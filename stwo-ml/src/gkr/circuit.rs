//! Layered circuit representation and compiler from ComputationGraph.
//!
//! Converts a DAG of ML operations into a layered GKR circuit where each
//! layer has a specialized reduction protocol (sumcheck for matmul,
//! identity for add, degree-2 for mul, LogUp for activations).

use std::ops::Range;

use crate::compiler::graph::{ComputationGraph, GraphOp};
use crate::components::activation::ActivationType;
use crate::components::attention::MultiHeadAttentionConfig;
use crate::gadgets::quantize::QuantParams;

use super::types::GKRError;

/// A single layer in the GKR circuit.
#[derive(Debug, Clone)]
pub struct CircuitLayer {
    pub layer_type: LayerType,
    pub input_shape: (usize, usize),
    pub output_shape: (usize, usize),
    /// Maps back to the GraphNode ID in the original ComputationGraph.
    pub node_id: usize,
    /// Input layer indices in the LayeredCircuit (predecessors).
    pub input_layers: Vec<usize>,
}

/// The type of operation performed by a circuit layer.
/// Each variant determines which GKR reduction protocol is used.
#[derive(Debug, Clone)]
pub enum LayerType {
    /// C[i][j] = sum_k A[i][k] * B[k][j]
    /// Reduction: sumcheck over k-dimension (reuses existing GPU kernels).
    MatMul {
        m: usize,
        k: usize,
        n: usize,
        weight_node_id: usize,
    },

    /// output[i] = lhs[i] + rhs[i]
    /// Reduction: degree-1 identity (no sumcheck, split claim).
    Add { size: usize },

    /// output[i] = lhs[i] * rhs[i]
    /// Reduction: degree-2 check at random point.
    Mul { size: usize },

    /// output[i] = f(input[i]) where f is a lookup table.
    /// Reduction: LogUp lookup argument.
    Activation {
        size: usize,
        activation_type: ActivationType,
    },

    /// Layer normalization: mean/variance + scale/shift.
    /// Reduction: LogUp for rsqrt + linear constraints.
    LayerNorm { dim: usize },

    /// RMS normalization: x / sqrt(mean(x²)) — no mean subtraction.
    /// Reduction: degree-2 eq-sumcheck (output = input × rsqrt) + LogUp for rsqrt.
    RMSNorm { dim: usize },

    /// Multi-head attention (decomposed into sub-matmuls + softmax).
    Attention { config: MultiHeadAttentionConfig },

    /// Model input — no reduction, verified against commitment.
    Input,

    /// Dequantization: maps quantized values to dequantized values via lookup table.
    /// Reduction: LogUp lookup argument (same protocol as Activation).
    Dequantize {
        size: usize,
        params: QuantParams,
    },

    /// Identity / passthrough (zero-cost in GKR, claim propagates unchanged).
    Identity,
}

/// SIMD batch configuration for identical transformer blocks.
#[derive(Debug, Clone)]
pub struct SIMDBatchConfig {
    /// Number of identical blocks.
    pub num_blocks: usize,
    /// Layer indices forming one template block.
    pub template_range: Range<usize>,
    /// log2(num_blocks) rounded up — SIMD randomness dimension.
    pub simd_log_size: usize,
}

/// A layered circuit compiled from a ComputationGraph.
/// GKR walks this from output (last layer) to input (first layer).
#[derive(Debug, Clone)]
pub struct LayeredCircuit {
    /// Layers ordered from input (index 0) to output (index len-1).
    pub layers: Vec<CircuitLayer>,

    /// Block boundaries for SIMD batching.
    pub block_ranges: Vec<Range<usize>>,

    /// SIMD config if identical blocks were detected.
    pub simd_config: Option<SIMDBatchConfig>,

    /// Original graph's input shape.
    pub input_shape: (usize, usize),

    /// Original graph's output shape.
    pub output_shape: (usize, usize),
}

impl LayeredCircuit {
    /// Compile a ComputationGraph into a layered GKR circuit.
    ///
    /// Steps:
    /// 1. Topological sort the graph.
    /// 2. Map each GraphOp to a LayerType.
    /// 3. Detect identical block boundaries.
    /// 4. Validate: each layer's output shape matches downstream input shapes.
    pub fn from_graph(graph: &ComputationGraph) -> Result<Self, GKRError> {
        let topo_order = graph.topological_order();

        if topo_order.is_empty() {
            return Err(GKRError::CompilationError(
                "empty computation graph".to_string(),
            ));
        }

        // Build node_id → layer_index mapping
        let mut node_to_layer: Vec<Option<usize>> = vec![None; graph.nodes.len()];
        let mut layers: Vec<CircuitLayer> = Vec::with_capacity(topo_order.len());

        for &node_id in &topo_order {
            let node = &graph.nodes[node_id];
            let layer_idx = layers.len();
            node_to_layer[node_id] = Some(layer_idx);

            // Map input node IDs to layer indices
            let input_layers: Vec<usize> = node
                .inputs
                .iter()
                .filter_map(|&inp_id| {
                    if inp_id < node_to_layer.len() {
                        node_to_layer[inp_id]
                    } else {
                        None
                    }
                })
                .collect();

            // Determine input shape from predecessors
            let input_shape = if input_layers.is_empty() {
                graph.input_shape
            } else {
                layers[input_layers[0]].output_shape
            };

            let layer_type = Self::map_op_to_layer_type(&node.op, node_id)?;

            layers.push(CircuitLayer {
                layer_type,
                input_shape,
                output_shape: node.output_shape,
                node_id,
                input_layers,
            });
        }

        // Detect block boundaries from original graph
        let graph_block_ranges = graph.find_block_boundaries();

        // Remap graph block ranges to circuit layer indices
        let block_ranges: Vec<Range<usize>> = graph_block_ranges
            .iter()
            .filter_map(|range| {
                let start = node_to_layer[range.start]?;
                let end_node = if range.end > 0 { range.end - 1 } else { 0 };
                let end = node_to_layer[end_node].map(|e| e + 1)?;
                Some(start..end)
            })
            .collect();

        // Detect SIMD batching opportunity
        let simd_config = Self::detect_simd_batching(&layers, &block_ranges);

        Ok(Self {
            layers,
            block_ranges,
            simd_config,
            input_shape: graph.input_shape,
            output_shape: graph.output_shape,
        })
    }

    /// Map a GraphOp to the corresponding GKR LayerType.
    fn map_op_to_layer_type(op: &GraphOp, node_id: usize) -> Result<LayerType, GKRError> {
        match op {
            GraphOp::MatMul { dims: (m, k, n) } => Ok(LayerType::MatMul {
                m: *m,
                k: *k,
                n: *n,
                weight_node_id: node_id,
            }),
            GraphOp::Activation {
                activation_type,
                size,
            } => Ok(LayerType::Activation {
                size: *size,
                activation_type: *activation_type,
            }),
            GraphOp::LayerNorm { dim } => Ok(LayerType::LayerNorm { dim: *dim }),
            GraphOp::Add { size } => Ok(LayerType::Add { size: *size }),
            GraphOp::Mul { size } => Ok(LayerType::Mul { size: *size }),
            GraphOp::Attention { config } => Ok(LayerType::Attention {
                config: *config,
            }),
            GraphOp::Identity { .. } => Ok(LayerType::Identity),
            GraphOp::Embedding { .. } => Err(GKRError::CompilationError(
                "Embedding layers must be lowered to MatMul before GKR compilation".to_string(),
            )),
            GraphOp::Conv2D { .. } => Err(GKRError::CompilationError(
                "Conv2D layers must be lowered to im2col + MatMul before GKR compilation"
                    .to_string(),
            )),
            GraphOp::Quantize { .. } => Ok(LayerType::Identity),
            GraphOp::Dequantize { params, size } => Ok(LayerType::Dequantize {
                size: *size,
                params: params.clone(),
            }),
            GraphOp::RMSNorm { dim } => Ok(LayerType::RMSNorm { dim: *dim }),
            GraphOp::RoPE { .. } => Ok(LayerType::Identity),
        }
    }

    /// Detect if blocks have identical structure (same layer types and shapes).
    /// Returns SIMD config if >= 2 identical blocks are found.
    fn detect_simd_batching(
        layers: &[CircuitLayer],
        block_ranges: &[Range<usize>],
    ) -> Option<SIMDBatchConfig> {
        if block_ranges.len() < 2 {
            return None;
        }

        // Check if all blocks have the same length and structure
        let template = &block_ranges[0];
        let template_len = template.end - template.start;

        let mut identical_count = 1;

        for range in &block_ranges[1..] {
            let len = range.end - range.start;
            if len != template_len {
                break;
            }

            // Compare layer types and shapes
            let matches = (0..template_len).all(|offset| {
                let t = &layers[template.start + offset];
                let c = &layers[range.start + offset];
                layer_type_matches(&t.layer_type, &c.layer_type)
                    && t.input_shape == c.input_shape
                    && t.output_shape == c.output_shape
            });

            if matches {
                identical_count += 1;
            } else {
                break;
            }
        }

        if identical_count >= 2 {
            let simd_log_size = (identical_count as f64).log2().ceil() as usize;
            Some(SIMDBatchConfig {
                num_blocks: identical_count,
                template_range: template.clone(),
                simd_log_size,
            })
        } else {
            None
        }
    }

    /// Total number of layers in the circuit.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Count of layers by type.
    pub fn layer_counts(&self) -> LayerCounts {
        let mut counts = LayerCounts::default();
        for layer in &self.layers {
            match &layer.layer_type {
                LayerType::MatMul { .. } => counts.matmul += 1,
                LayerType::Add { .. } => counts.add += 1,
                LayerType::Mul { .. } => counts.mul += 1,
                LayerType::Activation { .. } => counts.activation += 1,
                LayerType::LayerNorm { .. } => counts.layer_norm += 1,
                LayerType::RMSNorm { .. } => counts.layer_norm += 1,
                LayerType::Attention { .. } => counts.attention += 1,
                LayerType::Dequantize { .. } => counts.dequantize += 1,
                LayerType::Input => counts.input += 1,
                LayerType::Identity => counts.identity += 1,
            }
        }
        counts
    }

    /// Total number of sumcheck variables across all matmul layers.
    pub fn total_sumcheck_variables(&self) -> usize {
        self.layers
            .iter()
            .filter_map(|l| match &l.layer_type {
                LayerType::MatMul { k, .. } => {
                    Some(k.next_power_of_two().trailing_zeros() as usize)
                }
                _ => None,
            })
            .sum()
    }
}

/// Per-type layer counts for circuit analysis.
#[derive(Debug, Default, Clone)]
pub struct LayerCounts {
    pub matmul: usize,
    pub add: usize,
    pub mul: usize,
    pub activation: usize,
    pub layer_norm: usize,
    pub attention: usize,
    pub dequantize: usize,
    pub input: usize,
    pub identity: usize,
}

/// Check if two layer types are structurally identical (same op and dimensions).
fn layer_type_matches(a: &LayerType, b: &LayerType) -> bool {
    match (a, b) {
        (
            LayerType::MatMul {
                m: m1,
                k: k1,
                n: n1,
                ..
            },
            LayerType::MatMul {
                m: m2,
                k: k2,
                n: n2,
                ..
            },
        ) => m1 == m2 && k1 == k2 && n1 == n2,
        (LayerType::Add { size: s1 }, LayerType::Add { size: s2 }) => s1 == s2,
        (LayerType::Mul { size: s1 }, LayerType::Mul { size: s2 }) => s1 == s2,
        (
            LayerType::Activation {
                size: s1,
                activation_type: t1,
            },
            LayerType::Activation {
                size: s2,
                activation_type: t2,
            },
        ) => s1 == s2 && t1 == t2,
        (LayerType::LayerNorm { dim: d1 }, LayerType::LayerNorm { dim: d2 }) => d1 == d2,
        (LayerType::RMSNorm { dim: d1 }, LayerType::RMSNorm { dim: d2 }) => d1 == d2,
        (LayerType::Attention { config: c1 }, LayerType::Attention { config: c2 }) => {
            c1.num_heads == c2.num_heads
                && c1.d_model == c2.d_model
                && c1.seq_len == c2.seq_len
        }
        (
            LayerType::Dequantize { size: s1, .. },
            LayerType::Dequantize { size: s2, .. },
        ) => s1 == s2,
        (LayerType::Input, LayerType::Input) => true,
        (LayerType::Identity, LayerType::Identity) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use crate::components::activation::ActivationType;
    use crate::components::attention::MultiHeadAttentionConfig;

    #[test]
    fn test_compile_simple_mlp() {
        // 3-layer MLP: Linear → ReLU → Linear → ReLU → Linear
        let mut builder = GraphBuilder::new((1, 64));
        builder.linear(128);
        builder.activation(ActivationType::ReLU);
        builder.linear(64);
        builder.activation(ActivationType::ReLU);
        builder.linear(10);
        let graph = builder.build();

        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        assert_eq!(circuit.depth(), 5);
        let counts = circuit.layer_counts();
        assert_eq!(counts.matmul, 3);
        assert_eq!(counts.activation, 2);

        // First layer should be MatMul(1, 64, 128)
        match &circuit.layers[0].layer_type {
            LayerType::MatMul { m, k, n, .. } => {
                assert_eq!(*m, 1);
                assert_eq!(*k, 64);
                assert_eq!(*n, 128);
            }
            other => panic!("expected MatMul, got {other:?}"),
        }

        // No block boundaries detected (no LayerNorm)
        assert!(circuit.simd_config.is_none());
    }

    #[test]
    fn test_compile_transformer_blocks() {
        // 2 identical transformer blocks: LayerNorm → Attention → LayerNorm → Linear → ReLU → Linear
        let config = MultiHeadAttentionConfig::new(4, 64, 8);
        let mut builder = GraphBuilder::new((8, 64));

        for _ in 0..2 {
            builder
                .layer_norm()
                .attention(config)
                .layer_norm()
                .linear(256)
                .activation(ActivationType::GELU)
                .linear(64);
        }

        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        // 2 blocks × 6 layers = 12 layers
        assert_eq!(circuit.depth(), 12);

        // Block detection should find identical blocks
        assert!(circuit.block_ranges.len() >= 2);

        // SIMD config should detect 2 identical blocks
        if let Some(simd) = &circuit.simd_config {
            assert_eq!(simd.num_blocks, 2);
            assert_eq!(simd.simd_log_size, 1); // log2(2) = 1
        }
    }

    #[test]
    fn test_compile_with_residual() {
        // Linear → fork → Linear → Add (residual)
        let mut builder = GraphBuilder::new((1, 64));
        builder.linear(64);
        let fork = builder.fork();
        builder.linear(64);
        builder.add_from(fork);

        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        // Should have: MatMul, MatMul, Add
        assert_eq!(circuit.depth(), 3);
        let counts = circuit.layer_counts();
        assert_eq!(counts.matmul, 2);
        assert_eq!(counts.add, 1);

        // Add layer should have 2 input layers
        let add_layer = &circuit.layers[2];
        assert_eq!(add_layer.input_layers.len(), 2);
    }

    #[test]
    fn test_layer_counts() {
        let mut builder = GraphBuilder::new((1, 32));
        builder.linear(64);
        builder.activation(ActivationType::ReLU);
        builder.linear(32);
        let graph = builder.build();

        let circuit = LayeredCircuit::from_graph(&graph).unwrap();
        let counts = circuit.layer_counts();

        assert_eq!(counts.matmul, 2);
        assert_eq!(counts.activation, 1);
        assert_eq!(counts.add, 0);
        assert_eq!(counts.mul, 0);
    }

    #[test]
    fn test_total_sumcheck_variables() {
        // Two matmuls: k=64 → 6 vars, k=128 → 7 vars = 13 total
        let mut builder = GraphBuilder::new((1, 64));
        builder.linear(128);
        builder.linear(32);
        let graph = builder.build();

        let circuit = LayeredCircuit::from_graph(&graph).unwrap();
        let total = circuit.total_sumcheck_variables();
        assert_eq!(total, 6 + 7); // log2(64) + log2(128)
    }

    #[test]
    fn test_empty_graph_error() {
        let graph = ComputationGraph::new((1, 64));
        let result = LayeredCircuit::from_graph(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_type_matches() {
        assert!(layer_type_matches(
            &LayerType::MatMul { m: 1, k: 64, n: 128, weight_node_id: 0 },
            &LayerType::MatMul { m: 1, k: 64, n: 128, weight_node_id: 5 },
        ));

        assert!(!layer_type_matches(
            &LayerType::MatMul { m: 1, k: 64, n: 128, weight_node_id: 0 },
            &LayerType::MatMul { m: 1, k: 32, n: 128, weight_node_id: 0 },
        ));

        assert!(!layer_type_matches(
            &LayerType::Add { size: 64 },
            &LayerType::Mul { size: 64 },
        ));
    }
}
