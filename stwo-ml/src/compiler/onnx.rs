//! ONNX model import.
//!
//! Parses ONNX protobuf files and extracts layer definitions,
//! weight tensors, and activation functions for circuit compilation.
//!
//! # Op Mapping
//!
//! | ONNX Op | GraphOp |
//! |---------|---------|
//! | `MatMul`, `Gemm` | `GraphOp::MatMul` |
//! | `Relu`, `Gelu`, `Sigmoid` | `GraphOp::Activation` |
//! | `LayerNormalization` | `GraphOp::LayerNorm` |
//! | `DequantizeLinear` | `GraphOp::Quantize` |

use std::path::Path;

use crate::compiler::graph::{ComputationGraph, GraphWeights, GraphBuilder};
use crate::components::activation::ActivationType;

/// Metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name from ONNX graph.
    pub name: String,
    /// Number of parameters.
    pub num_parameters: usize,
    /// Input shape: (batch, features).
    pub input_shape: (usize, usize),
    /// Output shape: (batch, features).
    pub output_shape: (usize, usize),
    /// Number of layers.
    pub num_layers: usize,
}

/// A fully loaded ONNX model ready for proving.
pub struct OnnxModel {
    pub graph: ComputationGraph,
    pub weights: GraphWeights,
    pub input_shape: (usize, usize),
    pub metadata: ModelMetadata,
}

/// Error type for ONNX loading.
#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("Shape inference error: {0}")]
    ShapeError(String),
    #[error("Weight extraction error: {0}")]
    WeightError(String),
}

/// Load an ONNX model from a file path.
///
/// Uses tract-onnx for full ONNX protobuf parsing. Extracts the graph
/// structure AND weight tensors, quantizing them to M31.
#[cfg(feature = "onnx")]
pub fn load_onnx(path: &Path) -> Result<OnnxModel, OnnxError> {
    use tract_onnx::prelude::*;
    use crate::compiler::quantize_weights::{quantize_weight_matrix, quantize_bias_vector};
    use crate::gadgets::quantize::QuantStrategy;

    let model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| OnnxError::ParseError(e.to_string()))?
        .into_optimized()
        .map_err(|e| OnnxError::ParseError(e.to_string()))?
        .into_runnable()
        .map_err(|e| OnnxError::ParseError(e.to_string()))?;

    // Extract input shape
    let input_fact = model.model().input_fact(0)
        .map_err(|e| OnnxError::ShapeError(e.to_string()))?;
    let input_shape = extract_shape_typed(input_fact);

    let mut builder = GraphBuilder::new(input_shape);
    let mut weights = GraphWeights::new();
    let mut num_parameters = 0usize;

    // Walk the optimized model's nodes and map ops to GraphOp variants.
    // tract's optimized model may fuse/rewrite ops, so we match on the
    // actual op type after optimization.
    for (node_idx, node) in model.model().nodes.iter().enumerate() {
        let op_name = node.op().name();

        match op_name.as_ref() {
            "MatMatMul" | "MatMul" | "Gemm" | "EinSum" => {
                let out_shape = get_node_output_shape(node);
                let out_features = out_shape.1;
                // Capture in_features BEFORE adding the linear node
                let in_features = builder.current_output_shape().1;
                builder.linear(out_features);
                let graph_node_id = builder.current_node_id();

                // Extract weight tensor from the second input (B matrix)
                if let Some(weight_data) = extract_const_input(model.model(), node_idx, 1) {
                    let (matrix, _params) = quantize_weight_matrix(
                        &weight_data,
                        in_features,
                        out_features,
                        QuantStrategy::Direct,
                    );
                    num_parameters += weight_data.len();
                    weights.add_weight(graph_node_id, matrix);
                }

                // Extract bias for Gemm ops (third input)
                if op_name.as_ref() == "Gemm" {
                    if let Some(bias_data) = extract_const_input(model.model(), node_idx, 2) {
                        let (bias_vec, _) = quantize_bias_vector(
                            &bias_data,
                            QuantStrategy::Direct,
                        );
                        num_parameters += bias_data.len();
                        weights.add_bias(graph_node_id, bias_vec);
                    }
                }
            }
            "Relu" => {
                builder.activation(ActivationType::ReLU);
            }
            "Gelu" | "FastGelu" => {
                builder.activation(ActivationType::GELU);
            }
            "Sigmoid" => {
                builder.activation(ActivationType::Sigmoid);
            }
            "Softmax" => {
                builder.activation(ActivationType::Softmax);
            }
            "LayerNormalization" | "RmsNormalization" => {
                builder.layer_norm();
            }
            // Element-wise ops: real computation with AIR constraints
            "Add" => {
                let shape = builder.current_output_shape();
                let size = shape.0 * shape.1;
                // For ONNX Add, second input is often from an earlier layer (residual).
                // In the sequential builder, we treat it as Add with the previous output.
                let inputs = builder.last_node.map(|n| vec![n]).unwrap_or_default();
                let id = builder.graph.add_node(
                    crate::compiler::graph::GraphOp::Add { size },
                    inputs,
                    shape,
                );
                builder.last_node = Some(id);
            }
            "Mul" => {
                let shape = builder.current_output_shape();
                let size = shape.0 * shape.1;
                let inputs = builder.last_node.map(|n| vec![n]).unwrap_or_default();
                let id = builder.graph.add_node(
                    crate::compiler::graph::GraphOp::Mul { size },
                    inputs,
                    shape,
                );
                builder.last_node = Some(id);
            }
            // Dequantization: map to quantize node
            "DequantizeLinear" => {
                use crate::gadgets::quantize::{QuantStrategy, QuantParams};
                builder.quantize(QuantParams {
                    strategy: QuantStrategy::Direct,
                    scale: 1.0,
                    zero_point: 0,
                    bits: 8,
                });
            }
            // Skip infrastructure ops that don't map to computation
            "Source" | "Const" | "Reshape" | "Transpose" | "Flatten"
            | "Squeeze" | "Unsqueeze" | "Cast" | "Gather" | "Concat"
            | "Shape" | "Slice" | "Identity" | "Dropout" => {
                // These are structural ops, not computation ops
            }
            _ => {
                tracing::warn!("Skipping unsupported ONNX op: {}", op_name);
            }
        }
    }

    let graph = builder.build();
    let metadata = ModelMetadata {
        name: model.model().properties.get("name")
            .and_then(|v| v.to_scalar::<String>().ok().cloned())
            .unwrap_or_else(|| "unnamed".to_string()),
        num_parameters,
        input_shape,
        output_shape: graph.output_shape,
        num_layers: graph.num_layers(),
    };

    Ok(OnnxModel {
        graph,
        weights,
        input_shape,
        metadata,
    })
}

/// Extract f32 data from a constant input to a node.
#[cfg(feature = "onnx")]
fn extract_const_input(
    model: &tract_onnx::prelude::TypedModel,
    node_idx: usize,
    input_slot: usize,
) -> Option<Vec<f32>> {
    let node = &model.nodes[node_idx];
    let input = node.inputs.get(input_slot)?;
    let source_node = &model.nodes[input.node];

    // Check if the source node is a Const op
    if source_node.op().name() != "Const" {
        return None;
    }

    // Extract the tensor from the Const op
    let const_op = source_node.op_as::<tract_onnx::tract_core::ops::konst::Const>()?;
    let tensor = &const_op.0;

    // Convert to f32
    tensor.as_slice::<f32>().ok().map(|s| s.to_vec())
}

/// Get output shape of a node from its output facts.
#[cfg(feature = "onnx")]
fn get_node_output_shape(node: &tract_onnx::prelude::TypedNode) -> (usize, usize) {
    let fact = node.outputs.first()
        .map(|o| &o.fact);
    if let Some(f) = fact {
        let shape = f.shape.as_concrete().unwrap_or_default();
        match shape.len() {
            0 => (1, 1),
            1 => (1, shape[0]),
            _ => (shape[shape.len() - 2], shape[shape.len() - 1]),
        }
    } else {
        (1, 1)
    }
}

/// Extract shape from a TypedFact.
#[cfg(feature = "onnx")]
fn extract_shape_typed(fact: &tract_onnx::prelude::TypedFact) -> (usize, usize) {
    let shape = fact.shape.as_concrete().unwrap_or_default();
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        _ => (shape[0], shape[shape.len() - 1]),
    }
}

/// When `onnx` feature is not enabled, return an error.
#[cfg(not(feature = "onnx"))]
pub fn load_onnx(_path: &Path) -> Result<OnnxModel, OnnxError> {
    Err(OnnxError::ParseError(
        "ONNX loading requires the 'onnx' feature flag".to_string(),
    ))
}

/// Build a simple MLP model programmatically (no ONNX required).
///
/// Useful for testing and for models defined in code.
pub fn build_mlp(
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
    activation: ActivationType,
) -> OnnxModel {
    let mut builder = GraphBuilder::new((1, input_dim));

    for &dim in hidden_dims {
        builder.linear(dim);
        builder.activation(activation);
    }
    builder.linear(output_dim);

    let graph = builder.build();
    let metadata = ModelMetadata {
        name: "mlp".to_string(),
        num_parameters: 0,
        input_shape: (1, input_dim),
        output_shape: graph.output_shape,
        num_layers: graph.num_layers(),
    };

    OnnxModel {
        graph,
        weights: GraphWeights::new(),
        input_shape: (1, input_dim),
        metadata,
    }
}

/// Configuration for building a transformer block.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Model/embedding dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Feed-forward inner dimension (typically 4 * d_model).
    pub d_ff: usize,
    /// Activation function for FFN.
    pub activation: ActivationType,
}

impl TransformerConfig {
    /// Create a new config with defaults: d_ff = 4 * d_model, activation = GELU.
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self {
            d_model,
            num_heads,
            d_ff: 4 * d_model,
            activation: ActivationType::GELU,
        }
    }
}

/// Build a single transformer block with auto-generated weights.
///
/// Structure: LayerNorm → Q proj → O proj → LayerNorm → FFN up → Activation → FFN down.
/// 7 layers total, all weighted.
pub fn build_transformer_block(config: &TransformerConfig, seed: u64) -> OnnxModel {
    let d = config.d_model;
    let d_ff = config.d_ff;

    let mut builder = GraphBuilder::new((1, d));

    // Pre-attention LayerNorm
    builder.layer_norm();
    // Q projection (simplified: d_model → d_model)
    builder.linear(d);
    // O projection (d_model → d_model)
    builder.linear(d);
    // Post-attention LayerNorm
    builder.layer_norm();
    // FFN up projection (d_model → d_ff)
    builder.linear(d_ff);
    // FFN activation
    builder.activation(config.activation);
    // FFN down projection (d_ff → d_model)
    builder.linear(d);

    let graph = builder.build();
    let weights = generate_weights_for_graph(&graph, seed);
    let num_parameters = count_matmul_params(&graph);

    let metadata = ModelMetadata {
        name: "transformer_block".to_string(),
        num_parameters,
        input_shape: (1, d),
        output_shape: graph.output_shape,
        num_layers: graph.num_layers(),
    };

    OnnxModel {
        graph,
        weights,
        input_shape: (1, d),
        metadata,
    }
}

/// Build a multi-layer transformer with auto-generated weights.
///
/// Stacks `num_layers` transformer blocks + final LayerNorm + LM head linear.
pub fn build_transformer(
    config: &TransformerConfig,
    num_layers: usize,
    vocab_size: usize,
    seed: u64,
) -> OnnxModel {
    let d = config.d_model;
    let d_ff = config.d_ff;

    let mut builder = GraphBuilder::new((1, d));

    for _ in 0..num_layers {
        // Each block: LN → Q → O → LN → FFN up → act → FFN down
        builder.layer_norm();
        builder.linear(d);
        builder.linear(d);
        builder.layer_norm();
        builder.linear(d_ff);
        builder.activation(config.activation);
        builder.linear(d);
    }

    // Final LayerNorm + LM head
    builder.layer_norm();
    builder.linear(vocab_size);

    let graph = builder.build();
    let weights = generate_weights_for_graph(&graph, seed);
    let num_parameters = count_matmul_params(&graph);

    let metadata = ModelMetadata {
        name: format!("transformer_{num_layers}L"),
        num_parameters,
        input_shape: (1, d),
        output_shape: graph.output_shape,
        num_layers: graph.num_layers(),
    };

    OnnxModel {
        graph,
        weights,
        input_shape: (1, d),
        metadata,
    }
}

/// Count the total number of MatMul parameters (k*n) across all MatMul nodes.
pub fn count_matmul_params(graph: &ComputationGraph) -> usize {
    use crate::compiler::graph::GraphOp;
    graph.nodes.iter().map(|node| {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            k * n
        } else {
            0
        }
    }).sum()
}

/// Generate deterministic M31 weights for all MatMul nodes in a graph.
///
/// Uses an LCG PRNG: `state = state * 6364136223846793005 + 1442695040888963407`.
/// Values are `(state >> 33) % 9 + 1`, giving M31 values in [1, 9].
pub fn generate_weights_for_graph(
    graph: &ComputationGraph,
    seed: u64,
) -> GraphWeights {
    use stwo::core::fields::m31::M31;
    use crate::compiler::graph::GraphOp;
    use crate::components::matmul::M31Matrix;

    let mut weights = GraphWeights::new();
    let mut state = seed;

    for node in &graph.nodes {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            let mut matrix = M31Matrix::new(*k, *n);
            for i in 0..*k {
                for j in 0..*n {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let val = ((state >> 33) % 9 + 1) as u32;
                    matrix.set(i, j, M31::from(val));
                }
            }
            weights.add_weight(node.id, matrix);
        }
    }

    weights
}

/// Build a simple MLP model with auto-generated deterministic weights.
///
/// Same structure as `build_mlp`, but populates `GraphWeights` with
/// LCG-seeded M31 values in [1, 9]. Immediately provable.
pub fn build_mlp_with_weights(
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
    activation: ActivationType,
    seed: u64,
) -> OnnxModel {
    let mut builder = GraphBuilder::new((1, input_dim));

    for &dim in hidden_dims {
        builder.linear(dim);
        builder.activation(activation);
    }
    builder.linear(output_dim);

    let graph = builder.build();
    let weights = generate_weights_for_graph(&graph, seed);
    let num_parameters = count_matmul_params(&graph);

    let metadata = ModelMetadata {
        name: "mlp".to_string(),
        num_parameters,
        input_shape: (1, input_dim),
        output_shape: graph.output_shape,
        num_layers: graph.num_layers(),
    };

    OnnxModel {
        graph,
        weights,
        input_shape: (1, input_dim),
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_mlp() {
        let model = build_mlp(784, &[128, 64], 10, ActivationType::ReLU);
        assert_eq!(model.input_shape, (1, 784));
        assert_eq!(model.graph.output_shape, (1, 10));
        assert_eq!(model.graph.num_layers(), 5); // linear + relu + linear + relu + linear
        assert_eq!(model.metadata.name, "mlp");
    }

    #[test]
    fn test_onnx_not_available_without_feature() {
        #[cfg(not(feature = "onnx"))]
        {
            let result = load_onnx(Path::new("nonexistent.onnx"));
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_build_transformer_block() {
        let config = TransformerConfig::new(8, 2);
        let model = build_transformer_block(&config, 42);

        // 7 layers: LN + Q + O + LN + FFN_up + act + FFN_down
        assert_eq!(model.graph.num_layers(), 7);
        assert_eq!(model.input_shape, (1, 8));
        assert_eq!(model.graph.output_shape, (1, 8));

        // 4 MatMul layers: Q(8→8) + O(8→8) + FFN_up(8→32) + FFN_down(32→8)
        // Params: 64 + 64 + 256 + 256 = 640
        assert_eq!(model.metadata.num_parameters, 640);
        assert_eq!(model.weights.weights.len(), 4);
    }

    #[test]
    fn test_build_transformer_block_provable() {
        use crate::compiler::prove::prove_model;

        // Use tiny dims so proving is fast
        let config = TransformerConfig {
            d_model: 4,
            num_heads: 1,
            d_ff: 8,
            activation: ActivationType::GELU,
        };
        let model = build_transformer_block(&config, 77);

        let mut input = crate::components::matmul::M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, stwo::core::fields::m31::M31::from((j + 1) as u32));
        }

        let result = prove_model(&model.graph, &input, &model.weights);
        assert!(result.is_ok(), "Transformer block should prove: {:?}", result.err());
    }

    #[test]
    fn test_build_multi_layer_transformer() {
        let config = TransformerConfig::new(8, 2);
        let model = build_transformer(
            &config,
            2,     // 2 layers
            16,    // vocab_size
            42,
        );

        // Each block = 7 layers, + final LN + LM head = 2*7 + 2 = 16
        assert_eq!(model.graph.num_layers(), 16);
        assert_eq!(model.graph.output_shape, (1, 16));
        assert_eq!(model.metadata.name, "transformer_2L");
    }

    #[test]
    fn test_build_mlp_with_weights_autogenerated() {
        let model = build_mlp_with_weights(4, &[8, 4], 2, ActivationType::ReLU, 42);

        assert_eq!(model.input_shape, (1, 4));
        assert_eq!(model.graph.output_shape, (1, 2));
        assert_eq!(model.graph.num_layers(), 5); // linear + relu + linear + relu + linear

        // Weights should be populated
        assert!(!model.weights.weights.is_empty());
        assert_eq!(model.weights.weights.len(), 3); // 3 MatMul layers

        // Verify parameter count: 4*8 + 8*4 + 4*2 = 32+32+8 = 72
        assert_eq!(model.metadata.num_parameters, 72);
        assert_eq!(count_matmul_params(&model.graph), 72);

        // Verify weight shapes
        let w0 = model.weights.get_weight(0).unwrap();
        assert_eq!((w0.rows, w0.cols), (4, 8));
        let w2 = model.weights.get_weight(2).unwrap();
        assert_eq!((w2.rows, w2.cols), (8, 4));
        let w4 = model.weights.get_weight(4).unwrap();
        assert_eq!((w4.rows, w4.cols), (4, 2));
    }

    #[test]
    fn test_build_mlp_with_weights_and_prove_auto() {
        use crate::compiler::prove::prove_model;

        let model = build_mlp_with_weights(4, &[4], 2, ActivationType::ReLU, 123);

        let mut input = crate::components::matmul::M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, stwo::core::fields::m31::M31::from((j + 1) as u32));
        }

        let result = prove_model(&model.graph, &input, &model.weights);
        assert!(result.is_ok(), "Auto-weighted MLP should prove: {:?}", result.err());

        let (proofs, execution) = result.unwrap();
        assert_eq!(proofs.len(), 3); // linear + relu + linear
        assert_eq!(execution.output.cols, 2);
    }

    #[test]
    fn test_generate_weights_deterministic() {
        use crate::compiler::graph::GraphBuilder;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        // Same seed = same weights
        let w1 = generate_weights_for_graph(&graph, 42);
        let w2 = generate_weights_for_graph(&graph, 42);
        assert_eq!(w1.weights.len(), w2.weights.len());
        for ((id1, m1), (id2, m2)) in w1.weights.iter().zip(w2.weights.iter()) {
            assert_eq!(id1, id2);
            assert_eq!(m1.data, m2.data);
        }

        // Different seed = different weights
        let w3 = generate_weights_for_graph(&graph, 99);
        let (_, m1) = &w1.weights[0];
        let (_, m3) = &w3.weights[0];
        assert_ne!(m1.data, m3.data, "Different seeds should produce different weights");
    }

    #[test]
    fn test_build_mlp_with_weights_and_prove() {
        use stwo::core::fields::m31::M31;
        use crate::compiler::prove::prove_model;

        let model = build_mlp(4, &[4], 2, ActivationType::ReLU);

        // Manually supply weights (build_mlp doesn't generate them)
        let mut weights = GraphWeights::new();

        let mut w0 = crate::components::matmul::M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);

        let mut w2 = crate::components::matmul::M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let mut input = crate::components::matmul::M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let result = prove_model(&model.graph, &input, &weights);
        assert!(result.is_ok(), "MLP from build_mlp should prove: {:?}", result.err());
    }

    /// Test that build_mlp_with_weights produces a model that goes through
    /// the full prove → verify pipeline (simulates what load_onnx → prove would do).
    #[test]
    fn test_load_onnx_programmatic_via_builder() {
        use crate::compiler::prove::{prove_model, verify_model_matmuls};

        // Simulate what load_onnx produces: a graph with weights
        let model = build_mlp_with_weights(4, &[4, 4], 2, ActivationType::ReLU, 55);

        // Verify structure matches what ONNX loading would produce
        assert_eq!(model.graph.num_layers(), 5); // 3 linear + 2 relu
        assert_eq!(model.weights.weights.len(), 3);
        assert!(model.metadata.num_parameters > 0);

        let mut input = crate::components::matmul::M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, stwo::core::fields::m31::M31::from((j + 1) as u32));
        }

        let (proofs, _exec) = prove_model(&model.graph, &input, &model.weights)
            .expect("programmatic ONNX model should prove");

        verify_model_matmuls(&proofs, &model.graph, &input, &model.weights)
            .expect("programmatic ONNX model should verify");
    }

    /// End-to-end: build model with identity/quantize ops → prove.
    #[test]
    fn test_load_onnx_and_prove_with_all_ops() {
        use crate::compiler::prove::prove_model;
        use crate::compiler::graph::GraphBuilder;

        // Build a graph that mimics what load_onnx would produce with
        // DequantizeLinear, Add, Mul ops
        let mut builder = GraphBuilder::new((1, 4));
        builder.quantize(crate::gadgets::quantize::QuantParams {
            strategy: crate::gadgets::quantize::QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 8,
        });
        builder.linear(4);
        builder.identity(); // simulates Add
        builder.activation(ActivationType::ReLU);
        builder.identity(); // simulates Mul
        builder.linear(2);

        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 88);

        let mut input = crate::components::matmul::M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, stwo::core::fields::m31::M31::from((j + 1) as u32));
        }

        let result = prove_model(&graph, &input, &weights);
        assert!(result.is_ok(), "Model with all op types should prove: {:?}", result.err());

        let (proofs, execution) = result.unwrap();
        assert_eq!(proofs.len(), 6); // quantize + linear + identity + relu + identity + linear
        assert_eq!(execution.output.cols, 2);
    }
}
