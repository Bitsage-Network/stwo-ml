//! Computation graph builder.
//!
//! Constructs a directed acyclic graph of ML operations, assigns
//! each node to a STWO component, and computes the total trace budget.

use stwo::core::fields::m31::M31;

use crate::components::activation::ActivationType;
use crate::components::attention::MultiHeadAttentionConfig;
use crate::components::matmul::M31Matrix;
use crate::gadgets::quantize::QuantParams;

/// An operation in the computation graph.
#[derive(Debug, Clone)]
pub enum GraphOp {
    /// Matrix multiplication: C = A × B.
    MatMul {
        /// (rows_a, cols_a/rows_b, cols_b).
        dims: (usize, usize, usize),
    },
    /// Activation function applied element-wise.
    Activation {
        activation_type: ActivationType,
        size: usize,
    },
    /// Layer normalization over the last dimension.
    LayerNorm {
        /// Dimension being normalized.
        dim: usize,
    },
    /// RMS normalization over the last dimension (no mean subtraction).
    RMSNorm {
        /// Dimension being normalized.
        dim: usize,
    },
    /// Quantization / dequantization step.
    Quantize { params: QuantParams, size: usize },
    /// Multi-head attention block.
    Attention { config: MultiHeadAttentionConfig },
    /// Element-wise addition: output[i] = lhs[i] + rhs[i].
    Add { size: usize },
    /// Element-wise multiplication: output[i] = lhs[i] * rhs[i].
    Mul { size: usize },
    /// Embedding table lookup: output = table[token_ids].
    Embedding { vocab_size: usize, embed_dim: usize },
    /// 2D convolution via im2col + MatMul.
    Conv2D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    },
    /// Dequantization step: maps quantized M31 values back to dequantized M31 values
    /// via a lookup table. Provable via LogUp (table size = 2^bits).
    Dequantize { params: QuantParams, size: usize },
    /// Rotary Positional Embedding applied to Q/K vectors.
    /// Applies position-dependent rotations to adjacent element pairs.
    RoPE {
        config: crate::components::rope::RoPEConfig,
    },
    /// Identity / passthrough (for graph structure).
    Identity { size: usize },
}

impl GraphOp {
    /// Estimated trace rows for this operation (sumcheck-based).
    pub fn trace_rows(&self) -> usize {
        match self {
            GraphOp::MatMul { dims: (m, k, n) } => {
                // Sumcheck cost: O(m*n + m*k + k*n)
                m * n + m * k + k * n
            }
            GraphOp::Activation { size, .. } => {
                // One lookup per element
                *size
            }
            GraphOp::LayerNorm { dim } => {
                // Mean + variance + rsqrt lookup + scale/shift
                dim * 4
            }
            GraphOp::RMSNorm { dim } => {
                // rms² + rsqrt lookup + scale (no mean)
                dim * 3
            }
            GraphOp::Quantize { size, .. } => {
                // Range check per element
                *size
            }
            GraphOp::Dequantize { size, params } => {
                // Trace must be >= table size (2^bits)
                (*size).max(1usize << params.bits)
            }
            GraphOp::Attention { config } => config.sumcheck_trace_rows(),
            GraphOp::Add { size } | GraphOp::Mul { size } => {
                // One constraint per element
                *size
            }
            GraphOp::Embedding {
                vocab_size,
                embed_dim,
            } => {
                // LogUp lookup per token
                *vocab_size + *embed_dim
            }
            GraphOp::Conv2D {
                in_channels,
                out_channels,
                kernel_size,
                ..
            } => {
                // im2col + matmul cost
                in_channels * kernel_size * kernel_size * out_channels
            }
            GraphOp::RoPE { config } => {
                // 7 trace columns per dimension pair: input_x, input_y, cos, sin, out_x, out_y, mult
                config.seq_len * config.num_pairs()
            }
            GraphOp::Identity { .. } => 0,
        }
    }
}

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique node ID.
    pub id: usize,
    /// The operation this node performs.
    pub op: GraphOp,
    /// Input node IDs.
    pub inputs: Vec<usize>,
    /// Output shape: (rows, cols).
    pub output_shape: (usize, usize),
}

/// Weights for the entire graph.
#[derive(Debug, Clone)]
pub struct GraphWeights {
    /// Per-node weight matrices (node_id → weight data).
    pub weights: Vec<(usize, M31Matrix)>,
    /// Per-node bias vectors.
    pub biases: Vec<(usize, Vec<M31>)>,
    /// Named weight matrices: (node_id, name, weight).
    pub named_weights: Vec<(usize, String, M31Matrix)>,
}

impl GraphWeights {
    pub fn new() -> Self {
        Self {
            weights: Vec::new(),
            biases: Vec::new(),
            named_weights: Vec::new(),
        }
    }

    pub fn add_weight(&mut self, node_id: usize, weight: M31Matrix) {
        self.weights.push((node_id, weight));
    }

    /// Add a named weight for a node (e.g. "w_q", "w_k" for attention).
    pub fn add_named_weight(&mut self, node_id: usize, name: &str, weight: M31Matrix) {
        self.named_weights.push((node_id, name.to_string(), weight));
    }

    pub fn add_bias(&mut self, node_id: usize, bias: Vec<M31>) {
        self.biases.push((node_id, bias));
    }

    pub fn get_weight(&self, node_id: usize) -> Option<&M31Matrix> {
        self.weights
            .iter()
            .find(|(id, _)| *id == node_id)
            .map(|(_, w)| w)
    }

    /// Get a named weight for a node.
    pub fn get_named_weight(&self, node_id: usize, name: &str) -> Option<&M31Matrix> {
        self.named_weights
            .iter()
            .find(|(id, n, _)| *id == node_id && n == name)
            .map(|(_, _, w)| w)
    }

    pub fn get_bias(&self, node_id: usize) -> Option<&Vec<M31>> {
        self.biases
            .iter()
            .find(|(id, _)| *id == node_id)
            .map(|(_, b)| b)
    }

    /// Extract weights for a node range `[start..end)`, remapping node IDs to `[0..len)`.
    pub fn subset(&self, range: std::ops::Range<usize>) -> GraphWeights {
        let start = range.start;
        let end = range.end;
        let mut sub = GraphWeights::new();

        for (id, w) in &self.weights {
            if *id >= start && *id < end {
                sub.weights.push((*id - start, w.clone()));
            }
        }
        for (id, b) in &self.biases {
            if *id >= start && *id < end {
                sub.biases.push((*id - start, b.clone()));
            }
        }
        for (id, name, w) in &self.named_weights {
            if *id >= start && *id < end {
                sub.named_weights
                    .push((*id - start, name.clone(), w.clone()));
            }
        }

        sub
    }
}

impl Default for GraphWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// A computation graph representing a neural network.
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    pub nodes: Vec<GraphNode>,
    /// Input shape to the graph.
    pub input_shape: (usize, usize),
    /// Output shape of the graph.
    pub output_shape: (usize, usize),
}

impl ComputationGraph {
    pub fn new(input_shape: (usize, usize)) -> Self {
        Self {
            nodes: Vec::new(),
            input_shape,
            output_shape: input_shape,
        }
    }

    /// Add a node to the graph and return its ID.
    pub fn add_node(
        &mut self,
        op: GraphOp,
        inputs: Vec<usize>,
        output_shape: (usize, usize),
    ) -> usize {
        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            id,
            op,
            inputs,
            output_shape,
        });
        self.output_shape = output_shape;
        id
    }

    /// Total estimated trace rows for the entire graph.
    pub fn total_trace_rows(&self) -> usize {
        self.nodes.iter().map(|n| n.op.trace_rows()).sum()
    }

    /// Number of operations/layers in the graph.
    pub fn num_layers(&self) -> usize {
        self.nodes.len()
    }

    /// Topological order of node IDs using Kahn's algorithm.
    ///
    /// Handles multi-input nodes (e.g. Add/Mul with residual connections)
    /// correctly. For purely sequential graphs, produces `[0, 1, 2, ...]`.
    pub fn topological_order(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for node in &self.nodes {
            for &inp in &node.inputs {
                if inp < n {
                    adj[inp].push(node.id);
                    in_degree[node.id] += 1;
                }
            }
        }

        let mut queue: std::collections::VecDeque<usize> =
            (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        // If cycle detected (should never happen), fall back to sequential
        if order.len() != n {
            (0..n).collect()
        } else {
            order
        }
    }

    /// Extract a subgraph containing nodes `[start..end)`, remapping node IDs to `[0..len)`.
    ///
    /// The subgraph's `input_shape` is derived from the previous node's output shape
    /// (or the original `input_shape` if `start == 0`).
    pub fn subgraph(&self, range: std::ops::Range<usize>) -> ComputationGraph {
        let start = range.start;
        let end = range.end.min(self.nodes.len());

        let input_shape = if start == 0 {
            self.input_shape
        } else {
            self.nodes[start - 1].output_shape
        };

        let mut sub = ComputationGraph::new(input_shape);

        for (new_id, old_idx) in (start..end).enumerate() {
            let node = &self.nodes[old_idx];
            let remapped_inputs: Vec<usize> = node
                .inputs
                .iter()
                .filter_map(|&old_id| {
                    if old_id >= start && old_id < end {
                        Some(old_id - start)
                    } else {
                        None
                    }
                })
                .collect();

            sub.add_node(node.op.clone(), remapped_inputs, node.output_shape);
            let _ = new_id; // used implicitly by add_node
        }

        sub
    }

    /// Detect transformer block boundaries.
    ///
    /// A transformer block is a repeating pattern of LayerNorm → MatMul (attention) →
    /// Activation → MatMul (FFN). Returns ranges where each range covers one block.
    /// Falls back to splitting at LayerNorm boundaries, or uniform chunks if no
    /// LayerNorm is found.
    pub fn find_block_boundaries(&self) -> Vec<std::ops::Range<usize>> {
        // Find LayerNorm positions — these typically start transformer blocks
        let ln_positions: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(&n.op, GraphOp::LayerNorm { .. }).then_some(i))
            .collect();

        if ln_positions.len() >= 2 {
            // Split at LayerNorm boundaries
            let mut ranges = Vec::new();
            for i in 0..ln_positions.len() {
                let start = ln_positions[i];
                let end = if i + 1 < ln_positions.len() {
                    ln_positions[i + 1]
                } else {
                    self.nodes.len()
                };
                ranges.push(start..end);
            }
            // If first LayerNorm is not at position 0, add the prefix
            if ln_positions[0] > 0 {
                ranges.insert(0, 0..ln_positions[0]);
            }
            ranges
        } else {
            // No clear block structure — return single range
            vec![0..self.nodes.len()]
        }
    }

    /// Estimate peak proving memory across all nodes (bytes).
    ///
    /// For MatMul nodes, uses `estimate_sumcheck_memory()`. For other ops,
    /// estimates based on activation size × 16 bytes per SecureField.
    pub fn estimate_peak_memory(&self) -> usize {
        use crate::components::matmul::estimate_sumcheck_memory;

        self.nodes
            .iter()
            .map(|node| match &node.op {
                GraphOp::MatMul { dims: (m, k, n) } => {
                    let (_, total) = estimate_sumcheck_memory(*m, *k, *n);
                    total
                }
                GraphOp::Activation { size, .. } => size * 16, // SecureField per element
                GraphOp::LayerNorm { dim } => dim * 4 * 16,
                GraphOp::Attention { config } => {
                    // Q, K, V projections each d_model × d_model
                    let d = config.d_model;
                    let (_, total) = estimate_sumcheck_memory(config.seq_len, d, d);
                    total * 4 // Q, K, V, output projections
                }
                GraphOp::Dequantize { size, .. } => size * 16,
                GraphOp::Add { size } | GraphOp::Mul { size } => size * 16,
                GraphOp::Embedding {
                    vocab_size,
                    embed_dim,
                } => vocab_size * embed_dim * 4,
                GraphOp::Conv2D {
                    in_channels,
                    out_channels,
                    kernel_size,
                    ..
                } => {
                    let k = in_channels * kernel_size * kernel_size;
                    let (_, total) = estimate_sumcheck_memory(1, k, *out_channels);
                    total
                }
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }
}

/// Builder for constructing computation graphs from layer descriptions.
pub struct GraphBuilder {
    pub(crate) graph: ComputationGraph,
    pub(crate) last_node: Option<usize>,
}

impl GraphBuilder {
    pub fn new(input_shape: (usize, usize)) -> Self {
        Self {
            graph: ComputationGraph::new(input_shape),
            last_node: None,
        }
    }

    /// Add a linear (matmul) layer.
    pub fn linear(&mut self, out_features: usize) -> &mut Self {
        let input_shape = self.current_output_shape();
        let (batch, in_features) = input_shape;
        let dims = (batch, in_features, out_features);
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self
            .graph
            .add_node(GraphOp::MatMul { dims }, inputs, (batch, out_features));
        self.last_node = Some(id);
        self
    }

    /// Add an activation layer.
    pub fn activation(&mut self, act_type: ActivationType) -> &mut Self {
        let shape = self.current_output_shape();
        let size = shape.0 * shape.1;
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self.graph.add_node(
            GraphOp::Activation {
                activation_type: act_type,
                size,
            },
            inputs,
            shape,
        );
        self.last_node = Some(id);
        self
    }

    /// Add a multi-head attention block.
    pub fn attention(&mut self, config: MultiHeadAttentionConfig) -> &mut Self {
        let shape = self.current_output_shape();
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self
            .graph
            .add_node(GraphOp::Attention { config }, inputs, shape);
        self.last_node = Some(id);
        self
    }

    /// Add a GQA attention block (num_heads query heads, num_kv_heads key/value heads).
    pub fn gqa_attention(
        &mut self,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len: usize,
        causal: bool,
    ) -> &mut Self {
        let shape = self.current_output_shape();
        let d_model = shape.1;
        let config =
            MultiHeadAttentionConfig::new_gqa(num_heads, num_kv_heads, d_model, seq_len, causal);
        self.attention(config)
    }

    /// Add a layer normalization layer.
    pub fn layer_norm(&mut self) -> &mut Self {
        let shape = self.current_output_shape();
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self
            .graph
            .add_node(GraphOp::LayerNorm { dim: shape.1 }, inputs, shape);
        self.last_node = Some(id);
        self
    }

    /// Add an RMS normalization layer (no mean subtraction).
    pub fn rms_norm(&mut self) -> &mut Self {
        let shape = self.current_output_shape();
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self
            .graph
            .add_node(GraphOp::RMSNorm { dim: shape.1 }, inputs, shape);
        self.last_node = Some(id);
        self
    }

    /// Add a RoPE (Rotary Positional Embedding) layer.
    /// Applied to Q or K matrices before attention.
    /// Shape is preserved: (seq_len, head_dim) → (seq_len, head_dim).
    pub fn rope(&mut self, head_dim: usize) -> &mut Self {
        let shape = self.current_output_shape();
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();
        let config = crate::components::rope::RoPEConfig::new(shape.0, head_dim);

        let id = self.graph.add_node(GraphOp::RoPE { config }, inputs, shape);
        self.last_node = Some(id);
        self
    }

    /// Returns the ID of the most recently added node.
    pub fn current_node_id(&self) -> usize {
        self.last_node.expect("no nodes added yet")
    }

    /// Add an identity (passthrough) layer.
    pub fn identity(&mut self) -> &mut Self {
        let shape = self.current_output_shape();
        let size = shape.0 * shape.1;
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self
            .graph
            .add_node(GraphOp::Identity { size }, inputs, shape);
        self.last_node = Some(id);
        self
    }

    /// Add a dequantization layer (quantized M31 → dequantized M31 via lookup table).
    pub fn dequantize(&mut self, params: QuantParams) -> &mut Self {
        let shape = self.current_output_shape();
        let size = shape.0 * shape.1;
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self
            .graph
            .add_node(GraphOp::Dequantize { params, size }, inputs, shape);
        self.last_node = Some(id);
        self
    }

    /// Convenience: quantize → linear → dequantize.
    ///
    /// Models quantized inference: the input is quantized, the matmul runs in quantized
    /// domain, and the output is dequantized back to full precision.
    pub fn quantized_linear(&mut self, out_features: usize, params: QuantParams) -> &mut Self {
        let deq_params = params.clone();
        self.quantize(params)
            .linear(out_features)
            .dequantize(deq_params)
    }

    /// Add a quantization layer.
    pub fn quantize(&mut self, params: QuantParams) -> &mut Self {
        let shape = self.current_output_shape();
        let size = shape.0 * shape.1;
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self
            .graph
            .add_node(GraphOp::Quantize { params, size }, inputs, shape);
        self.last_node = Some(id);
        self
    }

    /// Returns the ID of the most recently added node, for use as a branch point.
    ///
    /// Use with `add_from()` / `mul_from()` to create residual connections:
    /// ```ignore
    /// let branch = builder.fork();
    /// builder.linear(64).activation(ActivationType::ReLU);
    /// builder.add_from(branch); // residual: output = branch + relu(linear(branch))
    /// ```
    pub fn fork(&self) -> usize {
        self.last_node.expect("fork() requires at least one node")
    }

    /// Add an element-wise addition node combining `lhs` (saved branch) and the current output.
    pub fn add_from(&mut self, lhs: usize) -> &mut Self {
        let shape = self.current_output_shape();
        let size = shape.0 * shape.1;
        let rhs = self.last_node.expect("add_from requires a current node");
        let id = self
            .graph
            .add_node(GraphOp::Add { size }, vec![lhs, rhs], shape);
        self.last_node = Some(id);
        self
    }

    /// Add an element-wise multiplication node combining `lhs` (saved branch) and the current output.
    pub fn mul_from(&mut self, lhs: usize) -> &mut Self {
        let shape = self.current_output_shape();
        let size = shape.0 * shape.1;
        let rhs = self.last_node.expect("mul_from requires a current node");
        let id = self
            .graph
            .add_node(GraphOp::Mul { size }, vec![lhs, rhs], shape);
        self.last_node = Some(id);
        self
    }

    /// Add an embedding lookup layer.
    pub fn embedding(&mut self, vocab_size: usize, embed_dim: usize) -> &mut Self {
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();
        let id = self.graph.add_node(
            GraphOp::Embedding {
                vocab_size,
                embed_dim,
            },
            inputs,
            (1, embed_dim),
        );
        self.last_node = Some(id);
        self
    }

    /// Add a 2D convolution layer.
    pub fn conv2d(
        &mut self,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> &mut Self {
        let shape = self.current_output_shape();
        let in_channels = shape.1;
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        // Output spatial dim: (input_dim + 2*padding - kernel_size) / stride + 1
        // Simplified: we track (batch, out_channels) as shape
        let id = self.graph.add_node(
            GraphOp::Conv2D {
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            },
            inputs,
            (shape.0, out_channels),
        );
        self.last_node = Some(id);
        self
    }

    /// Build the final computation graph.
    pub fn build(self) -> ComputationGraph {
        self.graph
    }

    /// Returns the current output shape (shape of the last node, or input_shape if empty).
    pub fn current_output_shape(&self) -> (usize, usize) {
        match self.last_node {
            Some(id) => self.graph.nodes[id].output_shape,
            None => self.graph.input_shape,
        }
    }

    /// Build a Llama-style transformer block:
    ///
    /// ```text
    /// residual = x
    /// x = RMSNorm(x)
    /// x = Attention(x)          // with GQA support
    /// x = x + residual          // residual connection
    /// residual = x
    /// x = RMSNorm(x)
    /// x = Linear(ffn_dim)       // FFN up-projection
    /// x = GELU(x)
    /// x = Linear(d_model)       // FFN down-projection
    /// x = x + residual          // residual connection
    /// ```
    pub fn transformer_block(
        &mut self,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len: usize,
        ffn_dim: usize,
    ) -> &mut Self {
        // Ensure there's a node to fork from (needed for residual connections)
        if self.last_node.is_none() {
            self.identity();
        }

        let d_model = self.current_output_shape().1;
        let attn_config = crate::components::attention::MultiHeadAttentionConfig::new_gqa(
            num_heads,
            num_kv_heads,
            d_model,
            seq_len,
            true,
        );

        // Pre-attention norm + attention + residual
        let residual1 = self.fork();
        self.rms_norm().attention(attn_config).add_from(residual1);

        // Pre-FFN norm + FFN + residual
        let residual2 = self.fork();
        self.rms_norm()
            .linear(ffn_dim)
            .activation(ActivationType::GELU)
            .linear(d_model)
            .add_from(residual2)
    }
}

/// Result of executing a computation graph.
#[derive(Debug, Clone)]
pub struct GraphExecution {
    /// Per-node intermediate results (input to each node).
    pub intermediates: Vec<(usize, M31Matrix)>,
    /// Per-node output results. Used by `get_binary_op_intermediates` to
    /// look up the OUTPUT of input layers (Add/Mul need outputs, not inputs).
    pub node_outputs: std::collections::HashMap<usize, M31Matrix>,
    /// Final output.
    pub output: M31Matrix,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_mlp() {
        let mut builder = GraphBuilder::new((1, 784)); // MNIST input

        builder
            .linear(128)
            .activation(ActivationType::ReLU)
            .linear(64)
            .activation(ActivationType::ReLU)
            .linear(10);

        let graph = builder.build();

        assert_eq!(graph.num_layers(), 5);
        assert_eq!(graph.input_shape, (1, 784));
        assert_eq!(graph.output_shape, (1, 10));

        let total = graph.total_trace_rows();
        assert!(total > 0, "trace rows should be positive: {total}");
    }

    #[test]
    fn test_graph_trace_budget() {
        let mut builder = GraphBuilder::new((1, 768));
        builder
            .linear(768)
            .layer_norm()
            .activation(ActivationType::GELU)
            .linear(3072) // FFN expansion
            .activation(ActivationType::GELU)
            .linear(768); // FFN contraction

        let graph = builder.build();
        let total = graph.total_trace_rows();

        // Transformer block should have manageable trace
        println!("Transformer block trace rows: {total}");
        assert!(total < 10_000_000, "trace budget exceeded");
    }

    #[test]
    fn test_builder_identity_and_quantize() {
        let mut builder = GraphBuilder::new((1, 8));
        builder
            .linear(8)
            .identity()
            .quantize(QuantParams {
                strategy: crate::gadgets::quantize::QuantStrategy::Direct,
                scale: 1.0,
                zero_point: 0,
                bits: 8,
            })
            .linear(4);

        let graph = builder.build();

        assert_eq!(graph.num_layers(), 4);
        // Identity and Quantize preserve shape
        assert_eq!(graph.nodes[1].output_shape, (1, 8));
        assert_eq!(graph.nodes[2].output_shape, (1, 8));
        assert_eq!(graph.output_shape, (1, 4));

        // Identity has 0 trace rows, Quantize has size trace rows
        assert_eq!(graph.nodes[1].op.trace_rows(), 0);
        assert_eq!(graph.nodes[2].op.trace_rows(), 8); // 1*8
    }

    #[test]
    fn test_graph_weights() {
        let mut weights = GraphWeights::new();
        let w = M31Matrix::new(2, 3);
        weights.add_weight(0, w);
        weights.add_bias(0, vec![M31::from(1), M31::from(2)]);

        assert!(weights.get_weight(0).is_some());
        assert!(weights.get_bias(0).is_some());
        assert!(weights.get_weight(1).is_none());
    }

    #[test]
    fn test_subgraph_basic() {
        // 6-node graph: matmul, relu, matmul, relu, matmul, relu
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(4)
            .activation(ActivationType::ReLU);
        let graph = builder.build();
        assert_eq!(graph.num_layers(), 6);

        // Extract first 3 nodes (matmul, relu, matmul)
        let sub = graph.subgraph(0..3);
        assert_eq!(sub.num_layers(), 3);
        assert_eq!(sub.input_shape, (1, 4));
        assert_eq!(sub.output_shape, (1, 4));

        // Extract nodes 3..6 (relu, matmul, relu)
        let sub2 = graph.subgraph(3..6);
        assert_eq!(sub2.num_layers(), 3);
        assert_eq!(sub2.input_shape, (1, 4)); // from node 2's output
    }

    #[test]
    fn test_subgraph_single_node() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let sub = graph.subgraph(0..1);
        assert_eq!(sub.num_layers(), 1);
        assert_eq!(sub.input_shape, (1, 4));
        assert_eq!(sub.output_shape, (1, 2));
    }

    #[test]
    fn test_weights_subset() {
        let mut weights = GraphWeights::new();
        weights.add_weight(0, M31Matrix::new(4, 4));
        weights.add_weight(2, M31Matrix::new(4, 4));
        weights.add_weight(4, M31Matrix::new(4, 2));
        weights.add_bias(0, vec![M31::from(1)]);
        weights.add_bias(2, vec![M31::from(2)]);

        // Subset for nodes 2..5 → remapped to 0..3
        let sub = weights.subset(2..5);
        assert!(sub.get_weight(0).is_some()); // was node 2
        assert!(sub.get_weight(2).is_some()); // was node 4
        assert!(sub.get_weight(1).is_none()); // node 3 had no weight
        assert!(sub.get_bias(0).is_some()); // was node 2
    }

    #[test]
    fn test_find_block_boundaries_with_layernorm() {
        let mut builder = GraphBuilder::new((1, 64));
        // Block 0: LN → MatMul → ReLU → MatMul
        builder
            .layer_norm()
            .linear(64)
            .activation(ActivationType::ReLU)
            .linear(64);
        // Block 1: LN → MatMul → ReLU → MatMul
        builder
            .layer_norm()
            .linear(64)
            .activation(ActivationType::ReLU)
            .linear(64);
        let graph = builder.build();

        let blocks = graph.find_block_boundaries();
        assert_eq!(blocks.len(), 2, "should detect 2 transformer blocks");
        assert_eq!(blocks[0].start, 0);
        assert_eq!(blocks[1].start, 4);
    }

    #[test]
    fn test_find_block_boundaries_no_layernorm() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let blocks = graph.find_block_boundaries();
        assert_eq!(blocks.len(), 1, "no layernorms → single block");
        assert_eq!(blocks[0], 0..3);
    }

    #[test]
    fn test_estimate_peak_memory() {
        let mut builder = GraphBuilder::new((1, 128));
        builder.linear(128);
        let graph = builder.build();

        let peak = graph.estimate_peak_memory();
        assert!(peak > 0, "peak memory should be positive");
    }

    #[test]
    fn test_add_mul_graph_ops() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let branch = builder.fork();
        builder.activation(ActivationType::ReLU);
        builder.add_from(branch); // residual connection: x + relu(linear(x))

        let graph = builder.build();
        assert_eq!(graph.num_layers(), 3); // linear, relu, add
        assert_eq!(graph.output_shape, (1, 4));

        // The Add node should have 2 inputs
        let add_node = &graph.nodes[2];
        assert!(matches!(add_node.op, GraphOp::Add { .. }));
        assert_eq!(add_node.inputs.len(), 2);
        assert_eq!(add_node.inputs[0], 0); // branch = linear output
        assert_eq!(add_node.inputs[1], 1); // relu output
    }

    #[test]
    fn test_mul_op() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let branch = builder.fork();
        builder.linear(4);
        builder.mul_from(branch);

        let graph = builder.build();
        assert_eq!(graph.num_layers(), 3); // linear, linear, mul
        let mul_node = &graph.nodes[2];
        assert!(matches!(mul_node.op, GraphOp::Mul { .. }));
        assert_eq!(mul_node.inputs.len(), 2);
    }

    #[test]
    fn test_residual_connection_graph() {
        // Build a residual block: y = x + matmul(relu(matmul(x)))
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let residual = builder.fork(); // save after first linear
        builder.activation(ActivationType::ReLU);
        builder.linear(4);
        builder.add_from(residual); // residual connection

        let graph = builder.build();
        assert_eq!(graph.num_layers(), 4); // linear, relu, linear, add
        assert_eq!(graph.output_shape, (1, 4));

        // Verify topo order handles multi-input
        let topo = graph.topological_order();
        assert_eq!(topo.len(), 4);
        // Add node (3) should come after both linear(0) and linear(2)
        let add_pos = topo.iter().position(|&x| x == 3).unwrap();
        let lin0_pos = topo.iter().position(|&x| x == 0).unwrap();
        let lin2_pos = topo.iter().position(|&x| x == 2).unwrap();
        assert!(add_pos > lin0_pos);
        assert!(add_pos > lin2_pos);
    }

    #[test]
    fn test_topo_sort_diamond() {
        // Diamond graph: 0 → 1, 0 → 2, 1+2 → 3(add)
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4); // node 0
        let branch = builder.fork(); // save node 0
        builder.activation(ActivationType::ReLU); // node 1, depends on 0
                                                  // Now we want node 2 to also depend on 0, then add 1+2
                                                  // But with current API, 'current' is node 1.
                                                  // So we use add_from(branch) to combine node 0 + node 1
        builder.add_from(branch); // node 2 = add(0, 1)

        let graph = builder.build();
        let topo = graph.topological_order();
        assert_eq!(topo.len(), 3);

        // Node 2 must come after both 0 and 1
        let add_pos = topo.iter().position(|&x| x == 2).unwrap();
        assert!(add_pos > topo.iter().position(|&x| x == 0).unwrap());
        assert!(add_pos > topo.iter().position(|&x| x == 1).unwrap());
    }

    #[test]
    fn test_embedding_op() {
        let mut builder = GraphBuilder::new((1, 1));
        builder.embedding(1000, 768);
        let graph = builder.build();

        assert_eq!(graph.num_layers(), 1);
        assert_eq!(graph.output_shape, (1, 768));
        assert!(matches!(
            graph.nodes[0].op,
            GraphOp::Embedding {
                vocab_size: 1000,
                embed_dim: 768
            }
        ));
    }

    #[test]
    fn test_conv2d_op() {
        let mut builder = GraphBuilder::new((1, 3));
        builder.conv2d(16, 3, 1, 1);
        let graph = builder.build();

        assert_eq!(graph.num_layers(), 1);
        assert!(matches!(graph.nodes[0].op, GraphOp::Conv2D { .. }));
    }

    #[test]
    fn test_add_trace_rows() {
        let op = GraphOp::Add { size: 256 };
        assert_eq!(op.trace_rows(), 256);
    }

    #[test]
    fn test_mul_trace_rows() {
        let op = GraphOp::Mul { size: 512 };
        assert_eq!(op.trace_rows(), 512);
    }

    #[test]
    fn test_builder_dequantize() {
        use crate::gadgets::quantize::QuantStrategy;
        let params = QuantParams {
            strategy: QuantStrategy::Symmetric8,
            scale: 0.01,
            zero_point: 127,
            bits: 8,
        };
        let mut builder = GraphBuilder::new((1, 8));
        builder.linear(8).dequantize(params);
        let graph = builder.build();

        assert_eq!(graph.num_layers(), 2);
        assert!(matches!(graph.nodes[1].op, GraphOp::Dequantize { .. }));
        assert_eq!(graph.nodes[1].output_shape, (1, 8));
        // Trace rows: max(size=8, 2^8=256) = 256
        assert_eq!(graph.nodes[1].op.trace_rows(), 256);
    }

    #[test]
    fn test_quantized_linear_convenience() {
        use crate::gadgets::quantize::QuantStrategy;
        let params = QuantParams {
            strategy: QuantStrategy::Symmetric4,
            scale: 0.1,
            zero_point: 7,
            bits: 4,
        };
        let mut builder = GraphBuilder::new((1, 8));
        builder.quantized_linear(4, params);
        let graph = builder.build();

        // Should produce: Quantize → MatMul → Dequantize
        assert_eq!(graph.num_layers(), 3);
        assert!(matches!(graph.nodes[0].op, GraphOp::Quantize { .. }));
        assert!(matches!(graph.nodes[1].op, GraphOp::MatMul { .. }));
        assert!(matches!(graph.nodes[2].op, GraphOp::Dequantize { .. }));
        assert_eq!(graph.output_shape, (1, 4));
    }

    #[test]
    fn test_dequantize_trace_rows_int4() {
        use crate::gadgets::quantize::QuantStrategy;
        // For INT4 with 2 elements: max(2, 16) = 16
        let op = GraphOp::Dequantize {
            params: QuantParams {
                strategy: QuantStrategy::Symmetric4,
                scale: 0.1,
                zero_point: 7,
                bits: 4,
            },
            size: 2,
        };
        assert_eq!(op.trace_rows(), 16);
    }

    #[test]
    fn test_transformer_block_builder() {
        // Llama-style: 8 Q heads, 2 KV heads (GQA), d_model=32, seq_len=4
        let mut builder = GraphBuilder::new((4, 32)); // (seq_len, d_model)
        builder.transformer_block(
            8,  // num_heads
            2,  // num_kv_heads (GQA)
            4,  // seq_len
            64, // ffn_dim (2× expansion)
        );
        let graph = builder.build();

        // Expected ops:
        // Identity(input anchor),
        // RMSNorm, Attention, Add(residual),
        // RMSNorm, MatMul(up), GELU, MatMul(down), Add(residual)
        // = 9 nodes
        assert_eq!(graph.num_layers(), 9);
        assert_eq!(graph.input_shape, (4, 32));
        assert_eq!(graph.output_shape, (4, 32)); // preserved through block
    }

    #[test]
    fn test_multi_transformer_block() {
        // Stack 3 transformer blocks
        let mut builder = GraphBuilder::new((4, 16));
        for _ in 0..3 {
            builder.transformer_block(4, 2, 4, 32);
        }
        let graph = builder.build();

        // 1 identity + 8 ops per block × 3 = 25 (identity only on first block)
        assert_eq!(graph.num_layers(), 25);
        assert_eq!(graph.output_shape, (4, 16));
    }
}
