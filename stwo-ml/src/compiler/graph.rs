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
    /// Quantization / dequantization step.
    Quantize {
        params: QuantParams,
        size: usize,
    },
    /// Multi-head attention block.
    Attention {
        config: MultiHeadAttentionConfig,
    },
    /// Identity / passthrough (for graph structure).
    Identity {
        size: usize,
    },
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
            GraphOp::Quantize { size, .. } => {
                // Range check per element
                *size
            }
            GraphOp::Attention { config } => {
                config.sumcheck_trace_rows()
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
    pub fn add_node(&mut self, op: GraphOp, inputs: Vec<usize>, output_shape: (usize, usize)) -> usize {
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

    /// Topological order of node IDs (already in order since we add sequentially).
    pub fn topological_order(&self) -> Vec<usize> {
        (0..self.nodes.len()).collect()
    }
}

/// Builder for constructing computation graphs from layer descriptions.
pub struct GraphBuilder {
    graph: ComputationGraph,
    last_node: Option<usize>,
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

        let id = self.graph.add_node(
            GraphOp::MatMul { dims },
            inputs,
            (batch, out_features),
        );
        self.last_node = Some(id);
        self
    }

    /// Add an activation layer.
    pub fn activation(&mut self, act_type: ActivationType) -> &mut Self {
        let shape = self.current_output_shape();
        let size = shape.0 * shape.1;
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self.graph.add_node(
            GraphOp::Activation { activation_type: act_type, size },
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

        let id = self.graph.add_node(
            GraphOp::Attention { config },
            inputs,
            shape,
        );
        self.last_node = Some(id);
        self
    }

    /// Add a layer normalization layer.
    pub fn layer_norm(&mut self) -> &mut Self {
        let shape = self.current_output_shape();
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self.graph.add_node(
            GraphOp::LayerNorm { dim: shape.1 },
            inputs,
            shape,
        );
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

        let id = self.graph.add_node(
            GraphOp::Identity { size },
            inputs,
            shape,
        );
        self.last_node = Some(id);
        self
    }

    /// Add a quantization layer.
    pub fn quantize(&mut self, params: QuantParams) -> &mut Self {
        let shape = self.current_output_shape();
        let size = shape.0 * shape.1;
        let inputs = self.last_node.map(|n| vec![n]).unwrap_or_default();

        let id = self.graph.add_node(
            GraphOp::Quantize { params, size },
            inputs,
            shape,
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
}

/// Result of executing a computation graph.
#[derive(Debug, Clone)]
pub struct GraphExecution {
    /// Per-node intermediate results.
    pub intermediates: Vec<(usize, M31Matrix)>,
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
            .linear(3072)  // FFN expansion
            .activation(ActivationType::GELU)
            .linear(768);  // FFN contraction

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
}
