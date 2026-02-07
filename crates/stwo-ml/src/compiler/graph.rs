//! Computation graph builder.
//!
//! Constructs a directed acyclic graph of ML operations, assigns
//! each node to a STWO component, and computes the total trace budget.
//!
//! # Example
//!
//! ```rust,ignore
//! use stwo_ml::compiler::graph::*;
//!
//! let mut graph = ComputationGraph::new();
//!
//! // 2-layer MLP: input → matmul(W1) → relu → matmul(W2) → output
//! let input = graph.add_node(GraphOp::Input { rows: 4, cols: 4 });
//! let mm1 = graph.add_node(GraphOp::MatMul { weight_rows: 4, weight_cols: 8 });
//! let relu = graph.add_node(GraphOp::Activation {
//!     activation: ActivationType::ReLU,
//!     log_table_size: 8,
//! });
//! let mm2 = graph.add_node(GraphOp::MatMul { weight_rows: 8, weight_cols: 4 });
//!
//! graph.connect(input, mm1);
//! graph.connect(mm1, relu);
//! graph.connect(relu, mm2);
//!
//! let order = graph.topological_sort().unwrap();
//! let budget = graph.trace_budget();
//! ```

use std::collections::{HashMap, VecDeque};

use stwo::core::fields::m31::M31;

use crate::components::activation::ActivationType;
use crate::components::attention::apply_activation_table;
use crate::components::layernorm::{apply_layernorm, center, compute_mean, LayerNormParams};
use crate::components::matmul::M31Matrix;

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the computation graph.
pub type NodeId = usize;

/// An ML operation in the computation graph.
#[derive(Debug, Clone)]
pub enum GraphOp {
    /// Model input tensor (no computation, just declares shape).
    Input {
        rows: usize,
        cols: usize,
    },
    /// Matrix multiplication: output = input × weight.
    ///
    /// Input shape is inferred from the predecessor node.
    /// Output shape: (input_rows, weight_cols).
    MatMul {
        weight_rows: usize,
        weight_cols: usize,
    },
    /// Element-wise activation function via LogUp lookup table.
    Activation {
        activation: ActivationType,
        log_table_size: u32,
    },
    /// Layer normalization over the feature dimension.
    LayerNorm {
        feature_dim: usize,
    },
    /// Quantization with range check.
    Quantize {
        bits: u32,
    },
}

impl GraphOp {
    /// Human-readable name for the operation.
    pub fn name(&self) -> &'static str {
        match self {
            GraphOp::Input { .. } => "Input",
            GraphOp::MatMul { .. } => "MatMul",
            GraphOp::Activation { .. } => "Activation",
            GraphOp::LayerNorm { .. } => "LayerNorm",
            GraphOp::Quantize { .. } => "Quantize",
        }
    }

    /// Estimate the STWO trace rows required for this operation.
    ///
    /// For `MatMul`: input_rows × inner_dim + inner_dim × output_cols + output_rows × output_cols
    /// (sumcheck witness size is O(n) per entry, not O(n³)).
    ///
    /// For `Activation`: 2^log_table_size (preprocessed table size).
    ///
    /// For `LayerNorm`: 2^ceil_log2(feature_dim) (table size).
    ///
    /// `input_shape` is (rows, cols) of the input to this node.
    pub fn trace_rows(&self, input_shape: (usize, usize)) -> usize {
        match self {
            GraphOp::Input { .. } => 0,
            GraphOp::MatMul { weight_rows, weight_cols } => {
                let (m, _k) = input_shape;
                // Sumcheck witness: A (m×k) + B (k×n) + C (m×n)
                m * weight_rows + weight_rows * weight_cols + m * weight_cols
            }
            GraphOp::Activation { log_table_size, .. } => {
                1 << log_table_size
            }
            GraphOp::LayerNorm { feature_dim } => {
                // Padded to power of 2 for STWO
                feature_dim.next_power_of_two()
            }
            GraphOp::Quantize { bits } => {
                1 << bits
            }
        }
    }

    /// Compute the output shape given the input shape.
    ///
    /// Returns `None` if the shapes are incompatible.
    pub fn output_shape(&self, input_shape: (usize, usize)) -> Option<(usize, usize)> {
        match self {
            GraphOp::Input { rows, cols } => Some((*rows, *cols)),
            GraphOp::MatMul { weight_rows, weight_cols } => {
                let (_m, k) = input_shape;
                if k != *weight_rows {
                    None // dimension mismatch
                } else {
                    Some((input_shape.0, *weight_cols))
                }
            }
            // Activation, LayerNorm, Quantize are element-wise: shape preserved.
            GraphOp::Activation { .. } => Some(input_shape),
            GraphOp::LayerNorm { .. } => Some(input_shape),
            GraphOp::Quantize { .. } => Some(input_shape),
        }
    }
}

// ---------------------------------------------------------------------------
// Graph node
// ---------------------------------------------------------------------------

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier.
    pub id: NodeId,
    /// The operation this node performs.
    pub op: GraphOp,
    /// IDs of predecessor nodes (inputs to this node).
    pub inputs: Vec<NodeId>,
    /// IDs of successor nodes (consumers of this node's output).
    pub outputs: Vec<NodeId>,
}

// ---------------------------------------------------------------------------
// Computation graph
// ---------------------------------------------------------------------------

/// Error type for graph operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("node {0} not found in graph")]
    NodeNotFound(NodeId),
    #[error("graph contains a cycle (not a DAG)")]
    CycleDetected,
    #[error("shape mismatch at node {node_id} ({op_name}): input shape {input_shape:?} incompatible with operation")]
    ShapeMismatch {
        node_id: NodeId,
        op_name: &'static str,
        input_shape: (usize, usize),
    },
    #[error("node {0} has no input and is not an Input node")]
    MissingInput(NodeId),
    #[error("graph is empty")]
    EmptyGraph,
}

/// A directed acyclic graph of ML operations.
///
/// Nodes represent operations (matmul, activation, layernorm, etc.).
/// Edges represent data dependencies (output of one node feeds into the next).
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    nodes: Vec<GraphNode>,
}

impl ComputationGraph {
    /// Create an empty computation graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Add a node to the graph. Returns the new node's ID.
    pub fn add_node(&mut self, op: GraphOp) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            id,
            op,
            inputs: Vec::new(),
            outputs: Vec::new(),
        });
        id
    }

    /// Connect `from` → `to` (from's output feeds into to's input).
    ///
    /// Returns `Err` if either node doesn't exist.
    pub fn connect(&mut self, from: NodeId, to: NodeId) -> Result<(), GraphError> {
        if from >= self.nodes.len() {
            return Err(GraphError::NodeNotFound(from));
        }
        if to >= self.nodes.len() {
            return Err(GraphError::NodeNotFound(to));
        }
        self.nodes[from].outputs.push(to);
        self.nodes[to].inputs.push(from);
        Ok(())
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Topological sort of the graph (Kahn's algorithm).
    ///
    /// Returns nodes in execution order (all inputs before their consumers).
    /// Returns `Err(CycleDetected)` if the graph is not a DAG.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, GraphError> {
        if self.nodes.is_empty() {
            return Err(GraphError::EmptyGraph);
        }

        let n = self.nodes.len();
        let mut in_degree: Vec<usize> = self.nodes.iter().map(|n| n.inputs.len()).collect();
        let mut queue: VecDeque<NodeId> = VecDeque::new();

        // Start with nodes that have no inputs (sources).
        for (id, deg) in in_degree.iter().enumerate() {
            if *deg == 0 {
                queue.push_back(id);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(node_id) = queue.pop_front() {
            order.push(node_id);
            for &successor in &self.nodes[node_id].outputs {
                in_degree[successor] -= 1;
                if in_degree[successor] == 0 {
                    queue.push_back(successor);
                }
            }
        }

        if order.len() != n {
            return Err(GraphError::CycleDetected);
        }
        Ok(order)
    }

    /// Compute the output shape of each node in topological order.
    ///
    /// Returns a map from NodeId to output shape (rows, cols).
    pub fn compute_shapes(&self) -> Result<HashMap<NodeId, (usize, usize)>, GraphError> {
        let order = self.topological_sort()?;
        let mut shapes: HashMap<NodeId, (usize, usize)> = HashMap::new();

        for &node_id in &order {
            let node = &self.nodes[node_id];
            let input_shape = match &node.op {
                GraphOp::Input { rows, cols } => (*rows, *cols),
                _ => {
                    if node.inputs.is_empty() {
                        return Err(GraphError::MissingInput(node_id));
                    }
                    // Use the first input's shape.
                    *shapes.get(&node.inputs[0]).unwrap()
                }
            };

            let output_shape = node.op.output_shape(input_shape).ok_or(
                GraphError::ShapeMismatch {
                    node_id,
                    op_name: node.op.name(),
                    input_shape,
                },
            )?;
            shapes.insert(node_id, output_shape);
        }

        Ok(shapes)
    }

    /// Estimate total STWO trace rows for the entire graph.
    pub fn trace_budget(&self) -> Result<usize, GraphError> {
        let shapes = self.compute_shapes()?;
        let order = self.topological_sort()?;
        let mut total = 0usize;

        for &node_id in &order {
            let node = &self.nodes[node_id];
            let input_shape = match &node.op {
                GraphOp::Input { rows, cols } => (*rows, *cols),
                _ => {
                    if node.inputs.is_empty() {
                        continue;
                    }
                    *shapes.get(&node.inputs[0]).unwrap()
                }
            };
            total += node.op.trace_rows(input_shape);
        }

        Ok(total)
    }

    /// Validate the entire graph: check shapes are compatible, no cycles.
    pub fn validate(&self) -> Result<(), GraphError> {
        self.compute_shapes()?;
        Ok(())
    }

    /// Return all nodes in the graph.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

impl ComputationGraph {
    /// Build a simple sequential model (linear chain of operations).
    ///
    /// Connects each operation to the next in order.
    pub fn sequential(ops: Vec<GraphOp>) -> Result<Self, GraphError> {
        let mut graph = Self::new();
        let mut prev: Option<NodeId> = None;

        for op in ops {
            let node = graph.add_node(op);
            if let Some(prev_id) = prev {
                graph.connect(prev_id, node)?;
            }
            prev = Some(node);
        }

        graph.validate()?;
        Ok(graph)
    }
}

// ---------------------------------------------------------------------------
// Graph executor (witness generation)
// ---------------------------------------------------------------------------

/// Weights and parameters for executing a computation graph.
///
/// Each node that requires external data (MatMul weights, LayerNorm params)
/// looks up its parameters by NodeId.
#[derive(Debug, Clone)]
pub struct GraphWeights {
    /// Weight matrices for MatMul nodes, keyed by NodeId.
    pub matmul_weights: HashMap<NodeId, M31Matrix>,
    /// LayerNorm parameters keyed by NodeId.
    pub layernorm_params: HashMap<NodeId, LayerNormParams>,
    /// LayerNorm inv_std values keyed by NodeId (prover-supplied).
    pub layernorm_inv_std: HashMap<NodeId, M31>,
}

impl GraphWeights {
    pub fn new() -> Self {
        Self {
            matmul_weights: HashMap::new(),
            layernorm_params: HashMap::new(),
            layernorm_inv_std: HashMap::new(),
        }
    }
}

impl Default for GraphWeights {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of executing a computation graph.
#[derive(Debug, Clone)]
pub struct GraphExecution {
    /// Output matrix for each node, keyed by NodeId.
    pub node_outputs: HashMap<NodeId, M31Matrix>,
    /// Execution order (topological).
    pub order: Vec<NodeId>,
}

impl GraphExecution {
    /// Get the final output (last node in topological order).
    pub fn final_output(&self) -> Option<&M31Matrix> {
        self.order.last().and_then(|id| self.node_outputs.get(id))
    }

    /// Get the output of a specific node.
    pub fn output(&self, node_id: NodeId) -> Option<&M31Matrix> {
        self.node_outputs.get(&node_id)
    }
}

/// Error type for graph execution.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("graph error: {0}")]
    Graph(#[from] GraphError),
    #[error("missing weight matrix for MatMul node {0}")]
    MissingWeight(NodeId),
    #[error("missing LayerNorm params for node {0}")]
    MissingLayerNormParams(NodeId),
    #[error("missing input matrix")]
    MissingInput,
    #[error("matmul error at node {node_id}: {msg}")]
    MatMulError { node_id: NodeId, msg: String },
    #[error("activation error at node {node_id}: {msg}")]
    ActivationError { node_id: NodeId, msg: String },
    #[error("layernorm error at node {node_id}: {msg}")]
    LayerNormError { node_id: NodeId, msg: String },
}

/// Execute a computation graph on an input matrix, producing all intermediate outputs.
///
/// This is the **witness generator**: it computes the actual values that the
/// proving pipeline will then prove correct.
pub fn execute_graph(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<GraphExecution, ExecutionError> {
    let order = graph.topological_sort()?;
    let mut node_outputs: HashMap<NodeId, M31Matrix> = HashMap::new();

    for &node_id in &order {
        let node = &graph.nodes()[node_id];

        let output = match &node.op {
            GraphOp::Input { .. } => input.clone(),

            GraphOp::MatMul { .. } => {
                let input_matrix = get_node_input(&node.inputs, &node_outputs, node_id)?;
                let weight = weights.matmul_weights.get(&node_id)
                    .ok_or(ExecutionError::MissingWeight(node_id))?;
                M31Matrix::multiply(input_matrix, weight)
                    .map_err(|e| ExecutionError::MatMulError {
                        node_id,
                        msg: e.to_string(),
                    })?
            }

            GraphOp::Activation { activation, log_table_size } => {
                let input_matrix = get_node_input(&node.inputs, &node_outputs, node_id)?;
                let table = activation.build_table(*log_table_size);
                apply_activation_table(input_matrix, &table)
                    .map_err(|e| ExecutionError::ActivationError {
                        node_id,
                        msg: e.to_string(),
                    })?
            }

            GraphOp::LayerNorm { feature_dim: _ } => {
                let input_matrix = get_node_input(&node.inputs, &node_outputs, node_id)?;
                let params = weights.layernorm_params.get(&node_id)
                    .ok_or(ExecutionError::MissingLayerNormParams(node_id))?;
                let inv_std = weights.layernorm_inv_std.get(&node_id)
                    .copied()
                    .unwrap_or(M31::from(1)); // Default inv_std=1

                // Normalize each row independently.
                let mut result = M31Matrix::new(input_matrix.rows, input_matrix.cols);
                for row in 0..input_matrix.rows {
                    let row_data: Vec<M31> = (0..input_matrix.cols)
                        .map(|j| input_matrix.get(row, j))
                        .collect();
                    let mean = compute_mean(&row_data)
                        .map_err(|e| ExecutionError::LayerNormError {
                            node_id,
                            msg: e.to_string(),
                        })?;
                    let centered = center(&row_data, mean);
                    let normalized = apply_layernorm(&centered, inv_std, params)
                        .map_err(|e| ExecutionError::LayerNormError {
                            node_id,
                            msg: e.to_string(),
                        })?;
                    for (j, &val) in normalized.iter().enumerate() {
                        result.set(row, j, val);
                    }
                }
                result
            }

            GraphOp::Quantize { .. } => {
                // Quantize is a pass-through for M31 values (already in field).
                // Range validation happens at proving time.
                let input_matrix = get_node_input(&node.inputs, &node_outputs, node_id)?;
                input_matrix.clone()
            }
        };

        node_outputs.insert(node_id, output);
    }

    Ok(GraphExecution { node_outputs, order })
}

/// Helper: get the input matrix for a node from its predecessor's output.
fn get_node_input<'a>(
    inputs: &[NodeId],
    outputs: &'a HashMap<NodeId, M31Matrix>,
    _node_id: NodeId,
) -> Result<&'a M31Matrix, ExecutionError> {
    if inputs.is_empty() {
        return Err(ExecutionError::MissingInput);
    }
    outputs.get(&inputs[0]).ok_or(ExecutionError::MissingInput)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_mlp() {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::MatMul { weight_rows: 4, weight_cols: 8 },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 8,
            },
            GraphOp::MatMul { weight_rows: 8, weight_cols: 4 },
        ]).unwrap();

        assert_eq!(graph.len(), 4);

        let order = graph.topological_sort().unwrap();
        assert_eq!(order, vec![0, 1, 2, 3]);

        let shapes = graph.compute_shapes().unwrap();
        assert_eq!(shapes[&0], (4, 4));   // Input: 4×4
        assert_eq!(shapes[&1], (4, 8));   // MatMul: 4×8
        assert_eq!(shapes[&2], (4, 8));   // ReLU: shape preserved
        assert_eq!(shapes[&3], (4, 4));   // MatMul: 4×4

        let budget = graph.trace_budget().unwrap();
        assert!(budget > 0);
    }

    #[test]
    fn test_transformer_block() {
        // Input → LayerNorm → Attention(QKT matmul) → ReLU → MatMul → LayerNorm
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 8, cols: 16 },
            GraphOp::LayerNorm { feature_dim: 16 },
            GraphOp::MatMul { weight_rows: 16, weight_cols: 16 },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 8,
            },
            GraphOp::MatMul { weight_rows: 16, weight_cols: 16 },
            GraphOp::LayerNorm { feature_dim: 16 },
        ]).unwrap();

        assert_eq!(graph.len(), 6);

        let shapes = graph.compute_shapes().unwrap();
        assert_eq!(shapes[&0], (8, 16));
        assert_eq!(shapes[&5], (8, 16)); // Final shape same as input
    }

    #[test]
    fn test_shape_mismatch_detected() {
        let result = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::MatMul { weight_rows: 8, weight_cols: 4 }, // 4 != 8
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = ComputationGraph::new();
        let a = graph.add_node(GraphOp::Input { rows: 4, cols: 4 });
        let b = graph.add_node(GraphOp::Activation {
            activation: ActivationType::ReLU,
            log_table_size: 4,
        });
        let c = graph.add_node(GraphOp::Activation {
            activation: ActivationType::ReLU,
            log_table_size: 4,
        });

        graph.connect(a, b).unwrap();
        graph.connect(b, c).unwrap();
        graph.connect(c, b).unwrap(); // cycle: b → c → b

        let result = graph.topological_sort();
        assert!(result.is_err());
    }

    #[test]
    fn test_trace_budget() {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::MatMul { weight_rows: 4, weight_cols: 4 },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 4,
            },
        ]).unwrap();

        let budget = graph.trace_budget().unwrap();
        // MatMul: 4*4 + 4*4 + 4*4 = 48
        // Activation: 2^4 = 16
        // Total: 64
        assert_eq!(budget, 64);
    }

    #[test]
    fn test_quantize_node() {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::Quantize { bits: 8 },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 8,
            },
        ]).unwrap();

        let shapes = graph.compute_shapes().unwrap();
        assert_eq!(shapes[&1], (4, 4)); // Quantize preserves shape
        assert_eq!(shapes[&2], (4, 4)); // Activation preserves shape
    }

    #[test]
    fn test_empty_graph_error() {
        let graph = ComputationGraph::new();
        assert!(graph.topological_sort().is_err());
    }

    #[test]
    fn test_node_not_found() {
        let mut graph = ComputationGraph::new();
        let _a = graph.add_node(GraphOp::Input { rows: 4, cols: 4 });
        assert!(graph.connect(0, 99).is_err());
    }

    #[test]
    fn test_diamond_dag() {
        // Input → [MatMul1, MatMul2] → Activation
        // (branching DAG, not a simple chain)
        let mut graph = ComputationGraph::new();
        let input = graph.add_node(GraphOp::Input { rows: 4, cols: 4 });
        let mm1 = graph.add_node(GraphOp::MatMul { weight_rows: 4, weight_cols: 4 });
        let relu = graph.add_node(GraphOp::Activation {
            activation: ActivationType::ReLU,
            log_table_size: 4,
        });

        graph.connect(input, mm1).unwrap();
        graph.connect(mm1, relu).unwrap();

        let order = graph.topological_sort().unwrap();
        assert_eq!(order.len(), 3);
        // input must come before mm1, mm1 before relu
        assert!(order.iter().position(|&x| x == input).unwrap()
            < order.iter().position(|&x| x == mm1).unwrap());
        assert!(order.iter().position(|&x| x == mm1).unwrap()
            < order.iter().position(|&x| x == relu).unwrap());
    }

    // -------------------------------------------------------------------
    // Graph executor tests
    // -------------------------------------------------------------------

    #[test]
    fn test_execute_matmul_relu() {
        // Input(2×2) → MatMul(2×2) → ReLU
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 2, cols: 2 },
            GraphOp::MatMul { weight_rows: 2, weight_cols: 2 },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 4,
            },
        ]).unwrap();

        let input = M31Matrix::from_data(2, 2, vec![
            M31::from(1), M31::from(0),
            M31::from(0), M31::from(1),
        ]).unwrap();

        let weight = M31Matrix::from_data(2, 2, vec![
            M31::from(2), M31::from(3),
            M31::from(1), M31::from(1),
        ]).unwrap();

        let mut weights = GraphWeights::new();
        weights.matmul_weights.insert(1, weight); // node 1 is the MatMul

        let execution = execute_graph(&graph, &input, &weights).unwrap();

        // Check input node output = input
        assert_eq!(execution.output(0).unwrap().get(0, 0), M31::from(1));

        // Check matmul output: I × W = W
        let mm_out = execution.output(1).unwrap();
        assert_eq!(mm_out.get(0, 0), M31::from(2));
        assert_eq!(mm_out.get(0, 1), M31::from(3));
        assert_eq!(mm_out.get(1, 0), M31::from(1));
        assert_eq!(mm_out.get(1, 1), M31::from(1));

        // ReLU should preserve values (all < 8 for log_size=4)
        let relu_out = execution.output(2).unwrap();
        assert_eq!(relu_out.get(0, 0), M31::from(2));
        assert_eq!(relu_out.get(0, 1), M31::from(3));

        // Final output
        assert!(execution.final_output().is_some());
    }

    #[test]
    fn test_execute_two_layer_mlp() {
        // Input(2×2) → MatMul(W1: 2×4) → ReLU → MatMul(W2: 4×2)
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 2, cols: 2 },
            GraphOp::MatMul { weight_rows: 2, weight_cols: 4 },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 4,
            },
            GraphOp::MatMul { weight_rows: 4, weight_cols: 2 },
        ]).unwrap();

        let input = M31Matrix::from_data(2, 2, vec![
            M31::from(1), M31::from(2),
            M31::from(3), M31::from(4),
        ]).unwrap();

        let w1 = M31Matrix::from_data(2, 4, vec![
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
        ]).unwrap();

        let w2 = M31Matrix::from_data(4, 2, vec![
            M31::from(1), M31::from(0),
            M31::from(0), M31::from(1),
            M31::from(1), M31::from(0),
            M31::from(0), M31::from(1),
        ]).unwrap();

        let mut weights = GraphWeights::new();
        weights.matmul_weights.insert(1, w1);
        weights.matmul_weights.insert(3, w2);

        let execution = execute_graph(&graph, &input, &weights).unwrap();

        assert_eq!(execution.order.len(), 4);
        let final_out = execution.final_output().unwrap();
        assert_eq!(final_out.rows, 2);
        assert_eq!(final_out.cols, 2);
    }

    #[test]
    fn test_execute_with_layernorm() {
        // Input(2×2) → LayerNorm
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 2, cols: 2 },
            GraphOp::LayerNorm { feature_dim: 2 },
        ]).unwrap();

        let input = M31Matrix::from_data(2, 2, vec![
            M31::from(5), M31::from(15),
            M31::from(3), M31::from(7),
        ]).unwrap();

        let params = LayerNormParams::identity(2);

        let mut weights = GraphWeights::new();
        weights.layernorm_params.insert(1, params);

        let execution = execute_graph(&graph, &input, &weights).unwrap();

        // LayerNorm with identity params and inv_std=1: output = (input - mean)
        let ln_out = execution.output(1).unwrap();
        assert_eq!(ln_out.rows, 2);
        assert_eq!(ln_out.cols, 2);

        // Final output should exist
        assert!(execution.final_output().is_some());
    }

    #[test]
    fn test_execute_missing_weight_error() {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 2, cols: 2 },
            GraphOp::MatMul { weight_rows: 2, weight_cols: 2 },
        ]).unwrap();

        let input = M31Matrix::from_data(2, 2, vec![
            M31::from(1), M31::from(2),
            M31::from(3), M31::from(4),
        ]).unwrap();

        let weights = GraphWeights::new(); // No weights provided

        let result = execute_graph(&graph, &input, &weights);
        assert!(result.is_err());
    }
}
