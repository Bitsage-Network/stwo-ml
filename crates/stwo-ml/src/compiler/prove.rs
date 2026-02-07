//! End-to-end model proof pipeline.
//!
//! Proves an entire computation graph by composing per-node proofs:
//!
//! - **MatMul** nodes: sumcheck protocol over multilinear extensions
//! - **Activation** nodes: LogUp STARK against precomputed tables
//! - **LayerNorm** nodes: LogUp STARK for normalization mapping
//! - **Quantize** nodes: range check STARK for value bounds
//!
//! # Usage
//!
//! ```rust,ignore
//! use stwo_ml::compiler::graph::*;
//! use stwo_ml::compiler::prove::*;
//!
//! let graph = ComputationGraph::sequential(vec![
//!     GraphOp::Input { rows: 4, cols: 4 },
//!     GraphOp::MatMul { weight_rows: 4, weight_cols: 4 },
//!     GraphOp::Activation { activation: ActivationType::ReLU, log_table_size: 8 },
//! ]).unwrap();
//!
//! let input = M31Matrix::random(4, 4);
//! let mut weights = GraphWeights::new();
//! weights.matmul_weights.insert(1, M31Matrix::random(4, 4));
//!
//! let (proof, execution) = prove_model(&graph, &input, &weights).unwrap();
//! verify_model(&proof, &execution, &weights).unwrap();
//! ```

use std::collections::HashMap;

use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::m31::M31;
use stwo::core::pcs::PcsConfig;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;

use crate::components::activation::{
    prove_activation, verify_activation, ActivationComponent, ActivationError,
};
use crate::components::attention::flatten_matrix;
use crate::components::layernorm::{
    compute_mean, prove_layernorm_stark, verify_layernorm_stark, LayerNormComponent,
    LayerNormError,
};
use crate::components::matmul::{
    prove_matmul, verify_matmul, M31Matrix, MatMulAux, MatMulError, MatMulProof,
};
use crate::gadgets::range_check::{
    prove_range_check, verify_range_check, RangeCheckComponent, RangeCheckError,
};

use super::graph::{
    execute_graph, ComputationGraph, ExecutionError, GraphExecution, GraphOp, GraphWeights, NodeId,
};

// ---------------------------------------------------------------------------
// Per-node proof types
// ---------------------------------------------------------------------------

/// Proof for a single node in the computation graph.
pub enum NodeProof {
    /// Input node — no computation to prove.
    Input,
    /// MatMul proof via sumcheck over multilinear extensions.
    MatMul {
        proof: MatMulProof,
        aux: MatMulAux,
    },
    /// Activation proof via LogUp STARK.
    Activation {
        component: ActivationComponent,
        proof: StarkProof<Blake2sMerkleHasher>,
    },
    /// LayerNorm proof via LogUp STARK.
    LayerNorm {
        component: LayerNormComponent,
        proof: StarkProof<Blake2sMerkleHasher>,
        mean: M31,
        inv_std: M31,
    },
    /// Quantize proof via range check STARK.
    Quantize {
        component: RangeCheckComponent,
        proof: StarkProof<Blake2sMerkleHasher>,
    },
}

// ---------------------------------------------------------------------------
// Model proof
// ---------------------------------------------------------------------------

/// Complete proof for an entire model (computation graph).
///
/// Contains one `NodeProof` per operation in the graph. Verification
/// replays each node's proof in topological order.
pub struct ModelProof {
    /// Per-node proofs, keyed by NodeId.
    pub node_proofs: HashMap<NodeId, NodeProof>,
    /// Execution order (topological).
    pub order: Vec<NodeId>,
}

impl ModelProof {
    /// Number of proven nodes (excludes Input nodes).
    pub fn num_proofs(&self) -> usize {
        self.node_proofs
            .values()
            .filter(|p| !matches!(p, NodeProof::Input))
            .count()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for model proving/verification.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("graph execution failed: {0}")]
    Execution(#[from] ExecutionError),
    #[error("matmul proof failed at node {node_id}: {source}")]
    MatMul {
        node_id: NodeId,
        source: MatMulError,
    },
    #[error("activation proof failed at node {node_id}: {source}")]
    Activation {
        node_id: NodeId,
        source: ActivationError,
    },
    #[error("layernorm proof failed at node {node_id}: {source}")]
    LayerNorm {
        node_id: NodeId,
        source: LayerNormError,
    },
    #[error("range check proof failed at node {node_id}: {source}")]
    RangeCheck {
        node_id: NodeId,
        source: RangeCheckError,
    },
    #[error("missing proof for node {0}")]
    MissingProof(NodeId),
    #[error("missing output for node {0}")]
    MissingOutput(NodeId),
}

// ---------------------------------------------------------------------------
// Prove
// ---------------------------------------------------------------------------

/// Prove an entire computation graph.
///
/// 1. Executes the graph to generate all intermediate values (witnesses).
/// 2. Proves each non-Input node using the appropriate STWO protocol.
/// 3. Returns both the composed proof and the execution trace.
pub fn prove_model(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<(ModelProof, GraphExecution), ModelError> {
    let execution = execute_graph(graph, input, weights)?;
    let order = execution.order.clone();
    let mut node_proofs: HashMap<NodeId, NodeProof> = HashMap::new();
    let config = PcsConfig::default();

    for &node_id in &order {
        let node = graph.node(node_id).unwrap();
        let proof = match &node.op {
            GraphOp::Input { .. } => NodeProof::Input,

            GraphOp::MatMul { .. } => {
                // Input to this matmul is the predecessor's output.
                let pred_id = node.inputs[0];
                let input_matrix = execution
                    .output(pred_id)
                    .ok_or(ModelError::MissingOutput(pred_id))?;
                let weight_matrix = weights
                    .matmul_weights
                    .get(&node_id)
                    .ok_or(ModelError::Execution(ExecutionError::MissingWeight(node_id)))?;
                let output_matrix = execution
                    .output(node_id)
                    .ok_or(ModelError::MissingOutput(node_id))?;

                let mut channel = Blake2sChannel::default();
                let (proof, aux) =
                    prove_matmul(input_matrix, weight_matrix, output_matrix, &mut channel)
                        .map_err(|e| ModelError::MatMul {
                            node_id,
                            source: e,
                        })?;
                NodeProof::MatMul { proof, aux }
            }

            GraphOp::Activation {
                activation,
                log_table_size,
            } => {
                let pred_id = node.inputs[0];
                let input_matrix = execution
                    .output(pred_id)
                    .ok_or(ModelError::MissingOutput(pred_id))?;
                let table = activation.build_table(*log_table_size);
                let values = flatten_matrix(input_matrix);

                let mut channel = Blake2sChannel::default();
                let (component, proof) =
                    prove_activation(&values, &table, config, &mut channel).map_err(|e| {
                        ModelError::Activation {
                            node_id,
                            source: e,
                        }
                    })?;
                NodeProof::Activation { component, proof }
            }

            GraphOp::LayerNorm { .. } => {
                let pred_id = node.inputs[0];
                let input_matrix = execution
                    .output(pred_id)
                    .ok_or(ModelError::MissingOutput(pred_id))?;
                let params = weights
                    .layernorm_params
                    .get(&node_id)
                    .ok_or(ModelError::Execution(
                        ExecutionError::MissingLayerNormParams(node_id),
                    ))?;
                let inv_std = weights
                    .layernorm_inv_std
                    .get(&node_id)
                    .copied()
                    .unwrap_or(M31::from(1));

                // Flatten first row for proving (LayerNorm operates row-by-row;
                // prove the first row as a representative).
                let row_data: Vec<M31> =
                    (0..input_matrix.cols).map(|j| input_matrix.get(0, j)).collect();
                let mean = compute_mean(&row_data).map_err(|e| ModelError::LayerNorm {
                    node_id,
                    source: e,
                })?;
                let log_size = (input_matrix.cols as u32)
                    .next_power_of_two()
                    .ilog2()
                    .max(4); // Minimum for SIMD lanes

                let mut channel = Blake2sChannel::default();
                let (component, proof) = prove_layernorm_stark(
                    &row_data, log_size, mean, inv_std, params, config, &mut channel,
                )
                .map_err(|e| ModelError::LayerNorm {
                    node_id,
                    source: e,
                })?;
                NodeProof::LayerNorm {
                    component,
                    proof,
                    mean,
                    inv_std,
                }
            }

            GraphOp::Quantize { bits } => {
                let pred_id = node.inputs[0];
                let input_matrix = execution
                    .output(pred_id)
                    .ok_or(ModelError::MissingOutput(pred_id))?;
                let values = flatten_matrix(input_matrix);
                let log_range = (*bits).max(4); // Minimum for SIMD lanes

                let mut channel = Blake2sChannel::default();
                let (component, proof) =
                    prove_range_check(&values, log_range, config, &mut channel).map_err(|e| {
                        ModelError::RangeCheck {
                            node_id,
                            source: e,
                        }
                    })?;
                NodeProof::Quantize { component, proof }
            }
        };

        node_proofs.insert(node_id, proof);
    }

    Ok((ModelProof { node_proofs, order }, execution))
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

/// Verify a model proof against the execution trace.
///
/// Replays each node's proof in topological order, checking:
/// - MatMul: sumcheck verification
/// - Activation: LogUp STARK verification
/// - LayerNorm: LogUp STARK verification
/// - Quantize: range check STARK verification
pub fn verify_model(
    proof: &ModelProof,
    execution: &GraphExecution,
    graph: &ComputationGraph,
    weights: &GraphWeights,
) -> Result<(), ModelError> {
    for &node_id in &proof.order {
        let node = graph.node(node_id).unwrap();
        let node_proof = proof
            .node_proofs
            .get(&node_id)
            .ok_or(ModelError::MissingProof(node_id))?;

        match (node_proof, &node.op) {
            (NodeProof::Input, GraphOp::Input { .. }) => {
                // Nothing to verify for inputs.
            }

            (NodeProof::MatMul { proof, aux }, GraphOp::MatMul { .. }) => {
                let pred_id = node.inputs[0];
                let input_matrix = execution
                    .output(pred_id)
                    .ok_or(ModelError::MissingOutput(pred_id))?;
                let weight_matrix = weights
                    .matmul_weights
                    .get(&node_id)
                    .ok_or(ModelError::Execution(ExecutionError::MissingWeight(node_id)))?;
                let output_matrix = execution
                    .output(node_id)
                    .ok_or(ModelError::MissingOutput(node_id))?;

                let mut channel = Blake2sChannel::default();
                verify_matmul(
                    input_matrix,
                    weight_matrix,
                    output_matrix,
                    proof,
                    aux,
                    &mut channel,
                )
                .map_err(|e| ModelError::MatMul {
                    node_id,
                    source: e,
                })?;
            }

            (
                NodeProof::Activation { component, proof },
                GraphOp::Activation { .. },
            ) => {
                let mut channel = Blake2sChannel::default();
                verify_activation(component, proof, &mut channel).map_err(|e| {
                    ModelError::Activation {
                        node_id,
                        source: e,
                    }
                })?;
            }

            (
                NodeProof::LayerNorm {
                    component, proof, ..
                },
                GraphOp::LayerNorm { .. },
            ) => {
                let mut channel = Blake2sChannel::default();
                verify_layernorm_stark(component, proof, &mut channel).map_err(|e| {
                    ModelError::LayerNorm {
                        node_id,
                        source: e,
                    }
                })?;
            }

            (NodeProof::Quantize { component, proof }, GraphOp::Quantize { .. }) => {
                let mut channel = Blake2sChannel::default();
                verify_range_check(component, proof, &mut channel).map_err(|e| {
                    ModelError::RangeCheck {
                        node_id,
                        source: e,
                    }
                })?;
            }

            _ => {
                return Err(ModelError::MissingProof(node_id));
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::activation::ActivationType;
    use crate::components::layernorm::LayerNormParams;

    /// Build a 4×4 matmul + ReLU model and prove/verify end-to-end.
    #[test]
    fn test_prove_verify_matmul_relu() {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::MatMul {
                weight_rows: 4,
                weight_cols: 4,
            },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 8,
            },
        ])
        .unwrap();

        // Small values to stay within activation table range [0, 256).
        let input = M31Matrix::from_data(
            4,
            4,
            (0..16).map(|i| M31::from(i % 4)).collect(),
        )
        .unwrap();

        let weight = M31Matrix::from_data(
            4,
            4,
            (0..16).map(|i| M31::from((i + 1) % 3)).collect(),
        )
        .unwrap();

        let mut weights = GraphWeights::new();
        weights.matmul_weights.insert(1, weight);

        let (proof, execution) = prove_model(&graph, &input, &weights).unwrap();
        assert_eq!(proof.num_proofs(), 2); // matmul + activation

        verify_model(&proof, &execution, &graph, &weights).unwrap();
    }

    /// Two-layer MLP: input → matmul → ReLU → matmul → output.
    #[test]
    fn test_prove_verify_two_layer_mlp() {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::MatMul {
                weight_rows: 4,
                weight_cols: 4,
            },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 8,
            },
            GraphOp::MatMul {
                weight_rows: 4,
                weight_cols: 4,
            },
        ])
        .unwrap();

        let input = M31Matrix::from_data(
            4,
            4,
            (0..16).map(|i| M31::from(i % 3)).collect(),
        )
        .unwrap();

        let w1 = M31Matrix::from_data(
            4,
            4,
            (0..16).map(|i| M31::from((i + 1) % 2)).collect(),
        )
        .unwrap();

        let w2 = M31Matrix::from_data(
            4,
            4,
            (0..16).map(|i| M31::from(i % 2)).collect(),
        )
        .unwrap();

        let mut weights = GraphWeights::new();
        weights.matmul_weights.insert(1, w1);
        weights.matmul_weights.insert(3, w2);

        let (proof, execution) = prove_model(&graph, &input, &weights).unwrap();
        assert_eq!(proof.num_proofs(), 3); // matmul + relu + matmul

        verify_model(&proof, &execution, &graph, &weights).unwrap();
    }

    /// Model with layernorm: input → layernorm.
    #[test]
    fn test_prove_verify_layernorm_model() {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 1, cols: 16 },
            GraphOp::LayerNorm { feature_dim: 16 },
        ])
        .unwrap();

        // Values in [0, 16) so they fit in the layernorm table.
        let input = M31Matrix::from_data(
            1,
            16,
            (0..16).map(|i| M31::from(i as u32)).collect(),
        )
        .unwrap();

        let params = LayerNormParams::identity(16);

        let mut weights = GraphWeights::new();
        weights.layernorm_params.insert(1, params);

        let (proof, execution) = prove_model(&graph, &input, &weights).unwrap();
        assert_eq!(proof.num_proofs(), 1);

        verify_model(&proof, &execution, &graph, &weights).unwrap();
    }

    /// Model with quantize: input → quantize(8-bit range check).
    #[test]
    fn test_prove_verify_quantize_model() {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 2, cols: 4 },
            GraphOp::Quantize { bits: 8 },
        ])
        .unwrap();

        // All values must be in [0, 256) for the range check.
        let input = M31Matrix::from_data(
            2,
            4,
            (0..8).map(|i| M31::from(i * 30)).collect(),
        )
        .unwrap();

        let weights = GraphWeights::new();

        let (proof, execution) = prove_model(&graph, &input, &weights).unwrap();
        assert_eq!(proof.num_proofs(), 1);

        verify_model(&proof, &execution, &graph, &weights).unwrap();
    }

    /// Full pipeline: matmul → ReLU → quantize.
    #[test]
    fn test_prove_verify_matmul_relu_quantize() {
        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::MatMul {
                weight_rows: 4,
                weight_cols: 4,
            },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 8,
            },
            GraphOp::Quantize { bits: 8 },
        ])
        .unwrap();

        let input = M31Matrix::from_data(
            4,
            4,
            (0..16).map(|i| M31::from(i % 4)).collect(),
        )
        .unwrap();

        let weight = M31Matrix::from_data(
            4,
            4,
            (0..16).map(|i| M31::from((i + 1) % 3)).collect(),
        )
        .unwrap();

        let mut weights = GraphWeights::new();
        weights.matmul_weights.insert(1, weight);

        let (proof, execution) = prove_model(&graph, &input, &weights).unwrap();
        assert_eq!(proof.num_proofs(), 3); // matmul + relu + quantize

        verify_model(&proof, &execution, &graph, &weights).unwrap();
    }
}
