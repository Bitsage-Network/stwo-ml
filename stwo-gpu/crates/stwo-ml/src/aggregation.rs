//! Proof aggregation for on-chain verification.
//!
//! Composes multiple activation layer STARK components into a single
//! STARK proof. Matmul sumcheck proofs remain separate (different proof system).
//!
//! # Architecture
//!
//! ```text
//! Input → [MatMul₀ (sumcheck)] → [ReLU₀ ─┐
//!       → [MatMul₁ (sumcheck)] → [ReLU₁ ─┤──→ Single STARK proof
//!       → [MatMul₂ (sumcheck)]            │    (all activations)
//!                                          └───────────────────┘
//! ```
//!
//! The single STARK proof covers all activation/layernorm constraints,
//! verified on-chain via a single Cairo verifier call.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::channel::{Blake2sChannel, MerkleChannel};
use stwo::core::proof::StarkProof;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::CommitmentSchemeProver;
use stwo::prover::prove;
use stwo::prover::ComponentProver;

use stwo_constraint_framework::{
    FrameworkComponent, TraceLocationAllocator,
    LogupTraceGenerator,
};

use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use crate::compiler::prove::{GraphExecution, ModelError};
use crate::components::activation::{
    ActivationEval, ActivationRelation,
    compute_multiplicities,
};
use crate::components::matmul::{
    M31Matrix, matmul_m31,
    MatMulSumcheckProof, prove_matmul_sumcheck,
};
use crate::gadgets::lookup_table::PrecomputedTable;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

/// A claim about a single layer's computation.
#[derive(Debug, Clone)]
pub struct LayerClaim {
    pub layer_index: usize,
    pub claimed_sum: SecureField,
    pub trace_rows: usize,
}

/// Aggregated model proof: single STARK (activations) + per-matmul sumchecks.
pub struct AggregatedModelProof {
    /// Single STARK proof covering all activation layers.
    /// `None` if the model has no activation layers.
    pub activation_stark: Option<StarkProof<Blake2sHash>>,
    /// Per-matmul sumcheck proofs, in layer order.
    pub matmul_proofs: Vec<(usize, MatMulSumcheckProof)>,
    /// Forward pass execution trace.
    pub execution: GraphExecution,
    /// Per-activation-layer claims (for verification).
    pub activation_claims: Vec<LayerClaim>,
}

impl AggregatedModelProof {
    /// Total number of proven layers (matmul + activation).
    pub fn num_proven_layers(&self) -> usize {
        self.matmul_proofs.len() + self.activation_claims.len()
    }

    /// Estimated calldata size in bytes for on-chain submission.
    pub fn estimated_calldata_bytes(&self) -> usize {
        // Commitments: 3 trees × 32 bytes each
        let commitment_size = 3 * 32;
        // FRI proof: ~1KB per component (rough estimate)
        let fri_size = 1024 * (self.activation_claims.len().max(1));
        // Sumcheck proofs: ~256 bytes each
        let sumcheck_size = self.matmul_proofs.len() * 256;
        commitment_size + fri_size + sumcheck_size
    }
}

/// Error type for aggregation.
#[derive(Debug, thiserror::Error)]
pub enum AggregationError {
    #[error("No components to aggregate")]
    EmptyComponents,
    #[error("Proving error: {0}")]
    ProvingError(String),
    #[error("Model error: {0}")]
    ModelError(#[from] ModelError),
}

/// Collected activation layer data for aggregation.
struct ActivationLayerData {
    node_id: usize,
    inputs: Vec<M31>,
    outputs: Vec<M31>,
    table: PrecomputedTable,
}

/// Prove an entire computation graph with aggregated STARK proof.
///
/// All activation layers are combined into a **single STARK proof**,
/// while each matmul gets its own sumcheck proof. This minimizes
/// on-chain verification cost.
pub fn prove_model_aggregated(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProof, AggregationError> {
    let config = PcsConfig::default();
    let mut intermediates: Vec<(usize, M31Matrix)> = Vec::new();
    let mut current = input.clone();

    // Collect matmul proofs and activation data during forward pass
    let mut matmul_proofs: Vec<(usize, MatMulSumcheckProof)> = Vec::new();
    let mut activation_layers: Vec<ActivationLayerData> = Vec::new();

    for node in &graph.nodes {
        match &node.op {
            GraphOp::MatMul { .. } => {
                let weight = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;

                let output = matmul_m31(&current, weight);

                let proof = prove_matmul_sumcheck(&current, weight, &output)
                    .map_err(|e| ModelError::ProvingError {
                        layer: node.id,
                        message: format!("MatMul sumcheck: {e}"),
                    })?;

                intermediates.push((node.id, current.clone()));
                matmul_proofs.push((node.id, proof));
                current = output;
            }

            GraphOp::Activation { activation_type, .. } => {
                let f = activation_type.as_fn();
                let output = crate::compiler::prove::apply_activation_pub(&current, &*f);

                let table = PrecomputedTable::build(|x| (*f)(x), 4);
                let flat_inputs = current.data.clone();
                let flat_outputs = output.data.clone();

                activation_layers.push(ActivationLayerData {
                    node_id: node.id,
                    inputs: flat_inputs,
                    outputs: flat_outputs,
                    table,
                });

                intermediates.push((node.id, current.clone()));
                current = output;
            }

            _ => {
                intermediates.push((node.id, current.clone()));
            }
        }
    }

    let execution = GraphExecution {
        intermediates,
        output: current,
    };

    // If no activations, return early with just sumcheck proofs
    if activation_layers.is_empty() {
        return Ok(AggregatedModelProof {
            activation_stark: None,
            matmul_proofs,
            execution,
            activation_claims: Vec::new(),
        });
    }

    // === Aggregate all activation layers into a single STARK ===
    let log_size = 4u32; // All activation tables use log_size=4 for now
    let size = 1usize << log_size;
    let vec_size = size >> LOG_N_LANES;
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Twiddles must cover the max constraint degree
    let max_degree_bound = log_size + 1;
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_degree_bound + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Tree 0: Preprocessed columns — all activation tables concatenated
    let mut tree_builder = commitment_scheme.tree_builder();
    for layer in &activation_layers {
        let (table_input_col, table_output_col) = build_table_columns(&layer.table, size);
        tree_builder.extend_evals(vec![
            CircleEvaluation::new(domain, table_input_col),
            CircleEvaluation::new(domain, table_output_col),
        ]);
    }
    tree_builder.commit(channel);

    // Tree 1: Execution traces — all trace columns concatenated
    let mut tree_builder = commitment_scheme.tree_builder();
    let mut layer_mults: Vec<Vec<M31>> = Vec::new();
    for layer in &activation_layers {
        let pad_input = layer.table.inputs[0];
        let pad_output = layer.table.outputs[0];
        let padding_count = size.saturating_sub(layer.inputs.len());

        let mut mults = compute_multiplicities(&layer.inputs, &layer.table);
        if padding_count > 0 {
            mults[0] += M31::from(padding_count as u32);
        }

        let (trace_in, trace_out, mult_col) = build_trace_columns(
            &layer.inputs, &layer.outputs, &mults,
            pad_input, pad_output, size,
        );
        tree_builder.extend_evals(vec![
            CircleEvaluation::new(domain, trace_in),
            CircleEvaluation::new(domain, trace_out),
            CircleEvaluation::new(domain, mult_col),
        ]);
        layer_mults.push(mults);
    }
    tree_builder.commit(channel);

    // Draw lookup elements (shared across all activation layers)
    let lookup_elements = ActivationRelation::draw(channel);

    // Tree 2: Interaction traces — all LogUp columns concatenated
    let mut tree_builder = commitment_scheme.tree_builder();
    let mut claimed_sums: Vec<SecureField> = Vec::new();

    for (idx, layer) in activation_layers.iter().enumerate() {
        let pad_input = layer.table.inputs[0];
        let pad_output = layer.table.outputs[0];

        let (table_in_col, table_out_col) = build_table_columns(&layer.table, size);
        let (trace_in_col, trace_out_col, _) = build_trace_columns(
            &layer.inputs, &layer.outputs, &layer_mults[idx],
            pad_input, pad_output, size,
        );

        let mut logup_gen = LogupTraceGenerator::new(log_size);
        let mut col_gen = logup_gen.new_col();

        for vec_row in 0..vec_size {
            let q_table: PackedSecureField = lookup_elements.lookup_elements().combine(
                &[table_in_col.data[vec_row], table_out_col.data[vec_row]],
            );
            let q_trace: PackedSecureField = lookup_elements.lookup_elements().combine(
                &[trace_in_col.data[vec_row], trace_out_col.data[vec_row]],
            );

            let mult_packed = pack_multiplicities(&layer_mults[idx], vec_row);
            let numerator = q_table - mult_packed * q_trace;
            let denominator = q_table * q_trace;

            col_gen.write_frac(vec_row, numerator, denominator);
        }
        col_gen.finalize_col();

        let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
        tree_builder.extend_evals(interaction_trace);
        claimed_sums.push(claimed_sum);
    }
    tree_builder.commit(channel);

    // Build all activation components with shared allocator
    let mut allocator = TraceLocationAllocator::default();
    let mut components: Vec<FrameworkComponent<ActivationEval>> = Vec::new();
    let mut activation_claims: Vec<LayerClaim> = Vec::new();

    for (idx, layer) in activation_layers.iter().enumerate() {
        let claimed_sum = claimed_sums[idx];
        let component = FrameworkComponent::new(
            &mut allocator,
            ActivationEval {
                log_n_rows: log_size,
                lookup_elements: lookup_elements.clone(),
                claimed_sum,
                total_sum: claimed_sum,
            },
            claimed_sum,
        );
        components.push(component);
        activation_claims.push(LayerClaim {
            layer_index: layer.node_id,
            claimed_sum,
            trace_rows: size,
        });
    }

    // Single prove() call with all activation components
    let component_refs: Vec<&dyn ComponentProver<SimdBackend>> =
        components.iter().map(|c| c as &dyn ComponentProver<SimdBackend>).collect();

    let stark_proof = prove::<SimdBackend, Blake2sMerkleChannel>(
        &component_refs,
        channel,
        commitment_scheme,
    ).map_err(|e| AggregationError::ProvingError(format!("{e:?}")))?;

    Ok(AggregatedModelProof {
        activation_stark: Some(stark_proof),
        matmul_proofs,
        execution,
        activation_claims,
    })
}

// --- Helper functions ---

fn build_table_columns(
    table: &PrecomputedTable,
    size: usize,
) -> (Col<SimdBackend, BaseField>, Col<SimdBackend, BaseField>) {
    let mut input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut output_col = Col::<SimdBackend, BaseField>::zeros(size);
    for (i, (&inp, &out)) in table.inputs.iter().zip(table.outputs.iter()).enumerate().take(size) {
        input_col.set(i, inp);
        output_col.set(i, out);
    }
    (input_col, output_col)
}

fn build_trace_columns(
    inputs: &[M31],
    outputs: &[M31],
    multiplicities: &[M31],
    pad_input: M31,
    pad_output: M31,
    size: usize,
) -> (
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
) {
    let mut trace_in = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_out = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);

    for (i, (&inp, &out)) in inputs.iter().zip(outputs.iter()).enumerate().take(size) {
        trace_in.set(i, inp);
        trace_out.set(i, out);
    }
    for i in inputs.len()..size {
        trace_in.set(i, pad_input);
        trace_out.set(i, pad_output);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    (trace_in, trace_out, mult_col)
}

fn pack_multiplicities(
    multiplicities: &[M31],
    vec_row: usize,
) -> PackedSecureField {
    use stwo::prover::backend::simd::m31::PackedBaseField;

    let n_lanes = 16usize;
    let base = vec_row * n_lanes;
    let mut vals = [M31::from(0); 16];
    for (i, val) in vals.iter_mut().enumerate() {
        let idx = base + i;
        if idx < multiplicities.len() {
            *val = multiplicities[idx];
        }
    }
    let packed_base = PackedBaseField::from_array(std::array::from_fn(|i| vals[i]));
    packed_base.into()
}

/// Aggregate multiple claims into a summary (layer count, total rows).
pub fn summarize_claims(claims: &[LayerClaim]) -> (usize, usize) {
    let total_rows: usize = claims.iter().map(|c| c.trace_rows).sum();
    (claims.len(), total_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use crate::components::activation::ActivationType;

    #[test]
    fn test_aggregated_matmul_only() {
        // Model with no activations — should return None for activation_stark
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated proving should succeed");

        assert!(proof.activation_stark.is_none());
        assert_eq!(proof.matmul_proofs.len(), 1);
        assert_eq!(proof.num_proven_layers(), 1);
    }

    #[test]
    fn test_aggregated_mlp_with_activations() {
        // 5-layer MLP: matmul → relu → matmul → relu → matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();

        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32)); } }
        weights.add_weight(0, w0);

        let mut w2 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w2.set(i, j, M31::from(((i * j) % 5 + 1) as u32)); } }
        weights.add_weight(2, w2);

        let mut w4 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w4.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(4, w4);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated MLP proving should succeed");

        // 1 STARK proof covering both ReLU layers
        assert!(proof.activation_stark.is_some(), "should have aggregated activation STARK");
        // 3 sumcheck proofs (one per matmul)
        assert_eq!(proof.matmul_proofs.len(), 3, "3 matmul sumcheck proofs");
        // 2 activation claims
        assert_eq!(proof.activation_claims.len(), 2, "2 activation layers in STARK");
        // Total: 5 proven layers
        assert_eq!(proof.num_proven_layers(), 5);
        // Output shape
        assert_eq!(proof.execution.output.rows, 1);
        assert_eq!(proof.execution.output.cols, 2);
    }

    #[test]
    fn test_aggregated_calldata_estimate() {
        let proof = AggregatedModelProof {
            activation_stark: None,
            matmul_proofs: Vec::new(),
            execution: GraphExecution {
                intermediates: Vec::new(),
                output: M31Matrix::new(1, 1),
            },
            activation_claims: vec![
                LayerClaim { layer_index: 0, claimed_sum: SecureField::from(M31::from(0)), trace_rows: 16 },
                LayerClaim { layer_index: 1, claimed_sum: SecureField::from(M31::from(0)), trace_rows: 16 },
            ],
        };
        let calldata = proof.estimated_calldata_bytes();
        assert!(calldata > 0);
    }

    #[test]
    fn test_summarize_claims() {
        let claims = vec![
            LayerClaim { layer_index: 0, claimed_sum: SecureField::from(M31::from(0)), trace_rows: 1000 },
            LayerClaim { layer_index: 1, claimed_sum: SecureField::from(M31::from(0)), trace_rows: 2000 },
        ];
        let (num, total) = summarize_claims(&claims);
        assert_eq!(num, 2);
        assert_eq!(total, 3000);
    }
}
