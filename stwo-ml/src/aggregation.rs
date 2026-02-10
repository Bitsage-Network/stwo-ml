//! Proof aggregation for on-chain verification.
//!
//! Composes all non-matmul STARK components (activations, Add, Mul, LayerNorm)
//! into a **single unified STARK proof**. Matmul sumcheck proofs remain separate
//! (different proof system).
//!
//! # Architecture
//!
//! ```text
//! Input → [MatMul₀ (sumcheck)] → [ReLU₀ ─┐
//!       → [MatMul₁ (sumcheck)] → [ReLU₁ ─┤
//!       → [Add (residual)]     ───────────┤──→ Single STARK proof
//!       → [Mul (gating)]       ───────────┤    (all non-matmul components)
//!       → [LayerNorm]          ───────────┘
//! ```
//!
//! The single STARK proof covers all activation/add/mul/layernorm constraints,
//! verified on-chain via a single Cairo verifier call.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::channel::MerkleChannel;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::{Col, Column, BackendForChannel};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::CommitmentSchemeProver;
use stwo::prover::prove;
use stwo::prover::ComponentProver;

use stwo_constraint_framework::{
    FrameworkComponent, TraceLocationAllocator,
    LogupTraceGenerator,
};

use std::collections::HashMap;

use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use crate::compiler::prove::{
    GraphExecution, ModelError, elementwise_add, elementwise_mul,
    apply_layernorm_detailed,
};
use crate::components::elementwise::{ElementwiseAddEval, ElementwiseMulEval};
use crate::components::layernorm::{LayerNormConfig, LayerNormEval, LayerNormRelation, build_rsqrt_table};
use crate::components::activation::{
    ActivationEval, ActivationRelation,
    compute_multiplicities,
};
use crate::components::matmul::{
    M31Matrix, matmul_m31,
    MatMulSumcheckProof, prove_matmul_sumcheck,
    MatMulSumcheckProofOnChain, prove_matmul_sumcheck_onchain,
    estimate_sumcheck_memory,
};
use crate::components::tiled_matmul::{
    TiledMatMulConfig, prove_tiled_matmul, compose_tiled_proof,
};
use crate::gadgets::lookup_table::PrecomputedTable;
use crate::backend::convert_evaluations;
use tracing::info;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

/// Compute the minimum log_size for a data vector: next power of two, at least SIMD width.
fn data_log_size(data_len: usize) -> u32 {
    let min_size = data_len.next_power_of_two().max(1 << LOG_N_LANES);
    min_size.ilog2()
}

/// A claim about a single layer's computation.
#[derive(Debug, Clone)]
pub struct LayerClaim {
    pub layer_index: usize,
    pub claimed_sum: SecureField,
    pub trace_rows: usize,
}

/// Aggregated model proof: single unified STARK (all non-matmul components) + per-matmul sumchecks.
/// Generic over the Merkle hash type `H`.
pub struct AggregatedModelProofFor<H: MerkleHasherLifted> {
    /// Single STARK proof covering all non-matmul components
    /// (activations, Add, Mul, LayerNorm).
    /// `None` if the model has no non-matmul layers.
    pub unified_stark: Option<StarkProof<H>>,
    /// Per-matmul sumcheck proofs, in layer order.
    pub matmul_proofs: Vec<(usize, MatMulSumcheckProof)>,
    /// Per-Add layer claims (verified inside unified STARK).
    pub add_claims: Vec<LayerClaim>,
    /// Per-Mul layer claims (verified inside unified STARK).
    pub mul_claims: Vec<LayerClaim>,
    /// Per-LayerNorm layer claims (verified inside unified STARK).
    pub layernorm_claims: Vec<LayerClaim>,
    /// Forward pass execution trace.
    pub execution: GraphExecution,
    /// Per-activation-layer claims (for verification).
    pub activation_claims: Vec<LayerClaim>,
}

/// Aggregated model proof using Blake2s (default).
pub type AggregatedModelProof = AggregatedModelProofFor<Blake2sHash>;

impl<H: MerkleHasherLifted> AggregatedModelProofFor<H> {
    /// Total number of proven layers (matmul + activation + add + mul + layernorm).
    pub fn num_proven_layers(&self) -> usize {
        self.matmul_proofs.len() + self.activation_claims.len()
            + self.add_claims.len() + self.mul_claims.len() + self.layernorm_claims.len()
    }

    /// Estimated calldata size in bytes for on-chain submission.
    pub fn estimated_calldata_bytes(&self) -> usize {
        // Commitments: 3 trees × 32 bytes each
        let commitment_size = 3 * 32;
        // FRI proof: ~1KB per component (rough estimate)
        let num_components = self.activation_claims.len()
            + self.add_claims.len() + self.mul_claims.len() + self.layernorm_claims.len();
        let fri_size = 1024 * num_components.max(1);
        // Sumcheck proofs: ~256 bytes each
        let sumcheck_size = self.matmul_proofs.len() * 256;
        // Claims are lightweight (no separate STARK proofs)
        let claim_size = (self.add_claims.len() + self.mul_claims.len()
            + self.layernorm_claims.len()) * 32;
        commitment_size + fri_size + sumcheck_size + claim_size
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
    log_size: u32,
}

/// Collected Add layer data for unified STARK aggregation.
struct AddLayerData {
    node_id: usize,
    lhs: Vec<M31>,
    rhs: Vec<M31>,
    output: Vec<M31>,
    log_size: u32,
}

/// Collected Mul layer data for unified STARK aggregation.
struct MulLayerData {
    node_id: usize,
    lhs: Vec<M31>,
    rhs: Vec<M31>,
    output: Vec<M31>,
    log_size: u32,
}

/// Collected LayerNorm layer data for unified STARK aggregation.
struct LayerNormLayerData {
    node_id: usize,
    inputs: Vec<M31>,
    means: Vec<M31>,
    variances: Vec<M31>,
    rsqrt_vals: Vec<M31>,
    outputs: Vec<M31>,
    rsqrt_table: PrecomputedTable,
    log_size: u32,
}

/// Prove an entire computation graph with aggregated STARK proof,
/// generic over backend and Merkle channel.
///
/// All non-matmul layers are combined into a **single unified STARK proof**,
/// while each matmul gets its own sumcheck proof. Trace generation uses
/// `SimdBackend`; commitment and proving use backend `B`, enabling GPU
/// acceleration of Merkle hashing, FRI, and quotient evaluation.
pub fn prove_model_aggregated_with<B, MC>(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofFor<<MC as MerkleChannel>::H>, AggregationError>
where
    B: BackendForChannel<MC> + PolyOps,
    MC: MerkleChannel,
    FrameworkComponent<ActivationEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseAddEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseMulEval>: ComponentProver<B>,
    FrameworkComponent<LayerNormEval>: ComponentProver<B>,
{
    info!(
        backend = std::any::type_name::<B>(),
        channel = std::any::type_name::<MC>(),
        "Proving unified STARK (off-chain aggregation)"
    );
    let config = PcsConfig::default();
    let mut intermediates: Vec<(usize, M31Matrix)> = Vec::new();
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();
    let mut current = input.clone();

    // Collect matmul proofs and layer data during forward pass
    let mut matmul_proofs: Vec<(usize, MatMulSumcheckProof)> = Vec::new();
    let mut activation_layers: Vec<ActivationLayerData> = Vec::new();
    let mut add_layers: Vec<AddLayerData> = Vec::new();
    let mut mul_layers: Vec<MulLayerData> = Vec::new();
    let mut layernorm_layers: Vec<LayerNormLayerData> = Vec::new();

    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];

        // Resolve current input from first dependency
        if let Some(&first_input) = node.inputs.first() {
            if let Some(inp) = node_outputs.get(&first_input) {
                current = inp.clone();
            }
        }

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
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Activation { activation_type, .. } => {
                let f = activation_type.as_fn();
                let output = crate::compiler::prove::apply_activation_pub(&current, &*f);

                let act_log_size = activation_type.recommended_table_log_size();
                let table = PrecomputedTable::build(|x| (*f)(x), act_log_size);
                let flat_inputs = current.data.clone();
                let flat_outputs = output.data.clone();

                activation_layers.push(ActivationLayerData {
                    node_id: node.id,
                    inputs: flat_inputs,
                    outputs: flat_outputs,
                    table,
                    log_size: act_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Add { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_add(&lhs, &rhs);

                let add_log_size = data_log_size(output.data.len());
                add_layers.push(AddLayerData {
                    node_id: node.id,
                    lhs: lhs.data.clone(),
                    rhs: rhs.data.clone(),
                    output: output.data.clone(),
                    log_size: add_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Mul { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_mul(&lhs, &rhs);

                let mul_log_size = data_log_size(output.data.len());
                mul_layers.push(MulLayerData {
                    node_id: node.id,
                    lhs: lhs.data.clone(),
                    rhs: rhs.data.clone(),
                    output: output.data.clone(),
                    log_size: mul_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::LayerNorm { dim } => {
                let ln_log_size = LayerNormConfig::new(*dim).rsqrt_table_log_size;
                let ln = apply_layernorm_detailed(&current, *dim);
                let rsqrt_table = build_rsqrt_table(ln_log_size);

                layernorm_layers.push(LayerNormLayerData {
                    node_id: node.id,
                    inputs: ln.inputs.clone(),
                    means: ln.means.clone(),
                    variances: ln.variances.clone(),
                    rsqrt_vals: ln.rsqrt_vals.clone(),
                    outputs: ln.outputs.clone(),
                    rsqrt_table,
                    log_size: ln_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, ln.output_matrix.clone());
                current = ln.output_matrix;
            }

            _ => {
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, current.clone());
            }
        }
    }

    let execution = GraphExecution {
        intermediates,
        output: current,
    };

    // Check if there are any non-matmul components to aggregate
    let has_components = !activation_layers.is_empty()
        || !add_layers.is_empty() || !mul_layers.is_empty() || !layernorm_layers.is_empty();

    if !has_components {
        return Ok(AggregatedModelProofFor {
            unified_stark: None,
            matmul_proofs,
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            execution,
            activation_claims: Vec::new(),
        });
    }

    // === Build unified STARK for all non-matmul components ===
    // Per-component log_sizes: each component uses its own size derived from
    // its table or data length. The max_log_size drives twiddle precomputation.
    let all_log_sizes: Vec<u32> = activation_layers.iter().map(|l| l.log_size)
        .chain(add_layers.iter().map(|l| l.log_size))
        .chain(mul_layers.iter().map(|l| l.log_size))
        .chain(layernorm_layers.iter().map(|l| l.log_size))
        .collect();
    let max_log_size = *all_log_sizes.iter().max().unwrap();

    let max_degree_bound = max_log_size + 1;
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(max_degree_bound + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::new(config, &twiddles);

    let has_logup = !activation_layers.is_empty() || !layernorm_layers.is_empty();

    // Tree 0: Preprocessed columns (always committed, may be empty)
    // - Activation tables: 2 cols per layer (table_input, table_output)
    // - LayerNorm rsqrt tables: 2 cols per layer (table_var, table_rsqrt)
    // - Add/Mul: 0 preprocessed cols
    {
        let mut tree_builder = commitment_scheme.tree_builder();
        for layer in &activation_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let (table_input_col, table_output_col) = build_table_columns(&layer.table, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, table_input_col),
                CircleEvaluation::new(layer_domain, table_output_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        for layer in &layernorm_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let (table_var_col, table_rsqrt_col) = build_table_columns(&layer.rsqrt_table, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, table_var_col),
                CircleEvaluation::new(layer_domain, table_rsqrt_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        tree_builder.commit(channel);
    }

    // Tree 1: Execution traces — concatenated in order:
    // 1. Activation traces: 3 cols per layer (trace_in, trace_out, multiplicity)
    // 2. Add traces: 3 cols per layer (lhs, rhs, output)
    // 3. Mul traces: 3 cols per layer (lhs, rhs, output)
    // 4. LayerNorm traces: 6 cols per layer (input, mean, var, rsqrt, output, multiplicity)
    let mut tree_builder = commitment_scheme.tree_builder();
    let mut activation_mults: Vec<Vec<M31>> = Vec::new();
    for layer in &activation_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let pad_input = layer.table.inputs[0];
        let pad_output = layer.table.outputs[0];
        let padding_count = layer_size.saturating_sub(layer.inputs.len());

        let mut mults = compute_multiplicities(&layer.inputs, &layer.table);
        if padding_count > 0 {
            mults[0] += M31::from(padding_count as u32);
        }

        let (trace_in, trace_out, mult_col) = build_trace_columns(
            &layer.inputs, &layer.outputs, &mults,
            pad_input, pad_output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, trace_in),
            CircleEvaluation::new(layer_domain, trace_out),
            CircleEvaluation::new(layer_domain, mult_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        activation_mults.push(mults);
    }
    for layer in &add_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let (lhs_col, rhs_col, out_col) = build_elementwise_trace_columns(
            &layer.lhs, &layer.rhs, &layer.output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, lhs_col),
            CircleEvaluation::new(layer_domain, rhs_col),
            CircleEvaluation::new(layer_domain, out_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    for layer in &mul_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let (lhs_col, rhs_col, out_col) = build_elementwise_trace_columns(
            &layer.lhs, &layer.rhs, &layer.output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, lhs_col),
            CircleEvaluation::new(layer_domain, rhs_col),
            CircleEvaluation::new(layer_domain, out_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    let mut layernorm_mults: Vec<Vec<M31>> = Vec::new();
    for layer in &layernorm_layers {
        let layer_size = 1usize << layer.log_size;
        let mults = compute_multiplicities(&layer.variances, &layer.rsqrt_table);
        let cols = build_layernorm_trace_columns(
            &layer.inputs, &layer.means, &layer.variances,
            &layer.rsqrt_vals, &layer.outputs, &mults,
            &layer.rsqrt_table, layer_size,
        );
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(cols));
        layernorm_mults.push(mults);
    }
    tree_builder.commit(channel);

    // Draw relation elements and build Tree 2 — only if LogUp components exist
    let mut activation_lookup: Option<ActivationRelation> = None;
    let mut layernorm_lookup: Option<LayerNormRelation> = None;
    let mut activation_claimed_sums: Vec<SecureField> = Vec::new();
    let mut layernorm_claimed_sums: Vec<SecureField> = Vec::new();

    if has_logup {
        // Draw relation elements — activation first, then layernorm
        if !activation_layers.is_empty() {
            activation_lookup = Some(ActivationRelation::draw(channel));
        }
        if !layernorm_layers.is_empty() {
            layernorm_lookup = Some(LayerNormRelation::draw(channel));
        }

        // Tree 2: Interaction traces (LogUp) — only for activation and layernorm
        // Add/Mul are pure AIR (no interaction columns)
        let mut tree_builder = commitment_scheme.tree_builder();

        if let Some(ref lookup) = activation_lookup {
        for (idx, layer) in activation_layers.iter().enumerate() {
            let layer_size = 1usize << layer.log_size;
            let layer_vec_size = layer_size >> LOG_N_LANES;
            let pad_input = layer.table.inputs[0];
            let pad_output = layer.table.outputs[0];

            let (table_in_col, table_out_col) = build_table_columns(&layer.table, layer_size);
            let (trace_in_col, trace_out_col, _) = build_trace_columns(
                &layer.inputs, &layer.outputs, &activation_mults[idx],
                pad_input, pad_output, layer_size,
            );

            let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
            let mut col_gen = logup_gen.new_col();

            for vec_row in 0..layer_vec_size {
                let q_table: PackedSecureField = lookup.lookup_elements().combine(
                    &[table_in_col.data[vec_row], table_out_col.data[vec_row]],
                );
                let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                    &[trace_in_col.data[vec_row], trace_out_col.data[vec_row]],
                );

                let mult_packed = pack_multiplicities(&activation_mults[idx], vec_row);
                let numerator = q_table - mult_packed * q_trace;
                let denominator = q_table * q_trace;

                col_gen.write_frac(vec_row, numerator, denominator);
            }
            col_gen.finalize_col();

            let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
            activation_claimed_sums.push(claimed_sum);
        }
    }

        if let Some(ref lookup) = layernorm_lookup {
            for (idx, layer) in layernorm_layers.iter().enumerate() {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;
                let (table_var_col, table_rsqrt_col) = build_table_columns(&layer.rsqrt_table, layer_size);

                // Build variance and rsqrt trace columns for LogUp
                let mut var_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut rsqrt_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let n = layer.variances.len().min(layer_size);
                for i in 0..n {
                    var_col.set(i, layer.variances[i]);
                    rsqrt_col.set(i, layer.rsqrt_vals[i]);
                }
                // Pad with table[0] values
                let pad_var = layer.rsqrt_table.inputs.first().copied().unwrap_or(M31::from(0));
                let pad_rsqrt = layer.rsqrt_table.outputs.first().copied().unwrap_or(M31::from(0));
                for i in n..layer_size {
                    var_col.set(i, pad_var);
                    rsqrt_col.set(i, pad_rsqrt);
                }

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
                let mut col_gen = logup_gen.new_col();

                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[table_var_col.data[vec_row], table_rsqrt_col.data[vec_row]],
                    );
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[var_col.data[vec_row], rsqrt_col.data[vec_row]],
                    );

                    let mult_packed = pack_multiplicities(&layernorm_mults[idx], vec_row);
                    let numerator = q_table - mult_packed * q_trace;
                    let denominator = q_table * q_trace;

                    col_gen.write_frac(vec_row, numerator, denominator);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                layernorm_claimed_sums.push(claimed_sum);
            }
        }
        tree_builder.commit(channel);
    } // end if has_logup

    // Build all components with shared allocator (same order as trace columns)
    let mut allocator = TraceLocationAllocator::default();
    let mut component_refs_storage: Vec<Box<dyn ComponentProverErased<B>>> = Vec::new();
    let mut activation_claims: Vec<LayerClaim> = Vec::new();
    let mut add_claims: Vec<LayerClaim> = Vec::new();
    let mut mul_claims: Vec<LayerClaim> = Vec::new();
    let mut layernorm_claims: Vec<LayerClaim> = Vec::new();

    // Activation components
    if let Some(ref lookup) = activation_lookup {
        for (idx, layer) in activation_layers.iter().enumerate() {
            let claimed_sum = activation_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                ActivationEval {
                    log_n_rows: layer.log_size,
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                    total_sum: claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            activation_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Add components (pure AIR, no LogUp)
    for layer in &add_layers {
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseAddEval { log_n_rows: layer.log_size },
            SecureField::default(),
        );
        component_refs_storage.push(Box::new(component));
        add_claims.push(LayerClaim {
            layer_index: layer.node_id,
            claimed_sum: SecureField::default(),
            trace_rows: 1 << layer.log_size,
        });
    }

    // Mul components (pure AIR, no LogUp)
    for layer in &mul_layers {
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseMulEval { log_n_rows: layer.log_size },
            SecureField::default(),
        );
        component_refs_storage.push(Box::new(component));
        mul_claims.push(LayerClaim {
            layer_index: layer.node_id,
            claimed_sum: SecureField::default(),
            trace_rows: 1 << layer.log_size,
        });
    }

    // LayerNorm components (LogUp)
    if let Some(ref lookup) = layernorm_lookup {
        for (idx, layer) in layernorm_layers.iter().enumerate() {
            let claimed_sum = layernorm_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                LayerNormEval {
                    log_n_rows: layer.log_size,
                    dim: layer.inputs.len(),
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            layernorm_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Single prove() call with all component refs
    let component_refs: Vec<&dyn ComponentProver<B>> = component_refs_storage
        .iter()
        .map(|c| c.as_component_prover())
        .collect();

    let stark_proof = prove::<B, MC>(
        &component_refs,
        channel,
        commitment_scheme,
    ).map_err(|e| AggregationError::ProvingError(format!("{e:?}")))?;

    Ok(AggregatedModelProofFor {
        unified_stark: Some(stark_proof),
        matmul_proofs,
        add_claims,
        mul_claims,
        layernorm_claims,
        execution,
        activation_claims,
    })
}

/// Prove an entire computation graph with aggregated STARK proof.
///
/// Convenience wrapper using `SimdBackend` + `Blake2sMerkleChannel`.
pub fn prove_model_aggregated(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProof, AggregationError> {
    prove_model_aggregated_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
}

/// Prove a model with aggregation using the best available backend.
///
/// Uses `GpuBackend` when CUDA is available, otherwise `SimdBackend`.
pub fn prove_model_aggregated_auto(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProof, AggregationError> {
    let gpu_available = crate::backend::gpu_is_available();
    info!(
        gpu_available,
        "Auto-selecting backend for off-chain aggregation"
    );
    crate::backend::with_best_backend(
        || {
            info!("Using SimdBackend for off-chain aggregation");
            prove_model_aggregated_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
        },
        || {
            info!("Using GpuBackend for off-chain aggregation");
            prove_model_aggregated_gpu(graph, input, weights)
        },
    )
}

/// GPU aggregated proving path — dispatches to `GpuBackend` when `cuda-runtime` is enabled.
fn prove_model_aggregated_gpu(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProof, AggregationError> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return prove_model_aggregated_with::<GpuBackend, Blake2sMerkleChannel>(
            graph, input, weights,
        );
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        prove_model_aggregated_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
    }
}

// ===== On-Chain Aggregated Proof =====

/// Aggregated model proof formatted for on-chain Cairo verification.
///
/// Uses `MatMulSumcheckProofOnChain` (Poseidon channel + MLE commitments)
/// instead of `MatMulSumcheckProof` (Blake2s).
pub struct AggregatedModelProofOnChain {
    /// Single STARK proof covering all non-matmul components.
    pub unified_stark: Option<StarkProof<Blake2sHash>>,
    /// Per-matmul on-chain sumcheck proofs, in layer order.
    pub matmul_proofs: Vec<(usize, MatMulSumcheckProofOnChain)>,
    /// Per-Add layer claims (verified inside unified STARK).
    pub add_claims: Vec<LayerClaim>,
    /// Per-Mul layer claims (verified inside unified STARK).
    pub mul_claims: Vec<LayerClaim>,
    /// Per-LayerNorm layer claims (verified inside unified STARK).
    pub layernorm_claims: Vec<LayerClaim>,
    /// Forward pass execution trace.
    pub execution: GraphExecution,
    /// Per-activation-layer claims.
    pub activation_claims: Vec<LayerClaim>,
}

impl AggregatedModelProofOnChain {
    /// Total number of proven layers across all proof types.
    pub fn num_proven_layers(&self) -> usize {
        self.matmul_proofs.len() + self.activation_claims.len()
            + self.add_claims.len() + self.mul_claims.len() + self.layernorm_claims.len()
    }
}

/// Prove an entire computation graph with on-chain Poseidon-based matmul proofs.
///
/// Same as `prove_model_aggregated` but calls `prove_matmul_sumcheck_onchain()`
/// for each matmul layer, producing proofs with Poseidon Merkle commitments
/// and MLE opening proofs compatible with the Cairo verifier.
///
/// The unified STARK covers all non-matmul components (already uses Blake2s).
pub fn prove_model_aggregated_onchain(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, AggregationError> {
    prove_model_aggregated_onchain_with::<SimdBackend>(graph, input, weights)
}

/// On-chain aggregated proving, generic over backend `B`.
///
/// The unified STARK always uses `Blake2sMerkleChannel` (matching the on-chain
/// verifier). Matmul sumcheck proofs use Poseidon (independent of `B`).
/// Genericizing `B` allows GPU acceleration of Merkle hashing, FRI, and
/// quotient evaluation for the unified STARK portion.
pub(crate) fn prove_model_aggregated_onchain_with<B>(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, AggregationError>
where
    B: BackendForChannel<Blake2sMerkleChannel> + PolyOps,
    FrameworkComponent<ActivationEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseAddEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseMulEval>: ComponentProver<B>,
    FrameworkComponent<LayerNormEval>: ComponentProver<B>,
{
    info!(
        backend = std::any::type_name::<B>(),
        "Proving unified STARK (on-chain aggregation, Blake2sMerkleChannel)"
    );
    let config = PcsConfig::default();
    let mut intermediates: Vec<(usize, M31Matrix)> = Vec::new();
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();
    let mut current = input.clone();

    let mut matmul_proofs: Vec<(usize, MatMulSumcheckProofOnChain)> = Vec::new();
    let mut activation_layers: Vec<ActivationLayerData> = Vec::new();
    let mut add_layers: Vec<AddLayerData> = Vec::new();
    let mut mul_layers: Vec<MulLayerData> = Vec::new();
    let mut layernorm_layers: Vec<LayerNormLayerData> = Vec::new();

    // Memory budget for tiled matmul auto-dispatch (4GB default)
    const TILED_MEMORY_BUDGET: usize = 4 * 1024 * 1024 * 1024;

    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];

        // Resolve current input from first dependency
        if let Some(&first_input) = node.inputs.first() {
            if let Some(inp) = node_outputs.get(&first_input) {
                current = inp.clone();
            }
        }

        match &node.op {
            GraphOp::MatMul { dims } => {
                let weight = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;

                let output = matmul_m31(&current, weight);

                // Auto-dispatch: use tiled proving for large matmuls
                let (m, k, n) = *dims;
                let (_, estimated_mem) = estimate_sumcheck_memory(m, k, n);

                let proof = if estimated_mem > TILED_MEMORY_BUDGET {
                    info!(
                        node_id = node.id,
                        m, k, n,
                        estimated_mem,
                        "Using tiled matmul proving (exceeds memory budget)"
                    );
                    let config = TiledMatMulConfig::from_memory_budget(
                        m, k, n, TILED_MEMORY_BUDGET,
                    );
                    let tiled = prove_tiled_matmul(&current, weight, &output, &config)
                        .map_err(|e| ModelError::ProvingError {
                            layer: node.id,
                            message: format!("Tiled matmul: {e}"),
                        })?;
                    compose_tiled_proof(&tiled)
                } else {
                    // Use standard on-chain proving
                    prove_matmul_sumcheck_onchain(&current, weight, &output)
                        .map_err(|e| ModelError::ProvingError {
                            layer: node.id,
                            message: format!("MatMul sumcheck (on-chain): {e}"),
                        })?
                };

                intermediates.push((node.id, current.clone()));
                matmul_proofs.push((node.id, proof));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Activation { activation_type, .. } => {
                let f = activation_type.as_fn();
                let output = crate::compiler::prove::apply_activation_pub(&current, &*f);

                let act_log_size = activation_type.recommended_table_log_size();
                let table = PrecomputedTable::build(|x| (*f)(x), act_log_size);
                let flat_inputs = current.data.clone();
                let flat_outputs = output.data.clone();

                activation_layers.push(ActivationLayerData {
                    node_id: node.id,
                    inputs: flat_inputs,
                    outputs: flat_outputs,
                    table,
                    log_size: act_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Add { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_add(&lhs, &rhs);

                let add_log_size = data_log_size(output.data.len());
                add_layers.push(AddLayerData {
                    node_id: node.id,
                    lhs: lhs.data.clone(),
                    rhs: rhs.data.clone(),
                    output: output.data.clone(),
                    log_size: add_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Mul { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_mul(&lhs, &rhs);

                let mul_log_size = data_log_size(output.data.len());
                mul_layers.push(MulLayerData {
                    node_id: node.id,
                    lhs: lhs.data.clone(),
                    rhs: rhs.data.clone(),
                    output: output.data.clone(),
                    log_size: mul_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::LayerNorm { dim } => {
                let ln_log_size = LayerNormConfig::new(*dim).rsqrt_table_log_size;
                let ln = apply_layernorm_detailed(&current, *dim);
                let rsqrt_table = build_rsqrt_table(ln_log_size);

                layernorm_layers.push(LayerNormLayerData {
                    node_id: node.id,
                    inputs: ln.inputs.clone(),
                    means: ln.means.clone(),
                    variances: ln.variances.clone(),
                    rsqrt_vals: ln.rsqrt_vals.clone(),
                    outputs: ln.outputs.clone(),
                    rsqrt_table,
                    log_size: ln_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, ln.output_matrix.clone());
                current = ln.output_matrix;
            }

            _ => {
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, current.clone());
            }
        }
    }

    let execution = GraphExecution {
        intermediates,
        output: current,
    };

    // Check if there are any non-matmul components to aggregate
    let has_components = !activation_layers.is_empty()
        || !add_layers.is_empty() || !mul_layers.is_empty() || !layernorm_layers.is_empty();

    if !has_components {
        return Ok(AggregatedModelProofOnChain {
            unified_stark: None,
            matmul_proofs,
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            execution,
            activation_claims: Vec::new(),
        });
    }

    // Build unified STARK using backend B + Blake2sMerkleChannel
    // Per-component log_sizes: each component uses its own size derived from
    // its table or data length. The max_log_size drives twiddle precomputation.
    let all_log_sizes: Vec<u32> = activation_layers.iter().map(|l| l.log_size)
        .chain(add_layers.iter().map(|l| l.log_size))
        .chain(mul_layers.iter().map(|l| l.log_size))
        .chain(layernorm_layers.iter().map(|l| l.log_size))
        .collect();
    let max_log_size = *all_log_sizes.iter().max().unwrap();

    let max_degree_bound = max_log_size + 1;
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(max_degree_bound + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, Blake2sMerkleChannel>::new(config, &twiddles);

    let has_logup = !activation_layers.is_empty() || !layernorm_layers.is_empty();

    // Tree 0: Preprocessed (activation tables + layernorm rsqrt tables)
    // Always committed (may be empty for pure-AIR models).
    {
        let mut tree_builder = commitment_scheme.tree_builder();
        for layer in &activation_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let (table_input_col, table_output_col) = build_table_columns(&layer.table, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, table_input_col),
                CircleEvaluation::new(layer_domain, table_output_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        for layer in &layernorm_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let (table_var_col, table_rsqrt_col) = build_table_columns(&layer.rsqrt_table, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, table_var_col),
                CircleEvaluation::new(layer_domain, table_rsqrt_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        tree_builder.commit(channel);
    }

    // Tree 1: Execution traces (activation + add + mul + layernorm)
    let mut tree_builder = commitment_scheme.tree_builder();
    let mut activation_mults: Vec<Vec<M31>> = Vec::new();
    for layer in &activation_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let pad_input = layer.table.inputs[0];
        let pad_output = layer.table.outputs[0];
        let padding_count = layer_size.saturating_sub(layer.inputs.len());

        let mut mults = compute_multiplicities(&layer.inputs, &layer.table);
        if padding_count > 0 {
            mults[0] += M31::from(padding_count as u32);
        }

        let (trace_in, trace_out, mult_col) = build_trace_columns(
            &layer.inputs, &layer.outputs, &mults,
            pad_input, pad_output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, trace_in),
            CircleEvaluation::new(layer_domain, trace_out),
            CircleEvaluation::new(layer_domain, mult_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        activation_mults.push(mults);
    }
    for layer in &add_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let (lhs_col, rhs_col, out_col) = build_elementwise_trace_columns(
            &layer.lhs, &layer.rhs, &layer.output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, lhs_col),
            CircleEvaluation::new(layer_domain, rhs_col),
            CircleEvaluation::new(layer_domain, out_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    for layer in &mul_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let (lhs_col, rhs_col, out_col) = build_elementwise_trace_columns(
            &layer.lhs, &layer.rhs, &layer.output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, lhs_col),
            CircleEvaluation::new(layer_domain, rhs_col),
            CircleEvaluation::new(layer_domain, out_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    let mut layernorm_mults: Vec<Vec<M31>> = Vec::new();
    for layer in &layernorm_layers {
        let layer_size = 1usize << layer.log_size;
        let mults = compute_multiplicities(&layer.variances, &layer.rsqrt_table);
        let cols = build_layernorm_trace_columns(
            &layer.inputs, &layer.means, &layer.variances,
            &layer.rsqrt_vals, &layer.outputs, &mults,
            &layer.rsqrt_table, layer_size,
        );
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(cols));
        layernorm_mults.push(mults);
    }
    tree_builder.commit(channel);

    // Draw relation elements and build Tree 2 — only if LogUp components exist
    let mut activation_lookup: Option<ActivationRelation> = None;
    let mut layernorm_lookup: Option<LayerNormRelation> = None;
    let mut activation_claimed_sums: Vec<SecureField> = Vec::new();
    let mut layernorm_claimed_sums: Vec<SecureField> = Vec::new();

    if has_logup {
        if !activation_layers.is_empty() {
            activation_lookup = Some(ActivationRelation::draw(channel));
        }
        if !layernorm_layers.is_empty() {
            layernorm_lookup = Some(LayerNormRelation::draw(channel));
        }

        // Tree 2: Interaction traces (LogUp for activation + layernorm)
        let mut tree_builder = commitment_scheme.tree_builder();

        if let Some(ref lookup) = activation_lookup {
            for (idx, layer) in activation_layers.iter().enumerate() {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;
                let pad_input = layer.table.inputs[0];
                let pad_output = layer.table.outputs[0];

                let (table_in_col, table_out_col) = build_table_columns(&layer.table, layer_size);
                let (trace_in_col, trace_out_col, _) = build_trace_columns(
                    &layer.inputs, &layer.outputs, &activation_mults[idx],
                    pad_input, pad_output, layer_size,
                );

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
                let mut col_gen = logup_gen.new_col();

                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[table_in_col.data[vec_row], table_out_col.data[vec_row]],
                    );
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[trace_in_col.data[vec_row], trace_out_col.data[vec_row]],
                    );

                    let mult_packed = pack_multiplicities(&activation_mults[idx], vec_row);
                    let numerator = q_table - mult_packed * q_trace;
                    let denominator = q_table * q_trace;

                    col_gen.write_frac(vec_row, numerator, denominator);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                activation_claimed_sums.push(claimed_sum);
            }
        }

        if let Some(ref lookup) = layernorm_lookup {
            for (idx, layer) in layernorm_layers.iter().enumerate() {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;
                let (table_var_col, table_rsqrt_col) = build_table_columns(&layer.rsqrt_table, layer_size);

                let mut var_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut rsqrt_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let n = layer.variances.len().min(layer_size);
                for i in 0..n {
                    var_col.set(i, layer.variances[i]);
                    rsqrt_col.set(i, layer.rsqrt_vals[i]);
                }
                let pad_var = layer.rsqrt_table.inputs.first().copied().unwrap_or(M31::from(0));
                let pad_rsqrt = layer.rsqrt_table.outputs.first().copied().unwrap_or(M31::from(0));
                for i in n..layer_size {
                    var_col.set(i, pad_var);
                    rsqrt_col.set(i, pad_rsqrt);
                }

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
                let mut col_gen = logup_gen.new_col();

                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[table_var_col.data[vec_row], table_rsqrt_col.data[vec_row]],
                    );
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[var_col.data[vec_row], rsqrt_col.data[vec_row]],
                    );

                    let mult_packed = pack_multiplicities(&layernorm_mults[idx], vec_row);
                    let numerator = q_table - mult_packed * q_trace;
                    let denominator = q_table * q_trace;

                    col_gen.write_frac(vec_row, numerator, denominator);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                layernorm_claimed_sums.push(claimed_sum);
            }
        }
        tree_builder.commit(channel);
    } // end if has_logup

    // Build all components with shared allocator
    let mut allocator = TraceLocationAllocator::default();
    let mut component_refs_storage: Vec<Box<dyn ComponentProverErased<B>>> = Vec::new();
    let mut activation_claims: Vec<LayerClaim> = Vec::new();
    let mut add_claims: Vec<LayerClaim> = Vec::new();
    let mut mul_claims: Vec<LayerClaim> = Vec::new();
    let mut layernorm_claims: Vec<LayerClaim> = Vec::new();

    // Activation components
    if let Some(ref lookup) = activation_lookup {
        for (idx, layer) in activation_layers.iter().enumerate() {
            let claimed_sum = activation_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                ActivationEval {
                    log_n_rows: layer.log_size,
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                    total_sum: claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            activation_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Add components
    for layer in &add_layers {
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseAddEval { log_n_rows: layer.log_size },
            SecureField::default(),
        );
        component_refs_storage.push(Box::new(component));
        add_claims.push(LayerClaim {
            layer_index: layer.node_id,
            claimed_sum: SecureField::default(),
            trace_rows: 1 << layer.log_size,
        });
    }

    // Mul components
    for layer in &mul_layers {
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseMulEval { log_n_rows: layer.log_size },
            SecureField::default(),
        );
        component_refs_storage.push(Box::new(component));
        mul_claims.push(LayerClaim {
            layer_index: layer.node_id,
            claimed_sum: SecureField::default(),
            trace_rows: 1 << layer.log_size,
        });
    }

    // LayerNorm components
    if let Some(ref lookup) = layernorm_lookup {
        for (idx, layer) in layernorm_layers.iter().enumerate() {
            let claimed_sum = layernorm_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                LayerNormEval {
                    log_n_rows: layer.log_size,
                    dim: layer.inputs.len(),
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            layernorm_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    let component_refs: Vec<&dyn ComponentProver<B>> = component_refs_storage
        .iter()
        .map(|c| c.as_component_prover())
        .collect();

    let stark_proof = prove::<B, Blake2sMerkleChannel>(
        &component_refs,
        channel,
        commitment_scheme,
    ).map_err(|e| AggregationError::ProvingError(format!("{e:?}")))?;

    Ok(AggregatedModelProofOnChain {
        unified_stark: Some(stark_proof),
        matmul_proofs,
        add_claims,
        mul_claims,
        layernorm_claims,
        execution,
        activation_claims,
    })
}

/// Prove with auto GPU dispatch for on-chain format.
///
/// Uses `GpuBackend` when CUDA is available, otherwise `SimdBackend`.
/// GPU accelerates the unified STARK (Merkle hashing, FRI, quotient eval).
/// Matmul sumcheck proofs use Poseidon and are unaffected by the backend choice.
pub fn prove_model_aggregated_onchain_auto(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, AggregationError> {
    let gpu_available = crate::backend::gpu_is_available();
    info!(
        gpu_available,
        "Auto-selecting backend for on-chain aggregation"
    );
    crate::backend::with_best_backend(
        || {
            info!("Using SimdBackend for on-chain aggregation");
            prove_model_aggregated_onchain_with::<SimdBackend>(graph, input, weights)
        },
        || {
            info!("Using GpuBackend for on-chain aggregation");
            prove_model_aggregated_onchain_gpu(graph, input, weights)
        },
    )
}

/// GPU proving path for on-chain aggregation.
fn prove_model_aggregated_onchain_gpu(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, AggregationError> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return prove_model_aggregated_onchain_with::<GpuBackend>(
            graph, input, weights,
        );
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        prove_model_aggregated_onchain_with::<SimdBackend>(graph, input, weights)
    }
}

// --- Type-erased ComponentProver trait for heterogeneous component storage ---

/// Trait object wrapper to hold different `FrameworkComponent<E>` types
/// in a single Vec for the unified prove() call.
trait ComponentProverErased<B: stwo::prover::backend::Backend> {
    fn as_component_prover(&self) -> &dyn ComponentProver<B>;
}

impl<B, E> ComponentProverErased<B> for FrameworkComponent<E>
where
    B: stwo::prover::backend::Backend,
    E: stwo_constraint_framework::FrameworkEval,
    FrameworkComponent<E>: ComponentProver<B>,
{
    fn as_component_prover(&self) -> &dyn ComponentProver<B> {
        self
    }
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

/// Build 3 trace columns for elementwise Add/Mul (lhs, rhs, output).
fn build_elementwise_trace_columns(
    lhs: &[M31],
    rhs: &[M31],
    output: &[M31],
    size: usize,
) -> (
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
) {
    let mut lhs_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rhs_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut out_col = Col::<SimdBackend, BaseField>::zeros(size);

    let n = lhs.len().min(rhs.len()).min(output.len()).min(size);
    for i in 0..n {
        lhs_col.set(i, lhs[i]);
        rhs_col.set(i, rhs[i]);
        out_col.set(i, output[i]);
    }
    // Pad remaining rows with zeros (identity for Add/Mul constraints)

    (lhs_col, rhs_col, out_col)
}

/// Build 6 trace columns for LayerNorm (input, mean, var, rsqrt, output, multiplicity).
fn build_layernorm_trace_columns(
    inputs: &[M31],
    means: &[M31],
    variances: &[M31],
    rsqrt_vals: &[M31],
    outputs: &[M31],
    multiplicities: &[M31],
    rsqrt_table: &PrecomputedTable,
    size: usize,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, stwo::prover::poly::BitReversedOrder>> {
    let domain = CanonicCoset::new(size.ilog2()).circle_domain();

    let pad_var = rsqrt_table.inputs.first().copied().unwrap_or(M31::from(0));
    let pad_rsqrt = rsqrt_table.outputs.first().copied().unwrap_or(M31::from(0));

    let mut input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mean_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut var_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rsqrt_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);

    let n = inputs.len().min(size);
    for i in 0..n {
        input_col.set(i, inputs[i]);
        mean_col.set(i, means[i]);
        var_col.set(i, variances[i]);
        rsqrt_col.set(i, rsqrt_vals[i]);
        output_col.set(i, outputs[i]);
    }
    for i in n..size {
        var_col.set(i, pad_var);
        rsqrt_col.set(i, pad_rsqrt);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    vec![
        CircleEvaluation::new(domain, input_col),
        CircleEvaluation::new(domain, mean_col),
        CircleEvaluation::new(domain, var_col),
        CircleEvaluation::new(domain, rsqrt_col),
        CircleEvaluation::new(domain, output_col),
        CircleEvaluation::new(domain, mult_col),
    ]
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

/// Collect all matmul proofs and activation claims from chunked proving results.
///
/// Useful for composing chunk-level proofs into a single logical proof payload
/// for recursive STARK verification.
pub fn collect_chunk_proofs(
    results: &[crate::compiler::chunked::ChunkProofResult],
) -> (Vec<(usize, MatMulSumcheckProofOnChain)>, Vec<LayerClaim>) {
    let mut all_matmul: Vec<(usize, MatMulSumcheckProofOnChain)> = Vec::new();
    let mut all_claims: Vec<LayerClaim> = Vec::new();

    for chunk in results {
        for (layer_idx, proof) in &chunk.proof.matmul_proofs {
            // Remap to original graph indices
            let original_idx = chunk.node_range.start + layer_idx;
            all_matmul.push((original_idx, proof.clone()));
        }
        for claim in &chunk.proof.activation_claims {
            let mut remapped = claim.clone();
            remapped.layer_index += chunk.node_range.start;
            all_claims.push(remapped);
        }
    }

    (all_matmul, all_claims)
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
        // Model with no activations — should return None for unified_stark
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

        assert!(proof.unified_stark.is_none());
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
        assert!(proof.unified_stark.is_some(), "should have unified STARK");
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
        let proof: AggregatedModelProof = AggregatedModelProofFor {
            unified_stark: None,
            matmul_proofs: Vec::new(),
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
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

    // === On-Chain Aggregation Tests ===

    #[test]
    fn test_aggregated_onchain_mlp() {
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

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain aggregated MLP proving should succeed");

        assert!(proof.unified_stark.is_some());
        assert_eq!(proof.matmul_proofs.len(), 3);
        assert_eq!(proof.activation_claims.len(), 2);

        // All matmul proofs should have Poseidon commitments
        for (_, mp) in &proof.matmul_proofs {
            assert_ne!(mp.a_commitment, starknet_ff::FieldElement::ZERO);
            assert_ne!(mp.b_commitment, starknet_ff::FieldElement::ZERO);
        }
    }

    #[test]
    fn test_aggregated_onchain_matmul_only() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain matmul-only proving should succeed");

        assert!(proof.unified_stark.is_none());
        assert_eq!(proof.matmul_proofs.len(), 1);

        // Verify the matmul proof has on-chain format
        let (_, mp) = &proof.matmul_proofs[0];
        assert_eq!(mp.m, 1);
        assert_eq!(mp.k, 4);
        assert_eq!(mp.n, 2);
    }

    #[test]
    fn test_aggregated_onchain_auto_mlp() {
        let mut builder = GraphBuilder::new((1, 4));
        builder
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
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let proof = prove_model_aggregated_onchain_auto(&graph, &input, &weights)
            .expect("on-chain auto aggregated proving should succeed");

        assert!(proof.unified_stark.is_some());
        assert_eq!(proof.matmul_proofs.len(), 2);
        assert_eq!(proof.activation_claims.len(), 1);
    }

    #[test]
    fn test_aggregated_onchain_auto_matmul_only() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated_onchain_auto(&graph, &input, &weights)
            .expect("on-chain auto matmul-only proving should succeed");

        assert!(proof.unified_stark.is_none());
        assert_eq!(proof.matmul_proofs.len(), 1);
    }

    #[test]
    fn test_aggregated_with_add_residual() {
        // Residual connection: matmul → relu → matmul → add(skip) → matmul
        let mut builder = GraphBuilder::new((1, 8));
        builder.linear(8);
        let branch = builder.fork();
        builder.activation(ActivationType::ReLU);
        builder.linear(8);
        builder.add_from(branch);
        builder.linear(4);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 8);
        for j in 0..8 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { w2.set(i, j, M31::from(((i * j) % 7 + 1) as u32)); } }
        weights.add_weight(2, w2);
        let mut w4 = M31Matrix::new(8, 4);
        for i in 0..8 { for j in 0..4 { w4.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(4, w4);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated proving with residual Add should succeed");

        assert_eq!(proof.matmul_proofs.len(), 3, "3 matmul proofs");
        assert_eq!(proof.activation_claims.len(), 1, "1 activation claim (ReLU)");
        assert_eq!(proof.add_claims.len(), 1, "1 Add claim");
        assert_eq!(proof.mul_claims.len(), 0, "no Mul claims");
        assert_eq!(proof.layernorm_claims.len(), 0, "no LayerNorm claims");
        assert_eq!(proof.num_proven_layers(), 5, "total: 3 matmul + 1 activation + 1 add");
        // Unified STARK covers both activation and add
        assert!(proof.unified_stark.is_some(), "unified STARK covers activation + add");
    }

    #[test]
    fn test_aggregated_onchain_with_add() {
        let mut builder = GraphBuilder::new((1, 8));
        builder.linear(8);
        let branch = builder.fork();
        builder.activation(ActivationType::ReLU);
        builder.linear(8);
        builder.add_from(branch);
        builder.linear(4);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 8);
        for j in 0..8 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { w2.set(i, j, M31::from(((i * j) % 7 + 1) as u32)); } }
        weights.add_weight(2, w2);
        let mut w4 = M31Matrix::new(8, 4);
        for i in 0..8 { for j in 0..4 { w4.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(4, w4);

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain aggregated proving with Add should succeed");

        assert_eq!(proof.matmul_proofs.len(), 3);
        assert_eq!(proof.activation_claims.len(), 1);
        assert_eq!(proof.add_claims.len(), 1, "1 Add claim (on-chain)");
    }

    #[test]
    fn test_aggregated_with_mul() {
        // Element-wise multiply: matmul → fork → matmul → mul(branch) → matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let branch = builder.fork();
        builder.linear(4);
        builder.mul_from(branch);
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w1 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w1.set(i, j, M31::from(((i + j + 1) % 7 + 1) as u32)); } }
        weights.add_weight(1, w1);
        let mut w3 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w3.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(3, w3);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated proving with Mul should succeed");

        assert_eq!(proof.matmul_proofs.len(), 3, "3 matmul proofs");
        assert_eq!(proof.mul_claims.len(), 1, "1 Mul claim");
        assert_eq!(proof.add_claims.len(), 0, "no Add claims");
        // Unified STARK covers mul
        assert!(proof.unified_stark.is_some(), "unified STARK covers mul");
    }
}
