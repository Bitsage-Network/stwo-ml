//! Proving pipeline for computation graphs.
//!
//! Takes a `ComputationGraph`, input data, and weights, then generates
//! STARK proofs for each layer using the appropriate component.

use stwo::core::channel::MerkleChannel;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{BackendForChannel, Col, Column, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::prove;
use stwo::prover::CommitmentSchemeProver;

use stwo_constraint_framework::{FrameworkComponent, LogupTraceGenerator, TraceLocationAllocator};

use crate::backend::convert_evaluations;
use tracing::info;

use std::collections::HashMap;

use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use crate::components::activation::{compute_multiplicities, ActivationEval, ActivationRelation};
use crate::components::elementwise::{ElementwiseAddEval, ElementwiseMulEval};
use crate::components::layernorm::{
    build_rsqrt_table, LayerNormConfig, LayerNormEval, LayerNormRelation,
};
use crate::components::matmul::{
    matmul_m31, prove_matmul_sumcheck_auto, M31Matrix, MatMulSumcheckProof,
};
use crate::gadgets::lookup_table::PrecomputedTable;

/// Proof for a single layer — either a STARK (activation/layernorm) or sumcheck (matmul).
#[derive(Debug)]
pub enum LayerProofKind<H: MerkleHasherLifted> {
    /// Activation/LayerNorm: LogUp-based STARK proof.
    Stark(StarkProof<H>),
    /// MatMul: Sumcheck proof over multilinear extensions.
    Sumcheck(MatMulSumcheckProof),
    /// Element-wise Add: AIR constraint proof (output - lhs - rhs = 0).
    ElementwiseAdd(StarkProof<H>),
    /// Element-wise Mul: AIR constraint proof (output - lhs * rhs = 0).
    ElementwiseMul(StarkProof<H>),
    /// LayerNorm: LogUp proof for rsqrt lookup + output constraint.
    LayerNorm(StarkProof<H>),
    /// RMSNorm: LogUp proof for rsqrt lookup + output = input * rsqrt constraint.
    RMSNorm(StarkProof<H>),
    /// Identity: no proof needed (structural op only).
    Passthrough,
}

/// Result of proving a single layer.
#[derive(Debug)]
pub struct LayerProof<H: MerkleHasherLifted> {
    pub kind: LayerProofKind<H>,
    pub claimed_sum: SecureField,
    pub layer_index: usize,
}

/// Result of proving an entire model (forward pass + per-layer proofs).
pub struct ModelProof<H: MerkleHasherLifted> {
    pub layer_proofs: Vec<LayerProof<H>>,
    pub output: M31Matrix,
}

/// Intermediate execution state for the forward pass.
pub struct GraphExecution {
    pub intermediates: Vec<(usize, M31Matrix)>,
    pub output: M31Matrix,
}

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

/// Raw SIMD columns built for activation layer proving.
///
/// Contains both the `CircleEvaluation`s (for commitment) and the raw
/// packed column data (for LogUp computation after lookup elements are drawn).
struct ActivationColumns {
    /// Tree 0: preprocessed lookup table columns.
    preprocessed: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    /// Tree 1: execution trace columns.
    execution: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    /// Raw packed columns — kept for LogUp computation.
    table_input_col: Col<SimdBackend, BaseField>,
    table_output_col: Col<SimdBackend, BaseField>,
    trace_input_col: Col<SimdBackend, BaseField>,
    trace_output_col: Col<SimdBackend, BaseField>,
    multiplicities: Vec<M31>,
    log_size: u32,
}

/// Generic model proof result: per-layer proofs + execution trace.
pub type ModelProofResultFor<H> = (Vec<LayerProof<H>>, GraphExecution);

/// Model proof result using Blake2s (default).
pub type ModelProofResult = ModelProofResultFor<Blake2sHash>;

/// Error type for model proving.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Proving error at layer {layer}: {message}")]
    ProvingError { layer: usize, message: String },
    #[error("Verification error at layer {layer}: {message}")]
    VerificationError { layer: usize, message: String },
    #[error("Graph error: {0}")]
    GraphError(String),
    #[error("Weight not found for node {0}")]
    MissingWeight(usize),
}

/// Prove a single activation layer, generic over backend and Merkle channel.
///
/// Uses LogUp to verify every (input, output) pair exists in the precomputed table.
/// Tree layout: [preprocessed table | execution trace | LogUp interaction].
///
/// Trace generation uses `SimdBackend` (fast column building). Commitment and
/// proving use backend `B`, so GPU acceleration applies to the cryptographic
/// operations (Merkle trees, FRI, quotient evaluation). LogUp computation uses
/// `SimdBackend` packed operations but draws lookup elements from the real
/// `MC::C` channel, making this fully generic over any `MerkleChannel`.
pub fn prove_activation_layer<B, MC>(
    inputs: &[M31],
    outputs: &[M31],
    table: &PrecomputedTable,
    config: PcsConfig,
) -> Result<
    (
        FrameworkComponent<ActivationEval>,
        StarkProof<<MC as MerkleChannel>::H>,
    ),
    ModelError,
>
where
    B: BackendForChannel<MC> + PolyOps + ColumnOps<BaseField>,
    <B as ColumnOps<BaseField>>::Column: 'static,
    MC: MerkleChannel,
    FrameworkComponent<ActivationEval>: stwo::prover::ComponentProver<B>,
{
    // --- Phase A: Build raw SIMD columns (no commitment yet) ---
    let cols = build_activation_columns_simd(inputs, outputs, table);
    let log_size = cols.log_size;

    // --- Phase B: Commit with real B+MC scheme ---
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::new(config, &twiddles);

    // Tree 0: Preprocessed columns
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(
        cols.preprocessed,
    ));
    tree_builder.commit(channel);

    // Tree 1: Execution trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(
        cols.execution,
    ));
    tree_builder.commit(channel);

    // --- Phase C: Draw lookup elements from REAL MC::C channel ---
    // The channel state is now deterministic based on the Merkle roots committed
    // above with backend B and hash MC::H. This works with any MerkleChannel.
    let lookup_elements: ActivationRelation = ActivationRelation::draw(channel);

    // --- Phase D: Compute LogUp interaction trace on SimdBackend ---
    // LogupTraceGenerator is SimdBackend-only. The lookup elements are just
    // SecureField scalars — they work identically regardless of which channel drew them.
    let (interaction_simd, claimed_sum) = compute_activation_logup_simd(
        &cols.table_input_col,
        &cols.table_output_col,
        &cols.trace_input_col,
        &cols.trace_output_col,
        &cols.multiplicities,
        log_size,
        &lookup_elements,
    );

    // Tree 2: LogUp interaction trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(
        interaction_simd,
    ));
    tree_builder.commit(channel);

    // --- Phase E: Build component and prove ---
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        ActivationEval {
            log_n_rows: log_size,
            lookup_elements,
            claimed_sum,
            total_sum: claimed_sum,
            activation_type_tag: 0, // standalone proof — no cross-type batching
        },
        claimed_sum,
    );

    let proof = prove::<B, MC>(&[&component], channel, commitment_scheme).map_err(|e| {
        ModelError::ProvingError {
            layer: 0,
            message: format!("{e:?}"),
        }
    })?;

    Ok((component, proof))
}

/// Build raw SIMD columns for activation layer proving (no commitment).
///
/// Returns the packed columns and CircleEvaluations needed for:
/// - Commitment (preprocessed + execution evals)
/// - LogUp computation (raw packed column data)
fn build_activation_columns_simd(
    inputs: &[M31],
    outputs: &[M31],
    table: &PrecomputedTable,
) -> ActivationColumns {
    let log_size = table.log_size.max(4);
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Pad trace with the first table entry for rows beyond real data.
    let pad_input = table.inputs[0];
    let pad_output = table.outputs[0];
    let padding_count = size.saturating_sub(inputs.len());

    // Build multiplicities: count real uses + padding uses
    let mut multiplicities = compute_multiplicities(inputs, table);
    if padding_count > 0 {
        multiplicities[0] += M31::from(padding_count as u32);
    }

    // Build packed columns for the execution trace
    let mut trace_input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);

    for (i, (&inp, &out)) in inputs.iter().zip(outputs.iter()).enumerate().take(size) {
        trace_input_col.set(i, inp);
        trace_output_col.set(i, out);
    }
    for i in inputs.len()..size {
        trace_input_col.set(i, pad_input);
        trace_output_col.set(i, pad_output);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    // Build packed columns for the preprocessed table
    let mut table_input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut table_output_col = Col::<SimdBackend, BaseField>::zeros(size);
    for (i, (&inp, &out)) in table
        .inputs
        .iter()
        .zip(table.outputs.iter())
        .enumerate()
        .take(size)
    {
        table_input_col.set(i, inp);
        table_output_col.set(i, out);
    }

    // Build CircleEvaluations for commitment (clone columns since LogUp needs originals)
    let preprocessed = vec![
        CircleEvaluation::new(domain, table_input_col.clone()),
        CircleEvaluation::new(domain, table_output_col.clone()),
    ];
    let execution = vec![
        CircleEvaluation::new(domain, trace_input_col.clone()),
        CircleEvaluation::new(domain, trace_output_col.clone()),
        CircleEvaluation::new(domain, mult_col),
    ];

    ActivationColumns {
        preprocessed,
        execution,
        table_input_col,
        table_output_col,
        trace_input_col,
        trace_output_col,
        multiplicities,
        log_size,
    }
}

/// Compute LogUp interaction trace on SimdBackend given lookup elements.
///
/// The lookup elements can come from any `Channel` implementation — they're
/// just `SecureField` scalars used for random linear combinations.
fn compute_activation_logup_simd(
    table_input_col: &Col<SimdBackend, BaseField>,
    table_output_col: &Col<SimdBackend, BaseField>,
    trace_input_col: &Col<SimdBackend, BaseField>,
    trace_output_col: &Col<SimdBackend, BaseField>,
    multiplicities: &[M31],
    log_size: u32,
    lookup_elements: &ActivationRelation,
) -> (
    Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};

    let size = 1usize << log_size;
    let vec_size = size >> LOG_N_LANES;

    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let mut col_gen = logup_gen.new_col();

    // Type tag = 0 for standalone proofs (no cross-type batching).
    let tag_packed = PackedBaseField::broadcast(M31::from(0u32));

    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            tag_packed,
            table_input_col.data[vec_row],
            table_output_col.data[vec_row],
        ]);
        let q_trace: PackedSecureField = lookup_elements.lookup_elements().combine(&[
            tag_packed,
            trace_input_col.data[vec_row],
            trace_output_col.data[vec_row],
        ]);

        let mult_packed: PackedSecureField = mult_col_data_at(multiplicities, vec_row, log_size);

        let numerator = q_table - mult_packed * q_trace;
        let denominator = q_table * q_trace;

        col_gen.write_frac(vec_row, numerator, denominator);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

/// Helper: get packed multiplicity values at a given vec_row.
fn mult_col_data_at(multiplicities: &[M31], vec_row: usize, _log_size: u32) -> PackedSecureField {
    use stwo::prover::backend::simd::m31::PackedBaseField;

    let n_lanes = 16usize; // N_LANES = 2^LOG_N_LANES = 2^4 = 16
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

/// Prove an element-wise Add layer: `output[i] = lhs[i] + rhs[i]`.
///
/// Pure AIR constraint — no LogUp. Two Merkle trees:
/// - Tree 0: Empty preprocessed (required by protocol)
/// - Tree 1: Execution trace [lhs, rhs, output] (3 columns)
pub fn prove_elementwise_add_layer<B, MC>(
    lhs: &[M31],
    rhs: &[M31],
    output: &[M31],
    config: PcsConfig,
) -> Result<
    (
        FrameworkComponent<ElementwiseAddEval>,
        StarkProof<<MC as MerkleChannel>::H>,
    ),
    ModelError,
>
where
    B: BackendForChannel<MC> + PolyOps + ColumnOps<BaseField>,
    <B as ColumnOps<BaseField>>::Column: 'static,
    MC: MerkleChannel,
    FrameworkComponent<ElementwiseAddEval>: stwo::prover::ComponentProver<B>,
{
    let n = lhs.len().max(rhs.len()).max(output.len()).max(16);
    let log_size = (n as f64).log2().ceil() as u32;
    let log_size = log_size.max(4); // STWO minimum
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Build trace columns
    let mut lhs_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rhs_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut out_col = Col::<SimdBackend, BaseField>::zeros(size);

    for (i, (&l, &r)) in lhs.iter().zip(rhs.iter()).enumerate().take(size) {
        lhs_col.set(i, l);
        rhs_col.set(i, r);
    }
    for (i, &o) in output.iter().enumerate().take(size) {
        out_col.set(i, o);
    }

    let trace = vec![
        CircleEvaluation::new(domain, lhs_col),
        CircleEvaluation::new(domain, rhs_col),
        CircleEvaluation::new(domain, out_col),
    ];

    // Commit
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::new(config, &twiddles);

    // Tree 0: Empty preprocessed
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(vec![]));
    tree_builder.commit(channel);

    // Tree 1: Execution trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(trace));
    tree_builder.commit(channel);

    // Build component — no LogUp, claimed_sum = 0
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        ElementwiseAddEval {
            log_n_rows: log_size,
        },
        SecureField::from(M31::from(0)),
    );

    let proof = prove::<B, MC>(&[&component], channel, commitment_scheme).map_err(|e| {
        ModelError::ProvingError {
            layer: 0,
            message: format!("ElementwiseAdd STARK: {e:?}"),
        }
    })?;

    Ok((component, proof))
}

/// Prove an element-wise Mul layer: `output[i] = lhs[i] * rhs[i]`.
///
/// Same structure as Add — pure AIR, no LogUp.
pub fn prove_elementwise_mul_layer<B, MC>(
    lhs: &[M31],
    rhs: &[M31],
    output: &[M31],
    config: PcsConfig,
) -> Result<
    (
        FrameworkComponent<ElementwiseMulEval>,
        StarkProof<<MC as MerkleChannel>::H>,
    ),
    ModelError,
>
where
    B: BackendForChannel<MC> + PolyOps + ColumnOps<BaseField>,
    <B as ColumnOps<BaseField>>::Column: 'static,
    MC: MerkleChannel,
    FrameworkComponent<ElementwiseMulEval>: stwo::prover::ComponentProver<B>,
{
    let n = lhs.len().max(rhs.len()).max(output.len()).max(16);
    let log_size = (n as f64).log2().ceil() as u32;
    let log_size = log_size.max(4);
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    let mut lhs_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rhs_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut out_col = Col::<SimdBackend, BaseField>::zeros(size);

    for (i, (&l, &r)) in lhs.iter().zip(rhs.iter()).enumerate().take(size) {
        lhs_col.set(i, l);
        rhs_col.set(i, r);
    }
    for (i, &o) in output.iter().enumerate().take(size) {
        out_col.set(i, o);
    }

    let trace = vec![
        CircleEvaluation::new(domain, lhs_col),
        CircleEvaluation::new(domain, rhs_col),
        CircleEvaluation::new(domain, out_col),
    ];

    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::new(config, &twiddles);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(vec![]));
    tree_builder.commit(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(trace));
    tree_builder.commit(channel);

    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        ElementwiseMulEval {
            log_n_rows: log_size,
        },
        SecureField::from(M31::from(0)),
    );

    let proof = prove::<B, MC>(&[&component], channel, commitment_scheme).map_err(|e| {
        ModelError::ProvingError {
            layer: 0,
            message: format!("ElementwiseMul STARK: {e:?}"),
        }
    })?;

    Ok((component, proof))
}

/// Prove a LayerNorm layer using LogUp for rsqrt lookup verification.
///
/// Three Merkle trees:
/// - Tree 0: Preprocessed rsqrt table (2 columns: var_input, rsqrt_output)
/// - Tree 1: Execution trace (6 columns: input, mean, variance, rsqrt_val, output, multiplicity)
/// - Tree 2: LogUp interaction trace
///
/// The AIR constraint verifies `output = (input - mean) * rsqrt_val` and the
/// LogUp protocol verifies every `(variance, rsqrt_val)` pair exists in the rsqrt table.
pub fn prove_layernorm_layer<B, MC>(
    inputs: &[M31],
    means: &[M31],
    variances: &[M31],
    rsqrt_vals: &[M31],
    outputs: &[M31],
    rsqrt_table: &PrecomputedTable,
    config: PcsConfig,
) -> Result<
    (
        FrameworkComponent<LayerNormEval>,
        StarkProof<<MC as MerkleChannel>::H>,
    ),
    ModelError,
>
where
    B: BackendForChannel<MC> + PolyOps + ColumnOps<BaseField>,
    <B as ColumnOps<BaseField>>::Column: 'static,
    MC: MerkleChannel,
    FrameworkComponent<LayerNormEval>: stwo::prover::ComponentProver<B>,
{
    let log_size = rsqrt_table.log_size.max(4);
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Pad traces with identity values for rows beyond real data
    let pad_val = M31::from(0);
    let pad_rsqrt = rsqrt_table.outputs.first().copied().unwrap_or(M31::from(0));

    // Compute multiplicities: count how many times each variance appears in the table
    let multiplicities = compute_multiplicities(variances, rsqrt_table);

    // Build preprocessed columns (rsqrt table)
    let mut table_var_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut table_rsqrt_col = Col::<SimdBackend, BaseField>::zeros(size);
    for (i, (&inp, &out)) in rsqrt_table
        .inputs
        .iter()
        .zip(rsqrt_table.outputs.iter())
        .enumerate()
        .take(size)
    {
        table_var_col.set(i, inp);
        table_rsqrt_col.set(i, out);
    }

    let preprocessed = vec![
        CircleEvaluation::new(domain, table_var_col.clone()),
        CircleEvaluation::new(domain, table_rsqrt_col.clone()),
    ];

    // Build execution trace columns (6 columns)
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
    // Pad remaining rows
    for i in n..size {
        input_col.set(i, pad_val);
        mean_col.set(i, pad_val);
        var_col.set(i, rsqrt_table.inputs.first().copied().unwrap_or(pad_val));
        rsqrt_col.set(i, pad_rsqrt);
        output_col.set(i, pad_val);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    let execution = vec![
        CircleEvaluation::new(domain, input_col),
        CircleEvaluation::new(domain, mean_col),
        CircleEvaluation::new(domain, var_col.clone()),
        CircleEvaluation::new(domain, rsqrt_col.clone()),
        CircleEvaluation::new(domain, output_col),
        CircleEvaluation::new(domain, mult_col),
    ];

    // Commit
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::new(config, &twiddles);

    // Tree 0: Preprocessed rsqrt table
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(
        preprocessed,
    ));
    tree_builder.commit(channel);

    // Tree 1: Execution trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(execution));
    tree_builder.commit(channel);

    // Draw lookup elements from channel
    let lookup_elements: LayerNormRelation = LayerNormRelation::draw(channel);

    // Compute LogUp interaction trace
    use stwo::prover::backend::simd::m31::LOG_N_LANES;
    let vec_size = size >> LOG_N_LANES;

    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let mut col_gen = logup_gen.new_col();

    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements
            .lookup_elements()
            .combine(&[table_var_col.data[vec_row], table_rsqrt_col.data[vec_row]]);
        let q_trace: PackedSecureField = lookup_elements
            .lookup_elements()
            .combine(&[var_col.data[vec_row], rsqrt_col.data[vec_row]]);

        let mult_packed = mult_col_data_at(&multiplicities, vec_row, log_size);

        let numerator = q_table - mult_packed * q_trace;
        let denominator = q_table * q_trace;

        col_gen.write_frac(vec_row, numerator, denominator);
    }
    col_gen.finalize_col();

    let (interaction_trace, claimed_sum) = logup_gen.finalize_last();

    // Tree 2: LogUp interaction trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(
        interaction_trace,
    ));
    tree_builder.commit(channel);

    // Build component and prove
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        LayerNormEval {
            log_n_rows: log_size,
            dim: inputs.len(),
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );

    let proof = prove::<B, MC>(&[&component], channel, commitment_scheme).map_err(|e| {
        ModelError::ProvingError {
            layer: 0,
            message: format!("LayerNorm STARK: {e:?}"),
        }
    })?;

    Ok((component, proof))
}

/// Prove an RMSNorm layer via LogUp STARK.
///
/// Same structure as `prove_layernorm_layer` but with 5 trace columns (no mean).
pub fn prove_rmsnorm_layer<B, MC>(
    inputs: &[M31],
    rms_sq_vals: &[M31],
    rsqrt_vals: &[M31],
    outputs: &[M31],
    rsqrt_table: &PrecomputedTable,
    config: PcsConfig,
) -> Result<
    (
        FrameworkComponent<crate::components::rmsnorm::RMSNormEval>,
        StarkProof<<MC as MerkleChannel>::H>,
    ),
    ModelError,
>
where
    B: BackendForChannel<MC> + PolyOps + ColumnOps<BaseField>,
    <B as ColumnOps<BaseField>>::Column: 'static,
    MC: MerkleChannel,
    FrameworkComponent<crate::components::rmsnorm::RMSNormEval>: stwo::prover::ComponentProver<B>,
{
    use crate::components::rmsnorm::{RMSNormEval, RMSNormRelation};

    let log_size = rsqrt_table.log_size.max(4);
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    let pad_val = M31::from(0);
    let pad_rsqrt = rsqrt_table.outputs.first().copied().unwrap_or(M31::from(0));

    let multiplicities = compute_multiplicities(rms_sq_vals, rsqrt_table);

    // Preprocessed columns (rsqrt table)
    let mut table_rms_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut table_rsqrt_col = Col::<SimdBackend, BaseField>::zeros(size);
    for (i, (&inp, &out)) in rsqrt_table
        .inputs
        .iter()
        .zip(rsqrt_table.outputs.iter())
        .enumerate()
        .take(size)
    {
        table_rms_col.set(i, inp);
        table_rsqrt_col.set(i, out);
    }

    let preprocessed = vec![
        CircleEvaluation::new(domain, table_rms_col.clone()),
        CircleEvaluation::new(domain, table_rsqrt_col.clone()),
    ];

    // Execution trace (5 columns: input, rms_sq, rsqrt, output, multiplicity)
    let mut input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rms_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rsqrt_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);

    let n = inputs.len().min(size);
    for i in 0..n {
        input_col.set(i, inputs[i]);
        rms_col.set(i, rms_sq_vals[i]);
        rsqrt_col.set(i, rsqrt_vals[i]);
        output_col.set(i, outputs[i]);
    }
    for i in n..size {
        input_col.set(i, pad_val);
        rms_col.set(i, rsqrt_table.inputs.first().copied().unwrap_or(pad_val));
        rsqrt_col.set(i, pad_rsqrt);
        output_col.set(i, pad_val);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    let execution = vec![
        CircleEvaluation::new(domain, input_col),
        CircleEvaluation::new(domain, rms_col.clone()),
        CircleEvaluation::new(domain, rsqrt_col.clone()),
        CircleEvaluation::new(domain, output_col),
        CircleEvaluation::new(domain, mult_col),
    ];

    // Commit
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::new(config, &twiddles);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(
        preprocessed,
    ));
    tree_builder.commit(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(execution));
    tree_builder.commit(channel);

    let lookup_elements: RMSNormRelation = RMSNormRelation::draw(channel);

    // LogUp interaction trace
    use stwo::prover::backend::simd::m31::LOG_N_LANES;
    let vec_size = size >> LOG_N_LANES;

    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let mut col_gen = logup_gen.new_col();

    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements
            .lookup_elements()
            .combine(&[table_rms_col.data[vec_row], table_rsqrt_col.data[vec_row]]);
        let q_trace: PackedSecureField = lookup_elements
            .lookup_elements()
            .combine(&[rms_col.data[vec_row], rsqrt_col.data[vec_row]]);

        let mult_packed = mult_col_data_at(&multiplicities, vec_row, log_size);

        let numerator = q_table - mult_packed * q_trace;
        let denominator = q_table * q_trace;

        col_gen.write_frac(vec_row, numerator, denominator);
    }
    col_gen.finalize_col();

    let (interaction_trace, claimed_sum) = logup_gen.finalize_last();

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(
        interaction_trace,
    ));
    tree_builder.commit(channel);

    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        RMSNormEval {
            log_n_rows: log_size,
            dim: inputs.len(),
            lookup_elements,
            claimed_sum,
        },
        claimed_sum,
    );

    let proof = prove::<B, MC>(&[&component], channel, commitment_scheme).map_err(|e| {
        ModelError::ProvingError {
            layer: 0,
            message: format!("RMSNorm STARK: {e:?}"),
        }
    })?;

    Ok((component, proof))
}

/// Element-wise addition of two matrices.
/// Parallelized with rayon for arrays >= 4096 elements.
pub fn elementwise_add(lhs: &M31Matrix, rhs: &M31Matrix) -> M31Matrix {
    use rayon::prelude::*;

    let rows = lhs.rows.max(rhs.rows);
    let cols = lhs.cols.max(rhs.cols);
    let len = lhs.data.len().min(rhs.data.len());

    let mut data = if len >= 4096 {
        lhs.data[..len]
            .par_iter()
            .zip(rhs.data[..len].par_iter())
            .map(|(&a, &b)| a + b)
            .collect::<Vec<_>>()
    } else {
        (0..len)
            .map(|i| lhs.data[i] + rhs.data[i])
            .collect::<Vec<_>>()
    };

    // Pad to full output size + copy remaining from lhs
    data.resize(rows * cols, M31::from(0));
    for i in len..lhs.data.len().min(data.len()) {
        data[i] = lhs.data[i];
    }

    M31Matrix { rows, cols, data }
}

/// Element-wise multiplication of two matrices.
/// Parallelized with rayon for arrays >= 4096 elements.
pub fn elementwise_mul(lhs: &M31Matrix, rhs: &M31Matrix) -> M31Matrix {
    use rayon::prelude::*;

    let rows = lhs.rows.max(rhs.rows);
    let cols = lhs.cols.max(rhs.cols);
    let len = lhs.data.len().min(rhs.data.len());

    let mut data = if len >= 4096 {
        lhs.data[..len]
            .par_iter()
            .zip(rhs.data[..len].par_iter())
            .map(|(&a, &b)| a * b)
            .collect::<Vec<_>>()
    } else {
        (0..len)
            .map(|i| lhs.data[i] * rhs.data[i])
            .collect::<Vec<_>>()
    };

    data.resize(rows * cols, M31::from(0));
    M31Matrix { rows, cols, data }
}

/// Apply an activation function element-wise to a matrix (public for aggregation).
pub fn apply_activation_pub(input: &M31Matrix, f: &(dyn Fn(M31) -> M31 + Sync)) -> M31Matrix {
    apply_activation(input, f)
}

/// Apply LayerNorm (public for aggregation).
pub fn apply_layernorm_pub(input: &M31Matrix, dim: usize) -> M31Matrix {
    apply_layernorm(input, dim)
}

/// Apply an activation function element-wise to a matrix.
/// Parallelized with rayon for arrays >= 4096 elements.
fn apply_activation(input: &M31Matrix, f: &(dyn Fn(M31) -> M31 + Sync)) -> M31Matrix {
    use rayon::prelude::*;

    let data = if input.data.len() >= 4096 {
        input.data.par_iter().map(|&v| f(v)).collect()
    } else {
        input.data.iter().map(|&v| f(v)).collect()
    };

    M31Matrix {
        rows: input.rows,
        cols: input.cols,
        data,
    }
}

/// LayerNorm intermediates for proving.
pub(crate) struct LayerNormIntermediates {
    pub inputs: Vec<M31>,
    pub means: Vec<M31>,
    pub variances: Vec<M31>,
    pub rsqrt_vals: Vec<M31>,
    pub outputs: Vec<M31>,
    pub output_matrix: M31Matrix,
}

/// Compute LayerNorm forward pass in M31 field arithmetic, returning intermediates for proving.
///
/// y = (x - mean) * rsqrt(variance)
///
/// Division by n uses modular inverse (Fermat's little theorem).
/// Reciprocal sqrt looked up in precomputed table.
pub(crate) fn apply_layernorm_detailed(input: &M31Matrix, dim: usize) -> LayerNormIntermediates {
    use rayon::prelude::*;

    let rsqrt_table = build_rsqrt_table(LayerNormConfig::new(dim).rsqrt_table_log_size);
    let n = dim.min(input.cols);
    let inv_n = m31_mod_inverse(n as u32);
    let cols = input.cols;

    // Each row is independent: compute per-row results in parallel, then flatten.
    let process_row = |row: usize| -> (Vec<M31>, Vec<M31>, Vec<M31>, Vec<M31>, Vec<M31>, Vec<M31>) {
        let row_start = row * cols;

        // Mean: sum(x) / n
        let mut sum = M31::from(0);
        for col in 0..n {
            sum += input.data[row_start + col];
        }
        let mean = sum * inv_n;

        // Variance: sum((x - mean)^2) / n
        let mut var_sum = M31::from(0);
        for col in 0..n {
            let diff = input.data[row_start + col] - mean;
            var_sum += diff * diff;
        }
        // Reduce variance to rsqrt_table range [0, 2^16) so the LogUp
        // lookup always succeeds.  M31 modular arithmetic can produce
        // variance values anywhere in [0, P-1]; masking to 16 bits keeps
        // the computation deterministic and provable.
        let variance_raw = var_sum * inv_n;
        let variance = M31::from(variance_raw.0 & ((1u32 << 16) - 1));

        let rsqrt = rsqrt_table
            .lookup(variance)
            .expect("variance reduced to table range; lookup must succeed");

        let mut row_out = Vec::with_capacity(cols);
        let mut row_inputs = Vec::with_capacity(cols);
        let mut row_means = Vec::with_capacity(cols);
        let mut row_vars = Vec::with_capacity(cols);
        let mut row_rsqrt = Vec::with_capacity(cols);
        let mut row_outputs = Vec::with_capacity(cols);

        for col in 0..n {
            let x = input.data[row_start + col];
            let centered = x - mean;
            let out_val = centered * rsqrt;
            row_out.push(out_val);
            row_inputs.push(x);
            row_means.push(mean);
            row_vars.push(variance);
            row_rsqrt.push(rsqrt);
            row_outputs.push(out_val);
        }
        for col in n..cols {
            let x = input.data[row_start + col];
            row_out.push(x);
            row_inputs.push(x);
            row_means.push(M31::from(0));
            row_vars.push(M31::from(0));
            row_rsqrt.push(M31::from(1u32 << 16));
            row_outputs.push(x);
        }

        (
            row_out,
            row_inputs,
            row_means,
            row_vars,
            row_rsqrt,
            row_outputs,
        )
    };

    let row_results: Vec<_> = if input.rows >= 64 {
        (0..input.rows).into_par_iter().map(process_row).collect()
    } else {
        (0..input.rows).map(process_row).collect()
    };

    let total = input.rows * cols;
    let mut out_data = Vec::with_capacity(total);
    let mut all_inputs = Vec::with_capacity(total);
    let mut all_means = Vec::with_capacity(total);
    let mut all_variances = Vec::with_capacity(total);
    let mut all_rsqrt = Vec::with_capacity(total);
    let mut all_outputs = Vec::with_capacity(total);

    for (row_out, ri, rm, rv, rr, ro) in row_results {
        out_data.extend(row_out);
        all_inputs.extend(ri);
        all_means.extend(rm);
        all_variances.extend(rv);
        all_rsqrt.extend(rr);
        all_outputs.extend(ro);
    }

    LayerNormIntermediates {
        inputs: all_inputs,
        means: all_means,
        variances: all_variances,
        rsqrt_vals: all_rsqrt,
        outputs: all_outputs,
        output_matrix: M31Matrix {
            rows: input.rows,
            cols,
            data: out_data,
        },
    }
}

/// Compute LayerNorm forward pass (output matrix only).
fn apply_layernorm(input: &M31Matrix, dim: usize) -> M31Matrix {
    apply_layernorm_detailed(input, dim).output_matrix
}

/// RMSNorm intermediates for proving.
pub(crate) struct RMSNormIntermediates {
    pub inputs: Vec<M31>,
    pub rms_sq_vals: Vec<M31>,
    pub rsqrt_vals: Vec<M31>,
    pub outputs: Vec<M31>,
    pub output_matrix: M31Matrix,
}

/// Compute RMSNorm forward pass in M31 field arithmetic, returning intermediates for proving.
///
/// y = x × rsqrt(mean(x²))
///
/// Unlike LayerNorm, there is no mean subtraction. Only:
/// 1. rms² = sum(x²) / n
/// 2. rsqrt = lookup_table(rms²)
/// 3. output = input × rsqrt
pub(crate) fn apply_rmsnorm_detailed(input: &M31Matrix, dim: usize) -> RMSNormIntermediates {
    use crate::components::rmsnorm::{build_rsqrt_table, RMSNormConfig};
    use rayon::prelude::*;

    let rsqrt_table = build_rsqrt_table(RMSNormConfig::new(dim).rsqrt_table_log_size);
    let n = dim.min(input.cols);
    let inv_n = m31_mod_inverse(n as u32);
    let cols = input.cols;

    let process_row = |row: usize| -> (Vec<M31>, Vec<M31>, Vec<M31>, Vec<M31>, Vec<M31>) {
        let row_start = row * cols;

        // RMS²: sum(x²) / n
        let mut sq_sum = M31::from(0);
        for col in 0..n {
            let x = input.data[row_start + col];
            sq_sum += x * x;
        }
        // Reduce rms_sq to rsqrt_table range [0, 2^16) for provable LogUp.
        let rms_sq_raw = sq_sum * inv_n;
        let rms_sq = M31::from(rms_sq_raw.0 & ((1u32 << 16) - 1));

        let rsqrt = rsqrt_table
            .lookup(rms_sq)
            .expect("rms_sq reduced to table range; lookup must succeed");

        let mut row_out = Vec::with_capacity(cols);
        let mut row_inputs = Vec::with_capacity(cols);
        let mut row_rms = Vec::with_capacity(cols);
        let mut row_rsqrt = Vec::with_capacity(cols);
        let mut row_outputs = Vec::with_capacity(cols);

        for col in 0..n {
            let x = input.data[row_start + col];
            let out_val = x * rsqrt;
            row_out.push(out_val);
            row_inputs.push(x);
            row_rms.push(rms_sq);
            row_rsqrt.push(rsqrt);
            row_outputs.push(out_val);
        }
        for col in n..cols {
            let x = input.data[row_start + col];
            row_out.push(x);
            row_inputs.push(x);
            row_rms.push(M31::from(0));
            row_rsqrt.push(M31::from(1u32 << 16));
            row_outputs.push(x);
        }

        (row_out, row_inputs, row_rms, row_rsqrt, row_outputs)
    };

    let row_results: Vec<_> = if input.rows >= 64 {
        (0..input.rows).into_par_iter().map(process_row).collect()
    } else {
        (0..input.rows).map(process_row).collect()
    };

    let total = input.rows * cols;
    let mut out_data = Vec::with_capacity(total);
    let mut all_inputs = Vec::with_capacity(total);
    let mut all_rms_sq = Vec::with_capacity(total);
    let mut all_rsqrt = Vec::with_capacity(total);
    let mut all_outputs = Vec::with_capacity(total);

    for (row_out, ri, rr, rs, ro) in row_results {
        out_data.extend(row_out);
        all_inputs.extend(ri);
        all_rms_sq.extend(rr);
        all_rsqrt.extend(rs);
        all_outputs.extend(ro);
    }

    RMSNormIntermediates {
        inputs: all_inputs,
        rms_sq_vals: all_rms_sq,
        rsqrt_vals: all_rsqrt,
        outputs: all_outputs,
        output_matrix: M31Matrix {
            rows: input.rows,
            cols,
            data: out_data,
        },
    }
}

/// Modular inverse of n in M31 via Fermat's little theorem: n^(P-2) mod P.
fn m31_mod_inverse(n: u32) -> M31 {
    if n == 0 {
        return M31::from(0);
    }
    let p: u64 = (1u64 << 31) - 1;
    let mut result: u64 = 1;
    let mut base = n as u64 % p;
    let mut exp = p - 2;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % p;
        }
        base = base * base % p;
        exp >>= 1;
    }
    M31::from(result as u32)
}

/// Prove an entire computation graph, generic over backend and Merkle channel.
///
/// Executes the forward pass and generates real STARK proofs (activations)
/// and sumcheck proofs (matmuls). Trace generation uses `SimdBackend`
/// internally; commitment and proving use backend `B`, enabling GPU
/// acceleration of Merkle hashing, FRI, and quotient evaluation.
pub fn prove_model_with<B, MC>(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<ModelProofResultFor<<MC as MerkleChannel>::H>, ModelError>
where
    B: BackendForChannel<MC> + PolyOps + ColumnOps<BaseField>,
    <B as ColumnOps<BaseField>>::Column: 'static,
    MC: MerkleChannel,
    FrameworkComponent<ActivationEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<ElementwiseAddEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<ElementwiseMulEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<LayerNormEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<crate::components::rmsnorm::RMSNormEval>: stwo::prover::ComponentProver<B>,
{
    info!(
        backend = std::any::type_name::<B>(),
        channel = std::any::type_name::<MC>(),
        num_nodes = graph.nodes.len(),
        "Proving model per-layer"
    );
    let mut layer_proofs = Vec::new();
    let mut intermediates: Vec<(usize, M31Matrix)> = Vec::new();

    // Map from node ID to its output, for resolving multi-input nodes
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();

    // Current activation flowing through the network (for sequential chains)
    let mut current = input.clone();

    // Process nodes in topological order
    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];

        // Resolve current input: use first input from node_outputs if available
        if let Some(&first_input) = node.inputs.first() {
            if let Some(inp) = node_outputs.get(&first_input) {
                current = inp.clone();
            }
        }

        match &node.op {
            GraphOp::MatMul { dims } => {
                let (_m, _k, _n) = *dims;

                // Get weight matrix for this layer
                let weight = weights
                    .get_weight(node.id)
                    .ok_or(ModelError::MissingWeight(node.id))?;

                // Forward pass: C = current × weight
                let output = matmul_m31(&current, weight);

                // Generate sumcheck proof
                let proof = prove_matmul_sumcheck_auto(&current, weight, &output).map_err(|e| {
                    ModelError::ProvingError {
                        layer: node.id,
                        message: format!("MatMul sumcheck: {e}"),
                    }
                })?;

                let claimed_sum = proof.claimed_sum;
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;

                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Sumcheck(proof),
                    claimed_sum,
                    layer_index: node.id,
                });
            }

            GraphOp::Activation {
                activation_type,
                size: _,
            } => {
                let f = activation_type.as_fn();
                let output = apply_activation(&current, &*f);

                // Build lookup table — use production size
                let log_size = activation_type.recommended_table_log_size();
                let table = PrecomputedTable::build(|x| (*f)(x), log_size);
                let config = PcsConfig::default();

                // Flatten current matrix to input/output slices
                let flat_inputs: Vec<M31> = current.data.clone();
                let flat_outputs: Vec<M31> = output.data.clone();

                // Generate real STARK proof via LogUp
                let (_component, proof) =
                    prove_activation_layer::<B, MC>(&flat_inputs, &flat_outputs, &table, config)
                        .map_err(|e| ModelError::ProvingError {
                            layer: node.id,
                            message: format!("Activation STARK: {e}"),
                        })?;

                let claimed_sum = SecureField::from(M31::from(0));
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;

                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Stark(proof),
                    claimed_sum,
                    layer_index: node.id,
                });
            }

            GraphOp::LayerNorm { dim } => {
                let ln = apply_layernorm_detailed(&current, *dim);
                let rsqrt_table =
                    build_rsqrt_table(LayerNormConfig::new(*dim).rsqrt_table_log_size);
                let config = PcsConfig::default();

                let (_component, proof) = prove_layernorm_layer::<B, MC>(
                    &ln.inputs,
                    &ln.means,
                    &ln.variances,
                    &ln.rsqrt_vals,
                    &ln.outputs,
                    &rsqrt_table,
                    config,
                )
                .map_err(|e| ModelError::ProvingError {
                    layer: node.id,
                    message: format!("LayerNorm STARK: {e}"),
                })?;

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, ln.output_matrix.clone());
                current = ln.output_matrix;

                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::LayerNorm(proof),
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }

            GraphOp::RMSNorm { dim } => {
                let rn = apply_rmsnorm_detailed(&current, *dim);
                let rsqrt_table = crate::components::rmsnorm::build_rsqrt_table(
                    crate::components::rmsnorm::RMSNormConfig::new(*dim).rsqrt_table_log_size,
                );
                let config = PcsConfig::default();

                let (_component, proof) = prove_rmsnorm_layer::<B, MC>(
                    &rn.inputs,
                    &rn.rms_sq_vals,
                    &rn.rsqrt_vals,
                    &rn.outputs,
                    &rsqrt_table,
                    config,
                )
                .map_err(|e| ModelError::ProvingError {
                    layer: node.id,
                    message: format!("RMSNorm STARK: {e}"),
                })?;

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, rn.output_matrix.clone());
                current = rn.output_matrix;

                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::RMSNorm(proof),
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }

            GraphOp::Add { size: _ } => {
                // Resolve both inputs
                let lhs = node
                    .inputs
                    .get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node
                    .inputs
                    .get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());

                // Forward pass: element-wise addition
                let output = elementwise_add(&lhs, &rhs);

                // Generate real STARK proof
                let config = PcsConfig::default();
                let (_component, proof) = prove_elementwise_add_layer::<B, MC>(
                    &lhs.data,
                    &rhs.data,
                    &output.data,
                    config,
                )
                .map_err(|e| ModelError::ProvingError {
                    layer: node.id,
                    message: format!("Add STARK: {e}"),
                })?;

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;

                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::ElementwiseAdd(proof),
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }

            GraphOp::Mul { size: _ } => {
                // Resolve both inputs
                let lhs = node
                    .inputs
                    .get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node
                    .inputs
                    .get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());

                // Forward pass: element-wise multiplication
                let output = elementwise_mul(&lhs, &rhs);

                // Generate real STARK proof
                let config = PcsConfig::default();
                let (_component, proof) = prove_elementwise_mul_layer::<B, MC>(
                    &lhs.data,
                    &rhs.data,
                    &output.data,
                    config,
                )
                .map_err(|e| ModelError::ProvingError {
                    layer: node.id,
                    message: format!("Mul STARK: {e}"),
                })?;

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;

                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::ElementwiseMul(proof),
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }

            GraphOp::Attention { config } => {
                // Attempt to extract attention weights and run real prover
                let w_q = weights.get_named_weight(node.id, "w_q");
                let w_k = weights.get_named_weight(node.id, "w_k");
                let w_v = weights.get_named_weight(node.id, "w_v");
                let w_o = weights.get_named_weight(node.id, "w_o");

                if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (w_q, w_k, w_v, w_o) {
                    use crate::components::attention::{prove_attention_with, AttentionWeights};

                    let attn_weights = AttentionWeights {
                        w_q: wq.clone(),
                        w_k: wk.clone(),
                        w_v: wv.clone(),
                        w_o: wo.clone(),
                    };

                    let attn_proof =
                        prove_attention_with::<B, MC>(&current, &attn_weights, config, false)
                            .map_err(|e| ModelError::ProvingError {
                                layer: node.id,
                                message: format!("Attention: {e}"),
                            })?;

                    let output = attn_proof.intermediates.final_output.clone();
                    intermediates.push((node.id, current.clone()));
                    node_outputs.insert(node.id, output.clone());
                    current = output;

                    // Use the softmax exp STARK as the layer proof
                    layer_proofs.push(LayerProof {
                        kind: LayerProofKind::Stark(attn_proof.softmax_exp_proof),
                        claimed_sum: SecureField::from(M31::from(0)),
                        layer_index: node.id,
                    });
                } else {
                    // No named weights available — passthrough
                    intermediates.push((node.id, current.clone()));
                    node_outputs.insert(node.id, current.clone());
                    layer_proofs.push(LayerProof {
                        kind: LayerProofKind::Passthrough,
                        claimed_sum: SecureField::from(M31::from(0)),
                        layer_index: node.id,
                    });
                }
            }

            GraphOp::Embedding { .. } | GraphOp::Conv2D { .. } => {
                // Forward pass only — full proving deferred to aggregation
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, current.clone());
                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Passthrough,
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }

            GraphOp::Quantize { .. }
            | GraphOp::Dequantize { .. }
            | GraphOp::Identity { .. }
            | GraphOp::RoPE { .. } => {
                // Quantize/Dequantize: proven in aggregation via LogUp.
                // RoPE: proven in aggregation via LogUp rotation table.
                // Identity: no-op by definition.
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, current.clone());
                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Passthrough,
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }
        }
    }

    let execution = GraphExecution {
        intermediates,
        output: current,
    };

    Ok((layer_proofs, execution))
}

/// Prove an entire computation graph with streaming intermediates.
///
/// Like [`prove_model_with`] but frees intermediate activations as soon as
/// all downstream nodes have consumed them. This reduces peak memory from
/// O(total_activations) to O(max_layer_width), enabling larger models.
///
/// The tradeoff is that verification cannot be done after proving since
/// intermediates are freed. Use this when memory is the constraint and
/// you only need the proofs and final output.
pub fn prove_model_streaming(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<ModelProofResult, ModelError> {
    prove_model_streaming_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
}

/// Generic streaming prover.
pub fn prove_model_streaming_with<B, MC>(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<ModelProofResultFor<<MC as MerkleChannel>::H>, ModelError>
where
    B: BackendForChannel<MC> + PolyOps + ColumnOps<BaseField>,
    <B as ColumnOps<BaseField>>::Column: 'static,
    MC: MerkleChannel,
    FrameworkComponent<ActivationEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<ElementwiseAddEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<ElementwiseMulEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<LayerNormEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<crate::components::rmsnorm::RMSNormEval>: stwo::prover::ComponentProver<B>,
{
    info!(
        backend = std::any::type_name::<B>(),
        num_nodes = graph.nodes.len(),
        "Proving model with streaming intermediates"
    );
    let mut layer_proofs = Vec::new();

    // Track node outputs and their remaining consumer count
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();
    let mut consumer_count: HashMap<usize, usize> = HashMap::new();

    // Count how many downstream nodes consume each node's output
    for node in &graph.nodes {
        for &inp in &node.inputs {
            *consumer_count.entry(inp).or_insert(0) += 1;
        }
    }

    let mut current = input.clone();
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
            GraphOp::MatMul { dims: _ } => {
                let weight = weights
                    .get_weight(node.id)
                    .ok_or(ModelError::MissingWeight(node.id))?;
                let output = matmul_m31(&current, weight);
                let proof = prove_matmul_sumcheck_auto(&current, weight, &output).map_err(|e| {
                    ModelError::ProvingError {
                        layer: node.id,
                        message: format!("MatMul sumcheck: {e}"),
                    }
                })?;
                let claimed_sum = proof.claimed_sum;
                node_outputs.insert(node.id, output.clone());
                current = output;
                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Sumcheck(proof),
                    claimed_sum,
                    layer_index: node.id,
                });
            }
            GraphOp::Activation {
                activation_type,
                size: _,
            } => {
                let f = activation_type.as_fn();
                let output = apply_activation(&current, &*f);
                let log_size = activation_type.recommended_table_log_size();
                let table = PrecomputedTable::build(|x| (*f)(x), log_size);
                let config = PcsConfig::default();
                let flat_inputs: Vec<M31> = current.data.clone();
                let flat_outputs: Vec<M31> = output.data.clone();
                let (_component, proof) =
                    prove_activation_layer::<B, MC>(&flat_inputs, &flat_outputs, &table, config)
                        .map_err(|e| ModelError::ProvingError {
                            layer: node.id,
                            message: format!("Activation STARK: {e}"),
                        })?;
                node_outputs.insert(node.id, output.clone());
                current = output;
                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Stark(proof),
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }
            GraphOp::Add { .. } => {
                let lhs = node
                    .inputs
                    .get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node
                    .inputs
                    .get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_add(&lhs, &rhs);
                let config = PcsConfig::default();
                let (_component, proof) = prove_elementwise_add_layer::<B, MC>(
                    &lhs.data,
                    &rhs.data,
                    &output.data,
                    config,
                )
                .map_err(|e| ModelError::ProvingError {
                    layer: node.id,
                    message: format!("Add STARK: {e}"),
                })?;
                node_outputs.insert(node.id, output.clone());
                current = output;
                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::ElementwiseAdd(proof),
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }
            GraphOp::Mul { .. } => {
                let lhs = node
                    .inputs
                    .get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node
                    .inputs
                    .get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_mul(&lhs, &rhs);
                let config = PcsConfig::default();
                let (_component, proof) = prove_elementwise_mul_layer::<B, MC>(
                    &lhs.data,
                    &rhs.data,
                    &output.data,
                    config,
                )
                .map_err(|e| ModelError::ProvingError {
                    layer: node.id,
                    message: format!("Mul STARK: {e}"),
                })?;
                node_outputs.insert(node.id, output.clone());
                current = output;
                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::ElementwiseMul(proof),
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }
            GraphOp::LayerNorm { dim } => {
                let ln = apply_layernorm_detailed(&current, *dim);
                let rsqrt_table =
                    build_rsqrt_table(LayerNormConfig::new(*dim).rsqrt_table_log_size);
                let config = PcsConfig::default();
                let (_component, proof) = prove_layernorm_layer::<B, MC>(
                    &ln.inputs,
                    &ln.means,
                    &ln.variances,
                    &ln.rsqrt_vals,
                    &ln.outputs,
                    &rsqrt_table,
                    config,
                )
                .map_err(|e| ModelError::ProvingError {
                    layer: node.id,
                    message: format!("LayerNorm STARK: {e}"),
                })?;
                node_outputs.insert(node.id, ln.output_matrix.clone());
                current = ln.output_matrix;
                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::LayerNorm(proof),
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }
            GraphOp::RMSNorm { dim } => {
                let rn = apply_rmsnorm_detailed(&current, *dim);
                let rsqrt_table = crate::components::rmsnorm::build_rsqrt_table(
                    crate::components::rmsnorm::RMSNormConfig::new(*dim).rsqrt_table_log_size,
                );
                let config = PcsConfig::default();
                let (_component, proof) = prove_rmsnorm_layer::<B, MC>(
                    &rn.inputs,
                    &rn.rms_sq_vals,
                    &rn.rsqrt_vals,
                    &rn.outputs,
                    &rsqrt_table,
                    config,
                )
                .map_err(|e| ModelError::ProvingError {
                    layer: node.id,
                    message: format!("RMSNorm STARK: {e}"),
                })?;
                node_outputs.insert(node.id, rn.output_matrix.clone());
                current = rn.output_matrix;
                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::RMSNorm(proof),
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }
            _ => {
                // Attention, Embedding, Conv2D, Quantize, Identity
                node_outputs.insert(node.id, current.clone());
                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Passthrough,
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }
        }

        // Free intermediates that are no longer needed
        for &inp in &node.inputs {
            if let Some(count) = consumer_count.get_mut(&inp) {
                *count -= 1;
                if *count == 0 {
                    node_outputs.remove(&inp);
                }
            }
        }
    }

    let execution = GraphExecution {
        intermediates: Vec::new(), // freed during proving
        output: current,
    };

    Ok((layer_proofs, execution))
}

/// Prove an entire computation graph using `SimdBackend` + `Blake2sMerkleChannel`.
///
/// Convenience wrapper around [`prove_model_with`]. Use `prove_model_with::<GpuBackend, Blake2sMerkleChannel>`
/// for GPU-accelerated proving (requires `cuda-runtime` feature).
pub fn prove_model(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<ModelProofResult, ModelError> {
    prove_model_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
}

/// Verify all matmul sumcheck proofs in a model's layer proofs.
///
/// For each Sumcheck proof, replays the Fiat-Shamir transcript
/// and checks the final MLE evaluations against the original matrices.
/// Generic over the Merkle hash type `H`.
pub fn verify_model_matmuls<H: MerkleHasherLifted>(
    layer_proofs: &[LayerProof<H>],
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<(), ModelError> {
    use crate::components::matmul::verify_matmul_sumcheck;

    let mut current = input.clone();
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();

    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        let proof = layer_proofs.iter().find(|p| p.layer_index == node_id);

        // Resolve current input from first dependency
        if let Some(&first_input) = node.inputs.first() {
            if let Some(inp) = node_outputs.get(&first_input) {
                current = inp.clone();
            }
        }

        match &node.op {
            GraphOp::MatMul { .. } => {
                let weight = weights
                    .get_weight(node.id)
                    .ok_or(ModelError::MissingWeight(node.id))?;

                let c = matmul_m31(&current, weight);

                if let Some(p) = proof {
                    if let LayerProofKind::Sumcheck(matmul_proof) = &p.kind {
                        verify_matmul_sumcheck(matmul_proof, &current, weight, &c).map_err(
                            |e| ModelError::VerificationError {
                                layer: node.id,
                                message: format!("MatMul verification failed: {e}"),
                            },
                        )?;
                    }
                }

                node_outputs.insert(node.id, c.clone());
                current = c;
            }
            GraphOp::Activation {
                activation_type, ..
            } => {
                let f = activation_type.as_fn();
                let output = apply_activation(&current, &*f);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::LayerNorm { dim } => {
                let output = apply_layernorm(&current, *dim);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::RMSNorm { dim } => {
                let rn = apply_rmsnorm_detailed(&current, *dim);
                node_outputs.insert(node.id, rn.output_matrix.clone());
                current = rn.output_matrix;
            }
            GraphOp::Add { .. } => {
                let lhs = node
                    .inputs
                    .get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node
                    .inputs
                    .get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_add(&lhs, &rhs);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Mul { .. } => {
                let lhs = node
                    .inputs
                    .get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node
                    .inputs
                    .get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_mul(&lhs, &rhs);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Attention { .. } => {
                // Attention proofs verified separately; skip in matmul verifier.
                node_outputs.insert(node.id, current.clone());
            }
            _ => {
                node_outputs.insert(node.id, current.clone());
            }
        }
    }

    Ok(())
}

/// Prove a model using the best available backend.
///
/// If `cuda-runtime` feature is enabled and a GPU is detected, uses `GpuBackend`
/// for commitment and proving (Merkle trees, FRI, quotient evaluation on GPU).
/// Otherwise falls back to `SimdBackend`.
///
/// `GpuBackend` is a drop-in replacement — same column types, same twiddles.
/// Trace generation stays on SIMD; the GPU accelerates cryptographic operations.
pub fn prove_model_auto(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<ModelProofResult, ModelError> {
    let gpu_available = crate::backend::gpu_is_available();
    info!(
        gpu_available,
        "Auto-selecting backend for per-layer proving"
    );
    crate::backend::with_best_backend(
        || {
            info!("Using SimdBackend for per-layer proving");
            prove_model_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
        },
        || {
            info!("Using GpuBackend for per-layer proving");
            prove_model_gpu(graph, input, weights)
        },
    )
}

/// GPU proving path — dispatches to `GpuBackend` when `cuda-runtime` is enabled.
fn prove_model_gpu(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<ModelProofResult, ModelError> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return prove_model_with::<GpuBackend, Blake2sMerkleChannel>(graph, input, weights);
    }

    // Fallback — unreachable when with_best_backend routes here,
    // but needed for compilation without cuda-runtime.
    #[cfg(not(feature = "cuda-runtime"))]
    {
        prove_model_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use crate::components::activation::ActivationType;

    #[test]
    fn test_prove_model_matmul_only() {
        // 1×4 input × 4×2 weight = 1×2 output
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let (proofs, execution) =
            prove_model(&graph, &input, &weights).expect("model proving should succeed");

        assert_eq!(proofs.len(), 1);
        assert!(matches!(proofs[0].kind, LayerProofKind::Sumcheck(_)));
        assert_eq!(execution.output.rows, 1);
        assert_eq!(execution.output.cols, 2);
    }

    #[test]
    fn test_prove_model_mlp_matmul_activation_matmul() {
        // 2-layer MLP: input(1×4) → linear(4→4) → ReLU → linear(4→2)
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();

        // Layer 0: 4×4 weight
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        // Layer 2: 4×2 weight (node 1 is activation, node 2 is second linear)
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let (proofs, execution) =
            prove_model(&graph, &input, &weights).expect("MLP proving should succeed");

        assert_eq!(proofs.len(), 3, "3 layers: matmul, activation, matmul");
        assert!(
            matches!(proofs[0].kind, LayerProofKind::Sumcheck(_)),
            "layer 0 = matmul"
        );
        assert!(
            matches!(proofs[1].kind, LayerProofKind::Stark(_)),
            "layer 1 = activation (STARK proof)"
        );
        assert!(
            matches!(proofs[2].kind, LayerProofKind::Sumcheck(_)),
            "layer 2 = matmul"
        );
        assert_eq!(execution.output.rows, 1);
        assert_eq!(execution.output.cols, 2);
    }

    /// The "not looking like fools" test.
    ///
    /// Builds a 3-layer MLP, executes the forward pass in M31 arithmetic,
    /// generates real sumcheck proofs for every matmul, and verifies them.
    #[test]
    fn test_end_to_end_mlp_prove_and_verify() {
        // Architecture: 1×4 → Linear(4→4) → ReLU → Linear(4→4) → ReLU → Linear(4→2)
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
        let graph = builder.build();

        // Input: [1, 2, 3, 4]
        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        // Weights for each linear layer
        let mut weights = GraphWeights::new();

        // Layer 0 (node 0): 4×4 weight
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i * 4 + j) % 5 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        // Layer 2 (node 2): 4×4 weight
        let mut w2 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w2.set(i, j, M31::from(((i + j * 3) % 7 + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        // Layer 4 (node 4): 4×2 weight
        let mut w4 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w4.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(4, w4);

        // === PROVE ===
        let (proofs, execution) =
            prove_model(&graph, &input, &weights).expect("End-to-end proving should succeed");

        // Check proof structure: matmul=sumcheck, relu=STARK, matmul, relu, matmul
        assert_eq!(
            proofs.len(),
            5,
            "5 layers: matmul, relu, matmul, relu, matmul"
        );
        assert!(
            matches!(proofs[0].kind, LayerProofKind::Sumcheck(_)),
            "layer 0 = matmul sumcheck"
        );
        assert!(
            matches!(proofs[1].kind, LayerProofKind::Stark(_)),
            "layer 1 = ReLU STARK"
        );
        assert!(
            matches!(proofs[2].kind, LayerProofKind::Sumcheck(_)),
            "layer 2 = matmul sumcheck"
        );
        assert!(
            matches!(proofs[3].kind, LayerProofKind::Stark(_)),
            "layer 3 = ReLU STARK"
        );
        assert!(
            matches!(proofs[4].kind, LayerProofKind::Sumcheck(_)),
            "layer 4 = matmul sumcheck"
        );

        // Check output shape
        assert_eq!(execution.output.rows, 1);
        assert_eq!(execution.output.cols, 2);

        // === VERIFY ===
        verify_model_matmuls(&proofs, &graph, &input, &weights)
            .expect("End-to-end verification should succeed");
    }

    /// Test that verification catches a tampered weight matrix.
    #[test]
    fn test_verification_catches_wrong_weights() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        // Prove with correct weights
        let (proofs, _) = prove_model(&graph, &input, &weights).expect("proving should succeed");

        // Tamper with weights and try to verify
        let mut tampered_weights = GraphWeights::new();
        let mut w_bad = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w_bad.set(i, j, M31::from(99));
            }
        }
        tampered_weights.add_weight(0, w_bad);

        let result = verify_model_matmuls(&proofs, &graph, &input, &tampered_weights);
        assert!(
            result.is_err(),
            "Verification with tampered weights should fail"
        );
    }

    #[test]
    fn test_prove_activation_layer_standalone() {
        use crate::gadgets::lookup_table::PrecomputedTable;

        // Build a ReLU table with log_size=4 (16 entries)
        let table = PrecomputedTable::build(crate::gadgets::lookup_table::activations::relu, 4);

        // 4 inputs, all in-range for the table
        let inputs = vec![M31::from(0), M31::from(1), M31::from(3), M31::from(5)];
        let outputs: Vec<M31> = inputs
            .iter()
            .map(|&x| crate::gadgets::lookup_table::activations::relu(x))
            .collect();

        let config = PcsConfig::default();
        let result = prove_activation_layer::<SimdBackend, Blake2sMerkleChannel>(
            &inputs, &outputs, &table, config,
        );
        assert!(
            result.is_ok(),
            "Standalone activation proving failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_prove_layernorm_layer_standalone() {
        // Minimal test for prove_layernorm_layer
        let rsqrt_table = build_rsqrt_table(4);

        // 4 elements: simple LayerNorm
        let inputs = vec![M31::from(10), M31::from(20), M31::from(30), M31::from(40)];
        let mean = M31::from(25); // simplified
        let means = vec![mean; 4];
        // variance reduced to table range (log_size=4 → [0,15])
        let var = M31::from(125u32 & ((1u32 << 4) - 1)); // 125 & 0xF = 13
        let variances = vec![var; 4];
        let rsqrt = rsqrt_table
            .lookup(var)
            .expect("variance reduced to table range");
        let rsqrt_vals = vec![rsqrt; 4];
        let outputs: Vec<M31> = inputs.iter().map(|&x| (x - mean) * rsqrt).collect();

        let config = PcsConfig::default();
        let result = prove_layernorm_layer::<SimdBackend, Blake2sMerkleChannel>(
            &inputs,
            &means,
            &variances,
            &rsqrt_vals,
            &outputs,
            &rsqrt_table,
            config,
        );
        assert!(
            result.is_ok(),
            "Standalone LayerNorm proving failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_prove_model_auto() {
        // prove_model_auto should produce identical results to prove_model
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let (proofs, execution) =
            prove_model_auto(&graph, &input, &weights).expect("prove_model_auto should succeed");

        assert_eq!(proofs.len(), 3);
        assert_eq!(execution.output.rows, 1);
        assert_eq!(execution.output.cols, 2);
    }

    /// GPU proving test — only runs with cuda-runtime feature.
    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn test_prove_model_gpu_backend() {
        use stwo::prover::backend::gpu::GpuBackend;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        // Use GpuBackend explicitly
        let (proofs, execution) =
            prove_model_with::<GpuBackend, Blake2sMerkleChannel>(&graph, &input, &weights)
                .expect("GPU proving should succeed");

        assert_eq!(proofs.len(), 3);
        assert_eq!(execution.output.cols, 2);

        // Verify the proofs
        verify_model_matmuls(&proofs, &graph, &input, &weights)
            .expect("GPU proof verification should succeed");
    }

    // === Phase 1A: Add/Mul soundness tests ===

    #[test]
    fn test_add_residual_connection() {
        // Build: linear(4→4) → fork → relu → add_from(fork)
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let branch = builder.fork();
        builder.activation(ActivationType::ReLU);
        builder.add_from(branch);

        let graph = builder.build();
        let weights = crate::compiler::onnx::generate_weights_for_graph(&graph, 42);

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let (proofs, execution) =
            prove_model(&graph, &input, &weights).expect("Add residual proving should succeed");

        assert_eq!(proofs.len(), 3, "3 layers: matmul, relu, add");

        // Verify the Add output is NOT identity — it should be linear_out + relu(linear_out)
        let linear_out = &execution.intermediates[1].1; // relu's input = linear output
        assert_ne!(
            execution.output.data, linear_out.data,
            "Add should modify the output (not identity)"
        );

        // Verify matmul sumcheck
        verify_model_matmuls(&proofs, &graph, &input, &weights)
            .expect("Residual connection verification should succeed");
    }

    #[test]
    fn test_mul_elementwise() {
        // Build: linear(4→4) → fork → linear(4→4) → mul_from(fork)
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let branch = builder.fork();
        builder.linear(4);
        builder.mul_from(branch);

        let graph = builder.build();
        let weights = crate::compiler::onnx::generate_weights_for_graph(&graph, 42);

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let (proofs, _execution) =
            prove_model(&graph, &input, &weights).expect("Mul elementwise proving should succeed");

        assert_eq!(proofs.len(), 3, "3 layers: matmul, matmul, mul");

        // Verify
        verify_model_matmuls(&proofs, &graph, &input, &weights)
            .expect("Mul elementwise verification should succeed");
    }

    #[test]
    fn test_onnx_add_mul_not_identity() {
        // Verify that Add/Mul are no longer mapped to Identity
        use crate::compiler::graph::GraphOp;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let branch = builder.fork();
        builder.activation(ActivationType::ReLU);
        builder.add_from(branch);

        let graph = builder.build();

        // The third node should be Add, NOT Identity
        assert!(
            matches!(graph.nodes[2].op, GraphOp::Add { .. }),
            "Add op should be GraphOp::Add, got: {:?}",
            graph.nodes[2].op
        );
    }

    #[test]
    fn test_elementwise_add_function() {
        let mut a = M31Matrix::new(1, 4);
        let mut b = M31Matrix::new(1, 4);
        for j in 0..4 {
            a.set(0, j, M31::from((j + 1) as u32));
            b.set(0, j, M31::from(10));
        }
        let result = elementwise_add(&a, &b);
        assert_eq!(result.get(0, 0), M31::from(11));
        assert_eq!(result.get(0, 1), M31::from(12));
        assert_eq!(result.get(0, 2), M31::from(13));
        assert_eq!(result.get(0, 3), M31::from(14));
    }

    #[test]
    fn test_elementwise_mul_function() {
        let mut a = M31Matrix::new(1, 4);
        let mut b = M31Matrix::new(1, 4);
        for j in 0..4 {
            a.set(0, j, M31::from((j + 1) as u32));
            b.set(0, j, M31::from(3));
        }
        let result = elementwise_mul(&a, &b);
        assert_eq!(result.get(0, 0), M31::from(3));
        assert_eq!(result.get(0, 1), M31::from(6));
        assert_eq!(result.get(0, 2), M31::from(9));
        assert_eq!(result.get(0, 3), M31::from(12));
    }

    // === Phase 5C: Streaming intermediates tests ===

    #[test]
    fn test_prove_model_streaming_basic() {
        // Same MLP as test_prove_model_mlp_matmul_activation_matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        // Streaming prover
        let (streaming_proofs, streaming_exec) = prove_model_streaming(&graph, &input, &weights)
            .expect("streaming proving should succeed");

        // Standard prover
        let (standard_proofs, standard_exec) =
            prove_model(&graph, &input, &weights).expect("standard proving should succeed");

        // Same number of proofs
        assert_eq!(streaming_proofs.len(), standard_proofs.len());

        // Same output
        assert_eq!(streaming_exec.output.data, standard_exec.output.data);

        // Streaming should have empty intermediates (freed during proving)
        assert!(streaming_exec.intermediates.is_empty());
    }

    #[test]
    fn test_prove_model_streaming_residual() {
        // Residual: linear(4→4) → fork → relu → add_from(fork) → linear(4→2)
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let branch = builder.fork();
        builder.activation(ActivationType::ReLU);
        builder.add_from(branch);
        builder.linear(2);
        let graph = builder.build();
        let weights = crate::compiler::onnx::generate_weights_for_graph(&graph, 42);

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let (streaming_proofs, streaming_exec) = prove_model_streaming(&graph, &input, &weights)
            .expect("streaming residual proving should succeed");

        let (_standard_proofs, standard_exec) = prove_model(&graph, &input, &weights)
            .expect("standard residual proving should succeed");

        assert_eq!(streaming_exec.output.data, standard_exec.output.data);
        assert_eq!(streaming_proofs.len(), 4); // matmul, relu, add, matmul
    }
}
