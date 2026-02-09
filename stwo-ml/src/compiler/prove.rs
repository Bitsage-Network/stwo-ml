//! Proving pipeline for computation graphs.
//!
//! Takes a `ComputationGraph`, input data, and weights, then generates
//! STARK proofs for each layer using the appropriate component.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::channel::MerkleChannel;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::{Col, Column, BackendForChannel};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::CommitmentSchemeProver;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::prove;

use stwo_constraint_framework::{
    FrameworkComponent, TraceLocationAllocator,
    LogupTraceGenerator,
};

use crate::backend::convert_evaluations;

use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use crate::components::activation::{
    ActivationEval, ActivationRelation,
    compute_multiplicities,
};
use crate::components::matmul::{
    M31Matrix, matmul_m31,
    MatMulSumcheckProof, prove_matmul_sumcheck,
};
use crate::gadgets::lookup_table::PrecomputedTable;

/// Proof for a single layer — either a STARK (activation/layernorm) or sumcheck (matmul).
#[derive(Debug)]
pub enum LayerProofKind<H: MerkleHasherLifted> {
    /// Activation/LayerNorm: LogUp-based STARK proof.
    Stark(StarkProof<H>),
    /// MatMul: Sumcheck proof over multilinear extensions.
    Sumcheck(MatMulSumcheckProof),
    /// Identity/Quantize: no proof needed.
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
) -> Result<(FrameworkComponent<ActivationEval>, StarkProof<<MC as MerkleChannel>::H>), ModelError>
where
    B: BackendForChannel<MC> + PolyOps,
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
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(cols.preprocessed));
    tree_builder.commit(channel);

    // Tree 1: Execution trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(cols.execution));
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
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_simd));
    tree_builder.commit(channel);

    // --- Phase E: Build component and prove ---
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        ActivationEval {
            log_n_rows: log_size,
            lookup_elements,
            claimed_sum,
            total_sum: claimed_sum,
        },
        claimed_sum,
    );

    let proof = prove::<B, MC>(
        &[&component],
        channel,
        commitment_scheme,
    ).map_err(|e| ModelError::ProvingError {
        layer: 0,
        message: format!("{e:?}"),
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
    for (i, (&inp, &out)) in table.inputs.iter().zip(table.outputs.iter()).enumerate().take(size) {
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
) -> (Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>, SecureField) {
    use stwo::prover::backend::simd::m31::LOG_N_LANES;

    let size = 1usize << log_size;
    let vec_size = size >> LOG_N_LANES;

    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let mut col_gen = logup_gen.new_col();

    for vec_row in 0..vec_size {
        let q_table: PackedSecureField = lookup_elements.lookup_elements().combine(
            &[table_input_col.data[vec_row], table_output_col.data[vec_row]],
        );
        let q_trace: PackedSecureField = lookup_elements.lookup_elements().combine(
            &[trace_input_col.data[vec_row], trace_output_col.data[vec_row]],
        );

        let mult_packed: PackedSecureField = mult_col_data_at(
            multiplicities, vec_row, log_size,
        );

        let numerator = q_table - mult_packed * q_trace;
        let denominator = q_table * q_trace;

        col_gen.write_frac(vec_row, numerator, denominator);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

/// Helper: get packed multiplicity values at a given vec_row.
fn mult_col_data_at(
    multiplicities: &[M31],
    vec_row: usize,
    _log_size: u32,
) -> PackedSecureField {
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

/// Apply an activation function element-wise to a matrix (public for aggregation).
pub fn apply_activation_pub(input: &M31Matrix, f: &dyn Fn(M31) -> M31) -> M31Matrix {
    apply_activation(input, f)
}

/// Apply an activation function element-wise to a matrix.
fn apply_activation(input: &M31Matrix, f: &dyn Fn(M31) -> M31) -> M31Matrix {
    let mut output = M31Matrix::new(input.rows, input.cols);
    for i in 0..input.data.len() {
        output.data[i] = f(input.data[i]);
    }
    output
}

/// Compute LayerNorm forward pass in M31 field arithmetic.
///
/// y = (x - mean) * rsqrt(variance)
///
/// Division by n uses modular inverse (Fermat's little theorem).
/// Reciprocal sqrt looked up in precomputed table.
fn apply_layernorm(input: &M31Matrix, dim: usize) -> M31Matrix {
    use crate::components::layernorm::build_rsqrt_table;

    let rsqrt_table = build_rsqrt_table(4);
    let mut output = M31Matrix::new(input.rows, input.cols);
    let n = dim.min(input.cols);
    let inv_n = m31_mod_inverse(n as u32);

    for row in 0..input.rows {
        // Mean: sum(x) / n
        let mut sum = M31::from(0);
        for col in 0..n {
            sum += input.data[row * input.cols + col];
        }
        let mean = sum * inv_n;

        // Variance: sum((x - mean)^2) / n
        let mut var_sum = M31::from(0);
        for col in 0..n {
            let diff = input.data[row * input.cols + col] - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum * inv_n;

        // rsqrt(variance) from lookup table. Fallback to scale=1.0 (2^16).
        let rsqrt = rsqrt_table
            .lookup(variance)
            .unwrap_or(M31::from(1u32 << 16));

        // Output: (x - mean) * rsqrt
        for col in 0..n {
            let centered = input.data[row * input.cols + col] - mean;
            output.data[row * input.cols + col] = centered * rsqrt;
        }
        // Pass through any columns beyond the normalization dimension
        for col in n..input.cols {
            output.data[row * input.cols + col] = input.data[row * input.cols + col];
        }
    }

    output
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
    B: BackendForChannel<MC> + PolyOps,
    MC: MerkleChannel,
    FrameworkComponent<ActivationEval>: stwo::prover::ComponentProver<B>,
{
    let mut layer_proofs = Vec::new();
    let mut intermediates: Vec<(usize, M31Matrix)> = Vec::new();

    // Current activation flowing through the network
    let mut current = input.clone();

    for node in &graph.nodes {
        match &node.op {
            GraphOp::MatMul { dims } => {
                let (_m, _k, _n) = *dims;

                // Get weight matrix for this layer
                let weight = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;

                // Forward pass: C = current × weight
                let output = matmul_m31(&current, weight);

                // Generate sumcheck proof
                let proof = prove_matmul_sumcheck(&current, weight, &output)
                    .map_err(|e| ModelError::ProvingError {
                        layer: node.id,
                        message: format!("MatMul sumcheck: {e}"),
                    })?;

                let claimed_sum = proof.claimed_sum;
                intermediates.push((node.id, current.clone()));
                current = output;

                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Sumcheck(proof),
                    claimed_sum,
                    layer_index: node.id,
                });
            }

            GraphOp::Activation { activation_type, size: _ } => {
                let f = activation_type.as_fn();
                let output = apply_activation(&current, &*f);

                // Build lookup table for this activation type
                let table = PrecomputedTable::build(
                    |x| (*f)(x),
                    4, // log_size=4 (16 entries) for small models
                );
                let config = PcsConfig::default();

                // Flatten current matrix to input/output slices
                let flat_inputs: Vec<M31> = current.data.clone();
                let flat_outputs: Vec<M31> = output.data.clone();

                // Generate real STARK proof via LogUp
                let (_component, proof) = prove_activation_layer::<B, MC>(
                    &flat_inputs,
                    &flat_outputs,
                    &table,
                    config,
                ).map_err(|e| ModelError::ProvingError {
                    layer: node.id,
                    message: format!("Activation STARK: {e}"),
                })?;

                let claimed_sum = SecureField::from(M31::from(0));
                intermediates.push((node.id, current.clone()));
                current = output;

                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Stark(proof),
                    claimed_sum,
                    layer_index: node.id,
                });
            }

            GraphOp::LayerNorm { dim } => {
                // Real LayerNorm forward pass in M31 arithmetic.
                // STARK proof for rsqrt lookup is generated at the aggregation layer.
                let output = apply_layernorm(&current, *dim);
                intermediates.push((node.id, current.clone()));
                current = output;

                layer_proofs.push(LayerProof {
                    kind: LayerProofKind::Passthrough,
                    claimed_sum: SecureField::from(M31::from(0)),
                    layer_index: node.id,
                });
            }

            GraphOp::Quantize { .. }
            | GraphOp::Identity { .. } => {
                // Quantize: applied to weights at load time, identity in forward pass.
                // Identity: no-op by definition.
                intermediates.push((node.id, current.clone()));
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

    for (proof, node) in layer_proofs.iter().zip(graph.nodes.iter()) {
        match (&proof.kind, &node.op) {
            (LayerProofKind::Sumcheck(matmul_proof), GraphOp::MatMul { .. }) => {
                let weight = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;

                let c = matmul_m31(&current, weight);

                verify_matmul_sumcheck(matmul_proof, &current, weight, &c)
                    .map_err(|e| ModelError::VerificationError {
                        layer: node.id,
                        message: format!("MatMul verification failed: {e}"),
                    })?;

                current = c;
            }
            (LayerProofKind::Stark(_), GraphOp::Activation { activation_type, .. })
            | (LayerProofKind::Passthrough, GraphOp::Activation { activation_type, .. }) => {
                let f = activation_type.as_fn();
                current = apply_activation(&current, &*f);
            }
            (LayerProofKind::Passthrough, GraphOp::LayerNorm { dim }) => {
                current = apply_layernorm(&current, *dim);
            }
            _ => {}
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
    crate::backend::with_best_backend(
        || prove_model_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights),
        || prove_model_gpu(graph, input, weights),
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

        let (proofs, execution) = prove_model(&graph, &input, &weights)
            .expect("model proving should succeed");

        assert_eq!(proofs.len(), 1);
        assert!(matches!(proofs[0].kind, LayerProofKind::Sumcheck(_)));
        assert_eq!(execution.output.rows, 1);
        assert_eq!(execution.output.cols, 2);
    }

    #[test]
    fn test_prove_model_mlp_matmul_activation_matmul() {
        // 2-layer MLP: input(1×4) → linear(4→4) → ReLU → linear(4→2)
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();

        // Layer 0: 4×4 weight
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32)); } }
        weights.add_weight(0, w0);

        // Layer 2: 4×2 weight (node 1 is activation, node 2 is second linear)
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let (proofs, execution) = prove_model(&graph, &input, &weights)
            .expect("MLP proving should succeed");

        assert_eq!(proofs.len(), 3, "3 layers: matmul, activation, matmul");
        assert!(matches!(proofs[0].kind, LayerProofKind::Sumcheck(_)), "layer 0 = matmul");
        assert!(matches!(proofs[1].kind, LayerProofKind::Stark(_)), "layer 1 = activation (STARK proof)");
        assert!(matches!(proofs[2].kind, LayerProofKind::Sumcheck(_)), "layer 2 = matmul");
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
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        // Weights for each linear layer
        let mut weights = GraphWeights::new();

        // Layer 0 (node 0): 4×4 weight
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 {
            w0.set(i, j, M31::from(((i * 4 + j) % 5 + 1) as u32));
        }}
        weights.add_weight(0, w0);

        // Layer 2 (node 2): 4×4 weight
        let mut w2 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 {
            w2.set(i, j, M31::from(((i + j * 3) % 7 + 1) as u32));
        }}
        weights.add_weight(2, w2);

        // Layer 4 (node 4): 4×2 weight
        let mut w4 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 {
            w4.set(i, j, M31::from((i * 2 + j + 1) as u32));
        }}
        weights.add_weight(4, w4);

        // === PROVE ===
        let (proofs, execution) = prove_model(&graph, &input, &weights)
            .expect("End-to-end proving should succeed");

        // Check proof structure: matmul=sumcheck, relu=STARK, matmul, relu, matmul
        assert_eq!(proofs.len(), 5, "5 layers: matmul, relu, matmul, relu, matmul");
        assert!(matches!(proofs[0].kind, LayerProofKind::Sumcheck(_)), "layer 0 = matmul sumcheck");
        assert!(matches!(proofs[1].kind, LayerProofKind::Stark(_)), "layer 1 = ReLU STARK");
        assert!(matches!(proofs[2].kind, LayerProofKind::Sumcheck(_)), "layer 2 = matmul sumcheck");
        assert!(matches!(proofs[3].kind, LayerProofKind::Stark(_)), "layer 3 = ReLU STARK");
        assert!(matches!(proofs[4].kind, LayerProofKind::Sumcheck(_)), "layer 4 = matmul sumcheck");

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
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        // Prove with correct weights
        let (proofs, _) = prove_model(&graph, &input, &weights)
            .expect("proving should succeed");

        // Tamper with weights and try to verify
        let mut tampered_weights = GraphWeights::new();
        let mut w_bad = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w_bad.set(i, j, M31::from(99)); } }
        tampered_weights.add_weight(0, w_bad);

        let result = verify_model_matmuls(&proofs, &graph, &input, &tampered_weights);
        assert!(result.is_err(), "Verification with tampered weights should fail");
    }

    #[test]
    fn test_prove_activation_layer_standalone() {
        use crate::gadgets::lookup_table::PrecomputedTable;

        // Build a ReLU table with log_size=4 (16 entries)
        let table = PrecomputedTable::build(
            crate::gadgets::lookup_table::activations::relu,
            4,
        );

        // 4 inputs, all in-range for the table
        let inputs = vec![M31::from(0), M31::from(1), M31::from(3), M31::from(5)];
        let outputs: Vec<M31> = inputs.iter().map(|&x| {
            crate::gadgets::lookup_table::activations::relu(x)
        }).collect();

        let config = PcsConfig::default();
        let result = prove_activation_layer::<SimdBackend, Blake2sMerkleChannel>(&inputs, &outputs, &table, config);
        assert!(result.is_ok(), "Standalone activation proving failed: {:?}", result.err());
    }

    #[test]
    fn test_prove_model_auto() {
        // prove_model_auto should produce identical results to prove_model
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

        let (proofs, execution) = prove_model_auto(&graph, &input, &weights)
            .expect("prove_model_auto should succeed");

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

        // Use GpuBackend explicitly
        let (proofs, execution) = prove_model_with::<GpuBackend, Blake2sMerkleChannel>(
            &graph, &input, &weights,
        ).expect("GPU proving should succeed");

        assert_eq!(proofs.len(), 3);
        assert_eq!(execution.output.cols, 2);

        // Verify the proofs
        verify_model_matmuls(&proofs, &graph, &input, &weights)
            .expect("GPU proof verification should succeed");
    }
}
