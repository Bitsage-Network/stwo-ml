//! Proving pipeline for computation graphs.
//!
//! Takes a `ComputationGraph`, input data, and weights, then generates
//! STARK proofs for each layer using the appropriate component.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::channel::{Blake2sChannel, MerkleChannel};
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::CommitmentSchemeProver;
use stwo::prover::poly::circle::PolyOps;
use stwo::prover::prove;

use stwo_constraint_framework::{
    FrameworkComponent, TraceLocationAllocator,
    LogupTraceGenerator,
};

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

/// Model proof result: per-layer proofs + execution trace.
pub type ModelProofResult = (Vec<LayerProof<Blake2sHash>>, GraphExecution);

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

/// Prove a single activation layer using SimdBackend + Blake2s.
///
/// Uses LogUp to verify every (input, output) pair exists in the precomputed table.
/// Tree layout: [preprocessed table | execution trace | LogUp interaction].
pub fn prove_activation_layer(
    inputs: &[M31],
    outputs: &[M31],
    table: &PrecomputedTable,
    config: PcsConfig,
) -> Result<(FrameworkComponent<ActivationEval>, StarkProof<<Blake2sMerkleChannel as MerkleChannel>::H>), ModelError> {
    use stwo::prover::backend::simd::m31::LOG_N_LANES;

    let log_size = table.log_size.max(4);
    let size = 1usize << log_size;
    let vec_size = size >> LOG_N_LANES;
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Pad trace with the first table entry for rows beyond real data.
    // This ensures padding rows reference a valid table entry.
    let pad_input = table.inputs[0];
    let pad_output = table.outputs[0];
    let padding_count = size.saturating_sub(inputs.len());

    // Build multiplicities: count real uses + padding uses
    let mut multiplicities = compute_multiplicities(inputs, table);
    if padding_count > 0 {
        // Padding rows all "use" the first table entry
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
    // Padding rows use valid table entry
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

    // --- Commitment scheme setup ---
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Tree 0: Preprocessed columns (activation table: input + output)
    let preprocessed_trace = vec![
        CircleEvaluation::new(domain, table_input_col.clone()),
        CircleEvaluation::new(domain, table_output_col.clone()),
    ];
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(preprocessed_trace);
    tree_builder.commit(channel);

    // Tree 1: Execution trace (trace_input, trace_output, multiplicity)
    let execution_trace = vec![
        CircleEvaluation::new(domain, trace_input_col.clone()),
        CircleEvaluation::new(domain, trace_output_col.clone()),
        CircleEvaluation::new(domain, mult_col),
    ];
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(execution_trace);
    tree_builder.commit(channel);

    // Draw lookup elements (Fiat-Shamir after committing trees 0 + 1)
    let lookup_elements = ActivationRelation::draw(channel);

    // Tree 2: Interaction trace — 1 LogUp column with combined fractions.
    //
    // Each row combines table-yield + trace-use into one fraction:
    //   frac = -mult/q_table + 1/q_trace
    //        = (q_table - mult * q_trace) / (q_table * q_trace)
    //
    // where q_table = combine([table_in, table_out])
    //       q_trace = combine([trace_in, trace_out])
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
            &multiplicities, vec_row, log_size,
        );

        let numerator = q_table - mult_packed * q_trace;
        let denominator = q_table * q_trace;

        col_gen.write_frac(vec_row, numerator, denominator);
    }
    col_gen.finalize_col();

    let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace);
    tree_builder.commit(channel);

    // Build component
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

    // Prove
    let proof = prove::<SimdBackend, Blake2sMerkleChannel>(
        &[&component],
        channel,
        commitment_scheme,
    ).map_err(|e| ModelError::ProvingError {
        layer: 0,
        message: format!("{e:?}"),
    })?;

    Ok((component, proof))
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

/// Prove an entire computation graph: execute forward pass and generate
/// real STARK proofs (activations) and sumcheck proofs (matmuls).
///
/// Returns per-layer proofs and the final output matrix.
pub fn prove_model(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<ModelProofResult, ModelError> {
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
                let (_component, proof) = prove_activation_layer(
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

            GraphOp::LayerNorm { .. }
            | GraphOp::Quantize { .. }
            | GraphOp::Identity { .. } => {
                // Passthrough for now — LayerNorm proving can be added
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

/// Verify all matmul sumcheck proofs in a model's layer proofs.
///
/// For each Sumcheck proof, replays the Fiat-Shamir transcript
/// and checks the final MLE evaluations against the original matrices.
pub fn verify_model_matmuls(
    layer_proofs: &[LayerProof<<Blake2sMerkleChannel as MerkleChannel>::H>],
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
                // STARK proof verified separately; here just replay the forward pass
                let f = activation_type.as_fn();
                current = apply_activation(&current, &*f);
            }
            _ => {}
        }
    }

    Ok(())
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
        let result = prove_activation_layer(&inputs, &outputs, &table, config);
        assert!(result.is_ok(), "Standalone activation proving failed: {:?}", result.err());
    }
}
