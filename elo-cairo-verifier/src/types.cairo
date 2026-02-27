// Proof types for on-chain sumcheck verification.
//
// All types derive Serde for calldata deserialization and match
// the serialization layout from stwo-ml's cairo_serde.rs.

use crate::field::QM31;

/// A single round polynomial: p(x) = c0 + c1·x + c2·x².
/// Coefficients in monomial basis, matching STWO's UnivariatePoly<SecureField>.
#[derive(Drop, Copy, Serde)]
pub struct RoundPoly {
    pub c0: QM31,
    pub c1: QM31,
    pub c2: QM31,
}

/// Compressed degree-2 round polynomial: omits c1 (verifier reconstructs it).
/// Since p(0) + p(1) = current_sum and p(1) = c0 + c1 + c2,
/// we get: c1 = current_sum - 2*c0 - c2.
/// Saves 1 QM31 (1 packed felt or 4 unpacked) per sumcheck round.
#[derive(Drop, Copy)]
pub struct CompressedRoundPoly {
    pub c0: QM31,
    pub c2: QM31,
}

/// Compressed degree-3 round polynomial: omits c1 (verifier reconstructs it).
/// Since p(0) + p(1) = current_sum and p(1) = c0 + c1 + c2 + c3,
/// we get: c1 = current_sum - 2*c0 - c2 - c3.
/// Saves 1 QM31 per sumcheck round.
#[derive(Drop, Copy)]
pub struct CompressedGkrRoundPoly {
    pub c0: QM31,
    pub c2: QM31,
    pub c3: QM31,
}

/// Data for a single query at a single folding round of the MLE opening protocol.
#[derive(Drop, Serde)]
pub struct MleQueryRoundData {
    /// Value at the lo half (L_i[idx]).
    pub left_value: QM31,
    /// Value at the hi half (L_i[mid + idx]).
    pub right_value: QM31,
    /// Merkle path siblings for the left value (bottom-up).
    pub left_siblings: Array<felt252>,
    /// Merkle path siblings for the right value (bottom-up).
    pub right_siblings: Array<felt252>,
}

/// Complete data for a single query across all folding rounds.
#[derive(Drop, Serde)]
pub struct MleQueryProof {
    /// Initial query index in layer 0.
    pub initial_pair_index: u32,
    /// Authentication data at each folding round.
    pub rounds: Array<MleQueryRoundData>,
}

/// Opening proof for MLE(point) = claimed_eval using multilinear folding.
#[derive(Drop, Serde)]
pub struct MleOpeningProof {
    /// Merkle roots of intermediate folded layers (R_1, ..., R_{n-1}).
    pub intermediate_roots: Array<felt252>,
    /// Spot-check query proofs.
    pub queries: Array<MleQueryProof>,
    /// The final value after all folds.
    pub final_value: QM31,
}

/// Complete sumcheck proof with MLE commitment openings for on-chain verification.
///
/// Field order matches cairo_serde.rs serialize_matmul_sumcheck_proof():
/// m, k, n, num_rounds, claimed_sum, round_polys, final_a_eval, final_b_eval,
/// a_commitment, b_commitment, a_opening, b_opening
#[derive(Drop, Serde)]
pub struct MatMulSumcheckProof {
    /// Matrix dimensions: A is m×k, B is k×n, C is m×n.
    pub m: u32,
    pub k: u32,
    pub n: u32,
    /// Number of sumcheck rounds (= ceil_log2(k)).
    pub num_rounds: u32,
    /// The claimed value: MLE_C evaluated at the random point.
    pub claimed_sum: QM31,
    /// One degree-2 polynomial per sumcheck round.
    pub round_polys: Array<RoundPoly>,
    /// MLE_A evaluated at (row_challenges, assignment).
    pub final_a_eval: QM31,
    /// MLE_B evaluated at (assignment, col_challenges).
    pub final_b_eval: QM31,
    /// Poseidon Merkle root of matrix A entries.
    pub a_commitment: felt252,
    /// Poseidon Merkle root of matrix B entries.
    pub b_commitment: felt252,
    /// MLE opening proof verifying final_a_eval against a_commitment.
    pub a_opening: MleOpeningProof,
    /// MLE opening proof verifying final_b_eval against b_commitment.
    pub b_opening: MleOpeningProof,
}

// ============================================================================
// Batched Matmul Sumcheck Types
// ============================================================================

/// Per-matmul entry within a batched proof.
///
/// Field order matches cairo_serde.rs serialize_batched_matmul_for_recursive()
/// per-entry layout: node_id, m, n, claimed_sum, final_a_eval, final_b_eval,
/// a_commitment, b_commitment.
#[derive(Drop, Copy, Serde)]
pub struct BatchedMatMulEntry {
    /// Graph node identifier for this matmul.
    pub node_id: u32,
    /// Row dimension of matrix A.
    pub m: u32,
    /// Column dimension of matrix B.
    pub n: u32,
    /// MLE_C evaluated at (r_i, r_j) for this matmul.
    pub claimed_sum: QM31,
    /// MLE_A evaluated at (row_challenges, assignment).
    pub final_a_eval: QM31,
    /// MLE_B evaluated at (assignment, col_challenges).
    pub final_b_eval: QM31,
    /// Poseidon Merkle root of restricted MLE_A.
    pub a_commitment: felt252,
    /// Poseidon Merkle root of restricted MLE_B.
    pub b_commitment: felt252,
}

/// Batched matmul sumcheck proof — multiple matmuls combined with lambda weighting.
///
/// Instead of N individual sumcheck proofs, a batch combines them:
///   h(x) = Σ λ^i · f_a_i(x) · f_b_i(x)
/// One set of shared round polynomials + per-matmul final evaluations.
///
/// Field order matches cairo_serde.rs serialize_batched_matmul_for_recursive():
/// k, num_rounds, lambda, combined_claimed_sum, round_polys[], entries[].
#[derive(Drop, Serde)]
pub struct BatchedMatMulProof {
    /// Padded k dimension (shared by all entries in this batch).
    pub k: u32,
    /// Number of sumcheck rounds (= log2(k)).
    pub num_rounds: u32,
    /// Lambda batching weight drawn from Fiat-Shamir.
    pub lambda: QM31,
    /// Combined claimed sum: Σ λ^i · claimed_sum_i.
    pub combined_claimed_sum: QM31,
    /// Shared round polynomials (one degree-2 polynomial per round).
    pub round_polys: Array<RoundPoly>,
    /// Per-matmul entries with individual evaluations and commitments.
    pub entries: Array<BatchedMatMulEntry>,
}

// ============================================================================
// GKR Batch Verification Types
// ============================================================================

/// Gate type: 0 = GrandProduct, 1 = LogUp.
/// GrandProduct: 1 column, output = a * b.
/// LogUp: 2 columns (numerator, denominator), output = fraction addition.
#[derive(Drop, Copy, Serde)]
pub struct GateType {
    pub gate_id: u32,
}

/// GKR round polynomial with variable degree (up to 3).
/// STWO's UnivariatePoly truncates leading zeros, so num_coeffs can be 1..4.
/// For Fiat-Shamir mixing, only the first num_coeffs values are mixed.
#[derive(Drop, Copy, Serde)]
pub struct GkrRoundPoly {
    pub c0: QM31,
    pub c1: QM31,
    pub c2: QM31,
    pub c3: QM31,
    pub num_coeffs: u32,
}

/// Stores two evaluations per column in a GKR layer mask.
/// Flattened: [col0_v0, col0_v1, col1_v0, col1_v1, ...].
/// num_columns indicates how many column pairs are stored.
#[derive(Drop, Serde)]
pub struct GkrMask {
    pub values: Array<QM31>,
    pub num_columns: u32,
}

/// Sumcheck proof for a single GKR layer.
#[derive(Drop, Serde)]
pub struct GkrSumcheckProof {
    pub round_polys: Array<GkrRoundPoly>,
}

/// Proof data for a single GKR layer: sumcheck proof + masks for active instances.
#[derive(Drop, Serde)]
pub struct GkrLayerProof {
    pub sumcheck_proof: GkrSumcheckProof,
    pub masks: Array<GkrMask>,
}

/// Per-instance metadata for the GKR batch proof.
#[derive(Drop, Serde)]
pub struct GkrInstance {
    pub gate: GateType,
    pub n_variables: u32,
    pub output_claims: Array<QM31>,
}

/// Complete batch GKR proof matching STWO's GkrBatchProof structure.
#[derive(Drop, Serde)]
pub struct GkrBatchProof {
    pub instances: Array<GkrInstance>,
    pub layer_proofs: Array<GkrLayerProof>,
}

/// Output artifact from GKR verification.
/// Contains the OOD point and per-instance claims to verify against input layer.
#[derive(Drop, Serde)]
pub struct GkrArtifact {
    pub ood_point: Array<QM31>,
    pub claims_to_verify: Array<Array<QM31>>,
    pub n_variables_by_instance: Array<u32>,
}

/// Verification result emitted as event data.
#[derive(Drop, Copy, Serde)]
pub struct VerificationResult {
    pub verified: bool,
    pub proof_hash: felt252,
    pub num_rounds: u32,
}

// ============================================================================
// Unified Model Verification Types (Phase 8)
// ============================================================================

/// PCS security configuration (flat version for elo-cairo-verifier's own proofs).
#[derive(Drop, Copy, Serde)]
pub struct PcsConfig {
    pub pow_bits: u32,
    pub log_blowup_factor: u32,
    pub log_last_layer_degree_bound: u32,
    pub n_queries: u32,
}

/// Unified proof for a full ZKML model.
///
/// Combines all verification sub-proofs into a single structure.
/// The Rust prover serializes this via `build_starknet_proof_onchain()`.
///
/// Layout matches the combined_calldata format:
///   [pcs_config(4)] [raw_io_data(Array)] [layer_chain_commitment(1)]
///   [matmul_count(1)] [per-matmul: len + data]...
///   [batch_count(1)] [per-batch: data]...
///   [has_gkr(1)] [gkr_len + gkr_data]?
#[derive(Drop, Serde)]
pub struct ModelProof {
    /// PCS security parameters.
    pub pcs_config: PcsConfig,
    /// Raw IO data: [in_rows, in_cols, in_len, in_data..., out_rows, out_cols, out_len, out_data...].
    /// The verifier recomputes Poseidon(raw_io_data) on-chain — never trusts a caller-supplied hash.
    pub raw_io_data: Array<felt252>,
    /// Running Poseidon hash of intermediate layer outputs.
    pub layer_chain_commitment: felt252,
    /// Individual matmul sumcheck proofs (node_id, proof).
    pub matmul_proofs: Array<MatMulSumcheckProof>,
    /// Batched matmul proofs (groups of same-k matmuls).
    pub batched_matmul_proofs: Array<BatchedMatMulProof>,
    /// Optional GKR proof for LogUp/GrandProduct lookup arguments.
    pub has_gkr: bool,
    /// GKR proof (only present if has_gkr == true).
    pub gkr_proof: Array<GkrBatchProof>,
}

// ============================================================================
// Direct Model Verification Types (Phase 9 — eliminates Cairo VM Stage 2)
// ============================================================================

/// Direct model proof for on-chain verification without recursive proving.
///
/// Contains:
///   1. Batch sumcheck proofs (matmul verification)
///   2. Activation STARK proof (unified STARK via Air<MLAir>)
///
/// This replaces the 3-stage pipeline:
///   BEFORE: GPU prove → Cairo VM (46.8s) → on-chain verify recursive proof
///   AFTER:  GPU prove → on-chain verify_model_direct() (0s Stage 2)
#[derive(Drop, Serde)]
pub struct DirectModelProof {
    /// Model metadata.
    pub model_id: felt252,
    /// Raw IO data for on-chain recomputation of Poseidon(inputs || outputs).
    pub raw_io_data: Array<felt252>,
    /// Poseidon hash of all weight matrices.
    pub weight_commitment: felt252,
    /// Number of model layers.
    pub num_layers: u32,
    /// Activation function type: 0=ReLU, 1=GELU, 2=Sigmoid.
    pub activation_type: u8,
    /// Batched matmul sumcheck proofs (groups of same-k matmuls).
    pub batched_matmul_proofs: Array<BatchedMatMulProof>,
    /// Whether an activation STARK proof is present.
    pub has_activation_stark: bool,
    /// Number of activation STARK data chunks (0 if no activation STARK).
    pub num_stark_chunks: u32,
}

// ============================================================================
// ML GKR Full Verification Types (100% on-chain, no STARK/FRI/dicts)
// ============================================================================

/// LogUp sub-proof for activation/layernorm/dequantize lookup verification.
///
/// Encapsulates the degree-3 eq-sumcheck over the LogUp relation plus
/// the per-table-entry multiplicities for table-side sum computation.
#[derive(Drop, Serde)]
pub struct LogUpProof {
    /// Total LogUp sum (trace-side): S = sum_i 1/(gamma - in_i - beta*out_i).
    pub claimed_sum: QM31,
    /// Number of eq-sumcheck rounds (= log2(table_size)).
    pub num_rounds: u32,
    /// Degree-3 round polynomials for the eq-sumcheck.
    pub eq_round_polys: Array<GkrRoundPoly>,
    /// LogUp weight w(s) at the final sumcheck point.
    pub final_w_eval: QM31,
    /// Input MLE at the final sumcheck point.
    pub final_in_eval: QM31,
    /// Output MLE at the final sumcheck point.
    pub final_out_eval: QM31,
    /// Number of table entries with non-zero multiplicities.
    pub num_multiplicities: u32,
    /// Per-table-entry multiplicity counts.
    pub multiplicities: Array<u32>,
}

/// GKR claim: an evaluation point + claimed value on a multilinear polynomial.
///
/// Used to thread claims through the GKR layer walk:
///   output claim -> layer verifier -> input claim -> next layer...
#[derive(Drop, Clone, Serde)]
pub struct GKRClaim {
    /// Evaluation point in the boolean hypercube extension.
    pub point: Array<QM31>,
    /// Claimed MLE value at this point.
    pub value: QM31,
}

/// Complete GKR model proof for full on-chain verification.
///
/// The layer_proof_data is a flat felt252 array containing tag-dispatched
/// per-layer proofs. Cairo deserializes via manual offset-based reading
/// (not auto-derived Serde) because each layer type has different fields.
///
/// Layout matches stwo-ml/src/cairo_serde.rs:serialize_gkr_model_proof().
/// Tags: 0=MatMul, 1=Add, 2=Mul, 3=Activation, 4=LayerNorm,
///       5=Attention, 6=Dequantize, 7=MatMulDualSimd, 8=RMSNorm
#[derive(Drop, Serde)]
pub struct GKRModelProof {
    /// Number of layers in the model.
    pub num_layers: u32,
    /// Flat serialized layer proofs (tag + variant-specific data per layer).
    pub layer_proof_data: Array<felt252>,
    /// Length of the input claim evaluation point.
    pub input_claim_point_len: u32,
    /// Input claim evaluation point (final reduced point from GKR walk).
    pub input_claim_point: Array<QM31>,
    /// Input claim value (expected MLE(raw_input, point)).
    pub input_claim_value: QM31,
    /// Number of weight MLE commitment roots.
    pub num_weight_commitments: u32,
    /// Poseidon Merkle roots of weight MLEs (one per MatMul layer).
    pub weight_commitments: Array<felt252>,
    /// Raw IO data for on-chain recomputation of Poseidon(inputs || outputs).
    /// Layout: [in_rows, in_cols, in_len, in_data..., out_rows, out_cols, out_len, out_data...].
    pub raw_io_data: Array<felt252>,
}

/// Weight MLE opening proof: verifies a weight matrix commitment.
#[derive(Drop, Serde)]
pub struct WeightMleOpening {
    /// Index of the layer this weight belongs to.
    pub layer_index: u32,
    /// The GKR claim (point, value) reduced to this weight matrix.
    pub claim: GKRClaim,
    /// Merkle-based MLE opening proof against the registered root.
    pub opening_proof: MleOpeningProof,
}
