//! GKR protocol proof types and error types.

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::components::activation::ActivationType;
use crate::components::matmul::RoundPoly;
use crate::crypto::aggregated_opening::AggregatedWeightBindingProof;
use crate::crypto::mle_opening::MleOpeningProof;
use crate::crypto::poseidon_channel::{securefield_to_felt, PoseidonChannel};

/// QM31 is the secure (extension) field used throughout stwo-ml.
pub type SecureField = QM31;

/// A claim in the GKR protocol: "the MLE evaluated at `point` equals `value`."
#[derive(Debug, Clone)]
pub struct GKRClaim {
    pub point: Vec<SecureField>,
    pub value: SecureField,
}

/// Degree-3 round polynomial for eq-sumcheck (Mul layer).
///
/// p(t) = c0 + c1*t + c2*t² + c3*t³
/// Used when the sumcheck involves eq(r,x)·a(x)·b(x) — three degree-1 factors
/// give a degree-3 univariate in each round.
#[derive(Debug, Clone, Copy)]
pub struct RoundPolyDeg3 {
    pub c0: SecureField,
    pub c1: SecureField,
    pub c2: SecureField,
    pub c3: SecureField,
}

impl RoundPolyDeg3 {
    /// Evaluate p(t) = c0 + c1*t + c2*t² + c3*t³.
    pub fn eval(&self, t: SecureField) -> SecureField {
        self.c0 + self.c1 * t + self.c2 * t * t + self.c3 * t * t * t
    }

    /// Compress by omitting c1 (reconstructible from the sumcheck consistency equation).
    ///
    /// Since `p(0) + p(1) = current_sum` and `p(1) = c0 + c1 + c2 + c3`,
    /// we get: `c1 = current_sum - 2*c0 - c2 - c3`.
    pub fn compress(&self) -> CompressedRoundPolyDeg3 {
        CompressedRoundPolyDeg3 {
            c0: self.c0,
            c2: self.c2,
            c3: self.c3,
        }
    }
}

/// Compressed degree-3 round polynomial: omits c1 (verifier reconstructs it).
///
/// Saves 4 felt252s (1 QM31) per sumcheck round — 25% reduction in round poly calldata.
/// Verifier reconstructs: `c1 = current_sum - 2*c0 - c2 - c3`.
#[derive(Debug, Clone, Copy)]
pub struct CompressedRoundPolyDeg3 {
    pub c0: SecureField,
    pub c2: SecureField,
    pub c3: SecureField,
}

impl CompressedRoundPolyDeg3 {
    /// Reconstruct the full polynomial given the running sumcheck sum.
    pub fn decompress(&self, current_sum: SecureField) -> RoundPolyDeg3 {
        let two = SecureField::from(stwo::core::fields::m31::M31::from(2));
        let c1 = current_sum - two * self.c0 - self.c2 - self.c3;
        RoundPolyDeg3 {
            c0: self.c0,
            c1,
            c2: self.c2,
            c3: self.c3,
        }
    }
}

/// LogUp proof for activation/layernorm lookup arguments.
///
/// Proves that every (input, output) pair exists in a precomputed lookup table
/// using a degree-3 eq-sumcheck on inverse witnesses.
///
/// The protocol:
/// 1. Encode each pair: encode_i = in_i + β·out_i
/// 2. Compute denominators: d_i = γ - encode_i
/// 3. Compute inverse witnesses: w_i = 1/d_i
/// 4. Verify LogUp sum: Σ w_i = Σ mult_j/(γ - table_encode_j)
/// 5. Eq-sumcheck proves: Σ_{x} eq(r,x)·w(x)·d(x) = 1
///    (i.e., w(x)·d(x) = 1 for all boolean x — fractions are correctly formed)
#[derive(Debug, Clone)]
pub struct LogUpProof {
    /// Degree-3 eq-sumcheck round polynomials.
    /// Proves Σ eq(r,x) · w(x) · (γ - Ṽ_in(x) - β·Ṽ_out(x)) = 1.
    pub eq_round_polys: Vec<RoundPolyDeg3>,

    /// Final evaluations at the sumcheck challenge point s:
    /// (w(s), in(s), out(s)).
    pub final_evals: (SecureField, SecureField, SecureField),

    /// The claimed total LogUp sum: Σ_i 1/(γ - encode_i).
    pub claimed_sum: SecureField,

    /// Multiplicities for each table entry (prover sends these; verifier uses
    /// them to independently compute the table-side sum).
    pub multiplicities: Vec<u32>,
}

/// LogUp proof for embedding lookups: (token_id, column, value) relation.
///
/// Unlike activation/dequantize, the embedding table is model-dependent, so the
/// proof carries sparse table multiplicities keyed by `(token_id, column)`.
#[derive(Debug, Clone)]
pub struct EmbeddingLogUpProof {
    /// Degree-3 eq-sumcheck round polynomials proving w(x) * d(x) = 1 over the trace domain.
    pub eq_round_polys: Vec<RoundPolyDeg3>,
    /// Final evaluations at the sumcheck challenge point s: (w(s), tok(s), col(s), val(s)).
    pub final_evals: (SecureField, SecureField, SecureField, SecureField),
    /// Claimed trace-side LogUp sum Σ_i 1 / (γ - encode_i).
    pub claimed_sum: SecureField,
    /// Sparse table multiplicity keys and counts (same length; one entry per non-zero cell).
    pub table_tokens: Vec<u32>,
    pub table_cols: Vec<u32>,
    pub multiplicities: Vec<u32>,
}

/// Per-layer proof in the GKR protocol.
#[derive(Debug, Clone)]
pub enum LayerProof {
    /// Sumcheck proof for matmul layer.
    /// Reduces claim on C = A×B to claims on A and B.
    MatMul {
        round_polys: Vec<RoundPoly>,
        final_a_eval: SecureField,
        final_b_eval: SecureField,
    },

    /// Element-wise add: c = a + b.
    /// No sumcheck needed — linearity gives us the split directly.
    Add {
        lhs_eval: SecureField,
        rhs_eval: SecureField,
        /// 0 = lhs is trunk (sequential walk follows lhs), 1 = rhs is trunk.
        /// In DAG circuits (residual connections), the trunk is the input with
        /// the higher layer index. For simple chains, trunk_idx = 0.
        trunk_idx: u8,
    },

    /// Element-wise multiply: c = a * b.
    /// Sound eq-sumcheck: proves Ṽ_c(r) = Σ_x eq(r,x)·Ṽ_a(x)·Ṽ_b(x).
    /// Degree-3 sumcheck (eq × a × b), with 4-point evaluation per round.
    Mul {
        eq_round_polys: Vec<RoundPolyDeg3>,
        lhs_eval: SecureField,
        rhs_eval: SecureField,
    },

    /// LogUp proof for non-linear activation (ReLU, GELU, SiLU, Sigmoid).
    /// Proves every (input, output) pair exists in the precomputed activation table.
    /// `logup_proof` is `None` when using the simple MLE-evaluation reduction path.
    Activation {
        activation_type: ActivationType,
        logup_proof: Option<LogUpProof>,
        input_eval: SecureField,
        output_eval: SecureField,
        table_commitment: starknet_ff::FieldElement,
    },

    /// Layer normalization reduction.
    /// Decomposes into:
    /// 1. Linear transform eq-sumcheck: output = (input - mean) * rsqrt
    ///    Degree-3: eq(r,x) × centered(x) × rsqrt(x).
    /// 2. rsqrt lookup (LogUp): proves (variance, rsqrt) ∈ rsqrt_table.
    LayerNorm {
        logup_proof: Option<LogUpProof>,
        /// Degree-3 eq-sumcheck for output = centered × rsqrt.
        linear_round_polys: Vec<RoundPolyDeg3>,
        /// Final evaluations at the linear sumcheck challenge point s:
        /// (centered(s), rsqrt(s)).
        linear_final_evals: (SecureField, SecureField),
        input_eval: SecureField,
        output_eval: SecureField,
        /// Ṽ_mean(r) — mean MLE evaluated at the claim point.
        mean: SecureField,
        /// Ṽ_rsqrt(r) — rsqrt MLE evaluated at the claim point.
        rsqrt_var: SecureField,
        /// Poseidon commitment to the rsqrt lookup table.
        rsqrt_table_commitment: starknet_ff::FieldElement,
        /// True if this was produced by the SIMD combined-product path.
        /// SIMD path skips LogUp since combined variance/rsqrt are QM31
        /// sums that don't map to individual table entries.
        simd_combined: bool,
    },

    /// RMS normalization reduction.
    /// Like LayerNorm but simpler: output = input × rsqrt(mean(x²))
    /// 1. Degree-2 eq-sumcheck: output = input × rsqrt
    /// 2. rsqrt lookup (LogUp): proves (rms_sq, rsqrt) ∈ rsqrt_table
    RMSNorm {
        logup_proof: Option<LogUpProof>,
        /// Degree-3 eq-sumcheck for output = input × rsqrt.
        linear_round_polys: Vec<RoundPolyDeg3>,
        /// Final evaluations at the linear sumcheck challenge point s:
        /// (input(s), rsqrt(s)).
        linear_final_evals: (SecureField, SecureField),
        input_eval: SecureField,
        output_eval: SecureField,
        /// Ṽ_rms²(r) — rms² MLE evaluated at the claim point.
        rms_sq_eval: SecureField,
        /// Ṽ_rsqrt(r) — rsqrt MLE evaluated at the claim point.
        rsqrt_eval: SecureField,
        /// Poseidon commitment to the rsqrt lookup table.
        rsqrt_table_commitment: starknet_ff::FieldElement,
        /// True if this was produced by the SIMD combined-product path.
        simd_combined: bool,
    },

    /// LogUp proof for dequantization (quantized → dequantized lookup).
    /// Same protocol as Activation but with a 2-element relation (input, output).
    Dequantize {
        logup_proof: Option<LogUpProof>,
        input_eval: SecureField,
        output_eval: SecureField,
        table_commitment: starknet_ff::FieldElement,
    },

    /// LogUp proof for quantization (input → quantized output lookup).
    ///
    /// The table is layer-instance dependent (built from observed input values),
    /// so the proof carries explicit table columns.
    Quantize {
        logup_proof: Option<LogUpProof>,
        input_eval: SecureField,
        output_eval: SecureField,
        table_inputs: Vec<M31>,
        table_outputs: Vec<M31>,
    },

    /// LogUp proof for embedding lookup (token_id, col_idx, value).
    ///
    /// Verifier reconstructs table values from model weights and checks sparse
    /// multiplicity balance.
    Embedding {
        logup_proof: Option<EmbeddingLogUpProof>,
        input_eval: SecureField,
        output_eval: SecureField,
        /// Number of variables in the projected embedding-input claim.
        input_num_vars: usize,
    },

    /// Block-extended 3-factor sumcheck for SIMD matmul where both operands vary.
    /// Used for attention per-head score/context matmuls across SIMD blocks.
    ///
    /// Proves: Σ_b w_b · Σ_k A_b(r_row, k) · B_b(k, r_col) = claimed_value
    ///
    /// Constructs extended MLEs of length N×K (blocks × inner dimension):
    ///   ext_w[b*K + k] = block_weight[b]
    ///   ext_a[b*K + k] = Ṽ_{A_b}(r_row, k)
    ///   ext_b[b*K + k] = Ṽ_{B_b}(k, r_col)
    ///
    /// Then runs a standard 3-factor eq-sumcheck: Σ_i ext_w[i]·ext_a[i]·ext_b[i].
    MatMulDualSimd {
        round_polys: Vec<RoundPolyDeg3>,
        final_a_eval: SecureField,
        final_b_eval: SecureField,
        /// log2(num_blocks) — number of block-variable rounds.
        /// Verifier uses this to split sumcheck challenges into block and k parts.
        n_block_vars: usize,
    },

    /// Attention block (decomposed into sub-layer proofs).
    ///
    /// Each sub-matmul gets a fresh random evaluation point drawn from the
    /// Fiat-Shamir channel. The `sub_claim_values` store the output MLE
    /// evaluation at that point for each sub-proof, allowing the verifier to
    /// reconstruct the claims without access to intermediate matrices.
    Attention {
        sub_proofs: Vec<LayerProof>,
        /// Output MLE evaluations for each sub-matmul's fresh claim.
        sub_claim_values: Vec<SecureField>,
    },
}

/// Weight claim from a MatMul layer: the evaluation point and expected value
/// for the weight MLE. Used to bind the sumcheck to the registered weight matrix.
#[derive(Debug, Clone)]
pub struct WeightClaim {
    /// MatMul weight node id in the compiled circuit graph.
    pub weight_node_id: usize,
    /// Evaluation point in the weight MLE: [r_j || sumcheck_challenges]
    /// where r_j = output column challenges, sumcheck_challenges from reduction.
    pub eval_point: Vec<SecureField>,
    /// Expected value: final_b_eval from the MatMul sumcheck.
    pub expected_value: SecureField,
}

/// Transcript strategy for weight opening proofs.
///
/// - `Sequential`: legacy mode; one shared channel threaded through all openings.
/// - `BatchedSubchannelV1`: draws one master seed from the main channel and derives
///   independent per-opening sub-channels, enabling batched/parallel opening proving.
/// - `BatchedRlcDirectEvalV1`: skips per-weight Merkle openings and binds all
///   weight claims with a single random-linear-combination check against model
///   weights (off-chain verifier path).
/// - `AggregatedTrustlessV2`: Phase-3 trustless on-chain binding mode.
///   Current implementation keeps full opening proofs while adding
///   mode-2 binding payload checks in the Starknet v3 verifier path.
///   Openings use per-opening sub-channel transcripts for deterministic
///   parallelizable proving/verifying.
/// - `AggregatedOpeningsV4Experimental`: experimental mode-3 envelope for
///   Starknet v4 integration scaffolding. Current implementation still uses
///   full opening proofs and emits mode-3 binding metadata.
/// - `AggregatedOracleSumcheck`: unified oracle mismatch sumcheck — all M
///   weight claims aggregated into one sumcheck + one MLE opening. Produces
///   ~17K felts calldata instead of ~2.4M. This is the production submit mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightOpeningTranscriptMode {
    Sequential,
    BatchedSubchannelV1,
    BatchedRlcDirectEvalV1,
    AggregatedTrustlessV2,
    AggregatedOpeningsV4Experimental,
    AggregatedOracleSumcheck,
}

/// Deterministically derive a per-opening sub-channel from a master seed.
///
/// Domain separation includes opening index, eval point, and expected value.
///
/// Note: `weight_node_id` is intentionally excluded so the transcript can be
/// reproduced by on-chain verifiers from serialized weight claims alone.
pub(crate) fn derive_weight_opening_subchannel(
    master_seed: starknet_ff::FieldElement,
    opening_index: usize,
    claim: &WeightClaim,
) -> PoseidonChannel {
    let mut ch = PoseidonChannel::new();
    ch.mix_felt(master_seed);
    ch.mix_u64(opening_index as u64);
    ch.mix_felts(&claim.eval_point);
    ch.mix_felt(securefield_to_felt(claim.expected_value));
    ch
}

/// Complete GKR proof for a full model forward pass.
#[derive(Debug, Clone)]
pub struct GKRProof {
    /// Per-layer proofs, ordered output → input.
    pub layer_proofs: Vec<LayerProof>,

    /// Initial claim on the model output MLE (start of GKR walk).
    pub output_claim: GKRClaim,

    /// Final claim on the model input MLE (end of GKR walk).
    pub input_claim: GKRClaim,

    /// Weight commitments (Poseidon Merkle roots), one per MatMul layer.
    pub weight_commitments: Vec<starknet_ff::FieldElement>,

    /// MLE opening proofs for weight matrices, one per MatMul layer.
    /// Generated AFTER the GKR walk using the post-walk channel state.
    /// Proves that final_b_eval == MLE(weight, eval_point) for each MatMul.
    pub weight_openings: Vec<MleOpeningProof>,

    /// Weight claims: (eval_point, expected_value) per MatMul layer.
    /// Used by the verifier to check weight_openings[i].final_value == weight_claims[i].expected_value.
    pub weight_claims: Vec<WeightClaim>,

    /// Transcript mode used for generating/verifying `weight_openings`.
    pub weight_opening_transcript_mode: WeightOpeningTranscriptMode,

    /// IO commitment binding model input and output.
    pub io_commitment: starknet_ff::FieldElement,

    /// Deferred proofs for rhs branches of DAG Add layers (residual connections).
    /// When Add inputs come from different paths, the main walk follows the lhs
    /// branch and each rhs branch gets a separate deferred proof.
    pub deferred_proofs: Vec<DeferredProof>,

    /// Aggregated weight binding proof (mode = AggregatedOracleSumcheck).
    /// When present, `weight_openings` is empty — all claims are proven
    /// in one shot via a mismatch sumcheck + single MLE opening.
    pub aggregated_binding: Option<AggregatedWeightBindingProof>,
}

/// Discriminant for deferred proof kind: weight-bearing (MatMul) vs weightless
/// (Quantize/Dequantize — LogUp layers with no weight matrix).
#[derive(Debug, Clone)]
pub enum DeferredProofKind {
    /// MatMul deferred proof — carries weight data (commitment, opening, claim, dims).
    MatMul {
        /// MatMul dimensions (m, k, n).
        dims: (usize, usize, usize),
        /// Weight commitment (Poseidon Merkle root).
        weight_commitment: starknet_ff::FieldElement,
        /// Weight MLE opening proof.
        weight_opening: MleOpeningProof,
        /// Weight claim.
        weight_claim: WeightClaim,
    },
    /// Weightless deferred proof (Quantize, Dequantize) — no weight matrix.
    Weightless,
}

/// Deferred proof for the rhs branch of a DAG Add layer.
///
/// In residual connections (e.g., output = FFN(x) + x), the Add layer's
/// inputs come from different branches. The main GKR walk follows the lhs
/// branch. The rhs branch gets a separate deferred proof.
///
/// For MatMul branches, this is a matmul sumcheck proof binding `rhs_eval`
/// to the weights and model input — self-contained with its own weight
/// commitment and opening proof.
///
/// For Quantize/Dequantize branches, this is a LogUp proof with no weight
/// data (weightless).
#[derive(Debug, Clone)]
pub struct DeferredProof {
    /// The deferred claim: "rhs branch MLE at point r equals rhs_eval"
    pub claim: GKRClaim,

    /// Layer reduction proof for the deferred branch.
    pub layer_proof: LayerProof,

    /// Final input claim from the deferred reduction.
    pub input_claim: GKRClaim,

    /// Kind of deferred proof: MatMul (with weight data) or Weightless.
    pub kind: DeferredProofKind,
}

impl DeferredProof {
    /// Returns the weight claim if this is a MatMul deferred proof.
    pub fn weight_claim(&self) -> Option<&WeightClaim> {
        match &self.kind {
            DeferredProofKind::MatMul { weight_claim, .. } => Some(weight_claim),
            DeferredProofKind::Weightless => None,
        }
    }

    /// Returns the weight commitment if this is a MatMul deferred proof.
    pub fn weight_commitment(&self) -> Option<starknet_ff::FieldElement> {
        match &self.kind {
            DeferredProofKind::MatMul {
                weight_commitment, ..
            } => Some(*weight_commitment),
            DeferredProofKind::Weightless => None,
        }
    }

    /// Returns a mutable reference to the weight commitment if this is a MatMul deferred proof.
    pub fn weight_commitment_mut(&mut self) -> Option<&mut starknet_ff::FieldElement> {
        match &mut self.kind {
            DeferredProofKind::MatMul {
                weight_commitment, ..
            } => Some(weight_commitment),
            DeferredProofKind::Weightless => None,
        }
    }

    /// Returns the weight opening if this is a MatMul deferred proof.
    pub fn weight_opening(&self) -> Option<&MleOpeningProof> {
        match &self.kind {
            DeferredProofKind::MatMul { weight_opening, .. } => Some(weight_opening),
            DeferredProofKind::Weightless => None,
        }
    }

    /// Returns mutable reference to the weight opening if MatMul.
    pub fn weight_opening_mut(&mut self) -> Option<&mut MleOpeningProof> {
        match &mut self.kind {
            DeferredProofKind::MatMul { weight_opening, .. } => Some(weight_opening),
            DeferredProofKind::Weightless => None,
        }
    }

    /// Returns the MatMul dimensions (m, k, n) if this is a MatMul deferred proof.
    pub fn dims(&self) -> Option<(usize, usize, usize)> {
        match &self.kind {
            DeferredProofKind::MatMul { dims, .. } => Some(*dims),
            DeferredProofKind::Weightless => None,
        }
    }

    /// Returns true if this deferred proof carries weight data.
    pub fn has_weights(&self) -> bool {
        matches!(self.kind, DeferredProofKind::MatMul { .. })
    }
}

/// Intermediate claim produced by a layer reduction.
/// Maps from output claim to one or two input claims.
#[derive(Debug, Clone)]
pub enum ReductionOutput {
    /// Single input claim (MatMul, Activation, LayerNorm).
    Single(GKRClaim),

    /// Two input claims (Add, Mul — one for each operand).
    Split {
        lhs_claim: GKRClaim,
        rhs_claim: GKRClaim,
    },
}

/// Errors in the GKR protocol.
#[derive(Debug, thiserror::Error)]
pub enum GKRError {
    #[error("circuit compilation failed: {0}")]
    CompilationError(String),

    #[error("layer {layer_idx} reduction failed: {reason}")]
    ReductionError { layer_idx: usize, reason: String },

    #[error("verification failed at layer {layer_idx}: {reason}")]
    VerificationError { layer_idx: usize, reason: String },

    #[error("SIMD batch error: {0}")]
    SimdError(String),

    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },

    #[error("missing weight for node {node_id}")]
    MissingWeight { node_id: usize },

    #[error("missing intermediate for node {node_id}")]
    MissingIntermediate { node_id: usize },

    #[error("sumcheck error: {0}")]
    SumcheckError(String),

    #[error("logup error: {0}")]
    LogUpError(String),

    #[error("lookup table error: {0}")]
    LookupTableError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::QM31;

    #[test]
    fn test_compressed_round_poly_deg3_roundtrip() {
        let rp = RoundPolyDeg3 {
            c0: QM31(
                CM31(M31::from(10), M31::from(20)),
                CM31(M31::from(30), M31::from(40)),
            ),
            c1: QM31(
                CM31(M31::from(50), M31::from(60)),
                CM31(M31::from(70), M31::from(80)),
            ),
            c2: QM31(
                CM31(M31::from(90), M31::from(100)),
                CM31(M31::from(110), M31::from(120)),
            ),
            c3: QM31(
                CM31(M31::from(130), M31::from(140)),
                CM31(M31::from(150), M31::from(160)),
            ),
        };

        // current_sum = p(0) + p(1) = c0 + (c0 + c1 + c2 + c3)
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;
        let current_sum = p0 + p1;

        let compressed = rp.compress();
        let decompressed = compressed.decompress(current_sum);

        assert_eq!(decompressed.c0, rp.c0, "c0 mismatch");
        assert_eq!(decompressed.c1, rp.c1, "c1 mismatch after reconstruction");
        assert_eq!(decompressed.c2, rp.c2, "c2 mismatch");
        assert_eq!(decompressed.c3, rp.c3, "c3 mismatch");
    }

    #[test]
    fn test_round_poly_deg3_eval() {
        let rp = RoundPolyDeg3 {
            c0: QM31(
                CM31(M31::from(1), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            c1: QM31(
                CM31(M31::from(2), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            c2: QM31(
                CM31(M31::from(3), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            c3: QM31(
                CM31(M31::from(4), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
        };

        // p(0) = 1, p(1) = 1+2+3+4 = 10
        let t0 = SecureField::from(M31::from(0));
        let t1 = SecureField::from(M31::from(1));
        assert_eq!(rp.eval(t0), rp.c0);
        let expected_p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;
        assert_eq!(rp.eval(t1), expected_p1);
    }
}
