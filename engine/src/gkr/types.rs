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

/// Proof that the softmax normalization sum is correct for one attention head.
///
/// For a score matrix of shape (seq_len × seq_len), proves that for each row r:
///   sum_exp[r] = Σ_{col} exp(scores[r][col])
///
/// Uses a plain sumcheck over the exp MLE, with eq-weighting to bind
/// per-row sums to the claimed sum_exp values. The verifier can then
/// confirm that `softmax_weight[i] = exp[i] / sum_exp[row_of_i]`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftmaxSumProof {
    /// Round polynomials from the plain sumcheck proving Σ exp = sum_exp.
    /// Length = log(seq_len) (number of column variables).
    pub round_polys: Vec<RoundPoly>,
    /// Final evaluation of the exp MLE at the sumcheck challenge point.
    pub final_exp_eval: SecureField,
    /// Final evaluation of the sum_exp MLE (broadcast per row) at the challenge point.
    pub final_sum_eval: SecureField,
    /// Total claimed sum (before eq-weighting).
    pub claimed_sum: SecureField,
    /// Per-row sum_exp values (M31), used for verification binding.
    pub row_sums: Vec<M31>,
}

/// A claim in the GKR protocol: "the MLE evaluated at `point` equals `value`."
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Multiplicity sumcheck proof for large-table LogUp layers.
///
/// Instead of sending raw multiplicities (up to 65K entries), the prover runs
/// a degree-1 sumcheck over the multiplicity MLE to prove that the table-side
/// sum is consistent with the trace-side claimed sum. ~35 felts per layer.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiplicitySumcheckProof {
    /// Degree-1 round polynomials (n_vars rounds, 2 coefficients each).
    /// p(t) = c0 + c1*t, so p(0)=c0, p(1)=c0+c1.
    pub round_polys: Vec<(SecureField, SecureField)>,
    /// Final evaluation of multiplicity MLE at the challenge point.
    pub final_eval: SecureField,
    /// Claimed sum: should equal the trace-side LogUp sum for balanced LogUp.
    pub claimed_sum: SecureField,
}

/// LogUp proof for embedding lookups: (token_id, column, value) relation.
///
/// Unlike activation/dequantize, the embedding table is model-dependent, so the
/// proof carries sparse table multiplicities keyed by `(token_id, column)`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Algebraic product+binary eq-sumcheck proof for activation layers (ReLU).
///
/// Proves that `output = input * indicator` where `indicator ∈ {0,1}^n` via a
/// single degree-3 sumcheck combining:
///   1. Product constraint: `Σ eq(r,x)·b(x)·in(x) = V_out(r)`
///   2. Binary constraint:  `Σ eq(r,x)·b(x)·(1-b(x)) = 0`
///
/// Phase B extends this with sign consistency via bit decomposition:
///   3. Decomposition: `input - Σ 2^j·bit_j - 2^30·(1-indicator) = 0`
///   4. Binary bits:   `bit_j·(1-bit_j) = 0` for j=0..29
///
/// Combined with random linear combination powers `η^0..η^32`:
///   `Σ eq(r,x) · [b·in + η·b·(1-b) + η²·decomp + Σ η^{j+3}·bit_j·(1-bit_j)] = V_out(r)`
///
/// Degree 3: eq × bit × (1-bit) — three linear factors after folding.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ActivationProductProof {
    /// Degree-3 round polynomials from the combined eq-sumcheck.
    pub round_polys: Vec<RoundPolyDeg3>,
    /// V_input(s) — input MLE evaluated at sumcheck challenge point.
    pub input_eval: SecureField,
    /// V_b(s) — indicator MLE evaluated at sumcheck challenge point.
    pub indicator_eval: SecureField,
    /// Phase B: 30 bit MLE evaluations at the sumcheck challenge point.
    /// `None` = Phase A-only (backward compat). `Some(vec![...])` = Phase A+B.
    pub bit_evals: Option<Vec<SecureField>>,
}

/// Piecewise-linear algebraic activation proof.
///
/// Proves activation correctness via 16-segment linear approximation over the
/// full M31 domain. Uses a combined degree-3 eq-sumcheck with 18 constraints:
///   - η^0: output matches piecewise evaluation
///   - η^1: partition of unity (indicators sum to 1)
///   - η^{2..17}: binary indicator enforcement (I_i ∈ {0,1})
///
/// Total degree: 3 (eq × indicator × (1 - indicator)).
/// Eliminates lookup tables — ~82% calldata reduction vs LogUp.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PiecewiseAlgebraicProof {
    /// Degree-3 round polynomials from the combined eq-sumcheck (one per variable).
    pub round_polys: Vec<RoundPolyDeg3>,
    /// V_input(s) — input MLE evaluated at the final sumcheck challenge point.
    pub input_eval: SecureField,
    /// V_output(s) — output MLE evaluated at the final sumcheck challenge point.
    pub output_eval: SecureField,
    /// V_{I_i}(s) — indicator MLE evaluations at the final sumcheck challenge point.
    pub indicator_evals: [SecureField; 16],
    /// Segment bit MLE evaluations at the final sumcheck challenge point.
    /// 4 bits encoding the segment index (top 4 bits of input).
    /// None for legacy proofs without segment-input binding.
    pub seg_bit_evals: Option<[SecureField; 4]>,
}

/// Per-layer proof in the GKR protocol.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        /// Multiplicity sumcheck proof for table-side sum verification.
        /// Present when the table is too large to inline multiplicities.
        multiplicity_sumcheck: Option<MultiplicitySumcheckProof>,
        /// Algebraic product+binary eq-sumcheck proof (Phase A soundness).
        /// Present for ReLU: proves `output = input * indicator` with `indicator ∈ {0,1}`.
        /// `None` for non-ReLU activations and legacy proofs.
        activation_proof: Option<ActivationProductProof>,
        /// Piecewise-linear algebraic proof (GELU/Sigmoid/Softmax).
        /// Default for non-ReLU activations. Opt-out: `STWO_PIECEWISE_ACTIVATION=0`.
        piecewise_proof: Option<PiecewiseAlgebraicProof>,
        input_eval: SecureField,
        output_eval: SecureField,
        table_commitment: starknet_ff::FieldElement,
        /// True when this activation was proved via SIMD block-combination.
        /// SIMD combined activations operate on combined SecureField MLEs and
        /// cannot produce individual LogUp/algebraic/piecewise proofs.
        simd_combined: bool,
    },

    /// Layer normalization reduction.
    /// Decomposes into:
    /// 1. Linear transform eq-sumcheck: output = (input - mean) * rsqrt
    ///    Degree-3: eq(r,x) × centered(x) × rsqrt(x).
    /// 2. rsqrt lookup (LogUp): proves (variance, rsqrt) ∈ rsqrt_table.
    LayerNorm {
        logup_proof: Option<LogUpProof>,
        /// Multiplicity sumcheck proof for table-side sum verification.
        multiplicity_sumcheck: Option<MultiplicitySumcheckProof>,
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
        /// Part 0: Batched mean + variance plain sumcheck round polynomials.
        /// Proves: Σ_x [η·input(x) + η²·centered(x)²] = η·total_input_sum + η²·total_centered_sq_sum
        mean_var_round_polys: Option<Vec<RoundPolyDeg3>>,
        /// Final evaluations at the mean/variance sumcheck challenge point: (input(s₀), mean(s₀)).
        mean_var_final_evals: Option<(SecureField, SecureField)>,
        /// Ṽ_variance(r) — variance MLE evaluated at the claim point.
        var_eval: Option<SecureField>,
        /// Centered-consistency binding evals at Part 1 challenge: (input(s₁), mean(s₁)).
        centered_binding_evals: Option<(SecureField, SecureField)>,
        /// Claimed sums for plain sumcheck: (total_input_sum, total_centered_sq_sum).
        mv_claimed_sums: Option<(SecureField, SecureField)>,
        /// Number of active columns (for Part 0 channel mixing: mix_u64(n_active)).
        /// Serialized into Part 0 so the verifier can replay `mix_u64(n_active)`.
        n_active: Option<usize>,
        /// Per-row mean values for multi-row binding verification.
        /// When rows > 1, the verifier uses these to reconstruct mean(s₀)
        /// from the Part 0 sumcheck challenge, preventing fake per-row means.
        row_means: Option<Vec<M31>>,
        /// Per-row variance values for multi-row variance binding verification.
        /// When rows > 1, the verifier uses these to reconstruct var_eval
        /// from the Part 0 sumcheck challenge, preventing fake per-row variances.
        row_variances: Option<Vec<M31>>,
    },

    /// RMS normalization reduction.
    /// Like LayerNorm but simpler: output = input × rsqrt(mean(x²))
    /// 1. Degree-2 eq-sumcheck: output = input × rsqrt
    /// 2. rsqrt lookup (LogUp): proves (rms_sq, rsqrt) ∈ rsqrt_table
    RMSNorm {
        logup_proof: Option<LogUpProof>,
        /// Multiplicity sumcheck proof for table-side sum verification.
        multiplicity_sumcheck: Option<MultiplicitySumcheckProof>,
        /// Degree-3 eq-sumcheck for output = input × rsqrt (× γ if present).
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
        /// Poseidon commitment to the learned affine scale vector γ.
        /// When present, the verifier checks that the committed γ matches
        /// the model registration. When `None`, the norm layer has no
        /// learned scale (raw normalization only).
        gamma_commitment: Option<starknet_ff::FieldElement>,
        /// Ṽ_γ(s) — gamma MLE evaluated at the linear sumcheck challenge point.
        /// Present when `gamma_commitment` is `Some`.
        gamma_eval: Option<SecureField>,
        /// Part 0: RMS² verification plain sumcheck round polynomials.
        /// Proves: Σ_x input(x)² = total_sq_sum (no eq weighting).
        rms_sq_round_polys: Option<Vec<RoundPolyDeg3>>,
        /// Final input(s₀) evaluation at the RMS² sumcheck challenge point.
        rms_sq_input_final: Option<SecureField>,
        /// Claimed total sum of input squares: Σ input(x)².
        /// Verifier uses this to derive rms_sq and check against rms_sq_eval.
        rms_sq_claimed_sq_sum: Option<SecureField>,
        /// Number of active (un-padded) columns for the RMS² sumcheck.
        /// Serialized into Part 0 so the verifier can replay `mix_u64(n_active)`.
        rms_sq_n_active: Option<usize>,
        /// Per-row rms² values for multi-row binding verification.
        /// When rows > 1, the verifier uses these to reconstruct rms_sq(s₀)
        /// from the Part 0 sumcheck challenge, preventing fake per-row rms².
        row_rms_sq: Option<Vec<M31>>,
    },

    /// LogUp proof for dequantization (quantized → dequantized lookup).
    /// Same protocol as Activation but with a 2-element relation (input, output).
    Dequantize {
        logup_proof: Option<LogUpProof>,
        /// Multiplicity sumcheck proof for table-side sum verification.
        multiplicity_sumcheck: Option<MultiplicitySumcheckProof>,
        input_eval: SecureField,
        output_eval: SecureField,
        table_commitment: starknet_ff::FieldElement,
        /// Quantization bit-width (mixed into Fiat-Shamir channel for domain separation).
        bits: u32,
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
        /// Quantization bit-width (mixed into channel).
        bits: u32,
        /// |zero_point| (mixed into channel).
        zero_point_abs: u32,
        /// scale × 2³² truncated to u64, serialized as two u32s (mixed into channel).
        scale_fixed: u64,
        /// Strategy tag: 0=Direct, 1=Sym8, 2=Asym8, 3=Sym4, 4=Asym4 (mixed into channel).
        strategy_tag: u32,
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
        /// Vocabulary size (mixed into channel for replay).
        vocab_size: u32,
        /// Embedding dimension (mixed into channel for replay).
        embed_dim: u32,
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
        /// Number of attention heads (mixed into channel for replay).
        num_heads: u32,
        /// Sequence length (mixed into channel for replay).
        seq_len: u32,
        /// Model dimension (mixed into channel for replay).
        d_model: u32,
        /// Whether causal masking is applied (mixed into channel for replay).
        causal: bool,
        /// Per-head softmax normalization sum proofs.
        /// Each entry proves: sum_exp[row] = Σ_col exp(scores[row][col]) for one head.
        /// Without this, the prover could claim arbitrary softmax weights.
        softmax_sum_proofs: Vec<SoftmaxSumProof>,
    },

    /// Attention decode step (cached KV, sub-matmuls have asymmetric dims).
    ///
    /// Like `Attention`, but sub-matmul dimensions differ from prefill:
    /// - Score: (new_tokens, d_k, full_seq_len)
    /// - Context: (new_tokens, full_seq_len, d_k)
    /// - Projections: (new_tokens, d_model, d_model)
    AttentionDecode {
        sub_proofs: Vec<LayerProof>,
        sub_claim_values: Vec<SecureField>,
        num_heads: usize,
        new_tokens: usize,
        full_seq_len: usize,
        d_model: usize,
        causal: bool,
        position_offset: usize,
    },

    /// Top-K expert selection proof for MoE routing.
    ///
    /// Proves that the K selected experts have the K largest router logits.
    /// The proof consists of:
    /// 1. Selected indices + values (K entries)
    /// 2. Rejected indices + values (N-K entries)
    /// 3. Threshold proof: min(selected) ≥ max(rejected) via range check
    /// 4. Completeness: all N indices covered exactly once
    ///
    /// The router logits MLE binding is done via the parent MatMul layer proof
    /// (the router projection is a standard MatMul → TopK takes its output).
    TopK {
        /// Number of experts (N).
        num_experts: usize,
        /// Number selected (K).
        top_k: usize,
        /// Selected expert indices.
        selected_indices: Vec<u32>,
        /// Selected expert logit values (as SecureField for channel mixing).
        selected_values: Vec<SecureField>,
        /// min(selected) - max(rejected), must be non-negative.
        /// Encoded as M31 value in [0, P/2] (non-negative signed range).
        threshold_gap: SecureField,
        /// Poseidon commitment to the full router logits vector.
        /// The verifier checks this matches the router MatMul output.
        logits_commitment: starknet_ff::FieldElement,
    },
}

/// Weight claim from a MatMul layer: the evaluation point and expected value
/// for the weight MLE. Used to bind the sumcheck to the registered weight matrix.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// For batch proofs, this is the batch IO commitment (Poseidon of N individual commitments).
    pub io_commitment: starknet_ff::FieldElement,

    /// Deferred proofs for rhs branches of DAG Add layers (residual connections).
    /// When Add inputs come from different paths, the main walk follows the lhs
    /// branch and each rhs branch gets a separate deferred proof.
    pub deferred_proofs: Vec<DeferredProof>,

    /// Aggregated weight binding proof (mode = AggregatedOracleSumcheck).
    /// When present, `weight_openings` is empty — all claims are proven
    /// in one shot via a mismatch sumcheck + single MLE opening.
    /// For ≤ GROUP_SIZE matrices, this holds the single binding proof.
    pub aggregated_binding: Option<AggregatedWeightBindingProof>,

    /// Grouped binding proofs for large models (> GROUP_SIZE matrices).
    /// Each group covers one transformer layer's weight matrices (typically 4).
    /// When non-empty, `aggregated_binding` is None — the verifier uses
    /// these groups instead, verifying each independently through the same
    /// Fiat-Shamir channel.
    pub binding_groups: Vec<AggregatedWeightBindingProof>,

    /// KV-cache commitment (Poseidon hash of current KV state after this inference).
    /// Present when the proof covers a model with KV-cache (autoregressive decoding).
    pub kv_cache_commitment: Option<starknet_ff::FieldElement>,

    /// Previous KV-cache commitment (for chaining sequential inferences).
    /// Present when `kv_cache_commitment` is set.
    pub prev_kv_cache_commitment: Option<starknet_ff::FieldElement>,
}

/// Discriminant for deferred proof kind: weight-bearing (MatMul) vs weightless
/// (Quantize/Dequantize — LogUp layers with no weight matrix).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    #[error("verification failed: {0}")]
    VerificationFailed(String),
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
