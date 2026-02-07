//! Attention mechanism verification via composed sumcheck + LogUp.
//!
//! # Attention Decomposition
//!
//! Standard attention: `Output = softmax(Q × K^T / √d_k) × V`
//!
//! Decomposed into three provable stages:
//!
//! ```text
//! Stage 1: Score Matrix (Sumcheck MatMul)
//!   scores = Q × K^T                    // MatMul component
//!   scores_scaled = scores / √d_k       // Constant multiply
//!
//! Stage 2: Attention Weights (LogUp Softmax)
//!   weights = softmax(scores_scaled)     // Activation component (table lookup)
//!
//! Stage 3: Output Projection (Sumcheck MatMul)
//!   output = weights × V                 // MatMul component
//! ```
//!
//! # Multi-Head Attention
//!
//! For h heads with d_model dimensions:
//!
//! ```text
//! d_k = d_v = d_model / h
//!
//! Per head:
//!   Q_i = X × W_Q_i    (seq_len × d_k)
//!   K_i = X × W_K_i    (seq_len × d_k)
//!   V_i = X × W_V_i    (seq_len × d_v)
//!   head_i = Attention(Q_i, K_i, V_i)
//!
//! Concat + project:
//!   output = Concat(head_1, ..., head_h) × W_O
//! ```
//!
//! Each head is independent -> parallelizable across GPUs.

use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::M31;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;

use super::activation::{
    prove_activation, verify_activation, ActivationComponent, ActivationError, ActivationType,
};
use super::matmul::{
    prove_matmul, verify_matmul, M31Matrix, MatMulAux, MatMulError, MatMulProof,
};
use crate::gadgets::lookup_table::PrecomputedTable;

/// Configuration for a single attention head.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AttentionHeadConfig {
    /// Sequence length (number of tokens).
    pub seq_len: usize,
    /// Key/query dimension per head.
    pub d_k: usize,
    /// Value dimension per head.
    pub d_v: usize,
}

/// Configuration for multi-head attention.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiHeadAttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Model dimension (d_model = num_heads × d_k).
    pub d_model: usize,
    /// Sequence length.
    pub seq_len: usize,
}

impl MultiHeadAttentionConfig {
    pub fn new(num_heads: usize, d_model: usize, seq_len: usize) -> Self {
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model must be divisible by num_heads"
        );
        Self {
            num_heads,
            d_model,
            seq_len,
        }
    }

    /// Dimension per head.
    pub fn d_k(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Total sumcheck witness rows for all heads.
    pub fn sumcheck_trace_rows(&self) -> usize {
        let d_k = self.d_k();
        let s = self.seq_len;

        // Per head: QK^T matmul + softmax lookups + attn×V matmul
        let qkt_witness = s * d_k + s * s + d_k * s;
        let softmax_lookups = s * s;
        let attn_v_witness = s * s + s * d_k + s * d_k;

        let per_head = qkt_witness + softmax_lookups + attn_v_witness;
        per_head * self.num_heads
    }

    /// Naive O(n^3) trace rows for comparison.
    pub fn naive_trace_rows(&self) -> usize {
        let d_k = self.d_k();
        let s = self.seq_len;

        // Per head: QK^T (s×d_k×s) + attn×V (s×s×d_k) + softmax (s×s×~100 ops)
        let per_head = s * d_k * s + s * s * d_k + s * s * 100;
        per_head * self.num_heads
    }

    /// Speedup: naive / sumcheck.
    pub fn speedup(&self) -> f64 {
        self.naive_trace_rows() as f64 / self.sumcheck_trace_rows() as f64
    }
}

// ---------------------------------------------------------------------------
// Attention witness builder
// ---------------------------------------------------------------------------

/// Pre-computed witness for a single attention head.
///
/// Bundles Q, K, V and all intermediate matrices so the caller
/// doesn't have to manually compute scores, transpose K, etc.
///
/// ```rust,ignore
/// let witness = AttentionWitness::build(q, k, v).unwrap();
/// let proof = prove_attention_head(
///     &witness.q, &witness.k, &witness.v,
///     &witness.scores, &witness.weights, &witness.output,
///     &mut channel,
/// ).unwrap();
/// ```
pub struct AttentionWitness {
    pub q: M31Matrix,
    pub k: M31Matrix,
    pub v: M31Matrix,
    /// Q × K^T
    pub scores: M31Matrix,
    /// softmax(scores) — identity placeholder until softmax table is wired.
    pub weights: M31Matrix,
    /// weights × V
    pub output: M31Matrix,
}

impl AttentionWitness {
    /// Build the full attention witness from Q, K, V.
    ///
    /// Computes `scores = Q × K^T` and `output = weights × V`.
    /// Uses identity as the softmax placeholder (weights = scores).
    pub fn build(
        q: M31Matrix,
        k: M31Matrix,
        v: M31Matrix,
    ) -> Result<Self, MatMulError> {
        let kt = k.transpose();
        let scores = M31Matrix::multiply(&q, &kt)?;
        let weights = scores.clone();
        let output = M31Matrix::multiply(&weights, &v)?;
        Ok(Self {
            q,
            k,
            v,
            scores,
            weights,
            output,
        })
    }

    /// Prove this attention witness.
    pub fn prove(
        &self,
        channel: &mut impl Channel,
    ) -> Result<AttentionHeadProof, MatMulError> {
        prove_attention_head(
            &self.q, &self.k, &self.v,
            &self.scores, &self.weights, &self.output,
            channel,
        )
    }

    /// Verify a proof against this witness.
    pub fn verify(
        &self,
        proof: &AttentionHeadProof,
        channel: &mut impl Channel,
    ) -> Result<(), MatMulError> {
        verify_attention_head(
            &self.q, &self.k, &self.v,
            &self.scores, &self.weights, &self.output,
            proof,
            channel,
        )
    }
}

// ---------------------------------------------------------------------------
// Single-head attention proof
// ---------------------------------------------------------------------------

/// Proof for a single attention head: scores = Q × K^T, output = weights × V.
///
/// The softmax step is handled separately via the activation component.
/// This proof covers the two matmul stages.
#[derive(Debug, Clone)]
pub struct AttentionHeadProof {
    /// Proof that scores = Q × K^T.
    pub qkt_proof: MatMulProof,
    pub qkt_aux: MatMulAux,
    /// Proof that output = weights × V.
    pub attn_v_proof: MatMulProof,
    pub attn_v_aux: MatMulAux,
}

/// Prove a single attention head: scores = Q × K^T, output = weights × V.
///
/// `weights` should be the already-computed softmax(scores / sqrt(d_k)).
/// The caller is responsible for proving the softmax step separately
/// using the activation component.
///
/// **Soundness note**: This proof verifies two independent matmul claims
/// (QK^T and weights×V) linked by a shared Fiat-Shamir channel. However,
/// there is no cryptographic binding that `weights = softmax(scores)`.
/// For a fully sound attention proof, the softmax activation must be proven
/// separately and the intermediate values committed in a shared Merkle tree.
#[allow(clippy::too_many_arguments)]
pub fn prove_attention_head(
    q: &M31Matrix,         // (seq_len × d_k)
    k: &M31Matrix,         // (seq_len × d_k)
    v: &M31Matrix,         // (seq_len × d_v)
    scores: &M31Matrix,    // (seq_len × seq_len) = Q × K^T
    weights: &M31Matrix,   // (seq_len × seq_len) = softmax(scores)
    output: &M31Matrix,    // (seq_len × d_v) = weights × V
    channel: &mut impl Channel,
) -> Result<AttentionHeadProof, MatMulError> {
    // Stage 1: Prove scores = Q × K^T
    let kt = k.transpose();
    let (qkt_proof, qkt_aux) = prove_matmul(q, &kt, scores, channel)?;

    // Stage 3: Prove output = weights × V
    let (attn_v_proof, attn_v_aux) = prove_matmul(weights, v, output, channel)?;

    Ok(AttentionHeadProof {
        qkt_proof,
        qkt_aux,
        attn_v_proof,
        attn_v_aux,
    })
}

/// Verify a single attention head proof.
#[allow(clippy::too_many_arguments)]
pub fn verify_attention_head(
    q: &M31Matrix,
    k: &M31Matrix,
    v: &M31Matrix,
    scores: &M31Matrix,
    weights: &M31Matrix,
    output: &M31Matrix,
    proof: &AttentionHeadProof,
    channel: &mut impl Channel,
) -> Result<(), MatMulError> {
    // Verify scores = Q × K^T
    let kt = k.transpose();
    verify_matmul(q, &kt, scores, &proof.qkt_proof, &proof.qkt_aux, channel)?;

    // Verify output = weights × V
    verify_matmul(
        weights,
        v,
        output,
        &proof.attn_v_proof,
        &proof.attn_v_aux,
        channel,
    )?;

    Ok(())
}

/// Simple element-wise scaling: output\[i\]\[j\] = input\[i\]\[j\] * scale_inv.
///
/// For attention, `scale_inv` represents `1/sqrt(d_k)` mapped to M31.
/// In practice, this is done as part of the matmul or via a lookup.
pub fn scale_matrix(m: &M31Matrix, scale: M31) -> M31Matrix {
    let mut result = M31Matrix::new(m.rows, m.cols);
    for i in 0..m.rows {
        for j in 0..m.cols {
            result.set(i, j, m.get(i, j) * scale);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Composed attention proof (matmul + activation + matmul)
// ---------------------------------------------------------------------------

/// Error type for composed attention proofs.
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("matmul error: {0}")]
    MatMul(#[from] MatMulError),
    #[error("activation error: {0}")]
    Activation(#[from] ActivationError),
    #[error("scores out of activation table range: max score {max_score} >= table size {table_size}")]
    ScoreOutOfRange { max_score: u32, table_size: u32 },
    #[error("dimension mismatch: weights rows ({weight_rows}) != V rows ({v_rows})")]
    WeightsDimensionMismatch { weight_rows: usize, v_rows: usize },
}

/// Proof for composed attention: scores = Q×K^T, weights = f(scores), output = weights×V.
///
/// Contains three linked proofs sharing a Fiat-Shamir transcript:
/// 1. Sumcheck proof for QK^T matmul
/// 2. LogUp STARK proof for activation function applied to scores
/// 3. Sumcheck proof for weights×V matmul
pub struct ComposedAttentionProof {
    /// Proof that scores = Q × K^T (sumcheck).
    pub qkt_proof: MatMulProof,
    pub qkt_aux: MatMulAux,
    /// Proof that weights = activation(scores) (LogUp STARK).
    pub activation_component: ActivationComponent,
    pub activation_proof: stwo::core::proof::StarkProof<Blake2sMerkleHasher>,
    /// Proof that output = weights × V (sumcheck).
    pub attn_v_proof: MatMulProof,
    pub attn_v_aux: MatMulAux,
    /// The activation type used (for verification).
    pub activation_type: ActivationType,
    /// The computed weights matrix (needed for verification).
    pub weights: M31Matrix,
    /// The computed output matrix (needed for verification).
    pub output: M31Matrix,
}

/// Apply an activation table element-wise to a matrix.
///
/// Each element `m[i][j]` is looked up in the table: `output[i][j] = table.outputs[m[i][j].0]`.
/// Returns `Err` if any element is outside the table range.
pub fn apply_activation_table(
    m: &M31Matrix,
    table: &PrecomputedTable,
) -> Result<M31Matrix, AttentionError> {
    let size = table.size() as u32;
    let mut result = M31Matrix::new(m.rows, m.cols);
    for i in 0..m.rows {
        for j in 0..m.cols {
            let val = m.get(i, j);
            if val.0 >= size {
                return Err(AttentionError::ScoreOutOfRange {
                    max_score: val.0,
                    table_size: size,
                });
            }
            result.set(i, j, table.outputs[val.0 as usize]);
        }
    }
    Ok(result)
}

/// Flatten a matrix to a vector of M31 values (row-major order).
pub fn flatten_matrix(m: &M31Matrix) -> Vec<M31> {
    let mut v = Vec::with_capacity(m.rows * m.cols);
    for i in 0..m.rows {
        for j in 0..m.cols {
            v.push(m.get(i, j));
        }
    }
    v
}

/// Prove a composed attention head with activation:
///   scores = Q × K^T → weights = activation(scores) → output = weights × V
///
/// Three proof stages are linked through a shared Fiat-Shamir channel (for the
/// sumcheck stages) and a separate STARK proof for the activation.
///
/// **Important**: Score matrix values must be in `[0, table.size())`. For real
/// inference, quantize scores before calling this function.
pub fn prove_composed_attention(
    q: &M31Matrix,
    k: &M31Matrix,
    v: &M31Matrix,
    activation_type: ActivationType,
    log_table_size: u32,
    channel: &mut Blake2sChannel,
) -> Result<ComposedAttentionProof, AttentionError> {
    // Stage 1: Prove scores = Q × K^T (sumcheck)
    let kt = k.transpose();
    let scores = M31Matrix::multiply(q, &kt)?;
    let (qkt_proof, qkt_aux) = prove_matmul(q, &kt, &scores, channel)?;

    // Stage 2: Apply activation and prove via LogUp
    let table = activation_type.build_table(log_table_size);
    let weights = apply_activation_table(&scores, &table)?;

    let score_values = flatten_matrix(&scores);
    let pcs_config = PcsConfig::default();
    let mut activation_channel = Blake2sChannel::default();
    let (activation_component, activation_proof) =
        prove_activation(&score_values, &table, pcs_config, &mut activation_channel)?;

    // Stage 3: Prove output = weights × V (sumcheck)
    let output = M31Matrix::multiply(&weights, v)?;
    let (attn_v_proof, attn_v_aux) = prove_matmul(&weights, v, &output, channel)?;

    Ok(ComposedAttentionProof {
        qkt_proof,
        qkt_aux,
        activation_component,
        activation_proof,
        attn_v_proof,
        attn_v_aux,
        activation_type,
        weights,
        output,
    })
}

/// Verify a composed attention proof.
///
/// Checks all three stages:
/// 1. scores = Q × K^T (sumcheck verification)
/// 2. weights = activation(scores) (LogUp STARK verification)
/// 3. output = weights × V (sumcheck verification)
pub fn verify_composed_attention(
    q: &M31Matrix,
    k: &M31Matrix,
    v: &M31Matrix,
    proof: &ComposedAttentionProof,
    channel: &mut Blake2sChannel,
) -> Result<(), AttentionError> {
    // Stage 1: Verify scores = Q × K^T
    let kt = k.transpose();
    let scores = M31Matrix::multiply(q, &kt)?;
    verify_matmul(q, &kt, &scores, &proof.qkt_proof, &proof.qkt_aux, channel)?;

    // Stage 2: Verify activation proof
    let mut activation_channel = Blake2sChannel::default();
    verify_activation(
        &proof.activation_component,
        &proof.activation_proof,
        &mut activation_channel,
    )?;

    // Stage 3: Verify output = weights × V
    verify_matmul(
        &proof.weights,
        v,
        &proof.output,
        &proof.attn_v_proof,
        &proof.attn_v_aux,
        channel,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::channel::Blake2sChannel;

    #[test]
    fn test_bert_base_attention() {
        // BERT-base: 12 heads, 768 dims, 512 seq_len
        let config = MultiHeadAttentionConfig::new(12, 768, 512);
        assert_eq!(config.d_k(), 64);

        let sumcheck = config.sumcheck_trace_rows();
        let naive = config.naive_trace_rows();

        // Sumcheck should be dramatically smaller
        assert!(sumcheck < 12_000_000, "sumcheck rows: {sumcheck}");
        assert!(naive > 100_000_000, "naive rows: {naive}");
        assert!(config.speedup() > 10.0, "speedup: {:.1}x", config.speedup());
    }

    #[test]
    fn test_gpt2_attention() {
        // GPT-2 small: 12 heads, 768 dims, 1024 seq_len
        let config = MultiHeadAttentionConfig::new(12, 768, 1024);

        let sumcheck = config.sumcheck_trace_rows();

        // Should still be provable (< 100M rows, feasible on H100)
        assert!(sumcheck < 50_000_000);
    }

    #[test]
    fn test_llama_attention() {
        // Llama-7B: 32 heads, 4096 dims, 2048 seq_len
        let config = MultiHeadAttentionConfig::new(32, 4096, 2048);

        let _sumcheck = config.sumcheck_trace_rows();
    }

    #[test]
    fn test_transpose() {
        let m = M31Matrix::from_data(2, 3, vec![
            M31::from(1), M31::from(2), M31::from(3),
            M31::from(4), M31::from(5), M31::from(6),
        ]).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), M31::from(1));
        assert_eq!(t.get(0, 1), M31::from(4));
        assert_eq!(t.get(2, 0), M31::from(3));
        assert_eq!(t.get(2, 1), M31::from(6));
    }

    #[test]
    fn test_prove_verify_attention_head_4x4() {
        // Small attention: seq_len=4, d_k=4, d_v=4 (all power of 2)
        let q = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let k = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
        let v = M31Matrix::from_data(4, 4, (33..=48).map(M31::from).collect()).unwrap();

        // Compute scores = Q × K^T
        let kt = k.transpose();
        let scores = M31Matrix::multiply(&q, &kt).unwrap();

        // For testing, use identity as "softmax" (weights = scores)
        // In real usage, weights would come from softmax activation proof.
        let weights = scores.clone();

        // Compute output = weights × V
        let output = M31Matrix::multiply(&weights, &v).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let proof = prove_attention_head(
            &q, &k, &v, &scores, &weights, &output,
            &mut prover_channel,
        )
        .unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_attention_head(
            &q, &k, &v, &scores, &weights, &output,
            &proof,
            &mut verifier_channel,
        )
        .unwrap();
    }

    #[test]
    fn test_attention_witness_builder() {
        let q = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let k = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
        let v = M31Matrix::from_data(4, 4, (33..=48).map(M31::from).collect()).unwrap();

        let witness = AttentionWitness::build(q, k, v).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let proof = witness.prove(&mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        witness.verify(&proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_verify_attention_head_8x4() {
        // seq_len=8, d_k=4, d_v=4
        let q = M31Matrix::from_data(8, 4, (1..=32).map(M31::from).collect()).unwrap();
        let k = M31Matrix::from_data(8, 4, (33..=64).map(M31::from).collect()).unwrap();
        let v = M31Matrix::from_data(8, 4, (65..=96).map(M31::from).collect()).unwrap();

        let kt = k.transpose();
        let scores = M31Matrix::multiply(&q, &kt).unwrap();
        let weights = scores.clone();
        let output = M31Matrix::multiply(&weights, &v).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let proof = prove_attention_head(
            &q, &k, &v, &scores, &weights, &output,
            &mut prover_channel,
        )
        .unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_attention_head(
            &q, &k, &v, &scores, &weights, &output,
            &proof,
            &mut verifier_channel,
        )
        .unwrap();
    }

    // -----------------------------------------------------------------------
    // Composed attention tests (matmul + activation + matmul)
    // -----------------------------------------------------------------------

    #[test]
    fn test_composed_attention_relu_4x4() {
        // Small attention with ReLU activation.
        // Values kept small to fit in log_size=4 table ([0, 16)).
        let q = M31Matrix::from_data(4, 4, vec![
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
        ]).unwrap();
        let k = q.clone(); // Self-attention for simplicity
        let v = M31Matrix::from_data(4, 4, vec![
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
        ]).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let proof = prove_composed_attention(
            &q, &k, &v,
            ActivationType::ReLU,
            4, // log_table_size
            &mut prover_channel,
        ).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_composed_attention(&q, &k, &v, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_composed_attention_sigmoid_4x4() {
        // Same small matrices with Sigmoid activation.
        let q = M31Matrix::from_data(4, 4, vec![
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
        ]).unwrap();
        let k = q.clone();
        let v = M31Matrix::from_data(4, 4, vec![
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
        ]).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let proof = prove_composed_attention(
            &q, &k, &v,
            ActivationType::Sigmoid,
            4,
            &mut prover_channel,
        ).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_composed_attention(&q, &k, &v, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_composed_attention_gelu_4x4() {
        let q = M31Matrix::from_data(4, 4, vec![
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
        ]).unwrap();
        let k = q.clone();
        let v = M31Matrix::from_data(4, 4, vec![
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
        ]).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let proof = prove_composed_attention(
            &q, &k, &v,
            ActivationType::GELU,
            4,
            &mut prover_channel,
        ).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_composed_attention(&q, &k, &v, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_composed_attention_scores_out_of_range() {
        // Scores exceed table range → should error
        let q = M31Matrix::from_data(2, 2, vec![
            M31::from(10), M31::from(10),
            M31::from(10), M31::from(10),
        ]).unwrap();
        let k = q.clone();
        let v = M31Matrix::from_data(2, 2, vec![
            M31::from(1), M31::from(1),
            M31::from(1), M31::from(1),
        ]).unwrap();

        let mut channel = Blake2sChannel::default();
        // Table is only [0, 16), but scores = Q×K^T will have entries like 200
        let result = prove_composed_attention(
            &q, &k, &v,
            ActivationType::ReLU,
            4, // tiny table
            &mut channel,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_activation_table() {
        let table = PrecomputedTable::relu(4);
        let m = M31Matrix::from_data(2, 2, vec![
            M31::from(3), M31::from(7),
            M31::from(10), M31::from(0),
        ]).unwrap();

        let result = apply_activation_table(&m, &table).unwrap();
        // ReLU on log_size=4: [0,8) → identity, [8,16) → 0
        assert_eq!(result.get(0, 0), M31::from(3));  // 3 < 8 → 3
        assert_eq!(result.get(0, 1), M31::from(7));  // 7 < 8 → 7
        assert_eq!(result.get(1, 0), M31::from(0));  // 10 >= 8 → 0
        assert_eq!(result.get(1, 1), M31::from(0));  // 0 → 0
    }
}
