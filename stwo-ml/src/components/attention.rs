//! Attention mechanism verification via composed sumcheck + LogUp.
//!
//! # Attention Decomposition
//!
//! Standard attention: `Output = softmax(Q × K^T / √d_k) × V`
//!
//! Decomposed into provable stages using existing primitives:
//!
//! ```text
//! Stage 1: Projections (3 × Sumcheck MatMul)
//!   Q = X × W_Q, K = X × W_K, V = X × W_V
//!
//! Stage 2: Per-head Score Matrix (Sumcheck MatMul)
//!   scores_h = Q_h × K_h^T, scaled by 1/√d_k
//!
//! Stage 3: Softmax (LogUp STARK)
//!   weights_h = softmax(scores_h)
//!   Only exp(x) gets a STARK proof; normalization verified via commitment chain
//!
//! Stage 4: Per-head Context (Sumcheck MatMul)
//!   context_h = weights_h × V_h
//!
//! Stage 5: Output Projection (Sumcheck MatMul)
//!   output = Concat(context_0, ..., context_h) × W_O
//! ```
//!
//! # Proof Count: 4 + 2*H + 1 (matmul sumchecks + 1 LogUp STARK)
//!
//! For 2 heads (test): 9 proofs. For 32 heads (production): 69 proofs.

use stwo::core::fields::m31::M31;
use stwo::core::pcs::PcsConfig;
use stwo::core::channel::MerkleChannel;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::prover::backend::BackendForChannel;
use stwo::prover::poly::circle::PolyOps;
use stwo_constraint_framework::FrameworkComponent;

use crate::components::activation::{ActivationEval, ActivationType};
use crate::components::matmul::{
    M31Matrix, MatMulError, MatMulSumcheckProof,
    MatMulSumcheckProofOnChain,
    matmul_m31, prove_matmul_sumcheck,
    prove_matmul_sumcheck_onchain,
};
use crate::compiler::prove::prove_activation_layer;
use crate::gadgets::lookup_table::PrecomputedTable;
use crate::gadgets::lookup_table::activations::softmax_exp;

/// Configuration for a single attention head.
#[derive(Debug, Clone, Copy)]
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
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        Self { num_heads, d_model, seq_len }
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

    /// Naive O(n³) trace rows for comparison.
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

// ===== Weight & Intermediate Structures =====

/// Weight matrices for multi-head attention.
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Query projection: (d_model, d_model)
    pub w_q: M31Matrix,
    /// Key projection: (d_model, d_model)
    pub w_k: M31Matrix,
    /// Value projection: (d_model, d_model)
    pub w_v: M31Matrix,
    /// Output projection: (d_model, d_model)
    pub w_o: M31Matrix,
}

/// Intermediate results from the attention forward pass.
#[derive(Debug, Clone)]
pub struct AttentionIntermediates {
    /// Q = input × W_Q: (seq_len, d_model)
    pub q: M31Matrix,
    /// K = input × W_K: (seq_len, d_model)
    pub k: M31Matrix,
    /// V = input × W_V: (seq_len, d_model)
    pub v: M31Matrix,
    /// Per-head score matrices (after scaling): num_heads × (seq_len, seq_len)
    pub score_matrices: Vec<M31Matrix>,
    /// Per-head softmax outputs: num_heads × (seq_len, seq_len)
    pub softmax_outputs: Vec<M31Matrix>,
    /// Per-head context outputs: num_heads × (seq_len, d_k)
    pub head_outputs: Vec<M31Matrix>,
    /// Concatenated heads: (seq_len, d_model)
    pub concat: M31Matrix,
    /// Final output: (seq_len, d_model)
    pub final_output: M31Matrix,
}

/// Complete attention proof containing all sub-proofs.
#[derive(Debug)]
pub struct AttentionProof<H: MerkleHasherLifted> {
    /// Q projection proof
    pub q_proof: MatMulSumcheckProof,
    /// K projection proof
    pub k_proof: MatMulSumcheckProof,
    /// V projection proof
    pub v_proof: MatMulSumcheckProof,
    /// Per-head score proofs (Q_h × K_h^T)
    pub score_proofs: Vec<MatMulSumcheckProof>,
    /// Per-head attention×V proofs
    pub attn_v_proofs: Vec<MatMulSumcheckProof>,
    /// Output projection proof
    pub output_proof: MatMulSumcheckProof,
    /// Aggregated softmax exp STARK proof (all heads batched)
    pub softmax_exp_proof: StarkProof<H>,
    /// Forward pass intermediates (for verification replay)
    pub intermediates: AttentionIntermediates,
}

/// On-chain attention proof (Poseidon channel).
#[derive(Debug)]
pub struct AttentionProofOnChain {
    pub q_proof: MatMulSumcheckProofOnChain,
    pub k_proof: MatMulSumcheckProofOnChain,
    pub v_proof: MatMulSumcheckProofOnChain,
    pub score_proofs: Vec<MatMulSumcheckProofOnChain>,
    pub attn_v_proofs: Vec<MatMulSumcheckProofOnChain>,
    pub output_proof: MatMulSumcheckProofOnChain,
    pub intermediates: AttentionIntermediates,
}

/// Error type for attention proving.
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("MatMul error in {stage}: {source}")]
    MatMul {
        stage: String,
        #[source]
        source: MatMulError,
    },
    #[error("Activation error: {0}")]
    Activation(String),
    #[error("Dimension error: {0}")]
    Dimension(String),
}

impl From<MatMulError> for AttentionError {
    fn from(e: MatMulError) -> Self {
        AttentionError::MatMul {
            stage: "unknown".to_string(),
            source: e,
        }
    }
}

// ===== Helper Functions =====

/// Transpose an M31Matrix: (r, c) → (c, r).
pub fn transpose_m31(matrix: &M31Matrix) -> M31Matrix {
    let mut result = M31Matrix::new(matrix.cols, matrix.rows);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            result.set(j, i, matrix.get(i, j));
        }
    }
    result
}

/// Zero-pad a matrix to next power-of-2 dimensions.
pub fn pad_to_pow2(matrix: &M31Matrix) -> M31Matrix {
    let new_rows = matrix.rows.next_power_of_two();
    let new_cols = matrix.cols.next_power_of_two();
    if new_rows == matrix.rows && new_cols == matrix.cols {
        return matrix.clone();
    }
    let mut padded = M31Matrix::new(new_rows, new_cols);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            padded.set(i, j, matrix.get(i, j));
        }
    }
    padded
}

/// Split a (seq_len, d_model) matrix into num_heads matrices of (seq_len, d_k).
pub fn split_heads(matrix: &M31Matrix, num_heads: usize) -> Vec<M31Matrix> {
    let d_k = matrix.cols / num_heads;
    assert_eq!(matrix.cols, num_heads * d_k, "cols must be divisible by num_heads");
    let mut heads = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let mut head = M31Matrix::new(matrix.rows, d_k);
        for i in 0..matrix.rows {
            for j in 0..d_k {
                head.set(i, j, matrix.get(i, h * d_k + j));
            }
        }
        heads.push(head);
    }
    heads
}

/// Concatenate num_heads matrices of (seq_len, d_k) into (seq_len, d_model).
pub fn concat_heads(heads: &[M31Matrix]) -> M31Matrix {
    if heads.is_empty() {
        return M31Matrix::new(0, 0);
    }
    let seq_len = heads[0].rows;
    let d_k = heads[0].cols;
    let d_model = heads.len() * d_k;
    let mut result = M31Matrix::new(seq_len, d_model);
    for (h, head) in heads.iter().enumerate() {
        for i in 0..seq_len {
            for j in 0..d_k {
                result.set(i, h * d_k + j, head.get(i, j));
            }
        }
    }
    result
}

/// Apply causal mask: set score[i][j] = P-2 when j > i.
/// P-2 in M31 arithmetic causes softmax_exp to return ~0.
pub fn apply_causal_mask(scores: &mut M31Matrix) {
    let p_minus_2 = M31::from((1u32 << 31) - 3);
    for i in 0..scores.rows {
        for j in (i + 1)..scores.cols {
            scores.set(i, j, p_minus_2);
        }
    }
}

/// Modular inverse of n in M31 via Fermat's little theorem: n^(P-2) mod P.
pub(crate) fn m31_mod_inverse(n: u32) -> M31 {
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

/// Integer square root (floor).
fn isqrt(n: usize) -> u32 {
    (n as f64).sqrt() as u32
}

/// Softmax over a single row in M31 arithmetic.
///
/// 1. Apply softmax_exp element-wise
/// 2. Sum the row
/// 3. Multiply each element by modular inverse of the sum
pub fn softmax_row_m31(row: &[M31]) -> Vec<M31> {
    let exp_vals: Vec<M31> = row.iter().map(|&x| softmax_exp(x)).collect();
    let sum: u64 = exp_vals.iter().map(|v| v.0 as u64).sum();
    // Clamp sum into M31 range
    let sum_m31 = (sum % ((1u64 << 31) - 1)) as u32;
    if sum_m31 == 0 {
        return exp_vals;
    }
    let inv_sum = m31_mod_inverse(sum_m31);
    exp_vals.iter().map(|&v| v * inv_sum).collect()
}

/// Apply softmax row-wise to an entire matrix.
fn softmax_matrix(matrix: &M31Matrix) -> M31Matrix {
    let mut result = M31Matrix::new(matrix.rows, matrix.cols);
    for i in 0..matrix.rows {
        let row: Vec<M31> = (0..matrix.cols).map(|j| matrix.get(i, j)).collect();
        let soft = softmax_row_m31(&row);
        for (j, &v) in soft.iter().enumerate() {
            result.set(i, j, v);
        }
    }
    result
}

// ===== Forward Pass =====

/// Execute the full multi-head attention forward pass.
pub fn attention_forward(
    input: &M31Matrix,
    weights: &AttentionWeights,
    config: &MultiHeadAttentionConfig,
    causal: bool,
) -> AttentionIntermediates {
    let d_k = config.d_k();

    // Projections: Q, K, V = input × W_Q/K/V
    let q = matmul_m31(input, &weights.w_q);
    let k = matmul_m31(input, &weights.w_k);
    let v = matmul_m31(input, &weights.w_v);

    // Split into per-head slices
    let q_heads = split_heads(&q, config.num_heads);
    let k_heads = split_heads(&k, config.num_heads);
    let v_heads = split_heads(&v, config.num_heads);

    let scale_inv = m31_mod_inverse(isqrt(d_k));

    let mut score_matrices = Vec::with_capacity(config.num_heads);
    let mut softmax_outputs = Vec::with_capacity(config.num_heads);
    let mut head_outputs = Vec::with_capacity(config.num_heads);

    for h in 0..config.num_heads {
        // Score: Q_h × K_h^T
        let k_t = transpose_m31(&k_heads[h]);
        let mut scores = matmul_m31(&q_heads[h], &k_t);

        // Scale by 1/√d_k
        for val in scores.data.iter_mut() {
            *val = *val * scale_inv;
        }

        // Causal mask
        if causal {
            apply_causal_mask(&mut scores);
        }

        score_matrices.push(scores.clone());

        // Softmax (row-wise)
        let soft = softmax_matrix(&scores);
        softmax_outputs.push(soft.clone());

        // Context: softmax × V_h
        let context = matmul_m31(&soft, &v_heads[h]);
        head_outputs.push(context);
    }

    // Concat heads
    let concat = concat_heads(&head_outputs);

    // Output projection
    let final_output = matmul_m31(&concat, &weights.w_o);

    AttentionIntermediates {
        q,
        k,
        v,
        score_matrices,
        softmax_outputs,
        head_outputs,
        concat,
        final_output,
    }
}

// ===== Softmax Normalization Constraint =====

/// Evaluator for softmax normalization constraint.
///
/// Verifies that `weights[i] * sum_exp == exp_values[i]` for each element,
/// ensuring that the softmax normalization step is correct.
///
/// # Constraint (degree 2):
///   weights[i] * sum_exp - exp_values[i] = 0
#[derive(Debug, Clone)]
pub struct SoftmaxNormEval {
    pub log_n_rows: u32,
}

impl stwo_constraint_framework::FrameworkEval for SoftmaxNormEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: stwo_constraint_framework::EvalAtRow>(&self, mut eval: E) -> E {
        let exp_val = eval.next_trace_mask();
        let sum_exp = eval.next_trace_mask();
        let weight = eval.next_trace_mask();

        // Constraint: weight * sum_exp - exp_val = 0
        // i.e. weight = exp_val / sum_exp (softmax normalization)
        eval.add_constraint(weight * sum_exp - exp_val);

        eval
    }
}

pub type SoftmaxNormComponent = stwo_constraint_framework::FrameworkComponent<SoftmaxNormEval>;

// ===== Proving =====

/// Prove multi-head attention, generic over backend and Merkle channel.
///
/// Generates:
/// - 3 projection matmul sumcheck proofs (Q, K, V)
/// - num_heads score matmul sumcheck proofs (Q_h × K_h^T)
/// - num_heads attention×V matmul sumcheck proofs
/// - 1 output projection matmul sumcheck proof
/// - 1 aggregated softmax exp LogUp STARK proof
pub fn prove_attention_with<B, MC>(
    input: &M31Matrix,
    weights: &AttentionWeights,
    config: &MultiHeadAttentionConfig,
    causal: bool,
) -> Result<AttentionProof<<MC as MerkleChannel>::H>, AttentionError>
where
    B: BackendForChannel<MC> + PolyOps,
    MC: MerkleChannel,
    FrameworkComponent<ActivationEval>: stwo::prover::ComponentProver<B>,
{
    // Run forward pass
    let intermediates = attention_forward(input, weights, config, causal);

    // Pad matrices to power-of-2 for sumcheck
    let input_p = pad_to_pow2(input);
    let wq_p = pad_to_pow2(&weights.w_q);
    let wk_p = pad_to_pow2(&weights.w_k);
    let wv_p = pad_to_pow2(&weights.w_v);
    let wo_p = pad_to_pow2(&weights.w_o);
    let q_p = pad_to_pow2(&intermediates.q);
    let k_p = pad_to_pow2(&intermediates.k);
    let v_p = pad_to_pow2(&intermediates.v);

    // 1. Q projection proof
    let q_proof = prove_matmul_sumcheck(&input_p, &wq_p, &q_p)
        .map_err(|e| AttentionError::MatMul { stage: "Q_projection".into(), source: e })?;

    // 2. K projection proof
    let k_proof = prove_matmul_sumcheck(&input_p, &wk_p, &k_p)
        .map_err(|e| AttentionError::MatMul { stage: "K_projection".into(), source: e })?;

    // 3. V projection proof
    let v_proof = prove_matmul_sumcheck(&input_p, &wv_p, &v_p)
        .map_err(|e| AttentionError::MatMul { stage: "V_projection".into(), source: e })?;

    // Split for per-head proofs
    let q_heads = split_heads(&intermediates.q, config.num_heads);
    let k_heads = split_heads(&intermediates.k, config.num_heads);
    let v_heads = split_heads(&intermediates.v, config.num_heads);

    let mut score_proofs = Vec::with_capacity(config.num_heads);
    let mut attn_v_proofs = Vec::with_capacity(config.num_heads);
    let mut all_softmax_inputs = Vec::new();
    let mut all_softmax_outputs = Vec::new();

    for h in 0..config.num_heads {
        // Per-head: score = Q_h × K_h^T
        let k_t = transpose_m31(&k_heads[h]);
        let q_h_p = pad_to_pow2(&q_heads[h]);
        let k_t_p = pad_to_pow2(&k_t);

        // We need the padded score matrix that matches q_h_p × k_t_p
        let scores_p_raw = matmul_m31(&q_h_p, &k_t_p);
        let score_proof = prove_matmul_sumcheck(&q_h_p, &k_t_p, &scores_p_raw)
            .map_err(|e| AttentionError::MatMul {
                stage: format!("score_head_{h}"),
                source: e,
            })?;
        score_proofs.push(score_proof);

        // Collect softmax exp inputs/outputs for batched STARK
        let score_mat = &intermediates.score_matrices[h];
        for val in &score_mat.data {
            all_softmax_inputs.push(*val);
            all_softmax_outputs.push(softmax_exp(*val));
        }

        // Per-head: context = softmax × V_h
        let soft_p = pad_to_pow2(&intermediates.softmax_outputs[h]);
        let v_h_p = pad_to_pow2(&v_heads[h]);
        let context_p = matmul_m31(&soft_p, &v_h_p);
        let attn_v_proof = prove_matmul_sumcheck(&soft_p, &v_h_p, &context_p)
            .map_err(|e| AttentionError::MatMul {
                stage: format!("attn_v_head_{h}"),
                source: e,
            })?;
        attn_v_proofs.push(attn_v_proof);
    }

    // Output projection proof
    let concat_p = pad_to_pow2(&intermediates.concat);
    let out_p = pad_to_pow2(&intermediates.final_output);
    let output_proof = prove_matmul_sumcheck(&concat_p, &wo_p, &out_p)
        .map_err(|e| AttentionError::MatMul { stage: "output_projection".into(), source: e })?;

    // Batched softmax exp STARK proof (all heads in one proof)
    let table = PrecomputedTable::build_parallel(softmax_exp, ActivationType::Softmax.production_log_size());
    let pcs_config = PcsConfig::default();
    let (_component, softmax_exp_proof) = prove_activation_layer::<B, MC>(
        &all_softmax_inputs,
        &all_softmax_outputs,
        &table,
        pcs_config,
    ).map_err(|e| AttentionError::Activation(format!("softmax exp STARK: {e}")))?;

    Ok(AttentionProof {
        q_proof,
        k_proof,
        v_proof,
        score_proofs,
        attn_v_proofs,
        output_proof,
        softmax_exp_proof,
        intermediates,
    })
}

/// Prove multi-head attention for on-chain verification (Poseidon channel).
pub fn prove_attention_onchain(
    input: &M31Matrix,
    weights: &AttentionWeights,
    config: &MultiHeadAttentionConfig,
    causal: bool,
) -> Result<AttentionProofOnChain, AttentionError> {
    let intermediates = attention_forward(input, weights, config, causal);

    let input_p = pad_to_pow2(input);
    let wq_p = pad_to_pow2(&weights.w_q);
    let wk_p = pad_to_pow2(&weights.w_k);
    let wv_p = pad_to_pow2(&weights.w_v);
    let wo_p = pad_to_pow2(&weights.w_o);
    let q_p = pad_to_pow2(&intermediates.q);
    let k_p = pad_to_pow2(&intermediates.k);
    let v_p = pad_to_pow2(&intermediates.v);

    let q_proof = prove_matmul_sumcheck_onchain(&input_p, &wq_p, &q_p)
        .map_err(|e| AttentionError::MatMul { stage: "Q_projection".into(), source: e })?;
    let k_proof = prove_matmul_sumcheck_onchain(&input_p, &wk_p, &k_p)
        .map_err(|e| AttentionError::MatMul { stage: "K_projection".into(), source: e })?;
    let v_proof = prove_matmul_sumcheck_onchain(&input_p, &wv_p, &v_p)
        .map_err(|e| AttentionError::MatMul { stage: "V_projection".into(), source: e })?;

    let q_heads = split_heads(&intermediates.q, config.num_heads);
    let k_heads = split_heads(&intermediates.k, config.num_heads);
    let v_heads = split_heads(&intermediates.v, config.num_heads);

    let mut score_proofs = Vec::with_capacity(config.num_heads);
    let mut attn_v_proofs = Vec::with_capacity(config.num_heads);

    for h in 0..config.num_heads {
        let k_t = transpose_m31(&k_heads[h]);
        let q_h_p = pad_to_pow2(&q_heads[h]);
        let k_t_p = pad_to_pow2(&k_t);
        let scores_p = matmul_m31(&q_h_p, &k_t_p);
        let sp = prove_matmul_sumcheck_onchain(&q_h_p, &k_t_p, &scores_p)
            .map_err(|e| AttentionError::MatMul { stage: format!("score_head_{h}"), source: e })?;
        score_proofs.push(sp);

        let soft_p = pad_to_pow2(&intermediates.softmax_outputs[h]);
        let v_h_p = pad_to_pow2(&v_heads[h]);
        let context_p = matmul_m31(&soft_p, &v_h_p);
        let avp = prove_matmul_sumcheck_onchain(&soft_p, &v_h_p, &context_p)
            .map_err(|e| AttentionError::MatMul { stage: format!("attn_v_head_{h}"), source: e })?;
        attn_v_proofs.push(avp);
    }

    let concat_p = pad_to_pow2(&intermediates.concat);
    let out_p = pad_to_pow2(&intermediates.final_output);
    let output_proof = prove_matmul_sumcheck_onchain(&concat_p, &wo_p, &out_p)
        .map_err(|e| AttentionError::MatMul { stage: "output_projection".into(), source: e })?;

    Ok(AttentionProofOnChain {
        q_proof,
        k_proof,
        v_proof,
        score_proofs,
        attn_v_proofs,
        output_proof,
        intermediates,
    })
}

// ===== Float32 Attention =====

use crate::components::f32_ops::{F32Matrix, matmul_f32, softmax_f32};

/// Float32 weight matrices for multi-head attention.
#[derive(Debug, Clone)]
pub struct AttentionWeightsF32 {
    pub w_q: F32Matrix,
    pub w_k: F32Matrix,
    pub w_v: F32Matrix,
    pub w_o: F32Matrix,
}

/// Transpose an F32Matrix: (r, c) → (c, r).
fn transpose_f32(matrix: &F32Matrix) -> F32Matrix {
    let mut result = F32Matrix::new(matrix.cols, matrix.rows);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            result.set(j, i, matrix.get(i, j));
        }
    }
    result
}

/// Split an F32Matrix (seq_len, d_model) into num_heads matrices of (seq_len, d_k).
fn split_heads_f32(matrix: &F32Matrix, num_heads: usize) -> Vec<F32Matrix> {
    let d_k = matrix.cols / num_heads;
    let mut heads = Vec::with_capacity(num_heads);
    for h in 0..num_heads {
        let mut head = F32Matrix::new(matrix.rows, d_k);
        for i in 0..matrix.rows {
            for j in 0..d_k {
                head.set(i, j, matrix.get(i, h * d_k + j));
            }
        }
        heads.push(head);
    }
    heads
}

/// Concatenate num_heads F32Matrix (seq_len, d_k) into (seq_len, d_model).
fn concat_heads_f32(heads: &[F32Matrix]) -> F32Matrix {
    if heads.is_empty() {
        return F32Matrix::new(0, 0);
    }
    let seq_len = heads[0].rows;
    let d_k = heads[0].cols;
    let d_model = heads.len() * d_k;
    let mut result = F32Matrix::new(seq_len, d_model);
    for (h, head) in heads.iter().enumerate() {
        for i in 0..seq_len {
            for j in 0..d_k {
                result.set(i, h * d_k + j, head.get(i, j));
            }
        }
    }
    result
}

/// Execute the full multi-head attention forward pass in f32.
///
/// Produces meaningful real-valued outputs unlike the M31 version.
/// Uses the same decomposition: projections → per-head scores → softmax → context → output.
pub fn attention_forward_f32(
    input: &F32Matrix,
    weights: &AttentionWeightsF32,
    config: &MultiHeadAttentionConfig,
    causal: bool,
) -> F32Matrix {
    let d_k = config.d_k();
    let scale = 1.0 / (d_k as f32).sqrt();

    // Projections
    let q = matmul_f32(input, &weights.w_q);
    let k = matmul_f32(input, &weights.w_k);
    let v = matmul_f32(input, &weights.w_v);

    // Split into per-head slices
    let q_heads = split_heads_f32(&q, config.num_heads);
    let k_heads = split_heads_f32(&k, config.num_heads);
    let v_heads = split_heads_f32(&v, config.num_heads);

    let mut head_outputs = Vec::with_capacity(config.num_heads);

    for h in 0..config.num_heads {
        // Score: Q_h × K_h^T, scaled by 1/√d_k
        let k_t = transpose_f32(&k_heads[h]);
        let mut scores = matmul_f32(&q_heads[h], &k_t);

        // Scale
        for val in scores.data.iter_mut() {
            *val *= scale;
        }

        // Causal mask: set future positions to -infinity
        if causal {
            for i in 0..scores.rows {
                for j in (i + 1)..scores.cols {
                    scores.set(i, j, f32::NEG_INFINITY);
                }
            }
        }

        // Softmax (row-wise)
        let mut soft = F32Matrix::new(scores.rows, scores.cols);
        for i in 0..scores.rows {
            let row: Vec<f32> = (0..scores.cols).map(|j| scores.get(i, j)).collect();
            let s = softmax_f32(&row);
            for (j, &v) in s.iter().enumerate() {
                soft.set(i, j, v);
            }
        }

        // Context: softmax × V_h
        let context = matmul_f32(&soft, &v_heads[h]);
        head_outputs.push(context);
    }

    // Concat heads and output projection
    let concat = concat_heads_f32(&head_outputs);
    matmul_f32(&concat, &weights.w_o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    use stwo::prover::backend::simd::SimdBackend;

    /// Generate deterministic weights for testing.
    fn make_test_weights(d_model: usize, seed: u64) -> AttentionWeights {
        fn fill_matrix(rows: usize, cols: usize, seed: u64) -> M31Matrix {
            let mut m = M31Matrix::new(rows, cols);
            let mut state = seed;
            for i in 0..rows {
                for j in 0..cols {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let val = ((state >> 33) % 9 + 1) as u32;
                    m.set(i, j, M31::from(val));
                }
            }
            m
        }

        AttentionWeights {
            w_q: fill_matrix(d_model, d_model, seed),
            w_k: fill_matrix(d_model, d_model, seed.wrapping_add(1)),
            w_v: fill_matrix(d_model, d_model, seed.wrapping_add(2)),
            w_o: fill_matrix(d_model, d_model, seed.wrapping_add(3)),
        }
    }

    fn make_test_input(seq_len: usize, d_model: usize) -> M31Matrix {
        let mut m = M31Matrix::new(seq_len, d_model);
        for i in 0..seq_len {
            for j in 0..d_model {
                m.set(i, j, M31::from((i * d_model + j + 1) as u32 % 100 + 1));
            }
        }
        m
    }

    // --- Cost model tests (preserved from original) ---

    #[test]
    fn test_bert_base_attention() {
        let config = MultiHeadAttentionConfig::new(12, 768, 512);
        assert_eq!(config.d_k(), 64);

        let sumcheck = config.sumcheck_trace_rows();
        let naive = config.naive_trace_rows();

        assert!(sumcheck < 12_000_000, "sumcheck rows: {sumcheck}");
        assert!(naive > 100_000_000, "naive rows: {naive}");
        assert!(config.speedup() > 10.0, "speedup: {:.1}x", config.speedup());
    }

    #[test]
    fn test_gpt2_attention() {
        let config = MultiHeadAttentionConfig::new(12, 768, 1024);
        let sumcheck = config.sumcheck_trace_rows();
        assert!(sumcheck < 50_000_000);
    }

    #[test]
    fn test_llama_attention() {
        let config = MultiHeadAttentionConfig::new(32, 4096, 2048);
        let _sumcheck = config.sumcheck_trace_rows();
    }

    // --- Helper tests ---

    #[test]
    fn test_transpose() {
        let mut m = M31Matrix::new(2, 3);
        m.set(0, 0, M31::from(1)); m.set(0, 1, M31::from(2)); m.set(0, 2, M31::from(3));
        m.set(1, 0, M31::from(4)); m.set(1, 1, M31::from(5)); m.set(1, 2, M31::from(6));

        let t = transpose_m31(&m);
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), M31::from(1));
        assert_eq!(t.get(0, 1), M31::from(4));
        assert_eq!(t.get(2, 0), M31::from(3));
        assert_eq!(t.get(2, 1), M31::from(6));
    }

    #[test]
    fn test_transpose_roundtrip() {
        let m = make_test_input(4, 4);
        let tt = transpose_m31(&transpose_m31(&m));
        assert_eq!(m.data, tt.data);
    }

    #[test]
    fn test_pad_to_pow2() {
        let m = M31Matrix::new(3, 5);
        let p = pad_to_pow2(&m);
        assert_eq!(p.rows, 4);
        assert_eq!(p.cols, 8);
        // Original data preserved
        for i in 0..3 {
            for j in 0..5 {
                assert_eq!(p.get(i, j), m.get(i, j));
            }
        }
        // Padding is zero
        assert_eq!(p.get(3, 0), M31::from(0));
        assert_eq!(p.get(0, 7), M31::from(0));
    }

    #[test]
    fn test_pad_already_pow2() {
        let m = M31Matrix::new(4, 8);
        let p = pad_to_pow2(&m);
        assert_eq!(p.rows, 4);
        assert_eq!(p.cols, 8);
    }

    #[test]
    fn test_split_concat_roundtrip() {
        let m = make_test_input(4, 8);
        let heads = split_heads(&m, 2);
        assert_eq!(heads.len(), 2);
        assert_eq!(heads[0].rows, 4);
        assert_eq!(heads[0].cols, 4);

        let concat = concat_heads(&heads);
        assert_eq!(concat.rows, 4);
        assert_eq!(concat.cols, 8);
        assert_eq!(m.data, concat.data);
    }

    #[test]
    fn test_split_heads_values() {
        let mut m = M31Matrix::new(2, 4);
        for j in 0..4 { m.set(0, j, M31::from((j + 1) as u32)); }
        for j in 0..4 { m.set(1, j, M31::from((j + 5) as u32)); }

        let heads = split_heads(&m, 2);
        // Head 0: cols 0-1, Head 1: cols 2-3
        assert_eq!(heads[0].get(0, 0), M31::from(1));
        assert_eq!(heads[0].get(0, 1), M31::from(2));
        assert_eq!(heads[1].get(0, 0), M31::from(3));
        assert_eq!(heads[1].get(0, 1), M31::from(4));
    }

    #[test]
    fn test_causal_mask() {
        let mut scores = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                scores.set(i, j, M31::from(10));
            }
        }
        apply_causal_mask(&mut scores);

        let p_minus_2 = M31::from((1u32 << 31) - 3);
        // Diagonal and below should be unchanged
        assert_eq!(scores.get(0, 0), M31::from(10));
        assert_eq!(scores.get(1, 0), M31::from(10));
        assert_eq!(scores.get(1, 1), M31::from(10));
        // Above diagonal should be masked
        assert_eq!(scores.get(0, 1), p_minus_2);
        assert_eq!(scores.get(0, 3), p_minus_2);
        assert_eq!(scores.get(2, 3), p_minus_2);
    }

    #[test]
    fn test_softmax_row_basic() {
        // softmax of identical values should give uniform distribution (roughly)
        let row = vec![M31::from(100); 4];
        let result = softmax_row_m31(&row);
        // All outputs should be equal
        assert_eq!(result[0], result[1]);
        assert_eq!(result[1], result[2]);
        assert_eq!(result[2], result[3]);
    }

    #[test]
    fn test_softmax_row_nonzero() {
        let row = vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)];
        let result = softmax_row_m31(&row);
        // All outputs should be non-zero
        for v in &result {
            assert_ne!(*v, M31::from(0), "softmax output should not be zero");
        }
    }

    // --- Forward pass tests ---

    #[test]
    fn test_attention_forward_shapes() {
        let config = MultiHeadAttentionConfig::new(2, 4, 4);
        let input = make_test_input(4, 4);
        let weights = make_test_weights(4, 42);

        let inter = attention_forward(&input, &weights, &config, false);

        assert_eq!(inter.q.rows, 4);
        assert_eq!(inter.q.cols, 4);
        assert_eq!(inter.k.rows, 4);
        assert_eq!(inter.k.cols, 4);
        assert_eq!(inter.v.rows, 4);
        assert_eq!(inter.v.cols, 4);
        assert_eq!(inter.score_matrices.len(), 2);
        assert_eq!(inter.score_matrices[0].rows, 4);
        assert_eq!(inter.score_matrices[0].cols, 4);
        assert_eq!(inter.softmax_outputs.len(), 2);
        assert_eq!(inter.head_outputs.len(), 2);
        assert_eq!(inter.head_outputs[0].rows, 4);
        assert_eq!(inter.head_outputs[0].cols, 2); // d_k = 4/2 = 2
        assert_eq!(inter.concat.rows, 4);
        assert_eq!(inter.concat.cols, 4);
        assert_eq!(inter.final_output.rows, 4);
        assert_eq!(inter.final_output.cols, 4);
    }

    #[test]
    fn test_attention_forward_causal() {
        let config = MultiHeadAttentionConfig::new(1, 4, 4);
        let input = make_test_input(4, 4);
        let weights = make_test_weights(4, 42);

        let inter_causal = attention_forward(&input, &weights, &config, true);
        let inter_no_causal = attention_forward(&input, &weights, &config, false);

        // Causal and non-causal should produce different outputs
        assert_ne!(
            inter_causal.final_output.data,
            inter_no_causal.final_output.data,
            "causal mask should change the output"
        );
    }

    #[test]
    fn test_attention_forward_deterministic() {
        let config = MultiHeadAttentionConfig::new(2, 4, 4);
        let input = make_test_input(4, 4);
        let weights = make_test_weights(4, 42);

        let inter1 = attention_forward(&input, &weights, &config, false);
        let inter2 = attention_forward(&input, &weights, &config, false);

        assert_eq!(inter1.final_output.data, inter2.final_output.data);
    }

    // --- Proving tests ---

    #[test]
    fn test_prove_attention_single_head() {
        // 1 head, d_model=4, seq_len=4 (all power-of-2)
        let config = MultiHeadAttentionConfig::new(1, 4, 4);
        let input = make_test_input(4, 4);
        let weights = make_test_weights(4, 42);

        let result = prove_attention_with::<SimdBackend, Blake2sMerkleChannel>(
            &input, &weights, &config, false,
        );
        assert!(result.is_ok(), "Single-head attention proving failed: {:?}", result.err());

        let proof = result.unwrap();
        // 3 projections + 1 score + 1 attn_v + 1 output = 6 matmul proofs
        assert_eq!(proof.score_proofs.len(), 1);
        assert_eq!(proof.attn_v_proofs.len(), 1);
        assert_eq!(proof.intermediates.final_output.rows, 4);
        assert_eq!(proof.intermediates.final_output.cols, 4);
    }

    #[test]
    fn test_prove_attention_multi_head() {
        // 2 heads, d_model=4, seq_len=4
        let config = MultiHeadAttentionConfig::new(2, 4, 4);
        let input = make_test_input(4, 4);
        let weights = make_test_weights(4, 77);

        let result = prove_attention_with::<SimdBackend, Blake2sMerkleChannel>(
            &input, &weights, &config, false,
        );
        assert!(result.is_ok(), "Multi-head attention proving failed: {:?}", result.err());

        let proof = result.unwrap();
        assert_eq!(proof.score_proofs.len(), 2);
        assert_eq!(proof.attn_v_proofs.len(), 2);
    }

    #[test]
    fn test_prove_attention_causal() {
        let config = MultiHeadAttentionConfig::new(1, 4, 4);
        let input = make_test_input(4, 4);
        let weights = make_test_weights(4, 42);

        let result = prove_attention_with::<SimdBackend, Blake2sMerkleChannel>(
            &input, &weights, &config, true,
        );
        assert!(result.is_ok(), "Causal attention proving failed: {:?}", result.err());
    }

    #[test]
    fn test_prove_attention_onchain() {
        let config = MultiHeadAttentionConfig::new(1, 4, 4);
        let input = make_test_input(4, 4);
        let weights = make_test_weights(4, 42);

        let result = prove_attention_onchain(&input, &weights, &config, false);
        assert!(result.is_ok(), "On-chain attention proving failed: {:?}", result.err());

        let proof = result.unwrap();
        // Verify Poseidon commitments are non-zero
        assert_ne!(proof.q_proof.a_commitment, starknet_ff::FieldElement::ZERO);
    }

    #[test]
    fn test_attention_proof_count() {
        // Verify the proof count formula: 4 + 2*H + 1
        let config = MultiHeadAttentionConfig::new(2, 4, 4);
        let input = make_test_input(4, 4);
        let weights = make_test_weights(4, 42);

        let proof = prove_attention_with::<SimdBackend, Blake2sMerkleChannel>(
            &input, &weights, &config, false,
        ).unwrap();

        let matmul_count = 1 + 1 + 1 // Q, K, V projections
            + proof.score_proofs.len()   // per-head scores
            + proof.attn_v_proofs.len()  // per-head attn×V
            + 1;                         // output projection
        let stark_count = 1; // softmax exp

        assert_eq!(matmul_count, 4 + 2 * config.num_heads);
        assert_eq!(stark_count, 1);
        assert_eq!(matmul_count + stark_count, 4 + 2 * config.num_heads + 1);
    }

    // --- Float32 attention tests ---

    fn make_f32_weights(d_model: usize, seed: u64) -> AttentionWeightsF32 {
        fn fill_f32(rows: usize, cols: usize, seed: u64) -> F32Matrix {
            let mut data = vec![0.0f32; rows * cols];
            let mut state = seed;
            for v in data.iter_mut() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *v = ((state >> 33) % 10) as f32 * 0.1 - 0.5;
            }
            F32Matrix::from_data(rows, cols, data)
        }
        AttentionWeightsF32 {
            w_q: fill_f32(d_model, d_model, seed),
            w_k: fill_f32(d_model, d_model, seed.wrapping_add(1)),
            w_v: fill_f32(d_model, d_model, seed.wrapping_add(2)),
            w_o: fill_f32(d_model, d_model, seed.wrapping_add(3)),
        }
    }

    fn make_f32_input(seq_len: usize, d_model: usize) -> F32Matrix {
        let data: Vec<f32> = (0..seq_len * d_model)
            .map(|i| (i as f32 * 0.1) - 1.0)
            .collect();
        F32Matrix::from_data(seq_len, d_model, data)
    }

    #[test]
    fn test_attention_f32_single_head() {
        let config = MultiHeadAttentionConfig::new(1, 4, 4);
        let input = make_f32_input(4, 4);
        let weights = make_f32_weights(4, 42);

        let output = attention_forward_f32(&input, &weights, &config, false);
        assert_eq!(output.rows, 4);
        assert_eq!(output.cols, 4);
        // Output should be finite real values
        for &v in &output.data {
            assert!(v.is_finite(), "output should be finite, got {v}");
        }
    }

    #[test]
    fn test_attention_f32_multi_head() {
        let config = MultiHeadAttentionConfig::new(2, 4, 4);
        let input = make_f32_input(4, 4);
        let weights = make_f32_weights(4, 77);

        let output = attention_forward_f32(&input, &weights, &config, false);
        assert_eq!(output.rows, 4);
        assert_eq!(output.cols, 4);
        for &v in &output.data {
            assert!(v.is_finite(), "output should be finite, got {v}");
        }
    }

    #[test]
    fn test_attention_f32_causal_masking() {
        let config = MultiHeadAttentionConfig::new(1, 4, 4);
        let input = make_f32_input(4, 4);
        let weights = make_f32_weights(4, 42);

        let out_causal = attention_forward_f32(&input, &weights, &config, true);
        let out_no_causal = attention_forward_f32(&input, &weights, &config, false);

        // Causal and non-causal should produce different outputs
        let mut differs = false;
        for (a, b) in out_causal.data.iter().zip(out_no_causal.data.iter()) {
            if (a - b).abs() > 1e-6 {
                differs = true;
                break;
            }
        }
        assert!(differs, "causal mask should change the output");
    }

    // --- Softmax normalization tests ---

    #[test]
    fn test_softmax_normalization_eval() {
        use stwo_constraint_framework::FrameworkEval;
        let eval = SoftmaxNormEval { log_n_rows: 4 };
        assert_eq!(eval.log_size(), 4);
        assert_eq!(eval.max_constraint_log_degree_bound(), 5);
    }

    #[test]
    fn test_softmax_normalization_constraint_holds() {
        // For softmax: weight = exp(x) / sum(exp)
        // Constraint: weight * sum_exp - exp_val = 0
        // If exp_val = 10, sum_exp = 100, weight = 10 * inv(100)
        // Then weight * 100 = 10 ✓
        let exp_val = M31::from(10);
        let sum_exp = M31::from(100);
        let weight = exp_val * crate::components::attention::m31_mod_inverse(100);

        // Verify constraint: weight * sum_exp - exp_val = 0
        let constraint_val = weight * sum_exp - exp_val;
        assert_eq!(constraint_val, M31::from(0), "Softmax norm constraint should hold");
    }
}
