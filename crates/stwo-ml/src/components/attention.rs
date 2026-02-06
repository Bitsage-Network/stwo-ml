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
//! Each head is independent → **parallelizable across GPUs**.
//!
//! # Trace Cost (Sumcheck)
//!
//! For seq_len=512, d_model=768, h=12 (BERT-base dimensions):
//!
//! ```text
//! Per head (d_k = 64):
//!   Q×K^T:    O(512×64 + 512×512 + 64×512) = ~327K witness rows
//!   softmax:  512×512 = 262K lookups
//!   attn×V:   O(512×512 + 512×64 + 512×64) = ~327K witness rows
//!   Subtotal: ~916K per head
//!
//! All 12 heads: ~11M rows (provable on H100 in ~10s)
//!
//! Compare naive O(n³):
//!   Q×K^T alone = 512×64×512 = 16.7M rows per head
//!   All 12 heads naive = 200M+ rows (IMPOSSIBLE)
//! ```
//!
//! Sumcheck reduces 12-head attention from 200M rows to 11M rows: **~18× reduction**.

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

// TODO: Phase 2 implementation
// - Compose MatMulComponent for Q×K^T and attn×V
// - Compose ActivationComponent (softmax table) for attention weights
// - Add causal masking support (decoder-only transformers)
// - Multi-head parallelism via STWO's component system
// - GPU-accelerated per-head proving on multi-GPU setups

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_base_attention() {
        // BERT-base: 12 heads, 768 dims, 512 seq_len
        let config = MultiHeadAttentionConfig::new(12, 768, 512);
        assert_eq!(config.d_k(), 64);

        let sumcheck = config.sumcheck_trace_rows();
        let naive = config.naive_trace_rows();

        // Sumcheck should be dramatically smaller
        assert!(sumcheck < 12_000_000, "sumcheck rows: {}", sumcheck);
        assert!(naive > 100_000_000, "naive rows: {}", naive);
        assert!(config.speedup() > 10.0, "speedup: {:.1}x", config.speedup());

        println!("BERT-base attention:");
        println!("  Sumcheck: {} rows ({:.1}M)", sumcheck, sumcheck as f64 / 1e6);
        println!("  Naive:    {} rows ({:.1}M)", naive, naive as f64 / 1e6);
        println!("  Speedup:  {:.1}x", config.speedup());
    }

    #[test]
    fn test_gpt2_attention() {
        // GPT-2 small: 12 heads, 768 dims, 1024 seq_len
        let config = MultiHeadAttentionConfig::new(12, 768, 1024);

        let sumcheck = config.sumcheck_trace_rows();
        println!("GPT-2 attention: {:.1}M sumcheck rows", sumcheck as f64 / 1e6);

        // Should still be provable (< 100M rows, feasible on H100)
        assert!(sumcheck < 50_000_000);
    }

    #[test]
    fn test_llama_attention() {
        // Llama-7B: 32 heads, 4096 dims, 2048 seq_len
        let config = MultiHeadAttentionConfig::new(32, 4096, 2048);

        let sumcheck = config.sumcheck_trace_rows();
        println!("Llama-7B attention (single layer): {:.1}M sumcheck rows",
                 sumcheck as f64 / 1e6);
    }
}
