//! Helpers for building `ProofEvent::AttentionHeatmap` from raw score matrices.

use crate::events::ProofEvent;

/// Maximum attention grid dimension to stream. Larger matrices are downsampled.
pub const MAX_ATTN_DIM: usize = 64;

/// Build an `AttentionHeatmap` event by sampling the attention score matrix.
///
/// `scores_row_major` must have length `seq_len * seq_len`.
/// If `seq_len > MAX_ATTN_DIM` the matrix is subsampled (stride sampling).
pub fn attention_heatmap_event(
    layer_idx: usize,
    head_idx: usize,
    num_heads: usize,
    seq_len: usize,
    scores_row_major: &[f32],
) -> ProofEvent {
    let out_dim = seq_len.min(MAX_ATTN_DIM);
    let stride = if seq_len > MAX_ATTN_DIM {
        (seq_len + MAX_ATTN_DIM - 1) / MAX_ATTN_DIM
    } else {
        1
    };

    let mut sampled = Vec::with_capacity(out_dim * out_dim);
    for r in (0..seq_len).step_by(stride).take(out_dim) {
        for c in (0..seq_len).step_by(stride).take(out_dim) {
            let idx = r * seq_len + c;
            sampled.push(if idx < scores_row_major.len() {
                scores_row_major[idx]
            } else {
                0.0
            });
        }
    }

    ProofEvent::AttentionHeatmap {
        layer_idx,
        head_idx,
        num_heads,
        seq_len: out_dim,
        scores: sampled,
    }
}

/// Build an `AttentionHeatmap` from raw M31 values (u32) converted to f32.
pub fn attention_heatmap_from_u32(
    layer_idx: usize,
    head_idx: usize,
    num_heads: usize,
    seq_len: usize,
    scores_u32: &[u32],
) -> ProofEvent {
    let scores_f32: Vec<f32> = scores_u32
        .iter()
        .map(|&v| v as f32 / 0x7fff_ffff_u32 as f32)
        .collect();
    attention_heatmap_event(layer_idx, head_idx, num_heads, seq_len, &scores_f32)
}
