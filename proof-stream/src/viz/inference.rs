//! Helpers for building `ProofEvent::LayerActivation` from intermediate tensors.

use crate::events::{ActivationStats, LayerKind, ProofEvent};

/// Build a `LayerActivation` event from a flat slice of M31 raw values (u32).
///
/// At most `max_sample` elements are included in `output_sample`.
/// Stats (mean, std, min, max, sparsity) are computed over the full slice.
pub fn layer_activation_event(
    layer_idx: usize,
    node_id: usize,
    kind: LayerKind,
    output_shape: (usize, usize),
    values: &[u32],
    max_sample: usize,
) -> ProofEvent {
    let stats = compute_stats(values);
    let sample: Vec<u32> = values.iter().copied().take(max_sample).collect();

    ProofEvent::LayerActivation {
        layer_idx,
        node_id,
        kind,
        output_shape,
        output_sample: sample,
        stats,
    }
}

/// Compute activation statistics over raw M31 values.
fn compute_stats(values: &[u32]) -> ActivationStats {
    if values.is_empty() {
        return ActivationStats {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            sparsity: 1.0,
        };
    }

    let p = 0x7fff_ffff_u32 as f64;
    let n = values.len() as f64;

    let sum: f64 = values.iter().map(|&v| v as f64 / p).sum();
    let mean = (sum / n) as f32;

    let var: f64 = values
        .iter()
        .map(|&v| {
            let x = v as f64 / p - mean as f64;
            x * x
        })
        .sum::<f64>()
        / n;
    let std_dev = var.sqrt() as f32;

    let min = values.iter().map(|&v| v as f64 / p).fold(f64::INFINITY, f64::min) as f32;
    let max = values.iter().map(|&v| v as f64 / p).fold(f64::NEG_INFINITY, f64::max) as f32;

    let zeros = values.iter().filter(|&&v| v == 0).count();
    let sparsity = zeros as f32 / values.len() as f32;

    ActivationStats {
        mean,
        std_dev,
        min,
        max,
        sparsity,
    }
}
