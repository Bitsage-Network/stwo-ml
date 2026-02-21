//! Helpers for building `ProofEvent::CircuitCompiled` from stwo-ml circuit types.
//!
//! This module is called from stwo-ml (feature-gated) — it may import stwo-ml
//! types directly. The output is a plain `ProofEvent` with no stwo/nightly deps.

use crate::events::{CircuitNodeMeta, LayerKind, ProofEvent};

/// A minimal description of one circuit layer, collected by stwo-ml before
/// calling `circuit_compiled_event`. This avoids importing stwo-ml types
/// directly into proof-stream.
#[derive(Debug, Clone)]
pub struct CircuitLayerDesc {
    pub layer_idx: usize,
    pub node_id: usize,
    pub kind: LayerKind,
    pub input_shape: (usize, usize),
    pub output_shape: (usize, usize),
    /// Estimated number of trace cells (e.g. m*k*n for MatMul).
    pub trace_cost: usize,
    /// Predecessor layer indices.
    pub input_layers: Vec<usize>,
}

/// Build `ProofEvent::CircuitCompiled` from a flat description list.
pub fn circuit_compiled_event(
    layers: &[CircuitLayerDesc],
    total_layers: usize,
    input_shape: (usize, usize),
    output_shape: (usize, usize),
) -> ProofEvent {
    let nodes: Vec<CircuitNodeMeta> = layers
        .iter()
        .map(|l| CircuitNodeMeta {
            layer_idx: l.layer_idx,
            node_id: l.node_id,
            kind: l.kind,
            input_shape: l.input_shape,
            output_shape: l.output_shape,
            trace_cost: l.trace_cost,
            input_layers: l.input_layers.clone(),
        })
        .collect();

    ProofEvent::CircuitCompiled {
        total_layers,
        input_shape,
        output_shape,
        nodes,
        has_simd: false,
        simd_num_blocks: 0,
    }
}

/// Estimate trace cost for a MatMul layer: m × k × n rounded up to next
/// power of two (proportional to sumcheck size).
pub fn matmul_trace_cost(m: usize, k: usize, n: usize) -> usize {
    let raw = m * k * n;
    raw.next_power_of_two()
}
