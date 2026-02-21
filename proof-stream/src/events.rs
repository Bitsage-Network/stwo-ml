//! `ProofEvent` — the central event type streamed from stwo-ml proving sessions.
//!
//! All variants are `#[non_exhaustive]` so that future additions don't break
//! downstream event consumers (e.g. sinks compiled against an older version).

use serde::{Deserialize, Serialize};

// ─── Mirror types (no dependency on stwo/nightly toolchain) ─────────────────

/// Mirror of QM31 / SecureField — four M31 limbs stored as bare u32.
/// The `a` field is the "real" M31 component; use it for scalar visualization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SecureFieldMirror {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
}

impl SecureFieldMirror {
    /// Normalize `a` to [0,1] for plotting.
    pub fn as_f32(&self) -> f32 {
        self.a as f32 / (0x7fff_ffff_u32 as f32)
    }
}

/// Degree-2 round polynomial [c0, c1, c2] from a MatMul sumcheck round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundPolyViz {
    pub c0: SecureFieldMirror,
    pub c1: SecureFieldMirror,
    pub c2: SecureFieldMirror,
}

/// Degree-3 round polynomial [c0, c1, c2, c3] (MulLayer / higher-degree oracle).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundPolyDeg3Viz {
    pub c0: SecureFieldMirror,
    pub c1: SecureFieldMirror,
    pub c2: SecureFieldMirror,
    pub c3: SecureFieldMirror,
}

// ─── Metadata types ───────────────────────────────────────────────────────────

/// Classification of a circuit layer used for coloring and filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerKind {
    MatMul,
    Activation,
    LayerNorm,
    RMSNorm,
    Add,
    Mul,
    Attention,
    Embedding,
    Dequantize,
    Quantize,
    Input,
    Unknown,
}

impl LayerKind {
    /// RGB color (0–255) used to tint this layer in the 3D circuit view.
    pub fn color_rgb(&self) -> [u8; 3] {
        match self {
            LayerKind::MatMul => [0x42, 0x9b, 0xf5],    // blue
            LayerKind::Activation => [0xf5, 0xa5, 0x42], // orange
            LayerKind::LayerNorm => [0x4c, 0xc6, 0x71],  // green
            LayerKind::RMSNorm => [0x38, 0xb2, 0xac],    // teal
            LayerKind::Add => [0xa0, 0xa0, 0xa0],        // gray
            LayerKind::Mul => [0xa8, 0x55, 0xd4],        // purple
            LayerKind::Attention => [0xf5, 0xd0, 0x42],  // yellow
            LayerKind::Embedding => [0xec, 0x6e, 0xad],  // pink
            LayerKind::Dequantize => [0xff, 0x8c, 0x00], // dark orange
            LayerKind::Quantize => [0xff, 0x4c, 0x4c],   // red
            LayerKind::Input => [0x80, 0x80, 0xff],      // light blue
            LayerKind::Unknown => [0x60, 0x60, 0x60],    // dark gray
        }
    }
}

/// Classification of the proof artifact produced by a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerProofKind {
    Sumcheck,
    LogUp,
    Linear,
    Skipped,
    Deferred,
}

/// A single node in the circuit DAG (static metadata emitted in `CircuitCompiled`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitNodeMeta {
    pub layer_idx: usize,
    pub node_id: usize,
    pub kind: LayerKind,
    pub input_shape: (usize, usize),
    pub output_shape: (usize, usize),
    /// Estimated number of trace cells (used as node radius in 3D view).
    pub trace_cost: usize,
    /// Predecessor layer indices in the `LayeredCircuit`.
    pub input_layers: Vec<usize>,
}

/// Snapshot of a single GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSnapshot {
    pub device_id: usize,
    pub device_name: String,
    /// Utilization in [0.0, 1.0].
    pub utilization: f32,
    /// Free memory in bytes (None if unknown).
    pub free_memory_bytes: Option<usize>,
}

/// Basic descriptive statistics for a layer activation tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub sparsity: f32,
}

/// Severity level for log events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

// ─── ProofEvent ──────────────────────────────────────────────────────────────

/// All events that can be emitted during a stwo-ml proof session.
///
/// Consumers (e.g. `RerunSink`) pattern-match this enum to route events to
/// the appropriate Rerun entity paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProofEvent {
    // ── Circuit (once, static) ──────────────────────────────────────────────
    /// Emitted once after the `LayeredCircuit` is compiled from the `ComputationGraph`.
    CircuitCompiled {
        total_layers: usize,
        input_shape: (usize, usize),
        output_shape: (usize, usize),
        nodes: Vec<CircuitNodeMeta>,
        has_simd: bool,
        simd_num_blocks: usize,
    },

    // ── Forward pass intermediates (viz #2 Inference DVR, #3 Robotics) ──────
    /// Raw layer activation output sampled for visualization.
    LayerActivation {
        layer_idx: usize,
        node_id: usize,
        kind: LayerKind,
        output_shape: (usize, usize),
        /// Up to `max_sample` elements from the output tensor (M31 raw values).
        output_sample: Vec<u32>,
        stats: ActivationStats,
    },
    /// Attention score matrix for one head (sampled to ≤64×64).
    AttentionHeatmap {
        layer_idx: usize,
        head_idx: usize,
        num_heads: usize,
        seq_len: usize,
        /// Row-major, length = min(seq_len,64)².
        scores: Vec<f32>,
    },

    // ── GKR walk (viz #1 ZK Debugger, #5 Circuit Layout) ───────────────────
    /// Start of the entire proof session.
    ProofStart {
        model_name: Option<String>,
        backend: String,
        num_layers: usize,
        input_shape: (usize, usize),
        output_shape: (usize, usize),
    },
    /// Beginning of a single layer's reduction.
    LayerStart {
        layer_idx: usize,
        kind: LayerKind,
        input_shape: (usize, usize),
        output_shape: (usize, usize),
        /// Estimated trace cost for this layer.
        trace_cost: usize,
        /// Initial claim value (QM31.a normalized to f32).
        claim_value_approx: f32,
        gpu_device: Option<usize>,
    },
    /// One round of sumcheck within a layer.
    SumcheckRound {
        layer_idx: usize,
        round: usize,
        total_rounds: usize,
        /// Degree-2 polynomial (MatMul / LogUp rounds).
        poly_deg2: Option<RoundPolyViz>,
        /// Degree-3 polynomial (Mul layer rounds).
        poly_deg3: Option<RoundPolyDeg3Viz>,
        /// Reduced claim after this round.
        claim_value_approx: f32,
    },
    /// End of a single layer's reduction.
    LayerEnd {
        layer_idx: usize,
        kind: LayerProofKind,
        final_claim_value_approx: f32,
        duration_ms: u64,
        rounds_completed: usize,
    },
    /// The entire proof session completed.
    ProofComplete {
        duration_ms: u64,
        num_layer_proofs: usize,
        num_weight_openings: usize,
        weight_binding_mode: String,
    },

    // ── Weight openings (viz #5: highlighted edges) ──────────────────────────
    WeightOpeningStart {
        weight_node_id: usize,
        eval_point_len: usize,
    },
    WeightOpeningEnd {
        weight_node_id: usize,
        duration_ms: u64,
        /// Hex-encoded commitment (first 8 bytes).
        commitment_hex: String,
    },

    // ── Aggregated oracle sumcheck ───────────────────────────────────────────
    AggregatedBindingStart {
        num_claims: usize,
        num_matrices: usize,
    },
    AggregatedBindingEnd {
        duration_ms: u64,
        estimated_calldata_felts: usize,
    },

    // ── GPU cluster (viz #4 ProofTV) ─────────────────────────────────────────
    GpuStatus {
        devices: Vec<GpuSnapshot>,
        matmul_done: usize,
        matmul_total: usize,
        layers_done: usize,
        layers_total: usize,
    },

    // ── Unified STARK (aggregation.rs) ───────────────────────────────────────
    StarkProofStart {
        num_activation_layers: usize,
        num_add_layers: usize,
        num_layernorm_layers: usize,
    },
    StarkProofEnd {
        duration_ms: u64,
    },

    /// Free-form log entry.
    Log {
        level: LogLevel,
        message: String,
    },
}
