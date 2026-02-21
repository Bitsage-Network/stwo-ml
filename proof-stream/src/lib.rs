//! `proof-stream` — real-time 3D streaming visualization of stwo-ml ZK proof sessions.
//!
//! # Overview
//!
//! This crate intercepts events emitted during stwo-ml proving (GKR walk,
//! sumcheck rounds, GPU utilization, layer activations) and streams them to a
//! [Rerun](https://rerun.io) viewer in real time via a background thread.
//!
//! # Quick start
//!
//! ```bash
//! # With the `rerun` feature enabled:
//! cargo run --bin prove-model --features proof-stream-rerun -- \
//!   --model-dir ./tiny-model --rerun spawn
//! ```
//!
//! # Feature flags
//!
//! - `rerun` — enables the `RerunSink` and its background thread. Without this
//!   flag the crate compiles with zero Rerun dependency.
//!
//! # Architecture
//!
//! 1. stwo-ml installs a `ProofSink` via `set_proof_sink()` (thread-local, RAII).
//! 2. The prover calls `emit_proof_event!(|| ...)` at hot points; the closure is
//!    never evaluated when no sink is active.
//! 3. Active sinks (e.g. `RerunSink`) forward events through a bounded
//!    crossbeam channel (8 192 slots, `try_send` — never blocks).
//! 4. A background thread drains the channel and calls the Rerun SDK.

pub mod blueprint;
pub mod events;
pub mod rerun_sink;
pub mod sink;
pub mod viz;
#[cfg(feature = "ws-server")]
pub mod ws_sink;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use events::{
    ActivationStats, CircuitNodeMeta, GpuSnapshot, LayerKind, LayerProofKind,
    LogLevel, ProofEvent, RoundPolyDeg3Viz, RoundPolyViz, SecureFieldMirror,
};
pub use sink::{ChannelSink, CollectingSink, NullSink, ProofEventSink, ProofSink};

#[cfg(feature = "rerun")]
pub use rerun_sink::{RerunConnection, RerunSink};

#[cfg(feature = "ws-server")]
pub use ws_sink::WsBroadcastSink;

// ── Convenience constructor ───────────────────────────────────────────────────

/// Parse a connection string and return a `ProofSink` wrapping a `RerunSink`.
///
/// Connection string formats:
/// - `"spawn"` — spawn a Rerun viewer subprocess
/// - `"file:<path>"` — write a `.rrd` replay file
/// - `"tcp://host:port"` or `"host:port"` — connect to a running viewer
///
/// Returns an error if the connection fails (e.g. viewer not running).
/// Error type returned by `sink_from_str`.
#[cfg(feature = "rerun")]
pub type SinkError = Box<dyn std::error::Error + Send + Sync>;

#[cfg(feature = "rerun")]
pub fn sink_from_str(addr: &str, app_id: &str) -> Result<ProofSink, SinkError> {
    let conn = RerunConnection::from_str(addr);
    let sink = RerunSink::connect(conn, app_id)?;
    Ok(ProofSink::new(sink))
}
