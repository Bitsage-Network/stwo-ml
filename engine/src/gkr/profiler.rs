//! Phase profiler for GKR prover and outer proving pipeline — zero overhead when disabled.
//!
//! # Overview
//!
//! The profiler tracks wall-clock time, Poseidon hash counts, and sub-phase breakdowns
//! for both the inner GKR prover and the outer proving pipeline (forward pass, attention,
//! commitments, unified STARK, serialization).
//!
//! # Activation
//!
//! - **Env var**: `STWO_PROFILE=1` enables profiling globally.
//! - **Programmatic**: `PhaseProfiler::new(true)` or `crate::set_profile(true)`.
//! - When disabled, all methods are no-ops — zero allocation, zero timing overhead.
//!
//! # Phase Names
//!
//! **GKR inner phases** (recorded by `gkr/prover.rs`):
//! - `channel_init` — Fiat-Shamir channel setup
//! - `layer_walk` — per-layer sumcheck reductions (contains MatMul sub-breakdown)
//! - `final_claims` — final claim extraction and binding
//! - `weight_binding` — aggregated weight commitment verification
//!
//! **Outer phases** (recorded by `aggregation.rs::prove_model_pure_gkr_inner`):
//! - `forward_pass` — model forward execution (contains per-op-type sub-breakdown)
//! - `gkr_proof` — GKR proving pass
//! - `attention_proofs` — attention layer proving (independent from GKR)
//! - `commitments_io` — layer chain + IO + LayerNorm commitment computation
//! - `commitments_kv_cache` — KV-cache rehash commitment
//! - `unified_stark` — non-matmul component STARK proving
//!
//! **Serialization phases** (recorded by `starknet.rs`):
//! - `serialize` — proof → calldata felts
//! - `self_verify` — replay_verify_serialized_proof
//!
//! # JSON Output Format
//!
//! ```json
//! {
//!   "total_elapsed_ms": 107920.0,
//!   "total_hashes": 8602,
//!   "phases": [...],
//!   "layer_type_counts": { "matmul": 160, "activation": 120 },
//!   "forward_pass_ops": { "matmul": { "count": 160, "elapsed_ms": 1200.0 }, ... },
//!   "serialization": { "serialize_ms": 2300.0, "self_verify_ms": 8100.0 }
//! }
//! ```
//!
//! # Extending
//!
//! To add a new phase:
//! 1. Call `profiler.begin_phase("new_phase", hash_count)` before the section.
//! 2. Call `profiler.end_phase(hash_count)` after.
//! 3. For sub-phase accumulators, add a new struct and `record_*` method.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Timing data for a single proving phase.
#[derive(Debug, Clone)]
pub struct PhaseTiming {
    /// Wall-clock time spent in this phase.
    pub elapsed: Duration,
    /// Number of Poseidon hashes (mix + draw) during this phase.
    pub hash_count: u64,
    /// Human-readable detail, e.g. "160 matmuls, 120 activations".
    pub detail: Option<String>,
}

/// Sub-phase timing within a single MatMul reduction.
#[derive(Debug, Clone, Default)]
pub struct MatMulTimings {
    /// Time spent seeding channel with dims + claim.
    pub channel_seed: Duration,
    /// Time spent in backend reduction (GPU or CPU sumcheck).
    pub backend_reduce: Duration,
    /// Time spent binding final evals to transcript.
    pub output_bind: Duration,
}

/// Accumulated MatMul sub-phase statistics across all reductions.
#[derive(Debug, Clone, Default)]
struct MatMulAccumulator {
    count: usize,
    channel_seed: Duration,
    backend_reduce: Duration,
    output_bind: Duration,
}

/// Per-op-type timing accumulators for the forward pass.
#[derive(Debug, Clone, Default)]
pub struct ForwardPassTimings {
    pub matmul: Duration,
    pub matmul_count: usize,
    pub activation: Duration,
    pub activation_count: usize,
    pub add: Duration,
    pub add_count: usize,
    pub mul: Duration,
    pub mul_count: usize,
    pub layernorm: Duration,
    pub layernorm_count: usize,
    pub rmsnorm: Duration,
    pub rmsnorm_count: usize,
    pub attention: Duration,
    pub attention_count: usize,
    pub embedding: Duration,
    pub embedding_count: usize,
    pub other: Duration,
    pub other_count: usize,
}

/// Serialization sub-phase timing (proof → calldata + self-verification).
#[derive(Debug, Clone, Default)]
pub struct SerializationTimings {
    /// Time spent serializing the proof into calldata felts.
    pub serialize: Duration,
    /// Time spent in `replay_verify_serialized_proof`.
    pub self_verify: Duration,
}

impl SerializationTimings {
    /// Total serialization time (serialize + self_verify).
    pub fn total(&self) -> Duration {
        self.serialize + self.self_verify
    }
}

/// Lightweight phase profiler for GKR prover instrumentation.
///
/// When `enabled` is false, all methods are no-ops with zero overhead.
/// Designed to be created per-proof and printed/exported at the end.
pub struct PhaseProfiler {
    phases: BTreeMap<&'static str, PhaseTiming>,
    /// Insertion order tracking (BTreeMap sorts alphabetically).
    phase_order: Vec<&'static str>,
    current_phase: Option<(&'static str, Instant, u64)>,
    matmul_acc: MatMulAccumulator,
    /// Per-layer-type counters within the layer_walk phase.
    layer_type_counts: BTreeMap<&'static str, usize>,
    /// Per-op-type forward pass timing accumulators.
    forward_ops: ForwardPassTimings,
    /// Serialization sub-phase timing (set externally by starknet.rs).
    serialization: Option<SerializationTimings>,
    enabled: bool,
}

impl PhaseProfiler {
    /// Create a new profiler. When `enabled` is false, all operations are no-ops.
    pub fn new(enabled: bool) -> Self {
        Self {
            phases: BTreeMap::new(),
            phase_order: Vec::new(),
            current_phase: None,
            matmul_acc: MatMulAccumulator::default(),
            layer_type_counts: BTreeMap::new(),
            forward_ops: ForwardPassTimings::default(),
            serialization: None,
            enabled,
        }
    }

    /// Returns true if profiling is active.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Begin a named phase. `hash_count` is the channel's current hash count.
    #[inline]
    pub fn begin_phase(&mut self, name: &'static str, hash_count: u64) {
        if !self.enabled {
            return;
        }
        self.current_phase = Some((name, Instant::now(), hash_count));
    }

    /// End the current phase. `hash_count` is the channel's current hash count.
    #[inline]
    pub fn end_phase(&mut self, hash_count: u64) {
        if !self.enabled {
            return;
        }
        if let Some((name, start, start_hashes)) = self.current_phase.take() {
            let timing = PhaseTiming {
                elapsed: start.elapsed(),
                hash_count: hash_count.saturating_sub(start_hashes),
                detail: None,
            };
            if !self.phases.contains_key(name) {
                self.phase_order.push(name);
            }
            self.phases.insert(name, timing);
        }
    }

    /// End the current phase with an attached detail string.
    #[inline]
    pub fn end_phase_with_detail(&mut self, hash_count: u64, detail: String) {
        if !self.enabled {
            return;
        }
        if let Some((name, start, start_hashes)) = self.current_phase.take() {
            let timing = PhaseTiming {
                elapsed: start.elapsed(),
                hash_count: hash_count.saturating_sub(start_hashes),
                detail: Some(detail),
            };
            if !self.phases.contains_key(name) {
                self.phase_order.push(name);
            }
            self.phases.insert(name, timing);
        }
    }

    /// Record sub-phase timing for a single MatMul reduction.
    #[inline]
    pub fn record_matmul(&mut self, timings: MatMulTimings) {
        if !self.enabled {
            return;
        }
        self.matmul_acc.count += 1;
        self.matmul_acc.channel_seed += timings.channel_seed;
        self.matmul_acc.backend_reduce += timings.backend_reduce;
        self.matmul_acc.output_bind += timings.output_bind;
    }

    /// Record that a layer of the given type was processed during the walk.
    #[inline]
    pub fn record_layer_type(&mut self, kind: &'static str) {
        if !self.enabled {
            return;
        }
        *self.layer_type_counts.entry(kind).or_insert(0) += 1;
    }

    /// Record a forward-pass operation's elapsed time by op kind.
    ///
    /// Called once per graph node during the forward pass loop.
    /// `kind` should be one of: `"matmul"`, `"activation"`, `"add"`, `"mul"`,
    /// `"layernorm"`, `"rmsnorm"`, `"attention"`, `"embedding"`, or `"other"`.
    #[inline]
    pub fn record_forward_op(&mut self, kind: &'static str, elapsed: Duration) {
        if !self.enabled {
            return;
        }
        match kind {
            "matmul" => {
                self.forward_ops.matmul += elapsed;
                self.forward_ops.matmul_count += 1;
            }
            "activation" => {
                self.forward_ops.activation += elapsed;
                self.forward_ops.activation_count += 1;
            }
            "add" => {
                self.forward_ops.add += elapsed;
                self.forward_ops.add_count += 1;
            }
            "mul" => {
                self.forward_ops.mul += elapsed;
                self.forward_ops.mul_count += 1;
            }
            "layernorm" => {
                self.forward_ops.layernorm += elapsed;
                self.forward_ops.layernorm_count += 1;
            }
            "rmsnorm" => {
                self.forward_ops.rmsnorm += elapsed;
                self.forward_ops.rmsnorm_count += 1;
            }
            "attention" => {
                self.forward_ops.attention += elapsed;
                self.forward_ops.attention_count += 1;
            }
            "embedding" => {
                self.forward_ops.embedding += elapsed;
                self.forward_ops.embedding_count += 1;
            }
            _ => {
                self.forward_ops.other += elapsed;
                self.forward_ops.other_count += 1;
            }
        }
    }

    /// Record serialization sub-phase timings (set by `starknet.rs`).
    #[inline]
    pub fn record_serialization(&mut self, timings: SerializationTimings) {
        if !self.enabled {
            return;
        }
        self.serialization = Some(timings);
    }

    /// Access the forward pass sub-timings.
    pub fn forward_ops(&self) -> &ForwardPassTimings {
        &self.forward_ops
    }

    /// Access the serialization sub-timings.
    pub fn serialization(&self) -> Option<&SerializationTimings> {
        self.serialization.as_ref()
    }

    /// Access the collected phase timings.
    pub fn phases(&self) -> &BTreeMap<&'static str, PhaseTiming> {
        &self.phases
    }

    /// Total wall-clock time across all phases.
    pub fn total_elapsed(&self) -> Duration {
        self.phases.values().map(|p| p.elapsed).sum()
    }

    /// Total hash count across all phases.
    pub fn total_hashes(&self) -> u64 {
        self.phases.values().map(|p| p.hash_count).sum()
    }

    /// Merge phases from an inner profiler (e.g. GKR sub-phases) into this
    /// outer profiler with a `"gkr/"` prefix so they appear in the JSON output.
    ///
    /// Existing phases in `self` are preserved. Inner phases are prefixed to
    /// avoid name collisions (e.g. `"channel_init"` → `"gkr/channel_init"`).
    pub fn merge_inner(&mut self, inner: &PhaseProfiler) {
        if !self.enabled || !inner.enabled {
            return;
        }
        for &name in &inner.phase_order {
            if let Some(timing) = inner.phases.get(name) {
                // Leak a prefixed name — same lifetime pattern as other &'static str phases
                let prefixed: &'static str =
                    Box::leak(format!("gkr/{name}").into_boxed_str());
                if !self.phases.contains_key(prefixed) {
                    self.phase_order.push(prefixed);
                }
                self.phases.insert(prefixed, timing.clone());
            }
        }
        // Merge matmul accumulator
        self.matmul_acc.count += inner.matmul_acc.count;
        self.matmul_acc.channel_seed += inner.matmul_acc.channel_seed;
        self.matmul_acc.backend_reduce += inner.matmul_acc.backend_reduce;
        self.matmul_acc.output_bind += inner.matmul_acc.output_bind;
        // Merge layer type counts
        for (&kind, &count) in &inner.layer_type_counts {
            *self.layer_type_counts.entry(kind).or_insert(0) += count;
        }
    }

    /// Format a human-readable summary table for stderr output.
    pub fn summary(&self) -> String {
        if self.phases.is_empty() {
            return String::new();
        }

        let total_elapsed = self.total_elapsed();
        let total_secs = total_elapsed.as_secs_f64().max(1e-9);
        let total_hashes = self.total_hashes();

        let mut lines = Vec::new();
        lines.push("[profile] ── Outer Phase Breakdown ────────────────────".to_string());

        for &name in &self.phase_order {
            if let Some(timing) = self.phases.get(name) {
                let secs = timing.elapsed.as_secs_f64();
                let pct = (secs / total_secs) * 100.0;
                lines.push(format!(
                    "[profile]  {:<20} {:>7.3}s {:>6} hashes {:>5.1}%",
                    name, secs, timing.hash_count, pct,
                ));

                // Forward-pass sub-breakdown
                if name == "forward_pass" {
                    self.format_forward_pass_breakdown(&mut lines);
                }

                // Sub-phase breakdown for layer_walk (GKR inner)
                if name == "layer_walk" && self.matmul_acc.count > 0 {
                    let mm = &self.matmul_acc;
                    let mm_total = mm.channel_seed + mm.backend_reduce + mm.output_bind;
                    lines.push(format!(
                        "[profile]    ├─ matmul ({})    {:>7.3}s (channel: {:.3}s, reduce: {:.3}s, bind: {:.3}s)",
                        mm.count,
                        mm_total.as_secs_f64(),
                        mm.channel_seed.as_secs_f64(),
                        mm.backend_reduce.as_secs_f64(),
                        mm.output_bind.as_secs_f64(),
                    ));
                    // Show non-matmul layer types
                    for (kind, count) in &self.layer_type_counts {
                        if *kind != "matmul" {
                            lines.push(format!(
                                "[profile]    ├─ {} ({})",
                                kind, count,
                            ));
                        }
                    }
                }

                if let Some(ref detail) = timing.detail {
                    lines.push(format!("[profile]    └─ {}", detail));
                }
            }
        }

        lines.push("[profile] ─────────────────────────────────────────────".to_string());
        lines.push(format!(
            "[profile]  {:<20} {:>7.3}s {:>6} hashes",
            "TOTAL",
            total_secs,
            total_hashes,
        ));

        // Serialization section
        if let Some(ref ser) = self.serialization {
            lines.push(String::new());
            lines.push("[profile] ── Serialization ────────────────────────────".to_string());
            lines.push(format!(
                "[profile]  {:<20} {:>7.3}s",
                "serialize",
                ser.serialize.as_secs_f64(),
            ));
            lines.push(format!(
                "[profile]  {:<20} {:>7.3}s",
                "self_verify",
                ser.self_verify.as_secs_f64(),
            ));
            lines.push(format!(
                "[profile]  {:<20} {:>7.3}s",
                "TOTAL",
                ser.total().as_secs_f64(),
            ));
        }

        lines.join("\n")
    }

    /// Format forward-pass op sub-breakdown lines.
    fn format_forward_pass_breakdown(&self, lines: &mut Vec<String>) {
        let ops = &self.forward_ops;
        let entries: Vec<(&str, usize, Duration)> = [
            ("matmul", ops.matmul_count, ops.matmul),
            ("activation", ops.activation_count, ops.activation),
            ("add", ops.add_count, ops.add),
            ("mul", ops.mul_count, ops.mul),
            ("layernorm", ops.layernorm_count, ops.layernorm),
            ("rmsnorm", ops.rmsnorm_count, ops.rmsnorm),
            ("attention", ops.attention_count, ops.attention),
            ("embedding", ops.embedding_count, ops.embedding),
            ("other", ops.other_count, ops.other),
        ]
        .into_iter()
        .filter(|(_, count, _)| *count > 0)
        .collect();

        for (i, (name, count, dur)) in entries.iter().enumerate() {
            let prefix = if i + 1 < entries.len() { "├─" } else { "└─" };
            lines.push(format!(
                "[profile]    {} {} ({})  {:>7.3}s",
                prefix,
                name,
                count,
                dur.as_secs_f64(),
            ));
        }
    }

    /// Serialize profile data as JSON string (for `--profile` file output).
    pub fn to_json(&self) -> String {
        let total_elapsed = self.total_elapsed();
        let total_hashes = self.total_hashes();

        let mut phases_json = Vec::new();
        for &name in &self.phase_order {
            if let Some(timing) = self.phases.get(name) {
                phases_json.push(format!(
                    "    {{\n      \"name\": \"{}\",\n      \"elapsed_ms\": {:.1},\n      \"hash_count\": {},\n      \"detail\": {}{}",
                    name,
                    timing.elapsed.as_secs_f64() * 1000.0,
                    timing.hash_count,
                    match &timing.detail {
                        Some(d) => format!("\"{}\"", d.replace('"', "\\\"")),
                        None => "null".to_string(),
                    },
                    if name == "layer_walk" && self.matmul_acc.count > 0 {
                        let mm = &self.matmul_acc;
                        format!(
                            ",\n      \"matmul_breakdown\": {{\n        \"count\": {},\n        \"channel_seed_ms\": {:.1},\n        \"backend_reduce_ms\": {:.1},\n        \"output_bind_ms\": {:.1}\n      }}\n    }}",
                            mm.count,
                            mm.channel_seed.as_secs_f64() * 1000.0,
                            mm.backend_reduce.as_secs_f64() * 1000.0,
                            mm.output_bind.as_secs_f64() * 1000.0,
                        )
                    } else {
                        "\n    }".to_string()
                    },
                ));
            }
        }

        let layer_types_json: Vec<String> = self
            .layer_type_counts
            .iter()
            .map(|(k, v)| format!("    \"{}\": {}", k, v))
            .collect();

        // Forward-pass ops JSON
        let fwd = &self.forward_ops;
        let fwd_entries: Vec<String> = [
            ("matmul", fwd.matmul_count, fwd.matmul),
            ("activation", fwd.activation_count, fwd.activation),
            ("add", fwd.add_count, fwd.add),
            ("mul", fwd.mul_count, fwd.mul),
            ("layernorm", fwd.layernorm_count, fwd.layernorm),
            ("rmsnorm", fwd.rmsnorm_count, fwd.rmsnorm),
            ("attention", fwd.attention_count, fwd.attention),
            ("embedding", fwd.embedding_count, fwd.embedding),
            ("other", fwd.other_count, fwd.other),
        ]
        .into_iter()
        .filter(|(_, count, _)| *count > 0)
        .map(|(name, count, dur)| {
            format!(
                "    \"{}\": {{ \"count\": {}, \"elapsed_ms\": {:.1} }}",
                name,
                count,
                dur.as_secs_f64() * 1000.0,
            )
        })
        .collect();

        // Serialization JSON
        let ser_json = match &self.serialization {
            Some(s) => format!(
                ",\n  \"serialization\": {{\n    \"serialize_ms\": {:.1},\n    \"self_verify_ms\": {:.1}\n  }}",
                s.serialize.as_secs_f64() * 1000.0,
                s.self_verify.as_secs_f64() * 1000.0,
            ),
            None => String::new(),
        };

        format!(
            "{{\n  \"total_elapsed_ms\": {:.1},\n  \"total_hashes\": {},\n  \"phases\": [\n{}\n  ],\n  \"layer_type_counts\": {{\n{}\n  }},\n  \"forward_pass_ops\": {{\n{}\n  }}{}\n}}",
            total_elapsed.as_secs_f64() * 1000.0,
            total_hashes,
            phases_json.join(",\n"),
            layer_types_json.join(",\n"),
            fwd_entries.join(",\n"),
            ser_json,
        )
    }
}

use std::cell::RefCell;

thread_local! {
    /// Stores the last profiler JSON output for retrieval by the CLI.
    static LAST_PROFILE_JSON: RefCell<Option<String>> = const { RefCell::new(None) };
    /// Stores the inner GKR profiler for retrieval by the outer pipeline.
    static LAST_INNER_PROFILER: RefCell<Option<PhaseProfiler>> = const { RefCell::new(None) };
}

/// Store profiler JSON in thread-local storage (called by aggregation.rs after proving).
pub fn store_profile_json(json: String) {
    LAST_PROFILE_JSON.with(|cell| {
        *cell.borrow_mut() = Some(json);
    });
}

/// Retrieve and consume the last profiler JSON from thread-local storage.
pub fn take_profile_json() -> Option<String> {
    LAST_PROFILE_JSON.with(|cell| cell.borrow_mut().take())
}

/// Store the inner GKR profiler for later merging into the outer pipeline profiler.
pub fn store_inner_profiler(profiler: PhaseProfiler) {
    LAST_INNER_PROFILER.with(|cell| {
        *cell.borrow_mut() = Some(profiler);
    });
}

/// Retrieve and consume the inner GKR profiler from thread-local storage.
pub fn take_inner_profiler() -> Option<PhaseProfiler> {
    LAST_INNER_PROFILER.with(|cell| cell.borrow_mut().take())
}

/// Print serialization timing to stderr (env-var gated, standalone use from starknet.rs).
pub fn print_serialization_timing(timings: &SerializationTimings) {
    if !crate::is_profile() {
        return;
    }
    eprintln!("[profile] ── Serialization ────────────────────────────");
    eprintln!(
        "[profile]  {:<20} {:>7.3}s",
        "serialize",
        timings.serialize.as_secs_f64(),
    );
    eprintln!(
        "[profile]  {:<20} {:>7.3}s",
        "self_verify",
        timings.self_verify.as_secs_f64(),
    );
    eprintln!(
        "[profile]  {:<20} {:>7.3}s",
        "TOTAL",
        timings.total().as_secs_f64(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_disabled_is_noop() {
        let mut p = PhaseProfiler::new(false);
        p.begin_phase("test", 0);
        p.end_phase(10);
        p.record_matmul(MatMulTimings {
            channel_seed: Duration::from_millis(5),
            backend_reduce: Duration::from_millis(100),
            output_bind: Duration::from_millis(2),
        });
        p.record_forward_op("matmul", Duration::from_millis(10));
        p.record_serialization(SerializationTimings {
            serialize: Duration::from_millis(100),
            self_verify: Duration::from_millis(200),
        });
        assert!(p.phases().is_empty());
        assert_eq!(p.summary(), "");
        assert_eq!(p.forward_ops().matmul_count, 0);
        assert!(p.serialization().is_none());
    }

    #[test]
    fn test_profiler_records_phases() {
        let mut p = PhaseProfiler::new(true);

        p.begin_phase("channel_init", 0);
        std::thread::sleep(Duration::from_millis(5));
        p.end_phase(42);

        p.begin_phase("layer_walk", 42);
        p.record_matmul(MatMulTimings {
            channel_seed: Duration::from_millis(1),
            backend_reduce: Duration::from_millis(10),
            output_bind: Duration::from_millis(1),
        });
        p.record_layer_type("matmul");
        p.record_layer_type("activation");
        std::thread::sleep(Duration::from_millis(5));
        p.end_phase(100);

        assert_eq!(p.phases().len(), 2);
        assert!(p.phases()["channel_init"].hash_count == 42);
        assert!(p.phases()["layer_walk"].hash_count == 58);
        assert_eq!(p.total_hashes(), 100);

        let summary = p.summary();
        assert!(summary.contains("channel_init"));
        assert!(summary.contains("layer_walk"));
        assert!(summary.contains("matmul (1)"));
        assert!(summary.contains("TOTAL"));
    }

    #[test]
    fn test_profiler_json_output() {
        let mut p = PhaseProfiler::new(true);
        p.begin_phase("test_phase", 0);
        p.end_phase(10);

        p.record_forward_op("matmul", Duration::from_millis(50));
        p.record_forward_op("matmul", Duration::from_millis(30));
        p.record_forward_op("activation", Duration::from_millis(10));

        p.record_serialization(SerializationTimings {
            serialize: Duration::from_millis(100),
            self_verify: Duration::from_millis(200),
        });

        let json = p.to_json();
        assert!(json.contains("\"test_phase\""));
        assert!(json.contains("\"total_hashes\": 10"));
        assert!(json.contains("\"forward_pass_ops\""));
        assert!(json.contains("\"matmul\""));
        assert!(json.contains("\"count\": 2"));
        assert!(json.contains("\"serialization\""));
        assert!(json.contains("\"serialize_ms\""));
        assert!(json.contains("\"self_verify_ms\""));
    }

    #[test]
    fn test_phase_order_preserved() {
        let mut p = PhaseProfiler::new(true);
        p.begin_phase("zebra", 0);
        p.end_phase(1);
        p.begin_phase("alpha", 1);
        p.end_phase(2);
        p.begin_phase("middle", 2);
        p.end_phase(3);

        // Order should be insertion order, not alphabetical
        assert_eq!(p.phase_order, vec!["zebra", "alpha", "middle"]);
    }

    #[test]
    fn test_forward_pass_breakdown() {
        let mut p = PhaseProfiler::new(true);
        p.begin_phase("forward_pass", 0);
        p.record_forward_op("matmul", Duration::from_millis(100));
        p.record_forward_op("matmul", Duration::from_millis(50));
        p.record_forward_op("activation", Duration::from_millis(20));
        p.record_forward_op("layernorm", Duration::from_millis(10));
        p.end_phase(0);

        let summary = p.summary();
        assert!(summary.contains("matmul (2)"));
        assert!(summary.contains("activation (1)"));
        assert!(summary.contains("layernorm (1)"));
        // Should use tree chars
        assert!(summary.contains("├─") || summary.contains("└─"));
    }

    #[test]
    fn test_serialization_in_summary() {
        let mut p = PhaseProfiler::new(true);
        p.begin_phase("gkr_proof", 0);
        p.end_phase(100);
        p.record_serialization(SerializationTimings {
            serialize: Duration::from_millis(2300),
            self_verify: Duration::from_millis(8100),
        });

        let summary = p.summary();
        assert!(summary.contains("Serialization"));
        assert!(summary.contains("serialize"));
        assert!(summary.contains("self_verify"));
    }

    #[test]
    fn test_store_and_take_profile_json() {
        // Ensure thread-local storage works correctly
        let json = r#"{"test": true}"#.to_string();
        store_profile_json(json.clone());
        let retrieved = take_profile_json();
        assert_eq!(retrieved, Some(json));
        // Second take should return None (consumed)
        assert_eq!(take_profile_json(), None);
    }

    #[test]
    fn test_profiler_inner_merge() {
        // Create an inner profiler simulating GKR sub-phases
        let mut inner = PhaseProfiler::new(true);
        inner.begin_phase("channel_init", 0);
        std::thread::sleep(Duration::from_millis(1));
        inner.end_phase(42);
        inner.begin_phase("layer_walk", 42);
        inner.record_matmul(MatMulTimings {
            channel_seed: Duration::from_millis(5),
            backend_reduce: Duration::from_millis(50),
            output_bind: Duration::from_millis(3),
        });
        inner.record_layer_type("matmul");
        inner.record_layer_type("matmul");
        inner.record_layer_type("activation");
        inner.end_phase(100);

        // Create an outer profiler and merge
        let mut outer = PhaseProfiler::new(true);
        outer.begin_phase("forward_pass", 0);
        outer.end_phase(0);
        outer.begin_phase("gkr_proof", 0);
        outer.end_phase(0);

        outer.merge_inner(&inner);

        // Inner phases should appear with gkr/ prefix
        assert!(outer.phases().contains_key("gkr/channel_init"));
        assert!(outer.phases().contains_key("gkr/layer_walk"));
        assert_eq!(outer.phases()["gkr/channel_init"].hash_count, 42);
        assert_eq!(outer.phases()["gkr/layer_walk"].hash_count, 58);

        // Matmul accumulator should be merged
        assert_eq!(outer.matmul_acc.count, 1);
        assert_eq!(outer.matmul_acc.channel_seed, Duration::from_millis(5));

        // Layer type counts should be merged
        assert_eq!(outer.layer_type_counts["matmul"], 2);
        assert_eq!(outer.layer_type_counts["activation"], 1);

        // Phase order should have outer phases first, then gkr/ prefixed
        assert_eq!(outer.phase_order[0], "forward_pass");
        assert_eq!(outer.phase_order[1], "gkr_proof");
        assert_eq!(outer.phase_order[2], "gkr/channel_init");
        assert_eq!(outer.phase_order[3], "gkr/layer_walk");

        // JSON output should include inner phases
        let json = outer.to_json();
        assert!(json.contains("\"gkr/channel_init\""));
        assert!(json.contains("\"gkr/layer_walk\""));
    }

    #[test]
    fn test_store_and_take_inner_profiler() {
        let mut p = PhaseProfiler::new(true);
        p.begin_phase("test", 0);
        p.end_phase(10);
        store_inner_profiler(p);
        let retrieved = take_inner_profiler();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().total_hashes(), 10);
        // Second take should return None (consumed)
        assert!(take_inner_profiler().is_none());
    }
}
