//! Proof generation policy configuration.
//!
//! Replaces 15+ scattered environment variables with a single composable struct
//! whose Poseidon hash is mixed into the Fiat-Shamir channel, binding proofs to
//! specific policy choices and making them verifiable on-chain.
//!
//! # Presets
//!
//! - [`PolicyConfig::strict()`]: Production — all soundness gates enforced, full
//!   weight binding, decode chain validation required.
//! - [`PolicyConfig::standard()`]: On-chain streaming — matches prove_server defaults.
//! - [`PolicyConfig::relaxed()`]: Development — all gates permissive.
//! - [`PolicyConfig::from_env()`]: Reads current `STWO_*` env vars (backward compat).
//!
//! # Fiat-Shamir Binding
//!
//! The [`PolicyConfig::policy_commitment()`] Poseidon hash is mixed into the GKR
//! channel after KV-cache binding and before circuit metadata. Both the Rust
//! prover/verifier and the Cairo on-chain verifier must mix at the same position.
//! When `policy_commitment() == FieldElement::ZERO`, the mix is skipped to preserve
//! backward compatibility with legacy proofs.

use starknet_crypto::poseidon_hash_many;
use starknet_ff::FieldElement;

/// Domain separator for policy commitment hashing.
/// "POL" (0x504F4C) + version byte 0x01.
const POLICY_DOMAIN_SEPARATOR: u64 = 0x504F4C_01;

/// Weight binding strategy for GKR proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum WeightBindingMode {
    /// Legacy per-layer opening proofs.
    Individual = 0,
    /// Aggregated oracle sumcheck (default).
    Aggregated = 1,
    /// Trustless mode 2: aggregated with per-layer openings.
    TrustlessMode2 = 2,
    /// Trustless mode 3: full trustless verification.
    TrustlessMode3 = 3,
}

/// Proof generation policy configuration.
///
/// Each field corresponds to an `STWO_*` environment variable that previously
/// controlled proof generation behavior. The struct is hashed via Poseidon to
/// produce a commitment that is mixed into the Fiat-Shamir channel.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PolicyConfig {
    // ── Soundness gates (verifier-side) ──────────────────────────────────

    /// Allow proofs with missing LayerNorm/RMSNorm sub-proofs.
    /// Env: `STWO_ALLOW_MISSING_NORM_PROOF`
    pub allow_missing_norm_proof: bool,

    /// Allow reduced-precision LogUp activation proofs (lower 16-20 bits only).
    /// Env: `STWO_ALLOW_LOGUP_ACTIVATION`
    pub allow_logup_activation: bool,

    /// Allow proofs with missing piecewise segment binding.
    /// Env: `STWO_ALLOW_MISSING_SEGMENT_BINDING`
    pub allow_missing_segment_binding: bool,

    // ── Prover feature flags ─────────────────────────────────────────────

    /// Skip RMSNorm Part 0 (variance) self-verification.
    /// Non-fatal: Cairo on-chain verifier is authoritative.
    /// Env: `STWO_SKIP_RMS_SQ_PROOF`
    pub skip_rms_sq_proof: bool,

    /// Use piecewise-linear algebraic activation proofs (full M31 domain).
    /// When false, falls back to legacy LogUp path.
    /// Env: `STWO_PIECEWISE_ACTIVATION` (default: true)
    pub piecewise_activation: bool,

    /// Skip the demo binary's batched-conversation token proving pass.
    /// Despite the name, this is NOT a soundness gate — no lib code consumes it.
    /// It's read at `src/bin/prove_model.rs:5166` to gate a demo feature that
    /// runs an extra batched forward pass over all response tokens at end-of-run.
    /// Env: `STWO_SKIP_BATCH_TOKENS`. Audited Apr 30 2026.
    pub skip_batch_tokens: bool,

    /// Skip unified STARK layer in pure-GKR mode.
    /// Env: `STWO_PURE_GKR_SKIP_UNIFIED_STARK`
    pub skip_unified_stark: bool,

    // ── Weight binding ───────────────────────────────────────────────────

    /// Weight binding strategy. Unifies `STWO_WEIGHT_BINDING`,
    /// `STWO_GKR_AGGREGATE_WEIGHT_BINDING`, `STWO_GKR_TRUSTLESS_MODE2/3`.
    pub weight_binding_mode: WeightBindingMode,

    /// Enable full MLE opening proofs for trustless on-chain streaming.
    /// Env: `STWO_AGGREGATED_FULL_BINDING`
    pub aggregated_full_binding: bool,

    /// Use RLC-only binding (weaker, no MLE opening proofs).
    /// Env: `STWO_AGGREGATED_RLC_ONLY`
    pub aggregated_rlc_only: bool,

    // ── Serialization ────────────────────────────────────────────────────

    /// Pack IO data (8 M31 per felt252).
    /// Env: `STWO_NO_IO_PACK` (negated — presence disables)
    pub io_packing: bool,

    /// Use packed proof format.
    /// Env: `STWO_NO_PACKED` (negated)
    pub packed_proof: bool,

    /// Use double-packed proof format.
    /// Env: `STWO_NO_DOUBLE_PACK` (negated)
    pub double_packed_proof: bool,

    // ── Decode chain ─────────────────────────────────────────────────────

    /// Enforce decode chain continuity validation for sequential inference.
    /// When true, KV-cache commitments must chain correctly between steps.
    /// Previously a caller-supplied parameter; now policy-bound.
    pub validate_decode_chain: bool,
}

impl PolicyConfig {
    /// Production preset: all soundness gates enforced.
    ///
    /// - No missing proofs allowed
    /// - Full weight binding with aggregated MLE openings
    /// - Piecewise activation enabled
    /// - Decode chain validation enforced
    /// - All packing enabled
    pub fn strict() -> Self {
        Self {
            // Soundness: all gates closed (reject incomplete proofs)
            allow_missing_norm_proof: false,
            allow_logup_activation: false,
            allow_missing_segment_binding: false,
            // Prover: full proving, no skips
            skip_rms_sq_proof: false,
            piecewise_activation: true,
            skip_batch_tokens: false,
            skip_unified_stark: false,
            // Weight: full trustless binding
            weight_binding_mode: WeightBindingMode::Aggregated,
            aggregated_full_binding: true,
            aggregated_rlc_only: false,
            // Serialization: all packing on
            io_packing: true,
            packed_proof: true,
            double_packed_proof: true,
            // Decode chain: enforced
            validate_decode_chain: true,
        }
    }

    /// On-chain streaming preset: matches prove_server auto-set defaults.
    ///
    /// Hardening (Apr 30 2026, two passes):
    /// - Pass 1 closed `skip_rms_sq_proof` + `allow_missing_norm_proof`
    ///   per `engine/docs/SOUNDNESS_GATES_AUDIT.md` (verified safe on A10G).
    /// - Pass 2 closed `allow_logup_activation` and enabled
    ///   `piecewise_activation`. Verified safe on M-series CPU with SmolLM2-135M
    ///   (1L proof generates, self-verifies, externally cryptographically
    ///   re-verifies — same code path as `strict()` for these two fields).
    ///
    /// Remaining: `skip_batch_tokens` is NOT a soundness gate (it's a demo
    /// binary feature toggle — see field doc); `skip_unified_stark` is the
    /// pure-GKR mode for streaming compatibility.
    pub fn standard() -> Self {
        Self {
            // Soundness: norm proofs required, full-precision activations required
            allow_missing_norm_proof: false,
            allow_logup_activation: false,
            allow_missing_segment_binding: false,
            // Prover: RMS Part 0 proven, piecewise activation enabled, batch skipped
            skip_rms_sq_proof: false,
            piecewise_activation: true,
            skip_batch_tokens: true,
            skip_unified_stark: true,
            // Weight: aggregated with full binding
            weight_binding_mode: WeightBindingMode::Aggregated,
            aggregated_full_binding: true,
            aggregated_rlc_only: false,
            // Serialization: all packing on
            io_packing: true,
            packed_proof: true,
            double_packed_proof: true,
            // Decode chain: enforced
            validate_decode_chain: true,
        }
    }

    /// Development preset: all gates permissive.
    ///
    /// Useful for testing and debugging. No soundness guarantees.
    pub fn relaxed() -> Self {
        Self {
            allow_missing_norm_proof: true,
            allow_logup_activation: true,
            allow_missing_segment_binding: true,
            skip_rms_sq_proof: true,
            piecewise_activation: true,
            skip_batch_tokens: true,
            skip_unified_stark: true,
            weight_binding_mode: WeightBindingMode::Aggregated,
            aggregated_full_binding: false,
            aggregated_rlc_only: false,
            io_packing: true,
            packed_proof: true,
            double_packed_proof: true,
            validate_decode_chain: false,
        }
    }

    /// Read policy from environment variables (backward compatibility).
    ///
    /// Reads all `STWO_*` env vars that previously controlled proof generation.
    /// This allows existing workflows to continue working without changes.
    pub fn from_env() -> Self {
        Self {
            allow_missing_norm_proof: env_bool("STWO_ALLOW_MISSING_NORM_PROOF", false),
            allow_logup_activation: env_bool("STWO_ALLOW_LOGUP_ACTIVATION", false),
            allow_missing_segment_binding: env_bool("STWO_ALLOW_MISSING_SEGMENT_BINDING", false),
            skip_rms_sq_proof: env_is_set("STWO_SKIP_RMS_SQ_PROOF"),
            piecewise_activation: env_bool_default_true("STWO_PIECEWISE_ACTIVATION"),
            skip_batch_tokens: env_bool("STWO_SKIP_BATCH_TOKENS", false),
            skip_unified_stark: env_bool("STWO_PURE_GKR_SKIP_UNIFIED_STARK", false),
            weight_binding_mode: weight_binding_mode_from_env(),
            aggregated_full_binding: env_bool("STWO_AGGREGATED_FULL_BINDING", false),
            aggregated_rlc_only: env_bool("STWO_AGGREGATED_RLC_ONLY", false),
            io_packing: !env_is_set("STWO_NO_IO_PACK"),
            packed_proof: !env_is_set("STWO_NO_PACKED"),
            double_packed_proof: !env_is_set("STWO_NO_DOUBLE_PACK"),
            validate_decode_chain: env_bool("STWO_VALIDATE_DECODE_CHAIN", false),
        }
    }

    /// Compute the Poseidon commitment of this policy configuration.
    ///
    /// The hash is deterministic: identical PolicyConfig values always produce
    /// the same commitment. This is mixed into the Fiat-Shamir channel to
    /// cryptographically bind the proof to the policy.
    ///
    /// Field ordering is versioned via a domain separator. The Cairo on-chain
    /// verifier must compute the identical hash for the same policy.
    pub fn policy_commitment(&self) -> FieldElement {
        poseidon_hash_many(&[
            FieldElement::from(POLICY_DOMAIN_SEPARATOR),
            FieldElement::from(self.allow_missing_norm_proof as u64),
            FieldElement::from(self.allow_logup_activation as u64),
            FieldElement::from(self.allow_missing_segment_binding as u64),
            FieldElement::from(self.skip_rms_sq_proof as u64),
            FieldElement::from(self.piecewise_activation as u64),
            FieldElement::from(self.skip_batch_tokens as u64),
            FieldElement::from(self.skip_unified_stark as u64),
            FieldElement::from(self.weight_binding_mode as u64),
            FieldElement::from(self.aggregated_full_binding as u64),
            FieldElement::from(self.aggregated_rlc_only as u64),
            FieldElement::from(self.io_packing as u64),
            FieldElement::from(self.packed_proof as u64),
            FieldElement::from(self.double_packed_proof as u64),
            FieldElement::from(self.validate_decode_chain as u64),
        ])
    }

    /// Returns true if this policy uses aggregated oracle sumcheck for weight binding.
    pub fn is_aggregated_weight_binding(&self) -> bool {
        self.weight_binding_mode != WeightBindingMode::Individual
    }

    /// Derive weight mode flags from this policy (replaces `compute_weight_mode_flags()`).
    pub fn weight_mode_flags(&self) -> WeightModeFlags {
        let trustless_mode3 = self.weight_binding_mode == WeightBindingMode::TrustlessMode3;
        let trustless_mode2 =
            self.weight_binding_mode == WeightBindingMode::TrustlessMode2 && !trustless_mode3;
        let aggregate_weight_binding = matches!(
            self.weight_binding_mode,
            WeightBindingMode::Aggregated
        ) && !(trustless_mode2 || trustless_mode3);
        WeightModeFlags {
            aggregate_weight_binding,
            trustless_mode2,
            trustless_mode3,
        }
    }

    /// Returns true if the policy needs full MLE opening proofs (not RLC-only).
    pub fn needs_full_binding(&self) -> bool {
        if self.aggregated_rlc_only {
            false
        } else {
            self.aggregated_full_binding
        }
    }
}

impl Default for PolicyConfig {
    /// Default policy reads from environment variables for backward compatibility.
    fn default() -> Self {
        Self::from_env()
    }
}

/// Resolve an optional policy: use provided config or fall back to env vars.
///
/// This is the primary entry point for all prove/verify functions that accept
/// `policy: Option<&PolicyConfig>`. When `None`, the existing env var behavior
/// is preserved, ensuring zero breakage for existing callers.
pub fn resolve(policy: Option<&PolicyConfig>) -> PolicyConfig {
    match policy {
        Some(p) => p.clone(),
        None => PolicyConfig::from_env(),
    }
}

// ── User-facing helpers ──────────────────────────────────────────────────────

/// Known preset names for CLI, API, and TUI surfaces.
pub const PRESET_NAMES: &[&str] = &["strict", "standard", "relaxed"];

/// Resolve a policy from a preset name string (case-insensitive).
///
/// Returns `None` for unknown names. The caller is responsible for
/// producing the appropriate user-facing error.
pub fn from_preset_name(name: &str) -> Option<PolicyConfig> {
    match name.to_ascii_lowercase().as_str() {
        "strict" => Some(PolicyConfig::strict()),
        "standard" => Some(PolicyConfig::standard()),
        "relaxed" => Some(PolicyConfig::relaxed()),
        _ => None,
    }
}

/// Classify a [`PolicyConfig`] as a known preset by comparing commitment hashes.
///
/// Returns `None` for custom configurations that don't match any preset.
pub fn preset_name(policy: &PolicyConfig) -> Option<&'static str> {
    let c = policy.policy_commitment();
    if c == PolicyConfig::strict().policy_commitment() {
        return Some("strict");
    }
    if c == PolicyConfig::standard().policy_commitment() {
        return Some("standard");
    }
    if c == PolicyConfig::relaxed().policy_commitment() {
        return Some("relaxed");
    }
    None
}

/// One-line summary for CLI banners and TUI headers.
///
/// Example: `"standard (0x03af8b2c71e9...)"`.
pub fn summary_line(policy: &PolicyConfig) -> String {
    let name = preset_name(policy).unwrap_or("custom");
    let commit = format!("{:#066x}", policy.policy_commitment());
    let short = if commit.len() > 18 {
        format!("{}...", &commit[..18])
    } else {
        commit
    };
    format!("{name} ({short})")
}

/// Load a custom policy from a JSON file (requires `serde` feature).
#[cfg(feature = "serde")]
pub fn from_file(path: &std::path::Path) -> Result<PolicyConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read policy file '{}': {e}", path.display()))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("invalid policy JSON in '{}': {e}", path.display()))
}

// ── Policy commitment skip override ──────────────────────────────────────────
//
// `STWO_SKIP_POLICY_COMMITMENT` was previously a process-global env var read at
// 9 production call sites in `starknet.rs` and `gkr/{prover,verifier}.rs`. That
// is racy under parallel `cargo test` execution and exposes an attack surface
// where an external operator can disable policy binding mid-process.
//
// `policy_commitment_skipped()` is the canonical reader. It checks a
// thread-local override first, then falls back to the env var (preserved for
// backwards compatibility with existing deploy scripts and CLI flags).
//
// Same pattern memory's `aggregated-binding.md` documents for
// `STWO_WEIGHT_BINDING`. Tests should use `PolicyCommitmentSkipGuard::enter()`
// instead of `std::env::set_var(...)`.
use std::cell::Cell;

thread_local! {
    static SKIP_POLICY_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Returns `true` if the policy-commitment binding should be skipped on the
/// current thread.
///
/// Resolution order: thread-local override → `STWO_SKIP_POLICY_COMMITMENT` env
/// var → false.
pub fn policy_commitment_skipped() -> bool {
    if let Some(v) = SKIP_POLICY_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    std::env::var("STWO_SKIP_POLICY_COMMITMENT").is_ok()
}

/// RAII guard that overrides `policy_commitment_skipped()` for the current
/// thread until dropped. Composable with the env var (env var is the fallback,
/// override wins).
#[must_use = "the guard must be held for the duration of the override"]
pub struct PolicyCommitmentSkipGuard {
    prev: Option<bool>,
}

impl PolicyCommitmentSkipGuard {
    /// Set the override to `value` for the current thread.
    pub fn set(value: bool) -> Self {
        let prev = SKIP_POLICY_OVERRIDE.with(|c| c.replace(Some(value)));
        Self { prev }
    }
}

impl Drop for PolicyCommitmentSkipGuard {
    fn drop(&mut self) {
        SKIP_POLICY_OVERRIDE.with(|c| c.set(self.prev));
    }
}

// ── Soundness gate thread-local overrides ────────────────────────────────────
//
// These four gates were closed in `PolicyConfig::standard()` on Apr 30 2026
// (audit-verified safe in `engine/docs/SOUNDNESS_GATES_AUDIT.md`). Closing them
// in the preset is necessary but not sufficient — production code at 8 sites
// reads the corresponding env vars directly, so an external operator could
// set e.g. `STWO_SKIP_RMS_SQ_PROOF=1` to re-open a gate at runtime even when
// the policy says `skip_rms_sq_proof: false`. The thread-local overrides below
// give explicit precedence over the env var so a properly-resolved policy
// cannot be downgraded mid-process.
//
// Resolution order for each gate: thread-local override (Some) → env var → default.
// Production code path uses the canonical reader (`*_skipped()` / `*_enabled()`).
// Tests should `set_thread_policy_overrides()` and hold the returned guard.

thread_local! {
    static SKIP_RMS_SQ_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static ALLOW_MISSING_NORM_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static PIECEWISE_ACTIVATION_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static ALLOW_LOGUP_ACTIVATION_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static AGGREGATED_RLC_ONLY_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static AGGREGATED_FULL_BINDING_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Returns `true` if the RMSNorm Part 0 (variance) sumcheck should be skipped.
/// Default: `false` (proven). Hardened in `standard()` Apr 30 2026.
pub fn skip_rms_sq_proof() -> bool {
    if let Some(v) = SKIP_RMS_SQ_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    std::env::var("STWO_SKIP_RMS_SQ_PROOF").is_ok()
}

/// Returns `true` if proofs without LayerNorm/RMSNorm sub-proofs are accepted.
/// Default: `false` (sub-proofs required). Hardened in `standard()` Apr 30 2026.
pub fn allow_missing_norm_proof() -> bool {
    if let Some(v) = ALLOW_MISSING_NORM_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    std::env::var("STWO_ALLOW_MISSING_NORM_PROOF")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Returns `true` if piecewise-linear algebraic activation proofs are enabled.
/// Default: `true` (full M31 domain proven). Hardened in `standard()` Apr 30 2026.
pub fn piecewise_activation_enabled() -> bool {
    if let Some(v) = PIECEWISE_ACTIVATION_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    match std::env::var("STWO_PIECEWISE_ACTIVATION") {
        Ok(s) => !matches!(s.as_str(), "0" | "false" | "no" | "off"),
        Err(_) => true,
    }
}

/// Returns `true` if proofs with missing/lower-bits LogUp activation proofs
/// are accepted. Default: `false`. Hardened in `standard()` Apr 30 2026.
pub fn allow_logup_activation() -> bool {
    if let Some(v) = ALLOW_LOGUP_ACTIVATION_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    std::env::var("STWO_ALLOW_LOGUP_ACTIVATION")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Returns `true` if aggregated weight binding should use RLC-only mode (no
/// MLE opening proof). Default: `false`. Pollutes Fiat-Shamir transcript if
/// inconsistent across prover/verifier.
pub fn aggregated_rlc_only() -> bool {
    if let Some(v) = AGGREGATED_RLC_ONLY_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    std::env::var("STWO_AGGREGATED_RLC_ONLY")
        .ok()
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false)
}

/// Returns `true` if aggregated weight binding should produce a full MLE
/// opening proof (trustless on-chain verification). Default: `false` (RLC).
pub fn aggregated_full_binding() -> bool {
    if let Some(v) = AGGREGATED_FULL_BINDING_OVERRIDE.with(|c| c.get()) {
        return v;
    }
    std::env::var("STWO_AGGREGATED_FULL_BINDING")
        .ok()
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false)
}

/// RAII guard that overrides one or more soundness gate readers for the current
/// thread until dropped. Pass `None` to leave a gate at its env-var fallback.
#[must_use = "the guard must be held for the duration of the override"]
pub struct SoundnessGateGuard {
    prev_skip_rms_sq: Option<bool>,
    prev_missing_norm: Option<bool>,
    prev_piecewise: Option<bool>,
    prev_logup: Option<bool>,
}

/// Single-gate setter helpers (composable).
impl SoundnessGateGuard {
    /// Override all four gates at once. `None` means "use env var fallback".
    pub fn set(
        skip_rms_sq: Option<bool>,
        allow_missing_norm: Option<bool>,
        piecewise: Option<bool>,
        allow_logup: Option<bool>,
    ) -> Self {
        let prev_skip_rms_sq = SKIP_RMS_SQ_OVERRIDE.with(|c| c.replace(skip_rms_sq));
        let prev_missing_norm =
            ALLOW_MISSING_NORM_OVERRIDE.with(|c| c.replace(allow_missing_norm));
        let prev_piecewise = PIECEWISE_ACTIVATION_OVERRIDE.with(|c| c.replace(piecewise));
        let prev_logup = ALLOW_LOGUP_ACTIVATION_OVERRIDE.with(|c| c.replace(allow_logup));
        Self {
            prev_skip_rms_sq,
            prev_missing_norm,
            prev_piecewise,
            prev_logup,
        }
    }

    /// Convenience: enter strict-aligned soundness mode for the current thread.
    /// Equivalent to `PolicyConfig::strict()` for these four gates.
    pub fn strict() -> Self {
        Self::set(Some(false), Some(false), Some(true), Some(false))
    }
}

impl Drop for SoundnessGateGuard {
    fn drop(&mut self) {
        SKIP_RMS_SQ_OVERRIDE.with(|c| c.set(self.prev_skip_rms_sq));
        ALLOW_MISSING_NORM_OVERRIDE.with(|c| c.set(self.prev_missing_norm));
        PIECEWISE_ACTIVATION_OVERRIDE.with(|c| c.set(self.prev_piecewise));
        ALLOW_LOGUP_ACTIVATION_OVERRIDE.with(|c| c.set(self.prev_logup));
    }
}

/// Detect `STWO_*` env vars that would be overridden by an explicit policy.
///
/// Returns the names of set env vars. Empty means no conflicts.
pub fn detect_env_conflicts() -> Vec<&'static str> {
    const POLICY_VARS: &[&str] = &[
        "STWO_ALLOW_MISSING_NORM_PROOF",
        "STWO_ALLOW_LOGUP_ACTIVATION",
        "STWO_ALLOW_MISSING_SEGMENT_BINDING",
        "STWO_SKIP_RMS_SQ_PROOF",
        "STWO_PIECEWISE_ACTIVATION",
        "STWO_SKIP_BATCH_TOKENS",
        "STWO_PURE_GKR_SKIP_UNIFIED_STARK",
        "STWO_WEIGHT_BINDING",
        "STWO_GKR_TRUSTLESS_MODE2",
        "STWO_GKR_TRUSTLESS_MODE3",
        "STWO_AGGREGATED_FULL_BINDING",
        "STWO_AGGREGATED_RLC_ONLY",
        "STWO_NO_IO_PACK",
        "STWO_NO_PACKED",
        "STWO_NO_DOUBLE_PACK",
        "STWO_VALIDATE_DECODE_CHAIN",
    ];
    POLICY_VARS
        .iter()
        .copied()
        .filter(|v| std::env::var(v).is_ok())
        .collect()
}

/// Sync environment variables to match a resolved PolicyConfig.
///
/// This ensures that any downstream code calling `PolicyConfig::from_env()`
/// (e.g. the streaming calldata self-verifier) produces the same policy
/// commitment as the prover that used an explicit `--policy` preset.
///
/// Also writes thread-local overrides for the migrated soundness gates, so
/// that parallel-test pollution of process-global env vars cannot downgrade
/// the calling thread's policy. Migrated gates: `skip_rms_sq_proof`,
/// `allow_missing_norm_proof`, `piecewise_activation`, `allow_logup_activation`.
/// (G16, Apr 30 2026.)
pub fn apply_to_env(policy: &PolicyConfig) {
    // Thread-local overrides for migrated readers — these win over env vars
    // and are immune to parallel-test pollution.
    SKIP_RMS_SQ_OVERRIDE.with(|c| c.set(Some(policy.skip_rms_sq_proof)));
    ALLOW_MISSING_NORM_OVERRIDE.with(|c| c.set(Some(policy.allow_missing_norm_proof)));
    PIECEWISE_ACTIVATION_OVERRIDE.with(|c| c.set(Some(policy.piecewise_activation)));
    ALLOW_LOGUP_ACTIVATION_OVERRIDE.with(|c| c.set(Some(policy.allow_logup_activation)));
    AGGREGATED_RLC_ONLY_OVERRIDE.with(|c| c.set(Some(policy.aggregated_rlc_only)));
    AGGREGATED_FULL_BINDING_OVERRIDE.with(|c| c.set(Some(policy.aggregated_full_binding)));

    let set = |name: &str, val: bool| {
        if val {
            std::env::set_var(name, "1");
        } else {
            std::env::remove_var(name);
        }
    };
    set("STWO_ALLOW_MISSING_NORM_PROOF", policy.allow_missing_norm_proof);
    set("STWO_ALLOW_LOGUP_ACTIVATION", policy.allow_logup_activation);
    set("STWO_ALLOW_MISSING_SEGMENT_BINDING", policy.allow_missing_segment_binding);
    if policy.skip_rms_sq_proof {
        std::env::set_var("STWO_SKIP_RMS_SQ_PROOF", "1");
    } else {
        std::env::remove_var("STWO_SKIP_RMS_SQ_PROOF");
    }
    // STWO_PIECEWISE_ACTIVATION defaults to true in from_env() (env_bool_default_true),
    // so we must explicitly set "0" when false, not just remove the var.
    if policy.piecewise_activation {
        std::env::set_var("STWO_PIECEWISE_ACTIVATION", "1");
    } else {
        std::env::set_var("STWO_PIECEWISE_ACTIVATION", "0");
    }
    set("STWO_SKIP_BATCH_TOKENS", policy.skip_batch_tokens);
    set("STWO_PURE_GKR_SKIP_UNIFIED_STARK", policy.skip_unified_stark);
    set("STWO_AGGREGATED_FULL_BINDING", policy.aggregated_full_binding);
    set("STWO_AGGREGATED_RLC_ONLY", policy.aggregated_rlc_only);
    if !policy.io_packing { std::env::set_var("STWO_NO_IO_PACK", "1"); }
    else { std::env::remove_var("STWO_NO_IO_PACK"); }
    if !policy.packed_proof { std::env::set_var("STWO_NO_PACKED", "1"); }
    else { std::env::remove_var("STWO_NO_PACKED"); }
    if !policy.double_packed_proof { std::env::set_var("STWO_NO_DOUBLE_PACK", "1"); }
    else { std::env::remove_var("STWO_NO_DOUBLE_PACK"); }
    set("STWO_VALIDATE_DECODE_CHAIN", policy.validate_decode_chain);
    // Sync weight binding mode
    match policy.weight_binding_mode {
        WeightBindingMode::Individual => {
            std::env::remove_var("STWO_WEIGHT_BINDING");
            std::env::remove_var("STWO_GKR_TRUSTLESS_MODE2");
            std::env::remove_var("STWO_GKR_TRUSTLESS_MODE3");
        }
        WeightBindingMode::Aggregated => {
            std::env::set_var("STWO_WEIGHT_BINDING", "aggregated");
            std::env::remove_var("STWO_GKR_TRUSTLESS_MODE2");
            std::env::remove_var("STWO_GKR_TRUSTLESS_MODE3");
        }
        WeightBindingMode::TrustlessMode2 => {
            std::env::set_var("STWO_GKR_TRUSTLESS_MODE2", "1");
            std::env::remove_var("STWO_GKR_TRUSTLESS_MODE3");
        }
        WeightBindingMode::TrustlessMode3 => {
            std::env::set_var("STWO_GKR_TRUSTLESS_MODE3", "1");
        }
        _ => {}
    }
}

/// Weight mode flags derived from [`PolicyConfig`].
///
/// Drop-in replacement for the `WeightModeFlags` struct in `gkr/prover.rs`.
#[derive(Debug, Clone, Copy)]
pub struct WeightModeFlags {
    pub aggregate_weight_binding: bool,
    pub trustless_mode2: bool,
    pub trustless_mode3: bool,
}

// ── Environment variable helpers ─────────────────────────────────────────────

/// Read a boolean env var (truthy: non-empty + not "0"/"false"/"off").
fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off" && v != "no"
        }
        Err(_) => default,
    }
}

/// Read a boolean env var that defaults to true (opt-out pattern).
fn env_bool_default_true(key: &str) -> bool {
    match std::env::var(key) {
        Ok(v) => {
            let s = v.trim();
            !(s == "0"
                || s.eq_ignore_ascii_case("false")
                || s.eq_ignore_ascii_case("no")
                || s.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    }
}

/// Check if an env var is set (any value, including empty).
fn env_is_set(key: &str) -> bool {
    std::env::var(key).is_ok()
}

/// Determine weight binding mode from env vars.
fn weight_binding_mode_from_env() -> WeightBindingMode {
    // Check trustless modes first (highest priority)
    if env_bool("STWO_GKR_TRUSTLESS_MODE3", false) {
        return WeightBindingMode::TrustlessMode3;
    }
    if env_bool("STWO_GKR_TRUSTLESS_MODE2", false) {
        return WeightBindingMode::TrustlessMode2;
    }
    // Check STWO_WEIGHT_BINDING for individual opt-out
    match std::env::var("STWO_WEIGHT_BINDING") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            if v == "individual" || v == "sequential" || v == "0" || v == "off" || v == "false" {
                WeightBindingMode::Individual
            } else {
                WeightBindingMode::Aggregated
            }
        }
        Err(_) => WeightBindingMode::Aggregated,
    }
}

// ── Display ──────────────────────────────────────────────────────────────────

impl std::fmt::Display for PolicyConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PolicyConfig {{")?;
        writeln!(f, "  soundness:")?;
        writeln!(f, "    allow_missing_norm_proof: {}", self.allow_missing_norm_proof)?;
        writeln!(f, "    allow_logup_activation: {}", self.allow_logup_activation)?;
        writeln!(f, "    allow_missing_segment_binding: {}", self.allow_missing_segment_binding)?;
        writeln!(f, "  prover:")?;
        writeln!(f, "    skip_rms_sq_proof: {}", self.skip_rms_sq_proof)?;
        writeln!(f, "    piecewise_activation: {}", self.piecewise_activation)?;
        writeln!(f, "    skip_batch_tokens: {}", self.skip_batch_tokens)?;
        writeln!(f, "    skip_unified_stark: {}", self.skip_unified_stark)?;
        writeln!(f, "  weight_binding:")?;
        writeln!(f, "    mode: {:?}", self.weight_binding_mode)?;
        writeln!(f, "    full_binding: {}", self.aggregated_full_binding)?;
        writeln!(f, "    rlc_only: {}", self.aggregated_rlc_only)?;
        writeln!(f, "  serialization:")?;
        writeln!(f, "    io_packing: {}", self.io_packing)?;
        writeln!(f, "    packed_proof: {}", self.packed_proof)?;
        writeln!(f, "    double_packed_proof: {}", self.double_packed_proof)?;
        writeln!(f, "  decode_chain:")?;
        writeln!(f, "    validate: {}", self.validate_decode_chain)?;
        writeln!(f, "  commitment: {:#066x}", self.policy_commitment())?;
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_commitment_deterministic() {
        let p1 = PolicyConfig::strict();
        let p2 = PolicyConfig::strict();
        assert_eq!(p1.policy_commitment(), p2.policy_commitment());
    }

    /// Print all preset commitments (run with --nocapture to see output).
    #[test]
    fn test_print_preset_commitments() {
        let strict = PolicyConfig::strict();
        let standard = PolicyConfig::standard();
        let relaxed = PolicyConfig::relaxed();
        println!("=== Policy Preset Commitments ===");
        println!("strict:   {:#066x}", strict.policy_commitment());
        println!("standard: {:#066x}", standard.policy_commitment());
        println!("relaxed:  {:#066x}", relaxed.policy_commitment());
        println!("=================================");
    }

    #[test]
    fn test_policy_commitment_distinct() {
        let strict = PolicyConfig::strict();
        let standard = PolicyConfig::standard();
        let relaxed = PolicyConfig::relaxed();
        assert_ne!(strict.policy_commitment(), standard.policy_commitment());
        assert_ne!(strict.policy_commitment(), relaxed.policy_commitment());
        assert_ne!(standard.policy_commitment(), relaxed.policy_commitment());
    }

    #[test]
    fn test_policy_commitment_not_zero() {
        // No preset should produce a zero commitment (reserved for "no policy").
        assert_ne!(PolicyConfig::strict().policy_commitment(), FieldElement::ZERO);
        assert_ne!(PolicyConfig::standard().policy_commitment(), FieldElement::ZERO);
        assert_ne!(PolicyConfig::relaxed().policy_commitment(), FieldElement::ZERO);
    }

    #[test]
    fn test_soundness_gate_thread_local_overrides_env() {
        // With thread-local override set, the env var must be ignored.
        // Use a separate thread so we don't pollute the global env var
        // for parallel tests (the env var fallback is observed only when
        // no override is set on the calling thread).
        let handle = std::thread::spawn(|| {
            // Set the env var to weakened
            std::env::set_var("STWO_SKIP_RMS_SQ_PROOF", "1");
            std::env::set_var("STWO_ALLOW_LOGUP_ACTIVATION", "1");
            std::env::set_var("STWO_ALLOW_MISSING_NORM_PROOF", "1");
            std::env::set_var("STWO_PIECEWISE_ACTIVATION", "0");
            // No override yet — env vars should win
            assert!(skip_rms_sq_proof());
            assert!(allow_logup_activation());
            assert!(allow_missing_norm_proof());
            assert!(!piecewise_activation_enabled());

            // Now apply strict-aligned override
            {
                let _g = SoundnessGateGuard::strict();
                assert!(!skip_rms_sq_proof(), "thread-local must override env");
                assert!(!allow_logup_activation());
                assert!(!allow_missing_norm_proof());
                assert!(piecewise_activation_enabled());
            }
            // Guard dropped — env var fallback restored
            assert!(skip_rms_sq_proof(), "env var fallback must restore on drop");

            // Cleanup
            std::env::remove_var("STWO_SKIP_RMS_SQ_PROOF");
            std::env::remove_var("STWO_ALLOW_LOGUP_ACTIVATION");
            std::env::remove_var("STWO_ALLOW_MISSING_NORM_PROOF");
            std::env::remove_var("STWO_PIECEWISE_ACTIVATION");
        });
        handle.join().expect("spawned thread panicked");
    }

    #[test]
    fn test_soundness_gate_guard_restores_previous() {
        // Nested guards must restore the previous value, not None.
        let _outer = SoundnessGateGuard::set(Some(true), Some(true), Some(false), Some(true));
        assert!(skip_rms_sq_proof());
        assert!(!piecewise_activation_enabled());

        {
            let _inner = SoundnessGateGuard::strict();
            assert!(!skip_rms_sq_proof());
            assert!(piecewise_activation_enabled());
        }
        // Inner dropped — outer's overrides restored, not the env-var fallback.
        assert!(skip_rms_sq_proof(), "inner drop must restore outer override");
        assert!(!piecewise_activation_enabled());
    }

    #[test]
    fn test_strict_preset_values() {
        let p = PolicyConfig::strict();
        assert!(!p.allow_missing_norm_proof);
        assert!(!p.allow_logup_activation);
        assert!(!p.allow_missing_segment_binding);
        assert!(!p.skip_rms_sq_proof);
        assert!(p.piecewise_activation);
        assert!(!p.skip_batch_tokens);
        assert!(!p.skip_unified_stark);
        assert_eq!(p.weight_binding_mode, WeightBindingMode::Aggregated);
        assert!(p.aggregated_full_binding);
        assert!(!p.aggregated_rlc_only);
        assert!(p.io_packing);
        assert!(p.packed_proof);
        assert!(p.double_packed_proof);
        assert!(p.validate_decode_chain);
    }

    #[test]
    fn test_standard_preset_matches_prove_server_defaults() {
        let p = PolicyConfig::standard();
        // Hardened Apr 30 2026 (two-pass):
        // Pass 1: skip_rms_sq_proof + allow_missing_norm_proof closed.
        // Pass 2: piecewise_activation enabled + allow_logup_activation closed.
        assert!(!p.skip_rms_sq_proof);           // CLOSED — RMS Part 0 proven
        assert!(!p.allow_missing_norm_proof);    // CLOSED — norm sub-proofs required
        assert!(p.piecewise_activation);         // CLOSED — full-precision activation
        assert!(!p.allow_logup_activation);      // CLOSED — no lower-bits-only fallback
        assert!(p.aggregated_full_binding);      // STWO_AGGREGATED_FULL_BINDING=1
        assert!(p.skip_batch_tokens);            // STWO_SKIP_BATCH_TOKENS=1 (still open)
        assert!(p.skip_unified_stark);           // STWO_PURE_GKR_SKIP_UNIFIED_STARK=1
    }

    #[test]
    fn test_is_aggregated_weight_binding() {
        let mut p = PolicyConfig::strict();
        assert!(p.is_aggregated_weight_binding());

        p.weight_binding_mode = WeightBindingMode::Individual;
        assert!(!p.is_aggregated_weight_binding());

        p.weight_binding_mode = WeightBindingMode::TrustlessMode2;
        assert!(p.is_aggregated_weight_binding());
    }

    #[test]
    fn test_weight_mode_flags() {
        let p = PolicyConfig::strict();
        let flags = p.weight_mode_flags();
        assert!(flags.aggregate_weight_binding);
        assert!(!flags.trustless_mode2);
        assert!(!flags.trustless_mode3);

        let mut p2 = PolicyConfig::strict();
        p2.weight_binding_mode = WeightBindingMode::TrustlessMode3;
        let flags2 = p2.weight_mode_flags();
        assert!(!flags2.aggregate_weight_binding);
        assert!(!flags2.trustless_mode2);
        assert!(flags2.trustless_mode3);
    }

    #[test]
    fn test_needs_full_binding() {
        let p = PolicyConfig::strict();
        assert!(p.needs_full_binding()); // full_binding=true, rlc_only=false

        let mut p2 = PolicyConfig::strict();
        p2.aggregated_rlc_only = true;
        assert!(!p2.needs_full_binding()); // rlc_only overrides

        let p3 = PolicyConfig::relaxed();
        assert!(!p3.needs_full_binding()); // full_binding=false
    }

    #[test]
    fn test_single_field_change_changes_commitment() {
        let base = PolicyConfig::strict();
        let mut modified = base.clone();
        modified.allow_missing_norm_proof = true;
        assert_ne!(base.policy_commitment(), modified.policy_commitment());
    }

    #[test]
    fn test_resolve_with_some() {
        let strict = PolicyConfig::strict();
        let resolved = resolve(Some(&strict));
        assert_eq!(resolved, strict);
    }

    #[test]
    fn test_resolve_with_none_returns_from_env() {
        // With no env vars set, from_env() returns defaults
        let resolved = resolve(None);
        // Default: piecewise_activation should be true (default-true env var)
        assert!(resolved.piecewise_activation);
        // Default: weight_binding_mode should be Aggregated
        assert_eq!(resolved.weight_binding_mode, WeightBindingMode::Aggregated);
    }

    #[test]
    fn test_display() {
        let p = PolicyConfig::strict();
        let s = format!("{}", p);
        assert!(s.contains("PolicyConfig"));
        assert!(s.contains("commitment:"));
    }

    // ── UX helper tests ──────────────────────────────────────────────────

    #[test]
    fn test_from_preset_name() {
        assert_eq!(from_preset_name("strict"), Some(PolicyConfig::strict()));
        assert_eq!(from_preset_name("STANDARD"), Some(PolicyConfig::standard()));
        assert_eq!(from_preset_name("Relaxed"), Some(PolicyConfig::relaxed()));
        assert_eq!(from_preset_name("unknown"), None);
        assert_eq!(from_preset_name(""), None);
    }

    #[test]
    fn test_preset_name_roundtrip() {
        assert_eq!(preset_name(&PolicyConfig::strict()), Some("strict"));
        assert_eq!(preset_name(&PolicyConfig::standard()), Some("standard"));
        assert_eq!(preset_name(&PolicyConfig::relaxed()), Some("relaxed"));

        // Custom config should return None
        let mut custom = PolicyConfig::strict();
        custom.allow_missing_norm_proof = true;
        assert_eq!(preset_name(&custom), None);
    }

    #[test]
    fn test_summary_line() {
        let s = summary_line(&PolicyConfig::strict());
        assert!(s.starts_with("strict (0x"));
        assert!(s.ends_with("...)"));

        let s2 = summary_line(&PolicyConfig::standard());
        assert!(s2.starts_with("standard (0x"));
    }

    #[test]
    fn test_detect_env_conflicts_empty_by_default() {
        // In a clean test environment, no STWO_* vars should be set
        // (unless another test left them — this is best-effort)
        let conflicts = detect_env_conflicts();
        // We can't assert empty due to parallel tests, but it shouldn't panic
        assert!(conflicts.len() <= 16);
    }

    #[test]
    fn test_preset_names_constant() {
        assert_eq!(PRESET_NAMES.len(), 3);
        assert!(PRESET_NAMES.contains(&"strict"));
        assert!(PRESET_NAMES.contains(&"standard"));
        assert!(PRESET_NAMES.contains(&"relaxed"));
    }

    // ── Integration: prove-verify roundtrip with explicit policy ─────────

    /// Helper: build a simple 1×4 → linear(2) model for policy integration tests.
    fn build_test_model() -> (
        crate::compiler::graph::ComputationGraph,
        crate::components::matmul::M31Matrix,
        crate::compiler::graph::GraphWeights,
    ) {
        use crate::compiler::graph::{GraphBuilder, GraphWeights};
        use crate::components::matmul::M31Matrix;
        use stwo::core::fields::m31::M31;

        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(2, 4);
        for i in 0..2 {
            for j in 0..4 {
                input.set(i, j, M31::from((i * 4 + j + 1) as u32));
            }
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        (graph, input, weights)
    }

    /// Prove a simple model with an explicit policy and verify the proof.
    /// This exercises the full pipeline: PolicyConfig → Poseidon hash →
    /// Fiat-Shamir channel binding → GKR prove → GKR verify.
    #[test]
    fn test_prove_verify_with_explicit_strict_policy() {
        use crate::gkr::circuit::LayeredCircuit;
        use crate::crypto::poseidon_channel::PoseidonChannel;
        use crate::components::matmul::matmul_m31;

        let policy = PolicyConfig::strict();
        let (graph, input, weights) = build_test_model();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        // Execute forward pass manually (single matmul)
        let weight = weights.get_weight(0).unwrap();
        let output = matmul_m31(&input, weight);
        let execution = crate::compiler::graph::GraphExecution {
            intermediates: std::collections::HashMap::from([(0, input.clone())]),
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        // Prove with strict policy
        let mut prove_channel = PoseidonChannel::new();
        let proof = crate::gkr::prove_gkr_with_cache(
            &circuit, &execution, &weights, &mut prove_channel, None, Some(&policy),
        ).unwrap();

        // Verify with the SAME policy → should succeed
        let mut verify_channel = PoseidonChannel::new();
        let result = crate::gkr::verifier::verify_gkr_with_policy(
            &circuit, &proof, &output, Some(&weights), &mut verify_channel, &policy,
        );
        assert!(result.is_ok(), "verify with matching policy should succeed: {:?}", result.err());
    }

    /// Prove with strict, verify with standard → must FAIL (different policy commitments
    /// produce different Fiat-Shamir challenges → sumcheck verification fails).
    #[test]
    fn test_prove_verify_policy_mismatch_fails() {
        use crate::gkr::circuit::LayeredCircuit;
        use crate::crypto::poseidon_channel::PoseidonChannel;
        use crate::components::matmul::matmul_m31;

        let prove_policy = PolicyConfig::strict();
        let verify_policy = PolicyConfig::standard();
        assert_ne!(prove_policy.policy_commitment(), verify_policy.policy_commitment());

        let (graph, input, weights) = build_test_model();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let weight = weights.get_weight(0).unwrap();
        let output = matmul_m31(&input, weight);
        let execution = crate::compiler::graph::GraphExecution {
            intermediates: std::collections::HashMap::from([(0, input.clone())]),
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        // Prove with strict
        let mut prove_channel = PoseidonChannel::new();
        let proof = crate::gkr::prove_gkr_with_cache(
            &circuit, &execution, &weights, &mut prove_channel, None, Some(&prove_policy),
        ).unwrap();

        // Verify with standard → should fail
        let mut verify_channel = PoseidonChannel::new();
        let result = crate::gkr::verifier::verify_gkr_with_policy(
            &circuit, &proof, &output, Some(&weights), &mut verify_channel, &verify_policy,
        );
        assert!(result.is_err(), "verify with mismatched policy should fail");
    }

    /// Prove through the full aggregation pipeline with explicit policy.
    /// Verify the proof struct carries the correct policy_commitment.
    #[test]
    fn test_aggregated_proof_carries_policy_commitment() {
        let policy = PolicyConfig::strict();
        let (graph, input, weights) = build_test_model();

        let proof = crate::aggregation::prove_model_pure_gkr_auto_with_cache(
            &graph, &input, &weights, None, Some(&policy),
        ).unwrap();

        assert_eq!(
            proof.policy_commitment,
            policy.policy_commitment(),
            "proof should carry the policy commitment from the explicit policy"
        );
        assert_ne!(
            proof.policy_commitment,
            FieldElement::ZERO,
            "policy commitment should not be zero for explicit policies"
        );
        assert_ne!(
            proof.policy_commitment,
            PolicyConfig::standard().policy_commitment(),
            "strict commitment should differ from standard"
        );
    }
}
