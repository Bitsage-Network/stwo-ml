//! # Recursive STARK Composition (Phase 4A)
//!
//! Compresses the GKR proof (~112K felts, 18 TXs) into a constant-size STARK
//! proof (~500 felts, 1 TX) by proving "I verified the GKR proof and it passed."
//!
//! ## Architecture
//!
//! ```text
//! prove_model()          →  GKR Proof (112K felts)
//! generate_witness()     →  Verifier execution trace (every Poseidon call + QM31 op)
//! prove_recursive()      →  STARK proof (~500 felts)
//! verify_recursive()     →  accept/reject (Rust pre-flight)
//! On-chain (1 TX)        →  stwo-cairo-verifier verifies the STARK
//! ```
//!
//! ## Key Design Principle
//!
//! The GKR verifier is refactored into a generic function parameterized over the
//! channel type. Both production verification (`PoseidonChannel`) and witness
//! generation (`InstrumentedChannel`) call the same code path, guaranteeing
//! Fiat-Shamir transcript consistency.
//!
//! ## See Also
//!
//! - `docs/RECURSIVE_STARK.md` — full design document
//! - `src/gkr/verifier.rs` — the verifier being arithmetized

pub mod air;
pub mod hades_air;
pub mod prover;
#[cfg(all(test, feature = "red-team-recursive-forgery"))]
mod red_team;
pub mod types;
pub mod verifier;
pub mod witness;

#[cfg(test)]
mod tests;

pub use air::{
    build_recursive_trace, RecursiveTraceData, RecursiveVerifierComponent, RecursiveVerifierEval,
    COLS_PER_ROW,
};
pub use prover::{
    export_hades_pairs_cairo_args, prove_recursive, prove_recursive_with_policy,
    verify_hades_perms_offline, RecursiveError,
};
pub use types::{GkrVerifierWitness, RecursiveProof, RecursivePublicInputs, WitnessOp};
pub use verifier::verify_recursive;
pub use witness::{generate_witness, InstrumentedChannel};

// ── Test-only PcsConfig downgrade ────────────────────────────────────────────
//
// Production builds never use these (the `#[cfg(test)]`-gated reader paths
// in `prover.rs` and `verifier.rs` compile out entirely in release).
//
// Tests previously called `std::env::set_var("OBELYZK_RECURSIVE_SECURITY",
// "test")` which is process-global and stomps across `cargo test`'s parallel
// threads — that's the proximate cause of the flaky `test_verify_recursive_
// roundtrip` failure observed in mixed runs (passes in isolation, fails in
// the full suite). Switch to a thread-local override that's invisible to
// other threads.
//
// Same pattern memory's `aggregated-binding.md` documents for
// `STWO_WEIGHT_BINDING`. Production code reads no env var, so this cannot
// affect deployment.
#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    /// Thread-local override for recursive STARK security level.
    /// `Some(true)` = use 13-bit `PcsConfig::default()` for fast tests.
    /// `None` = use production 120-bit config.
    pub(crate) static RECURSIVE_TEST_MODE: Cell<bool> = Cell::new(false);
}

/// Activate the low-security PcsConfig path for the current thread until
/// the returned guard is dropped. **Test-only.**
#[cfg(test)]
#[must_use = "the guard must be held for the duration of the test"]
pub(crate) struct RecursiveTestModeGuard {
    prev: bool,
}

#[cfg(test)]
impl RecursiveTestModeGuard {
    pub(crate) fn enter() -> Self {
        let prev = RECURSIVE_TEST_MODE.with(|c| c.replace(true));
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for RecursiveTestModeGuard {
    fn drop(&mut self) {
        RECURSIVE_TEST_MODE.with(|c| c.set(self.prev));
    }
}

#[cfg(test)]
pub(crate) fn recursive_test_mode_active() -> bool {
    RECURSIVE_TEST_MODE.with(|c| c.get())
}
