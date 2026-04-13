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
