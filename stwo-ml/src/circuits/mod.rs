//! Provable circuit implementations for cryptographic primitives.
//!
//! Standalone proving modules for operations needed by the VM31 privacy protocol.
//! Each circuit generates execution traces, produces sumcheck-based proofs, and
//! provides verification using Fiat-Shamir transcript replay.
//!
//! ## Modules
//!
//! - [`poseidon_circuit`] — Poseidon2-M31 permutation proving (batch of N permutations).
//! - [`helpers`] — Permutation-recording helpers for composed circuit proofs.
//! - [`deposit`] — Deposit circuit (public amount → shielded note).
//! - [`withdraw`] — Withdraw circuit (shielded note → public amount).
//! - [`spend`] — 2-in/2-out spend circuit (private transfer).

pub mod poseidon_circuit;
pub mod helpers;
pub mod deposit;
pub mod withdraw;
pub mod spend;
pub mod stark_deposit;
pub mod stark_withdraw;
pub mod stark_spend;
pub mod batch;
