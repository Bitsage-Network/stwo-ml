//! Cryptographic primitives for on-chain proof verification.
//!
//! Provides Poseidon-based hashing, Merkle trees, MLE commitments,
//! and a Fiat-Shamir channel matching the Cairo verifier's implementation.

pub mod aggregated_opening;
pub mod commitment;
pub mod encryption;
pub mod hades;
pub mod merkle_cache;
pub mod merkle_m31;
pub mod mle_opening;
pub mod poseidon2_m31;
pub mod poseidon_channel;
pub mod poseidon_merkle;
