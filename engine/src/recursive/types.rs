//! Recursive STARK proof types.
//!
//! These types represent the recursive composition output — a STARK proof
//! attesting that the GKR verifier accepted the original proof.

use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::components::matmul::RoundPoly;
use crate::gkr::types::{RoundPolyDeg3, SecureField};

/// A single recorded operation from the GKR verifier execution.
///
/// The witness generator records one of these for every channel operation
/// and QM31 arithmetic operation that the verifier performs. These become
/// the execution trace rows in the recursive STARK.
#[derive(Debug, Clone)]
pub enum WitnessOp {
    /// Hades permutation over felt252: `state = [digest, value, capacity]; hades(&state);`
    /// Records the full 3-element input and output state.
    /// This is Starknet's Poseidon (felt252 Hades), NOT the M31 Poseidon2.
    HadesPerm {
        input: [FieldElement; 3],
        output: [FieldElement; 3],
    },

    /// A channel operation that transforms digest_before → digest_after.
    /// This is the unit of chaining — one per mix/draw/mix_poly_coeffs call.
    /// May internally involve multiple Hades permutations (e.g., poseidon_hash_many).
    ChannelOp {
        digest_before: FieldElement,
        digest_after: FieldElement,
    },

    /// Sumcheck round verification (degree-2, MatMul layers).
    /// Verifies: `c0 + (c0 + c1 + c2) == claim`, then `next = c0 + c1*r + c2*r^2`.
    SumcheckRoundDeg2 {
        round_poly: RoundPoly,
        claim: SecureField,
        challenge: SecureField,
        next_claim: SecureField,
    },

    /// Sumcheck round verification (degree-3, Mul/Activation/Norm layers).
    /// Verifies: `c0 + (c0+c1+c2+c3) == claim`, then Horner eval at challenge.
    SumcheckRoundDeg3 {
        round_poly: RoundPolyDeg3,
        claim: SecureField,
        challenge: SecureField,
        next_claim: SecureField,
    },

    /// QM31 multiplication: `result = a * b`.
    QM31Mul {
        a: SecureField,
        b: SecureField,
        result: SecureField,
    },

    /// QM31 addition: `result = a + b`.
    QM31Add {
        a: SecureField,
        b: SecureField,
        result: SecureField,
    },

    /// Equality assertion: `lhs == rhs`.
    /// The AIR constrains `lhs - rhs = 0`.
    EqualityCheck { lhs: SecureField, rhs: SecureField },

    /// Mix a SecureField into the Fiat-Shamir channel.
    /// This is a Hades permutation with specific packing — recorded for
    /// consistency but implemented via HadesPerm in the AIR.
    ChannelMix { value: SecureField },

    /// Draw a SecureField from the Fiat-Shamir channel.
    ChannelDraw { result: SecureField },
}

/// The complete witness of a GKR verifier execution.
///
/// Contains every operation the verifier performed, in order.
/// This becomes the execution trace for the recursive STARK.
#[derive(Debug, Clone)]
pub struct GkrVerifierWitness {
    /// All operations, in execution order.
    pub ops: Vec<WitnessOp>,

    /// Public inputs — committed in the recursive STARK and checked on-chain.
    pub public_inputs: RecursivePublicInputs,

    /// Total Poseidon2 permutations (for trace sizing).
    pub n_poseidon_perms: usize,

    /// Total sumcheck rounds verified.
    pub n_sumcheck_rounds: usize,

    /// Total QM31 arithmetic operations.
    pub n_qm31_ops: usize,

    /// Final channel digest after production verification (felt252).
    /// Used as boundary constraint in the recursive STARK.
    pub final_digest: FieldElement,

    /// Total equality checks.
    pub n_equality_checks: usize,
}

/// Public inputs for the recursive STARK proof.
///
/// These values are committed inside the recursive circuit and verified
/// on-chain against the registered model. They are the only data the
/// on-chain verifier needs — the full GKR proof is never sent on-chain.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecursivePublicInputs {
    /// Poseidon hash of the LayeredCircuit descriptor.
    /// Binds the proof to a specific model architecture (layer types, shapes).
    pub circuit_hash: QM31,

    /// Poseidon hash of the packed IO felts (input + output commitment).
    /// Binds the proof to a specific inference (what went in, what came out).
    pub io_commitment: QM31,

    /// Poseidon Merkle root of all weight matrices.
    /// Binds the proof to specific model weights (prevents weight substitution).
    pub weight_super_root: QM31,

    /// Total number of GKR layers verified (for the AIR log_size).
    pub n_layers: u32,

    /// Total Poseidon permutations in the verifier execution.
    /// SECURITY: This is validated on-chain against the registered model's
    /// expected complexity. Without this, an attacker could submit a trivially
    /// small trace (2 Hades permutations) that satisfies all chain AIR constraints
    /// without ever running the GKR verifier. The AIR uses this to reconstruct
    /// the preprocessed columns (is_first, is_last, is_chain) deterministically.
    pub n_poseidon_perms: u32,

    /// Poseidon channel digest after the 3 circuit-seeding operations:
    ///   mix_u64(n_layers), mix_u64(input_shape.0), mix_u64(input_shape.1)
    ///
    /// SECURITY: This checkpoint constrains the chain's early rows to match
    /// the GKR verifier's deterministic seeding. An attacker fabricating a
    /// chain cannot pass this constraint without knowing the model dimensions
    /// and producing the exact same Hades permutations the real verifier would.
    /// This value is deterministic per model (doesn't depend on inference data).
    pub seed_digest: QM31,
}

/// The recursive STARK proof — replaces the 112K felt GKR calldata.
///
/// Contains a standard STWO STARK proof plus the public inputs.
/// On-chain verification uses stwo-cairo-verifier's generic `verify()`.
#[derive(Debug, Clone)]
pub struct RecursiveProof {
    /// The actual STARK proof (Poseidon252MerkleHasher).
    /// Used for Rust-side verification via `verify_recursive()`.
    pub stark_proof: stwo::core::proof::StarkProof<
        stwo::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleHasher,
    >,

    /// Public inputs committed in the circuit.
    pub public_inputs: RecursivePublicInputs,

    /// Full felt252 IO commitment (Poseidon hash of packed IO).
    pub io_commitment_felt252: FieldElement,

    /// Pass 1 (full GKR verification) final channel digest.
    /// Mixed into the Fiat-Shamir channel to bind the STARK proof to a
    /// COMPLETE GKR verification. Without this, an attacker could fabricate
    /// a partial witness that passes the chain AIR without running the full verifier.
    pub pass1_final_digest: FieldElement,

    /// Pass 2 (instrumented replay) final digest — used for chain AIR boundary.
    pub final_digest: FieldElement,

    /// LogUp claimed sum for the chain component (0 if LogUp disabled).
    pub logup_claimed_sum: QM31,

    /// Number of real (active) rows in the chain trace.
    /// This is the ChannelOp count from Pass 2, NOT n_poseidon_perms (which
    /// counts ALL Poseidon calls including non-ChannelOp ones from Pass 1).
    /// The AIR's amortized accumulator uses this for the correction term.
    pub n_real_rows: u32,

    /// Trace log_size (needed for verifier to reconstruct the AIR).
    pub log_size: u32,

    /// Hades permutation pairs for two-level recursion.
    /// Each pair is (input[3], output[3]) verified by the prover offline.
    /// Used to generate the Level 1 Hades recursive proof via cairo-prove.
    pub hades_pairs: Vec<([starknet_ff::FieldElement; 3], [starknet_ff::FieldElement; 3])>,

    /// Proof metadata for debugging/display.
    pub metadata: RecursiveProofMetadata,
}

/// Metadata about the recursive proof (not security-critical).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RecursiveProofMetadata {
    /// Time to generate the recursive proof (seconds).
    pub recursive_prove_time_secs: f64,

    /// Time to generate the original GKR proof (seconds).
    pub gkr_prove_time_secs: f64,

    /// Number of Poseidon2 permutations in the verifier trace.
    pub n_poseidon_perms: usize,

    /// Number of sumcheck rounds verified.
    pub n_sumcheck_rounds: usize,

    /// Trace log_size (log2 of number of rows).
    pub trace_log_size: u32,

    /// Number of trace columns.
    pub n_trace_columns: usize,
}
