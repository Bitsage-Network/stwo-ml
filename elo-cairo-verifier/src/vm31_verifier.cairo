// VM31 Batch STARK Verifier Adapter
//
// Wraps the batch privacy transaction STARK proof with IO binding.
// The Rust prover (stwo-ml/src/circuits/batch.rs) generates a single STARK proof
// that covers N deposits + M withdrawals + K spends. This module:
//
//   1. Deserializes the batch public inputs from calldata
//   2. Hashes all public inputs using Poseidon2-M31 (matching Rust's hash_batch_public_inputs)
//   3. Mixes the hash into the Fiat-Shamir channel (binding public inputs to the proof)
//   4. Delegates STARK verification to the existing verifier infrastructure
//   5. Returns the verified public inputs for the pool contract to process
//
// Architecture: submit_and_verify_with_io_binding (direct verification, no chunking)
// The STARK proof itself is uploaded via chunk mechanism if too large for one TX.
// Public input verification is O(N) in the number of transactions — always one TX.

use crate::vm31_merkle::{PackedDigest, poseidon2_m31_hash_packed};

// ============================================================================
// Batch Public Input Types
// ============================================================================

// Public inputs for a single deposit transaction
#[derive(Drop, Copy, Serde)]
pub struct DepositPublicInput {
    pub commitment: PackedDigest,  // note commitment (8 M31)
    pub amount_lo: u64,            // amount low limb (M31)
    pub amount_hi: u64,            // amount high limb (M31)
    pub asset_id: u64,             // asset identifier (M31)
}

// Public inputs for a single withdrawal transaction
#[derive(Drop, Copy, Serde)]
pub struct WithdrawPublicInput {
    pub merkle_root: PackedDigest, // Merkle tree root at time of withdraw
    pub nullifier: PackedDigest,   // spend nullifier (8 M31)
    pub amount_lo: u64,
    pub amount_hi: u64,
    pub asset_id: u64,
    // Bridge/app memo hash committed in proof public inputs.
    pub withdrawal_binding: PackedDigest,
}

// Public inputs for a single spend (private transfer) transaction
#[derive(Drop, Copy, Serde)]
pub struct SpendPublicInput {
    pub merkle_root: PackedDigest,        // Merkle root for input notes
    pub nullifier_0: PackedDigest,        // nullifier for input note 0
    pub nullifier_1: PackedDigest,        // nullifier for input note 1
    pub output_commitment_0: PackedDigest, // output note 0 commitment
    pub output_commitment_1: PackedDigest, // output note 1 commitment
}

// All public inputs for a batch
#[derive(Drop, Serde)]
pub struct BatchPublicInputs {
    pub deposits: Array<DepositPublicInput>,
    pub withdrawals: Array<WithdrawPublicInput>,
    pub spends: Array<SpendPublicInput>,
}

// Result of batch verification — the verified, binding public inputs
#[derive(Drop)]
pub struct VerifiedBatch {
    pub deposits: Array<DepositPublicInput>,
    pub withdrawals: Array<WithdrawPublicInput>,
    pub spends: Array<SpendPublicInput>,
    pub batch_hash: PackedDigest,  // Poseidon2-M31 hash of all public inputs
}

// ============================================================================
// Batch Public Input Hashing
// ============================================================================

// Hash all batch public inputs using Poseidon2-M31.
// Must match Rust's hash_batch_public_inputs() exactly for Fiat-Shamir binding.
//
// Layout:
//   [n_deposits, for each deposit: commitment[8], amount_lo, amount_hi, asset_id,
//    n_withdrawals, for each withdraw: merkle_root[8], nullifier[8], amount_lo, amount_hi, asset_id, withdrawal_binding[8],
//    n_spends, for each spend: merkle_root[8], nullifier_0[8], nullifier_1[8], out_commit_0[8], out_commit_1[8]]
pub fn hash_batch_public_inputs(inputs: @BatchPublicInputs) -> PackedDigest {
    let mut data: Array<u64> = array![];

    // Deposits
    let n_dep: u64 = inputs.deposits.len().into();
    data.append(n_dep);
    let mut i: u32 = 0;
    loop {
        if i >= inputs.deposits.len() {
            break;
        }
        let dep = inputs.deposits.at(i);
        // commitment: unpack 8 M31 values
        append_packed_digest(ref data, dep.commitment);
        data.append(*dep.amount_lo);
        data.append(*dep.amount_hi);
        data.append(*dep.asset_id);
        i += 1;
    };

    // Withdrawals
    let n_wit: u64 = inputs.withdrawals.len().into();
    data.append(n_wit);
    let mut i: u32 = 0;
    loop {
        if i >= inputs.withdrawals.len() {
            break;
        }
        let wit = inputs.withdrawals.at(i);
        append_packed_digest(ref data, wit.merkle_root);
        append_packed_digest(ref data, wit.nullifier);
        data.append(*wit.amount_lo);
        data.append(*wit.amount_hi);
        data.append(*wit.asset_id);
        append_packed_digest(ref data, wit.withdrawal_binding);
        i += 1;
    };

    // Spends
    let n_spe: u64 = inputs.spends.len().into();
    data.append(n_spe);
    let mut i: u32 = 0;
    loop {
        if i >= inputs.spends.len() {
            break;
        }
        let spe = inputs.spends.at(i);
        append_packed_digest(ref data, spe.merkle_root);
        append_packed_digest(ref data, spe.nullifier_0);
        append_packed_digest(ref data, spe.nullifier_1);
        append_packed_digest(ref data, spe.output_commitment_0);
        append_packed_digest(ref data, spe.output_commitment_1);
        i += 1;
    };

    poseidon2_m31_hash_packed(data.span())
}

// Helper: append 8 unpacked M31 values from a PackedDigest to a data array
fn append_packed_digest(ref data: Array<u64>, digest: @PackedDigest) {
    let vals = crate::vm31_merkle::unpack_m31x8(*digest);
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        data.append(*vals.at(i));
        i += 1;
    };
}

// ============================================================================
// Fiat-Shamir Channel Binding
// ============================================================================

// Mix the batch public input hash into a Poseidon252 Fiat-Shamir channel.
// This binds the public inputs to the STARK proof transcript:
//   channel_mix_felt(packed_hash.lo)
//   channel_mix_felt(packed_hash.hi)
//
// Both prover and verifier must call this before any tree commitments.
// Any mismatch causes channel divergence → verification failure.
pub fn mix_batch_hash_into_channel(
    ref ch: crate::channel::PoseidonChannel,
    batch_hash: PackedDigest,
) {
    crate::channel::channel_mix_felt(ref ch, batch_hash.lo);
    crate::channel::channel_mix_felt(ref ch, batch_hash.hi);
}

// ============================================================================
// Verification Entry Point
// ============================================================================

// Verify a batch proof and return the verified public inputs.
//
// This is the direct IO-binding verification path:
//   1. Hash public inputs using Poseidon2-M31
//   2. The STARK proof has already been committed to the same hash
//   3. If the STARK verifies, the public inputs are sound
//
// The actual STARK verification is delegated to the main verifier contract
// (which handles FRI, commitment scheme, etc.). This function handles
// only the public input binding layer.
//
// Returns: VerifiedBatch if verification succeeds
// Panics: if public input hash doesn't match the proof's committed hash
pub fn verify_batch_public_inputs(
    inputs: @BatchPublicInputs,
    committed_batch_hash: PackedDigest,
) -> PackedDigest {
    let computed_hash = hash_batch_public_inputs(inputs);
    assert!(
        computed_hash == committed_batch_hash,
        "VM31: batch public input hash mismatch"
    );
    computed_hash
}

// ============================================================================
// Deposit Amount Reconstruction
// ============================================================================

const M31_P: u64 = 0x7FFFFFFF;

// Reconstruct a u64 amount from M31 limbs: amount = lo + hi * 2^31
// Safe for amounts up to 2^62 - 1 (sufficient for token amounts)
pub fn reconstruct_amount(amount_lo: u64, amount_hi: u64) -> u64 {
    assert!(amount_lo < M31_P, "VM31: amount_lo out of M31 range");
    assert!(amount_hi < M31_P, "VM31: amount_hi out of M31 range");
    amount_lo + amount_hi * 0x80000000 // 2^31
}

// ============================================================================
// Batch Statistics
// ============================================================================

// Count total transactions in a batch
pub fn batch_tx_count(inputs: @BatchPublicInputs) -> u32 {
    inputs.deposits.len() + inputs.withdrawals.len() + inputs.spends.len()
}
