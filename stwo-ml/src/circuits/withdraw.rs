//! Withdraw circuit: consumes one shielded note and reveals the amount publicly.
//!
//! Like half a spend: one input note consumed (ownership + commitment + nullifier + Merkle),
//! amount exits the shielded pool publicly.
//!
//! Permutations: 25 (1 ownership + 2 commitment + 2 nullifier + 20 Merkle) → padded to 32.
//!
//! Composes `Poseidon2BatchProof` + `RangeCheckProof` into a single `WithdrawProof`.

use stwo::core::fields::m31::BaseField as M31;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::channel::MerkleChannel;

use crate::crypto::commitment::{Note, NoteCommitment, Nullifier, SpendingKey, PublicKey};
use crate::crypto::poseidon2_m31::{poseidon2_permutation, STATE_WIDTH, RATE};
use crate::crypto::merkle_m31::{Digest, MerklePath};
use crate::crypto::poseidon_channel::PoseidonChannel;
use crate::circuits::poseidon_circuit::{prove_poseidon2_batch, verify_poseidon2_batch, Poseidon2BatchProof};
use crate::circuits::helpers::{
    record_hash_permutations, record_merkle_permutations, record_ownership_permutations,
    verify_sponge_chain,
};
use crate::components::range_check::{prove_range_check, verify_range_check, RangeCheckProof};
use crate::gadgets::range_check::RangeCheckConfig;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

pub const MERKLE_DEPTH: usize = 20;
pub const INPUT_PERMS: usize = 25; // 1 ownership + 2 commitment + 2 nullifier + 20 Merkle
pub const BATCH_SIZE: usize = 32;  // next power of 2

#[derive(Debug, thiserror::Error)]
pub enum WithdrawError {
    #[error("ownership verification failed: derived pk {derived:?} != note pk {expected:?}")]
    OwnershipFailed { derived: PublicKey, expected: PublicKey },
    #[error("Merkle inclusion failed: computed root {computed:?} != expected {expected:?}")]
    MerkleInclusionFailed { computed: Digest, expected: Digest },
    #[error("commitment wiring failed")]
    CommitmentWiringFailed,
    #[error("nullifier wiring failed")]
    NullifierWiringFailed,
    #[error("range check decomposition failed: {0}")]
    RangeDecomposition(String),
    #[error("poseidon proof error: {0}")]
    PoseidonError(String),
    #[error("range check error: {0}")]
    RangeCheckError(String),
    #[error("permutation count mismatch: expected {expected}, got {actual}")]
    PermutationCountMismatch { expected: usize, actual: usize },
    #[error("public input mismatch: {0}")]
    PublicInputMismatch(String),
    #[error("cross-wiring verification failed: {0}")]
    CrossWiringFailed(String),
}

/// Witness for a withdraw transaction.
#[derive(Clone, Debug)]
pub struct WithdrawWitness {
    pub note: Note,
    pub spending_key: SpendingKey,
    pub merkle_path: MerklePath,
    pub merkle_root: Digest,
    // Optional app-level binding (e.g., bridge recipient memo hash).
    // This value is not used in arithmetic constraints, but is committed in
    // batch public-input hashing so relayers cannot rewrite it post-proving.
    pub withdrawal_binding: Digest,
}

/// Public inputs for a withdraw transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WithdrawPublicInputs {
    pub merkle_root: Digest,
    pub nullifier: Nullifier,
    pub amount_lo: M31,
    pub amount_hi: M31,
    pub asset_id: M31,
    pub withdrawal_binding: Digest,
}

/// Permutation index ranges for wiring verification.
#[derive(Clone, Debug)]
pub struct WithdrawPermRanges {
    pub ownership: (usize, usize),   // 1 perm
    pub commitment: (usize, usize),  // 2 perms
    pub nullifier: (usize, usize),   // 2 perms
    pub merkle: (usize, usize),      // 20 perms
}

/// Execution trace for the withdraw circuit.
#[derive(Clone, Debug)]
pub struct WithdrawExecution {
    pub commitment: NoteCommitment,
    pub nullifier: Nullifier,
    pub derived_pubkey: PublicKey,
    pub computed_merkle_root: Digest,
    pub all_permutation_inputs: Vec<[M31; STATE_WIDTH]>,
    pub all_permutation_outputs: Vec<[M31; STATE_WIDTH]>,
    pub perm_ranges: WithdrawPermRanges,
    pub range_check_limbs: Vec<M31>,
}

/// Complete withdraw proof.
#[derive(Debug)]
pub struct WithdrawProof {
    pub poseidon_proof: Poseidon2BatchProof,
    pub range_check_proof: RangeCheckProof<Blake2sHash>,
    pub execution: WithdrawExecution,
    pub public_inputs: WithdrawPublicInputs,
}

/// Execute the withdraw circuit.
pub fn execute_withdraw(
    witness: &WithdrawWitness,
) -> Result<(WithdrawExecution, WithdrawPublicInputs), WithdrawError> {
    let mut all_inputs: Vec<[M31; STATE_WIDTH]> = Vec::with_capacity(BATCH_SIZE);
    let mut all_outputs: Vec<[M31; STATE_WIDTH]> = Vec::with_capacity(BATCH_SIZE);
    let mut offset = 0;

    // 1. Ownership: derive pubkey from spending key
    let (derived_pk, ownership_perms) = record_ownership_permutations(&witness.spending_key);
    if derived_pk != witness.note.owner_pubkey {
        return Err(WithdrawError::OwnershipFailed {
            derived: derived_pk,
            expected: witness.note.owner_pubkey,
        });
    }
    let ownership_start = offset;
    for (inp, out) in &ownership_perms {
        all_inputs.push(*inp);
        all_outputs.push(*out);
        offset += 1;
    }
    let ownership_end = offset;

    // 2. Commitment hash (11 M31 → 2 perms)
    let commitment_input = [
        witness.note.owner_pubkey[0],
        witness.note.owner_pubkey[1],
        witness.note.owner_pubkey[2],
        witness.note.owner_pubkey[3],
        witness.note.asset_id,
        witness.note.amount_lo,
        witness.note.amount_hi,
        witness.note.blinding[0],
        witness.note.blinding[1],
        witness.note.blinding[2],
        witness.note.blinding[3],
    ];
    let (commitment, commitment_perms) = record_hash_permutations(&commitment_input);
    let commitment_start = offset;
    for (inp, out) in &commitment_perms {
        all_inputs.push(*inp);
        all_outputs.push(*out);
        offset += 1;
    }
    let commitment_end = offset;

    // 3. Nullifier hash (12 M31 → 2 perms)
    let mut nullifier_input = [M31::from_u32_unchecked(0); 12];
    nullifier_input[..4].copy_from_slice(&witness.spending_key);
    nullifier_input[4..12].copy_from_slice(&commitment);
    let (nullifier, nullifier_perms) = record_hash_permutations(&nullifier_input);
    let nullifier_start = offset;
    for (inp, out) in &nullifier_perms {
        all_inputs.push(*inp);
        all_outputs.push(*out);
        offset += 1;
    }
    let nullifier_end = offset;

    // 4. Merkle path verification (20 compress perms)
    let (computed_root, merkle_perms) = record_merkle_permutations(&commitment, &witness.merkle_path);
    if computed_root != witness.merkle_root {
        return Err(WithdrawError::MerkleInclusionFailed {
            computed: computed_root,
            expected: witness.merkle_root,
        });
    }
    let merkle_start = offset;
    for (inp, out) in &merkle_perms {
        all_inputs.push(*inp);
        all_outputs.push(*out);
        offset += 1;
    }
    let merkle_end = offset;

    assert_eq!(offset, INPUT_PERMS, "expected {INPUT_PERMS} perms, got {offset}");

    // 5. Pad to BATCH_SIZE with identity permutations
    pad_permutations(&mut all_inputs, &mut all_outputs, BATCH_SIZE);

    // 6. Decompose amount limbs for range checking
    let range_check_limbs = decompose_amount_limbs(
        witness.note.amount_lo,
        witness.note.amount_hi,
    )?;

    let execution = WithdrawExecution {
        commitment,
        nullifier,
        derived_pubkey: derived_pk,
        computed_merkle_root: computed_root,
        all_permutation_inputs: all_inputs,
        all_permutation_outputs: all_outputs,
        perm_ranges: WithdrawPermRanges {
            ownership: (ownership_start, ownership_end),
            commitment: (commitment_start, commitment_end),
            nullifier: (nullifier_start, nullifier_end),
            merkle: (merkle_start, merkle_end),
        },
        range_check_limbs,
    };

    let public_inputs = WithdrawPublicInputs {
        merkle_root: witness.merkle_root,
        nullifier,
        amount_lo: witness.note.amount_lo,
        amount_hi: witness.note.amount_hi,
        asset_id: witness.note.asset_id,
        withdrawal_binding: witness.withdrawal_binding,
    };

    Ok((execution, public_inputs))
}

/// Prove a withdraw transaction.
pub fn prove_withdraw(witness: &WithdrawWitness) -> Result<WithdrawProof, WithdrawError> {
    let (execution, public_inputs) = execute_withdraw(witness)?;

    // Prove Poseidon2 batch
    let mut channel = PoseidonChannel::default();
    let poseidon_proof = prove_poseidon2_batch(
        &execution.all_permutation_inputs,
        &mut channel,
    );

    // Prove range check
    let config = RangeCheckConfig::uint16();
    let range_check_proof = prove_range_check(&execution.range_check_limbs, &config)
        .map_err(|e| WithdrawError::RangeCheckError(format!("{e}")))?;

    Ok(WithdrawProof {
        poseidon_proof,
        range_check_proof,
        execution,
        public_inputs,
    })
}

/// Verify a withdraw proof.
pub fn verify_withdraw(proof: &WithdrawProof) -> Result<(), WithdrawError> {
    let exec = &proof.execution;
    let pub_in = &proof.public_inputs;

    // 1. Verify Poseidon2 batch proof
    let mut channel = PoseidonChannel::default();
    verify_poseidon2_batch(
        &proof.poseidon_proof,
        &exec.all_permutation_inputs,
        &exec.all_permutation_outputs,
        &mut channel,
    ).map_err(|e| WithdrawError::PoseidonError(format!("{e}")))?;

    // 2. Verify range check proof
    let config = RangeCheckConfig::uint16();
    verify_range_check(&proof.range_check_proof, &config, exec.range_check_limbs.len())
        .map_err(|e| WithdrawError::RangeCheckError(format!("{e}")))?;

    // 3. Verify permutation count
    if exec.all_permutation_inputs.len() != BATCH_SIZE {
        return Err(WithdrawError::PermutationCountMismatch {
            expected: BATCH_SIZE,
            actual: exec.all_permutation_inputs.len(),
        });
    }

    // --- Cross-wiring checks (P1-P5) ---

    // P2: Ownership pk → commitment input
    // commitment_perm0_input[0..4] must equal ownership_perm0_output[0..4]
    let ownership_start = exec.perm_ranges.ownership.0;
    let commitment_start = exec.perm_ranges.commitment.0;
    for j in 0..4 {
        if exec.all_permutation_inputs[commitment_start][j]
            != exec.all_permutation_outputs[ownership_start][j]
        {
            return Err(WithdrawError::CrossWiringFailed(format!(
                "P2: ownership pk[{j}] ({}) != commitment input[{j}] ({})",
                exec.all_permutation_outputs[ownership_start][j].0,
                exec.all_permutation_inputs[commitment_start][j].0,
            )));
        }
    }

    // P3: Nullifier sk = ownership sk
    // nullifier_perm0_input[0..4] must equal ownership_perm0_input[1..5]
    let nullifier_start = exec.perm_ranges.nullifier.0;
    for j in 0..4 {
        if exec.all_permutation_inputs[nullifier_start][j]
            != exec.all_permutation_inputs[ownership_start][j + 1]
        {
            return Err(WithdrawError::CrossWiringFailed(format!(
                "P3: nullifier sk[{j}] ({}) != ownership sk[{j}] ({})",
                exec.all_permutation_inputs[nullifier_start][j].0,
                exec.all_permutation_inputs[ownership_start][j + 1].0,
            )));
        }
    }

    // P4: Nullifier commitment = commitment output
    // nullifier_perm0_input[4..8] must equal commitment[0..4]
    let commitment_end = exec.perm_ranges.commitment.1;
    let actual_commitment_out = &exec.all_permutation_outputs[commitment_end - 1];
    for j in 0..4 {
        if exec.all_permutation_inputs[nullifier_start][4 + j] != actual_commitment_out[j] {
            return Err(WithdrawError::CrossWiringFailed(format!(
                "P4: nullifier perm0 input[{}] ({}) != commitment[{j}] ({})",
                4 + j,
                exec.all_permutation_inputs[nullifier_start][4 + j].0,
                actual_commitment_out[j].0,
            )));
        }
    }
    // nullifier_perm1_input[j] - nullifier_perm0_output[j] must equal commitment[4+j] for j=0..4
    for j in 0..4 {
        let delta = exec.all_permutation_inputs[nullifier_start + 1][j]
            - exec.all_permutation_outputs[nullifier_start][j];
        if delta != actual_commitment_out[4 + j] {
            return Err(WithdrawError::CrossWiringFailed(format!(
                "P4: nullifier sponge absorption at position {j}: delta {} != commitment[{}] ({})",
                delta.0, 4 + j, actual_commitment_out[4 + j].0,
            )));
        }
    }

    // P5: Merkle leaf = commitment
    let merkle_start = exec.perm_ranges.merkle.0;
    let merkle_input = &exec.all_permutation_inputs[merkle_start];
    let left_matches = (0..RATE).all(|j| merkle_input[j] == actual_commitment_out[j]);
    let right_matches = (0..RATE).all(|j| merkle_input[RATE + j] == actual_commitment_out[j]);
    if !left_matches && !right_matches {
        return Err(WithdrawError::CrossWiringFailed(
            "P5: commitment not found in Merkle leaf compress input (neither left nor right half)".into(),
        ));
    }

    // P1: Sponge chain continuity
    verify_sponge_chain(
        &exec.all_permutation_inputs,
        &exec.all_permutation_outputs,
        exec.perm_ranges.commitment.0,
        exec.perm_ranges.commitment.1 - exec.perm_ranges.commitment.0,
        11, // commitment: 11 elements
    ).map_err(|e| WithdrawError::CrossWiringFailed(format!("P1 commitment: {e}")))?;

    verify_sponge_chain(
        &exec.all_permutation_inputs,
        &exec.all_permutation_outputs,
        exec.perm_ranges.nullifier.0,
        exec.perm_ranges.nullifier.1 - exec.perm_ranges.nullifier.0,
        12, // nullifier: 12 elements
    ).map_err(|e| WithdrawError::CrossWiringFailed(format!("P1 nullifier: {e}")))?;

    // --- Existing output-value checks ---

    // 4. Wiring: ownership output matches derived pubkey
    let ownership_range = &exec.perm_ranges.ownership;
    let ownership_out = &exec.all_permutation_outputs[ownership_range.1 - 1];
    let ownership_pk = [ownership_out[0], ownership_out[1], ownership_out[2], ownership_out[3]];
    if ownership_pk != exec.derived_pubkey {
        return Err(WithdrawError::OwnershipFailed {
            derived: ownership_pk,
            expected: exec.derived_pubkey,
        });
    }

    // 5. Wiring: commitment output
    let commitment_range = &exec.perm_ranges.commitment;
    let commitment_out = &exec.all_permutation_outputs[commitment_range.1 - 1];
    let mut computed_commitment = [M31::from_u32_unchecked(0); RATE];
    computed_commitment.copy_from_slice(&commitment_out[..RATE]);
    if computed_commitment != exec.commitment {
        return Err(WithdrawError::CommitmentWiringFailed);
    }

    // 6. Wiring: nullifier output
    let nullifier_range = &exec.perm_ranges.nullifier;
    let nullifier_out = &exec.all_permutation_outputs[nullifier_range.1 - 1];
    let mut computed_nullifier = [M31::from_u32_unchecked(0); RATE];
    computed_nullifier.copy_from_slice(&nullifier_out[..RATE]);
    if computed_nullifier != exec.nullifier {
        return Err(WithdrawError::NullifierWiringFailed);
    }

    // 7. Wiring: Merkle root
    let merkle_range = &exec.perm_ranges.merkle;
    let merkle_out = &exec.all_permutation_outputs[merkle_range.1 - 1];
    let mut computed_root = [M31::from_u32_unchecked(0); RATE];
    computed_root.copy_from_slice(&merkle_out[..RATE]);
    if computed_root != exec.computed_merkle_root {
        return Err(WithdrawError::MerkleInclusionFailed {
            computed: computed_root,
            expected: exec.computed_merkle_root,
        });
    }

    // 8. Public input consistency
    if exec.nullifier != pub_in.nullifier {
        return Err(WithdrawError::PublicInputMismatch("nullifier mismatch".into()));
    }
    if exec.computed_merkle_root != pub_in.merkle_root {
        return Err(WithdrawError::PublicInputMismatch("merkle root mismatch".into()));
    }

    // 9. Amount decomposition verification
    let amount = pub_in.amount_lo.0 as u64 + (pub_in.amount_hi.0 as u64) * (1u64 << 31);
    verify_amount_decomposition(&exec.range_check_limbs, amount)?;

    Ok(())
}

/// Pad permutations to the target batch size with identity (zero-input) permutations.
fn pad_permutations(
    inputs: &mut Vec<[M31; STATE_WIDTH]>,
    outputs: &mut Vec<[M31; STATE_WIDTH]>,
    target: usize,
) {
    while inputs.len() < target {
        let pad_input = [M31::from_u32_unchecked(0); STATE_WIDTH];
        let mut pad_output = pad_input;
        poseidon2_permutation(&mut pad_output);
        inputs.push(pad_input);
        outputs.push(pad_output);
    }
}

fn decompose_amount_limbs(amount_lo: M31, amount_hi: M31) -> Result<Vec<M31>, WithdrawError> {
    let lo_val = amount_lo.0;
    let hi_val = amount_hi.0;

    Ok(vec![
        M31::from_u32_unchecked(lo_val & 0xFFFF),
        M31::from_u32_unchecked(lo_val >> 16),
        M31::from_u32_unchecked(hi_val & 0xFFFF),
        M31::from_u32_unchecked(hi_val >> 16),
    ])
}

fn verify_amount_decomposition(limbs: &[M31], amount: u64) -> Result<(), WithdrawError> {
    if limbs.len() != 4 {
        return Err(WithdrawError::RangeDecomposition(
            format!("expected 4 sub-limbs, got {}", limbs.len()),
        ));
    }

    let amount_lo = limbs[0].0 as u64 + (limbs[1].0 as u64) * 65536;
    let amount_hi = limbs[2].0 as u64 + (limbs[3].0 as u64) * 65536;
    let reconstructed = amount_lo + amount_hi * (1u64 << 31);

    if reconstructed != amount {
        return Err(WithdrawError::RangeDecomposition(
            format!("reconstructed {} != amount {}", reconstructed, amount),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::commitment::derive_pubkey;
    use crate::crypto::merkle_m31::PoseidonMerkleTreeM31;

    fn make_withdraw_witness(amount: u64) -> WithdrawWitness {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let pk = derive_pubkey(&sk);
        let amount_lo = M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32);
        let amount_hi = M31::from_u32_unchecked((amount >> 31) as u32);
        let asset_id = M31::from_u32_unchecked(0);
        let blinding = [10, 20, 30, 40].map(M31::from_u32_unchecked);
        let note = Note::new(pk, asset_id, amount_lo, amount_hi, blinding);

        // Build Merkle tree with this note
        let commitment = note.commitment();
        let mut tree = PoseidonMerkleTreeM31::new(MERKLE_DEPTH);
        tree.append(commitment);

        let merkle_path = tree.prove(0);
        let merkle_root = tree.root();

        WithdrawWitness {
            note,
            spending_key: sk,
            merkle_path,
            merkle_root,
            withdrawal_binding: [M31::from_u32_unchecked(0); RATE],
        }
    }

    #[test]
    fn test_withdraw_prove_verify_basic() {
        let witness = make_withdraw_witness(1000);
        let proof = prove_withdraw(&witness).expect("prove should succeed");
        verify_withdraw(&proof).expect("verify should succeed");
    }

    #[test]
    fn test_withdraw_wrong_key_rejected() {
        let mut witness = make_withdraw_witness(1000);
        witness.spending_key = [1, 2, 3, 4].map(M31::from_u32_unchecked); // wrong key
        let result = prove_withdraw(&witness);
        assert!(result.is_err());
        match result.unwrap_err() {
            WithdrawError::OwnershipFailed { .. } => {}
            e => panic!("expected OwnershipFailed, got: {e}"),
        }
    }

    #[test]
    fn test_withdraw_wrong_merkle_path_rejected() {
        let mut witness = make_withdraw_witness(1000);
        // Corrupt the merkle root
        witness.merkle_root[0] = M31::from_u32_unchecked(999999);
        let result = prove_withdraw(&witness);
        assert!(result.is_err());
        match result.unwrap_err() {
            WithdrawError::MerkleInclusionFailed { .. } => {}
            e => panic!("expected MerkleInclusionFailed, got: {e}"),
        }
    }

    #[test]
    fn test_withdraw_public_inputs_correct() {
        let witness = make_withdraw_witness(500);
        let (_, pub_in) = execute_withdraw(&witness).expect("execute should succeed");

        assert_eq!(pub_in.merkle_root, witness.merkle_root);
        assert_eq!(pub_in.nullifier, witness.note.nullifier(&witness.spending_key));
        assert_eq!(pub_in.amount_lo, witness.note.amount_lo);
        assert_eq!(pub_in.amount_hi, witness.note.amount_hi);
        assert_eq!(pub_in.asset_id, witness.note.asset_id);
    }

    // --- Cross-wiring property tests ---

    /// Helper: create a valid execution, tamper it, re-prove, and verify.
    fn tamper_and_verify(
        witness: &WithdrawWitness,
        tamper: impl FnOnce(&mut WithdrawExecution),
    ) -> Result<(), WithdrawError> {
        let (mut exec, pub_in) = execute_withdraw(witness).unwrap();
        tamper(&mut exec);

        // Re-prove batch with tampered perms
        let mut channel = PoseidonChannel::default();
        let poseidon_proof = prove_poseidon2_batch(&exec.all_permutation_inputs, &mut channel);

        let config = RangeCheckConfig::uint16();
        let range_check_proof = prove_range_check(&exec.range_check_limbs, &config).unwrap();

        let proof = WithdrawProof {
            poseidon_proof,
            range_check_proof,
            execution: exec,
            public_inputs: pub_in,
        };
        verify_withdraw(&proof)
    }

    /// Tamper a permutation input at `perm_idx`, position `pos`, and recompute its output.
    fn tamper_perm(exec: &mut WithdrawExecution, perm_idx: usize, pos: usize, val: u32) {
        exec.all_permutation_inputs[perm_idx][pos] = M31::from_u32_unchecked(val);
        let mut state = exec.all_permutation_inputs[perm_idx];
        poseidon2_permutation(&mut state);
        exec.all_permutation_outputs[perm_idx] = state;
    }

    #[test]
    fn test_withdraw_p1_sponge_chain() {
        let witness = make_withdraw_witness(1000);
        let result = tamper_and_verify(&witness, |exec| {
            // Break nullifier perm1's capacity position 10
            // (Tampering commitment perm1 would change commitment output and trip P4 first.
            // Nullifier perm1 capacity position 10 is outside P4's j=0..3 range, so only P1 fires.)
            let perm1_idx = exec.perm_ranges.nullifier.0 + 1;
            tamper_perm(exec, perm1_idx, 10, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            WithdrawError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P1"), "expected P1 error, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_withdraw_p2_ownership_commitment() {
        let witness = make_withdraw_witness(1000);
        let result = tamper_and_verify(&witness, |exec| {
            // Replace ownership perm with a different valid one (different sk → different pk)
            let own_idx = exec.perm_ranges.ownership.0;
            // Build a new ownership perm input: [DOMAIN_SPEND, new_sk...]
            let mut new_input = [M31::from_u32_unchecked(0); STATE_WIDTH];
            new_input[0] = M31::from_u32_unchecked(0x766D3331); // DOMAIN_SPEND
            new_input[1] = M31::from_u32_unchecked(1); // different sk
            new_input[2] = M31::from_u32_unchecked(2);
            new_input[3] = M31::from_u32_unchecked(3);
            new_input[4] = M31::from_u32_unchecked(4);
            new_input[8] = M31::from_u32_unchecked(5); // domain sep for 5 elements
            exec.all_permutation_inputs[own_idx] = new_input;
            let mut state = new_input;
            poseidon2_permutation(&mut state);
            exec.all_permutation_outputs[own_idx] = state;
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            WithdrawError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P2"), "expected P2 error, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_withdraw_p3_nullifier_sk() {
        let witness = make_withdraw_witness(1000);
        let result = tamper_and_verify(&witness, |exec| {
            // Change nullifier perm0 input[0] (sk[0]) to differ from ownership sk
            let nul_idx = exec.perm_ranges.nullifier.0;
            tamper_perm(exec, nul_idx, 0, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            WithdrawError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P3"), "expected P3 error, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_withdraw_p4_nullifier_commitment_direct() {
        // Test P4 first half: nullifier perm0 input[4..8] vs commitment[0..4]
        let witness = make_withdraw_witness(1000);
        let result = tamper_and_verify(&witness, |exec| {
            let nul_idx = exec.perm_ranges.nullifier.0;
            tamper_perm(exec, nul_idx, 4, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            WithdrawError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P4"), "expected P4 error, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_withdraw_p4_nullifier_commitment_delta() {
        // Test P4 second half: nullifier perm1 absorption delta
        // perm1_input[j] - perm0_output[j] should equal commitment[4+j]
        let witness = make_withdraw_witness(1000);
        let result = tamper_and_verify(&witness, |exec| {
            // Change nullifier perm1 input[0] — breaks the sponge absorption delta
            // while leaving perm0 input[4..8] (first half of P4) intact
            let nul_perm1_idx = exec.perm_ranges.nullifier.0 + 1;
            tamper_perm(exec, nul_perm1_idx, 0, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            WithdrawError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P4"), "expected P4 error, got: {msg}");
                assert!(msg.contains("absorption"), "expected absorption delta error, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_withdraw_p5_merkle_leaf() {
        let witness = make_withdraw_witness(1000);
        let result = tamper_and_verify(&witness, |exec| {
            // Change merkle perm0 input[0] — breaks left-half commitment match
            // (for index 0, commitment is on the left)
            let merkle_idx = exec.perm_ranges.merkle.0;
            tamper_perm(exec, merkle_idx, 0, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            WithdrawError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P5"), "expected P5 error, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_withdraw_permutation_count() {
        let witness = make_withdraw_witness(1000);
        let (exec, _) = execute_withdraw(&witness).expect("execute should succeed");
        assert_eq!(exec.all_permutation_inputs.len(), BATCH_SIZE);
        assert_eq!(exec.all_permutation_outputs.len(), BATCH_SIZE);

        // Verify actual perm count before padding
        let ranges = &exec.perm_ranges;
        let actual_perms = ranges.merkle.1; // last range end
        assert_eq!(actual_perms, INPUT_PERMS);
    }
}
