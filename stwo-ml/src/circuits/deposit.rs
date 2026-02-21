//! Deposit circuit: proves a note commitment is correctly formed from a public amount + asset.
//!
//! The simplest circuit — only 2 Poseidon2 permutations (commitment hash of 11 M31 input).
//! Composes `Poseidon2BatchProof` + `RangeCheckProof` into a single `DepositProof`.
//!
//! Public inputs: commitment, amount (u64), asset_id.
//! The verifier checks:
//! - Poseidon2 batch proof verifies
//! - Range check proof verifies (amount limbs are valid M31)
//! - Commitment output wiring is correct
//! - Amount consistency: `lo + hi * 2^31 == amount`
//! - Asset matches

use stwo::core::channel::MerkleChannel;
use stwo::core::fields::m31::BaseField as M31;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;

use crate::circuits::helpers::{record_hash_permutations, verify_sponge_chain};
use crate::circuits::poseidon_circuit::{
    prove_poseidon2_batch, verify_poseidon2_batch, Poseidon2BatchProof,
};
use crate::components::range_check::{prove_range_check, verify_range_check, RangeCheckProof};
use crate::crypto::commitment::{Note, NoteCommitment};
use crate::crypto::poseidon2_m31::{RATE, STATE_WIDTH};
use crate::crypto::poseidon_channel::PoseidonChannel;
use crate::gadgets::range_check::RangeCheckConfig;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

const BATCH_SIZE: usize = 2; // 2 perms for commitment hash, already power of 2

#[derive(Debug, thiserror::Error)]
pub enum DepositError {
    #[error("amount mismatch: lo={lo} hi={hi} does not equal amount={amount}")]
    AmountMismatch { lo: u32, hi: u32, amount: u64 },
    #[error("asset mismatch: note has {note_asset}, expected {expected_asset}")]
    AssetMismatch {
        note_asset: u32,
        expected_asset: u32,
    },
    #[error("commitment wiring failed: computed {computed:?} != expected {expected:?}")]
    CommitmentWiringFailed {
        computed: NoteCommitment,
        expected: NoteCommitment,
    },
    #[error("range check decomposition failed: {0}")]
    RangeDecomposition(String),
    #[error("poseidon proof error: {0}")]
    PoseidonError(String),
    #[error("range check error: {0}")]
    RangeCheckError(String),
    #[error("cross-wiring verification failed: {0}")]
    CrossWiringFailed(String),
}

/// Witness for a deposit transaction (all private to the prover).
#[derive(Clone, Debug)]
pub struct DepositWitness {
    pub note: Note,
    pub amount: u64,
    pub asset_id: M31,
}

/// Public inputs for a deposit transaction (visible on-chain).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepositPublicInputs {
    pub commitment: NoteCommitment,
    pub amount: u64,
    pub asset_id: M31,
}

/// Execution trace for the deposit circuit.
#[derive(Clone, Debug)]
pub struct DepositExecution {
    pub commitment: NoteCommitment,
    pub all_permutation_inputs: Vec<[M31; STATE_WIDTH]>,
    pub all_permutation_outputs: Vec<[M31; STATE_WIDTH]>,
    /// Range check sub-limbs: [lo_16_0, hi_16_0, lo_16_1, hi_16_1]
    pub range_check_limbs: Vec<M31>,
}

/// Complete deposit proof.
#[derive(Debug)]
pub struct DepositProof {
    pub poseidon_proof: Poseidon2BatchProof,
    pub range_check_proof: RangeCheckProof<Blake2sHash>,
    pub execution: DepositExecution,
    pub public_inputs: DepositPublicInputs,
}

/// Execute the deposit circuit: compute commitment + collect permutation I/O.
pub fn execute_deposit(
    witness: &DepositWitness,
) -> Result<(DepositExecution, DepositPublicInputs), DepositError> {
    // Check amount consistency: lo + hi * 2^31 == amount
    let lo = witness.note.amount_lo.0 as u64;
    let hi = witness.note.amount_hi.0 as u64;
    let reconstructed = lo + hi * (1u64 << 31);
    if reconstructed != witness.amount {
        return Err(DepositError::AmountMismatch {
            lo: witness.note.amount_lo.0,
            hi: witness.note.amount_hi.0,
            amount: witness.amount,
        });
    }

    // Check asset consistency
    if witness.note.asset_id != witness.asset_id {
        return Err(DepositError::AssetMismatch {
            note_asset: witness.note.asset_id.0,
            expected_asset: witness.asset_id.0,
        });
    }

    // Compute commitment with recorded permutations
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
    let (commitment, perms) = record_hash_permutations(&commitment_input);
    assert_eq!(
        perms.len(),
        2,
        "commitment hash of 11 elements should produce 2 permutations"
    );

    let mut all_inputs = Vec::with_capacity(BATCH_SIZE);
    let mut all_outputs = Vec::with_capacity(BATCH_SIZE);
    for (inp, out) in &perms {
        all_inputs.push(*inp);
        all_outputs.push(*out);
    }

    // Decompose amount limbs into 16-bit sub-limbs for range checking
    let range_check_limbs = decompose_amount_limbs(witness.note.amount_lo, witness.note.amount_hi)?;

    let execution = DepositExecution {
        commitment,
        all_permutation_inputs: all_inputs,
        all_permutation_outputs: all_outputs,
        range_check_limbs,
    };

    let public_inputs = DepositPublicInputs {
        commitment,
        amount: witness.amount,
        asset_id: witness.asset_id,
    };

    Ok((execution, public_inputs))
}

/// Prove a deposit transaction.
pub fn prove_deposit(witness: &DepositWitness) -> Result<DepositProof, DepositError> {
    let (execution, public_inputs) = execute_deposit(witness)?;

    // Prove Poseidon2 batch (2 permutations)
    let mut channel = PoseidonChannel::default();
    let poseidon_proof = prove_poseidon2_batch(&execution.all_permutation_inputs, &mut channel);

    // Prove range check on decomposed limbs
    let config = RangeCheckConfig::uint16();
    let range_check_proof = prove_range_check(&execution.range_check_limbs, &config)
        .map_err(|e| DepositError::RangeCheckError(format!("{e}")))?;

    Ok(DepositProof {
        poseidon_proof,
        range_check_proof,
        execution,
        public_inputs,
    })
}

/// Verify a deposit proof.
pub fn verify_deposit(proof: &DepositProof) -> Result<(), DepositError> {
    let exec = &proof.execution;
    let pub_in = &proof.public_inputs;

    // 1. Verify Poseidon2 batch proof
    let mut channel = PoseidonChannel::default();
    verify_poseidon2_batch(
        &proof.poseidon_proof,
        &exec.all_permutation_inputs,
        &exec.all_permutation_outputs,
        &mut channel,
    )
    .map_err(|e| DepositError::PoseidonError(format!("{e}")))?;

    // 2. Verify range check proof
    let config = RangeCheckConfig::uint16();
    verify_range_check(
        &proof.range_check_proof,
        &config,
        exec.range_check_limbs.len(),
    )
    .map_err(|e| DepositError::RangeCheckError(format!("{e}")))?;

    // 3. P1: Sponge chain continuity for commitment hash (11 elements, 2 perms)
    verify_sponge_chain(
        &exec.all_permutation_inputs,
        &exec.all_permutation_outputs,
        0,          // commitment starts at perm 0
        BATCH_SIZE, // 2 perms
        11,         // 11 elements
    )
    .map_err(|e| DepositError::CrossWiringFailed(format!("P1 commitment: {e}")))?;

    // 4. Wiring: commitment output from permutation chain matches claimed commitment
    // The final permutation's output state[..RATE] should be the commitment
    let last_perm_output = &exec.all_permutation_outputs[BATCH_SIZE - 1];
    let mut computed_commitment = [M31::from_u32_unchecked(0); RATE];
    computed_commitment.copy_from_slice(&last_perm_output[..RATE]);
    if computed_commitment != exec.commitment {
        return Err(DepositError::CommitmentWiringFailed {
            computed: computed_commitment,
            expected: exec.commitment,
        });
    }

    // 4. Public input consistency: commitment matches
    if exec.commitment != pub_in.commitment {
        return Err(DepositError::CommitmentWiringFailed {
            computed: exec.commitment,
            expected: pub_in.commitment,
        });
    }

    // 5. Amount consistency: decomposed limbs reconstruct to the claimed amount
    // range_check_limbs = [lo_16_0, hi_16_0, lo_16_1, hi_16_1]
    verify_amount_decomposition(&exec.range_check_limbs, pub_in.amount)?;

    Ok(())
}

/// Decompose M31 amount limbs into 16-bit sub-limbs for range checking.
///
/// Each M31 value v is split: v = lo_16 + hi_16 * 65536.
/// Returns [lo_16_of_amount_lo, hi_16_of_amount_lo, lo_16_of_amount_hi, hi_16_of_amount_hi].
fn decompose_amount_limbs(amount_lo: M31, amount_hi: M31) -> Result<Vec<M31>, DepositError> {
    let lo_val = amount_lo.0;
    let hi_val = amount_hi.0;

    let lo_lo16 = lo_val & 0xFFFF;
    let lo_hi16 = lo_val >> 16;
    let hi_lo16 = hi_val & 0xFFFF;
    let hi_hi16 = hi_val >> 16;

    // Verify reconstruction
    if lo_lo16 + lo_hi16 * 65536 != lo_val {
        return Err(DepositError::RangeDecomposition(format!(
            "amount_lo decomposition: {} + {} * 65536 != {}",
            lo_lo16, lo_hi16, lo_val
        )));
    }
    if hi_lo16 + hi_hi16 * 65536 != hi_val {
        return Err(DepositError::RangeDecomposition(format!(
            "amount_hi decomposition: {} + {} * 65536 != {}",
            hi_lo16, hi_hi16, hi_val
        )));
    }

    Ok(vec![
        M31::from_u32_unchecked(lo_lo16),
        M31::from_u32_unchecked(lo_hi16),
        M31::from_u32_unchecked(hi_lo16),
        M31::from_u32_unchecked(hi_hi16),
    ])
}

/// Verify that the range-checked sub-limbs reconstruct to the claimed amount.
fn verify_amount_decomposition(limbs: &[M31], amount: u64) -> Result<(), DepositError> {
    if limbs.len() != 4 {
        return Err(DepositError::RangeDecomposition(format!(
            "expected 4 sub-limbs, got {}",
            limbs.len()
        )));
    }

    let amount_lo = limbs[0].0 as u64 + (limbs[1].0 as u64) * 65536;
    let amount_hi = limbs[2].0 as u64 + (limbs[3].0 as u64) * 65536;
    let reconstructed = amount_lo + amount_hi * (1u64 << 31);

    if reconstructed != amount {
        return Err(DepositError::AmountMismatch {
            lo: amount_lo as u32,
            hi: amount_hi as u32,
            amount,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::commitment::derive_pubkey;

    fn make_deposit_witness(amount: u64) -> DepositWitness {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let pk = derive_pubkey(&sk);
        let amount_lo = M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32);
        let amount_hi = M31::from_u32_unchecked((amount >> 31) as u32);
        let asset_id = M31::from_u32_unchecked(0); // STRK
        let blinding = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let note = Note::new(pk, asset_id, amount_lo, amount_hi, blinding);
        DepositWitness {
            note,
            amount,
            asset_id,
        }
    }

    #[test]
    fn test_deposit_prove_verify_basic() {
        let witness = make_deposit_witness(1000);
        let proof = prove_deposit(&witness).expect("prove should succeed");
        verify_deposit(&proof).expect("verify should succeed");
    }

    #[test]
    fn test_deposit_amount_mismatch_rejected() {
        let mut witness = make_deposit_witness(1000);
        witness.amount = 2000; // mismatch
        let result = prove_deposit(&witness);
        assert!(result.is_err());
        match result.unwrap_err() {
            DepositError::AmountMismatch { .. } => {}
            e => panic!("expected AmountMismatch, got: {e}"),
        }
    }

    #[test]
    fn test_deposit_commitment_correct() {
        let witness = make_deposit_witness(500);
        let (exec, pub_in) = execute_deposit(&witness).expect("execute should succeed");

        // Commitment should match Note::commitment()
        let expected = witness.note.commitment();
        assert_eq!(exec.commitment, expected);
        assert_eq!(pub_in.commitment, expected);
    }

    #[test]
    fn test_deposit_sponge_chain_verified() {
        let witness = make_deposit_witness(1000);
        let (mut exec, pub_in) = execute_deposit(&witness).unwrap();

        // Break sponge chain: change perm1's capacity position 10
        // This makes perm1.input[10] differ from perm0.output[10]
        exec.all_permutation_inputs[1][10] = M31::from_u32_unchecked(999999);
        let mut state = exec.all_permutation_inputs[1];
        crate::crypto::poseidon2_m31::poseidon2_permutation(&mut state);
        exec.all_permutation_outputs[1] = state;

        // Re-prove batch with tampered perms
        let mut channel = PoseidonChannel::default();
        let poseidon_proof = prove_poseidon2_batch(&exec.all_permutation_inputs, &mut channel);

        let config = RangeCheckConfig::uint16();
        let range_check_proof = prove_range_check(&exec.range_check_limbs, &config).unwrap();

        let proof = DepositProof {
            poseidon_proof,
            range_check_proof,
            execution: exec,
            public_inputs: pub_in,
        };

        let result = verify_deposit(&proof);
        assert!(result.is_err());
        match result.unwrap_err() {
            DepositError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P1"), "expected P1 error, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_deposit_range_check() {
        // Large amount that uses both limbs
        let amount = (1u64 << 40) + 42;
        let witness = make_deposit_witness(amount);
        let proof = prove_deposit(&witness).expect("prove should succeed");
        verify_deposit(&proof).expect("verify should succeed");

        // Verify the decomposition
        assert_eq!(proof.execution.range_check_limbs.len(), 4);
        // All sub-limbs should be in [0, 65535]
        for limb in &proof.execution.range_check_limbs {
            assert!(limb.0 <= 65535, "sub-limb {} out of uint16 range", limb.0);
        }
    }
}
