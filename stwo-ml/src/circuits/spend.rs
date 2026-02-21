//! 2-in/2-out spend circuit: private transfer within the shielded pool.
//!
//! The most complex circuit. For each of 2 inputs: ownership + commitment + nullifier + Merkle.
//! For each of 2 outputs: commitment. Checks balance conservation and asset consistency.
//!
//! Per input note: 1 ownership + 2 commitment + 2 nullifier + 20 Merkle = 25 perms.
//! Per output note: 2 commitment perms.
//! Total: 2×25 + 2×2 = 54 perms → padded to 64 (next power of 2).
//!
//! Composes `Poseidon2BatchProof` + `RangeCheckProof` into a single `SpendProof`.

use stwo::core::channel::MerkleChannel;
use stwo::core::fields::m31::BaseField as M31;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;

use crate::circuits::helpers::{
    record_hash_permutations, record_merkle_permutations, record_ownership_permutations,
    verify_sponge_chain,
};
use crate::circuits::poseidon_circuit::{
    prove_poseidon2_batch, verify_poseidon2_batch, Poseidon2BatchProof,
};
use crate::components::range_check::{prove_range_check, verify_range_check, RangeCheckProof};
use crate::crypto::commitment::{Note, NoteCommitment, Nullifier, PublicKey, SpendingKey};
use crate::crypto::merkle_m31::{Digest, MerklePath};
use crate::crypto::poseidon2_m31::{poseidon2_permutation, DOMAIN_COMPRESS, RATE, STATE_WIDTH};
use crate::crypto::poseidon_channel::PoseidonChannel;
use crate::gadgets::range_check::RangeCheckConfig;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

pub const SPEND_NUM_INPUTS: usize = 2;
pub const SPEND_NUM_OUTPUTS: usize = 2;
pub const MERKLE_DEPTH: usize = 20;
pub const PERMS_PER_INPUT: usize = 25; // 1 + 2 + 2 + 20
pub const PERMS_PER_OUTPUT: usize = 2;
pub const TOTAL_PERMS: usize =
    SPEND_NUM_INPUTS * PERMS_PER_INPUT + SPEND_NUM_OUTPUTS * PERMS_PER_OUTPUT; // 54
pub const BATCH_SIZE: usize = 64; // next power of 2

#[derive(Debug, thiserror::Error)]
pub enum SpendError {
    #[error("ownership failed for input {index}: derived pk {derived:?} != note pk {expected:?}")]
    OwnershipFailed {
        index: usize,
        derived: PublicKey,
        expected: PublicKey,
    },
    #[error("Merkle inclusion failed for input {index}: computed root {computed:?} != expected {expected:?}")]
    MerkleInclusionFailed {
        index: usize,
        computed: Digest,
        expected: Digest,
    },
    #[error("balance check failed: input sum {input_sum} != output sum {output_sum}")]
    BalanceCheckFailed { input_sum: u64, output_sum: u64 },
    #[error("asset mismatch: input {index} has asset {got}, expected {expected}")]
    AssetMismatch {
        index: usize,
        got: u32,
        expected: u32,
    },
    #[error("commitment wiring failed for {which}")]
    CommitmentWiringFailed { which: String },
    #[error("nullifier wiring failed for input {index}")]
    NullifierWiringFailed { index: usize },
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
    #[error("tampered execution: {0}")]
    TamperedExecution(String),
    #[error("cross-wiring verification failed: {0}")]
    CrossWiringFailed(String),
    #[error("zero amount in output {index} (recipient output must be non-zero)")]
    ZeroAmountOutput { index: usize },
}

/// Witness for a single input note.
#[derive(Clone, Debug)]
pub struct InputNoteWitness {
    pub note: Note,
    pub spending_key: SpendingKey,
    pub merkle_path: MerklePath,
}

/// Witness for a single output note.
#[derive(Clone, Debug)]
pub struct OutputNoteWitness {
    pub note: Note,
}

/// Complete spend witness (private).
#[derive(Clone, Debug)]
pub struct SpendWitness {
    pub inputs: [InputNoteWitness; SPEND_NUM_INPUTS],
    pub outputs: [OutputNoteWitness; SPEND_NUM_OUTPUTS],
    pub merkle_root: Digest,
}

/// Public inputs for a spend transaction (visible on-chain).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpendPublicInputs {
    pub merkle_root: Digest,
    pub nullifiers: [Nullifier; SPEND_NUM_INPUTS],
    pub output_commitments: [NoteCommitment; SPEND_NUM_OUTPUTS],
}

/// Permutation index ranges for a single input note.
#[derive(Clone, Debug)]
pub struct InputPermRanges {
    pub ownership: (usize, usize),
    pub commitment: (usize, usize),
    pub nullifier: (usize, usize),
    pub merkle: (usize, usize),
}

/// Permutation index range for a single output note.
#[derive(Clone, Debug)]
pub struct OutputPermRanges {
    pub commitment: (usize, usize),
}

/// Execution trace for the spend circuit (prover-private, NEVER share).
///
/// Contains full Poseidon2 permutation I/O including spending keys,
/// blinding factors, and Merkle path siblings. Zeroized on drop.
#[derive(Clone)]
pub struct SpendExecution {
    pub input_commitments: [NoteCommitment; SPEND_NUM_INPUTS],
    pub input_nullifiers: [Nullifier; SPEND_NUM_INPUTS],
    pub derived_pubkeys: [PublicKey; SPEND_NUM_INPUTS],
    pub computed_merkle_roots: [Digest; SPEND_NUM_INPUTS],
    pub output_commitments: [NoteCommitment; SPEND_NUM_OUTPUTS],
    pub all_permutation_inputs: Vec<[M31; STATE_WIDTH]>,
    pub all_permutation_outputs: Vec<[M31; STATE_WIDTH]>,
    pub input_perm_ranges: [InputPermRanges; SPEND_NUM_INPUTS],
    pub output_perm_ranges: [OutputPermRanges; SPEND_NUM_OUTPUTS],
    pub range_check_limbs: Vec<M31>,
}

impl std::fmt::Debug for SpendExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpendExecution")
            .field("input_nullifiers", &self.input_nullifiers)
            .field("output_commitments", &self.output_commitments)
            .field("num_perms", &self.all_permutation_inputs.len())
            .finish_non_exhaustive()
    }
}

impl Drop for SpendExecution {
    fn drop(&mut self) {
        for perm in &mut self.all_permutation_inputs {
            for v in perm.iter_mut() {
                *v = M31::from_u32_unchecked(0);
            }
        }
        for perm in &mut self.all_permutation_outputs {
            for v in perm.iter_mut() {
                *v = M31::from_u32_unchecked(0);
            }
        }
    }
}

/// Complete spend proof. Does NOT contain the execution trace.
///
/// Safe to serialize, log, or transmit — contains only cryptographic
/// commitments and public inputs, no witness secrets.
#[derive(Debug)]
pub struct SpendProof {
    pub poseidon_proof: Poseidon2BatchProof,
    pub range_check_proof: RangeCheckProof<Blake2sHash>,
    pub public_inputs: SpendPublicInputs,
}

/// Execute the spend circuit: compute all intermediates and collect permutation I/O.
pub fn execute_spend(
    witness: &SpendWitness,
) -> Result<(SpendExecution, SpendPublicInputs), SpendError> {
    let mut all_inputs: Vec<[M31; STATE_WIDTH]> = Vec::with_capacity(BATCH_SIZE);
    let mut all_outputs: Vec<[M31; STATE_WIDTH]> = Vec::with_capacity(BATCH_SIZE);
    let mut offset = 0;

    let mut input_commitments = [[M31::from_u32_unchecked(0); RATE]; SPEND_NUM_INPUTS];
    let mut input_nullifiers = [[M31::from_u32_unchecked(0); RATE]; SPEND_NUM_INPUTS];
    let mut derived_pubkeys = [[M31::from_u32_unchecked(0); 4]; SPEND_NUM_INPUTS];
    let mut computed_merkle_roots = [[M31::from_u32_unchecked(0); RATE]; SPEND_NUM_INPUTS];
    let mut input_perm_ranges_vec = Vec::with_capacity(SPEND_NUM_INPUTS);

    // Check: all notes must have the same asset_id
    let asset_id = witness.inputs[0].note.asset_id;
    for (i, inp) in witness.inputs.iter().enumerate() {
        if inp.note.asset_id != asset_id {
            return Err(SpendError::AssetMismatch {
                index: i,
                got: inp.note.asset_id.0,
                expected: asset_id.0,
            });
        }
    }
    for (i, out) in witness.outputs.iter().enumerate() {
        if out.note.asset_id != asset_id {
            return Err(SpendError::AssetMismatch {
                index: SPEND_NUM_INPUTS + i,
                got: out.note.asset_id.0,
                expected: asset_id.0,
            });
        }
    }

    // Reject zero-amount recipient output (output 0).
    // Zero change (output 1) is allowed for exact-amount transfers.
    {
        let recipient = &witness.outputs[0];
        let recipient_amount =
            recipient.note.amount_lo.0 as u64 + (recipient.note.amount_hi.0 as u64) * (1u64 << 31);
        if recipient_amount == 0 {
            return Err(SpendError::ZeroAmountOutput { index: 0 });
        }
    }

    // Process each input note
    for (i, inp) in witness.inputs.iter().enumerate() {
        // 1. Ownership
        let (derived_pk, ownership_perms) = record_ownership_permutations(&inp.spending_key);
        if derived_pk != inp.note.owner_pubkey {
            return Err(SpendError::OwnershipFailed {
                index: i,
                derived: derived_pk,
                expected: inp.note.owner_pubkey,
            });
        }
        derived_pubkeys[i] = derived_pk;
        let ownership_start = offset;
        for (inp_s, out_s) in &ownership_perms {
            all_inputs.push(*inp_s);
            all_outputs.push(*out_s);
            offset += 1;
        }
        let ownership_end = offset;

        // 2. Commitment hash (11 M31 → 2 perms)
        let commitment_input = [
            inp.note.owner_pubkey[0],
            inp.note.owner_pubkey[1],
            inp.note.owner_pubkey[2],
            inp.note.owner_pubkey[3],
            inp.note.asset_id,
            inp.note.amount_lo,
            inp.note.amount_hi,
            inp.note.blinding[0],
            inp.note.blinding[1],
            inp.note.blinding[2],
            inp.note.blinding[3],
        ];
        let (commitment, commitment_perms) = record_hash_permutations(&commitment_input);
        input_commitments[i] = commitment;
        let commitment_start = offset;
        for (inp_s, out_s) in &commitment_perms {
            all_inputs.push(*inp_s);
            all_outputs.push(*out_s);
            offset += 1;
        }
        let commitment_end = offset;

        // 3. Nullifier hash (12 M31 → 2 perms)
        let mut nullifier_input = [M31::from_u32_unchecked(0); 12];
        nullifier_input[..4].copy_from_slice(&inp.spending_key);
        nullifier_input[4..12].copy_from_slice(&commitment);
        let (nullifier, nullifier_perms) = record_hash_permutations(&nullifier_input);
        input_nullifiers[i] = nullifier;
        let nullifier_start = offset;
        for (inp_s, out_s) in &nullifier_perms {
            all_inputs.push(*inp_s);
            all_outputs.push(*out_s);
            offset += 1;
        }
        let nullifier_end = offset;

        // 4. Merkle path verification (20 compress perms)
        let (computed_root, merkle_perms) =
            record_merkle_permutations(&commitment, &inp.merkle_path);
        if computed_root != witness.merkle_root {
            return Err(SpendError::MerkleInclusionFailed {
                index: i,
                computed: computed_root,
                expected: witness.merkle_root,
            });
        }
        computed_merkle_roots[i] = computed_root;
        let merkle_start = offset;
        for (inp_s, out_s) in &merkle_perms {
            all_inputs.push(*inp_s);
            all_outputs.push(*out_s);
            offset += 1;
        }
        let merkle_end = offset;

        input_perm_ranges_vec.push(InputPermRanges {
            ownership: (ownership_start, ownership_end),
            commitment: (commitment_start, commitment_end),
            nullifier: (nullifier_start, nullifier_end),
            merkle: (merkle_start, merkle_end),
        });
    }

    // Process each output note
    let mut output_commitments = [[M31::from_u32_unchecked(0); RATE]; SPEND_NUM_OUTPUTS];
    let mut output_perm_ranges_vec = Vec::with_capacity(SPEND_NUM_OUTPUTS);

    for (j, out) in witness.outputs.iter().enumerate() {
        let commitment_input = [
            out.note.owner_pubkey[0],
            out.note.owner_pubkey[1],
            out.note.owner_pubkey[2],
            out.note.owner_pubkey[3],
            out.note.asset_id,
            out.note.amount_lo,
            out.note.amount_hi,
            out.note.blinding[0],
            out.note.blinding[1],
            out.note.blinding[2],
            out.note.blinding[3],
        ];
        let (commitment, commitment_perms) = record_hash_permutations(&commitment_input);
        output_commitments[j] = commitment;
        let commitment_start = offset;
        for (inp_s, out_s) in &commitment_perms {
            all_inputs.push(*inp_s);
            all_outputs.push(*out_s);
            offset += 1;
        }
        let commitment_end = offset;

        output_perm_ranges_vec.push(OutputPermRanges {
            commitment: (commitment_start, commitment_end),
        });
    }

    assert_eq!(
        offset, TOTAL_PERMS,
        "expected {TOTAL_PERMS} perms, got {offset}"
    );

    // Balance check: sum of input amounts == sum of output amounts (u64)
    let input_sum: u64 = witness
        .inputs
        .iter()
        .map(|inp| inp.note.amount_lo.0 as u64 + (inp.note.amount_hi.0 as u64) * (1u64 << 31))
        .sum();
    let output_sum: u64 = witness
        .outputs
        .iter()
        .map(|out| out.note.amount_lo.0 as u64 + (out.note.amount_hi.0 as u64) * (1u64 << 31))
        .sum();
    if input_sum != output_sum {
        return Err(SpendError::BalanceCheckFailed {
            input_sum,
            output_sum,
        });
    }

    // Pad to BATCH_SIZE
    while all_inputs.len() < BATCH_SIZE {
        let pad_input = [M31::from_u32_unchecked(0); STATE_WIDTH];
        let mut pad_output = pad_input;
        poseidon2_permutation(&mut pad_output);
        all_inputs.push(pad_input);
        all_outputs.push(pad_output);
    }

    // Range check: decompose output amount limbs into 16-bit sub-limbs
    // 2 outputs × 2 limbs × 2 sub-limbs = 8 sub-limbs
    let mut range_check_limbs = Vec::with_capacity(8);
    for out in &witness.outputs {
        let lo = out.note.amount_lo.0;
        let hi = out.note.amount_hi.0;
        range_check_limbs.push(M31::from_u32_unchecked(lo & 0xFFFF));
        range_check_limbs.push(M31::from_u32_unchecked(lo >> 16));
        range_check_limbs.push(M31::from_u32_unchecked(hi & 0xFFFF));
        range_check_limbs.push(M31::from_u32_unchecked(hi >> 16));
    }

    let input_perm_ranges: [InputPermRanges; SPEND_NUM_INPUTS] = [
        input_perm_ranges_vec.remove(0),
        input_perm_ranges_vec.remove(0),
    ];
    let output_perm_ranges: [OutputPermRanges; SPEND_NUM_OUTPUTS] = [
        output_perm_ranges_vec.remove(0),
        output_perm_ranges_vec.remove(0),
    ];

    let execution = SpendExecution {
        input_commitments,
        input_nullifiers,
        derived_pubkeys,
        computed_merkle_roots,
        output_commitments,
        all_permutation_inputs: all_inputs,
        all_permutation_outputs: all_outputs,
        input_perm_ranges,
        output_perm_ranges,
        range_check_limbs,
    };

    let public_inputs = SpendPublicInputs {
        merkle_root: witness.merkle_root,
        nullifiers: input_nullifiers,
        output_commitments,
    };

    Ok((execution, public_inputs))
}

/// Prove a 2-in/2-out spend transaction.
///
/// Returns `(proof, execution)` separately. The `execution` contains witness
/// secrets (spending keys, blinding, Merkle siblings) and MUST NOT be shared —
/// it is zeroized on drop.
pub fn prove_spend(witness: &SpendWitness) -> Result<(SpendProof, SpendExecution), SpendError> {
    let (execution, public_inputs) = execute_spend(witness)?;

    // Prove Poseidon2 batch (64 permutations)
    let mut channel = PoseidonChannel::default();
    let poseidon_proof = prove_poseidon2_batch(&execution.all_permutation_inputs, &mut channel);

    // Prove range check on output amount sub-limbs
    let config = RangeCheckConfig::uint16();
    let range_check_proof = prove_range_check(&execution.range_check_limbs, &config)
        .map_err(|e| SpendError::RangeCheckError(format!("{e}")))?;

    let proof = SpendProof {
        poseidon_proof,
        range_check_proof,
        public_inputs,
    };
    Ok((proof, execution))
}

/// Verify a spend proof against its execution trace.
///
/// The `execution` is the prover-private trace needed for cross-wiring checks.
/// In production, only the STARK proof path (stark_spend.rs) should be used
/// for third-party verification — it does not require the execution trace.
pub fn verify_spend(proof: &SpendProof, execution: &SpendExecution) -> Result<(), SpendError> {
    let exec = execution;
    let pub_in = &proof.public_inputs;

    // 1. Verify Poseidon2 batch proof
    let mut channel = PoseidonChannel::default();
    verify_poseidon2_batch(
        &proof.poseidon_proof,
        &exec.all_permutation_inputs,
        &exec.all_permutation_outputs,
        &mut channel,
    )
    .map_err(|e| SpendError::PoseidonError(format!("{e}")))?;

    // 2. Verify range check proof
    let config = RangeCheckConfig::uint16();
    verify_range_check(
        &proof.range_check_proof,
        &config,
        exec.range_check_limbs.len(),
    )
    .map_err(|e| SpendError::RangeCheckError(format!("{e}")))?;

    // 3. Verify permutation count
    if exec.all_permutation_inputs.len() != BATCH_SIZE {
        return Err(SpendError::PermutationCountMismatch {
            expected: BATCH_SIZE,
            actual: exec.all_permutation_inputs.len(),
        });
    }

    // --- Cross-wiring checks (P1-P7) ---

    for i in 0..SPEND_NUM_INPUTS {
        let ranges = &exec.input_perm_ranges[i];
        let ownership_start = ranges.ownership.0;
        let commitment_start = ranges.commitment.0;
        let commitment_end = ranges.commitment.1;
        let nullifier_start = ranges.nullifier.0;
        let merkle_start = ranges.merkle.0;
        let actual_commitment_out = &exec.all_permutation_outputs[commitment_end - 1];

        // P2: Ownership pk → commitment input
        for j in 0..4 {
            if exec.all_permutation_inputs[commitment_start][j]
                != exec.all_permutation_outputs[ownership_start][j]
            {
                return Err(SpendError::CrossWiringFailed(format!(
                    "P2 input {i}: ownership pk[{j}] ({}) != commitment input[{j}] ({})",
                    exec.all_permutation_outputs[ownership_start][j].0,
                    exec.all_permutation_inputs[commitment_start][j].0,
                )));
            }
        }

        // P3: Nullifier sk = ownership sk
        for j in 0..4 {
            if exec.all_permutation_inputs[nullifier_start][j]
                != exec.all_permutation_inputs[ownership_start][j + 1]
            {
                return Err(SpendError::CrossWiringFailed(format!(
                    "P3 input {i}: nullifier sk[{j}] ({}) != ownership sk[{j}] ({})",
                    exec.all_permutation_inputs[nullifier_start][j].0,
                    exec.all_permutation_inputs[ownership_start][j + 1].0,
                )));
            }
        }

        // P4: Nullifier commitment = commitment output
        for j in 0..4 {
            if exec.all_permutation_inputs[nullifier_start][4 + j] != actual_commitment_out[j] {
                return Err(SpendError::CrossWiringFailed(format!(
                    "P4 input {i}: nullifier perm0 input[{}] ({}) != commitment[{j}] ({})",
                    4 + j,
                    exec.all_permutation_inputs[nullifier_start][4 + j].0,
                    actual_commitment_out[j].0,
                )));
            }
        }
        for j in 0..4 {
            let delta = exec.all_permutation_inputs[nullifier_start + 1][j]
                - exec.all_permutation_outputs[nullifier_start][j];
            if delta != actual_commitment_out[4 + j] {
                return Err(SpendError::CrossWiringFailed(format!(
                    "P4 input {i}: nullifier sponge absorption at position {j}: delta {} != commitment[{}] ({})",
                    delta.0, 4 + j, actual_commitment_out[4 + j].0,
                )));
            }
        }

        // P5: Merkle leaf = commitment
        // Note: after C5 domain separation, state[RATE+1] in compress inputs
        // has DOMAIN_COMPRESS added, so the right-child check must account for it.
        let merkle_input = &exec.all_permutation_inputs[merkle_start];
        let left_matches = (0..RATE).all(|j| merkle_input[j] == actual_commitment_out[j]);
        let right_matches = (0..RATE).all(|j| {
            let expected = if j == 1 {
                actual_commitment_out[j] + DOMAIN_COMPRESS
            } else {
                actual_commitment_out[j]
            };
            merkle_input[RATE + j] == expected
        });
        if !left_matches && !right_matches {
            return Err(SpendError::CrossWiringFailed(format!(
                "P5 input {i}: commitment not found in Merkle leaf compress input",
            )));
        }
    }

    // P6: Balance conservation — extract amounts from commitment preimages
    {
        let mut input_sum: u64 = 0;
        for i in 0..SPEND_NUM_INPUTS {
            let commitment_start = exec.input_perm_ranges[i].commitment.0;
            let lo = exec.all_permutation_inputs[commitment_start][5].0 as u64;
            let hi = exec.all_permutation_inputs[commitment_start][6].0 as u64;
            input_sum += lo + hi * (1u64 << 31);
        }
        let mut output_sum: u64 = 0;
        for j in 0..SPEND_NUM_OUTPUTS {
            let commitment_start = exec.output_perm_ranges[j].commitment.0;
            let lo = exec.all_permutation_inputs[commitment_start][5].0 as u64;
            let hi = exec.all_permutation_inputs[commitment_start][6].0 as u64;
            output_sum += lo + hi * (1u64 << 31);
        }
        if input_sum != output_sum {
            return Err(SpendError::CrossWiringFailed(format!(
                "P6 balance: input sum {input_sum} != output sum {output_sum}",
            )));
        }
    }

    // P7: Asset consistency — all commitment preimages must have the same asset_id
    {
        let first_asset = exec.all_permutation_inputs[exec.input_perm_ranges[0].commitment.0][4];
        for i in 1..SPEND_NUM_INPUTS {
            let asset = exec.all_permutation_inputs[exec.input_perm_ranges[i].commitment.0][4];
            if asset != first_asset {
                return Err(SpendError::CrossWiringFailed(format!(
                    "P7 asset: input {i} has asset {}, expected {}",
                    asset.0, first_asset.0,
                )));
            }
        }
        for j in 0..SPEND_NUM_OUTPUTS {
            let asset = exec.all_permutation_inputs[exec.output_perm_ranges[j].commitment.0][4];
            if asset != first_asset {
                return Err(SpendError::CrossWiringFailed(format!(
                    "P7 asset: output {j} has asset {}, expected {}",
                    asset.0, first_asset.0,
                )));
            }
        }
    }

    // P1: Sponge chain continuity for all multi-perm hashes
    for i in 0..SPEND_NUM_INPUTS {
        let ranges = &exec.input_perm_ranges[i];
        verify_sponge_chain(
            &exec.all_permutation_inputs,
            &exec.all_permutation_outputs,
            ranges.commitment.0,
            ranges.commitment.1 - ranges.commitment.0,
            11,
        )
        .map_err(|e| SpendError::CrossWiringFailed(format!("P1 input {i} commitment: {e}")))?;

        verify_sponge_chain(
            &exec.all_permutation_inputs,
            &exec.all_permutation_outputs,
            ranges.nullifier.0,
            ranges.nullifier.1 - ranges.nullifier.0,
            12,
        )
        .map_err(|e| SpendError::CrossWiringFailed(format!("P1 input {i} nullifier: {e}")))?;
    }
    for j in 0..SPEND_NUM_OUTPUTS {
        let ranges = &exec.output_perm_ranges[j];
        verify_sponge_chain(
            &exec.all_permutation_inputs,
            &exec.all_permutation_outputs,
            ranges.commitment.0,
            ranges.commitment.1 - ranges.commitment.0,
            11,
        )
        .map_err(|e| SpendError::CrossWiringFailed(format!("P1 output {j} commitment: {e}")))?;
    }

    // --- Existing output-value checks ---

    // 4. Wiring checks per input
    for i in 0..SPEND_NUM_INPUTS {
        let ranges = &exec.input_perm_ranges[i];

        // Ownership output
        let ownership_out = &exec.all_permutation_outputs[ranges.ownership.1 - 1];
        let ownership_pk = [
            ownership_out[0],
            ownership_out[1],
            ownership_out[2],
            ownership_out[3],
        ];
        if ownership_pk != exec.derived_pubkeys[i] {
            return Err(SpendError::OwnershipFailed {
                index: i,
                derived: ownership_pk,
                expected: exec.derived_pubkeys[i],
            });
        }

        // Commitment output
        let commitment_out = &exec.all_permutation_outputs[ranges.commitment.1 - 1];
        let mut computed_commitment = [M31::from_u32_unchecked(0); RATE];
        computed_commitment.copy_from_slice(&commitment_out[..RATE]);
        if computed_commitment != exec.input_commitments[i] {
            return Err(SpendError::CommitmentWiringFailed {
                which: format!("input {i}"),
            });
        }

        // Nullifier output
        let nullifier_out = &exec.all_permutation_outputs[ranges.nullifier.1 - 1];
        let mut computed_nullifier = [M31::from_u32_unchecked(0); RATE];
        computed_nullifier.copy_from_slice(&nullifier_out[..RATE]);
        if computed_nullifier != exec.input_nullifiers[i] {
            return Err(SpendError::NullifierWiringFailed { index: i });
        }

        // Merkle root
        let merkle_out = &exec.all_permutation_outputs[ranges.merkle.1 - 1];
        let mut computed_root = [M31::from_u32_unchecked(0); RATE];
        computed_root.copy_from_slice(&merkle_out[..RATE]);
        if computed_root != exec.computed_merkle_roots[i] {
            return Err(SpendError::MerkleInclusionFailed {
                index: i,
                computed: computed_root,
                expected: exec.computed_merkle_roots[i],
            });
        }
    }

    // 5. Wiring checks per output
    for j in 0..SPEND_NUM_OUTPUTS {
        let ranges = &exec.output_perm_ranges[j];
        let commitment_out = &exec.all_permutation_outputs[ranges.commitment.1 - 1];
        let mut computed_commitment = [M31::from_u32_unchecked(0); RATE];
        computed_commitment.copy_from_slice(&commitment_out[..RATE]);
        if computed_commitment != exec.output_commitments[j] {
            return Err(SpendError::CommitmentWiringFailed {
                which: format!("output {j}"),
            });
        }
    }

    // 6. Public input consistency
    for i in 0..SPEND_NUM_INPUTS {
        if exec.input_nullifiers[i] != pub_in.nullifiers[i] {
            return Err(SpendError::PublicInputMismatch(format!(
                "nullifier {i} mismatch"
            )));
        }
        if exec.computed_merkle_roots[i] != pub_in.merkle_root {
            return Err(SpendError::PublicInputMismatch(format!(
                "merkle root for input {i} doesn't match public root"
            )));
        }
    }
    for j in 0..SPEND_NUM_OUTPUTS {
        if exec.output_commitments[j] != pub_in.output_commitments[j] {
            return Err(SpendError::PublicInputMismatch(format!(
                "output commitment {j} mismatch"
            )));
        }
    }

    // 7. Range check limbs count
    if exec.range_check_limbs.len() != SPEND_NUM_OUTPUTS * 4 {
        return Err(SpendError::RangeDecomposition(format!(
            "expected {} sub-limbs, got {}",
            SPEND_NUM_OUTPUTS * 4,
            exec.range_check_limbs.len()
        )));
    }

    // Balance conservation (P6) and asset consistency (P7) are verified above
    // in the cross-wiring section by reading amounts/assets directly from
    // commitment preimages bound by the Poseidon2 batch proof.

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::commitment::derive_pubkey;
    use crate::crypto::merkle_m31::PoseidonMerkleTreeM31;

    fn make_spend_witness(amounts_in: [u64; 2], amounts_out: [u64; 2]) -> SpendWitness {
        let sk1 = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let sk2 = [50, 60, 70, 80].map(M31::from_u32_unchecked);
        let pk1 = derive_pubkey(&sk1);
        let pk2 = derive_pubkey(&sk2);

        let asset_id = M31::from_u32_unchecked(0); // STRK

        // Create input notes
        let note1 = Note::new(
            pk1,
            asset_id,
            M31::from_u32_unchecked((amounts_in[0] & 0x7FFFFFFF) as u32),
            M31::from_u32_unchecked((amounts_in[0] >> 31) as u32),
            [1, 2, 3, 4].map(M31::from_u32_unchecked),
        );
        let note2 = Note::new(
            pk2,
            asset_id,
            M31::from_u32_unchecked((amounts_in[1] & 0x7FFFFFFF) as u32),
            M31::from_u32_unchecked((amounts_in[1] >> 31) as u32),
            [5, 6, 7, 8].map(M31::from_u32_unchecked),
        );

        // Build Merkle tree with both input notes
        let commitment1 = note1.commitment();
        let commitment2 = note2.commitment();
        let mut tree = PoseidonMerkleTreeM31::new(MERKLE_DEPTH);
        tree.append(commitment1);
        tree.append(commitment2);
        let merkle_root = tree.root();
        let path1 = tree.prove(0).unwrap();
        let path2 = tree.prove(1).unwrap();

        // Create output notes (new recipients)
        let out_sk1 = [100, 200, 300, 400].map(M31::from_u32_unchecked);
        let out_sk2 = [500, 600, 700, 800].map(M31::from_u32_unchecked);
        let out_pk1 = derive_pubkey(&out_sk1);
        let out_pk2 = derive_pubkey(&out_sk2);

        let out_note1 = Note::new(
            out_pk1,
            asset_id,
            M31::from_u32_unchecked((amounts_out[0] & 0x7FFFFFFF) as u32),
            M31::from_u32_unchecked((amounts_out[0] >> 31) as u32),
            [10, 20, 30, 40].map(M31::from_u32_unchecked),
        );
        let out_note2 = Note::new(
            out_pk2,
            asset_id,
            M31::from_u32_unchecked((amounts_out[1] & 0x7FFFFFFF) as u32),
            M31::from_u32_unchecked((amounts_out[1] >> 31) as u32),
            [50, 60, 70, 80].map(M31::from_u32_unchecked),
        );

        SpendWitness {
            inputs: [
                InputNoteWitness {
                    note: note1,
                    spending_key: sk1,
                    merkle_path: path1,
                },
                InputNoteWitness {
                    note: note2,
                    spending_key: sk2,
                    merkle_path: path2,
                },
            ],
            outputs: [
                OutputNoteWitness { note: out_note1 },
                OutputNoteWitness { note: out_note2 },
            ],
            merkle_root,
        }
    }

    #[test]
    fn test_spend_prove_verify_basic() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let (proof, exec) = prove_spend(&witness).expect("prove should succeed");
        verify_spend(&proof, &exec).expect("verify should succeed");
    }

    #[test]
    fn test_spend_wrong_key_rejected() {
        let mut witness = make_spend_witness([1000, 2000], [1500, 1500]);
        witness.inputs[0].spending_key = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let result = prove_spend(&witness);
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::OwnershipFailed { index: 0, .. } => {}
            e => panic!("expected OwnershipFailed index=0, got: {e}"),
        }
    }

    #[test]
    fn test_spend_wrong_merkle_root_rejected() {
        let mut witness = make_spend_witness([1000, 2000], [1500, 1500]);
        witness.merkle_root[0] = M31::from_u32_unchecked(999999);
        let result = prove_spend(&witness);
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::MerkleInclusionFailed { .. } => {}
            e => panic!("expected MerkleInclusionFailed, got: {e}"),
        }
    }

    #[test]
    fn test_spend_unbalanced_rejected() {
        let witness = make_spend_witness([1000, 2000], [1500, 2000]); // 3000 != 3500
        let result = prove_spend(&witness);
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::BalanceCheckFailed {
                input_sum: 3000,
                output_sum: 3500,
            } => {}
            e => panic!("expected BalanceCheckFailed, got: {e}"),
        }
    }

    #[test]
    fn test_spend_different_assets_rejected() {
        let mut witness = make_spend_witness([1000, 2000], [1500, 1500]);
        witness.inputs[1].note.asset_id = M31::from_u32_unchecked(1); // ETH
        let result = prove_spend(&witness);
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::AssetMismatch { index: 1, .. } => {}
            e => panic!("expected AssetMismatch index=1, got: {e}"),
        }
    }

    #[test]
    fn test_spend_permutation_count() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let (exec, _) = execute_spend(&witness).expect("execute should succeed");
        assert_eq!(exec.all_permutation_inputs.len(), BATCH_SIZE);
        assert_eq!(exec.all_permutation_outputs.len(), BATCH_SIZE);

        // Verify actual perm counts before padding
        let last_output_range = &exec.output_perm_ranges[1].commitment;
        assert_eq!(last_output_range.1, TOTAL_PERMS);
    }

    #[test]
    fn test_spend_public_inputs_correctness() {
        let witness = make_spend_witness([500, 500], [300, 700]);
        let (exec, pub_in) = execute_spend(&witness).expect("execute should succeed");

        // Nullifiers match
        for i in 0..SPEND_NUM_INPUTS {
            let expected_nullifier = witness.inputs[i]
                .note
                .nullifier(&witness.inputs[i].spending_key);
            assert_eq!(pub_in.nullifiers[i], expected_nullifier);
            assert_eq!(exec.input_nullifiers[i], expected_nullifier);
        }

        // Output commitments match
        for j in 0..SPEND_NUM_OUTPUTS {
            let expected_commitment = witness.outputs[j].note.commitment();
            assert_eq!(pub_in.output_commitments[j], expected_commitment);
            assert_eq!(exec.output_commitments[j], expected_commitment);
        }

        // Merkle root matches
        assert_eq!(pub_in.merkle_root, witness.merkle_root);
    }

    #[test]
    fn test_spend_large_amounts() {
        // Use amounts that require both lo and hi limbs
        let large = (1u64 << 40) + 42;
        let witness = make_spend_witness([large, 100], [large - 50, 150]);
        let (proof, exec) = prove_spend(&witness).expect("prove should succeed");
        verify_spend(&proof, &exec).expect("verify should succeed");
    }

    #[test]
    fn test_spend_tampered_proof_rejected() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let (proof, mut exec) = prove_spend(&witness).expect("prove should succeed");

        // Tamper with a commitment in the execution
        exec.input_commitments[0][0] = M31::from_u32_unchecked(999999);

        let result = verify_spend(&proof, &exec);
        assert!(result.is_err(), "tampered proof should fail verification");
    }

    #[test]
    fn test_spend_zero_amount() {
        // One zero-amount output is valid (dust/change)
        let witness = make_spend_witness([1000, 0], [1000, 0]);
        let (proof, exec) = prove_spend(&witness).expect("prove should succeed");
        verify_spend(&proof, &exec).expect("verify should succeed");
    }

    #[test]
    fn test_spend_zero_recipient_output_rejected() {
        // Output[0] = 0 (recipient), output[1] = 3000 (change). Zero recipient is rejected.
        let witness = make_spend_witness([1000, 2000], [0, 3000]);
        let result = prove_spend(&witness);
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::ZeroAmountOutput { index: 0 } => {}
            e => panic!("expected ZeroAmountOutput index=0, got: {e}"),
        }
    }

    /// M3 regression: proof struct must NOT contain execution trace.
    #[test]
    fn test_m3_proof_does_not_contain_execution() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let (proof, _exec) = prove_spend(&witness).expect("prove should succeed");

        let debug_str = format!("{:?}", proof);
        assert!(
            !debug_str.contains("all_permutation"),
            "proof Debug must not leak permutation I/O"
        );
        assert!(
            !debug_str.contains("spending_key"),
            "proof Debug must not leak spending keys"
        );
    }

    // --- Cross-wiring property tests ---

    /// Helper: create a valid execution, tamper it, re-prove, and verify.
    fn tamper_and_verify(
        witness: &SpendWitness,
        tamper: impl FnOnce(&mut SpendExecution),
    ) -> Result<(), SpendError> {
        let (mut exec, pub_in) = execute_spend(witness).unwrap();
        tamper(&mut exec);

        let mut channel = PoseidonChannel::default();
        let poseidon_proof = prove_poseidon2_batch(&exec.all_permutation_inputs, &mut channel);

        let config = RangeCheckConfig::uint16();
        let range_check_proof = prove_range_check(&exec.range_check_limbs, &config).unwrap();

        let proof = SpendProof {
            poseidon_proof,
            range_check_proof,
            public_inputs: pub_in,
        };
        verify_spend(&proof, &exec)
    }

    fn tamper_perm(exec: &mut SpendExecution, perm_idx: usize, pos: usize, val: u32) {
        exec.all_permutation_inputs[perm_idx][pos] = M31::from_u32_unchecked(val);
        let mut state = exec.all_permutation_inputs[perm_idx];
        poseidon2_permutation(&mut state);
        exec.all_permutation_outputs[perm_idx] = state;
    }

    #[test]
    fn test_spend_p1_sponge_chain() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let result = tamper_and_verify(&witness, |exec| {
            // Break input 0's nullifier perm1 capacity position 10
            // (Commitment perm1 tampering would change commitment output and trip P4 first.)
            let perm1_idx = exec.input_perm_ranges[0].nullifier.0 + 1;
            tamper_perm(exec, perm1_idx, 10, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P1"), "expected P1, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_spend_p2_ownership_commitment() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let result = tamper_and_verify(&witness, |exec| {
            // Replace ownership perm for input 0 with a different valid one
            let own_idx = exec.input_perm_ranges[0].ownership.0;
            let mut new_input = [M31::from_u32_unchecked(0); STATE_WIDTH];
            new_input[0] = M31::from_u32_unchecked(0x766D3331); // DOMAIN_SPEND
            new_input[1] = M31::from_u32_unchecked(1);
            new_input[2] = M31::from_u32_unchecked(2);
            new_input[3] = M31::from_u32_unchecked(3);
            new_input[4] = M31::from_u32_unchecked(4);
            new_input[8] = M31::from_u32_unchecked(5); // domain sep
            exec.all_permutation_inputs[own_idx] = new_input;
            let mut state = new_input;
            poseidon2_permutation(&mut state);
            exec.all_permutation_outputs[own_idx] = state;
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P2"), "expected P2, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_spend_p3_nullifier_sk() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let result = tamper_and_verify(&witness, |exec| {
            // Change nullifier perm0 input[0] (sk[0]) for input 0
            let nul_idx = exec.input_perm_ranges[0].nullifier.0;
            tamper_perm(exec, nul_idx, 0, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P3"), "expected P3, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_spend_p4_nullifier_commitment() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let result = tamper_and_verify(&witness, |exec| {
            // Change nullifier perm0 input[4] (first commitment element) for input 0
            let nul_idx = exec.input_perm_ranges[0].nullifier.0;
            tamper_perm(exec, nul_idx, 4, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P4"), "expected P4, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_spend_p5_merkle_leaf_left() {
        // Input 0 has Merkle index 0 (even) → commitment is left child
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let result = tamper_and_verify(&witness, |exec| {
            let merkle_idx = exec.input_perm_ranges[0].merkle.0;
            // Tamper left half position 0 — breaks left-child commitment match
            tamper_perm(exec, merkle_idx, 0, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P5"), "expected P5, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_spend_p5_merkle_leaf_right() {
        // Input 1 has Merkle index 1 (odd) → commitment is right child
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let result = tamper_and_verify(&witness, |exec| {
            let merkle_idx = exec.input_perm_ranges[1].merkle.0;
            // Tamper right half position RATE+0 — breaks right-child commitment match
            tamper_perm(exec, merkle_idx, RATE, 999999);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P5"), "expected P5, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_spend_p6_balance() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let result = tamper_and_verify(&witness, |exec| {
            // Change output 0's amount_lo in commitment preimage to break balance
            let out_commit_idx = exec.output_perm_ranges[0].commitment.0;
            // perm0_input[5] = amount_lo. Change from 1500 to 2000.
            tamper_perm(exec, out_commit_idx, 5, 2000);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P6"), "expected P6, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    #[test]
    fn test_spend_p7_asset() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let result = tamper_and_verify(&witness, |exec| {
            // Change output 0's asset_id in commitment preimage
            let out_commit_idx = exec.output_perm_ranges[0].commitment.0;
            // perm0_input[4] = asset_id. Change from 0 to 1.
            tamper_perm(exec, out_commit_idx, 4, 1);
        });
        assert!(result.is_err());
        match result.unwrap_err() {
            SpendError::CrossWiringFailed(msg) => {
                assert!(msg.contains("P7"), "expected P7, got: {msg}");
            }
            e => panic!("expected CrossWiringFailed, got: {e}"),
        }
    }

    // Integration tests

    #[test]
    fn test_deposit_then_spend() {
        use crate::circuits::deposit::{prove_deposit, verify_deposit, DepositWitness};

        let sk1 = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let sk2 = [50, 60, 70, 80].map(M31::from_u32_unchecked);
        let pk1 = derive_pubkey(&sk1);
        let pk2 = derive_pubkey(&sk2);
        let asset_id = M31::from_u32_unchecked(0);

        // Deposit 2 notes
        let note1 = Note::new(
            pk1,
            asset_id,
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            [1, 2, 3, 4].map(M31::from_u32_unchecked),
        );
        let note2 = Note::new(
            pk2,
            asset_id,
            M31::from_u32_unchecked(2000),
            M31::from_u32_unchecked(0),
            [5, 6, 7, 8].map(M31::from_u32_unchecked),
        );

        let dep1 = DepositWitness {
            note: note1.clone(),
            amount: 1000,
            asset_id,
        };
        let dep2 = DepositWitness {
            note: note2.clone(),
            amount: 2000,
            asset_id,
        };
        let (dp1, dp1_exec) = prove_deposit(&dep1).expect("deposit 1 prove");
        verify_deposit(&dp1, &dp1_exec).expect("deposit 1 verify");
        let (dp2, dp2_exec) = prove_deposit(&dep2).expect("deposit 2 prove");
        verify_deposit(&dp2, &dp2_exec).expect("deposit 2 verify");

        // Build Merkle tree from deposited commitments
        let c1 = note1.commitment();
        let c2 = note2.commitment();
        let mut tree = PoseidonMerkleTreeM31::new(MERKLE_DEPTH);
        tree.append(c1);
        tree.append(c2);
        let root = tree.root();
        let path1 = tree.prove(0).unwrap();
        let path2 = tree.prove(1).unwrap();

        // Spend into 2 new outputs
        let out_pk1 = derive_pubkey(&[100, 200, 300, 400].map(M31::from_u32_unchecked));
        let out_pk2 = derive_pubkey(&[500, 600, 700, 800].map(M31::from_u32_unchecked));
        let out_note1 = Note::new(
            out_pk1,
            asset_id,
            M31::from_u32_unchecked(1500),
            M31::from_u32_unchecked(0),
            [10, 20, 30, 40].map(M31::from_u32_unchecked),
        );
        let out_note2 = Note::new(
            out_pk2,
            asset_id,
            M31::from_u32_unchecked(1500),
            M31::from_u32_unchecked(0),
            [50, 60, 70, 80].map(M31::from_u32_unchecked),
        );

        let spend_witness = SpendWitness {
            inputs: [
                InputNoteWitness {
                    note: note1,
                    spending_key: sk1,
                    merkle_path: path1,
                },
                InputNoteWitness {
                    note: note2,
                    spending_key: sk2,
                    merkle_path: path2,
                },
            ],
            outputs: [
                OutputNoteWitness { note: out_note1 },
                OutputNoteWitness { note: out_note2 },
            ],
            merkle_root: root,
        };

        let (spend_proof, spend_exec) = prove_spend(&spend_witness).expect("spend prove");
        verify_spend(&spend_proof, &spend_exec).expect("spend verify");
    }

    #[test]
    fn test_deposit_then_withdraw() {
        use crate::circuits::deposit::{prove_deposit, verify_deposit, DepositWitness};
        use crate::circuits::withdraw::{prove_withdraw, verify_withdraw, WithdrawWitness};

        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let pk = derive_pubkey(&sk);
        let asset_id = M31::from_u32_unchecked(0);

        // Deposit
        let note = Note::new(
            pk,
            asset_id,
            M31::from_u32_unchecked(5000),
            M31::from_u32_unchecked(0),
            [1, 2, 3, 4].map(M31::from_u32_unchecked),
        );
        let dep = DepositWitness {
            note: note.clone(),
            amount: 5000,
            asset_id,
        };
        let (dp, dp_exec) = prove_deposit(&dep).expect("deposit prove");
        verify_deposit(&dp, &dp_exec).expect("deposit verify");

        // Build Merkle tree
        let commitment = note.commitment();
        let mut tree = PoseidonMerkleTreeM31::new(MERKLE_DEPTH);
        tree.append(commitment);

        // Withdraw
        let withdraw_witness = WithdrawWitness {
            note: note.clone(),
            spending_key: sk,
            merkle_path: tree.prove(0).unwrap(),
            merkle_root: tree.root(),
            withdrawal_binding: [M31::from_u32_unchecked(0); RATE],
        };
        let (wp, wp_exec) = prove_withdraw(&withdraw_witness).expect("withdraw prove");
        verify_withdraw(&wp, &wp_exec).expect("withdraw verify");

        // Verify public inputs match
        assert_eq!(wp.public_inputs.amount_lo, note.amount_lo);
        assert_eq!(wp.public_inputs.amount_hi, note.amount_hi);
        assert_eq!(wp.public_inputs.asset_id, note.asset_id);
    }
}
