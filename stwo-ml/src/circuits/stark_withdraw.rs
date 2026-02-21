//! Withdraw Transaction STARK: wraps the withdraw circuit in a single STWO STARK proof.
//!
//! Proves knowledge of a spending key + note + Merkle path such that:
//!   1. The derived pubkey matches the note owner
//!   2. The commitment is correctly formed
//!   3. The nullifier is correctly derived
//!   4. The note is included in the Merkle tree
//!
//! The verifier only sees: merkle_root, nullifier, amount_lo, amount_hi, asset_id.
//!
//! Trace layout (~20,932 columns):
//!   - Perms 0..31: 32 × 652 = 20,864 (25 real + 7 padding permutations)
//!   - Sub-limbs: 4 columns
//!   - Bit decomposition: 64 columns

use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::channel::MerkleChannel;
use stwo::core::fields::m31::BaseField as M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::verifier::verify as stwo_verify;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{BackendForChannel, Col, Column, ColumnOps};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::prove;
use stwo::prover::CommitmentSchemeProver;
use stwo::prover::ComponentProver;

use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator,
};

use crate::backend::convert_evaluations;
use crate::circuits::withdraw::{execute_withdraw, WithdrawPublicInputs, WithdrawWitness};
use crate::components::poseidon2_air::{
    compute_merkle_chain_padding, compute_permutation_trace, constrain_poseidon2_permutation,
    decompose_to_bits, dummy_permutation_trace, write_permutation_to_trace, Poseidon2Columns,
    COLS_PER_PERM,
};
use crate::crypto::poseidon2_m31::{RATE, STATE_WIDTH};

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

const NUM_PERMS: usize = 32; // 25 real (1 ownership + 2 commit + 2 nullifier + 20 Merkle) + 7 padding for padding
const NUM_SUB_LIMBS: usize = 4;
const BITS_PER_LIMB: usize = 16;
const NUM_BIT_COLS: usize = NUM_SUB_LIMBS * BITS_PER_LIMB;
const TOTAL_EXEC_COLS: usize = NUM_PERMS * COLS_PER_PERM + NUM_SUB_LIMBS + NUM_BIT_COLS;
const LOG_SIZE: u32 = 4;

// Permutation index ranges for wiring constraints
const OWNERSHIP_PERM: usize = 0;
const COMMITMENT_PERM_START: usize = 1;
const COMMITMENT_PERM_END: usize = 3; // exclusive
const NULLIFIER_PERM_START: usize = 3;
const NULLIFIER_PERM_END: usize = 5;
pub const MERKLE_PERM_START: usize = 5;
pub const MERKLE_PERM_END: usize = 25;

#[derive(Debug, thiserror::Error)]
pub enum WithdrawStarkError {
    #[error("execution error: {0}")]
    Execution(String),
    #[error("proving error: {0}")]
    Proving(String),
    #[error("verification error: {0}")]
    Verification(String),
}

// ──────────────────────────── Shared wiring constraints ────────────────

/// Withdraw wiring constraints shared between single-tx and batched evaluators.
pub fn constrain_withdraw_wiring<E: EvalAtRow>(
    eval: &mut E,
    is_real: &E::F,
    perms: &[Poseidon2Columns<E::F>],
    sub_limbs: &[E::F; NUM_SUB_LIMBS],
    merkle_root: &[E::F; RATE],
    nullifier: &[E::F; RATE],
    amount_lo: &E::F,
    amount_hi: &E::F,
    asset_id: &E::F,
) {
    // 1. Ownership: perm0 is hash([DOMAIN_SPEND, sk[0..3]])
    eval.add_constraint(
        is_real.clone()
            * (perms[OWNERSHIP_PERM].input()[0].clone()
                - E::F::from(M31::from_u32_unchecked(0x766D3331))),
    );
    for j in 5..RATE {
        eval.add_constraint(is_real.clone() * perms[OWNERSHIP_PERM].input()[j].clone());
    }
    eval.add_constraint(
        is_real.clone()
            * (perms[OWNERSHIP_PERM].input()[RATE].clone()
                - E::F::from(M31::from_u32_unchecked(5))),
    );
    for j in (RATE + 1)..STATE_WIDTH {
        eval.add_constraint(is_real.clone() * perms[OWNERSHIP_PERM].input()[j].clone());
    }

    // 2. Ownership → Commitment: derived pk = perm0.output[0..3]
    eval.add_constraint(
        is_real.clone()
            * (perms[COMMITMENT_PERM_START].input()[RATE].clone()
                - E::F::from(M31::from_u32_unchecked(11))),
    );
    for j in (RATE + 1)..STATE_WIDTH {
        eval.add_constraint(is_real.clone() * perms[COMMITMENT_PERM_START].input()[j].clone());
    }
    for j in 0..4 {
        eval.add_constraint(
            is_real.clone()
                * (perms[COMMITMENT_PERM_START].input()[j].clone()
                    - perms[OWNERSHIP_PERM].output()[j].clone()),
        );
    }

    // 3. Commitment sponge chain
    for j in 3..STATE_WIDTH {
        eval.add_constraint(
            is_real.clone()
                * (perms[COMMITMENT_PERM_END - 1].input()[j].clone()
                    - perms[COMMITMENT_PERM_START].output()[j].clone()),
        );
    }

    // 4. Nullifier hash: domain sep = 12
    eval.add_constraint(
        is_real.clone()
            * (perms[NULLIFIER_PERM_START].input()[RATE].clone()
                - E::F::from(M31::from_u32_unchecked(12))),
    );
    for j in (RATE + 1)..STATE_WIDTH {
        eval.add_constraint(is_real.clone() * perms[NULLIFIER_PERM_START].input()[j].clone());
    }

    // S1: Nullifier sk must match ownership sk
    for j in 0..4 {
        eval.add_constraint(
            is_real.clone()
                * (perms[NULLIFIER_PERM_START].input()[j].clone()
                    - perms[OWNERSHIP_PERM].input()[j + 1].clone()),
        );
    }

    // Nullifier input[4..7] = commitment[0..3]
    for j in 4..RATE {
        eval.add_constraint(
            is_real.clone()
                * (perms[NULLIFIER_PERM_START].input()[j].clone()
                    - perms[COMMITMENT_PERM_END - 1].output()[j - 4].clone()),
        );
    }

    // Nullifier sponge chain
    for j in 4..STATE_WIDTH {
        eval.add_constraint(
            is_real.clone()
                * (perms[NULLIFIER_PERM_END - 1].input()[j].clone()
                    - perms[NULLIFIER_PERM_START].output()[j].clone()),
        );
    }

    // S2: Nullifier chunk 2 absorption
    for j in 0..4 {
        eval.add_constraint(
            is_real.clone()
                * (perms[NULLIFIER_PERM_END - 1].input()[j].clone()
                    - perms[NULLIFIER_PERM_START].output()[j].clone()
                    - perms[COMMITMENT_PERM_END - 1].output()[j + 4].clone()),
        );
    }

    // Nullifier output = public nullifier
    for j in 0..RATE {
        eval.add_constraint(
            is_real.clone()
                * (perms[NULLIFIER_PERM_END - 1].output()[j].clone() - nullifier[j].clone()),
        );
    }

    // 5. Merkle path: degree 2 constraints (no is_real)
    let commitment_perm = COMMITMENT_PERM_END - 1;
    for j in 0..RATE {
        let commit_out = perms[commitment_perm].output()[j].clone();
        let curr_left = perms[MERKLE_PERM_START].input()[j].clone();
        let curr_right = perms[MERKLE_PERM_START].input()[j + RATE].clone();
        eval.add_constraint((curr_left - commit_out.clone()) * (curr_right - commit_out));
    }

    for i in (MERKLE_PERM_START + 1)..MERKLE_PERM_END {
        for j in 0..RATE {
            let prev_out = perms[i - 1].output()[j].clone();
            let curr_left = perms[i].input()[j].clone();
            let curr_right = perms[i].input()[j + RATE].clone();
            eval.add_constraint((curr_left - prev_out.clone()) * (curr_right - prev_out));
        }
    }

    // Merkle root output
    for j in 0..RATE {
        eval.add_constraint(
            is_real.clone()
                * (perms[MERKLE_PERM_END - 1].output()[j].clone() - merkle_root[j].clone()),
        );
    }

    // 6. Amount/asset binding
    eval.add_constraint(
        is_real.clone() * (perms[COMMITMENT_PERM_START].input()[5].clone() - amount_lo.clone()),
    );
    eval.add_constraint(
        is_real.clone() * (perms[COMMITMENT_PERM_START].input()[6].clone() - amount_hi.clone()),
    );
    eval.add_constraint(
        is_real.clone() * (perms[COMMITMENT_PERM_START].input()[4].clone() - asset_id.clone()),
    );

    // Sub-limb ↔ amount binding
    let c65536 = M31::from_u32_unchecked(65536);
    eval.add_constraint(
        is_real.clone()
            * (sub_limbs[0].clone() + sub_limbs[1].clone() * c65536 - amount_lo.clone()),
    );
    eval.add_constraint(
        is_real.clone()
            * (sub_limbs[2].clone() + sub_limbs[3].clone() * c65536 - amount_hi.clone()),
    );
}

// ──────────────────────────── FrameworkEval ────────────────────────────

#[derive(Debug, Clone)]
pub struct WithdrawStarkEval {
    pub log_n_rows: u32,
    pub merkle_root: [M31; RATE],
    pub nullifier: [M31; RATE],
    pub amount_lo: M31,
    pub amount_hi: M31,
    pub asset_id: M31,
}

impl FrameworkEval for WithdrawStarkEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_real = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "is_real".into(),
        });

        // ── Read all permutation columns ──
        let perms: Vec<_> = (0..NUM_PERMS)
            .map(|_| constrain_poseidon2_permutation(&mut eval))
            .collect();

        // ── Sub-limb & bit columns ──
        let sub_limbs: [E::F; NUM_SUB_LIMBS] = std::array::from_fn(|_| eval.next_trace_mask());
        let bits: [[E::F; BITS_PER_LIMB]; NUM_SUB_LIMBS] =
            std::array::from_fn(|_| std::array::from_fn(|_| eval.next_trace_mask()));

        // ── Bit constraints (all rows) ──
        let one = M31::from_u32_unchecked(1);
        for limb_bits in &bits {
            for bit in limb_bits {
                eval.add_constraint(bit.clone() * (bit.clone() - E::F::from(one)));
            }
        }
        for (k, sub_limb) in sub_limbs.iter().enumerate() {
            let mut reconstructed = E::F::zero();
            for (i, bit) in bits[k].iter().enumerate() {
                reconstructed += bit.clone() * M31::from_u32_unchecked(1u32 << i);
            }
            eval.add_constraint(reconstructed - sub_limb.clone());
        }

        // ── Wiring constraints via shared helper ──
        let merkle_root: [E::F; RATE] = std::array::from_fn(|j| E::F::from(self.merkle_root[j]));
        let nullifier: [E::F; RATE] = std::array::from_fn(|j| E::F::from(self.nullifier[j]));
        let amount_lo = E::F::from(self.amount_lo);
        let amount_hi = E::F::from(self.amount_hi);
        let asset_id = E::F::from(self.asset_id);

        constrain_withdraw_wiring(
            &mut eval,
            &is_real,
            &perms,
            &sub_limbs,
            &merkle_root,
            &nullifier,
            &amount_lo,
            &amount_hi,
            &asset_id,
        );

        eval
    }
}

pub type WithdrawStarkComponent = FrameworkComponent<WithdrawStarkEval>;

// ──────────────────────────── Proof type ───────────────────────────────

#[derive(Debug)]
pub struct WithdrawStarkProof {
    pub stark_proof: StarkProof<Blake2sHash>,
    pub public_inputs: WithdrawPublicInputs,
}

// ──────────────────────────── Prover ──────────────────────────────────

pub fn prove_withdraw_stark(
    witness: &WithdrawWitness,
) -> Result<WithdrawStarkProof, WithdrawStarkError> {
    #[cfg(feature = "cuda-runtime")]
    {
        if crate::backend::force_gpu() || crate::backend::gpu_is_available() {
            use stwo::prover::backend::gpu::GpuBackend;
            return prove_withdraw_stark_with::<GpuBackend>(witness);
        }
    }
    prove_withdraw_stark_with::<SimdBackend>(witness)
}

fn prove_withdraw_stark_with<B>(
    witness: &WithdrawWitness,
) -> Result<WithdrawStarkProof, WithdrawStarkError>
where
    B: BackendForChannel<Blake2sMerkleChannel> + PolyOps + ColumnOps<M31>,
    Col<B, M31>: Column<M31>,
    <B as ColumnOps<M31>>::Column: 'static,
    FrameworkComponent<WithdrawStarkEval>: ComponentProver<B>,
{
    let (execution, public_inputs) =
        execute_withdraw(witness).map_err(|e| WithdrawStarkError::Execution(format!("{e}")))?;

    let table_size = 1usize << LOG_SIZE;

    // Compute permutation traces
    let perm_traces: Vec<_> = execution
        .all_permutation_inputs
        .iter()
        .map(compute_permutation_trace)
        .collect();
    let dummy = dummy_permutation_trace();
    // Dummy output hash (first RATE elements) — used as start for Merkle chain padding
    let dummy_hash: [M31; RATE] = std::array::from_fn(|j| dummy.states[22][j]);
    // Chained Merkle padding traces so degree-2 chain constraints hold on padding rows
    let merkle_levels = MERKLE_PERM_END - MERKLE_PERM_START;
    let merkle_padding = compute_merkle_chain_padding(&dummy_hash, merkle_levels);

    // Build execution columns
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let mut exec_cols: Vec<Col<B, M31>> = (0..TOTAL_EXEC_COLS)
        .map(|_| Col::<B, M31>::zeros(table_size))
        .collect();

    // Row 0: real traces
    for (p, trace) in perm_traces.iter().enumerate() {
        write_permutation_to_trace::<B>(trace, &mut exec_cols, p * COLS_PER_PERM, 0);
    }
    // Padding rows: non-Merkle perms use dummy traces, Merkle perms use chained traces
    for row in 1..table_size {
        for p in 0..NUM_PERMS {
            if p >= MERKLE_PERM_START && p < MERKLE_PERM_END {
                let merkle_idx = p - MERKLE_PERM_START;
                write_permutation_to_trace::<B>(
                    &merkle_padding[merkle_idx],
                    &mut exec_cols,
                    p * COLS_PER_PERM,
                    row,
                );
            } else {
                write_permutation_to_trace::<B>(&dummy, &mut exec_cols, p * COLS_PER_PERM, row);
            }
        }
    }

    // Sub-limbs and bits
    let sub_offset = NUM_PERMS * COLS_PER_PERM;
    for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
        exec_cols[sub_offset + k].set(0, limb);
    }
    let bit_offset = sub_offset + NUM_SUB_LIMBS;
    for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
        let bit_vals = decompose_to_bits(limb.0);
        for (i, &bit) in bit_vals.iter().enumerate() {
            exec_cols[bit_offset + k * BITS_PER_LIMB + i].set(0, M31::from_u32_unchecked(bit));
        }
    }

    // Preprocessed
    let mut is_real_col = Col::<B, M31>::zeros(table_size);
    is_real_col.set(0, M31::from_u32_unchecked(1));
    let preprocessed = vec![CircleEvaluation::<B, M31, BitReversedOrder>::new(
        domain,
        is_real_col,
    )];
    let execution_evals: Vec<CircleEvaluation<B, M31, BitReversedOrder>> = exec_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    // PCS
    let pcs_config = PcsConfig::default();
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(LOG_SIZE + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<B, Blake2sMerkleChannel>::new(pcs_config, &twiddles);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<B, B, M31>(preprocessed));
    tree_builder.commit(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<B, B, M31>(execution_evals));
    tree_builder.commit(channel);

    let eval = WithdrawStarkEval {
        log_n_rows: LOG_SIZE,
        merkle_root: public_inputs.merkle_root,
        nullifier: public_inputs.nullifier,
        amount_lo: public_inputs.amount_lo,
        amount_hi: public_inputs.amount_hi,
        asset_id: public_inputs.asset_id,
    };
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        eval,
        SecureField::zero(),
    );

    let stark_proof = prove::<B, Blake2sMerkleChannel>(&[&component], channel, commitment_scheme)
        .map_err(|e| WithdrawStarkError::Proving(format!("{e:?}")))?;

    Ok(WithdrawStarkProof {
        stark_proof,
        public_inputs,
    })
}

// ──────────────────────────── Verifier ─────────────────────────────────

pub fn verify_withdraw_stark(
    proof: &WithdrawStarkProof,
    public_inputs: &WithdrawPublicInputs,
) -> Result<(), WithdrawStarkError> {
    let pcs_config = PcsConfig::default();

    let dummy_eval = WithdrawStarkEval {
        log_n_rows: LOG_SIZE,
        merkle_root: public_inputs.merkle_root,
        nullifier: public_inputs.nullifier,
        amount_lo: public_inputs.amount_lo,
        amount_hi: public_inputs.amount_hi,
        asset_id: public_inputs.asset_id,
    };
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component = FrameworkComponent::new(&mut allocator, dummy_eval, SecureField::zero());
    let bounds = Component::trace_log_degree_bounds(&dummy_component);

    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);

    commitment_scheme.commit(proof.stark_proof.commitments[0], &bounds[0], channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &bounds[1], channel);

    let real_eval = WithdrawStarkEval {
        log_n_rows: LOG_SIZE,
        merkle_root: public_inputs.merkle_root,
        nullifier: public_inputs.nullifier,
        amount_lo: public_inputs.amount_lo,
        amount_hi: public_inputs.amount_hi,
        asset_id: public_inputs.asset_id,
    };
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(&mut allocator, real_eval, SecureField::zero());

    stwo_verify::<Blake2sMerkleChannel>(
        &[&component as &dyn Component],
        channel,
        &mut commitment_scheme,
        proof.stark_proof.clone(),
    )
    .map_err(|e| WithdrawStarkError::Verification(format!("{e:?}")))
}

// ──────────────────────────── Tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::commitment::{derive_pubkey, Note};
    use crate::crypto::merkle_m31::PoseidonMerkleTreeM31;

    fn make_withdraw_witness(amount: u64) -> WithdrawWitness {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let pk = derive_pubkey(&sk);
        let amount_lo = M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32);
        let amount_hi = M31::from_u32_unchecked((amount >> 31) as u32);
        let asset_id = M31::from_u32_unchecked(0);
        let blinding = [10, 20, 30, 40].map(M31::from_u32_unchecked);
        let note = Note::new(pk, asset_id, amount_lo, amount_hi, blinding);

        let commitment = note.commitment();
        let mut tree = PoseidonMerkleTreeM31::new(crate::circuits::withdraw::MERKLE_DEPTH);
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
    fn test_withdraw_stark_prove_verify() {
        let witness = make_withdraw_witness(1000);
        let proof = prove_withdraw_stark(&witness).expect("proving should succeed");
        verify_withdraw_stark(&proof, &proof.public_inputs).expect("verification should succeed");
    }

    #[test]
    fn test_withdraw_stark_wrong_root_rejected() {
        let witness = make_withdraw_witness(1000);
        let proof = prove_withdraw_stark(&witness).expect("proving should succeed");

        let mut bad_inputs = proof.public_inputs.clone();
        bad_inputs.merkle_root[0] = M31::from_u32_unchecked(999999);

        let result = verify_withdraw_stark(&proof, &bad_inputs);
        assert!(result.is_err(), "modified root should fail");
    }

    #[test]
    fn test_withdraw_stark_wrong_nullifier_rejected() {
        let witness = make_withdraw_witness(1000);
        let proof = prove_withdraw_stark(&witness).expect("proving should succeed");

        let mut bad_inputs = proof.public_inputs.clone();
        bad_inputs.nullifier[0] = M31::from_u32_unchecked(999999);

        let result = verify_withdraw_stark(&proof, &bad_inputs);
        assert!(result.is_err(), "modified nullifier should fail");
    }

    #[test]
    fn test_withdraw_stark_wrong_key_rejected() {
        let mut witness = make_withdraw_witness(1000);
        witness.spending_key = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let result = prove_withdraw_stark(&witness);
        assert!(result.is_err(), "wrong key should fail at execution");
    }

    #[test]
    fn test_withdraw_stark_permutation_count() {
        let witness = make_withdraw_witness(1000);
        let (exec, _) = execute_withdraw(&witness).expect("execute");
        assert_eq!(exec.all_permutation_inputs.len(), NUM_PERMS);
    }

    #[test]
    fn test_withdraw_stark_consistent_with_phase3() {
        use crate::circuits::withdraw::{prove_withdraw, verify_withdraw};

        let witness = make_withdraw_witness(500);

        let phase3_proof = prove_withdraw(&witness).expect("phase3 prove");
        verify_withdraw(&phase3_proof).expect("phase3 verify");

        let stark_proof = prove_withdraw_stark(&witness).expect("stark prove");
        verify_withdraw_stark(&stark_proof, &stark_proof.public_inputs).expect("stark verify");

        assert_eq!(
            phase3_proof.public_inputs.nullifier,
            stark_proof.public_inputs.nullifier
        );
        assert_eq!(
            phase3_proof.public_inputs.merkle_root,
            stark_proof.public_inputs.merkle_root
        );
    }
}
