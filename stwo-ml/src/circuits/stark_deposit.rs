//! Deposit Transaction STARK: wraps the deposit circuit in a single STWO STARK proof.
//!
//! The deposit STARK proves that a note commitment is correctly formed from a public
//! amount and asset, without revealing the note contents (owner, blinding).
//!
//! Trace layout (1,372 columns):
//!   - Perm 0: 652 columns (commitment hash chunk 1)
//!   - Perm 1: 652 columns (commitment hash chunk 2)
//!   - Sub-limbs: 4 columns (uint16 decomposition of amount limbs)
//!   - Bit decomposition: 64 columns (4 × 16 bits)
//!
//! Public inputs: commitment (8 M31), amount_lo, amount_hi, asset_id.

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
use stwo::prover::CommitmentSchemeProver;
use stwo::prover::ComponentProver;
use stwo::prover::prove;

use stwo_constraint_framework::{
    EvalAtRow, FrameworkComponent, FrameworkEval, TraceLocationAllocator,
};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::backend::convert_evaluations;
use crate::circuits::deposit::{DepositPublicInputs, DepositWitness, execute_deposit};
use crate::components::poseidon2_air::{
    constrain_poseidon2_permutation, compute_permutation_trace, decompose_to_bits,
    dummy_permutation_trace, write_permutation_to_trace, Poseidon2Columns, COLS_PER_PERM,
};
use crate::crypto::poseidon2_m31::{RATE, STATE_WIDTH};

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

const NUM_PERMS: usize = 2;
pub const NUM_SUB_LIMBS: usize = 4;
const BITS_PER_LIMB: usize = 16;
const NUM_BIT_COLS: usize = NUM_SUB_LIMBS * BITS_PER_LIMB;
const TOTAL_EXEC_COLS: usize = NUM_PERMS * COLS_PER_PERM + NUM_SUB_LIMBS + NUM_BIT_COLS;
const LOG_SIZE: u32 = 4; // 16 rows (STWO minimum)

#[derive(Debug, thiserror::Error)]
pub enum DepositStarkError {
    #[error("execution error: {0}")]
    Execution(String),
    #[error("proving error: {0}")]
    Proving(String),
    #[error("verification error: {0}")]
    Verification(String),
}

// ──────────────────────────── Shared wiring constraints ────────────────

/// Deposit wiring constraints shared between single-tx and batched evaluators.
///
/// Public inputs are passed as generic `E::F` values — the single-tx evaluator
/// converts struct constants via `E::F::from()`, while the batched evaluator
/// reads them from preprocessed columns.
pub fn constrain_deposit_wiring<E: EvalAtRow>(
    eval: &mut E,
    is_real: &E::F,
    perm0: &Poseidon2Columns<E::F>,
    perm1: &Poseidon2Columns<E::F>,
    sub_limbs: &[E::F; NUM_SUB_LIMBS],
    commitment: &[E::F; RATE],
    amount_lo: &E::F,
    amount_hi: &E::F,
    asset_id: &E::F,
) {
    // Domain separation: perm0.input[8] = 11 (commitment hash input length)
    eval.add_constraint(
        is_real.clone()
            * (perm0.input()[RATE].clone() - E::F::from(M31::from_u32_unchecked(11))),
    );

    // Zero padding: perm0.input[9..15] = 0
    for j in (RATE + 1)..STATE_WIDTH {
        eval.add_constraint(is_real.clone() * perm0.input()[j].clone());
    }

    // Sponge chain: perm1.input[3..15] = perm0.output[3..15]
    for j in 3..STATE_WIDTH {
        eval.add_constraint(
            is_real.clone()
                * (perm1.input()[j].clone() - perm0.output()[j].clone()),
        );
    }

    // Commitment output: perm1.output[0..7] = public commitment
    for j in 0..RATE {
        eval.add_constraint(
            is_real.clone()
                * (perm1.output()[j].clone() - commitment[j].clone()),
        );
    }

    // Amount binding: perm0.input[5] = amount_lo, [6] = amount_hi
    eval.add_constraint(
        is_real.clone()
            * (perm0.input()[5].clone() - amount_lo.clone()),
    );
    eval.add_constraint(
        is_real.clone()
            * (perm0.input()[6].clone() - amount_hi.clone()),
    );

    // Asset binding: perm0.input[4] = asset_id
    eval.add_constraint(
        is_real.clone()
            * (perm0.input()[4].clone() - asset_id.clone()),
    );

    // Sub-limb ↔ amount binding:
    // sub_limbs[0] + sub_limbs[1] * 65536 = amount_lo (mod p)
    // sub_limbs[2] + sub_limbs[3] * 65536 = amount_hi (mod p)
    let c65536 = M31::from_u32_unchecked(65536);
    eval.add_constraint(
        is_real.clone()
            * (sub_limbs[0].clone() + sub_limbs[1].clone() * c65536
                - amount_lo.clone()),
    );
    eval.add_constraint(
        is_real.clone()
            * (sub_limbs[2].clone() + sub_limbs[3].clone() * c65536
                - amount_hi.clone()),
    );
}

// ──────────────────────────── FrameworkEval ────────────────────────────

/// Evaluator for the deposit transaction STARK.
///
/// All Poseidon2 permutation and bit constraints apply to every row.
/// Wiring/public-input constraints are gated by `is_real` (row 0 only).
#[derive(Debug, Clone)]
pub struct DepositStarkEval {
    pub log_n_rows: u32,
    pub commitment: [M31; RATE],
    pub amount_lo: M31,
    pub amount_hi: M31,
    pub asset_id: M31,
}

impl FrameworkEval for DepositStarkEval {
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

        // ── Permutation round constraints (all rows) ──
        let perm0 = constrain_poseidon2_permutation(&mut eval);
        let perm1 = constrain_poseidon2_permutation(&mut eval);

        // ── Sub-limb columns (4) ──
        let sub_limbs: [E::F; NUM_SUB_LIMBS] =
            std::array::from_fn(|_| eval.next_trace_mask());

        // ── Bit columns (64 = 4 × 16) ──
        let bits: [[E::F; BITS_PER_LIMB]; NUM_SUB_LIMBS] =
            std::array::from_fn(|_| std::array::from_fn(|_| eval.next_trace_mask()));

        // ── Bit constraints: bi * (bi - 1) = 0 (all rows) ──
        let one = M31::from_u32_unchecked(1);
        for limb_bits in &bits {
            for bit in limb_bits {
                eval.add_constraint(bit.clone() * (bit.clone() - E::F::from(one)));
            }
        }

        // ── Reconstruction: Σ bi * 2^i = sub_limb (all rows) ──
        for (k, sub_limb) in sub_limbs.iter().enumerate() {
            let mut reconstructed = E::F::zero();
            for (i, bit) in bits[k].iter().enumerate() {
                let coeff = M31::from_u32_unchecked(1u32 << i);
                reconstructed += bit.clone() * coeff;
            }
            eval.add_constraint(reconstructed - sub_limb.clone());
        }

        // ── Wiring constraints via shared helper ──
        let commitment: [E::F; RATE] = std::array::from_fn(|j| E::F::from(self.commitment[j]));
        let amount_lo = E::F::from(self.amount_lo);
        let amount_hi = E::F::from(self.amount_hi);
        let asset_id = E::F::from(self.asset_id);

        constrain_deposit_wiring(
            &mut eval, &is_real,
            &perm0, &perm1,
            &sub_limbs,
            &commitment, &amount_lo, &amount_hi, &asset_id,
        );

        eval
    }
}

pub type DepositStarkComponent = FrameworkComponent<DepositStarkEval>;

// ──────────────────────────── Proof type ───────────────────────────────

#[derive(Debug)]
pub struct DepositStarkProof {
    pub stark_proof: StarkProof<Blake2sHash>,
    pub public_inputs: DepositPublicInputs,
}

// ──────────────────────────── Prover ──────────────────────────────────

/// Prove a deposit transaction with a STARK proof.
///
/// The verifier only sees the public inputs (commitment, amount, asset_id).
/// The trace (note contents, blinding) is hidden by the STARK.
pub fn prove_deposit_stark(
    witness: &DepositWitness,
) -> Result<DepositStarkProof, DepositStarkError> {
    #[cfg(feature = "cuda-runtime")]
    {
        if crate::backend::force_gpu() || crate::backend::gpu_is_available() {
            use stwo::prover::backend::gpu::GpuBackend;
            return prove_deposit_stark_with::<GpuBackend>(witness);
        }
    }
    prove_deposit_stark_with::<SimdBackend>(witness)
}

fn prove_deposit_stark_with<B>(
    witness: &DepositWitness,
) -> Result<DepositStarkProof, DepositStarkError>
where
    B: BackendForChannel<Blake2sMerkleChannel> + PolyOps + ColumnOps<M31>,
    Col<B, M31>: Column<M31>,
    <B as ColumnOps<M31>>::Column: 'static,
    FrameworkComponent<DepositStarkEval>: ComponentProver<B>,
{
    let (execution, public_inputs) = execute_deposit(witness)
        .map_err(|e| DepositStarkError::Execution(format!("{e}")))?;

    let table_size = 1usize << LOG_SIZE;

    // Compute permutation traces for the real row
    let perm_traces: Vec<_> = execution
        .all_permutation_inputs
        .iter()
        .map(compute_permutation_trace)
        .collect();

    // Dummy trace for padding rows
    let dummy = dummy_permutation_trace();

    // Build execution columns
    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let mut exec_cols: Vec<Col<B, M31>> = (0..TOTAL_EXEC_COLS)
        .map(|_| Col::<B, M31>::zeros(table_size))
        .collect();

    // Write permutation traces
    for (p, trace) in perm_traces.iter().enumerate() {
        write_permutation_to_trace::<B>(trace, &mut exec_cols, p * COLS_PER_PERM, 0);
    }
    for row in 1..table_size {
        for p in 0..NUM_PERMS {
            write_permutation_to_trace::<B>(&dummy, &mut exec_cols, p * COLS_PER_PERM, row);
        }
    }

    // Write sub-limbs (row 0 only, padding rows stay zero)
    let sub_offset = NUM_PERMS * COLS_PER_PERM;
    for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
        exec_cols[sub_offset + k].set(0, limb);
    }

    // Write bit decomposition (row 0 only)
    let bit_offset = sub_offset + NUM_SUB_LIMBS;
    for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
        let bits = decompose_to_bits(limb.0);
        for (i, &bit) in bits.iter().enumerate() {
            exec_cols[bit_offset + k * BITS_PER_LIMB + i].set(0, M31::from_u32_unchecked(bit));
        }
    }

    // Build preprocessed column (is_real)
    let mut is_real_col = Col::<B, M31>::zeros(table_size);
    is_real_col.set(0, M31::from_u32_unchecked(1));

    let preprocessed = vec![CircleEvaluation::<B, M31, BitReversedOrder>::new(
        domain, is_real_col,
    )];
    let execution_evals: Vec<CircleEvaluation<B, M31, BitReversedOrder>> = exec_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    // PCS setup
    let pcs_config = PcsConfig::default();
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(LOG_SIZE + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<B, Blake2sMerkleChannel>::new(pcs_config, &twiddles);

    // Tree 0: Preprocessed (is_real)
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<B, B, M31>(preprocessed));
    tree_builder.commit(channel);

    // Tree 1: Execution trace
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<B, B, M31>(
        execution_evals,
    ));
    tree_builder.commit(channel);

    // Build component (no LogUp → claimed_sum = 0)
    let eval = DepositStarkEval {
        log_n_rows: LOG_SIZE,
        commitment: public_inputs.commitment,
        amount_lo: M31::from_u32_unchecked((public_inputs.amount & 0x7FFFFFFF) as u32),
        amount_hi: M31::from_u32_unchecked((public_inputs.amount >> 31) as u32),
        asset_id: public_inputs.asset_id,
    };
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        eval,
        SecureField::zero(),
    );

    let stark_proof = prove::<B, Blake2sMerkleChannel>(
        &[&component],
        channel,
        commitment_scheme,
    )
    .map_err(|e| DepositStarkError::Proving(format!("{e:?}")))?;

    Ok(DepositStarkProof {
        stark_proof,
        public_inputs,
    })
}

// ──────────────────────────── Verifier ─────────────────────────────────

/// Verify a deposit STARK proof against the given public inputs.
pub fn verify_deposit_stark(
    proof: &DepositStarkProof,
    public_inputs: &DepositPublicInputs,
) -> Result<(), DepositStarkError> {
    let amount_lo = M31::from_u32_unchecked((public_inputs.amount & 0x7FFFFFFF) as u32);
    let amount_hi = M31::from_u32_unchecked((public_inputs.amount >> 31) as u32);
    let pcs_config = PcsConfig::default();

    // Build dummy component to get trace_log_degree_bounds
    let dummy_eval = DepositStarkEval {
        log_n_rows: LOG_SIZE,
        commitment: public_inputs.commitment,
        amount_lo,
        amount_hi,
        asset_id: public_inputs.asset_id,
    };
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component =
        FrameworkComponent::new(&mut allocator, dummy_eval, SecureField::zero());
    let bounds = Component::trace_log_degree_bounds(&dummy_component);

    // Set up channel and verifier
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme =
        CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);

    // Replay commitments
    commitment_scheme.commit(proof.stark_proof.commitments[0], &bounds[0], channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &bounds[1], channel);

    // Build real component (same eval — no lookup elements to draw)
    let real_eval = DepositStarkEval {
        log_n_rows: LOG_SIZE,
        commitment: public_inputs.commitment,
        amount_lo,
        amount_hi,
        asset_id: public_inputs.asset_id,
    };
    let mut allocator = TraceLocationAllocator::default();
    let component =
        FrameworkComponent::new(&mut allocator, real_eval, SecureField::zero());

    stwo_verify::<Blake2sMerkleChannel>(
        &[&component as &dyn Component],
        channel,
        &mut commitment_scheme,
        proof.stark_proof.clone(),
    )
    .map_err(|e| DepositStarkError::Verification(format!("{e:?}")))
}

// ──────────────────────────── Tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::commitment::{derive_pubkey, Note};

    fn make_deposit_witness(amount: u64) -> DepositWitness {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let pk = derive_pubkey(&sk);
        let amount_lo = M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32);
        let amount_hi = M31::from_u32_unchecked((amount >> 31) as u32);
        let asset_id = M31::from_u32_unchecked(0);
        let blinding = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let note = Note::new(pk, asset_id, amount_lo, amount_hi, blinding);
        DepositWitness {
            note,
            amount,
            asset_id,
        }
    }

    #[test]
    fn test_deposit_stark_prove_verify() {
        let witness = make_deposit_witness(1000);
        let proof = prove_deposit_stark(&witness).expect("proving should succeed");
        verify_deposit_stark(&proof, &proof.public_inputs).expect("verification should succeed");
    }

    #[test]
    fn test_deposit_stark_wrong_commitment_rejected() {
        let witness = make_deposit_witness(1000);
        let proof = prove_deposit_stark(&witness).expect("proving should succeed");

        let mut bad_inputs = proof.public_inputs.clone();
        bad_inputs.commitment[0] = M31::from_u32_unchecked(999999);

        let result = verify_deposit_stark(&proof, &bad_inputs);
        assert!(result.is_err(), "modified commitment should fail");
    }

    #[test]
    fn test_deposit_stark_wrong_amount_rejected() {
        let witness = make_deposit_witness(1000);
        let proof = prove_deposit_stark(&witness).expect("proving should succeed");

        let mut bad_inputs = proof.public_inputs.clone();
        bad_inputs.amount = 2000;

        let result = verify_deposit_stark(&proof, &bad_inputs);
        assert!(result.is_err(), "modified amount should fail");
    }

    #[test]
    fn test_deposit_stark_consistent_with_phase3() {
        use crate::circuits::deposit::{prove_deposit, verify_deposit};

        let witness = make_deposit_witness(500);

        // Phase 3 proof
        let phase3_proof = prove_deposit(&witness).expect("phase3 prove");
        verify_deposit(&phase3_proof).expect("phase3 verify");

        // Phase 4 STARK proof
        let stark_proof = prove_deposit_stark(&witness).expect("stark prove");
        verify_deposit_stark(&stark_proof, &stark_proof.public_inputs)
            .expect("stark verify");

        // Public inputs should match
        assert_eq!(
            phase3_proof.public_inputs.commitment,
            stark_proof.public_inputs.commitment
        );
        assert_eq!(
            phase3_proof.public_inputs.amount,
            stark_proof.public_inputs.amount
        );
    }

    #[test]
    fn test_deposit_stark_zero_amount() {
        let witness = make_deposit_witness(0);
        let proof = prove_deposit_stark(&witness).expect("proving should succeed");
        verify_deposit_stark(&proof, &proof.public_inputs).expect("verification should succeed");
    }

    #[test]
    fn test_deposit_stark_large_amount() {
        let amount = (1u64 << 40) + 42;
        let witness = make_deposit_witness(amount);
        let proof = prove_deposit_stark(&witness).expect("proving should succeed");
        verify_deposit_stark(&proof, &proof.public_inputs).expect("verification should succeed");
    }
}
