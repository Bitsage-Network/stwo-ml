//! Spend Transaction STARK: wraps the 2-in/2-out spend circuit in a STWO STARK proof.
//!
//! The most complex transaction STARK. For each of 2 inputs:
//!   ownership + commitment + nullifier + Merkle path verification.
//! For each of 2 outputs: commitment hash.
//! Plus balance conservation and asset consistency.
//!
//! The verifier only sees: merkle_root, 2 nullifiers, 2 output commitments.
//!
//! Trace layout (~41,867 columns):
//!   - Perms 0..63: 64 × 652 = 41,728 (54 real + 10 padding)
//!   - Sub-limbs: 8 columns (4 per output note)
//!   - Bit decomposition: 128 columns (8 × 16)
//!   - Carry columns: 3 (carry, carry_pos, carry_neg)
//!
//! Permutation index map:
//!   0:     Input 0 ownership
//!   1-2:   Input 0 commitment hash
//!   3-4:   Input 0 nullifier hash
//!   5-24:  Input 0 Merkle path (20 levels)
//!   25:    Input 1 ownership
//!   26-27: Input 1 commitment hash
//!   28-29: Input 1 nullifier hash
//!   30-49: Input 1 Merkle path (20 levels)
//!   50-51: Output 0 commitment hash
//!   52-53: Output 1 commitment hash
//!   54-63: Padding (dummy perms)

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
use crate::circuits::spend::{
    execute_spend, SpendPublicInputs, SpendWitness, MERKLE_DEPTH, SPEND_NUM_INPUTS,
    SPEND_NUM_OUTPUTS,
};
use crate::components::poseidon2_air::{
    compute_merkle_chain_padding, compute_permutation_trace, constrain_poseidon2_permutation,
    decompose_to_bits, dummy_permutation_trace, write_permutation_to_trace, Poseidon2Columns,
    COLS_PER_PERM,
};
use crate::crypto::poseidon2_m31::{RATE, STATE_WIDTH};

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

const PERMS_PER_INPUT: usize = 25; // 1 + 2 + 2 + 20
const PERMS_PER_OUTPUT: usize = 2;
const NUM_PERMS: usize = 64; // 54 real (2×25 input + 2×2 output) + 10 padding
const NUM_SUB_LIMBS: usize = SPEND_NUM_OUTPUTS * 4; // 8
const BITS_PER_LIMB: usize = 16;
const NUM_BIT_COLS: usize = NUM_SUB_LIMBS * BITS_PER_LIMB; // 128
const NUM_CARRY_COLS: usize = 3; // carry, carry_pos, carry_neg
const TOTAL_EXEC_COLS: usize =
    NUM_PERMS * COLS_PER_PERM + NUM_SUB_LIMBS + NUM_BIT_COLS + NUM_CARRY_COLS;
const LOG_SIZE: u32 = 4;

// Permutation index offsets for each input
pub fn input_ownership_perm(i: usize) -> usize {
    i * PERMS_PER_INPUT
}
pub fn input_commitment_start(i: usize) -> usize {
    i * PERMS_PER_INPUT + 1
}
pub fn input_nullifier_start(i: usize) -> usize {
    i * PERMS_PER_INPUT + 3
}
pub fn input_merkle_start(i: usize) -> usize {
    i * PERMS_PER_INPUT + 5
}
pub fn input_merkle_end(i: usize) -> usize {
    i * PERMS_PER_INPUT + 25
}
pub fn output_commitment_start(j: usize) -> usize {
    SPEND_NUM_INPUTS * PERMS_PER_INPUT + j * PERMS_PER_OUTPUT
}

#[derive(Debug, thiserror::Error)]
pub enum SpendStarkError {
    #[error("execution error: {0}")]
    Execution(String),
    #[error("proving error: {0}")]
    Proving(String),
    #[error("verification error: {0}")]
    Verification(String),
}

// ──────────────────────────── Shared wiring constraints ────────────────

/// Spend wiring constraints shared between single-tx and batched evaluators.
pub fn constrain_spend_wiring<E: EvalAtRow>(
    eval: &mut E,
    is_real: &E::F,
    perms: &[Poseidon2Columns<E::F>],
    sub_limbs: &[E::F; NUM_SUB_LIMBS],
    carry: &E::F,
    merkle_root: &[E::F; RATE],
    nullifiers: &[[E::F; RATE]; SPEND_NUM_INPUTS],
    output_commitments: &[[E::F; RATE]; SPEND_NUM_OUTPUTS],
) {
    // ── Per-input wiring constraints (× is_real) ──
    for inp_idx in 0..SPEND_NUM_INPUTS {
        let own = input_ownership_perm(inp_idx);
        let com_start = input_commitment_start(inp_idx);
        let com_end = com_start + 2;
        let nul_start = input_nullifier_start(inp_idx);
        let nul_end = nul_start + 2;
        let mrk_start = input_merkle_start(inp_idx);
        let mrk_end = input_merkle_end(inp_idx);

        // Ownership: domain tag + padding
        eval.add_constraint(
            is_real.clone()
                * (perms[own].input()[0].clone() - E::F::from(M31::from_u32_unchecked(0x766D3331))),
        );
        for j in 5..RATE {
            eval.add_constraint(is_real.clone() * perms[own].input()[j].clone());
        }
        eval.add_constraint(
            is_real.clone()
                * (perms[own].input()[RATE].clone() - E::F::from(M31::from_u32_unchecked(5))),
        );
        for j in (RATE + 1)..STATE_WIDTH {
            eval.add_constraint(is_real.clone() * perms[own].input()[j].clone());
        }

        // Ownership → Commitment
        eval.add_constraint(
            is_real.clone()
                * (perms[com_start].input()[RATE].clone()
                    - E::F::from(M31::from_u32_unchecked(11))),
        );
        for j in (RATE + 1)..STATE_WIDTH {
            eval.add_constraint(is_real.clone() * perms[com_start].input()[j].clone());
        }
        for j in 0..4 {
            eval.add_constraint(
                is_real.clone()
                    * (perms[com_start].input()[j].clone() - perms[own].output()[j].clone()),
            );
        }

        // Commitment sponge chain
        for j in 3..STATE_WIDTH {
            eval.add_constraint(
                is_real.clone()
                    * (perms[com_end - 1].input()[j].clone()
                        - perms[com_start].output()[j].clone()),
            );
        }

        // Nullifier hash: domain sep = 12
        eval.add_constraint(
            is_real.clone()
                * (perms[nul_start].input()[RATE].clone()
                    - E::F::from(M31::from_u32_unchecked(12))),
        );
        for j in (RATE + 1)..STATE_WIDTH {
            eval.add_constraint(is_real.clone() * perms[nul_start].input()[j].clone());
        }
        // S1: Nullifier sk must match ownership sk
        for j in 0..4 {
            eval.add_constraint(
                is_real.clone()
                    * (perms[nul_start].input()[j].clone() - perms[own].input()[j + 1].clone()),
            );
        }
        // Nullifier input[4..7] = commitment[0..3]
        for j in 4..RATE {
            eval.add_constraint(
                is_real.clone()
                    * (perms[nul_start].input()[j].clone()
                        - perms[com_end - 1].output()[j - 4].clone()),
            );
        }
        // Nullifier sponge chain
        for j in 4..STATE_WIDTH {
            eval.add_constraint(
                is_real.clone()
                    * (perms[nul_end - 1].input()[j].clone()
                        - perms[nul_start].output()[j].clone()),
            );
        }
        // S2: Nullifier chunk 2 absorption
        for j in 0..4 {
            eval.add_constraint(
                is_real.clone()
                    * (perms[nul_end - 1].input()[j].clone()
                        - perms[nul_start].output()[j].clone()
                        - perms[com_end - 1].output()[j + 4].clone()),
            );
        }

        // Nullifier output = public nullifier
        for j in 0..RATE {
            eval.add_constraint(
                is_real.clone()
                    * (perms[nul_end - 1].output()[j].clone() - nullifiers[inp_idx][j].clone()),
            );
        }

        // Merkle chain (degree 2, all rows)
        for j in 0..RATE {
            let commit_out = perms[com_end - 1].output()[j].clone();
            let curr_left = perms[mrk_start].input()[j].clone();
            let curr_right = perms[mrk_start].input()[j + RATE].clone();
            eval.add_constraint((curr_left - commit_out.clone()) * (curr_right - commit_out));
        }
        for i in (mrk_start + 1)..mrk_end {
            for j in 0..RATE {
                let prev_out = perms[i - 1].output()[j].clone();
                let curr_left = perms[i].input()[j].clone();
                let curr_right = perms[i].input()[j + RATE].clone();
                eval.add_constraint((curr_left - prev_out.clone()) * (curr_right - prev_out));
            }
        }

        // Merkle root = public root
        for j in 0..RATE {
            eval.add_constraint(
                is_real.clone() * (perms[mrk_end - 1].output()[j].clone() - merkle_root[j].clone()),
            );
        }
    }

    // ── Per-output wiring constraints (× is_real) ──
    for out_idx in 0..SPEND_NUM_OUTPUTS {
        let com_start = output_commitment_start(out_idx);
        let com_end = com_start + 2;

        eval.add_constraint(
            is_real.clone()
                * (perms[com_start].input()[RATE].clone()
                    - E::F::from(M31::from_u32_unchecked(11))),
        );
        for j in (RATE + 1)..STATE_WIDTH {
            eval.add_constraint(is_real.clone() * perms[com_start].input()[j].clone());
        }

        // Sponge chain
        for j in 3..STATE_WIDTH {
            eval.add_constraint(
                is_real.clone()
                    * (perms[com_end - 1].input()[j].clone()
                        - perms[com_start].output()[j].clone()),
            );
        }

        // Output commitment = public commitment
        for j in 0..RATE {
            eval.add_constraint(
                is_real.clone()
                    * (perms[com_end - 1].output()[j].clone()
                        - output_commitments[out_idx][j].clone()),
            );
        }

        // Sub-limb ↔ output amount binding
        let sub_base = out_idx * 4;
        let c65536 = M31::from_u32_unchecked(65536);
        eval.add_constraint(
            is_real.clone()
                * (sub_limbs[sub_base].clone() + sub_limbs[sub_base + 1].clone() * c65536
                    - perms[com_start].input()[5].clone()),
        );
        eval.add_constraint(
            is_real.clone()
                * (sub_limbs[sub_base + 2].clone() + sub_limbs[sub_base + 3].clone() * c65536
                    - perms[com_start].input()[6].clone()),
        );
    }

    // ── Balance conservation with carry (× is_real) ──
    let mut sum_in_lo = E::F::zero();
    let mut sum_in_hi = E::F::zero();
    for inp_idx in 0..SPEND_NUM_INPUTS {
        let com_start = input_commitment_start(inp_idx);
        sum_in_lo += perms[com_start].input()[5].clone();
        sum_in_hi += perms[com_start].input()[6].clone();
    }

    let mut sum_out_lo = E::F::zero();
    let mut sum_out_hi = E::F::zero();
    for out_idx in 0..SPEND_NUM_OUTPUTS {
        let com_start = output_commitment_start(out_idx);
        sum_out_lo += perms[com_start].input()[5].clone();
        sum_out_hi += perms[com_start].input()[6].clone();
    }

    eval.add_constraint(is_real.clone() * (sum_in_lo - sum_out_lo + carry.clone()));
    eval.add_constraint(is_real.clone() * (sum_in_hi - sum_out_hi - carry.clone()));

    // ── Asset consistency (× is_real) ──
    let reference_asset = perms[input_commitment_start(0)].input()[4].clone();
    for inp_idx in 1..SPEND_NUM_INPUTS {
        let com_start = input_commitment_start(inp_idx);
        eval.add_constraint(
            is_real.clone() * (perms[com_start].input()[4].clone() - reference_asset.clone()),
        );
    }
    for out_idx in 0..SPEND_NUM_OUTPUTS {
        let com_start = output_commitment_start(out_idx);
        eval.add_constraint(
            is_real.clone() * (perms[com_start].input()[4].clone() - reference_asset.clone()),
        );
    }
}

// ──────────────────────────── FrameworkEval ────────────────────────────

#[derive(Debug, Clone)]
pub struct SpendStarkEval {
    pub log_n_rows: u32,
    pub merkle_root: [M31; RATE],
    pub nullifiers: [[M31; RATE]; SPEND_NUM_INPUTS],
    pub output_commitments: [[M31; RATE]; SPEND_NUM_OUTPUTS],
}

impl FrameworkEval for SpendStarkEval {
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

        // ── Carry columns for cross-limb balance ──
        let carry = eval.next_trace_mask();
        let carry_pos = eval.next_trace_mask();
        let carry_neg = eval.next_trace_mask();

        // Carry range constraints (all rows): c ∈ {-1, 0, 1}
        eval.add_constraint(carry_pos.clone() * (carry_pos.clone() - E::F::from(one)));
        eval.add_constraint(carry_neg.clone() * (carry_neg.clone() - E::F::from(one)));
        eval.add_constraint(carry_neg.clone() * carry_pos.clone());
        eval.add_constraint(carry.clone() - carry_pos + carry_neg);

        // ── Wiring constraints via shared helper ──
        let merkle_root: [E::F; RATE] = std::array::from_fn(|j| E::F::from(self.merkle_root[j]));
        let nullifiers: [[E::F; RATE]; SPEND_NUM_INPUTS] =
            std::array::from_fn(|i| std::array::from_fn(|j| E::F::from(self.nullifiers[i][j])));
        let output_commitments: [[E::F; RATE]; SPEND_NUM_OUTPUTS] = std::array::from_fn(|i| {
            std::array::from_fn(|j| E::F::from(self.output_commitments[i][j]))
        });

        constrain_spend_wiring(
            &mut eval,
            &is_real,
            &perms,
            &sub_limbs,
            &carry,
            &merkle_root,
            &nullifiers,
            &output_commitments,
        );

        eval
    }
}

pub type SpendStarkComponent = FrameworkComponent<SpendStarkEval>;

// ──────────────────────────── Proof type ───────────────────────────────

#[derive(Debug)]
pub struct SpendStarkProof {
    pub stark_proof: StarkProof<Blake2sHash>,
    pub public_inputs: SpendPublicInputs,
}

// ──────────────────────────── Prover ──────────────────────────────────

pub fn prove_spend_stark(witness: &SpendWitness) -> Result<SpendStarkProof, SpendStarkError> {
    #[cfg(feature = "cuda-runtime")]
    {
        if crate::backend::force_gpu() || crate::backend::gpu_is_available() {
            use stwo::prover::backend::gpu::GpuBackend;
            return prove_spend_stark_with::<GpuBackend>(witness);
        }
    }
    prove_spend_stark_with::<SimdBackend>(witness)
}

fn prove_spend_stark_with<B>(witness: &SpendWitness) -> Result<SpendStarkProof, SpendStarkError>
where
    B: BackendForChannel<Blake2sMerkleChannel> + PolyOps + ColumnOps<M31>,
    Col<B, M31>: Column<M31>,
    <B as ColumnOps<M31>>::Column: 'static,
    FrameworkComponent<SpendStarkEval>: ComponentProver<B>,
{
    let (execution, public_inputs) =
        execute_spend(witness).map_err(|e| SpendStarkError::Execution(format!("{e}")))?;

    let table_size = 1usize << LOG_SIZE;

    let perm_traces: Vec<_> = execution
        .all_permutation_inputs
        .iter()
        .map(compute_permutation_trace)
        .collect();
    let dummy = dummy_permutation_trace();
    // Dummy output hash — used as starting point for Merkle chain padding
    let dummy_hash: [M31; RATE] = std::array::from_fn(|j| dummy.states[22][j]);
    // Compute chained Merkle padding traces for each input's Merkle region
    let merkle_levels = MERKLE_DEPTH;
    let merkle_padding = compute_merkle_chain_padding(&dummy_hash, merkle_levels);

    let domain = CanonicCoset::new(LOG_SIZE).circle_domain();
    let mut exec_cols: Vec<Col<B, M31>> = (0..TOTAL_EXEC_COLS)
        .map(|_| Col::<B, M31>::zeros(table_size))
        .collect();

    for (p, trace) in perm_traces.iter().enumerate() {
        write_permutation_to_trace::<B>(trace, &mut exec_cols, p * COLS_PER_PERM, 0);
    }
    // Padding rows: Merkle perms use chained traces, others use dummy
    for row in 1..table_size {
        for p in 0..NUM_PERMS {
            // Check if this perm slot is in any input's Merkle region
            let mut is_merkle = false;
            for inp_idx in 0..SPEND_NUM_INPUTS {
                let mrk_start = input_merkle_start(inp_idx);
                let mrk_end = input_merkle_end(inp_idx);
                if p >= mrk_start && p < mrk_end {
                    let merkle_idx = p - mrk_start;
                    write_permutation_to_trace::<B>(
                        &merkle_padding[merkle_idx],
                        &mut exec_cols,
                        p * COLS_PER_PERM,
                        row,
                    );
                    is_merkle = true;
                    break;
                }
            }
            if !is_merkle {
                write_permutation_to_trace::<B>(&dummy, &mut exec_cols, p * COLS_PER_PERM, row);
            }
        }
    }

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

    // Carry computation from integer amounts
    let carry_offset = bit_offset + NUM_BIT_COLS;
    let hi_in: i64 = witness
        .inputs
        .iter()
        .map(|i| i.note.amount_hi.0 as i64)
        .sum();
    let hi_out: i64 = witness
        .outputs
        .iter()
        .map(|o| o.note.amount_hi.0 as i64)
        .sum();
    let carry_val = hi_in - hi_out; // must be -1, 0, or 1
    debug_assert!(
        (-1..=1).contains(&carry_val),
        "carry out of range: {carry_val}"
    );
    let p = 0x7FFFFFFFu32;
    let carry_m31 = if carry_val >= 0 {
        M31::from_u32_unchecked(carry_val as u32)
    } else {
        M31::from_u32_unchecked(p - ((-carry_val) as u32))
    };
    exec_cols[carry_offset].set(0, carry_m31);
    exec_cols[carry_offset + 1].set(
        0,
        M31::from_u32_unchecked(if carry_val > 0 { 1 } else { 0 }),
    );
    exec_cols[carry_offset + 2].set(
        0,
        M31::from_u32_unchecked(if carry_val < 0 { 1 } else { 0 }),
    );
    // Padding rows: carry=0, carry_pos=0, carry_neg=0 (already zeroed by Col::zeros)

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

    let eval = SpendStarkEval {
        log_n_rows: LOG_SIZE,
        merkle_root: public_inputs.merkle_root,
        nullifiers: public_inputs.nullifiers,
        output_commitments: public_inputs.output_commitments,
    };
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        eval,
        SecureField::zero(),
    );

    let stark_proof = prove::<B, Blake2sMerkleChannel>(&[&component], channel, commitment_scheme)
        .map_err(|e| SpendStarkError::Proving(format!("{e:?}")))?;

    Ok(SpendStarkProof {
        stark_proof,
        public_inputs,
    })
}

// ──────────────────────────── Verifier ─────────────────────────────────

pub fn verify_spend_stark(
    proof: &SpendStarkProof,
    public_inputs: &SpendPublicInputs,
) -> Result<(), SpendStarkError> {
    let pcs_config = PcsConfig::default();

    let dummy_eval = SpendStarkEval {
        log_n_rows: LOG_SIZE,
        merkle_root: public_inputs.merkle_root,
        nullifiers: public_inputs.nullifiers,
        output_commitments: public_inputs.output_commitments,
    };
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component = FrameworkComponent::new(&mut allocator, dummy_eval, SecureField::zero());
    let bounds = Component::trace_log_degree_bounds(&dummy_component);

    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);

    commitment_scheme.commit(proof.stark_proof.commitments[0], &bounds[0], channel);
    commitment_scheme.commit(proof.stark_proof.commitments[1], &bounds[1], channel);

    let real_eval = SpendStarkEval {
        log_n_rows: LOG_SIZE,
        merkle_root: public_inputs.merkle_root,
        nullifiers: public_inputs.nullifiers,
        output_commitments: public_inputs.output_commitments,
    };
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(&mut allocator, real_eval, SecureField::zero());

    stwo_verify::<Blake2sMerkleChannel>(
        &[&component as &dyn Component],
        channel,
        &mut commitment_scheme,
        proof.stark_proof.clone(),
    )
    .map_err(|e| SpendStarkError::Verification(format!("{e:?}")))
}

// ──────────────────────────── Tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::spend::{InputNoteWitness, OutputNoteWitness};
    use crate::crypto::commitment::{derive_pubkey, Note};
    use crate::crypto::merkle_m31::PoseidonMerkleTreeM31;

    fn make_spend_witness(amounts_in: [u64; 2], amounts_out: [u64; 2]) -> SpendWitness {
        let sk1 = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let sk2 = [50, 60, 70, 80].map(M31::from_u32_unchecked);
        let pk1 = derive_pubkey(&sk1);
        let pk2 = derive_pubkey(&sk2);
        let asset_id = M31::from_u32_unchecked(0);

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

        let c1 = note1.commitment();
        let c2 = note2.commitment();
        let mut tree = PoseidonMerkleTreeM31::new(MERKLE_DEPTH);
        tree.append(c1);
        tree.append(c2);
        let merkle_root = tree.root();
        let path1 = tree.prove(0);
        let path2 = tree.prove(1);

        let out_pk1 = derive_pubkey(&[100, 200, 300, 400].map(M31::from_u32_unchecked));
        let out_pk2 = derive_pubkey(&[500, 600, 700, 800].map(M31::from_u32_unchecked));
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
    fn test_spend_stark_prove_verify() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let proof = prove_spend_stark(&witness).expect("proving should succeed");
        verify_spend_stark(&proof, &proof.public_inputs).expect("verification should succeed");
    }

    #[test]
    fn test_spend_stark_wrong_key_rejected() {
        let mut witness = make_spend_witness([1000, 2000], [1500, 1500]);
        witness.inputs[0].spending_key = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let result = prove_spend_stark(&witness);
        assert!(result.is_err(), "wrong key should fail at execution");
    }

    #[test]
    fn test_spend_stark_unbalanced_rejected() {
        let witness = make_spend_witness([1000, 2000], [1500, 2000]);
        let result = prove_spend_stark(&witness);
        assert!(result.is_err(), "unbalanced should fail at execution");
    }

    #[test]
    fn test_spend_stark_wrong_nullifier_rejected() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let proof = prove_spend_stark(&witness).expect("proving should succeed");

        let mut bad_inputs = proof.public_inputs.clone();
        bad_inputs.nullifiers[0][0] = M31::from_u32_unchecked(999999);

        let result = verify_spend_stark(&proof, &bad_inputs);
        assert!(result.is_err(), "modified nullifier should fail");
    }

    #[test]
    fn test_spend_stark_wrong_root_rejected() {
        let witness = make_spend_witness([1000, 2000], [1500, 1500]);
        let proof = prove_spend_stark(&witness).expect("proving should succeed");

        let mut bad_inputs = proof.public_inputs.clone();
        bad_inputs.merkle_root[0] = M31::from_u32_unchecked(999999);

        let result = verify_spend_stark(&proof, &bad_inputs);
        assert!(result.is_err(), "modified root should fail");
    }

    #[test]
    fn test_spend_stark_different_assets_rejected() {
        let mut witness = make_spend_witness([1000, 2000], [1500, 1500]);
        witness.inputs[1].note.asset_id = M31::from_u32_unchecked(1);
        let result = prove_spend_stark(&witness);
        assert!(result.is_err(), "different assets should fail at execution");
    }

    #[test]
    fn test_spend_stark_large_amounts() {
        // Use amounts with non-zero hi limbs that balance per-limb (carry=0).
        let a = 3 * (1u64 << 31) + 500; // lo=500, hi=3
        let b = 2 * (1u64 << 31) + 300; // lo=300, hi=2
        let c = 4 * (1u64 << 31) + 600; // lo=600, hi=4
        let d = 1 * (1u64 << 31) + 200; // lo=200, hi=1
        let witness = make_spend_witness([a, b], [c, d]);
        let proof = prove_spend_stark(&witness).expect("proving should succeed");
        verify_spend_stark(&proof, &proof.public_inputs).expect("verification should succeed");
    }

    #[test]
    fn test_spend_stark_carry_positive() {
        // carry = +1: hi_in_sum > hi_out_sum, lo overflow compensated by hi.
        // Input: (hi=1,lo=100) + (hi=1,lo=200) → hi_in=2, lo_in=300
        // Output: (hi=1,lo=1147483948) + (hi=0,lo=1000000000) → hi_out=1, lo_out=2147483948
        // carry = 2-1 = 1, D_lo = 300-2147483948 = -2^31, D_lo + 1*2^31 = 0 ✓
        let a = 1 * (1u64 << 31) + 100;
        let b = 1 * (1u64 << 31) + 200;
        let c = 1 * (1u64 << 31) + 1_147_483_948;
        let d = 1_000_000_000;
        assert_eq!(a + b, c + d);
        let witness = make_spend_witness([a, b], [c, d]);
        let proof = prove_spend_stark(&witness).expect("positive carry should succeed");
        verify_spend_stark(&proof, &proof.public_inputs)
            .expect("verification with positive carry should succeed");
    }

    #[test]
    fn test_spend_stark_carry_negative() {
        // carry = -1: hi_out_sum > hi_in_sum, lo_in compensates.
        // Input: (hi=1,lo=2B) + (hi=1,lo=1B) → hi_in=2, lo_in=3B
        // Output: (hi=2,lo=500M) + (hi=1,lo=352516352) → hi_out=3, lo_out=852516352
        // carry = 2-3 = -1, D_lo = 3B-852516352 = 2^31, D_lo + (-1)*2^31 = 0 ✓
        let a = 1 * (1u64 << 31) + 2_000_000_000;
        let b = 1 * (1u64 << 31) + 1_000_000_000;
        let c = 2 * (1u64 << 31) + 500_000_000;
        let d = 1 * (1u64 << 31) + 352_516_352;
        assert_eq!(a + b, c + d);
        let witness = make_spend_witness([a, b], [c, d]);
        let proof = prove_spend_stark(&witness).expect("negative carry should work");
        verify_spend_stark(&proof, &proof.public_inputs)
            .expect("verification with negative carry should succeed");
    }

    #[test]
    fn test_spend_stark_consistent_with_phase3() {
        use crate::circuits::spend::{prove_spend, verify_spend};

        let witness = make_spend_witness([500, 500], [300, 700]);

        let phase3_proof = prove_spend(&witness).expect("phase3 prove");
        verify_spend(&phase3_proof).expect("phase3 verify");

        let stark_proof = prove_spend_stark(&witness).expect("stark prove");
        verify_spend_stark(&stark_proof, &stark_proof.public_inputs).expect("stark verify");

        assert_eq!(
            phase3_proof.public_inputs.nullifiers,
            stark_proof.public_inputs.nullifiers
        );
        assert_eq!(
            phase3_proof.public_inputs.output_commitments,
            stark_proof.public_inputs.output_commitments
        );
    }
}
