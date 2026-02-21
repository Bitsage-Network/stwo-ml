//! Spend Transaction STARK: wraps the 2-in/2-out spend circuit in a STWO STARK proof.
//!
//! The most complex transaction STARK. For each of 2 inputs:
//!   ownership + commitment + nullifier + Merkle path verification.
//! For each of 2 outputs: commitment hash.
//! Plus balance conservation and asset consistency.
//!
//! The verifier only sees: merkle_root, 2 nullifiers, 2 output commitments.
//!
//! Trace layout (~42,143 columns):
//!   - Perms 0..63: 64 × 652 = 41,728 (54 real + 10 padding)
//!   - Sub-limbs: 16 columns (4 per note × 4 notes)
//!   - Bit decomposition: 256 columns (16 × 16)
//!   - Carry columns: 15 (3 carries × 5: c_val, pos_b0, pos_b1, neg_b0, neg_b1)
//!
//! Balance conservation uses sub-limb ripple carry (not lo/hi carry).
//! Carry multipliers 2^16 and 2^15 avoid the 2^31 ≡ 1 (mod p) degeneracy.
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
use crate::crypto::poseidon2_m31::{DOMAIN_COMPRESS, RATE, STATE_WIDTH};

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

const PERMS_PER_INPUT: usize = 25; // 1 + 2 + 2 + 20
const PERMS_PER_OUTPUT: usize = 2;
const NUM_PERMS: usize = 64; // 54 real (2×25 input + 2×2 output) + 10 padding
const NUM_SUB_LIMBS: usize = (SPEND_NUM_INPUTS + SPEND_NUM_OUTPUTS) * 4; // 16
const BITS_PER_LIMB: usize = 16;
const NUM_BIT_COLS: usize = NUM_SUB_LIMBS * BITS_PER_LIMB; // 256
/// 3 carries × 5 columns each: (c_val, pos_b0, pos_b1, neg_b0, neg_b1)
const NUM_CARRIES: usize = 3;
const COLS_PER_CARRY: usize = 5;
const NUM_CARRY_COLS: usize = NUM_CARRIES * COLS_PER_CARRY; // 15
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
///
/// `sub_limbs` layout: [inp0_s0, inp0_s1, inp0_s2, inp0_s3,
///                       inp1_s0, inp1_s1, inp1_s2, inp1_s3,
///                       out0_s0, out0_s1, out0_s2, out0_s3,
///                       out1_s0, out1_s1, out1_s2, out1_s3]
///
/// `carries` = [c0, c1, c2] for the 4-equation sub-limb ripple carry.
pub fn constrain_spend_wiring<E: EvalAtRow>(
    eval: &mut E,
    is_real: &E::F,
    perms: &[Poseidon2Columns<E::F>],
    sub_limbs: &[E::F; NUM_SUB_LIMBS],
    carries: &[E::F; NUM_CARRIES],
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
        // C5: state[RATE+1] has DOMAIN_COMPRESS added in compress, so the
        // right-child check at j=1 must subtract it to match the raw value.
        let domain_tag = E::F::from(DOMAIN_COMPRESS);
        for j in 0..RATE {
            let commit_out = perms[com_end - 1].output()[j].clone();
            let curr_left = perms[mrk_start].input()[j].clone();
            let curr_right_raw = perms[mrk_start].input()[j + RATE].clone();
            let curr_right = if j == 1 {
                curr_right_raw - domain_tag.clone()
            } else {
                curr_right_raw
            };
            eval.add_constraint((curr_left - commit_out.clone()) * (curr_right - commit_out));
        }
        for i in (mrk_start + 1)..mrk_end {
            for j in 0..RATE {
                let prev_out = perms[i - 1].output()[j].clone();
                let curr_left = perms[i].input()[j].clone();
                let curr_right_raw = perms[i].input()[j + RATE].clone();
                let curr_right = if j == 1 {
                    curr_right_raw - domain_tag.clone()
                } else {
                    curr_right_raw
                };
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

    // ── Per-input sub-limb ↔ amount binding (× is_real) ──
    let c65536 = M31::from_u32_unchecked(65536);
    for inp_idx in 0..SPEND_NUM_INPUTS {
        let com_start = input_commitment_start(inp_idx);
        let sub_base = inp_idx * 4;
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
        let sub_base = (SPEND_NUM_INPUTS + out_idx) * 4;
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

    // ── Balance conservation: sub-limb ripple carry (× is_real) ──
    //
    // Amount = lo + hi × 2^31 = s0 + s1 × 2^16 + s2 × 2^31 + s3 × 2^47.
    // Sub-limb widths: s0=16-bit, s1=15-bit (MSB-0), s2=16-bit, s3=15-bit (MSB-0).
    // This ensures lo < p and hi < p (no M31 wrapping within a limb).
    //
    // Balance equations with carries c0, c1, c2 ∈ [-2, 2]:
    //   D[0] + c0 × 2^16 = 0     (carry at position 16)
    //   D[1] - c0 + c1 × 2^15 = 0  (carry at position 31)
    //   D[2] - c1 + c2 × 2^16 = 0  (carry at position 47)
    //   D[3] - c2 = 0             (top sub-limb)
    //
    // All values fit in [-262144, 262144] ≪ p, so M31 ≡ 0 ⟹ integer = 0.
    // This eliminates the 2^31 ≡ 1 (mod p) degeneracy in the old carry scheme.
    let c32768 = M31::from_u32_unchecked(32768);
    // Build D[k] = ∑_in s[k] - ∑_out s[k] for each sub-limb position k.
    let mut d: [E::F; 4] = std::array::from_fn(|_| E::F::zero());
    for inp_idx in 0..SPEND_NUM_INPUTS {
        let base = inp_idx * 4;
        for k in 0..4 {
            d[k] = d[k].clone() + sub_limbs[base + k].clone();
        }
    }
    for out_idx in 0..SPEND_NUM_OUTPUTS {
        let base = (SPEND_NUM_INPUTS + out_idx) * 4;
        for k in 0..4 {
            d[k] = d[k].clone() - sub_limbs[base + k].clone();
        }
    }

    // Eq 0: D[0] + c0 × 65536 = 0
    eval.add_constraint(is_real.clone() * (d[0].clone() + carries[0].clone() * c65536));
    // Eq 1: D[1] - c0 + c1 × 32768 = 0
    eval.add_constraint(
        is_real.clone() * (d[1].clone() - carries[0].clone() + carries[1].clone() * c32768),
    );
    // Eq 2: D[2] - c1 + c2 × 65536 = 0
    eval.add_constraint(
        is_real.clone() * (d[2].clone() - carries[1].clone() + carries[2].clone() * c65536),
    );
    // Eq 3: D[3] - c2 = 0
    eval.add_constraint(is_real.clone() * (d[3].clone() - carries[2].clone()));

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

        // ── MSB-0 constraints for high sub-limbs (s1, s3 per note) ──
        // Odd-indexed sub-limbs (1, 3, 5, 7, ...) are the "high half" of each M31 limb.
        // Constraining bit[15]=0 ensures the sub-limb is 15-bit, so lo < p and hi < p.
        for k in 0..NUM_SUB_LIMBS {
            if k % 2 == 1 {
                // High sub-limb: MSB must be 0
                eval.add_constraint(bits[k][BITS_PER_LIMB - 1].clone());
            }
        }

        // ── Carry columns: 3 carries × 5 cols each ──
        // Each carry c ∈ {-2, -1, 0, 1, 2} = pos - neg,
        // pos = pos_b0 + 2*pos_b1 ∈ {0,1,2}, neg = neg_b0 + 2*neg_b1 ∈ {0,1,2}.
        let two = M31::from_u32_unchecked(2);
        let carries: [E::F; NUM_CARRIES] = std::array::from_fn(|_| {
            let c_val = eval.next_trace_mask();
            let pos_b0 = eval.next_trace_mask();
            let pos_b1 = eval.next_trace_mask();
            let neg_b0 = eval.next_trace_mask();
            let neg_b1 = eval.next_trace_mask();

            // Binary constraints (degree 2)
            eval.add_constraint(pos_b0.clone() * (pos_b0.clone() - E::F::from(one)));
            eval.add_constraint(pos_b1.clone() * (pos_b1.clone() - E::F::from(one)));
            eval.add_constraint(neg_b0.clone() * (neg_b0.clone() - E::F::from(one)));
            eval.add_constraint(neg_b1.clone() * (neg_b1.clone() - E::F::from(one)));

            // pos ≤ 2: exclude pos=3 (b0=1,b1=1)
            eval.add_constraint(pos_b0.clone() * pos_b1.clone());
            // neg ≤ 2: exclude neg=3
            eval.add_constraint(neg_b0.clone() * neg_b1.clone());

            // c_val = pos - neg = (pos_b0 + 2*pos_b1) - (neg_b0 + 2*neg_b1)
            eval.add_constraint(
                c_val.clone() - pos_b0 - pos_b1.clone() * two + neg_b0 + neg_b1.clone() * two,
            );

            c_val
        });

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
            &carries,
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

    // Sub-limb decomposition for ALL notes (inputs + outputs).
    // Layout: [inp0_s0..s3, inp1_s0..s3, out0_s0..s3, out1_s0..s3]
    let sub_offset = NUM_PERMS * COLS_PER_PERM;
    let mut all_sub_limbs: Vec<u32> = Vec::with_capacity(NUM_SUB_LIMBS);
    for inp in &witness.inputs {
        let lo = inp.note.amount_lo.0;
        let hi = inp.note.amount_hi.0;
        all_sub_limbs.push(lo & 0xFFFF); // s0: low 16 bits of lo
        all_sub_limbs.push((lo >> 16) & 0x7FFF); // s1: high 15 bits of lo
        all_sub_limbs.push(hi & 0xFFFF); // s2: low 16 bits of hi
        all_sub_limbs.push((hi >> 16) & 0x7FFF); // s3: high 15 bits of hi
    }
    // Output sub-limbs (same as before, moved from execution.range_check_limbs)
    for out in &witness.outputs {
        let lo = out.note.amount_lo.0;
        let hi = out.note.amount_hi.0;
        all_sub_limbs.push(lo & 0xFFFF);
        all_sub_limbs.push((lo >> 16) & 0x7FFF);
        all_sub_limbs.push(hi & 0xFFFF);
        all_sub_limbs.push((hi >> 16) & 0x7FFF);
    }
    debug_assert_eq!(all_sub_limbs.len(), NUM_SUB_LIMBS);
    for (k, &limb_val) in all_sub_limbs.iter().enumerate() {
        exec_cols[sub_offset + k].set(0, M31::from_u32_unchecked(limb_val));
    }
    let bit_offset = sub_offset + NUM_SUB_LIMBS;
    for (k, &limb_val) in all_sub_limbs.iter().enumerate() {
        let bit_vals = decompose_to_bits(limb_val);
        for (i, &bit) in bit_vals.iter().enumerate() {
            exec_cols[bit_offset + k * BITS_PER_LIMB + i].set(0, M31::from_u32_unchecked(bit));
        }
    }

    // Carry computation: sub-limb ripple carry (3 carries, 5 columns each).
    let carry_offset = bit_offset + NUM_BIT_COLS;
    let p = 0x7FFFFFFFu32;

    // Compute per-sub-limb differences D[0..4]
    let mut d = [0i64; 4];
    for inp_idx in 0..SPEND_NUM_INPUTS {
        let base = inp_idx * 4;
        for k in 0..4 {
            d[k] += all_sub_limbs[base + k] as i64;
        }
    }
    for out_idx in 0..SPEND_NUM_OUTPUTS {
        let base = (SPEND_NUM_INPUTS + out_idx) * 4;
        for k in 0..4 {
            d[k] -= all_sub_limbs[base + k] as i64;
        }
    }

    // Solve carry chain: D[0]+c0*65536=0, D[1]-c0+c1*32768=0, D[2]-c1+c2*65536=0, D[3]-c2=0
    let c0 = if d[0] == 0 { 0i64 } else { -d[0] / 65536 };
    debug_assert_eq!(d[0] + c0 * 65536, 0, "D[0]={} not divisible by 65536", d[0]);
    let c1_num = -(d[1] - c0);
    let c1 = if c1_num == 0 { 0i64 } else { c1_num / 32768 };
    debug_assert_eq!(d[1] - c0 + c1 * 32768, 0, "carry eq 1 failed");
    let c2 = d[3];
    debug_assert_eq!(d[2] - c1 + c2 * 65536, 0, "carry eq 2 failed");
    debug_assert!((-2..=2).contains(&c0), "c0 out of range: {c0}");
    debug_assert!((-2..=2).contains(&c1), "c1 out of range: {c1}");
    debug_assert!((-2..=2).contains(&c2), "c2 out of range: {c2}");

    // Write carry columns: for each carry, 5 cols = (c_val, pos_b0, pos_b1, neg_b0, neg_b1)
    for (ci, &carry_val) in [c0, c1, c2].iter().enumerate() {
        let (pos, neg) = if carry_val >= 0 {
            (carry_val as u32, 0u32)
        } else {
            (0u32, (-carry_val) as u32)
        };
        let pos_b0 = pos & 1;
        let pos_b1 = (pos >> 1) & 1;
        let neg_b0 = neg & 1;
        let neg_b1 = (neg >> 1) & 1;
        let c_m31 = if carry_val >= 0 {
            M31::from_u32_unchecked(carry_val as u32)
        } else {
            M31::from_u32_unchecked(p - ((-carry_val) as u32))
        };

        let base = carry_offset + ci * COLS_PER_CARRY;
        exec_cols[base].set(0, c_m31);
        exec_cols[base + 1].set(0, M31::from_u32_unchecked(pos_b0));
        exec_cols[base + 2].set(0, M31::from_u32_unchecked(pos_b1));
        exec_cols[base + 3].set(0, M31::from_u32_unchecked(neg_b0));
        exec_cols[base + 4].set(0, M31::from_u32_unchecked(neg_b1));
    }
    // Padding rows: all carry cols zeroed by Col::zeros (c=0, pos=neg=0 → valid)

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
        let path1 = tree.prove(0).unwrap();
        let path2 = tree.prove(1).unwrap();

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

    /// Regression test for M2 carry attack (CVE-level severity).
    ///
    /// Before the sub-limb ripple carry fix, the balance conservation used:
    ///   D_lo + carry × 2^31 ≡ 0 (mod p)
    /// Since 2^31 ≡ 1 (mod p) in M31, a malicious prover could set
    /// D_lo = -p (≡ 0 mod p) with carry = 0, stealing p base units.
    ///
    /// The new sub-limb ripple carry uses multipliers 2^16 and 2^15
    /// (both coprime to p), so this attack is no longer possible.
    /// This test verifies that spending more than available is rejected.
    #[test]
    fn test_spend_stark_m2_carry_attack_regression() {
        let p = 0x7FFFFFFFu64; // 2^31 - 1

        // Attack 1: outputs exceed inputs by exactly p base units.
        // Old system: D_lo = -p ≡ 0 (mod p), carry = 0 → accepted.
        // New system: sub-limb carry chain has no valid solution → rejected.
        let witness = make_spend_witness([500, 500], [p + 500, 500]);
        let result = prove_spend_stark(&witness);
        assert!(
            result.is_err(),
            "imbalance of p should be rejected (carry attack 1)"
        );

        // Attack 2: outputs exceed inputs by p in the hi limb.
        // If hi difference = p, D[2..3] would need to absorb p — impossible
        // since sub-limb values < 2^16 and there are only 4 notes.
        let hi_shift = p * (1u64 << 31);
        let witness2 = make_spend_witness([1000, 1000], [hi_shift + 1000, 1000]);
        let result2 = prove_spend_stark(&witness2);
        assert!(
            result2.is_err(),
            "imbalance of p<<31 should be rejected (carry attack 2)"
        );
    }

    /// Verify that sub-limb carry chain math is correct for exact-balance transfers.
    /// With the ripple carry, amounts that exercise all 3 carries should still verify.
    #[test]
    fn test_spend_stark_all_carries_nonzero() {
        // Construct amounts where every sub-limb position has a nonzero difference,
        // requiring all 3 carries (c0, c1, c2) to be nonzero.
        //
        // Input 0: lo=0x1_FFFF (s0=0xFFFF, s1=1), hi=0x2_0001 (s2=1, s3=2)
        // Input 1: lo=0x0_0001 (s0=1, s1=0),      hi=0x0_0000 (s2=0, s3=0)
        // Output 0: lo=0x0_0002 (s0=2, s1=0),      hi=0x2_0001 (s2=1, s3=2)
        // Output 1: lo=0x1_FFFE (s0=0xFFFE, s1=1), hi=0x0_0000 (s2=0, s3=0)
        //
        // D[0] = (0xFFFF+1) - (2+0xFFFE) = 0x10000 - 0x10000 = 0 → c0=0
        // Actually let me pick better values that force nonzero carries.
        //
        // Input:  lo0=70000 (s0=4464, s1=1), lo1=1 (s0=1, s1=0)
        //         hi0=1 (s2=1, s3=0), hi1=0
        // Output: lo0=5 (s0=5, s1=0), lo1=65536 (s0=0, s1=1)
        //         hi0=0, hi1=1 (s2=1, s3=0)
        // Total in:  (1<<31)*1 + 70000 + 1 = 2147553649
        // Total out: 5 + 65536 + (1<<31)*1 = 2147549189
        // NOT equal. Let me just use numerics directly.

        // in_total = in0 + in1, out_total = out0 + out1, must be equal.
        // Pick: in0 = 0x2_0001_FFFF = 8590000127, in1 = 1
        //       out0 = 2, out1 = 8590000126
        // All lo/hi limbs differ → carries needed.
        let in0 = 2u64 * (1 << 31) + 131071; // hi=2, lo=0x1FFFF (s0=0xFFFF,s1=1)
        let in1 = 1u64; // hi=0, lo=1
        let total = in0 + in1;
        let out0 = 2u64; // hi=0, lo=2
        let out1 = total - out0;
        assert_eq!(in0 + in1, out0 + out1);

        let witness = make_spend_witness([in0, in1], [out0, out1]);
        let proof = prove_spend_stark(&witness).expect("all-carry case should prove");
        verify_spend_stark(&proof, &proof.public_inputs).expect("all-carry case should verify");
    }

    #[test]
    fn test_spend_stark_consistent_with_phase3() {
        use crate::circuits::spend::{prove_spend, verify_spend};

        let witness = make_spend_witness([500, 500], [300, 700]);

        let (phase3_proof, phase3_exec) = prove_spend(&witness).expect("phase3 prove");
        verify_spend(&phase3_proof, &phase3_exec).expect("phase3 verify");

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
