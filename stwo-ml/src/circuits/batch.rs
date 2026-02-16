//! Privacy Transaction Batch Proving: aggregates N transactions into a single STARK proof.
//!
//! Individual transaction STARKs each produce a separate proof. On-chain, each proof
//! requires ~280K gas to verify. Batch proving aggregates N transactions into one proof,
//! amortizing verification cost. At batch size 1000, per-tx drops to ~280 gas.
//!
//! Architecture: multi-row batching per transaction type + multi-component composition.
//! Each row in a component = one transaction. Different types can have different log_sizes.
//! Empty types are skipped.
//!
//! Key design: per-row public inputs live in preprocessed columns (Tree 0).
//! Fiat-Shamir binding: hash all public inputs into channel before tree commitments.

use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::m31::BaseField as M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::verifier::verify as stwo_verify;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
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
use crate::circuits::withdraw::{WithdrawPublicInputs, WithdrawWitness, execute_withdraw, MERKLE_DEPTH};
use crate::circuits::spend::{
    SpendPublicInputs, SpendWitness, execute_spend,
    SPEND_NUM_INPUTS, SPEND_NUM_OUTPUTS,
};
use crate::circuits::stark_deposit::{constrain_deposit_wiring, NUM_SUB_LIMBS as DEP_NUM_SUB_LIMBS};
use crate::circuits::stark_withdraw::constrain_withdraw_wiring;
use crate::circuits::stark_spend::constrain_spend_wiring;
use crate::components::poseidon2_air::{
    constrain_poseidon2_permutation, compute_merkle_chain_padding, compute_permutation_trace,
    decompose_to_bits, dummy_permutation_trace, write_permutation_to_trace,
    COLS_PER_PERM,
};
use crate::crypto::poseidon2_m31::{poseidon2_hash, RATE};

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

// ──────────────────────── Constants ──────────────────────────────────

const MIN_LOG_SIZE: u32 = 4; // STWO minimum

// Deposit constants
const DEP_NUM_PERMS: usize = 2;
const DEP_BITS_PER_LIMB: usize = 16;
const DEP_NUM_BIT_COLS: usize = DEP_NUM_SUB_LIMBS * DEP_BITS_PER_LIMB;
const DEP_EXEC_COLS: usize = DEP_NUM_PERMS * COLS_PER_PERM + DEP_NUM_SUB_LIMBS + DEP_NUM_BIT_COLS;
// Preprocessed: is_real(1) + commitment(8) + amount_lo(1) + amount_hi(1) + asset(1) = 12
const DEP_PREPROC_COLS: usize = 12;

// Withdraw constants
const WDR_NUM_PERMS: usize = 32;
const WDR_NUM_SUB_LIMBS: usize = 4;
const WDR_BITS_PER_LIMB: usize = 16;
const WDR_NUM_BIT_COLS: usize = WDR_NUM_SUB_LIMBS * WDR_BITS_PER_LIMB;
const WDR_EXEC_COLS: usize = WDR_NUM_PERMS * COLS_PER_PERM + WDR_NUM_SUB_LIMBS + WDR_NUM_BIT_COLS;
// Preprocessed: is_real(1) + merkle_root(8) + nullifier(8) + amount_lo(1) + amount_hi(1) + asset(1) = 20
const WDR_PREPROC_COLS: usize = 20;

// Spend constants
const SPD_NUM_PERMS: usize = 64;
const SPD_NUM_SUB_LIMBS: usize = SPEND_NUM_OUTPUTS * 4; // 8
const SPD_BITS_PER_LIMB: usize = 16;
const SPD_NUM_BIT_COLS: usize = SPD_NUM_SUB_LIMBS * SPD_BITS_PER_LIMB;
const SPD_NUM_CARRY_COLS: usize = 3;
const SPD_EXEC_COLS: usize = SPD_NUM_PERMS * COLS_PER_PERM + SPD_NUM_SUB_LIMBS + SPD_NUM_BIT_COLS + SPD_NUM_CARRY_COLS;
// Preprocessed: is_real(1) + merkle_root(8) + nullifiers(2×8=16) + output_commitments(2×8=16) = 41
const SPD_PREPROC_COLS: usize = 41;

// ──────────────────────── Data structures ────────────────────────────

/// A batch of privacy transactions to be proved in a single STARK.
pub struct PrivacyBatch {
    pub deposits: Vec<DepositWitness>,
    pub withdrawals: Vec<WithdrawWitness>,
    pub spends: Vec<SpendWitness>,
}

/// Collected public inputs for the batch.
#[derive(Clone, Debug)]
pub struct BatchPublicInputs {
    pub deposits: Vec<DepositPublicInputs>,
    pub withdrawals: Vec<WithdrawPublicInputs>,
    pub spends: Vec<SpendPublicInputs>,
}

/// A single STARK proof covering all transactions in the batch.
#[derive(Debug)]
pub struct PrivacyBatchProof {
    pub stark_proof: StarkProof<Blake2sHash>,
    pub public_inputs: BatchPublicInputs,
}

#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    #[error("empty batch")]
    EmptyBatch,
    #[error("deposit execution error: {0}")]
    DepositExecution(String),
    #[error("withdraw execution error: {0}")]
    WithdrawExecution(String),
    #[error("spend execution error: {0}")]
    SpendExecution(String),
    #[error("proving error: {0}")]
    Proving(String),
    #[error("verification error: {0}")]
    Verification(String),
}

// ──────────────────────── Utility functions ──────────────────────────

fn batch_log_size(count: usize) -> u32 {
    if count <= 1 {
        return MIN_LOG_SIZE;
    }
    let log = (count as f64).log2().ceil() as u32;
    log.max(MIN_LOG_SIZE)
}

/// Hash all public inputs into a deterministic sequence of M31 elements
/// for Fiat-Shamir binding.
fn hash_batch_public_inputs(inputs: &BatchPublicInputs) -> Vec<M31> {
    let mut data: Vec<M31> = Vec::new();

    // Encode counts
    data.push(M31::from_u32_unchecked(inputs.deposits.len() as u32));
    data.push(M31::from_u32_unchecked(inputs.withdrawals.len() as u32));
    data.push(M31::from_u32_unchecked(inputs.spends.len() as u32));

    for dep in &inputs.deposits {
        data.extend_from_slice(&dep.commitment);
        data.push(M31::from_u32_unchecked((dep.amount & 0x7FFFFFFF) as u32));
        data.push(M31::from_u32_unchecked((dep.amount >> 31) as u32));
        data.push(dep.asset_id);
    }

    for wdr in &inputs.withdrawals {
        data.extend_from_slice(&wdr.merkle_root);
        data.extend_from_slice(&wdr.nullifier);
        data.push(wdr.amount_lo);
        data.push(wdr.amount_hi);
        data.push(wdr.asset_id);
        data.extend_from_slice(&wdr.withdrawal_binding);
    }

    for spd in &inputs.spends {
        data.extend_from_slice(&spd.merkle_root);
        for nul in &spd.nullifiers {
            data.extend_from_slice(nul);
        }
        for com in &spd.output_commitments {
            data.extend_from_slice(com);
        }
    }

    // Hash the collected data using Poseidon2
    poseidon2_hash(&data).to_vec()
}

fn mix_public_inputs_into_channel(
    channel: &mut <Blake2sMerkleChannel as MerkleChannel>::C,
    inputs: &BatchPublicInputs,
) {
    let hash = hash_batch_public_inputs(inputs);
    let felts: Vec<SecureField> = hash
        .iter()
        .map(|&m| SecureField::from(m))
        .collect();
    channel.mix_felts(&felts);
}

// ──────────────────────── ComponentProverErased ──────────────────────

trait ComponentProverErased {
    fn as_component_prover(&self) -> &dyn ComponentProver<SimdBackend>;
}

impl<E> ComponentProverErased for FrameworkComponent<E>
where
    E: FrameworkEval,
    FrameworkComponent<E>: ComponentProver<SimdBackend>,
{
    fn as_component_prover(&self) -> &dyn ComponentProver<SimdBackend> {
        self
    }
}

// ──────────────────────── Batched Deposit Eval ──────────────────────

#[derive(Debug, Clone)]
pub struct BatchedDepositEval {
    pub log_n_rows: u32,
}

impl FrameworkEval for BatchedDepositEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Read preprocessed columns: is_real, commitment[8], amount_lo, amount_hi, asset_id
        let is_real = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "batch_dep_is_real".into(),
        });
        let commitment: [E::F; RATE] = std::array::from_fn(|j| {
            eval.get_preprocessed_column(PreProcessedColumnId {
                id: format!("batch_dep_commitment_{j}").into(),
            })
        });
        let amount_lo = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "batch_dep_amount_lo".into(),
        });
        let amount_hi = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "batch_dep_amount_hi".into(),
        });
        let asset_id = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "batch_dep_asset_id".into(),
        });

        // Permutation round constraints (all rows)
        let perm0 = constrain_poseidon2_permutation(&mut eval);
        let perm1 = constrain_poseidon2_permutation(&mut eval);

        // Sub-limb & bit columns
        let sub_limbs: [E::F; DEP_NUM_SUB_LIMBS] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let bits: [[E::F; DEP_BITS_PER_LIMB]; DEP_NUM_SUB_LIMBS] =
            std::array::from_fn(|_| std::array::from_fn(|_| eval.next_trace_mask()));

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

        constrain_deposit_wiring(
            &mut eval, &is_real,
            &perm0, &perm1,
            &sub_limbs,
            &commitment, &amount_lo, &amount_hi, &asset_id,
        );

        eval
    }
}

// ──────────────────────── Batched Withdraw Eval ─────────────────────

#[derive(Debug, Clone)]
pub struct BatchedWithdrawEval {
    pub log_n_rows: u32,
}

impl FrameworkEval for BatchedWithdrawEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_real = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "batch_wdr_is_real".into(),
        });
        let merkle_root: [E::F; RATE] = std::array::from_fn(|j| {
            eval.get_preprocessed_column(PreProcessedColumnId {
                id: format!("batch_wdr_merkle_root_{j}").into(),
            })
        });
        let nullifier: [E::F; RATE] = std::array::from_fn(|j| {
            eval.get_preprocessed_column(PreProcessedColumnId {
                id: format!("batch_wdr_nullifier_{j}").into(),
            })
        });
        let amount_lo = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "batch_wdr_amount_lo".into(),
        });
        let amount_hi = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "batch_wdr_amount_hi".into(),
        });
        let asset_id = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "batch_wdr_asset_id".into(),
        });

        // Permutation round constraints (all rows)
        let perms: Vec<_> = (0..WDR_NUM_PERMS)
            .map(|_| constrain_poseidon2_permutation(&mut eval))
            .collect();

        let sub_limbs: [E::F; WDR_NUM_SUB_LIMBS] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let bits: [[E::F; WDR_BITS_PER_LIMB]; WDR_NUM_SUB_LIMBS] =
            std::array::from_fn(|_| std::array::from_fn(|_| eval.next_trace_mask()));

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

        constrain_withdraw_wiring(
            &mut eval, &is_real,
            &perms, &sub_limbs,
            &merkle_root, &nullifier,
            &amount_lo, &amount_hi, &asset_id,
        );

        eval
    }
}

// ──────────────────────── Batched Spend Eval ────────────────────────

#[derive(Debug, Clone)]
pub struct BatchedSpendEval {
    pub log_n_rows: u32,
}

impl FrameworkEval for BatchedSpendEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let is_real = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "batch_spd_is_real".into(),
        });
        let merkle_root: [E::F; RATE] = std::array::from_fn(|j| {
            eval.get_preprocessed_column(PreProcessedColumnId {
                id: format!("batch_spd_merkle_root_{j}").into(),
            })
        });
        let nullifiers: [[E::F; RATE]; SPEND_NUM_INPUTS] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                eval.get_preprocessed_column(PreProcessedColumnId {
                    id: format!("batch_spd_nullifier_{i}_{j}").into(),
                })
            })
        });
        let output_commitments: [[E::F; RATE]; SPEND_NUM_OUTPUTS] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                eval.get_preprocessed_column(PreProcessedColumnId {
                    id: format!("batch_spd_output_commitment_{i}_{j}").into(),
                })
            })
        });

        // Permutation round constraints (all rows)
        let perms: Vec<_> = (0..SPD_NUM_PERMS)
            .map(|_| constrain_poseidon2_permutation(&mut eval))
            .collect();

        let sub_limbs: [E::F; SPD_NUM_SUB_LIMBS] =
            std::array::from_fn(|_| eval.next_trace_mask());
        let bits: [[E::F; SPD_BITS_PER_LIMB]; SPD_NUM_SUB_LIMBS] =
            std::array::from_fn(|_| std::array::from_fn(|_| eval.next_trace_mask()));

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

        // Carry columns
        let carry = eval.next_trace_mask();
        let carry_pos = eval.next_trace_mask();
        let carry_neg = eval.next_trace_mask();
        eval.add_constraint(carry_pos.clone() * (carry_pos.clone() - E::F::from(one)));
        eval.add_constraint(carry_neg.clone() * (carry_neg.clone() - E::F::from(one)));
        eval.add_constraint(carry_neg.clone() * carry_pos.clone());
        eval.add_constraint(carry.clone() - carry_pos + carry_neg);

        constrain_spend_wiring(
            &mut eval, &is_real,
            &perms, &sub_limbs,
            &carry,
            &merkle_root, &nullifiers, &output_commitments,
        );

        eval
    }
}

// ──────────────────────── Trace builders ─────────────────────────────

/// Build deposit batch traces (preprocessed + execution) for N deposits.
fn build_deposit_batch_traces(
    witnesses: &[DepositWitness],
    public_inputs: &[DepositPublicInputs],
    log_size: u32,
) -> Result<
    (
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    ),
    BatchError,
> {
    let table_size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Preprocessed columns: is_real, commitment[8], amount_lo, amount_hi, asset_id
    let mut preproc_cols: Vec<Col<SimdBackend, M31>> = (0..DEP_PREPROC_COLS)
        .map(|_| Col::<SimdBackend, M31>::zeros(table_size))
        .collect();

    // Execution columns
    let mut exec_cols: Vec<Col<SimdBackend, M31>> = (0..DEP_EXEC_COLS)
        .map(|_| Col::<SimdBackend, M31>::zeros(table_size))
        .collect();

    let dummy = dummy_permutation_trace();

    for (row, (witness, pub_in)) in witnesses.iter().zip(public_inputs.iter()).enumerate() {
        let (execution, _) = execute_deposit(witness)
            .map_err(|e| BatchError::DepositExecution(format!("{e}")))?;

        // Preprocessed: set per-row public inputs
        preproc_cols[0].set(row, M31::from_u32_unchecked(1)); // is_real
        for j in 0..RATE {
            preproc_cols[1 + j].set(row, pub_in.commitment[j]);
        }
        let amount_lo = M31::from_u32_unchecked((pub_in.amount & 0x7FFFFFFF) as u32);
        let amount_hi = M31::from_u32_unchecked((pub_in.amount >> 31) as u32);
        preproc_cols[9].set(row, amount_lo);
        preproc_cols[10].set(row, amount_hi);
        preproc_cols[11].set(row, pub_in.asset_id);

        // Execution: permutation traces
        let perm_traces: Vec<_> = execution
            .all_permutation_inputs
            .iter()
            .map(compute_permutation_trace)
            .collect();
        for (p, trace) in perm_traces.iter().enumerate() {
            write_permutation_to_trace(trace, &mut exec_cols, p * COLS_PER_PERM, row);
        }

        // Sub-limbs
        let sub_offset = DEP_NUM_PERMS * COLS_PER_PERM;
        for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
            exec_cols[sub_offset + k].set(row, limb);
        }

        // Bit decomposition
        let bit_offset = sub_offset + DEP_NUM_SUB_LIMBS;
        for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
            let bits = decompose_to_bits(limb.0);
            for (i, &bit) in bits.iter().enumerate() {
                exec_cols[bit_offset + k * DEP_BITS_PER_LIMB + i]
                    .set(row, M31::from_u32_unchecked(bit));
            }
        }
    }

    // Padding rows: dummy permutation traces (all values zero → constraints satisfied)
    for row in witnesses.len()..table_size {
        for p in 0..DEP_NUM_PERMS {
            write_permutation_to_trace(&dummy, &mut exec_cols, p * COLS_PER_PERM, row);
        }
    }

    let preprocessed = preproc_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();
    let execution = exec_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    Ok((preprocessed, execution))
}

/// Build withdraw batch traces.
fn build_withdraw_batch_traces(
    witnesses: &[WithdrawWitness],
    public_inputs: &[WithdrawPublicInputs],
    log_size: u32,
) -> Result<
    (
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    ),
    BatchError,
> {
    use crate::circuits::stark_withdraw::{MERKLE_PERM_START, MERKLE_PERM_END};

    let table_size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    let mut preproc_cols: Vec<Col<SimdBackend, M31>> = (0..WDR_PREPROC_COLS)
        .map(|_| Col::<SimdBackend, M31>::zeros(table_size))
        .collect();

    let mut exec_cols: Vec<Col<SimdBackend, M31>> = (0..WDR_EXEC_COLS)
        .map(|_| Col::<SimdBackend, M31>::zeros(table_size))
        .collect();

    let dummy = dummy_permutation_trace();
    let dummy_hash: [M31; RATE] = std::array::from_fn(|j| dummy.states[22][j]);
    let merkle_levels = MERKLE_PERM_END - MERKLE_PERM_START;
    let merkle_padding = compute_merkle_chain_padding(&dummy_hash, merkle_levels);

    for (row, (witness, pub_in)) in witnesses.iter().zip(public_inputs.iter()).enumerate() {
        let (execution, _) = execute_withdraw(witness)
            .map_err(|e| BatchError::WithdrawExecution(format!("{e}")))?;

        // Preprocessed
        preproc_cols[0].set(row, M31::from_u32_unchecked(1)); // is_real
        for j in 0..RATE {
            preproc_cols[1 + j].set(row, pub_in.merkle_root[j]);
        }
        for j in 0..RATE {
            preproc_cols[9 + j].set(row, pub_in.nullifier[j]);
        }
        preproc_cols[17].set(row, pub_in.amount_lo);
        preproc_cols[18].set(row, pub_in.amount_hi);
        preproc_cols[19].set(row, pub_in.asset_id);

        // Execution
        let perm_traces: Vec<_> = execution
            .all_permutation_inputs
            .iter()
            .map(compute_permutation_trace)
            .collect();
        for (p, trace) in perm_traces.iter().enumerate() {
            write_permutation_to_trace(trace, &mut exec_cols, p * COLS_PER_PERM, row);
        }

        let sub_offset = WDR_NUM_PERMS * COLS_PER_PERM;
        for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
            exec_cols[sub_offset + k].set(row, limb);
        }
        let bit_offset = sub_offset + WDR_NUM_SUB_LIMBS;
        for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
            let bits = decompose_to_bits(limb.0);
            for (i, &bit) in bits.iter().enumerate() {
                exec_cols[bit_offset + k * WDR_BITS_PER_LIMB + i]
                    .set(row, M31::from_u32_unchecked(bit));
            }
        }
    }

    // Padding rows
    for row in witnesses.len()..table_size {
        for p in 0..WDR_NUM_PERMS {
            if p >= MERKLE_PERM_START && p < MERKLE_PERM_END {
                let merkle_idx = p - MERKLE_PERM_START;
                write_permutation_to_trace(
                    &merkle_padding[merkle_idx],
                    &mut exec_cols,
                    p * COLS_PER_PERM,
                    row,
                );
            } else {
                write_permutation_to_trace(&dummy, &mut exec_cols, p * COLS_PER_PERM, row);
            }
        }
    }

    let preprocessed = preproc_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();
    let execution = exec_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    Ok((preprocessed, execution))
}

/// Build spend batch traces.
fn build_spend_batch_traces(
    witnesses: &[SpendWitness],
    public_inputs: &[SpendPublicInputs],
    log_size: u32,
) -> Result<
    (
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
        Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    ),
    BatchError,
> {
    let table_size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();

    let mut preproc_cols: Vec<Col<SimdBackend, M31>> = (0..SPD_PREPROC_COLS)
        .map(|_| Col::<SimdBackend, M31>::zeros(table_size))
        .collect();

    let mut exec_cols: Vec<Col<SimdBackend, M31>> = (0..SPD_EXEC_COLS)
        .map(|_| Col::<SimdBackend, M31>::zeros(table_size))
        .collect();

    let dummy = dummy_permutation_trace();
    let dummy_hash: [M31; RATE] = std::array::from_fn(|j| dummy.states[22][j]);
    let merkle_padding = compute_merkle_chain_padding(&dummy_hash, MERKLE_DEPTH);

    for (row, (witness, pub_in)) in witnesses.iter().zip(public_inputs.iter()).enumerate() {
        let (execution, _) = execute_spend(witness)
            .map_err(|e| BatchError::SpendExecution(format!("{e}")))?;

        // Preprocessed: is_real(1), merkle_root(8), nullifiers(16), output_commitments(16)
        preproc_cols[0].set(row, M31::from_u32_unchecked(1)); // is_real
        let mut idx = 1;
        for j in 0..RATE {
            preproc_cols[idx + j].set(row, pub_in.merkle_root[j]);
        }
        idx += RATE;
        for i in 0..SPEND_NUM_INPUTS {
            for j in 0..RATE {
                preproc_cols[idx + i * RATE + j].set(row, pub_in.nullifiers[i][j]);
            }
        }
        idx += SPEND_NUM_INPUTS * RATE;
        for i in 0..SPEND_NUM_OUTPUTS {
            for j in 0..RATE {
                preproc_cols[idx + i * RATE + j].set(row, pub_in.output_commitments[i][j]);
            }
        }

        // Execution
        let perm_traces: Vec<_> = execution
            .all_permutation_inputs
            .iter()
            .map(compute_permutation_trace)
            .collect();
        for (p, trace) in perm_traces.iter().enumerate() {
            write_permutation_to_trace(trace, &mut exec_cols, p * COLS_PER_PERM, row);
        }

        let sub_offset = SPD_NUM_PERMS * COLS_PER_PERM;
        for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
            exec_cols[sub_offset + k].set(row, limb);
        }
        let bit_offset = sub_offset + SPD_NUM_SUB_LIMBS;
        for (k, &limb) in execution.range_check_limbs.iter().enumerate() {
            let bits = decompose_to_bits(limb.0);
            for (i, &bit) in bits.iter().enumerate() {
                exec_cols[bit_offset + k * SPD_BITS_PER_LIMB + i]
                    .set(row, M31::from_u32_unchecked(bit));
            }
        }

        // Carry columns
        let carry_offset = bit_offset + SPD_NUM_BIT_COLS;
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
        let carry_val = hi_in - hi_out;
        let p = 0x7FFFFFFFu32;
        let carry_m31 = if carry_val >= 0 {
            M31::from_u32_unchecked(carry_val as u32)
        } else {
            M31::from_u32_unchecked(p - ((-carry_val) as u32))
        };
        exec_cols[carry_offset].set(row, carry_m31);
        exec_cols[carry_offset + 1]
            .set(row, M31::from_u32_unchecked(if carry_val > 0 { 1 } else { 0 }));
        exec_cols[carry_offset + 2]
            .set(row, M31::from_u32_unchecked(if carry_val < 0 { 1 } else { 0 }));
    }

    // Padding rows
    for row in witnesses.len()..table_size {
        for p_idx in 0..SPD_NUM_PERMS {
            let mut is_merkle = false;
            for inp_idx in 0..SPEND_NUM_INPUTS {
                let mrk_start = crate::circuits::stark_spend::input_merkle_start(inp_idx);
                let mrk_end = crate::circuits::stark_spend::input_merkle_end(inp_idx);
                if p_idx >= mrk_start && p_idx < mrk_end {
                    let merkle_idx = p_idx - mrk_start;
                    write_permutation_to_trace(
                        &merkle_padding[merkle_idx],
                        &mut exec_cols,
                        p_idx * COLS_PER_PERM,
                        row,
                    );
                    is_merkle = true;
                    break;
                }
            }
            if !is_merkle {
                write_permutation_to_trace(&dummy, &mut exec_cols, p_idx * COLS_PER_PERM, row);
            }
        }
    }

    let preprocessed = preproc_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();
    let execution = exec_cols
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    Ok((preprocessed, execution))
}

// ──────────────────────── Prover ─────────────────────────────────────

/// Prove a batch of privacy transactions with a single STARK proof.
pub fn prove_privacy_batch(
    batch: &PrivacyBatch,
) -> Result<PrivacyBatchProof, BatchError> {
    if batch.deposits.is_empty() && batch.withdrawals.is_empty() && batch.spends.is_empty() {
        return Err(BatchError::EmptyBatch);
    }

    // Execute all witnesses and collect public inputs
    let mut dep_pub: Vec<DepositPublicInputs> = Vec::with_capacity(batch.deposits.len());
    for w in &batch.deposits {
        let (_, pi) = execute_deposit(w)
            .map_err(|e| BatchError::DepositExecution(format!("{e}")))?;
        dep_pub.push(pi);
    }

    let mut wdr_pub: Vec<WithdrawPublicInputs> = Vec::with_capacity(batch.withdrawals.len());
    for w in &batch.withdrawals {
        let (_, pi) = execute_withdraw(w)
            .map_err(|e| BatchError::WithdrawExecution(format!("{e}")))?;
        wdr_pub.push(pi);
    }

    let mut spd_pub: Vec<SpendPublicInputs> = Vec::with_capacity(batch.spends.len());
    for w in &batch.spends {
        let (_, pi) = execute_spend(w)
            .map_err(|e| BatchError::SpendExecution(format!("{e}")))?;
        spd_pub.push(pi);
    }

    let public_inputs = BatchPublicInputs {
        deposits: dep_pub.clone(),
        withdrawals: wdr_pub.clone(),
        spends: spd_pub.clone(),
    };

    // Determine max log_size across all components for twiddles
    let dep_log = if !batch.deposits.is_empty() {
        Some(batch_log_size(batch.deposits.len()))
    } else {
        None
    };
    let wdr_log = if !batch.withdrawals.is_empty() {
        Some(batch_log_size(batch.withdrawals.len()))
    } else {
        None
    };
    let spd_log = if !batch.spends.is_empty() {
        Some(batch_log_size(batch.spends.len()))
    } else {
        None
    };
    let max_log = [dep_log, wdr_log, spd_log]
        .iter()
        .filter_map(|x| *x)
        .max()
        .unwrap();

    // PCS setup
    let pcs_config = PcsConfig::default();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_log + 1 + pcs_config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();

    // Fiat-Shamir: mix public inputs before any tree commitment
    mix_public_inputs_into_channel(channel, &public_inputs);

    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(pcs_config, &twiddles);

    // Build traces for each non-empty type
    let mut all_preprocessed: Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> =
        Vec::new();
    let mut all_execution: Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> =
        Vec::new();

    if let Some(log) = dep_log {
        let (prep, exec) =
            build_deposit_batch_traces(&batch.deposits, &dep_pub, log)?;
        all_preprocessed.extend(prep);
        all_execution.extend(exec);
    }
    if let Some(log) = wdr_log {
        let (prep, exec) =
            build_withdraw_batch_traces(&batch.withdrawals, &wdr_pub, log)?;
        all_preprocessed.extend(prep);
        all_execution.extend(exec);
    }
    if let Some(log) = spd_log {
        let (prep, exec) =
            build_spend_batch_traces(&batch.spends, &spd_pub, log)?;
        all_preprocessed.extend(prep);
        all_execution.extend(exec);
    }

    // Tree 0: Preprocessed
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
        all_preprocessed,
    ));
    tree_builder.commit(channel);

    // Tree 1: Execution
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
        all_execution,
    ));
    tree_builder.commit(channel);

    // Build components with shared allocator (same order as traces)
    let mut allocator = TraceLocationAllocator::default();
    let mut comp_storage: Vec<Box<dyn ComponentProverErased>> = Vec::new();

    if let Some(log) = dep_log {
        let eval = BatchedDepositEval { log_n_rows: log };
        let comp = FrameworkComponent::new(&mut allocator, eval, SecureField::zero());
        comp_storage.push(Box::new(comp));
    }
    if let Some(log) = wdr_log {
        let eval = BatchedWithdrawEval { log_n_rows: log };
        let comp = FrameworkComponent::new(&mut allocator, eval, SecureField::zero());
        comp_storage.push(Box::new(comp));
    }
    if let Some(log) = spd_log {
        let eval = BatchedSpendEval { log_n_rows: log };
        let comp = FrameworkComponent::new(&mut allocator, eval, SecureField::zero());
        comp_storage.push(Box::new(comp));
    }

    let component_refs: Vec<&dyn ComponentProver<SimdBackend>> = comp_storage
        .iter()
        .map(|c| c.as_component_prover())
        .collect();

    let stark_proof = prove::<SimdBackend, Blake2sMerkleChannel>(
        &component_refs,
        channel,
        commitment_scheme,
    )
    .map_err(|e| BatchError::Proving(format!("{e:?}")))?;

    Ok(PrivacyBatchProof {
        stark_proof,
        public_inputs,
    })
}

// ──────────────────────── Verifier ───────────────────────────────────

/// Verify a batch proof against the given public inputs.
pub fn verify_privacy_batch(
    proof: &PrivacyBatchProof,
    public_inputs: &BatchPublicInputs,
) -> Result<(), BatchError> {
    let pcs_config = PcsConfig::default();

    let has_deposits = !public_inputs.deposits.is_empty();
    let has_withdrawals = !public_inputs.withdrawals.is_empty();
    let has_spends = !public_inputs.spends.is_empty();

    if !has_deposits && !has_withdrawals && !has_spends {
        return Err(BatchError::EmptyBatch);
    }

    let dep_log = if has_deposits {
        Some(batch_log_size(public_inputs.deposits.len()))
    } else {
        None
    };
    let wdr_log = if has_withdrawals {
        Some(batch_log_size(public_inputs.withdrawals.len()))
    } else {
        None
    };
    let spd_log = if has_spends {
        Some(batch_log_size(public_inputs.spends.len()))
    } else {
        None
    };

    // Build dummy components to get trace_log_degree_bounds, then merge per-tree
    let mut dummy_allocator = TraceLocationAllocator::default();

    let mut typed_deps_dummy: Vec<FrameworkComponent<BatchedDepositEval>> = Vec::new();
    let mut typed_wdrs_dummy: Vec<FrameworkComponent<BatchedWithdrawEval>> = Vec::new();
    let mut typed_spds_dummy: Vec<FrameworkComponent<BatchedSpendEval>> = Vec::new();

    if let Some(log) = dep_log {
        let eval = BatchedDepositEval { log_n_rows: log };
        typed_deps_dummy.push(FrameworkComponent::new(&mut dummy_allocator, eval, SecureField::zero()));
    }
    if let Some(log) = wdr_log {
        let eval = BatchedWithdrawEval { log_n_rows: log };
        typed_wdrs_dummy.push(FrameworkComponent::new(&mut dummy_allocator, eval, SecureField::zero()));
    }
    if let Some(log) = spd_log {
        let eval = BatchedSpendEval { log_n_rows: log };
        typed_spds_dummy.push(FrameworkComponent::new(&mut dummy_allocator, eval, SecureField::zero()));
    }

    // Collect all components as dyn Component for bounds
    let mut dummy_refs: Vec<&dyn Component> = Vec::new();
    for c in &typed_deps_dummy { dummy_refs.push(c); }
    for c in &typed_wdrs_dummy { dummy_refs.push(c); }
    for c in &typed_spds_dummy { dummy_refs.push(c); }

    // Merge bounds: each component returns TreeVec<Vec<u32>>, we merge per tree index
    let num_trees = 2; // Tree 0 = preprocessed, Tree 1 = execution
    let mut merged_bounds: Vec<Vec<u32>> = vec![Vec::new(); num_trees];
    for comp in &dummy_refs {
        let bounds = comp.trace_log_degree_bounds();
        for (tree_idx, tree_bounds) in bounds.iter().enumerate() {
            merged_bounds[tree_idx].extend(tree_bounds);
        }
    }

    // Set up channel and verifier
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    mix_public_inputs_into_channel(channel, public_inputs);

    let mut commitment_scheme =
        CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);

    // Replay commitments for each tree
    for (tree_idx, bounds) in merged_bounds.iter().enumerate() {
        commitment_scheme.commit(
            proof.stark_proof.commitments[tree_idx],
            bounds,
            channel,
        );
    }

    // Build real components (same allocator order as prover)
    let mut allocator = TraceLocationAllocator::default();

    let mut typed_deps: Vec<FrameworkComponent<BatchedDepositEval>> = Vec::new();
    let mut typed_wdrs: Vec<FrameworkComponent<BatchedWithdrawEval>> = Vec::new();
    let mut typed_spds: Vec<FrameworkComponent<BatchedSpendEval>> = Vec::new();

    if let Some(log) = dep_log {
        let eval = BatchedDepositEval { log_n_rows: log };
        typed_deps.push(FrameworkComponent::new(&mut allocator, eval, SecureField::zero()));
    }
    if let Some(log) = wdr_log {
        let eval = BatchedWithdrawEval { log_n_rows: log };
        typed_wdrs.push(FrameworkComponent::new(&mut allocator, eval, SecureField::zero()));
    }
    if let Some(log) = spd_log {
        let eval = BatchedSpendEval { log_n_rows: log };
        typed_spds.push(FrameworkComponent::new(&mut allocator, eval, SecureField::zero()));
    }

    let mut component_refs: Vec<&dyn Component> = Vec::new();
    for c in &typed_deps { component_refs.push(c); }
    for c in &typed_wdrs { component_refs.push(c); }
    for c in &typed_spds { component_refs.push(c); }

    stwo_verify::<Blake2sMerkleChannel>(
        &component_refs,
        channel,
        &mut commitment_scheme,
        proof.stark_proof.clone(),
    )
    .map_err(|e| BatchError::Verification(format!("{e:?}")))
}

// ──────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::spend::{InputNoteWitness, OutputNoteWitness};
    use crate::crypto::commitment::{derive_pubkey, Note};
    use crate::crypto::merkle_m31::PoseidonMerkleTreeM31;

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

    fn make_deposit_witness_with_blinding(amount: u64, blinding_seed: u32) -> DepositWitness {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let pk = derive_pubkey(&sk);
        let amount_lo = M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32);
        let amount_hi = M31::from_u32_unchecked((amount >> 31) as u32);
        let asset_id = M31::from_u32_unchecked(0);
        let blinding = [blinding_seed, blinding_seed + 1, blinding_seed + 2, blinding_seed + 3]
            .map(M31::from_u32_unchecked);
        let note = Note::new(pk, asset_id, amount_lo, amount_hi, blinding);
        DepositWitness {
            note,
            amount,
            asset_id,
        }
    }

    fn make_withdraw_witness(amount: u64) -> WithdrawWitness {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let pk = derive_pubkey(&sk);
        let amount_lo = M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32);
        let amount_hi = M31::from_u32_unchecked((amount >> 31) as u32);
        let asset_id = M31::from_u32_unchecked(0);
        let blinding = [10, 20, 30, 40].map(M31::from_u32_unchecked);
        let note = Note::new(pk, asset_id, amount_lo, amount_hi, blinding);

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
        let mut tree = PoseidonMerkleTreeM31::new(crate::circuits::spend::MERKLE_DEPTH);
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

    // ── Test 1: Single deposit batch ──

    #[test]
    fn test_batch_single_deposit() {
        let batch = PrivacyBatch {
            deposits: vec![make_deposit_witness(1000)],
            withdrawals: vec![],
            spends: vec![],
        };
        let proof = prove_privacy_batch(&batch).expect("proving should succeed");
        verify_privacy_batch(&proof, &proof.public_inputs)
            .expect("verification should succeed");

        // Public inputs should match single-tx execution
        assert_eq!(proof.public_inputs.deposits.len(), 1);
        assert_eq!(proof.public_inputs.deposits[0].amount, 1000);
    }

    // ── Test 2: Multiple deposits ──

    #[test]
    fn test_batch_multiple_deposits() {
        let batch = PrivacyBatch {
            deposits: vec![
                make_deposit_witness_with_blinding(100, 1),
                make_deposit_witness_with_blinding(200, 5),
                make_deposit_witness_with_blinding(300, 9),
                make_deposit_witness_with_blinding(400, 13),
            ],
            withdrawals: vec![],
            spends: vec![],
        };
        let proof = prove_privacy_batch(&batch).expect("proving should succeed");
        verify_privacy_batch(&proof, &proof.public_inputs)
            .expect("verification should succeed");
        assert_eq!(proof.public_inputs.deposits.len(), 4);
    }

    // ── Test 3: Single withdraw ──

    #[test]
    fn test_batch_single_withdraw() {
        let batch = PrivacyBatch {
            deposits: vec![],
            withdrawals: vec![make_withdraw_witness(1000)],
            spends: vec![],
        };
        let proof = prove_privacy_batch(&batch).expect("proving should succeed");
        verify_privacy_batch(&proof, &proof.public_inputs)
            .expect("verification should succeed");
        assert_eq!(proof.public_inputs.withdrawals.len(), 1);
    }

    // ── Test 4: Single spend ──

    #[test]
    fn test_batch_single_spend() {
        let batch = PrivacyBatch {
            deposits: vec![],
            withdrawals: vec![],
            spends: vec![make_spend_witness([1000, 2000], [1500, 1500])],
        };
        let proof = prove_privacy_batch(&batch).expect("proving should succeed");
        verify_privacy_batch(&proof, &proof.public_inputs)
            .expect("verification should succeed");
        assert_eq!(proof.public_inputs.spends.len(), 1);
    }

    // ── Test 5: Mixed types ──

    #[test]
    fn test_batch_mixed_types() {
        let batch = PrivacyBatch {
            deposits: vec![
                make_deposit_witness_with_blinding(100, 1),
                make_deposit_witness_with_blinding(200, 5),
                make_deposit_witness_with_blinding(300, 9),
            ],
            withdrawals: vec![
                make_withdraw_witness(500),
                make_withdraw_witness(600),
            ],
            spends: vec![make_spend_witness([1000, 2000], [1500, 1500])],
        };
        let proof = prove_privacy_batch(&batch).expect("proving should succeed");
        verify_privacy_batch(&proof, &proof.public_inputs)
            .expect("verification should succeed");
        assert_eq!(proof.public_inputs.deposits.len(), 3);
        assert_eq!(proof.public_inputs.withdrawals.len(), 2);
        assert_eq!(proof.public_inputs.spends.len(), 1);
    }

    // ── Test 6: Deposits only (no withdrawals or spends) ──

    #[test]
    fn test_batch_deposits_only() {
        let batch = PrivacyBatch {
            deposits: vec![
                make_deposit_witness_with_blinding(100, 1),
                make_deposit_witness_with_blinding(200, 5),
            ],
            withdrawals: vec![],
            spends: vec![],
        };
        let proof = prove_privacy_batch(&batch).expect("proving should succeed");
        verify_privacy_batch(&proof, &proof.public_inputs)
            .expect("verification should succeed");
    }

    // ── Test 7: Wrong commitment rejected ──

    #[test]
    fn test_batch_wrong_commitment_rejected() {
        let batch = PrivacyBatch {
            deposits: vec![make_deposit_witness(1000)],
            withdrawals: vec![],
            spends: vec![],
        };
        let proof = prove_privacy_batch(&batch).expect("proving should succeed");

        let mut bad_inputs = proof.public_inputs.clone();
        bad_inputs.deposits[0].commitment[0] = M31::from_u32_unchecked(999999);

        let result = verify_privacy_batch(&proof, &bad_inputs);
        assert!(result.is_err(), "modified commitment should fail verification");
    }

    // ── Test 8: Wrong nullifier rejected ──

    #[test]
    fn test_batch_wrong_nullifier_rejected() {
        let batch = PrivacyBatch {
            deposits: vec![],
            withdrawals: vec![make_withdraw_witness(1000)],
            spends: vec![],
        };
        let proof = prove_privacy_batch(&batch).expect("proving should succeed");

        let mut bad_inputs = proof.public_inputs.clone();
        bad_inputs.withdrawals[0].nullifier[0] = M31::from_u32_unchecked(999999);

        let result = verify_privacy_batch(&proof, &bad_inputs);
        assert!(result.is_err(), "modified nullifier should fail verification");
    }

    // ── Test 9: Empty batch rejected ──

    #[test]
    fn test_batch_empty_rejected() {
        let batch = PrivacyBatch {
            deposits: vec![],
            withdrawals: vec![],
            spends: vec![],
        };
        let result = prove_privacy_batch(&batch);
        assert!(result.is_err(), "empty batch should fail");
        match result.unwrap_err() {
            BatchError::EmptyBatch => {}
            e => panic!("expected EmptyBatch, got: {e}"),
        }
    }

    // ── Test 10: Public input hash deterministic ──

    #[test]
    fn test_batch_public_input_hash_deterministic() {
        let w = make_deposit_witness(1000);
        let (_, pi) = execute_deposit(&w).unwrap();

        let inputs = BatchPublicInputs {
            deposits: vec![pi.clone()],
            withdrawals: vec![],
            spends: vec![],
        };

        let hash1 = hash_batch_public_inputs(&inputs);
        let hash2 = hash_batch_public_inputs(&inputs);
        assert_eq!(hash1, hash2);
    }

    // ── Test 11: Log size minimum ──

    #[test]
    fn test_batch_log_size_minimum() {
        assert_eq!(batch_log_size(1), MIN_LOG_SIZE);
        assert_eq!(batch_log_size(0), MIN_LOG_SIZE);
        assert_eq!(batch_log_size(16), MIN_LOG_SIZE);
        assert_eq!(batch_log_size(17), 5);
        assert_eq!(batch_log_size(32), 5);
        assert_eq!(batch_log_size(33), 6);
    }

    // ── Test 12: Padding rows satisfy constraints ──

    #[test]
    fn test_batch_padding_rows_correct() {
        // 1 deposit in a batch → 15 padding rows (log_size=4 means 16 rows)
        let batch = PrivacyBatch {
            deposits: vec![make_deposit_witness(42)],
            withdrawals: vec![],
            spends: vec![],
        };
        let proof = prove_privacy_batch(&batch).expect("should prove with padding");
        verify_privacy_batch(&proof, &proof.public_inputs)
            .expect("padding rows should satisfy constraints");
    }

    // ── Test 13: Larger batch ──

    #[test]
    fn test_batch_larger_batch() {
        // 16 deposits → fills exactly log_size=4
        // Plus 1 more to force log_size=5
        let mut deposits: Vec<DepositWitness> = Vec::new();
        for i in 0..17 {
            deposits.push(make_deposit_witness_with_blinding(100 + i as u64, (i * 4) as u32 + 1));
        }
        let batch = PrivacyBatch {
            deposits,
            withdrawals: vec![],
            spends: vec![],
        };
        let proof = prove_privacy_batch(&batch).expect("larger batch should prove");
        verify_privacy_batch(&proof, &proof.public_inputs)
            .expect("larger batch should verify");
        assert_eq!(proof.public_inputs.deposits.len(), 17);
    }
}
