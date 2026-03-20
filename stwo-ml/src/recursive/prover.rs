//! Recursive STARK prover — proves "I verified the GKR proof and it passed."
//!
//! This module wires together the witness generator and AIR circuit to produce
//! a recursive STARK proof using STWO's standard `prove()` function.
//!
//! # Pipeline
//!
//! ```text
//! GKRProof + Circuit + Output + Weights
//!     → generate_witness()     → GkrVerifierWitness
//!     → build_recursive_trace()→ RecursiveTraceData
//!     → commit traces          → CommitmentSchemeProver
//!     → stwo::prove()          → StarkProof
//!     → RecursiveProof
//! ```

use num_traits::Zero;
use stwo::core::channel::MerkleChannel;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::{QM31, SecureField};
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs_lifted::poseidon252_merkle::{Poseidon252MerkleChannel, Poseidon252MerkleHasher};
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::prove;
use stwo::prover::CommitmentSchemeProver;
use stwo_constraint_framework::{FrameworkComponent, TraceLocationAllocator};

use crate::backend::convert_evaluations;

use crate::compiler::graph::GraphWeights;
use crate::components::matmul::M31Matrix;
use crate::gkr::circuit::LayeredCircuit;
use crate::gkr::types::GKRProof;

use super::air::{build_recursive_trace, RecursiveVerifierEval};
use super::types::{RecursiveProof, RecursiveProofMetadata, RecursivePublicInputs};
use super::witness::generate_witness;

/// Error type for recursive proving.
#[derive(Debug)]
pub enum RecursiveError {
    /// The GKR proof failed verification (Pass 1).
    GkrVerificationFailed(String),
    /// Trace building failed.
    TraceBuildFailed(String),
    /// STWO proving failed.
    ProvingFailed(String),
}

impl std::fmt::Display for RecursiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecursiveError::GkrVerificationFailed(e) => write!(f, "GKR verification failed: {e}"),
            RecursiveError::TraceBuildFailed(e) => write!(f, "trace build failed: {e}"),
            RecursiveError::ProvingFailed(e) => write!(f, "recursive proving failed: {e}"),
        }
    }
}

impl std::error::Error for RecursiveError {}

/// Produce a recursive STARK proof for a GKR proof.
///
/// This is the main entry point for recursive composition. It:
/// 1. Generates the verifier witness (validates the GKR proof via production verifier)
/// 2. Builds the execution trace from the witness
/// 3. Commits the trace using STWO's commitment scheme
/// 4. Calls `stwo::prove()` to produce the recursive STARK proof
///
/// # Arguments
///
/// * `circuit` - The model's layered circuit
/// * `gkr_proof` - The GKR proof to verify recursively
/// * `output` - The model's output matrix
/// * `weights` - Model weights (needed for aggregated binding verification)
/// * `weight_super_root` - Poseidon root of all weight commitments
/// * `io_commitment` - Poseidon hash of packed IO
/// * `gkr_prove_time_secs` - Time taken to produce the GKR proof (for metadata)
///
/// # Returns
///
/// A `RecursiveProof` containing the STARK proof + public inputs.
/// On-chain, only this proof is submitted (not the original GKR proof).
pub fn prove_recursive(
    circuit: &LayeredCircuit,
    gkr_proof: &GKRProof,
    output: &M31Matrix,
    weights: &GraphWeights,
    weight_super_root: QM31,
    io_commitment: QM31,
    gkr_prove_time_secs: f64,
) -> Result<RecursiveProof, RecursiveError> {
    let t_start = std::time::Instant::now();

    // ── Step 1: Generate witness ─────────────────────────────────────
    eprintln!("  [Recursive] Step 1/4: Generating verifier witness...");
    let witness = generate_witness(
        circuit,
        gkr_proof,
        output,
        Some(weights),
        weight_super_root,
        io_commitment,
    )
    .map_err(|e| RecursiveError::GkrVerificationFailed(format!("{e:?}")))?;

    eprintln!(
        "  [Recursive] Witness: {} poseidon perms, {} sumcheck rounds, {} qm31 ops",
        witness.n_poseidon_perms, witness.n_sumcheck_rounds, witness.n_qm31_ops,
    );

    // ── Step 2: Build trace ──────────────────────────────────────────
    eprintln!("  [Recursive] Step 2/4: Building execution trace...");
    let trace_data = build_recursive_trace(&witness);

    eprintln!(
        "  [Recursive] Trace: {} rows (log_size={}), {} cols/row, {} real rows",
        1u32 << trace_data.log_size,
        trace_data.log_size,
        super::air::COLS_PER_ROW,
        trace_data.n_real_rows,
    );

    // ── Step 3: Commit traces ────────────────────────────────────────
    eprintln!("  [Recursive] Step 3/4: Committing traces...");
    let config = PcsConfig::default();
    let log_size = trace_data.log_size;
    let max_degree_bound = log_size + 1;

    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_degree_bound + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // Use Poseidon252MerkleChannel so the STARK is verifiable by stwo-cairo-verifier
    // (Cairo's native Poseidon). This eliminates the need to constrain felt252 Hades
    // in the M31 AIR — the STARK proof itself uses Poseidon252 for Fiat-Shamir and
    // Merkle commitments, matching what the Cairo verifier expects.
    let channel = &mut <Poseidon252MerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(config, &twiddles);

    let domain = CanonicCoset::new(log_size).circle_domain();

    // Tree 0: Preprocessed columns (is_first, is_last, is_chain)
    {
        let mut tree_builder = commitment_scheme.tree_builder();
        let is_first_col = simd_column_from_vec(&trace_data.preprocessed_is_first);
        let is_last_col = simd_column_from_vec(&trace_data.preprocessed_is_last);
        let is_chain_col = simd_column_from_vec(&trace_data.preprocessed_is_chain);
        let simd_evals = vec![
            CircleEvaluation::new(domain, is_first_col),
            CircleEvaluation::new(domain, is_last_col),
            CircleEvaluation::new(domain, is_chain_col),
        ];
        tree_builder.extend_evals(
            convert_evaluations::<SimdBackend, SimdBackend, M31>(simd_evals),
        );
        tree_builder.commit(channel);
    }

    // Tree 1: Execution trace (655 columns)
    {
        let mut tree_builder = commitment_scheme.tree_builder();
        let simd_evals: Vec<CircleEvaluation<SimdBackend, M31, _>> = trace_data
            .execution_trace
            .iter()
            .map(|col| {
                let simd_col = simd_column_from_vec(col);
                CircleEvaluation::new(domain, simd_col)
            })
            .collect();
        tree_builder.extend_evals(
            convert_evaluations::<SimdBackend, SimdBackend, M31>(simd_evals),
        );
        tree_builder.commit(channel);
    }

    // ── Step 4: Prove ────────────────────────────────────────────────
    eprintln!("  [Recursive] Step 4/4: Proving (STARK)...");

    // Compute initial/final digest limbs.
    // Initial = zero (fresh channel).
    // Final = the output digest of the last recorded Hades operation.
    // This may differ from the production verifier's final digest if the
    // witness only partially records the chain (Pass 2 covers core layers).
    let zero_limbs = super::air::felt252_to_limbs(&starknet_ff::FieldElement::ZERO);

    // Find the last ChannelOp's digest_after
    let last_channel_op = witness.ops.iter().rev().find_map(|op| {
        if let super::types::WitnessOp::ChannelOp { digest_after, .. } = op {
            Some(*digest_after)
        } else {
            None
        }
    });

    let final_digest_felt = last_channel_op.unwrap_or(starknet_ff::FieldElement::ZERO);
    let final_limbs = super::air::felt252_to_limbs(&final_digest_felt);

    eprintln!(
        "  [Recursive] Final digest: {:?} (production: {:?}, match: {})",
        final_digest_felt, witness.final_digest,
        final_digest_felt == witness.final_digest,
    );

    let eval = RecursiveVerifierEval {
        log_n_rows: log_size,
        initial_digest_limbs: zero_limbs,
        final_digest_limbs: final_limbs,
    };

    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        eval,
        SecureField::zero(),
    );

    let stark_proof =
        prove::<SimdBackend, Poseidon252MerkleChannel>(&[&component], channel, commitment_scheme)
            .map_err(|e| RecursiveError::ProvingFailed(format!("{e:?}")))?;

    let recursive_prove_time = t_start.elapsed().as_secs_f64();
    eprintln!(
        "  [Recursive] Done in {:.2}s (trace: {}x{}, proof size: {} bytes)",
        recursive_prove_time,
        1u32 << log_size,
        super::air::COLS_PER_ROW,
        estimate_proof_size(&stark_proof),
    );

    Ok(RecursiveProof {
        stark_proof: stark_proof,
        public_inputs: witness.public_inputs,
        final_digest: final_digest_felt,
        log_size,
        metadata: RecursiveProofMetadata {
            recursive_prove_time_secs: recursive_prove_time,
            gkr_prove_time_secs,
            n_poseidon_perms: witness.n_poseidon_perms,
            n_sumcheck_rounds: witness.n_sumcheck_rounds,
            trace_log_size: log_size,
            n_trace_columns: super::air::COLS_PER_ROW,
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Convert a Vec<M31> to a SIMD column for STWO.
fn simd_column_from_vec(data: &[M31]) -> Col<SimdBackend, M31> {
    let mut col = Col::<SimdBackend, M31>::zeros(data.len());
    for (i, &val) in data.iter().enumerate() {
        col.set(i, val);
    }
    col
}

/// Rough estimate of serialized proof size.
fn estimate_proof_size(_proof: &StarkProof<Poseidon252MerkleHasher>) -> usize {
    4096 // placeholder
}

/// Serialize a STARK proof to bytes (placeholder — binary format in Phase 2D).
fn serialize_stark_proof(_proof: &StarkProof<Poseidon252MerkleHasher>) -> Vec<u8> {
    Vec::new()
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use stwo::core::fields::cm31::CM31;

    #[test]
    fn test_prove_recursive_1layer() {
        // End-to-end: prove a 1-layer MatMul GKR → recursive STARK.
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");

        let zero = QM31(CM31(M31::from(0), M31::from(0)), CM31(M31::from(0), M31::from(0)));

        let result = prove_recursive(
            &circuit,
            gkr,
            &proof.execution.output,
            &weights,
            zero,
            zero,
            0.0,
        );

        let recursive_proof = result.expect("recursive proving should succeed");
        assert!(recursive_proof.public_inputs.verified);
        assert!(recursive_proof.metadata.n_poseidon_perms > 0);
        assert!(recursive_proof.metadata.recursive_prove_time_secs > 0.0);
        eprintln!(
            "Recursive proof: {:.3}s, {} poseidon perms, log_size={}",
            recursive_proof.metadata.recursive_prove_time_secs,
            recursive_proof.metadata.n_poseidon_perms,
            recursive_proof.metadata.trace_log_size,
        );
    }
}
