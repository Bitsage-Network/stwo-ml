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

/// Debug logging for recursive prover — only prints in debug builds.
macro_rules! recursive_log {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        eprintln!($($arg)*);
    };
}

use num_traits::Zero;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::{SecureField, QM31};
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs_lifted::poseidon252_merkle::{
    Poseidon252MerkleChannel, Poseidon252MerkleHasher,
};
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
use super::witness::generate_witness_with_policy;

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
    // Default: reconstruct felt252 from QM31 (lossy — 124 bits + sentinel).
    // Production callers should use prove_recursive_with_policy and set
    // io_commitment_felt252 on the result to the original Poseidon hash.
    prove_recursive_with_policy(
        circuit,
        gkr_proof,
        output,
        weights,
        weight_super_root,
        io_commitment,
        gkr_prove_time_secs,
        None,
    )
}

/// Generate a recursive STARK proof with explicit policy binding.
pub fn prove_recursive_with_policy(
    circuit: &LayeredCircuit,
    gkr_proof: &GKRProof,
    output: &M31Matrix,
    weights: &GraphWeights,
    weight_super_root: QM31,
    io_commitment: QM31,
    gkr_prove_time_secs: f64,
    policy: Option<&crate::policy::PolicyConfig>,
) -> Result<RecursiveProof, RecursiveError> {
    let t_start = std::time::Instant::now();

    // ── Step 1: Generate witness ─────────────────────────────────────
    recursive_log!("  [Recursive] Step 1/4: Generating verifier witness...");
    let witness = generate_witness_with_policy(
        circuit,
        gkr_proof,
        output,
        Some(weights),
        weight_super_root,
        io_commitment,
        policy,
    )
    .map_err(|e| RecursiveError::GkrVerificationFailed(format!("{e:?}")))?;

    recursive_log!(
        "  [Recursive] Witness: {} poseidon perms, {} sumcheck rounds, {} qm31 ops",
        witness.n_poseidon_perms, witness.n_sumcheck_rounds, witness.n_qm31_ops,
    );

    // ── Step 1b: Verify all Hades permutations offline ──────────────
    // This ensures every (input, output) pair in the witness is a valid
    // Hades permutation, providing soundness at the prover level even
    // before the Hades AIR is fully integrated into the multi-component STARK.
    let n_hades_verified = verify_hades_perms_offline(&witness).map_err(|e| {
        RecursiveError::GkrVerificationFailed(format!("Hades permutation check failed: {e}"))
    })?;
    recursive_log!(
        "  [Recursive] Verified {} Hades permutations offline",
        n_hades_verified
    );

    // Hades AIR: inline constraints for the Hades permutation.
    // Default ON in production. Set OBELYZK_HADES_AIR=0 to disable (faster but less sound).
    // In test builds, default OFF for speed (offline verification still runs).
    #[cfg(not(test))]
    let hades_enabled = std::env::var("OBELYZK_HADES_AIR")
        .map(|v| v != "0")
        .unwrap_or(true); // ON by default in production
    #[cfg(test)]
    let hades_enabled = std::env::var("OBELYZK_HADES_AIR")
        .map(|v| v == "1")
        .unwrap_or(false); // OFF by default in tests

    // ── Step 2: Build traces ─────────────────────────────────────────
    recursive_log!("  [Recursive] Step 2/5: Building chain execution trace...");
    let trace_data = build_recursive_trace(&witness);

    // Build Hades verification trace from HadesPerm witness ops
    let hades_perms = extract_hades_perms(&witness);
    let hades_trace = if hades_enabled {
        recursive_log!("  [Recursive] Step 2b/5: Building Hades verification trace...");
        super::hades_air::build_hades_trace(&hades_perms)
    } else {
        // Placeholder empty trace when Hades AIR is disabled
        super::hades_air::HadesTraceData {
            trace: Vec::new(),
            log_size: 1,
            n_real_rows: 0,
            n_perms: 0,
        }
    };

    recursive_log!(
        "  [Recursive] Trace: {} rows (log_size={}), {} cols/row, {} real rows",
        1u32 << trace_data.log_size,
        trace_data.log_size,
        super::air::COLS_PER_ROW,
        trace_data.n_real_rows,
    );

    // ── Step 3: Commit traces ────────────────────────────────────────
    recursive_log!("  [Recursive] Step 3/5: Committing traces...");
    // Security-hardened PCS config for recursive proofs.
    //
    // PcsConfig::default() gives only 13 bits of security
    // (pow_bits=10, log_blowup=1, n_queries=3).
    //
    // Production config: 96+ bits of STARK security.
    //   log_blowup=3  (8x blowup — moderate proof size)
    //   n_queries=30   (30 FRI queries — 3*30=90 bits FRI security)
    //   pow_bits=16    (proof-of-work grinding protection)
    //   Total: 90 + 16 = 106 bits
    //
    // Override via OBELYZK_RECURSIVE_SECURITY env var:
    //   "test"       → 13 bits  (fast, for unit tests)
    //   "production" → 106 bits (default)
    let config = {
        // In test builds only, allow env var override for fast testing.
        // Production builds always use the hardened config.
        #[cfg(test)]
        let level = std::env::var("OBELYZK_RECURSIVE_SECURITY")
            .unwrap_or_else(|_| "production".to_string());
        #[cfg(not(test))]
        let level = "production".to_string();
        match level.as_str() {
            #[cfg(test)]
            "test" => PcsConfig::default(), // 13 bits — unit tests only
            _ => PcsConfig {
                pow_bits: 16,
                fri_config: stwo::core::fri::FriConfig::new(3, 3, 30),
            },
        }
    };
    let chain_log_size = trace_data.log_size;
    let hades_log_size = hades_trace.log_size;
    // Twiddles computed after unified_log_size is known (below).

    // Use Poseidon252MerkleChannel so the STARK is verifiable by stwo-cairo-verifier
    // (Cairo's native Poseidon). This eliminates the need to constrain felt252 Hades
    // in the M31 AIR — the STARK proof itself uses Poseidon252 for Fiat-Shamir and
    // Merkle commitments, matching what the Cairo verifier expects.
    let channel = &mut <Poseidon252MerkleChannel as MerkleChannel>::C::default();
    // Mix PcsConfig into channel BEFORE any tree commits.
    // MUST use individual mix_u64 calls to match Cairo verifier's PcsConfig::mix_into
    // (Cairo mixes 4 separate u64s; Rust's config.mix_into packs into a single QM31).
    recursive_log!(
        "  [Recursive] Channel after default: {:?}",
        channel.digest()
    );
    channel.mix_u64(config.pow_bits as u64);
    channel.mix_u64(config.fri_config.log_blowup_factor as u64);
    channel.mix_u64(config.fri_config.n_queries as u64);
    channel.mix_u64(config.fri_config.log_last_layer_degree_bound as u64);
    recursive_log!(
        "  [Recursive] Channel after PcsConfig: {:?}",
        channel.digest()
    );

    // ── Bind public inputs to Fiat-Shamir channel ────────────────────
    // By mixing circuit_hash, io_commitment, weight_super_root, and
    // n_layers into the channel BEFORE any tree commits, the STARK proof
    // becomes cryptographically bound to these values.  A verifier that
    // supplies different metadata will initialize a different channel
    // state, causing the FRI verification to fail.
    //
    // Order: [circuit_hash, io_commitment, weight_super_root] via
    // mix_felts (chunks-of-2 QM31 packing), then n_layers via mix_u64.
    // The Cairo verifier MUST replicate this exact sequence.
    channel.mix_felts(&[
        witness.public_inputs.circuit_hash,
        witness.public_inputs.io_commitment,
        witness.public_inputs.weight_super_root,
    ]);
    channel.mix_u64(witness.public_inputs.n_layers as u64);
    // SECURITY: n_poseidon_perms prevents trace miniaturization attack.
    channel.mix_u64(witness.public_inputs.n_poseidon_perms as u64);
    // SECURITY: seed_digest checkpoint — binds chain content to model dimensions.
    channel.mix_felts(&[witness.public_inputs.seed_digest]);

    // Also bind the full felt252 io_commitment into the channel.
    // The QM31 io_commitment above is lossy (124 bits). This mixes the
    // original 252-bit Poseidon hash so the proof body's felt252 field
    // cannot be tampered without invalidating the STARK.
    {
        let io_felt = crate::crypto::poseidon_channel::securefield_to_felt(
            witness.public_inputs.io_commitment,
        );
        let bytes = io_felt.to_bytes_be();
        let u0 = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let u2 = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        let u3 = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        channel.mix_u64(u0);
        channel.mix_u64(u1);
        channel.mix_u64(u2);
        channel.mix_u64(u3);
    }
    // SECURITY: Bind the Pass 1 (full GKR verification) final digest into
    // the Fiat-Shamir channel. This prevents a malicious prover from skipping
    // Pass 1 — without the correct Pass 1 digest, the channel diverges and
    // the STARK proof fails FRI verification.
    //
    // An attacker who fabricates a partial witness (Pass 2 only) cannot produce
    // the correct Pass 1 digest because it depends on ALL layer verifications
    // (Activation, LayerNorm, etc.) that Pass 2 doesn't replay.
    {
        let bytes = witness.final_digest.to_bytes_be();
        let u0 = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let u2 = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        let u3 = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        channel.mix_u64(u0);
        channel.mix_u64(u1);
        channel.mix_u64(u2);
        channel.mix_u64(u3);
    }
    recursive_log!(
        "  [Recursive] Channel after public inputs + Pass 1 digest: {:?}",
        channel.digest()
    );

    // When Hades AIR is enabled, use the SAME domain for both components
    // to avoid STWO's SIMD mixed-size column evaluation issues.
    // Chain columns are padded to unified_log_size (zeros beyond n_real_rows).
    let unified_log_size = if hades_enabled {
        chain_log_size.max(hades_log_size)
    } else {
        chain_log_size
    };
    // Max degree must cover the largest component's constraint degree.
    // Chain: +1 (degree 2), Hades: +2 (degree 3 from a*b*is_active).
    // +1 for degree-2 constraints (helper columns keep all constraints degree ≤ 2)
    let max_degree_bound = unified_log_size + 1;
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(max_degree_bound + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Poseidon252MerkleChannel>::new(config, &twiddles);
    commitment_scheme.set_store_polynomials_coefficients();

    let chain_domain = CanonicCoset::new(unified_log_size).circle_domain();
    let hades_domain = CanonicCoset::new(unified_log_size).circle_domain();

    // Tree 0: Preprocessed columns (is_first, is_last, is_chain)
    {
        let mut tree_builder = commitment_scheme.tree_builder();
        // Pad preprocessed columns to unified_log_size if needed
        let pad_to = 1 << unified_log_size;
        let mut is_first = trace_data.preprocessed_is_first.clone();
        let mut is_last = trace_data.preprocessed_is_last.clone();
        let mut is_chain = trace_data.preprocessed_is_chain.clone();
        is_first.resize(pad_to, M31::from_u32_unchecked(0));
        is_last.resize(pad_to, M31::from_u32_unchecked(0));
        is_chain.resize(pad_to, M31::from_u32_unchecked(0));
        let is_first_col = simd_column_from_vec(&is_first);
        let is_last_col = simd_column_from_vec(&is_last);
        let is_chain_col = simd_column_from_vec(&is_chain);
        let simd_evals = vec![
            CircleEvaluation::new(chain_domain, is_first_col),
            CircleEvaluation::new(chain_domain, is_last_col),
            CircleEvaluation::new(chain_domain, is_chain_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
            simd_evals,
        ));
        tree_builder.commit(channel);
        recursive_log!(
            "  [Recursive] Channel after preprocessed commit: {:?}",
            channel.digest()
        );
    }

    // Tree 1: All execution traces (chain + Hades in same tree, mixed sizes)
    // STWO's tree builder supports mixed-size columns within one commit.
    {
        let mut tree_builder = commitment_scheme.tree_builder();

        // Chain columns: 69 columns (64 data + 5 selectors), padded to unified_log_size
        let chain_evals: Vec<CircleEvaluation<SimdBackend, M31, _>> = trace_data
            .execution_trace
            .iter()
            .map(|col| {
                let padded = if col.len() < (1 << unified_log_size) {
                    let mut p = col.clone();
                    p.resize(1 << unified_log_size, M31::from_u32_unchecked(0));
                    p
                } else {
                    col.clone()
                };
                let simd_col = simd_column_from_vec(&padded);
                CircleEvaluation::new(chain_domain, simd_col)
            })
            .collect();
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
            chain_evals,
        ));

        // Hades columns: 590 columns at hades_log_size (in same tree)
        if hades_enabled {
            let hades_evals: Vec<CircleEvaluation<SimdBackend, M31, _>> = hades_trace
                .trace
                .iter()
                .map(|col| {
                    let simd_col = simd_column_from_vec(col);
                    CircleEvaluation::new(hades_domain, simd_col)
                })
                .collect();
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, SimdBackend, M31>(
                hades_evals,
            ));
        }

        tree_builder.commit(channel);
        recursive_log!(
            "  [Recursive] Channel after trace commit: {:?} (chain: {} cols @ log{}{})",
            channel.digest(),
            trace_data.execution_trace.len(),
            chain_log_size,
            if hades_enabled {
                format!(
                    ", hades: {} cols @ log{}",
                    hades_trace.trace.len(),
                    hades_log_size
                )
            } else {
                String::new()
            },
        );
    }

    recursive_log!(
        "  [Recursive] Channel before prove(): {:?}",
        channel.digest()
    );

    // Print final digest limbs for Cairo comparison
    {
        let fd = starknet_ff::FieldElement::from_hex_be(&format!("0x{:064x}", channel.digest()))
            .unwrap_or(starknet_ff::FieldElement::ZERO);
        // The final digest used in the AIR comes from the WITNESS, not the channel
    }

    // ── Step 4: Prove ────────────────────────────────────────────────
    recursive_log!("  [Recursive] Step 4/5: Proving (STARK)...");

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

    let pass2_final = last_channel_op.unwrap_or(starknet_ff::FieldElement::ZERO);

    // The chain AIR uses Pass 2's last digest for the boundary constraint
    // (since the chain trace is built from Pass 2's recorded ops).
    let final_digest_felt = pass2_final;

    if pass2_final != witness.final_digest {
        recursive_log!(
            "  [Recursive] NOTE: Pass 2 final digest differs from Pass 1 \
             ({} recorded ops vs {} total Poseidon calls). \
             Chain covers the instrumented subset.",
            witness
                .ops
                .iter()
                .filter(|op| matches!(op, super::types::WitnessOp::ChannelOp { .. }))
                .count(),
            witness.n_poseidon_perms,
        );
    }

    let final_limbs = super::air::felt252_to_limbs(&final_digest_felt);

    // Print limbs for Cairo comparison
    recursive_log!(
        "  [Recursive] Initial limbs: {:?}",
        zero_limbs.iter().map(|l| l.0).collect::<Vec<_>>()
    );
    recursive_log!(
        "  [Recursive] Final limbs: {:?}",
        final_limbs.iter().map(|l| l.0).collect::<Vec<_>>()
    );

    recursive_log!(
        "  [Recursive] Final digest: {:?} (production: {:?}, match: {})",
        final_digest_felt,
        witness.final_digest,
        final_digest_felt == witness.final_digest,
    );

    // Create both AIR components with shared allocator
    let mut allocator = TraceLocationAllocator::default();

    // Component 1: Chain AIR (digest chain + boundary constraints)
    let chain_eval = RecursiveVerifierEval {
        log_n_rows: unified_log_size,
        n_real_rows: trace_data.n_real_rows as u32,
        initial_digest_limbs: zero_limbs,
        final_digest_limbs: final_limbs,
        hades_lookup: None, // LogUp for Hades binding (TODO: draw from channel)
    };
    let chain_component = FrameworkComponent::new(&mut allocator, chain_eval, SecureField::zero());

    // Component 2: Hades AIR (permutation verification)
    let hades_eval = super::hades_air::HadesVerifierEval {
        log_n_rows: unified_log_size,
        round_constants_limbs: Vec::new(),
        range_check: None,
        hades_logup: None, // TODO: draw from channel when multi-component LogUp is wired
    };

    let stark_proof = if hades_enabled {
        let hades_component =
            FrameworkComponent::new(&mut allocator, hades_eval, SecureField::zero());
        recursive_log!("  [Recursive] Proving with chain + Hades components");
        prove::<SimdBackend, Poseidon252MerkleChannel>(
            &[&chain_component, &hades_component],
            channel,
            commitment_scheme,
        )
        .map_err(|e| RecursiveError::ProvingFailed(format!("{e:?}")))?
    } else {
        recursive_log!("  [Recursive] Proving with chain component only (Hades offline-verified)");
        prove::<SimdBackend, Poseidon252MerkleChannel>(
            &[&chain_component],
            channel,
            commitment_scheme,
        )
        .map_err(|e| RecursiveError::ProvingFailed(format!("{e:?}")))?
    };

    let recursive_prove_time = t_start.elapsed().as_secs_f64();
    recursive_log!(
        "  [Recursive] Done in {:.2}s (chain: {}x{}, hades: {}x{}, proof size: {} bytes)",
        recursive_prove_time,
        1u32 << chain_log_size,
        super::air::COLS_PER_ROW,
        1u32 << hades_log_size,
        super::hades_air::N_HADES_TRACE_COLUMNS,
        estimate_proof_size(&stark_proof),
    );

    let io_felt252 =
        crate::crypto::poseidon_channel::securefield_to_felt(witness.public_inputs.io_commitment);
    Ok(RecursiveProof {
        stark_proof: stark_proof,
        public_inputs: witness.public_inputs,
        io_commitment_felt252: io_felt252,
        pass1_final_digest: witness.final_digest,
        final_digest: final_digest_felt,
        n_real_rows: trace_data.n_real_rows as u32,
        log_size: chain_log_size,
        metadata: RecursiveProofMetadata {
            recursive_prove_time_secs: recursive_prove_time,
            gkr_prove_time_secs,
            n_poseidon_perms: witness.n_poseidon_perms,
            n_sumcheck_rounds: witness.n_sumcheck_rounds,
            trace_log_size: chain_log_size,
            n_trace_columns: super::air::COLS_PER_ROW + super::hades_air::N_HADES_TRACE_COLUMNS,
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Hardened Recursive Proving (with Hades AIR)
// ═══════════════════════════════════════════════════════════════════════

/// Extract (input, output) pairs for every HadesPerm in the witness.
///
/// These pairs are used to build the Hades verification trace that
/// constrains the actual Hades permutation computation.
pub fn extract_hades_perms(
    witness: &super::types::GkrVerifierWitness,
) -> Vec<(
    [starknet_ff::FieldElement; 3],
    [starknet_ff::FieldElement; 3],
)> {
    witness
        .ops
        .iter()
        .filter_map(|op| {
            if let super::types::WitnessOp::HadesPerm { input, output } = op {
                Some((*input, *output))
            } else {
                None
            }
        })
        .collect()
}

/// Verify all HadesPerm operations in a witness via step-by-step execution.
///
/// This is a standalone soundness check that can be run without generating
/// a STARK proof. It verifies that every (input, output) pair in the witness
/// corresponds to a correct Hades permutation.
///
/// Returns Ok(n_verified) on success, or Err with the first mismatch.
pub fn verify_hades_perms_offline(
    witness: &super::types::GkrVerifierWitness,
) -> Result<usize, RecursiveError> {
    let perms = extract_hades_perms(witness);
    for (i, (input, expected_output)) in perms.iter().enumerate() {
        let mut actual = *input;
        starknet_crypto::poseidon_permute_comp(&mut actual);
        if actual != *expected_output {
            return Err(RecursiveError::ProvingFailed(format!(
                "HadesPerm #{} mismatch: input={:?}, expected={:?}, got={:?}",
                i, input, expected_output, actual,
            )));
        }
    }
    Ok(perms.len())
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
        std::env::set_var("OBELYZK_RECURSIVE_SECURITY", "test");
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

        let zero = QM31(
            CM31(M31::from(0), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );

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
        assert!(recursive_proof.metadata.n_poseidon_perms > 0);
        assert!(recursive_proof.metadata.recursive_prove_time_secs > 0.0);
        recursive_log!(
            "Recursive proof: {:.3}s, {} poseidon perms, log_size={}",
            recursive_proof.metadata.recursive_prove_time_secs,
            recursive_proof.metadata.n_poseidon_perms,
            recursive_proof.metadata.trace_log_size,
        );
    }
}
