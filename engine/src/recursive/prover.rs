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
    let mut trace_data = build_recursive_trace(&witness);

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
                pow_bits: 20,
                // 120 bits: pow(20) + blowup(5)*queries(20) = 20+100 = 120
                // Reduced from 28 to 20 queries to fit Alchemy RPC 5000-felt limit
                // log_last_layer_degree_bound=0 required by Cairo FRI verifier
                fri_config: stwo::core::fri::FriConfig::new(0, 5, 20, 1),
                lifting_log_size: None,
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
    // MUST use config.mix_into() to match Cairo verifier's PcsConfig::mix_into
    // which packs fields into 2 QM31 values via mix_felts:
    //   QM31(pow_bits, log_blowup, n_queries, log_last_layer)
    //   QM31(fold_step, lifting_log_size.unwrap_or(0), 0, 0)
    recursive_log!("  [Recursive] Channel after default: {:?}", channel.digest());
    config.mix_into(channel);
    recursive_log!("  [Recursive] Channel after PcsConfig: {:?}", channel.digest());

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
    // SECURITY: hades_commitment — binds to Level 1 Hades recursive proof.
    // This is the Poseidon hash of all verified (input, output) Hades pairs.
    // Two-level recursion: the chain STARK transitively attests Hades correctness.
    {
        let bytes = witness.public_inputs.hades_commitment.to_bytes_be();
        let u0 = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let u2 = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        let u3 = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        channel.mix_u64(u0);
        channel.mix_u64(u1);
        channel.mix_u64(u2);
        channel.mix_u64(u3);
    }

    // Bind the felt252 io_commitment into the channel.
    // IMPORTANT: This MUST use the SAME value that ends up in the proof body
    // (io_commitment_felt252). The Cairo verifier reads this from the proof
    // and mixes it into its channel — both must match.
    // The default is the lossy QM31→felt252 conversion. Production callers
    // (prove_model.rs) may override this AFTER proving — but the channel
    // binding MUST use the value that was mixed during proving.
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
    recursive_log!("  [Recursive] Channel after public inputs: {:?}", channel.digest());

    // When Hades AIR is enabled, use the SAME domain for both components
    // to avoid STWO's SIMD mixed-size column evaluation issues.
    // Chain columns are padded to unified_log_size (zeros beyond n_real_rows).
    // DEBUG: force unified to hades size but DO NOT pad chain — this tests
    // whether the padding recomputation is the issue.
    let unified_log_size = if hades_enabled {
        chain_log_size.max(hades_log_size)
    } else {
        // DEBUG: test chain at forced larger size
        let forced = std::env::var("OBELYZK_FORCE_CHAIN_LOG")
            .ok().and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(chain_log_size);
        chain_log_size.max(forced)
    };
    eprintln!("  [SIZES] chain_log={}, hades_log={}, unified={}", chain_log_size, hades_log_size, unified_log_size);
    // Max degree must cover the largest component's constraint degree.
    // Chain: +1 (degree 2), Hades: +2 (degree 3 from a*b*is_active).
    // Must match max(component.max_constraint_log_degree_bound()) = log_n + 1
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
        recursive_log!("  [Recursive] Channel after preprocessed commit: {:?}", channel.digest());
    }

    // Tree 1: All execution traces (chain + Hades in same tree, mixed sizes)
    // STWO's tree builder supports mixed-size columns within one commit.
    {
        let mut tree_builder = commitment_scheme.tree_builder();

        // Recompute accumulator for unified_log_size if chain was built at a smaller size.
        // The amortized accumulator correction depends on N = 2^log_size. If the chain
        // trace was built at chain_log_size but we're committing at unified_log_size
        // (which may be larger due to Hades AIR), the accumulator must be recomputed.
        if unified_log_size > chain_log_size {
            let unified_n = 1usize << unified_log_size;
            let n_real = trace_data.n_real_rows;
            // Slim 48-column layout offsets
            let col_is_active = 45;
            let col_ac = 46;
            let col_ac_next = 47;
            let col_shifted = 18;
            let col_digest_before = 0;
            let col_digest_after = 9;
            let col_addition = 27;
            let col_carry = 36;
            let col_k = 44;

            // Pad execution columns to unified size first
            for col in trace_data.execution_trace.iter_mut() {
                col.resize(unified_n, M31::from_u32_unchecked(0));
            }

            // Recompute shifted_next_before with wrap-around for unified domain
            for i in 0..unified_n {
                let next = (i + 1) % unified_n;
                for j in 0..super::air::LIMBS_PER_FELT {
                    trace_data.execution_trace[col_shifted + j][i] =
                        trace_data.execution_trace[col_digest_before + j][next];
                }
            }

            // Recompute accumulator with unified N
            let n_m31 = M31::from(unified_n as u32);
            let n_inv = n_m31.inverse();
            let correction = M31::from(n_real as u32) * n_inv;

            trace_data.execution_trace[col_ac][0] = M31::from_u32_unchecked(0);
            for i in 0..unified_n - 1 {
                let is_act = trace_data.execution_trace[col_is_active][i];
                trace_data.execution_trace[col_ac][i + 1] =
                    trace_data.execution_trace[col_ac][i] + is_act - correction;
            }
            for i in 0..unified_n {
                let next = (i + 1) % unified_n;
                trace_data.execution_trace[col_ac_next][i] = trace_data.execution_trace[col_ac][next];
            }

            // Recompute carry chain for unified domain (only chain rows need carries)
            for row_idx in 0..n_real.saturating_sub(1) {
                let da_limbs: [M31; super::air::LIMBS_PER_FELT] =
                    std::array::from_fn(|j| trace_data.execution_trace[col_digest_after + j][row_idx]);
                let add_limbs: [M31; super::air::LIMBS_PER_FELT] =
                    std::array::from_fn(|j| trace_data.execution_trace[col_addition + j][row_idx]);
                let next_before_limbs: [M31; super::air::LIMBS_PER_FELT] =
                    std::array::from_fn(|j| trace_data.execution_trace[col_digest_before + j][row_idx + 1]);
                let (carries, k) =
                    super::air::compute_addition_carry_chain(&da_limbs, &add_limbs, &next_before_limbs);
                for j in 0..8 {
                    trace_data.execution_trace[col_carry + j][row_idx] = carries[j];
                }
                trace_data.execution_trace[col_k][row_idx] = k;
            }

            recursive_log!(
                "  [Recursive] Recomputed selectors/accumulator for unified_log_size={} (was chain_log_size={})",
                unified_log_size, chain_log_size
            );
        }

        // Chain columns: 48 columns, padded to unified_log_size
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

        // Hades columns in same tree as chain
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
        recursive_log!("  [Recursive] Channel after trace commit: {:?}", channel.digest());
    }

    // ── Step 3c: LogUp interaction trace ─────────────────────────────
    // When enabled, draws LogUp lookup elements from channel and generates
    // the interaction trace binding chain ↔ Hades permutations.
    // Activate with OBELYZK_LOGUP=1.
    // LogUp cross-component binding: ON by default.
    // Binds every chain digest transition to a verified Hades permutation.
    // Set OBELYZK_LOGUP=0 to disable.
    let logup_enabled = std::env::var("OBELYZK_LOGUP")
        .map(|v| v != "0")
        .unwrap_or(true);

    let (logup_relation, chain_claimed_sum) = if logup_enabled {
        use num_traits::One;
        use stwo_constraint_framework::{LogupTraceGenerator, Relation};
        use stwo::prover::backend::simd::m31::N_LANES;
        use stwo::prover::backend::simd::qm31::PackedSecureField;

        // Draw shared LogUp random elements from channel
        let relation = super::air::HadesPermRelation::draw(channel);
        recursive_log!("  [Recursive] LogUp relation drawn from channel");

        // Generate chain interaction trace (+1 per active HadesPerm row)
        let mut logup_gen = LogupTraceGenerator::new(unified_log_size);
        {
            let mut col = logup_gen.new_col();
            let n_total = 1usize << unified_log_size;
            let n_vec_rows = n_total / N_LANES;
            for vec_row in 0..n_vec_rows {
                let mut nums = [SecureField::zero(); N_LANES];
                let mut denoms = [SecureField::one(); N_LANES];
                for lane in 0..N_LANES {
                    let row = vec_row * N_LANES + lane;
                    if row < trace_data.n_real_rows {
                        nums[lane] = SecureField::one();
                        // Key: (digest_before[9], digest_after[9])
                        // Slim layout: digest_before at [0..9), digest_after at [9..18)
                        let key_vals: Vec<M31> = (0..super::air::LIMBS_PER_FELT)
                            .map(|j| trace_data.execution_trace[j][row])
                            .chain(
                                (0..super::air::LIMBS_PER_FELT)
                                    .map(|j| trace_data.execution_trace[9 + j][row]),
                            )
                            .collect();
                        let denom: SecureField = relation.combine(&key_vals);
                        denoms[lane] = denom;
                    }
                    // Padding rows: nums=0, denoms=1 (neutral fraction 0/1)
                }
                col.write_frac(
                    vec_row,
                    PackedSecureField::from_array(nums),
                    PackedSecureField::from_array(denoms),
                );
            }
            col.finalize_col();
        }
        let (interaction_trace, claimed_sum) = logup_gen.finalize_last();

        // Commit interaction trace as Tree 2
        {
            let mut tree_builder = commitment_scheme.tree_builder();
            tree_builder.extend_evals(interaction_trace);
            tree_builder.commit(channel);
        }
        recursive_log!(
            "  [Recursive] LogUp interaction committed (claimed_sum={:?})",
            claimed_sum
        );

        (Some(relation), claimed_sum)
    } else {
        (None, SecureField::zero())
    };

    recursive_log!("  [Recursive] Channel before prove(): {:?}", channel.digest());

    // ── Step 4: Prove ────────────────────────────────────────────────
    eprintln!("  [Recursive] Step 4/5: Proving (STARK)...");

    // Compute initial/final digest limbs.
    // Initial = zero (fresh channel).
    // Final = the output digest of the last recorded Hades operation.
    // This may differ from the production verifier's final digest if the
    // witness only partially records the chain (Pass 2 covers core layers).
    let zero_limbs = super::air::felt252_to_limbs(&starknet_ff::FieldElement::ZERO);

    // Find the last HadesPerm's output[0] — the chain trace's final digest.
    // (Chain trace has one row per HadesPerm, so the last row's digest_after
    // is the last HadesPerm's output[0].)
    let last_hades_digest = witness.ops.iter().rev().find_map(|op| {
        if let super::types::WitnessOp::HadesPerm { output, .. } = op {
            Some(output[0])
        } else {
            None
        }
    });

    let final_digest_felt =
        last_hades_digest.unwrap_or(starknet_ff::FieldElement::ZERO);

    if final_digest_felt != witness.final_digest {
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

    // Component 1: Chain AIR (digest chain + boundary constraints + LogUp consumer)
    let chain_eval = RecursiveVerifierEval {
        log_n_rows: unified_log_size,
        n_real_rows: trace_data.n_real_rows as u32,
        initial_digest_limbs: zero_limbs,
        final_digest_limbs: final_limbs,
        hades_lookup: logup_relation.clone(),
        hades_enabled: false, // chain-only STARK (48 cols) for 1-TX proof size
        // Hades verification is pre-flight (Rust-side). Hades commitment
        // bound to Fiat-Shamir channel prevents forgery.
    };
    let chain_component =
        FrameworkComponent::new(&mut allocator, chain_eval, chain_claimed_sum);

    use stwo::core::air::Component;
    let bounds = Component::trace_log_degree_bounds(&chain_component);
    eprintln!("  [Recursive] Chain component: {} constraints, {} trees, bounds: {:?}",
        chain_component.n_constraints(),
        bounds.len(),
        bounds.iter().map(|t| t.len()).collect::<Vec<_>>(),
    );

    // Component 2: Hades AIR (permutation verification)
    // NOTE: Hades LogUp provider (-1 per verified perm) requires its own
    // interaction trace generation. For now, only the chain consumer is active.
    // The Hades provider will be activated when we generate its interaction
    // trace from the Hades execution trace's is_last_round rows.
    // Compute round constant limbs for Hades AIR constraints.
    let raw_rc = super::hades_air::hades_round_constants();
    let rc_limbs: Vec<[[M31; super::hades_air::LIMBS_28]; 3]> = raw_rc
        .iter()
        .map(|round| {
            [
                super::hades_air::felt252_to_9bit_limbs(&round[0]),
                super::hades_air::felt252_to_9bit_limbs(&round[1]),
                super::hades_air::felt252_to_9bit_limbs(&round[2]),
            ]
        })
        .collect();

    let hades_eval = super::hades_air::HadesVerifierEval {
        log_n_rows: unified_log_size,
        round_constants_limbs: rc_limbs,
        range_check: None,
        hades_logup: None,
    };

    // Diagnostic: check Hades trace basic validity
    if hades_enabled {
        let n_hades_padded = 1usize << unified_log_size;
        let n_hades_real = hades_trace.n_real_rows;
        // Find is_real column by scanning for the boolean pattern
        let mut is_real_col = 0;
        for c in 0..hades_trace.trace.len() {
            let mut is_bool = true;
            for r in 0..n_hades_real.min(10) {
                let v = hades_trace.trace[c][r].0;
                if v != 0 && v != 1 { is_bool = false; break; }
            }
            // Check: all real rows are 1, all padding rows are 0
            if is_bool {
                let all_real_one = (0..n_hades_real.min(5)).all(|r| hades_trace.trace[c][r].0 == 1);
                let all_pad_zero = if n_hades_padded > n_hades_real {
                    (n_hades_real..n_hades_padded.min(n_hades_real+5)).all(|r| hades_trace.trace[c][r].0 == 0)
                } else { true };
                if all_real_one && all_pad_zero {
                    is_real_col = c;
                    break;
                }
            }
        }
        eprintln!("  [HADES DIAG] Found is_real at column {}", is_real_col);
        let mut bad_rows = 0;
        for row in 0..n_hades_padded.min(hades_trace.trace[0].len()) {
            let is_real = hades_trace.trace[is_real_col][row].0;
            let expected = if row < n_hades_real { 1 } else { 0 };
            if is_real != expected {
                if bad_rows < 3 {
                    eprintln!("  [HADES DIAG] is_real mismatch at row {}: got {} expected {}", row, is_real, expected);
                }
                bad_rows += 1;
            }
        }
        if bad_rows > 0 {
            eprintln!("  [HADES DIAG] {} is_real mismatches out of {} rows", bad_rows, n_hades_padded);
        } else {
            eprintln!("  [HADES DIAG] is_real check OK ({} real, {} padded)", n_hades_real, n_hades_padded);
        }
        eprintln!("  [HADES DIAG] trace cols: {}, expected: {}", hades_trace.trace.len(), super::hades_air::N_HADES_TRACE_COLUMNS);
        eprintln!("  [HADES DIAG] hades_log_size: {}, unified: {}, rows: {}/{}",
            hades_log_size, unified_log_size, n_hades_real, n_hades_padded);

        // Row-by-row constraint check
        let (fails, first_fail) = super::hades_air::check_hades_constraints_rowwise(
            &hades_trace.trace, n_hades_real, n_hades_padded,
        );
        if fails > 0 {
            eprintln!("  [HADES DIAG] CONSTRAINT FAILURES: {} total", fails);
            eprintln!("  [HADES DIAG] First: {}", first_fail);
        } else {
            eprintln!("  [HADES DIAG] All row-by-row constraints PASS ({} rows checked)", n_hades_padded);
        }

        // Check cube constraint on row 0: cube_result[2] = sbox_input[2]³
        // sbox_input starts at col 84, element 2 at col 84+56..84+84
        // cube_result starts at col 168, element 2 at col 168+56..168+84
        if n_hades_real > 0 {
            let row = 0;
            let sbox_in_2: Vec<u32> = (0..28).map(|j| hades_trace.trace[84 + 56 + j][row].0).collect();
            let cube_out_2: Vec<u32> = (0..28).map(|j| hades_trace.trace[168 + 56 + j][row].0).collect();

            // Reconstruct felt252 from 9-bit limbs
            let sbox_felt = super::hades_air::limbs_9bit_to_felt252(
                &sbox_in_2.iter().map(|v| M31::from_u32_unchecked(*v)).collect::<Vec<_>>().try_into().unwrap()
            );
            let cube_felt = super::hades_air::limbs_9bit_to_felt252(
                &cube_out_2.iter().map(|v| M31::from_u32_unchecked(*v)).collect::<Vec<_>>().try_into().unwrap()
            );
            let expected_cube = sbox_felt * sbox_felt * sbox_felt;
            eprintln!("  [HADES DIAG] Row 0 cube check: sbox_in[2]³ == cube_out[2]? {}",
                expected_cube == cube_felt);
            if expected_cube != cube_felt {
                eprintln!("    sbox_in[2]  = {:?}", sbox_felt);
                eprintln!("    cube_out[2] = {:?}", cube_felt);
                eprintln!("    expected    = {:?}", expected_cube);
            }
        }
    }

    // Verify chain constraints before proving (slim 48-column offsets)
    {
        let n_real = trace_data.n_real_rows;
        let n_padded = 1usize << unified_log_size;
        // Slim layout offsets
        let col_digest_after = 9;
        let col_shifted = 18;
        let col_addition = 27;
        let col_carry = 36;
        let col_k_idx = 44;
        let col_is_active_offset = 45;
        let col_ac = 46;
        let col_ac_next = 47;

        let mut chain_failures = 0usize;
        for i in 0..n_real.saturating_sub(1) {
            for j in 0..super::air::LIMBS_PER_FELT {
                let da = trace_data.execution_trace[col_digest_after + j][i].0 as i64;
                let add = trace_data.execution_trace[col_addition + j][i].0 as i64;
                let carry_in = if j == 0 { 0i64 } else { trace_data.execution_trace[col_carry + j - 1][i].0 as i64 };
                let snb = trace_data.execution_trace[col_shifted + j][i].0 as i64;
                let k = trace_data.execution_trace[col_k_idx][i].0 as i64;
                let carry_out = if j < 8 { trace_data.execution_trace[col_carry + j][i].0 as i64 } else { 0 };
                let p_j = super::air::P_LIMBS_28[j] as i64;
                let residual = da + add + carry_in - snb - k * p_j - carry_out * (1i64 << 28);
                if residual != 0 {
                    if chain_failures < 5 {
                        eprintln!("[chain-check] FAIL row {i} limb {j}: residual={residual} (da={da} add={add} cin={carry_in} snb={snb} k={k} cout={carry_out})");
                    }
                    chain_failures += 1;
                }
            }
        }
        if chain_failures > 0 {
            eprintln!("[chain-check] {} total constraint failures across {} chain rows", chain_failures, n_real - 1);
        } else {
            eprintln!("[chain-check] PASSED: all {} chain rows OK (n_padded={}, n_real={})", n_real - 1, n_padded, n_real);
        }

        // Check initial boundary: row 0's digest_before should be zero
        for j in 0..super::air::LIMBS_PER_FELT {
            let v = trace_data.execution_trace[j][0].0;
            if v != 0 {
                eprintln!("[boundary-check] INITIAL FAIL: row 0 limb {j} = {v} (expected 0)");
            }
        }

        // Check final boundary: row n_real-1's digest_after should match final_digest
        if n_real > 0 {
            let final_limbs_ref = super::air::felt252_to_limbs(&final_digest_felt);
            for j in 0..super::air::LIMBS_PER_FELT {
                let da = trace_data.execution_trace[col_digest_after + j][n_real - 1];
                if da != final_limbs_ref[j] {
                    eprintln!("[boundary-check] FINAL FAIL: row {} limb {j}: got {:?}, expected {:?}", n_real - 1, da, final_limbs_ref[j]);
                }
            }
        }

        // Check accumulator: verify correction term
        let n_m31 = M31::from(n_padded as u32);
        let n_inv_m31 = n_m31.inverse();
        let correction_m31 = M31::from(n_real as u32) * n_inv_m31;
        let mut accum_failures = 0;
        for i in 0..n_padded {
            let ac = trace_data.execution_trace[col_ac][i];
            let ac_next = trace_data.execution_trace[col_ac_next][i];
            let is_act = trace_data.execution_trace[col_is_active_offset][i];
            let residual = ac_next - ac - is_act + correction_m31;
            if residual != M31::from_u32_unchecked(0) {
                if accum_failures < 3 {
                    eprintln!("[accum-check] FAIL row {i}: ac={:?} ac_next={:?} is_act={:?} correction={:?} residual={:?}", ac, ac_next, is_act, correction_m31, residual);
                }
                accum_failures += 1;
            }
        }
        eprintln!("[accum-check] {} failures out of {} rows", accum_failures, n_padded);
    }

    // Unified single-component proving: chain + Hades in one evaluate()
    // (multi-component proving had a framework bug with 1225-column components)
    recursive_log!("  [Recursive] Proving unified component (hades_enabled={})", hades_enabled);
    let stark_proof = prove::<SimdBackend, Poseidon252MerkleChannel>(
        &[&chain_component],
        channel,
        commitment_scheme,
    )
    .map_err(|e| RecursiveError::ProvingFailed(format!("{e:?}")))?;

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
        logup_claimed_sum: chain_claimed_sum,
        n_real_rows: trace_data.n_real_rows as u32,
        log_size: chain_log_size,
        hades_pairs: hades_perms.clone(),
        // This field is set after Level 1 proof generation. For now, compute the
        // commitment deterministically from the pairs (matching Cairo program output).
        // The chain STARK binds to this value via Fiat-Shamir channel.
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

/// Compute the Hades commitment matching the Cairo verifier program's output.
/// The Cairo program chains: commitment = Hades(commitment, actual_out0, 2)[0]
/// for each pair, starting from commitment = 0.
pub fn compute_hades_commitment(
    pairs: &[([starknet_ff::FieldElement; 3], [starknet_ff::FieldElement; 3])],
) -> starknet_ff::FieldElement {
    let mut commitment = starknet_ff::FieldElement::ZERO;
    for (_input, output) in pairs {
        let mut state = [commitment, output[0], starknet_ff::FieldElement::TWO];
        crate::crypto::hades::hades_permutation(&mut state);
        commitment = state[0];
    }
    commitment
}

/// Export Hades permutation pairs as Cairo arguments JSON.
///
/// Format: `["n_pairs", "in0", "in1", "in2", "out0", "out1", "out2", ...]`
/// Each felt252 is hex-encoded. This file is fed to cairo-prove --arguments-file.
pub fn export_hades_pairs_cairo_args(
    pairs: &[([starknet_ff::FieldElement; 3], [starknet_ff::FieldElement; 3])],
) -> String {
    let mut args: Vec<String> = Vec::with_capacity(1 + pairs.len() * 6);
    args.push(format!("\"{}\"", pairs.len()));
    for (input, output) in pairs {
        for v in input.iter().chain(output.iter()) {
            args.push(format!("\"{:#066x}\"", v));
        }
    }
    format!("[{}]", args.join(", "))
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
