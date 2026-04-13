use cairo_air::CairoProof;
use cairo_vm::types::builtin_name::BuiltinName;
use cairo_vm::vm::runners::cairo_runner::CairoRunner;
use log::info;
use stwo_cairo_adapter::adapter::adapt;
use stwo_cairo_adapter::{ProverInput, PublicSegmentContext};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::prover::{prove_cairo, ChannelHash, ProverParameters};
use stwo_cairo_prover::stwo::core::pcs::PcsConfig;
use stwo_cairo_prover::stwo::core::vcs_lifted::blake2_merkle::{
    Blake2sMerkleChannel, Blake2sMerkleHasher,
};

use crate::error::{CairoProveError, Result};

/// Extracts artifacts from a finished cairo runner, to later be used for proving.
///
/// Uses `bootloader_context()` by default (all 11 builtins present), which is required for
/// STARK-in-STARK recursion where the Cairo verifier expects all 11 segments. For standalone
/// Cairo programs that declare fewer builtins, overrides with actual builtins to match the
/// AP stack layout — these programs cannot be used for STARK-in-STARK.
///
/// # Errors
///
/// Returns an error if the adapter fails to process the runner.
pub fn prover_input_from_runner(runner: &CairoRunner) -> Result<ProverInput> {
    info!("Generating input for the prover...");
    let mut input = adapt(runner)
        .map_err(|e| CairoProveError::AdapterError(format!("{:?}", e)))?;

    // If the program declares fewer than 11 builtins, override bootloader_context
    // with actual builtins. This is needed for standalone programs (JSON/verify path)
    // but means the proof cannot be used for STARK-in-STARK recursion.
    let builtins: Vec<BuiltinName> = runner.get_program_builtins().to_vec();
    if builtins.len() < 11 {
        input.public_segment_context = PublicSegmentContext::new(&builtins);
    }

    info!("Input for the prover generated successfully.");
    Ok(input)
}

/// Deduces the preprocessed trace variant needed for the specific execution, and proves.
///
/// # Errors
///
/// Returns an error if proof generation fails.
pub fn prove(input: ProverInput, pcs_config: PcsConfig) -> Result<CairoProof<Blake2sMerkleHasher>> {
    // Always use Canonical (with pedersen) to match the default Cairo verifier's
    // preprocessed trace root. This ensures STARK-in-STARK compatibility.
    // The pedersen builtin columns are included in the preprocessed trace even if
    // the program doesn't use pedersen — they just have empty evaluations.
    let preprocessed_trace = PreProcessedTraceVariant::Canonical;
    prove_inner(input, preprocessed_trace, pcs_config)
}

/// Prove with GPU backend. Uses GpuBackend for STARK proving (FFT, FRI, Merkle,
/// constraints) while keeping witness generation on SimdBackend (CPU).
///
/// Requires the `gpu` feature to be enabled at compile time, which pulls in the
/// stwo-gpu CUDA runtime.
#[cfg(feature = "gpu")]
pub fn prove_gpu(input: ProverInput, pcs_config: PcsConfig) -> Result<CairoProof<Blake2sMerkleHasher>> {
    use std::sync::Arc;
    use stwo_cairo_prover::stwo::prover::backend::gpu::GpuBackend;
    use stwo_cairo_prover::stwo::prover::backend::simd::SimdBackend;
    use stwo_cairo_prover::stwo::prover::backend::BackendForChannel;
    use stwo_cairo_prover::stwo::prover::pcs::CommitmentTreeProver;
    use stwo_cairo_prover::stwo::prover::poly::circle::CanonicCoset;
    use stwo_cairo_prover::stwo::prover::MaybeOwned;
    use stwo_cairo_prover::prover::{prove_cairo_with_precompute, MAX_CANONICAL_COSET_LOG_SIZE};
    use stwo_cairo_prover::witness::preprocessed_trace::gen_trace;
    use stwo_cairo_prover::witness::utils::BaseColumnPool;

    let preprocessed_trace_variant = match input.public_segment_context[1] {
        true => PreProcessedTraceVariant::Canonical,
        false => PreProcessedTraceVariant::CanonicalWithoutPedersen,
    };

    let prover_params = ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        channel_salt: 0,
        pcs_config,
        preprocessed_trace: preprocessed_trace_variant,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
    };

    let max_domain_size = prover_params.pcs_config.lifting_log_size
        .unwrap_or(MAX_CANONICAL_COSET_LOG_SIZE);

    // Precompute twiddles on GPU
    let twiddles = GpuBackend::precompute_twiddles(
        CanonicCoset::new(max_domain_size).circle_domain().half_coset,
    );

    let preprocessed_trace = Arc::new(prover_params.preprocessed_trace.to_preprocessed_trace());
    let preprocessed_trace_polys =
        GpuBackend::interpolate_columns(gen_trace(preprocessed_trace.clone()), &twiddles);

    let base_column_pool = BaseColumnPool::new();
    let preprocessed_tree = CommitmentTreeProver::<GpuBackend, Blake2sMerkleChannel>::new(
        preprocessed_trace_polys,
        prover_params.pcs_config.fri_config.log_blowup_factor,
        &twiddles,
        prover_params.store_polynomials_coefficients,
        prover_params.pcs_config.lifting_log_size,
        &base_column_pool,
    );

    prove_cairo_with_precompute::<Blake2sMerkleChannel>(
        &base_column_pool,
        &twiddles,
        preprocessed_trace,
        MaybeOwned::Owned(preprocessed_tree),
        input,
        prover_params,
    )
    .map_err(|e| CairoProveError::ProofGeneration(format!("{:?}", e)))
}

fn prove_inner(
    input: ProverInput,
    preprocessed_trace: PreProcessedTraceVariant,
    pcs_config: PcsConfig,
) -> Result<CairoProof<Blake2sMerkleHasher>> {
    let prover_params = ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        channel_salt: 0,
        pcs_config,
        preprocessed_trace,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
    };

    prove_cairo::<Blake2sMerkleChannel>(input, prover_params)
        .map_err(|e| CairoProveError::ProofGeneration(format!("{:?}", e)))
}
