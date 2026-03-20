//! # stwo-ml: ML Inference Proving on Circle STARKs
//!
//! ML-specific proving circuits built on STWO — the fastest STARK prover in the world.
//!
//! ## Feature Flags
//!
//! - `std` (default): Standard library support + STWO prover.
//! - `gpu`: Enable GPU acceleration (kernel source, no runtime).
//! - `cuda-runtime`: Full CUDA execution (requires CUDA toolkit).
//! - `multi-gpu`: Multi-GPU distributed proving.
//! - `tee`: TEE attestation (requires `cuda-runtime`).
//! - `onnx`: ONNX model loading via tract-onnx.
//! - `safetensors`: SafeTensors weight loading.
//! - `model-loading`: Both ONNX + SafeTensors.

#![feature(portable_simd)]
#![feature(once_cell_try)]

use std::sync::atomic::{AtomicBool, Ordering};

/// Global quiet flag: when true, suppresses verbose diagnostic output.
/// Set via `set_quiet(true)` from the CLI binary.
static QUIET: AtomicBool = AtomicBool::new(false);

/// Enable or disable quiet mode globally.
pub fn set_quiet(quiet: bool) {
    QUIET.store(quiet, Ordering::Relaxed);
}

/// Returns true if quiet mode is enabled.
#[inline]
pub fn is_quiet() -> bool {
    QUIET.load(Ordering::Relaxed)
}

/// Global profiling flag: when true, enables per-phase timing in GKR provers.
/// Set via `set_profile(true)` or `STWO_PROFILE=1` env var.
static PROFILE: AtomicBool = AtomicBool::new(false);

/// Enable or disable phase profiling globally.
pub fn set_profile(enabled: bool) {
    PROFILE.store(enabled, Ordering::Relaxed);
}

/// Returns true if phase profiling is enabled.
#[inline]
pub fn is_profile() -> bool {
    PROFILE.load(Ordering::Relaxed)
        || std::env::var("STWO_PROFILE").map_or(false, |v| v == "1" || v == "true")
}

pub mod aggregation;
pub mod backend;
pub mod cairo_serde;
pub mod circuits;
pub mod compiler;
pub mod components;
pub mod crypto;
pub mod gadgets;
pub mod gkr;
pub mod gpu;
pub mod json_serde;
#[cfg(any(feature = "cli", feature = "audit", feature = "server"))]
pub mod kv_state;
pub mod receipt;
pub mod starknet;
pub mod recursive;
pub mod tee;
pub mod weight_cache;

#[cfg(any(feature = "cli", feature = "audit", feature = "server"))]
pub mod privacy;

#[cfg(feature = "audit")]
pub mod audit;

#[cfg(feature = "cuda-runtime")]
pub mod gpu_sumcheck;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "tui")]
pub mod tui;

#[cfg(feature = "multi-gpu")]
pub mod multi_gpu;

#[cfg(feature = "multi-query")]
pub mod gpu_scheduler;

/// Re-export core STWO types used throughout stwo-ml.
pub mod prelude {
    pub use stwo::core::fields::m31::M31;
    pub use stwo::core::fields::qm31::QM31;
    pub use stwo::core::pcs::PcsConfig;
    pub use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    pub use stwo::prover::backend::simd::SimdBackend;
    pub use stwo::prover::backend::BackendForChannel;

    pub use crate::aggregation::{compute_io_commitment, compute_io_commitment_packed};
    pub use crate::aggregation::{
        prove_model_aggregated, prove_model_aggregated_auto, prove_model_aggregated_onchain,
        prove_model_aggregated_onchain_auto, prove_model_aggregated_onchain_auto_cached,
        prove_model_aggregated_onchain_gkr, prove_model_aggregated_onchain_gkr_auto,
        prove_model_aggregated_onchain_logup_gkr, prove_model_aggregated_onchain_logup_gkr_auto,
        prove_model_aggregated_with, verify_aggregated_model_proof,
        verify_aggregated_model_proof_onchain, GkrBatchData,
    };
    pub use crate::backend::{BackendInfo, MatMulReduction, ZkmlOps};
    pub use crate::cairo_serde::DirectProofMetadata;
    #[cfg(feature = "multi-gpu")]
    pub use crate::compiler::chunked::{
        prove_model_chunked_multi_gpu, prove_model_chunked_multi_gpu_with_metrics,
    };
    pub use crate::compiler::graph::{ComputationGraph, GraphBuilder, GraphWeights};
    pub use crate::compiler::inspect::{summarize_graph, summarize_model, ModelSummary};
    pub use crate::compiler::onnx::{
        build_mlp, build_mlp_with_weights, build_transformer, build_transformer_block,
        generate_weights_for_graph, load_onnx, ModelMetadata, OnnxError, OnnxModel,
        TransformerConfig,
    };
    pub use crate::compiler::prove::{prove_model, prove_model_auto, prove_model_with};
    pub use crate::components::activation::ActivationType;
    pub use crate::components::matmul::M31Matrix;
    pub use crate::components::matmul::{
        prove_matmul_sumcheck_auto, prove_matmul_sumcheck_onchain,
        prove_matmul_sumcheck_onchain_auto, MatMulSumcheckProofOnChain, RoundPoly,
    };
    pub use crate::crypto::mle_opening::{
        commit_mle, commit_mle_root_only, prove_mle_opening, MleOpeningProof,
    };
    pub use crate::crypto::poseidon_channel::PoseidonChannel;
    pub use crate::gpu::GpuModelProver;
    pub use crate::gpu::ProofWithAttestation;
    #[cfg(feature = "multi-gpu")]
    pub use crate::multi_gpu::{
        ChunkWorkload, DeviceAssignment, DeviceGuard, DeviceProvingStat, GpuDeviceInfo,
        MultiGpuError, MultiGpuExecutor, MultiGpuProvingResult,
    };
    pub use crate::receipt::{prove_receipt, prove_receipt_batch, prove_receipt_batch_auto};
    #[cfg(feature = "multi-gpu")]
    pub use crate::starknet::prove_for_starknet_direct_multi_gpu;
    pub use crate::starknet::{
        build_starknet_proof_direct, build_starknet_proof_onchain_with_tee,
        build_starknet_proof_with_tee, prove_for_starknet_direct,
        prove_for_starknet_onchain_cached, DirectStarknetProof,
    };
    pub use crate::tee::{
        detect_tee_capability, verify_attestation, SecurityLevel, TeeAttestation, TeeCapability,
        TeeModelProver,
    };
    pub use crate::weight_cache::{
        compute_weight_fingerprint, save_shared_cache, shared_cache, shared_cache_for_model,
        shared_cache_for_model_mmap, shared_cache_for_model_validated, shared_cache_from_file,
        CachedWeight, SharedWeightCache, WeightCommitmentCache,
    };
}

/// Re-export GPU backend when available.
#[cfg(feature = "gpu")]
pub use stwo::prover::backend::gpu::GpuBackend;

/// Shared test utilities for env var isolation.
#[cfg(test)]
pub(crate) mod test_utils {
    /// Global mutex serializing tests that mutate process-wide env vars.
    /// Both `starknet::tests` and `gkr::verifier::tests` use this to
    /// prevent parallel races on `STWO_WEIGHT_BINDING` etc.
    pub static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
}
