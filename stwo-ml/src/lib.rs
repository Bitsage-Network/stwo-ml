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
pub mod receipt;
pub mod starknet;
pub mod tee;
pub mod weight_cache;

#[cfg(any(feature = "cli", feature = "audit", feature = "server"))]
pub mod privacy;

#[cfg(feature = "audit")]
pub mod audit;

#[cfg(feature = "cuda-runtime")]
pub mod gpu_sumcheck;

#[cfg(feature = "multi-gpu")]
pub mod multi_gpu;

/// Re-export core STWO types used throughout stwo-ml.
pub mod prelude {
    pub use stwo::core::fields::m31::M31;
    pub use stwo::core::fields::qm31::QM31;
    pub use stwo::core::pcs::PcsConfig;
    pub use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
    pub use stwo::prover::backend::simd::SimdBackend;
    pub use stwo::prover::backend::BackendForChannel;

    pub use crate::aggregation::compute_io_commitment;
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
        shared_cache, shared_cache_from_file, CachedWeight, SharedWeightCache,
        WeightCommitmentCache,
    };
}

/// Re-export GPU backend when available.
#[cfg(feature = "gpu")]
pub use stwo::prover::backend::gpu::GpuBackend;
