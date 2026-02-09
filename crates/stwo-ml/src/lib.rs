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

pub mod components;
pub mod compiler;
pub mod gadgets;
pub mod backend;
pub mod gpu;
pub mod aggregation;
pub mod starknet;
pub mod tee;
pub mod cairo_serde;
pub mod receipt;
pub mod crypto;
pub mod pipeline;
pub mod recursive;

/// Re-export core STWO types used throughout stwo-ml.
pub mod prelude {
    pub use stwo::core::fields::m31::M31;
    pub use stwo::core::fields::qm31::QM31;
    pub use stwo::core::pcs::PcsConfig;
    pub use stwo::prover::backend::simd::SimdBackend;
    pub use stwo::prover::backend::BackendForChannel;
    pub use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;

    pub use crate::components::matmul::M31Matrix;
    pub use crate::components::activation::ActivationType;
    pub use crate::components::attention::{
        MultiHeadAttentionConfig, AttentionWeights,
        AttentionProof, AttentionProofOnChain, AttentionError,
        attention_forward, prove_attention_with, prove_attention_onchain,
    };
    pub use crate::compiler::graph::{ComputationGraph, GraphBuilder, GraphWeights};
    pub use crate::compiler::prove::{prove_model, prove_model_with, prove_model_auto};
    pub use crate::compiler::onnx::{
        OnnxModel, ModelMetadata, OnnxError, load_onnx,
        build_mlp, build_mlp_with_weights, generate_weights_for_graph,
        TransformerConfig, build_transformer_block, build_transformer,
    };
    pub use crate::compiler::inspect::{ModelSummary, summarize_model, summarize_graph};
    pub use crate::aggregation::{
        prove_model_aggregated, prove_model_aggregated_with, prove_model_aggregated_auto,
        prove_model_aggregated_onchain, prove_model_aggregated_onchain_auto,
    };
    pub use crate::receipt::{prove_receipt, prove_receipt_batch, prove_receipt_batch_auto};
    pub use crate::backend::BackendInfo;
    pub use crate::gpu::GpuModelProver;
    pub use crate::crypto::poseidon_channel::PoseidonChannel;
    pub use crate::crypto::mle_opening::{MleOpeningProof, commit_mle, prove_mle_opening};
    pub use crate::components::matmul::{
        MatMulSumcheckProofOnChain, RoundPoly,
        prove_matmul_sumcheck_onchain,
        pad_matrix_pow2, estimate_sumcheck_memory,
    };
    pub use crate::starknet::compute_io_commitment;
    pub use crate::pipeline::types::{ModelPipelineProof, PipelineConfig, commit_matrix};
    pub use crate::pipeline::prover::prove_model_pipeline;
    #[cfg(feature = "safetensors")]
    pub use crate::pipeline::prover::prove_model_pipeline_streaming;
    #[cfg(feature = "safetensors")]
    pub use crate::compiler::safetensors::LazyWeights;
    pub use crate::pipeline::verifier::{verify_pipeline_proof, VerificationResult};

    // Dual-track f32 inference + M31 proving
    pub use crate::components::f32_ops::{
        F32Matrix, matmul_f32, apply_activation_f32, layernorm_f32,
        relu_f32, gelu_f32, sigmoid_f32, softmax_f32,
    };
    pub use crate::compiler::dual::{
        DualWeights, DualOutput, f32_forward, prove_model_dual, prove_model_dual_default,
    };
    pub use crate::components::attention::{
        AttentionWeightsF32, attention_forward_f32,
    };
}

/// Re-export GPU backend when available.
#[cfg(feature = "gpu")]
pub use stwo::prover::backend::gpu::GpuBackend;
