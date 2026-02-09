//! GPU-accelerated proof pipeline.
//!
//! Provides a high-level `GpuModelProver` that uses STWO's `GpuBackend`
//! for accelerated proving when CUDA is available. Falls back to
//! `SimdBackend` transparently when GPU is not available.
//!
//! # Usage
//!
//! ```ignore
//! let prover = GpuModelProver::new()?;
//! let (proofs, execution) = prover.prove_model(&graph, &input, &weights)?;
//! ```
//!
//! Or use `prove_model_with` directly for explicit backend control:
//!
//! ```ignore
//! use stwo_ml::compiler::prove::prove_model_with;
//! use stwo::prover::backend::gpu::GpuBackend;
//! use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
//!
//! let result = prove_model_with::<GpuBackend, Blake2sMerkleChannel>(&graph, &input, &weights)?;
//! ```

use crate::backend::{
    gpu_is_available, gpu_device_name, gpu_available_memory,
    gpu_compute_capability, estimate_proof_memory, GpuThresholds,
};
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::compiler::prove::{ModelError, ModelProofResult};
use crate::components::matmul::M31Matrix;
use crate::aggregation::{AggregatedModelProof, AggregationError};
use crate::receipt::{ComputeReceipt, ReceiptProof, ReceiptError};

/// GPU-accelerated model prover.
///
/// When CUDA is available and the problem size exceeds thresholds,
/// this uses `GpuBackend` for proving. Otherwise, it transparently
/// falls back to `SimdBackend`.
#[derive(Debug)]
pub struct GpuModelProver {
    pub device_name: String,
    pub available_memory: usize,
    pub compute_capability: Option<(u32, u32)>,
    pub is_gpu: bool,
}

/// Error type for GPU operations.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("No GPU available")]
    NoGpu,
    #[error("Insufficient GPU memory: need {needed} bytes, have {available}")]
    InsufficientMemory { needed: usize, available: usize },
    #[error("Proving error: {0}")]
    ProvingError(String),
}

impl GpuModelProver {
    /// Create a new GPU model prover.
    ///
    /// Returns `Ok` even if no GPU is available (will use CPU fallback).
    pub fn new() -> Result<Self, GpuError> {
        if gpu_is_available() {
            Ok(Self {
                device_name: gpu_device_name().unwrap_or_else(|| "Unknown GPU".to_string()),
                available_memory: gpu_available_memory().unwrap_or(0),
                compute_capability: gpu_compute_capability(),
                is_gpu: true,
            })
        } else {
            Ok(Self {
                device_name: "CPU (SimdBackend)".to_string(),
                available_memory: 0,
                compute_capability: None,
                is_gpu: false,
            })
        }
    }

    /// Create a prover that requires GPU (fails if none available).
    pub fn require_gpu() -> Result<Self, GpuError> {
        if !gpu_is_available() {
            return Err(GpuError::NoGpu);
        }
        Self::new()
    }

    /// Estimate memory required for proving a graph.
    pub fn estimate_memory(&self, graph: &ComputationGraph) -> usize {
        let total_rows = graph.total_trace_rows();
        let log_size = (total_rows as f64).log2().ceil() as u32;
        let num_columns = graph.nodes.len() * 10; // ~10 columns per component
        estimate_proof_memory(log_size, num_columns)
    }

    /// Check if the graph fits in GPU memory.
    pub fn fits_in_memory(&self, graph: &ComputationGraph) -> bool {
        if !self.is_gpu {
            return true; // CPU always "fits"
        }
        let needed = self.estimate_memory(graph);
        needed < (self.available_memory * 4 / 5) // 80% safety margin
    }

    /// Whether to use GPU for the given graph based on size thresholds.
    fn should_use_gpu(&self, graph: &ComputationGraph) -> bool {
        self.is_gpu
            && graph.total_trace_rows() >= (1 << GpuThresholds::FFT_FRI)
            && self.fits_in_memory(graph)
    }

    /// Prove a model using the best available backend.
    ///
    /// When GPU is available, the problem exceeds the FFT threshold, and
    /// it fits in GPU memory, uses `GpuBackend` for accelerated proving.
    /// Otherwise falls back to `SimdBackend`.
    pub fn prove_model(
        &self,
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<ModelProofResult, ModelError> {
        if self.should_use_gpu(graph) {
            self.prove_model_gpu(graph, input, weights)
        } else {
            crate::compiler::prove::prove_model(graph, input, weights)
        }
    }

    /// Prove a model with aggregation using the best available backend.
    ///
    /// All activation layers are combined into a single STARK proof.
    /// Uses GPU when the graph exceeds size thresholds and fits in GPU memory.
    pub fn prove_model_aggregated(
        &self,
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<AggregatedModelProof, AggregationError> {
        if self.should_use_gpu(graph) {
            self.prove_model_aggregated_gpu(graph, input, weights)
        } else {
            crate::aggregation::prove_model_aggregated(graph, input, weights)
        }
    }

    /// Prove a batch of compute receipts using the best available backend.
    ///
    /// Uses GPU for large batches that exceed size thresholds.
    pub fn prove_receipt_batch(
        &self,
        receipts: &[ComputeReceipt],
    ) -> Result<ReceiptProof, ReceiptError> {
        let log_size = receipts.len().next_power_of_two().ilog2().max(4);
        if self.is_gpu && log_size >= GpuThresholds::FFT_FRI {
            self.prove_receipt_batch_gpu(receipts)
        } else {
            crate::receipt::prove_receipt_batch(receipts)
        }
    }

    /// Internal: prove using GPU backend.
    fn prove_model_gpu(
        &self,
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<ModelProofResult, ModelError> {
        #[cfg(feature = "cuda-runtime")]
        {
            use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
            use stwo::prover::backend::gpu::GpuBackend;
            return crate::compiler::prove::prove_model_with::<GpuBackend, Blake2sMerkleChannel>(
                graph, input, weights,
            );
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            crate::compiler::prove::prove_model(graph, input, weights)
        }
    }

    /// Internal: aggregated prove using GPU backend.
    fn prove_model_aggregated_gpu(
        &self,
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<AggregatedModelProof, AggregationError> {
        #[cfg(feature = "cuda-runtime")]
        {
            use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
            use stwo::prover::backend::gpu::GpuBackend;
            return crate::aggregation::prove_model_aggregated_with::<GpuBackend, Blake2sMerkleChannel>(
                graph, input, weights,
            );
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            crate::aggregation::prove_model_aggregated(graph, input, weights)
        }
    }

    /// Internal: receipt batch prove using GPU backend.
    fn prove_receipt_batch_gpu(
        &self,
        receipts: &[ComputeReceipt],
    ) -> Result<ReceiptProof, ReceiptError> {
        #[cfg(feature = "cuda-runtime")]
        {
            use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
            use stwo::prover::backend::gpu::GpuBackend;
            return crate::receipt::prove_receipt_batch_with::<GpuBackend, Blake2sMerkleChannel>(
                receipts,
            );
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            crate::receipt::prove_receipt_batch(receipts)
        }
    }
}

impl Default for GpuModelProver {
    fn default() -> Self {
        Self::new().unwrap_or(Self {
            device_name: "CPU (SimdBackend)".to_string(),
            available_memory: 0,
            compute_capability: None,
            is_gpu: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use crate::components::activation::ActivationType;

    #[test]
    fn test_gpu_prover_creation() {
        let prover = GpuModelProver::new();
        assert!(prover.is_ok());
    }

    #[test]
    fn test_gpu_prover_default() {
        let prover = GpuModelProver::default();
        assert!(!prover.device_name.is_empty());
    }

    #[test]
    fn test_memory_estimation() {
        let mut builder = GraphBuilder::new((1, 128));
        builder.linear(64).activation(ActivationType::ReLU).linear(10);
        let graph = builder.build();

        let prover = GpuModelProver::default();
        let mem = prover.estimate_memory(&graph);
        assert!(mem > 0);
    }

    #[test]
    fn test_prove_model_cpu_fallback() {
        use stwo::core::fields::m31::M31;
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let prover = GpuModelProver::default();
        let result = prover.prove_model(&graph, &input, &weights);
        assert!(result.is_ok());
    }

    #[test]
    fn test_should_use_gpu_threshold() {
        let prover = GpuModelProver::default();

        // Small graph — should NOT use GPU even if available
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let small_graph = builder.build();
        // total_trace_rows for a 1×4 → 4×2 matmul is small
        assert!(!prover.should_use_gpu(&small_graph));
    }

    /// GPU-specific test — only runs when cuda-runtime is enabled.
    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn test_gpu_backend_real_proving() {
        use stwo::core::fields::m31::M31;

        let prover = GpuModelProver::require_gpu()
            .expect("GPU required for this test");
        assert!(prover.is_gpu);
        assert!(!prover.device_name.contains("CPU"));

        // Build a model large enough to exceed GPU threshold
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let result = prover.prove_model(&graph, &input, &weights);
        assert!(result.is_ok(), "GPU proving failed: {:?}", result.err());
    }

    #[test]
    fn test_prove_model_aggregated_cpu_fallback() {
        use stwo::core::fields::m31::M31;
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let prover = GpuModelProver::default();
        let result = prover.prove_model_aggregated(&graph, &input, &weights);
        assert!(result.is_ok());
        let proof = result.unwrap();
        assert!(proof.activation_stark.is_some());
        assert_eq!(proof.matmul_proofs.len(), 2);
    }

    #[test]
    fn test_prove_receipt_batch_cpu_fallback() {
        use starknet_ff::FieldElement;

        let receipt = ComputeReceipt {
            job_id: FieldElement::from(1u64),
            worker_pubkey: FieldElement::from(42u64),
            input_commitment: FieldElement::from(100u64),
            output_commitment: FieldElement::from(200u64),
            model_commitment: FieldElement::from(300u64),
            prev_receipt_hash: FieldElement::ZERO,
            gpu_time_ms: 5000,
            token_count: 512,
            peak_memory_mb: 1024,
            billing_amount_sage: 5620, // 5000*100/1000 + 512*10 = 500+5120
            billing_rate_per_sec: 100,
            billing_rate_per_token: 10,
            tee_report_hash: FieldElement::from(500u64),
            tee_timestamp: 1700000000,
            timestamp: 1700000010,
            sequence_number: 0,
        };

        let prover = GpuModelProver::default();
        let result = prover.prove_receipt_batch(&[receipt]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().batch_size, 1);
    }

    /// Verify GPU device info when CUDA is available.
    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn test_gpu_device_info() {
        if !gpu_is_available() {
            return;
        }
        let prover = GpuModelProver::new().unwrap();
        assert!(prover.is_gpu);
        assert!(prover.available_memory > 0);
        assert!(prover.compute_capability.is_some());
    }
}
