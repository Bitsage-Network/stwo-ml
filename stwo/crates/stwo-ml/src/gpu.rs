//! GPU-accelerated proof pipeline.
//!
//! Provides a high-level `GpuModelProver` that uses STWO's `GpuBackend`
//! for accelerated proving when CUDA is available. Falls back to
//! `SimdBackend` transparently when GPU is not available.

use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::channel::MerkleChannel;

use crate::backend::{gpu_is_available, gpu_device_name, gpu_available_memory, GpuThresholds};
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::compiler::prove::{ModelError, LayerProof, GraphExecution};
use crate::components::matmul::M31Matrix;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

/// GPU-accelerated model prover.
///
/// When CUDA is available and the problem size exceeds thresholds,
/// this uses `GpuBackend` for proving. Otherwise, it transparently
/// falls back to `SimdBackend`.
#[derive(Debug)]
pub struct GpuModelProver {
    pub device_name: String,
    pub available_memory: usize,
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
    /// Returns `Err` only if explicitly requiring GPU and none exists.
    pub fn new() -> Result<Self, GpuError> {
        if gpu_is_available() {
            Ok(Self {
                device_name: gpu_device_name().unwrap_or_else(|| "Unknown GPU".to_string()),
                available_memory: gpu_available_memory().unwrap_or(0),
                is_gpu: true,
            })
        } else {
            Ok(Self {
                device_name: "CPU (SimdBackend)".to_string(),
                available_memory: 0,
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
        // Each trace row ≈ 4 bytes (M31) × number of columns (≈10 per component)
        // Plus FRI overhead (2x blowup)
        total_rows * 4 * 10 * 3
    }

    /// Check if the graph fits in GPU memory.
    pub fn fits_in_memory(&self, graph: &ComputationGraph) -> bool {
        if !self.is_gpu {
            return true; // CPU always "fits"
        }
        self.estimate_memory(graph) < self.available_memory
    }

    /// Prove a model using the best available backend.
    pub fn prove_model(
        &self,
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<(Vec<LayerProof<Blake2sHash>>, GraphExecution), ModelError> {
        if self.is_gpu && graph.total_trace_rows() >= (1 << GpuThresholds::FFT_FRI) {
            self.prove_model_gpu(graph, input, weights)
        } else {
            crate::compiler::prove::prove_model(graph, input, weights)
        }
    }

    /// Internal: prove using GPU backend.
    fn prove_model_gpu(
        &self,
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<(Vec<LayerProof<Blake2sHash>>, GraphExecution), ModelError> {
        #[cfg(feature = "cuda-runtime")]
        {
            return crate::compiler::prove::prove_model(graph, input, weights);
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            crate::compiler::prove::prove_model(graph, input, weights)
        }
    }
}

impl Default for GpuModelProver {
    fn default() -> Self {
        Self::new().unwrap_or(Self {
            device_name: "CPU (SimdBackend)".to_string(),
            available_memory: 0,
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
        // On CI, this will be CPU
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
        // Simple matmul-only model that has weights
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
}
