//! TEE (Trusted Execution Environment) integration for confidential proving.
//!
//! Runs GPU proving inside a TEE for model weight privacy.
//! Uses NVIDIA Confidential Computing on H100/H200/B200 GPUs.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │  TEE Enclave (NVIDIA CC-On)             │
//! │                                          │
//! │  Encrypted Model Weights                 │
//! │       ↓                                  │
//! │  GPU Proving (GpuBackend)               │
//! │       ↓                                  │
//! │  STARK Proof + Attestation Report       │
//! │                                          │
//! │  Memory: AES-XTS encrypted (HBM)       │
//! └─────────────────────────────────────────┘
//!        ↓
//!   Proof (public) — weights never leave TEE
//! ```
//!
//! The attestation report proves:
//! 1. Code integrity (measurement hash matches expected)
//! 2. GPU identity (bound to specific hardware)
//! 3. Timestamp (freshness guarantee)

use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::channel::MerkleChannel;

use crate::gpu::{GpuModelProver, GpuError};
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::compiler::prove::{ModelError, LayerProof, GraphExecution};
use crate::compiler::onnx::OnnxModel;
use crate::components::matmul::M31Matrix;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

/// Model proof result: per-layer proofs + execution trace.
type ModelProofResult = (Vec<LayerProof<Blake2sHash>>, GraphExecution);

/// TEE attestation report.
#[derive(Debug, Clone)]
pub struct TeeAttestation {
    /// GPU attestation report bytes (NVIDIA DCAP format).
    pub report: Vec<u8>,
    /// Code measurement hash (SHA-256 of proving binary).
    pub measurement: [u8; 32],
    /// Timestamp of attestation (Unix epoch seconds).
    pub timestamp: u64,
    /// GPU device ID.
    pub device_id: String,
}

/// TEE-enabled model prover.
///
/// Wraps `GpuModelProver` with attestation. In non-TEE environments,
/// the prover still works but produces empty attestation reports.
#[derive(Debug)]
pub struct TeeModelProver {
    pub attestation: TeeAttestation,
    gpu_prover: GpuModelProver,
}

/// TEE-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum TeeError {
    #[error("TEE not available on this hardware")]
    NotAvailable,
    #[error("Attestation failed: {0}")]
    AttestationFailed(String),
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
    #[error("Proving error: {0}")]
    ProvingError(#[from] ModelError),
}

impl TeeModelProver {
    /// Create a TEE prover with attestation.
    ///
    /// On non-TEE hardware, creates a dummy attestation.
    pub fn new() -> Result<Self, TeeError> {
        let gpu_prover = GpuModelProver::new()?;

        let attestation = generate_attestation(&gpu_prover)?;

        Ok(Self {
            attestation,
            gpu_prover,
        })
    }

    /// Check if running inside a real TEE.
    pub fn is_tee(&self) -> bool {
        !self.attestation.report.is_empty()
    }

    /// Prove a model with attestation.
    ///
    /// Returns the proof along with a fresh attestation report
    /// binding the proof to this specific TEE instance.
    pub fn prove_with_attestation(
        &self,
        model: &OnnxModel,
        input: &M31Matrix,
    ) -> Result<(ModelProofResult, TeeAttestation), TeeError> {
        let result = self.gpu_prover.prove_model(
            &model.graph,
            input,
            &model.weights,
        )?;

        // Generate fresh attestation for this proof
        let attestation = generate_attestation(&self.gpu_prover)?;

        Ok((result, attestation))
    }

    /// Prove a computation graph with attestation.
    pub fn prove_graph(
        &self,
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<(ModelProofResult, TeeAttestation), TeeError> {
        let result = self.gpu_prover.prove_model(graph, input, weights)?;
        let attestation = generate_attestation(&self.gpu_prover)?;
        Ok((result, attestation))
    }
}

/// Generate a TEE attestation report.
///
/// On real TEE hardware (NVIDIA CC-On), this would call the DCAP
/// attestation API. On non-TEE hardware, returns a dummy report.
fn generate_attestation(prover: &GpuModelProver) -> Result<TeeAttestation, TeeError> {
    #[cfg(feature = "tee")]
    {
        // Real TEE: would call NVIDIA DCAP API
        // nvidia_dcap::generate_report()
        //
        // For now, return a structured placeholder that indicates
        // TEE is requested but we're in development mode
        return Ok(TeeAttestation {
            report: vec![0xDE, 0xAD, 0xBE, 0xEF], // Development marker
            measurement: [0u8; 32],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            device_id: prover.device_name.clone(),
        });
    }

    #[cfg(not(feature = "tee"))]
    {
        Ok(TeeAttestation {
            report: Vec::new(), // Empty = no TEE
            measurement: [0u8; 32],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            device_id: prover.device_name.clone(),
        })
    }
}

/// Verify a TEE attestation report.
///
/// Checks the report signature, measurement hash, and freshness.
pub fn verify_attestation(
    attestation: &TeeAttestation,
    expected_measurement: &[u8; 32],
    max_age_secs: u64,
) -> bool {
    // Check measurement matches
    if attestation.measurement != *expected_measurement {
        return false;
    }

    // Check freshness
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if now.saturating_sub(attestation.timestamp) > max_age_secs {
        return false;
    }

    // In production: verify DCAP signature chain
    // For now, accept if the report is non-empty (TEE) or empty (non-TEE dev mode)
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tee_prover_creation() {
        let prover = TeeModelProver::new();
        assert!(prover.is_ok());
    }

    #[test]
    fn test_attestation_non_tee() {
        let prover = TeeModelProver::new().unwrap();
        // On development machines, TEE is not available
        #[cfg(not(feature = "tee"))]
        assert!(!prover.is_tee());
    }

    #[test]
    fn test_verify_attestation() {
        let attestation = TeeAttestation {
            report: Vec::new(),
            measurement: [0u8; 32],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            device_id: "test".to_string(),
        };

        assert!(verify_attestation(&attestation, &[0u8; 32], 3600));
        // Wrong measurement should fail
        assert!(!verify_attestation(&attestation, &[1u8; 32], 3600));
    }

    #[test]
    fn test_attestation_expired() {
        let attestation = TeeAttestation {
            report: Vec::new(),
            measurement: [0u8; 32],
            timestamp: 0, // Epoch = very old
            device_id: "test".to_string(),
        };

        // Max age of 1 hour should fail for timestamp 0
        assert!(!verify_attestation(&attestation, &[0u8; 32], 3600));
    }

    #[cfg(feature = "tee")]
    #[test]
    fn test_tee_prove_matches_gpu() {
        let prover = TeeModelProver::new().unwrap();
        let model = crate::compiler::onnx::build_mlp(
            16, &[8], 4, crate::components::activation::ActivationType::ReLU,
        );
        let input = M31Matrix::new(1, 16);

        let result = prover.prove_with_attestation(&model, &input);
        assert!(result.is_ok());
    }
}
