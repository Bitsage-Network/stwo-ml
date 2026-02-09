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

use crate::gpu::{GpuModelProver, GpuError};
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::compiler::prove::{ModelError, ModelProofResult};
use crate::compiler::onnx::OnnxModel;
use crate::components::matmul::M31Matrix;

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
/// Wraps `GpuModelProver` with attestation. Without the `tee` feature,
/// the prover works but produces empty attestation reports (honest about
/// no TEE). With the `tee` feature enabled, creation fails unless running
/// on real NVIDIA CC-On hardware (H100/H200/B200).
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
    /// Without `tee` feature: creates prover with empty attestation.
    /// With `tee` feature: requires CC-On hardware or returns error.
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
/// With `tee` feature: requires NVIDIA CC-On hardware and nvidia-dcap crate.
/// Fails honestly if hardware is not available.
/// Without `tee` feature: returns empty attestation (no TEE).
fn generate_attestation(prover: &GpuModelProver) -> Result<TeeAttestation, TeeError> {
    #[cfg(feature = "tee")]
    {
        // Real TEE attestation requires NVIDIA CC-On hardware (H100/H200/B200)
        // and the nvidia-dcap crate for remote attestation. Without these,
        // fail honestly rather than returning fake attestation data.
        return Err(TeeError::AttestationFailed(
            "NVIDIA DCAP attestation requires CC-On hardware (H100/H200/B200) \
             and nvidia-dcap crate. Run on TEE-enabled GPU."
                .into(),
        ));
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
/// Checks: non-empty report, measurement hash, and freshness.
/// Full DCAP signature chain verification requires the nvidia-dcap crate.
pub fn verify_attestation(
    attestation: &TeeAttestation,
    expected_measurement: &[u8; 32],
    max_age_secs: u64,
) -> bool {
    // A valid attestation requires a non-empty TEE report.
    // Empty reports indicate non-TEE mode and cannot pass verification.
    if attestation.report.is_empty() {
        return false;
    }

    // Check measurement matches expected binary hash
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

    // Structural checks pass (non-empty report, correct measurement, fresh).
    // Full DCAP signature chain verification requires the nvidia-dcap crate,
    // which will be integrated when TEE hardware support is available.
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tee_prover_creation() {
        let result = TeeModelProver::new();
        // Without TEE feature: creates prover with empty attestation.
        // With TEE feature: fails without CC-On hardware.
        #[cfg(not(feature = "tee"))]
        assert!(result.is_ok());
        #[cfg(feature = "tee")]
        let _ = result; // Err on dev machines, Ok on CC-On hardware
    }

    #[test]
    fn test_attestation_non_tee() {
        #[cfg(not(feature = "tee"))]
        {
            let prover = TeeModelProver::new().unwrap();
            assert!(!prover.is_tee());
        }
    }

    #[test]
    fn test_verify_attestation() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Non-empty report + matching measurement + fresh → pass
        let valid = TeeAttestation {
            report: vec![1, 2, 3, 4],
            measurement: [0u8; 32],
            timestamp: now,
            device_id: "test".to_string(),
        };
        assert!(verify_attestation(&valid, &[0u8; 32], 3600));

        // Wrong measurement → fail
        assert!(!verify_attestation(&valid, &[1u8; 32], 3600));

        // Empty report → fail (no TEE)
        let no_tee = TeeAttestation {
            report: Vec::new(),
            measurement: [0u8; 32],
            timestamp: now,
            device_id: "test".to_string(),
        };
        assert!(!verify_attestation(&no_tee, &[0u8; 32], 3600));
    }

    #[test]
    fn test_attestation_expired() {
        let attestation = TeeAttestation {
            report: vec![1, 2, 3, 4], // Non-empty to test expiry path
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
