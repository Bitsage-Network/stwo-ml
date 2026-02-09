//! Pipeline types: per-layer proofs, commitment chain, model pipeline proof.

use starknet_crypto::poseidon_hash_many;
use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::channel::MerkleChannel;

use crate::components::attention::AttentionProofOnChain;
use crate::components::matmul::{M31Matrix, MatMulSumcheckProofOnChain};
use crate::receipt::ComputeReceipt;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

/// What kind of proof a layer carries.
#[derive(Debug)]
pub enum LayerProofKindOnChain {
    /// MatMul: sumcheck with Poseidon Fiat-Shamir + MLE commitments.
    MatMulSumcheck(MatMulSumcheckProofOnChain),
    /// Activation: LogUp-based STARK proof.
    ActivationStark(StarkProof<Blake2sHash>),
    /// Multi-head attention: composed matmul sumcheck + softmax proofs.
    Attention(Box<AttentionProofOnChain>),
    /// LayerNorm: passthrough (proven at aggregation layer).
    Passthrough,
}

/// Proof for a single layer with commitment linking.
#[derive(Debug)]
pub struct LayerPipelineProof {
    /// Layer index in the model.
    pub layer_index: usize,
    /// The actual proof.
    pub kind: LayerProofKindOnChain,
    /// Poseidon commitment of the layer's input matrix.
    pub input_commitment: FieldElement,
    /// Poseidon commitment of the layer's output matrix.
    pub output_commitment: FieldElement,
}

/// Configuration for the pipeline prover.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Whether to generate on-chain (Poseidon) matmul proofs.
    /// Off-chain uses Blake2s (faster, for local testing).
    pub onchain_matmul: bool,
    /// Whether to generate activation STARK proofs.
    pub prove_activations: bool,
    /// Whether to generate a compute receipt.
    pub generate_receipt: bool,
    /// Optional precomputed model commitment (skips expensive weight hashing).
    /// When `Some`, the pipeline uses this value directly instead of hashing
    /// all weight matrices with Poseidon (which can take minutes for large models).
    pub precomputed_model_commitment: Option<FieldElement>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            onchain_matmul: true,
            prove_activations: true,
            generate_receipt: true,
            precomputed_model_commitment: None,
        }
    }
}

/// Complete model pipeline proof with commitment chain.
#[derive(Debug)]
pub struct ModelPipelineProof {
    /// Poseidon commitment over all model weights.
    pub model_commitment: FieldElement,
    /// Poseidon(input_data || output_data).
    pub io_commitment: FieldElement,
    /// Per-layer proofs with commitment linking.
    pub layer_proofs: Vec<LayerPipelineProof>,
    /// Optional compute receipt.
    pub receipt: Option<ComputeReceipt>,
    /// TEE attestation report hash (if available).
    pub tee_report_hash: Option<FieldElement>,
}

impl ModelPipelineProof {
    /// Number of layers with real proofs (matmul + activation).
    pub fn num_proven_layers(&self) -> usize {
        self.layer_proofs
            .iter()
            .filter(|p| !matches!(p.kind, LayerProofKindOnChain::Passthrough))
            .count()
    }

    /// Total number of matmul sumcheck proofs.
    pub fn num_matmul_proofs(&self) -> usize {
        self.layer_proofs
            .iter()
            .filter(|p| matches!(p.kind, LayerProofKindOnChain::MatMulSumcheck(_)))
            .count()
    }

    /// Total number of activation STARK proofs.
    pub fn num_activation_proofs(&self) -> usize {
        self.layer_proofs
            .iter()
            .filter(|p| matches!(p.kind, LayerProofKindOnChain::ActivationStark(_)))
            .count()
    }

    /// Total number of attention proofs.
    pub fn num_attention_proofs(&self) -> usize {
        self.layer_proofs
            .iter()
            .filter(|p| matches!(p.kind, LayerProofKindOnChain::Attention(_)))
            .count()
    }

    /// Verify the commitment chain: each layer's output == next layer's input.
    pub fn verify_commitment_chain(&self) -> bool {
        for window in self.layer_proofs.windows(2) {
            // Skip passthrough → passthrough transitions (layernorm chains)
            if matches!(window[0].kind, LayerProofKindOnChain::Passthrough)
                && matches!(window[1].kind, LayerProofKindOnChain::Passthrough)
            {
                continue;
            }
            if window[0].output_commitment != window[1].input_commitment {
                return false;
            }
        }
        true
    }
}

/// Commit to a matrix by Poseidon-hashing its flattened M31 values.
///
/// Used for: weight matrices, intermediate activations, model I/O.
pub fn commit_matrix(matrix: &M31Matrix) -> FieldElement {
    let felts: Vec<FieldElement> = matrix
        .data
        .iter()
        .map(|v| FieldElement::from(v.0 as u64))
        .collect();
    if felts.is_empty() {
        return FieldElement::ZERO;
    }
    poseidon_hash_many(&felts)
}

/// Commit to a slice of M31 values.
pub fn commit_values(values: &[M31]) -> FieldElement {
    let felts: Vec<FieldElement> = values
        .iter()
        .map(|v| FieldElement::from(v.0 as u64))
        .collect();
    if felts.is_empty() {
        return FieldElement::ZERO;
    }
    poseidon_hash_many(&felts)
}

/// Commit to all model weights (Poseidon hash of concatenated weight commitments).
pub fn commit_model_weights(
    weight_commitments: &[FieldElement],
) -> FieldElement {
    if weight_commitments.is_empty() {
        return FieldElement::ZERO;
    }
    poseidon_hash_many(weight_commitments)
}

/// Compute I/O commitment: Poseidon(input_data || output_data).
pub fn compute_pipeline_io_commitment(
    input: &M31Matrix,
    output: &M31Matrix,
) -> FieldElement {
    let mut felts = Vec::with_capacity(input.data.len() + output.data.len());
    for v in &input.data {
        felts.push(FieldElement::from(v.0 as u64));
    }
    for v in &output.data {
        felts.push(FieldElement::from(v.0 as u64));
    }
    if felts.is_empty() {
        return FieldElement::ZERO;
    }
    poseidon_hash_many(&felts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_matrix_deterministic() {
        let mut m = M31Matrix::new(2, 2);
        m.set(0, 0, M31::from(1u32));
        m.set(0, 1, M31::from(2u32));
        m.set(1, 0, M31::from(3u32));
        m.set(1, 1, M31::from(4u32));

        let c1 = commit_matrix(&m);
        let c2 = commit_matrix(&m);
        assert_eq!(c1, c2, "Commitment must be deterministic");
        assert_ne!(c1, FieldElement::ZERO, "Non-trivial matrix should have non-zero commitment");
    }

    #[test]
    fn test_commit_matrix_different_data() {
        let mut m1 = M31Matrix::new(2, 2);
        m1.set(0, 0, M31::from(1u32));
        m1.set(0, 1, M31::from(2u32));
        m1.set(1, 0, M31::from(3u32));
        m1.set(1, 1, M31::from(4u32));

        let mut m2 = M31Matrix::new(2, 2);
        m2.set(0, 0, M31::from(5u32));
        m2.set(0, 1, M31::from(6u32));
        m2.set(1, 0, M31::from(7u32));
        m2.set(1, 1, M31::from(8u32));

        assert_ne!(
            commit_matrix(&m1),
            commit_matrix(&m2),
            "Different matrices must produce different commitments"
        );
    }

    #[test]
    fn test_commit_empty_matrix() {
        let m = M31Matrix::new(0, 0);
        assert_eq!(commit_matrix(&m), FieldElement::ZERO);
    }

    #[test]
    fn test_commit_model_weights() {
        let c1 = FieldElement::from(42u64);
        let c2 = FieldElement::from(99u64);
        let model_commit = commit_model_weights(&[c1, c2]);
        assert_ne!(model_commit, FieldElement::ZERO);

        // Different order should produce different commitment
        let model_commit_rev = commit_model_weights(&[c2, c1]);
        assert_ne!(model_commit, model_commit_rev);
    }

    #[test]
    fn test_io_commitment() {
        let mut input = M31Matrix::new(1, 4);
        for i in 0..4 {
            input.set(0, i, M31::from((i + 1) as u32));
        }
        let mut output = M31Matrix::new(1, 2);
        output.set(0, 0, M31::from(10u32));
        output.set(0, 1, M31::from(20u32));

        let io = compute_pipeline_io_commitment(&input, &output);
        assert_ne!(io, FieldElement::ZERO);

        // Deterministic
        let io2 = compute_pipeline_io_commitment(&input, &output);
        assert_eq!(io, io2);
    }
}
