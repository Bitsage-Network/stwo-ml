//! Shared types for inference providers.

use starknet_ff::FieldElement;

/// Trust model for a given inference.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrustModel {
    /// Full ZK proof — we have the weights and proved the computation.
    ZkProof {
        weight_commitment: String,
    },
    /// TLS attestation — cryptographic proof the API call happened.
    TlsAttestation {
        server_domain: String,
    },
    /// Commitment only — we hash the IO but don't prove computation.
    CommitmentOnly,
}

/// Result of an inference call from any provider.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct InferenceResult {
    /// The generated text response.
    pub text: String,
    /// Token IDs of the generated response.
    pub token_ids: Vec<u32>,
    /// Number of generated tokens.
    pub num_tokens: usize,
    /// Model identifier.
    pub model_id: String,
    /// Provider name ("local", "openai", "anthropic", etc.).
    pub provider: String,
    /// Trust model used.
    pub trust_model: TrustModel,
    /// Poseidon hash of (prompt || response).
    pub io_commitment: Option<FieldElement>,
    /// Inference latency in milliseconds.
    pub inference_time_ms: u64,
}

/// Attestation record for any inference (regardless of provider).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct InferenceAttestation {
    pub attestation_id: String,
    pub model_id: String,
    pub provider: String,
    pub trust_model: TrustModel,
    pub input_commitment: FieldElement,
    pub output_commitment: FieldElement,
    pub io_commitment: FieldElement,
    pub timestamp_epoch_ms: u64,
    pub proof_id: Option<String>,
    pub proof_status: ProofStatus,
}

/// Status of the proof for an inference.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ProofStatus {
    /// No proof requested or applicable.
    None,
    /// Proof job queued.
    Queued,
    /// Proof being generated.
    Proving,
    /// Proof complete.
    Complete,
    /// Proof generation failed.
    Failed,
}

/// Chat message in OpenAI/Anthropic format.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Provider-agnostic inference request.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: bool,
}

/// Compute IO commitment from prompt and response text.
pub fn compute_text_io_commitment(prompt: &str, response: &str) -> FieldElement {
    let prompt_hash = hash_text(prompt);
    let response_hash = hash_text(response);
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(0x494F434Du64), // "IOCM"
        prompt_hash,
        response_hash,
    ])
}

fn hash_text(text: &str) -> FieldElement {
    let bytes = text.as_bytes();
    let mut felts = vec![FieldElement::from(bytes.len() as u64)];
    for chunk in bytes.chunks(31) {
        let mut buf = [0u8; 32];
        buf[32 - chunk.len()..].copy_from_slice(chunk);
        felts.push(FieldElement::from_bytes_be(&buf).unwrap_or(FieldElement::ZERO));
    }
    starknet_crypto::poseidon_hash_many(&felts)
}
