//! Execution trace types for the ObelyZK VM.
//!
//! An `ExecutionTrace` captures the result of a forward pass + proving run,
//! including the output, commitments, and the cryptographic proof itself.

use starknet_ff::FieldElement;

use crate::components::matmul::M31Matrix;

/// Complete execution trace for one inference call.
///
/// Contains the output, commitments, and proof from a single inference.
/// This is the unit of work tracked by the VM's proving queue.
pub struct ExecutionTrace {
    /// Model identifier (weight commitment hex or name).
    pub model_id: String,
    /// Input token IDs (from tokenizer).
    pub input_tokens: Vec<u32>,
    /// Final output matrix (last layer's output).
    pub output: M31Matrix,
    /// IO commitment: Poseidon(input || output).
    pub io_commitment: Option<FieldElement>,
    /// Policy commitment bound into the Fiat-Shamir channel.
    pub policy_commitment: FieldElement,
    /// KV-cache commitment before this inference (None for first turn).
    pub kv_commitment_before: Option<FieldElement>,
    /// KV-cache commitment after this inference.
    pub kv_commitment_after: Option<FieldElement>,
    /// Tokenization commitment: H(text, token_ids, tokenizer_config).
    pub tokenization_commitment: Option<FieldElement>,
    /// Wall-clock time in milliseconds (execution + proving).
    pub inference_time_ms: u64,
    /// Number of tokens in this trace.
    pub num_tokens: usize,
    /// Position offset (for decode steps — 0 for prefill).
    pub position_offset: usize,
    /// The cryptographic proof produced by the proving pipeline.
    pub proof: Option<crate::aggregation::AggregatedModelProofOnChain>,
}

/// Metadata about a proven chunk within a conversation.
pub struct ProvenChunkMeta {
    pub chunk_index: usize,
    pub position_start: usize,
    pub position_end: usize,
    pub num_tokens: usize,
    pub kv_commitment: FieldElement,
    pub prev_kv_commitment: FieldElement,
    pub io_commitment: FieldElement,
    pub proof_id: String,
    pub prove_time_ms: u64,
}

/// Commitment over the tokenization step.
///
/// Proves: "for this text hash and tokenizer config, these token IDs were produced."
/// The tokenizer is deterministic, so a commitment (not a ZK proof) is sufficient.
pub fn commit_tokenization(
    text: &str,
    token_ids: &[u32],
    tokenizer_config_hash: FieldElement,
) -> FieldElement {
    let text_hash = {
        let bytes = text.as_bytes();
        let mut felts = vec![FieldElement::from(0x544F4B4Eu64)]; // "TOKN"
        felts.push(FieldElement::from(bytes.len() as u64));
        for chunk in bytes.chunks(31) {
            let mut buf = [0u8; 32];
            buf[32 - chunk.len()..].copy_from_slice(chunk);
            felts.push(FieldElement::from_bytes_be(&buf).unwrap_or(FieldElement::ZERO));
        }
        starknet_crypto::poseidon_hash_many(&felts)
    };

    let token_hash = {
        let mut felts = vec![FieldElement::from(token_ids.len() as u64)];
        for &tid in token_ids {
            felts.push(FieldElement::from(tid as u64));
        }
        starknet_crypto::poseidon_hash_many(&felts)
    };

    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(0x544F4B4Eu64),
        text_hash,
        token_hash,
        tokenizer_config_hash,
    ])
}

/// Commitment over the sampling/decoding step.
pub fn commit_sampling(
    logits_commitment: FieldElement,
    selected_token_id: u32,
    temperature: u32,
    top_k: u32,
    seed: u64,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(0x53414D50u64), // "SAMP"
        logits_commitment,
        FieldElement::from(selected_token_id as u64),
        FieldElement::from(temperature as u64),
        FieldElement::from(top_k as u64),
        FieldElement::from(seed),
    ])
}

/// Conversation-level commitment chain.
///
/// `C_n = H(C_{n-1}, turn, model_id, io, kv, policy)`
pub fn chain_conversation_commitment(
    prev: FieldElement,
    turn_number: u64,
    model_id: FieldElement,
    io_commitment: FieldElement,
    kv_commitment: FieldElement,
    policy_commitment: FieldElement,
) -> FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(0x434F4E56u64), // "CONV"
        prev,
        FieldElement::from(turn_number),
        model_id,
        io_commitment,
        kv_commitment,
        policy_commitment,
    ])
}
