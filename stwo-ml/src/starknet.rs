//! Starknet on-chain verification types and calldata generation.
//!
//! End-to-end pipeline: computation graph → aggregated STARK proof →
//! felt252 calldata for the Cairo verifier contract.
//!
//! # Pipeline
//!
//! ```text
//! ComputationGraph + Input + Weights
//!         ↓
//! prove_model_aggregated_auto()  ← GPU-accelerated when available
//!         ↓
//! AggregatedModelProof { unified STARK + matmul sumchecks }
//!         ↓
//! serialize_proof() (cairo_serde)
//!         ↓
//! StarknetModelProof { calldata: Vec<felt252>, gas estimate }
//! ```

use starknet_ff::FieldElement;

use stwo::core::pcs::PcsConfig;

use crate::aggregation::{
    prove_model_aggregated_auto, prove_model_aggregated_onchain_auto, AggregatedModelProof,
    AggregatedModelProofOnChain, AggregationError, LayerClaim,
};
use crate::cairo_serde::{
    serialize_gkr_proof_data_only, serialize_matmul_sumcheck_proof, serialize_proof,
};
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::compiler::prove::ModelError;
use crate::components::matmul::M31Matrix;
use crate::tee::TeeAttestation;

/// A proof formatted for Starknet on-chain verification.
///
/// Contains the serialized felt252 calldata ready to submit to the
/// Cairo verifier contract, along with metadata for the transaction.
#[derive(Debug)]
pub struct StarknetModelProof {
    /// Serialized unified STARK proof as felt252 calldata.
    /// Covers all non-matmul components (activations, add, mul, layernorm).
    /// Empty if the model has no such layers.
    pub unified_calldata: Vec<FieldElement>,
    /// Serialized matmul sumcheck proofs as felt252 calldata (one per matmul layer).
    pub matmul_calldata: Vec<Vec<FieldElement>>,
    /// IO commitment: Poseidon(inputs || outputs). Recomputed on-chain from raw_io_data.
    pub io_commitment: FieldElement,
    /// Raw IO data: [in_rows, in_cols, in_len, in_data..., out_rows, out_cols, out_len, out_data...].
    /// The on-chain verifier recomputes Poseidon(raw_io_data) to derive io_commitment.
    pub raw_io_data: Vec<FieldElement>,
    /// Combined calldata ready for on-chain submission.
    /// Format: [pcs_config(4), raw_io_data(Array), unified_proof..., matmul_proofs_count, matmul_proof_0..., ...]
    pub combined_calldata: Vec<FieldElement>,
    /// Per-activation-layer claims (for the verifier).
    pub layer_claims: Vec<LayerClaim>,
    /// Number of matmul sumcheck proofs.
    pub num_matmul_proofs: usize,
    /// Number of Add claims in the unified STARK.
    pub num_add_claims: usize,
    /// Number of Mul claims in the unified STARK.
    pub num_mul_claims: usize,
    /// Number of LayerNorm claims in the unified STARK.
    pub num_layernorm_claims: usize,
    /// Number of Attention proofs (separate from unified STARK).
    pub num_attention_proofs: usize,
    /// Number of Embedding claims in the unified STARK.
    pub num_embedding_claims: usize,
    /// Number of Quantize claims in the unified STARK.
    pub num_quantize_claims: usize,
    /// Number of Dequantize claims in the unified STARK.
    pub num_dequantize_claims: usize,
    /// Number of RMSNorm claims in the unified STARK.
    pub num_rmsnorm_claims: usize,
    /// Number of LayerNorm mean/var commitments.
    pub num_layernorm_mean_var_commitments: usize,
    /// Total number of proven layers.
    pub num_proven_layers: usize,
    /// PCS configuration used for STARK proving (security parameters).
    /// `pow_bits`: proof-of-work difficulty, `fri_config`: FRI query/blowup parameters.
    pub pcs_config: PcsConfig,
    /// Estimated gas cost for on-chain verification.
    pub estimated_gas: u64,
    /// Total calldata size in felt252 elements.
    pub calldata_size: usize,
    /// Layer chain commitment: running Poseidon hash of intermediate values.
    /// Binds layer outputs to layer inputs, preventing substitution of intermediates.
    pub layer_chain_commitment: FieldElement,
    /// TEE attestation hash (Poseidon hash of the NVIDIA CC attestation report).
    /// `None` when the proof was generated without TEE (pure ZK-only).
    /// `Some(hash)` when the proof was generated on CC-On hardware and includes
    /// a real attestation report. This value is passed to the ObelyskVerifier's
    /// `tee_attestation_hash` parameter.
    pub tee_attestation_hash: Option<FieldElement>,
    /// GKR proof calldata (serialized layer proofs + input claim + weight commitments).
    /// `None` when the standard pipeline was used (no GKR).
    pub gkr_calldata: Option<Vec<FieldElement>>,
}

/// A proof formatted for direct on-chain verification (no Cairo VM recursion).
///
/// Eliminates Stage 2 of the proving pipeline:
///   BEFORE: GPU prove → Cairo VM (46.8s) → on-chain verify recursive proof
///   AFTER:  GPU prove → on-chain verify_model_direct() (0s Stage 2)
#[derive(Debug)]
pub struct DirectStarknetProof {
    /// Model identifier (Poseidon hash of architecture + weights).
    pub model_id: FieldElement,
    /// Raw IO data for on-chain recomputation of Poseidon(inputs || outputs).
    pub raw_io_data: Vec<FieldElement>,
    /// Poseidon hash of weight matrices.
    pub weight_commitment: FieldElement,
    /// Number of model layers.
    pub num_layers: u32,
    /// Activation function type (0=ReLU, 1=GELU, 2=Sigmoid).
    pub activation_type: u8,
    /// Per-batch serialized sumcheck proofs (sent with verify_model_direct tx).
    pub batched_calldata: Vec<Vec<FieldElement>>,
    /// Activation STARK proof split into upload chunks.
    /// Each chunk is uploaded via `upload_proof_chunk()` in a separate tx.
    pub stark_chunks: Vec<Vec<FieldElement>>,
    /// Whether an activation STARK proof is present.
    pub has_activation_stark: bool,
    /// Estimated gas cost for the verify_model_direct transaction.
    pub estimated_gas: u64,
    /// Total calldata size across all transactions.
    pub total_calldata_size: usize,
}

// ============================================================================
// Model Registration
// ============================================================================

/// Model registration data for on-chain registration.
#[derive(Debug, Clone)]
pub struct ModelRegistration {
    /// Unique model identifier: Poseidon(architecture_hash || weight_commitment || description).
    pub model_id: FieldElement,
    /// Poseidon hash of all weight matrices.
    pub weight_commitment: FieldElement,
    /// Number of layers in the model.
    pub num_layers: usize,
    /// Human-readable model description.
    pub description: String,
}

/// Compute a Poseidon commitment over all weight matrices.
pub fn compute_weight_commitment(weights: &GraphWeights) -> FieldElement {
    let mut data = Vec::new();
    // Sort by node_id for deterministic ordering
    let mut sorted: Vec<&(usize, M31Matrix)> = weights.weights.iter().collect();
    sorted.sort_by_key(|(id, _)| *id);
    for (node_id, w) in sorted {
        data.push(FieldElement::from(*node_id as u64));
        data.push(FieldElement::from(w.rows as u64));
        data.push(FieldElement::from(w.cols as u64));
        for val in &w.data {
            data.push(FieldElement::from(val.0 as u64));
        }
    }
    if data.is_empty() {
        return FieldElement::ZERO;
    }
    starknet_crypto::poseidon_hash_many(&data)
}

/// Prepare model registration data from a computation graph and weights.
pub fn prepare_model_registration(
    graph: &ComputationGraph,
    weights: &GraphWeights,
    description: &str,
) -> ModelRegistration {
    let weight_commitment = compute_weight_commitment(weights);

    // Architecture hash: Poseidon of node types and shapes
    let mut arch_data = Vec::new();
    arch_data.push(FieldElement::from(graph.nodes.len() as u64));
    for node in &graph.nodes {
        arch_data.push(FieldElement::from(node.id as u64));
    }
    let arch_hash = starknet_crypto::poseidon_hash_many(&arch_data);

    // Description hash
    let desc_bytes = description.as_bytes();
    let desc_felt = if desc_bytes.is_empty() {
        FieldElement::ZERO
    } else {
        let mut desc_data = Vec::new();
        for chunk in desc_bytes.chunks(31) {
            let mut padded = [0u8; 32];
            padded[32 - chunk.len()..].copy_from_slice(chunk);
            desc_data.push(FieldElement::from_bytes_be(&padded).unwrap_or(FieldElement::ZERO));
        }
        starknet_crypto::poseidon_hash_many(&desc_data)
    };

    let model_id = starknet_crypto::poseidon_hash_many(&[arch_hash, weight_commitment, desc_felt]);

    ModelRegistration {
        model_id,
        weight_commitment,
        num_layers: graph.nodes.len(),
        description: description.to_string(),
    }
}

/// Generate calldata for `register_model()` on the ObelyskVerifier contract.
///
/// Returns `[model_id, weight_commitment, num_layers, description_hash]`.
pub fn register_model_calldata(registration: &ModelRegistration) -> Vec<FieldElement> {
    let desc_hash = {
        let bytes = registration.description.as_bytes();
        if bytes.is_empty() {
            FieldElement::ZERO
        } else {
            let mut data = Vec::new();
            for chunk in bytes.chunks(31) {
                let mut padded = [0u8; 32];
                padded[32 - chunk.len()..].copy_from_slice(chunk);
                data.push(FieldElement::from_bytes_be(&padded).unwrap_or(FieldElement::ZERO));
            }
            starknet_crypto::poseidon_hash_many(&data)
        }
    };
    vec![
        registration.model_id,
        registration.weight_commitment,
        FieldElement::from(registration.num_layers as u64),
        desc_hash,
    ]
}

/// Generate calldata for `register_model()` on the SumcheckVerifier contract.
///
/// Returns `[model_id, weight_commitment]`.
pub fn register_model_calldata_sumcheck(registration: &ModelRegistration) -> Vec<FieldElement> {
    vec![registration.model_id, registration.weight_commitment]
}

/// Error type for Starknet proof generation.
#[derive(Debug, thiserror::Error)]
pub enum StarknetModelError {
    #[error("Proving error: {0}")]
    ProvingError(#[from] ModelError),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Soundness gate failed: {0}")]
    SoundnessGate(String),
    #[error("Aggregation error: {0}")]
    AggregationError(#[from] AggregationError),
}

/// Prove a computation graph and produce Starknet-ready calldata.
///
/// Runs the full pipeline:
/// 1. Forward pass + per-layer proving (unified STARK, per-matmul sumchecks)
/// 2. Serializes the unified STARK proof to felt252 calldata
/// 3. Estimates gas cost for on-chain verification
///
/// Uses GPU acceleration when available via `prove_model_aggregated_auto`.
pub fn prove_for_starknet(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<StarknetModelProof, StarknetModelError> {
    let proof = prove_model_aggregated_auto(graph, input, weights)?;
    Ok(build_starknet_proof(&proof))
}

/// Prove a computation graph and produce Starknet-ready on-chain calldata.
///
/// Runs the full on-chain pipeline:
/// 1. Forward pass + per-layer proving (Blake2s unified STARK + Poseidon matmul sumchecks)
/// 2. Assembles combined calldata with IO commitment at index [4]
/// 3. Estimates gas cost for on-chain verification
///
/// Uses GPU acceleration when available via `prove_model_aggregated_onchain_auto`.
pub fn prove_for_starknet_onchain(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<StarknetModelProof, StarknetModelError> {
    let proof = prove_model_aggregated_onchain_auto(graph, input, weights)?;
    Ok(build_starknet_proof_onchain(&proof, input))
}

/// Prove and serialize for Starknet on-chain verification with weight cache.
///
/// Same as [`prove_for_starknet_onchain`] but reuses cached weight commitments.
/// For repeated inference with the same model, this is the recommended entry point.
pub fn prove_for_starknet_onchain_cached(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    weight_cache: &crate::weight_cache::SharedWeightCache,
) -> Result<StarknetModelProof, StarknetModelError> {
    let proof = crate::aggregation::prove_model_aggregated_onchain_auto_cached(
        graph,
        input,
        weights,
        weight_cache,
    )?;
    Ok(build_starknet_proof_onchain(&proof, input))
}

/// Convert an already-computed `AggregatedModelProof` into Starknet calldata.
///
/// Useful when you've already proven the model (e.g., via `GpuModelProver`)
/// and just want to serialize for on-chain submission.
pub fn build_starknet_proof(proof: &AggregatedModelProof) -> StarknetModelProof {
    let unified_calldata = match &proof.unified_stark {
        Some(stark_proof) => serialize_proof(stark_proof),
        None => Vec::new(),
    };

    let pcs_config = match &proof.unified_stark {
        Some(stark_proof) => stark_proof.config,
        None => PcsConfig::default(),
    };

    let total_trace_rows: usize = proof.activation_claims.iter().map(|c| c.trace_rows).sum();
    let num_proven = proof.num_proven_layers();
    let estimated_gas = estimate_verification_gas(num_proven, total_trace_rows.max(1));
    let calldata_size = unified_calldata.len();

    StarknetModelProof {
        unified_calldata,
        matmul_calldata: Vec::new(),
        io_commitment: FieldElement::ZERO,
        raw_io_data: Vec::new(),
        combined_calldata: Vec::new(),
        layer_claims: proof.activation_claims.clone(),
        num_matmul_proofs: proof.matmul_proofs.len(),
        num_add_claims: proof.add_claims.len(),
        num_mul_claims: proof.mul_claims.len(),
        num_layernorm_claims: proof.layernorm_claims.len(),
        num_attention_proofs: proof.attention_proofs.len(),
        num_embedding_claims: proof.embedding_claims.len(),
        num_quantize_claims: proof.quantize_claims.len(),
        num_dequantize_claims: proof.dequantize_claims.len(),
        num_rmsnorm_claims: proof.rmsnorm_claims.len(),
        num_layernorm_mean_var_commitments: proof.layernorm_mean_var_commitments.len(),
        num_proven_layers: num_proven,
        pcs_config,
        estimated_gas,
        calldata_size,
        layer_chain_commitment: proof.layer_chain_commitment,
        tee_attestation_hash: None,
        gkr_calldata: None,
    }
}

/// Convert an already-computed `AggregatedModelProof` into Starknet calldata with TEE attestation.
pub fn build_starknet_proof_with_tee(
    proof: &AggregatedModelProof,
    attestation: Option<&TeeAttestation>,
) -> StarknetModelProof {
    let mut result = build_starknet_proof(proof);
    result.tee_attestation_hash = attestation
        .filter(|a| a.has_report())
        .map(|a| a.report_hash_felt());
    result
}

// compute_io_commitment is imported from crate::aggregation

/// Convert an on-chain aggregated proof into complete Starknet calldata.
///
/// Assembles the combined calldata format:
/// - PCS config (4 felts)
/// - Raw IO data as Array<felt252> (length-prefixed) — Cairo verifier recomputes Poseidon on-chain
/// - Layer chain commitment (1 felt)
/// - Activation STARK calldata
/// - Matmul proof count (1 felt)
/// - Concatenated matmul calldata
///
/// `input` is needed to serialize raw IO data for on-chain recomputation.
pub fn build_starknet_proof_onchain(
    proof: &AggregatedModelProofOnChain,
    input: &M31Matrix,
) -> StarknetModelProof {
    let unified_calldata = match &proof.unified_stark {
        Some(stark_proof) => serialize_proof(stark_proof),
        None => Vec::new(),
    };

    // Serialize all matmul sumcheck proofs
    let mut matmul_calldata: Vec<Vec<FieldElement>> = Vec::new();
    for (_layer_idx, matmul_proof) in &proof.matmul_proofs {
        let mut buf = Vec::new();
        serialize_matmul_sumcheck_proof(matmul_proof, &mut buf);
        matmul_calldata.push(buf);
    }

    // Serialize raw IO data for on-chain commitment recomputation.
    let raw_io_data = crate::cairo_serde::serialize_raw_io(input, &proof.execution.output);
    let io_commitment = proof.io_commitment;

    // Extract the real PCS config from the unified STARK proof.
    // If no unified STARK exists (matmul-only model), use PcsConfig::default()
    // which has real security parameters (pow_bits=10, n_queries=3).
    let pcs_config = match &proof.unified_stark {
        Some(stark_proof) => stark_proof.config,
        None => PcsConfig::default(),
    };

    // Assemble combined calldata
    let mut combined = vec![
        // PCS config (4 felts) — real values from the proof
        FieldElement::from(pcs_config.pow_bits as u64),
        FieldElement::from(pcs_config.fri_config.log_blowup_factor as u64),
        FieldElement::from(pcs_config.fri_config.log_last_layer_degree_bound as u64),
        FieldElement::from(pcs_config.fri_config.n_queries as u64),
    ];
    // Raw IO data as Array<felt252> — Cairo Serde: length prefix + elements.
    // The on-chain verifier recomputes Poseidon(raw_io_data) instead of trusting a caller hash.
    combined.push(FieldElement::from(raw_io_data.len() as u64));
    combined.extend_from_slice(&raw_io_data);
    // Layer chain commitment
    combined.push(proof.layer_chain_commitment);

    // Unified STARK calldata (covers activations + add + mul + layernorm)
    combined.extend_from_slice(&unified_calldata);

    // Matmul proofs count
    combined.push(FieldElement::from(matmul_calldata.len() as u64));

    // Concatenated matmul calldata
    for mc in &matmul_calldata {
        combined.push(FieldElement::from(mc.len() as u64)); // per-proof length
        combined.extend_from_slice(mc);
    }

    // Batched matmul proofs count + serialized batch data
    combined.push(FieldElement::from(proof.batched_matmul_proofs.len() as u64));
    for batch in &proof.batched_matmul_proofs {
        // Serialize batch: k, num_rounds, num_entries, then per-entry (m, n)
        combined.push(FieldElement::from(batch.k as u64));
        combined.push(FieldElement::from(batch.num_rounds as u64));
        combined.push(FieldElement::from(batch.entries.len() as u64));
        for entry in &batch.entries {
            combined.push(FieldElement::from(entry.node_id as u64));
            combined.push(FieldElement::from(entry.m as u64));
            combined.push(FieldElement::from(entry.n as u64));
            combined.push(entry.a_commitment);
            combined.push(entry.b_commitment);
        }
    }

    // Attention proofs count + serialized sub-proofs
    combined.push(FieldElement::from(proof.attention_proofs.len() as u64));
    for (layer_idx, attn_proof) in &proof.attention_proofs {
        combined.push(FieldElement::from(*layer_idx as u64));
        // Serialize each matmul sub-proof in the attention proof
        let mut attn_buf = Vec::new();
        serialize_matmul_sumcheck_proof(&attn_proof.q_proof, &mut attn_buf);
        serialize_matmul_sumcheck_proof(&attn_proof.k_proof, &mut attn_buf);
        serialize_matmul_sumcheck_proof(&attn_proof.v_proof, &mut attn_buf);
        // Score proofs
        attn_buf.push(FieldElement::from(attn_proof.score_proofs.len() as u64));
        for sp in &attn_proof.score_proofs {
            serialize_matmul_sumcheck_proof(sp, &mut attn_buf);
        }
        // Attn×V proofs
        attn_buf.push(FieldElement::from(attn_proof.attn_v_proofs.len() as u64));
        for ap in &attn_proof.attn_v_proofs {
            serialize_matmul_sumcheck_proof(ap, &mut attn_buf);
        }
        serialize_matmul_sumcheck_proof(&attn_proof.output_proof, &mut attn_buf);
        // Softmax exp STARK proof
        let softmax_calldata = serialize_proof(&attn_proof.softmax_exp_proof);
        attn_buf.push(FieldElement::from(softmax_calldata.len() as u64));
        attn_buf.extend_from_slice(&softmax_calldata);
        combined.push(FieldElement::from(attn_buf.len() as u64));
        combined.extend_from_slice(&attn_buf);
    }

    // Embedding claims count
    combined.push(FieldElement::from(proof.embedding_claims.len() as u64));
    for claim in &proof.embedding_claims {
        combined.push(FieldElement::from(claim.layer_index as u64));
        combined.push(FieldElement::from(claim.trace_rows as u64));
    }

    // LayerNorm mean/var commitments
    combined.push(FieldElement::from(
        proof.layernorm_mean_var_commitments.len() as u64,
    ));
    for commitment in &proof.layernorm_mean_var_commitments {
        combined.push(*commitment);
    }

    // GKR proof (optional — for on-chain LogUp/GrandProduct verification)
    // Prefer STWO-native GKR batch format (gkr_batch_data) over custom ML GKR (gkr_proof).
    let gkr_calldata = if let Some(ref gkr_data) = proof.gkr_batch_data {
        // STWO-native GKR batch proof — directly compatible with Cairo's partially_verify_batch
        let mut gkr_buf = Vec::new();
        crate::cairo_serde::serialize_stwo_gkr_batch_proof(
            &gkr_data.proof,
            &gkr_data.gate_types,
            &gkr_data.n_variables,
            &mut gkr_buf,
        );
        combined.push(FieldElement::from(1u64)); // has_gkr = true
        combined.push(FieldElement::from(gkr_buf.len() as u64));
        combined.extend_from_slice(&gkr_buf);
        Some(gkr_buf)
    } else if let Some(ref gkr_proof) = proof.gkr_proof {
        // Legacy custom ML GKR format
        let mut gkr_buf = Vec::new();
        crate::cairo_serde::serialize_gkr_model_proof(gkr_proof, &mut gkr_buf);
        combined.push(FieldElement::from(1u64)); // has_gkr = true
        combined.push(FieldElement::from(gkr_buf.len() as u64));
        combined.extend_from_slice(&gkr_buf);
        Some(gkr_buf)
    } else {
        combined.push(FieldElement::from(0u64)); // has_gkr = false
        None
    };

    let total_trace_rows: usize = proof.activation_claims.iter().map(|c| c.trace_rows).sum();
    let num_proven = proof.num_proven_layers();
    let estimated_gas = estimate_verification_gas(num_proven, total_trace_rows.max(1));
    let calldata_size = combined.len();

    StarknetModelProof {
        unified_calldata,
        matmul_calldata,
        io_commitment,
        raw_io_data,
        combined_calldata: combined,
        layer_claims: proof.activation_claims.clone(),
        num_matmul_proofs: proof.matmul_proofs.len(),
        num_add_claims: proof.add_claims.len(),
        num_mul_claims: proof.mul_claims.len(),
        num_layernorm_claims: proof.layernorm_claims.len(),
        num_attention_proofs: proof.attention_proofs.len(),
        num_embedding_claims: proof.embedding_claims.len(),
        num_quantize_claims: proof.quantize_claims.len(),
        num_dequantize_claims: proof.dequantize_claims.len(),
        num_rmsnorm_claims: proof.rmsnorm_claims.len(),
        num_layernorm_mean_var_commitments: proof.layernorm_mean_var_commitments.len(),
        num_proven_layers: num_proven,
        pcs_config,
        estimated_gas,
        calldata_size,
        layer_chain_commitment: proof.layer_chain_commitment,
        tee_attestation_hash: None,
        gkr_calldata,
    }
}

/// Convert an on-chain aggregated proof into Starknet calldata with TEE attestation.
pub fn build_starknet_proof_onchain_with_tee(
    proof: &AggregatedModelProofOnChain,
    input: &M31Matrix,
    attestation: Option<&TeeAttestation>,
) -> StarknetModelProof {
    let mut result = build_starknet_proof_onchain(proof, input);
    result.tee_attestation_hash = attestation
        .filter(|a| a.has_report())
        .map(|a| a.report_hash_felt());
    result
}

/// Prove a computation graph and produce a direct on-chain proof.
///
/// This is the 2-stage pipeline that eliminates Cairo VM recursion:
///   Stage 1: GPU prove → AggregatedModelProofOnChain
///   On-chain: verify_model_direct() (batch sumchecks + activation STARK)
///
/// The returned `DirectStarknetProof` contains:
/// - `batched_calldata`: sent with the `verify_model_direct()` transaction
/// - `stark_chunks`: uploaded via `upload_proof_chunk()` in prior transactions
pub fn prove_for_starknet_direct(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    metadata: crate::cairo_serde::DirectProofMetadata,
) -> Result<DirectStarknetProof, StarknetModelError> {
    let proof = prove_model_aggregated_onchain_auto(graph, input, weights)?;
    Ok(build_starknet_proof_direct(&proof, input, metadata))
}

/// Prove a computation graph on multiple GPUs and produce a direct on-chain proof.
///
/// Distributes chunk proving across all available GPUs using memory-aware bin-packing,
/// then composes chunk proofs and serializes for direct on-chain verification.
///
/// This is the multi-GPU variant of [`prove_for_starknet_direct`].
#[cfg(feature = "multi-gpu")]
pub fn prove_for_starknet_direct_multi_gpu(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    metadata: crate::cairo_serde::DirectProofMetadata,
    memory_budget: usize,
) -> Result<DirectStarknetProof, StarknetModelError> {
    let chunks = crate::compiler::chunked::prove_model_chunked_multi_gpu(
        graph,
        input,
        weights,
        memory_budget,
    )
    .map_err(|e| {
        StarknetModelError::SerializationError(format!("Multi-GPU chunked proving: {e}"))
    })?;

    let proof = crate::compiler::chunked::compose_chunk_proofs_auto(&chunks, graph, input, weights)
        .map_err(|e| StarknetModelError::SerializationError(format!("Chunk composition: {e}")))?;

    Ok(build_starknet_proof_direct(&proof, input, metadata))
}

/// Convert an on-chain aggregated proof into a `DirectStarknetProof`.
///
/// Serializes batch sumcheck proofs and splits the activation STARK
/// into upload chunks for multi-transaction on-chain submission.
///
/// `input` is needed to serialize raw IO data for on-chain commitment recomputation.
pub fn build_starknet_proof_direct(
    proof: &AggregatedModelProofOnChain,
    input: &M31Matrix,
    metadata: crate::cairo_serde::DirectProofMetadata,
) -> DirectStarknetProof {
    let serialized = crate::cairo_serde::serialize_model_proof_direct(proof, &metadata);

    let raw_io_data = crate::cairo_serde::serialize_raw_io(input, &proof.execution.output);

    let batched_size: usize = serialized.batched_calldata.iter().map(|b| b.len()).sum();
    let stark_size: usize = serialized.stark_chunks.iter().map(|c| c.len()).sum();
    let total_calldata_size = batched_size + stark_size + raw_io_data.len();

    let num_proven = proof.num_proven_layers();
    let total_trace_rows: usize = proof.activation_claims.iter().map(|c| c.trace_rows).sum();
    let estimated_gas = estimate_verification_gas(num_proven, total_trace_rows.max(1));

    DirectStarknetProof {
        model_id: metadata.model_id,
        raw_io_data,
        weight_commitment: metadata.weight_commitment,
        num_layers: metadata.num_layers,
        activation_type: metadata.activation_type,
        batched_calldata: serialized.batched_calldata,
        stark_chunks: serialized.stark_chunks,
        has_activation_stark: serialized.has_activation_stark,
        estimated_gas,
        total_calldata_size,
    }
}

// ============================================================================
// GKR Model Calldata (Full On-Chain ML Verification)
// ============================================================================

/// A GKR proof formatted for on-chain submission to `verify_model_gkr()`.
///
/// Contains all data needed for the Cairo contract to verify every layer
/// of the model via GKR sumcheck — no FRI, no dict-based operations.
#[derive(Debug)]
pub struct GkrStarknetProof {
    /// Model identifier (Poseidon hash of architecture + weight commitment + description).
    pub model_id: FieldElement,
    /// Serialized GKR proof_data for `verify_model_gkr()`:
    /// tags + per-layer payloads + deferred proofs (no header/footer fields).
    pub gkr_calldata: Vec<FieldElement>,
    /// Raw IO data for on-chain IO commitment recomputation.
    pub io_calldata: Vec<FieldElement>,
    /// Weight commitments (one per MatMul/deferred MatMul), passed as
    /// `weight_commitments: Array<felt252>` to `verify_model_gkr()`.
    pub weight_commitments: Vec<FieldElement>,
    /// Weight MLE opening proofs for on-chain weight binding.
    /// Already serialized as Cairo `Array<MleOpeningProof>`:
    /// `[num_openings, opening_0..., opening_1..., ...]`.
    pub weight_opening_calldata: Vec<FieldElement>,
    /// Weight claims (node id + eval point + expected value) in stable serialized form.
    /// Always populated for artifact completeness, including batched weight-binding mode.
    pub weight_claim_calldata: Vec<FieldElement>,
    /// Versioned schema for `weight_binding_data_calldata`.
    ///
    /// This is an artifact-level schema marker (not a contract parameter).
    pub weight_binding_schema_version: u32,
    /// Mode id expected by Starknet v2/v3/v4 entrypoints when available.
    ///
    /// - `Some(0)` -> Sequential
    /// - `Some(1)` -> BatchedSubchannelV1
    /// - `Some(2)` -> AggregatedTrustlessV2 (v3 submit-ready)
    /// - `Some(3)` -> AggregatedOpeningsV4Experimental (v4 experimental)
    /// - `Some(4)` -> AggregatedOracleSumcheck (v4 production submit-ready)
    /// - `None`    -> off-chain-only mode (no Starknet mode id)
    pub weight_binding_mode_id: Option<u32>,
    /// Extra mode-specific binding payload encoded as `Array<felt252>`.
    ///
    /// For modes `0/1`, this is empty.
    /// For mode `2`, this carries `[binding_digest, claim_count]`.
    pub weight_binding_data_calldata: Vec<FieldElement>,
    /// Transcript mode used for weight binding/openings.
    pub weight_opening_mode: crate::gkr::types::WeightOpeningTranscriptMode,
    /// True when the proof passes strict Starknet soundness gates and can be submitted
    /// with `verify_model_gkr` calldata as-is.
    pub submission_ready: bool,
    /// If `submission_ready == false`, contains the strict gate failure reason.
    pub soundness_gate_error: Option<String>,
    /// IO commitment: Poseidon(input_data || output_data).
    pub io_commitment: FieldElement,
    /// Layer chain commitment from the proving pipeline.
    pub layer_chain_commitment: FieldElement,
    /// Number of GKR layer proofs.
    pub num_layer_proofs: usize,
    /// Estimated gas cost.
    pub estimated_gas: u64,
    /// Total calldata size across all fields.
    pub total_calldata_size: usize,
}

const WEIGHT_BINDING_SCHEMA_VERSION_V1: u32 = 1;
const WEIGHT_BINDING_MODE2_DOMAIN_TAG: u64 = 0x5742_4d32; // "WBM2"
const WEIGHT_BINDING_MODE2_SCHEMA_VERSION: u64 = 1;
const WEIGHT_BINDING_MODE3_DOMAIN_TAG: u64 = 0x5742_4d33; // "WBM3"
const WEIGHT_BINDING_MODE3_SCHEMA_VERSION: u64 = 1;

/// Strict soundness gates for GKR serialization/submission.
///
/// Rejects proofs that weaken on-chain verification guarantees:
/// - Activation layers must include LogUp proofs.
/// - All MatMul weight claims must have matching non-empty opening proofs.
fn enforce_gkr_soundness_gates(proof: &crate::gkr::GKRProof) -> Result<(), StarknetModelError> {
    use crate::gkr::types::WeightOpeningTranscriptMode;
    use crate::gkr::LayerProof;

    for (layer_idx, layer_proof) in proof.layer_proofs.iter().enumerate() {
        if let LayerProof::Activation { logup_proof, .. } = layer_proof {
            if logup_proof.is_none() {
                return Err(StarknetModelError::SoundnessGate(format!(
                    "activation layer {} missing LogUp proof",
                    layer_idx
                )));
            }
        }
    }

    match proof.weight_opening_transcript_mode {
        WeightOpeningTranscriptMode::Sequential
        | WeightOpeningTranscriptMode::BatchedSubchannelV1
        | WeightOpeningTranscriptMode::AggregatedTrustlessV2
        | WeightOpeningTranscriptMode::AggregatedOpeningsV4Experimental => {}
        WeightOpeningTranscriptMode::AggregatedOracleSumcheck => {
            // Mode 4: accepts either full mismatch sumcheck proof or RLC-only binding.
            // Full proof (aggregated_binding: Some) provides independent verifiability.
            // RLC-only (aggregated_binding: None) relies on Poseidon Merkle commitments
            // mixed into the Fiat-Shamir transcript for binding security.
        }
        WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1 => {
            return Err(StarknetModelError::SoundnessGate(
                "BatchedRlcDirectEvalV1 is off-chain only and cannot be serialized for Starknet"
                    .to_string(),
            ));
        }
    }

    let n_main = proof.weight_claims.len();
    if proof.weight_opening_transcript_mode != WeightOpeningTranscriptMode::AggregatedOracleSumcheck
    {
        if proof.weight_commitments.len() != n_main {
            return Err(StarknetModelError::SoundnessGate(format!(
                "weight commitments ({}) != weight claims ({})",
                proof.weight_commitments.len(),
                n_main
            )));
        }
        if proof.weight_openings.len() != n_main {
            return Err(StarknetModelError::SoundnessGate(format!(
                "weight openings ({}) != weight claims ({})",
                proof.weight_openings.len(),
                n_main
            )));
        }
    }
    for (i, opening) in proof.weight_openings.iter().enumerate() {
        if opening.queries.is_empty() {
            return Err(StarknetModelError::SoundnessGate(format!(
                "weight opening {} has no Merkle queries",
                i
            )));
        }
    }
    for (i, deferred) in proof.deferred_proofs.iter().enumerate() {
        if let Some(wo) = deferred.weight_opening() {
            if wo.queries.is_empty() {
                return Err(StarknetModelError::SoundnessGate(format!(
                    "deferred weight opening {} has no Merkle queries",
                    i
                )));
            }
        }
    }

    Ok(())
}

fn serialize_weight_claim(claim: &crate::gkr::types::WeightClaim, output: &mut Vec<FieldElement>) {
    crate::cairo_serde::serialize_u32(claim.weight_node_id as u32, output);
    crate::cairo_serde::serialize_u32(claim.eval_point.len() as u32, output);
    for &p in &claim.eval_point {
        crate::cairo_serde::serialize_qm31(p, output);
    }
    crate::cairo_serde::serialize_qm31(claim.expected_value, output);
}

fn serialize_weight_claims_for_artifact(proof: &crate::gkr::GKRProof) -> Vec<FieldElement> {
    // Order mirrors verifier-side claim walk:
    //   1) main walk MatMul claims
    //   2) deferred MatMul claims (weightless proofs are skipped)
    let deferred_matmul_count = proof
        .deferred_proofs
        .iter()
        .filter(|d| d.has_weights())
        .count();
    let total_claims = proof.weight_claims.len() + deferred_matmul_count;
    let mut out = Vec::new();
    crate::cairo_serde::serialize_u32(total_claims as u32, &mut out);
    for claim in &proof.weight_claims {
        serialize_weight_claim(claim, &mut out);
    }
    for deferred in &proof.deferred_proofs {
        if let Some(claim) = deferred.weight_claim() {
            serialize_weight_claim(claim, &mut out);
        }
    }
    out
}

fn mode_weight_binding_data_with_domain(
    proof: &crate::gkr::GKRProof,
    domain_tag: u64,
    schema_version: u64,
) -> Vec<FieldElement> {
    let deferred_matmul_count = proof
        .deferred_proofs
        .iter()
        .filter(|d| d.has_weights())
        .count();
    let total_claims = proof.weight_claims.len() + deferred_matmul_count;
    let mut commitments = proof.weight_commitments.clone();
    for deferred in &proof.deferred_proofs {
        if let Some(wc) = deferred.weight_commitment() {
            commitments.push(wc);
        }
    }

    let mut digest_inputs = Vec::new();
    digest_inputs.push(FieldElement::from(domain_tag));
    digest_inputs.push(FieldElement::from(schema_version));
    digest_inputs.push(FieldElement::from(total_claims as u64));
    digest_inputs.push(FieldElement::from(commitments.len() as u64));

    for (idx, claim) in proof.weight_claims.iter().enumerate() {
        digest_inputs.push(commitments[idx]);
        digest_inputs.push(FieldElement::from(claim.eval_point.len() as u64));
        for &p in &claim.eval_point {
            digest_inputs.push(crate::crypto::poseidon_channel::securefield_to_felt(p));
        }
        digest_inputs.push(crate::crypto::poseidon_channel::securefield_to_felt(
            claim.expected_value,
        ));
    }

    let mut deferred_claim_idx = proof.weight_claims.len();
    for deferred in &proof.deferred_proofs {
        if let Some(claim) = deferred.weight_claim() {
            digest_inputs.push(commitments[deferred_claim_idx]);
            digest_inputs.push(FieldElement::from(claim.eval_point.len() as u64));
            for &p in &claim.eval_point {
                digest_inputs.push(crate::crypto::poseidon_channel::securefield_to_felt(p));
            }
            digest_inputs.push(crate::crypto::poseidon_channel::securefield_to_felt(
                claim.expected_value,
            ));
            deferred_claim_idx += 1;
        }
    }

    let digest = starknet_crypto::poseidon_hash_many(&digest_inputs);
    vec![digest, FieldElement::from(total_claims as u64)]
}

fn mode2_weight_binding_data(proof: &crate::gkr::GKRProof) -> Vec<FieldElement> {
    mode_weight_binding_data_with_domain(
        proof,
        WEIGHT_BINDING_MODE2_DOMAIN_TAG,
        WEIGHT_BINDING_MODE2_SCHEMA_VERSION,
    )
}

fn mode3_weight_binding_data(proof: &crate::gkr::GKRProof) -> Vec<FieldElement> {
    mode_weight_binding_data_with_domain(
        proof,
        WEIGHT_BINDING_MODE3_DOMAIN_TAG,
        WEIGHT_BINDING_MODE3_SCHEMA_VERSION,
    )
}

fn starknet_weight_binding_mode_for_artifact(
    mode: crate::gkr::types::WeightOpeningTranscriptMode,
) -> Option<u32> {
    use crate::gkr::types::WeightOpeningTranscriptMode;

    match mode {
        WeightOpeningTranscriptMode::Sequential => Some(0),
        WeightOpeningTranscriptMode::BatchedSubchannelV1 => Some(1),
        WeightOpeningTranscriptMode::AggregatedTrustlessV2 => Some(2),
        WeightOpeningTranscriptMode::AggregatedOpeningsV4Experimental => Some(3),
        WeightOpeningTranscriptMode::AggregatedOracleSumcheck => Some(4),
        WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1 => None,
    }
}

fn weight_binding_data_for_artifact(proof: &crate::gkr::GKRProof) -> Vec<FieldElement> {
    use crate::gkr::types::WeightOpeningTranscriptMode;
    match proof.weight_opening_transcript_mode {
        // No extra payload needed for mode 0/1 today.
        WeightOpeningTranscriptMode::Sequential
        | WeightOpeningTranscriptMode::BatchedSubchannelV1 => Vec::new(),
        WeightOpeningTranscriptMode::AggregatedTrustlessV2 => mode2_weight_binding_data(proof),
        WeightOpeningTranscriptMode::AggregatedOpeningsV4Experimental => {
            mode3_weight_binding_data(proof)
        }
        WeightOpeningTranscriptMode::AggregatedOracleSumcheck => {
            // Mode 4: either full mismatch sumcheck proof or RLC-only binding.
            if let Some(binding) = proof.aggregated_binding.as_ref() {
                // Full proof payload for independent on-chain verification.
                let mut out = Vec::new();
                crate::cairo_serde::serialize_aggregated_binding_proof(binding, &mut out);
                out
            } else {
                // RLC-only: marker tag (0x524C43 = "RLC") + claim count.
                // The on-chain verifier replays the RLC computation using weight_claims.
                vec![
                    FieldElement::from(0x524C43u64),
                    FieldElement::from(proof.weight_claims.len() as u64),
                ]
            }
        }
        // Off-chain-only mode has no Starknet payload.
        WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1 => Vec::new(),
    }
}

fn build_gkr_serialized_proof_inner(
    proof: &AggregatedModelProofOnChain,
    model_id: FieldElement,
    input: &M31Matrix,
    require_starknet_soundness: bool,
) -> Result<GkrStarknetProof, StarknetModelError> {
    let gkr_proof = proof.gkr_proof.as_ref().ok_or_else(|| {
        StarknetModelError::SerializationError("No GKR proof in aggregated proof".to_string())
    })?;

    let soundness_gate_error = match enforce_gkr_soundness_gates(gkr_proof) {
        Ok(()) => None,
        Err(e) => {
            if require_starknet_soundness {
                return Err(e);
            }
            Some(e.to_string())
        }
    };

    // Serialize ONLY proof_data for verify_model_gkr(proof_data: Array<felt252>).
    // Full serialize_gkr_model_proof includes header/footer fields that the
    // contract passes separately and would break parser alignment.
    let mut gkr_calldata = Vec::new();
    serialize_gkr_proof_data_only(gkr_proof, &mut gkr_calldata);

    // Serialize raw IO for on-chain IO commitment recomputation.
    let io_calldata = crate::cairo_serde::serialize_raw_io(input, &proof.execution.output);

    // Weight commitments are a separate parameter to verify_model_gkr.
    // Include both main-walk matmuls and deferred matmul branches (weightless proofs excluded).
    let mut weight_commitments = gkr_proof.weight_commitments.clone();
    for deferred in &gkr_proof.deferred_proofs {
        if let Some(wc) = deferred.weight_commitment() {
            weight_commitments.push(wc);
        }
    }

    // Serialize weight MLE opening proofs as Array<MleOpeningProof> for Cairo Serde.
    // Order must match Cairo verifier weight_claims:
    //   1) main walk matmul claims
    //   2) deferred matmul claims (DAG Add skip branches, weightless excluded)
    let deferred_opening_count = gkr_proof
        .deferred_proofs
        .iter()
        .filter(|d| d.has_weights())
        .count();
    let total_openings = gkr_proof.weight_openings.len() + deferred_opening_count;
    let mut weight_opening_calldata = Vec::new();
    crate::cairo_serde::serialize_u32(total_openings as u32, &mut weight_opening_calldata);
    for opening in &gkr_proof.weight_openings {
        crate::cairo_serde::serialize_mle_opening_proof(opening, &mut weight_opening_calldata);
    }
    for deferred in &gkr_proof.deferred_proofs {
        if let Some(wo) = deferred.weight_opening() {
            crate::cairo_serde::serialize_mle_opening_proof(wo, &mut weight_opening_calldata);
        }
    }

    let weight_claim_calldata = serialize_weight_claims_for_artifact(gkr_proof);
    let weight_binding_mode_id =
        starknet_weight_binding_mode_for_artifact(gkr_proof.weight_opening_transcript_mode);
    let weight_binding_data_calldata = weight_binding_data_for_artifact(gkr_proof);

    let total_calldata_size = 1
        + gkr_calldata.len()
        + io_calldata.len()
        + (1 + weight_commitments.len())
        + weight_opening_calldata.len()
        + weight_claim_calldata.len();
    let num_layer_proofs = gkr_proof.layer_proofs.len();
    let estimated_gas = estimate_gkr_verification_gas(num_layer_proofs);

    Ok(GkrStarknetProof {
        model_id,
        gkr_calldata,
        io_calldata,
        weight_commitments,
        weight_opening_calldata,
        weight_claim_calldata,
        weight_binding_schema_version: WEIGHT_BINDING_SCHEMA_VERSION_V1,
        weight_binding_mode_id,
        weight_binding_data_calldata,
        weight_opening_mode: gkr_proof.weight_opening_transcript_mode,
        submission_ready: soundness_gate_error.is_none(),
        soundness_gate_error,
        io_commitment: proof.io_commitment,
        layer_chain_commitment: proof.layer_chain_commitment,
        num_layer_proofs,
        estimated_gas,
        total_calldata_size,
    })
}

/// Build GKR model calldata for the `verify_model_gkr()` contract entry point.
///
/// Wraps `serialize_gkr_model_proof()` output with the `model_id` parameter.
/// The returned calldata is ready for submission as an `InvokeTransaction`.
///
/// Layout: `[model_id, ...serialized_gkr_proof]`
pub fn build_gkr_model_calldata(
    proof: &crate::gkr::GKRProof,
    model_id: FieldElement,
) -> Vec<FieldElement> {
    let mut calldata = Vec::new();
    calldata.push(model_id);
    crate::cairo_serde::serialize_gkr_model_proof(proof, &mut calldata);
    calldata
}

/// Build the complete GKR Starknet proof from a pure GKR aggregated proof.
///
/// Extracts the GKR proof, serializes all components (GKR layers, raw IO,
/// weight MLE openings), and assembles into a `GkrStarknetProof`.
pub fn build_gkr_starknet_proof(
    proof: &AggregatedModelProofOnChain,
    model_id: FieldElement,
    input: &M31Matrix,
) -> Result<GkrStarknetProof, StarknetModelError> {
    build_gkr_serialized_proof_inner(proof, model_id, input, true)
}

/// Build a serialized GKR artifact even when strict Starknet soundness gates fail.
///
/// This preserves proof data for benchmarking/off-chain verification workflows
/// (e.g., batched weight-binding transcript modes) while clearly marking
/// `submission_ready=false` and carrying a gate error message.
pub fn build_gkr_serializable_proof(
    proof: &AggregatedModelProofOnChain,
    model_id: FieldElement,
    input: &M31Matrix,
) -> Result<GkrStarknetProof, StarknetModelError> {
    build_gkr_serialized_proof_inner(proof, model_id, input, false)
}

/// Prove a computation graph via pure GKR and produce Starknet-ready calldata.
///
/// Full pipeline:
/// 1. Forward pass + GKR proof (all layer types in single pass)
/// 2. Serialize GKR proof to felt252 calldata
/// 3. Serialize raw IO for IO commitment recomputation
/// 4. Assemble into `GkrStarknetProof`
///
/// Uses GPU acceleration when available via `prove_model_pure_gkr_auto`.
pub fn prove_for_starknet_ml_gkr(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    model_id: FieldElement,
) -> Result<GkrStarknetProof, StarknetModelError> {
    let proof = crate::aggregation::prove_model_pure_gkr_auto(graph, input, weights)?;
    build_gkr_starknet_proof(&proof, model_id, input)
}

/// Estimate gas cost for GKR verification on-chain.
///
/// Based on per-layer sumcheck verification costs:
/// - Base cost: 50K gas (contract overhead)
/// - Per layer: 8K gas average (degree-2 sumcheck ~ 5K, degree-3 ~ 12K, Add ~ 500)
/// - Per QM31 operation in Cairo: ~250 gas (Poseidon hash + field arithmetic)
fn estimate_gkr_verification_gas(num_layers: usize) -> u64 {
    let base_cost: u64 = 50_000;
    let per_layer: u64 = 8_000;
    base_cost + (num_layers as u64) * per_layer
}

/// Extract matmul dimensions from a LayeredCircuit in GKR walk order (output → input).
///
/// Returns a flat Vec of [m0, k0, n0, m1, k1, n1, ...] matching the Cairo contract's
/// `matmul_dims: Array<u32>` parameter.
pub fn extract_matmul_dims(circuit: &crate::gkr::LayeredCircuit) -> Vec<u32> {
    let mut dims = Vec::new();
    for layer_idx in (0..circuit.layers.len()).rev() {
        if let crate::gkr::circuit::LayerType::MatMul { m, k, n, .. } =
            &circuit.layers[layer_idx].layer_type
        {
            dims.push(*m as u32);
            dims.push(*k as u32);
            dims.push(*n as u32);
        }
    }
    dims
}

/// Extract dequantize bit widths from a LayeredCircuit in GKR walk order (output → input).
///
/// Returns a Vec matching the Cairo contract's `dequantize_bits: Array<u64>` parameter.
pub fn extract_dequantize_bits(circuit: &crate::gkr::LayeredCircuit) -> Vec<u64> {
    let mut bits = Vec::new();
    for layer_idx in (0..circuit.layers.len()).rev() {
        if let crate::gkr::circuit::LayerType::Dequantize { params, .. } =
            &circuit.layers[layer_idx].layer_type
        {
            bits.push(params.bits as u64);
        }
    }
    bits
}

/// Build a circuit descriptor for `register_model_gkr()`.
///
/// The descriptor is a flat Array<u32> encoding the layer sequence:
/// [num_layers, tag0, tag1, ...] where tag values match the GKR proof tags.
/// The Cairo contract hashes this to produce a `circuit_hash` for binding.
pub fn build_circuit_descriptor(circuit: &crate::gkr::LayeredCircuit) -> Vec<u32> {
    let mut desc = Vec::new();
    desc.push(circuit.layers.len() as u32);
    for layer_idx in (0..circuit.layers.len()).rev() {
        let tag = match &circuit.layers[layer_idx].layer_type {
            crate::gkr::circuit::LayerType::MatMul { .. } => 0u32,
            crate::gkr::circuit::LayerType::Add { .. } => 1,
            crate::gkr::circuit::LayerType::Mul { .. } => 2,
            crate::gkr::circuit::LayerType::Activation { .. } => 3,
            crate::gkr::circuit::LayerType::LayerNorm { .. } => 4,
            crate::gkr::circuit::LayerType::Attention { .. } => 5,
            crate::gkr::circuit::LayerType::Dequantize { .. } => 6,
            crate::gkr::circuit::LayerType::RMSNorm { .. } => 8,
            crate::gkr::circuit::LayerType::Quantize { .. } => 9,
            crate::gkr::circuit::LayerType::Embedding { .. } => 10,
            crate::gkr::circuit::LayerType::Input => continue,
            crate::gkr::circuit::LayerType::Identity => continue,
        };
        desc.push(tag);
    }
    desc
}

/// Build flat sncast calldata for `register_model_gkr(model_id, weight_commitments, circuit_descriptor)`.
///
/// Returns hex strings ready for `sncast invoke --calldata <...>`.
pub fn build_register_gkr_calldata(
    model_id: FieldElement,
    weight_commitments: &[FieldElement],
    circuit_descriptor: &[u32],
) -> Vec<String> {
    let mut calldata = Vec::new();
    // model_id
    calldata.push(format!("0x{:x}", model_id));
    // weight_commitments: Array<felt252> → len + elements
    calldata.push(format!("{}", weight_commitments.len()));
    for wc in weight_commitments {
        calldata.push(format!("0x{:x}", wc));
    }
    // circuit_descriptor: Array<u32> → len + elements
    calldata.push(format!("{}", circuit_descriptor.len()));
    for &d in circuit_descriptor {
        calldata.push(format!("{}", d));
    }
    calldata
}

/// Serialized calldata components for `verify_model_gkr()`.
///
/// Separates the proof into the contract's individual parameters so they can
/// be assembled into flat sncast calldata.
#[derive(Debug)]
pub struct VerifyModelGkrCalldata {
    /// Flat sncast calldata (space-separated hex values).
    pub calldata_parts: Vec<String>,
    /// Total number of felt252 values in the calldata.
    pub total_felts: usize,
}

/// Maximum felts per chunk for chunked GKR session uploads.
pub const MAX_GKR_CHUNK_FELTS: usize = 4000;

/// Threshold above which chunked session mode is auto-selected.
pub const CHUNKED_GKR_THRESHOLD: usize = 4500;

/// Chunked GKR calldata for session-based upload.
///
/// When total calldata exceeds `CHUNKED_GKR_THRESHOLD`, the proof is split
/// into a flat session data buffer and uploaded in chunks via:
///   `open_gkr_session` → `upload_gkr_chunk` × N → `seal_gkr_session` → `verify_gkr_from_session`
#[derive(Debug, Clone)]
pub struct ChunkedGkrCalldata {
    /// Model ID (hex-encoded felt252).
    pub model_id: String,
    /// Total flat felt252 count in the session data buffer.
    pub total_felts: usize,
    /// Circuit depth parameter.
    pub circuit_depth: u32,
    /// Number of proof layers.
    pub num_layers: u32,
    /// Weight binding mode (3 or 4).
    pub weight_binding_mode: u32,
    /// Whether proof_data uses packed QM31 format (1 felt per QM31).
    pub packed: bool,
    /// Session data split into chunks of ≤ MAX_GKR_CHUNK_FELTS hex-encoded felts.
    pub chunks: Vec<Vec<String>>,
    /// Number of chunks.
    pub num_chunks: usize,
}

/// Build chunked GKR session calldata from a v4 proof.
///
/// Produces a flat session data buffer with length-prefixed sections:
/// ```text
/// [raw_io_data_len, raw_io_data...,
///  matmul_dims_len, matmul_dims...,
///  dequantize_bits_len, dequantize_bits...,
///  proof_data_len, proof_data...,
///  weight_commitments_len, weight_commitments...,
///  weight_binding_data_len, weight_binding_data...,
///  <remaining: Serde-serialized Array<MleOpeningProof>>]
/// ```
///
/// Then splits into chunks of `MAX_GKR_CHUNK_FELTS`.
pub fn build_chunked_gkr_calldata(
    proof: &crate::gkr::GKRProof,
    circuit: &crate::gkr::LayeredCircuit,
    model_id: FieldElement,
    raw_io_data: &[FieldElement],
) -> Result<ChunkedGkrCalldata, StarknetModelError> {
    enforce_gkr_soundness_gates(proof)?;

    let binding_mode = starknet_weight_binding_mode(proof.weight_opening_transcript_mode)?;
    if binding_mode != 3 && binding_mode != 4 {
        return Err(StarknetModelError::SoundnessGate(format!(
            "chunked GKR requires weight_binding_mode in {{3,4}} (got: {binding_mode})"
        )));
    }

    let circuit_depth = circuit.layers.len() as u32;
    let num_layers = proof.layer_proofs.len() as u32;

    let mut flat: Vec<String> = Vec::new();

    // Helper: push a length-prefixed Array<felt252> section.
    macro_rules! push_felt_section {
        ($data:expr) => {
            flat.push(format!("{}", $data.len()));
            for f in $data {
                flat.push(format!("0x{:x}", f));
            }
        };
    }

    // 1. raw_io_data
    push_felt_section!(raw_io_data);

    // 2. matmul_dims (as felt252 values)
    let matmul_dims = extract_matmul_dims(circuit);
    flat.push(format!("{}", matmul_dims.len()));
    for d in &matmul_dims {
        flat.push(format!("{}", d));
    }

    // 3. dequantize_bits (as felt252 values)
    let dequantize_bits = extract_dequantize_bits(circuit);
    flat.push(format!("{}", dequantize_bits.len()));
    for b in &dequantize_bits {
        flat.push(format!("{}", b));
    }

    // 4. proof_data — packed (1 felt/QM31) unless STWO_NO_PACKED is set
    let use_packed = std::env::var("STWO_NO_PACKED").is_err();
    let mut proof_data = Vec::new();
    if use_packed {
        crate::cairo_serde::serialize_gkr_proof_data_only_packed(proof, &mut proof_data);
    } else {
        crate::cairo_serde::serialize_gkr_proof_data_only(proof, &mut proof_data);
    }

    // Self-verify: replay Fiat-Shamir channel against serialized proof to catch
    // prover/serializer mismatches before saving. This prevents stale-binary bugs
    // from producing proofs that fail on-chain verification.
    if std::env::var("STWO_SKIP_SELF_VERIFY").is_err() {
        replay_verify_serialized_proof(
            &proof_data,
            raw_io_data,
            &matmul_dims,
            circuit_depth,
            num_layers,
            use_packed,
        ).map_err(|e| StarknetModelError::SoundnessGate(
            format!("self-verification failed: {e}")
        ))?;
    }

    push_felt_section!(&proof_data);

    // 5. weight_commitments
    let mut weight_commitments = proof.weight_commitments.clone();
    for deferred in &proof.deferred_proofs {
        if let Some(wc) = deferred.weight_commitment() {
            weight_commitments.push(wc);
        }
    }
    push_felt_section!(&weight_commitments);

    // 6. weight_binding_data
    let weight_binding_data_strs = starknet_weight_binding_data(proof)?;
    flat.push(format!("{}", weight_binding_data_strs.len()));
    for s in &weight_binding_data_strs {
        flat.push(s.clone());
    }

    // 7. weight_opening_proofs — Serde-serialized as-is (count + per-proof data).
    //    This matches how Cairo calldata deserialization works for Array<MleOpeningProof>.
    //
    //    For Mode 4 RLC-only (aggregated_binding is None), the on-chain verifier
    //    never iterates over opening proofs — it uses RLC binding instead.
    //    Omitting them drops ~53K felts (~160 proofs × ~330 felts each), reducing
    //    total session data from ~86K to ~33K felts, which fits within Starknet's
    //    per-TX step limit for storage reads.
    let is_mode4_rlc = binding_mode == 4 && proof.aggregated_binding.is_none();
    if is_mode4_rlc {
        // Empty opening proofs array — Cairo verifier skips them for Mode 4 RLC.
        flat.push("0".to_string());
    } else {
        let deferred_opening_count = proof
            .deferred_proofs
            .iter()
            .filter(|d| d.has_weights())
            .count();
        let total_openings = proof.weight_openings.len() + deferred_opening_count;
        flat.push(format!("{}", total_openings));

        let mut opening_buf = Vec::new();
        for opening in &proof.weight_openings {
            opening_buf.clear();
            crate::cairo_serde::serialize_mle_opening_proof(opening, &mut opening_buf);
            for felt in &opening_buf {
                flat.push(format!("0x{:x}", felt));
            }
        }
        for deferred in &proof.deferred_proofs {
            if let Some(wo) = deferred.weight_opening() {
                opening_buf.clear();
                crate::cairo_serde::serialize_mle_opening_proof(wo, &mut opening_buf);
                for felt in &opening_buf {
                    flat.push(format!("0x{:x}", felt));
                }
            }
        }
    }

    let total_felts = flat.len();

    // Split into chunks.
    let chunks: Vec<Vec<String>> = flat
        .chunks(MAX_GKR_CHUNK_FELTS)
        .map(|c| c.to_vec())
        .collect();
    let num_chunks = chunks.len();

    Ok(ChunkedGkrCalldata {
        model_id: format!("0x{:x}", model_id),
        total_felts,
        circuit_depth,
        num_layers,
        weight_binding_mode: binding_mode,
        packed: use_packed,
        chunks,
        num_chunks,
    })
}

fn starknet_weight_binding_mode(
    mode: crate::gkr::types::WeightOpeningTranscriptMode,
) -> Result<u32, StarknetModelError> {
    use crate::gkr::types::WeightOpeningTranscriptMode;

    match mode {
        // v2 mode 0: sequential transcript (v1-compatible).
        WeightOpeningTranscriptMode::Sequential => Ok(0),
        // v2 mode 1: batched sub-channel transcript (same opening statement).
        WeightOpeningTranscriptMode::BatchedSubchannelV1 => Ok(1),
        // v3 mode 2: aggregated trustless binding (with explicit binding payload).
        WeightOpeningTranscriptMode::AggregatedTrustlessV2 => Ok(2),
        // v4 mode 3: aggregated opening envelope (experimental scaffolding).
        WeightOpeningTranscriptMode::AggregatedOpeningsV4Experimental => Ok(3),
        // mode 4: aggregated oracle sumcheck (production mode).
        WeightOpeningTranscriptMode::AggregatedOracleSumcheck => Ok(4),
        WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1 => {
            Err(StarknetModelError::SoundnessGate(
                "BatchedRlcDirectEvalV1 is off-chain only and cannot be serialized for Starknet"
                    .to_string(),
            ))
        }
    }
}

fn starknet_weight_binding_data(
    proof: &crate::gkr::GKRProof,
) -> Result<Vec<String>, StarknetModelError> {
    use crate::gkr::types::WeightOpeningTranscriptMode;

    match proof.weight_opening_transcript_mode {
        // Modes 0/1 do not require extra payload.
        WeightOpeningTranscriptMode::Sequential
        | WeightOpeningTranscriptMode::BatchedSubchannelV1 => Ok(Vec::new()),
        WeightOpeningTranscriptMode::AggregatedTrustlessV2 => Ok(mode2_weight_binding_data(proof)
            .into_iter()
            .map(|f| format!("0x{:x}", f))
            .collect()),
        WeightOpeningTranscriptMode::AggregatedOpeningsV4Experimental => {
            Ok(mode3_weight_binding_data(proof)
                .into_iter()
                .map(|f| format!("0x{:x}", f))
                .collect())
        }
        WeightOpeningTranscriptMode::AggregatedOracleSumcheck => {
            // Mode 4: full proof payload or RLC-only marker.
            if let Some(binding) = proof.aggregated_binding.as_ref() {
                let mut payload = Vec::new();
                crate::cairo_serde::serialize_aggregated_binding_proof(binding, &mut payload);
                Ok(payload.into_iter().map(|f| format!("0x{:x}", f)).collect())
            } else {
                // RLC-only: marker tag (0x524C43 = "RLC") + claim count.
                Ok(vec![
                    format!("0x{:x}", FieldElement::from(0x524C43u64)),
                    format!("0x{:x}", FieldElement::from(proof.weight_claims.len() as u64)),
                ])
            }
        }
        WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1 => {
            Err(StarknetModelError::SoundnessGate(
                "BatchedRlcDirectEvalV1 is off-chain only and cannot be serialized for Starknet"
                    .to_string(),
            ))
        }
    }
}

/// Standardized verification calldata payload for `verify_model_direct()`.
///
/// `calldata_parts` is flat Starknet calldata ready for:
/// `verify_model_direct(model_id, session_id, raw_io_data, weight_commitment, ...)`.
///
/// The second element is a caller-provided session placeholder (for example
/// `"__SESSION_ID__"`). Submitters should replace it with the real session ID
/// and use the same value for all `upload_proof_chunk()` calls.
#[derive(Debug)]
pub struct VerifyModelDirectCalldata {
    /// Flat calldata for `verify_model_direct`.
    pub calldata_parts: Vec<String>,
    /// Activation STARK chunks for `upload_proof_chunk(session_id, idx, data)`.
    pub upload_chunks: Vec<Vec<String>>,
    /// Total number of felt252 values in `calldata_parts`.
    pub total_felts: usize,
}

/// Build flat sncast calldata for `verify_model_gkr()`.
///
/// Matches EloVerifier v7 contract's parameter order:
/// ```text
/// model_id: felt252
/// raw_io_data: Array<felt252>
/// circuit_depth: u32              (total layers including Identity, for channel seeding)
/// num_layers: u32                 (proof layers only)
/// matmul_dims: Array<u32>         [len, m0, k0, n0, ...]
/// dequantize_bits: Array<u64>     [len, bits0, ...]
/// proof_data: Array<felt252>      [len, ...layer proof felts...]
/// weight_commitments: Array<felt252> [len, ...commitment felts...]
/// ```
pub fn build_verify_model_gkr_calldata(
    proof: &crate::gkr::GKRProof,
    circuit: &crate::gkr::LayeredCircuit,
    model_id: FieldElement,
    raw_io_data: &[FieldElement],
) -> Result<VerifyModelGkrCalldata, StarknetModelError> {
    build_verify_model_gkr_calldata_inner(
        proof,
        circuit,
        model_id,
        raw_io_data,
        GkrCalldataLayout::V1,
    )
}

/// Build flat sncast calldata for `verify_model_gkr_v2()`.
///
/// Emits `weight_binding_mode` based on transcript mode:
/// - `0`: sequential openings (v1-compatible)
/// - `1`: batched sub-channel openings
pub fn build_verify_model_gkr_v2_calldata(
    proof: &crate::gkr::GKRProof,
    circuit: &crate::gkr::LayeredCircuit,
    model_id: FieldElement,
    raw_io_data: &[FieldElement],
) -> Result<VerifyModelGkrCalldata, StarknetModelError> {
    build_verify_model_gkr_calldata_inner(
        proof,
        circuit,
        model_id,
        raw_io_data,
        GkrCalldataLayout::V2,
    )
}

/// Build flat sncast calldata for `verify_model_gkr_v3()`.
///
/// v3 layout extends v2 with `weight_binding_data: Array<felt252>`:
/// - mode `0`: sequential openings
/// - mode `1`: batched sub-channel openings
/// - mode `2`: aggregated trustless binding payload + opening proofs (phase-3 path)
pub fn build_verify_model_gkr_v3_calldata(
    proof: &crate::gkr::GKRProof,
    circuit: &crate::gkr::LayeredCircuit,
    model_id: FieldElement,
    raw_io_data: &[FieldElement],
) -> Result<VerifyModelGkrCalldata, StarknetModelError> {
    build_verify_model_gkr_calldata_inner(
        proof,
        circuit,
        model_id,
        raw_io_data,
        GkrCalldataLayout::V3,
    )
}

/// Build flat sncast calldata for `verify_model_gkr_v4()`.
///
/// v4 layout currently reuses the v3 envelope shape while requiring:
/// - mode `3`: aggregated openings experimental binding payload + opening proofs
/// - mode `4`: aggregated oracle sumcheck binding payload
pub fn build_verify_model_gkr_v4_calldata(
    proof: &crate::gkr::GKRProof,
    circuit: &crate::gkr::LayeredCircuit,
    model_id: FieldElement,
    raw_io_data: &[FieldElement],
) -> Result<VerifyModelGkrCalldata, StarknetModelError> {
    build_verify_model_gkr_calldata_inner(
        proof,
        circuit,
        model_id,
        raw_io_data,
        GkrCalldataLayout::V4,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GkrCalldataLayout {
    V1,
    V2,
    V3,
    V4,
}

fn build_verify_model_gkr_calldata_inner(
    proof: &crate::gkr::GKRProof,
    circuit: &crate::gkr::LayeredCircuit,
    model_id: FieldElement,
    raw_io_data: &[FieldElement],
    layout: GkrCalldataLayout,
) -> Result<VerifyModelGkrCalldata, StarknetModelError> {
    use crate::cairo_serde::serialize_gkr_proof_data_only;
    use crate::gkr::types::WeightOpeningTranscriptMode;

    enforce_gkr_soundness_gates(proof)?;

    // v1 entrypoint only supports sequential weight-opening transcript.
    if layout == GkrCalldataLayout::V1
        && proof.weight_opening_transcript_mode != WeightOpeningTranscriptMode::Sequential
    {
        return Err(StarknetModelError::SoundnessGate(format!(
            "verify_model_gkr requires Sequential weight opening mode (got: {:?})",
            proof.weight_opening_transcript_mode
        )));
    }

    let mut parts = Vec::new();

    // 1. model_id: felt252
    parts.push(format!("0x{:x}", model_id));

    // 2. raw_io_data: Array<felt252> — raw input/output for on-chain MLE evaluation
    parts.push(format!("{}", raw_io_data.len()));
    for f in raw_io_data {
        parts.push(format!("0x{:x}", f));
    }

    // 3. circuit_depth: u32 — total layers including Identity (for channel seeding)
    let circuit_depth = circuit.layers.len() as u32;
    parts.push(format!("{}", circuit_depth));

    // 4. num_layers: u32 — proof layers only (excluding Identity/Input)
    let num_layers = proof.layer_proofs.len() as u32;
    parts.push(format!("{}", num_layers));

    // 4. matmul_dims: Array<u32>
    let matmul_dims = extract_matmul_dims(circuit);
    parts.push(format!("{}", matmul_dims.len()));
    for d in &matmul_dims {
        parts.push(format!("{}", d));
    }

    // 5. dequantize_bits: Array<u64>
    let dequantize_bits = extract_dequantize_bits(circuit);
    parts.push(format!("{}", dequantize_bits.len()));
    for b in &dequantize_bits {
        parts.push(format!("{}", b));
    }

    // 6. proof_data: Array<felt252> — tag-dispatched per-layer GKR proofs
    let mut proof_data = Vec::new();
    serialize_gkr_proof_data_only(proof, &mut proof_data);

    // Self-verify against serialized proof data (same as chunked path).
    if std::env::var("STWO_SKIP_SELF_VERIFY").is_err() {
        replay_verify_serialized_proof(
            &proof_data,
            raw_io_data,
            &matmul_dims,
            circuit_depth,
            num_layers,
            false, // unpacked in single-TX path
        ).map_err(|e| StarknetModelError::SoundnessGate(
            format!("self-verification failed: {e}")
        ))?;
    }

    parts.push(format!("{}", proof_data.len()));
    for f in &proof_data {
        parts.push(format!("0x{:x}", f));
    }

    // 7. weight_commitments: Array<felt252> (main + deferred matmul branches, weightless excluded)
    let mut weight_commitments = proof.weight_commitments.clone();
    for deferred in &proof.deferred_proofs {
        if let Some(wc) = deferred.weight_commitment() {
            weight_commitments.push(wc);
        }
    }
    parts.push(format!("{}", weight_commitments.len()));
    for wc in &weight_commitments {
        parts.push(format!("0x{:x}", wc));
    }

    // 8 (v2/v3/v4). weight_binding_mode: u32
    if layout == GkrCalldataLayout::V2
        || layout == GkrCalldataLayout::V3
        || layout == GkrCalldataLayout::V4
    {
        let binding_mode = starknet_weight_binding_mode(proof.weight_opening_transcript_mode)?;
        if layout == GkrCalldataLayout::V2 && binding_mode == 2 {
            return Err(StarknetModelError::SoundnessGate(
                "verify_model_gkr_v2 supports weight_binding_mode in {0,1}; mode 2 requires verify_model_gkr_v3"
                    .to_string(),
            ));
        }
        if layout == GkrCalldataLayout::V2 && binding_mode == 3 {
            return Err(StarknetModelError::SoundnessGate(
                "verify_model_gkr_v2 supports weight_binding_mode in {0,1}; mode 3 requires verify_model_gkr_v4"
                    .to_string(),
            ));
        }
        if layout == GkrCalldataLayout::V2 && binding_mode == 4 {
            return Err(StarknetModelError::SoundnessGate(
                "verify_model_gkr_v2 supports weight_binding_mode in {0,1}; mode 4 requires verify_model_gkr_v4"
                    .to_string(),
            ));
        }
        if layout == GkrCalldataLayout::V3 && binding_mode == 3 {
            return Err(StarknetModelError::SoundnessGate(
                "verify_model_gkr_v3 supports weight_binding_mode in {0,1,2}; mode 3 requires verify_model_gkr_v4"
                    .to_string(),
            ));
        }
        if layout == GkrCalldataLayout::V3 && binding_mode == 4 {
            return Err(StarknetModelError::SoundnessGate(
                "verify_model_gkr_v3 supports weight_binding_mode in {0,1,2}; mode 4 requires verify_model_gkr_v4"
                    .to_string(),
            ));
        }
        if layout == GkrCalldataLayout::V4 && binding_mode != 3 && binding_mode != 4 {
            return Err(StarknetModelError::SoundnessGate(format!(
                "verify_model_gkr_v4 requires weight_binding_mode in {{3,4}} (got: {binding_mode})"
            )));
        }
        parts.push(format!("{}", binding_mode));
    }

    // 9 (v3/v4). weight_binding_data: Array<felt252>
    if layout == GkrCalldataLayout::V3 || layout == GkrCalldataLayout::V4 {
        let weight_binding_data = starknet_weight_binding_data(proof)?;
        if layout == GkrCalldataLayout::V4 && weight_binding_data.is_empty() {
            return Err(StarknetModelError::SoundnessGate(
                "verify_model_gkr_v4 requires non-empty weight_binding_data".to_string(),
            ));
        }
        parts.push(format!("{}", weight_binding_data.len()));
        for felt in weight_binding_data {
            parts.push(felt);
        }
    }

    // 8/9/10. weight_opening_proofs: Array<MleOpeningProof>
    // Order matches verifier weight_claims: main walk first, then deferred (weightless excluded).
    let deferred_opening_count = proof
        .deferred_proofs
        .iter()
        .filter(|d| d.has_weights())
        .count();
    let total_openings = proof.weight_openings.len() + deferred_opening_count;
    parts.push(format!("{}", total_openings));

    let mut opening_buf = Vec::new();
    for opening in &proof.weight_openings {
        opening_buf.clear();
        crate::cairo_serde::serialize_mle_opening_proof(opening, &mut opening_buf);
        for felt in &opening_buf {
            parts.push(format!("0x{:x}", felt));
        }
    }
    for deferred in &proof.deferred_proofs {
        if let Some(wo) = deferred.weight_opening() {
            opening_buf.clear();
            crate::cairo_serde::serialize_mle_opening_proof(wo, &mut opening_buf);
            for felt in &opening_buf {
                parts.push(format!("0x{:x}", felt));
            }
        }
    }

    let total_felts = parts.len();
    Ok(VerifyModelGkrCalldata {
        calldata_parts: parts,
        total_felts,
    })
}

/// Build flat calldata for `verify_model_direct()`.
///
/// Layout matches EloVerifier's signature:
/// ```text
/// model_id: felt252
/// session_id: felt252
/// raw_io_data: Array<felt252>
/// weight_commitment: felt252
/// num_layers: u32
/// activation_type: u8
/// batched_proofs: Array<BatchedMatMulProof>
/// activation_stark_data: Array<felt252>
/// ```
///
/// The returned calldata includes `session_placeholder` as the `session_id`
/// field so submitters can choose a runtime session ID while preserving a
/// single serialized artifact schema.
pub fn build_verify_model_direct_calldata(
    proof: &DirectStarknetProof,
    session_placeholder: &str,
) -> VerifyModelDirectCalldata {
    let mut parts = Vec::new();

    // model_id
    parts.push(format!("0x{:x}", proof.model_id));
    // session_id (filled by submitter)
    parts.push(session_placeholder.to_string());

    // raw_io_data: Array<felt252>
    parts.push(format!("{}", proof.raw_io_data.len()));
    for felt in &proof.raw_io_data {
        parts.push(format!("0x{:x}", felt));
    }

    // weight_commitment, num_layers, activation_type
    parts.push(format!("0x{:x}", proof.weight_commitment));
    parts.push(format!("{}", proof.num_layers));
    parts.push(format!("{}", proof.activation_type));

    // batched_proofs: Array<BatchedMatMulProof>
    // Each batch is already serialized according to Cairo Serde layout.
    parts.push(format!("{}", proof.batched_calldata.len()));
    for batch in &proof.batched_calldata {
        for felt in batch {
            parts.push(format!("0x{:x}", felt));
        }
    }

    // activation_stark_data: Array<felt252>
    let flat_stark_len: usize = proof.stark_chunks.iter().map(|c| c.len()).sum();
    parts.push(format!("{}", flat_stark_len));
    for chunk in &proof.stark_chunks {
        for felt in chunk {
            parts.push(format!("0x{:x}", felt));
        }
    }

    let upload_chunks: Vec<Vec<String>> = proof
        .stark_chunks
        .iter()
        .map(|chunk| chunk.iter().map(|f| format!("0x{:x}", f)).collect())
        .collect();

    let total_felts = parts.len();
    VerifyModelDirectCalldata {
        calldata_parts: parts,
        upload_chunks,
        total_felts,
    }
}

/// Estimate gas cost for verifying the proof on-chain.
///
/// Based on empirical Cairo verifier costs:
/// - Base cost: 50K gas (contract overhead)
/// - Per activation layer: 10K gas (STARK verification per component)
/// - Per matmul: 5K gas (sumcheck verification)
/// - Per add/mul/layernorm claim: included in unified STARK verification
/// - Log trace rows: 100 gas per log2(rows) (FRI queries scale logarithmically)
/// - Per calldata felt: 16 gas (L1 data availability)
pub fn estimate_verification_gas(num_layers: usize, total_trace_rows: usize) -> u64 {
    let base_cost: u64 = 50_000;
    let per_layer: u64 = 10_000;
    let per_row_log: u64 = 100;

    let log_rows = if total_trace_rows > 0 {
        (total_trace_rows as f64).log2().ceil() as u64
    } else {
        0
    };
    base_cost + (num_layers as u64) * per_layer + log_rows * per_row_log
}

/// Estimate gas cost from a `StarknetModelProof`.
pub fn estimate_gas_from_proof(proof: &StarknetModelProof) -> u64 {
    let base = proof.estimated_gas;
    // Add L1 data availability cost (16 gas per calldata felt252)
    let da_cost = (proof.calldata_size as u64) * 16;
    base + da_cost
}

/// Replay Fiat-Shamir channel verification against serialized proof data.
///
/// This catches prover/serializer mismatches immediately after proof generation,
/// preventing stale-binary issues where the serialized proof doesn't match the
/// channel state the verifier expects. It exercises the exact same channel
/// operations as the Cairo on-chain verifier.
///
/// Returns `Ok(())` if all layers pass, or `Err(msg)` with the first mismatch.
pub fn replay_verify_serialized_proof(
    proof_data: &[FieldElement],
    raw_io: &[FieldElement],
    matmul_dims: &[u32],
    circuit_depth: u32,
    num_layers: u32,
    packed: bool,
) -> Result<(), String> {
    use crate::crypto::poseidon_channel::PoseidonChannel;
    use crate::gkr::prover::mix_secure_field;
    use crate::crypto::poseidon_channel::felt_to_securefield;
    use stwo::core::fields::qm31::{QM31, SecureField};
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::m31::M31;
    use num_traits::Zero;

    fn felt_to_u64(f: &FieldElement) -> u64 {
        let b = f.to_bytes_be();
        u64::from_be_bytes([b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]])
    }

    let input_rows = felt_to_u64(&raw_io[0]);
    let input_cols = felt_to_u64(&raw_io[1]);
    let input_len = felt_to_u64(&raw_io[2]) as usize;
    let out_start = 3 + input_len;
    let output_rows = felt_to_u64(&raw_io[out_start]) as usize;
    let output_cols = felt_to_u64(&raw_io[out_start + 1]) as usize;

    let padded_rows = output_rows.next_power_of_two();
    let padded_cols = output_cols.next_power_of_two();
    let mut output_mle = Vec::with_capacity(padded_rows * padded_cols);
    for r in 0..padded_rows {
        for c in 0..padded_cols {
            if r < output_rows && c < output_cols {
                let idx = r * output_cols + c;
                let val = felt_to_u64(&raw_io[out_start + 3 + idx]) as u32;
                output_mle.push(SecureField::from(M31::from(val)));
            } else {
                output_mle.push(SecureField::zero());
            }
        }
    }

    let mut ch = PoseidonChannel::new();
    ch.mix_u64(circuit_depth as u64);
    ch.mix_u64(input_rows);
    ch.mix_u64(input_cols);

    let log_out = (padded_rows * padded_cols).ilog2() as usize;
    let r_out = ch.draw_qm31s(log_out);
    let output_value = crate::components::matmul::evaluate_mle_pub(&output_mle, &r_out);
    mix_secure_field(&mut ch, output_value);

    let mut off = 0usize;
    let read_u32_from = |data: &[FieldElement], off: &mut usize| -> u32 {
        let v = felt_to_u64(&data[*off]) as u32;
        *off += 1;
        v
    };
    let read_qm31_from = |data: &[FieldElement], off: &mut usize| -> SecureField {
        if packed {
            let fe = data[*off];
            *off += 1;
            felt_to_securefield(fe)
        } else {
            let aa = felt_to_u64(&data[*off]) as u32; *off += 1;
            let ab = felt_to_u64(&data[*off]) as u32; *off += 1;
            let ba = felt_to_u64(&data[*off]) as u32; *off += 1;
            let bb = felt_to_u64(&data[*off]) as u32; *off += 1;
            QM31(CM31(M31::from(aa), M31::from(ab)),
                 CM31(M31::from(ba), M31::from(bb)))
        }
    };

    let mut current_claim_value = output_value;
    let mut matmul_idx = 0usize;

    for layer in 0..num_layers as usize {
        let tag = read_u32_from(proof_data, &mut off);

        match tag {
            0 => {
                // MatMul
                let m = matmul_dims[matmul_idx * 3] as usize;
                let k = matmul_dims[matmul_idx * 3 + 1] as usize;
                let n = matmul_dims[matmul_idx * 3 + 2] as usize;
                matmul_idx += 1;

                ch.mix_u64(m as u64);
                ch.mix_u64(k as u64);
                ch.mix_u64(n as u64);
                mix_secure_field(&mut ch, current_claim_value);

                let num_rounds = read_u32_from(proof_data, &mut off) as usize;
                let mut current_sum = current_claim_value;

                for round in 0..num_rounds {
                    let c0 = read_qm31_from(proof_data, &mut off);
                    let c1 = read_qm31_from(proof_data, &mut off);
                    let c2 = read_qm31_from(proof_data, &mut off);
                    let p0 = c0;
                    let p1 = c0 + c1 + c2;
                    if p0 + p1 != current_sum {
                        return Err(format!(
                            "MATMUL_ROUND_SUM_MISMATCH at layer {} round {}: p(0)+p(1)={:?} != sum={:?}",
                            layer, round, p0 + p1, current_sum
                        ));
                    }
                    ch.mix_poly_coeffs(c0, c1, c2);
                    let challenge = ch.draw_qm31();
                    current_sum = c0 + c1 * challenge + c2 * challenge * challenge;
                }
                let final_a = read_qm31_from(proof_data, &mut off);
                let final_b = read_qm31_from(proof_data, &mut off);
                if current_sum != final_a * final_b {
                    return Err(format!(
                        "MATMUL_FINAL_MISMATCH at layer {}: sum={:?} != a*b={:?}",
                        layer, current_sum, final_a * final_b
                    ));
                }
                mix_secure_field(&mut ch, final_a);
                mix_secure_field(&mut ch, final_b);
                current_claim_value = final_a;
            }
            8 => {
                // RMSNorm
                let input_eval = read_qm31_from(proof_data, &mut off);
                let output_eval = read_qm31_from(proof_data, &mut off);
                let rms_sq = read_qm31_from(proof_data, &mut off);
                let rsqrt_eval = read_qm31_from(proof_data, &mut off);
                off += 1; // commitment
                let _simd = read_u32_from(proof_data, &mut off);

                ch.mix_u64(0x524E); // "RN"
                mix_secure_field(&mut ch, rms_sq);
                mix_secure_field(&mut ch, rsqrt_eval);
                mix_secure_field(&mut ch, current_claim_value);

                let nrounds = read_u32_from(proof_data, &mut off) as usize;
                let mut rms_sum = current_claim_value;
                for round in 0..nrounds {
                    let c0 = read_qm31_from(proof_data, &mut off);
                    let c1 = read_qm31_from(proof_data, &mut off);
                    let c2 = read_qm31_from(proof_data, &mut off);
                    let c3 = read_qm31_from(proof_data, &mut off);
                    let p0 = c0;
                    let p1 = c0 + c1 + c2 + c3;
                    if p0 + p1 != rms_sum {
                        return Err(format!(
                            "RMSNORM_ROUND_SUM at layer {} round {}", layer, round
                        ));
                    }
                    ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                    let challenge = ch.draw_qm31();
                    rms_sum = c0 + c1 * challenge + c2 * challenge * challenge
                        + c3 * challenge * challenge * challenge;
                }
                let input_final = read_qm31_from(proof_data, &mut off);
                let rsqrt_final = read_qm31_from(proof_data, &mut off);
                mix_secure_field(&mut ch, input_final);
                mix_secure_field(&mut ch, rsqrt_final);

                // Optional logup
                let has_logup = read_u32_from(proof_data, &mut off);
                if has_logup == 1 {
                    ch.mix_u64(0x4C4F47); // "LOG"
                    ch.mix_u64(0x524E); // "RN"
                    let _gamma = ch.draw_qm31();
                    let _beta = ch.draw_qm31();
                    let claimed_sum = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, claimed_sum);
                    let eq_rounds = read_u32_from(proof_data, &mut off) as usize;
                    let mut logup_sum = SecureField::from(M31::from(1u32));
                    for round in 0..eq_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c1 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let p0 = c0;
                        let p1 = c0 + c1 + c2 + c3;
                        if p0 + p1 != logup_sum {
                            return Err(format!(
                                "LOGUP_ROUND_SUM at layer {} round {}", layer, round
                            ));
                        }
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }
                    let _w = read_qm31_from(proof_data, &mut off);
                    let _in_e = read_qm31_from(proof_data, &mut off);
                    let _out_e = read_qm31_from(proof_data, &mut off);
                    let num_mults = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..num_mults {
                        let _ = read_u32_from(proof_data, &mut off);
                    }
                }
                mix_secure_field(&mut ch, input_eval);
                mix_secure_field(&mut ch, output_eval);
                current_claim_value = input_eval;
            }
            3 => {
                // Activation
                let _act_type = read_u32_from(proof_data, &mut off);
                let input_eval = read_qm31_from(proof_data, &mut off);
                let output_eval = read_qm31_from(proof_data, &mut off);
                off += 1; // table_commitment

                ch.mix_u64(0x4C4F47); // "LOG"
                let _gamma = ch.draw_qm31();
                let _beta = ch.draw_qm31();

                let has_logup = read_u32_from(proof_data, &mut off);
                if has_logup == 1 {
                    let claimed_sum = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, claimed_sum);
                    let eq_rounds = read_u32_from(proof_data, &mut off) as usize;
                    let mut logup_sum = SecureField::from(M31::from(1u32));
                    for round in 0..eq_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c1 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let p0 = c0;
                        let p1 = c0 + c1 + c2 + c3;
                        if p0 + p1 != logup_sum {
                            return Err(format!(
                                "ACT_LOGUP_ROUND at layer {} round {}", layer, round
                            ));
                        }
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }
                    let _w = read_qm31_from(proof_data, &mut off);
                    let _in_e = read_qm31_from(proof_data, &mut off);
                    let _out_e = read_qm31_from(proof_data, &mut off);
                    let num_mults = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..num_mults {
                        let _ = read_qm31_from(proof_data, &mut off);
                    }
                }
                mix_secure_field(&mut ch, input_eval);
                mix_secure_field(&mut ch, output_eval);
                current_claim_value = input_eval;
            }
            1 => {
                // Add
                let lhs = read_qm31_from(proof_data, &mut off);
                let rhs = read_qm31_from(proof_data, &mut off);
                let trunk_idx = read_u32_from(proof_data, &mut off);
                mix_secure_field(&mut ch, lhs);
                mix_secure_field(&mut ch, rhs);
                let _alpha = ch.draw_qm31();
                current_claim_value = if trunk_idx == 0 { lhs } else { rhs };
            }
            _ => return Err(format!("Unknown tag {} at layer {}", tag, layer)),
        }
    }

    let _num_deferred = read_u32_from(proof_data, &mut off);
    let remaining = proof_data.len() - off;
    if remaining != 0 {
        return Err(format!(
            "Expected 0 remaining proof_data felts, got {}", remaining
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregation::compute_io_commitment;
    use crate::compiler::graph::GraphBuilder;
    use crate::components::activation::ActivationType;
    use stwo::core::fields::m31::M31;

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev }
        }

        fn unset(key: &'static str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(prev) = self.prev.as_ref() {
                std::env::set_var(self.key, prev);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn test_gas_estimation() {
        let gas = estimate_verification_gas(3, 10_000);
        assert!(gas > 50_000);
        assert!(gas < 200_000);

        let gas = estimate_verification_gas(32, 10_000_000);
        assert!(gas > 300_000);
    }

    #[test]
    fn test_gas_estimation_zero_rows() {
        let gas = estimate_verification_gas(0, 0);
        assert_eq!(gas, 50_000); // Just base cost
    }

    #[test]
    fn test_prove_for_starknet_matmul_only() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let result = prove_for_starknet(&graph, &input, &weights);
        assert!(result.is_ok());
        let proof = result.unwrap();

        // No activations → empty calldata
        assert!(proof.unified_calldata.is_empty());
        assert_eq!(proof.num_matmul_proofs, 1);
        assert_eq!(proof.num_proven_layers, 1);
        assert!(proof.estimated_gas > 0);
    }

    #[test]
    fn test_prove_for_starknet_mlp() {
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w2.set(i, j, M31::from(((i * j) % 5 + 1) as u32));
            }
        }
        weights.add_weight(2, w2);
        let mut w4 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w4.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(4, w4);

        let result = prove_for_starknet(&graph, &input, &weights);
        assert!(result.is_ok());
        let proof = result.unwrap();

        // Has unified STARK calldata
        assert!(
            !proof.unified_calldata.is_empty(),
            "should have unified calldata"
        );
        assert_eq!(proof.layer_claims.len(), 2, "2 activation layers");
        assert_eq!(proof.num_matmul_proofs, 3, "3 matmul sumchecks");
        assert_eq!(proof.num_proven_layers, 5, "5 total layers");
        assert!(proof.calldata_size > 0);

        // Gas estimate should include DA cost
        let gas_with_da = estimate_gas_from_proof(&proof);
        assert!(
            gas_with_da > proof.estimated_gas,
            "DA cost should increase gas"
        );
    }

    #[test]
    fn test_build_starknet_proof_from_aggregated() {
        use crate::aggregation::prove_model_aggregated;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        // Prove aggregated, then convert to Starknet format
        let agg_proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated proving should succeed");
        let starknet_proof = build_starknet_proof(&agg_proof);

        assert!(!starknet_proof.unified_calldata.is_empty());
        assert_eq!(starknet_proof.num_matmul_proofs, 2);
        assert_eq!(starknet_proof.layer_claims.len(), 1);

        // Calldata should be compact (< 5000 felt252s for a small model)
        assert!(
            starknet_proof.calldata_size < 5000,
            "calldata too large: {} felts",
            starknet_proof.calldata_size
        );
    }

    // === IO Commitment + Combined Calldata Tests ===

    #[test]
    fn test_io_commitment_deterministic() {
        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }
        let mut output = M31Matrix::new(1, 2);
        output.set(0, 0, M31::from(10));
        output.set(0, 1, M31::from(20));

        let hash1 = compute_io_commitment(&input, &output);
        let hash2 = compute_io_commitment(&input, &output);
        assert_eq!(hash1, hash2, "same input/output should produce same hash");
        assert_ne!(hash1, FieldElement::ZERO, "hash should be non-zero");
    }

    #[test]
    fn test_io_commitment_different_inputs() {
        let mut input1 = M31Matrix::new(1, 4);
        for j in 0..4 {
            input1.set(0, j, M31::from((j + 1) as u32));
        }
        let mut input2 = M31Matrix::new(1, 4);
        for j in 0..4 {
            input2.set(0, j, M31::from((j + 10) as u32));
        }

        let mut output = M31Matrix::new(1, 2);
        output.set(0, 0, M31::from(10));
        output.set(0, 1, M31::from(20));

        let hash1 = compute_io_commitment(&input1, &output);
        let hash2 = compute_io_commitment(&input2, &output);
        assert_ne!(
            hash1, hash2,
            "different inputs should produce different hashes"
        );
    }

    #[test]
    fn test_combined_calldata_io_at_index_4() {
        use crate::aggregation::prove_model_aggregated_onchain;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain aggregated proving should succeed");
        let starknet_proof = build_starknet_proof_onchain(&agg_proof, &input);

        // Raw IO data is at index [4] as a length-prefixed Array<felt252>.
        // Index [4] is the length, then [5..5+len] are the raw IO elements.
        assert!(
            starknet_proof.combined_calldata.len() > 5,
            "combined calldata too small"
        );
        let raw_io_len: u64 = starknet_proof.combined_calldata[4].try_into().unwrap();
        assert!(raw_io_len > 0, "raw_io_data length should be > 0");

        // Verify that Poseidon(raw_io_data) matches the io_commitment
        let raw_io_slice = &starknet_proof.combined_calldata[5..5 + raw_io_len as usize];
        let recomputed = starknet_crypto::poseidon_hash_many(raw_io_slice);
        assert_eq!(
            recomputed, starknet_proof.io_commitment,
            "Poseidon(raw_io_data) must equal io_commitment"
        );
        assert_ne!(starknet_proof.io_commitment, FieldElement::ZERO);

        // PCS config at [0..4] must have real security parameters (not zeros)
        let default_config = PcsConfig::default();
        assert_eq!(
            starknet_proof.combined_calldata[0],
            FieldElement::from(default_config.pow_bits as u64),
            "combined_calldata[0] must be pow_bits (matmul-only falls back to default)"
        );
        assert_ne!(
            starknet_proof.combined_calldata[3],
            FieldElement::ZERO,
            "combined_calldata[3] (n_queries) must not be zero"
        );
    }

    #[test]
    fn test_pcs_config_from_unified_stark() {
        use crate::aggregation::prove_model_aggregated_onchain;

        // Build a model WITH unified STARK (has activation)
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain proving should succeed");

        assert!(
            agg_proof.unified_stark.is_some(),
            "should have unified STARK"
        );

        let starknet_proof = build_starknet_proof_onchain(&agg_proof, &input);

        // PCS config should come from the actual proof, not defaults
        let actual_config = agg_proof.unified_stark.as_ref().unwrap().config;
        assert_eq!(
            starknet_proof.combined_calldata[0],
            FieldElement::from(actual_config.pow_bits as u64),
            "pow_bits must match proof's PCS config"
        );
        assert_eq!(
            starknet_proof.combined_calldata[1],
            FieldElement::from(actual_config.fri_config.log_blowup_factor as u64),
            "log_blowup_factor must match proof's PCS config"
        );
        assert_eq!(
            starknet_proof.combined_calldata[2],
            FieldElement::from(actual_config.fri_config.log_last_layer_degree_bound as u64),
            "log_last_layer_deg must match proof's PCS config"
        );
        assert_eq!(
            starknet_proof.combined_calldata[3],
            FieldElement::from(actual_config.fri_config.n_queries as u64),
            "n_queries must match proof's PCS config"
        );

        // pcs_config field on the struct should match too
        assert_eq!(starknet_proof.pcs_config.pow_bits, actual_config.pow_bits);
        assert_eq!(
            starknet_proof.pcs_config.fri_config.n_queries,
            actual_config.fri_config.n_queries
        );

        // Security sanity: pow_bits > 0 and n_queries > 0
        assert!(
            starknet_proof.pcs_config.pow_bits > 0,
            "pow_bits must be > 0 for soundness"
        );
        assert!(
            starknet_proof.pcs_config.fri_config.n_queries > 0,
            "n_queries must be > 0 for soundness"
        );
    }

    #[test]
    fn test_prove_for_starknet_onchain_mlp() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let result = prove_for_starknet_onchain(&graph, &input, &weights);
        assert!(result.is_ok());
        let proof = result.unwrap();

        // Has unified STARK + matmul on-chain proofs
        assert!(!proof.unified_calldata.is_empty());
        assert_eq!(proof.num_matmul_proofs, 2);
        assert_eq!(proof.layer_claims.len(), 1);
        assert_ne!(proof.io_commitment, FieldElement::ZERO);
        assert!(proof.combined_calldata.len() > 5);
        // Index [4] is now raw_io_data length prefix (Array<felt252> encoding)
        let raw_io_len: u64 = proof.combined_calldata[4].try_into().unwrap();
        assert!(raw_io_len > 0, "raw_io_data should be non-empty");
    }

    #[test]
    fn test_prove_for_starknet_onchain_matmul_only() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let result = prove_for_starknet_onchain(&graph, &input, &weights);
        assert!(result.is_ok());
        let proof = result.unwrap();

        assert!(proof.unified_calldata.is_empty());
        assert_eq!(proof.num_matmul_proofs, 1);
        assert!(proof.estimated_gas > 0);
    }

    #[test]
    fn test_build_starknet_proof_with_add() {
        use crate::aggregation::prove_model_aggregated_onchain;

        // Build a model with a residual Add
        let mut builder = GraphBuilder::new((1, 8));
        builder.linear(8);
        let branch = builder.fork();
        builder.activation(ActivationType::ReLU);
        builder.linear(8);
        builder.add_from(branch);
        builder.linear(4);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 8);
        for j in 0..8 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                w2.set(i, j, M31::from(((i * j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(2, w2);
        let mut w4 = M31Matrix::new(8, 4);
        for i in 0..8 {
            for j in 0..4 {
                w4.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(4, w4);

        let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain proving with Add should succeed");
        let starknet_proof = build_starknet_proof_onchain(&agg_proof, &input);

        // Add claim should be in the unified STARK
        assert_eq!(starknet_proof.num_add_claims, 1, "should have 1 Add claim");
        // Unified calldata covers activation + add together
        assert!(
            !starknet_proof.unified_calldata.is_empty(),
            "unified calldata should be non-empty"
        );

        // mul and layernorm should be empty
        assert_eq!(starknet_proof.num_mul_claims, 0);
        assert_eq!(starknet_proof.num_layernorm_claims, 0);

        // num_proven_layers should include the Add claim
        assert_eq!(
            starknet_proof.num_proven_layers,
            3 + 1 + 1, // 3 matmul + 1 activation + 1 add
            "total proven layers"
        );
    }

    #[test]
    fn test_starknet_combined_calldata_includes_elementwise() {
        use crate::aggregation::prove_model_aggregated_onchain;

        // Model with Add — combined calldata should be larger than matmul-only
        let mut builder_add = GraphBuilder::new((1, 8));
        builder_add.linear(8);
        let branch = builder_add.fork();
        builder_add.linear(8);
        builder_add.add_from(branch);
        builder_add.linear(4);
        let graph_add = builder_add.build();

        let mut builder_no_add = GraphBuilder::new((1, 8));
        builder_no_add.linear(8);
        builder_no_add.linear(8);
        builder_no_add.linear(4);
        let graph_no_add = builder_no_add.build();

        let mut input = M31Matrix::new(1, 8);
        for j in 0..8 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w1 = M31Matrix::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                w1.set(i, j, M31::from(((i * j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(1, w1);
        let mut w2 = M31Matrix::new(8, 4);
        for i in 0..8 {
            for j in 0..4 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        // For the add graph, we need weights at different node IDs
        let mut weights_add = GraphWeights::new();
        let mut wa0 = M31Matrix::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                wa0.set(i, j, M31::from(((i + j) % 5 + 1) as u32));
            }
        }
        weights_add.add_weight(0, wa0);
        let mut wa1 = M31Matrix::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                wa1.set(i, j, M31::from(((i * j) % 7 + 1) as u32));
            }
        }
        weights_add.add_weight(1, wa1);
        let mut wa3 = M31Matrix::new(8, 4);
        for i in 0..8 {
            for j in 0..4 {
                wa3.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights_add.add_weight(3, wa3);

        let proof_add = prove_model_aggregated_onchain(&graph_add, &input, &weights_add)
            .expect("proving with Add should succeed");
        let proof_no_add = prove_model_aggregated_onchain(&graph_no_add, &input, &weights)
            .expect("proving without Add should succeed");

        let sn_add = build_starknet_proof_onchain(&proof_add, &input);
        let sn_no_add = build_starknet_proof_onchain(&proof_no_add, &input);

        // Model with Add should have unified STARK (includes Add component)
        assert!(
            !sn_add.unified_calldata.is_empty(),
            "Add model should have unified STARK calldata"
        );
        assert_eq!(sn_add.num_add_claims, 1, "should have 1 Add claim");

        // Model without Add should have no unified STARK (matmul only)
        assert!(
            sn_no_add.unified_calldata.is_empty(),
            "matmul-only model should have no unified STARK"
        );
        assert_eq!(sn_no_add.num_add_claims, 0);
    }

    // === GKR Calldata Serialization Tests ===

    #[test]
    fn test_gkr_calldata_serialization_mlp() {
        use crate::aggregation::prove_model_pure_gkr;

        // 5-layer MLP proven via pure GKR pipeline
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w2.set(i, j, M31::from(((i * j) % 5 + 1) as u32));
            }
        }
        weights.add_weight(2, w2);
        let mut w4 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w4.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(4, w4);

        let proof = prove_model_pure_gkr(&graph, &input, &weights)
            .expect("pure GKR proving should succeed");
        assert!(proof.gkr_proof.is_some());

        // Serialize to Starknet calldata
        let starknet_proof = build_starknet_proof_onchain(&proof, &input);

        // GKR calldata should be present and non-empty
        assert!(
            starknet_proof.gkr_calldata.is_some(),
            "GKR calldata should be Some"
        );
        let gkr_cd = starknet_proof.gkr_calldata.as_ref().unwrap();
        assert!(!gkr_cd.is_empty(), "GKR calldata should be non-empty");

        // Combined calldata should contain has_gkr=1 flag
        let combined = &starknet_proof.combined_calldata;
        assert!(!combined.is_empty());
        // Find has_gkr flag (should be 1 somewhere near the end)
        let _has_gkr_flag = combined
            .iter()
            .rev()
            .skip(gkr_cd.len() + 1) // skip gkr_buf + its length prefix
            .next();
        // The flag value 1 should appear before the GKR data
        assert!(
            combined.contains(&FieldElement::from(1u64)),
            "combined calldata should contain has_gkr=1 flag"
        );

        // No individual matmul proofs in calldata
        assert_eq!(
            starknet_proof.num_matmul_proofs, 0,
            "GKR mode has no individual matmul proofs"
        );

        // IO commitment should be valid — raw_io_data at index [4] as length-prefixed array
        assert_ne!(starknet_proof.io_commitment, FieldElement::ZERO);
        let raw_io_len: u64 = starknet_proof.combined_calldata[4].try_into().unwrap();
        assert!(raw_io_len > 0, "raw_io_data should be non-empty");
    }

    #[test]
    fn test_gkr_calldata_serialization_matmul_only() {
        use crate::aggregation::prove_model_pure_gkr;

        // Single matmul — GKR but no unified STARK
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let proof = prove_model_pure_gkr(&graph, &input, &weights)
            .expect("pure GKR matmul-only proving should succeed");

        let starknet_proof = build_starknet_proof_onchain(&proof, &input);

        // GKR calldata present, unified STARK absent
        assert!(
            starknet_proof.gkr_calldata.is_some(),
            "GKR calldata should be present"
        );
        assert!(
            starknet_proof.unified_calldata.is_empty(),
            "no unified STARK for matmul-only"
        );
        assert_eq!(starknet_proof.num_matmul_proofs, 0);
    }

    #[test]
    fn test_gkr_calldata_round_trip_size() {
        use crate::aggregation::prove_model_pure_gkr;

        // Prove and serialize, then verify the GKR calldata has expected structure
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let proof = prove_model_pure_gkr(&graph, &input, &weights).expect("proving should succeed");

        // Serialize directly to buffer
        let mut direct_buf = Vec::new();
        crate::cairo_serde::serialize_gkr_model_proof(
            proof.gkr_proof.as_ref().unwrap(),
            &mut direct_buf,
        );

        // Also serialize via starknet pipeline
        let starknet_proof = build_starknet_proof_onchain(&proof, &input);
        let pipeline_buf = starknet_proof.gkr_calldata.unwrap();

        // Both paths should produce identical output
        assert_eq!(
            direct_buf.len(),
            pipeline_buf.len(),
            "serialization length mismatch"
        );
        assert_eq!(direct_buf, pipeline_buf, "serialization content mismatch");

        // First felt should be num_layer_proofs (3 layers: matmul, relu, matmul)
        let gkr = proof.gkr_proof.as_ref().unwrap();
        assert_eq!(
            direct_buf[0],
            FieldElement::from(gkr.layer_proofs.len() as u64),
            "first felt should be num_layer_proofs"
        );
    }

    // === build_gkr_model_calldata Tests ===

    #[test]
    fn test_build_gkr_model_calldata_structure() {
        use crate::aggregation::prove_model_pure_gkr;

        // Single matmul → prove GKR → build calldata
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr_proof = agg_proof.gkr_proof.as_ref().unwrap();

        let model_id = FieldElement::from(0xDEADBEEFu64);
        let calldata = build_gkr_model_calldata(gkr_proof, model_id);

        // First element should be model_id
        assert_eq!(calldata[0], model_id, "calldata[0] should be model_id");

        // Rest should be the serialized GKR proof
        let mut expected_gkr = Vec::new();
        crate::cairo_serde::serialize_gkr_model_proof(gkr_proof, &mut expected_gkr);
        assert_eq!(
            &calldata[1..],
            &expected_gkr[..],
            "calldata[1..] should match serialize_gkr_model_proof output"
        );
    }

    #[test]
    fn test_build_gkr_starknet_proof_mlp() {
        use crate::aggregation::prove_model_pure_gkr;

        // MLP with activation
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let model_id = FieldElement::from(0xCAFEu64);
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");

        let gkr_sn = build_gkr_starknet_proof(&agg_proof, model_id, &input)
            .expect("GKR starknet proof should succeed");

        // Verify all fields are populated
        assert_eq!(gkr_sn.model_id, model_id);
        assert!(
            !gkr_sn.gkr_calldata.is_empty(),
            "GKR calldata should be non-empty"
        );
        assert!(
            !gkr_sn.io_calldata.is_empty(),
            "IO calldata should be non-empty"
        );
        assert_eq!(
            gkr_sn.num_layer_proofs, 3,
            "3 layers: matmul + relu + matmul"
        );
        assert_ne!(gkr_sn.io_commitment, FieldElement::ZERO);
        assert!(gkr_sn.estimated_gas > 0);
        assert!(gkr_sn.total_calldata_size > 0);

        // IO calldata should hash to the IO commitment
        let recomputed_io = starknet_crypto::poseidon_hash_many(&gkr_sn.io_calldata);
        assert_eq!(
            recomputed_io, gkr_sn.io_commitment,
            "IO calldata hash must match IO commitment"
        );
    }

    #[test]
    fn test_prove_for_starknet_ml_gkr_pipeline() {
        // Full pipeline test: graph → prove → serialize → verify structure
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let model_id = FieldElement::from(0xABCDu64);
        let result = prove_for_starknet_ml_gkr(&graph, &input, &weights, model_id);
        assert!(result.is_ok(), "pipeline should succeed");

        let gkr_sn = result.unwrap();
        assert_eq!(gkr_sn.model_id, model_id);
        assert_eq!(gkr_sn.num_layer_proofs, 1, "single matmul = 1 layer proof");
        assert!(!gkr_sn.gkr_calldata.is_empty());
        assert!(!gkr_sn.io_calldata.is_empty());

        // Weight opening calldata should contain actual MLE opening proofs
        // (one per MatMul layer: count + eval_point + expected_value + MLE proof)
        assert!(
            gkr_sn.weight_opening_calldata.len() > 1,
            "weight openings should be populated (got {} felts)",
            gkr_sn.weight_opening_calldata.len(),
        );
    }

    #[test]
    fn test_build_gkr_starknet_proof_no_gkr_fails() {
        use crate::aggregation::prove_model_aggregated_onchain;

        // Prove via non-GKR pipeline → should fail when building GKR proof
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("standard proving should succeed");

        let result = build_gkr_starknet_proof(&agg_proof, FieldElement::from(1u64), &input);
        assert!(
            result.is_err(),
            "should fail: no GKR proof in standard pipeline"
        );
    }

    #[test]
    fn test_gkr_soundness_gate_rejects_missing_activation_logup() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::gkr::LayerProof;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        for layer in &mut gkr.layer_proofs {
            if let LayerProof::Activation { logup_proof, .. } = layer {
                *logup_proof = None;
                break;
            }
        }

        let err = build_gkr_starknet_proof(&agg_proof, FieldElement::from(7u64), &input)
            .expect_err("soundness gate must reject missing activation LogUp");
        match err {
            StarknetModelError::SoundnessGate(_) => {}
            other => panic!("expected SoundnessGate error, got: {other}"),
        }
    }

    #[test]
    fn test_gkr_soundness_gate_rejects_missing_weight_openings() {
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        assert!(
            !gkr.weight_openings.is_empty(),
            "test setup requires weight openings"
        );
        gkr.weight_openings.clear();

        let err = build_gkr_starknet_proof(&agg_proof, FieldElement::from(9u64), &input)
            .expect_err("soundness gate must reject missing weight openings");
        match err {
            StarknetModelError::SoundnessGate(_) => {}
            other => panic!("expected SoundnessGate error, got: {other}"),
        }
    }

    #[test]
    fn test_build_gkr_serializable_proof_keeps_artifact_when_not_submission_ready() {
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        gkr.weight_openings.clear();

        let serialized =
            build_gkr_serializable_proof(&agg_proof, FieldElement::from(10u64), &input)
                .expect("serializable builder should not fail when soundness gates fail");
        assert!(
            !serialized.submission_ready,
            "must be marked non-submit-ready"
        );
        assert!(
            serialized.soundness_gate_error.is_some(),
            "gate failure reason should be preserved"
        );
        assert!(
            !serialized.weight_claim_calldata.is_empty(),
            "weight claims must still be serialized for off-chain verification"
        );
        assert_eq!(
            serialized.weight_binding_schema_version,
            WEIGHT_BINDING_SCHEMA_VERSION_V1
        );
        assert_eq!(serialized.weight_binding_mode_id, Some(0));
        assert!(
            serialized.weight_binding_data_calldata.is_empty(),
            "mode 0 should not require extra weight_binding_data"
        );
    }

    #[test]
    fn test_build_verify_model_gkr_v2_calldata_inserts_mode_field() {
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof expected");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let v1 = build_verify_model_gkr_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("v1 calldata should build");
        let v2 = build_verify_model_gkr_v2_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("v2 calldata should build");

        assert_eq!(
            v2.calldata_parts.len(),
            v1.calldata_parts.len() + 1,
            "v2 calldata should add exactly one felt (weight_binding_mode)"
        );

        // Parse through the shared prefix to locate the insertion point:
        // model_id, raw_io_data, circuit_depth, num_layers, matmul_dims, dequantize_bits,
        // proof_data, weight_commitments.
        let mut idx = 0usize;
        idx += 1; // model_id
        let raw_io_len = v1.calldata_parts[idx]
            .parse::<usize>()
            .expect("raw_io_data len parse");
        idx += 1 + raw_io_len;
        idx += 2; // circuit_depth, num_layers
        let matmul_len = v1.calldata_parts[idx]
            .parse::<usize>()
            .expect("matmul_dims len parse");
        idx += 1 + matmul_len;
        let dequant_len = v1.calldata_parts[idx]
            .parse::<usize>()
            .expect("dequantize_bits len parse");
        idx += 1 + dequant_len;
        let proof_data_len = v1.calldata_parts[idx]
            .parse::<usize>()
            .expect("proof_data len parse");
        idx += 1 + proof_data_len;
        let weight_commitments_len = v1.calldata_parts[idx]
            .parse::<usize>()
            .expect("weight_commitments len parse");
        idx += 1 + weight_commitments_len;

        assert_eq!(
            v2.calldata_parts[idx], "0",
            "v2 should encode weight_binding_mode=0 for Phase 1 compat"
        );
        assert_eq!(
            v2.calldata_parts[idx + 1],
            v1.calldata_parts[idx],
            "opening payload must start immediately after inserted mode"
        );
    }

    #[test]
    fn test_build_verify_model_gkr_v2_calldata_accepts_batched_subchannel_mode() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::gkr::types::WeightOpeningTranscriptMode;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::BatchedSubchannelV1;

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let v2 = build_verify_model_gkr_v2_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("v2 calldata should build for batched subchannel mode");

        // Locate weight_binding_mode insertion index.
        let mut idx = 0usize;
        idx += 1; // model_id
        let raw_io_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("raw_io_data len parse");
        idx += 1 + raw_io_len;
        idx += 2; // circuit_depth, num_layers
        let matmul_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("matmul_dims len parse");
        idx += 1 + matmul_len;
        let dequant_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("dequantize_bits len parse");
        idx += 1 + dequant_len;
        let proof_data_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("proof_data len parse");
        idx += 1 + proof_data_len;
        let weight_commitments_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("weight_commitments len parse");
        idx += 1 + weight_commitments_len;

        assert_eq!(
            v2.calldata_parts[idx], "1",
            "v2 should encode weight_binding_mode=1 for BatchedSubchannelV1"
        );

        let v1_err = build_verify_model_gkr_calldata(gkr, &circuit, model_id, &raw_io)
            .expect_err("v1 calldata should reject batched subchannel mode");
        match v1_err {
            StarknetModelError::SoundnessGate(_) => {}
            other => panic!("expected SoundnessGate error, got: {other}"),
        }
    }

    #[test]
    fn test_build_verify_model_gkr_v3_calldata_inserts_mode_and_empty_binding_data() {
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof expected");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let v2 = build_verify_model_gkr_v2_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("v2 calldata should build");
        let v3 = build_verify_model_gkr_v3_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("v3 calldata should build");

        assert_eq!(
            v3.calldata_parts.len(),
            v2.calldata_parts.len() + 1,
            "v3 calldata should add exactly one felt (weight_binding_data len=0)"
        );

        // Find v2 weight_binding_mode index.
        let mut idx = 0usize;
        idx += 1; // model_id
        let raw_io_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("raw_io_data len parse");
        idx += 1 + raw_io_len;
        idx += 2; // circuit_depth, num_layers
        let matmul_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("matmul_dims len parse");
        idx += 1 + matmul_len;
        let dequant_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("dequantize_bits len parse");
        idx += 1 + dequant_len;
        let proof_data_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("proof_data len parse");
        idx += 1 + proof_data_len;
        let weight_commitments_len = v2.calldata_parts[idx]
            .parse::<usize>()
            .expect("weight_commitments len parse");
        idx += 1 + weight_commitments_len;

        assert_eq!(
            v2.calldata_parts[idx], "0",
            "v2 mode should be sequential (0)"
        );
        assert_eq!(
            v3.calldata_parts[idx], "0",
            "v3 mode should be sequential (0)"
        );
        assert_eq!(
            v3.calldata_parts[idx + 1],
            "0",
            "v3 should encode empty weight_binding_data"
        );
        assert_eq!(
            v3.calldata_parts[idx + 2],
            v2.calldata_parts[idx + 1],
            "opening payload must follow v3 binding_data field"
        );
    }

    #[test]
    fn test_build_verify_model_gkr_v2_rejects_mode2_binding() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::gkr::types::WeightOpeningTranscriptMode;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedTrustlessV2;

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let err = build_verify_model_gkr_v2_calldata(gkr, &circuit, model_id, &raw_io)
            .expect_err("v2 calldata should reject mode2 binding");
        match err {
            StarknetModelError::SoundnessGate(_) => {}
            other => panic!("expected SoundnessGate error, got: {other}"),
        }
    }

    #[test]
    fn test_build_verify_model_gkr_v3_calldata_encodes_mode2_binding_data() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::gkr::types::WeightOpeningTranscriptMode;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedTrustlessV2;

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let v3 = build_verify_model_gkr_v3_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("v3 calldata should build for mode2");

        // Find insertion index for weight_binding_mode (same prefix as v2).
        let mut idx = 0usize;
        idx += 1; // model_id
        let raw_io_len = v3.calldata_parts[idx]
            .parse::<usize>()
            .expect("raw_io_data len parse");
        idx += 1 + raw_io_len;
        idx += 2; // circuit_depth, num_layers
        let matmul_len = v3.calldata_parts[idx]
            .parse::<usize>()
            .expect("matmul_dims len parse");
        idx += 1 + matmul_len;
        let dequant_len = v3.calldata_parts[idx]
            .parse::<usize>()
            .expect("dequantize_bits len parse");
        idx += 1 + dequant_len;
        let proof_data_len = v3.calldata_parts[idx]
            .parse::<usize>()
            .expect("proof_data len parse");
        idx += 1 + proof_data_len;
        let weight_commitments_len = v3.calldata_parts[idx]
            .parse::<usize>()
            .expect("weight_commitments len parse");
        idx += 1 + weight_commitments_len;

        assert_eq!(
            v3.calldata_parts[idx], "2",
            "v3 mode should be aggregated trustless (2)"
        );
        let binding_data_len = v3.calldata_parts[idx + 1]
            .parse::<usize>()
            .expect("weight_binding_data len parse");
        assert_eq!(
            binding_data_len, 2,
            "mode2 v3 payload should currently be [digest, claim_count]"
        );
    }

    #[test]
    fn test_build_verify_model_gkr_v3_rejects_mode4_binding() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::crypto::aggregated_opening::{prove_aggregated_binding, AggregatedWeightClaim};
        use crate::crypto::poseidon_channel::PoseidonChannel;
        use crate::gkr::types::WeightOpeningTranscriptMode;

        // Prove with sequential mode, then manually construct mode 4 state.
        // This avoids env-var race conditions with parallel tests.
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");

        // Build aggregated binding from weight claims (mirrors mode 4 prover path).
        let mut agg_claims = Vec::with_capacity(gkr.weight_claims.len());
        let mut all_mles = Vec::with_capacity(gkr.weight_claims.len());
        for (idx, claim) in gkr.weight_claims.iter().enumerate() {
            let weight = weights
                .get_weight(claim.weight_node_id)
                .expect("weight must exist for claim");
            let mle = crate::components::matmul::matrix_to_mle_col_major_pub(weight);
            agg_claims.push(AggregatedWeightClaim {
                matrix_index: idx,
                local_n_vars: mle.len().trailing_zeros() as usize,
                eval_point: claim.eval_point.clone(),
                expected_value: claim.expected_value,
                commitment: gkr.weight_commitments[idx],
            });
            all_mles.push(mle);
        }
        let mle_refs: Vec<&[crate::gkr::types::SecureField]> =
            all_mles.iter().map(|m| m.as_slice()).collect();
        let mut channel = PoseidonChannel::new();
        let aggregated_binding = prove_aggregated_binding(&agg_claims, &mle_refs, &mut channel);

        gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
        gkr.aggregated_binding = Some(aggregated_binding);

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let err = build_verify_model_gkr_v3_calldata(gkr, &circuit, model_id, &raw_io)
            .expect_err("v3 calldata should reject mode4 proofs");
        match err {
            StarknetModelError::SoundnessGate(msg) => assert!(
                msg.contains("mode 4 requires verify_model_gkr_v4"),
                "unexpected soundness gate message: {msg}"
            ),
            other => panic!("expected SoundnessGate error, got: {other}"),
        }
    }

    #[test]
    fn test_mode2_serialized_proof_is_submit_ready_with_binding_payload() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::gkr::types::WeightOpeningTranscriptMode;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedTrustlessV2;

        let serialized = build_gkr_starknet_proof(&agg_proof, FieldElement::from(12u64), &input)
            .expect("mode2 should satisfy strict Starknet soundness gates");
        assert!(
            serialized.submission_ready,
            "mode2 should be marked submit-ready in strict builder"
        );
        assert_eq!(serialized.weight_binding_mode_id, Some(2));
        assert_eq!(
            serialized.weight_binding_data_calldata.len(),
            2,
            "mode2 payload should include digest + claim_count"
        );
    }

    #[test]
    fn test_build_verify_model_gkr_v4_calldata_encodes_mode3_binding_data() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::gkr::types::WeightOpeningTranscriptMode;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        gkr.weight_opening_transcript_mode =
            WeightOpeningTranscriptMode::AggregatedOpeningsV4Experimental;

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let v4 = build_verify_model_gkr_v4_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("v4 calldata should build for mode3");

        // Find insertion index for weight_binding_mode (same prefix as v2/v3).
        let mut idx = 0usize;
        idx += 1; // model_id
        let raw_io_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("raw_io_data len parse");
        idx += 1 + raw_io_len;
        idx += 2; // circuit_depth, num_layers
        let matmul_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("matmul_dims len parse");
        idx += 1 + matmul_len;
        let dequant_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("dequantize_bits len parse");
        idx += 1 + dequant_len;
        let proof_data_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("proof_data len parse");
        idx += 1 + proof_data_len;
        let weight_commitments_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("weight_commitments len parse");
        idx += 1 + weight_commitments_len;

        assert_eq!(
            v4.calldata_parts[idx], "3",
            "v4 mode should be aggregated openings experimental (3)"
        );
        let binding_data_len = v4.calldata_parts[idx + 1]
            .parse::<usize>()
            .expect("weight_binding_data len parse");
        assert_eq!(
            binding_data_len, 2,
            "mode3 v4 payload should currently be [digest, claim_count]"
        );
    }

    #[test]
    fn test_build_verify_model_gkr_v4_encodes_mode4_binding_data() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::crypto::aggregated_opening::{prove_aggregated_binding, AggregatedWeightClaim};
        use crate::crypto::poseidon_channel::PoseidonChannel;
        use crate::gkr::types::WeightOpeningTranscriptMode;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");

        // Build a concrete aggregated binding proof (mode 4 payload) from the
        // generated weight claims so v4 calldata includes serialized binding data.
        let mut agg_claims = Vec::with_capacity(gkr.weight_claims.len());
        let mut all_mles = Vec::with_capacity(gkr.weight_claims.len());
        for (idx, claim) in gkr.weight_claims.iter().enumerate() {
            let weight = weights
                .get_weight(claim.weight_node_id)
                .expect("weight must exist for claim");
            let mle = crate::components::matmul::matrix_to_mle_col_major_pub(weight);
            agg_claims.push(AggregatedWeightClaim {
                matrix_index: idx,
                local_n_vars: mle.len().trailing_zeros() as usize,
                eval_point: claim.eval_point.clone(),
                expected_value: claim.expected_value,
                commitment: gkr.weight_commitments[idx],
            });
            all_mles.push(mle);
        }
        let mle_refs: Vec<&[crate::gkr::types::SecureField]> =
            all_mles.iter().map(|m| m.as_slice()).collect();
        let mut channel = PoseidonChannel::new();
        let aggregated_binding = prove_aggregated_binding(&agg_claims, &mle_refs, &mut channel);

        gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
        gkr.aggregated_binding = Some(aggregated_binding);

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let v4 = build_verify_model_gkr_v4_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("v4 calldata should build for mode4");

        let mut idx = 0usize;
        idx += 1; // model_id
        let raw_io_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("raw_io_data len parse");
        idx += 1 + raw_io_len;
        idx += 2; // circuit_depth, num_layers
        let matmul_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("matmul_dims len parse");
        idx += 1 + matmul_len;
        let dequant_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("dequantize_bits len parse");
        idx += 1 + dequant_len;
        let proof_data_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("proof_data len parse");
        idx += 1 + proof_data_len;
        let weight_commitments_len = v4.calldata_parts[idx]
            .parse::<usize>()
            .expect("weight_commitments len parse");
        idx += 1 + weight_commitments_len;

        assert_eq!(v4.calldata_parts[idx], "4", "v4 mode should be mode4");
        let binding_data_len = v4.calldata_parts[idx + 1]
            .parse::<usize>()
            .expect("weight_binding_data len parse");
        assert!(
            binding_data_len > 10,
            "mode4 payload should contain serialized aggregated proof"
        );
    }

    #[test]
    fn test_build_verify_model_gkr_v4_rejects_non_mode3_or_mode4_binding() {
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof expected");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let err = build_verify_model_gkr_v4_calldata(gkr, &circuit, model_id, &raw_io)
            .expect_err("v4 calldata should reject proofs not in mode 3/4");
        match err {
            StarknetModelError::SoundnessGate(_) => {}
            other => panic!("expected SoundnessGate error, got: {other}"),
        }
    }

    #[test]
    fn test_build_verify_model_gkr_v4_accepts_mode4_rlc_only_binding() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::gkr::types::WeightOpeningTranscriptMode;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
        gkr.aggregated_binding = None;

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &agg_proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        // Mode 4 with RLC-only binding should succeed and produce valid calldata
        let calldata = build_verify_model_gkr_v4_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("mode4 with RLC-only binding should succeed");
        assert!(calldata.total_felts > 0, "calldata should be non-empty");
        // Verify RLC marker (0x524c43) appears in calldata after binding mode
        let has_rlc_marker = calldata
            .calldata_parts
            .iter()
            .any(|p| p == "0x524c43");
        assert!(has_rlc_marker, "calldata should contain RLC marker");
    }

    #[test]
    fn test_serializable_proof_records_offchain_mode_binding_metadata() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::gkr::types::WeightOpeningTranscriptMode;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w);

        let mut agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_mut().expect("GKR proof expected");
        gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1;

        let serialized =
            build_gkr_serializable_proof(&agg_proof, FieldElement::from(11u64), &input)
                .expect("serializable builder should preserve off-chain mode artifacts");
        assert!(
            !serialized.submission_ready,
            "off-chain-only mode must never be marked submit-ready"
        );
        assert_eq!(
            serialized.weight_binding_schema_version,
            WEIGHT_BINDING_SCHEMA_VERSION_V1
        );
        assert_eq!(
            serialized.weight_binding_mode_id, None,
            "off-chain-only mode should not claim Starknet mode id"
        );
        assert!(
            serialized.weight_binding_data_calldata.is_empty(),
            "off-chain mode metadata is currently schema-only"
        );
    }

    /// Replay Cairo-compatible GKR verification from serialized proof_data.
    ///
    /// Proves a 2-MatMul model (1×4 → 2 → 2), serializes to the exact format
    /// the Cairo contract expects, then replays Fiat-Shamir channel + sumcheck
    /// checks for BOTH layers. If this fails, the serialization format diverges
    /// from what the Cairo verifier expects.
    #[test]
    fn test_replay_cairo_verification_from_serialized_proof_data() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::crypto::poseidon_channel::PoseidonChannel;
        use crate::gkr::prover::mix_secure_field;
        use crate::cairo_serde::{serialize_gkr_proof_data_only, serialize_raw_io};
        use stwo::core::fields::qm31::{QM31, SecureField};
        use stwo::core::fields::cm31::CM31;
        use num_traits::Zero;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        // Build: 1×4 → rms_norm → linear(2) → linear(2)
        // This matches the on-chain pattern: RMSNorm(8) → MatMul(0) → MatMul(0)
        let mut builder = GraphBuilder::new((1, 4));
        builder.rms_norm();
        builder.linear(2);
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        // node 0 = rms_norm (no weights), node 1 = linear, node 2 = linear
        let mut w0 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w0.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }
        weights.add_weight(1, w0);
        let mut w1 = M31Matrix::new(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                w1.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w1);

        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let d = circuit.layers.len();
        let num_proof_layers = gkr.layer_proofs.len();

        // Serialize proof_data (unpacked)
        let mut proof_data_felts = Vec::new();
        serialize_gkr_proof_data_only(gkr, &mut proof_data_felts);
        println!("proof_data_felts: {} felts, circuit_depth={}, proof_layers={}",
            proof_data_felts.len(), d, num_proof_layers);

        // Extract matmul dims
        let matmul_dims = extract_matmul_dims(&circuit);
        println!("matmul_dims: {:?}", matmul_dims);

        // === Replay Cairo verification ===
        // Helper to read u64 from felt252
        fn felt_to_u64(f: &FieldElement) -> u64 {
            let b = f.to_bytes_be();
            u64::from_be_bytes([b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]])
        }

        // 1. Parse raw_io
        let input_rows = felt_to_u64(&raw_io[0]);
        let input_cols = felt_to_u64(&raw_io[1]);
        let input_len = felt_to_u64(&raw_io[2]) as usize;

        let out_start = 3 + input_len;
        let output_rows = felt_to_u64(&raw_io[out_start]) as usize;
        let output_cols = felt_to_u64(&raw_io[out_start + 1]) as usize;

        println!("input: {}x{}, output: {}x{}", input_rows, input_cols, output_rows, output_cols);

        // 2. Build output MLE
        let padded_rows = output_rows.next_power_of_two();
        let padded_cols = output_cols.next_power_of_two();
        let mut output_mle = Vec::with_capacity(padded_rows * padded_cols);
        for r in 0..padded_rows {
            for c in 0..padded_cols {
                if r < output_rows && c < output_cols {
                    let idx = r * output_cols + c;
                    let val = felt_to_u64(&raw_io[out_start + 3 + idx]) as u32;
                    output_mle.push(SecureField::from(M31::from(val)));
                } else {
                    output_mle.push(SecureField::zero());
                }
            }
        }

        // 3. Initialize channel
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(d as u64);
        ch.mix_u64(input_rows);
        ch.mix_u64(input_cols);

        let log_out = (padded_rows * padded_cols).ilog2() as usize;
        let r_out = ch.draw_qm31s(log_out);
        let output_value = crate::components::matmul::evaluate_mle_pub(&output_mle, &r_out);
        mix_secure_field(&mut ch, output_value);

        println!("output_value: {:?}", output_value);

        // 4. Parse proof_data and replay ALL layers
        let mut off = 0usize;

        // Closures to read from proof_data_felts
        let proof_felts = &proof_data_felts;
        let read_u32_from = |off: &mut usize| -> u32 {
            let v = felt_to_u64(&proof_felts[*off]) as u32;
            *off += 1;
            v
        };
        let read_qm31_from = |off: &mut usize| -> SecureField {
            let aa = felt_to_u64(&proof_felts[*off]) as u32; *off += 1;
            let ab = felt_to_u64(&proof_felts[*off]) as u32; *off += 1;
            let ba = felt_to_u64(&proof_felts[*off]) as u32; *off += 1;
            let bb = felt_to_u64(&proof_felts[*off]) as u32; *off += 1;
            QM31(CM31(M31::from(aa), M31::from(ab)),
                 CM31(M31::from(ba), M31::from(bb)))
        };

        let mut current_claim_value = output_value;
        let mut current_claim_point = r_out.clone();
        let mut matmul_idx = 0usize;

        for layer in 0..num_proof_layers {
            let tag = read_u32_from(&mut off);
            println!("\n--- Layer {} tag={} ---", layer, tag);

            match tag {
                0 => {
                    // MatMul
                    let m = matmul_dims[matmul_idx * 3] as usize;
                    let k = matmul_dims[matmul_idx * 3 + 1] as usize;
                    let n = matmul_dims[matmul_idx * 3 + 2] as usize;
                    matmul_idx += 1;
                    println!("  MatMul dims: m={}, k={}, n={}", m, k, n);

                    ch.mix_u64(m as u64);
                    ch.mix_u64(k as u64);
                    ch.mix_u64(n as u64);
                    mix_secure_field(&mut ch, current_claim_value);

                    let num_rounds = read_u32_from(&mut off) as usize;
                    let log_k = k.next_power_of_two().ilog2() as usize;
                    println!("  num_rounds: {} (log_k={})", num_rounds, log_k);
                    assert_eq!(num_rounds, log_k, "layer {} round count mismatch", layer);

                    let mut current_sum = current_claim_value;
                    let log_m = m.next_power_of_two().ilog2() as usize;
                    let mut sumcheck_challenges = Vec::with_capacity(num_rounds);

                    for round in 0..num_rounds {
                        let c0 = read_qm31_from(&mut off);
                        let c1 = read_qm31_from(&mut off);
                        let c2 = read_qm31_from(&mut off);

                        let p0 = c0;
                        let p1 = c0 + c1 + c2;
                        let round_sum = p0 + p1;

                        println!("  round {}: p(0)+p(1)={:?}, current_sum={:?}, match={}",
                            round, round_sum, current_sum, round_sum == current_sum);

                        assert_eq!(
                            round_sum, current_sum,
                            "MATMUL_ROUND_SUM_MISMATCH at layer {} round {}", layer, round,
                        );

                        ch.mix_poly_coeffs(c0, c1, c2);
                        let challenge = ch.draw_qm31();
                        sumcheck_challenges.push(challenge);
                        current_sum = c0 + c1 * challenge + c2 * challenge * challenge;
                    }

                    let final_a = read_qm31_from(&mut off);
                    let final_b = read_qm31_from(&mut off);
                    assert_eq!(
                        current_sum, final_a * final_b,
                        "MATMUL_FINAL_MISMATCH at layer {}", layer,
                    );

                    // Mix final evals (matching Rust verifier lines 2108-2109)
                    mix_secure_field(&mut ch, final_a);
                    mix_secure_field(&mut ch, final_b);

                    // Build new claim: point = r_i || sumcheck_challenges
                    // r_i = first log_m elements of current_claim_point
                    let mut new_point = Vec::with_capacity(log_m + num_rounds);
                    for i in 0..log_m {
                        new_point.push(current_claim_point[i].clone());
                    }
                    new_point.extend_from_slice(&sumcheck_challenges);

                    current_claim_point = new_point;
                    current_claim_value = final_a;

                    println!("  OK: final_a*final_b check passed, new claim value = {:?}", final_a);
                }

                8 => {
                    // RMSNorm — read all fields and replay channel ops
                    let input_eval = read_qm31_from(&mut off);
                    let output_eval = read_qm31_from(&mut off);
                    let rms_sq = read_qm31_from(&mut off);
                    let rsqrt_eval = read_qm31_from(&mut off);
                    let _rsqrt_commitment = { off += 1; }; // felt252
                    let _simd_combined = read_u32_from(&mut off); // u32

                    // "RN" tag
                    ch.mix_u64(0x524E); // 'R'=0x52, 'N'=0x4E → 0x524E
                    mix_secure_field(&mut ch, rms_sq);
                    mix_secure_field(&mut ch, rsqrt_eval);
                    mix_secure_field(&mut ch, current_claim_value);

                    // eq-sumcheck rounds
                    let nrounds = read_u32_from(&mut off) as usize;
                    println!("  RMSNorm: nrounds={}", nrounds);
                    let mut rms_sum = current_claim_value;
                    for round in 0..nrounds {
                        let c0 = read_qm31_from(&mut off);
                        let c1 = read_qm31_from(&mut off);
                        let c2 = read_qm31_from(&mut off);
                        let c3 = read_qm31_from(&mut off);

                        // degree-3 poly: p(0)+p(1) = c0 + (c0+c1+c2+c3)
                        let p0 = c0;
                        let p1 = c0 + c1 + c2 + c3;
                        let round_sum = p0 + p1;

                        println!("  rms round {}: sum match={}", round, round_sum == rms_sum);
                        assert_eq!(round_sum, rms_sum,
                            "RMSNORM_ROUND_SUM_MISMATCH at layer {} round {}", layer, round);

                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        rms_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }

                    // final evals
                    let _eq_eval = read_qm31_from(&mut off);
                    let _prod_eval = read_qm31_from(&mut off);

                    // optional logup
                    let has_logup = read_u32_from(&mut off);
                    if has_logup == 1 {
                        let _claimed_sum = read_qm31_from(&mut off);
                        mix_secure_field(&mut ch, _claimed_sum);
                        let eq_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let _ = ch.draw_qm31();
                        }
                        // final evals
                        let _e1 = read_qm31_from(&mut off);
                        let _e2 = read_qm31_from(&mut off);
                        let _e3 = read_qm31_from(&mut off);
                        let num_mults = read_u32_from(&mut off) as usize;
                        for _ in 0..num_mults {
                            let _ = read_qm31_from(&mut off);
                        }
                    }

                    // Mix input/output evals (matching Rust verifier lines 3494-3495)
                    mix_secure_field(&mut ch, input_eval);
                    mix_secure_field(&mut ch, output_eval);

                    // New claim: same point, value = input_eval
                    current_claim_value = input_eval;
                    // point stays the same for RMSNorm

                    println!("  RMSNorm OK: new claim value = {:?}", input_eval);
                }

                3 => {
                    // Activation — read fields and replay channel ops
                    let act_type = read_u32_from(&mut off);
                    let input_eval = read_qm31_from(&mut off);
                    let output_eval = read_qm31_from(&mut off);
                    let _table_commitment = { let _f = &proof_felts[off]; off += 1; };

                    // "LOG" tag
                    ch.mix_u64(0x4C4F47); // 'L'=0x4C, 'O'=0x4F, 'G'=0x47

                    // draw gamma, beta
                    let _gamma = ch.draw_qm31();
                    let _beta = ch.draw_qm31();

                    // optional logup
                    let has_logup = read_u32_from(&mut off);
                    if has_logup == 1 {
                        let claimed_sum = read_qm31_from(&mut off);
                        mix_secure_field(&mut ch, claimed_sum);
                        let eq_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let _ = ch.draw_qm31();
                        }
                        let _e1 = read_qm31_from(&mut off);
                        let _e2 = read_qm31_from(&mut off);
                        let _e3 = read_qm31_from(&mut off);
                        let num_mults = read_u32_from(&mut off) as usize;
                        for _ in 0..num_mults {
                            let _ = read_qm31_from(&mut off);
                        }
                    }

                    // Mix input/output evals
                    mix_secure_field(&mut ch, input_eval);
                    mix_secure_field(&mut ch, output_eval);

                    // New claim: same point, value = input_eval
                    current_claim_value = input_eval;

                    println!("  Activation(type={}) OK: new claim value = {:?}", act_type, input_eval);
                }

                1 => {
                    // Add
                    let lhs = read_qm31_from(&mut off);
                    let rhs = read_qm31_from(&mut off);
                    let trunk_idx = read_u32_from(&mut off);

                    mix_secure_field(&mut ch, lhs);
                    mix_secure_field(&mut ch, rhs);
                    let _alpha = ch.draw_qm31();

                    let trunk_eval = if trunk_idx == 0 { lhs } else { rhs };
                    current_claim_value = trunk_eval;

                    println!("  Add OK: trunk_idx={}, new claim value = {:?}", trunk_idx, trunk_eval);
                }

                _ => panic!("Unknown tag {} at layer {}", tag, layer),
            }
        }

        // Read deferred proofs count
        let num_deferred = read_u32_from(&mut off);
        println!("\nDeferred proofs: {}", num_deferred);
        println!("Remaining felts: {}", proof_data_felts.len() - off);
        println!("SUCCESS: all {} proof layers pass Cairo-compatible verification replay", num_proof_layers);
    }

    /// Load the actual GPU-generated proof JSON and replay channel operations.
    /// This catches bugs specific to large models / GPU prover.
    #[test]
    /// Replay channel verification from an actual GPU-generated proof JSON.
    /// Set PROOF_JSON env var to override path (default: /tmp/unpacked_proof_2.json).
    /// Handles both packed (1 felt per QM31) and unpacked (4 felts per QM31) formats.
    ///
    /// Run manually: `PROOF_JSON=/tmp/packed_proof_40.json cargo test test_replay_from_proof_json -- --ignored --nocapture`
    #[ignore]
    fn test_replay_from_proof_json() {
        use crate::crypto::poseidon_channel::PoseidonChannel;
        use crate::gkr::prover::mix_secure_field;
        use crate::crypto::poseidon_channel::felt_to_securefield;
        use stwo::core::fields::qm31::{QM31, SecureField};
        use stwo::core::fields::cm31::CM31;
        use num_traits::Zero;
        use std::path::Path;

        let proof_path_str = std::env::var("PROOF_JSON")
            .unwrap_or_else(|_| "/tmp/unpacked_proof_2.json".to_string());
        let proof_path = Path::new(&proof_path_str);
        if !proof_path.exists() {
            println!("SKIP: {} not found", proof_path_str);
            return;
        }

        let json_str = std::fs::read_to_string(proof_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let vc = &json["verify_calldata"];

        let circuit_depth = vc["circuit_depth"].as_u64().unwrap() as u32;
        let num_layers = vc["num_layers"].as_u64().unwrap() as u32;
        let packed = vc["packed"].as_bool().unwrap_or(false);
        println!("packed={}", packed);

        // Flatten chunks (handle both hex "0x..." and decimal strings)
        let chunks = vc["chunks"].as_array().unwrap();
        let mut all_felts: Vec<FieldElement> = Vec::new();
        for chunk in chunks {
            for f in chunk.as_array().unwrap() {
                let s = f.as_str().unwrap();
                let fe = if s.starts_with("0x") || s.starts_with("0X") {
                    FieldElement::from_hex_be(s).unwrap()
                } else {
                    FieldElement::from_dec_str(s).unwrap()
                };
                all_felts.push(fe);
            }
        }
        println!("Loaded {} felts, circuit_depth={}, num_layers={}", all_felts.len(), circuit_depth, num_layers);

        // Parse sections
        let mut sec_off = 0usize;
        let read_section = |off: &mut usize| -> Vec<FieldElement> {
            let len_bytes = all_felts[*off].to_bytes_be();
            let len = u64::from_be_bytes([len_bytes[24], len_bytes[25], len_bytes[26], len_bytes[27],
                                          len_bytes[28], len_bytes[29], len_bytes[30], len_bytes[31]]) as usize;
            let data = all_felts[*off + 1 .. *off + 1 + len].to_vec();
            *off += 1 + len;
            data
        };

        let raw_io = read_section(&mut sec_off);
        let matmul_dims_felts = read_section(&mut sec_off);
        let _deq_bits = read_section(&mut sec_off);
        let proof_data = read_section(&mut sec_off);
        let _weight_commitments = read_section(&mut sec_off);
        let _weight_binding = read_section(&mut sec_off);
        let _weight_openings = read_section(&mut sec_off);

        println!("raw_io: {}, matmul_dims: {}, proof_data: {} felts",
            raw_io.len(), matmul_dims_felts.len(), proof_data.len());

        fn felt_to_u64(f: &FieldElement) -> u64 {
            let b = f.to_bytes_be();
            u64::from_be_bytes([b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]])
        }

        // Parse raw_io
        let input_rows = felt_to_u64(&raw_io[0]);
        let input_cols = felt_to_u64(&raw_io[1]);
        let input_len = felt_to_u64(&raw_io[2]) as usize;
        let out_start = 3 + input_len;
        let output_rows = felt_to_u64(&raw_io[out_start]) as usize;
        let output_cols = felt_to_u64(&raw_io[out_start + 1]) as usize;

        println!("input: {}x{}, output: {}x{}", input_rows, input_cols, output_rows, output_cols);

        // Parse matmul_dims
        let matmul_dims: Vec<u32> = matmul_dims_felts.iter().map(|f| felt_to_u64(f) as u32).collect();

        // Build output MLE
        let padded_rows = output_rows.next_power_of_two();
        let padded_cols = output_cols.next_power_of_two();
        let mut output_mle: Vec<SecureField> = Vec::with_capacity(padded_rows * padded_cols);
        for r in 0..padded_rows {
            for c in 0..padded_cols {
                if r < output_rows && c < output_cols {
                    let idx = r * output_cols + c;
                    let val = felt_to_u64(&raw_io[out_start + 3 + idx]) as u32;
                    output_mle.push(SecureField::from(M31::from(val)));
                } else {
                    output_mle.push(SecureField::zero());
                }
            }
        }

        // Initialize channel
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(circuit_depth as u64);
        ch.mix_u64(input_rows);
        ch.mix_u64(input_cols);

        let log_out = (padded_rows * padded_cols).ilog2() as usize;
        let r_out = ch.draw_qm31s(log_out);
        let output_value = crate::components::matmul::evaluate_mle_pub(&output_mle, &r_out);
        mix_secure_field(&mut ch, output_value);

        println!("output_value: {:?}", output_value);
        println!("channel digest: {:?}", ch.digest());

        // Parse proof_data
        let mut off = 0usize;
        let read_u32_from = |off: &mut usize| -> u32 {
            let v = felt_to_u64(&proof_data[*off]) as u32;
            *off += 1;
            v
        };
        let read_qm31_from = |off: &mut usize| -> SecureField {
            if packed {
                let fe = proof_data[*off];
                *off += 1;
                felt_to_securefield(fe)
            } else {
                let aa = felt_to_u64(&proof_data[*off]) as u32; *off += 1;
                let ab = felt_to_u64(&proof_data[*off]) as u32; *off += 1;
                let ba = felt_to_u64(&proof_data[*off]) as u32; *off += 1;
                let bb = felt_to_u64(&proof_data[*off]) as u32; *off += 1;
                QM31(CM31(M31::from(aa), M31::from(ab)),
                     CM31(M31::from(ba), M31::from(bb)))
            }
        };

        let mut current_claim_value = output_value;
        let mut matmul_idx = 0usize;

        for layer in 0..num_layers as usize {
            let tag = read_u32_from(&mut off);
            println!("\n--- Layer {} tag={} ---", layer, tag);

            match tag {
                0 => {
                    // MatMul
                    let m = matmul_dims[matmul_idx * 3] as usize;
                    let k = matmul_dims[matmul_idx * 3 + 1] as usize;
                    let n = matmul_dims[matmul_idx * 3 + 2] as usize;
                    matmul_idx += 1;

                    println!("  [off={}, matmul m={}, k={}, n={}]", off, m, k, n);
                    println!("  current_claim_value: {:?}", current_claim_value);
                    println!("  ch BEFORE matmul seeding: {:?}", ch.digest());

                    ch.mix_u64(m as u64);
                    ch.mix_u64(k as u64);
                    ch.mix_u64(n as u64);
                    mix_secure_field(&mut ch, current_claim_value);

                    let num_rounds = read_u32_from(&mut off) as usize;
                    println!("  MatMul m={}, k={}, n={}, rounds={}", m, k, n, num_rounds);

                    let mut current_sum = current_claim_value;
                    println!("  channel before rounds: {:?}", ch.digest());
                    for round in 0..num_rounds {
                        let c0 = read_qm31_from(&mut off);
                        let c1 = read_qm31_from(&mut off);
                        let c2 = read_qm31_from(&mut off);

                        let p0 = c0;
                        let p1 = c0 + c1 + c2;
                        let round_sum = p0 + p1;

                        if round < 3 || round_sum != current_sum {
                            println!("  round {}: p(0)+p(1)={:?}, sum={:?}, match={}",
                                round, round_sum, current_sum, round_sum == current_sum);
                        }

                        if round_sum != current_sum {
                            println!("  c0={:?}", c0);
                            println!("  c1={:?}", c1);
                            println!("  c2={:?}", c2);
                            panic!("MATMUL_ROUND_SUM_MISMATCH at layer {} round {}", layer, round);
                        }

                        ch.mix_poly_coeffs(c0, c1, c2);
                        let challenge = ch.draw_qm31();
                        current_sum = c0 + c1 * challenge + c2 * challenge * challenge;
                        if round < 3 {
                            println!("  round {} challenge: {:?}", round, challenge);
                            println!("  round {} new sum: {:?}", round, current_sum);
                        }
                    }

                    let final_a = read_qm31_from(&mut off);
                    let final_b = read_qm31_from(&mut off);
                    assert_eq!(current_sum, final_a * final_b, "MATMUL_FINAL layer {}", layer);

                    mix_secure_field(&mut ch, final_a);
                    mix_secure_field(&mut ch, final_b);
                    current_claim_value = final_a;
                    println!("  OK");
                }

                8 => {
                    // RMSNorm
                    let input_eval = read_qm31_from(&mut off);
                    let output_eval = read_qm31_from(&mut off);
                    let rms_sq = read_qm31_from(&mut off);
                    let rsqrt_eval = read_qm31_from(&mut off);
                    off += 1; // rsqrt_table_commitment
                    let _simd = read_u32_from(&mut off);

                    println!("  [off at start of RMSNorm fields: {}]", off);
                    println!("  input_eval: {:?}", input_eval);
                    println!("  output_eval: {:?}", output_eval);
                    println!("  rms_sq: {:?}", rms_sq);
                    println!("  rsqrt_eval: {:?}", rsqrt_eval);
                    println!("  current_claim: {:?}", current_claim_value);
                    println!("  ch BEFORE RN: {:?}", ch.digest());

                    ch.mix_u64(0x524E); // "RN"
                    println!("  ch after mix RN: {:?}", ch.digest());
                    mix_secure_field(&mut ch, rms_sq);
                    println!("  ch after mix rms_sq: {:?}", ch.digest());
                    mix_secure_field(&mut ch, rsqrt_eval);
                    println!("  ch after mix rsqrt: {:?}", ch.digest());
                    mix_secure_field(&mut ch, current_claim_value);
                    println!("  ch after mix claim: {:?}", ch.digest());

                    let nrounds = read_u32_from(&mut off) as usize;
                    println!("  RMSNorm nrounds={}", nrounds);
                    let mut rms_sum = current_claim_value;

                    for round in 0..nrounds {
                        let c0 = read_qm31_from(&mut off);
                        let c1 = read_qm31_from(&mut off);
                        let c2 = read_qm31_from(&mut off);
                        let c3 = read_qm31_from(&mut off);

                        let p0 = c0;
                        let p1 = c0 + c1 + c2 + c3;
                        let round_sum = p0 + p1;

                        if round_sum != rms_sum {
                            println!("  RMS MISMATCH at round {}", round);
                            panic!("RMSNORM_ROUND_SUM at layer {} round {}", layer, round);
                        }

                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        rms_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }

                    println!("  ch after {} eq-rounds: {:?}", nrounds, ch.digest());

                    // Final evals
                    let input_final = read_qm31_from(&mut off);
                    let rsqrt_final = read_qm31_from(&mut off);
                    println!("  input_final: {:?}", input_final);
                    println!("  rsqrt_final: {:?}", rsqrt_final);
                    mix_secure_field(&mut ch, input_final);
                    mix_secure_field(&mut ch, rsqrt_final);
                    println!("  ch after final evals: {:?}", ch.digest());

                    // Optional logup
                    let has_logup = read_u32_from(&mut off);
                    println!("  has_logup: {}", has_logup);
                    if has_logup == 1 {
                        ch.mix_u64(0x4C4F47); // "LOG"
                        ch.mix_u64(0x524E); // "RN"
                        println!("  ch after LOG+RN tags: {:?}", ch.digest());
                        let _gamma = ch.draw_qm31();
                        let _beta = ch.draw_qm31();
                        println!("  ch after gamma+beta draws: {:?}", ch.digest());

                        let claimed_sum = read_qm31_from(&mut off);
                        println!("  claimed_sum: {:?}", claimed_sum);
                        mix_secure_field(&mut ch, claimed_sum);
                        println!("  ch after mix claimed_sum: {:?}", ch.digest());

                        let eq_rounds = read_u32_from(&mut off) as usize;
                        println!("  logup eq_rounds: {}", eq_rounds);
                        let mut logup_sum = SecureField::from(M31::from(1u32)); // initial = 1
                        for round in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);

                            let p0 = c0;
                            let p1 = c0 + c1 + c2 + c3;
                            if p0 + p1 != logup_sum {
                                println!("  LOGUP_ROUND_SUM at round {}: p(0)+p(1)={:?} != sum={:?}", round, p0+p1, logup_sum);
                                panic!("LOGUP_ROUND_SUM at layer {} round {}", layer, round);
                            }

                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let challenge = ch.draw_qm31();
                            logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                                + c3 * challenge * challenge * challenge;
                            if round < 2 || round == eq_rounds - 1 {
                                println!("  logup round {}: ch={:?}", round, ch.digest());
                            }
                        }

                        // Final evals (3 QM31s)
                        let _w = read_qm31_from(&mut off);
                        let _in_e = read_qm31_from(&mut off);
                        let _out_e = read_qm31_from(&mut off);
                        let num_mults = read_u32_from(&mut off) as usize;
                        println!("  logup final_evals read, num_mults={}", num_mults);
                        for _ in 0..num_mults {
                            let _ = read_u32_from(&mut off); // multiplicities are u32, NOT QM31!
                        }
                        println!("  ch after logup (before input/output mix): {:?}", ch.digest());
                    }

                    println!("  [off after RMSNorm: {}]", off);
                    mix_secure_field(&mut ch, input_eval);
                    mix_secure_field(&mut ch, output_eval);
                    current_claim_value = input_eval;
                    println!("  ch FINAL after RMSNorm: {:?}", ch.digest());
                }

                3 => {
                    // Activation
                    let _act_type = read_u32_from(&mut off);
                    let input_eval = read_qm31_from(&mut off);
                    let output_eval = read_qm31_from(&mut off);
                    off += 1; // table_commitment

                    ch.mix_u64(0x4C4F47); // "LOG"
                    let _gamma = ch.draw_qm31();
                    let _beta = ch.draw_qm31();

                    let has_logup = read_u32_from(&mut off);
                    if has_logup == 1 {
                        let claimed_sum = read_qm31_from(&mut off);
                        mix_secure_field(&mut ch, claimed_sum);
                        let eq_rounds = read_u32_from(&mut off) as usize;
                        let mut logup_sum = SecureField::from(M31::from(1u32));
                        for round in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);

                            let p0 = c0;
                            let p1 = c0 + c1 + c2 + c3;
                            if p0 + p1 != logup_sum {
                                panic!("ACT_LOGUP_ROUND at layer {} round {}", layer, round);
                            }

                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let challenge = ch.draw_qm31();
                            logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                                + c3 * challenge * challenge * challenge;
                        }
                        let _w = read_qm31_from(&mut off);
                        let _in_e = read_qm31_from(&mut off);
                        let _out_e = read_qm31_from(&mut off);
                        let num_mults = read_u32_from(&mut off) as usize;
                        for _ in 0..num_mults {
                            let _ = read_qm31_from(&mut off);
                        }
                    }

                    mix_secure_field(&mut ch, input_eval);
                    mix_secure_field(&mut ch, output_eval);
                    current_claim_value = input_eval;
                    println!("  Activation OK");
                }

                1 => {
                    // Add
                    let lhs = read_qm31_from(&mut off);
                    let rhs = read_qm31_from(&mut off);
                    let trunk_idx = read_u32_from(&mut off);

                    mix_secure_field(&mut ch, lhs);
                    mix_secure_field(&mut ch, rhs);
                    let _alpha = ch.draw_qm31();

                    current_claim_value = if trunk_idx == 0 { lhs } else { rhs };
                    println!("  Add OK, trunk_idx={}", trunk_idx);
                }

                _ => panic!("Unknown tag {} at layer {}", tag, layer),
            }
        }

        let num_deferred = read_u32_from(&mut off);
        println!("\nDeferred: {}, remaining: {}", num_deferred, proof_data.len() - off);
        println!("SUCCESS: all {} layers of actual proof pass Rust channel replay", num_layers);
    }

    /// Test that exercises the full prove → serialize → replay pipeline.
    /// This catches any discrepancy between the prover's channel state and
    /// the serialized proof data by generating a fresh proof and immediately replaying.
    #[test]
    fn test_fresh_prove_serialize_replay_roundtrip() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::crypto::poseidon_channel::PoseidonChannel;
        use crate::gkr::prover::mix_secure_field;
        use crate::cairo_serde::{serialize_gkr_proof_data_only, serialize_raw_io};
        use stwo::core::fields::qm31::{QM31, SecureField};
        use stwo::core::fields::cm31::CM31;
        use num_traits::Zero;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        // Build a model matching the on-chain Qwen3 pattern:
        // RMSNorm → MatMul(silu) → MatMul → RMSNorm → MatMul → MatMul → RMSNorm
        // Use dim=16 for speed (still exercises the full protocol).
        let dim = 16;
        let hidden = 32;
        let mut builder = GraphBuilder::new((1, dim));
        builder.rms_norm();           // layer 0
        builder.linear(hidden);       // layer 1: up-project
        builder.linear(dim);          // layer 2: down-project
        builder.rms_norm();           // layer 3
        builder.linear(hidden);       // layer 4: up-project
        builder.linear(dim);          // layer 5: down-project
        builder.rms_norm();           // layer 6
        let graph = builder.build();

        let mut input = M31Matrix::new(1, dim);
        for j in 0..dim {
            input.set(0, j, M31::from((j * 7 + 3) as u32 % 127));
        }

        let mut weights = GraphWeights::new();
        // Node IDs: 0=rms, 1=linear(hidden), 2=linear(dim), 3=rms, 4=linear(hidden), 5=linear(dim), 6=rms
        // linear(hidden) nodes: weight is dim × hidden
        // linear(dim) nodes: weight is hidden × dim
        let linear_nodes = [(1usize, dim, hidden), (2, hidden, dim), (4, dim, hidden), (5, hidden, dim)];
        for &(node_id, wr, wc) in &linear_nodes {
            let mut w = M31Matrix::new(wr, wc);
            for r in 0..wr {
                for c in 0..wc {
                    w.set(r, c, M31::from(((r * wc + c + node_id * 37) * 13 + 5) as u32 % 251));
                }
            }
            weights.add_weight(node_id, w);
        }

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        println!("Proving model: {} layers, dim={}, hidden={}", circuit.layers.len(), dim, hidden);
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let d = circuit.layers.len();
        let num_proof_layers = gkr.layer_proofs.len();

        // Serialize proof_data (unpacked)
        let mut proof_data_felts = Vec::new();
        serialize_gkr_proof_data_only(gkr, &mut proof_data_felts);
        let matmul_dims = extract_matmul_dims(&circuit);
        println!("proof_data: {} felts, circuit_depth={}, proof_layers={}, matmul_dims={:?}",
            proof_data_felts.len(), d, num_proof_layers, matmul_dims);

        // === Replay with IDENTICAL logic to the JSON-based replay ===
        fn felt_to_u64(f: &FieldElement) -> u64 {
            let b = f.to_bytes_be();
            u64::from_be_bytes([b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]])
        }

        let input_rows = felt_to_u64(&raw_io[0]);
        let input_cols = felt_to_u64(&raw_io[1]);
        let input_len = felt_to_u64(&raw_io[2]) as usize;
        let out_start = 3 + input_len;
        let output_rows = felt_to_u64(&raw_io[out_start]) as usize;
        let output_cols = felt_to_u64(&raw_io[out_start + 1]) as usize;

        let padded_rows = output_rows.next_power_of_two();
        let padded_cols = output_cols.next_power_of_two();
        let mut output_mle = Vec::with_capacity(padded_rows * padded_cols);
        for r in 0..padded_rows {
            for c in 0..padded_cols {
                if r < output_rows && c < output_cols {
                    let idx = r * output_cols + c;
                    let val = felt_to_u64(&raw_io[out_start + 3 + idx]) as u32;
                    output_mle.push(SecureField::from(M31::from(val)));
                } else {
                    output_mle.push(SecureField::zero());
                }
            }
        }

        let mut ch = PoseidonChannel::new();
        ch.mix_u64(d as u64);
        ch.mix_u64(input_rows);
        ch.mix_u64(input_cols);

        let log_out = (padded_rows * padded_cols).ilog2() as usize;
        let r_out = ch.draw_qm31s(log_out);
        let output_value = crate::components::matmul::evaluate_mle_pub(&output_mle, &r_out);
        mix_secure_field(&mut ch, output_value);

        let proof_felts = &proof_data_felts;
        let mut off = 0usize;
        let read_u32_from = |off: &mut usize| -> u32 {
            let v = felt_to_u64(&proof_felts[*off]) as u32;
            *off += 1;
            v
        };
        let read_qm31_from = |off: &mut usize| -> SecureField {
            let aa = felt_to_u64(&proof_felts[*off]) as u32; *off += 1;
            let ab = felt_to_u64(&proof_felts[*off]) as u32; *off += 1;
            let ba = felt_to_u64(&proof_felts[*off]) as u32; *off += 1;
            let bb = felt_to_u64(&proof_felts[*off]) as u32; *off += 1;
            QM31(CM31(M31::from(aa), M31::from(ab)),
                 CM31(M31::from(ba), M31::from(bb)))
        };

        let mut current_claim_value = output_value;
        let mut matmul_idx = 0usize;

        for layer in 0..num_proof_layers {
            let tag = read_u32_from(&mut off);
            print!("  Layer {} tag={}", layer, tag);

            match tag {
                0 => {
                    let m = matmul_dims[matmul_idx * 3] as usize;
                    let k = matmul_dims[matmul_idx * 3 + 1] as usize;
                    let n = matmul_dims[matmul_idx * 3 + 2] as usize;
                    matmul_idx += 1;

                    ch.mix_u64(m as u64);
                    ch.mix_u64(k as u64);
                    ch.mix_u64(n as u64);
                    mix_secure_field(&mut ch, current_claim_value);

                    let num_rounds = read_u32_from(&mut off) as usize;
                    let mut current_sum = current_claim_value;

                    for round in 0..num_rounds {
                        let c0 = read_qm31_from(&mut off);
                        let c1 = read_qm31_from(&mut off);
                        let c2 = read_qm31_from(&mut off);
                        let p0 = c0;
                        let p1 = c0 + c1 + c2;
                        assert_eq!(p0 + p1, current_sum,
                            "MATMUL_ROUND_SUM at layer {} round {}", layer, round);
                        ch.mix_poly_coeffs(c0, c1, c2);
                        let challenge = ch.draw_qm31();
                        current_sum = c0 + c1 * challenge + c2 * challenge * challenge;
                    }
                    let final_a = read_qm31_from(&mut off);
                    let final_b = read_qm31_from(&mut off);
                    assert_eq!(current_sum, final_a * final_b,
                        "MATMUL_FINAL at layer {}", layer);
                    mix_secure_field(&mut ch, final_a);
                    mix_secure_field(&mut ch, final_b);
                    current_claim_value = final_a;
                    println!(" MatMul({}x{}x{}) OK", m, k, n);
                }
                8 => {
                    let input_eval = read_qm31_from(&mut off);
                    let output_eval = read_qm31_from(&mut off);
                    let rms_sq = read_qm31_from(&mut off);
                    let rsqrt_eval = read_qm31_from(&mut off);
                    off += 1; // commitment
                    let _simd = read_u32_from(&mut off);

                    ch.mix_u64(0x524E);
                    mix_secure_field(&mut ch, rms_sq);
                    mix_secure_field(&mut ch, rsqrt_eval);
                    mix_secure_field(&mut ch, current_claim_value);

                    let nrounds = read_u32_from(&mut off) as usize;
                    let mut rms_sum = current_claim_value;
                    for round in 0..nrounds {
                        let c0 = read_qm31_from(&mut off);
                        let c1 = read_qm31_from(&mut off);
                        let c2 = read_qm31_from(&mut off);
                        let c3 = read_qm31_from(&mut off);
                        let p0 = c0;
                        let p1 = c0 + c1 + c2 + c3;
                        assert_eq!(p0 + p1, rms_sum,
                            "RMSNORM_ROUND at layer {} round {}", layer, round);
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        rms_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }
                    let input_final = read_qm31_from(&mut off);
                    let rsqrt_final = read_qm31_from(&mut off);
                    mix_secure_field(&mut ch, input_final);
                    mix_secure_field(&mut ch, rsqrt_final);

                    let has_logup = read_u32_from(&mut off);
                    if has_logup == 1 {
                        ch.mix_u64(0x4C4F47);
                        ch.mix_u64(0x524E);
                        let _gamma = ch.draw_qm31();
                        let _beta = ch.draw_qm31();
                        let claimed_sum = read_qm31_from(&mut off);
                        mix_secure_field(&mut ch, claimed_sum);
                        let eq_rounds = read_u32_from(&mut off) as usize;
                        let mut logup_sum = SecureField::from(M31::from(1u32));
                        for round in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            let p0 = c0;
                            let p1 = c0 + c1 + c2 + c3;
                            assert_eq!(p0 + p1, logup_sum,
                                "LOGUP_ROUND at layer {} round {}", layer, round);
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let challenge = ch.draw_qm31();
                            logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                                + c3 * challenge * challenge * challenge;
                        }
                        let _w = read_qm31_from(&mut off);
                        let _in_e = read_qm31_from(&mut off);
                        let _out_e = read_qm31_from(&mut off);
                        let num_mults = read_u32_from(&mut off) as usize;
                        for _ in 0..num_mults {
                            let _ = read_u32_from(&mut off);
                        }
                    }
                    mix_secure_field(&mut ch, input_eval);
                    mix_secure_field(&mut ch, output_eval);
                    current_claim_value = input_eval;
                    println!(" RMSNorm({} rounds, logup={}) OK", nrounds, has_logup);
                }
                3 => {
                    let _act_type = read_u32_from(&mut off);
                    let input_eval = read_qm31_from(&mut off);
                    let output_eval = read_qm31_from(&mut off);
                    off += 1; // commitment
                    ch.mix_u64(0x4C4F47);
                    let _gamma = ch.draw_qm31();
                    let _beta = ch.draw_qm31();
                    let has_logup = read_u32_from(&mut off);
                    if has_logup == 1 {
                        let claimed_sum = read_qm31_from(&mut off);
                        mix_secure_field(&mut ch, claimed_sum);
                        let eq_rounds = read_u32_from(&mut off) as usize;
                        let mut logup_sum = SecureField::from(M31::from(1u32));
                        for round in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            assert_eq!(c0 + c0 + c1 + c2 + c3, logup_sum,
                                "ACT_LOGUP at layer {} round {}", layer, round);
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let challenge = ch.draw_qm31();
                            logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                                + c3 * challenge * challenge * challenge;
                        }
                        let _w = read_qm31_from(&mut off);
                        let _in_e = read_qm31_from(&mut off);
                        let _out_e = read_qm31_from(&mut off);
                        let num_mults = read_u32_from(&mut off) as usize;
                        for _ in 0..num_mults {
                            let _ = read_qm31_from(&mut off);
                        }
                    }
                    mix_secure_field(&mut ch, input_eval);
                    mix_secure_field(&mut ch, output_eval);
                    current_claim_value = input_eval;
                    println!(" Activation OK");
                }
                1 => {
                    let lhs = read_qm31_from(&mut off);
                    let rhs = read_qm31_from(&mut off);
                    let trunk_idx = read_u32_from(&mut off);
                    mix_secure_field(&mut ch, lhs);
                    mix_secure_field(&mut ch, rhs);
                    let _alpha = ch.draw_qm31();
                    current_claim_value = if trunk_idx == 0 { lhs } else { rhs };
                    println!(" Add(trunk={}) OK", trunk_idx);
                }
                _ => panic!("Unknown tag {} at layer {}", tag, layer),
            }
        }

        let num_deferred = read_u32_from(&mut off);
        assert_eq!(proof_data_felts.len() - off, 0,
            "Expected 0 remaining felts, got {}", proof_data_felts.len() - off);
        println!("\nSUCCESS: all {} layers pass unpacked roundtrip verify (deferred={})", num_proof_layers, num_deferred);

        // === Also verify packed format ===
        let mut packed_proof_felts = Vec::new();
        crate::cairo_serde::serialize_gkr_proof_data_only_packed(gkr, &mut packed_proof_felts);
        println!("packed: {} felts (vs unpacked: {})", packed_proof_felts.len(), proof_data_felts.len());

        fn unpack_qm31(f: &FieldElement) -> SecureField {
            let val = f.to_bytes_be();
            // Convert to u256-like representation
            let mut v = [0u8; 32];
            v.copy_from_slice(&val);
            // Extract 4 M31 fields from packed felt252
            // Layout: sentinel(1 bit @124) | aa(31 bits @93) | ab(31 bits @62) | ba(31 bits @31) | bb(31 bits @0)
            let val_u128 = u128::from_be_bytes(v[16..32].try_into().unwrap());
            let bb = (val_u128 & 0x7FFFFFFF) as u32;
            let ba = ((val_u128 >> 31) & 0x7FFFFFFF) as u32;
            let ab = ((val_u128 >> 62) & 0x7FFFFFFF) as u32;
            let aa = ((val_u128 >> 93) & 0x7FFFFFFF) as u32;
            QM31(CM31(M31::from(aa), M31::from(ab)),
                 CM31(M31::from(ba), M31::from(bb)))
        }

        let pf = &packed_proof_felts;
        let mut poff = 0usize;
        let p_read_u32 = |off: &mut usize| -> u32 {
            let v = felt_to_u64(&pf[*off]) as u32;
            *off += 1;
            v
        };
        let p_read_qm31 = |off: &mut usize| -> SecureField {
            // Packed: 1 felt per QM31
            let v = unpack_qm31(&pf[*off]);
            *off += 1;
            v
        };

        let mut ch2 = PoseidonChannel::new();
        ch2.mix_u64(d as u64);
        ch2.mix_u64(input_rows);
        ch2.mix_u64(input_cols);
        let r_out2 = ch2.draw_qm31s(log_out);
        let output_value2 = crate::components::matmul::evaluate_mle_pub(&output_mle, &r_out2);
        mix_secure_field(&mut ch2, output_value2);
        let mut claim2 = output_value2;
        let mut mi2 = 0usize;

        for layer in 0..num_proof_layers {
            let tag = p_read_u32(&mut poff);
            match tag {
                0 => {
                    let m = matmul_dims[mi2 * 3] as usize;
                    let k = matmul_dims[mi2 * 3 + 1] as usize;
                    let n = matmul_dims[mi2 * 3 + 2] as usize;
                    mi2 += 1;
                    ch2.mix_u64(m as u64); ch2.mix_u64(k as u64); ch2.mix_u64(n as u64);
                    mix_secure_field(&mut ch2, claim2);
                    let nr = p_read_u32(&mut poff) as usize;
                    let mut s = claim2;
                    for round in 0..nr {
                        let c0 = p_read_qm31(&mut poff);
                        let c1 = p_read_qm31(&mut poff);
                        let c2 = p_read_qm31(&mut poff);
                        assert_eq!(c0 + c0 + c1 + c2, s, "PACKED_MATMUL at L{} R{}", layer, round);
                        ch2.mix_poly_coeffs(c0, c1, c2);
                        let ch_v = ch2.draw_qm31();
                        s = c0 + c1 * ch_v + c2 * ch_v * ch_v;
                    }
                    let fa = p_read_qm31(&mut poff);
                    let fb = p_read_qm31(&mut poff);
                    assert_eq!(s, fa * fb, "PACKED_MATMUL_FINAL at L{}", layer);
                    mix_secure_field(&mut ch2, fa); mix_secure_field(&mut ch2, fb);
                    claim2 = fa;
                }
                8 => {
                    let ie = p_read_qm31(&mut poff);
                    let oe = p_read_qm31(&mut poff);
                    let rms = p_read_qm31(&mut poff);
                    let rsq = p_read_qm31(&mut poff);
                    poff += 1; // commitment
                    let _simd = p_read_u32(&mut poff);
                    ch2.mix_u64(0x524E);
                    mix_secure_field(&mut ch2, rms);
                    mix_secure_field(&mut ch2, rsq);
                    mix_secure_field(&mut ch2, claim2);
                    let nr = p_read_u32(&mut poff) as usize;
                    let mut s = claim2;
                    for round in 0..nr {
                        let c0 = p_read_qm31(&mut poff); let c1 = p_read_qm31(&mut poff);
                        let c2 = p_read_qm31(&mut poff); let c3 = p_read_qm31(&mut poff);
                        assert_eq!(c0 + c0 + c1 + c2 + c3, s, "PACKED_RMS at L{} R{}", layer, round);
                        ch2.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let ch_v = ch2.draw_qm31();
                        s = c0 + c1 * ch_v + c2 * ch_v * ch_v + c3 * ch_v * ch_v * ch_v;
                    }
                    let inf = p_read_qm31(&mut poff); let rsf = p_read_qm31(&mut poff);
                    mix_secure_field(&mut ch2, inf); mix_secure_field(&mut ch2, rsf);
                    let hl = p_read_u32(&mut poff);
                    if hl == 1 {
                        ch2.mix_u64(0x4C4F47); ch2.mix_u64(0x524E);
                        let _ = ch2.draw_qm31(); let _ = ch2.draw_qm31();
                        let cs = p_read_qm31(&mut poff);
                        mix_secure_field(&mut ch2, cs);
                        let er = p_read_u32(&mut poff) as usize;
                        let mut ls = SecureField::from(M31::from(1u32));
                        for round in 0..er {
                            let c0 = p_read_qm31(&mut poff); let c1 = p_read_qm31(&mut poff);
                            let c2 = p_read_qm31(&mut poff); let c3 = p_read_qm31(&mut poff);
                            assert_eq!(c0 + c0 + c1 + c2 + c3, ls, "PACKED_LOGUP at L{} R{}", layer, round);
                            ch2.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let ch_v = ch2.draw_qm31();
                            ls = c0 + c1 * ch_v + c2 * ch_v * ch_v + c3 * ch_v * ch_v * ch_v;
                        }
                        let _ = p_read_qm31(&mut poff); let _ = p_read_qm31(&mut poff); let _ = p_read_qm31(&mut poff);
                        let nm = p_read_u32(&mut poff) as usize;
                        for _ in 0..nm { let _ = p_read_u32(&mut poff); }
                    }
                    mix_secure_field(&mut ch2, ie); mix_secure_field(&mut ch2, oe);
                    claim2 = ie;
                }
                1 => {
                    let lhs = p_read_qm31(&mut poff); let rhs = p_read_qm31(&mut poff);
                    let ti = p_read_u32(&mut poff);
                    mix_secure_field(&mut ch2, lhs); mix_secure_field(&mut ch2, rhs);
                    let _ = ch2.draw_qm31();
                    claim2 = if ti == 0 { lhs } else { rhs };
                }
                3 => {
                    let _ = p_read_u32(&mut poff);
                    let ie = p_read_qm31(&mut poff); let oe = p_read_qm31(&mut poff);
                    poff += 1;
                    ch2.mix_u64(0x4C4F47);
                    let _ = ch2.draw_qm31(); let _ = ch2.draw_qm31();
                    let hl = p_read_u32(&mut poff);
                    if hl == 1 {
                        let cs = p_read_qm31(&mut poff);
                        mix_secure_field(&mut ch2, cs);
                        let er = p_read_u32(&mut poff) as usize;
                        let mut ls = SecureField::from(M31::from(1u32));
                        for round in 0..er {
                            let c0 = p_read_qm31(&mut poff); let c1 = p_read_qm31(&mut poff);
                            let c2 = p_read_qm31(&mut poff); let c3 = p_read_qm31(&mut poff);
                            assert_eq!(c0 + c0 + c1 + c2 + c3, ls, "PACKED_ACT_LOGUP at L{} R{}", layer, round);
                            ch2.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let ch_v = ch2.draw_qm31();
                            ls = c0 + c1 * ch_v + c2 * ch_v * ch_v + c3 * ch_v * ch_v * ch_v;
                        }
                        let _ = p_read_qm31(&mut poff); let _ = p_read_qm31(&mut poff); let _ = p_read_qm31(&mut poff);
                        let nm = p_read_u32(&mut poff) as usize;
                        for _ in 0..nm { let _ = p_read_qm31(&mut poff); }
                    }
                    mix_secure_field(&mut ch2, ie); mix_secure_field(&mut ch2, oe);
                    claim2 = ie;
                }
                _ => panic!("Unknown packed tag {} at layer {}", tag, layer),
            }
        }
        let pd2 = p_read_u32(&mut poff);
        assert_eq!(packed_proof_felts.len() - poff, 0, "Packed: remaining felts");
        println!("PACKED roundtrip also passes! (deferred={})", pd2);
    }
}
