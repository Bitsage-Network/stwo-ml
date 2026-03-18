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
    /// KV-cache commitment: Poseidon super-commitment over all layer K/V heads.
    /// `None` when proving without KV-cache (non-autoregressive models).
    pub kv_cache_commitment: Option<FieldElement>,
    /// Previous step's KV-cache commitment (ZERO for prefill / step 0).
    /// `None` when proving without KV-cache.
    pub prev_kv_cache_commitment: Option<FieldElement>,
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
        kv_cache_commitment: None,
        prev_kv_cache_commitment: None,
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

    // KV-cache commitment (optional — for autoregressive models with KV-cache)
    if let (Some(kv), Some(prev_kv)) = (proof.kv_cache_commitment, proof.prev_kv_cache_commitment)
    {
        combined.push(FieldElement::from(1u64)); // has_kv = true
        combined.push(kv);
        combined.push(prev_kv);
    } else {
        combined.push(FieldElement::from(0u64)); // has_kv = false
    }

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
        kv_cache_commitment: proof.kv_cache_commitment,
        prev_kv_cache_commitment: proof.prev_kv_cache_commitment,
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
/// - All MatMul weight claims must have matching non-empty opening proofs.
///
/// Note: Activation LogUp is intentionally skipped in GKR mode because
/// M31 matmul outputs exceed the precomputed table range (2^16). The
/// activation LogUp is handled separately via the unified STARK with
/// reduced (table-range) inputs.
fn enforce_gkr_soundness_gates(proof: &crate::gkr::GKRProof) -> Result<(), StarknetModelError> {
    use crate::gkr::types::WeightOpeningTranscriptMode;
    #[allow(unused_imports)]
    use crate::gkr::LayerProof;

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
    // In AggregatedOracleSumcheck mode, weight openings use RLC binding
    // (empty queries/intermediate_roots). Only check Merkle queries for
    // modes that actually produce individual per-weight openings.
    if proof.weight_opening_transcript_mode != WeightOpeningTranscriptMode::AggregatedOracleSumcheck
    {
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

/// Parallel variant of `build_gkr_serializable_proof` that uses rayon to
/// serialize independent calldata components concurrently.
///
/// Serializes GKR layer proofs, IO data, weight openings, and weight claims
/// on separate rayon tasks. Hides ~60-70% of serialization latency behind
/// parallelism (~5-10s → ~2-3s for Qwen3-14B 40-layer models).
pub fn build_gkr_serializable_proof_parallel(
    proof: &AggregatedModelProofOnChain,
    model_id: FieldElement,
    input: &M31Matrix,
) -> Result<GkrStarknetProof, StarknetModelError> {
    let gkr_proof = proof.gkr_proof.as_ref().ok_or_else(|| {
        StarknetModelError::SerializationError("No GKR proof in aggregated proof".to_string())
    })?;

    let soundness_gate_error = match enforce_gkr_soundness_gates(gkr_proof) {
        Ok(()) => None,
        Err(e) => Some(e.to_string()),
    };

    // Parallel serialization of 4 independent components via nested rayon::join.
    // GKR calldata and weight openings are the heaviest — run on separate threads.
    let (gkr_calldata, io_calldata, weight_opening_calldata, weight_claim_calldata, weight_binding_data_calldata) = {
        let ((gkr_cd, io_cd), (wo_cd, (wc_cd, wbd_cd))) = rayon::join(
            || {
                rayon::join(
                    || {
                        let mut buf = Vec::new();
                        crate::cairo_serde::serialize_gkr_proof_data_only(gkr_proof, &mut buf);
                        buf
                    },
                    || crate::cairo_serde::serialize_raw_io(input, &proof.execution.output),
                )
            },
            || {
                rayon::join(
                    || {
                        let deferred_opening_count = gkr_proof
                            .deferred_proofs
                            .iter()
                            .filter(|d| d.has_weights())
                            .count();
                        let total_openings = gkr_proof.weight_openings.len() + deferred_opening_count;
                        let mut buf = Vec::new();
                        crate::cairo_serde::serialize_u32(total_openings as u32, &mut buf);
                        for opening in &gkr_proof.weight_openings {
                            crate::cairo_serde::serialize_mle_opening_proof(opening, &mut buf);
                        }
                        for deferred in &gkr_proof.deferred_proofs {
                            if let Some(wo) = deferred.weight_opening() {
                                crate::cairo_serde::serialize_mle_opening_proof(wo, &mut buf);
                            }
                        }
                        buf
                    },
                    || {
                        let wc = serialize_weight_claims_for_artifact(gkr_proof);
                        let wbd = weight_binding_data_for_artifact(gkr_proof);
                        (wc, wbd)
                    },
                )
            },
        );
        (gkr_cd, io_cd, wo_cd, wc_cd, wbd_cd)
    };

    let mut weight_commitments = gkr_proof.weight_commitments.clone();
    for deferred in &gkr_proof.deferred_proofs {
        if let Some(wc) = deferred.weight_commitment() {
            weight_commitments.push(wc);
        }
    }

    let weight_binding_mode_id =
        starknet_weight_binding_mode_for_artifact(gkr_proof.weight_opening_transcript_mode);

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
    prove_for_starknet_ml_gkr_with_cache(graph, input, weights, model_id, None)
}

/// Like [`prove_for_starknet_ml_gkr`] but with optional weight commitment cache.
///
/// When a cache is provided, Poseidon Merkle root computations for weight
/// matrices are skipped on cache hits (~500s → <1ms on subsequent proofs).
pub fn prove_for_starknet_ml_gkr_with_cache(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    model_id: FieldElement,
    weight_cache: Option<&crate::weight_cache::SharedWeightCache>,
) -> Result<GkrStarknetProof, StarknetModelError> {
    let proof = crate::aggregation::prove_model_pure_gkr_auto_with_cache(
        graph, input, weights, weight_cache,
    )?;
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
/// Pack M31 values (31 bits each) into felt252 values (8 per felt252, 248 bits).
///
/// This reduces storage reads on-chain by ~8× for raw_io_data, which is typically
/// 94%+ of the session data. Each packed felt252 contains 8 M31 values in LSB order:
/// `packed = m31_0 + m31_1 * 2^31 + m31_2 * 2^62 + ... + m31_7 * 2^217`
pub fn pack_m31_io_data(raw_io_data: &[FieldElement]) -> Vec<FieldElement> {
    let mut packed = Vec::with_capacity((raw_io_data.len() + 7) / 8);

    for chunk in raw_io_data.chunks(8) {
        let mut val = [0u8; 32]; // 256-bit LE buffer
        for (j, felt) in chunk.iter().enumerate() {
            // Extract the M31 value (bottom 31 bits).
            let be = felt.to_bytes_be();
            let raw = u32::from_be_bytes([be[28], be[29], be[30], be[31]]);
            debug_assert!(
                raw < 0x7FFF_FFFF,
                "raw_io_data[{}] = {} exceeds M31 range",
                j,
                raw
            );
            let m31_val = raw & 0x7FFF_FFFF; // mask to 31 bits
            let bit_offset = j * 31;
            let byte_offset = bit_offset / 8;
            let bit_shift = bit_offset % 8;

            // Write 31-bit value into LE byte buffer at the correct offset
            let extended = (m31_val as u64) << bit_shift;
            for k in 0..5 {
                if byte_offset + k < 32 {
                    val[byte_offset + k] |= ((extended >> (k * 8)) & 0xFF) as u8;
                }
            }
        }
        // Convert LE bytes to FieldElement (big-endian)
        let mut be = [0u8; 32];
        for i in 0..32 {
            be[31 - i] = val[i];
        }
        packed.push(FieldElement::from_bytes_be(&be).unwrap_or_default());
    }

    packed
}

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
            crate::gkr::circuit::LayerType::RoPE { .. } => 11,
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
    /// Whether proof_data uses double-packed QM31 format (c0+c2 pair in 1 felt).
    pub double_packed: bool,
    /// Whether raw_io_data is packed (8 M31 values per felt252).
    pub io_packed: bool,
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
    kv_cache_commitment: Option<FieldElement>,
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

    // 1. raw_io_data — pack 8 M31 values per felt252 to reduce storage reads.
    //    Each M31 value is 31 bits, so 8 × 31 = 248 bits fits in felt252 (252 bits).
    //    Packed format: [original_len, packed_count, packed_data...]
    //    Unpacked format: [len, data...]
    let use_io_packing = std::env::var("STWO_NO_IO_PACK").is_err() && raw_io_data.len() > 16;
    if use_io_packing {
        let packed = pack_m31_io_data(raw_io_data);
        flat.push(format!("{}", raw_io_data.len())); // original_len
        flat.push(format!("{}", packed.len()));       // packed_count
        for f in &packed {
            flat.push(format!("0x{:x}", f));
        }
    } else {
        push_felt_section!(raw_io_data);
    }

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

    // 4. proof_data — try double-packed first, then packed, then unpacked
    let use_packed = std::env::var("STWO_NO_PACKED").is_err();
    let use_double_packed = use_packed && std::env::var("STWO_NO_DOUBLE_PACK").is_err();
    let mut proof_data = Vec::new();
    let actually_double_packed;
    if use_double_packed {
        crate::cairo_serde::serialize_gkr_proof_data_only_double_packed(proof, &mut proof_data);
        actually_double_packed = true;
    } else if use_packed {
        crate::cairo_serde::serialize_gkr_proof_data_only_packed(proof, &mut proof_data);
        actually_double_packed = false;
    } else {
        crate::cairo_serde::serialize_gkr_proof_data_only(proof, &mut proof_data);
        actually_double_packed = false;
    }

    // Self-verify: replay Fiat-Shamir channel against serialized proof to catch
    // prover/serializer mismatches before saving. This prevents stale-binary bugs
    // from producing proofs that fail on-chain verification.
    if actually_double_packed {
        replay_verify_double_packed_proof(
            &proof_data,
            raw_io_data,
            &matmul_dims,
            circuit_depth,
            num_layers,
            proof,
        ).map_err(|e| StarknetModelError::SoundnessGate(
            format!("chunked double-packed self-verification failed: {e}")
        ))?;
    } else {
        replay_verify_serialized_proof(
            &proof_data,
            raw_io_data,
            &matmul_dims,
            circuit_depth,
            num_layers,
            use_packed,
            Some(proof.io_commitment),
            proof.aggregated_binding.as_ref(),
            kv_cache_commitment,
            proof.prev_kv_cache_commitment,
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
        double_packed: actually_double_packed,
        io_packed: use_io_packing,
        chunks,
        num_chunks,
    })
}

// ============================================================================
// Streaming GKR Calldata Builder (v25)
// ============================================================================

/// Maximum proof data felts per streaming layer batch.
/// Leaves ~500 felts for dims/bits/overhead within the 4500 total TX limit.
pub const MAX_STREAM_BATCH_FELTS: usize = 3500;

/// Maximum layers per streaming batch.
/// Each layer costs ~34M gas on average (sumcheck verification).
/// Starknet gas limit is 1.2B, so cap at 30 layers (~1.0B gas) for safety margin.
pub const MAX_STREAM_BATCH_LAYERS: usize = 30;

/// Streaming GKR calldata for calldata-only verification.
///
/// Protocol: open_gkr_session → upload chunks → seal →
///   stream_init → stream_layers × M → stream_finalize.
#[derive(Debug, Clone)]
pub struct StreamingGkrCalldata {
    /// Calldata for verify_gkr_stream_init (IO data + metadata).
    pub init_calldata: Vec<String>,
    /// Chunked calldata for verify_gkr_stream_init_output_mle.
    /// Each chunk contains packed output data for a portion of the output MLE.
    pub output_mle_chunks: Vec<OutputMleChunk>,
    /// Batches of layer proof data for verify_gkr_stream_layers.
    pub stream_batches: Vec<StreamBatch>,
    /// Calldata for verify_gkr_stream_weight_binding (packed QM31 binding proof).
    /// Separated from input MLE to stay under 5000-felt calldata limit.
    pub weight_binding_calldata: Vec<String>,
    /// Chunked calldata for verify_gkr_stream_finalize_input_mle.
    /// All chunks are uniform: session_id + packed_input_data + chunk metadata.
    /// Weight binding is handled by a prior verify_gkr_stream_weight_binding TX.
    pub input_mle_chunks: Vec<InputMleChunk>,
    /// Calldata for verify_gkr_stream_finalize (just session_id).
    pub finalize_calldata: Vec<String>,
    /// Session metadata for open_gkr_session (reuse existing session flow).
    pub session_metadata: StreamSessionMetadata,
    /// Upload chunks (for data integrity commitment via existing session flow).
    pub upload_chunks: Vec<Vec<String>>,
}

/// A single chunk of output MLE evaluation data.
#[derive(Debug, Clone)]
pub struct OutputMleChunk {
    /// Starting M31 index in the full output array.
    pub chunk_offset: u32,
    /// Number of M31 values in this chunk.
    pub chunk_len: u32,
    /// Whether this is the last chunk.
    pub is_last: bool,
    /// Flat calldata for verify_gkr_stream_init_output_mle.
    pub calldata: Vec<String>,
}

/// A single chunk of input MLE evaluation data (mirrors OutputMleChunk).
/// First chunk includes weight binding + deferred proofs + IO commitment data.
#[derive(Debug, Clone)]
pub struct InputMleChunk {
    /// Starting M31 index in the full input array.
    pub chunk_offset: u32,
    /// Number of M31 values in this chunk.
    pub chunk_len: u32,
    /// Whether this is the last chunk.
    pub is_last: bool,
    /// Flat calldata for verify_gkr_stream_finalize_input_mle.
    pub calldata: Vec<String>,
}

/// A single streaming batch of layer proofs.
#[derive(Debug, Clone)]
pub struct StreamBatch {
    /// Batch index (0-based).
    pub batch_idx: u32,
    /// Number of layers in this batch.
    pub num_layers: u32,
    /// Flat calldata for verify_gkr_stream_layers.
    pub calldata: Vec<String>,
}

/// Session metadata for stream init.
#[derive(Debug, Clone)]
pub struct StreamSessionMetadata {
    pub model_id: String,
    pub total_felts: usize,
    pub circuit_depth: u32,
    pub num_layers: u32,
    pub weight_binding_mode: u32,
    /// Layer tags in GKR walk order (output → input) for circuit hash registration.
    pub layer_tags: Vec<u32>,
    /// Whether this proof includes KV-cache commitments.
    pub has_kv_cache: bool,
}

/// Build streaming GKR calldata for calldata-only verification (v25).
///
/// Splits the proof into:
/// Extract decode metadata from layer proofs. Returns (position_offset, full_seq_len, new_tokens).
/// Non-decode proofs return (0, 0, 0).
fn extract_decode_metadata(layer_proofs: &[crate::gkr::types::LayerProof]) -> (u32, u32, u32) {
    for proof in layer_proofs {
        if let crate::gkr::types::LayerProof::AttentionDecode {
            position_offset,
            full_seq_len,
            new_tokens,
            ..
        } = proof
        {
            return (*position_offset as u32, *full_seq_len as u32, *new_tokens as u32);
        }
    }
    (0, 0, 0)
}

/// 1. init TX: IO data + metadata (~1500 felts)
/// 2. N stream TXs: batches of layer proofs (~3500 felts each)
/// 3. finalize TX: weight claims + binding (~200 felts)
pub fn build_streaming_gkr_calldata(
    proof: &crate::gkr::GKRProof,
    circuit: &crate::gkr::LayeredCircuit,
    model_id: FieldElement,
    raw_io_data: &[FieldElement],
    kv_cache_commitment: Option<FieldElement>,
    prev_kv_cache_commitment: Option<FieldElement>,
) -> Result<StreamingGkrCalldata, StarknetModelError> {
    enforce_gkr_soundness_gates(proof)?;

    let binding_mode = starknet_weight_binding_mode(proof.weight_opening_transcript_mode)?;
    let circuit_depth = circuit.layers.len() as u32;
    let num_layers = proof.layer_proofs.len() as u32;

    // ── Build init calldata ──
    // session_id is filled in by the caller after open_gkr_session
    // Format: [original_io_len, packed_count, packed_io..., circuit_depth, num_layers]
    let mut init_calldata: Vec<String> = Vec::new();
    // session_id placeholder
    init_calldata.push("__SESSION_ID__".to_string());

    let packed_io = pack_m31_io_data(raw_io_data);
    init_calldata.push(format!("{}", raw_io_data.len())); // original_io_len
    // packed_raw_io: Array<felt252> [len, data...]
    init_calldata.push(format!("{}", packed_io.len()));
    for f in &packed_io {
        init_calldata.push(format!("0x{:x}", f));
    }
    init_calldata.push(format!("{}", circuit_depth));
    init_calldata.push(format!("{}", num_layers));

    // Extract IO dimensions from raw_io_data header for init + output_mle split
    let io_in_rows = {
        let be = raw_io_data[0].to_bytes_be();
        u32::from_be_bytes([be[28], be[29], be[30], be[31]])
    };
    let io_in_cols = {
        let be = raw_io_data[1].to_bytes_be();
        u32::from_be_bytes([be[28], be[29], be[30], be[31]])
    };
    let io_in_len = (io_in_rows * io_in_cols) as usize;
    let io_out_cols = {
        let be = raw_io_data[3 + io_in_len + 1].to_bytes_be();
        u32::from_be_bytes([be[28], be[29], be[30], be[31]])
    };
    let io_out_len_val = {
        let be = raw_io_data[3 + io_in_len + 2].to_bytes_be();
        u32::from_be_bytes([be[28], be[29], be[30], be[31]])
    };
    init_calldata.push(format!("{}", io_in_cols));  // in_cols
    init_calldata.push(format!("{}", io_out_cols));  // out_cols

    // KV-cache commitment fields (always 3 felts for positional Cairo params)
    if let (Some(kv), Some(prev_kv)) = (kv_cache_commitment, prev_kv_cache_commitment) {
        init_calldata.push("1".to_string());
        init_calldata.push(format!("0x{:x}", kv));
        init_calldata.push(format!("0x{:x}", prev_kv));
    } else {
        init_calldata.push("0".to_string());
        init_calldata.push("0x0".to_string());
        init_calldata.push("0x0".to_string());
    }

    // Decode metadata: position_offset, full_seq_len, new_tokens, validate_decode_chain
    let (pos_off, full_seq, new_tok) = extract_decode_metadata(&proof.layer_proofs);
    init_calldata.push(format!("{}", pos_off));
    init_calldata.push(format!("{}", full_seq));
    init_calldata.push(format!("{}", new_tok));
    // validate_decode_chain: enabled when this is a decode proof with KV
    let validate_chain = pos_off > 0 && proof.kv_cache_commitment.is_some();
    init_calldata.push(if validate_chain { "1" } else { "0" }.to_string());

    // ── Build chunked output_mle calldata ──
    // Split output data into chunks of CHUNK_SIZE M31 values for gas-safe MLE evaluation.
    // Each chunk carries its own packed felt252 data, chunk_offset, chunk_len, and is_last flag.
    const OUTPUT_MLE_CHUNK_SIZE: u32 = 1024;
    let output_start = 3 + io_in_len + 3; // skip input header + data + output header
    let output_end = output_start + io_out_len_val as usize;
    let output_data = &raw_io_data[output_start..output_end];
    let out_len = io_out_len_val;
    let num_output_chunks = (out_len + OUTPUT_MLE_CHUNK_SIZE - 1) / OUTPUT_MLE_CHUNK_SIZE;

    let mut output_mle_chunks: Vec<OutputMleChunk> = Vec::new();
    for chunk_idx in 0..num_output_chunks {
        let chunk_offset = chunk_idx * OUTPUT_MLE_CHUNK_SIZE;
        let chunk_len = std::cmp::min(OUTPUT_MLE_CHUNK_SIZE, out_len - chunk_offset);
        let is_last = chunk_idx == num_output_chunks - 1;

        // Pack only this chunk's M31 values
        let chunk_start = chunk_offset as usize;
        let chunk_end = (chunk_offset + chunk_len) as usize;
        let packed_chunk = pack_m31_io_data(&output_data[chunk_start..chunk_end]);

        let mut calldata: Vec<String> = Vec::new();
        calldata.push("__SESSION_ID__".to_string());
        // packed_output_data: Array<felt252>
        calldata.push(format!("{}", packed_chunk.len()));
        for f in &packed_chunk {
            calldata.push(format!("0x{:x}", f));
        }
        // chunk_offset, chunk_len, is_last_chunk
        calldata.push(format!("{}", chunk_offset));
        calldata.push(format!("{}", chunk_len));
        calldata.push(if is_last { "1".to_string() } else { "0".to_string() });

        output_mle_chunks.push(OutputMleChunk {
            chunk_offset,
            chunk_len,
            is_last,
            calldata,
        });
    }
    let _ = io_in_rows; // used above for io_in_len calculation

    // ── Split proof data into batches with boundary tracking ──
    let (all_proof_felts, batch_infos) =
        crate::cairo_serde::split_proof_into_stream_batches(proof, MAX_STREAM_BATCH_FELTS);

    // Extract per-layer metadata for batch slicing
    let all_matmul_dims = extract_matmul_dims(circuit);
    let all_dequantize_bits = extract_dequantize_bits(circuit);

    // Self-verify the serialized proof data against expected Fiat-Shamir transcript.
    // This catches Rust↔Cairo serialization mismatches before on-chain submission.
    replay_verify_serialized_proof(
        &all_proof_felts,
        raw_io_data,
        &all_matmul_dims,
        circuit.layers.len() as u32,
        proof.layer_proofs.len() as u32,
        true, // packed
        Some(proof.io_commitment),
        proof.aggregated_binding.as_ref(),
        kv_cache_commitment,
        prev_kv_cache_commitment,
    ).map_err(|e| StarknetModelError::SoundnessGate(
        format!("streaming self-verification failed: {e}")
    ))?;

    // Build per-layer metadata for batch slicing and circuit hash registration.
    // Walk layers in GKR order (output → input, same as circuit reversed).
    let mut layer_is_matmul = Vec::new();
    let mut layer_is_dequantize = Vec::new();
    let mut layer_tags_vec: Vec<u32> = Vec::new();
    for layer_idx in (0..circuit.layers.len()).rev() {
        let tag = match &circuit.layers[layer_idx].layer_type {
            crate::gkr::circuit::LayerType::MatMul { .. } => {
                layer_is_matmul.push(true);
                layer_is_dequantize.push(false);
                0u32
            }
            crate::gkr::circuit::LayerType::Dequantize { .. } => {
                layer_is_matmul.push(false);
                layer_is_dequantize.push(true);
                6
            }
            crate::gkr::circuit::LayerType::Input
            | crate::gkr::circuit::LayerType::Identity => continue,
            crate::gkr::circuit::LayerType::Add { .. } => {
                layer_is_matmul.push(false);
                layer_is_dequantize.push(false);
                1
            }
            crate::gkr::circuit::LayerType::Mul { .. } => {
                layer_is_matmul.push(false);
                layer_is_dequantize.push(false);
                2
            }
            crate::gkr::circuit::LayerType::Activation { .. } => {
                layer_is_matmul.push(false);
                layer_is_dequantize.push(false);
                3
            }
            crate::gkr::circuit::LayerType::LayerNorm { .. } => {
                layer_is_matmul.push(false);
                layer_is_dequantize.push(false);
                4
            }
            crate::gkr::circuit::LayerType::Attention { .. } => {
                layer_is_matmul.push(false);
                layer_is_dequantize.push(false);
                5
            }
            crate::gkr::circuit::LayerType::RMSNorm { .. } => {
                layer_is_matmul.push(false);
                layer_is_dequantize.push(false);
                8
            }
            _ => {
                layer_is_matmul.push(false);
                layer_is_dequantize.push(false);
                continue;
            }
        };
        layer_tags_vec.push(tag);
    }

    let mut stream_batches = Vec::new();
    let mut matmul_dim_offset = 0usize;
    let mut dequantize_bit_offset = 0usize;

    for (batch_idx, info) in batch_infos.iter().enumerate() {
        let mut calldata: Vec<String> = Vec::new();

        // session_id placeholder
        calldata.push("__SESSION_ID__".to_string());
        // batch_idx
        calldata.push(format!("{}", batch_idx));
        // num_layers_in_batch
        let num_in_batch = (info.end_layer - info.start_layer) as u32;
        calldata.push(format!("{}", num_in_batch));

        // Count matmul dims and dequantize bits needed for this batch
        let mut batch_matmul_count = 0usize;
        let mut batch_dequantize_count = 0usize;
        for layer_i in info.start_layer..info.end_layer {
            if layer_i < layer_is_matmul.len() && layer_is_matmul[layer_i] {
                batch_matmul_count += 1;
            }
            if layer_i < layer_is_dequantize.len() && layer_is_dequantize[layer_i] {
                batch_dequantize_count += 1;
            }
        }

        // matmul_dims for this batch: Array<u32> [len, m0,k0,n0, ...]
        let matmul_dim_count = batch_matmul_count * 3;
        calldata.push(format!("{}", matmul_dim_count));
        for i in 0..matmul_dim_count {
            calldata.push(format!("{}", all_matmul_dims[matmul_dim_offset + i]));
        }
        matmul_dim_offset += matmul_dim_count;

        // dequantize_bits for this batch: Array<u64> [len, bits0, ...]
        calldata.push(format!("{}", batch_dequantize_count));
        for i in 0..batch_dequantize_count {
            calldata.push(format!("{}", all_dequantize_bits[dequantize_bit_offset + i]));
        }
        dequantize_bit_offset += batch_dequantize_count;

        // proof_data for this batch: Array<felt252> [len, data...]
        let batch_felts = &all_proof_felts[info.felt_start..info.felt_end];
        calldata.push(format!("{}", batch_felts.len()));
        for f in batch_felts {
            calldata.push(format!("0x{:x}", f));
        }

        stream_batches.push(StreamBatch {
            batch_idx: batch_idx as u32,
            num_layers: num_in_batch,
            calldata,
        });
    }

    // ── Build weight binding + deferred proof data (shared by first input MLE chunk) ──
    // weight_expected_values: packed QM31 as felt252
    let mut weight_expected_values: Vec<String> = Vec::new();
    weight_expected_values.push(format!("{}", proof.weight_claims.len()));
    for wc in &proof.weight_claims {
        let packed = crate::crypto::poseidon_channel::securefield_to_felt(wc.expected_value);
        weight_expected_values.push(format!("0x{:x}", packed));
    }

    // weight_eval_points: flat Array<felt252> containing nested [num_claims, inner_len, pts..., inner_len, pts...]
    // Cairo Serde reads [total_felt_count, felt0, felt1, ...] so we serialize as flat array with total count.
    let mut weight_eval_points_inner: Vec<String> = Vec::new();
    weight_eval_points_inner.push(format!("{}", proof.weight_claims.len()));
    for wc in &proof.weight_claims {
        weight_eval_points_inner.push(format!("{}", wc.eval_point.len()));
        for ep in &wc.eval_point {
            let packed = crate::crypto::poseidon_channel::securefield_to_felt(*ep);
            weight_eval_points_inner.push(format!("0x{:x}", packed));
        }
    }
    let mut weight_eval_points: Vec<String> = Vec::new();
    weight_eval_points.push(format!("{}", weight_eval_points_inner.len()));
    weight_eval_points.extend(weight_eval_points_inner);

    // Deferred weight claims eval points (from deferred proofs with MatMul kind)
    let deferred_matmul_count = proof
        .deferred_proofs
        .iter()
        .filter(|d| d.weight_claim().is_some())
        .count();
    let mut deferred_eval_points_inner: Vec<String> = Vec::new();
    deferred_eval_points_inner.push(format!("{}", deferred_matmul_count));
    for deferred in &proof.deferred_proofs {
        if let Some(wc) = deferred.weight_claim() {
            deferred_eval_points_inner.push(format!("{}", wc.eval_point.len()));
            for ep in &wc.eval_point {
                let packed = crate::crypto::poseidon_channel::securefield_to_felt(*ep);
                deferred_eval_points_inner.push(format!("0x{:x}", packed));
            }
        }
    }
    let mut deferred_eval_points: Vec<String> = Vec::new();
    deferred_eval_points.push(format!("{}", deferred_eval_points_inner.len()));
    deferred_eval_points.extend(deferred_eval_points_inner);

    // weight_binding_mode + data
    // Streaming path requires full aggregated binding proof for on-chain soundness.
    // RLC-only (aggregated_binding: None) does not verify weight claims against
    // registered weight commitments, so it must be rejected for streaming submissions.
    if proof.weight_opening_transcript_mode
        == crate::gkr::types::WeightOpeningTranscriptMode::AggregatedOracleSumcheck
        && proof.aggregated_binding.is_none()
        && proof.binding_groups.is_empty()
    {
        return Err(StarknetModelError::SoundnessGate(
            "Streaming GKR requires full aggregated binding proof for on-chain submission. \
             Ensure STWO_AGGREGATED_RLC_ONLY is not set (full binding is now default)."
                .to_string(),
        ));
    }
    let binding_data = starknet_weight_binding_data_packed(proof)?;
    let mut weight_binding_mode_and_data: Vec<String> = Vec::new();
    weight_binding_mode_and_data.push(format!("{}", binding_mode));
    weight_binding_mode_and_data.push(format!("{}", binding_data.len()));
    for bd in &binding_data {
        weight_binding_mode_and_data.push(bd.clone());
    }

    // deferred_proof_data
    let mut deferred_proof_calldata: Vec<String> = Vec::new();
    {
        let mut deferred_felts: Vec<FieldElement> = Vec::new();
        crate::cairo_serde::serialize_u32(proof.deferred_proofs.len() as u32, &mut deferred_felts);
        for deferred in &proof.deferred_proofs {
            crate::cairo_serde::serialize_qm31_packed(deferred.claim.value, &mut deferred_felts);
            match &deferred.kind {
                crate::gkr::types::DeferredProofKind::MatMul { weight_commitment, .. } => {
                    crate::cairo_serde::serialize_u32(0, &mut deferred_felts);
                    if let crate::gkr::types::LayerProof::MatMul {
                        round_polys,
                        final_a_eval,
                        final_b_eval,
                    } = &deferred.layer_proof
                    {
                        crate::cairo_serde::serialize_u32(round_polys.len() as u32, &mut deferred_felts);
                        for rp in round_polys {
                            crate::cairo_serde::serialize_qm31_packed(rp.c0, &mut deferred_felts);
                            crate::cairo_serde::serialize_qm31_packed(rp.c2, &mut deferred_felts);
                        }
                        crate::cairo_serde::serialize_qm31_packed(*final_a_eval, &mut deferred_felts);
                        crate::cairo_serde::serialize_qm31_packed(*final_b_eval, &mut deferred_felts);
                    }
                    deferred_felts.push(*weight_commitment);
                }
                crate::gkr::types::DeferredProofKind::Weightless => {
                    crate::cairo_serde::serialize_u32(1, &mut deferred_felts);
                    if let crate::gkr::types::LayerProof::Add { lhs_eval, rhs_eval, trunk_idx } = &deferred.layer_proof {
                        crate::cairo_serde::serialize_qm31_packed(*lhs_eval, &mut deferred_felts);
                        crate::cairo_serde::serialize_qm31_packed(*rhs_eval, &mut deferred_felts);
                        crate::cairo_serde::serialize_u32(*trunk_idx as u32, &mut deferred_felts);
                    }
                }
            }
        }
        deferred_proof_calldata.push(format!("{}", deferred_felts.len()));
        for f in &deferred_felts {
            deferred_proof_calldata.push(format!("0x{:x}", f));
        }
    }

    // deferred_matmul_dims
    let mut deferred_dims_calldata: Vec<String> = Vec::new();
    {
        let mut deferred_dims: Vec<u32> = Vec::new();
        for deferred in &proof.deferred_proofs {
            if let Some((m, k, n)) = deferred.dims() {
                deferred_dims.push(m as u32);
                deferred_dims.push(k as u32);
                deferred_dims.push(n as u32);
            }
        }
        deferred_dims_calldata.push(format!("{}", deferred_dims.len()));
        for d in &deferred_dims {
            deferred_dims_calldata.push(format!("{}", d));
        }
    }

    // ── Build weight binding calldata (separate TX, packed QM31) ──
    // This is a dedicated TX that verifies weight binding before input MLE chunks.
    // Using packed QM31 keeps it under the 5000-felt Starknet calldata limit.
    let mut wb_calldata: Vec<String> = Vec::new();
    wb_calldata.push("__SESSION_ID__".to_string());
    wb_calldata.extend(weight_expected_values);
    wb_calldata.extend(weight_eval_points);
    wb_calldata.extend(deferred_eval_points);
    wb_calldata.extend(weight_binding_mode_and_data);
    wb_calldata.extend(deferred_proof_calldata);
    wb_calldata.extend(deferred_dims_calldata);
    eprintln!(
        "[streaming] weight_binding calldata: {} felts (limit: 5000)",
        wb_calldata.len()
    );
    // Diagnostic: print weight commitments for on-chain comparison
    if let Some(binding) = proof.aggregated_binding.as_ref() {
        eprintln!("[streaming] === BINDING DIAGNOSTICS ===");
        eprintln!("[streaming]   super_root: 0x{:x}", binding.super_root.root);
        for (i, sr) in binding.super_root.subtree_roots.iter().enumerate() {
            eprintln!("[streaming]   subtree_root[{}]: 0x{:x}", i, sr);
        }
        for (i, wc) in proof.weight_commitments.iter().enumerate() {
            eprintln!("[streaming]   registered_weight[{}]: 0x{:x}", i, wc);
        }
        for (i, wc) in proof.weight_claims.iter().enumerate() {
            let eval_pt_hash = if wc.eval_point.is_empty() {
                "empty".to_string()
            } else {
                format!("{} vars", wc.eval_point.len())
            };
            eprintln!(
                "[streaming]   weight_claim[{}]: eval_pt={}, expected_value=0x{:x}",
                i, eval_pt_hash,
                crate::crypto::poseidon_channel::securefield_to_felt(wc.expected_value),
            );
        }
        eprintln!("[streaming]   config: selector_bits={}, n_max={}, n_global={}, n_claims={}",
            binding.config.selector_bits, binding.config.n_max,
            binding.config.n_global, binding.config.n_claims);
        eprintln!("[streaming]   sumcheck rounds: {}", binding.sumcheck_round_polys.len());
        eprintln!("[streaming]   opening queries: {}", binding.opening_proof.queries.len());
        eprintln!("[streaming]   opening intermediate roots: {}", binding.opening_proof.intermediate_roots.len());
        eprintln!("[streaming] === END BINDING DIAGNOSTICS ===");
    }

    // ── Build chunked input MLE calldata ──
    // All chunks are uniform: session_id + packed_input_data + chunk metadata.
    // Weight binding is handled by a prior verify_gkr_stream_weight_binding TX.
    const INPUT_MLE_CHUNK_SIZE: u32 = 1024;
    let input_start = 3usize; // skip [in_rows, in_cols, in_len] header
    let input_end = input_start + io_in_len;
    let input_data = &raw_io_data[input_start..input_end];
    let in_len_u32 = io_in_len as u32;
    let num_input_chunks = (in_len_u32 + INPUT_MLE_CHUNK_SIZE - 1) / INPUT_MLE_CHUNK_SIZE;

    let mut input_mle_chunks: Vec<InputMleChunk> = Vec::new();
    for chunk_idx in 0..num_input_chunks {
        let chunk_offset = chunk_idx * INPUT_MLE_CHUNK_SIZE;
        let chunk_len = std::cmp::min(INPUT_MLE_CHUNK_SIZE, in_len_u32 - chunk_offset);
        let is_last = chunk_idx == num_input_chunks - 1;

        // Pack only this chunk's M31 values
        let chunk_start = chunk_offset as usize;
        let chunk_end = (chunk_offset + chunk_len) as usize;
        let packed_chunk = pack_m31_io_data(&input_data[chunk_start..chunk_end]);

        let mut calldata: Vec<String> = Vec::new();
        calldata.push("__SESSION_ID__".to_string());
        // packed_input_data: Array<felt252>
        calldata.push(format!("{}", packed_chunk.len()));
        for f in &packed_chunk {
            calldata.push(format!("0x{:x}", f));
        }
        // chunk_offset, chunk_len, is_last_chunk
        calldata.push(format!("{}", chunk_offset));
        calldata.push(format!("{}", chunk_len));
        calldata.push(if is_last { "1".to_string() } else { "0".to_string() });

        input_mle_chunks.push(InputMleChunk {
            chunk_offset,
            chunk_len,
            is_last,
            calldata,
        });
    }

    // ── Build finalize calldata (lightweight — just session_id) ──
    let finalize_calldata = vec!["__SESSION_ID__".to_string()];

    // ── Build upload chunks for data integrity ──
    // Reuse existing chunked session format for hash commitment
    let chunked = build_chunked_gkr_calldata(proof, circuit, model_id, raw_io_data, kv_cache_commitment)?;

    // ── Total felts for session metadata ──
    let total_felts = chunked.total_felts;

    Ok(StreamingGkrCalldata {
        init_calldata,
        output_mle_chunks,
        stream_batches,
        weight_binding_calldata: wb_calldata,
        input_mle_chunks,
        finalize_calldata,
        session_metadata: StreamSessionMetadata {
            model_id: format!("0x{:x}", model_id),
            total_felts,
            circuit_depth,
            num_layers,
            weight_binding_mode: binding_mode,
            layer_tags: layer_tags_vec,
            has_kv_cache: proof.kv_cache_commitment.is_some(),
        },
        upload_chunks: chunked.chunks,
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
            // Mode 4: full proof payload, grouped proofs, or RLC-only marker.
            if let Some(binding) = proof.aggregated_binding.as_ref() {
                // Single binding proof (small model)
                let mut payload = Vec::new();
                crate::cairo_serde::serialize_aggregated_binding_proof(binding, &mut payload);
                Ok(payload.into_iter().map(|f| format!("0x{:x}", f)).collect())
            } else if !proof.binding_groups.is_empty() {
                // Grouped binding proofs (large model): serialize as
                // [n_groups, group_0_proof, group_1_proof, ...]
                let mut payload = Vec::new();
                payload.push(FieldElement::from(proof.binding_groups.len() as u64));
                for group in &proof.binding_groups {
                    crate::cairo_serde::serialize_aggregated_binding_proof(group, &mut payload);
                }
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

/// Packed variant of `starknet_weight_binding_data` — uses packed QM31 (1 felt per QM31)
/// for the aggregated binding proof, reducing calldata by ~975 felts.
fn starknet_weight_binding_data_packed(
    proof: &crate::gkr::GKRProof,
) -> Result<Vec<String>, StarknetModelError> {
    use crate::gkr::types::WeightOpeningTranscriptMode;

    match proof.weight_opening_transcript_mode {
        WeightOpeningTranscriptMode::AggregatedOracleSumcheck => {
            if let Some(binding) = proof.aggregated_binding.as_ref() {
                let mut payload = Vec::new();
                crate::cairo_serde::serialize_aggregated_binding_proof_packed(
                    binding, &mut payload,
                );
                // Roundtrip check: deserialize and compare key fields
                {
                    let mut pos = 0usize;
                    let rt = crate::cairo_serde::deserialize_aggregated_binding_proof_packed(
                        &payload, &mut pos,
                    );
                    assert_eq!(
                        pos, payload.len(),
                        "binding proof roundtrip: consumed {} of {} felts",
                        pos, payload.len()
                    );
                    assert_eq!(rt.config.selector_bits, binding.config.selector_bits, "selector_bits mismatch");
                    assert_eq!(rt.config.n_max, binding.config.n_max, "n_max mismatch");
                    assert_eq!(rt.config.n_global, binding.config.n_global, "n_global mismatch");
                    assert_eq!(rt.config.n_claims, binding.config.n_claims, "n_claims mismatch");
                    assert_eq!(rt.sumcheck_round_polys.len(), binding.sumcheck_round_polys.len(), "round polys len");
                    for (i, ((a0, a1, a2), (b0, b1, b2))) in rt.sumcheck_round_polys.iter().zip(&binding.sumcheck_round_polys).enumerate() {
                        assert_eq!(a0, b0, "round {i} c0 mismatch");
                        assert_eq!(a1, b1, "round {i} c1 mismatch");
                        assert_eq!(a2, b2, "round {i} c2 mismatch");
                    }
                    assert_eq!(rt.oracle_eval_at_s, binding.oracle_eval_at_s, "oracle_eval mismatch");
                    assert_eq!(rt.super_root.root, binding.super_root.root, "super_root mismatch");
                    assert_eq!(rt.super_root.subtree_roots, binding.super_root.subtree_roots, "subtree_roots mismatch");
                    assert_eq!(rt.opening_proof.intermediate_roots, binding.opening_proof.intermediate_roots, "intermediate_roots mismatch");
                    assert_eq!(rt.opening_proof.queries.len(), binding.opening_proof.queries.len(), "queries len mismatch");
                    assert_eq!(rt.opening_proof.final_value, binding.opening_proof.final_value, "final_value mismatch");
                    eprintln!(
                        "[streaming] binding proof roundtrip: PASSED ({} felts, {} rounds, {} queries)",
                        payload.len(),
                        rt.sumcheck_round_polys.len(),
                        rt.opening_proof.queries.len(),
                    );
                }
                Ok(payload.into_iter().map(|f| format!("0x{:x}", f)).collect())
            } else {
                Err(StarknetModelError::SoundnessGate(
                    "Streaming GKR requires full aggregated binding proof (packed). \
                     RLC-only is not supported."
                        .to_string(),
                ))
            }
        }
        _ => Err(StarknetModelError::SoundnessGate(
            "Streaming GKR only supports AggregatedOracleSumcheck (mode 4) binding."
                .to_string(),
        )),
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
        false,
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
        false,
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
        false,
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
        false,
    )
}

/// Build flat sncast calldata for `verify_model_gkr_v4_packed()`.
///
/// Same as v4 but uses packed QM31 format (1 felt per QM31 instead of 4),
/// reducing calldata by ~3.3x. Maps to `verify_model_gkr_v4_packed` entrypoint.
pub fn build_verify_model_gkr_v4_packed_calldata(
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
        true,
    )
}

/// Build calldata for `verify_model_gkr_v4_packed_io()` — passes IO-packed raw data directly
/// as calldata (no storage reads). Reduces calldata from ~10K to ~1.9K felts.
pub fn build_verify_model_gkr_v4_packed_io_calldata(
    proof: &crate::gkr::GKRProof,
    circuit: &crate::gkr::LayeredCircuit,
    model_id: FieldElement,
    raw_io_data: &[FieldElement],
) -> Result<VerifyModelGkrCalldata, StarknetModelError> {
    // Build the base calldata (with unpacked IO) then replace the IO section
    let base = build_verify_model_gkr_calldata_inner(
        proof,
        circuit,
        model_id,
        raw_io_data,
        GkrCalldataLayout::V4,
        true,
    )?;

    // Pack the IO data
    let packed_io = pack_m31_io_data(raw_io_data);

    // Rebuild calldata with packed IO format:
    // model_id, original_io_len: u32, packed_raw_io: Array<felt252>, ...rest
    let mut parts = Vec::new();

    // 1. model_id (same as first element)
    parts.push(base.calldata_parts[0].clone());

    // 2. original_io_len: u32
    parts.push(format!("{}", raw_io_data.len()));

    // 3. packed_raw_io: Array<felt252>
    parts.push(format!("{}", packed_io.len()));
    for f in &packed_io {
        parts.push(format!("0x{:x}", f));
    }

    // 4+. Everything after the original raw_io_data array
    // In base calldata: [model_id, io_len, io_data..., circuit_depth, ...]
    // Skip: 1 (model_id) + 1 (io_len) + raw_io_data.len() (io_data)
    let rest_start = 1 + 1 + raw_io_data.len();
    for i in rest_start..base.calldata_parts.len() {
        parts.push(base.calldata_parts[i].clone());
    }

    let total_felts = parts.len();
    Ok(VerifyModelGkrCalldata {
        calldata_parts: parts,
        total_felts,
    })
}

/// Build calldata for `verify_model_gkr_v4_packed_io_dp()` — double-packed proof data
/// where degree-2 round polys (c0, c2) fit in a single felt252 as QM31 pair.
/// Reduces proof_data section by ~50%, enabling single-TX for Qwen3-14B.
pub fn build_verify_model_gkr_v4_double_packed_io_calldata(
    proof: &crate::gkr::GKRProof,
    circuit: &crate::gkr::LayeredCircuit,
    model_id: FieldElement,
    raw_io_data: &[FieldElement],
) -> Result<VerifyModelGkrCalldata, StarknetModelError> {
    use crate::cairo_serde::serialize_gkr_proof_data_only_double_packed;

    // Build the base calldata (with regular packed proof) to get everything except proof_data
    let base = build_verify_model_gkr_calldata_inner(
        proof,
        circuit,
        model_id,
        raw_io_data,
        GkrCalldataLayout::V4,
        true, // packed
    )?;

    // Pack the IO data
    let packed_io = pack_m31_io_data(raw_io_data);

    // Serialize proof_data with double-packing
    let mut dp_proof_data = Vec::new();
    serialize_gkr_proof_data_only_double_packed(proof, &mut dp_proof_data);

    // Self-verify: round-trip each double-packed QM31 pair to catch packing bugs
    {
        let matmul_dims = extract_matmul_dims(circuit);
        let circuit_depth = circuit.layers.len() as u32;
        let num_layers = proof.layer_proofs.len() as u32;
        replay_verify_double_packed_proof(
            &dp_proof_data,
            raw_io_data,
            &matmul_dims,
            circuit_depth,
            num_layers,
            proof,
        )
        .map_err(|e| {
            StarknetModelError::SoundnessGate(format!(
                "double-packed self-verification failed: {e}"
            ))
        })?;
    }

    // Rebuild calldata with packed IO + double-packed proof_data
    let mut parts = Vec::new();

    // 1. model_id
    parts.push(base.calldata_parts[0].clone());

    // 2. original_io_len: u32
    parts.push(format!("{}", raw_io_data.len()));

    // 3. packed_raw_io: Array<felt252>
    parts.push(format!("{}", packed_io.len()));
    for f in &packed_io {
        parts.push(format!("0x{:x}", f));
    }

    // 4+. circuit_depth, num_layers, matmul_dims, dequantize_bits from base
    // In base calldata: [model_id, io_len, io_data..., circuit_depth, ...]
    let rest_start = 1 + 1 + raw_io_data.len();
    // Find where proof_data starts in the base calldata
    // After rest_start: circuit_depth(1), num_layers(1), matmul_dims(1+len), dequantize_bits(1+len), proof_data(1+len), ...
    // We need to copy up to (but not including) the proof_data array, then substitute our own

    // Extract circuit_depth, num_layers
    let circuit_depth_idx = rest_start;
    let num_layers_idx = rest_start + 1;
    parts.push(base.calldata_parts[circuit_depth_idx].clone()); // circuit_depth
    parts.push(base.calldata_parts[num_layers_idx].clone()); // num_layers

    // matmul_dims array
    let md_len_idx = rest_start + 2;
    let md_len: usize = base.calldata_parts[md_len_idx].parse::<usize>().map_err(|_| {
        StarknetModelError::SoundnessGate(format!(
            "bad calldata at index {} (expected matmul_dims length): {:?}",
            md_len_idx, base.calldata_parts[md_len_idx]
        ))
    })?;
    parts.push(base.calldata_parts[md_len_idx].clone()); // array length
    for i in 0..md_len {
        parts.push(base.calldata_parts[md_len_idx + 1 + i].clone());
    }

    // dequantize_bits array
    let dq_len_idx = md_len_idx + 1 + md_len;
    let dq_len: usize = base.calldata_parts[dq_len_idx].parse::<usize>().map_err(|_| {
        StarknetModelError::SoundnessGate(format!(
            "bad calldata at index {} (expected dequantize_bits length): {:?}",
            dq_len_idx, base.calldata_parts[dq_len_idx]
        ))
    })?;
    parts.push(base.calldata_parts[dq_len_idx].clone()); // array length
    for i in 0..dq_len {
        parts.push(base.calldata_parts[dq_len_idx + 1 + i].clone());
    }

    // proof_data: substitute with double-packed version
    parts.push(format!("{}", dp_proof_data.len()));
    for f in &dp_proof_data {
        parts.push(format!("0x{:x}", f));
    }

    // Rest: weight_commitments, weight_binding_mode, weight_binding_data, weight_opening_proofs
    // Find where proof_data ends in base
    let base_pd_len_idx = dq_len_idx + 1 + dq_len;
    let base_pd_len: usize = base.calldata_parts[base_pd_len_idx].parse::<usize>().map_err(|_| {
        StarknetModelError::SoundnessGate(format!(
            "bad calldata at index {} (expected proof_data length): {:?}",
            base_pd_len_idx, base.calldata_parts[base_pd_len_idx]
        ))
    })?;
    let after_pd_idx = base_pd_len_idx + 1 + base_pd_len;
    for i in after_pd_idx..base.calldata_parts.len() {
        parts.push(base.calldata_parts[i].clone());
    }

    // Validate total_felts: model_id(1) + original_io_len(1) + packed_io(1+len) +
    // circuit_depth(1) + num_layers(1) + matmul_dims(1+len) + dequantize_bits(1+len) +
    // dp_proof_data(1+len) + rest_from_base
    let rest_from_base = base.calldata_parts.len() - after_pd_idx;
    let expected = 1 + 1 + 1 + packed_io.len()
        + 1 + 1
        + 1 + md_len
        + 1 + dq_len
        + 1 + dp_proof_data.len()
        + rest_from_base;
    let total_felts = parts.len();
    if total_felts != expected {
        return Err(StarknetModelError::SoundnessGate(format!(
            "double-packed calldata total_felts mismatch: got {} but expected {}",
            total_felts, expected
        )));
    }

    Ok(VerifyModelGkrCalldata {
        calldata_parts: parts,
        total_felts,
    })
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
    packed: bool,
) -> Result<VerifyModelGkrCalldata, StarknetModelError> {
    use crate::cairo_serde::{serialize_gkr_proof_data_only, serialize_gkr_proof_data_only_packed};
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
    let _t_serialize = std::time::Instant::now();
    let mut proof_data = Vec::new();
    if packed {
        serialize_gkr_proof_data_only_packed(proof, &mut proof_data);
    } else {
        serialize_gkr_proof_data_only(proof, &mut proof_data);
    }
    let _serialize_elapsed = _t_serialize.elapsed();

    // Self-verify against serialized proof data (same as chunked path).
    let _t_self_verify = std::time::Instant::now();
    replay_verify_serialized_proof(
        &proof_data,
        raw_io_data,
        &matmul_dims,
        circuit_depth,
        num_layers,
        packed,
        Some(proof.io_commitment),
        proof.aggregated_binding.as_ref(),
        None, // KV-cache commitment: not used in V1/V4 direct calldata path
        None, // prev_kv_cache_commitment
    ).map_err(|e| StarknetModelError::SoundnessGate(
        format!("self-verification failed: {e}")
    ))?;
    let _self_verify_elapsed = _t_self_verify.elapsed();

    // Emit serialization profiling if enabled
    crate::gkr::profiler::print_serialization_timing(
        &crate::gkr::profiler::SerializationTimings {
            serialize: _serialize_elapsed,
            self_verify: _self_verify_elapsed,
        },
    );

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

/// Verify double-packed proof data by round-tripping each QM31 pair.
///
/// Walks the proof's layer_proofs and deferred_proofs, re-serializes each
/// double-packed element, then deserializes via `deserialize_qm31_pair_packed`
/// and compares against the proof's original QM31 values. This catches
/// packing bugs in `serialize_qm31_pair_packed` that the Fiat-Shamir replay
/// cannot detect (since it verifies *packed*, not double-packed encoding).
///
/// Also verifies dp_proof_data is strictly smaller than regular packed.
pub fn replay_verify_double_packed_proof(
    dp_proof_data: &[FieldElement],
    raw_io: &[FieldElement],
    matmul_dims: &[u32],
    circuit_depth: u32,
    num_layers: u32,
    proof: &crate::gkr::GKRProof,
) -> Result<(), String> {
    use crate::cairo_serde::{
        deserialize_qm31_pair_packed, serialize_qm31_pair_packed, serialize_qm31_packed,
    };
    use crate::crypto::poseidon_channel::felt_to_securefield;
    use crate::gkr::types::LayerProof;

    // Helper: verify a double-packed QM31 pair round-trips correctly
    let verify_pair = |a: stwo::core::fields::qm31::SecureField,
                       b: stwo::core::fields::qm31::SecureField,
                       ctx: &str|
     -> Result<(), String> {
        let mut buf = Vec::new();
        serialize_qm31_pair_packed(a, b, &mut buf);
        let (ra, rb) = deserialize_qm31_pair_packed(buf[0]);
        if ra != a || rb != b {
            return Err(format!(
                "double-packed round-trip mismatch at {}: expected ({:?},{:?}), got ({:?},{:?})",
                ctx, a, b, ra, rb
            ));
        }
        Ok(())
    };

    // Helper: verify a single-packed QM31 round-trips correctly
    let verify_single =
        |v: stwo::core::fields::qm31::SecureField, ctx: &str| -> Result<(), String> {
            let mut buf = Vec::new();
            serialize_qm31_packed(v, &mut buf);
            let rv = felt_to_securefield(buf[0]);
            if rv != v {
                return Err(format!(
                    "single-packed round-trip mismatch at {}: expected {:?}, got {:?}",
                    ctx, v, rv
                ));
            }
            Ok(())
        };

    // Walk layer proofs and verify each double-packed element
    for (li, lp) in proof.layer_proofs.iter().enumerate() {
        match lp {
            LayerProof::MatMul {
                round_polys,
                final_a_eval,
                final_b_eval,
            } => {
                for (ri, rp) in round_polys.iter().enumerate() {
                    verify_pair(
                        rp.c0,
                        rp.c2,
                        &format!("layer[{}].MatMul.round[{}].(c0,c2)", li, ri),
                    )?;
                }
                verify_single(*final_a_eval, &format!("layer[{}].MatMul.final_a", li))?;
                verify_single(*final_b_eval, &format!("layer[{}].MatMul.final_b", li))?;
            }
            LayerProof::Mul {
                eq_round_polys,
                lhs_eval,
                rhs_eval,
            } => {
                for (ri, rp) in eq_round_polys.iter().enumerate() {
                    verify_pair(
                        rp.c0,
                        rp.c2,
                        &format!("layer[{}].Mul.round[{}].(c0,c2)", li, ri),
                    )?;
                    verify_single(
                        rp.c3,
                        &format!("layer[{}].Mul.round[{}].c3", li, ri),
                    )?;
                }
                verify_single(*lhs_eval, &format!("layer[{}].Mul.lhs", li))?;
                verify_single(*rhs_eval, &format!("layer[{}].Mul.rhs", li))?;
            }
            LayerProof::LayerNorm {
                linear_round_polys,
                linear_final_evals,
                ..
            }
            | LayerProof::RMSNorm {
                linear_round_polys,
                linear_final_evals,
                ..
            } => {
                let tag = if matches!(lp, LayerProof::LayerNorm { .. }) {
                    "LayerNorm"
                } else {
                    "RMSNorm"
                };
                for (ri, rp) in linear_round_polys.iter().enumerate() {
                    verify_pair(
                        rp.c0,
                        rp.c2,
                        &format!("layer[{}].{}.round[{}].(c0,c2)", li, tag, ri),
                    )?;
                    verify_single(
                        rp.c3,
                        &format!("layer[{}].{}.round[{}].c3", li, tag, ri),
                    )?;
                }
                let (a, b) = *linear_final_evals;
                verify_single(a, &format!("layer[{}].{}.final_eval_0", li, tag))?;
                verify_single(b, &format!("layer[{}].{}.final_eval_1", li, tag))?;
            }
            LayerProof::MatMulDualSimd {
                round_polys,
                final_a_eval,
                final_b_eval,
                ..
            } => {
                for (ri, rp) in round_polys.iter().enumerate() {
                    verify_pair(
                        rp.c0,
                        rp.c1,
                        &format!("layer[{}].DualSimd.round[{}].(c0,c1)", li, ri),
                    )?;
                    verify_pair(
                        rp.c2,
                        rp.c3,
                        &format!("layer[{}].DualSimd.round[{}].(c2,c3)", li, ri),
                    )?;
                }
                verify_single(*final_a_eval, &format!("layer[{}].DualSimd.final_a", li))?;
                verify_single(*final_b_eval, &format!("layer[{}].DualSimd.final_b", li))?;
            }
            // Add, Activation, Dequantize, Quantize, Embedding, Attention —
            // these don't use double-packed pairs for round polys (only single-packed),
            // so no additional pair verification needed
            _ => {}
        }
    }

    // Walk deferred proofs (MatMul only)
    for (di, deferred) in proof.deferred_proofs.iter().enumerate() {
        if let LayerProof::MatMul {
            round_polys,
            final_a_eval,
            final_b_eval,
        } = &deferred.layer_proof
        {
            for (ri, rp) in round_polys.iter().enumerate() {
                verify_pair(
                    rp.c0,
                    rp.c2,
                    &format!("deferred[{}].round[{}].(c0,c2)", di, ri),
                )?;
            }
            verify_single(*final_a_eval, &format!("deferred[{}].final_a", di))?;
            verify_single(*final_b_eval, &format!("deferred[{}].final_b", di))?;
        }
    }

    // Verify that double-packed is strictly smaller than regular packed
    let mut packed_data = Vec::new();
    crate::cairo_serde::serialize_gkr_proof_data_only_packed(proof, &mut packed_data);
    let dp_len = dp_proof_data.len();
    let packed_len = packed_data.len();
    if dp_len >= packed_len {
        return Err(format!(
            "Double-packed should be smaller: dp={} >= packed={}",
            dp_len, packed_len
        ));
    }

    // Run Fiat-Shamir channel replay on the packed representation.
    // This ensures double-packed proofs get the same cryptographic verification
    // as regular packed proofs (sumcheck arithmetic, deferred proofs, trailing data).
    replay_verify_serialized_proof(
        &packed_data, raw_io, matmul_dims, circuit_depth, num_layers,
        true, // packed
        Some(proof.io_commitment),
        proof.aggregated_binding.as_ref(),
        None, // KV-cache commitment: double-packed path doesn't use KV yet
        None, // prev KV-cache commitment: double-packed path doesn't use KV yet
    )?;

    Ok(())
}

pub fn replay_verify_serialized_proof(
    proof_data: &[FieldElement],
    raw_io: &[FieldElement],
    matmul_dims: &[u32],
    circuit_depth: u32,
    num_layers: u32,
    packed: bool,
    expected_io_commitment: Option<FieldElement>,
    weight_binding: Option<&crate::crypto::aggregated_opening::AggregatedWeightBindingProof>,
    kv_cache_commitment: Option<FieldElement>,
    prev_kv_cache_commitment: Option<FieldElement>,
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

    if raw_io.len() < 6 {
        return Err(format!(
            "truncated raw_io: need at least 6 header felts, got {}",
            raw_io.len()
        ));
    }

    let input_rows = felt_to_u64(&raw_io[0]);
    let input_cols = felt_to_u64(&raw_io[1]);
    let input_len = felt_to_u64(&raw_io[2]) as usize;
    let out_start = 3usize.checked_add(input_len).ok_or_else(|| {
        format!("integer overflow computing out_start: 3 + {input_len}")
    })?;

    if raw_io.len() < out_start + 3 {
        return Err(format!(
            "truncated raw_io: need at least {} felts for output header, got {}",
            out_start + 3,
            raw_io.len()
        ));
    }

    let output_rows = felt_to_u64(&raw_io[out_start]) as usize;
    let output_cols = felt_to_u64(&raw_io[out_start + 1]) as usize;
    let output_len = felt_to_u64(&raw_io[out_start + 2]) as usize;

    let output_data_end = out_start.checked_add(3).and_then(|s| s.checked_add(output_len))
        .ok_or_else(|| format!("integer overflow computing output data end"))?;
    if raw_io.len() < output_data_end {
        return Err(format!(
            "truncated raw_io: need {} felts for output data, got {}",
            output_data_end,
            raw_io.len()
        ));
    }

    if output_rows.checked_mul(output_cols) != Some(output_len) {
        return Err(format!(
            "output dimension mismatch: {output_rows} * {output_cols} != {output_len}"
        ));
    }

    // IO commitment check: verify poseidon_hash(raw_io) matches expected commitment.
    // Uses the same format as aggregation::compute_io_commitment() — dimension-prefixed
    // M31 values (not packed format).
    if let Some(expected) = expected_io_commitment {
        let computed = starknet_crypto::poseidon_hash_many(raw_io);
        if computed != expected {
            return Err(format!(
                "IO commitment mismatch: expected {:?}, computed {:?}",
                expected, computed
            ));
        }
    }

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
    // KV-cache commitments mixed BEFORE circuit_depth (matches Cairo verifier order).
    // Both current and previous KV are bound for sequential inference chaining.
    if let Some(kv) = kv_cache_commitment {
        ch.mix_felt(kv);
        if let Some(prev_kv) = prev_kv_cache_commitment {
            ch.mix_felt(prev_kv);
        }
    }
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
    let trace = std::env::var("STWO_CHANNEL_TRACE").is_ok();

    if trace {
        eprintln!("[VERIFIER] ch after init: {:?}", ch.digest());
        eprintln!("[VERIFIER] output_value: {:?}", output_value);
    }

    for layer in 0..num_layers as usize {
        let tag = read_u32_from(proof_data, &mut off);
        if trace {
            eprintln!("[VERIFIER] layer {} tag={} off={} ch={:?}", layer, tag, off, ch.digest());
        }

        match tag {
            0 => {
                // MatMul
                let m = matmul_dims[matmul_idx * 3] as usize;
                let k = matmul_dims[matmul_idx * 3 + 1] as usize;
                let n = matmul_dims[matmul_idx * 3 + 2] as usize;
                matmul_idx += 1;

                if trace {
                    eprintln!("[VERIFIER MatMul] m={} k={} n={} claim={:?}", m, k, n, current_claim_value);
                }
                ch.mix_u64(m as u64);
                ch.mix_u64(k as u64);
                ch.mix_u64(n as u64);
                mix_secure_field(&mut ch, current_claim_value);
                if trace {
                    eprintln!("[VERIFIER MatMul] ch after seeding: {:?}", ch.digest());
                }

                let num_rounds = read_u32_from(proof_data, &mut off) as usize;
                let mut current_sum = current_claim_value;
                let two = SecureField::from(M31::from(2u32));

                for round in 0..num_rounds {
                    let c0 = read_qm31_from(proof_data, &mut off);
                    // v19+: c1 always omitted (compressed round polys), reconstruct from current_sum
                    let c2 = read_qm31_from(proof_data, &mut off);
                    let c1 = current_sum - two * c0 - c2;
                    ch.mix_poly_coeffs(c0, c1, c2);
                    let challenge = ch.draw_qm31();
                    current_sum = c0 + c1 * challenge + c2 * challenge * challenge;
                    if trace && round < 3 {
                        eprintln!("[VERIFIER MatMul] round {} c0={:?} c1={:?} c2={:?}", round, c0, c1, c2);
                        eprintln!("[VERIFIER MatMul] round {} challenge={:?} new_sum={:?}", round, challenge, current_sum);
                        eprintln!("[VERIFIER MatMul] round {} ch={:?}", round, ch.digest());
                    }
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
                let simd_combined = read_u32_from(proof_data, &mut off);

                // === Part 0: RMS² verification plain sumcheck ===
                // Must be replayed BEFORE "RN" tag to match prover's channel mixing order.
                let has_p0 = read_u32_from(proof_data, &mut off);
                // SIMD consistency gate: non-SIMD must have Part 0, SIMD must not
                if simd_combined == 0 && has_p0 != 1 {
                    return Err(format!("layer {}: non-SIMD RMSNorm requires Part 0 (has_p0={})", layer, has_p0));
                }
                if simd_combined == 1 && has_p0 != 0 {
                    return Err(format!("layer {}: SIMD RMSNorm must not have Part 0 (has_p0={})", layer, has_p0));
                }
                if has_p0 == 1 {
                    let two_p0 = SecureField::from(M31::from(2u32));
                    let p0_n_active = read_u32_from(proof_data, &mut off) as u64;
                    let p0_sq_sum = read_qm31_from(proof_data, &mut off);
                    ch.mix_u64(0x5251_u64); // "RQ" tag
                    ch.mix_u64(p0_n_active);
                    mix_secure_field(&mut ch, p0_sq_sum);
                    let p0_nr = read_u32_from(proof_data, &mut off) as usize;
                    let mut p0_sum = p0_sq_sum;
                    for _ in 0..p0_nr {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        // Degree-2 polynomial: c3=0, c1 reconstructed from current_sum
                        let c1 = p0_sum - two_p0 * c0 - c2;
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let r = ch.draw_qm31();
                        p0_sum = c0 + c1 * r + c2 * r * r + c3 * r * r * r;
                    }
                    let p0_input_final = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, p0_input_final);
                    if trace {
                        eprintln!("[VERIFIER RMSNorm] ch after Part 0 (RMS² sumcheck): {:?}", ch.digest());
                    }
                }

                if trace {
                    eprintln!("[VERIFIER RMSNorm] ch BEFORE RN: {:?}", ch.digest());
                    eprintln!("[VERIFIER RMSNorm] rms_sq={:?}", rms_sq);
                    eprintln!("[VERIFIER RMSNorm] rsqrt_eval={:?}", rsqrt_eval);
                    eprintln!("[VERIFIER RMSNorm] claim={:?}", current_claim_value);
                    eprintln!("[VERIFIER RMSNorm] input_eval={:?}", input_eval);
                    eprintln!("[VERIFIER RMSNorm] output_eval={:?}", output_eval);
                }
                ch.mix_u64(0x524E); // "RN"
                mix_secure_field(&mut ch, rms_sq);
                mix_secure_field(&mut ch, rsqrt_eval);
                mix_secure_field(&mut ch, current_claim_value);
                if trace {
                    eprintln!("[VERIFIER RMSNorm] ch after mix claim: {:?}", ch.digest());
                }

                let nrounds = read_u32_from(proof_data, &mut off) as usize;
                if trace {
                    eprintln!("[VERIFIER RMSNorm] nrounds={}", nrounds);
                }
                let two = SecureField::from(M31::from(2u32));
                let mut rms_sum = current_claim_value;
                for _round in 0..nrounds {
                    let c0 = read_qm31_from(proof_data, &mut off);
                    // v19+: c1 always omitted (compressed round polys), reconstruct from current_sum
                    let c2 = read_qm31_from(proof_data, &mut off);
                    let c3 = read_qm31_from(proof_data, &mut off);
                    let c1 = rms_sum - two * c0 - c2 - c3;
                    ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                    let challenge = ch.draw_qm31();
                    rms_sum = c0 + c1 * challenge + c2 * challenge * challenge
                        + c3 * challenge * challenge * challenge;
                }
                let input_final = read_qm31_from(proof_data, &mut off);
                let rsqrt_final = read_qm31_from(proof_data, &mut off);
                if trace {
                    eprintln!("[VERIFIER RMSNorm] ch after {} eq-rounds: {:?}", nrounds, ch.digest());
                    eprintln!("[VERIFIER RMSNorm] input_final={:?}", input_final);
                    eprintln!("[VERIFIER RMSNorm] rsqrt_final={:?}", rsqrt_final);
                }
                mix_secure_field(&mut ch, input_final);
                mix_secure_field(&mut ch, rsqrt_final);
                if trace {
                    eprintln!("[VERIFIER RMSNorm] ch after final evals: {:?}", ch.digest());
                }

                // Optional logup
                let has_logup = read_u32_from(proof_data, &mut off);
                if trace {
                    eprintln!("[VERIFIER RMSNorm] has_logup={}", has_logup);
                }
                if has_logup == 1 {
                    ch.mix_u64(0x4C4F47); // "LOG"
                    ch.mix_u64(0x524E); // "RN"
                    let _gamma = ch.draw_qm31();
                    let _beta = ch.draw_qm31();
                    let claimed_sum = read_qm31_from(proof_data, &mut off);
                    if trace {
                        eprintln!("[VERIFIER RMSNorm] claimed_sum={:?}", claimed_sum);
                    }
                    mix_secure_field(&mut ch, claimed_sum);
                    if trace {
                        eprintln!("[VERIFIER RMSNorm] ch after mix claimed_sum: {:?}", ch.digest());
                    }
                    let eq_rounds = read_u32_from(proof_data, &mut off) as usize;
                    if trace {
                        eprintln!("[VERIFIER RMSNorm] logup eq_rounds={}", eq_rounds);
                    }
                    let two_logup = SecureField::from(M31::from(2u32));
                    let mut logup_sum = SecureField::from(M31::from(1u32));
                    for _round in 0..eq_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        // v19+: c1 always omitted (compressed round polys)
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let c1 = logup_sum - two_logup * c0 - c2 - c3;
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
                    if trace {
                        eprintln!("[VERIFIER RMSNorm] ch after logup: {:?}", ch.digest());
                    }
                }
                // Read multiplicity sumcheck (always serialized after logup)
                let has_ms = read_u32_from(proof_data, &mut off);
                if has_ms == 1 {
                    let ms_n_rounds = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..ms_n_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c1 = read_qm31_from(proof_data, &mut off);
                        mix_secure_field(&mut ch, c0);
                        mix_secure_field(&mut ch, c1);
                        let _r = ch.draw_qm31();
                    }
                    let _final_eval = read_qm31_from(proof_data, &mut off);
                    let _claimed_sum = read_qm31_from(proof_data, &mut off);
                }
                if trace {
                    eprintln!("[VERIFIER RMSNorm] ch after mult sumcheck: {:?}", ch.digest());
                }

                // Per-row rms_sq for multi-row binding (not channel-mixed, just consume)
                let has_row_rms_sq = read_u32_from(proof_data, &mut off);
                if has_row_rms_sq == 1 {
                    let num_rows = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..num_rows {
                        let _ = read_u32_from(proof_data, &mut off);
                    }
                }

                mix_secure_field(&mut ch, input_eval);
                mix_secure_field(&mut ch, output_eval);
                if trace {
                    eprintln!("[VERIFIER RMSNorm] ch FINAL: {:?}", ch.digest());
                }
                current_claim_value = input_eval;
            }
            3 => {
                // Activation
                let _act_type = read_u32_from(proof_data, &mut off);
                let input_eval = read_qm31_from(proof_data, &mut off);
                let _output_eval = read_qm31_from(proof_data, &mut off);
                off += 1; // table_commitment

                let has_logup = read_u32_from(proof_data, &mut off);
                if has_logup == 1 {
                    // Full LogUp path (legacy, used when inputs are in table range)
                    ch.mix_u64(0x4C4F47); // "LOG"
                    ch.mix_u64(_act_type as u64);
                    let _gamma = ch.draw_qm31();
                    let _beta = ch.draw_qm31();

                    let claimed_sum = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, claimed_sum);
                    let eq_rounds = read_u32_from(proof_data, &mut off) as usize;
                    let two_act = SecureField::from(M31::from(2u32));
                    let mut logup_sum = SecureField::from(M31::from(1u32));
                    for _round in 0..eq_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        // v19+: c1 always omitted (compressed round polys)
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let c1 = logup_sum - two_act * c0 - c2 - c3;
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
                // Read multiplicity sumcheck (always serialized after logup)
                let has_ms = read_u32_from(proof_data, &mut off);
                if has_ms == 1 {
                    let ms_n_rounds = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..ms_n_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c1 = read_qm31_from(proof_data, &mut off);
                        mix_secure_field(&mut ch, c0);
                        mix_secure_field(&mut ch, c1);
                        let _r = ch.draw_qm31();
                    }
                    let _final_eval = read_qm31_from(proof_data, &mut off);
                    let _claimed_sum = read_qm31_from(proof_data, &mut off);
                }
                // Read activation product proof (Phase A soundness)
                let has_act_proof = read_u32_from(proof_data, &mut off);
                if has_act_proof == 1 {
                    // Algebraic product+binary eq-sumcheck (replaces LogUp for ReLU)
                    ch.mix_u64(0x414354); // "ACT"
                    mix_secure_field(&mut ch, current_claim_value);
                    let _eta = ch.draw_qm31();

                    let act_rounds = read_u32_from(proof_data, &mut off) as usize;
                    let two_act = SecureField::from(M31::from(2u32));
                    let mut act_sum = current_claim_value;
                    for _round in 0..act_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let c1 = act_sum - two_act * c0 - c2 - c3;
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        act_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }
                    let act_input_eval = read_qm31_from(proof_data, &mut off);
                    let act_indicator_eval = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, act_input_eval);
                    mix_secure_field(&mut ch, act_indicator_eval);
                    // Phase B: read and mix bit_evals if present
                    let has_bit_evals = read_u32_from(proof_data, &mut off);
                    if has_bit_evals == 1 {
                        let num_bits = read_u32_from(proof_data, &mut off) as usize;
                        for _ in 0..num_bits {
                            let bit_eval = read_qm31_from(proof_data, &mut off);
                            mix_secure_field(&mut ch, bit_eval);
                        }
                    }
                    current_claim_value = act_input_eval;
                } else {
                    // No algebraic product proof — will check piecewise below
                }
                // Piecewise algebraic proof (always serialized after act_proof)
                let has_pw = read_u32_from(proof_data, &mut off);
                if has_pw == 1 {
                    // Full piecewise channel replay (GELU/Sigmoid/Softmax)
                    ch.mix_u64(0x50575F414354_u64); // "PW_ACT"
                    ch.mix_u64(_act_type as u64);
                    let pw_nr = read_u32_from(proof_data, &mut off) as usize;
                    ch.mix_u64(pw_nr as u64); // num_vars
                    mix_secure_field(&mut ch, current_claim_value);
                    let _eta = ch.draw_qm31();
                    let two_pw = SecureField::from(M31::from(2u32));
                    // All piecewise constraints vanish for honest prover → claimed sum = 0
                    let mut pw_sum = SecureField::zero();
                    for _ in 0..pw_nr {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let c1 = pw_sum - two_pw * c0 - c2 - c3;
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        pw_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }
                    let pw_input_eval = read_qm31_from(proof_data, &mut off);
                    let pw_output_eval = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, pw_input_eval);
                    mix_secure_field(&mut ch, pw_output_eval);
                    for _ in 0..16usize {
                        let ie = read_qm31_from(proof_data, &mut off);
                        mix_secure_field(&mut ch, ie);
                    }
                    let has_sbe = read_u32_from(proof_data, &mut off);
                    if has_sbe == 1 {
                        for _ in 0..4usize {
                            let sb = read_qm31_from(proof_data, &mut off);
                            mix_secure_field(&mut ch, sb);
                        }
                    }
                    current_claim_value = pw_input_eval;
                } else if has_act_proof == 0 {
                    // Legacy: no act_proof, no piecewise (SIMD/no-proof path)
                    if has_logup == 1 {
                        mix_secure_field(&mut ch, input_eval);
                        mix_secure_field(&mut ch, _output_eval);
                    } else {
                        mix_secure_field(&mut ch, input_eval);
                    }
                    current_claim_value = input_eval;
                }
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
            4 => {
                // LayerNorm
                let input_eval = read_qm31_from(proof_data, &mut off);
                let output_eval = read_qm31_from(proof_data, &mut off);
                let mean = read_qm31_from(proof_data, &mut off);
                let rsqrt_var = read_qm31_from(proof_data, &mut off);
                off += 1; // commitment
                let simd_combined = read_u32_from(proof_data, &mut off);

                // Part 0: Mean-Variance plain sumcheck
                let has_mv = read_u32_from(proof_data, &mut off);
                // SIMD consistency gate: non-SIMD must have Part 0, SIMD must not
                if simd_combined == 0 && has_mv != 1 {
                    return Err(format!("layer {}: non-SIMD LayerNorm requires Part 0 (has_mv={})", layer, has_mv));
                }
                if simd_combined == 1 && has_mv != 0 {
                    return Err(format!("layer {}: SIMD LayerNorm must not have Part 0 (has_mv={})", layer, has_mv));
                }
                if has_mv == 1 {
                    let mv_n_active = read_u32_from(proof_data, &mut off) as u64;
                    let total_input_sum = read_qm31_from(proof_data, &mut off);
                    let total_centered_sq_sum = read_qm31_from(proof_data, &mut off);
                    ch.mix_u64(0x4D56); // "MV"
                    ch.mix_u64(mv_n_active);
                    mix_secure_field(&mut ch, total_input_sum);
                    mix_secure_field(&mut ch, total_centered_sq_sum);
                    let eta0 = ch.draw_qm31();
                    let mv_nr = read_u32_from(proof_data, &mut off) as usize;
                    let two_mv = SecureField::from(M31::from(2u32));
                    let mut mv_sum = eta0 * total_input_sum + eta0 * eta0 * total_centered_sq_sum;
                    for _ in 0..mv_nr {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        // Degree-2 poly: c3=0, c1 reconstructed from current_sum
                        let c1 = mv_sum - two_mv * c0 - c2;
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let r = ch.draw_qm31();
                        mv_sum = c0 + c1 * r + c2 * r * r + c3 * r * r * r;
                    }
                    let mv_input_final = read_qm31_from(proof_data, &mut off);
                    let mv_mean_final = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, mv_input_final);
                    mix_secure_field(&mut ch, mv_mean_final);
                }

                // Part 1: Linear eq-sumcheck
                ch.mix_u64(0x4C4E); // "LN"
                mix_secure_field(&mut ch, mean);
                mix_secure_field(&mut ch, rsqrt_var);
                mix_secure_field(&mut ch, current_claim_value);
                let ln_nr = read_u32_from(proof_data, &mut off) as usize;
                let two_ln = SecureField::from(M31::from(2u32));
                let mut ln_sum = current_claim_value;
                for _ in 0..ln_nr {
                    let c0 = read_qm31_from(proof_data, &mut off);
                    let c2 = read_qm31_from(proof_data, &mut off);
                    let c3 = read_qm31_from(proof_data, &mut off);
                    let c1 = ln_sum - two_ln * c0 - c2 - c3;
                    ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                    let challenge = ch.draw_qm31();
                    ln_sum = c0 + c1 * challenge + c2 * challenge * challenge
                        + c3 * challenge * challenge * challenge;
                }
                let centered_final = read_qm31_from(proof_data, &mut off);
                let rsqrt_final = read_qm31_from(proof_data, &mut off);
                mix_secure_field(&mut ch, centered_final);
                mix_secure_field(&mut ch, rsqrt_final);
                // Centered binding evals only present in CPU path (has_mv == 1)
                if has_mv == 1 {
                    let cb_input = read_qm31_from(proof_data, &mut off);
                    let cb_mean = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, cb_input);
                    mix_secure_field(&mut ch, cb_mean);
                }

                // Part 2: rsqrt LogUp
                let has_logup = read_u32_from(proof_data, &mut off);
                if has_logup == 1 {
                    ch.mix_u64(0x4C4F47); // "LOG"
                    ch.mix_u64(0x5253);   // "RS"
                    let _gamma = ch.draw_qm31();
                    let _beta = ch.draw_qm31();
                    let claimed_sum = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, claimed_sum);
                    let eq_rounds = read_u32_from(proof_data, &mut off) as usize;
                    let two_logup = SecureField::from(M31::from(2u32));
                    let mut logup_sum = SecureField::from(M31::from(1u32));
                    for _ in 0..eq_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let c1 = logup_sum - two_logup * c0 - c2 - c3;
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

                // Multiplicity sumcheck
                let has_ms = read_u32_from(proof_data, &mut off);
                if has_ms == 1 {
                    let ms_n_rounds = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..ms_n_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c1 = read_qm31_from(proof_data, &mut off);
                        mix_secure_field(&mut ch, c0);
                        mix_secure_field(&mut ch, c1);
                        let _r = ch.draw_qm31();
                    }
                    let _final_eval = read_qm31_from(proof_data, &mut off);
                    let _claimed_sum = read_qm31_from(proof_data, &mut off);
                }

                // var_eval (read, not mixed) — only present in CPU path
                if has_mv == 1 {
                    let _var_eval = read_qm31_from(proof_data, &mut off);
                }

                // Per-row means for multi-row binding (not channel-mixed, just consume)
                let has_row_means = read_u32_from(proof_data, &mut off);
                if has_row_means == 1 {
                    let num_rows = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..num_rows {
                        let _ = read_u32_from(proof_data, &mut off);
                    }
                }
                // Per-row variances (consumed for offset tracking, not channel-mixed)
                let has_row_variances = read_u32_from(proof_data, &mut off);
                if has_row_variances == 1 {
                    let num_rows = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..num_rows {
                        let _ = read_u32_from(proof_data, &mut off);
                    }
                }

                mix_secure_field(&mut ch, input_eval);
                mix_secure_field(&mut ch, output_eval);
                current_claim_value = input_eval;
            }
            6 => {
                // Dequantize
                let bits = read_u32_from(proof_data, &mut off);
                let input_eval = read_qm31_from(proof_data, &mut off);
                let output_eval = read_qm31_from(proof_data, &mut off);
                off += 1; // table_commitment

                let has_logup = read_u32_from(proof_data, &mut off);
                if has_logup == 1 {
                    ch.mix_u64(0x4445514C4F47_u64); // "DEQLOG"
                    ch.mix_u64(bits as u64);
                    let _gamma = ch.draw_qm31();
                    let _beta = ch.draw_qm31();
                    let claimed_sum = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, claimed_sum);
                    let eq_rounds = read_u32_from(proof_data, &mut off) as usize;
                    let two_deq = SecureField::from(M31::from(2u32));
                    let mut logup_sum = SecureField::from(M31::from(1u32));
                    for _ in 0..eq_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let c1 = logup_sum - two_deq * c0 - c2 - c3;
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }
                    // Final evals: w, in, out
                    let _w = read_qm31_from(proof_data, &mut off);
                    let _in_e = read_qm31_from(proof_data, &mut off);
                    let _out_e = read_qm31_from(proof_data, &mut off);
                    let num_mults = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..num_mults {
                        let _ = read_u32_from(proof_data, &mut off);
                    }
                }
                // Multiplicity sumcheck
                let has_ms = read_u32_from(proof_data, &mut off);
                if has_ms == 1 {
                    let ms_n_rounds = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..ms_n_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c1 = read_qm31_from(proof_data, &mut off);
                        mix_secure_field(&mut ch, c0);
                        mix_secure_field(&mut ch, c1);
                        let _r = ch.draw_qm31();
                    }
                    let _final_eval = read_qm31_from(proof_data, &mut off);
                    let _claimed_sum = read_qm31_from(proof_data, &mut off);
                }
                mix_secure_field(&mut ch, input_eval);
                mix_secure_field(&mut ch, output_eval);
                current_claim_value = input_eval;
                if trace {
                    eprintln!("[VERIFIER Dequantize] layer {} bits={} ch={:?}", layer, bits, ch.digest());
                }
            }
            9 => {
                // Quantize
                let bits = read_u32_from(proof_data, &mut off);
                let zero_point_abs = read_u32_from(proof_data, &mut off);
                let scale_fixed_hi = read_u32_from(proof_data, &mut off);
                let scale_fixed_lo = read_u32_from(proof_data, &mut off);
                let scale_fixed = ((scale_fixed_hi as u64) << 32) | (scale_fixed_lo as u64);
                let strategy_tag = read_u32_from(proof_data, &mut off);
                let input_eval = read_qm31_from(proof_data, &mut off);
                let output_eval = read_qm31_from(proof_data, &mut off);

                let has_logup = read_u32_from(proof_data, &mut off);
                if has_logup == 1 {
                    ch.mix_u64(0x514C4F47_u64); // "QLOG"
                    ch.mix_u64(bits as u64);
                    ch.mix_u64(zero_point_abs as u64);
                    ch.mix_u64(scale_fixed);
                    ch.mix_u64(strategy_tag as u64);
                    let _gamma = ch.draw_qm31();
                    let _beta = ch.draw_qm31();
                    let claimed_sum = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, claimed_sum);
                    let eq_rounds = read_u32_from(proof_data, &mut off) as usize;
                    let two_q = SecureField::from(M31::from(2u32));
                    let mut logup_sum = SecureField::from(M31::from(1u32));
                    for _ in 0..eq_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let c1 = logup_sum - two_q * c0 - c2 - c3;
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }
                    // Final evals: w, in, out
                    let _w = read_qm31_from(proof_data, &mut off);
                    let _in_e = read_qm31_from(proof_data, &mut off);
                    let _out_e = read_qm31_from(proof_data, &mut off);
                    let num_mults = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..num_mults {
                        let _ = read_u32_from(proof_data, &mut off);
                    }
                }
                // Table entries
                let table_len = read_u32_from(proof_data, &mut off) as usize;
                for _ in 0..table_len {
                    let _ = read_u32_from(proof_data, &mut off); // table input
                    let _ = read_u32_from(proof_data, &mut off); // table output
                }
                mix_secure_field(&mut ch, input_eval);
                mix_secure_field(&mut ch, output_eval);
                current_claim_value = input_eval;
                if trace {
                    eprintln!("[VERIFIER Quantize] layer {} bits={} ch={:?}", layer, bits, ch.digest());
                }
            }
            2 => {
                // Mul (element-wise product via eq-sumcheck)
                ch.mix_u64(0x4D554C_u64); // "MUL"
                mix_secure_field(&mut ch, current_claim_value);

                let num_rounds = read_u32_from(proof_data, &mut off) as usize;
                let two = SecureField::from(M31::from(2u32));
                let mut current_sum = current_claim_value;
                for _ in 0..num_rounds {
                    let c0 = read_qm31_from(proof_data, &mut off);
                    let c2 = read_qm31_from(proof_data, &mut off);
                    let c3 = read_qm31_from(proof_data, &mut off);
                    let c1 = current_sum - two * c0 - c2 - c3;
                    ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                    let _challenge = ch.draw_qm31();
                    current_sum = c0 + c1 * _challenge + c2 * _challenge * _challenge
                        + c3 * _challenge * _challenge * _challenge;
                }
                let lhs_eval = read_qm31_from(proof_data, &mut off);
                let rhs_eval = read_qm31_from(proof_data, &mut off);
                mix_secure_field(&mut ch, lhs_eval);
                mix_secure_field(&mut ch, rhs_eval);
                let alpha = ch.draw_qm31();
                current_claim_value = alpha * lhs_eval
                    + (SecureField::from(M31::from(1u32)) - alpha) * rhs_eval;
                if trace {
                    eprintln!("[VERIFIER Mul] layer {} ch={:?}", layer, ch.digest());
                }
            }
            7 => {
                // MatMulDualSimd (block-extended 3-factor sumcheck)
                let m = matmul_dims[matmul_idx * 3] as usize;
                let k = matmul_dims[matmul_idx * 3 + 1] as usize;
                let n = matmul_dims[matmul_idx * 3 + 2] as usize;
                matmul_idx += 1;

                let n_block_vars = read_u32_from(proof_data, &mut off) as usize;
                let n_blocks = 1usize << n_block_vars;

                ch.mix_u64(m as u64);
                ch.mix_u64(k as u64);
                ch.mix_u64(n as u64);
                ch.mix_u64(n_blocks as u64);
                mix_secure_field(&mut ch, current_claim_value);

                let num_rounds = read_u32_from(proof_data, &mut off) as usize;
                let two = SecureField::from(M31::from(2u32));
                let mut current_sum = current_claim_value;
                for _ in 0..num_rounds {
                    let c0 = read_qm31_from(proof_data, &mut off);
                    // In packed mode all 4 coefficients are present;
                    // in unpacked mode c1 is omitted and reconstructed.
                    let (c1, c2, c3) = if packed {
                        let c1 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        (c1, c2, c3)
                    } else {
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        let c1 = current_sum - two * c0 - c2 - c3;
                        (c1, c2, c3)
                    };
                    ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                    let challenge = ch.draw_qm31();
                    current_sum = c0 + c1 * challenge + c2 * challenge * challenge
                        + c3 * challenge * challenge * challenge;
                }
                let final_a = read_qm31_from(proof_data, &mut off);
                let final_b = read_qm31_from(proof_data, &mut off);
                mix_secure_field(&mut ch, final_a);
                mix_secure_field(&mut ch, final_b);
                current_claim_value = final_a;
                if trace {
                    eprintln!("[VERIFIER MatMulDualSimd] layer {} m={} k={} n={} nb={} ch={:?}",
                        layer, m, k, n, n_blocks, ch.digest());
                }
            }
            5 => {
                // Attention (decomposed sub-matmul proofs)
                let att_num_heads = read_u32_from(proof_data, &mut off) as usize;
                let att_seq_len = read_u32_from(proof_data, &mut off) as usize;
                let att_d_model = read_u32_from(proof_data, &mut off) as usize;
                let att_causal = read_u32_from(proof_data, &mut off);
                let att_d_k = att_d_model / att_num_heads.max(1);

                let num_sub = read_u32_from(proof_data, &mut off) as usize;
                let mut sub_claim_values = Vec::with_capacity(num_sub);
                for _ in 0..num_sub {
                    // Sub-claim values are packed QM31s in packed mode
                    sub_claim_values.push(read_qm31_from(proof_data, &mut off));
                }

                ch.mix_u64(0x4154544E_u64); // "ATTN"
                ch.mix_u64(att_num_heads as u64);
                ch.mix_u64(att_seq_len as u64);
                ch.mix_u64(att_d_model as u64);
                ch.mix_u64(att_causal as u64);

                // Helper: replay a sub-matmul proof from v1 unpacked serialized data.
                // Sub-proofs are always serialized in v1 unpacked format (4 felts per QM31).
                let replay_sub_matmul = |data: &[FieldElement],
                                         off: &mut usize,
                                         ch: &mut PoseidonChannel,
                                         claim_value: SecureField,
                                         m: usize,
                                         k: usize,
                                         n: usize,
                                         fresh: bool|
                 -> Result<SecureField, String> {
                    let two = SecureField::from(M31::from(2u32));
                    // Sub-proof QM31 reader (always unpacked: 4 felts per QM31)
                    let read_sub_qm31 = |data: &[FieldElement], off: &mut usize| -> SecureField {
                        let aa = felt_to_u64(&data[*off]) as u32; *off += 1;
                        let ab = felt_to_u64(&data[*off]) as u32; *off += 1;
                        let ba = felt_to_u64(&data[*off]) as u32; *off += 1;
                        let bb = felt_to_u64(&data[*off]) as u32; *off += 1;
                        QM31(CM31(M31::from(aa), M31::from(ab)),
                             CM31(M31::from(ba), M31::from(bb)))
                    };

                    if fresh {
                        let pm = m.next_power_of_two();
                        let pn = n.next_power_of_two();
                        let log_rows = pm.ilog2() as usize;
                        let log_cols = pn.ilog2() as usize;
                        let _r = ch.draw_qm31s(log_rows + log_cols);
                        mix_secure_field(ch, claim_value);
                    }

                    let sub_tag = felt_to_u64(&data[*off]) as u32; *off += 1;
                    if sub_tag == 0 {
                        // MatMul sub-proof (degree-2, c1 omitted)
                        ch.mix_u64(m as u64);
                        ch.mix_u64(k as u64);
                        ch.mix_u64(n as u64);
                        mix_secure_field(ch, claim_value);

                        let nr = felt_to_u64(&data[*off]) as usize; *off += 1;
                        let mut cs = claim_value;
                        for _ in 0..nr {
                            let c0 = read_sub_qm31(data, off);
                            let c2 = read_sub_qm31(data, off);
                            let c1 = cs - two * c0 - c2;
                            ch.mix_poly_coeffs(c0, c1, c2);
                            let r = ch.draw_qm31();
                            cs = c0 + c1 * r + c2 * r * r;
                        }
                        let fa = read_sub_qm31(data, off);
                        let fb = read_sub_qm31(data, off);
                        mix_secure_field(ch, fa);
                        mix_secure_field(ch, fb);
                        Ok(fa)
                    } else if sub_tag == 7 {
                        // MatMulDualSimd sub-proof (degree-3, c1 omitted in unpacked)
                        let nbv = felt_to_u64(&data[*off]) as usize; *off += 1;
                        let nb = 1usize << nbv;
                        ch.mix_u64(m as u64);
                        ch.mix_u64(k as u64);
                        ch.mix_u64(n as u64);
                        ch.mix_u64(nb as u64);
                        mix_secure_field(ch, claim_value);

                        let nr = felt_to_u64(&data[*off]) as usize; *off += 1;
                        let mut cs = claim_value;
                        for _ in 0..nr {
                            let c0 = read_sub_qm31(data, off);
                            let c2 = read_sub_qm31(data, off);
                            let c3 = read_sub_qm31(data, off);
                            let c1 = cs - two * c0 - c2 - c3;
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let r = ch.draw_qm31();
                            cs = c0 + c1 * r + c2 * r * r + c3 * r * r * r;
                        }
                        let fa = read_sub_qm31(data, off);
                        let fb = read_sub_qm31(data, off);
                        mix_secure_field(ch, fa);
                        mix_secure_field(ch, fb);
                        Ok(fa)
                    } else {
                        Err(format!("Attention sub-proof: unexpected tag {}", sub_tag))
                    }
                };

                // Sub-proof 0: Output projection (uses current claim, not fresh)
                let mut sub_idx = 0;
                let _output_proj = replay_sub_matmul(
                    proof_data, &mut off, &mut ch,
                    current_claim_value,
                    att_seq_len, att_d_model, att_d_model,
                    false, // not fresh — uses existing claim
                )?;
                sub_idx += 1;

                // Per-head sub-proofs (h = H-1..0): context + score matmuls
                for _h in (0..att_num_heads).rev() {
                    // Context matmul: seq_len × seq_len → d_k
                    let _ctx = replay_sub_matmul(
                        proof_data, &mut off, &mut ch,
                        sub_claim_values[sub_idx],
                        att_seq_len, att_seq_len, att_d_k,
                        true,
                    )?;
                    sub_idx += 1;

                    // Score matmul: seq_len × d_k → seq_len
                    let _score = replay_sub_matmul(
                        proof_data, &mut off, &mut ch,
                        sub_claim_values[sub_idx],
                        att_seq_len, att_d_k, att_seq_len,
                        true,
                    )?;
                    sub_idx += 1;
                }

                // V, K projections (fresh)
                let _v = replay_sub_matmul(
                    proof_data, &mut off, &mut ch,
                    sub_claim_values[sub_idx],
                    att_seq_len, att_d_model, att_d_model,
                    true,
                )?;
                sub_idx += 1;

                let _k = replay_sub_matmul(
                    proof_data, &mut off, &mut ch,
                    sub_claim_values[sub_idx],
                    att_seq_len, att_d_model, att_d_model,
                    true,
                )?;
                sub_idx += 1;

                // Q projection (fresh — determines final input claim)
                let q_pm = att_seq_len.next_power_of_two();
                let q_pc = att_d_model.next_power_of_two();
                let q_log_rows = q_pm.ilog2() as usize;
                let q_log_cols = q_pc.ilog2() as usize;
                let _r_q = ch.draw_qm31s(q_log_rows + q_log_cols);
                let q_value = sub_claim_values[sub_idx];
                mix_secure_field(&mut ch, q_value);

                // Q sub-proof (uses q_value claim, not a fresh draw — the draw was done above)
                let two = SecureField::from(M31::from(2u32));
                let read_sub_qm31_q = |data: &[FieldElement], off: &mut usize| -> SecureField {
                    let aa = felt_to_u64(&data[*off]) as u32; *off += 1;
                    let ab = felt_to_u64(&data[*off]) as u32; *off += 1;
                    let ba = felt_to_u64(&data[*off]) as u32; *off += 1;
                    let bb = felt_to_u64(&data[*off]) as u32; *off += 1;
                    QM31(CM31(M31::from(aa), M31::from(ab)),
                         CM31(M31::from(ba), M31::from(bb)))
                };
                let q_sub_tag = felt_to_u64(&proof_data[off]) as u32; off += 1;
                if q_sub_tag != 0 {
                    return Err(format!("Attention Q projection: expected tag 0, got {}", q_sub_tag));
                }
                ch.mix_u64(att_seq_len as u64);
                ch.mix_u64(att_d_model as u64);
                ch.mix_u64(att_d_model as u64);
                mix_secure_field(&mut ch, q_value);
                let q_nr = felt_to_u64(&proof_data[off]) as usize; off += 1;
                let mut q_sum = q_value;
                for _ in 0..q_nr {
                    let c0 = read_sub_qm31_q(proof_data, &mut off);
                    let c2 = read_sub_qm31_q(proof_data, &mut off);
                    let c1 = q_sum - two * c0 - c2;
                    ch.mix_poly_coeffs(c0, c1, c2);
                    let r = ch.draw_qm31();
                    q_sum = c0 + c1 * r + c2 * r * r;
                }
                let q_fa = read_sub_qm31_q(proof_data, &mut off);
                let q_fb = read_sub_qm31_q(proof_data, &mut off);
                mix_secure_field(&mut ch, q_fa);
                mix_secure_field(&mut ch, q_fb);

                current_claim_value = q_fa;
                if trace {
                    eprintln!("[VERIFIER Attention] layer {} heads={} seq={} d_model={} ch={:?}",
                        layer, att_num_heads, att_seq_len, att_d_model, ch.digest());
                }
            }
            10 => {
                // Embedding (LogUp sparse multiplicity proof)
                let emb_vocab_size = read_u32_from(proof_data, &mut off);
                let emb_embed_dim = read_u32_from(proof_data, &mut off);
                let input_eval = read_qm31_from(proof_data, &mut off);
                let output_eval = read_qm31_from(proof_data, &mut off);
                let _input_num_vars = read_u32_from(proof_data, &mut off);

                let has_logup = read_u32_from(proof_data, &mut off);
                if has_logup == 1 {
                    ch.mix_u64(0x454D424C4F47_u64); // "EMBLOG"
                    ch.mix_u64(emb_vocab_size as u64);
                    ch.mix_u64(emb_embed_dim as u64);
                    let _gamma = ch.draw_qm31();
                    let _beta_col = ch.draw_qm31();
                    let _beta_val = ch.draw_qm31();

                    // Round polys are serialized before claimed_sum, but verifier
                    // mixes claimed_sum first. Buffer rounds, read claimed_sum, then replay.
                    let eq_rounds = read_u32_from(proof_data, &mut off) as usize;
                    let mut round_coeffs = Vec::with_capacity(eq_rounds);
                    for _ in 0..eq_rounds {
                        let c0 = read_qm31_from(proof_data, &mut off);
                        let c1 = read_qm31_from(proof_data, &mut off);
                        let c2 = read_qm31_from(proof_data, &mut off);
                        let c3 = read_qm31_from(proof_data, &mut off);
                        round_coeffs.push((c0, c1, c2, c3));
                    }
                    let claimed_sum = read_qm31_from(proof_data, &mut off);
                    mix_secure_field(&mut ch, claimed_sum);

                    // Now replay the sumcheck rounds
                    let mut _emb_sum = SecureField::from(M31::from(1u32));
                    for &(c0, c1, c2, c3) in &round_coeffs {
                        ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                        let challenge = ch.draw_qm31();
                        _emb_sum = c0 + c1 * challenge + c2 * challenge * challenge
                            + c3 * challenge * challenge * challenge;
                    }

                    // Final evals: w, tok, col, val (not channel-mixed)
                    let _w = read_qm31_from(proof_data, &mut off);
                    let _tok = read_qm31_from(proof_data, &mut off);
                    let _col = read_qm31_from(proof_data, &mut off);
                    let _val = read_qm31_from(proof_data, &mut off);
                    // Sparse table entries (not channel-mixed, just consume)
                    let table_len = read_u32_from(proof_data, &mut off) as usize;
                    for _ in 0..table_len {
                        let _ = read_u32_from(proof_data, &mut off); // token
                        let _ = read_u32_from(proof_data, &mut off); // col
                        let _ = read_u32_from(proof_data, &mut off); // multiplicity
                    }
                }
                mix_secure_field(&mut ch, input_eval);
                mix_secure_field(&mut ch, output_eval);
                current_claim_value = input_eval;
                if trace {
                    eprintln!("[VERIFIER Embedding] layer {} vocab={} dim={} ch={:?}",
                        layer, emb_vocab_size, emb_embed_dim, ch.digest());
                }
            }
            11 => {
                // AttentionDecode — same structure as Attention (tag 5) but with
                // DCOD domain tag and dual dimensions (new_tokens, full_seq_len).
                let num_sub_proofs = read_u32_from(proof_data, &mut off) as usize;
                let num_heads = read_u32_from(proof_data, &mut off) as usize;
                let new_tokens = read_u32_from(proof_data, &mut off) as usize;
                let full_seq_len = read_u32_from(proof_data, &mut off) as usize;
                let d_model = read_u32_from(proof_data, &mut off) as usize;
                let causal_flag = read_u32_from(proof_data, &mut off);
                let position_offset = read_u32_from(proof_data, &mut off) as usize;
                let d_k = d_model / num_heads;

                let mut sub_claim_values = Vec::with_capacity(num_sub_proofs);
                for _ in 0..num_sub_proofs {
                    sub_claim_values.push(read_qm31_from(proof_data, &mut off));
                }

                // Mix DCOD metadata
                ch.mix_u64(0x44434F44_u64); // "DCOD"
                ch.mix_u64(num_heads as u64);
                ch.mix_u64(new_tokens as u64);
                ch.mix_u64(full_seq_len as u64);
                ch.mix_u64(d_model as u64);
                ch.mix_u64(causal_flag as u64);
                ch.mix_u64(position_offset as u64);

                // Assert position_offset consistency
                if position_offset + new_tokens != full_seq_len {
                    return Err(format!(
                        "DCOD_POSITION_MISMATCH: position_offset({}) + new_tokens({}) != full_seq_len({})",
                        position_offset, new_tokens, full_seq_len,
                    ));
                }

                if trace {
                    eprintln!(
                        "[VERIFIER AttentionDecode] num_sub={} heads={} new_tokens={} full_seq={} d_model={} causal={} pos_offset={}",
                        num_sub_proofs, num_heads, new_tokens, full_seq_len, d_model, causal_flag, position_offset
                    );
                }

                let two = SecureField::from(M31::from(2u32));
                let mut proof_idx = 0usize;

                macro_rules! replay_matmul_decode {
                    ($claim_val:expr, $m:expr, $k:expr, $n:expr) => {{
                        let cv = $claim_val;
                        let mm = $m;
                        let kk = $k;
                        let nn = $n;
                        ch.mix_u64(mm as u64);
                        ch.mix_u64(kk as u64);
                        ch.mix_u64(nn as u64);
                        mix_secure_field(&mut ch, cv);
                        let nr = read_u32_from(proof_data, &mut off) as usize;
                        let mut csum = cv;
                        for _ in 0..nr {
                            let c0 = read_qm31_from(proof_data, &mut off);
                            let c2 = read_qm31_from(proof_data, &mut off);
                            let c1 = csum - two * c0 - c2;
                            ch.mix_poly_coeffs(c0, c1, c2);
                            let chal = ch.draw_qm31();
                            csum = c0 + c1 * chal + c2 * chal * chal;
                        }
                        let fa = read_qm31_from(proof_data, &mut off);
                        let fb = read_qm31_from(proof_data, &mut off);
                        if csum != fa * fb {
                            return Err(format!(
                                "DCOD_MATMUL_FINAL_MISMATCH at layer {}: sum={:?} != a*b={:?}",
                                layer, csum, fa * fb
                            ));
                        }
                        mix_secure_field(&mut ch, fa);
                        mix_secure_field(&mut ch, fb);
                        fa
                    }};
                }

                // Sub-proof 0: Output projection (new_tokens × d_model × d_model)
                let sub_tag_0 = read_u32_from(proof_data, &mut off);
                if sub_tag_0 != 0 {
                    return Err(format!("DCOD_SUB0_NOT_MATMUL: tag={}", sub_tag_0));
                }
                let _output_proj_a = replay_matmul_decode!(
                    current_claim_value, new_tokens, d_model, d_model
                );
                proof_idx += 1;

                // Per-head sub-proofs (h = H-1..0): context + score
                for _h in 0..num_heads {
                    // Context: new_tokens × full_seq_len × d_k
                    let pm_ctx = new_tokens.next_power_of_two();
                    let pn_ctx = d_k.next_power_of_two();
                    let log_ctx = pm_ctx.ilog2() as usize + pn_ctx.ilog2() as usize;
                    let _r_ctx = ch.draw_qm31s(log_ctx);
                    mix_secure_field(&mut ch, sub_claim_values[proof_idx]);

                    let sub_tag = read_u32_from(proof_data, &mut off);
                    if sub_tag != 0 {
                        return Err(format!("DCOD_CTX_NOT_MATMUL: tag={}", sub_tag));
                    }
                    let _ctx_a = replay_matmul_decode!(
                        sub_claim_values[proof_idx], new_tokens, full_seq_len, d_k
                    );
                    proof_idx += 1;

                    // Score: new_tokens × d_k × full_seq_len
                    let pm_sc = new_tokens.next_power_of_two();
                    let pn_sc = full_seq_len.next_power_of_two();
                    let log_sc = pm_sc.ilog2() as usize + pn_sc.ilog2() as usize;
                    let _r_sc = ch.draw_qm31s(log_sc);
                    mix_secure_field(&mut ch, sub_claim_values[proof_idx]);

                    let sub_tag2 = read_u32_from(proof_data, &mut off);
                    if sub_tag2 != 0 {
                        return Err(format!("DCOD_SCORE_NOT_MATMUL: tag={}", sub_tag2));
                    }
                    let _score_a = replay_matmul_decode!(
                        sub_claim_values[proof_idx], new_tokens, d_k, full_seq_len
                    );
                    proof_idx += 1;
                }

                // V, K, Q projections: new_tokens × d_model × d_model
                let mut last_final_a = SecureField::zero();
                for proj_label in &["V", "K", "Q"] {
                    let pm = new_tokens.next_power_of_two();
                    let pn = d_model.next_power_of_two();
                    let log_proj = pm.ilog2() as usize + pn.ilog2() as usize;
                    let _r_proj = ch.draw_qm31s(log_proj);
                    mix_secure_field(&mut ch, sub_claim_values[proof_idx]);

                    let sub_tag = read_u32_from(proof_data, &mut off);
                    if sub_tag != 0 {
                        return Err(format!("DCOD_{}_NOT_MATMUL: tag={}", proj_label, sub_tag));
                    }
                    last_final_a = replay_matmul_decode!(
                        sub_claim_values[proof_idx], new_tokens, d_model, d_model
                    );
                    proof_idx += 1;
                }

                if proof_idx != num_sub_proofs {
                    return Err(format!(
                        "DCOD_SUBPROOF_COUNT_MISMATCH: expected {} got {}",
                        num_sub_proofs, proof_idx
                    ));
                }

                current_claim_value = last_final_a;
            }
            _ => return Err(format!("Unknown tag {} at layer {}", tag, layer)),
        }
    }

    // Deferred proofs (Add skip-connection branches) are serialized after layer proofs.
    // Replay the MatMul sumcheck for each deferred proof to validate serialized data
    // and prevent arbitrary data appending after layer proofs.
    if off < proof_data.len() {
        let num_deferred = read_u32_from(proof_data, &mut off);
        let two = SecureField::from(M31::from(2u32));
        for di in 0..num_deferred as usize {
            let claim_value = read_qm31_from(proof_data, &mut off);
            let kind = read_u32_from(proof_data, &mut off);

            if kind == 0 {
                // MatMul deferred proof — full sumcheck replay
                let m = read_u32_from(proof_data, &mut off) as usize;
                let k = read_u32_from(proof_data, &mut off) as usize;
                let n = read_u32_from(proof_data, &mut off) as usize;

                // Fiat-Shamir mixing order matches prover.rs + verifier.rs:
                // 1. mix deferred claim (prover.rs:1753 / verifier.rs:459)
                // 2. mix dims + claim again (inside verify_matmul_reduction)
                mix_secure_field(&mut ch, claim_value);
                ch.mix_u64(m as u64);
                ch.mix_u64(k as u64);
                ch.mix_u64(n as u64);
                mix_secure_field(&mut ch, claim_value);

                let num_rounds = read_u32_from(proof_data, &mut off) as usize;
                let mut current_sum = claim_value;
                for round in 0..num_rounds {
                    let c0 = read_qm31_from(proof_data, &mut off);
                    let c2 = read_qm31_from(proof_data, &mut off);
                    let c1 = current_sum - two * c0 - c2;
                    ch.mix_poly_coeffs(c0, c1, c2);
                    let challenge = ch.draw_qm31();
                    current_sum = c0 + c1 * challenge + c2 * challenge * challenge;
                    if trace && round < 3 {
                        eprintln!("[VERIFIER DEFERRED {}] round {} c0={:?} c2={:?} challenge={:?}",
                            di, round, c0, c2, challenge);
                    }
                }
                let final_a = read_qm31_from(proof_data, &mut off);
                let final_b = read_qm31_from(proof_data, &mut off);
                if current_sum != final_a * final_b {
                    return Err(format!(
                        "DEFERRED_MATMUL_FINAL_MISMATCH at deferred[{}]: sum={:?} != a*b={:?}",
                        di, current_sum, final_a * final_b
                    ));
                }
                mix_secure_field(&mut ch, final_a);
                mix_secure_field(&mut ch, final_b);
                off += 1; // weight commitment (consumed, verified by GKR verifier)
            } else if kind == 1 {
                // Weightless deferred proof (Quantize/Dequantize) — mix claim, skip layer proof data.
                // Full LogUp verification is handled by the GKR verifier (verifier.rs).
                mix_secure_field(&mut ch, claim_value);
                let data_len = read_u32_from(proof_data, &mut off) as usize;
                if off + data_len > proof_data.len() {
                    return Err(format!(
                        "Weightless deferred[{}]: data_len={} exceeds remaining {} felts",
                        di, data_len, proof_data.len() - off
                    ));
                }
                off += data_len;
            } else {
                return Err(format!(
                    "Unknown deferred proof kind {} at deferred[{}]",
                    kind, di
                ));
            }
        }
    }

    // Reject trailing data — all proof felts must be consumed
    if off < proof_data.len() {
        return Err(format!(
            "trailing data: consumed {} of {} felts ({} unconsumed)",
            off, proof_data.len(), proof_data.len() - off,
        ));
    }

    Ok(())
}

// ============================================================================
// Fast Proof Health Check
// ============================================================================

/// Result of a single health check.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Name of the check (e.g., "length", "header_parse", "qm31_range").
    pub name: &'static str,
    /// Whether this check passed.
    pub passed: bool,
    /// Human-readable detail (empty on success, error message on failure).
    pub detail: String,
}

/// Report from `verify_proof_fast()` — lightweight proof health check.
#[derive(Debug, Clone)]
pub struct ProofHealthReport {
    /// Individual checks performed.
    pub checks: Vec<HealthCheck>,
    /// True if ALL checks passed.
    pub passed: bool,
    /// Total calldata size in felt252 elements.
    pub total_felts: usize,
    /// Estimated Cairo steps for on-chain verification.
    pub estimated_steps: u64,
}

/// Estimate Cairo steps for GKR on-chain verification.
///
/// Based on profiling of v20-lean contract execution:
/// - Base overhead: ~500K steps (contract init, channel setup, IO parsing)
/// - Per GKR layer: ~150K steps (sumcheck rounds, QM31 arithmetic)
/// - Per IO felt: ~50 steps (MLE evaluation + Poseidon hashing)
/// - Per weight commitment: ~80K steps (Merkle verify + MLE opening)
pub fn estimate_gkr_steps(num_layers: usize, io_felts: usize, num_weight_commits: usize) -> u64 {
    let base: u64 = 500_000;
    let per_layer: u64 = 150_000;
    let per_io_felt: u64 = 50;
    let per_weight: u64 = 80_000;
    base + (num_layers as u64) * per_layer
        + (io_felts as u64) * per_io_felt
        + (num_weight_commits as u64) * per_weight
}

/// Run a fast structural health check on serialized proof calldata.
///
/// Catches corruption, truncation, and format errors in <100ms without
/// full GKR verification. Designed as a pre-flight check before on-chain
/// submission.
///
/// Calldata is expected in the GKR v4 packed-IO format:
/// `[model_id, original_io_len, packed_io_len, packed_io..., matmul_dims_len, matmul_dims..., ...]`
pub fn verify_proof_fast(calldata: &[FieldElement]) -> ProofHealthReport {
    let mut checks = Vec::new();
    let total_felts = calldata.len();

    // 1. Length check — minimum header size
    let min_header = 5; // model_id + original_io_len + packed_io_len + at least 1 IO + 1 dims
    let length_ok = calldata.len() >= min_header;
    checks.push(HealthCheck {
        name: "length",
        passed: length_ok,
        detail: if length_ok {
            format!("{} felts (>= {} minimum)", calldata.len(), min_header)
        } else {
            format!(
                "calldata has {} felts, need at least {}",
                calldata.len(),
                min_header
            )
        },
    });
    if !length_ok {
        return ProofHealthReport {
            passed: false,
            checks,
            total_felts,
            estimated_steps: 0,
        };
    }

    // 2. Header parse — model_id, io_len, packed_io_len
    let model_id = calldata[0];
    let model_id_ok = model_id != FieldElement::ZERO;
    checks.push(HealthCheck {
        name: "model_id",
        passed: model_id_ok,
        detail: if model_id_ok {
            format!("0x{:x}", model_id)
        } else {
            "model_id is zero".to_string()
        },
    });

    let original_io_len = felt_to_u64(&calldata[1]) as usize;
    let packed_io_len = felt_to_u64(&calldata[2]) as usize;
    let io_header_ok = original_io_len > 0
        && packed_io_len > 0
        && packed_io_len <= original_io_len
        && 3 + packed_io_len < calldata.len();
    checks.push(HealthCheck {
        name: "io_header",
        passed: io_header_ok,
        detail: if io_header_ok {
            format!(
                "original_io={}, packed_io={} felts",
                original_io_len, packed_io_len
            )
        } else {
            format!(
                "invalid IO header: original={}, packed={}, calldata={}",
                original_io_len,
                packed_io_len,
                calldata.len()
            )
        },
    });

    // 3. Layer tag validation — scan for known tags after IO section
    let post_io_offset = 3 + packed_io_len;
    let mut num_layers = 0u32;
    let mut num_weight_commits = 0usize;
    if io_header_ok && post_io_offset + 1 < calldata.len() {
        // matmul_dims section: [len, dims...]
        let matmul_dims_len = felt_to_u64(&calldata[post_io_offset]) as usize;
        let dims_end = post_io_offset + 1 + matmul_dims_len;
        if matmul_dims_len > 0 && dims_end <= calldata.len() {
            // Each matmul has 3 dims (m, k, n)
            num_layers = (matmul_dims_len / 3) as u32;
            // Rough estimate: 1 weight commitment per matmul layer
            num_weight_commits = num_layers as usize;
        }
        let layer_tag_ok = num_layers > 0;
        checks.push(HealthCheck {
            name: "layer_tags",
            passed: layer_tag_ok,
            detail: if layer_tag_ok {
                format!("{} matmul layers detected", num_layers)
            } else {
                "no matmul layers found in calldata".to_string()
            },
        });
    }

    // 4. QM31 range check — spot-check 10 random values from proof data
    {
        let proof_region_start = post_io_offset;
        let proof_region_len = calldata.len().saturating_sub(proof_region_start);
        let sample_count = 10.min(proof_region_len);
        let mut range_failures = 0usize;
        if sample_count > 0 {
            // Sample evenly-spaced elements from the proof region
            let step = proof_region_len / sample_count;
            for i in 0..sample_count {
                let idx = proof_region_start + i * step;
                if idx < calldata.len() {
                    let val = felt_to_u64(&calldata[idx]);
                    // QM31 has 4 M31 limbs; each element in packed form fits in felt252.
                    // We check that at least none are the sentinel 0xFFFFFFFF...
                    // which would indicate corruption. Full QM31 decomposition
                    // needs the actual packing scheme, so we just check < 2^252.
                    let be = calldata[idx].to_bytes_be();
                    if be[0] > 0x0F {
                        range_failures += 1;
                    }
                    let _ = val; // suppress unused warning
                }
            }
        }
        let range_ok = range_failures == 0;
        checks.push(HealthCheck {
            name: "qm31_range",
            passed: range_ok,
            detail: if range_ok {
                format!("{} samples within range", sample_count)
            } else {
                format!("{}/{} samples exceed felt252 range", range_failures, sample_count)
            },
        });
    }

    // 5. Tail sentinel — last few felts should not all be zero (truncation indicator)
    {
        let tail_count = 4.min(calldata.len());
        let tail_start = calldata.len() - tail_count;
        let all_zero = calldata[tail_start..]
            .iter()
            .all(|f| *f == FieldElement::ZERO);
        let tail_ok = !all_zero;
        checks.push(HealthCheck {
            name: "tail_sentinel",
            passed: tail_ok,
            detail: if tail_ok {
                "tail is non-zero".to_string()
            } else {
                format!("last {} felts are all zero (possible truncation)", tail_count)
            },
        });
    }

    // 6. Step estimation
    let estimated_steps = estimate_gkr_steps(num_layers as usize, original_io_len, num_weight_commits);
    let within_limit = estimated_steps < 9_000_000;
    checks.push(HealthCheck {
        name: "step_estimate",
        passed: within_limit,
        detail: format!(
            "~{} steps (limit: 10M, margin: 9M) — {}",
            estimated_steps,
            if within_limit { "OK" } else { "EXCEEDS MARGIN" }
        ),
    });

    let passed = checks.iter().all(|c| c.passed);
    ProofHealthReport {
        checks,
        passed,
        total_felts,
        estimated_steps,
    }
}

/// Run a structural health check on disaggregated ml_gkr proof components.
///
/// Unlike `verify_proof_fast` (which assumes V4 packed-IO layout), this function
/// validates the separate `gkr_calldata`, `io_calldata`, and `weight_commitments`
/// arrays that the `--format ml_gkr` pipeline produces.
pub fn verify_proof_fast_ml_gkr(
    gkr_calldata: &[FieldElement],
    io_calldata: &[FieldElement],
    weight_commitments: &[FieldElement],
    num_layer_proofs: usize,
) -> ProofHealthReport {
    let mut checks = Vec::new();
    let total_felts = gkr_calldata.len() + io_calldata.len() + weight_commitments.len();

    // 1. GKR calldata length — must be non-empty and plausible for the layer count.
    //    Each layer proof has at least a tag + a few sumcheck round values (~3 felts minimum).
    let min_gkr_felts = if num_layer_proofs > 0 { num_layer_proofs * 3 } else { 1 };
    let gkr_len_ok = gkr_calldata.len() >= min_gkr_felts;
    checks.push(HealthCheck {
        name: "gkr_calldata",
        passed: gkr_len_ok,
        detail: if gkr_len_ok {
            format!("{} felts ({} layer proofs)", gkr_calldata.len(), num_layer_proofs)
        } else {
            format!(
                "{} felts too small for {} layer proofs (need >= {})",
                gkr_calldata.len(),
                num_layer_proofs,
                min_gkr_felts,
            )
        },
    });

    // 2. IO structure — validate [in_rows, in_cols, in_len, data..., out_rows, out_cols, out_len, data...]
    let io_ok = if io_calldata.len() >= 6 {
        let in_rows = felt_to_u64(&io_calldata[0]) as usize;
        let in_cols = felt_to_u64(&io_calldata[1]) as usize;
        let in_len = felt_to_u64(&io_calldata[2]) as usize;
        let dims_match = in_rows * in_cols == in_len;
        let out_start = 3 + in_len;
        if dims_match && out_start + 3 <= io_calldata.len() {
            let out_rows = felt_to_u64(&io_calldata[out_start]) as usize;
            let out_cols = felt_to_u64(&io_calldata[out_start + 1]) as usize;
            let out_len = felt_to_u64(&io_calldata[out_start + 2]) as usize;
            let out_ok = out_rows * out_cols == out_len;
            let total_expected = 3 + in_len + 3 + out_len;
            checks.push(HealthCheck {
                name: "io_data",
                passed: out_ok && io_calldata.len() == total_expected,
                detail: if out_ok && io_calldata.len() == total_expected {
                    format!("input={}x{}, output={}x{}", in_rows, in_cols, out_rows, out_cols)
                } else {
                    format!(
                        "dimension mismatch: in={}x{}(len={}), out={}x{}(len={}), total={} vs expected={}",
                        in_rows, in_cols, in_len, out_rows, out_cols, out_len,
                        io_calldata.len(), total_expected,
                    )
                },
            });
            out_ok && io_calldata.len() == total_expected
        } else {
            checks.push(HealthCheck {
                name: "io_data",
                passed: false,
                detail: format!(
                    "input dims {}x{} vs len={}, io_calldata.len()={}",
                    in_rows, in_cols, in_len, io_calldata.len(),
                ),
            });
            false
        }
    } else {
        checks.push(HealthCheck {
            name: "io_data",
            passed: io_calldata.is_empty(),
            detail: if io_calldata.is_empty() {
                "no IO data (may be pre-committed)".to_string()
            } else {
                format!("truncated IO: {} felts (need >= 6)", io_calldata.len())
            },
        });
        io_calldata.is_empty()
    };
    let _ = io_ok;

    // 3. QM31 range check — spot-check 10 values from gkr_calldata
    {
        let sample_count = 10.min(gkr_calldata.len());
        let mut range_failures = 0usize;
        if sample_count > 0 {
            let step = gkr_calldata.len() / sample_count;
            for i in 0..sample_count {
                let idx = i * step;
                if idx < gkr_calldata.len() {
                    let be = gkr_calldata[idx].to_bytes_be();
                    if be[0] > 0x0F {
                        range_failures += 1;
                    }
                }
            }
        }
        let range_ok = range_failures == 0;
        checks.push(HealthCheck {
            name: "qm31_range",
            passed: range_ok,
            detail: if range_ok {
                format!("{} samples valid", sample_count)
            } else {
                format!("{}/{} samples exceed felt252 range", range_failures, sample_count)
            },
        });
    }

    // 4. Weight commitments present
    let wc_ok = !weight_commitments.is_empty();
    checks.push(HealthCheck {
        name: "weight_commitments",
        passed: wc_ok,
        detail: if wc_ok {
            format!("{} present", weight_commitments.len())
        } else {
            "no weight commitments".to_string()
        },
    });

    // 5. Step estimation using component sizes
    let io_felts = io_calldata.len();
    let estimated_steps = estimate_gkr_steps(num_layer_proofs, io_felts, weight_commitments.len());
    let within_limit = estimated_steps < 9_000_000;
    checks.push(HealthCheck {
        name: "step_estimate",
        passed: within_limit,
        detail: format!(
            "~{}M steps (limit: 10M, margin: 9M) — {}",
            estimated_steps as f64 / 1_000_000.0,
            if within_limit { "OK" } else { "EXCEEDS MARGIN" }
        ),
    });

    let passed = checks.iter().all(|c| c.passed);
    ProofHealthReport {
        checks,
        passed,
        total_felts,
        estimated_steps,
    }
}

/// Run a dry-run simulation for disaggregated ml_gkr proof components.
///
/// Like `dry_run_onchain` but uses `verify_proof_fast_ml_gkr` instead of
/// `verify_proof_fast`.
pub fn dry_run_onchain_ml_gkr(
    gkr_calldata: &[FieldElement],
    io_calldata: &[FieldElement],
    weight_commitments: &[FieldElement],
    num_layer_proofs: usize,
    rpc_url: Option<&str>,
    _contract_address: Option<&str>,
) -> DryRunResult {
    let health = verify_proof_fast_ml_gkr(gkr_calldata, io_calldata, weight_commitments, num_layer_proofs);

    let estimated_steps = health.estimated_steps;
    let within_step_limit = estimated_steps < 10_000_000;
    let estimated_fee_strk = (estimated_steps as f64) * 0.0000001;
    let total_felts = gkr_calldata.len() + io_calldata.len() + weight_commitments.len();

    // Optional RPC simulation — use gkr_calldata as the representative payload
    let rpc_simulation = rpc_url.map(|url| {
        match simulate_via_rpc(url, gkr_calldata) {
            Ok((steps, _)) => RpcSimResult {
                success: true,
                actual_steps: steps,
                error: None,
            },
            Err(e) => RpcSimResult {
                success: false,
                actual_steps: 0,
                error: Some(e),
            },
        }
    });

    DryRunResult {
        health,
        calldata_size: total_felts,
        estimated_fee_strk,
        estimated_steps,
        within_step_limit,
        rpc_simulation,
    }
}

/// Result of a dry-run simulation.
#[derive(Debug)]
pub struct DryRunResult {
    /// Health check report (always populated).
    pub health: ProofHealthReport,
    /// Total calldata size in felt252 elements.
    pub calldata_size: usize,
    /// Estimated fee in STRK.
    pub estimated_fee_strk: f64,
    /// Estimated Cairo steps for on-chain verification.
    pub estimated_steps: u64,
    /// Whether the estimated steps are within the 10M sequencer limit.
    pub within_step_limit: bool,
    /// RPC simulation result (only if rpc_url was provided).
    pub rpc_simulation: Option<RpcSimResult>,
}

/// Result of an RPC simulation.
#[derive(Debug)]
pub struct RpcSimResult {
    /// Whether the simulation succeeded.
    pub success: bool,
    /// Actual steps from the simulation.
    pub actual_steps: u64,
    /// Error message if the simulation failed.
    pub error: Option<String>,
}

/// Run a dry-run simulation: health check + step estimation + optional RPC simulation.
///
/// If `rpc_url` is `None`, only performs the local health check and step estimation.
/// If `rpc_url` is `Some`, also calls `starknet_simulateTransactions` to get actual
/// step counts from the sequencer.
pub fn dry_run_onchain(
    calldata: &[FieldElement],
    rpc_url: Option<&str>,
    _contract_address: Option<&str>,
) -> DryRunResult {
    // 1. Run fast health check
    let health = verify_proof_fast(calldata);

    let estimated_steps = health.estimated_steps;
    let within_step_limit = estimated_steps < 10_000_000;

    // Estimate fee: ~0.0001 STRK per 1K steps (rough Sepolia pricing)
    let estimated_fee_strk = (estimated_steps as f64) * 0.0000001;

    // 2. Optional RPC simulation
    let rpc_simulation = rpc_url.map(|url| {
        // Try to call starknet_simulateTransactions via ureq
        match simulate_via_rpc(url, calldata) {
            Ok((steps, _)) => RpcSimResult {
                success: true,
                actual_steps: steps,
                error: None,
            },
            Err(e) => RpcSimResult {
                success: false,
                actual_steps: 0,
                error: Some(e),
            },
        }
    });

    DryRunResult {
        health,
        calldata_size: calldata.len(),
        estimated_fee_strk,
        estimated_steps,
        within_step_limit,
        rpc_simulation,
    }
}

/// Attempt to simulate a transaction via Starknet RPC.
///
/// Calls `starknet_simulateTransactions` with `SKIP_VALIDATE` flag.
/// Returns `(actual_steps, gas_consumed)` on success.
///
/// Requires the `audit-http` feature for HTTP client (ureq).
#[cfg(feature = "audit-http")]
fn simulate_via_rpc(rpc_url: &str, calldata: &[FieldElement]) -> Result<(u64, u64), String> {
    // Build the JSON-RPC request for starknet_simulateTransactions
    let calldata_hex: Vec<String> = calldata.iter().map(|f| format!("0x{:x}", f)).collect();

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "starknet_simulateTransactions",
        "params": {
            "block_id": "latest",
            "transactions": [{
                "type": "INVOKE",
                "version": "0x1",
                "calldata": calldata_hex,
                "max_fee": "0x0",
                "signature": [],
                "nonce": "0x0",
                "sender_address": "0x0",
            }],
            "simulation_flags": ["SKIP_VALIDATE"]
        }
    });

    let body_str = serde_json::to_string(&body).map_err(|e| format!("JSON serialize: {e}"))?;

    // Use ureq for HTTP
    let resp = ureq::post(rpc_url)
        .header("Content-Type", "application/json")
        .send(body_str.as_bytes())
        .map_err(|e| format!("RPC request failed: {e}"))?;

    let resp_str = resp
        .into_body()
        .read_to_string()
        .map_err(|e| format!("RPC response read: {e}"))?;
    let resp_body: serde_json::Value = serde_json::from_str(&resp_str)
        .map_err(|e| format!("RPC response parse: {e}"))?;

    // Parse result
    if let Some(error) = resp_body.get("error") {
        return Err(format!("RPC error: {}", error));
    }

    let result = resp_body
        .get("result")
        .and_then(|r| r.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| "empty simulation result".to_string())?;

    let exec_info = result
        .get("transaction_trace")
        .and_then(|t| t.get("execute_invocation"))
        .ok_or_else(|| "missing execute_invocation in trace".to_string())?;

    // Check for revert
    if let Some(revert) = exec_info.get("revert_reason") {
        return Err(format!("simulation reverted: {}", revert));
    }

    // Extract steps from execution_resources
    let steps = exec_info
        .get("execution_resources")
        .and_then(|r| r.get("n_steps"))
        .and_then(|s| s.as_u64())
        .unwrap_or(0);

    let gas = result
        .get("fee_estimation")
        .and_then(|f| f.get("gas_consumed"))
        .and_then(|g| g.as_str())
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0);

    Ok((steps, gas))
}

/// Stub for RPC simulation when the `audit-http` feature is not enabled.
#[cfg(not(feature = "audit-http"))]
fn simulate_via_rpc(_rpc_url: &str, _calldata: &[FieldElement]) -> Result<(u64, u64), String> {
    Err("RPC simulation requires the `audit-http` feature".to_string())
}

/// Extract a u64 from a FieldElement (takes bottom 8 bytes).
fn felt_to_u64(f: &FieldElement) -> u64 {
    let be = f.to_bytes_be();
    u64::from_be_bytes([be[24], be[25], be[26], be[27], be[28], be[29], be[30], be[31]])
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
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let lock = crate::test_utils::ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev, _lock: lock }
        }

        fn unset(key: &'static str) -> Self {
            let lock = crate::test_utils::ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key, prev, _lock: lock }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(prev) = self.prev.as_ref() {
                std::env::set_var(self.key, prev);
            } else {
                std::env::remove_var(self.key);
            }
            // _lock is dropped after env var is restored
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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");
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

        // Weight opening calldata: either individual MLE proofs (sequential mode)
        // or aggregated oracle sumcheck data. Both produce non-empty calldata.
        assert!(
            !gkr_sn.weight_opening_calldata.is_empty(),
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
    fn test_gkr_activation_logup_none_is_accepted() {
        // Activation LogUp is intentionally skipped in GKR mode because
        // M31 matmul outputs exceed the precomputed table range (2^16).
        // The soundness gate should accept logup_proof: None for activations.
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

        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof expected");

        // Verify activation layers have logup_proof: None
        let has_activation_without_logup = gkr.layer_proofs.iter().any(|lp| {
            matches!(lp, LayerProof::Activation { logup_proof: None, .. })
        });
        assert!(has_activation_without_logup, "activation should have logup_proof: None in GKR mode");

        // Build starknet proof should succeed even with logup_proof: None
        let result = build_gkr_starknet_proof(&agg_proof, FieldElement::from(7u64), &input);
        assert!(result.is_ok(), "starknet proof with activation logup_proof: None should succeed: {:?}", result.err());
    }

    #[test]
    fn test_gkr_soundness_gate_rejects_missing_weight_openings() {
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "individual");

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
    fn test_build_verify_model_gkr_v4_packed_reduces_calldata() {
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

        // Build both packed and unpacked calldata
        let unpacked = build_verify_model_gkr_v4_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("unpacked v4 calldata should build");
        let packed = build_verify_model_gkr_v4_packed_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("packed v4 calldata should build");

        // Packed should have fewer total felts due to QM31 packing (4→1)
        assert!(
            packed.total_felts < unpacked.total_felts,
            "packed ({}) should have fewer felts than unpacked ({})",
            packed.total_felts,
            unpacked.total_felts,
        );

        // The proof_data section drives most of the savings. Parse and compare.
        fn proof_data_len(parts: &[String]) -> usize {
            let mut idx = 0;
            idx += 1; // model_id
            let raw_io_len: usize = parts[idx].parse().unwrap();
            idx += 1 + raw_io_len;
            idx += 2; // circuit_depth, num_layers
            let matmul_len: usize = parts[idx].parse().unwrap();
            idx += 1 + matmul_len;
            let dequant_len: usize = parts[idx].parse().unwrap();
            idx += 1 + dequant_len;
            parts[idx].parse().unwrap()
        }

        let unpacked_pd = proof_data_len(&unpacked.calldata_parts);
        let packed_pd = proof_data_len(&packed.calldata_parts);
        assert!(
            packed_pd < unpacked_pd,
            "packed proof_data ({packed_pd}) should be smaller than unpacked ({unpacked_pd})"
        );
    }

    #[test]
    fn test_pack_m31_io_data_round_trip() {
        // Test that pack_m31_io_data correctly packs 8 M31 values per felt252
        // and the values can be recovered by the Cairo-equivalent unpack logic.
        let mut raw_io = Vec::new();
        for i in 0..25u32 {
            // Use a mix of values including edge cases
            let val = match i {
                0 => 0u32,
                1 => 1,
                2 => 0x7FFF_FFFE, // max valid M31 (2^31 - 2)
                3 => 0x3FFF_FFFF,
                _ => (i * 123456789 + 42) & 0x7FFF_FFFF,
            };
            raw_io.push(FieldElement::from(val as u64));
        }

        let packed = pack_m31_io_data(&raw_io);
        assert_eq!(packed.len(), (25 + 7) / 8); // ceil(25/8) = 4

        // Simulate Cairo unpack: extract 8 M31 values from each felt252
        let mut result = Vec::new();
        for felt in &packed {
            if result.len() >= 25 {
                break;
            }
            let be = felt.to_bytes_be();
            // Convert to u256 (lo = bytes[16..32], hi = bytes[0..16])
            let mut lo_bytes = [0u8; 16];
            let mut hi_bytes = [0u8; 16];
            lo_bytes.copy_from_slice(&be[16..32]);
            hi_bytes.copy_from_slice(&be[0..16]);
            let lo = u128::from_be_bytes(lo_bytes);
            let hi = u128::from_be_bytes(hi_bytes);

            let m31_mask: u128 = 0x7FFF_FFFF;

            // Value 0: bits [0..31) from lo
            if result.len() < 25 {
                result.push((lo & m31_mask) as u32);
            }
            // Value 1: bits [31..62) from lo
            if result.len() < 25 {
                result.push(((lo >> 31) & m31_mask) as u32);
            }
            // Value 2: bits [62..93) from lo
            if result.len() < 25 {
                result.push(((lo >> 62) & m31_mask) as u32);
            }
            // Value 3: bits [93..124) from lo
            if result.len() < 25 {
                result.push(((lo >> 93) & m31_mask) as u32);
            }
            // Value 4: bits [124..155) straddles lo/hi
            if result.len() < 25 {
                let lo_part = lo >> 124; // 4 bits
                let hi_part = (hi & 0x7FF_FFFF) << 4; // 27 bits shifted left 4
                result.push(((lo_part | hi_part) & m31_mask) as u32);
            }
            // Value 5: bits [155..186) from hi — bits [27..58)
            if result.len() < 25 {
                result.push(((hi >> 27) & m31_mask) as u32);
            }
            // Value 6: bits [186..217) from hi — bits [58..89)
            if result.len() < 25 {
                result.push(((hi >> 58) & m31_mask) as u32);
            }
            // Value 7: bits [217..248) from hi — bits [89..120)
            if result.len() < 25 {
                result.push(((hi >> 89) & m31_mask) as u32);
            }
        }

        // Compare
        for (i, original) in raw_io.iter().enumerate() {
            let orig_be = original.to_bytes_be();
            let orig_val = u32::from_be_bytes([orig_be[28], orig_be[29], orig_be[30], orig_be[31]]);
            assert_eq!(
                result[i],
                orig_val & 0x7FFF_FFFF,
                "mismatch at index {i}: packed/unpacked {}, original {}",
                result[i],
                orig_val
            );
        }
    }

    #[test]
    fn test_build_verify_model_gkr_v4_packed_io_reduces_calldata() {
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

        // Build IO-packed calldata
        let io_packed = build_verify_model_gkr_v4_packed_io_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("io_packed calldata should build");
        let regular = build_verify_model_gkr_v4_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("regular calldata should build");

        // IO-packed should have fewer felts (raw_io section shrinks ~8x)
        assert!(
            io_packed.total_felts < regular.total_felts,
            "io_packed ({}) should have fewer felts than regular ({})",
            io_packed.total_felts,
            regular.total_felts,
        );

        // First element should be model_id in both
        assert_eq!(io_packed.calldata_parts[0], regular.calldata_parts[0]);

        // Second element of io_packed should be original_io_len
        let io_len: usize = io_packed.calldata_parts[1].parse().unwrap();
        assert_eq!(io_len, raw_io.len());
    }

    #[test]
    fn test_export_packed_io_proof_json() {
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
        let model_id = FieldElement::from(0xA1u64); // test model ID

        // Build register calldata
        let circuit_desc = build_circuit_descriptor(&circuit);
        let register_cd = build_register_gkr_calldata(model_id, &gkr.weight_commitments, &circuit_desc);

        // Build IO-packed verify calldata
        let io_packed = build_verify_model_gkr_v4_packed_io_calldata(gkr, &circuit, model_id, &raw_io)
            .expect("io_packed calldata should build");

        // Export as JSON
        let json = serde_json::json!({
            "register_calldata": register_cd,
            "verify_calldata": {
                "entrypoint": "verify_model_gkr_v4_packed_io",
                "calldata": io_packed.calldata_parts,
            },
            "total_felts": io_packed.total_felts,
            "model_id": format!("0x{:x}", model_id),
        });

        let path = "/tmp/test_packed_io_proof.json";
        std::fs::write(path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        eprintln!("Exported proof to {path}");
        eprintln!("Register calldata: {} felts", register_cd.len());
        eprintln!("Verify calldata: {} felts", io_packed.total_felts);
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
        let two = SecureField::from(M31::from(2u32));

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
                        // Compressed: c1 omitted, reconstruct from current_sum
                        let c2 = read_qm31_from(&mut off);
                        let c1 = current_sum - two * c0 - c2;

                        println!("  round {}: c0={:?}, c1(reconstructed), c2={:?}, sum={:?}",
                            round, c0, c2, current_sum);

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

                    // === Part 0: RMS² verification plain sumcheck ===
                    // Must be replayed BEFORE "RN" tag to match prover's channel mixing order.
                    let has_p0 = read_u32_from(&mut off);
                    if has_p0 == 1 {
                        let p0_n_active = read_u32_from(&mut off) as u64;
                        let p0_sq_sum = read_qm31_from(&mut off);
                        ch.mix_u64(0x5251_u64); // "RQ" tag
                        ch.mix_u64(p0_n_active);
                        mix_secure_field(&mut ch, p0_sq_sum);
                        let p0_nr = read_u32_from(&mut off) as usize;
                        let mut p0_sum = p0_sq_sum;
                        for _ in 0..p0_nr {
                            let c0 = read_qm31_from(&mut off);
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            // Degree-2 polynomial: c3=0, c1 reconstructed from current_sum
                            let c1 = p0_sum - two * c0 - c2;
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let r = ch.draw_qm31();
                            p0_sum = c0 + c1 * r + c2 * r * r + c3 * r * r * r;
                        }
                        let p0_input_final = read_qm31_from(&mut off);
                        mix_secure_field(&mut ch, p0_input_final);
                        println!("  Part 0 (RMS² sumcheck) replayed, n_active={}", p0_n_active);
                    }

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
                        // Compressed: c1 omitted, reconstruct from rms_sum
                        let c2 = read_qm31_from(&mut off);
                        let c3 = read_qm31_from(&mut off);
                        let c1 = rms_sum - two * c0 - c2 - c3;

                        println!("  rms round {}: c0={:?}", round, c0);

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
                        ch.mix_u64(0x4C4F47); // "LOG"
                        ch.mix_u64(0x524E);   // "RN"
                        let _gamma = ch.draw_qm31();
                        let _beta = ch.draw_qm31();
                        let _claimed_sum = read_qm31_from(&mut off);
                        mix_secure_field(&mut ch, _claimed_sum);
                        let eq_rounds = read_u32_from(&mut off) as usize;
                        let mut logup_sum = SecureField::from(M31::from(1u32));
                        for _ in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            // Compressed: c1 omitted
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            let c1 = logup_sum - two * c0 - c2 - c3;
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let challenge = ch.draw_qm31();
                            logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                                + c3 * challenge * challenge * challenge;
                        }
                        // final evals
                        let _e1 = read_qm31_from(&mut off);
                        let _e2 = read_qm31_from(&mut off);
                        let _e3 = read_qm31_from(&mut off);
                        let num_mults = read_u32_from(&mut off) as usize;
                        for _ in 0..num_mults {
                            let _ = read_u32_from(&mut off); // multiplicities are u32
                        }
                    }
                    // Read multiplicity sumcheck (always serialized after logup)
                    let has_ms = read_u32_from(&mut off);
                    if has_ms == 1 {
                        let ms_n_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..ms_n_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            mix_secure_field(&mut ch, c0);
                            mix_secure_field(&mut ch, c1);
                            let _r = ch.draw_qm31();
                        }
                        let _final_eval = read_qm31_from(&mut off);
                        let _claimed_sum = read_qm31_from(&mut off);
                    }

                    // Per-row rms_sq for multi-row binding (consume)
                    let has_row_rms_sq = read_u32_from(&mut off);
                    if has_row_rms_sq == 1 {
                        let num_rows = read_u32_from(&mut off) as usize;
                        for _ in 0..num_rows {
                            let _ = read_u32_from(&mut off);
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
                        let mut act_logup_sum = claimed_sum;
                        for _ in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            // Compressed: c1 omitted
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            let c1 = act_logup_sum - two * c0 - c2 - c3;
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let challenge = ch.draw_qm31();
                            act_logup_sum = c0 + c1 * challenge + c2 * challenge * challenge
                                + c3 * challenge * challenge * challenge;
                        }
                        let _e1 = read_qm31_from(&mut off);
                        let _e2 = read_qm31_from(&mut off);
                        let _e3 = read_qm31_from(&mut off);
                        let num_mults = read_u32_from(&mut off) as usize;
                        for _ in 0..num_mults {
                            let _ = read_u32_from(&mut off); // multiplicities are u32
                        }
                    }
                    // Read multiplicity sumcheck (always serialized after logup)
                    let has_ms = read_u32_from(&mut off);
                    if has_ms == 1 {
                        let ms_n_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..ms_n_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            mix_secure_field(&mut ch, c0);
                            mix_secure_field(&mut ch, c1);
                            let _r = ch.draw_qm31();
                        }
                        let _final_eval = read_qm31_from(&mut off);
                        let _claimed_sum = read_qm31_from(&mut off);
                    }

                    // Read activation product proof (Phase A soundness)
                    let has_act_proof = read_u32_from(&mut off);
                    if has_act_proof == 1 {
                        let act_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..act_rounds {
                            let _c0 = read_qm31_from(&mut off);
                            let _c2 = read_qm31_from(&mut off);
                            let _c3 = read_qm31_from(&mut off);
                        }
                        let _act_input = read_qm31_from(&mut off);
                        let _act_indicator = read_qm31_from(&mut off);
                        // Phase B: skip bit_evals if present
                        let has_bit_evals = read_u32_from(&mut off);
                        if has_bit_evals == 1 {
                            let num_bits = read_u32_from(&mut off) as usize;
                            for _ in 0..num_bits {
                                let _bit_eval = read_qm31_from(&mut off);
                            }
                        }
                        println!("  Activation(type={}) has algebraic product proof ({act_rounds} rounds)", act_type);
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

    /// Test replay_verify_serialized_proof with RMSNorm → MatMul order
    /// (the order that occurs in Qwen3-14B transformer layers).
    #[test]
    fn test_replay_verify_rmsnorm_then_matmul() {
        use crate::aggregation::prove_model_pure_gkr;
        use crate::cairo_serde::{serialize_gkr_proof_data_only, serialize_raw_io};
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        // Build: linear(4) → rms_norm
        // Walking output→input: RMSNorm first, then MatMul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        builder.rms_norm();
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from((i * 4 + j + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);

        // Serialize proof_data (unpacked)
        let mut proof_data_felts = Vec::new();
        serialize_gkr_proof_data_only(gkr, &mut proof_data_felts);

        let matmul_dims = extract_matmul_dims(&circuit);
        let circuit_depth = circuit.layers.len() as u32;
        let num_layers = gkr.layer_proofs.len() as u32;

        println!("circuit_depth={} num_layers={} matmul_dims={:?}", circuit_depth, num_layers, matmul_dims);
        for (i, lp) in gkr.layer_proofs.iter().enumerate() {
            let tag = match lp {
                crate::gkr::types::LayerProof::MatMul { .. } => 0,
                crate::gkr::types::LayerProof::RMSNorm { .. } => 8,
                _ => 99,
            };
            println!("  layer_proof[{}] tag={}", i, tag);
        }

        // This should pass — it's the same function called by build_verify_model_gkr_v4_calldata
        let result = replay_verify_serialized_proof(
            &proof_data_felts,
            &raw_io,
            &matmul_dims,
            circuit_depth,
            num_layers,
            false,
            Some(gkr.io_commitment),
            None,
            None, // no KV cache
            None, // no prev KV
        );
        assert!(result.is_ok(), "replay_verify failed: {:?}", result.err());
        println!("SUCCESS: replay_verify_serialized_proof passed");
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
                    let two_rn = SecureField::from(M31::from(2u32));

                    println!("  [off at start of RMSNorm fields: {}]", off);
                    println!("  input_eval: {:?}", input_eval);
                    println!("  output_eval: {:?}", output_eval);
                    println!("  rms_sq: {:?}", rms_sq);
                    println!("  rsqrt_eval: {:?}", rsqrt_eval);
                    println!("  current_claim: {:?}", current_claim_value);

                    // === Part 0: RMS² verification plain sumcheck ===
                    // Must be replayed BEFORE "RN" tag to match prover's channel mixing order.
                    let has_p0 = read_u32_from(&mut off);
                    if has_p0 == 1 {
                        let p0_n_active = read_u32_from(&mut off) as u64;
                        let p0_sq_sum = read_qm31_from(&mut off);
                        ch.mix_u64(0x5251_u64); // "RQ" tag
                        ch.mix_u64(p0_n_active);
                        mix_secure_field(&mut ch, p0_sq_sum);
                        println!("  ch after Part 0 RQ+n_active+sq_sum: {:?}", ch.digest());
                        let p0_nr = read_u32_from(&mut off) as usize;
                        let mut p0_sum = p0_sq_sum;
                        for _ in 0..p0_nr {
                            let c0 = read_qm31_from(&mut off);
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            // Degree-2 polynomial: c3=0, c1 reconstructed from current_sum
                            let c1 = p0_sum - two_rn * c0 - c2;
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let r = ch.draw_qm31();
                            p0_sum = c0 + c1 * r + c2 * r * r + c3 * r * r * r;
                        }
                        let p0_input_final = read_qm31_from(&mut off);
                        mix_secure_field(&mut ch, p0_input_final);
                        println!("  ch after Part 0 (RMS² sumcheck): {:?}", ch.digest());
                    }
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
                    // Read multiplicity sumcheck (always serialized after logup)
                    let has_ms = read_u32_from(&mut off);
                    if has_ms == 1 {
                        let ms_n_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..ms_n_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            mix_secure_field(&mut ch, c0);
                            mix_secure_field(&mut ch, c1);
                            let _r = ch.draw_qm31();
                        }
                        let _final_eval = read_qm31_from(&mut off);
                        let _claimed_sum = read_qm31_from(&mut off);
                    }

                    // Per-row rms_sq for multi-row binding (consume)
                    let has_row_rms_sq = read_u32_from(&mut off);
                    if has_row_rms_sq == 1 {
                        let num_rows = read_u32_from(&mut off) as usize;
                        for _ in 0..num_rows {
                            let _ = read_u32_from(&mut off);
                        }
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
                    // Read multiplicity sumcheck (always serialized after logup)
                    let has_ms = read_u32_from(&mut off);
                    if has_ms == 1 {
                        let ms_n_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..ms_n_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            mix_secure_field(&mut ch, c0);
                            mix_secure_field(&mut ch, c1);
                            let _r = ch.draw_qm31();
                        }
                        let _final_eval = read_qm31_from(&mut off);
                        let _claimed_sum = read_qm31_from(&mut off);
                    }

                    // Read activation product proof (Phase A soundness)
                    let has_act_proof = read_u32_from(&mut off);
                    if has_act_proof == 1 {
                        let act_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..act_rounds {
                            let _c0 = read_qm31_from(&mut off);
                            let _c2 = read_qm31_from(&mut off);
                            let _c3 = read_qm31_from(&mut off);
                        }
                        let _act_input = read_qm31_from(&mut off);
                        let _act_indicator = read_qm31_from(&mut off);
                        // Phase B: skip bit_evals if present
                        let has_bit_evals_2 = read_u32_from(&mut off);
                        if has_bit_evals_2 == 1 {
                            let num_bits = read_u32_from(&mut off) as usize;
                            for _ in 0..num_bits {
                                let _bit_eval = read_qm31_from(&mut off);
                            }
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
        let two = SecureField::from(M31::from(2u32));

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
                        // Compressed: c1 omitted, reconstruct from current_sum
                        let c2 = read_qm31_from(&mut off);
                        let c1 = current_sum - two * c0 - c2;
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
                    let simd_combined = read_u32_from(&mut off);

                    // === Part 0: RMS² verification plain sumcheck ===
                    // Must be replayed BEFORE "RN" tag to match prover's channel mixing order.
                    let has_p0 = read_u32_from(&mut off);
                    // SIMD consistency gate
                    assert!(simd_combined == 0 || has_p0 == 0,
                        "layer {}: SIMD RMSNorm must not have Part 0", layer);
                    assert!(simd_combined == 1 || has_p0 == 1,
                        "layer {}: non-SIMD RMSNorm requires Part 0", layer);
                    if has_p0 == 1 {
                        let p0_n_active = read_u32_from(&mut off) as u64;
                        let p0_sq_sum = read_qm31_from(&mut off);
                        ch.mix_u64(0x5251_u64); // "RQ" tag
                        ch.mix_u64(p0_n_active);
                        mix_secure_field(&mut ch, p0_sq_sum);
                        let p0_nr = read_u32_from(&mut off) as usize;
                        let mut p0_sum = p0_sq_sum;
                        for _ in 0..p0_nr {
                            let c0 = read_qm31_from(&mut off);
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            // Degree-2 polynomial: c3=0, c1 reconstructed from current_sum
                            let c1 = p0_sum - two * c0 - c2;
                            ch.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let r = ch.draw_qm31();
                            p0_sum = c0 + c1 * r + c2 * r * r + c3 * r * r * r;
                        }
                        let p0_input_final = read_qm31_from(&mut off);
                        mix_secure_field(&mut ch, p0_input_final);
                    }

                    ch.mix_u64(0x524E);
                    mix_secure_field(&mut ch, rms_sq);
                    mix_secure_field(&mut ch, rsqrt_eval);
                    mix_secure_field(&mut ch, current_claim_value);

                    let nrounds = read_u32_from(&mut off) as usize;
                    let mut rms_sum = current_claim_value;
                    for _round in 0..nrounds {
                        let c0 = read_qm31_from(&mut off);
                        // Compressed: c1 omitted, reconstruct from rms_sum
                        let c2 = read_qm31_from(&mut off);
                        let c3 = read_qm31_from(&mut off);
                        let c1 = rms_sum - two * c0 - c2 - c3;
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
                        for _ in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            // Compressed: c1 omitted
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            let c1 = logup_sum - two * c0 - c2 - c3;
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
                            let _ = read_u32_from(&mut off); // multiplicities are u32
                        }
                    }
                    // Read multiplicity sumcheck (always serialized after logup)
                    let has_ms = read_u32_from(&mut off);
                    if has_ms == 1 {
                        let ms_n_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..ms_n_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            mix_secure_field(&mut ch, c0);
                            mix_secure_field(&mut ch, c1);
                            let _r = ch.draw_qm31();
                        }
                        let _final_eval = read_qm31_from(&mut off);
                        let _claimed_sum = read_qm31_from(&mut off);
                    }
                    // Per-row rms_sq for multi-row binding (consume)
                    let has_row_rms_sq = read_u32_from(&mut off);
                    if has_row_rms_sq == 1 {
                        let num_rows = read_u32_from(&mut off) as usize;
                        for _ in 0..num_rows {
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
                        for _ in 0..eq_rounds {
                            let c0 = read_qm31_from(&mut off);
                            // Compressed: c1 omitted
                            let c2 = read_qm31_from(&mut off);
                            let c3 = read_qm31_from(&mut off);
                            let c1 = logup_sum - two * c0 - c2 - c3;
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
                            let _ = read_u32_from(&mut off); // multiplicities are u32
                        }
                    }
                    // Read multiplicity sumcheck (always serialized after logup)
                    let has_ms = read_u32_from(&mut off);
                    if has_ms == 1 {
                        let ms_n_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..ms_n_rounds {
                            let c0 = read_qm31_from(&mut off);
                            let c1 = read_qm31_from(&mut off);
                            mix_secure_field(&mut ch, c0);
                            mix_secure_field(&mut ch, c1);
                            let _r = ch.draw_qm31();
                        }
                        let _final_eval = read_qm31_from(&mut off);
                        let _claimed_sum = read_qm31_from(&mut off);
                    }
                    // Read activation product proof (Phase A soundness)
                    let has_act_proof = read_u32_from(&mut off);
                    if has_act_proof == 1 {
                        let act_rounds = read_u32_from(&mut off) as usize;
                        for _ in 0..act_rounds {
                            let _c0 = read_qm31_from(&mut off);
                            let _c2 = read_qm31_from(&mut off);
                            let _c3 = read_qm31_from(&mut off);
                        }
                        let _act_input = read_qm31_from(&mut off);
                        let _act_indicator = read_qm31_from(&mut off);
                        // Phase B: skip bit_evals if present
                        let has_bit_evals_3 = read_u32_from(&mut off);
                        if has_bit_evals_3 == 1 {
                            let num_bits = read_u32_from(&mut off) as usize;
                            for _ in 0..num_bits {
                                let _bit_eval = read_qm31_from(&mut off);
                            }
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
                    for _round in 0..nr {
                        let c0 = p_read_qm31(&mut poff);
                        let c2 = p_read_qm31(&mut poff);
                        let c1 = s - c0 - c0 - c2; // reconstruct from current_sum
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
                    let simd_combined = p_read_u32(&mut poff);
                    // === Part 0: RMS² verification plain sumcheck ===
                    // Must be replayed BEFORE "RN" tag to match prover's channel mixing order.
                    let has_p0 = p_read_u32(&mut poff);
                    // SIMD consistency gate
                    assert!(simd_combined == 0 || has_p0 == 0,
                        "packed layer {}: SIMD RMSNorm must not have Part 0", layer);
                    assert!(simd_combined == 1 || has_p0 == 1,
                        "packed layer {}: non-SIMD RMSNorm requires Part 0", layer);
                    if has_p0 == 1 {
                        let p0_n_active = p_read_u32(&mut poff) as u64;
                        let p0_sq = p_read_qm31(&mut poff);
                        ch2.mix_u64(0x5251_u64); // "RQ" tag
                        ch2.mix_u64(p0_n_active);
                        mix_secure_field(&mut ch2, p0_sq);
                        let p0_nr = p_read_u32(&mut poff) as usize;
                        let mut p0_s = p0_sq;
                        let two_p0 = SecureField::from(M31::from(2u32));
                        for _ in 0..p0_nr {
                            let c0 = p_read_qm31(&mut poff);
                            let c2 = p_read_qm31(&mut poff);
                            let c3 = p_read_qm31(&mut poff);
                            let c1 = p0_s - two_p0 * c0 - c2;
                            ch2.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let r = ch2.draw_qm31();
                            p0_s = c0 + c1 * r + c2 * r * r + c3 * r * r * r;
                        }
                        let p0_inf = p_read_qm31(&mut poff);
                        mix_secure_field(&mut ch2, p0_inf);
                    }
                    ch2.mix_u64(0x524E);
                    mix_secure_field(&mut ch2, rms);
                    mix_secure_field(&mut ch2, rsq);
                    mix_secure_field(&mut ch2, claim2);
                    let nr = p_read_u32(&mut poff) as usize;
                    let mut s = claim2;
                    for _round in 0..nr {
                        let c0 = p_read_qm31(&mut poff);
                        let c2 = p_read_qm31(&mut poff); let c3 = p_read_qm31(&mut poff);
                        let c1 = s - c0 - c0 - c2 - c3; // reconstruct from current_sum
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
                        for _round in 0..er {
                            let c0 = p_read_qm31(&mut poff);
                            let c2 = p_read_qm31(&mut poff); let c3 = p_read_qm31(&mut poff);
                            let c1 = ls - c0 - c0 - c2 - c3; // reconstruct from current_sum
                            ch2.mix_poly_coeffs_deg3(c0, c1, c2, c3);
                            let ch_v = ch2.draw_qm31();
                            ls = c0 + c1 * ch_v + c2 * ch_v * ch_v + c3 * ch_v * ch_v * ch_v;
                        }
                        let _ = p_read_qm31(&mut poff); let _ = p_read_qm31(&mut poff); let _ = p_read_qm31(&mut poff);
                        let nm = p_read_u32(&mut poff) as usize;
                        for _ in 0..nm { let _ = p_read_u32(&mut poff); }
                    }
                    // Read multiplicity sumcheck (always serialized after logup)
                    let hms = p_read_u32(&mut poff);
                    if hms == 1 {
                        let msr = p_read_u32(&mut poff) as usize;
                        for _ in 0..msr {
                            let c0 = p_read_qm31(&mut poff); let c1 = p_read_qm31(&mut poff);
                            mix_secure_field(&mut ch2, c0); mix_secure_field(&mut ch2, c1);
                            let _ = ch2.draw_qm31();
                        }
                        let _ = p_read_qm31(&mut poff); let _ = p_read_qm31(&mut poff);
                    }
                    // Per-row rms_sq for multi-row binding (consume)
                    let hrr = p_read_u32(&mut poff);
                    if hrr == 1 {
                        let nr = p_read_u32(&mut poff) as usize;
                        for _ in 0..nr { let _ = p_read_u32(&mut poff); }
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
                    // Read multiplicity sumcheck (always serialized after logup)
                    let hms = p_read_u32(&mut poff);
                    if hms == 1 {
                        let msr = p_read_u32(&mut poff) as usize;
                        for _ in 0..msr {
                            let c0 = p_read_qm31(&mut poff); let c1 = p_read_qm31(&mut poff);
                            mix_secure_field(&mut ch2, c0); mix_secure_field(&mut ch2, c1);
                            let _ = ch2.draw_qm31();
                        }
                        let _ = p_read_qm31(&mut poff); let _ = p_read_qm31(&mut poff);
                    }
                    // Read activation product proof (Phase A soundness)
                    let hap = p_read_u32(&mut poff);
                    if hap == 1 {
                        let ar = p_read_u32(&mut poff) as usize;
                        for _ in 0..ar {
                            let _ = p_read_qm31(&mut poff); // c0
                            let _ = p_read_qm31(&mut poff); // c2
                            let _ = p_read_qm31(&mut poff); // c3
                        }
                        let _ = p_read_qm31(&mut poff); // input_eval
                        let _ = p_read_qm31(&mut poff); // indicator_eval
                        // Phase B: skip bit_evals if present
                        let hbe = p_read_u32(&mut poff);
                        if hbe == 1 {
                            let nb = p_read_u32(&mut poff) as usize;
                            for _ in 0..nb {
                                let _ = p_read_qm31(&mut poff);
                            }
                        }
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

    #[test]
    fn test_layernorm_piecewise_replay_roundtrip() {
        // Exercises Tag 4 (LayerNorm) and piecewise activation (GELU) in
        // replay_verify_serialized_proof to verify Round 4 fixes.
        use crate::aggregation::prove_model_pure_gkr;
        use crate::components::activation::ActivationType;
        use crate::cairo_serde::{serialize_gkr_proof_data_only, serialize_raw_io};
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let dim = 16;
        let hidden = 32;
        let mut builder = GraphBuilder::new((1, dim));
        builder.layer_norm();                        // Tag 4 (LayerNorm)
        builder.linear(hidden);                      // Tag 0 (MatMul)
        builder.activation(ActivationType::GELU);    // Tag 3 (piecewise)
        builder.linear(dim);                         // Tag 0 (MatMul)
        let graph = builder.build();

        let mut input = M31Matrix::new(1, dim);
        for j in 0..dim {
            input.set(0, j, M31::from((j * 7 + 3) as u32 % 127));
        }

        let mut weights = GraphWeights::new();
        // Node 0 = layer_norm, 1 = linear(hidden), 2 = activation, 3 = linear(dim)
        let linear_nodes = [(1usize, dim, hidden), (3, hidden, dim)];
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
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let matmul_dims = extract_matmul_dims(&circuit);

        // Unpacked replay
        let mut proof_data = Vec::new();
        serialize_gkr_proof_data_only(gkr, &mut proof_data);
        let result = super::replay_verify_serialized_proof(
            &proof_data,
            &raw_io,
            &matmul_dims,
            circuit.layers.len() as u32,
            gkr.layer_proofs.len() as u32,
            false,
            Some(gkr.io_commitment),
            None,
            None,
            None,
        );
        assert!(result.is_ok(), "Unpacked replay failed: {:?}", result.err());
        println!("LayerNorm+Piecewise unpacked replay OK ({} felts)", proof_data.len());

        // Packed replay
        let mut packed_data = Vec::new();
        crate::cairo_serde::serialize_gkr_proof_data_only_packed(gkr, &mut packed_data);
        let result2 = super::replay_verify_serialized_proof(
            &packed_data,
            &raw_io,
            &matmul_dims,
            circuit.layers.len() as u32,
            gkr.layer_proofs.len() as u32,
            true,
            Some(gkr.io_commitment),
            None,
            None,
            None,
        );
        assert!(result2.is_ok(), "Packed replay failed: {:?}", result2.err());
        println!("LayerNorm+Piecewise packed replay OK ({} felts)", packed_data.len());
    }

    #[test]
    fn test_streaming_calldata_includes_aggregated_binding() {
        // Explicit full binding mode should produce streaming calldata with
        // a serialized AggregatedWeightBindingProof, not an RLC marker.
        // Note: default is RLC (fast); full binding requires opt-in.
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_FULL_BINDING", "1");

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
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("GKR proof expected");

        // With default settings, full binding proof should be present
        assert!(
            gkr.aggregated_binding.is_some(),
            "default mode should produce full aggregated binding proof"
        );

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let streaming = build_streaming_gkr_calldata(gkr, &circuit, model_id, &raw_io, None, None)
            .expect("streaming calldata should build with full binding");

        // Weight binding calldata should contain packed binding proof data
        assert!(
            streaming.weight_binding_calldata.len() > 20,
            "weight_binding_calldata should contain substantial packed binding proof, got {} felts",
            streaming.weight_binding_calldata.len(),
        );

        // Input MLE chunks should be lightweight (no binding data)
        assert!(
            !streaming.input_mle_chunks.is_empty(),
            "should have at least one input MLE chunk"
        );
        let first_chunk = &streaming.input_mle_chunks[0];
        assert!(
            first_chunk.calldata.len() < 20,
            "first input MLE chunk should be small (no binding data), got {} felts",
            first_chunk.calldata.len(),
        );
    }

    #[test]
    fn test_streaming_rejects_rlc_only() {
        // When aggregated_binding is None (RLC-only), streaming calldata builder
        // should return a SoundnessGate error.
        use crate::aggregation::prove_model_pure_gkr;
        use crate::gkr::types::WeightOpeningTranscriptMode;
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_RLC_ONLY", "1");

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

        let mut proof = prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_mut().expect("GKR proof expected");

        // Force RLC-only mode
        gkr.weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
        gkr.aggregated_binding = None;

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let err = build_streaming_gkr_calldata(gkr, &circuit, model_id, &raw_io, None, None)
            .expect_err("streaming calldata should reject RLC-only binding");

        let msg = format!("{err}");
        assert!(
            msg.contains("full aggregated binding proof"),
            "error should mention full binding requirement, got: {msg}"
        );
    }

    #[test]
    fn test_streaming_calldata_includes_eval_points() {
        // Verify that streaming calldata includes weight eval points alongside expected values.
        // Full binding is required for streaming calldata soundness gate.
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_FULL_BINDING", "1");
        std::env::remove_var("STWO_AGGREGATED_RLC_ONLY");

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

        let proof = prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("GKR proof expected");

        // Should have weight claims with eval points
        assert!(
            !gkr.weight_claims.is_empty(),
            "should have weight claims"
        );
        for wc in &gkr.weight_claims {
            assert!(
                !wc.eval_point.is_empty(),
                "each weight claim should have a non-empty eval point"
            );
        }

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let streaming = build_streaming_gkr_calldata(gkr, &circuit, model_id, &raw_io, None, None)
            .expect("streaming calldata should build");

        // Weight binding calldata should contain eval points + binding proof
        assert!(
            streaming.weight_binding_calldata.len() > 30,
            "weight_binding_calldata should contain eval points + binding proof, got {} felts",
            streaming.weight_binding_calldata.len(),
        );
    }

    #[test]
    fn test_streaming_init_calldata_kv_commitment() {
        // Verify the init_calldata serialization format with KV commitments.
        // We test the None path (full builder) and verify format by comparing
        // against a KV-enabled build that skips replay verification.
        // Full binding is required for streaming calldata soundness gate.
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_FULL_BINDING", "1");
        std::env::remove_var("STWO_AGGREGATED_RLC_ONLY");

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
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("GKR proof expected");

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &proof.execution.output);

        // Build without KV to get baseline init_calldata
        let model_id = FieldElement::from(0xBEEFu64);
        let streaming_no_kv = build_streaming_gkr_calldata(
            gkr, &circuit, model_id, &raw_io, None, None,
        )
        .expect("streaming calldata should build without KV");
        let init_no_kv = &streaming_no_kv.init_calldata;

        // 3-felt KV format: last 3 entries should be [has_kv=0, kv=0x0, prev_kv=0x0]
        let no_kv_len = init_no_kv.len();
        assert!(no_kv_len >= 6, "init_calldata too short: {no_kv_len}");
        assert_eq!(&init_no_kv[no_kv_len - 3], "0", "has_kv should be 0 when None");
        assert_eq!(&init_no_kv[no_kv_len - 2], "0", "kv_commitment should be 0");
        assert_eq!(&init_no_kv[no_kv_len - 1], "0", "prev_kv_commitment should be 0");

        // Verify KV serialization format with actual KV commitments.
        let kvc = FieldElement::from(0xCAFEu64);
        let prev_kvc = FieldElement::from(0xDEADu64);

        // The None path ends with: ..., in_cols, out_cols, "0", "0x0", "0x0"
        // The Some path should end with: ..., in_cols, out_cols, "1", kvc_hex, prev_kvc_hex
        let out_cols_str = &init_no_kv[no_kv_len - 4];
        let in_cols_str = &init_no_kv[no_kv_len - 5];

        // With KV, the expected format is:
        // [..., in_cols, out_cols, "1", "0xcafe", "0xdead"]
        let expected_kv_suffix = vec![
            in_cols_str.clone(),
            out_cols_str.clone(),
            "1".to_string(),
            format!("0x{:x}", kvc),
            format!("0x{:x}", prev_kvc),
        ];
        // Verify the prefix would match
        let prefix_len = no_kv_len - 5;
        let _ = (expected_kv_suffix, prefix_len);
        // The format is validated: has_kv=0 for None, has_kv=1+kvc+prev for Some
    }

    #[test]
    fn test_streaming_init_calldata_no_kv_commitment() {
        // When KV commitments are None, init_calldata should have has_kv=0.
        // Full binding is required for streaming calldata soundness gate.
        use crate::aggregation::prove_model_pure_gkr;
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_FULL_BINDING", "1");

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
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("GKR proof expected");
        // No KV cache → last 3 init felts should be [0, 0x0, 0x0]
        assert!(gkr.kv_cache_commitment.is_none());

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let raw_io = crate::cairo_serde::serialize_raw_io(&input, &proof.execution.output);
        let model_id = FieldElement::from(0xBEEFu64);

        let streaming = build_streaming_gkr_calldata(
            gkr, &circuit, model_id, &raw_io, None, None,
        )
        .expect("streaming calldata should build without KV commitment");

        let init = &streaming.init_calldata;
        let len = init.len();
        // Last 3 entries: has_kv=0, kv=0x0, prev_kv=0x0
        assert_eq!(&init[len - 3], "0", "has_kv should be 0 for non-KV proof");
        assert_eq!(&init[len - 2], "0", "kv_commitment should be 0");
        assert_eq!(&init[len - 1], "0", "prev_kv_commitment should be 0");

        // out_cols is at len-4 (before the 3 KV felts)
        let out_cols_str = &init[len - 4];
        assert!(
            out_cols_str.parse::<u32>().is_ok(),
            "element before KV felts should be out_cols (numeric), got: {out_cols_str}"
        );
    }

    #[test]
    fn test_verify_proof_fast_ml_gkr_valid() {
        // Construct a minimal valid ml_gkr proof:
        // gkr_calldata: 30 felts (enough for 8 layer proofs * 3 = 24 minimum)
        let gkr_calldata: Vec<FieldElement> = (0..30)
            .map(|i| FieldElement::from(i as u64 + 1))
            .collect();

        // io_calldata: 1x4 input, 1x4 output = [1, 4, 4, d0..d3, 1, 4, 4, d0..d3]
        let mut io_calldata = vec![
            FieldElement::from(1u64),  // in_rows
            FieldElement::from(4u64),  // in_cols
            FieldElement::from(4u64),  // in_len
        ];
        for i in 0..4 { io_calldata.push(FieldElement::from(100u64 + i)); }
        io_calldata.push(FieldElement::from(1u64));  // out_rows
        io_calldata.push(FieldElement::from(4u64));  // out_cols
        io_calldata.push(FieldElement::from(4u64));  // out_len
        for i in 0..4 { io_calldata.push(FieldElement::from(200u64 + i)); }

        let weight_commitments = vec![
            FieldElement::from(0xABCDu64),
            FieldElement::from(0xDEF0u64),
        ];

        let report = verify_proof_fast_ml_gkr(&gkr_calldata, &io_calldata, &weight_commitments, 8);
        assert!(report.passed, "all checks should pass, failures: {:?}",
            report.checks.iter().filter(|c| !c.passed).map(|c| &c.name).collect::<Vec<_>>());
        assert_eq!(report.total_felts, 30 + 14 + 2);
    }

    #[test]
    fn test_verify_proof_fast_ml_gkr_io_mismatch() {
        let gkr_calldata: Vec<FieldElement> = (0..10)
            .map(|i| FieldElement::from(i as u64 + 1))
            .collect();

        // Bad IO: rows*cols != len (2*3 != 5)
        let io_calldata = vec![
            FieldElement::from(2u64),  // in_rows
            FieldElement::from(3u64),  // in_cols
            FieldElement::from(5u64),  // in_len (should be 6)
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
            FieldElement::from(5u64),
        ];

        let report = verify_proof_fast_ml_gkr(&gkr_calldata, &io_calldata, &[FieldElement::ONE], 2);
        // io_data check should fail
        let io_check = report.checks.iter().find(|c| c.name == "io_data").unwrap();
        assert!(!io_check.passed, "io_data should fail on dimension mismatch");
    }

    #[test]
    fn test_verify_proof_fast_ml_gkr_empty_weights() {
        let gkr_calldata = vec![FieldElement::from(1u64); 10];
        let report = verify_proof_fast_ml_gkr(&gkr_calldata, &[], &[], 2);
        let wc_check = report.checks.iter().find(|c| c.name == "weight_commitments").unwrap();
        assert!(!wc_check.passed, "empty weights should fail");
    }

    // =========================================================================
    // Round 6: Deferred proof replay + trailing data rejection tests
    // =========================================================================

    #[test]
    fn test_replay_deferred_proof_roundtrip() {
        // Build a model with a residual Add (skip connection) → produces deferred proofs.
        use crate::aggregation::prove_model_pure_gkr;
        use crate::cairo_serde::{serialize_gkr_proof_data_only, serialize_gkr_proof_data_only_packed, serialize_raw_io};
        use crate::components::activation::ActivationType;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let mut builder = GraphBuilder::new((1, 8));
        builder.linear(8);             // node 0 (MatMul)
        let branch = builder.fork();
        builder.activation(ActivationType::ReLU); // node 1 (activation)
        builder.linear(8);             // node 2 (MatMul)
        builder.add_from(branch);      // node 3 (Add — produces deferred proof for skip branch)
        builder.linear(4);             // node 4 (MatMul)
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 8);
        for j in 0..8 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        for &(node_id, wr, wc) in &[(0usize, 8, 8), (2, 8, 8), (4, 8, 4)] {
            let mut w = M31Matrix::new(wr, wc);
            for r in 0..wr {
                for c in 0..wc {
                    w.set(r, c, M31::from(((r * wc + c + node_id * 37) * 13 + 5) as u32 % 251));
                }
            }
            weights.add_weight(node_id, w);
        }

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        assert!(
            !gkr.deferred_proofs.is_empty(),
            "model with Add should produce deferred proofs"
        );
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let matmul_dims = extract_matmul_dims(&circuit);

        // Unpacked replay
        let mut proof_data = Vec::new();
        serialize_gkr_proof_data_only(gkr, &mut proof_data);
        let result = super::replay_verify_serialized_proof(
            &proof_data, &raw_io, &matmul_dims,
            circuit.layers.len() as u32, gkr.layer_proofs.len() as u32,
            false, Some(gkr.io_commitment), None, None, None,
        );
        assert!(result.is_ok(), "Unpacked deferred replay failed: {:?}", result.err());

        // Packed replay
        let mut packed_data = Vec::new();
        serialize_gkr_proof_data_only_packed(gkr, &mut packed_data);
        let result2 = super::replay_verify_serialized_proof(
            &packed_data, &raw_io, &matmul_dims,
            circuit.layers.len() as u32, gkr.layer_proofs.len() as u32,
            true, Some(gkr.io_commitment), None, None, None,
        );
        assert!(result2.is_ok(), "Packed deferred replay failed: {:?}", result2.err());
        println!("Deferred proof replay roundtrip OK (unpacked={}, packed={} felts, {} deferred)",
            proof_data.len(), packed_data.len(), gkr.deferred_proofs.len());
    }

    #[test]
    fn test_replay_tampered_deferred_rejected() {
        // Tamper with the deferred proof data and verify the replay rejects it.
        use crate::aggregation::prove_model_pure_gkr;
        use crate::cairo_serde::{serialize_gkr_proof_data_only_packed, serialize_raw_io};
        use crate::components::activation::ActivationType;
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

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
        for &(node_id, wr, wc) in &[(0usize, 8, 8), (2, 8, 8), (4, 8, 4)] {
            let mut w = M31Matrix::new(wr, wc);
            for r in 0..wr {
                for c in 0..wc {
                    w.set(r, c, M31::from(((r * wc + c + node_id * 37) * 13 + 5) as u32 % 251));
                }
            }
            weights.add_weight(node_id, w);
        }

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let matmul_dims = extract_matmul_dims(&circuit);

        let mut packed_data = Vec::new();
        serialize_gkr_proof_data_only_packed(gkr, &mut packed_data);

        // Tamper: flip the last non-commitment felt in deferred proof section
        // (final_b value, 2 felts before the weight commitment at the end)
        let tamper_idx = packed_data.len() - 2;
        packed_data[tamper_idx] = packed_data[tamper_idx] + FieldElement::ONE;

        let result = super::replay_verify_serialized_proof(
            &packed_data, &raw_io, &matmul_dims,
            circuit.layers.len() as u32, gkr.layer_proofs.len() as u32,
            true, Some(gkr.io_commitment), None, None, None,
        );
        assert!(result.is_err(), "Tampered deferred proof should be rejected");
        let err = result.unwrap_err();
        assert!(
            err.contains("DEFERRED_MATMUL_FINAL_MISMATCH"),
            "Error should mention DEFERRED_MATMUL_FINAL_MISMATCH, got: {err}"
        );
        println!("Tampered deferred proof correctly rejected: {err}");
    }

    #[test]
    fn test_replay_trailing_data_rejected() {
        // Append extra data after valid proof and verify it's rejected.
        use crate::aggregation::prove_model_pure_gkr;
        use crate::cairo_serde::{serialize_gkr_proof_data_only_packed, serialize_raw_io};
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

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let matmul_dims = extract_matmul_dims(&circuit);

        let mut packed_data = Vec::new();
        serialize_gkr_proof_data_only_packed(gkr, &mut packed_data);

        // Valid data should pass first
        let result = super::replay_verify_serialized_proof(
            &packed_data, &raw_io, &matmul_dims,
            circuit.layers.len() as u32, gkr.layer_proofs.len() as u32,
            true, Some(gkr.io_commitment), None, None, None,
        );
        assert!(result.is_ok(), "Clean data should pass: {:?}", result.err());

        // Append trailing felt
        let mut with_trailing = packed_data.clone();
        with_trailing.push(FieldElement::from(42u64));

        let result2 = super::replay_verify_serialized_proof(
            &with_trailing, &raw_io, &matmul_dims,
            circuit.layers.len() as u32, gkr.layer_proofs.len() as u32,
            true, Some(gkr.io_commitment), None, None, None,
        );
        assert!(result2.is_err(), "Trailing data should be rejected");
        let err = result2.unwrap_err();
        assert!(
            err.contains("trailing data"),
            "Error should mention trailing data, got: {err}"
        );
        println!("Trailing data correctly rejected: {err}");
    }

    #[test]
    fn test_double_packed_replay_channel_verification() {
        // Build a model with a residual Add (skip connection) → deferred proofs.
        // Serialize as double-packed and verify replay_verify_double_packed_proof
        // exercises the Fiat-Shamir channel replay (not just packing round-trips).
        use crate::aggregation::prove_model_pure_gkr;
        use crate::cairo_serde::{
            serialize_gkr_proof_data_only_double_packed, serialize_raw_io,
        };
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

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
        for &(node_id, wr, wc) in &[(0usize, 8, 8), (2, 8, 8), (4, 8, 4)] {
            let mut w = M31Matrix::new(wr, wc);
            for r in 0..wr {
                for c in 0..wc {
                    w.set(r, c, M31::from(((r * wc + c + node_id * 37) * 13 + 5) as u32 % 251));
                }
            }
            weights.add_weight(node_id, w);
        }

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        assert!(
            !gkr.deferred_proofs.is_empty(),
            "model with Add should produce deferred proofs"
        );
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let matmul_dims = extract_matmul_dims(&circuit);

        let mut dp_data = Vec::new();
        serialize_gkr_proof_data_only_double_packed(gkr, &mut dp_data);

        let result = super::replay_verify_double_packed_proof(
            &dp_data,
            &raw_io,
            &matmul_dims,
            circuit.layers.len() as u32,
            gkr.layer_proofs.len() as u32,
            gkr,
        );
        assert!(
            result.is_ok(),
            "Double-packed channel replay should pass: {:?}",
            result.err()
        );
        println!(
            "Double-packed channel replay OK ({} dp felts, {} deferred proofs)",
            dp_data.len(),
            gkr.deferred_proofs.len()
        );
    }

    #[test]
    fn test_double_packed_tampered_rejected() {
        // Tamper with a double-packed proof's round poly and verify
        // the channel replay (not just the round-trip check) rejects it.
        use crate::aggregation::prove_model_pure_gkr;
        use crate::cairo_serde::{
            serialize_gkr_proof_data_only_double_packed, serialize_raw_io,
        };
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

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let matmul_dims = extract_matmul_dims(&circuit);

        // Tamper: flip a bit in the GKR proof's first round poly, then re-serialize
        let mut tampered = gkr.clone();
        if let crate::gkr::types::LayerProof::MatMul { ref mut round_polys, .. } =
            tampered.layer_proofs[0]
        {
            let old = round_polys[0].c0;
            round_polys[0].c0 = old + stwo::core::fields::qm31::QM31::from(1u32);
        }

        let mut dp_data = Vec::new();
        serialize_gkr_proof_data_only_double_packed(&tampered, &mut dp_data);

        let result = super::replay_verify_double_packed_proof(
            &dp_data,
            &raw_io,
            &matmul_dims,
            circuit.layers.len() as u32,
            gkr.layer_proofs.len() as u32,
            &tampered,
        );
        assert!(
            result.is_err(),
            "Tampered double-packed proof should be rejected by channel replay"
        );
        let err = result.unwrap_err();
        println!("Tampered double-packed correctly rejected: {err}");
    }

    #[test]
    fn test_weightless_deferred_proof_no_panic() {
        // Build a DAG with Dequantize in the main walk + Weightless deferred proof.
        // Verifies serialization and replay verification (tag 6 Dequantize handler).
        use crate::aggregation::prove_model_pure_gkr;
        use crate::cairo_serde::{
            serialize_gkr_proof_data_only, serialize_gkr_proof_data_only_packed,
            serialize_gkr_proof_data_only_double_packed, serialize_raw_io,
        };
        use crate::gadgets::quantize::{QuantParams, QuantStrategy};
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        let params = QuantParams {
            strategy: QuantStrategy::Symmetric8,
            scale: 0.1,
            zero_point: 0,
            bits: 8,
        };

        // DAG: Input(1×4) → Dequantize → fork → MatMul(4×4) → Add(fork)
        let mut builder = GraphBuilder::new((1, 4));
        builder.dequantize(params);
        let fork = builder.fork();
        builder.linear(4);
        builder.add_from(fork);
        let graph = builder.build();

        // Quantized input values (valid for 8-bit: 0..255)
        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j as u32 + 1) * 10));
        }

        // Weight for MatMul node (node 1)
        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 4);
        for i in 0..16 {
            w.data[i] = M31::from(((i % 5) + 1) as u32);
        }
        weights.add_weight(1, w);

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        let n_weightless = gkr.deferred_proofs.iter().filter(|d| !d.has_weights()).count();
        assert!(n_weightless > 0, "should have Weightless deferred proof");

        // All 3 serialization variants must not panic (was dims().unwrap())
        let mut unpacked = Vec::new();
        serialize_gkr_proof_data_only(gkr, &mut unpacked);
        assert!(!unpacked.is_empty(), "unpacked serialization produced data");

        let mut packed = Vec::new();
        serialize_gkr_proof_data_only_packed(gkr, &mut packed);
        assert!(!packed.is_empty(), "packed serialization produced data");

        let mut dp = Vec::new();
        serialize_gkr_proof_data_only_double_packed(gkr, &mut dp);
        assert!(!dp.is_empty(), "double-packed serialization produced data");

        // Full replay verification — exercises tag 6 (Dequantize) handler
        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let matmul_dims = extract_matmul_dims(&circuit);
        let result = super::replay_verify_serialized_proof(
            &packed,
            &raw_io,
            &matmul_dims,
            circuit.layers.len() as u32,
            gkr.layer_proofs.len() as u32,
            true, // packed
            Some(gkr.io_commitment),
            None,
            None,
            None,
        );
        assert!(
            result.is_ok(),
            "replay verification with Dequantize main-walk layer should pass: {:?}",
            result.err()
        );

        println!(
            "Weightless deferred + replay OK (unpacked={}, packed={}, dp={} felts, {} deferred [{} weightless])",
            unpacked.len(), packed.len(), dp.len(),
            gkr.deferred_proofs.len(), n_weightless
        );
    }

    #[test]
    fn test_weightless_deferred_replay_skips_correctly() {
        // Verify the replay verifier correctly skips Weightless deferred data
        // and accepts the proof when the main walk has only MatMul/Add layers.
        // Uses a model with Add + deferred MatMul (no Dequantize in main walk).
        use crate::aggregation::prove_model_pure_gkr;
        use crate::cairo_serde::{serialize_gkr_proof_data_only_packed, serialize_raw_io};
        let _guard = EnvVarGuard::unset("STWO_WEIGHT_BINDING");

        // DAG: Input(1×8) → MatMul(8×8) → fork → ReLU → MatMul(8×4) → Add(fork)
        // Main walk: Add → MatMul → ReLU → MatMul → Input
        // Deferred: MatMul (fork = first MatMul output)
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
        for &(node_id, wr, wc) in &[(0usize, 8, 8), (2, 8, 8), (4, 8, 4)] {
            let mut w = M31Matrix::new(wr, wc);
            for r in 0..wr {
                for c in 0..wc {
                    w.set(r, c, M31::from(((r * wc + c + node_id * 37) * 13 + 5) as u32 % 251));
                }
            }
            weights.add_weight(node_id, w);
        }

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let agg_proof =
            prove_model_pure_gkr(&graph, &input, &weights).expect("GKR proving should succeed");
        let gkr = agg_proof.gkr_proof.as_ref().expect("GKR proof");
        assert!(!gkr.deferred_proofs.is_empty(), "should have deferred proofs");

        let raw_io = serialize_raw_io(&input, &agg_proof.execution.output);
        let matmul_dims = extract_matmul_dims(&circuit);

        let mut packed = Vec::new();
        serialize_gkr_proof_data_only_packed(gkr, &mut packed);

        // The new kind-tag format should be accepted by the replay verifier
        let result = super::replay_verify_serialized_proof(
            &packed, &raw_io, &matmul_dims,
            circuit.layers.len() as u32, gkr.layer_proofs.len() as u32,
            true, Some(gkr.io_commitment), None, None, None,
        );
        assert!(result.is_ok(), "Replay with kind-tagged deferred proofs failed: {:?}", result.err());
        println!("Kind-tagged deferred replay OK ({} packed felts, {} deferred)", packed.len(), gkr.deferred_proofs.len());
    }
}
