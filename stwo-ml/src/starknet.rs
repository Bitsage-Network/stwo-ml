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
    AggregatedModelProof, AggregatedModelProofOnChain,
    AggregationError, LayerClaim,
    prove_model_aggregated_auto, prove_model_aggregated_onchain_auto,
};
use crate::cairo_serde::{serialize_proof, serialize_matmul_sumcheck_proof};
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
    /// IO commitment: Poseidon(inputs || outputs).
    pub io_commitment: FieldElement,
    /// Combined calldata ready for on-chain submission.
    /// Format: [pcs_config(4), io_commitment(1), unified_proof..., matmul_proofs_count, matmul_proof_0..., ...]
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
}

/// Error type for Starknet proof generation.
#[derive(Debug, thiserror::Error)]
pub enum StarknetModelError {
    #[error("Proving error: {0}")]
    ProvingError(#[from] ModelError),
    #[error("Serialization error: {0}")]
    SerializationError(String),
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
    Ok(build_starknet_proof_onchain(&proof))
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
        combined_calldata: Vec::new(),
        layer_claims: proof.activation_claims.clone(),
        num_matmul_proofs: proof.matmul_proofs.len(),
        num_add_claims: proof.add_claims.len(),
        num_mul_claims: proof.mul_claims.len(),
        num_layernorm_claims: proof.layernorm_claims.len(),
        num_attention_proofs: proof.attention_proofs.len(),
        num_embedding_claims: proof.embedding_claims.len(),
        num_layernorm_mean_var_commitments: proof.layernorm_mean_var_commitments.len(),
        num_proven_layers: num_proven,
        pcs_config,
        estimated_gas,
        calldata_size,
        layer_chain_commitment: proof.layer_chain_commitment,
        tee_attestation_hash: None,
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
/// - IO commitment at index [4] (1 felt) — critical: Cairo verifier reads proof_data[4]
/// - Activation STARK calldata
/// - Matmul proof count (1 felt)
/// - Concatenated matmul calldata
pub fn build_starknet_proof_onchain(proof: &AggregatedModelProofOnChain) -> StarknetModelProof {
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

    // Use the IO commitment from the proof (computed during proving and verified).
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
        // IO commitment at index [4]
        io_commitment,
        // Layer chain commitment at index [5]
        proof.layer_chain_commitment,
    ];

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
    combined.push(FieldElement::from(proof.layernorm_mean_var_commitments.len() as u64));
    for commitment in &proof.layernorm_mean_var_commitments {
        combined.push(*commitment);
    }

    let total_trace_rows: usize = proof.activation_claims.iter().map(|c| c.trace_rows).sum();
    let num_proven = proof.num_proven_layers();
    let estimated_gas = estimate_verification_gas(num_proven, total_trace_rows.max(1));
    let calldata_size = combined.len();

    StarknetModelProof {
        unified_calldata,
        matmul_calldata,
        io_commitment,
        combined_calldata: combined,
        layer_claims: proof.activation_claims.clone(),
        num_matmul_proofs: proof.matmul_proofs.len(),
        num_add_claims: proof.add_claims.len(),
        num_mul_claims: proof.mul_claims.len(),
        num_layernorm_claims: proof.layernorm_claims.len(),
        num_attention_proofs: proof.attention_proofs.len(),
        num_embedding_claims: proof.embedding_claims.len(),
        num_layernorm_mean_var_commitments: proof.layernorm_mean_var_commitments.len(),
        num_proven_layers: num_proven,
        pcs_config,
        estimated_gas,
        calldata_size,
        layer_chain_commitment: proof.layer_chain_commitment,
        tee_attestation_hash: None,
    }
}

/// Convert an on-chain aggregated proof into Starknet calldata with TEE attestation.
pub fn build_starknet_proof_onchain_with_tee(
    proof: &AggregatedModelProofOnChain,
    attestation: Option<&TeeAttestation>,
) -> StarknetModelProof {
    let mut result = build_starknet_proof_onchain(proof);
    result.tee_attestation_hash = attestation
        .filter(|a| a.has_report())
        .map(|a| a.report_hash_felt());
    result
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregation::compute_io_commitment;
    use crate::compiler::graph::GraphBuilder;
    use crate::components::activation::ActivationType;
    use stwo::core::fields::m31::M31;

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
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
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
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w2.set(i, j, M31::from(((i * j) % 5 + 1) as u32)); } }
        weights.add_weight(2, w2);
        let mut w4 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w4.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(4, w4);

        let result = prove_for_starknet(&graph, &input, &weights);
        assert!(result.is_ok());
        let proof = result.unwrap();

        // Has unified STARK calldata
        assert!(!proof.unified_calldata.is_empty(), "should have unified calldata");
        assert_eq!(proof.layer_claims.len(), 2, "2 activation layers");
        assert_eq!(proof.num_matmul_proofs, 3, "3 matmul sumchecks");
        assert_eq!(proof.num_proven_layers, 5, "5 total layers");
        assert!(proof.calldata_size > 0);

        // Gas estimate should include DA cost
        let gas_with_da = estimate_gas_from_proof(&proof);
        assert!(gas_with_da > proof.estimated_gas, "DA cost should increase gas");
    }

    #[test]
    fn test_build_starknet_proof_from_aggregated() {
        use crate::aggregation::prove_model_aggregated;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
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
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }
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
        for j in 0..4 { input1.set(0, j, M31::from((j + 1) as u32)); }
        let mut input2 = M31Matrix::new(1, 4);
        for j in 0..4 { input2.set(0, j, M31::from((j + 10) as u32)); }

        let mut output = M31Matrix::new(1, 2);
        output.set(0, 0, M31::from(10));
        output.set(0, 1, M31::from(20));

        let hash1 = compute_io_commitment(&input1, &output);
        let hash2 = compute_io_commitment(&input2, &output);
        assert_ne!(hash1, hash2, "different inputs should produce different hashes");
    }

    #[test]
    fn test_combined_calldata_io_at_index_4() {
        use crate::aggregation::prove_model_aggregated_onchain;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain aggregated proving should succeed");
        let starknet_proof = build_starknet_proof_onchain(&agg_proof);

        // IO commitment should be at index [4]
        assert!(starknet_proof.combined_calldata.len() > 5, "combined calldata too small");
        assert_eq!(
            starknet_proof.combined_calldata[4],
            starknet_proof.io_commitment,
            "combined_calldata[4] must equal io_commitment"
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
            starknet_proof.combined_calldata[3], FieldElement::ZERO,
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
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain proving should succeed");

        assert!(agg_proof.unified_stark.is_some(), "should have unified STARK");

        let starknet_proof = build_starknet_proof_onchain(&agg_proof);

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
        assert_eq!(starknet_proof.pcs_config.fri_config.n_queries, actual_config.fri_config.n_queries);

        // Security sanity: pow_bits > 0 and n_queries > 0
        assert!(starknet_proof.pcs_config.pow_bits > 0, "pow_bits must be > 0 for soundness");
        assert!(starknet_proof.pcs_config.fri_config.n_queries > 0, "n_queries must be > 0 for soundness");
    }

    #[test]
    fn test_prove_for_starknet_onchain_mlp() {
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

        let result = prove_for_starknet_onchain(&graph, &input, &weights);
        assert!(result.is_ok());
        let proof = result.unwrap();

        // Has unified STARK + matmul on-chain proofs
        assert!(!proof.unified_calldata.is_empty());
        assert_eq!(proof.num_matmul_proofs, 2);
        assert_eq!(proof.layer_claims.len(), 1);
        assert_ne!(proof.io_commitment, FieldElement::ZERO);
        assert!(proof.combined_calldata.len() > 5);
        assert_eq!(proof.combined_calldata[4], proof.io_commitment);
    }

    #[test]
    fn test_prove_for_starknet_onchain_matmul_only() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
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
        for j in 0..8 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { w2.set(i, j, M31::from(((i * j) % 7 + 1) as u32)); } }
        weights.add_weight(2, w2);
        let mut w4 = M31Matrix::new(8, 4);
        for i in 0..8 { for j in 0..4 { w4.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(4, w4);

        let agg_proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain proving with Add should succeed");
        let starknet_proof = build_starknet_proof_onchain(&agg_proof);

        // Add claim should be in the unified STARK
        assert_eq!(starknet_proof.num_add_claims, 1, "should have 1 Add claim");
        // Unified calldata covers activation + add together
        assert!(!starknet_proof.unified_calldata.is_empty(), "unified calldata should be non-empty");

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
        for j in 0..8 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w1 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { w1.set(i, j, M31::from(((i * j) % 7 + 1) as u32)); } }
        weights.add_weight(1, w1);
        let mut w2 = M31Matrix::new(8, 4);
        for i in 0..8 { for j in 0..4 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        // For the add graph, we need weights at different node IDs
        let mut weights_add = GraphWeights::new();
        let mut wa0 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { wa0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights_add.add_weight(0, wa0);
        let mut wa1 = M31Matrix::new(8, 8);
        for i in 0..8 { for j in 0..8 { wa1.set(i, j, M31::from(((i * j) % 7 + 1) as u32)); } }
        weights_add.add_weight(1, wa1);
        let mut wa3 = M31Matrix::new(8, 4);
        for i in 0..8 { for j in 0..4 { wa3.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights_add.add_weight(3, wa3);

        let proof_add = prove_model_aggregated_onchain(&graph_add, &input, &weights_add)
            .expect("proving with Add should succeed");
        let proof_no_add = prove_model_aggregated_onchain(&graph_no_add, &input, &weights)
            .expect("proving without Add should succeed");

        let sn_add = build_starknet_proof_onchain(&proof_add);
        let sn_no_add = build_starknet_proof_onchain(&proof_no_add);

        // Model with Add should have unified STARK (includes Add component)
        assert!(!sn_add.unified_calldata.is_empty(), "Add model should have unified STARK calldata");
        assert_eq!(sn_add.num_add_claims, 1, "should have 1 Add claim");

        // Model without Add should have no unified STARK (matmul only)
        assert!(sn_no_add.unified_calldata.is_empty(), "matmul-only model should have no unified STARK");
        assert_eq!(sn_no_add.num_add_claims, 0);
    }
}
