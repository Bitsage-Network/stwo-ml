//! Proof aggregation for on-chain verification.
//!
//! Composes all non-matmul STARK components (activations, Add, Mul, LayerNorm)
//! into a **single unified STARK proof**. Matmul sumcheck proofs remain separate
//! (different proof system).
//!
//! # Architecture
//!
//! ```text
//! Input → [MatMul₀ (sumcheck)] → [ReLU₀ ─┐
//!       → [MatMul₁ (sumcheck)] → [ReLU₁ ─┤
//!       → [Add (residual)]     ───────────┤──→ Single STARK proof
//!       → [Mul (gating)]       ───────────┤    (all non-matmul components)
//!       → [LayerNorm]          ───────────┘
//! ```
//!
//! The single STARK proof covers all activation/add/mul/layernorm constraints,
//! verified on-chain via a single Cairo verifier call.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::channel::MerkleChannel;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::core::pcs::CommitmentSchemeVerifier;
use stwo::core::air::Component;
use stwo::core::verifier::verify as stwo_verify;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::simd::qm31::PackedSecureField;
use stwo::prover::backend::{Col, Column, BackendForChannel};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::CommitmentSchemeProver;
use stwo::prover::prove;
use stwo::prover::ComponentProver;

use stwo_constraint_framework::{
    FrameworkComponent, TraceLocationAllocator,
    LogupTraceGenerator,
};

use num_traits::One;
#[cfg(feature = "cuda-runtime")]
use num_traits::Zero;
use starknet_ff::FieldElement;
use std::collections::HashMap;

use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use crate::compiler::prove::{
    GraphExecution, ModelError, elementwise_add, elementwise_mul,
    apply_layernorm_detailed,
};
use crate::components::elementwise::{ElementwiseAddEval, ElementwiseMulEval};
use crate::components::layernorm::{LayerNormConfig, LayerNormEval, LayerNormRelation, build_rsqrt_table};
use crate::components::activation::{
    ActivationEval, ActivationRelation,
    compute_multiplicities,
};
use crate::components::matmul::{
    M31Matrix, matmul_m31,
    MatMulSumcheckProof, prove_matmul_sumcheck_auto,
    MatMulSumcheckProofOnChain, prove_matmul_sumcheck_onchain_auto,
    estimate_sumcheck_memory,
};
use crate::components::tiled_matmul::{
    TiledMatMulConfig, prove_tiled_matmul, compose_tiled_proof,
};
use crate::components::attention::{
    AttentionWeights, AttentionProof, AttentionProofOnChain,
    MultiHeadAttentionConfig,
    prove_attention_with, prove_attention_onchain,
    attention_forward,
    pad_to_pow2, split_heads, transpose_m31,
};
use crate::components::embedding::{
    EmbeddingEval, EmbeddingRelation,
    embedding_lookup, build_embedding_table_columns,
};
use crate::components::conv2d::{Im2ColConfig, conv2d_forward};
use crate::components::quantize::{QuantizeEval, QuantizeRelation};
use crate::gadgets::lookup_table::PrecomputedTable;
use crate::backend::convert_evaluations;
use tracing::info;

type Blake2sHash = <Blake2sMerkleChannel as MerkleChannel>::H;

/// Compute the minimum log_size for a data vector: next power of two, at least SIMD width.
fn data_log_size(data_len: usize) -> u32 {
    let min_size = data_len.next_power_of_two().max(1 << LOG_N_LANES);
    min_size.ilog2()
}

/// Compute a layer chain commitment: a running Poseidon hash over ALL intermediate
/// values at each layer boundary.
///
/// This binds layer outputs to layer inputs cryptographically, preventing an attacker
/// from substituting intermediate values between independently-verified layers.
///
/// The chain is: `H₀ = Poseidon(rows, cols, ALL input_data); Hₙ = Poseidon(Hₙ₋₁, node_id, rows, cols, ALL output_data)`
///
/// Every element of every intermediate matrix is hashed — no sampling, no shortcuts.
/// For Qwen3-14B (d_model=5120, ~40 layers), this adds ~0.2s of Poseidon hashing,
/// negligible compared to ~40s proving time.
pub fn compute_layer_chain_commitment(
    input: &M31Matrix,
    intermediates: &[(usize, M31Matrix)],
    final_output: &M31Matrix,
) -> FieldElement {
    /// Convert all M31 elements in a matrix to felt252 for hashing.
    fn matrix_to_felts(m: &M31Matrix) -> Vec<FieldElement> {
        if m.data.is_empty() {
            return vec![FieldElement::ZERO];
        }
        m.data.iter().map(|v| FieldElement::from(v.0 as u64)).collect()
    }

    // H₀ = Poseidon(rows, cols, ALL input elements)
    let mut hash_parts: Vec<FieldElement> = vec![
        FieldElement::from(input.rows as u64),
        FieldElement::from(input.cols as u64),
    ];
    hash_parts.extend(matrix_to_felts(input));
    let mut chain = starknet_crypto::poseidon_hash_many(&hash_parts);

    // Hₙ = Poseidon(Hₙ₋₁, node_id, rows, cols, ALL output elements)
    for (node_id, output) in intermediates {
        let mut parts = vec![
            chain,
            FieldElement::from(*node_id as u64),
            FieldElement::from(output.rows as u64),
            FieldElement::from(output.cols as u64),
        ];
        parts.extend(matrix_to_felts(output));
        chain = starknet_crypto::poseidon_hash_many(&parts);
    }

    // Final: include the final output (ALL elements)
    let mut parts = vec![
        chain,
        FieldElement::from(u64::MAX), // sentinel for final output
        FieldElement::from(final_output.rows as u64),
        FieldElement::from(final_output.cols as u64),
    ];
    parts.extend(matrix_to_felts(final_output));
    starknet_crypto::poseidon_hash_many(&parts)
}

/// Compute IO commitment: Poseidon hash of flattened input and output M31 values.
///
/// This binds the proof to specific computation inputs/outputs, preventing replay
/// of a valid proof with different I/O claims.
pub fn compute_io_commitment(input: &M31Matrix, output: &M31Matrix) -> FieldElement {
    let mut hash_inputs = Vec::new();
    for &v in &input.data {
        hash_inputs.push(FieldElement::from(v.0 as u64));
    }
    for &v in &output.data {
        hash_inputs.push(FieldElement::from(v.0 as u64));
    }
    starknet_crypto::poseidon_hash_many(&hash_inputs)
}

/// Compute a Poseidon commitment over LayerNorm mean and variance values.
///
/// Binds the prover to specific mean/variance choices, preventing a malicious prover
/// from substituting arbitrary values. The verifier independently recomputes mean/variance
/// from the forward pass and checks this commitment.
///
/// Follows the same `poseidon_hash_many` pattern as `compute_layer_chain_commitment`
/// and `compute_io_commitment`.
pub fn compute_layernorm_mean_var_commitment(means: &[M31], variances: &[M31]) -> FieldElement {
    let mut hash_inputs = Vec::with_capacity(means.len() + variances.len());
    for &m in means {
        hash_inputs.push(FieldElement::from(m.0 as u64));
    }
    for &v in variances {
        hash_inputs.push(FieldElement::from(v.0 as u64));
    }
    starknet_crypto::poseidon_hash_many(&hash_inputs)
}

/// A claim about a single layer's computation.
#[derive(Debug, Clone)]
pub struct LayerClaim {
    pub layer_index: usize,
    pub claimed_sum: SecureField,
    pub trace_rows: usize,
}

/// Aggregated model proof: single unified STARK (all non-matmul components) + per-matmul sumchecks.
/// Generic over the Merkle hash type `H`.
pub struct AggregatedModelProofFor<H: MerkleHasherLifted> {
    /// Single STARK proof covering all non-matmul components
    /// (activations, Add, Mul, LayerNorm, Embedding).
    /// `None` if the model has no non-matmul layers.
    pub unified_stark: Option<StarkProof<H>>,
    /// Per-matmul sumcheck proofs, in layer order.
    pub matmul_proofs: Vec<(usize, MatMulSumcheckProof)>,
    /// Per-Add layer claims (verified inside unified STARK).
    pub add_claims: Vec<LayerClaim>,
    /// Per-Mul layer claims (verified inside unified STARK).
    pub mul_claims: Vec<LayerClaim>,
    /// Per-LayerNorm layer claims (verified inside unified STARK).
    pub layernorm_claims: Vec<LayerClaim>,
    /// Forward pass execution trace.
    pub execution: GraphExecution,
    /// Per-activation-layer claims (for verification).
    pub activation_claims: Vec<LayerClaim>,
    /// Per-attention proofs (separate from unified STARK).
    pub attention_proofs: Vec<(usize, AttentionProof<H>)>,
    /// Per-embedding layer claims (verified inside unified STARK).
    pub embedding_claims: Vec<LayerClaim>,
    /// Per-quantize layer claims (verified inside unified STARK via LogUp range-check).
    pub quantize_claims: Vec<LayerClaim>,
    /// Layer chain commitment: running Poseidon hash of intermediate values.
    /// Binds layer outputs to layer inputs, preventing substitution of intermediates.
    pub layer_chain_commitment: FieldElement,
    /// IO commitment: Poseidon(input_data || output_data).
    /// Binds the proof to specific input/output data, preventing replay with different I/O.
    pub io_commitment: FieldElement,
    /// Per-LayerNorm Poseidon commitments of (means, variances).
    /// Binds the prover to specific mean/variance values, preventing substitution.
    pub layernorm_mean_var_commitments: Vec<FieldElement>,
}

/// Aggregated model proof using Blake2s (default).
pub type AggregatedModelProof = AggregatedModelProofFor<Blake2sHash>;

impl<H: MerkleHasherLifted> AggregatedModelProofFor<H> {
    /// Total number of proven layers (matmul + activation + add + mul + layernorm + attention + embedding + quantize).
    pub fn num_proven_layers(&self) -> usize {
        self.matmul_proofs.len() + self.activation_claims.len()
            + self.add_claims.len() + self.mul_claims.len() + self.layernorm_claims.len()
            + self.attention_proofs.len() + self.embedding_claims.len()
            + self.quantize_claims.len()
    }

    /// Estimated calldata size in bytes for on-chain submission.
    pub fn estimated_calldata_bytes(&self) -> usize {
        // Commitments: 3 trees × 32 bytes each
        let commitment_size = 3 * 32;
        // FRI proof: ~1KB per component (rough estimate)
        let num_components = self.activation_claims.len()
            + self.add_claims.len() + self.mul_claims.len() + self.layernorm_claims.len()
            + self.embedding_claims.len();
        let fri_size = 1024 * num_components.max(1);
        // Sumcheck proofs: ~256 bytes each
        let sumcheck_size = self.matmul_proofs.len() * 256;
        // Claims are lightweight (no separate STARK proofs)
        let claim_size = (self.add_claims.len() + self.mul_claims.len()
            + self.layernorm_claims.len() + self.embedding_claims.len()) * 32;
        // Attention proofs: ~2KB each (multiple matmul sumchecks + softmax STARK)
        let attention_size = self.attention_proofs.len() * 2048;
        let layernorm_mv_size = self.layernorm_mean_var_commitments.len() * 32;
        commitment_size + fri_size + sumcheck_size + claim_size + attention_size + layernorm_mv_size
    }
}

/// Error type for aggregation.
#[derive(Debug, thiserror::Error)]
pub enum AggregationError {
    #[error("No components to aggregate")]
    EmptyComponents,
    #[error("Proving error: {0}")]
    ProvingError(String),
    #[error("Model error: {0}")]
    ModelError(#[from] ModelError),
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
}

/// Batched matmul sumcheck proof — multiple matmuls proved in one combined sumcheck.
///
/// Instead of N individual sumcheck proofs (each with its own round polynomials),
/// a batch combines them with random lambda weighting: h(x) = Σ λ^i · f_a_i(x)·f_b_i(x).
/// One set of shared round polynomials + per-matmul final evaluations.
#[derive(Debug, Clone)]
pub struct BatchedMatMulProofOnChain {
    /// Padded k dimension (shared by all entries in this batch).
    pub k: u32,
    /// Number of sumcheck rounds (log2(k)).
    pub num_rounds: u32,
    /// Lambda batching weight drawn from Fiat-Shamir.
    /// The verifier re-derives this from the channel and asserts equality.
    pub lambda: SecureField,
    /// Combined claimed sum: Σ λ^i · claimed_sum_i.
    pub combined_claimed_sum: SecureField,
    /// Shared round polynomials (one per round).
    pub round_polys: Vec<crate::components::matmul::RoundPoly>,
    /// Per-matmul entries with individual evaluations and commitments.
    pub entries: Vec<BatchedMatMulEntryOnChain>,
}

/// Per-matmul entry within a batched proof.
#[derive(Debug, Clone)]
pub struct BatchedMatMulEntryOnChain {
    pub node_id: usize,
    pub m: u32,
    pub n: u32,
    pub claimed_sum: SecureField,
    pub final_a_eval: SecureField,
    pub final_b_eval: SecureField,
    pub a_commitment: starknet_ff::FieldElement,
    pub b_commitment: starknet_ff::FieldElement,
}

/// Deferred matmul data: forward pass computed, proving deferred to Phase 2.
/// Weight matrix (B) is NOT cloned — looked up from GraphWeights during Phase 2
/// to avoid duplicating ~37 GB of weight data in memory.
struct DeferredMatMul {
    node_id: usize,
    a: M31Matrix,
    c: M31Matrix,
    dims: (usize, usize, usize),
}

/// Collected activation layer data for aggregation.
struct ActivationLayerData {
    node_id: usize,
    inputs: Vec<M31>,
    outputs: Vec<M31>,
    table: PrecomputedTable,
    log_size: u32,
}

/// Collected Add layer data for unified STARK aggregation.
struct AddLayerData {
    node_id: usize,
    lhs: Vec<M31>,
    rhs: Vec<M31>,
    output: Vec<M31>,
    log_size: u32,
}

/// Collected Mul layer data for unified STARK aggregation.
struct MulLayerData {
    node_id: usize,
    lhs: Vec<M31>,
    rhs: Vec<M31>,
    output: Vec<M31>,
    log_size: u32,
}

/// Collected LayerNorm layer data for unified STARK aggregation.
struct LayerNormLayerData {
    node_id: usize,
    inputs: Vec<M31>,
    means: Vec<M31>,
    variances: Vec<M31>,
    rsqrt_vals: Vec<M31>,
    outputs: Vec<M31>,
    rsqrt_table: PrecomputedTable,
    log_size: u32,
}

/// Collected Attention layer data for aggregation.
struct AttentionLayerData {
    node_id: usize,
    config: MultiHeadAttentionConfig,
    weights: AttentionWeights,
    input: M31Matrix,
}

/// Collected Embedding layer data for unified STARK aggregation.
struct EmbeddingLayerData {
    node_id: usize,
    token_ids: Vec<M31>,
    col_indices: Vec<M31>,
    values: Vec<M31>,
    multiplicities: Vec<M31>,
    table_tokens: Vec<M31>,
    table_cols: Vec<M31>,
    table_values: Vec<M31>,
    log_size: u32,
}

/// Collected Quantize layer data for unified STARK aggregation.
struct QuantizeLayerData {
    node_id: usize,
    /// The quantized values (trace).
    values: Vec<M31>,
    /// Multiplicity of each range table entry (how many trace values hit it).
    multiplicities: Vec<M31>,
    /// Number of bits in the range table (e.g. 8 for INT8).
    bits: u32,
    log_size: u32,
}

/// Prove an entire computation graph with aggregated STARK proof,
/// generic over backend and Merkle channel.
///
/// All non-matmul layers are combined into a **single unified STARK proof**,
/// while each matmul gets its own sumcheck proof. Trace generation uses
/// `SimdBackend`; commitment and proving use backend `B`, enabling GPU
/// acceleration of Merkle hashing, FRI, and quotient evaluation.
pub fn prove_model_aggregated_with<B, MC>(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofFor<<MC as MerkleChannel>::H>, AggregationError>
where
    B: BackendForChannel<MC> + PolyOps,
    MC: MerkleChannel,
    FrameworkComponent<ActivationEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseAddEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseMulEval>: ComponentProver<B>,
    FrameworkComponent<LayerNormEval>: ComponentProver<B>,
    FrameworkComponent<EmbeddingEval>: ComponentProver<B>,
    FrameworkComponent<QuantizeEval>: ComponentProver<B>,
{
    info!(
        backend = std::any::type_name::<B>(),
        channel = std::any::type_name::<MC>(),
        "Proving unified STARK (off-chain aggregation)"
    );
    let config = PcsConfig::default();
    let mut intermediates: Vec<(usize, M31Matrix)> = Vec::new();
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();
    let mut current = input.clone();

    // Collect matmul proofs and layer data during forward pass
    let mut matmul_proofs: Vec<(usize, MatMulSumcheckProof)> = Vec::new();
    let mut activation_layers: Vec<ActivationLayerData> = Vec::new();
    let mut add_layers: Vec<AddLayerData> = Vec::new();
    let mut mul_layers: Vec<MulLayerData> = Vec::new();
    let mut layernorm_layers: Vec<LayerNormLayerData> = Vec::new();
    let mut attention_layers: Vec<AttentionLayerData> = Vec::new();
    let mut embedding_layers: Vec<EmbeddingLayerData> = Vec::new();
    let mut quantize_layers: Vec<QuantizeLayerData> = Vec::new();

    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];

        // Resolve current input from first dependency
        if let Some(&first_input) = node.inputs.first() {
            if let Some(inp) = node_outputs.get(&first_input) {
                current = inp.clone();
            }
        }

        match &node.op {
            GraphOp::MatMul { .. } => {
                let weight = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;

                // GPU GEMM for all matrix sizes, CPU fallback
                #[cfg(feature = "cuda-runtime")]
                let output = crate::gpu_sumcheck::gpu_matmul_m31_full(&current, weight)
                    .unwrap_or_else(|_| matmul_m31(&current, weight));
                #[cfg(not(feature = "cuda-runtime"))]
                let output = matmul_m31(&current, weight);

                let proof = prove_matmul_sumcheck_auto(&current, weight, &output)
                    .map_err(|e| ModelError::ProvingError {
                        layer: node.id,
                        message: format!("MatMul sumcheck: {e}"),
                    })?;

                intermediates.push((node.id, current.clone()));
                matmul_proofs.push((node.id, proof));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Activation { activation_type, .. } => {
                let f = activation_type.as_fn();
                let output = crate::compiler::prove::apply_activation_pub(&current, &*f);

                let act_log_size = activation_type.recommended_table_log_size();
                let table = PrecomputedTable::build(|x| (*f)(x), act_log_size);
                let flat_inputs = current.data.clone();
                let flat_outputs = output.data.clone();

                activation_layers.push(ActivationLayerData {
                    node_id: node.id,
                    inputs: flat_inputs,
                    outputs: flat_outputs,
                    table,
                    log_size: act_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Add { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());

                #[cfg(feature = "cuda-runtime")]
                let output = {
                    let rows = lhs.rows.max(rhs.rows);
                    let cols = lhs.cols.max(rhs.cols);
                    crate::gpu_sumcheck::gpu_elementwise_add(&lhs.data, &rhs.data)
                        .map(|data| M31Matrix { rows, cols, data })
                        .unwrap_or_else(|_| elementwise_add(&lhs, &rhs))
                };
                #[cfg(not(feature = "cuda-runtime"))]
                let output = elementwise_add(&lhs, &rhs);

                let add_log_size = data_log_size(output.data.len());
                add_layers.push(AddLayerData {
                    node_id: node.id,
                    lhs: lhs.data.clone(),
                    rhs: rhs.data.clone(),
                    output: output.data.clone(),
                    log_size: add_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Mul { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());

                #[cfg(feature = "cuda-runtime")]
                let output = {
                    let rows = lhs.rows.max(rhs.rows);
                    let cols = lhs.cols.max(rhs.cols);
                    crate::gpu_sumcheck::gpu_elementwise_mul(&lhs.data, &rhs.data)
                        .map(|data| M31Matrix { rows, cols, data })
                        .unwrap_or_else(|_| elementwise_mul(&lhs, &rhs))
                };
                #[cfg(not(feature = "cuda-runtime"))]
                let output = elementwise_mul(&lhs, &rhs);

                let mul_log_size = data_log_size(output.data.len());
                mul_layers.push(MulLayerData {
                    node_id: node.id,
                    lhs: lhs.data.clone(),
                    rhs: rhs.data.clone(),
                    output: output.data.clone(),
                    log_size: mul_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::LayerNorm { dim } => {
                let ln_log_size = LayerNormConfig::new(*dim).rsqrt_table_log_size;
                let ln = apply_layernorm_detailed(&current, *dim);
                let rsqrt_table = build_rsqrt_table(ln_log_size);

                layernorm_layers.push(LayerNormLayerData {
                    node_id: node.id,
                    inputs: ln.inputs.clone(),
                    means: ln.means.clone(),
                    variances: ln.variances.clone(),
                    rsqrt_vals: ln.rsqrt_vals.clone(),
                    outputs: ln.outputs.clone(),
                    rsqrt_table,
                    log_size: ln_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, ln.output_matrix.clone());
                current = ln.output_matrix;
            }

            GraphOp::Attention { config: attn_config } => {
                let w_q = weights.get_named_weight(node.id, "w_q");
                let w_k = weights.get_named_weight(node.id, "w_k");
                let w_v = weights.get_named_weight(node.id, "w_v");
                let w_o = weights.get_named_weight(node.id, "w_o");

                if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (w_q, w_k, w_v, w_o) {
                    let attn_weights = AttentionWeights {
                        w_q: wq.clone(), w_k: wk.clone(),
                        w_v: wv.clone(), w_o: wo.clone(),
                    };
                    // Run forward pass to get correct output for downstream nodes
                    let inter = attention_forward(&current, &attn_weights, attn_config, false);
                    attention_layers.push(AttentionLayerData {
                        node_id: node.id,
                        config: *attn_config,
                        weights: attn_weights,
                        input: current.clone(),
                    });
                    intermediates.push((node.id, current.clone()));
                    node_outputs.insert(node.id, inter.final_output.clone());
                    current = inter.final_output;
                } else {
                    // No weights found — passthrough (log warning)
                    info!(node_id = node.id, "Attention node missing weights, passthrough");
                    intermediates.push((node.id, current.clone()));
                    node_outputs.insert(node.id, current.clone());
                }
            }

            GraphOp::Embedding { vocab_size: _, embed_dim: _ } => {
                let embed_table = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let token_u32s: Vec<u32> = current.data.iter().map(|m| m.0).collect();
                let (output, token_ids, col_indices, values, multiplicities) =
                    embedding_lookup(&token_u32s, embed_table);
                let (table_tokens, table_cols, table_values) = build_embedding_table_columns(embed_table);
                let log_size = data_log_size(values.len());
                embedding_layers.push(EmbeddingLayerData {
                    node_id: node.id,
                    token_ids,
                    col_indices,
                    values,
                    multiplicities,
                    table_tokens,
                    table_cols,
                    table_values,
                    log_size,
                });
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Conv2D { in_channels, out_channels, kernel_size, stride, padding } => {
                let kernel = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let im2col_config = Im2ColConfig {
                    in_channels: *in_channels,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    input_h: current.rows,
                    input_w: current.cols / in_channels,
                };
                let (_im2col_mat, _kernel_mat, output) =
                    conv2d_forward(&current.data, &kernel.data, &im2col_config, *out_channels);
                // Prove the im2col×kernel matmul
                let proof = prove_matmul_sumcheck_auto(&_im2col_mat, &_kernel_mat, &output)
                    .map_err(|e| ModelError::ProvingError {
                        layer: node.id,
                        message: format!("Conv2D matmul: {e}"),
                    })?;
                intermediates.push((node.id, current.clone()));
                matmul_proofs.push((node.id, proof));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Quantize { params, size: _ } => {
                // Apply quantization forward pass
                let quantized: Vec<M31> = current.data.iter()
                    .map(|v| {
                        let val = v.0;
                        let max_val = (1u32 << params.bits) - 1;
                        M31::from(val.min(max_val))
                    })
                    .collect();

                // Build range table [0, 1, ..., 2^bits - 1] and compute multiplicities
                let table_size = 1u32 << params.bits;
                let mut multiplicities = vec![M31::from(0); table_size as usize];
                for v in &quantized {
                    let idx = v.0 as usize;
                    if idx < multiplicities.len() {
                        multiplicities[idx] = M31::from(multiplicities[idx].0 + 1);
                    }
                }

                let log_size = data_log_size(quantized.len().max(table_size as usize));
                quantize_layers.push(QuantizeLayerData {
                    node_id: node.id,
                    values: quantized.clone(),
                    multiplicities,
                    bits: params.bits,
                    log_size,
                });

                let output = M31Matrix {
                    rows: current.rows,
                    cols: current.cols,
                    data: quantized,
                };
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            _ => {
                // Identity, etc. — passthrough
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, current.clone());
            }
        }
    }

    // Prove attention layers (separate from unified STARK)
    let mut attention_proofs = Vec::new();
    for layer in &attention_layers {
        let proof = prove_attention_with::<B, MC>(
            &layer.input, &layer.weights, &layer.config, false,
        ).map_err(|e| AggregationError::ProvingError(
            format!("Attention node {}: {e}", layer.node_id),
        ))?;
        attention_proofs.push((layer.node_id, proof));
    }

    // Compute layer chain commitment before consuming `intermediates`.
    let layer_chain_commitment = compute_layer_chain_commitment(
        input, &intermediates, &current,
    );
    // Compute IO commitment binding this proof to specific input/output.
    let io_commitment = compute_io_commitment(input, &current);

    // Compute per-LayerNorm mean/variance commitments (defense-in-depth).
    let layernorm_mean_var_commitments: Vec<FieldElement> = layernorm_layers.iter()
        .map(|layer| compute_layernorm_mean_var_commitment(&layer.means, &layer.variances))
        .collect();

    let execution = GraphExecution {
        intermediates,
        output: current,
    };

    // Check if there are any non-matmul components to aggregate
    let has_components = !activation_layers.is_empty()
        || !add_layers.is_empty() || !mul_layers.is_empty() || !layernorm_layers.is_empty()
        || !embedding_layers.is_empty() || !quantize_layers.is_empty();

    if !has_components {
        return Ok(AggregatedModelProofFor {
            unified_stark: None,
            matmul_proofs,
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            execution,
            activation_claims: Vec::new(),
            attention_proofs,
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            layer_chain_commitment,
            io_commitment,
            layernorm_mean_var_commitments: Vec::new(),
        });
    }

    // === Build unified STARK for all non-matmul components ===
    // Per-component log_sizes: each component uses its own size derived from
    // its table or data length. The max_log_size drives twiddle precomputation.
    let all_log_sizes: Vec<u32> = activation_layers.iter().map(|l| l.log_size)
        .chain(add_layers.iter().map(|l| l.log_size))
        .chain(mul_layers.iter().map(|l| l.log_size))
        .chain(layernorm_layers.iter().map(|l| l.log_size))
        .chain(embedding_layers.iter().map(|l| l.log_size))
        .chain(quantize_layers.iter().map(|l| l.log_size))
        .collect();
    let max_log_size = *all_log_sizes.iter().max().unwrap();

    let max_degree_bound = max_log_size + 1;
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(max_degree_bound + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::new(config, &twiddles);

    let has_logup = !activation_layers.is_empty() || !layernorm_layers.is_empty()
        || !embedding_layers.is_empty() || !quantize_layers.is_empty();

    // Tree 0: Preprocessed columns (always committed, may be empty)
    // - Activation tables: 2 cols per layer (table_input, table_output)
    // - LayerNorm rsqrt tables: 2 cols per layer (table_var, table_rsqrt)
    // - Embedding tables: 3 cols per layer (table_token, table_col, table_value)
    // - Quantize: 1 col per layer (range table [0..2^bits))
    // - Add/Mul: 0 preprocessed cols
    {
        let mut tree_builder = commitment_scheme.tree_builder();
        for layer in &activation_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let (table_input_col, table_output_col) = build_table_columns(&layer.table, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, table_input_col),
                CircleEvaluation::new(layer_domain, table_output_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        for layer in &layernorm_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let (table_var_col, table_rsqrt_col) = build_table_columns(&layer.rsqrt_table, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, table_var_col),
                CircleEvaluation::new(layer_domain, table_rsqrt_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        for layer in &embedding_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let simd_evals = build_embedding_preprocessed_columns(
                &layer.table_tokens, &layer.table_cols, &layer.table_values, layer_size, layer_domain,
            );
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        for layer in &quantize_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let range_table_col = build_quantize_range_table(layer.bits, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, range_table_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        tree_builder.commit(channel);
    }

    // Tree 1: Execution traces — concatenated in order:
    // 1. Activation traces: 3 cols per layer (trace_in, trace_out, multiplicity)
    // 2. Add traces: 3 cols per layer (lhs, rhs, output)
    // 3. Mul traces: 3 cols per layer (lhs, rhs, output)
    // 4. LayerNorm traces: 6 cols per layer (input, mean, var, rsqrt, output, multiplicity)
    // 5. Embedding traces: 4 cols per layer (token_id, col_idx, value, multiplicity)
    let mut tree_builder = commitment_scheme.tree_builder();
    let mut activation_mults: Vec<Vec<M31>> = Vec::new();
    for layer in &activation_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let pad_input = layer.table.inputs[0];
        let pad_output = layer.table.outputs[0];
        let padding_count = layer_size.saturating_sub(layer.inputs.len());

        let mut mults = compute_multiplicities(&layer.inputs, &layer.table);
        if padding_count > 0 {
            mults[0] += M31::from(padding_count as u32);
        }

        let (trace_in, trace_out, mult_col) = build_trace_columns(
            &layer.inputs, &layer.outputs, &mults,
            pad_input, pad_output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, trace_in),
            CircleEvaluation::new(layer_domain, trace_out),
            CircleEvaluation::new(layer_domain, mult_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        activation_mults.push(mults);
    }
    for layer in &add_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let (lhs_col, rhs_col, out_col) = build_elementwise_trace_columns(
            &layer.lhs, &layer.rhs, &layer.output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, lhs_col),
            CircleEvaluation::new(layer_domain, rhs_col),
            CircleEvaluation::new(layer_domain, out_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    for layer in &mul_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let (lhs_col, rhs_col, out_col) = build_elementwise_trace_columns(
            &layer.lhs, &layer.rhs, &layer.output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, lhs_col),
            CircleEvaluation::new(layer_domain, rhs_col),
            CircleEvaluation::new(layer_domain, out_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    let mut layernorm_mults: Vec<Vec<M31>> = Vec::new();
    for layer in &layernorm_layers {
        let layer_size = 1usize << layer.log_size;
        let mults = compute_multiplicities(&layer.variances, &layer.rsqrt_table);
        let cols = build_layernorm_trace_columns(
            &layer.inputs, &layer.means, &layer.variances,
            &layer.rsqrt_vals, &layer.outputs, &mults,
            &layer.rsqrt_table, layer_size,
        );
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(cols));
        layernorm_mults.push(mults);
    }
    for layer in &embedding_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let simd_evals = build_embedding_trace_columns(
            &layer.token_ids, &layer.col_indices, &layer.values, &layer.multiplicities,
            layer_size, layer_domain,
        );
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    for layer in &quantize_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let simd_evals = build_quantize_trace_columns(
            &layer.values, &layer.multiplicities, layer.bits, layer_size, layer_domain,
        );
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    tree_builder.commit(channel);

    // Draw relation elements and build Tree 2 — only if LogUp components exist
    let mut activation_lookup: Option<ActivationRelation> = None;
    let mut layernorm_lookup: Option<LayerNormRelation> = None;
    let mut embedding_lookup_rel: Option<EmbeddingRelation> = None;
    let mut quantize_lookup: Option<QuantizeRelation> = None;
    let mut activation_claimed_sums: Vec<SecureField> = Vec::new();
    let mut layernorm_claimed_sums: Vec<SecureField> = Vec::new();
    let mut embedding_claimed_sums: Vec<SecureField> = Vec::new();
    let mut quantize_claimed_sums: Vec<SecureField> = Vec::new();

    if has_logup {
        // Draw relation elements — activation first, then layernorm, then embedding, then quantize
        if !activation_layers.is_empty() {
            activation_lookup = Some(ActivationRelation::draw(channel));
        }
        if !layernorm_layers.is_empty() {
            layernorm_lookup = Some(LayerNormRelation::draw(channel));
        }
        if !embedding_layers.is_empty() {
            embedding_lookup_rel = Some(EmbeddingRelation::draw(channel));
        }
        if !quantize_layers.is_empty() {
            quantize_lookup = Some(QuantizeRelation::draw(channel));
        }

        // Tree 2: Interaction traces (LogUp) — for activation, layernorm, embedding, and quantize
        // Add/Mul are pure AIR (no interaction columns)
        let mut tree_builder = commitment_scheme.tree_builder();

        if let Some(ref lookup) = activation_lookup {
        for (idx, layer) in activation_layers.iter().enumerate() {
            let layer_size = 1usize << layer.log_size;
            let layer_vec_size = layer_size >> LOG_N_LANES;
            let pad_input = layer.table.inputs[0];
            let pad_output = layer.table.outputs[0];

            let (table_in_col, table_out_col) = build_table_columns(&layer.table, layer_size);
            let (trace_in_col, trace_out_col, _) = build_trace_columns(
                &layer.inputs, &layer.outputs, &activation_mults[idx],
                pad_input, pad_output, layer_size,
            );

            let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
            let mut col_gen = logup_gen.new_col();

            for vec_row in 0..layer_vec_size {
                let q_table: PackedSecureField = lookup.lookup_elements().combine(
                    &[table_in_col.data[vec_row], table_out_col.data[vec_row]],
                );
                let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                    &[trace_in_col.data[vec_row], trace_out_col.data[vec_row]],
                );

                let mult_packed = pack_multiplicities(&activation_mults[idx], vec_row);
                let numerator = q_table - mult_packed * q_trace;
                let denominator = q_table * q_trace;

                col_gen.write_frac(vec_row, numerator, denominator);
            }
            col_gen.finalize_col();

            let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
            activation_claimed_sums.push(claimed_sum);
        }
    }

        if let Some(ref lookup) = layernorm_lookup {
            for (idx, layer) in layernorm_layers.iter().enumerate() {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;
                let (table_var_col, table_rsqrt_col) = build_table_columns(&layer.rsqrt_table, layer_size);

                // Build variance and rsqrt trace columns for LogUp
                let mut var_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut rsqrt_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let n = layer.variances.len().min(layer_size);
                for i in 0..n {
                    var_col.set(i, layer.variances[i]);
                    rsqrt_col.set(i, layer.rsqrt_vals[i]);
                }
                // Pad with table[0] values
                let pad_var = layer.rsqrt_table.inputs.first().copied().unwrap_or(M31::from(0));
                let pad_rsqrt = layer.rsqrt_table.outputs.first().copied().unwrap_or(M31::from(0));
                for i in n..layer_size {
                    var_col.set(i, pad_var);
                    rsqrt_col.set(i, pad_rsqrt);
                }

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
                let mut col_gen = logup_gen.new_col();

                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[table_var_col.data[vec_row], table_rsqrt_col.data[vec_row]],
                    );
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[var_col.data[vec_row], rsqrt_col.data[vec_row]],
                    );

                    let mult_packed = pack_multiplicities(&layernorm_mults[idx], vec_row);
                    let numerator = q_table - mult_packed * q_trace;
                    let denominator = q_table * q_trace;

                    col_gen.write_frac(vec_row, numerator, denominator);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                layernorm_claimed_sums.push(claimed_sum);
            }
        }

        // Embedding LogUp interaction traces
        if let Some(ref lookup) = embedding_lookup_rel {
            for layer in &embedding_layers {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;

                // Build SIMD columns for preprocessed table and trace
                let mut tbl_tok = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut tbl_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut tbl_val = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let n_table = layer.table_tokens.len().min(layer_size);
                for i in 0..n_table {
                    tbl_tok.set(i, layer.table_tokens[i]);
                    tbl_col.set(i, layer.table_cols[i]);
                    tbl_val.set(i, layer.table_values[i]);
                }

                let mut tr_tok = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut tr_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut tr_val = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let n_trace = layer.token_ids.len().min(layer_size);
                for i in 0..n_trace {
                    tr_tok.set(i, layer.token_ids[i]);
                    tr_col.set(i, layer.col_indices[i]);
                    tr_val.set(i, layer.values[i]);
                }

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);

                // Table side (yield with -multiplicity)
                let mut col_gen = logup_gen.new_col();
                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[tbl_tok.data[vec_row], tbl_col.data[vec_row], tbl_val.data[vec_row]],
                    );
                    let mult_packed = pack_multiplicities(&layer.multiplicities, vec_row);
                    col_gen.write_frac(vec_row, -mult_packed, q_table);
                }
                col_gen.finalize_col();

                // Trace side (use with +1)
                let mut col_gen = logup_gen.new_col();
                for vec_row in 0..layer_vec_size {
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[tr_tok.data[vec_row], tr_col.data[vec_row], tr_val.data[vec_row]],
                    );
                    col_gen.write_frac(vec_row, PackedSecureField::one(), q_trace);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                embedding_claimed_sums.push(claimed_sum);
            }
        }

        // Quantize LogUp interaction traces
        // Pattern: combine table-side (-mult) and trace-side (+1) into single column,
        // matching QuantizeEval's finalize_logup_in_pairs() which pairs both add_to_relation calls.
        if let Some(ref lookup) = quantize_lookup {
            for layer in &quantize_layers {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;

                // Build SIMD columns for range table and trace
                let range_table_col = build_quantize_range_table(layer.bits, layer_size);
                let (value_col, _mult_col) = build_quantize_trace_simd(
                    &layer.values, &layer.multiplicities, layer.bits, layer_size,
                );

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
                let mut col_gen = logup_gen.new_col();

                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[range_table_col.data[vec_row]],
                    );
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[value_col.data[vec_row]],
                    );

                    let mult_packed = pack_multiplicities(&layer.multiplicities, vec_row);
                    // Combined fraction: (-mult / q_table) + (1 / q_trace)
                    // = (-mult * q_trace + q_table) / (q_table * q_trace)
                    let numerator = q_table - mult_packed * q_trace;
                    let denominator = q_table * q_trace;

                    col_gen.write_frac(vec_row, numerator, denominator);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                quantize_claimed_sums.push(claimed_sum);
            }
        }

        tree_builder.commit(channel);
    } // end if has_logup

    // Build all components with shared allocator (same order as trace columns)
    let mut allocator = TraceLocationAllocator::default();
    let mut component_refs_storage: Vec<Box<dyn ComponentProverErased<B>>> = Vec::new();
    let mut activation_claims: Vec<LayerClaim> = Vec::new();
    let mut add_claims: Vec<LayerClaim> = Vec::new();
    let mut mul_claims: Vec<LayerClaim> = Vec::new();
    let mut layernorm_claims: Vec<LayerClaim> = Vec::new();
    let mut embedding_claims: Vec<LayerClaim> = Vec::new();
    let mut quantize_claims_vec: Vec<LayerClaim> = Vec::new();

    // Activation components
    if let Some(ref lookup) = activation_lookup {
        for (idx, layer) in activation_layers.iter().enumerate() {
            let claimed_sum = activation_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                ActivationEval {
                    log_n_rows: layer.log_size,
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                    total_sum: claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            activation_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Add components (pure AIR, no LogUp)
    for layer in &add_layers {
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseAddEval { log_n_rows: layer.log_size },
            SecureField::default(),
        );
        component_refs_storage.push(Box::new(component));
        add_claims.push(LayerClaim {
            layer_index: layer.node_id,
            claimed_sum: SecureField::default(),
            trace_rows: 1 << layer.log_size,
        });
    }

    // Mul components (pure AIR, no LogUp)
    for layer in &mul_layers {
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseMulEval { log_n_rows: layer.log_size },
            SecureField::default(),
        );
        component_refs_storage.push(Box::new(component));
        mul_claims.push(LayerClaim {
            layer_index: layer.node_id,
            claimed_sum: SecureField::default(),
            trace_rows: 1 << layer.log_size,
        });
    }

    // LayerNorm components (LogUp)
    if let Some(ref lookup) = layernorm_lookup {
        for (idx, layer) in layernorm_layers.iter().enumerate() {
            let claimed_sum = layernorm_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                LayerNormEval {
                    log_n_rows: layer.log_size,
                    dim: layer.inputs.len(),
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            layernorm_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Embedding components (LogUp)
    if let Some(ref lookup) = embedding_lookup_rel {
        for (idx, layer) in embedding_layers.iter().enumerate() {
            let claimed_sum = embedding_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                EmbeddingEval {
                    log_n_rows: layer.log_size,
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            embedding_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Quantize components (LogUp range-check)
    if let Some(ref lookup) = quantize_lookup {
        for (idx, layer) in quantize_layers.iter().enumerate() {
            let claimed_sum = quantize_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                QuantizeEval {
                    log_n_rows: layer.log_size,
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            quantize_claims_vec.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Single prove() call with all component refs
    let component_refs: Vec<&dyn ComponentProver<B>> = component_refs_storage
        .iter()
        .map(|c| c.as_component_prover())
        .collect();

    let stark_proof = prove::<B, MC>(
        &component_refs,
        channel,
        commitment_scheme,
    ).map_err(|e| AggregationError::ProvingError(format!("{e:?}")))?;

    Ok(AggregatedModelProofFor {
        unified_stark: Some(stark_proof),
        matmul_proofs,
        add_claims,
        mul_claims,
        layernorm_claims,
        execution,
        activation_claims,
        attention_proofs,
        embedding_claims,
        quantize_claims: quantize_claims_vec,
        layer_chain_commitment,
        io_commitment,
        layernorm_mean_var_commitments,
    })
}

/// Prove an entire computation graph with aggregated STARK proof.
///
/// Convenience wrapper using `SimdBackend` + `Blake2sMerkleChannel`.
pub fn prove_model_aggregated(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProof, AggregationError> {
    prove_model_aggregated_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
}

/// Prove a model with aggregation using the best available backend.
///
/// Uses `GpuBackend` when CUDA is available, otherwise `SimdBackend`.
pub fn prove_model_aggregated_auto(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProof, AggregationError> {
    let gpu_available = crate::backend::gpu_is_available();
    info!(
        gpu_available,
        "Auto-selecting backend for off-chain aggregation"
    );
    crate::backend::with_best_backend(
        || {
            info!("Using SimdBackend for off-chain aggregation");
            prove_model_aggregated_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
        },
        || {
            info!("Using GpuBackend for off-chain aggregation");
            prove_model_aggregated_gpu(graph, input, weights)
        },
    )
}

/// GPU aggregated proving path — dispatches to `GpuBackend` when `cuda-runtime` is enabled.
fn prove_model_aggregated_gpu(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProof, AggregationError> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return prove_model_aggregated_with::<GpuBackend, Blake2sMerkleChannel>(
            graph, input, weights,
        );
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        prove_model_aggregated_with::<SimdBackend, Blake2sMerkleChannel>(graph, input, weights)
    }
}

// ===== On-Chain Aggregated Proof =====

/// Aggregated model proof formatted for on-chain Cairo verification.
///
/// Uses `MatMulSumcheckProofOnChain` (Poseidon channel + MLE commitments)
/// instead of `MatMulSumcheckProof` (Blake2s).
pub struct AggregatedModelProofOnChain {
    /// Single STARK proof covering all non-matmul components.
    pub unified_stark: Option<StarkProof<Blake2sHash>>,
    /// Per-matmul on-chain sumcheck proofs, in layer order.
    /// Empty when `batched_matmul_proofs` is used instead.
    pub matmul_proofs: Vec<(usize, MatMulSumcheckProofOnChain)>,
    /// Batched matmul proofs — grouped by k dimension.
    /// When non-empty, these replace `matmul_proofs` (which will be empty).
    pub batched_matmul_proofs: Vec<BatchedMatMulProofOnChain>,
    /// Per-Add layer claims (verified inside unified STARK).
    pub add_claims: Vec<LayerClaim>,
    /// Per-Mul layer claims (verified inside unified STARK).
    pub mul_claims: Vec<LayerClaim>,
    /// Per-LayerNorm layer claims (verified inside unified STARK).
    pub layernorm_claims: Vec<LayerClaim>,
    /// Forward pass execution trace.
    pub execution: GraphExecution,
    /// Per-activation-layer claims.
    pub activation_claims: Vec<LayerClaim>,
    /// Per-attention on-chain proofs.
    pub attention_proofs: Vec<(usize, AttentionProofOnChain)>,
    /// Per-embedding layer claims (verified inside unified STARK).
    pub embedding_claims: Vec<LayerClaim>,
    /// Per-quantize layer claims (verified inside unified STARK via LogUp range-check).
    pub quantize_claims: Vec<LayerClaim>,
    /// Layer chain commitment: running Poseidon hash of intermediate values.
    /// Binds layer outputs to layer inputs, preventing substitution of intermediates.
    pub layer_chain_commitment: FieldElement,
    /// IO commitment: Poseidon(input_data || output_data).
    /// Binds the proof to specific input/output data, preventing replay with different I/O.
    pub io_commitment: FieldElement,
    /// Per-LayerNorm Poseidon commitments of (means, variances).
    /// Binds the prover to specific mean/variance values, preventing substitution.
    pub layernorm_mean_var_commitments: Vec<FieldElement>,
}

impl AggregatedModelProofOnChain {
    /// Total number of proven layers across all proof types.
    pub fn num_proven_layers(&self) -> usize {
        let batched_count: usize = self.batched_matmul_proofs.iter()
            .map(|b| b.entries.len()).sum();
        self.matmul_proofs.len() + batched_count + self.activation_claims.len()
            + self.add_claims.len() + self.mul_claims.len() + self.layernorm_claims.len()
            + self.attention_proofs.len() + self.embedding_claims.len()
            + self.quantize_claims.len()
    }

    /// Total number of matmul proofs (individual + batched entries).
    pub fn total_matmul_count(&self) -> usize {
        let batched: usize = self.batched_matmul_proofs.iter()
            .map(|b| b.entries.len()).sum();
        self.matmul_proofs.len() + batched
    }
}

/// Prove an entire computation graph with on-chain Poseidon-based matmul proofs.
///
/// Same as `prove_model_aggregated` but calls `prove_matmul_sumcheck_onchain_auto()`
/// for each matmul layer, producing proofs with Poseidon Merkle commitments
/// and MLE opening proofs compatible with the Cairo verifier.
///
/// The unified STARK covers all non-matmul components (already uses Blake2s).
pub fn prove_model_aggregated_onchain(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, AggregationError> {
    prove_model_aggregated_onchain_with::<SimdBackend>(graph, input, weights)
}

/// On-chain aggregated proving, generic over backend `B`.
///
/// The unified STARK always uses `Blake2sMerkleChannel` (matching the on-chain
/// verifier). Matmul sumcheck proofs use Poseidon (independent of `B`).
/// Genericizing `B` allows GPU acceleration of Merkle hashing, FRI, and
/// quotient evaluation for the unified STARK portion.
pub(crate) fn prove_model_aggregated_onchain_with<B>(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, AggregationError>
where
    B: BackendForChannel<Blake2sMerkleChannel> + PolyOps,
    FrameworkComponent<ActivationEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseAddEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseMulEval>: ComponentProver<B>,
    FrameworkComponent<LayerNormEval>: ComponentProver<B>,
    FrameworkComponent<EmbeddingEval>: ComponentProver<B>,
    FrameworkComponent<QuantizeEval>: ComponentProver<B>,
{
    prove_model_aggregated_onchain_with_cache::<B>(graph, input, weights, None)
}

/// On-chain aggregated proving with optional weight commitment cache.
///
/// When `weight_cache` is `Some`, uses cached restricted weight MLEs and
/// commitments for batch entry preparation. On cache miss, computes and stores.
pub(crate) fn prove_model_aggregated_onchain_with_cache<B>(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    _weight_cache: Option<&crate::weight_cache::SharedWeightCache>,
) -> Result<AggregatedModelProofOnChain, AggregationError>
where
    B: BackendForChannel<Blake2sMerkleChannel> + PolyOps,
    FrameworkComponent<ActivationEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseAddEval>: ComponentProver<B>,
    FrameworkComponent<ElementwiseMulEval>: ComponentProver<B>,
    FrameworkComponent<LayerNormEval>: ComponentProver<B>,
    FrameworkComponent<EmbeddingEval>: ComponentProver<B>,
    FrameworkComponent<QuantizeEval>: ComponentProver<B>,
{
    info!(
        backend = std::any::type_name::<B>(),
        "Proving unified STARK (on-chain aggregation, Blake2sMerkleChannel)"
    );

    // Print GPU status banner
    let gpu_active = crate::backend::gpu_is_available();
    let backend_name = std::any::type_name::<B>();
    let is_gpu_backend = backend_name.contains("Gpu");
    eprintln!("=== Proving Pipeline ===");
    eprintln!("  Backend: {} {}", backend_name,
        if is_gpu_backend { "[GPU]" } else { "[CPU/SIMD]" });
    eprintln!("  Sumcheck: {} {}",
        if gpu_active { "GPU-accelerated" } else { "CPU" },
        if gpu_active { "[GPU]" } else { "[CPU]" });
    if let Some(dev) = crate::backend::gpu_device_name() {
        if let Some(mem) = crate::backend::gpu_available_memory() {
            eprintln!("  Device: {} ({:.1} GB)", dev, mem as f64 / 1e9);
        } else {
            eprintln!("  Device: {}", dev);
        }
    }
    eprintln!("========================");

    let config = PcsConfig::default();
    let mut intermediates: Vec<(usize, M31Matrix)> = Vec::new();
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();
    let mut current = input.clone();

    let mut matmul_data: Vec<DeferredMatMul> = Vec::new();
    let mut matmul_proofs: Vec<(usize, MatMulSumcheckProofOnChain)> = Vec::new();
    let mut activation_layers: Vec<ActivationLayerData> = Vec::new();
    let mut add_layers: Vec<AddLayerData> = Vec::new();
    let mut mul_layers: Vec<MulLayerData> = Vec::new();
    let mut layernorm_layers: Vec<LayerNormLayerData> = Vec::new();
    let mut attention_layers: Vec<AttentionLayerData> = Vec::new();
    let mut embedding_layers: Vec<EmbeddingLayerData> = Vec::new();
    let mut quantize_layers: Vec<QuantizeLayerData> = Vec::new();

    // Memory budget for tiled matmul auto-dispatch.
    // 64GB — conservative for H200 (143GB). A 5120x17408 sumcheck needs ~4.6GB,
    // well within GPU memory. Only tile truly massive matmuls.
    const TILED_MEMORY_BUDGET: usize = 64 * 1024 * 1024 * 1024;

    let topo = graph.topological_order();
    let total_nodes = topo.len();
    let t_start = std::time::Instant::now();
    eprintln!("Phase 1/3: Forward pass ({} nodes)...", total_nodes);
    for (step, &node_id) in topo.iter().enumerate() {
        let node = &graph.nodes[node_id];

        // Resolve current input from first dependency
        if let Some(&first_input) = node.inputs.first() {
            if let Some(inp) = node_outputs.get(&first_input) {
                current = inp.clone();
            }
        }

        match &node.op {
            GraphOp::MatMul { dims } => {
                let (m, k, n) = *dims;
                eprintln!(
                    "  [{}/{}] Node {} MatMul {}x{}x{} — forward pass",
                    step + 1, total_nodes, node.id, m, k, n,
                );

                let weight = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;

                // GPU GEMM for all matrix sizes, CPU fallback
                #[cfg(feature = "cuda-runtime")]
                let output = crate::gpu_sumcheck::gpu_matmul_m31_full(&current, weight)
                    .unwrap_or_else(|_| matmul_m31(&current, weight));
                #[cfg(not(feature = "cuda-runtime"))]
                let output = matmul_m31(&current, weight);

                // Collect for deferred proving (Phase 2)
                matmul_data.push(DeferredMatMul {
                    node_id: node.id,
                    a: current.clone(),
                    c: output.clone(),
                    dims: (m, k, n),
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Activation { activation_type, .. } => {
                eprintln!(
                    "[{}/{}] Node {} Activation {:?}",
                    step + 1, total_nodes, node.id, activation_type,
                );
                let f = activation_type.as_fn();
                let output = crate::compiler::prove::apply_activation_pub(&current, &*f);

                let act_log_size = activation_type.recommended_table_log_size();
                let table = PrecomputedTable::build(|x| (*f)(x), act_log_size);
                let flat_inputs = current.data.clone();
                let flat_outputs = output.data.clone();

                activation_layers.push(ActivationLayerData {
                    node_id: node.id,
                    inputs: flat_inputs,
                    outputs: flat_outputs,
                    table,
                    log_size: act_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Add { .. } => {
                eprintln!(
                    "[{}/{}] Node {} Add ({} elements)",
                    step + 1, total_nodes, node.id, current.data.len(),
                );
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());

                #[cfg(feature = "cuda-runtime")]
                let output = {
                    let rows = lhs.rows.max(rhs.rows);
                    let cols = lhs.cols.max(rhs.cols);
                    crate::gpu_sumcheck::gpu_elementwise_add(&lhs.data, &rhs.data)
                        .map(|data| M31Matrix { rows, cols, data })
                        .unwrap_or_else(|_| elementwise_add(&lhs, &rhs))
                };
                #[cfg(not(feature = "cuda-runtime"))]
                let output = elementwise_add(&lhs, &rhs);

                let add_log_size = data_log_size(output.data.len());
                add_layers.push(AddLayerData {
                    node_id: node.id,
                    lhs: lhs.data.clone(),
                    rhs: rhs.data.clone(),
                    output: output.data.clone(),
                    log_size: add_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Mul { .. } => {
                eprintln!(
                    "[{}/{}] Node {} Mul ({} elements)",
                    step + 1, total_nodes, node.id, current.data.len(),
                );
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());

                #[cfg(feature = "cuda-runtime")]
                let output = {
                    let rows = lhs.rows.max(rhs.rows);
                    let cols = lhs.cols.max(rhs.cols);
                    crate::gpu_sumcheck::gpu_elementwise_mul(&lhs.data, &rhs.data)
                        .map(|data| M31Matrix { rows, cols, data })
                        .unwrap_or_else(|_| elementwise_mul(&lhs, &rhs))
                };
                #[cfg(not(feature = "cuda-runtime"))]
                let output = elementwise_mul(&lhs, &rhs);

                let mul_log_size = data_log_size(output.data.len());
                mul_layers.push(MulLayerData {
                    node_id: node.id,
                    lhs: lhs.data.clone(),
                    rhs: rhs.data.clone(),
                    output: output.data.clone(),
                    log_size: mul_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::LayerNorm { dim } => {
                eprintln!(
                    "[{}/{}] Node {} LayerNorm (dim={})",
                    step + 1, total_nodes, node.id, dim,
                );
                let ln_log_size = LayerNormConfig::new(*dim).rsqrt_table_log_size;
                let ln = apply_layernorm_detailed(&current, *dim);
                let rsqrt_table = build_rsqrt_table(ln_log_size);

                layernorm_layers.push(LayerNormLayerData {
                    node_id: node.id,
                    inputs: ln.inputs.clone(),
                    means: ln.means.clone(),
                    variances: ln.variances.clone(),
                    rsqrt_vals: ln.rsqrt_vals.clone(),
                    outputs: ln.outputs.clone(),
                    rsqrt_table,
                    log_size: ln_log_size,
                });

                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, ln.output_matrix.clone());
                current = ln.output_matrix;
            }

            GraphOp::Attention { config: attn_config } => {
                let tag = if gpu_active { "[GPU]" } else { "[CPU]" };
                eprintln!(
                    "{} [{}/{}] Node {} Attention (heads={}, d_model={})",
                    tag, step + 1, total_nodes, node.id,
                    attn_config.num_heads, attn_config.d_model,
                );
                let t_node = std::time::Instant::now();

                let w_q = weights.get_named_weight(node.id, "w_q");
                let w_k = weights.get_named_weight(node.id, "w_k");
                let w_v = weights.get_named_weight(node.id, "w_v");
                let w_o = weights.get_named_weight(node.id, "w_o");

                if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (w_q, w_k, w_v, w_o) {
                    let attn_weights = AttentionWeights {
                        w_q: wq.clone(), w_k: wk.clone(),
                        w_v: wv.clone(), w_o: wo.clone(),
                    };
                    let inter = attention_forward(&current, &attn_weights, attn_config, false);
                    attention_layers.push(AttentionLayerData {
                        node_id: node.id,
                        config: *attn_config,
                        weights: attn_weights,
                        input: current.clone(),
                    });
                    intermediates.push((node.id, current.clone()));
                    node_outputs.insert(node.id, inter.final_output.clone());
                    current = inter.final_output;
                    eprintln!(
                        "  -> {} done in {:.2}s (elapsed: {:.1}s)",
                        tag, t_node.elapsed().as_secs_f64(),
                        t_start.elapsed().as_secs_f64(),
                    );
                } else {
                    eprintln!("  -> missing weights, passthrough");
                    intermediates.push((node.id, current.clone()));
                    node_outputs.insert(node.id, current.clone());
                }
            }

            GraphOp::Embedding { vocab_size, embed_dim } => {
                eprintln!(
                    "[{}/{}] Node {} Embedding (vocab={}, dim={})",
                    step + 1, total_nodes, node.id, vocab_size, embed_dim,
                );
                let embed_table = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let token_u32s: Vec<u32> = current.data.iter().map(|m| m.0).collect();
                let (output, token_ids, col_indices, values, multiplicities) =
                    embedding_lookup(&token_u32s, embed_table);
                let (table_tokens, table_cols, table_values) = build_embedding_table_columns(embed_table);
                let log_size = data_log_size(values.len());
                embedding_layers.push(EmbeddingLayerData {
                    node_id: node.id,
                    token_ids,
                    col_indices,
                    values,
                    multiplicities,
                    table_tokens,
                    table_cols,
                    table_values,
                    log_size,
                });
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            GraphOp::Conv2D { in_channels, out_channels, kernel_size, stride, padding } => {
                let tag = if gpu_active { "[GPU]" } else { "[CPU]" };
                eprintln!(
                    "{} [{}/{}] Node {} Conv2D ({}->{}ch, k={}, s={}, p={})",
                    tag, step + 1, total_nodes, node.id,
                    in_channels, out_channels, kernel_size, stride, padding,
                );
                let t_node = std::time::Instant::now();

                let kernel = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let im2col_config = Im2ColConfig {
                    in_channels: *in_channels,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    input_h: current.rows,
                    input_w: current.cols / in_channels,
                };
                let (im2col_mat, kernel_mat, output) =
                    conv2d_forward(&current.data, &kernel.data, &im2col_config, *out_channels);
                // Conv2D matmul uses on-chain proving
                let proof = prove_matmul_sumcheck_onchain_auto(&im2col_mat, &kernel_mat, &output)
                    .map_err(|e| ModelError::ProvingError {
                        layer: node.id,
                        message: format!("Conv2D matmul (on-chain): {e}"),
                    })?;
                intermediates.push((node.id, current.clone()));
                matmul_proofs.push((node.id, proof));
                node_outputs.insert(node.id, output.clone());
                current = output;
                eprintln!(
                    "  -> done in {:.2}s (elapsed: {:.1}s)",
                    t_node.elapsed().as_secs_f64(),
                    t_start.elapsed().as_secs_f64(),
                );
            }

            GraphOp::Quantize { params, size: _ } => {
                eprintln!(
                    "[{}/{}] Node {} Quantize (bits={})",
                    step + 1, total_nodes, node.id, params.bits,
                );
                let quantized: Vec<M31> = current.data.iter()
                    .map(|v| {
                        let val = v.0;
                        let max_val = (1u32 << params.bits) - 1;
                        M31::from(val.min(max_val))
                    })
                    .collect();

                let table_size = 1u32 << params.bits;
                let mut multiplicities = vec![M31::from(0); table_size as usize];
                for v in &quantized {
                    let idx = v.0 as usize;
                    if idx < multiplicities.len() {
                        multiplicities[idx] = M31::from(multiplicities[idx].0 + 1);
                    }
                }

                let log_size = data_log_size(quantized.len().max(table_size as usize));
                quantize_layers.push(QuantizeLayerData {
                    node_id: node.id,
                    values: quantized.clone(),
                    multiplicities,
                    bits: params.bits,
                    log_size,
                });

                let output = M31Matrix {
                    rows: current.rows,
                    cols: current.cols,
                    data: quantized,
                };
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }

            _ => {
                // Identity, etc. — passthrough
                intermediates.push((node.id, current.clone()));
                node_outputs.insert(node.id, current.clone());
            }
        }
    }

    eprintln!(
        "Phase 1 complete: forward pass ({} nodes in {:.1}s)",
        total_nodes, t_start.elapsed().as_secs_f64(),
    );
    eprintln!(
        "  Deferred: {} matmuls, {} attention, {} activations, {} add, {} mul, {} layernorm, {} quantize",
        matmul_data.len(), attention_layers.len(), activation_layers.len(),
        add_layers.len(), mul_layers.len(), layernorm_layers.len(), quantize_layers.len(),
    );

    // =====================================================================
    // Phase 2: Prove all matmul sumchecks (GPU batch when available)
    // =====================================================================
    let tag = if gpu_active { "[GPU]" } else { "[CPU]" };
    let t_phase2 = std::time::Instant::now();
    #[allow(unused_mut)]
    let mut batched_matmul_proofs: Vec<BatchedMatMulProofOnChain> = Vec::new();

    // Try batch proving: group matmuls by k dimension, prove each group in one combined sumcheck.
    // Falls back to individual proving if GPU batch isn't available.
    let batch_available = {
        #[cfg(feature = "cuda-runtime")]
        { gpu_active }
        #[cfg(not(feature = "cuda-runtime"))]
        { false }
    };

    if batch_available && !matmul_data.is_empty() {
        // Group matmuls by padded k dimension (all entries in a batch must share k).
        let mut dim_groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        for (idx, dm) in matmul_data.iter().enumerate() {
            let k = dm.dims.1.next_power_of_two();
            dim_groups.entry(k).or_default().push(idx);
        }
        eprintln!(
            "Phase 2/3: {} Batch proving {} matmuls in {} dimension groups...",
            tag, matmul_data.len(), dim_groups.len(),
        );
        for (group_idx, (&k_dim, indices)) in dim_groups.iter().enumerate() {
            eprintln!(
                "  {} Group {}/{}: k={}, {} matmuls",
                tag, group_idx + 1, dim_groups.len(), k_dim, indices.len(),
            );
            let t_group = std::time::Instant::now();

            // Prepare batch entries and GPU batch prove (all behind cuda-runtime gate)
            #[cfg(feature = "cuda-runtime")]
            {
                use rayon::prelude::*;

                eprintln!("    Preparing {} batch entries (parallel chunks of 16)...", indices.len());
                let t_prep = std::time::Instant::now();

                // Extract A/C from matmul_data for parallel processing.
                // Weight B is looked up from GraphWeights (shared ref, thread-safe).
                let prep_inputs: Vec<(usize, M31Matrix, M31Matrix)> = indices.iter().map(|&idx| {
                    let dm = &mut matmul_data[idx];
                    let a = std::mem::replace(&mut dm.a, M31Matrix::new(0, 0));
                    let c = std::mem::replace(&mut dm.c, M31Matrix::new(0, 0));
                    (dm.node_id, a, c)
                }).collect();
                let total = prep_inputs.len();

                // Process in chunks of 16 to bound peak memory:
                // Each entry peaks at ~5 GB temp (padded MLE + restrict),
                // freed after restriction to ~128 KB via shrink_to_fit().
                // Peak: 37 GB weights + 16 × 5 GB temp = ~117 GB < 196 GB RAM.
                let done_count = std::sync::atomic::AtomicUsize::new(0);
                let mut entries: Vec<crate::gpu_sumcheck::BatchEntry> = Vec::with_capacity(total);
                let mut prep_error: Option<ModelError> = None;
                let num_chunks = (total + 15) / 16;

                for (chunk_idx, chunk) in prep_inputs.chunks(16).enumerate() {
                    let chunk_entries: Vec<Result<crate::gpu_sumcheck::BatchEntry, ModelError>> = chunk
                        .par_iter()
                        .map(|(node_id, a, c)| {
                            let weight_b = weights.get_weight(*node_id).expect("weight exists");
                            let entry = if let Some(cache) = _weight_cache {
                                crate::gpu_sumcheck::prepare_batch_entry_cached(*node_id, a, weight_b, c, cache)
                                    .map_err(|e| ModelError::ProvingError {
                                        layer: *node_id,
                                        message: format!("Batch prep (cached): {e}"),
                                    })?
                            } else {
                                crate::gpu_sumcheck::prepare_batch_entry(*node_id, a, weight_b, c)
                                    .map_err(|e| ModelError::ProvingError {
                                        layer: *node_id,
                                        message: format!("Batch prep: {e}"),
                                    })?
                            };
                            done_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            Ok(entry)
                        })
                        .collect();

                    for result in chunk_entries {
                        match result {
                            Ok(entry) => entries.push(entry),
                            Err(e) => { prep_error = Some(e); break; }
                        }
                    }
                    if prep_error.is_some() { break; }

                    // Report progress with RSS
                    let rss_mb = {
                        #[cfg(target_os = "linux")]
                        {
                            std::fs::read_to_string("/proc/self/status")
                                .ok()
                                .and_then(|s| s.lines()
                                    .find(|l| l.starts_with("VmRSS:"))
                                    .and_then(|l| l.split_whitespace().nth(1))
                                    .and_then(|v| v.parse::<u64>().ok()))
                                .map(|kb| kb / 1024)
                                .unwrap_or(0)
                        }
                        #[cfg(not(target_os = "linux"))]
                        { 0u64 }
                    };
                    eprintln!(
                        "      Chunk {}/{}: {}/{} entries prepared ({:.1}s, RSS: {} MB)",
                        chunk_idx + 1, num_chunks, entries.len(), total,
                        t_prep.elapsed().as_secs_f64(), rss_mb,
                    );
                }
                if let Some(e) = prep_error {
                    return Err(AggregationError::ProvingError(format!("Batch prep: {e:?}")));
                }
                eprintln!("    Prep done in {:.1}s", t_prep.elapsed().as_secs_f64());

                // GPU batch prove
                eprintln!("    {} Batch sumcheck ({} entries, k={})...", tag, entries.len(), k_dim);
                let t_prove = std::time::Instant::now();

                let batch_result = crate::gpu_sumcheck::prove_matmul_batch_onchain_gpu(&entries)
                    .map_err(|e| AggregationError::ProvingError(
                        format!("Batch sumcheck k={k_dim}: {e}"),
                    ))?;

                // Compute combined claimed sum for verification
                let mut combined = SecureField::zero();
                let mut lambda_pow = SecureField::one();
                for entry in &entries {
                    combined = combined + lambda_pow * entry.claimed_sum;
                    lambda_pow = lambda_pow * batch_result.lambda;
                }

                // Build batched proof struct
                let batch_entries: Vec<BatchedMatMulEntryOnChain> = entries.iter()
                    .zip(batch_result.per_matmul.iter())
                    .map(|(entry, result)| BatchedMatMulEntryOnChain {
                        node_id: entry.node_id,
                        m: entry.m as u32,
                        n: entry.n as u32,
                        claimed_sum: entry.claimed_sum,
                        final_a_eval: result.final_a_eval,
                        final_b_eval: result.final_b_eval,
                        a_commitment: entry.a_commitment,
                        b_commitment: entry.b_commitment,
                    })
                    .collect();

                batched_matmul_proofs.push(BatchedMatMulProofOnChain {
                    k: k_dim as u32,
                    num_rounds: batch_result.round_polys.len() as u32,
                    lambda: batch_result.lambda,
                    combined_claimed_sum: combined,
                    round_polys: batch_result.round_polys,
                    entries: batch_entries,
                });

                // Explicitly free batch entries (f_a/f_b vectors) before next group
                drop(entries);
            }
            eprintln!(
                "    -> {} Group done in {:.2}s (elapsed: {:.1}s)",
                tag, t_group.elapsed().as_secs_f64(),
                t_start.elapsed().as_secs_f64(),
            );
        }
        eprintln!(
            "Phase 2 complete: {} batched groups ({} total matmuls) in {:.1}s",
            batched_matmul_proofs.len(),
            batched_matmul_proofs.iter().map(|b| b.entries.len()).sum::<usize>(),
            t_phase2.elapsed().as_secs_f64(),
        );
    } else {
        // Fallback: individual proving (no GPU batch available)
        eprintln!(
            "Phase 2/3: {} Proving {} matmul sumchecks (individual)...",
            tag, matmul_data.len(),
        );
        for (i, dm) in matmul_data.iter().enumerate() {
            let (m, k, n) = dm.dims;
            eprintln!(
                "  {} [{}/{}] Node {} MatMul {}x{}x{}",
                tag, i + 1, matmul_data.len(), dm.node_id, m, k, n,
            );
            let t_node = std::time::Instant::now();

            let (_, estimated_mem) = estimate_sumcheck_memory(m, k, n);

            let weight_b = weights.get_weight(dm.node_id).expect("weight exists");
            let proof = if estimated_mem > TILED_MEMORY_BUDGET {
                let config = TiledMatMulConfig::from_memory_budget(
                    m, k, n, TILED_MEMORY_BUDGET,
                );
                let tiled = prove_tiled_matmul(&dm.a, weight_b, &dm.c, &config)
                    .map_err(|e| ModelError::ProvingError {
                        layer: dm.node_id,
                        message: format!("Tiled matmul: {e}"),
                    })?;
                compose_tiled_proof(&tiled)
            } else {
                prove_matmul_sumcheck_onchain_auto(&dm.a, weight_b, &dm.c)
                    .map_err(|e| ModelError::ProvingError {
                        layer: dm.node_id,
                        message: format!("MatMul sumcheck (on-chain): {e}"),
                    })?
            };

            eprintln!(
                "    -> {} done in {:.2}s (elapsed: {:.1}s)",
                tag, t_node.elapsed().as_secs_f64(), t_start.elapsed().as_secs_f64(),
            );

            matmul_proofs.push((dm.node_id, proof));
        }
        eprintln!(
            "Phase 2 complete: {} individual matmul proofs in {:.1}s",
            matmul_proofs.len(), t_phase2.elapsed().as_secs_f64(),
        );
    }

    // =====================================================================
    // Phase 2b: Prove attention layers (on-chain format, GPU sumchecks)
    // =====================================================================
    let attn_tag = if gpu_active { "[GPU]" } else { "[CPU]" };
    let mut attention_proofs = Vec::new();
    for (i, layer) in attention_layers.iter().enumerate() {
        eprintln!(
            "{} Proving attention sumchecks [{}/{}] (node {}, 6 matmuls)...",
            attn_tag, i + 1, attention_layers.len(), layer.node_id,
        );
        let t_attn = std::time::Instant::now();
        let proof = prove_attention_onchain(
            &layer.input, &layer.weights, &layer.config, false,
        ).map_err(|e| AggregationError::ProvingError(
            format!("Attention node {} (on-chain): {e}", layer.node_id),
        ))?;
        eprintln!(
            "  -> {} attention node {} proved in {:.2}s (elapsed: {:.1}s)",
            attn_tag, layer.node_id, t_attn.elapsed().as_secs_f64(), t_start.elapsed().as_secs_f64(),
        );
        attention_proofs.push((layer.node_id, proof));
    }

    // Compute layer chain commitment before consuming `intermediates`.
    let layer_chain_commitment = compute_layer_chain_commitment(
        input, &intermediates, &current,
    );
    // Compute IO commitment binding this proof to specific input/output.
    let io_commitment = compute_io_commitment(input, &current);

    // Compute per-LayerNorm mean/variance commitments (defense-in-depth).
    let layernorm_mean_var_commitments: Vec<FieldElement> = layernorm_layers.iter()
        .map(|layer| compute_layernorm_mean_var_commitment(&layer.means, &layer.variances))
        .collect();

    let execution = GraphExecution {
        intermediates,
        output: current,
    };

    // Check if there are any non-matmul components to aggregate
    let has_components = !activation_layers.is_empty()
        || !add_layers.is_empty() || !mul_layers.is_empty() || !layernorm_layers.is_empty()
        || !embedding_layers.is_empty() || !quantize_layers.is_empty();

    if !has_components {
        return Ok(AggregatedModelProofOnChain {
            unified_stark: None,
            matmul_proofs,
            batched_matmul_proofs,
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            execution,
            activation_claims: Vec::new(),
            attention_proofs,
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            layer_chain_commitment,
            io_commitment,
            layernorm_mean_var_commitments: Vec::new(),
        });
    }

    // Build unified STARK using backend B + Blake2sMerkleChannel
    eprintln!("Phase 3/3: {} Building unified STARK proof (all non-matmul components)...", tag);
    let t_stark = std::time::Instant::now();
    // Per-component log_sizes: each component uses its own size derived from
    // its table or data length. The max_log_size drives twiddle precomputation.
    let all_log_sizes: Vec<u32> = activation_layers.iter().map(|l| l.log_size)
        .chain(add_layers.iter().map(|l| l.log_size))
        .chain(mul_layers.iter().map(|l| l.log_size))
        .chain(layernorm_layers.iter().map(|l| l.log_size))
        .chain(embedding_layers.iter().map(|l| l.log_size))
        .chain(quantize_layers.iter().map(|l| l.log_size))
        .collect();
    let max_log_size = *all_log_sizes.iter().max().unwrap();

    let max_degree_bound = max_log_size + 1;
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(max_degree_bound + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, Blake2sMerkleChannel>::new(config, &twiddles);

    let has_logup = !activation_layers.is_empty() || !layernorm_layers.is_empty()
        || !embedding_layers.is_empty() || !quantize_layers.is_empty();

    // Tree 0: Preprocessed (activation tables + layernorm rsqrt tables + embedding tables + quantize range tables)
    {
        let mut tree_builder = commitment_scheme.tree_builder();
        for layer in &activation_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let (table_input_col, table_output_col) = build_table_columns(&layer.table, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, table_input_col),
                CircleEvaluation::new(layer_domain, table_output_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        for layer in &layernorm_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let (table_var_col, table_rsqrt_col) = build_table_columns(&layer.rsqrt_table, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, table_var_col),
                CircleEvaluation::new(layer_domain, table_rsqrt_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        for layer in &embedding_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let simd_evals = build_embedding_preprocessed_columns(
                &layer.table_tokens, &layer.table_cols, &layer.table_values, layer_size, layer_domain,
            );
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        for layer in &quantize_layers {
            let layer_size = 1usize << layer.log_size;
            let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
            let range_table_col = build_quantize_range_table(layer.bits, layer_size);
            let simd_evals = vec![
                CircleEvaluation::new(layer_domain, range_table_col),
            ];
            tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        }
        tree_builder.commit(channel);
    }

    // Tree 1: Execution traces (activation + add + mul + layernorm + embedding + quantize)
    let mut tree_builder = commitment_scheme.tree_builder();
    let mut activation_mults: Vec<Vec<M31>> = Vec::new();
    for layer in &activation_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let pad_input = layer.table.inputs[0];
        let pad_output = layer.table.outputs[0];
        let padding_count = layer_size.saturating_sub(layer.inputs.len());

        let mut mults = compute_multiplicities(&layer.inputs, &layer.table);
        if padding_count > 0 {
            mults[0] += M31::from(padding_count as u32);
        }

        let (trace_in, trace_out, mult_col) = build_trace_columns(
            &layer.inputs, &layer.outputs, &mults,
            pad_input, pad_output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, trace_in),
            CircleEvaluation::new(layer_domain, trace_out),
            CircleEvaluation::new(layer_domain, mult_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
        activation_mults.push(mults);
    }
    for layer in &add_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let (lhs_col, rhs_col, out_col) = build_elementwise_trace_columns(
            &layer.lhs, &layer.rhs, &layer.output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, lhs_col),
            CircleEvaluation::new(layer_domain, rhs_col),
            CircleEvaluation::new(layer_domain, out_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    for layer in &mul_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let (lhs_col, rhs_col, out_col) = build_elementwise_trace_columns(
            &layer.lhs, &layer.rhs, &layer.output, layer_size,
        );
        let simd_evals = vec![
            CircleEvaluation::new(layer_domain, lhs_col),
            CircleEvaluation::new(layer_domain, rhs_col),
            CircleEvaluation::new(layer_domain, out_col),
        ];
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    let mut layernorm_mults: Vec<Vec<M31>> = Vec::new();
    for layer in &layernorm_layers {
        let layer_size = 1usize << layer.log_size;
        let mults = compute_multiplicities(&layer.variances, &layer.rsqrt_table);
        let cols = build_layernorm_trace_columns(
            &layer.inputs, &layer.means, &layer.variances,
            &layer.rsqrt_vals, &layer.outputs, &mults,
            &layer.rsqrt_table, layer_size,
        );
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(cols));
        layernorm_mults.push(mults);
    }
    for layer in &embedding_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let simd_evals = build_embedding_trace_columns(
            &layer.token_ids, &layer.col_indices, &layer.values, &layer.multiplicities,
            layer_size, layer_domain,
        );
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    for layer in &quantize_layers {
        let layer_size = 1usize << layer.log_size;
        let layer_domain = CanonicCoset::new(layer.log_size).circle_domain();
        let simd_evals = build_quantize_trace_columns(
            &layer.values, &layer.multiplicities, layer.bits, layer_size, layer_domain,
        );
        tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(simd_evals));
    }
    tree_builder.commit(channel);

    // Draw relation elements and build Tree 2 — only if LogUp components exist
    let mut activation_lookup: Option<ActivationRelation> = None;
    let mut layernorm_lookup: Option<LayerNormRelation> = None;
    let mut embedding_lookup_rel: Option<EmbeddingRelation> = None;
    let mut quantize_lookup: Option<QuantizeRelation> = None;
    let mut activation_claimed_sums: Vec<SecureField> = Vec::new();
    let mut layernorm_claimed_sums: Vec<SecureField> = Vec::new();
    let mut embedding_claimed_sums: Vec<SecureField> = Vec::new();
    let mut quantize_claimed_sums: Vec<SecureField> = Vec::new();

    if has_logup {
        if !activation_layers.is_empty() {
            activation_lookup = Some(ActivationRelation::draw(channel));
        }
        if !layernorm_layers.is_empty() {
            layernorm_lookup = Some(LayerNormRelation::draw(channel));
        }
        if !embedding_layers.is_empty() {
            embedding_lookup_rel = Some(EmbeddingRelation::draw(channel));
        }
        if !quantize_layers.is_empty() {
            quantize_lookup = Some(QuantizeRelation::draw(channel));
        }

        // Tree 2: Interaction traces (LogUp for activation + layernorm + embedding)
        let mut tree_builder = commitment_scheme.tree_builder();

        if let Some(ref lookup) = activation_lookup {
            for (idx, layer) in activation_layers.iter().enumerate() {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;
                let pad_input = layer.table.inputs[0];
                let pad_output = layer.table.outputs[0];

                let (table_in_col, table_out_col) = build_table_columns(&layer.table, layer_size);
                let (trace_in_col, trace_out_col, _) = build_trace_columns(
                    &layer.inputs, &layer.outputs, &activation_mults[idx],
                    pad_input, pad_output, layer_size,
                );

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
                let mut col_gen = logup_gen.new_col();

                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[table_in_col.data[vec_row], table_out_col.data[vec_row]],
                    );
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[trace_in_col.data[vec_row], trace_out_col.data[vec_row]],
                    );

                    let mult_packed = pack_multiplicities(&activation_mults[idx], vec_row);
                    let numerator = q_table - mult_packed * q_trace;
                    let denominator = q_table * q_trace;

                    col_gen.write_frac(vec_row, numerator, denominator);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                activation_claimed_sums.push(claimed_sum);
            }
        }

        if let Some(ref lookup) = layernorm_lookup {
            for (idx, layer) in layernorm_layers.iter().enumerate() {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;
                let (table_var_col, table_rsqrt_col) = build_table_columns(&layer.rsqrt_table, layer_size);

                let mut var_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut rsqrt_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let n = layer.variances.len().min(layer_size);
                for i in 0..n {
                    var_col.set(i, layer.variances[i]);
                    rsqrt_col.set(i, layer.rsqrt_vals[i]);
                }
                let pad_var = layer.rsqrt_table.inputs.first().copied().unwrap_or(M31::from(0));
                let pad_rsqrt = layer.rsqrt_table.outputs.first().copied().unwrap_or(M31::from(0));
                for i in n..layer_size {
                    var_col.set(i, pad_var);
                    rsqrt_col.set(i, pad_rsqrt);
                }

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
                let mut col_gen = logup_gen.new_col();

                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[table_var_col.data[vec_row], table_rsqrt_col.data[vec_row]],
                    );
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[var_col.data[vec_row], rsqrt_col.data[vec_row]],
                    );

                    let mult_packed = pack_multiplicities(&layernorm_mults[idx], vec_row);
                    let numerator = q_table - mult_packed * q_trace;
                    let denominator = q_table * q_trace;

                    col_gen.write_frac(vec_row, numerator, denominator);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                layernorm_claimed_sums.push(claimed_sum);
            }
        }

        // Embedding LogUp interaction traces
        if let Some(ref lookup) = embedding_lookup_rel {
            for layer in &embedding_layers {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;

                let mut tbl_tok = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut tbl_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut tbl_val = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let n_table = layer.table_tokens.len().min(layer_size);
                for i in 0..n_table {
                    tbl_tok.set(i, layer.table_tokens[i]);
                    tbl_col.set(i, layer.table_cols[i]);
                    tbl_val.set(i, layer.table_values[i]);
                }

                let mut tr_tok = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut tr_col = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let mut tr_val = Col::<SimdBackend, BaseField>::zeros(layer_size);
                let n_trace = layer.token_ids.len().min(layer_size);
                for i in 0..n_trace {
                    tr_tok.set(i, layer.token_ids[i]);
                    tr_col.set(i, layer.col_indices[i]);
                    tr_val.set(i, layer.values[i]);
                }

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);

                let mut col_gen = logup_gen.new_col();
                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[tbl_tok.data[vec_row], tbl_col.data[vec_row], tbl_val.data[vec_row]],
                    );
                    let mult_packed = pack_multiplicities(&layer.multiplicities, vec_row);
                    col_gen.write_frac(vec_row, -mult_packed, q_table);
                }
                col_gen.finalize_col();

                let mut col_gen = logup_gen.new_col();
                for vec_row in 0..layer_vec_size {
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[tr_tok.data[vec_row], tr_col.data[vec_row], tr_val.data[vec_row]],
                    );
                    col_gen.write_frac(vec_row, PackedSecureField::one(), q_trace);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                embedding_claimed_sums.push(claimed_sum);
            }
        }

        // Quantize LogUp interaction traces
        // Combined single-col pattern matching QuantizeEval's finalize_logup_in_pairs()
        if let Some(ref lookup) = quantize_lookup {
            for layer in &quantize_layers {
                let layer_size = 1usize << layer.log_size;
                let layer_vec_size = layer_size >> LOG_N_LANES;

                let range_table_col = build_quantize_range_table(layer.bits, layer_size);
                let (value_col, _mult_col) = build_quantize_trace_simd(
                    &layer.values, &layer.multiplicities, layer.bits, layer_size,
                );

                let mut logup_gen = LogupTraceGenerator::new(layer.log_size);
                let mut col_gen = logup_gen.new_col();

                for vec_row in 0..layer_vec_size {
                    let q_table: PackedSecureField = lookup.lookup_elements().combine(
                        &[range_table_col.data[vec_row]],
                    );
                    let q_trace: PackedSecureField = lookup.lookup_elements().combine(
                        &[value_col.data[vec_row]],
                    );

                    let mult_packed = pack_multiplicities(&layer.multiplicities, vec_row);
                    let numerator = q_table - mult_packed * q_trace;
                    let denominator = q_table * q_trace;

                    col_gen.write_frac(vec_row, numerator, denominator);
                }
                col_gen.finalize_col();

                let (interaction_trace, claimed_sum) = logup_gen.finalize_last();
                tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(interaction_trace));
                quantize_claimed_sums.push(claimed_sum);
            }
        }

        tree_builder.commit(channel);
    } // end if has_logup

    // Build all components with shared allocator
    let mut allocator = TraceLocationAllocator::default();
    let mut component_refs_storage: Vec<Box<dyn ComponentProverErased<B>>> = Vec::new();
    let mut activation_claims: Vec<LayerClaim> = Vec::new();
    let mut add_claims: Vec<LayerClaim> = Vec::new();
    let mut mul_claims: Vec<LayerClaim> = Vec::new();
    let mut layernorm_claims: Vec<LayerClaim> = Vec::new();
    let mut embedding_claims: Vec<LayerClaim> = Vec::new();
    let mut quantize_claims_vec: Vec<LayerClaim> = Vec::new();

    // Activation components
    if let Some(ref lookup) = activation_lookup {
        for (idx, layer) in activation_layers.iter().enumerate() {
            let claimed_sum = activation_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                ActivationEval {
                    log_n_rows: layer.log_size,
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                    total_sum: claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            activation_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Add components
    for layer in &add_layers {
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseAddEval { log_n_rows: layer.log_size },
            SecureField::default(),
        );
        component_refs_storage.push(Box::new(component));
        add_claims.push(LayerClaim {
            layer_index: layer.node_id,
            claimed_sum: SecureField::default(),
            trace_rows: 1 << layer.log_size,
        });
    }

    // Mul components
    for layer in &mul_layers {
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseMulEval { log_n_rows: layer.log_size },
            SecureField::default(),
        );
        component_refs_storage.push(Box::new(component));
        mul_claims.push(LayerClaim {
            layer_index: layer.node_id,
            claimed_sum: SecureField::default(),
            trace_rows: 1 << layer.log_size,
        });
    }

    // LayerNorm components
    if let Some(ref lookup) = layernorm_lookup {
        for (idx, layer) in layernorm_layers.iter().enumerate() {
            let claimed_sum = layernorm_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                LayerNormEval {
                    log_n_rows: layer.log_size,
                    dim: layer.inputs.len(),
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            layernorm_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Embedding components (LogUp)
    if let Some(ref lookup) = embedding_lookup_rel {
        for (idx, layer) in embedding_layers.iter().enumerate() {
            let claimed_sum = embedding_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                EmbeddingEval {
                    log_n_rows: layer.log_size,
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            embedding_claims.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    // Quantize components (LogUp range-check)
    if let Some(ref lookup) = quantize_lookup {
        for (idx, layer) in quantize_layers.iter().enumerate() {
            let claimed_sum = quantize_claimed_sums[idx];
            let component = FrameworkComponent::new(
                &mut allocator,
                QuantizeEval {
                    log_n_rows: layer.log_size,
                    lookup_elements: lookup.clone(),
                    claimed_sum,
                },
                claimed_sum,
            );
            component_refs_storage.push(Box::new(component));
            quantize_claims_vec.push(LayerClaim {
                layer_index: layer.node_id,
                claimed_sum,
                trace_rows: 1 << layer.log_size,
            });
        }
    }

    let component_refs: Vec<&dyn ComponentProver<B>> = component_refs_storage
        .iter()
        .map(|c| c.as_component_prover())
        .collect();

    let stark_proof = prove::<B, Blake2sMerkleChannel>(
        &component_refs,
        channel,
        commitment_scheme,
    ).map_err(|e| AggregationError::ProvingError(format!("{e:?}")))?;

    eprintln!(
        "Unified STARK proof complete in {:.2}s (total pipeline: {:.1}s)",
        t_stark.elapsed().as_secs_f64(),
        t_start.elapsed().as_secs_f64(),
    );

    Ok(AggregatedModelProofOnChain {
        unified_stark: Some(stark_proof),
        matmul_proofs,
        batched_matmul_proofs,
        add_claims,
        mul_claims,
        layernorm_claims,
        execution,
        activation_claims,
        attention_proofs,
        embedding_claims,
        quantize_claims: quantize_claims_vec,
        layer_chain_commitment,
        io_commitment,
        layernorm_mean_var_commitments,
    })
}

/// Prove with auto GPU dispatch for on-chain format.
///
/// Uses `GpuBackend` when CUDA is available, otherwise `SimdBackend`.
/// GPU accelerates the unified STARK (Merkle hashing, FRI, quotient eval).
/// Matmul sumcheck proofs use Poseidon and are unaffected by the backend choice.
pub fn prove_model_aggregated_onchain_auto(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, AggregationError> {
    let gpu_available = crate::backend::gpu_is_available();
    info!(
        gpu_available,
        "Auto-selecting backend for on-chain aggregation"
    );
    crate::backend::with_best_backend(
        || {
            info!("Using SimdBackend for on-chain aggregation");
            prove_model_aggregated_onchain_with::<SimdBackend>(graph, input, weights)
        },
        || {
            info!("Using GpuBackend for on-chain aggregation");
            prove_model_aggregated_onchain_gpu(graph, input, weights)
        },
    )
}

/// GPU proving path for on-chain aggregation.
fn prove_model_aggregated_onchain_gpu(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<AggregatedModelProofOnChain, AggregationError> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return prove_model_aggregated_onchain_with::<GpuBackend>(
            graph, input, weights,
        );
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        prove_model_aggregated_onchain_with::<SimdBackend>(graph, input, weights)
    }
}

/// Prove with auto GPU dispatch for on-chain format, with weight commitment cache.
///
/// Same as [`prove_model_aggregated_onchain_auto`] but reuses cached weight
/// commitments from previous inferences. For repeated inference with the same
/// model, this skips `restrict_cols` + `commit_mle_root_only` for all weight
/// matrices — saving 30-50% of per-inference proving time.
///
/// The cache is populated on first inference (miss) and reused on subsequent
/// calls (hit). Callers should persist the cache to disk between sessions via
/// [`WeightCommitmentCache::save`].
pub fn prove_model_aggregated_onchain_auto_cached(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    weight_cache: &crate::weight_cache::SharedWeightCache,
) -> Result<AggregatedModelProofOnChain, AggregationError> {
    let _cache = weight_cache; // used in cuda-runtime path only

    #[cfg(feature = "cuda-runtime")]
    {
        let gpu_available = crate::backend::gpu_is_available();
        if gpu_available {
            return prove_model_aggregated_onchain_gpu_cached(graph, input, weights, _cache);
        }
    }

    // Non-GPU path: no batch entry prep, cache unused
    prove_model_aggregated_onchain_with::<SimdBackend>(graph, input, weights)
}

/// GPU proving path with weight cache.
#[cfg(feature = "cuda-runtime")]
fn prove_model_aggregated_onchain_gpu_cached(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    weight_cache: &crate::weight_cache::SharedWeightCache,
) -> Result<AggregatedModelProofOnChain, AggregationError> {
    use stwo::prover::backend::gpu::GpuBackend;
    prove_model_aggregated_onchain_with_cache::<GpuBackend>(
        graph, input, weights, Some(weight_cache),
    )
}

// --- Type-erased ComponentProver trait for heterogeneous component storage ---

/// Trait object wrapper to hold different `FrameworkComponent<E>` types
/// in a single Vec for the unified prove() call.
trait ComponentProverErased<B: stwo::prover::backend::Backend> {
    fn as_component_prover(&self) -> &dyn ComponentProver<B>;
}

impl<B, E> ComponentProverErased<B> for FrameworkComponent<E>
where
    B: stwo::prover::backend::Backend,
    E: stwo_constraint_framework::FrameworkEval,
    FrameworkComponent<E>: ComponentProver<B>,
{
    fn as_component_prover(&self) -> &dyn ComponentProver<B> {
        self
    }
}

// --- Helper functions ---

fn build_table_columns(
    table: &PrecomputedTable,
    size: usize,
) -> (Col<SimdBackend, BaseField>, Col<SimdBackend, BaseField>) {
    let mut input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut output_col = Col::<SimdBackend, BaseField>::zeros(size);
    for (i, (&inp, &out)) in table.inputs.iter().zip(table.outputs.iter()).enumerate().take(size) {
        input_col.set(i, inp);
        output_col.set(i, out);
    }
    (input_col, output_col)
}

fn build_trace_columns(
    inputs: &[M31],
    outputs: &[M31],
    multiplicities: &[M31],
    pad_input: M31,
    pad_output: M31,
    size: usize,
) -> (
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
) {
    let mut trace_in = Col::<SimdBackend, BaseField>::zeros(size);
    let mut trace_out = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);

    for (i, (&inp, &out)) in inputs.iter().zip(outputs.iter()).enumerate().take(size) {
        trace_in.set(i, inp);
        trace_out.set(i, out);
    }
    for i in inputs.len()..size {
        trace_in.set(i, pad_input);
        trace_out.set(i, pad_output);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    (trace_in, trace_out, mult_col)
}

/// Build 3 trace columns for elementwise Add/Mul (lhs, rhs, output).
fn build_elementwise_trace_columns(
    lhs: &[M31],
    rhs: &[M31],
    output: &[M31],
    size: usize,
) -> (
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
    Col<SimdBackend, BaseField>,
) {
    let mut lhs_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rhs_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut out_col = Col::<SimdBackend, BaseField>::zeros(size);

    let n = lhs.len().min(rhs.len()).min(output.len()).min(size);
    for i in 0..n {
        lhs_col.set(i, lhs[i]);
        rhs_col.set(i, rhs[i]);
        out_col.set(i, output[i]);
    }
    // Pad remaining rows with zeros (identity for Add/Mul constraints)

    (lhs_col, rhs_col, out_col)
}

/// Build 6 trace columns for LayerNorm (input, mean, var, rsqrt, output, multiplicity).
fn build_layernorm_trace_columns(
    inputs: &[M31],
    means: &[M31],
    variances: &[M31],
    rsqrt_vals: &[M31],
    outputs: &[M31],
    multiplicities: &[M31],
    rsqrt_table: &PrecomputedTable,
    size: usize,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, stwo::prover::poly::BitReversedOrder>> {
    let domain = CanonicCoset::new(size.ilog2()).circle_domain();

    let pad_var = rsqrt_table.inputs.first().copied().unwrap_or(M31::from(0));
    let pad_rsqrt = rsqrt_table.outputs.first().copied().unwrap_or(M31::from(0));

    let mut input_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mean_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut var_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut rsqrt_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut output_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);

    let n = inputs.len().min(size);
    for i in 0..n {
        input_col.set(i, inputs[i]);
        mean_col.set(i, means[i]);
        var_col.set(i, variances[i]);
        rsqrt_col.set(i, rsqrt_vals[i]);
        output_col.set(i, outputs[i]);
    }
    for i in n..size {
        var_col.set(i, pad_var);
        rsqrt_col.set(i, pad_rsqrt);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    vec![
        CircleEvaluation::new(domain, input_col),
        CircleEvaluation::new(domain, mean_col),
        CircleEvaluation::new(domain, var_col),
        CircleEvaluation::new(domain, rsqrt_col),
        CircleEvaluation::new(domain, output_col),
        CircleEvaluation::new(domain, mult_col),
    ]
}

/// Build 3 preprocessed columns for embedding LogUp (table_token, table_col, table_value).
fn build_embedding_preprocessed_columns(
    table_tokens: &[M31],
    table_cols: &[M31],
    table_values: &[M31],
    size: usize,
    domain: stwo::core::poly::circle::CircleDomain,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, stwo::prover::poly::BitReversedOrder>> {
    let mut tok_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut col_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut val_col = Col::<SimdBackend, BaseField>::zeros(size);

    let n = table_tokens.len().min(size);
    for i in 0..n {
        tok_col.set(i, table_tokens[i]);
        col_col.set(i, table_cols[i]);
        val_col.set(i, table_values[i]);
    }

    vec![
        CircleEvaluation::new(domain, tok_col),
        CircleEvaluation::new(domain, col_col),
        CircleEvaluation::new(domain, val_col),
    ]
}

/// Build 4 execution trace columns for embedding (token_id, col_idx, value, multiplicity).
fn build_embedding_trace_columns(
    token_ids: &[M31],
    col_indices: &[M31],
    values: &[M31],
    multiplicities: &[M31],
    size: usize,
    domain: stwo::core::poly::circle::CircleDomain,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, stwo::prover::poly::BitReversedOrder>> {
    let mut tok_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut col_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut val_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);

    let n = token_ids.len().min(size);
    for i in 0..n {
        tok_col.set(i, token_ids[i]);
        col_col.set(i, col_indices[i]);
        val_col.set(i, values[i]);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    vec![
        CircleEvaluation::new(domain, tok_col),
        CircleEvaluation::new(domain, col_col),
        CircleEvaluation::new(domain, val_col),
        CircleEvaluation::new(domain, mult_col),
    ]
}

/// Build 1 preprocessed column for quantize range table [0, 1, ..., 2^bits - 1], padded to `size`.
fn build_quantize_range_table(bits: u32, size: usize) -> Col<SimdBackend, BaseField> {
    let table_size = (1u32 << bits) as usize;
    let mut col = Col::<SimdBackend, BaseField>::zeros(size);
    for i in 0..table_size.min(size) {
        col.set(i, M31::from(i as u32));
    }
    // Remaining rows are zero (valid padding for range table)
    col
}

/// Build 2 execution trace columns for quantize (value, multiplicity).
fn build_quantize_trace_columns(
    values: &[M31],
    multiplicities: &[M31],
    bits: u32,
    size: usize,
    domain: stwo::core::poly::circle::CircleDomain,
) -> Vec<CircleEvaluation<SimdBackend, BaseField, stwo::prover::poly::BitReversedOrder>> {
    let (value_col, mult_col) = build_quantize_trace_simd(values, multiplicities, bits, size);
    vec![
        CircleEvaluation::new(domain, value_col),
        CircleEvaluation::new(domain, mult_col),
    ]
}

/// Build SIMD columns for quantize trace data (value, multiplicity).
fn build_quantize_trace_simd(
    values: &[M31],
    multiplicities: &[M31],
    _bits: u32,
    size: usize,
) -> (Col<SimdBackend, BaseField>, Col<SimdBackend, BaseField>) {
    let mut value_col = Col::<SimdBackend, BaseField>::zeros(size);
    let mut mult_col = Col::<SimdBackend, BaseField>::zeros(size);

    for (i, &v) in values.iter().enumerate().take(size) {
        value_col.set(i, v);
    }
    for (i, &m) in multiplicities.iter().enumerate().take(size) {
        mult_col.set(i, m);
    }

    (value_col, mult_col)
}

fn pack_multiplicities(
    multiplicities: &[M31],
    vec_row: usize,
) -> PackedSecureField {
    use stwo::prover::backend::simd::m31::PackedBaseField;

    let n_lanes = 16usize;
    let base = vec_row * n_lanes;
    let mut vals = [M31::from(0); 16];
    for (i, val) in vals.iter_mut().enumerate() {
        let idx = base + i;
        if idx < multiplicities.len() {
            *val = multiplicities[idx];
        }
    }
    let packed_base = PackedBaseField::from_array(std::array::from_fn(|i| vals[i]));
    packed_base.into()
}

/// Collect all matmul proofs and activation claims from chunked proving results.
///
/// Useful for composing chunk-level proofs into a single logical proof payload
/// for recursive STARK verification.
pub fn collect_chunk_proofs(
    results: &[crate::compiler::chunked::ChunkProofResult],
) -> (Vec<(usize, MatMulSumcheckProofOnChain)>, Vec<LayerClaim>) {
    let mut all_matmul: Vec<(usize, MatMulSumcheckProofOnChain)> = Vec::new();
    let mut all_claims: Vec<LayerClaim> = Vec::new();

    for chunk in results {
        for (layer_idx, proof) in &chunk.proof.matmul_proofs {
            // Remap to original graph indices
            let original_idx = chunk.node_range.start + layer_idx;
            all_matmul.push((original_idx, proof.clone()));
        }
        for claim in &chunk.proof.activation_claims {
            let mut remapped = claim.clone();
            remapped.layer_index += chunk.node_range.start;
            all_claims.push(remapped);
        }
    }

    (all_matmul, all_claims)
}

/// Aggregate multiple claims into a summary (layer count, total rows).
pub fn summarize_claims(claims: &[LayerClaim]) -> (usize, usize) {
    let total_rows: usize = claims.iter().map(|c| c.trace_rows).sum();
    (claims.len(), total_rows)
}

// === Verification ===

/// Type-erased Component trait for heterogeneous component storage during verification.
trait ComponentRefErased {
    fn as_component(&self) -> &dyn Component;
}

impl<E> ComponentRefErased for FrameworkComponent<E>
where
    E: stwo_constraint_framework::FrameworkEval,
{
    fn as_component(&self) -> &dyn Component {
        self
    }
}

/// Verify an aggregated model proof (matmul sumchecks + unified STARK).
///
/// Re-runs the forward pass to reconstruct expected matrices, then:
/// 1. Verifies each matmul sumcheck proof against recomputed A × B = C
/// 2. Verifies the unified STARK proof (activation, add, mul, layernorm, embedding)
pub fn verify_aggregated_model_proof(
    proof: AggregatedModelProof,
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<(), AggregationError> {
    use crate::components::matmul::verify_matmul_sumcheck;

    // 1. Re-run forward pass to collect (A, B, C) for matmul verification
    let mut current = input.clone();
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();
    let mut matmul_matrices: HashMap<usize, (M31Matrix, M31Matrix, M31Matrix)> = HashMap::new();
    let mut verifier_intermediates: Vec<(usize, M31Matrix)> = Vec::new();
    let mut attention_verify_data: Vec<(usize, MultiHeadAttentionConfig, AttentionWeights, M31Matrix)> = Vec::new();
    let mut ln_verify_data: Vec<(usize, Vec<M31>, Vec<M31>)> = Vec::new(); // (node_id, means, variances)

    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let Some(&first_input) = node.inputs.first() {
            if let Some(inp) = node_outputs.get(&first_input) {
                current = inp.clone();
            }
        }

        // Record the input to this node (matches prover's intermediates.push pattern)
        verifier_intermediates.push((node_id, current.clone()));

        match &node.op {
            GraphOp::MatMul { .. } => {
                let weight = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let output = matmul_m31(&current, weight);
                matmul_matrices.insert(node.id, (current.clone(), weight.clone(), output.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Activation { activation_type, .. } => {
                let f = activation_type.as_fn();
                let output = crate::compiler::prove::apply_activation_pub(&current, &*f);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Add { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_add(&lhs, &rhs);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Mul { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_mul(&lhs, &rhs);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::LayerNorm { dim } => {
                let ln = apply_layernorm_detailed(&current, *dim);
                ln_verify_data.push((node.id, ln.means.clone(), ln.variances.clone()));
                node_outputs.insert(node.id, ln.output_matrix.clone());
                current = ln.output_matrix;
            }
            GraphOp::Attention { config: attn_config } => {
                let w_q = weights.get_named_weight(node.id, "w_q");
                let w_k = weights.get_named_weight(node.id, "w_k");
                let w_v = weights.get_named_weight(node.id, "w_v");
                let w_o = weights.get_named_weight(node.id, "w_o");
                if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (w_q, w_k, w_v, w_o) {
                    let attn_weights = AttentionWeights {
                        w_q: wq.clone(), w_k: wk.clone(),
                        w_v: wv.clone(), w_o: wo.clone(),
                    };
                    attention_verify_data.push((
                        node.id, *attn_config, attn_weights.clone(), current.clone(),
                    ));
                    let inter = attention_forward(&current, &attn_weights, attn_config, false);
                    node_outputs.insert(node.id, inter.final_output.clone());
                    current = inter.final_output;
                } else {
                    node_outputs.insert(node.id, current.clone());
                }
            }
            GraphOp::Embedding { .. } => {
                let embed_table = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let token_u32s: Vec<u32> = current.data.iter().map(|m| m.0).collect();
                let (output, _, _, _, _) = embedding_lookup(&token_u32s, embed_table);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Conv2D { in_channels, out_channels, kernel_size, stride, padding } => {
                let kernel = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let im2col_config = Im2ColConfig {
                    in_channels: *in_channels,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    input_h: current.rows,
                    input_w: current.cols / in_channels,
                };
                let (im2col_mat, kernel_mat, output) =
                    conv2d_forward(&current.data, &kernel.data, &im2col_config, *out_channels);
                matmul_matrices.insert(node.id, (im2col_mat, kernel_mat, output.clone()));
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            _ => {
                node_outputs.insert(node.id, current.clone());
            }
        }
    }

    // 1b. Verify layer chain commitment
    let expected_chain = compute_layer_chain_commitment(
        input, &verifier_intermediates, &current,
    );
    if expected_chain != proof.layer_chain_commitment {
        return Err(AggregationError::VerificationFailed(
            format!(
                "Layer chain commitment mismatch: expected {:#x}, got {:#x}",
                expected_chain, proof.layer_chain_commitment,
            )
        ));
    }

    // 1c. Verify IO commitment (binds proof to specific input/output)
    let expected_io = compute_io_commitment(input, &current);
    if expected_io != proof.io_commitment {
        return Err(AggregationError::VerificationFailed(
            format!(
                "IO commitment mismatch: expected {:#x}, got {:#x}",
                expected_io, proof.io_commitment,
            )
        ));
    }

    // 1d. Verify LayerNorm mean/variance commitments
    for (idx, (node_id, means, variances)) in ln_verify_data.iter().enumerate() {
        if idx < proof.layernorm_mean_var_commitments.len() {
            let expected = compute_layernorm_mean_var_commitment(means, variances);
            if expected != proof.layernorm_mean_var_commitments[idx] {
                return Err(AggregationError::VerificationFailed(
                    format!("LayerNorm mean/var commitment mismatch at node {node_id}")
                ));
            }
        }
    }

    // 2. Verify matmul sumcheck proofs
    for (node_id, matmul_proof) in &proof.matmul_proofs {
        let (a, b, c) = matmul_matrices.get(node_id).ok_or_else(|| {
            AggregationError::VerificationFailed(
                format!("No matmul data for node {node_id}")
            )
        })?;
        verify_matmul_sumcheck(matmul_proof, a, b, c).map_err(|e| {
            AggregationError::VerificationFailed(
                format!("MatMul sumcheck node {node_id}: {e}")
            )
        })?;
    }

    // 2b. Verify attention proofs (matmul sub-proofs + softmax STARK)
    let attention_proof_count = proof.attention_proofs.len();
    for (node_id, attn_proof) in proof.attention_proofs {
        let (_, config, attn_weights, attn_input) = attention_verify_data.iter()
            .find(|(id, _, _, _)| *id == node_id)
            .ok_or_else(|| AggregationError::VerificationFailed(
                format!("No attention verify data for node {node_id}")
            ))?;
        verify_attention_proof_blake2s(
            attn_proof, attn_input, attn_weights, config,
        )?;
    }

    // 3. Verify unified STARK
    if let Some(stark_proof) = proof.unified_stark {
        verify_unified_stark_blake2s(
            stark_proof, graph,
            &proof.activation_claims,
            &proof.add_claims,
            &proof.mul_claims,
            &proof.layernorm_claims,
            &proof.embedding_claims,
            &proof.quantize_claims,
        )?;
    }

    // 4. Verify model completeness: every provable layer has a proof/claim
    verify_model_completeness(
        graph,
        proof.matmul_proofs.len(),
        proof.activation_claims.len(),
        proof.add_claims.len(),
        proof.mul_claims.len(),
        proof.layernorm_claims.len(),
        attention_proof_count,
        proof.embedding_claims.len(),
        proof.quantize_claims.len(),
    )?;

    Ok(())
}

/// Verify an on-chain aggregated model proof (Poseidon matmul sumchecks + Blake2s unified STARK).
///
/// Mirror of `verify_aggregated_model_proof` for the on-chain proof format:
/// 1. Re-runs the forward pass to verify IO and layer chain commitments.
/// 2. Verifies individual matmul sumcheck proofs via `verify_matmul_sumcheck_onchain`
///    (self-contained: Fiat-Shamir transcript + MLE opening proofs).
/// 3. Verifies batched matmul proofs (combined sumcheck + per-entry final evaluation).
/// 4. Verifies attention on-chain proofs (matmul sub-proofs).
/// 5. Verifies the unified STARK proof (activation, add, mul, layernorm, embedding).
/// 6. Checks model completeness.
pub fn verify_aggregated_model_proof_onchain(
    proof: AggregatedModelProofOnChain,
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<(), AggregationError> {
    use crate::components::matmul::verify_matmul_sumcheck_onchain;

    // 1. Re-run forward pass to verify commitments and collect attention data
    let mut current = input.clone();
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();
    let mut verifier_intermediates: Vec<(usize, M31Matrix)> = Vec::new();
    let mut attention_verify_data: Vec<(usize, MultiHeadAttentionConfig, AttentionWeights, M31Matrix)> = Vec::new();
    let mut ln_verify_data: Vec<(usize, Vec<M31>, Vec<M31>)> = Vec::new(); // (node_id, means, variances)

    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];
        if let Some(&first_input) = node.inputs.first() {
            if let Some(inp) = node_outputs.get(&first_input) {
                current = inp.clone();
            }
        }

        verifier_intermediates.push((node_id, current.clone()));

        match &node.op {
            GraphOp::MatMul { .. } => {
                let weight = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let output = matmul_m31(&current, weight);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Activation { activation_type, .. } => {
                let f = activation_type.as_fn();
                let output = crate::compiler::prove::apply_activation_pub(&current, &*f);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Add { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_add(&lhs, &rhs);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Mul { .. } => {
                let lhs = node.inputs.get(0)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let rhs = node.inputs.get(1)
                    .and_then(|id| node_outputs.get(id))
                    .cloned()
                    .unwrap_or_else(|| current.clone());
                let output = elementwise_mul(&lhs, &rhs);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::LayerNorm { dim } => {
                let ln = apply_layernorm_detailed(&current, *dim);
                ln_verify_data.push((node.id, ln.means.clone(), ln.variances.clone()));
                node_outputs.insert(node.id, ln.output_matrix.clone());
                current = ln.output_matrix;
            }
            GraphOp::Attention { config: attn_config } => {
                let w_q = weights.get_named_weight(node.id, "w_q");
                let w_k = weights.get_named_weight(node.id, "w_k");
                let w_v = weights.get_named_weight(node.id, "w_v");
                let w_o = weights.get_named_weight(node.id, "w_o");
                if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (w_q, w_k, w_v, w_o) {
                    let attn_weights = AttentionWeights {
                        w_q: wq.clone(), w_k: wk.clone(),
                        w_v: wv.clone(), w_o: wo.clone(),
                    };
                    attention_verify_data.push((
                        node.id, *attn_config, attn_weights.clone(), current.clone(),
                    ));
                    let inter = attention_forward(&current, &attn_weights, attn_config, false);
                    node_outputs.insert(node.id, inter.final_output.clone());
                    current = inter.final_output;
                } else {
                    node_outputs.insert(node.id, current.clone());
                }
            }
            GraphOp::Embedding { .. } => {
                let embed_table = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let token_u32s: Vec<u32> = current.data.iter().map(|m| m.0).collect();
                let (output, _, _, _, _) = embedding_lookup(&token_u32s, embed_table);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            GraphOp::Conv2D { in_channels, out_channels, kernel_size, stride, padding } => {
                let kernel = weights.get_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id)
                )?;
                let im2col_config = Im2ColConfig {
                    in_channels: *in_channels,
                    kernel_size: *kernel_size,
                    stride: *stride,
                    padding: *padding,
                    input_h: current.rows,
                    input_w: current.cols / in_channels,
                };
                let (_im2col_mat, _kernel_mat, output) =
                    conv2d_forward(&current.data, &kernel.data, &im2col_config, *out_channels);
                node_outputs.insert(node.id, output.clone());
                current = output;
            }
            _ => {
                node_outputs.insert(node.id, current.clone());
            }
        }
    }

    // 1b. Verify layer chain commitment
    let expected_chain = compute_layer_chain_commitment(
        input, &verifier_intermediates, &current,
    );
    if expected_chain != proof.layer_chain_commitment {
        return Err(AggregationError::VerificationFailed(
            format!(
                "Layer chain commitment mismatch: expected {:#x}, got {:#x}",
                expected_chain, proof.layer_chain_commitment,
            )
        ));
    }

    // 1c. Verify IO commitment
    let expected_io = compute_io_commitment(input, &current);
    if expected_io != proof.io_commitment {
        return Err(AggregationError::VerificationFailed(
            format!(
                "IO commitment mismatch: expected {:#x}, got {:#x}",
                expected_io, proof.io_commitment,
            )
        ));
    }

    // 1d. Verify LayerNorm mean/variance commitments
    for (idx, (node_id, means, variances)) in ln_verify_data.iter().enumerate() {
        if idx < proof.layernorm_mean_var_commitments.len() {
            let expected = compute_layernorm_mean_var_commitment(means, variances);
            if expected != proof.layernorm_mean_var_commitments[idx] {
                return Err(AggregationError::VerificationFailed(
                    format!("LayerNorm mean/var commitment mismatch at node {node_id}")
                ));
            }
        }
    }

    // 2. Verify individual matmul sumcheck proofs (on-chain format, self-contained)
    for (node_id, matmul_proof) in &proof.matmul_proofs {
        verify_matmul_sumcheck_onchain(matmul_proof).map_err(|e| {
            AggregationError::VerificationFailed(
                format!("On-chain matmul sumcheck node {node_id}: {e}")
            )
        })?;
    }

    // 2a. Verify batched matmul proofs
    for (batch_idx, batch) in proof.batched_matmul_proofs.iter().enumerate() {
        verify_batched_matmul_onchain(batch).map_err(|e| {
            AggregationError::VerificationFailed(
                format!("Batched matmul batch {batch_idx}: {e}")
            )
        })?;
    }

    // 2b. Verify attention on-chain proofs (matmul sub-proofs + softmax STARK)
    let attention_proof_count = proof.attention_proofs.len();
    for (node_id, attn_proof) in proof.attention_proofs {
        let (_, config, _, _) = attention_verify_data.iter()
            .find(|(id, _, _, _)| *id == node_id)
            .ok_or_else(|| AggregationError::VerificationFailed(
                format!("No attention verify data for node {node_id}")
            ))?;
        verify_attention_proof_onchain(attn_proof, config)?;
    }

    // 3. Verify unified STARK (same Blake2s format for both paths)
    if let Some(stark_proof) = proof.unified_stark {
        verify_unified_stark_blake2s(
            stark_proof, graph,
            &proof.activation_claims,
            &proof.add_claims,
            &proof.mul_claims,
            &proof.layernorm_claims,
            &proof.embedding_claims,
            &proof.quantize_claims,
        )?;
    }

    // 4. Verify model completeness
    let total_matmul = proof.matmul_proofs.len()
        + proof.batched_matmul_proofs.iter().map(|b| b.entries.len()).sum::<usize>();
    verify_model_completeness(
        graph,
        total_matmul,
        proof.activation_claims.len(),
        proof.add_claims.len(),
        proof.mul_claims.len(),
        proof.layernorm_claims.len(),
        attention_proof_count,
        proof.embedding_claims.len(),
        proof.quantize_claims.len(),
    )?;

    Ok(())
}

/// Verify a batched on-chain matmul proof with full Fiat-Shamir replay.
///
/// Batched proofs combine multiple matmuls with the same k dimension into
/// a single sumcheck. The verifier replays the exact Fiat-Shamir transcript
/// that the prover used (see gpu_sumcheck::prove_matmul_batch_onchain_gpu):
///
/// 1. Initialize Poseidon channel with batch metadata (num_entries, k).
/// 2. Mix all per-matmul commitments and claimed sums.
/// 3. Re-derive lambda from the channel (must match prover's lambda).
/// 4. Verify combined_claimed_sum = Σ λ^i * claimed_sum_i.
/// 5. Per round: verify p(0)+p(1) = current_sum, mix poly, draw challenge r_k,
///    update current_sum = p(r_k).
/// 6. Final check: current_sum = Σ λ^i * final_a_eval_i * final_b_eval_i.
fn verify_batched_matmul_onchain(
    batch: &BatchedMatMulProofOnChain,
) -> Result<(), AggregationError> {
    use crate::crypto::poseidon_channel::{PoseidonChannel, securefield_to_felt};

    let log_k = batch.num_rounds as usize;
    if batch.round_polys.len() != log_k {
        return Err(AggregationError::VerificationFailed(
            format!("Batched round_polys count {} != num_rounds {}", batch.round_polys.len(), log_k)
        ));
    }
    if batch.entries.is_empty() {
        return Err(AggregationError::VerificationFailed(
            "Batched proof has no entries".into()
        ));
    }

    // Replay Fiat-Shamir: initialize channel with batch metadata
    // (mirrors prove_matmul_batch_onchain_gpu steps 1-2)
    let mut channel = PoseidonChannel::new();
    channel.mix_u64(batch.entries.len() as u64);
    channel.mix_u64(batch.k as u64);

    for entry in &batch.entries {
        channel.mix_felt(securefield_to_felt(entry.claimed_sum));
        channel.mix_felt(entry.a_commitment);
        channel.mix_felt(entry.b_commitment);
    }

    // Re-derive lambda from channel (must match prover's lambda)
    let lambda = channel.draw_qm31();
    if lambda != batch.lambda {
        return Err(AggregationError::VerificationFailed(
            format!(
                "Batched lambda mismatch: re-derived {:?}, proof contains {:?}",
                lambda, batch.lambda
            )
        ));
    }

    // Verify combined_claimed_sum = Σ λ^i * claimed_sum_i
    let mut expected_combined = SecureField::default();
    let mut lambda_pow = SecureField::one();
    for entry in &batch.entries {
        expected_combined += lambda_pow * entry.claimed_sum;
        lambda_pow = lambda_pow * lambda;
    }
    if expected_combined != batch.combined_claimed_sum {
        return Err(AggregationError::VerificationFailed(
            format!(
                "Batched combined_claimed_sum mismatch: expected {:?}, got {:?}",
                expected_combined, batch.combined_claimed_sum
            )
        ));
    }

    // Verify sumcheck rounds with Fiat-Shamir challenge derivation
    let mut current_sum = batch.combined_claimed_sum;
    for rp in &batch.round_polys {
        let p_at_0 = rp.c0;
        let p_at_1 = rp.c0 + rp.c1 + rp.c2;
        let round_sum = p_at_0 + p_at_1;

        if round_sum != current_sum {
            return Err(AggregationError::VerificationFailed(
                format!("Batched round sum {:?} != expected {:?}", round_sum, current_sum)
            ));
        }

        // Mix round polynomial into channel and draw per-round challenge
        // (mirrors prover's channel.mix_poly_coeffs + channel.draw_qm31)
        channel.mix_poly_coeffs(rp.c0, rp.c1, rp.c2);
        let r_k = channel.draw_qm31();

        // Evaluate p(r_k) = c0 + c1*r_k + c2*r_k^2
        current_sum = rp.c0 + rp.c1 * r_k + rp.c2 * r_k * r_k;
    }

    // Verify final evaluation: current_sum = Σ λ^i * a_i * b_i
    let mut expected_final = SecureField::default();
    lambda_pow = SecureField::one();
    for entry in &batch.entries {
        expected_final += lambda_pow * entry.final_a_eval * entry.final_b_eval;
        lambda_pow = lambda_pow * lambda;
    }
    if current_sum != expected_final {
        return Err(AggregationError::VerificationFailed(
            format!(
                "Batched final eval mismatch: current_sum {:?} != Σ λ^i·a_i·b_i {:?}",
                current_sum, expected_final
            )
        ));
    }

    Ok(())
}

/// Verify an on-chain attention proof.
///
/// Verifies all matmul sub-proofs (Q/K/V projections, per-head scores,
/// per-head attn×V, output projection) using the on-chain Poseidon verifier,
/// plus the softmax exp STARK proof (Blake2s channel).
fn verify_attention_proof_onchain(
    proof: AttentionProofOnChain,
    config: &MultiHeadAttentionConfig,
) -> Result<(), AggregationError> {
    use crate::components::matmul::verify_matmul_sumcheck_onchain;

    // Q projection
    verify_matmul_sumcheck_onchain(&proof.q_proof)
        .map_err(|e| AggregationError::VerificationFailed(
            format!("Attention on-chain Q projection: {e}")
        ))?;

    // K projection
    verify_matmul_sumcheck_onchain(&proof.k_proof)
        .map_err(|e| AggregationError::VerificationFailed(
            format!("Attention on-chain K projection: {e}")
        ))?;

    // V projection
    verify_matmul_sumcheck_onchain(&proof.v_proof)
        .map_err(|e| AggregationError::VerificationFailed(
            format!("Attention on-chain V projection: {e}")
        ))?;

    // Per-head score proofs
    if proof.score_proofs.len() != config.num_heads {
        return Err(AggregationError::VerificationFailed(
            format!("Expected {} on-chain score proofs, got {}", config.num_heads, proof.score_proofs.len())
        ));
    }
    for (h, sp) in proof.score_proofs.iter().enumerate() {
        verify_matmul_sumcheck_onchain(sp)
            .map_err(|e| AggregationError::VerificationFailed(
                format!("Attention on-chain score head {h}: {e}")
            ))?;
    }

    // Per-head attn×V proofs
    if proof.attn_v_proofs.len() != config.num_heads {
        return Err(AggregationError::VerificationFailed(
            format!("Expected {} on-chain attn_v proofs, got {}", config.num_heads, proof.attn_v_proofs.len())
        ));
    }
    for (h, avp) in proof.attn_v_proofs.iter().enumerate() {
        verify_matmul_sumcheck_onchain(avp)
            .map_err(|e| AggregationError::VerificationFailed(
                format!("Attention on-chain attn_v head {h}: {e}")
            ))?;
    }

    // Output projection
    verify_matmul_sumcheck_onchain(&proof.output_proof)
        .map_err(|e| AggregationError::VerificationFailed(
            format!("Attention on-chain output projection: {e}")
        ))?;

    // Softmax exp STARK
    verify_attention_softmax_stark(
        proof.softmax_exp_proof,
        proof.softmax_claimed_sum,
        proof.softmax_log_size,
    )?;

    Ok(())
}

/// Verify model completeness: every provable layer in the graph has a corresponding proof/claim.
///
/// Counts the expected number of each operation type from the graph, then verifies
/// the proof contains exactly that many proofs/claims. Conv2D counts as a matmul.
/// Identity and Quantize are passthrough (not proven).
fn verify_model_completeness(
    graph: &ComputationGraph,
    num_matmul_proofs: usize,
    num_activation_claims: usize,
    num_add_claims: usize,
    num_mul_claims: usize,
    num_layernorm_claims: usize,
    num_attention_proofs: usize,
    num_embedding_claims: usize,
    num_quantize_claims: usize,
) -> Result<(), AggregationError> {
    let mut expected_matmul = 0usize;
    let mut expected_activation = 0usize;
    let mut expected_add = 0usize;
    let mut expected_mul = 0usize;
    let mut expected_layernorm = 0usize;
    let mut expected_attention = 0usize;
    let mut expected_embedding = 0usize;
    let mut expected_quantize = 0usize;

    for node in &graph.nodes {
        match &node.op {
            GraphOp::MatMul { .. } | GraphOp::Conv2D { .. } => expected_matmul += 1,
            GraphOp::Activation { .. } => expected_activation += 1,
            GraphOp::Add { .. } => expected_add += 1,
            GraphOp::Mul { .. } => expected_mul += 1,
            GraphOp::LayerNorm { .. } => expected_layernorm += 1,
            GraphOp::Attention { .. } => expected_attention += 1,
            GraphOp::Embedding { .. } => expected_embedding += 1,
            GraphOp::Quantize { .. } => expected_quantize += 1,
            // Identity — passthrough, not proven
            _ => {}
        }
    }

    let mut mismatches = Vec::new();

    if num_matmul_proofs != expected_matmul {
        mismatches.push(format!("matmul: expected {expected_matmul}, got {num_matmul_proofs}"));
    }
    if num_activation_claims != expected_activation {
        mismatches.push(format!("activation: expected {expected_activation}, got {num_activation_claims}"));
    }
    if num_add_claims != expected_add {
        mismatches.push(format!("add: expected {expected_add}, got {num_add_claims}"));
    }
    if num_mul_claims != expected_mul {
        mismatches.push(format!("mul: expected {expected_mul}, got {num_mul_claims}"));
    }
    if num_layernorm_claims != expected_layernorm {
        mismatches.push(format!("layernorm: expected {expected_layernorm}, got {num_layernorm_claims}"));
    }
    if num_attention_proofs != expected_attention {
        mismatches.push(format!("attention: expected {expected_attention}, got {num_attention_proofs}"));
    }
    if num_embedding_claims != expected_embedding {
        mismatches.push(format!("embedding: expected {expected_embedding}, got {num_embedding_claims}"));
    }
    if num_quantize_claims != expected_quantize {
        mismatches.push(format!("quantize: expected {expected_quantize}, got {num_quantize_claims}"));
    }

    if !mismatches.is_empty() {
        return Err(AggregationError::VerificationFailed(
            format!("Model completeness check failed: {}", mismatches.join(", "))
        ));
    }

    Ok(())
}

/// Verify the unified STARK proof covering all non-matmul components.
///
/// Reconstructs the same component definitions the prover used, sets up the
/// commitment scheme verifier with the proof's tree commitments, and calls
/// STWO's verify function to check the DEEP-ALI + FRI proof.
fn verify_unified_stark_blake2s(
    proof: StarkProof<Blake2sHash>,
    graph: &ComputationGraph,
    activation_claims: &[LayerClaim],
    add_claims: &[LayerClaim],
    mul_claims: &[LayerClaim],
    layernorm_claims: &[LayerClaim],
    embedding_claims: &[LayerClaim],
    quantize_claims: &[LayerClaim],
) -> Result<(), AggregationError> {
    let config = PcsConfig::default();

    // Collect LayerNorm dims from graph nodes
    let mut layernorm_dims: HashMap<usize, usize> = HashMap::new();
    for node in &graph.nodes {
        if let GraphOp::LayerNorm { dim } = &node.op {
            layernorm_dims.insert(node.id, *dim);
        }
    }

    let has_logup = !activation_claims.is_empty()
        || !layernorm_claims.is_empty()
        || !embedding_claims.is_empty()
        || !quantize_claims.is_empty();

    // Step 1: Build dummy components to get per-tree column sizes.
    // Column structure depends only on log_n_rows and evaluator type, NOT on lookup elements.
    let dummy_sizes = build_component_tree_sizes(
        activation_claims, add_claims, mul_claims,
        layernorm_claims, embedding_claims, quantize_claims, &layernorm_dims,
    );

    // Step 2: Set up channel and commitment scheme verifier
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Step 3: Commit Tree 0 (preprocessed) and Tree 1 (execution) from proof
    commitment_scheme.commit(proof.commitments[0], &dummy_sizes.tree0, channel);
    commitment_scheme.commit(proof.commitments[1], &dummy_sizes.tree1, channel);

    // Step 4: Draw relation elements (same order as prover)
    let activation_lookup = if !activation_claims.is_empty() {
        Some(ActivationRelation::draw(channel))
    } else {
        None
    };
    let layernorm_lookup = if !layernorm_claims.is_empty() {
        Some(LayerNormRelation::draw(channel))
    } else {
        None
    };
    let embedding_lookup_rel = if !embedding_claims.is_empty() {
        Some(EmbeddingRelation::draw(channel))
    } else {
        None
    };
    let quantize_lookup = if !quantize_claims.is_empty() {
        Some(QuantizeRelation::draw(channel))
    } else {
        None
    };

    // Step 5: Commit Tree 2 (interaction) if LogUp components exist
    if has_logup {
        commitment_scheme.commit(proof.commitments[2], &dummy_sizes.tree2, channel);
    }

    // Step 6: Build real components with drawn relation elements
    let mut allocator = TraceLocationAllocator::default();
    let mut component_storage: Vec<Box<dyn ComponentRefErased>> = Vec::new();

    // Activation components
    if let Some(ref lookup) = activation_lookup {
        for claim in activation_claims {
            let ls = claim.trace_rows.ilog2();
            let component = FrameworkComponent::new(
                &mut allocator,
                ActivationEval {
                    log_n_rows: ls,
                    lookup_elements: lookup.clone(),
                    claimed_sum: claim.claimed_sum,
                    total_sum: claim.claimed_sum,
                },
                claim.claimed_sum,
            );
            component_storage.push(Box::new(component));
        }
    }

    // Add components (pure AIR)
    for claim in add_claims {
        let ls = claim.trace_rows.ilog2();
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseAddEval { log_n_rows: ls },
            SecureField::default(),
        );
        component_storage.push(Box::new(component));
    }

    // Mul components (pure AIR)
    for claim in mul_claims {
        let ls = claim.trace_rows.ilog2();
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseMulEval { log_n_rows: ls },
            SecureField::default(),
        );
        component_storage.push(Box::new(component));
    }

    // LayerNorm components
    if let Some(ref lookup) = layernorm_lookup {
        for claim in layernorm_claims {
            let ls = claim.trace_rows.ilog2();
            let dim = layernorm_dims.get(&claim.layer_index).copied().unwrap_or(1);
            let component = FrameworkComponent::new(
                &mut allocator,
                LayerNormEval {
                    log_n_rows: ls,
                    dim,
                    lookup_elements: lookup.clone(),
                    claimed_sum: claim.claimed_sum,
                },
                claim.claimed_sum,
            );
            component_storage.push(Box::new(component));
        }
    }

    // Embedding components
    if let Some(ref lookup) = embedding_lookup_rel {
        for claim in embedding_claims {
            let ls = claim.trace_rows.ilog2();
            let component = FrameworkComponent::new(
                &mut allocator,
                EmbeddingEval {
                    log_n_rows: ls,
                    lookup_elements: lookup.clone(),
                    claimed_sum: claim.claimed_sum,
                },
                claim.claimed_sum,
            );
            component_storage.push(Box::new(component));
        }
    }

    // Quantize components (LogUp range-check)
    if let Some(ref lookup) = quantize_lookup {
        for claim in quantize_claims {
            let ls = claim.trace_rows.ilog2();
            let component = FrameworkComponent::new(
                &mut allocator,
                QuantizeEval {
                    log_n_rows: ls,
                    lookup_elements: lookup.clone(),
                    claimed_sum: claim.claimed_sum,
                },
                claim.claimed_sum,
            );
            component_storage.push(Box::new(component));
        }
    }

    // Step 7: Verify
    let component_refs: Vec<&dyn Component> = component_storage
        .iter()
        .map(|c| c.as_component())
        .collect();

    stwo_verify::<Blake2sMerkleChannel>(
        &component_refs,
        channel,
        &mut commitment_scheme,
        proof,
    ).map_err(|e| AggregationError::VerificationFailed(format!("Unified STARK: {e}")))?;

    Ok(())
}

/// Verify a single attention proof: all matmul sub-proofs + softmax exp STARK.
///
/// Re-runs the attention forward pass from the given input and weights, then:
/// 1. Verifies Q/K/V projection sumcheck proofs
/// 2. Verifies per-head score and attn×V sumcheck proofs
/// 3. Verifies the output projection sumcheck proof
/// 4. Verifies the softmax exp STARK proof (LogUp activation)
fn verify_attention_proof_blake2s(
    proof: AttentionProof<Blake2sHash>,
    input: &M31Matrix,
    weights: &AttentionWeights,
    config: &MultiHeadAttentionConfig,
) -> Result<(), AggregationError> {
    use crate::components::matmul::verify_matmul_sumcheck;

    // Re-run forward pass to get correct intermediates
    let inter = attention_forward(input, weights, config, false);

    // Verify forward pass output consistency
    if inter.final_output.data != proof.intermediates.final_output.data {
        return Err(AggregationError::VerificationFailed(
            "Attention forward pass output mismatch".into()
        ));
    }

    // Pad matrices for sumcheck verification (must match prover's padding)
    let input_p = pad_to_pow2(input);
    let wq_p = pad_to_pow2(&weights.w_q);
    let wk_p = pad_to_pow2(&weights.w_k);
    let wv_p = pad_to_pow2(&weights.w_v);
    let wo_p = pad_to_pow2(&weights.w_o);
    let q_p = pad_to_pow2(&inter.q);
    let k_p = pad_to_pow2(&inter.k);
    let v_p = pad_to_pow2(&inter.v);

    // 1. Q projection: input × W_Q = Q
    verify_matmul_sumcheck(&proof.q_proof, &input_p, &wq_p, &q_p)
        .map_err(|e| AggregationError::VerificationFailed(
            format!("Attention Q projection: {e}")
        ))?;

    // 2. K projection: input × W_K = K
    verify_matmul_sumcheck(&proof.k_proof, &input_p, &wk_p, &k_p)
        .map_err(|e| AggregationError::VerificationFailed(
            format!("Attention K projection: {e}")
        ))?;

    // 3. V projection: input × W_V = V
    verify_matmul_sumcheck(&proof.v_proof, &input_p, &wv_p, &v_p)
        .map_err(|e| AggregationError::VerificationFailed(
            format!("Attention V projection: {e}")
        ))?;

    // Split for per-head verification
    let q_heads = split_heads(&inter.q, config.num_heads);
    let k_heads = split_heads(&inter.k, config.num_heads);
    let v_heads = split_heads(&inter.v, config.num_heads);

    if proof.score_proofs.len() != config.num_heads {
        return Err(AggregationError::VerificationFailed(
            format!("Expected {} score proofs, got {}", config.num_heads, proof.score_proofs.len())
        ));
    }
    if proof.attn_v_proofs.len() != config.num_heads {
        return Err(AggregationError::VerificationFailed(
            format!("Expected {} attn_v proofs, got {}", config.num_heads, proof.attn_v_proofs.len())
        ));
    }

    for h in 0..config.num_heads {
        // Per-head: score = Q_h × K_h^T (padded)
        let k_t = transpose_m31(&k_heads[h]);
        let q_h_p = pad_to_pow2(&q_heads[h]);
        let k_t_p = pad_to_pow2(&k_t);
        let scores_p = matmul_m31(&q_h_p, &k_t_p);

        verify_matmul_sumcheck(&proof.score_proofs[h], &q_h_p, &k_t_p, &scores_p)
            .map_err(|e| AggregationError::VerificationFailed(
                format!("Attention score head {h}: {e}")
            ))?;

        // Per-head: context = softmax × V_h (padded)
        let soft_p = pad_to_pow2(&inter.softmax_outputs[h]);
        let v_h_p = pad_to_pow2(&v_heads[h]);
        let context_p = matmul_m31(&soft_p, &v_h_p);

        verify_matmul_sumcheck(&proof.attn_v_proofs[h], &soft_p, &v_h_p, &context_p)
            .map_err(|e| AggregationError::VerificationFailed(
                format!("Attention attn_v head {h}: {e}")
            ))?;
    }

    // Output projection: concat × W_O = output (padded)
    let concat_p = pad_to_pow2(&inter.concat);
    let out_p = pad_to_pow2(&inter.final_output);
    verify_matmul_sumcheck(&proof.output_proof, &concat_p, &wo_p, &out_p)
        .map_err(|e| AggregationError::VerificationFailed(
            format!("Attention output projection: {e}")
        ))?;

    // Verify softmax exp STARK
    verify_attention_softmax_stark(
        proof.softmax_exp_proof,
        proof.softmax_claimed_sum,
        proof.softmax_log_size,
    )?;

    Ok(())
}

/// Verify the softmax exp STARK proof for an attention layer.
///
/// Reconstructs the ActivationEval component from the proof metadata,
/// sets up the commitment scheme verifier, and calls STWO's verify.
fn verify_attention_softmax_stark(
    proof: StarkProof<Blake2sHash>,
    claimed_sum: SecureField,
    log_size: u32,
) -> Result<(), AggregationError> {
    let config = PcsConfig::default();

    // Build dummy component to get per-tree column sizes
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component = FrameworkComponent::new(
        &mut allocator,
        ActivationEval {
            log_n_rows: log_size,
            lookup_elements: ActivationRelation::dummy(),
            claimed_sum,
            total_sum: claimed_sum,
        },
        claimed_sum,
    );

    let bounds = Component::trace_log_degree_bounds(&dummy_component);
    let tree0 = bounds[0].clone();
    let tree1 = bounds[1].clone();
    let tree2 = bounds[2].clone();

    // Set up channel and commitment scheme verifier
    let channel = &mut <Blake2sMerkleChannel as MerkleChannel>::C::default();
    let mut commitment_scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Commit Trees 0 (preprocessed) and 1 (execution) from proof
    commitment_scheme.commit(proof.commitments[0], &tree0, channel);
    commitment_scheme.commit(proof.commitments[1], &tree1, channel);

    // Draw lookup elements (must match prover's channel state)
    let lookup_elements = ActivationRelation::draw(channel);

    // Commit Tree 2 (LogUp interaction)
    commitment_scheme.commit(proof.commitments[2], &tree2, channel);

    // Build real component with drawn relation elements
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        ActivationEval {
            log_n_rows: log_size,
            lookup_elements,
            claimed_sum,
            total_sum: claimed_sum,
        },
        claimed_sum,
    );

    // Verify
    stwo_verify::<Blake2sMerkleChannel>(
        &[&component as &dyn Component],
        channel,
        &mut commitment_scheme,
        proof,
    ).map_err(|e| AggregationError::VerificationFailed(
        format!("Attention softmax STARK: {e}")
    ))?;

    Ok(())
}

/// Per-tree column sizes for the unified STARK.
struct TreeColumnSizes {
    tree0: Vec<u32>, // preprocessed
    tree1: Vec<u32>, // execution
    tree2: Vec<u32>, // interaction (LogUp)
}

/// Build column sizes for each commitment tree using dummy components.
///
/// The column structure depends only on log_n_rows and evaluator type,
/// NOT on the specific lookup element values, so dummy relations are used.
fn build_component_tree_sizes(
    activation_claims: &[LayerClaim],
    add_claims: &[LayerClaim],
    mul_claims: &[LayerClaim],
    layernorm_claims: &[LayerClaim],
    embedding_claims: &[LayerClaim],
    quantize_claims: &[LayerClaim],
    layernorm_dims: &HashMap<usize, usize>,
) -> TreeColumnSizes {
    let mut allocator = TraceLocationAllocator::default();
    let mut all_bounds: Vec<Vec<Vec<u32>>> = Vec::new();

    // Activation (dummy)
    for claim in activation_claims {
        let ls = claim.trace_rows.ilog2();
        let component = FrameworkComponent::new(
            &mut allocator,
            ActivationEval {
                log_n_rows: ls,
                lookup_elements: ActivationRelation::dummy(),
                claimed_sum: claim.claimed_sum,
                total_sum: claim.claimed_sum,
            },
            claim.claimed_sum,
        );
        let bounds = component.trace_log_degree_bounds();
        all_bounds.push(bounds.iter().map(|v| v.clone()).collect());
    }

    // Add (dummy)
    for claim in add_claims {
        let ls = claim.trace_rows.ilog2();
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseAddEval { log_n_rows: ls },
            SecureField::default(),
        );
        let bounds = component.trace_log_degree_bounds();
        all_bounds.push(bounds.iter().map(|v| v.clone()).collect());
    }

    // Mul (dummy)
    for claim in mul_claims {
        let ls = claim.trace_rows.ilog2();
        let component = FrameworkComponent::new(
            &mut allocator,
            ElementwiseMulEval { log_n_rows: ls },
            SecureField::default(),
        );
        let bounds = component.trace_log_degree_bounds();
        all_bounds.push(bounds.iter().map(|v| v.clone()).collect());
    }

    // LayerNorm (dummy)
    for claim in layernorm_claims {
        let ls = claim.trace_rows.ilog2();
        let dim = layernorm_dims.get(&claim.layer_index).copied().unwrap_or(1);
        let component = FrameworkComponent::new(
            &mut allocator,
            LayerNormEval {
                log_n_rows: ls,
                dim,
                lookup_elements: LayerNormRelation::dummy(),
                claimed_sum: claim.claimed_sum,
            },
            claim.claimed_sum,
        );
        let bounds = component.trace_log_degree_bounds();
        all_bounds.push(bounds.iter().map(|v| v.clone()).collect());
    }

    // Embedding (dummy)
    for claim in embedding_claims {
        let ls = claim.trace_rows.ilog2();
        let component = FrameworkComponent::new(
            &mut allocator,
            EmbeddingEval {
                log_n_rows: ls,
                lookup_elements: EmbeddingRelation::dummy(),
                claimed_sum: claim.claimed_sum,
            },
            claim.claimed_sum,
        );
        let bounds = component.trace_log_degree_bounds();
        all_bounds.push(bounds.iter().map(|v| v.clone()).collect());
    }

    // Quantize (dummy)
    for claim in quantize_claims {
        let ls = claim.trace_rows.ilog2();
        let component = FrameworkComponent::new(
            &mut allocator,
            QuantizeEval {
                log_n_rows: ls,
                lookup_elements: QuantizeRelation::dummy(),
                claimed_sum: claim.claimed_sum,
            },
            claim.claimed_sum,
        );
        let bounds = component.trace_log_degree_bounds();
        all_bounds.push(bounds.iter().map(|v| v.clone()).collect());
    }

    // Merge per-tree
    let mut tree0 = Vec::new();
    let mut tree1 = Vec::new();
    let mut tree2 = Vec::new();

    for bounds in &all_bounds {
        if bounds.len() > 0 { tree0.extend_from_slice(&bounds[0]); }
        if bounds.len() > 1 { tree1.extend_from_slice(&bounds[1]); }
        if bounds.len() > 2 { tree2.extend_from_slice(&bounds[2]); }
    }

    TreeColumnSizes { tree0, tree1, tree2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use crate::components::activation::ActivationType;
    use num_traits::Zero;
    use stwo::core::fields::FieldExpOps;

    #[test]
    fn test_aggregated_matmul_only() {
        // Model with no activations — should return None for unified_stark
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated proving should succeed");

        assert!(proof.unified_stark.is_none());
        assert_eq!(proof.matmul_proofs.len(), 1);
        assert_eq!(proof.num_proven_layers(), 1);
    }

    #[test]
    fn test_aggregated_mlp_with_activations() {
        // 5-layer MLP: matmul → relu → matmul → relu → matmul
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

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated MLP proving should succeed");

        // 1 STARK proof covering both ReLU layers
        assert!(proof.unified_stark.is_some(), "should have unified STARK");
        // 3 sumcheck proofs (one per matmul)
        assert_eq!(proof.matmul_proofs.len(), 3, "3 matmul sumcheck proofs");
        // 2 activation claims
        assert_eq!(proof.activation_claims.len(), 2, "2 activation layers in STARK");
        // Total: 5 proven layers
        assert_eq!(proof.num_proven_layers(), 5);
        // Output shape
        assert_eq!(proof.execution.output.rows, 1);
        assert_eq!(proof.execution.output.cols, 2);
    }

    #[test]
    fn test_aggregated_calldata_estimate() {
        let proof: AggregatedModelProof = AggregatedModelProofFor {
            unified_stark: None,
            matmul_proofs: Vec::new(),
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            execution: GraphExecution {
                intermediates: Vec::new(),
                output: M31Matrix::new(1, 1),
            },
            activation_claims: vec![
                LayerClaim { layer_index: 0, claimed_sum: SecureField::from(M31::from(0)), trace_rows: 16 },
                LayerClaim { layer_index: 1, claimed_sum: SecureField::from(M31::from(0)), trace_rows: 16 },
            ],
            attention_proofs: Vec::new(),
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            layer_chain_commitment: FieldElement::ZERO,
            io_commitment: FieldElement::ZERO,
            layernorm_mean_var_commitments: Vec::new(),
        };
        let calldata = proof.estimated_calldata_bytes();
        assert!(calldata > 0);
    }

    #[test]
    fn test_summarize_claims() {
        let claims = vec![
            LayerClaim { layer_index: 0, claimed_sum: SecureField::from(M31::from(0)), trace_rows: 1000 },
            LayerClaim { layer_index: 1, claimed_sum: SecureField::from(M31::from(0)), trace_rows: 2000 },
        ];
        let (num, total) = summarize_claims(&claims);
        assert_eq!(num, 2);
        assert_eq!(total, 3000);
    }

    // === On-Chain Aggregation Tests ===

    #[test]
    fn test_aggregated_onchain_mlp() {
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

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain aggregated MLP proving should succeed");

        assert!(proof.unified_stark.is_some());
        assert_eq!(proof.matmul_proofs.len(), 3);
        assert_eq!(proof.activation_claims.len(), 2);

        // All matmul proofs should have Poseidon commitments
        for (_, mp) in &proof.matmul_proofs {
            assert_ne!(mp.a_commitment, starknet_ff::FieldElement::ZERO);
            assert_ne!(mp.b_commitment, starknet_ff::FieldElement::ZERO);
        }
    }

    #[test]
    fn test_aggregated_onchain_matmul_only() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain matmul-only proving should succeed");

        assert!(proof.unified_stark.is_none());
        assert_eq!(proof.matmul_proofs.len(), 1);

        // Verify the matmul proof has on-chain format
        let (_, mp) = &proof.matmul_proofs[0];
        assert_eq!(mp.m, 1);
        assert_eq!(mp.k, 4);
        assert_eq!(mp.n, 2);
    }

    #[test]
    fn test_aggregated_onchain_auto_mlp() {
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

        let proof = prove_model_aggregated_onchain_auto(&graph, &input, &weights)
            .expect("on-chain auto aggregated proving should succeed");

        assert!(proof.unified_stark.is_some());
        assert_eq!(proof.matmul_proofs.len(), 2);
        assert_eq!(proof.activation_claims.len(), 1);
    }

    #[test]
    fn test_aggregated_onchain_auto_matmul_only() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated_onchain_auto(&graph, &input, &weights)
            .expect("on-chain auto matmul-only proving should succeed");

        assert!(proof.unified_stark.is_none());
        assert_eq!(proof.matmul_proofs.len(), 1);
    }

    #[test]
    fn test_aggregated_with_add_residual() {
        // Residual connection: matmul → relu → matmul → add(skip) → matmul
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

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated proving with residual Add should succeed");

        assert_eq!(proof.matmul_proofs.len(), 3, "3 matmul proofs");
        assert_eq!(proof.activation_claims.len(), 1, "1 activation claim (ReLU)");
        assert_eq!(proof.add_claims.len(), 1, "1 Add claim");
        assert_eq!(proof.mul_claims.len(), 0, "no Mul claims");
        assert_eq!(proof.layernorm_claims.len(), 0, "no LayerNorm claims");
        assert_eq!(proof.num_proven_layers(), 5, "total: 3 matmul + 1 activation + 1 add");
        // Unified STARK covers both activation and add
        assert!(proof.unified_stark.is_some(), "unified STARK covers activation + add");
    }

    #[test]
    fn test_aggregated_onchain_with_add() {
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

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain aggregated proving with Add should succeed");

        assert_eq!(proof.matmul_proofs.len(), 3);
        assert_eq!(proof.activation_claims.len(), 1);
        assert_eq!(proof.add_claims.len(), 1, "1 Add claim (on-chain)");
    }

    #[test]
    fn test_aggregated_with_mul() {
        // Element-wise multiply: matmul → fork → matmul → mul(branch) → matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let branch = builder.fork();
        builder.linear(4);
        builder.mul_from(branch);
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w1 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w1.set(i, j, M31::from(((i + j + 1) % 7 + 1) as u32)); } }
        weights.add_weight(1, w1);
        let mut w3 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w3.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(3, w3);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated proving with Mul should succeed");

        assert_eq!(proof.matmul_proofs.len(), 3, "3 matmul proofs");
        assert_eq!(proof.mul_claims.len(), 1, "1 Mul claim");
        assert_eq!(proof.add_claims.len(), 0, "no Add claims");
        // Unified STARK covers mul
        assert!(proof.unified_stark.is_some(), "unified STARK covers mul");
    }

    // === Verification Tests ===

    #[test]
    fn test_verify_aggregated_matmul_only() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");
        assert!(proof.unified_stark.is_none());

        verify_aggregated_model_proof(proof, &graph, &input, &weights)
            .expect("verification should succeed");
    }

    #[test]
    fn test_verify_aggregated_mlp() {
        // matmul → relu → matmul → relu → matmul
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

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");
        assert!(proof.unified_stark.is_some());
        assert_eq!(proof.matmul_proofs.len(), 3);
        assert_eq!(proof.activation_claims.len(), 2);

        verify_aggregated_model_proof(proof, &graph, &input, &weights)
            .expect("verification of MLP with activations should succeed");
    }

    // === Layer Chain Commitment Tests ===

    #[test]
    fn test_layer_chain_commitment_deterministic() {
        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut output = M31Matrix::new(1, 2);
        output.set(0, 0, M31::from(10));
        output.set(0, 1, M31::from(20));

        let intermediates = vec![(0, input.clone())];
        let c1 = compute_layer_chain_commitment(&input, &intermediates, &output);
        let c2 = compute_layer_chain_commitment(&input, &intermediates, &output);
        assert_eq!(c1, c2, "same data should produce same commitment");
        assert_ne!(c1, FieldElement::ZERO, "commitment should be non-zero");
    }

    #[test]
    fn test_layer_chain_commitment_different_intermediates() {
        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }
        let mut output = M31Matrix::new(1, 2);
        output.set(0, 0, M31::from(10));
        output.set(0, 1, M31::from(20));

        let inter1 = vec![(0, input.clone())];
        let mut tampered = input.clone();
        tampered.set(0, 0, M31::from(999));
        let inter2 = vec![(0, tampered)];

        let c1 = compute_layer_chain_commitment(&input, &inter1, &output);
        let c2 = compute_layer_chain_commitment(&input, &inter2, &output);
        assert_ne!(c1, c2, "different intermediates should produce different commitments");
    }

    #[test]
    fn test_layer_chain_commitment_any_element_change_detected() {
        // Verify that changing ANY element in a large matrix is detected.
        // This was the C5 vulnerability: the old 16-sample approach only checked
        // evenly-spaced positions, so an attacker could tamper with unsampled elements.
        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }
        let output = M31Matrix::new(1, 2);

        // Create a 1×64 intermediate (larger than the old SAMPLE_SIZE=16)
        let mut inter = M31Matrix::new(1, 64);
        for j in 0..64 { inter.set(0, j, M31::from(j as u32)); }
        let intermediates = vec![(0, inter.clone())];
        let original = compute_layer_chain_commitment(&input, &intermediates, &output);

        // Tamper with EVERY position — each must produce a different commitment
        for pos in 0..64 {
            let mut tampered = inter.clone();
            tampered.data[pos] = M31::from(9999);
            let tampered_intermediates = vec![(0, tampered)];
            let tampered_hash = compute_layer_chain_commitment(&input, &tampered_intermediates, &output);
            assert_ne!(
                original, tampered_hash,
                "Changing element at position {pos} must change the commitment"
            );
        }
    }

    #[test]
    fn test_layer_chain_commitment_in_proof() {
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

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        // Layer chain commitment should be non-zero
        assert_ne!(
            proof.layer_chain_commitment, FieldElement::ZERO,
            "layer chain commitment should be non-zero for multi-layer model"
        );

        // Verification should pass
        verify_aggregated_model_proof(proof, &graph, &input, &weights)
            .expect("verification should succeed with correct chain commitment");
    }

    #[test]
    fn test_layer_chain_commitment_tampered_fails() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let mut proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        // Tamper with the layer chain commitment
        proof.layer_chain_commitment = FieldElement::from(0xdeadbeefu64);

        // Verification should fail
        let result = verify_aggregated_model_proof(proof, &graph, &input, &weights);
        assert!(result.is_err(), "tampered chain commitment should fail verification");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Layer chain commitment mismatch"),
            "error should mention chain commitment, got: {err_msg}"
        );
    }

    // === IO Commitment (Proof Replay Prevention) Tests ===

    #[test]
    fn test_io_commitment_in_proof() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        // IO commitment should be non-zero
        assert_ne!(proof.io_commitment, FieldElement::ZERO);

        // Should match manual computation
        let expected_io = compute_io_commitment(&input, &proof.execution.output);
        assert_eq!(proof.io_commitment, expected_io);

        // Verification should pass
        verify_aggregated_model_proof(proof, &graph, &input, &weights)
            .expect("verification should succeed with correct io commitment");
    }

    #[test]
    fn test_io_commitment_tampered_fails() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let mut proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        // Tamper with the IO commitment (simulate replay attack)
        proof.io_commitment = FieldElement::from(0xcafebabeu64);

        // Verification should fail
        let result = verify_aggregated_model_proof(proof, &graph, &input, &weights);
        assert!(result.is_err(), "tampered IO commitment should fail verification");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("IO commitment mismatch"),
            "error should mention IO commitment, got: {err_msg}"
        );
    }

    #[test]
    fn test_io_commitment_different_inputs() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        // Prove with input 1
        let mut input1 = M31Matrix::new(1, 4);
        for j in 0..4 { input1.set(0, j, M31::from((j + 1) as u32)); }
        let proof1 = prove_model_aggregated(&graph, &input1, &weights)
            .expect("proving should succeed");

        // Prove with input 2
        let mut input2 = M31Matrix::new(1, 4);
        for j in 0..4 { input2.set(0, j, M31::from((j + 10) as u32)); }
        let proof2 = prove_model_aggregated(&graph, &input2, &weights)
            .expect("proving should succeed");

        // IO commitments should differ
        assert_ne!(
            proof1.io_commitment, proof2.io_commitment,
            "different inputs should produce different IO commitments"
        );

        // Trying to verify proof1 against input2 should fail (chain commitment will catch it first)
        let result = verify_aggregated_model_proof(proof1, &graph, &input2, &weights);
        assert!(result.is_err(), "proof for input1 should not verify against input2");
    }

    // === Model Completeness Tests ===

    #[test]
    fn test_completeness_check_passes() {
        // A proof with the right number of proofs/claims should pass
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

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        // Should have 2 matmul proofs + 1 activation claim = 3 proven layers
        assert_eq!(proof.matmul_proofs.len(), 2);
        assert_eq!(proof.activation_claims.len(), 1);

        // Verification (including completeness) should pass
        verify_aggregated_model_proof(proof, &graph, &input, &weights)
            .expect("completeness check should pass");
    }

    #[test]
    fn test_completeness_check_missing_matmul_fails() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w1 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w1.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(1, w1);

        let mut proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        // Remove one matmul proof to simulate incomplete proof
        assert_eq!(proof.matmul_proofs.len(), 2);
        proof.matmul_proofs.pop();

        // Verification should fail completeness check
        let result = verify_aggregated_model_proof(proof, &graph, &input, &weights);
        assert!(result.is_err(), "missing matmul proof should fail completeness");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Model completeness") && err_msg.contains("matmul"),
            "error should mention model completeness and matmul, got: {err_msg}"
        );
    }

    /// Helper: build a graph with an explicit Attention node and matching weights.
    fn build_attention_test_model() -> (crate::compiler::graph::ComputationGraph, GraphWeights, M31Matrix) {
        use crate::components::attention::MultiHeadAttentionConfig;

        let d_model = 4;
        let num_heads = 1;
        let config = MultiHeadAttentionConfig::new(num_heads, d_model, 1);

        let mut builder = GraphBuilder::new((1, d_model));
        builder.attention(config);
        builder.linear(d_model); // output linear to have a matmul too
        let graph = builder.build();

        // Build attention weights (d_model × d_model each)
        let mut weights = GraphWeights::new();
        let attn_node_id = 0; // first node is attention
        for (name, seed_offset) in &[("w_q", 1u32), ("w_k", 2), ("w_v", 3), ("w_o", 4)] {
            let mut w = M31Matrix::new(d_model, d_model);
            for i in 0..d_model {
                for j in 0..d_model {
                    w.set(i, j, M31::from((i * d_model + j + *seed_offset as usize) as u32 % 7 + 1));
                }
            }
            weights.add_named_weight(attn_node_id, name, w);
        }
        // MatMul weight for node 1 (the output linear)
        let matmul_node_id = 1;
        let mut w = M31Matrix::new(d_model, d_model);
        for i in 0..d_model {
            for j in 0..d_model {
                w.set(i, j, M31::from((i * d_model + j + 5) as u32 % 9 + 1));
            }
        }
        weights.add_weight(matmul_node_id, w);

        let mut input = M31Matrix::new(1, d_model);
        for j in 0..d_model { input.set(0, j, M31::from((j + 1) as u32)); }

        (graph, weights, input)
    }

    #[test]
    fn test_attention_proof_verified_in_aggregation() {
        let (graph, weights, input) = build_attention_test_model();

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        // The proof should contain attention proofs
        assert!(!proof.attention_proofs.is_empty(), "should have attention proofs");
        assert_eq!(proof.attention_proofs.len(), 1, "one attention node");

        // Verification should pass (attention proofs are now verified)
        verify_aggregated_model_proof(proof, &graph, &input, &weights)
            .expect("verification with attention proofs should succeed");
    }

    #[test]
    fn test_attention_proof_tampered_fails_verification() {
        let (graph, weights, input) = build_attention_test_model();

        let mut proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        assert!(!proof.attention_proofs.is_empty(), "should have attention proofs");

        // Tamper with the attention intermediates (corrupt the final output)
        proof.attention_proofs[0].1.intermediates.final_output.data[0] = M31::from(99999u32);

        // Verification should fail because forward pass output doesn't match
        let result = verify_aggregated_model_proof(proof, &graph, &input, &weights);
        assert!(result.is_err(), "tampered attention proof should fail verification");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Attention") || err_msg.contains("attention"),
            "error should mention attention, got: {err_msg}"
        );
    }

    // === On-chain verification tests ===

    #[test]
    fn test_verify_onchain_matmul_only() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain proving should succeed");

        verify_aggregated_model_proof_onchain(proof, &graph, &input, &weights)
            .expect("on-chain verification should succeed");
    }

    #[test]
    fn test_verify_onchain_mlp_with_activation() {
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
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain MLP proving should succeed");

        assert!(proof.unified_stark.is_some(), "should have unified STARK");
        assert_eq!(proof.matmul_proofs.len(), 2);

        verify_aggregated_model_proof_onchain(proof, &graph, &input, &weights)
            .expect("on-chain MLP verification should succeed");
    }

    #[test]
    fn test_verify_onchain_tampered_fails() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let mut proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain proving should succeed");

        // Tamper with a matmul proof's round polynomial
        if let Some((_, mp)) = proof.matmul_proofs.first_mut() {
            mp.round_polys[0].c0 = mp.round_polys[0].c0 + SecureField::one();
        }

        let result = verify_aggregated_model_proof_onchain(proof, &graph, &input, &weights);
        assert!(result.is_err(), "tampered on-chain proof should fail verification");
    }

    #[test]
    fn test_verify_onchain_wrong_input_fails() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain proving should succeed");

        // Verify with different input — IO commitment should mismatch
        let mut wrong_input = M31Matrix::new(1, 4);
        for j in 0..4 { wrong_input.set(0, j, M31::from((j + 10) as u32)); }

        let result = verify_aggregated_model_proof_onchain(proof, &graph, &wrong_input, &weights);
        assert!(result.is_err(), "wrong input should fail IO commitment");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("commitment mismatch"),
            "error should mention commitment mismatch, got: {err_msg}"
        );
    }

    #[test]
    fn test_verify_onchain_with_residual() {
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

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain residual proving should succeed");

        assert_eq!(proof.add_claims.len(), 1, "should have 1 Add claim");

        verify_aggregated_model_proof_onchain(proof, &graph, &input, &weights)
            .expect("on-chain residual verification should succeed");
    }

    #[test]
    fn test_aggregated_with_quantize() {
        use crate::gadgets::quantize::{QuantParams, QuantStrategy};

        // matmul → quantize(8-bit) → matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        builder.quantize(QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 8,
        });
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        // Small weights so matmul output fits in [0, 255]
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 3 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("aggregated proving with Quantize should succeed");

        assert_eq!(proof.quantize_claims.len(), 1, "should have 1 Quantize claim");

        verify_aggregated_model_proof(proof, &graph, &input, &weights)
            .expect("aggregated Quantize verification should succeed");
    }

    #[test]
    fn test_aggregated_onchain_with_quantize() {
        use crate::gadgets::quantize::{QuantParams, QuantStrategy};

        // matmul → quantize(8-bit) → matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        builder.quantize(QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 8,
        });
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 3 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain proving with Quantize should succeed");

        assert_eq!(proof.quantize_claims.len(), 1, "should have 1 Quantize claim (on-chain)");

        verify_aggregated_model_proof_onchain(proof, &graph, &input, &weights)
            .expect("on-chain Quantize verification should succeed");
    }

    #[test]
    fn test_quantize_tampered_fails_verification() {
        use crate::gadgets::quantize::{QuantParams, QuantStrategy};

        // matmul → quantize(8-bit) → matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        builder.quantize(QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 8,
        });
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 3 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        // Verify with wrong input — should fail
        let mut wrong_input = M31Matrix::new(1, 4);
        for j in 0..4 { wrong_input.set(0, j, M31::from((j + 10) as u32)); }

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        let result = verify_aggregated_model_proof(proof, &graph, &wrong_input, &weights);
        assert!(result.is_err(), "verification with wrong input should fail");
    }

    // === LayerNorm Mean/Var Commitment Tests ===

    #[test]
    fn test_layernorm_mean_var_commitment_verified() {
        // matmul → LayerNorm → matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).layer_norm().linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        // Should have exactly 1 LayerNorm mean/var commitment
        assert_eq!(proof.layernorm_mean_var_commitments.len(), 1,
            "should have 1 LayerNorm mean/var commitment");
        assert_ne!(proof.layernorm_mean_var_commitments[0], FieldElement::ZERO,
            "commitment should be non-zero");

        // Independently recompute and verify the commitment matches
        let matmul_output = matmul_m31(&input, weights.get_weight(0).unwrap());
        let ln = apply_layernorm_detailed(&matmul_output, 4);
        let expected = compute_layernorm_mean_var_commitment(&ln.means, &ln.variances);
        assert_eq!(proof.layernorm_mean_var_commitments[0], expected,
            "commitment should match independently computed value");

        verify_aggregated_model_proof(proof, &graph, &input, &weights)
            .expect("verification should succeed");
    }

    #[test]
    fn test_layernorm_tampered_commitment_fails() {
        // matmul → LayerNorm → matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).layer_norm().linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let mut proof = prove_model_aggregated(&graph, &input, &weights)
            .expect("proving should succeed");

        // Tamper with the LayerNorm mean/var commitment
        assert_eq!(proof.layernorm_mean_var_commitments.len(), 1);
        proof.layernorm_mean_var_commitments[0] = FieldElement::from(0xDEADu64);

        let result = verify_aggregated_model_proof(proof, &graph, &input, &weights);
        assert!(result.is_err(), "tampered commitment should fail verification");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("mean/var commitment mismatch"),
            "error should mention mean/var commitment mismatch, got: {err_msg}");
    }

    // Helper: build a valid BatchedMatMulProofOnChain by running the prover's
    // Fiat-Shamir logic on CPU. This lets us test the verifier without GPU.
    fn build_cpu_batch_proof(
        entries: &[(usize, SecureField, SecureField, SecureField, FieldElement, FieldElement, u32, u32)],
        k: u32,
    ) -> BatchedMatMulProofOnChain {
        use crate::crypto::poseidon_channel::{PoseidonChannel, securefield_to_felt};
        use crate::components::matmul::RoundPoly;

        let log_k = k.ilog2() as usize;
        let num_entries = entries.len();

        // Build per-entry MLE vectors (f_a, f_b) as simple polynomials
        // so we can compute real sumcheck round polynomials.
        let mut f_a_list: Vec<Vec<SecureField>> = Vec::new();
        let mut f_b_list: Vec<Vec<SecureField>> = Vec::new();
        let mut batch_entries: Vec<BatchedMatMulEntryOnChain> = Vec::new();

        for &(node_id, claimed_a, claimed_b, claimed_sum, a_commit, b_commit, m, n) in entries {
            // Simple MLEs: f_a = [claimed_a, 0, 0, ...], f_b = [claimed_b, 0, 0, ...]
            // adjusted so inner product = claimed_sum at index 0
            let mut f_a = vec![SecureField::zero(); k as usize];
            let mut f_b = vec![SecureField::zero(); k as usize];
            // Set f_a[0] and f_b[0] so their product-sum gives claimed_sum
            f_a[0] = claimed_a;
            f_b[0] = claimed_b;
            // Verify: Σ f_a[i]*f_b[i] = claimed_a * claimed_b = claimed_sum
            assert_eq!(claimed_a * claimed_b, claimed_sum);

            f_a_list.push(f_a);
            f_b_list.push(f_b);
            batch_entries.push(BatchedMatMulEntryOnChain {
                node_id,
                m,
                n,
                claimed_sum,
                final_a_eval: SecureField::zero(), // filled after sumcheck
                final_b_eval: SecureField::zero(),
                a_commitment: a_commit,
                b_commitment: b_commit,
            });
        }

        // Fiat-Shamir channel — exactly mirrors prove_matmul_batch_onchain_gpu
        let mut channel = PoseidonChannel::new();
        channel.mix_u64(num_entries as u64);
        channel.mix_u64(k as u64);

        for e in &batch_entries {
            channel.mix_felt(securefield_to_felt(e.claimed_sum));
            channel.mix_felt(e.a_commitment);
            channel.mix_felt(e.b_commitment);
        }

        let lambda = channel.draw_qm31();

        // Compute combined claimed sum
        let mut combined = SecureField::zero();
        let mut lambda_pow = SecureField::one();
        for e in &batch_entries {
            combined = combined + lambda_pow * e.claimed_sum;
            lambda_pow = lambda_pow * lambda;
        }

        // Run sumcheck rounds on CPU
        let mut round_polys = Vec::with_capacity(log_k);
        let mut cur_n = k as usize;

        for _ in 0..log_k {
            let mid = cur_n / 2;

            // Compute combined (s0, s1, s2) across all entries
            let mut combined_s0 = SecureField::zero();
            let mut combined_s1 = SecureField::zero();
            let mut combined_s2 = SecureField::zero();
            let mut lp = SecureField::one();

            for i in 0..num_entries {
                let f_a = &f_a_list[i];
                let f_b = &f_b_list[i];

                // s0 = Σ_{j<mid} f_a[j] * f_b[j]  (evaluate at t=0)
                let mut s0 = SecureField::zero();
                for j in 0..mid {
                    s0 = s0 + f_a[j] * f_b[j];
                }

                // s1 = Σ_{j<mid} f_a[mid+j] * f_b[mid+j]  (evaluate at t=1)
                let mut s1 = SecureField::zero();
                for j in 0..mid {
                    s1 = s1 + f_a[mid + j] * f_b[mid + j];
                }

                // s2 = Σ_{j<mid} (2*f_a[mid+j] - f_a[j]) * (2*f_b[mid+j] - f_b[j])  (evaluate at t=2)
                let two = SecureField::from(M31::from(2));
                let mut s2 = SecureField::zero();
                for j in 0..mid {
                    let a2 = two * f_a[mid + j] - f_a[j];
                    let b2 = two * f_b[mid + j] - f_b[j];
                    s2 = s2 + a2 * b2;
                }

                combined_s0 = combined_s0 + lp * s0;
                combined_s1 = combined_s1 + lp * s1;
                combined_s2 = combined_s2 + lp * s2;
                lp = lp * lambda;
            }

            // Lagrange interpolation → coefficients
            let c0 = combined_s0;
            let two = SecureField::from(M31::from(2));
            let c2 = (combined_s2 - two * combined_s1 + combined_s0) * two.inverse();
            let c1 = combined_s1 - combined_s0 - c2;

            round_polys.push(RoundPoly { c0, c1, c2 });

            // Fiat-Shamir: mix poly, draw challenge
            channel.mix_poly_coeffs(c0, c1, c2);
            let r_k = channel.draw_qm31();

            // Fold all MLEs with challenge r_k
            for i in 0..num_entries {
                let f_a = &f_a_list[i];
                let f_b = &f_b_list[i];
                let mut new_f_a = vec![SecureField::zero(); mid];
                let mut new_f_b = vec![SecureField::zero(); mid];
                for j in 0..mid {
                    new_f_a[j] = f_a[j] + r_k * (f_a[mid + j] - f_a[j]);
                    new_f_b[j] = f_b[j] + r_k * (f_b[mid + j] - f_b[j]);
                }
                f_a_list[i] = new_f_a;
                f_b_list[i] = new_f_b;
            }

            cur_n = mid;
        }

        // Extract final evaluations
        assert_eq!(cur_n, 1);
        for i in 0..num_entries {
            batch_entries[i].final_a_eval = f_a_list[i][0];
            batch_entries[i].final_b_eval = f_b_list[i][0];
        }

        BatchedMatMulProofOnChain {
            k,
            num_rounds: log_k as u32,
            lambda,
            combined_claimed_sum: combined,
            round_polys,
            entries: batch_entries,
        }
    }

    #[test]
    fn test_batch_fiat_shamir_replay_valid() {
        // Build a valid batch proof via CPU prover and verify it passes
        let entries = vec![
            // (node_id, f_a[0], f_b[0], claimed_sum, a_commit, b_commit, m, n)
            (0, SecureField::from(M31::from(3)), SecureField::from(M31::from(5)),
             SecureField::from(M31::from(15)),
             FieldElement::from(0xA1u64), FieldElement::from(0xB1u64), 1, 1),
            (1, SecureField::from(M31::from(7)), SecureField::from(M31::from(2)),
             SecureField::from(M31::from(14)),
             FieldElement::from(0xA2u64), FieldElement::from(0xB2u64), 1, 1),
        ];

        let batch = build_cpu_batch_proof(&entries, 4); // k=4, log_k=2 rounds
        let result = verify_batched_matmul_onchain(&batch);
        assert!(result.is_ok(), "Valid batch proof should verify: {:?}", result.err());
    }

    #[test]
    fn test_batch_fiat_shamir_replay_single_entry() {
        let entries = vec![
            (0, SecureField::from(M31::from(10)), SecureField::from(M31::from(20)),
             SecureField::from(M31::from(200)),
             FieldElement::from(0xC1u64), FieldElement::from(0xD1u64), 1, 1),
        ];

        let batch = build_cpu_batch_proof(&entries, 8); // k=8, log_k=3 rounds
        let result = verify_batched_matmul_onchain(&batch);
        assert!(result.is_ok(), "Single-entry batch should verify: {:?}", result.err());
    }

    #[test]
    fn test_batch_fiat_shamir_tampered_lambda_rejected() {
        let entries = vec![
            (0, SecureField::from(M31::from(3)), SecureField::from(M31::from(5)),
             SecureField::from(M31::from(15)),
             FieldElement::from(0xA1u64), FieldElement::from(0xB1u64), 1, 1),
            (1, SecureField::from(M31::from(7)), SecureField::from(M31::from(2)),
             SecureField::from(M31::from(14)),
             FieldElement::from(0xA2u64), FieldElement::from(0xB2u64), 1, 1),
        ];

        let mut batch = build_cpu_batch_proof(&entries, 4);
        // Tamper with stored lambda
        batch.lambda = SecureField::from(M31::from(999));

        let result = verify_batched_matmul_onchain(&batch);
        assert!(result.is_err(), "Tampered lambda should be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("lambda mismatch"), "Error should mention lambda: {err}");
    }

    #[test]
    fn test_batch_fiat_shamir_tampered_round_poly_rejected() {
        let entries = vec![
            (0, SecureField::from(M31::from(3)), SecureField::from(M31::from(5)),
             SecureField::from(M31::from(15)),
             FieldElement::from(0xA1u64), FieldElement::from(0xB1u64), 1, 1),
            (1, SecureField::from(M31::from(7)), SecureField::from(M31::from(2)),
             SecureField::from(M31::from(14)),
             FieldElement::from(0xA2u64), FieldElement::from(0xB2u64), 1, 1),
        ];

        let mut batch = build_cpu_batch_proof(&entries, 4);
        // Tamper with a round polynomial coefficient
        batch.round_polys[0].c1 = batch.round_polys[0].c1 + SecureField::from(M31::from(1));

        let result = verify_batched_matmul_onchain(&batch);
        assert!(result.is_err(), "Tampered round poly should be rejected");
    }

    #[test]
    fn test_batch_fiat_shamir_tampered_claimed_sum_rejected() {
        let entries = vec![
            (0, SecureField::from(M31::from(3)), SecureField::from(M31::from(5)),
             SecureField::from(M31::from(15)),
             FieldElement::from(0xA1u64), FieldElement::from(0xB1u64), 1, 1),
        ];

        let mut batch = build_cpu_batch_proof(&entries, 4);
        // Tamper with combined claimed sum
        batch.combined_claimed_sum = batch.combined_claimed_sum + SecureField::from(M31::from(1));

        let result = verify_batched_matmul_onchain(&batch);
        assert!(result.is_err(), "Tampered claimed sum should be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("combined_claimed_sum mismatch"), "Error should mention claimed sum: {err}");
    }

    #[test]
    fn test_batch_fiat_shamir_tampered_final_eval_rejected() {
        let entries = vec![
            (0, SecureField::from(M31::from(3)), SecureField::from(M31::from(5)),
             SecureField::from(M31::from(15)),
             FieldElement::from(0xA1u64), FieldElement::from(0xB1u64), 1, 1),
            (1, SecureField::from(M31::from(7)), SecureField::from(M31::from(2)),
             SecureField::from(M31::from(14)),
             FieldElement::from(0xA2u64), FieldElement::from(0xB2u64), 1, 1),
        ];

        let mut batch = build_cpu_batch_proof(&entries, 4);
        // Tamper with a final evaluation
        batch.entries[0].final_a_eval = batch.entries[0].final_a_eval + SecureField::from(M31::from(1));

        let result = verify_batched_matmul_onchain(&batch);
        assert!(result.is_err(), "Tampered final eval should be rejected");
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("final eval mismatch"), "Error should mention final eval: {err}");
    }

    #[test]
    fn test_batch_fiat_shamir_four_entries() {
        // Larger batch with 4 matmuls
        let entries = vec![
            (0, SecureField::from(M31::from(1)), SecureField::from(M31::from(1)),
             SecureField::from(M31::from(1)),
             FieldElement::from(0x10u64), FieldElement::from(0x20u64), 1, 1),
            (1, SecureField::from(M31::from(2)), SecureField::from(M31::from(3)),
             SecureField::from(M31::from(6)),
             FieldElement::from(0x30u64), FieldElement::from(0x40u64), 1, 1),
            (2, SecureField::from(M31::from(4)), SecureField::from(M31::from(5)),
             SecureField::from(M31::from(20)),
             FieldElement::from(0x50u64), FieldElement::from(0x60u64), 2, 2),
            (3, SecureField::from(M31::from(6)), SecureField::from(M31::from(7)),
             SecureField::from(M31::from(42)),
             FieldElement::from(0x70u64), FieldElement::from(0x80u64), 2, 2),
        ];

        let batch = build_cpu_batch_proof(&entries, 16); // k=16, log_k=4 rounds
        let result = verify_batched_matmul_onchain(&batch);
        assert!(result.is_ok(), "4-entry batch should verify: {:?}", result.err());
    }
}
