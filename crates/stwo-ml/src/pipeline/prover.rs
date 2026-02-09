//! Pipeline prover: chains per-layer proofs with Poseidon commitment linking.
//!
//! Proves each transformer layer (matmul → activation → layernorm) independently,
//! linking outputs to next layer's inputs via Poseidon commitments.

use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;

use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
#[cfg(feature = "safetensors")]
use crate::compiler::safetensors::LazyWeights;
use crate::compiler::prove::{prove_activation_layer, ModelError};
use crate::components::activation::ActivationType;
use crate::components::attention::{
    AttentionWeights, attention_forward, prove_attention_onchain,
};
use crate::components::matmul::{
    M31Matrix, matmul_m31,
    prove_matmul_sumcheck_onchain,
};
use crate::gadgets::lookup_table::PrecomputedTable;
use crate::receipt::ComputeReceipt;

use super::types::{
    LayerPipelineProof, LayerProofKindOnChain,
    ModelPipelineProof, PipelineConfig,
    commit_matrix, commit_model_weights,
    compute_pipeline_io_commitment,
};

/// Error type for pipeline proving.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Model error at layer {layer}: {message}")]
    ModelError { layer: usize, message: String },
    #[error("MatMul proving failed at layer {layer}: {source}")]
    MatMulError {
        layer: usize,
        #[source]
        source: crate::components::matmul::MatMulError,
    },
    #[error("Weight not found for node {0}")]
    MissingWeight(usize),
    #[error("Empty graph")]
    EmptyGraph,
}

impl From<ModelError> for PipelineError {
    fn from(e: ModelError) -> Self {
        PipelineError::ModelError {
            layer: 0,
            message: e.to_string(),
        }
    }
}

/// Apply an activation function element-wise to a matrix.
fn apply_activation(input: &M31Matrix, f: &dyn Fn(M31) -> M31) -> M31Matrix {
    let mut output = M31Matrix::new(input.rows, input.cols);
    for i in 0..input.data.len() {
        output.data[i] = f(input.data[i]);
    }
    output
}

/// Apply LayerNorm in M31 arithmetic (mean-center and scale by rsqrt).
fn apply_layernorm(input: &M31Matrix, dim: usize) -> M31Matrix {
    use crate::components::layernorm::build_rsqrt_table;

    let rsqrt_table = build_rsqrt_table(16);
    let mut output = M31Matrix::new(input.rows, input.cols);
    let n = dim.min(input.cols);
    let inv_n = m31_mod_inverse(n as u32);

    for row in 0..input.rows {
        let mut sum = M31::from(0u32);
        for col in 0..n {
            sum += input.data[row * input.cols + col];
        }
        let mean = sum * inv_n;

        let mut var_sum = M31::from(0u32);
        for col in 0..n {
            let diff = input.data[row * input.cols + col] - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum * inv_n;

        let rsqrt = rsqrt_table
            .lookup(variance)
            .unwrap_or(M31::from(1u32 << 16));

        for col in 0..n {
            let centered = input.data[row * input.cols + col] - mean;
            output.data[row * input.cols + col] = centered * rsqrt;
        }
        for col in n..input.cols {
            output.data[row * input.cols + col] = input.data[row * input.cols + col];
        }
    }
    output
}

/// Modular inverse of n in M31 via Fermat's little theorem.
fn m31_mod_inverse(n: u32) -> M31 {
    if n == 0 {
        return M31::from(0u32);
    }
    let p: u64 = (1u64 << 31) - 1;
    let mut result: u64 = 1;
    let mut base = n as u64 % p;
    let mut exp = p - 2;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % p;
        }
        base = base * base % p;
        exp >>= 1;
    }
    M31::from(result as u32)
}

/// Prove a single transformer layer: matmul → optional activation.
///
/// Returns the layer proof and the output matrix from this layer.
fn prove_matmul_layer(
    layer_index: usize,
    current: &M31Matrix,
    weight: &M31Matrix,
) -> Result<(LayerPipelineProof, M31Matrix), PipelineError> {
    // Forward pass
    let output = matmul_m31(current, weight);

    // Commitment linking
    let input_commitment = commit_matrix(current);
    let output_commitment = commit_matrix(&output);

    // On-chain sumcheck proof (Poseidon Fiat-Shamir)
    let proof = prove_matmul_sumcheck_onchain(current, weight, &output)
        .map_err(|e| PipelineError::MatMulError {
            layer: layer_index,
            source: e,
        })?;

    Ok((
        LayerPipelineProof {
            layer_index,
            kind: LayerProofKindOnChain::MatMulSumcheck(proof),
            input_commitment,
            output_commitment,
        },
        output,
    ))
}

/// Prove a single activation layer with STARK proof.
fn prove_activation_layer_pipeline(
    layer_index: usize,
    current: &M31Matrix,
    activation_type: &ActivationType,
) -> Result<(LayerPipelineProof, M31Matrix), PipelineError> {
    let f = activation_type.as_fn();
    let output = apply_activation(current, &*f);

    let input_commitment = commit_matrix(current);
    let output_commitment = commit_matrix(&output);

    // Build production-sized lookup table
    let table = PrecomputedTable::build_parallel(move |x| (*f)(x), activation_type.production_log_size());
    let config = PcsConfig::default();

    let flat_inputs: Vec<M31> = current.data.clone();
    let flat_outputs: Vec<M31> = output.data.clone();

    let (_component, stark_proof) =
        prove_activation_layer::<SimdBackend, Blake2sMerkleChannel>(
            &flat_inputs,
            &flat_outputs,
            &table,
            config,
        )
        .map_err(|e| PipelineError::ModelError {
            layer: layer_index,
            message: format!("Activation STARK: {e}"),
        })?;

    Ok((
        LayerPipelineProof {
            layer_index,
            kind: LayerProofKindOnChain::ActivationStark(stark_proof),
            input_commitment,
            output_commitment,
        },
        output,
    ))
}

/// Prove an entire model pipeline: all layers with commitment chaining.
///
/// Iterates through the computation graph, generating per-layer proofs
/// and linking them via Poseidon commitments. Memory-efficient: only
/// one layer's data is live at a time.
pub fn prove_model_pipeline(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
    config: &PipelineConfig,
) -> Result<ModelPipelineProof, PipelineError> {
    if graph.nodes.is_empty() {
        return Err(PipelineError::EmptyGraph);
    }

    // Commit all weight matrices → model commitment
    let model_commitment = if let Some(precomputed) = config.precomputed_model_commitment {
        precomputed
    } else {
        let mut weight_commitments = Vec::new();
        for node in &graph.nodes {
            match &node.op {
                GraphOp::MatMul { .. } => {
                    if let Some(w) = weights.get_weight(node.id) {
                        weight_commitments.push(commit_matrix(w));
                    }
                }
                GraphOp::Attention { .. } => {
                    for name in &["w_q", "w_k", "w_v", "w_o"] {
                        if let Some(w) = weights.get_named_weight(node.id, name) {
                            weight_commitments.push(commit_matrix(w));
                        }
                    }
                }
                _ => {}
            }
        }
        commit_model_weights(&weight_commitments)
    };

    let mut layer_proofs = Vec::new();
    let mut current = input.clone();
    let original_input = input.clone();

    for node in &graph.nodes {
        match &node.op {
            GraphOp::MatMul { .. } => {
                let weight = weights
                    .get_weight(node.id)
                    .ok_or(PipelineError::MissingWeight(node.id))?;

                if config.onchain_matmul {
                    let (proof, output) = prove_matmul_layer(node.id, &current, weight)?;
                    layer_proofs.push(proof);
                    current = output;
                } else {
                    // Off-chain: compute forward pass, no proof
                    let output = matmul_m31(&current, weight);
                    let input_commitment = commit_matrix(&current);
                    let output_commitment = commit_matrix(&output);
                    layer_proofs.push(LayerPipelineProof {
                        layer_index: node.id,
                        kind: LayerProofKindOnChain::Passthrough,
                        input_commitment,
                        output_commitment,
                    });
                    current = output;
                }
            }

            GraphOp::Activation { activation_type, .. } => {
                if config.prove_activations {
                    let (proof, output) =
                        prove_activation_layer_pipeline(node.id, &current, activation_type)?;
                    layer_proofs.push(proof);
                    current = output;
                } else {
                    let f = activation_type.as_fn();
                    let output = apply_activation(&current, &*f);
                    let input_commitment = commit_matrix(&current);
                    let output_commitment = commit_matrix(&output);
                    layer_proofs.push(LayerPipelineProof {
                        layer_index: node.id,
                        kind: LayerProofKindOnChain::Passthrough,
                        input_commitment,
                        output_commitment,
                    });
                    current = output;
                }
            }

            GraphOp::LayerNorm { dim } => {
                let output = apply_layernorm(&current, *dim);
                let input_commitment = commit_matrix(&current);
                let output_commitment = commit_matrix(&output);
                layer_proofs.push(LayerPipelineProof {
                    layer_index: node.id,
                    kind: LayerProofKindOnChain::Passthrough,
                    input_commitment,
                    output_commitment,
                });
                current = output;
            }

            GraphOp::Attention { config: attn_config } => {
                let attn_weights = AttentionWeights {
                    w_q: weights.get_named_weight(node.id, "w_q")
                        .ok_or(PipelineError::MissingWeight(node.id))?.clone(),
                    w_k: weights.get_named_weight(node.id, "w_k")
                        .ok_or(PipelineError::MissingWeight(node.id))?.clone(),
                    w_v: weights.get_named_weight(node.id, "w_v")
                        .ok_or(PipelineError::MissingWeight(node.id))?.clone(),
                    w_o: weights.get_named_weight(node.id, "w_o")
                        .ok_or(PipelineError::MissingWeight(node.id))?.clone(),
                };

                let input_commitment = commit_matrix(&current);

                if config.onchain_matmul {
                    let proof = prove_attention_onchain(
                        &current, &attn_weights, attn_config, false,
                    ).map_err(|e| PipelineError::ModelError {
                        layer: node.id,
                        message: format!("Attention: {e}"),
                    })?;

                    current = proof.intermediates.final_output.clone();
                    let output_commitment = commit_matrix(&current);

                    layer_proofs.push(LayerPipelineProof {
                        layer_index: node.id,
                        kind: LayerProofKindOnChain::Attention(Box::new(proof)),
                        input_commitment,
                        output_commitment,
                    });
                } else {
                    let inter = attention_forward(&current, &attn_weights, attn_config, false);
                    current = inter.final_output;
                    let output_commitment = commit_matrix(&current);

                    layer_proofs.push(LayerPipelineProof {
                        layer_index: node.id,
                        kind: LayerProofKindOnChain::Passthrough,
                        input_commitment,
                        output_commitment,
                    });
                }
            }

            GraphOp::Quantize { .. } | GraphOp::Identity { .. } => {
                let input_commitment = commit_matrix(&current);
                let output_commitment = input_commitment;
                layer_proofs.push(LayerPipelineProof {
                    layer_index: node.id,
                    kind: LayerProofKindOnChain::Passthrough,
                    input_commitment,
                    output_commitment,
                });
            }
        }
    }

    let io_commitment = compute_pipeline_io_commitment(&original_input, &current);

    // Generate receipt if configured
    let receipt = if config.generate_receipt {
        Some(build_receipt(
            &original_input,
            &current,
            model_commitment,
            &layer_proofs,
        ))
    } else {
        None
    };

    Ok(ModelPipelineProof {
        model_commitment,
        io_commitment,
        layer_proofs,
        receipt,
        tee_report_hash: None,
    })
}

/// Prove an entire model pipeline with streaming weight loading.
///
/// Identical to [`prove_model_pipeline`] but takes `&mut LazyWeights` instead
/// of `&GraphWeights`. Weights are loaded (quantized) per-layer from mmap'd
/// SafeTensors and evicted after use, keeping peak memory to the size of a
/// single layer's weights.
///
/// The proving happens in two streaming passes:
/// 1. **Commitment pass**: load → commit → evict each weight-bearing node.
/// 2. **Proving pass**: load → forward + prove → evict each node.
///
/// Re-quantization in pass 2 is deterministic (same mmap bytes → same M31),
/// so commitments computed in pass 1 match the weights used in pass 2.
#[cfg(feature = "safetensors")]
pub fn prove_model_pipeline_streaming(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &mut LazyWeights,
    config: &PipelineConfig,
) -> Result<ModelPipelineProof, PipelineError> {
    if graph.nodes.is_empty() {
        return Err(PipelineError::EmptyGraph);
    }

    // --- Pass 1: Streaming commitment ---
    let mut weight_commitments = Vec::new();
    for node in &graph.nodes {
        match &node.op {
            GraphOp::MatMul { .. } => {
                weights.load_layer(node.id).map_err(|e| PipelineError::ModelError {
                    layer: node.id,
                    message: format!("weight load: {e}"),
                })?;
                if let Some(w) = weights.get_weight(node.id) {
                    weight_commitments.push(commit_matrix(w));
                }
                weights.evict_layer(node.id);
            }
            // Attention would need named weights — skip for now (not supported
            // in streaming mode; fall back to eager for attention models).
            _ => {}
        }
    }
    let model_commitment = commit_model_weights(&weight_commitments);

    // --- Pass 2: Streaming proving ---
    let mut layer_proofs = Vec::new();
    let mut current = input.clone();
    let original_input = input.clone();

    for node in &graph.nodes {
        match &node.op {
            GraphOp::MatMul { .. } => {
                weights.load_layer(node.id).map_err(|e| PipelineError::ModelError {
                    layer: node.id,
                    message: format!("weight load: {e}"),
                })?;
                let weight = weights
                    .get_weight(node.id)
                    .ok_or(PipelineError::MissingWeight(node.id))?;

                if config.onchain_matmul {
                    let (proof, output) = prove_matmul_layer(node.id, &current, weight)?;
                    layer_proofs.push(proof);
                    current = output;
                } else {
                    let output = matmul_m31(&current, weight);
                    let input_commitment = commit_matrix(&current);
                    let output_commitment = commit_matrix(&output);
                    layer_proofs.push(LayerPipelineProof {
                        layer_index: node.id,
                        kind: LayerProofKindOnChain::Passthrough,
                        input_commitment,
                        output_commitment,
                    });
                    current = output;
                }
                weights.evict_layer(node.id);
            }

            GraphOp::Activation { activation_type, .. } => {
                if config.prove_activations {
                    let (proof, output) =
                        prove_activation_layer_pipeline(node.id, &current, activation_type)?;
                    layer_proofs.push(proof);
                    current = output;
                } else {
                    let f = activation_type.as_fn();
                    let output = apply_activation(&current, &*f);
                    let input_commitment = commit_matrix(&current);
                    let output_commitment = commit_matrix(&output);
                    layer_proofs.push(LayerPipelineProof {
                        layer_index: node.id,
                        kind: LayerProofKindOnChain::Passthrough,
                        input_commitment,
                        output_commitment,
                    });
                    current = output;
                }
            }

            GraphOp::LayerNorm { dim } => {
                let output = apply_layernorm(&current, *dim);
                let input_commitment = commit_matrix(&current);
                let output_commitment = commit_matrix(&output);
                layer_proofs.push(LayerPipelineProof {
                    layer_index: node.id,
                    kind: LayerProofKindOnChain::Passthrough,
                    input_commitment,
                    output_commitment,
                });
                current = output;
            }

            GraphOp::Quantize { .. } | GraphOp::Identity { .. } => {
                let input_commitment = commit_matrix(&current);
                let output_commitment = input_commitment;
                layer_proofs.push(LayerPipelineProof {
                    layer_index: node.id,
                    kind: LayerProofKindOnChain::Passthrough,
                    input_commitment,
                    output_commitment,
                });
            }

            // Attention layers not yet supported in streaming mode.
            GraphOp::Attention { .. } => {
                return Err(PipelineError::ModelError {
                    layer: node.id,
                    message: "Attention not supported in streaming mode; use prove_model_pipeline".into(),
                });
            }
        }
    }

    let io_commitment = compute_pipeline_io_commitment(&original_input, &current);

    let receipt = if config.generate_receipt {
        Some(build_receipt(
            &original_input,
            &current,
            model_commitment,
            &layer_proofs,
        ))
    } else {
        None
    };

    Ok(ModelPipelineProof {
        model_commitment,
        io_commitment,
        layer_proofs,
        receipt,
        tee_report_hash: None,
    })
}

/// Build a compute receipt for the pipeline proof.
fn build_receipt(
    input: &M31Matrix,
    output: &M31Matrix,
    model_commitment: FieldElement,
    layer_proofs: &[LayerPipelineProof],
) -> ComputeReceipt {
    use super::types::commit_values;

    let input_commitment = commit_values(&input.data);
    let output_commitment = commit_values(&output.data);
    let num_proven = layer_proofs
        .iter()
        .filter(|p| !matches!(p.kind, LayerProofKindOnChain::Passthrough))
        .count();

    ComputeReceipt {
        job_id: FieldElement::from(1u64),
        worker_pubkey: FieldElement::from(0u64),
        input_commitment,
        output_commitment,
        model_commitment,
        prev_receipt_hash: FieldElement::ZERO,
        gpu_time_ms: 0,
        token_count: num_proven as u32,
        peak_memory_mb: 0,
        billing_amount_sage: 0,
        billing_rate_per_sec: 0,
        billing_rate_per_token: 0,
        tee_report_hash: FieldElement::ZERO,
        tee_timestamp: 0,
        timestamp: 0,
        sequence_number: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use crate::compiler::onnx::generate_weights_for_graph;
    use crate::components::activation::ActivationType;

    #[test]
    fn test_prove_single_matmul_layer() {
        // 4×4 matmul (power of 2 required by sumcheck)
        let mut builder = GraphBuilder::new((4, 4));
        builder.linear(4);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 42);
        let input = {
            let mut m = M31Matrix::new(4, 4);
            for i in 0..4 {
                for j in 0..4 {
                    m.set(i, j, M31::from((i * 4 + j + 1) as u32));
                }
            }
            m
        };

        let config = PipelineConfig {
            onchain_matmul: true,
            prove_activations: false,
            generate_receipt: false,
            precomputed_model_commitment: None,
        };

        let proof = prove_model_pipeline(&graph, &input, &weights, &config).unwrap();

        assert_eq!(proof.layer_proofs.len(), 1);
        assert!(matches!(
            proof.layer_proofs[0].kind,
            LayerProofKindOnChain::MatMulSumcheck(_)
        ));
        assert_ne!(proof.model_commitment, FieldElement::ZERO);
        assert_ne!(proof.io_commitment, FieldElement::ZERO);
    }

    #[test]
    fn test_prove_matmul_activation_chain() {
        let mut builder = GraphBuilder::new((4, 4));
        builder.linear(4).activation(ActivationType::ReLU);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 42);
        let input = {
            let mut m = M31Matrix::new(4, 4);
            for i in 0..16 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };

        let config = PipelineConfig::default();
        let proof = prove_model_pipeline(&graph, &input, &weights, &config).unwrap();

        assert_eq!(proof.layer_proofs.len(), 2);
        assert!(matches!(
            proof.layer_proofs[0].kind,
            LayerProofKindOnChain::MatMulSumcheck(_)
        ));
        assert!(matches!(
            proof.layer_proofs[1].kind,
            LayerProofKindOnChain::ActivationStark(_)
        ));

        // Commitment chain: matmul output == activation input
        assert_eq!(
            proof.layer_proofs[0].output_commitment,
            proof.layer_proofs[1].input_commitment,
            "Commitment chain must be continuous"
        );
    }

    #[test]
    fn test_commitment_chain_continuity() {
        // 2-layer MLP: linear → relu → linear → relu
        let mut builder = GraphBuilder::new((4, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(4)
            .activation(ActivationType::ReLU);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 123);
        let input = {
            let mut m = M31Matrix::new(4, 4);
            for i in 0..16 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };

        let config = PipelineConfig::default();
        let proof = prove_model_pipeline(&graph, &input, &weights, &config).unwrap();

        assert_eq!(proof.layer_proofs.len(), 4);
        assert!(proof.verify_commitment_chain());
    }

    #[test]
    fn test_receipt_generation() {
        let mut builder = GraphBuilder::new((4, 4));
        builder.linear(4);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 42);
        let input = {
            let mut m = M31Matrix::new(4, 4);
            for i in 0..16 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };

        let config = PipelineConfig {
            onchain_matmul: true,
            prove_activations: false,
            generate_receipt: true,
            precomputed_model_commitment: None,
        };

        let proof = prove_model_pipeline(&graph, &input, &weights, &config).unwrap();
        assert!(proof.receipt.is_some());

        let receipt = proof.receipt.as_ref().unwrap();
        assert_eq!(receipt.model_commitment, proof.model_commitment);
        assert_ne!(receipt.receipt_hash(), FieldElement::ZERO);
    }

    /// Helper: build a SafeTensors file from GraphWeights for streaming tests.
    #[cfg(feature = "safetensors")]
    fn build_safetensors_for_graph(
        graph: &ComputationGraph,
        weights: &GraphWeights,
    ) -> std::path::PathBuf {
        use std::collections::HashMap as StdMap;

        let tmp = std::env::temp_dir().join(format!(
            "stwo_ml_stream_test_{}.safetensors",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        // Collect byte data with stable ownership before building tensor views
        let mut byte_data: StdMap<String, (Vec<u8>, usize, usize)> = StdMap::new();
        for (idx, node) in graph.nodes.iter().enumerate() {
            if let GraphOp::MatMul { .. } = &node.op {
                if let Some(w) = weights.get_weight(idx) {
                    let bytes: Vec<u8> = w.data.iter()
                        .flat_map(|v| (v.0 as f32).to_le_bytes())
                        .collect();
                    byte_data.insert(format!("weight.{idx}"), (bytes, w.rows, w.cols));
                }
            }
        }

        let mut tensors_map = StdMap::new();
        for (name, (bytes, rows, cols)) in &byte_data {
            tensors_map.insert(
                name.clone(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![*rows, *cols],
                    bytes,
                ).unwrap(),
            );
        }

        let serialized = safetensors::serialize(&tensors_map, &None).unwrap();
        std::fs::write(&tmp, &serialized).unwrap();
        tmp
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_streaming_proves_same_as_eager() {
        use crate::compiler::safetensors::{LazyWeights, load_weights};
        use crate::gadgets::quantize::QuantStrategy;

        let mut builder = GraphBuilder::new((4, 4));
        builder.linear(4);
        let graph = builder.build();

        // Create a SafeTensors file with known f32 data
        let tmp = {
            use std::collections::HashMap as StdMap;

            let path = std::env::temp_dir().join(format!(
                "stwo_ml_stream_eager_{}.safetensors",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let w_data: Vec<f32> = (0..16).map(|i| (i + 1) as f32 * 0.1).collect();
            let w_bytes: Vec<u8> = w_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let mut tensors = StdMap::new();
            tensors.insert(
                "weight.0".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32, vec![4, 4], &w_bytes,
                ).unwrap(),
            );
            let serialized = safetensors::serialize(&tensors, &None).unwrap();
            std::fs::write(&path, &serialized).unwrap();
            path
        };

        let input = {
            let mut m = M31Matrix::new(4, 4);
            for i in 0..16 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };

        let config = PipelineConfig {
            onchain_matmul: true,
            prove_activations: false,
            generate_receipt: false,
            precomputed_model_commitment: None,
        };

        // Both paths load from the same SafeTensors file → same quantization
        let eager_weights = load_weights(&tmp, &graph, QuantStrategy::Direct).unwrap();
        let eager_proof = prove_model_pipeline(&graph, &input, &eager_weights, &config).unwrap();

        let mut lazy = LazyWeights::open(&tmp, &graph, QuantStrategy::Direct).unwrap();
        let stream_proof = prove_model_pipeline_streaming(
            &graph, &input, &mut lazy, &config,
        ).unwrap();

        // Model commitments must match (same weights → same Poseidon hash)
        assert_eq!(
            eager_proof.model_commitment,
            stream_proof.model_commitment,
            "model commitments must match"
        );
        assert_eq!(
            eager_proof.io_commitment,
            stream_proof.io_commitment,
            "IO commitments must match"
        );
        assert_eq!(
            eager_proof.layer_proofs.len(),
            stream_proof.layer_proofs.len(),
        );

        std::fs::remove_file(&tmp).ok();
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_streaming_commitment_chain_continuity() {
        use crate::compiler::safetensors::LazyWeights;
        use crate::gadgets::quantize::QuantStrategy;

        let mut builder = GraphBuilder::new((4, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(4)
            .activation(ActivationType::ReLU);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 123);
        let input = {
            let mut m = M31Matrix::new(4, 4);
            for i in 0..16 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };

        let config = PipelineConfig::default();

        let tmp = build_safetensors_for_graph(&graph, &weights);
        let mut lazy = LazyWeights::open(&tmp, &graph, QuantStrategy::Direct).unwrap();
        let proof = prove_model_pipeline_streaming(
            &graph, &input, &mut lazy, &config,
        ).unwrap();

        assert_eq!(proof.layer_proofs.len(), 4);
        assert!(
            proof.verify_commitment_chain(),
            "streaming proof must have continuous commitment chain"
        );

        std::fs::remove_file(&tmp).ok();
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn test_streaming_eviction_between_layers() {
        use crate::compiler::safetensors::LazyWeights;
        use crate::gadgets::quantize::QuantStrategy;

        let mut builder = GraphBuilder::new((4, 4));
        builder.linear(4).linear(4);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 55);
        let input = {
            let mut m = M31Matrix::new(4, 4);
            for i in 0..16 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };

        let config = PipelineConfig {
            onchain_matmul: false, // passthrough — faster for this test
            prove_activations: false,
            generate_receipt: false,
            precomputed_model_commitment: None,
        };

        let tmp = build_safetensors_for_graph(&graph, &weights);
        let mut lazy = LazyWeights::open(&tmp, &graph, QuantStrategy::Direct).unwrap();

        let proof = prove_model_pipeline_streaming(
            &graph, &input, &mut lazy, &config,
        ).unwrap();

        // After streaming, no layers should be cached (all evicted)
        assert!(
            lazy.cached_layers().is_empty(),
            "all layers should be evicted after streaming prove"
        );
        assert_eq!(proof.layer_proofs.len(), 2);
        assert_ne!(proof.model_commitment, FieldElement::ZERO);

        std::fs::remove_file(&tmp).ok();
    }
}
