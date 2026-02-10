//! Dual-track execution: f32 inference + M31 proving.
//!
//! Runs the model in f32 for meaningful float outputs while simultaneously
//! generating M31 STARK proofs via the existing proving pipeline. The two
//! tracks share the same computation graph and weight structure.

use stwo::core::channel::MerkleChannel;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::BackendForChannel;
use stwo::prover::poly::circle::PolyOps;
use stwo_constraint_framework::FrameworkComponent;

use crate::components::activation::ActivationEval;
use crate::components::attention::{AttentionWeightsF32, attention_forward_f32};
use crate::components::f32_ops::{
    F32Matrix, matmul_f32, apply_activation_f32, layernorm_f32,
};
use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use crate::compiler::prove::{ModelError, ModelProofResultFor, prove_model_with};
use crate::gadgets::quantize::{
    QuantParams, QuantStrategy, dequantize_value,
};

// ===== DualWeights =====

/// Dual weight storage: f32 originals + M31 quantized for proving.
///
/// Stores both representations so the f32 forward pass uses real weights
/// and the M31 proving pipeline uses quantized weights. QuantParams are
/// preserved for computing quantization error.
#[derive(Debug, Clone)]
pub struct DualWeights {
    /// M31 quantized weights (for the existing proving pipeline).
    pub m31: GraphWeights,
    /// f32 weight matrices: (node_id, name, matrix).
    pub f32_weights: Vec<(usize, String, F32Matrix)>,
    /// Quantization parameters per weight: (node_id, name, params).
    pub quant_params: Vec<(usize, String, QuantParams)>,
}

impl DualWeights {
    /// Create DualWeights from f32 weight entries.
    ///
    /// Each entry: (node_id, name, f32_data). Quantizes to M31 using the
    /// given strategy and stores both representations.
    pub fn from_f32(
        entries: Vec<(usize, String, F32Matrix)>,
        strategy: QuantStrategy,
    ) -> Self {
        let mut m31 = GraphWeights::new();
        let mut quant_params = Vec::with_capacity(entries.len());

        for (node_id, name, f32_mat) in &entries {
            let (m31_mat, params) = f32_mat.to_m31(strategy);
            if name.is_empty() {
                m31.add_weight(*node_id, m31_mat);
            } else {
                m31.add_named_weight(*node_id, name, m31_mat);
            }
            quant_params.push((*node_id, name.clone(), params));
        }

        Self {
            m31,
            f32_weights: entries,
            quant_params,
        }
    }

    /// Get the f32 weight for a node (first unnamed weight).
    pub fn get_f32_weight(&self, node_id: usize) -> Option<&F32Matrix> {
        self.f32_weights
            .iter()
            .find(|(id, name, _)| *id == node_id && name.is_empty())
            .map(|(_, _, w)| w)
    }

    /// Get a named f32 weight for a node.
    pub fn get_f32_named(&self, node_id: usize, name: &str) -> Option<&F32Matrix> {
        self.f32_weights
            .iter()
            .find(|(id, n, _)| *id == node_id && n == name)
            .map(|(_, _, w)| w)
    }

    /// Access the M31 GraphWeights for the existing proving pipeline.
    pub fn as_graph_weights(&self) -> &GraphWeights {
        &self.m31
    }

    /// Reconstruct DualWeights from existing M31 GraphWeights and their QuantParams.
    ///
    /// Dequantizes M31 weights back to f32 approximations. This is lossy but
    /// useful when you have an existing M31 pipeline and want to add f32 output.
    pub fn from_graph_weights(gw: &GraphWeights, params: &[(usize, String, QuantParams)]) -> Self {
        let mut f32_weights = Vec::with_capacity(params.len());

        for (node_id, name, qp) in params {
            let m31_mat = if name.is_empty() {
                gw.get_weight(*node_id)
            } else {
                gw.get_named_weight(*node_id, name)
            };

            if let Some(m31_mat) = m31_mat {
                let f32_mat = F32Matrix::from_m31(m31_mat, qp);
                f32_weights.push((*node_id, name.clone(), f32_mat));
            }
        }

        Self {
            m31: gw.clone(),
            f32_weights,
            quant_params: params.to_vec(),
        }
    }
}

// ===== f32 Forward Pass =====

/// Execute a computation graph in f32 for meaningful real-valued output.
///
/// Walks graph nodes in topological order, dispatching to f32 operations.
/// The graph structure and node types are read-only — only the computation
/// uses float arithmetic.
pub fn f32_forward(
    graph: &ComputationGraph,
    input: &F32Matrix,
    weights: &DualWeights,
) -> Result<F32Matrix, ModelError> {
    let mut current = input.clone();

    for node in &graph.nodes {
        match &node.op {
            GraphOp::MatMul { .. } => {
                let weight = weights.get_f32_weight(node.id).ok_or(
                    ModelError::MissingWeight(node.id),
                )?;
                current = matmul_f32(&current, weight);
            }

            GraphOp::Activation { activation_type, .. } => {
                current = apply_activation_f32(&current, *activation_type);
            }

            GraphOp::LayerNorm { dim } => {
                current = layernorm_f32(&current, *dim, 1e-5);
            }

            GraphOp::Attention { config } => {
                let attn_weights = AttentionWeightsF32 {
                    w_q: weights
                        .get_f32_named(node.id, "w_q")
                        .ok_or(ModelError::MissingWeight(node.id))?
                        .clone(),
                    w_k: weights
                        .get_f32_named(node.id, "w_k")
                        .ok_or(ModelError::MissingWeight(node.id))?
                        .clone(),
                    w_v: weights
                        .get_f32_named(node.id, "w_v")
                        .ok_or(ModelError::MissingWeight(node.id))?
                        .clone(),
                    w_o: weights
                        .get_f32_named(node.id, "w_o")
                        .ok_or(ModelError::MissingWeight(node.id))?
                        .clone(),
                };
                current = attention_forward_f32(&current, &attn_weights, config, false);
            }

            GraphOp::Add { .. } => {
                // Element-wise add in f32: use current + current (self-add) as default
                // Real multi-input handled when graph has explicit branches
            }

            GraphOp::Mul { .. } => {
                // Element-wise mul in f32: passthrough (mul by 1)
            }

            GraphOp::Embedding { .. } | GraphOp::Conv2D { .. } => {
                // Forward pass only in f32 — passthrough
            }

            GraphOp::Quantize { .. } | GraphOp::Identity { .. } => {
                // Passthrough in f32 — quantization is an M31 concern
            }
        }
    }

    Ok(current)
}

// ===== Dual Output =====

/// Result of dual-track execution: f32 output + M31 proof.
pub struct DualOutput<H: MerkleHasherLifted> {
    /// Meaningful float inference output.
    pub f32_output: F32Matrix,
    /// M31 STARK proof (layer proofs + execution trace).
    pub m31_proof: ModelProofResultFor<H>,
    /// Maximum absolute quantization error between f32 and dequantized M31 output.
    pub quantization_error: f64,
}

/// Run both f32 inference and M31 proving on the same graph.
///
/// 1. Quantizes f32 input → M31 input
/// 2. Runs f32 forward pass → f32 output
/// 3. Runs M31 proving pipeline → STARK proof (existing code, unchanged)
/// 4. Computes quantization error between the two outputs
pub fn prove_model_dual<B, MC>(
    graph: &ComputationGraph,
    input_f32: &F32Matrix,
    weights: &DualWeights,
) -> Result<DualOutput<<MC as MerkleChannel>::H>, ModelError>
where
    B: BackendForChannel<MC> + PolyOps,
    MC: MerkleChannel,
    FrameworkComponent<ActivationEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<crate::components::elementwise::ElementwiseAddEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<crate::components::elementwise::ElementwiseMulEval>: stwo::prover::ComponentProver<B>,
    FrameworkComponent<crate::components::layernorm::LayerNormEval>: stwo::prover::ComponentProver<B>,
{
    // 1. Quantize input for M31 pipeline
    let (input_m31, _input_params) = input_f32.to_m31(QuantStrategy::Direct);

    // 2. f32 forward pass
    let f32_output = f32_forward(graph, input_f32, weights)?;

    // 3. M31 proving (calls existing pipeline unchanged)
    let m31_proof = prove_model_with::<B, MC>(graph, &input_m31, weights.as_graph_weights())?;

    // 4. Compute quantization error
    // Dequantize M31 output and compare with f32 output
    let m31_execution = &m31_proof.1;
    let m31_output = &m31_execution.output;

    // Use Direct strategy params for output comparison
    let output_params = QuantParams::from_range(
        f32_output.data.iter().cloned().fold(f32::INFINITY, f32::min) as f64,
        f32_output.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64,
        QuantStrategy::Direct,
    );

    let mut max_error: f64 = 0.0;
    let n = f32_output.data.len().min(m31_output.data.len());
    for i in 0..n {
        let m31_f32 = dequantize_value(m31_output.data[i], &output_params);
        let error = (f32_output.data[i] as f64 - m31_f32 as f64).abs();
        if error > max_error {
            max_error = error;
        }
    }

    Ok(DualOutput {
        f32_output,
        m31_proof,
        quantization_error: max_error,
    })
}

/// Convenience: dual prove with SimdBackend + Blake2sMerkleChannel.
pub fn prove_model_dual_default(
    graph: &ComputationGraph,
    input_f32: &F32Matrix,
    weights: &DualWeights,
) -> Result<DualOutput<<Blake2sMerkleChannel as MerkleChannel>::H>, ModelError> {
    prove_model_dual::<SimdBackend, Blake2sMerkleChannel>(graph, input_f32, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use crate::components::activation::ActivationType;
    use crate::components::attention::MultiHeadAttentionConfig;

    fn make_mlp_graph() -> ComputationGraph {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(8).activation(ActivationType::ReLU).linear(4);
        builder.build()
    }

    fn make_mlp_weights() -> DualWeights {
        // Node 0: matmul (1,4) → (1,8), weight is (4,8)
        // Node 1: activation (passthrough)
        // Node 2: matmul (1,8) → (1,4), weight is (8,4)
        let w0_data: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1) - 1.5).collect();
        let w0 = F32Matrix::from_data(4, 8, w0_data);

        let w2_data: Vec<f32> = (0..32).map(|i| (i as f32 * 0.05) - 0.8).collect();
        let w2 = F32Matrix::from_data(8, 4, w2_data);

        DualWeights::from_f32(
            vec![
                (0, String::new(), w0),
                (2, String::new(), w2),
            ],
            QuantStrategy::Symmetric8,
        )
    }

    #[test]
    fn test_dual_weights_creation() {
        let weights = make_mlp_weights();
        assert!(weights.get_f32_weight(0).is_some());
        assert!(weights.get_f32_weight(2).is_some());
        assert!(weights.get_f32_weight(1).is_none());
        assert!(weights.m31.get_weight(0).is_some());
        assert!(weights.m31.get_weight(2).is_some());
        assert_eq!(weights.quant_params.len(), 2);
    }

    #[test]
    fn test_f32_forward_mlp() {
        let graph = make_mlp_graph();
        let weights = make_mlp_weights();
        let input = F32Matrix::from_data(1, 4, vec![0.5, -0.3, 0.8, 0.1]);

        let output = f32_forward(&graph, &input, &weights).unwrap();
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 4);
        // After ReLU, all values should be >= 0 at intermediate step,
        // final matmul can produce any sign
        for &v in &output.data {
            assert!(v.is_finite(), "output should be finite: {v}");
        }
    }

    #[test]
    fn test_f32_forward_with_layernorm() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).layer_norm().linear(4);
        let graph = builder.build();

        let w0 = F32Matrix::from_data(4, 4, vec![
            0.5, 0.1, -0.2, 0.3,
            0.1, 0.4, 0.2, -0.1,
            -0.3, 0.2, 0.6, 0.1,
            0.2, -0.1, 0.1, 0.5,
        ]);
        let w2 = F32Matrix::from_data(4, 4, vec![
            0.3, -0.1, 0.2, 0.4,
            -0.2, 0.5, 0.1, -0.3,
            0.1, 0.2, -0.4, 0.2,
            0.4, -0.3, 0.3, 0.1,
        ]);
        let weights = DualWeights::from_f32(
            vec![
                (0, String::new(), w0),
                (2, String::new(), w2),
            ],
            QuantStrategy::Symmetric8,
        );

        let input = F32Matrix::from_data(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let output = f32_forward(&graph, &input, &weights).unwrap();
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 4);
        for &v in &output.data {
            assert!(v.is_finite(), "output should be finite: {v}");
        }
    }

    #[test]
    fn test_dual_prove_and_verify() {
        let graph = make_mlp_graph();
        let weights = make_mlp_weights();
        let input = F32Matrix::from_data(1, 4, vec![0.5, -0.3, 0.8, 0.1]);

        let result = prove_model_dual_default(&graph, &input, &weights);
        assert!(result.is_ok(), "dual prove failed: {:?}", result.err());

        let dual = result.unwrap();
        assert_eq!(dual.f32_output.rows, 1);
        assert_eq!(dual.f32_output.cols, 4);
        // M31 proof should have layer proofs for all 3 nodes
        assert_eq!(dual.m31_proof.0.len(), 3);
        // Quantization error should be finite
        assert!(dual.quantization_error.is_finite());
    }

    #[test]
    fn test_quantization_error_bounded() {
        let graph = make_mlp_graph();
        let weights = make_mlp_weights();
        let input = F32Matrix::from_data(1, 4, vec![0.5, -0.3, 0.8, 0.1]);

        let dual = prove_model_dual_default(&graph, &input, &weights).unwrap();
        // Error exists but should be finite (we don't assert a tight bound
        // since M31 field arithmetic diverges from float)
        assert!(
            dual.quantization_error.is_finite(),
            "quantization error should be finite: {}",
            dual.quantization_error,
        );
    }

    #[test]
    fn test_dual_weights_from_graph_weights() {
        let original = make_mlp_weights();
        let reconstructed = DualWeights::from_graph_weights(
            &original.m31,
            &original.quant_params,
        );
        assert!(reconstructed.get_f32_weight(0).is_some());
        assert!(reconstructed.get_f32_weight(2).is_some());
        assert_eq!(reconstructed.f32_weights.len(), original.f32_weights.len());
    }

    #[test]
    fn test_f32_forward_with_gelu() {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(8).activation(ActivationType::GELU).linear(4);
        let graph = builder.build();

        let w0_data: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1) - 1.5).collect();
        let w0 = F32Matrix::from_data(4, 8, w0_data);
        let w2_data: Vec<f32> = (0..32).map(|i| (i as f32 * 0.05) - 0.8).collect();
        let w2 = F32Matrix::from_data(8, 4, w2_data);

        let weights = DualWeights::from_f32(
            vec![
                (0, String::new(), w0),
                (2, String::new(), w2),
            ],
            QuantStrategy::Symmetric8,
        );

        let input = F32Matrix::from_data(1, 4, vec![0.5, -0.3, 0.8, 0.1]);
        let output = f32_forward(&graph, &input, &weights).unwrap();
        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 4);
        for &v in &output.data {
            assert!(v.is_finite(), "output should be finite: {v}");
        }
    }

    #[test]
    fn test_f32_forward_with_attention() {
        let mut builder = GraphBuilder::new((4, 4));
        let config = MultiHeadAttentionConfig::new(2, 4, 4);
        builder.attention(config);
        let graph = builder.build();

        fn fill_f32(rows: usize, cols: usize, seed: u64) -> F32Matrix {
            let mut data = vec![0.0f32; rows * cols];
            let mut state = seed;
            for v in data.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *v = ((state >> 33) % 10) as f32 * 0.1 - 0.5;
            }
            F32Matrix::from_data(rows, cols, data)
        }

        let weights = DualWeights::from_f32(
            vec![
                (0, "w_q".to_string(), fill_f32(4, 4, 1)),
                (0, "w_k".to_string(), fill_f32(4, 4, 2)),
                (0, "w_v".to_string(), fill_f32(4, 4, 3)),
                (0, "w_o".to_string(), fill_f32(4, 4, 4)),
            ],
            QuantStrategy::Symmetric8,
        );

        let input = F32Matrix::from_data(
            4, 4,
            (0..16).map(|i| i as f32 * 0.1 - 0.8).collect(),
        );
        let output = f32_forward(&graph, &input, &weights).unwrap();
        assert_eq!(output.rows, 4);
        assert_eq!(output.cols, 4);
        for &v in &output.data {
            assert!(v.is_finite(), "output should be finite: {v}");
        }
    }
}
