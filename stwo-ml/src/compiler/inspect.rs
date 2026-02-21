//! Model inspection and summarization.
//!
//! Provides structured summaries of computation graphs including
//! per-layer breakdown, parameter counts, and memory estimates.

use std::fmt;

use crate::backend::estimate_proof_memory;
use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};

/// Summary of a single layer in the model.
#[derive(Debug, Clone)]
pub struct LayerSummary {
    /// Layer index.
    pub index: usize,
    /// Operation name (e.g. "MatMul", "Activation", "LayerNorm").
    pub op_name: String,
    /// Output shape: (rows, cols).
    pub output_shape: (usize, usize),
    /// Number of weight parameters in this layer.
    pub num_parameters: usize,
    /// Estimated trace rows for proving.
    pub trace_rows: usize,
    /// Whether this layer has an associated weight matrix.
    pub has_weight: bool,
}

/// Summary of an entire model.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// Model name.
    pub name: String,
    /// Input shape.
    pub input_shape: (usize, usize),
    /// Output shape.
    pub output_shape: (usize, usize),
    /// Per-layer summaries.
    pub layers: Vec<LayerSummary>,
    /// Total weight parameters.
    pub total_parameters: usize,
    /// Total estimated trace rows.
    pub total_trace_rows: usize,
    /// Estimated weight memory in bytes (M31 = 4 bytes each).
    pub weight_memory_bytes: usize,
    /// Estimated proof memory in bytes.
    pub proof_memory_bytes: usize,
    /// Count of MatMul layers.
    pub num_matmul_layers: usize,
    /// Count of Activation layers.
    pub num_activation_layers: usize,
    /// Count of LayerNorm layers.
    pub num_layernorm_layers: usize,
    /// Count of other layers (Identity, Quantize).
    pub num_other_layers: usize,
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model: {}", self.name)?;
        writeln!(
            f,
            "Input:  {:?}  Output: {:?}",
            self.input_shape, self.output_shape
        )?;
        writeln!(f, "{:-<72}", "")?;
        writeln!(
            f,
            "{:<6} {:<16} {:<14} {:<10} {:<10} {:<6}",
            "Layer", "Op", "Shape", "Params", "TraceRows", "Weight"
        )?;
        writeln!(f, "{:-<72}", "")?;

        for layer in &self.layers {
            writeln!(
                f,
                "{:<6} {:<16} {:?}{:<4} {:<10} {:<10} {:<6}",
                layer.index,
                layer.op_name,
                layer.output_shape,
                "",
                layer.num_parameters,
                layer.trace_rows,
                if layer.has_weight { "yes" } else { "-" },
            )?;
        }

        writeln!(f, "{:-<72}", "")?;
        writeln!(
            f,
            "Layers: {} total ({} MatMul, {} Activation, {} LayerNorm, {} Other)",
            self.layers.len(),
            self.num_matmul_layers,
            self.num_activation_layers,
            self.num_layernorm_layers,
            self.num_other_layers,
        )?;
        writeln!(f, "Parameters: {}", self.total_parameters)?;
        writeln!(f, "Trace rows:  {}", self.total_trace_rows)?;
        writeln!(
            f,
            "Weight memory: {:.2} KB",
            self.weight_memory_bytes as f64 / 1024.0
        )?;
        writeln!(
            f,
            "Proof memory:  {:.2} KB",
            self.proof_memory_bytes as f64 / 1024.0
        )?;

        Ok(())
    }
}

/// Summarize a computation graph with optional weights.
pub fn summarize_graph(graph: &ComputationGraph, weights: &GraphWeights) -> ModelSummary {
    let mut layers = Vec::new();
    let mut total_parameters = 0usize;
    let mut num_matmul = 0usize;
    let mut num_activation = 0usize;
    let mut num_layernorm = 0usize;
    let mut num_other = 0usize;

    for node in &graph.nodes {
        let (op_name, num_params) = match &node.op {
            GraphOp::MatMul { dims: (_m, k, n) } => {
                num_matmul += 1;
                ("MatMul".to_string(), k * n)
            }
            GraphOp::Activation {
                activation_type, ..
            } => {
                num_activation += 1;
                (format!("Activation({activation_type:?})"), 0)
            }
            GraphOp::LayerNorm { .. } => {
                num_layernorm += 1;
                ("LayerNorm".to_string(), 0)
            }
            GraphOp::RMSNorm { .. } => {
                num_layernorm += 1;
                ("RMSNorm".to_string(), 0)
            }
            GraphOp::Quantize { .. } => {
                num_other += 1;
                ("Quantize".to_string(), 0)
            }
            GraphOp::Attention { config } => {
                num_other += 1;
                let d = config.d_model;
                // 4 weight matrices: W_Q, W_K, W_V, W_O each (d_model, d_model)
                (format!("Attention({}h)", config.num_heads), 4 * d * d)
            }
            GraphOp::Add { .. } => {
                num_other += 1;
                ("Add".to_string(), 0)
            }
            GraphOp::Mul { .. } => {
                num_other += 1;
                ("Mul".to_string(), 0)
            }
            GraphOp::Embedding {
                vocab_size,
                embed_dim,
            } => {
                num_other += 1;
                ("Embedding".to_string(), vocab_size * embed_dim)
            }
            GraphOp::Conv2D {
                in_channels,
                out_channels,
                kernel_size,
                ..
            } => {
                num_other += 1;
                (
                    "Conv2D".to_string(),
                    in_channels * out_channels * kernel_size * kernel_size,
                )
            }
            GraphOp::Dequantize { .. } => {
                num_other += 1;
                ("Dequantize".to_string(), 0)
            }
            GraphOp::RoPE { config } => {
                num_other += 1;
                (format!("RoPE(d={})", config.head_dim), 0)
            }
            GraphOp::Identity { .. } => {
                num_other += 1;
                ("Identity".to_string(), 0)
            }
        };

        let has_weight = weights.get_weight(node.id).is_some();
        total_parameters += num_params;

        layers.push(LayerSummary {
            index: node.id,
            op_name,
            output_shape: node.output_shape,
            num_parameters: num_params,
            trace_rows: node.op.trace_rows(),
            has_weight,
        });
    }

    let total_trace_rows = graph.total_trace_rows();
    let weight_memory_bytes = total_parameters * 4; // M31 = 4 bytes

    // Estimate proof memory: use log2 of total trace rows, assume ~5 columns per layer
    let log_size = if total_trace_rows > 0 {
        (total_trace_rows as f64).log2().ceil() as u32
    } else {
        4
    };
    let num_columns = graph.num_layers() * 5;
    let proof_memory_bytes = estimate_proof_memory(log_size.max(4), num_columns);

    ModelSummary {
        name: String::new(),
        input_shape: graph.input_shape,
        output_shape: graph.output_shape,
        layers,
        total_parameters,
        total_trace_rows,
        weight_memory_bytes,
        proof_memory_bytes,
        num_matmul_layers: num_matmul,
        num_activation_layers: num_activation,
        num_layernorm_layers: num_layernorm,
        num_other_layers: num_other,
    }
}

/// Summarize an OnnxModel.
pub fn summarize_model(model: &super::onnx::OnnxModel) -> ModelSummary {
    let mut summary = summarize_graph(&model.graph, &model.weights);
    summary.name = model.metadata.name.clone();
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::onnx::{build_mlp, build_mlp_with_weights};
    use crate::components::activation::ActivationType;

    #[test]
    fn test_summarize_mlp() {
        let model = build_mlp_with_weights(4, &[8, 4], 2, ActivationType::ReLU, 42);
        let summary = summarize_model(&model);

        assert_eq!(summary.num_matmul_layers, 3);
        assert_eq!(summary.num_activation_layers, 2);
        assert_eq!(summary.total_parameters, 72); // 4*8 + 8*4 + 4*2
        assert_eq!(summary.layers.len(), 5);

        // All MatMul layers should have weights
        for layer in &summary.layers {
            if layer.op_name == "MatMul" {
                assert!(
                    layer.has_weight,
                    "MatMul layer {} should have weight",
                    layer.index
                );
            }
        }

        // Display output should contain "MatMul"
        let display = format!("{summary}");
        assert!(
            display.contains("MatMul"),
            "Display should contain 'MatMul'"
        );
        assert!(
            display.contains("Parameters:"),
            "Display should contain 'Parameters:'"
        );
    }

    #[test]
    fn test_summarize_graph_without_weights() {
        let model = build_mlp(4, &[8], 2, ActivationType::ReLU);
        let summary = summarize_graph(&model.graph, &model.weights);

        // No weights in build_mlp
        for layer in &summary.layers {
            assert!(
                !layer.has_weight,
                "Layer {} should not have weight",
                layer.index
            );
        }

        assert_eq!(summary.num_matmul_layers, 2);
        assert_eq!(summary.total_parameters, 4 * 8 + 8 * 2);
    }
}
