//! ONNX model import.
//!
//! Parses ONNX protobuf files and extracts layer definitions,
//! weight tensors, and activation functions for circuit compilation.
//!
//! # Op Mapping
//!
//! | ONNX Op | GraphOp |
//! |---------|---------|
//! | `MatMul`, `Gemm` | `GraphOp::MatMul` |
//! | `Relu`, `Gelu`, `Sigmoid` | `GraphOp::Activation` |
//! | `LayerNormalization` | `GraphOp::LayerNorm` |
//! | `DequantizeLinear` | `GraphOp::Quantize` |

use std::path::Path;

use crate::compiler::graph::{ComputationGraph, GraphWeights, GraphBuilder};
use crate::components::activation::ActivationType;

/// Metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name from ONNX graph.
    pub name: String,
    /// Number of parameters.
    pub num_parameters: usize,
    /// Input shape: (batch, features).
    pub input_shape: (usize, usize),
    /// Output shape: (batch, features).
    pub output_shape: (usize, usize),
    /// Number of layers.
    pub num_layers: usize,
}

/// A fully loaded ONNX model ready for proving.
pub struct OnnxModel {
    pub graph: ComputationGraph,
    pub weights: GraphWeights,
    pub input_shape: (usize, usize),
    pub metadata: ModelMetadata,
}

/// Error type for ONNX loading.
#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Parse error: {0}")]
    ParseError(String),
    #[error("Unsupported op: {0}")]
    UnsupportedOp(String),
    #[error("Shape inference error: {0}")]
    ShapeError(String),
    #[error("Weight extraction error: {0}")]
    WeightError(String),
}

/// Load an ONNX model from a file path.
///
/// Uses tract-onnx for full ONNX protobuf parsing. Extracts the graph
/// structure AND weight tensors, quantizing them to M31.
#[cfg(feature = "onnx")]
pub fn load_onnx(path: &Path) -> Result<OnnxModel, OnnxError> {
    use tract_onnx::prelude::*;
    use crate::compiler::quantize_weights::quantize_weight_matrix;
    use crate::gadgets::quantize::QuantStrategy;

    let model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| OnnxError::ParseError(e.to_string()))?
        .into_optimized()
        .map_err(|e| OnnxError::ParseError(e.to_string()))?
        .into_runnable()
        .map_err(|e| OnnxError::ParseError(e.to_string()))?;

    // Extract input shape
    let input_fact = model.model().input_fact(0)
        .map_err(|e| OnnxError::ShapeError(e.to_string()))?;
    let input_shape = extract_shape_typed(input_fact);

    let mut builder = GraphBuilder::new(input_shape);
    let mut weights = GraphWeights::new();
    let mut num_parameters = 0usize;

    // Walk the optimized model's nodes and map ops to GraphOp variants.
    // tract's optimized model may fuse/rewrite ops, so we match on the
    // actual op type after optimization.
    for (node_idx, node) in model.model().nodes.iter().enumerate() {
        let op_name = node.op().name();

        match op_name.as_ref() {
            "MatMatMul" | "MatMul" | "Gemm" | "EinSum" => {
                let out_shape = get_node_output_shape(node);
                let out_features = out_shape.1;
                // Capture in_features BEFORE adding the linear node
                let in_features = builder.current_output_shape().1;
                builder.linear(out_features);
                let graph_node_id = builder.current_node_id();

                // Extract weight tensor from the second input (B matrix)
                if let Some(weight_data) = extract_const_input(model.model(), node_idx, 1) {
                    let (matrix, _params) = quantize_weight_matrix(
                        &weight_data,
                        in_features,
                        out_features,
                        QuantStrategy::Direct,
                    );
                    num_parameters += weight_data.len();
                    weights.add_weight(graph_node_id, matrix);
                }
            }
            "Relu" => {
                builder.activation(ActivationType::ReLU);
            }
            "Gelu" | "FastGelu" => {
                builder.activation(ActivationType::GELU);
            }
            "Sigmoid" => {
                builder.activation(ActivationType::Sigmoid);
            }
            "Softmax" => {
                builder.activation(ActivationType::Softmax);
            }
            "LayerNormalization" | "RmsNormalization" => {
                builder.layer_norm();
            }
            // Skip infrastructure ops that don't map to computation
            "Source" | "Const" | "Reshape" | "Transpose" | "Flatten"
            | "Squeeze" | "Unsqueeze" | "Cast" | "Gather" | "Concat"
            | "Shape" | "Slice" | "Identity" | "Dropout" | "Add" | "Mul" => {
                // These are structural ops, not computation ops
            }
            _ => {
                tracing::warn!("Skipping unsupported ONNX op: {}", op_name);
            }
        }
    }

    let graph = builder.build();
    let metadata = ModelMetadata {
        name: model.model().properties.get("name")
            .and_then(|v| v.to_scalar::<String>().ok().cloned())
            .unwrap_or_else(|| "unnamed".to_string()),
        num_parameters,
        input_shape,
        output_shape: graph.output_shape,
        num_layers: graph.num_layers(),
    };

    Ok(OnnxModel {
        graph,
        weights,
        input_shape,
        metadata,
    })
}

/// Extract f32 data from a constant input to a node.
#[cfg(feature = "onnx")]
fn extract_const_input(
    model: &tract_onnx::prelude::TypedModel,
    node_idx: usize,
    input_slot: usize,
) -> Option<Vec<f32>> {
    let node = &model.nodes[node_idx];
    let input = node.inputs.get(input_slot)?;
    let source_node = &model.nodes[input.node];

    // Check if the source node is a Const op
    if source_node.op().name() != "Const" {
        return None;
    }

    // Extract the tensor from the Const op
    let const_op = source_node.op_as::<tract_onnx::tract_core::ops::konst::Const>()?;
    let tensor = &const_op.0;

    // Convert to f32
    tensor.as_slice::<f32>().ok().map(|s| s.to_vec())
}

/// Get output shape of a node from its output facts.
#[cfg(feature = "onnx")]
fn get_node_output_shape(node: &tract_onnx::prelude::TypedNode) -> (usize, usize) {
    let fact = node.outputs.first()
        .map(|o| &o.fact);
    if let Some(f) = fact {
        let shape = f.shape.as_concrete().unwrap_or_default();
        match shape.len() {
            0 => (1, 1),
            1 => (1, shape[0]),
            _ => (shape[shape.len() - 2], shape[shape.len() - 1]),
        }
    } else {
        (1, 1)
    }
}

/// Extract shape from a TypedFact.
#[cfg(feature = "onnx")]
fn extract_shape_typed(fact: &tract_onnx::prelude::TypedFact) -> (usize, usize) {
    let shape = fact.shape.as_concrete().unwrap_or_default();
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        _ => (shape[0], shape[shape.len() - 1]),
    }
}

/// When `onnx` feature is not enabled, return an error.
#[cfg(not(feature = "onnx"))]
pub fn load_onnx(_path: &Path) -> Result<OnnxModel, OnnxError> {
    Err(OnnxError::ParseError(
        "ONNX loading requires the 'onnx' feature flag".to_string(),
    ))
}

/// Build a simple MLP model programmatically (no ONNX required).
///
/// Useful for testing and for models defined in code.
pub fn build_mlp(
    input_dim: usize,
    hidden_dims: &[usize],
    output_dim: usize,
    activation: ActivationType,
) -> OnnxModel {
    let mut builder = GraphBuilder::new((1, input_dim));

    for &dim in hidden_dims {
        builder.linear(dim);
        builder.activation(activation);
    }
    builder.linear(output_dim);

    let graph = builder.build();
    let metadata = ModelMetadata {
        name: "mlp".to_string(),
        num_parameters: 0,
        input_shape: (1, input_dim),
        output_shape: graph.output_shape,
        num_layers: graph.num_layers(),
    };

    OnnxModel {
        graph,
        weights: GraphWeights::new(),
        input_shape: (1, input_dim),
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_mlp() {
        let model = build_mlp(784, &[128, 64], 10, ActivationType::ReLU);
        assert_eq!(model.input_shape, (1, 784));
        assert_eq!(model.graph.output_shape, (1, 10));
        assert_eq!(model.graph.num_layers(), 5); // linear + relu + linear + relu + linear
        assert_eq!(model.metadata.name, "mlp");
    }

    #[test]
    fn test_onnx_not_available_without_feature() {
        #[cfg(not(feature = "onnx"))]
        {
            let result = load_onnx(Path::new("nonexistent.onnx"));
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_build_mlp_with_weights_and_prove() {
        use stwo::core::fields::m31::M31;
        use crate::compiler::prove::prove_model;

        let model = build_mlp(4, &[4], 2, ActivationType::ReLU);

        // Manually supply weights (build_mlp doesn't generate them)
        let mut weights = GraphWeights::new();

        let mut w0 = crate::components::matmul::M31Matrix::new(4, 4);
        for i in 0..4 { for j in 0..4 { w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32)); } }
        weights.add_weight(0, w0);

        let mut w2 = crate::components::matmul::M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w2.set(i, j, M31::from((i + j + 1) as u32)); } }
        weights.add_weight(2, w2);

        let mut input = crate::components::matmul::M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let result = prove_model(&model.graph, &input, &weights);
        assert!(result.is_ok(), "MLP from build_mlp should prove: {:?}", result.err());
    }
}
