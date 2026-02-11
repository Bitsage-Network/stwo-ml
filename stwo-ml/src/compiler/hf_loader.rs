//! HuggingFace model directory loader.
//!
//! Loads a transformer model from a HuggingFace-format directory containing:
//! - `config.json`: Model architecture config
//! - `*.safetensors`: Weight shard files
//!
//! Builds a [`ComputationGraph`] from config.json and loads weights from SafeTensors.

use std::path::Path;

use crate::compiler::graph::{ComputationGraph, GraphBuilder};
use crate::compiler::onnx::{OnnxModel, OnnxError, ModelMetadata, TransformerConfig};
use crate::components::activation::ActivationType;
use crate::compiler::safetensors::{discover_shards, load_weights_sharded, list_tensors_sharded};
use crate::gadgets::quantize::QuantStrategy;

/// Parsed HuggingFace config.json.
#[derive(Debug, Clone)]
pub struct HfConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
}

impl HfConfig {
    /// Parse a HuggingFace config.json file.
    pub fn from_file(path: &Path) -> Result<Self, OnnxError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| OnnxError::IoError(format!("Cannot read config.json: {e}")))?;

        let json: serde_json::Value = serde_json::from_str(&contents)
            .map_err(|e| OnnxError::ParseError(format!("Invalid config.json: {e}")))?;

        Ok(Self {
            model_type: json["model_type"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
            hidden_size: json["hidden_size"]
                .as_u64()
                .ok_or_else(|| OnnxError::ParseError("missing hidden_size".into()))?
                as usize,
            num_attention_heads: json["num_attention_heads"]
                .as_u64()
                .ok_or_else(|| OnnxError::ParseError("missing num_attention_heads".into()))?
                as usize,
            num_key_value_heads: json["num_key_value_heads"]
                .as_u64()
                .unwrap_or(json["num_attention_heads"].as_u64().unwrap_or(1))
                as usize,
            intermediate_size: json["intermediate_size"]
                .as_u64()
                .ok_or_else(|| OnnxError::ParseError("missing intermediate_size".into()))?
                as usize,
            num_hidden_layers: json["num_hidden_layers"]
                .as_u64()
                .ok_or_else(|| OnnxError::ParseError("missing num_hidden_layers".into()))?
                as usize,
            vocab_size: json["vocab_size"]
                .as_u64()
                .unwrap_or(32000)
                as usize,
            hidden_act: json["hidden_act"]
                .as_str()
                .or_else(|| json["hidden_activation"].as_str())
                .unwrap_or("silu")
                .to_string(),
            max_position_embeddings: json["max_position_embeddings"]
                .as_u64()
                .unwrap_or(2048)
                as usize,
        })
    }

    /// Convert to internal TransformerConfig.
    pub fn to_transformer_config(&self) -> TransformerConfig {
        let activation = match self.hidden_act.as_str() {
            "gelu" | "gelu_new" | "gelu_fast" => ActivationType::GELU,
            "relu" => ActivationType::ReLU,
            "silu" | "swiglu" => ActivationType::GELU, // Map SiLU to GELU for now
            _ => ActivationType::GELU,
        };

        TransformerConfig {
            d_model: self.hidden_size,
            num_heads: self.num_attention_heads,
            d_ff: self.intermediate_size,
            activation,
        }
    }
}

/// Load a model from a HuggingFace directory.
///
/// The directory should contain `config.json` and `*.safetensors` files.
///
/// # Arguments
/// * `model_dir` - Path to the model directory
/// * `num_layers` - Number of transformer layers to load (use 0 or config value for all)
///
/// Returns an `OnnxModel` with the graph and loaded weights.
pub fn load_hf_model(
    model_dir: &Path,
    num_layers: Option<usize>,
) -> Result<OnnxModel, OnnxError> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Err(OnnxError::IoError(format!(
            "config.json not found in {}",
            model_dir.display()
        )));
    }

    // Parse config
    let hf_config = HfConfig::from_file(&config_path)?;
    let transformer_config = hf_config.to_transformer_config();

    let layers = num_layers.unwrap_or(hf_config.num_hidden_layers);
    let layers = if layers == 0 { hf_config.num_hidden_layers } else { layers };

    eprintln!("Model: {} ({})", hf_config.model_type, model_dir.display());
    eprintln!(
        "  hidden_size={}, heads={}, ff={}, layers={}/{}",
        hf_config.hidden_size,
        hf_config.num_attention_heads,
        hf_config.intermediate_size,
        layers,
        hf_config.num_hidden_layers,
    );

    // Build computation graph
    let graph = build_hf_transformer_graph(&transformer_config, layers);

    // Discover and load SafeTensors weights
    let shard_paths = discover_shards(model_dir, "model")
        .map_err(|e| OnnxError::WeightError(format!("Cannot discover shards: {e}")))?;

    if shard_paths.is_empty() {
        eprintln!(
            "  WARNING: No SafeTensors shard files found in {}. Using auto-generated weights.",
            model_dir.display()
        );
        // Fall back to auto-generated weights for testing
        let weights = crate::compiler::onnx::generate_weights_for_graph(&graph, 42);
        let num_parameters = crate::compiler::onnx::count_matmul_params(&graph);

        let metadata = ModelMetadata {
            name: format!("{}_{}L", hf_config.model_type, layers),
            num_parameters,
            input_shape: graph.input_shape,
            output_shape: graph.output_shape,
            num_layers: graph.num_layers(),
        };

        return Ok(OnnxModel {
            input_shape: graph.input_shape,
            graph,
            weights,
            metadata,
        });
    }

    eprintln!("  Loading weights from {} shards...", shard_paths.len());

    // List tensors for diagnostics
    if let Ok(all_tensors) = list_tensors_sharded(&shard_paths) {
        eprintln!("  Total tensors across shards: {}", all_tensors.len());
    }

    let weights = load_weights_sharded(&shard_paths, &graph, QuantStrategy::Symmetric8)
        .map_err(|e| OnnxError::WeightError(format!("Cannot load weights: {e}")))?;

    let loaded_count = graph.nodes.iter().enumerate()
        .filter(|(idx, _)| weights.get_weight(*idx).is_some())
        .count();

    eprintln!(
        "  Loaded weights for {}/{} MatMul layers",
        loaded_count,
        graph.nodes.iter().filter(|n| matches!(n.op, crate::compiler::graph::GraphOp::MatMul { .. })).count()
    );

    let num_parameters = crate::compiler::onnx::count_matmul_params(&graph);

    let metadata = ModelMetadata {
        name: format!("{}_{}L", hf_config.model_type, layers),
        num_parameters,
        input_shape: graph.input_shape,
        output_shape: graph.output_shape,
        num_layers: graph.num_layers(),
    };

    Ok(OnnxModel {
        input_shape: graph.input_shape,
        graph,
        weights,
        metadata,
    })
}

/// Build a transformer computation graph matching a HuggingFace architecture.
///
/// Each transformer block: LayerNorm → Q proj → O proj → LayerNorm → FFN up → act → FFN down
fn build_hf_transformer_graph(
    config: &TransformerConfig,
    num_layers: usize,
) -> ComputationGraph {
    let d = config.d_model;
    let d_ff = config.d_ff;

    let mut builder = GraphBuilder::new((1, d));

    for _ in 0..num_layers {
        // Pre-attention LayerNorm
        builder.layer_norm();
        // Q projection (d → d)
        builder.linear(d);
        // O projection (d → d)
        builder.linear(d);
        // Post-attention LayerNorm
        builder.layer_norm();
        // FFN up projection (d → d_ff)
        builder.linear(d_ff);
        // FFN activation
        builder.activation(config.activation);
        // FFN down projection (d_ff → d)
        builder.linear(d);
    }

    // Final LayerNorm
    builder.layer_norm();

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_config_parse() {
        let json = r#"{
            "model_type": "qwen3",
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "intermediate_size": 13824,
            "num_hidden_layers": 40,
            "vocab_size": 152064,
            "hidden_act": "silu",
            "max_position_embeddings": 40960
        }"#;

        let tmp = std::env::temp_dir().join("test_config.json");
        std::fs::write(&tmp, json).unwrap();

        let config = HfConfig::from_file(&tmp).unwrap();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_attention_heads, 40);
        assert_eq!(config.intermediate_size, 13824);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.model_type, "qwen3");

        let tc = config.to_transformer_config();
        assert_eq!(tc.d_model, 5120);
        assert_eq!(tc.num_heads, 40);
        assert_eq!(tc.d_ff, 13824);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_build_hf_transformer_graph() {
        let config = TransformerConfig {
            d_model: 8,
            num_heads: 2,
            d_ff: 16,
            activation: ActivationType::GELU,
        };

        let graph = build_hf_transformer_graph(&config, 2);

        // 2 blocks × 7 ops + 1 final LN = 15
        assert_eq!(graph.num_layers(), 15);
        assert_eq!(graph.input_shape, (1, 8));
        assert_eq!(graph.output_shape, (1, 8));
    }
}
