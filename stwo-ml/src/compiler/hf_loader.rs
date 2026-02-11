//! HuggingFace model directory loader with mandatory validation.
//!
//! Loads a transformer model from a HuggingFace-format directory containing:
//! - `config.json`: Model architecture config
//! - `*.safetensors`: Weight shard files
//! - `model.safetensors.index.json`: Shard index (for multi-shard models)
//!
//! **All validation checks must pass before the model is returned.**
//! Proofs over a broken or incomplete model are meaningless.
//!
//! Builds a [`ComputationGraph`] from config.json and loads weights from SafeTensors.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::compiler::graph::{ComputationGraph, GraphBuilder, GraphOp, GraphWeights};
use crate::compiler::onnx::{OnnxModel, OnnxError, ModelMetadata, TransformerConfig};
use crate::compiler::quantize_weights::quantize_weight_matrix;
use crate::compiler::safetensors::{discover_shards, list_tensors_sharded, tensor_to_f32};
use crate::components::activation::ActivationType;
use crate::gadgets::quantize::QuantStrategy;

// ─────────────────────────────────────────────────────────────────────────────
// Model Validation
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a single validation check.
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Full validation report for a model directory.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub model_dir: PathBuf,
    pub checks: Vec<ValidationCheck>,
}

impl ValidationReport {
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|c| c.passed)
    }

    pub fn num_passed(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    pub fn num_failed(&self) -> usize {
        self.checks.iter().filter(|c| !c.passed).count()
    }

    /// Format as a human-readable report for terminal output.
    pub fn format_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("  Model directory: {}", self.model_dir.display()));
        lines.push(String::new());

        for check in &self.checks {
            let icon = if check.passed { "✓" } else { "✗" };
            let color = if check.passed { "\x1b[0;32m" } else { "\x1b[0;31m" };
            let reset = "\x1b[0m";
            if check.detail.is_empty() {
                lines.push(format!("  {color}{icon}{reset} {}", check.name));
            } else {
                lines.push(format!("  {color}{icon}{reset} {}  ({})", check.name, check.detail));
            }
        }

        lines.push(String::new());
        lines.push(format!("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"));
        let total = self.checks.len();
        let passed = self.num_passed();
        let failed = self.num_failed();
        if failed > 0 {
            lines.push(format!(
                "  Checks: {passed}/{total} passed, \x1b[0;31m{failed} FAILED\x1b[0m"
            ));
            lines.push(String::new());
            lines.push("  Proofs over an incomplete model are meaningless.".to_string());
            lines.push("  Fix the issues above before attempting to prove.".to_string());
        } else {
            lines.push(format!(
                "  Checks: {passed}/{total} passed — \x1b[0;32mALL PASSED\x1b[0m"
            ));
        }
        lines.push(format!("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"));

        lines.join("\n")
    }
}

/// Validate a model directory end-to-end.
///
/// Checks performed:
/// 1. Directory exists
/// 2. config.json exists and is parseable
/// 3. Architecture parameters are valid (hidden_size > 0, layers > 0)
/// 4. SafeTensors weight files exist
/// 5. Shard index exists (for multi-shard models)
/// 6. Weight files are readable (binary header parses)
/// 7. Required weight tensors exist for the requested layers
/// 8. Weight dimensions match expected graph dimensions
pub fn validate_model_directory(
    model_dir: &Path,
    num_layers: Option<usize>,
) -> ValidationReport {
    let mut checks = Vec::new();

    // Check 1: Directory exists
    checks.push(ValidationCheck {
        name: "Model directory exists".into(),
        passed: model_dir.is_dir(),
        detail: model_dir.display().to_string(),
    });

    if !model_dir.is_dir() {
        return ValidationReport { model_dir: model_dir.to_path_buf(), checks };
    }

    // Check 2: config.json exists
    let config_path = model_dir.join("config.json");
    checks.push(ValidationCheck {
        name: "config.json present".into(),
        passed: config_path.is_file(),
        detail: if config_path.is_file() { "found".into() } else { "MISSING".into() },
    });

    if !config_path.is_file() {
        return ValidationReport { model_dir: model_dir.to_path_buf(), checks };
    }

    // Check 3: config.json parseable
    let hf_config = match HfConfig::from_file(&config_path) {
        Ok(c) => {
            checks.push(ValidationCheck {
                name: "config.json parseable".into(),
                passed: true,
                detail: format!(
                    "{}: d={}, heads={}, ff={}, layers={}, vocab={}",
                    c.model_type, c.hidden_size, c.num_attention_heads,
                    c.intermediate_size, c.num_hidden_layers, c.vocab_size,
                ),
            });
            Some(c)
        }
        Err(e) => {
            checks.push(ValidationCheck {
                name: "config.json parseable".into(),
                passed: false,
                detail: e.to_string(),
            });
            None
        }
    };

    // Check 4: Architecture parameters valid
    if let Some(ref cfg) = hf_config {
        checks.push(ValidationCheck {
            name: "Architecture parameters valid".into(),
            passed: cfg.hidden_size > 0 && cfg.num_hidden_layers > 0 && cfg.num_attention_heads > 0,
            detail: if cfg.hidden_size == 0 {
                "hidden_size is 0".into()
            } else if cfg.num_hidden_layers == 0 {
                "num_hidden_layers is 0".into()
            } else {
                format!("hidden_size={}, layers={}", cfg.hidden_size, cfg.num_hidden_layers)
            },
        });
    }

    // Check 5: SafeTensors weight files exist
    let shard_paths = discover_shards(model_dir, "model").unwrap_or_default();
    // Also try without "model" filter (some models use different naming)
    let shard_paths = if shard_paths.is_empty() {
        discover_shards(model_dir, "").unwrap_or_default()
            .into_iter()
            .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
            .collect()
    } else {
        shard_paths
    };

    let total_weight_bytes: u64 = shard_paths.iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();

    checks.push(ValidationCheck {
        name: "SafeTensors weight files present".into(),
        passed: !shard_paths.is_empty(),
        detail: format!("{} shards, {:.1} GB", shard_paths.len(), total_weight_bytes as f64 / 1e9),
    });

    // Check 6: Shard index (for multi-shard models)
    if shard_paths.len() > 1 {
        let has_index = model_dir.join("model.safetensors.index.json").is_file();
        checks.push(ValidationCheck {
            name: "Shard index file present".into(),
            passed: has_index,
            detail: if has_index {
                "model.safetensors.index.json".into()
            } else {
                "MISSING — multi-shard model needs index file".into()
            },
        });
    }

    // Check 7: Weight shards are readable
    if !shard_paths.is_empty() {
        match validate_shard_headers(&shard_paths) {
            Ok(total_tensors) => {
                checks.push(ValidationCheck {
                    name: "Weight shards readable".into(),
                    passed: true,
                    detail: format!("{} tensors across {} shards", total_tensors, shard_paths.len()),
                });
            }
            Err(e) => {
                checks.push(ValidationCheck {
                    name: "Weight shards readable".into(),
                    passed: false,
                    detail: e,
                });
            }
        }
    }

    // Check 8: Required weight tensors exist for requested layers
    if let Some(ref cfg) = hf_config {
        let layers = num_layers.unwrap_or(cfg.num_hidden_layers);
        let layers = if layers == 0 { cfg.num_hidden_layers } else { layers };

        if let Ok(all_tensors) = list_tensors_sharded(&shard_paths) {
            let transformer_config = cfg.to_transformer_config();
            let graph = build_hf_transformer_graph(&transformer_config, layers);
            let name_map = build_weight_name_map(&graph, layers, &all_tensors);

            let matmul_count = graph.nodes.iter()
                .filter(|n| matches!(n.op, GraphOp::MatMul { .. }))
                .count();
            let mapped_count = name_map.len();

            checks.push(ValidationCheck {
                name: "Required weight tensors found".into(),
                passed: mapped_count == matmul_count,
                detail: format!("{}/{} MatMul weights mapped", mapped_count, matmul_count),
            });

            // Check 9: List which specific weights are missing
            if mapped_count < matmul_count {
                let missing: Vec<String> = graph.nodes.iter().enumerate()
                    .filter(|(_, n)| matches!(n.op, GraphOp::MatMul { .. }))
                    .filter(|(idx, _)| !name_map.contains_key(idx))
                    .map(|(idx, _)| format!("node {} (MatMul)", idx))
                    .collect();
                checks.push(ValidationCheck {
                    name: "Missing weights".into(),
                    passed: false,
                    detail: missing.join(", "),
                });
            }

            // Check 10: Verify weight dimensions match
            if mapped_count > 0 {
                match validate_weight_dimensions(&shard_paths, &graph, &name_map) {
                    Ok(()) => {
                        checks.push(ValidationCheck {
                            name: "Weight dimensions match graph".into(),
                            passed: true,
                            detail: format!("all {} weights have correct shapes", mapped_count),
                        });
                    }
                    Err(mismatches) => {
                        checks.push(ValidationCheck {
                            name: "Weight dimensions match graph".into(),
                            passed: false,
                            detail: mismatches,
                        });
                    }
                }
            }
        }
    }

    ValidationReport { model_dir: model_dir.to_path_buf(), checks }
}

/// Validate that SafeTensors shard files can be opened and headers parsed.
fn validate_shard_headers(shard_paths: &[PathBuf]) -> Result<usize, String> {
    let mut total_tensors = 0;
    for path in shard_paths {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("Cannot mmap {}: {}", path.display(), e))?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("Cannot parse {}: {}", path.display(), e))?;
        total_tensors += tensors.names().len();
    }
    Ok(total_tensors)
}

/// Validate that weight tensor dimensions match the expected graph dimensions.
fn validate_weight_dimensions(
    shard_paths: &[PathBuf],
    graph: &ComputationGraph,
    name_map: &HashMap<usize, String>,
) -> Result<(), String> {
    let mut shard_data: Vec<memmap2::Mmap> = Vec::new();
    for path in shard_paths {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
        shard_data.push(mmap);
    }

    let mut mismatches = Vec::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            if let Some(tensor_name) = name_map.get(&idx) {
                for mmap in &shard_data {
                    let tensors = safetensors::SafeTensors::deserialize(mmap)
                        .map_err(|e| e.to_string())?;
                    if let Ok(tensor) = tensors.tensor(tensor_name) {
                        let shape = tensor.shape();
                        if shape.len() == 2 {
                            let (rows, cols) = (shape[0], shape[1]);
                            // HF stores (out, in) or (in, out) — either orientation is valid
                            let ok = (rows == *k && cols == *n)
                                || (rows == *n && cols == *k);
                            if !ok {
                                mismatches.push(format!(
                                    "{}: shape ({}, {}) does not match expected ({}, {})",
                                    tensor_name, rows, cols, k, n
                                ));
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(mismatches.join("; "))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config Parsing
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Model Loading (with mandatory validation)
// ─────────────────────────────────────────────────────────────────────────────

/// Load a model from a HuggingFace directory.
///
/// **Performs full validation before returning.** If any critical check fails
/// (missing config, missing weights, dimension mismatch), this returns an error.
/// Proofs over an incomplete model are refused.
///
/// # Arguments
/// * `model_dir` - Path to the model directory
/// * `num_layers` - Number of transformer layers to load (None = all from config)
///
/// Returns an `OnnxModel` with the graph and loaded weights.
pub fn load_hf_model(
    model_dir: &Path,
    num_layers: Option<usize>,
) -> Result<OnnxModel, OnnxError> {
    // ── Step 1: Run full validation ──
    let report = validate_model_directory(model_dir, num_layers);

    // Print the validation report
    eprintln!();
    eprintln!("  ── Model Validation ──");
    eprintln!("{}", report.format_report());
    eprintln!();

    // Refuse to load if validation failed
    if !report.passed() {
        return Err(OnnxError::WeightError(format!(
            "Model validation failed: {}/{} checks passed. \
             Fix the issues above before proving. \
             Proofs over an incomplete model are meaningless.",
            report.num_passed(),
            report.checks.len(),
        )));
    }

    // ── Step 2: Parse config ──
    let config_path = model_dir.join("config.json");
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

    // ── Step 3: Build computation graph ──
    let graph = build_hf_transformer_graph(&transformer_config, layers);

    // ── Step 4: Load weights (guaranteed to exist by validation) ──
    let shard_paths = discover_shards(model_dir, "model")
        .map_err(|e| OnnxError::WeightError(format!("Cannot discover shards: {e}")))?;

    if shard_paths.is_empty() {
        return Err(OnnxError::WeightError(format!(
            "No SafeTensors weight files found in {}. \
             Download the model first: \
             huggingface-cli download <model-id> --local-dir {}",
            model_dir.display(),
            model_dir.display(),
        )));
    }

    eprintln!("  Loading weights from {} shards...", shard_paths.len());

    let all_tensor_names: Vec<(String, usize)> = list_tensors_sharded(&shard_paths)
        .map_err(|e| OnnxError::WeightError(format!("Cannot list tensors: {e}")))?;
    eprintln!("  Total tensors across shards: {}", all_tensor_names.len());

    let name_map = build_weight_name_map(&graph, layers, &all_tensor_names);
    eprintln!("  Weight name mapping: {} entries", name_map.len());
    for (idx, name) in &name_map {
        eprintln!("    node {} → {}", idx, name);
    }

    let weights = load_weights_from_shards(
        &shard_paths, &graph, &name_map, QuantStrategy::Symmetric8,
    ).map_err(|e| OnnxError::WeightError(format!("Cannot load weights: {e}")))?;

    // ── Step 5: Verify all MatMul nodes have weights ──
    let matmul_nodes: Vec<usize> = graph.nodes.iter().enumerate()
        .filter(|(_, n)| matches!(n.op, GraphOp::MatMul { .. }))
        .map(|(idx, _)| idx)
        .collect();

    let mut missing_weights = Vec::new();
    for &idx in &matmul_nodes {
        if weights.get_weight(idx).is_none() {
            missing_weights.push(idx);
        }
    }

    if !missing_weights.is_empty() {
        return Err(OnnxError::WeightError(format!(
            "Weight not found for {} MatMul node(s): {:?}. \
             All weights must be present before proving. \
             Check that the model directory has all SafeTensors shards.",
            missing_weights.len(),
            missing_weights,
        )));
    }

    let loaded_count = matmul_nodes.len() - missing_weights.len();
    eprintln!("  Loaded weights for {}/{} MatMul layers ✓", loaded_count, matmul_nodes.len());

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

// ─────────────────────────────────────────────────────────────────────────────
// Graph Construction
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Weight Name Mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Build a mapping from graph node indices to HuggingFace tensor names.
///
/// For each transformer block, the graph has 7 nodes:
///   0: LayerNorm (pre-attention)
///   1: MatMul    (Q projection)    → model.layers.{L}.self_attn.q_proj.weight
///   2: MatMul    (O projection)    → model.layers.{L}.self_attn.o_proj.weight
///   3: LayerNorm (post-attention)
///   4: MatMul    (FFN up)          → model.layers.{L}.mlp.up_proj.weight (or gate_proj)
///   5: Activation
///   6: MatMul    (FFN down)        → model.layers.{L}.mlp.down_proj.weight
///
/// Plus a final LayerNorm at the end.
fn build_weight_name_map(
    graph: &ComputationGraph,
    num_layers: usize,
    available_tensors: &[(String, usize)],
) -> HashMap<usize, String> {
    let mut map = HashMap::new();
    let tensor_set: std::collections::HashSet<&str> = available_tensors
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    // Each block has 7 nodes. MatMul nodes are at offsets 1, 2, 4, 6 within a block.
    let nodes_per_block = 7;

    for layer_idx in 0..num_layers {
        let block_start = layer_idx * nodes_per_block;

        // Node offset 1: Q projection
        let q_node = block_start + 1;
        // Node offset 2: O projection
        let o_node = block_start + 2;
        // Node offset 4: FFN up
        let up_node = block_start + 4;
        // Node offset 6: FFN down
        let down_node = block_start + 6;

        // Try common HuggingFace naming patterns for the Q projection
        let q_candidates = [
            format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
            format!("model.layers.{layer_idx}.attention.wq.weight"),
            format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
        ];
        for name in &q_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(q_node, name.clone());
                break;
            }
        }

        // O projection
        let o_candidates = [
            format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
            format!("model.layers.{layer_idx}.attention.wo.weight"),
            format!("transformer.h.{layer_idx}.attn.o_proj.weight"),
        ];
        for name in &o_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(o_node, name.clone());
                break;
            }
        }

        // FFN up projection (some models use gate_proj, some use up_proj)
        let up_candidates = [
            format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
            format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
            format!("model.layers.{layer_idx}.feed_forward.w1.weight"),
            format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
        ];
        for name in &up_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(up_node, name.clone());
                break;
            }
        }

        // FFN down projection
        let down_candidates = [
            format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
            format!("model.layers.{layer_idx}.feed_forward.w2.weight"),
            format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
        ];
        for name in &down_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(down_node, name.clone());
                break;
            }
        }
    }

    map
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight Loading
// ─────────────────────────────────────────────────────────────────────────────

/// Load weights from multiple SafeTensors shards using an explicit name mapping.
fn load_weights_from_shards(
    shard_paths: &[std::path::PathBuf],
    graph: &ComputationGraph,
    name_map: &HashMap<usize, String>,
    strategy: QuantStrategy,
) -> Result<GraphWeights, crate::compiler::quantize_weights::WeightError> {
    use crate::compiler::quantize_weights::WeightError;

    // Memory-map all shards
    let mut shard_data: Vec<(std::fs::File, memmap2::Mmap)> = Vec::new();
    for path in shard_paths {
        let file = std::fs::File::open(path)
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        shard_data.push((file, mmap));
    }

    let mut weights = GraphWeights::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            if let Some(tensor_name) = name_map.get(&idx) {
                // Search through shards for this tensor
                let mut found = false;
                for (_file, mmap) in &shard_data {
                    let tensors = safetensors::SafeTensors::deserialize(mmap)
                        .map_err(|e| WeightError::IoError(e.to_string()))?;

                    if let Ok(tensor) = tensors.tensor(tensor_name) {
                        let data = tensor_to_f32(tensor.data(), tensor.dtype());
                        // HuggingFace stores weights as (out_features, in_features)
                        // Our MatMul expects (k, n) = (in_features, out_features)
                        // So we may need to transpose
                        let shape = tensor.shape();
                        let (weight_data, wk, wn) = if shape.len() == 2 {
                            let rows = shape[0]; // out_features
                            let cols = shape[1]; // in_features
                            if rows == *n && cols == *k {
                                // Already (out, in) — transpose to (in, out) = (k, n)
                                let mut transposed = vec![0.0f32; data.len()];
                                for r in 0..rows {
                                    for c in 0..cols {
                                        transposed[c * rows + r] = data[r * cols + c];
                                    }
                                }
                                (transposed, *k, *n)
                            } else if rows == *k && cols == *n {
                                // Already (k, n)
                                (data, *k, *n)
                            } else {
                                eprintln!(
                                    "    WARNING: tensor {} shape ({}, {}) doesn't match expected ({}, {}), using as-is",
                                    tensor_name, rows, cols, k, n
                                );
                                (data, *k, *n)
                            }
                        } else {
                            (data, *k, *n)
                        };

                        let (matrix, _params) = quantize_weight_matrix(
                            &weight_data, wk, wn, strategy,
                        );
                        weights.add_weight(idx, matrix);
                        found = true;
                        break;
                    }
                }
                if !found {
                    eprintln!(
                        "    ERROR: tensor '{}' not found in any shard for node {}",
                        tensor_name, idx
                    );
                }
            }
        }
    }

    Ok(weights)
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

    #[test]
    fn test_validation_missing_dir() {
        let report = validate_model_directory(Path::new("/nonexistent/dir"), Some(1));
        assert!(!report.passed());
        assert_eq!(report.num_failed(), 1);
    }

    #[test]
    fn test_validation_empty_dir() {
        let tmp = std::env::temp_dir().join("stwo_ml_empty_model");
        std::fs::create_dir_all(&tmp).unwrap();

        let report = validate_model_directory(&tmp, Some(1));
        assert!(!report.passed());

        std::fs::remove_dir_all(&tmp).ok();
    }
}
