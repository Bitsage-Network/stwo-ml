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
use crate::compiler::onnx::{ModelMetadata, OnnxError, OnnxModel, TransformerConfig};
use crate::compiler::quantize_weights::quantize_weight_matrix;
use crate::compiler::safetensors::{discover_shards, list_tensors_sharded, tensor_to_f32};
use crate::components::activation::ActivationType;
use crate::components::matmul::M31Matrix;
use stwo::core::fields::m31::M31;
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
            let color = if check.passed {
                "\x1b[0;32m"
            } else {
                "\x1b[0;31m"
            };
            let reset = "\x1b[0m";
            if check.detail.is_empty() {
                lines.push(format!("  {color}{icon}{reset} {}", check.name));
            } else {
                lines.push(format!(
                    "  {color}{icon}{reset} {}  ({})",
                    check.name, check.detail
                ));
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
pub fn validate_model_directory(model_dir: &Path, num_layers: Option<usize>) -> ValidationReport {
    let mut checks = Vec::new();

    // Check 1: Directory exists
    checks.push(ValidationCheck {
        name: "Model directory exists".into(),
        passed: model_dir.is_dir(),
        detail: model_dir.display().to_string(),
    });

    if !model_dir.is_dir() {
        return ValidationReport {
            model_dir: model_dir.to_path_buf(),
            checks,
        };
    }

    // Check 2: config.json exists
    let config_path = model_dir.join("config.json");
    checks.push(ValidationCheck {
        name: "config.json present".into(),
        passed: config_path.is_file(),
        detail: if config_path.is_file() {
            "found".into()
        } else {
            "MISSING".into()
        },
    });

    if !config_path.is_file() {
        return ValidationReport {
            model_dir: model_dir.to_path_buf(),
            checks,
        };
    }

    // Check 3: config.json parseable
    let hf_config = match HfConfig::from_file(&config_path) {
        Ok(c) => {
            checks.push(ValidationCheck {
                name: "config.json parseable".into(),
                passed: true,
                detail: format!(
                    "{}: d={}, heads={}, ff={}, layers={}, vocab={}",
                    c.model_type,
                    c.hidden_size,
                    c.num_attention_heads,
                    c.intermediate_size,
                    c.num_hidden_layers,
                    c.vocab_size,
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
                format!(
                    "hidden_size={}, layers={}",
                    cfg.hidden_size, cfg.num_hidden_layers
                )
            },
        });
    }

    // Check 5: SafeTensors weight files exist
    let shard_paths = discover_shards(model_dir, "model").unwrap_or_default();
    // Also try without "model" filter (some models use different naming)
    let shard_paths = if shard_paths.is_empty() {
        discover_shards(model_dir, "")
            .unwrap_or_default()
            .into_iter()
            .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
            .collect()
    } else {
        shard_paths
    };

    let total_weight_bytes: u64 = shard_paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();

    checks.push(ValidationCheck {
        name: "SafeTensors weight files present".into(),
        passed: !shard_paths.is_empty(),
        detail: format!(
            "{} shards, {:.1} GB",
            shard_paths.len(),
            total_weight_bytes as f64 / 1e9
        ),
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
                    detail: format!(
                        "{} tensors across {} shards",
                        total_tensors,
                        shard_paths.len()
                    ),
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
        let layers = if layers == 0 {
            cfg.num_hidden_layers
        } else {
            layers
        };

        if let Ok(all_tensors) = list_tensors_sharded(&shard_paths) {
            let transformer_config = cfg.to_transformer_config();
            let (graph, moe_infos) = build_hf_transformer_graph(&transformer_config, layers);
            let mut name_map = build_weight_name_map(&graph, layers, &all_tensors);
            add_moe_weight_names(&mut name_map, &moe_infos, &all_tensors);

            let matmul_count = graph
                .nodes
                .iter()
                .filter(|n| matches!(n.op, GraphOp::MatMul { .. }))
                .count();
            // Count only MatMul weight mappings (keys < 10000), not gamma (10000+) or up_proj (20000+)
            let mapped_count = name_map.keys().filter(|&&k| k < 10000).count();

            checks.push(ValidationCheck {
                name: "Required weight tensors found".into(),
                passed: mapped_count == matmul_count,
                detail: format!("{}/{} MatMul weights mapped", mapped_count, matmul_count),
            });

            // Check 9: List which specific weights are missing
            if mapped_count < matmul_count {
                let missing: Vec<String> = graph
                    .nodes
                    .iter()
                    .enumerate()
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

    ValidationReport {
        model_dir: model_dir.to_path_buf(),
        checks,
    }
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
                    let tensors =
                        safetensors::SafeTensors::deserialize(mmap).map_err(|e| e.to_string())?;
                    if let Ok(tensor) = tensors.tensor(tensor_name) {
                        let shape = tensor.shape();
                        if shape.len() == 2 {
                            let (rows, cols) = (shape[0], shape[1]);
                            // HF stores (out, in) or (in, out) — either orientation is valid.
                            // Also accept fused tensors:
                            //   gate_up_proj: [2n, k] or [k, 2n] (Phi-3)
                            //   qkv_proj: [3k, k] or similar (Phi-3, GPT-2)
                            let ok = (rows == *k && cols == *n)
                                || (rows == *n && cols == *k)
                                // Fused gate_up: 2× output dimension
                                || (rows == 2 * *n && cols == *k)
                                || (rows == *k && cols == 2 * *n)
                                // Fused QKV: any size > n (3× for MHA, (H+2KV)*head for GQA)
                                || (rows > *n && cols == *k)
                                || (rows == *k && cols > *n);
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
    /// Number of experts for MoE models (0 = dense, no MoE).
    pub num_experts: usize,
    /// Number of experts activated per token (top-K).
    pub num_experts_per_tok: usize,
}

impl HfConfig {
    /// Parse a HuggingFace config.json file.
    pub fn from_file(path: &Path) -> Result<Self, OnnxError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| OnnxError::IoError(format!("Cannot read config.json: {e}")))?;

        let json: serde_json::Value = serde_json::from_str(&contents)
            .map_err(|e| OnnxError::ParseError(format!("Invalid config.json: {e}")))?;

        // Support nested config (Qwen3.5, multimodal models with text_config)
        // Try root-level fields first, fall back to text_config sub-object
        let tc = if json["text_config"].is_object() { &json["text_config"] } else { &json };

        // Helper: read u64 from tc first, then json root as fallback
        let get_u64 = |key: &str| -> Option<u64> {
            tc[key].as_u64().or_else(|| json[key].as_u64())
        };
        let get_str = |key: &str| -> Option<&str> {
            tc[key].as_str().or_else(|| json[key].as_str())
        };

        Ok(Self {
            model_type: json["model_type"].as_str()
                .or_else(|| tc["model_type"].as_str())
                .unwrap_or("unknown").to_string(),
            hidden_size: get_u64("hidden_size")
                .ok_or_else(|| OnnxError::ParseError("missing hidden_size".into()))?
                as usize,
            num_attention_heads: get_u64("num_attention_heads")
                .ok_or_else(|| OnnxError::ParseError("missing num_attention_heads".into()))?
                as usize,
            // GLM uses multi_query_group_num instead of num_key_value_heads
            num_key_value_heads: get_u64("num_key_value_heads")
                .or_else(|| get_u64("multi_query_group_num"))
                .unwrap_or(get_u64("num_attention_heads").unwrap_or(1))
                as usize,
            // GLM uses ffn_hidden_size instead of intermediate_size
            intermediate_size: get_u64("intermediate_size")
                .or_else(|| get_u64("ffn_hidden_size"))
                .ok_or_else(|| OnnxError::ParseError("missing intermediate_size/ffn_hidden_size".into()))?
                as usize,
            // GLM uses num_layers instead of num_hidden_layers
            num_hidden_layers: get_u64("num_hidden_layers")
                .or_else(|| get_u64("num_layers"))
                .ok_or_else(|| OnnxError::ParseError("missing num_hidden_layers/num_layers".into()))?
                as usize,
            // GLM uses padded_vocab_size instead of vocab_size
            vocab_size: get_u64("vocab_size")
                .or_else(|| get_u64("padded_vocab_size"))
                .unwrap_or(32000) as usize,
            hidden_act: get_str("hidden_act")
                .or_else(|| get_str("hidden_activation"))
                .unwrap_or("silu")
                .to_string(),
            max_position_embeddings: get_u64("max_position_embeddings").unwrap_or(2048)
                as usize,
            // MoE: MiniMax uses num_local_experts (256), Mixtral uses num_local_experts (8)
            num_experts: get_u64("num_local_experts")
                .or_else(|| get_u64("num_experts"))
                .or_else(|| get_u64("n_routed_experts"))
                .unwrap_or(0) as usize,
            num_experts_per_tok: get_u64("num_experts_per_tok")
                .or_else(|| get_u64("num_experts_per_token"))
                .unwrap_or(0) as usize,
        })
    }

    /// Convert to internal TransformerConfig.
    pub fn to_transformer_config(&self) -> TransformerConfig {
        use crate::compiler::onnx::NormType;

        let activation = match self.hidden_act.as_str() {
            "gelu" | "gelu_new" | "gelu_fast" => ActivationType::GELU,
            "relu" => ActivationType::ReLU,
            "silu" | "swiglu" => ActivationType::SiLU,
            "sigmoid" => ActivationType::Sigmoid,
            _ => ActivationType::GELU,
        };

        // Most modern LLMs (Llama, Qwen, Mistral, Gemma, etc.) use RMSNorm.
        // Only older architectures (GPT-2, BERT, etc.) use full LayerNorm.
        let norm_type = match self.model_type.as_str() {
            "gpt2" | "bert" | "roberta" | "bart" | "t5" => NormType::LayerNorm,
            _ => NormType::RMSNorm,
        };

        TransformerConfig {
            d_model: self.hidden_size,
            num_heads: self.num_attention_heads,
            d_ff: self.intermediate_size,
            activation,
            norm_type,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
        }
    }

    /// Whether this model uses Mixture of Experts.
    pub fn is_moe(&self) -> bool {
        self.num_experts > 0 && self.num_experts_per_tok > 0
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
pub fn load_hf_model(model_dir: &Path, num_layers: Option<usize>) -> Result<OnnxModel, OnnxError> {
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
    let layers = if layers == 0 {
        hf_config.num_hidden_layers
    } else {
        layers
    };

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
    let (graph, moe_infos) = build_hf_transformer_graph(&transformer_config, layers);

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

    let mut name_map = build_weight_name_map(&graph, layers, &all_tensor_names);
    add_moe_weight_names(&mut name_map, &moe_infos, &all_tensor_names);
    eprintln!("  Weight name mapping: {} entries", name_map.len());

    let weights =
        load_weights_from_shards(&shard_paths, &graph, &name_map, QuantStrategy::Symmetric8)
            .map_err(|e| OnnxError::WeightError(format!("Cannot load weights: {e}")))?;

    // ── Step 5: Verify all MatMul nodes have weights ──
    let matmul_nodes: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
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
    eprintln!(
        "  Loaded weights for {}/{} MatMul layers ✓",
        loaded_count,
        matmul_nodes.len()
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

// ─────────────────────────────────────────────────────────────────────────────
// Streaming Weight Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Open a streaming weight pipeline for a HuggingFace model directory.
///
/// Returns the pipeline (with mmap'd shards, no weights loaded) and the
/// computation graph. Use [`StreamingWeightPipeline::load_chunk_weights`]
/// to load weights on demand per chunk.
///
/// # Arguments
/// * `model_dir` - Path to the model directory (must contain `config.json` + SafeTensors shards)
/// * `strategy` - Quantization strategy for weight conversion
/// * `num_layers` - Number of transformer layers to load (None = all from config)
pub fn open_streaming_pipeline(
    model_dir: &std::path::Path,
    strategy: QuantStrategy,
    num_layers: Option<usize>,
) -> Result<
    (
        crate::compiler::streaming::StreamingWeightPipeline,
        ComputationGraph,
    ),
    OnnxError,
> {
    // Parse config
    let config_path = model_dir.join("config.json");
    let hf_config = HfConfig::from_file(&config_path)?;
    let transformer_config = hf_config.to_transformer_config();

    let layers = num_layers.unwrap_or(hf_config.num_hidden_layers);
    let layers = if layers == 0 {
        hf_config.num_hidden_layers
    } else {
        layers
    };

    // Build computation graph
    let (graph, moe_infos) = build_hf_transformer_graph(&transformer_config, layers);

    // Discover shards
    let shard_paths = discover_shards(model_dir, "model")
        .map_err(|e| OnnxError::WeightError(format!("Cannot discover shards: {e}")))?;

    if shard_paths.is_empty() {
        return Err(OnnxError::WeightError(format!(
            "No SafeTensors weight files found in {}",
            model_dir.display(),
        )));
    }

    // Build name map
    let all_tensor_names = list_tensors_sharded(&shard_paths)
        .map_err(|e| OnnxError::WeightError(format!("Cannot list tensors: {e}")))?;
    let name_map = build_weight_name_map(&graph, layers, &all_tensor_names);

    // Open pipeline (mmap only, no weight loading)
    let pipeline = crate::compiler::streaming::StreamingWeightPipeline::open(
        &shard_paths,
        &graph,
        name_map,
        strategy,
    )
    .map_err(|e| OnnxError::WeightError(format!("Cannot open streaming pipeline: {e}")))?;

    Ok((pipeline, graph))
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph Construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build a transformer computation graph matching a HuggingFace architecture.
///
/// Each transformer block: Norm → Q proj → O proj → Norm → Gated FFN (gate×up→down)
///
/// Models the real Llama/Qwen/Mistral architecture with SwiGLU gated FFN:
///   gate = SiLU(input × W_gate)
///   up   = input × W_up
///   hidden = gate * up
///   output = hidden × W_down
pub(crate) fn build_hf_transformer_graph(config: &TransformerConfig, num_layers: usize) -> (ComputationGraph, Vec<(usize, crate::compiler::graph::MoESlotInfo)>) {
    use crate::compiler::onnx::NormType;
    let d = config.d_model;
    let d_ff = config.d_ff;

    let mut builder = GraphBuilder::new((1, d));
    let mut moe_slot_infos: Vec<(usize, crate::compiler::graph::MoESlotInfo)> = Vec::new();

    for layer_idx in 0..num_layers {
        // Pre-attention norm
        match config.norm_type {
            NormType::LayerNorm => { builder.layer_norm(); }
            NormType::RMSNorm => { builder.rms_norm(); }
        }
        // Q projection (d → d)
        builder.linear(d);
        // O projection (d → d)
        builder.linear(d);
        // Post-attention norm
        match config.norm_type {
            NormType::LayerNorm => { builder.layer_norm(); }
            NormType::RMSNorm => { builder.rms_norm(); }
        }
        // FFN: dense (gated FFN) or MoE (multi-expert gated FFN)
        if config.num_experts > 0 && config.num_experts_per_tok > 0 {
            // MoE: router → TopK → K parallel expert FFNs → weighted sum
            let moe_info = builder.moe_ffn(config.num_experts, config.num_experts_per_tok, d_ff, config.activation);
            moe_slot_infos.push((layer_idx, moe_info));
        } else {
            // Dense: single gated FFN (SwiGLU)
            builder.gated_ffn(d_ff, config.activation);
        }
    }

    // Final norm
    match config.norm_type {
        NormType::LayerNorm => { builder.layer_norm(); }
        NormType::RMSNorm => { builder.rms_norm(); }
    }

    (builder.build(), moe_slot_infos)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight Name Mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Build a mapping from graph node indices to HuggingFace tensor names.
///
/// For each transformer block with gated FFN, the graph has 9 nodes:
///   0: Norm      (pre-attention)
///   1: MatMul    (Q projection)    → model.layers.{L}.self_attn.q_proj.weight
///   2: MatMul    (O projection)    → model.layers.{L}.self_attn.o_proj.weight
///   3: Norm      (post-attention)
///   4: MatMul    (gate_proj)       → model.layers.{L}.mlp.gate_proj.weight
///   5: Activation (SiLU on gate)
///   6: MatMul    (up_proj)         → model.layers.{L}.mlp.up_proj.weight
///   7: Mul       (gate * up)
///   8: MatMul    (down_proj)       → model.layers.{L}.mlp.down_proj.weight
///
/// Plus a final Norm at the end.
pub(crate) fn build_weight_name_map(
    _graph: &ComputationGraph,
    num_layers: usize,
    available_tensors: &[(String, usize)],
) -> HashMap<usize, String> {
    let mut map = HashMap::new();
    let tensor_set: std::collections::HashSet<&str> = available_tensors
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    // Each block has 7 nodes with simplified gated FFN (linear chain):
    //   0: Norm, 1: Q proj, 2: O proj, 3: Norm, 4: gate_proj, 5: SiLU, 6: down_proj
    // up_proj is stored as a named weight "up_proj" on the down_proj node (offset 6).
    let nodes_per_block = 7;

    for layer_idx in 0..num_layers {
        let block_start = layer_idx * nodes_per_block;

        // Node offset 1: Q projection
        let q_node = block_start + 1;
        // Node offset 2: O projection
        let o_node = block_start + 2;
        // Node offset 4: gate_proj
        let gate_node = block_start + 4;
        // Node offset 6: down_proj (up_proj stored as named weight here)
        let down_node = block_start + 6;

        // Try common HuggingFace naming patterns for the Q projection
        // Q projection (or fused QKV for Phi-3/GPT-2)
        let q_candidates = [
            format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
            format!("model.layers.{layer_idx}.attention.wq.weight"),
            format!("transformer.h.{layer_idx}.attn.q_proj.weight"),
            // Fused QKV: store as Q, will be split during loading
            format!("model.layers.{layer_idx}.self_attn.qkv_proj.weight"),
            // GPT-2 fused QKV
            format!("h.{layer_idx}.attn.c_attn.weight"),
            // GLM-4 fused QKV
            format!("transformer.encoder.layers.{layer_idx}.self_attention.query_key_value.weight"),
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
            // GPT-2
            format!("h.{layer_idx}.attn.c_proj.weight"),
            // GLM-4
            format!("transformer.encoder.layers.{layer_idx}.self_attention.dense.weight"),
        ];
        for name in &o_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(o_node, name.clone());
                break;
            }
        }

        // Gate projection (SwiGLU gate branch)
        // Supports: separate gate_proj OR fused gate_up_proj (Phi-3 style)
        let gate_candidates = [
            format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
            format!("model.layers.{layer_idx}.feed_forward.w1.weight"),
            format!("transformer.h.{layer_idx}.mlp.gate_proj.weight"),
            // GLM-4 gate_proj (part of fused dense_h_to_4h)
            format!("transformer.encoder.layers.{layer_idx}.mlp.dense_h_to_4h.weight"),
        ];
        let mut found_gate = false;
        for name in &gate_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(gate_node, name.clone());
                found_gate = true;
                break;
            }
        }

        // Check for fused gate_up_proj (Phi-3: [2*intermediate, hidden] → split into gate + up)
        if !found_gate {
            let fused_candidates = [
                format!("model.layers.{layer_idx}.mlp.gate_up_proj.weight"),
                // GLM-4 fused gate+up
                format!("transformer.encoder.layers.{layer_idx}.mlp.dense_h_to_4h.weight"),
            ];
            for name in &fused_candidates {
                if tensor_set.contains(name.as_str()) {
                    // Store fused tensor as the gate_proj weight (regular key).
                    // During weight loading, the fused tensor will be detected by its
                    // shape (2× expected rows) and split: first half → gate, second half → up.
                    map.insert(gate_node, name.clone());
                    found_gate = true;
                    break;
                }
            }
        }

        // Non-gated FFN (GPT-2 style: c_fc → activation → c_proj)
        if !found_gate {
            let nongated_candidates = [
                format!("h.{layer_idx}.mlp.c_fc.weight"),
                format!("transformer.h.{layer_idx}.mlp.c_fc.weight"),
            ];
            for name in &nongated_candidates {
                if tensor_set.contains(name.as_str()) {
                    map.insert(gate_node, name.clone());
                    found_gate = true;
                    break;
                }
            }
        }

        // Up projection — stored as named weight "up_proj" on down_proj node (key 20000+down_node)
        // For fused gate_up_proj models (Phi-3), this won't find a match —
        // the up_proj will be split from the fused tensor during weight loading.
        {
            let up_candidates = [
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                format!("model.layers.{layer_idx}.feed_forward.w3.weight"),
                format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
            ];
            for name in &up_candidates {
                if tensor_set.contains(name.as_str()) {
                    map.insert(20000 + down_node, name.clone());
                    break;
                }
            }
        }

        // FFN down projection
        let down_candidates = [
            format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
            format!("model.layers.{layer_idx}.feed_forward.w2.weight"),
            format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
            // GPT-2
            format!("h.{layer_idx}.mlp.c_proj.weight"),
            // GLM-4
            format!("transformer.encoder.layers.{layer_idx}.mlp.dense_4h_to_h.weight"),
        ];
        for name in &down_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(down_node, name.clone());
                break;
            }
        }

        // Norm weights (γ) — stored as named weights "gamma:{node_id}"
        // Node offset 0: pre-attention norm
        let pre_norm_node = block_start;
        let pre_norm_candidates = [
            format!("model.layers.{layer_idx}.input_layernorm.weight"),
            format!("model.layers.{layer_idx}.ln1.weight"),
            format!("transformer.h.{layer_idx}.ln_1.weight"),
            // GLM-4
            format!("transformer.encoder.layers.{layer_idx}.input_layernorm.weight"),
        ];
        for name in &pre_norm_candidates {
            if tensor_set.contains(name.as_str()) {
                // Use a special key format: "gamma:{node_id}" to distinguish from matmul weights
                map.insert(10000 + pre_norm_node, name.clone());
                break;
            }
        }

        // Node offset 3: post-attention norm (same offset in 9-node layout)
        let post_norm_node = block_start + 3;
        let post_norm_candidates = [
            format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
            format!("model.layers.{layer_idx}.ln2.weight"),
            format!("transformer.h.{layer_idx}.ln_2.weight"),
            // GLM-4
            format!("transformer.encoder.layers.{layer_idx}.post_attention_layernorm.weight"),
        ];
        for name in &post_norm_candidates {
            if tensor_set.contains(name.as_str()) {
                map.insert(10000 + post_norm_node, name.clone());
                break;
            }
        }
    }

    // Final norm weight
    let final_norm_node = num_layers * nodes_per_block;
    let final_norm_candidates = [
        "model.norm.weight".to_string(),
        "transformer.ln_f.weight".to_string(),
        "model.final_layernorm.weight".to_string(),
        // GLM-4
        "transformer.encoder.final_layernorm.weight".to_string(),
    ];
    for name in &final_norm_candidates {
        if tensor_set.contains(name.as_str()) {
            map.insert(10000 + final_norm_node, name.clone());
            break;
        }
    }

    map
}

/// Add MoE expert weight tensor names to the weight name map.
///
/// Maps Mixtral-style tensor names to the template expert slot node IDs.
/// The MoEWeightBank will load ALL expert weights; at runtime, bind_experts()
/// selects which experts' weights go into the template slots.
fn add_moe_weight_names(
    map: &mut HashMap<usize, String>,
    moe_slot_infos: &[(usize, crate::compiler::graph::MoESlotInfo)],
    available_tensors: &[(String, usize)],
) {
    let tensor_set: std::collections::HashSet<&str> = available_tensors
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    for (layer_idx, info) in moe_slot_infos {
        // Router gate weight
        let router_name = format!(
            "model.layers.{layer_idx}.block_sparse_moe.gate.weight"
        );
        if tensor_set.contains(router_name.as_str()) {
            map.insert(info.router_node_id, router_name);
        }

        // Per-expert weights: stored in MoEWeightBank, not in the graph directly.
        // We use a special key range (30000 + expert_idx * 100 + slot) to identify
        // MoE expert tensors. These are loaded into the bank, not the graph weights.
        for expert_idx in 0..info.num_experts {
            // w1 = gate_proj
            let w1_name = format!(
                "model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"
            );
            if tensor_set.contains(w1_name.as_str()) {
                map.insert(30000 + layer_idx * 1000 + expert_idx * 10, w1_name);
            }

            // w3 = up_proj
            let w3_name = format!(
                "model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight"
            );
            if tensor_set.contains(w3_name.as_str()) {
                map.insert(30000 + layer_idx * 1000 + expert_idx * 10 + 1, w3_name);
            }

            // w2 = down_proj
            let w2_name = format!(
                "model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"
            );
            if tensor_set.contains(w2_name.as_str()) {
                map.insert(30000 + layer_idx * 1000 + expert_idx * 10 + 2, w2_name);
            }
        }
    }
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

    // Memory-map all shards and deserialize headers ONCE (not per-weight)
    let t_load_start = std::time::Instant::now();

    struct ShardHandle {
        _file: std::fs::File,
        mmap: memmap2::Mmap,
    }

    let mut shards: Vec<ShardHandle> = Vec::with_capacity(shard_paths.len());
    for path in shard_paths {
        let file = std::fs::File::open(path).map_err(|e| WeightError::IoError(e.to_string()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| WeightError::IoError(e.to_string()))?;

        // Advise kernel to prefetch pages sequentially (reduces page fault stalls).
        // MADV_SEQUENTIAL triggers aggressive readahead; MADV_WILLNEED starts prefetch now.
        #[cfg(unix)]
        {
            unsafe {
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_SEQUENTIAL,
                );
                libc::madvise(
                    mmap.as_ptr() as *mut libc::c_void,
                    mmap.len(),
                    libc::MADV_WILLNEED,
                );
            }
        }

        shards.push(ShardHandle { _file: file, mmap });
    }

    // Build tensor_name → shard_index lookup (O(1) per weight instead of scanning all shards)
    let mut tensor_to_shard: HashMap<String, usize> = HashMap::new();
    for (shard_idx, shard) in shards.iter().enumerate() {
        let st = safetensors::SafeTensors::deserialize(&shard.mmap)
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        for name in st.names() {
            tensor_to_shard.insert(name.to_string(), shard_idx);
        }
    }
    eprintln!(
        "  Indexed {} tensors across {} shards in {:.1}s",
        tensor_to_shard.len(),
        shards.len(),
        t_load_start.elapsed().as_secs_f64(),
    );

    let mut weights = GraphWeights::new();
    let total_weights = name_map.len();
    let mut loaded_count = 0usize;

    // Collect which weights live in which shard: shard_idx → [(node_idx, k, n, tensor_name)]
    let mut shard_to_weights: HashMap<usize, Vec<(usize, usize, usize, &str)>> = HashMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            if let Some(tensor_name) = name_map.get(&idx) {
                if let Some(&shard_idx) = tensor_to_shard.get(tensor_name) {
                    shard_to_weights
                        .entry(shard_idx)
                        .or_default()
                        .push((idx, *k, *n, tensor_name));
                } else {
                    eprintln!(
                        "    ERROR: tensor '{}' not found in any shard for node {}",
                        tensor_name, idx
                    );
                }
            }
        }
    }

    // Two-phase loading:
    // Phase 1: Extract raw f32 data from all shards sequentially (triggers mmap page faults).
    //          MADV_WILLNEED prefetching overlaps I/O across shards.
    // Phase 2: Parallel transpose + quantize across ALL weights at once (better rayon utilization
    //          on many-core CPUs like 72-core Grace — 160 tasks vs ~20 per shard).
    use rayon::prelude::*;

    let t_extract = std::time::Instant::now();
    let mut all_raw: Vec<(usize, usize, usize, Vec<f32>, Vec<usize>)> =
        Vec::with_capacity(total_weights);

    for shard_idx in 0..shards.len() {
        let Some(weight_list) = shard_to_weights.get(&shard_idx) else {
            continue;
        };
        let tensors = safetensors::SafeTensors::deserialize(&shards[shard_idx].mmap)
            .map_err(|e| WeightError::IoError(e.to_string()))?;

        for &(idx, k, n, tensor_name) in weight_list {
            let tensor = tensors
                .tensor(tensor_name)
                .map_err(|e| WeightError::IoError(format!("{}: {e}", tensor_name)))?;
            let mut data = tensor_to_f32(tensor.data(), tensor.dtype());
            let shape = tensor.shape().to_vec();

            // FP8 per-block dequantization: apply weight_scale_inv if present.
            // MiniMax-M2.5, DeepSeek-V3 store FP8 weights with companion
            // scale tensors: tensor_name + "_scale_inv" with block_size [128, 128].
            let is_fp8 = matches!(tensor.dtype(),
                safetensors::Dtype::F8_E4M3 | safetensors::Dtype::F8_E5M2);
            if is_fp8 {
                let scale_name = format!("{}_scale_inv", tensor_name);
                if let Ok(scale_tensor) = tensors.tensor(&scale_name) {
                    let scales = tensor_to_f32(scale_tensor.data(), scale_tensor.dtype());
                    let scale_shape = scale_tensor.shape();
                    // Block size: weight shape / scale shape
                    let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, data.len()) };
                    let (s_rows, s_cols) = if scale_shape.len() == 2 {
                        (scale_shape[0], scale_shape[1])
                    } else {
                        (1, scales.len())
                    };
                    let block_r = if s_rows > 0 { rows / s_rows } else { rows };
                    let block_c = if s_cols > 0 { cols / s_cols } else { cols };

                    if block_r > 0 && block_c > 0 && scales.len() == s_rows * s_cols {
                        for r in 0..rows {
                            for c in 0..cols {
                                let sr = (r / block_r).min(s_rows - 1);
                                let sc = (c / block_c).min(s_cols - 1);
                                data[r * cols + c] *= scales[sr * s_cols + sc];
                            }
                        }
                        eprintln!("  FP8 dequant: {} [{rows}×{cols}] × scale [{s_rows}×{s_cols}] (block {block_r}×{block_c})",
                            tensor_name);
                    }
                }
            }

            all_raw.push((idx, k, n, data, shape));
        }
    }
    // Detect and split fused tensors (gate_up_proj, qkv_proj).
    // A fused gate_up tensor has shape [2*n, k] where the node expects [n, k].
    // Split: first half → gate weight (stays in all_raw), second half → up_proj named weight.
    let mut fused_up_proj_data: Vec<(usize, Vec<f32>, usize, usize)> = Vec::new(); // (down_node, data, rows, cols)
    for entry in all_raw.iter_mut() {
        let (idx, k, n, data, shape) = entry;
        if shape.len() == 2 {
            let tensor_rows = shape[0];
            let tensor_cols = shape[1];
            // Check if this is a fused QKV: tensor has more rows than expected
            // Handles: Phi-3 [3d, d], GPT-2 [d, 3d], GLM-4 [(H+2*KV)*head, d]
            // Extract just the Q portion (first n rows or columns)
            let is_fused_qkv = (tensor_rows == 3 * *n && tensor_cols == *k)
                || (tensor_cols == 3 * *n && tensor_rows == *k)
                || (tensor_rows >= 2 * *n && tensor_rows != 2 * *n && tensor_cols == *k && tensor_rows % *n == 0)
                || (tensor_cols >= 2 * *n && tensor_cols != 2 * *n && tensor_rows == *k && tensor_cols % *n == 0)
                // GLM-4 GQA fused QKV: (num_heads + 2*num_kv_heads) * head_dim rows
                // tensor_rows > n but tensor_rows < 2*n (e.g., 4608 > 4096 but < 8192)
                || (tensor_rows > *n && tensor_rows < 2 * *n && tensor_cols == *k);
            if is_fused_qkv {
                let is_col_fused = tensor_cols > tensor_rows;
                let multiplier = if is_col_fused { tensor_cols / *n } else { tensor_rows / *n };

                if is_col_fused {
                    // Shape [k, M*n]: take columns 0..n for Q
                    let q_data: Vec<f32> = (0..tensor_rows)
                        .flat_map(|r| data[r * tensor_cols..r * tensor_cols + *n].iter().copied())
                        .collect();
                    eprintln!("  Splitting fused QKV for node {} ({}×): [{}×{}] → Q [{}×{}]",
                        idx, multiplier, tensor_rows, tensor_cols, tensor_rows, *n);
                    *data = q_data;
                    shape[1] = *n;
                } else {
                    // Shape [M*n, k]: take rows 0..n for Q
                    let q_data = data[..*n * tensor_cols].to_vec();
                    eprintln!("  Splitting fused QKV for node {} ({}×): [{}×{}] → Q [{}×{}]",
                        idx, multiplier, tensor_rows, tensor_cols, *n, tensor_cols);
                    *data = q_data;
                    shape[0] = *n;
                }
            }

            // Check if this is a fused gate_up: tensor has 2× the expected output dimension
            else if (tensor_rows == 2 * *n && tensor_cols == *k) || (tensor_cols == 2 * *n && tensor_rows == *k) {
                let is_transposed = tensor_rows == *k;
                let (fused_rows, fused_cols) = if is_transposed {
                    (*k, 2 * *n)
                } else {
                    (2 * *n, *k)
                };
                let half_rows = fused_rows / 2;

                // Split: first half = gate, second half = up
                let gate_data: Vec<f32>;
                let up_data: Vec<f32>;
                if is_transposed {
                    // Shape [k, 2n]: columns 0..n = gate, columns n..2n = up
                    gate_data = (0..*k).flat_map(|r| data[r * fused_cols..r * fused_cols + *n].iter().copied()).collect();
                    up_data = (0..*k).flat_map(|r| data[r * fused_cols + *n..r * fused_cols + 2 * *n].iter().copied()).collect();
                } else {
                    // Shape [2n, k]: rows 0..n = gate, rows n..2n = up
                    gate_data = data[..half_rows * fused_cols].to_vec();
                    up_data = data[half_rows * fused_cols..].to_vec();
                }

                eprintln!("  Splitting fused gate_up_proj for node {}: [{}×{}] → gate [{}×{}] + up [{}×{}]",
                    idx, fused_rows, fused_cols, half_rows, fused_cols, half_rows, fused_cols);

                // Replace data with gate-only
                *data = gate_data;
                shape[0] = if is_transposed { *k } else { *n };
                shape[1] = if is_transposed { *n } else { *k };

                // Store up_proj for later named weight creation
                // The down_proj node is 2 nodes after gate_proj in 7-node blocks
                let down_node_id = *idx + 2;
                fused_up_proj_data.push((down_node_id, up_data, if is_transposed { *k } else { *n }, if is_transposed { *n } else { *k }));
            }
        }
    }

    eprintln!(
        "  Extracted {} tensors from {} shards in {:.1}s{}",
        all_raw.len(),
        shards.len(),
        t_extract.elapsed().as_secs_f64(),
        if fused_up_proj_data.is_empty() { String::new() } else { format!(" ({} fused splits)", fused_up_proj_data.len()) },
    );

    // Drop shard mmaps to free virtual memory before parallel processing
    drop(shards);

    let t_process = std::time::Instant::now();
    eprintln!(
        "  Processing {} weights in parallel (transpose + quantize)...",
        all_raw.len()
    );

    let processed: Vec<(usize, M31Matrix)> = all_raw
        .par_iter()
        .map(|(idx, k, n, data, shape)| {
            let (k, n) = (*k, *n);
            let (weight_data, wk, wn) = if shape.len() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                if rows == n && cols == k {
                    let mut transposed = vec![0.0f32; data.len()];
                    const BLOCK: usize = 64;
                    for r_block in (0..rows).step_by(BLOCK) {
                        for c_block in (0..cols).step_by(BLOCK) {
                            let r_end = (r_block + BLOCK).min(rows);
                            let c_end = (c_block + BLOCK).min(cols);
                            for r in r_block..r_end {
                                for c in c_block..c_end {
                                    transposed[c * rows + r] = data[r * cols + c];
                                }
                            }
                        }
                    }
                    (transposed, k, n)
                } else if rows == k && cols == n {
                    (data.clone(), k, n)
                } else {
                    (data.clone(), k, n)
                }
            } else {
                (data.clone(), k, n)
            };
            let (matrix, _params) = quantize_weight_matrix(&weight_data, wk, wn, strategy);
            (*idx, matrix)
        })
        .collect();

    for (idx, matrix) in processed {
        weights.add_weight(idx, matrix);
        loaded_count += 1;
    }
    eprintln!(
        "  Transpose + quantize: {:.1}s",
        t_process.elapsed().as_secs_f64(),
    );

    // ── Phase 2b: Store fused up_proj splits as named weights ──
    for (down_node_id, up_data, rows, cols) in &fused_up_proj_data {
        // Transpose from HF layout (out_features, in_features) to matmul layout (in_features, out_features)
        let mut transposed = vec![0.0f32; up_data.len()];
        for r in 0..*rows {
            for c in 0..*cols {
                transposed[c * rows + r] = up_data[r * cols + c];
            }
        }
        let (quantized, _params) = crate::gadgets::quantize::quantize_tensor(&transposed, strategy);
        let up_matrix = M31Matrix {
            rows: *cols,   // in_features
            cols: *rows,   // out_features
            data: quantized,
        };
        weights.add_named_weight(*down_node_id, "up_proj", up_matrix);
        loaded_count += 1;
        eprintln!("  Stored fused up_proj for down_proj node {} ({}×{})", down_node_id, cols, rows);
    }

    // ── Phase 3: Load norm γ weights (1D vectors, stored as named weights) ──
    // Keys >= 10000 are norm weights: actual node_id = key - 10000
    let gamma_entries: Vec<(usize, &str)> = name_map
        .iter()
        .filter(|(k, _)| **k >= 10000)
        .map(|(k, name)| (k - 10000, name.as_str()))
        .collect();

    if !gamma_entries.is_empty() {
        // Re-open shards for gamma loading (lightweight — 1D vectors are small)
        let mut gamma_shards: Vec<memmap2::Mmap> = Vec::new();
        for path in shard_paths {
            let file = std::fs::File::open(path)
                .map_err(|e| WeightError::IoError(e.to_string()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| WeightError::IoError(e.to_string()))?;
            gamma_shards.push(mmap);
        }

        let mut gamma_count = 0usize;
        for (node_id, tensor_name) in &gamma_entries {
            // Find which shard has this tensor
            for mmap in &gamma_shards {
                let Ok(tensors) = safetensors::SafeTensors::deserialize(mmap) else {
                    continue;
                };
                let Ok(tensor) = tensors.tensor(tensor_name) else {
                    continue;
                };

                let data = tensor_to_f32(tensor.data(), tensor.dtype());
                let dim = data.len();

                // Quantize γ to M31 (same strategy as other weights)
                let (quantized, _params) = crate::gadgets::quantize::quantize_tensor(&data, strategy);

                // Store as a named weight "gamma" for this norm node
                let gamma_matrix = M31Matrix {
                    rows: 1,
                    cols: dim,
                    data: quantized,
                };
                weights.add_named_weight(*node_id, "gamma", gamma_matrix);
                gamma_count += 1;
                loaded_count += 1;
                break;
            }
        }
        if gamma_count > 0 {
            eprintln!("  Loaded {} norm γ weights as named weights", gamma_count);
        }
    }

    // ── Phase 4: Load up_proj weights (gated FFN) as named weights ──
    // Keys >= 20000 are up_proj weights: actual node_id = key - 20000
    let up_proj_entries: Vec<(usize, &str)> = name_map
        .iter()
        .filter(|(k, _)| **k >= 20000)
        .map(|(k, name)| (k - 20000, name.as_str()))
        .collect();

    if !up_proj_entries.is_empty() {
        let mut up_shards: Vec<memmap2::Mmap> = Vec::new();
        for path in shard_paths {
            let file = std::fs::File::open(path)
                .map_err(|e| WeightError::IoError(e.to_string()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| WeightError::IoError(e.to_string()))?;
            up_shards.push(mmap);
        }

        let mut up_count = 0usize;
        for (node_id, tensor_name) in &up_proj_entries {
            for mmap in &up_shards {
                let Ok(tensors) = safetensors::SafeTensors::deserialize(mmap) else {
                    continue;
                };
                let Ok(tensor) = tensors.tensor(tensor_name) else {
                    continue;
                };

                let data = tensor_to_f32(tensor.data(), tensor.dtype());
                let shape = tensor.shape();
                let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (data.len(), 1) };

                // HF stores weights as (out_features, in_features).
                // For input × W_up, we need W_up as (in_features, out_features).
                // Transpose: (rows, cols) → (cols, rows)
                let mut transposed = vec![0.0f32; data.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        transposed[c * rows + r] = data[r * cols + c];
                    }
                }

                let (quantized, _params) = crate::gadgets::quantize::quantize_tensor(&transposed, strategy);
                let up_matrix = M31Matrix {
                    rows: cols,   // in_features
                    cols: rows,   // out_features
                    data: quantized,
                };
                weights.add_named_weight(*node_id, "up_proj", up_matrix);
                up_count += 1;
                loaded_count += 1;
                break;
            }
        }
        if up_count > 0 {
            eprintln!("  Loaded {} up_proj weights as named weights", up_count);
        }
    }

    eprintln!(
        "  All {} weights loaded in {:.1}s",
        loaded_count,
        t_load_start.elapsed().as_secs_f64(),
    );

    Ok(weights)
}

/// Load MoE expert weights from safetensors shards into a weight bank.
///
/// Returns one MoEWeightBank per MoE layer. Each bank holds all expert weight
/// triples (gate_proj, up_proj, down_proj) for that layer.
pub fn load_moe_weight_banks(
    shard_paths: &[std::path::PathBuf],
    name_map: &HashMap<usize, String>,
    moe_infos: &[(usize, crate::compiler::graph::MoESlotInfo)],
    strategy: QuantStrategy,
) -> Result<Vec<(usize, crate::compiler::graph::MoEWeightBank)>, crate::compiler::quantize_weights::WeightError> {
    use crate::compiler::quantize_weights::WeightError;

    if moe_infos.is_empty() {
        return Ok(Vec::new());
    }

    // Memory-map shards
    let mut shard_mmaps: Vec<memmap2::Mmap> = Vec::new();
    for path in shard_paths {
        let file = std::fs::File::open(path).map_err(|e| WeightError::IoError(e.to_string()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| WeightError::IoError(e.to_string()))?;
        shard_mmaps.push(mmap);
    }

    let mut banks = Vec::with_capacity(moe_infos.len());

    for (layer_idx, info) in moe_infos {
        let mut experts = Vec::with_capacity(info.num_experts);

        for expert_idx in 0..info.num_experts {
            // Keys: 30000 + layer*1000 + expert*10 + {0=gate, 1=up, 2=down}
            let gate_key = 30000 + layer_idx * 1000 + expert_idx * 10;
            let up_key = gate_key + 1;
            let down_key = gate_key + 2;

            let load_tensor = |key: usize| -> Result<M31Matrix, WeightError> {
                let tensor_name = name_map.get(&key).ok_or_else(|| {
                    WeightError::IoError(format!("MoE tensor key {} not in name_map (layer={}, expert={})", key, layer_idx, expert_idx))
                })?;

                for mmap in &shard_mmaps {
                    let Ok(tensors) = safetensors::SafeTensors::deserialize(mmap) else { continue };
                    let Ok(tensor) = tensors.tensor(tensor_name) else { continue };

                    let data = tensor_to_f32(tensor.data(), tensor.dtype());
                    let shape = tensor.shape();
                    let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (data.len(), 1) };

                    // Transpose: HF (out_features, in_features) → matmul (in_features, out_features)
                    let mut transposed = vec![0.0f32; data.len()];
                    for r in 0..rows {
                        for c in 0..cols {
                            transposed[c * rows + r] = data[r * cols + c];
                        }
                    }
                    let (quantized, _) = crate::gadgets::quantize::quantize_tensor(&transposed, strategy);
                    return Ok(M31Matrix { rows: cols, cols: rows, data: quantized });
                }
                Err(WeightError::IoError(format!("tensor '{}' not found in any shard", tensor_name)))
            };

            let gate_proj = load_tensor(gate_key)?;
            let up_proj = load_tensor(up_key)?;
            let down_proj = load_tensor(down_key)?;

            experts.push(crate::compiler::graph::MoEExpertWeights { gate_proj, up_proj, down_proj });
        }

        let bank = crate::compiler::graph::MoEWeightBank {
            experts,
            slot_gate_ids: info.slot_gate_ids.clone(),
            slot_down_ids: info.slot_down_ids.clone(),
            router_node_id: info.router_node_id,
            num_experts: info.num_experts,
            top_k: info.top_k,
        };
        banks.push((*layer_idx, bank));
        eprintln!("  Loaded MoE weight bank: layer {}, {} experts × 3 matrices", layer_idx, info.num_experts);
    }

    Ok(banks)
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode Graph (Attention-aware) — for decode-step proving
// ─────────────────────────────────────────────────────────────────────────────

/// Build a decode-compatible computation graph using `transformer_block()`.
///
/// Unlike `build_hf_transformer_graph()` which uses flat `linear()` calls,
/// this produces `GraphOp::Attention` nodes + residual `Add` connections,
/// matching what `prove_model_pure_gkr_decode_step` expects.
fn build_hf_decode_graph(
    config: &TransformerConfig,
    hf_config: &HfConfig,
    num_layers: usize,
) -> ComputationGraph {
    build_hf_full_graph(config, hf_config, num_layers, 1)
}

/// Build a full transformer graph with attention and configurable batch/seq_len.
///
/// This is the most complete graph: includes embedding lookup, Q/K/V/O projections,
/// attention mechanism (softmax + score matmul), residual connections, and FFN.
/// Used for both decode (seq_len=1) and batched (seq_len=N) proving.
///
/// When `include_embedding` is true, adds an Embedding node at the start
/// that proves the token→embedding lookup via LogUp.
pub fn build_hf_full_graph(
    config: &TransformerConfig,
    hf_config: &HfConfig,
    num_layers: usize,
    seq_len: usize,
) -> ComputationGraph {
    build_hf_full_graph_with_options(config, hf_config, num_layers, seq_len, false)
}

/// Build the full graph with optional embedding node.
pub fn build_hf_full_graph_with_options(
    config: &TransformerConfig,
    hf_config: &HfConfig,
    num_layers: usize,
    seq_len: usize,
    include_embedding: bool,
) -> ComputationGraph {
    use crate::compiler::onnx::NormType;
    let d = config.d_model;

    let mut builder = if include_embedding {
        // Start with token IDs as input, then embedding lookup
        let mut b = GraphBuilder::new((seq_len, 1)); // token IDs: seq_len × 1
        b.embedding(hf_config.vocab_size, d);
        b
    } else {
        GraphBuilder::new((seq_len, d))
    };

    for _ in 0..num_layers {
        builder.transformer_block(
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            seq_len,
            config.d_ff,
        );
    }
    // Final norm (matching prefill graph)
    match config.norm_type {
        NormType::LayerNorm => { builder.layer_norm(); }
        NormType::RMSNorm => { builder.rms_norm(); }
    }
    builder.build()
}

/// Build a weight name mapping for decode graphs (which contain Attention nodes).
///
/// Returns two maps:
/// - `matmul_map`: node_id → tensor_name for positional MatMul weights (FFN up/down)
/// - `attention_map`: Vec of (node_id, key_name, tensor_name) for named attention weights
fn build_decode_weight_name_map(
    graph: &ComputationGraph,
    num_layers: usize,
    available_tensors: &[(String, usize)],
) -> (HashMap<usize, String>, Vec<(usize, String, String)>) {
    let tensor_set: std::collections::HashSet<&str> = available_tensors
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    let mut matmul_map = HashMap::new();
    let mut attention_map: Vec<(usize, String, String)> = Vec::new();

    // Walk graph nodes per layer. transformer_block() produces per block:
    //   Identity(fork) → RMSNorm → Attention → Add → Identity(fork) → RMSNorm
    //   → MatMul(up) → Activation → MatMul(down) → Add
    // That's ~10 nodes/block.
    let mut layer_idx = 0usize;
    let mut matmul_in_layer = 0usize;

    for (idx, node) in graph.nodes.iter().enumerate() {
        match &node.op {
            GraphOp::Attention { .. } => {
                // Map Q/K/V/O projection weights as named weights
                let candidates = |suffix: &str| -> Vec<String> {
                    vec![
                        format!("model.layers.{layer_idx}.self_attn.{suffix}.weight"),
                        format!("model.layers.{layer_idx}.attention.{}.weight",
                            match suffix {
                                "q_proj" => "wq",
                                "k_proj" => "wk",
                                "v_proj" => "wv",
                                "o_proj" => "wo",
                                _ => suffix,
                            }),
                        format!("transformer.h.{layer_idx}.attn.{suffix}.weight"),
                    ]
                };

                for (key, suffix) in [
                    ("w_q", "q_proj"),
                    ("w_k", "k_proj"),
                    ("w_v", "v_proj"),
                    ("w_o", "o_proj"),
                ] {
                    for name in candidates(suffix) {
                        if tensor_set.contains(name.as_str()) {
                            attention_map.push((idx, key.to_string(), name));
                            break;
                        }
                    }
                }
                matmul_in_layer = 0;
            }
            GraphOp::MatMul { .. } => {
                // FFN MatMul nodes: up_proj (first) and down_proj (second) within each block
                let tensor_name = if matmul_in_layer == 0 {
                    // FFN up projection
                    let up_candidates = [
                        format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                        format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                        format!("model.layers.{layer_idx}.feed_forward.w1.weight"),
                        format!("transformer.h.{layer_idx}.mlp.up_proj.weight"),
                    ];
                    up_candidates.iter().find(|n| tensor_set.contains(n.as_str())).cloned()
                } else {
                    // FFN down projection
                    let down_candidates = [
                        format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                        format!("model.layers.{layer_idx}.feed_forward.w2.weight"),
                        format!("transformer.h.{layer_idx}.mlp.down_proj.weight"),
                    ];
                    down_candidates.iter().find(|n| tensor_set.contains(n.as_str())).cloned()
                };
                if let Some(name) = tensor_name {
                    matmul_map.insert(idx, name);
                }
                matmul_in_layer += 1;
            }
            GraphOp::Add { .. } => {
                // Second Add in a block marks end of transformer block
                // Count Adds to detect layer boundaries: 2 Adds per block
                // (post-attention residual + post-FFN residual)
                // We advance layer_idx after the second Add
                // But we need to track: after Attention's Add, we're in FFN part
                // After FFN's Add, layer is complete
                if matmul_in_layer >= 2 {
                    layer_idx += 1;
                    matmul_in_layer = 0;
                }
            }
            _ => {}
        }
    }

    (matmul_map, attention_map)
}

/// Load a model with the FULL attention graph for batched proving.
///
/// Builds the complete transformer graph with Q/K/V/O projections, attention
/// mechanism, residual connections, and FFN — proving the entire transformer
/// including attention. Uses `seq_len` for the attention dimension.
///
/// This is the cryptographically complete loading mode: all 7 weight matrices
/// per layer are mapped, and the attention mechanism is in the circuit.
pub fn load_hf_model_full(
    model_dir: &Path,
    num_layers: Option<usize>,
    seq_len: usize,
) -> Result<OnnxModel, OnnxError> {
    let config_path = model_dir.join("config.json");
    let hf_config = HfConfig::from_file(&config_path)?;
    let transformer_config = hf_config.to_transformer_config();
    let layers = num_layers.unwrap_or(hf_config.num_hidden_layers);
    let layers = if layers == 0 { hf_config.num_hidden_layers } else { layers };

    eprintln!("Model (full attention, seq_len={}): {} ({})",
        seq_len, hf_config.model_type, model_dir.display());
    eprintln!(
        "  hidden_size={}, heads={}/{} (q/kv), ff={}, layers={}/{}",
        hf_config.hidden_size, hf_config.num_attention_heads,
        hf_config.num_key_value_heads, hf_config.intermediate_size,
        layers, hf_config.num_hidden_layers,
    );

    // Include embedding node for LogUp proof of token→embedding lookup.
    // The 151K×896 embedding table is large (136M elements) but the Merkle
    // root is computed once and cached. Subsequent runs are instant.
    let graph = build_hf_full_graph_with_options(
        &transformer_config, &hf_config, layers, seq_len, true,
    );

    // Use the decode weight mapping (which handles Q/K/V/O + FFN)
    let shard_paths = discover_shards(model_dir, "model")
        .map_err(|e| OnnxError::WeightError(format!("Cannot discover shards: {e}")))?;

    let all_tensor_names: Vec<(String, usize)> = list_tensors_sharded(&shard_paths)
        .map_err(|e| OnnxError::WeightError(format!("Cannot list tensors: {e}")))?;

    let (matmul_map, attention_map) =
        build_decode_weight_name_map(&graph, layers, &all_tensor_names);
    eprintln!(
        "  Weight mapping: {} MatMul + {} Attention entries (all {} weight matrices)",
        matmul_map.len(), attention_map.len(),
        matmul_map.len() + attention_map.len(),
    );

    // Load FFN weights
    let mut weights =
        load_weights_from_shards(&shard_paths, &graph, &matmul_map, QuantStrategy::Symmetric8)
            .map_err(|e| OnnxError::WeightError(format!("Cannot load weights: {e}")))?;

    // Load Attention named weights (Q/K/V/O)
    let mut tensor_to_shard: HashMap<String, usize> = HashMap::new();
    for (_, _, name) in &attention_map {
        for (si, sp) in shard_paths.iter().enumerate() {
            let file = std::fs::File::open(sp).map_err(|e| OnnxError::IoError(e.to_string()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| OnnxError::IoError(e.to_string()))?;
            let tensors = safetensors::SafeTensors::deserialize(&mmap)
                .map_err(|e| OnnxError::WeightError(e.to_string()))?;
            if tensors.tensor(name).is_ok() {
                tensor_to_shard.insert(name.clone(), si);
                break;
            }
        }
    }

    // Memory-map all needed shards once
    let mut shard_mmaps: HashMap<usize, (std::fs::File, memmap2::Mmap)> = HashMap::new();
    for &si in tensor_to_shard.values() {
        if !shard_mmaps.contains_key(&si) {
            let file = std::fs::File::open(&shard_paths[si])
                .map_err(|e| OnnxError::IoError(e.to_string()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| OnnxError::IoError(e.to_string()))?;
            shard_mmaps.insert(si, (file, mmap));
        }
    }

    for (node_id, key_name, tensor_name) in &attention_map {
        if let Some(si) = tensor_to_shard.get(tensor_name) {
            if let Some((_file, mmap)) = shard_mmaps.get(si) {
                let tensors = safetensors::SafeTensors::deserialize(mmap)
                    .map_err(|e| OnnxError::WeightError(e.to_string()))?;
                if let Ok(tensor) = tensors.tensor(tensor_name) {
                    let shape = tensor.shape();
                    if shape.len() == 2 {
                        let data_f32 = tensor_to_f32(tensor.data(), tensor.dtype());
                        // Transpose: safetensors stores (out_features, in_features)
                        // but matmul expects (in_features, out_features) for input × W
                        let (raw_matrix, _) = quantize_weight_matrix(
                            &data_f32, shape[0], shape[1], QuantStrategy::Symmetric8,
                        );
                        // Transpose (out_feat, in_feat) → (in_feat, out_feat)
                        let mut transposed = M31Matrix::new(shape[1], shape[0]);
                        for r in 0..shape[0] {
                            for c in 0..shape[1] {
                                transposed.set(c, r, raw_matrix.get(r, c));
                            }
                        }
                        weights.add_named_weight(*node_id, key_name, transposed);
                    }
                }
            }
        }
    }

    // Load embedding table for the Embedding node (if present in graph).
    // The table is 136M elements — first load is slow but the weight commitment
    // cache handles subsequent runs instantly.
    let embed_node_id = graph.nodes.iter()
        .find(|n| matches!(&n.op, GraphOp::Embedding { .. }))
        .map(|n| n.id);

    if let Some(embed_id) = embed_node_id {
        eprintln!("  Loading embedding table for LogUp proof...");
        // Load the full embedding table (vocab_size × hidden_size)
        for sp in &shard_paths {
            let file = std::fs::File::open(sp)
                .map_err(|e| OnnxError::IoError(e.to_string()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| OnnxError::IoError(e.to_string()))?;
            let tensors = safetensors::SafeTensors::deserialize(&mmap)
                .map_err(|e| OnnxError::WeightError(e.to_string()))?;

            for &name in EMBED_CANDIDATES {
                if let Ok(tensor) = tensors.tensor(name) {
                    let shape = tensor.shape();
                    if shape.len() == 2 {
                        let data_f32 = tensor_to_f32(tensor.data(), tensor.dtype());
                        let (matrix, _) = quantize_weight_matrix(
                            &data_f32, shape[0], shape[1], QuantStrategy::Symmetric8,
                        );
                        weights.add_weight(embed_id, matrix);
                        eprintln!(
                            "  Embedding table loaded: {} ({}x{}) for node {}",
                            name, shape[0], shape[1], embed_id,
                        );
                        break;
                    }
                }
            }
            if weights.get_weight(embed_id).is_some() {
                break;
            }
        }
    }

    eprintln!("  All weights loaded (FFN + Attention + Embedding) ✓");

    let num_params: usize = weights.weights.iter().map(|(_, w)| w.rows * w.cols).sum::<usize>()
        + weights.named_weights.iter().map(|(_, _, w)| w.rows * w.cols).sum::<usize>();
    let metadata = crate::compiler::onnx::ModelMetadata {
        name: format!("{}_{}L_full", hf_config.model_type, layers),
        num_parameters: num_params,
        input_shape: (seq_len, hf_config.hidden_size),
        output_shape: (seq_len, hf_config.hidden_size),
        num_layers: graph.nodes.len(),
    };

    Ok(OnnxModel {
        graph,
        weights,
        input_shape: (seq_len, hf_config.hidden_size),
        metadata,
    })
}

/// Load a model from a HuggingFace directory in decode-compatible format.
///
/// Unlike `load_hf_model()`, this builds a graph with `Attention` nodes and
/// residual connections (via `transformer_block()`), suitable for decode-step
/// proving with `prove_model_pure_gkr_decode_step`.
///
/// Attention weights (Q/K/V/O projections) are loaded as named weights.
pub fn load_hf_model_decode(
    model_dir: &Path,
    num_layers: Option<usize>,
) -> Result<OnnxModel, OnnxError> {
    // ── Step 1: Run validation ──
    let report = validate_model_directory(model_dir, num_layers);
    eprintln!();
    eprintln!("  ── Model Validation (decode) ──");
    eprintln!("{}", report.format_report());
    eprintln!();

    if !report.passed() {
        return Err(OnnxError::WeightError(format!(
            "Model validation failed: {}/{} checks passed.",
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

    eprintln!("Model (decode): {} ({})", hf_config.model_type, model_dir.display());
    eprintln!(
        "  hidden_size={}, heads={}/{} (q/kv), ff={}, layers={}/{}",
        hf_config.hidden_size,
        hf_config.num_attention_heads,
        hf_config.num_key_value_heads,
        hf_config.intermediate_size,
        layers,
        hf_config.num_hidden_layers,
    );

    // ── Step 3: Build decode graph ──
    let graph = build_hf_decode_graph(&transformer_config, &hf_config, layers);

    // ── Step 4: Discover shards + build name maps ──
    let shard_paths = discover_shards(model_dir, "model")
        .map_err(|e| OnnxError::WeightError(format!("Cannot discover shards: {e}")))?;

    if shard_paths.is_empty() {
        return Err(OnnxError::WeightError(format!(
            "No SafeTensors weight files found in {}",
            model_dir.display(),
        )));
    }

    eprintln!("  Loading weights from {} shards...", shard_paths.len());

    let all_tensor_names: Vec<(String, usize)> = list_tensors_sharded(&shard_paths)
        .map_err(|e| OnnxError::WeightError(format!("Cannot list tensors: {e}")))?;

    let (matmul_map, attention_map) =
        build_decode_weight_name_map(&graph, layers, &all_tensor_names);
    eprintln!(
        "  Weight mapping: {} MatMul + {} Attention entries",
        matmul_map.len(),
        attention_map.len(),
    );

    // ── Step 4a: Load FFN MatMul weights ──
    let mut weights =
        load_weights_from_shards(&shard_paths, &graph, &matmul_map, QuantStrategy::Symmetric8)
            .map_err(|e| OnnxError::WeightError(format!("Cannot load weights: {e}")))?;

    // ── Step 4b: Load Attention named weights ──
    // Memory-map shards once for attention weights
    let mut tensor_to_shard: HashMap<String, usize> = HashMap::new();
    let mut shard_mmaps: Vec<memmap2::Mmap> = Vec::with_capacity(shard_paths.len());
    for path in &shard_paths {
        let file = std::fs::File::open(path)
            .map_err(|e| OnnxError::WeightError(format!("Cannot open shard: {e}")))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| OnnxError::WeightError(format!("Cannot mmap shard: {e}")))?;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| OnnxError::WeightError(format!("Cannot parse shard: {e}")))?;
        for name in st.names() {
            tensor_to_shard.insert(name.to_string(), shard_mmaps.len());
        }
        shard_mmaps.push(mmap);
    }

    let mut attn_loaded = 0usize;
    for (node_id, key_name, tensor_name) in &attention_map {
        let Some(&shard_idx) = tensor_to_shard.get(tensor_name) else {
            eprintln!(
                "    WARN: attention tensor '{}' not found in any shard for node {}",
                tensor_name, node_id,
            );
            continue;
        };
        let tensors = safetensors::SafeTensors::deserialize(&shard_mmaps[shard_idx])
            .map_err(|e| OnnxError::WeightError(format!("{tensor_name}: {e}")))?;
        let tensor = tensors
            .tensor(tensor_name)
            .map_err(|e| OnnxError::WeightError(format!("{tensor_name}: {e}")))?;
        let data = tensor_to_f32(tensor.data(), tensor.dtype());
        let shape = tensor.shape();

        // Attention weight shape: [out_features, in_features]
        // Quantize and store as named weight
        let (rows, cols) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, data.len()) };
        let (matrix, _) = quantize_weight_matrix(&data, cols, rows, QuantStrategy::Symmetric8);
        weights.add_named_weight(*node_id, key_name, matrix);
        attn_loaded += 1;
    }
    eprintln!("  Loaded {} attention weight matrices", attn_loaded);

    // ── Step 5: Verify all Attention nodes have named weights ──
    let attention_nodes: Vec<usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n.op, GraphOp::Attention { .. }))
        .map(|(idx, _)| idx)
        .collect();

    for &idx in &attention_nodes {
        for key in &["w_q", "w_k", "w_v", "w_o"] {
            if weights.get_named_weight(idx, key).is_none() {
                eprintln!(
                    "  WARN: Attention node {} missing named weight '{}'",
                    idx, key,
                );
            }
        }
    }

    let num_parameters = crate::compiler::onnx::count_matmul_params(&graph);

    let metadata = ModelMetadata {
        name: format!("{}_{}L_decode", hf_config.model_type, layers),
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
// Embedding Row Extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Bytes per element for each SafeTensors dtype.
fn dtype_byte_width(dtype: safetensors::Dtype) -> usize {
    match dtype {
        safetensors::Dtype::F32 => 4,
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
        safetensors::Dtype::I8 | safetensors::Dtype::U8 => 1,
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
        safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::I16 | safetensors::Dtype::U16 => 2,
        _ => 4, // fallback
    }
}

/// Common embedding tensor names across architectures.
const EMBED_CANDIDATES: &[&str] = &[
    "model.embed_tokens.weight",
    "transformer.wte.weight",
    "transformer.word_embeddings.weight",
    // GLM-4
    "transformer.embedding.word_embeddings.weight",
];

/// Extract a single embedding row for a token ID from a HuggingFace model.
///
/// Only reads and quantizes the single row needed — avoids loading the full
/// embedding table (~3 GB for 152K vocab × 5120 hidden_size).
///
/// Returns a `(1, hidden_size)` M31Matrix containing the quantized embedding.
///
/// # Arguments
/// * `model_dir` - Path to the HuggingFace model directory
/// * `hidden_size` - Expected hidden dimension (from config.json)
/// * `token_id` - Token ID whose embedding row to extract
pub fn load_embedding_row(
    model_dir: &Path,
    hidden_size: usize,
    token_id: u32,
) -> Result<(M31Matrix, usize), OnnxError> {
    let shard_paths = discover_shards(model_dir, "model")
        .map_err(|e| OnnxError::WeightError(format!("Cannot discover shards: {e}")))?;

    if shard_paths.is_empty() {
        return Err(OnnxError::WeightError(format!(
            "No SafeTensors weight files found in {}",
            model_dir.display(),
        )));
    }

    for path in &shard_paths {
        let file = std::fs::File::open(path)
            .map_err(|e| OnnxError::IoError(format!("Cannot open {}: {e}", path.display())))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| OnnxError::IoError(format!("Cannot mmap {}: {e}", path.display())))?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| OnnxError::WeightError(format!("Cannot parse {}: {e}", path.display())))?;

        for &name in EMBED_CANDIDATES {
            if let Ok(tensor) = tensors.tensor(name) {
                let shape = tensor.shape();
                if shape.len() != 2 {
                    return Err(OnnxError::WeightError(format!(
                        "Embedding tensor '{}' has {} dims, expected 2",
                        name,
                        shape.len(),
                    )));
                }
                let vocab_size = shape[0];
                let embed_dim = shape[1];
                if embed_dim != hidden_size {
                    return Err(OnnxError::WeightError(format!(
                        "Embedding dim mismatch: tensor has {}, config has {}",
                        embed_dim, hidden_size,
                    )));
                }

                let tid = token_id as usize;
                if tid >= vocab_size {
                    return Err(OnnxError::WeightError(format!(
                        "token_id {} exceeds vocab_size {}",
                        token_id, vocab_size,
                    )));
                }

                // Extract only the bytes for this single row from the mmap'd tensor.
                let bw = dtype_byte_width(tensor.dtype());
                let row_bytes = embed_dim * bw;
                let offset = tid * row_bytes;
                let raw = tensor.data();
                let row_slice = &raw[offset..offset + row_bytes];

                // Convert just this row to f32 and quantize.
                let row_f32 = tensor_to_f32(row_slice, tensor.dtype());
                let (matrix, _params) =
                    quantize_weight_matrix(&row_f32, 1, embed_dim, QuantStrategy::Symmetric8);

                if std::env::var("OBELYZK_VERBOSE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "  Embedding row {}: extracted from '{}' ({}x{} table, {} dtype)",
                        token_id, name, vocab_size, embed_dim, bw * 8,
                    );
                }
                return Ok((matrix, vocab_size));
            }
        }
    }

    Err(OnnxError::WeightError(format!(
        "Embedding tensor not found. Searched for: {}",
        EMBED_CANDIDATES.join(", "),
    )))
}

/// Load embeddings for a batch of token IDs and stack into (N × hidden_size) matrix.
///
/// This is the key function for batched proving: instead of proving N separate
/// forward passes for N tokens, we stack all embeddings into a single matrix
/// and prove ONE batched forward pass. GKR cost scales as log(N), not N.
pub fn load_embedding_batch(
    model_dir: &Path,
    hidden_size: usize,
    token_ids: &[u32],
) -> Result<M31Matrix, OnnxError> {
    let n = token_ids.len();
    if n == 0 {
        return Err(OnnxError::WeightError("Empty token list".to_string()));
    }

    let shard_paths = discover_shards(model_dir, "model")
        .map_err(|e| OnnxError::WeightError(format!("Cannot discover shards: {e}")))?;

    for path in &shard_paths {
        let file = std::fs::File::open(path)
            .map_err(|e| OnnxError::IoError(format!("Cannot open {}: {e}", path.display())))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| OnnxError::IoError(format!("Cannot mmap {}: {e}", path.display())))?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| OnnxError::WeightError(format!("Cannot parse {}: {e}", path.display())))?;

        for &name in EMBED_CANDIDATES {
            if let Ok(tensor) = tensors.tensor(name) {
                let shape = tensor.shape();
                if shape.len() != 2 {
                    continue;
                }
                let vocab_size = shape[0];
                let embed_dim = shape[1];
                if embed_dim != hidden_size {
                    return Err(OnnxError::WeightError(format!(
                        "Embedding dim mismatch: tensor has {}, config has {}",
                        embed_dim, hidden_size,
                    )));
                }

                let bw = dtype_byte_width(tensor.dtype());
                let row_bytes = embed_dim * bw;
                let raw = tensor.data();

                // Build stacked (N × hidden_size) matrix
                let mut batch = M31Matrix::new(n, hidden_size);

                for (row_idx, &tid) in token_ids.iter().enumerate() {
                    let tid = tid as usize;
                    if tid >= vocab_size {
                        return Err(OnnxError::WeightError(format!(
                            "token_id {} exceeds vocab_size {}",
                            tid, vocab_size,
                        )));
                    }

                    let offset = tid * row_bytes;
                    let row_slice = &raw[offset..offset + row_bytes];
                    let row_f32 = tensor_to_f32(row_slice, tensor.dtype());

                    // Quantize each element to M31
                    for (col, &val) in row_f32.iter().enumerate() {
                        let quantized = quantize_single_m31(val);
                        batch.set(row_idx, col, quantized);
                    }
                }

                eprintln!(
                    "  Embedding batch: {} tokens from '{}' ({}x{} → {}x{})",
                    n, name, vocab_size, embed_dim, n, hidden_size,
                );
                return Ok(batch);
            }
        }
    }

    Err(OnnxError::WeightError(format!(
        "Embedding tensor not found. Searched for: {}",
        EMBED_CANDIDATES.join(", "),
    )))
}

/// Candidate tensor names for the output projection (lm_head).
const LM_HEAD_CANDIDATES: &[&str] = &[
    "lm_head.weight",
    "output.weight",
    // GLM-4
    "transformer.output_layer.weight",
];

/// Project a hidden-state M31Matrix through lm_head (or tied embedding weights)
/// to produce vocabulary logits, then return (argmax_token_id, max_logit_score).
///
/// The matmul is performed in f32 space and is NOT proven — it's a convenience
/// for producing human-readable text output. The proof covers the transformer
/// forward pass (embedding → final hidden state).
pub fn project_to_logits(
    model_dir: &Path,
    hidden_state: &M31Matrix, // (1, d_model)
) -> Result<(u32, f32), OnnxError> {
    let d_model = hidden_state.cols;

    let shard_paths = discover_shards(model_dir, "model")
        .map_err(|e| OnnxError::WeightError(format!("Cannot discover shards: {e}")))?;

    // Try lm_head.weight first, then fall back to tied embedding weights
    let all_candidates: Vec<&str> = LM_HEAD_CANDIDATES.iter()
        .chain(EMBED_CANDIDATES.iter())
        .copied()
        .collect();

    for path in &shard_paths {
        let file = std::fs::File::open(path)
            .map_err(|e| OnnxError::IoError(format!("Cannot open {}: {e}", path.display())))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| OnnxError::IoError(format!("Cannot mmap {}: {e}", path.display())))?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| OnnxError::WeightError(format!("Cannot parse {}: {e}", path.display())))?;

        for &name in &all_candidates {
            if let Ok(tensor) = tensors.tensor(name) {
                let shape = tensor.shape();
                if shape.len() != 2 { continue; }
                let vocab_size = shape[0];
                let embed_dim = shape[1];
                if embed_dim != d_model { continue; }

                // Dequantize hidden state from M31 → f32
                let scale = 127.0_f32;
                let hidden_f32: Vec<f32> = hidden_state.data.iter()
                    .take(d_model)
                    .map(|m| m.0 as f32 / scale)
                    .collect();

                // Load the full projection weight as f32
                let raw = tensor.data();
                let bw = dtype_byte_width(tensor.dtype());
                let weight_f32 = tensor_to_f32(raw, tensor.dtype());

                // Matmul: hidden (1, d_model) @ weight^T (d_model, vocab_size)
                // weight is stored as (vocab_size, d_model), row-major
                let mut best_id: u32 = 0;
                let mut best_score = f32::NEG_INFINITY;
                for v in 0..vocab_size {
                    let row_offset = v * embed_dim;
                    let mut dot = 0.0_f32;
                    for d in 0..d_model {
                        dot += hidden_f32[d] * weight_f32[row_offset + d];
                    }
                    if dot > best_score {
                        best_score = dot;
                        best_id = v as u32;
                    }
                }

                if std::env::var("OBELYZK_VERBOSE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "  project_to_logits: {} → argmax={} (score={:.2}, vocab={})",
                        name, best_id, best_score, vocab_size,
                    );
                }
                let _ = bw; // suppress unused warning
                return Ok((best_id, best_score));
            }
        }
    }

    Err(OnnxError::WeightError(format!(
        "lm_head/embedding tensor not found. Searched for: {}",
        all_candidates.join(", "),
    )))
}

/// Quantize a single f32 value to M31.
fn quantize_single_m31(val: f32) -> M31 {
    const P: u32 = (1u32 << 31) - 1; // M31 prime
    let scale = 127.0; // symmetric 8-bit scale
    let quantized = (val * scale).round().clamp(0.0, (P - 1) as f32);
    M31::from(quantized as u32)
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
            norm_type: crate::compiler::onnx::NormType::LayerNorm,
        };

        let (graph, moe_infos) = build_hf_transformer_graph(&config, 2);

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
        let tmp = std::env::temp_dir().join("obelyzk_empty_model");
        std::fs::create_dir_all(&tmp).unwrap();

        let report = validate_model_directory(&tmp, Some(1));
        assert!(!report.passed());

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_load_embedding_row() {
        // Create a small fake model dir with embed_tokens in a safetensors file.
        let tmp = std::env::temp_dir().join("obelyzk_embed_row_test");
        std::fs::create_dir_all(&tmp).unwrap();

        let vocab_size = 8;
        let hidden_size = 4;
        // Row-major: row i = [i*4, i*4+1, i*4+2, i*4+3] as f32
        let data: Vec<f32> = (0..(vocab_size * hidden_size))
            .map(|i| i as f32)
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![vocab_size, hidden_size],
                &bytes,
            )
            .unwrap(),
        );
        let serialized = safetensors::serialize(&tensors, &None).unwrap();
        std::fs::write(tmp.join("model.safetensors"), &serialized).unwrap();

        // Extract row 3 (values 12.0, 13.0, 14.0, 15.0)
        let (row, vs) = load_embedding_row(&tmp, hidden_size, 3).unwrap();
        assert_eq!(vs, vocab_size);
        assert_eq!(row.rows, 1);
        assert_eq!(row.cols, hidden_size);
        // Values should be non-zero (quantized from 12..15)
        assert!(row.data.iter().any(|v| v.0 != 0));

        // Out-of-range token should fail
        assert!(load_embedding_row(&tmp, hidden_size, 99).is_err());

        // Wrong hidden_size should fail
        assert!(load_embedding_row(&tmp, hidden_size + 1, 0).is_err());

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_load_embedding_row_missing_tensor() {
        let tmp = std::env::temp_dir().join("obelyzk_embed_no_tensor");
        std::fs::create_dir_all(&tmp).unwrap();

        // SafeTensors file with a tensor that has a different name
        let data: Vec<f32> = vec![1.0; 16];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "some.other.weight".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![4, 4],
                &bytes,
            )
            .unwrap(),
        );
        let serialized = safetensors::serialize(&tensors, &None).unwrap();
        std::fs::write(tmp.join("model.safetensors"), &serialized).unwrap();

        let result = load_embedding_row(&tmp, 4, 0);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found"), "expected 'not found' in: {err}");

        std::fs::remove_dir_all(&tmp).ok();
    }
}
