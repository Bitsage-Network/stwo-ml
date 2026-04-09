//! Self-evaluation: the model grades its own outputs.
//!
//! For each inference, constructs an evaluation prompt, runs the model's
//! forward pass on it, and extracts a quality score from the output.
//! The evaluation forward pass can optionally be proved (expensive).

use starknet_ff::FieldElement;

use crate::aggregation::compute_io_commitment;
use crate::audit::deterministic::evaluate_deterministic;
use crate::audit::types::{AuditError, InferenceEvaluation, InferenceLogEntry};
use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::components::matmul::M31Matrix;

use stwo::core::fields::m31::M31;

// ─── Eval Templates ─────────────────────────────────────────────────────────

/// Template for constructing evaluation prompts.
#[derive(Debug, Clone, Copy)]
pub enum EvalTemplate {
    /// General quality assessment.
    GeneralQuality,
    /// Task adherence (did the output match the request?).
    TaskAdherence,
    /// Factual consistency (no hallucinations).
    FactualConsistency,
    /// Code quality (correctness, style, efficiency).
    CodeQuality,
}

impl EvalTemplate {
    /// Select the best template for a given task category.
    pub fn for_category(category: &str) -> Self {
        match category {
            "code_generation" => Self::CodeQuality,
            "qa" => Self::FactualConsistency,
            _ => Self::GeneralQuality,
        }
    }

    fn system_prompt(&self) -> &'static str {
        match self {
            Self::GeneralQuality => {
                "Rate the quality of the following AI response on a scale of 0-10. \
                 Consider: accuracy, helpfulness, clarity, and completeness. \
                 Respond with ONLY a number between 0 and 10."
            }
            Self::TaskAdherence => {
                "Rate how well the following AI response addresses the user's request \
                 on a scale of 0-10. Consider: relevance, completeness, and format. \
                 Respond with ONLY a number between 0 and 10."
            }
            Self::FactualConsistency => {
                "Rate the factual accuracy of the following AI response on a scale of 0-10. \
                 Check for hallucinations, incorrect claims, and logical errors. \
                 Respond with ONLY a number between 0 and 10."
            }
            Self::CodeQuality => {
                "Rate the quality of the following code on a scale of 0-10. \
                 Consider: correctness, efficiency, readability, and error handling. \
                 Respond with ONLY a number between 0 and 10."
            }
        }
    }
}

// ─── Prompt Construction ────────────────────────────────────────────────────

/// Build an evaluation prompt from the original input/output and a template.
///
/// The returned string is what gets tokenized and fed into the model forward pass.
pub fn build_eval_prompt(input: &str, output: &str, template: EvalTemplate) -> String {
    format!(
        "{}\n\nUser request:\n{}\n\nAI response:\n{}\n\nScore:",
        template.system_prompt(),
        truncate(input, 500),
        truncate(output, 1000),
    )
}

fn truncate(s: &str, max_chars: usize) -> &str {
    if s.len() <= max_chars {
        s
    } else {
        // Find a safe UTF-8 boundary.
        let mut end = max_chars;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

// ─── Score Parsing ──────────────────────────────────────────────────────────

/// Parse a quality score (0-10) from model output, normalize to 0.0-1.0.
///
/// Handles formats like "8", "8.5", "Score: 8", "8/10", "8 out of 10".
pub fn parse_score(model_output: &str) -> Option<f32> {
    let trimmed = model_output.trim();

    // Try direct number first.
    if let Ok(n) = trimmed.parse::<f32>() {
        return normalize_score(n);
    }

    // Try "X/10" format.
    if let Some(slash_pos) = trimmed.find("/10") {
        let before = trimmed[..slash_pos].trim();
        if let Some(n) = extract_last_number(before) {
            return normalize_score(n);
        }
    }

    // Try "X out of 10" format.
    if let Some(out_pos) = trimmed.to_lowercase().find("out of 10") {
        let before = trimmed[..out_pos].trim();
        if let Some(n) = extract_last_number(before) {
            return normalize_score(n);
        }
    }

    // Try extracting the first number from the output.
    extract_first_number(trimmed).and_then(normalize_score)
}

fn normalize_score(raw: f32) -> Option<f32> {
    if raw < 0.0 || raw > 10.0 {
        None
    } else {
        Some(raw / 10.0)
    }
}

fn extract_first_number(s: &str) -> Option<f32> {
    let mut num_str = String::new();
    let mut found_digit = false;
    let mut found_dot = false;

    for ch in s.chars() {
        if ch.is_ascii_digit() {
            num_str.push(ch);
            found_digit = true;
        } else if ch == '.' && found_digit && !found_dot {
            num_str.push(ch);
            found_dot = true;
        } else if found_digit {
            break;
        }
    }

    if found_digit {
        num_str.parse::<f32>().ok()
    } else {
        None
    }
}

fn extract_last_number(s: &str) -> Option<f32> {
    // Walk backward to find the last number.
    let s = s.trim();
    let mut end = s.len();
    let mut start = end;

    let chars: Vec<char> = s.chars().collect();
    let mut i = chars.len();
    let mut found = false;

    while i > 0 {
        i -= 1;
        if chars[i].is_ascii_digit() || chars[i] == '.' {
            if !found {
                end = i + 1;
                found = true;
            }
            start = i;
        } else if found {
            break;
        }
    }

    if found {
        // Reconstruct the byte range.
        let byte_start: usize = chars[..start].iter().map(|c| c.len_utf8()).sum();
        let byte_end: usize = chars[..end].iter().map(|c| c.len_utf8()).sum();
        s[byte_start..byte_end].parse::<f32>().ok()
    } else {
        None
    }
}

// ─── Evaluation ─────────────────────────────────────────────────────────────

/// Configuration for self-evaluation.
pub struct SelfEvalConfig {
    /// Which template to use (None = auto-detect from category).
    pub template: Option<EvalTemplate>,
    /// Whether to prove the evaluation forward passes.
    pub prove_evaluations: bool,
}

impl Default for SelfEvalConfig {
    fn default() -> Self {
        Self {
            template: None,
            prove_evaluations: false,
        }
    }
}

/// Evaluate a single inference using deterministic checks + model self-eval.
///
/// If `graph` and `weights` are provided, runs a forward pass on the eval
/// prompt and extracts a semantic score. Otherwise, only deterministic checks
/// are returned.
pub fn evaluate_inference(
    entry: &InferenceLogEntry,
    graph: Option<&ComputationGraph>,
    weights: Option<&GraphWeights>,
    config: &SelfEvalConfig,
) -> InferenceEvaluation {
    let input_text = entry.input_preview.as_deref().unwrap_or("");
    let output_text = entry.output_preview.as_deref().unwrap_or("");

    let category = entry.task_category.as_deref().unwrap_or_else(|| "general");

    // Deterministic checks.
    let deterministic_checks = evaluate_deterministic(input_text, output_text, Some(category));

    // Self-evaluation via model forward pass (if model available).
    let (semantic_score, eval_io_commitment) = if let (Some(g), Some(w)) = (graph, weights) {
        let template = config
            .template
            .unwrap_or_else(|| EvalTemplate::for_category(category));
        let eval_prompt = build_eval_prompt(input_text, output_text, template);

        match run_eval_forward_pass(g, w, &eval_prompt) {
            Ok((score, commitment)) => (score, Some(format!("{:#066x}", commitment))),
            Err(_) => (None, None),
        }
    } else {
        (None, None)
    };

    InferenceEvaluation {
        sequence: entry.sequence_number,
        deterministic_checks,
        semantic_score,
        eval_io_commitment,
        evaluation_proved: false, // Proving is done separately if requested.
    }
}

/// Batch-evaluate multiple inferences.
pub fn evaluate_batch(
    entries: &[InferenceLogEntry],
    graph: Option<&ComputationGraph>,
    weights: Option<&GraphWeights>,
    config: &SelfEvalConfig,
) -> Vec<InferenceEvaluation> {
    entries
        .iter()
        .map(|entry| evaluate_inference(entry, graph, weights, config))
        .collect()
}

/// Run the model forward pass on an evaluation prompt and extract a score.
///
/// The prompt is tokenized into M31 values (each char → M31), padded to
/// the graph's input shape, then run through `execute_forward_pass`.
fn run_eval_forward_pass(
    graph: &ComputationGraph,
    weights: &GraphWeights,
    eval_prompt: &str,
) -> Result<(Option<f32>, FieldElement), AuditError> {
    let (rows, cols) = graph.input_shape;

    // Tokenize: each byte of the prompt → one M31 element.
    let bytes = eval_prompt.as_bytes();
    let total = rows * cols;
    let mut data = vec![M31::from(0u32); total];
    for (i, &b) in bytes.iter().take(total).enumerate() {
        data[i] = M31::from(b as u32);
    }

    let input = M31Matrix { rows, cols, data };

    let output = crate::audit::replay::execute_forward_pass(graph, &input, weights)?;
    let commitment = compute_io_commitment(&input, &output);

    // Extract score from output: interpret first elements as ASCII.
    let output_text: String = output
        .data
        .iter()
        .take(20)
        .filter_map(|m| {
            let v = m.0;
            if v > 0 && v < 128 {
                Some(v as u8 as char)
            } else {
                None
            }
        })
        .collect();

    let score = parse_score(&output_text);

    Ok((score, commitment))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_score_direct() {
        assert_eq!(parse_score("8"), Some(0.8));
        assert_eq!(parse_score("8.5"), Some(0.85));
        assert_eq!(parse_score("10"), Some(1.0));
        assert_eq!(parse_score("0"), Some(0.0));
    }

    #[test]
    fn test_parse_score_formats() {
        assert_eq!(parse_score("8/10"), Some(0.8));
        assert_eq!(parse_score("Score: 7"), Some(0.7));
        assert_eq!(parse_score("8 out of 10"), Some(0.8));
        assert_eq!(parse_score("I rate this 9.5"), Some(0.95));
    }

    #[test]
    fn test_parse_score_invalid() {
        assert_eq!(parse_score(""), None);
        assert_eq!(parse_score("no number here"), None);
        assert_eq!(parse_score("11"), None); // Out of range.
        assert_eq!(parse_score("-1"), None);
    }

    #[test]
    fn test_build_eval_prompt() {
        let prompt = build_eval_prompt("hello", "world", EvalTemplate::GeneralQuality);
        assert!(prompt.contains("Rate the quality"));
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("world"));
        assert!(prompt.contains("Score:"));
    }

    #[test]
    fn test_eval_template_selection() {
        assert!(matches!(
            EvalTemplate::for_category("code_generation"),
            EvalTemplate::CodeQuality
        ));
        assert!(matches!(
            EvalTemplate::for_category("qa"),
            EvalTemplate::FactualConsistency
        ));
        assert!(matches!(
            EvalTemplate::for_category("general"),
            EvalTemplate::GeneralQuality
        ));
    }

    #[test]
    fn test_evaluate_inference_deterministic_only() {
        let entry = InferenceLogEntry {
            inference_id: 0,
            sequence_number: 42,
            model_id: "0x1".to_string(),
            weight_commitment: "0x0".to_string(),
            model_name: "test".to_string(),
            num_layers: 1,
            input_tokens: vec![],
            output_tokens: vec![],
            matrix_offset: 0,
            matrix_size: 0,
            input_rows: 0,
            input_cols: 0,
            output_rows: 0,
            output_cols: 0,
            io_commitment: "0x0".to_string(),
            layer_chain_commitment: "0x0".to_string(),
            prev_entry_hash: "0x0".to_string(),
            entry_hash: "0x0".to_string(),
            timestamp_ns: 0,
            latency_ms: 0,
            gpu_device: "test".to_string(),
            tee_report_hash: "0x0".to_string(),
            task_category: Some("json".to_string()),
            input_preview: Some("Return JSON".to_string()),
            output_preview: Some("{\"key\": 1}".to_string()),
        };

        let config = SelfEvalConfig::default();
        let eval = evaluate_inference(&entry, None, None, &config);

        assert_eq!(eval.sequence, 42);
        assert!(!eval.deterministic_checks.is_empty());
        assert!(eval.semantic_score.is_none()); // No model provided.
        assert!(eval.eval_io_commitment.is_none());
    }
}
