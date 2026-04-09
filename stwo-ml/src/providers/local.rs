//! Local inference provider — runs the M31 forward pass with full ZK proof.
//!
//! This is the highest-trust provider: the computation runs inside the VM
//! and every operation is provable via GKR sumcheck + recursive STARK.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::compiler::graph::{ComputationGraph, GraphWeights};
use crate::components::attention::ModelKVCache;
use crate::components::matmul::M31Matrix;
use crate::policy::PolicyConfig;
use crate::providers::types::*;
use crate::vm::trace::ExecutionTrace;

/// Local inference provider — loads HuggingFace models, runs M31 forward pass.
pub struct LocalProvider {
    pub model_name: String,
    pub model_dir: PathBuf,
    pub graph: Arc<ComputationGraph>,
    pub weights: Arc<GraphWeights>,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    pub weight_commitment: String,
    pub hidden_size: usize,
}

impl LocalProvider {
    /// Load a model from a HuggingFace directory.
    pub fn load(model_dir: &Path, name: Option<&str>) -> Result<Self, LocalProviderError> {
        let dir_name = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());

        let model_name = name.unwrap_or(&dir_name).to_string();

        eprintln!("[local] Loading model from {}...", model_dir.display());
        let hf = crate::compiler::hf_loader::load_hf_model(model_dir, None)
            .map_err(|e| LocalProviderError::LoadFailed(format!("{e}")))?;

        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| LocalProviderError::LoadFailed(format!("tokenizer: {e}")))?;

        let hidden_size = hf.input_shape.1;
        eprintln!(
            "[local] Loaded: {} ({} layers, {} weights, d_model={})",
            model_name,
            hf.graph.num_layers(),
            hf.weights.weights.len(),
            hidden_size,
        );

        Ok(Self {
            model_name,
            model_dir: model_dir.to_path_buf(),
            graph: Arc::new(hf.graph),
            weights: Arc::new(hf.weights),
            tokenizer: Arc::new(tokenizer),
            weight_commitment: "deferred".into(),
            hidden_size,
        })
    }

    /// Run inference from a text prompt.
    ///
    /// Returns (output_text, inference_result, execution_trace).
    pub fn infer_text(
        &self,
        prompt: &str,
        kv_cache: Option<&mut ModelKVCache>,
    ) -> Result<(String, InferenceResult, ExecutionTrace), LocalProviderError> {
        let t_start = Instant::now();

        // 1. Tokenize
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| LocalProviderError::TokenizeFailed(format!("{e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            return Err(LocalProviderError::EmptyPrompt);
        }
        let last_token_id = *token_ids.last().unwrap();

        // 2. Embed
        let (input_matrix, _vocab_size) = crate::compiler::hf_loader::load_embedding_row(
            &self.model_dir, self.hidden_size, last_token_id,
        ).map_err(|e| LocalProviderError::EmbedFailed(format!("{e}")))?;

        // 3. Execute + prove
        let trace = crate::vm::executor::execute_and_prove(
            &self.graph,
            &input_matrix,
            &self.weights,
            kv_cache,
            Some(&PolicyConfig::standard()),
            token_ids.clone(),
            self.model_name.clone(),
        ).map_err(|e| LocalProviderError::ProveFailed(format!("{e}")))?;

        // 4. Project to logits for predicted text
        let predicted_text = crate::compiler::hf_loader::project_to_logits(
            &self.model_dir, &trace.output,
        )
        .ok()
        .and_then(|(tid, _)| self.tokenizer.decode(&[tid], true).ok())
        .unwrap_or_default();

        let inference_time_ms = t_start.elapsed().as_millis() as u64;

        let io_commitment = trace.io_commitment;
        let result = InferenceResult {
            text: predicted_text.clone(),
            token_ids: token_ids.clone(),
            num_tokens: token_ids.len(),
            model_id: self.model_name.clone(),
            provider: "local".into(),
            trust_model: TrustModel::ZkProof {
                weight_commitment: self.weight_commitment.clone(),
            },
            io_commitment,
            inference_time_ms,
        };

        Ok((predicted_text, result, trace))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LocalProviderError {
    #[error("Failed to load model: {0}")]
    LoadFailed(String),
    #[error("Tokenization failed: {0}")]
    TokenizeFailed(String),
    #[error("Prompt produced zero tokens")]
    EmptyPrompt,
    #[error("Embedding lookup failed: {0}")]
    EmbedFailed(String),
    #[error("Proving failed: {0}")]
    ProveFailed(String),
}
