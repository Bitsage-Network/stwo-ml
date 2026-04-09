//! OpenAI-compatible provider — forwards to any vLLM/TGI/Ollama/OpenAI API.
//!
//! This provider:
//! 1. Forwards the chat completion request to an upstream API
//! 2. Captures the response tokens
//! 3. Computes a commitment over (prompt, response)
//! 4. Optionally triggers M31 replay proving for open-weight models
//!
//! Trust model: CommitmentOnly (unless paired with a local replay prover).

use std::time::Instant;

use crate::providers::types::*;

/// OpenAI-compatible inference provider.
pub struct OpenAiCompatProvider {
    /// Upstream API base URL (e.g., "http://localhost:8000/v1").
    pub base_url: String,
    /// API key for the upstream (optional).
    pub api_key: Option<String>,
    /// Model name to send in the request.
    pub model: String,
    /// Provider label for attestation records.
    pub provider_name: String,
}

impl OpenAiCompatProvider {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: None,
            model: model.to_string(),
            provider_name: "openai-compat".into(),
        }
    }

    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    pub fn with_provider_name(mut self, name: &str) -> Self {
        self.provider_name = name.to_string();
        self
    }

    /// Send a chat completion request and capture the response.
    #[cfg(feature = "server")]
    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    ) -> Result<InferenceResult, OpenAiError> {
        let t_start = Instant::now();

        // Build request body
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages.iter().map(|m| {
                serde_json::json!({"role": &m.role, "content": &m.content})
            }).collect::<Vec<_>>(),
        });
        if let Some(max) = max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }
        if let Some(temp) = temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        let url = format!("{}/chat/completions", self.base_url);
        let client = reqwest::Client::new();
        let mut req = client.post(&url)
            .header("Content-Type", "application/json")
            .json(&body);

        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let resp = req.send().await
            .map_err(|e| OpenAiError::RequestFailed(format!("{e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(OpenAiError::ApiError(format!("{status}: {body}")));
        }

        let json: serde_json::Value = resp.json().await
            .map_err(|e| OpenAiError::ParseFailed(format!("{e}")))?;

        // Extract response text
        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let inference_time_ms = t_start.elapsed().as_millis() as u64;

        // Compute IO commitment
        let prompt_text = messages.iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");
        let io_commitment = compute_text_io_commitment(&prompt_text, &text);

        Ok(InferenceResult {
            text,
            token_ids: Vec::new(), // OpenAI API doesn't always return token IDs
            num_tokens: json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize,
            model_id: self.model.clone(),
            provider: self.provider_name.clone(),
            trust_model: TrustModel::CommitmentOnly,
            io_commitment: Some(io_commitment),
            inference_time_ms,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiError {
    #[error("Request failed: {0}")]
    RequestFailed(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Parse failed: {0}")]
    ParseFailed(String),
}
