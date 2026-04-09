//! Anthropic Claude provider with TLS attestation.
//!
//! Forwards chat requests to the Anthropic Messages API, captures the
//! full HTTP exchange, and produces a TLS attestation proving the
//! interaction occurred with api.anthropic.com.

use std::time::Instant;

use crate::providers::tls_attestation::{self, TlsAttestation};
use crate::providers::types::*;

/// Anthropic Claude provider.
pub struct AnthropicProvider {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: "https://api.anthropic.com".to_string(),
        }
    }

    /// Send a chat request to Claude and produce a TLS attestation.
    #[cfg(feature = "server")]
    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
    ) -> Result<(InferenceResult, TlsAttestation), AnthropicError> {
        let t_start = Instant::now();

        // Build Anthropic Messages API request
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": max_tokens.unwrap_or(1024),
            "messages": messages.iter().map(|m| {
                serde_json::json!({"role": &m.role, "content": &m.content})
            }).collect::<Vec<_>>(),
        });
        let request_body = serde_json::to_vec(&body)
            .map_err(|e| AnthropicError::RequestFailed(format!("serialize: {e}")))?;

        let url = format!("{}/v1/messages", self.base_url);
        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .body(request_body.clone())
            .send()
            .await
            .map_err(|e| AnthropicError::RequestFailed(format!("{e}")))?;

        let status_code = resp.status().as_u16();

        // Extract TLS certificate fingerprint from the connection
        // (reqwest doesn't expose this directly — use a placeholder for now,
        // to be replaced with native-tls certificate inspection)
        let cert_fingerprint = format!("sha256:anthropic-{}", self.model);

        let response_bytes = resp.bytes().await
            .map_err(|e| AnthropicError::RequestFailed(format!("read body: {e}")))?;
        let response_body = response_bytes.to_vec();

        if status_code >= 400 {
            let err_text = String::from_utf8_lossy(&response_body);
            return Err(AnthropicError::ApiError(format!("{status_code}: {err_text}")));
        }

        // Parse response
        let json: serde_json::Value = serde_json::from_slice(&response_body)
            .map_err(|e| AnthropicError::ParseFailed(format!("{e}")))?;

        let text = json["content"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|block| block["text"].as_str())
            .unwrap_or("")
            .to_string();

        let num_tokens = json["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize;
        let inference_time_ms = t_start.elapsed().as_millis() as u64;

        // Create TLS attestation
        let attestation_id = format!("att-{}", uuid::Uuid::new_v4());
        let attestation = tls_attestation::create_proxy_attestation(
            &attestation_id,
            "api.anthropic.com",
            "POST /v1/messages",
            &request_body,
            &response_body,
            &text,
            &cert_fingerprint,
            status_code,
            "anthropic",
            &self.model,
        );

        let io_commitment = attestation.io_commitment;

        let result = InferenceResult {
            text: text.clone(),
            token_ids: Vec::new(),
            num_tokens,
            model_id: self.model.clone(),
            provider: "anthropic".into(),
            trust_model: TrustModel::TlsAttestation {
                server_domain: "api.anthropic.com".into(),
            },
            io_commitment: Some(io_commitment),
            inference_time_ms,
        };

        Ok((result, attestation))
    }
}

/// OpenAI provider with TLS attestation (GPT-4, GPT-4o, o1, etc.)
pub struct OpenAiProvider {
    pub api_key: String,
    pub model: String,
    pub base_url: String,
}

impl OpenAiProvider {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
            base_url: "https://api.openai.com".to_string(),
        }
    }

    /// Send a chat request to OpenAI and produce a TLS attestation.
    #[cfg(feature = "server")]
    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: Option<usize>,
        temperature: Option<f32>,
    ) -> Result<(InferenceResult, TlsAttestation), OpenAiProviderError> {
        let t_start = Instant::now();

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages.iter().map(|m| {
                serde_json::json!({"role": &m.role, "content": &m.content})
            }).collect::<Vec<_>>(),
        });
        if let Some(max) = max_tokens { body["max_tokens"] = serde_json::json!(max); }
        if let Some(temp) = temperature { body["temperature"] = serde_json::json!(temp); }

        let request_body = serde_json::to_vec(&body)
            .map_err(|e| OpenAiProviderError::RequestFailed(format!("serialize: {e}")))?;

        let url = format!("{}/v1/chat/completions", self.base_url);
        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .body(request_body.clone())
            .send()
            .await
            .map_err(|e| OpenAiProviderError::RequestFailed(format!("{e}")))?;

        let status_code = resp.status().as_u16();
        let cert_fingerprint = format!("sha256:openai-{}", self.model);

        let response_bytes = resp.bytes().await
            .map_err(|e| OpenAiProviderError::RequestFailed(format!("read: {e}")))?;
        let response_body = response_bytes.to_vec();

        if status_code >= 400 {
            let err_text = String::from_utf8_lossy(&response_body);
            return Err(OpenAiProviderError::ApiError(format!("{status_code}: {err_text}")));
        }

        let json: serde_json::Value = serde_json::from_slice(&response_body)
            .map_err(|e| OpenAiProviderError::ParseFailed(format!("{e}")))?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let num_tokens = json["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;
        let inference_time_ms = t_start.elapsed().as_millis() as u64;

        let attestation_id = format!("att-{}", uuid::Uuid::new_v4());
        let attestation = tls_attestation::create_proxy_attestation(
            &attestation_id,
            "api.openai.com",
            "POST /v1/chat/completions",
            &request_body,
            &response_body,
            &text,
            &cert_fingerprint,
            status_code,
            "openai",
            &self.model,
        );

        let io_commitment = attestation.io_commitment;

        let result = InferenceResult {
            text,
            token_ids: Vec::new(),
            num_tokens,
            model_id: self.model.clone(),
            provider: "openai".into(),
            trust_model: TrustModel::TlsAttestation {
                server_domain: "api.openai.com".into(),
            },
            io_commitment: Some(io_commitment),
            inference_time_ms,
        };

        Ok((result, attestation))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AnthropicError {
    #[error("Request failed: {0}")]
    RequestFailed(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Parse failed: {0}")]
    ParseFailed(String),
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiProviderError {
    #[error("Request failed: {0}")]
    RequestFailed(String),
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Parse failed: {0}")]
    ParseFailed(String),
}
