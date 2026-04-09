//! Inference provider adapters for the ObelyZK VM.
//!
//! Each provider implements the `InferenceProvider` trait, enabling the VM
//! to route inference requests to different backends:
//!
//! - **Local**: Our M31 forward pass with full ZK proof
//! - **OpenAI-compatible**: Any vLLM, TGI, Ollama, or OpenAI-format API
//! - **Anthropic**: Claude API with TLS attestation
//!
//! All providers produce an `InferenceAttestation` regardless of backend.

pub mod types;
pub mod tls_attestation;
#[cfg(feature = "model-loading")]
pub mod local;
#[cfg(feature = "server")]
pub mod openai_compat;
#[cfg(feature = "server")]
pub mod anthropic;

pub use types::*;
