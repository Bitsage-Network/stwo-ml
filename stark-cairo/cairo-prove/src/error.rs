//! Error types for Cairo proving and verification.
//!
//! This module provides comprehensive error handling for the Cairo proof pipeline,
//! replacing panics with proper Result-based error propagation.

use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// Main error type for Cairo proving operations.
#[derive(Error, Debug)]
pub enum CairoProveError {
    /// Error reading or writing files.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Error parsing JSON (executable or proof).
    #[error("JSON parsing error: {0}")]
    JsonParse(#[from] serde_json::Error),

    /// Error during Cairo program execution.
    #[error("Execution error: {0}")]
    Execution(String),

    /// Error converting path to string.
    #[error("Invalid path: {path:?} - path contains invalid UTF-8")]
    InvalidPath { path: PathBuf },

    /// Error finding entrypoint in executable.
    #[error("No standalone entrypoint found in executable")]
    NoEntrypoint,

    /// Error creating Cairo program.
    #[error("Failed to create Cairo program: {0}")]
    ProgramCreation(String),

    /// Error extracting trace from runner.
    #[error("Failed to extract trace: relocated trace not available")]
    NoRelocatedTrace,

    /// Error getting public input from runner.
    #[error("Failed to get public input: {0}")]
    PublicInputError(String),

    /// Error adapting input to STWO format.
    #[error("Adapter error: {0}")]
    AdapterError(String),

    /// Error during proof generation.
    #[error("Proof generation failed: {0}")]
    ProofGeneration(String),

    /// Error during proof verification.
    #[error("Verification failed: {0}")]
    Verification(String),

    /// Error serializing proof.
    #[error("Proof serialization failed: {0}")]
    ProofSerialization(String),
}

/// Result type alias for Cairo prove operations.
pub type Result<T> = std::result::Result<T, CairoProveError>;

impl CairoProveError {
    /// Create an execution error from any error type.
    pub fn execution<E: std::fmt::Display>(err: E) -> Self {
        CairoProveError::Execution(err.to_string())
    }

    /// Create an adapter error from any error type.
    pub fn adapter<E: std::fmt::Display>(err: E) -> Self {
        CairoProveError::AdapterError(err.to_string())
    }

    /// Create a proof generation error from any error type.
    pub fn proof<E: std::fmt::Display>(err: E) -> Self {
        CairoProveError::ProofGeneration(err.to_string())
    }

    /// Create a verification error from any error type.
    pub fn verification<E: std::fmt::Display>(err: E) -> Self {
        CairoProveError::Verification(err.to_string())
    }
}
