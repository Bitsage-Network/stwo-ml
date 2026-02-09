//! Error types for the Cairo adapter.
//!
//! This module provides error types for fallible operations in the adapter,
//! allowing callers to handle errors gracefully instead of panicking.

use thiserror::Error;

/// Error type for adapter operations.
#[derive(Error, Debug, Clone)]
pub enum AdapterError {
    /// Attempted to access an empty memory cell.
    #[error("Accessing empty memory cell at address {address}")]
    EmptyMemoryCell { address: u32 },

    /// Memory segment contains values when it should be empty.
    #[error("Memory segment not empty: address {address} has value ID {value_id}")]
    SegmentNotEmpty { address: u32, value_id: u32 },

    /// Invalid memory value tag encountered during decoding.
    #[error("Invalid memory value tag: {tag}")]
    InvalidTag { tag: u32 },

    /// Invalid builtin name encountered.
    #[error("Invalid builtin name: {name}")]
    InvalidBuiltinName { name: String },

    /// Builtin segment contains holes (missing values).
    #[error("Builtin segment '{builtin}' at index {segment_index} contains a hole")]
    BuiltinSegmentHasHole {
        builtin: String,
        segment_index: usize,
    },

    /// Builtin segment size is not divisible by cells per instance.
    #[error(
        "Builtin segment '{builtin}' size {size} is not divisible by {cells_per_instance}"
    )]
    InvalidBuiltinSegmentSize {
        builtin: String,
        size: usize,
        cells_per_instance: usize,
    },

    /// Adapter conversion failed.
    #[error("Adapter conversion failed: {0}")]
    ConversionError(String),
}

/// Result type alias for adapter operations.
pub type Result<T> = std::result::Result<T, AdapterError>;
