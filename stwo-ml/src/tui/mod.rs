//! Terminal UI for Obelysk proof visualization.
//!
//! Uses ratatui for beautiful terminal dashboards showing:
//! - Proof progress with live progress bars
//! - Audit report summary with commitment hashes
//! - On-chain verification status
//! - Tamper detection results
//! - VM runtime: GPU workers, proving queue, conversation sessions

#[cfg(feature = "tui")]
pub mod dashboard;

#[cfg(feature = "tui")]
pub mod vm_dashboard;
