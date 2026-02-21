//! Verifiable Inference Audit System.
//!
//! Captures real inference into an append-only log, then batch-proves
//! correctness over a time window with optional semantic evaluation.
//!
//! # Architecture
//!
//! ```text
//! HOT PATH (serving)              AUDIT PATH (on-demand)
//! User → Server → Response        "Prove last hour" →
//!           │                              │
//!           v                    ┌─────────┼─────────┐
//!    Inference Log               v         v         v
//!    (append-only)          Summary   Prover    Evaluator
//!                           (instant) (GPU)     (parallel)
//!                               │         │         │
//!                               └─────────┼─────────┘
//!                                         v
//!                                   Audit Report
//!                                   → Encrypt → Arweave → On-chain
//! ```
//!
//! # Modules
//!
//! - [`types`] — Shared types (InferenceLogEntry, AuditReport, etc.)
//! - [`log`] — Append-only inference log with Merkle commitment (Dev A)
//! - [`capture`] — Non-blocking capture hook for model servers (Dev A)
//! - [`replay`] — Replay verification before proving (Dev A)
//! - [`prover`] — Batch audit prover over time windows (Dev A)
//! - [`submit`] — On-chain submission of audit records (Dev A)
//! - [`deterministic`] — Deterministic output checks (Dev B)
//! - [`self_eval`] — Self-evaluation via model forward pass (Dev B)
//! - [`scoring`] — Aggregate semantic scoring (Dev B)
//! - [`report`] — Report builder and hash computation (Dev B)
//! - [`storage`] — Arweave upload/download client (Dev B)
//! - [`encryption`] — Encryption integration with VM31 (Dev B)
//! - [`orchestrator`] — E2E audit orchestration (Dev B)

pub mod digest;
pub mod types;

// ── Dev A modules (inference log + proving) ──────────────────────────────
pub mod capture;
pub mod log;
pub mod prover;
pub mod replay;
pub mod submit;

// ── Dev B modules (evaluation + report + storage) ────────────────────────
pub mod deterministic;
pub mod encryption;
pub mod orchestrator;
pub mod report;
pub mod scoring;
pub mod self_eval;
pub mod storage;
