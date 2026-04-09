//! ObelyZK Virtual Machine — verifiable inference execution environment.
//!
//! The VM separates execution from proving:
//! - **Executor**: Runs the forward pass, captures a complete execution trace
//! - **Prover**: Takes a captured trace + weights, generates a GKR proof asynchronously
//! - **Queue**: Manages async proof jobs across multiple GPU workers
//!
//! This decoupling enables:
//! - Instant inference (user sees tokens immediately)
//! - Async proving (proofs arrive 30-60s later)
//! - Multi-GPU pipelining (chunks prove in parallel)
//! - Trace replay (re-prove from saved trace without re-executing)

pub mod trace;
pub mod executor;
pub mod prover;
pub mod queue;
