//! ZKML Transaction Classifier: verifiable AI guardrails for agent transactions.
//!
//! A small MLP classifier (64→64→32→3, 6339 parameters) scores transaction
//! intent as safe/suspicious/malicious. The inference is proven with GKR+STARK
//! under [`PolicyConfig::strict()`], producing a cryptographic proof that
//! verifies on-chain.
//!
//! # Architecture
//!
//! ```text
//! TransactionFeatures → encode_transaction() → M31Matrix (1×64)
//!                                                    ↓
//!                                          Classifier MLP forward pass
//!                                                    ↓
//!                                          GKR proof (PolicyConfig::strict)
//!                                                    ↓
//!                                   ClassifierResult { threat_score, decision, proof }
//!                                                    ↓
//!                                          On-chain: verify proof → approve/block
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use stwo_ml::classifier::{evaluate_transaction, build_test_classifier, TransactionFeatures};
//! use stwo_ml::policy::PolicyConfig;
//!
//! let model = build_test_classifier();
//! let policy = PolicyConfig::strict();
//! let tx = TransactionFeatures { /* ... */ };
//!
//! let result = evaluate_transaction(&tx, &model, &policy).unwrap();
//! println!("Decision: {} (score: {})", result.decision, result.threat_score);
//! ```
//!
//! # On-Chain Integration
//!
//! The classifier result contains:
//! - `io_commitment`: Poseidon hash of inputs + outputs (verified on-chain)
//! - `policy_commitment`: proves the strict policy was used
//! - `scores`: raw classifier output (extractable from `raw_io_data`)
//!
//! The `AgentFirewallZK` Cairo contract calls `is_proof_verified(proof_hash)`
//! on the ObelyskVerifier, then applies the proven threat score to
//! approve/escalate/block the agent's transaction.

pub mod encoder;
pub mod model;
pub mod prove;
pub mod types;

// Re-export key types for convenience
pub use encoder::encode_transaction;
pub use model::{build_classifier_graph, build_test_classifier, ClassifierModel};
pub use prove::evaluate_transaction;
pub use types::*;
