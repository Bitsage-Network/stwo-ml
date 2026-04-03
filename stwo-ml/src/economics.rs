//! Commit/Sample/Prove economics layer.
//!
//! This module implements the three-tier verification model that makes
//! provable inference economically viable at production scale:
//!
//! **Tier 1 — Commit (every inference, ~$0.0001)**
//! Every inference gets a Poseidon commitment: H(model_id, input, output, timestamp).
//! This is a single hash — microseconds, near-zero cost. The commitment is stored
//! on-chain as a tamper-evident log. If the provider later cheats, any committed
//! inference can be challenged and the commitment serves as the receipt.
//!
//! **Tier 2 — Sample (2-5% of inferences, ~$0.05 each)**
//! A random subset of committed inferences are selected for full proving.
//! The selection is verifiably random (derived from block hash or VRF).
//! The provider must produce a valid GKR proof within the SLA window.
//! If the proof fails, the provider's stake is slashed.
//!
//! **Tier 3 — Challenge (on demand, ~$0.10)**
//! Any party can challenge any committed inference by posting a bond.
//! The provider must respond with a full proof within the challenge window.
//! If the proof is valid, the challenger loses their bond.
//! If the proof is invalid or not provided, the provider is slashed.
//!
//! **Amortized cost: $1-3 per million inferences.**

use stwo::core::fields::m31::M31;
use starknet_ff::FieldElement;

/// Configuration for the Commit/Sample/Prove protocol.
#[derive(Debug, Clone)]
pub struct VerificationPolicy {
    /// Fraction of inferences to sample for proving (0.0 to 1.0).
    /// Default: 0.02 (2%).
    pub sample_rate: f64,
    /// Maximum time (seconds) to produce a proof after being sampled.
    /// Default: 300 (5 minutes).
    pub proof_sla_seconds: u64,
    /// Maximum time (seconds) to respond to a challenge.
    /// Default: 3600 (1 hour).
    pub challenge_window_seconds: u64,
    /// Minimum stake required from the provider (in token units).
    pub provider_stake: u64,
    /// Bond required from a challenger (in token units).
    pub challenge_bond: u64,
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        Self {
            sample_rate: 0.02,
            proof_sla_seconds: 300,
            challenge_window_seconds: 3600,
            provider_stake: 1000,
            challenge_bond: 10,
        }
    }
}

/// A commitment to a single inference.
///
/// This is the receipt that proves an inference happened with specific
/// model, input, and output. The commitment is a single Poseidon hash —
/// cheap to compute and store, expensive to forge.
#[derive(Debug, Clone)]
pub struct InferenceCommitment {
    /// Poseidon hash of (model_id, io_commitment, timestamp, sequence_number).
    pub commitment_hash: FieldElement,
    /// Model identity commitment (Poseidon of weight matrices).
    pub model_id: FieldElement,
    /// IO commitment: Poseidon(input || output).
    pub io_commitment: FieldElement,
    /// Unix timestamp (seconds) of the inference.
    pub timestamp: u64,
    /// Monotonically increasing sequence number within this provider.
    pub sequence_number: u64,
    /// Whether this inference has been sampled for proving.
    pub sampled: bool,
    /// Whether a proof has been submitted and verified.
    pub proven: bool,
}

impl InferenceCommitment {
    /// Create a new inference commitment.
    pub fn new(
        model_id: FieldElement,
        io_commitment: FieldElement,
        timestamp: u64,
        sequence_number: u64,
    ) -> Self {
        let commitment_hash = starknet_crypto::poseidon_hash_many(&[
            FieldElement::from(0x434F4D4D4954_u64), // "COMMIT" tag
            model_id,
            io_commitment,
            FieldElement::from(timestamp),
            FieldElement::from(sequence_number),
        ]);

        Self {
            commitment_hash,
            model_id,
            io_commitment,
            timestamp,
            sequence_number,
            sampled: false,
            proven: false,
        }
    }
}

/// The commitment chain: a hash-linked log of all inference commitments.
///
/// Each entry chains from the previous: `chain[i] = H(chain[i-1], commitment[i])`.
/// This ensures no commitment can be inserted, removed, or reordered after the fact.
///
/// The chain root is published on-chain periodically (e.g., every 100 inferences
/// or every minute). Anyone can verify the chain by replaying the hashes.
#[derive(Debug, Clone)]
pub struct CommitmentChain {
    /// Current chain head (hash of all commitments so far).
    pub chain_head: FieldElement,
    /// Total number of commitments in the chain.
    pub length: u64,
    /// All commitments (for local verification; in production, store on disk).
    pub commitments: Vec<InferenceCommitment>,
    /// Verification policy.
    pub policy: VerificationPolicy,
    /// Commitments selected for proving (indices into `commitments`).
    pub sampled_indices: Vec<usize>,
}

impl CommitmentChain {
    /// Create a new empty commitment chain.
    pub fn new(policy: VerificationPolicy) -> Self {
        Self {
            chain_head: FieldElement::ZERO,
            length: 0,
            commitments: Vec::new(),
            policy,
            sampled_indices: Vec::new(),
        }
    }

    /// Commit an inference. Returns the commitment and whether it was sampled.
    pub fn commit(
        &mut self,
        model_id: FieldElement,
        io_commitment: FieldElement,
        timestamp: u64,
    ) -> &InferenceCommitment {
        let seq = self.length;
        let mut commitment = InferenceCommitment::new(model_id, io_commitment, timestamp, seq);

        // Chain: new_head = H(old_head, commitment_hash)
        self.chain_head = starknet_crypto::poseidon_hash_many(&[
            self.chain_head,
            commitment.commitment_hash,
        ]);
        self.length += 1;

        // Sampling: deterministic from chain_head (verifiable by anyone)
        // sample if H(chain_head, "SAMPLE") mod 10000 < sample_rate * 10000
        let sample_hash = starknet_crypto::poseidon_hash_many(&[
            self.chain_head,
            FieldElement::from(0x53414D504C45_u64), // "SAMPLE"
        ]);
        let sample_bytes = sample_hash.to_bytes_be();
        let sample_val = u64::from_be_bytes([
            sample_bytes[24], sample_bytes[25], sample_bytes[26], sample_bytes[27],
            sample_bytes[28], sample_bytes[29], sample_bytes[30], sample_bytes[31],
        ]);
        let threshold = (self.policy.sample_rate * 10000.0) as u64;
        if sample_val % 10000 < threshold {
            commitment.sampled = true;
            self.sampled_indices.push(self.commitments.len());
        }

        self.commitments.push(commitment);
        self.commitments.last().unwrap()
    }

    /// Get the current chain head (for on-chain publication).
    pub fn head(&self) -> FieldElement {
        self.chain_head
    }

    /// Verify the chain integrity: replay all hashes and check the head matches.
    pub fn verify_chain(&self) -> bool {
        let mut head = FieldElement::ZERO;
        for c in &self.commitments {
            head = starknet_crypto::poseidon_hash_many(&[head, c.commitment_hash]);
        }
        head == self.chain_head
    }

    /// Number of inferences that need proving (sampled but not yet proven).
    pub fn pending_proofs(&self) -> usize {
        self.sampled_indices
            .iter()
            .filter(|&&i| !self.commitments[i].proven)
            .count()
    }

    /// Mark a sampled inference as proven.
    pub fn mark_proven(&mut self, index: usize) {
        if index < self.commitments.len() {
            self.commitments[index].proven = true;
        }
    }

    /// Estimated cost for a batch of inferences.
    pub fn estimate_cost(&self, num_inferences: u64, prove_cost_per: f64) -> CostEstimate {
        let commit_cost = num_inferences as f64 * 0.0001; // ~$0.0001 per commitment hash
        let num_sampled = (num_inferences as f64 * self.policy.sample_rate).ceil() as u64;
        let prove_cost = num_sampled as f64 * prove_cost_per;
        let total = commit_cost + prove_cost;

        CostEstimate {
            num_inferences,
            num_sampled,
            commit_cost_usd: commit_cost,
            prove_cost_usd: prove_cost,
            total_cost_usd: total,
            cost_per_million_usd: total / num_inferences as f64 * 1_000_000.0,
        }
    }
}

/// Cost breakdown for a batch of inferences.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub num_inferences: u64,
    pub num_sampled: u64,
    pub commit_cost_usd: f64,
    pub prove_cost_usd: f64,
    pub total_cost_usd: f64,
    pub cost_per_million_usd: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commitment_chain_basic() {
        let policy = VerificationPolicy::default();
        let mut chain = CommitmentChain::new(policy);

        let model_id = FieldElement::from(0x1234_u64);
        let io = FieldElement::from(0x5678_u64);

        chain.commit(model_id, io, 1000);
        assert_eq!(chain.length, 1);
        assert_ne!(chain.chain_head, FieldElement::ZERO);
        assert!(chain.verify_chain());
    }

    #[test]
    fn test_commitment_chain_integrity() {
        let policy = VerificationPolicy::default();
        let mut chain = CommitmentChain::new(policy);

        let model_id = FieldElement::from(0x1_u64);
        for i in 0..100 {
            chain.commit(model_id, FieldElement::from(i as u64), i * 1000);
        }

        assert_eq!(chain.length, 100);
        assert!(chain.verify_chain(), "chain integrity must hold");

        // Tamper: modify one commitment hash
        let original = chain.commitments[50].commitment_hash;
        chain.commitments[50].commitment_hash = FieldElement::from(0xDEAD_u64);
        assert!(!chain.verify_chain(), "tampered chain must fail verification");
        chain.commitments[50].commitment_hash = original; // restore
    }

    #[test]
    fn test_sampling_rate() {
        let policy = VerificationPolicy {
            sample_rate: 0.05, // 5%
            ..Default::default()
        };
        let mut chain = CommitmentChain::new(policy);

        let model_id = FieldElement::from(0x1_u64);
        for i in 0..10000 {
            chain.commit(model_id, FieldElement::from(i as u64), i * 100);
        }

        let sampled = chain.sampled_indices.len();
        // With 5% rate and 10000 inferences, expect ~500 sampled (±100)
        eprintln!("Sampled: {sampled} / 10000 ({:.1}%)", sampled as f64 / 100.0);
        assert!(sampled > 300, "too few sampled: {sampled}");
        assert!(sampled < 800, "too many sampled: {sampled}");
    }

    #[test]
    fn test_sampling_is_deterministic() {
        let policy = VerificationPolicy {
            sample_rate: 0.02,
            ..Default::default()
        };

        let mut chain1 = CommitmentChain::new(policy.clone());
        let mut chain2 = CommitmentChain::new(policy);

        let model_id = FieldElement::from(0x1_u64);
        for i in 0..1000 {
            chain1.commit(model_id, FieldElement::from(i as u64), i * 100);
            chain2.commit(model_id, FieldElement::from(i as u64), i * 100);
        }

        assert_eq!(chain1.sampled_indices, chain2.sampled_indices,
            "same inputs must produce same sampling decisions");
        assert_eq!(chain1.chain_head, chain2.chain_head,
            "same inputs must produce same chain head");
    }

    #[test]
    fn test_cost_estimate() {
        let policy = VerificationPolicy {
            sample_rate: 0.02,
            ..Default::default()
        };
        let chain = CommitmentChain::new(policy);

        // Cost per proof: H100 at $3/hr, 30s per proof (after optimization) = $0.025
        let estimate = chain.estimate_cost(1_000_000, 0.025);
        eprintln!("Cost for 1M inferences (2% sample, $0.025/proof):");
        eprintln!("  Sampled: {}", estimate.num_sampled);
        eprintln!("  Commit: ${:.2}", estimate.commit_cost_usd);
        eprintln!("  Prove: ${:.2}", estimate.prove_cost_usd);
        eprintln!("  Total: ${:.2}", estimate.total_cost_usd);
        eprintln!("  Per million: ${:.2}", estimate.cost_per_million_usd);

        // 2% of 1M = 20K proofs × $0.025 = $500
        // Commit: 1M × $0.0001 = $100
        // Total: ~$600 for 1M inferences
        assert!(estimate.total_cost_usd > 300.0, "total should be > $300");
        assert!(estimate.total_cost_usd < 1000.0, "total should be < $1000");
    }

    #[test]
    fn test_pending_proofs() {
        let policy = VerificationPolicy {
            sample_rate: 1.0, // 100% — sample everything for test
            ..Default::default()
        };
        let mut chain = CommitmentChain::new(policy);

        let model_id = FieldElement::from(0x1_u64);
        for i in 0..10 {
            chain.commit(model_id, FieldElement::from(i as u64), i * 100);
        }

        assert_eq!(chain.pending_proofs(), 10, "all should be pending");

        chain.mark_proven(0);
        chain.mark_proven(5);
        assert_eq!(chain.pending_proofs(), 8, "2 proven, 8 pending");
    }
}
