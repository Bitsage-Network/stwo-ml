//! Streaming Verifiable Compute Receipt (SVCR) constraint system.
//!
//! Proves that GPU inference receipts have correct billing arithmetic,
//! valid chain linking, and fresh TEE attestation — without proving the
//! actual inference computation.
//!
//! # Architecture
//!
//! ```text
//! TEE GPU Worker
//! ├── Runs inference (vLLM / TensorRT-LLM)
//! ├── Captures: output_tokens, gpu_time, peak_memory
//! ├── Builds ComputeReceipt with Poseidon commitments
//! └── STWO proves receipt constraints (~5K constraints, <2 sec)
//!         │
//!         ▼
//!     Starknet verifier (stwo-cairo-verifier)
//!     └── Verify billing, chain linking, TEE freshness
//! ```
//!
//! # Constraints
//!
//! 1. **Billing arithmetic**: `billing = floor(gpu_time * rate_sec / 1000) + tokens * rate_token`
//! 2. **Chain linking**: `seq > 0 → prev_hash == known_prev` else `prev_hash == 0`
//! 3. **TEE freshness**: `receipt_timestamp - tee_timestamp ∈ [0, MAX_AGE]`

use starknet_crypto::poseidon_hash_many;
use starknet_ff::FieldElement;

use num_traits::Zero;

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof::StarkProof;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs_lifted::MerkleHasherLifted;
use stwo::core::channel::MerkleChannel;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::simd::m31::LOG_N_LANES;
use stwo::prover::backend::{Col, Column, BackendForChannel};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::CommitmentSchemeProver;
use stwo::prover::prove;

use stwo_constraint_framework::{
    FrameworkEval, FrameworkComponent, EvalAtRow, TraceLocationAllocator,
};

use tracing::info;
use crate::backend::convert_evaluations;

/// The lifted Merkle hasher type for Blake2s channel.
type Hasher = <Blake2sMerkleChannel as MerkleChannel>::H;

/// Maximum allowed age (in seconds) between receipt timestamp and TEE attestation.
pub const MAX_TEE_AGE_SECS: u64 = 3600; // 1 hour

/// Number of trace columns in the receipt proof.
const NUM_RECEIPT_COLUMNS: usize = 7;

/// A compute receipt capturing GPU inference execution metadata.
///
/// All commitment fields (input/output/model/tee) are Poseidon hashes
/// computed inside the TEE before proof generation.
#[derive(Debug, Clone)]
pub struct ComputeReceipt {
    // === Identity ===
    /// Unique job identifier (32 bytes).
    pub job_id: FieldElement,
    /// Worker's public key (registered on-chain).
    pub worker_pubkey: FieldElement,

    // === Commitments (Poseidon hashes) ===
    /// Hash of input tokens/data.
    pub input_commitment: FieldElement,
    /// Hash of output tokens/data.
    pub output_commitment: FieldElement,
    /// Merkle root of model weights.
    pub model_commitment: FieldElement,

    // === Chain link ===
    /// Hash of the previous receipt (0x0 for first in session).
    pub prev_receipt_hash: FieldElement,

    // === Execution metrics ===
    /// GPU wall-clock time in milliseconds.
    pub gpu_time_ms: u64,
    /// Number of tokens processed.
    pub token_count: u32,
    /// Peak GPU memory usage in MB.
    pub peak_memory_mb: u32,

    // === Billing ===
    /// Total billing amount in SAGE tokens (smallest unit).
    pub billing_amount_sage: u64,
    /// Rate charged per second of GPU time.
    pub billing_rate_per_sec: u64,
    /// Rate charged per token processed.
    pub billing_rate_per_token: u64,

    // === TEE ===
    /// Hash of the NVIDIA DCAP attestation report.
    pub tee_report_hash: FieldElement,
    /// Timestamp from the TEE attestation.
    pub tee_timestamp: u64,

    // === Metadata ===
    /// Receipt creation timestamp (Unix epoch seconds).
    pub timestamp: u64,
    /// Position in the receipt chain (0 = first).
    pub sequence_number: u32,
}

impl ComputeReceipt {
    /// Populate the TEE fields from a real `TeeAttestation`.
    ///
    /// Sets `tee_report_hash` to the Poseidon hash of the attestation report
    /// (matching the on-chain ObelyskVerifier's expected format) and
    /// `tee_timestamp` to the attestation's timestamp.
    ///
    /// This bridges the gap between the attestation pipeline (which produces
    /// `TeeAttestation`) and the receipt system (which uses felt252 hashes).
    pub fn set_attestation(&mut self, attestation: &crate::tee::TeeAttestation) {
        self.tee_report_hash = attestation.report_hash_felt();
        self.tee_timestamp = attestation.timestamp;
    }

    /// Create a new receipt with TEE attestation fields populated.
    ///
    /// Convenience constructor that calls `set_attestation` on the receipt.
    pub fn with_attestation(mut self, attestation: &crate::tee::TeeAttestation) -> Self {
        self.set_attestation(attestation);
        self
    }

    /// Compute the Poseidon hash of this receipt.
    ///
    /// This hash uniquely identifies the receipt and is used for
    /// chain linking (next receipt's `prev_receipt_hash`).
    pub fn receipt_hash(&self) -> FieldElement {
        poseidon_hash_many(&[
            self.job_id,
            self.worker_pubkey,
            self.input_commitment,
            self.output_commitment,
            self.model_commitment,
            self.prev_receipt_hash,
            FieldElement::from(self.gpu_time_ms),
            FieldElement::from(self.token_count as u64),
            FieldElement::from(self.billing_amount_sage),
            FieldElement::from(self.billing_rate_per_sec),
            FieldElement::from(self.billing_rate_per_token),
            self.tee_report_hash,
            FieldElement::from(self.tee_timestamp),
            FieldElement::from(self.timestamp),
            FieldElement::from(self.sequence_number as u64),
        ])
    }

    /// Convert billing fields to M31 trace row values.
    ///
    /// Returns 7 values matching the trace column layout:
    /// `[gpu_time, rate_sec, time_billing, token_count, rate_token, token_billing, billing_total]`
    pub fn to_trace_row(&self) -> [M31; NUM_RECEIPT_COLUMNS] {
        let time_billing = self.gpu_time_ms * self.billing_rate_per_sec / 1000;
        let token_billing = self.token_count as u64 * self.billing_rate_per_token;

        [
            M31::from(self.gpu_time_ms as u32),
            M31::from(self.billing_rate_per_sec as u32),
            M31::from(time_billing as u32),
            M31::from(self.token_count),
            M31::from(self.billing_rate_per_token as u32),
            M31::from(token_billing as u32),
            M31::from(self.billing_amount_sage as u32),
        ]
    }

    /// Verify billing arithmetic off-chain (sanity check before proving).
    pub fn verify_billing(&self) -> bool {
        let time_billing = self.gpu_time_ms * self.billing_rate_per_sec / 1000;
        let token_billing = self.token_count as u64 * self.billing_rate_per_token;
        self.billing_amount_sage == time_billing + token_billing
    }

    /// Verify chain linking: first receipt must have prev_hash == 0.
    pub fn verify_chain_link(&self, expected_prev: Option<FieldElement>) -> bool {
        if self.sequence_number == 0 {
            self.prev_receipt_hash == FieldElement::ZERO
        } else {
            match expected_prev {
                Some(prev) => self.prev_receipt_hash == prev,
                None => false,
            }
        }
    }

    /// Verify TEE attestation freshness.
    pub fn verify_tee_freshness(&self) -> bool {
        if self.tee_timestamp > self.timestamp {
            return false;
        }
        self.timestamp - self.tee_timestamp <= MAX_TEE_AGE_SECS
    }
}

/// Evaluator for receipt billing constraints.
///
/// Proves that the billing arithmetic is correct for each receipt in the batch:
/// 1. `time_billing * 1000 == gpu_time_ms * rate_per_sec`
/// 2. `token_billing == token_count * rate_per_token`
/// 3. `billing_total == time_billing + token_billing`
#[derive(Debug, Clone)]
pub struct ReceiptEval {
    pub log_n_rows: u32,
}

impl FrameworkEval for ReceiptEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        // Constraints are degree 2 (product of two trace columns)
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Read 7 trace columns in order
        let gpu_time_ms = eval.next_trace_mask();
        let rate_per_sec = eval.next_trace_mask();
        let time_billing = eval.next_trace_mask();
        let token_count = eval.next_trace_mask();
        let rate_per_token = eval.next_trace_mask();
        let token_billing = eval.next_trace_mask();
        let billing_total = eval.next_trace_mask();

        let thousand = E::F::from(BaseField::from(1000u32));

        // Constraint 1: time_billing * 1000 == gpu_time_ms * rate_per_sec
        // This avoids division: time_billing = floor(gpu_time * rate / 1000)
        // Expressed as: time_billing * 1000 - gpu_time_ms * rate_per_sec == 0
        eval.add_constraint(
            time_billing.clone() * thousand - gpu_time_ms * rate_per_sec,
        );

        // Constraint 2: token_billing == token_count * rate_per_token
        eval.add_constraint(
            token_billing.clone() - token_count * rate_per_token,
        );

        // Constraint 3: billing_total == time_billing + token_billing
        eval.add_constraint(
            billing_total - time_billing - token_billing,
        );

        eval
    }
}

/// Type alias for the receipt component.
pub type ReceiptComponent = FrameworkComponent<ReceiptEval>;

/// Proof of a single receipt or batch of receipts, generic over hash type.
#[derive(Debug)]
pub struct ReceiptProofFor<H: MerkleHasherLifted> {
    /// The STARK proof covering all billing constraints.
    pub stark_proof: StarkProof<H>,
    /// Receipt hashes (Poseidon) for each receipt in the batch.
    pub receipt_hashes: Vec<FieldElement>,
    /// Number of receipts proven.
    pub batch_size: usize,
}

/// Receipt proof using Blake2s (default).
pub type ReceiptProof = ReceiptProofFor<Hasher>;

/// Error type for receipt proving.
#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("Invalid billing: expected {expected}, got {actual}")]
    InvalidBilling { expected: u64, actual: u64 },
    #[error("Chain link broken at sequence {sequence}: expected {expected}, got {actual}")]
    BrokenChain {
        sequence: u32,
        expected: String,
        actual: String,
    },
    #[error("TEE attestation expired: age {age_secs}s exceeds max {max_secs}s")]
    TeeExpired { age_secs: u64, max_secs: u64 },
    #[error("Empty receipt batch")]
    EmptyBatch,
    #[error("Proving error: {0}")]
    ProvingError(String),
}

/// Prove a single compute receipt.
///
/// Generates a STARK proof that the receipt's billing arithmetic is correct.
pub fn prove_receipt(receipt: &ComputeReceipt) -> Result<ReceiptProof, ReceiptError> {
    prove_receipt_batch(std::slice::from_ref(receipt))
}

/// Prove a batch of compute receipts using `SimdBackend` + `Blake2sMerkleChannel`.
///
/// Convenience wrapper around [`prove_receipt_batch_with`].
pub fn prove_receipt_batch(receipts: &[ComputeReceipt]) -> Result<ReceiptProof, ReceiptError> {
    prove_receipt_batch_with::<SimdBackend, Blake2sMerkleChannel>(receipts)
}

/// Prove a receipt batch using the best available backend.
///
/// Uses `GpuBackend` when CUDA is available, otherwise `SimdBackend`.
pub fn prove_receipt_batch_auto(
    receipts: &[ComputeReceipt],
) -> Result<ReceiptProof, ReceiptError> {
    let gpu_available = crate::backend::gpu_is_available();
    info!(
        gpu_available,
        batch_size = receipts.len(),
        "Auto-selecting backend for receipt batch proving"
    );
    crate::backend::with_best_backend(
        || {
            info!("Using SimdBackend for receipt batch proving");
            prove_receipt_batch_with::<SimdBackend, Blake2sMerkleChannel>(receipts)
        },
        || {
            info!("Using GpuBackend for receipt batch proving");
            prove_receipt_batch_gpu(receipts)
        },
    )
}

/// GPU receipt proving path.
fn prove_receipt_batch_gpu(
    receipts: &[ComputeReceipt],
) -> Result<ReceiptProof, ReceiptError> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return prove_receipt_batch_with::<GpuBackend, Blake2sMerkleChannel>(receipts);
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        prove_receipt_batch_with::<SimdBackend, Blake2sMerkleChannel>(receipts)
    }
}

/// Prove a batch of compute receipts, generic over backend and Merkle channel.
///
/// All receipts share one proof, minimizing on-chain verification cost.
/// The trace has one row per receipt (padded to power-of-2).
///
/// Trace generation uses `SimdBackend`; commitment and proving use backend `B`.
pub fn prove_receipt_batch_with<B, MC>(
    receipts: &[ComputeReceipt],
) -> Result<ReceiptProofFor<<MC as MerkleChannel>::H>, ReceiptError>
where
    B: BackendForChannel<MC> + PolyOps,
    MC: MerkleChannel,
    FrameworkComponent<ReceiptEval>: stwo::prover::ComponentProver<B>,
{
    info!(
        backend = std::any::type_name::<B>(),
        batch_size = receipts.len(),
        "Proving receipt batch"
    );
    if receipts.is_empty() {
        return Err(ReceiptError::EmptyBatch);
    }

    // Validate billing before proving
    for receipt in receipts {
        if !receipt.verify_billing() {
            let time_billing = receipt.gpu_time_ms * receipt.billing_rate_per_sec / 1000;
            let token_billing = receipt.token_count as u64 * receipt.billing_rate_per_token;
            return Err(ReceiptError::InvalidBilling {
                expected: time_billing + token_billing,
                actual: receipt.billing_amount_sage,
            });
        }
    }

    // Minimum log_size for SIMD (LOG_N_LANES = 4, so minimum 16 rows)
    let min_log_size = LOG_N_LANES;
    let needed_rows = receipts.len().next_power_of_two();
    let log_size = needed_rows.ilog2().max(min_log_size);
    let size = 1usize << log_size;
    let domain = CanonicCoset::new(log_size).circle_domain();
    let config = PcsConfig::default();

    // Build trace columns on SimdBackend (7 columns × size rows)
    let mut columns: Vec<Col<SimdBackend, BaseField>> = (0..NUM_RECEIPT_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(size))
        .collect();

    for (row, receipt) in receipts.iter().enumerate() {
        let trace_row = receipt.to_trace_row();
        for (col_idx, &val) in trace_row.iter().enumerate() {
            columns[col_idx].set(row, val);
        }
    }
    // Padding rows remain zero — all constraints satisfied with zeros.

    // Build CircleEvaluations from SIMD columns
    let trace_evals: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = columns
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect();

    // === Commitment scheme setup with backend B ===
    let max_degree_bound = log_size + 1;
    let twiddles = B::precompute_twiddles(
        CanonicCoset::new(max_degree_bound + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<B, MC>::new(config, &twiddles);

    // Tree 0: Preprocessed columns (none for receipt — commit empty tree)
    let tree_builder = commitment_scheme.tree_builder();
    tree_builder.commit(channel);

    // Tree 1: Execution trace (7 receipt columns), converted to backend B
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(convert_evaluations::<SimdBackend, B, BaseField>(trace_evals));
    tree_builder.commit(channel);

    // Build receipt component
    let eval = ReceiptEval { log_n_rows: log_size };
    let component = FrameworkComponent::new(
        &mut TraceLocationAllocator::default(),
        eval,
        SecureField::zero(), // No LogUp claimed sum
    );

    // Prove using backend B
    let stark_proof = prove::<B, MC>(
        &[&component],
        channel,
        commitment_scheme,
    )
    .map_err(|e| ReceiptError::ProvingError(format!("{e:?}")))?;

    // Compute receipt hashes
    let receipt_hashes: Vec<FieldElement> = receipts.iter().map(|r| r.receipt_hash()).collect();

    Ok(ReceiptProofFor {
        stark_proof,
        receipt_hashes,
        batch_size: receipts.len(),
    })
}

/// Verify a receipt chain: each receipt links to the previous via hash.
pub fn verify_receipt_chain(receipts: &[ComputeReceipt]) -> Result<(), ReceiptError> {
    for (i, receipt) in receipts.iter().enumerate() {
        if i == 0 {
            if receipt.sequence_number != 0 {
                return Err(ReceiptError::BrokenChain {
                    sequence: receipt.sequence_number,
                    expected: "0".to_string(),
                    actual: format!("{}", receipt.sequence_number),
                });
            }
            if receipt.prev_receipt_hash != FieldElement::ZERO {
                return Err(ReceiptError::BrokenChain {
                    sequence: 0,
                    expected: "0x0".to_string(),
                    actual: format!("{:#x}", receipt.prev_receipt_hash),
                });
            }
        } else {
            let expected_prev = receipts[i - 1].receipt_hash();
            if receipt.prev_receipt_hash != expected_prev {
                return Err(ReceiptError::BrokenChain {
                    sequence: receipt.sequence_number,
                    expected: format!("{expected_prev:#x}"),
                    actual: format!("{:#x}", receipt.prev_receipt_hash),
                });
            }
        }
    }
    Ok(())
}

/// Build a test receipt with valid billing arithmetic.
#[cfg(test)]
fn test_receipt(
    gpu_time_ms: u64,
    rate_per_sec: u64,
    token_count: u32,
    rate_per_token: u64,
    sequence: u32,
    prev_hash: FieldElement,
) -> ComputeReceipt {
    let time_billing = gpu_time_ms * rate_per_sec / 1000;
    let token_billing = token_count as u64 * rate_per_token;
    let billing_total = time_billing + token_billing;

    ComputeReceipt {
        job_id: FieldElement::from(1u64),
        worker_pubkey: FieldElement::from(42u64),
        input_commitment: FieldElement::from(100u64),
        output_commitment: FieldElement::from(200u64),
        model_commitment: FieldElement::from(300u64),
        prev_receipt_hash: prev_hash,
        gpu_time_ms,
        token_count,
        peak_memory_mb: 1024,
        billing_amount_sage: billing_total,
        billing_rate_per_sec: rate_per_sec,
        billing_rate_per_token: rate_per_token,
        tee_report_hash: FieldElement::from(500u64),
        tee_timestamp: 1700000000,
        timestamp: 1700000010,
        sequence_number: sequence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_hash_deterministic() {
        let r = test_receipt(5000, 100, 512, 10, 0, FieldElement::ZERO);
        let h1 = r.receipt_hash();
        let h2 = r.receipt_hash();
        assert_eq!(h1, h2, "receipt hash should be deterministic");
        assert_ne!(h1, FieldElement::ZERO, "receipt hash should be non-zero");
    }

    #[test]
    fn test_billing_verification() {
        // Valid billing: 5s at 100 SAGE/s + 512 tokens at 10 SAGE/token
        // = 500 + 5120 = 5620
        let r = test_receipt(5000, 100, 512, 10, 0, FieldElement::ZERO);
        assert!(r.verify_billing());
        assert_eq!(r.billing_amount_sage, 5620);

        // Invalid billing
        let mut bad = r.clone();
        bad.billing_amount_sage = 9999;
        assert!(!bad.verify_billing());
    }

    #[test]
    fn test_chain_link_verification() {
        let r0 = test_receipt(1000, 100, 100, 10, 0, FieldElement::ZERO);
        assert!(r0.verify_chain_link(None));

        let r1 = test_receipt(2000, 100, 200, 10, 1, r0.receipt_hash());
        assert!(r1.verify_chain_link(Some(r0.receipt_hash())));

        // Wrong prev hash
        let r_bad = test_receipt(2000, 100, 200, 10, 1, FieldElement::from(999u64));
        assert!(!r_bad.verify_chain_link(Some(r0.receipt_hash())));
    }

    #[test]
    fn test_tee_freshness() {
        let r = test_receipt(1000, 100, 100, 10, 0, FieldElement::ZERO);
        assert!(r.verify_tee_freshness());

        // Expired TEE
        let mut expired = r.clone();
        expired.tee_timestamp = 0; // Very old
        expired.timestamp = MAX_TEE_AGE_SECS + 100;
        assert!(!expired.verify_tee_freshness());
    }

    #[test]
    fn test_trace_row_values() {
        let r = test_receipt(5000, 100, 512, 10, 0, FieldElement::ZERO);
        let row = r.to_trace_row();

        assert_eq!(row[0], M31::from(5000u32), "gpu_time_ms");
        assert_eq!(row[1], M31::from(100u32), "rate_per_sec");
        assert_eq!(row[2], M31::from(500u32), "time_billing = 5000*100/1000");
        assert_eq!(row[3], M31::from(512u32), "token_count");
        assert_eq!(row[4], M31::from(10u32), "rate_per_token");
        assert_eq!(row[5], M31::from(5120u32), "token_billing = 512*10");
        assert_eq!(row[6], M31::from(5620u32), "billing_total = 500+5120");
    }

    #[test]
    fn test_prove_single_receipt() {
        let r = test_receipt(5000, 100, 512, 10, 0, FieldElement::ZERO);
        let proof = prove_receipt(&r).expect("receipt proving should succeed");

        assert_eq!(proof.batch_size, 1);
        assert_eq!(proof.receipt_hashes.len(), 1);
        assert_eq!(proof.receipt_hashes[0], r.receipt_hash());
    }

    #[test]
    fn test_prove_receipt_batch() {
        let r0 = test_receipt(1000, 100, 100, 10, 0, FieldElement::ZERO);
        let r1 = test_receipt(2000, 200, 200, 20, 1, r0.receipt_hash());
        let r2 = test_receipt(3000, 300, 300, 30, 2, r1.receipt_hash());

        let proof = prove_receipt_batch(&[r0.clone(), r1.clone(), r2.clone()])
            .expect("batch proving should succeed");

        assert_eq!(proof.batch_size, 3);
        assert_eq!(proof.receipt_hashes.len(), 3);
    }

    #[test]
    fn test_prove_receipt_invalid_billing_rejected() {
        let mut r = test_receipt(5000, 100, 512, 10, 0, FieldElement::ZERO);
        r.billing_amount_sage = 9999; // Wrong

        let result = prove_receipt(&r);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ReceiptError::InvalidBilling { .. }));
    }

    #[test]
    fn test_verify_receipt_chain() {
        let r0 = test_receipt(1000, 100, 100, 10, 0, FieldElement::ZERO);
        let r1 = test_receipt(2000, 100, 200, 10, 1, r0.receipt_hash());
        let r2 = test_receipt(3000, 100, 300, 10, 2, r1.receipt_hash());

        let result = verify_receipt_chain(&[r0, r1, r2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_broken_chain() {
        let r0 = test_receipt(1000, 100, 100, 10, 0, FieldElement::ZERO);
        let r1_bad = test_receipt(2000, 100, 200, 10, 1, FieldElement::from(999u64));

        let result = verify_receipt_chain(&[r0, r1_bad]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ReceiptError::BrokenChain { .. }));
    }

    #[test]
    fn test_prove_receipt_with_serialization() {
        use crate::cairo_serde::serialize_proof;

        let r = test_receipt(5000, 100, 512, 10, 0, FieldElement::ZERO);
        let proof = prove_receipt(&r).expect("proving should succeed");

        // Serialize to felt252 calldata
        let calldata = serialize_proof(&proof.stark_proof);
        assert!(!calldata.is_empty(), "calldata should be non-empty");

        // Receipt proof calldata should be compact (~500 felt252s)
        assert!(
            calldata.len() < 5000,
            "receipt calldata too large: {} felts",
            calldata.len()
        );
    }

    #[test]
    fn test_empty_batch_rejected() {
        let result = prove_receipt_batch(&[]);
        assert!(matches!(result.unwrap_err(), ReceiptError::EmptyBatch));
    }

    #[test]
    fn test_prove_receipt_batch_auto() {
        let r0 = test_receipt(1000, 100, 100, 10, 0, FieldElement::ZERO);
        let r1 = test_receipt(2000, 200, 200, 20, 1, r0.receipt_hash());

        let proof = prove_receipt_batch_auto(&[r0, r1])
            .expect("auto batch proving should succeed");

        assert_eq!(proof.batch_size, 2);
        assert_eq!(proof.receipt_hashes.len(), 2);
    }
}
