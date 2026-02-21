//! VM31 relayer flow orchestration for on-chain batch settlement.
//!
//! Production sequence:
//! 1. Verify proof hash is marked verified in verifier
//! 2. Bind VM31 public-input hash to proof hash in verifier
//! 3. Submit batch to pool
//! 4. Apply chunks until processed == total
//! 5. Finalize batch
//!
//! Each step is idempotent and retried with bounded backoff.

use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

use starknet_crypto::poseidon_hash_many;
use starknet_ff::FieldElement;
use stwo::core::fields::m31::BaseField as M31;

use crate::circuits::batch::BatchPublicInputs;
use crate::crypto::poseidon2_m31::{poseidon2_hash, RATE};

const M31_P: u64 = 0x7FFF_FFFF;
const WITHDRAW_BINDING_DOMAIN_V1_HEX: &str = "0x564D33315F5744525F42494E445F5631";
const WITHDRAW_BINDING_DOMAIN_V2_HEX: &str = "0x564D33315F5744525F42494E445F5632";

#[derive(Debug, thiserror::Error)]
pub enum RelayerError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("proof hash is not verified: {0}")]
    ProofNotVerified(String),
    #[error("batch hash mismatch for proof {proof_hash}: expected ({expected_lo}, {expected_hi}), got ({actual_lo}, {actual_hi})")]
    BatchHashMismatch {
        proof_hash: String,
        expected_lo: String,
        expected_hi: String,
        actual_lo: String,
        actual_hi: String,
    },
    #[error("unexpected state: {0}")]
    UnexpectedState(String),
    #[error("step '{step}' failed after {attempts} attempts: {last_error}")]
    RetriesExhausted {
        step: String,
        attempts: u32,
        last_error: String,
    },
}

#[derive(Clone, Debug)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff: Duration::from_secs(2),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Vm31RelayerConfig {
    pub chunk_size: u32,
    pub retries: RetryPolicy,
}

impl Default for Vm31RelayerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 32,
            retries: RetryPolicy::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RelayOutcome {
    pub proof_hash: String,
    pub batch_id: String,
    pub batch_hash: [M31; RATE],
    pub total_txs: u32,
    pub processed_txs: u32,
    pub finalized: bool,
}

#[derive(Clone, Debug)]
pub struct WithdrawalRecipients {
    pub payout: Vec<String>,
    pub credit: Vec<String>,
}

impl WithdrawalRecipients {
    pub fn new(payout: Vec<String>, credit: Vec<String>) -> Self {
        Self { payout, credit }
    }

    // Legacy helper for migration flows where payout and credit were the same recipient.
    pub fn mirrored(recipients: Vec<String>) -> Self {
        Self {
            payout: recipients.clone(),
            credit: recipients,
        }
    }
}

pub trait Vm31RelayerBackend {
    fn is_proof_verified(&self, proof_hash: &str) -> Result<bool, RelayerError>;
    fn get_vm31_public_hash(
        &self,
        proof_hash: &str,
    ) -> Result<Option<(String, String)>, RelayerError>;
    fn bind_vm31_public_hash(
        &self,
        proof_hash: &str,
        batch_hash_lo: &str,
        batch_hash_hi: &str,
    ) -> Result<(), RelayerError>;

    fn get_batch_for_proof_hash(&self, proof_hash: &str) -> Result<Option<String>, RelayerError>;
    fn submit_batch_proof(
        &self,
        inputs: &BatchPublicInputs,
        proof_hash: &str,
        recipients: &WithdrawalRecipients,
    ) -> Result<(), RelayerError>;

    fn get_batch_status(&self, batch_id: &str) -> Result<u8, RelayerError>;
    fn get_batch_total_txs(&self, batch_id: &str) -> Result<u32, RelayerError>;
    fn get_batch_processed_count(&self, batch_id: &str) -> Result<u32, RelayerError>;
    fn apply_batch_chunk(&self, batch_id: &str, start: u32, count: u32)
        -> Result<(), RelayerError>;
    fn finalize_batch(&self, batch_id: &str) -> Result<(), RelayerError>;
}

/// `sncast`-based backend for production relayer execution.
#[derive(Clone, Debug)]
pub struct SncastVm31Backend {
    pub account: String,
    pub rpc_url: String,
    pub verifier_contract: String,
    pub pool_contract: String,
}

impl SncastVm31Backend {
    pub fn new(
        account: impl Into<String>,
        rpc_url: impl Into<String>,
        verifier_contract: impl Into<String>,
        pool_contract: impl Into<String>,
    ) -> Self {
        Self {
            account: account.into(),
            rpc_url: rpc_url.into(),
            verifier_contract: normalize_hex(&verifier_contract.into()),
            pool_contract: normalize_hex(&pool_contract.into()),
        }
    }

    fn sncast_call(
        &self,
        contract: &str,
        function: &str,
        calldata: &[String],
    ) -> Result<Vec<String>, RelayerError> {
        let mut cmd = Command::new("sncast");
        cmd.arg("--account")
            .arg(&self.account)
            .arg("--url")
            .arg(&self.rpc_url)
            .arg("call")
            .arg("--contract-address")
            .arg(contract)
            .arg("--function")
            .arg(function);
        if !calldata.is_empty() {
            cmd.arg("--calldata");
            for arg in calldata {
                cmd.arg(arg);
            }
        }
        let output = cmd
            .output()
            .map_err(|e| RelayerError::Backend(format!("failed to run sncast call: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(RelayerError::Backend(format!(
                "sncast call {} failed: {} {}",
                function,
                stdout.trim(),
                stderr.trim()
            )));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(extract_hex_tokens(&stdout))
    }

    fn sncast_invoke(
        &self,
        contract: &str,
        function: &str,
        calldata: &[String],
    ) -> Result<(), RelayerError> {
        let mut cmd = Command::new("sncast");
        cmd.arg("--account")
            .arg(&self.account)
            .arg("--url")
            .arg(&self.rpc_url)
            .arg("invoke")
            .arg("--contract-address")
            .arg(contract)
            .arg("--function")
            .arg(function);
        if !calldata.is_empty() {
            cmd.arg("--calldata");
            for arg in calldata {
                cmd.arg(arg);
            }
        }
        let output = cmd
            .output()
            .map_err(|e| RelayerError::Backend(format!("failed to run sncast invoke: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(RelayerError::Backend(format!(
                "sncast invoke {} failed: {} {}",
                function,
                stdout.trim(),
                stderr.trim()
            )));
        }
        Ok(())
    }
}

impl Vm31RelayerBackend for SncastVm31Backend {
    fn is_proof_verified(&self, proof_hash: &str) -> Result<bool, RelayerError> {
        let out = self.sncast_call(
            &self.verifier_contract,
            "is_proof_verified",
            &[normalize_hex(proof_hash)],
        )?;
        Ok(!out.first().map(|v| is_zero_felt(v)).unwrap_or(true))
    }

    fn get_vm31_public_hash(
        &self,
        proof_hash: &str,
    ) -> Result<Option<(String, String)>, RelayerError> {
        match self.sncast_call(
            &self.verifier_contract,
            "get_vm31_public_hash",
            &[normalize_hex(proof_hash)],
        ) {
            Ok(out) => {
                if out.len() < 2 {
                    return Err(RelayerError::Backend(
                        "get_vm31_public_hash returned fewer than 2 felts".to_string(),
                    ));
                }
                Ok(Some((normalize_hex(&out[0]), normalize_hex(&out[1]))))
            }
            Err(RelayerError::Backend(msg))
                if msg.contains("VM31 hash not bound")
                    || msg.contains("public hash not bound")
                    || msg.contains("VM31: public hash not bound") =>
            {
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    fn bind_vm31_public_hash(
        &self,
        proof_hash: &str,
        batch_hash_lo: &str,
        batch_hash_hi: &str,
    ) -> Result<(), RelayerError> {
        self.sncast_invoke(
            &self.verifier_contract,
            "bind_vm31_public_hash",
            &[
                normalize_hex(proof_hash),
                normalize_hex(batch_hash_lo),
                normalize_hex(batch_hash_hi),
            ],
        )
    }

    fn get_batch_for_proof_hash(&self, proof_hash: &str) -> Result<Option<String>, RelayerError> {
        let out = self.sncast_call(
            &self.pool_contract,
            "get_batch_for_proof_hash",
            &[normalize_hex(proof_hash)],
        )?;
        let first = out.first().cloned().unwrap_or_else(|| "0x0".to_string());
        if is_zero_felt(&first) {
            Ok(None)
        } else {
            Ok(Some(normalize_hex(&first)))
        }
    }

    fn submit_batch_proof(
        &self,
        inputs: &BatchPublicInputs,
        proof_hash: &str,
        recipients: &WithdrawalRecipients,
    ) -> Result<(), RelayerError> {
        let calldata = build_submit_batch_proof_calldata(inputs, proof_hash, recipients)?;
        self.sncast_invoke(&self.pool_contract, "submit_batch_proof", &calldata)
    }

    fn get_batch_status(&self, batch_id: &str) -> Result<u8, RelayerError> {
        let out = self.sncast_call(
            &self.pool_contract,
            "get_batch_status",
            &[normalize_hex(batch_id)],
        )?;
        let v = parse_felt_to_u64(out.first().map(|s| s.as_str()).unwrap_or("0x0"))?;
        Ok(v as u8)
    }

    fn get_batch_total_txs(&self, batch_id: &str) -> Result<u32, RelayerError> {
        let out = self.sncast_call(
            &self.pool_contract,
            "get_batch_total_txs",
            &[normalize_hex(batch_id)],
        )?;
        let v = parse_felt_to_u64(out.first().map(|s| s.as_str()).unwrap_or("0x0"))?;
        Ok(v as u32)
    }

    fn get_batch_processed_count(&self, batch_id: &str) -> Result<u32, RelayerError> {
        let out = self.sncast_call(
            &self.pool_contract,
            "get_batch_processed_count",
            &[normalize_hex(batch_id)],
        )?;
        let v = parse_felt_to_u64(out.first().map(|s| s.as_str()).unwrap_or("0x0"))?;
        Ok(v as u32)
    }

    fn apply_batch_chunk(
        &self,
        batch_id: &str,
        start: u32,
        count: u32,
    ) -> Result<(), RelayerError> {
        let calldata = build_apply_batch_chunk_calldata(batch_id, start, count);
        self.sncast_invoke(&self.pool_contract, "apply_batch_chunk", &calldata)
    }

    fn finalize_batch(&self, batch_id: &str) -> Result<(), RelayerError> {
        let calldata = build_finalize_batch_calldata(batch_id);
        self.sncast_invoke(&self.pool_contract, "finalize_batch", &calldata)
    }
}

pub fn run_vm31_relayer_flow<B: Vm31RelayerBackend>(
    backend: &B,
    inputs: &BatchPublicInputs,
    proof_hash: &str,
    recipients: &WithdrawalRecipients,
    cfg: &Vm31RelayerConfig,
) -> Result<RelayOutcome, RelayerError> {
    if cfg.chunk_size == 0 {
        return Err(RelayerError::InvalidInput(
            "chunk_size must be > 0".to_string(),
        ));
    }
    let proof_hash = normalize_hex(proof_hash);
    if recipients.payout.len() != inputs.withdrawals.len() {
        return Err(RelayerError::InvalidInput(format!(
            "payout recipient count mismatch: expected {}, got {}",
            inputs.withdrawals.len(),
            recipients.payout.len()
        )));
    }
    if recipients.credit.len() != inputs.withdrawals.len() {
        return Err(RelayerError::InvalidInput(format!(
            "credit recipient count mismatch: expected {}, got {}",
            inputs.withdrawals.len(),
            recipients.credit.len()
        )));
    }
    validate_withdrawal_bindings(inputs, recipients)?;
    let batch_hash = hash_batch_public_inputs_for_cairo(inputs)?;
    let (batch_hash_lo, batch_hash_hi) = pack_digest_to_felt_hex(&batch_hash)?;

    // Step 1: Proof verification guard.
    let verified = with_retries(&cfg.retries, "verify_proof_hash", || {
        backend.is_proof_verified(&proof_hash)
    })?;
    if !verified {
        return Err(RelayerError::ProofNotVerified(proof_hash));
    }

    // Step 2: Bind public input hash (idempotent).
    match with_retries(&cfg.retries, "read_vm31_binding", || {
        backend.get_vm31_public_hash(&proof_hash)
    })? {
        Some((lo, hi)) => {
            if normalize_hex(&lo) != batch_hash_lo || normalize_hex(&hi) != batch_hash_hi {
                return Err(RelayerError::BatchHashMismatch {
                    proof_hash: proof_hash.clone(),
                    expected_lo: batch_hash_lo,
                    expected_hi: batch_hash_hi,
                    actual_lo: normalize_hex(&lo),
                    actual_hi: normalize_hex(&hi),
                });
            }
        }
        None => {
            with_retries(&cfg.retries, "bind_vm31_public_hash", || {
                backend.bind_vm31_public_hash(&proof_hash, &batch_hash_lo, &batch_hash_hi)
            })?;
            let bound = with_retries(&cfg.retries, "read_vm31_binding_after_bind", || {
                backend.get_vm31_public_hash(&proof_hash)
            })?;
            match bound {
                Some((lo, hi))
                    if normalize_hex(&lo) == batch_hash_lo
                        && normalize_hex(&hi) == batch_hash_hi => {}
                Some((lo, hi)) => {
                    return Err(RelayerError::BatchHashMismatch {
                        proof_hash: proof_hash.clone(),
                        expected_lo: batch_hash_lo,
                        expected_hi: batch_hash_hi,
                        actual_lo: normalize_hex(&lo),
                        actual_hi: normalize_hex(&hi),
                    });
                }
                None => {
                    return Err(RelayerError::UnexpectedState(
                        "binding transaction succeeded but verifier hash is still unset"
                            .to_string(),
                    ));
                }
            }
        }
    }

    // Step 3: Submit batch (idempotent via proof_hash -> batch_id mapping).
    let mut batch_id = with_retries(&cfg.retries, "read_batch_for_proof_hash", || {
        backend.get_batch_for_proof_hash(&proof_hash)
    })?;
    if batch_id.is_none() {
        with_retries(&cfg.retries, "submit_batch_proof", || {
            backend.submit_batch_proof(inputs, &proof_hash, recipients)
        })?;
        batch_id = with_retries(
            &cfg.retries,
            "read_batch_for_proof_hash_after_submit",
            || backend.get_batch_for_proof_hash(&proof_hash),
        )?;
    }
    let batch_id = batch_id.ok_or_else(|| {
        RelayerError::UnexpectedState(
            "submit completed but batch id is not indexed by proof hash".to_string(),
        )
    })?;

    // Step 4: Apply chunks until fully processed.
    let total_txs = with_retries(&cfg.retries, "read_batch_total_txs", || {
        backend.get_batch_total_txs(&batch_id)
    })?;
    if total_txs == 0 {
        return Err(RelayerError::UnexpectedState(format!(
            "batch {batch_id} has zero transactions"
        )));
    }

    loop {
        let status = with_retries(&cfg.retries, "read_batch_status", || {
            backend.get_batch_status(&batch_id)
        })?;
        if status == 2 {
            break;
        }

        let processed = with_retries(&cfg.retries, "read_batch_processed_count", || {
            backend.get_batch_processed_count(&batch_id)
        })?;
        if processed >= total_txs {
            break;
        }

        let remaining = total_txs - processed;
        let count = remaining.min(cfg.chunk_size);
        let start = processed;

        with_retries(&cfg.retries, "apply_batch_chunk", || {
            // Idempotency: skip if already advanced by a prior tx/runner.
            let now_processed = backend.get_batch_processed_count(&batch_id)?;
            if now_processed > start {
                return Ok(());
            }
            backend.apply_batch_chunk(&batch_id, start, count)
        })?;
    }

    // Step 5: Finalize (idempotent).
    let status = with_retries(&cfg.retries, "read_batch_status_before_finalize", || {
        backend.get_batch_status(&batch_id)
    })?;
    if status != 2 {
        with_retries(&cfg.retries, "finalize_batch", || {
            // Idempotency: skip if already finalized.
            let now_status = backend.get_batch_status(&batch_id)?;
            if now_status == 2 {
                return Ok(());
            }
            backend.finalize_batch(&batch_id)
        })?;
    }

    let final_status = with_retries(&cfg.retries, "read_batch_status_after_finalize", || {
        backend.get_batch_status(&batch_id)
    })?;
    let processed_txs = with_retries(&cfg.retries, "read_processed_after_finalize", || {
        backend.get_batch_processed_count(&batch_id)
    })?;

    Ok(RelayOutcome {
        proof_hash,
        batch_id,
        batch_hash,
        total_txs,
        processed_txs,
        finalized: final_status == 2,
    })
}

fn validate_withdrawal_bindings(
    inputs: &BatchPublicInputs,
    recipients: &WithdrawalRecipients,
) -> Result<(), RelayerError> {
    for (i, ((wdr, payout_recipient), credit_recipient)) in inputs
        .withdrawals
        .iter()
        .zip(recipients.payout.iter())
        .zip(recipients.credit.iter())
        .enumerate()
    {
        let expected_v2 = compute_withdrawal_binding_digest(
            payout_recipient,
            credit_recipient,
            wdr.asset_id.0 as u64,
            wdr.amount_lo.0 as u64,
            wdr.amount_hi.0 as u64,
            i as u32,
        )?;
        if wdr.withdrawal_binding != expected_v2 {
            let expected_v1 = compute_withdrawal_binding_digest_v1(
                payout_recipient,
                wdr.asset_id.0 as u64,
                wdr.amount_lo.0 as u64,
                wdr.amount_hi.0 as u64,
                i as u32,
            )?;
            if wdr.withdrawal_binding != expected_v1 {
                return Err(RelayerError::InvalidInput(format!(
                    "withdrawal binding mismatch at index {i}"
                )));
            }
            if normalize_hex(payout_recipient) != normalize_hex(credit_recipient) {
                return Err(RelayerError::InvalidInput(format!(
                    "legacy v1 binding requires payout==credit at index {i}"
                )));
            }
        }
    }
    Ok(())
}

pub fn hash_batch_public_inputs_for_cairo(
    inputs: &BatchPublicInputs,
) -> Result<[M31; RATE], RelayerError> {
    let dep_len: u32 = inputs
        .deposits
        .len()
        .try_into()
        .map_err(|_| RelayerError::InvalidInput("too many deposits".to_string()))?;
    let wdr_len: u32 = inputs
        .withdrawals
        .len()
        .try_into()
        .map_err(|_| RelayerError::InvalidInput("too many withdrawals".to_string()))?;
    let spd_len: u32 = inputs
        .spends
        .len()
        .try_into()
        .map_err(|_| RelayerError::InvalidInput("too many spends".to_string()))?;

    let mut data = Vec::new();

    // Deposits block
    data.push(M31::from_u32_unchecked(dep_len));
    for dep in &inputs.deposits {
        data.extend_from_slice(&dep.commitment);
        let amount_lo = (dep.amount & 0x7FFF_FFFF) as u32;
        let amount_hi = (dep.amount >> 31) as u32;
        data.push(M31::from_u32_unchecked(amount_lo));
        data.push(M31::from_u32_unchecked(amount_hi));
        data.push(dep.asset_id);
    }

    // Withdrawals block
    data.push(M31::from_u32_unchecked(wdr_len));
    for wdr in &inputs.withdrawals {
        data.extend_from_slice(&wdr.merkle_root);
        data.extend_from_slice(&wdr.nullifier);
        data.push(wdr.amount_lo);
        data.push(wdr.amount_hi);
        data.push(wdr.asset_id);
        data.extend_from_slice(&wdr.withdrawal_binding);
    }

    // Spends block
    data.push(M31::from_u32_unchecked(spd_len));
    for spd in &inputs.spends {
        data.extend_from_slice(&spd.merkle_root);
        data.extend_from_slice(&spd.nullifiers[0]);
        data.extend_from_slice(&spd.nullifiers[1]);
        data.extend_from_slice(&spd.output_commitments[0]);
        data.extend_from_slice(&spd.output_commitments[1]);
    }

    let digest = poseidon2_hash(&data);
    Ok(digest)
}

/// Compute the proof-bound withdrawal binding digest from bridge tuple fields.
///
/// The encoding matches Cairo `vm31_pool.compute_withdrawal_binding_felt`:
///   poseidon252(domain, payout_recipient, credit_recipient, asset_id, amount_lo, amount_hi, withdraw_idx)
/// then truncated to 8 canonical M31 limbs (low 248 bits, with limb==p mapped to 0).
pub fn compute_withdrawal_binding_digest(
    payout_recipient: &str,
    credit_recipient: &str,
    asset_id: u64,
    amount_lo: u64,
    amount_hi: u64,
    withdraw_idx: u32,
) -> Result<[M31; RATE], RelayerError> {
    if amount_lo >= M31_P || amount_hi >= M31_P {
        return Err(RelayerError::InvalidInput(
            "amount limbs must be canonical M31".to_string(),
        ));
    }
    if asset_id >= M31_P {
        return Err(RelayerError::InvalidInput(
            "asset_id must be canonical M31".to_string(),
        ));
    }

    let domain = FieldElement::from_hex_be(WITHDRAW_BINDING_DOMAIN_V2_HEX)
        .map_err(|e| RelayerError::InvalidInput(format!("invalid binding domain constant: {e}")))?;
    let payout_recipient_fe = parse_field_element(payout_recipient)?;
    let credit_recipient_fe = parse_field_element(credit_recipient)?;
    let hash = poseidon_hash_many(&[
        domain,
        payout_recipient_fe,
        credit_recipient_fe,
        FieldElement::from(asset_id),
        FieldElement::from(amount_lo),
        FieldElement::from(amount_hi),
        FieldElement::from(withdraw_idx as u64),
    ]);
    Ok(field_element_to_binding_digest(hash))
}

/// Compute the legacy V1 binding digest:
///   poseidon252(domain_v1, payout_recipient, asset_id, amount_lo, amount_hi, withdraw_idx)
pub fn compute_withdrawal_binding_digest_v1(
    payout_recipient: &str,
    asset_id: u64,
    amount_lo: u64,
    amount_hi: u64,
    withdraw_idx: u32,
) -> Result<[M31; RATE], RelayerError> {
    if amount_lo >= M31_P || amount_hi >= M31_P {
        return Err(RelayerError::InvalidInput(
            "amount limbs must be canonical M31".to_string(),
        ));
    }
    if asset_id >= M31_P {
        return Err(RelayerError::InvalidInput(
            "asset_id must be canonical M31".to_string(),
        ));
    }

    let domain = FieldElement::from_hex_be(WITHDRAW_BINDING_DOMAIN_V1_HEX)
        .map_err(|e| RelayerError::InvalidInput(format!("invalid binding domain constant: {e}")))?;
    let payout_recipient_fe = parse_field_element(payout_recipient)?;
    let hash = poseidon_hash_many(&[
        domain,
        payout_recipient_fe,
        FieldElement::from(asset_id),
        FieldElement::from(amount_lo),
        FieldElement::from(amount_hi),
        FieldElement::from(withdraw_idx as u64),
    ]);
    Ok(field_element_to_binding_digest(hash))
}

pub fn pack_digest_to_felt_hex(digest: &[M31; RATE]) -> Result<(String, String), RelayerError> {
    let lo = pack4_to_u128(&digest[0..4])?;
    let hi = pack4_to_u128(&digest[4..8])?;
    Ok((format!("0x{lo:x}"), format!("0x{hi:x}")))
}

pub fn build_bind_vm31_public_hash_calldata(
    proof_hash: &str,
    batch_hash: &[M31; RATE],
) -> Result<Vec<String>, RelayerError> {
    let (lo, hi) = pack_digest_to_felt_hex(batch_hash)?;
    Ok(vec![normalize_hex(proof_hash), lo, hi])
}

pub fn build_submit_batch_proof_calldata(
    inputs: &BatchPublicInputs,
    proof_hash: &str,
    recipients: &WithdrawalRecipients,
) -> Result<Vec<String>, RelayerError> {
    if recipients.payout.len() != inputs.withdrawals.len() {
        return Err(RelayerError::InvalidInput(format!(
            "payout recipient count mismatch: expected {}, got {}",
            inputs.withdrawals.len(),
            recipients.payout.len()
        )));
    }
    if recipients.credit.len() != inputs.withdrawals.len() {
        return Err(RelayerError::InvalidInput(format!(
            "credit recipient count mismatch: expected {}, got {}",
            inputs.withdrawals.len(),
            recipients.credit.len()
        )));
    }

    let mut calldata = Vec::new();

    // deposits: Array<DepositPublicInput>
    calldata.push(to_hex_u64(inputs.deposits.len() as u64));
    for dep in &inputs.deposits {
        append_packed_digest(&mut calldata, &dep.commitment)?;
        let amount_lo = dep.amount & 0x7FFF_FFFF;
        let amount_hi = dep.amount >> 31;
        calldata.push(to_hex_u64(amount_lo));
        calldata.push(to_hex_u64(amount_hi));
        calldata.push(to_hex_u64(dep.asset_id.0 as u64));
    }

    // withdrawals: Array<WithdrawPublicInput>
    calldata.push(to_hex_u64(inputs.withdrawals.len() as u64));
    for wdr in &inputs.withdrawals {
        append_packed_digest(&mut calldata, &wdr.merkle_root)?;
        append_packed_digest(&mut calldata, &wdr.nullifier)?;
        calldata.push(to_hex_u64(wdr.amount_lo.0 as u64));
        calldata.push(to_hex_u64(wdr.amount_hi.0 as u64));
        calldata.push(to_hex_u64(wdr.asset_id.0 as u64));
        append_packed_digest(&mut calldata, &wdr.withdrawal_binding)?;
    }

    // spends: Array<SpendPublicInput>
    calldata.push(to_hex_u64(inputs.spends.len() as u64));
    for spd in &inputs.spends {
        append_packed_digest(&mut calldata, &spd.merkle_root)?;
        append_packed_digest(&mut calldata, &spd.nullifiers[0])?;
        append_packed_digest(&mut calldata, &spd.nullifiers[1])?;
        append_packed_digest(&mut calldata, &spd.output_commitments[0])?;
        append_packed_digest(&mut calldata, &spd.output_commitments[1])?;
    }

    calldata.push(normalize_hex(proof_hash));

    // payout_recipients: Array<ContractAddress>
    calldata.push(to_hex_u64(recipients.payout.len() as u64));
    for recipient in &recipients.payout {
        calldata.push(normalize_hex(recipient));
    }
    // credit_recipients: Array<ContractAddress>
    calldata.push(to_hex_u64(recipients.credit.len() as u64));
    for recipient in &recipients.credit {
        calldata.push(normalize_hex(recipient));
    }
    Ok(calldata)
}

pub fn build_apply_batch_chunk_calldata(batch_id: &str, start: u32, count: u32) -> Vec<String> {
    vec![
        normalize_hex(batch_id),
        to_hex_u64(start as u64),
        to_hex_u64(count as u64),
    ]
}

pub fn build_finalize_batch_calldata(batch_id: &str) -> Vec<String> {
    vec![normalize_hex(batch_id)]
}

fn append_packed_digest(
    calldata: &mut Vec<String>,
    digest: &[M31; RATE],
) -> Result<(), RelayerError> {
    let (lo, hi) = pack_digest_to_felt_hex(digest)?;
    calldata.push(lo);
    calldata.push(hi);
    Ok(())
}

fn parse_field_element(value: &str) -> Result<FieldElement, RelayerError> {
    let normalized = normalize_hex(value);
    FieldElement::from_hex_be(normalized.as_str())
        .map_err(|e| RelayerError::InvalidInput(format!("invalid felt '{}': {e}", value)))
}

fn field_element_to_binding_digest(fe: FieldElement) -> [M31; RATE] {
    let mut bytes_le = fe.to_bytes_be();
    bytes_le.reverse();
    let mut out = [M31::from_u32_unchecked(0); RATE];
    for (i, item) in out.iter_mut().enumerate() {
        let limb = extract_bits_le(&bytes_le, i * 31, 31);
        // Canonical M31 element: map limb==p to 0.
        let canonical = if limb == M31_P as u32 { 0 } else { limb };
        *item = M31::from_u32_unchecked(canonical);
    }
    out
}

fn extract_bits_le(bytes_le: &[u8; 32], start: usize, width: usize) -> u32 {
    let mut out = 0u32;
    for i in 0..width {
        let bit_index = start + i;
        let byte = bytes_le[bit_index / 8];
        let bit = (byte >> (bit_index % 8)) & 1;
        out |= (bit as u32) << i;
    }
    out
}

fn pack4_to_u128(elems: &[M31]) -> Result<u128, RelayerError> {
    if elems.len() != 4 {
        return Err(RelayerError::InvalidInput(
            "digest packing requires exactly 4 M31 elements".to_string(),
        ));
    }
    let shift = 1u128 << 31;
    let mut out = 0u128;
    for (i, m) in elems.iter().enumerate() {
        let v = m.0 as u64;
        if v >= M31_P {
            return Err(RelayerError::InvalidInput(format!(
                "value {v} out of M31 range"
            )));
        }
        out += (v as u128) * shift.pow(i as u32);
    }
    Ok(out)
}

fn with_retries<T, F>(policy: &RetryPolicy, step: &str, mut op: F) -> Result<T, RelayerError>
where
    F: FnMut() -> Result<T, RelayerError>,
{
    let attempts = policy.max_attempts.max(1);
    let mut last_error = None;
    for i in 0..attempts {
        match op() {
            Ok(v) => return Ok(v),
            Err(e) => {
                last_error = Some(e.to_string());
                if i + 1 < attempts {
                    sleep(policy.backoff);
                }
            }
        }
    }
    Err(RelayerError::RetriesExhausted {
        step: step.to_string(),
        attempts,
        last_error: last_error.unwrap_or_else(|| "unknown".to_string()),
    })
}

fn normalize_hex(value: &str) -> String {
    let trimmed = value.trim();
    let hex = if let Some(stripped) = trimmed.strip_prefix("0x") {
        stripped
    } else if let Some(stripped) = trimmed.strip_prefix("0X") {
        stripped
    } else {
        trimmed
    };
    let normalized = hex.trim_start_matches('0').to_ascii_lowercase();
    if normalized.is_empty() {
        "0x0".to_string()
    } else {
        format!("0x{normalized}")
    }
}

fn to_hex_u64(v: u64) -> String {
    format!("0x{v:x}")
}

fn parse_felt_to_u64(value: &str) -> Result<u64, RelayerError> {
    let trimmed = value.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|e| {
            RelayerError::Backend(format!("failed to parse felt '{value}' as u64: {e}"))
        })
    } else {
        trimmed.parse::<u64>().map_err(|e| {
            RelayerError::Backend(format!("failed to parse felt '{value}' as u64: {e}"))
        })
    }
}

fn is_zero_felt(value: &str) -> bool {
    normalize_hex(value) == "0x0"
}

fn extract_hex_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i + 2 <= bytes.len() {
        if bytes[i] == b'0' && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
            let start = i;
            i += 2;
            let mut j = i;
            while j < bytes.len() && (bytes[j] as char).is_ascii_hexdigit() {
                j += 1;
            }
            if j > i {
                out.push(normalize_hex(&text[start..j]));
                i = j;
                continue;
            }
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use crate::circuits::deposit::DepositPublicInputs;
    use crate::circuits::spend::SpendPublicInputs;
    use crate::circuits::withdraw::WithdrawPublicInputs;

    fn sample_inputs() -> BatchPublicInputs {
        let withdrawal_binding =
            compute_withdrawal_binding_digest("0x1234", "0x1234", 7, 5, 0, 0).unwrap();
        BatchPublicInputs {
            deposits: vec![DepositPublicInputs {
                commitment: [M31::from_u32_unchecked(1); RATE],
                amount: 123,
                asset_id: M31::from_u32_unchecked(7),
            }],
            withdrawals: vec![WithdrawPublicInputs {
                merkle_root: [M31::from_u32_unchecked(2); RATE],
                nullifier: [M31::from_u32_unchecked(3); RATE],
                amount_lo: M31::from_u32_unchecked(5),
                amount_hi: M31::from_u32_unchecked(0),
                asset_id: M31::from_u32_unchecked(7),
                withdrawal_binding,
            }],
            spends: vec![SpendPublicInputs {
                merkle_root: [M31::from_u32_unchecked(4); RATE],
                nullifiers: [
                    [M31::from_u32_unchecked(5); RATE],
                    [M31::from_u32_unchecked(6); RATE],
                ],
                output_commitments: [
                    [M31::from_u32_unchecked(7); RATE],
                    [M31::from_u32_unchecked(8); RATE],
                ],
            }],
        }
    }

    struct MockBackend {
        verified: bool,
        bound: RefCell<Option<(String, String)>>,
        batch_id: RefCell<Option<String>>,
        status: RefCell<u8>,
        total: u32,
        processed: RefCell<u32>,
        bind_fail_once: RefCell<bool>,
        bind_calls: RefCell<u32>,
        submit_calls: RefCell<u32>,
        apply_calls: RefCell<u32>,
        finalize_calls: RefCell<u32>,
    }

    impl MockBackend {
        fn new(total: u32) -> Self {
            Self {
                verified: true,
                bound: RefCell::new(None),
                batch_id: RefCell::new(None),
                status: RefCell::new(1),
                total,
                processed: RefCell::new(0),
                bind_fail_once: RefCell::new(false),
                bind_calls: RefCell::new(0),
                submit_calls: RefCell::new(0),
                apply_calls: RefCell::new(0),
                finalize_calls: RefCell::new(0),
            }
        }
    }

    impl Vm31RelayerBackend for MockBackend {
        fn is_proof_verified(&self, _proof_hash: &str) -> Result<bool, RelayerError> {
            Ok(self.verified)
        }

        fn get_vm31_public_hash(
            &self,
            _proof_hash: &str,
        ) -> Result<Option<(String, String)>, RelayerError> {
            Ok(self.bound.borrow().clone())
        }

        fn bind_vm31_public_hash(
            &self,
            _proof_hash: &str,
            batch_hash_lo: &str,
            batch_hash_hi: &str,
        ) -> Result<(), RelayerError> {
            *self.bind_calls.borrow_mut() += 1;
            if *self.bind_fail_once.borrow() {
                *self.bind_fail_once.borrow_mut() = false;
                return Err(RelayerError::Backend("temporary bind failure".to_string()));
            }
            self.bound
                .replace(Some((batch_hash_lo.to_string(), batch_hash_hi.to_string())));
            Ok(())
        }

        fn get_batch_for_proof_hash(
            &self,
            _proof_hash: &str,
        ) -> Result<Option<String>, RelayerError> {
            Ok(self.batch_id.borrow().clone())
        }

        fn submit_batch_proof(
            &self,
            _inputs: &BatchPublicInputs,
            _proof_hash: &str,
            _recipients: &WithdrawalRecipients,
        ) -> Result<(), RelayerError> {
            *self.submit_calls.borrow_mut() += 1;
            self.batch_id.replace(Some("0xbeef".to_string()));
            self.status.replace(1);
            Ok(())
        }

        fn get_batch_status(&self, _batch_id: &str) -> Result<u8, RelayerError> {
            Ok(*self.status.borrow())
        }

        fn get_batch_total_txs(&self, _batch_id: &str) -> Result<u32, RelayerError> {
            Ok(self.total)
        }

        fn get_batch_processed_count(&self, _batch_id: &str) -> Result<u32, RelayerError> {
            Ok(*self.processed.borrow())
        }

        fn apply_batch_chunk(
            &self,
            _batch_id: &str,
            _start: u32,
            count: u32,
        ) -> Result<(), RelayerError> {
            *self.apply_calls.borrow_mut() += 1;
            let current = *self.processed.borrow();
            let next = (current + count).min(self.total);
            self.processed.replace(next);
            Ok(())
        }

        fn finalize_batch(&self, _batch_id: &str) -> Result<(), RelayerError> {
            *self.finalize_calls.borrow_mut() += 1;
            self.status.replace(2);
            Ok(())
        }
    }

    #[test]
    fn test_pack_digest_to_felt_hex_roundtrip_shape() {
        let digest = [M31::from_u32_unchecked(1); RATE];
        let (lo, hi) = pack_digest_to_felt_hex(&digest).unwrap();
        assert!(lo.starts_with("0x"));
        assert!(hi.starts_with("0x"));
    }

    #[test]
    fn test_build_submit_batch_proof_calldata_has_expected_prefixes() {
        let inputs = sample_inputs();
        let recipients =
            WithdrawalRecipients::new(vec!["0x1234".to_string()], vec!["0x5678".to_string()]);
        let calldata = build_submit_batch_proof_calldata(&inputs, "0xabc", &recipients).unwrap();
        // deposits_len, deposit struct, withdrawals_len, withdraw struct, spends_len, spend struct, proof hash
        assert_eq!(calldata[0], "0x1");
        // withdrawals length appears after deposit payload (2 packed + 3 scalars)
        assert_eq!(calldata[6], "0x1");
        // proof hash then payout array(len + 1 recipient) then credit array(len + 1 recipient)
        assert_eq!(calldata[calldata.len() - 5], "0xabc");
        assert_eq!(calldata[calldata.len() - 4], "0x1");
        assert_eq!(calldata[calldata.len() - 3], "0x1234");
        assert_eq!(calldata[calldata.len() - 2], "0x1");
        assert_eq!(calldata[calldata.len() - 1], "0x5678");
    }

    #[test]
    fn test_relayer_flow_happy_path() {
        let backend = MockBackend::new(5);
        let inputs = sample_inputs();
        let cfg = Vm31RelayerConfig {
            chunk_size: 2,
            retries: RetryPolicy {
                max_attempts: 2,
                backoff: Duration::from_millis(1),
            },
        };
        let recipients = WithdrawalRecipients::mirrored(vec!["0x1234".to_string()]);
        let outcome = run_vm31_relayer_flow(&backend, &inputs, "0xabc", &recipients, &cfg).unwrap();

        assert_eq!(outcome.batch_id, "0xbeef");
        assert_eq!(outcome.total_txs, 5);
        assert_eq!(outcome.processed_txs, 5);
        assert!(outcome.finalized);
        assert_eq!(*backend.submit_calls.borrow(), 1);
        assert_eq!(*backend.finalize_calls.borrow(), 1);
    }

    #[test]
    fn test_relayer_flow_idempotent_resume() {
        let backend = MockBackend::new(3);
        backend
            .bound
            .replace(Some(("0x1".to_string(), "0x2".to_string())));
        backend.batch_id.replace(Some("0xbeef".to_string()));
        backend.processed.replace(3);
        backend.status.replace(2);

        let inputs = sample_inputs();
        let expected = hash_batch_public_inputs_for_cairo(&inputs).unwrap();
        let packed = pack_digest_to_felt_hex(&expected).unwrap();
        backend.bound.replace(Some(packed));

        let cfg = Vm31RelayerConfig::default();
        let recipients = WithdrawalRecipients::mirrored(vec!["0x1234".to_string()]);
        let outcome = run_vm31_relayer_flow(&backend, &inputs, "0xabc", &recipients, &cfg).unwrap();

        assert_eq!(outcome.batch_id, "0xbeef");
        assert!(outcome.finalized);
        assert_eq!(*backend.bind_calls.borrow(), 0);
        assert_eq!(*backend.submit_calls.borrow(), 0);
        assert_eq!(*backend.apply_calls.borrow(), 0);
        assert_eq!(*backend.finalize_calls.borrow(), 0);
    }

    #[test]
    fn test_relayer_flow_retries_bind() {
        let backend = MockBackend::new(1);
        backend.bind_fail_once.replace(true);
        let inputs = sample_inputs();
        let cfg = Vm31RelayerConfig {
            chunk_size: 1,
            retries: RetryPolicy {
                max_attempts: 3,
                backoff: Duration::from_millis(1),
            },
        };
        let recipients = WithdrawalRecipients::mirrored(vec!["0x1234".to_string()]);
        let outcome = run_vm31_relayer_flow(&backend, &inputs, "0xabc", &recipients, &cfg).unwrap();

        assert!(outcome.finalized);
        assert_eq!(*backend.bind_calls.borrow(), 2);
    }

    #[test]
    fn test_relayer_flow_rejects_payout_recipient_mismatch() {
        let backend = MockBackend::new(1);
        let inputs = sample_inputs();
        let cfg = Vm31RelayerConfig::default();
        let recipients = WithdrawalRecipients::new(vec![], vec!["0x1234".to_string()]);
        let err = run_vm31_relayer_flow(&backend, &inputs, "0xabc", &recipients, &cfg).unwrap_err();
        assert!(matches!(err, RelayerError::InvalidInput(_)));
        assert!(err.to_string().contains("payout recipient count mismatch"));
    }

    #[test]
    fn test_relayer_flow_rejects_credit_recipient_mismatch() {
        let backend = MockBackend::new(1);
        let inputs = sample_inputs();
        let cfg = Vm31RelayerConfig::default();
        let recipients = WithdrawalRecipients::new(vec!["0x1234".to_string()], vec![]);
        let err = run_vm31_relayer_flow(&backend, &inputs, "0xabc", &recipients, &cfg).unwrap_err();
        assert!(matches!(err, RelayerError::InvalidInput(_)));
        assert!(err.to_string().contains("credit recipient count mismatch"));
    }

    #[test]
    fn test_relayer_flow_rejects_withdraw_binding_mismatch_on_payout() {
        let backend = MockBackend::new(1);
        let inputs = sample_inputs();
        let cfg = Vm31RelayerConfig::default();
        let recipients =
            WithdrawalRecipients::new(vec!["0x9999".to_string()], vec!["0x1234".to_string()]);
        let err = run_vm31_relayer_flow(&backend, &inputs, "0xabc", &recipients, &cfg).unwrap_err();
        assert!(matches!(err, RelayerError::InvalidInput(_)));
        assert!(err
            .to_string()
            .contains("withdrawal binding mismatch at index 0"));
    }

    #[test]
    fn test_relayer_flow_rejects_withdraw_binding_mismatch_on_credit() {
        let backend = MockBackend::new(1);
        let inputs = sample_inputs();
        let cfg = Vm31RelayerConfig::default();
        let recipients =
            WithdrawalRecipients::new(vec!["0x1234".to_string()], vec!["0x9999".to_string()]);
        let err = run_vm31_relayer_flow(&backend, &inputs, "0xabc", &recipients, &cfg).unwrap_err();
        assert!(matches!(err, RelayerError::InvalidInput(_)));
        assert!(err
            .to_string()
            .contains("withdrawal binding mismatch at index 0"));
    }

    #[test]
    fn test_relayer_flow_accepts_legacy_v1_with_mirrored_recipients() {
        let backend = MockBackend::new(1);
        let mut inputs = sample_inputs();
        inputs.withdrawals[0].withdrawal_binding =
            compute_withdrawal_binding_digest_v1("0x1234", 7, 5, 0, 0).unwrap();
        let cfg = Vm31RelayerConfig::default();
        let recipients = WithdrawalRecipients::mirrored(vec!["0x1234".to_string()]);
        let outcome = run_vm31_relayer_flow(&backend, &inputs, "0xabc", &recipients, &cfg).unwrap();
        assert!(outcome.finalized);
    }

    #[test]
    fn test_relayer_flow_rejects_legacy_v1_with_split_recipients() {
        let backend = MockBackend::new(1);
        let mut inputs = sample_inputs();
        inputs.withdrawals[0].withdrawal_binding =
            compute_withdrawal_binding_digest_v1("0x1234", 7, 5, 0, 0).unwrap();
        let cfg = Vm31RelayerConfig::default();
        let recipients =
            WithdrawalRecipients::new(vec!["0x1234".to_string()], vec!["0x5678".to_string()]);
        let err = run_vm31_relayer_flow(&backend, &inputs, "0xabc", &recipients, &cfg).unwrap_err();
        assert!(matches!(err, RelayerError::InvalidInput(_)));
        assert!(err
            .to_string()
            .contains("legacy v1 binding requires payout==credit"));
    }

    #[test]
    fn test_compute_withdrawal_binding_digest_deterministic() {
        let a = compute_withdrawal_binding_digest("0x1234", "0x5678", 7, 10, 0, 3).unwrap();
        let b = compute_withdrawal_binding_digest("0x1234", "0x5678", 7, 10, 0, 3).unwrap();
        let c = compute_withdrawal_binding_digest("0x9999", "0x5678", 7, 10, 0, 3).unwrap();
        let d = compute_withdrawal_binding_digest("0x1234", "0x9999", 7, 10, 0, 3).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);

        let v1 = compute_withdrawal_binding_digest_v1("0x1234", 7, 10, 0, 3).unwrap();
        assert_ne!(a, v1);
    }
}
