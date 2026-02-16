// Audit Record — on-chain storage for batch inference audits.
//
// Extends the existing SumcheckVerifier contract with audit record
// storage, submission, and query functions. Backward compatible:
// existing verify_model_gkr and is_verified continue to work.
//
// Storage model:
//   audit_records:    audit_id  →  AuditRecord
//   model_audit_ids:  (model_id, index)  →  audit_id
//   model_audit_count: model_id  →  u32
//   next_audit_nonce: global counter

use starknet::ContractAddress;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Packed M31 digest (8 M31 elements packed into two felt252 values).
///
/// `lo = m31[0] + m31[1]*2^31 + m31[2]*2^62 + m31[3]*2^93` (124 bits)
/// `hi = m31[4] + m31[5]*2^31 + m31[6]*2^62 + m31[7]*2^93` (124 bits)
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct PackedDigest8 {
    pub lo: felt252,
    pub hi: felt252,
}

/// Compact on-chain audit record.
///
/// Contains only hashes and metadata — enough to verify the off-chain
/// report without storing its full content on-chain.
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct AuditRecord {
    /// Model that was audited.
    pub model_id: felt252,
    /// Poseidon2-M31 hash of the full off-chain audit report (packed digest).
    pub audit_report_hash: PackedDigest8,
    /// M31 Merkle root of the inference log entries in the audit window.
    pub inference_log_merkle_root: PackedDigest8,
    /// Weight commitment (felt252, matches registered model).
    pub weight_commitment: felt252,
    /// Audit window start (Unix epoch seconds).
    pub time_start: u64,
    /// Audit window end (Unix epoch seconds).
    pub time_end: u64,
    /// Number of inferences covered by this audit.
    pub inference_count: u32,
    /// Whether the ZK proof was verified on-chain.
    pub proof_verified: bool,
    /// Who submitted this audit.
    pub submitter: ContractAddress,
    /// Block number when audit was submitted.
    pub submitted_at_block: u64,
    /// TEE attestation hash (0x0 if no TEE).
    pub tee_attestation_hash: felt252,
    /// Privacy tier: 0 = public, 1 = encrypted, 2 = selective.
    pub privacy_tier: u8,
}

// ─── Events ─────────────────────────────────────────────────────────────────

#[derive(Drop, starknet::Event)]
pub struct AuditSubmitted {
    #[key]
    pub audit_id: felt252,
    #[key]
    pub model_id: felt252,
    pub submitter: ContractAddress,
    pub report_hash_lo: felt252,
    pub report_hash_hi: felt252,
    pub merkle_root_lo: felt252,
    pub merkle_root_hi: felt252,
    pub time_start: u64,
    pub time_end: u64,
    pub inference_count: u32,
    pub proof_verified: bool,
    pub privacy_tier: u8,
}

// ─── Interface ──────────────────────────────────────────────────────────────

#[starknet::interface]
pub trait IAuditVerifier<TContractState> {
    /// Submit an audit with M31-native digest fields.
    ///
    /// 1. Checks weight_commitment matches registered model
    /// 2. Validates time window and inference count
    /// 3. Stores AuditRecord with PackedDigest8 for report_hash and merkle_root
    /// 4. Emits AuditSubmitted event
    /// 5. Returns audit_id (Poseidon hash of nonce + model + submitter + time)
    fn submit_audit(
        ref self: TContractState,
        model_id: felt252,
        report_hash_lo: felt252,
        report_hash_hi: felt252,
        merkle_root_lo: felt252,
        merkle_root_hi: felt252,
        weight_commitment: felt252,
        time_start: u64,
        time_end: u64,
        inference_count: u32,
        tee_attestation_hash: felt252,
        privacy_tier: u8,
    ) -> felt252;

    /// Get an audit record by its ID.
    fn get_audit(self: @TContractState, audit_id: felt252) -> AuditRecord;

    /// Get all audit IDs for a model (returns array of audit_id).
    fn get_model_audits(self: @TContractState, model_id: felt252) -> Array<felt252>;

    /// Get the total number of audits for a model.
    fn get_audit_count(self: @TContractState, model_id: felt252) -> u32;

    /// Get the most recent audit for a model.
    fn get_latest_audit(self: @TContractState, model_id: felt252) -> AuditRecord;

    /// Check if a model has been audited within a time range.
    /// Returns true if any proof-verified audit overlaps [since, until].
    fn is_audited_in_range(
        self: @TContractState,
        model_id: felt252,
        since: u64,
        until: u64,
    ) -> bool;

    /// Get total inferences proven across all audits for a model.
    fn get_total_proven_inferences(self: @TContractState, model_id: felt252) -> u64;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Generate a unique audit ID.
///
/// Poseidon hash of (nonce, model_id, submitter, time_start).
pub fn generate_audit_id(
    nonce: u32,
    model_id: felt252,
    submitter: ContractAddress,
    time_start: u64,
) -> felt252 {
    let submitter_felt: felt252 = submitter.into();
    core::poseidon::poseidon_hash_span(
        array![nonce.into(), model_id, submitter_felt, time_start.into()].span(),
    )
}
