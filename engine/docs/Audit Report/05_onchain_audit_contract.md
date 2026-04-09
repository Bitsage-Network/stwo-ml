# Stage 5: On-Chain Audit Contract

**Status**: Extension of Existing Contract
**Readiness**: 40% — Verifier contract exists, need `AuditRecord` storage + `submit_audit`
**Depends on**: Stage 2 (Audit Proving), Stage 4 (Audit Report Format)
**Blocks**: Stage 6 (Privacy/Encryption), Stage 7 (Access Control)

---

## Purpose

Add an `AuditRecord` storage layer to the existing on-chain verifier. Currently, `verify_model_gkr` proves the math is correct and stores `model_id -> verified: true`. That's a boolean with no context.

The audit contract extends this with:
- **Who** submitted the audit
- **When** the audit covers (time window)
- **What** was proven (inference count, commitments)
- **Where** the full report lives (off-chain hash binding)
- **History** — all audits for a model, queryable

---

## Existing Contract

### `SumcheckVerifierContract` (Sepolia: `0x005928ac...`)

Current interface in `libs/elo-cairo-verifier/src/verifier.cairo`:

```cairo
#[starknet::interface]
trait ISumcheckVerifier<TContractState> {
    fn verify_model_gkr(ref self: TContractState, model_id: felt252, calldata: Span<felt252>);
    fn verify_model_direct(ref self: TContractState, model_id: felt252, calldata: Span<felt252>);
    fn is_verified(self: @TContractState, model_id: felt252) -> bool;
    fn get_verification_count(self: @TContractState, model_id: felt252) -> u32;
    fn register_model(ref self: TContractState, model_id: felt252, weight_commitment: felt252);
}
```

Storage:
```cairo
#[storage]
struct Storage {
    verified: LegacyMap<felt252, bool>,              // model_id -> verified
    verification_count: LegacyMap<felt252, u32>,     // model_id -> count
    weight_commitments: LegacyMap<felt252, felt252>,  // model_id -> weight hash
    owner: ContractAddress,
}
```

### `ObelyskVerifier` (in `libs/stwo-ml-verifier/src/contract.cairo`)

Has events for proof verification:
```cairo
#[event]
enum Event {
    ProofVerified: ProofVerified,
    ModelRegistered: ModelRegistered,
    BatchVerified: BatchVerified,
    ProofRejected: ProofRejected,
    WeightCommitmentUpdated: WeightCommitmentUpdated,
    TeeAttestationVerified: TeeAttestationVerified,
    InferenceRecorded: InferenceRecorded,
}
```

---

## New Contract Interface

### `AuditRecord` Storage

```cairo
/// Compact on-chain audit record.
///
/// Contains only hashes and metadata — enough to verify the off-chain
/// report without storing its full content on-chain.
#[derive(Drop, Serde, starknet::Store)]
struct AuditRecord {
    /// Model that was audited.
    model_id: felt252,
    /// Poseidon hash of the full off-chain audit report.
    /// Anyone can fetch the report and verify hash(report) == this value.
    audit_report_hash: felt252,
    /// Merkle root of the inference log entries in the audit window.
    /// Proves which specific inferences are covered.
    inference_log_merkle_root: felt252,
    /// Poseidon hash of model weights used.
    /// Must match the registered weight_commitment for this model_id.
    weight_commitment: felt252,
    /// Audit window start (Unix epoch seconds).
    time_start: u64,
    /// Audit window end (Unix epoch seconds).
    time_end: u64,
    /// Number of inferences covered by this audit.
    inference_count: u32,
    /// Whether the ZK proof was verified on-chain.
    proof_verified: bool,
    /// Who submitted this audit.
    submitter: ContractAddress,
    /// Block number when audit was submitted.
    submitted_at_block: u64,
    /// Optional: TEE attestation hash (0x0 if no TEE).
    tee_attestation_hash: felt252,
    /// Privacy tier: 0 = public, 1 = encrypted (owner only), 2 = selective disclosure
    privacy_tier: u8,
    /// Arweave transaction ID for the encrypted report blob.
    /// Content-addressed — anyone can fetch, only key holders can decrypt.
    /// Stored as felt252 (first 31 bytes of the 43-byte base64url tx_id).
    arweave_tx_id: felt252,
}
```

### New Functions

```cairo
#[starknet::interface]
trait IAuditVerifier<TContractState> {
    // ─── Submit Audit ────────────────────────────────────────────
    /// Submit an audit with ZK proof verification.
    ///
    /// 1. Verifies the ZK proof (GKR or STARK) against the model
    /// 2. Checks weight_commitment matches registered model
    /// 3. Stores AuditRecord
    /// 4. Emits AuditSubmitted event
    /// 5. Returns audit_id
    fn submit_audit(
        ref self: TContractState,
        model_id: felt252,
        report_hash: felt252,
        merkle_root: felt252,
        weight_commitment: felt252,
        time_start: u64,
        time_end: u64,
        inference_count: u32,
        tee_attestation_hash: felt252,
        privacy_tier: u8,
        proof_calldata: Span<felt252>,
    ) -> felt252;  // returns audit_id

    /// Submit an audit without proof verification (off-chain verified).
    ///
    /// For cases where proof was verified locally or by a trusted prover.
    /// Still stores the AuditRecord with proof_verified = false.
    fn submit_audit_record(
        ref self: TContractState,
        model_id: felt252,
        report_hash: felt252,
        merkle_root: felt252,
        weight_commitment: felt252,
        time_start: u64,
        time_end: u64,
        inference_count: u32,
        tee_attestation_hash: felt252,
        privacy_tier: u8,
    ) -> felt252;

    // ─── Query Audits ────────────────────────────────────────────
    /// Get an audit record by its ID.
    fn get_audit(self: @TContractState, audit_id: felt252) -> AuditRecord;

    /// Get all audit IDs for a model.
    fn get_model_audits(self: @TContractState, model_id: felt252) -> Span<felt252>;

    /// Get the total number of audits for a model.
    fn get_audit_count(self: @TContractState, model_id: felt252) -> u32;

    /// Get the most recent audit for a model.
    fn get_latest_audit(self: @TContractState, model_id: felt252) -> AuditRecord;

    /// Check if a model has been audited within a time range.
    fn is_audited_in_range(
        self: @TContractState,
        model_id: felt252,
        since: u64,
        until: u64,
    ) -> bool;

    /// Get total inferences proven across all audits for a model.
    fn get_total_proven_inferences(self: @TContractState, model_id: felt252) -> u64;
}
```

### Events

```cairo
#[derive(Drop, starknet::Event)]
struct AuditSubmitted {
    #[key]
    audit_id: felt252,
    #[key]
    model_id: felt252,
    submitter: ContractAddress,
    report_hash: felt252,
    merkle_root: felt252,
    time_start: u64,
    time_end: u64,
    inference_count: u32,
    proof_verified: bool,
    privacy_tier: u8,
}

#[derive(Drop, starknet::Event)]
struct AuditAccessGranted {
    #[key]
    audit_id: felt252,
    grantee: ContractAddress,
    granted_by: ContractAddress,
}

#[derive(Drop, starknet::Event)]
struct AuditAccessRevoked {
    #[key]
    audit_id: felt252,
    revokee: ContractAddress,
    revoked_by: ContractAddress,
}
```

### Storage Layout

```cairo
#[storage]
struct Storage {
    // ─── Existing ────────────────────────────────────
    verified: LegacyMap<felt252, bool>,
    verification_count: LegacyMap<felt252, u32>,
    weight_commitments: LegacyMap<felt252, felt252>,
    owner: ContractAddress,

    // ─── New: Audit Records ──────────────────────────
    /// audit_id -> AuditRecord
    audit_records: LegacyMap<felt252, AuditRecord>,
    /// model_id -> [audit_id, audit_id, ...] (append-only list)
    model_audit_ids: LegacyMap<(felt252, u32), felt252>,  // (model_id, index) -> audit_id
    /// model_id -> number of audits
    model_audit_count: LegacyMap<felt252, u32>,
    /// Global audit counter (for generating unique audit_ids).
    next_audit_nonce: u32,
    /// model_id -> total proven inferences across all audits
    total_proven_inferences: LegacyMap<felt252, u64>,
}
```

### Audit ID Generation

```cairo
/// Generate a unique audit ID.
///
/// Uses Poseidon hash of (nonce, model_id, submitter, time_start) for uniqueness.
fn generate_audit_id(
    nonce: u32,
    model_id: felt252,
    submitter: ContractAddress,
    time_start: u64,
) -> felt252 {
    let submitter_felt: felt252 = submitter.into();
    poseidon_hash_many(
        array![
            nonce.into(),
            model_id,
            submitter_felt,
            time_start.into(),
        ].span()
    )
}
```

---

## Implementation: `submit_audit`

```cairo
fn submit_audit(
    ref self: ContractState,
    model_id: felt252,
    report_hash: felt252,
    merkle_root: felt252,
    weight_commitment: felt252,
    time_start: u64,
    time_end: u64,
    inference_count: u32,
    tee_attestation_hash: felt252,
    privacy_tier: u8,
    proof_calldata: Span<felt252>,
) -> felt252 {
    // 1. Verify weight commitment matches registered model
    let registered_weight = self.weight_commitments.read(model_id);
    assert(registered_weight != 0, 'Model not registered');
    assert(weight_commitment == registered_weight, 'Weight commitment mismatch');

    // 2. Verify time window is valid
    assert(time_end > time_start, 'Invalid time window');
    assert(inference_count > 0, 'Empty audit');

    // 3. Verify the ZK proof
    //    Calls existing verify_model_gkr internally
    let proof_valid = self._verify_proof(model_id, proof_calldata);

    // 4. Generate audit ID
    let nonce = self.next_audit_nonce.read();
    let submitter = get_caller_address();
    let audit_id = generate_audit_id(nonce, model_id, submitter, time_start);
    self.next_audit_nonce.write(nonce + 1);

    // 5. Store audit record
    let record = AuditRecord {
        model_id,
        audit_report_hash: report_hash,
        inference_log_merkle_root: merkle_root,
        weight_commitment,
        time_start,
        time_end,
        inference_count,
        proof_verified: proof_valid,
        submitter,
        submitted_at_block: starknet::get_block_number(),
        tee_attestation_hash,
        privacy_tier,
    };
    self.audit_records.write(audit_id, record);

    // 6. Append to model's audit list
    let count = self.model_audit_count.read(model_id);
    self.model_audit_ids.write((model_id, count), audit_id);
    self.model_audit_count.write(model_id, count + 1);

    // 7. Update total proven inferences
    let total = self.total_proven_inferences.read(model_id);
    self.total_proven_inferences.write(model_id, total + inference_count.into());

    // 8. Emit event
    self.emit(AuditSubmitted {
        audit_id,
        model_id,
        submitter,
        report_hash,
        merkle_root,
        time_start,
        time_end,
        inference_count,
        proof_verified: proof_valid,
        privacy_tier,
    });

    audit_id
}
```

---

## Query Patterns

### "Has this model been audited recently?"

```cairo
fn is_audited_in_range(self: @ContractState, model_id: felt252, since: u64, until: u64) -> bool {
    let count = self.model_audit_count.read(model_id);
    let mut i = count;
    loop {
        if i == 0 { break false; }
        i -= 1;
        let audit_id = self.model_audit_ids.read((model_id, i));
        let record = self.audit_records.read(audit_id);
        if record.time_start >= since && record.time_end <= until && record.proof_verified {
            break true;
        }
    }
}
```

### "Show me all audits for Qwen3-14B"

```cairo
fn get_model_audits(self: @ContractState, model_id: felt252) -> Span<felt252> {
    let count = self.model_audit_count.read(model_id);
    let mut ids = ArrayTrait::new();
    let mut i = 0;
    loop {
        if i >= count { break; }
        ids.append(self.model_audit_ids.read((model_id, i)));
        i += 1;
    };
    ids.span()
}
```

---

## Backward Compatibility

The new functions are **additive** — existing `verify_model_gkr` and `is_verified` continue to work unchanged. The audit system is an optional layer on top.

Models can be:
1. **Verified** (existing) — at least one proof submitted (boolean)
2. **Audited** (new) — has audit records with time windows, inference counts, report hashes

---

## Gas Estimates

| Operation | Estimated Gas | STRK Cost (Sepolia) |
|-----------|-------------|-------------------|
| `submit_audit` (no proof) | ~50K steps | ~0.005 STRK |
| `submit_audit` (with GKR proof) | ~200K steps | ~0.02 STRK |
| `get_audit` (read) | ~5K steps | Free (view) |
| `get_model_audits` (10 audits) | ~15K steps | Free (view) |

With AVNU paymaster, `submit_audit` is gasless for the submitter on Sepolia.

---

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `libs/elo-cairo-verifier/src/audit.cairo` | **New** | ~300 (AuditRecord, submit_audit, queries) |
| `libs/elo-cairo-verifier/src/verifier.cairo` | **Modify** | ~50 (add audit module, storage) |
| `libs/elo-cairo-verifier/src/lib.cairo` | **Modify** | +1 (module declaration) |
| `libs/elo-cairo-verifier/tests/test_audit.cairo` | **New** | ~200 (unit tests) |

---

## Verification Criteria

- [ ] `submit_audit` reverts if model not registered
- [ ] `submit_audit` reverts if weight commitment doesn't match
- [ ] `submit_audit` reverts if time window is invalid
- [ ] Audit ID is unique across all submissions
- [ ] `get_model_audits` returns all audit IDs for a model
- [ ] `get_latest_audit` returns the most recent audit
- [ ] `is_audited_in_range` correctly checks time overlap
- [ ] `get_total_proven_inferences` accumulates across audits
- [ ] Existing `verify_model_gkr` and `is_verified` still work
- [ ] Events contain all necessary fields for indexing
