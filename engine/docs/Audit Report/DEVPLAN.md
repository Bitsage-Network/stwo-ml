# Dev Plan: Audit System Implementation

**Dev A** + **Dev B** — 5 weeks, developing locally
**VM31 Dev** — delivers `PoseidonM31Encryption` by end of week 2
**Marketplace Dev** — parallel, independent

---

## Day 0: Shared Interface Contract

Before any code, both devs agree on the shared types. Create this file together:

### File: `src/audit/types.rs`

```rust
//! Shared types for the audit system.
//!
//! Dev A and Dev B code against these types independently.
//! Integration happens when both sides implement their producers/consumers.

use starknet_ff::FieldElement;
use crate::components::matmul::M31Matrix;
use serde::{Serialize, Deserialize};

// ─── Stage 1: Inference Logging (Dev A produces, Dev B consumes) ───

/// A single inference record in the append-only log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceLogEntry {
    pub inference_id: u64,
    pub sequence_number: u64,
    pub model_id: String,                    // felt252 hex
    pub weight_commitment: String,           // felt252 hex
    pub model_name: String,
    pub num_layers: u32,
    pub input_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub io_commitment: String,               // felt252 hex
    pub layer_chain_commitment: String,      // felt252 hex
    pub prev_entry_hash: String,             // felt252 hex
    pub timestamp_ns: u64,
    pub latency_ms: u64,
    pub gpu_device: String,
    pub tee_report_hash: String,             // felt252 hex
    pub task_category: Option<String>,
    pub input_preview: Option<String>,
    pub output_preview: Option<String>,
}

/// Query result from the inference log.
#[derive(Debug)]
pub struct LogWindow {
    pub entries: Vec<InferenceLogEntry>,
    pub merkle_root: FieldElement,
    pub start_ns: u64,
    pub end_ns: u64,
    pub chain_verified: bool,
}

// ─── Stage 2: Audit Proving (Dev A produces, Dev B consumes) ───

/// Request for an audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRequest {
    pub start_ns: u64,
    pub end_ns: u64,
    pub model_id: String,
    pub mode: String,                        // "gkr", "direct", "recursive"
    pub evaluate_semantics: bool,
    pub max_inferences: usize,               // 0 = all
    pub gpu_device: Option<usize>,
}

/// Result of a single inference proof within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceProofResult {
    pub sequence: u64,
    pub io_commitment: String,
    pub layer_chain_commitment: String,
    pub timestamp: u64,
    pub proof_size_bytes: usize,
    pub proving_time_ms: u64,
}

/// Result of batch audit proving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchAuditResult {
    pub time_start: u64,
    pub time_end: u64,
    pub inference_count: u32,
    pub io_merkle_root: String,
    pub log_merkle_root: String,
    pub weight_commitment: String,
    pub combined_chain_commitment: String,
    pub inference_results: Vec<InferenceProofResult>,
    pub model_id: String,
    pub proving_time_ms: u64,
    pub proof_calldata: Vec<String>,         // felt252 hex for on-chain
    pub tee_attestation_hash: Option<String>,
}

// ─── Stage 3: Semantic Evaluation (Dev B produces, Dev A consumes via report) ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicCheck {
    pub check_type: String,
    pub passed: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEvaluation {
    pub sequence: u64,
    pub deterministic_checks: Vec<DeterministicCheck>,
    pub semantic_score: Option<f32>,         // 0.0 - 1.0
    pub eval_io_commitment: Option<String>,  // if evaluation was a forward pass
    pub evaluation_proved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSemanticSummary {
    pub method: String,                      // "deterministic", "self_eval", "cross_model"
    pub avg_quality_score: f32,
    pub excellent_count: u32,                // 0.9-1.0
    pub good_count: u32,                     // 0.7-0.9
    pub fair_count: u32,                     // 0.5-0.7
    pub poor_count: u32,                     // 0.0-0.5
    pub deterministic_pass: u32,
    pub deterministic_fail: u32,
    pub evaluated_count: u32,
    pub evaluations_proved: bool,
    pub eval_merkle_root: Option<String>,
    pub per_inference: Vec<InferenceEvaluation>,
}

// ─── Stage 4: Report Format (Dev B produces) ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub version: String,
    pub audit_id: String,
    pub time_window: TimeWindow,
    pub model: ModelInfo,
    pub infrastructure: InfrastructureInfo,
    pub inference_summary: InferenceSummary,
    pub semantic_evaluation: Option<AuditSemanticSummary>,
    pub commitments: AuditCommitments,
    pub proof: ProofInfo,
    pub privacy: Option<PrivacyInfo>,
    pub inferences: Vec<InferenceEntry>,
    pub billing: Option<BillingInfo>,
    pub metadata: ReportMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: String,
    pub end: String,
    pub start_epoch_ns: u64,
    pub end_epoch_ns: u64,
    pub duration_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub name: String,
    pub architecture: String,
    pub parameters: String,
    pub layers: u32,
    pub weight_commitment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfrastructureInfo {
    pub gpu_device: String,
    pub gpu_count: u32,
    pub cuda_version: String,
    pub prover_version: String,
    pub tee_active: bool,
    pub tee_attestation_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceSummary {
    pub total_inferences: u32,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub avg_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub throughput_tokens_per_sec: f32,
    pub categories: std::collections::HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditCommitments {
    pub inference_log_merkle_root: String,
    pub io_merkle_root: String,
    pub weight_commitment: String,
    pub combined_chain_commitment: String,
    pub audit_report_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofInfo {
    pub mode: String,
    pub proving_time_seconds: u64,
    pub proof_size_bytes: usize,
    pub on_chain_tx: Option<String>,
    pub on_chain_verified: Option<bool>,
    pub arweave_tx_id: Option<String>,
    pub audit_record_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyInfo {
    pub tier: String,
    pub encryption_scheme: String,          // "poseidon_m31" or "aes_256_gcm"
    pub arweave_tx_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEntry {
    pub index: u32,
    pub sequence: u64,
    pub timestamp: String,
    pub io_commitment: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u64,
    pub category: Option<String>,
    pub semantic_score: Option<f32>,
    pub input_preview: Option<String>,
    pub output_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingInfo {
    pub total_gpu_time_ms: u64,
    pub total_tokens_processed: u64,
    pub total_sage_billed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub generated_at: String,
    pub generator: String,
}

// ─── Stage 6: Encryption (Dev B integrates VM31 dev's output) ───

/// Trait for swappable encryption backends.
///
/// VM31 dev implements PoseidonM31Encryption.
/// AES-GCM fallback implements Aes256GcmEncryption.
pub trait AuditEncryption: Send + Sync {
    fn encrypt(
        &self,
        plaintext: &[u8],
        owner_pubkey: &[u8],
    ) -> Result<EncryptedBlob, EncryptionError>;

    fn decrypt(
        &self,
        blob: &EncryptedBlob,
        privkey: &[u8],
    ) -> Result<Vec<u8>, EncryptionError>;

    fn wrap_key_for(
        &self,
        blob: &EncryptedBlob,
        owner_privkey: &[u8],
        grantee_pubkey: &[u8],
    ) -> Result<Vec<u8>, EncryptionError>;

    fn scheme_name(&self) -> &str;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedBlob {
    pub ciphertext: Vec<u8>,
    pub scheme: String,                      // "poseidon_m31" or "aes_256_gcm"
    pub nonce: Vec<u8>,
    pub wrapped_keys: Vec<WrappedKey>,
    pub plaintext_hash: String,              // Poseidon hash of original
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrappedKey {
    pub recipient: String,                   // Starknet address hex
    pub encrypted_key: Vec<u8>,
    pub role: String,
}

#[derive(Debug, thiserror::Error)]
pub enum EncryptionError {
    #[error("Access denied: no wrapped key for this recipient")]
    AccessDenied,
    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),
    #[error("Key wrapping failed: {0}")]
    WrappingFailed(String),
    #[error("Hash mismatch: report was tampered")]
    HashMismatch,
}
```

### File: `src/audit/mod.rs`

```rust
pub mod types;
// Dev A adds:
// pub mod log;
// pub mod prover;
// pub mod replay;
// Dev B adds:
// pub mod evaluation;
// pub mod report;
// pub mod storage;
// pub mod encryption;
```

### Add to `src/lib.rs`:
```rust
pub mod audit;
```

**Both devs commit this together on day 0, then branch.**

---

## Dev A — Tasks

### Branch: `feat/audit-inference-log`

---

#### A1. Inference Log Core
**File**: `src/audit/log.rs` (~300 lines)
**Time**: 2-3 days

Build the append-only inference log:

- [ ] `InferenceLog::new(log_dir, model_id, weight_commitment)` — creates session directory
- [ ] `InferenceLog::append(entry)` — appends to JSONL file + Merkle tree
- [ ] `InferenceLog::query_window(start_ns, end_ns) -> LogWindow` — time-range query
- [ ] `InferenceLog::merkle_root_at(seq) -> FieldElement` — root at sequence
- [ ] `InferenceLog::merkle_proof(seq) -> Vec<FieldElement>` — proof for entry
- [ ] `InferenceLog::verify_chain() -> Result<()>` — full chain integrity check
- [ ] `InferenceLog::load(log_dir)` — recover from disk after restart

Implementation details:
- Use `BufWriter` for streaming JSONL append (one line per entry via `serde_json::to_writer`)
- M31 matrices stored in binary sidecar: `matrices.bin` (sequential `[rows: u32, cols: u32, data: [u32; rows*cols]]`)
- Merkle tree: Vec-backed binary tree, Poseidon hash at each node
- Entry hash: `poseidon_hash_many` over identity + commitment fields (see `InferenceLogEntry` in types.rs)
- Chain link: each entry's `prev_entry_hash` = previous entry's hash, first = `0x0`

Storage layout:
```
~/.obelysk/logs/session_{timestamp}/
  log.jsonl         ← InferenceLogEntry per line
  matrices.bin      ← binary M31 matrices
  merkle.bin        ← serialized Merkle tree (for fast recovery)
  meta.json         ← { model_id, weight_commitment, model_name, created_at }
```

Tests:
- [ ] Append 100 entries, verify chain integrity
- [ ] Query window returns correct subset
- [ ] Merkle root is deterministic for same entries
- [ ] Merkle proof verifies for any entry
- [ ] Log survives drop + reload from disk
- [ ] Empty log returns empty window

---

#### A2. Capture Hook
**File**: `src/audit/capture.rs` (~150 lines)
**Time**: 1-2 days

Non-blocking hook between model server and log:

- [ ] `CaptureHook::new(log_dir, model)` — spawns background thread
- [ ] `CaptureHook::record(input_tokens, output_tokens, timestamp, latency, gpu, category)` — sends to background, returns immediately
- [ ] `CaptureHook::flush()` — blocks until all pending records written
- [ ] `CaptureHook::entry_count() -> u64` — how many recorded so far
- [ ] Background thread: receives `CaptureJob` via `crossbeam::channel`, quantizes input/output to M31, computes `io_commitment` and `layer_chain_commitment`, appends to log

Implementation details:
- `crossbeam::channel::unbounded()` for job queue (non-blocking send)
- Background thread calls `compute_io_commitment` from `aggregation.rs`
- M31 quantization reuses existing `quantize_to_m31` from compiler
- On drop: flush remaining jobs, close channel

Tests:
- [ ] Record 1000 entries, verify all flushed
- [ ] Record is non-blocking (returns in < 1ms)
- [ ] Commitments match manual computation

---

#### A3. Replay Verification
**File**: `src/audit/replay.rs` (~100 lines)
**Time**: 1 day

Verify logged inference is reproducible before proving:

- [ ] `verify_replay(model, entry) -> Result<()>` — re-runs forward pass, checks io_commitment matches
- [ ] `verify_replay_batch(model, entries) -> Result<Vec<ReplayResult>>` — batch with error collection

Implementation details:
- Uses existing `forward_pass` from `compiler/prove.rs`
- Loads M31 input from binary sidecar
- Compares `compute_io_commitment(input, replayed_output)` with logged `io_commitment`
- If mismatch → `AuditError::ReplayMismatch { sequence, expected, actual }`

Tests:
- [ ] Valid replay passes
- [ ] Tampered input matrix fails
- [ ] Tampered log entry fails

---

#### A4. Audit Prover
**File**: `src/audit/prover.rs` (~400 lines)
**Time**: 4-5 days

Batch prove all inferences in a window:

- [ ] `AuditProver::new(model, mode)` — create prover for a model
- [ ] `AuditProver::prove_window(log, request) -> Result<BatchAuditResult>` — main entry point
- [ ] Internal: `prove_inference(entry) -> Result<InferenceProofResult>` — per-entry proof
- [ ] Internal: `compute_batch_commitments(results) -> (io_merkle_root, combined_chain)`

Implementation details:
- Step 1: `log.query_window(start, end)` → get entries
- Step 2: `log.verify_chain()` → ensure no tampering
- Step 3: For each entry:
  - Load M31 input from sidecar
  - `verify_replay(model, entry)` → ensure reproducible
  - Prove via `prove_model_gkr` (GKR mode) or `prove_model` (direct mode)
  - Extract calldata via `serialize_gkr_proof` (existing in `cairo_serde.rs`)
  - Record `InferenceProofResult`
- Step 4: Build io_merkle_root (Merkle tree of all io_commitments)
- Step 5: Build combined_chain_commitment (Poseidon hash of all per-inference chain commits)
- Step 6: Optionally prove billing receipts via `prove_receipt_batch`
- Step 7: Optionally attach TEE attestation

GKR mode integration:
- Reuse `prove_model_gkr` from existing code
- Calldata serialization via `serialize_gkr_calldata` (existing)
- Weight MLE is shared across all inferences (committed once)

Tests:
- [ ] Prove 1 inference, verify calldata is non-empty
- [ ] Prove 5 inferences, verify io_merkle_root is deterministic
- [ ] Prove with GKR mode
- [ ] Weight commitment constant across batch
- [ ] Replay failure prevents proving (no wasted GPU time)

---

#### A5. On-Chain Submission
**File**: `src/audit/submit.rs` (~200 lines)
**Time**: 2-3 days

Serialize batch proof for on-chain `submit_audit`:

- [ ] `serialize_audit_calldata(result: &BatchAuditResult) -> Vec<FieldElement>` — build calldata
- [ ] `submit_audit_onchain(result, config) -> Result<SubmitReceipt>` — call contract via paymaster
- [ ] `check_audit_onchain(audit_id, config) -> Result<AuditRecord>` — query contract

Implementation details:
- Calldata format: `[model_id, report_hash, merkle_root, weight_commitment, time_start, time_end, inference_count, tee_hash, privacy_tier, ...proof_calldata]`
- Submission reuses existing `paymaster_submit.mjs` (already built)
- Or direct `starknet_rs` if preferred
- `SubmitReceipt`: `{ audit_id, tx_hash, block_number, explorer_url }`

Tests:
- [ ] Calldata serialization round-trips
- [ ] Calldata size reasonable (< 100K felts for 10 inferences)
- [ ] Mock contract call succeeds (dry-run mode)

---

#### A6. Cairo Contract — AuditRecord
**File**: `libs/elo-cairo-verifier/src/audit.cairo` (~300 lines)
**Time**: 3-4 days

New Cairo module for audit storage:

- [ ] `AuditRecord` struct (from 05_onchain_audit_contract.md)
- [ ] `submit_audit()` — verify proof + store record
- [ ] `submit_audit_record()` — store without proof verification
- [ ] `get_audit(audit_id) -> AuditRecord`
- [ ] `get_model_audits(model_id) -> Span<felt252>`
- [ ] `get_audit_count(model_id) -> u32`
- [ ] `get_latest_audit(model_id) -> AuditRecord`
- [ ] `is_audited_in_range(model_id, since, until) -> bool`
- [ ] `get_total_proven_inferences(model_id) -> u64`
- [ ] `AuditSubmitted` event
- [ ] Storage: `audit_records`, `model_audit_ids`, `model_audit_count`, `next_audit_nonce`
- [ ] Integrate into existing `verifier.cairo` (add module, storage fields)

Tests (`tests/test_audit.cairo`):
- [ ] Submit audit with mock proof data
- [ ] Reject if model not registered
- [ ] Reject if weight commitment mismatch
- [ ] Query by model returns correct list
- [ ] Audit count increments
- [ ] Total proven inferences accumulates

---

### Dev A Summary

| Task | File(s) | Days | Depends On |
|------|---------|------|------------|
| A1. Log core | `audit/log.rs` | 2-3 | types.rs (day 0) |
| A2. Capture hook | `audit/capture.rs` | 1-2 | A1 |
| A3. Replay | `audit/replay.rs` | 1 | A1 |
| A4. Audit prover | `audit/prover.rs` | 4-5 | A1, A3 |
| A5. On-chain submit | `audit/submit.rs` | 2-3 | A4 |
| A6. Cairo contract | `elo-cairo-verifier/src/audit.cairo` | 3-4 | A5 (for calldata format) |
| **Total** | | **~15 days (3 weeks)** | |

---

## Dev B — Tasks

### Branch: `feat/audit-evaluation-report`

---

#### B1. Deterministic Checks
**File**: `src/audit/deterministic.rs` (~200 lines)
**Time**: 2 days

Instant programmatic checks for verifiable outputs:

- [ ] `evaluate_deterministic(input, output, task_hint) -> Vec<DeterministicCheck>`
- [ ] JSON validation check (parse output as JSON, check schema)
- [ ] Math check (extract math expressions, evaluate)
- [ ] Code syntax check (basic bracket/paren matching, keyword detection)
- [ ] SQL parse check (basic SQL grammar validation)
- [ ] Structured output check (regex pattern matching)
- [ ] Category detection: infer task_category from input if not provided

Implementation details:
- No model calls — pure Rust string processing
- Each check returns `DeterministicCheck { check_type, passed, detail }`
- Category detection: keyword matching ("write code" → code_generation, "what is" → qa, etc.)
- Keep it simple — these are sanity checks, not full parsers

Tests:
- [ ] JSON valid/invalid detection
- [ ] Math "2+2=4" passes, "2+2=5" fails
- [ ] Category detection for 10 example inputs
- [ ] All checks run in < 1ms per inference

---

#### B2. Self-Evaluation
**File**: `src/audit/self_eval.rs` (~250 lines)
**Time**: 3 days

Model evaluates its own outputs:

- [ ] `evaluate_self(model, inferences) -> Vec<InferenceEvaluation>`
- [ ] `build_eval_prompt(input, output, template) -> String` — construct evaluation prompt
- [ ] `parse_score(model_output) -> f32` — extract 0-10 score from model output, normalize to 0-1
- [ ] `EvalTemplate` enum: GeneralQuality, TaskAdherence, FactualConsistency, CodeQuality
- [ ] For each inference: tokenize eval prompt → M31 → forward pass → extract score → compute eval_io_commitment

Implementation details:
- Reuse existing tokenizer/quantization pipeline
- Forward pass via `forward_pass` from `compiler/prove.rs`
- Score extraction: regex for digits, fallback to first number in output
- eval_io_commitment: `compute_io_commitment(eval_input, eval_output)` from `aggregation.rs`
- Optional proving: if `evaluation_proved=true`, prove each eval forward pass (expensive)

Tests:
- [ ] Eval prompt construction is correct
- [ ] Score parsing handles "8", "8.5", "Score: 8", "8/10"
- [ ] eval_io_commitment is non-zero
- [ ] Batch evaluation of 10 inferences completes

---

#### B3. Aggregate Scoring
**File**: `src/audit/scoring.rs` (~100 lines)
**Time**: 1 day

Aggregate individual evaluations into summary:

- [ ] `aggregate_evaluations(evals) -> AuditSemanticSummary`
- [ ] Compute avg_quality_score
- [ ] Compute score distribution (excellent/good/fair/poor)
- [ ] Count deterministic pass/fail
- [ ] Build eval_merkle_root (Merkle of eval_io_commitments)

Tests:
- [ ] 100 evaluations aggregate correctly
- [ ] Distribution counts sum to total
- [ ] Empty evaluations handled gracefully

---

#### B4. Report Builder
**File**: `src/audit/report.rs` (~300 lines)
**Time**: 3 days

Assemble the full audit report from all components:

- [ ] `AuditReportBuilder::new()`
- [ ] `.with_audit_result(result: &BatchAuditResult)` — from Dev A's prover
- [ ] `.with_semantic_eval(eval: &AuditSemanticSummary)` — from B1-B3
- [ ] `.with_infrastructure()` — auto-detect GPU, CUDA, OS, prover version
- [ ] `.with_billing(receipts)` — from receipt chain
- [ ] `.build() -> AuditReport`
- [ ] `compute_report_hash(report) -> FieldElement` — Poseidon hash for on-chain binding

Implementation details:
- Builder pattern — each `.with_*()` sets a section
- `build()` validates completeness, generates audit_id, computes report_hash
- `compute_report_hash`: canonical JSON → chunk into 31-byte felt252s → `poseidon_hash_many`
- Infrastructure auto-detection: `nvidia-smi` for GPU, `uname` for OS, `env!("CARGO_PKG_VERSION")` for prover
- Inference entries built from `InferenceLogEntry` + `InferenceEvaluation` (join on sequence)

Tests:
- [ ] Builder produces valid JSON
- [ ] Report hash is deterministic
- [ ] Missing required fields → error
- [ ] Round-trip: serialize → deserialize → re-serialize matches

---

#### B5. Arweave Storage Client
**File**: `src/audit/storage.rs` (~200 lines)
**Time**: 2 days

Upload/download encrypted reports to Arweave:

- [ ] `ArweaveClient::new(gateway, wallet_path)` — connect to Arweave
- [ ] `ArweaveClient::upload(data, tags) -> Result<String>` — upload, return tx_id
- [ ] `ArweaveClient::download(tx_id) -> Result<Vec<u8>>` — fetch by tx_id
- [ ] `ArweaveClient::status(tx_id) -> Result<TxStatus>` — check confirmation
- [ ] Tags: `App-Name: Obelysk-Audit`, `Audit-ID: {id}`, `Model-ID: {id}`

Implementation details:
- Use `reqwest` for HTTP calls to Arweave gateway
- Upload via Irys/Bundlr for instant finality (POST to bundler endpoint)
- Alternatively: direct Arweave TX via `arweave-rs` crate
- Download: `GET https://arweave.net/{tx_id}`
- Fallback gateway: `https://ar-io.dev/{tx_id}`

Tests:
- [ ] Upload small blob, get valid tx_id
- [ ] Download returns same bytes
- [ ] Tags are queryable via GraphQL
- [ ] Integration test with Arweave testnet

---

#### B6. Encryption Integration
**File**: `src/audit/encryption.rs` (~150 lines)
**Time**: 2 days (week 3, after VM31 dev delivers)

Integrate VM31 dev's `PoseidonM31Encryption` (or AES fallback):

- [ ] `encrypt_and_store(report, owner_pubkey, storage) -> Result<StorageReceipt>`
- [ ] `fetch_and_decrypt(audit_id, privkey, storage) -> Result<AuditReport>`
- [ ] `grant_access(blob, owner_privkey, grantee_pubkey) -> Result<WrappedKey>`
- [ ] `revoke_access(report, remaining_keys) -> Result<EncryptedBlob>` — re-encrypt

Implementation details:
- Import VM31 dev's `PoseidonM31Encryption` implementing `AuditEncryption` trait
- Fallback: implement `Aes256GcmEncryption` if VM31 not ready (1 day, `aes-gcm` crate)
- `encrypt_and_store`: encrypt → upload to Arweave → return receipt
- `fetch_and_decrypt`: download from Arweave → find wrapped key → decrypt → verify hash

Feature flag:
```toml
[features]
default = ["poseidon-m31-encryption"]
poseidon-m31-encryption = ["stwo-ml-vm31"]
aes-fallback = ["aes-gcm"]
```

Tests:
- [ ] Encrypt → decrypt round-trip
- [ ] Hash before encryption matches after decryption
- [ ] Grant access → grantee can decrypt
- [ ] Revoke → old key fails on re-encrypted blob

---

#### B7. E2E Orchestrator
**File**: `src/audit/orchestrator.rs` (~200 lines)
**Time**: 2 days (week 4, integration with Dev A)

Wire everything together into a single `run_audit` function:

- [ ] `run_audit(log, model, request, encryption, storage) -> Result<AuditReport>`

Steps:
```
1. log.query_window(start, end)              → entries
2. prover.prove_window(log, request)         → BatchAuditResult      (Dev A)
3. evaluate_batch(model, entries)            → AuditSemanticSummary   (Dev B)
4. AuditReportBuilder::new()
     .with_audit_result(&result)
     .with_semantic_eval(&eval)
     .with_infrastructure()
     .build()                                → AuditReport
5. encrypt_and_store(report, pubkey, arweave) → StorageReceipt
6. submit_audit_onchain(result, config)      → SubmitReceipt          (Dev A)
7. Update report with on-chain tx hash
```

- [ ] Steps 2 and 3 run in parallel (prover on GPU, eval on CPU or second GPU)
- [ ] CLI command: `prove_model audit --start "2h ago" --end now --model-name qwen3-14b`

Tests:
- [ ] Full E2E with small model (2-layer MLP)
- [ ] Parallel prover + evaluator both complete
- [ ] Report is valid, encrypted, uploaded, on-chain record submitted
- [ ] Dry-run mode (no on-chain, no Arweave)

---

### Dev B Summary

| Task | File(s) | Days | Depends On |
|------|---------|------|------------|
| B1. Deterministic checks | `audit/deterministic.rs` | 2 | types.rs (day 0) |
| B2. Self-evaluation | `audit/self_eval.rs` | 3 | B1 |
| B3. Aggregate scoring | `audit/scoring.rs` | 1 | B1, B2 |
| B4. Report builder | `audit/report.rs` | 3 | types.rs |
| B5. Arweave client | `audit/storage.rs` | 2 | — |
| B6. Encryption integration | `audit/encryption.rs` | 2 | VM31 dev (week 2) |
| B7. E2E orchestrator | `audit/orchestrator.rs` | 2 | A4, B4, B5, B6 |
| **Total** | | **~15 days (3 weeks)** | |

---

## Integration Points

### Week 1 Sync (end of week)

Both devs meet to verify:
- [ ] Dev A's `InferenceLog` writes entries that Dev B's evaluator can read
- [ ] `InferenceLogEntry` serialization is compatible
- [ ] Merkle root computation matches between log and report builder

### Week 2 Sync (end of week)

- [ ] Dev A's `BatchAuditResult` feeds into Dev B's `AuditReportBuilder`
- [ ] VM31 dev delivers `PoseidonM31Encryption` implementing `AuditEncryption` trait
- [ ] Dev B verifies encryption trait works end-to-end

### Week 3 Sync (merge branches)

- [ ] Merge `feat/audit-inference-log` and `feat/audit-evaluation-report`
- [ ] Dev B builds orchestrator (B7) using both codepaths
- [ ] Dev A deploys Cairo contract to Sepolia testnet

### Week 4 (E2E testing)

- [ ] Full pipeline: serve → log → audit → prove → evaluate → report → encrypt → Arweave → on-chain
- [ ] Test with real model (phi3-mini for speed, qwen3-14b for production)
- [ ] Marketplace dev integrates: audit browser, decrypt flow

---

## File Tree (Final)

```
src/audit/
  mod.rs                  ← module declarations
  types.rs                ← shared types (day 0, both devs)
  log.rs                  ← A1: inference log
  capture.rs              ← A2: capture hook
  replay.rs               ← A3: replay verification
  prover.rs               ← A4: batch audit prover
  submit.rs               ← A5: on-chain submission
  deterministic.rs        ← B1: deterministic checks
  self_eval.rs            ← B2: self-evaluation
  scoring.rs              ← B3: aggregate scoring
  report.rs               ← B4: report builder + hash
  storage.rs              ← B5: Arweave client
  encryption.rs           ← B6: encryption integration
  orchestrator.rs         ← B7: E2E wiring

libs/elo-cairo-verifier/src/
  audit.cairo              ← A6: on-chain audit contract
  tests/test_audit.cairo   ← A6: contract tests
```

---

## Dependencies to Add

```toml
# Cargo.toml additions
[dependencies]
crossbeam = "0.8"          # A2: channel for capture hook
reqwest = { version = "0.12", features = ["json"] }  # B5: Arweave HTTP
chrono = "0.4"             # B4: timestamp formatting

[dev-dependencies]
tempfile = "3"             # A1, B5: test directories

[features]
audit = []                 # Feature gate the whole module
poseidon-m31-encryption = []  # VM31 encryption (default)
aes-fallback = ["aes-gcm"]   # Fallback if VM31 not ready
```

---

## Definition of Done

The audit system is done when this command works:

```bash
# 1. Model serves inferences (capture hook running)
./serve_model.sh --model qwen3-14b --capture

# 2. After some time, trigger audit
./prove_model audit \
  --start "1h ago" \
  --end now \
  --model-name qwen3-14b \
  --evaluate \
  --submit

# Output:
# Queried 147 inferences from log [10:00 → 11:00]
# Chain integrity: verified
# Replaying inferences... 147/147
# Proving batch (GKR mode)... 127s
# Semantic evaluation... 89% avg quality
# Report generated: audit_0xa7b3.json
# Encrypted (poseidon_m31) → Arweave tx: bIj9E1os...
# On-chain: submit_audit TX 0x9e1f... ✓ verified
# Audit ID: 0x003f...
# Explorer: https://sepolia.starkscan.co/tx/0x9e1f...
```
