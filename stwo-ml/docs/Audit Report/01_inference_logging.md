# Stage 1: Inference Logging

**Status**: Design Ready
**Readiness**: 80% — `ComputeReceipt` + chain-linking exist, need log format + capture hook
**Depends on**: Nothing (foundation stage)
**Blocks**: Stage 2 (Audit Proving), Stage 4 (Audit Report Format)

---

## Purpose

Capture every inference during real-time model serving into an append-only, chain-linked, Merkle-committed log. The log is the **source of truth** for all subsequent auditing — it binds real user queries to real model outputs with cryptographic commitments.

The inference log lives entirely on the operator's machine. Nothing leaves the server during normal serving. When an audit is requested, the log provides the raw data for replay and proving.

---

## Existing Infrastructure

### `ComputeReceipt` (`src/receipt.rs`)

Already has most of what we need for a per-inference record:

```rust
pub struct ComputeReceipt {
    pub job_id: FieldElement,           // Unique inference ID
    pub worker_pubkey: FieldElement,     // Server identity
    pub input_commitment: FieldElement,  // Poseidon(input)
    pub output_commitment: FieldElement, // Poseidon(output)
    pub model_commitment: FieldElement,  // Poseidon(weights)
    pub prev_receipt_hash: FieldElement, // Chain link
    pub gpu_time_ms: u64,               // Execution time
    pub token_count: u32,               // Tokens processed
    pub peak_memory_mb: u32,            // GPU memory
    pub billing_amount_sage: u64,        // Billing
    pub billing_rate_per_sec: u64,
    pub billing_rate_per_token: u64,
    pub tee_report_hash: FieldElement,   // TEE attestation
    pub tee_timestamp: u64,
    pub timestamp: u64,                  // Unix epoch
    pub sequence_number: u32,            // Position in chain
}
```

### Chain Linking (`src/receipt.rs`)

`receipt_hash()` computes Poseidon over all fields. `prev_receipt_hash` links to previous entry. `verify_receipt_chain()` validates the full chain.

### IO Commitment (`src/aggregation.rs`)

`compute_io_commitment(input, output) -> FieldElement` binds proof to specific I/O data.

---

## Design

### Inference Log Entry

Extends `ComputeReceipt` with inference-specific fields needed for audit replay:

```rust
/// A single inference record in the append-only log.
///
/// Contains everything needed to replay the inference during audit proving:
/// the tokenized input/output, the model identifier, and all commitments
/// that will be verified in the aggregated proof.
pub struct InferenceLogEntry {
    // === Identity ===
    /// Unique inference ID (UUID or sequential).
    pub inference_id: u64,
    /// Position in the log (0-indexed, monotonically increasing).
    pub sequence_number: u64,

    // === Model ===
    /// Model identifier (matches on-chain model_id).
    pub model_id: FieldElement,
    /// Poseidon hash of all weight matrices.
    pub weight_commitment: FieldElement,
    /// Model name (human-readable, not hashed).
    pub model_name: String,
    /// Number of layers in the model.
    pub num_layers: u32,
    /// Architecture type (transformer, cnn, mlp).
    pub architecture: String,

    // === Inference Data ===
    /// Tokenized input (raw token IDs before M31 quantization).
    pub input_tokens: Vec<u32>,
    /// Tokenized output (raw token IDs).
    pub output_tokens: Vec<u32>,
    /// M31-quantized input matrix (for prover replay).
    pub input_m31: M31Matrix,
    /// M31-quantized output matrix (computed during inference).
    pub output_m31: M31Matrix,

    // === Commitments ===
    /// Poseidon(input || output) — binds proof to this specific I/O.
    pub io_commitment: FieldElement,
    /// Running Poseidon hash through all layer intermediates.
    pub layer_chain_commitment: FieldElement,

    // === Chain Link ===
    /// Poseidon hash of the previous log entry (0x0 for first).
    pub prev_entry_hash: FieldElement,

    // === Metadata ===
    /// Inference start timestamp (Unix epoch nanoseconds).
    pub timestamp_ns: u64,
    /// Inference latency in milliseconds.
    pub latency_ms: u64,
    /// GPU device used.
    pub gpu_device: String,
    /// TEE attestation hash (0x0 if no TEE).
    pub tee_report_hash: FieldElement,
    /// Task category (classification, generation, etc.) — for report aggregation.
    pub task_category: Option<String>,
    /// Input preview (first 100 chars of detokenized input, for report).
    pub input_preview: Option<String>,
    /// Output preview (first 200 chars of detokenized output, for report).
    pub output_preview: Option<String>,
}

impl InferenceLogEntry {
    /// Compute the Poseidon hash of this entry (used for chain linking).
    pub fn entry_hash(&self) -> FieldElement {
        poseidon_hash_many(&[
            FieldElement::from(self.inference_id),
            FieldElement::from(self.sequence_number),
            self.model_id,
            self.weight_commitment,
            self.io_commitment,
            self.layer_chain_commitment,
            self.prev_entry_hash,
            FieldElement::from(self.timestamp_ns),
            FieldElement::from(self.latency_ms),
            self.tee_report_hash,
        ])
    }
}
```

### Inference Log

The log itself is an append-only file + in-memory Merkle tree:

```rust
/// Append-only inference log with Merkle commitment.
///
/// During serving, entries are appended to a file on disk and to an
/// in-memory Merkle tree. The Merkle root changes with each append,
/// enabling audit proofs over any subset of entries.
pub struct InferenceLog {
    /// Path to the log file on disk.
    pub log_path: PathBuf,
    /// Poseidon-based Merkle tree of entry hashes.
    pub merkle_tree: PoseidonMerkleTree,
    /// Current Merkle root.
    pub merkle_root: FieldElement,
    /// Number of entries.
    pub entry_count: u64,
    /// Hash of the most recent entry (for chain linking).
    pub latest_entry_hash: FieldElement,
    /// Weight commitment (constant for a model session).
    pub weight_commitment: FieldElement,
    /// Model identifier.
    pub model_id: FieldElement,
}

impl InferenceLog {
    /// Create a new log for a model session.
    pub fn new(log_dir: &Path, model_id: FieldElement, weight_commitment: FieldElement) -> Self;

    /// Append an inference entry to the log.
    ///
    /// 1. Computes entry_hash()
    /// 2. Appends entry to disk (streaming JSON lines)
    /// 3. Inserts entry_hash into Merkle tree
    /// 4. Updates merkle_root and latest_entry_hash
    ///
    /// This is designed to be non-blocking: serialization and disk I/O
    /// happen on a background thread. The Merkle tree update is O(log N).
    pub fn append(&mut self, entry: InferenceLogEntry) -> Result<(), LogError>;

    /// Query entries within a time window.
    pub fn query_window(&self, start_ns: u64, end_ns: u64) -> Vec<&InferenceLogEntry>;

    /// Get the Merkle root covering all entries up to a given sequence number.
    pub fn merkle_root_at(&self, seq: u64) -> FieldElement;

    /// Get a Merkle proof for a specific entry.
    pub fn merkle_proof(&self, seq: u64) -> Vec<FieldElement>;

    /// Verify the chain link integrity of all entries.
    pub fn verify_chain(&self) -> Result<(), LogError>;
}
```

### Capture Hook

The capture hook sits between the model server and the log. It intercepts inference calls without adding latency to the serving path:

```rust
/// Hook that captures inference I/O for the log.
///
/// Designed to wrap any inference server (vLLM, TensorRT-LLM, llama.cpp).
/// The hook:
/// 1. Receives (input_tokens, output_tokens) after inference completes
/// 2. Quantizes to M31 (same quantization the prover uses)
/// 3. Computes io_commitment and layer_chain_commitment
/// 4. Appends InferenceLogEntry to the log
///
/// The commitment computation runs on a background thread — it does NOT
/// block the response to the user.
pub struct InferenceCaptureHook {
    log: Arc<Mutex<InferenceLog>>,
    model: Arc<OnnxModel>,        // For M31 quantization params
    background_tx: Sender<CaptureJob>,
}

struct CaptureJob {
    input_tokens: Vec<u32>,
    output_tokens: Vec<u32>,
    timestamp_ns: u64,
    latency_ms: u64,
    gpu_device: String,
    task_category: Option<String>,
}

impl InferenceCaptureHook {
    /// Create a hook for a model session.
    ///
    /// Spawns a background thread for commitment computation and log I/O.
    pub fn new(log_dir: &Path, model: Arc<OnnxModel>) -> Self;

    /// Record an inference. Non-blocking — returns immediately.
    ///
    /// The actual commitment computation and log append happen asynchronously.
    pub fn record(&self, job: CaptureJob);

    /// Flush all pending records to disk. Blocks until complete.
    pub fn flush(&self);
}
```

### Log File Format

Each entry is a single JSON line (JSON Lines format) for streaming append:

```jsonl
{"seq":0,"ts":1707926400000000000,"model_id":"0x2","io_commitment":"0xa1b2...","prev_hash":"0x0","input_tokens":[1,234,567,...],"output_tokens":[89,101,...],"latency_ms":234,"gpu":"H100","category":"qa","input_preview":"What causes rain?","output_preview":"Rain is caused by..."}
{"seq":1,"ts":1707926401234000000,"model_id":"0x2","io_commitment":"0xc3d4...","prev_hash":"0x8f2c...","input_tokens":[2,345,678,...],"output_tokens":[90,102,...],"latency_ms":189,"gpu":"H100","category":"code","input_preview":"Write a Python function","output_preview":"def foo():..."}
```

The M31 matrices are stored separately in a binary sidecar file (too large for JSON):

```
~/.obelysk/logs/
  session_20260214T100000/
    log.jsonl              <- JSON Lines (metadata + commitments)
    matrices.bin           <- Binary M31 matrices (input/output for each entry)
    merkle.bin             <- Serialized Merkle tree state
    meta.json              <- Session metadata (model name, weight_commit, etc.)
```

---

## Merkle Tree Design

The Merkle tree uses Poseidon hashing (same as on-chain) for consistency:

```
                    Root
                   /    \
              H(0,1)    H(2,3)
              /    \    /    \
           h(e0) h(e1) h(e2) h(e3)
```

- **Hash function**: `starknet_crypto::poseidon_hash_many` (felt252)
- **Leaf**: `entry_hash()` of each `InferenceLogEntry`
- **Depth**: Dynamic (grows as entries are appended)
- **Empty leaf**: `FieldElement::ZERO`
- **Append-only**: Entries never modified or deleted
- **Proof size**: O(log N) felt252 values per entry

**Future (VM31)**: When Poseidon-M31 is ready (12.2x cheaper), the Merkle tree can switch to native M31 hashing. The on-chain verifier would need to support Poseidon-M31 roots.

---

## Integration with Existing Code

### Reuse `ComputeReceipt` for Billing

The billing fields from `ComputeReceipt` map directly into the log:

```rust
impl From<&InferenceLogEntry> for ComputeReceipt {
    fn from(entry: &InferenceLogEntry) -> Self {
        ComputeReceipt {
            job_id: FieldElement::from(entry.inference_id),
            worker_pubkey: FieldElement::ZERO, // Set by server
            input_commitment: entry.io_commitment, // Simplified
            output_commitment: entry.io_commitment,
            model_commitment: entry.weight_commitment,
            prev_receipt_hash: entry.prev_entry_hash,
            gpu_time_ms: entry.latency_ms,
            token_count: (entry.input_tokens.len() + entry.output_tokens.len()) as u32,
            peak_memory_mb: 0, // Captured at runtime
            billing_amount_sage: 0, // Computed by billing module
            billing_rate_per_sec: 0,
            billing_rate_per_token: 0,
            tee_report_hash: entry.tee_report_hash,
            tee_timestamp: entry.timestamp_ns / 1_000_000_000,
            timestamp: entry.timestamp_ns / 1_000_000_000,
            sequence_number: entry.sequence_number as u32,
        }
    }
}
```

### Reuse `compute_io_commitment`

The existing function from `aggregation.rs` computes exactly what we need:

```rust
// During inference capture:
let io_commitment = compute_io_commitment(&input_m31, &output_m31);
```

### Reuse `verify_receipt_chain`

The chain-linking verification in `receipt.rs` uses the same pattern — each entry links to the previous via Poseidon hash.

---

## Performance Considerations

| Operation | Time | Blocking? |
|-----------|------|-----------|
| M31 quantization of input | ~0.1ms per 1K tokens | No (background) |
| `compute_io_commitment` | ~0.5ms per inference | No (background) |
| Merkle tree insert | ~0.01ms | No (background) |
| JSON line serialization | ~0.05ms | No (background) |
| Disk I/O (append) | ~0.1ms | No (background) |
| **Total overhead on serving** | **0ms** | **Everything is async** |

The capture hook sends a message to a background thread and returns immediately. The user sees zero additional latency.

---

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `src/inference_log.rs` | **New** | ~400 |
| `src/capture_hook.rs` | **New** | ~200 |
| `src/lib.rs` | **Modify** | +2 (module declarations) |
| `src/receipt.rs` | **Modify** | +20 (`From<InferenceLogEntry>` impl) |

---

## Verification Criteria

- [ ] `InferenceLog::append` is non-blocking (background thread)
- [ ] Chain link verification passes for 1000+ entries
- [ ] Merkle root is deterministic for the same entry sequence
- [ ] Merkle proof verifies for any entry in the log
- [ ] `InferenceLogEntry -> ComputeReceipt` conversion preserves all commitment fields
- [ ] Log file format is human-readable (JSON Lines)
- [ ] Binary matrix sidecar enables exact replay without re-quantization
- [ ] Log survives process restart (disk persistence + recovery)
