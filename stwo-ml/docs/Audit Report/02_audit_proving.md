# Stage 2: Audit Proving

**Status**: Core Exists
**Readiness**: 60% — `prove_model` + `AggregatedModelProof` exist, need batch wrapper + replay
**Depends on**: Stage 1 (Inference Logging)
**Blocks**: Stage 4 (Audit Report Format), Stage 5 (On-Chain Contract)

---

## Purpose

When an audit is triggered ("prove the last hour"), the audit prover:

1. Reads the inference log for the requested time window
2. Replays each inference through the model (re-computing all intermediates)
3. Generates a cryptographic proof for each inference
4. Aggregates all proofs into a single batch proof
5. Produces a Merkle root binding the batch to the inference log

The result is a single on-chain submission that proves N inferences were computed correctly in time window [t_start, t_end].

---

## Existing Infrastructure

### `AggregatedModelProof` (`src/aggregation.rs`)

Already proves a single forward pass with full commitment binding:

```rust
pub struct AggregatedModelProofFor<H: MerkleHasherLifted> {
    pub unified_stark: Option<StarkProof<H>>,        // All non-matmul components
    pub matmul_proofs: Vec<(usize, MatMulSumcheckProof)>,  // Per-matmul sumchecks
    pub execution: GraphExecution,                    // Forward pass trace
    pub layer_chain_commitment: FieldElement,         // Running hash
    pub io_commitment: FieldElement,                  // Poseidon(input||output)
    pub layernorm_mean_var_commitments: Vec<FieldElement>,
    pub quantize_params_commitment: FieldElement,
    // ... claims for each layer type
}
```

### `prove_model` function

Existing prover takes a model + input and produces `AggregatedModelProof`. Located in `src/compiler/prove.rs` (CPU path) and `src/gpu/` (GPU path).

### `prove_receipt_batch` (`src/receipt.rs`)

Already batch-proves receipts with a single STARK. Uses same commitment scheme pattern.

### GKR Prover (`src/aggregation.rs`)

GKR mode (`prove_model_gkr`) is the fastest path — uses GKR sumcheck for matmuls instead of individual sumcheck proofs.

---

## Design

### Audit Request

```rust
/// An audit request specifying what to prove.
pub struct AuditRequest {
    /// Start of audit window (Unix epoch nanoseconds).
    pub start_ns: u64,
    /// End of audit window (Unix epoch nanoseconds).
    pub end_ns: u64,
    /// Model ID (must match log entries).
    pub model_id: FieldElement,
    /// Proving mode.
    pub mode: AuditProofMode,
    /// Whether to run semantic evaluation (Stage 3).
    pub evaluate_semantics: bool,
    /// Maximum number of inferences to prove (0 = all in window).
    pub max_inferences: usize,
    /// GPU device index (for multi-GPU).
    pub gpu_device: Option<usize>,
}

/// Proof mode for the audit.
pub enum AuditProofMode {
    /// GKR sumcheck — fastest, recommended.
    Gkr,
    /// Direct STARK per matmul — moderate speed.
    Direct,
    /// Recursive composition — slowest, highest security.
    Recursive,
}
```

### Batch Audit Proof

```rust
/// Aggregated proof covering all inferences in an audit window.
pub struct BatchAuditProof {
    // === Window ===
    /// Audit window start (Unix epoch seconds).
    pub time_start: u64,
    /// Audit window end (Unix epoch seconds).
    pub time_end: u64,
    /// Number of inferences proven.
    pub inference_count: u32,

    // === Per-Inference Proofs ===
    /// Individual proof for each inference (GKR or STARK).
    /// Each proves: correct forward pass, io_commitment, layer_chain_commitment.
    pub inference_proofs: Vec<InferenceProof>,

    // === Aggregated Commitments ===
    /// Merkle root of all io_commitments in the batch.
    pub io_merkle_root: FieldElement,
    /// Merkle root of the inference log entries in the window.
    pub log_merkle_root: FieldElement,
    /// Weight commitment (must be constant across all inferences).
    pub weight_commitment: FieldElement,
    /// Combined layer chain commitment (XOR/hash of all per-inference chain commits).
    pub combined_chain_commitment: FieldElement,

    // === Receipt Batch ===
    /// STARK proof of billing correctness for all receipts.
    pub receipt_proof: Option<ReceiptProof>,
    /// Chain of receipt hashes.
    pub receipt_chain: Vec<FieldElement>,

    // === TEE ===
    /// TEE attestation binding the audit to hardware.
    pub tee_attestation: Option<TeeAttestation>,

    // === Metadata ===
    /// Model identifier.
    pub model_id: FieldElement,
    /// Prover version.
    pub prover_version: String,
    /// Total proving time in milliseconds.
    pub proving_time_ms: u64,
}

/// Proof for a single inference within the batch.
pub struct InferenceProof {
    /// Inference sequence number in the log.
    pub sequence: u64,
    /// IO commitment for this inference.
    pub io_commitment: FieldElement,
    /// Layer chain commitment for this inference.
    pub layer_chain_commitment: FieldElement,
    /// The proof itself (GKR calldata or STARK proof).
    pub proof_data: InferenceProofData,
    /// Timestamp of the inference.
    pub timestamp: u64,
}

/// Proof data — either GKR calldata (compact) or full STARK.
pub enum InferenceProofData {
    /// GKR mode — compact calldata for on-chain verification.
    Gkr(Vec<FieldElement>),
    /// Direct STARK — full aggregated model proof.
    Stark(AggregatedModelProof),
}
```

### Audit Prover

```rust
/// The batch audit prover.
///
/// Replays inferences from the log and generates a single aggregated proof
/// covering the entire audit window.
pub struct AuditProver {
    model: Arc<OnnxModel>,
    gpu_prover: Option<GpuModelProver>,
    mode: AuditProofMode,
}

impl AuditProver {
    /// Create a prover for a specific model.
    pub fn new(model: Arc<OnnxModel>, mode: AuditProofMode) -> Self;

    /// Prove all inferences in a time window.
    ///
    /// Steps:
    /// 1. Query log for entries in [start_ns, end_ns]
    /// 2. Verify chain integrity of queried entries
    /// 3. For each entry:
    ///    a. Load M31 input from binary sidecar
    ///    b. Replay forward pass (recomputes all intermediates)
    ///    c. Verify io_commitment matches log entry
    ///    d. Generate proof (GKR or STARK)
    /// 4. Aggregate into BatchAuditProof
    /// 5. Compute batch Merkle roots
    /// 6. Optionally prove billing receipts
    /// 7. Optionally attach TEE attestation
    pub fn prove_window(
        &self,
        log: &InferenceLog,
        request: &AuditRequest,
    ) -> Result<BatchAuditProof, AuditError>;

    /// Prove a single inference (internal, called per-entry).
    fn prove_inference(
        &self,
        entry: &InferenceLogEntry,
    ) -> Result<InferenceProof, AuditError>;
}
```

### Replay Verification

Before proving, the prover verifies that the logged inference is reproducible:

```rust
/// Replay an inference and verify it matches the log entry.
///
/// This catches log corruption or tampering before spending GPU time proving.
fn verify_replay(
    model: &OnnxModel,
    entry: &InferenceLogEntry,
) -> Result<(), AuditError> {
    // 1. Re-run forward pass with logged M31 input
    let output = forward_pass(&model.graph, &entry.input_m31, &model.weights);

    // 2. Verify output matches logged output
    let replayed_io = compute_io_commitment(&entry.input_m31, &output);
    if replayed_io != entry.io_commitment {
        return Err(AuditError::ReplayMismatch {
            sequence: entry.sequence_number,
            expected: entry.io_commitment,
            actual: replayed_io,
        });
    }

    Ok(())
}
```

---

## Batch Strategies

### Strategy 1: Independent Proofs (Simple, Parallelizable)

Prove each inference independently, aggregate commitments:

```
Inference 0 ──> Proof 0 ──┐
Inference 1 ──> Proof 1 ──┤
Inference 2 ──> Proof 2 ──┤──> BatchAuditProof
...                       │    (Merkle root of all io_commitments)
Inference N ──> Proof N ──┘
```

- Each proof is independent — fully parallelizable across GPUs
- On-chain: verify each proof + verify Merkle root
- Calldata: N * per_proof_size (can be large for N > 100)

### Strategy 2: Recursive Aggregation (Compact, Sequential)

Recursively compose proofs into a single proof:

```
Proof 0 + Proof 1 ──> AggProof 01 ──┐
Proof 2 + Proof 3 ──> AggProof 23 ──┤──> FinalProof
...                                  │
Proof N-1 + Proof N ──> AggProof   ──┘
```

- Single final proof regardless of N
- On-chain: verify one proof
- Proving time: O(N log N) due to recursive composition
- Requires recursive STARK verifier circuit

### Strategy 3: Batch GKR (Recommended)

Extend GKR to prove multiple forward passes in one batch:

```
Input 0 ──┐
Input 1 ──┤──> Batch GKR Prover ──> Single GKR Proof
Input 2 ──┤    (shared weight MLE,   (covers all N inferences)
...       │     batched sumcheck)
Input N ──┘
```

- Weights are committed once (shared across all inferences)
- Sumcheck batches across inferences (amortized cost)
- Single on-chain submission
- Most efficient for same-model audits

---

## Multi-GPU Parallelism

For large audit windows (100+ inferences), distribute across GPUs:

```rust
/// Parallel audit proving across multiple GPUs.
///
/// Uses the existing MultiGpuExecutor from src/gpu/multi_gpu.rs.
/// Inferences are distributed round-robin across available GPUs.
pub fn prove_window_multi_gpu(
    model: &OnnxModel,
    entries: &[InferenceLogEntry],
    gpu_count: usize,
) -> Result<Vec<InferenceProof>, AuditError> {
    // Partition entries across GPUs
    let chunks: Vec<_> = entries.chunks((entries.len() + gpu_count - 1) / gpu_count).collect();

    // Prove in parallel (using rayon + DeviceGuard)
    let proofs: Vec<_> = chunks.par_iter()
        .enumerate()
        .map(|(gpu_id, chunk)| {
            let _guard = DeviceGuard::new(gpu_id);
            // ... prove each entry on this GPU
        })
        .collect();

    Ok(proofs.into_iter().flatten().collect())
}
```

---

## On-Chain Submission

The `BatchAuditProof` is serialized to felt252 calldata for the on-chain contract:

```rust
/// Serialize a batch audit proof to on-chain calldata.
pub fn serialize_audit_proof(proof: &BatchAuditProof) -> Vec<FieldElement> {
    let mut calldata = Vec::new();

    // Header
    calldata.push(FieldElement::from(proof.model_id));
    calldata.push(FieldElement::from(proof.time_start));
    calldata.push(FieldElement::from(proof.time_end));
    calldata.push(FieldElement::from(proof.inference_count as u64));
    calldata.push(proof.io_merkle_root);
    calldata.push(proof.log_merkle_root);
    calldata.push(proof.weight_commitment);
    calldata.push(proof.combined_chain_commitment);

    // Per-inference proof calldata
    for iproof in &proof.inference_proofs {
        calldata.push(FieldElement::from(iproof.sequence));
        calldata.push(iproof.io_commitment);
        match &iproof.proof_data {
            InferenceProofData::Gkr(data) => {
                calldata.push(FieldElement::from(data.len() as u64));
                calldata.extend(data);
            }
            InferenceProofData::Stark(_) => {
                // Serialize STARK proof using existing cairo_serde
                // ...
            }
        }
    }

    calldata
}
```

---

## Performance Estimates

| Audit Window | Inferences | Proving Time (1x H100) | Proving Time (4x H100) |
|-------------|-----------|----------------------|----------------------|
| 10 minutes | ~20 | ~30s | ~10s |
| 1 hour | ~150 | ~4min | ~1min |
| 24 hours | ~3600 | ~90min | ~25min |

Estimates assume GKR mode with GPU acceleration. Actual times depend on model size and inference complexity.

---

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `src/audit/mod.rs` | **New** | ~50 (module structure) |
| `src/audit/prover.rs` | **New** | ~500 (AuditProver, BatchAuditProof) |
| `src/audit/replay.rs` | **New** | ~150 (replay verification) |
| `src/audit/serialize.rs` | **New** | ~200 (on-chain calldata serialization) |
| `src/lib.rs` | **Modify** | +1 (module declaration) |

---

## Verification Criteria

- [ ] Replay of logged inference produces same `io_commitment`
- [ ] Batch proof covers all inferences in requested window
- [ ] Chain integrity verified before proving
- [ ] Weight commitment is constant across all inferences in batch
- [ ] Merkle root of io_commitments is deterministic
- [ ] Multi-GPU distribution produces same result as single-GPU
- [ ] Calldata serialization round-trips correctly
- [ ] Proving time scales linearly with inference count (within GPU memory)
