# Stage 4: Audit Report Format

**Status**: Design Phase
**Readiness**: 10% — Schema design, no existing code
**Depends on**: Stage 2 (Audit Proving), Stage 3 (Semantic Evaluation)
**Blocks**: Stage 5 (On-Chain Contract), Stage 6 (Privacy/Encryption)

---

## Purpose

The audit report is a comprehensive JSON document containing everything about an audit session: what was proven, for which model, on what hardware, with what semantic quality, and where the on-chain proof lives. It is the off-chain counterpart to the compact on-chain `AuditRecord`.

The report is designed for three audiences:
1. **Operators** — full detail for internal review
2. **Auditors** — cryptographic binding to on-chain record for independent verification
3. **Smart contracts** — report hash stored on-chain, linking off-chain detail to on-chain truth

---

## Full Report Schema

```json
{
  "version": "1.0.0",
  "audit_id": "0xa7b3c4d5...",
  "schema": "obelysk-audit-report-v1",

  "time_window": {
    "start": "2026-02-14T10:00:00Z",
    "end": "2026-02-14T11:00:00Z",
    "start_epoch_ns": 1707926400000000000,
    "end_epoch_ns": 1707930000000000000,
    "duration_seconds": 3600
  },

  "model": {
    "model_id": "0x2",
    "name": "Qwen3-14B",
    "version": "v1.0",
    "architecture": "transformer",
    "parameters": "14.7B",
    "layers": 40,
    "hidden_dim": 5120,
    "num_heads": 40,
    "context_length": 8192,
    "vocab_size": 151936,
    "quantization": {
      "field": "M31",
      "strategy": "symmetric8",
      "bits": 8
    },
    "weight_commitment": "0x5a3f...",
    "source": "Qwen/Qwen3-14B",
    "weight_hash_method": "poseidon_hash_many"
  },

  "infrastructure": {
    "gpu": {
      "device": "NVIDIA H100 80GB",
      "count": 4,
      "cuda_version": "12.4",
      "driver_version": "550.54.14",
      "vram_total_gb": 320,
      "compute_capability": "9.0"
    },
    "system": {
      "os": "Ubuntu 22.04",
      "kernel": "5.15.0-91-generic",
      "cpu": "AMD EPYC 9654 96-Core",
      "ram_gb": 768
    },
    "prover": {
      "name": "stwo-ml",
      "version": "0.2.0",
      "mode": "gkr",
      "backend": "gpu"
    },
    "tee": {
      "available": true,
      "active": true,
      "level": "zk+tee",
      "device": "H100",
      "secure_boot": true,
      "debug_status": "disabled-since-boot",
      "attestation_hash": "0x7f8e..."
    }
  },

  "inference_summary": {
    "total_inferences": 147,
    "total_input_tokens": 50274,
    "total_output_tokens": 131097,
    "avg_input_tokens": 342,
    "avg_output_tokens": 891,
    "max_input_tokens": 4096,
    "max_output_tokens": 8192,
    "avg_latency_ms": 234,
    "p50_latency_ms": 189,
    "p95_latency_ms": 892,
    "p99_latency_ms": 1203,
    "peak_latency_ms": 1845,
    "throughput_tokens_per_sec": 48.2,
    "categories": {
      "question_answering": 89,
      "code_generation": 23,
      "summarization": 35
    }
  },

  "semantic_evaluation": {
    "method": "self_evaluation",
    "evaluator_model": "same",
    "evaluator_weight_commitment": "0x5a3f...",
    "avg_quality_score": 0.89,
    "distribution": {
      "excellent_0.9_to_1.0": 98,
      "good_0.7_to_0.9": 37,
      "fair_0.5_to_0.7": 10,
      "poor_0.0_to_0.5": 2
    },
    "deterministic_checks": {
      "total": 23,
      "passed": 21,
      "failed": 2,
      "check_types": ["json_valid", "code_compiles", "math_correct"]
    },
    "evaluation_proved": true,
    "eval_merkle_root": "0xb2c3..."
  },

  "commitments": {
    "inference_log_merkle_root": "0xd41a...",
    "io_merkle_root": "0x8f2c...",
    "weight_commitment": "0x5a3f...",
    "combined_chain_commitment": "0xe5f6...",
    "audit_report_hash": "0xa7b3...",
    "receipt_chain_hash": "0x1234...",
    "quantize_params_commitment": "0x9a0b..."
  },

  "proof": {
    "mode": "gkr",
    "proving_time_seconds": 127,
    "proving_gpu_hours": 0.141,
    "proof_size_bytes": 452300,
    "inference_proofs_count": 147,
    "receipt_proof_included": true,
    "on_chain": {
      "network": "starknet-sepolia",
      "contract": "0x005928ac548dc2719ef1b34869db2b61c2a55a4b148012fad742262a8d674fba",
      "tx_hash": "0x9e1f...",
      "block_number": 847291,
      "verified": true,
      "gas_sponsored": true,
      "gas_paid_by": "avnu_paymaster",
      "explorer_url": "https://sepolia.starkscan.co/tx/0x9e1f..."
    },
    "audit_record_id": "0x003f...",
    "submitted_at": "2026-02-14T11:02:07Z"
  },

  "privacy": {
    "tier": "selective_disclosure",
    "encryption": "aes-256-gcm",
    "key_wrapping": "starknet_ec_elgamal",
    "access_list": [
      {
        "address": "0x04a7...",
        "role": "owner",
        "granted_at": "2026-02-14T11:02:07Z"
      },
      {
        "address": "0x0812...",
        "role": "auditor",
        "granted_at": "2026-02-14T12:00:00Z"
      }
    ],
    "arweave_tx_id": "bIj9E1osYl_56bJejIVS-bLMfmhPMAlOxRvPsUvKzWI",
    "storage_backend": "arweave",
    "marketplace_cache": true
  },

  "inferences": [
    {
      "index": 0,
      "sequence": 0,
      "timestamp": "2026-02-14T10:00:12Z",
      "input_hash": "0xf1e2...",
      "output_hash": "0xd3c4...",
      "io_commitment": "0xa1b2...",
      "layer_chain_commitment": "0xc3d4...",
      "input_tokens": 45,
      "output_tokens": 234,
      "latency_ms": 189,
      "category": "question_answering",
      "semantic_score": 0.95,
      "input_preview": "What causes rain?",
      "output_preview": "Rain is caused by the condensation of water vapor..."
    }
  ],

  "billing": {
    "total_gpu_time_ms": 34398,
    "total_tokens_processed": 181371,
    "total_sage_billed": 523400,
    "billing_rate_per_sec": 100,
    "billing_rate_per_token": 2,
    "billing_proof_verified": true
  },

  "metadata": {
    "generated_at": "2026-02-14T11:02:07Z",
    "generator": "stwo-ml 0.2.0",
    "report_format": "obelysk-audit-report-v1",
    "report_hash_method": "poseidon_hash_many"
  }
}
```

---

## On-Chain / Off-Chain Split

### On-Chain (Compact — `AuditRecord`)

What gets stored permanently on Starknet:

```
model_id:                   0x2
audit_report_hash:          0xa7b3...   ← Poseidon of full report
inference_log_merkle_root:  0xd41a...   ← Merkle root of log entries
weight_commitment:          0x5a3f...   ← Proves which weights
time_start:                 1707926400  ← Window start (epoch)
time_end:                   1707930000  ← Window end (epoch)
inference_count:            147         ← How many inferences
proof_verified:             true        ← ZK proof passed
submitter:                  0x04a7...   ← Who submitted
```

~10 felt252s. Constant size regardless of inference count.

### Off-Chain (Detailed — Full Report)

What lives on the operator's storage (or IPFS/Bitsage Marketplace):

- Full JSON report (schema above)
- Inference log (raw entries with full input/output)
- Binary M31 matrices (for re-proving)
- Evaluation details

The `audit_report_hash` on-chain binds to this off-chain data. Anyone can fetch the report, hash it, and verify it matches what's on-chain.

---

## Report Hash Computation

```rust
/// Compute the Poseidon hash of an audit report.
///
/// This hash is stored on-chain as `audit_report_hash` in the AuditRecord.
/// It binds the full off-chain report to the on-chain record.
///
/// Hashes the canonical JSON serialization (sorted keys, no whitespace).
pub fn compute_report_hash(report: &AuditReport) -> FieldElement {
    let canonical_json = serde_json::to_string(report).unwrap();
    let bytes = canonical_json.as_bytes();

    // Chunk bytes into 31-byte felt252 segments
    let mut felts = Vec::new();
    for chunk in bytes.chunks(31) {
        let mut padded = [0u8; 32];
        padded[32 - chunk.len()..].copy_from_slice(chunk);
        felts.push(FieldElement::from_bytes_be(&padded).unwrap_or(FieldElement::ZERO));
    }

    poseidon_hash_many(&felts)
}
```

---

## Rust Types

```rust
/// Complete audit report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub version: String,
    pub audit_id: String,
    pub time_window: TimeWindow,
    pub model: ModelInfo,
    pub infrastructure: InfrastructureInfo,
    pub inference_summary: InferenceSummary,
    pub semantic_evaluation: Option<SemanticEvaluation>,
    pub commitments: AuditCommitments,
    pub proof: ProofInfo,
    pub privacy: Option<PrivacyInfo>,
    pub inferences: Vec<InferenceEntry>,
    pub billing: Option<BillingInfo>,
    pub metadata: ReportMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start: String,  // ISO 8601
    pub end: String,
    pub start_epoch_ns: u64,
    pub end_epoch_ns: u64,
    pub duration_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditCommitments {
    pub inference_log_merkle_root: String,
    pub io_merkle_root: String,
    pub weight_commitment: String,
    pub combined_chain_commitment: String,
    pub audit_report_hash: String,
    pub receipt_chain_hash: Option<String>,
    pub quantize_params_commitment: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceEntry {
    pub index: u32,
    pub sequence: u64,
    pub timestamp: String,
    pub input_hash: String,
    pub output_hash: String,
    pub io_commitment: String,
    pub layer_chain_commitment: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u64,
    pub category: Option<String>,
    pub semantic_score: Option<f32>,
    pub input_preview: Option<String>,
    pub output_preview: Option<String>,
}
```

---

## Storage Architecture: Arweave + Bitsage Marketplace

Encrypted audit reports are stored on **Arweave** for permanent, decentralized, cheap storage. The **Bitsage Marketplace** acts as the UX/caching layer on top — indexing audits per-user, providing fast access, and managing the access control UI.

```
Audit complete
  → Encrypt report (AES-256-GCM)
  → Upload encrypted blob to Arweave (~$0.005, permanent) → arweave_tx_id
  → Index in Bitsage Marketplace DB (fast lookup, per-wallet)
  → Submit AuditRecord on-chain (report_hash + arweave_tx_id)

User reads report:
  → Marketplace UI lists their audits (SQL query by wallet)
  → Click → fetch from Marketplace cache (or Arweave gateway fallback)
  → Decrypt with wrapped key from on-chain contract
  → Verify poseidon(report) == on-chain report_hash
```

### Why This Split

| Layer | Purpose | Cost | Permanence |
|-------|---------|------|-----------|
| **Starknet** | Compact anchor (hashes, metadata, ~320 bytes) | ~$0.02/audit (paymaster-sponsored) | Permanent |
| **Arweave** | Encrypted blob (full report, 1-5MB) | ~$0.005/audit (one-time) | Permanent, decentralized |
| **Marketplace** | UX layer (index, cache, file browser, access control UI) | Infra cost | While service runs |

- **Arweave** is the source of truth for the encrypted blob — content-addressed, tamper-proof, no monthly bills, no vendor lock-in. Even if the Marketplace goes down, the data is recoverable from Arweave.
- **Marketplace** provides the user-facing experience — wallet-authenticated file browser, search/filter by model/date/status, share/revoke UI, fast cached access.
- **Starknet** provides the cryptographic anchor — report_hash on-chain binds to the Arweave blob, verifiable by anyone.

### Storage Backend

```rust
/// Upload an audit report to storage.
pub async fn store_report(
    report: &AuditReport,
    storage: &StorageBackend,
    encryption: Option<&EncryptionConfig>,
) -> Result<StorageReceipt, StorageError> {
    let serialized = serde_json::to_vec(report)?;

    let data = match encryption {
        Some(config) => encrypt_report(&serialized, config)?,
        None => serialized,
    };

    storage.upload(&data, &report.audit_id).await
}

pub enum StorageBackend {
    /// Local filesystem (~/.obelysk/reports/) — dev/testing only.
    Local(PathBuf),
    /// Arweave permanent storage — primary backend for production.
    /// Returns arweave_tx_id (content-addressed, ~$0.005/MB, permanent).
    Arweave { gateway: String, wallet_jwk: PathBuf },
    /// Bitsage Marketplace — caching/indexing layer on top of Arweave.
    /// Stores metadata in PostgreSQL, caches blobs for fast access,
    /// falls back to Arweave gateway if cache miss.
    BitsageMarketplace {
        api_url: String,
        wallet_address: String,
        arweave_gateway: String,  // Fallback for cache misses
    },
}

/// Receipt from a successful storage upload.
pub struct StorageReceipt {
    /// Arweave transaction ID (permanent, content-addressed).
    pub arweave_tx_id: Option<String>,
    /// Marketplace file ID (for fast lookup via API).
    pub marketplace_file_id: Option<String>,
    /// Local file path (dev only).
    pub local_path: Option<PathBuf>,
    /// Size of the stored blob in bytes.
    pub size_bytes: usize,
    /// Timestamp of upload.
    pub uploaded_at: u64,
}
```

### Cost per Audit

| Component | Cost | Notes |
|-----------|------|-------|
| On-chain `AuditRecord` | ~$0.02 | Paymaster-sponsored on Sepolia |
| Arweave blob (1MB encrypted report) | ~$0.005 | One-time, permanent |
| Marketplace indexing | $0 | Internal infra |
| **Total per audit** | **~$0.03** | |

For 1,000 audits/month: ~$30/month total. Compare to S3 alone at ~$0.023/GB/month (similar cost but not permanent, not decentralized).

---

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `src/audit/report.rs` | **New** | ~400 (report types + serialization) |
| `src/audit/report_hash.rs` | **New** | ~50 (Poseidon hash computation) |
| `src/audit/storage.rs` | **New** | ~250 (Arweave + Marketplace backends) |

---

## Verification Criteria

- [ ] Report JSON is valid and parseable by any JSON parser
- [ ] `compute_report_hash(report)` is deterministic for same report content
- [ ] Report hash matches what would be stored on-chain
- [ ] All commitment fields in report match values from the prover
- [ ] Inference entries have consistent ordering (by sequence number)
- [ ] Time window fields are consistent (end - start = duration)
- [ ] Report round-trips through serialize/deserialize without data loss
- [ ] Arweave upload returns a valid tx_id that resolves via gateway
- [ ] Marketplace indexes the Arweave tx_id for fast wallet-scoped lookup
- [ ] Cache miss on Marketplace falls back to Arweave gateway transparently
