# Session Notes — Feb 14, 2026

## What Was Discussed

### 1. The Core Problem

The stwo-ml pipeline proves a forward pass with **random test input**. That's a tech demo — it proves the prover works but has zero practical value. Real proofs must verify real inference: actual user queries, actual model outputs, actual computation.

### 2. What We Decided

- **Audit model, not per-inference proofs.** Per-inference proofs are too slow, too expensive, and unnecessary. The right model is like financial auditing: "prove the last hour of inference."
- **Three-speed audit output.** Summary is instant (read the log). Semantic evaluation is fast (parallel forward pass). ZK proof is async (GPU proving). All three bind to the same Merkle root.
- **Privacy is fundamental.** Reports contain sensitive user queries. Encryption (AES-256-GCM) with on-chain access control. Three tiers: full privacy, selective disclosure, public.
- **On-chain = compact anchor.** The full report lives off-chain (Bitsage Marketplace). On-chain stores only: model_id, report_hash, merkle_root, time_window, inference_count, verified.
- **Zero-config on-chain submission works.** Implemented during this session — AVNU paymaster, ephemeral accounts, no env vars needed on Sepolia.

### 3. Architecture

```
HOT PATH                              AUDIT PATH
User → Model Server → Response        "Prove last hour" →
            │                                │
            v                     ┌──────────┼──────────┐
     Inference Log                v          v          v
     (append-only)           Summary     Prover    Evaluator
                             (instant)   (GPU)     (parallel)
                                │          │          │
                                └──────────┼──────────┘
                                           v
                                    Audit Report
                                    ┌─────────┐
                                    │ Encrypt  │
                                    │ Store    │
                                    │ On-chain │
                                    └─────────┘
```

### 4. What Was Implemented

| Item | Status |
|------|--------|
| AVNU paymaster integration (`paymaster_submit.mjs`) | Done, pushed to main |
| Ephemeral account deployment via `deploymentData` | Done, pushed to main |
| Zero-config `04_verify_onchain.sh` (no env vars on Sepolia) | Done, pushed to main |
| `GETTING_STARTED.md` updated | Done, pushed to main |
| `run_e2e.sh` passthrough flags | Done, pushed to main |

### 5. What Was Planned (This Document Set)

| Stage | Document | Summary |
|-------|----------|---------|
| Overview | [`00_SCOPE_OVERVIEW.md`](./00_SCOPE_OVERVIEW.md) | Full scope, readiness matrix, dependency graph |
| 1 | [`01_inference_logging.md`](./01_inference_logging.md) | Append-only Merkle log capturing every inference |
| 2 | [`02_audit_proving.md`](./02_audit_proving.md) | Batch replay + aggregated proof over time windows |
| 3 | [`03_semantic_evaluation.md`](./03_semantic_evaluation.md) | Self-evaluation + deterministic checks |
| 4 | [`04_audit_report_format.md`](./04_audit_report_format.md) | Comprehensive JSON schema with all metadata |
| 5 | [`05_onchain_audit_contract.md`](./05_onchain_audit_contract.md) | Cairo `submit_audit` + `AuditRecord` storage |
| 6 | [`06_privacy_encryption.md`](./06_privacy_encryption.md) | AES-256-GCM + ElGamal key wrapping |
| 7 | [`07_access_control.md`](./07_access_control.md) | On-chain ACL + view key delegation |

### 6. Key Technical Decisions

**Why Poseidon for commitments?** Same hash used on-chain in Cairo contracts. Off-chain commitment matches on-chain verification — no hash translation needed.

**Why AES-256-GCM + ElGamal wrapping?** AES-GCM for speed (hardware-accelerated). ElGamal for per-recipient key wrapping (same keypair as Starknet account). This is the pattern in `Obelysk-Protocol/contracts/src/elgamal.cairo`.

**Why Arweave for blob storage?** ~$0.005/MB one-time, permanent, decentralized, content-addressed. Encrypted blobs are safe on public storage — Arweave nodes store ciphertext they can't read. No monthly bills, no vendor lock-in, no "oops we deleted it." Total cost per audit: ~$0.03 (Arweave + on-chain).

**Why Bitsage Marketplace as the UX layer?** Arweave has no per-user indexing or auth. The Marketplace provides: wallet-authenticated file browser, per-user audit listing, cached access, share/revoke UI. It indexes Arweave tx_ids in PostgreSQL and falls back to Arweave gateway on cache miss. Even if the Marketplace goes down, data is recoverable from Arweave.

**Why GKR for batch proving?** GKR mode is fastest. Weights committed once, sumcheck batched across inferences. Single on-chain submission regardless of inference count.

### 7. Existing Code That's Reused

| Component | File | Reuse |
|-----------|------|-------|
| `ComputeReceipt` | `src/receipt.rs` | Becomes per-inference log entry |
| `verify_receipt_chain` | `src/receipt.rs` | Validates log chain integrity |
| `compute_io_commitment` | `src/aggregation.rs` | Per-inference I/O binding |
| `compute_layer_chain_commitment` | `src/aggregation.rs` | Proves no layer skipped |
| `AggregatedModelProof` | `src/aggregation.rs` | Individual inference proof |
| `TeeAttestation` | `src/tee.rs` | Hardware trust binding |
| `prove_receipt_batch` | `src/receipt.rs` | Batch billing proof |
| `GpuModelProver` | `src/gpu/` | Forward pass replay |
| ElGamal encryption | `Obelysk-Protocol/contracts/src/elgamal.cairo` | Key wrapping |
| IndexedDB key store | `bitsage-marketplace/apps/web/src/lib/crypto/keyStore.ts` | Client-side keys |
| Privacy Pools ASP | `Obelysk-Protocol/contracts/src/privacy_pools.cairo` | Compliance proofs |

### 8. Open Questions

1. **Tokenization consistency**: How do we ensure the M31-quantized tokens during audit replay exactly match what was served? Need deterministic quantization or to store M31 matrices alongside the log.
2. **Log completeness**: How do we prove the server didn't omit entries? TEE attestation helps but isn't available on all hardware. Need a complementary mechanism (e.g., external log auditor).
3. **Recursive proof composition**: For 1000+ inference audits, independent proofs create large calldata. Recursive STARK verifier would collapse to single proof but is more complex to implement.
4. **Storage decided**: Arweave for permanent encrypted blobs (~$0.005/MB), Bitsage Marketplace as caching/UX layer, Starknet for compact on-chain anchor (~320 bytes). Total ~$0.03/audit.
5. **View key rotation**: How often should view keys rotate? Per-session? Per-day? On explicit revocation only?
