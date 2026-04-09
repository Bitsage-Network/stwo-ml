# Verifiable Inference Audit System — Scope Overview

**Status**: Planning
**Date**: February 14, 2026
**Version**: 0.1.0-draft

---

## Vision

Transform stwo-ml from a "can the model compute a forward pass?" demo into a **production audit system** that cryptographically proves real inference happened correctly — binding specific inputs, outputs, weights, and execution metadata into a single verifiable on-chain record.

The core insight: **proofs should verify actual inference, not random test vectors.** A proof of random noise has zero practical value. A proof that a medical AI used the FDA-approved model to diagnose a patient — that's what changes industries.

---

## Architecture

```
HOT PATH (real-time, zero overhead on serving)
┌──────────┐     ┌──────────────┐
│  User     │────>│ Model Server │──> Response to user
│  Query    │     │ (vLLM, etc.) │
└──────────┘     └──────┬───────┘
                        │ append (async, non-blocking)
                        v
                 ┌──────────────────┐
                 │  Inference Log    │  Append-only Poseidon-M31 Merkle log
                 │  - input tokens   │  Every entry: {input, output, timestamp,
                 │  - output tokens  │   model_hash, io_commitment, seq_number}
                 │  - timestamp      │
                 │  - model hash     │  Chain-linked via prev_entry_hash
                 │  - io_commitment  │
                 └──────────────────┘

AUDIT PATH (on-demand, async, GPU-accelerated)
                 ┌──────────────────┐
  "Audit last    │  Inference Log    │
   hour" ───────>│  [t_0 -> t_1]    │
                 └──────┬───────────┘
                        │
          ┌─────────────┼─────────────┐
          v             v             v
   ┌────────────┐ ┌──────────┐ ┌───────────┐
   │  Summary   │ │  Batch   │ │ Semantic  │
   │ Generator  │ │  Prover  │ │ Evaluator │
   │ (instant)  │ │ (GPU)    │ │ (parallel)│
   └─────┬──────┘ └────┬─────┘ └─────┬─────┘
         │              │             │
         v              v             v
   ┌─────────────────────────────────────────┐
   │         Audit Report (JSON)             │
   │  - Session summary (instant)            │
   │  - Semantic scores (fast)               │
   │  - ZK proof (async, on-chain)           │
   │  - All bound to same Merkle root        │
   └─────────────────────────┬───────────────┘
                             │
              ┌──────────────┼──────────────┐
              v              v              v
       ┌────────────┐ ┌──────────┐ ┌──────────────┐
       │  Encrypt   │ │ On-Chain │ │  Arweave      │
       │  (AES-GCM) │ │ Record   │ │  (permanent)  │
       │  + Key Wrap │ │ (~320B)  │ │  (~$0.005/MB) │
       └────────────┘ └──────────┘ └──────┬───────┘
                                          │ cached by
                                   ┌──────┴───────┐
                                   │  Bitsage      │
                                   │  Marketplace  │
                                   │  (UX layer)   │
                                   └──────────────┘
```

---

## Stages

| # | Stage | Document | What It Does | Readiness |
|---|-------|----------|-------------|-----------|
| 1 | Inference Logging | [`01_inference_logging.md`](./01_inference_logging.md) | Append-only Merkle log capturing every inference during serving | **Design ready** — extends existing `ComputeReceipt` + chain-linking |
| 2 | Audit Proving | [`02_audit_proving.md`](./02_audit_proving.md) | Batch replay + aggregated proof over a time window | **Core exists** — `prove_model` + `AggregatedModelProof` need batch wrapper |
| 3 | Semantic Evaluation | [`03_semantic_evaluation.md`](./03_semantic_evaluation.md) | Self-evaluation + deterministic checks for quality scoring | **New module** — forward pass infrastructure exists, evaluation prompts are new |
| 4 | Audit Report Format | [`04_audit_report_format.md`](./04_audit_report_format.md) | Comprehensive JSON report with all metadata | **New module** — schema design, serialization |
| 5 | On-Chain Audit Contract | [`05_onchain_audit_contract.md`](./05_onchain_audit_contract.md) | Cairo contract: `submit_audit` + `AuditRecord` storage | **Extension** — adds to existing `SumcheckVerifierContract` |
| 6 | Privacy & Encryption | [`06_privacy_encryption.md`](./06_privacy_encryption.md) | AES-256-GCM encryption, Arweave permanent storage, Marketplace UX cache | **Design ready** — Obelysk Protocol has ElGamal, Pedersen, Poseidon-M31 |
| 7 | Access Control | [`07_access_control.md`](./07_access_control.md) | On-chain ACL: grant/revoke per Starknet address | **New contract** — draws from stealth payments + view key delegation |

---

## Existing Infrastructure (What's Real)

### stwo-ml (Rust)

| Component | File | What It Does | Audit Integration |
|-----------|------|-------------|-------------------|
| `ComputeReceipt` | `src/receipt.rs` | GPU inference billing + chain-linking + TEE | **Direct reuse** — becomes per-inference log entry |
| `TeeAttestation` | `src/tee.rs` | NVIDIA CC attestation, `report_hash_felt()` | **Direct reuse** — binds audit to hardware |
| `compute_io_commitment` | `src/aggregation.rs` | Poseidon(input \|\| output) | **Direct reuse** — per-inference binding |
| `compute_layer_chain_commitment` | `src/aggregation.rs` | Running hash through all layers | **Direct reuse** — proves no layer skipped |
| `compute_layernorm_mean_var_commitment` | `src/aggregation.rs` | Poseidon(means, variances) | **Direct reuse** — binds normalization params |
| `compute_quantize_params_commitment` | `src/aggregation.rs` | Poseidon(scale, zp, bits) per layer | **Direct reuse** — proves quantization integrity |
| `AggregatedModelProof` | `src/aggregation.rs` | Unified STARK + per-matmul sumchecks | **Core proof** — wrapped into batch audit proof |
| `prove_receipt_batch` | `src/receipt.rs` | Batch STARK proof for billing constraints | **Direct reuse** — proves billing over audit window |
| `verify_receipt_chain` | `src/receipt.rs` | Hash-chain verification for receipts | **Direct reuse** — proves log continuity |
| `GpuModelProver` | `src/gpu/` | CUDA-accelerated forward pass + proving | **Direct reuse** — replays inference for proving |
| `TeeModelProver` | `src/tee.rs` | TEE-wrapped proving with attestation | **Direct reuse** — optional hardware trust layer |

### Smart Contracts (Cairo)

| Contract | Address (Sepolia) | What It Does | Audit Integration |
|----------|------------------|-------------|-------------------|
| `SumcheckVerifierContract` | `0x005928ac...` | `verify_model_gkr`, `verify_model_direct` | **Extend** — add `submit_audit` function |
| `ObelyskVerifier` | (in elo-cairo-verifier) | `verify_and_pay` with TEE binding | **Extend** — add `AuditRecord` storage |
| `AgentAccountFactory` | `0x2f69e5...` | Deploy agent accounts | **Reuse** — audit submitter identity |
| `IdentityRegistry` | `0x72eb37...` | ERC-8004 identity NFT | **Reuse** — bind audits to identity |

### Obelysk Protocol (Privacy Primitives)

| Primitive | File | What It Does | Audit Integration |
|-----------|------|-------------|-------------------|
| ElGamal encryption | `contracts/src/elgamal.cairo` | Homomorphic encryption over STARK curve | **Key wrapping** — encrypt report keys per-recipient |
| Pedersen commitments | `contracts/src/pedersen_commitments.cairo` | `C = v*G + r*H` | **Report binding** — commit to report without revealing |
| Poseidon-M31 | (VM31 spec) | 12.2x cheaper than Poseidon-252 | **Future** — native M31 hashing for inference log |
| View key delegation | `contracts/src/stealth_payments.cairo` | Viewing without spending authority | **Audit access** — grant read-only access without ownership transfer |
| Privacy Pools ASP | `contracts/src/privacy_pools.cairo` | Association set compliance | **Compliance** — prove audit submitter is in approved set |

---

## Readiness Matrix

```
Stage 1 (Inference Logging)     [████████░░] 80% — Receipt + chain-link exist, need log format + hook
Stage 2 (Audit Proving)         [██████░░░░] 60% — prove_model + AggregatedModelProof exist, need batch + replay
Stage 3 (Semantic Evaluation)   [██░░░░░░░░] 20% — Forward pass infra exists, evaluation logic is new
Stage 4 (Audit Report Format)   [█░░░░░░░░░] 10% — Pure design work, no existing code
Stage 5 (On-Chain Contract)     [████░░░░░░] 40% — Verifier contract exists, need AuditRecord extension
Stage 6 (Privacy/Encryption)    [███░░░░░░░] 30% — Obelysk primitives exist, integration is new
Stage 7 (Access Control)        [██░░░░░░░░] 20% — View keys exist, ACL contract is new
```

---

## Implementation Order

### Phase 1: Foundation (Stages 1-2)
Build the inference log and batch prover. This is the minimum viable product — real inference captured and provable.

### Phase 2: Intelligence (Stages 3-4)
Add semantic evaluation and structured reporting. The audit goes from "computation correct" to "computation correct AND meaningful."

### Phase 3: On-Chain (Stage 5)
Deploy the `submit_audit` contract extension. Audits become permanently verifiable on Starknet.

### Phase 4: Privacy (Stages 6-7)
Encrypt reports, add access control. Full production privacy — public verifiability with private content.

---

## What the Proof Guarantees

For each inference in the audit window:

| Guarantee | How | Existing? |
|-----------|-----|-----------|
| **Model identity** | `weight_commitment` = Poseidon of all weight matrices | Yes (`aggregation.rs`) |
| **Input integrity** | `io_commitment` = Poseidon(input \|\| output) | Yes (`aggregation.rs`) |
| **Output integrity** | Same `io_commitment` binds output | Yes (`aggregation.rs`) |
| **All layers ran** | `layer_chain_commitment` = running hash through every layer | Yes (`aggregation.rs`) |
| **No layer skipped** | Chain breaks if any intermediate is substituted | Yes (`aggregation.rs`) |
| **Correct arithmetic** | STARK proof of every matmul, activation, norm, etc. | Yes (core prover) |
| **Quantization integrity** | `quantize_params_commitment` binds scale/zp/bits | Yes (`aggregation.rs`) |
| **Hardware attestation** | TEE report hash bound to proof (optional) | Yes (`tee.rs`) |
| **Billing accuracy** | Receipt STARK proves billing arithmetic | Yes (`receipt.rs`) |
| **Log continuity** | `prev_receipt_hash` chain-links entries | Yes (`receipt.rs`) |
| **Time window** | Audit covers [t_start, t_end] with all entries | New (Stage 1) |
| **Batch coverage** | Merkle root of log covers all N inferences | New (Stage 1) |
| **Semantic quality** | Self-evaluation scores (also provable) | New (Stage 3) |

## What the Proof Does NOT Guarantee

- **Semantic correctness** — If the model hallucinates, the proof proves it hallucinated genuinely with those weights. The proof is about computational integrity, not truth.
- **Model quality** — The proof doesn't say the model is "good." That comes from benchmarks, evals, red-teaming. The proof says the model you evaluated IS the model that ran.
- **Completeness of log** — If the server operator omits entries from the log, the proof covers what's in the log. TEE attestation mitigates this by binding computation to hardware-verified execution.

---

## Dependencies

```
Stage 1 ──> Stage 2 ──> Stage 5
                │
                └──> Stage 4
                        │
Stage 3 ───────────────>│
                        │
                        └──> Stage 6 ──> Stage 7
```

- Stage 2 depends on Stage 1 (need log to prove)
- Stage 4 depends on Stage 2 + Stage 3 (report contains proof + evaluation)
- Stage 5 depends on Stage 2 + Stage 4 (contract stores proof + report hash)
- Stage 6 depends on Stage 4 (encrypt the report)
- Stage 7 depends on Stage 6 (control access to encrypted report)
- Stage 3 can run in parallel with Stage 2 (independent forward pass)
