# ObelyZK ZKML Proving System -- Technical Status Report

**Date**: April 6, 2026
**Version**: 0.2.0
**Author**: Internal Engineering
**Audience**: Engineering team, investors, auditors
**Branch**: `development`

---

## 1. Executive Summary

ObelyZK is a zero-knowledge ML inference proving system built on STWO Circle STARKs. It can take a HuggingFace model (ONNX or safetensors), execute the forward pass over the M31 finite field, generate a GKR sumcheck proof for every layer, compress the proof via recursive STARK composition, and verify the result on Starknet in a single transaction. The system has been demonstrated end-to-end on Starknet Sepolia with SmolLM2-135M (30 layers, 211 GKR layers, 120 weight matrices). Seven model families across five architectures have been proven locally. The core GKR prover and recursive STARK are real, tested, and functional. However, critical gaps exist in weight binding, soundness gate enforcement, and production infrastructure that must be closed before the system can be considered production-ready.

---

## 2. What's Production-Ready

### 2.1 GKR Prover

The GKR sumcheck prover is the core of the system. It is real, not scaffolding.

| Metric | Value |
|--------|-------|
| Rust source files | ~128 |
| Feature flags | 29 (`std`, `gpu`, `cuda-runtime`, `multi-gpu`, `tee`, `onnx`, `safetensors`, `model-loading`, `audit`, `proof-stream`, etc.) |
| Test count | 935+, 0 failures |
| Field arithmetic | M31 -> CM31 -> QM31 (128-bit algebraic security) |
| Integer-only | Zero f64 in the proving path; cos/sin tables, integer sigmoid/gelu/silu |

**Supported operations**: MatMul, RMSNorm, LayerNorm, SiLU, GELU, Softmax, Sigmoid, attention (multi-head, GQA), residual connections, RoPE, KV-cache commitment chaining.

**Proven models** (CPU self-verification, Apple Silicon):

| Model | Parameters | Prove Time |
|-------|-----------|------------|
| Qwen2-0.5B | 0.5B | 0.57s |
| Qwen2-1.5B | 1.5B | 1.14s |
| SmolLM2-135M | 135M | 3.41s |
| Phi-3 Mini 3.8B | 3.8B | 48.86s |
| Llama-3.2-3B | 3B | 48.48s |
| Yi-1.5-6B | 6B | 86.58s |
| Mistral-7B-v0.3 | 7B | 88.19s |

**GPU performance** (A10G):

| Model | Layers | GKR Proving | Total |
|-------|--------|-------------|-------|
| SmolLM2-135M | 30 | ~2.5s | ~6.0s (with recursive) |
| Qwen3-14B | 40 | 103s | ~148s (projected with recursive) |

**What's real in the prover**:
- MatMul sumcheck: Real sumcheck over MLE with STWO's `prove_batch`/`partially_verify`. Fiat-Shamir channel absorbs round polynomials before drawing challenges.
- Activation STARK: Real LogUp proof via `FrameworkComponent` (`finalize_logup_in_pairs`).
- Proof aggregation: Multiple activation STARKs composed into single STARK proof.
- Forward pass execution: Real M31 arithmetic through matmul + activation layers.
- Weight binding: Aggregated oracle sumcheck with full MLE opening proofs. Poseidon Merkle roots computed for all weight matrices.
- IO commitment: Poseidon hash of packed input/output felts (8 M31 per felt252).
- GPU CUDA kernels: Sumcheck MLE restriction on GPU, CM31/QM31 field arithmetic (corrected February 2026).
- CPU SIMD fallback: Full prover path without GPU.

### 2.2 Recursive STARK

The recursive STARK compresses the GKR proof into a constant-size STARK proof, enabling single-transaction verification regardless of model size. Two versions are deployed:

#### v2 (Upgraded, April 12, 2026)

| Metric | Value |
|--------|-------|
| GKR felts (SmolLM2-135M 30-layer) | 46,148 |
| Recursive felts | ~4,934 |
| Security | 160-bit (pow_bits=20, log_blowup=5, n_queries=28) |
| Chain AIR | 48 columns, 38 constraints |
| Hades AIR | 1225 columns |
| Security layers | 9 independent layers |
| Merkle channel | Poseidon252MerkleChannel |
| Latest verified TX | [`0x021512dd...`](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8) |
| Verification count | 4 on Sepolia |
| Contract | `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005` |

**How it works**: The GKR verifier is re-executed with an `InstrumentedChannel` that records every Poseidon permutation. The recorded operations become the witness for a STARK AIR. The v2 AIR uses 48 columns (was 89 -- 41 unused columns removed) and 38 constraints including an amortized accumulator (unconditional constraint blocking all-zeros-selector attacks), carry-chain modular addition for HadesPerm-level chain integrity, boundary constraints, seed_digest checkpoints, pass1_final_digest binding, and hades_commitment binding for two-level recursion. A separate Hades AIR (1225 columns) handles S-box, MDS, and round transition constraints. Two-level recursion: Level 1 cairo-prove verifies 145 Hades permutations (10s, 278K felts, off-chain), Level 2 chain STARK binds to Level 1 commitment (6.5s, ~4,934 felts, on-chain).

The 9 security layers are: (1) Fiat-Shamir channel binding, (2) amortized accumulator, (3) n_poseidon_perms on-chain validation, (4) seed_digest checkpoint, (5) pass1_final_digest binding, (6) carry-chain modular addition, (7) hades_commitment binding, (8) boundary constraints, (9) 160-bit STARK security (pow=20, blowup=5, queries=28).

#### v1 (Original)

| Metric | Value |
|--------|-------|
| Recursive felts | 942 |
| Compression ratio | 49x |
| Recursive proving time (A10G) | 3.55s |
| AIR constraints | 27 |
| Trace columns | 28 execution + 3 preprocessed |
| Trace rows | 16,384 (log_size = 14) |
| Poseidon permutations | 14,126 |

**How it works**: The 27 constraints are: 9 boundary (first row digest = zero), 9 boundary (last row digest = final), 9 chain (row i output = row i+1 input). All degree 2. The STARK uses `Poseidon252MerkleChannel` so that stwo-cairo-verifier can verify it natively.

### 2.3 On-Chain Trustless Verification

Recursive STARK proofs have been verified on Starknet Sepolia in single transactions with full cryptographic verification.

#### v2 Contract (Upgraded)

| Field | Value |
|-------|-------|
| Contract | `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005` |
| First verified proof TX | [`0x055c2bf89f43d9b65580862e0b81e6b47842b9dda3b862c134f35b61b0ae620f`](https://sepolia.starkscan.co/tx/0x055c2bf89f43d9b65580862e0b81e6b47842b9dda3b862c134f35b61b0ae620f) |
| AIR | 48-column chain (38 constraints) + 1225-column Hades |
| Security | 160-bit (pow_bits=20, log_blowup=5, n_queries=28) |
| On-chain TXs required | 1 |

#### v1 Contract (Original)

| Field | Value |
|-------|-------|
| Contract | `0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7` |
| Class hash | `0x056a8b05376d4133e14451884dcef650d469c137bed273dd1bba3f39e5df28a5` |
| First verified proof TX | `0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc` |
| MIN_POW_BITS | 10 (production) |
| On-chain TXs required | 1 |

**What both contracts check**:
- OODS (Out-of-Domain Sampling): Verifies polynomial evaluations at a random out-of-domain point.
- Merkle decommitment: Verifies Poseidon252 Merkle paths for all queried trace columns.
- FRI proximity proof: Verifies FRI folding layers.
- Proof-of-work: Requires valid PoW nonce.

This is the same verification that stwo-cairo-verifier performs for any STARK proof. The GKR semantics are encoded entirely in the AIR public inputs (`circuit_hash`, `io_commitment`, `weight_super_root`), not in the contract logic.

### 2.4 Streaming GKR Verification (Fallback)

A second verification path exists for smaller models or when recursive proving is not desired.

| Field | Value |
|-------|-------|
| Contract | `0x376fa0c4a9cf3d069e6a5b91bad6e131e7a800f9fced49bd72253a0b0983039` |
| Verification steps | 14/14 passing on-chain |
| Model | 1-layer Qwen2 |
| TX count | 6+ (session-based: init, output MLE, layers, weight binding, input MLE, finalize) |

### 2.5 Infrastructure

| Component | Status |
|-----------|--------|
| A10G GPU instance | Running at `44.251.24.184` |
| Juno Sepolia node | Configured with snap-sync for class declaration |
| Hosted API | **Live** at `https://api.bitsage.network` (HTTPS, Bearer auth, 60 req/min rate limit) |
| Automated deployment | `deploy_node.sh`, `deploy_api.sh` (nginx + certbot), `deploy_recursive_v2.mjs` |
| Policy system | `PolicyConfig` with strict/standard/relaxed presets, Poseidon commitment hashing |
| Phase profiler | Per-phase wall-clock + Poseidon hash counts (`STWO_PROFILE=1`) |
| Proof streaming | WebSocket broadcast via `prove-server` with Three.js dashboard |

### 2.6 Additional Components

- **ZKML Transaction Classifier**: MLP-based classifier (64->64->32->3) for verifiable AI guardrails. GKR + STARK proof of classifier inference.
- **KV-Cache Commitment Chain**: Poseidon binding for incremental decode-step proving. `AttentionDecode` variant with Merkle append proofs.
- **VM31 Privacy Protocol**: Deposit/withdraw/spend circuits + STARKs + batch proving (PAUSED).
- **Audit Pipeline**: M31-native Poseidon2 hashing, inference logging, semantic evaluation.

---

## 3. Known Gaps and Weaknesses

### 3.1 Self-Verification Bug (Streaming Path)

**Severity**: Medium (recursive path unaffected)

The GKR proof's streaming format self-verification fails at layer 1 (`MATMUL_FINAL_MISMATCH`) and layer 210 (RMSNorm linear round). Root cause: Fiat-Shamir transcript divergence between `prove_model_pure_gkr` and `replay_verify_serialized_proof`. The prover and replay verifier produce different Poseidon channel states at these layers.

The recursive STARK path works because `prove_recursive_with_policy` uses a different verification code path (the `InstrumentedChannel` witness generator) that does not hit this divergence.

**Impact**: Streaming GKR submission for 30-layer models would fail if the streaming path were used. The recursive path (which is the primary path) is unaffected. This is a latent bug that would surface if anyone tried to use the streaming fallback for large models.

### 3.2 Weight Binding Placeholders

**Severity**: Critical

Model registration on the trustless recursive contract uses placeholder values:
```
circuit_hash: 0x1
weight_super_root: 0x1
```

The actual Poseidon Merkle roots are computed during proving (`compute_io_commitment_packed()`, weight commitment in `apply_aggregated_oracle_sumcheck`), but they are not passed through to `register_model_recursive()` in the submission scripts.

**Impact**: The contract verifies that the STARK proof is mathematically valid (OODS, Merkle, FRI, PoW all pass), but it does not bind the proof to specific model weights or architecture. An attacker could:
1. Register model_id X with `circuit_hash: 0x1, weight_super_root: 0x1`.
2. Prove a completely different (weaker/malicious) model.
3. Submit the recursive STARK proof.
4. The contract would accept it because the STARK is valid and the registered circuit_hash is a placeholder that matches anything.

This is the single most important gap to close.

### 3.3 Policy Commitment Skipped

**Severity**: Medium

`STWO_SKIP_POLICY_COMMITMENT=1` is set in all deployment scripts and is part of the `standard` policy preset. The `PolicyConfig` system exists in Rust with Poseidon commitment hashing and three presets (strict/standard/relaxed), but the commitment is not mixed into the Fiat-Shamir channel during proving when this flag is set.

**Impact**: Proofs do not carry enforceable policy guarantees. A prover using the `relaxed` policy (which skips several soundness checks) produces a proof indistinguishable from one using the `strict` policy. The on-chain contract has no way to know which policy was used.

### 3.4 Weakened Soundness Gates

**Severity**: High

Five of six environment variable gates are set to permissive mode in the `standard` policy (the default for all on-chain submissions):

| Gate | Setting | What It Skips |
|------|---------|---------------|
| `STWO_SKIP_RMS_SQ_PROOF=1` | Permissive | Skips RMSNorm Part 0 (variance) sub-proof generation and self-verification |
| `STWO_ALLOW_MISSING_NORM_PROOF=1` | Permissive | Accepts proofs without LayerNorm/RMSNorm LogUp verification |
| `STWO_PIECEWISE_ACTIVATION=0` | Disabled | Disables piecewise algebraic activation proofs (falls back to legacy LogUp, which is also skipped) |
| `STWO_ALLOW_LOGUP_ACTIVATION=1` | Permissive | Accepts proofs with missing LogUp activation sub-proofs |
| `STWO_SKIP_BATCH_TOKENS=1` | Permissive | Skips batch token accumulation proofs |

These are set to permissive because the on-chain Cairo verifier (both recursive and streaming) does not check these sub-proof components. The recursive STARK only wraps the GKR sumcheck verification and Fiat-Shamir chain -- it does not include activation LogUp, norm sub-proofs, or batch token proofs in its AIR.

**Impact**: A malicious prover can substitute arbitrary activation function outputs, normalization parameters, or batch aggregation without detection. The GKR sumcheck for MatMul layers is fully verified, but the gaps in activation/norm verification mean that only the linear algebra is provably correct. A sophisticated attacker could manipulate non-linear operations while preserving valid MatMul proofs.

### 3.5 SDK/API Mismatch

**Severity**: Medium

The published packages have mismatched contents:

- **`@obelyzk/sdk` (npm)**: Contains BitSage distributed compute client code, not ObelyZK prover-specific APIs. The `createProverClient()` class references endpoints that may not be live.
- **`obelyzk` (PyPI)**: Same issue -- contains BitSage client wrappers, not prover APIs.
- **`@obelyzk/cli`**: Commands (`obelysk prove`) do not match the actual binary entry points (`bitsage deploy`, `bitsage stake`).

**Impact**: Developers who install the published SDK cannot actually prove or verify models without building from source. The "3 lines of code" promise in documentation is aspirational.

### 3.6 Hosted API

**Severity**: Resolved

The hosted API is live at `https://api.bitsage.network` and `https://prover.bitsage.network` with HTTPS (nginx + Let's Encrypt). Authentication is via Bearer token. Rate limit: 60 requests/minute. Deployment is automated via `scripts/deploy_api.sh` which handles build, systemd service, nginx reverse proxy, and certbot certificate provisioning.

### 3.7 No Adversarial Testing

**Severity**: High

No malicious proof submissions have been tested against the trustless recursive contract. Specifically:

- No bit-flip testing in any proof section.
- No malformed calldata submissions.
- No oversized proof testing (what happens if calldata exceeds expectations).
- No proof replay attacks (submitting the same proof for different inputs).
- No fuzzing of the Cairo verifier deserialization.
- No testing of wrong `model_id` with valid proof.
- No testing of `io_commitment` manipulation.

The Rust-side prover has tamper tests (9/9 adversarial attacks detected in the GKR layer), but the on-chain contract has not been subjected to adversarial inputs.

### 3.8 Cairo Test Coverage for Recursive Verifier

**Severity**: Medium (improved)

The trustless recursive verifier contract (`elo-cairo-verifier/src/recursive_verifier.cairo`) now has 32 passing Cairo tests (up from 0), including 8 upgrade timelock tests. Coverage includes model registration, proof verification, duplicate detection, and upgrade lifecycle (propose/execute/cancel). Remaining gaps:
- No tampered proof rejection tests (bit-flip in STARK data).
- No fuzzing of deserialization edge cases.
- No adversarial calldata submissions.

### 3.9 Single Model Verified On-Chain

**Severity**: Low

Only SmolLM2-135M has been end-to-end verified on Starknet Sepolia via the recursive path. Seven model families have been proven locally with CPU self-verification, but only one has produced an on-chain verified proof. Qwen2-0.5B, Phi-3-mini, Qwen3-14B, Llama-3.2-3B, Yi-1.5-6B, and Mistral-7B-v0.3 have not been submitted on-chain.

### 3.10 Crates.io Not Published

**Severity**: Low

`stwo-ml` cannot be published to crates.io because it depends on the STWO library via a path dependency (local fork). Users must clone the repo and build from source.

### 3.11 Key Management

**Severity**: Medium

The deployer account v2 private key (`0x0123456789abcdef...`) is stored in plaintext in `docs/DEPLOYER_ACCOUNTS.md` and referenced in deployment scripts. While this is a testnet key, the documentation explicitly warns against using it on mainnet -- but the rotation procedure is not documented and there is no multisig or timelock on the deployer account.

### 3.12 Contract Upgrade Timelock

**Severity**: Low (resolved)

Both the streaming GKR verifier and the recursive verifier contract now have a timelock upgrade mechanism (`propose_upgrade` -> 5-minute delay -> `execute_upgrade` / `cancel_upgrade`). The recursive contract was upgraded to class `0x056a8b05376d4133e14451884dcef650d469c137bed273dd1bba3f39e5df28a5` which includes full upgrade support. 32 Cairo tests pass, including 8 upgrade-specific tests.

---

## 4. Security Assessment

| Area | Rating | Notes |
|------|--------|-------|
| STARK verification soundness | **STRONG** | Real OODS + Merkle decommitment + FRI proximity proof + PoW. Uses stwo-cairo-verifier, which is a well-tested STARK verifier. |
| GKR sumcheck soundness | **STRONG** | Correct Fiat-Shamir channel (challenge absorbs round polys). MLE opening proofs via Merkle decommitment. 9/9 adversarial tamper tests pass in Rust. |
| Recursive AIR correctness | **MODERATE** | 27 constraints verified by differential testing (witness generator vs. production verifier). But no formal verification, no independent audit, and constraint count is low enough that subtle bugs could exist. |
| Weight binding | **WEAK** | Poseidon Merkle roots are computed during proving but on-chain registration uses placeholder values (0x1). The contract cannot distinguish proofs from different models. |
| Model identity binding | **WEAK** | `circuit_hash` is 0x1 on-chain. No binding between proof and model architecture. |
| IO commitment | **MODERATE** | Correctly computed as Poseidon hash of packed IO felts. Mixed into the Fiat-Shamir channel. But not independently verifiable by a third party without access to the raw inputs/outputs. |
| Activation/norm verification | **WEAK** | All soundness gates permissive in standard policy. Activation functions, normalization layers, and batch tokens are not verified in the recursive AIR. Only MatMul sumcheck is fully verified end-to-end. |
| Policy enforcement | **NOT ACTIVE** | Policy commitment skipped in all deployment configurations. |
| Access control | **BASIC** | Owner-only model registration. No multisig, no role-based access, no governance. |
| Key management | **WEAK** | Testnet keys in plaintext in docs and scripts. Single deployer account. No rotation procedure. |
| Denial of service | **BASIC** | Hosted API has rate limiting (60 req/min) and Bearer token auth. Contract-level DoS not tested. |
| Contract upgradeability | **COMPLETE** | Both streaming and recursive contracts have timelock upgrade mechanism (propose/execute/cancel, 5-minute delay). |
| Supply chain | **MODERATE** | Depends on STWO fork (path dep), stwo-cairo-verifier, starknet.js. No vendoring or pinning of Cairo dependencies. |

---

## 5. What We Can Honestly Claim

| Claim | Verdict | Explanation |
|-------|---------|-------------|
| "We can prove ML inference and verify the STARK proof on-chain in 1 TX" | **TRUE** | Demonstrated on Sepolia with SmolLM2-135M. TX `0x276c6a44...` is verifiable on-chain. Full STARK verification (OODS + Merkle + FRI + PoW). |
| "The verification is fully trustless (OODS + Merkle + FRI + PoW)" | **TRUE for the STARK layer** | The STARK proof itself is verified with full cryptographic rigor. However, the STARK only proves that the GKR verifier's Fiat-Shamir chain was executed correctly -- it does not separately verify activations, norms, or weight binding due to the weakened soundness gates. |
| "The proof binds to specific model weights" | **FALSE** | On-chain registration uses placeholder `weight_super_root: 0x1`. The Rust prover computes real weight roots, but they are not threaded to the contract. |
| "The proof binds to specific model architecture" | **FALSE** | On-chain registration uses placeholder `circuit_hash: 0x1`. |
| "Any model can be verified" | **PARTIALLY TRUE** | Seven model families have been proven locally. Only SmolLM2-135M has been verified on-chain. The system should work for other models of similar or smaller size, but this has not been demonstrated. |
| "The SDKs let you prove and verify in 3 lines of code" | **ASPIRATIONAL** | The published npm/PyPI packages contain BitSage client code, not ObelyZK prover APIs. Building from source and running the Rust binary works, but the SDK developer experience does not match documentation. |
| "All operations in the forward pass are verified" | **FALSE** | MatMul layers are verified via GKR sumcheck. Activation functions, normalization layers, and batch tokens are not verified when using the standard policy (which is the default for all on-chain submissions). |
| "Recursive proof is constant-size regardless of model" | **TRUE in theory** | The recursive STARK calldata is determined by the STARK parameters, not the model size. Demonstrated at 942 felts for 30-layer SmolLM2. Projected ~950 for 40-layer Qwen3-14B. Not yet demonstrated for models > 30 layers on-chain. |
| "103s proving time for Qwen3-14B on H100" | **TRUE** | Benchmarked on H100 NVL. 40 layers, 160 MatMuls, cached weights. |

---

## 6. Codebase Statistics

| Metric | Value |
|--------|-------|
| Total Rust source files | ~128 |
| Lines of Rust (estimated) | ~45,000 |
| Cairo contract files | ~15 (streaming + recursive) |
| JavaScript/TypeScript scripts | ~20 (deployment, submission, testing) |
| Feature flags | 29 |
| Test count | 935+ |
| Test pass rate | 100% |
| CI | GitHub Actions (Rust tests, clippy, format) |
| Toolchain | nightly-2025-07-14 |
| STWO dependency | Local fork (path dependency) |

---

## 7. Contract Registry

| Contract | Address | Purpose | Status |
|----------|---------|---------|--------|
| Recursive Verifier v2 | `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005` | Production: 48-col/38-cst AIR, 160-bit security, 9 security layers, two-level recursion | Live on Sepolia |
| Recursive Verifier v1 | `0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7` | Original: 28-col/27-cst AIR, trustless STARK verification (1 TX) | Live on Sepolia |
| Streaming Verifier v32 | `0x376fa0c4a9cf3d069e6a5b91bad6e131e7a800f9fced49bd72253a0b0983039` | Multi-TX GKR verification | Live on Sepolia |
| Deployer v2 | `0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f` | Account for contract operations | Active |
| Deployer v1 | `0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344` | Legacy deployer | DEPRECATED |

---

## 8. Audit History

An internal security audit (February 2026) identified 24 findings across four severity tiers:

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 8 | All fixed |
| High | 5 | All fixed |
| Medium | 6 | All fixed |
| Quantize | 6 | All fixed |

Key critical findings that were fixed:
- C1: Activation proofs used identity (no real STARK) -- now real FrameworkComponent proofs.
- C2: MatMul sumcheck used constant challenge (soundness collapse) -- now absorbs round polys.
- C3: No domain separation in unified STARK -- now writes domain separators per component.
- C4: MLE opening proof missing from sumcheck -- now includes Merkle decommitment.
- C5: LayerNorm mean/variance not committed -- now Poseidon-hashed and bound.

No external third-party audit has been conducted.

---

*This document was prepared for internal use. It reflects the system state as of April 6, 2026. All claims have been cross-referenced against the codebase, deployment scripts, and on-chain transaction records.*

## Appendix: Design Issue Found During E2E Testing (April 8, 2026)

### weight_super_root is Input-Dependent

**Finding**: The `weight_super_root` stored in the recursive proof changes with each inference input, even for the same model. This is because it's computed as `Poseidon(num_claims, claim[0].expected_value, ...)` where `expected_value` is the weight MLE evaluated at a Fiat-Shamir challenge point — which depends on the input through the transcript.

**Impact**: Model registration on-chain must be updated for each unique input's proof, defeating the purpose of one-time registration.

**Root cause**: The weight binding uses evaluated claims (input-dependent) instead of committed roots (input-independent).

**Fix**: Replace `weight_super_root` with `Poseidon(weight_merkle_roots)` — a hash of the individual weight Poseidon Merkle roots, which are model-fixed and input-independent. The Merkle roots are already computed during GKR proving (in `apply_aggregated_oracle_sumcheck`).

**Workaround**: Register with `weight_super_root=0` (skip check) until the fix is implemented. The STARK proof still cryptographically verifies the computation — the binding is weaker but the proof is still valid.
