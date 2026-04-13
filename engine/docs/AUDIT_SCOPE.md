# ObelyZK Audit Scope

**Version**: 1.0
**Date**: April 6, 2026
**Engagement type**: Third-party security audit
**Contact**: dev@obelysk.xyz

---

## 1. Scope Overview

This document defines the scope of a security audit for the ObelyZK ZKML proving
system. The audit covers the cryptographic core: GKR prover/verifier, recursive
STARK composition, Cairo on-chain verifier, serialization layer, policy system, and
supporting cryptographic primitives. The goal is to verify the soundness of the
proof system, correctness of the Rust-to-Cairo serialization, and security of the
on-chain verification contracts.

The system takes a neural network model (weights + circuit topology) and an input,
executes inference in the M31 field, generates a GKR proof, composes it into a
STARK, and verifies it on Starknet via a Cairo smart contract.

---

## 2. In-Scope Components

### 2.1 GKR Prover and Verifier

**Path**: `src/gkr/`
**LOC**: ~26,500

| File | Lines | Description |
|------|-------|-------------|
| `prover.rs` | 13,840 | GKR prover: sumcheck, MLE evaluation, weight binding, GPU dispatch |
| `verifier.rs` | 10,218 | GKR verifier: sumcheck verification, claim reduction, streaming verification |
| `types.rs` | 875 | Proof types, layer descriptors, serialization |
| `profiler.rs` | 817 | Phase profiling (non-security-critical) |
| `circuit.rs` | 672 | LayeredCircuit topology, circuit hash computation |
| `mod.rs` | 72 | Module re-exports |

**Key security properties**:
- Sumcheck soundness for every layer type (MatMul, RMSNorm, LayerNorm, Activation,
  Embedding, Conv2D, Attention, Dequantize, Quantize)
- Correct MLE evaluation at verifier-chosen challenge points
- Aggregated weight binding: RLC computation + MLE opening proofs
- Fiat-Shamir channel state consistency between prover and verifier
- Forward pass execution correctness in M31 arithmetic

### 2.2 Recursive STARK

**Path**: `src/recursive/`
**LOC**: ~2,328

| File | Lines | Description |
|------|-------|-------------|
| `witness.rs` | 636 | Witness generation for recursive AIR |
| `air.rs` | 498 | AIR constraint definitions for recursive composition |
| `tests.rs` | 439 | Recursive verifier tests |
| `prover.rs` | 376 | Recursive STARK prover: wraps GKR proof in STARK |
| `types.rs` | 190 | Recursive proof types |
| `verifier.rs` | 145 | Recursive STARK verifier |
| `mod.rs` | 44 | Module re-exports |

**Key security properties**:
- Inner GKR proof correctly encoded as AIR witness
- Recursive AIR constraints faithfully reproduce GKR verifier logic
- STARK soundness (delegates to STWO's `prove()`/`verify()`)
- Proof compression preserves soundness (~4,934 felts for v2 AIR, 981 felts for v1)
- 160-bit security via PcsConfig (pow_bits=20, log_blowup=5, n_queries=28)
- 48-column chain AIR with 38 constraints (v2), including amortized accumulator
  (41 unused columns removed from previous 89-column design)
- Hades AIR with 1225 columns for Poseidon permutation verification
- Two-level recursion: Level 1 cairo-prove (145 Hades perms, off-chain) + Level 2
  chain STARK (on-chain)
- 9 security layers: Fiat-Shamir binding, amortized accumulator, n_poseidon_perms
  validation, seed_digest checkpoint, pass1_final_digest binding, carry-chain
  modular addition, hades_commitment binding, boundary constraints, 160-bit STARK
  security

### 2.3 Cairo Serialization

**Path**: `src/cairo_serde.rs`
**LOC**: ~6,310

**Key security properties**:
- Lossless encoding of M31, CM31, QM31 values as felt252
- Correct packed IO format (8 M31 per felt252)
- Deterministic serialization (same proof always produces same calldata)
- No truncation, overflow, or off-by-one in field element packing
- Alignment between Rust serializer and Cairo deserializer

### 2.4 Policy System

**Path**: `src/policy.rs`
**LOC**: ~820

**Key security properties**:
- Policy commitment correctly computed via Poseidon hash with domain separator
- `PolicyConfig::strict()` preset closes all soundness gates
- Environment variable parsing cannot be exploited to produce unexpected configs
- Policy commitment mixed into Fiat-Shamir channel at correct position
- Backward compatibility: zero commitment skips mix (legacy proofs)

### 2.5 Cryptographic Primitives

**Path**: `src/crypto/`
**LOC**: ~11,100

| File | Lines | Description |
|------|-------|-------------|
| `aggregated_opening.rs` | 2,929 | Batched MLE opening proofs for weight binding |
| `mle_opening.rs` | 2,295 | Single MLE opening proof |
| `poseidon_channel.rs` | 1,709 | Poseidon2-based Fiat-Shamir channel |
| `poseidon_merkle.rs` | 1,042 | Poseidon Merkle tree (CPU + GPU) |
| `encryption.rs` | 854 | Symmetric encryption (lower priority) |
| `poseidon2_m31.rs` | 740 | M31-native Poseidon2 hash |
| `commitment.rs` | 494 | Weight commitment schemes |
| `merkle_cache.rs` | 491 | Merkle tree caching layer |
| `merkle_m31.rs` | 456 | M31 Merkle tree |
| `hades.rs` | 75 | Hades permutation constants |
| `mod.rs` | 15 | Module re-exports |

**Key security properties**:
- Poseidon2 M31 implementation matches reference specification
- Merkle tree construction is collision-resistant
- Aggregated opening proofs are sound (batched verification is equivalent to
  individual verification)
- Channel absorb/squeeze operations are deterministic and order-dependent

### 2.6 Cairo Verifier Contract

**Path**: `../elo-cairo-verifier/src/recursive_verifier.cairo`
**LOC**: ~379

**Key security properties**:
- Correct deserialization of proof calldata
- Fiat-Shamir channel state matches Rust prover exactly
- OODS, FRI, Merkle verification correctly implemented
- Proof-of-work nonce verified
- Proof hash deduplication prevents replay
- Model registry access control (owner-only registration)
- IO commitment cross-checked against proof data

### 2.7 Cairo Recursive AIR

**Path**: `../elo-cairo-verifier/src/recursive_air.cairo`
**LOC**: ~132

**Key security properties**:
- AIR constraints match the Rust `src/recursive/air.rs` definitions exactly
  (38 constraints in v2, 27 constraints in v1)
- Constraint degree bounds correctly specified
- No missing constraints that would allow invalid witnesses
- Amortized accumulator constraint is unconditional (blocks all-zeros-selector attack)
- Carry-chain modular addition constraints enforce HadesPerm-level integrity

### 2.8 STWO Cairo Verifier Modifications

**Path**: `../stwo-cairo/stwo_cairo_verifier/` (6 files modified)

The ObelyZK project modifies the upstream STWO Cairo verifier to remove all
`Felt252Dict` usage, which is necessary for Starknet deployment (dictionaries
exceed gas limits). These modifications must preserve STARK verification soundness.

**Key security properties**:
- Dict-free implementations are functionally equivalent to original dict-based code
- No verification steps were accidentally removed during dict elimination
- FRI layer verification remains sound without dict-based deduplication

### 2.9 Aggregation and Starknet Integration

**Path**: `src/aggregation.rs` (~12,729 LOC), `src/starknet.rs` (~10,457 LOC)

| File | Lines | Description |
|------|-------|-------------|
| `aggregation.rs` | 12,729 | Proof aggregation, IO commitment, activation STARK |
| `starknet.rs` | 10,457 | Calldata building, self-verification, on-chain submission |

**Key security properties**:
- `compute_io_commitment_packed()` produces deterministic commitments
- Self-verification (Rust replay) matches on-chain verification
- Calldata construction is lossless and deterministic
- Streaming proof chunking does not lose or reorder data
- Packed/double-packed formats preserve all proof elements

---

## 3. Out-of-Scope

The following components are explicitly excluded from this audit engagement:

| Component | Path | Reason |
|-----------|------|--------|
| SDK clients (TypeScript, Python) | External repos | Application layer, not cryptographic |
| CLI binary | `src/bin/obelysk.rs` | User interface wrapper, delegates to in-scope code |
| prove-server HTTP layer | `src/bin/prove_server.rs` | Network layer, not cryptographic core |
| Privacy module | `src/privacy/` | Paused development, separate protocol (VM31) |
| Audit module | `src/audit/` | Logging/reporting infrastructure |
| TUI/dashboard | `src/tui/` | Visualization, no security impact |
| Docker/deployment | `Dockerfile`, `scripts/` | Infrastructure, not cryptographic |
| ZKML Classifier | `src/classifier/` | Separate feature, not core proving |
| Economics module | `src/economics.rs` | Pricing logic, not cryptographic |
| GPU kernel code | `src/gpu.rs`, `src/gpu_sumcheck.rs`, `src/metal/` | Same cryptography as CPU path; GPU correctness is tested via equivalence tests |
| Component STARKs | `src/components/` | Legacy proving path, superseded by GKR |
| Gadgets | `src/gadgets/` | Lookup table infrastructure |

---

## 4. Feature Flags

The crate uses Cargo feature flags to control compilation. The following flags are
relevant to the audit:

| Flag | Required | Description |
|------|----------|-------------|
| `std` | Yes | Standard library support. Required for all builds. |
| `gpu` | No | GPU acceleration via CUDA. Uses the same cryptographic operations as CPU; the GPU path is an optimization, not a different protocol. |
| `cuda-runtime` | No | CUDA runtime linking. Required when `gpu` is enabled on systems with NVIDIA GPUs. |
| `multi-gpu` | No | Multi-GPU support for parallel weight commitment. Same cryptography, partitioned across devices. |
| `cli` | No | Command-line interface binary (`obelysk`). |
| `model-loading` | No | HuggingFace/ONNX/SafeTensors model loading. Affects input parsing, not proving. |
| `onnx` | No | ONNX model format support. |
| `safetensors` | No | SafeTensors weight format support. |
| `tee` | No | Trusted Execution Environment attestation. |
| `audit` | No | Audit logging and reporting. |
| `audit-http` | No | HTTP transport for audit submissions. |
| `proof-stream` | No | Real-time proof streaming via channels. |
| `proof-stream-rerun` | No | Rerun visualization integration. |
| `proof-stream-ws` | No | WebSocket proof streaming. |
| `server-stream` | No | prove-server WebSocket streaming endpoint. |
| `serde` | No | Serde serialization derives. |

**Audit recommendation**: Focus on the default feature set (`std`) plus
`model-loading` (which exercises the full proving pipeline). GPU features use
identical cryptography and are covered by equivalence tests.

---

## 5. Deployment Targets

### 5.1 Starknet Sepolia (Testnet)

| Item | Value |
|------|-------|
| **Recursive verifier v2 contract** | `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005` |
| **Recursive verifier v1 contract** | `0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7` |
| **Recursive verifier v1 class hash** | `0x056a8b05376d4133e14451884dcef650d469c137bed273dd1bba3f39e5df28a5` |
| **Streaming GKR class (v31)** | `0x6a6b7a75d5ec1f63d715617d352bc0d353042b2a033d98fa28ffbaf6c5b5439` |
| **Deployer account** | `0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344` |
| **Network** | Starknet Sepolia (chain ID: SN_SEPOLIA) |
| **Status** | Active, 14/14 streaming verification steps passing |

### 5.2 Starknet Mainnet

| Item | Value |
|------|-------|
| **Status** | Pending deployment |
| **Blocker** | Audit completion, soundness gate closure |

---

## 6. Test Infrastructure

### 6.1 Rust Tests

- **Total**: 1,287 `#[test]` functions across 102 source files
- **GKR prover tests**: 46 tests covering sumcheck, MLE evaluation, weight binding,
  forward pass, and self-verification
- **GKR verifier tests**: 53 tests covering claim reduction, streaming verification,
  and adversarial tampering
- **Recursive STARK tests**: 13 tests covering proof generation, verification, and
  compression
- **Cairo serde tests**: 46 tests covering felt252 encoding/decoding, packed IO,
  and round-trip consistency
- **Crypto tests**: 106 tests across all primitives (Poseidon, Merkle, MLE opening,
  aggregated opening, encryption)
- **Aggregation tests**: 72 tests covering IO commitment, activation STARK, and
  proof composition
- **Starknet integration tests**: 63 tests covering calldata construction,
  self-verification, and on-chain submission

### 6.2 Cairo Tests

- **Total**: 149 test functions across 9 test files
- **Recursive verifier tests**: 20 tests including adversarial tests (tampered
  proofs, wrong model ID, wrong IO commitment, replay attempts)
- **Cross-language tests**: 9 tests (`test_sp3_cross_language.cairo`) verifying
  Rust-Cairo serialization consistency
- **Field arithmetic tests**: 28 tests for M31/CM31/QM31 operations
- **Layer verifier tests**: 25 tests for per-layer GKR verification
- **Firewall tests**: 30 tests for access control and input validation
- **Channel tests**: 10 tests for Poseidon channel state consistency
- **Model verifier tests**: 12 tests for end-to-end model verification
- **Contract integration tests**: 10 tests for full contract interaction

### 6.3 Cross-Language Test Vectors

Rust and Cairo share test vectors for critical operations:
- Poseidon2 hash outputs
- M31/QM31 field arithmetic
- MLE evaluation at specific challenge points
- Packed IO commitment computation
- Fiat-Shamir channel state after specific mixing sequences

These vectors ensure the Rust prover and Cairo verifier produce identical
cryptographic outputs.

### 6.4 End-to-End On-Chain Verification

Three models have been verified end-to-end on Starknet Sepolia:
- SmolLM2-135M (v2 AIR, 48-col/38-constraint, 160-bit security): recursive STARK,
  single TX ([`0x055c2bf8...`](https://sepolia.starkscan.co/tx/0x055c2bf89f43d9b65580862e0b81e6b47842b9dda3b862c134f35b61b0ae620f))
- SmolLM2-135M (30 layers, 211 GKR layers): recursive STARK v1, single TX
- 5-layer MLP: streaming GKR, 14 sequential transactions

---

## 7. Build Reproducibility

### 7.1 Rust

| Item | Value |
|------|-------|
| **Toolchain** | `nightly-2025-07-14` (pinned in `rust-toolchain.toml`) |
| **Dependency lock** | `Cargo.lock` committed (8,389 lines) |
| **STWO dependency** | Pinned git revision in `Cargo.toml` |
| **Target** | `x86_64-unknown-linux-gnu` (production), `aarch64-apple-darwin` (dev) |

### 7.2 Cairo

| Item | Value |
|------|-------|
| **Scarb** | 2.12.2 |
| **snforge** | 0.54.1 (pinned via git tag in `Scarb.toml`) |
| **Package version** | `elo_cairo_verifier` v0.4.0 |
| **STWO Cairo dependency** | Local path `../stwo-cairo/stwo_cairo_verifier/crates/` |

### 7.3 Reproducible Build Steps

```bash
# Rust: build and test
rustup install nightly-2025-07-14
cargo +nightly-2025-07-14 build --features std,model-loading
cargo +nightly-2025-07-14 test --features std,model-loading --lib

# Cairo: build and test
cd ../elo-cairo-verifier
scarb build
snforge test
```

### 7.4 Docker

A `Dockerfile` is available for reproducible builds in an isolated environment.
The Docker image pins the Rust toolchain, Scarb version, and all dependencies.

---

## 8. Key Security Properties to Verify

### 8.1 STARK Soundness

- **OODS (Out-of-Domain Sampling)**: Verify that the OODS point is derived from
  the Fiat-Shamir channel and that polynomial evaluations at the OODS point are
  correctly checked against committed traces.
- **Merkle commitment**: Verify that Merkle proofs are validated against committed
  roots and that leaf indices are correctly computed from FRI query positions.
- **FRI (Fast Reed-Solomon IOP)**: Verify that FRI folding, query, and
  decommitment steps correctly reduce polynomial degree. Check that the number of
  FRI layers and the final polynomial degree bound are consistent.
- **Proof-of-Work**: Verify that the PoW nonce satisfies the required difficulty
  (`pow_bits >= 10`) and that the PoW challenge is derived from the channel state
  after all commitments.

### 8.2 GKR Completeness

Verify that every layer type is correctly reduced via sumcheck:

| Layer Type | Reduction | Critical Check |
|------------|-----------|----------------|
| MatMul | Product of two MLEs | Degree-2 sumcheck (3 evaluation points) |
| RMSNorm | Variance + reciprocal sqrt | Part 0 (variance sum) + Part 1 (rsqrt table) |
| LayerNorm | Mean + variance + normalize | Multi-part reduction with masking |
| Activation (GELU, SiLU, etc.) | Piecewise-linear segments | Segment binding proof |
| Embedding | Table lookup | Index-to-vector MLE |
| Conv2D | im2col + MatMul lowering | Correct lowering to MatMul reduction |
| Attention | QKV + softmax + output | Multi-head attention with RoPE |
| Dequantize | Scale + zero-point | Fixed-point to M31 conversion |
| Quantize | Clamp + round | M31 to fixed-point conversion |

### 8.3 Fiat-Shamir Binding

- **Channel initialization**: Verify that the channel is seeded with all public
  parameters (circuit hash, weight super root, IO commitment, policy commitment)
  in the correct order.
- **Challenge derivation**: Verify that all verifier challenges (sumcheck, FRI,
  OODS) are derived by squeezing the channel after absorbing the corresponding
  prover messages.
- **Rust-Cairo consistency**: Verify that `PoseidonChannel` in Rust
  (`src/crypto/poseidon_channel.rs`) and the Cairo channel implementation produce
  identical states given identical inputs.

### 8.4 Public Input Binding

- **Circuit hash**: Verify that the `circuit_hash` mixed into the channel matches
  the one stored in the on-chain model registry for the given `model_id`.
- **Weight super root**: Verify that the `weight_super_root` mixed into the channel
  is the Poseidon Merkle root over all weight matrices, and that aggregated opening
  proofs correctly verify weight MLE evaluations against this root.
- **IO commitment**: Verify that the packed IO commitment correctly encodes the
  input and output tensors and is mixed into the channel.
- **Policy commitment**: Verify that the policy commitment correctly encodes the
  `PolicyConfig` fields and is mixed into the channel at the correct position.

### 8.5 Serialization Safety

- **Rust to felt252**: Verify that `cairo_serde.rs` correctly encodes all proof
  elements (M31, CM31, QM31, Merkle paths, FRI layers, sumcheck rounds) as felt252
  values without truncation or overflow.
- **felt252 to Cairo**: Verify that the Cairo deserializer reconstructs the exact
  same proof elements from the felt252 array.
- **Round-trip consistency**: Verify that encode(decode(x)) == x and
  decode(encode(x)) == x for all proof element types.
- **Packed IO format**: Verify that the 8-M31-per-felt252 packing is lossless and
  that the Cairo unpacking produces the same M31 values.
- **Edge cases**: Verify behavior for zero values, maximum field element values
  (p-1 = 2^31 - 2), and boundary conditions in array serialization.

---

## 9. Known Issues (for Auditor Context)

### 9.1 Open Soundness Gates

Four environment-variable-controlled soundness gates can weaken verification.
See `docs/SOUNDNESS_GATES_AUDIT.md` for a detailed analysis including test
results and closure status for each gate:

| Gate | Default | Status |
|------|---------|--------|
| `STWO_ALLOW_MISSING_NORM_PROOF` | `false` | Cannot close yet; prover does not generate full norm binding proofs |
| `STWO_PIECEWISE_ACTIVATION` | `true` | Default is secure; legacy `false` path has upper-bit gap |
| `STWO_ALLOW_LOGUP_ACTIVATION` | `false` | Coupled with piecewise; both must be addressed together |
| `STWO_SKIP_BATCH_TOKENS` | `false` | Not yet tested; low priority (single-token unaffected) |

The `PolicyConfig` system binds these gates into the Fiat-Shamir channel, so a
proof generated under a relaxed policy cannot pass verification under a strict
policy. However, if the on-chain verifier does not enforce policy checking (legacy
contracts), this binding is ineffective.

### 9.2 IO Commitment Encoding Mismatch

The IO commitment parameter passed to the on-chain verifier uses packed felt252
encoding (8 M31 per felt), while the proof header embeds IO values in a different
format (raw M31 arrays with layer metadata). The on-chain verifier does not
currently cross-check the packed IO commitment against the IO values in the proof
header. This means a prover could theoretically pass a different IO commitment
than what the proof actually computes over, though Fiat-Shamir binding provides
indirect protection since the commitment is mixed into the channel.

### 9.3 No Contract Upgrade Timelock

The recursive verifier contract supports upgrades via a propose-then-execute
pattern. The streaming GKR verifier has a 5-minute timelock between proposal and
execution. The recursive verifier may not have an equivalent timelock, which
could allow the contract owner to upgrade to a malicious verifier without
community notice.

### 9.4 CM31/QM31 CUDA Kernel Fix (Historical)

In February 2026, six CUDA kernel multiplication functions were found to have
incorrect formulas for CM31 and QM31 arithmetic. These were fixed:
- CM31: imaginary part was `ac + 2*bd`, corrected to `ac - bd`
- QM31: cross term was `(2r+2s)`, corrected to `(2r-s)`

The CPU and SIMD paths were unaffected. GPU equivalence tests now verify that
GPU and CPU produce identical results. Auditors should verify that the current
CUDA kernels in `src/gpu_sumcheck.rs` match the M31/CM31/QM31 arithmetic
specifications.

### 9.5 Activation Intermediate Storage

The forward pass stores unreduced M31 values as activation intermediates for GKR
claim chaining. A historical bug stored reduced values instead, breaking GKR
claim continuity. This was fixed in February 2026 across all 5 forward pass
locations and 2 verifier forward passes. Auditors should verify that intermediate
storage is consistent between prover and verifier.

---

## 10. Contact

| Role | Contact |
|------|---------|
| **Engineering lead** | dev@obelysk.xyz |
| **Security questions** | dev@obelysk.xyz |
| **Repository access** | Provided via GitHub invitation upon engagement |
| **Test environment** | Starknet Sepolia (public) |
| **Prover instance** | Available upon request for end-to-end testing |
