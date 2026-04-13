# ObelyZK On-Chain Verification Pipeline

**Last updated**: April 10, 2026
**Version**: 0.3.0
**Network**: Starknet Sepolia

**Latest verified TX (production recursive STARK)**: [`0x021512dd...`](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8) — SmolLM2-135M, 48-column chain AIR, 38 constraints, ~4,934 calldata felts, 160-bit security, 4 verified TXs on Sepolia

**Previous verified TX**: [`0x5ce1b4...edfd3`](https://sepolia.starkscan.co/tx/0x5ce1b41815e29a7b3dd03b77187cf32c8c5f0e2607960303174cbea303edfd3) — Qwen2.5-14B, 192 matmuls, 337 layers, 946 calldata felts

---

## 1. Overview

ObelyZK proves the correctness of ML inference over Circle STARKs (STWO). Given a
model (weights + architecture) and an input, the system executes the forward pass
over the M31 field, generates a GKR sumcheck proof for every layer, and produces
cryptographic calldata that a Starknet smart contract can verify on-chain.

There are two verification paths:

| Path | TXs | Calldata | Security | Status |
|------|-----|----------|----------|--------|
| **Recursive STARK v2** (preferred) | 1 | ~4,934 felts | 160-bit (pow_bits=20, log_blowup=5, n_queries=28) | **Verified on Sepolia** ([TX](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8)), 4 TXs verified |
| **Recursive STARK v1** | 1 | ~950 felts | Standard | **Verified on Sepolia**, 5+ verifications live |
| **Streaming GKR** (fallback) | 6+ | 8,744--255,100 felts | Standard | Fully deployed, 14/14 steps passing on-chain |

**Why recursive is preferred.** The streaming pipeline sends the raw GKR proof
data on-chain across multiple transactions. For large models (30+ layers) this
approaches or exceeds the Starknet sequencer's 10M Cairo step limit per TX.
The recursive STARK compresses the proof into a constant-size STARK that is
model-size-independent --- a 14B-parameter model and a 400B-parameter model
cost the same to verify.

### Fully Trustless Verification (OODS + Merkle + FRI + PoW)

The recursive STARK verifier on devnet performs full cryptographic verification:
OODS sampling, Merkle decommitment, FRI layer folding, and proof-of-work
validation all pass. This is the same verification that stwo-cairo-verifier
performs for any STARK proof -- applied to our recursive AIR.

Six key fixes made this work:

1. **Poseidon252MerkleChannel alignment** -- the recursive trace uses the same
   Merkle channel that stwo-cairo-verifier expects, so Fiat-Shamir transcripts
   match exactly between Rust prover and Cairo verifier.
2. **Felt252 9-limb decomposition** -- both Rust serialization and Cairo
   deserialization use identical 28-bit limb packing (9 limbs per felt252).
3. **CommitmentSchemeProof Serde** -- flat felt252 layout matches the
   `commitments -> sampled_values -> decommitments -> fri_proof` ordering.
4. **Preprocessed tree commitment** -- Tree 0 (selectors) committed before
   Tree 1 (execution), matching the AIR's `column_count_per_interaction`.
5. **Boundary constraint domain fix** -- `is_first` / `is_last` selectors
   evaluate on the correct coset points for the CircleSTARK domain.
6. **FRI proof nesting** -- inner FRI layers serialized with correct length
   prefixes for Cairo's `Serde::deserialize`.

### Compression Ratios

| Model | GKR felts | Recursive felts | AIR | Compression | Verified |
|-------|-----------|-----------------|-----|-------------|----------|
| SmolLM2 30-layer (v2) | 46,148 | ~4,934 | 48-col/38-constraint | 9.4x | **Sepolia** ([TX](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8)) |
| SmolLM2 1-layer (v1) | 8,744 | 718 | 28-col/27-constraint | 12.2x | Devnet |
| SmolLM2 30-layer (v1) | 46,148 | 942 | 28-col/27-constraint | 49x | **Sepolia** |
| Qwen3-14B 40-layer | ~112,000 | ~4,934 (v2) | 48-col/38-constraint | ~23x | Pending |

Note: The v2 AIR produces larger calldata (~4,934 felts vs ~950 felts) due to the
48-column trace with 38 constraints and two-level recursion, but provides significantly stronger
security (160-bit vs standard) and 9 independent security layers.

---

## 2. Architecture

### End-to-End Flow

```
Model Loading (ONNX / HuggingFace safetensors)
    |
    v
Quantization (f32 -> M31 via QuantStrategy)
    |
    v
Forward Pass (M31 matmul + activation over LayeredCircuit)
    |
    v
GKR Proof (per-layer sumcheck proofs + weight binding + IO commitment)
    |
    +---> [Streaming Path] Serialize to 6-step calldata -> multi-TX submission
    |
    +---> [Recursive Path] GKR Verifier as witness -> STARK prove() -> single TX
```

### Recursive Composition (Phase 4A)

The recursive path treats the GKR verifier itself as a computation to be proved:

```
prove_model()           ->  GKR Proof
generate_witness()      ->  GkrVerifierWitness (every Poseidon call recorded)
build_recursive_trace() ->  Execution trace (28 columns x 2^log_size rows)
stwo::prove()           ->  Recursive STARK proof (Poseidon252MerkleChannel)
verify_recursive()      ->  Rust pre-flight check
On-chain (1 TX)         ->  stwo-cairo-verifier verifies the STARK
```

The key insight: the GKR verifier's Fiat-Shamir transcript is a chain of Poseidon
permutations. The production recursive AIR (v2) constrains this chain with a 48-column
chain AIR (was 89 -- 41 unused columns removed) and 38 constraints, providing 160-bit
cryptographic security via a two-level recursion architecture.

**Chain AIR** (48 columns, 38 constraints):

- Boundary constraints binding initial and final digests
- Amortized accumulator constraint (unconditional -- blocks all-zeros-selector attack)
- Carry-chain modular addition for HadesPerm-level chain integrity
- seed_digest checkpoint binding chain to model dimensions
- pass1_final_digest binding proving full GKR verification ran
- hades_commitment binding for two-level recursion

**Two-level recursion:**
- Level 1: cairo-prove verifies 145 Hades permutations (10s, 278K felts, OFF-CHAIN)
- Level 2: Chain STARK binds to Level 1 commitment (6.5s, ~4,934 felts, ON-CHAIN)

**Hades AIR** (1225 columns):

- S-box constraints for Poseidon/Hades nonlinear layer
- MDS matrix multiplication constraints
- Round transition constraints

Cross-component integrity is enforced via LogUp chain-to-Hades binding.

**PcsConfig** (160-bit security):

- `pow_bits=20`: proof-of-work grinding difficulty
- `log_blowup=5`: FRI blowup factor (2^5 = 32x)
- `n_queries=28`: FRI query count
- `log_last_layer_deg=0`: final FRI layer is degree 1
- Security: pow_bits + log_blowup * n_queries = 20 + 5*28 = 160 bits

**9 Security Layers**:

1. Fiat-Shamir channel binding (all public inputs mixed before tree commits)
2. Amortized accumulator (unconditional constraint, blocks all-zeros-selector attack)
3. n_poseidon_perms on-chain validation (prevents trace miniaturization)
4. seed_digest checkpoint (binds chain to model dimensions)
5. pass1_final_digest binding (proves full GKR verification ran)
6. Carry-chain modular addition (HadesPerm-level chain integrity)
7. hades_commitment binding (two-level recursion)
8. Boundary constraints (initial/final digest)
9. 160-bit STARK security (pow=20, blowup=5, queries=28)

The previous v1 AIR used 28 execution columns, 3 preprocessed selectors, and
27 constraints. The v1 system remains operational but the v2 system is preferred
for new deployments.

The STARK uses `Poseidon252MerkleChannel` so that the proof is natively verifiable
by stwo-cairo-verifier (Cairo's native Poseidon for Fiat-Shamir and Merkle).

---

## 3. Contracts on Starknet Sepolia

### 3.1 Recursive Verifier v2 (Production, 48-Column Chain AIR)

| Field | Value |
|-------|-------|
| **Contract address** | [`0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) |
| **Latest verified TX** | [`0x021512dd...`](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8) |
| **Verification count** | 4 on Sepolia |
| **AIR** | 48 columns (chain) + 1225 columns (Hades), 38 constraints |
| **Security** | 160-bit (pow_bits=20, log_blowup=5, n_queries=28) |
| **Source** | `elo-cairo-verifier/src/recursive_verifier.cairo` + `recursive_air.cairo` (38 constraints matching Rust) |
| **Status** | **Live on Sepolia. Fully trustless STARK verification with 9 security layers. Two-level recursion.** |
| **Explorer** | [View on Starkscan](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) |

### 3.1.1 Recursive Verifier v1 (Original)

| Field | Value |
|-------|-------|
| **Contract address** | [`0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7`](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) |
| **Class hash** | [`0x056a8b05376d4133e14451884dcef650d469c137bed273dd1bba3f39e5df28a5`](https://sepolia.starkscan.co/class/0x056a8b05376d4133e14451884dcef650d469c137bed273dd1bba3f39e5df28a5) |
| **Source** | `elo-cairo-verifier/src/recursive_verifier.cairo` |
| **Deployer** | `0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344` |
| **Declare TX** | [`0x0684d0b2914a16a6637cfe2ba1b5da4f705f4156e2220e36b0e369ba7bab7a61`](https://sepolia.starkscan.co/tx/0x0684d0b2914a16a6637cfe2ba1b5da4f705f4156e2220e36b0e369ba7bab7a61) |
| **Deploy TX** | [`0x7b7715e4710b7f9e329bb91cffbdc05ac54b1e68b88989bee9fa60ec2dcdb9c`](https://sepolia.starkscan.co/tx/0x7b7715e4710b7f9e329bb91cffbdc05ac54b1e68b88989bee9fa60ec2dcdb9c) |
| **First verification** | [`0x61a60a7fcf899d38da5e0f4632746f48843e1c537dabe57ea7df42ad71c0ba6`](https://sepolia.starkscan.co/tx/0x61a60a7fcf899d38da5e0f4632746f48843e1c537dabe57ea7df42ad71c0ba6) |
| **MIN_POW_BITS** | 10 (production hardened) |
| **AIR** | 28 columns, 27 constraints |
| **Status** | **Live on Sepolia. Superseded by v2 for new deployments.** |
| **Explorer** | [View on Starkscan](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) |

This contract performs **full cryptographic STARK verification** on-chain:

- **OODS sampling** -- verifies polynomial evaluations at the out-of-domain point
- **Merkle decommitment** -- verifies Poseidon252 Merkle paths for all queried columns
- **FRI layer folding** -- verifies the full FRI proximity proof (14 layers)
- **PoW validation** -- requires proof-of-work nonce (MIN_POW_BITS=10)

**Entrypoints:**

| Function | Description |
|----------|-------------|
| `register_model_recursive(model_id, circuit_hash, weight_super_root)` | Register a model for verification |
| `verify_recursive(model_id, io_commitment, stark_proof_data)` | Submit and verify a recursive STARK proof |
| `is_recursive_proof_verified(proof_hash)` | Query whether a proof has been verified |
| `get_recursive_verification_count(model_id)` | Get total verifications for a model |
| `get_recursive_model_info(model_id)` | Get model registration details |
| `propose_upgrade(new_class_hash)` | Owner-only. Start upgrade with 5-minute timelock |
| `execute_upgrade()` | Owner-only. Finalize upgrade after timelock expires |
| `cancel_upgrade()` | Owner-only. Cancel a pending upgrade |

**Key deployment details:**

The Cairo contract was deployed by removing all `Felt252Dict` usage from the
stwo-cairo verifier (which generates `squashed_felt252_dict_entries`, a Sierra
1.8.0 libfunc not supported by stable Scarb). Replaced with array-based
`QueryPositionMap` for query position lookups, insertion sort for query
deduplication, and bucket arrays for column grouping.

### 3.2 Streaming GKR Verifier v32

| Field | Value |
|-------|-------|
| **Contract address** | `0x376fa0c4a9cf3d069e6a5b91bad6e131e7a800f9fced49bd72253a0b0983039` |
| **Class hash** | `0x5dca646786c36f9d68bab802d5c5c4995c37aa7c25bfa59ff20144a283f0956` |
| **Source** | `elo-cairo-verifier/src/verifier.cairo` |
| **Previous contract (v4)** | `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005` |

This is the production streaming verifier. It supports:

- Model registration with weight commitments and circuit descriptors
- Session-based streaming verification (open/upload/seal/stream)
- 14/14 streaming verification steps passing on-chain
- Aggregated weight binding with full MLE opening proofs
- Packed IO (8 M31 per felt252) for calldata efficiency
- KV-cache commitment chaining for decode-step proving
- Policy commitment binding

### 3.3 Deployer Account v2 (ACTIVE)

| Field | Value |
|-------|-------|
| **Address** | `0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f` |
| **Type** | OpenZeppelin v0.14.0 Account |
| **Status** | Active, compatible with Starknet v0.8 RPCs |

### 3.4 Previous Contract Versions

| Version | Address | Class Hash | Notes |
|---------|---------|------------|-------|
| GKR v4 | `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005` | -- | First streaming verifier |
| GKR v31 | -- | `0x6a6b7a75d5ec1f63d715617d352bc0d353042b2a033d98fa28ffbaf6c5b5439` | 14/14 steps pass, poseidon_hash_2 fix |
| GKR v30 | -- | `0x38e9f407..` | Diagnostic build |
| GKR v29 | -- | `0x316aa715..` | -- |
| Ephemeral | `0x52c2f627d6dfc1a663247f3696300ff5a66716a18b2762913b37f22c684e1f7` | -- | Ephemeral test contract |

---

## 4. Recursive STARK Pipeline (Main Path)

### Step 1: GKR Proof Generation

```bash
prove-model \
  --model-dir ./smollm2-135m \
  --gkr \
  --format ml_gkr \
  --recursive \
  --policy standard \
  --output proof.json
```

The `--recursive` flag triggers Phase 4 after GKR proving completes.

### Step 2: Recursive STARK Composition

The `prove_recursive()` function in `src/recursive/prover.rs` executes:

1. **Witness generation** (`generate_witness`): Re-runs the GKR verifier with an
   `InstrumentedChannel` that records every Poseidon permutation. The instrumented
   channel wraps `PoseidonChannel` and produces identical Fiat-Shamir output ---
   differential tests confirm transcript consistency.

2. **Trace building** (`build_recursive_trace`): Converts the witness ops into a
   2D execution trace. Each row represents one channel operation with its
   `digest_before` and `digest_after` decomposed into M31 limbs.

3. **Commitment** (Trees 0 and 1): Preprocessed selectors committed to Tree 0,
   execution trace committed to Tree 1, both using `Poseidon252MerkleChannel`.

4. **STARK proving** (`stwo::prove`): Standard STWO prove with the
   `RecursiveVerifierComponent` (a `FrameworkComponent<RecursiveVerifierEval>`).

Performance:

| Model | GKR Time | Recursive Time | Total |
|-------|----------|---------------|-------|
| SmolLM2 1-layer | 0.24s | 0.18s | 0.42s |
| SmolLM2 30-layer | ~2.5s | ~3.55s | ~6.0s |
| Qwen3-14B 40-layer | 103s | ~45s (projected) | ~148s |

### Step 3: Calldata Serialization

The `serialize_recursive_proof_calldata()` function in `src/cairo_serde.rs` produces:

```
+----------------------------------------------------------+
| circuit_hash: QM31              [4 felts]                |
| io_commitment: QM31             [4 felts]                |
| weight_super_root: QM31         [4 felts]                |
| n_layers: u32                   [1 felt]                 |
| verified: u32                   [1 felt]                 |
| final_digest: felt252           [1 felt]                 |
| log_size: u32                   [1 felt]                 |
+----------------------------------------------------------+
| StarkProof<Poseidon252MerkleHasher>                      |
|   (CommitmentSchemeProof layout)                         |
+----------------------------------------------------------+
```

Total: 16 header felts + STARK proof body. For log_size=10, typically 200--600 felts.

### Step 4: Single-TX Submission

```javascript
// deploy_recursive_v2.mjs
const tx = await account.execute({
  contractAddress: RECURSIVE_CONTRACT,
  entrypoint: "verify_recursive",
  calldata: CallData.compile({
    model_id: modelId,
    io_commitment: proof.io_commitment,
    stark_proof_data: proof.recursive_proof.calldata,
  }),
});
```

One transaction. No session management. No chunking.

---

## 5. Streaming Pipeline (Fallback Path)

The streaming pipeline splits the GKR proof into 6 verification steps, each
submitted as a separate Starknet transaction.

### 5.1 Verification Steps

| Step | Entrypoint | Description | Typical felts |
|------|-----------|-------------|---------------|
| 1. Init | `verify_gkr_stream_init` | Opens session, sends packed IO + metadata | ~200 |
| 2. Output MLE | `verify_gkr_stream_init_output_mle` | Chunks of output MLE data | ~500/chunk |
| 3. Layers | `verify_gkr_stream_layers` | Batched layer proofs (up to 30 layers/batch) | ~3,500/batch |
| 4. Weight Binding | `verify_gkr_stream_weight_binding` | Aggregated weight binding verification | ~1,000/chunk |
| 5. Input MLE | `verify_gkr_stream_finalize_input_mle` | Input MLE data + deferred proofs | ~500/chunk |
| 6. Finalize | `verify_gkr_stream_finalize` | Final check + proof recording | ~10 |

### 5.2 Session Management

The streaming protocol uses a session-based flow:

1. **`open_gkr_session`**: Creates a session with model_id, total_felts, circuit_depth,
   num_layers, and weight_binding_mode. Returns a `session_id` via event.

2. **`upload_gkr_chunk`**: Uploads proof data chunks to contract storage. Each chunk
   includes session_id, chunk_index, chunk_length, and the data.

3. **`seal_gkr_session`**: Seals the session to prevent further uploads. Triggers
   integrity checks on the uploaded data.

4. **Steps 1--6**: The six verification entrypoints above, each referencing the
   session_id. The `__SESSION_ID__` placeholder in pre-serialized calldata is
   replaced with the actual session_id returned from step 1.

### 5.3 Chunking Rules

- **Layer batches**: Maximum 30 layers per batch (`MAX_STREAM_BATCH_LAYERS`).
  Each layer costs ~34M gas. Starknet gas limit is 1.2B, so 30 layers uses ~1.0B.

- **Weight binding chunks**: Maximum 1,000 felts per chunk
  (`WEIGHT_BINDING_CHUNK_MAX_FELTS`). Non-final chunks call
  `verify_gkr_stream_weight_binding_chunk` (accumulate). The final chunk calls
  `verify_gkr_stream_weight_binding` (verify).

- **IO MLE chunks**: Output and input MLE data are chunked with offset/length
  metadata and an `is_last` flag.

### 5.4 TX Counts by Model Size

| Model | Layers | Init | Output MLE | Layer Batches | Weight Binding | Input MLE | Finalize | Total |
|-------|--------|------|-----------|---------------|----------------|-----------|----------|-------|
| SmolLM2 1-layer | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| SmolLM2 30-layer | 30 | 1 | 1 | 1 | 3--4 | 1 | 1 | 7--8 |
| Qwen3-14B 40-layer | 160+ | 1 | 2 | 6 | 5 | 2 | 1 | 17--18 |

### 5.5 Submission Scripts

- **`scripts/pipeline/streaming_submit.mjs`**: Orchestrates the 6-step flow with
  exponential backoff retry. Supports gasless mode via AVNU paymaster.

- **`scripts/pipeline/register_and_submit.mjs`**: Full pipeline including model
  registration, session creation, chunk upload, and streaming submission.

- **`scripts/submit_full.mjs`**: Standalone submission to the v32 contract with
  explicit gas bounds per step type.

---

## 6. Environment Variables

### Policy Presets (Recommended)

Use `--policy` instead of individual env vars. The policy is Poseidon-committed
into the Fiat-Shamir channel.

```bash
prove-model --gkr --policy standard   # default, for on-chain streaming
prove-model --gkr --policy strict     # all gates enforced
prove-model --gkr --policy relaxed    # development, fastest proving
```

### Individual Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_SKIP_RMS_SQ_PROOF` | unset | Skip RMSNorm Part 0 (variance) self-verification. Non-fatal: Cairo is authoritative. |
| `STWO_ALLOW_MISSING_NORM_PROOF` | `0` | Accept proofs with missing LayerNorm/RMSNorm sub-proofs. |
| `STWO_PIECEWISE_ACTIVATION` | `1` (true) | Use piecewise-linear algebraic activation proofs. Set to `0` to use legacy LogUp. |
| `STWO_ALLOW_LOGUP_ACTIVATION` | `0` | Allow reduced-precision LogUp activation proofs (lower 16--20 bits). |
| `STWO_AGGREGATED_FULL_BINDING` | `1` (true) | Enable full MLE opening proofs for weight binding. Required for streaming. |
| `STWO_SKIP_BATCH_TOKENS` | `0` | Skip batch token accumulation proofs. |
| `STWO_PURE_GKR_SKIP_UNIFIED_STARK` | `0` | Skip the unified STARK in pure-GKR mode (GKR handles all layers). |
| `STWO_SKIP_POLICY_COMMITMENT` | unset | Skip policy commitment mixing. Required when Cairo contract does not support policy. |
| `STWO_WEIGHT_BINDING` | `aggregated` | Weight binding strategy: `aggregated`, `individual`, `sequential`, `off`. |
| `STWO_AGGREGATED_RLC_ONLY` | `0` | Use RLC-only binding (weaker). Rejected by streaming pipeline soundness gates. |
| `STWO_GPU_MERKLE_THRESHOLD` | `4096` | Minimum tree size for GPU Merkle hashing. |
| `STWO_PROFILE` | unset | Enable per-phase profiling (JSON export to `<output>.profile.json`). |

### Standard Policy Env Var Equivalent

The `standard` policy (default for prove-server) is equivalent to:

```bash
export STWO_SKIP_RMS_SQ_PROOF=1
export STWO_ALLOW_MISSING_NORM_PROOF=1
export STWO_PIECEWISE_ACTIVATION=0
export STWO_ALLOW_LOGUP_ACTIVATION=1
export STWO_AGGREGATED_FULL_BINDING=1
export STWO_SKIP_BATCH_TOKENS=1
export STWO_PURE_GKR_SKIP_UNIFIED_STARK=1
export STWO_SKIP_POLICY_COMMITMENT=1
```

---

## 7. Key Bug Fixes

### 7.1 RMSNorm Gamma (April 2026)

**Problem**: GPU, decode, and SIMD prover paths called `reduce_rmsnorm_layer()`
which does not apply the learnable gamma scaling, instead of
`reduce_rmsnorm_layer_with_gamma()`. This produced incorrect GKR claims for any
model with RMSNorm layers (Qwen, LLaMA, Mistral, etc.).

**Fix**: All three paths updated to call `reduce_rmsnorm_layer_with_gamma()`.
The gamma scaling is applied after the reciprocal-square-root normalization,
matching the forward pass exactly.

**Affected**: `src/gkr/prover.rs` (GPU path), decode path, SIMD path.

### 7.2 Policy Commitment (March 2026)

**Problem**: `replay_verify_serialized_proof()` was missing the policy commitment
mix step. The Rust-side self-verifier mixed the policy hash into the Fiat-Shamir
channel, but the Cairo on-chain contract does not support policy yet. This caused
a transcript mismatch.

**Fix**: Set `STWO_SKIP_POLICY_COMMITMENT=1` to skip policy mixing in both prover
and self-verifier. The policy commitment is still recorded in proof metadata for
auditing but is not mixed into the cryptographic transcript.

**Workaround**: Always use `--policy standard` with `STWO_SKIP_POLICY_COMMITMENT=1`
until the Cairo contract is upgraded to support policy commitments.

### 7.3 Account v0.13.4 Compatibility (March 2026)

**Problem**: The original deployer account (v1) was deployed with an older
OpenZeppelin account class that is incompatible with Starknet Sepolia v0.8 RPC
endpoints. Calls to `starknet_estimateFee` and `starknet_addInvokeTransaction`
failed with `UNEXPECTED_ERROR` or `Contract not found`.

**Fix**: Deployed a new deployer account (v2) using OpenZeppelin v0.14.0 Account
contract, which supports the v3 transaction format required by v0.8 RPCs.

**Deployer v1 (DEPRECATED)**: `0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344`

### 7.4 CM31/QM31 CUDA Multiplication (February 2026)

**Problem**: Six CUDA kernel multiplication functions in `gpu_sumcheck.rs` had
wrong formulas. CM31 imaginary part was `ac + 2*bd` instead of `ad + bc`. QM31
cross term used `(2r+2s)` instead of `(2r-s)`.

**Fix**: Corrected all six variants (base, `_r`, `_l`) to match the field
arithmetic specification. Affected GPU-accelerated sumcheck for all layer types.

### 7.5 Activation Intermediate Storage (February 2026)

**Problem**: Forward pass stored reduced activation values as intermediates,
breaking GKR claim chaining. The GKR verifier expects the unreduced M31 output
from each layer as the input to the next.

**Fix**: Store `current.clone()` (full M31 values) before reduction. Only pass
the reduced values to the activation LogUp trace builder. Applied to all 5
forward pass locations + 2 verifier forward passes in `aggregation.rs`.

---

## 8. Fully Trustless Recursive Verification

The fully trustless recursive verifier is complete. OODS sampling, Merkle
decommitment, FRI layer folding, and proof-of-work all pass on devnet. The
remaining step is deploying the class to Sepolia.

### 8.1 What Works

- **RecursiveAir** (`elo-cairo-verifier/src/recursive_air.cairo`): Full Cairo
  implementation of the AIR with all 38 constraints. Implements the `Air` trait
  from `stwo_verifier_core`. Matches the Rust `RecursiveVerifierEval` exactly.

- **RecursiveVerifierContract** (`elo-cairo-verifier/src/recursive_verifier.cairo`):
  Starknet contract with `register_model_recursive`, `verify_recursive`, event
  emission, dedup via proof hash, and model info storage.

- **CommitmentSchemeProof Serde**: Full deserialization of `CommitmentSchemeProof<Poseidon252MerkleHasher>`
  from flat felt252 calldata. Handles nested structures: `commitments`,
  `sampled_values`, `decommitments`, `fri_proof`.

- **Felt252 limb decomposition**: Both Rust and Cairo use identical 9-limb
  (28 bits each) decomposition of felt252 values.

- **Poseidon252MerkleChannel**: The recursive STARK uses the same Merkle channel
  that stwo-cairo-verifier expects, eliminating the need to constrain Hades
  permutations inside the M31 AIR.

- **Full STARK verify()**: The `verify_recursive` entrypoint deserializes the
  proof, reconstructs the AIR, and calls stwo-cairo-verifier's `verify()` with
  the correct commitment scheme and security parameters.

### 8.2 Verified Models

| Model | Calldata | Prove Time | Contract | TX |
|-------|----------|------------|----------|-----|
| SmolLM2-135M (v2 AIR) | ~4,934 felts | 102s | v2 (`0x0121d1...`) | [`0x021512dd...`](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8) |
| SmolLM2-135M (30-layer) | 942 felts | 102s | v1 (`0x1c208a...`) | [`0x276c6a44...`](https://sepolia.starkscan.co/tx/0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc) |
| Qwen2-0.5B | 924 felts | 287s | v1 (`0x1c208a...`) | Verified on Sepolia |

### 8.3 Sepolia Deployment

Two recursive verifier contracts are deployed on Sepolia:

**v2 (production, preferred)**: Contract `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`.
Uses the 48-column chain AIR with 38 constraints and 160-bit security (pow_bits=20,
log_blowup=5, n_queries=28). Two-level recursion: Level 1 cairo-prove (145 Hades perms,
off-chain) + Level 2 chain STARK (on-chain). Includes 9 security layers (Fiat-Shamir
binding, amortized accumulator, n_poseidon_perms validation, seed_digest checkpoint,
pass1_final_digest binding, carry-chain modular addition, hades_commitment binding,
boundary constraints, 160-bit STARK security). 4 verified TXs on Sepolia. Contract
includes upgrade timelock.

**v1 (original)**: Contract `0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7`
with class `0x056a8b05376d4133e14451884dcef650d469c137bed273dd1bba3f39e5df28a5`.
Uses 28-column AIR with 27 constraints. Remains operational for backward compatibility.

A single Starknet transaction cryptographically verifies that the GKR verifier
accepted the original ML inference proof. The streaming pipeline is fully optional.

32 Cairo tests pass, including 8 upgrade timelock tests.

---

## 9. CLI and API Usage

### CLI: `--on-chain` Flag

The `obelysk` CLI supports proving and submitting in one command:

```bash
# Prove and submit on-chain (recursive by default)
obelysk prove --model smollm2-135m --input "Hello world" --on-chain

# Prove with explicit recursive flag
obelysk prove --model smollm2-135m --input "Hello world" --recursive --on-chain
```

The `--on-chain` flag triggers on-chain submission after proving completes. It
requires `STARKNET_PRIVATE_KEY` to be set. The `--recursive` flag is the default
proving mode.

Environment variables for on-chain submission:

```bash
export STARKNET_PRIVATE_KEY="<your-key>"
export STARKNET_RPC="<your-rpc-url>"              # defaults to Alchemy Sepolia
export RECURSIVE_CONTRACT="0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7"
```

### API: `/api/v1/attest` Endpoint

The prove-server exposes an attestation endpoint that proves and submits in one
request:

```bash
curl -X POST http://localhost:8080/api/v1/attest \
  -H "Content-Type: application/json" \
  -d '{"model": "smollm2-135m", "input": "Hello", "recursive": true}'
```

Response:

```json
{
  "output": "Hello world ...",
  "proof_hash": "0x03af8b...",
  "tx_hash": "0x07c1a2...",
  "prove_time_ms": 9500,
  "calldata_felts": 981,
  "submitted": true,
  "verified": true
}
```

### Script: `submit_recursive.mjs`

For manual submission of pre-generated proofs:

```bash
node scripts/submit_recursive.mjs /tmp/proof.json
```

The script reads `STARKNET_PRIVATE_KEY`, `STARKNET_RPC`, `RECURSIVE_CONTRACT`,
and `STARKNET_ACCOUNT` from environment variables. It registers the model if
needed, submits the recursive proof in a single transaction, and outputs a
structured JSON result.

---

## Appendix A: File Locations

| Component | Path |
|-----------|------|
| Recursive AIR (Rust) | `stwo-ml/src/recursive/air.rs` |
| Recursive prover | `stwo-ml/src/recursive/prover.rs` |
| Recursive verifier (Rust pre-flight) | `stwo-ml/src/recursive/verifier.rs` |
| Witness generator | `stwo-ml/src/recursive/witness.rs` |
| Types | `stwo-ml/src/recursive/types.rs` |
| Calldata serialization | `stwo-ml/src/cairo_serde.rs` (fn `serialize_recursive_proof_calldata`) |
| Streaming calldata builder | `stwo-ml/src/starknet.rs` (fn `build_streaming_gkr_calldata`) |
| Recursive AIR (Cairo) | `elo-cairo-verifier/src/recursive_air.cairo` |
| Recursive verifier (Cairo) | `elo-cairo-verifier/src/recursive_verifier.cairo` |
| Streaming verifier (Cairo) | `elo-cairo-verifier/src/verifier.cairo` |
| Policy configuration | `stwo-ml/src/policy.rs` |
| CLI binary | `stwo-ml/src/bin/prove_model.rs` |
| Recursive submission | `stwo-ml/scripts/submit_recursive.mjs` |
| Deploy scripts | `stwo-ml/scripts/deploy_recursive_v2.mjs`, `deploy_contract.mjs` |
| Submission scripts | `stwo-ml/scripts/pipeline/streaming_submit.mjs`, `register_and_submit.mjs` |

## Appendix B: Contract Upgrade Procedure

Both the streaming GKR verifier and the recursive verifier support upgrades via a timelock mechanism:

1. Declare the new class: `account.declare({ contract: sierra, casm })`
2. Propose upgrade: `account.execute({ entrypoint: "propose_upgrade", calldata: [new_class_hash] })`
3. Wait 5 minutes (timelock)
4. Execute upgrade: `account.execute({ entrypoint: "execute_upgrade" })`

Use `scripts/declare_and_upgrade.mjs` for the combined declare + propose flow.
