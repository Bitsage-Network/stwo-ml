# elo-cairo-verifier

```
+===========================================================================+
|                                                                           |
|    ON-CHAIN ZKML VERIFIER FOR STARKNET                                   |
|                                                                           |
|    Fully trustless ML inference verification.                            |
|    Single transaction. Full STARK verification (OODS+Merkle+FRI+PoW).   |
|                                                                           |
+===========================================================================+
```

Verifies STWO ML proofs on Starknet — recursive STARK verification in a single TX, plus streaming GKR verification for fallback. Pure cryptographic verification, no trust assumptions.

## Verified On-Chain (Starknet Sepolia)

### Recursive STARK Verification — 1 TX, Fully Trustless (April 2026)

Single-transaction verification of a 30-layer SmolLM2-135M transformer proof. The Cairo contract performs full OODS + Merkle + FRI + PoW STARK verification on-chain.

| Field | Value |
|-------|-------|
| **Contract** | [`0x526fcdb940f92dc50bc3a234ffafe6d08d7b2e3b69f6cb41678331ee6a5a03c`](https://sepolia.starkscan.co/contract/0x526fcdb940f92dc50bc3a234ffafe6d08d7b2e3b69f6cb41678331ee6a5a03c) |
| **Class hash** | `0x0300ff964fe615d094af601074b76b7193b564e0c7215c7b98bc046334c35bcf` |
| **Verified proof TX** | [`0x276c6a44...`](https://sepolia.starkscan.co/tx/0x61a60a7fcf899d38da5e0f4632746f48843e1c537dabe57ea7df42ad71c0ba6) |
| **Calldata** | 942 felts (49x compression from 46,148 GKR felts) |
| **Model** | SmolLM2-135M, 30 layers, 211 GKR layers |
| **Prove time** | 102s (A10G GPU) + 3.55s recursive |

**Key**: This is NOT record-based. The Cairo contract re-executes the STARK verifier, checking every polynomial evaluation, every Merkle path, every FRI fold, and the proof-of-work nonce.

### Full GKR Streaming Verification — 6/6 TX SUCCEEDED (March 2026)

First-ever full GKR streaming proof verification of ML inference on Starknet. Model: Qwen2-0.5B, 1 transformer layer (8 GKR layers: 3 RMSNorm + 4 MatMul + 1 SiLU). Streaming v25 protocol, 5,526 felts calldata.

| Step | Entrypoint | TX Hash | Status |
|------|-----------|---------|--------|
| 1 | `verify_gkr_stream_init` | [`0x5493...1a1`](https://sepolia.starkscan.co/tx/0x5493310a8e2deb5d2f25b07e2402e84692aaf5926141b5acc203a1892a181a1) | SUCCEEDED |
| 2 | `verify_gkr_stream_output_mle` | [`0x7cab...7e`](https://sepolia.starkscan.co/tx/0x7cabd35f5382c11334c6509e40b7a758ccd7e03e83e75b66a3c569f5d7b7a7e) | SUCCEEDED |
| 3 | `verify_gkr_stream_layers` | [`0x5346...918`](https://sepolia.starkscan.co/tx/0x53465edc957c5f8a6054739a0633beecf814ee37e3e22c23a570448a5be5918) | SUCCEEDED |
| 4 | `verify_gkr_stream_weight_binding` | [`0x5f54...3fc`](https://sepolia.starkscan.co/tx/0x5f549a1e6cc1ebefea3615c2458cdd0fd8f45fd505bf72e5b9dd8417c9be3fc) | SUCCEEDED |
| 5 | `verify_gkr_stream_input_mle` | [`0x2395...bab`](https://sepolia.starkscan.co/tx/0x239545b66f94387a3d1b5dbc55dedba6b7de1d5384f1930e77e152a986d5bab) | SUCCEEDED |
| 6 | `verify_gkr_stream_finalize` | [`0x4b08...a41`](https://sepolia.starkscan.co/tx/0x4b081156d4be88ea159533223d2597d76cd3f99911501d8326e156f12051a41) | SUCCEEDED |

**Contract**: [`0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005)
**Class hash (v39)**: `0x0473c81da9df0522f5c239f022889f7730ef866fb97e4f092ad1e8793fb22feb`

**Technical stack**: STWO Circle STARKs, GKR sumcheck over M31, Poseidon252 Fiat-Shamir channel, aggregated weight binding with Poseidon Merkle trees, LogUp STARK proofs for activation/normalization layers.

### Policy-Bound Verification

Proofs are generated under a **policy** (strict, standard, or relaxed) that controls soundness gates. The policy is cryptographically bound to the proof via a Poseidon commitment mixed into the Fiat-Shamir channel.

**On-chain enforcement**:
- `register_model_policy(model_id, policy_hash)` locks a model to a specific policy
- `verify_gkr_stream_init()` validates the submitted `policy_hash` matches the registered policy
- `verify_gkr_stream_finalize()` includes `policy_hash` in the proof hash
- `ModelGkrVerified` event emits the `policy_hash` for off-chain indexers

When a model has a registered policy:
- Proofs generated under a different policy are rejected at `stream_init`
- Decode chain validation is automatically enforced for KV-cache sessions
- The `policy_hash` appears in the `ModelGkrVerified` event alongside the proof hash

Models without a registered policy (legacy) continue to work with `policy_hash = 0`.

### Earlier Verification Milestones

| Model | Architecture | Layers Verified | Tx Status |
|-------|-------------|-----------------|-----------|
| **D8** | Single MatMul | MatMul | Accepted on L2 |
| **D9** | MLP (MatMul + ReLU + MatMul) | MatMul, Activation, MatMul | Accepted on L2 |
| **D10** | LayerNorm Chain | MatMul, LayerNorm, MatMul | Accepted on L2 |
| **D11** | Residual Network (DAG with skip connection) | MatMul, ReLU, MatMul, Add + Deferred | Accepted on L2 |

> **Full documentation**: [`../docs/onchain-zkml-verification.md`](../docs/onchain-zkml-verification.md)

### AgentFirewallZK — DEPLOYED (April 2026)

On-chain guardrails for AI agent transactions. Deployed at [`0x043b51f6f571137d0e7c3afa4ca689e84271ba97c5b6fc83349a3fe1275634f0`](https://sepolia.starkscan.co/contract/0x043b51f6f571137d0e7c3afa4ca689e84271ba97c5b6fc83349a3fe1275634f0).

The contract scores transactions using a ZKML-proven classifier and enforces decisions on-chain with 21 sequential security checks.

```cairo
// 1. Register agent
firewall.register_agent(agent_id);

// 2. Submit action for evaluation
let action_id = firewall.submit_action(agent_id, target, value, selector, io_commitment);

// 3. After ZKML proof is verified on ObelyskVerifier, resolve with proven IO data
firewall.resolve_action_with_proof(action_id, proof_hash, original_io_len, packed_raw_io);
// → threat_score computed ON-CHAIN from proven output neurons
// → EMA trust score updated with asymmetric decay
// → strikes incremented if score >= escalation threshold

// 4. External contracts check before executing
if firewall.is_action_approved(action_id) {
    // safe to execute the agent's transaction
}
```

**Also deployed:** [`ContractRegistry`](https://sepolia.starkscan.co/contract/0x075f9812753666ee506509de0de10bdea3ad1a79d4ed31817a0e2534c9d90607) for `is_verified`/`has_source` target attestations.

**Security**: 21 sequential checks in `resolve_action_with_proof`, asymmetric EMA (fast up 0.5, slow down 0.1), auto-freeze at 5 strikes, emergency pause, 2-step ownership transfer, per-agent rate limiting, proof replay protection, on-chain score extraction (not caller-supplied), 47/48 input features verified against on-chain state.

**30 Cairo tests, 4 Rust E2E tests, all passing.** See [`../stwo-ml/docs/CLASSIFIER.md`](../stwo-ml/docs/CLASSIFIER.md) for the full architecture.

## Deployed Contracts (Starknet Sepolia)

| Version | Address | Class Hash | Features |
|---------|---------|------------|----------|
| **v39 (Current)** | [`0x0121d1...c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) | `0x0473c8...2feb` | 6-step streaming GKR verification (v25 protocol), aggregated weight binding, full ML inference verified on-chain |
| v12 | [`0x0121d1...c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) | `0x46016a...ffa7` | Packed QM31 single-TX entrypoint (`verify_model_gkr_v4_packed`), ~3.3x calldata reduction |
| v11 | [`0x0121d1...c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) | `0x08a5b7...c527` | Subtree commitment fix, zero-tree cache, input validation, deprecated modes removed |
| v10 | [`0x0121d1...c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) | `0x644792...e273` | GKR v4 path, deferred proofs, all layer types, 5-min upgrade |
| v4 | [`0x0068c7...86eb7`](https://sepolia.starkscan.co/contract/0x0068c7023d6edcb1c086bed57e0ce2b3b5dd007f50f0d6beaec3e57427c86eb7) | `0x3b870a...79a56` | Sumcheck, batch, GKR, unified, direct, upgrade |
| v3 | [`0x048070...29160`](https://sepolia.voyager.online/contract/0x048070fbd531a0192f3d4a37eb019ae3174600cae15e08c737982fae5d929160) | `0x32d1a0...d4a02` | Sumcheck, batch, GKR, unified |
| v2 | [`0x01c102...bd7f8`](https://sepolia.voyager.online/contract/0x01c102bbf1b8a7c2c37b02a7cef7e2baf06dcce94432e2aecda233b79adbd7f8) | `0x7845b0...2efe5` | Sumcheck, batch |
| v1 | [`0x053118...8569e`](https://sepolia.voyager.online/contract/0x0531182369ea82331ac39854faab986ba61907c2f88aa75120636a427ff8569e) | `0x55175...cd40` | Sumcheck only |

## VM31 Privacy Pool Contract

The VM31Pool contract manages a shielded transaction pool using a Poseidon2-M31 Merkle tree for note commitments. Supports deposits (public -> shielded), withdrawals (shielded -> public), and private transfers (2-in/2-out spends) — all verified via STARK proofs.

### Deployed (Starknet Sepolia)

| Field | Value |
|-------|-------|
| **Address** | [`0x07cf94...e1f9`](https://sepolia.starkscan.co/contract/0x07cf94e27a60b94658ec908a00a9bb6dfff03358e952d9d48a8ed0be080ce1f9) |
| **Class Hash** | `0x046d316ca9ffe36adfdd3760003e9f8aa433cb34105619edcdc275315a2c8405` |
| **Owner/Relayer** | `0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344` |
| **Verifier** | EloVerifier v12 (`0x0121d1...c005`) |

### Batch Processing Protocol (3-step)

```
  1. submit_batch_proof(deposits, withdrawals, spends, proof_hash, recipients)
     -> Verifies STARK, stores batch metadata, returns batch_id

  2. apply_batch_chunk(batch_id, start, count)
     -> Processes Merkle insertions + nullifiers for a chunk of transactions
     -> Relayer-only before timeout (120 blocks), permissionless after

  3. finalize_batch(batch_id)
     -> Verifies all transactions processed, updates canonical root
```

### Key Features

| Feature | Detail |
|---------|--------|
| **Merkle tree** | Depth-20 on-chain Poseidon2-M31 tree (packed `lo`/`hi` felt252 digests) |
| **Multi-asset** | Per-asset ERC-20 vault accounting, `register_asset()` for new tokens |
| **Nullifier set** | O(1) lookup for double-spend prevention |
| **Root history** | Ring buffer of 256 historical roots for concurrent proof generation |
| **Upgradability** | 5-minute timelocked `propose_upgrade` / `execute_upgrade` |
| **Verifier change** | Timelocked `propose_verifier_change` / `execute_verifier_change` |
| **Pause/unpause** | Owner-controlled emergency pause |
| **Batch timeout** | 120-block relayer exclusivity, then permissionless finalization |

### Deployment

```bash
# Deploy the pool contract
cd elo-cairo-verifier
./scripts/deploy.sh \
  --contract vm31-pool \
  --relayer 0x<relayer_address> \
  --verifier 0x<elo_verifier_address> \
  0x<owner_address>
```

The deploy script auto-updates `scripts/pipeline/lib/contract_addresses.sh`.

## Architecture

```
  elo_cairo_verifier/
  +-- src/
  |   +-- field.cairo            M31/CM31/QM31 field tower + qm31_inverse
  |   +-- channel.cairo          Poseidon252 Fiat-Shamir channel
  |   +-- types.cairo            Proof type definitions (Serde)
  |   +-- sumcheck.cairo         Sumcheck round verification (degree-2, degree-3)
  |   +-- mle.cairo              MLE opening proof verification (14 Merkle queries)
  |   +-- gkr.cairo              GKR batch verification (~350 lines)
  |   +-- logup.cairo            LogUp table-side sum + batch Montgomery inversion
  |   +-- layer_verifiers.cairo  Per-layer verifiers (MatMul, Add, Mul, Activation,
  |   |                          LayerNorm, RMSNorm, Dequantize, Attention)
  |   +-- model_verifier.cairo   Full GKR model walk + deferred proofs (~500 lines)
  |   +-- ml_air.cairo           ML Air trait + STARK verification (~950 lines)
  |   +-- verifier.cairo         ISumcheckVerifier contract (~1300 lines)
  |   +-- audit.cairo            On-chain audit record storage
  |   +-- access_control.cairo   Role-based access control for audit/admin
  |   +-- view_key.cairo         View key registration for note decryption
  |   +-- vm31_merkle.cairo      Poseidon2-M31 Merkle tree (packed lo/hi digests)
  |   +-- vm31_verifier.cairo    Batch public input verification for privacy pool
  |   +-- vm31_pool.cairo        VM31PoolContract — privacy pool with batch protocol
  |   +-- mock_proof_verifier.cairo  Mock verifier for pool integration tests
  |   +-- mock_erc20.cairo       Mock ERC-20 for pool deposit/withdraw tests
  |   +-- lib.cairo              Module registry
  +-- tests/
      +-- test_field.cairo                 27 field arithmetic tests
      +-- test_channel.cairo               10 Poseidon channel tests
      +-- test_verifier.cairo              6 single/batched matmul tests
      +-- test_batch.cairo                 25 batch sumcheck tests
      +-- test_gkr.cairo                   34 GKR verification tests
      +-- test_unified.cairo               19 verify_model() integration tests
      +-- test_direct.cairo                27 verify_model_direct() + upgrade tests
      +-- test_model_gkr_contract.cairo    10 GKR registration + negative tests
      +-- test_layer_verifiers.cairo       Per-layer type verification tests
      +-- test_logup.cairo                 LogUp table verification tests
      +-- test_model_verifier.cairo        Full model GKR walk tests
      +-- test_sp3_cross_language.cairo    Cross-language Serde roundtrip tests
      +-- test_access_control.cairo        13 role-based access control tests
      +-- test_audit.cairo                 Audit record storage tests
      +-- test_vm31_merkle.cairo           Poseidon2-M31 Merkle tree tests
      +-- test_vm31_verifier.cairo         Batch public input verification tests
      +-- test_vm31_pool.cairo             41 pool contract integration tests
      +-- test_vm31_binder.cairo           Cross-component binding tests
```

## GKR Model Verification Pipeline

The recommended path — `verify_model_gkr` — runs 100% on-chain:

```
  PROVER (off-chain, GPU)                    VERIFIER (on-chain, Cairo)
  =======================                    =========================

  prove_model_pure_gkr()                     register_model_gkr()
       |                                          |
       v                                          v
  GKR proof:                                 Store: weight commits
  - Per-layer sumcheck proofs                      circuit descriptor hash
  - Weight MLE openings
  - Deferred proofs (DAG)                    verify_model_gkr()
       |                                          |
       v                                          v
  cairo_serde.rs                             Phase 0: Setup
  serialize_gkr_model_proof()                  - Check model registered
       |                                       - Recompute IO commitment
       v                                       - Init Poseidon channel
  starknet.rs                                      |
  build_verify_model_gkr_calldata()                v
       |                                     Phase 1: GKR Walk
       v                                       - Output claim from IO
  sncast invoke                                - For each layer:
  --function verify_model_gkr -------->          Tag 0: MatMul sumcheck
                                                 Tag 1: Add split + defer
                                                 Tag 3: Activation LogUp
                                                 Tag 4: LayerNorm eq-check
                                              - Process deferred proofs
                                              - Verify input MLE
                                                   |
                                                   v
                                             Phase 2: Weight verify
                                              - MLE opening checks
                                                   |
                                                   v
                                             Phase 3: Finalize
                                              - Store proof_hash
                                              - Emit event
                                              - Return true
```

## Residual / DAG Circuit Support

v9 introduces **deferred proofs** for skip connections:

```
  Input --> MatMul --> FORK --+--> ReLU --> MatMul --> ADD --> Output
                              |                        ^
                              +---- skip connection ---+

  GKR Walk (trunk path):              Deferred Proofs:
  1. Add: split claim                 After walk:
  2. MatMul (trunk): sumcheck         1. Skip MatMul sumcheck
  3. ReLU: LogUp verify                  (using saved claim point)
  4. MatMul: sumcheck
  --> verify input MLE                --> collect weight claim
```

The verifier saves claim points during the walk when hitting Add layers, then uses them to reconstruct deferred claims and verify skip-branch proofs.

## Verification Modes

| Mode | Function | Use Case |
|------|----------|----------|
| **GKR Model (packed)** | `verify_model_gkr_v4_packed` | Packed QM31 format — ~3.3x calldata reduction, single TX for most models |
| **GKR Model** | `verify_model_gkr` | 100% on-chain GKR walk — all layer types, DAG circuits |
| **Direct** | `verify_model_direct` | Batch sumchecks + STARK hash binding (eliminates Stage 2) |
| **Unified** | `verify_model` | All matmuls + batched + GKR in one transaction |
| **GKR** | `verify_gkr` | LogUp/GrandProduct lookup argument proofs |
| **Batched matmul** | `verify_batched_matmul` | Lambda-weighted combined sumcheck |
| **Single matmul** | `verify_matmul` | One matrix multiplication proof |

## Layer Types Supported

| Tag | Layer | Verification Method | Status |
|:---:|-------|--------------------:|:------:|
| 0 | MatMul | Sumcheck (degree-2, log_k rounds) | Verified on-chain |
| 1 | Add | Direct claim split + deferred proof | Verified on-chain |
| 2 | Mul | Eq-sumcheck (degree-3) | Tested |
| 3 | Activation (ReLU/GELU/Sigmoid) | LogUp lookup table | Verified on-chain |
| 4 | LayerNorm | Eq-sumcheck + LogUp rsqrt | Verified on-chain |
| 5 | Attention | Composed sub-matmul sumchecks | Prover only |
| 6 | Dequantize (INT4/INT8) | LogUp 2D lookup table | Tested |
| 7 | MatMulDualSimd | Sumcheck (prover variant) | Prover only |
| 8 | RMSNorm | Eq-sumcheck + LogUp rms | Tested |

## IO Commitment Recomputation

All verification paths recompute the IO commitment on-chain from raw data:

```
  Raw IO Layout: [in_rows, in_cols, in_len, in_data...,
                  out_rows, out_cols, out_len, out_data...]

  On-chain:  io_commitment = Poseidon(raw_io_data)

  GKR path additional checks:
  - MLE(output, r_out) == output_claim   (random evaluation)
  - MLE(input, r_final) == final_claim   (GKR walk endpoint)
```

## Contract Interface

```cairo
#[starknet::interface]
trait ISumcheckVerifier<TContractState> {
    // -- Registration --
    fn register_model(ref self: TContractState, model_id: felt252, weight_commitment: felt252);
    fn register_model_gkr(ref self: TContractState, model_id: felt252,
        weight_commitments: Array<felt252>, circuit_descriptor: Array<felt252>);

    // -- GKR Model Verification (recommended) --
    fn verify_model_gkr(ref self: TContractState, model_id: felt252,
        raw_io_data: Array<felt252>, circuit_depth: u32,
        num_layers: u32, matmul_dims: Array<u32>,
        dequantize_bits: Array<u64>, proof_data: Array<felt252>,
        weight_commitments: Array<felt252>,
        weight_opening_proofs: Array<felt252>) -> bool;

    // -- GKR v4 Packed (3.3x smaller calldata via QM31 packing) --
    fn verify_model_gkr_v4_packed(ref self: TContractState, model_id: felt252,
        raw_io_data: Array<felt252>, circuit_depth: u32,
        num_layers: u32, matmul_dims: Array<u32>,
        dequantize_bits: Array<u64>, proof_data: Array<felt252>,
        weight_commitments: Array<felt252>,
        weight_binding_mode: u32, weight_binding_data: Array<felt252>,
        weight_opening_proofs: Array<felt252>) -> bool;

    // -- Other verification modes --
    fn verify_matmul(ref self: TContractState, model_id: felt252,
        proof: MatMulSumcheckProof) -> bool;
    fn verify_batched_matmul(ref self: TContractState, model_id: felt252,
        proof: BatchedMatMulProof) -> bool;
    fn verify_gkr(ref self: TContractState, model_id: felt252,
        proof: GkrBatchProof) -> bool;
    fn verify_model(ref self: TContractState, model_id: felt252,
        proof: ModelProof) -> bool;
    fn verify_model_direct(ref self: TContractState, model_id: felt252,
        session_id: felt252, raw_io_data: Array<felt252>,
        weight_commitment: felt252, num_layers: u32, activation_type: u8,
        batched_proofs: Array<BatchedMatMulProof>,
        activation_stark_data: Array<felt252>) -> bool;

    // -- Policy --
    fn register_model_policy(ref self: TContractState, model_id: felt252, policy_hash: felt252);
    fn get_model_policy(self: @TContractState, model_id: felt252) -> felt252;

    // -- Queries --
    fn get_verification_count(self: @TContractState, model_id: felt252) -> u64;
    fn is_proof_verified(self: @TContractState, proof_hash: felt252) -> bool;
    fn get_model_commitment(self: @TContractState, model_id: felt252) -> felt252;
    fn get_model_circuit_hash(self: @TContractState, model_id: felt252) -> felt252;
    fn get_model_gkr_weight_count(self: @TContractState, model_id: felt252) -> u32;
    fn get_model_policy(self: @TContractState, model_id: felt252) -> felt252;
    fn get_owner(self: @TContractState) -> ContractAddress;

    // -- Upgradability (5-minute timelock) --
    fn propose_upgrade(ref self: TContractState, new_class_hash: ClassHash);
    fn execute_upgrade(ref self: TContractState);
    fn cancel_upgrade(ref self: TContractState);
    fn get_pending_upgrade(self: @TContractState) -> (ClassHash, u64);
}
```

## Field Tower

All arithmetic uses `u64` (not `felt252`) to avoid overflow:

```
  M31:  p = 2^31 - 1              Base field, single-cycle reduction
  CM31: M31[i] / (i^2 + 1)       Complex extension
  QM31: CM31[j] / (j^2 - 2 - i)  Degree-4 secure field (128-bit security)

  Key operations in field.cairo:
  - qm31_mul (Karatsuba)
  - qm31_inverse (conjugation + Fermat's little theorem)
  - eq_eval(x, y)              equality polynomial
  - fold_mle_eval(x, v0, v1)   MLE folding
  - poly_eval_degree3           degree-3 polynomial evaluation
```

## Upgrade Mechanism

Owner-only 5-minute timelock:

```
  1. propose_upgrade(new_class_hash)  -- starts 300s countdown
  2. wait 5 minutes
  3. execute_upgrade()                -- replace_class_syscall
  4. cancel_upgrade()                 -- cancel anytime before execution
```

Automated via script:

```bash
# Full upgrade (build + declare + propose + wait + execute)
./scripts/upgrade.sh --skip-delay

# Declare only (get the new class hash without upgrading)
./scripts/upgrade.sh --declare-only

# Explicit contract target
./scripts/upgrade.sh --contract 0x0121d1...c005
```

## Building

```bash
# Build contract (Sierra + CASM)
scarb build

# Run tests
scarb test
```

Requires Scarb 2.12+ and starknet 2.12.0.

## Deployment

```bash
# Declare
sncast --account deployer declare \
  --url https://api.cartridge.gg/x/starknet/sepolia \
  --contract-name SumcheckVerifierContract

# Deploy (owner = your wallet address)
sncast --account deployer deploy \
  --url https://api.cartridge.gg/x/starknet/sepolia \
  --class-hash <CLASS_HASH> \
  --arguments '<OWNER_ADDRESS>'

# Register a model
sncast invoke --function register_model_gkr \
  --calldata '<model_id> <num_weights> <weight_commit_0> ... <num_layers> <tag_0> ...'

# Verify a model
sncast invoke --function verify_model_gkr \
  --calldata $(cat d11_verify_gkr_calldata.txt)

# Check verification
sncast call --function get_verification_count --calldata '<model_id>'
```

Note: `sncast 0.54+` requires network-keyed account JSON: `{"alpha-sepolia": {"deployer": {...}}}`.

## Serialization

Proof structs match the Rust serialization in `libs/stwo-ml/src/cairo_serde.rs`:

- `serialize_gkr_model_proof()` — full GKR model proof with per-layer tags
- `serialize_matmul_sumcheck_proof()` — single matmul proof
- `serialize_batched_matmul_proof()` — batched matmul proof
- `serialize_gkr_batch_proof()` — GKR batch proof
- `serialize_model_proof_direct()` — direct verification proof

## Dependencies

```toml
[dependencies]
starknet = "2.12.0"
stwo_verifier_core = { path = "../stwo-cairo/stwo_cairo_verifier/crates/verifier_core" }
stwo_constraint_framework = { path = "../stwo-cairo/stwo_cairo_verifier/crates/constraint_framework" }
stwo_verifier_utils = { path = "../stwo-cairo/stwo_cairo_verifier/crates/verifier_utils" }
bounded_int = { path = "../stwo-cairo/stwo_cairo_verifier/crates/bounded_int" }
```

## Known Limitations

**STARK verification on-chain**: The STWO FRI verifier uses `Felt252Dict` internally, generating `squashed_felt252_dict_entries` which is not in Starknet's allowed libfuncs. STARK data is hash-bound on-chain, with full verification off-chain. The **GKR path bypasses this entirely** — no FRI needed.

**Weight MLE openings (Phase 2)**: Currently `weight_opening_proofs` is passed as an empty array. Weight commitments are verified against registration. Full MLE opening verification with Merkle proofs is planned.

## License

Apache 2.0
