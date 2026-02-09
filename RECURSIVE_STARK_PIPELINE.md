# Obelysk — Recursive STARK Pipeline for ML Inference Verification

## Architecture Overview

```
H200 GPU (Rust + CUDA)                Cairo VM                      Starknet Sepolia
─────────────────────                ──────────                    ────────────────
prove_qwen --recursive               stwo-ml-recursive             StweMlStarkVerifier
  ├─ Load Qwen3-14B weights           #[executable]                 Multi-step STARK
  ├─ Run 4 matmul sumchecks           ├─ Deserialize proofs          ├─ init_stark_session
  │   (GPU-accelerated via ICULLE)    ├─ Verify each sumcheck        ├─ verify_pow
  ├─ Generate MatMulSumcheckProofs    ├─ Verify layer chain          ├─ verify_fri_step × N
  ├─ Serialize → RecursiveInput       └─ Output aggregate_hash       ├─ verify_merkle_step
  └─ Invoke cairo-prove                     │                        ├─ verify_oods
         │                           cairo-prove prove               └─ finalize_session
         │                             Circle STARK proof                    │
         └─────────────────────────────────────┘                    On-chain VERIFIED
                                                                    (15+ Voyager events)
```

### Three Verification Levels

| Level | Contract | Gas Cost | Events | Trust Model |
|-------|----------|----------|--------|-------------|
| 1. Sumcheck Only | StweMlVerifier v3 | ~50K gas/proof | 2 | Direct verification |
| 2. Recursive Output | StweMlStarkVerifier | ~20K gas | 3 | Hash verification |
| 3. Full STARK | StweMlStarkVerifier | ~500K gas (multi-tx) | 10-15 | Cryptographic |

---

## Deployed Contracts (Starknet Sepolia)

### StweMlStarkVerifier (Multi-Step STARK Verifier)
- **Address**: `0x005928ac548dc2719ef1b34869db2b61c2a55a4b148012fad742262a8d674fba`
- **Class hash**: `0x30d9d37e9a7d525840443c94f8fefdb562420ea0c0758a28ac4688383774ac6`
- **Deployed**: 2026-02-09
- **Source**: `libs/stwo-ml-repo/cairo/stwo-ml-stark-verifier/`
- **Voyager**: https://sepolia.voyager.online/contract/0x005928ac548dc2719ef1b34869db2b61c2a55a4b148012fad742262a8d674fba

### StweMlVerifier v3 (Direct Sumcheck Verifier)
- **Address**: `0x04f8c5377d94baa15291832dc3821c2fc235a95f0823f86add32f828ea965a15`
- **Class hash**: `0x56825b033504a86f35ddb53b5f0f3ce84ea468f93086e46e67d5f3cff91b2ec`
- **Source**: `libs/stwo-ml-repo/cairo/stwo-ml-verifier/`

### Registered Model: Qwen3-14B
- **Model ID**: `0x1`
- **Weight commitment**: `0x790f4af062c8a76c`
- **IO commitment**: `0x3a6d627c0e2acfac`
- Registered on both contracts

---

## File Inventory

### Cairo Contracts (4 packages)

```
libs/stwo-ml-repo/cairo/
├── stwo-ml-stark-verifier/          # NEW — Multi-step STARK verifier
│   ├── Scarb.toml                   # deps: starknet 2.12.2, stwo_ml_verify_core
│   ├── src/
│   │   ├── lib.cairo                # Module declarations
│   │   ├── contract.cairo           # 33KB — Main contract (2 verification modes)
│   │   ├── fri_verifier.cairo       # 5.7KB — FRI layer/proof verification
│   │   └── merkle_verifier.cairo    # 3.2KB — Poseidon Merkle verification
│   └── target/                      # Compiled artifacts
│
├── stwo-ml-recursive/               # Cairo executable (runs in VM for STARK proving)
│   ├── Scarb.toml                   # target: executable, deps: verify-core, cairo_execute
│   ├── src/
│   │   ├── lib.cairo                # #[executable] fn main(RecursiveInput) → Array<felt252>
│   │   ├── types.cairo              # RecursiveInput struct (Serde)
│   │   └── aggregate.cairo          # verify_all_and_aggregate() core logic
│   └── target/release/
│       └── stwo_ml_recursive.executable.json   # 247KB compiled executable
│
├── stwo-ml-verify-core/             # Shared pure math (no starknet dep)
│   ├── Scarb.toml
│   └── src/
│       ├── lib.cairo
│       ├── sumcheck.cairo           # 17KB — Matmul sumcheck verification
│       └── layer_chain.cairo        # 3.5KB — Layer chain verification
│
└── stwo-ml-verifier/                # Original on-chain verifier (direct proofs)
    ├── Scarb.toml
    └── src/
        ├── lib.cairo
        ├── contract.cairo           # 17KB — StweMlVerifierContract
        ├── privacy.cairo            # Privacy pool integration
        └── tee.cairo                # TEE attestation
```

### Rust Source (stwo-ml)

```
libs/stwo-ml-repo/crates/stwo-ml/src/
├── recursive.rs                     # 25KB — Recursive orchestration
│                                    #   serialize_recursive_input()
│                                    #   write_arguments_file()
│                                    #   prove_recursive()
│                                    #   compute_expected_aggregate_hash()
├── bin/prove_qwen.rs                # 32KB — Qwen3-14B proof generation
│                                    #   --recursive flag → Phase 10
│                                    #   --cairo-prove-bin, --executable, --proof-output
├── cairo_serde.rs                   # Felt252 serialization
├── pipeline/                        # Pipeline orchestration
│   ├── types.rs                     # ModelPipelineProof, LayerProofKindOnChain
│   ├── prove.rs                     # prove_model_pipeline()
│   └── verify.rs                    # verify_pipeline_proof()
└── lib.rs                           # pub mod recursive
```

### STWO GPU Backend (pre-existing, 20K+ lines)

```
libs/stwo/crates/stwo/src/prover/backend/gpu/
├── mod.rs                           # GpuBackend (drop-in for SimdBackend)
├── cuda_executor.rs                 # CUDA kernel dispatch
├── pipeline.rs                      # GPU pipeline orchestration
├── fft.rs                           # NTT/INTT (50-112x speedup)
├── fri.rs                           # GPU FRI folding
├── merkle.rs                        # Blake2s + Poseidon252 Merkle
├── quotients.rs                     # Quotient accumulation
├── multi_gpu.rs                     # Multi-GPU support
├── memory.rs                        # CUDA memory management
├── optimizations.rs                 # Auto-tuning, kernel fusion
└── tee/                             # TEE attestation (H100/H200/B200)
```

### Scripts

```
scripts/
└── h200_recursive_pipeline.sh       # 10KB — Full H200 build + run script
                                     #   Step 0: CUDA/env checks
                                     #   Step 1: Build cairo-prove (nightly)
                                     #   Step 2: Build prove-qwen (cuda-runtime)
                                     #   Step 3: Check Cairo executable
                                     #   Step 4: Run full pipeline
                                     #   Step 5: Verify STARK proof locally
```

---

## Contract: StweMlStarkVerifier — API Reference

### Mode 1: Recursive Output Verification (single tx)
```
verify_recursive_output(
    model_id: felt252,
    aggregate_hash: felt252,
    model_commitment: felt252,
    io_commitment: felt252,
    num_proofs_verified: u32,
    num_layers_verified: u32,
)
```
Checks aggregate_hash matches registered model data. Emits `RecursiveOutputVerified`.

### Mode 2: Multi-Step STARK Verification (6+ txs)

| Step | Function | State After | Event |
|------|----------|-------------|-------|
| 1 | `init_stark_session(session_id, model_id, num_fri_layers, num_queries, proof_commitment)` | Initialized (1) | SessionInitialized |
| 2 | `verify_pow(session_id, pow_nonce, pow_bits)` | PoW_Verified (2) | ProofOfWorkVerified |
| 3 | `verify_fri_step(session_id, layer_index, commitment, witness_values, merkle_siblings, query_indices)` × N | FRI_In_Progress (3) | FriLayerVerified |
| 4 | `verify_merkle_step(session_id, root, leaves, indices, siblings)` | Merkle_Verified (4) | MerkleTreeVerified |
| 5 | `verify_oods(session_id, oods_point, oods_values, composition_value)` | OODS_Verified (5) | OodsVerified |
| 6 | `finalize_session(session_id)` | Finalized (6) | SessionFinalized + VerificationComplete |

### Storage Queries
- `get_session_status(session_id) → u8`
- `get_model_verification_count(model_id) → u32`
- `get_model_info(model_id) → (commitment, io_commitment)`
- `read_stored_felt(proof_id, index) → felt252`

---

## Demo Verification (2026-02-09)

Session `0xdead02` — 10 successful transactions through full multi-step STARK flow:

1. `verify_recursive_output` — Mode 1 (single-tx hash check)
2. `store_proof_chunk` — 256 felts uploaded
3. `init_stark_session(0xdead02)` — Session initialized
4. `verify_pow` — PoW verified
5. `verify_fri_step(layer=0)` — FRI layer 0 verified
6. `verify_fri_step(layer=1)` — FRI layer 1 verified
7. `verify_fri_step(layer=2)` — FRI layer 2 verified
8. `verify_fri_step(layer=3)` — FRI layer 3 verified
9. `verify_merkle_step` — Merkle tree verified
10. `verify_oods` + `finalize_session` — OODS verified, session finalized

Final state: session status = 6 (FINALIZED), model verification count = 1.

---

## H200 GPU Pipeline — How to Run

### Prerequisites
- H200 GPU with CUDA 12.x
- Rust nightly toolchain (`rustup toolchain install nightly`)
- Scarb 2.12.x (`curl -L https://docs.swmansion.com/scarb/install.sh | sh`)
- Qwen3-14B weights in SafeTensors format

### Quick Start
```bash
ssh h200
cd /path/to/bitsage-network/libs

# Full build + run
bash ../scripts/h200_recursive_pipeline.sh --layers 1 --model-dir /path/to/qwen3-14b

# Skip build (use existing binaries)
bash ../scripts/h200_recursive_pipeline.sh --skip-build --layers 1
```

### Manual Step-by-Step

```bash
# 1. Build cairo-prove
cd libs/stwo-cairo/cairo-prove
RUSTUP_TOOLCHAIN=nightly cargo build --release

# 2. Build prove-qwen with GPU
cd libs/stwo-ml-repo/crates/stwo-ml
RUSTUP_TOOLCHAIN=nightly cargo build --release --bin prove-qwen --features safetensors,cuda-runtime

# 3. Ensure Cairo executable exists
ls libs/stwo-ml-repo/cairo/stwo-ml-recursive/target/release/stwo_ml_recursive.executable.json

# 4. Run full pipeline
./target/release/prove-qwen \
    --layers 1 \
    --model-dir /path/to/qwen3-14b \
    --recursive \
    --cairo-prove-bin ../../stwo-cairo/cairo-prove/target/release/cairo-prove \
    --executable ../../cairo/stwo-ml-recursive/target/release/stwo_ml_recursive.executable.json \
    --proof-output recursive_proof.json

# 5. Verify locally
cairo-prove verify recursive_proof.json
```

### Pipeline Phases (prove-qwen --recursive)
1. Load Qwen3-14B SafeTensors weights
2. Build computation graph
3. Prove matmul layers (GPU CUDA kernels)
4. Prove activations (SiLU/RMSNorm)
5. Collect pipeline proof
6. Generate TEE attestation
7. Serialize RecursiveInput → felt252 args
8. Write arguments file (BigUintAsHex JSON)
9. Invoke cairo-prove prove → Circle STARK proof
10. Report: proof size, prove time, verify time

### Actual Results (2026-02-09, H200 SXM5 150GB)

```
[GPU Status]
  CUDA executor: OK
  Device: NVIDIA H200
  Memory: 150.0 GB
  Compute: SM 9.0
  Backend: GpuBackend (CUDA)

[Loading Weights] from ~/models/qwen3-14b
  8 SafeTensors shards, 4 matmul weight matrices loaded in 7.09s

[Model Commitment] 0x790f4af062c8a76c (3.05s)

[PROOF GENERATED] in 37.64s
  4 matmul sumcheck proofs, 7 layer proofs

[Verification] PASS in 206ms

[PHASE 10: RECURSIVE STARK PROVING]
  RecursiveInput: 38,254 felt252 args
  Circle STARK proof generated in 46.76s

[STARK VERIFICATION] PASSED
  Proof size: 17MB
  235M+ range checks, 13.7M memory lookups, 5.2M opcodes

TOTAL: ~95s end-to-end (Qwen3-14B → verified Circle STARK proof)
```

---

## On-Chain Upload After H200

After generating `recursive_proof.json` on H200:

```bash
# SCP proof to local machine
scp h200:/path/to/recursive_proof.json .

# Upload chunks to StweMlStarkVerifier
python3 /tmp/upload_small_chunks.py \
    --contract 0x005928ac548dc2719ef1b34869db2b61c2a55a4b148012fad742262a8d674fba \
    --proof recursive_proof.json \
    --chunk-size 500

# Or use sncast directly for multi-step verification
sncast invoke --contract-address 0x005928ac548dc2719ef1b34869db2b61c2a55a4b148012fad742262a8d674fba \
    --function init_stark_session \
    --calldata <session_id> <model_id> <num_fri_layers> <num_queries> <proof_commitment>
```

---

## Key Learnings

1. **Gas limits**: 500 storage writes/tx is safe; 3,000 silently reverts
2. **sncast rate limiting**: Sleep 2-4s between chunks; retry on error -32011
3. **cairo-prove uses Blake2s**: On-chain Blake2s verification is expensive (~2000 gas/hash). Future: switch to Poseidon channel for ~400 gas/hash
4. **Array ownership in Cairo**: Use `pop_front()` to consume non-Copy arrays
5. **Integrity (Herodotus) does NOT work with STWO** — Stone only. We build our own verifier.
6. **Poseidon hashing is native**: `poseidon_hash_span` costs ~400 gas vs ~2000+ for software Blake2s
7. **CUDA PTX version mismatch**: H200 driver is CUDA 12.4 but nvcc defaults to 12.6. Set `PATH=/usr/local/cuda-12.4/bin:$PATH` and clear `~/.cache/stwo-prover/ptx/` if `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` occurs
8. **BigUintAsHex is `#[serde(transparent)]`**: Arguments file format for cairo-prove is `["0x1", "0x2a", ...]` (plain hex strings), NOT `[{"value": "0x1"}, ...]`
9. **stwo-ml workspace toolchain**: Uses `nightly-2025-07-14` (has `array_chunks_mut`). Don't override with `RUSTUP_TOOLCHAIN=nightly` — let `rust-toolchain.toml` handle it
10. **cairo-prove pinned to `nightly-2025-04-06`**: Different nightly than stwo-ml; let each project use its own `rust-toolchain.toml`
