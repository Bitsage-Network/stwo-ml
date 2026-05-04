# Multi-Token Decode Chain — Reproducibility Guide

This guide walks through reproducing the trustless multi-token autoregressive
generation demo on Starknet Sepolia using obelyzk. It targets the v4 recursive
verifier deployed at:

- **Contract**: `0x05736b0fb338a5de1e00f751bae3e2b65f0d8051952a5888d9cbf2f0a929e92a`
- **Class hash**: `0x1403842938e35a934f5d5502f9e23a11ab4beb85891ba78f1eb1a2655853578`
- **Network**: Starknet Sepolia
- **Branch**: `feat/multi-token-hades-trustless` on `Bitsage-Network/obelyzk.rs`

## Live demo on-chain (May 3 2026)

**Session 4 — Llama-3.2-1B (16 layers), 3 decode steps, 100-bit FRI strict policy**:

| Step | Tx |
|---|---|
| Register model | [`0x425c73c1...`](https://sepolia.starkscan.co/tx/0x425c73c1fd23bd88262ec75a09e79fb1a8f40bcba918b4a10251d41a568dff1) |
| Start session | [`0x6ff88242...`](https://sepolia.starkscan.co/tx/0x6ff88242f36fa50d821b5bc755e50ab02a50f915c9edda0bed321d2ac79dfef) |
| Step 0 | [`0x5eb85b82...`](https://sepolia.starkscan.co/tx/0x5eb85b826d44ad3b12d83e281c78471362c1953688bb9e522592d726069fa34) |
| Step 1 | [`0x6a88a2e4...`](https://sepolia.starkscan.co/tx/0x6a88a2e4f33253dde26ae8436c996f4634ab4ee2aa59ee915d6b9083b930552) |
| Step 2 | [`0x303ea3d2...`](https://sepolia.starkscan.co/tx/0x303ea3d27ba5ed41dea39701983bb70f48f5e32b5f9b4e8594f18c82b9c0575) |
| Finalize | [`0x4cb0fe57...`](https://sepolia.starkscan.co/tx/0x4cb0fe5726edc43e816d0a95fe1058870f6d03ba3589d023ba16b0447497c28) |

Each step proof: ~4,690 felts in single TX. Total prove time on M1 CPU: 3h 18min.
Total Sepolia STRK fees: ~6 STRK. Hades Phase A aggregate hash registered:
`0x61d98230ea7dd05038b45b8a41406c3b8d62583dbba23f204c948221bcde02`.

## What this demonstrates

Each forward pass generates one recursive STARK (~3-5K felts) submitted in a
single TX. A session contract on-chain enforces continuity across submissions:

1. `start_decode_session(model_id, initial_kv_commitment) → session_id`
2. `verify_decode_step(session_id, step_idx, ...recursive_proof)` — the proof
   body contains a `prev_kv_cache_commitment` (header felt [24]) which **must**
   equal the session's stored `last_kv_commitment`. On success, the session
   rolls forward to the proof's `kv_cache_commitment` (header felt [25]).
3. `finalize_decode_session(session_id) → final_kv_commitment`

If any step is reordered, skipped, or has a tampered KV commitment, the
on-chain verification reverts at the continuity check (or earlier, at the
STARK's Fiat-Shamir verification — the kv commitments are channel-bound).

## Scope notes

**Path A / real decode-step KV-cache binding**: each step's
`kv_cache_commitment` is the actual incremental KV-cache Merkle commitment
from `prove_model_pure_gkr_decode_step_incremental`. The recursive witness
correctly replays the decode prover's channel-mixed kv commitments
(witness.rs:446 — fix landed May 3). On-chain validated Session 7
(`0x58cca48a5ae40cf1...`) on SmolLM2-135M.

**Path A scope limitation**: the chain-AIR's carry-chain validity constraint
enforces `c ∈ {0, 1}` (degree 2). For larger models (Llama-1B+), some
decode-mode channel ops produce limb-level integer sums where `c = -1`
(borrow) is needed — caused by P's high-bit structure
(`P = 2^251 + 17*2^192 + 1`). When this happens, the prover's
`compute_addition_carry_chain` returns the FALLBACK and the recursive STARK
fails with `ConstraintsNotSatisfied`. Resolving requires either:
- bumping `max_constraint_log_degree_bound` to `log_n+2` with degree-3
  validity constraint `c*(c-1)*(c+1) == 0`, AND increasing stwo's
  `COMPOSITION_LOG_SPLIT` (workspace-level change), OR
- adding 8 borrow columns to the chain trace at the cost of breaking the
  deployed Cairo verifier's hardcoded `n_trace=48`.

Either is a multi-day refactor beyond the current scope.

**Path B / synthetic chain**: regular `prove_model_pure_gkr` + per-step
input-hash kv commitments. Used for Llama-1B in Session 4
(`0x4cb0fe5726edc43e...`). Cryptographic chain via Fiat-Shamir is real;
the binding is over input-token continuity rather than the actual decode
KV cache. Valid for any model size since it doesn't trigger the carry-chain
edge case.

**Recommendation**: use path A for SmolLM2-class (≤8-layer) models; use
path B for larger. The CLI selects based on model size internally as a
follow-up; for now `--decode --recursive` uses path A and falls back if
it hits ConstraintsNotSatisfied.

## Required env

Set before running:

```bash
export STWO_AGGREGATED_FULL_BINDING=1
export STWO_SKIP_BATCH_TOKENS=1
export STWO_MLE_N_QUERIES=5
export OBELYZK_HADES_AIR=0      # chain-only 3-tree (5K-felt cap)
export OBELYZK_LOGUP=0          # disable LogUp interaction tree
```

Strict policy (`PolicyConfig::standard()`) is the default — all four soundness
gates closed. Do **not** set `STWO_SKIP_RMS_SQ_PROOF`, `STWO_ALLOW_MISSING_NORM_PROOF`,
`STWO_PIECEWISE_ACTIVATION=0`, or `STWO_ALLOW_LOGUP_ACTIVATION`.

## Step 1 — Build prove-model

```bash
cd libs/engine
cargo build --release --bin prove-model \
  --features "std,cli,model-loading,safetensors,audit,parallel-audit"
```

## Step 2 — Generate a chain

For SmolLM2-135M (smoke test, 1 layer, ~1 min/step):

```bash
mkdir -p /tmp/zkml_runs/chain_smoke
prove-model \
  --model-dir ~/.obelysk/models/smollm2-135m \
  --layers 1 --decode --decode-steps 3 --prefill-len 4 \
  --gkr --format ml_gkr --recursive \
  --output /tmp/zkml_runs/chain_smoke/chain.json
```

For Llama-3.2-1B (real demo, 16 layers, ~50 min/step on M1 CPU):

```bash
mkdir -p /tmp/zkml_runs/chain_llama
prove-model \
  --model-dir ~/.obelysk/models/llama-3.2-1b \
  --decode --decode-steps 3 --prefill-len 4 \
  --gkr --format ml_gkr --recursive \
  --output /tmp/zkml_runs/chain_llama/chain.json
```

Expected outputs in the proof directory:
- `chain.chain_manifest.json` — index file
- `chain_step_<N>.recursive.json` — per-step recursive proof (~3-5K felts)
- `chain_step_<N>.recursive.json.hades_args.json` — Hades pairs sidecar

## Step 3 — Compute Hades Phase A hash

```bash
libs/engine/scripts/prove_hades_level1.sh /tmp/zkml_runs/chain_llama
```

This computes a per-step keccak256 over the canonical 32-byte felt
concatenation of each step's Hades sidecar, then an aggregate keccak across
all steps, truncated to felt252. Updates `chain_manifest.json::level1_proof_hash`
and writes `level1_aggregate.hash.txt`.

The aggregate hash is registered on-chain at `register_model_recursive(...)`.
Anyone can later fetch the sidecars, recompute the hash via this script, and
compare to `get_level1_proof_hash(model_id)` on-chain.

## Step 4 — Submit chain on-chain

```bash
KEYSTORE_PATH=libs/deployment/sepolia_keystore.json \
KEYSTORE_PASSWORD="<password>" \
ACCOUNT_ADDRESS=0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344 \
CONTRACT=0x05736b0fb338a5de1e00f751bae3e2b65f0d8051952a5888d9cbf2f0a929e92a \
PROOF_DIR=/tmp/zkml_runs/chain_llama \
N_MATMULS=<from manifest>  HIDDEN_SIZE=2048  NUM_TRANSFORMER_BLOCKS=16 \
node libs/engine/scripts/submit_decode_chain.mjs
```

The script registers the model (idempotent), starts a session anchored to
`initial_kv_commitment`, submits N decode-step verify TXs, and finalizes.
Each TX prints its hash and a Starkscan explorer link.

## Step 5 — Independently audit a finalized chain

Given an `EXPLORER_TX` link to a `finalize_decode_session` TX:

1. Pull the session_id from event keys.
2. Fetch session state: `get_decode_session(session_id) → DecodeSession`
   - Confirms `step_count`, `last_kv_commitment`, `finalized: true`.
3. For each step's tx, fetch the `verify_decode_step` calldata and the
   bound `kv_cache_commitment` (header [25]) and `prev_kv_cache_commitment`
   (header [24]). Verify continuity: step *N*'s prev == step *N-1*'s curr.
4. Fetch `level1_proof_hash` via `get_level1_proof_hash(model_id)`.
5. Pull the off-chain Hades sidecars (published via the proof_file URLs
   in chain_manifest.json), run `prove_hades_level1.sh`, confirm the
   aggregate hash matches the on-chain value.

If steps 1-5 all match, the chain is independently validated:
- Each step's STARK is on-chain verified (the contract ran stwo's verifier).
- KV continuity is on-chain verified (session storage holds the rolling
  commitment; each step's prev_kv was checked against it).
- Hades pairs witness integrity is auditable off-chain (sidecar bytes
  hash-bound to on-chain registration).

## Verification cost (Sepolia, May 2 baseline)

| Action | L2 gas | STRK fee |
|---|---|---|
| `register_model_recursive` | ~1M | ~0.01 STRK |
| `start_decode_session` | ~200K | ~0.002 STRK |
| `verify_decode_step` (per step) | ~250M | ~2 STRK |
| `finalize_decode_session` | ~150K | ~0.001 STRK |

5-step chain ≈ ~10 STRK total at current Sepolia rates.

## Known limitations / future work

- **Decode-step KV-cache binding (Path A)**: the real KV-cache prover path
  has an RMS² sumcheck mismatch in the recursive witness replay. Resolving
  this enables true decode-step amortization (each step ~5x cheaper).
- **Hades Phase B (Level-0 compressor)**: today's Phase A binds the Hades
  pairs sidecar to a hash; full on-chain Hades validation needs a separate
  Level-0 recursive STARK over the cairo-prove output.
- **Mainnet**: no audit yet, calldata cap is 4K on mainnet (vs 5K Sepolia)
  so n_queries may need to drop to 14 (≈92-bit security).
- **Single-prover model**: prove-model is the only prover; no multi-party
  prove-aggregator yet.

## Source pointers

- Cairo contract: `libs/elo-cairo-verifier/src/recursive_verifier.cairo`
- Recursive prover: `libs/engine/src/recursive/`
  - Public-input plumbing: `recursive/types.rs`, `recursive/witness.rs`
  - Fiat-Shamir mixing: `recursive/prover.rs`, `recursive/verifier.rs`
- Calldata serialization: `libs/engine/src/cairo_serde.rs`
- Decode-mode CLI: `libs/engine/src/bin/prove_model.rs::run_decode_mode`
- Submission scripts: `libs/engine/scripts/submit_decode_chain.mjs`,
  `prove_hades_level1.sh`
