# Multi-Token Decode Chain — Reproducibility Guide

This guide walks through reproducing the trustless multi-token autoregressive
generation demo on Starknet Sepolia using obelyzk. It targets the v4 recursive
verifier deployed at:

- **Contract**: `0x05736b0fb338a5de1e00f751bae3e2b65f0d8051952a5888d9cbf2f0a929e92a`
- **Class hash**: `0x1403842938e35a934f5d5502f9e23a11ab4beb85891ba78f1eb1a2655853578`
- **Network**: Starknet Sepolia
- **Branch**: `feat/multi-token-hades-trustless` on `Bitsage-Network/obelyzk.rs`

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

## Scope notes (read before reproducing)

- **Path B / synthetic kv binding (current)**: each step's `kv_cache_commitment`
  is `Poseidon(prev_kv, hash_of_input_token)`. This is a real cryptographic
  chain over input continuity, but it is *not* a chain over the actual decode-
  step KV cache state. The full forward-pass prover is used at every step
  (no KV-cache amortization).
- **Path A / real decode KV-cache binding (future work)**: the decode-step
  recursive prover hits a pre-existing RMS² sumcheck mismatch in the recursive
  witness's GKR verifier replay. Once that's fixed, swap in
  `prove_model_pure_gkr_decode_step_incremental` and the `kv_cache_commitment`
  becomes the real Merkle-rooted KV cache state. Architecture is unchanged.

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
