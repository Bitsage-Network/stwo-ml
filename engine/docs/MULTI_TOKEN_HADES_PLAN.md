# Multi-Token Decode Chain + Hades Two-Level Recursion — Implementation Plan

**Branch**: `feat/multi-token-hades-trustless`
**Target model**: Llama-3.2-1B-Instruct (from unsloth public mirror)
**Scope**: Goal A — architecture supports unlimited length; demo proves shape on M1

## Status as of branch start

Three milestones already on Sepolia from prior session
(verifier `0x3b041fae89d75b871ee29a3f4351c1afd6230a6d8eb87832f214ce3113ba9a4`):

| Tx | Layers | Policy | Felts | Note |
|---|---|---|---|---|
| `0x778ef128…` | 30 (full SmolLM2) | strict | 4,871 | Single forward pass |
| `0x37fb9182…` | 1 | strict | 4,035 | Format demo |
| `0x72d7433e…` | 30 | strict | 4,886 | Tampered → REVERTED (Merkle root mismatch) |

Two architectural gaps remain:
1. Each on-chain proof is a **single forward pass**. Real LLM use is autoregressive
   — N decode steps where step *i*'s input KV cache = step *i-1*'s output cache.
   The contract has no mechanism to enforce continuity across submissions.
2. The `hades_commitment` value at proof position [18] is computed and channel-bound,
   but **no Level-1 cairo-prove proof of those Hades permutations is ever generated
   or verified**. Soundness story is half-circular.

This plan closes both.

## Architectural design

### Multi-token chain (per-step recursive STARK + on-chain session)

Each decode step is its own recursive STARK (~5K felts, 1 TX). The **contract**
enforces continuity by storing per-session state and validating each new step's
`prev_kv_cache_commitment` parameter against the session's `last_kv_commitment`
storage cell.

```
                 ┌── SESSION ON-CHAIN ──────────────────────────────────────┐
                 │  session_id → (model_id, last_kv, n_steps, finalized?)   │
                 └──────────────────────────────────────────────────────────┘
                              ▲                  ▲                  ▲
        ┌────────────────────┘                  │                  │
        │                                        │                  │
   prefill TX                              decode TX 1        decode TX 2 …
  start_decode_session                  verify_decode_step  verify_decode_step
   + prefill proof                       (binds prev_kv ==   (binds prev_kv ==
   sets last_kv                            session.last_kv)    session.last_kv)
                                          updates last_kv     updates last_kv
```

`finalize_decode_session(session_id) → final_kv_commitment` emits the chain
commitment as one event for off-chain consumers.

### Hades two-level recursion

**Phase A — auditable hash bind (this PR)**: generate Level-1 cairo-prove proof
off-chain, keccak-hash it, store hash on-chain at registration. Anyone can fetch
the proof, run `cairo-prove verify`, and confirm the on-chain `hades_commitment`
matches the proof's content. Closes the circular trust without changing chain
verify path.

**Phase B — full on-chain Level-0 compressor (follow-up PR, not in this branch)**:
build a recursive STARK that compresses Level-1's ~278K felts into ~5K. Add
`verify_hades_level0` contract entrypoint. Cross-link to chain proof's
`hades_commitment` for end-to-end on-chain verification.

## Concrete diff inventory

### Cairo (`libs/elo-cairo-verifier/src/recursive_verifier.cairo`)

```diff
+ struct DecodeSession {
+   model_id: felt252,
+   started_at: u64,
+   step_count: u32,
+   last_kv_commitment: felt252,
+   finalized: bool,
+ }
+
+ #[storage] struct Storage {
+   ...existing...
+ + decode_sessions: Map<u64, DecodeSession>,
+ + next_session_id: u64,
+ + level1_proof_hashes: Map<felt252, felt252>,  // model_id → keccak256(L1)
+ }
+
+ fn start_decode_session(
+   model_id: felt252,
+   prefill_recursive_proof: ...,  // same args as verify_recursive
+   initial_kv_commitment: felt252,
+ ) -> u64;  // session_id
+
+ fn verify_decode_step(
+   session_id: u64,
+   expected_step_idx: u32,
+   prev_kv_commitment: felt252,  // must match session.last_kv
+   new_kv_commitment: felt252,
+   recursive_proof: ...,
+ ) -> bool;
+
+ fn finalize_decode_session(session_id: u64) -> felt252;  // final_kv
+ fn get_decode_session(session_id: u64) -> DecodeSession;
+
+ fn register_model_recursive(
+   ...,
+   level1_proof_hash: felt252,  // NEW: keccak256 of cairo-prove L1 output (Phase A)
+ );
```

Estimated +180 LoC including events.

### Rust additions

`engine/src/recursive/types.rs`:
```diff
 pub struct RecursivePublicInputs {
   ...existing fields...
+  pub kv_cache_commitment: QM31,
+  pub prev_kv_cache_commitment: QM31,
 }
```

`engine/src/recursive/prover.rs`:
- Read `kv_cache_commitment` and `prev_kv_cache_commitment` from `GKRProof`
- Mix into Fiat-Shamir channel (after `seed_digest`, before `hades_commitment`)
- Pass through to public inputs

`engine/src/cairo_serde.rs`:
- `serialize_recursive_proof_calldata`: emit the two new QM31s in header
  (positions [22..30) become [24..32), shifting log_size and n_real_rows)
- Update header_felts count from 16 to 24 in `RecursiveCalldataSummary`

`engine/src/bin/prove_model.rs`:
- `--decode --recursive --decode-steps N --output base.json`:
  - Run prefill, save `base.prefill.recursive.json`
  - Loop N decode steps; for each step *i* save `base.step_{i}.recursive.json`
  - Emit `base.chain_manifest.json` with the full sequence + KV-cache commitments

Estimated +120 LoC.

### Submission scripts

`engine/scripts/submit_decode_chain.mjs`:
- Read `chain_manifest.json`
- Tx 1: `start_decode_session(model_id, prefill_proof, initial_kv)`
- Tx 2..N+1: `verify_decode_step(session_id, i, prev_kv, new_kv, proof)`
- Tx N+2: `finalize_decode_session(session_id)`
- Print summary table + Starkscan links

`engine/scripts/prove_hades_level1.sh`:
- For each generated `*.hades_args.json`, run `cairo-prove prove`
- Output `*.hades_level1.proof.json`
- Compute keccak hash, write `*.hades_level1.hash.txt`
- Used at registration time as `level1_proof_hash`

Estimated +250 LoC.

### Tests

`engine/tests/decode_chain_e2e.rs`:
- Prove 3-step decode chain locally
- Verify each proof individually (Rust pre-flight)
- Verify continuity: step *i*'s `prev_kv_cache_commitment` == step *i-1*'s `kv_cache_commitment`
- Adversarial: reorder steps → simulator reject; tamper kv commitment → simulator reject
- ~150 LoC

### Contract redeploy

- Compile new Cairo class (target `~/bitsage-network/libs/elo-cairo-verifier/target/release/`)
- Declare via sncast (~25 STRK)
- Deploy fresh instance (~0.5 STRK)
- Re-register Llama-3.2-1B model on the new contract (with `level1_proof_hash`)
- Total cost: ~26 STRK from v1 deployer (currently ~801 STRK after yesterday's ops)

## Phased execution

| # | Phase | Wall time | Output |
|---|---|---|---|
| 0 | Download Llama-3.2-1B + validate single-pass on existing v3 contract | 2-4 hrs | Confirms Llama loader works; baseline timing |
| 1 | Implement multi-token Cairo session contract + Rust public input plumbing | 1 day | Code, no on-chain submit yet |
| 2 | Build `submit_decode_chain.mjs` + decode_chain_e2e.rs test | 0.5 day | Local 3-step chain verifies |
| 3 | Compile + declare + deploy v4 verifier | 1 hr | New contract address, STRK spent |
| 4 | Demo run: prefill + 5-10 token decode chain on Llama-3.2-1B (overnight) | 3-7 hrs | TX hashes, on-chain session events |
| 5 | Hades Phase A: `prove_hades_level1.sh` + registration hash | 0.5 day | Level-1 proof file, hash on-chain |
| 6 | Document audit procedure + reproducible README | 0.5 day | `HADES_AUDIT.md`, `MULTI_TOKEN_README.md` |

Total: **~3-4 working days** for both features end-to-end.

Hades Phase B (full on-chain Level-0 compressor) is deferred to a follow-up
branch; it's significant work (~3-5 more days, ~600 LoC) and the design depends
on what Phase A reveals about Level-1 size + cairo-prove behavior at scale.

## Risks & mitigations tracked

| Risk | Mitigation |
|---|---|
| Llama-3.2-1B too slow on M1 (>1 hr/token) | Fall back to SmolLM2-360M or run demo on H100 cloud |
| `cairo-prove` on 278K-felt Hades fails on M1 | Run on Brev H100 (~$1) or split Hades pairs |
| New Cairo class exceeds 81K-felt cap | Use library_call splitting (already pattern in current verifier) |
| Format drift from stwo-gpu version bumps | Pin stwo-gpu in Cargo.toml; document upgrade procedure |
| Session state grows unboundedly | Add session expiry (gc after N days) in v5+ |
| Adversarial concurrency (two parties claim same session) | session_id is monotonic + bound to caller via storage; first to start wins |
| KV cache commitment trivially forgeable if not tied to weights | `kv_cache_commitment` is hashed with `weight_super_root` in Fiat-Shamir |

## Public claim post-completion

> "First end-to-end trustless multi-token autoregressive generation on Starknet:
> N-step decode chain of Llama-3.2-1B verified on Sepolia, with each step's KV
> cache cryptographically chained to the prior step's via on-chain session
> contract, and Hades permutation correctness publicly auditable via Level-1
> cairo-prove proof matched to the chain STARK's `hades_commitment`."

Both halves are demonstrably verified; the architecture supports arbitrary
session length (capped by storage cost only, not protocol design).
