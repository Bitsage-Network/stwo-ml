# Fast GKR Fully Trustless On-Chain (v4): Aggregated Weight-Opening Argument

## Status

This document is the Phase-0/Phase-1 design spec for the **major protocol upgrade**
that removes the per-weight opening bottleneck from submit-ready Starknet GKR.

Current production behavior:

- v1/v2/v3 are trustless and submit-ready.
- v3 mode2 is trustless, but still carries per-weight openings.
- Large models still spend most wall time in opening proofs.

Target behavior:

- **Trustless, fully on-chain, submit-ready**.
- **No 160 independent weight openings** in calldata.
- Same structural guarantees as current GKR submit path.

---

## Problem Statement

For large transformer models, prover runtime is dominated by:

1. Per-weight MLE opening proof generation.
2. Per-weight opening serialization and on-chain replay costs.

Even with GPU-resident improvements, this remains approximately `O(M * opening_cost)`,
where `M` is number of weight claims.

We need a protocol where verification cost is dominated by a **single aggregated
binding argument** instead of `M` independent opening arguments.

---

## Security Objective

v4 must prove, on-chain and trustlessly:

1. The GKR layer walk is valid for the claimed model circuit.
2. The IO anchors are valid (output claim + input claim).
3. Every referenced weight evaluation is bound to Starknet-registered commitments.
4. The transcript is domain-separated by version/mode, preventing replay/mode confusion.

No weakening of soundness versus v3 mode2 is acceptable.

---

## Design Direction (Chosen)

### High-level

Introduce a new weight-binding mode (`mode=3`) where:

- The verifier checks one aggregated weight-binding proof object.
- The prover supplies one accumulator proof, not `M` independent opening proofs.

### Core algebraic relation

Given weight claims:

- `C_i = (weight_id_i, z_i, v_i)` for `i in [0..M-1]`
- `W` is the committed weight oracle (see commitment model below)

Define random coefficients `beta_i` from Fiat-Shamir transcript.
Define mismatch polynomial over oracle domain variable `t`:

`R(t) = Σ_i beta_i * eq(z_i, t) * (W(t) - v_i)`

If all claims are correct, then `R(t) = 0` over the full domain.

v4 proves this relation through one accumulator/sumcheck path and one final opening path.

Important: this is the design statement; formal proof obligations are listed below.

---

## Commitment Model

### Why commitment upgrade is needed

With separate per-matrix roots only, removing all independent openings is not clean.
v4 introduces a unified commitment view to support a single accumulator argument.

### v4 commitment layout

1. `weight_super_root`:
   - Commitment to a canonical global weight oracle `W`.
   - Encodes `(weight_id, local_index) -> value` in a single domain.

2. `weight_meta_root`:
   - Commitment to matrix metadata used to map `weight_id` into global offsets:
     shape, stride, quantization metadata, and canonical index range.

3. Optional compatibility:
   - Keep existing per-matrix roots for transition/debug paths.

Registration requirement:

- New registration path stores `weight_super_root` and `weight_meta_root`.

---

## On-Chain Interface (v4)

Add new entrypoint (keep v1/v2/v3 intact):

```cairo
fn verify_model_gkr_v4(
    ref self: ContractState,
    model_id: felt252,
    raw_io_data: Array<felt252>,
    circuit_depth: u32,
    num_layers: u32,
    matmul_dims: Array<u32>,
    dequantize_bits: Array<u64>,
    proof_data: Array<felt252>,
    weight_binding_mode: u32,            // must be 3
    weight_binding_data: Array<felt252>, // fixed schema header + public accum values
    weight_binding_proof: Array<felt252> // accumulator proof transcript payload
) -> bool;
```

### `weight_binding_data` (proposed schema v1)

`[schema_version, domain_tag, claims_count, weight_super_root, weight_meta_root, ...public_accumulator_terms]`

- `schema_version` starts at `1`.
- `domain_tag` is version+mode specific.
- `claims_count` must match claim extraction from `proof_data`.

### `weight_binding_proof`

Opaque felt array interpreted by v4 verifier as:

1. Aggregation sumcheck rounds
2. Final oracle opening object
3. Any required metadata/authentication payloads

No `weight_opening_proofs: Array<MleOpeningProof>` in v4.

---

## Transcript and Domain Separation

v4 transcript must mix, in order:

1. Existing GKR anchors:
   - model id
   - circuit hash/descriptor inputs
   - output claim anchor
2. v4 binding namespace:
   - `WEIGHT_BINDING_V4_DOMAIN_TAG`
   - `WEIGHT_BINDING_MODE = 3`
   - `WEIGHT_BINDING_SCHEMA_VERSION`
3. Canonical claim list hash:
   - deterministic order by layer walk / deferred ordering
   - includes `(weight_id, eval_point, expected_value)`
4. Accumulator rounds/messages

Replay safety rules:

- Different mode/version must never share transcript domain tags.
- Contract rejects mismatched mode/schema/tag triples.

---

## Verifier Algorithm (Conceptual)

1. Run existing GKR walk and reconstruct weight claim list.
2. Resolve registered `weight_super_root` + `weight_meta_root` for model.
3. Recompute canonical claim hash and derive `beta_i`.
4. Verify accumulator proof (`weight_binding_proof`) against:
   - reconstructed claims,
   - registered roots,
   - transcript challenges.
5. Accept only if accumulator check passes and all standard GKR checks pass.

---

## Soundness Obligations (Must be Proven/Checked)

1. If any claim `W(z_i) != v_i`, verifier rejects except negligible probability.
2. Canonical claim ordering uniqueness (no claim omission/permutation attacks).
3. Binding from `weight_id` to oracle location via `weight_meta_root`.
4. No cross-mode replay (v3 artifacts must fail on v4 path and vice versa).
5. Field/domain sizing for `claims_count` and accumulator degrees are enforced.

---

## Migration Plan

### Phase A: Spec + Reference

- Finalize v4 statement and transcript.
- Add Rust reference verifier for mode3 (off-chain check).
- Add deterministic test vectors.

### Phase B: Dual Proving

- Prover emits both:
  - v3 mode2 artifact (current submit-ready baseline)
  - v4 mode3 experimental artifact
- Compare acceptance and outputs off-chain.

### Phase C: Sepolia Shadow

- Deploy v4 contract entrypoint.
- Submit shadow proofs in parallel with v3 mode2.
- Enforce no-regression gate.

### Phase D: Default Switch

- `--starknet-ready` defaults to v4 mode3.
- Keep `--legacy-gkr-v3-mode2` escape hatch for rollback window.

---

## Engineering Plan (Concrete Work Items)

## 1. Cairo verifier

- File: `elo-cairo-verifier/src/verifier.cairo`
  - Add `verify_model_gkr_v4`.
  - Add mode3 parser and strict schema validation.
  - Add accumulator verifier logic.

## 2. Rust serialization + submit gates

- File: `stwo-ml/src/starknet.rs`
  - Add `build_verify_model_gkr_v4_calldata`.
  - Add mode3 artifact metadata fields.

- Files:
  - `scripts/pipeline/03_prove.sh`
  - `scripts/pipeline/04_verify_onchain.sh`
  - `scripts/pipeline/lib/paymaster_submit.mjs`
  - `scripts/pipeline/run_e2e.sh`
  - Add mode3 gating and entrypoint handling.

## 3. Prover

- File: `stwo-ml/src/gkr/prover.rs`
  - Add mode3 aggregator construction path.
  - Keep mode2 as fallback during rollout.

- File: `stwo-ml/src/gkr/verifier.rs`
  - Add reference mode3 verifier.

## 4. Tests

- `stwo-ml/src/starknet.rs` unit tests for v4 calldata layout.
- Cairo tests for:
  - happy path
  - malformed schema
  - claim count mismatch
  - transcript tag mismatch
  - tampered binding payload
- Pipeline E2E:
  - dry-run and submit preflight gates for mode3.

---

## Benchmark Targets

For Qwen3-14B-class shapes, target:

1. Weight-binding phase reduction: `>= 5x` versus v3 mode2 opening-heavy path.
2. Total proof wall-time: strong path to `< 15 min` end-to-end in submit-ready mode.
3. Calldata size reduction material enough for reliable paymaster submission.

---

## Open Questions (Must resolve before implementation freeze)

1. Exact accumulator construction choice:
   - single-oracle sumcheck accumulator,
   - or equivalent batch opening argument with explicit reduction proof.
2. Final commitment encoding details for `weight_super_root`.
3. Compatibility policy for already-registered models without super-root metadata.
4. Gas tradeoff between richer `weight_binding_data` vs proof payload size.

---

## Non-Goals

1. Marking off-chain-only RLC mode as submit-ready without new on-chain checks.
2. Removing fallback modes before v4 is production-proven.

