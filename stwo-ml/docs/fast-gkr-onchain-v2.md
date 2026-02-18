# Fast GKR Fully Trustless On-Chain (v2)

## Problem

Current `ml_gkr` has two practical modes:

1. `Sequential` weight openings (`--starknet-ready`)
2. `BatchedRlcDirectEvalV1` fast mode

`Sequential` is trustless on-chain but too large/slow for production-scale models.
`BatchedRlcDirectEvalV1` is fast but not currently submit-ready, because it removes
on-chain weight-opening binding against registered commitments.

## Why fast mode is blocked today

The current contract path `verify_model_gkr` verifies per-weight MLE openings against
registered Poseidon-Merkle roots.

Fast mode removes those openings and does direct matrix evaluation in verifier memory.
That is valid off-chain, but not trustless for Starknet unless the contract can verify
an equivalent binding argument against on-chain registered commitments.

## v2 objective

Deliver a submit-ready fast path with:

1. Same trust model as `verify_model_gkr` (no trusted prover)
2. Same structural integrity guarantees (layer walk + IO claim + weight binding)
3. Production calldata/latency profile suitable for large models

## v2 interface (draft)

Add a versioned entrypoint (keep v1 unchanged):

```cairo
fn verify_model_gkr_v2(
    ref self: ContractState,
    model_id: felt252,
    raw_io_data: Array<felt252>,
    circuit_depth: u32,
    num_layers: u32,
    matmul_dims: Array<u32>,
    dequantize_bits: Array<u64>,
    proof_data: Array<felt252>,
    weight_binding_mode: u32,
    weight_binding_data: Array<felt252>,
) -> bool;
```

`weight_binding_mode` is explicit and domain-separated in transcript:

- `0`: sequential openings (compat mode)
- `1`: batched-subchannel openings (parallel opening transcript)
- `2`: aggregated trustless binding (new fast trustless mode)

## Soundness requirements (must hold for every mode)

1. Output claim anchor: `MLE(raw_output, r_out)` from Fiat-Shamir `r_out`
2. Full GKR layer walk correctness
3. Circuit descriptor hash match
4. Input claim anchor: `MLE(raw_input, final_claim.point) == final_claim.value`
5. Weight binding against registered model commitments (or equivalent cryptographic reduction)
6. Transcript domain separation by version + mode

## Critical constraint

With independent per-matrix Merkle commitments, removing all per-matrix openings
without replacing them by an equivalent argument breaks trustlessness.

So v2 fast mode is not "toggle existing RLC on-chain"; it requires a new
weight-binding argument format and verifier logic.

## Recommended phased rollout

### Phase 1: Versioned verifier plumbing

- Introduce `verify_model_gkr_v2`
- Add `weight_binding_mode`
- Keep mode `0` behavior equivalent to v1

### Phase 2: Batched-subchannel on-chain support (low risk)

- Support mode `1` on-chain
- Same proof statement as sequential, different opening transcript derivation
- Validates mode-aware transcript handling and v2 tooling

### Phase 3: Aggregated trustless weight binding (main speed win)

- Implement mode `2` with a new on-chain-verifiable binding argument
- Remove dependency on 160 independent opening payloads
- Preserve cryptographic binding to registered commitments

### Phase 4: Coverage parity

Ensure v2 covers all needed layer tags in Cairo walk for target production models.

## Rust / pipeline changes (v2-ready)

1. Add v2 calldata builder in `stwo-ml/src/starknet.rs`
2. Emit `verify_calldata.entrypoint = verify_model_gkr_v2` when selected
3. Keep `submission_ready` gate strict per mode
4. Update submit scripts/paymaster to accept both v1 and v2 entrypoints

## Non-goals

1. Marking current off-chain RLC artifacts as submit-ready without new on-chain checks
2. Weakening soundness gates for convenience

## Success criteria

1. v2 fast mode submits and verifies on-chain
2. No trust assumptions beyond Starknet verifier correctness + crypto assumptions
3. End-to-end proving/submission profile materially better than sequential mode

## Phase 1 hardening (implemented)

1. Contract:
   - Added `verify_model_gkr_v2(...)`.
   - Initial deployment enforced `weight_binding_mode == 0` and delegated to v1 logic.
2. Rust serializers:
   - Added `build_verify_model_gkr_v2_calldata(...)`.
   - Inserts explicit `weight_binding_mode` felt into calldata.
3. Pipeline:
   - `03_prove.sh --gkr-v2` support.
   - `run_e2e.sh --gkr-v2` passthrough support.
   - Submit parsers now validate:
     - supported entrypoint (`verify_model_gkr` / `verify_model_gkr_v2`),
     - `submission_ready != false`,
    - `weight_opening_mode`/`weight_binding_mode` consistency checks.
4. Backward compatibility:
   - v1 (`verify_model_gkr`) remains unchanged.
   - v2 is opt-in until target deployments include the new entrypoint.

## Phase 2 (implemented): BatchedSubchannelV1 on-chain

1. Contract:
   - `verify_model_gkr_v2` now accepts `weight_binding_mode in {0,1}`.
   - Mode `1` verifies the same MLE openings using per-opening sub-channels
     derived from a single transcript seed.
2. Rust + serialization:
   - `WeightOpeningTranscriptMode::BatchedSubchannelV1` maps to
     `weight_binding_mode=1` in v2 calldata.
   - v1 calldata path still rejects non-sequential modes.
3. Pipeline gates:
   - Hardened submit scripts accept `verify_model_gkr_v2` with
     `weight_opening_mode=BatchedSubchannelV1` and enforce mode consistency.
4. Safety:
   - Aggregated RLC direct-eval mode remains non-submit-ready and unchanged.

## Phase 2 hardening + integration

1. Runner integration:
   - `run_e2e.sh` now supports `--gkr-v2-mode auto|sequential|batched`.
   - `auto` preserves safe defaults and prefers batched mode on GPU submit flow.
2. Submission gating:
   - v1 submit path still enforces sequential-only.
   - v2 submit path enforces `weight_opening_mode` ↔ `weight_binding_mode` consistency.
3. Paymaster preflight:
   - Submission now checks target contract ABI for requested entrypoint before TX build.
   - Missing `verify_model_gkr_v2` fails fast with actionable guidance.

## Phase 3 step-2 (implemented): v3 mode-2 trustless payload checks

1. Contract:
   - Added `verify_model_gkr_v3(...)` entrypoint.
   - Added explicit `weight_binding_data: Array<felt252>`.
   - Modes `0/1` delegate to existing trustless opening verification.
   - Mode `2` is now submit-ready with strict payload checks:
     - `weight_binding_data == [binding_digest, claim_count]`
     - digest is recomputed on-chain from resolved commitments + weight claims.
   - Full MLE opening proofs are still required in mode `2` (no soundness downgrade),
     and mode-2 openings now follow the sub-channel transcript path (same as batched mode).
2. Rust serializers:
   - Added `build_verify_model_gkr_v3_calldata(...)`.
   - v3 layout = v2 + `weight_binding_data` array.
   - For submit-ready modes:
     - mode `0/1`: `weight_binding_data=[]`
     - mode `2`: non-empty payload `[binding_digest, claim_count]`
   - Artifact now carries versioned binding metadata:
     - `weight_binding_schema_version`
     - `weight_binding_mode_id`
     - `weight_binding_data_calldata`
3. Pipeline hardening:
   - Added `--gkr-v3` and `--gkr-v3-mode2` in proving/e2e scripts.
   - Submit parsers/paymaster now accept `verify_model_gkr_v3` and enforce:
     - `weight_binding_mode` consistency with artifact mode.
     - `weight_binding_data=[]` for modes `0/1`.
     - non-empty `weight_binding_data` for mode `2`.
4. Scope:
   - This is a trustless v3 mode-2 baseline with full opening checks retained.
   - The major speed-win redesign (eliminate per-weight openings with an equivalent
     on-chain binding argument) is still pending.
