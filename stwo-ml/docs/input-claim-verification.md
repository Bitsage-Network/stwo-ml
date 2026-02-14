# Input Claim Verification — Anchoring GKR Proofs to Real Data

**Date**: February 2026
**Branch**: `feat/batch-sumcheck-verifier`

---

## Problem

The GKR protocol proves that a layered computation is internally consistent: given an output claim, each layer reduction produces a valid sub-claim for the next layer. After the full walk, the verifier holds a `final_claim` — an evaluation of the input MLE at a random point.

However, internal consistency alone is insufficient. Without checking that:

1. The **output claim** matches actual raw output data
2. The **input claim** matches actual raw input data

...the proof only shows "some computation with some I/O is internally consistent," not "this specific computation on this specific data was performed correctly."

### The Soundness Gap (Before)

```
                     Prover-supplied
                     (not verified)
                          │
                          ▼
    Output MLE claim ──► GKR walk ──► final_claim
         │                                │
         │  r_out drawn from              │  final_claim.point
         │  channel but DISCARDED         │  and .value DISCARDED
         │  (_discard)                    │  (_final_claim)
         ▼                                ▼
    ❌ Not anchored                  ❌ Not anchored
       to raw output                    to raw input
```

A malicious prover could:
- Construct a valid GKR proof for arbitrary I/O
- Submit it with different raw data than what was actually computed
- The verifier would accept it because it never checked the endpoints

---

## Solution

Both ends of the GKR walk are now anchored to raw data passed in calldata. The verifier evaluates MLEs on-chain and asserts equality with the claims from the GKR walk.

### Architecture (After)

```
    Raw I/O in calldata
         │
    ┌────┴────┐
    ▼         ▼
  Output    Input
  parsing   parsing
    │         │
    ▼         ▼
  Pad to    Pad to
  pow2      pow2
    │         │
    ▼         ▼
  Build     Build
  MLE       MLE
    │         │
    ▼         ▼
  Draw      GKR walk
  r_out     returns
  from ch   final_claim
    │         │
    ▼         ▼
  Evaluate  Evaluate
  MLE at    MLE at
  r_out     final_claim
    │       .point
    ▼         │
  output_     ▼
  value     input_
    │       value
    ▼         │
  Use as      ▼
  initial   Assert ==
  claim     final_claim
  for GKR   .value
  walk        │
    │         ▼
    └──► ✅ Proof anchored to real data
```

### Output Side (A-side)

1. Parse raw output data from calldata: `[out_rows, out_cols, out_len, data...]`
2. Build padded MLE: zero-pad to `next_pow2(rows) × next_pow2(cols)`, row-major layout
3. Draw `r_out` from Fiat-Shamir channel (was previously drawn but discarded)
4. Evaluate `MLE(output, r_out)` on-chain → `output_value`
5. Mix `output_value` into channel
6. Use `GKRClaim { point: r_out, value: output_value }` as the initial claim for the GKR walk

### Input Side (B-side)

1. GKR walk completes → returns `final_claim: GKRClaim`
2. Parse raw input data from calldata: `[in_rows, in_cols, in_len, data...]`
3. Build padded input MLE: same padding/layout as output
4. Evaluate `MLE(input, final_claim.point)` on-chain → `input_value`
5. **Assert `input_value == final_claim.value`** (error: `INPUT_CLAIM_MISMATCH`)

---

## On-Chain MLE Evaluation

The `evaluate_mle` function in Cairo implements iterative folding over the boolean hypercube. For `n` evaluation points and `2^n` MLE coefficients, it runs in `O(2^n)` time and `O(2^n)` memory.

### Algorithm

```
Input:  evals[0..2^n]  — MLE values on {0,1}^n (row-major)
        point[0..n]    — evaluation point (QM31 elements)
Output: MLE(evals, point)

current = evals
for var_idx in 0..n:
    r = point[var_idx]
    mid = |current| / 2
    next = []
    for j in 0..mid:
        lo = current[j]         // var_idx = 0
        hi = current[j + mid]   // var_idx = 1
        next[j] = lo + r * (hi - lo)   // linear interpolation
    current = next
return current[0]
```

This matches the Rust-side `evaluate_mle` in `components/matmul.rs:118-132`.

### Power-of-2 Padding

Raw matrix dimensions are padded to the next power of 2 before MLE construction:
- `(3, 5)` → `(4, 8)` → 32 entries
- `(1, 4)` → `(1, 4)` → 4 entries (already power of 2)

Padding entries are zero-filled. Row-major layout: `A[i][j]` at index `i * padded_cols + j`.

This matches `pad_matrix_pow2` + `matrix_to_mle` on the Rust side.

---

## Contract Interface Change

### Before

```cairo
fn verify_model_gkr(
    ref self: TContractState,
    model_id: felt252,
    io_commitment: felt252,        // Poseidon hash (opaque)
    num_layers: u32,
    input_rows: u64,               // Prover-supplied
    input_cols: u64,               // Prover-supplied
    matmul_dims: Array<u32>,
    dequantize_bits: Array<u64>,
    initial_claim_point: Array<QM31>,  // Prover-supplied
    initial_claim_value: QM31,         // Prover-supplied
    proof_data: Array<felt252>,
    weight_commitments: Array<felt252>,
) -> bool;
```

### After

```cairo
fn verify_model_gkr(
    ref self: TContractState,
    model_id: felt252,
    raw_io_data: Array<felt252>,   // Full raw I/O (verifier evaluates MLEs)
    num_layers: u32,
    matmul_dims: Array<u32>,
    dequantize_bits: Array<u64>,
    proof_data: Array<felt252>,
    weight_commitments: Array<felt252>,
) -> bool;
```

**Removed parameters** (now computed on-chain from `raw_io_data`):
- `io_commitment` — replaced by raw data verification
- `input_rows`, `input_cols` — parsed from `raw_io_data` header
- `initial_claim_point` — drawn from channel on-chain
- `initial_claim_value` — evaluated from output MLE on-chain

### raw_io_data Format

Serialized by `serialize_raw_io()` in `cairo_serde.rs`:

```
[in_rows, in_cols, in_len, in_data[0..in_len],
 out_rows, out_cols, out_len, out_data[0..out_len]]
```

Each value is a `felt252` encoding an M31 field element.

---

## Calldata Layout (Rust → Cairo)

`build_verify_model_gkr_calldata()` in `starknet.rs` produces:

| Index | Field | Type |
|-------|-------|------|
| 0 | `model_id` | `felt252` |
| 1 | `raw_io_data.len()` | `u32` |
| 2..2+N | `raw_io_data` | `felt252[]` |
| 2+N | `num_layers` | `u32` |
| 3+N | `matmul_dims.len()` | `u32` |
| ... | `matmul_dims` | `u32[]` |
| ... | `dequantize_bits.len()` | `u32` |
| ... | `dequantize_bits` | `u64[]` |
| ... | `proof_data.len()` | `u32` |
| ... | `proof_data` | `felt252[]` |
| ... | `weight_commitments.len()` | `u32` |
| ... | `weight_commitments` | `felt252[]` |

---

## Files Changed

| File | Change |
|------|--------|
| `elo-cairo-verifier/src/field.cairo` | Added `evaluate_mle()`, `pad_and_embed_m31s()` |
| `elo-cairo-verifier/src/verifier.cairo` | New contract interface + on-chain MLE evaluation |
| `elo-cairo-verifier/src/model_verifier.cairo` | Return type updated to `(GKRClaim, Array<WeightClaimData>)` |
| `elo-cairo-verifier/tests/test_gkr.cairo` | 6 new MLE evaluation tests |
| `elo-cairo-verifier/tests/test_sp3_cross_language.cairo` | Updated for new return type |
| `stwo-ml/src/starknet.rs` | `build_verify_model_gkr_calldata` takes `raw_io_data` |
| `stwo-ml/tests/e2e_cairo_verify.rs` | All 6 call sites + structural assertions updated |

## Test Coverage

### Cairo MLE Tests (`test_gkr.cairo`)

| Test | Description |
|------|-------------|
| `test_evaluate_mle_single_variable` | 1-var MLE: f(0)=3, f(1)=7 at r=0 and r=1 |
| `test_evaluate_mle_two_variables` | 2-var MLE at all 4 boolean corners |
| `test_evaluate_mle_matches_fold_mle_eval` | Consistency with existing `fold_mle_eval` |
| `test_evaluate_mle_with_qm31_point` | Non-trivial QM31 evaluation point |
| `test_pad_and_embed_m31s` | M31→QM31 embedding + zero-padding |
| `test_evaluate_mle_padded_matrix` | Padded matrix MLE at (0,0) |

### Rust E2E Tests (`e2e_cairo_verify.rs`)

All 24 tests pass, including 6 that exercise `build_verify_model_gkr_calldata` with `raw_io_data`:

| Test | Model Type |
|------|------------|
| `test_verify_model_gkr_calldata_matmul_only` | Single matmul (1×4 × 4×2) |
| `test_verify_model_gkr_calldata_mlp_relu` | MLP: MatMul→ReLU→MatMul |
| `test_d7_export_onchain_calldata` | Matmul-only deployment artifact |
| `test_d9_export_mlp_onchain_calldata` | MLP deployment artifact |
| `test_d10_export_layernorm_onchain_calldata` | MatMul→LayerNorm→MatMul |
| `test_d11_export_residual_onchain_calldata` | Residual connection (Add layer) |

---

## Security Impact

This fix addresses a **Critical** soundness gap. Before this change, the on-chain verifier accepted proofs that were internally consistent but not bound to any specific input/output data. A malicious prover could:

1. Prove inference on arbitrary data
2. Submit the proof with different claimed I/O
3. The verifier would accept because it never checked the endpoints

After this fix, both endpoints of the GKR walk are cryptographically bound to the raw data in calldata. The verifier independently evaluates the MLE at the Fiat-Shamir challenge points and asserts equality with the GKR claims.

### Remaining Work

- **Weight MLE opening verification** (`verifier.cairo:1216`): Weight claims are collected but not yet verified against registered Merkle roots on-chain. The `TODO` marks where `weight_opening_data` should be added as a contract parameter.
