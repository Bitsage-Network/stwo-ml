# Security Audit — stwo-ml ZKML Prover

**Date**: February 2026
**Scope**: Full prover/verifier/serialization pipeline
**Branch**: `feat/batch-sumcheck-verifier`

---

## Overview

Deep security audit of the stwo-ml ZKML proving system covering 24 findings across four severity tiers. All findings have been addressed.

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 8     | All fixed |
| High     | 5     | All fixed |
| Medium   | 6     | All fixed |
| Quantize | 6     | All fixed |

---

## Critical Findings

### C1: Activation Proof Uses Identity (No Real STARK)

**File**: `aggregation.rs`
**Impact**: Prover could claim arbitrary activation outputs without cryptographic verification.

Add, Mul, and LayerNorm operations in the unified STARK were generating forward-pass results but not producing real STARK proofs. The `ComponentProverErased<B>` trait was implemented but the actual proof generation skipped constraint evaluation.

**Fix**: All non-matmul components (Add, Mul, LayerNorm, activations) now generate genuine STARK proofs through the unified STARK framework. Each component commits execution traces to Trees 0/1/2 (preprocessed/execution/interaction) and the verifier replays the full commitment scheme.

---

### C2: Matmul Sumcheck Uses Constant Challenge

**File**: `components/matmul.rs`
**Impact**: Soundness collapse — prover could forge sumcheck proofs by precomputing against known challenges.

The Fiat-Shamir channel was not absorbing round polynomials before squeezing the next challenge. Each round's challenge was independent of the prover's messages, reducing the protocol to a fixed-point check.

**Fix**: Each sumcheck round polynomial is mixed into the Fiat-Shamir channel (Blake2s or Poseidon) before the next challenge is drawn. This applies to both `prove_matmul_sumcheck` (Blake2s) and `prove_matmul_sumcheck_onchain` (Poseidon) paths.

---

### C3: No Domain Separation in Unified STARK

**File**: `aggregation.rs`
**Impact**: Cross-component proof confusion — a valid ReLU proof could be substituted for a GELU proof.

The unified STARK combined multiple component types (activation lookups, elementwise AIR, LayerNorm) into a single commitment scheme without distinguishing which component produced which columns.

**Fix**: Each component type writes a domain separator into the Fiat-Shamir channel before its trace columns are committed. The verifier checks the same separators, preventing component proof substitution.

---

### C4: MLE Opening Proof Missing from Sumcheck

**File**: `components/matmul.rs`
**Impact**: Prover could claim arbitrary MLE evaluations at the sumcheck verification point without proving they match the committed polynomials.

The sumcheck protocol verified that round polynomials are consistent and that the final evaluation equals `a(r) * b(r)`, but never checked that `a(r)` and `b(r)` actually correspond to the committed weight/activation matrices.

**Fix**: Added `MleOpeningProof` generation after sumcheck completion. The prover commits the MLE via Merkle tree, then opens at the random evaluation point with a 14-query decommitment. The verifier checks the Merkle path before accepting the final evaluation.

---

### C5: LayerNorm Mean/Variance Not Committed

**File**: `aggregation.rs`
**Impact**: Prover could use fabricated statistics for normalization, producing any desired output while the proof checks pass.

LayerNorm computed mean and variance during proving but never committed these intermediate values. The verifier had no way to check that the normalization used correct statistics.

**Fix**: Mean and variance are Poseidon-hashed into a commitment that binds to the LayerNorm proof. The verifier recomputes the commitment from claimed values and checks it matches before verifying the normalization constraints. (PR #36, merged)

---

### C6: Softmax Normalization Bypass

**File**: `components/activation.rs`
**Impact**: Softmax outputs could be arbitrary values that don't sum to 1, breaking attention weight semantics.

The `SoftmaxNormEval` component verified individual `exp(x)` lookups but never constrained that the row sums equal 1 (or the claimed sum). A prover could output unnormalized exponentials.

**Fix**: Added a sum-check constraint to `SoftmaxNormEval` that verifies `sum(exp(x_i)) = claimed_sum` for each row. The attention component passes the claimed sum through the Fiat-Shamir channel, binding it to the proof transcript.

---

### C7: Attention QKV Weight Binding

**File**: `components/attention.rs`
**Impact**: Prover could use different weight matrices for Q, K, V projections than what was registered, computing attention over arbitrary transformations.

The attention proof contained sumcheck sub-proofs for Q, K, V, and output projections, but nothing bound these to the registered model weights. A prover could substitute any matrices.

**Fix**: Weight commitments (Poseidon hash of MLE coefficients) are included in the `AttentionProof` and checked against the registered model's weight commitment during verification.

---

### C8: GKR Input/Output Claims Not Anchored to Raw Data

**File**: `elo-cairo-verifier/src/verifier.cairo`, `stwo-ml/src/starknet.rs`
**Impact**: On-chain GKR verifier accepted proofs that were internally consistent but not bound to any specific input/output data. A malicious prover could prove inference on arbitrary data, submit the proof with different claimed I/O, and the verifier would accept.

The `verify_model_gkr` contract function received `initial_claim_point` and `initial_claim_value` directly from the prover without verification. The drawn `r_out` challenge was discarded (`_discard`), and the GKR walk's `final_claim` was also discarded (`_final_claim`). Neither endpoint was checked against actual data.

**Fix**: Replaced prover-supplied claims with on-chain MLE evaluation. The contract now:
1. Receives raw I/O data in calldata (`raw_io_data: Array<felt252>`)
2. Builds padded MLEs on-chain (power-of-2 padding, row-major layout)
3. **Output side**: Draws `r_out` from channel, evaluates `MLE(output, r_out)` on-chain, uses as initial GKR claim
4. **Input side**: After GKR walk returns `final_claim`, evaluates `MLE(input, final_claim.point)` and asserts equality with `final_claim.value`

Added `evaluate_mle()` and `pad_and_embed_m31s()` to Cairo field module. Removed 4 prover-supplied parameters (`input_rows`, `input_cols`, `initial_claim_point`, `initial_claim_value`) from the contract interface. See [`docs/input-claim-verification.md`](input-claim-verification.md) for full details.

---

## High Findings

### H1: LayerNorm Mean/Variance Commitment

**File**: `aggregation.rs`
**Impact**: Same as C5 — this was the implementation-level finding corresponding to C5's design gap.

**Fix**: Poseidon commitment of mean/variance values, verified in both Blake2s and Poseidon verification paths. Merged as PR #36 (`fix: constrain LayerNorm mean/variance via Poseidon commitment`).

---

### H2: Softmax STARK Missing in On-Chain Attention Path

**File**: `components/attention.rs`
**Impact**: On-chain attention verification skipped softmax entirely. The `AttentionProofOnChain` struct had no softmax fields, and `prove_attention_onchain()` never generated a softmax proof.

The off-chain path (`AttentionProof<H>`) correctly included `softmax_exp_proof: StarkProof<H>`, `softmax_claimed_sum`, and `softmax_log_size`. The on-chain path omitted all three.

**Fix**: Added three fields to `AttentionProofOnChain`:
```rust
pub softmax_exp_proof: StarkProof<Blake2sHash>,
pub softmax_claimed_sum: SecureField,
pub softmax_log_size: u32,
```
`prove_attention_onchain()` now collects softmax inputs/outputs per head and generates a batched LogUp STARK proof via `prove_activation_layer`. The verifier (`verify_attention_proof_onchain`) calls `verify_attention_softmax_stark()` — the same function used by the off-chain path.

---

### H3: Batch Sumcheck Lambda Commitment

**File**: `gpu_sumcheck.rs`
**Impact**: Batch sumcheck combined multiple matmul proofs with random lambdas, but the lambdas were not committed to the transcript. A malicious prover could choose lambdas after seeing the combined polynomial.

**Fix**: Lambda weights are drawn from the Fiat-Shamir channel after all individual claimed sums are mixed in, ensuring they are determined by the transcript. Committed as `245f06f`.

---

### H4: IO Commitment Binding → On-Chain Recomputation

**File**: `aggregation.rs`, `starknet.rs`, `elo-cairo-verifier/src/verifier.cairo`, `elo-cairo-verifier/src/types.cairo`
**Impact**: The model's input/output commitment was not bound to the proof transcript. A verifier could be tricked into accepting a proof for different I/O than what was actually computed. All three Cairo verification paths (`verify_model`, `verify_model_direct`, `verify_model_gkr`) blindly trusted a caller-supplied `io_commitment: felt252`.

**Fix (Phase 1)**: `compute_io_commitment()` Poseidon-hashes the flattened input and output vectors. This commitment is mixed into the Fiat-Shamir channel at the start of both proving and verification, binding the entire proof to specific I/O. Committed as `154f8cd`.

**Fix (Phase 2 — On-Chain Recomputation)**: Eliminated the trusted `io_commitment` parameter entirely. All verification paths now accept `raw_io_data: Array<felt252>` (layout: `[in_rows, in_cols, in_len, in_data..., out_rows, out_cols, out_len, out_data...]`) and recompute `Poseidon(raw_io_data)` on-chain. The GKR path additionally evaluates `MLE(output, r_out)` and `MLE(input, final_point)` on-chain, cryptographically binding the proof to the exact computation I/O without trusting any caller-supplied value.

Changes:
- `types.cairo`: `ModelProof.io_commitment` → `raw_io_data: Array<felt252>`, same for `DirectModelProof`
- `verifier.cairo`: All 3 paths validate and Poseidon-hash raw data on-chain
- `starknet.rs`: `build_starknet_proof_onchain(proof, input)` serializes raw I/O as length-prefixed array in calldata
- `cairo_serde.rs`: Updated serialization for new raw_io_data format

---

### H5: Weight Commitment Scope

**File**: `starknet.rs`
**Impact**: `compute_weight_commitment()` only hashed the first layer's weights, not the full model. A prover could register a small model but prove with different weights for deeper layers.

**Fix**: `compute_weight_commitment()` now iterates all layers in the `GraphWeights` and hashes every weight matrix into a single Poseidon commitment. The registration and verification both use this full-scope commitment. Committed as `154f8cd`.

---

## Medium Findings

### M1: Activation Type Tag in LogUp Relation

**File**: `components/activation.rs`
**Impact**: Without a type tag, a ReLU lookup entry `(x, relu(x))` could be accepted as a valid GELU entry if the values happened to match a GELU table row. Domain separation between activation types was missing.

**Fix**: The `ActivationRelation` now uses a 3-element tuple `(type_tag, input, output)` instead of 2-element `(input, output)`. Each `ActivationType` variant maps to a unique M31 tag value. The precomputed lookup table includes the tag, and the LogUp multiplicity check enforces type consistency.

---

### M2: LayerClaim Activation Type Binding

**File**: `aggregation.rs`, `cairo_serde.rs`
**Impact**: `LayerClaim` carried no record of which activation type was used. The verifier could not check that claimed activation outputs matched the graph's specified activation for that layer.

**Fix**: Added `activation_type: Option<ActivationType>` to `LayerClaim` and `act_type: ActivationType` to `ActivationLayerData`. During proving, activation layers set `activation_type: Some(layer.act_type)`. During verification, the verifier cross-references against the computation graph:
```rust
if let Some(claimed_type) = claim.activation_type {
    if graph_act_type != Some(claimed_type) {
        return Err(AggregationError::VerificationFailed(...));
    }
}
```

---

### M3: Non-On-Chain `build_starknet_proof` Has Zero Security Fields

**File**: `starknet.rs`
**Impact**: `build_starknet_proof()` (the Blake2s-channel variant) returned a `StarknetModelProof` with `matmul_calldata: Vec::new()`, `io_commitment: FieldElement::ZERO`, and `combined_calldata: Vec::new()`. Any consumer using this function instead of `build_starknet_proof_onchain()` received an incomplete proof with no cryptographic content.

**Fix**: `build_starknet_proof()` now serializes all matmul proofs via `serialize_matmul_sumcheck_proof_blake2s` (new function), uses the real `io_commitment` from the proof, and builds a full `combined_calldata` containing PCS config, IO commitment, layer chain commitment, matmul proofs, attention sub-proofs, embedding claims, and LayerNorm commitments. A new serialization function was added to `cairo_serde.rs` that handles `MatMulSumcheckProof` (Blake2s) as opposed to the existing `serialize_matmul_sumcheck_proof` which handles `MatMulSumcheckProofOnChain` (Poseidon).

---

### M4: Recursive Serialization Drops MLE Openings

**File**: `cairo_serde.rs`
**Impact**: `serialize_matmul_for_recursive()` serialized only 10 of 12 fields per matmul proof, omitting `a_opening` and `b_opening` MLE opening proofs. The recursive Cairo verifier received sumcheck round data but had no MLE opening proofs to verify that evaluations matched committed polynomials.

**Fix**: Added serialization of both MLE opening proofs:
```rust
serialize_mle_opening_proof(&proof.a_opening, output);
serialize_mle_opening_proof(&proof.b_opening, output);
```
Updated doc comments from "10-field" to "12-field format". Fixed two tests that used hardcoded byte offsets — replaced with dynamic size measurement of serialized matmul proofs.

---

### M5: Duplicate PCS Config in Serialization

**File**: `cairo_serde.rs`
**Impact**: `serialize_unified_stark_proof()` serialized PCS config twice — once explicitly as field #9, and once embedded inside `CommitmentSchemeProof` (field #11). This wasted gas on-chain and created a maintenance hazard where the two copies could diverge.

**Fix**: Removed the explicit PCS config serialization (old field #9). The single copy inside `CommitmentSchemeProof` is the authoritative source. Renumbered remaining fields (#10 → #9 interaction_pow, #11 → #10 stark_proof). Updated doc comments and test assertions to match new field layout.

---

### M6: Causal Mask Always Disabled

**File**: `aggregation.rs`, `components/attention.rs`
**Impact**: All 7 call sites for attention in `aggregation.rs` hardcoded `causal = false`. Autoregressive models (GPT, Llama, Qwen) require causal masking to prevent attention to future tokens. Without it, proving these models produces incorrect results.

**Fix**: Added `pub causal: bool` to `MultiHeadAttentionConfig`. The default constructor `::new()` sets `causal: false` for backward compatibility. Added `::new_causal()` for autoregressive models. All 7 call sites in `aggregation.rs` now read from `config.causal` / `attn_config.causal` / `layer.config.causal` instead of hardcoded `false`.

---

### Q1: Quantize Forward Pass Uses Simple Clamp (Not Real Quantization)

**File**: `aggregation.rs`, `components/quantize.rs`
**Impact**: All 6 `GraphOp::Quantize` sites (4 prover + 2 verifier) used `val.min(max_val)` (simple clamping) instead of the real `quantize_value()` formula (`round(val / scale) + zp`). The proven output did not match actual INT8/INT4 quantization. A malicious prover could produce proofs over clamped values that diverge from the claimed quantization scheme.

**Root cause**: The quantize forward pass was implemented as a simple range clamp, not the actual quantize formula. This likely originated from an early stub that was never upgraded.

**Fix**: All 6 sites now apply the real quantization formula:
```rust
let f32_val = dequantize_value(v, &direct_params); // M31 → f32
let quantized = quantize_value(f32_val, params);    // f32 → quantized M31
```
This ensures the proven output exactly matches `round(input / scale) + zero_point`, clamped to the M31 field.

---

### Q2: QuantizeEval Was 1D (Range-Check Only, No Input→Output Binding)

**File**: `components/quantize.rs`
**Impact**: `QuantizeEval` used `relation!(QuantizeRelation, 1)` — a 1-element range-check proving only that `output ∈ [0, 2^bits)`. It did **not** verify the `input → output` mapping. A malicious prover could output any in-range value for any input and the proof would verify. Compare with `DequantizeEval` which correctly used `relation!(_, 2)` to prove `(input, output)` pairs exist in a lookup table.

**Root cause**: QuantizeEval was modeled after a simple range-check instead of mirroring DequantizeEval's 2D LogUp pattern.

**Fix**: Upgraded to 2D LogUp, mirroring `DequantizeEval` exactly:
- `relation!(QuantizeRelation, 2)` — proves `(input, output)` pair membership
- 2 preprocessed columns: `quantize_table_input`, `quantize_table_output`
- 3 execution columns: `trace_input`, `trace_output`, `multiplicity`
- New `build_quantize_table(params, input_values)` builds a data-dependent lookup table mapping each observed input to `quantize_value(input, params)`
- The STARK now proves every `(input, output)` pair in the trace exists in this table

---

### Q3: Quantization Parameters Never Committed

**File**: `aggregation.rs`
**Impact**: The `scale`, `zero_point`, `bits`, and `strategy` from `QuantParams` were not part of any Poseidon commitment. A malicious prover could use different quantization parameters than those registered with the model, producing valid-looking proofs over incorrectly quantized values.

**Root cause**: No `compute_quantize_params_commitment()` existed, unlike LayerNorm which had `compute_layernorm_mean_var_commitment()`.

**Fix**: Added `compute_quantize_params_commitment()` which Poseidon-hashes all layers' quantization parameters:
```rust
// Per layer: strategy (1) + scale_lo (1) + scale_hi (1) + zero_point (1) + bits (1) = 5 felts
pub(crate) fn compute_quantize_params_commitment(layers: &[QuantizeLayerData]) -> FieldElement
```
The commitment is stored in both `AggregatedModelProofFor` and `AggregatedModelProofOnChain` as `quantize_params_commitment: FieldElement`, enabling verifiers to check that the prover used the correct quantization parameters.

---

### Q4: QuantizeLayerData Missing Input Values and Parameters

**File**: `aggregation.rs`
**Impact**: `QuantizeLayerData` only stored `values` (outputs), `multiplicities`, and `bits`. It was missing `input_values` (needed for the 2D lookup table) and the full `QuantParams` (needed for table construction and commitment). This made it impossible to build a 2D quantize table or commit to the parameters.

**Fix**: Updated `QuantizeLayerData` to include:
- `input_values: Vec<M31>` — original input values before quantization
- `params: QuantParams` — full quantization parameters (strategy, scale, zero_point, bits)
- Removed standalone `bits: u32` field (now accessible via `params.bits`)

---

### Q5: GQA Verifier Splits K/V by Wrong Head Count

**File**: `aggregation.rs` (`verify_attention_proof_blake2s`)
**Impact**: Verification always failed for GQA/MQA models. K and V heads were split by `num_heads` (Q heads) instead of `num_kv_heads`, causing dimension mismatches when `num_kv_heads < num_heads`. For MQA (1 KV head, 32 Q heads), this tried to split a `(seq, d_k)` matrix into 32 heads instead of 1, crashing with `zip_eq() reached end of one iterator before the other`.

**Root cause**: `split_heads(&inter.k, config.num_heads)` and `split_heads(&inter.v, config.num_heads)` used the Q head count. In MHA where `num_kv_heads == num_heads`, this worked by coincidence. In GQA/MQA it failed.

**Fix**: Split K/V by `num_kv_heads`, compute `group_size = num_heads / num_kv_heads`, and index KV heads via `kv_idx = h / group_size`:
```rust
let kv_heads_k = split_heads(&inter.k, config.num_kv_heads);
let kv_heads_v = split_heads(&inter.v, config.num_kv_heads);
let group_size = config.group_size();
for h in 0..config.num_heads {
    let kv_idx = h / group_size;
    let k_t = transpose_m31(&kv_heads_k[kv_idx]);
    // ...
    let v_h_p = pad_to_pow2(&kv_heads_v[kv_idx]);
}
```

---

### Q6: Trace Building Was 1D (Insufficient Columns)

**File**: `aggregation.rs`
**Impact**: The quantize trace used 1 preprocessed column (range table `[0..2^bits)`) and 2 execution columns (value, multiplicity). This was structurally insufficient for a 2D lookup — there was no column binding input values to output values. The 1D LogUp interaction trace used single-element `combine()`, which could not verify input→output mapping.

**Fix**: Updated all 3 proving paths (Blake2s, Poseidon on-chain, unified) across Trees 0/1/2:
- **Tree 0 (preprocessed)**: 1 column → 2 columns (`table_input`, `table_output`) via `build_quantize_table_columns()`
- **Tree 1 (execution)**: 2 columns → 3 columns (`trace_input`, `trace_output`, `multiplicity`) via `build_quantize_trace_columns_2d()` with padding using valid table entries
- **Tree 2 (LogUp interaction)**: Single-element `combine(&[value])` → two-element `combine(&[input, output])` for both table-side and trace-side fractions

---

## Verification

All fixes verified with:

| Test Suite | Count | Status |
|------------|-------|--------|
| Full library (`cargo test --lib`) | 442 | Pass |
| Cross-verify (`cargo test --test cross_verify`) | 5 | Pass |
| E2E Cairo (`cargo test --test e2e_cairo_verify`) | 24 | Pass |
| E2E Full Pipeline (`cargo test --test e2e_full_pipeline`) | 3 | Pass |
| Transcript Vectors (`cargo test --test transcript_vectors`) | 4 | Pass |
| Cairo verifier (`snforge test`) | 246 | Pass |

## Files Changed

| File | Findings Addressed |
|------|-------------------|
| `src/aggregation.rs` | C1, C3, C5, H1, H4, M2, M6, Q1, Q3, Q4, Q5, Q6 |
| `src/components/matmul.rs` | C2, C4 |
| `src/components/activation.rs` | C6, M1 |
| `src/components/attention.rs` | C7, H2, M6, Q5 |
| `src/components/quantize.rs` | Q1, Q2, Q6 |
| `src/gpu_sumcheck.rs` | H3 |
| `src/starknet.rs` | C8, H4, H5, M3 |
| `src/cairo_serde.rs` | M2, M3, M4, M5 |
| `elo-cairo-verifier/src/field.cairo` | C8 |
| `elo-cairo-verifier/src/verifier.cairo` | C8 |
