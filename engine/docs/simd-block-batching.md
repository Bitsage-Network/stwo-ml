# SIMD Block Batching

## Overview

Transformer models repeat identical blocks (layers with shared weights). Instead of proving each block independently, SIMD block batching proves N identical blocks in a **single GKR pass** by introducing a randomized block selection dimension.

For Qwen3-14B with 8 SIMD blocks, this reduces the GKR from 8 independent proofs to 1 proof with `log₂(8) = 3` extra sumcheck rounds per layer.

## Core Idea

Given N blocks with identical circuit structure, the verifier:

1. Draws random SIMD challenges `r_simd ∈ F^{log₂(N)}`
2. Computes block weights via Lagrange basis: `w_b = eq(r_simd, b)` for each block index `b`
3. The combined output is `combined[i] = Σ_b w_b · output_b[i]`

If the combined output satisfies the GKR reduction, then with overwhelming probability all individual blocks are correct (Schwartz-Zippel lemma).

## Block Weights

The SIMD weights are the multilinear Lagrange basis evaluated at the random point:

```
w_b = Π_{i=0}^{log₂(N)-1} [(1 - r_simd[i])(1 - b_i) + r_simd[i] · b_i]
```

where `b_i` is the i-th bit of block index b. For N=2:
- `w_0 = 1 - r_simd[0]`
- `w_1 = r_simd[0]`

## Shared-Weight vs Dual-Operand Matmuls

### Shared-Weight (degree-2)

When only the input A varies per block but weight B is shared:

```
Σ_b w_b · (A_b × B) = (Σ_b w_b · A_b) × B
```

**Linearity allows combining A first**, then running a standard degree-2 sumcheck:
- GPU combines: `combined_A = Σ_b w_b · MLE(A_b)` via `gpu.combine_blocks()`
- Standard matmul sumcheck: `restrict(combined_A, r_i) · restrict_col(B, r_j)`
- Same number of rounds as non-SIMD (no extra overhead)

Used for: output projection, Q/K/V projections.

### Dual-Operand (degree-3, block-extended)

When **both** A and B vary per block (per-head attention matmuls):

```
Σ_b w_b · Σ_k A_b(r_row, k) · B_b(k, r_col)
```

The linearity trick fails because `(Σ_b w_b · A_b) × (Σ_c w_c · B_c)` includes cross-terms where `b ≠ c`.

**Solution**: Block-extended 3-factor sumcheck.

Define extended MLEs of length `N × K`:

```
ext_w[b·K + k] = w_b          (block weight, replicated K times)
ext_a[b·K + k] = f_a_b[k]     (restricted A for block b)
ext_b[b·K + k] = f_b_b[k]     (restricted B for block b)
```

Then: `claim = Σ_i ext_w[i] · ext_a[i] · ext_b[i]`

This is a standard 3-factor sumcheck over `log₂(N×K)` variables — exactly `log₂(N)` extra rounds compared to non-SIMD. The degree-3 round polynomial uses `RoundPolyDeg3` with Newton interpolation at `t = 0, 1, 2, 3`.

The verification final check uses the eq evaluation:

```
running_sum == eq(r_simd, block_challenges) · final_a · final_b
```

where `block_challenges` are the first `log₂(N)` sumcheck challenges (corresponding to the block-index variables).

Used for: per-head score matmul (`Q_h × K_h^T`), per-head context matmul (`softmax_h × V_h`).

## Attention Layer Batching

### Sub-matmul Types

| Sub-matmul | A varies? | B varies? | Protocol |
|-----------|-----------|-----------|----------|
| Output: `concat × W_O` | Yes (concat differs) | No (shared weight) | Shared-weight (degree-2) |
| Context: `softmax_h × V_h` | Yes | Yes | Dual-operand (degree-3) |
| Score: `Q_h × K_h^T` | Yes | Yes | Dual-operand (degree-3) |
| V proj: `input × W_V` | Yes | No (shared weight) | Shared-weight (degree-2) |
| K proj: `input × W_K` | Yes | No (shared weight) | Shared-weight (degree-2) |
| Q proj: `input × W_Q` | Yes | No (shared weight) | Shared-weight (degree-2) |

### Score Matrix Scaling Gotcha

`score_matrices[h]` in `AttentionIntermediates` includes the `1/√d_k` scaling factor applied after `Q_h × K_h^T`. The sumcheck operates on the **raw unscaled** product. You must compute `matmul_m31(Q_h, K_h^T)` fresh for the combined output MLE — using `score_matrices[h]` directly causes an exact `√d_k` factor mismatch.

For `d_k = 64` (typical): `scale_inv = 1/8`, so the mismatch would be 8×.

### Prover Entry Point

```rust
reduce_attention_layer_simd_gpu(
    gpu, output_claim, block_executions,
    attn_weights, config, block_weights, r_simd, channel,
) → Result<(LayerProof::Attention, GKRClaim), GKRError>
```

### Verifier Dispatch

`verify_attention_reduction` accepts `r_simd: Option<&[SecureField]>`:
- `None` → non-SIMD path, all sub-proofs must be `LayerProof::MatMul`
- `Some(r_simd)` → SIMD path, per-head sub-proofs can be `MatMulDualSimd`

Sub-proof 0 (output projection) and the Q projection (last sub-proof) must always be `LayerProof::MatMul`.

## LayerNorm Non-Linearity

LayerNorm involves mean and rsqrt — non-linear operations that cause cross-terms when combining across blocks:

```
Σ_b w_b · centered_b × Σ_c w_c · rsqrt_c  ≠  Σ_b w_b · (centered_b × rsqrt_b)
```

### Combined-Product MLE Solution

Pre-compute the per-element product before combining:

```
combined_product[i] = Σ_b w_b · (centered_b[i] × rsqrt_b[i])
```

On the boolean hypercube, this equals the combined output. The eq-sumcheck proves:

```
Σ eq(r, x) · combined_product(x) · 1 = output_claim.value
```

The constant-1 MLE as the second factor means `rsqrt_final = 1` after all folds.

### Prover Flow

1. **Per-block CPU forward pass** → compute product/mean/rsqrt/input MLEs
2. **GPU combine** → 4× `gpu.combine_blocks()` calls
3. **GPU MLE evaluation** at claim point
4. **Degree-3 eq-sumcheck** over `combined_product × ones`
5. LogUp skipped → `logup_proof: None`

## SIMD Proving Flow

```rust
prove_gkr_simd_gpu(circuit, block_executions, weights, channel)
```

1. **Seed channel**: mix dimensions, block count
2. **Draw SIMD challenges**: `r_simd = channel.draw_qm31s(log₂(n_blocks))`
3. **Compute block weights**: Lagrange basis evaluation
4. **Per-block forward passes**: execute all blocks to get intermediates
5. **Walk layers** (output → input):
   - **MatMul (shared weight)**: `reduce_matmul_layer_simd_gpu()` — combine A, standard sumcheck
   - **MatMul (dual operand)**: `reduce_matmul_layer_dual_simd_gpu()` — block-extended 3-factor
   - **Activation**: `reduce_activation_layer()` with combined input/output MLEs
   - **LayerNorm**: `reduce_layernorm_layer_simd()` with combined-product approach
   - **Attention**: `reduce_attention_layer_simd_gpu()` — mixed shared/dual sub-proofs
6. **Return proof** with combined input claim

## Verification

```rust
verify_gkr_simd(circuit, proof, combined_output, channel)
```

Mirrors the prover's channel state exactly:
1. Same seeding and SIMD challenge derivation
2. Per-layer verification with `r_simd` context
3. `MatMulDualSimd` sub-proofs verified via `verify_matmul_dual_simd_reduction`
4. Final input claim checked against public input
