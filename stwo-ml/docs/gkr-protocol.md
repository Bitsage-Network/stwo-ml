# GKR Protocol — Layer-by-Layer Interactive Proof

## Overview

The GKR (Goldwasser-Kalai-Rothblum) protocol replaces per-layer independent STARK proofs with a single interactive proof that walks the computation graph from output to input. Instead of generating O(L) independent proofs for L layers, GKR produces one proof whose verification cost is proportional to the circuit depth.

For a transformer with 40 layers, each containing matmul + activation + layernorm, this eliminates ~120 independent STARK proofs and replaces them with a single GKR proof verified in O(depth) time.

## Architecture

```
 Output MLE claim
       │
       ▼
 ┌─────────────┐
 │  Layer L-1   │  ← MatMul sumcheck (log k rounds)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Layer L-2   │  ← Activation LogUp (eq-sumcheck + lookup proof)
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Layer L-3   │  ← LayerNorm (combined-product sumcheck)
 └──────┬──────┘
        ▼
       ...
        ▼
 ┌─────────────┐
 │  Layer 0     │  ← Input claim (verified against public input)
 └─────────────┘
```

## Module Layout

| File | Purpose |
|------|---------|
| `src/gkr/types.rs` | `GKRProof`, `LayerProof`, `GKRClaim`, `RoundPolyDeg3`, `LogUpProof` |
| `src/gkr/circuit.rs` | `LayeredCircuit` compiler from `ComputationGraph` |
| `src/gkr/prover.rs` | Layer reduction protocols + GPU/SIMD variants |
| `src/gkr/verifier.rs` | Fiat-Shamir transcript replay verifier |

## Layer Types and Reduction Protocols

### MatMul (degree-2 sumcheck)

For `C = A × B` with dimensions `(m × k) × (k × n)`:

The matmul identity enables efficient proof:

```
MLE_C(r_i, r_j) = Σ_k restrict(A, r_i)[k] · restrict_col(B, r_j)[k]
```

This converts a claim about C into an inner product of two restricted MLEs over the inner dimension k, proved via `log₂(k)` rounds of sumcheck. Each round produces a degree-2 polynomial `p(t) = c₀ + c₁t + c₂t²`.

**Channel protocol**: `mix(m, k, n)`, `mix(claimed_value)`, per-round `mix_poly_coeffs(c₀, c₁, c₂)` + `draw()`, then `mix(final_a)`, `mix(final_b)`.

### Add (degree-1 split)

For `C = A + B`, the MLE decomposes linearly:

```
MLE_C(r) = MLE_A(r) + MLE_B(r)
```

No sumcheck needed — the claim splits into two sub-claims that are recursively verified.

### Mul (degree-3 eq-sumcheck)

For element-wise `C = A ⊙ B`:

```
claim = Σ_x eq(r, x) · MLE_A(x) · MLE_B(x)
```

Three degree-1 factors → degree-3 univariate per round. Uses `RoundPolyDeg3` with coefficients `(c₀, c₁, c₂, c₃)` and Newton divided differences for interpolation.

### Activation (LogUp eq-sumcheck)

For `y = f(x)` where `f` is a non-linear activation (ReLU, GELU, Sigmoid):

1. **Precomputed table**: All valid `(input, output)` pairs in the M31 field
2. **Eq-sumcheck**: Proves the activation output matches the table via LogUp

```
claim = Σ_x eq(r, x) · MLE_activation(x) · MLE_multiplicity(x)
```

The `LogUpProof` contains the eq-sumcheck round polynomials, final evaluations, claimed sum, and multiplicities.

### LayerNorm (combined-product sumcheck)

LayerNorm is non-linear (involves mean + rsqrt), which causes cross-terms when batching across SIMD blocks. The solution uses a **combined-product MLE**:

```
combined_product[i] = Σ_b w_b · (centered_b[i] × rsqrt_b[i])
```

This equals the combined output on the boolean hypercube. The eq-sumcheck proves:

```
Σ eq(r, x) · combined_product(x) · 1 = output_claim.value
```

A constant-1 MLE serves as the second factor, giving `rsqrt_final = 1` after all folds.

### RMSNorm (LogUp rsqrt lookup)

RMSNorm (`y = x / sqrt(mean(x^2) + epsilon) * gamma`) is handled similarly to LayerNorm but without mean subtraction. The reciprocal square root is looked up in a precomputed table via LogUp, then the eq-sumcheck proves:

```
claim = Σ_x eq(r, x) · MLE_output(x) · MLE_rsqrt(x)
```

The `LogUpProof` follows the same structure as Activation — eq-sumcheck round polynomials, final evaluations, claimed sum, and multiplicities.

### RoPE (LogUp rotation table)

Rotary Positional Embedding applies position-dependent (cos, sin) rotations to Q/K vectors. The rotation factors are deterministic from `(seq_len, head_dim, base)`, so the verifier reconstructs the table. LogUp proves each (cos, sin) pair used in the trace exists in the table.

### Dequantize (LogUp 2D table)

For quantized models (INT4/INT8), dequantization maps quantized integer values to their M31 equivalents via a small lookup table (16 entries for INT4, 256 for INT8). LogUp proves each `(quantized_input, dequantized_output)` pair matches the table. Follows the Activation pattern with `finalize_logup_in_pairs()`.

### Attention (composed sub-matmuls)

Attention decomposes into `4 + 2H` sub-matmuls (H = num_heads):

| Index | Sub-matmul | Type | Operands |
|-------|-----------|------|----------|
| 0 | Output projection | Shared-weight | `concat × W_O` |
| 1..2H | Per-head context | Dual-operand | `softmax_h × V_h` |
| 1..2H | Per-head score | Dual-operand | `Q_h × K_h^T` (unscaled) |
| 2H+1 | V projection | Shared-weight | `input × W_V` |
| 2H+2 | K projection | Shared-weight | `input × W_K` |
| 2H+3 | Q projection | Shared-weight | `input × W_Q` |

**Important**: `score_matrices[h]` includes the `1/√d_k` scaling factor. The sumcheck must use the **raw** `Q_h × K_h^T` product (unscaled), not the stored score matrix.

See [SIMD Block Batching](simd-block-batching.md) for how dual-operand matmuls work.

## Proof Types

```rust
/// A claim: "MLE evaluated at `point` equals `value`"
pub struct GKRClaim {
    pub point: Vec<SecureField>,
    pub value: SecureField,
}

/// Per-layer proof variant
pub enum LayerProof {
    MatMul { round_polys, final_a_eval, final_b_eval },
    MatMulDualSimd { round_polys, final_a_eval, final_b_eval, n_block_vars },
    Add { lhs_eval, rhs_eval },
    Mul { round_polys: Vec<RoundPolyDeg3>, final_a_eval, final_b_eval },
    Activation { activation_type, round_polys, final_input_eval, final_output_eval, logup_proof },
    LayerNorm { round_polys, final_input_eval, final_output_eval, logup_proof },
    RMSNorm { round_polys, final_input_eval, final_output_eval, logup_proof },
    RoPE { logup_proof, input_eval, output_eval },
    Dequantize { logup_proof, input_eval, output_eval, table_commitment },
    Attention { sub_proofs: Vec<LayerProof>, sub_claim_values },
}
```

## Circuit Compilation

`LayeredCircuit::from_graph(graph)` converts a `ComputationGraph` (topologically sorted DAG of ML operations) into a layered circuit where each layer has:

- `layer_type`: Which reduction protocol to use
- `input_shape` / `output_shape`: Matrix dimensions (for MLE sizing)
- `node_id`: Back-reference to the original graph node
- `input_layers`: Predecessor layer indices

## Fiat-Shamir Transcript

The prover and verifier share a `PoseidonChannel` that absorbs all public data:

1. **Circuit metadata**: dimensions, layer count, block count
2. **Output claim**: evaluation point and value
3. **Per-layer**: dimension metadata, claimed values, round polynomial coefficients, final evaluations

The channel state must be **byte-identical** between prover and verifier at every step. Any divergence causes verification failure. This is the single most common source of bugs — the `mix_secure_field`, `mix_poly_coeffs`, and `draw_qm31` calls must appear in exactly the same order.

## Entry Points

### Proving

| Function | Description |
|----------|-------------|
| `prove_gkr(circuit, execution, weights, channel)` | CPU-only GKR proof |
| `prove_gkr_gpu(circuit, execution, weights, channel)` | GPU-accelerated (single block) |
| `prove_gkr_simd_gpu(circuit, block_executions, weights, channel)` | GPU + SIMD batching across blocks |

### Verification

| Function | Description |
|----------|-------------|
| `verify_gkr(circuit, proof, output, channel)` | Standard verification |
| `verify_gkr_with_execution(circuit, proof, execution, channel)` | Verification with intermediate checks |
| `verify_gkr_simd(circuit, proof, combined_output, channel)` | SIMD-aware verification |

## Input/Output Claim Verification

The GKR walk proves internal consistency but must be anchored to real data at both endpoints. See [`docs/input-claim-verification.md`](input-claim-verification.md) for the full design.

**Output side**: The verifier draws `r_out` from the Fiat-Shamir channel, evaluates `MLE(raw_output, r_out)` on-chain, and uses the result as the initial GKR claim.

**Input side**: After the GKR walk completes with `final_claim`, the verifier evaluates `MLE(raw_input, final_claim.point)` on-chain and asserts equality with `final_claim.value`:

```
assert!(MLE(input, final_claim.point) == final_claim.value, "INPUT_CLAIM_MISMATCH");
```

Both MLEs are constructed from raw data in calldata with power-of-2 padding and row-major layout, matching the Rust-side `pad_matrix_pow2` + `matrix_to_mle`.

## Integration with Aggregation Pipeline

GKR is **optional and additive** — the standard STARK pipeline runs first, then GKR produces an additional proof:

```rust
// Standard pipeline
let stark_proof = prove_model_aggregated_onchain(graph, input, weights);

// With GKR (additional verification layer)
let (stark_proof, gkr_proof) = prove_model_aggregated_onchain_gkr(graph, input, weights);
```

The verifier checks both proofs independently. GKR provides a second, complementary verification path.
