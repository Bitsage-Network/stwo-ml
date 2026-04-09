# GKR Protocol Engine — Design Document

**Author**: Dev C (Prover Core)
**Date**: Feb 2026
**Phase**: 4 (Architecture Leap)
**Status**: DESIGN — awaiting team review before implementation

---

## 1. Motivation

The current stwo-ml proving pipeline treats each matmul as an independent
sumcheck proof and each activation as an independent LogUp STARK. For a
Qwen3-14B model with 160 matmuls, this produces ~160 independent proofs that
must each be verified separately. The on-chain cost scales as:

    O(num_matmuls × log(k))

GKR (Goldwasser-Kalai-Rothblum) replaces this with a single layered interactive
proof that walks the computation graph from output to input, reducing claims
layer-by-layer. The cost becomes:

    O(depth × log(width))

For a transformer with 40 identical decoder blocks, SIMD batching further
reduces the verifier's work by 40×, since the circuit description is shared.

**Concrete target**: prove a full Qwen3-14B forward pass in a single GKR proof
(with LogUp sub-proofs for non-linear ops), verified on-chain in ~6 batched
verification calls instead of ~160 individual ones.

---

## 2. Protocol Overview

### 2.1 Standard GKR Recap

Given a layered arithmetic circuit C with depth d and width w:

1. Verifier receives claimed output y and picks random point r₀.
2. For each layer i (output → input):
   - Verifier holds claim: Ṽᵢ(rᵢ) = vᵢ
   - Prover and verifier run sumcheck on:
     ```
     vᵢ = Σ_{x,y} [ add_i(rᵢ, x, y)·(Ṽᵢ₊₁(x) + Ṽᵢ₊₁(y))
                    + mul_i(rᵢ, x, y)·(Ṽᵢ₊₁(x) · Ṽᵢ₊₁(y)) ]
     ```
   - After sumcheck, verifier obtains claims on Ṽᵢ₊₁ at two points (x*, y*).
   - Combine into single claim via random linear combination → rᵢ₊₁.
3. At input layer: verifier checks claim against committed input MLE.

**Proof size**: d rounds × O(log w) per round = O(d log w).
**Verifier time**: O(|input| + |output| + d log w).

### 2.2 Our Adaptation: Layer-Typed GKR

Standard GKR decomposes everything into add/mul gates. This is wasteful for
matmul, which is already a sumcheck-native operation. Instead, we use
**layer-typed GKR** where each layer has a specialized reduction:

| Layer Type | Reduction Protocol | Existing Code |
|---|---|---|
| MatMul | Sumcheck over inner product (degree 2) | `compute_round_poly` GPU kernel |
| Add | Degree-1 constraint: c = a + b | `elementwise_add` + AIR |
| Mul | Degree-2 constraint: c = a × b | `elementwise_mul` + AIR |
| Activation | LogUp lookup argument | `finalize_logup_in_pairs` |
| LayerNorm | LogUp for rsqrt + linear constraints | existing STARK |
| Attention | Decomposed into sub-matmuls + softmax LogUp | `prove_attention` |

Each layer type defines:
- `reduce(claim_on_output) → claim_on_inputs` — the per-layer sumcheck
- `verify_reduction(claim_on_output, proof) → claim_on_inputs` — verifier-side

### 2.3 Claim Propagation Chain

```
claim₀ on model output
  ↓ [MatMul reduction: sumcheck over k-dimension]
claim₁ on matmul input (= previous activation output)
  ↓ [Activation reduction: LogUp lookup proof]
claim₂ on activation input (= previous matmul output)
  ↓ [MatMul reduction: sumcheck over k-dimension]
claim₃ on matmul input
  ↓ ...
claimₙ on model input → verify against committed input
```

At each step, the verifier's claim on one layer's output becomes a claim on the
previous layer's output. No intermediate layer values need to be committed.

---

## 3. Layered Circuit Representation

### 3.1 Module: `gkr/circuit.rs`

```rust
/// A layer in the GKR circuit. Each layer has a type that determines
/// its reduction protocol and a shape that determines variable counts.
pub struct CircuitLayer {
    pub layer_type: LayerType,
    pub input_shape: (usize, usize),   // (rows, cols) of input MLE
    pub output_shape: (usize, usize),  // (rows, cols) of output MLE
    pub node_id: usize,                // maps back to ComputationGraph node
}

pub enum LayerType {
    /// C[i][j] = Σ_k A[i][k] * B[k][j]
    /// Reduction: sumcheck over k-dimension (reuses existing GPU kernels).
    /// Weight index references GraphWeights for matrix B.
    MatMul {
        m: usize, k: usize, n: usize,
        weight_node_id: usize,
    },

    /// output[i] = lhs[i] + rhs[i]
    /// Reduction: degree-1 identity — no sumcheck needed, just
    /// split the claim into two sub-claims on lhs and rhs.
    Add { size: usize, lhs_layer: usize, rhs_layer: usize },

    /// output[i] = lhs[i] * rhs[i]
    /// Reduction: degree-2 check at random point.
    Mul { size: usize, lhs_layer: usize, rhs_layer: usize },

    /// output[i] = f(input[i]) where f is a lookup table.
    /// Reduction: LogUp argument (existing STARK infrastructure).
    Activation { size: usize, activation_type: ActivationType },

    /// Multi-head attention block (decomposed internally).
    /// Contains sub-layers for Q/K/V projections, scores, softmax, output.
    Attention { config: MultiHeadAttentionConfig },

    /// Layer normalization with mean/variance computation.
    LayerNorm { dim: usize },

    /// Input layer — no reduction, verified directly against commitment.
    Input,
}

/// The full layered circuit compiled from a ComputationGraph.
pub struct LayeredCircuit {
    /// Layers ordered from input (index 0) to output (index len-1).
    /// GKR walks backwards: output → input.
    pub layers: Vec<CircuitLayer>,

    /// Block boundaries for SIMD batching.
    /// Each range is a contiguous set of layers that form one
    /// identical transformer block.
    pub block_ranges: Vec<std::ops::Range<usize>>,

    /// Number of identical blocks (for SIMD multiplier).
    pub num_identical_blocks: usize,
}
```

### 3.2 Compiler: `ComputationGraph → LayeredCircuit`

```rust
impl LayeredCircuit {
    /// Compile a ComputationGraph into a layered GKR circuit.
    ///
    /// Steps:
    /// 1. Topological sort the graph.
    /// 2. Map each GraphOp to a LayerType.
    /// 3. Detect identical block boundaries via find_block_boundaries().
    /// 4. Validate: each layer's output shape matches next layer's input shape.
    pub fn from_graph(graph: &ComputationGraph) -> Self { ... }
}
```

The compiler uses `graph.find_block_boundaries()` (already implemented) to
detect repeated transformer blocks and set `num_identical_blocks`.

### 3.3 Wiring: How Layers Connect

Unlike standard GKR which uses explicit wiring predicates add_i(g,x,y) and
mul_i(g,x,y) encoded as sparse multilinear polynomials, our layer-typed approach
uses **implicit wiring**: each layer knows its input/output shapes, and the
reduction protocol maps output claims to input claims structurally.

For **MatMul** C = A × B:
- Output claim: Ṽ_C(r_i, r_j) = v
- Sumcheck reduces to: v = Σ_k Ṽ_A(r_i, k) · Ṽ_B(k, r_j)
- After sumcheck: claims on Ṽ_A(r_i, r*) and Ṽ_B(r*, r_j) at challenge point r*
- These become input claims for the previous layers

For **Add** c = a + b:
- Output claim: Ṽ_c(r) = v
- Trivially: Ṽ_a(r) + Ṽ_b(r) = v
- Verifier picks random α, reduces to: α·Ṽ_a(r) + (1-α)·Ṽ_b(r) = α·a* + (1-α)·b*
- Two sub-claims propagated to respective input layers

For **Activation** y = f(x):
- Output claim: Ṽ_y(r) = v
- LogUp proves that (x_i, y_i) entries exist in precomputed table
- Reduces to claim on Ṽ_x(r) = x*

---

## 4. GKR Prover

### 4.1 Module: `gkr/prover.rs`

```rust
/// Per-layer proof in the GKR protocol.
pub enum LayerProof {
    /// Sumcheck proof for matmul layer (reuses RoundPoly format).
    MatMul {
        round_polys: Vec<RoundPoly>,
        final_a_eval: SecureField,
        final_b_eval: SecureField,
    },

    /// For Add: no proof needed, just the split values.
    Add {
        lhs_eval: SecureField,
        rhs_eval: SecureField,
    },

    /// For Mul: product check at random point.
    Mul {
        lhs_eval: SecureField,
        rhs_eval: SecureField,
    },

    /// LogUp sub-proof for activation layer.
    Activation {
        logup_proof: ActivationLogUpProof,
    },

    /// Attention block decomposition.
    Attention {
        sub_proofs: Vec<LayerProof>,
    },
}

/// Complete GKR proof for the full model.
pub struct GKRProof {
    /// Per-layer proofs, ordered output → input.
    pub layer_proofs: Vec<LayerProof>,

    /// Final claim on input MLE (verified against commitment).
    pub input_claim: (Vec<SecureField>, SecureField),

    /// Weight commitments (Poseidon Merkle roots).
    pub weight_commitments: Vec<FieldElement>,

    /// IO commitment for binding.
    pub io_commitment: FieldElement,
}
```

### 4.2 Proving Algorithm

```rust
pub fn prove_gkr(
    circuit: &LayeredCircuit,
    execution: &GraphExecution,     // All intermediate values from forward pass
    weights: &GraphWeights,
    channel: &mut PoseidonChannel,
) -> Result<GKRProof, GKRError> {
    let d = circuit.layers.len();
    let mut layer_proofs = Vec::with_capacity(d);

    // Start with claim on output layer
    let output = &execution.output;
    let output_mle = matrix_to_mle(output);

    // Verifier draws random evaluation point
    let log_rows = output.rows.next_power_of_two().ilog2() as usize;
    let log_cols = output.cols.next_power_of_two().ilog2() as usize;
    let r_out = channel.draw_qm31s(log_rows + log_cols);
    let mut current_claim = evaluate_mle(&output_mle, &r_out);

    // Walk layers from output → input
    for layer_idx in (0..d).rev() {
        let layer = &circuit.layers[layer_idx];

        match &layer.layer_type {
            LayerType::MatMul { m, k, n, weight_node_id } => {
                // Get intermediate values (A matrix) and weight (B matrix)
                let a_matrix = get_intermediate(execution, layer.node_id);
                let b_matrix = weights.get_weight(*weight_node_id);

                // Run sumcheck reduction — reuses existing GPU kernel!
                let (proof, new_claim) = reduce_matmul_layer(
                    &current_claim, a_matrix, b_matrix,
                    *m, *k, *n, channel,
                )?;

                layer_proofs.push(LayerProof::MatMul(proof));
                current_claim = new_claim;
            }

            LayerType::Add { lhs_layer, rhs_layer, .. } => {
                let (proof, new_claim) = reduce_add_layer(
                    &current_claim, execution,
                    *lhs_layer, *rhs_layer, channel,
                )?;
                layer_proofs.push(LayerProof::Add(proof));
                current_claim = new_claim;
            }

            LayerType::Activation { activation_type, .. } => {
                let (proof, new_claim) = reduce_activation_layer(
                    &current_claim, execution,
                    layer.node_id, *activation_type, channel,
                )?;
                layer_proofs.push(LayerProof::Activation(proof));
                current_claim = new_claim;
            }

            // ... other layer types
        }
    }

    Ok(GKRProof { layer_proofs, input_claim: current_claim, ... })
}
```

### 4.3 MatMul Layer Reduction (GPU-Accelerated)

This is the critical function that reuses existing GPU sumcheck infrastructure:

```rust
fn reduce_matmul_layer(
    output_claim: &(Vec<SecureField>, SecureField),
    a: &M31Matrix,
    b: &M31Matrix,
    m: usize, k: usize, n: usize,
    channel: &mut PoseidonChannel,
) -> Result<(MatMulLayerProof, InputClaim), GKRError> {
    // The output claim is: Ṽ_C(r_i, r_j) = v
    // where C = A × B, so Ṽ_C(r_i, r_j) = Σ_k Ṽ_A(r_i, k) · Ṽ_B(k, r_j)
    //
    // This is EXACTLY the sumcheck we already implement!
    // - f_a(k) = Ṽ_A(r_i, k)  (restrict A's MLE to r_i)
    // - f_b(k) = Ṽ_B(k, r_j)  (restrict B's MLE to r_j)
    // - claim: Σ_k f_a(k) · f_b(k) = v
    //
    // Reuse: GpuSumcheckExecutor::compute_round_poly() + mle_fold()

    let gpu = GpuSumcheckExecutor::cached()?;

    // GPU fused restrict (existing kernel)
    let d_f_a = gpu.restrict_rows(a, &r_i, k)?;
    let d_f_b = gpu.restrict_cols(b, &r_j, k)?;

    // GPU sumcheck loop (existing kernels)
    let log_k = k.ilog2() as usize;
    let mut round_polys = Vec::with_capacity(log_k);
    let mut cur_n = k;

    for _ in 0..log_k {
        let mid = cur_n / 2;
        let (s0, s1, s2) = gpu.compute_round_poly(&d_f_a, &d_f_b, mid)?;

        // Lagrange interpolation → coefficients
        let c0 = s0;
        let c2 = (s2 - 2*s1 + s0) / 2;
        let c1 = s1 - s0 - c2;

        round_polys.push(RoundPoly { c0, c1, c2 });

        // Fiat-Shamir
        channel.mix_poly_coeffs(c0, c1, c2);
        let r_k = channel.draw_qm31();

        // GPU fold (existing kernel)
        d_f_a = gpu.mle_fold(&d_f_a, cur_n, &r_k)?;
        d_f_b = gpu.mle_fold(&d_f_b, cur_n, &r_k)?;
        cur_n = mid;
    }

    // Final evaluations
    let final_a = download_single(&d_f_a)?;
    let final_b = download_single(&d_f_b)?;

    // Output: claim on A at (r_i, r*) and claim on B at (r*, r_j)
    // where r* is the vector of sumcheck challenges
    // → propagates backward as input claim for previous layer
    Ok((proof, InputClaim { a_point: (r_i, r_star), a_val: final_a, ... }))
}
```

**Zero new GPU kernel code needed for matmul reduction.** All three kernels
(compute_round_poly, mle_fold, restrict_rows/cols) are reused directly.

### 4.4 Add/Mul Layer Reduction

These are trivial — no sumcheck needed:

```rust
fn reduce_add_layer(
    claim: &SecureField,  // Ṽ_c(r) = v
    r: &[SecureField],
    a_vals: &[M31],       // lhs values
    b_vals: &[M31],       // rhs values
) -> (AddLayerProof, (SecureField, SecureField)) {
    // c[i] = a[i] + b[i] for all i
    // Ṽ_c(r) = Ṽ_a(r) + Ṽ_b(r) (linearity of MLE)
    let a_eval = evaluate_mle(&to_mle(a_vals), r);
    let b_eval = evaluate_mle(&to_mle(b_vals), r);

    // Prover sends a_eval, b_eval
    // Verifier checks: a_eval + b_eval == claim
    (AddLayerProof { a_eval, b_eval }, (a_eval, b_eval))
}
```

### 4.5 Activation Layer Reduction (LogUp)

For non-linear activations, we use the existing LogUp STARK infrastructure:

```rust
fn reduce_activation_layer(
    claim: &SecureField,  // Ṽ_output(r) = v
    r: &[SecureField],
    inputs: &[M31],
    outputs: &[M31],
    table: &PrecomputedTable,
    activation_type: ActivationType,
) -> (ActivationLayerProof, SecureField) {
    // 1. Prover evaluates input MLE at r: input_eval = Ṽ_input(r)
    // 2. Prover generates LogUp proof that (input[i], output[i]) are in table
    // 3. LogUp proof binds input ↔ output relationship
    // 4. Verifier accepts claim on Ṽ_input(r) = input_eval

    // Reuse existing LogUp infrastructure:
    //   finalize_logup_in_pairs(), FrameworkComponent, etc.
    let logup_proof = prove_activation_logup(inputs, outputs, table)?;
    let input_eval = evaluate_mle(&to_mle(inputs), r);

    (ActivationLayerProof { logup_proof, input_eval }, input_eval)
}
```

---

## 5. SIMD Batching for Identical Blocks

### 5.1 Problem

A transformer has 40 identical decoder blocks. Without SIMD, the GKR prover
runs 40 independent reductions through the same circuit structure. The verifier
must process all 40 independently.

### 5.2 Solution: Randomized Block Selection

Add a SIMD dimension `r_simd` that selects which block to verify:

```
Ṽ_output(r_simd, r_layer) = Σ_block L_block(r_simd) · Ṽ_block_output(r_layer)
```

where L_block is the Lagrange basis for the block index.

The verifier draws r_simd randomly. With overwhelming probability, if any single
block was computed incorrectly, the combined evaluation will be wrong.

### 5.3 Implementation

```rust
pub struct SIMDBatchConfig {
    /// Number of identical blocks to batch.
    pub num_blocks: usize,

    /// Layer range for each block (must be identical structure).
    pub block_template: Vec<CircuitLayer>,

    /// SIMD randomness dimension (log2 of num_blocks, rounded up).
    pub simd_log_size: usize,
}

/// Prove all identical blocks in one GKR pass.
pub fn prove_gkr_simd(
    circuit: &LayeredCircuit,
    simd_config: &SIMDBatchConfig,
    executions: &[GraphExecution],  // One per block
    channel: &mut PoseidonChannel,
) -> Result<GKRProof, GKRError> {
    // 1. Draw r_simd from channel
    let r_simd = channel.draw_qm31s(simd_config.simd_log_size);

    // 2. Compute Lagrange basis for block selection
    let block_weights = compute_lagrange_basis(&r_simd);

    // 3. Combine all block outputs into single weighted MLE
    //    combined_output[i] = Σ_block weight[block] · block_output[block][i]
    let combined_output = combine_block_outputs(executions, &block_weights);

    // 4. Run standard GKR on the combined output
    //    Each layer reduction works on the combined MLE
    //    Sumcheck kernels handle the combined values transparently
    prove_gkr(circuit, &combined_execution, weights, channel)
}
```

**Key insight**: The GPU sumcheck kernels don't care whether the MLE values
come from a single execution or a weighted combination of 40 executions. The
combination happens at the MLE level, then the existing kernels process the
combined MLE identically.

### 5.4 Cost Analysis

| Metric | Without SIMD | With SIMD (40 blocks) |
|---|---|---|
| Prover per-layer sumcheck | 40 × O(k) | 1 × O(k) + O(40) combine |
| Verifier per-layer check | 40 × O(log k) | 1 × O(log k) + O(log 40) |
| Proof size | 40 × O(d_block log w) | O(d_block log w) + O(log 40) |

---

## 6. Hybrid Architecture: GKR + LogUp STARK

### 6.1 Design Principle

GKR handles **linear layers** (matmul, add, mul) natively via sumcheck.
LogUp STARK handles **non-linear layers** (activation, softmax, layernorm rsqrt)
via lookup arguments.

These compose because:
- GKR reduces output claim → input claim (MLE evaluation)
- LogUp reduces output claim → input claim (lookup verification)
- Both produce claims of the same form: "MLE at point r has value v"

### 6.2 Unified Proof Type

```rust
/// Complete model proof using hybrid GKR + LogUp architecture.
pub struct HybridModelProof {
    /// GKR component: layer-by-layer reductions for the full model.
    pub gkr_proof: GKRProof,

    /// LogUp sub-proofs embedded within GKR layers.
    /// Each activation/layernorm layer has its own LogUp proof,
    /// referenced by layer index in the GKR proof.
    pub logup_proofs: Vec<(usize, StarkProof<Blake2sHash>)>,

    /// Weight commitments (same as current pipeline).
    pub weight_commitments: Vec<FieldElement>,

    /// IO commitment (input + output binding).
    pub io_commitment: FieldElement,

    /// SIMD config (if transformer blocks were batched).
    pub simd_config: Option<SIMDBatchConfig>,
}
```

### 6.3 Verification Flow

```
1. Verifier receives HybridModelProof
2. Compute output MLE, draw random point r₀
3. For each GKR layer (output → input):
   a. If MatMul: verify sumcheck round polynomials + final eval
   b. If Add: check a_eval + b_eval == claimed_sum
   c. If Mul: check a_eval * b_eval == claimed_sum
   d. If Activation: verify embedded LogUp STARK proof
   e. Update claim for next layer
4. At input layer: verify claim against committed input MLE
5. Verify weight commitments match registered model
```

---

## 7. Module Structure

```
src/gkr/
  mod.rs          Public API: prove_gkr, verify_gkr, LayeredCircuit
  circuit.rs      LayeredCircuit, CircuitLayer, LayerType, compiler
  prover.rs       GKR prover: layer reductions, GPU dispatch
  verifier.rs     GKR verifier: Rust-side pre-flight check
  types.rs        GKRProof, LayerProof, InputClaim, GKRError
  simd.rs         SIMD batching for identical transformer blocks
```

### 7.1 Dependencies on Existing Modules

| GKR Module | Depends On | What It Uses |
|---|---|---|
| `circuit.rs` | `compiler/graph.rs` | ComputationGraph, GraphOp, topological_order |
| `prover.rs` | `gpu_sumcheck.rs` | GpuSumcheckExecutor, compute_round_poly, mle_fold |
| `prover.rs` | `components/matmul.rs` | restrict_mle, evaluate_mle, matrix_to_mle |
| `prover.rs` | `components/activation.rs` | ActivationEval, LogupTraceGenerator |
| `prover.rs` | `crypto/poseidon_channel.rs` | PoseidonChannel (Fiat-Shamir) |
| `verifier.rs` | `crypto/poseidon_channel.rs` | PoseidonChannel (transcript replay) |
| `simd.rs` | `components/matmul.rs` | compute_lagrange_basis |

### 7.2 What's New vs Reused

**Reused (zero changes)**:
- GPU sumcheck kernels (compute_round_poly, mle_fold, restrict)
- MLE operations (restrict, evaluate, matrix_to_mle)
- PoseidonChannel (Fiat-Shamir)
- LogUp STARK infrastructure (finalize_logup_in_pairs, FrameworkComponent)
- Lagrange basis computation
- Weight cache infrastructure

**New code**:
- `circuit.rs` — LayeredCircuit type + graph compiler (~400 LOC)
- `prover.rs` — GKR prover loop + per-layer reducers (~600 LOC)
- `verifier.rs` — GKR verifier (~300 LOC)
- `types.rs` — proof types (~100 LOC)
- `simd.rs` — SIMD batching (~200 LOC)
- **GPU**: wiring predicate evaluation kernel (~100 LOC CUDA) — only needed
  if we go full gate-level GKR. With layer-typed approach, no new kernels.

**Total new**: ~1600 LOC Rust, 0-100 LOC CUDA.

---

## 8. Comparison with Current Pipeline

### 8.1 Proof Size

For Qwen3-14B (40 decoder blocks, ~4 matmuls per block = 160 matmuls):

| Approach | Individual Proofs | Batch | GKR | GKR + SIMD |
|---|---|---|---|---|
| Matmul proofs | 160 | ~6 batches | 1 (integrated) | 1 (integrated) |
| Activation proofs | ~80 | 1 unified | ~80 LogUp sub | ~2 LogUp sub |
| On-chain verify calls | ~166 | ~7 | ~1 GKR + ~80 LogUp | ~1 GKR + ~2 LogUp |
| Proof size (approx) | ~160 × 2KB | ~6 × 8KB | ~50KB + LogUp | ~5KB + LogUp |

### 8.2 Prover Time

GKR does NOT reduce prover time significantly — the sumcheck work per matmul is
the same. The gains are in:
1. **Proof size**: O(d log w) vs O(m log w)
2. **Verifier time**: single integrated verification
3. **On-chain gas**: fewer verification calls
4. **SIMD**: 40× reduction for identical blocks

### 8.3 Migration Path

The GKR engine runs alongside the existing pipeline initially:
- `prove_model_aggregated_onchain` → existing pipeline (production)
- `prove_model_gkr` → new GKR pipeline (experimental)

Once GKR is validated, swap the default.

---

## 9. Implementation Plan

### Week 3: Foundation (Dev C)

1. `gkr/types.rs` — Proof types, error types
2. `gkr/circuit.rs` — LayeredCircuit + ComputationGraph compiler
3. `gkr/circuit.rs` — Unit tests: compile MLP, compile transformer block

### Week 4: Prover (Dev C + Dev B for GPU)

4. `gkr/prover.rs` — GKR prover loop (MatMul + Add + Mul reductions)
5. `gkr/prover.rs` — Activation reduction (LogUp integration)
6. `gkr/prover.rs` — Integration tests: prove 5-layer MLP, verify claims

### Week 5: SIMD + Verification (Dev C + Dev A for Cairo)

7. `gkr/simd.rs` — SIMD batching for identical blocks
8. `gkr/verifier.rs` — Rust-side GKR verifier
9. Integration: `prove_model_gkr()` in aggregation.rs
10. Dev A: Cairo GKR verifier (gkr_verifier.cairo)

### Week 5 Deliverable

End-to-end: `ComputationGraph → LayeredCircuit → GKR proof → verify`
for a 5-layer MLP and a 2-block transformer model.

---

## 10. Open Questions

1. **Attention decomposition in GKR**: Should attention be one GKR layer or
   decomposed into sub-layers (Q/K/V projections, scores, softmax, output)?
   Current leaning: decompose, since the sub-operations are different types.

2. **LogUp embedding in GKR**: Should LogUp sub-proofs be inline (part of the
   GKR layer proof) or separate (referenced by index)? Separate is cleaner
   for Cairo verification.

3. **Weight MLE commitment**: In the current pipeline, weights are committed
   per-matmul. In GKR, should there be a single aggregate weight commitment?
   Depends on the on-chain verification model.

4. **Fiat-Shamir transcript ordering**: The GKR transcript must be deterministic.
   Need to specify exact mix order for Cairo verifier compatibility. Dev A
   coordination point.

5. **Backward compatibility**: Should GKR proofs be convertible to the current
   `AggregatedModelProofOnChain` format for gradual rollout?

---

## 11. References

- Goldwasser, Kalai, Rothblum. "Delegating Computation: Interactive Proofs for Muggles" (2015, JACM)
- Thaler. "A Note on the GKR Protocol" (2013)
- zkPyTorch: Hierarchical Optimized Compiler for ZKML (2025, ePrint 2025/535)
- SUMMER: Recursive ZK Proofs for RNN Training (2025, ePrint 2025/1688)
- Expander: Polyhedra Network GKR implementation (2024-2025)
- LogUp-GKR: Improving logarithmic derivative lookups (2023, ePrint 2023/1284)
