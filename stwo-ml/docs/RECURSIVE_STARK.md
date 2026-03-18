# Recursive STARK Composition — Phase 4A Design Document

**Author**: Bitsage Network
**Date**: March 17, 2026
**Branch**: `feat/recursive-stark`
**Status**: Implementation in progress

---

## 1. Motivation

The current ObelyZK proof pipeline produces a GKR proof for Qwen3-14B (40 layers,
160 MatMuls) that generates ~112K felts of calldata. On-chain verification requires
18 streaming transactions totaling ~56M Cairo steps — which **exceeds the 10M
Starknet sequencer limit** and forces a chunked fallback.

Recursive STARK composition compresses this into **~500 felts, 1 transaction,
~2-3M Cairo steps**. Verification cost becomes **constant regardless of model size**.

| Metric | Current (streaming GKR) | After Recursive STARK |
|--------|------------------------|-----------------------|
| Calldata | ~87K felts | ~500 felts |
| On-chain TXs | 18 streaming | **1** |
| Cairo steps | ~56M (exceeds limit) | ~2-3M |
| Gas cost | ~50 STRK | ~5 STRK |
| Prover time | 103s | 103s + ~45s recursive |
| Proof size | ~7 MB JSON | ~50 KB |

A 14B-parameter model and a 400B-parameter model have the same on-chain
verification cost.

---

## 2. Architecture

### 2.1 Current Flow

```
┌─────────────────┐     ┌────────────────┐     ┌────────────────────┐
│ prove_model()   │────>│ GKR Proof      │────>│ 18 streaming TXs   │
│ (103s on H100)  │     │ (112K felts)   │     │ Cairo GKR verifier │
└─────────────────┘     └────────────────┘     └────────────────────┘
```

### 2.2 Recursive Flow

```
┌─────────────────┐     ┌────────────────┐     ┌─────────────────────┐
│ prove_model()   │────>│ GKR Proof      │────>│ GKR Verifier        │
│ (103s on H100)  │     │ (112K felts)   │     │ as STWO AIR witness │
└─────────────────┘     └────────────────┘     └──────────┬──────────┘
                                                          │
                                                          v
                                               ┌─────────────────────┐
                                               │ STWO prove()        │
                                               │ (~45s, GPU)         │
                                               └──────────┬──────────┘
                                                          │
                                                          v
                                               ┌─────────────────────┐
                                               │ Recursive STARK     │
                                               │ (~500 felts)        │
                                               └──────────┬──────────┘
                                                          │
                                                          v
                                               ┌─────────────────────┐
                                               │ 1 TX on Starknet    │
                                               │ Generic STARK       │
                                               │ verifier (stwo-     │
                                               │ cairo-verifier)     │
                                               └─────────────────────┘
```

The key insight: **the GKR verifier becomes the witness**. We prove "I ran the GKR
verifier on this proof and it accepted" using a standard STARK. On-chain, we verify
only the STARK — no GKR-specific logic needed.

---

## 3. What Gets Arithmetized

The GKR verifier (`src/gkr/verifier.rs`) performs these operations. Each must
become an AIR constraint in the recursive circuit:

### 3.1 Fiat-Shamir Channel (Poseidon2)

**The dominant cost.** Every `mix_u64`, `mix_felt`, `draw_qm31`,
`mix_poly_coeffs` call in the Fiat-Shamir transcript is a Poseidon2-M31
permutation. The existing `poseidon2_air.rs` provides reusable constraints
(652 columns per permutation).

For a 40-layer Qwen3-14B model:
- ~160 MatMul layers x ~13 rounds each = ~2,080 sumcheck rounds
- ~80 activation/norm layers with degree-3 sumchecks
- Each sumcheck round = ~2 Poseidon permutations (mix round poly + draw challenge)
- Channel init + misc: ~500 Poseidon calls
- Aggregated binding sumcheck: ~35 rounds = ~70 permutations
- MLE opening Merkle paths: ~5K Poseidon hashes
- **Total: ~10K-15K Poseidon2 permutations**

### 3.2 Sumcheck Round Verification

For each MatMul layer round (degree-2 polynomial):
```
Constraint: c0 + (c0 + c1 + c2) == current_sum
Next sum:   next_sum = c0 + c1*r + c2*r^2       (Horner evaluation)
```

For activation/norm layers (degree-3 polynomial):
```
Constraint: c0 + (c0 + c1 + c2 + c3) == current_sum
Next sum:   next_sum = c0 + c1*r + c2*r^2 + c3*r^3
```

All arithmetic is over QM31 (4 M31 components).

### 3.3 Final Evaluation Checks

After all sumcheck rounds for a layer:
```
Constraint: current_sum == final_a_eval * final_b_eval
```

QM31 multiply + equality check.

### 3.4 MLE Evaluation

Input/output MLE evaluation at the Fiat-Shamir random point:
```
V(r) = sum_x eq(r, x) * v_x
```

For packed IO (1-row batch), this is `evaluate_mle_from_packed_1row()` —
a sequence of QM31 multiply-accumulates.

### 3.5 Weight Binding Verification

`verify_aggregated_binding()` includes its own sumcheck + MLE opening.
The sumcheck uses the same constraints as 3.2. The MLE opening is a
Poseidon Merkle path verification (sequence of Poseidon2 hash calls).

### 3.6 Merkle Path Verification

Each weight MLE opening requires verifying a Merkle authentication path.
Each path step is one Poseidon2 permutation (hash left || right child).

---

## 4. AIR Circuit Design

### 4.1 Component Decomposition

The recursive circuit is composed of 4 sub-components, following the
unified STARK pattern from `aggregation.rs`:

#### PoseidonChainComponent

Chains N Poseidon2 permutations sequentially. Each row = one permutation.
Reuses `constrain_poseidon2_permutation()` from `poseidon2_air.rs`.

Constraint: the output state of row i feeds the input state of row i+1
according to the Fiat-Shamir channel protocol (mix/draw/squeeze).

This is by far the largest component — ~15K rows x 652 columns.

#### SumcheckRoundComponent

Verifies sumcheck round polynomial consistency. Two modes:

- **Degree-2** (MatMul): row has `(c0, c1, c2, current_sum, challenge, next_sum)`.
  Constraints: `c0 + c0 + c1 + c2 == current_sum` and
  `c0 + c1*challenge + c2*challenge^2 == next_sum`.

- **Degree-3** (Activation/Norm): row has `(c0, c1, c2, c3, current_sum, challenge, next_sum)`.
  Same structure with cubic evaluation.

Each entry is a QM31 (4 M31 columns). Total: ~2,200 rows.

#### QM31ArithComponent

General-purpose QM31 arithmetic verification. Each row:
`(a, b, result, op_type)` where:
- `op_type = 0`: `result == a + b`
- `op_type = 1`: `result == a * b`

QM31 multiply decomposes into 16 M31 multiplies:
```
QM31 = (CM31_a, CM31_b) where CM31 = (M31_real, M31_imag)
CM31 multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
QM31 multiply with u^2 = 2+i: standard extension field formula
```

Constraint degree: 2 (multiply is degree 2 over trace columns).

#### PublicInputComponent

Binds the trace to public inputs:
- `circuit_hash`: Poseidon hash of the LayeredCircuit descriptor
- `io_commitment`: hash of input/output packed felts
- `weight_super_root`: Poseidon Merkle root of all weight matrices

These are read from dedicated "public input" rows and constrained to
equal the first/last Poseidon chain output at the appropriate positions.

### 4.2 Trace Layout

```
Tree 0 (preprocessed):
  - Step type selector (which component owns each row)
  - Poseidon round constants
  - Sumcheck mode flags (degree-2 vs degree-3)

Tree 1 (execution):
  - Poseidon permutation state (652 columns per perm)
  - Sumcheck round values (c0..c3, sum, challenge, next_sum)
  - QM31 arithmetic values (a, b, result)

Tree 2 (interaction / LogUp):
  - Cross-component linking (Poseidon output → sumcheck challenge)
  - Public input bindings
```

### 4.3 Trace Dimensions

| Component | Rows | Columns | Cells |
|-----------|------|---------|-------|
| PoseidonChain | ~15K | 652 | ~9.8M |
| SumcheckRound | ~2.2K | 28 | ~62K |
| QM31Arith | ~5K | 16 | ~80K |
| PublicInput | ~10 | 8 | ~80 |
| **Total** | ~22K (log_size ≈ 15) | — | ~10M |

This is within STWO's capabilities. The existing unified STARK for
40-layer Qwen3-14B handles similar trace sizes.

---

## 5. Module Structure

```
src/recursive/
├── mod.rs              Public API: prove_recursive(), verify_recursive()
├── types.rs            RecursiveProof, RecursivePublicInputs, WitnessOp
├── witness.rs          GKR verifier witness generator (instrumented channel)
├── air.rs              FrameworkEval implementations (4 sub-components)
├── prover.rs           Recursive STARK proving pipeline
├── verifier.rs         Rust-side pre-flight verification
└── tests.rs            Unit + integration + tamper tests
```

### 5.1 File Descriptions

**`types.rs`** (~100 LOC)

```rust
/// The recursive STARK proof — this replaces the 112K felt GKR calldata
pub struct RecursiveProof {
    pub stark_proof: StarkProof<Blake2sMerkleHasher>,
    pub public_inputs: RecursivePublicInputs,
}

/// Public inputs committed in the recursive circuit
pub struct RecursivePublicInputs {
    /// Poseidon hash of LayeredCircuit descriptor (layer types, shapes)
    pub circuit_hash: QM31,
    /// Poseidon hash of packed IO felts
    pub io_commitment: QM31,
    /// Poseidon Merkle root of all weight matrices
    pub weight_super_root: QM31,
    /// Model identifier (registered on-chain)
    pub model_id: FieldElement,
}

/// A single recorded operation from the verifier execution
pub enum WitnessOp {
    Poseidon2Perm { input: [M31; 16], output: [M31; 16] },
    SumcheckRound { coeffs: Vec<QM31>, claim: QM31, challenge: QM31, next_claim: QM31 },
    QM31Mul { a: QM31, b: QM31, result: QM31 },
    QM31Add { a: QM31, b: QM31, result: QM31 },
    EqualityCheck { lhs: QM31, rhs: QM31 },
}
```

**`witness.rs`** (~800 LOC)

The witness generator re-implements `verify_gkr_inner` but records every
operation instead of asserting. It uses an `InstrumentedPoseidonChannel`
that wraps the real `PoseidonChannel` and logs every permutation.

```rust
/// Wrapper around PoseidonChannel that records every Poseidon2 call
pub struct InstrumentedChannel {
    inner: PoseidonChannel,
    ops: Vec<WitnessOp>,
}

impl InstrumentedChannel {
    pub fn mix_felts(&mut self, values: &[SecureField]) { /* record + delegate */ }
    pub fn draw_felt(&mut self) -> SecureField { /* record + delegate */ }
}

/// Generate the witness trace by replaying the GKR verifier
pub fn generate_witness(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    output: &M31Matrix,
    weight_commitments: &[QM31],
) -> Result<GkrVerifierWitness, RecursiveError>
```

Key design principle: the witness generator executes the **exact same code path**
as `verify_gkr_inner`, just with an instrumented channel. This is achieved by
extracting the verifier's core logic into a generic function parameterized over
the channel type:

```rust
fn verify_gkr_generic<C: VerifierChannel>(
    channel: &mut C,
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    // ...
) -> Result<(), GKRError>
```

Both `verify_gkr_inner` (production) and `generate_witness` (recursive) call
this same function — the former with `PoseidonChannel`, the latter with
`InstrumentedChannel`.

**`air.rs`** (~1200 LOC)

Four FrameworkEval implementations:

```rust
pub struct PoseidonChainEval { pub log_n_rows: u32 }
pub struct SumcheckRoundEval { pub log_n_rows: u32, pub max_degree: u32 }
pub struct QM31ArithEval { pub log_n_rows: u32 }
pub struct PublicInputEval { pub log_n_rows: u32 }

// Each implements FrameworkEval with evaluate<E: EvalAtRow>()
```

The PoseidonChainEval reuses `constrain_poseidon2_permutation()` from
`poseidon2_air.rs` — the same function that powers the existing activation
STARK. It adds chaining constraints that link output_state[row] to
input_state[row+1].

**`prover.rs`** (~500 LOC)

Follows the established pattern from `aggregation.rs::prove_unified_stark_inner`:

```rust
pub fn prove_recursive(
    circuit: &LayeredCircuit,
    gkr_proof: &GKRProof,
    output: &M31Matrix,
    weight_commitments: &[QM31],
) -> Result<RecursiveProof, RecursiveError> {
    // 1. Generate witness
    let witness = generate_witness(circuit, gkr_proof, output, weight_commitments)?;

    // 2. Build trace columns from witness ops
    let (preprocessed, execution, interaction) = build_traces(&witness);

    // 3. Commit traces using STWO's CommitmentSchemeProver
    // 4. Call stwo::prove()
    // 5. Package into RecursiveProof
}
```

**`verifier.rs`** (~200 LOC)

Standard STWO `verify()` call for pre-flight checking before on-chain submission:

```rust
pub fn verify_recursive(proof: &RecursiveProof) -> Result<(), RecursiveError> {
    // Reconstruct AIR components from public inputs
    // Call stwo::verify()
}
```

**`tests.rs`** (~400 LOC)

- `test_recursive_1layer_roundtrip`: prove 1-layer GKR → recursive → verify
- `test_recursive_40layer_roundtrip`: full model end-to-end
- `test_recursive_tampered_proof_rejected`: flip a bit in GKR proof, recursive fails
- `test_recursive_tampered_io_rejected`: wrong IO commitment, rejected
- `test_recursive_tampered_weights_rejected`: wrong weight root, rejected
- `test_recursive_witness_matches_verifier`: differential test — witness generator
  and production verifier produce identical Poseidon call sequences
- `test_recursive_public_inputs_binding`: public inputs correctly constrained

---

## 6. Cairo Recursive Verifier

### 6.1 Contract Design

```cairo
#[starknet::contract]
mod RecursiveVerifier {
    use stwo_cairo_verifier::verifier::verify;

    #[storage]
    struct Storage {
        model_registry: Map<felt252, ModelInfo>,
        verified_proofs: Map<felt252, bool>,
    }

    #[derive(Drop)]
    struct ModelInfo {
        circuit_hash: felt252,
        weight_super_root: felt252,
        owner: ContractAddress,
    }

    #[external(v0)]
    fn verify_recursive_gkr(
        ref self: ContractState,
        stark_proof: Span<felt252>,
        model_id: felt252,
        io_commitment: felt252,
    ) -> bool {
        // 1. Look up registered model
        let model = self.model_registry.read(model_id);

        // 2. Build AIR definition for the GKR verifier circuit
        let air = build_recursive_air(model.circuit_hash, io_commitment, model.weight_super_root);

        // 3. Verify using stwo-cairo-verifier's generic verify()
        verify(air, stark_proof);

        // 4. Record verification
        let proof_hash = poseidon_hash(io_commitment, model_id);
        self.verified_proofs.write(proof_hash, true);

        // 5. Emit event
        self.emit(ProofVerified { model_id, io_commitment, proof_hash });

        true
    }

    #[external(v0)]
    fn register_model(
        ref self: ContractState,
        model_id: felt252,
        circuit_hash: felt252,
        weight_super_root: felt252,
    ) {
        self.model_registry.write(model_id, ModelInfo {
            circuit_hash, weight_super_root, owner: get_caller_address()
        });
    }

    #[external(v0)]
    fn is_verified(self: @ContractState, proof_hash: felt252) -> bool {
        self.verified_proofs.read(proof_hash)
    }
}
```

### 6.2 Key Property

The Cairo contract does NOT contain any GKR-specific logic. It verifies a
standard STARK proof using stwo-cairo-verifier. The GKR semantics are
encoded entirely in the AIR definition, which is determined by the
`circuit_hash` public input.

This means the same contract works for:
- 1-layer proofs and 40-layer proofs
- Dense models and MoE models (Phase 3B)
- Any future model architecture

---

## 7. Security Properties

### 7.1 Soundness

The recursive proof is sound if:
1. The STWO STARK proof system is sound (standard assumption)
2. The AIR circuit correctly constrains the GKR verifier execution
3. The public inputs are correctly bound (circuit_hash, io_commitment, weight_super_root)

Property (2) is the critical implementation challenge. We ensure it by:
- Extracting the verifier into a generic function shared between production and witness generation
- Running differential tests (>50 fuzz tests) comparing both paths
- Formal review of each AIR constraint against the verifier operation it represents

### 7.2 Completeness

If the GKR proof is valid, the recursive proof will be valid. This follows from:
- The witness generator succeeds whenever the production verifier succeeds
- The AIR constraints are satisfiable by the witness trace

### 7.3 Binding

A recursive proof binds to specific:
- **Model architecture**: via `circuit_hash` (Poseidon hash of layer types and shapes)
- **Model weights**: via `weight_super_root` (Poseidon Merkle root of weight matrices)
- **Inference IO**: via `io_commitment` (Poseidon hash of packed input/output)

Substituting any of these requires breaking Poseidon2 preimage resistance.

### 7.4 Zero-Knowledge

The recursive STARK proof does NOT reveal:
- The GKR proof itself (only that it was valid)
- Intermediate layer outputs
- Weight values (only the commitment)
- Inference inputs beyond the commitment

This is stronger privacy than the current streaming approach, which sends
the full GKR proof on-chain.

---

## 8. Risks and Mitigations

### Risk 1: Poseidon2 Trace Explosion

~15K permutations x 652 columns = ~9.8M cells.

**Mitigation**: At log_size 15 (32K rows), this is 652 columns x 32K rows = ~21M
field elements. STWO handles this — the existing unified STARK for 40-layer
Qwen3-14B processes similar trace sizes. GPU acceleration via the existing
`prove_unified_stark_with_gpu_pipeline` path.

### Risk 2: Fiat-Shamir Transcript Consistency

The recursive AIR must constrain the **exact same** Poseidon call sequence as
`verify_gkr_inner`. Any divergence breaks soundness.

**Mitigation**: Generic verifier function shared between production and witness
paths (see Section 5.1, witness.rs). Differential fuzz tests with >50 random
valid proofs. CI gate: witness trace hash must match verifier transcript hash.

### Risk 3: QM31 Arithmetic Correctness

QM31 multiply over the extension field (CM31 x CM31 with u^2 = 2+i) requires
16 M31 multiplies. This was previously buggy in the CUDA kernels (fixed Feb 23).

**Mitigation**: Isolated `QM31ArithComponent` with exhaustive unit tests against
`stwo::core::fields::qm31` reference implementation. Test all edge cases:
zero, one, field boundary, maximum values.

### Risk 4: Circuit Descriptor Binding

Without binding the recursive proof to a specific circuit, a prover could
substitute a simpler circuit that's easier to satisfy.

**Mitigation**: `circuit_hash` public input = Poseidon hash of the full
`LayeredCircuit` descriptor (layer types, shapes, weight node IDs). On-chain
verifier checks this against the registered model's circuit hash.

### Risk 5: Recursive Proving Overhead

Estimated ~45s additional proving time on top of the 103s GKR proof.

**Mitigation**: The 45s overhead is a one-time cost that replaces 18 streaming
TXs (~5 minutes of submission time + ~50 STRK gas). Net improvement is massive.
GPU acceleration reduces the overhead further. Future: the recursive proof can
be computed while the previous inference is serving (pipelined).

---

## 9. Implementation Timeline

| Week | Deliverable | Test Gate |
|------|-------------|-----------|
| 1-2 | `types.rs` + `witness.rs` | Witness matches production verifier for 1-layer and 40-layer |
| 2-3 | `air.rs` (4 sub-components) | AIR constraints satisfiable by witness trace |
| 3-4 | `prover.rs` + `verifier.rs` | Recursive proof verifies for 1-layer model |
| 4-5 | Cairo contract + `cairo_serde.rs` | 1 TX submission on Starknet Sepolia |
| 5-6 | Pipeline integration + benchmarks | Full 40-layer recursive proof on H100 |

---

## 10. Dependencies

- **STWO library**: `prove()`, `verify()`, `FrameworkEval`, `CommitmentSchemeProver`
- **poseidon2_air.rs**: `constrain_poseidon2_permutation()` — reused directly
- **stwo-cairo-verifier**: Generic `verify()` function + `Air` trait
- **Existing GKR verifier**: `verify_gkr_inner` refactored into generic form

No external dependencies need to be added.

---

## 11. Success Criteria

The feature is complete when:

1. `prove_recursive()` produces a valid STARK proof for a 40-layer Qwen3-14B GKR proof
2. `verify_recursive()` accepts valid proofs and rejects tampered proofs
3. A single Starknet transaction verifies the recursive proof (~2-3M steps)
4. End-to-end test passes on H100: prove → recursive → submit → verify on-chain
5. All 850+ existing tests still pass
6. Recursive proving overhead < 60s on H100
7. Paper updated with recursive STARK benchmarks
