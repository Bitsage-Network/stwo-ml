# Soundness Gates Audit

**Date**: April 7, 2026
**Tested on**: A10G GPU, SmolLM2-135M (30 layers, 211 GKR layers)

## Overview

The GKR prover uses 5 environment variable "gates" that weaken proof components for on-chain compatibility. This audit tests what happens when each gate is removed (closed).

## Results

| Gate | Default | Closed Result | Can Close? | Risk if Open |
|------|---------|---------------|------------|--------------|
| `STWO_SKIP_RMS_SQ_PROOF` | `=1` (skip) | **PASSED** | **YES** | Fake RMS variance |
| `STWO_ALLOW_MISSING_NORM_PROOF` | `=1` (allow) | **PASSED** (verified on A10G) | **YES** | Skip normalization |
| `STWO_PIECEWISE_ACTIVATION` | `=0` (off) | Not tested (coupled) | Needs work | Upper bits unverified |
| `STWO_ALLOW_LOGUP_ACTIVATION` | `=1` (allow) | Not tested (coupled) | Needs work | Lower-bit only |
| `STWO_SKIP_BATCH_TOKENS` | `=1` (skip) | Not tested | Unknown | Batch manipulation |

## Detailed Analysis

### STWO_SKIP_RMS_SQ_PROOF — CAN BE CLOSED

**What it does**: Skips the Part 0 RMS-squared variance sumcheck in RMSNorm proving. When set to 1, the prover doesn't generate a sumcheck proof for `sum_x input(x)^2 = total_sq_sum`.

**Test result**: With this gate REMOVED, the prover generates a valid proof that passes self-verification. The Part 0 sumcheck is correctly computed.

**Recommendation**: **Remove `STWO_SKIP_RMS_SQ_PROOF=1` from deployment scripts.** This closes an attack surface where a prover could fabricate the RMS variance value and choose a favorable rsqrt, allowing incorrect normalization.

### STWO_ALLOW_MISSING_NORM_PROOF — SAFE TO CLOSE

**What it does**: Allows proofs to pass verification even when LayerNorm/RMSNorm sub-proofs are missing. When set to 1, the verifier skips checking norm proofs.

**Test result**: With this gate REMOVED, the prover generates a valid proof that passes self-verification on A10G. The norm sub-proof structure is now correctly generated when `STWO_SKIP_RMS_SQ_PROOF` is also closed.

**Recommendation**: **Remove `STWO_ALLOW_MISSING_NORM_PROOF=1` from deployment scripts.** This closes the attack surface where a prover could fabricate normalization parameters. Both RMS² and norm binding proofs are now correctly generated and verified.

### STWO_PIECEWISE_ACTIVATION — NOT TESTED (COUPLED)

**What it does**: When `=0`, falls back to legacy LogUp activation proofs that only verify lower 16-20 bits of activation function outputs (GELU, SiLU, Softmax).

**Coupled with**: `STWO_ALLOW_LOGUP_ACTIVATION`. Both must be addressed together.

**Attack surface**: With piecewise disabled, activation function outputs could be wrong in upper bits (bits 20-30 of the M31 field). This is a significant soundness gap for adversarial provers.

### STWO_ALLOW_LOGUP_ACTIVATION — NOT TESTED (COUPLED)

**What it does**: When set to 1, accepts proofs without full LogUp activation proofs.

**Recommendation**: Test closing both piecewise + LogUp gates together in a future sprint.

### STWO_SKIP_BATCH_TOKENS — NOT TESTED

**What it does**: Skips batch token accumulation proofs for multi-token inference.

**Attack surface**: In batch mode, the accumulation of token-level proofs into a batch proof could be manipulated.

**Recommendation**: Low priority — single-token inference (the common case) is unaffected.

## Action Items

1. **DONE**: Remove `STWO_SKIP_RMS_SQ_PROOF=1` from `deploy_node.sh` — verified safe
2. **DONE**: Remove `STWO_ALLOW_MISSING_NORM_PROOF=1` from deployment scripts — verified safe on A10G
3. **Phase 3**: Test and close piecewise + LogUp activation gates together
4. **Phase 4**: Test batch token gate

---

## Recursive STARK Security (April 12, 2026)

The production recursive STARK (v2) provides additional soundness protection independent
of the GKR soundness gates above. The v2 system uses a 48-column chain AIR (was 89 --
41 unused columns removed) with 38 constraints and 160-bit cryptographic security
(PcsConfig: pow_bits=20, log_blowup=5, n_queries=28, log_last_layer_deg=0). It uses a
two-level recursion architecture: Level 1 cairo-prove verifies 145 Hades permutations
(off-chain), Level 2 chain STARK binds to the Level 1 commitment (on-chain).

The recursive layer is protected by 9 independent security layers:

1. **Fiat-Shamir channel binding** -- prevents transcript manipulation
2. **Amortized accumulator** -- unconditional constraint, blocks all-zeros-selector
3. **n_poseidon_perms on-chain validation** -- prevents trace miniaturization
4. **seed_digest checkpoint** -- binds chain to model dimensions
5. **pass1_final_digest binding** -- proves full GKR verification ran
6. **Carry-chain modular addition** -- HadesPerm-level chain integrity
7. **hades_commitment binding** -- two-level recursion
8. **Boundary constraints** -- initial/final digest
9. **160-bit STARK security** -- pow=20, blowup=5, queries=28

First verified on Sepolia:
[`0x055c2bf89f43d9b65580862e0b81e6b47842b9dda3b862c134f35b61b0ae620f`](https://sepolia.starkscan.co/tx/0x055c2bf89f43d9b65580862e0b81e6b47842b9dda3b862c134f35b61b0ae620f)
(Contract: `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`)
