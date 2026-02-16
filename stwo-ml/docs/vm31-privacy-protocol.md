# VM31 Privacy Protocol

Design document for the shielded transaction system built on STWO Circle STARKs over M31.

## Overview

VM31 implements a UTXO-based shielded pool where:

1. **Deposit** — Convert public funds to a shielded note (commitment added to Merkle tree)
2. **Withdraw** — Destroy a shielded note and release public funds (nullifier published)
3. **Spend** — Transfer privately: consume 2 input notes, produce 2 output notes

All operations are proved with STWO STARK proofs. The verifier sees **only public inputs** — all private data (spending keys, blinding factors, note contents, Merkle paths) is hidden by the FRI protocol's zero-knowledge property.

## Cryptographic Primitives

### Poseidon2-M31 (`crypto/poseidon2_m31.rs`)

Native Poseidon2 hash function operating directly over the Mersenne-31 field (`p = 2^31 - 1`). No field conversions — M31 in, M31 out.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| State width (t) | 16 | Matches Plonky3/STWO standard |
| Rate | 8 | 128-bit capacity security |
| Capacity | 8 | |
| S-box | x^5 | gcd(5, p-1) = 1 → invertible |
| Full rounds (R_f) | 8 (4+4) | |
| Partial rounds (R_p) | 14 | |
| External matrix | circ(2M4, M4, M4, M4) | HorizenLabs standard |
| Internal diagonal | Plonky3 validated | `[-2, 1, 2, 4, 8, ..., 65536]` |
| Round constants | 142 | xorshift64 PRNG seeded with "Poseidon2-M31" |

**Hash output**: 8 M31 elements (248 bits, ~124-bit collision resistance).

**Compression**: `poseidon2_compress(left, right)` — 16 M31 in (8+8), 8 M31 out. Overwrites the input state and returns the first 8 elements of the permutation output.

**Key implementation detail**: The S-box inverse `d_inv = modinv(5, p-1) = 1717986917`. This is used in security analysis but not in the proving circuits (which only compute x^5 forward).

### Note Structure (`crypto/commitment.rs`)

```rust
pub struct Note {
    pub owner_pubkey: [M31; 4],  // Poseidon2 hash of spending key
    pub asset_id: M31,           // 0 = STRK, 1 = ETH, ...
    pub amount_lo: M31,          // Low 31-bit limb
    pub amount_hi: M31,          // High 31-bit limb (total = lo + hi * 2^31)
    pub blinding: [M31; 4],      // 124-bit hiding factor
}
```

**Amount encoding**: Values up to ~2^62 are encoded as two M31 limbs. Each limb is further decomposed into two uint16 sub-limbs for range checking: `sub_lo + sub_hi * 65536 = limb_value`.

### Commitment Scheme

```
NoteCommitment = Poseidon2(pk[0..3] || asset_id || amount_lo || amount_hi || blinding[0..3])
```

11 M31 elements absorbed via sponge construction:
- Rate = 8, so first chunk absorbs elements 0..7, second chunk absorbs elements 8..10
- Domain separation tag at position 8 of the first permutation input
- Output: first 8 elements of the final permutation state

### Nullifier

```
Nullifier = Poseidon2(spending_key[0..3] || commitment[0..7])
```

12 M31 elements absorbed:
- First chunk: `sk[0..3] || commitment[0..3]` (8 elements at rate positions)
- Second chunk: `commitment[4..7]` (4 elements at rate positions 0..3)
- Sponge chain preserves positions 4..15 from previous permutation output

### Key Derivation

```
pubkey       = Poseidon2("spend" || sk[0..3])[0..3]    // 4 M31 elements
viewing_key  = Poseidon2("view"  || sk[0..3])[0..3]    // 4 M31 elements
```

Domain separation via ASCII tags (`"spend"` = `0x7370656E64`, `"view"` = `0x76696577`).

### Merkle Tree (`crypto/merkle_m31.rs`)

Append-only binary Merkle tree using Poseidon2-M31 compression.

| Property | Value |
|----------|-------|
| Depth | 20 |
| Capacity | 1,048,576 notes |
| Leaf | 8 M31 elements (note commitment) |
| Internal node | `poseidon2_compress(left_child, right_child)` |
| Empty leaf | `[0; 8]` |
| Proof | 20 sibling digests + leaf index |

### Encryption (`crypto/encryption.rs`)

Poseidon2-M31 counter-mode encryption for note memos.

- **Key derivation**: `key = Poseidon2("kdf" || shared_secret)[0..8]`
- **Keystream**: `block_i = Poseidon2_perm(key || counter_i || "encr" || padding)[0..8]`
- **Encrypt**: `ciphertext[i] = plaintext[i] + keystream[i]` (M31 addition)
- **Checksum**: Last block includes a hash of the plaintext for integrity verification

## Transaction Circuits (Phase 3)

Phase 3 implements **computational integrity** — the verifier can see the full execution trace and confirms it was computed correctly. This is the foundation that Phase 4 wraps with zero-knowledge.

### Shared Helpers (`circuits/helpers.rs`)

Permutation-recording functions that capture all intermediate Poseidon2 states:

| Helper | Description | Permutations |
|--------|-------------|-------------|
| `record_hash_permutations` | Sponge hash of N M31 elements | ceil(N/rate) |
| `record_compress_permutation` | Merkle node compression | 1 |
| `record_merkle_permutations` | Full Merkle path verification | depth (20) |
| `record_ownership_permutations` | Key derivation proof | 1 |

### Deposit Circuit (`circuits/deposit.rs`)

**Operation**: Public amount → shielded note commitment.

**Witness**: spending key, note (pubkey, asset, amount, blinding)

**Public outputs**: commitment (8 M31 elements)

**Permutations**: 2 (commitment hash via sponge)

**Proof structure**: `Poseidon2BatchProof` (batch of 2 permutations) + `RangeCheckProof` (4 sub-limbs)

### Withdraw Circuit (`circuits/withdraw.rs`)

**Operation**: Shielded note → public amount.

**Witness**: spending key, note, Merkle path

**Public outputs**: Merkle root, nullifier, amount, asset_id

**Permutations**: 25 total
- 1 ownership derivation (key → pubkey)
- 2 commitment hash (sponge)
- 2 nullifier hash (sponge)
- 20 Merkle path compress (one per tree level)

**Proof structure**: `Poseidon2BatchProof` (batch of 32, padded) + `RangeCheckProof` (4 sub-limbs)

### Spend Circuit (`circuits/spend.rs`)

**Operation**: 2-in/2-out private transfer.

**Witness**: 2 spending keys, 2 input notes, 2 Merkle paths, 2 output notes

**Public outputs**: Merkle root, 2 nullifiers, 2 output commitments

**Permutations**: 54 total
- Per input (×2): 1 ownership + 2 commitment + 2 nullifier + 20 Merkle = 25
- Per output (×2): 2 commitment hash = 2

**Balance conservation**: Carry witness `c ∈ {-1, 0, 1}` handles cross-limb overflow:
- `Σ lo_in - Σ lo_out + c ≡ 0 (mod p)` — lo limb balance with carry
- `Σ hi_in - Σ hi_out - c ≡ 0 (mod p)` — hi limb balance with carry
- `c = carry_pos - carry_neg` where both are constrained binary and mutually exclusive

**Asset consistency**: All 4 notes must have the same `asset_id`.

**Proof structure**: `Poseidon2BatchProof` (batch of 64) + `RangeCheckProof` (8 sub-limbs) + `SpendExecution`

## Transaction STARKs (Phase 4)

Phase 4 wraps each transaction circuit into a **STWO STARK proof** where the verifier sees only public inputs. The full execution trace (all private data) is committed via FRI and hidden by the protocol's zero-knowledge property.

### Architecture

Each transaction type is a **single `FrameworkEval`** component where each row = 1 complete transaction. All Poseidon2 permutations are unrolled within the row — no cross-row constraints.

**Why single-row?** STWO handles wide traces efficiently (the ML components already use thousands of columns). A single row eliminates the need for cross-row constraint plumbing while keeping the architecture simple.

### Poseidon2 AIR Constraints (`components/poseidon2_air.rs`)

Shared constraint functions reused by all three transaction STARKs. Each Poseidon2 permutation occupies **652 execution trace columns**:

| Column Group | Count | Description |
|---|---|---|
| `state[0..22][0..15]` | 23 × 16 = 368 | State before each round + output state |
| `sq` (full rounds) | 8 × 16 = 128 | x^2 auxiliary for S-box (all 16 elements) |
| `quad` (full rounds) | 8 × 16 = 128 | x^4 auxiliary for S-box (all 16 elements) |
| `sq` (partial rounds) | 14 × 1 = 14 | x^2 for element 0 only |
| `quad` (partial rounds) | 14 × 1 = 14 | x^4 for element 0 only |

**S-box decomposition** (x^5 → degree 2):

The Poseidon2 S-box computes x^5. Direct constraint would be degree 5, which exceeds STWO's degree-2 limit. Decomposed with auxiliary columns:

```
sq   = (state + round_constant)^2          constraint: sq - (state + rc)^2 = 0       (degree 2)
quad = sq^2                                 constraint: quad - sq^2 = 0               (degree 2)
sbox_output = quad * (state + round_constant)   // used inline in matrix constraint   (degree 2)
```

Round constants are embedded as M31 literals in constraint expressions — not preprocessed columns.

**Full round constraints** (rounds 0..3 and 18..21):
- S-box applied to all 16 state elements → 16 sq columns + 16 quad columns
- External matrix: `circ(2M4, M4, M4, M4)` applied to S-box outputs
- 16 constraints per round linking S-box outputs through the matrix to next state

**Partial round constraints** (rounds 4..17):
- S-box applied to element 0 only → 1 sq column + 1 quad column
- Elements 1..15 pass through unchanged
- Internal matrix: `(J + diag(INTERNAL_DIAG))` applied to [sbox_out[0], state[1..15]]

### STWO Degree Constraint

**Critical**: STWO requires `max_constraint_log_degree_bound = log_size + 1` when using the default `PcsConfig` (`log_blowup_factor = 1`). This means all constraints must be degree ≤ 2. Higher degrees cause FRI decommitment failures because the composition polynomial's committed domain doesn't match the trace domain.

This constraint drove several architectural decisions:
- S-box decomposition with auxiliary columns (degree 5 → degree 2)
- Merkle chain constraints without `is_real` selector (see below)

### Merkle Chain Constraints

Merkle path verification requires chaining consecutive compress operations:

```
level 0: compress(leaf, sibling_0) → hash_0
level 1: compress(hash_0, sibling_1) → hash_1  (or compress(sibling_1, hash_0))
...
```

The position bit (left/right child) determines which half of the compress input holds the previous output. The constraint must enforce "previous output appears in either the left OR right half" — but the naive approach `is_real * (left - prev) * (right - prev) = 0` is degree 3.

**Solution**: Per-element multiplicative constraint without `is_real`:

```
(input[j] - prev_output[j]) * (input[j + RATE] - prev_output[j]) = 0
```

This is degree 2 and works on all rows — including padding — because padding rows use **chained traces** where each padding permutation feeds its output into the next padding permutation's left half. The constraint `(left - prev) = 0` is trivially satisfied on padding rows.

The `compute_merkle_chain_padding` function generates these chained dummy traces.

### Deposit STARK (`circuits/stark_deposit.rs`)

| Property | Value |
|----------|-------|
| Permutations | 2 (commitment hash) |
| Trace width | ~1,372 columns |
| Log table size | 4 (16 rows) |
| Public inputs | commitment[8], amount_lo, amount_hi, asset_id |

Constraints:
1. Poseidon2 round constraints (×2 permutations)
2. Domain separation tag binding
3. Sponge chain (perm 0 output → perm 1 input)
4. Commitment output binding to public input
5. Amount/asset input binding
6. Range check: 4 sub-limbs × 16 bits = 64 bit columns

### Withdraw STARK (`circuits/stark_withdraw.rs`)

| Property | Value |
|----------|-------|
| Permutations | 32 (25 real + 7 padding) |
| Trace width | ~20,932 columns |
| Log table size | 4 (16 rows) |
| Public inputs | merkle_root[8], nullifier[8], amount_lo, amount_hi, asset_id |

Permutation layout:
| Perms | Operation |
|-------|-----------|
| 0 | Ownership derivation (sk → pk) |
| 1-2 | Commitment hash (sponge) |
| 3-4 | Nullifier hash (sponge) |
| 5-24 | Merkle path compress (20 levels) |
| 25-31 | Padding (chained dummy) |

Wiring constraints (all × `is_real`):
1. Ownership: `input[0] = DOMAIN_SPEND`, `input[5..7] = 0`, `input[8] = 5`, `input[9..15] = 0`
2. Ownership output[0..3] = commitment input pk[0..3]
3. Commitment sponge chain (perm 1 → perm 2, positions 3..15)
4. Nullifier sk = ownership sk: `nul.input[0..3] = own.input[1..4]`
5. Commitment output[0..3] → nullifier input[4..7]
6. Nullifier sponge chain (perm 3 → perm 4, positions 4..15)
7. Nullifier chunk 2 absorption: `nul2.input[0..3] - nul1.output[0..3] = commitment.output[4..7]`
8. Nullifier output = public nullifier
9. Merkle leaf = commitment (degree-2 left/right selector)
10. Merkle chain (20 levels, degree-2 constraints)
11. Merkle root = public merkle_root
12. Amount/asset binding + range check

### Spend STARK (`circuits/stark_spend.rs`)

| Property | Value |
|----------|-------|
| Permutations | 64 (54 real + 10 padding) |
| Trace width | ~41,867 columns |
| Log table size | 4 (16 rows) |
| Public inputs | merkle_root[8], nullifiers[2][8], output_commitments[2][8] |

Permutation layout:
| Perms | Operation |
|-------|-----------|
| 0 | Input 0: ownership |
| 1-2 | Input 0: commitment hash |
| 3-4 | Input 0: nullifier hash |
| 5-24 | Input 0: Merkle path (20 levels) |
| 25 | Input 1: ownership |
| 26-27 | Input 1: commitment hash |
| 28-29 | Input 1: nullifier hash |
| 30-49 | Input 1: Merkle path (20 levels) |
| 50-51 | Output 0: commitment hash |
| 52-53 | Output 1: commitment hash |
| 54-63 | Padding (chained dummy) |

Additional constraints:
- **Balance with carry**: 3 carry columns (carry, carry_pos, carry_neg) + 6 constraints:
  - `carry_pos ∈ {0,1}`, `carry_neg ∈ {0,1}`, `carry_pos × carry_neg = 0` (mutual exclusion)
  - `carry = carry_pos - carry_neg` (reconstruction, c ∈ {-1, 0, 1})
  - `is_real × (Σ lo_in - Σ lo_out + carry) = 0`
  - `is_real × (Σ hi_in - Σ hi_out - carry) = 0`
- **Ownership**: `input[0] = DOMAIN_SPEND`, `input[5..7] = 0`, `input[8] = 5`, `input[9..15] = 0`
- **Nullifier sk = ownership sk**: `nul.input[0..3] = own.input[1..4]` (per input)
- **Nullifier chunk 2 absorption**: `nul2.input[0..3] - nul1.output[0..3] = commitment.output[4..7]`
- **Asset consistency**: All 4 notes have the same asset_id
- Range check: 8 sub-limbs × 16 bits = 128 bit columns

### Carry Witness for Cross-Limb Balance (Spend STARK)

Amounts are encoded as `amount = lo + hi * 2^31` where each limb is in `[0, p-1]` with `p = 2^31 - 1`. When lo-limb sums overflow (e.g., inputs `(lo=2B, hi=0) + (lo=1.5B, hi=0)` produce outputs `(lo=852516352, hi=1) + (lo=500M, hi=0)`), the lo sums differ by `2^31` and the hi sums differ by 1.

A carry witness `c ∈ {-1, 0, 1}` accounts for this overflow:

```
D_lo + c * 2^31 = 0    (integer equation)
D_hi - c = 0            (integer equation)
```

In M31 arithmetic, `2^31 ≡ 1 (mod p)`, so both reduce to degree-1:

```
D_lo + c ≡ 0  (mod p)
D_hi - c ≡ 0  (mod p)
```

**Soundness argument**: From the constraints, `D_lo = -c + k1*p` and `D_hi = c + k2*p`. Since each limb is in `[0, p-1]` and there are 2 inputs/outputs, `|D_lo| <= 2(p-1)` and `|D_hi| <= 2(p-1)`. Substituting into the integer balance equation `D_lo + D_hi * 2^31 = 0` yields `p(c + k1 + k2(p+1)) = 0`. Since `|c + k1| <= 3` but `|k2(p+1)| >= p+1 > 3` when `k2 != 0`, we must have `k2 = 0` and `k1 = -c`, recovering the exact integer carry.

**Implementation**: 3 execution trace columns:
- `carry_pos`, `carry_neg`: binary, mutually exclusive (4 degree-2 constraints)
- `carry = carry_pos - carry_neg`: the signed carry value
- Prover computes `carry_val = hi_in_sum - hi_out_sum` and encodes `-1` as `p - 1` in M31

### Proving Flow

All three STARKs follow the same pattern:

1. **Execute** — Run the transaction logic, collect all Poseidon2 intermediate states
2. **Build trace** — Populate all 652 columns per permutation (states + sq/quad auxiliaries) + bit decomposition columns
3. **Build preprocessed column** — `is_real`: 1 at row 0, 0 at rows 1..N
4. **Commit** — Tree 0 (preprocessed) → channel, Tree 1 (execution) → channel
5. **Build component** — `FrameworkComponent` with `TraceLocationAllocator`
6. **Prove** — `stwo::prover::prove::<SimdBackend, Blake2sMerkleChannel>()` → `StarkProof`

### Verification Flow

1. Receive `StarkProof` + public inputs
2. Construct the `FrameworkEval` with public inputs embedded (same constraints)
3. Build dummy component → extract `trace_log_degree_bounds`
4. Create `CommitmentSchemeVerifier`, replay tree commitments from proof
5. Call `stwo::core::prover::verify()` → `Result<(), VerificationError>`

## Known Limitations

### Computational Integrity vs Zero-Knowledge

- **Phase 3 circuits** (`deposit.rs`, `withdraw.rs`, `spend.rs`) provide computational integrity only — the verifier sees all intermediate values including private data.
- **Phase 4 STARKs** (`stark_deposit.rs`, `stark_withdraw.rs`, `stark_spend.rs`) provide zero-knowledge — the verifier sees only public inputs.

Both phases are available simultaneously. Phase 3 is useful for testing and debugging; Phase 4 is required for production privacy.

## Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| `crypto/poseidon2_m31` | 21 | Permutation, hash, compress, round constants, S-box |
| `crypto/commitment` | 12 | Note creation, commitment, nullifier, key derivation |
| `crypto/merkle_m31` | 13 | Insert, prove, verify, depth, append-only |
| `crypto/encryption` | 11 | Encrypt, decrypt, key derivation, checksum |
| `circuits/helpers` | 6 | Permutation recording helpers |
| `circuits/deposit` | 4 | Deposit circuit execution |
| `circuits/withdraw` | 5 | Withdraw circuit execution |
| `circuits/spend` | 12 | Spend circuit execution + integration |
| `components/poseidon2_air` | 8 | AIR constraints, M4 matrix, S-box decomposition |
| `circuits/stark_deposit` | 6 | Deposit STARK prove/verify round-trip |
| `circuits/stark_withdraw` | 6 | Withdraw STARK prove/verify round-trip |
| `circuits/stark_spend` | 10 | Spend STARK prove/verify + carry witness |
| **Total** | **114** | |

## Future Work (Phase 5+)

- **Recursive STARK** — Prove the verifier itself (STARK of a STARK)
- **Batch proving** — Multiple transactions in one STARK for amortized cost
- **On-chain verifier contract** — Cairo contract for Phase 4 STARKs on Starknet
- **Privacy Pools / ASP** — Association Set Provider compliance for regulatory compatibility
