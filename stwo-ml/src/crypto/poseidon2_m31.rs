//! Poseidon2 hash function over the Mersenne-31 field.
//!
//! Parameters (from Plonky3 / Poseidon2 paper eprint.iacr.org/2023/323):
//!   - State width: t = 16
//!   - Rate: 8 (128-bit capacity security)
//!   - Capacity: 8
//!   - S-box: x^5 (valid: gcd(5, 2^31-2) = 1)
//!   - Full rounds: R_f = 8 (4 + 4)
//!   - Partial rounds: R_p = 14
//!   - External matrix: circ(2*M4, M4, M4, M4) from HorizenLabs
//!   - Internal diagonal: Plonky3 DiffusionMatrixMersenne31 (validated)
//!
//! Round constant generation: xorshift64 PRNG seeded with "Poseidon2-M31"
//! (nothing-up-my-sleeve, deterministic, reproducible).

use std::sync::OnceLock;

use stwo::core::fields::m31::BaseField as M31;

// ──────────────────────────── Constants ────────────────────────────

pub const STATE_WIDTH: usize = 16;
pub const RATE: usize = 8;
pub const CAPACITY: usize = 8;
pub const N_FULL_ROUNDS: usize = 8;
pub const N_HALF_FULL_ROUNDS: usize = 4;
pub const N_PARTIAL_ROUNDS: usize = 14;

/// Plonky3's validated internal diagonal vector for M31 width-16.
/// The internal diffusion matrix is M_I = J + diag(v), where J is all-ones.
/// Application: result[i] = v[i] * state[i] + sum(state).
/// -2 in M31 = 2^31 - 1 - 2 = 2147483645.
pub const INTERNAL_DIAG_U32: [u32; STATE_WIDTH] = [
    2147483645, // -2 mod p
    1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 8192, 16384, 32768, 65536,
];

/// PRNG seed: "Poseidon2-M31" as big-endian ASCII bytes packed into u64s.
/// 0x506F736569646F6E = "Poseidon", 0x322D4D3331 = "2-M31"
const SEED_HI: u64 = 0x506F736569646F6E;
const SEED_LO: u64 = 0x322D4D3331_000000;

// ──────────────────────────── Round constants ──────────────────────

pub struct RoundConstants {
    pub external: [[M31; STATE_WIDTH]; N_FULL_ROUNDS],
    pub internal: [M31; N_PARTIAL_ROUNDS],
}

static ROUND_CONSTANTS: OnceLock<RoundConstants> = OnceLock::new();

pub fn get_round_constants() -> &'static RoundConstants {
    ROUND_CONSTANTS.get_or_init(generate_round_constants)
}

fn generate_round_constants() -> RoundConstants {
    let p = (1u64 << 31) - 1;
    let mut state = SEED_HI ^ SEED_LO;

    let mut next_m31 = || -> M31 {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let val = (state % p) as u32;
        // Ensure non-zero (zero constants weaken the hash)
        M31::from_u32_unchecked(if val == 0 { 1 } else { val })
    };

    let mut external = [[M31::from_u32_unchecked(0); STATE_WIDTH]; N_FULL_ROUNDS];
    for round in external.iter_mut() {
        for c in round.iter_mut() {
            *c = next_m31();
        }
    }

    let mut internal = [M31::from_u32_unchecked(0); N_PARTIAL_ROUNDS];
    for c in internal.iter_mut() {
        *c = next_m31();
    }

    RoundConstants { external, internal }
}

// ──────────────────────────── Matrix operations ───────────────────

/// Apply the 4x4 MDS sub-matrix (HorizenLabs / STWO).
/// Equivalent to multiplication by:
///   [[5, 7, 1, 3],
///    [4, 6, 1, 1],
///    [1, 3, 5, 7],
///    [1, 1, 4, 6]]
/// Implemented with only additions (no multiplications).
#[inline(always)]
pub fn apply_m4(x: &mut [M31; 4]) {
    let t0 = x[0] + x[1];
    let t02 = t0 + t0;
    let t1 = x[2] + x[3];
    let t12 = t1 + t1;
    let t2 = x[1] + x[1] + t1;
    let t3 = x[3] + x[3] + t0;
    let t4 = t12 + t12 + t3;
    let t5 = t02 + t02 + t2;
    x[0] = t3 + t5;
    x[1] = t5;
    x[2] = t2 + t4;
    x[3] = t4;
}

/// Apply external round matrix: circ(2*M4, M4, M4, M4).
/// First apply M4 to each 4-element block, then add cross-block sums.
#[inline]
pub fn apply_external_round_matrix(state: &mut [M31; STATE_WIDTH]) {
    // Apply M4 to each 4-element chunk
    for i in 0..4 {
        let base = 4 * i;
        let mut chunk = [state[base], state[base + 1], state[base + 2], state[base + 3]];
        apply_m4(&mut chunk);
        state[base] = chunk[0];
        state[base + 1] = chunk[1];
        state[base + 2] = chunk[2];
        state[base + 3] = chunk[3];
    }

    // Add cross-block column sums
    for j in 0..4 {
        let s = state[j] + state[j + 4] + state[j + 8] + state[j + 12];
        state[j] += s;
        state[j + 4] += s;
        state[j + 8] += s;
        state[j + 12] += s;
    }
}

/// Apply internal round matrix: M_I = J + diag(INTERNAL_DIAG).
/// result[i] = INTERNAL_DIAG[i] * state[i] + sum(state)
#[inline]
pub fn apply_internal_round_matrix(state: &mut [M31; STATE_WIDTH]) {
    let sum: M31 = state.iter().copied().fold(M31::from_u32_unchecked(0), |a, b| a + b);

    for (i, s) in state.iter_mut().enumerate() {
        *s = *s * M31::from_u32_unchecked(INTERNAL_DIAG_U32[i]) + sum;
    }
}

// ──────────────────────────── S-box ───────────────────────────────

/// S-box: x^5 over M31. Costs 3 multiplications.
#[inline(always)]
pub fn sbox(x: M31) -> M31 {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x
}

// ──────────────────────────── Permutation ─────────────────────────

/// Poseidon2 permutation over M31[16].
///
/// Structure (following Poseidon2 paper):
///   - First half full rounds (4): AddConst → S-box(all) → External matrix
///   - Partial rounds (14): AddConst[0] → S-box[0] → Internal matrix
///   - Second half full rounds (4): AddConst → S-box(all) → External matrix
pub fn poseidon2_permutation(state: &mut [M31; STATE_WIDTH]) {
    let rc = get_round_constants();

    // First half: 4 full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        // Add round constants
        for i in 0..STATE_WIDTH {
            state[i] += rc.external[round][i];
        }
        // S-box on all elements
        for s in state.iter_mut() {
            *s = sbox(*s);
        }
        // External linear layer
        apply_external_round_matrix(state);
    }

    // Middle: 14 partial rounds
    for round in 0..N_PARTIAL_ROUNDS {
        // Add round constant to first element only
        state[0] += rc.internal[round];
        // S-box on first element only
        state[0] = sbox(state[0]);
        // Internal linear layer
        apply_internal_round_matrix(state);
    }

    // Second half: 4 full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        let rc_idx = round + N_HALF_FULL_ROUNDS;
        for i in 0..STATE_WIDTH {
            state[i] += rc.external[rc_idx][i];
        }
        for s in state.iter_mut() {
            *s = sbox(*s);
        }
        apply_external_round_matrix(state);
    }
}

// ──────────────────────────── Sponge hash ─────────────────────────

/// Hash variable-length M31 input to 8 M31 elements (248-bit, ~124-bit collision resistance).
///
/// Uses sponge construction with rate=8, capacity=8.
/// Absorb: add input to state[0..8], permute.
/// Squeeze: return state[0..8].
pub fn poseidon2_hash(input: &[M31]) -> [M31; RATE] {
    let mut state = [M31::from_u32_unchecked(0); STATE_WIDTH];

    // Domain separation: encode input length in first capacity element
    state[RATE] = M31::from_u32_unchecked(input.len() as u32);

    // Absorb phase: process input in chunks of RATE
    for chunk in input.chunks(RATE) {
        for (i, &val) in chunk.iter().enumerate() {
            state[i] += val;
        }
        poseidon2_permutation(&mut state);
    }

    // For empty input, still apply one permutation (domain separation is in capacity)
    if input.is_empty() {
        poseidon2_permutation(&mut state);
    }

    // Squeeze: return rate portion
    let mut output = [M31::from_u32_unchecked(0); RATE];
    output.copy_from_slice(&state[..RATE]);
    output
}

/// Hash variable-length input to 4 M31 elements (compact, ~62-bit collision resistance).
///
/// Use for Fiat-Shamir challenges or when only preimage resistance is needed.
/// For commitments and Merkle trees, use `poseidon2_hash` (8 elements) instead.
pub fn poseidon2_hash_4(input: &[M31]) -> [M31; 4] {
    let full = poseidon2_hash(input);
    [full[0], full[1], full[2], full[3]]
}

/// 2-to-1 compression function for Merkle trees.
///
/// Takes two 8-element digests, loads them into the full 16-element state,
/// applies one permutation, returns the rate portion.
///
/// This is the Jive/overwrite mode: secure for fixed-length 2-to-1 compression
/// (standard construction used by Zcash Orchard, Mina, Plonky3).
pub fn poseidon2_compress(left: &[M31; RATE], right: &[M31; RATE]) -> [M31; RATE] {
    let mut state = [M31::from_u32_unchecked(0); STATE_WIDTH];
    state[..RATE].copy_from_slice(left);
    state[RATE..].copy_from_slice(right);
    poseidon2_permutation(&mut state);

    let mut output = [M31::from_u32_unchecked(0); RATE];
    output.copy_from_slice(&state[..RATE]);
    output
}

/// Hash exactly 2 M31 elements (convenience for simple commitments).
pub fn poseidon2_hash_pair(a: M31, b: M31) -> [M31; RATE] {
    poseidon2_hash(&[a, b])
}

// ──────────────────────────── Tests ───────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox_correctness() {
        // x^5 for small values
        let x = M31::from_u32_unchecked(3);
        assert_eq!(sbox(x), M31::from_u32_unchecked(243)); // 3^5 = 243

        let x = M31::from_u32_unchecked(7);
        assert_eq!(sbox(x), M31::from_u32_unchecked(16807)); // 7^5 = 16807

        // x^5 = 0 iff x = 0
        let zero = M31::from_u32_unchecked(0);
        assert_eq!(sbox(zero), zero);

        // x^5 = 1 iff x = 1
        let one = M31::from_u32_unchecked(1);
        assert_eq!(sbox(one), one);
    }

    #[test]
    fn test_sbox_is_permutation() {
        // Verify that x^5 is a permutation over M31.
        // gcd(5, p-1) = gcd(5, 2^31-2) = 1 (since p-1 has prime factors 2,3,7,11,31,151,331)
        // Test: applying inverse exponent d_inv where 5 * d_inv ≡ 1 mod (p-1) recovers x.
        // d_inv = modinv(5, 2^31 - 2) = 1717986917
        // Proof: 5 * 1717986917 = 8589934585, and 8589934585 mod 2147483646 = 1
        let d_inv = 1717986917u64;
        let p_minus_1 = (1u64 << 31) - 2;
        assert_eq!((5 * d_inv) % p_minus_1, 1);

        // Test roundtrip for a few values
        for val in [2u32, 42, 1000, 2147483646] {
            let x = M31::from_u32_unchecked(val);
            let y = sbox(x); // x^5

            // Compute y^{d_inv} to recover x
            let mut result = M31::from_u32_unchecked(1);
            let mut base = y;
            let mut exp = d_inv;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = result * base;
                }
                base = base * base;
                exp >>= 1;
            }
            assert_eq!(result, x, "S-box roundtrip failed for {val}");
        }
    }

    #[test]
    fn test_m4_matches_stwo() {
        // Verify our M4 matches STWO's matrix:
        // [[5, 7, 1, 3], [4, 6, 1, 1], [1, 3, 5, 7], [1, 1, 4, 6]]
        let m4 = [
            [5u32, 7, 1, 3],
            [4, 6, 1, 1],
            [1, 3, 5, 7],
            [1, 1, 4, 6],
        ];

        for test_input in [[0u32, 1, 2, 3], [1, 0, 0, 0], [0, 0, 0, 1], [10, 20, 30, 40]] {
            let mut state: [M31; 4] = test_input.map(M31::from_u32_unchecked);
            apply_m4(&mut state);

            // Manual matrix multiply
            let mut expected = [0u64; 4];
            for i in 0..4 {
                for j in 0..4 {
                    expected[i] += m4[i][j] as u64 * test_input[j] as u64;
                }
            }
            let p = (1u64 << 31) - 1;
            let expected: [M31; 4] = expected.map(|v| M31::from_u32_unchecked((v % p) as u32));

            assert_eq!(state, expected, "M4 mismatch for input {test_input:?}");
        }
    }

    #[test]
    fn test_external_matrix_deterministic() {
        let mut s1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            .map(M31::from_u32_unchecked);
        let mut s2 = s1;

        apply_external_round_matrix(&mut s1);
        apply_external_round_matrix(&mut s2);

        assert_eq!(s1, s2);
    }

    #[test]
    fn test_internal_matrix_structure() {
        // Verify: result[i] = diag[i] * state[i] + sum(state)
        let input = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
            .map(M31::from_u32_unchecked);
        let mut state = input;

        let sum: M31 = input.iter().copied().fold(M31::from_u32_unchecked(0), |a, b| a + b);

        apply_internal_round_matrix(&mut state);

        for i in 0..STATE_WIDTH {
            let expected = input[i] * M31::from_u32_unchecked(INTERNAL_DIAG_U32[i]) + sum;
            assert_eq!(
                state[i], expected,
                "Internal matrix mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_permutation_deterministic() {
        let mut s1 = [0u32; STATE_WIDTH].map(M31::from_u32_unchecked);
        let mut s2 = [0u32; STATE_WIDTH].map(M31::from_u32_unchecked);

        poseidon2_permutation(&mut s1);
        poseidon2_permutation(&mut s2);

        assert_eq!(s1, s2, "Permutation must be deterministic");
    }

    #[test]
    fn test_permutation_not_identity() {
        let input: [M31; STATE_WIDTH] =
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                .map(M31::from_u32_unchecked);
        let mut state = input;
        poseidon2_permutation(&mut state);

        // Output should differ from input
        assert_ne!(state, input, "Permutation should not be identity");

        // Output should not be all zeros
        let all_zero = state.iter().all(|&x| x == M31::from_u32_unchecked(0));
        assert!(!all_zero, "Permutation output should not be all zeros");
    }

    #[test]
    fn test_permutation_diffusion() {
        // Changing one input element should change all output elements
        let mut s1 = [0u32; STATE_WIDTH].map(M31::from_u32_unchecked);
        let mut s2 = [0u32; STATE_WIDTH].map(M31::from_u32_unchecked);
        s2[0] = M31::from_u32_unchecked(1); // Change one element

        poseidon2_permutation(&mut s1);
        poseidon2_permutation(&mut s2);

        // Count differing positions
        let diffs = s1
            .iter()
            .zip(s2.iter())
            .filter(|(a, b)| a != b)
            .count();

        // All 16 elements should differ (full diffusion)
        assert_eq!(
            diffs, STATE_WIDTH,
            "Full diffusion: all elements should differ, got {diffs}/16"
        );
    }

    #[test]
    fn test_hash_deterministic() {
        let input = [1, 2, 3, 4, 5].map(M31::from_u32_unchecked);
        let h1 = poseidon2_hash(&input);
        let h2 = poseidon2_hash(&input);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let h1 = poseidon2_hash(&[M31::from_u32_unchecked(1)]);
        let h2 = poseidon2_hash(&[M31::from_u32_unchecked(2)]);
        assert_ne!(h1, h2, "Different inputs should produce different hashes");
    }

    #[test]
    fn test_hash_length_domain_separation() {
        // Hash of [1, 2] should differ from hash of [1, 2, 0] due to length encoding
        let h1 = poseidon2_hash(&[M31::from_u32_unchecked(1), M31::from_u32_unchecked(2)]);
        let h2 = poseidon2_hash(&[
            M31::from_u32_unchecked(1),
            M31::from_u32_unchecked(2),
            M31::from_u32_unchecked(0),
        ]);
        assert_ne!(h1, h2, "Length domain separation should differentiate");
    }

    #[test]
    fn test_hash_empty_input() {
        let h = poseidon2_hash(&[]);
        // Should produce a non-zero output
        let all_zero = h.iter().all(|&x| x == M31::from_u32_unchecked(0));
        assert!(!all_zero, "Hash of empty input should not be all zeros");
    }

    #[test]
    fn test_hash_long_input() {
        // Input longer than rate (>8 elements) triggers multiple absorb rounds
        let input: Vec<M31> = (0..20).map(|i| M31::from_u32_unchecked(i + 1)).collect();
        let h = poseidon2_hash(&input);

        // Should be deterministic
        let h2 = poseidon2_hash(&input);
        assert_eq!(h, h2);

        // Should differ from shorter input
        let h_short = poseidon2_hash(&input[..8]);
        assert_ne!(h, h_short);
    }

    #[test]
    fn test_hash_4_is_prefix() {
        let input = [42, 99, 7, 13, 256].map(M31::from_u32_unchecked);
        let full = poseidon2_hash(&input);
        let compact = poseidon2_hash_4(&input);
        assert_eq!(compact, [full[0], full[1], full[2], full[3]]);
    }

    #[test]
    fn test_compress_deterministic() {
        let left = [1, 2, 3, 4, 5, 6, 7, 8].map(M31::from_u32_unchecked);
        let right = [9, 10, 11, 12, 13, 14, 15, 16].map(M31::from_u32_unchecked);

        let h1 = poseidon2_compress(&left, &right);
        let h2 = poseidon2_compress(&left, &right);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_compress_order_matters() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8].map(M31::from_u32_unchecked);
        let b = [9, 10, 11, 12, 13, 14, 15, 16].map(M31::from_u32_unchecked);

        let h_ab = poseidon2_compress(&a, &b);
        let h_ba = poseidon2_compress(&b, &a);
        assert_ne!(h_ab, h_ba, "compress(a,b) != compress(b,a)");
    }

    #[test]
    fn test_compress_collision_resistance() {
        // Different inputs should produce different outputs
        let a = [1, 0, 0, 0, 0, 0, 0, 0].map(M31::from_u32_unchecked);
        let b = [0, 0, 0, 0, 0, 0, 0, 0].map(M31::from_u32_unchecked);
        let c = [2, 0, 0, 0, 0, 0, 0, 0].map(M31::from_u32_unchecked);

        let h1 = poseidon2_compress(&a, &b);
        let h2 = poseidon2_compress(&c, &b);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_round_constants_non_trivial() {
        let rc = get_round_constants();

        // External constants should all be non-zero
        for (r, round) in rc.external.iter().enumerate() {
            for (i, &c) in round.iter().enumerate() {
                assert_ne!(
                    c,
                    M31::from_u32_unchecked(0),
                    "External constant [{r}][{i}] is zero"
                );
            }
        }

        // Internal constants should all be non-zero
        for (i, &c) in rc.internal.iter().enumerate() {
            assert_ne!(
                c,
                M31::from_u32_unchecked(0),
                "Internal constant [{i}] is zero"
            );
        }

        // Constants should not all be the same
        let first = rc.external[0][0];
        let all_same = rc.external.iter().all(|r| r.iter().all(|&c| c == first));
        assert!(!all_same, "Round constants should not all be identical");
    }

    #[test]
    fn test_round_constant_generation_deterministic() {
        let rc1 = generate_round_constants();
        let rc2 = generate_round_constants();
        assert_eq!(rc1.external, rc2.external);
        assert_eq!(rc1.internal, rc2.internal);
    }

    #[test]
    fn test_statistical_distribution() {
        // Hash many inputs, verify outputs cover the field reasonably
        let mut outputs = Vec::new();
        for i in 0..100 {
            let h = poseidon2_hash(&[M31::from_u32_unchecked(i)]);
            outputs.push(h[0]);
        }

        // Check that we get at least 90 unique values out of 100
        let mut unique = outputs.clone();
        unique.sort_by_key(|m| m.0);
        unique.dedup();
        assert!(
            unique.len() >= 90,
            "Expected >= 90 unique outputs, got {}",
            unique.len()
        );
    }

    #[test]
    fn test_hash_pair_convenience() {
        let a = M31::from_u32_unchecked(42);
        let b = M31::from_u32_unchecked(99);
        let h1 = poseidon2_hash_pair(a, b);
        let h2 = poseidon2_hash(&[a, b]);
        assert_eq!(h1, h2);
    }
}
