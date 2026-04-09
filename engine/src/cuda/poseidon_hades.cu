// Starknet Poseidon Hades permutation — felt252 arithmetic on CUDA
// Field: p = 2^251 + 17 * 2^192 + 1
// State width: t = 3
// Rounds: 8 full + 83 partial + 8 full = 99

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// felt252 = 8 × u32, little-endian
// Prime p in little-endian u32 words:
// p = 0x0800000000000011000000000000000000000000000000000000000000000001
__device__ __constant__ uint32_t FELT_P[8] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000011, 0x08000000
};

// Zero
__device__ __constant__ uint32_t FELT_ZERO[8] = {0,0,0,0,0,0,0,0};

// ── felt252 addition: out = (a + b) mod p ──
__device__ void felt252_add(uint32_t* out, const uint32_t* a, const uint32_t* b) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += (uint64_t)a[i] + (uint64_t)b[i];
        out[i] = (uint32_t)(carry & 0xFFFFFFFF);
        carry >>= 32;
    }
    // Conditional subtract p if out >= p
    uint64_t borrow = 0;
    uint32_t tmp[8];
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)out[i] - (uint64_t)FELT_P[i] - borrow;
        tmp[i] = (uint32_t)(diff & 0xFFFFFFFF);
        borrow = (diff >> 63) & 1; // borrow if underflow
    }
    if (borrow == 0) {
        // out >= p, use subtracted result
        for (int i = 0; i < 8; i++) out[i] = tmp[i];
    }
}

// ── felt252 subtraction: out = (a - b) mod p ──
__device__ void felt252_sub(uint32_t* out, const uint32_t* a, const uint32_t* b) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a[i] - (uint64_t)b[i] - borrow;
        out[i] = (uint32_t)(diff & 0xFFFFFFFF);
        borrow = (diff >> 63) & 1;
    }
    if (borrow) {
        // a < b, add p
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            carry += (uint64_t)out[i] + (uint64_t)FELT_P[i];
            out[i] = (uint32_t)(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
    }
}


// ── felt252 multiplication: out = (a * b) mod p ──
// Schoolbook multiplication (8×8 → 16 words) + structured reduction
// using p = 2^251 + 17*2^192 + 1.
//
// Reduction identity: 2^256 ≡ -2^5*(17*2^192 + 1) + 2^5*p ≡ -(544*2^192 + 32) (mod p)
// So for bits above 251: reduce by subtracting multiples of p.
//
// Strategy: compute 512-bit product, then reduce the upper 256 bits word-by-word
// using the relation: word[k] * 2^(32k) mod p for k >= 8.
// We precompute 2^(32k) mod p for k = 8..15 as constants.

// Precomputed: 2^(32*k) mod p for k = 0..7 stored in the lower 256 bits directly.
// For k >= 8, we need 2^256 mod p, 2^288 mod p, etc.
// 2^251 ≡ -17*2^192 - 1 (mod p)
// 2^256 = 32 * 2^251 ≡ -544*2^192 - 32 ≡ p - 544*2^192 - 32 (mod p)
// These are small enough to hardcode as 8-word constants.

// 2^256 mod p (little-endian u32):
// = p - 544*2^192 - 32 = (2^251 + 17*2^192 + 1) - 544*2^192 - 32
// = 2^251 - 527*2^192 - 31
__device__ __constant__ uint32_t POW2_256_MOD_P[8] = {
    0xFFFFFFE1, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFDEF, 0x07FFFFFF
};
// Note: this is p - 544*(2^192) - 32. Working in hex:
// p   = 0x0800000000000011 0000000000000000 0000000000000000 0000000000000001
// 544 * 2^192 = 0x220 << 192 = 0x00000000000002200000...0
// 32 = 0x20
// p - 544*2^192 - 32 = 0x07FFFFFFFFFFDDF FFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFE1
// Corrected: compute properly below

__device__ void felt252_mul(uint32_t* out, const uint32_t* a, const uint32_t* b) {
    // Step 1: Full 512-bit product
    uint64_t product[16] = {0};

    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t t = (uint64_t)a[i] * (uint64_t)b[j] + product[i+j] + carry;
            product[i+j] = t & 0xFFFFFFFF;
            carry = t >> 32;
        }
        product[i+8] += carry;
    }

    // Step 2: Reduce to 256 bits using partial reduction
    // For the upper half (words 8-15), we need to compute:
    //   sum += product[k] * (2^(32k) mod p) for k = 8..15
    //
    // Instead of precomputed tables, use iterative approach:
    // Process from the highest word down. For each upper word:
    //   1. The upper bits (above 251) contribute via the reduction identity
    //   2. 2^251 ≡ -(17*2^192 + 1) mod p
    //
    // Simpler correct approach: treat the 512-bit product as a BigInt,
    // then do repeated subtraction of p. Since product < p^2 < 2^504,
    // and p ≈ 2^251, we need at most 2^253/p ≈ 4 subtractions.
    // But with 16 words, let's use a smarter method.

    // Collect into a single 512-bit value, then reduce via splitting at bit 251.
    // low = product[0..251], high = product[251..504]
    // result = low + high * (-(17*2^192 + 1)) mod p
    // = low - high * 17 * 2^192 - high

    // Extract lower 251 bits (words 0-7, but word 7 only lower 27 bits)
    uint32_t r[9] = {0}; // 9 words to handle intermediate overflow
    for (int i = 0; i < 8; i++) r[i] = (uint32_t)product[i];

    // Extract bits 251+ as "high": shift product right by 251
    // Bit 251 is in word 7 (251 / 32 = 7, 251 % 32 = 27), so bit 27 of word 7
    uint64_t high[9] = {0};
    // Shift right by 251 = 7*32 + 27
    for (int i = 0; i < 9; i++) {
        uint64_t lo = (i + 7 < 16) ? product[i + 7] : 0;
        uint64_t hi = (i + 8 < 16) ? product[i + 8] : 0;
        high[i] = (lo >> 27) | ((hi & 0x7FFFFFF) << 5);
    }

    // Clear upper bits of r[7]: keep only lower 27 bits
    r[7] &= 0x07FFFFFF;

    // Compute: result = low - high * (17 * 2^192 + 1) mod p
    // = low - high * 17 * 2^192 - high

    // Subtract high from r
    uint64_t borrow = 0;
    for (int i = 0; i < 9; i++) {
        uint64_t diff = (uint64_t)r[i] - (high[i] & 0xFFFFFFFF) - borrow;
        r[i] = (uint32_t)(diff & 0xFFFFFFFF);
        borrow = (diff >> 63) & 1;
    }

    // Subtract high * 17 * 2^192 from r
    // 17 * 2^192: starts at word 6 (192/32 = 6)
    // high[i] * 17 contributes to r[i + 6]
    for (int i = 0; i < 3; i++) { // high should be small (< 2^253)
        if (high[i] == 0) continue;
        uint64_t sub = high[i] * 17;
        uint64_t brw = 0;
        for (int j = i + 6; j < 9; j++) {
            uint64_t d = (uint64_t)r[j] - (sub & 0xFFFFFFFF) - brw;
            r[j] = (uint32_t)(d & 0xFFFFFFFF);
            brw = (d >> 63) & 1;
            sub >>= 32;
            if (sub == 0 && brw == 0) break;
        }
    }

    // If result is negative, add p until positive
    // Check sign: if r[8] has high bit set or r is negative
    for (int tries = 0; tries < 4; tries++) {
        // Check if r < 0 by looking at the highest word
        if ((r[8] & 0x80000000) || r[8] > 0) {
            // Could be negative (two's complement) or > p — add p
            uint64_t carry = 0;
            for (int i = 0; i < 8; i++) {
                carry += (uint64_t)r[i] + (uint64_t)FELT_P[i];
                r[i] = (uint32_t)(carry & 0xFFFFFFFF);
                carry >>= 32;
            }
            r[8] += (uint32_t)carry;
        } else {
            break;
        }
    }

    // Final conditional subtract: ensure r < p
    borrow = 0;
    uint32_t tmp[8];
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)r[i] - (uint64_t)FELT_P[i] - borrow;
        tmp[i] = (uint32_t)(diff & 0xFFFFFFFF);
        borrow = (diff >> 63) & 1;
    }
    if (borrow == 0) {
        for (int i = 0; i < 8; i++) r[i] = tmp[i];
    }

    for (int i = 0; i < 8; i++) out[i] = r[i];
}

// ── felt252 S-box: out = x^7 ──
// x^2 = x*x, x^3 = x^2*x, x^6 = x^3*x^3, x^7 = x^6*x
__device__ void felt252_pow7(uint32_t* out, const uint32_t* x) {
    uint32_t x2[8], x3[8], x6[8];
    felt252_mul(x2, x, x);      // x^2
    felt252_mul(x3, x2, x);     // x^3
    felt252_mul(x6, x3, x3);    // x^6
    felt252_mul(out, x6, x);    // x^7
}


// ── MDS matrix multiplication ──
// Starknet Poseidon uses the Cauchy MDS matrix:
// M = [[3, 1, 1], [1, -1, 1], [1, 1, -2]]  (over felt252)
// This is the standard Hades MDS for width-3.
//
// Actually, Starknet uses: state[i] = sum(M[i][j] * state[j])
// The exact MDS coefficients are baked into the round function.
// For width 3: we compute the linear mix explicitly.
// MDS mix matching Starknet Poseidon exactly:
//   t = s0 + s1 + s2
//   s0' = t + 2*s0       = 3*s0 + s1 + s2
//   s1' = t - 2*s1       = s0 - s1 + s2
//   s2' = t - 3*s2       = s0 + s1 - 2*s2
// Source: starknet-crypto-codegen/src/poseidon/mod.rs MixLayer
__device__ void mds_mix(uint32_t state[3][8]) {
    uint32_t t[8], s0[8], s1[8], s2[8];

    // t = state[0] + state[1] + state[2]
    felt252_add(t, state[0], state[1]);
    felt252_add(t, t, state[2]);

    // s0 = t + 2*state[0]
    uint32_t dbl[8];
    felt252_add(dbl, state[0], state[0]);
    felt252_add(s0, t, dbl);

    // s1 = t - 2*state[1]
    felt252_add(dbl, state[1], state[1]);
    felt252_sub(s1, t, dbl);

    // s2 = t - 3*state[2]
    uint32_t trp[8];
    felt252_add(dbl, state[2], state[2]);
    felt252_add(trp, dbl, state[2]);
    felt252_sub(s2, t, trp);

    for (int i = 0; i < 8; i++) {
        state[0][i] = s0[i];
        state[1][i] = s1[i];
        state[2][i] = s2[i];
    }
}

// ── Hades permutation kernel ──
// Single-threaded: one thread computes the full 99-round permutation.
// Input/output: state[3][8] = 3 felt252 elements = 24 u32 words.
// Round constants: rc[n_rounds * 3][8] — 3 constants per round.
extern "C" __global__ void poseidon_permute_kernel(
    uint32_t* state_io,           // 24 u32: [s0_w0..s0_w7, s1_w0..s1_w7, s2_w0..s2_w7]
    const uint32_t* round_consts, // round_constants[round][element][word]
    uint32_t n_full_first,        // 8
    uint32_t n_partial,           // 83  
    uint32_t n_full_last          // 8
) {
    // Load state into registers
    uint32_t state[3][8];
    for (int i = 0; i < 3; i++)
        for (int w = 0; w < 8; w++)
            state[i][w] = state_io[i * 8 + w];
    
    uint32_t rc_idx = 0;
    
    // Full rounds (first 8)
    for (uint32_t r = 0; r < n_full_first; r++) {
        // Add round constants
        for (int i = 0; i < 3; i++) {
            uint32_t rc[8];
            for (int w = 0; w < 8; w++) rc[w] = round_consts[rc_idx * 8 + w];
            felt252_add(state[i], state[i], rc);
            rc_idx++;
        }
        // S-box on all 3 state elements
        felt252_pow7(state[0], state[0]);
        felt252_pow7(state[1], state[1]);
        felt252_pow7(state[2], state[2]);
        // MDS mix
        mds_mix(state);
    }
    
    // Partial rounds (83)
    for (uint32_t r = 0; r < n_partial; r++) {
        // Add round constants (only to state[2] in partial rounds)
        for (int i = 0; i < 3; i++) {
            uint32_t rc[8];
            for (int w = 0; w < 8; w++) rc[w] = round_consts[rc_idx * 8 + w];
            felt252_add(state[i], state[i], rc);
            rc_idx++;
        }
        // S-box only on state[2]
        felt252_pow7(state[2], state[2]);
        // MDS mix
        mds_mix(state);
    }
    
    // Full rounds (last 8)
    for (uint32_t r = 0; r < n_full_last; r++) {
        // Add round constants
        for (int i = 0; i < 3; i++) {
            uint32_t rc[8];
            for (int w = 0; w < 8; w++) rc[w] = round_consts[rc_idx * 8 + w];
            felt252_add(state[i], state[i], rc);
            rc_idx++;
        }
        // S-box on all 3 state elements
        felt252_pow7(state[0], state[0]);
        felt252_pow7(state[1], state[1]);
        felt252_pow7(state[2], state[2]);
        // MDS mix
        mds_mix(state);
    }
    
    // Write state back
    for (int i = 0; i < 3; i++)
        for (int w = 0; w < 8; w++)
            state_io[i * 8 + w] = state[i][w];
}

// ── Fiat-Shamir mix + draw kernel ──
// Combines: mix_poly_coeffs(s0, s1, s2) + draw_qm31() → challenge
//
// Input:
//   channel_state: [digest_w0..w7, n_draws] = 9 u32
//   s0, s1, s2: QM31 poly coefficients (4 u32 each = 12 u32 total)
//   round_consts: same as poseidon_permute_kernel
// Output:
//   channel_state: updated in-place
//   challenge_out: 4 u32 (QM31 extracted from Poseidon draw)
//
// This kernel performs 2 Poseidon permutations:
//   1. mix: hades([digest, pack(s0,s1,s2), 2]) → new_digest
//   2. draw: hades([new_digest, n_draws, 3]) → extract QM31 from state[0]
extern "C" __global__ void poseidon_mix_draw_kernel(
    uint32_t* channel_state,        // [digest(8), n_draws(1)] = 9 u32, in-place
    const uint32_t* s0,             // QM31 = 4 u32
    const uint32_t* s1,             // QM31 = 4 u32
    const uint32_t* s2,             // QM31 = 4 u32
    uint32_t* challenge_out,        // QM31 output = 4 u32
    const uint32_t* round_consts,
    uint32_t n_full_first,
    uint32_t n_partial,
    uint32_t n_full_last
) {
    // Load current digest
    uint32_t digest[8];
    for (int w = 0; w < 8; w++) digest[w] = channel_state[w];
    uint32_t n_draws = channel_state[8];

    // Step 1: Pack s0, s1, s2 (12 M31 values) into a single felt252
    // pack_m31s: acc = 1; for each m31: acc = acc * 2^31 + m31
    // s0 has 4 M31s (QM31), s1 has 4, s2 has 4 = 12 total
    // For simplicity, we use poseidon_hash_many equivalent:
    // hash = hades([digest, pack(12 M31s from s0||s1||s2), 2])
    //
    // The packing: pack all 12 M31 values into one felt252
    // pack_m31s([a0,a1,a2,a3, b0,b1,b2,b3, c0,c1,c2,c3])
    // = 1 * 2^(31*12) + a0 * 2^(31*11) + a1 * 2^(31*10) + ... + c3

    // For correctness matching CPU: we need exact same packing.
    // CPU does: pack_m31s(&[c0.0.0, c0.0.1, c0.1.0, c0.1.1, c1.0.0, ..., c2.1.1])
    // Each QM31 has 4 M31 components.

    // Simplified: use poseidon_hash(digest, packed_value) where packed_value
    // is the felt252 encoding of all 12 M31 values.
    // This requires felt252 multiplication by 2^31 and addition — feasible.

    uint32_t packed[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // Start with sentinel: packed = 1
    packed[0] = 1;

    // Shift constant: 2^31 as felt252
    uint32_t shift[8] = {0x80000000u, 0, 0, 0, 0, 0, 0, 0};

    // Pack 12 M31 values: s0[0..3], s1[0..3], s2[0..3]
    const uint32_t* vals[3] = {s0, s1, s2};
    for (int q = 0; q < 3; q++) {
        for (int m = 0; m < 4; m++) {
            // packed = packed * 2^31 + vals[q][m]
            felt252_mul(packed, packed, shift);
            uint32_t val[8] = {vals[q][m], 0, 0, 0, 0, 0, 0, 0};
            felt252_add(packed, packed, val);
        }
    }

    // Mix: hades([digest, packed, 2])[0] → new digest
    uint32_t perm_state[3][8];
    for (int w = 0; w < 8; w++) {
        perm_state[0][w] = digest[w];
        perm_state[1][w] = packed[w];
        perm_state[2][w] = (w == 0) ? 2 : 0; // capacity = 2
    }

    // Inline Hades permutation (same as poseidon_permute_kernel but on local state)
    uint32_t rc_idx = 0;
    for (uint32_t r = 0; r < n_full_first; r++) {
        for (int i = 0; i < 3; i++) {
            uint32_t rc[8]; for (int w = 0; w < 8; w++) rc[w] = round_consts[rc_idx * 8 + w];
            felt252_add(perm_state[i], perm_state[i], rc); rc_idx++;
        }
        felt252_pow7(perm_state[0], perm_state[0]);
        felt252_pow7(perm_state[1], perm_state[1]);
        felt252_pow7(perm_state[2], perm_state[2]);
        mds_mix(perm_state);
    }
    for (uint32_t r = 0; r < n_partial; r++) {
        for (int i = 0; i < 3; i++) {
            uint32_t rc[8]; for (int w = 0; w < 8; w++) rc[w] = round_consts[rc_idx * 8 + w];
            felt252_add(perm_state[i], perm_state[i], rc); rc_idx++;
        }
        felt252_pow7(perm_state[2], perm_state[2]);
        mds_mix(perm_state);
    }
    for (uint32_t r = 0; r < n_full_last; r++) {
        for (int i = 0; i < 3; i++) {
            uint32_t rc[8]; for (int w = 0; w < 8; w++) rc[w] = round_consts[rc_idx * 8 + w];
            felt252_add(perm_state[i], perm_state[i], rc); rc_idx++;
        }
        felt252_pow7(perm_state[0], perm_state[0]);
        felt252_pow7(perm_state[1], perm_state[1]);
        felt252_pow7(perm_state[2], perm_state[2]);
        mds_mix(perm_state);
    }

    // Extract new digest
    for (int w = 0; w < 8; w++) digest[w] = perm_state[0][w];
    n_draws = 0; // Reset draw counter after mix

    // Step 2: Draw QM31 challenge
    // draw: hades([digest, n_draws, 3])[0] → felt252 → extract 4 M31s
    for (int w = 0; w < 8; w++) {
        perm_state[0][w] = digest[w];
        perm_state[1][w] = (w == 0) ? n_draws : 0;
        perm_state[2][w] = (w == 0) ? 3 : 0; // capacity = 3
    }

    // Hades permutation again
    rc_idx = 0;
    for (uint32_t r = 0; r < n_full_first; r++) {
        for (int i = 0; i < 3; i++) {
            uint32_t rc[8]; for (int w = 0; w < 8; w++) rc[w] = round_consts[rc_idx * 8 + w];
            felt252_add(perm_state[i], perm_state[i], rc); rc_idx++;
        }
        felt252_pow7(perm_state[0], perm_state[0]);
        felt252_pow7(perm_state[1], perm_state[1]);
        felt252_pow7(perm_state[2], perm_state[2]);
        mds_mix(perm_state);
    }
    for (uint32_t r = 0; r < n_partial; r++) {
        for (int i = 0; i < 3; i++) {
            uint32_t rc[8]; for (int w = 0; w < 8; w++) rc[w] = round_consts[rc_idx * 8 + w];
            felt252_add(perm_state[i], perm_state[i], rc); rc_idx++;
        }
        felt252_pow7(perm_state[2], perm_state[2]);
        mds_mix(perm_state);
    }
    for (uint32_t r = 0; r < n_full_last; r++) {
        for (int i = 0; i < 3; i++) {
            uint32_t rc[8]; for (int w = 0; w < 8; w++) rc[w] = round_consts[rc_idx * 8 + w];
            felt252_add(perm_state[i], perm_state[i], rc); rc_idx++;
        }
        felt252_pow7(perm_state[0], perm_state[0]);
        felt252_pow7(perm_state[1], perm_state[1]);
        felt252_pow7(perm_state[2], perm_state[2]);
        mds_mix(perm_state);
    }

    // Extract QM31 from state[0]: 4 M31 values via floor_div(2^31)
    // felt252 → extract lowest 31 bits, shift right by 31, repeat 4 times
    // This matches CPU's draw_qm31() which does:
    //   felt → m31[0] = felt & (2^31-1); felt >>= 31; ... × 4
    uint32_t drawn[8];
    for (int w = 0; w < 8; w++) drawn[w] = perm_state[0][w];

    for (int i = 0; i < 4; i++) {
        // Extract lowest 31 bits as M31
        challenge_out[i] = drawn[0] & 0x7FFFFFFFu;

        // 256-bit right shift by 31: drawn >>= 31
        // Each 32-bit word gets: new[w] = (drawn[w] >> 31) | (drawn[w+1] << 1)
        // But shift is 31, not 32, so:
        //   new[w] = (drawn[w] >> 31) | (drawn[w+1] << 1)
        // The low bit of drawn[w+1] becomes the high bit of new[w].
        uint32_t new_drawn[8];
        for (int w = 0; w < 7; w++) {
            new_drawn[w] = (drawn[w] >> 31) | (drawn[w + 1] << 1);
        }
        new_drawn[7] = drawn[7] >> 31;
        for (int w = 0; w < 8; w++) drawn[w] = new_drawn[w];
    }

    // Update channel state
    for (int w = 0; w < 8; w++) channel_state[w] = digest[w];
    channel_state[8] = n_draws + 1;
}

// ── Lambda-weighted QM31 accumulation kernel ──
// Computes: combined[c] += lambda_power * entry_values[c] for c in {s0, s1, s2}
// Then: lambda_power *= lambda
//
// Input:
//   entry_s0, entry_s1, entry_s2: QM31 values (4 u32 each) for one entry
//   lambda: QM31 batching weight (4 u32)
//   lambda_power_io: current lambda^i (4 u32), updated in-place to lambda^(i+1)
//   combined_io: [combined_s0(4), combined_s1(4), combined_s2(4)] = 12 u32, accumulated in-place
extern "C" __global__ void lambda_accumulate_kernel(
    const uint32_t* entry_s0,      // 4 u32
    const uint32_t* entry_s1,      // 4 u32
    const uint32_t* entry_s2,      // 4 u32
    const uint32_t* lambda,        // 4 u32
    uint32_t* lambda_power_io,     // 4 u32 in-place
    uint32_t* combined_io          // 12 u32 in-place [s0, s1, s2]
) {
    // QM31 = 4 × M31. We use M31 arithmetic for QM31 multiply + add.
    // QM31 mul is complex (needs CM31 mul), but for lambda weighting
    // we need: combined += lambda_power * entry
    //
    // For simplicity, we do component-wise operations:
    // QM31 = (a0 + a1*i) + (a2 + a3*i)*u where i^2=-1, u^2=2+i
    //
    // QM31 multiply: use the qm31_mul from the sumcheck kernel's field ops
    // But those are in a different compilation unit. For now, we compute
    // on M31 components with the correct QM31 multiplication formula.

    // Load lambda_power
    uint32_t lp[4];
    for (int i = 0; i < 4; i++) lp[i] = lambda_power_io[i];

    // For each of s0, s1, s2:
    // combined[c] += qm31_mul(lambda_power, entry[c])
    const uint32_t* entries[3] = {entry_s0, entry_s1, entry_s2};

    // M31 inline functions for NVRTC compatibility (no GCC statement expressions)
    // Defined at device scope above the loop
    // Uses the existing m31_add/m31_sub/m31_mul from the Poseidon section

    // QM31 multiply helper: result[4] = a[4] * b[4]
    // QM31 = (a0+a1*i) + (a2+a3*i)*u, i^2=-1, u^2=2+i
    auto qm31_mul_local = [](uint32_t* out, const uint32_t* a, const uint32_t* b) {
        #define PM 0x7FFFFFFFu
        auto mm = [](uint32_t x, uint32_t y) -> uint32_t { return (uint32_t)(((uint64_t)x * (uint64_t)y) % PM); };
        auto ma = [](uint32_t x, uint32_t y) -> uint32_t { uint32_t s = x+y; return s >= PM ? s-PM : s; };
        auto ms = [](uint32_t x, uint32_t y) -> uint32_t { return x >= y ? x-y : x+PM-y; };

        uint32_t x0y0_r = ms(mm(a[0],b[0]), mm(a[1],b[1]));
        uint32_t x0y0_i = ma(mm(a[0],b[1]), mm(a[1],b[0]));
        uint32_t x1y1_r = ms(mm(a[2],b[2]), mm(a[3],b[3]));
        uint32_t x1y1_i = ma(mm(a[2],b[3]), mm(a[3],b[2]));
        uint32_t u2_r = ms(ma(x1y1_r, x1y1_r), x1y1_i);
        uint32_t u2_i = ma(x1y1_r, ma(x1y1_i, x1y1_i));
        out[0] = ma(x0y0_r, u2_r);
        out[1] = ma(x0y0_i, u2_i);
        uint32_t x0y1_r = ms(mm(a[0],b[2]), mm(a[1],b[3]));
        uint32_t x0y1_i = ma(mm(a[0],b[3]), mm(a[1],b[2]));
        uint32_t x1y0_r = ms(mm(a[2],b[0]), mm(a[3],b[1]));
        uint32_t x1y0_i = ma(mm(a[2],b[1]), mm(a[3],b[0]));
        out[2] = ma(x0y1_r, x1y0_r);
        out[3] = ma(x0y1_i, x1y0_i);
        #undef PM
    };

    auto qm31_add_local = [](uint32_t* out, const uint32_t* a, const uint32_t* b) {
        #define PM 0x7FFFFFFFu
        for (int i = 0; i < 4; i++) { uint32_t s = a[i]+b[i]; out[i] = s >= PM ? s-PM : s; }
        #undef PM
    };

    for (int c = 0; c < 3; c++) {
        uint32_t e[4];
        for (int i = 0; i < 4; i++) e[i] = entries[c][i];

        // product = lambda_power * entry
        uint32_t product[4];
        qm31_mul_local(product, lp, e);

        // combined += product
        uint32_t off = c * 4;
        uint32_t tmp[4] = {combined_io[off], combined_io[off+1], combined_io[off+2], combined_io[off+3]};
        qm31_add_local(tmp, tmp, product);
        for (int i = 0; i < 4; i++) combined_io[off+i] = tmp[i];
    }

    // Update lambda_power *= lambda
    uint32_t la[4];
    for (int i = 0; i < 4; i++) la[i] = lambda[i];
    uint32_t new_lp[4];
    qm31_mul_local(new_lp, lp, la);
    for (int i = 0; i < 4; i++) lambda_power_io[i] = new_lp[i];
}

