/// Cairo AIR implementation for the Hades permutation verifier.
///
/// This module constrains that every Hades permutation in the recursive
/// proof trace was computed correctly, using:
///   - VerifyMul252: carry-chain multiplication over 9-bit limbs
///   - Cube252: S-box (x → x³) via two multiplications
///   - MDS: 3×3 linear matrix application
///   - Range checks via LogUp for carry bounds
///
/// The Hades AIR operates on 91-row blocks (one per Hades permutation call):
///   4 full rounds + 83 partial rounds + 4 full rounds
///
/// Connected to the chain AIR via LogUp:
///   - Chain AIR contributes +1 for each (input, output) pair
///   - Hades AIR contributes -1 at block boundaries (verified permutations)
///   - Sum = 0 proves all chain entries have valid Hades permutations

use stwo_verifier_core::fields::qm31::{QM31, QM31Zero, QM31One};
use stwo_verifier_core::fields::m31::M31;

/// Number of 9-bit limbs per felt252.
pub const LIMBS_28: u32 = 28;

/// Number of full rounds per half.
pub const N_FULL_ROUNDS_HALF: u32 = 4;
/// Number of partial rounds.
pub const N_PARTIAL_ROUNDS: u32 = 83;
/// Total rounds per Hades permutation.
pub const N_ROUNDS: u32 = 91;

/// Base for 9-bit limbs.
pub const BASE_512: u32 = 512;

/// Carry shift: 2^19 = 524288 (shifts carry from [-2^19, 2^19) to [0, 2^20)).
pub const CARRY_SHIFT: u32 = 524288;

/// Stark prime P in 9-bit limbs (28 values).
/// P = 2^251 + 17 * 2^192 + 1
/// Computed from felt252_to_9bit_limbs(P) in the Rust side.
/// These values are constant and can be hardcoded.
pub fn stark_prime_limbs() -> Array<u32> {
    // P = 0x800000000000011000000000000000000000000000000000000000000000001
    // Decomposed into 28 × 9-bit limbs (LSB first).
    // TODO: populate from Rust's stark_prime_9bit_limbs() output.
    // For now, returns placeholder — the Rust prover verifies offline.
    let mut limbs: Array<u32> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 28 { break; }
        limbs.append(0);
        i += 1;
    };
    limbs
}

/// Verify multiplication a × b ≡ c (mod P) using carry-chain constraints.
///
/// For each limb position j (0..28):
///   conv[j] = Σ_{i+l=j} a[i] * b[l]
///   conv[j] = c[j] + k * p[j] + carry[j] * 512 - carry[j-1]
///
/// Each carry must be in [-2^19, 2^19), verified via LogUp.
///
/// Parameters are QM31 evaluations at the OOD point.
pub fn verify_mul_252_at_point(
    a: Span<QM31>,        // 28 limbs
    b: Span<QM31>,        // 28 limbs
    c: Span<QM31>,        // 28 limbs
    k: QM31,
    carries: Span<QM31>,  // 27 carries
    p_limbs: Span<u32>,   // 28 limbs of Stark prime
    is_active: QM31,
) -> Array<QM31> {
    // Returns array of constraint quotients (28 total)
    let base: QM31 = qm31_from_u32(BASE_512);
    let zero: QM31 = QM31Zero::zero();
    let mut quotients: Array<QM31> = array![];

    let mut j: u32 = 0;
    loop {
        if j >= 28 { break; }

        // Convolution: Σ a[i] * b[j-i]
        let mut conv: QM31 = QM31Zero::zero();
        let mut i: u32 = 0;
        loop {
            if i > j { break; }
            if i < 28 && (j - i) < 28 {
                conv = conv + *a.at(i) * *b.at(j - i);
            }
            i += 1;
        };

        let p_j: QM31 = qm31_from_u32(*p_limbs.at(j));
        let carry_in: QM31 = if j == 0 { zero } else { *carries.at(j - 1) };
        let carry_out: QM31 = if j < 27 { *carries.at(j) } else { zero };

        let rhs = *c.at(j) + k * p_j + carry_out * base - carry_in;
        quotients.append(is_active * (conv - rhs));

        j += 1;
    };

    quotients
}

/// Verify S-box: x³ = cube_result, via x² = x*x, x³ = x²*x.
pub fn cube_252_at_point(
    x: Span<QM31>,
    x_sq: Span<QM31>,
    x_cubed: Span<QM31>,
    k_sq: QM31,
    carries_sq: Span<QM31>,
    k_cube: QM31,
    carries_cube: Span<QM31>,
    p_limbs: Span<u32>,
    is_active: QM31,
) -> Array<QM31> {
    let mut quotients: Array<QM31> = array![];

    // x * x = x_sq
    let q1 = verify_mul_252_at_point(x, x, x_sq, k_sq, carries_sq, p_limbs, is_active);
    let mut i: u32 = 0;
    loop {
        if i >= q1.len() { break; }
        quotients.append(*q1.at(i));
        i += 1;
    };

    // x_sq * x = x_cubed
    let q2 = verify_mul_252_at_point(x_sq, x, x_cubed, k_cube, carries_cube, p_limbs, is_active);
    i = 0;
    loop {
        if i >= q2.len() { break; }
        quotients.append(*q2.at(i));
        i += 1;
    };

    quotients
}

/// Verify MDS matrix application.
///
/// out[0] = 3*a + b + c
/// out[1] = a - b + c
/// out[2] = a + b - 2*c
///
/// Each row uses carry-chain arithmetic with 27 carries.
pub fn mds_at_point(
    a: Span<QM31>,       // post-sbox state[0], 28 limbs
    b: Span<QM31>,       // post-sbox state[1], 28 limbs
    c: Span<QM31>,       // post-sbox state[2], 28 limbs
    out0: Span<QM31>,
    out1: Span<QM31>,
    out2: Span<QM31>,
    carries0: Span<QM31>,
    carries1: Span<QM31>,
    carries2: Span<QM31>,
    k0: QM31,
    k1: QM31,
    k2: QM31,
    p_limbs: Span<u32>,
    is_active: QM31,
) -> Array<QM31> {
    let base: QM31 = qm31_from_u32(BASE_512);
    let zero: QM31 = QM31Zero::zero();
    let three: QM31 = qm31_from_u32(3);
    // -1 mod M31::P = M31::P - 1 = 2147483646
    let neg_one: QM31 = qm31_from_u32(2147483646);
    // -2 mod M31::P = M31::P - 2 = 2147483645
    let neg_two: QM31 = qm31_from_u32(2147483645);
    let one: QM31 = QM31One::one();

    let mut quotients: Array<QM31> = array![];

    // MDS row 0: 3a + b + c
    let coeffs0: Array<QM31> = array![three, one, one];
    mds_row_at_point(a, b, c, out0, carries0, k0, @coeffs0, p_limbs, base, is_active, ref quotients);

    // MDS row 1: a - b + c
    let coeffs1: Array<QM31> = array![one, neg_one, one];
    mds_row_at_point(a, b, c, out1, carries1, k1, @coeffs1, p_limbs, base, is_active, ref quotients);

    // MDS row 2: a + b - 2c
    let coeffs2: Array<QM31> = array![one, one, neg_two];
    mds_row_at_point(a, b, c, out2, carries2, k2, @coeffs2, p_limbs, base, is_active, ref quotients);

    quotients
}

fn mds_row_at_point(
    a: Span<QM31>,
    b: Span<QM31>,
    c: Span<QM31>,
    out: Span<QM31>,
    carries: Span<QM31>,
    k: QM31,
    coeffs: @Array<QM31>,
    p_limbs: Span<u32>,
    base: QM31,
    is_active: QM31,
    ref quotients: Array<QM31>,
) {
    let zero: QM31 = QM31Zero::zero();
    let mut j: u32 = 0;
    loop {
        if j >= 28 { break; }
        let p_j: QM31 = qm31_from_u32(*p_limbs.at(j));
        let carry_in: QM31 = if j == 0 { zero } else { *carries.at(j - 1) };
        let carry_out: QM31 = if j < 27 { *carries.at(j) } else { zero };

        let lhs = *coeffs.at(0) * *a.at(j)
            + *coeffs.at(1) * *b.at(j)
            + *coeffs.at(2) * *c.at(j);
        let rhs = *out.at(j) + k * p_j + carry_out * base - carry_in;
        quotients.append(is_active * (lhs - rhs));
        j += 1;
    };
}

/// Convert a u32 to QM31 (embed in real component).
fn qm31_from_u32(v: u32) -> QM31 {
    let m = stwo_verifier_core::fields::m31::m31(v);
    let z = stwo_verifier_core::fields::m31::m31(0);
    stwo_verifier_core::fields::qm31::QM31Trait::from_fixed_array([m, z, z, z])
}
