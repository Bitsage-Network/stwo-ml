use elo_cairo_verifier::field::{
    M31_P, m31_add, m31_sub, m31_mul, m31_reduce,
    CM31, cm31_add, cm31_sub, cm31_mul,
    qm31_new, qm31_zero, qm31_one, qm31_add, qm31_sub, qm31_mul,
    poly_eval_degree2, pack_qm31_to_felt,
    log2_ceil, next_power_of_two, pow2,
};

// ============================================================================
// M31 Tests
// ============================================================================

#[test]
fn test_m31_add_basic() {
    assert!(m31_add(1, 2) == 3, "1 + 2 = 3");
    assert!(m31_add(0, 0) == 0, "0 + 0 = 0");
    assert!(m31_add(100, 200) == 300, "100 + 200 = 300");
}

#[test]
fn test_m31_add_wraparound() {
    // P-1 + 1 should wrap to 0
    assert!(m31_add(M31_P - 1, 1) == 0, "P-1 + 1 = 0");
    // P-1 + 2 should wrap to 1
    assert!(m31_add(M31_P - 1, 2) == 1, "P-1 + 2 = 1");
}

#[test]
fn test_m31_sub_basic() {
    assert!(m31_sub(5, 3) == 2, "5 - 3 = 2");
    assert!(m31_sub(0, 0) == 0, "0 - 0 = 0");
}

#[test]
fn test_m31_sub_wraparound() {
    // 0 - 1 should give P-1
    assert!(m31_sub(0, 1) == M31_P - 1, "0 - 1 = P-1");
    // 3 - 5 should give P-2
    assert!(m31_sub(3, 5) == M31_P - 2, "3 - 5 = P-2");
}

#[test]
fn test_m31_mul_basic() {
    assert!(m31_mul(2, 3) == 6, "2 * 3 = 6");
    assert!(m31_mul(0, 42) == 0, "0 * 42 = 0");
    assert!(m31_mul(1, 42) == 42, "1 * 42 = 42");
}

#[test]
fn test_m31_mul_large() {
    // (P-1) * (P-1) mod P should be 1 since (-1)*(-1)=1
    let result = m31_mul(M31_P - 1, M31_P - 1);
    assert!(result == 1, "(-1)*(-1) = 1");
}

#[test]
fn test_m31_inverse_of_2() {
    // inv(2) in M31 is (P+1)/2 = 0x40000000 = 1073741824
    let inv2: u64 = 0x40000000;
    let result = m31_mul(inv2, 2);
    assert!(result == 1, "inv2 * 2 = 1");
}

#[test]
fn test_m31_reduce() {
    assert!(m31_reduce(0) == 0, "reduce(0) = 0");
    assert!(m31_reduce(M31_P) == 0, "reduce(P) = 0");
    assert!(m31_reduce(M31_P + 1) == 1, "reduce(P+1) = 1");
}

// ============================================================================
// CM31 Tests
// ============================================================================

#[test]
fn test_cm31_add() {
    let x = CM31 { a: 3, b: 5 };
    let y = CM31 { a: 7, b: 11 };
    let result = cm31_add(x, y);
    assert!(result.a == 10 && result.b == 16, "cm31 add");
}

#[test]
fn test_cm31_sub() {
    let x = CM31 { a: 10, b: 20 };
    let y = CM31 { a: 3, b: 5 };
    let result = cm31_sub(x, y);
    assert!(result.a == 7 && result.b == 15, "cm31 sub");
}

#[test]
fn test_cm31_mul_identity() {
    // (1 + 0i) * (a + bi) = (a + bi)
    let one = CM31 { a: 1, b: 0 };
    let x = CM31 { a: 42, b: 17 };
    let result = cm31_mul(one, x);
    assert!(result.a == 42 && result.b == 17, "cm31 identity mul");
}

#[test]
fn test_cm31_mul_i_squared() {
    // i * i = -1 = (P-1, 0)
    let i = CM31 { a: 0, b: 1 };
    let result = cm31_mul(i, i);
    assert!(result.a == M31_P - 1 && result.b == 0, "i^2 = -1");
}

#[test]
fn test_cm31_mul_general() {
    // (2 + 3i)(4 + 5i) = 8 + 10i + 12i + 15i^2 = (8-15) + (10+12)i = (-7, 22)
    let x = CM31 { a: 2, b: 3 };
    let y = CM31 { a: 4, b: 5 };
    let result = cm31_mul(x, y);
    // -7 mod P = P - 7
    assert!(result.a == M31_P - 7, "cm31 mul real part");
    assert!(result.b == 22, "cm31 mul imag part");
}

// ============================================================================
// QM31 Tests
// ============================================================================

#[test]
fn test_qm31_add() {
    let x = qm31_new(1, 2, 3, 4);
    let y = qm31_new(10, 20, 30, 40);
    let result = qm31_add(x, y);
    assert!(result.a.a == 11, "add a.a");
    assert!(result.a.b == 22, "add a.b");
    assert!(result.b.a == 33, "add b.a");
    assert!(result.b.b == 44, "add b.b");
}

#[test]
fn test_qm31_sub() {
    let x = qm31_new(10, 20, 30, 40);
    let y = qm31_new(1, 2, 3, 4);
    let result = qm31_sub(x, y);
    assert!(result.a.a == 9, "sub a.a");
    assert!(result.a.b == 18, "sub a.b");
    assert!(result.b.a == 27, "sub b.a");
    assert!(result.b.b == 36, "sub b.b");
}

#[test]
fn test_qm31_mul_identity() {
    // 1 * x = x
    let one = qm31_one();
    let x = qm31_new(42, 17, 99, 3);
    let result = qm31_mul(one, x);
    assert!(result == x, "qm31 identity mul");
}

#[test]
fn test_qm31_mul_zero() {
    let zero = qm31_zero();
    let x = qm31_new(42, 17, 99, 3);
    let result = qm31_mul(zero, x);
    assert!(result == zero, "qm31 zero mul");
}

#[test]
fn test_qm31_inv2_times_2() {
    // inv(2) = (0x40000000, 0, 0, 0) in QM31
    let inv2 = qm31_new(0x40000000, 0, 0, 0);
    let two = qm31_add(qm31_one(), qm31_one());
    let result = qm31_mul(inv2, two);
    assert!(result == qm31_one(), "inv2 * 2 = 1 in QM31");
}

#[test]
fn test_qm31_mul_pure_j() {
    // j * j = 2 + i = (2, 1, 0, 0)
    let j = qm31_new(0, 0, 1, 0);
    let result = qm31_mul(j, j);
    assert!(result.a.a == 2, "j^2 real.a = 2");
    assert!(result.a.b == 1, "j^2 real.b = 1");
    assert!(result.b.a == 0, "j^2 j_part.a = 0");
    assert!(result.b.b == 0, "j^2 j_part.b = 0");
}

#[test]
fn test_qm31_mul_commutativity() {
    let x = qm31_new(3, 5, 7, 11);
    let y = qm31_new(13, 17, 19, 23);
    let xy = qm31_mul(x, y);
    let yx = qm31_mul(y, x);
    assert!(xy == yx, "qm31 mul should be commutative");
}

#[test]
fn test_qm31_mul_distributive() {
    // x * (y + z) = x*y + x*z
    let x = qm31_new(2, 0, 0, 0);
    let y = qm31_new(3, 0, 0, 0);
    let z = qm31_new(5, 0, 0, 0);
    let yz = qm31_add(y, z);
    let x_yz = qm31_mul(x, yz);
    let xy = qm31_mul(x, y);
    let xz = qm31_mul(x, z);
    let xy_plus_xz = qm31_add(xy, xz);
    assert!(x_yz == xy_plus_xz, "distributive law");
}

// ============================================================================
// Polynomial Evaluation Tests
// ============================================================================

#[test]
fn test_poly_eval_at_zero() {
    // p(0) = c0
    let c0 = qm31_new(42, 0, 0, 0);
    let c1 = qm31_new(7, 0, 0, 0);
    let c2 = qm31_new(3, 0, 0, 0);
    let result = poly_eval_degree2(c0, c1, c2, qm31_zero());
    assert!(result == c0, "p(0) = c0");
}

#[test]
fn test_poly_eval_at_one() {
    // p(1) = c0 + c1 + c2
    let c0 = qm31_new(10, 0, 0, 0);
    let c1 = qm31_new(20, 0, 0, 0);
    let c2 = qm31_new(30, 0, 0, 0);
    let result = poly_eval_degree2(c0, c1, c2, qm31_one());
    let expected = qm31_new(60, 0, 0, 0);
    assert!(result == expected, "p(1) = c0 + c1 + c2 = 60");
}

// ============================================================================
// Pack QM31 Tests
// ============================================================================

#[test]
fn test_pack_qm31_nonzero() {
    let v = qm31_new(1, 2, 3, 4);
    let packed = pack_qm31_to_felt(v);
    // Should be: 1 * 2^(31*4) + 1 * 2^(31*3) + 2 * 2^(31*2) + 3 * 2^31 + 4
    // Just verify it's non-zero and deterministic
    assert!(packed != 0, "packed should be non-zero");

    let packed2 = pack_qm31_to_felt(v);
    assert!(packed == packed2, "packing should be deterministic");
}

#[test]
fn test_pack_qm31_different_values() {
    let v1 = qm31_new(1, 0, 0, 0);
    let v2 = qm31_new(0, 1, 0, 0);
    let p1 = pack_qm31_to_felt(v1);
    let p2 = pack_qm31_to_felt(v2);
    assert!(p1 != p2, "different QM31s should pack differently");
}

// ============================================================================
// Utility Tests
// ============================================================================

#[test]
fn test_log2_ceil() {
    assert!(log2_ceil(1) == 0, "log2(1) = 0");
    assert!(log2_ceil(2) == 1, "log2(2) = 1");
    assert!(log2_ceil(4) == 2, "log2(4) = 2");
    assert!(log2_ceil(8) == 3, "log2(8) = 3");
    assert!(log2_ceil(3) == 2, "log2_ceil(3) = 2");
    assert!(log2_ceil(5) == 3, "log2_ceil(5) = 3");
}

#[test]
fn test_next_power_of_two() {
    assert!(next_power_of_two(1) == 1, "npo2(1) = 1");
    assert!(next_power_of_two(2) == 2, "npo2(2) = 2");
    assert!(next_power_of_two(3) == 4, "npo2(3) = 4");
    assert!(next_power_of_two(5) == 8, "npo2(5) = 8");
    assert!(next_power_of_two(8) == 8, "npo2(8) = 8");
    assert!(next_power_of_two(9) == 16, "npo2(9) = 16");
}

#[test]
fn test_pow2() {
    assert!(pow2(0) == 1, "2^0 = 1");
    assert!(pow2(1) == 2, "2^1 = 2");
    assert!(pow2(10) == 1024, "2^10 = 1024");
}
