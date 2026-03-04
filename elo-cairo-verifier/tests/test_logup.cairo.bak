use elo_cairo_verifier::field::{
    M31_P, m31_mul, m31_inverse, cm31_inverse, cm31_mul, qm31_inverse, qm31_one, qm31_mul,
    qm31_new, qm31_zero, qm31_add, qm31_eq,
    m31_to_qm31, qm31_from_u32, batch_inverse, m31_pow,
    CM31, QM31,
};
use elo_cairo_verifier::channel::{
    channel_default, channel_mix_u64, channel_mix_secure_field,
    channel_mix_felts,
};
use elo_cairo_verifier::logup::verify_logup_table_sum;

// ============================================================================
// M31 Inverse Tests
// ============================================================================

#[test]
fn test_m31_pow_basic() {
    // 2^10 = 1024
    assert!(m31_pow(2, 10) == 1024, "2^10 = 1024");
    // a^0 = 1
    assert!(m31_pow(42, 0) == 1, "a^0 = 1");
    // a^1 = a
    assert!(m31_pow(42, 1) == 42, "a^1 = a");
}

#[test]
fn test_m31_inverse_of_2() {
    let inv2 = m31_inverse(2);
    // inv(2) in M31 is (P+1)/2 = 0x40000000
    assert!(inv2 == 0x40000000, "inv(2) = (P+1)/2");
    assert!(m31_mul(inv2, 2) == 1, "inv(2) * 2 = 1");
}

#[test]
fn test_m31_inverse_of_7() {
    let inv7 = m31_inverse(7);
    assert!(m31_mul(inv7, 7) == 1, "inv(7) * 7 = 1");
}

#[test]
fn test_m31_inverse_of_p_minus_1() {
    // P-1 = -1, so inv(-1) = -1 = P-1
    let inv = m31_inverse(M31_P - 1);
    assert!(inv == M31_P - 1, "inv(-1) = -1");
    assert!(m31_mul(inv, M31_P - 1) == 1, "inv(P-1) * (P-1) = 1");
}

#[test]
#[should_panic(expected: "M31_ZERO_INVERSE")]
fn test_m31_inverse_zero_panics() {
    m31_inverse(0);
}

// ============================================================================
// CM31 Inverse Tests
// ============================================================================

#[test]
fn test_cm31_inverse_real() {
    // inv(3 + 0i) = inv(3) + 0i
    let z = CM31 { a: 3, b: 0 };
    let inv = cm31_inverse(z);
    let expected_a = m31_inverse(3);
    assert!(inv.a == expected_a, "cm31 inv real part");
    assert!(inv.b == 0, "cm31 inv imag part = 0");
}

#[test]
fn test_cm31_inverse_identity() {
    // z * z^{-1} = 1
    let z = CM31 { a: 5, b: 7 };
    let inv = cm31_inverse(z);
    let product = cm31_mul(z, inv);
    assert!(product.a == 1, "cm31 inv product real = 1");
    assert!(product.b == 0, "cm31 inv product imag = 0");
}

// ============================================================================
// QM31 Inverse Tests
// ============================================================================

#[test]
fn test_qm31_inverse_real() {
    // inv(5, 0, 0, 0) = (inv(5), 0, 0, 0)
    let x = qm31_new(5, 0, 0, 0);
    let inv = qm31_inverse(x);
    let product = qm31_mul(x, inv);
    assert!(qm31_eq(product, qm31_one()), "qm31 inv of real = 1");
}

#[test]
fn test_qm31_inverse_general() {
    // x * x^{-1} = 1 for a general element
    let x = qm31_new(3, 5, 7, 11);
    let inv = qm31_inverse(x);
    let product = qm31_mul(x, inv);
    assert!(qm31_eq(product, qm31_one()), "qm31 general inv identity");
}

#[test]
fn test_qm31_inverse_pure_j() {
    // x = (0, 0, 1, 0) = j, j^{-1} should satisfy j * j^{-1} = 1
    let x = qm31_new(0, 0, 1, 0);
    let inv = qm31_inverse(x);
    let product = qm31_mul(x, inv);
    assert!(qm31_eq(product, qm31_one()), "j * inv(j) = 1");
}

#[test]
fn test_qm31_inverse_complex() {
    // x = (1, 1, 0, 0) = 1 + i, inv * x = 1
    let x = qm31_new(1, 1, 0, 0);
    let inv = qm31_inverse(x);
    let product = qm31_mul(x, inv);
    assert!(qm31_eq(product, qm31_one()), "(1+i) * inv(1+i) = 1");
}

// ============================================================================
// Helper Tests
// ============================================================================

#[test]
fn test_m31_to_qm31() {
    let v = m31_to_qm31(42);
    assert!(v.a.a == 42, "m31_to_qm31 a.a");
    assert!(v.a.b == 0, "m31_to_qm31 a.b");
    assert!(v.b.a == 0, "m31_to_qm31 b.a");
    assert!(v.b.b == 0, "m31_to_qm31 b.b");
}

#[test]
fn test_qm31_from_u32() {
    let v = qm31_from_u32(100);
    assert!(v.a.a == 100, "qm31_from_u32 value");
    assert!(v.a.b == 0 && v.b.a == 0 && v.b.b == 0, "qm31_from_u32 zeros");
}

// ============================================================================
// Batch Inverse Tests
// ============================================================================

#[test]
fn test_batch_inverse_single() {
    let values = array![qm31_new(5, 0, 0, 0)];
    let inv = batch_inverse(values.span());
    let product = qm31_mul(*values.at(0), *inv.at(0));
    assert!(qm31_eq(product, qm31_one()), "batch single inv");
}

#[test]
fn test_batch_inverse_3_elements() {
    let v0 = qm31_new(3, 0, 0, 0);
    let v1 = qm31_new(7, 0, 0, 0);
    let v2 = qm31_new(11, 0, 0, 0);
    let values = array![v0, v1, v2];
    let inv = batch_inverse(values.span());
    assert!(inv.len() == 3, "batch inverse length");
    assert!(qm31_eq(qm31_mul(v0, *inv.at(0)), qm31_one()), "batch inv 0");
    assert!(qm31_eq(qm31_mul(v1, *inv.at(1)), qm31_one()), "batch inv 1");
    assert!(qm31_eq(qm31_mul(v2, *inv.at(2)), qm31_one()), "batch inv 2");
}

#[test]
fn test_batch_inverse_matches_sequential() {
    let v0 = qm31_new(2, 3, 5, 7);
    let v1 = qm31_new(11, 13, 17, 19);
    let values = array![v0, v1];
    let batch_inv = batch_inverse(values.span());
    let seq_inv0 = qm31_inverse(v0);
    let seq_inv1 = qm31_inverse(v1);
    assert!(qm31_eq(*batch_inv.at(0), seq_inv0), "batch == seq inv 0");
    assert!(qm31_eq(*batch_inv.at(1), seq_inv1), "batch == seq inv 1");
}

#[test]
#[should_panic(expected: "EMPTY_BATCH")]
fn test_batch_inverse_empty_panics() {
    let values: Array<QM31> = ArrayTrait::new();
    batch_inverse(values.span());
}

// ============================================================================
// channel_mix_secure_field Tests
// ============================================================================

#[test]
fn test_mix_secure_field_changes_digest() {
    let mut ch = channel_default();
    let d0 = ch.digest;
    let v = qm31_new(1, 2, 3, 4);
    channel_mix_secure_field(ref ch, v);
    assert!(ch.digest != d0, "mix_secure_field should change digest");
}

#[test]
fn test_mix_secure_field_deterministic() {
    let mut ch1 = channel_default();
    channel_mix_u64(ref ch1, 42);
    let v = qm31_new(10, 20, 30, 40);
    channel_mix_secure_field(ref ch1, v);
    let d1 = ch1.digest;

    let mut ch2 = channel_default();
    channel_mix_u64(ref ch2, 42);
    channel_mix_secure_field(ref ch2, v);
    let d2 = ch2.digest;

    assert!(d1 == d2, "mix_secure_field should be deterministic");
}

#[test]
fn test_mix_secure_field_not_same_as_mix_felts() {
    // CRITICAL: mix_secure_field (4x mix_u64) != mix_felts (packed into felt252)
    let v = qm31_new(1, 2, 3, 4);

    let mut ch1 = channel_default();
    channel_mix_secure_field(ref ch1, v);
    let d1 = ch1.digest;

    let mut ch2 = channel_default();
    channel_mix_felts(ref ch2, array![v].span());
    let d2 = ch2.digest;

    assert!(d1 != d2, "mix_secure_field must differ from mix_felts");
}

#[test]
fn test_mix_secure_field_equivalent_to_4_mix_u64() {
    let v = qm31_new(10, 20, 30, 40);

    // Path A: use mix_secure_field
    let mut ch1 = channel_default();
    channel_mix_secure_field(ref ch1, v);

    // Path B: manual 4x mix_u64
    let mut ch2 = channel_default();
    channel_mix_u64(ref ch2, 10);
    channel_mix_u64(ref ch2, 20);
    channel_mix_u64(ref ch2, 30);
    channel_mix_u64(ref ch2, 40);

    assert!(ch1.digest == ch2.digest, "mix_secure_field == 4x mix_u64");
}

// ============================================================================
// LogUp Table-Side Sum Tests
// ============================================================================

#[test]
fn test_logup_table_sum_single_entry() {
    // Table: input=5, output=3, multiplicity=1
    // gamma=100, beta=7
    // d = gamma - 5 - 7*3 = 100 - 5 - 21 = 74
    // sum = 1/74
    let gamma = qm31_new(100, 0, 0, 0);
    let beta = qm31_new(7, 0, 0, 0);
    let d = qm31_new(74, 0, 0, 0);
    let inv_d = qm31_inverse(d);
    // claimed_sum = 1 * inv(74) = inv(74)
    let result = verify_logup_table_sum(
        gamma, beta,
        array![5_u64].span(),
        array![3_u64].span(),
        array![1_u32].span(),
        inv_d,
    );
    assert!(result, "single entry LogUp should verify");
}

#[test]
fn test_logup_table_sum_2_entries() {
    // Table: entry0 = (in=2, out=4, mult=3), entry1 = (in=6, out=8, mult=2)
    // gamma=50, beta=3
    // d0 = 50 - 2 - 3*4 = 36
    // d1 = 50 - 6 - 3*8 = 20
    // sum = 3/36 + 2/20 = 3*inv(36) + 2*inv(20)
    let gamma = qm31_new(50, 0, 0, 0);
    let beta = qm31_new(3, 0, 0, 0);
    let inv36 = qm31_inverse(qm31_new(36, 0, 0, 0));
    let inv20 = qm31_inverse(qm31_new(20, 0, 0, 0));
    let three = qm31_new(3, 0, 0, 0);
    let two = qm31_new(2, 0, 0, 0);
    let claimed = qm31_add(qm31_mul(three, inv36), qm31_mul(two, inv20));

    let result = verify_logup_table_sum(
        gamma, beta,
        array![2_u64, 6_u64].span(),
        array![4_u64, 8_u64].span(),
        array![3_u32, 2_u32].span(),
        claimed,
    );
    assert!(result, "2-entry LogUp should verify");
}

#[test]
fn test_logup_table_sum_wrong_sum_fails() {
    let gamma = qm31_new(100, 0, 0, 0);
    let beta = qm31_new(7, 0, 0, 0);
    let wrong_sum = qm31_new(999, 0, 0, 0);
    let result = verify_logup_table_sum(
        gamma, beta,
        array![5_u64].span(),
        array![3_u64].span(),
        array![1_u32].span(),
        wrong_sum,
    );
    assert!(!result, "wrong sum should fail");
}

#[test]
fn test_logup_table_sum_zero_multiplicities() {
    // All multiplicities are zero: sum should be zero
    let gamma = qm31_new(100, 0, 0, 0);
    let beta = qm31_new(7, 0, 0, 0);
    let result = verify_logup_table_sum(
        gamma, beta,
        array![5_u64, 10_u64].span(),
        array![3_u64, 6_u64].span(),
        array![0_u32, 0_u32].span(),
        qm31_zero(),
    );
    assert!(result, "all-zero multiplicities should give zero sum");
}
