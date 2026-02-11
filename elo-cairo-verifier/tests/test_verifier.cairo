use elo_cairo_verifier::field::{qm31_new, qm31_mul};
use elo_cairo_verifier::sumcheck::check_round_sum;
use elo_cairo_verifier::types::RoundPoly;

#[test]
fn test_round_sum_check_valid() {
    // p(0) = c0, p(1) = c0 + c1 + c2
    // expected_sum = p(0) + p(1) = 2*c0 + c1 + c2
    let c0 = qm31_new(10, 0, 0, 0);
    let c1 = qm31_new(20, 0, 0, 0);
    let c2 = qm31_new(30, 0, 0, 0);

    // expected_sum = 10 + (10 + 20 + 30) = 10 + 60 = 70
    let expected_sum = qm31_new(70, 0, 0, 0);

    let poly = RoundPoly { c0, c1, c2 };
    assert!(check_round_sum(poly, expected_sum), "valid round sum");
}

#[test]
fn test_round_sum_check_invalid() {
    let c0 = qm31_new(10, 0, 0, 0);
    let c1 = qm31_new(20, 0, 0, 0);
    let c2 = qm31_new(30, 0, 0, 0);

    // Wrong expected sum
    let wrong_sum = qm31_new(999, 0, 0, 0);
    let poly = RoundPoly { c0, c1, c2 };
    assert!(!check_round_sum(poly, wrong_sum), "invalid round sum should fail");
}

#[test]
fn test_round_sum_check_with_extension_field() {
    // Test with non-trivial QM31 values
    let c0 = qm31_new(5, 3, 7, 2);
    let c1 = qm31_new(11, 13, 17, 19);
    let c2 = qm31_new(23, 29, 31, 37);

    // p(0) = c0 = (5, 3, 7, 2)
    // p(1) = c0 + c1 + c2 = (5+11+23, 3+13+29, 7+17+31, 2+19+37) = (39, 45, 55, 58)
    // expected = p(0) + p(1) = (44, 48, 62, 60)
    let expected_sum = qm31_new(44, 48, 62, 60);

    let poly = RoundPoly { c0, c1, c2 };
    assert!(check_round_sum(poly, expected_sum), "extension field round sum");
}

#[test]
fn test_final_check_product() {
    // Simulate the final check: expected_sum = a_eval * b_eval
    let a_eval = qm31_new(3, 0, 0, 0);
    let b_eval = qm31_new(7, 0, 0, 0);
    let expected = qm31_mul(a_eval, b_eval);
    // 3 * 7 = 21
    assert!(expected == qm31_new(21, 0, 0, 0), "3 * 7 = 21");
}

#[test]
fn test_final_check_product_extension() {
    // Test with extension field values
    let a_eval = qm31_new(2, 1, 0, 0); // 2 + i in the real CM31 part
    let b_eval = qm31_new(3, 0, 0, 0); // just 3
    let product = qm31_mul(a_eval, b_eval);
    // (2+i) * 3 = 6 + 3i
    assert!(product == qm31_new(6, 3, 0, 0), "(2+i)*3 = 6+3i");
}
