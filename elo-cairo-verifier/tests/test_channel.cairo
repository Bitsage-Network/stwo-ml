use elo_cairo_verifier::channel::{
    channel_default, channel_mix_u64, channel_mix_felt,
    channel_draw_qm31, channel_draw_qm31s, channel_mix_poly_coeffs,
    channel_draw_query_indices,
};
use elo_cairo_verifier::field::{M31_P, qm31_new};

#[test]
fn test_channel_initial_state() {
    let ch = channel_default();
    assert!(ch.digest == 0, "initial digest is 0");
    assert!(ch.n_draws == 0, "initial n_draws is 0");
}

#[test]
fn test_channel_mix_u64_changes_digest() {
    let mut ch = channel_default();
    let initial_digest = ch.digest;
    channel_mix_u64(ref ch, 42);
    assert!(ch.digest != initial_digest, "mix should change digest");
    assert!(ch.n_draws == 0, "mix resets n_draws");
}

#[test]
fn test_channel_mix_felt_changes_digest() {
    let mut ch = channel_default();
    channel_mix_felt(ref ch, 0x1234);
    let d1 = ch.digest;
    channel_mix_felt(ref ch, 0x5678);
    assert!(ch.digest != d1, "different values should produce different digests");
}

#[test]
fn test_channel_draw_deterministic() {
    let mut ch1 = channel_default();
    channel_mix_u64(ref ch1, 42);
    let q1 = channel_draw_qm31(ref ch1);

    let mut ch2 = channel_default();
    channel_mix_u64(ref ch2, 42);
    let q2 = channel_draw_qm31(ref ch2);

    assert!(q1 == q2, "same state should produce same draw");
}

#[test]
fn test_channel_draw_different_from_different_state() {
    let mut ch1 = channel_default();
    channel_mix_u64(ref ch1, 1);
    let q1 = channel_draw_qm31(ref ch1);

    let mut ch2 = channel_default();
    channel_mix_u64(ref ch2, 2);
    let q2 = channel_draw_qm31(ref ch2);

    // Different states should (almost certainly) produce different draws
    assert!(q1 != q2, "different state should produce different draw");
}

#[test]
fn test_channel_draw_qm31_in_range() {
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 12345);
    let q = channel_draw_qm31(ref ch);

    // Each M31 component should be < P
    assert!(q.a.a < M31_P, "a.a in range");
    assert!(q.a.b < M31_P, "a.b in range");
    assert!(q.b.a < M31_P, "b.a in range");
    assert!(q.b.b < M31_P, "b.b in range");
}

#[test]
fn test_channel_draw_qm31s_count() {
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 99);
    let draws = channel_draw_qm31s(ref ch, 5);
    assert!(draws.len() == 5, "should draw 5 QM31s");
}

#[test]
fn test_channel_mix_poly_coeffs() {
    let mut ch = channel_default();
    let d0 = ch.digest;
    let c0 = qm31_new(1, 2, 3, 4);
    let c1 = qm31_new(5, 6, 7, 8);
    let c2 = qm31_new(9, 10, 11, 12);
    channel_mix_poly_coeffs(ref ch, c0, c1, c2);
    assert!(ch.digest != d0, "mix_poly_coeffs should change digest");
    assert!(ch.n_draws == 0, "mix resets n_draws");
}

#[test]
fn test_channel_draw_query_indices() {
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 42);
    let indices = channel_draw_query_indices(ref ch, 64, 4);
    assert!(indices.len() == 4, "should draw 4 indices");

    // All indices should be < half_n
    let mut i: u32 = 0;
    loop {
        if i >= 4 {
            break;
        }
        assert!(*indices.at(i) < 64, "index should be < half_n");
        i += 1;
    };
}

#[test]
fn test_channel_sequential_draws_differ() {
    let mut ch = channel_default();
    channel_mix_u64(ref ch, 42);
    let q1 = channel_draw_qm31(ref ch);
    let q2 = channel_draw_qm31(ref ch);
    // n_draws increments, so sequential draws should differ
    assert!(q1 != q2, "sequential draws should differ");
}
