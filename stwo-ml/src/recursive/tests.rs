//! Tests for recursive STARK composition.
//!
//! These tests verify that the witness generator produces correct traces
//! and that the instrumented channel matches the production channel.

#[cfg(test)]
mod unit_tests {
    use super::super::witness::InstrumentedChannel;
    use crate::crypto::poseidon_channel::PoseidonChannel;
    use crate::gkr::types::SecureField;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::QM31;
    use stwo::core::fields::cm31::CM31;

    #[test]
    fn test_instrumented_channel_matches_production() {
        // The instrumented channel must produce identical output to the production channel.
        let mut prod = PoseidonChannel::new();
        let mut inst = InstrumentedChannel::new();

        // Mix some values
        prod.mix_u64(42);
        inst.mix_u64(42);

        prod.mix_u64(100);
        inst.mix_u64(100);

        // Draw and compare
        let prod_draw = prod.draw_qm31();
        let inst_draw = inst.draw_qm31();
        assert_eq!(prod_draw, inst_draw, "draw_qm31 mismatch after mix_u64");

        // Mix a SecureField
        let sf = QM31(CM31(M31::from(7), M31::from(13)), CM31(M31::from(21), M31::from(31)));
        prod.mix_felt(crate::crypto::poseidon_channel::securefield_to_felt(sf));
        inst.mix_securefield(sf);

        let prod_draw2 = prod.draw_qm31();
        let inst_draw2 = inst.draw_qm31();
        assert_eq!(prod_draw2, inst_draw2, "draw_qm31 mismatch after mix_securefield");

        // Mix poly coefficients
        let c0 = QM31(CM31(M31::from(1), M31::from(2)), CM31(M31::from(3), M31::from(4)));
        let c1 = QM31(CM31(M31::from(5), M31::from(6)), CM31(M31::from(7), M31::from(8)));
        let c2 = QM31(CM31(M31::from(9), M31::from(10)), CM31(M31::from(11), M31::from(12)));

        prod.mix_poly_coeffs(c0, c1, c2);
        inst.mix_poly_coeffs(c0, c1, c2);

        let prod_draw3 = prod.draw_qm31();
        let inst_draw3 = inst.draw_qm31();
        assert_eq!(prod_draw3, inst_draw3, "draw_qm31 mismatch after mix_poly_coeffs");

        // Verify ops were recorded
        let ops = inst.ops();
        assert!(!ops.is_empty(), "instrumented channel should have recorded ops");
    }

    #[test]
    fn test_instrumented_channel_records_ops() {
        let mut channel = InstrumentedChannel::new();

        channel.mix_u64(1);
        channel.mix_u64(2);
        let _r = channel.draw_qm31();

        let sf = QM31(CM31(M31::from(1), M31::from(0)), CM31(M31::from(0), M31::from(0)));
        channel.record_mul(sf, sf, sf * sf);
        channel.record_equality_check(sf, sf);

        let (n_poseidon, n_sumcheck, n_qm31, n_eq) = channel.counters();
        assert_eq!(n_poseidon, 3, "expected 3 poseidon perms (2 mix + 1 draw)");
        assert_eq!(n_sumcheck, 0);
        assert_eq!(n_qm31, 1);
        assert_eq!(n_eq, 1);
    }

    #[test]
    fn test_instrumented_channel_many_draws() {
        // Verify that multiple consecutive draws remain consistent.
        let mut prod = PoseidonChannel::new();
        let mut inst = InstrumentedChannel::new();

        prod.mix_u64(0xDEAD);
        inst.mix_u64(0xDEAD);

        for _ in 0..50 {
            let p = prod.draw_qm31();
            let i = inst.draw_qm31();
            assert_eq!(p, i);
        }
    }
}

// Integration tests that require the full GKR pipeline will be added
// once the witness generator supports all layer types.
//
// TODO(recursive-stark):
// - test_witness_1layer_matmul: prove 1-layer MatMul, generate witness, verify shape
// - test_witness_matches_verifier: differential test against verify_gkr_inner
// - test_witness_40layer_full: full Qwen3-14B witness generation
// - test_witness_tampered_proof: tampered GKR proof → witness generation fails
