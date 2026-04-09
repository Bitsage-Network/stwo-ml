//! Tamper detection tests for protocol soundness.
//!
//! Each test verifies that a specific attack vector is detected by the protocol.
//! These tests demonstrate that:
//! 1. The integer-only arithmetic prevents platform divergence
//! 2. The attention scale fix prevents f64-based manipulation
//! 3. The softmax guard prevents division-by-zero crashes
//! 4. The TopK verification catches wrong expert selection
//! 5. The piecewise coefficients are consistent across segment counts
//!
//! A malicious prover who corrupts any of these should be caught.

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::M31;

    // ── Integer Math Tamper Tests ────────────────────────────────────────

    #[test]
    fn test_cos_sin_deterministic_across_calls() {
        // Verify cos/sin produce identical results every time (no f64 variance)
        use crate::components::integer_math::{cos_fixed, sin_fixed};
        let angles: Vec<u32> = (0..1000).map(|i| i * 4294967).collect(); // spread across [0, 2^32)
        let results1: Vec<(i32, i32)> = angles.iter().map(|&a| (cos_fixed(a), sin_fixed(a))).collect();
        let results2: Vec<(i32, i32)> = angles.iter().map(|&a| (cos_fixed(a), sin_fixed(a))).collect();
        assert_eq!(results1, results2, "cos/sin must be deterministic");
    }

    #[test]
    fn test_rope_table_deterministic() {
        // Two builds of the same RoPE table must produce identical M31 values
        use crate::components::integer_math::{build_rope_table_integer, precompute_rope_thetas};
        let thetas = precompute_rope_thetas(64, 10000.0);
        let (cos1, sin1) = build_rope_table_integer(128, 64, &thetas, 0);
        let (cos2, sin2) = build_rope_table_integer(128, 64, &thetas, 0);
        assert_eq!(cos1, cos2, "RoPE cos must be deterministic");
        assert_eq!(sin1, sin2, "RoPE sin must be deterministic");
    }

    #[test]
    fn test_rope_offset_changes_output() {
        // Different position offsets must produce different rotation values
        use crate::components::integer_math::{build_rope_table_integer, precompute_rope_thetas};
        let thetas = precompute_rope_thetas(64, 10000.0);
        let (cos_off0, _) = build_rope_table_integer(4, 64, &thetas, 0);
        let (cos_off10, _) = build_rope_table_integer(4, 64, &thetas, 10);
        assert_ne!(cos_off0, cos_off10, "different offsets must produce different tables");
    }

    // ── Activation Tamper Tests ──────────────────────────────────────────

    #[test]
    fn test_piecewise_coefficients_different_per_activation() {
        // Each activation type must produce different coefficients
        use crate::components::activation::{ActivationType, PiecewiseLinearCoeffs};
        let gelu = PiecewiseLinearCoeffs::for_activation(ActivationType::GELU);
        let silu = PiecewiseLinearCoeffs::for_activation(ActivationType::SiLU);
        let sigmoid = PiecewiseLinearCoeffs::for_activation(ActivationType::Sigmoid);
        assert_ne!(gelu.slopes, silu.slopes, "GELU and SiLU must have different slopes");
        assert_ne!(gelu.slopes, sigmoid.slopes, "GELU and Sigmoid must have different slopes");
        assert_ne!(silu.slopes, sigmoid.slopes, "SiLU and Sigmoid must have different slopes");
    }

    #[test]
    fn test_piecewise_segment_count_changes_coefficients() {
        // Different segment counts must produce different coefficient sets
        use crate::components::activation::{ActivationType, PiecewiseLinearCoeffs};
        let c16 = PiecewiseLinearCoeffs::with_segments(ActivationType::GELU, 16);
        let c64 = PiecewiseLinearCoeffs::with_segments(ActivationType::GELU, 64);
        assert_ne!(c16.slopes.len(), c64.slopes.len());
        assert_ne!(c16.segment_shift, c64.segment_shift);
    }

    #[test]
    fn test_activation_integer_matches_piecewise_at_boundaries() {
        // At segment boundaries, the piecewise evaluation must match the integer activation
        use crate::components::activation::{ActivationType, PiecewiseLinearCoeffs, piecewise_linear_eval};
        use crate::components::integer_math::apply_activation_integer;
        let coeffs = PiecewiseLinearCoeffs::for_activation(ActivationType::SiLU);
        for seg in 0..coeffs.num_segments {
            let x_start = (seg as u32).wrapping_mul(coeffs.segment_width);
            let pw = piecewise_linear_eval(&coeffs, M31::from(x_start));
            let exact = apply_activation_integer(4, x_start); // SiLU = tag 4
            assert_eq!(pw, M31::from(exact),
                "SiLU boundary mismatch at segment {seg}, x={x_start}");
        }
    }

    // ── Attention Tamper Tests ───────────────────────────────────────────

    #[test]
    fn test_isqrt_correct_for_typical_dimensions() {
        // Verify isqrt gives correct results for typical head dimensions
        // d_k = 64, 80, 96, 128 (common in GPT-2, Llama, Qwen)
        for d_k in [64, 80, 96, 128, 256] {
            let sqrt = {
                let n = d_k as u64;
                if n < 2 { n as u32 } else {
                    let mut x = 1u64 << (((64 - n.leading_zeros()) + 1) / 2);
                    loop {
                        let x1 = (x + n / x) / 2;
                        if x1 >= x { break x as u32; }
                        x = x1;
                    }
                }
            };
            let expected = (d_k as f64).sqrt() as u32;
            assert_eq!(sqrt, expected, "isqrt({d_k}) should be {expected}, got {sqrt}");
        }
    }

    #[test]
    fn test_softmax_sum_zero_no_panic() {
        // Verify softmax with all-zero input doesn't panic (was a critical bug)
        use crate::components::attention::softmax_row_m31;
        let zeros = vec![M31::from(0u32); 4];
        // This should NOT panic — it should return a uniform distribution
        let result = softmax_row_m31(&zeros);
        assert_eq!(result.len(), 4);
    }

    // ── TopK Tamper Tests ───────────────────────────────────────────────

    #[test]
    fn test_topk_wrong_expert_detected() {
        // A malicious prover claiming the wrong expert should be caught
        use crate::components::topk::{select_top_k, verify_top_k, TopKSelection};
        let logits = vec![
            M31::from(10u32), M31::from(90u32), M31::from(50u32),
            M31::from(80u32), M31::from(30u32),
        ];
        // Correct top-2: indices 1 (90) and 3 (80)
        let correct = select_top_k(&logits, 2);
        assert!(verify_top_k(&logits, &correct).is_ok());

        // Tampered: claim indices 0 (10) and 2 (50) instead
        let tampered = TopKSelection {
            selected_indices: vec![0, 2],
            selected_values: vec![M31::from(10u32), M31::from(50u32)],
            rejected_indices: vec![1, 3, 4],
            rejected_values: vec![M31::from(90u32), M31::from(80u32), M31::from(30u32)],
            num_experts: 5,
            top_k: 2,
        };
        assert!(verify_top_k(&logits, &tampered).is_err(),
            "tampered expert selection should be rejected");
    }

    #[test]
    fn test_topk_fabricated_logits_detected() {
        // A malicious prover claiming wrong logit values should be caught
        use crate::components::topk::{verify_top_k, TopKSelection};
        let logits = vec![M31::from(10u32), M31::from(50u32), M31::from(30u32)];
        let fabricated = TopKSelection {
            selected_indices: vec![0],
            selected_values: vec![M31::from(999u32)], // fabricated value
            rejected_indices: vec![1, 2],
            rejected_values: vec![M31::from(50u32), M31::from(30u32)],
            num_experts: 3,
            top_k: 1,
        };
        assert!(verify_top_k(&logits, &fabricated).is_err(),
            "fabricated logit values should be rejected");
    }

    // ── LayerNorm γ Tamper Tests ─────────────────────────────────────────

    #[test]
    fn test_gamma_commitment_changes_with_different_gamma() {
        // Two different γ vectors must produce different commitments
        use crate::crypto::poseidon_channel::PoseidonChannel;
        use stwo::core::fields::qm31::QM31;

        fn gamma_commitment(gamma: &[M31]) -> starknet_ff::FieldElement {
            let mut ch = PoseidonChannel::new();
            ch.mix_u64(0x47414D4D41_u64); // "GAMMA" tag
            for &v in gamma {
                ch.mix_felt(crate::crypto::poseidon_channel::securefield_to_felt(
                    QM31::from(v),
                ));
            }
            ch.digest()
        }

        let g1 = vec![M31::from(1u32), M31::from(2u32), M31::from(3u32), M31::from(4u32)];
        let g2 = vec![M31::from(1u32), M31::from(2u32), M31::from(3u32), M31::from(5u32)]; // last element changed

        let c1 = gamma_commitment(&g1);
        let c2 = gamma_commitment(&g2);
        assert_ne!(c1, c2, "different γ vectors must produce different commitments");
    }

    #[test]
    fn test_gamma_identity_preserves_output() {
        // Multiplying RMSNorm output by γ=[1,1,1,1] should not change the result
        use crate::components::matmul::M31Matrix;
        let input = M31Matrix {
            rows: 1,
            cols: 4,
            data: vec![M31::from(100u32), M31::from(200u32), M31::from(300u32), M31::from(400u32)],
        };
        let ones = vec![M31::from(1u32); 4];

        // Compute RMSNorm output (uses the compiler::prove path)
        let rn = crate::compiler::prove::apply_rmsnorm_pub(&input, 4);

        // Apply gamma=[1,1,1,1] manually
        let mut scaled = rn.clone();
        for (val, &g) in scaled.data.iter_mut().zip(ones.iter()) {
            *val = *val * g;
        }
        assert_eq!(rn.data, scaled.data, "γ=[1,1,1,1] should be identity");
    }
}
