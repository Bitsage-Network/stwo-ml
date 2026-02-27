//! Minimal test to regenerate SP3 matmul-only constants for Cairo tests (v19 transcript).

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use num_traits::One;
use stwo_ml::cairo_serde::{serialize_gkr_proof_data_only, serialize_mle_opening_proof};
use stwo_ml::compiler::graph::{GraphBuilder, GraphWeights};
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::crypto::poseidon_channel::PoseidonChannel;
use stwo_ml::gkr::LayeredCircuit;

fn build_gkr_matmul_only() -> (stwo_ml::compiler::graph::ComputationGraph, M31Matrix, GraphWeights) {
    let mut builder = GraphBuilder::new((1, 4));
    builder.linear(2);
    let graph = builder.build();

    let mut input = M31Matrix::new(1, 4);
    for j in 0..4 {
        input.set(0, j, M31::from((j + 1) as u32));
    }

    let mut weights = GraphWeights::new();
    let mut w = M31Matrix::new(4, 2);
    for i in 0..4 {
        for j in 0..2 {
            w.set(i, j, M31::from((i * 2 + j + 1) as u32));
        }
    }
    weights.add_weight(0, w);

    (graph, input, weights)
}

fn mix_secure_field(channel: &mut PoseidonChannel, v: SecureField) {
    channel.mix_felt(stwo_ml::crypto::poseidon_channel::securefield_to_felt(v));
}

fn evaluate_mle_2(vals: [SecureField; 2], r: SecureField) -> SecureField {
    vals[0] * (SecureField::one() - r) + vals[1] * r
}

fn poly_eval_deg2(c0: SecureField, c1: SecureField, c2: SecureField, t: SecureField) -> SecureField {
    c0 + c1 * t + c2 * t * t
}

#[test]
fn gen_sp3_matmul_only_constants() {
    // Force individual weight binding mode for MLE opening proofs
    std::env::set_var("STWO_WEIGHT_BINDING", "individual");

    let (graph, input, weights) = build_gkr_matmul_only();

    let proof = stwo_ml::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
        .expect("proving should succeed");
    let gkr = proof.gkr_proof.as_ref().unwrap();

    // Serialize proof_data_only (just layer proofs + deferred)
    let mut proof_data = Vec::new();
    serialize_gkr_proof_data_only(gkr, &mut proof_data);

    let qm31_hex = |v: SecureField| -> String {
        let c = v.to_m31_array();
        format!("0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}", c[0].0, c[1].0, c[2].0, c[3].0)
    };

    eprintln!("=== SP3 v19 COMPRESSED PROOF DATA ===");
    eprintln!();
    // Print compressed proof_data (c1 removed)
    // tag
    eprintln!("0x{:x},", proof_data[0]);
    eprintln!("0x{:x},", proof_data[1]);
    for r in 0..2usize {
        let base = 2 + r * 12;
        eprintln!("// round[{}].c0", r);
        eprintln!("0x{:x}, 0x{:x}, 0x{:x}, 0x{:x},", proof_data[base], proof_data[base+1], proof_data[base+2], proof_data[base+3]);
        eprintln!("// round[{}].c2 (c1 omitted)", r);
        eprintln!("0x{:x}, 0x{:x}, 0x{:x}, 0x{:x},", proof_data[base+8], proof_data[base+9], proof_data[base+10], proof_data[base+11]);
    }
    let fa = 2 + 2 * 12;
    eprintln!("// final_a_eval");
    eprintln!("0x{:x}, 0x{:x}, 0x{:x}, 0x{:x},", proof_data[fa], proof_data[fa+1], proof_data[fa+2], proof_data[fa+3]);
    eprintln!("// final_b_eval");
    eprintln!("0x{:x}, 0x{:x}, 0x{:x}, 0x{:x},", proof_data[fa+4], proof_data[fa+5], proof_data[fa+6], proof_data[fa+7]);
    eprintln!("0x{:x},", proof_data[fa+8]);
    eprintln!();

    // === Manually replicate the GKR walk transcript (matches Cairo verify_gkr_model) ===
    // This gives us the channel state AFTER the GKR walk but BEFORE weight openings.
    let mut ch = PoseidonChannel::new();

    // 1. Seed: mix_u64(d=1), mix_u64(input_rows=1), mix_u64(input_cols=4)
    ch.mix_u64(1);
    ch.mix_u64(1);
    ch.mix_u64(4);

    // 2. Draw r_out (1 challenge for 1x2 output)
    let r_out = ch.draw_qm31();

    // 3. Evaluate output MLE: fold([50, 60], r_out)
    let out0 = SecureField::from(M31::from(50u32));
    let out1 = SecureField::from(M31::from(60u32));
    let output_value = evaluate_mle_2([out0, out1], r_out);

    // 4. Mix output value
    mix_secure_field(&mut ch, output_value);

    // 5. MatMul verification: mix dims, mix claim
    ch.mix_u64(1); // m
    ch.mix_u64(4); // k
    ch.mix_u64(2); // n

    mix_secure_field(&mut ch, output_value); // initial claim value

    // 6. Sumcheck rounds (read c0, c1, c2 from proof data, reconstruct c1 from claim)
    let mut current_sum = output_value;
    let mut k_challenges = Vec::new();

    // Extract round polys from uncompressed proof_data
    use stwo::core::fields::cm31::CM31;
    let read_qm31 = |data: &[starknet_ff::FieldElement], offset: usize| -> SecureField {
        let a = data[offset].to_string().parse::<u64>().unwrap();
        let b = data[offset+1].to_string().parse::<u64>().unwrap();
        let c = data[offset+2].to_string().parse::<u64>().unwrap();
        let d = data[offset+3].to_string().parse::<u64>().unwrap();
        SecureField::from_m31_array(std::array::from_fn(|i| M31::from([a,b,c,d][i] as u32)))
    };

    for r in 0..2 {
        let base = 2 + r * 12;
        let c0 = read_qm31(&proof_data, base);
        let c1 = read_qm31(&proof_data, base + 4);
        let c2 = read_qm31(&proof_data, base + 8);

        // Mix c0, c1, c2 into channel (matches channel_mix_poly_coeffs)
        ch.mix_poly_coeffs(c0, c1, c2);

        // Draw challenge
        let challenge = ch.draw_qm31();
        k_challenges.push(challenge);

        // Update sum
        current_sum = poly_eval_deg2(c0, c1, c2, challenge);
    }

    let final_a = read_qm31(&proof_data, fa);
    let final_b = read_qm31(&proof_data, fa + 4);

    // Verify final: current_sum == final_a * final_b
    assert_eq!(current_sum, final_a * final_b, "MATMUL_FINAL_MISMATCH in manual transcript");

    // Mix final evaluations
    mix_secure_field(&mut ch, final_a);
    mix_secure_field(&mut ch, final_b);

    // === Now ch has the "after GKR walk" state (before weight openings) ===
    eprintln!("=== EXPECTED VALUES ===");
    eprintln!();
    eprintln!("// Test 1: digest after GKR walk");
    eprintln!("0x{:064x}", ch.digest());
    eprintln!();
    eprintln!("// Test 2: final claim value (= final_a)");
    eprintln!("{}", qm31_hex(final_a));
    eprintln!();
    eprintln!("// Test 3: final claim point [k_challenges]");
    for (i, kc) in k_challenges.iter().enumerate() {
        eprintln!("p{}: {}", i, qm31_hex(*kc));
    }
    eprintln!();

    // Weight claim
    eprintln!("// Test 7: weight claims");
    eprintln!("eval_point[0] (r_j): {}", qm31_hex(r_out)); // r_j = r_out for m=1, n=2
    for (i, kc) in k_challenges.iter().enumerate() {
        eprintln!("eval_point[{}] (k_challenge): {}", i + 1, qm31_hex(*kc));
    }
    eprintln!("expected_value (final_b): {}", qm31_hex(final_b));
    eprintln!();

    // Weight commitment (doesn't depend on transcript)
    eprintln!("// Weight commitment");
    eprintln!("0x{:064x}", gkr.weight_commitments[0]);
    eprintln!();

    // Weight opening proof data
    if !gkr.weight_openings.is_empty() {
        let mut wo_data = Vec::new();
        serialize_mle_opening_proof(&gkr.weight_openings[0], &mut wo_data);
        eprintln!("// Weight opening proof ({} felts):", wo_data.len());
        for f in &wo_data {
            eprintln!("  0x{:x},", f);
        }
        eprintln!();

        // Verify weight opening to get post-opening digest
        let valid = stwo_ml::crypto::mle_opening::verify_mle_opening(
            gkr.weight_commitments[0],
            &gkr.weight_openings[0],
            &gkr.weight_claims[0].eval_point,
            &mut ch,
        );
        assert!(valid, "weight opening verification failed");
        eprintln!("// Test 8: digest after weight openings");
        eprintln!("0x{:064x}", ch.digest());
    }

    std::env::remove_var("STWO_WEIGHT_BINDING");
}
