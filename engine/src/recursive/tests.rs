//! Tests for recursive STARK composition.
//!
//! These tests verify that the witness generator produces correct traces
//! and that the instrumented channel matches the production channel.

#[cfg(test)]
mod unit_tests {
    use super::super::witness::InstrumentedChannel;
    use crate::crypto::poseidon_channel::PoseidonChannel;
    use crate::gkr::types::SecureField;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::QM31;

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
        let sf = QM31(
            CM31(M31::from(7), M31::from(13)),
            CM31(M31::from(21), M31::from(31)),
        );
        prod.mix_felt(crate::crypto::poseidon_channel::securefield_to_felt(sf));
        inst.mix_securefield(sf);

        let prod_draw2 = prod.draw_qm31();
        let inst_draw2 = inst.draw_qm31();
        assert_eq!(
            prod_draw2, inst_draw2,
            "draw_qm31 mismatch after mix_securefield"
        );

        // Mix poly coefficients
        let c0 = QM31(
            CM31(M31::from(1), M31::from(2)),
            CM31(M31::from(3), M31::from(4)),
        );
        let c1 = QM31(
            CM31(M31::from(5), M31::from(6)),
            CM31(M31::from(7), M31::from(8)),
        );
        let c2 = QM31(
            CM31(M31::from(9), M31::from(10)),
            CM31(M31::from(11), M31::from(12)),
        );

        prod.mix_poly_coeffs(c0, c1, c2);
        inst.mix_poly_coeffs(c0, c1, c2);

        let prod_draw3 = prod.draw_qm31();
        let inst_draw3 = inst.draw_qm31();
        assert_eq!(
            prod_draw3, inst_draw3,
            "draw_qm31 mismatch after mix_poly_coeffs"
        );

        // Verify ops were recorded
        let ops = inst.ops();
        assert!(
            !ops.is_empty(),
            "instrumented channel should have recorded ops"
        );
    }

    #[test]
    fn test_instrumented_channel_records_ops() {
        let mut channel = InstrumentedChannel::new();

        channel.mix_u64(1);
        channel.mix_u64(2);
        let _r = channel.draw_qm31();

        let sf = QM31(
            CM31(M31::from(1), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );
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

    #[test]
    fn test_instrumented_channel_deg3_poly() {
        // Verify degree-3 polynomial mixing matches production.
        let mut prod = PoseidonChannel::new();
        let mut inst = InstrumentedChannel::new();

        prod.mix_u64(0xCAFE);
        inst.mix_u64(0xCAFE);

        let c0 = QM31(
            CM31(M31::from(10), M31::from(20)),
            CM31(M31::from(30), M31::from(40)),
        );
        let c1 = QM31(
            CM31(M31::from(50), M31::from(60)),
            CM31(M31::from(70), M31::from(80)),
        );
        let c2 = QM31(
            CM31(M31::from(90), M31::from(100)),
            CM31(M31::from(110), M31::from(120)),
        );
        let c3 = QM31(
            CM31(M31::from(130), M31::from(140)),
            CM31(M31::from(150), M31::from(160)),
        );

        prod.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        inst.mix_poly_coeffs_deg3(c0, c1, c2, c3);

        let p = prod.draw_qm31();
        let i = inst.draw_qm31();
        assert_eq!(p, i, "draw mismatch after mix_poly_coeffs_deg3");
    }

    #[test]
    fn test_instrumented_channel_from_existing() {
        // Verify InstrumentedChannel::from_channel preserves state.
        let mut prod = PoseidonChannel::new();
        prod.mix_u64(42);
        prod.mix_u64(100);

        let cloned = prod.clone();
        let mut inst = InstrumentedChannel::from_channel(cloned);

        // Both should produce identical draws from the same state
        let p = prod.draw_qm31();
        let i = inst.draw_qm31();
        assert_eq!(p, i);
    }

    #[test]
    fn test_witness_op_variants() {
        // Verify all WitnessOp variants can be constructed and pattern-matched.
        use super::super::types::WitnessOp;
        use crate::components::matmul::RoundPoly;
        use crate::gkr::types::RoundPolyDeg3;

        let zero = SecureField::from(M31::from(0));
        let one = SecureField::from(M31::from(1));

        let ops: Vec<WitnessOp> = vec![
            WitnessOp::HadesPerm {
                input: [starknet_ff::FieldElement::ZERO; 3],
                output: [starknet_ff::FieldElement::ONE; 3],
            },
            WitnessOp::SumcheckRoundDeg2 {
                round_poly: RoundPoly {
                    c0: zero,
                    c1: zero,
                    c2: zero,
                },
                claim: zero,
                challenge: one,
                next_claim: zero,
            },
            WitnessOp::SumcheckRoundDeg3 {
                round_poly: RoundPolyDeg3 {
                    c0: zero,
                    c1: zero,
                    c2: zero,
                    c3: zero,
                },
                claim: zero,
                challenge: one,
                next_claim: zero,
            },
            WitnessOp::QM31Mul {
                a: one,
                b: one,
                result: one,
            },
            WitnessOp::QM31Add {
                a: one,
                b: zero,
                result: one,
            },
            WitnessOp::EqualityCheck { lhs: one, rhs: one },
            WitnessOp::ChannelMix { value: one },
            WitnessOp::ChannelDraw { result: one },
            WitnessOp::ChannelOp {
                digest_before: starknet_ff::FieldElement::ZERO,
                digest_after: starknet_ff::FieldElement::ONE,
            },
        ];

        assert_eq!(ops.len(), 9, "should have all 9 WitnessOp variants");

        for op in &ops {
            match op {
                WitnessOp::HadesPerm { .. } => {}
                WitnessOp::SumcheckRoundDeg2 { .. } => {}
                WitnessOp::SumcheckRoundDeg3 { .. } => {}
                WitnessOp::QM31Mul { .. } => {}
                WitnessOp::QM31Add { .. } => {}
                WitnessOp::EqualityCheck { .. } => {}
                WitnessOp::ChannelMix { .. } => {}
                WitnessOp::ChannelDraw { .. } => {}
                WitnessOp::ChannelOp { .. } => {}
            }
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::super::witness::generate_witness;
    use crate::compiler::graph::{GraphBuilder, GraphWeights};
    use crate::components::matmul::M31Matrix;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::QM31;

    #[test]
    fn test_witness_1layer_matmul() {
        // Prove a simple 1-layer MatMul circuit and generate witness.
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

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");

        let zero = QM31(
            CM31(M31::from(0), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );

        let witness = generate_witness(
            &circuit,
            gkr,
            &proof.execution.output,
            Some(&weights),
            zero, // weight_super_root placeholder
            zero, // io_commitment placeholder
        )
        .expect("witness generation should succeed");

        // Verify witness has content
        assert!(
            !witness.ops.is_empty(),
            "witness should have recorded operations"
        );
        assert!(
            witness.n_poseidon_perms > 0,
            "should have at least 1 Poseidon perm"
        );
        assert!(
            witness.n_sumcheck_rounds > 0,
            "should have sumcheck rounds for MatMul"
        );
        assert!(witness.n_equality_checks > 0, "should have equality checks");

        // Verify public inputs
        assert!(witness.public_inputs.n_layers > 0);

        println!("Witness stats:");
        println!("  ops: {}", witness.ops.len());
        println!("  poseidon_perms: {}", witness.n_poseidon_perms);
        println!("  sumcheck_rounds: {}", witness.n_sumcheck_rounds);
        println!("  qm31_ops: {}", witness.n_qm31_ops);
        println!("  equality_checks: {}", witness.n_equality_checks);
    }

    #[test]
    fn test_witness_matmul_chain() {
        // 2-layer MatMul chain: verify witness captures both layers.
        use crate::components::activation::ActivationType;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).activation(ActivationType::ReLU).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");
        let zero = QM31(
            CM31(M31::from(0), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );

        let witness = generate_witness(
            &circuit,
            gkr,
            &proof.execution.output,
            Some(&weights),
            zero,
            zero,
        )
        .expect("witness generation should succeed");

        // Should have more ops than 1-layer (2 MatMul + 1 Activation)
        assert!(
            witness.ops.len() > 10,
            "multi-layer witness should have many ops"
        );
        assert!(
            witness.n_sumcheck_rounds >= 2,
            "should have sumcheck rounds for both MatMuls"
        );

        println!(
            "Multi-layer witness: {} ops, {} poseidon, {} sumcheck, {} qm31, {} eq",
            witness.ops.len(),
            witness.n_poseidon_perms,
            witness.n_sumcheck_rounds,
            witness.n_qm31_ops,
            witness.n_equality_checks,
        );
    }

    #[test]
    fn test_witness_circuit_hash_deterministic() {
        // Same circuit should produce the same hash every time.
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");

        let hash1 = super::super::witness::compute_circuit_hash(&circuit);
        let hash2 = super::super::witness::compute_circuit_hash(&circuit);
        assert_eq!(hash1, hash2, "circuit hash should be deterministic");
    }

    #[test]
    fn test_witness_different_circuits_different_hashes() {
        // Different circuits should produce different hashes.
        let mut builder1 = GraphBuilder::new((1, 4));
        builder1.linear(2);
        let graph1 = builder1.build();

        let mut builder2 = GraphBuilder::new((1, 4));
        builder2.linear(8);
        let graph2 = builder2.build();

        let circuit1 = crate::gkr::LayeredCircuit::from_graph(&graph1).expect("circuit1");
        let circuit2 = crate::gkr::LayeredCircuit::from_graph(&graph2).expect("circuit2");

        let hash1 = super::super::witness::compute_circuit_hash(&circuit1);
        let hash2 = super::super::witness::compute_circuit_hash(&circuit2);
        assert_ne!(
            hash1, hash2,
            "different circuits should have different hashes"
        );
    }
}

#[cfg(test)]
mod chain_debug {
    use crate::compiler::graph::{GraphBuilder, GraphWeights};
    use crate::components::matmul::M31Matrix;
    use crate::recursive::air::{build_recursive_trace, LIMBS_PER_FELT};
    use crate::recursive::witness::generate_witness;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::QM31;

    #[test]
    #[ignore = "stale: indexes columns at 2*COLS_PER_STATE assuming 89-col layout; current slim layout is 48 cols (G7)"]
    fn test_real_witness_chain_integrity() {
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

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving");
        let gkr = proof.gkr_proof.as_ref().expect("GKR proof");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit");
        let zero = QM31(
            CM31(M31::from(0), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );

        let witness = generate_witness(
            &circuit,
            gkr,
            &proof.execution.output,
            Some(&weights),
            zero,
            zero,
        )
        .expect("witness");

        let trace = build_recursive_trace(&witness);

        // Check chain integrity: digest_after[row] == digest_before[row+1]
        let mut chain_breaks = 0;
        for row in 0..trace.n_real_rows.saturating_sub(1) {
            for j in 0..LIMBS_PER_FELT {
                // Expanded layout: digest_after at COLS_PER_STATE, shifted at 2*COLS_PER_STATE
                let digest_after =
                    trace.execution_trace[super::super::air::COLS_PER_STATE + j][row];
                let next_in_digest = trace.execution_trace[j][row + 1];
                let shifted = trace.execution_trace[2 * super::super::air::COLS_PER_STATE + j][row];

                if digest_after != next_in_digest {
                    if chain_breaks < 3 {
                        eprintln!("Chain break at row {row} limb {j}: digest_after={:?}, next_digest_before={:?}", digest_after, next_in_digest);
                    }
                    chain_breaks += 1;
                }
                assert_eq!(
                    shifted, next_in_digest,
                    "shifted column mismatch at row {row} limb {j}"
                );
            }
        }
        eprintln!(
            "Chain breaks: {chain_breaks} out of {} checks",
            (trace.n_real_rows - 1) * LIMBS_PER_FELT
        );
        assert_eq!(
            chain_breaks, 0,
            "Chain has {chain_breaks} breaks — transcript not continuous"
        );
    }
}

#[test]
fn test_poseidon_hash_many_matches_reference() {
    use starknet_crypto::poseidon_hash_many;
    use starknet_ff::FieldElement;

    // From starknet.js computePoseidonHashOnElements:
    let h0 = poseidon_hash_many(&[]);
    assert_eq!(
        format!("{:#066x}", h0),
        "0x02272be0f580fd156823304800919530eaa97430e972d7213ee13f4fbf7a5dbc"
    );

    let h1 = poseidon_hash_many(&[FieldElement::ONE]);
    assert_eq!(
        format!("{:#066x}", h1),
        "0x00579e8877c7755365d5ec1ec7d3a94a457eff5d1f40482bbe9729c064cdead2"
    );

    let h2 = poseidon_hash_many(&[FieldElement::ONE, FieldElement::TWO]);
    assert_eq!(
        format!("{:#066x}", h2),
        "0x0371cb6995ea5e7effcd2e174de264b5b407027a75a231a70c2c8d196107f0e7"
    );

    eprintln!("All poseidon_hash_many tests passed — matches starknet.js reference");
}

#[test]
fn test_mix_felts_poseidon252_matches_js() {
    use stwo::core::channel::MerkleChannel;
    use stwo::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleChannel;
    type Poseidon252Channel = <Poseidon252MerkleChannel as MerkleChannel>::C;
    use stwo::core::channel::Channel;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::SecureField;

    let mut ch = Poseidon252Channel::default();
    let qm31 =
        SecureField::from_m31_array([M31::from(1), M31::from(2), M31::from(3), M31::from(4)]);
    ch.mix_felts(&[qm31]);
    let digest = ch.digest();
    eprintln!("Rust mix_felts digest: 0x{:064x}", digest);
    // From starknet.js: poseidon_hash_many([0, 0x10000000200000008000000180000004])
    // = 0xe029e34eb10ff7719b4a23d283495d23f65760b9d9a9b7885f0350879a7f1c
    let expected = starknet_ff::FieldElement::from_hex_be(
        "0x0e029e34eb10ff7719b4a23d283495d23f65760b9d9a9b7885f0350879a7f1c",
    )
    .unwrap();
    assert_eq!(
        digest, expected,
        "mix_felts must match starknet.js reference"
    );
}
