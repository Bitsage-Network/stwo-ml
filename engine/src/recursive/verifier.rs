//! Rust-side verification of recursive STARK proofs.
//!
//! This is the pre-flight check before on-chain submission. It reconstructs
//! the AIR component from public inputs and calls `stwo::verify()`.

use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::vcs_lifted::poseidon252_merkle::{
    Poseidon252MerkleChannel, Poseidon252MerkleHasher,
};
use stwo::core::verifier::verify as stwo_verify;
use stwo_constraint_framework::{FrameworkComponent, TraceLocationAllocator};

use super::air::RecursiveVerifierEval;
use super::prover::RecursiveError;
use super::types::RecursivePublicInputs;

/// Verify a recursive STARK proof against its public inputs.
///
/// This is the Rust-side pre-flight check. It:
/// 1. Reconstructs the AIR evaluator from public inputs
/// 2. Builds a dummy component to get trace degree bounds
/// 3. Replays the commitment scheme
/// 4. Calls `stwo::verify()`
///
/// If this passes, the proof is valid and can be submitted on-chain.
pub fn verify_recursive(
    stark_proof: &stwo::core::proof::StarkProof<Poseidon252MerkleHasher>,
    public_inputs: &RecursivePublicInputs,
    pass1_final_digest: starknet_ff::FieldElement,
    n_real_rows: u32,
    log_size: u32,
    final_digest: starknet_ff::FieldElement,
    logup_claimed_sum: SecureField,
) -> Result<(), RecursiveError> {
    // PcsConfig must match what the prover used. The Cairo verifier reads
    // it from the proof body. For Rust pre-flight, match the prover's config.
    let pcs_config = {
        #[cfg(test)]
        let level = std::env::var("OBELYZK_RECURSIVE_SECURITY")
            .unwrap_or_else(|_| "production".to_string());
        #[cfg(not(test))]
        let level = "production".to_string();
        match level.as_str() {
            #[cfg(test)]
            "test" => PcsConfig::default(),
            _ => PcsConfig {
                pow_bits: 20,
                fri_config: stwo::core::fri::FriConfig::new(0, 5, 20, 1),
                lifting_log_size: None,
            },
        }
    };

    // Build evaluator from public inputs
    let zero_limbs = super::air::felt252_to_limbs(&starknet_ff::FieldElement::ZERO);
    let eval = RecursiveVerifierEval {
        log_n_rows: log_size,
        n_real_rows,
        initial_digest_limbs: zero_limbs,
        final_digest_limbs: super::air::felt252_to_limbs(&final_digest),
        hades_lookup: None, // LogUp disabled until multi-component STARK is wired
        hades_enabled: false,
    };

    // Build dummy component to get trace_log_degree_bounds
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component =
        FrameworkComponent::new(&mut allocator, eval.clone(), SecureField::zero());
    let bounds = Component::trace_log_degree_bounds(&dummy_component);

    // Set up channel and commitment scheme verifier
    let channel = &mut <Poseidon252MerkleChannel as MerkleChannel>::C::default();

    // Mix PcsConfig into channel (must match prover's config.mix_into() call)
    pcs_config.mix_into(channel);

    // Mix public inputs into channel (must match prover's binding)
    channel.mix_felts(&[
        public_inputs.circuit_hash,
        public_inputs.io_commitment,
        public_inputs.weight_super_root,
    ]);
    channel.mix_u64(public_inputs.n_layers as u64);
    channel.mix_u64(public_inputs.n_poseidon_perms as u64);
    channel.mix_felts(&[public_inputs.seed_digest]);
    // hades_commitment binding (two-level recursion)
    {
        let bytes = public_inputs.hades_commitment.to_bytes_be();
        channel.mix_u64(u64::from_be_bytes(bytes[0..8].try_into().unwrap()));
        channel.mix_u64(u64::from_be_bytes(bytes[8..16].try_into().unwrap()));
        channel.mix_u64(u64::from_be_bytes(bytes[16..24].try_into().unwrap()));
        channel.mix_u64(u64::from_be_bytes(bytes[24..32].try_into().unwrap()));
    }

    // Also bind the felt252 io_commitment (4 × u64, matching prover)
    {
        let io_felt =
            crate::crypto::poseidon_channel::securefield_to_felt(public_inputs.io_commitment);
        let bytes = io_felt.to_bytes_be();
        let u0 = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let u2 = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        let u3 = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        channel.mix_u64(u0);
        channel.mix_u64(u1);
        channel.mix_u64(u2);
        channel.mix_u64(u3);
    }

    // Bind Pass 1 (full GKR verification) digest — must match prover
    {
        let bytes = pass1_final_digest.to_bytes_be();
        let u0 = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let u1 = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        let u2 = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        let u3 = u64::from_be_bytes(bytes[24..32].try_into().unwrap());
        channel.mix_u64(u0);
        channel.mix_u64(u1);
        channel.mix_u64(u2);
        channel.mix_u64(u3);
    }

    let mut commitment_scheme =
        CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(pcs_config);

    if stark_proof.commitments.len() < 2 {
        return Err(RecursiveError::ProvingFailed(format!(
            "expected at least 2 commitments, got {}",
            stark_proof.commitments.len()
        )));
    }

    // Commit Tree 0 (preprocessed) and Tree 1 (execution)
    commitment_scheme.commit(stark_proof.commitments[0], &bounds[0], channel);
    commitment_scheme.commit(stark_proof.commitments[1], &bounds[1], channel);

    // Detect LogUp: 3+ commitments means Tree 2 (interaction) is present.
    let logup_active = stark_proof.commitments.len() >= 3
        && logup_claimed_sum != SecureField::zero();

    let logup_relation = if logup_active {
        Some(super::air::HadesPermRelation::draw(channel))
    } else {
        None
    };

    // Build evaluator with LogUp if active
    let eval_final = RecursiveVerifierEval {
        hades_lookup: logup_relation,
        ..eval
    };

    // Build component — bounds now include interaction trace if LogUp active
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(
        &mut allocator,
        eval_final,
        if logup_active { logup_claimed_sum } else { SecureField::zero() },
    );
    let all_bounds = Component::trace_log_degree_bounds(&component);

    // Commit Tree 2 (interaction) if present, using bounds from the component
    if logup_active && all_bounds.len() > 2 {
        commitment_scheme.commit(
            stark_proof.commitments[2],
            &all_bounds[2],
            channel,
        );
    }

    // Verify
    stwo_verify::<Poseidon252MerkleChannel>(
        &[&component as &dyn Component],
        channel,
        &mut commitment_scheme,
        stark_proof.clone(),
    )
    .map_err(|e| {
        RecursiveError::ProvingFailed(format!("recursive STARK verification failed: {e:?}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{GraphBuilder, GraphWeights};
    use crate::components::matmul::M31Matrix;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::qm31::QM31;

    #[test]
    fn test_poseidon_channel_config_mixing() {
        use stwo::core::channel::Channel;
        // Verify that config.mix_into() produces a DIFFERENT digest than manual mix_u64 calls.
        // The Cairo contract uses mix_into(), so the prover must too.
        let config = PcsConfig {
            pow_bits: 20,
            fri_config: stwo::core::fri::FriConfig::new(0, 5, 20, 1),
            lifting_log_size: None,
        };

        let ch1 = &mut <Poseidon252MerkleChannel as MerkleChannel>::C::default();
        config.mix_into(ch1);

        let ch2 = &mut <Poseidon252MerkleChannel as MerkleChannel>::C::default();
        ch2.mix_u64(20);
        ch2.mix_u64(5);
        ch2.mix_u64(28);
        ch2.mix_u64(0);

        eprintln!("config.mix_into digest: {:?}", ch1.digest());
        eprintln!("manual mix_u64 digest:  {:?}", ch2.digest());
        assert_ne!(
            ch1.digest(), ch2.digest(),
            "mix_into and manual mix_u64 MUST produce different digests"
        );
    }

    #[test]
    fn test_verify_recursive_roundtrip() {
        std::env::set_var("OBELYZK_RECURSIVE_SECURITY", "test");
        // Full roundtrip: prove GKR → prove recursive → verify recursive.
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

        // Generate witness + build trace + prove
        let witness = crate::recursive::generate_witness(
            &circuit,
            gkr,
            &proof.execution.output,
            Some(&weights),
            zero,
            zero,
        )
        .expect("witness generation should succeed");

        let trace_data = crate::recursive::air::build_recursive_trace(&witness);
        let log_size = trace_data.log_size;

        // Prove recursive STARK
        let recursive_result = crate::recursive::prove_recursive(
            &circuit,
            gkr,
            &proof.execution.output,
            &weights,
            zero,
            zero,
            0.0,
        );

        let recursive_proof = recursive_result.expect("recursive proving should succeed");
        eprintln!(
            "Recursive proof produced (log_size={}, {} poseidon perms, {:.3}s)",
            recursive_proof.log_size,
            witness.n_poseidon_perms,
            recursive_proof.metadata.recursive_prove_time_secs,
        );

        // Full roundtrip: verify the recursive STARK proof
        let verify_result = verify_recursive(
            &recursive_proof.stark_proof,
            &recursive_proof.public_inputs,
            recursive_proof.pass1_final_digest,
            recursive_proof.n_real_rows,
            recursive_proof.log_size,
            recursive_proof.final_digest,
            recursive_proof.logup_claimed_sum,
        );
        assert!(
            verify_result.is_ok(),
            "recursive verification should succeed: {:?}",
            verify_result.err()
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // Adversarial tests — metadata relabeling + proof integrity
    // ═══════════════════════════════════════════════════════════════

    /// Helper: produce a valid recursive proof for adversarial testing.
    fn adversarial_proof() -> super::super::types::RecursiveProof {
        std::env::set_var("OBELYZK_RECURSIVE_SECURITY", "test");
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
        crate::recursive::prove_recursive(
            &circuit,
            gkr,
            &proof.execution.output,
            &weights,
            zero,
            zero,
            0.0,
        )
        .expect("recursive proving should succeed")
    }

    #[test]
    fn test_oods_diagnostic() {
        use num_traits::One;
        use stwo::core::air::accumulation::PointEvaluationAccumulator;
        use stwo::core::constraints::coset_vanishing;
        use stwo::core::fields::FieldExpOps;
        use stwo::core::poly::circle::CanonicCoset;

        std::env::set_var("OBELYZK_RECURSIVE_SECURITY", "test");
        std::env::set_var("OBELYZK_LOGUP", "0");
        let rp = adversarial_proof();

        // Replay the FULL verification channel to extract random_coeff + oods_point
        let pcs_config = stwo::core::pcs::PcsConfig::default();
        let channel = &mut <Poseidon252MerkleChannel as stwo::core::channel::MerkleChannel>::C::default();

        channel.mix_u64(pcs_config.pow_bits as u64);
        channel.mix_u64(pcs_config.fri_config.log_blowup_factor as u64);
        channel.mix_u64(pcs_config.fri_config.n_queries as u64);
        channel.mix_u64(pcs_config.fri_config.log_last_layer_degree_bound as u64);
        channel.mix_felts(&[
            rp.public_inputs.circuit_hash,
            rp.public_inputs.io_commitment,
            rp.public_inputs.weight_super_root,
        ]);
        channel.mix_u64(rp.public_inputs.n_layers as u64);
        channel.mix_u64(rp.public_inputs.n_poseidon_perms as u64);
        channel.mix_felts(&[rp.public_inputs.seed_digest]);
        {
            let io_felt = crate::crypto::poseidon_channel::securefield_to_felt(rp.public_inputs.io_commitment);
            let bytes = io_felt.to_bytes_be();
            channel.mix_u64(u64::from_be_bytes(bytes[0..8].try_into().unwrap()));
            channel.mix_u64(u64::from_be_bytes(bytes[8..16].try_into().unwrap()));
            channel.mix_u64(u64::from_be_bytes(bytes[16..24].try_into().unwrap()));
            channel.mix_u64(u64::from_be_bytes(bytes[24..32].try_into().unwrap()));
        }
        {
            let bytes = rp.pass1_final_digest.to_bytes_be();
            channel.mix_u64(u64::from_be_bytes(bytes[0..8].try_into().unwrap()));
            channel.mix_u64(u64::from_be_bytes(bytes[8..16].try_into().unwrap()));
            channel.mix_u64(u64::from_be_bytes(bytes[16..24].try_into().unwrap()));
            channel.mix_u64(u64::from_be_bytes(bytes[24..32].try_into().unwrap()));
        }

        let zero_limbs = super::super::air::felt252_to_limbs(&starknet_ff::FieldElement::ZERO);
        let eval = RecursiveVerifierEval {
            log_n_rows: rp.log_size,
            n_real_rows: rp.n_real_rows,
            initial_digest_limbs: zero_limbs,
            final_digest_limbs: super::super::air::felt252_to_limbs(&rp.final_digest),
            hades_lookup: None,
            hades_enabled: false,
        };
        let mut allocator = TraceLocationAllocator::default();
        let component = FrameworkComponent::new(&mut allocator, eval, SecureField::zero());
        let bounds = Component::trace_log_degree_bounds(&component);

        let mut cs = stwo::core::pcs::CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(pcs_config);
        cs.commit(rp.stark_proof.commitments[0], &bounds[0], channel);
        cs.commit(rp.stark_proof.commitments[1], &bounds[1], channel);

        let comp_log_size = component.max_constraint_log_degree_bound();
        cs.commit(
            *rp.stark_proof.commitments.last().unwrap(),
            &[comp_log_size - 1; 2 * 4],
            channel,
        );

        let random_coeff = channel.draw_secure_felt();
        let oods_point = stwo::core::circle::CirclePoint::<SecureField>::get_random_point(channel);
        eprintln!("random_coeff: {:?}", random_coeff);
        eprintln!("oods_point.x: {:?}", oods_point.x);

        // Compute via FrameworkComponent
        let max_log_deg = comp_log_size - 1;
        let rust_eval = {
            let mut acc = PointEvaluationAccumulator::new(random_coeff);
            component.evaluate_constraint_quotients_at_point(
                oods_point, &rp.stark_proof.sampled_values, &mut acc, max_log_deg,
            );
            acc.finalize()
        };
        eprintln!("rust_eval: {:?}", rust_eval);

        // Manual "Cairo-style" evaluation
        let denom_inv = coset_vanishing(CanonicCoset::new(max_log_deg).coset, oods_point).inverse();
        let sv = &rp.stark_proof.sampled_values;
        let prep = &sv[0];
        let trace = &sv[1];

        eprintln!("prep cols: {}, trace cols: {}", prep.len(), trace.len());

        let one = SecureField::one();
        let zero_sf = SecureField::zero();
        let is_first = prep[0][0];
        let is_last = prep[1][0];
        let is_chain = prep[2][0];
        let is_active = trace[45][0];
        let ac = trace[46][0];
        let ac_next = trace[47][0];
        let add_k = trace[44][0];

        let n = 1u32 << rp.log_size;
        let n_inv = M31::from(n).inverse();
        let corr = SecureField::from(M31::from(rp.n_real_rows)) * SecureField::from(n_inv);

        let mut cairo = zero_sf;
        // C1
        cairo = cairo * random_coeff + denom_inv * is_active * (one - is_active);
        // C2
        cairo = cairo * random_coeff + denom_inv * (ac_next - ac - is_active + corr);
        // C3: initial boundary ×9
        let init_limbs = super::super::air::felt252_to_limbs(&starknet_ff::FieldElement::ZERO);
        for j in 0..9usize {
            cairo = cairo * random_coeff + denom_inv * is_first * (trace[j][0] - SecureField::from(init_limbs[j]));
        }
        // C4: final boundary ×9
        let final_limbs = super::super::air::felt252_to_limbs(&rp.final_digest);
        for j in 0..9usize {
            cairo = cairo * random_coeff + denom_inv * is_last * (trace[9 + j][0] - SecureField::from(final_limbs[j]));
        }
        // C5k: k boolean
        cairo = cairo * random_coeff + denom_inv * is_chain * add_k * (add_k - one);
        // C5c: carry booleans ×8
        for j in 0..8usize {
            let cj = trace[36 + j][0];
            cairo = cairo * random_coeff + denom_inv * is_chain * cj * (cj - one);
        }
        // C5: carry chain ×9
        let p_28: [u32; 9] = [1, 0, 0, 0, 0, 0, 16777216, 1, 134217728];
        let two28 = SecureField::from(M31::from(1u32 << 28));
        for j in 0..9usize {
            let da = trace[9 + j][0];
            let add = trace[27 + j][0];
            let snb = trace[18 + j][0];
            let pj = SecureField::from(M31::from(p_28[j]));
            let cin = if j == 0 { zero_sf } else { trace[36 + j - 1][0] };
            let cout = if j < 8 { trace[36 + j][0] * two28 } else { zero_sf };
            cairo = cairo * random_coeff + denom_inv * is_chain * (da + add + cin - snb - add_k * pj - cout);
        }

        eprintln!("cairo_eval: {:?}", cairo);
        eprintln!("MATCH: {}", rust_eval == cairo);

        if rust_eval != cairo {
            // Find first diverging constraint
            let mut rust_acc = PointEvaluationAccumulator::new(random_coeff);
            let mut cairo_terms: Vec<SecureField> = Vec::new();

            // We can't easily get per-constraint values from FrameworkComponent.
            // But we know they should match. Let me print the first few constraint values.
            eprintln!("\nPer-constraint Cairo values:");
            eprintln!("  C1 (is_active bool): {:?}", denom_inv * is_active * (one - is_active));
            eprintln!("  C2 (accumulator):    {:?}", denom_inv * (ac_next - ac - is_active + corr));
            eprintln!("  C3[0] (init bnd):    {:?}", denom_inv * is_first * (trace[0][0] - SecureField::from(init_limbs[0])));
        }
    }

    #[test]
    fn test_adversarial_tampered_io_commitment_rejected() {
        // Relabeling attack: same proof body, different io_commitment.
        // Fiat-Shamir channel binding causes FRI divergence → rejection.
        let rp = adversarial_proof();

        // Valid proof passes
        let ok = verify_recursive(
            &rp.stark_proof,
            &rp.public_inputs,
            rp.pass1_final_digest,
            rp.n_real_rows,
            rp.log_size,
            rp.final_digest,
            rp.logup_claimed_sum,
        );
        assert!(ok.is_ok(), "valid proof must verify");

        // Tampered io_commitment → MUST fail
        let tampered = RecursivePublicInputs {
            io_commitment: QM31(
                CM31(M31::from(999), M31::from(888)),
                CM31(M31::from(777), M31::from(666)),
            ),
            ..rp.public_inputs
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.pass1_final_digest, rp.n_real_rows, rp.log_size, rp.final_digest, rp.logup_claimed_sum);
        assert!(
            err.is_err(),
            "SECURITY: tampered io_commitment MUST be rejected"
        );
        eprintln!("[adversarial] io_commitment tampering rejected ✓");
    }

    #[test]
    fn test_adversarial_tampered_n_layers_rejected() {
        // n_layers tampering: different layer count with same proof body.
        let rp = adversarial_proof();

        let tampered = RecursivePublicInputs {
            n_layers: rp.public_inputs.n_layers + 100,
            ..rp.public_inputs
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.pass1_final_digest, rp.n_real_rows, rp.log_size, rp.final_digest, rp.logup_claimed_sum);
        assert!(err.is_err(), "SECURITY: tampered n_layers MUST be rejected");
        eprintln!("[adversarial] n_layers tampering rejected ✓");
    }

    #[test]
    fn test_adversarial_tampered_weight_super_root_rejected() {
        let rp = adversarial_proof();

        let tampered = RecursivePublicInputs {
            weight_super_root: QM31(
                CM31(M31::from(42), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            ..rp.public_inputs
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.pass1_final_digest, rp.n_real_rows, rp.log_size, rp.final_digest, rp.logup_claimed_sum);
        assert!(
            err.is_err(),
            "SECURITY: tampered weight_super_root MUST be rejected"
        );
        eprintln!("[adversarial] weight_super_root tampering rejected ✓");
    }

    #[test]
    fn test_adversarial_tampered_circuit_hash_rejected() {
        let rp = adversarial_proof();

        let tampered = RecursivePublicInputs {
            circuit_hash: QM31(
                CM31(M31::from(111), M31::from(222)),
                CM31(M31::from(333), M31::from(444)),
            ),
            ..rp.public_inputs
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.pass1_final_digest, rp.n_real_rows, rp.log_size, rp.final_digest, rp.logup_claimed_sum);
        assert!(
            err.is_err(),
            "SECURITY: tampered circuit_hash MUST be rejected"
        );
        eprintln!("[adversarial] circuit_hash tampering rejected ✓");
    }

    #[test]
    fn test_adversarial_full_metadata_relabeling() {
        // Full relabeling attack: change ALL metadata fields
        // while keeping the proof body unchanged.
        let rp = adversarial_proof();

        let tampered = RecursivePublicInputs {
            circuit_hash: QM31(
                CM31(M31::from(1), M31::from(2)),
                CM31(M31::from(3), M31::from(4)),
            ),
            io_commitment: QM31(
                CM31(M31::from(5), M31::from(6)),
                CM31(M31::from(7), M31::from(8)),
            ),
            weight_super_root: QM31(
                CM31(M31::from(9), M31::from(10)),
                CM31(M31::from(11), M31::from(12)),
            ),
            n_layers: 337,
            n_poseidon_perms: 9999,
            seed_digest: QM31::default(),
            hades_commitment: starknet_ff::FieldElement::ZERO,
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.pass1_final_digest, rp.n_real_rows, rp.log_size, rp.final_digest, rp.logup_claimed_sum);
        assert!(
            err.is_err(),
            "SECURITY: fully tampered metadata MUST be rejected"
        );

        // Original metadata still passes
        let ok = verify_recursive(
            &rp.stark_proof,
            &rp.public_inputs,
            rp.pass1_final_digest,
            rp.n_real_rows,
            rp.log_size,
            rp.final_digest,
            rp.logup_claimed_sum,
        );
        assert!(
            ok.is_ok(),
            "original metadata must still verify after adversarial test"
        );
        eprintln!("[adversarial] full metadata relabeling blocked ✓");
    }

    #[test]
    fn test_adversarial_trace_miniaturization_rejected() {
        // VULNERABILITY: Trace miniaturization attack.
        // An attacker claims a different n_poseidon_perms to make the trace
        // trivially small. Without the n_poseidon_perms channel binding, this
        // would produce a valid STARK proof without running the GKR verifier.
        let rp = adversarial_proof();

        // Tampered: claim only 2 Poseidon perms (trivially small chain)
        let tampered = RecursivePublicInputs {
            n_poseidon_perms: 2,
            ..rp.public_inputs
        };
        let err = verify_recursive(
            &rp.stark_proof, &tampered, rp.pass1_final_digest, rp.n_real_rows, rp.log_size, rp.final_digest, rp.logup_claimed_sum,
        );
        assert!(
            err.is_err(),
            "SECURITY: miniaturized n_poseidon_perms MUST be rejected"
        );

        // Also test inflated count
        let tampered = RecursivePublicInputs {
            n_poseidon_perms: rp.public_inputs.n_poseidon_perms + 1000,
            ..rp.public_inputs
        };
        let err = verify_recursive(
            &rp.stark_proof, &tampered, rp.pass1_final_digest, rp.n_real_rows, rp.log_size, rp.final_digest, rp.logup_claimed_sum,
        );
        assert!(
            err.is_err(),
            "SECURITY: inflated n_poseidon_perms MUST be rejected"
        );
        eprintln!("[adversarial] trace miniaturization attack blocked ✓");
    }

    #[test]
    fn test_adversarial_hades_tampered_witness_detected() {
        // Hades permutation integrity: tampered witness must be rejected.
        use crate::recursive::types::WitnessOp;

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

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights).unwrap();
        let gkr = proof.gkr_proof.as_ref().unwrap();
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).unwrap();

        let zero = QM31(
            CM31(M31::from(0), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );
        let mut witness = crate::recursive::generate_witness(
            &circuit,
            gkr,
            &proof.execution.output,
            Some(&weights),
            zero,
            zero,
        )
        .unwrap();

        // Honest witness passes
        let n = crate::recursive::verify_hades_perms_offline(&witness).unwrap();
        assert!(n > 0);

        // Tamper a HadesPerm output
        for op in witness.ops.iter_mut() {
            if let WitnessOp::HadesPerm { output, .. } = op {
                output[0] = starknet_ff::FieldElement::from(12345u64);
                break;
            }
        }
        let err = crate::recursive::verify_hades_perms_offline(&witness);
        assert!(
            err.is_err(),
            "SECURITY: tampered Hades permutation MUST be detected"
        );
        eprintln!("[adversarial] Hades witness tampering detected ✓");
    }
}
