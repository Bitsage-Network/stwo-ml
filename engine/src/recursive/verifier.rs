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
use stwo::core::vcs_lifted::poseidon252_merkle::{Poseidon252MerkleChannel, Poseidon252MerkleHasher};
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
    log_size: u32,
    final_digest: starknet_ff::FieldElement,
) -> Result<(), RecursiveError> {
    let pcs_config = PcsConfig::default();

    // Build evaluator from public inputs
    let zero_limbs = super::air::felt252_to_limbs(&starknet_ff::FieldElement::ZERO);
    let eval = RecursiveVerifierEval {
        log_n_rows: log_size,
        initial_digest_limbs: zero_limbs,
        final_digest_limbs: super::air::felt252_to_limbs(&final_digest),
        hades_lookup: None, // LogUp disabled until multi-component STARK is wired
    };

    // Build dummy component to get trace_log_degree_bounds
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component =
        FrameworkComponent::new(&mut allocator, eval.clone(), SecureField::zero());
    let bounds = Component::trace_log_degree_bounds(&dummy_component);

    // Set up channel and commitment scheme verifier
    let channel = &mut <Poseidon252MerkleChannel as MerkleChannel>::C::default();

    // Mix PcsConfig into channel (must match prover's individual mix_u64 calls)
    channel.mix_u64(pcs_config.pow_bits as u64);
    channel.mix_u64(pcs_config.fri_config.log_blowup_factor as u64);
    channel.mix_u64(pcs_config.fri_config.n_queries as u64);
    channel.mix_u64(pcs_config.fri_config.log_last_layer_degree_bound as u64);

    // Mix public inputs into channel (must match prover's binding)
    channel.mix_felts(&[
        public_inputs.circuit_hash,
        public_inputs.io_commitment,
        public_inputs.weight_super_root,
    ]);
    channel.mix_u64(public_inputs.n_layers as u64);

    let mut commitment_scheme = CommitmentSchemeVerifier::<Poseidon252MerkleChannel>::new(pcs_config);

    // Replay commitments (Tree 0 = preprocessed, Tree 1 = execution)
    if stark_proof.commitments.len() < 2 {
        return Err(RecursiveError::ProvingFailed(format!(
            "expected at least 2 commitments, got {}",
            stark_proof.commitments.len()
        )));
    }
    commitment_scheme.commit(stark_proof.commitments[0], &bounds[0], channel);
    commitment_scheme.commit(stark_proof.commitments[1], &bounds[1], channel);

    // Build real component for verification
    let mut allocator = TraceLocationAllocator::default();
    let component = FrameworkComponent::new(&mut allocator, eval, SecureField::zero());

    // Verify
    stwo_verify::<Poseidon252MerkleChannel>(
        &[&component as &dyn Component],
        channel,
        &mut commitment_scheme,
        stark_proof.clone(),
    )
    .map_err(|e| RecursiveError::ProvingFailed(format!("recursive STARK verification failed: {e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{GraphBuilder, GraphWeights};
    use crate::components::matmul::M31Matrix;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::qm31::QM31;

    #[test]
    fn test_verify_recursive_roundtrip() {
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

        let zero = QM31(CM31(M31::from(0), M31::from(0)), CM31(M31::from(0), M31::from(0)));

        // Generate witness + build trace + prove
        let witness = crate::recursive::generate_witness(
            &circuit, gkr, &proof.execution.output, Some(&weights), zero, zero,
        )
        .expect("witness generation should succeed");

        let trace_data = crate::recursive::air::build_recursive_trace(&witness);
        let log_size = trace_data.log_size;

        // Prove recursive STARK
        let recursive_result = crate::recursive::prove_recursive(
            &circuit, gkr, &proof.execution.output, &weights, zero, zero, 0.0,
        );

        let recursive_proof = recursive_result.expect("recursive proving should succeed");
        eprintln!(
            "Recursive proof produced (log_size={}, {} poseidon perms, {:.3}s)",
            recursive_proof.log_size, witness.n_poseidon_perms,
            recursive_proof.metadata.recursive_prove_time_secs,
        );

        // Full roundtrip: verify the recursive STARK proof
        let verify_result = verify_recursive(
            &recursive_proof.stark_proof,
            &recursive_proof.public_inputs,
            recursive_proof.log_size,
            recursive_proof.final_digest,
        );
        assert!(verify_result.is_ok(), "recursive verification should succeed: {:?}", verify_result.err());
    }

    // ═══════════════════════════════════════════════════════════════
    // Adversarial tests — reproduce Omar Espejel's review findings
    // ═══════════════════════════════════════════════════════════════

    /// Helper: produce a valid recursive proof for adversarial testing.
    fn adversarial_proof() -> super::super::types::RecursiveProof {
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights)
            .expect("GKR proving should succeed");
        let gkr = proof.gkr_proof.as_ref().expect("should have GKR proof");
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).expect("circuit compile");

        let zero = QM31(CM31(M31::from(0), M31::from(0)), CM31(M31::from(0), M31::from(0)));
        crate::recursive::prove_recursive(
            &circuit, gkr, &proof.execution.output, &weights, zero, zero, 0.0,
        ).expect("recursive proving should succeed")
    }

    #[test]
    fn test_adversarial_tampered_io_commitment_rejected() {
        // Omar's Finding 1: same proof body, different io_commitment.
        // Before fix (commit c3071846): starknet_call returned success (0x1).
        // After fix (commit 618fcc79): Fiat-Shamir channel diverges → FRI fails.
        let rp = adversarial_proof();

        // Valid proof passes
        let ok = verify_recursive(&rp.stark_proof, &rp.public_inputs, rp.log_size, rp.final_digest);
        assert!(ok.is_ok(), "valid proof must verify");

        // Tampered io_commitment → MUST fail
        let tampered = RecursivePublicInputs {
            io_commitment: QM31(CM31(M31::from(999), M31::from(888)), CM31(M31::from(777), M31::from(666))),
            ..rp.public_inputs
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.log_size, rp.final_digest);
        assert!(err.is_err(), "SECURITY: tampered io_commitment MUST be rejected");
        eprintln!("[adversarial] io_commitment tampering rejected ✓");
    }

    #[test]
    fn test_adversarial_tampered_n_layers_rejected() {
        // Omar specifically changed n_layers in his test.
        let rp = adversarial_proof();

        let tampered = RecursivePublicInputs {
            n_layers: rp.public_inputs.n_layers + 100,
            ..rp.public_inputs
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.log_size, rp.final_digest);
        assert!(err.is_err(), "SECURITY: tampered n_layers MUST be rejected");
        eprintln!("[adversarial] n_layers tampering rejected ✓");
    }

    #[test]
    fn test_adversarial_tampered_weight_super_root_rejected() {
        let rp = adversarial_proof();

        let tampered = RecursivePublicInputs {
            weight_super_root: QM31(CM31(M31::from(42), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            ..rp.public_inputs
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.log_size, rp.final_digest);
        assert!(err.is_err(), "SECURITY: tampered weight_super_root MUST be rejected");
        eprintln!("[adversarial] weight_super_root tampering rejected ✓");
    }

    #[test]
    fn test_adversarial_tampered_circuit_hash_rejected() {
        let rp = adversarial_proof();

        let tampered = RecursivePublicInputs {
            circuit_hash: QM31(CM31(M31::from(111), M31::from(222)), CM31(M31::from(333), M31::from(444))),
            ..rp.public_inputs
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.log_size, rp.final_digest);
        assert!(err.is_err(), "SECURITY: tampered circuit_hash MUST be rejected");
        eprintln!("[adversarial] circuit_hash tampering rejected ✓");
    }

    #[test]
    fn test_adversarial_omar_full_scenario() {
        // Reproduce Omar's exact test: change ALL metadata fields
        // while keeping the proof body unchanged.
        let rp = adversarial_proof();

        let tampered = RecursivePublicInputs {
            circuit_hash: QM31(CM31(M31::from(1), M31::from(2)), CM31(M31::from(3), M31::from(4))),
            io_commitment: QM31(CM31(M31::from(5), M31::from(6)), CM31(M31::from(7), M31::from(8))),
            weight_super_root: QM31(CM31(M31::from(9), M31::from(10)), CM31(M31::from(11), M31::from(12))),
            n_layers: 337,
            verified: true,
        };
        let err = verify_recursive(&rp.stark_proof, &tampered, rp.log_size, rp.final_digest);
        assert!(err.is_err(), "SECURITY: fully tampered metadata MUST be rejected");

        // Original metadata still passes
        let ok = verify_recursive(&rp.stark_proof, &rp.public_inputs, rp.log_size, rp.final_digest);
        assert!(ok.is_ok(), "original metadata must still verify after adversarial test");
        eprintln!("[adversarial] Omar's full relabeling scenario: blocked ✓");
    }

    #[test]
    fn test_adversarial_hades_tampered_witness_detected() {
        // Omar's Finding 2: verify Hades permutation integrity.
        use crate::recursive::types::WitnessOp;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 { input.set(0, j, M31::from((j + 1) as u32)); }

        let mut weights = GraphWeights::new();
        let mut w = M31Matrix::new(4, 2);
        for i in 0..4 { for j in 0..2 { w.set(i, j, M31::from((i * 2 + j + 1) as u32)); } }
        weights.add_weight(0, w);

        let proof = crate::aggregation::prove_model_pure_gkr(&graph, &input, &weights).unwrap();
        let gkr = proof.gkr_proof.as_ref().unwrap();
        let circuit = crate::gkr::LayeredCircuit::from_graph(&graph).unwrap();

        let zero = QM31(CM31(M31::from(0), M31::from(0)), CM31(M31::from(0), M31::from(0)));
        let mut witness = crate::recursive::generate_witness(
            &circuit, gkr, &proof.execution.output, Some(&weights), zero, zero,
        ).unwrap();

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
        assert!(err.is_err(), "SECURITY: tampered Hades permutation MUST be detected");
        eprintln!("[adversarial] Hades witness tampering detected ✓");
    }
}
