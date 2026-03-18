//! Rust-side verification of recursive STARK proofs.
//!
//! This is the pre-flight check before on-chain submission. It reconstructs
//! the AIR component from public inputs and calls `stwo::verify()`.

use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::channel::MerkleChannel;
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
    };

    // Build dummy component to get trace_log_degree_bounds
    let mut allocator = TraceLocationAllocator::default();
    let dummy_component =
        FrameworkComponent::new(&mut allocator, eval.clone(), SecureField::zero());
    let bounds = Component::trace_log_degree_bounds(&dummy_component);

    // Set up channel and commitment scheme verifier
    let channel = &mut <Poseidon252MerkleChannel as MerkleChannel>::C::default();
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
}
