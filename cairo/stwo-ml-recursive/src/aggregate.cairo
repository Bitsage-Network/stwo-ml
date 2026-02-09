/// Core aggregation logic: verify all matmul proofs + layer chain,
/// then compute a single aggregate hash.
///
/// This runs inside the Cairo VM (not on-chain). The stwo-cairo-prover
/// then proves this execution into a compact Circle STARK.

use core::poseidon::poseidon_hash_span;
use stwo_ml_verify_core::sumcheck::verify_matmul_sumcheck;
use stwo_ml_verify_core::layer_chain::{verify_layer_chain, compute_chain_commitment};
use super::types::RecursiveInput;

/// Domain separator for the aggregate hash.
const AGGREGATE_DOMAIN: felt252 = 'OBELYSK_RECURSIVE_V1';

/// Verify all matmul proofs and the layer chain, returning the aggregate hash
/// and verification counts.
///
/// Panics if any proof is invalid or the chain is broken.
///
/// Returns: (aggregate_hash, num_verified, num_layers)
pub fn verify_all_and_aggregate(input: RecursiveInput) -> (felt252, u32, u32) {
    let RecursiveInput {
        model_id,
        model_commitment,
        io_commitment,
        num_matmul_proofs,
        mut matmul_proofs,
        layer_headers,
        model_input_commitment,
        model_output_commitment,
        tee_report_hash,
    } = input;

    // 1. Validate proof count
    assert!(
        matmul_proofs.len() == num_matmul_proofs,
        "Proof count mismatch: expected {}, got {}",
        num_matmul_proofs,
        matmul_proofs.len(),
    );

    // 2. Verify each matmul sumcheck proof, collecting proof hashes.
    //    Use pop_front() to consume Array ownership (MatMulSumcheckProof is non-Copy).
    let mut proof_hashes: Array<felt252> = array![];
    let mut verified: u32 = 0;

    loop {
        match matmul_proofs.pop_front() {
            Option::Some(proof) => {
                let (is_valid, proof_hash) = verify_matmul_sumcheck(proof);
                assert!(is_valid, "Matmul proof {} failed verification", verified);
                proof_hashes.append(proof_hash);
                verified += 1;
            },
            Option::None => { break; },
        }
    };

    // 3. Verify the layer commitment chain
    let num_layers: u32 = layer_headers.len();
    let chain_result = verify_layer_chain(
        layer_headers.span(),
        model_input_commitment,
        model_output_commitment,
    );
    assert!(
        chain_result.is_valid,
        "Layer chain broken at index {}",
        chain_result.broken_at,
    );

    // 4. Compute chain commitment
    let chain_commitment = compute_chain_commitment(layer_headers.span());

    // 5. Compute aggregate hash:
    //    Poseidon(domain, model_id, model_commitment, io_commitment,
    //             chain_commitment, tee_hash, proof_hash_0, proof_hash_1, ...)
    let mut hash_inputs: Array<felt252> = array![
        AGGREGATE_DOMAIN,
        model_id,
        model_commitment,
        io_commitment,
        chain_commitment,
        tee_report_hash,
    ];

    let proof_hashes_span = proof_hashes.span();
    let mut i: u32 = 0;
    loop {
        if i >= proof_hashes_span.len() {
            break;
        }
        hash_inputs.append(*proof_hashes_span.at(i));
        i += 1;
    };

    let aggregate_hash = poseidon_hash_span(hash_inputs.span());

    (aggregate_hash, verified, num_layers)
}
