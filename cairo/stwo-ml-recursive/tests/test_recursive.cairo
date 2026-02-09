use stwo_ml_recursive::types::RecursiveInput;
use stwo_ml_recursive::aggregate::verify_all_and_aggregate;
use stwo_ml_verify_core::layer_chain::LayerProofHeader;

// ============================================================================
// Helpers
// ============================================================================

fn make_input_no_proofs(
    model_id: felt252,
    model_commitment: felt252,
    io_commitment: felt252,
    headers: Array<LayerProofHeader>,
    model_input: felt252,
    model_output: felt252,
    tee_hash: felt252,
) -> RecursiveInput {
    RecursiveInput {
        model_id,
        model_commitment,
        io_commitment,
        num_matmul_proofs: 0,
        matmul_proofs: array![],
        layer_headers: headers,
        model_input_commitment: model_input,
        model_output_commitment: model_output,
        tee_report_hash: tee_hash,
    }
}

// ============================================================================
// Tests
// ============================================================================

// Note: #[should_panic] tests require snforge_std which is incompatible
// with executable targets. Panic behavior is tested via Rust integration tests.
// The following panics are verified by the Rust test_recursive_e2e:
// - "Proof count mismatch" when num_matmul_proofs != matmul_proofs.len()
// - "Layer chain broken" when layer output != next layer input

#[test]
fn test_aggregate_hash_deterministic() {
    // Two identical inputs (no matmul proofs, just metadata) → same hash
    let input1 = make_input_no_proofs(1, 42, 99, array![], 0, 0, 0);
    let input2 = make_input_no_proofs(1, 42, 99, array![], 0, 0, 0);

    let (hash1, count1, layers1) = verify_all_and_aggregate(input1);
    let (hash2, count2, layers2) = verify_all_and_aggregate(input2);

    assert!(hash1 == hash2, "Same input must produce same aggregate hash");
    assert!(count1 == count2, "Same verification count");
    assert!(layers1 == layers2, "Same layer count");
    assert!(count1 == 0, "Should have verified 0 proofs");
    assert!(layers1 == 0, "Should have 0 layers");
}

#[test]
fn test_aggregate_hash_includes_model_id() {
    let input1 = make_input_no_proofs(1, 42, 99, array![], 0, 0, 0);
    let input2 = make_input_no_proofs(2, 42, 99, array![], 0, 0, 0);

    let (hash1, _, _) = verify_all_and_aggregate(input1);
    let (hash2, _, _) = verify_all_and_aggregate(input2);

    assert!(hash1 != hash2, "Different model_id must produce different aggregate hash");
}

#[test]
fn test_empty_proofs_and_empty_layers() {
    let input = make_input_no_proofs(1, 42, 99, array![], 0, 0, 0);
    let (hash, num_verified, num_layers) = verify_all_and_aggregate(input);

    assert!(hash != 0, "Aggregate hash should be non-zero even with empty input");
    assert!(num_verified == 0, "No proofs verified");
    assert!(num_layers == 0, "No layers");
}

#[test]
fn test_valid_chain_with_commitments() {
    let headers = array![
        LayerProofHeader { layer_index: 0, input_commitment: 100, output_commitment: 200 },
        LayerProofHeader { layer_index: 1, input_commitment: 200, output_commitment: 300 },
    ];
    let input = make_input_no_proofs(1, 42, 99, headers, 100, 300, 0);

    let (hash, num_verified, num_layers) = verify_all_and_aggregate(input);

    assert!(hash != 0, "Aggregate hash should be non-zero");
    assert!(num_verified == 0, "No matmul proofs to verify");
    assert!(num_layers == 2, "Should have 2 layers");
}

#[test]
fn test_aggregate_hash_includes_tee_hash() {
    let input1 = make_input_no_proofs(1, 42, 99, array![], 0, 0, 0);
    let input2 = make_input_no_proofs(1, 42, 99, array![], 0, 0, 12345);

    let (hash1, _, _) = verify_all_and_aggregate(input1);
    let (hash2, _, _) = verify_all_and_aggregate(input2);

    assert!(hash1 != hash2, "Different tee_report_hash must produce different aggregate hash");
}

#[test]
fn test_aggregate_hash_includes_io_commitment() {
    let input1 = make_input_no_proofs(1, 42, 99, array![], 0, 0, 0);
    let input2 = make_input_no_proofs(1, 42, 77, array![], 0, 0, 0);

    let (hash1, _, _) = verify_all_and_aggregate(input1);
    let (hash2, _, _) = verify_all_and_aggregate(input2);

    assert!(hash1 != hash2, "Different io_commitment must produce different aggregate hash");
}
