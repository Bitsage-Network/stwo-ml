/// Tests for A7: Contract-level GKR model verification.
///
/// Tests register_model_gkr() and verify_model_gkr() through the
/// ISumcheckVerifier dispatcher (full contract deploy + call cycle).
///
/// Note: verify_model_gkr() now recomputes IO commitment on-chain from
/// raw_io_data and evaluates MLE(output, r_out) + MLE(input, final_point).
/// Hand-crafted proofs cannot satisfy the MLE checks, so positive verification
/// tests require Rust-generated test vectors (see e2e_cairo_verify.rs).
/// The tests here focus on registration, access control, and early rejection.

use snforge_std::{declare, DeclareResultTrait, ContractClassTrait};
use starknet::ContractAddress;
use elo_cairo_verifier::field::{QM31, CM31, qm31_new, qm31_add};
use elo_cairo_verifier::verifier::{
    ISumcheckVerifierDispatcher, ISumcheckVerifierDispatcherTrait,
};

// ============================================================================
// Helpers
// ============================================================================

fn deploy_verifier() -> ISumcheckVerifierDispatcher {
    let contract = declare("SumcheckVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    let (address, _) = contract.deploy(@array![owner.into()]).unwrap();
    ISumcheckVerifierDispatcher { contract_address: address }
}

/// Real-only QM31.
fn mk(a: u64) -> QM31 {
    QM31 { a: CM31 { a, b: 0 }, b: CM31 { a: 0, b: 0 } }
}

/// Build a minimal raw_io_data array.
/// Layout: [in_rows, in_cols, in_len, in_data..., out_rows, out_cols, out_len, out_data...]
fn build_raw_io_data() -> Array<felt252> {
    array![
        1,          // in_rows
        4,          // in_cols
        4,          // in_len
        1, 2, 3, 4, // in_data
        1,          // out_rows
        2,          // out_cols
        2,          // out_len
        100, 200,   // out_data
    ]
}

// ============================================================================
// Test 1: Register GKR model and verify registration state
// ============================================================================

#[test]
fn test_register_and_verify_model_gkr() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    let model_id: felt252 = 0xABC;

    // Register model for GKR (no matmul layers -> no weight commitments)
    dispatcher.register_model_gkr(
        model_id,
        array![],                    // no weight commitments
        array![1],                   // circuit descriptor: [Add]
    );

    // Verify registration
    let circuit_hash = dispatcher.get_model_circuit_hash(model_id);
    assert!(circuit_hash != 0, "circuit hash should be set");

    let weight_count = dispatcher.get_model_gkr_weight_count(model_id);
    assert!(weight_count == 0, "no weight commitments");
}

// ============================================================================
// Test 2: Non-owner cannot register_model_gkr
// ============================================================================

#[test]
#[should_panic(expected: "Only owner")]
fn test_register_model_gkr_non_owner_rejected() {
    let dispatcher = deploy_verifier();
    let attacker: ContractAddress = 0xBAD_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, attacker);

    dispatcher.register_model_gkr(
        0xABC,
        array![0x111],
        array![0, 1],   // MatMul, Add
    );
}

// ============================================================================
// Test 3: Cannot verify unregistered model
// ============================================================================

#[test]
#[should_panic(expected: "Model not registered for GKR")]
fn test_verify_model_gkr_not_registered() {
    let dispatcher = deploy_verifier();

    dispatcher.verify_model_gkr(
        0xDEAD,                          // unregistered model_id
        build_raw_io_data(),             // raw_io_data
        1,                               // circuit_depth
        1,                               // num_layers
        array![],                        // matmul_dims
        array![],                        // dequantize_bits
        array![],                        // proof_data
        array![],                        // weight_commitments
        array![],                        // weight_opening_proofs
    );
}

// ============================================================================
// Test 4: Weight commitment mismatch rejected
// ============================================================================

#[test]
#[should_panic(expected: "Weight commitment mismatch")]
fn test_verify_model_gkr_weight_mismatch() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    let model_id: felt252 = 0xABC;

    // Register with weight commitments [0x111, 0x222]
    dispatcher.register_model_gkr(
        model_id,
        array![0x111, 0x222],
        array![0, 1],   // MatMul, Add
    );

    // Try to verify with WRONG weight commitments
    dispatcher.verify_model_gkr(
        model_id,
        build_raw_io_data(),
        1,                               // circuit_depth
        1,
        array![],
        array![],
        array![],
        array![0x111, 0x999],   // 0x999 != registered 0x222
        array![],                        // weight_opening_proofs
    );
}

// ============================================================================
// Test 5: Weight commitment count mismatch rejected
// ============================================================================

#[test]
#[should_panic(expected: "Weight commitment count mismatch")]
fn test_verify_model_gkr_weight_count_mismatch() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    let model_id: felt252 = 0xABC;

    // Register with 2 weight commitments
    dispatcher.register_model_gkr(
        model_id,
        array![0x111, 0x222],
        array![0, 1],
    );

    // Try to verify with only 1 weight commitment
    dispatcher.verify_model_gkr(
        model_id,
        build_raw_io_data(),
        1,                               // circuit_depth
        1,
        array![],
        array![],
        array![],
        array![0x111],   // count 1 != registered 2
        array![],                        // weight_opening_proofs
    );
}

// ============================================================================
// Test 6: Cannot re-register a model for GKR
// ============================================================================

#[test]
#[should_panic(expected: "Model already registered for GKR")]
fn test_register_model_gkr_duplicate_rejected() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    dispatcher.register_model_gkr(0xABC, array![], array![1]);

    // Try to register same model again
    dispatcher.register_model_gkr(0xABC, array![], array![1, 2]);
}

// ============================================================================
// Test 7: Empty raw_io_data rejected (replaces old "zero io_commitment" test)
// ============================================================================

#[test]
#[should_panic(expected: "IO_DATA_TOO_SHORT")]
fn test_verify_model_gkr_empty_io_data() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    dispatcher.register_model_gkr(0xABC, array![], array![1]);

    // Empty raw_io_data should fail validation
    dispatcher.verify_model_gkr(
        0xABC,
        array![],                        // empty raw_io_data
        1,                               // circuit_depth
        1,
        array![],
        array![],
        array![],
        array![],
        array![],                        // weight_opening_proofs
    );
}

// ============================================================================
// Test 8: Zero weight commitment in array rejected at registration
// ============================================================================

#[test]
#[should_panic(expected: "Weight commitment cannot be zero")]
fn test_register_model_gkr_zero_weight_rejected() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    dispatcher.register_model_gkr(
        0xABC,
        array![0x111, 0],   // second commitment is zero
        array![0, 0],
    );
}

// ============================================================================
// Test 9: GKR and legacy model registration are independent
// ============================================================================

#[test]
fn test_gkr_and_legacy_registration_independent() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    let model_id: felt252 = 0xABC;

    // Register legacy (no owner gate)
    dispatcher.register_model(model_id, 42);
    let legacy_commitment = dispatcher.get_model_commitment(model_id);
    assert!(legacy_commitment == 42, "legacy should be registered");

    // Register GKR (owner gated)
    dispatcher.register_model_gkr(model_id, array![0x111], array![0]);
    let circuit_hash = dispatcher.get_model_circuit_hash(model_id);
    assert!(circuit_hash != 0, "GKR should be registered");

    // Both coexist
    assert!(dispatcher.get_model_commitment(model_id) == 42, "legacy still works");
    assert!(dispatcher.get_model_gkr_weight_count(model_id) == 1, "GKR still works");
}

// ============================================================================
// Test 10: Short raw_io_data rejected (< 6 elements)
// ============================================================================

#[test]
#[should_panic(expected: "IO_DATA_TOO_SHORT")]
fn test_verify_model_gkr_short_io_data() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    dispatcher.register_model_gkr(0xABC, array![], array![1]);

    // raw_io_data with only 5 elements (needs >= 6)
    dispatcher.verify_model_gkr(
        0xABC,
        array![1, 2, 3, 4, 5],          // too short
        1,                               // circuit_depth
        1,
        array![],
        array![],
        array![],
        array![],
        array![],                        // weight_opening_proofs
    );
}

// ============================================================================
// Test 11: Circuit hash mismatch rejected (proof tags/depth must match registration)
// ============================================================================

#[test]
#[should_panic(expected: "CIRCUIT_HASH_MISMATCH")]
fn test_verify_model_gkr_circuit_hash_mismatch() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    let model_id: felt252 = 0xABC;

    // Register with descriptor [1]
    dispatcher.register_model_gkr(model_id, array![], array![1]);

    // Verify with a mismatched circuit_depth so descriptor hash diverges.
    dispatcher.verify_model_gkr(
        model_id,
        build_raw_io_data(),
        2,              // wrong circuit_depth
        0,              // no layer proofs
        array![],
        array![],
        array![0],     // proof_data: num_deferred = 0 (well-formed minimal payload)
        array![],
        array![],
    );
}
