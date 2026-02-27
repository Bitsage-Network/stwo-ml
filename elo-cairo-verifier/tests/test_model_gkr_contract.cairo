/// Tests for A7: Contract-level GKR model verification.
///
/// Tests register_model_gkr() and verify_model_gkr_v4_packed_io() through the
/// ISumcheckVerifier dispatcher (full contract deploy + call cycle).
///
/// Note: verify_model_gkr_v4_packed_io() processes packed IO data, runs the
/// full GKR model walk, and only checks weight/circuit bindings AFTER verification.
/// This means deep verification tests (weight mismatch, registration check, etc.)
/// require Rust-generated test vectors (see e2e_cairo_verify.rs).
/// The tests here focus on registration, access control, and early rejection.

use snforge_std::{declare, DeclareResultTrait, ContractClassTrait};
use starknet::ContractAddress;
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
// Test 3: Cannot re-register a model for GKR
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
// Test 4: Zero weight commitment in array rejected at registration
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
// Test 5: GKR and legacy model registration are independent
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
// Test 6: Unsupported weight binding mode rejected (earliest verify rejection)
// ============================================================================

/// With the IO-packed interface, UNSUPPORTED_WEIGHT_BINDING_MODE is the
/// first assert in verify_model_gkr_v4_packed_io_core (before IO hashing,
/// dimension extraction, or GKR walk). This is the earliest rejection point.
#[test]
#[should_panic(expected: "UNSUPPORTED_WEIGHT_BINDING_MODE")]
fn test_verify_model_gkr_unsupported_binding_mode() {
    let dispatcher = deploy_verifier();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    snforge_std::start_cheat_caller_address(dispatcher.contract_address, owner);

    dispatcher.register_model_gkr(0xABC, array![], array![1]);

    // Use invalid weight_binding_mode=0 — triggers earliest assert
    dispatcher.verify_model_gkr_v4_packed_io(
        0xABC,
        8,                               // original_io_len
        array![],                        // packed_raw_io
        1,                               // circuit_depth
        1,                               // num_layers
        array![],                        // matmul_dims
        array![],                        // dequantize_bits
        array![],                        // proof_data
        array![],                        // weight_commitments
        0,                               // weight_binding_mode = 0 (unsupported)
        array![],                        // weight_binding_data
        array![],                        // weight_opening_proofs
    );
}
