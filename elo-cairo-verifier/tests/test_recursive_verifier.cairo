use snforge_std::{declare, DeclareResultTrait, ContractClassTrait, start_cheat_caller_address, stop_cheat_caller_address};
use starknet::ContractAddress;
use elo_cairo_verifier::recursive_verifier::{
    IRecursiveVerifierDispatcher, IRecursiveVerifierDispatcherTrait,
};

const OWNER_ADDR: felt252 = 0x1234;
const ATTACKER_ADDR: felt252 = 0xBAD;
const MODEL_ID: felt252 = 0xABC;
const CIRCUIT_HASH: felt252 = 0x123456;
const WEIGHT_ROOT: felt252 = 0x789ABC;
const IO_COMMITMENT: felt252 = 0xDEF123;

fn deploy_verifier() -> IRecursiveVerifierDispatcher {
    let contract = declare("RecursiveVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    let (address, _) = contract.deploy(@array![owner.into()]).unwrap();
    IRecursiveVerifierDispatcher { contract_address: address }
}

fn as_owner(verifier: @IRecursiveVerifierDispatcher) {
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(*verifier.contract_address, owner);
}

/// Build a fake proof header (25 felts) with specified circuit_hash and weight_root.
/// Values are placed in the low limb (ch3/wr3) for simplicity.
/// This proof will fail at STARK verification but is sufficient to test
/// pre-STARK asserts (circuit hash, weight binding, io commitment).
fn build_fake_proof(circuit_hash: felt252, io_commit: felt252, weight_root: felt252) -> Array<felt252> {
    let mut data: Array<felt252> = array![];
    // circuit_hash: QM31 as 4 M31 limbs [ch0, ch1, ch2, ch3]
    data.append(0); data.append(0); data.append(0); data.append(circuit_hash);
    // io_commitment: QM31 as 4 M31 limbs
    data.append(0); data.append(0); data.append(0); data.append(io_commit);
    // weight_super_root: QM31 as 4 M31 limbs
    data.append(0); data.append(0); data.append(0); data.append(weight_root);
    // n_layers (u32), verified (u32)
    data.append(30); data.append(1);
    // final_digest (felt252), log_size (u32)
    data.append(0x1234); data.append(14);
    // Padding to reach >= 20 felts + some extra for deserialize attempt
    data.append(0); data.append(0); data.append(0); data.append(0);
    data.append(0); data.append(0); data.append(0); data.append(0);
    data.append(0);
    data
}

// ═══════════════════════════════════════════════════════════════
// GROUP A: Registration CRUD
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_register_recursive_model() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.register_model_recursive(MODEL_ID, CIRCUIT_HASH, WEIGHT_ROOT);

    let info = verifier.get_recursive_model_info(MODEL_ID);
    assert!(info.circuit_hash == CIRCUIT_HASH, "circuit_hash mismatch");
    assert!(info.weight_super_root == WEIGHT_ROOT, "weight_root mismatch");

    let count = verifier.get_recursive_verification_count(MODEL_ID);
    assert!(count == 0, "count should start at 0");
}

#[test]
#[should_panic(expected: 'Only owner can register')]
fn test_register_recursive_model_non_owner_rejected() {
    let verifier = deploy_verifier();
    let attacker: ContractAddress = ATTACKER_ADDR.try_into().unwrap();
    start_cheat_caller_address(verifier.contract_address, attacker);

    verifier.register_model_recursive(MODEL_ID, CIRCUIT_HASH, WEIGHT_ROOT);
}

#[test]
fn test_query_unregistered_model_returns_zero() {
    let verifier = deploy_verifier();
    let info = verifier.get_recursive_model_info(0xDEAD);
    assert!(info.circuit_hash == 0, "unregistered model should have circuit_hash=0");
}

#[test]
fn test_is_recursive_proof_verified_default_false() {
    let verifier = deploy_verifier();
    let verified = verifier.is_recursive_proof_verified(0x999);
    assert!(!verified, "should default to false");
}

#[test]
fn test_verification_count_default_zero() {
    let verifier = deploy_verifier();
    let count = verifier.get_recursive_verification_count(0xDEAD);
    assert!(count == 0, "unregistered model count should be 0");
}

// ═══════════════════════════════════════════════════════════════
// GROUP A2: Registration Edge Cases
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_register_multiple_models() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.register_model_recursive(0x1, 0xAA, 0xBB);
    verifier.register_model_recursive(0x2, 0xCC, 0xDD);

    let info1 = verifier.get_recursive_model_info(0x1);
    let info2 = verifier.get_recursive_model_info(0x2);

    assert!(info1.circuit_hash == 0xAA, "model 1 circuit_hash wrong");
    assert!(info2.circuit_hash == 0xCC, "model 2 circuit_hash wrong");
    assert!(info1.weight_super_root == 0xBB, "model 1 weight_root wrong");
    assert!(info2.weight_super_root == 0xDD, "model 2 weight_root wrong");
}

#[test]
fn test_re_register_model_overwrites() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.register_model_recursive(MODEL_ID, 0x111, 0x222);
    verifier.register_model_recursive(MODEL_ID, 0x333, 0x444);

    let info = verifier.get_recursive_model_info(MODEL_ID);
    assert!(info.circuit_hash == 0x333, "re-register should overwrite circuit_hash");
    assert!(info.weight_super_root == 0x444, "re-register should overwrite weight_root");
}

#[test]
fn test_register_model_owner_stored_correctly() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.register_model_recursive(MODEL_ID, CIRCUIT_HASH, WEIGHT_ROOT);

    let info = verifier.get_recursive_model_info(MODEL_ID);
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    assert!(info.owner == owner, "owner should be the registrar");
}

#[test]
fn test_register_with_zero_circuit_hash() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Zero circuit_hash is allowed at registration time
    // but verify_recursive will reject it with "Model not registered"
    // because the check is `model.circuit_hash != 0`
    verifier.register_model_recursive(MODEL_ID, 0, WEIGHT_ROOT);

    let info = verifier.get_recursive_model_info(MODEL_ID);
    assert!(info.circuit_hash == 0, "zero circuit_hash should be stored");
}

// ═══════════════════════════════════════════════════════════════
// GROUP B: Pre-STARK Adversarial Rejection Tests
// ═══════════════════════════════════════════════════════════════

#[test]
#[should_panic(expected: 'Model not registered')]
fn test_verify_unregistered_model_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Submit proof for model that was never registered
    let fake_proof = build_fake_proof(CIRCUIT_HASH, IO_COMMITMENT, WEIGHT_ROOT);
    verifier.verify_recursive(0xDEAD, IO_COMMITMENT, fake_proof);
}

#[test]
#[should_panic(expected: "Proof too short")]
fn test_verify_proof_too_short() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.register_model_recursive(MODEL_ID, CIRCUIT_HASH, WEIGHT_ROOT);

    // Submit proof with only 10 felts (need >= 20)
    let short_proof: Array<felt252> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    verifier.verify_recursive(MODEL_ID, IO_COMMITMENT, short_proof);
}

#[test]
#[should_panic(expected: 'Circuit hash mismatch')]
fn test_verify_circuit_hash_mismatch() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Register model with CIRCUIT_HASH
    verifier.register_model_recursive(MODEL_ID, CIRCUIT_HASH, WEIGHT_ROOT);

    // Build proof with DIFFERENT circuit_hash (0xBADBAD instead of 0x123456)
    let tampered_proof = build_fake_proof(0xBADBAD, IO_COMMITMENT, WEIGHT_ROOT);
    verifier.verify_recursive(MODEL_ID, IO_COMMITMENT, tampered_proof);
}

#[test]
#[should_panic(expected: 'Weight binding mismatch')]
fn test_verify_weight_binding_mismatch() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Register model with WEIGHT_ROOT
    verifier.register_model_recursive(MODEL_ID, CIRCUIT_HASH, WEIGHT_ROOT);

    // Build proof with correct circuit_hash but WRONG weight_root
    let tampered_proof = build_fake_proof(CIRCUIT_HASH, IO_COMMITMENT, 0xBADBAD);
    verifier.verify_recursive(MODEL_ID, IO_COMMITMENT, tampered_proof);
}

// NOTE: io_commitment is NOT directly comparable between the parameter (Poseidon hash)
// and the proof header (QM31 limbs). The STARK proof binds IO through Fiat-Shamir.
// A separate io_commitment mismatch test would require matching encodings.

#[test]
#[should_panic(expected: "Proof too short")]
fn test_verify_empty_proof_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);
    verifier.register_model_recursive(MODEL_ID, CIRCUIT_HASH, WEIGHT_ROOT);

    let empty_proof: Array<felt252> = array![];
    verifier.verify_recursive(MODEL_ID, IO_COMMITMENT, empty_proof);
}

#[test]
#[should_panic(expected: "Proof too short")]
fn test_verify_proof_boundary_19_felts_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);
    verifier.register_model_recursive(MODEL_ID, CIRCUIT_HASH, WEIGHT_ROOT);

    // Exactly 19 felts — one short of the 20 minimum
    let mut short: Array<felt252> = array![];
    let mut i: u32 = 0;
    while i < 19 {
        short.append(0);
        i += 1;
    };
    verifier.verify_recursive(MODEL_ID, IO_COMMITMENT, short);
}

#[test]
#[should_panic(expected: 'Model not registered')]
fn test_verify_model_with_zero_circuit_hash_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Register model with circuit_hash=0, then try to verify
    // The verify_recursive check is `model.circuit_hash != 0`
    // so this should be rejected even though the model was "registered"
    verifier.register_model_recursive(MODEL_ID, 0, WEIGHT_ROOT);

    let fake_proof = build_fake_proof(0, IO_COMMITMENT, WEIGHT_ROOT);
    verifier.verify_recursive(MODEL_ID, IO_COMMITMENT, fake_proof);
}

#[test]
#[should_panic(expected: 'Circuit hash mismatch')]
fn test_verify_swapped_models_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    // Register two models with different circuit hashes
    verifier.register_model_recursive(0x1, 0xAAA, WEIGHT_ROOT);
    verifier.register_model_recursive(0x2, 0xBBB, WEIGHT_ROOT);

    // Try to verify model 0x1 with model 0x2's circuit hash
    let wrong_proof = build_fake_proof(0xBBB, IO_COMMITMENT, WEIGHT_ROOT);
    verifier.verify_recursive(0x1, IO_COMMITMENT, wrong_proof);
}

// ═══════════════════════════════════════════════════════════════
// GROUP C: Real STARK Proof Tests (require test data from A10G)
// These tests are commented out until test_recursive_data.cairo
// is generated from a real proof run.
// ═══════════════════════════════════════════════════════════════

use super::test_recursive_data;

#[test]
#[ignore]  // Requires --max-n-steps 500000000 (STARK verification is expensive)
fn test_verify_recursive_proof_valid() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.register_model_recursive(
        test_recursive_data::smollm2_model_id(),
        test_recursive_data::smollm2_circuit_hash(),
        test_recursive_data::smollm2_weight_root(),
    );

    let result = verifier.verify_recursive(
        test_recursive_data::smollm2_model_id(),
        test_recursive_data::smollm2_io_commitment(),
        test_recursive_data::smollm2_calldata(),
    );
    assert!(result, "proof should be valid");

    let count = verifier.get_recursive_verification_count(
        test_recursive_data::smollm2_model_id()
    );
    assert!(count == 1, "count should be 1 after verification");
}

#[test]
#[ignore]  // Requires --max-n-steps 500000000
#[should_panic(expected: 'Already verified')]
fn test_verify_proof_replay_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.register_model_recursive(
        test_recursive_data::smollm2_model_id(),
        test_recursive_data::smollm2_circuit_hash(),
        test_recursive_data::smollm2_weight_root(),
    );

    verifier.verify_recursive(
        test_recursive_data::smollm2_model_id(),
        test_recursive_data::smollm2_io_commitment(),
        test_recursive_data::smollm2_calldata(),
    );

    // Second submission → should panic
    verifier.verify_recursive(
        test_recursive_data::smollm2_model_id(),
        test_recursive_data::smollm2_io_commitment(),
        test_recursive_data::smollm2_calldata(),
    );
}

#[test]
#[ignore]  // Requires --max-n-steps 500000000
#[should_panic]  // STARK verification rejects tampered proof
fn test_verify_bit_flip_rejected() {
    let verifier = deploy_verifier();
    as_owner(@verifier);

    verifier.register_model_recursive(
        test_recursive_data::smollm2_model_id(),
        test_recursive_data::smollm2_circuit_hash(),
        test_recursive_data::smollm2_weight_root(),
    );

    let real = test_recursive_data::smollm2_calldata();
    let mut tampered: Array<felt252> = array![];
    let real_span = real.span();
    let mut i: u32 = 0;
    loop {
        if i >= real_span.len() { break; }
        if i == 20 {
            tampered.append(*real_span.at(i) + 0xDEAD);
        } else {
            tampered.append(*real_span.at(i));
        }
        i += 1;
    };

    verifier.verify_recursive(
        test_recursive_data::smollm2_model_id(),
        test_recursive_data::smollm2_io_commitment(),
        tampered,
    );
}
