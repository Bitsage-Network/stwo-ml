use snforge_std::{
    declare, ContractClassTrait, DeclareResultTrait,
    start_cheat_caller_address, stop_cheat_caller_address,
};
use starknet::{ContractAddress, contract_address_const};
use core::serde::Serde;

use stwo_ml_verifier::interfaces::{IObelyskVerifierDispatcher, IObelyskVerifierDispatcherTrait};
use sage_contracts::interfaces::sage_token::{ISAGETokenDispatcher, ISAGETokenDispatcherTrait};

// ── Constants ────────────────────────────────────────────────────────────

fn OWNER() -> ContractAddress {
    contract_address_const::<'OWNER'>()
}

fn WORKER() -> ContractAddress {
    contract_address_const::<'WORKER'>()
}

fn ATTACKER() -> ContractAddress {
    contract_address_const::<'ATTACKER'>()
}

fn JOB_MANAGER() -> ContractAddress {
    contract_address_const::<'JOB_MANAGER'>()
}

fn CDC_POOL() -> ContractAddress {
    contract_address_const::<'CDC_POOL'>()
}

fn PAYMASTER() -> ContractAddress {
    contract_address_const::<'PAYMASTER'>()
}

fn TREASURY() -> ContractAddress {
    contract_address_const::<'TREASURY'>()
}

fn TEAM() -> ContractAddress {
    contract_address_const::<'TEAM'>()
}

fn LIQUIDITY() -> ContractAddress {
    contract_address_const::<'LIQUIDITY'>()
}

// ── Deploy Helpers ───────────────────────────────────────────────────────

fn deploy_sage_token(owner: ContractAddress) -> ISAGETokenDispatcher {
    let contract = declare("SAGEToken").unwrap().contract_class();
    let mut calldata = array![];
    owner.serialize(ref calldata);
    JOB_MANAGER().serialize(ref calldata);
    CDC_POOL().serialize(ref calldata);
    PAYMASTER().serialize(ref calldata);
    TREASURY().serialize(ref calldata);
    TEAM().serialize(ref calldata);
    LIQUIDITY().serialize(ref calldata);
    let (addr, _) = contract.deploy(@calldata).unwrap();
    ISAGETokenDispatcher { contract_address: addr }
}

fn deploy_verifier(
    owner: ContractAddress, sage_token: ContractAddress,
) -> IObelyskVerifierDispatcher {
    let contract = declare("ObelyskVerifier").unwrap().contract_class();
    let mut calldata = array![];
    owner.serialize(ref calldata);
    sage_token.serialize(ref calldata);
    let (addr, _) = contract.deploy(@calldata).unwrap();
    IObelyskVerifierDispatcher { contract_address: addr }
}

fn setup() -> (IObelyskVerifierDispatcher, ISAGETokenDispatcher) {
    let sage = deploy_sage_token(OWNER());
    let verifier = deploy_verifier(OWNER(), sage.contract_address);
    (verifier, sage)
}

// ── Test Helpers ─────────────────────────────────────────────────────────

const MODEL_ID: felt252 = 'qwen3-14b';
const MODEL_ID_2: felt252 = 'llama-70b';
const PROOF_HASH: felt252 = 'proof_hash_001';
const IO_COMMIT: felt252 = 'io_commit_001';
const WEIGHT_COMMIT: felt252 = 'weight_commit_001';
const DESCRIPTION: felt252 = 'Qwen3-14B model';
const DESCRIPTION_2: felt252 = 'Llama-70B model';
const JOB_ID: felt252 = 'job_001';
const JOB_ID_2: felt252 = 'job_002';
const NUM_LAYERS: u32 = 40;

// TGE_PUBLIC_SALE = 10_000_000 * 10^18
const TGE_PUBLIC_SALE: u256 = 10_000_000_000_000_000_000_000_000;

// ══════════════════════════════════════════════════════════════════════════
// Constructor & Deployment (3 tests)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_deploy_verifier() {
    let (verifier, _sage) = setup();
    assert!(verifier.get_owner() == OWNER(), "Owner mismatch");
}

#[test]
fn test_get_sage_token() {
    let (verifier, sage) = setup();
    assert!(verifier.get_sage_token() == sage.contract_address, "SAGE token mismatch");
}

#[test]
fn test_initial_state() {
    let (verifier, _sage) = setup();
    assert!(!verifier.is_verified('random_proof_id'), "Should not be verified");
    assert!(verifier.get_model_verification_count('random_model') == 0, "Count should be 0");
}

// ══════════════════════════════════════════════════════════════════════════
// Model Registration (4 tests)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_register_model() {
    let (verifier, _sage) = setup();
    start_cheat_caller_address(verifier.contract_address, OWNER());
    verifier.register_model(MODEL_ID, WEIGHT_COMMIT, NUM_LAYERS, DESCRIPTION);
    stop_cheat_caller_address(verifier.contract_address);

    assert!(verifier.get_model_verification_count(MODEL_ID) == 0, "Count should start at 0");
}

#[test]
#[should_panic(expected: "Only owner can register models")]
fn test_register_model_not_owner() {
    let (verifier, _sage) = setup();
    start_cheat_caller_address(verifier.contract_address, ATTACKER());
    verifier.register_model(MODEL_ID, WEIGHT_COMMIT, NUM_LAYERS, DESCRIPTION);
}

#[test]
#[should_panic(expected: "Model already registered")]
fn test_register_model_duplicate() {
    let (verifier, _sage) = setup();
    start_cheat_caller_address(verifier.contract_address, OWNER());
    verifier.register_model(MODEL_ID, WEIGHT_COMMIT, NUM_LAYERS, DESCRIPTION);
    verifier.register_model(MODEL_ID, WEIGHT_COMMIT, NUM_LAYERS, DESCRIPTION);
}

#[test]
fn test_register_two_models() {
    let (verifier, _sage) = setup();
    start_cheat_caller_address(verifier.contract_address, OWNER());
    verifier.register_model(MODEL_ID, WEIGHT_COMMIT, NUM_LAYERS, DESCRIPTION);
    verifier.register_model(MODEL_ID_2, 'weight_commit_002', 80, DESCRIPTION_2);
    stop_cheat_caller_address(verifier.contract_address);

    assert!(verifier.get_model_verification_count(MODEL_ID) == 0, "Model 1 count should be 0");
    assert!(verifier.get_model_verification_count(MODEL_ID_2) == 0, "Model 2 count should be 0");
}

// ══════════════════════════════════════════════════════════════════════════
// verify_and_pay — Zero Payment (5 tests)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_verify_zero_payment() {
    let (verifier, _sage) = setup();
    start_cheat_caller_address(verifier.contract_address, OWNER());
    let result = verifier
        .verify_and_pay(MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), 0, 0);
    stop_cheat_caller_address(verifier.contract_address);

    assert!(result, "Should return true");
}

#[test]
#[should_panic(expected: "Only owner can submit proofs")]
fn test_verify_not_owner() {
    let (verifier, _sage) = setup();
    start_cheat_caller_address(verifier.contract_address, ATTACKER());
    verifier
        .verify_and_pay(MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), 0, 0);
}

#[test]
#[should_panic(expected: "Job already verified")]
fn test_verify_double_job() {
    let (verifier, _sage) = setup();
    start_cheat_caller_address(verifier.contract_address, OWNER());
    verifier
        .verify_and_pay(MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), 0, 0);
    // Same job_id again should panic
    verifier
        .verify_and_pay(MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), 0, 0);
}

#[test]
fn test_verify_increments_count() {
    let (verifier, _sage) = setup();
    start_cheat_caller_address(verifier.contract_address, OWNER());

    assert!(verifier.get_model_verification_count(MODEL_ID) == 0, "Should start at 0");

    verifier
        .verify_and_pay(MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), 0, 0);

    assert!(verifier.get_model_verification_count(MODEL_ID) == 1, "Should be 1 after verify");
    stop_cheat_caller_address(verifier.contract_address);
}

#[test]
fn test_verify_proof_id_deterministic() {
    let (verifier, _sage) = setup();
    start_cheat_caller_address(verifier.contract_address, OWNER());

    // Verify job 1 with specific inputs
    verifier
        .verify_and_pay(MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), 0, 0);

    // Compute expected proof_id = poseidon(model_id, io_commitment, weight_commitment, proof_hash)
    let proof_id = core::poseidon::poseidon_hash_span(
        [MODEL_ID, IO_COMMIT, WEIGHT_COMMIT, PROOF_HASH].span(),
    );
    assert!(verifier.is_verified(proof_id), "Proof ID should be verified");

    // Same inputs, different job_id → same proof_id already marked verified
    // (but we can't re-verify the same job, so use a different job_id to test the proof_id collision)
    // The proof_id should already be true since same inputs produce same hash
    verifier
        .verify_and_pay(
            MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID_2, WORKER(), 0, 0,
        );

    // proof_id is still verified (same hash from same inputs)
    assert!(verifier.is_verified(proof_id), "Proof ID should still be verified");
    assert!(verifier.get_model_verification_count(MODEL_ID) == 2, "Count should be 2");
    stop_cheat_caller_address(verifier.contract_address);
}

// ══════════════════════════════════════════════════════════════════════════
// verify_and_pay — With Real SAGE Payment (4 tests)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_verify_with_sage_payment() {
    let (verifier, sage) = setup();

    let payment: u256 = 500_000_000_000_000_000_000; // 500 SAGE (18 decimals)

    // Owner approves verifier to spend SAGE
    start_cheat_caller_address(sage.contract_address, OWNER());
    sage.approve(verifier.contract_address, payment);
    stop_cheat_caller_address(sage.contract_address);

    // Owner calls verify_and_pay with SAGE payment → worker
    start_cheat_caller_address(verifier.contract_address, OWNER());
    let result = verifier
        .verify_and_pay(
            MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), payment, 0,
        );
    stop_cheat_caller_address(verifier.contract_address);

    assert!(result, "verify_and_pay should return true");

    // Worker received the SAGE
    let worker_balance = sage.balance_of(WORKER());
    assert!(worker_balance == payment, "Worker should have received 500 SAGE");

    // Owner's balance decreased
    let owner_balance = sage.balance_of(OWNER());
    assert!(owner_balance == TGE_PUBLIC_SALE - payment, "Owner balance should decrease by payment");
}

#[test]
fn test_verify_two_jobs_sage_payment() {
    let (verifier, sage) = setup();

    let payment: u256 = 250_000_000_000_000_000_000; // 250 SAGE each
    let total_payment: u256 = 500_000_000_000_000_000_000; // 500 SAGE total

    // Owner approves verifier for total
    start_cheat_caller_address(sage.contract_address, OWNER());
    sage.approve(verifier.contract_address, total_payment);
    stop_cheat_caller_address(sage.contract_address);

    // Verify two jobs
    start_cheat_caller_address(verifier.contract_address, OWNER());
    verifier
        .verify_and_pay(
            MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), payment, 0,
        );
    verifier
        .verify_and_pay(
            MODEL_ID, 'proof_hash_002', 'io_002', WEIGHT_COMMIT, NUM_LAYERS, JOB_ID_2, WORKER(),
            payment, 0,
        );
    stop_cheat_caller_address(verifier.contract_address);

    // Worker got 2x payment
    let worker_balance = sage.balance_of(WORKER());
    assert!(worker_balance == total_payment, "Worker should have 500 SAGE from 2 jobs");

    // Model verification count = 2
    assert!(verifier.get_model_verification_count(MODEL_ID) == 2, "Count should be 2");
}

#[test]
fn test_verify_sage_zero_amount_skips_transfer() {
    let (verifier, sage) = setup();

    let owner_before = sage.balance_of(OWNER());

    // Verify with 0 SAGE — no transfer should happen
    start_cheat_caller_address(verifier.contract_address, OWNER());
    verifier
        .verify_and_pay(MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), 0, 0);
    stop_cheat_caller_address(verifier.contract_address);

    // Balances unchanged
    let owner_after = sage.balance_of(OWNER());
    let worker_balance = sage.balance_of(WORKER());
    assert!(owner_after == owner_before, "Owner balance should be unchanged");
    assert!(worker_balance == 0, "Worker balance should be 0");
}

#[test]
#[should_panic]
fn test_verify_sage_no_allowance_panics() {
    let (verifier, _sage) = setup();

    let payment: u256 = 500_000_000_000_000_000_000;

    // No approval — should panic during transfer_from
    start_cheat_caller_address(verifier.contract_address, OWNER());
    verifier
        .verify_and_pay(
            MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), payment, 0,
        );
}

// ══════════════════════════════════════════════════════════════════════════
// verify_and_pay — With TEE Attestation (1 test)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_verify_with_tee_attestation() {
    let (verifier, _sage) = setup();

    let tee_hash: felt252 = 'tee_attestation_sha256_hash';

    start_cheat_caller_address(verifier.contract_address, OWNER());
    let result = verifier
        .verify_and_pay(
            MODEL_ID, PROOF_HASH, IO_COMMIT, WEIGHT_COMMIT, NUM_LAYERS, JOB_ID, WORKER(), 0,
            tee_hash,
        );
    stop_cheat_caller_address(verifier.contract_address);

    assert!(result, "verify_and_pay with TEE should return true");

    // Verify the proof was recorded
    let proof_id = core::poseidon::poseidon_hash_span(
        [MODEL_ID, IO_COMMIT, WEIGHT_COMMIT, PROOF_HASH].span(),
    );
    assert!(verifier.is_verified(proof_id), "Proof should be verified");
}

// ══════════════════════════════════════════════════════════════════════════
// View Functions (2 tests)
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_is_verified_unknown() {
    let (verifier, _sage) = setup();
    assert!(!verifier.is_verified('nonexistent_proof_id'), "Unknown proof should not be verified");
}

#[test]
fn test_verification_count_unknown_model() {
    let (verifier, _sage) = setup();
    assert!(
        verifier.get_model_verification_count('nonexistent_model') == 0,
        "Unknown model count should be 0",
    );
}
