// Tests for the audit record system (submit_audit, queries).

use snforge_std::{declare, DeclareResultTrait, ContractClassTrait, start_cheat_caller_address};
use starknet::ContractAddress;
use elo_cairo_verifier::audit::{IAuditVerifierDispatcher, IAuditVerifierDispatcherTrait};
use elo_cairo_verifier::verifier::{
    ISumcheckVerifierDispatcher, ISumcheckVerifierDispatcherTrait,
};

// ============================================================================
// Helpers
// ============================================================================

fn deploy_verifier() -> (ISumcheckVerifierDispatcher, IAuditVerifierDispatcher) {
    let contract = declare("SumcheckVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = 0x1234_felt252.try_into().unwrap();
    let (address, _) = contract.deploy(@array![owner.into()]).unwrap();
    (
        ISumcheckVerifierDispatcher { contract_address: address },
        IAuditVerifierDispatcher { contract_address: address },
    )
}

fn owner() -> ContractAddress {
    0x1234_felt252.try_into().unwrap()
}

fn submitter() -> ContractAddress {
    0x5678_felt252.try_into().unwrap()
}

/// Register a model via the existing interface so submit_audit can find it.
fn register_model(verifier: @ISumcheckVerifierDispatcher, model_id: felt252) {
    let addr = *verifier.contract_address;
    start_cheat_caller_address(addr, owner());
    verifier.register_model(model_id, 0xABC);
}

// ============================================================================
// Test 1: Submit a single audit and read it back
// ============================================================================

#[test]
fn test_submit_and_get_audit() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    register_model(@verifier, model_id);

    start_cheat_caller_address(audit.contract_address, submitter());
    let audit_id = audit.submit_audit(
        model_id,
        0xDEAD,    // report_hash_lo
        0xBEEF,    // report_hash_hi
        0xCA11,    // merkle_root_lo
        0xDA7A,    // merkle_root_hi
        0xABC,     // weight_commitment (must match registered)
        1000,      // time_start
        2000,      // time_end
        50,        // inference_count
        0,         // tee_attestation_hash
        0,         // privacy_tier (public)
    );

    assert!(audit_id != 0, "audit_id should be non-zero");

    // Read it back
    let record = audit.get_audit(audit_id);
    assert!(record.model_id == model_id, "model_id mismatch");
    assert!(record.audit_report_hash.lo == 0xDEAD, "report_hash lo mismatch");
    assert!(record.audit_report_hash.hi == 0xBEEF, "report_hash hi mismatch");
    assert!(record.inference_log_merkle_root.lo == 0xCA11, "merkle_root lo mismatch");
    assert!(record.inference_log_merkle_root.hi == 0xDA7A, "merkle_root hi mismatch");
    assert!(record.weight_commitment == 0xABC, "weight_commitment mismatch");
    assert!(record.time_start == 1000, "time_start mismatch");
    assert!(record.time_end == 2000, "time_end mismatch");
    assert!(record.inference_count == 50, "inference_count mismatch");
    assert!(record.proof_verified == false, "should not be proof-verified");
    assert!(record.submitter == submitter(), "submitter mismatch");
    assert!(record.privacy_tier == 0, "privacy_tier mismatch");
}

// ============================================================================
// Test 2: Audit count and model audit list
// ============================================================================

#[test]
fn test_audit_count_and_list() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    register_model(@verifier, model_id);
    start_cheat_caller_address(audit.contract_address, submitter());

    // Submit 3 audits
    let id1 = audit.submit_audit(model_id, 0x1, 0, 0x1, 0, 0xABC, 1000, 2000, 10, 0, 0);
    let id2 = audit.submit_audit(model_id, 0x2, 0, 0x2, 0, 0xABC, 2000, 3000, 20, 0, 0);
    let id3 = audit.submit_audit(model_id, 0x3, 0, 0x3, 0, 0xABC, 3000, 4000, 30, 0, 0);

    assert!(audit.get_audit_count(model_id) == 3, "should have 3 audits");

    let ids = audit.get_model_audits(model_id);
    assert!(ids.len() == 3, "should return 3 audit IDs");
    assert!(*ids.at(0) == id1, "first ID mismatch");
    assert!(*ids.at(1) == id2, "second ID mismatch");
    assert!(*ids.at(2) == id3, "third ID mismatch");
}

// ============================================================================
// Test 3: get_latest_audit returns most recent
// ============================================================================

#[test]
fn test_get_latest_audit() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    register_model(@verifier, model_id);
    start_cheat_caller_address(audit.contract_address, submitter());

    audit.submit_audit(model_id, 0x1, 0, 0x1, 0, 0xABC, 1000, 2000, 10, 0, 0);
    audit.submit_audit(model_id, 0x99, 0, 0x99, 0, 0xABC, 5000, 6000, 100, 0, 0);

    let latest = audit.get_latest_audit(model_id);
    assert!(latest.audit_report_hash.lo == 0x99, "should be the latest audit");
    assert!(latest.time_start == 5000, "time_start of latest");
    assert!(latest.inference_count == 100, "inference_count of latest");
}

// ============================================================================
// Test 4: is_audited_in_range
// ============================================================================

#[test]
fn test_is_audited_in_range() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    register_model(@verifier, model_id);
    start_cheat_caller_address(audit.contract_address, submitter());

    // Audit covers [1000, 2000]
    audit.submit_audit(model_id, 0x1, 0, 0x1, 0, 0xABC, 1000, 2000, 10, 0, 0);

    // Range that contains the audit window
    assert!(audit.is_audited_in_range(model_id, 500, 3000), "should be audited in [500, 3000]");

    // Range that doesn't contain any audit
    assert!(!audit.is_audited_in_range(model_id, 3000, 4000), "should not be audited in [3000, 4000]");
}

// ============================================================================
// Test 5: total proven inferences accumulates
// ============================================================================

#[test]
fn test_total_proven_inferences() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    register_model(@verifier, model_id);
    start_cheat_caller_address(audit.contract_address, submitter());

    assert!(audit.get_total_proven_inferences(model_id) == 0, "should start at 0");

    audit.submit_audit(model_id, 0x1, 0, 0x1, 0, 0xABC, 1000, 2000, 50, 0, 0);
    assert!(audit.get_total_proven_inferences(model_id) == 50, "should be 50 after first");

    audit.submit_audit(model_id, 0x2, 0, 0x2, 0, 0xABC, 2000, 3000, 30, 0, 0);
    assert!(audit.get_total_proven_inferences(model_id) == 80, "should be 80 after second");
}

// ============================================================================
// Test 6: submit_audit rejects unregistered model
// ============================================================================

#[test]
#[should_panic(expected: "Model not registered")]
fn test_submit_audit_rejects_unregistered() {
    let (_verifier, audit) = deploy_verifier();
    start_cheat_caller_address(audit.contract_address, submitter());

    // Model 0x999 is not registered
    audit.submit_audit(0x999, 0x1, 0, 0x1, 0, 0xABC, 1000, 2000, 10, 0, 0);
}

// ============================================================================
// Test 7: submit_audit rejects wrong weight commitment
// ============================================================================

#[test]
#[should_panic(expected: "Weight commitment mismatch")]
fn test_submit_audit_rejects_wrong_weight() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    register_model(@verifier, model_id); // registers with 0xABC

    start_cheat_caller_address(audit.contract_address, submitter());
    // Submit with wrong weight commitment 0xDEAD != 0xABC
    audit.submit_audit(model_id, 0x1, 0, 0x1, 0, 0xDEAD, 1000, 2000, 10, 0, 0);
}

// ============================================================================
// Test 8: submit_audit rejects invalid time window
// ============================================================================

#[test]
#[should_panic(expected: "Invalid time window")]
fn test_submit_audit_rejects_invalid_time() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    register_model(@verifier, model_id);
    start_cheat_caller_address(audit.contract_address, submitter());

    // time_end <= time_start
    audit.submit_audit(model_id, 0x1, 0, 0x1, 0, 0xABC, 2000, 1000, 10, 0, 0);
}

// ============================================================================
// Test 9: submit_audit rejects zero inference count
// ============================================================================

#[test]
#[should_panic(expected: "Empty audit")]
fn test_submit_audit_rejects_zero_inferences() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    register_model(@verifier, model_id);
    start_cheat_caller_address(audit.contract_address, submitter());

    audit.submit_audit(model_id, 0x1, 0, 0x1, 0, 0xABC, 1000, 2000, 0, 0, 0);
}

// ============================================================================
// Test 10: Audit IDs are unique
// ============================================================================

#[test]
fn test_audit_ids_unique() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    register_model(@verifier, model_id);
    start_cheat_caller_address(audit.contract_address, submitter());

    let id1 = audit.submit_audit(model_id, 0x1, 0, 0x1, 0, 0xABC, 1000, 2000, 10, 0, 0);
    let id2 = audit.submit_audit(model_id, 0x2, 0, 0x2, 0, 0xABC, 2000, 3000, 20, 0, 0);

    assert!(id1 != id2, "audit IDs should be unique");
}

// ============================================================================
// Test 11: Existing verification still works after audit additions
// ============================================================================

#[test]
fn test_existing_verification_unbroken() {
    let (verifier, audit) = deploy_verifier();
    let model_id: felt252 = 0x2;

    // Register via existing interface
    start_cheat_caller_address(verifier.contract_address, owner());
    verifier.register_model(model_id, 0xABC);

    // Existing query still works
    let commitment = verifier.get_model_commitment(model_id);
    assert!(commitment == 0xABC, "existing model commitment should work");

    let count = verifier.get_verification_count(model_id);
    assert!(count == 0, "verification count starts at 0");

    // Audit functions also work on same contract
    assert!(audit.get_audit_count(model_id) == 0, "no audits yet");
    assert!(audit.get_total_proven_inferences(model_id) == 0, "no inferences yet");
}
