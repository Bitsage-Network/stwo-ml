// Tests for access control and view key delegation.

use snforge_std::{declare, DeclareResultTrait, ContractClassTrait, start_cheat_caller_address, start_cheat_block_number};
use starknet::ContractAddress;
use elo_cairo_verifier::audit::{IAuditVerifierDispatcher, IAuditVerifierDispatcherTrait};
use elo_cairo_verifier::access_control::{IAuditAccessControlDispatcher, IAuditAccessControlDispatcherTrait};
use elo_cairo_verifier::view_key::{IViewKeyDelegationDispatcher, IViewKeyDelegationDispatcherTrait};
use elo_cairo_verifier::verifier::{
    ISumcheckVerifierDispatcher, ISumcheckVerifierDispatcherTrait,
};

// ============================================================================
// Helpers
// ============================================================================

fn deploy_verifier() -> (
    ISumcheckVerifierDispatcher,
    IAuditVerifierDispatcher,
    IAuditAccessControlDispatcher,
    IViewKeyDelegationDispatcher,
) {
    let contract = declare("SumcheckVerifierContract").unwrap().contract_class();
    let owner_addr: ContractAddress = 0x1234_felt252.try_into().unwrap();
    let (address, _) = contract.deploy(@array![owner_addr.into()]).unwrap();
    (
        ISumcheckVerifierDispatcher { contract_address: address },
        IAuditVerifierDispatcher { contract_address: address },
        IAuditAccessControlDispatcher { contract_address: address },
        IViewKeyDelegationDispatcher { contract_address: address },
    )
}

fn owner() -> ContractAddress {
    0x1234_felt252.try_into().unwrap()
}

fn submitter() -> ContractAddress {
    0x5678_felt252.try_into().unwrap()
}

fn auditor() -> ContractAddress {
    0xAAAA_felt252.try_into().unwrap()
}

fn regulator() -> ContractAddress {
    0xBBBB_felt252.try_into().unwrap()
}

/// Register a model and submit an audit, returns the audit_id.
fn setup_audit(
    verifier: @ISumcheckVerifierDispatcher,
    audit: @IAuditVerifierDispatcher,
) -> felt252 {
    let model_id: felt252 = 0x2;
    let addr = *verifier.contract_address;

    start_cheat_caller_address(addr, owner());
    verifier.register_model(model_id, 0xABC);

    start_cheat_caller_address(addr, submitter());
    audit.submit_audit(model_id, 0xDEAD, 0xBEEF, 0xCA11, 0xDA7A, 0xABC, 1000, 2000, 50, 0, 1)
}

// ============================================================================
// Access Control Tests
// ============================================================================

#[test]
fn test_grant_and_check_access() {
    let (verifier, audit, acl, _) = deploy_verifier();
    let audit_id = setup_audit(@verifier, @audit);
    let addr = acl.contract_address;

    // Owner is submitter
    assert!(acl.get_audit_owner(audit_id) == submitter(), "owner should be submitter");

    // No access yet
    assert!(!acl.has_audit_access(audit_id, auditor()), "should not have access yet");
    assert!(acl.get_access_count(audit_id) == 0, "should have 0 grants");

    // Grant access
    start_cheat_caller_address(addr, submitter());
    acl.grant_audit_access(audit_id, auditor(), 0xAE01, 1);

    assert!(acl.has_audit_access(audit_id, auditor()), "should have access");
    assert!(acl.get_access_count(audit_id) == 1, "should have 1 grant");
    assert!(acl.get_wrapped_key(audit_id, auditor()) == 0xAE01, "wrapped key mismatch");
}

#[test]
fn test_revoke_access() {
    let (verifier, audit, acl, _) = deploy_verifier();
    let audit_id = setup_audit(@verifier, @audit);
    let addr = acl.contract_address;

    // Grant then revoke
    start_cheat_caller_address(addr, submitter());
    acl.grant_audit_access(audit_id, auditor(), 0xAE01, 1);
    assert!(acl.has_audit_access(audit_id, auditor()), "should have access");

    acl.revoke_audit_access(audit_id, auditor());
    assert!(!acl.has_audit_access(audit_id, auditor()), "should not have access after revoke");
}

#[test]
fn test_batch_grant() {
    let (verifier, audit, acl, _) = deploy_verifier();
    let audit_id = setup_audit(@verifier, @audit);
    let addr = acl.contract_address;

    start_cheat_caller_address(addr, submitter());
    acl.grant_audit_access_batch(
        audit_id,
        array![auditor(), regulator()].span(),
        array![0xAE01, 0xAE02].span(),
        array![1, 2].span(),
    );

    assert!(acl.has_audit_access(audit_id, auditor()), "auditor should have access");
    assert!(acl.has_audit_access(audit_id, regulator()), "regulator should have access");
    assert!(acl.get_access_count(audit_id) == 2, "should have 2 grants");
    assert!(acl.get_wrapped_key(audit_id, auditor()) == 0xAE01, "auditor key");
    assert!(acl.get_wrapped_key(audit_id, regulator()) == 0xAE02, "regulator key");
}

#[test]
#[should_panic(expected: "Only audit owner can grant access")]
fn test_non_owner_cannot_grant() {
    let (verifier, audit, acl, _) = deploy_verifier();
    let audit_id = setup_audit(@verifier, @audit);

    // Auditor tries to grant — should fail
    start_cheat_caller_address(acl.contract_address, auditor());
    acl.grant_audit_access(audit_id, regulator(), 0xAE00, 2);
}

#[test]
#[should_panic(expected: "Only audit owner can revoke access")]
fn test_non_owner_cannot_revoke() {
    let (verifier, audit, acl, _) = deploy_verifier();
    let audit_id = setup_audit(@verifier, @audit);
    let addr = acl.contract_address;

    // Grant first
    start_cheat_caller_address(addr, submitter());
    acl.grant_audit_access(audit_id, auditor(), 0xAE01, 1);

    // Auditor tries to revoke themselves — should fail
    start_cheat_caller_address(addr, auditor());
    acl.revoke_audit_access(audit_id, auditor());
}

#[test]
#[should_panic(expected: "Access already granted")]
fn test_duplicate_grant_rejected() {
    let (verifier, audit, acl, _) = deploy_verifier();
    let audit_id = setup_audit(@verifier, @audit);
    let addr = acl.contract_address;

    start_cheat_caller_address(addr, submitter());
    acl.grant_audit_access(audit_id, auditor(), 0xAE01, 1);
    // Second grant should fail
    acl.grant_audit_access(audit_id, auditor(), 0xAE02, 1);
}

#[test]
#[should_panic(expected: "No active access")]
fn test_get_wrapped_key_without_access() {
    let (verifier, audit, acl, _) = deploy_verifier();
    let audit_id = setup_audit(@verifier, @audit);

    // No grant — should panic
    acl.get_wrapped_key(audit_id, auditor());
}

#[test]
fn test_revoked_wrapped_key_zeroed() {
    let (verifier, audit, acl, _) = deploy_verifier();
    let audit_id = setup_audit(@verifier, @audit);
    let addr = acl.contract_address;

    start_cheat_caller_address(addr, submitter());
    acl.grant_audit_access(audit_id, auditor(), 0xAE01, 1);
    acl.revoke_audit_access(audit_id, auditor());

    // Active access count decremented on revoke.
    assert!(!acl.has_audit_access(audit_id, auditor()), "should not have access");
    assert!(acl.get_access_count(audit_id) == 0, "active count should be 0 after revoke");
}

// ============================================================================
// View Key Delegation Tests
// ============================================================================

#[test]
fn test_delegate_and_check_view_key() {
    let (_, _, _, vk) = deploy_verifier();
    let addr = vk.contract_address;

    start_cheat_caller_address(addr, submitter());
    vk.delegate_view_key(auditor(), 0xF1E001, 0); // 0 = forever

    assert!(vk.has_view_key(submitter(), auditor()), "should have view key");
    assert!(vk.get_view_key(submitter(), auditor()) == 0xF1E001, "view key mismatch");
    assert!(vk.get_delegation_count(submitter()) == 1, "should have 1 delegation");
}

#[test]
fn test_revoke_view_key() {
    let (_, _, _, vk) = deploy_verifier();
    let addr = vk.contract_address;

    start_cheat_caller_address(addr, submitter());
    vk.delegate_view_key(auditor(), 0xF1E001, 0);
    assert!(vk.has_view_key(submitter(), auditor()), "should have view key");

    vk.revoke_view_key(auditor());
    assert!(!vk.has_view_key(submitter(), auditor()), "should not have view key after revoke");
}

#[test]
fn test_view_key_expiry() {
    let (_, _, _, vk) = deploy_verifier();
    let addr = vk.contract_address;

    // Delegate with expiry at block 100
    start_cheat_block_number(addr, 50);
    start_cheat_caller_address(addr, submitter());
    vk.delegate_view_key(auditor(), 0xF1E001, 100);

    // At block 80, still valid
    start_cheat_block_number(addr, 80);
    assert!(vk.has_view_key(submitter(), auditor()), "should be valid before expiry");

    // At block 101, expired
    start_cheat_block_number(addr, 101);
    assert!(!vk.has_view_key(submitter(), auditor()), "should be expired");
}

#[test]
#[should_panic(expected: "View key already delegated")]
fn test_duplicate_delegation_rejected() {
    let (_, _, _, vk) = deploy_verifier();
    let addr = vk.contract_address;

    start_cheat_caller_address(addr, submitter());
    vk.delegate_view_key(auditor(), 0xF1E001, 0);
    // Second delegation should fail
    vk.delegate_view_key(auditor(), 0xF1E002, 0);
}

#[test]
fn test_multiple_delegations() {
    let (_, _, _, vk) = deploy_verifier();
    let addr = vk.contract_address;

    start_cheat_caller_address(addr, submitter());
    vk.delegate_view_key(auditor(), 0xAE0A, 0);
    vk.delegate_view_key(regulator(), 0xAE0B, 0);

    assert!(vk.get_delegation_count(submitter()) == 2, "should have 2 delegations");
    assert!(vk.has_view_key(submitter(), auditor()), "auditor should have view key");
    assert!(vk.has_view_key(submitter(), regulator()), "regulator should have view key");
}
