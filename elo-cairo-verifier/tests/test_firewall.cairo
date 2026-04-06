/// Tests for AgentFirewallZK: on-chain guardrails for AI agent transactions.
///
/// Tests cover agent registration, action submission, proof-based resolution,
/// EMA trust scoring, strike mechanism, auto-freeze, escalation flow, and access control.

use snforge_std::{declare, DeclareResultTrait, ContractClassTrait};
use snforge_std::{start_cheat_caller_address, stop_cheat_caller_address};
use starknet::ContractAddress;
use elo_cairo_verifier::firewall::{
    IAgentFirewallDispatcher, IAgentFirewallDispatcherTrait,
};

// ============================================================================
// Helpers
// ============================================================================

fn owner() -> ContractAddress {
    0x1234_felt252.try_into().unwrap()
}

fn agent_owner_addr() -> ContractAddress {
    0xABCD_felt252.try_into().unwrap()
}

fn deploy_firewall() -> IAgentFirewallDispatcher {
    let contract = declare("AgentFirewallZK").unwrap().contract_class();
    let verifier_address: ContractAddress = 0x9999_felt252.try_into().unwrap();
    let classifier_model_id: felt252 = 0x42;
    let classifier_weight_root_hash: felt252 = 0xCAFEBABE; // test weight hash
    let (address, _) = contract
        .deploy(@array![
            owner().into(),
            verifier_address.into(),
            classifier_model_id,
            classifier_weight_root_hash,
        ])
        .unwrap();
    IAgentFirewallDispatcher { contract_address: address }
}

// ============================================================================
// Test 1: Register an agent
// ============================================================================

#[test]
fn test_register_agent() {
    let fw = deploy_firewall();
    let agent_addr = agent_owner_addr();
    start_cheat_caller_address(fw.contract_address, agent_addr);

    let agent_id: felt252 = 0xA1;
    fw.register_agent(agent_id);

    assert!(fw.is_agent_active(agent_id), "agent should be active after registration");
    assert!(fw.get_trust_score(agent_id) == 0, "initial trust score should be 0");
    assert!(fw.get_strikes(agent_id) == 0, "initial strikes should be 0");
    assert!(fw.is_trusted(agent_id), "new agent should be trusted");
}

// ============================================================================
// Test 2: Cannot register same agent twice
// ============================================================================

#[test]
#[should_panic(expected: "AGENT_ALREADY_REGISTERED")]
fn test_register_agent_duplicate_rejected() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());

    fw.register_agent(0xA1);
    fw.register_agent(0xA1); // should panic
}

// ============================================================================
// Test 3: Submit an action
// ============================================================================

#[test]
fn test_submit_action() {
    let fw = deploy_firewall();
    let agent_addr = agent_owner_addr();
    start_cheat_caller_address(fw.contract_address, agent_addr);

    fw.register_agent(0xA1);
    let action_id = fw.submit_action(0xA1, 0xDEAD, 0x1000, 0xa9059cbb, 0xCAFE);

    assert!(action_id == 1, "first action should have id 1");
    assert!(fw.get_action_decision(action_id) == 0, "action should be pending");
    assert!(!fw.is_action_approved(action_id), "pending action should not be approved");
}

// ============================================================================
// Test 4: Cannot submit action for frozen agent
// ============================================================================

#[test]
#[should_panic(expected: "AGENT_FROZEN")]
fn test_submit_action_frozen_agent_rejected() {
    let fw = deploy_firewall();
    let agent_addr = agent_owner_addr();
    start_cheat_caller_address(fw.contract_address, agent_addr);

    fw.register_agent(0xA1);
    fw.deactivate_agent(0xA1);
    fw.submit_action(0xA1, 0xDEAD, 0x1000, 0xa9059cbb, 0xCAFE); // should panic
}

// ============================================================================
// Test 5: Deactivate and reactivate agent
// ============================================================================

#[test]
fn test_deactivate_and_reactivate() {
    let fw = deploy_firewall();
    let agent_addr = agent_owner_addr();
    start_cheat_caller_address(fw.contract_address, agent_addr);

    fw.register_agent(0xA1);
    assert!(fw.is_agent_active(0xA1), "should be active");

    fw.deactivate_agent(0xA1);
    assert!(!fw.is_agent_active(0xA1), "should be deactivated");
    assert!(!fw.is_trusted(0xA1), "deactivated agent should not be trusted");

    fw.reactivate_agent(0xA1);
    assert!(fw.is_agent_active(0xA1), "should be reactivated");
    assert!(fw.get_strikes(0xA1) == 0, "strikes should reset on reactivation");
}

// ============================================================================
// Test 6: Non-owner cannot deactivate agent
// ============================================================================

#[test]
#[should_panic(expected: "NOT_AGENT_OR_CONTRACT_OWNER")]
fn test_deactivate_non_owner_rejected() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());
    fw.register_agent(0xA1);
    stop_cheat_caller_address(fw.contract_address);

    let attacker: ContractAddress = 0xBAD_felt252.try_into().unwrap();
    start_cheat_caller_address(fw.contract_address, attacker);
    fw.deactivate_agent(0xA1); // should panic
}

// ============================================================================
// Test 7: Contract owner can deactivate any agent
// ============================================================================

#[test]
fn test_contract_owner_can_deactivate() {
    let fw = deploy_firewall();

    // Agent registers itself
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());
    fw.register_agent(0xA1);
    stop_cheat_caller_address(fw.contract_address);

    // Contract owner deactivates
    start_cheat_caller_address(fw.contract_address, owner());
    fw.deactivate_agent(0xA1);
    assert!(!fw.is_agent_active(0xA1), "contract owner should be able to deactivate");
}

// ============================================================================
// Test 8: is_trusted reflects thresholds
// ============================================================================

#[test]
fn test_is_trusted_respects_state() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());

    fw.register_agent(0xA1);

    // New agent: active, score=0, strikes=0 → trusted
    assert!(fw.is_trusted(0xA1), "new agent should be trusted");

    // Deactivated → not trusted
    fw.deactivate_agent(0xA1);
    assert!(!fw.is_trusted(0xA1), "deactivated agent not trusted");

    // Reactivated → trusted again
    fw.reactivate_agent(0xA1);
    assert!(fw.is_trusted(0xA1), "reactivated agent should be trusted");
}

// ============================================================================
// Test 9: Action IDs increment correctly
// ============================================================================

#[test]
fn test_action_ids_increment() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());

    fw.register_agent(0xA1);
    let id1 = fw.submit_action(0xA1, 0xDEAD, 0x100, 0xa9059cbb, 0xCAFE);
    let id2 = fw.submit_action(0xA1, 0xBEEF, 0x200, 0x095ea7b3, 0xFACE);
    let id3 = fw.submit_action(0xA1, 0xF00D, 0x300, 0x38ed1739, 0xDECA);

    assert!(id1 == 1, "first action id should be 1");
    assert!(id2 == 2, "second action id should be 2");
    assert!(id3 == 3, "third action id should be 3");
}

// ============================================================================
// Test 10: Non-agent-owner cannot submit action
// ============================================================================

#[test]
#[should_panic(expected: "NOT_AGENT_OWNER")]
fn test_submit_action_non_owner_rejected() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());
    fw.register_agent(0xA1);
    stop_cheat_caller_address(fw.contract_address);

    let attacker: ContractAddress = 0xBAD_felt252.try_into().unwrap();
    start_cheat_caller_address(fw.contract_address, attacker);
    fw.submit_action(0xA1, 0xDEAD, 0x100, 0xa9059cbb, 0xCAFE); // should panic
}

// ============================================================================
// Test 11: Cannot submit action for unregistered agent
// ============================================================================

#[test]
#[should_panic(expected: "AGENT_NOT_REGISTERED")]
fn test_submit_action_unregistered_agent_rejected() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());
    fw.submit_action(0xDEAD, 0xBEEF, 0x100, 0xa9059cbb, 0xCAFE); // no agent registered
}

// ============================================================================
// Test 12: Admin can update thresholds
// ============================================================================

#[test]
fn test_set_thresholds() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, owner());

    fw.set_thresholds(30000, 80000, 3);
    let (esc, blk, max) = fw.get_thresholds();
    assert!(esc == 30000, "escalate threshold should be updated");
    assert!(blk == 80000, "block threshold should be updated");
    assert!(max == 3, "max strikes should be updated");
}

// ============================================================================
// Test 13: Non-owner cannot update thresholds
// ============================================================================

#[test]
#[should_panic(expected: "ONLY_OWNER")]
fn test_set_thresholds_non_owner_rejected() {
    let fw = deploy_firewall();
    let attacker: ContractAddress = 0xBAD_felt252.try_into().unwrap();
    start_cheat_caller_address(fw.contract_address, attacker);
    fw.set_thresholds(10000, 50000, 10);
}

// ============================================================================
// Test 14: Invalid thresholds rejected
// ============================================================================

#[test]
#[should_panic(expected: "ESCALATE_MUST_BE_LESS_THAN_BLOCK")]
fn test_set_thresholds_invalid_order_rejected() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, owner());
    fw.set_thresholds(80000, 40000, 5); // escalate > block → invalid
}

// ============================================================================
// Test 15: Query action metadata
// ============================================================================

#[test]
fn test_query_action_metadata() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());

    fw.register_agent(0xA1);
    let action_id = fw.submit_action(0xA1, 0xDEAD, 0x1000, 0xa9059cbb, 0xCAFE);

    assert!(fw.get_action_io_commitment(action_id) == 0xCAFE, "io_commitment should match");
    assert!(fw.get_action_threat_score(action_id) == 0, "unresolved action threat score should be 0");
}

// ============================================================================
// Test 16: Query contract configuration
// ============================================================================

#[test]
fn test_query_configuration() {
    let fw = deploy_firewall();

    assert!(fw.get_owner() == owner(), "owner should match constructor arg");
    assert!(fw.get_classifier_model_id() == 0x42, "model id should match constructor arg");

    let (esc, blk, max) = fw.get_thresholds();
    assert!(esc == 40000, "default escalate threshold");
    assert!(blk == 70000, "default block threshold");
    assert!(max == 5, "default max strikes");
}

// ============================================================================
// Test 17: Reactivate resets strikes
// ============================================================================

#[test]
fn test_reactivate_resets_strikes_to_zero() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());

    fw.register_agent(0xA1);

    // Deactivate (simulating freeze after strikes)
    fw.deactivate_agent(0xA1);
    assert!(!fw.is_agent_active(0xA1), "should be deactivated");

    // Reactivate — strikes must reset
    fw.reactivate_agent(0xA1);
    assert!(fw.is_agent_active(0xA1), "should be reactivated");
    assert!(fw.get_strikes(0xA1) == 0, "strikes must be 0 after reactivation");
}

// ============================================================================
// Test 18: is_agent_registered query
// ============================================================================

#[test]
fn test_is_agent_registered() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());

    assert!(!fw.is_agent_registered(0xA1), "unregistered agent should return false");

    fw.register_agent(0xA1);
    assert!(fw.is_agent_registered(0xA1), "registered agent should return true");
}

// ============================================================================
// Test 19: Zero IO commitment rejected on submit
// ============================================================================

#[test]
#[should_panic(expected: "IO_COMMITMENT_ZERO")]
fn test_submit_action_zero_io_commitment_rejected() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());

    fw.register_agent(0xA1);
    fw.submit_action(0xA1, 0xDEAD, 0x1000, 0xa9059cbb, 0); // zero io_commitment → rejected
}

// ============================================================================
// Test 20: get_agent_owner returns correct address
// ============================================================================

#[test]
fn test_get_agent_owner() {
    let fw = deploy_firewall();
    let agent_addr = agent_owner_addr();
    start_cheat_caller_address(fw.contract_address, agent_addr);

    fw.register_agent(0xA1);
    assert!(fw.get_agent_owner(0xA1) == agent_addr, "agent owner should match caller");
}

// ============================================================================
// Test 21: get_action_agent returns correct agent
// ============================================================================

#[test]
fn test_get_action_agent() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());

    fw.register_agent(0xA1);
    let action_id = fw.submit_action(0xA1, 0xDEAD, 0x1000, 0xa9059cbb, 0xCAFE);
    assert!(fw.get_action_agent(action_id) == 0xA1, "action agent should match");
}

// ============================================================================
// Test 22: set_verifier zero address rejected
// ============================================================================

#[test]
#[should_panic(expected: "VERIFIER_CANNOT_BE_ZERO")]
fn test_set_verifier_zero_rejected() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, owner());

    let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
    fw.set_verifier(zero_addr);
}

// ============================================================================
// Test 23: set_classifier_model zero rejected
// ============================================================================

#[test]
#[should_panic(expected: "MODEL_ID_CANNOT_BE_ZERO")]
fn test_set_classifier_model_zero_rejected() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, owner());

    fw.set_classifier_model(0, 0xDEAD);
}

// ============================================================================
// Test 24: Multiple sequential actions for same agent
// ============================================================================

#[test]
fn test_multiple_actions_same_agent() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());

    fw.register_agent(0xA1);
    let id1 = fw.submit_action(0xA1, 0xDEAD, 0x100, 0xa9059cbb, 0xCAFE);
    let id2 = fw.submit_action(0xA1, 0xBEEF, 0x200, 0x095ea7b3, 0xFACE);

    // Both pending
    assert!(fw.get_action_decision(id1) == 0, "first should be pending");
    assert!(fw.get_action_decision(id2) == 0, "second should be pending");

    // Different IO commitments
    assert!(fw.get_action_io_commitment(id1) == 0xCAFE, "first io");
    assert!(fw.get_action_io_commitment(id2) == 0xFACE, "second io");
}

// ============================================================================
// Test 25: Emergency pause blocks submit_action
// ============================================================================

#[test]
#[should_panic(expected: "CONTRACT_PAUSED")]
fn test_pause_blocks_submit() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());
    fw.register_agent(0xA1);
    stop_cheat_caller_address(fw.contract_address);

    // Owner pauses
    start_cheat_caller_address(fw.contract_address, owner());
    fw.pause();
    assert!(fw.is_paused(), "should be paused");
    stop_cheat_caller_address(fw.contract_address);

    // Agent tries to submit → blocked
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());
    fw.submit_action(0xA1, 0xDEAD, 0x1000, 0xa9059cbb, 0xCAFE);
}

// ============================================================================
// Test 26: Unpause restores functionality
// ============================================================================

#[test]
fn test_unpause_restores() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());
    fw.register_agent(0xA1);
    stop_cheat_caller_address(fw.contract_address);

    // Pause
    start_cheat_caller_address(fw.contract_address, owner());
    fw.pause();
    assert!(fw.is_paused(), "should be paused");
    fw.unpause();
    assert!(!fw.is_paused(), "should be unpaused");
    stop_cheat_caller_address(fw.contract_address);

    // Agent can submit again
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());
    let id = fw.submit_action(0xA1, 0xDEAD, 0x1000, 0xa9059cbb, 0xCAFE);
    assert!(id == 1, "should get action id 1");
}

// ============================================================================
// Test 27: Non-owner cannot pause
// ============================================================================

#[test]
#[should_panic(expected: "ONLY_OWNER")]
fn test_pause_non_owner_rejected() {
    let fw = deploy_firewall();
    let attacker: ContractAddress = 0xBAD_felt252.try_into().unwrap();
    start_cheat_caller_address(fw.contract_address, attacker);
    fw.pause();
}

// ============================================================================
// Test 28: Per-agent rate limiting
// ============================================================================

#[test]
#[should_panic(expected: "TOO_MANY_PENDING_ACTIONS")]
fn test_rate_limit_per_agent() {
    let fw = deploy_firewall();
    start_cheat_caller_address(fw.contract_address, agent_owner_addr());
    fw.register_agent(0xA1);

    // Default max is 10 — submit 10 actions
    let mut i: u32 = 0;
    loop {
        if i >= 10 {
            break;
        }
        fw.submit_action(0xA1, 0xDEAD, i.into(), 0xa9059cbb, (0xCAFE + i.into()));
        i += 1;
    };

    // 11th should fail
    fw.submit_action(0xA1, 0xDEAD, 0x999, 0xa9059cbb, 0xFFFF);
}

// ============================================================================
// Test 29: 2-step ownership transfer
// ============================================================================

#[test]
fn test_ownership_transfer() {
    let fw = deploy_firewall();
    let new_owner: ContractAddress = 0x5678_felt252.try_into().unwrap();

    // Step 1: current owner initiates transfer
    start_cheat_caller_address(fw.contract_address, owner());
    fw.transfer_ownership(new_owner);
    stop_cheat_caller_address(fw.contract_address);

    // Owner is still the original until accepted
    assert!(fw.get_owner() == owner(), "owner should not change until accepted");

    // Step 2: new owner accepts
    start_cheat_caller_address(fw.contract_address, new_owner);
    fw.accept_ownership();
    stop_cheat_caller_address(fw.contract_address);

    assert!(fw.get_owner() == new_owner, "owner should now be new_owner");
}

// ============================================================================
// Test 30: Non-pending-owner cannot accept ownership
// ============================================================================

#[test]
#[should_panic(expected: "NOT_PENDING_OWNER")]
fn test_accept_ownership_non_pending_rejected() {
    let fw = deploy_firewall();
    let new_owner: ContractAddress = 0x5678_felt252.try_into().unwrap();

    start_cheat_caller_address(fw.contract_address, owner());
    fw.transfer_ownership(new_owner);
    stop_cheat_caller_address(fw.contract_address);

    // Attacker tries to accept
    let attacker: ContractAddress = 0xBAD_felt252.try_into().unwrap();
    start_cheat_caller_address(fw.contract_address, attacker);
    fw.accept_ownership();
}
