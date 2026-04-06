// AgentFirewallZK: On-chain guardrails for AI agent transactions.
//
// Replaces LLM-based oracles (ENShell's Claude scoring) with ZKML-proven
// classifier inference. Every agent action is scored by a small MLP classifier
// whose inference is proven with GKR+STARK and verified on the ObelyskVerifier.
//
// Flow:
//   1. Agent SDK calls submit_action(agent_id, target, value, io_commitment)
//   2. Prover generates ZKML proof of classifier inference
//   3. Proof is verified on ObelyskVerifier (6-step streaming or 1-TX recursive)
//   4. SDK calls resolve_action_with_proof(action_id, proof_hash, threat_score)
//   5. Contract checks proof is verified, updates EMA trust score, applies decision
//   6. External contracts call is_action_approved(action_id) before executing

use starknet::ContractAddress;

/// Verifier contract interface (subset needed by firewall).
#[starknet::interface]
pub trait IVerifier<TContractState> {
    fn is_proof_verified(self: @TContractState, proof_hash: felt252) -> bool;
    fn get_model_policy(self: @TContractState, model_id: felt252) -> felt252;
    fn get_model_weight_root_hash(self: @TContractState, model_id: felt252) -> felt252;
    fn get_proof_io_commitment(self: @TContractState, proof_hash: felt252) -> felt252;
    fn get_proof_model_id(self: @TContractState, proof_hash: felt252) -> felt252;
}

/// Agent Firewall interface.
#[starknet::interface]
pub trait IAgentFirewall<TContractState> {
    /// Register a new agent. Caller becomes the agent owner.
    fn register_agent(ref self: TContractState, agent_id: felt252);

    /// Deactivate an agent (owner or contract owner only).
    fn deactivate_agent(ref self: TContractState, agent_id: felt252);

    /// Reactivate a deactivated agent (owner only, resets strikes to 0).
    fn reactivate_agent(ref self: TContractState, agent_id: felt252);

    /// Submit an action for classifier evaluation.
    /// Returns the action_id for tracking.
    fn submit_action(
        ref self: TContractState,
        agent_id: felt252,
        target: felt252,
        value: felt252,
        selector: u32,
        io_commitment: felt252,
    ) -> u64;

    /// Resolve an action using a verified ZKML classifier proof.
    ///
    /// The proof must have been verified on the ObelyskVerifier contract.
    /// The classifier model must have the strict policy registered.
    ///
    /// The threat score is NOT caller-supplied — it is extracted from the
    /// packed IO data and computed on-chain. The caller provides packed_raw_io
    /// as calldata, which is verified against the proof's io_commitment via
    /// Poseidon hash. The 3 output neurons (safe, suspicious, malicious) are
    /// extracted and the threat score is computed as:
    ///   threat_score = (malicious * 100000) / (safe + suspicious + malicious)
    fn resolve_action_with_proof(
        ref self: TContractState,
        action_id: u64,
        proof_hash: felt252,
        original_io_len: u32,
        packed_raw_io: Array<felt252>,
    );

    /// Approve an escalated action (agent owner only, for human-in-the-loop).
    fn approve_escalated(ref self: TContractState, action_id: u64);

    /// Reject an escalated action (agent owner only).
    fn reject_escalated(ref self: TContractState, action_id: u64);

    // ── Admin ─────────────────────────────────────────────────────────

    /// Update scoring thresholds (contract owner only).
    fn set_thresholds(
        ref self: TContractState,
        escalate_threshold: u32,
        block_threshold: u32,
        max_strikes: u32,
    );

    /// Emergency pause (contract owner only). Blocks submit + resolve.
    fn pause(ref self: TContractState);

    /// Unpause (contract owner only).
    fn unpause(ref self: TContractState);

    /// Initiate ownership transfer (2-step pattern, owner only).
    fn transfer_ownership(ref self: TContractState, new_owner: ContractAddress);

    /// Accept pending ownership transfer (new owner only).
    fn accept_ownership(ref self: TContractState);

    /// Update verifier contract address (contract owner only).
    fn set_verifier(ref self: TContractState, verifier_address: ContractAddress);

    /// Update classifier model ID and expected weight hash (contract owner only).
    /// Both must be provided — prevents model substitution attacks.
    fn set_classifier_model(
        ref self: TContractState, model_id: felt252, weight_root_hash: felt252,
    );

    // ── Queries ──────────────────────────────────────────────────────

    /// Check if a specific action has been approved.
    fn is_action_approved(self: @TContractState, action_id: u64) -> bool;

    /// Check if an agent is trusted (active, score below threshold, strikes below max).
    fn is_trusted(self: @TContractState, agent_id: felt252) -> bool;

    /// Get an agent's current trust score (0-100000).
    fn get_trust_score(self: @TContractState, agent_id: felt252) -> u64;

    /// Get an agent's strike count.
    fn get_strikes(self: @TContractState, agent_id: felt252) -> u32;

    /// Get whether an agent is active.
    fn is_agent_active(self: @TContractState, agent_id: felt252) -> bool;

    /// Get whether an agent is registered.
    fn is_agent_registered(self: @TContractState, agent_id: felt252) -> bool;

    /// Get the agent's owner address.
    fn get_agent_owner(self: @TContractState, agent_id: felt252) -> ContractAddress;

    /// Get which agent submitted an action.
    fn get_action_agent(self: @TContractState, action_id: u64) -> felt252;

    /// Get the action decision (0=pending, 1=approved, 2=escalated, 3=blocked).
    fn get_action_decision(self: @TContractState, action_id: u64) -> u8;

    /// Get the action's IO commitment.
    fn get_action_io_commitment(self: @TContractState, action_id: u64) -> felt252;

    /// Get the action's threat score (0 if unresolved).
    fn get_action_threat_score(self: @TContractState, action_id: u64) -> u32;

    /// Get the contract owner.
    fn get_owner(self: @TContractState) -> ContractAddress;

    /// Get the verifier contract address.
    fn get_verifier(self: @TContractState) -> ContractAddress;

    /// Get the classifier model ID.
    fn get_classifier_model_id(self: @TContractState) -> felt252;

    /// Get the current thresholds.
    fn get_thresholds(self: @TContractState) -> (u32, u32, u32);

    /// Check if the contract is paused.
    fn is_paused(self: @TContractState) -> bool;
}

/// Extract a single M31 value (31-bit unsigned) from packed felt252 array.
/// Each felt252 holds 8 M31 values. Re-exported from crate::field for firewall use.
fn extract_m31(packed_felts: Span<felt252>, m31_index: u32) -> u32 {
    crate::field::extract_m31_from_packed(packed_felts, m31_index)
}

/// Compute floor(log2(val)) + 1 for val > 0. Returns 0 for val == 0.
/// Matches Rust's `128 - val.leading_zeros()`.
fn bit_length_u128(val: u128) -> u32 {
    if val == 0 {
        return 0;
    }
    let mut bits: u32 = 0;
    let mut v: u128 = val;
    // Binary search: narrow down the bit position
    if v >= 0x10000000000000000 { // >= 2^64
        bits += 64;
        v = v / 0x10000000000000000;
    }
    if v >= 0x100000000 { // >= 2^32
        bits += 32;
        v = v / 0x100000000;
    }
    if v >= 0x10000 { // >= 2^16
        bits += 16;
        v = v / 0x10000;
    }
    if v >= 0x100 { // >= 2^8
        bits += 8;
        v = v / 0x100;
    }
    if v >= 0x10 { // >= 2^4
        bits += 4;
        v = v / 0x10;
    }
    if v >= 4 { // >= 2^2
        bits += 2;
        v = v / 4;
    }
    if v >= 2 {
        bits += 1;
        v = v / 2;
    }
    if v >= 1 {
        bits += 1;
    }
    bits
}

#[starknet::contract]
pub mod AgentFirewallZK {
    use starknet::storage::{
        StoragePointerReadAccess, StoragePointerWriteAccess, Map, StoragePathEntry,
    };
    use starknet::{ContractAddress, get_caller_address, get_block_timestamp};
    use super::{IVerifierDispatcher, IVerifierDispatcherTrait, extract_m31, bit_length_u128};

    // ── Constants ────────────────────────────────────────────────────

    /// EMA alpha for INCREASING scores (score > prev): fast up.
    /// 500 / 1000 = 0.5 — bad actions raise the score quickly.
    const EMA_ALPHA_UP_NUM: u64 = 500;
    /// EMA alpha for DECREASING scores (score <= prev): slow down.
    /// 100 / 1000 = 0.1 — safe actions lower the score slowly.
    /// This asymmetry prevents EMA dilution (Attack 4).
    const EMA_ALPHA_DOWN_NUM: u64 = 100;
    /// EMA alpha denominator.
    const EMA_ALPHA_DEN: u64 = 1000;

    /// Maximum age (seconds) for a pending action before it expires.
    /// Actions older than this cannot be resolved (Attack 6).
    const MAX_ACTION_AGE: u64 = 3600; // 1 hour

    /// Behavioral tracking window (seconds). Stats reset when window expires.
    const BEHAVIORAL_WINDOW: u64 = 86400; // 24 hours

    /// Default escalation threshold (40000 / 100000).
    const DEFAULT_ESCALATE_THRESHOLD: u32 = 40000;
    /// Default block threshold (70000 / 100000).
    const DEFAULT_BLOCK_THRESHOLD: u32 = 70000;
    /// Default max strikes before auto-freeze.
    const DEFAULT_MAX_STRIKES: u32 = 5;

    // ── Storage ──────────────────────────────────────────────────────

    #[storage]
    struct Storage {
        /// Contract owner.
        owner: ContractAddress,
        /// Pending ownership transfer target (2-step transfer).
        pending_owner: ContractAddress,
        /// Emergency pause flag — blocks submit_action and resolve_action_with_proof.
        paused: bool,
        /// ObelyskVerifier contract address.
        verifier_address: ContractAddress,
        /// Classifier model ID registered on the verifier.
        classifier_model_id: felt252,
        /// Expected weight root hash for the classifier model.
        /// Poseidon hash of all weight Merkle roots — prevents model substitution.
        /// Set at construction, verified during resolve_action_with_proof.
        classifier_weight_root_hash: felt252,

        // ── Agent registry ───────────────────────────────────────────
        agent_owner: Map<felt252, ContractAddress>,
        agent_trust_score: Map<felt252, u64>,
        agent_strikes: Map<felt252, u32>,
        agent_active: Map<felt252, bool>,
        agent_registered: Map<felt252, bool>,
        agent_registered_at: Map<felt252, u64>,

        // ── Action queue ─────────────────────────────────────────────
        next_action_id: u64,
        action_agent: Map<u64, felt252>,
        action_target: Map<u64, felt252>,
        action_value: Map<u64, felt252>,
        action_selector: Map<u64, u32>,
        action_io_commitment: Map<u64, felt252>,
        action_resolved: Map<u64, bool>,
        /// 0=pending, 1=approved, 2=escalated, 3=blocked
        action_decision: Map<u64, u8>,
        action_threat_score: Map<u64, u32>,
        action_proof_hash: Map<u64, felt252>,
        action_submitted_at: Map<u64, u64>,

        // ── Per-agent rate limiting ──────────────────────────────────
        /// agent_id → number of currently pending (unresolved) actions.
        agent_pending_count: Map<felt252, u32>,
        /// Maximum pending actions per agent (default: 10).
        max_pending_per_agent: u32,

        // ── Behavioral tracking (accumulated per submit_action) ─────
        /// (agent_id, target) → interaction count with this target.
        interaction_count: Map<(felt252, felt252), u32>,
        /// agent_id → total actions submitted (all-time, for frequency calc).
        agent_total_actions: Map<felt252, u32>,
        /// agent_id → timestamp of first action in current window.
        agent_window_start: Map<felt252, u64>,
        /// agent_id → actions in current window (for tx_frequency).
        agent_window_count: Map<felt252, u32>,
        /// agent_id → number of distinct targets in current window.
        agent_window_unique_targets: Map<felt252, u32>,
        /// agent_id → running sum of values in current window (for avg).
        agent_window_value_sum: Map<felt252, u64>,
        /// agent_id → max value in current window.
        agent_window_value_max: Map<felt252, u64>,

        // ── Proof replay protection ──────────────────────────────────
        /// proof_hash → whether this proof has been used to resolve an action.
        /// Prevents the same proof from being replayed across multiple actions.
        used_proof_hashes: Map<felt252, bool>,

        // ── Thresholds ───────────────────────────────────────────────
        escalate_threshold: u32,
        block_threshold: u32,
        max_strikes: u32,
    }

    // ── Events ───────────────────────────────────────────────────────

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        AgentRegistered: AgentRegistered,
        AgentDeactivated: AgentDeactivated,
        AgentReactivated: AgentReactivated,
        ActionSubmitted: ActionSubmitted,
        ActionResolved: ActionResolved,
        TrustScoreUpdated: TrustScoreUpdated,
        AgentFrozen: AgentFrozen,
        ThresholdsUpdated: ThresholdsUpdated,
        Paused: Paused,
        Unpaused: Unpaused,
        OwnershipTransferred: OwnershipTransferred,
    }

    #[derive(Drop, starknet::Event)]
    struct AgentRegistered {
        #[key]
        agent_id: felt252,
        owner: ContractAddress,
        registered_at: u64,
    }

    #[derive(Drop, starknet::Event)]
    struct AgentDeactivated {
        #[key]
        agent_id: felt252,
        deactivated_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct AgentReactivated {
        #[key]
        agent_id: felt252,
    }

    #[derive(Drop, starknet::Event)]
    struct ActionSubmitted {
        #[key]
        action_id: u64,
        agent_id: felt252,
        target: felt252,
        value: felt252,
        selector: u32,
        io_commitment: felt252,
    }

    #[derive(Drop, starknet::Event)]
    struct ActionResolved {
        #[key]
        action_id: u64,
        agent_id: felt252,
        decision: u8,
        threat_score: u32,
        proof_hash: felt252,
    }

    #[derive(Drop, starknet::Event)]
    struct TrustScoreUpdated {
        #[key]
        agent_id: felt252,
        old_score: u64,
        new_score: u64,
        raw_score: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct AgentFrozen {
        #[key]
        agent_id: felt252,
        strikes: u32,
        final_trust_score: u64,
    }

    #[derive(Drop, starknet::Event)]
    struct ThresholdsUpdated {
        escalate_threshold: u32,
        block_threshold: u32,
        max_strikes: u32,
        updated_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct Paused {
        by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct Unpaused {
        by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct OwnershipTransferred {
        previous_owner: ContractAddress,
        new_owner: ContractAddress,
    }

    // ── Constructor ──────────────────────────────────────────────────

    #[constructor]
    fn constructor(
        ref self: ContractState,
        owner: ContractAddress,
        verifier_address: ContractAddress,
        classifier_model_id: felt252,
        classifier_weight_root_hash: felt252,
    ) {
        self.owner.write(owner);
        self.verifier_address.write(verifier_address);
        self.classifier_model_id.write(classifier_model_id);
        self.classifier_weight_root_hash.write(classifier_weight_root_hash);
        self.paused.write(false);
        self.max_pending_per_agent.write(10);
        self.escalate_threshold.write(DEFAULT_ESCALATE_THRESHOLD);
        self.block_threshold.write(DEFAULT_BLOCK_THRESHOLD);
        self.max_strikes.write(DEFAULT_MAX_STRIKES);
        self.next_action_id.write(1);
    }

    // ── Implementation ───────────────────────────────────────────────

    #[abi(embed_v0)]
    impl AgentFirewallImpl of super::IAgentFirewall<ContractState> {
        fn register_agent(ref self: ContractState, agent_id: felt252) {
            assert!(!self.agent_registered.entry(agent_id).read(), "AGENT_ALREADY_REGISTERED");

            let caller = get_caller_address();
            let now = get_block_timestamp();

            self.agent_owner.entry(agent_id).write(caller);
            self.agent_trust_score.entry(agent_id).write(0); // starts clean
            self.agent_strikes.entry(agent_id).write(0);
            self.agent_active.entry(agent_id).write(true);
            self.agent_registered.entry(agent_id).write(true);
            self.agent_registered_at.entry(agent_id).write(now);

            self.emit(AgentRegistered { agent_id, owner: caller, registered_at: now });
        }

        fn deactivate_agent(ref self: ContractState, agent_id: felt252) {
            assert!(self.agent_registered.entry(agent_id).read(), "AGENT_NOT_REGISTERED");
            let caller = get_caller_address();
            let agent_owner = self.agent_owner.entry(agent_id).read();
            assert!(
                caller == agent_owner || caller == self.owner.read(),
                "NOT_AGENT_OR_CONTRACT_OWNER"
            );
            self.agent_active.entry(agent_id).write(false);
            self.emit(AgentDeactivated { agent_id, deactivated_by: caller });
        }

        fn reactivate_agent(ref self: ContractState, agent_id: felt252) {
            assert!(self.agent_registered.entry(agent_id).read(), "AGENT_NOT_REGISTERED");
            let caller = get_caller_address();
            assert!(caller == self.agent_owner.entry(agent_id).read(), "NOT_AGENT_OWNER");
            self.agent_active.entry(agent_id).write(true);
            self.agent_strikes.entry(agent_id).write(0); // reset strikes on reactivation
            self.emit(AgentReactivated { agent_id });
        }

        fn submit_action(
            ref self: ContractState,
            agent_id: felt252,
            target: felt252,
            value: felt252,
            selector: u32,
            io_commitment: felt252,
        ) -> u64 {
            // Contract must not be paused
            assert!(!self.paused.read(), "CONTRACT_PAUSED");

            // Agent must be registered and active
            assert!(self.agent_registered.entry(agent_id).read(), "AGENT_NOT_REGISTERED");
            assert!(self.agent_active.entry(agent_id).read(), "AGENT_FROZEN");

            // Caller must be agent owner
            let caller = get_caller_address();
            assert!(caller == self.agent_owner.entry(agent_id).read(), "NOT_AGENT_OWNER");

            // Per-agent rate limit: max pending actions
            let pending = self.agent_pending_count.entry(agent_id).read();
            assert!(pending < self.max_pending_per_agent.read(), "TOO_MANY_PENDING_ACTIONS");
            self.agent_pending_count.entry(agent_id).write(pending + 1);

            // IO commitment must be non-zero (zero is meaningless)
            assert!(io_commitment != 0, "IO_COMMITMENT_ZERO");

            let action_id = self.next_action_id.read();
            self.next_action_id.write(action_id + 1);

            self.action_agent.entry(action_id).write(agent_id);
            self.action_target.entry(action_id).write(target);
            self.action_value.entry(action_id).write(value);
            self.action_selector.entry(action_id).write(selector);
            self.action_io_commitment.entry(action_id).write(io_commitment);
            self.action_resolved.entry(action_id).write(false);
            self.action_decision.entry(action_id).write(0); // pending
            self.action_submitted_at.entry(action_id).write(get_block_timestamp());

            // ── Update behavioral tracking stats ──
            let now = get_block_timestamp();

            // Increment interaction count for (agent, target) pair
            let prev_interactions = self.interaction_count.entry((agent_id, target)).read();
            self.interaction_count.entry((agent_id, target)).write(prev_interactions + 1);

            // Check if behavioral window has expired → reset
            let window_start = self.agent_window_start.entry(agent_id).read();
            if window_start == 0 || now - window_start > BEHAVIORAL_WINDOW {
                // New window — reset all counters
                self.agent_window_start.entry(agent_id).write(now);
                self.agent_window_count.entry(agent_id).write(1);
                self.agent_window_unique_targets.entry(agent_id).write(1);
                let value_u256: u256 = value.into();
                let val_u64: u64 = (value_u256.low & 0xFFFFFFFFFFFFFFFF).try_into().unwrap();
                self.agent_window_value_sum.entry(agent_id).write(val_u64);
                self.agent_window_value_max.entry(agent_id).write(val_u64);
            } else {
                // Same window — accumulate
                let count = self.agent_window_count.entry(agent_id).read();
                self.agent_window_count.entry(agent_id).write(count + 1);

                // Unique targets: if this is first interaction with this target in this window,
                // increment. We use interaction_count == 1 as the signal (just incremented above).
                if prev_interactions == 0 {
                    let unique = self.agent_window_unique_targets.entry(agent_id).read();
                    self.agent_window_unique_targets.entry(agent_id).write(unique + 1);
                }

                // Value stats (truncate u128 to u64 for storage — sufficient for tracking)
                let value_u256: u256 = value.into();
                let val_u64: u64 = (value_u256.low & 0xFFFFFFFFFFFFFFFF).try_into().unwrap();
                let prev_sum = self.agent_window_value_sum.entry(agent_id).read();
                self.agent_window_value_sum.entry(agent_id).write(prev_sum + val_u64);
                let prev_max = self.agent_window_value_max.entry(agent_id).read();
                if val_u64 > prev_max {
                    self.agent_window_value_max.entry(agent_id).write(val_u64);
                }
            }

            self.agent_total_actions.entry(agent_id).write(
                self.agent_total_actions.entry(agent_id).read() + 1
            );

            self.emit(ActionSubmitted { action_id, agent_id, target, value, selector, io_commitment });

            action_id
        }

        fn resolve_action_with_proof(
            ref self: ContractState,
            action_id: u64,
            proof_hash: felt252,
            original_io_len: u32,
            packed_raw_io: Array<felt252>,
        ) {
            // Contract must not be paused
            assert!(!self.paused.read(), "CONTRACT_PAUSED");

            // Action must exist and not be resolved
            let agent_id = self.action_agent.entry(action_id).read();
            assert!(agent_id != 0, "ACTION_NOT_FOUND");
            assert!(!self.action_resolved.entry(action_id).read(), "ACTION_ALREADY_RESOLVED");

            // Action must not be expired (Attack 6: stale action exploitation)
            let submitted_at = self.action_submitted_at.entry(action_id).read();
            let now = get_block_timestamp();
            assert!(now - submitted_at <= MAX_ACTION_AGE, "ACTION_EXPIRED");

            // Caller must be agent owner or contract owner
            let caller = get_caller_address();
            let agent_owner = self.agent_owner.entry(agent_id).read();
            assert!(
                caller == agent_owner || caller == self.owner.read(),
                "NOT_AGENT_OR_CONTRACT_OWNER"
            );

            // Agent must still be active (frozen agents can't resolve)
            assert!(self.agent_active.entry(agent_id).read(), "AGENT_FROZEN");

            // 1. Verify the ZKML proof was verified on ObelyskVerifier
            let verifier = IVerifierDispatcher {
                contract_address: self.verifier_address.read()
            };
            assert!(verifier.is_proof_verified(proof_hash), "PROOF_NOT_VERIFIED");

            // 2. Verify proof hasn't been used for another action (replay protection)
            assert!(!self.used_proof_hashes.entry(proof_hash).read(), "PROOF_ALREADY_USED");
            self.used_proof_hashes.entry(proof_hash).write(true);

            // 3. Validate packed IO data is non-empty and well-formed.
            let packed_span = packed_raw_io.span();
            assert!(packed_span.len() > 0, "PACKED_IO_EMPTY");

            // Minimum: at least 1 felt (holds up to 8 M31 values).
            // For classifier (1x64 → 1x3): 73 M31 values → 10 packed felts.
            // We validate the exact structure below after io_commitment check.

            // 4. Recompute io_commitment from calldata and verify against proof.
            // This is the CRITICAL security check: we don't trust any caller-supplied
            // score — we verify the raw IO data matches the proof, then extract the
            // output neurons ourselves.
            let mut commitment_input: Array<felt252> = array![original_io_len.into()];
            let mut ci: u32 = 0;
            loop {
                if ci >= packed_span.len() {
                    break;
                }
                commitment_input.append(*packed_span.at(ci));
                ci += 1;
            };
            let recomputed_io = core::poseidon::poseidon_hash_span(commitment_input.span());

            // Verify against the stored action io_commitment
            let action_io = self.action_io_commitment.entry(action_id).read();
            assert!(recomputed_io == action_io, "IO_COMMITMENT_MISMATCH");

            // Also verify against the proof's stored io_commitment (belt + suspenders)
            let proof_io = verifier.get_proof_io_commitment(proof_hash);
            assert!(proof_io == action_io, "PROOF_IO_MISMATCH");

            // 5. Validate IO structure and extract output neurons.
            // Layout: [in_rows, in_cols, in_len, ...input_data...,
            //          out_rows, out_cols, out_len, ...output_data...]

            // Total M31 capacity of packed array
            let packed_m31_capacity: u32 = packed_span.len() * 8;

            // Header: must have at least 6 header felts (3 input meta + 3 output meta)
            assert!(original_io_len >= 6, "IO_TOO_SHORT");
            assert!(original_io_len <= packed_m31_capacity, "IO_LEN_EXCEEDS_PACKED");

            // Extract input dimensions
            let in_rows: u32 = extract_m31(packed_span, 0);
            let in_cols: u32 = extract_m31(packed_span, 1);
            let in_len: u32 = extract_m31(packed_span, 2);

            // Validate input dimensions are consistent
            assert!(in_rows == 1, "CLASSIFIER_BATCH_NOT_1");
            assert!(in_len == in_rows * in_cols, "INPUT_DIMENSION_MISMATCH");

            // Output section starts after input header + data
            let out_start: u32 = 3 + in_len;
            // Ensure we can read output header (3 values: out_rows, out_cols, out_len)
            assert!(out_start + 3 <= packed_m31_capacity, "OUTPUT_HEADER_OUT_OF_BOUNDS");

            let out_rows: u32 = extract_m31(packed_span, out_start);
            let out_cols: u32 = extract_m31(packed_span, out_start + 1);
            let out_len: u32 = extract_m31(packed_span, out_start + 2);

            // Validate output dimensions
            assert!(out_rows == 1, "OUTPUT_BATCH_NOT_1");
            assert!(out_cols == 3, "CLASSIFIER_MUST_HAVE_3_OUTPUTS");
            assert!(out_len == 3, "OUTPUT_LEN_MISMATCH");
            assert!(out_len == out_rows * out_cols, "OUTPUT_DIMENSION_MISMATCH");

            // Validate total IO length matches original_io_len
            let expected_io_len: u32 = 3 + in_len + 3 + out_len; // 3 input header + data + 3 output header + data
            assert!(original_io_len == expected_io_len, "IO_LEN_TOTAL_MISMATCH");

            // Ensure we can read all 3 output scores
            let score_start: u32 = out_start + 3;
            assert!(score_start + 3 <= packed_m31_capacity, "SCORES_OUT_OF_BOUNDS");

            // Extract the 3 classifier output neurons
            let score_safe: u64 = extract_m31(packed_span, score_start).into();
            let score_suspicious: u64 = extract_m31(packed_span, score_start + 1).into();
            let score_malicious: u64 = extract_m31(packed_span, score_start + 2).into();

            // 6. Cross-check hard transaction data against action's stored values.
            // Features 0-7 encode the target address as 8 x 31-bit M31 chunks.
            // Feature 16 encodes the function selector (31-bit masked).
            // These are inside the proven IO — an attacker cannot fake them
            // without breaking the io_commitment Poseidon hash.
            let input_start: u32 = 3; // after in_rows, in_cols, in_len header

            // Reconstruct target from 8 M31 chunks (31 bits each, big-endian chunked)
            // The encoding splits felt252 bytes into 4-byte chunks masked to 31 bits.
            // We reconstruct a truncated target and compare against the stored action_target.
            // Full felt252 reconstruction would require 9 chunks (252 bits / 31 = ~9).
            // We verify the lower 248 bits (8 x 31 = 248) which is sufficient for
            // Starknet addresses (251-bit felt252, top 3 bits usually zero).
            let stored_target: felt252 = self.action_target.entry(action_id).read();
            let mut reconstructed: u256 = 0;
            let mut shift: u256 = 1; // 2^0
            let mut chunk_idx: u32 = 0;
            loop {
                if chunk_idx >= 8 {
                    break;
                }
                // Chunks are stored big-endian by the encoder (chunk 0 = MSB bytes)
                // but M31 index 0 = chunk 0. Reconstruct by treating chunk 0 as lowest.
                let chunk_val: u256 = extract_m31(packed_span, input_start + chunk_idx).into();
                reconstructed = reconstructed + chunk_val * shift;
                shift = shift * 0x80000000; // 2^31
                chunk_idx += 1;
            };
            // Compare lower 248 bits of target against reconstructed
            let target_u256: u256 = stored_target.into();
            let mask_248: u256 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF; // 2^248 - 1
            assert!(
                (reconstructed & mask_248) == (target_u256 & mask_248),
                "INPUT_TARGET_MISMATCH"
            );

            // Verify selector: feature 16 at M31 index (input_start + 16)
            let encoded_selector: u32 = extract_m31(packed_span, input_start + 16);
            let stored_selector: u32 = self.action_selector.entry(action_id).read();
            // Selector is masked to 31 bits in the encoder (& 0x7FFFFFFF)
            assert!(
                encoded_selector == (stored_selector & 0x7FFFFFFF),
                "INPUT_SELECTOR_MISMATCH"
            );

            // Verify agent metadata features against on-chain state.
            // These are the most dangerous soft features to fake — an attacker
            // who lies about their trust score gets a more favorable classification.
            //
            // Feature 26: agent_trust_score (clamped to 100000 in encoder)
            // Feature 27: agent_strikes
            // Feature 28: agent_age_blocks
            //
            // We allow a tolerance of ±1 on trust_score because the score may
            // have changed between submit_action and resolve_action_with_proof.
            // Strikes and age are checked exactly.
            let encoded_trust: u32 = extract_m31(packed_span, input_start + 26);
            let onchain_trust: u64 = self.agent_trust_score.entry(agent_id).read();
            let onchain_trust_clamped: u32 = if onchain_trust > 100000 {
                100000
            } else {
                onchain_trust.try_into().unwrap()
            };
            // Allow ±5000 tolerance on trust score (score can change between submit and resolve)
            let trust_diff: u32 = if encoded_trust > onchain_trust_clamped {
                encoded_trust - onchain_trust_clamped
            } else {
                onchain_trust_clamped - encoded_trust
            };
            assert!(trust_diff <= 5000, "INPUT_TRUST_SCORE_MISMATCH");

            // Feature 27: strikes must match exactly
            let encoded_strikes: u32 = extract_m31(packed_span, input_start + 27);
            let onchain_strikes: u32 = self.agent_strikes.entry(agent_id).read();
            assert!(encoded_strikes == onchain_strikes, "INPUT_STRIKES_MISMATCH");

            // Feature 28: agent age — compute from registration timestamp
            // Allow ±100 blocks tolerance (block time variance)
            let registered_at: u64 = self.agent_registered_at.entry(agent_id).read();
            let current_time: u64 = get_block_timestamp();
            let onchain_age: u64 = if current_time > registered_at {
                current_time - registered_at
            } else {
                0
            };
            let encoded_age: u32 = extract_m31(packed_span, input_start + 28);
            let onchain_age_u32: u32 = if onchain_age > 0x7FFFFFFF {
                0x7FFFFFFF
            } else {
                onchain_age.try_into().unwrap()
            };
            let age_diff: u32 = if encoded_age > onchain_age_u32 {
                encoded_age - onchain_age_u32
            } else {
                onchain_age_u32 - encoded_age
            };
            assert!(age_diff <= 100, "INPUT_AGE_MISMATCH");

            // Verify value_features (indices 33-36) — derivable from stored action_value.
            // Feature 33: log2(value + 1)
            let stored_value: felt252 = self.action_value.entry(action_id).read();
            let value_u256: u256 = stored_value.into();
            let value_low: u128 = value_u256.low;
            let expected_log2: u32 = bit_length_u128(value_low);
            let encoded_log2: u32 = extract_m31(packed_span, input_start + 33);
            assert!(encoded_log2 == expected_log2, "INPUT_LOG2_VALUE_MISMATCH");

            // Feature 35: is_max_approval (value == u128::MAX)
            let expected_max_approval: u32 = if value_low == 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF {
                1
            } else {
                0
            };
            let encoded_max_approval: u32 = extract_m31(packed_span, input_start + 35);
            assert!(encoded_max_approval == expected_max_approval, "INPUT_MAX_APPROVAL_MISMATCH");

            // Feature 36: is_zero_value
            let expected_zero: u32 = if value_low == 0 { 1 } else { 0 };
            let encoded_zero: u32 = extract_m31(packed_span, input_start + 36);
            assert!(encoded_zero == expected_zero, "INPUT_ZERO_VALUE_MISMATCH");

            // Verify selector_features (indices 37-40) — derivable from stored selector.
            let sel: u32 = self.action_selector.entry(action_id).read();

            // Feature 37: is_transfer (ERC20 transfer / transferFrom)
            let expected_transfer: u32 = if sel == 0xa9059cbb || sel == 0x23b872dd { 1 } else { 0 };
            let encoded_transfer: u32 = extract_m31(packed_span, input_start + 37);
            assert!(encoded_transfer == expected_transfer, "INPUT_IS_TRANSFER_MISMATCH");

            // Feature 38: is_approve
            let expected_approve: u32 = if sel == 0x095ea7b3 { 1 } else { 0 };
            let encoded_approve: u32 = extract_m31(packed_span, input_start + 38);
            assert!(encoded_approve == expected_approve, "INPUT_IS_APPROVE_MISMATCH");

            // Feature 39: is_swap (Uniswap/Sushi common selectors)
            let expected_swap: u32 = if sel == 0x38ed1739 || sel == 0x7ff36ab5 || sel == 0x18cbafe5 {
                1
            } else {
                0
            };
            let encoded_swap: u32 = extract_m31(packed_span, input_start + 39);
            assert!(encoded_swap == expected_swap, "INPUT_IS_SWAP_MISMATCH");

            // Feature 40: is_unknown (selector == 0)
            let expected_unknown: u32 = if sel == 0 { 1 } else { 0 };
            let encoded_unknown: u32 = extract_m31(packed_span, input_start + 40);
            assert!(encoded_unknown == expected_unknown, "INPUT_IS_UNKNOWN_MISMATCH");

            // Verify behavioral features (indices 41-44) from contract-internal tracking.
            // Feature 32: interaction_count for this (agent, target) pair
            let stored_target_for_behavioral: felt252 = self.action_target.entry(action_id).read();
            let onchain_interactions: u32 = self.interaction_count.entry(
                (agent_id, stored_target_for_behavioral)
            ).read();
            let encoded_interactions: u32 = extract_m31(packed_span, input_start + 32);
            // Allow ±5 tolerance (interactions may have changed between submit and resolve)
            let interact_diff: u32 = if encoded_interactions > onchain_interactions {
                encoded_interactions - onchain_interactions
            } else {
                onchain_interactions - encoded_interactions
            };
            assert!(interact_diff <= 5, "INPUT_INTERACTION_COUNT_MISMATCH");

            // Feature 41: tx_frequency (actions in current window)
            let onchain_freq: u32 = self.agent_window_count.entry(agent_id).read();
            let encoded_freq: u32 = extract_m31(packed_span, input_start + 41);
            let freq_diff: u32 = if encoded_freq > onchain_freq {
                encoded_freq - onchain_freq
            } else {
                onchain_freq - encoded_freq
            };
            assert!(freq_diff <= 5, "INPUT_TX_FREQUENCY_MISMATCH");

            // Feature 42: unique_targets_24h
            let onchain_unique: u32 = self.agent_window_unique_targets.entry(agent_id).read();
            let encoded_unique: u32 = extract_m31(packed_span, input_start + 42);
            let unique_diff: u32 = if encoded_unique > onchain_unique {
                encoded_unique - onchain_unique
            } else {
                onchain_unique - encoded_unique
            };
            assert!(unique_diff <= 3, "INPUT_UNIQUE_TARGETS_MISMATCH");

            // Features 43-44: avg_value and max_value — allow wider tolerance
            // because these are aggregates that can shift significantly between
            // submit and resolve. We verify they're in the right ballpark.
            // A ±50% tolerance prevents gross fabrication while allowing natural drift.
            let onchain_max: u64 = self.agent_window_value_max.entry(agent_id).read();
            let encoded_max: u32 = extract_m31(packed_span, input_start + 44);
            let encoded_max_u64: u64 = encoded_max.into();
            // Max value: encoded should not exceed 2x on-chain max (anti-fabrication)
            if onchain_max > 0 && encoded_max_u64 > 0 {
                assert!(encoded_max_u64 <= onchain_max * 2 + 1, "INPUT_MAX_VALUE_IMPLAUSIBLE");
            }

            // 7. Compute threat score on-chain (not caller-supplied!)
            let total: u64 = score_safe + score_suspicious + score_malicious;
            let threat_score: u32 = if total == 0 {
                50000 // ambiguous → escalate by default
            } else {
                // threat = (malicious / total) * 100000
                // Max: 100000 (when malicious == total), always fits u32
                let raw: u64 = (score_malicious * 100000) / total;
                // Defensive clamp (mathematically unnecessary but safe)
                if raw > 100000 { 100000 } else { raw.try_into().unwrap() }
            };

            // 6. Verify the proof came from the registered classifier model.
            let model_id = self.classifier_model_id.read();
            let proof_model = verifier.get_proof_model_id(proof_hash);
            assert!(proof_model == model_id, "MODEL_ID_MISMATCH");

            // 7. Verify the model's weight commitments match the expected hash.
            // This prevents model substitution: even if the contract owner is
            // compromised and changes classifier_model_id, the weight hash
            // must still match. An attacker cannot register a rigged model
            // with different weights without changing this hash.
            let expected_weights = self.classifier_weight_root_hash.read();
            if expected_weights != 0 {
                let actual_weights = verifier.get_model_weight_root_hash(model_id);
                assert!(actual_weights == expected_weights, "WEIGHT_ROOT_HASH_MISMATCH");
            }

            // 8. Verify the classifier model has a policy registered (strict required)
            let registered_policy = verifier.get_model_policy(model_id);
            assert!(registered_policy != 0, "NO_POLICY_REGISTERED");

            // 7. Apply decision based on thresholds
            let block_threshold = self.block_threshold.read();
            let escalate_threshold = self.escalate_threshold.read();

            let decision: u8 = if threat_score >= block_threshold {
                3 // block
            } else if threat_score >= escalate_threshold {
                2 // escalate
            } else {
                1 // approve
            };

            // 8. Update EMA trust score with ASYMMETRIC decay.
            // Score going UP (bad action): alpha = 0.5 (fast)
            // Score going DOWN (safe action): alpha = 0.1 (slow)
            // This prevents EMA dilution — an attacker can't quickly wash
            // away a high score by submitting safe transactions.
            let prev_score = self.agent_trust_score.entry(agent_id).read();
            let threat_u64: u64 = threat_score.into();
            let alpha_num = if threat_u64 > prev_score {
                EMA_ALPHA_UP_NUM    // 0.5 — bad scores hit hard
            } else {
                EMA_ALPHA_DOWN_NUM  // 0.1 — safe scores forgive slowly
            };
            let new_score = (alpha_num * threat_u64
                + (EMA_ALPHA_DEN - alpha_num) * prev_score)
                / EMA_ALPHA_DEN;
            self.agent_trust_score.entry(agent_id).write(new_score);
            self.emit(TrustScoreUpdated {
                agent_id, old_score: prev_score, new_score, raw_score: threat_score
            });

            // 9. Strike mechanism (strikes on escalate or block)
            if threat_score >= escalate_threshold {
                let strikes = self.agent_strikes.entry(agent_id).read() + 1;
                self.agent_strikes.entry(agent_id).write(strikes);

                // Auto-freeze at max strikes
                if strikes >= self.max_strikes.read() {
                    self.agent_active.entry(agent_id).write(false);
                    self.emit(AgentFrozen {
                        agent_id, strikes, final_trust_score: new_score
                    });
                }
            }

            // 10. Record resolution + decrement pending count
            self.action_decision.entry(action_id).write(decision);
            self.action_threat_score.entry(action_id).write(threat_score);
            self.action_proof_hash.entry(action_id).write(proof_hash);
            self.action_resolved.entry(action_id).write(true);

            let pending = self.agent_pending_count.entry(agent_id).read();
            if pending > 0 {
                self.agent_pending_count.entry(agent_id).write(pending - 1);
            }

            self.emit(ActionResolved {
                action_id, agent_id, decision, threat_score, proof_hash
            });
        }

        fn approve_escalated(ref self: ContractState, action_id: u64) {
            let agent_id = self.action_agent.entry(action_id).read();
            assert!(agent_id != 0, "ACTION_NOT_FOUND");
            assert!(self.action_decision.entry(action_id).read() == 2, "NOT_ESCALATED");

            // Escalated actions also expire
            let submitted_at = self.action_submitted_at.entry(action_id).read();
            assert!(get_block_timestamp() - submitted_at <= MAX_ACTION_AGE, "ACTION_EXPIRED");

            // Agent owner OR contract owner can approve escalated
            let caller = get_caller_address();
            let agent_owner = self.agent_owner.entry(agent_id).read();
            assert!(
                caller == agent_owner || caller == self.owner.read(),
                "NOT_AGENT_OR_CONTRACT_OWNER"
            );

            self.action_decision.entry(action_id).write(1); // approved

            // Decrement pending count
            let pending = self.agent_pending_count.entry(agent_id).read();
            if pending > 0 {
                self.agent_pending_count.entry(agent_id).write(pending - 1);
            }

            self.emit(ActionResolved {
                action_id, agent_id, decision: 1,
                threat_score: self.action_threat_score.entry(action_id).read(),
                proof_hash: self.action_proof_hash.entry(action_id).read(),
            });
        }

        fn reject_escalated(ref self: ContractState, action_id: u64) {
            let agent_id = self.action_agent.entry(action_id).read();
            assert!(agent_id != 0, "ACTION_NOT_FOUND");
            assert!(self.action_decision.entry(action_id).read() == 2, "NOT_ESCALATED");

            // Escalated actions also expire
            let submitted_at = self.action_submitted_at.entry(action_id).read();
            assert!(get_block_timestamp() - submitted_at <= MAX_ACTION_AGE, "ACTION_EXPIRED");

            // Agent owner OR contract owner can reject
            let caller = get_caller_address();
            let agent_owner = self.agent_owner.entry(agent_id).read();
            assert!(
                caller == agent_owner || caller == self.owner.read(),
                "NOT_AGENT_OR_CONTRACT_OWNER"
            );

            self.action_decision.entry(action_id).write(3); // blocked

            // Rejection = 1 additional strike (owner agrees it was suspicious)
            let strikes = self.agent_strikes.entry(agent_id).read() + 1;
            self.agent_strikes.entry(agent_id).write(strikes);
            if strikes >= self.max_strikes.read() {
                self.agent_active.entry(agent_id).write(false);
                let trust = self.agent_trust_score.entry(agent_id).read();
                self.emit(AgentFrozen { agent_id, strikes, final_trust_score: trust });
            }

            // Decrement pending count
            let pending = self.agent_pending_count.entry(agent_id).read();
            if pending > 0 {
                self.agent_pending_count.entry(agent_id).write(pending - 1);
            }

            self.emit(ActionResolved {
                action_id, agent_id, decision: 3,
                threat_score: self.action_threat_score.entry(action_id).read(),
                proof_hash: self.action_proof_hash.entry(action_id).read(),
            });
        }

        // ── Admin ─────────────────────────────────────────────────────

        fn set_thresholds(
            ref self: ContractState,
            escalate_threshold: u32,
            block_threshold: u32,
            max_strikes: u32,
        ) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            assert!(escalate_threshold < block_threshold, "ESCALATE_MUST_BE_LESS_THAN_BLOCK");
            assert!(block_threshold <= 100000, "BLOCK_THRESHOLD_OUT_OF_RANGE");
            assert!(max_strikes > 0, "MAX_STRIKES_MUST_BE_POSITIVE");
            self.escalate_threshold.write(escalate_threshold);
            self.block_threshold.write(block_threshold);
            self.max_strikes.write(max_strikes);
            self.emit(ThresholdsUpdated {
                escalate_threshold, block_threshold, max_strikes,
                updated_by: get_caller_address(),
            });
        }

        fn pause(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            assert!(!self.paused.read(), "ALREADY_PAUSED");
            self.paused.write(true);
            self.emit(Paused { by: get_caller_address() });
        }

        fn unpause(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            assert!(self.paused.read(), "NOT_PAUSED");
            self.paused.write(false);
            self.emit(Unpaused { by: get_caller_address() });
        }

        fn transfer_ownership(ref self: ContractState, new_owner: ContractAddress) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            assert!(new_owner != zero_addr, "NEW_OWNER_CANNOT_BE_ZERO");
            self.pending_owner.write(new_owner);
        }

        fn accept_ownership(ref self: ContractState) {
            let caller = get_caller_address();
            assert!(caller == self.pending_owner.read(), "NOT_PENDING_OWNER");
            let previous = self.owner.read();
            self.owner.write(caller);
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            self.pending_owner.write(zero_addr);
            self.emit(OwnershipTransferred { previous_owner: previous, new_owner: caller });
        }

        fn set_verifier(ref self: ContractState, verifier_address: ContractAddress) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            assert!(verifier_address != zero_addr, "VERIFIER_CANNOT_BE_ZERO");
            self.verifier_address.write(verifier_address);
        }

        fn set_classifier_model(
            ref self: ContractState, model_id: felt252, weight_root_hash: felt252,
        ) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            assert!(model_id != 0, "MODEL_ID_CANNOT_BE_ZERO");
            assert!(weight_root_hash != 0, "WEIGHT_HASH_CANNOT_BE_ZERO");
            self.classifier_model_id.write(model_id);
            self.classifier_weight_root_hash.write(weight_root_hash);
        }

        // ── Queries ──────────────────────────────────────────────────

        fn is_action_approved(self: @ContractState, action_id: u64) -> bool {
            self.action_resolved.entry(action_id).read()
                && self.action_decision.entry(action_id).read() == 1
        }

        fn is_trusted(self: @ContractState, agent_id: felt252) -> bool {
            self.agent_active.entry(agent_id).read()
                && self.agent_trust_score.entry(agent_id).read()
                    < self.block_threshold.read().into()
                && self.agent_strikes.entry(agent_id).read()
                    < self.max_strikes.read()
        }

        fn get_trust_score(self: @ContractState, agent_id: felt252) -> u64 {
            self.agent_trust_score.entry(agent_id).read()
        }

        fn get_strikes(self: @ContractState, agent_id: felt252) -> u32 {
            self.agent_strikes.entry(agent_id).read()
        }

        fn is_agent_active(self: @ContractState, agent_id: felt252) -> bool {
            self.agent_active.entry(agent_id).read()
        }

        fn is_agent_registered(self: @ContractState, agent_id: felt252) -> bool {
            self.agent_registered.entry(agent_id).read()
        }

        fn get_agent_owner(self: @ContractState, agent_id: felt252) -> ContractAddress {
            self.agent_owner.entry(agent_id).read()
        }

        fn get_action_agent(self: @ContractState, action_id: u64) -> felt252 {
            self.action_agent.entry(action_id).read()
        }

        fn get_action_decision(self: @ContractState, action_id: u64) -> u8 {
            self.action_decision.entry(action_id).read()
        }

        fn get_action_io_commitment(self: @ContractState, action_id: u64) -> felt252 {
            self.action_io_commitment.entry(action_id).read()
        }

        fn get_action_threat_score(self: @ContractState, action_id: u64) -> u32 {
            self.action_threat_score.entry(action_id).read()
        }

        fn get_owner(self: @ContractState) -> ContractAddress {
            self.owner.read()
        }

        fn get_verifier(self: @ContractState) -> ContractAddress {
            self.verifier_address.read()
        }

        fn get_classifier_model_id(self: @ContractState) -> felt252 {
            self.classifier_model_id.read()
        }

        fn get_thresholds(self: @ContractState) -> (u32, u32, u32) {
            (
                self.escalate_threshold.read(),
                self.block_threshold.read(),
                self.max_strikes.read(),
            )
        }

        fn is_paused(self: @ContractState) -> bool {
            self.paused.read()
        }
    }
}
