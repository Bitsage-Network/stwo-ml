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

    /// Update verifier contract address (contract owner only).
    fn set_verifier(ref self: TContractState, verifier_address: ContractAddress);

    /// Update classifier model ID (contract owner only).
    fn set_classifier_model(ref self: TContractState, model_id: felt252);

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
}

/// Extract a single M31 value (31-bit unsigned) from packed felt252 array.
/// Each felt252 holds 8 M31 values. Re-exported from crate::field for firewall use.
fn extract_m31(packed_felts: Span<felt252>, m31_index: u32) -> u32 {
    crate::field::extract_m31_from_packed(packed_felts, m31_index)
}

#[starknet::contract]
pub mod AgentFirewallZK {
    use starknet::storage::{
        StoragePointerReadAccess, StoragePointerWriteAccess, Map, StoragePathEntry,
    };
    use starknet::{ContractAddress, get_caller_address, get_block_timestamp};
    use super::{IVerifierDispatcher, IVerifierDispatcherTrait, extract_m31};

    // ── Constants ────────────────────────────────────────────────────

    /// EMA alpha numerator (300 / 1000 = 0.3, same as ENShell).
    const EMA_ALPHA_NUM: u64 = 300;
    /// EMA alpha denominator.
    const EMA_ALPHA_DEN: u64 = 1000;

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
        /// ObelyskVerifier contract address.
        verifier_address: ContractAddress,
        /// Classifier model ID registered on the verifier.
        classifier_model_id: felt252,

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
        action_io_commitment: Map<u64, felt252>,
        action_resolved: Map<u64, bool>,
        /// 0=pending, 1=approved, 2=escalated, 3=blocked
        action_decision: Map<u64, u8>,
        action_threat_score: Map<u64, u32>,
        action_proof_hash: Map<u64, felt252>,
        action_submitted_at: Map<u64, u64>,

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

    // ── Constructor ──────────────────────────────────────────────────

    #[constructor]
    fn constructor(
        ref self: ContractState,
        owner: ContractAddress,
        verifier_address: ContractAddress,
        classifier_model_id: felt252,
    ) {
        self.owner.write(owner);
        self.verifier_address.write(verifier_address);
        self.classifier_model_id.write(classifier_model_id);
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
            io_commitment: felt252,
        ) -> u64 {
            // Agent must be registered and active
            assert!(self.agent_registered.entry(agent_id).read(), "AGENT_NOT_REGISTERED");
            assert!(self.agent_active.entry(agent_id).read(), "AGENT_FROZEN");

            // Caller must be agent owner
            let caller = get_caller_address();
            assert!(caller == self.agent_owner.entry(agent_id).read(), "NOT_AGENT_OWNER");

            // IO commitment must be non-zero (zero is meaningless)
            assert!(io_commitment != 0, "IO_COMMITMENT_ZERO");

            let action_id = self.next_action_id.read();
            self.next_action_id.write(action_id + 1);

            self.action_agent.entry(action_id).write(agent_id);
            self.action_target.entry(action_id).write(target);
            self.action_value.entry(action_id).write(value);
            self.action_io_commitment.entry(action_id).write(io_commitment);
            self.action_resolved.entry(action_id).write(false);
            self.action_decision.entry(action_id).write(0); // pending
            self.action_submitted_at.entry(action_id).write(get_block_timestamp());

            self.emit(ActionSubmitted { action_id, agent_id, target, value, io_commitment });

            action_id
        }

        fn resolve_action_with_proof(
            ref self: ContractState,
            action_id: u64,
            proof_hash: felt252,
            original_io_len: u32,
            packed_raw_io: Array<felt252>,
        ) {
            // Action must exist and not be resolved
            let agent_id = self.action_agent.entry(action_id).read();
            assert!(agent_id != 0, "ACTION_NOT_FOUND");
            assert!(!self.action_resolved.entry(action_id).read(), "ACTION_ALREADY_RESOLVED");

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

            // 3. Recompute io_commitment from calldata and verify against proof.
            // This is the CRITICAL security check: we don't trust any caller-supplied
            // score — we verify the raw IO data matches the proof, then extract the
            // output neurons ourselves.
            let mut commitment_input: Array<felt252> = array![original_io_len.into()];
            let packed_span = packed_raw_io.span();
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

            // 4. Extract output neurons from packed IO data.
            // Layout: [in_rows, in_cols, in_len, ...input..., out_rows, out_cols, out_len, score0, score1, score2]
            let in_len: u32 = extract_m31(packed_span, 2);
            let out_start: u32 = 3 + in_len;
            let out_len: u32 = extract_m31(packed_span, out_start + 2);
            assert!(out_len >= 3, "CLASSIFIER_OUTPUT_TOO_SHORT");

            let score_safe: u64 = extract_m31(packed_span, out_start + 3).into();
            let score_suspicious: u64 = extract_m31(packed_span, out_start + 4).into();
            let score_malicious: u64 = extract_m31(packed_span, out_start + 5).into();

            // 5. Compute threat score on-chain (not caller-supplied!)
            let total = score_safe + score_suspicious + score_malicious;
            let threat_score: u32 = if total == 0 {
                50000 // ambiguous → escalate
            } else {
                // threat = (malicious / total) * 100000
                ((score_malicious * 100000) / total).try_into().unwrap()
            };

            // 6. Verify the proof came from the registered classifier model.
            let model_id = self.classifier_model_id.read();
            let proof_model = verifier.get_proof_model_id(proof_hash);
            assert!(proof_model == model_id, "MODEL_ID_MISMATCH");

            // 7. Verify the classifier model has a policy registered (strict required)
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

            // 8. Update EMA trust score (alpha = 0.3)
            let prev_score = self.agent_trust_score.entry(agent_id).read();
            let new_score = (EMA_ALPHA_NUM * threat_score.into()
                + (EMA_ALPHA_DEN - EMA_ALPHA_NUM) * prev_score)
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

            // 10. Record resolution
            self.action_decision.entry(action_id).write(decision);
            self.action_threat_score.entry(action_id).write(threat_score);
            self.action_proof_hash.entry(action_id).write(proof_hash);
            self.action_resolved.entry(action_id).write(true);

            self.emit(ActionResolved {
                action_id, agent_id, decision, threat_score, proof_hash
            });
        }

        fn approve_escalated(ref self: ContractState, action_id: u64) {
            let agent_id = self.action_agent.entry(action_id).read();
            assert!(agent_id != 0, "ACTION_NOT_FOUND");
            assert!(self.action_decision.entry(action_id).read() == 2, "NOT_ESCALATED");

            let caller = get_caller_address();
            assert!(caller == self.agent_owner.entry(agent_id).read(), "NOT_AGENT_OWNER");

            self.action_decision.entry(action_id).write(1); // approved
            self.emit(ActionResolved {
                action_id, agent_id, decision: 1, threat_score: 0, proof_hash: 0
            });
        }

        fn reject_escalated(ref self: ContractState, action_id: u64) {
            let agent_id = self.action_agent.entry(action_id).read();
            assert!(agent_id != 0, "ACTION_NOT_FOUND");
            assert!(self.action_decision.entry(action_id).read() == 2, "NOT_ESCALATED");

            let caller = get_caller_address();
            assert!(caller == self.agent_owner.entry(agent_id).read(), "NOT_AGENT_OWNER");

            self.action_decision.entry(action_id).write(3); // blocked
            self.emit(ActionResolved {
                action_id, agent_id, decision: 3, threat_score: 0, proof_hash: 0
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

        fn set_verifier(ref self: ContractState, verifier_address: ContractAddress) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            assert!(verifier_address != zero_addr, "VERIFIER_CANNOT_BE_ZERO");
            self.verifier_address.write(verifier_address);
        }

        fn set_classifier_model(ref self: ContractState, model_id: felt252) {
            assert!(get_caller_address() == self.owner.read(), "ONLY_OWNER");
            assert!(model_id != 0, "MODEL_ID_CANNOT_BE_ZERO");
            self.classifier_model_id.write(model_id);
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
    }
}
