// Generic On-Chain Sumcheck Verifier for ML Matrix Multiplication
//
// Verifies stwo-ml matmul proofs directly on Starknet.
// No token dependencies, no payment logic — pure cryptographic verification.
//
// Full transcript replay:
//   1. mix_u64(m), mix_u64(k), mix_u64(n)        — bind matrix dimensions
//   2. draw_qm31s(m_log), draw_qm31s(n_log)      — row/col challenges
//   3. mix_felt(pack(claimed_sum))                 — bind claimed evaluation
//   4. mix_felt(a_commitment), mix_felt(b_commitment) — bind MLE commitments
//   5. For each sumcheck round: verify + Fiat-Shamir
//   6. Final check: expected_sum = a_eval × b_eval
//   7. verify_mle_opening(A, assignment, channel)
//   8. verify_mle_opening(B, assignment, channel)

use crate::types::{GKRClaim, MleOpeningProof};
use crate::field::QM31;
use crate::vm31_merkle::PackedDigest;
use starknet::ClassHash;

/// Minimum delay (seconds) between propose_upgrade and execute_upgrade.
pub const UPGRADE_DELAY: u64 = 300; // 5 minutes

#[starknet::interface]
pub trait ISumcheckVerifier<TContractState> {
    /// Register a model's weight commitment on-chain.
    fn register_model(
        ref self: TContractState, model_id: felt252, weight_commitment: felt252,
    );

    /// Get the weight commitment for a registered model.
    fn get_model_commitment(self: @TContractState, model_id: felt252) -> felt252;

    /// Get the number of verified proofs for a model.
    fn get_verification_count(self: @TContractState, model_id: felt252) -> u64;

    /// Check if a specific proof hash has been verified.
    fn is_proof_verified(self: @TContractState, proof_hash: felt252) -> bool;

    /// Bind a verified proof hash to a VM31 batch public-input hash (owner only).
    fn bind_vm31_public_hash(
        ref self: TContractState, proof_hash: felt252, vm31_public_hash: PackedDigest,
    );

    /// Get the VM31 batch public-input hash bound to this proof hash.
    fn get_vm31_public_hash(self: @TContractState, proof_hash: felt252) -> PackedDigest;

    /// Get current VM31 binder role address.
    fn get_vm31_binder(self: @TContractState) -> starknet::ContractAddress;

    /// Set VM31 binder role (owner only).
    fn set_vm31_binder(ref self: TContractState, binder: starknet::ContractAddress);

    /// Get the contract owner.
    fn get_owner(self: @TContractState) -> starknet::ContractAddress;

    /// Propose a contract upgrade. Owner only. Starts the 5-min timelock.
    fn propose_upgrade(ref self: TContractState, new_class_hash: ClassHash);

    /// Execute a proposed upgrade after the timelock expires. Owner only.
    fn execute_upgrade(ref self: TContractState);

    /// Cancel a pending upgrade. Owner only.
    fn cancel_upgrade(ref self: TContractState);

    /// Get pending upgrade info: (class_hash, proposed_at_timestamp).
    /// Returns (0, 0) if no upgrade is pending.
    fn get_pending_upgrade(self: @TContractState) -> (ClassHash, u64);

    /// Register a model for full GKR verification (owner only).
    ///
    /// Stores per-MatMul weight commitments and a circuit descriptor hash.
    /// The circuit descriptor encodes layer types + dimensions (agreed off-chain).
    fn register_model_gkr(
        ref self: TContractState,
        model_id: felt252,
        weight_commitments: Array<felt252>,
        circuit_descriptor: Array<u32>,
    );

    /// Full on-chain ZKML verification via ML GKR walk with input claim verification.
    ///
    /// No STARK, no FRI, no dicts — only field ops + Poseidon + sumcheck.
    /// Walks all model layers output → input, verifying each via tag-dispatched
    /// per-layer verifiers (MatMul, Add, Mul, Activation, LayerNorm, etc.).
    ///
    /// **Soundness anchors (input claim verification):**
    ///   - OUTPUT side: draws r_out from channel, evaluates MLE(raw_output, r_out)
    ///     on-chain, uses the result as the initial claim value.
    ///   - INPUT side: after the GKR walk returns final_claim, evaluates
    ///     MLE(raw_input, final_claim.point) and asserts == final_claim.value.
    ///   - IO commitment: recomputed from raw_io_data via Poseidon on-chain.
    ///
    /// Parameters:
    ///   - model_id: registered model identifier
    ///   - raw_io_data: serialized [in_rows, in_cols, in_len, in_data...,
    ///     out_rows, out_cols, out_len, out_data...] for IO commitment + MLE eval
    ///   - circuit_depth: total number of layers in the circuit (including Identity)
    ///     Used for Fiat-Shamir channel seeding — must match prover's circuit.layers.len()
    ///   - num_layers: number of proof layers (excluding Identity/Input)
    ///   - matmul_dims: flat [m0,k0,n0, m1,k1,n1, ...] per MatMul layer
    ///   - dequantize_bits: [bits0, bits1, ...] per Dequantize layer
    ///   - proof_data: flat felt252 array of tag-dispatched per-layer proofs
    ///   - weight_commitments: Poseidon Merkle roots of weight MLEs
    fn verify_model_gkr(
        ref self: TContractState,
        model_id: felt252,
        raw_io_data: Array<felt252>,
        circuit_depth: u32,
        num_layers: u32,
        matmul_dims: Array<u32>,
        dequantize_bits: Array<u64>,
        proof_data: Array<felt252>,
        weight_commitments: Array<felt252>,
        weight_opening_proofs: Array<MleOpeningProof>,
    ) -> bool;

    /// Versioned full on-chain ZKML verification path.
    ///
    /// Supported `weight_binding_mode` values:
    ///   - `0`: sequential opening transcript (v1-compatible)
    ///   - `1`: batched sub-channel opening transcript
    fn verify_model_gkr_v2(
        ref self: TContractState,
        model_id: felt252,
        raw_io_data: Array<felt252>,
        circuit_depth: u32,
        num_layers: u32,
        matmul_dims: Array<u32>,
        dequantize_bits: Array<u64>,
        proof_data: Array<felt252>,
        weight_commitments: Array<felt252>,
        weight_binding_mode: u32,
        weight_opening_proofs: Array<MleOpeningProof>,
    ) -> bool;

    /// Phase-3 versioned interface for trustless aggregated weight binding.
    ///
    /// Supported `weight_binding_mode` values:
    ///   - `0`: sequential opening transcript (v1-compatible)
    ///   - `1`: batched sub-channel opening transcript
    ///   - `2`: aggregated trustless binding (v3 payload + opening checks, sub-channel opening transcript)
    ///
    /// `weight_binding_data`:
    ///   - mode `0|1`: must be empty
    ///   - mode `2`: `[binding_digest, claim_count]`
    fn verify_model_gkr_v3(
        ref self: TContractState,
        model_id: felt252,
        raw_io_data: Array<felt252>,
        circuit_depth: u32,
        num_layers: u32,
        matmul_dims: Array<u32>,
        dequantize_bits: Array<u64>,
        proof_data: Array<felt252>,
        weight_commitments: Array<felt252>,
        weight_binding_mode: u32,
        weight_binding_data: Array<felt252>,
        weight_opening_proofs: Array<MleOpeningProof>,
    ) -> bool;

    /// Phase-4 versioned interface for aggregated weight binding.
    ///
    /// Supported `weight_binding_mode` values:
    ///   - `3`: aggregated openings experimental envelope
    ///   - `4`: aggregated oracle mismatch sumcheck (production default)
    ///
    /// `weight_binding_data`:
    ///   - mode `3`: `[binding_digest, claim_count]`
    ///   - mode `4`: `[n_claims, n_max, m_padded, super_root, subtree_roots..., mismatch_sumcheck_proof...]`
    fn verify_model_gkr_v4(
        ref self: TContractState,
        model_id: felt252,
        raw_io_data: Array<felt252>,
        circuit_depth: u32,
        num_layers: u32,
        matmul_dims: Array<u32>,
        dequantize_bits: Array<u64>,
        proof_data: Array<felt252>,
        weight_commitments: Array<felt252>,
        weight_binding_mode: u32,
        weight_binding_data: Array<felt252>,
        weight_opening_proofs: Array<MleOpeningProof>,
    ) -> bool;

    /// Phase-4 versioned interface with packed QM31 proof data.
    ///
    /// Identical to `verify_model_gkr_v4` except `proof_data` uses packed QM31
    /// format (1 felt252 per QM31 instead of 4 u64s), reducing calldata by ~3.3x.
    fn verify_model_gkr_v4_packed(
        ref self: TContractState,
        model_id: felt252,
        raw_io_data: Array<felt252>,
        circuit_depth: u32,
        num_layers: u32,
        matmul_dims: Array<u32>,
        dequantize_bits: Array<u64>,
        proof_data: Array<felt252>,
        weight_commitments: Array<felt252>,
        weight_binding_mode: u32,
        weight_binding_data: Array<felt252>,
        weight_opening_proofs: Array<MleOpeningProof>,
    ) -> bool;

    /// Get the circuit descriptor hash for a GKR-registered model.
    fn get_model_circuit_hash(self: @TContractState, model_id: felt252) -> felt252;

    /// Get the number of GKR weight commitments for a model.
    fn get_model_gkr_weight_count(self: @TContractState, model_id: felt252) -> u32;

    // ─── Chunked GKR Session Protocol ────────────────────────────────────

    /// Open a new GKR upload session. Returns the session_id.
    ///
    /// The caller becomes the session owner and is the only address
    /// allowed to upload chunks.
    fn open_gkr_session(
        ref self: TContractState,
        model_id: felt252,
        total_felts: u32,
        circuit_depth: u32,
        num_layers: u32,
        weight_binding_mode: u32,
        packed: bool,
    ) -> felt252;

    /// Upload a chunk of proof data to an open session.
    ///
    /// Chunks must be uploaded in order (`chunk_idx == chunks_received`).
    /// Each chunk may contain at most 4000 felt252 values.
    fn upload_gkr_chunk(
        ref self: TContractState,
        session_id: felt252,
        chunk_idx: u32,
        data: Array<felt252>,
    );

    /// Seal a session after all chunks have been uploaded.
    ///
    /// Validates that the total received felts match the declared total.
    fn seal_gkr_session(ref self: TContractState, session_id: felt252);

    /// Verify a sealed session's proof data from storage.
    ///
    /// Reads all data back from `gkr_session_data`, deserializes into
    /// the standard verify_model_gkr_core parameters, and runs verification.
    fn verify_gkr_from_session(ref self: TContractState, session_id: felt252) -> bool;

    /// Expire a timed-out session and clean up metadata.
    ///
    /// Anyone can call this after `GKR_SESSION_TIMEOUT_BLOCKS` (~1000 blocks).
    /// Does NOT clean up data storage (too expensive); only resets metadata
    /// so the session_id cannot be used.
    fn expire_gkr_session(ref self: TContractState, session_id: felt252);

    /// Get the status of a GKR session (0=none, 1=uploading, 2=sealed, 3=verified).
    fn get_gkr_session_status(self: @TContractState, session_id: felt252) -> u8;
}

#[starknet::contract]
mod SumcheckVerifierContract {
    use super::{
        GKRClaim, MleOpeningProof, QM31, UPGRADE_DELAY,
    };
    use crate::field::{
        log2_ceil, next_power_of_two, pack_qm31_to_felt,
        evaluate_mle, m31_to_qm31, qm31_eq,
    };
    use crate::channel::{
        channel_default, channel_mix_u64, channel_mix_felt, channel_mix_felts,
        channel_draw_felt252, channel_draw_qm31, channel_draw_qm31s, channel_mix_secure_field,
    };
    use crate::mle::verify_mle_opening;
    use crate::model_verifier::{verify_gkr_model_with_trace, WeightClaimData};
    // NOTE: verify_unified_stark + UnifiedStarkProof deserialization pulls in
    // stwo_verifier_core's FRI verifier which uses Felt252Dict (squashed_felt252_dict_entries).
    // This libfunc is not yet in Starknet's allowed list, blocking deployment.
    // STARK verification is done off-chain; on-chain we bind via Poseidon hash.
    // Uncomment when Starknet adds squashed_felt252_dict_entries to allowed libfuncs:
    //   use crate::ml_air::{MLClaim, UnifiedStarkProof, verify_unified_stark};
    //   use stwo_verifier_core::channel::Channel as StwoChannel;
    use crate::audit::{AuditRecord, AuditSubmitted, generate_audit_id, PackedDigest8};
    use crate::access_control::{AuditAccess, AccessGranted, AccessRevoked};
    use crate::view_key::{ViewKeyDelegation, ViewKeyDelegated, ViewKeyRevoked};
    use crate::vm31_merkle::PackedDigest;
    use starknet::storage::{
        StoragePointerReadAccess, StoragePointerWriteAccess, Map, StoragePathEntry,
    };
    use starknet::{ClassHash, ContractAddress, get_caller_address, get_block_timestamp, get_block_number};

    #[storage]
    struct Storage {
        /// Contract owner (can register models).
        owner: ContractAddress,
        /// VM31 binder role (can bind proof_hash -> VM31 public hash).
        vm31_binder: ContractAddress,
        /// model_id → Poseidon hash of model weight matrices.
        model_commitments: Map<felt252, felt252>,
        /// model_id → number of successful verifications.
        verification_counts: Map<felt252, u64>,
        /// proof_hash → verified (true/false).
        verified_proofs: Map<felt252, bool>,
        /// proof_hash → VM31 batch public-input hash.
        vm31_public_hash: Map<felt252, PackedDigest>,
        /// proof_hash → whether a VM31 public hash binding exists.
        vm31_public_hash_set: Map<felt252, bool>,
        /// model_id → number of GKR weight commitments.
        model_gkr_weight_count: Map<felt252, u32>,
        /// (model_id, idx) → Poseidon Merkle root of weight MLE.
        model_gkr_weights: Map<(felt252, u32), felt252>,
        /// model_id → Poseidon hash of circuit descriptor.
        model_circuit_hash: Map<felt252, felt252>,
        /// Pending upgrade class hash (zero = no pending upgrade).
        pending_upgrade: ClassHash,
        /// Timestamp when the upgrade was proposed.
        upgrade_proposed_at: u64,
        // ─── Audit Records ──────────────────────────────────────
        /// audit_id → AuditRecord.
        audit_records: Map<felt252, AuditRecord>,
        /// (model_id, index) → audit_id (append-only list per model).
        model_audit_ids: Map<(felt252, u32), felt252>,
        /// model_id → number of audits.
        model_audit_count: Map<felt252, u32>,
        /// Global audit nonce (for generating unique audit_ids).
        next_audit_nonce: u32,
        /// model_id → total proven inferences across all audits.
        total_proven_inferences: Map<felt252, u64>,
        // ─── Access Control ──────────────────────────────────────
        /// (audit_id, grantee) → AuditAccess record.
        audit_access: Map<(felt252, ContractAddress), AuditAccess>,
        /// (audit_id, grantee) → wrapped encryption key.
        audit_wrapped_keys: Map<(felt252, ContractAddress), felt252>,
        /// (audit_id, index) → grantee address (for enumeration).
        audit_access_list: Map<(felt252, u32), ContractAddress>,
        /// audit_id → total grantees ever (for enumeration indexing).
        audit_access_count: Map<felt252, u32>,
        /// audit_id → number of currently active grantees.
        audit_active_access_count: Map<felt252, u32>,
        /// audit_id → owner address (set on submit_audit).
        audit_owner: Map<felt252, ContractAddress>,
        // ─── View Key Delegation ─────────────────────────────────
        /// (owner, delegate) → ViewKeyDelegation.
        view_delegations: Map<(ContractAddress, ContractAddress), ViewKeyDelegation>,
        /// (owner, index) → delegate address.
        view_delegation_list: Map<(ContractAddress, u32), ContractAddress>,
        /// owner → number of delegations.
        view_delegation_count: Map<ContractAddress, u32>,
        // ─── GKR Chunked Session Storage ─────────────────────────
        /// session_id → session owner address.
        gkr_session_owner: Map<felt252, ContractAddress>,
        /// session_id → model_id.
        gkr_session_model_id: Map<felt252, felt252>,
        /// session_id → total felts expected.
        gkr_session_total_felts: Map<felt252, u32>,
        /// session_id → felts received so far.
        gkr_session_received_felts: Map<felt252, u32>,
        /// session_id → total number of chunks expected.
        gkr_session_num_chunks: Map<felt252, u32>,
        /// session_id → number of chunks received so far.
        gkr_session_chunks_received: Map<felt252, u32>,
        /// session_id → session status (0=none, 1=uploading, 2=sealed, 3=verified).
        gkr_session_status: Map<felt252, u8>,
        /// session_id → block number when session was created.
        gkr_session_created_at: Map<felt252, u64>,
        /// (session_id, flat_index) → felt252 data.
        gkr_session_data: Map<(felt252, u32), felt252>,
        /// session_id → circuit_depth parameter.
        gkr_session_circuit_depth: Map<felt252, u32>,
        /// session_id → num_layers parameter.
        gkr_session_num_layers: Map<felt252, u32>,
        /// session_id → weight_binding_mode parameter.
        gkr_session_weight_binding_mode: Map<felt252, u32>,
        /// session_id → packed QM31 format flag (true = 1 felt per QM31, false = 4 felts).
        gkr_session_packed: Map<felt252, bool>,
        /// Monotonic nonce for generating unique session IDs.
        next_gkr_session_nonce: u64,
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        ModelRegistered: ModelRegistered,
        ModelGkrRegistered: ModelGkrRegistered,
        ModelGkrVerified: ModelGkrVerified,
        GkrSessionOpened: GkrSessionOpened,
        GkrChunkUploaded: GkrChunkUploaded,
        GkrSessionSealed: GkrSessionSealed,
        GkrSessionVerified: GkrSessionVerified,
        GkrSessionExpired: GkrSessionExpired,
        VerificationFailed: VerificationFailed,
        UpgradeProposed: UpgradeProposed,
        UpgradeExecuted: UpgradeExecuted,
        UpgradeCancelled: UpgradeCancelled,
        AuditSubmitted: AuditSubmitted,
        AccessGranted: AccessGranted,
        AccessRevoked: AccessRevoked,
        ViewKeyDelegated: ViewKeyDelegated,
        ViewKeyRevoked: ViewKeyRevoked,
    }

    #[derive(Drop, starknet::Event)]
    struct ModelRegistered {
        #[key]
        model_id: felt252,
        weight_commitment: felt252,
        registrar: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct ModelGkrRegistered {
        #[key]
        model_id: felt252,
        num_weight_commitments: u32,
        circuit_hash: felt252,
        registrar: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct ModelGkrVerified {
        #[key]
        model_id: felt252,
        proof_hash: felt252,
        io_commitment: felt252,
        num_layers: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct VerificationFailed {
        #[key]
        model_id: felt252,
        reason: felt252,
    }

    #[derive(Drop, starknet::Event)]
    struct UpgradeProposed {
        new_class_hash: ClassHash,
        proposed_at: u64,
        proposer: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct UpgradeExecuted {
        new_class_hash: ClassHash,
        executed_at: u64,
        executor: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct UpgradeCancelled {
        cancelled_class_hash: ClassHash,
        cancelled_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrSessionOpened {
        #[key]
        session_id: felt252,
        model_id: felt252,
        owner: ContractAddress,
        total_felts: u32,
        num_chunks: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrChunkUploaded {
        #[key]
        session_id: felt252,
        chunk_idx: u32,
        data_len: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrSessionSealed {
        #[key]
        session_id: felt252,
        total_felts: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrSessionVerified {
        #[key]
        session_id: felt252,
        model_id: felt252,
        proof_hash: felt252,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrSessionExpired {
        #[key]
        session_id: felt252,
        expired_by: ContractAddress,
    }

    // GKR session constants
    const MAX_GKR_CHUNK_FELTS: u32 = 4000;
    const GKR_SESSION_TIMEOUT_BLOCKS: u64 = 10000;
    // Session status values
    const GKR_SESSION_STATUS_NONE: u8 = 0;
    const GKR_SESSION_STATUS_UPLOADING: u8 = 1;
    const GKR_SESSION_STATUS_SEALED: u8 = 2;
    const GKR_SESSION_STATUS_VERIFIED: u8 = 3;

    const WEIGHT_BINDING_MODE_SEQUENTIAL: u32 = 0;
    const WEIGHT_BINDING_MODE_BATCHED_SUBCHANNEL_V1: u32 = 1;
    const WEIGHT_BINDING_MODE_AGGREGATED_TRUSTLESS_V2: u32 = 2;
    const WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL: u32 = 3;
    const WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK: u32 = 4;
    const WEIGHT_BINDING_MODE2_DOMAIN_TAG: felt252 = 0x57424d32;
    const WEIGHT_BINDING_MODE2_SCHEMA_VERSION: felt252 = 1;
    const WEIGHT_BINDING_MODE3_DOMAIN_TAG: felt252 = 0x57424d33;
    const WEIGHT_BINDING_MODE3_SCHEMA_VERSION: felt252 = 1;
    /// Marker tag for mode 4 RLC-only binding (0x524C43 = "RLC").
    const WEIGHT_BINDING_RLC_MARKER: felt252 = 0x524C43;

    fn derive_weight_opening_subchannel(
        opening_seed: felt252,
        opening_index: u32,
        claim: @WeightClaimData,
    ) -> crate::channel::PoseidonChannel {
        let mut ch = channel_default();
        channel_mix_felt(ref ch, opening_seed);
        channel_mix_u64(ref ch, opening_index.into());
        channel_mix_felts(ref ch, claim.eval_point.span());
        channel_mix_felt(ref ch, pack_qm31_to_felt(*claim.expected_value));
        ch
    }

    fn compute_mode_binding_digest(
        commitments: Span<felt252>,
        claims: Span<WeightClaimData>,
        domain_tag: felt252,
        schema_version: felt252,
    ) -> felt252 {
        let mut hasher_inputs: Array<felt252> = array![
            domain_tag,
            schema_version,
            claims.len().into(),
            commitments.len().into(),
        ];

        let mut c_i: u32 = 0;
        loop {
            if c_i >= claims.len() {
                break;
            }
            let claim = claims.at(c_i);
            hasher_inputs.append(*commitments.at(c_i));
            hasher_inputs.append(claim.eval_point.len().into());

            let mut p_i: u32 = 0;
            loop {
                if p_i >= claim.eval_point.len() {
                    break;
                }
                hasher_inputs.append(pack_qm31_to_felt(*claim.eval_point.at(p_i)));
                p_i += 1;
            };

            hasher_inputs.append(pack_qm31_to_felt(*claim.expected_value));
            c_i += 1;
        };

        core::poseidon::poseidon_hash_span(hasher_inputs.span())
    }

    fn compute_mode2_binding_digest(
        commitments: Span<felt252>,
        claims: Span<WeightClaimData>,
    ) -> felt252 {
        compute_mode_binding_digest(
            commitments,
            claims,
            WEIGHT_BINDING_MODE2_DOMAIN_TAG,
            WEIGHT_BINDING_MODE2_SCHEMA_VERSION,
        )
    }

    fn compute_mode3_binding_digest(
        commitments: Span<felt252>,
        claims: Span<WeightClaimData>,
    ) -> felt252 {
        compute_mode_binding_digest(
            commitments,
            claims,
            WEIGHT_BINDING_MODE3_DOMAIN_TAG,
            WEIGHT_BINDING_MODE3_SCHEMA_VERSION,
        )
    }

    fn verify_model_gkr_core(
        ref self: ContractState,
        model_id: felt252,
        raw_io_data: Array<felt252>,
        circuit_depth: u32,
        num_layers: u32,
        matmul_dims: Array<u32>,
        dequantize_bits: Array<u64>,
        proof_data: Array<felt252>,
        weight_commitments: Array<felt252>,
        weight_binding_mode: u32,
        weight_binding_data: Span<felt252>,
        weight_opening_proofs: Array<MleOpeningProof>,
        packed: bool,
    ) -> bool {
        assert!(
            weight_binding_mode == WEIGHT_BINDING_MODE_SEQUENTIAL
                || weight_binding_mode == WEIGHT_BINDING_MODE_BATCHED_SUBCHANNEL_V1
                || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_TRUSTLESS_V2
                || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL
                || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK,
            "UNSUPPORTED_WEIGHT_BINDING_MODE",
        );

        // 0b. Input validation guards — catch malformed calldata early
        assert!(matmul_dims.len() % 3 == 0, "MATMUL_DIMS_NOT_TRIPLE");
        let mut d_i: u32 = 0;
        loop {
            if d_i >= matmul_dims.len() {
                break;
            }
            assert!(*matmul_dims.at(d_i) > 0, "MATMUL_DIM_ZERO");
            assert!(*matmul_dims.at(d_i + 1) > 0, "MATMUL_DIM_ZERO");
            assert!(*matmul_dims.at(d_i + 2) > 0, "MATMUL_DIM_ZERO");
            d_i += 3;
        };
        assert!(num_layers > 0, "ZERO_LAYERS");
        assert!(circuit_depth > 0, "ZERO_CIRCUIT_DEPTH");

        // 1. Validate model is registered for GKR
        let circuit_hash = self.model_circuit_hash.entry(model_id).read();
        assert!(circuit_hash != 0, "Model not registered for GKR");

        // 2. Recompute IO commitment from raw data — never trust caller
        assert!(raw_io_data.len() >= 6, "IO_DATA_TOO_SHORT");
        let io_commitment = core::poseidon::poseidon_hash_span(raw_io_data.span());

        // 3. Verify weight commitments match registered
        let registered_count = self.model_gkr_weight_count.entry(model_id).read();
        assert!(
            weight_commitments.len() == registered_count,
            "Weight commitment count mismatch",
        );
        let mut w_idx: u32 = 0;
        loop {
            if w_idx >= registered_count {
                break;
            }
            let registered = self.model_gkr_weights.entry((model_id, w_idx)).read();
            assert!(
                *weight_commitments.at(w_idx) == registered,
                "Weight commitment mismatch",
            );
            w_idx += 1;
        };

        // ================================================================
        // 4. Parse raw_io_data to extract input and output matrices
        //
        // Layout (matches serialize_raw_io in cairo_serde.rs):
        //   [in_rows, in_cols, in_len, in_data...,
        //    out_rows, out_cols, out_len, out_data...]
        // ================================================================
        let io_span = raw_io_data.span();
        let io_len: u32 = io_span.len();
        let mut io_off: u32 = 0;

        // Parse input header
        assert!(io_off + 2 < io_len, "IO_DATA_TRUNCATED_INPUT_HEADER");
        let input_rows_felt: u256 = (*io_span.at(io_off)).into();
        let input_rows: u64 = input_rows_felt.try_into().unwrap();
        io_off += 1;
        let input_cols_felt: u256 = (*io_span.at(io_off)).into();
        let input_cols: u64 = input_cols_felt.try_into().unwrap();
        io_off += 1;
        let input_len_felt: u256 = (*io_span.at(io_off)).into();
        let input_len: u32 = input_len_felt.try_into().unwrap();
        io_off += 1;

        // Extract raw input M31 values
        assert!(io_off <= io_len, "IO_DATA_OFFSET_OOB");
        assert!(input_len <= io_len - io_off, "IO_INPUT_LENGTH_MISMATCH");
        let mut raw_input: Array<u64> = array![];
        let mut i: u32 = 0;
        loop {
            if i >= input_len {
                break;
            }
            let v: u256 = (*io_span.at(io_off + i)).into();
            raw_input.append(v.try_into().unwrap());
            i += 1;
        };
        io_off += input_len;

        // Parse output header
        assert!(io_off + 2 < io_len, "IO_DATA_TRUNCATED_OUTPUT_HEADER");
        let output_rows_felt: u256 = (*io_span.at(io_off)).into();
        let output_rows: u64 = output_rows_felt.try_into().unwrap();
        io_off += 1;
        let output_cols_felt: u256 = (*io_span.at(io_off)).into();
        let output_cols: u64 = output_cols_felt.try_into().unwrap();
        io_off += 1;
        let output_len_felt: u256 = (*io_span.at(io_off)).into();
        let output_len: u32 = output_len_felt.try_into().unwrap();
        io_off += 1;

        // Extract raw output M31 values
        assert!(io_off <= io_len, "IO_DATA_OFFSET_OOB");
        assert!(output_len <= io_len - io_off, "IO_OUTPUT_LENGTH_MISMATCH");
        let mut raw_output: Array<u64> = array![];
        i = 0;
        loop {
            if i >= output_len {
                break;
            }
            let v: u256 = (*io_span.at(io_off + i)).into();
            raw_output.append(v.try_into().unwrap());
            i += 1;
        };
        io_off += output_len;
        assert!(io_off == io_len, "IO_DATA_LENGTH_MISMATCH");

        // ================================================================
        // 5. Build output MLE (pad to power-of-2 dimensions, row-major)
        //
        // Matches Rust: pad_matrix_pow2(output) → matrix_to_mle()
        // The MLE has next_pow2(rows) * next_pow2(cols) entries.
        // ================================================================
        let out_rows_u32: u32 = output_rows.try_into().unwrap();
        let out_cols_u32: u32 = output_cols.try_into().unwrap();
        let padded_out_rows = next_power_of_two(out_rows_u32);
        let padded_out_cols = next_power_of_two(out_cols_u32);
        let _padded_out_len = padded_out_rows * padded_out_cols;

        // Build padded MLE: row i, col j → index i*padded_out_cols + j
        let mut output_mle: Array<QM31> = array![];
        let mut row: u32 = 0;
        loop {
            if row >= padded_out_rows {
                break;
            }
            let mut col: u32 = 0;
            loop {
                if col >= padded_out_cols {
                    break;
                }
                if row < out_rows_u32 && col < out_cols_u32 {
                    let idx: u32 = row * out_cols_u32 + col;
                    output_mle.append(m31_to_qm31(*raw_output.at(idx)));
                } else {
                    output_mle.append(crate::field::qm31_zero());
                }
                col += 1;
            };
            row += 1;
        };

        // ================================================================
        // 6. Initialize Fiat-Shamir channel and construct output claim
        // ================================================================
        let mut ch = channel_default();
        channel_mix_u64(ref ch, circuit_depth.into());
        channel_mix_u64(ref ch, input_rows);
        channel_mix_u64(ref ch, input_cols);

        let log_out_rows = log2_ceil(padded_out_rows);
        let log_out_cols = log2_ceil(padded_out_cols);
        let log_out_total = log_out_rows + log_out_cols;

        // Draw r_out from channel — this IS the claim point (not discarded)
        let r_out = channel_draw_qm31s(ref ch, log_out_total);

        // OUTPUT CLAIM VERIFICATION: evaluate MLE(raw_output, r_out) on-chain
        let output_value = evaluate_mle(output_mle.span(), r_out.span());

        // Mix the computed output value (matches prover's mix_secure_field)
        channel_mix_secure_field(ref ch, output_value);

        // ================================================================
        // 7. Run GKR model walk
        // ================================================================
        let initial_claim = GKRClaim { point: r_out, value: output_value };

        let (final_claim, weight_claims, layer_tags, deferred_weight_commitments) =
            verify_gkr_model_with_trace(
                proof_data.span(),
                num_layers,
                matmul_dims.span(),
                dequantize_bits.span(),
                initial_claim,
                ref ch,
                packed,
            );

        // ================================================================
        // 7a. CIRCUIT BINDING: hash(circuit_depth || layer_tags) must match
        //     registered model circuit hash.
        // ================================================================
        let mut descriptor_felts: Array<felt252> = array![circuit_depth.into()];
        let mut t_i: u32 = 0;
        loop {
            if t_i >= layer_tags.len() {
                break;
            }
            let tag_felt: felt252 = (*layer_tags.at(t_i)).into();
            descriptor_felts.append(tag_felt);
            t_i += 1;
        };
        let observed_circuit_hash = core::poseidon::poseidon_hash_span(descriptor_felts.span());
        assert!(observed_circuit_hash == circuit_hash, "CIRCUIT_HASH_MISMATCH");

        // ================================================================
        // 7b. WEIGHT BINDING: verify opening proofs for both main and
        //     deferred matmul claims.
        // ================================================================
        let expected_weight_claims = registered_count + deferred_weight_commitments.len();
        assert!(
            weight_claims.len() == expected_weight_claims,
            "WEIGHT_CLAIM_COUNT_MISMATCH",
        );
        if weight_binding_mode != WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK {
            assert!(
                weight_opening_proofs.len() == expected_weight_claims,
                "WEIGHT_OPENING_COUNT_MISMATCH",
            );
        }
        if weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_TRUSTLESS_V2
            || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL
        {
            assert!(
                weight_binding_data.len() == 2,
                "WEIGHT_BINDING_DATA_LENGTH_MISMATCH",
            );
        } else if weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK {
            // Mode 4: either full aggregated proof or RLC-only marker (2 felts)
            assert!(weight_binding_data.len() >= 2, "AGGREGATED_BINDING_DATA_TOO_SHORT");
        } else {
            assert!(weight_binding_data.len() == 0, "UNEXPECTED_WEIGHT_BINDING_DATA_FOR_MODE");
        }

        let opening_seed = if (
            weight_binding_mode == WEIGHT_BINDING_MODE_BATCHED_SUBCHANNEL_V1
                || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_TRUSTLESS_V2
                || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL
        ) && expected_weight_claims > 0 {
            channel_draw_felt252(ref ch)
        } else {
            0
        };

        let mut resolved_weight_commitments: Array<felt252> = array![];
        let mut w_i: u32 = 0;
        loop {
            if w_i >= expected_weight_claims {
                break;
            }

            let commitment = if w_i < registered_count {
                self.model_gkr_weights.entry((model_id, w_i)).read()
            } else {
                let deferred_root = *deferred_weight_commitments.at(w_i - registered_count);
                assert!(deferred_root != 0, "DEFERRED_WEIGHT_COMMITMENT_ZERO");

                let mut found = false;
                let mut reg_i: u32 = 0;
                loop {
                    if reg_i >= registered_count {
                        break;
                    }
                    let reg_root = self.model_gkr_weights.entry((model_id, reg_i)).read();
                    if reg_root == deferred_root {
                        found = true;
                        break;
                    }
                    reg_i += 1;
                };
                assert!(found, "DEFERRED_WEIGHT_NOT_REGISTERED");
                deferred_root
            };
            resolved_weight_commitments.append(commitment);
            w_i += 1;
        };

        if weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_TRUSTLESS_V2
            || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL
        {
            let advertised_claim_count_u256: u256 = (*weight_binding_data.at(1)).into();
            let advertised_claim_count: u32 = advertised_claim_count_u256.try_into().unwrap();
            assert!(
                advertised_claim_count == expected_weight_claims,
                "WEIGHT_BINDING_CLAIM_COUNT_MISMATCH",
            );
            let expected_digest = if weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_TRUSTLESS_V2 {
                compute_mode2_binding_digest(resolved_weight_commitments.span(), weight_claims.span())
            } else {
                compute_mode3_binding_digest(resolved_weight_commitments.span(), weight_claims.span())
            };
            assert!(
                expected_digest == *weight_binding_data.at(0),
                "WEIGHT_BINDING_DIGEST_MISMATCH",
            );
        }

        if weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK {
            // Mode 4: either full mismatch sumcheck proof or RLC-only binding.
            if weight_binding_data.len() == 2
                && *weight_binding_data.at(0) == WEIGHT_BINDING_RLC_MARKER {
                // RLC-only binding: replay prover's Fiat-Shamir transcript.
                // Draw rho from channel, compute combined = sum(rho^i * expected_value_i),
                // mix combined into channel. Commitments already bound via Fiat-Shamir.
                let rho = channel_draw_qm31(ref ch);
                let mut rho_pow = crate::field::qm31_one();
                let mut combined = crate::field::qm31_zero();
                let mut claim_i: u32 = 0;
                loop {
                    if claim_i >= expected_weight_claims {
                        break;
                    }
                    let claim = weight_claims.at(claim_i);
                    combined = crate::field::qm31_add(
                        combined,
                        crate::field::qm31_mul(rho_pow, *claim.expected_value),
                    );
                    rho_pow = crate::field::qm31_mul(rho_pow, rho);
                    claim_i += 1;
                };
                channel_mix_secure_field(ref ch, combined);
            } else {
                // Full mismatch sumcheck proof.
                let mut agg_data = weight_binding_data;
                let agg_proof: crate::aggregated_binding::AggregatedWeightBindingProof =
                    Serde::deserialize(ref agg_data).expect('AGG_BINDING_DESER_FAIL');

                let valid = crate::aggregated_binding::verify_aggregated_binding(
                    @agg_proof,
                    weight_claims.span(),
                    resolved_weight_commitments.span(),
                    ref ch,
                );
                assert!(valid, "AGGREGATED_WEIGHT_BINDING_FAILED");
            }
        } else {
            // Modes 0-3: Per-weight MLE opening verification loop.
            w_i = 0;
            loop {
                if w_i >= expected_weight_claims {
                    break;
                }
                let commitment = *resolved_weight_commitments.at(w_i);
                let opening = weight_opening_proofs.at(w_i);
                let claim = weight_claims.at(w_i);

                assert!(
                    qm31_eq(*opening.final_value, *claim.expected_value),
                    "WEIGHT_OPENING_VALUE_MISMATCH",
                );

                let valid = if weight_binding_mode == WEIGHT_BINDING_MODE_BATCHED_SUBCHANNEL_V1
                    || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_TRUSTLESS_V2
                    || weight_binding_mode
                        == WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL
                {
                    let mut sub_ch = derive_weight_opening_subchannel(opening_seed, w_i, claim);
                    verify_mle_opening(
                        commitment,
                        opening,
                        claim.eval_point.span(),
                        ref sub_ch,
                    )
                } else {
                    verify_mle_opening(
                        commitment,
                        opening,
                        claim.eval_point.span(),
                        ref ch,
                    )
                };
                assert!(valid, "WEIGHT_MLE_OPENING_FAILED");
                w_i += 1;
            };
        }

        // ================================================================
        // 8. INPUT CLAIM VERIFICATION: evaluate MLE(raw_input, final_claim.point)
        //    and assert it matches final_claim.value
        // ================================================================
        let in_rows_u32: u32 = input_rows.try_into().unwrap();
        let in_cols_u32: u32 = input_cols.try_into().unwrap();
        let padded_in_rows = next_power_of_two(in_rows_u32);
        let padded_in_cols = next_power_of_two(in_cols_u32);

        // Build padded input MLE (row-major, same layout as output)
        let mut input_mle: Array<QM31> = array![];
        row = 0;
        loop {
            if row >= padded_in_rows {
                break;
            }
            let mut col: u32 = 0;
            loop {
                if col >= padded_in_cols {
                    break;
                }
                if row < in_rows_u32 && col < in_cols_u32 {
                    let idx: u32 = row * in_cols_u32 + col;
                    input_mle.append(m31_to_qm31(*raw_input.at(idx)));
                } else {
                    input_mle.append(crate::field::qm31_zero());
                }
                col += 1;
            };
            row += 1;
        };

        // Evaluate input MLE at the GKR walk's final point
        let input_value = evaluate_mle(input_mle.span(), final_claim.point.span());

        // THE CRITICAL CHECK: input MLE evaluation must match final claim
        assert!(crate::field::qm31_eq(input_value, final_claim.value), "INPUT_CLAIM_MISMATCH");

        // ================================================================
        // 9. Compute proof hash and record
        // ================================================================
        let proof_hash = core::poseidon::poseidon_hash_span(
            array![ch.digest, io_commitment, model_id, num_layers.into()].span(),
        );

        // Replay prevention
        assert!(
            !self.verified_proofs.entry(proof_hash).read(),
            "PROOF_ALREADY_VERIFIED",
        );

        // Record verification
        self.verified_proofs.entry(proof_hash).write(true);
        let count = self.verification_counts.entry(model_id).read();
        self.verification_counts.entry(model_id).write(count + 1);

        self.emit(ModelGkrVerified {
            model_id, proof_hash, io_commitment, num_layers,
        });

        true
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
        self.vm31_binder.write(owner);
    }

    #[abi(embed_v0)]
    impl SumcheckVerifierImpl of super::ISumcheckVerifier<ContractState> {
        fn register_model(
            ref self: ContractState, model_id: felt252, weight_commitment: felt252,
        ) {
            let existing = self.model_commitments.entry(model_id).read();
            assert!(existing == 0, "Model already registered");
            assert!(weight_commitment != 0, "Commitment cannot be zero");

            self.model_commitments.entry(model_id).write(weight_commitment);

            self
                .emit(
                    ModelRegistered {
                        model_id, weight_commitment, registrar: get_caller_address(),
                    },
                );
        }

        fn get_model_commitment(self: @ContractState, model_id: felt252) -> felt252 {
            self.model_commitments.entry(model_id).read()
        }

        fn get_verification_count(self: @ContractState, model_id: felt252) -> u64 {
            self.verification_counts.entry(model_id).read()
        }

        fn is_proof_verified(self: @ContractState, proof_hash: felt252) -> bool {
            self.verified_proofs.entry(proof_hash).read()
        }

        fn bind_vm31_public_hash(
            ref self: ContractState, proof_hash: felt252, vm31_public_hash: PackedDigest,
        ) {
            assert!(get_caller_address() == self.vm31_binder.read(), "Only vm31 binder");
            assert!(
                self.verified_proofs.entry(proof_hash).read(),
                "Proof hash not verified"
            );
            assert!(
                !self.vm31_public_hash_set.entry(proof_hash).read(),
                "VM31 public hash already bound"
            );
            self.vm31_public_hash.entry(proof_hash).write(vm31_public_hash);
            self.vm31_public_hash_set.entry(proof_hash).write(true);
        }

        fn get_vm31_public_hash(self: @ContractState, proof_hash: felt252) -> PackedDigest {
            assert!(
                self.vm31_public_hash_set.entry(proof_hash).read(),
                "VM31 public hash not bound"
            );
            self.vm31_public_hash.entry(proof_hash).read()
        }

        fn get_vm31_binder(self: @ContractState) -> ContractAddress {
            self.vm31_binder.read()
        }

        fn set_vm31_binder(ref self: ContractState, binder: ContractAddress) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");
            let binder_felt: felt252 = binder.into();
            assert!(binder_felt != 0, "VM31 binder cannot be zero");
            self.vm31_binder.write(binder);
        }

        fn get_owner(self: @ContractState) -> ContractAddress {
            self.owner.read()
        }

        fn propose_upgrade(ref self: ContractState, new_class_hash: ClassHash) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");
            assert!(new_class_hash.into() != 0_felt252, "Class hash cannot be zero");

            // Cannot propose while another upgrade is pending
            let existing: felt252 = self.pending_upgrade.read().into();
            assert!(existing == 0, "Upgrade already pending, cancel first");

            let now = get_block_timestamp();
            self.pending_upgrade.write(new_class_hash);
            self.upgrade_proposed_at.write(now);

            self.emit(UpgradeProposed {
                new_class_hash, proposed_at: now, proposer: get_caller_address(),
            });
        }

        fn execute_upgrade(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");

            let new_class_hash = self.pending_upgrade.read();
            assert!(new_class_hash.into() != 0_felt252, "No upgrade pending");

            let proposed_at = self.upgrade_proposed_at.read();
            let now = get_block_timestamp();
            assert!(now >= proposed_at + UPGRADE_DELAY, "Upgrade delay not elapsed");

            // Clear pending state before syscall
            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);

            let executed_at = now;
            self.emit(UpgradeExecuted {
                new_class_hash, executed_at, executor: get_caller_address(),
            });

            starknet::syscalls::replace_class_syscall(new_class_hash).unwrap();
        }

        fn cancel_upgrade(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");

            let pending: ClassHash = self.pending_upgrade.read();
            assert!(pending.into() != 0_felt252, "No upgrade pending");

            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);

            self.emit(UpgradeCancelled {
                cancelled_class_hash: pending, cancelled_by: get_caller_address(),
            });
        }

        fn get_pending_upgrade(self: @ContractState) -> (ClassHash, u64) {
            (self.pending_upgrade.read(), self.upgrade_proposed_at.read())
        }

        fn register_model_gkr(
            ref self: ContractState,
            model_id: felt252,
            weight_commitments: Array<felt252>,
            circuit_descriptor: Array<u32>,
        ) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");

            // Ensure not already registered for GKR
            let existing = self.model_circuit_hash.entry(model_id).read();
            assert!(existing == 0, "Model already registered for GKR");

            // Store weight commitments (one per MatMul layer)
            let num_weights: u32 = weight_commitments.len();
            self.model_gkr_weight_count.entry(model_id).write(num_weights);
            let mut i: u32 = 0;
            loop {
                if i >= num_weights {
                    break;
                }
                let root = *weight_commitments.at(i);
                assert!(root != 0, "Weight commitment cannot be zero");
                self.model_gkr_weights.entry((model_id, i)).write(root);
                i += 1;
            };

            // Store circuit descriptor hash
            let mut desc_felts: Array<felt252> = array![];
            let mut j: u32 = 0;
            loop {
                if j >= circuit_descriptor.len() {
                    break;
                }
                let v: felt252 = (*circuit_descriptor.at(j)).into();
                desc_felts.append(v);
                j += 1;
            };
            let circuit_hash = core::poseidon::poseidon_hash_span(desc_felts.span());
            self.model_circuit_hash.entry(model_id).write(circuit_hash);

            self.emit(ModelGkrRegistered {
                model_id, num_weight_commitments: num_weights, circuit_hash,
                registrar: get_caller_address(),
            });
        }

        fn verify_model_gkr(
            ref self: ContractState,
            model_id: felt252,
            raw_io_data: Array<felt252>,
            circuit_depth: u32,
            num_layers: u32,
            matmul_dims: Array<u32>,
            dequantize_bits: Array<u64>,
            proof_data: Array<felt252>,
            weight_commitments: Array<felt252>,
            weight_opening_proofs: Array<MleOpeningProof>,
        ) -> bool {
            let empty_binding_data: Array<felt252> = array![];
            verify_model_gkr_core(
                ref self,
                model_id,
                raw_io_data,
                circuit_depth,
                num_layers,
                matmul_dims,
                dequantize_bits,
                proof_data,
                weight_commitments,
                WEIGHT_BINDING_MODE_SEQUENTIAL,
                empty_binding_data.span(),
                weight_opening_proofs,
                false,
            )
        }

        fn verify_model_gkr_v2(
            ref self: ContractState,
            model_id: felt252,
            raw_io_data: Array<felt252>,
            circuit_depth: u32,
            num_layers: u32,
            matmul_dims: Array<u32>,
            dequantize_bits: Array<u64>,
            proof_data: Array<felt252>,
            weight_commitments: Array<felt252>,
            weight_binding_mode: u32,
            weight_opening_proofs: Array<MleOpeningProof>,
        ) -> bool {
            assert!(
                weight_binding_mode == WEIGHT_BINDING_MODE_SEQUENTIAL
                    || weight_binding_mode == WEIGHT_BINDING_MODE_BATCHED_SUBCHANNEL_V1,
                "UNSUPPORTED_WEIGHT_BINDING_MODE",
            );
            let empty_binding_data: Array<felt252> = array![];
            verify_model_gkr_core(
                ref self,
                model_id,
                raw_io_data,
                circuit_depth,
                num_layers,
                matmul_dims,
                dequantize_bits,
                proof_data,
                weight_commitments,
                weight_binding_mode,
                empty_binding_data.span(),
                weight_opening_proofs,
                false,
            )
        }

        fn verify_model_gkr_v3(
            ref self: ContractState,
            model_id: felt252,
            raw_io_data: Array<felt252>,
            circuit_depth: u32,
            num_layers: u32,
            matmul_dims: Array<u32>,
            dequantize_bits: Array<u64>,
            proof_data: Array<felt252>,
            weight_commitments: Array<felt252>,
            weight_binding_mode: u32,
            weight_binding_data: Array<felt252>,
            weight_opening_proofs: Array<MleOpeningProof>,
        ) -> bool {
            assert!(
                weight_binding_mode == WEIGHT_BINDING_MODE_SEQUENTIAL
                    || weight_binding_mode == WEIGHT_BINDING_MODE_BATCHED_SUBCHANNEL_V1
                    || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_TRUSTLESS_V2,
                "UNSUPPORTED_WEIGHT_BINDING_MODE",
            );

            verify_model_gkr_core(
                ref self,
                model_id,
                raw_io_data,
                circuit_depth,
                num_layers,
                matmul_dims,
                dequantize_bits,
                proof_data,
                weight_commitments,
                weight_binding_mode,
                weight_binding_data.span(),
                weight_opening_proofs,
                false,
            )
        }

        fn verify_model_gkr_v4(
            ref self: ContractState,
            model_id: felt252,
            raw_io_data: Array<felt252>,
            circuit_depth: u32,
            num_layers: u32,
            matmul_dims: Array<u32>,
            dequantize_bits: Array<u64>,
            proof_data: Array<felt252>,
            weight_commitments: Array<felt252>,
            weight_binding_mode: u32,
            weight_binding_data: Array<felt252>,
            weight_opening_proofs: Array<MleOpeningProof>,
        ) -> bool {
            assert!(
                weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL
                    || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK,
                "UNSUPPORTED_WEIGHT_BINDING_MODE",
            );

            verify_model_gkr_core(
                ref self,
                model_id,
                raw_io_data,
                circuit_depth,
                num_layers,
                matmul_dims,
                dequantize_bits,
                proof_data,
                weight_commitments,
                weight_binding_mode,
                weight_binding_data.span(),
                weight_opening_proofs,
                false,
            )
        }

        fn verify_model_gkr_v4_packed(
            ref self: ContractState,
            model_id: felt252,
            raw_io_data: Array<felt252>,
            circuit_depth: u32,
            num_layers: u32,
            matmul_dims: Array<u32>,
            dequantize_bits: Array<u64>,
            proof_data: Array<felt252>,
            weight_commitments: Array<felt252>,
            weight_binding_mode: u32,
            weight_binding_data: Array<felt252>,
            weight_opening_proofs: Array<MleOpeningProof>,
        ) -> bool {
            assert!(
                weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL
                    || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK,
                "UNSUPPORTED_WEIGHT_BINDING_MODE",
            );

            verify_model_gkr_core(
                ref self,
                model_id,
                raw_io_data,
                circuit_depth,
                num_layers,
                matmul_dims,
                dequantize_bits,
                proof_data,
                weight_commitments,
                weight_binding_mode,
                weight_binding_data.span(),
                weight_opening_proofs,
                true,
            )
        }

        fn get_model_circuit_hash(self: @ContractState, model_id: felt252) -> felt252 {
            self.model_circuit_hash.entry(model_id).read()
        }

        fn get_model_gkr_weight_count(self: @ContractState, model_id: felt252) -> u32 {
            self.model_gkr_weight_count.entry(model_id).read()
        }

        // ─── Chunked GKR Session Protocol ────────────────────────────────

        fn open_gkr_session(
            ref self: ContractState,
            model_id: felt252,
            total_felts: u32,
            circuit_depth: u32,
            num_layers: u32,
            weight_binding_mode: u32,
            packed: bool,
        ) -> felt252 {
            assert!(total_felts > 0, "GKR_SESSION_ZERO_FELTS");
            assert!(circuit_depth > 0, "GKR_SESSION_ZERO_CIRCUIT_DEPTH");
            assert!(num_layers > 0, "GKR_SESSION_ZERO_NUM_LAYERS");
            assert!(
                weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL
                    || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK,
                "GKR_SESSION_UNSUPPORTED_BINDING_MODE",
            );

            // Note: model registration is NOT required for the session path,
            // matching verify_model_gkr_v4 which also skips it.

            // Compute number of chunks: ceil(total_felts / MAX_GKR_CHUNK_FELTS).
            let num_chunks = (total_felts + MAX_GKR_CHUNK_FELTS - 1) / MAX_GKR_CHUNK_FELTS;

            // Generate session_id via Poseidon(nonce, caller, block).
            let nonce = self.next_gkr_session_nonce.read();
            let caller = get_caller_address();
            let block = get_block_number();
            let session_id = core::poseidon::poseidon_hash_span(
                array![nonce.into(), caller.into(), block.into(), model_id].span(),
            );
            self.next_gkr_session_nonce.write(nonce + 1);

            // Write session metadata.
            self.gkr_session_owner.entry(session_id).write(caller);
            self.gkr_session_model_id.entry(session_id).write(model_id);
            self.gkr_session_total_felts.entry(session_id).write(total_felts);
            self.gkr_session_received_felts.entry(session_id).write(0);
            self.gkr_session_num_chunks.entry(session_id).write(num_chunks);
            self.gkr_session_chunks_received.entry(session_id).write(0);
            self.gkr_session_status.entry(session_id).write(GKR_SESSION_STATUS_UPLOADING);
            self.gkr_session_created_at.entry(session_id).write(block);
            self.gkr_session_circuit_depth.entry(session_id).write(circuit_depth);
            self.gkr_session_num_layers.entry(session_id).write(num_layers);
            self.gkr_session_weight_binding_mode.entry(session_id).write(weight_binding_mode);
            self.gkr_session_packed.entry(session_id).write(packed);

            self.emit(GkrSessionOpened {
                session_id, model_id, owner: caller, total_felts, num_chunks,
            });

            session_id
        }

        fn upload_gkr_chunk(
            ref self: ContractState,
            session_id: felt252,
            chunk_idx: u32,
            data: Array<felt252>,
        ) {
            // Session must be in uploading state.
            let status = self.gkr_session_status.entry(session_id).read();
            assert!(status == GKR_SESSION_STATUS_UPLOADING, "GKR_SESSION_NOT_UPLOADING");

            // Only the session owner can upload.
            let caller = get_caller_address();
            let owner = self.gkr_session_owner.entry(session_id).read();
            assert!(caller == owner, "GKR_SESSION_NOT_OWNER");

            // Check timeout.
            let created_at = self.gkr_session_created_at.entry(session_id).read();
            let current_block = get_block_number();
            assert!(
                current_block <= created_at + GKR_SESSION_TIMEOUT_BLOCKS,
                "GKR_SESSION_EXPIRED",
            );

            // Chunks must arrive in order.
            let chunks_received = self.gkr_session_chunks_received.entry(session_id).read();
            assert!(chunk_idx == chunks_received, "GKR_SESSION_CHUNK_OUT_OF_ORDER");

            // Chunk must not exceed limit.
            let data_len: u32 = data.len();
            assert!(data_len > 0, "GKR_SESSION_EMPTY_CHUNK");
            assert!(data_len <= MAX_GKR_CHUNK_FELTS, "GKR_SESSION_CHUNK_TOO_LARGE");

            // Must not exceed total_felts.
            let received = self.gkr_session_received_felts.entry(session_id).read();
            let total_felts = self.gkr_session_total_felts.entry(session_id).read();
            assert!(received + data_len <= total_felts, "GKR_SESSION_FELTS_OVERFLOW");

            // Write data to flat storage.
            let base_offset = received;
            let data_span = data.span();
            let mut i: u32 = 0;
            loop {
                if i >= data_len {
                    break;
                }
                self.gkr_session_data.entry((session_id, base_offset + i)).write(*data_span.at(i));
                i += 1;
            };

            // Update counters.
            self.gkr_session_received_felts.entry(session_id).write(received + data_len);
            self.gkr_session_chunks_received.entry(session_id).write(chunks_received + 1);

            self.emit(GkrChunkUploaded { session_id, chunk_idx, data_len });
        }

        fn seal_gkr_session(ref self: ContractState, session_id: felt252) {
            let status = self.gkr_session_status.entry(session_id).read();
            assert!(status == GKR_SESSION_STATUS_UPLOADING, "GKR_SESSION_NOT_UPLOADING");

            let caller = get_caller_address();
            let owner = self.gkr_session_owner.entry(session_id).read();
            assert!(caller == owner, "GKR_SESSION_NOT_OWNER");

            // Check timeout.
            let created_at = self.gkr_session_created_at.entry(session_id).read();
            let current_block = get_block_number();
            assert!(
                current_block <= created_at + GKR_SESSION_TIMEOUT_BLOCKS,
                "GKR_SESSION_EXPIRED",
            );

            // All felts must be received.
            let received = self.gkr_session_received_felts.entry(session_id).read();
            let total_felts = self.gkr_session_total_felts.entry(session_id).read();
            assert!(received == total_felts, "GKR_SESSION_INCOMPLETE");

            self.gkr_session_status.entry(session_id).write(GKR_SESSION_STATUS_SEALED);

            self.emit(GkrSessionSealed { session_id, total_felts });
        }

        fn verify_gkr_from_session(ref self: ContractState, session_id: felt252) -> bool {
            let status = self.gkr_session_status.entry(session_id).read();
            assert!(status == GKR_SESSION_STATUS_SEALED, "GKR_SESSION_NOT_SEALED");

            // Check timeout.
            let created_at = self.gkr_session_created_at.entry(session_id).read();
            let current_block = get_block_number();
            assert!(
                current_block <= created_at + GKR_SESSION_TIMEOUT_BLOCKS,
                "GKR_SESSION_EXPIRED",
            );

            let model_id = self.gkr_session_model_id.entry(session_id).read();
            let total_felts = self.gkr_session_total_felts.entry(session_id).read();
            let circuit_depth = self.gkr_session_circuit_depth.entry(session_id).read();
            let num_layers = self.gkr_session_num_layers.entry(session_id).read();
            let weight_binding_mode = self.gkr_session_weight_binding_mode.entry(session_id).read();
            let packed = self.gkr_session_packed.entry(session_id).read();

            // ── Read all data from flat storage into memory ──
            // Layout: [raw_io_data_len, raw_io_data...,
            //          matmul_dims_len, matmul_dims...,
            //          dequantize_bits_len, dequantize_bits...,
            //          proof_data_len, proof_data...,
            //          weight_commitments_len, weight_commitments...,
            //          weight_binding_data_len, weight_binding_data...,
            //          weight_opening_proofs_len, weight_opening_proofs_flat...]
            let mut flat: Array<felt252> = array![];
            let mut idx: u32 = 0;
            loop {
                if idx >= total_felts {
                    break;
                }
                flat.append(self.gkr_session_data.entry((session_id, idx)).read());
                idx += 1;
            };
            let flat_span = flat.span();

            // ── Parse sections from flat data ──
            let mut off: u32 = 0;

            // 1. raw_io_data
            assert!(off < total_felts, "GKR_SESSION_DATA_TRUNCATED");
            let raw_io_len_u256: u256 = (*flat_span.at(off)).into();
            let raw_io_len: u32 = raw_io_len_u256.try_into().unwrap();
            off += 1;
            let mut raw_io_data: Array<felt252> = array![];
            let mut i: u32 = 0;
            loop {
                if i >= raw_io_len {
                    break;
                }
                raw_io_data.append(*flat_span.at(off + i));
                i += 1;
            };
            off += raw_io_len;

            // 2. matmul_dims
            assert!(off < total_felts, "GKR_SESSION_DATA_TRUNCATED");
            let matmul_dims_len_u256: u256 = (*flat_span.at(off)).into();
            let matmul_dims_len: u32 = matmul_dims_len_u256.try_into().unwrap();
            off += 1;
            let mut matmul_dims: Array<u32> = array![];
            i = 0;
            loop {
                if i >= matmul_dims_len {
                    break;
                }
                let v_u256: u256 = (*flat_span.at(off + i)).into();
                matmul_dims.append(v_u256.try_into().unwrap());
                i += 1;
            };
            off += matmul_dims_len;

            // 3. dequantize_bits
            assert!(off < total_felts, "GKR_SESSION_DATA_TRUNCATED");
            let deq_bits_len_u256: u256 = (*flat_span.at(off)).into();
            let deq_bits_len: u32 = deq_bits_len_u256.try_into().unwrap();
            off += 1;
            let mut dequantize_bits: Array<u64> = array![];
            i = 0;
            loop {
                if i >= deq_bits_len {
                    break;
                }
                let v_u256: u256 = (*flat_span.at(off + i)).into();
                dequantize_bits.append(v_u256.try_into().unwrap());
                i += 1;
            };
            off += deq_bits_len;

            // 4. proof_data
            assert!(off < total_felts, "GKR_SESSION_DATA_TRUNCATED");
            let proof_data_len_u256: u256 = (*flat_span.at(off)).into();
            let proof_data_len: u32 = proof_data_len_u256.try_into().unwrap();
            off += 1;
            let mut proof_data: Array<felt252> = array![];
            i = 0;
            loop {
                if i >= proof_data_len {
                    break;
                }
                proof_data.append(*flat_span.at(off + i));
                i += 1;
            };
            off += proof_data_len;

            // 5. weight_commitments
            assert!(off < total_felts, "GKR_SESSION_DATA_TRUNCATED");
            let wc_len_u256: u256 = (*flat_span.at(off)).into();
            let wc_len: u32 = wc_len_u256.try_into().unwrap();
            off += 1;
            let mut weight_commitments: Array<felt252> = array![];
            i = 0;
            loop {
                if i >= wc_len {
                    break;
                }
                weight_commitments.append(*flat_span.at(off + i));
                i += 1;
            };
            off += wc_len;

            // 6. weight_binding_data
            assert!(off < total_felts, "GKR_SESSION_DATA_TRUNCATED");
            let wbd_len_u256: u256 = (*flat_span.at(off)).into();
            let wbd_len: u32 = wbd_len_u256.try_into().unwrap();
            off += 1;
            let mut weight_binding_data: Array<felt252> = array![];
            i = 0;
            loop {
                if i >= wbd_len {
                    break;
                }
                weight_binding_data.append(*flat_span.at(off + i));
                i += 1;
            };
            off += wbd_len;

            // 7. weight_opening_proofs (remaining felts are Serde-serialized Array<MleOpeningProof>)
            assert!(off <= total_felts, "GKR_SESSION_DATA_TRUNCATED");
            let wop_len = total_felts - off;
            let mut wop_span = flat_span.slice(off, wop_len);
            let weight_opening_proofs: Array<MleOpeningProof> =
                Serde::deserialize(ref wop_span).expect('GKR_SESSION_WOP_DESER_FAIL');

            // ── Delegate to core verifier ──
            let result = verify_model_gkr_core(
                ref self,
                model_id,
                raw_io_data,
                circuit_depth,
                num_layers,
                matmul_dims,
                dequantize_bits,
                proof_data,
                weight_commitments,
                weight_binding_mode,
                weight_binding_data.span(),
                weight_opening_proofs,
                packed,
            );

            // Mark session as verified.
            self.gkr_session_status.entry(session_id).write(GKR_SESSION_STATUS_VERIFIED);

            // proof_hash was already emitted by verify_model_gkr_core via ModelGkrVerified.
            // Session event uses a session-derived hash for correlation.
            let session_proof_hash = core::poseidon::poseidon_hash_span(
                array![session_id, model_id].span(),
            );
            self.emit(GkrSessionVerified { session_id, model_id, proof_hash: session_proof_hash });

            result
        }

        fn expire_gkr_session(ref self: ContractState, session_id: felt252) {
            let status = self.gkr_session_status.entry(session_id).read();
            assert!(
                status == GKR_SESSION_STATUS_UPLOADING || status == GKR_SESSION_STATUS_SEALED,
                "GKR_SESSION_CANNOT_EXPIRE",
            );

            let created_at = self.gkr_session_created_at.entry(session_id).read();
            let current_block = get_block_number();
            assert!(
                current_block > created_at + GKR_SESSION_TIMEOUT_BLOCKS,
                "GKR_SESSION_NOT_EXPIRED",
            );

            // Reset metadata (data storage is left — too expensive to clean).
            self.gkr_session_status.entry(session_id).write(GKR_SESSION_STATUS_NONE);

            self.emit(GkrSessionExpired { session_id, expired_by: get_caller_address() });
        }

        fn get_gkr_session_status(self: @ContractState, session_id: felt252) -> u8 {
            self.gkr_session_status.entry(session_id).read()
        }
    }

    // ─── Audit Verifier Implementation ──────────────────────────────────────

    #[abi(embed_v0)]
    impl AuditVerifierImpl of crate::audit::IAuditVerifier<ContractState> {
        fn submit_audit(
            ref self: ContractState,
            model_id: felt252,
            report_hash_lo: felt252,
            report_hash_hi: felt252,
            merkle_root_lo: felt252,
            merkle_root_hi: felt252,
            weight_commitment: felt252,
            time_start: u64,
            time_end: u64,
            inference_count: u32,
            tee_attestation_hash: felt252,
            privacy_tier: u8,
        ) -> felt252 {
            // 1. Verify weight commitment matches registered model
            let registered_weight = self.model_commitments.entry(model_id).read();
            assert!(registered_weight != 0, "Model not registered");
            assert!(weight_commitment == registered_weight, "Weight commitment mismatch");

            // 2. Validate time window
            assert!(time_end > time_start, "Invalid time window");
            assert!(inference_count > 0, "Empty audit");

            // 3. Generate audit ID
            let nonce = self.next_audit_nonce.read();
            let submitter = get_caller_address();
            let audit_id = generate_audit_id(nonce, model_id, submitter, time_start);
            self.next_audit_nonce.write(nonce + 1);

            // 4. Store audit record with PackedDigest8 fields
            let report_hash = PackedDigest8 { lo: report_hash_lo, hi: report_hash_hi };
            let merkle_root = PackedDigest8 { lo: merkle_root_lo, hi: merkle_root_hi };

            let record = AuditRecord {
                model_id,
                audit_report_hash: report_hash,
                inference_log_merkle_root: merkle_root,
                weight_commitment,
                time_start,
                time_end,
                inference_count,
                proof_verified: false,
                submitter,
                submitted_at_block: get_block_number(),
                tee_attestation_hash,
                privacy_tier,
            };
            self.audit_records.entry(audit_id).write(record);

            // 4b. Set audit owner for access control
            self.audit_owner.entry(audit_id).write(submitter);

            // 5. Append to model's audit list
            let count = self.model_audit_count.entry(model_id).read();
            self.model_audit_ids.entry((model_id, count)).write(audit_id);
            self.model_audit_count.entry(model_id).write(count + 1);

            // 6. Update total proven inferences
            let total = self.total_proven_inferences.entry(model_id).read();
            self.total_proven_inferences.entry(model_id).write(total + inference_count.into());

            // 7. Emit event
            self.emit(AuditSubmitted {
                audit_id,
                model_id,
                submitter,
                report_hash_lo,
                report_hash_hi,
                merkle_root_lo,
                merkle_root_hi,
                time_start,
                time_end,
                inference_count,
                proof_verified: false,
                privacy_tier,
            });

            audit_id
        }

        fn get_audit(self: @ContractState, audit_id: felt252) -> AuditRecord {
            self.audit_records.entry(audit_id).read()
        }

        fn get_model_audits(self: @ContractState, model_id: felt252) -> Array<felt252> {
            let count = self.model_audit_count.entry(model_id).read();
            let mut ids: Array<felt252> = array![];
            let mut i: u32 = 0;
            loop {
                if i >= count {
                    break;
                }
                ids.append(self.model_audit_ids.entry((model_id, i)).read());
                i += 1;
            };
            ids
        }

        fn get_audit_count(self: @ContractState, model_id: felt252) -> u32 {
            self.model_audit_count.entry(model_id).read()
        }

        fn get_latest_audit(self: @ContractState, model_id: felt252) -> AuditRecord {
            let count = self.model_audit_count.entry(model_id).read();
            assert!(count > 0, "No audits for model");
            let audit_id = self.model_audit_ids.entry((model_id, count - 1)).read();
            self.audit_records.entry(audit_id).read()
        }

        fn is_audited_in_range(
            self: @ContractState,
            model_id: felt252,
            since: u64,
            until: u64,
        ) -> bool {
            let count = self.model_audit_count.entry(model_id).read();
            let mut i = count;
            loop {
                if i == 0 {
                    break false;
                }
                i -= 1;
                let audit_id = self.model_audit_ids.entry((model_id, i)).read();
                let record = self.audit_records.entry(audit_id).read();
                if record.time_start >= since && record.time_end <= until {
                    break true;
                }
            }
        }

        fn get_total_proven_inferences(self: @ContractState, model_id: felt252) -> u64 {
            self.total_proven_inferences.entry(model_id).read()
        }
    }

    // ─── Access Control Implementation ──────────────────────────────────────

    #[abi(embed_v0)]
    impl AccessControlImpl of crate::access_control::IAuditAccessControl<ContractState> {
        fn grant_audit_access(
            ref self: ContractState,
            audit_id: felt252,
            grantee: ContractAddress,
            wrapped_key: felt252,
            role: u8,
        ) {
            // Only the audit owner can grant access.
            let caller = get_caller_address();
            let owner = self.audit_owner.entry(audit_id).read();
            assert!(owner == caller, "Only audit owner can grant access");

            // Check grantee doesn't already have active access.
            let existing = self.audit_access.entry((audit_id, grantee)).read();
            assert!(!existing.is_active, "Access already granted");

            // Store access record.
            let access = AuditAccess {
                address: grantee,
                role,
                granted_at_block: get_block_number(),
                is_active: true,
            };
            self.audit_access.entry((audit_id, grantee)).write(access);
            self.audit_wrapped_keys.entry((audit_id, grantee)).write(wrapped_key);

            // Append to enumeration list.
            let count = self.audit_access_count.entry(audit_id).read();
            self.audit_access_list.entry((audit_id, count)).write(grantee);
            self.audit_access_count.entry(audit_id).write(count + 1);

            // Increment active counter.
            let active = self.audit_active_access_count.entry(audit_id).read();
            self.audit_active_access_count.entry(audit_id).write(active + 1);

            self.emit(AccessGranted {
                audit_id,
                grantee,
                role,
                granted_by: caller,
            });
        }

        fn revoke_audit_access(
            ref self: ContractState,
            audit_id: felt252,
            revokee: ContractAddress,
        ) {
            let caller = get_caller_address();
            let owner = self.audit_owner.entry(audit_id).read();
            assert!(owner == caller, "Only audit owner can revoke access");

            let mut access = self.audit_access.entry((audit_id, revokee)).read();
            assert!(access.is_active, "Access not active");

            // Deactivate and zero wrapped key.
            access.is_active = false;
            self.audit_access.entry((audit_id, revokee)).write(access);
            self.audit_wrapped_keys.entry((audit_id, revokee)).write(0);

            // Decrement active counter.
            let active = self.audit_active_access_count.entry(audit_id).read();
            self.audit_active_access_count.entry(audit_id).write(active - 1);

            self.emit(AccessRevoked {
                audit_id,
                revokee,
                revoked_by: caller,
            });
        }

        fn grant_audit_access_batch(
            ref self: ContractState,
            audit_id: felt252,
            grantees: Span<ContractAddress>,
            wrapped_keys: Span<felt252>,
            roles: Span<u8>,
        ) {
            assert!(grantees.len() == wrapped_keys.len(), "Length mismatch");
            assert!(grantees.len() == roles.len(), "Length mismatch");

            let caller = get_caller_address();
            let owner = self.audit_owner.entry(audit_id).read();
            assert!(owner == caller, "Only audit owner can grant access");

            let mut i: u32 = 0;
            loop {
                if i >= grantees.len() {
                    break;
                }
                let grantee = *grantees.at(i);
                let wrapped_key = *wrapped_keys.at(i);
                let role = *roles.at(i);

                let existing = self.audit_access.entry((audit_id, grantee)).read();
                assert!(!existing.is_active, "Access already granted");

                let access = AuditAccess {
                    address: grantee,
                    role,
                    granted_at_block: get_block_number(),
                    is_active: true,
                };
                self.audit_access.entry((audit_id, grantee)).write(access);
                self.audit_wrapped_keys.entry((audit_id, grantee)).write(wrapped_key);

                let count = self.audit_access_count.entry(audit_id).read();
                self.audit_access_list.entry((audit_id, count)).write(grantee);
                self.audit_access_count.entry(audit_id).write(count + 1);

                // Increment active counter.
                let active = self.audit_active_access_count.entry(audit_id).read();
                self.audit_active_access_count.entry(audit_id).write(active + 1);

                self.emit(AccessGranted {
                    audit_id,
                    grantee,
                    role,
                    granted_by: caller,
                });

                i += 1;
            };
        }

        fn has_audit_access(
            self: @ContractState,
            audit_id: felt252,
            address: ContractAddress,
        ) -> bool {
            let access = self.audit_access.entry((audit_id, address)).read();
            access.is_active
        }

        fn get_wrapped_key(
            self: @ContractState,
            audit_id: felt252,
            grantee: ContractAddress,
        ) -> felt252 {
            let access = self.audit_access.entry((audit_id, grantee)).read();
            assert!(access.is_active, "No active access");
            self.audit_wrapped_keys.entry((audit_id, grantee)).read()
        }

        fn get_audit_owner(
            self: @ContractState,
            audit_id: felt252,
        ) -> ContractAddress {
            self.audit_owner.entry(audit_id).read()
        }

        fn get_access_count(
            self: @ContractState,
            audit_id: felt252,
        ) -> u32 {
            self.audit_active_access_count.entry(audit_id).read()
        }
    }

    // ─── View Key Delegation Implementation ─────────────────────────────────

    #[abi(embed_v0)]
    impl ViewKeyDelegationImpl of crate::view_key::IViewKeyDelegation<ContractState> {
        fn delegate_view_key(
            ref self: ContractState,
            delegate: ContractAddress,
            encrypted_view_key: felt252,
            valid_until: u64,
        ) {
            let owner = get_caller_address();

            // Check not already delegated.
            let existing = self.view_delegations.entry((owner, delegate)).read();
            assert!(!existing.is_active, "View key already delegated");

            let delegation = ViewKeyDelegation {
                owner,
                delegate,
                encrypted_view_key,
                valid_from: get_block_number(),
                valid_until,
                is_active: true,
            };
            self.view_delegations.entry((owner, delegate)).write(delegation);

            // Append to enumeration list.
            let count = self.view_delegation_count.entry(owner).read();
            self.view_delegation_list.entry((owner, count)).write(delegate);
            self.view_delegation_count.entry(owner).write(count + 1);

            self.emit(ViewKeyDelegated { owner, delegate, valid_until });
        }

        fn revoke_view_key(
            ref self: ContractState,
            delegate: ContractAddress,
        ) {
            let owner = get_caller_address();

            let mut delegation = self.view_delegations.entry((owner, delegate)).read();
            assert!(delegation.is_active, "View key not active");

            delegation.is_active = false;
            delegation.encrypted_view_key = 0;
            self.view_delegations.entry((owner, delegate)).write(delegation);

            self.emit(ViewKeyRevoked { owner, delegate });
        }

        fn has_view_key(
            self: @ContractState,
            owner: ContractAddress,
            delegate: ContractAddress,
        ) -> bool {
            let delegation = self.view_delegations.entry((owner, delegate)).read();
            if !delegation.is_active {
                return false;
            }
            // Check expiry (valid_until == 0 means forever).
            if delegation.valid_until != 0 {
                let current_block = get_block_number();
                if current_block > delegation.valid_until {
                    return false;
                }
            }
            true
        }

        fn get_view_key(
            self: @ContractState,
            owner: ContractAddress,
            delegate: ContractAddress,
        ) -> felt252 {
            let delegation = self.view_delegations.entry((owner, delegate)).read();
            assert!(delegation.is_active, "No active view key");
            // Check expiry (valid_until == 0 means forever).
            if delegation.valid_until != 0 {
                let current_block = get_block_number();
                assert!(current_block <= delegation.valid_until, "View key expired");
            }
            delegation.encrypted_view_key
        }

        fn get_delegation_count(
            self: @ContractState,
            owner: ContractAddress,
        ) -> u32 {
            self.view_delegation_count.entry(owner).read()
        }
    }
}
