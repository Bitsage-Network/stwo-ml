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
use crate::vm31_merkle::PackedDigest;
// QM31 imported only by v4_packed_io (inside module scope)
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

    /// Re-register model circuit hash for streaming (v25) verification.
    /// Uses incremental Poseidon: poseidon(circuit_depth, poseidon(...poseidon(0, tag_0)..., tag_N)).
    /// Only updates the circuit_hash — weight commitments remain from register_model_gkr.
    fn register_model_gkr_streaming_circuit(
        ref self: TContractState,
        model_id: felt252,
        circuit_depth: u32,
        layer_tags: Array<u32>,
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

    /// On-chain GKR verification with IO-packed data.
    ///
    /// `packed_raw_io` contains 8 M31 values per felt252 (248 bits).
    /// `original_io_len` is the unpacked count. This reduces calldata from
    /// ~10K to ~1.3K felts, fitting within the 5000 felt TX limit.
    fn verify_model_gkr_v4_packed_io(
        ref self: TContractState,
        model_id: felt252,
        original_io_len: u32,
        packed_raw_io: Array<felt252>,
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

    /// On-chain GKR verification with double-packed proof data.
    ///
    /// Same as verify_model_gkr_v4_packed_io but proof_data uses double-packed
    /// QM31 pairs: degree-2 round polys (c0, c2) fit in a single felt252 (248 bits).
    /// This halves the proof_data section, enabling single-TX for Qwen3-14B.
    fn verify_model_gkr_v4_packed_io_dp(
        ref self: TContractState,
        model_id: felt252,
        original_io_len: u32,
        packed_raw_io: Array<felt252>,
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

    /// View-only verification: identical to verify_model_gkr_v4_packed_io but
    /// skips proof recording (no storage writes, no counters, no events).
    /// Use when you only need the boolean result, not on-chain proof tracking.
    /// Saves ~12.5K steps + ~40K gas per call.
    fn verify_model_gkr_v4_packed_io_view(
        self: @TContractState,
        model_id: felt252,
        original_io_len: u32,
        packed_raw_io: Array<felt252>,
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

    // ─── Chunked GKR Session Entrypoints ─────────────────────────────
    // For proofs exceeding Starknet's per-TX calldata limit (~5K felts).
    // 4-step protocol: open → upload chunks → seal → verify.

    /// Open a new chunked GKR session. Returns the session ID.
    fn open_gkr_session(
        ref self: TContractState,
        model_id: felt252,
        total_felts: u32,
        circuit_depth: u32,
        num_layers: u32,
        weight_binding_mode: u32,
        packed: u32,
        io_packed: u32,
    ) -> u64;

    /// Upload a chunk of session data. Chunks must be uploaded sequentially.
    /// chunk_data length is used as the chunk felt count (no separate param needed).
    fn upload_gkr_chunk(
        ref self: TContractState,
        session_id: u64,
        chunk_idx: u32,
        chunk_data: Array<felt252>,
    );

    /// Seal a session after all chunks are uploaded. Verifies total felt count.
    fn seal_gkr_session(ref self: TContractState, session_id: u64);

    /// Verify a sealed session's proof data. Reassembles chunks and runs
    /// the same verification as verify_model_gkr_v4_packed_io.
    fn verify_gkr_from_session(ref self: TContractState, session_id: u64) -> bool;

    // ─── Two-Phase Verification (v24+) ───────────────────────────────
    // For proofs where verify_gkr_from_session exceeds the step limit.
    // Phase 1: feed chunks as calldata (hash-verified against upload hashes)
    // Phase 2: execute verification from stored data

    /// Feed a chunk of data for verification. Verifies Poseidon hash matches
    /// the hash stored during upload, then stores data with flat index.
    /// Chunks must be fed in order (chunk_idx = 0, 1, 2, ...).
    fn verify_gkr_feed_chunk(
        ref self: TContractState,
        session_id: u64,
        chunk_idx: u32,
        chunk_data: Array<felt252>,
    );

    /// Execute verification after all chunks have been fed.
    /// Reads data from flat-indexed storage and runs the GKR walk.
    fn verify_gkr_execute(ref self: TContractState, session_id: u64) -> bool;

    // ─── Streaming GKR Verification (v25) ────────────────────────────
    // Zero-storage-read verification: proof data flows as calldata only.
    // Protocol: stream_init → stream_layers × M → stream_finalize.
    // Only ~28 felts of checkpoint state stored between TXs.

    /// Initialize a streaming GKR verification session.
    /// Computes IO commitment, validates dimensions, seeds channel,
    /// stores initial checkpoint state. Does NOT evaluate output MLE.
    fn verify_gkr_stream_init(
        ref self: TContractState,
        session_id: u64,
        original_io_len: u32,
        packed_raw_io: Array<felt252>,
        circuit_depth: u32,
        num_layers: u32,
        in_cols: u32,
        out_cols: u32,
    );

    /// Evaluate output MLE in chunked TXs (splits expensive computation across
    /// multiple TXs to fit within Starknet's gas limit).
    /// Must be called after stream_init and before stream_layers.
    ///
    /// First call: draws r-points from channel, stores them, processes first chunk.
    /// Subsequent calls: reads stored r-points, processes chunk, accumulates.
    /// Last call (is_last_chunk=true): mixes final sum into channel, stores initial claim.
    ///
    /// Parameters:
    /// - packed_output_data: packed felt252s for this chunk (8 M31 per felt)
    /// - chunk_offset: starting M31 index in the full output array
    /// - chunk_len: number of M31 values in this chunk
    /// - is_last_chunk: if true, finalize the output MLE evaluation
    fn verify_gkr_stream_init_output_mle(
        ref self: TContractState,
        session_id: u64,
        packed_output_data: Array<felt252>,
        chunk_offset: u32,
        chunk_len: u32,
        is_last_chunk: bool,
    );

    /// Verify a batch of GKR layers. Proof data for this batch arrives
    /// as calldata (never stored). Updates checkpoint state.
    fn verify_gkr_stream_layers(
        ref self: TContractState,
        session_id: u64,
        batch_idx: u32,
        num_layers_in_batch: u32,
        matmul_dims: Array<u32>,
        dequantize_bits: Array<u64>,
        proof_data: Array<felt252>,
    );

    /// Chunked input MLE evaluation for finalize step.
    /// First call: performs weight binding + deferred proofs + IO commitment check,
    ///   draws no extra randomness (input MLE uses final_claim.point from walk),
    ///   stores r-points from final_claim.point, processes first chunk.
    /// Subsequent calls: reload stored r-points, process chunk, accumulate partial sum.
    /// Last call: store final input MLE value for finalize to read.
    fn verify_gkr_stream_finalize_input_mle(
        ref self: TContractState,
        session_id: u64,
        packed_input_data: Array<felt252>,
        chunk_offset: u32,
        chunk_len: u32,
        is_last_chunk: bool,
        // Only needed on first call (weight binding + IO commitment):
        weight_expected_values: Array<felt252>,
        weight_binding_mode: u32,
        weight_binding_data: Array<felt252>,
        deferred_proof_data: Array<felt252>,
        deferred_matmul_dims: Array<u32>,
        original_io_len: u32,
        packed_raw_io: Array<felt252>,
    );

    /// Finalize streaming verification: reads pre-computed input MLE value,
    /// asserts it matches the final GKR claim, and records the proof on-chain.
    fn verify_gkr_stream_finalize(
        ref self: TContractState,
        session_id: u64,
    ) -> bool;
}

#[starknet::contract]
mod SumcheckVerifierContract {
    use super::{
        GKRClaim, MleOpeningProof, UPGRADE_DELAY,
    };
    use crate::field::{
        QM31, log2_ceil, next_power_of_two,
        evaluate_mle_from_packed_1row,
        evaluate_mle_eq_dot_partial,
        extract_m31_from_packed,
        pack_qm31_to_felt, unpack_qm31_from_felt,
        qm31_zero, qm31_one, qm31_add, qm31_mul,
    };
    use crate::channel::{
        PoseidonChannel,
        channel_default, channel_mix_u64,
        channel_draw_qm31, channel_draw_qm31s, channel_mix_secure_field,
    };
    // mle stripped for lean v18b
    use crate::model_verifier::{
        verify_gkr_model_with_trace_dp,
        verify_gkr_layers_batch,
        reader_new as mv_reader_new, read_u32 as mv_read_u32,
        read_qm31 as mv_read_qm31, dispatch_matmul as mv_dispatch_matmul,
    };
    use super::PackedDigest;
    use starknet::storage::{
        StoragePointerReadAccess, StoragePointerWriteAccess, Map, StoragePathEntry,
    };
    use starknet::{ClassHash, ContractAddress, get_caller_address, get_block_timestamp};

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
        /// model_id → aggregate Poseidon hash of all weight commitment roots.
        /// Computed during registration: poseidon_hash_span([root_0, root_1, ..., root_N-1]).
        /// Saves N-1 storage reads during verification (1 read + hash vs N reads).
        model_weight_root_hash: Map<felt252, felt252>,
        /// Pending upgrade class hash (zero = no pending upgrade).
        pending_upgrade: ClassHash,
        /// Timestamp when the upgrade was proposed.
        upgrade_proposed_at: u64,
        // Audit/Access/ViewKey storage stripped for lean v18b deploy.
        // Storage slots preserved on-chain. Will be restored in next version.

        // ─── Chunked GKR Session Storage ─────────────────────────────────
        /// Next session ID (incremented on each open_gkr_session call).
        next_session_id: u64,
        /// session_id → session metadata.
        session_owner: Map<u64, ContractAddress>,
        session_model_id: Map<u64, felt252>,
        session_circuit_depth: Map<u64, u32>,
        session_num_layers: Map<u64, u32>,
        session_weight_binding_mode: Map<u64, u32>,
        session_packed: Map<u64, bool>,
        session_io_packed: Map<u64, bool>,
        session_total_felts: Map<u64, u32>,
        session_num_chunks: Map<u64, u32>,
        session_chunks_uploaded: Map<u64, u32>,
        session_sealed: Map<u64, bool>,
        /// (session_id, chunk_idx, felt_idx) → felt252 value (legacy, kept for storage compat).
        session_data: Map<(u64, u32, u32), felt252>,
        /// (session_id, chunk_idx) → felt count in this chunk.
        session_chunk_len: Map<(u64, u32), u32>,
        /// (session_id, chunk_idx) → Poseidon hash of chunk data (v24+).
        session_chunk_hash: Map<(u64, u32), felt252>,
        /// (session_id, flat_idx) → felt252 value for verification data (v24+).
        /// Flat index across all chunks, written during verify_gkr_feed_chunk.
        session_verify_data: Map<(u64, u32), felt252>,
        /// session_id → total felts fed for verification (v24+).
        session_verify_fed_total: Map<u64, u32>,
        /// session_id → number of chunks fed for verification (v24+).
        session_verify_chunks_fed: Map<u64, u32>,

        // ─── Streaming GKR Verification State (v25) ─────────────────────
        // Checkpoint state between streaming TXs (~28 felts per TX).
        // Each streaming session processes layers in batches via calldata,
        // storing only intermediate verification state (no proof data).

        /// session_id → Fiat-Shamir channel digest at checkpoint.
        stream_channel_digest: Map<u64, felt252>,
        /// session_id → Fiat-Shamir channel draw counter at checkpoint.
        stream_channel_counter: Map<u64, u32>,
        /// session_id → packed QM31 claim value at checkpoint.
        stream_claim_value: Map<u64, felt252>,
        /// session_id → claim point length at checkpoint.
        stream_claim_point_len: Map<u64, u32>,
        /// (session_id, index) → packed QM31 claim point element.
        stream_claim_point: Map<(u64, u32), felt252>,
        /// session_id → number of layers verified so far.
        stream_layers_verified: Map<u64, u32>,
        /// session_id → running Poseidon hash of packed weight expected_values.
        stream_weight_hash: Map<u64, felt252>,
        /// session_id → running Poseidon hash of layer tags.
        stream_tags_hash: Map<u64, felt252>,
        /// session_id → number of weight claims accumulated.
        stream_weight_count: Map<u64, u32>,
        /// session_id → IO commitment (Poseidon hash of packed IO data).
        stream_io_commitment: Map<u64, felt252>,
        /// session_id → whether stream_init has been called.
        stream_initialized: Map<u64, bool>,
        /// session_id → whether stream_finalize has been called.
        stream_finalized: Map<u64, bool>,
        /// session_id → total number of layers in the model circuit.
        stream_total_layers: Map<u64, u32>,
        /// session_id → circuit_depth for Fiat-Shamir seeding.
        stream_circuit_depth: Map<u64, u32>,

        // ─── Streaming IO metadata (for input MLE check in finalize) ────
        /// session_id → packed_raw_io stored for finalize input MLE evaluation.
        /// (session_id, felt_idx) → packed felt252 value.
        stream_packed_io: Map<(u64, u32), felt252>,
        /// session_id → number of packed IO felts stored.
        stream_packed_io_len: Map<u64, u32>,
        /// session_id → original_io_len (M31 element count).
        stream_original_io_len: Map<u64, u32>,
        /// session_id → output column count (for output MLE step).
        stream_out_cols: Map<u64, u32>,
        /// session_id → input column count (for input MLE step).
        stream_in_cols: Map<u64, u32>,
        /// session_id → M31 start index of output data in packed IO.
        stream_out_data_m31_start: Map<u64, u32>,
        /// session_id → whether output MLE has been evaluated.
        stream_output_mle_done: Map<u64, bool>,
        /// session_id → number of output MLE chunks processed so far.
        stream_output_mle_chunks_done: Map<u64, u32>,
        /// session_id → number of r-points stored (= n_vars = log2(padded_out_cols)).
        stream_output_mle_r_count: Map<u64, u32>,
        /// (session_id, var_idx) → packed QM31 r-point for output MLE evaluation.
        stream_output_mle_r_point: Map<(u64, u32), felt252>,
        /// session_id → packed QM31 running partial sum for output MLE.
        stream_output_mle_partial_sum: Map<u64, felt252>,
        /// session_id → whether input MLE chunked evaluation is done.
        stream_input_mle_done: Map<u64, bool>,
        /// session_id → number of input MLE chunks processed so far.
        stream_input_mle_chunks_done: Map<u64, u32>,
        /// session_id → number of r-points stored for input MLE.
        stream_input_mle_r_count: Map<u64, u32>,
        /// (session_id, var_idx) → packed QM31 r-point for input MLE.
        stream_input_mle_r_point: Map<(u64, u32), felt252>,
        /// session_id → packed QM31 running partial sum for input MLE.
        stream_input_mle_partial_sum: Map<u64, felt252>,
        /// session_id → packed QM31 final input MLE value (set when all chunks done).
        stream_input_mle_value: Map<u64, felt252>,
        /// session_id → whether weight binding has been done in finalize_input_mle.
        stream_weight_binding_done: Map<u64, bool>,
        /// session_id → expected next batch index (for sequential enforcement).
        stream_next_batch_idx: Map<u64, u32>,
        /// session_id → total deferred Add points accumulated across batches.
        stream_deferred_count: Map<u64, u32>,
        /// (session_id, deferred_idx) → point length for this deferred point.
        stream_deferred_point_len: Map<(u64, u32), u32>,
        /// (session_id, deferred_idx, coord_idx) → packed QM31 coordinate.
        stream_deferred_point: Map<(u64, u32, u32), felt252>,
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        ModelRegistered: ModelRegistered,
        ModelGkrRegistered: ModelGkrRegistered,
        ModelGkrVerified: ModelGkrVerified,
        VerificationFailed: VerificationFailed,
        UpgradeProposed: UpgradeProposed,
        UpgradeExecuted: UpgradeExecuted,
        UpgradeCancelled: UpgradeCancelled,
        GkrSessionOpened: GkrSessionOpened,
        GkrSessionSealed: GkrSessionSealed,
        GkrStreamStarted: GkrStreamStarted,
        GkrStreamProgress: GkrStreamProgress,
        GkrStreamFinalized: GkrStreamFinalized,
        // Audit/access/view events stripped for lean v18b deploy.
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
        session_id: u64,
        caller: ContractAddress,
        model_id: felt252,
        total_felts: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrSessionSealed {
        #[key]
        session_id: u64,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrStreamStarted {
        #[key]
        session_id: u64,
        model_id: felt252,
        num_layers: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrStreamProgress {
        #[key]
        session_id: u64,
        layers_verified: u32,
        total_layers: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrStreamFinalized {
        #[key]
        session_id: u64,
        proof_hash: felt252,
    }

    const WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL: u32 = 3;
    const WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK: u32 = 4;
    /// Marker tag for mode 4 RLC-only binding (0x524C43 = "RLC").
    const WEIGHT_BINDING_RLC_MARKER: felt252 = 0x524C43;

    // M31 packing: 8 M31 values (31 bits each) packed into one felt252 (248 bits).
    // Layout: v0 | (v1 << 31) | (v2 << 62) | ... | (v7 << 217)
    //
    // Unpacking uses u128 limb arithmetic to avoid expensive u256 division.
    // The packed value's low 128 bits contain values 0..3 (bits 0..123),
    // and high 128 bits contain values 4..7 (bits 124..247).
    // Value 4 straddles the boundary: bits 124..127 in low, bits 0..26 in high.
    const M31_MASK_128: u128 = 0x7FFFFFFF; // 2^31 - 1

    // unpack_m31_packed_io, verify_model_gkr_core, verify_model_gkr_core_with_io_commitment
    // stripped for lean v18b — packed_io entrypoint is self-contained.
    // See git history for full implementations.
    //

    // Core functions stripped for lean v18b deploy:
    // - unpack_m31_packed_io (no longer needed — packed_io evaluates directly)
    // - verify_model_gkr_core (only called by non-packed entrypoint)
    // - verify_model_gkr_core_with_io_commitment (only called by core)
    // See git history for full implementations.

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

            // Store weight commitments (one per MatMul layer) + aggregate hash
            let num_weights: u32 = weight_commitments.len();
            self.model_gkr_weight_count.entry(model_id).write(num_weights);
            let mut weight_hash_input: Array<felt252> = array![];
            let mut i: u32 = 0;
            loop {
                if i >= num_weights {
                    break;
                }
                let root = *weight_commitments.at(i);
                assert!(root != 0, "Weight commitment cannot be zero");
                self.model_gkr_weights.entry((model_id, i)).write(root);
                weight_hash_input.append(root);
                i += 1;
            };
            // Single aggregate hash: saves N-1 storage reads during verification
            let weight_root_hash = core::poseidon::poseidon_hash_span(weight_hash_input.span());
            self.model_weight_root_hash.entry(model_id).write(weight_root_hash);

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

        fn register_model_gkr_streaming_circuit(
            ref self: ContractState,
            model_id: felt252,
            circuit_depth: u32,
            layer_tags: Array<u32>,
        ) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");
            // Model must already be registered for GKR (weights exist)
            let weight_count = self.model_gkr_weight_count.entry(model_id).read();
            assert!(weight_count > 0, "Model not registered for GKR");

            // Compute incremental tags hash: poseidon(poseidon(...poseidon(0, tag_0)...), tag_N)
            let mut tags_hash: felt252 = 0;
            let mut i: u32 = 0;
            loop {
                if i >= layer_tags.len() {
                    break;
                }
                let tag_felt: felt252 = (*layer_tags.at(i)).into();
                tags_hash = core::poseidon::poseidon_hash_span(
                    array![tags_hash, tag_felt].span(),
                );
                i += 1;
            };
            // Final circuit hash: poseidon(circuit_depth, tags_hash)
            let streaming_circuit_hash = core::poseidon::poseidon_hash_span(
                array![circuit_depth.into(), tags_hash].span(),
            );

            // Overwrite the circuit hash with streaming-compatible version
            self.model_circuit_hash.entry(model_id).write(streaming_circuit_hash);
        }

        fn verify_model_gkr_v4_packed_io(
            ref self: ContractState,
            model_id: felt252,
            original_io_len: u32,
            packed_raw_io: Array<felt252>,
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
            let (proof_hash, io_commitment) = InternalImpl::verify_model_gkr_v4_packed_io_core(
                @self, model_id, original_io_len, packed_raw_io, circuit_depth,
                num_layers, matmul_dims, dequantize_bits, proof_data,
                weight_commitments, weight_binding_mode, weight_binding_data,
                weight_opening_proofs,
            );

            // Record proof on-chain
            assert!(!self.verified_proofs.entry(proof_hash).read(), "PROOF_ALREADY_VERIFIED");
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);
            self.emit(ModelGkrVerified { model_id, proof_hash, io_commitment, num_layers });
            true
        }

        fn verify_model_gkr_v4_packed_io_dp(
            ref self: ContractState,
            model_id: felt252,
            original_io_len: u32,
            packed_raw_io: Array<felt252>,
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
            let (proof_hash, io_commitment) = InternalImpl::verify_model_gkr_v4_packed_io_core_dp(
                @self, model_id, original_io_len, packed_raw_io, circuit_depth,
                num_layers, matmul_dims, dequantize_bits, proof_data,
                weight_commitments, weight_binding_mode, weight_binding_data,
                weight_opening_proofs, true,
            );

            // Record proof on-chain
            assert!(!self.verified_proofs.entry(proof_hash).read(), "PROOF_ALREADY_VERIFIED");
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);
            self.emit(ModelGkrVerified { model_id, proof_hash, io_commitment, num_layers });
            true
        }

        fn verify_model_gkr_v4_packed_io_view(
            self: @ContractState,
            model_id: felt252,
            original_io_len: u32,
            packed_raw_io: Array<felt252>,
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
            // View-only: verify but skip proof recording (no storage writes/events)
            let (_proof_hash, _io_commitment) = InternalImpl::verify_model_gkr_v4_packed_io_core(
                self, model_id, original_io_len, packed_raw_io, circuit_depth,
                num_layers, matmul_dims, dequantize_bits, proof_data,
                weight_commitments, weight_binding_mode, weight_binding_data,
                weight_opening_proofs,
            );
            true
        }

        fn get_model_circuit_hash(self: @ContractState, model_id: felt252) -> felt252 {
            self.model_circuit_hash.entry(model_id).read()
        }

        fn get_model_gkr_weight_count(self: @ContractState, model_id: felt252) -> u32 {
            self.model_gkr_weight_count.entry(model_id).read()
        }

        // ─── Chunked GKR Session Entrypoints ─────────────────────────────

        fn open_gkr_session(
            ref self: ContractState,
            model_id: felt252,
            total_felts: u32,
            circuit_depth: u32,
            num_layers: u32,
            weight_binding_mode: u32,
            packed: u32,
            io_packed: u32,
        ) -> u64 {
            let caller = get_caller_address();
            let session_id = self.next_session_id.read() + 1;
            self.next_session_id.write(session_id);

            self.session_owner.entry(session_id).write(caller);
            self.session_model_id.entry(session_id).write(model_id);
            self.session_circuit_depth.entry(session_id).write(circuit_depth);
            self.session_num_layers.entry(session_id).write(num_layers);
            self.session_weight_binding_mode.entry(session_id).write(weight_binding_mode);
            self.session_packed.entry(session_id).write(packed != 0);
            self.session_io_packed.entry(session_id).write(io_packed != 0);
            self.session_total_felts.entry(session_id).write(total_felts);
            self.session_sealed.entry(session_id).write(false);
            self.session_chunks_uploaded.entry(session_id).write(0);

            self.emit(GkrSessionOpened {
                session_id, caller, model_id, total_felts,
            });

            session_id
        }

        fn upload_gkr_chunk(
            ref self: ContractState,
            session_id: u64,
            chunk_idx: u32,
            chunk_data: Array<felt252>,
        ) {
            let caller = get_caller_address();
            let owner = self.session_owner.entry(session_id).read();
            let owner_felt: felt252 = owner.into();
            assert!(owner_felt != 0, "SESSION_NOT_FOUND");
            assert!(caller == owner, "NOT_SESSION_OWNER");
            assert!(!self.session_sealed.entry(session_id).read(), "SESSION_ALREADY_SEALED");

            let expected_idx = self.session_chunks_uploaded.entry(session_id).read();
            assert!(chunk_idx == expected_idx, "CHUNK_IDX_OUT_OF_ORDER");

            let chunk_felt_count = chunk_data.len();
            assert!(chunk_felt_count > 0, "EMPTY_CHUNK");

            // v24: Store only Poseidon hash of chunk data (not individual felts).
            // Data is re-submitted as calldata during verify_gkr_feed_chunk.
            let chunk_hash = core::poseidon::poseidon_hash_span(chunk_data.span());
            self.session_chunk_hash.entry((session_id, chunk_idx)).write(chunk_hash);
            self.session_chunk_len.entry((session_id, chunk_idx)).write(chunk_felt_count);
            self.session_chunks_uploaded.entry(session_id).write(expected_idx + 1);
        }

        fn seal_gkr_session(ref self: ContractState, session_id: u64) {
            let caller = get_caller_address();
            let owner = self.session_owner.entry(session_id).read();
            let owner_felt: felt252 = owner.into();
            assert!(owner_felt != 0, "SESSION_NOT_FOUND");
            assert!(caller == owner, "NOT_SESSION_OWNER");
            assert!(!self.session_sealed.entry(session_id).read(), "SESSION_ALREADY_SEALED");

            // Verify total felt count matches sum of chunk lengths
            let num_chunks = self.session_chunks_uploaded.entry(session_id).read();
            let expected_total = self.session_total_felts.entry(session_id).read();
            let mut actual_total: u32 = 0;
            let mut c: u32 = 0;
            loop {
                if c >= num_chunks {
                    break;
                }
                actual_total += self.session_chunk_len.entry((session_id, c)).read();
                c += 1;
            };
            assert!(actual_total == expected_total, "TOTAL_FELTS_MISMATCH");

            self.session_sealed.entry(session_id).write(true);
            self.session_num_chunks.entry(session_id).write(num_chunks);

            self.emit(GkrSessionSealed { session_id });
        }

        fn verify_gkr_from_session(ref self: ContractState, session_id: u64) -> bool {
            assert!(self.session_sealed.entry(session_id).read(), "SESSION_NOT_SEALED");

            // Read session metadata
            let model_id = self.session_model_id.entry(session_id).read();
            let circuit_depth = self.session_circuit_depth.entry(session_id).read();
            let num_layers = self.session_num_layers.entry(session_id).read();
            let weight_binding_mode = self.session_weight_binding_mode.entry(session_id).read();
            let _is_packed = self.session_packed.entry(session_id).read();
            let is_io_packed = self.session_io_packed.entry(session_id).read();
            let num_chunks = self.session_num_chunks.entry(session_id).read();

            // Reassemble flat data array from chunks
            let mut flat: Array<felt252> = array![];
            let mut c: u32 = 0;
            loop {
                if c >= num_chunks {
                    break;
                }
                let chunk_len = self.session_chunk_len.entry((session_id, c)).read();
                let mut i: u32 = 0;
                loop {
                    if i >= chunk_len {
                        break;
                    }
                    flat.append(self.session_data.entry((session_id, c, i)).read());
                    i += 1;
                };
                c += 1;
            };

            let flat_span = flat.span();
            let flat_len: u32 = flat_span.len();
            let mut off: u32 = 0;

            // Parse 7 length-prefixed sections from flat data.
            // Layout matches paymaster_submit.mjs sessionData builder:
            //   1. raw_io (IO-packed: [original_len, packed_count, packed_data...])
            //   2. matmul_dims: [len, data...]
            //   3. dequantize_bits: [len, data...]
            //   4. proof_data: [len, data...]
            //   5. weight_commitments: [len, data...]
            //   6. weight_binding_data: [len, data...]
            //   7. weight_opening_proofs: [count, ...]

            // Section 1: raw_io
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_IO");
            let mut original_io_len: u32 = 0;
            let mut packed_raw_io: Array<felt252> = array![];

            if is_io_packed {
                // IO-packed: [original_io_len, packed_count, packed_data...]
                let orig_len_felt: u256 = (*flat_span.at(off)).into();
                original_io_len = orig_len_felt.try_into().unwrap();
                off += 1;
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_IO_PACKED_COUNT");
                let packed_count_felt: u256 = (*flat_span.at(off)).into();
                let packed_count: u32 = packed_count_felt.try_into().unwrap();
                off += 1;
                let mut pi: u32 = 0;
                loop {
                    if pi >= packed_count {
                        break;
                    }
                    assert!(off < flat_len, "SESSION_DATA_TRUNCATED_IO_PACKED_DATA");
                    packed_raw_io.append(*flat_span.at(off));
                    off += 1;
                    pi += 1;
                };
            } else {
                // Non-packed: [len, raw_io_data...]
                let sec_len_felt: u256 = (*flat_span.at(off)).into();
                let sec_len: u32 = sec_len_felt.try_into().unwrap();
                off += 1;
                original_io_len = sec_len;
                let mut ri: u32 = 0;
                loop {
                    if ri >= sec_len {
                        break;
                    }
                    assert!(off < flat_len, "SESSION_DATA_TRUNCATED_RAW_IO");
                    packed_raw_io.append(*flat_span.at(off));
                    off += 1;
                    ri += 1;
                };
            };

            // Section 2: matmul_dims
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_MATMUL_DIMS");
            let md_len_felt: u256 = (*flat_span.at(off)).into();
            let md_len: u32 = md_len_felt.try_into().unwrap();
            off += 1;
            let mut matmul_dims: Array<u32> = array![];
            let mut mi: u32 = 0;
            loop {
                if mi >= md_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_MATMUL_DIMS_DATA");
                let v: u256 = (*flat_span.at(off)).into();
                matmul_dims.append(v.try_into().unwrap());
                off += 1;
                mi += 1;
            };

            // Section 3: dequantize_bits
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_DEQ");
            let db_len_felt: u256 = (*flat_span.at(off)).into();
            let db_len: u32 = db_len_felt.try_into().unwrap();
            off += 1;
            let mut dequantize_bits: Array<u64> = array![];
            let mut di: u32 = 0;
            loop {
                if di >= db_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_DEQ_DATA");
                let v: u256 = (*flat_span.at(off)).into();
                dequantize_bits.append(v.try_into().unwrap());
                off += 1;
                di += 1;
            };

            // Section 4: proof_data
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_PROOF");
            let pd_len_felt: u256 = (*flat_span.at(off)).into();
            let pd_len: u32 = pd_len_felt.try_into().unwrap();
            off += 1;
            let mut proof_data: Array<felt252> = array![];
            let mut pdi: u32 = 0;
            loop {
                if pdi >= pd_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_PROOF_DATA");
                proof_data.append(*flat_span.at(off));
                off += 1;
                pdi += 1;
            };

            // Section 5: weight_commitments
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WC");
            let wc_len_felt: u256 = (*flat_span.at(off)).into();
            let wc_len: u32 = wc_len_felt.try_into().unwrap();
            off += 1;
            let mut weight_commitments: Array<felt252> = array![];
            let mut wi: u32 = 0;
            loop {
                if wi >= wc_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WC_DATA");
                weight_commitments.append(*flat_span.at(off));
                off += 1;
                wi += 1;
            };

            // Section 6: weight_binding_data
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WBD");
            let wbd_len_felt: u256 = (*flat_span.at(off)).into();
            let wbd_len: u32 = wbd_len_felt.try_into().unwrap();
            off += 1;
            let mut weight_binding_data: Array<felt252> = array![];
            let mut wbi: u32 = 0;
            loop {
                if wbi >= wbd_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WBD_DATA");
                weight_binding_data.append(*flat_span.at(off));
                off += 1;
                wbi += 1;
            };

            // Section 7: weight_opening_proofs
            // For Mode 4 RLC, this is just [0] (empty array).
            // Full MLE opening proofs would require Serde deserialization
            // from flat felts, but we only support RLC mode in this entrypoint.
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WOP");
            let wop_count_felt: u256 = (*flat_span.at(off)).into();
            let wop_count: u32 = wop_count_felt.try_into().unwrap();
            off += 1;
            let weight_opening_proofs: Array<MleOpeningProof> = array![];
            assert!(wop_count == 0, "SESSION_VERIFY_ONLY_SUPPORTS_RLC_MODE");
            // For wop_count == 0, no further data to parse.

            assert!(off == flat_len, "SESSION_DATA_TRAILING");

            // Run core verification (same as verify_model_gkr_v4_packed_io)
            let (proof_hash, io_commitment) = InternalImpl::verify_model_gkr_v4_packed_io_core(
                @self, model_id, original_io_len, packed_raw_io, circuit_depth,
                num_layers, matmul_dims, dequantize_bits, proof_data,
                weight_commitments, weight_binding_mode, weight_binding_data,
                weight_opening_proofs,
            );

            // Record proof on-chain
            assert!(!self.verified_proofs.entry(proof_hash).read(), "PROOF_ALREADY_VERIFIED");
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);
            self.emit(ModelGkrVerified { model_id, proof_hash, io_commitment, num_layers });
            true
        }

        // ─── Two-Phase Verification (v24+) ───────────────────────────────

        fn verify_gkr_feed_chunk(
            ref self: ContractState,
            session_id: u64,
            chunk_idx: u32,
            chunk_data: Array<felt252>,
        ) {
            assert!(self.session_sealed.entry(session_id).read(), "SESSION_NOT_SEALED");
            let caller = get_caller_address();
            let owner = self.session_owner.entry(session_id).read();
            assert!(caller == owner, "NOT_SESSION_OWNER");

            // Verify chunk is fed in order
            let expected_idx = self.session_verify_chunks_fed.entry(session_id).read();
            assert!(chunk_idx == expected_idx, "FEED_CHUNK_IDX_OUT_OF_ORDER");

            // Verify chunk hash matches what was stored during upload
            let stored_hash = self.session_chunk_hash.entry((session_id, chunk_idx)).read();
            assert!(stored_hash != 0, "CHUNK_HASH_NOT_FOUND");
            let data_hash = core::poseidon::poseidon_hash_span(chunk_data.span());
            assert!(data_hash == stored_hash, "FEED_CHUNK_HASH_MISMATCH");

            // Verify chunk length matches
            let stored_len = self.session_chunk_len.entry((session_id, chunk_idx)).read();
            assert!(chunk_data.len() == stored_len, "FEED_CHUNK_LEN_MISMATCH");

            // Store data with flat index for efficient reading during execute
            let flat_offset = self.session_verify_fed_total.entry(session_id).read();
            let chunk_len = chunk_data.len();
            let mut i: u32 = 0;
            loop {
                if i >= chunk_len {
                    break;
                }
                self.session_verify_data.entry((session_id, flat_offset + i)).write(*chunk_data.at(i));
                i += 1;
            };

            self.session_verify_fed_total.entry(session_id).write(flat_offset + chunk_len);
            self.session_verify_chunks_fed.entry(session_id).write(expected_idx + 1);
        }

        fn verify_gkr_execute(ref self: ContractState, session_id: u64) -> bool {
            assert!(self.session_sealed.entry(session_id).read(), "SESSION_NOT_SEALED");

            // Verify all chunks have been fed
            let num_chunks = self.session_num_chunks.entry(session_id).read();
            let chunks_fed = self.session_verify_chunks_fed.entry(session_id).read();
            assert!(chunks_fed == num_chunks, "NOT_ALL_CHUNKS_FED");

            let total_felts = self.session_verify_fed_total.entry(session_id).read();
            let expected_total = self.session_total_felts.entry(session_id).read();
            assert!(total_felts == expected_total, "FED_TOTAL_MISMATCH");

            // Read session metadata
            let model_id = self.session_model_id.entry(session_id).read();
            let circuit_depth = self.session_circuit_depth.entry(session_id).read();
            let num_layers = self.session_num_layers.entry(session_id).read();
            let weight_binding_mode = self.session_weight_binding_mode.entry(session_id).read();
            let is_io_packed = self.session_io_packed.entry(session_id).read();

            // Read all data from flat-indexed storage into array
            let mut flat: Array<felt252> = array![];
            let mut fi: u32 = 0;
            loop {
                if fi >= total_felts {
                    break;
                }
                flat.append(self.session_verify_data.entry((session_id, fi)).read());
                fi += 1;
            };

            // Parse sections and verify (same as verify_gkr_from_session)
            let flat_span = flat.span();
            let flat_len: u32 = flat_span.len();
            let mut off: u32 = 0;

            // Section 1: raw_io
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_IO");
            let mut original_io_len: u32 = 0;
            let mut packed_raw_io: Array<felt252> = array![];

            if is_io_packed {
                let orig_len_felt: u256 = (*flat_span.at(off)).into();
                original_io_len = orig_len_felt.try_into().unwrap();
                off += 1;
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_IO_PACKED_COUNT");
                let packed_count_felt: u256 = (*flat_span.at(off)).into();
                let packed_count: u32 = packed_count_felt.try_into().unwrap();
                off += 1;
                let mut pi: u32 = 0;
                loop {
                    if pi >= packed_count {
                        break;
                    }
                    assert!(off < flat_len, "SESSION_DATA_TRUNCATED_IO_PACKED_DATA");
                    packed_raw_io.append(*flat_span.at(off));
                    off += 1;
                    pi += 1;
                };
            } else {
                let sec_len_felt: u256 = (*flat_span.at(off)).into();
                let sec_len: u32 = sec_len_felt.try_into().unwrap();
                off += 1;
                original_io_len = sec_len;
                let mut ri: u32 = 0;
                loop {
                    if ri >= sec_len {
                        break;
                    }
                    assert!(off < flat_len, "SESSION_DATA_TRUNCATED_RAW_IO");
                    packed_raw_io.append(*flat_span.at(off));
                    off += 1;
                    ri += 1;
                };
            };

            // Section 2: matmul_dims
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_MATMUL_DIMS");
            let md_len_felt: u256 = (*flat_span.at(off)).into();
            let md_len: u32 = md_len_felt.try_into().unwrap();
            off += 1;
            let mut matmul_dims: Array<u32> = array![];
            let mut mi: u32 = 0;
            loop {
                if mi >= md_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_MATMUL_DIMS_DATA");
                let v: u256 = (*flat_span.at(off)).into();
                matmul_dims.append(v.try_into().unwrap());
                off += 1;
                mi += 1;
            };

            // Section 3: dequantize_bits
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_DEQ");
            let db_len_felt: u256 = (*flat_span.at(off)).into();
            let db_len: u32 = db_len_felt.try_into().unwrap();
            off += 1;
            let mut dequantize_bits: Array<u64> = array![];
            let mut di: u32 = 0;
            loop {
                if di >= db_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_DEQ_DATA");
                let v: u256 = (*flat_span.at(off)).into();
                dequantize_bits.append(v.try_into().unwrap());
                off += 1;
                di += 1;
            };

            // Section 4: proof_data
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_PROOF");
            let pd_len_felt: u256 = (*flat_span.at(off)).into();
            let pd_len: u32 = pd_len_felt.try_into().unwrap();
            off += 1;
            let mut proof_data: Array<felt252> = array![];
            let mut pdi: u32 = 0;
            loop {
                if pdi >= pd_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_PROOF_DATA");
                proof_data.append(*flat_span.at(off));
                off += 1;
                pdi += 1;
            };

            // Section 5: weight_commitments
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WC");
            let wc_len_felt: u256 = (*flat_span.at(off)).into();
            let wc_len: u32 = wc_len_felt.try_into().unwrap();
            off += 1;
            let mut weight_commitments: Array<felt252> = array![];
            let mut wi: u32 = 0;
            loop {
                if wi >= wc_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WC_DATA");
                weight_commitments.append(*flat_span.at(off));
                off += 1;
                wi += 1;
            };

            // Section 6: weight_binding_data
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WBD");
            let wbd_len_felt: u256 = (*flat_span.at(off)).into();
            let wbd_len: u32 = wbd_len_felt.try_into().unwrap();
            off += 1;
            let mut weight_binding_data: Array<felt252> = array![];
            let mut wbi: u32 = 0;
            loop {
                if wbi >= wbd_len {
                    break;
                }
                assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WBD_DATA");
                weight_binding_data.append(*flat_span.at(off));
                off += 1;
                wbi += 1;
            };

            // Section 7: weight_opening_proofs
            assert!(off < flat_len, "SESSION_DATA_TRUNCATED_WOP");
            let wop_count_felt: u256 = (*flat_span.at(off)).into();
            let wop_count: u32 = wop_count_felt.try_into().unwrap();
            off += 1;
            let weight_opening_proofs: Array<MleOpeningProof> = array![];
            assert!(wop_count == 0, "SESSION_VERIFY_ONLY_SUPPORTS_RLC_MODE");

            assert!(off == flat_len, "SESSION_DATA_TRAILING");

            // Run core verification
            let (proof_hash, io_commitment) = InternalImpl::verify_model_gkr_v4_packed_io_core(
                @self, model_id, original_io_len, packed_raw_io, circuit_depth,
                num_layers, matmul_dims, dequantize_bits, proof_data,
                weight_commitments, weight_binding_mode, weight_binding_data,
                weight_opening_proofs,
            );

            // Record proof on-chain
            assert!(!self.verified_proofs.entry(proof_hash).read(), "PROOF_ALREADY_VERIFIED");
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);
            self.emit(ModelGkrVerified { model_id, proof_hash, io_commitment, num_layers });
            true
        }

        // ─── Streaming GKR Verification (v25) ───────────────────────────

        fn verify_gkr_stream_init(
            ref self: ContractState,
            session_id: u64,
            original_io_len: u32,
            packed_raw_io: Array<felt252>,
            circuit_depth: u32,
            num_layers: u32,
            in_cols: u32,
            out_cols: u32,
        ) {
            // Auth: must be session owner, session must be sealed
            assert!(self.session_sealed.entry(session_id).read(), "SESSION_NOT_SEALED");
            let caller = get_caller_address();
            let owner = self.session_owner.entry(session_id).read();
            assert!(caller == owner, "NOT_SESSION_OWNER");
            assert!(!self.stream_initialized.entry(session_id).read(), "STREAM_ALREADY_INITIALIZED");

            // Compute IO commitment from packed felts
            let mut commitment_input: Array<felt252> = array![original_io_len.into()];
            let mut ci: u32 = 0;
            loop {
                if ci >= packed_raw_io.len() {
                    break;
                }
                commitment_input.append(*packed_raw_io.at(ci));
                ci += 1;
            };
            let io_commitment = core::poseidon::poseidon_hash_span(commitment_input.span());

            // Validate dimensions against packed data
            let packed_span = packed_raw_io.span();
            let in_rows_v: u32 = extract_m31_from_packed(packed_span, 0);
            let in_cols_v: u32 = extract_m31_from_packed(packed_span, 1);
            let in_len: u32 = extract_m31_from_packed(packed_span, 2);
            assert!(in_cols_v == in_cols, "IN_COLS_MISMATCH");
            assert!(in_len == in_rows_v * in_cols_v, "IN_LEN_MISMATCH");
            let out_rows_v: u32 = extract_m31_from_packed(packed_span, 3 + in_len);
            let out_cols_v: u32 = extract_m31_from_packed(packed_span, 3 + in_len + 1);
            let out_len: u32 = extract_m31_from_packed(packed_span, 3 + in_len + 2);
            assert!(out_cols_v == out_cols, "OUT_COLS_MISMATCH");
            assert!(out_len == out_rows_v * out_cols_v, "OUT_LEN_MISMATCH");
            assert!(original_io_len == 6 + in_len + out_len, "PACKED_IO_LEN_MISMATCH");

            // Seed Fiat-Shamir channel (deterministic, no MLE eval here)
            let _padded_out_cols = next_power_of_two(out_cols);
            let mut ch = channel_default();
            channel_mix_u64(ref ch, circuit_depth.into());
            channel_mix_u64(ref ch, in_rows_v.into());
            channel_mix_u64(ref ch, in_cols.into());

            // Save checkpoint state — output MLE will be evaluated in stream_init_output_mle
            self.stream_channel_digest.entry(session_id).write(ch.digest);
            self.stream_channel_counter.entry(session_id).write(ch.n_draws);

            self.stream_layers_verified.entry(session_id).write(0);
            self.stream_weight_hash.entry(session_id).write(0);
            self.stream_tags_hash.entry(session_id).write(0);
            self.stream_weight_count.entry(session_id).write(0);
            self.stream_io_commitment.entry(session_id).write(io_commitment);
            self.stream_total_layers.entry(session_id).write(num_layers);
            self.stream_circuit_depth.entry(session_id).write(circuit_depth);
            self.stream_next_batch_idx.entry(session_id).write(0);
            self.stream_deferred_count.entry(session_id).write(0);
            self.stream_original_io_len.entry(session_id).write(original_io_len);
            self.stream_packed_io_len.entry(session_id).write(packed_raw_io.len());

            // Store dimensions for output MLE + input MLE steps
            self.stream_out_cols.entry(session_id).write(out_cols);
            self.stream_in_cols.entry(session_id).write(in_cols);
            self.stream_out_data_m31_start.entry(session_id).write(3 + in_len + 3);

            self.stream_initialized.entry(session_id).write(true);
            self.stream_output_mle_done.entry(session_id).write(false);
            self.stream_finalized.entry(session_id).write(false);

            let model_id = self.session_model_id.entry(session_id).read();
            self.emit(GkrStreamStarted { session_id, model_id, num_layers });
        }

        /// Evaluate output MLE in chunked TXs using eq-table dot product.
        ///
        /// First call (chunks_done == 0): draws r-points from channel, stores them,
        /// processes first chunk.
        /// Subsequent calls: reads stored r-points, processes chunk, accumulates.
        /// Last call (is_last_chunk): mixes final sum into channel, stores initial claim.
        fn verify_gkr_stream_init_output_mle(
            ref self: ContractState,
            session_id: u64,
            packed_output_data: Array<felt252>,
            chunk_offset: u32,
            chunk_len: u32,
            is_last_chunk: bool,
        ) {
            assert!(self.stream_initialized.entry(session_id).read(), "STREAM_NOT_INITIALIZED");
            assert!(!self.stream_output_mle_done.entry(session_id).read(), "OUTPUT_MLE_ALREADY_DONE");
            let caller = get_caller_address();
            let owner = self.session_owner.entry(session_id).read();
            assert!(caller == owner, "NOT_SESSION_OWNER");

            let out_cols = self.stream_out_cols.entry(session_id).read();
            let padded_out_cols = next_power_of_two(out_cols);
            let n_vars = log2_ceil(padded_out_cols);

            let chunks_done = self.stream_output_mle_chunks_done.entry(session_id).read();

            // First chunk: draw r-points from channel and store them
            let r_out = if chunks_done == 0 {
                // Restore channel state from init
                let mut ch = PoseidonChannel {
                    digest: self.stream_channel_digest.entry(session_id).read(),
                    n_draws: self.stream_channel_counter.entry(session_id).read(),
                };

                let r_out = channel_draw_qm31s(ref ch, n_vars);

                // Store channel state after drawing (before mixing output value)
                self.stream_channel_digest.entry(session_id).write(ch.digest);
                self.stream_channel_counter.entry(session_id).write(ch.n_draws);

                // Store r-points for subsequent chunk TXs
                self.stream_output_mle_r_count.entry(session_id).write(n_vars);
                let mut ri: u32 = 0;
                loop {
                    if ri >= n_vars {
                        break;
                    }
                    self.stream_output_mle_r_point.entry((session_id, ri)).write(
                        pack_qm31_to_felt(*r_out.at(ri)),
                    );
                    ri += 1;
                };

                // Initialize partial sum to zero
                self.stream_output_mle_partial_sum.entry(session_id).write(
                    pack_qm31_to_felt(qm31_zero()),
                );

                r_out
            } else {
                // Subsequent chunks: reload stored r-points
                let stored_n = self.stream_output_mle_r_count.entry(session_id).read();
                assert!(stored_n == n_vars, "R_POINTS_MISMATCH");
                let mut r_out: Array<QM31> = array![];
                let mut ri: u32 = 0;
                loop {
                    if ri >= n_vars {
                        break;
                    }
                    r_out.append(unpack_qm31_from_felt(
                        self.stream_output_mle_r_point.entry((session_id, ri)).read(),
                    ));
                    ri += 1;
                };
                r_out
            };

            // Compute partial sum for this chunk using eq-table dot product
            let partial = evaluate_mle_eq_dot_partial(
                packed_output_data.span(),
                0,              // m31_start within this chunk's packed data
                chunk_len,
                chunk_offset,   // global offset in padded output
                padded_out_cols,
                r_out.span(),
            );

            // Accumulate into running partial sum
            let prev_sum = unpack_qm31_from_felt(
                self.stream_output_mle_partial_sum.entry(session_id).read(),
            );
            let new_sum = qm31_add(prev_sum, partial);
            self.stream_output_mle_partial_sum.entry(session_id).write(
                pack_qm31_to_felt(new_sum),
            );
            self.stream_output_mle_chunks_done.entry(session_id).write(chunks_done + 1);

            // Last chunk: finalize — mix output value into channel and store initial claim
            if is_last_chunk {
                let output_value = new_sum;

                // Restore channel state (after r-point drawing, before output mix)
                let mut ch = PoseidonChannel {
                    digest: self.stream_channel_digest.entry(session_id).read(),
                    n_draws: self.stream_channel_counter.entry(session_id).read(),
                };

                channel_mix_secure_field(ref ch, output_value);

                // Store initial claim
                let initial_claim = GKRClaim { point: r_out, value: output_value };
                self.stream_channel_digest.entry(session_id).write(ch.digest);
                self.stream_channel_counter.entry(session_id).write(ch.n_draws);
                self.stream_claim_value.entry(session_id).write(
                    pack_qm31_to_felt(initial_claim.value),
                );
                let point_len = initial_claim.point.len();
                self.stream_claim_point_len.entry(session_id).write(point_len);
                let mut pi: u32 = 0;
                loop {
                    if pi >= point_len {
                        break;
                    }
                    self.stream_claim_point.entry((session_id, pi)).write(
                        pack_qm31_to_felt(*initial_claim.point.at(pi)),
                    );
                    pi += 1;
                };

                self.stream_output_mle_done.entry(session_id).write(true);
            }
        }

        fn verify_gkr_stream_layers(
            ref self: ContractState,
            session_id: u64,
            batch_idx: u32,
            num_layers_in_batch: u32,
            matmul_dims: Array<u32>,
            dequantize_bits: Array<u64>,
            proof_data: Array<felt252>,
        ) {
            // Auth
            assert!(self.stream_initialized.entry(session_id).read(), "STREAM_NOT_INITIALIZED");
            assert!(self.stream_output_mle_done.entry(session_id).read(), "OUTPUT_MLE_NOT_DONE");
            assert!(!self.stream_finalized.entry(session_id).read(), "STREAM_ALREADY_FINALIZED");
            let caller = get_caller_address();
            let owner = self.session_owner.entry(session_id).read();
            assert!(caller == owner, "NOT_SESSION_OWNER");

            // Enforce strict sequential batch ordering
            let expected_batch_idx = self.stream_next_batch_idx.entry(session_id).read();
            assert!(batch_idx == expected_batch_idx, "STREAM_BATCH_OUT_OF_ORDER");
            self.stream_next_batch_idx.entry(session_id).write(expected_batch_idx + 1);

            let layers_so_far = self.stream_layers_verified.entry(session_id).read();
            let total_layers = self.stream_total_layers.entry(session_id).read();
            assert!(layers_so_far + num_layers_in_batch <= total_layers, "STREAM_LAYERS_OVERFLOW");

            // Restore checkpoint state
            let mut ch = PoseidonChannel {
                digest: self.stream_channel_digest.entry(session_id).read(),
                n_draws: self.stream_channel_counter.entry(session_id).read(),
            };

            let point_len = self.stream_claim_point_len.entry(session_id).read();
            let mut point: Array<QM31> = array![];
            let mut pi: u32 = 0;
            loop {
                if pi >= point_len {
                    break;
                }
                point.append(
                    unpack_qm31_from_felt(
                        self.stream_claim_point.entry((session_id, pi)).read(),
                    ),
                );
                pi += 1;
            };

            let claim_value = unpack_qm31_from_felt(
                self.stream_claim_value.entry(session_id).read(),
            );
            let initial_claim = GKRClaim { point, value: claim_value };

            let prev_weight_hash = self.stream_weight_hash.entry(session_id).read();
            let prev_tags_hash = self.stream_tags_hash.entry(session_id).read();
            let prev_weight_count = self.stream_weight_count.entry(session_id).read();

            // Run batch verification
            let result = verify_gkr_layers_batch(
                proof_data.span(),
                num_layers_in_batch,
                matmul_dims.span(),
                dequantize_bits.span(),
                initial_claim,
                ref ch,
                true, // always packed for streaming
                prev_weight_hash,
                prev_tags_hash,
                prev_weight_count,
            );

            // Save updated checkpoint state
            self.stream_channel_digest.entry(session_id).write(ch.digest);
            self.stream_channel_counter.entry(session_id).write(ch.n_draws);
            self.stream_claim_value.entry(session_id).write(
                pack_qm31_to_felt(result.next_claim.value),
            );
            let old_point_len = point_len;
            let new_point_len = result.next_claim.point.len();
            self.stream_claim_point_len.entry(session_id).write(new_point_len);
            let mut pi: u32 = 0;
            loop {
                if pi >= new_point_len {
                    break;
                }
                self.stream_claim_point.entry((session_id, pi)).write(
                    pack_qm31_to_felt(*result.next_claim.point.at(pi)),
                );
                pi += 1;
            };
            // Clear stale claim point entries if the point shrank
            loop {
                if pi >= old_point_len {
                    break;
                }
                self.stream_claim_point.entry((session_id, pi)).write(0);
                pi += 1;
            };

            self.stream_weight_hash.entry(session_id).write(result.weight_hash);
            self.stream_tags_hash.entry(session_id).write(result.tags_hash);
            self.stream_weight_count.entry(session_id).write(result.weight_count);

            // Save deferred add points from this batch
            let mut deferred_base = self.stream_deferred_count.entry(session_id).read();
            let batch_deferred = result.deferred_add_points.span();
            let mut di: u32 = 0;
            loop {
                if di >= batch_deferred.len() {
                    break;
                }
                let pt = batch_deferred.at(di);
                let pt_len = pt.len();
                self.stream_deferred_point_len.entry((session_id, deferred_base)).write(pt_len);
                let mut ci: u32 = 0;
                loop {
                    if ci >= pt_len {
                        break;
                    }
                    self.stream_deferred_point.entry((session_id, deferred_base, ci)).write(
                        pack_qm31_to_felt(*pt.at(ci)),
                    );
                    ci += 1;
                };
                deferred_base += 1;
                di += 1;
            };
            self.stream_deferred_count.entry(session_id).write(deferred_base);

            let new_layers_verified = layers_so_far + num_layers_in_batch;
            self.stream_layers_verified.entry(session_id).write(new_layers_verified);

            self.emit(GkrStreamProgress {
                session_id,
                layers_verified: new_layers_verified,
                total_layers,
            });
        }

        fn verify_gkr_stream_finalize_input_mle(
            ref self: ContractState,
            session_id: u64,
            packed_input_data: Array<felt252>,
            chunk_offset: u32,
            chunk_len: u32,
            is_last_chunk: bool,
            weight_expected_values: Array<felt252>,
            weight_binding_mode: u32,
            weight_binding_data: Array<felt252>,
            deferred_proof_data: Array<felt252>,
            deferred_matmul_dims: Array<u32>,
            original_io_len: u32,
            packed_raw_io: Array<felt252>,
        ) {
            assert!(self.stream_initialized.entry(session_id).read(), "STREAM_NOT_INITIALIZED");
            assert!(!self.stream_finalized.entry(session_id).read(), "STREAM_ALREADY_FINALIZED");
            assert!(!self.stream_input_mle_done.entry(session_id).read(), "INPUT_MLE_ALREADY_DONE");
            let caller = get_caller_address();
            let owner = self.session_owner.entry(session_id).read();
            assert!(caller == owner, "NOT_SESSION_OWNER");

            let layers_verified = self.stream_layers_verified.entry(session_id).read();
            let total_layers = self.stream_total_layers.entry(session_id).read();
            assert!(layers_verified == total_layers, "STREAM_LAYERS_INCOMPLETE");

            let chunks_done = self.stream_input_mle_chunks_done.entry(session_id).read();

            // First chunk: weight binding + IO commitment + store r-points
            let r_in = if chunks_done == 0 {
                assert!(!self.stream_weight_binding_done.entry(session_id).read(), "WEIGHT_BINDING_ALREADY_DONE");

                let model_id = self.session_model_id.entry(session_id).read();

                // Restore checkpoint state
                let mut ch = PoseidonChannel {
                    digest: self.stream_channel_digest.entry(session_id).read(),
                    n_draws: self.stream_channel_counter.entry(session_id).read(),
                };

                let stream_weight_hash = self.stream_weight_hash.entry(session_id).read();
                let stream_tags_hash = self.stream_tags_hash.entry(session_id).read();
                let stream_weight_count = self.stream_weight_count.entry(session_id).read();

                // ── Circuit binding ──
                let circuit_depth = self.stream_circuit_depth.entry(session_id).read();
                let registered_circuit_hash = self.model_circuit_hash.entry(model_id).read();
                let observed_circuit_hash = core::poseidon::poseidon_hash_span(
                    array![circuit_depth.into(), stream_tags_hash].span(),
                );
                if registered_circuit_hash != 0 {
                    assert!(
                        observed_circuit_hash == registered_circuit_hash,
                        "STREAM_CIRCUIT_HASH_MISMATCH",
                    );
                }

                // ── Weight hash verification ──
                let expected_weight_count = weight_expected_values.len();
                assert!(expected_weight_count == stream_weight_count, "STREAM_WEIGHT_COUNT_MISMATCH");

                let mut recomputed_weight_hash: felt252 = 0;
                let mut wi: u32 = 0;
                loop {
                    if wi >= expected_weight_count {
                        break;
                    }
                    recomputed_weight_hash = core::poseidon::poseidon_hash_span(
                        array![recomputed_weight_hash, *weight_expected_values.at(wi)].span(),
                    );
                    wi += 1;
                };
                assert!(recomputed_weight_hash == stream_weight_hash, "STREAM_WEIGHT_HASH_MISMATCH");

                // ── Deferred proof verification ──
                let deferred_count = self.stream_deferred_count.entry(session_id).read();
                let deferred_proof_span = deferred_proof_data.span();
                let deferred_dims_span = deferred_matmul_dims.span();

                let mut deferred_reader = mv_reader_new(deferred_proof_span, true);
                let num_deferred = if deferred_proof_span.len() > 0 {
                    mv_read_u32(ref deferred_reader)
                } else {
                    0_u32
                };
                assert!(num_deferred <= deferred_count, "DEFERRED_COUNT_EXCEEDS_ADDS");

                let mut deferred_weight_evs: Array<felt252> = array![];
                let mut deferred_dims_idx: u32 = 0;
                let mut def_idx: u32 = 0;
                loop {
                    if def_idx >= num_deferred {
                        break;
                    }
                    let claim_value = mv_read_qm31(ref deferred_reader);
                    let pt_len = self.stream_deferred_point_len.entry((session_id, def_idx)).read();
                    let mut deferred_point: Array<QM31> = array![];
                    let mut ci: u32 = 0;
                    loop {
                        if ci >= pt_len {
                            break;
                        }
                        deferred_point.append(
                            unpack_qm31_from_felt(
                                self.stream_deferred_point.entry((session_id, def_idx, ci)).read(),
                            ),
                        );
                        ci += 1;
                    };
                    channel_mix_secure_field(ref ch, claim_value);
                    let dims_base = deferred_dims_idx * 3;
                    assert!(dims_base + 2 < deferred_dims_span.len(), "DEFERRED_DIMS_UNDERRUN");
                    let m = *deferred_dims_span.at(dims_base);
                    let k = *deferred_dims_span.at(dims_base + 1);
                    let n = *deferred_dims_span.at(dims_base + 2);
                    deferred_dims_idx += 1;
                    let deferred_claim = GKRClaim { point: deferred_point, value: claim_value };
                    let (_new_claim, final_b_eval) = mv_dispatch_matmul(
                        @deferred_claim, m, k, n, ref deferred_reader, ref ch,
                    );
                    deferred_weight_evs.append(pack_qm31_to_felt(final_b_eval));
                    def_idx += 1;
                };

                // ── Weight commitment verification ──
                let registered_weight_count = self.model_gkr_weight_count.entry(model_id).read();
                let total_weight_count = stream_weight_count + num_deferred;
                assert!(
                    total_weight_count == registered_weight_count,
                    "STREAM_WEIGHT_COUNT_VS_REGISTERED",
                );

                // ── Weight binding (RLC mode 4) ──
                assert!(
                    weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK,
                    "STREAM_ONLY_SUPPORTS_RLC_MODE",
                );
                let weight_binding_span = weight_binding_data.span();
                assert!(weight_binding_span.len() == 2, "STREAM_BINDING_DATA_LEN");
                assert!(*weight_binding_span.at(0) == WEIGHT_BINDING_RLC_MARKER, "STREAM_BINDING_NOT_RLC");

                let rho = channel_draw_qm31(ref ch);
                let mut rho_pow = qm31_one();
                let mut combined = qm31_zero();
                let mut claim_i: u32 = 0;
                loop {
                    if claim_i >= expected_weight_count {
                        break;
                    }
                    let ev = unpack_qm31_from_felt(*weight_expected_values.at(claim_i));
                    combined = qm31_add(combined, qm31_mul(rho_pow, ev));
                    rho_pow = qm31_mul(rho_pow, rho);
                    claim_i += 1;
                };
                let deferred_evs_span = deferred_weight_evs.span();
                let mut def_wi: u32 = 0;
                loop {
                    if def_wi >= deferred_evs_span.len() {
                        break;
                    }
                    let ev = unpack_qm31_from_felt(*deferred_evs_span.at(def_wi));
                    combined = qm31_add(combined, qm31_mul(rho_pow, ev));
                    rho_pow = qm31_mul(rho_pow, rho);
                    def_wi += 1;
                };
                channel_mix_secure_field(ref ch, combined);

                // ── IO commitment verification ──
                let stored_original_io_len = self.stream_original_io_len.entry(session_id).read();
                let stored_packed_io_len = self.stream_packed_io_len.entry(session_id).read();
                assert!(original_io_len == stored_original_io_len, "FINALIZE_IO_LEN_MISMATCH");
                assert!(packed_raw_io.len() == stored_packed_io_len, "FINALIZE_PACKED_IO_LEN_MISMATCH");
                let packed_span = packed_raw_io.span();

                let stored_io_commitment = self.stream_io_commitment.entry(session_id).read();
                let mut commitment_input: Array<felt252> = array![original_io_len.into()];
                let mut ci2: u32 = 0;
                loop {
                    if ci2 >= stored_packed_io_len {
                        break;
                    }
                    commitment_input.append(*packed_span.at(ci2));
                    ci2 += 1;
                };
                let recomputed_io_commitment = core::poseidon::poseidon_hash_span(
                    commitment_input.span(),
                );
                assert!(
                    recomputed_io_commitment == stored_io_commitment,
                    "STREAM_IO_COMMITMENT_MISMATCH",
                );

                // Save channel state after weight binding (before input MLE assertion)
                self.stream_channel_digest.entry(session_id).write(ch.digest);
                self.stream_channel_counter.entry(session_id).write(ch.n_draws);
                self.stream_weight_binding_done.entry(session_id).write(true);

                // Extract input dimensions and store r-points from final_claim.point
                let in_cols: u32 = extract_m31_from_packed(packed_span, 1);
                let padded_in_cols = next_power_of_two(in_cols);
                let n_vars = log2_ceil(padded_in_cols);

                // r-points for input MLE = final_claim.point (from the GKR walk)
                let point_len = self.stream_claim_point_len.entry(session_id).read();
                let mut r_in: Array<QM31> = array![];
                let mut pi: u32 = 0;
                loop {
                    if pi >= point_len {
                        break;
                    }
                    let rp = unpack_qm31_from_felt(
                        self.stream_claim_point.entry((session_id, pi)).read(),
                    );
                    r_in.append(rp);
                    pi += 1;
                };

                // Store r-points for subsequent chunk TXs
                self.stream_input_mle_r_count.entry(session_id).write(n_vars);
                let mut ri: u32 = 0;
                loop {
                    if ri >= n_vars {
                        break;
                    }
                    self.stream_input_mle_r_point.entry((session_id, ri)).write(
                        pack_qm31_to_felt(*r_in.at(ri)),
                    );
                    ri += 1;
                };

                // Initialize partial sum to zero
                self.stream_input_mle_partial_sum.entry(session_id).write(
                    pack_qm31_to_felt(qm31_zero()),
                );

                r_in
            } else {
                // Subsequent chunks: reload stored r-points
                let stored_n = self.stream_input_mle_r_count.entry(session_id).read();
                let mut r_in: Array<QM31> = array![];
                let mut ri: u32 = 0;
                loop {
                    if ri >= stored_n {
                        break;
                    }
                    r_in.append(unpack_qm31_from_felt(
                        self.stream_input_mle_r_point.entry((session_id, ri)).read(),
                    ));
                    ri += 1;
                };
                r_in
            };

            // Compute partial sum for this chunk
            // in_cols is stored during stream_init — packed_input_data contains
            // only the chunk's M31 values (no raw IO header).
            let in_cols = self.stream_in_cols.entry(session_id).read();
            let padded_in_cols = next_power_of_two(in_cols);

            let partial = evaluate_mle_eq_dot_partial(
                packed_input_data.span(),
                0,
                chunk_len,
                chunk_offset,
                padded_in_cols,
                r_in.span(),
            );

            // Accumulate
            let prev_sum = unpack_qm31_from_felt(
                self.stream_input_mle_partial_sum.entry(session_id).read(),
            );
            let new_sum = qm31_add(prev_sum, partial);
            self.stream_input_mle_partial_sum.entry(session_id).write(
                pack_qm31_to_felt(new_sum),
            );
            self.stream_input_mle_chunks_done.entry(session_id).write(chunks_done + 1);

            if is_last_chunk {
                self.stream_input_mle_value.entry(session_id).write(
                    pack_qm31_to_felt(new_sum),
                );
                self.stream_input_mle_done.entry(session_id).write(true);
            }
        }

        fn verify_gkr_stream_finalize(
            ref self: ContractState,
            session_id: u64,
        ) -> bool {
            assert!(self.stream_initialized.entry(session_id).read(), "STREAM_NOT_INITIALIZED");
            assert!(!self.stream_finalized.entry(session_id).read(), "STREAM_ALREADY_FINALIZED");
            assert!(self.stream_input_mle_done.entry(session_id).read(), "INPUT_MLE_NOT_DONE");
            assert!(self.stream_weight_binding_done.entry(session_id).read(), "WEIGHT_BINDING_NOT_DONE");
            let caller = get_caller_address();
            let owner = self.session_owner.entry(session_id).read();
            assert!(caller == owner, "NOT_SESSION_OWNER");

            let model_id = self.session_model_id.entry(session_id).read();

            // Read pre-computed input MLE value
            let input_value = unpack_qm31_from_felt(
                self.stream_input_mle_value.entry(session_id).read(),
            );

            // Read final claim from GKR walk
            let final_value = unpack_qm31_from_felt(
                self.stream_claim_value.entry(session_id).read(),
            );

            // Verify input MLE matches final GKR claim
            assert!(
                crate::field::qm31_eq(input_value, final_value),
                "STREAM_INPUT_CLAIM_MISMATCH",
            );

            // Compute proof hash using channel state after weight binding
            let ch_digest = self.stream_channel_digest.entry(session_id).read();
            let stored_io_commitment = self.stream_io_commitment.entry(session_id).read();
            let num_layers = self.stream_total_layers.entry(session_id).read();
            let proof_hash = core::poseidon::poseidon_hash_span(
                array![ch_digest, stored_io_commitment, model_id, num_layers.into()].span(),
            );

            // Record proof on-chain
            assert!(!self.verified_proofs.entry(proof_hash).read(), "PROOF_ALREADY_VERIFIED");
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);
            self.emit(ModelGkrVerified {
                model_id,
                proof_hash,
                io_commitment: stored_io_commitment,
                num_layers,
            });

            self.stream_finalized.entry(session_id).write(true);
            self.emit(GkrStreamFinalized { session_id, proof_hash });
            true
        }
    }

    // ─── Audit/Access/ViewKey impls stripped for lean v18b deploy ──────────
    // Will be restored in next version. Storage is preserved across upgrades.
    // See git history for full audit/access-control/view-key implementations.

    // ─── Private core verification logic (shared by record + view) ────────

    #[generate_trait]
    impl InternalImpl of InternalTrait {
        /// Core GKR verification: runs full Fiat-Shamir transcript replay and
        /// returns (proof_hash, io_commitment). Panics on verification failure.
        /// Shared by verify_model_gkr_v4_packed_io (records proof) and
        /// verify_model_gkr_v4_packed_io_view (view-only, no storage writes).
        fn verify_model_gkr_v4_packed_io_core(
            self: @ContractState,
            model_id: felt252,
            original_io_len: u32,
            packed_raw_io: Array<felt252>,
            circuit_depth: u32,
            num_layers: u32,
            matmul_dims: Array<u32>,
            dequantize_bits: Array<u64>,
            proof_data: Array<felt252>,
            weight_commitments: Array<felt252>,
            weight_binding_mode: u32,
            weight_binding_data: Array<felt252>,
            weight_opening_proofs: Array<MleOpeningProof>,
        ) -> (felt252, felt252) {
            Self::verify_model_gkr_v4_packed_io_core_dp(
                self, model_id, original_io_len, packed_raw_io, circuit_depth,
                num_layers, matmul_dims, dequantize_bits, proof_data,
                weight_commitments, weight_binding_mode, weight_binding_data,
                weight_opening_proofs, false,
            )
        }

        /// Core GKR verification with optional double-packed proof data.
        /// When `double_packed` is true, degree-2 round polys read (c0,c2) from
        /// one felt252 and degree-3 polys read (c0,c2) paired + c3 single.
        fn verify_model_gkr_v4_packed_io_core_dp(
            self: @ContractState,
            model_id: felt252,
            original_io_len: u32,
            packed_raw_io: Array<felt252>,
            circuit_depth: u32,
            num_layers: u32,
            matmul_dims: Array<u32>,
            dequantize_bits: Array<u64>,
            proof_data: Array<felt252>,
            weight_commitments: Array<felt252>,
            weight_binding_mode: u32,
            weight_binding_data: Array<felt252>,
            weight_opening_proofs: Array<MleOpeningProof>,
            double_packed: bool,
        ) -> (felt252, felt252) {
            assert!(
                weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_OPENINGS_V4_EXPERIMENTAL
                    || weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK,
                "UNSUPPORTED_WEIGHT_BINDING_MODE",
            );

            // Hash packed felts for IO commitment (1,281 felts vs 10,246)
            let mut commitment_input: Array<felt252> = array![original_io_len.into()];
            let mut ci: u32 = 0;
            loop {
                if ci >= packed_raw_io.len() {
                    break;
                }
                commitment_input.append(*packed_raw_io.at(ci));
                ci += 1;
            };
            let io_commitment = core::poseidon::poseidon_hash_span(commitment_input.span());

            // Extract IO dimensions from packed raw_io data.
            // raw_io layout: [in_rows, in_cols, in_len, in_data..., out_rows, out_cols, out_len, out_data...]
            // Each value is an M31 packed 8 per felt252.
            let packed_span = packed_raw_io.span();
            let in_rows: u32 = extract_m31_from_packed(packed_span, 0);
            let in_cols: u32 = extract_m31_from_packed(packed_span, 1);
            let in_len: u32 = extract_m31_from_packed(packed_span, 2);
            assert!(in_len == in_rows * in_cols, "IN_LEN_MISMATCH");
            let out_rows: u32 = extract_m31_from_packed(packed_span, 3 + in_len);
            let out_cols: u32 = extract_m31_from_packed(packed_span, 3 + in_len + 1);
            let out_len: u32 = extract_m31_from_packed(packed_span, 3 + in_len + 2);
            assert!(out_len == out_rows * out_cols, "OUT_LEN_MISMATCH");

            assert!(original_io_len == 6 + in_len + out_len, "PACKED_IO_LEN_MISMATCH");

            let in_data_m31_start: u32 = 3;
            let out_data_m31_start: u32 = 3 + in_len + 3;

            let padded_in_rows = next_power_of_two(in_rows);
            let padded_in_cols = next_power_of_two(in_cols);
            let padded_out_rows = next_power_of_two(out_rows);
            let padded_out_cols = next_power_of_two(out_cols);

            // Fiat-Shamir channel
            let mut ch = channel_default();
            channel_mix_u64(ref ch, circuit_depth.into());
            channel_mix_u64(ref ch, in_rows.into());
            channel_mix_u64(ref ch, in_cols.into());

            let log_out_rows = log2_ceil(padded_out_rows);
            let log_out_cols = log2_ceil(padded_out_cols);
            let log_out_total = log_out_rows + log_out_cols;
            let r_out = channel_draw_qm31s(ref ch, log_out_total);

            // OUTPUT MLE evaluation directly from packed data
            assert!(padded_out_rows == 1, "PACKED_IO_ONLY_1ROW");
            let output_value = evaluate_mle_from_packed_1row(
                packed_span, out_data_m31_start, out_cols, padded_out_cols, r_out.span(),
            );

            channel_mix_secure_field(ref ch, output_value);

            // GKR model walk
            let initial_claim = GKRClaim { point: r_out, value: output_value };
            let (final_claim, weight_claims, layer_tags, deferred_weight_commitments) =
                verify_gkr_model_with_trace_dp(
                    proof_data.span(), num_layers, matmul_dims.span(),
                    dequantize_bits.span(), initial_claim, ref ch, true, double_packed,
                );

            // Circuit binding
            let circuit_hash = self.model_circuit_hash.entry(model_id).read();
            assert!(circuit_hash != 0, "Model not registered for GKR");
            let mut descriptor_felts: Array<felt252> = array![circuit_depth.into()];
            let mut t_i: u32 = 0;
            loop {
                if t_i >= layer_tags.len() {
                    break;
                }
                descriptor_felts.append((*layer_tags.at(t_i)).into());
                t_i += 1;
            };
            let observed_circuit_hash = core::poseidon::poseidon_hash_span(descriptor_felts.span());
            assert!(observed_circuit_hash == circuit_hash, "CIRCUIT_HASH_MISMATCH");

            // Weight binding — aggregate hash comparison (1 storage read vs N)
            let registered_count = self.model_gkr_weight_count.entry(model_id).read();
            assert!(weight_commitments.len() == registered_count, "Weight commitment count mismatch");
            let mut weight_hash_input: Array<felt252> = array![];
            let mut w_idx: u32 = 0;
            loop {
                if w_idx >= registered_count {
                    break;
                }
                weight_hash_input.append(*weight_commitments.at(w_idx));
                w_idx += 1;
            };
            let calldata_weight_hash = core::poseidon::poseidon_hash_span(weight_hash_input.span());
            let registered_hash = self.model_weight_root_hash.entry(model_id).read();
            assert!(calldata_weight_hash == registered_hash, "WEIGHT_ROOT_HASH_MISMATCH");

            let expected_weight_claims = registered_count + deferred_weight_commitments.len();
            assert!(weight_claims.len() == expected_weight_claims, "WEIGHT_CLAIM_COUNT_MISMATCH");
            assert!(weight_binding_data.len() >= 2, "AGGREGATED_BINDING_DATA_TOO_SHORT");

            let weight_binding_span = weight_binding_data.span();

            if weight_binding_mode == WEIGHT_BINDING_MODE_AGGREGATED_ORACLE_SUMCHECK {
                if weight_binding_span.len() == 2
                    && *weight_binding_span.at(0) == WEIGHT_BINDING_RLC_MARKER {
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
                            combined, crate::field::qm31_mul(rho_pow, *claim.expected_value),
                        );
                        rho_pow = crate::field::qm31_mul(rho_pow, rho);
                        claim_i += 1;
                    };
                    channel_mix_secure_field(ref ch, combined);
                } else {
                    panic!("FULL_AGGREGATED_BINDING_NOT_IN_LEAN_BUILD");
                }
            } else {
                panic!("MODE_3_NOT_IN_LEAN_BUILD");
            }

            // INPUT MLE evaluation directly from packed data
            assert!(padded_in_rows == 1, "PACKED_IO_ONLY_1ROW");
            let input_value = evaluate_mle_from_packed_1row(
                packed_span, in_data_m31_start, in_cols, padded_in_cols,
                final_claim.point.span(),
            );
            assert!(crate::field::qm31_eq(input_value, final_claim.value), "INPUT_CLAIM_MISMATCH");

            // Return proof hash + io_commitment for caller to record (or discard for view)
            let proof_hash = core::poseidon::poseidon_hash_span(
                array![ch.digest, io_commitment, model_id, num_layers.into()].span(),
            );
            (proof_hash, io_commitment)
        }
    }
}
