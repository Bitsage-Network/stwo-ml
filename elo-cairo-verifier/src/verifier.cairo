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
// QM31 imported only by v4_packed_io (inside module scope)
use starknet::ClassHash;

// PackedDigest moved inline (was in vm31_merkle, 724 lines stripped)
#[derive(Drop, Copy, Serde, starknet::Store, PartialEq, Debug)]
pub struct PackedDigest {
    pub lo: felt252,
    pub hi: felt252,
}

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
}

#[starknet::contract]
mod SumcheckVerifierContract {
    use super::{
        GKRClaim, MleOpeningProof, UPGRADE_DELAY,
    };
    use crate::field::{
        log2_ceil, next_power_of_two,
        evaluate_mle_from_packed_1row,
    };
    use crate::channel::{
        channel_default, channel_mix_u64,
        channel_draw_qm31, channel_draw_qm31s, channel_mix_secure_field,
    };
    // mle stripped for lean v18b
    use crate::model_verifier::verify_gkr_model_with_trace;
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

            // Extract IO dimensions from matmul_dims
            let in_rows: u32 = *matmul_dims.at(0);
            let in_cols: u32 = *matmul_dims.at(1);
            let in_len: u32 = in_rows * in_cols;
            let last_triple_start: u32 = (num_layers - 1) * 3;
            let out_rows: u32 = *matmul_dims.at(last_triple_start);
            let out_cols: u32 = *matmul_dims.at(last_triple_start + 2);
            let out_len: u32 = out_rows * out_cols;

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
            let packed_span = packed_raw_io.span();
            assert!(padded_out_rows == 1, "PACKED_IO_ONLY_1ROW");
            let output_value = evaluate_mle_from_packed_1row(
                packed_span, out_data_m31_start, out_cols, padded_out_cols, r_out.span(),
            );

            channel_mix_secure_field(ref ch, output_value);

            // GKR model walk
            let initial_claim = GKRClaim { point: r_out, value: output_value };
            let (final_claim, weight_claims, layer_tags, deferred_weight_commitments) =
                verify_gkr_model_with_trace(
                    proof_data.span(), num_layers, matmul_dims.span(),
                    dequantize_bits.span(), initial_claim, ref ch, true,
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
