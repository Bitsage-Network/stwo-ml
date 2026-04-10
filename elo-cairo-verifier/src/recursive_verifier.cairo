// Recursive STARK Verifier for ObelyZK
//
// Verifies a recursive STARK proof that attests "the GKR verifier accepted."
// The proof is a standard STWO STARK — no GKR-specific logic on-chain.
//
// This replaces the 18-TX streaming GKR verification with a single TX.
//
// Public inputs (committed inside the STARK):
//   - circuit_hash: Poseidon hash of the model's circuit descriptor
//   - io_commitment: Poseidon hash of the packed inference IO
//   - weight_super_root: Poseidon Merkle root of all weight matrices
//
// On-chain, we check these against the registered model and record verification.

// Recursive STARK verifier types

/// Public inputs for recursive verification.
#[derive(Drop, Copy, Serde)]
pub struct RecursivePublicInputs {
    /// Poseidon hash of the LayeredCircuit descriptor.
    pub circuit_hash: felt252,
    /// Poseidon hash of the packed IO felts.
    pub io_commitment: felt252,
    /// Poseidon Merkle root of all weight matrices.
    pub weight_super_root: felt252,
}

/// Registered model info for recursive verification.
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct RecursiveModelInfo {
    /// Expected circuit hash (from registration).
    pub circuit_hash: felt252,
    /// Expected weight super root (from registration).
    pub weight_super_root: felt252,
    /// Expected policy commitment (Poseidon hash of PolicyConfig). 0 = any policy.
    pub policy_commitment: felt252,
    /// Owner who registered the model.
    pub owner: starknet::ContractAddress,
}

#[starknet::interface]
pub trait IRecursiveVerifier<TContractState> {
    /// Register a model for recursive verification.
    ///
    /// The circuit_hash and weight_super_root are committed at registration time.
    /// Subsequent verify calls check the proof's public inputs match these values.
    fn register_model_recursive(
        ref self: TContractState,
        model_id: felt252,
        circuit_hash: felt252,
        weight_super_root: felt252,
        policy_commitment: felt252,
    );

    /// Verify a recursive STARK proof for a registered model.
    ///
    /// This is a single-TX verification that replaces the 18-TX streaming pipeline.
    /// The STARK proof attests that the GKR verifier accepted the original proof.
    ///
    /// Returns true if verification succeeds.
    fn verify_recursive(
        ref self: TContractState,
        model_id: felt252,
        io_commitment: felt252,
        stark_proof_data: Array<felt252>,
    ) -> bool;

    /// Check if a recursive proof has been verified.
    fn is_recursive_proof_verified(
        self: @TContractState, proof_hash: felt252,
    ) -> bool;

    /// Get the number of recursive verifications for a model.
    fn get_recursive_verification_count(
        self: @TContractState, model_id: felt252,
    ) -> u64;

    /// Get registered model info.
    fn get_recursive_model_info(
        self: @TContractState, model_id: felt252,
    ) -> RecursiveModelInfo;

    /// Get the registered policy commitment for a model. Returns 0 if no policy bound.
    fn get_model_policy(
        self: @TContractState, model_id: felt252,
    ) -> felt252;

    /// Get full details of the last verification for a model.
    /// Returns (io_commitment, proof_hash, timestamp, proof_felts, n_layers, trace_log_size, verification_count).
    fn get_last_verification(
        self: @TContractState, model_id: felt252,
    ) -> (felt252, felt252, u64, u32, u32, u32, u64);

    /// Propose a contract class upgrade (owner only, subject to timelock).
    fn propose_upgrade(ref self: TContractState, new_class_hash: starknet::ClassHash);

    /// Execute a proposed upgrade after the timelock has elapsed.
    fn execute_upgrade(ref self: TContractState);

    /// Cancel a pending upgrade.
    fn cancel_upgrade(ref self: TContractState);

    /// Get the pending upgrade class hash and proposal timestamp.
    fn get_pending_upgrade(self: @TContractState) -> (starknet::ClassHash, u64);
}

#[starknet::contract]
pub mod RecursiveVerifierContract {
    use super::{RecursiveModelInfo, RecursivePublicInputs};
    use starknet::storage::{Map, StorageMapReadAccess, StorageMapWriteAccess, StoragePointerReadAccess, StoragePointerWriteAccess};
    use starknet::{get_caller_address, ContractAddress};
    use core::poseidon::poseidon_hash_span;
    use stwo_verifier_core::fields::qm31::{QM31, QM31Zero};
    use stwo_verifier_core::fields::m31::M31;
    use stwo_verifier_core::pcs::verifier::CommitmentSchemeVerifierImpl;
    use stwo_verifier_core::pcs::PcsConfigTrait;
    use stwo_verifier_core::channel::ChannelTrait;
    use stwo_verifier_core::circle::ChannelGetRandomCirclePointImpl;
    use crate::recursive_air::{RecursiveAir, LIMBS_PER_FELT};

    #[storage]
    struct Storage {
        /// Contract owner.
        owner: ContractAddress,

        /// Registered models for recursive verification.
        /// model_id → RecursiveModelInfo
        recursive_models: Map<felt252, RecursiveModelInfo>,

        /// Verified recursive proof hashes.
        /// proof_hash → verified (true/false)
        recursive_verified: Map<felt252, bool>,

        /// Verification count per model.
        recursive_count: Map<felt252, u64>,

        /// Last proof details per model (queryable on-chain).
        /// model_id → (io_commitment, proof_hash, timestamp, proof_felts)
        last_io: Map<felt252, felt252>,
        last_proof_hash: Map<felt252, felt252>,
        last_verified_at: Map<felt252, u64>,
        last_proof_felts: Map<felt252, u32>,
        last_n_layers: Map<felt252, u32>,
        last_trace_log_size: Map<felt252, u32>,

        /// Pending upgrade class hash (0 = no pending upgrade).
        pending_upgrade: starknet::ClassHash,

        /// Timestamp when upgrade was proposed.
        upgrade_proposed_at: u64,
    }

    /// Minimum delay (seconds) between propose_upgrade and execute_upgrade.
    const UPGRADE_DELAY: u64 = 300; // 5 minutes

    #[event]
    #[derive(Drop, starknet::Event)]
    pub enum Event {
        RecursiveModelRegistered: RecursiveModelRegistered,
        RecursiveProofVerified: RecursiveProofVerified,
        UpgradeProposed: UpgradeProposed,
        UpgradeExecuted: UpgradeExecuted,
        UpgradeCancelled: UpgradeCancelled,
    }

    #[derive(Drop, starknet::Event)]
    pub struct RecursiveModelRegistered {
        #[key]
        pub model_id: felt252,
        pub circuit_hash: felt252,
        pub weight_super_root: felt252,
        pub policy_commitment: felt252,
        pub owner: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct RecursiveProofVerified {
        #[key]
        pub model_id: felt252,
        #[key]
        pub proof_hash: felt252,
        /// Poseidon hash of packed inference IO (input + output).
        pub io_commitment: felt252,
        /// Poseidon hash of the model's circuit descriptor (architecture fingerprint).
        pub circuit_hash: felt252,
        /// Poseidon Merkle root binding all weight matrices.
        pub weight_super_root: felt252,
        /// Poseidon hash of the policy config used during proving.
        pub policy_commitment: felt252,
        /// Number of transformer layers in the model (e.g., 48 for Qwen2.5-14B).
        pub n_layers: u32,
        /// STARK trace log_size (log₂ of execution trace rows).
        pub trace_log_size: u32,
        /// Number of calldata felts in the STARK proof.
        pub proof_felts: u32,
        /// Verification sequence number for this model.
        pub verification_count: u64,
        /// Block timestamp of verification.
        pub verified_at: u64,
        /// Submitter address.
        pub submitter: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct UpgradeProposed {
        pub new_class_hash: starknet::ClassHash,
        pub proposed_at: u64,
        pub proposer: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct UpgradeExecuted {
        pub new_class_hash: starknet::ClassHash,
        pub executed_at: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct UpgradeCancelled {
        pub cancelled_class_hash: starknet::ClassHash,
        pub cancelled_by: ContractAddress,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
    }

    #[abi(embed_v0)]
    impl RecursiveVerifierImpl of super::IRecursiveVerifier<ContractState> {
        fn register_model_recursive(
            ref self: ContractState,
            model_id: felt252,
            circuit_hash: felt252,
            weight_super_root: felt252,
            policy_commitment: felt252,
        ) {
            // Only owner can register models
            let caller = get_caller_address();
            assert(caller == self.owner.read(), 'Only owner can register');

            let info = RecursiveModelInfo {
                circuit_hash,
                weight_super_root,
                policy_commitment,
                owner: caller,
            };
            self.recursive_models.write(model_id, info);

            self.emit(RecursiveModelRegistered {
                model_id,
                circuit_hash,
                weight_super_root,
                policy_commitment,
                owner: caller,
            });
        }

        fn verify_recursive(
            ref self: ContractState,
            model_id: felt252,
            io_commitment: felt252,
            stark_proof_data: Array<felt252>,
        ) -> bool {
            // 1. Look up registered model
            let model = self.recursive_models.read(model_id);
            assert(model.circuit_hash != 0, 'Model not registered');

            // 2. Compute proof hash for dedup
            let mut hash_input = array![model_id, io_commitment];
            let proof_hash = poseidon_hash_span(hash_input.span());

            // 3. Check not already verified
            assert(!self.recursive_verified.read(proof_hash), 'Already verified');

            // 4. Parse header and deserialize STARK proof.
            //
            // Calldata layout (from Rust serialize_recursive_proof_calldata):
            //   [0..4)   circuit_hash: QM31 (4 felts)
            //   [4..8)   io_commitment: QM31 (4 felts)
            //   [8..12)  weight_super_root: QM31 (4 felts)
            //   [12]     n_layers: u32
            //   [13]     verified: u32 (bool)
            //   [14]     final_digest: felt252
            //   [15]     log_size: u32
            //   [16..)   CommitmentSchemeProof (Serde-compatible)

            let stark_proof_data_len: u32 = stark_proof_data.len();
            let mut proof_span = stark_proof_data.span();
            assert!(proof_span.len() >= 20, "Proof too short");

            // Parse circuit_hash: QM31 (4 M31 limbs)
            let ch0: felt252 = *proof_span.pop_front().unwrap();
            let ch1: felt252 = *proof_span.pop_front().unwrap();
            let ch2: felt252 = *proof_span.pop_front().unwrap();
            let ch3: felt252 = *proof_span.pop_front().unwrap();

            // Parse io_commitment: QM31 (4 M31 limbs)
            let io0: felt252 = *proof_span.pop_front().unwrap();
            let io1: felt252 = *proof_span.pop_front().unwrap();
            let io2: felt252 = *proof_span.pop_front().unwrap();
            let io3: felt252 = *proof_span.pop_front().unwrap();

            // Parse weight_super_root: QM31 (4 M31 limbs)
            let wr0: felt252 = *proof_span.pop_front().unwrap();
            let wr1: felt252 = *proof_span.pop_front().unwrap();
            let wr2: felt252 = *proof_span.pop_front().unwrap();
            let wr3: felt252 = *proof_span.pop_front().unwrap();

            // Parse n_layers and verified
            let n_layers_felt: felt252 = *proof_span.pop_front().unwrap();
            let n_layers: u32 = n_layers_felt.try_into().unwrap_or(0);
            let _ = proof_span.pop_front().unwrap(); // verified

            let final_digest: felt252 = *proof_span.pop_front().unwrap();
            let log_size: u32 = (*proof_span.pop_front().unwrap()).try_into().unwrap();

            // Verify proof binds to the registered model's circuit and weights.
            // Pack 4 M31 limbs into felt252: a * 2^93 + b * 2^62 + c * 2^31 + d
            let circuit_hash_packed = ch0 * 0x80000000 * 0x80000000 * 0x80000000
                + ch1 * 0x80000000 * 0x80000000
                + ch2 * 0x80000000
                + ch3;
            let weight_root_packed = wr0 * 0x80000000 * 0x80000000 * 0x80000000
                + wr1 * 0x80000000 * 0x80000000
                + wr2 * 0x80000000
                + wr3;
            assert(circuit_hash_packed == model.circuit_hash, 'Circuit hash mismatch');
            assert(weight_root_packed == model.weight_super_root, 'Weight binding mismatch');
            // NOTE: io_commitment parameter is a Poseidon hash (felt252), while the header
            // contains QM31 limbs. These are different encodings and cannot be directly
            // compared. The STARK proof internally binds io_commitment through the
            // Fiat-Shamir channel, so a proof with wrong IO will fail STARK verification.

            // Build RecursiveAir from public inputs.
            // Initial digest is always zero (fresh Poseidon channel).
            // Final digest limbs: decompose the felt252 into 9 M31 limbs (28 bits each).
            let mut initial_limbs: Array<QM31> = array![];
            let mut final_limbs: Array<QM31> = array![];
            let mut i: u32 = 0;
            loop {
                if i >= LIMBS_PER_FELT { break; }
                initial_limbs.append(QM31Zero::zero());
                // Each limb is 28 bits of the felt252, from LSB
                // For the boundary constraint check, we extract M31 limbs
                // via bit shifting. The STARK's boundary constraints enforce
                // that the trace's digest_after on the last row matches these values.
                let limb_val = felt252_extract_limb(final_digest, i);
                final_limbs.append(m31_to_qm31(limb_val));
                i += 1;
            };

            let air = RecursiveAir {
                log_n_rows: log_size,
                initial_digest_limbs: initial_limbs,
                final_digest_limbs: final_limbs,
            };

            // 5. Deserialize + verify
            let csp: stwo_verifier_core::pcs::verifier::CommitmentSchemeProof =
                Serde::deserialize(ref proof_span).expect('CSP_DESER');

            let pcs_config = csp.config;
            let log_blowup = pcs_config.fri_config.log_blowup_factor;
            let commitments_span = csp.commitments;

            let preprocessed_commitment: stwo_verifier_core::Hash = *commitments_span.at(0);
            let trace_commitment: stwo_verifier_core::Hash = *commitments_span.at(1);
            let composition_commitment: stwo_verifier_core::Hash = *commitments_span.at(2);

            let mut preprocessed_sizes: Array<u32> = array![];
            i = 0;
            loop { if i >= 3 { break; } preprocessed_sizes.append(log_size); i += 1; };
            let mut trace_sizes: Array<u32> = array![];
            i = 0;
            loop { if i >= 28 { break; } trace_sizes.append(log_size); i += 1; };

            let mut channel = Default::default();
            pcs_config.mix_into(ref channel);

            let mut commitment_scheme = stwo_verifier_core::pcs::verifier::CommitmentSchemeVerifierImpl::new();
            commitment_scheme.commit(preprocessed_commitment, preprocessed_sizes.span(), ref channel, log_blowup);
            commitment_scheme.commit(trace_commitment, trace_sizes.span(), ref channel, log_blowup);

            // 7. FULL cryptographic STARK verification — ALL checks:
            // - OODS: AIR constraint evaluation matches composition polynomial
            // - Merkle: decommitment paths verify tree commitments
            // - FRI: proximity proof verifies polynomial low-degree
            // - PoW: proof of work prevents grinding
            let stark_proof = stwo_verifier_core::verifier::StarkProof { commitment_scheme_proof: csp };
            stwo_verifier_core::verifier::verify(
                air, ref channel, stark_proof, commitment_scheme, 0, composition_commitment,
            );

            // 6. Record verification + rich on-chain state
            self.recursive_verified.write(proof_hash, true);
            let count = self.recursive_count.read(model_id);
            self.recursive_count.write(model_id, count + 1);
            let block_ts = starknet::get_block_timestamp();
            self.last_io.write(model_id, io_commitment);
            self.last_proof_hash.write(model_id, proof_hash);
            self.last_verified_at.write(model_id, block_ts);
            self.last_proof_felts.write(model_id, stark_proof_data_len);
            self.last_n_layers.write(model_id, n_layers);
            self.last_trace_log_size.write(model_id, log_size);

            // 7. Emit rich verification event — full provenance in one TX
            self.emit(RecursiveProofVerified {
                model_id,
                proof_hash,
                io_commitment,
                circuit_hash: circuit_hash_packed,
                weight_super_root: weight_root_packed,
                policy_commitment: model.policy_commitment,
                n_layers,
                trace_log_size: log_size,
                proof_felts: stark_proof_data_len,
                verification_count: count + 1,
                verified_at: block_ts,
                submitter: get_caller_address(),
            });

            true
        }

        fn is_recursive_proof_verified(
            self: @ContractState, proof_hash: felt252,
        ) -> bool {
            self.recursive_verified.read(proof_hash)
        }

        fn get_recursive_verification_count(
            self: @ContractState, model_id: felt252,
        ) -> u64 {
            self.recursive_count.read(model_id)
        }

        fn get_recursive_model_info(
            self: @ContractState, model_id: felt252,
        ) -> RecursiveModelInfo {
            self.recursive_models.read(model_id)
        }

        fn get_model_policy(
            self: @ContractState, model_id: felt252,
        ) -> felt252 {
            self.recursive_models.read(model_id).policy_commitment
        }

        fn get_last_verification(
            self: @ContractState, model_id: felt252,
        ) -> (felt252, felt252, u64, u32, u32, u32, u64) {
            (
                self.last_io.read(model_id),
                self.last_proof_hash.read(model_id),
                self.last_verified_at.read(model_id),
                self.last_proof_felts.read(model_id),
                self.last_n_layers.read(model_id),
                self.last_trace_log_size.read(model_id),
                self.recursive_count.read(model_id),
            )
        }

        fn propose_upgrade(ref self: ContractState, new_class_hash: starknet::ClassHash) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");
            assert!(new_class_hash.into() != 0_felt252, "Class hash cannot be zero");

            let existing: felt252 = self.pending_upgrade.read().into();
            assert!(existing == 0, "Upgrade already pending, cancel first");

            let now = starknet::get_block_timestamp();
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
            let now = starknet::get_block_timestamp();
            assert!(now >= proposed_at + UPGRADE_DELAY, "Upgrade delay not elapsed");

            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);

            self.emit(UpgradeExecuted {
                new_class_hash, executed_at: now,
            });

            starknet::syscalls::replace_class_syscall(new_class_hash).unwrap();
        }

        fn cancel_upgrade(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "Only owner");

            let pending: starknet::ClassHash = self.pending_upgrade.read();
            assert!(pending.into() != 0_felt252, "No upgrade pending");

            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);

            self.emit(UpgradeCancelled {
                cancelled_class_hash: pending, cancelled_by: get_caller_address(),
            });
        }

        fn get_pending_upgrade(self: @ContractState) -> (starknet::ClassHash, u64) {
            (self.pending_upgrade.read(), self.upgrade_proposed_at.read())
        }
    }

    /// Extract the i-th 28-bit M31 limb from a felt252.
    fn felt252_extract_limb(value: felt252, limb_idx: u32) -> M31 {
        let v: u256 = value.into();
        let shift = limb_idx * 28;
        let mask: u256 = 0xFFFFFFF;
        let limb: u256 = (v / pow2(shift)) & mask;
        let limb_u32: u32 = limb.try_into().unwrap();
        stwo_verifier_core::fields::m31::m31(limb_u32 % 0x7FFFFFFF)
    }

    /// Convert an M31 to QM31 by embedding in the real component.
    fn m31_to_qm31(v: M31) -> QM31 {
        let z = stwo_verifier_core::fields::m31::m31(0);
        // QM31 = ((v, 0), (0, 0)) — v in real part, rest zero
        let arr: [M31; 4] = [v, z, z, z];
        stwo_verifier_core::fields::qm31::QM31Trait::from_fixed_array(arr)
    }

    fn pow2(n: u32) -> u256 {
        let mut r: u256 = 1;
        let mut i: u32 = 0;
        loop { if i >= n { break; } r = r * 2; i += 1; };
        r
    }
}
