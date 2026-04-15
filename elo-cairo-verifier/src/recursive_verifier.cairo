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
    /// Model architecture metadata (set at registration, validated at verification).
    pub n_matmuls: u32,
    pub hidden_size: u32,
    pub num_transformer_blocks: u32,
    /// Expected number of Poseidon permutations in the verifier trace.
    /// SECURITY: Prevents trace miniaturization attack. Without this, an attacker
    /// could submit a 2-row chain that satisfies all AIR constraints without
    /// running the GKR verifier. Set at registration from a reference proof.
    pub expected_n_poseidon_perms: u32,
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
        n_matmuls: u32,
        hidden_size: u32,
        num_transformer_blocks: u32,
        expected_n_poseidon_perms: u32,
    );

    /// Verify a recursive STARK proof for a registered model.
    ///
    /// Single-TX on-chain verification of a full ML inference proof.
    /// The STARK proof attests that the GKR verifier accepted the original
    /// proof covering all matmul, attention, norm, and activation layers.
    ///
    /// All parameters are visible in block explorers for full transparency.
    fn verify_recursive(
        ref self: TContractState,
        /// Unique model identifier (Poseidon hash of weight commitments).
        model_id: felt252,
        /// Poseidon hash of packed inference IO (input tokens + output logits).
        io_commitment: felt252,
        /// Model architecture fingerprint (Poseidon hash of circuit descriptor).
        circuit_hash: felt252,
        /// Poseidon Merkle root binding all weight matrices.
        weight_super_root: felt252,
        /// Number of GKR layers proven (e.g., 337 for 48-layer transformer).
        n_layers: u32,
        /// Number of matmul reductions in the GKR proof (e.g., 192 for Qwen2.5-14B).
        n_matmuls: u32,
        /// Model hidden dimension (e.g., 5120 for 14B params).
        hidden_size: u32,
        /// Number of transformer blocks (e.g., 48 for Qwen2.5-14B).
        num_transformer_blocks: u32,
        /// Proving policy commitment (Poseidon hash of PolicyConfig).
        policy_commitment: felt252,
        /// STARK execution trace log₂ size (e.g., 15 = 32768 rows).
        trace_log_size: u32,
        /// The recursive STARK proof body (FRI + Merkle decommitments).
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
    use super::RecursiveModelInfo;
    use starknet::storage::{Map, StorageMapReadAccess, StorageMapWriteAccess, StoragePointerReadAccess, StoragePointerWriteAccess};
    use starknet::{get_caller_address, ContractAddress};
    use core::poseidon::poseidon_hash_span;
    use stwo_verifier_core::fields::qm31::{QM31, QM31Zero, QM31Trait};
    use stwo_verifier_core::fields::m31::{M31, m31};
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
    // Development: 5 minutes. Set to 86400 (24h) before public mainnet launch.
    const UPGRADE_DELAY: u64 = 300;

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
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            expected_n_poseidon_perms: u32,
        ) {
            // Only owner can register models
            let caller = get_caller_address();
            assert(caller == self.owner.read(), 'Only owner can register');

            // Prevent accidental overwrite of existing registrations
            let existing = self.recursive_models.read(model_id);
            assert(existing.circuit_hash == 0, 'Model already registered');

            // SECURITY: n_poseidon_perms must be > 0 (prevents miniaturization attack)
            assert(expected_n_poseidon_perms > 0, 'n_poseidon_perms must be > 0');

            let info = RecursiveModelInfo {
                circuit_hash,
                weight_super_root,
                policy_commitment,
                n_matmuls,
                hidden_size,
                num_transformer_blocks,
                expected_n_poseidon_perms,
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
            circuit_hash: felt252,
            weight_super_root: felt252,
            n_layers: u32,
            n_matmuls: u32,
            hidden_size: u32,
            num_transformer_blocks: u32,
            policy_commitment: felt252,
            trace_log_size: u32,
            stark_proof_data: Array<felt252>,
        ) -> bool {
            // 1. Look up registered model
            let model = self.recursive_models.read(model_id);
            assert(model.circuit_hash != 0, 'Model not registered');

            // Verify caller-supplied metadata matches registration
            assert(circuit_hash == model.circuit_hash, 'Circuit hash mismatch (param)');
            assert(weight_super_root == model.weight_super_root, 'Weight root mismatch (param)');

            // 2. Compute proof hash for dedup.
            // Hash covers (model_id, io_commitment) — ensures each inference
            // can only be verified once. Does NOT include submitter address
            // to prevent the same IO from being double-counted.
            // NOTE: The submitter field in the event is not fraud-proof —
            // an MEV bot could frontrun and steal submission attribution.
            // A commit-reveal scheme would fix this but adds complexity.
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
            //   [13]     n_poseidon_perms: u32
            //   [14..18) seed_digest: QM31 (4 felts, channel seeding checkpoint)
            //   [18]     hades_commitment: felt252 (Level 1 Hades recursive proof binding)
            //   [19]     io_commitment_felt252: felt252 (full 252-bit hash)
            //   [20]     pass1_final_digest: felt252 (Pass 1 GKR verification digest)
            //   [21]     final_digest: felt252 (Pass 2 chain AIR boundary)
            //   [22]     log_size: u32
            //   [23]     n_real_rows: u32 (active HadesPerm rows for accumulator)
            //   [24..)   CommitmentSchemeProof (Serde-compatible)

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

            // Parse n_layers from proof body
            let proof_n_layers: felt252 = *proof_span.pop_front().unwrap();

            // Parse n_poseidon_perms from proof body
            let proof_n_poseidon_perms: u32 = (*proof_span.pop_front().unwrap()).try_into().unwrap();

            // Parse seed_digest: QM31 (4 M31 limbs) — channel seeding checkpoint
            let sd0: felt252 = *proof_span.pop_front().unwrap();
            let sd1: felt252 = *proof_span.pop_front().unwrap();
            let sd2: felt252 = *proof_span.pop_front().unwrap();
            let sd3: felt252 = *proof_span.pop_front().unwrap();

            // Level 1 Hades recursive proof commitment (two-level recursion)
            let hades_commitment: felt252 = *proof_span.pop_front().unwrap();

            // Full felt252 IO commitment (preserves all 252 bits)
            let proof_io_commitment_felt252: felt252 = *proof_span.pop_front().unwrap();

            // Pass 1 (full GKR verification) final digest — channel-bound.
            // This prevents a malicious prover from skipping Pass 1 (the full
            // GKR verification) and fabricating a partial witness from Pass 2 only.
            let pass1_final_digest: felt252 = *proof_span.pop_front().unwrap();

            let final_digest: felt252 = *proof_span.pop_front().unwrap();
            let proof_log_size: u32 = (*proof_span.pop_front().unwrap()).try_into().unwrap();
            let proof_n_real_rows: u32 = (*proof_span.pop_front().unwrap()).try_into().unwrap();

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

            // Cross-check ALL caller parameters against the proof body.
            // This prevents the relabeling attack: same proof body, different
            // caller metadata. Every value emitted in the event MUST match
            // what's cryptographically bound inside the proof.

            // io_commitment: full 252-bit felt252 from proof body, compare with caller.
            // The QM31 io_commitment only carries 124 bits (lossy conversion via
            // felt_to_securefield). The felt252 field preserves the full hash.
            assert(io_commitment == proof_io_commitment_felt252, 'io_commitment mismatch (proof)');

            // n_layers: compare caller param with proof body
            let proof_n_layers_u32: u32 = proof_n_layers.try_into().unwrap();
            assert(n_layers == proof_n_layers_u32, 'n_layers mismatch (param/proof)');

            // policy_commitment: check against registered model
            assert(
                policy_commitment == model.policy_commitment
                    || model.policy_commitment == 0,
                'Policy mismatch'
            );

            // trace_log_size: must match proof body
            assert(trace_log_size == proof_log_size, 'trace_log_size mismatch');

            // Model architecture metadata: must match registration.
            // These are fixed per model — set once at register_model_recursive,
            // validated here so the event cannot contain false architecture claims.
            assert(n_matmuls == model.n_matmuls, 'n_matmuls mismatch');
            assert(hidden_size == model.hidden_size, 'hidden_size mismatch');
            assert(
                num_transformer_blocks == model.num_transformer_blocks,
                'num_transformer_blocks mismatch'
            );

            // SECURITY: n_poseidon_perms from proof body must match registration.
            // This prevents the trace miniaturization attack: without this check,
            // an attacker could submit a proof with n_poseidon_perms=2 (trivially
            // small chain of 2 Hades permutations) that satisfies all chain AIR
            // constraints without ever running the GKR verifier.
            assert(
                proof_n_poseidon_perms == model.expected_n_poseidon_perms,
                'n_poseidon_perms mismatch'
            );

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
                log_n_rows: proof_log_size,
                n_real_rows: proof_n_real_rows,
                initial_digest_limbs: initial_limbs,
                final_digest_limbs: final_limbs,
            };

            // 5. Deserialize + verify
            let csp: stwo_verifier_core::pcs::verifier::CommitmentSchemeProof =
                Serde::deserialize(ref proof_span).expect('CSP_DESER');

            let pcs_config = csp.config;
            let log_blowup = pcs_config.fri_config.log_blowup_factor;

            // Enforce minimum proof security level.
            // Without this, an attacker could submit a proof with 1-bit security.
            // Minimum: pow_bits ≥ 10, log_blowup ≥ 1, n_queries ≥ 3 (≥ 13 bits total)
            assert(pcs_config.pow_bits >= 10, 'pow_bits too low');
            assert(log_blowup >= 1, 'log_blowup too low');
            assert(pcs_config.fri_config.n_queries >= 3, 'n_queries too low');
            let commitments_span = csp.commitments;

            // 3 trees: preprocessed (0), trace (1), composition (2)
            // No interaction tree (LogUp disabled for chain-only proof)
            let preprocessed_commitment: stwo_verifier_core::Hash = *commitments_span.at(0);
            let trace_commitment: stwo_verifier_core::Hash = *commitments_span.at(1);
            let composition_commitment: stwo_verifier_core::Hash = *commitments_span.at(2);

            let mut preprocessed_sizes: Array<u32> = array![];
            i = 0;
            loop { if i >= 3 { break; } preprocessed_sizes.append(proof_log_size); i += 1; };
            let mut trace_sizes: Array<u32> = array![];
            i = 0;
            // Slim trace: 48 columns (9+9+9+9 data + 8 carry + 1 k + 3 selectors)
            loop { if i >= 48 { break; } trace_sizes.append(proof_log_size); i += 1; };

            let mut channel = Default::default();
            pcs_config.mix_into(ref channel);

            // ── Bind public inputs to Fiat-Shamir channel ────────────
            // Reconstruct QM31 values from the M31 limbs parsed from the
            // proof header, then mix into the channel in the same order
            // as the Rust prover: [circuit_hash, io_commitment,
            // weight_super_root] via mix_felts, then n_layers via mix_u64.
            //
            // This makes the STARK proof cryptographically bound to these
            // values.  Submitting different metadata causes channel
            // divergence → FRI verification failure.
            let _z = m31(0);
            let circuit_hash_qm31 = QM31Trait::from_fixed_array([
                felt252_to_m31(ch0), felt252_to_m31(ch1),
                felt252_to_m31(ch2), felt252_to_m31(ch3),
            ]);
            let io_commitment_qm31 = QM31Trait::from_fixed_array([
                felt252_to_m31(io0), felt252_to_m31(io1),
                felt252_to_m31(io2), felt252_to_m31(io3),
            ]);
            let weight_root_qm31 = QM31Trait::from_fixed_array([
                felt252_to_m31(wr0), felt252_to_m31(wr1),
                felt252_to_m31(wr2), felt252_to_m31(wr3),
            ]);
            channel.mix_felts(
                array![circuit_hash_qm31, io_commitment_qm31, weight_root_qm31].span()
            );
            let proof_n_layers_u64: u64 = proof_n_layers.try_into().unwrap();
            channel.mix_u64(proof_n_layers_u64);
            // SECURITY: n_poseidon_perms bound to channel — prevents miniaturization
            let proof_n_poseidon_perms_u64: u64 = proof_n_poseidon_perms.into();
            channel.mix_u64(proof_n_poseidon_perms_u64);

            // SECURITY: seed_digest checkpoint — binds chain to model dimensions
            let seed_digest_qm31 = QM31Trait::from_fixed_array([
                felt252_to_m31(sd0), felt252_to_m31(sd1),
                felt252_to_m31(sd2), felt252_to_m31(sd3),
            ]);
            channel.mix_felts(array![seed_digest_qm31].span());

            // SECURITY: Bind Level 1 Hades recursive proof commitment.
            // This cryptographically ties the chain STARK to the set of verified
            // Hades permutations. An attacker cannot substitute different permutations
            // without changing the commitment, which invalidates the STARK proof.
            let hc_u256: u256 = hades_commitment.into();
            channel.mix_u64((hc_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256).try_into().unwrap());
            channel.mix_u64(((hc_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());
            channel.mix_u64(((hc_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());
            channel.mix_u64((hc_u256 & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());

            // Bind the full felt252 io_commitment into the channel.
            // This ensures the proof body's io_commitment_felt252 field
            // cannot be tampered without invalidating the STARK.
            // Split into 4 × u64 to match the Rust prover's 4 × mix_u64 calls.
            let io_u256: u256 = proof_io_commitment_felt252.into();
            channel.mix_u64((io_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256).try_into().unwrap());
            channel.mix_u64(((io_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());
            channel.mix_u64(((io_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());
            channel.mix_u64((io_u256 & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());

            // SECURITY: Bind Pass 1 (full GKR verification) final digest.
            // This prevents the Pass 2 fabrication attack: a malicious prover
            // cannot skip the full GKR verification and fabricate a partial
            // witness. Without the correct Pass 1 digest, the Fiat-Shamir
            // channel diverges and FRI verification fails.
            // Split into 4 × u64 to match the Rust prover's 4 × mix_u64 calls.
            let p1_u256: u256 = pass1_final_digest.into();
            channel.mix_u64((p1_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256).try_into().unwrap());
            channel.mix_u64(((p1_u256 / 0x10000000000000000_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());
            channel.mix_u64(((p1_u256 / 0x10000000000000000_u256) & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());
            channel.mix_u64((p1_u256 & 0xFFFFFFFFFFFFFFFF_u256).try_into().unwrap());

            let mut commitment_scheme = stwo_verifier_core::pcs::verifier::CommitmentSchemeVerifierImpl::new();
            commitment_scheme.commit(preprocessed_commitment, preprocessed_sizes.span(), ref channel, log_blowup);
            commitment_scheme.commit(trace_commitment, trace_sizes.span(), ref channel, log_blowup);

            // 7. FULL cryptographic STARK verification — ALL checks:
            // - OODS: AIR constraint evaluation matches composition polynomial
            // - Merkle: decommitment paths verify tree commitments
            // - FRI: proximity proof verifies polynomial low-degree
            // - PoW: proof of work prevents grinding
            let stark_proof = stwo_verifier_core::verifier::StarkProof { commitment_scheme_proof: csp };
            // v1.2.2 signature: verify(proof, air, composition_log_degree_bound,
            //   composition_commitment, commitment_scheme, ref channel, min_security_bits)
            let composition_log_degree_bound = proof_log_size + 1;

            stwo_verifier_core::verifier::verify(
                stark_proof, air, composition_log_degree_bound,
                composition_commitment, commitment_scheme, ref channel, 0,
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
            self.last_trace_log_size.write(model_id, trace_log_size);

            // 7. Emit rich verification event — full provenance in one TX
            self.emit(RecursiveProofVerified {
                model_id,
                proof_hash,
                io_commitment,
                circuit_hash,
                weight_super_root,
                policy_commitment,
                n_layers,
                trace_log_size,
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

    /// Convert a felt252 that holds an M31 value (0..2^31-1) to M31.
    /// Used to reconstruct QM31 from proof header limbs.
    fn felt252_to_m31(value: felt252) -> M31 {
        let v_u32: u32 = value.try_into().unwrap();
        m31(v_u32)
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
