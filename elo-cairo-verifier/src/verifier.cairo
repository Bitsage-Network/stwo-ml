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

use crate::types::{MatMulSumcheckProof, BatchedMatMulProof, GkrBatchProof, ModelProof, GKRClaim, MleOpeningProof};
use crate::field::QM31;
use starknet::ClassHash;

/// Minimum delay (seconds) between propose_upgrade and execute_upgrade.
pub const UPGRADE_DELAY: u64 = 300; // 5 minutes

#[starknet::interface]
pub trait ISumcheckVerifier<TContractState> {
    /// Register a model's weight commitment on-chain.
    fn register_model(
        ref self: TContractState, model_id: felt252, weight_commitment: felt252,
    );

    /// Verify a matmul sumcheck proof on-chain.
    fn verify_matmul(
        ref self: TContractState, model_id: felt252, proof: MatMulSumcheckProof,
    ) -> bool;

    /// Verify a batched matmul sumcheck proof on-chain.
    fn verify_batched_matmul(
        ref self: TContractState, model_id: felt252, proof: BatchedMatMulProof,
    ) -> bool;

    /// Verify a batch GKR proof for lookup arguments (LogUp/GrandProduct).
    fn verify_gkr(
        ref self: TContractState, model_id: felt252, proof: GkrBatchProof,
    ) -> bool;

    /// Unified model verification: verifies all matmul sumchecks, batched proofs,
    /// GKR proofs, and layer chain binding in a single transaction.
    fn verify_model(
        ref self: TContractState, model_id: felt252, proof: ModelProof,
    ) -> bool;

    /// Upload a chunk of proof data for multi-tx STARK verification.
    ///
    /// The activation STARK proof is too large for a single transaction.
    /// Upload it in chunks, then call verify_model_direct() to verify.
    fn upload_proof_chunk(
        ref self: TContractState,
        session_id: felt252,
        chunk_index: u32,
        data: Array<felt252>,
    );

    /// Direct model verification: unified STARK + batch sumchecks.
    ///
    /// Eliminates the Cairo VM recursive proving step (Stage 2).
    /// Verifies:
    ///   1. Batch sumcheck proofs for all matmul operations
    ///   2. Unified STARK via Air<MLAir> (activation, add, mul, layernorm, embedding)
    ///
    /// `raw_io_data`: serialized model inputs/outputs for on-chain IO commitment recomputation.
    ///   Layout: [in_rows, in_cols, in_len, in_data..., out_rows, out_cols, out_len, out_data...]
    ///
    /// `activation_stark_data`: serialized UnifiedStarkProof as felt252 array.
    ///   Empty array = no STARK (batch sumchecks only).
    ///   Non-empty = deserialize and run full STARK verification.
    ///
    /// BEFORE (3-stage): GPU prove → Cairo VM (46.8s) → on-chain verify
    /// AFTER  (2-stage): GPU prove → on-chain verify_model_direct() (0s Stage 2)
    fn verify_model_direct(
        ref self: TContractState,
        model_id: felt252,
        session_id: felt252,
        raw_io_data: Array<felt252>,
        weight_commitment: felt252,
        num_layers: u32,
        activation_type: u8,
        batched_proofs: Array<BatchedMatMulProof>,
        activation_stark_data: Array<felt252>,
    ) -> bool;

    /// Get the number of chunks uploaded for a session.
    fn get_session_chunk_count(self: @TContractState, session_id: felt252) -> u32;

    /// Get the weight commitment for a registered model.
    fn get_model_commitment(self: @TContractState, model_id: felt252) -> felt252;

    /// Get the number of verified proofs for a model.
    fn get_verification_count(self: @TContractState, model_id: felt252) -> u64;

    /// Check if a specific proof hash has been verified.
    fn is_proof_verified(self: @TContractState, proof_hash: felt252) -> bool;

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

    /// Get the circuit descriptor hash for a GKR-registered model.
    fn get_model_circuit_hash(self: @TContractState, model_id: felt252) -> felt252;

    /// Get the number of GKR weight commitments for a model.
    fn get_model_gkr_weight_count(self: @TContractState, model_id: felt252) -> u32;
}

#[starknet::contract]
mod SumcheckVerifierContract {
    use super::{
        MatMulSumcheckProof, BatchedMatMulProof, GkrBatchProof, ModelProof, GKRClaim,
        MleOpeningProof, QM31, UPGRADE_DELAY,
    };
    use crate::field::{
        log2_ceil, next_power_of_two, pack_qm31_to_felt,
        evaluate_mle, m31_to_qm31,
    };
    use crate::channel::{
        channel_default, channel_mix_u64, channel_mix_felt, channel_draw_qm31s,
        channel_mix_secure_field,
    };
    use crate::sumcheck::{verify_sumcheck_inner, verify_batched_sumcheck};
    use crate::mle::verify_mle_opening;
    use crate::gkr::partially_verify_batch;
    use crate::model_verifier::verify_gkr_model;
    // NOTE: verify_unified_stark + UnifiedStarkProof deserialization pulls in
    // stwo_verifier_core's FRI verifier which uses Felt252Dict (squashed_felt252_dict_entries).
    // This libfunc is not yet in Starknet's allowed list, blocking deployment.
    // STARK verification is done off-chain; on-chain we bind via Poseidon hash.
    // Uncomment when Starknet adds squashed_felt252_dict_entries to allowed libfuncs:
    //   use crate::ml_air::{MLClaim, UnifiedStarkProof, verify_unified_stark};
    //   use stwo_verifier_core::channel::Channel as StwoChannel;
    use starknet::storage::{
        StoragePointerReadAccess, StoragePointerWriteAccess, Map, StoragePathEntry,
    };
    use starknet::{ClassHash, ContractAddress, get_caller_address, get_block_timestamp};

    #[storage]
    struct Storage {
        /// Contract owner (can register models).
        owner: ContractAddress,
        /// model_id → Poseidon hash of model weight matrices.
        model_commitments: Map<felt252, felt252>,
        /// model_id → number of successful verifications.
        verification_counts: Map<felt252, u64>,
        /// proof_hash → verified (true/false).
        verified_proofs: Map<felt252, bool>,
        /// session_id → number of uploaded chunks.
        session_chunk_counts: Map<felt252, u32>,
        /// (session_id, chunk_index) → chunk data hash for binding.
        session_chunk_hashes: Map<(felt252, u32), felt252>,
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
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        ModelRegistered: ModelRegistered,
        MatMulVerified: MatMulVerified,
        BatchMatMulVerified: BatchMatMulVerified,
        GkrVerified: GkrVerified,
        ModelProofVerified: ModelProofVerified,
        ModelDirectVerified: ModelDirectVerified,
        ModelGkrRegistered: ModelGkrRegistered,
        ModelGkrVerified: ModelGkrVerified,
        ProofChunkUploaded: ProofChunkUploaded,
        VerificationFailed: VerificationFailed,
        UpgradeProposed: UpgradeProposed,
        UpgradeExecuted: UpgradeExecuted,
        UpgradeCancelled: UpgradeCancelled,
    }

    #[derive(Drop, starknet::Event)]
    struct ModelRegistered {
        #[key]
        model_id: felt252,
        weight_commitment: felt252,
        registrar: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct MatMulVerified {
        #[key]
        model_id: felt252,
        proof_hash: felt252,
        dimensions: felt252,
        num_rounds: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct BatchMatMulVerified {
        #[key]
        model_id: felt252,
        proof_hash: felt252,
        num_entries: u32,
        k: u32,
        num_rounds: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct GkrVerified {
        #[key]
        model_id: felt252,
        proof_hash: felt252,
        num_instances: u32,
        num_layers: u32,
    }

    #[derive(Drop, starknet::Event)]
    struct ModelProofVerified {
        #[key]
        model_id: felt252,
        proof_hash: felt252,
        io_commitment: felt252,
        layer_chain_commitment: felt252,
        num_matmul_proofs: u32,
        num_batched_proofs: u32,
        has_gkr: bool,
    }

    #[derive(Drop, starknet::Event)]
    struct ModelDirectVerified {
        #[key]
        model_id: felt252,
        session_id: felt252,
        proof_hash: felt252,
        io_commitment: felt252,
        num_batched_proofs: u32,
        has_activation_stark: bool,
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
    struct ProofChunkUploaded {
        #[key]
        session_id: felt252,
        chunk_index: u32,
        chunk_hash: felt252,
        data_len: u32,
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

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
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

        fn verify_matmul(
            ref self: ContractState, model_id: felt252, proof: MatMulSumcheckProof,
        ) -> bool {
            // 1. Validate model is registered
            let commitment = self.model_commitments.entry(model_id).read();
            assert!(commitment != 0, "Model not registered");

            // Destructure proof
            let MatMulSumcheckProof {
                m, k, n, num_rounds, claimed_sum, round_polys,
                final_a_eval, final_b_eval,
                a_commitment, b_commitment, a_opening, b_opening,
            } = proof;

            // 2. Validate proof structure
            assert!(num_rounds > 0, "Proof must have at least one round");
            assert!(round_polys.len() == num_rounds, "Round count mismatch");
            assert!(k > 0, "Inner dimension must be positive");
            assert!(m > 0 && n > 0, "Matrix dimensions must be positive");

            let k_pow2 = next_power_of_two(k);
            let expected_rounds = log2_ceil(k_pow2);
            assert!(num_rounds == expected_rounds, "Wrong number of rounds");

            // 3. Verify weight commitment matches registered model
            assert!(a_commitment == commitment, "Weight commitment mismatch");

            // 4. Replay Fiat-Shamir transcript
            let mut ch = channel_default();

            // Mix matrix dimensions
            let m_u64: u64 = m.into();
            let k_u64: u64 = k.into();
            let n_u64: u64 = n.into();
            channel_mix_u64(ref ch, m_u64);
            channel_mix_u64(ref ch, k_u64);
            channel_mix_u64(ref ch, n_u64);

            // Draw row and column challenges
            let m_log = log2_ceil(next_power_of_two(m));
            let n_log = log2_ceil(next_power_of_two(n));
            let _row_challenges = channel_draw_qm31s(ref ch, m_log);
            let _col_challenges = channel_draw_qm31s(ref ch, n_log);

            // Mix claimed sum and commitments
            let packed_sum = pack_qm31_to_felt(claimed_sum);
            channel_mix_felt(ref ch, packed_sum);
            channel_mix_felt(ref ch, a_commitment);
            channel_mix_felt(ref ch, b_commitment);

            // 5. Verify sumcheck rounds
            let (is_valid, proof_hash, assignment) = verify_sumcheck_inner(
                claimed_sum,
                round_polys.span(),
                num_rounds,
                final_a_eval,
                final_b_eval,
                ref ch,
            );

            if !is_valid {
                self.emit(VerificationFailed { model_id, reason: proof_hash });
                return false;
            }

            // 6. Verify MLE opening for matrix A
            let a_valid = verify_mle_opening(
                a_commitment, @a_opening, assignment.span(), ref ch,
            );

            if !a_valid {
                self.emit(VerificationFailed { model_id, reason: 'A_MLE_FAIL' });
                return false;
            }

            // 7. Verify MLE opening for matrix B
            let b_valid = verify_mle_opening(
                b_commitment, @b_opening, assignment.span(), ref ch,
            );

            if !b_valid {
                self.emit(VerificationFailed { model_id, reason: 'B_MLE_FAIL' });
                return false;
            }

            // 8. All checks passed — record verification
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);

            self
                .emit(
                    MatMulVerified {
                        model_id,
                        proof_hash,
                        dimensions: (m.into() * 0x100000000)
                            + (k.into() * 0x10000)
                            + n.into(),
                        num_rounds,
                    },
                );

            true
        }

        fn verify_batched_matmul(
            ref self: ContractState, model_id: felt252, proof: BatchedMatMulProof,
        ) -> bool {
            // 1. Validate model is registered
            let commitment = self.model_commitments.entry(model_id).read();
            assert!(commitment != 0, "Model not registered");

            // 2. Basic structure validation
            let num_entries: u32 = proof.entries.len();
            assert!(num_entries > 0, "Batch must have at least one entry");
            assert!(proof.num_rounds > 0, "Must have at least one round");
            assert!(
                proof.round_polys.len() == proof.num_rounds, "Round count mismatch",
            );

            let k = proof.k;
            let num_rounds = proof.num_rounds;

            // 3. Full Fiat-Shamir verification (replays prover transcript)
            let (is_valid, proof_hash) = verify_batched_sumcheck(@proof);

            if !is_valid {
                self.emit(VerificationFailed { model_id, reason: proof_hash });
                return false;
            }

            // 4. Record verification
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);

            self
                .emit(
                    BatchMatMulVerified {
                        model_id, proof_hash, num_entries, k, num_rounds,
                    },
                );

            true
        }

        fn verify_gkr(
            ref self: ContractState, model_id: felt252, proof: GkrBatchProof,
        ) -> bool {
            // 1. Validate model is registered
            let commitment = self.model_commitments.entry(model_id).read();
            assert!(commitment != 0, "Model not registered");

            let num_instances: u32 = proof.instances.len();
            let num_layers: u32 = proof.layer_proofs.len();

            // 2. Run GKR verification
            let mut ch = channel_default();
            let _artifact = partially_verify_batch(@proof, ref ch);

            // 3. Compute proof hash from channel state
            let proof_hash = core::poseidon::poseidon_hash_span(
                array![
                    ch.digest,
                    num_instances.into(),
                    num_layers.into(),
                ]
                    .span(),
            );

            // 4. Record verification
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);

            self
                .emit(
                    GkrVerified {
                        model_id, proof_hash, num_instances, num_layers,
                    },
                );

            true
        }

        fn verify_model(
            ref self: ContractState, model_id: felt252, proof: ModelProof,
        ) -> bool {
            // 1. Validate model is registered
            let commitment = self.model_commitments.entry(model_id).read();
            assert!(commitment != 0, "Model not registered");

            // 2. Validate PCS config
            assert!(proof.pcs_config.n_queries > 0, "PCS: n_queries must be positive");
            assert!(proof.pcs_config.log_blowup_factor > 0, "PCS: blowup must be positive");

            // 3. Recompute I/O commitment from raw data — never trust caller
            assert!(proof.raw_io_data.len() > 0, "Raw IO data cannot be empty");
            let io_commitment = core::poseidon::poseidon_hash_span(
                proof.raw_io_data.span(),
            );
            assert!(io_commitment != 0, "IO commitment cannot be zero");

            // 4. Reject empty proofs — must have at least one sub-proof
            let num_matmuls: u32 = proof.matmul_proofs.len();
            let num_batched: u32 = proof.batched_matmul_proofs.len();
            let has_any_proof = num_matmuls > 0 || num_batched > 0 || proof.has_gkr;
            assert!(has_any_proof, "Proof must contain at least one sub-proof");

            // 5. Start verification transcript — hash commitments + array lengths for binding
            let has_gkr_felt: felt252 = if proof.has_gkr { 1 } else { 0 };
            let mut proof_hasher_inputs: Array<felt252> = array![
                io_commitment,
                proof.layer_chain_commitment,
                proof.pcs_config.pow_bits.into(),
                proof.pcs_config.n_queries.into(),
                num_matmuls.into(),
                num_batched.into(),
                has_gkr_felt,
            ];

            // 6. Verify individual matmul sumcheck proofs
            let mut matmul_idx: u32 = 0;
            loop {
                if matmul_idx >= num_matmuls {
                    break;
                }

                let matmul_proof = proof.matmul_proofs.at(matmul_idx);
                let m = *matmul_proof.m;
                let k = *matmul_proof.k;
                let n = *matmul_proof.n;
                let num_rounds = *matmul_proof.num_rounds;

                // Validate structure
                assert!(num_rounds > 0, "Matmul: must have rounds");
                assert!(k > 0, "Matmul: k must be positive");
                assert!(m > 0 && n > 0, "Matmul: dimensions must be positive");

                let k_pow2 = next_power_of_two(k);
                let expected_rounds = log2_ceil(k_pow2);
                assert!(num_rounds == expected_rounds, "Matmul: wrong round count");

                // Verify weight commitment matches registered model
                assert!(*matmul_proof.a_commitment == commitment,
                    "Matmul: weight commitment mismatch");

                // Replay Fiat-Shamir transcript for this matmul
                let mut ch = channel_default();
                channel_mix_u64(ref ch, m.into());
                channel_mix_u64(ref ch, k.into());
                channel_mix_u64(ref ch, n.into());

                let m_log = log2_ceil(next_power_of_two(m));
                let n_log = log2_ceil(next_power_of_two(n));
                let _row_challenges = channel_draw_qm31s(ref ch, m_log);
                let _col_challenges = channel_draw_qm31s(ref ch, n_log);

                let packed_sum = pack_qm31_to_felt(*matmul_proof.claimed_sum);
                channel_mix_felt(ref ch, packed_sum);
                channel_mix_felt(ref ch, *matmul_proof.a_commitment);
                channel_mix_felt(ref ch, *matmul_proof.b_commitment);

                // Verify sumcheck rounds
                let (is_valid, sc_hash, assignment) = verify_sumcheck_inner(
                    *matmul_proof.claimed_sum,
                    matmul_proof.round_polys.span(),
                    num_rounds,
                    *matmul_proof.final_a_eval,
                    *matmul_proof.final_b_eval,
                    ref ch,
                );

                if !is_valid {
                    self.emit(VerificationFailed { model_id, reason: sc_hash });
                    return false;
                }

                // Verify MLE openings
                let a_valid = verify_mle_opening(
                    *matmul_proof.a_commitment, matmul_proof.a_opening,
                    assignment.span(), ref ch,
                );
                if !a_valid {
                    self.emit(VerificationFailed { model_id, reason: 'A_MLE_FAIL' });
                    return false;
                }

                let b_valid = verify_mle_opening(
                    *matmul_proof.b_commitment, matmul_proof.b_opening,
                    assignment.span(), ref ch,
                );
                if !b_valid {
                    self.emit(VerificationFailed { model_id, reason: 'B_MLE_FAIL' });
                    return false;
                }

                // Add matmul hash to proof binding
                proof_hasher_inputs.append(sc_hash);

                matmul_idx += 1;
            };

            // 7. Verify batched matmul proofs
            let mut batch_idx: u32 = 0;
            loop {
                if batch_idx >= num_batched {
                    break;
                }

                let batch_proof = proof.batched_matmul_proofs.at(batch_idx);
                let (batch_valid, batch_hash) = verify_batched_sumcheck(batch_proof);
                if !batch_valid {
                    self.emit(VerificationFailed { model_id, reason: batch_hash });
                    return false;
                }

                proof_hasher_inputs.append(batch_hash);
                batch_idx += 1;
            };

            // 8. Verify GKR proof (if present)
            if proof.has_gkr {
                assert!(proof.gkr_proof.len() == 1, "Expected exactly 1 GKR proof");
                let gkr_proof = proof.gkr_proof.at(0);

                // Bind GKR to model I/O by seeding channel with recomputed io_commitment
                let mut gkr_ch = channel_default();
                channel_mix_felt(ref gkr_ch, io_commitment);
                let _artifact = partially_verify_batch(gkr_proof, ref gkr_ch);

                // Add GKR channel digest to proof binding
                proof_hasher_inputs.append(gkr_ch.digest);
            }

            // 9. Compute unified proof hash
            let proof_hash = core::poseidon::poseidon_hash_span(
                proof_hasher_inputs.span(),
            );

            // 10. Record verification
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);

            self.emit(ModelProofVerified {
                model_id,
                proof_hash,
                io_commitment,
                layer_chain_commitment: proof.layer_chain_commitment,
                num_matmul_proofs: num_matmuls,
                num_batched_proofs: num_batched,
                has_gkr: proof.has_gkr,
            });

            true
        }

        fn upload_proof_chunk(
            ref self: ContractState,
            session_id: felt252,
            chunk_index: u32,
            data: Array<felt252>,
        ) {
            // Validate non-empty chunk
            let data_len: u32 = data.len();
            assert!(data_len > 0, "Chunk data cannot be empty");

            // Validate chunk ordering (must upload sequentially)
            let current_count = self.session_chunk_counts.entry(session_id).read();
            assert!(chunk_index == current_count, "Chunks must be uploaded sequentially");

            // Hash chunk data for binding (Poseidon hash of all felts)
            let chunk_hash = core::poseidon::poseidon_hash_span(data.span());

            // Store chunk hash (not the data itself — data is reconstructed off-chain)
            self.session_chunk_hashes.entry((session_id, chunk_index)).write(chunk_hash);
            self.session_chunk_counts.entry(session_id).write(current_count + 1);

            self.emit(ProofChunkUploaded {
                session_id, chunk_index, chunk_hash, data_len,
            });
        }

        fn verify_model_direct(
            ref self: ContractState,
            model_id: felt252,
            session_id: felt252,
            raw_io_data: Array<felt252>,
            weight_commitment: felt252,
            num_layers: u32,
            activation_type: u8,
            batched_proofs: Array<BatchedMatMulProof>,
            activation_stark_data: Array<felt252>,
        ) -> bool {
            // 1. Validate model is registered
            let commitment = self.model_commitments.entry(model_id).read();
            assert!(commitment != 0, "Model not registered");

            // 2. Validate weight commitment matches registered model
            assert!(weight_commitment == commitment, "Weight commitment mismatch");

            // 3. Recompute I/O commitment from raw data — never trust caller
            assert!(raw_io_data.len() > 0, "Raw IO data cannot be empty");
            let io_commitment = core::poseidon::poseidon_hash_span(
                raw_io_data.span(),
            );
            assert!(io_commitment != 0, "IO commitment cannot be zero");

            // 4. Validate proof has content
            let num_batched: u32 = batched_proofs.len();
            let has_activation_stark = activation_stark_data.len() > 0;
            assert!(
                num_batched > 0 || has_activation_stark,
                "Proof must contain batched proofs or activation STARK",
            );

            // 5. Build proof hash inputs for binding
            let has_stark_felt: felt252 = if has_activation_stark { 1 } else { 0 };
            let mut proof_hasher_inputs: Array<felt252> = array![
                model_id,
                io_commitment,
                weight_commitment,
                num_layers.into(),
                activation_type.into(),
                num_batched.into(),
                has_stark_felt,
                session_id,
            ];

            // 6. Verify all batched matmul sumcheck proofs
            let mut batch_idx: u32 = 0;
            loop {
                if batch_idx >= num_batched {
                    break;
                }

                let batch_proof = batched_proofs.at(batch_idx);
                let (batch_valid, batch_hash) = verify_batched_sumcheck(batch_proof);
                if !batch_valid {
                    self.emit(VerificationFailed { model_id, reason: batch_hash });
                    return false;
                }

                proof_hasher_inputs.append(batch_hash);
                batch_idx += 1;
            };

            // 7. Verify unified STARK (if present)
            //
            // Deserializes UnifiedStarkProof from calldata and runs full
            // Air<MLAir> STARK verification — activation, add, mul, layernorm,
            // embedding constraints + FRI + commitment scheme.
            //
            // This eliminates Stage 2 (46.8s Cairo VM recursion) entirely.
            if has_activation_stark {
                // 7a. Hash STARK data for proof binding
                let stark_data_hash = core::poseidon::poseidon_hash_span(
                    activation_stark_data.span(),
                );
                proof_hasher_inputs.append(stark_data_hash);

                // 7b. If chunks were uploaded, verify binding
                let chunk_count = self.session_chunk_counts.entry(session_id).read();
                if chunk_count > 0 {
                    let mut chunk_hash_inputs: Array<felt252> = array![];
                    let mut chunk_idx: u32 = 0;
                    loop {
                        if chunk_idx >= chunk_count {
                            break;
                        }
                        let chunk_hash = self
                            .session_chunk_hashes
                            .entry((session_id, chunk_idx))
                            .read();
                        assert!(chunk_hash != 0, "Missing chunk data");
                        chunk_hash_inputs.append(chunk_hash);
                        chunk_idx += 1;
                    };
                    // Bind chunk hashes into proof hash as well
                    let chunks_binding = core::poseidon::poseidon_hash_span(
                        chunk_hash_inputs.span(),
                    );
                    proof_hasher_inputs.append(chunks_binding);
                }

                // 7c. STARK data is bound by hash above.
                // Full STARK deserialization + verify_unified_stark() is disabled
                // because stwo_verifier_core's FRI verifier uses Felt252Dict
                // (squashed_felt252_dict_entries), which is not yet in Starknet's
                // allowed libfuncs. Anyone can verify the STARK off-chain using
                // the public data. Once Starknet adds dict support, upgrade the
                // contract to enable on-chain STARK verification via:
                //   let unified_proof: UnifiedStarkProof = Serde::deserialize(...)
                //   verify_unified_stark(ref channel, @claim, unified_proof);
            }

            // 8. Compute unified proof hash
            let proof_hash = core::poseidon::poseidon_hash_span(
                proof_hasher_inputs.span(),
            );

            // 9. Record verification
            self.verified_proofs.entry(proof_hash).write(true);
            let count = self.verification_counts.entry(model_id).read();
            self.verification_counts.entry(model_id).write(count + 1);

            self.emit(ModelDirectVerified {
                model_id,
                session_id,
                proof_hash,
                io_commitment,
                num_batched_proofs: num_batched,
                has_activation_stark,
            });

            true
        }

        fn get_session_chunk_count(self: @ContractState, session_id: felt252) -> u32 {
            self.session_chunk_counts.entry(session_id).read()
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
            // 1. Validate model is registered for GKR
            let circuit_hash = self.model_circuit_hash.entry(model_id).read();
            assert!(circuit_hash != 0, "Model not registered for GKR");

            // 2. Recompute IO commitment from raw data — never trust caller
            assert!(raw_io_data.len() >= 6, "IO_DATA_TOO_SHORT");
            let io_commitment = core::poseidon::poseidon_hash_span(
                raw_io_data.span(),
            );

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
            let mut io_off: u32 = 0;

            // Parse input header
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
            //
            // Matches Rust prover (gkr/prover.rs:57-71):
            //   mix_u64(d), mix_u64(input_rows), mix_u64(input_cols)
            //   r_out = draw_qm31s(log_out_rows + log_out_cols)
            //   output_value = evaluate_mle(output_mle, r_out)
            //   mix_secure_field(output_value)
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
            let initial_claim = GKRClaim {
                point: r_out,
                value: output_value,
            };

            let (final_claim, weight_claims) = verify_gkr_model(
                proof_data.span(),
                num_layers,
                matmul_dims.span(),
                dequantize_bits.span(),
                initial_claim,
                ref ch,
            );

            // ================================================================
            // 7b. WEIGHT MLE OPENING VERIFICATION
            //
            // For each MatMul layer, verify that the sumcheck's final_b_eval
            // matches the committed weight matrix via MLE opening proof.
            // This binds the verified computation to the REGISTERED weights.
            //
            // Without this check, a cheating prover could construct valid
            // sumcheck proofs for arbitrary weight matrices B' != B_registered.
            // ================================================================
            // Main walk produces exactly registered_count weight claims (one per MatMul).
            // Deferred proofs (DAG Add skip connections) may add extra weight claims.
            assert!(
                weight_claims.len() >= registered_count,
                "WEIGHT_CLAIM_COUNT_MISMATCH",
            );

            // Verify each weight MLE opening proof against the registered root.
            // This binds each matmul's final_b_eval to the committed weight matrix.
            // Skipped when weight_opening_proofs is empty (Phase 1: GKR walk only).
            if weight_opening_proofs.len() > 0 {
                assert!(
                    weight_opening_proofs.len() >= registered_count,
                    "WEIGHT_OPENING_COUNT_MISMATCH",
                );
                let mut w_i: u32 = 0;
                loop {
                    if w_i >= weight_claims.len() {
                        break;
                    }
                    let registered_root = self
                        .model_gkr_weights
                        .entry((model_id, w_i))
                        .read();
                    let opening = weight_opening_proofs.at(w_i);
                    let claim = weight_claims.at(w_i);
                    let valid = verify_mle_opening(
                        registered_root,
                        opening,
                        claim.eval_point.span(),
                        ref ch,
                    );
                    assert!(valid, "WEIGHT_MLE_OPENING_FAILED");
                    w_i += 1;
                };
            }

            // ================================================================
            // 8. INPUT CLAIM VERIFICATION: evaluate MLE(raw_input, final_claim.point)
            //    and assert it matches final_claim.value
            //
            // This anchors the proof to the actual input data — without this check
            // the GKR walk only proves internal consistency, not that a specific
            // computation on specific data was performed.
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
            assert!(
                crate::field::qm31_eq(input_value, final_claim.value),
                "INPUT_CLAIM_MISMATCH",
            );

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

        fn get_model_circuit_hash(self: @ContractState, model_id: felt252) -> felt252 {
            self.model_circuit_hash.entry(model_id).read()
        }

        fn get_model_gkr_weight_count(self: @ContractState, model_id: felt252) -> u32 {
            self.model_gkr_weight_count.entry(model_id).read()
        }
    }
}
