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

use crate::types::MatMulSumcheckProof;

#[starknet::interface]
pub trait ISumcheckVerifier<TContractState> {
    /// Register a model's weight commitment on-chain.
    fn register_model(
        ref self: TContractState, model_id: felt252, weight_commitment: felt252,
    );

    /// Verify a matmul sumcheck proof on-chain.
    /// Replays the full Fiat-Shamir transcript, verifies sumcheck rounds,
    /// final evaluation, and MLE opening proofs for both matrices.
    fn verify_matmul(
        ref self: TContractState, model_id: felt252, proof: MatMulSumcheckProof,
    ) -> bool;

    /// Get the weight commitment for a registered model.
    fn get_model_commitment(self: @TContractState, model_id: felt252) -> felt252;

    /// Get the number of verified proofs for a model.
    fn get_verification_count(self: @TContractState, model_id: felt252) -> u64;

    /// Check if a specific proof hash has been verified.
    fn is_proof_verified(self: @TContractState, proof_hash: felt252) -> bool;

    /// Get the contract owner.
    fn get_owner(self: @TContractState) -> starknet::ContractAddress;
}

#[starknet::contract]
mod SumcheckVerifierContract {
    use super::MatMulSumcheckProof;
    use crate::field::{log2_ceil, next_power_of_two, pack_qm31_to_felt};
    use crate::channel::{
        channel_default, channel_mix_u64, channel_mix_felt, channel_draw_qm31s,
    };
    use crate::sumcheck::verify_sumcheck_inner;
    use crate::mle::verify_mle_opening;
    use starknet::storage::{
        StoragePointerReadAccess, StoragePointerWriteAccess, Map, StoragePathEntry,
    };
    use starknet::{ContractAddress, get_caller_address};

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
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        ModelRegistered: ModelRegistered,
        MatMulVerified: MatMulVerified,
        VerificationFailed: VerificationFailed,
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
    struct VerificationFailed {
        #[key]
        model_id: felt252,
        reason: felt252,
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
    }
}
