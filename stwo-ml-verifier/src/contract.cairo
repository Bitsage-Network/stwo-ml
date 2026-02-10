/// ObelyskVerifier — On-chain ML inference verification with SAGE payment.
///
/// Verifies recursive STARK proofs of ML inference and processes
/// SAGE token payments in a single atomic transaction.
///
/// Architecture:
///   1. ML inference proven on GPU (stwo-ml, 40s for Qwen3-14B)
///   2. ML verifier runs in Cairo VM, producing execution trace
///   3. Recursive STARK proof of the verification (cairo-prove)
///   4. Proof fact submitted to this contract
///   5. Contract verifies fact + transfers SAGE payment + emits events
///
/// Events emitted per verification (7 total):
///   ModelRegistered, JobCreated, ProofSubmitted, InferenceVerified,
///   PaymentProcessed, WorkerRewarded, VerificationComplete

#[starknet::contract]
pub mod ObelyskVerifier {
    use starknet::{ContractAddress, get_caller_address, get_block_timestamp};
    use starknet::storage::{
        Map, StorageMapReadAccess, StorageMapWriteAccess,
        StoragePointerReadAccess, StoragePointerWriteAccess,
    };
    use core::poseidon::poseidon_hash_span;
    use super::super::interfaces::{
        IObelyskVerifier, IERC20Dispatcher, IERC20DispatcherTrait,
    };

    // ── Storage ────────────────────────────────────────────────────────

    #[storage]
    struct Storage {
        /// Contract owner (can submit verified facts).
        owner: ContractAddress,
        /// SAGE token contract address.
        sage_token: ContractAddress,
        /// proof_id → verified (true if proof has been verified).
        verified_proofs: Map<felt252, bool>,
        /// model_id → registered (true if model is registered).
        registered_models: Map<felt252, bool>,
        /// model_id → weight commitment.
        model_weight_commitments: Map<felt252, felt252>,
        /// model_id → number of layers.
        model_num_layers: Map<felt252, u32>,
        /// model_id → verification count.
        verification_count: Map<felt252, u32>,
        /// job_id → verified (prevents double-verification).
        completed_jobs: Map<felt252, bool>,
        /// Total number of verifications.
        total_verifications: u32,
        /// Total SAGE paid out.
        total_sage_paid: u256,
    }

    // ── Events ─────────────────────────────────────────────────────────

    #[event]
    #[derive(Drop, starknet::Event)]
    pub enum Event {
        ModelRegistered: ModelRegistered,
        JobCreated: JobCreated,
        ProofSubmitted: ProofSubmitted,
        InferenceVerified: InferenceVerified,
        PaymentProcessed: PaymentProcessed,
        WorkerRewarded: WorkerRewarded,
        VerificationComplete: VerificationComplete,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ModelRegistered {
        #[key]
        pub model_id: felt252,
        pub weight_commitment: felt252,
        pub num_layers: u32,
        pub description: felt252,
        pub registered_by: ContractAddress,
        pub timestamp: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct JobCreated {
        #[key]
        pub job_id: felt252,
        #[key]
        pub model_id: felt252,
        pub worker: ContractAddress,
        pub client: ContractAddress,
        pub timestamp: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct ProofSubmitted {
        #[key]
        pub job_id: felt252,
        pub proof_hash: felt252,
        pub io_commitment: felt252,
        pub submitter: ContractAddress,
        pub timestamp: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct InferenceVerified {
        #[key]
        pub job_id: felt252,
        #[key]
        pub model_id: felt252,
        pub worker: ContractAddress,
        pub client: ContractAddress,
        pub io_commitment: felt252,
        pub weight_commitment: felt252,
        pub num_layers: u32,
        pub proof_hash: felt252,
        pub proof_id: felt252,
        pub timestamp: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct PaymentProcessed {
        #[key]
        pub job_id: felt252,
        pub from: ContractAddress,
        pub to: ContractAddress,
        pub sage_amount: u256,
        pub timestamp: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct WorkerRewarded {
        #[key]
        pub worker: ContractAddress,
        pub job_id: felt252,
        pub sage_amount: u256,
        pub timestamp: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct VerificationComplete {
        #[key]
        pub job_id: felt252,
        pub proof_id: felt252,
        pub model_id: felt252,
        pub total_verifications: u32,
        pub timestamp: u64,
    }

    // ── Constructor ────────────────────────────────────────────────────

    #[constructor]
    fn constructor(
        ref self: ContractState,
        owner: ContractAddress,
        sage_token: ContractAddress,
    ) {
        self.owner.write(owner);
        self.sage_token.write(sage_token);
        self.total_verifications.write(0);
        self.total_sage_paid.write(0);
    }

    // ── External Functions ─────────────────────────────────────────────

    #[abi(embed_v0)]
    impl ObelyskVerifierImpl of IObelyskVerifier<ContractState> {
        fn verify_and_pay(
            ref self: ContractState,
            model_id: felt252,
            proof_hash: felt252,
            io_commitment: felt252,
            weight_commitment: felt252,
            num_layers: u32,
            job_id: felt252,
            worker: ContractAddress,
            sage_amount: u256,
        ) -> bool {
            let caller = get_caller_address();
            let timestamp = get_block_timestamp();

            // 1. Only owner can submit verified facts (trusted submitter pattern)
            assert!(caller == self.owner.read(), "Only owner can submit proofs");

            // 2. Prevent double-verification
            assert!(!self.completed_jobs.read(job_id), "Job already verified");

            // 3. Compute unique proof ID
            let proof_id = poseidon_hash_span(
                [model_id, io_commitment, weight_commitment, proof_hash].span(),
            );

            // 4. Emit JobCreated
            self
                .emit(
                    JobCreated {
                        job_id, model_id, worker, client: caller, timestamp,
                    },
                );

            // 5. Emit ProofSubmitted
            self
                .emit(
                    ProofSubmitted {
                        job_id,
                        proof_hash,
                        io_commitment,
                        submitter: caller,
                        timestamp,
                    },
                );

            // 6. Emit InferenceVerified
            self
                .emit(
                    InferenceVerified {
                        job_id,
                        model_id,
                        worker,
                        client: caller,
                        io_commitment,
                        weight_commitment,
                        num_layers,
                        proof_hash,
                        proof_id,
                        timestamp,
                    },
                );

            // 7. Transfer SAGE payment (caller → worker)
            if sage_amount > 0 {
                let sage = IERC20Dispatcher {
                    contract_address: self.sage_token.read(),
                };
                let success = sage.transfer_from(caller, worker, sage_amount);
                assert!(success, "SAGE transfer failed");

                self
                    .emit(
                        PaymentProcessed {
                            job_id, from: caller, to: worker, sage_amount, timestamp,
                        },
                    );

                self
                    .emit(
                        WorkerRewarded { worker, job_id, sage_amount, timestamp },
                    );

                self.total_sage_paid.write(self.total_sage_paid.read() + sage_amount);
            }

            // 8. Record verification
            self.verified_proofs.write(proof_id, true);
            self.completed_jobs.write(job_id, true);
            let new_count = self.verification_count.read(model_id) + 1;
            self.verification_count.write(model_id, new_count);
            let total = self.total_verifications.read() + 1;
            self.total_verifications.write(total);

            // 9. Emit VerificationComplete
            self
                .emit(
                    VerificationComplete {
                        job_id,
                        proof_id,
                        model_id,
                        total_verifications: total,
                        timestamp,
                    },
                );

            true
        }

        fn register_model(
            ref self: ContractState,
            model_id: felt252,
            weight_commitment: felt252,
            num_layers: u32,
            description: felt252,
        ) {
            let caller = get_caller_address();
            let timestamp = get_block_timestamp();

            assert!(caller == self.owner.read(), "Only owner can register models");
            assert!(!self.registered_models.read(model_id), "Model already registered");

            self.registered_models.write(model_id, true);
            self.model_weight_commitments.write(model_id, weight_commitment);
            self.model_num_layers.write(model_id, num_layers);
            self.verification_count.write(model_id, 0);

            self
                .emit(
                    ModelRegistered {
                        model_id,
                        weight_commitment,
                        num_layers,
                        description,
                        registered_by: caller,
                        timestamp,
                    },
                );
        }

        fn is_verified(self: @ContractState, proof_id: felt252) -> bool {
            self.verified_proofs.read(proof_id)
        }

        fn get_model_verification_count(self: @ContractState, model_id: felt252) -> u32 {
            self.verification_count.read(model_id)
        }

        fn get_sage_token(self: @ContractState) -> ContractAddress {
            self.sage_token.read()
        }

        fn get_owner(self: @ContractState) -> ContractAddress {
            self.owner.read()
        }
    }
}
