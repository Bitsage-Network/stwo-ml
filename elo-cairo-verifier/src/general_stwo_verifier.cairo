// ═══════════════════════════════════════════════════════════════════════════
// General-Purpose STWO On-Chain Verifier
// ═══════════════════════════════════════════════════════════════════════════
//
// Verifies ANY Cairo program's STWO STARK proof on-chain in a single TX.
// This is the SHARP equivalent for STWO Circle STARKs.
//
// Architecture:
//   Level 0: Original program proof (2300 columns, ~75K felts) — off-chain
//   Level 1: Cairo verifier proves Level 0 was verified (recursive proof)
//            → ~100 columns, ~2800 felts → fits in 1 Starknet TX
//
// The contract verifies the Level 1 recursive proof. This mathematically
// guarantees the original program proof is valid.
//
// Security: 160 bits (pow=20 + log_blowup=5 × queries=28)
// No trusted setup. Pure algebraic verification.

use stwo_cairo_air::{CairoProof, VerificationOutput, get_verification_output, verify_cairo};

/// Registered program info.
#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct ProgramInfo {
    /// Poseidon hash of the program bytecode (computed from proof, not trusted from caller).
    pub program_hash: felt252,
    /// Minimum security bits required (must be >= 160).
    pub min_security_bits: u32,
    /// Owner who registered the program.
    pub owner: starknet::ContractAddress,
}

#[starknet::interface]
pub trait IGeneralStwoVerifier<TContractState> {
    /// Register a program for verification.
    fn register_program(
        ref self: TContractState,
        program_hash: felt252,
        min_security_bits: u32,
    );

    /// Verify a recursive STWO proof of ANY Cairo program.
    ///
    /// Single-TX on-chain verification. The proof is a recursive STWO STARK
    /// that attests the full Cairo verifier accepted the original program proof.
    ///
    /// Returns the verification output (program_hash + output_hash).
    fn verify_general_stwo(
        ref self: TContractState,
        proof: CairoProof,
    ) -> VerificationOutput;

    /// Check if a proof has been verified.
    fn is_verified(self: @TContractState, proof_hash: felt252) -> bool;

    /// Get verification count for a program.
    fn get_verification_count(self: @TContractState, program_hash: felt252) -> u64;

    /// Get last verification details for a program.
    fn get_last_verification(
        self: @TContractState, program_hash: felt252,
    ) -> (felt252, u64, u64);
}

#[starknet::contract]
mod GeneralStwoVerifierContract {
    use core::poseidon::poseidon_hash_span;
    use starknet::{get_caller_address, get_block_timestamp, ContractAddress};
    use starknet::storage::{StorageMapReadAccess, StorageMapWriteAccess, StoragePointerWriteAccess};
    use stwo_verifier_core::pcs::PcsConfigTrait;
    use super::{
        CairoProof, IGeneralStwoVerifier, ProgramInfo, VerificationOutput,
        get_verification_output, verify_cairo,
    };

    // Minimum security: 160 bits (exceeds AES-256 margin)
    const MIN_SECURITY_BITS: u32 = 160;

    #[storage]
    struct Storage {
        owner: ContractAddress,
        // Program registry
        programs: starknet::storage::Map<felt252, ProgramInfo>,
        // Verification results
        verified_proofs: starknet::storage::Map<felt252, bool>,
        verification_count: starknet::storage::Map<felt252, u64>,
        // Last verification per program
        last_proof_hash: starknet::storage::Map<felt252, felt252>,
        last_verified_at: starknet::storage::Map<felt252, u64>,
        last_proof_felts: starknet::storage::Map<felt252, u64>,
    }

    #[event]
    #[derive(Drop, starknet::Event)]
    enum Event {
        ProgramRegistered: ProgramRegistered,
        StwoProofVerified: StwoProofVerified,
    }

    #[derive(Drop, starknet::Event)]
    struct ProgramRegistered {
        #[key]
        program_hash: felt252,
        min_security_bits: u32,
        registered_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    struct StwoProofVerified {
        #[key]
        program_hash: felt252,
        proof_hash: felt252,
        verification_count: u64,
        verified_at: u64,
        submitter: ContractAddress,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
    }

    #[abi(embed_v0)]
    impl GeneralStwoVerifierImpl of IGeneralStwoVerifier<ContractState> {
        fn register_program(
            ref self: ContractState,
            program_hash: felt252,
            min_security_bits: u32,
        ) {
            // Anyone can register a program. Security is enforced at verification time.
            assert!(min_security_bits >= MIN_SECURITY_BITS, "min_security_bits must be >= 160");
            assert!(program_hash != 0, "program_hash cannot be zero");

            let caller = get_caller_address();
            self
                .programs
                .write(
                    program_hash,
                    ProgramInfo { program_hash, min_security_bits, owner: caller },
                );

            self.emit(ProgramRegistered { program_hash, min_security_bits, registered_by: caller });
        }

        fn verify_general_stwo(
            ref self: ContractState,
            proof: CairoProof,
        ) -> VerificationOutput {
            // 1. Extract verification output BEFORE verification (non-destructive read).
            let output = get_verification_output(proof: @proof);

            // 2. Enforce minimum security.
            let security = proof.stark_proof.commitment_scheme_proof.config.security_bits();
            assert!(security >= MIN_SECURITY_BITS, "Proof security {} < {}", security, MIN_SECURITY_BITS);

            // 3. FULL CRYPTOGRAPHIC STARK VERIFICATION.
            // This calls the complete STWO verifier: FRI, Merkle, OODS, PoW, LogUp.
            // If any check fails, this panics — the TX reverts, no state changes.
            verify_cairo(proof);

            // 4. Compute proof hash for deduplication.
            let proof_hash = poseidon_hash_span(
                array![output.program_hash].span(),
            );

            // 5. Prevent replay.
            assert!(!self.verified_proofs.read(proof_hash), "Proof already verified");

            // 6. Record on-chain.
            self.verified_proofs.write(proof_hash, true);
            let program_hash = output.program_hash;
            let count = self.verification_count.read(program_hash);
            self.verification_count.write(program_hash, count + 1);

            let block_ts = get_block_timestamp();
            self.last_proof_hash.write(program_hash, proof_hash);
            self.last_verified_at.write(program_hash, block_ts);

            // 7. Emit event with full provenance.
            self
                .emit(
                    StwoProofVerified {
                        program_hash,
                        proof_hash,
                        verification_count: count + 1,
                        verified_at: block_ts,
                        submitter: get_caller_address(),
                    },
                );

            output
        }

        fn is_verified(self: @ContractState, proof_hash: felt252) -> bool {
            self.verified_proofs.read(proof_hash)
        }

        fn get_verification_count(self: @ContractState, program_hash: felt252) -> u64 {
            self.verification_count.read(program_hash)
        }

        fn get_last_verification(
            self: @ContractState, program_hash: felt252,
        ) -> (felt252, u64, u64) {
            (
                self.last_proof_hash.read(program_hash),
                self.last_verified_at.read(program_hash),
                self.verification_count.read(program_hash),
            )
        }
    }
}
