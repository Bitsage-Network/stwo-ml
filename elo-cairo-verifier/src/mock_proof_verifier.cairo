// Minimal verifier mock for VM31 pool integration tests.
// Exposes only the `is_proof_verified` surface that VM31Pool depends on.

use starknet::ContractAddress;
use crate::vm31_merkle::PackedDigest;

#[starknet::interface]
pub trait IMockProofVerifier<TContractState> {
    fn set_verified(ref self: TContractState, proof_hash: felt252, verified: bool);
    fn bind_vm31_public_hash(ref self: TContractState, proof_hash: felt252, vm31_public_hash: PackedDigest);
    fn is_proof_verified(self: @TContractState, proof_hash: felt252) -> bool;
    fn get_vm31_public_hash(self: @TContractState, proof_hash: felt252) -> PackedDigest;
    fn get_owner(self: @TContractState) -> ContractAddress;
}

#[starknet::contract]
pub mod MockProofVerifierContract {
    use super::{ContractAddress, PackedDigest};
    use starknet::get_caller_address;
    use starknet::storage::{Map, StoragePathEntry, StoragePointerReadAccess, StoragePointerWriteAccess};

    #[storage]
    struct Storage {
        owner: ContractAddress,
        verified: Map<felt252, bool>,
        vm31_public_hash: Map<felt252, PackedDigest>,
        vm31_public_hash_set: Map<felt252, bool>,
    }

    #[constructor]
    fn constructor(ref self: ContractState, owner: ContractAddress) {
        self.owner.write(owner);
    }

    #[abi(embed_v0)]
    impl MockProofVerifierImpl of super::IMockProofVerifier<ContractState> {
        fn set_verified(ref self: ContractState, proof_hash: felt252, verified: bool) {
            assert!(get_caller_address() == self.owner.read(), "Mock: owner only");
            self.verified.entry(proof_hash).write(verified);
        }

        fn bind_vm31_public_hash(
            ref self: ContractState, proof_hash: felt252, vm31_public_hash: PackedDigest,
        ) {
            assert!(get_caller_address() == self.owner.read(), "Mock: owner only");
            self.vm31_public_hash.entry(proof_hash).write(vm31_public_hash);
            self.vm31_public_hash_set.entry(proof_hash).write(true);
        }

        fn is_proof_verified(self: @ContractState, proof_hash: felt252) -> bool {
            self.verified.entry(proof_hash).read()
        }

        fn get_vm31_public_hash(self: @ContractState, proof_hash: felt252) -> PackedDigest {
            assert!(self.vm31_public_hash_set.entry(proof_hash).read(), "Mock: VM31 hash not bound");
            self.vm31_public_hash.entry(proof_hash).read()
        }

        fn get_owner(self: @ContractState) -> ContractAddress {
            self.owner.read()
        }
    }
}
