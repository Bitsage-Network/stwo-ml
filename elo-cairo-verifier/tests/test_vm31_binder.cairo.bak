// Tests for VM31 binder role in SumcheckVerifierContract.

use snforge_std::{declare, DeclareResultTrait, ContractClassTrait, start_cheat_caller_address};
use starknet::ContractAddress;
use elo_cairo_verifier::verifier::{ISumcheckVerifierDispatcher, ISumcheckVerifierDispatcherTrait};
use elo_cairo_verifier::vm31_merkle::pack_m31x8;

fn owner() -> ContractAddress {
    0x1234_felt252.try_into().unwrap()
}

fn binder() -> ContractAddress {
    0x5678_felt252.try_into().unwrap()
}

fn outsider() -> ContractAddress {
    0x9ABC_felt252.try_into().unwrap()
}

fn deploy_verifier() -> ISumcheckVerifierDispatcher {
    let contract = declare("SumcheckVerifierContract").unwrap().contract_class();
    let (address, _) = contract.deploy(@array![owner().into()]).unwrap();
    ISumcheckVerifierDispatcher { contract_address: address }
}

#[test]
fn test_default_vm31_binder_is_owner() {
    let verifier = deploy_verifier();
    assert!(verifier.get_vm31_binder() == owner(), "default binder should be owner");
}

#[test]
fn test_owner_can_set_vm31_binder() {
    let verifier = deploy_verifier();
    start_cheat_caller_address(verifier.contract_address, owner());
    verifier.set_vm31_binder(binder());
    assert!(verifier.get_vm31_binder() == binder(), "binder should update");
}

#[test]
#[should_panic(expected: "Only owner")]
fn test_non_owner_cannot_set_vm31_binder() {
    let verifier = deploy_verifier();
    start_cheat_caller_address(verifier.contract_address, outsider());
    verifier.set_vm31_binder(binder());
}

#[test]
#[should_panic(expected: "VM31 binder cannot be zero")]
fn test_set_vm31_binder_zero_rejected() {
    let verifier = deploy_verifier();
    let zero: ContractAddress = 0_felt252.try_into().unwrap();
    start_cheat_caller_address(verifier.contract_address, owner());
    verifier.set_vm31_binder(zero);
}

#[test]
#[should_panic(expected: "Only vm31 binder")]
fn test_bind_requires_vm31_binder_role() {
    let verifier = deploy_verifier();
    let digest = pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span());

    start_cheat_caller_address(verifier.contract_address, owner());
    verifier.set_vm31_binder(binder());

    // Owner is no longer binder; should fail before proof verification check.
    verifier.bind_vm31_public_hash(0xD001, digest);
}

#[test]
#[should_panic(expected: "Proof hash not verified")]
fn test_binder_cannot_bind_unverified_proof() {
    let verifier = deploy_verifier();
    let digest = pack_m31x8(array![11_u64, 12, 13, 14, 15, 16, 17, 18].span());

    start_cheat_caller_address(verifier.contract_address, owner());
    verifier.set_vm31_binder(binder());

    start_cheat_caller_address(verifier.contract_address, binder());
    verifier.bind_vm31_public_hash(0xD002, digest);
}
