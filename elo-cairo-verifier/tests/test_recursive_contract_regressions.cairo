use snforge_std::{
    declare, DeclareResultTrait, ContractClassTrait, start_cheat_caller_address,
};
use starknet::ContractAddress;

use elo_cairo_verifier::recursive_verifier::{
    IRecursiveVerifierDispatcher, IRecursiveVerifierDispatcherTrait,
};

const OWNER_ADDR: felt252 = 0x1234;
const ATTACKER_ADDR: felt252 = 0xBAD;
const MODEL_ID: felt252 = 0xABC;
const CIRCUIT_HASH: felt252 = 0x123456;
const WEIGHT_ROOT: felt252 = 0x789ABC;
const IO_COMMITMENT: felt252 = 0xDEF123;
const POLICY_COMMITMENT: felt252 = 0x0370c9;
const N_LAYERS: u32 = 30;
const TRACE_LOG_SIZE: u32 = 14;
const N_MATMULS: u32 = 192;
const HIDDEN_SIZE: u32 = 5120;
const NUM_TRANSFORMER_BLOCKS: u32 = 48;

fn deploy_verifier() -> IRecursiveVerifierDispatcher {
    let contract = declare("RecursiveVerifierContract").unwrap().contract_class();
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    let (address, _) = contract.deploy(@array![owner.into()]).unwrap();
    IRecursiveVerifierDispatcher { contract_address: address }
}

fn as_owner(verifier: @IRecursiveVerifierDispatcher) {
    let owner: ContractAddress = OWNER_ADDR.try_into().unwrap();
    start_cheat_caller_address(*verifier.contract_address, owner);
}

fn as_attacker(verifier: @IRecursiveVerifierDispatcher) {
    let attacker: ContractAddress = ATTACKER_ADDR.try_into().unwrap();
    start_cheat_caller_address(*verifier.contract_address, attacker);
}

fn register_default_model(ref verifier: IRecursiveVerifierDispatcher) {
    verifier.register_model_recursive(
        MODEL_ID,
        CIRCUIT_HASH,
        WEIGHT_ROOT,
        POLICY_COMMITMENT,
        N_MATMULS,
        HIDDEN_SIZE,
        NUM_TRANSFORMER_BLOCKS,
    );
}

/// Builds a minimal recursive proof header for tests that are expected to fail
/// before commitment-scheme deserialization.
fn build_fake_recursive_header(
    circuit_hash: felt252,
    io_commitment: felt252,
    weight_root: felt252,
    proof_n_layers: u32,
    final_digest: felt252,
    proof_log_size: u32,
) -> Array<felt252> {
    let mut data: Array<felt252> = array![];
    data.append(0); data.append(0); data.append(0); data.append(circuit_hash);
    data.append(0); data.append(0); data.append(0); data.append(io_commitment);
    data.append(0); data.append(0); data.append(0); data.append(weight_root);
    data.append(proof_n_layers.into());
    data.append(io_commitment);
    data.append(final_digest);
    data.append(proof_log_size.into());
    data.append(0); data.append(0); data.append(0); data.append(0);
    data
}

#[test]
fn test_register_model_current_abi_stores_metadata() {
    let mut verifier = deploy_verifier();
    as_owner(@verifier);
    register_default_model(ref verifier);

    let info = verifier.get_recursive_model_info(MODEL_ID);
    assert(info.circuit_hash == CIRCUIT_HASH, 'bad circuit hash');
    assert(info.weight_super_root == WEIGHT_ROOT, 'bad weight root');
    assert(info.policy_commitment == POLICY_COMMITMENT, 'bad policy');
    assert(info.n_matmuls == N_MATMULS, 'bad n_matmuls');
    assert(info.hidden_size == HIDDEN_SIZE, 'bad hidden_size');
    assert(info.num_transformer_blocks == NUM_TRANSFORMER_BLOCKS, 'bad num_blocks');
    assert(verifier.get_recursive_verification_count(MODEL_ID) == 0, 'count should start at 0');
}

#[test]
#[should_panic(expected: 'Only owner can register')]
fn test_register_model_current_abi_rejects_non_owner() {
    let mut verifier = deploy_verifier();
    as_attacker(@verifier);
    register_default_model(ref verifier);
}

#[test]
#[should_panic(expected: 'Policy mismatch')]
fn test_verify_recursive_rejects_policy_mismatch_before_stark() {
    let mut verifier = deploy_verifier();
    as_owner(@verifier);
    register_default_model(ref verifier);

    let proof = build_fake_recursive_header(
        CIRCUIT_HASH, IO_COMMITMENT, WEIGHT_ROOT, N_LAYERS, 0x1234, TRACE_LOG_SIZE,
    );
    verifier.verify_recursive(
        MODEL_ID,
        IO_COMMITMENT,
        CIRCUIT_HASH,
        WEIGHT_ROOT,
        N_LAYERS,
        N_MATMULS,
        HIDDEN_SIZE,
        NUM_TRANSFORMER_BLOCKS,
        0xBADBAD,
        TRACE_LOG_SIZE,
        proof,
    );
}

#[test]
#[should_panic(expected: 'n_layers mismatch (param/proof)')]
fn test_verify_recursive_rejects_n_layers_header_mismatch() {
    let mut verifier = deploy_verifier();
    as_owner(@verifier);
    register_default_model(ref verifier);

    let proof = build_fake_recursive_header(
        CIRCUIT_HASH, IO_COMMITMENT, WEIGHT_ROOT, N_LAYERS + 1, 0x1234, TRACE_LOG_SIZE,
    );
    verifier.verify_recursive(
        MODEL_ID,
        IO_COMMITMENT,
        CIRCUIT_HASH,
        WEIGHT_ROOT,
        N_LAYERS,
        N_MATMULS,
        HIDDEN_SIZE,
        NUM_TRANSFORMER_BLOCKS,
        POLICY_COMMITMENT,
        TRACE_LOG_SIZE,
        proof,
    );
}

#[test]
#[should_panic(expected: 'trace_log_size mismatch')]
fn test_verify_recursive_rejects_trace_log_size_header_mismatch() {
    let mut verifier = deploy_verifier();
    as_owner(@verifier);
    register_default_model(ref verifier);

    let proof = build_fake_recursive_header(
        CIRCUIT_HASH, IO_COMMITMENT, WEIGHT_ROOT, N_LAYERS, 0x1234, TRACE_LOG_SIZE + 1,
    );
    verifier.verify_recursive(
        MODEL_ID,
        IO_COMMITMENT,
        CIRCUIT_HASH,
        WEIGHT_ROOT,
        N_LAYERS,
        N_MATMULS,
        HIDDEN_SIZE,
        NUM_TRANSFORMER_BLOCKS,
        POLICY_COMMITMENT,
        TRACE_LOG_SIZE,
        proof,
    );
}

#[test]
#[should_panic(expected: 'io_commitment mismatch (proof)')]
fn test_verify_recursive_rejects_io_commitment_header_mismatch() {
    let mut verifier = deploy_verifier();
    as_owner(@verifier);
    register_default_model(ref verifier);

    let proof = build_fake_recursive_header(
        CIRCUIT_HASH, 0xBADBAD, WEIGHT_ROOT, N_LAYERS, 0x1234, TRACE_LOG_SIZE,
    );
    verifier.verify_recursive(
        MODEL_ID,
        IO_COMMITMENT,
        CIRCUIT_HASH,
        WEIGHT_ROOT,
        N_LAYERS,
        N_MATMULS,
        HIDDEN_SIZE,
        NUM_TRANSFORMER_BLOCKS,
        POLICY_COMMITMENT,
        TRACE_LOG_SIZE,
        proof,
    );
}

#[test]
#[should_panic(expected: 'n_matmuls mismatch')]
fn test_verify_recursive_rejects_external_arch_metadata_mismatch() {
    let mut verifier = deploy_verifier();
    as_owner(@verifier);
    register_default_model(ref verifier);

    let proof = build_fake_recursive_header(
        CIRCUIT_HASH, IO_COMMITMENT, WEIGHT_ROOT, N_LAYERS, 0x1234, TRACE_LOG_SIZE,
    );
    verifier.verify_recursive(
        MODEL_ID,
        IO_COMMITMENT,
        CIRCUIT_HASH,
        WEIGHT_ROOT,
        N_LAYERS,
        N_MATMULS + 1,
        HIDDEN_SIZE,
        NUM_TRANSFORMER_BLOCKS,
        POLICY_COMMITMENT,
        TRACE_LOG_SIZE,
        proof,
    );
}
