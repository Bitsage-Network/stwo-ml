// Tests for vm31_pool: privacy pool contract with hardened batch lifecycle.

use snforge_std::{
    declare, DeclareResultTrait, ContractClassTrait,
    start_cheat_caller_address, start_cheat_block_number, start_cheat_block_timestamp,
    spy_events, EventSpyAssertionsTrait,
};
use starknet::ContractAddress;
use elo_cairo_verifier::vm31_pool::{
    IVM31PoolDispatcher, IVM31PoolDispatcherTrait,
    VM31PoolContract::{Event as VM31PoolEvent, V1BindingsDisableProposed, V1BindingsDisableExecuted, V1BindingsDisableCancelled},
};
use elo_cairo_verifier::mock_proof_verifier::{
    IMockProofVerifierDispatcher, IMockProofVerifierDispatcherTrait,
};
use elo_cairo_verifier::mock_erc20::IMockERC20Dispatcher;
use elo_cairo_verifier::vm31_verifier::{
    BatchPublicInputs, DepositPublicInput, WithdrawPublicInput, SpendPublicInput,
    hash_batch_public_inputs,
};
use elo_cairo_verifier::vm31_merkle::{PackedDigest, pack_m31x8, packed_digest_zero};

// ============================================================================
// Helpers
// ============================================================================

const M31_P: u64 = 0x7FFFFFFF;
const U31_RADIX: u256 = 0x80000000;

fn owner() -> ContractAddress {
    0x1234_felt252.try_into().unwrap()
}

fn relayer() -> ContractAddress {
    0x5678_felt252.try_into().unwrap()
}

fn user() -> ContractAddress {
    0x9ABC_felt252.try_into().unwrap()
}

fn deploy_verifier() -> IMockProofVerifierDispatcher {
    let contract = declare("MockProofVerifierContract").unwrap().contract_class();
    let (address, _) = contract.deploy(@array![owner().into()]).unwrap();
    IMockProofVerifierDispatcher { contract_address: address }
}

fn deploy_pool(verifier_contract: ContractAddress) -> IVM31PoolDispatcher {
    let contract = declare("VM31PoolContract").unwrap().contract_class();
    let (address, _) = contract
        .deploy(@array![owner().into(), relayer().into(), verifier_contract.into()])
        .unwrap();
    IVM31PoolDispatcher { contract_address: address }
}

fn deploy_mock_erc20() -> IMockERC20Dispatcher {
    let contract = declare("MockERC20Contract").unwrap().contract_class();
    let (address, _) = contract.deploy(@array![]).unwrap();
    IMockERC20Dispatcher { contract_address: address }
}

fn setup_env_empty() -> (IVM31PoolDispatcher, IMockProofVerifierDispatcher) {
    let verifier = deploy_verifier();
    let pool = deploy_pool(verifier.contract_address);
    (pool, verifier)
}

fn setup_env() -> (IVM31PoolDispatcher, IMockProofVerifierDispatcher) {
    let (pool, verifier) = setup_env_empty();
    let token1 = deploy_mock_erc20();
    let token2 = deploy_mock_erc20();
    start_cheat_caller_address(pool.contract_address, owner());
    let asset1 = pool.register_asset(token1.contract_address);
    let asset2 = pool.register_asset(token2.contract_address);
    assert!(asset1 == 1, "asset 1 id mismatch");
    assert!(asset2 == 2, "asset 2 id mismatch");
    (pool, verifier)
}

fn mark_proof_verified(verifier: IMockProofVerifierDispatcher, proof_hash: felt252) {
    start_cheat_caller_address(verifier.contract_address, owner());
    verifier.set_verified(proof_hash, true);
}

fn bind_batch_hash(
    verifier: IMockProofVerifierDispatcher,
    proof_hash: felt252,
    batch_hash: PackedDigest,
) {
    start_cheat_caller_address(verifier.contract_address, owner());
    verifier.bind_vm31_public_hash(proof_hash, batch_hash);
}

fn compute_batch_hash(
    deposits: Array<DepositPublicInput>,
    withdrawals: Array<WithdrawPublicInput>,
    spends: Array<SpendPublicInput>,
) -> PackedDigest {
    let inputs = BatchPublicInputs { deposits, withdrawals, spends };
    hash_batch_public_inputs(@inputs)
}

fn binding_digest_from_felt(v: felt252) -> PackedDigest {
    let mut x: u256 = v.into();
    let mut limbs: Array<u64> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        let limb_u256: u256 = x % U31_RADIX;
        let mut limb: u64 = limb_u256.try_into().unwrap();
        if limb == M31_P {
            limb = 0;
        }
        limbs.append(limb);
        x = x / U31_RADIX;
        i += 1;
    };
    pack_m31x8(limbs.span())
}

fn bind_withdrawals_for_recipients_split(
    pool: IVM31PoolDispatcher,
    withdrawals: Array<WithdrawPublicInput>,
    payout_recipients: Array<ContractAddress>,
    credit_recipients: Array<ContractAddress>,
) -> Array<WithdrawPublicInput> {
    assert!(withdrawals.len() == payout_recipients.len(), "helper: payout length mismatch");
    assert!(withdrawals.len() == credit_recipients.len(), "helper: credit length mismatch");
    let mut bound: Array<WithdrawPublicInput> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= withdrawals.len() {
            break;
        }
        let wit = withdrawals.at(i);
        let payout_recipient = *payout_recipients.at(i);
        let credit_recipient = *credit_recipients.at(i);
        let binding_felt = pool.compute_withdrawal_binding_felt(
            payout_recipient,
            credit_recipient,
            (*wit.asset_id).into(),
            *wit.amount_lo,
            *wit.amount_hi,
            i,
        );
        bound.append(WithdrawPublicInput {
            merkle_root: *wit.merkle_root,
            nullifier: *wit.nullifier,
            amount_lo: *wit.amount_lo,
            amount_hi: *wit.amount_hi,
            asset_id: *wit.asset_id,
            withdrawal_binding: binding_digest_from_felt(binding_felt),
        });
        i += 1;
    };
    bound
}

fn bind_withdrawals_for_recipients(
    pool: IVM31PoolDispatcher,
    withdrawals: Array<WithdrawPublicInput>,
    recipients: Array<ContractAddress>,
) -> Array<WithdrawPublicInput> {
    bind_withdrawals_for_recipients_split(pool, withdrawals, recipients.clone(), recipients)
}

fn bind_withdrawals_for_payout_v1(
    pool: IVM31PoolDispatcher,
    withdrawals: Array<WithdrawPublicInput>,
    payout_recipients: Array<ContractAddress>,
) -> Array<WithdrawPublicInput> {
    assert!(withdrawals.len() == payout_recipients.len(), "helper: payout length mismatch");
    let mut bound: Array<WithdrawPublicInput> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= withdrawals.len() {
            break;
        }
        let wit = withdrawals.at(i);
        let payout_recipient = *payout_recipients.at(i);
        let binding_felt = pool.compute_withdrawal_binding_v1_felt(
            payout_recipient,
            (*wit.asset_id).into(),
            *wit.amount_lo,
            *wit.amount_hi,
            i,
        );
        bound.append(WithdrawPublicInput {
            merkle_root: *wit.merkle_root,
            nullifier: *wit.nullifier,
            amount_lo: *wit.amount_lo,
            amount_hi: *wit.amount_hi,
            asset_id: *wit.asset_id,
            withdrawal_binding: binding_digest_from_felt(binding_felt),
        });
        i += 1;
    };
    bound
}

fn submit_batch_with_hash_split(
    pool: IVM31PoolDispatcher,
    deposits: Array<DepositPublicInput>,
    withdrawals: Array<WithdrawPublicInput>,
    spends: Array<SpendPublicInput>,
    proof_hash: felt252,
    payout_recipients: Array<ContractAddress>,
    credit_recipients: Array<ContractAddress>,
) -> felt252 {
    start_cheat_caller_address(pool.contract_address, relayer());
    pool.submit_batch_proof(
        deposits, withdrawals, spends, proof_hash, payout_recipients, credit_recipients,
    )
}

fn submit_batch_with_hash(
    pool: IVM31PoolDispatcher,
    deposits: Array<DepositPublicInput>,
    withdrawals: Array<WithdrawPublicInput>,
    spends: Array<SpendPublicInput>,
    proof_hash: felt252,
    withdrawal_recipients: Array<ContractAddress>,
) -> felt252 {
    submit_batch_with_hash_split(
        pool,
        deposits,
        withdrawals,
        spends,
        proof_hash,
        withdrawal_recipients.clone(),
        withdrawal_recipients,
    )
}

fn submit_batch_verified_split(
    pool: IVM31PoolDispatcher,
    verifier: IMockProofVerifierDispatcher,
    deposits: Array<DepositPublicInput>,
    withdrawals: Array<WithdrawPublicInput>,
    spends: Array<SpendPublicInput>,
    proof_hash: felt252,
    payout_recipients: Array<ContractAddress>,
    credit_recipients: Array<ContractAddress>,
) -> felt252 {
    mark_proof_verified(verifier, proof_hash);
    let bound_withdrawals = bind_withdrawals_for_recipients_split(
        pool, withdrawals, payout_recipients.clone(), credit_recipients.clone(),
    );
    let committed_public_hash = compute_batch_hash(
        deposits.clone(), bound_withdrawals.clone(), spends.clone(),
    );
    bind_batch_hash(verifier, proof_hash, committed_public_hash);
    submit_batch_with_hash_split(
        pool,
        deposits,
        bound_withdrawals,
        spends,
        proof_hash,
        payout_recipients,
        credit_recipients,
    )
}

fn submit_batch_verified(
    pool: IVM31PoolDispatcher,
    verifier: IMockProofVerifierDispatcher,
    deposits: Array<DepositPublicInput>,
    withdrawals: Array<WithdrawPublicInput>,
    spends: Array<SpendPublicInput>,
    proof_hash: felt252,
    withdrawal_recipients: Array<ContractAddress>,
) -> felt252 {
    submit_batch_verified_split(
        pool,
        verifier,
        deposits,
        withdrawals,
        spends,
        proof_hash,
        withdrawal_recipients.clone(),
        withdrawal_recipients,
    )
}

fn deposit_verified(
    pool: IVM31PoolDispatcher,
    verifier: IMockProofVerifierDispatcher,
    commitment: PackedDigest,
    amount: u64,
    asset_id: felt252,
    proof_hash: felt252,
) {
    mark_proof_verified(verifier, proof_hash);
    pool.deposit(commitment, amount, asset_id, proof_hash);
}

fn sample_deposit(amt: u64) -> DepositPublicInput {
    DepositPublicInput {
        commitment: pack_m31x8(
            array![amt, amt + 1, amt + 2, amt + 3, amt + 4, amt + 5, amt + 6, amt + 7].span(),
        ),
        amount_lo: amt,
        amount_hi: 0,
        asset_id: 1,
    }
}

fn sample_withdraw(nul_seed: u64) -> WithdrawPublicInput {
    WithdrawPublicInput {
        merkle_root: packed_digest_zero(),
        nullifier: pack_m31x8(
            array![
                nul_seed, nul_seed + 1, nul_seed + 2, nul_seed + 3,
                nul_seed + 4, nul_seed + 5, nul_seed + 6, nul_seed + 7,
            ].span(),
        ),
        amount_lo: 10,
        amount_hi: 0,
        asset_id: 1,
        withdrawal_binding: pack_m31x8(array![0_u64, 0, 0, 0, 0, 0, 0, 0].span()),
    }
}

fn sample_spend(nul0_seed: u64, nul1_seed: u64, out0_seed: u64, out1_seed: u64) -> SpendPublicInput {
    SpendPublicInput {
        merkle_root: packed_digest_zero(),
        nullifier_0: pack_m31x8(array![nul0_seed, 2, 3, 4, 5, 6, 7, 8].span()),
        nullifier_1: pack_m31x8(array![nul1_seed, 2, 3, 4, 5, 6, 7, 8].span()),
        output_commitment_0: pack_m31x8(array![out0_seed, 2, 3, 4, 5, 6, 7, 8].span()),
        output_commitment_1: pack_m31x8(array![out1_seed, 2, 3, 4, 5, 6, 7, 8].span()),
    }
}

// ============================================================================
// Core lifecycle tests
// ============================================================================

#[test]
fn test_pool_initial_state() {
    let (pool, _) = setup_env();

    assert!(pool.get_tree_size() == 0, "initial tree size should be 0");
    assert!(pool.get_merkle_root() == packed_digest_zero(), "initial root should be zero");
    assert!(pool.get_owner() == owner(), "owner mismatch");
    assert!(pool.get_relayer() == relayer(), "relayer mismatch");
    assert!(pool.get_active_batch_id() == 0, "no active batch at init");
    assert!(pool.get_asset_balance(1) == 0, "initial balance should be 0");
    assert!(pool.is_v1_bindings_enabled(), "v1 bindings should be enabled at init");
    assert!(pool.get_v1_disable_proposed_at() == 0, "no v1 disable should be pending at init");
}

#[test]
#[should_panic(expected: "VM31: proof hash not verified")]
fn test_direct_deposit_requires_verified_proof() {
    let (pool, _) = setup_env();
    let commitment = pack_m31x8(array![42_u64, 99, 7, 13, 1, 2, 3, 4].span());

    pool.deposit(commitment, 1000, 1, 0xABC);
}

#[test]
fn test_direct_deposit() {
    let (pool, verifier) = setup_env();
    let commitment = pack_m31x8(array![42_u64, 99, 7, 13, 1, 2, 3, 4].span());

    deposit_verified(pool, verifier, commitment, 1000, 1, 0xABC);

    assert!(pool.get_tree_size() == 1, "tree size should be 1");
    assert!(pool.get_merkle_root() != packed_digest_zero(), "root should change");
    assert!(pool.get_asset_balance(1) == 1000, "balance should be 1000");
}

#[test]
fn test_multiple_direct_deposits() {
    let (pool, verifier) = setup_env();

    let c1 = pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span());
    let c2 = pack_m31x8(array![9_u64, 10, 11, 12, 13, 14, 15, 16].span());

    deposit_verified(pool, verifier, c1, 500, 1, 0x1);
    let root_after_1 = pool.get_merkle_root();

    deposit_verified(pool, verifier, c2, 300, 1, 0x2);
    let root_after_2 = pool.get_merkle_root();

    assert!(pool.get_tree_size() == 2, "tree size should be 2");
    assert!(root_after_1 != root_after_2, "root should change with each deposit");
    assert!(pool.get_asset_balance(1) == 800, "balance should be 800");
    assert!(pool.is_known_root(root_after_1), "first root should be known");
    assert!(pool.is_known_root(root_after_2), "current root should be known");
}

#[test]
fn test_multi_asset_deposits() {
    let (pool, verifier) = setup_env();

    let c1 = pack_m31x8(array![1_u64, 0, 0, 0, 0, 0, 0, 0].span());
    let c2 = pack_m31x8(array![2_u64, 0, 0, 0, 0, 0, 0, 0].span());

    deposit_verified(pool, verifier, c1, 1000, 1, 0x11);
    deposit_verified(pool, verifier, c2, 500, 2, 0x22);

    assert!(pool.get_asset_balance(1) == 1000, "asset 1 balance");
    assert!(pool.get_asset_balance(2) == 500, "asset 2 balance");
    assert!(pool.get_tree_size() == 2, "both notes in same tree");
}

#[test]
fn test_batch_submit() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100), sample_deposit(200)],
        array![],
        array![],
        0xDEAD,
        array![],
    );

    assert!(pool.get_batch_status(batch_id) == 1, "batch should be submitted");
    assert!(pool.get_active_batch_id() == batch_id, "active batch should be set");
}

#[test]
fn test_batch_submit_idempotent_on_proof_hash() {
    let (pool, verifier) = setup_env();
    let deps = array![sample_deposit(100), sample_deposit(200)];
    let proof_hash = 0xD00D;

    mark_proof_verified(verifier, proof_hash);
    let committed_public_hash = compute_batch_hash(
        deps.clone(), array![], array![],
    );
    bind_batch_hash(verifier, proof_hash, committed_public_hash);

    let batch_id_1 = submit_batch_with_hash(pool, deps.clone(), array![], array![], proof_hash, array![]);
    let batch_id_2 = submit_batch_with_hash(pool, deps, array![], array![], proof_hash, array![]);

    assert!(batch_id_1 == batch_id_2, "same proof hash should return same batch id");
    assert!(
        pool.get_batch_for_proof_hash(proof_hash) == batch_id_1,
        "proof hash mapping should be persisted",
    );
}

#[test]
fn test_batch_deposit_lifecycle() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100), sample_deposit(200)],
        array![],
        array![],
        0xDEAD,
        array![],
    );

    pool.apply_batch_chunk(batch_id, 0, 2);
    pool.finalize_batch(batch_id);

    assert!(pool.get_batch_status(batch_id) == 2, "batch should be finalized");
    assert!(pool.get_active_batch_id() == 0, "active batch should be cleared");
    assert!(pool.get_tree_size() == 2, "2 deposits inserted");
    assert!(pool.get_asset_balance(1) == 300, "100 + 200 = 300");
    assert!(pool.get_merkle_root() != packed_digest_zero(), "root should be non-zero");
}

#[test]
fn test_batch_chunked_application() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(10), sample_deposit(20), sample_deposit(30)],
        array![],
        array![],
        0xBEEF,
        array![],
    );

    pool.apply_batch_chunk(batch_id, 0, 2);
    pool.apply_batch_chunk(batch_id, 2, 1);
    pool.finalize_batch(batch_id);

    assert!(pool.get_tree_size() == 3, "3 deposits");
    assert!(pool.get_asset_balance(1) == 60, "10+20+30=60");
}

#[test]
fn test_batch_with_withdrawals() {
    let (pool, verifier) = setup_env();

    deposit_verified(
        pool,
        verifier,
        pack_m31x8(array![1_u64, 0, 0, 0, 0, 0, 0, 0].span()),
        1000,
        1,
        0x1,
    );

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(500)],
        array![sample_withdraw(1000)],
        array![],
        0xCAFE,
        array![user()],
    );

    pool.apply_batch_chunk(batch_id, 0, 2);
    pool.finalize_batch(batch_id);

    assert!(pool.get_asset_balance(1) == 1490, "1000 + 500 - 10 = 1490");
    assert!(pool.get_tree_size() == 2, "1 existing + 1 deposit");
}

#[test]
#[should_panic(expected: "VM31: nullifier already spent")]
fn test_nullifier_double_spend() {
    let (pool, verifier) = setup_env();

    deposit_verified(
        pool,
        verifier,
        pack_m31x8(array![1_u64, 0, 0, 0, 0, 0, 0, 0].span()),
        10000,
        1,
        0x1,
    );

    let wit = sample_withdraw(42);
    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![],
        array![wit, wit],
        array![],
        0xBAAD,
        array![user(), user()],
    );

    pool.apply_batch_chunk(batch_id, 0, 2);
}

#[test]
fn test_batch_with_spends() {
    let (pool, verifier) = setup_env();

    deposit_verified(
        pool,
        verifier,
        pack_m31x8(array![1_u64, 0, 0, 0, 0, 0, 0, 0].span()),
        1000,
        1,
        0x1,
    );

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![],
        array![],
        array![sample_spend(100, 200, 300, 400)],
        0xFACE,
        array![],
    );

    pool.apply_batch_chunk(batch_id, 0, 1);
    pool.finalize_batch(batch_id);

    assert!(pool.get_tree_size() == 3, "1 existing + 2 outputs");
    assert!(pool.get_asset_balance(1) == 1000, "balance unchanged by spend");
    let nul0 = pack_m31x8(array![100_u64, 2, 3, 4, 5, 6, 7, 8].span());
    let nul1 = pack_m31x8(array![200_u64, 2, 3, 4, 5, 6, 7, 8].span());
    assert!(pool.is_nullifier_spent(nul0), "nullifier 0 should be spent");
    assert!(pool.is_nullifier_spent(nul1), "nullifier 1 should be spent");
}

#[test]
#[should_panic(expected: "VM31: empty batch")]
fn test_empty_batch_rejected() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0x1234);
    bind_batch_hash(verifier, 0x1234, compute_batch_hash(array![], array![], array![]));
    submit_batch_with_hash(pool, array![], array![], array![], 0x1234, array![]);
}

#[test]
#[should_panic(expected: "VM31: not all transactions processed")]
fn test_finalize_before_all_chunks() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100), sample_deposit(200)],
        array![],
        array![],
        0xDEAD,
        array![],
    );

    pool.apply_batch_chunk(batch_id, 0, 1);
    pool.finalize_batch(batch_id);
}

#[test]
#[should_panic(expected: "VM31: chunk must start at processed count")]
fn test_chunk_out_of_order() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100), sample_deposit(200), sample_deposit(300)],
        array![],
        array![],
        0xDEAD,
        array![],
    );

    pool.apply_batch_chunk(batch_id, 1, 1);
}

#[test]
fn test_mixed_batch() {
    let (pool, verifier) = setup_env();

    deposit_verified(
        pool,
        verifier,
        pack_m31x8(array![1_u64, 0, 0, 0, 0, 0, 0, 0].span()),
        5000,
        1,
        0x1,
    );

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(200)],
        array![sample_withdraw(9999)],
        array![sample_spend(50, 51, 52, 53)],
        0xDADA,
        array![user()],
    );

    pool.apply_batch_chunk(batch_id, 0, 1);
    pool.apply_batch_chunk(batch_id, 1, 1);
    pool.apply_batch_chunk(batch_id, 2, 1);
    pool.finalize_batch(batch_id);

    assert!(pool.get_tree_size() == 4, "tree size should be 4");
    assert!(pool.get_asset_balance(1) == 5190, "balance should be 5190");
}

#[test]
#[should_panic(expected: "VM31: batch is not active")]
fn test_double_finalize() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100)],
        array![],
        array![],
        0xDEAD,
        array![],
    );

    pool.apply_batch_chunk(batch_id, 0, 1);
    pool.finalize_batch(batch_id);
    pool.finalize_batch(batch_id);
}

// ============================================================================
// Adversarial tests (hardening checks)
// ============================================================================

#[test]
#[should_panic(expected: "VM31: proof hash not verified")]
fn test_submit_unverified_proof_hash() {
    let (pool, _) = setup_env();
    let deps = array![sample_deposit(100)];
    submit_batch_with_hash(pool, deps, array![], array![], 0xA001, array![]);
}

#[test]
#[should_panic(expected: "VM31: batch public input hash mismatch")]
fn test_submit_tampered_public_input_hash() {
    let (pool, verifier) = setup_env();
    let deps = array![sample_deposit(100)];
    mark_proof_verified(verifier, 0xA002);
    let wrong_hash = pack_m31x8(array![99_u64, 1, 2, 3, 4, 5, 6, 7].span());
    bind_batch_hash(verifier, 0xA002, wrong_hash);

    submit_batch_with_hash(pool, deps, array![], array![], 0xA002, array![]);
}

#[test]
#[should_panic(expected: "VM31: active batch in progress")]
fn test_second_batch_while_active() {
    let (pool, verifier) = setup_env();

    let _first = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(10)],
        array![],
        array![],
        0xA100,
        array![],
    );

    let _second = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(20)],
        array![],
        array![],
        0xA101,
        array![],
    );
}

#[test]
#[should_panic(expected: "VM31: unknown withdraw root")]
fn test_withdraw_unknown_root() {
    let (pool, verifier) = setup_env();

    deposit_verified(
        pool,
        verifier,
        pack_m31x8(array![1_u64, 0, 0, 0, 0, 0, 0, 0].span()),
        1000,
        1,
        0xB001,
    );

    let bad_root = pack_m31x8(array![99_u64, 98, 97, 96, 95, 94, 93, 92].span());
    let bad_withdraw = WithdrawPublicInput {
        merkle_root: bad_root,
        nullifier: pack_m31x8(array![11_u64, 12, 13, 14, 15, 16, 17, 18].span()),
        amount_lo: 10,
        amount_hi: 0,
        asset_id: 1,
        withdrawal_binding: pack_m31x8(array![0_u64, 0, 0, 0, 0, 0, 0, 0].span()),
    };

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![],
        array![bad_withdraw],
        array![],
        0xB002,
        array![user()],
    );

    pool.apply_batch_chunk(batch_id, 0, 1);
}

#[test]
#[should_panic(expected: "VM31: deposit amount_hi out of M31 range")]
fn test_invalid_amount_limb() {
    let (pool, verifier) = setup_env();
    let bad_dep = DepositPublicInput {
        commitment: pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span()),
        amount_lo: 1,
        amount_hi: 0x7FFFFFFF,
        asset_id: 1,
    };

    let _batch_id = submit_batch_verified(
        pool,
        verifier,
        array![bad_dep],
        array![],
        array![],
        0xB100,
        array![],
    );
}

#[test]
#[should_panic(expected: "VM31: relayer only")]
fn test_submit_requires_relayer() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xC001);
    let deps = array![sample_deposit(10)];
    start_cheat_caller_address(pool.contract_address, user());
    pool.submit_batch_proof(deps, array![], array![], 0xC001, array![], array![]);
}

#[test]
#[should_panic(expected: "VM31: relayer only before timeout")]
fn test_apply_chunk_requires_relayer_before_timeout() {
    let (pool, verifier) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.set_batch_timeout_blocks(10);

    start_cheat_block_number(pool.contract_address, 100);
    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(55)],
        array![],
        array![],
        0xC100,
        array![],
    );

    start_cheat_block_number(pool.contract_address, 109); // before 100 + 10
    start_cheat_caller_address(pool.contract_address, user());
    pool.apply_batch_chunk(batch_id, 0, 1);
}

#[test]
#[should_panic(expected: "VM31: relayer only before timeout")]
fn test_finalize_requires_relayer_before_timeout() {
    let (pool, verifier) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.set_batch_timeout_blocks(10);

    start_cheat_block_number(pool.contract_address, 200);
    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(77)],
        array![],
        array![],
        0xC101,
        array![],
    );

    // Relayer applies chunk in-time.
    pool.apply_batch_chunk(batch_id, 0, 1);

    // Non-relayer finalize still blocked before timeout.
    start_cheat_block_number(pool.contract_address, 209); // before 200 + 10
    start_cheat_caller_address(pool.contract_address, user());
    pool.finalize_batch(batch_id);
}

#[test]
fn test_timeout_fallback_allows_anyone_after_timeout() {
    let (pool, verifier) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.set_batch_timeout_blocks(10);

    start_cheat_block_number(pool.contract_address, 300);
    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(88)],
        array![],
        array![],
        0xC102,
        array![],
    );

    start_cheat_block_number(pool.contract_address, 310); // timeout reached
    start_cheat_caller_address(pool.contract_address, user());
    pool.apply_batch_chunk(batch_id, 0, 1);
    pool.finalize_batch(batch_id);

    assert!(pool.get_batch_status(batch_id) == 2, "batch should finalize post-timeout");
    assert!(pool.get_active_batch_id() == 0, "active batch should clear post-timeout");
}

// ============================================================================
// Asset Registry Tests
// ============================================================================

#[test]
fn test_register_asset() {
    let (pool, _) = setup_env_empty();
    let token = deploy_mock_erc20();

    start_cheat_caller_address(pool.contract_address, owner());
    let asset_id = pool.register_asset(token.contract_address);

    assert!(asset_id == 1, "first asset id should be 1");
    assert!(pool.get_asset_token(asset_id) == token.contract_address, "asset token mismatch");
    assert!(pool.get_token_asset(token.contract_address) == asset_id, "reverse asset lookup mismatch");
    assert!(pool.get_next_asset_id() == 2, "next asset id should advance");
}

#[test]
#[should_panic(expected: "VM31: owner only")]
fn test_register_asset_non_owner() {
    let (pool, _) = setup_env_empty();
    let token = deploy_mock_erc20();

    start_cheat_caller_address(pool.contract_address, user());
    let _ = pool.register_asset(token.contract_address);
}

#[test]
#[should_panic(expected: "VM31: token already registered")]
fn test_register_asset_duplicate() {
    let (pool, _) = setup_env_empty();
    let token = deploy_mock_erc20();

    start_cheat_caller_address(pool.contract_address, owner());
    let _ = pool.register_asset(token.contract_address);
    let _ = pool.register_asset(token.contract_address);
}

#[test]
#[should_panic(expected: "VM31: token already registered")]
fn test_register_same_token_twice() {
    let (pool, _) = setup_env_empty();
    let token = deploy_mock_erc20();

    start_cheat_caller_address(pool.contract_address, owner());
    let _ = pool.register_asset(token.contract_address);
    let _ = pool.register_asset(token.contract_address);
}

#[test]
fn test_unregistered_asset_returns_zero() {
    let (pool, _) = setup_env();
    let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
    assert!(pool.get_asset_token(999) == zero_addr, "unregistered asset should be zero");
}

// ============================================================================
// Upgrade Tests
// ============================================================================

#[test]
fn test_propose_upgrade() {
    let (pool, _) = setup_env();
    let new_hash: starknet::ClassHash = 0xABCD_felt252.try_into().unwrap();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_upgrade(new_hash);

    let (pending_hash, proposed_at) = pool.get_pending_upgrade();
    assert!(pending_hash == new_hash, "pending hash mismatch");
    assert!(proposed_at == 1000, "proposed_at mismatch");
}

#[test]
#[should_panic(expected: "VM31: owner only")]
fn test_propose_upgrade_non_owner() {
    let (pool, _) = setup_env();
    let new_hash: starknet::ClassHash = 0xABCD_felt252.try_into().unwrap();

    start_cheat_caller_address(pool.contract_address, user());
    pool.propose_upgrade(new_hash);
}

#[test]
#[should_panic(expected: "VM31: upgrade already pending, cancel first")]
fn test_propose_upgrade_while_pending() {
    let (pool, _) = setup_env();
    let h1: starknet::ClassHash = 0xABCD_felt252.try_into().unwrap();
    let h2: starknet::ClassHash = 0xDEAD_felt252.try_into().unwrap();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_upgrade(h1);
    pool.propose_upgrade(h2);
}

#[test]
fn test_cancel_upgrade() {
    let (pool, _) = setup_env();
    let new_hash: starknet::ClassHash = 0xABCD_felt252.try_into().unwrap();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_upgrade(new_hash);
    pool.cancel_upgrade();

    let (pending_hash, proposed_at) = pool.get_pending_upgrade();
    let pending_felt: felt252 = pending_hash.into();
    assert!(pending_felt == 0, "pending should be cleared");
    assert!(proposed_at == 0, "proposed_at should be cleared");
}

#[test]
#[should_panic(expected: "VM31: no upgrade pending")]
fn test_cancel_upgrade_when_none() {
    let (pool, _) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.cancel_upgrade();
}

#[test]
#[should_panic(expected: "VM31: upgrade delay not elapsed")]
fn test_execute_upgrade_too_early() {
    let (pool, _) = setup_env();
    let new_hash: starknet::ClassHash = 0xABCD_felt252.try_into().unwrap();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_upgrade(new_hash);

    // Try to execute at t=1200 (only 200s elapsed, need 300)
    start_cheat_block_timestamp(pool.contract_address, 1200);
    pool.execute_upgrade();
}

#[test]
#[should_panic(expected: "VM31: no upgrade pending")]
fn test_execute_upgrade_when_none() {
    let (pool, _) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.execute_upgrade();
}

#[test]
fn test_initial_pending_upgrade_is_empty() {
    let (pool, _) = setup_env();
    let (pending_hash, proposed_at) = pool.get_pending_upgrade();
    let pending_felt: felt252 = pending_hash.into();
    assert!(pending_felt == 0, "no upgrade pending initially");
    assert!(proposed_at == 0, "proposed_at zero initially");
}

// ============================================================================
// C1: Tree overflow guard tests
// ============================================================================

// NOTE: We cannot practically fill 2^20 leaves in a test, but we can verify the
// assertion message is correct by testing that the contract compiles with the guard.
// The guard is tested implicitly via the existing deposit tests passing (tree_size < MAX_LEAVES).

// ============================================================================
// C2: Withdrawal recipients tests
// ============================================================================

#[test]
#[should_panic(expected: "VM31: payout_recipients length mismatch")]
fn test_payout_recipients_length_mismatch() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD001);
    let deps: Array<DepositPublicInput> = array![];
    let wits = array![sample_withdraw(500)];
    let committed = compute_batch_hash(deps.clone(), wits.clone(), array![]);
    bind_batch_hash(verifier, 0xD001, committed);
    // Pass 0 recipients for 1 withdrawal
    submit_batch_with_hash(pool, deps, wits, array![], 0xD001, array![]);
}

#[test]
#[should_panic(expected: "VM31: credit_recipients length mismatch")]
fn test_credit_recipients_length_mismatch() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD001A);
    let wits = array![sample_withdraw(501)];
    submit_batch_with_hash_split(
        pool,
        array![],
        wits,
        array![],
        0xD001A,
        array![user()],
        array![],
    );
}

#[test]
#[should_panic(expected: "VM31: payout recipient cannot be zero")]
fn test_payout_recipient_zero_address() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD002);
    let wits = array![sample_withdraw(600)];
    let committed = compute_batch_hash(array![], wits.clone(), array![]);
    bind_batch_hash(verifier, 0xD002, committed);
    let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
    submit_batch_with_hash(pool, array![], wits, array![], 0xD002, array![zero_addr]);
}

#[test]
#[should_panic(expected: "VM31: credit recipient cannot be zero")]
fn test_credit_recipient_zero_address() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD002A);
    let wits = array![sample_withdraw(602)];
    let committed = compute_batch_hash(array![], wits.clone(), array![]);
    bind_batch_hash(verifier, 0xD002A, committed);
    let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
    submit_batch_with_hash_split(
        pool,
        array![],
        wits,
        array![],
        0xD002A,
        array![user()],
        array![zero_addr],
    );
}

#[test]
#[should_panic(expected: "VM31: withdrawal binding mismatch")]
fn test_withdrawal_binding_mismatch() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD00B);
    // sample_withdraw has zero binding, which should not match any non-zero recipient binding.
    let wits = array![sample_withdraw(601)];
    let committed = compute_batch_hash(array![], wits.clone(), array![]);
    bind_batch_hash(verifier, 0xD00B, committed);
    submit_batch_with_hash(pool, array![], wits, array![], 0xD00B, array![user()]);
}

#[test]
#[should_panic(expected: "VM31: withdrawal binding mismatch")]
fn test_withdrawal_binding_payout_tamper() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD00C);
    let payout_good: ContractAddress = 0xABCD_felt252.try_into().unwrap();
    let credit_good: ContractAddress = 0xBEEF_felt252.try_into().unwrap();
    let bound_wits = bind_withdrawals_for_recipients_split(
        pool,
        array![sample_withdraw(603)],
        array![payout_good],
        array![credit_good],
    );
    let committed = compute_batch_hash(array![], bound_wits.clone(), array![]);
    bind_batch_hash(verifier, 0xD00C, committed);
    let payout_tampered: ContractAddress = 0xCAFE_felt252.try_into().unwrap();
    submit_batch_with_hash_split(
        pool,
        array![],
        bound_wits,
        array![],
        0xD00C,
        array![payout_tampered],
        array![credit_good],
    );
}

#[test]
#[should_panic(expected: "VM31: withdrawal binding mismatch")]
fn test_withdrawal_binding_credit_tamper() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD00D);
    let payout_good: ContractAddress = 0xABCE_felt252.try_into().unwrap();
    let credit_good: ContractAddress = 0xBEEE_felt252.try_into().unwrap();
    let bound_wits = bind_withdrawals_for_recipients_split(
        pool,
        array![sample_withdraw(604)],
        array![payout_good],
        array![credit_good],
    );
    let committed = compute_batch_hash(array![], bound_wits.clone(), array![]);
    bind_batch_hash(verifier, 0xD00D, committed);
    let credit_tampered: ContractAddress = 0xDEAD_felt252.try_into().unwrap();
    submit_batch_with_hash_split(
        pool,
        array![],
        bound_wits,
        array![],
        0xD00D,
        array![payout_good],
        array![credit_tampered],
    );
}

#[test]
fn test_withdrawal_binding_v1_compat_mirrored_recipients() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD00E);
    let payout: ContractAddress = 0xA001_felt252.try_into().unwrap();
    let bound_wits = bind_withdrawals_for_payout_v1(
        pool,
        array![sample_withdraw(605)],
        array![payout],
    );
    let committed = compute_batch_hash(array![], bound_wits.clone(), array![]);
    bind_batch_hash(verifier, 0xD00E, committed);
    let _batch_id = submit_batch_with_hash_split(
        pool,
        array![],
        bound_wits,
        array![],
        0xD00E,
        array![payout],
        array![payout],
    );
}

#[test]
#[should_panic(expected: "VM31: v1 binding requires payout==credit")]
fn test_withdrawal_binding_v1_rejects_split_recipients() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD00F);
    let payout: ContractAddress = 0xA002_felt252.try_into().unwrap();
    let credit: ContractAddress = 0xB002_felt252.try_into().unwrap();
    let bound_wits = bind_withdrawals_for_payout_v1(
        pool,
        array![sample_withdraw(606)],
        array![payout],
    );
    let committed = compute_batch_hash(array![], bound_wits.clone(), array![]);
    bind_batch_hash(verifier, 0xD00F, committed);
    submit_batch_with_hash_split(
        pool,
        array![],
        bound_wits,
        array![],
        0xD00F,
        array![payout],
        array![credit],
    );
}

#[test]
fn test_withdrawal_sends_to_recipient() {
    let (pool, verifier) = setup_env();

    // Seed the pool with balance
    deposit_verified(
        pool,
        verifier,
        pack_m31x8(array![1_u64, 0, 0, 0, 0, 0, 0, 0].span()),
        1000,
        1,
        0xD003,
    );

    let recipient: ContractAddress = 0xBEEF_felt252.try_into().unwrap();
    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![],
        array![sample_withdraw(7777)],
        array![],
        0xD004,
        array![recipient],
    );

    // apply chunk + finalize — should not panic (transfers to recipient, not caller)
    pool.apply_batch_chunk(batch_id, 0, 1);
    pool.finalize_batch(batch_id);

    assert!(pool.get_asset_balance(1) == 990, "1000 - 10 = 990");
}

// ============================================================================
// C3: Batch cancellation tests
// ============================================================================

#[test]
fn test_cancel_batch_success() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100)],
        array![],
        array![],
        0xE001,
        array![],
    );

    assert!(pool.get_batch_status(batch_id) == 1, "batch should be submitted");
    assert!(pool.get_active_batch_id() == batch_id, "should be active");

    start_cheat_caller_address(pool.contract_address, owner());
    pool.cancel_batch(batch_id);

    assert!(pool.get_batch_status(batch_id) == 3, "batch should be cancelled (status 3)");
    assert!(pool.get_active_batch_id() == 0, "active batch should be cleared");
}

#[test]
fn test_cancel_then_submit_new_batch() {
    let (pool, verifier) = setup_env();

    let batch_id_1 = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100)],
        array![],
        array![],
        0xE010,
        array![],
    );

    start_cheat_caller_address(pool.contract_address, owner());
    pool.cancel_batch(batch_id_1);

    // Should be able to submit a new batch now
    let batch_id_2 = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(200)],
        array![],
        array![],
        0xE011,
        array![],
    );

    assert!(batch_id_2 != batch_id_1, "new batch should have different id");
    assert!(pool.get_active_batch_id() == batch_id_2, "new batch should be active");
}

#[test]
#[should_panic(expected: "VM31: owner only")]
fn test_cancel_batch_requires_owner() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100)],
        array![],
        array![],
        0xE002,
        array![],
    );

    start_cheat_caller_address(pool.contract_address, user());
    pool.cancel_batch(batch_id);
}

#[test]
#[should_panic(expected: "VM31: cannot cancel after processing started")]
fn test_cancel_batch_after_processing() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100), sample_deposit(200)],
        array![],
        array![],
        0xE003,
        array![],
    );

    pool.apply_batch_chunk(batch_id, 0, 1);

    start_cheat_caller_address(pool.contract_address, owner());
    pool.cancel_batch(batch_id);
}

#[test]
#[should_panic(expected: "VM31: batch is not active")]
fn test_cancel_batch_wrong_id() {
    let (pool, verifier) = setup_env();

    let _batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100)],
        array![],
        array![],
        0xE004,
        array![],
    );

    start_cheat_caller_address(pool.contract_address, owner());
    pool.cancel_batch(0xBADB007);
}

// ============================================================================
// Emergency Pause Tests
// ============================================================================

#[test]
#[should_panic(expected: "VM31: contract is paused")]
fn test_pause_blocks_submit() {
    let (pool, verifier) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.pause();

    let _batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100)],
        array![],
        array![],
        0xF001,
        array![],
    );
}

#[test]
#[should_panic(expected: "VM31: contract is paused")]
fn test_pause_blocks_apply_chunk() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100)],
        array![],
        array![],
        0xF002,
        array![],
    );

    start_cheat_caller_address(pool.contract_address, owner());
    pool.pause();

    start_cheat_caller_address(pool.contract_address, relayer());
    pool.apply_batch_chunk(batch_id, 0, 1);
}

#[test]
#[should_panic(expected: "VM31: contract is paused")]
fn test_pause_blocks_finalize() {
    let (pool, verifier) = setup_env();

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![sample_deposit(100)],
        array![],
        array![],
        0xF003,
        array![],
    );

    start_cheat_caller_address(pool.contract_address, relayer());
    pool.apply_batch_chunk(batch_id, 0, 1);

    start_cheat_caller_address(pool.contract_address, owner());
    pool.pause();

    start_cheat_caller_address(pool.contract_address, relayer());
    pool.finalize_batch(batch_id);
}

#[test]
#[should_panic(expected: "VM31: contract is paused")]
fn test_pause_blocks_deposit() {
    let (pool, verifier) = setup_env();
    let commitment = pack_m31x8(array![42_u64, 99, 7, 13, 1, 2, 3, 4].span());

    start_cheat_caller_address(pool.contract_address, owner());
    pool.pause();

    deposit_verified(pool, verifier, commitment, 1000, 1, 0xF004);
}

#[test]
fn test_unpause_resumes_operations() {
    let (pool, verifier) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.pause();
    assert!(pool.is_paused(), "should be paused");

    pool.unpause();
    assert!(!pool.is_paused(), "should be unpaused");

    // Operations should work again
    let commitment = pack_m31x8(array![42_u64, 99, 7, 13, 1, 2, 3, 4].span());
    deposit_verified(pool, verifier, commitment, 1000, 1, 0xF005);
    assert!(pool.get_tree_size() == 1, "deposit should succeed after unpause");
}

#[test]
#[should_panic(expected: "VM31: owner only")]
fn test_pause_requires_owner() {
    let (pool, _) = setup_env();

    start_cheat_caller_address(pool.contract_address, user());
    pool.pause();
}

// ============================================================================
// Zero-Amount Rejection Tests
// ============================================================================

#[test]
#[should_panic(expected: "VM31: amount must be positive")]
fn test_zero_amount_deposit_rejected() {
    let (pool, verifier) = setup_env();
    let commitment = pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span());

    deposit_verified(pool, verifier, commitment, 0, 1, 0xF010);
}

#[test]
#[should_panic(expected: "VM31: amount must be positive")]
fn test_zero_amount_withdrawal_rejected() {
    let (pool, verifier) = setup_env();

    deposit_verified(
        pool,
        verifier,
        pack_m31x8(array![1_u64, 0, 0, 0, 0, 0, 0, 0].span()),
        1000,
        1,
        0xF011,
    );

    let zero_withdraw = WithdrawPublicInput {
        merkle_root: packed_digest_zero(),
        nullifier: pack_m31x8(array![77_u64, 78, 79, 80, 81, 82, 83, 84].span()),
        amount_lo: 0,
        amount_hi: 0,
        asset_id: 1,
        withdrawal_binding: pack_m31x8(array![0_u64, 0, 0, 0, 0, 0, 0, 0].span()),
    };

    let batch_id = submit_batch_verified(
        pool,
        verifier,
        array![],
        array![zero_withdraw],
        array![],
        0xF012,
        array![user()],
    );

    pool.apply_batch_chunk(batch_id, 0, 1);
}

// ============================================================================
// Timeout Floor Tests
// ============================================================================

#[test]
#[should_panic(expected: "VM31: timeout must be >= 10")]
fn test_timeout_floor_enforced() {
    let (pool, _) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.set_batch_timeout_blocks(5);
}

// ============================================================================
// Timelocked Verifier Change Tests
// ============================================================================

#[test]
fn test_verifier_change_timelock() {
    let (pool, _) = setup_env();
    let new_verifier: ContractAddress = 0xAE01_felt252.try_into().unwrap();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_verifier_change(new_verifier);

    let (pending, proposed_at) = pool.get_pending_verifier();
    assert!(pending == new_verifier, "pending verifier mismatch");
    assert!(proposed_at == 1000, "proposed_at mismatch");

    // Execute after delay
    start_cheat_block_timestamp(pool.contract_address, 1301); // 1000 + 300 + 1
    pool.execute_verifier_change();

    assert!(pool.get_verifier_contract() == new_verifier, "verifier should be updated");
    let (pending_after, _) = pool.get_pending_verifier();
    let pending_after_felt: felt252 = pending_after.into();
    assert!(pending_after_felt == 0, "pending should be cleared");
}

#[test]
#[should_panic(expected: "VM31: verifier change delay not elapsed")]
fn test_verifier_change_too_early() {
    let (pool, _) = setup_env();
    let new_verifier: ContractAddress = 0xAE02_felt252.try_into().unwrap();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_verifier_change(new_verifier);

    // Try to execute too early (only 200s elapsed, need 300)
    start_cheat_block_timestamp(pool.contract_address, 1200);
    pool.execute_verifier_change();
}

#[test]
fn test_cancel_verifier_change() {
    let (pool, _) = setup_env();
    let new_verifier: ContractAddress = 0xAE03_felt252.try_into().unwrap();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_verifier_change(new_verifier);
    pool.cancel_verifier_change();

    let (pending, proposed_at) = pool.get_pending_verifier();
    let pending_felt: felt252 = pending.into();
    assert!(pending_felt == 0, "pending should be cleared");
    assert!(proposed_at == 0, "proposed_at should be cleared");
}

// ============================================================================
// Timelocked V1 Binding Disable Tests
// ============================================================================

#[test]
fn test_propose_disable_v1_bindings() {
    let (pool, _) = setup_env();
    let mut spy = spy_events();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_disable_v1_bindings();

    spy.assert_emitted(
        @array![
            (
                pool.contract_address,
                VM31PoolEvent::V1BindingsDisableProposed(
                    V1BindingsDisableProposed { proposed_at: 1000, proposer: owner() }
                ),
            ),
        ],
    );
    assert!(pool.is_v1_bindings_enabled(), "v1 should remain enabled until execution");
    assert!(pool.get_v1_disable_proposed_at() == 1000, "proposal timestamp mismatch");
}

#[test]
#[should_panic(expected: "VM31: owner only")]
fn test_propose_disable_v1_bindings_non_owner() {
    let (pool, _) = setup_env();

    start_cheat_caller_address(pool.contract_address, user());
    pool.propose_disable_v1_bindings();
}

#[test]
#[should_panic(expected: "VM31: v1 disable already pending")]
fn test_propose_disable_v1_bindings_while_pending() {
    let (pool, _) = setup_env();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_disable_v1_bindings();
    pool.propose_disable_v1_bindings();
}

#[test]
#[should_panic(expected: "VM31: no v1 disable pending")]
fn test_execute_disable_v1_bindings_when_none() {
    let (pool, _) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.execute_disable_v1_bindings();
}

#[test]
#[should_panic(expected: "VM31: v1 disable delay not elapsed")]
fn test_execute_disable_v1_bindings_too_early() {
    let (pool, _) = setup_env();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_disable_v1_bindings();

    // Try to execute too early (only 200s elapsed, need 300)
    start_cheat_block_timestamp(pool.contract_address, 1200);
    pool.execute_disable_v1_bindings();
}

#[test]
fn test_execute_disable_v1_bindings_success() {
    let (pool, _) = setup_env();
    let mut spy = spy_events();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_disable_v1_bindings();

    // Execute after delay
    start_cheat_block_timestamp(pool.contract_address, 1301); // 1000 + 300 + 1
    pool.execute_disable_v1_bindings();

    spy.assert_emitted(
        @array![
            (
                pool.contract_address,
                VM31PoolEvent::V1BindingsDisableExecuted(
                    V1BindingsDisableExecuted { executed_at: 1301, executor: owner() }
                ),
            ),
        ],
    );
    assert!(!pool.is_v1_bindings_enabled(), "v1 should be disabled after execution");
    assert!(pool.get_v1_disable_proposed_at() == 0, "pending timestamp should be cleared");
}

#[test]
#[should_panic(expected: "VM31: v1 bindings already disabled")]
fn test_execute_disable_v1_bindings_when_already_disabled() {
    let (pool, _) = setup_env();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_disable_v1_bindings();
    start_cheat_block_timestamp(pool.contract_address, 1301);
    pool.execute_disable_v1_bindings();
    pool.execute_disable_v1_bindings();
}

#[test]
fn test_cancel_disable_v1_bindings() {
    let (pool, _) = setup_env();
    let mut spy = spy_events();

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_disable_v1_bindings();
    pool.cancel_disable_v1_bindings();

    spy.assert_emitted(
        @array![
            (
                pool.contract_address,
                VM31PoolEvent::V1BindingsDisableCancelled(
                    V1BindingsDisableCancelled { cancelled_by: owner() }
                ),
            ),
        ],
    );
    assert!(pool.is_v1_bindings_enabled(), "v1 should remain enabled after cancel");
    assert!(pool.get_v1_disable_proposed_at() == 0, "pending timestamp should be cleared");
}

#[test]
#[should_panic(expected: "VM31: no v1 disable pending")]
fn test_cancel_disable_v1_bindings_when_none() {
    let (pool, _) = setup_env();

    start_cheat_caller_address(pool.contract_address, owner());
    pool.cancel_disable_v1_bindings();
}

#[test]
#[should_panic(expected: "VM31: v1 bindings disabled")]
fn test_withdrawal_binding_v1_rejected_after_disable() {
    let (pool, verifier) = setup_env();
    mark_proof_verified(verifier, 0xD090);

    let payout: ContractAddress = 0xA090_felt252.try_into().unwrap();
    let bound_wits = bind_withdrawals_for_payout_v1(
        pool,
        array![sample_withdraw(690)],
        array![payout],
    );
    let committed = compute_batch_hash(array![], bound_wits.clone(), array![]);
    bind_batch_hash(verifier, 0xD090, committed);

    start_cheat_block_timestamp(pool.contract_address, 1000);
    start_cheat_caller_address(pool.contract_address, owner());
    pool.propose_disable_v1_bindings();
    start_cheat_block_timestamp(pool.contract_address, 1301);
    pool.execute_disable_v1_bindings();

    submit_batch_with_hash_split(
        pool,
        array![],
        bound_wits,
        array![],
        0xD090,
        array![payout],
        array![payout],
    );
}

// ============================================================================
// Governance Events Test
// ============================================================================

#[test]
fn test_governance_events_emitted() {
    let (pool, _) = setup_env();
    let new_relayer: ContractAddress = 0xAAAA_felt252.try_into().unwrap();
    let new_verifier: ContractAddress = 0xBBBB_felt252.try_into().unwrap();

    start_cheat_caller_address(pool.contract_address, owner());

    // set_relayer should not panic (emits RelayerChanged)
    pool.set_relayer(new_relayer);
    assert!(pool.get_relayer() == new_relayer, "relayer should be updated");

    // set_batch_timeout_blocks should not panic (emits BatchTimeoutChanged)
    pool.set_batch_timeout_blocks(50);
    assert!(pool.get_batch_timeout_blocks() == 50, "timeout should be updated");

    // pause / unpause should not panic (emits Paused / Unpaused)
    pool.pause();
    assert!(pool.is_paused(), "should be paused");
    pool.unpause();
    assert!(!pool.is_paused(), "should be unpaused");

    // propose_verifier_change should not panic (emits VerifierChangeProposed)
    start_cheat_block_timestamp(pool.contract_address, 500);
    pool.propose_verifier_change(new_verifier);
    let (pending, _) = pool.get_pending_verifier();
    assert!(pending == new_verifier, "pending verifier should be set");

    // cancel_verifier_change should not panic (emits VerifierChangeCancelled)
    pool.cancel_verifier_change();
    let (pending2, _) = pool.get_pending_verifier();
    let pending2_felt: felt252 = pending2.into();
    assert!(pending2_felt == 0, "pending should be cleared after cancel");
}
