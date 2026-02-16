// VM31 Privacy Pool Contract
//
// On-chain privacy pool using Poseidon2-M31 Merkle tree for note commitments.
// Supports deposits (public → shielded), withdrawals (shielded → public), and
// private transfers (shielded → shielded) via 2-in/2-out spends.
//
// Multi-asset from day one: single global note tree with asset_id inside each
// note commitment. Per-asset vault accounting tracks deposits/withdrawals.
//
// ERC20 custody: the pool holds actual tokens via IERC20.transferFrom on deposit
// and IERC20.transfer on withdrawal. Asset registry maps internal asset IDs to
// ERC20 token contract addresses.
//
// Batch Processing Protocol (3-step):
//   1. submit_batch_proof(public_inputs, proof_hash) — verifies STARK, stores batch metadata
//   2. apply_batch_chunk(batch_id, start, count) — processes chunk of leaves (Merkle updates)
//   3. finalize_batch(batch_id) — checks all chunks processed, updates root
//
// Upgradability: 5-minute timelocked upgrade via propose_upgrade / execute_upgrade.
//
// Gas amortization: Merkle tree updates are the expensive part (20 hashes per insert).
// Chunking spreads this across multiple transactions (e.g., 50 inserts per chunk).

use crate::vm31_merkle::{
    PackedDigest, packed_digest_zero, pack_m31x8, unpack_m31x8,
    poseidon2_m31_compress_packed,
    MERKLE_DEPTH,
};
use crate::vm31_verifier::{
    BatchPublicInputs, DepositPublicInput, WithdrawPublicInput, SpendPublicInput,
    verify_batch_public_inputs,
    reconstruct_amount,
};
use crate::verifier::ISumcheckVerifierDispatcher;
use starknet::{ContractAddress, ClassHash};

// ============================================================================
// ERC20 Interface (subset needed for custody)
// ============================================================================

#[starknet::interface]
pub trait IERC20<TContractState> {
    fn transfer(ref self: TContractState, recipient: ContractAddress, amount: u256) -> bool;
    fn transfer_from(
        ref self: TContractState,
        sender: ContractAddress,
        recipient: ContractAddress,
        amount: u256,
    ) -> bool;
    fn balance_of(self: @TContractState, account: ContractAddress) -> u256;
}

// Minimum delay (seconds) between propose_upgrade and execute_upgrade.
pub const UPGRADE_DELAY: u64 = 300; // 5 minutes
// After this many blocks from batch submission, anyone can finish chunk apply/finalize.
pub const DEFAULT_BATCH_TIMEOUT_BLOCKS: u64 = 120;

// ============================================================================
// Interface
// ============================================================================

#[starknet::interface]
pub trait IVM31Pool<TContractState> {
    // ── Batch Lifecycle ──

    // Step 1: Submit a batch proof with its public inputs.
    // The caller provides the batch public inputs and a reference to a verified STARK proof.
    // For withdrawals, recipient binding is split:
    //   - payout_recipients: on-chain ERC20 payout targets used during batch execution
    //   - credit_recipients: app/bridge credit targets committed inside withdrawal_binding
    // Returns a batch_id for subsequent chunk application.
    fn submit_batch_proof(
        ref self: TContractState,
        deposits: Array<DepositPublicInput>,
        withdrawals: Array<WithdrawPublicInput>,
        spends: Array<SpendPublicInput>,
        proof_stark_hash: felt252,
        payout_recipients: Array<ContractAddress>,
        credit_recipients: Array<ContractAddress>,
    ) -> felt252;

    // Step 2: Apply a chunk of the batch (process Merkle insertions + nullifiers).
    // start: index of first transaction in this chunk
    // count: number of transactions to process in this chunk
    // Access: relayer-only before timeout; permissionless after timeout.
    fn apply_batch_chunk(
        ref self: TContractState,
        batch_id: felt252,
        start: u32,
        count: u32,
    );

    // Step 3: Finalize the batch after all chunks are applied.
    // Verifies all transactions were processed and updates the canonical root.
    // Access: relayer-only before timeout; permissionless after timeout.
    fn finalize_batch(
        ref self: TContractState,
        batch_id: felt252,
    );

    // ── Direct Operations (single-tx, non-batched) ──

    // Deposit: submit a note commitment with a verified proof.
    // For small one-off deposits that don't justify batch overhead.
    fn deposit(
        ref self: TContractState,
        commitment: PackedDigest,
        amount: u64,
        asset_id: felt252,
        proof_hash: felt252,
    );

    // ── View Functions ──

    fn get_merkle_root(self: @TContractState) -> PackedDigest;
    fn get_tree_size(self: @TContractState) -> u64;
    fn is_nullifier_spent(self: @TContractState, nullifier: PackedDigest) -> bool;
    fn get_asset_balance(self: @TContractState, asset_id: felt252) -> u64;
    fn is_known_root(self: @TContractState, root: PackedDigest) -> bool;
    fn get_batch_status(self: @TContractState, batch_id: felt252) -> u8;
    fn get_batch_for_proof_hash(self: @TContractState, proof_stark_hash: felt252) -> felt252;
    fn get_batch_total_txs(self: @TContractState, batch_id: felt252) -> u32;
    fn get_batch_processed_count(self: @TContractState, batch_id: felt252) -> u32;
    fn get_batch_withdrawal_binding(
        self: @TContractState,
        batch_id: felt252,
        withdraw_idx: u32,
    ) -> PackedDigest;
    fn get_batch_withdrawal_binding_felt(
        self: @TContractState,
        batch_id: felt252,
        withdraw_idx: u32,
    ) -> felt252;
    fn compute_withdrawal_binding_felt(
        self: @TContractState,
        payout_recipient: ContractAddress,
        credit_recipient: ContractAddress,
        asset_id: felt252,
        amount_lo: u64,
        amount_hi: u64,
        withdraw_idx: u32,
    ) -> felt252;
    fn compute_withdrawal_binding_v1_felt(
        self: @TContractState,
        payout_recipient: ContractAddress,
        asset_id: felt252,
        amount_lo: u64,
        amount_hi: u64,
        withdraw_idx: u32,
    ) -> felt252;
    fn get_batch_timeout_blocks(self: @TContractState) -> u64;
    fn get_active_batch_id(self: @TContractState) -> felt252;
    fn get_owner(self: @TContractState) -> ContractAddress;
    fn get_relayer(self: @TContractState) -> ContractAddress;
    fn get_verifier_contract(self: @TContractState) -> ContractAddress;
    fn get_asset_token(self: @TContractState, asset_id: felt252) -> ContractAddress;
    fn get_token_asset(self: @TContractState, token_address: ContractAddress) -> felt252;
    fn get_next_asset_id(self: @TContractState) -> felt252;
    fn get_pending_upgrade(self: @TContractState) -> (ClassHash, u64);
    fn is_paused(self: @TContractState) -> bool;
    fn get_pending_verifier(self: @TContractState) -> (ContractAddress, u64);
    fn is_v1_bindings_enabled(self: @TContractState) -> bool;
    fn get_v1_disable_proposed_at(self: @TContractState) -> u64;

    // ── Admin ──

    fn set_relayer(ref self: TContractState, relayer: ContractAddress);
    fn cancel_batch(ref self: TContractState, batch_id: felt252);
    fn set_batch_timeout_blocks(ref self: TContractState, timeout_blocks: u64);
    fn register_asset(ref self: TContractState, token_address: ContractAddress) -> felt252;
    fn pause(ref self: TContractState);
    fn unpause(ref self: TContractState);

    // ── Timelocked Verifier Change ──

    fn propose_verifier_change(ref self: TContractState, new_verifier: ContractAddress);
    fn execute_verifier_change(ref self: TContractState);
    fn cancel_verifier_change(ref self: TContractState);

    // ── Timelocked V1 Binding Disable ──

    fn propose_disable_v1_bindings(ref self: TContractState);
    fn execute_disable_v1_bindings(ref self: TContractState);
    fn cancel_disable_v1_bindings(ref self: TContractState);

    // ── Upgradability ──

    fn propose_upgrade(ref self: TContractState, new_class_hash: ClassHash);
    fn execute_upgrade(ref self: TContractState);
    fn cancel_upgrade(ref self: TContractState);
}

// ============================================================================
// Contract
// ============================================================================

#[starknet::contract]
pub mod VM31PoolContract {
    use super::{
        PackedDigest, packed_digest_zero, pack_m31x8, unpack_m31x8,
        poseidon2_m31_compress_packed,
        BatchPublicInputs, DepositPublicInput, WithdrawPublicInput, SpendPublicInput,
        verify_batch_public_inputs,
        reconstruct_amount,
        ISumcheckVerifierDispatcher,
        IERC20Dispatcher, IERC20DispatcherTrait,
        MERKLE_DEPTH, UPGRADE_DELAY, DEFAULT_BATCH_TIMEOUT_BLOCKS,
    };
    use crate::verifier::ISumcheckVerifierDispatcherTrait;
    use starknet::{
        ClassHash, ContractAddress,
        get_caller_address, get_block_number, get_block_timestamp, get_contract_address,
    };
    use starknet::storage::{
        Map, StoragePathEntry, StoragePointerReadAccess, StoragePointerWriteAccess,
    };
    use core::poseidon::poseidon_hash_span;

    // Batch status constants
    const BATCH_STATUS_NONE: u8 = 0;
    const BATCH_STATUS_SUBMITTED: u8 = 1;
    const BATCH_STATUS_FINALIZED: u8 = 2;
    const BATCH_STATUS_CANCELLED: u8 = 3;

    // Root history ring buffer size
    const ROOT_HISTORY_SIZE: u32 = 256;

    // Maximum chunk size (gas limit safety)
    const MAX_CHUNK_SIZE: u32 = 64;

    // M31 modulus p = 2^31 - 1. Canonical limbs must be in [0, p).
    const M31_P: u64 = 0x7FFFFFFF;
    const U31_RADIX: u256 = 0x80000000; // 2^31
    const WITHDRAW_BINDING_DOMAIN_V1: felt252 = 0x564D33315F5744525F42494E445F5631;
    const WITHDRAW_BINDING_DOMAIN_V2: felt252 = 0x564D33315F5744525F42494E445F5632;

    // Maximum number of leaves the Merkle tree can hold (2^MERKLE_DEPTH).
    const MAX_LEAVES: u64 = 1_048_576;

    // ── Storage ──

    #[storage]
    struct Storage {
        // Owner
        owner: ContractAddress,
        relayer: ContractAddress,
        verifier_contract: ContractAddress,

        // Merkle tree state
        merkle_root: PackedDigest,
        tree_size: u64,
        current_root_seq: u64,
        // Sparse Merkle tree: leaf[index] = commitment
        // Internal nodes computed on-the-fly during insertion
        merkle_leaves: Map<u64, PackedDigest>,
        // Cached internal node hashes (level, index) → digest
        // Level 0 = leaf level, level MERKLE_DEPTH = root
        merkle_nodes: Map<(u32, u64), PackedDigest>,

        // Nullifier set
        nullifiers: Map<felt252, bool>,

        // Per-asset vault accounting
        asset_balances: Map<felt252, u64>,

        // Root history ring buffer
        root_history: Map<u32, PackedDigest>,
        root_history_index: u32,
        root_first_seen_seq: Map<felt252, u64>,

        // Batch state
        current_batch_id: felt252,
        batch_timeout_blocks: u64,
        next_batch_nonce: u64,
        batch_status: Map<felt252, u8>,
        batch_submitter: Map<felt252, ContractAddress>,
        batch_stark_hash: Map<felt252, felt252>,
        batch_for_proof_hash: Map<felt252, felt252>,
        batch_public_hash: Map<felt252, PackedDigest>,
        batch_submit_root_seq: Map<felt252, u64>,
        batch_total_txs: Map<felt252, u32>,
        batch_processed_count: Map<felt252, u32>,
        batch_block: Map<felt252, u64>,

        // Batch deposit data (stored for chunk processing)
        batch_n_deposits: Map<felt252, u32>,
        batch_deposit_commitment: Map<(felt252, u32), PackedDigest>,
        batch_deposit_amount_lo: Map<(felt252, u32), u64>,
        batch_deposit_amount_hi: Map<(felt252, u32), u64>,
        batch_deposit_asset: Map<(felt252, u32), u64>,

        // Batch withdrawal data
        batch_n_withdrawals: Map<felt252, u32>,
        batch_withdraw_merkle_root: Map<(felt252, u32), PackedDigest>,
        batch_withdraw_nullifier: Map<(felt252, u32), PackedDigest>,
        batch_withdraw_amount_lo: Map<(felt252, u32), u64>,
        batch_withdraw_amount_hi: Map<(felt252, u32), u64>,
        batch_withdraw_asset: Map<(felt252, u32), u64>,
        batch_withdraw_binding: Map<(felt252, u32), PackedDigest>,
        // v1/v2 note: `batch_withdraw_recipient` remains the payout recipient for storage-key compatibility.
        batch_withdraw_recipient: Map<(felt252, u32), ContractAddress>,
        batch_withdraw_credit_recipient: Map<(felt252, u32), ContractAddress>,

        // Batch spend data
        batch_n_spends: Map<felt252, u32>,
        batch_spend_merkle_root: Map<(felt252, u32), PackedDigest>,
        batch_spend_nullifier_0: Map<(felt252, u32), PackedDigest>,
        batch_spend_nullifier_1: Map<(felt252, u32), PackedDigest>,
        batch_spend_output_0: Map<(felt252, u32), PackedDigest>,
        batch_spend_output_1: Map<(felt252, u32), PackedDigest>,

        // Pending Merkle root during batch application
        batch_pending_root: Map<felt252, PackedDigest>,
        batch_pending_tree_size: Map<felt252, u64>,

        // Asset registry: internal asset_id <-> ERC20 token contract
        next_asset_id: u64,
        asset_token: Map<felt252, ContractAddress>,
        token_asset: Map<ContractAddress, felt252>,

        // Reentrancy guard
        reentrancy_locked: bool,

        // Timelocked upgradability
        pending_upgrade: ClassHash,
        upgrade_proposed_at: u64,

        // Emergency pause
        paused: bool,

        // Timelocked verifier change
        pending_verifier: ContractAddress,
        verifier_change_proposed_at: u64,

        // Timelocked kill-switch for legacy V1 withdrawal bindings
        v1_bindings_enabled: bool,
        v1_disable_proposed_at: u64,
    }

    // ── Events ──

    #[event]
    #[derive(Drop, starknet::Event)]
    pub enum Event {
        BatchSubmitted: BatchSubmitted,
        BatchChunkApplied: BatchChunkApplied,
        BatchFinalized: BatchFinalized,
        NoteInserted: NoteInserted,
        NullifierSpent: NullifierSpent,
        DepositProcessed: DepositProcessed,
        WithdrawProcessed: WithdrawProcessed,
        AssetRegistered: AssetRegistered,
        UpgradeProposed: UpgradeProposed,
        UpgradeExecuted: UpgradeExecuted,
        BatchCancelled: BatchCancelled,
        UpgradeCancelled: UpgradeCancelled,
        RelayerChanged: RelayerChanged,
        VerifierChangeProposed: VerifierChangeProposed,
        VerifierChangeExecuted: VerifierChangeExecuted,
        VerifierChangeCancelled: VerifierChangeCancelled,
        V1BindingsDisableProposed: V1BindingsDisableProposed,
        V1BindingsDisableExecuted: V1BindingsDisableExecuted,
        V1BindingsDisableCancelled: V1BindingsDisableCancelled,
        BatchTimeoutChanged: BatchTimeoutChanged,
        Paused: Paused,
        Unpaused: Unpaused,
    }

    #[derive(Drop, starknet::Event)]
    pub struct BatchSubmitted {
        pub batch_id: felt252,
        pub submitter: ContractAddress,
        pub n_deposits: u32,
        pub n_withdrawals: u32,
        pub n_spends: u32,
        pub block_number: u64,
    }

    #[derive(Drop, starknet::Event)]
    pub struct BatchChunkApplied {
        pub batch_id: felt252,
        pub start: u32,
        pub count: u32,
        pub processed_total: u32,
    }

    #[derive(Drop, starknet::Event)]
    pub struct BatchFinalized {
        pub batch_id: felt252,
        pub new_root: PackedDigest,
        pub new_tree_size: u64,
        pub total_txs: u32,
    }

    #[derive(Drop, starknet::Event)]
    pub struct NoteInserted {
        pub leaf_index: u64,
        pub commitment: PackedDigest,
    }

    #[derive(Drop, starknet::Event)]
    pub struct NullifierSpent {
        pub nullifier_hash: felt252,
    }

    #[derive(Drop, starknet::Event)]
    pub struct DepositProcessed {
        pub commitment: PackedDigest,
        pub amount: u64,
        pub asset_id: felt252,
    }

    #[derive(Drop, starknet::Event)]
    pub struct WithdrawProcessed {
        pub nullifier: PackedDigest,
        pub amount: u64,
        pub asset_id: felt252,
    }

    #[derive(Drop, starknet::Event)]
    pub struct AssetRegistered {
        pub asset_id: felt252,
        pub token_address: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct BatchCancelled {
        pub batch_id: felt252,
        pub cancelled_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct UpgradeProposed {
        pub new_class_hash: ClassHash,
        pub proposed_at: u64,
        pub proposer: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct UpgradeExecuted {
        pub new_class_hash: ClassHash,
        pub executed_at: u64,
        pub executor: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct UpgradeCancelled {
        pub cancelled_class_hash: ClassHash,
        pub cancelled_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct RelayerChanged {
        pub old_relayer: ContractAddress,
        pub new_relayer: ContractAddress,
        pub changed_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct VerifierChangeProposed {
        pub new_verifier: ContractAddress,
        pub proposed_at: u64,
        pub proposer: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct VerifierChangeExecuted {
        pub new_verifier: ContractAddress,
        pub executed_at: u64,
        pub executor: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct VerifierChangeCancelled {
        pub cancelled_verifier: ContractAddress,
        pub cancelled_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct V1BindingsDisableProposed {
        pub proposed_at: u64,
        pub proposer: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct V1BindingsDisableExecuted {
        pub executed_at: u64,
        pub executor: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct V1BindingsDisableCancelled {
        pub cancelled_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct BatchTimeoutChanged {
        pub old_timeout: u64,
        pub new_timeout: u64,
        pub changed_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct Paused {
        pub paused_by: ContractAddress,
    }

    #[derive(Drop, starknet::Event)]
    pub struct Unpaused {
        pub unpaused_by: ContractAddress,
    }

    // ── Constructor ──

    #[constructor]
    fn constructor(
        ref self: ContractState,
        owner: ContractAddress,
        relayer: ContractAddress,
        verifier_contract: ContractAddress,
    ) {
        self.owner.write(owner);
        self.relayer.write(relayer);
        self.verifier_contract.write(verifier_contract);
        self.merkle_root.write(packed_digest_zero());
        self.tree_size.write(0);
        self.current_root_seq.write(1);
        self.current_batch_id.write(0);
        self.batch_timeout_blocks.write(DEFAULT_BATCH_TIMEOUT_BLOCKS);
        self.next_batch_nonce.write(1);
        self.next_asset_id.write(1); // 0 = unregistered sentinel
        self.v1_bindings_enabled.write(true);
        self.v1_disable_proposed_at.write(0);
        self.root_history_index.write(0);
        // Store initial root in history
        let init_root = packed_digest_zero();
        self.root_history.entry(0).write(init_root);
        self.root_first_seen_seq.entry(pack_digest_to_felt(init_root)).write(1);
    }

    // ── Implementation ──

    #[abi(embed_v0)]
    impl VM31PoolImpl of super::IVM31Pool<ContractState> {

        // Step 1: Submit batch proof
        fn submit_batch_proof(
            ref self: ContractState,
            deposits: Array<DepositPublicInput>,
            withdrawals: Array<WithdrawPublicInput>,
            spends: Array<SpendPublicInput>,
            proof_stark_hash: felt252,
            payout_recipients: Array<ContractAddress>,
            credit_recipients: Array<ContractAddress>,
        ) -> felt252 {
            assert!(!self.paused.read(), "VM31: contract is paused");
            let caller = get_caller_address();
            let block = get_block_number();
            assert!(caller == self.relayer.read(), "VM31: relayer only");

            // Idempotency: if this proof hash already has a batch, return it.
            let existing_batch = self.batch_for_proof_hash.entry(proof_stark_hash).read();
            if existing_batch != 0 {
                let existing_status = self.batch_status.entry(existing_batch).read();
                assert!(existing_status != BATCH_STATUS_NONE, "VM31: stale proof mapping");
                return existing_batch;
            }
            assert!(self.current_batch_id.read() == 0, "VM31: active batch in progress");

            let verifier_addr = self.verifier_contract.read();
            let verifier_addr_felt: felt252 = verifier_addr.into();
            assert!(verifier_addr_felt != 0, "VM31: verifier contract not set");
            let verifier = ISumcheckVerifierDispatcher { contract_address: verifier_addr };
            assert!(
                verifier.is_proof_verified(proof_stark_hash),
                "VM31: proof hash not verified"
            );

            let n_dep = deposits.len();
            let n_wit = withdrawals.len();
            let n_spe = spends.len();
            let total = n_dep + n_wit + n_spe;
            assert!(total > 0, "VM31: empty batch");
            assert!(
                payout_recipients.len() == n_wit,
                "VM31: payout_recipients length mismatch"
            );
            assert!(
                credit_recipients.len() == n_wit,
                "VM31: credit_recipients length mismatch"
            );
            let committed_public_hash = verifier.get_vm31_public_hash(proof_stark_hash);

            // Generate batch ID
            let nonce = self.next_batch_nonce.read();
            let batch_id = poseidon_hash_span(
                array![nonce.into(), proof_stark_hash, caller.into()].span()
            );
            self.next_batch_nonce.write(nonce + 1);

            // Hash public inputs and bind them to the proof-committed hash.
            let batch_inputs = BatchPublicInputs {
                deposits: deposits.clone(),
                withdrawals: withdrawals.clone(),
                spends: spends.clone(),
            };
            let pub_hash = verify_batch_public_inputs(@batch_inputs, committed_public_hash);

            // Store batch metadata
            self.current_batch_id.write(batch_id);
            self.batch_status.entry(batch_id).write(BATCH_STATUS_SUBMITTED);
            self.batch_submitter.entry(batch_id).write(caller);
            self.batch_stark_hash.entry(batch_id).write(proof_stark_hash);
            self.batch_for_proof_hash.entry(proof_stark_hash).write(batch_id);
            self.batch_public_hash.entry(batch_id).write(pub_hash);
            self.batch_submit_root_seq.entry(batch_id).write(self.current_root_seq.read());
            self.batch_total_txs.entry(batch_id).write(total);
            self.batch_processed_count.entry(batch_id).write(0);
            self.batch_block.entry(batch_id).write(block);

            // Store deposit data for chunk processing
            self.batch_n_deposits.entry(batch_id).write(n_dep);
            let mut i: u32 = 0;
            loop {
                if i >= n_dep {
                    break;
                }
                let dep = deposits.at(i);
                assert!(*dep.amount_lo < M31_P, "VM31: deposit amount_lo out of M31 range");
                assert!(*dep.amount_hi < M31_P, "VM31: deposit amount_hi out of M31 range");
                assert!(*dep.asset_id < M31_P, "VM31: deposit asset_id out of M31 range");
                self.batch_deposit_commitment.entry((batch_id, i)).write(*dep.commitment);
                self.batch_deposit_amount_lo.entry((batch_id, i)).write(*dep.amount_lo);
                self.batch_deposit_amount_hi.entry((batch_id, i)).write(*dep.amount_hi);
                self.batch_deposit_asset.entry((batch_id, i)).write(*dep.asset_id);
                i += 1;
            };

            // Store withdrawal data
            self.batch_n_withdrawals.entry(batch_id).write(n_wit);
            let mut i: u32 = 0;
            loop {
                if i >= n_wit {
                    break;
                }
                let wit = withdrawals.at(i);
                assert!(*wit.amount_lo < M31_P, "VM31: withdraw amount_lo out of M31 range");
                assert!(*wit.amount_hi < M31_P, "VM31: withdraw amount_hi out of M31 range");
                assert!(*wit.asset_id < M31_P, "VM31: withdraw asset_id out of M31 range");
                self.batch_withdraw_merkle_root.entry((batch_id, i)).write(*wit.merkle_root);
                self.batch_withdraw_nullifier.entry((batch_id, i)).write(*wit.nullifier);
                self.batch_withdraw_amount_lo.entry((batch_id, i)).write(*wit.amount_lo);
                self.batch_withdraw_amount_hi.entry((batch_id, i)).write(*wit.amount_hi);
                self.batch_withdraw_asset.entry((batch_id, i)).write(*wit.asset_id);
                self.batch_withdraw_binding.entry((batch_id, i)).write(*wit.withdrawal_binding);
                let payout_recipient = *payout_recipients.at(i);
                let credit_recipient = *credit_recipients.at(i);
                let payout_recipient_felt: felt252 = payout_recipient.into();
                let credit_recipient_felt: felt252 = credit_recipient.into();
                assert!(payout_recipient_felt != 0, "VM31: payout recipient cannot be zero");
                assert!(credit_recipient_felt != 0, "VM31: credit recipient cannot be zero");
                let expected_binding_v2 = compute_withdrawal_binding_digest_v2(
                    payout_recipient,
                    credit_recipient,
                    (*wit.asset_id).into(),
                    *wit.amount_lo,
                    *wit.amount_hi,
                    i,
                );
                let expected_binding_v1 = compute_withdrawal_binding_digest_v1(
                    payout_recipient,
                    (*wit.asset_id).into(),
                    *wit.amount_lo,
                    *wit.amount_hi,
                    i,
                );
                if *wit.withdrawal_binding != expected_binding_v2 {
                    assert!(self.v1_bindings_enabled.read(), "VM31: v1 bindings disabled");
                    assert!(
                        *wit.withdrawal_binding == expected_binding_v1,
                        "VM31: withdrawal binding mismatch"
                    );
                    assert!(
                        payout_recipient == credit_recipient,
                        "VM31: v1 binding requires payout==credit"
                    );
                }
                self.batch_withdraw_recipient.entry((batch_id, i)).write(payout_recipient);
                self.batch_withdraw_credit_recipient.entry((batch_id, i)).write(credit_recipient);
                i += 1;
            };

            // Store spend data
            self.batch_n_spends.entry(batch_id).write(n_spe);
            let mut i: u32 = 0;
            loop {
                if i >= n_spe {
                    break;
                }
                let spe = spends.at(i);
                self.batch_spend_merkle_root.entry((batch_id, i)).write(*spe.merkle_root);
                self.batch_spend_nullifier_0.entry((batch_id, i)).write(*spe.nullifier_0);
                self.batch_spend_nullifier_1.entry((batch_id, i)).write(*spe.nullifier_1);
                self.batch_spend_output_0.entry((batch_id, i)).write(*spe.output_commitment_0);
                self.batch_spend_output_1.entry((batch_id, i)).write(*spe.output_commitment_1);
                i += 1;
            };

            // Initialize pending tree state from current
            self.batch_pending_root.entry(batch_id).write(self.merkle_root.read());
            self.batch_pending_tree_size.entry(batch_id).write(self.tree_size.read());

            self.emit(BatchSubmitted {
                batch_id,
                submitter: caller,
                n_deposits: n_dep,
                n_withdrawals: n_wit,
                n_spends: n_spe,
                block_number: block,
            });

            batch_id
        }

        // Step 2: Apply a chunk of the batch
        fn apply_batch_chunk(
            ref self: ContractState,
            batch_id: felt252,
            start: u32,
            count: u32,
        ) {
            assert!(!self.paused.read(), "VM31: contract is paused");
            assert!(!self.reentrancy_locked.read(), "VM31: reentrant call");
            let caller = get_caller_address();
            let now_block = get_block_number();
            assert!(self.current_batch_id.read() == batch_id, "VM31: batch is not active");
            let status = self.batch_status.entry(batch_id).read();
            assert!(status == BATCH_STATUS_SUBMITTED, "VM31: batch not in submitted state");
            let relayer = self.relayer.read();
            if caller != relayer {
                let submit_block = self.batch_block.entry(batch_id).read();
                let timeout_blocks = self.batch_timeout_blocks.read();
                let unlock_block = submit_block + timeout_blocks;
                assert!(unlock_block >= submit_block, "VM31: timeout overflow");
                assert!(now_block >= unlock_block, "VM31: relayer only before timeout");
            }
            assert!(count > 0 && count <= MAX_CHUNK_SIZE, "VM31: invalid chunk size");

            let total = self.batch_total_txs.entry(batch_id).read();
            let processed = self.batch_processed_count.entry(batch_id).read();
            assert!(start == processed, "VM31: chunk must start at processed count");
            assert!(start + count <= total, "VM31: chunk exceeds batch size");
            let submit_root_seq = self.batch_submit_root_seq.entry(batch_id).read();

            let n_dep = self.batch_n_deposits.entry(batch_id).read();
            let n_wit = self.batch_n_withdrawals.entry(batch_id).read();

            // Transaction ordering: deposits first, then withdrawals, then spends
            // Global index: [0..n_dep) = deposits, [n_dep..n_dep+n_wit) = withdrawals,
            //               [n_dep+n_wit..total) = spends
            let mut pending_tree_size = self.batch_pending_tree_size.entry(batch_id).read();
            let mut idx = start;
            let end = start + count;

            loop {
                if idx >= end {
                    break;
                }

                if idx < n_dep {
                    // Process deposit: insert commitment into tree
                    let commitment = self.batch_deposit_commitment.entry((batch_id, idx)).read();
                    let amount_lo = self.batch_deposit_amount_lo.entry((batch_id, idx)).read();
                    let amount_hi = self.batch_deposit_amount_hi.entry((batch_id, idx)).read();
                    let asset = self.batch_deposit_asset.entry((batch_id, idx)).read();

                    // C1: Tree overflow guard
                    assert!(pending_tree_size < MAX_LEAVES, "VM31: tree is full");

                    // Insert leaf into tree
                    let leaf_idx = pending_tree_size;
                    self.merkle_leaves.entry(leaf_idx).write(commitment);
                    InternalImpl::update_merkle_path(ref self, leaf_idx, commitment);
                    pending_tree_size += 1;

                    // Credit asset vault + ERC20 custody (pull from batch submitter)
                    let amount = reconstruct_amount(amount_lo, amount_hi);
                    assert!(amount > 0, "VM31: amount must be positive");
                    let asset_felt: felt252 = asset.into();
                    let token_addr = self.asset_token.entry(asset_felt).read();
                    let token_felt: felt252 = token_addr.into();
                    assert!(token_felt != 0, "VM31: asset not registered");
                    let submitter = self.batch_submitter.entry(batch_id).read();
                    let amount_u256: u256 = amount.into();
                    let token = IERC20Dispatcher { contract_address: token_addr };
                    self.reentrancy_locked.write(true);
                    let ok = token.transfer_from(submitter, get_contract_address(), amount_u256);
                    self.reentrancy_locked.write(false);
                    assert!(ok, "VM31: ERC20 transferFrom failed on deposit");
                    let cur_bal = self.asset_balances.entry(asset_felt).read();
                    self.asset_balances.entry(asset_felt).write(cur_bal + amount);

                    self.emit(NoteInserted { leaf_index: leaf_idx, commitment });
                    self.emit(DepositProcessed { commitment, amount, asset_id: asset_felt });

                } else if idx < n_dep + n_wit {
                    // Process withdrawal: check nullifier, debit vault
                    let wit_idx = idx - n_dep;
                    let merkle_root = self.batch_withdraw_merkle_root.entry((batch_id, wit_idx)).read();
                    assert!(
                        InternalImpl::is_root_known_at_or_before(ref self, merkle_root, submit_root_seq),
                        "VM31: unknown withdraw root"
                    );
                    let nullifier = self.batch_withdraw_nullifier.entry((batch_id, wit_idx)).read();
                    let amount_lo = self.batch_withdraw_amount_lo.entry((batch_id, wit_idx)).read();
                    let amount_hi = self.batch_withdraw_amount_hi.entry((batch_id, wit_idx)).read();
                    let asset = self.batch_withdraw_asset.entry((batch_id, wit_idx)).read();

                    // Nullifier must not be spent
                    let nul_key = pack_digest_to_felt(nullifier);
                    assert!(!self.nullifiers.entry(nul_key).read(), "VM31: nullifier already spent");
                    self.nullifiers.entry(nul_key).write(true);

                    // Debit asset vault + ERC20 transfer to stored recipient
                    let amount = reconstruct_amount(amount_lo, amount_hi);
                    assert!(amount > 0, "VM31: amount must be positive");
                    let asset_felt: felt252 = asset.into();
                    let cur_bal = self.asset_balances.entry(asset_felt).read();
                    assert!(cur_bal >= amount, "VM31: insufficient pool balance");
                    self.asset_balances.entry(asset_felt).write(cur_bal - amount);

                    let recipient = self.batch_withdraw_recipient.entry((batch_id, wit_idx)).read();
                    let token_addr = self.asset_token.entry(asset_felt).read();
                    let token_felt: felt252 = token_addr.into();
                    assert!(token_felt != 0, "VM31: asset not registered");
                    let amount_u256: u256 = amount.into();
                    let token = IERC20Dispatcher { contract_address: token_addr };
                    self.reentrancy_locked.write(true);
                    let ok = token.transfer(recipient, amount_u256);
                    self.reentrancy_locked.write(false);
                    assert!(ok, "VM31: ERC20 transfer failed on withdrawal");

                    self.emit(NullifierSpent { nullifier_hash: nul_key });
                    self.emit(WithdrawProcessed { nullifier, amount, asset_id: asset_felt });

                } else {
                    // Process spend: check nullifiers + insert output commitments
                    let spe_idx = idx - n_dep - n_wit;
                    let merkle_root = self.batch_spend_merkle_root.entry((batch_id, spe_idx)).read();
                    assert!(
                        InternalImpl::is_root_known_at_or_before(ref self, merkle_root, submit_root_seq),
                        "VM31: unknown spend root"
                    );
                    let nullifier_0 = self.batch_spend_nullifier_0.entry((batch_id, spe_idx)).read();
                    let nullifier_1 = self.batch_spend_nullifier_1.entry((batch_id, spe_idx)).read();
                    let output_0 = self.batch_spend_output_0.entry((batch_id, spe_idx)).read();
                    let output_1 = self.batch_spend_output_1.entry((batch_id, spe_idx)).read();

                    // Both nullifiers must not be spent
                    let nul_key_0 = pack_digest_to_felt(nullifier_0);
                    let nul_key_1 = pack_digest_to_felt(nullifier_1);
                    assert!(!self.nullifiers.entry(nul_key_0).read(), "VM31: nullifier 0 already spent");
                    assert!(!self.nullifiers.entry(nul_key_1).read(), "VM31: nullifier 1 already spent");
                    self.nullifiers.entry(nul_key_0).write(true);
                    self.nullifiers.entry(nul_key_1).write(true);

                    // C1: Tree overflow guard (need room for 2 outputs)
                    assert!(pending_tree_size + 1 < MAX_LEAVES, "VM31: tree is full");

                    // Insert both output commitments
                    let leaf_0 = pending_tree_size;
                    self.merkle_leaves.entry(leaf_0).write(output_0);
                    InternalImpl::update_merkle_path(ref self, leaf_0, output_0);
                    pending_tree_size += 1;

                    let leaf_1 = pending_tree_size;
                    self.merkle_leaves.entry(leaf_1).write(output_1);
                    InternalImpl::update_merkle_path(ref self, leaf_1, output_1);
                    pending_tree_size += 1;

                    self.emit(NoteInserted { leaf_index: leaf_0, commitment: output_0 });
                    self.emit(NoteInserted { leaf_index: leaf_1, commitment: output_1 });
                    self.emit(NullifierSpent { nullifier_hash: nul_key_0 });
                    self.emit(NullifierSpent { nullifier_hash: nul_key_1 });
                }

                idx += 1;
            };

            // Update pending state
            self.batch_pending_tree_size.entry(batch_id).write(pending_tree_size);
            let new_processed = processed + count;
            self.batch_processed_count.entry(batch_id).write(new_processed);

            // Recompute pending root from updated nodes
            let pending_root = InternalImpl::compute_root(ref self, pending_tree_size);
            self.batch_pending_root.entry(batch_id).write(pending_root);

            self.emit(BatchChunkApplied {
                batch_id,
                start,
                count,
                processed_total: new_processed,
            });
        }

        // Step 3: Finalize the batch
        fn finalize_batch(
            ref self: ContractState,
            batch_id: felt252,
        ) {
            assert!(!self.paused.read(), "VM31: contract is paused");
            let caller = get_caller_address();
            let now_block = get_block_number();
            assert!(self.current_batch_id.read() == batch_id, "VM31: batch is not active");
            let status = self.batch_status.entry(batch_id).read();
            assert!(status == BATCH_STATUS_SUBMITTED, "VM31: batch not in submitted state");
            let relayer = self.relayer.read();
            if caller != relayer {
                let submit_block = self.batch_block.entry(batch_id).read();
                let timeout_blocks = self.batch_timeout_blocks.read();
                let unlock_block = submit_block + timeout_blocks;
                assert!(unlock_block >= submit_block, "VM31: timeout overflow");
                assert!(now_block >= unlock_block, "VM31: relayer only before timeout");
            }

            let total = self.batch_total_txs.entry(batch_id).read();
            let processed = self.batch_processed_count.entry(batch_id).read();
            assert!(processed == total, "VM31: not all transactions processed");

            let new_root = self.batch_pending_root.entry(batch_id).read();
            let new_size = self.batch_pending_tree_size.entry(batch_id).read();

            // Update canonical state
            self.merkle_root.write(new_root);
            self.tree_size.write(new_size);
            InternalImpl::record_new_root(ref self, new_root);

            // Mark batch as finalized
            self.batch_status.entry(batch_id).write(BATCH_STATUS_FINALIZED);
            self.current_batch_id.write(0);

            self.emit(BatchFinalized {
                batch_id,
                new_root,
                new_tree_size: new_size,
                total_txs: total,
            });
        }

        // Direct deposit (non-batched)
        fn deposit(
            ref self: ContractState,
            commitment: PackedDigest,
            amount: u64,
            asset_id: felt252,
            proof_hash: felt252,
        ) {
            assert!(!self.paused.read(), "VM31: contract is paused");
            assert!(!self.reentrancy_locked.read(), "VM31: reentrant call");
            assert!(amount > 0, "VM31: amount must be positive");
            assert!(self.current_batch_id.read() == 0, "VM31: active batch in progress");
            let asset_id_u64: u64 = asset_id.try_into().unwrap();
            assert!(asset_id_u64 < M31_P, "VM31: asset_id out of M31 range");
            let verifier_addr = self.verifier_contract.read();
            let verifier_addr_felt: felt252 = verifier_addr.into();
            assert!(verifier_addr_felt != 0, "VM31: verifier contract not set");
            let verifier = ISumcheckVerifierDispatcher { contract_address: verifier_addr };
            assert!(
                verifier.is_proof_verified(proof_hash),
                "VM31: proof hash not verified"
            );

            // ERC20 custody: pull tokens from caller
            let token_addr = self.asset_token.entry(asset_id).read();
            let token_felt: felt252 = token_addr.into();
            assert!(token_felt != 0, "VM31: asset not registered");
            let caller = get_caller_address();
            let amount_u256: u256 = amount.into();
            let token = IERC20Dispatcher { contract_address: token_addr };
            self.reentrancy_locked.write(true);
            let ok = token.transfer_from(caller, get_contract_address(), amount_u256);
            self.reentrancy_locked.write(false);
            assert!(ok, "VM31: ERC20 transferFrom failed");

            // C1: Tree overflow guard
            let leaf_idx = self.tree_size.read();
            assert!(leaf_idx < MAX_LEAVES, "VM31: tree is full");
            self.merkle_leaves.entry(leaf_idx).write(commitment);
            InternalImpl::update_merkle_path(ref self, leaf_idx, commitment);
            let new_size = leaf_idx + 1;
            self.tree_size.write(new_size);

            // Recompute root
            let new_root = InternalImpl::compute_root(ref self, new_size);
            self.merkle_root.write(new_root);
            InternalImpl::record_new_root(ref self, new_root);

            // Credit vault
            let cur_bal = self.asset_balances.entry(asset_id).read();
            self.asset_balances.entry(asset_id).write(cur_bal + amount);

            self.emit(NoteInserted { leaf_index: leaf_idx, commitment });
            self.emit(DepositProcessed { commitment, amount, asset_id });
        }

        // ── View Functions ──

        fn get_merkle_root(self: @ContractState) -> PackedDigest {
            self.merkle_root.read()
        }

        fn get_tree_size(self: @ContractState) -> u64 {
            self.tree_size.read()
        }

        fn is_nullifier_spent(self: @ContractState, nullifier: PackedDigest) -> bool {
            let key = pack_digest_to_felt(nullifier);
            self.nullifiers.entry(key).read()
        }

        fn get_asset_balance(self: @ContractState, asset_id: felt252) -> u64 {
            self.asset_balances.entry(asset_id).read()
        }

        fn is_known_root(self: @ContractState, root: PackedDigest) -> bool {
            let key = pack_digest_to_felt(root);
            self.root_first_seen_seq.entry(key).read() > 0
        }

        fn get_batch_status(self: @ContractState, batch_id: felt252) -> u8 {
            self.batch_status.entry(batch_id).read()
        }

        fn get_batch_for_proof_hash(self: @ContractState, proof_stark_hash: felt252) -> felt252 {
            self.batch_for_proof_hash.entry(proof_stark_hash).read()
        }

        fn get_batch_total_txs(self: @ContractState, batch_id: felt252) -> u32 {
            self.batch_total_txs.entry(batch_id).read()
        }

        fn get_batch_processed_count(self: @ContractState, batch_id: felt252) -> u32 {
            self.batch_processed_count.entry(batch_id).read()
        }

        fn get_batch_withdrawal_binding(
            self: @ContractState,
            batch_id: felt252,
            withdraw_idx: u32,
        ) -> PackedDigest {
            self.batch_withdraw_binding.entry((batch_id, withdraw_idx)).read()
        }

        fn get_batch_withdrawal_binding_felt(
            self: @ContractState,
            batch_id: felt252,
            withdraw_idx: u32,
        ) -> felt252 {
            let binding = self.batch_withdraw_binding.entry((batch_id, withdraw_idx)).read();
            binding_digest_to_felt(binding)
        }

        fn compute_withdrawal_binding_felt(
            self: @ContractState,
            payout_recipient: ContractAddress,
            credit_recipient: ContractAddress,
            asset_id: felt252,
            amount_lo: u64,
            amount_hi: u64,
            withdraw_idx: u32,
        ) -> felt252 {
            let payout_recipient_felt: felt252 = payout_recipient.into();
            let credit_recipient_felt: felt252 = credit_recipient.into();
            assert!(payout_recipient_felt != 0, "VM31: payout recipient cannot be zero");
            assert!(credit_recipient_felt != 0, "VM31: credit recipient cannot be zero");
            let asset_u256: u256 = asset_id.into();
            let m31_p_u256: u256 = M31_P.into();
            assert!(asset_u256 < m31_p_u256, "VM31: asset_id out of M31 range");
            assert!(amount_lo < M31_P, "VM31: amount_lo out of M31 range");
            assert!(amount_hi < M31_P, "VM31: amount_hi out of M31 range");
            let binding = compute_withdrawal_binding_digest_v2(
                payout_recipient, credit_recipient, asset_id, amount_lo, amount_hi, withdraw_idx,
            );
            binding_digest_to_felt(binding)
        }

        fn compute_withdrawal_binding_v1_felt(
            self: @ContractState,
            payout_recipient: ContractAddress,
            asset_id: felt252,
            amount_lo: u64,
            amount_hi: u64,
            withdraw_idx: u32,
        ) -> felt252 {
            let payout_recipient_felt: felt252 = payout_recipient.into();
            assert!(payout_recipient_felt != 0, "VM31: payout recipient cannot be zero");
            let asset_u256: u256 = asset_id.into();
            let m31_p_u256: u256 = M31_P.into();
            assert!(asset_u256 < m31_p_u256, "VM31: asset_id out of M31 range");
            assert!(amount_lo < M31_P, "VM31: amount_lo out of M31 range");
            assert!(amount_hi < M31_P, "VM31: amount_hi out of M31 range");
            let binding = compute_withdrawal_binding_digest_v1(
                payout_recipient, asset_id, amount_lo, amount_hi, withdraw_idx,
            );
            binding_digest_to_felt(binding)
        }

        fn get_batch_timeout_blocks(self: @ContractState) -> u64 {
            self.batch_timeout_blocks.read()
        }

        fn get_active_batch_id(self: @ContractState) -> felt252 {
            self.current_batch_id.read()
        }

        fn get_owner(self: @ContractState) -> ContractAddress {
            self.owner.read()
        }

        fn get_relayer(self: @ContractState) -> ContractAddress {
            self.relayer.read()
        }

        fn get_verifier_contract(self: @ContractState) -> ContractAddress {
            self.verifier_contract.read()
        }

        fn get_asset_token(self: @ContractState, asset_id: felt252) -> ContractAddress {
            self.asset_token.entry(asset_id).read()
        }

        fn get_token_asset(self: @ContractState, token_address: ContractAddress) -> felt252 {
            self.token_asset.entry(token_address).read()
        }

        fn get_next_asset_id(self: @ContractState) -> felt252 {
            self.next_asset_id.read().into()
        }

        fn get_pending_upgrade(self: @ContractState) -> (ClassHash, u64) {
            (self.pending_upgrade.read(), self.upgrade_proposed_at.read())
        }

        fn is_paused(self: @ContractState) -> bool {
            self.paused.read()
        }

        fn get_pending_verifier(self: @ContractState) -> (ContractAddress, u64) {
            (self.pending_verifier.read(), self.verifier_change_proposed_at.read())
        }

        fn is_v1_bindings_enabled(self: @ContractState) -> bool {
            self.v1_bindings_enabled.read()
        }

        fn get_v1_disable_proposed_at(self: @ContractState) -> u64 {
            self.v1_disable_proposed_at.read()
        }

        // ── Admin ──

        fn cancel_batch(ref self: ContractState, batch_id: felt252) {
            assert!(get_caller_address() == self.owner.read(), "VM31: owner only");
            assert!(self.current_batch_id.read() == batch_id, "VM31: batch is not active");
            let status = self.batch_status.entry(batch_id).read();
            assert!(status == BATCH_STATUS_SUBMITTED, "VM31: batch not in submitted state");
            let processed = self.batch_processed_count.entry(batch_id).read();
            assert!(processed == 0, "VM31: cannot cancel after processing started");

            self.batch_status.entry(batch_id).write(BATCH_STATUS_CANCELLED);
            self.current_batch_id.write(0);

            self.emit(BatchCancelled { batch_id, cancelled_by: get_caller_address() });
        }

        fn set_relayer(ref self: ContractState, relayer: ContractAddress) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");
            let old_relayer = self.relayer.read();
            self.relayer.write(relayer);
            self.emit(RelayerChanged { old_relayer, new_relayer: relayer, changed_by: caller });
        }

        fn set_batch_timeout_blocks(ref self: ContractState, timeout_blocks: u64) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");
            assert!(timeout_blocks >= 10, "VM31: timeout must be >= 10");
            let old_timeout = self.batch_timeout_blocks.read();
            self.batch_timeout_blocks.write(timeout_blocks);
            self.emit(BatchTimeoutChanged {
                old_timeout, new_timeout: timeout_blocks, changed_by: caller,
            });
        }

        fn register_asset(ref self: ContractState, token_address: ContractAddress) -> felt252 {
            assert!(get_caller_address() == self.owner.read(), "VM31: owner only");
            let token_felt: felt252 = token_address.into();
            assert!(token_felt != 0, "VM31: token address cannot be zero");

            // Prevent duplicate token registration
            let existing_asset = self.token_asset.entry(token_address).read();
            assert!(existing_asset == 0, "VM31: token already registered");

            let asset_id: u64 = self.next_asset_id.read();
            assert!(asset_id < M31_P, "VM31: asset id space exhausted");
            let asset_id_felt: felt252 = asset_id.into();
            self.asset_token.entry(asset_id_felt).write(token_address);
            self.token_asset.entry(token_address).write(asset_id_felt);
            self.next_asset_id.write(asset_id + 1);

            self.emit(AssetRegistered { asset_id: asset_id_felt, token_address });
            asset_id_felt
        }

        fn pause(ref self: ContractState) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");
            assert!(!self.paused.read(), "VM31: already paused");
            self.paused.write(true);
            self.emit(Paused { paused_by: caller });
        }

        fn unpause(ref self: ContractState) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");
            assert!(self.paused.read(), "VM31: not paused");
            self.paused.write(false);
            self.emit(Unpaused { unpaused_by: caller });
        }

        // ── Timelocked Verifier Change ──

        fn propose_verifier_change(ref self: ContractState, new_verifier: ContractAddress) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");
            let new_verifier_felt: felt252 = new_verifier.into();
            assert!(new_verifier_felt != 0, "VM31: verifier address cannot be zero");

            let existing: felt252 = self.pending_verifier.read().into();
            assert!(existing == 0, "VM31: verifier change already pending, cancel first");

            let now = get_block_timestamp();
            self.pending_verifier.write(new_verifier);
            self.verifier_change_proposed_at.write(now);

            self.emit(VerifierChangeProposed {
                new_verifier, proposed_at: now, proposer: caller,
            });
        }

        fn execute_verifier_change(ref self: ContractState) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");

            let new_verifier = self.pending_verifier.read();
            let new_verifier_felt: felt252 = new_verifier.into();
            assert!(new_verifier_felt != 0, "VM31: no verifier change pending");

            let proposed_at = self.verifier_change_proposed_at.read();
            let now = get_block_timestamp();
            assert!(now >= proposed_at + UPGRADE_DELAY, "VM31: verifier change delay not elapsed");

            // Clear pending state and apply
            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            self.pending_verifier.write(zero_addr);
            self.verifier_change_proposed_at.write(0);
            self.verifier_contract.write(new_verifier);

            self.emit(VerifierChangeExecuted {
                new_verifier, executed_at: now, executor: caller,
            });
        }

        fn cancel_verifier_change(ref self: ContractState) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");

            let pending = self.pending_verifier.read();
            let pending_felt: felt252 = pending.into();
            assert!(pending_felt != 0, "VM31: no verifier change pending");

            let zero_addr: ContractAddress = 0_felt252.try_into().unwrap();
            self.pending_verifier.write(zero_addr);
            self.verifier_change_proposed_at.write(0);

            self.emit(VerifierChangeCancelled {
                cancelled_verifier: pending, cancelled_by: caller,
            });
        }

        // ── Timelocked V1 Binding Disable ──

        fn propose_disable_v1_bindings(ref self: ContractState) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");
            assert!(self.v1_bindings_enabled.read(), "VM31: v1 bindings already disabled");
            assert!(
                self.v1_disable_proposed_at.read() == 0,
                "VM31: v1 disable already pending"
            );

            let now = get_block_timestamp();
            self.v1_disable_proposed_at.write(now);
            self.emit(V1BindingsDisableProposed { proposed_at: now, proposer: caller });
        }

        fn execute_disable_v1_bindings(ref self: ContractState) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");
            assert!(self.v1_bindings_enabled.read(), "VM31: v1 bindings already disabled");
            let proposed_at = self.v1_disable_proposed_at.read();
            assert!(proposed_at != 0, "VM31: no v1 disable pending");

            let now = get_block_timestamp();
            assert!(now >= proposed_at + UPGRADE_DELAY, "VM31: v1 disable delay not elapsed");

            self.v1_bindings_enabled.write(false);
            self.v1_disable_proposed_at.write(0);
            self.emit(V1BindingsDisableExecuted { executed_at: now, executor: caller });
        }

        fn cancel_disable_v1_bindings(ref self: ContractState) {
            let caller = get_caller_address();
            assert!(caller == self.owner.read(), "VM31: owner only");
            let proposed_at = self.v1_disable_proposed_at.read();
            assert!(proposed_at != 0, "VM31: no v1 disable pending");

            self.v1_disable_proposed_at.write(0);
            self.emit(V1BindingsDisableCancelled { cancelled_by: caller });
        }

        // ── Upgradability ──

        fn propose_upgrade(ref self: ContractState, new_class_hash: ClassHash) {
            assert!(get_caller_address() == self.owner.read(), "VM31: owner only");
            assert!(new_class_hash.into() != 0_felt252, "VM31: class hash cannot be zero");

            let existing: felt252 = self.pending_upgrade.read().into();
            assert!(existing == 0, "VM31: upgrade already pending, cancel first");

            let now = get_block_timestamp();
            self.pending_upgrade.write(new_class_hash);
            self.upgrade_proposed_at.write(now);

            self.emit(UpgradeProposed {
                new_class_hash, proposed_at: now, proposer: get_caller_address(),
            });
        }

        fn execute_upgrade(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "VM31: owner only");

            let new_class_hash = self.pending_upgrade.read();
            assert!(new_class_hash.into() != 0_felt252, "VM31: no upgrade pending");

            let proposed_at = self.upgrade_proposed_at.read();
            let now = get_block_timestamp();
            assert!(now >= proposed_at + UPGRADE_DELAY, "VM31: upgrade delay not elapsed");

            // Clear pending state before syscall
            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);

            self.emit(UpgradeExecuted {
                new_class_hash, executed_at: now, executor: get_caller_address(),
            });

            starknet::syscalls::replace_class_syscall(new_class_hash).unwrap();
        }

        fn cancel_upgrade(ref self: ContractState) {
            assert!(get_caller_address() == self.owner.read(), "VM31: owner only");

            let pending: ClassHash = self.pending_upgrade.read();
            assert!(pending.into() != 0_felt252, "VM31: no upgrade pending");

            self.pending_upgrade.write(0.try_into().unwrap());
            self.upgrade_proposed_at.write(0);

            self.emit(UpgradeCancelled {
                cancelled_class_hash: pending, cancelled_by: get_caller_address(),
            });
        }
    }

    // ── Internal Helpers ──

    #[generate_trait]
    impl InternalImpl of InternalTrait {
        // Update the Merkle tree nodes along the path from a leaf to the root.
        // Uses a sparse representation: only non-empty nodes are stored.
        fn update_merkle_path(
            ref self: ContractState,
            leaf_index: u64,
            leaf_value: PackedDigest,
        ) {
            // Store at level 0
            self.merkle_nodes.entry((0, leaf_index)).write(leaf_value);

            // Walk up the tree
            let mut level: u32 = 0;
            let mut idx = leaf_index;
            loop {
                if level >= MERKLE_DEPTH {
                    break;
                }
                let parent_idx = idx / 2;
                let left_idx = parent_idx * 2;
                let right_idx = left_idx + 1;

                let left = self.merkle_nodes.entry((level, left_idx)).read();
                let right = self.merkle_nodes.entry((level, right_idx)).read();

                let parent_hash = poseidon2_m31_compress_packed(left, right);
                self.merkle_nodes.entry((level + 1, parent_idx)).write(parent_hash);

                idx = parent_idx;
                level += 1;
            };
        }

        // Compute the root from the top-level node
        fn compute_root(ref self: ContractState, _tree_size: u64) -> PackedDigest {
            // The root is at level MERKLE_DEPTH, index 0
            self.merkle_nodes.entry((MERKLE_DEPTH, 0)).read()
        }

        // Record a canonical root into both the ring buffer and monotonic root-sequence index.
        fn record_new_root(ref self: ContractState, root: PackedDigest) {
            let hist_idx = self.root_history_index.read();
            let next_idx = (hist_idx + 1) % ROOT_HISTORY_SIZE;
            self.root_history.entry(next_idx).write(root);
            self.root_history_index.write(next_idx);

            let next_seq = self.current_root_seq.read() + 1;
            self.current_root_seq.write(next_seq);

            let key = pack_digest_to_felt(root);
            let first_seen = self.root_first_seen_seq.entry(key).read();
            if first_seen == 0 {
                self.root_first_seen_seq.entry(key).write(next_seq);
            }
        }

        // Root validity check bound by the root sequence at batch submission.
        fn is_root_known_at_or_before(
            ref self: ContractState,
            root: PackedDigest,
            max_seq: u64,
        ) -> bool {
            let key = pack_digest_to_felt(root);
            let seen = self.root_first_seen_seq.entry(key).read();
            seen > 0 && seen <= max_seq
        }
    }

    // Pack a PackedDigest into a single felt252 for use as a storage key.
    // Uses Poseidon252 hash of (lo, hi) for collision resistance.
    fn pack_digest_to_felt(d: PackedDigest) -> felt252 {
        poseidon_hash_span(array![d.lo, d.hi].span())
    }

    // Compute the canonical V2 withdrawal binding digest from
    // (payout_recipient, credit_recipient, asset, amount, index).
    // This is proof-bound by inclusion in WithdrawPublicInput.withdrawal_binding.
    fn compute_withdrawal_binding_digest_v2(
        payout_recipient: ContractAddress,
        credit_recipient: ContractAddress,
        asset_id: felt252,
        amount_lo: u64,
        amount_hi: u64,
        withdraw_idx: u32,
    ) -> PackedDigest {
        let payout_recipient_felt: felt252 = payout_recipient.into();
        let credit_recipient_felt: felt252 = credit_recipient.into();
        let binding_hash_felt = poseidon_hash_span(
            array![
                WITHDRAW_BINDING_DOMAIN_V2,
                payout_recipient_felt,
                credit_recipient_felt,
                asset_id,
                amount_lo.into(),
                amount_hi.into(),
                withdraw_idx.into(),
            ].span(),
        );
        felt_to_binding_digest(binding_hash_felt)
    }

    // Compute the legacy V1 withdrawal binding digest from
    // (payout_recipient, asset, amount, index).
    // Accepted only for migration and only when payout_recipient == credit_recipient.
    fn compute_withdrawal_binding_digest_v1(
        payout_recipient: ContractAddress,
        asset_id: felt252,
        amount_lo: u64,
        amount_hi: u64,
        withdraw_idx: u32,
    ) -> PackedDigest {
        let payout_recipient_felt: felt252 = payout_recipient.into();
        let binding_hash_felt = poseidon_hash_span(
            array![
                WITHDRAW_BINDING_DOMAIN_V1,
                payout_recipient_felt,
                asset_id,
                amount_lo.into(),
                amount_hi.into(),
                withdraw_idx.into(),
            ].span(),
        );
        felt_to_binding_digest(binding_hash_felt)
    }

    // Truncate a felt252 hash to 8 canonical M31 limbs (low 248 bits).
    // Rare limbs equal to p are mapped to 0 so they remain valid M31 elements.
    fn felt_to_binding_digest(v: felt252) -> PackedDigest {
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

    // Convert an 8-limb M31 digest into a canonical felt252 value:
    //   sum_{i=0..7} limb_i * (2^31)^i
    fn binding_digest_to_felt(d: PackedDigest) -> felt252 {
        let limbs = unpack_m31x8(d);
        let radix: felt252 = 0x80000000; // 2^31
        let mut acc: felt252 = 0;
        let mut factor: felt252 = 1;
        let mut i: u32 = 0;
        loop {
            if i >= 8 {
                break;
            }
            let limb_felt: felt252 = (*limbs.at(i)).into();
            acc += limb_felt * factor;
            factor *= radix;
            i += 1;
        };
        acc
    }
}
