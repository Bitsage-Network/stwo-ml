// View Key Delegation for long-term audit access.
//
// Instead of per-audit grants, an audit owner can delegate a view key
// that gives read access to ALL their future audits. The delegate
// receives an encrypted view key via ElGamal and can scan for new
// audits without per-audit grants.
//
// Storage model:
//   view_delegations:       (owner, delegate)       → ViewKeyDelegation
//   view_delegation_list:   (owner, index)          → delegate address
//   view_delegation_count:  owner                   → u32

use starknet::ContractAddress;

// ─── Types ──────────────────────────────────────────────────────────────────

#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct ViewKeyDelegation {
    /// Who delegated the view key.
    pub owner: ContractAddress,
    /// Who received the view key.
    pub delegate: ContractAddress,
    /// Encrypted view key (ElGamal with delegate's public key).
    pub encrypted_view_key: felt252,
    /// Valid from (block number).
    pub valid_from: u64,
    /// Valid until (block number, 0 = forever).
    pub valid_until: u64,
    /// Whether currently active.
    pub is_active: bool,
}

// ─── Events ─────────────────────────────────────────────────────────────────

#[derive(Drop, starknet::Event)]
pub struct ViewKeyDelegated {
    #[key]
    pub owner: ContractAddress,
    #[key]
    pub delegate: ContractAddress,
    pub valid_until: u64,
}

#[derive(Drop, starknet::Event)]
pub struct ViewKeyRevoked {
    #[key]
    pub owner: ContractAddress,
    #[key]
    pub delegate: ContractAddress,
}

// ─── Interface ──────────────────────────────────────────────────────────────

#[starknet::interface]
pub trait IViewKeyDelegation<TContractState> {
    /// Delegate a view key to another address.
    ///
    /// The encrypted_view_key is the owner's view key encrypted with
    /// the delegate's public key. valid_until = 0 means forever.
    fn delegate_view_key(
        ref self: TContractState,
        delegate: ContractAddress,
        encrypted_view_key: felt252,
        valid_until: u64,
    );

    /// Revoke a previously delegated view key.
    fn revoke_view_key(
        ref self: TContractState,
        delegate: ContractAddress,
    );

    /// Check if a delegate has an active view key from an owner.
    fn has_view_key(
        self: @TContractState,
        owner: ContractAddress,
        delegate: ContractAddress,
    ) -> bool;

    /// Get the encrypted view key for a delegate.
    fn get_view_key(
        self: @TContractState,
        owner: ContractAddress,
        delegate: ContractAddress,
    ) -> felt252;

    /// Get the number of active view key delegations from an owner.
    fn get_delegation_count(
        self: @TContractState,
        owner: ContractAddress,
    ) -> u32;
}
