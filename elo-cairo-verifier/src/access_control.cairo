// Access Control for encrypted audit reports.
//
// On-chain ACL: the audit owner can grant/revoke read access to
// specific Starknet addresses. Each grant stores a wrapped key
// (the AES data-key encrypted with the grantee's public key).
//
// Storage model:
//   audit_access:       (audit_id, address) → AuditAccess
//   audit_wrapped_keys: (audit_id, address) → wrapped_key
//   audit_access_list:  (audit_id, index)   → address  (for enumeration)
//   audit_access_count: audit_id            → u32
//   audit_owner:        audit_id            → owner address

use starknet::ContractAddress;

// ─── Types ──────────────────────────────────────────────────────────────────

#[derive(Drop, Copy, Serde, starknet::Store)]
pub struct AuditAccess {
    /// Starknet address of the authorized party.
    pub address: ContractAddress,
    /// Role: 0 = owner, 1 = auditor, 2 = regulator, 3 = partner.
    pub role: u8,
    /// Block number when access was granted.
    pub granted_at_block: u64,
    /// Whether access is currently active.
    pub is_active: bool,
}

// ─── Events ─────────────────────────────────────────────────────────────────

#[derive(Drop, starknet::Event)]
pub struct AccessGranted {
    #[key]
    pub audit_id: felt252,
    #[key]
    pub grantee: ContractAddress,
    pub role: u8,
    pub granted_by: ContractAddress,
}

#[derive(Drop, starknet::Event)]
pub struct AccessRevoked {
    #[key]
    pub audit_id: felt252,
    #[key]
    pub revokee: ContractAddress,
    pub revoked_by: ContractAddress,
}

// ─── Interface ──────────────────────────────────────────────────────────────

#[starknet::interface]
pub trait IAuditAccessControl<TContractState> {
    /// Grant read access to an audit report.
    ///
    /// Only the audit owner can grant access. The wrapped_key is the
    /// AES data key encrypted with the grantee's public key.
    fn grant_audit_access(
        ref self: TContractState,
        audit_id: felt252,
        grantee: ContractAddress,
        wrapped_key: felt252,
        role: u8,
    );

    /// Revoke read access from a party.
    ///
    /// Only the owner can revoke. Zeroes the wrapped key on-chain.
    /// Off-chain: the owner should re-encrypt the report with a new
    /// data key and re-wrap for remaining parties.
    fn revoke_audit_access(
        ref self: TContractState,
        audit_id: felt252,
        revokee: ContractAddress,
    );

    /// Batch grant access to multiple parties.
    fn grant_audit_access_batch(
        ref self: TContractState,
        audit_id: felt252,
        grantees: Span<ContractAddress>,
        wrapped_keys: Span<felt252>,
        roles: Span<u8>,
    );

    /// Check if an address has active access to an audit.
    fn has_audit_access(
        self: @TContractState,
        audit_id: felt252,
        address: ContractAddress,
    ) -> bool;

    /// Get the wrapped key for a specific grantee.
    ///
    /// The grantee calls this to retrieve their encrypted data key,
    /// then decrypts it with their Starknet private key.
    fn get_wrapped_key(
        self: @TContractState,
        audit_id: felt252,
        grantee: ContractAddress,
    ) -> felt252;

    /// Get the owner of an audit (whoever called submit_audit).
    fn get_audit_owner(
        self: @TContractState,
        audit_id: felt252,
    ) -> ContractAddress;

    /// Get the number of active access grants for an audit.
    fn get_access_count(
        self: @TContractState,
        audit_id: felt252,
    ) -> u32;
}
