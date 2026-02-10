use starknet::ContractAddress;

/// ERC-20 dispatcher for SAGE token transfers.
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

/// ObelyskVerifier interface.
#[starknet::interface]
pub trait IObelyskVerifier<TContractState> {
    /// Verify a recursive ML proof and process payment in a single transaction.
    ///
    /// Flow:
    /// 1. Verify the recursive proof fact via trusted submitter (owner)
    /// 2. Record the verification on-chain
    /// 3. Transfer SAGE payment from caller to worker
    /// 4. Emit rich events for indexing
    fn verify_and_pay(
        ref self: TContractState,
        model_id: felt252,
        proof_hash: felt252,
        io_commitment: felt252,
        weight_commitment: felt252,
        num_layers: u32,
        job_id: felt252,
        worker: ContractAddress,
        sage_amount: u256,
    ) -> bool;

    /// Register a model for verification.
    fn register_model(
        ref self: TContractState,
        model_id: felt252,
        weight_commitment: felt252,
        num_layers: u32,
        description: felt252,
    );

    /// Check if a proof has been verified.
    fn is_verified(self: @TContractState, proof_id: felt252) -> bool;

    /// Get the number of verifications for a model.
    fn get_model_verification_count(self: @TContractState, model_id: felt252) -> u32;

    /// Get the SAGE token address.
    fn get_sage_token(self: @TContractState) -> ContractAddress;

    /// Get the contract owner.
    fn get_owner(self: @TContractState) -> ContractAddress;
}
