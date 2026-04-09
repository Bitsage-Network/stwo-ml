# stwo-ml-verifier

ObelyskVerifier contract for Starknet — verifies recursive STARK proofs of ML inference and processes SAGE token payments in a single atomic transaction.

This is the **trusted submitter** verification path (Path 1). The contract owner verifies recursive STARK proofs off-chain, then submits proof facts on-chain with payment settlement.

For **trustless** verification, see [`elo-cairo-verifier`](../elo-cairo-verifier/) which replays the full Fiat-Shamir transcript on-chain.

## Deployed (Starknet Sepolia)

| Contract | Address | Class Hash |
|----------|---------|------------|
| **ObelyskVerifier v3** | [`0x04f8c5...a15`](https://sepolia.voyager.online/contract/0x04f8c5377d94baa15291832dc3821c2fc235a95f0823f86add32f828ea965a15) | `0x56825b...2ec` |
| SAGE Token | [`0x07234...850`](https://sepolia.voyager.online/contract/0x072349097c8a802e7f66dc96b95aca84e4d78ddad22014904076c76293a99850) | `0x5e17a...5b7` |

## How It Works

```
GPU Inference (stwo-ml, ~40s)
    │
    ▼
Cairo ML Verifier (execution trace)
    │
    ▼
Recursive Circle STARK (cairo-prove, ~47s)
    │
    ▼
ObelyskVerifier.verify_and_pay()
    ├── Record proof fact on-chain
    ├── Transfer SAGE: caller → worker
    └── Emit 7 events for indexing
```

## Interface

```cairo
#[starknet::interface]
trait IObelyskVerifier<TContractState> {
    /// Verify recursive proof + pay worker in one transaction.
    /// tee_attestation_hash = 0 means no TEE was used.
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
        tee_attestation_hash: felt252,
    ) -> bool;

    /// Register a model for verification.
    fn register_model(
        ref self: TContractState,
        model_id: felt252,
        weight_commitment: felt252,
        num_layers: u32,
        description: felt252,
    );

    /// Read-only queries.
    fn is_verified(self: @TContractState, proof_id: felt252) -> bool;
    fn get_model_verification_count(self: @TContractState, model_id: felt252) -> u32;
    fn get_sage_token(self: @TContractState) -> ContractAddress;
    fn get_owner(self: @TContractState) -> ContractAddress;
}
```

## Events

| Event | Fields | Emitted When |
|-------|--------|-------------|
| `ModelRegistered` | model_id, weight_commitment, num_layers, description | `register_model()` |
| `JobCreated` | job_id, model_id, worker, sage_amount | `verify_and_pay()` start |
| `ProofSubmitted` | proof_hash, model_id, io_commitment | Proof fact recorded |
| `InferenceVerified` | proof_hash, model_id, num_layers | Verification confirmed |
| `PaymentProcessed` | job_id, sage_amount, from, to | SAGE transfer complete |
| `WorkerRewarded` | worker, sage_amount, model_id | Worker payment sent |
| `VerificationComplete` | job_id, proof_hash, model_id, total_verifications | All done |
| `TeeAttested` | proof_hash, attestation_hash | TEE attestation recorded |

## Storage

```cairo
owner: ContractAddress,                      // Trusted submitter
sage_token: ContractAddress,                 // SAGE ERC-20 contract
verified_proofs: Map<felt252, bool>,         // proof_id → verified
registered_models: Map<felt252, bool>,       // model_id → registered
model_weight_commitments: Map<felt252, felt252>,  // model_id → commitment
model_num_layers: Map<felt252, u32>,         // model_id → layers
verification_count: Map<felt252, u32>,       // model_id → count
completed_jobs: Map<felt252, bool>,          // job_id → done (anti-replay)
tee_attestation_hashes: Map<felt252, felt252>, // proof_id → TEE hash
total_verifications: u32,
total_sage_paid: u256,
```

## Building

```bash
# Build contract
scarb build

# Run tests (19 tests)
snforge test
```

Requires Scarb 2.12+ and `snforge_std` v0.54.0.

## Structure

```
stwo-ml-verifier/
├── src/
│   ├── contract.cairo      Main contract implementation
│   └── interfaces.cairo    IObelyskVerifier + IERC20 traits
├── tests/
│   └── test_obelysk_verifier.cairo  19 integration tests
└── Scarb.toml
```

## License

Apache 2.0
