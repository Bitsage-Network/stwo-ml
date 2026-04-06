# Starknet Deployer Accounts and Contract Registry

**Last updated**: April 5, 2026
**Network**: Starknet Sepolia

---

## 1. Deployer Account v1 (DEPRECATED)

| Field | Value |
|-------|-------|
| **Address** | `0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344` |
| **Status** | DEPRECATED -- incompatible with Starknet v0.8 RPCs |
| **Class** | Legacy OpenZeppelin Account (pre-v0.13.4) |

**Why deprecated**: This account was deployed with an older OpenZeppelin account
class that does not support the v3 transaction format. When Starknet Sepolia
upgraded to v0.8 RPC endpoints, calls to `starknet_estimateFee` and
`starknet_addInvokeTransaction` began failing with `UNEXPECTED_ERROR` or
`Contract not found`. The account class does not implement the updated
`__validate__` and `__execute__` entry points required by the v0.8 spec.

Do not use this account for any new operations. Existing contracts deployed by
this account remain functional, but new transactions must come from v2.

---

## 2. Deployer Account v2 (ACTIVE)

| Field | Value |
|-------|-------|
| **Address** | `0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f` |
| **Private Key** | `0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef` |
| **Public Key** | Derived from the private key via Stark curve |
| **Class** | OpenZeppelin Account v0.14.0 |
| **TX Format** | v3 (compatible with v0.8 RPCs) |
| **Status** | ACTIVE |

**IMPORTANT**: The private key shown above is the development/testnet key used in
deployment scripts. For production mainnet deployment, this key MUST be rotated.
Never use testnet keys on mainnet.

### Account Constructor

The v2 account uses `starknet.js` v8.9.2 Account constructor with the options
object format:

```javascript
import { Account, RpcProvider } from "starknet";

const provider = new RpcProvider({ nodeUrl: RPC_URL });
const account = new Account({
  provider,
  address: ADDR,
  signer: PRIVATE_KEY,
});
```

Note: starknet.js v8.9.2 changed the `Account` constructor from positional
arguments to an options object. Older versions used
`new Account(provider, address, privateKey)`.

### Funding

The deployer account requires STRK tokens for gas. Declare operations for
contract classes cost approximately:

| Contract Size | Declare Cost |
|---------------|-------------|
| Lean v18b (~15.7K Sierra felts) | ~18 STRK |
| Full verifier | ~40 STRK |
| Recursive verifier v2 | ~20 STRK |

---

## 3. Contract Addresses

### 3.1 Recursive Verifier -- Phase 1 (Record-Based, on Sepolia)

| Field | Value |
|-------|-------|
| **Contract** | `0x707819dea6210ab58b358151419a604ffdb16809b568bf6f8933067c2a28715` |
| **Deployer** | v2 account |
| **Source** | `elo-cairo-verifier/src/recursive_verifier.cairo` |
| **Deploy script** | `scripts/deploy_recursive_v2.mjs` |
| **Constructor** | `{ owner: DEPLOYER_ADDRESS }` |
| **Status** | Phase 1 -- records proof hashes, validates public input bindings |

Entrypoints:
- `register_model_recursive(model_id, circuit_hash, weight_super_root)`
- `verify_recursive(model_id, io_commitment, stark_proof_data)`
- `is_recursive_proof_verified(proof_hash) -> bool`
- `get_recursive_verification_count(model_id) -> u64`
- `get_recursive_model_info(model_id) -> RecursiveModelInfo`

### 3.1b Recursive Verifier -- Fully Trustless (Pending Deploy)

| Field | Value |
|-------|-------|
| **Class hash** | `0x006d4ff2332af0f7b1ac4601e266f7bcd7ef3b529f72012677b15445289ce820` |
| **Status** | Verified on devnet (OODS + Merkle + FRI + PoW all pass). Pending Sepolia declaration. |
| **Requires** | Juno full node for class declaration (class exceeds Alchemy gateway limits) |

This class will replace the Phase 1 contract via the timelock upgrade mechanism.
Once deployed, a single Starknet transaction performs full cryptographic STARK
verification of the recursive proof.

### 3.2 Streaming GKR Verifier v32

| Field | Value |
|-------|-------|
| **Contract** | `0x376fa0c4a9cf3d069e6a5b91bad6e131e7a800f9fced49bd72253a0b0983039` |
| **Class hash** | `0x5dca646786c36f9d68bab802d5c5c4995c37aa7c25bfa59ff20144a283f0956` |
| **Deployer** | v2 account |
| **Source** | `elo-cairo-verifier/src/verifier.cairo` |
| **Deploy script** | `scripts/deploy_contract.mjs` |
| **Constructor** | `{ admin: DEPLOYER_ADDRESS }` |

Key entrypoints (streaming flow):
- `register_model_gkr(model_id, weight_commitments, circuit_descriptor)`
- `register_model_gkr_streaming_circuit(model_id, layer_tags)`
- `open_gkr_session(model_id, total_felts, circuit_depth, num_layers, weight_binding_mode, packed, io_packed)`
- `upload_gkr_chunk(session_id, chunk_idx, chunk_len, data...)`
- `seal_gkr_session(session_id)`
- `verify_gkr_stream_init(session_id, ...)`
- `verify_gkr_stream_init_output_mle(session_id, ...)`
- `verify_gkr_stream_layers(session_id, ...)`
- `verify_gkr_stream_weight_binding(session_id, ...)`
- `verify_gkr_stream_weight_binding_chunk(session_id, ...)`
- `verify_gkr_stream_finalize_input_mle(session_id, ...)`
- `verify_gkr_stream_finalize(session_id)`

### 3.3 GKR Verifier v4 (Previous)

| Field | Value |
|-------|-------|
| **Contract** | `0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005` |
| **Deployer** | v2 account |
| **Status** | Superseded by v32, still functional |

### 3.4 Previous Recursive Verifier Addresses

| Version | Address | Notes |
|---------|---------|-------|
| Recursive v1 | `0x16919296b3990c10db6d714a04d2b6a1f62f007ed93e1b5816de1033beb248c` | Superseded by Phase 1 contract |

### 3.5 Previous Class Hashes

| Version | Class Hash | Notes |
|---------|------------|-------|
| Fully trustless | `0x006d4ff2332af0f7b1ac4601e266f7bcd7ef3b529f72012677b15445289ce820` | Pending Sepolia declaration |
| v32 | `0x5dca646786c36f9d68bab802d5c5c4995c37aa7c25bfa59ff20144a283f0956` | Production streaming |
| v31 | `0x6a6b7a75d5ec1f63d715617d352bc0d353042b2a033d98fa28ffbaf6c5b5439` | 14/14 steps pass |
| v30 | `0x38e9f407...` | Diagnostic |
| v29 | `0x316aa715...` | -- |
| v26 | `0x4859cb47...` | -- |
| v24 | `0x77ccc67d...` | -- |

### 3.6 Model IDs

| Model ID | Description |
|----------|-------------|
| `0x4` | v4 test model |
| `0x7` | v18 model |
| `0x8` | v18b model |
| `0x9` | v18b lean model |

### 3.7 Ephemeral Contract

| Field | Value |
|-------|-------|
| **Contract** | `0x52c2f627d6dfc1a663247f3696300ff5a66716a18b2762913b37f22c684e1f7` |
| **Purpose** | Temporary testing |

---

## 4. RPC Endpoints

### Working

| Provider | URL | API Version |
|----------|-----|-------------|
| **Alchemy** (recommended) | `https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/<YOUR_KEY>` | v0.8 |
| **Juno** (self-hosted) | `http://localhost:6060` | v0.8 |

Alchemy is the primary RPC for all deployment and submission scripts. A Juno
full node is required for declaring large contract classes (see section 6.4).

### Not Working / Unreliable

| Provider | URL | Issue |
|----------|-----|-------|
| **Cartridge** | `https://api.cartridge.gg/x/starknet/sepolia` | Intermittent failures with v3 transactions |
| **BlastAPI** | Various | RPC compatibility issues with v0.8 spec |
| **Lava** | `https://json-rpc.starknet-sepolia.public.lavanet.xyz` | Rate limiting, occasional timeouts |

### Environment Variable

All scripts read the RPC URL from `STARKNET_RPC`:

```bash
export STARKNET_RPC="https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/<YOUR_KEY>"
```

Some scripts also accept `--rpc-url` as a CLI flag.

---

## 5. starknet.js Version

**Required**: `starknet@8.9.2` (or later 8.x)

### Why v8.9.2

- Starknet Sepolia upgraded to v0.8 RPC endpoints in early 2026.
- starknet.js v5.x and v6.x do not support the v0.8 RPC spec.
- starknet.js v8.x added full v0.8 support including:
  - v3 transaction format (`INVOKE_TXN_V3`)
  - Resource bounds for L2 gas, L1 gas, and L1 data gas
  - Updated Account constructor (options object instead of positional args)

### Account Constructor Change

```javascript
// starknet.js v6 (OLD -- do not use):
const account = new Account(provider, address, privateKey);

// starknet.js v8.9.2 (CURRENT):
const account = new Account({ provider, address: ADDR, signer: KEY });
```

### Gas Bounds

v3 transactions require explicit resource bounds. The submission scripts use
predefined gas profiles:

```javascript
const GAS = {
  small: {
    l2_gas: { max_amount: 0x5000000n, max_price_per_unit: 0x2cb417800n },
    l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n },
    l1_data_gas: { max_amount: 0x5000n, max_price_per_unit: 0x7ed13c779n },
  },
  med: {
    l2_gas: { max_amount: 0x20000000n, max_price_per_unit: 0x2cb417800n },
    l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n },
    l1_data_gas: { max_amount: 0x30000n, max_price_per_unit: 0x7ed13c779n },
  },
  wbStore: {  // Weight binding storage (~1000 felts)
    l2_gas: { max_amount: 0x30000000n, max_price_per_unit: 0x2cb417800n },
    l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n },
    l1_data_gas: { max_amount: 0x30000n, max_price_per_unit: 0x7ed13c779n },
  },
  wbFinal: {  // Weight binding verify (read all + check)
    l2_gas: { max_amount: 0x40000000n, max_price_per_unit: 0x2cb417800n },
    l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n },
    l1_data_gas: { max_amount: 0x50000n, max_price_per_unit: 0x7ed13c779n },
  },
};
```

### AVNU Paymaster (Gasless)

For sponsored transactions, set the AVNU API key:

```bash
export AVNU_API_KEY="your-api-key"
```

Paymaster URLs:
- Sepolia: `https://sepolia.paymaster.avnu.fi`
- Mainnet: `https://starknet.paymaster.avnu.fi`

The `streaming_submit.mjs` script supports `--mode gasless` for paymaster-sponsored
submissions using STRK token approval.

---

## 6. Deployment Procedures

### 6.1 Declare a New Contract Class

```bash
node scripts/declare_and_upgrade.mjs
```

This script:
1. Reads Sierra and CASM JSON from `/tmp/contract_class.json` and `/tmp/contract_casm.json`
2. Declares via `account.declare({ contract, casm })`
3. Proposes upgrade to the existing contract (`propose_upgrade`)
4. Outputs the new class hash

### 6.2 Deploy a New Contract

```bash
node scripts/deploy_contract.mjs
```

Uses the class hash defined in the script, deploys with a unique salt, and
outputs the contract address.

### 6.3 Deploy Recursive Verifier

```bash
node scripts/deploy_recursive_v2.mjs
```

Declares the recursive verifier class from `/tmp/recursive_class_v2.json`,
deploys with `{ owner: DEPLOYER_ADDRESS }`, registers a test model, and
submits a recursive proof in a single TX.

### 6.4 Declaring Large Classes via Juno

The fully trustless recursive verifier class exceeds Alchemy's gateway size
limits for `starknet_addDeclareTransaction`. To declare it, run a Juno full
node pointed at Sepolia:

```bash
# Install Juno
git clone https://github.com/NethermindEth/juno.git
cd juno && go build -o juno ./cmd/juno

# Run Juno on Sepolia (syncs from L1)
./juno --network sepolia --http --http.port 6060

# Wait for sync to complete (check via /health endpoint)
curl http://localhost:6060/health
```

Once synced, point the declare script at your Juno node:

```bash
export STARKNET_RPC="http://localhost:6060"
node scripts/declare_and_upgrade.mjs
```

Juno has no payload size limits, so the large class declaration succeeds.

### 6.5 starkli Compatibility Note

`starkli` v0.4.2 has RPC compatibility issues with Starknet Sepolia. All
deployment scripts use `starknet.js` instead. The `scripts/pipeline/lib/declare_v13.mjs`
module handles declaration via the JS SDK.

---

## 7. Environment Variables Summary

| Variable | Description |
|----------|-------------|
| `STARKNET_RPC` | RPC endpoint URL (required) |
| `STARKNET_ACCOUNT` | Deployer account address |
| `STARKNET_PRIVATE_KEY` | Deployer private key |
| `RECURSIVE_CONTRACT` | Recursive verifier contract address |
| `OBELYSK_RECURSIVE_SCRIPT` | Path to `submit_recursive.mjs` (for CLI integration) |
| `AVNU_API_KEY` | AVNU paymaster API key (gasless mode) |
| `CONTRACT_ADDRESS` | Target streaming contract address (defaults to v32 in scripts) |
