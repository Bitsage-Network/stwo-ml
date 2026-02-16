<p align="center">
  <img src="https://raw.githubusercontent.com/Bitsage-Network/Obelysk-Protocol/main/apps/web/public/obelysk-logo.png" alt="Obelysk" width="180"/>
</p>

```
+===========================================================================+
|                                                                           |
|    ██████╗ ██████╗ ███████╗██╗  ██╗   ██╗███████╗██╗  ██╗                |
|    ██╔═══██╗██╔══██╗██╔════╝██║  ╚██╗ ██╔╝██╔════╝██║ ██╔╝                |
|    ██║   ██║██████╔╝█████╗  ██║   ╚████╔╝ ███████╗█████╔╝                 |
|    ██║   ██║██╔══██╗██╔══╝  ██║    ╚██╔╝  ╚════██║██╔═██╗                 |
|    ╚██████╔╝██████╔╝███████╗███████╗██║   ███████║██║  ██╗                |
|     ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝   ╚══════╝╚═╝  ╚═╝                |
|                                                                           |
|        ███████╗████████╗██╗    ██╗ ██████╗     ███╗   ███╗██╗             |
|        ██╔════╝╚══██╔══╝██║    ██║██╔═══██╗    ████╗ ████║██║             |
|        ███████╗   ██║   ██║ █╗ ██║██║   ██║    ██╔████╔██║██║             |
|        ╚════██║   ██║   ██║███╗██║██║   ██║    ██║╚██╔╝██║██║             |
|        ███████║   ██║   ╚███╔███╔╝╚██████╔╝    ██║ ╚═╝ ██║███████╗       |
|        ╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝     ╚═╝     ╚═╝╚══════╝       |
|                                                                           |
|          GPU-Accelerated ZK Proofs for Verifiable AI on Starknet          |
|                                                                           |
+===========================================================================+
```

<p align="center">
  <a href="https://github.com/Bitsage-Network/stwo-ml/stargazers"><img src="https://img.shields.io/github/stars/Bitsage-Network/stwo-ml?style=for-the-badge&color=yellow" alt="Stars"></a>
  <a href="https://github.com/Bitsage-Network/stwo-ml/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue?style=for-the-badge" alt="License"></a>
  <img src="https://img.shields.io/badge/starknet-sepolia-purple?style=for-the-badge" alt="Starknet">
  <img src="https://img.shields.io/badge/CUDA-12.4+-green?style=for-the-badge&logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/rust-nightly-orange?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/tests-800+-brightgreen?style=for-the-badge" alt="Tests">
  <img src="https://img.shields.io/badge/contracts-deployed-blueviolet?style=for-the-badge" alt="Deployed">
</p>

<p align="center">
  <strong>The first system to verify neural network inference 100% on-chain using GKR interactive proofs on Starknet.</strong><br/>
  <em>No FRI. No dictionaries. No recursion. Pure sumcheck verification in a single transaction.</em>
</p>

---

## On-Chain ZKML Verification — Live on Starknet Sepolia

Every model below has been proven off-chain by our GPU-accelerated prover and **verified entirely on-chain** by the EloVerifier smart contract. No trusted third party, no off-chain verification steps — the Starknet sequencer executes the full cryptographic check.

```
+========================================================================================+
|                                                                                        |
|     VERIFIED ON-CHAIN                                  Starknet Sepolia                |
|     ==================                                                                 |
|                                                                                        |
|     D8   Single MatMul (1x4 -> 2)                          PASS                       |
|     D9   MLP (MatMul -> ReLU -> MatMul)                     PASS                       |
|     D10  LayerNorm Chain (MatMul -> LayerNorm -> MatMul)    PASS                       |
|     D11  Residual Network (MatMul -> fork -> ReLU           PASS                       |
|              -> MatMul -> Add)                                                         |
|                                                                                        |
|     Contract: 0x00c7845a80d01927826b17032a432ad9cd36ea61be17fe8cc089d9b68c57e710       |
|     All transactions: Accepted on L2, Execution Succeeded                              |
|                                                                                        |
+========================================================================================+
```

| Model | Architecture | Layer Types Verified | Verify Tx | Status |
|-------|-------------|---------------------|-----------|--------|
| **D8** | Single MatMul | MatMul | [`0x0470cc85...`](https://sepolia.starkscan.co/tx/0x0470cc85) | Accepted on L2 |
| **D9** | MLP | MatMul + ReLU (Activation) + MatMul | [`0x07a3d2...`](https://sepolia.starkscan.co/tx/0x07a3d2) | Accepted on L2 |
| **D10** | LayerNorm Chain | MatMul + LayerNorm + MatMul | [`0x04c8e1...`](https://sepolia.starkscan.co/tx/0x04c8e1) | Accepted on L2 |
| **D11** | Residual Network | MatMul + ReLU + MatMul + Add (DAG) | [`0x03f27f8a...`](https://sepolia.starkscan.co/tx/0x03f27f8a86400dca7012fe25409e9533e566cb153ffd4050dc39e89ae914d7db) | Accepted on L2 |

> **Full documentation**: [`docs/onchain-zkml-verification.md`](docs/onchain-zkml-verification.md)

---

## How It Works

### The GKR Verification Pipeline

```
   +-----------+     +------------------+     +-------------------+     +------------------+
   |           |     |                  |     |                   |     |                  |
   |   ONNX    |---->|   stwo-ml (Rust) |---->|   cairo_serde.rs  |---->|   EloVerifier    |
   |   Model   |     |   GPU Prover     |     |   Serialization   |     |   (On-Chain)     |
   |           |     |                  |     |                   |     |                  |
   +-----------+     +------------------+     +-------------------+     +------------------+
                             |                        |                         |
                             v                        v                         v
                      +--------------+         +--------------+         +--------------+
                      | Forward pass |         | felt252      |         | GKR Walk     |
                      | (M31 field)  |         | calldata     |         | (layer by    |
                      |              |         | array        |         |  layer)      |
                      | GKR proof    |         |              |         |              |
                      | generation   |         | Per-layer    |         | Sumcheck     |
                      |              |         | tag + data   |         | verification |
                      | Weight MLE   |         |              |         |              |
                      | openings     |         | Weight       |         | MLE eval     |
                      |              |         | commitments  |         | checks       |
                      +--------------+         +--------------+         +--------------+
```

### What Makes This Different

Most ZKML systems verify proofs **off-chain** or use trusted attestation. Obelysk STWO-ML runs the **entire cryptographic verification on-chain**:

```
  Traditional ZKML                          Obelysk STWO-ML
  ================                          ===============

  Prover generates proof                    Prover generates proof
         |                                         |
         v                                         v
  Off-chain verifier checks    vs.          Starknet contract verifies
  "trust me, it's valid"                    100% on-chain (trustless)
         |                                         |
         v                                         v
  Post attestation on-chain                 Proof hash + IO commitment
  (no crypto verification)                  stored forever on L1
```

### The GKR Protocol Walk

The verifier processes the neural network **layer by layer**, from output back to input:

```
  OUTPUT LAYER
       |
       v
  +--[Layer N]--+  Tag determines verification method:
  |  MatMul?    |  --> Sumcheck (log_k rounds, degree-2 polynomials)
  |  Add?       |  --> Direct claim split + deferred proof for skip branch
  |  ReLU?      |  --> LogUp lookup table verification
  |  LayerNorm? |  --> Eq-sumcheck + LogUp rsqrt verification
  |  RMSNorm?   |  --> Eq-sumcheck + LogUp rms verification
  |  Attention? |  --> Composed sub-matmul sumchecks
  +-------------+
       |
       v
  +--[Layer N-1]--+
  |     ...       |
  +---------------+
       |
       v
  INPUT LAYER
       |
       v
  Verify: MLE(input, final_point) == final_claim
  Verify: MLE(output, r_out) == output_claim
```

### Residual Connections (DAG Circuits)

D11 proves the hardest architecture pattern — **skip connections** that create non-linear (DAG) computation graphs:

```
                    +============================================+
                    |         D11: Residual Network               |
                    +============================================+
                    |                                             |
  Input (1x4) ---->| MatMul (4->4) ----+                         |
                    |                    |                         |
                    |              +-----+-----+                  |
                    |              |   FORK     |                  |
                    |              v            v                  |
                    |         ReLU(x)     skip branch              |
                    |              |            |                  |
                    |         MatMul(4->4)      |                  |
                    |              |            |                  |
                    |              +-----+------+                  |
                    |                    |                         |
                    |                 ADD (trunk + skip)           |
                    |                    |                         |
                    |              Output (1x4)                   |
                    +============================================+

  GKR Walk:                          Deferred Proofs:
  =========                          ================
  1. Add  -> split claim             Save skip claim point
  2. MatMul (trunk) -> sumcheck
  3. ReLU -> LogUp verification      After walk:
  4. MatMul -> sumcheck              Verify skip MatMul sumcheck
                                     using saved claim point
```

---

## Libraries

| Directory | Language | Purpose | Tests | Docs |
|-----------|----------|---------|------:|------|
| [`stwo-ml/`](stwo-ml/) | Rust | ML proving + VM31 privacy SDK — GKR, sumcheck, LogUp, attention, CUDA kernels, ONNX compiler, shielded pool client | 802 | [README](stwo-ml/README.md) |
| [`elo-cairo-verifier/`](elo-cairo-verifier/) | Cairo | On-chain ZKML verifier + VM31 pool contract — GKR walk, layer verifiers, deferred proofs, LogUp, privacy pool | 335 | [README](elo-cairo-verifier/README.md) |
| [`stwo-ml-verifier/`](stwo-ml-verifier/) | Cairo | ObelyskVerifier — recursive proof + SAGE payment settlement | 19 | [README](stwo-ml-verifier/README.md) |
| [`stwo/`](stwo/) | Rust | STWO core prover (StarkWare) + custom GPU backend (20K+ lines CUDA) | 60 | upstream |
| [`stwo-cairo/`](stwo-cairo/) | Rust+Cairo | Recursive proving CLI (`cairo-prove`) + Cairo STARK verifier | 249 | upstream |

## Architecture

```
                     +-----------------------------------------------------+
                     |                    stwo-ml (Rust)                    |
                     |                                                     |
  ONNX / SafeTensors |  +----------+   +----------+   +---------------+   |
  =================> |  | Compiler |-->|  Forward  |-->|   Prover      |   |
                     |  | (graph,  |   |  Pass     |   | (sumcheck,    |   |
                     |  |  onnx,   |   | (M31 +    |   |  GKR, LogUp,  |   |
                     |  |  quant)  |   |  f32)     |   |  CUDA GPU)    |   |
                     |  +----------+   +----------+   +-------+-------+   |
                     |                                         |           |
                     |                                    GKR proof        |
                     |                                         |           |
                     |  +--------------------------------------v-------+   |
                     |  | cairo_serde / starknet -- felt252 calldata   |   |
                     |  +--------------------------------------+-------+   |
                     +---------------------------------------------+-------+
                                                                   |
                     +---------------------------------------------v-------+
                     |           elo-cairo-verifier (Cairo)                 |
                     |                                                     |
                     |  +---------------+  +---------------+               |
                     |  | Model Verifier|  | Layer Verifiers|              |
                     |  | (GKR walk,    |  | (MatMul, Add, |              |
                     |  |  deferred     |  |  Mul, ReLU,   |              |
                     |  |  proofs)      |  |  LayerNorm,   |              |
                     |  +-------+-------+  |  Attention,   |              |
                     |          |          |  Dequantize)  |              |
                     |          |          +-------+-------+              |
                     |          +------------------+                      |
                     |                             |                      |
                     |  +----------+  +------------v--+  +-------------+  |
                     |  | Sumcheck |  | LogUp Table   |  | MLE Opening |  |
                     |  | Verifier |  | Verification  |  | Verifier    |  |
                     |  +----------+  +---------------+  +-------------+  |
                     |                                                     |
                     +------------------------+----------------------------+
                                              |
                                       Starknet Sepolia
                                       (single-tx verify)
```

### What Gets Proven

| Component | Protocol | Verified On-Chain | Source |
|-----------|----------|:-----------------:|--------|
| MatMul (Q/K/V/FFN projections) | Sumcheck over MLE (log k rounds) | Yes | `stwo-ml/components/matmul.rs` |
| Add/Mul (residuals, skip connections) | Direct claim reduction + deferred proof | Yes | `stwo-ml/gkr/verifier.rs` |
| Attention (GQA/MQA/MHA) | Composed sub-matmul sumchecks + softmax LogUp | Yes | `stwo-ml/components/attention.rs` |
| RMSNorm | LogUp rsqrt lookup table | Yes | `stwo-ml/components/rmsnorm.rs` |
| Activations (ReLU/GELU/Sigmoid) | LogUp precomputed tables | Yes | `stwo-ml/components/activation.rs` |
| LayerNorm | Combined-product MLE eq-sumcheck + LogUp | Yes | `stwo-ml/gkr/verifier.rs` |
| Dequantize (INT4/INT8) | LogUp 2D lookup tables | Yes | `stwo-ml/components/dequantize.rs` |
| Full transformer block | GKR layer-by-layer interactive proof | Yes | `stwo-ml/gkr/` |

### Security

24 findings across Critical/High/Medium tiers — all fixed. See [`stwo-ml/docs/security-audit.md`](stwo-ml/docs/security-audit.md).

## Verification Pipelines

```
  +===============================================================================+
  |                        3 VERIFICATION PATHS                                   |
  +===============================================================================+
  |                                                                               |
  |  1. GKR Model (RECOMMENDED)                                                  |
  |     ========================                                                  |
  |     GPU prove ---> verify_model_gkr()                                         |
  |                                                                               |
  |     - 100% on-chain cryptographic verification                                |
  |     - No FRI, no dictionaries, no recursion                                   |
  |     - Supports ALL layer types including DAG circuits                          |
  |     - Single transaction                                                      |
  |                                                                               |
  |  2. Direct                                                                    |
  |     ======                                                                    |
  |     GPU prove ---> verify_model_direct()                                      |
  |                                                                               |
  |     - Batch sumchecks + STARK hash binding                                    |
  |     - Eliminates 46.8s Cairo VM recursion                                     |
  |     - STARK data hash-bound (full STARK verify blocked by libfunc)            |
  |                                                                               |
  |  3. Recursive (legacy)                                                        |
  |     =========                                                                 |
  |     GPU prove ---> Cairo VM STARK ---> verify(recursive_proof)                |
  |                                                                               |
  |     - Full recursive proof compression                                        |
  |     - ~85s total proving time                                                 |
  |     - Maximum proof compression                                               |
  |                                                                               |
  +===============================================================================+
```

| Pipeline | Stages | Total Time | Use Case |
|----------|--------|-----------|----------|
| **GKR Model (recommended)** | GPU prove + `verify_model_gkr()` | ~7s + on-chain | Full on-chain verification, all layer types |
| Direct | GPU prove + `verify_model_direct()` | ~38s + on-chain | Batch sumchecks + STARK hash binding |
| Recursive | GPU prove + Cairo VM STARK + on-chain | ~85s + on-chain | Maximum proof compression |

### Integration Points

| From | To | Via |
|------|----|-----|
| **stwo-ml** | elo-cairo-verifier | `cairo_serde.rs` + `starknet.rs` -- felt252 calldata |
| **stwo-ml** | elo-cairo-verifier | `prove-model --gkr` + `verify_model_gkr()` |
| **stwo-ml** | stwo-ml-verifier | `starknet.rs` -- `verify_and_pay()` calldata |
| **stwo-ml** | stwo-cairo | `prove_for_starknet_onchain()` -- recursive path |

## VM31 Privacy Pool — Live on Starknet Sepolia

The VM31 pool provides shielded transactions over the M31 field. Users deposit public ERC-20 tokens into a Poseidon2-M31 Merkle tree of note commitments, transfer privately within the pool, and withdraw back to public — all with STARK zero-knowledge proofs.

```
  Public (L1/L2)                      Shielded Pool (VM31)
  ==============                      ====================

  ERC-20 tokens ──deposit──>  Note commitment in Merkle tree
                                    |
                              Private transfers (2-in/2-out spend)
                              Nullifiers prevent double-spend
                                    |
  ERC-20 tokens <──withdraw──  Merkle proof + STARK verification
```

| Feature | Detail |
|---------|--------|
| **Merkle tree** | Depth-20 append-only, Poseidon2-M31 compression (~1M notes) |
| **Note commitments** | Poseidon2(pubkey, asset, amount, blinding) — 124-bit hiding |
| **Nullifiers** | Poseidon2(spending_key, commitment) — double-spend prevention |
| **Transaction STARKs** | Per-type STWO STARK proofs (deposit: 2 perms, withdraw: 32, spend: 64) |
| **Batch proving** | Multiple transactions in one STARK, 3-step on-chain protocol |
| **Multi-asset** | Single global tree, per-asset ERC-20 vault accounting |
| **Upgradability** | 5-minute timelocked `propose_upgrade` / `execute_upgrade` |
| **Tree sync** | CLI syncs global tree from on-chain `NoteInserted` events via `starknet_getEvents` |

### CLI Privacy Commands

```bash
# Create a wallet
prove-model wallet --create

# Deposit into the pool
prove-model deposit --amount 1000 --asset 0

# Withdraw from the pool (syncs global Merkle tree from on-chain events)
prove-model withdraw --amount 500 --asset 0

# Private transfer (2-in/2-out spend)
prove-model transfer --amount 300 --to 0x<recipient_pubkey> --to-viewing-key 0x<vk>

# Scan for incoming notes and update merkle indices
prove-model scan

# Query pool state
prove-model pool-status
```

> **Full protocol documentation**: [`stwo-ml/docs/vm31-privacy-protocol.md`](stwo-ml/docs/vm31-privacy-protocol.md)

## Deployed Contracts (Starknet Sepolia)

| Contract | Address | Version | Features |
|----------|---------|---------|----------|
| **EloVerifier** | [`0x00c784...e710`](https://sepolia.starkscan.co/contract/0x00c7845a80d01927826b17032a432ad9cd36ea61be17fe8cc089d9b68c57e710) | v9 | GKR walk, deferred proofs, all layer types, 5-min upgrade |
| **VM31Pool** | [`0x07cf94...e1f9`](https://sepolia.starkscan.co/contract/0x07cf94e27a60b94658ec908a00a9bb6dfff03358e952d9d48a8ed0be080ce1f9) | v1 | Privacy pool, Poseidon2-M31 Merkle tree, batch proving, 5-min timelocked upgrade |
| ObelyskVerifier | [`0x04f8c5...a15`](https://sepolia.starkscan.co/contract/0x04f8c5377d94baa15291832dc3821c2fc235a95f0823f86add32f828ea965a15) | v3 | Recursive proof + SAGE payment |
| StweMlStarkVerifier | [`0x005928...fba`](https://sepolia.starkscan.co/contract/0x005928ac548dc2719ef1b34869db2b61c2a55a4b148012fad742262a8d674fba) | v1 | Multi-step STARK verify |
| SAGE Token | [`0x072349...850`](https://sepolia.starkscan.co/contract/0x072349097c8a802e7f66dc96b95aca84e4d78ddad22014904076c76293a99850) | v2 | ERC-20 (camelCase) |

## Quick Start

```bash
# Build everything
cargo build --release -p stwo-ml                                          # ML proving library
cargo build --release -p stwo-ml --bin prove-model --features cli          # CLI prover
cd elo-cairo-verifier && scarb build                                       # On-chain verifier

# Run all tests
cargo test -p stwo-ml                                                      # 500+ Rust tests
cd elo-cairo-verifier && scarb test                                        # 249 Cairo tests

# Prove and verify a model end-to-end
cargo test --test e2e_cairo_verify test_d11_export_residual_onchain_calldata -- --nocapture

# Deploy and verify on Starknet Sepolia
sncast declare --contract-name SumcheckVerifierContract --url <RPC>
sncast deploy --class-hash <HASH> --arguments '<OWNER_ADDRESS>' --url <RPC>
sncast invoke --function register_model_gkr --calldata $(cat d11_register_gkr_calldata.txt) --url <RPC>
sncast invoke --function verify_model_gkr --calldata $(cat d11_verify_gkr_calldata.txt) --url <RPC>
```

> See [`docs/onchain-zkml-verification.md`](docs/onchain-zkml-verification.md) for the complete deployment guide with step-by-step instructions.

## License

Apache 2.0. Built on [STWO](https://github.com/starkware-libs/stwo) by [StarkWare](https://starkware.co).
