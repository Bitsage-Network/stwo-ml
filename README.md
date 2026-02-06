<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██████╗ ██╗████████╗███████╗ █████╗  ██████╗ ███████╗                      ║
║    ██╔══██╗██║╚══██╔══╝██╔════╝██╔══██╗██╔════╝ ██╔════╝                      ║
║    ██████╔╝██║   ██║   ███████╗███████║██║  ███╗█████╗                        ║
║    ██╔══██╗██║   ██║   ╚════██║██╔══██║██║   ██║██╔══╝                        ║
║    ██████╔╝██║   ██║   ███████║██║  ██║╚██████╔╝███████╗                      ║
║    ╚═════╝ ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝                      ║
║                                                                               ║
║                 The Economic Heart of Decentralized Compute                   ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║    ███████╗████████╗██╗    ██╗ ██████╗     ██████╗ ██████╗  ██████╗ ██╗   ██╗███████╗██████╗  ║
║    ██╔════╝╚══██╔══╝██║    ██║██╔═══██╗    ██╔══██╗██╔══██╗██╔═══██╗██║   ██║██╔════╝██╔══██╗ ║
║    ███████╗   ██║   ██║ █╗ ██║██║   ██║    ██████╔╝██████╔╝██║   ██║██║   ██║█████╗  ██████╔╝ ║
║    ╚════██║   ██║   ██║███╗██║██║   ██║    ██╔═══╝ ██╔══██╗██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗ ║
║    ███████║   ██║   ╚███╔███╔╝╚██████╔╝    ██║     ██║  ██║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║ ║
║    ╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝     ╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝ ║
║                                                                               ║
║              GPU-Accelerated Circle STARK Prover • ZK Proofs • Starknet       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

# BitSage Stwo Fork

**GPU-Optimized Circle STARK Prover for Verifiable Compute**

[![Starknet](https://img.shields.io/badge/Starknet-Sepolia-blue?style=for-the-badge)](https://sepolia.starkscan.co)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-H100-green?style=for-the-badge)](https://www.nvidia.com/en-us/data-center/h100/)
[![License](https://img.shields.io/badge/License-Apache_2.0-orange?style=for-the-badge)](LICENSE)

*Fork of [StarkWare's Stwo](https://github.com/starkware-libs/stwo) with GPU acceleration optimizations for the BitSage Network*

</div>

---

## Overview

This is BitSage Network's fork of the Stwo Circle STARK prover, optimized for GPU-accelerated proof generation. Our modifications enable **real-time verifiable compute** with on-chain proof verification on Starknet.

### What's Different from Upstream Stwo?

| Feature | Upstream Stwo | BitSage Fork |
|---------|---------------|--------------|
| **GPU Backend** | Experimental | Production-ready |
| **FRI Folding** | CPU-only | GPU-accelerated |
| **Memory Management** | Standard | GPU memory pool |
| **Starknet Integration** | None | Full pipeline |
| **On-chain Verification** | None | StwoVerifier contract |

---

## Latest Benchmark Results

**Hardware:** NVIDIA H100 PCIe 80GB
**Network:** Starknet Sepolia
**Date:** February 3, 2026

### Proof Generation Performance

| Workload | Trace Steps | GPU Time | CPU Time | Speedup | FRI Layers |
|----------|-------------|----------|----------|---------|------------|
| **ML Inference** | 132 | **21ms** | 18ms | 1.0x | 8 |
| **1K Steps** | 1,024 | **24ms** | 20ms | 0.8x | 10 |
| **64K Steps** | 65,536 | **159ms** | 164ms | 1.0x | 16 |
| **256K Steps** | 262,144 | **335ms** | 352ms | 1.1x | 18 |
| **1M Steps** | 1,048,576 | **1,107ms** | 1,125ms | 1.0x | 20 |

### On-Chain Verification

| Metric | Value |
|--------|-------|
| **Events per TX** | 12 |
| **Gas Cost (small proof)** | 0.307 STRK |
| **Gas Cost (large proof)** | 0.429 STRK |
| **Verification Success Rate** | 100% |
| **Calldata Size** | 173-333 felts |

### Verified Transactions (Starknet Sepolia)

```
ML_GPU:   https://sepolia.voyager.online/tx/0x068545dbe5b18a52328b0c0b74a661c6f0f7f689d4847247b055bd217a46cf53
ML_CPU:   https://sepolia.voyager.online/tx/0x051ee2466af84d94b439fae15bcb1662317a4a7116ee3e7ccb3a3f07ae731eac
GPU_1K:   https://sepolia.voyager.online/tx/0x03962dcd9b61dbcd7e5f24fab76132ad29ba4c6ba6e3b667b7f78055ee876e72
CPU_1K:   https://sepolia.voyager.online/tx/0x06661111810232815e84995dd64a4c69d7c027c00a4516a040dee5664c984528
GPU_64K:  https://sepolia.voyager.online/tx/0x03cc26baf34abbed4c753ce60e53854d8728723a73acc3f7fa9f687fc6f9bfb1
GPU_256K: https://sepolia.voyager.online/tx/0x0384d3daa5f08e083115c228b91d19a2a79d3d73117eb57f666f9ec8b3574607
GPU_1M:   https://sepolia.voyager.online/tx/0x05d0ae5280523e1ec31802a8aa7ffec28eea943c498d7b1694a495087557eec9
CPU_1M:   https://sepolia.voyager.online/tx/0x03494f9bd7eb9e5a1b323b12e0478d12876d8c943b9b92035b61d824ecd8a2fe
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BitSage Proof Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  ObelyskVM   │───►│  Stwo Prover │───►│   Starknet   │          │
│  │  (rust-node) │    │  (this repo) │    │  (on-chain)  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         │                   │                   │                   │
│    Execution            Proof Gen          Verification            │
│    Trace                                                            │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                    GPU Acceleration                       │      │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐         │      │
│  │  │  FFT   │  │  FRI   │  │ Merkle │  │ Memory │         │      │
│  │  │ Circle │  │ Folding│  │ Commit │  │  Pool  │         │      │
│  │  └────────┘  └────────┘  └────────┘  └────────┘         │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### On-Chain Multicall Flow

Each proof submission generates **12 events** through this call cascade:

```
__execute__ (account multicall)
├── Call 1: ProofGatedPayment.register_job_payment
│   └── Event: JobPaymentRegistered
├── Call 2: StwoVerifier.submit_and_verify_with_io_binding
│   ├── Event: ProofSubmitted
│   ├── Event: ProofVerified
│   └── Event: ProofLinkedToJob
├── Call 3: PaymentRouter.register_job
│   └── Event: JobRegistered
└── Call 4: PaymentRouter.pay_with_sage
    ├── Event: PaymentExecuted
    ├── Event: WorkerPaid
    ├── SAGE.Transfer (worker 80%)
    ├── SAGE.Transfer (treasury 18%)
    └── SAGE.Transfer (stakers 2%)
```

---

## Key Modifications

### 1. GPU FRI Folding (`crates/prover/src/core/backend/gpu/fri.rs`)

Optimized FRI folding to minimize CPU-GPU data transfers:

```rust
// GPU-resident folding - data stays on GPU between rounds
let d_output = executor.execute_fold_line_gpu_only(
    &d_cached, &d_itwiddles, &alpha_u32, n
)?;
```

### 2. Memory Pool (`crates/prover/src/core/backend/gpu/memory_pool.rs`)

Pre-allocated GPU memory pool to avoid allocation overhead:

```rust
pub struct GpuMemoryPool {
    pools: Vec<CudaSlice<u32>>,
    available: Vec<bool>,
    block_size: usize,
}
```

### 3. Circle FFT Optimizations

CUDA kernels for Circle-group FFT operations over the M31 field.

### 4. Proof Packing (`rust-node/src/obelysk/proof_packer.rs`)

Efficient serialization for Starknet calldata:

```rust
pub fn pack_proof(proof: &StarkProof) -> Result<PackedProof> {
    // Serialize to Cairo-compatible format
    // Output: 173-333 felts depending on proof size
}
```

---

## Deployed Contracts (Starknet Sepolia)

| Contract | Address | Purpose |
|----------|---------|---------|
| **StwoVerifier** | `0x575968af96f814da648442daf1b8a09d43b650c06986e17b2bab7719418ddfb` | Proof verification |
| **ProofGatedPayment** | `0x7e74d191b1cca7cac00adc03bc64eaa6236b81001f50c61d1d70ec4bfde8af0` | Payment gating |
| **PaymentRouter** | `0x001a7c5974eaa8a4d8c145765e507f73d56ee1d05419cbcffcae79ed3cd50f4d` | Fee distribution |
| **OracleWrapper** | `0x4d86bb472cb462a45d68a705a798b5e419359a5758d84b24af4bbe5441b6e5a` | Price feeds |
| **SAGE Token** | `0x072349097c8a802e7f66dc96b95aca84e4d78ddad22014904076c76293a99850` | Native token |

---

## Building

### Prerequisites

- Rust 1.75+
- CUDA Toolkit 12.x (for GPU support)
- NVIDIA Driver 535+

### CPU Build

```bash
cargo build --release
```

### GPU Build

```bash
cargo build --release --features cuda
```

### Run Benchmarks

```bash
# Single proof benchmark
cargo bench --features cuda

# Full pipeline benchmark (requires Starknet RPC)
cd ../rust-node
cargo run --release --bin benchmark_proof_pipeline
```

---

## Integration with rust-node

This Stwo fork is used by the BitSage rust-node via the `stwo_adapter`:

```rust
// rust-node/src/obelysk/stwo_adapter.rs
use stwo_prover::core::prover::prove;

pub fn prove_with_stwo_gpu(trace: &ExecutionTrace) -> Result<StarkProof> {
    // 1. Convert ObelyskVM trace to Stwo AIR
    // 2. Generate proof using GPU backend
    // 3. Return serializable proof
}
```

---

## Fee Distribution Model

When proofs are verified on-chain, fees are distributed:

| Recipient | Share | Description |
|-----------|-------|-------------|
| **Worker** | 80% | GPU operator reward |
| **Treasury** | 18% | Protocol development |
| **Stakers** | 2% | SAGE staker rewards |

---

## Upstream Sync Policy

We maintain compatibility with upstream Stwo while adding our optimizations:

```bash
# Add upstream remote
git remote add upstream https://github.com/starkware-libs/stwo.git

# Fetch and merge
git fetch upstream
git merge upstream/main --no-commit
```

Our modifications are primarily in:
- `crates/prover/src/core/backend/gpu/` - GPU optimizations
- Integration is handled in `rust-node/src/obelysk/`

---

## Related Repositories

| Repository | Description |
|------------|-------------|
| [rust-node](../rust-node) | BitSage GPU Worker Node |
| [BitSage-Cairo-Smart-Contracts](../../BitSage-Cairo-Smart-Contracts) | Starknet contracts |
| [stwo-cairo](../stwo-cairo) | Cairo verifier components |

---

## License

Apache 2.0 - Same as upstream Stwo

---

<div align="center">

**BitSage Network** - Verifiable Compute at Scale

[Website](https://bitsage.network) | [Discord](https://discord.gg/bitsage) | [Twitter](https://twitter.com/bitsagenetwork) | [Paper](https://eprint.iacr.org/2024/278) | [Benchmarks](https://starkware-libs.github.io/stwo/dev/bench/index.html)

</div>
>
> Stwo is currently under active development and is not recommended for production use.

## Table of Contents

- [About](#-about)
- [Key Features](#-key-features)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Building](#building)
  - [Running Examples](#running-examples)
- [Project Structure](#-project-structure)
- [Real-World Applications](#-real-world-applications)
- [Benchmarks](#-benchmarks)
- [License](#-license)

## 🌟 About

**Stwo** is a next-generation implementation of a [Circle STARK (CSTARK)](https://eprint.iacr.org/2024/278) prover and verifier framework, written in Rust 🦀. While it includes example implementations for demonstration purposes, the **primary purpose of this repository is to serve as a foundational framework** for building more complex, production-grade provers.

The framework provides a high-performance, flexible foundation for generating and verifying STARK proofs using the latest cryptographic research and innovations in STARK proof systems. It is designed to be extended and customized for specific use cases beyond the simple examples included in this repository.

## 🚀 Key Features

- **Circle STARKs:** Implementation based on the latest cryptographic research and innovations in STARK proof systems
- **High Performance:** Designed for extreme speed and efficiency with optimized SIMD operations

## 🛠️ Getting Started

### Prerequisites

- **Rust Toolchain:** This project requires Rust nightly (see `rust-toolchain.toml` for the specific version)

### Building

Clone the repository and build the project:

```bash
git clone https://github.com/starkware-libs/stwo.git
cd stwo
cargo build --release
```

### Running Examples

The project includes example implementations in the `crates/examples` directory:

```bash
# Run examples
cargo run --release --example <example_name>
```

## 📁 Project Structure

This is a Rust workspace containing multiple crates:

- **`crates/stwo`** - Core library implementing the Circle STARK prover and verifier
- **`crates/constraint-framework`** - Framework for expressing and evaluating constraints
- **`crates/air-utils`** - Utilities for working with Arithmetic Intermediate Representation (AIR)
- **`crates/air-utils-derive`** - Derive macros for AIR utilities
- **`crates/examples`** - Example implementations demonstrating basic usage
- **`crates/std-shims`** - Standard library shims for `no_std` compatibility

## 🌍 Real-World Applications

While the examples in this repository demonstrate basic usage patterns, Stwo is designed as a **framework for building sophisticated provers** that handle complex real-world scenarios. A prominent example of such an application is:

### [stwo-cairo](https://github.com/starkware-libs/stwo-cairo)

**stwo-cairo** is a production-oriented prover built on top of the Stwo framework that enables proving Cairo program executions.

## 📊 Benchmarks

### Quick Benchmark

Run a single-threaded Poseidon2 hash proof benchmark:

```bash
./poseidon_benchmark.sh
```

### Comprehensive Benchmarks

Run all benchmarks using Cargo:

```bash
cargo bench
```

### Benchmark Reports

Visual representation of benchmark results can be found at:
[https://starkware-libs.github.io/stwo/dev/bench/index.html](https://starkware-libs.github.io/stwo/dev/bench/index.html)

## 📜 License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for more information.
>>>>>>> upstream/dev
