# Contributing to obelyzk.rs

Thank you for your interest in contributing to obelyzk.rs — the verifiable AI engine. Every contribution helps build a future where AI computation is cryptographically trustworthy.

## Getting Started

### Prerequisites

- **Rust**: nightly-2025-07-14 (`rustup toolchain install nightly-2025-07-14`)
- **CUDA 12+** (optional, for GPU acceleration)
- **Scarb 2.12.2** (optional, for Cairo contract development)

### Building

```bash
git clone https://github.com/Bitsage-Network/obelyzk.rs.git
cd obelyzk.rs/engine

# CPU-only build
cargo build --release --bin obelyzk --features server

# GPU build (requires CUDA)
cargo build --release --bin obelyzk --features "server,cuda-runtime"
```

### Running Tests

```bash
# Full test suite (950 tests, ~10 min on CPU)
cargo test --lib --features std

# Quick smoke test
cargo test --lib --features std -- test_matmul --test-threads=4

# Cairo verifier tests
cd ../elo-cairo-verifier && snforge test
```

## Repository Structure

```
obelyzk.rs/
├── engine/              The proving engine (crate: obelyzk)
│   ├── src/
│   │   ├── gkr/         GKR sumcheck protocol
│   │   ├── vm/          VM runtime (trace, executor, queue)
│   │   ├── providers/   Inference providers (local, OpenAI, Anthropic)
│   │   ├── components/  Proof primitives (MatMul, Attention, Norm, etc.)
│   │   ├── compiler/    Model loading + graph compilation
│   │   ├── recursive/   Recursive STARK compression
│   │   ├── policy.rs    Crypto-bound policy framework
│   │   └── starknet.rs  On-chain calldata serialization
│   └── tests/
│
├── stwo-gpu/            STWO STARK prover (our GPU-enhanced fork)
├── elo-cairo-verifier/  Recursive verifier contract (Cairo/Starknet)
├── stark-cairo/         STWO STARK verifier in Cairo
├── proof-stream/        WebSocket proof visualization
└── sdk/                 Python, TypeScript, Rust SDKs
```

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/Bitsage-Network/obelyzk.rs/issues)
- Include: Rust version, GPU model (if applicable), steps to reproduce
- For security vulnerabilities: email security@bitsage.network (do not open a public issue)

### Pull Requests

1. **Fork** the repo and create a branch from `development`
2. **Write tests** for any new functionality
3. **Run the full test suite** before submitting: `cargo test --lib --features std`
4. **Follow the existing code style** — no external formatters, match the surrounding code
5. **Keep PRs focused** — one feature or fix per PR
6. **Open a PR** against the `development` branch (not `main`)

### Code Style

- **No `clippy` enforcement** — we prioritize correctness and readability over lint rules
- **Minimal comments** — code should be self-explanatory. Comment the *why*, not the *what*
- **Feature gates** — use `#[cfg(feature = "...")]` for optional dependencies
- **Error handling** — use `thiserror` for library errors, descriptive messages
- **No unwrap in library code** — use `?` or explicit error handling. `unwrap()` is OK in tests and binaries

### Areas for Contribution

#### Good First Issues

- Improve error messages in model loading (`engine/src/compiler/hf_loader.rs`)
- Add new activation functions (`engine/src/components/activation.rs`)
- Expand test coverage for edge cases

#### Intermediate

- **GGUF format support** — load llama.cpp models into the M31 pipeline
- **New model architectures** — add support for Mamba, RWKV, or custom architectures
- **Metal shaders** — expand Apple Silicon GPU support
- **Vulkan backend** — AMD GPU support via compute shaders

#### Advanced

- **GPU Poseidon kernels** — move Fiat-Shamir hashing to GPU for 2-3x proving speedup
- **GKR circuit parallelism** — parallel branches for Q/K/V attention projections
- **TLS Notary integration** — replace proxy attestation with MPC-based TLS proofs
- **WASM target** — compile the prover to WebAssembly for browser-based proving
- **New STARK backends** — integrate with Plonky3, Binius, or other proving systems

## Architecture Decisions

### Why M31 Field Arithmetic?

The Mersenne-31 prime (2^31 - 1) enables:
- Native 32-bit operations on all hardware
- Efficient modular reduction (bitshift + add)
- 128-bit security via QM31 quartic extension
- Direct compatibility with STWO Circle STARKs

### Why GKR Instead of Per-Layer STARKs?

GKR (Goldwasser-Kalai-Rothblum) provides:
- Single interactive proof covering all layers
- Logarithmic verifier complexity (vs linear for separate STARKs)
- Natural composition with weight binding
- GPU-friendly sumcheck kernels

### Why Recursive STARK?

Compresses the GKR proof (46,148 felts) into ~942 felts:
- One Starknet transaction instead of 18 streaming TXs
- 49x compression ratio
- Fixed-size proof regardless of model depth

## Commit Convention

We use descriptive commit messages:

```
feat: tokenizer integration — text-in/text-out proving
fix: gate multi_gpu::discover_devices behind multi-gpu feature
perf: warp-shuffle sumcheck reduction
docs: obelyzk.rs root README
refactor: rename directories — clean obelyzk.rs structure
```

Prefixes: `feat:`, `fix:`, `perf:`, `docs:`, `refactor:`, `test:`, `chore:`

## Community

- **Website**: [obelysk.xyz](https://obelysk.xyz)
- **Network**: [bitsage.network](https://bitsage.network)
- **Issues**: [GitHub Issues](https://github.com/Bitsage-Network/obelyzk.rs/issues)

## License

By contributing to obelyzk.rs, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).

---

**Thank you for helping build verifiable AI.**
