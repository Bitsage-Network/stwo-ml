//! # stwo-ml: ML Inference Proving on Circle STARKs
//!
//! ML-specific proving circuits built on STWO — the fastest STARK prover in the world.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  stwo-ml (this crate)                           │
//! │                                                  │
//! │  components/           ML AIR components         │
//! │  ├── matmul.rs         Sumcheck-based matmul    │
//! │  ├── activation.rs     LogUp-based non-linear   │
//! │  ├── attention.rs      Composed QKV attention    │
//! │  └── layernorm.rs      Normalization gadget      │
//! │                                                  │
//! │  gadgets/              Reusable constraint gadgets│
//! │  ├── range_check.rs    M31 range proofs          │
//! │  ├── lookup_table.rs   Precomputed function tables│
//! │  └── quantize.rs       INT8/FP8 quantization     │
//! │                                                  │
//! │  compiler/             Model → Circuit compiler  │
//! │  ├── onnx.rs           ONNX model import         │
//! │  └── graph.rs          Computation graph builder  │
//! │                                                  │
//! ├──────────────────────────────────────────────────┤
//! │  stwo (GPU backend)    Circle FFT, FRI, Merkle   │
//! │  stwo (constraint-fw)  LogUp, Sumcheck, GKR      │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Innovations
//!
//! - **Sumcheck-based MatMul**: Verify matrix multiplication in O(n) instead of O(n³)
//!   using multilinear extensions over the boolean hypercube.
//!
//! - **LogUp Activation Tables**: Non-linear operations (ReLU, GELU, sigmoid, softmax)
//!   via precomputed lookup tables verified with the LogUp protocol.
//!
//! - **M31 Integer Arithmetic**: Single-cycle field reduction on Mersenne-31 (2^31-1).
//!   2-4x faster per operation than 256-bit prime fields used by other zkML systems.
//!
//! - **GPU-Resident Proving**: Entire proof pipeline stays on GPU — one transfer in,
//!   one transfer out. CUDA Graphs eliminate kernel launch overhead.

pub mod components;
pub mod compiler;
pub mod gadgets;

/// Poseidon Merkle commitment and multilinear folding for MLE opening proofs.
#[cfg(not(target_arch = "wasm32"))]
pub mod commitment;

/// On-chain proof generation for Starknet's SumcheckVerifier contract.
#[cfg(not(target_arch = "wasm32"))]
pub mod starknet;

/// Re-export core STWO types used throughout stwo-ml.
pub mod prelude {
    pub use stwo::core::channel::{Blake2sChannel, Channel};
    #[cfg(not(target_arch = "wasm32"))]
    pub use stwo::core::channel::Poseidon252Channel;
    pub use stwo::core::fields::m31::{BaseField, M31};
    pub use stwo::core::fields::qm31::{QM31, SecureField};
    pub use stwo::core::fields::Field;
    pub use stwo::prover::backend::cpu::CpuBackend;
    pub use stwo::prover::backend::simd::SimdBackend;
    pub use stwo::prover::backend::{Col, Column, ColumnOps};
    pub use stwo::prover::lookups::mle::Mle;
    pub use stwo::prover::lookups::sumcheck::{
        partially_verify, prove_batch, MultivariatePolyOracle, SumcheckProof,
    };
    pub use stwo::prover::lookups::utils::UnivariatePoly;
}
