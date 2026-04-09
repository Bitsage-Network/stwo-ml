mod assert;
mod component_prover;
mod cpu_domain;
mod logup;
pub mod relation_tracker;
mod simd_domain;

// GPU support (requires stwo gpu feature)
#[cfg(feature = "gpu")]
mod gpu_domain;
#[cfg(feature = "gpu")]
mod gpu_component_prover;

pub use assert::{assert_constraints_on_polys, assert_constraints_on_trace, AssertEvaluator};
pub use cpu_domain::CpuDomainEvaluator;
pub use logup::{FractionWriter, LogupColGenerator, LogupTraceGenerator};
pub use simd_domain::SimdDomainEvaluator;

// GPU exports
#[cfg(feature = "gpu")]
pub use gpu_domain::GpuDomainEvaluator;
#[cfg(feature = "gpu")]
pub use gpu_component_prover::{
    is_gpu_constraint_kernels_enabled,
    set_gpu_constraint_kernels_enabled,
    will_use_gpu_kernels,
};
