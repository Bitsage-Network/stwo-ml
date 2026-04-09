//! Dispatch function that selects Metal GPU or CPU for matmul.
//!
//! Drop-in replacement for `matmul_m31()` that uses Metal when available.

use crate::components::matmul::M31Matrix;

/// Perform M31 matrix multiplication using the best available backend.
///
/// Priority: Metal GPU → CPU SIMD
/// The Metal path falls back to CPU for small matrices automatically.
pub fn matmul_m31_auto(a: &M31Matrix, b: &M31Matrix) -> M31Matrix {
    super::matmul::gpu_matmul_m31_metal(a, b)
}

/// Check if Metal GPU is available and report info.
pub fn metal_info() -> String {
    match std::panic::catch_unwind(|| {
        let dev = super::device::MetalDevice::global();
        format!("{} ({}MB)", dev.adapter_name, dev.max_buffer_size / (1024 * 1024))
    }) {
        Ok(info) => info,
        Err(_) => "not available".to_string(),
    }
}
