//! Backend selection and GPU runtime dispatch.

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::ExtensionOf;
use stwo::prover::backend::{Col, Column, ColumnOps};
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;

/// Convert a `CircleEvaluation` from one backend to another.
///
/// Extracts column data to CPU via `to_cpu()`, then reconstructs
/// the column for the destination backend via `FromIterator`.
///
/// When `Src == Dst` (e.g., both `SimdBackend`), the compiler will
/// optimize this to a simple clone. For `SimdBackend → GpuBackend`,
/// this uploads the trace data to the GPU.
pub fn convert_evaluation<Src, Dst, F>(
    eval: CircleEvaluation<Src, F, BitReversedOrder>,
) -> CircleEvaluation<Dst, F, BitReversedOrder>
where
    Src: ColumnOps<F>,
    Dst: ColumnOps<F>,
    F: ExtensionOf<BaseField> + Clone,
    Col<Src, F>: Column<F>,
    Col<Dst, F>: Column<F>,
{
    let cpu_vals = eval.values.to_cpu();
    let new_col: Col<Dst, F> = cpu_vals.into_iter().collect();
    CircleEvaluation::new(eval.domain, new_col)
}

/// Convert a batch of `CircleEvaluation`s from one backend to another.
pub fn convert_evaluations<Src, Dst, F>(
    evals: Vec<CircleEvaluation<Src, F, BitReversedOrder>>,
) -> Vec<CircleEvaluation<Dst, F, BitReversedOrder>>
where
    Src: ColumnOps<F>,
    Dst: ColumnOps<F>,
    F: ExtensionOf<BaseField> + Clone,
    Col<Src, F>: Column<F>,
    Col<Dst, F>: Column<F>,
{
    evals.into_iter().map(convert_evaluation::<Src, Dst, F>).collect()
}

/// Minimum problem sizes (log2) where GPU acceleration is beneficial.
///
/// Override with env var `OBELYSK_GPU_THRESHOLD` to set all thresholds to the same value.
/// Set to `0` on H200/A100 to always use GPU. Default values are conservative for consumer GPUs.
pub struct GpuThresholds;

impl GpuThresholds {
    const DEFAULT_FFT_FRI: u32 = 12;
    const DEFAULT_QUOTIENT: u32 = 14;
    const DEFAULT_COLUMN_OPS: u32 = 14;
    const DEFAULT_MERKLE: u32 = 14;
    const DEFAULT_MLE: u32 = 14;

    /// Read override from `OBELYSK_GPU_THRESHOLD` env var, or return the default.
    fn env_override(default: u32) -> u32 {
        std::env::var("OBELYSK_GPU_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    pub fn fft_fri() -> u32 { Self::env_override(Self::DEFAULT_FFT_FRI) }
    pub fn quotient() -> u32 { Self::env_override(Self::DEFAULT_QUOTIENT) }
    pub fn column_ops() -> u32 { Self::env_override(Self::DEFAULT_COLUMN_OPS) }
    pub fn merkle() -> u32 { Self::env_override(Self::DEFAULT_MERKLE) }
    pub fn mle() -> u32 { Self::env_override(Self::DEFAULT_MLE) }

    // Keep const values for backwards compatibility in tests
    pub const FFT_FRI: u32 = 12;
    pub const QUOTIENT: u32 = 14;
    pub const COLUMN_OPS: u32 = 14;
    pub const MERKLE: u32 = 14;
    pub const MLE: u32 = 14;
}

/// Returns `true` if a GPU backend is available at runtime.
pub fn gpu_is_available() -> bool {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return GpuBackend::is_available();
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        false
    }
}

/// Returns the GPU device name, if available.
pub fn gpu_device_name() -> Option<String> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return GpuBackend::device_name();
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        None
    }
}

/// Returns available GPU memory in bytes, if available.
pub fn gpu_available_memory() -> Option<usize> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return GpuBackend::available_memory();
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        None
    }
}

/// Returns GPU compute capability (major, minor), if available.
pub fn gpu_compute_capability() -> Option<(u32, u32)> {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        return GpuBackend::compute_capability();
    }
    #[cfg(not(feature = "cuda-runtime"))]
    {
        None
    }
}

/// Information about the current proving backend.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: &'static str,
    pub gpu_available: bool,
    pub gpu_device: Option<String>,
    pub gpu_memory_bytes: Option<usize>,
    pub gpu_compute_capability: Option<(u32, u32)>,
    /// Number of GPU devices available (multi-GPU feature).
    pub gpu_count: usize,
    /// Per-device information (multi-GPU feature).
    #[cfg(feature = "multi-gpu")]
    pub gpu_devices: Vec<crate::multi_gpu::GpuDeviceInfo>,
}

impl BackendInfo {
    pub fn detect() -> Self {
        #[cfg(feature = "multi-gpu")]
        let devices = crate::multi_gpu::discover_devices();
        #[cfg(not(feature = "multi-gpu"))]
        let device_count = if gpu_is_available() { 1 } else { 0 };

        Self {
            name: if gpu_is_available() { "GpuBackend" } else { "SimdBackend" },
            gpu_available: gpu_is_available(),
            gpu_device: gpu_device_name(),
            gpu_memory_bytes: gpu_available_memory(),
            gpu_compute_capability: gpu_compute_capability(),
            #[cfg(feature = "multi-gpu")]
            gpu_count: devices.len(),
            #[cfg(not(feature = "multi-gpu"))]
            gpu_count: device_count,
            #[cfg(feature = "multi-gpu")]
            gpu_devices: devices,
        }
    }
}

/// Estimate GPU memory required for proving a trace of the given log_size.
///
/// Each column ≈ `2^log_size × 4` bytes (M31 values). FRI blowup doubles the
/// domain, Merkle trees add ~2× overhead, and quotient accumulation needs scratch.
pub fn estimate_proof_memory(log_size: u32, num_columns: usize) -> usize {
    let row_bytes = 4; // M31 = 4 bytes
    let rows = 1usize << log_size;
    let blowup = 2; // default blowup factor
    // columns × rows × blowup × (data + Merkle + scratch)
    num_columns * rows * row_bytes * blowup * 3
}

/// Whether the given problem size fits in GPU memory with a safety margin.
pub fn fits_in_gpu(log_size: u32, num_columns: usize) -> bool {
    let needed = estimate_proof_memory(log_size, num_columns);
    match gpu_available_memory() {
        Some(available) => needed < (available * 4 / 5), // 80% safety margin
        None => false,
    }
}

/// Whether the given log_size is large enough for GPU to be beneficial.
///
/// Set `OBELYSK_GPU_THRESHOLD=0` to always use GPU on datacenter GPUs (H200, A100).
pub fn should_use_gpu(log_size: u32) -> bool {
    gpu_is_available() && log_size >= GpuThresholds::fft_fri()
}

/// Execute a closure with the best available backend.
///
/// Calls `f_gpu` if CUDA runtime is available and the GPU is detected,
/// otherwise falls back to `f_simd`.
///
/// # Example
/// ```ignore
/// use stwo_ml::backend::with_best_backend;
///
/// let result = with_best_backend(
///     || prove_model_simd(graph, input, weights),
///     || prove_model_gpu(graph, input, weights),
/// );
/// ```
pub fn with_best_backend<R>(
    f_simd: impl FnOnce() -> R,
    #[allow(unused_variables)]
    f_gpu: impl FnOnce() -> R,
) -> R {
    #[cfg(feature = "cuda-runtime")]
    {
        if gpu_is_available() {
            return f_gpu();
        }
    }
    f_simd()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_info() {
        let info = BackendInfo::detect();
        assert!(info.name == "SimdBackend" || info.name == "GpuBackend");
    }

    #[test]
    fn test_thresholds() {
        assert_eq!(GpuThresholds::FFT_FRI, 12);
        assert_eq!(GpuThresholds::QUOTIENT, 14);
    }

    #[test]
    fn test_gpu_detection() {
        let _available = gpu_is_available();
        let _name = gpu_device_name();
        let _mem = gpu_available_memory();
        let _cc = gpu_compute_capability();
    }

    #[test]
    fn test_estimate_proof_memory() {
        // log_size=16, 10 columns → should be non-zero and reasonable
        let mem = estimate_proof_memory(16, 10);
        assert!(mem > 0);
        // 10 cols × 2^16 rows × 4 bytes × 2 blowup × 3 overhead = 10 × 65536 × 24 = ~15MB
        assert!(mem < 100_000_000, "estimate should be < 100MB for log_size=16");
    }

    #[test]
    fn test_fits_in_gpu() {
        // Without CUDA, should always return false
        if !gpu_is_available() {
            assert!(!fits_in_gpu(16, 10));
        }
    }

    #[test]
    fn test_convert_evaluation_simd_to_simd() {
        use stwo::core::fields::m31::M31;
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::prover::backend::simd::SimdBackend;

        // Build a small evaluation on SimdBackend
        let log_size = 4;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let mut col = Col::<SimdBackend, BaseField>::zeros(1 << log_size);
        for i in 0..(1 << log_size) {
            col.set(i, M31::from(i as u32));
        }
        let eval = CircleEvaluation::new(domain, col);

        // Convert SimdBackend → SimdBackend (identity)
        let converted = convert_evaluation::<SimdBackend, SimdBackend, BaseField>(eval);
        assert_eq!(converted.values.to_cpu().len(), 1 << log_size);
        assert_eq!(converted.values.at(0), M31::from(0));
        assert_eq!(converted.values.at(5), M31::from(5));
    }

    #[test]
    fn test_convert_evaluations_batch() {
        use stwo::core::fields::m31::M31;
        use stwo::core::poly::circle::CanonicCoset;
        use stwo::prover::backend::simd::SimdBackend;

        let log_size = 4;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        let mut evals = Vec::new();
        for col_idx in 0..3u32 {
            let mut col = Col::<SimdBackend, BaseField>::zeros(size);
            for i in 0..size {
                col.set(i, M31::from(col_idx * 100 + i as u32));
            }
            evals.push(CircleEvaluation::new(domain, col));
        }

        let converted = convert_evaluations::<SimdBackend, SimdBackend, BaseField>(evals);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].values.at(0), M31::from(0));
        assert_eq!(converted[1].values.at(0), M31::from(100));
        assert_eq!(converted[2].values.at(0), M31::from(200));
    }

    #[test]
    fn test_with_best_backend() {
        // Without cuda-runtime, always uses simd path
        let result = with_best_backend(
            || "simd",
            || "gpu",
        );
        // In test environment (no CUDA), should always pick simd
        assert_eq!(result, "simd");
    }
}
