//! Backend selection and GPU runtime dispatch.

/// Minimum problem sizes (log2) where GPU acceleration is beneficial.
pub struct GpuThresholds;

impl GpuThresholds {
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

/// Information about the current proving backend.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: &'static str,
    pub gpu_available: bool,
    pub gpu_device: Option<String>,
    pub gpu_memory_bytes: Option<usize>,
}

impl BackendInfo {
    pub fn detect() -> Self {
        Self {
            name: if gpu_is_available() { "GpuBackend" } else { "SimdBackend" },
            gpu_available: gpu_is_available(),
            gpu_device: gpu_device_name(),
            gpu_memory_bytes: gpu_available_memory(),
        }
    }
}

/// Whether the given log_size is large enough for GPU to be beneficial.
pub fn should_use_gpu(log_size: u32) -> bool {
    gpu_is_available() && log_size >= GpuThresholds::FFT_FRI
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
    }
}
