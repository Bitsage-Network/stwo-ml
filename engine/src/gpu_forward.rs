//! GPU Forward Pass Executor — runs forward pass operations on GPU.
//!
//! Keeps intermediate tensors on GPU memory between operations,
//! eliminating CPU↔GPU roundtrips for matrix operations.

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::m31::M31;

#[cfg(feature = "cuda-runtime")]
use crate::components::matmul::M31Matrix;

#[cfg(feature = "cuda-runtime")]
use crate::gpu_sumcheck::{ForwardKernels, GpuSumcheckExecutor};

/// GPU forward pass error type.
#[derive(Debug, thiserror::Error)]
pub enum GpuForwardError {
    #[error("GPU kernel error: {0}")]
    KernelError(String),
    #[error("GPU memory error: {0}")]
    MemoryError(String),
    #[error("GPU init error: {0}")]
    InitError(String),
}

/// GPU-resident tensor — stays on device between operations.
#[cfg(feature = "cuda-runtime")]
pub struct GpuTensor {
    pub data: CudaSlice<u32>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "cuda-runtime")]
impl GpuTensor {
    pub fn len(&self) -> usize { self.rows * self.cols }
}

/// Orchestrates the forward pass on GPU with minimal CPU roundtrips.
#[cfg(feature = "cuda-runtime")]
pub struct GpuForwardExecutor {
    device: Arc<CudaDevice>,
    kernels: ForwardKernels,
}

#[cfg(feature = "cuda-runtime")]
impl GpuForwardExecutor {
    pub fn new() -> Result<Self, GpuForwardError> {
        let executor = GpuSumcheckExecutor::cached()
            .map_err(|e| GpuForwardError::InitError(format!("{e}")))?;
        let kernels = executor.get_forward_fns()
            .map_err(|e| GpuForwardError::InitError(format!("{e}")))?;
        Ok(Self { device: executor.device.clone(), kernels })
    }

    pub fn upload(&self, m: &M31Matrix) -> Result<GpuTensor, GpuForwardError> {
        let data_u32: Vec<u32> = m.data.iter().map(|v| v.0).collect();
        let d = self.device.htod_sync_copy(&data_u32)
            .map_err(|e| GpuForwardError::MemoryError(format!("upload: {e:?}")))?;
        Ok(GpuTensor { data: d, rows: m.rows, cols: m.cols })
    }

    pub fn download(&self, t: &GpuTensor) -> Result<M31Matrix, GpuForwardError> {
        let mut buf = vec![0u32; t.len()];
        self.device.dtoh_sync_copy_into(&t.data, &mut buf)
            .map_err(|e| GpuForwardError::MemoryError(format!("download: {e:?}")))?;
        Ok(M31Matrix { rows: t.rows, cols: t.cols, data: buf.into_iter().map(M31::from).collect() })
    }

    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuForwardError> {
        let (m, k, n) = (a.rows, a.cols, b.cols);
        if k != b.rows {
            return Err(GpuForwardError::KernelError(format!("matmul: a.cols={k} != b.rows={}", b.rows)));
        }
        let d_out: CudaSlice<u32> = self.device.alloc_zeros(m * n)
            .map_err(|e| GpuForwardError::MemoryError(format!("{e:?}")))?;

        if m == 1 {
            let bs = 256u32;
            let gs = (n as u32 + bs - 1) / bs;
            unsafe {
                self.kernels.gemv_fn.clone().launch(
                    LaunchConfig { grid_dim: (gs,1,1), block_dim: (bs,1,1), shared_mem_bytes: 0 },
                    (&a.data, &b.data, &d_out, k as u32, n as u32),
                ).map_err(|e| GpuForwardError::KernelError(format!("gemv: {e:?}")))?;
            }
        } else {
            let (bx, by) = (16u32, 16u32);
            unsafe {
                self.kernels.gemm_fn.clone().launch(
                    LaunchConfig { grid_dim: ((n as u32+bx-1)/bx, (m as u32+by-1)/by, 1), block_dim: (bx,by,1), shared_mem_bytes: 0 },
                    (&a.data, &b.data, &d_out, m as u32, k as u32, n as u32),
                ).map_err(|e| GpuForwardError::KernelError(format!("gemm: {e:?}")))?;
            }
        }
        Ok(GpuTensor { data: d_out, rows: m, cols: n })
    }

    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuForwardError> {
        let n = a.len();
        let d_out: CudaSlice<u32> = self.device.alloc_zeros(n)
            .map_err(|e| GpuForwardError::MemoryError(format!("{e:?}")))?;
        let (bs, gs) = (256u32, (n as u32 + 255) / 256);
        unsafe {
            self.kernels.add_fn.clone().launch(
                LaunchConfig { grid_dim: (gs,1,1), block_dim: (bs,1,1), shared_mem_bytes: 0 },
                (&a.data, &b.data, &d_out, n as u32),
            ).map_err(|e| GpuForwardError::KernelError(format!("add: {e:?}")))?;
        }
        Ok(GpuTensor { data: d_out, rows: a.rows, cols: a.cols })
    }

    pub fn mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, GpuForwardError> {
        let n = a.len();
        let d_out: CudaSlice<u32> = self.device.alloc_zeros(n)
            .map_err(|e| GpuForwardError::MemoryError(format!("{e:?}")))?;
        let (bs, gs) = (256u32, (n as u32 + 255) / 256);
        unsafe {
            self.kernels.mul_fn.clone().launch(
                LaunchConfig { grid_dim: (gs,1,1), block_dim: (bs,1,1), shared_mem_bytes: 0 },
                (&a.data, &b.data, &d_out, n as u32),
            ).map_err(|e| GpuForwardError::KernelError(format!("mul: {e:?}")))?;
        }
        Ok(GpuTensor { data: d_out, rows: a.rows, cols: a.cols })
    }

    pub fn relu(&self, a: &GpuTensor) -> Result<GpuTensor, GpuForwardError> {
        let n = a.len();
        let d_out: CudaSlice<u32> = self.device.alloc_zeros(n)
            .map_err(|e| GpuForwardError::MemoryError(format!("{e:?}")))?;
        let (bs, gs) = (256u32, (n as u32 + 255) / 256);
        unsafe {
            self.kernels.relu_fn.clone().launch(
                LaunchConfig { grid_dim: (gs,1,1), block_dim: (bs,1,1), shared_mem_bytes: 0 },
                (&a.data, &d_out, n as u32),
            ).map_err(|e| GpuForwardError::KernelError(format!("relu: {e:?}")))?;
        }
        Ok(GpuTensor { data: d_out, rows: a.rows, cols: a.cols })
    }

    /// Pre-upload all weight matrices to GPU.
    pub fn upload_weights(
        &self,
        weights: &crate::compiler::graph::GraphWeights,
    ) -> Result<std::collections::HashMap<usize, GpuTensor>, GpuForwardError> {
        let mut gpu_weights = std::collections::HashMap::new();
        for (node_id, matrix) in &weights.weights {
            let tensor = self.upload(matrix)?;
            gpu_weights.insert(*node_id, tensor);
        }
        eprintln!("[gpu-forward] Pre-uploaded {} weight matrices to GPU", gpu_weights.len());
        Ok(gpu_weights)
    }
}

// Stubs for non-CUDA builds
#[cfg(not(feature = "cuda-runtime"))]
pub struct GpuTensor;

#[cfg(not(feature = "cuda-runtime"))]
pub struct GpuForwardExecutor;
