//! GPU Forward Pass Executor — runs the entire forward pass on GPU.
//!
//! Keeps intermediate tensors on GPU memory between operations,
//! eliminating CPU↔GPU roundtrips. Downloads intermediates only
//! when needed for GKR proving.
//!
//! # Speedup
//!
//! By keeping data on-device between matmul → norm → activation → matmul,
//! the forward pass time drops from ~38s (CPU) to ~5-8s (GPU) for Qwen3-14B.

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchConfig};

#[cfg(feature = "cuda-runtime")]
use stwo::core::fields::m31::M31;

#[cfg(feature = "cuda-runtime")]
use crate::components::matmul::M31Matrix;

#[cfg(feature = "cuda-runtime")]
use crate::gpu_sumcheck::{CudaFftError, ForwardKernels, GpuSumcheckExecutor};

/// GPU-resident tensor — stays on device between operations.
#[cfg(feature = "cuda-runtime")]
pub struct GpuTensor {
    pub data: CudaSlice<u32>,
    pub rows: usize,
    pub cols: usize,
}

#[cfg(feature = "cuda-runtime")]
impl GpuTensor {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }
}

/// Orchestrates the forward pass on GPU with minimal CPU roundtrips.
#[cfg(feature = "cuda-runtime")]
pub struct GpuForwardExecutor {
    device: Arc<CudaDevice>,
    kernels: ForwardKernels,
}

#[cfg(feature = "cuda-runtime")]
impl GpuForwardExecutor {
    /// Create a new executor from the cached GPU sumcheck executor.
    pub fn new() -> Result<Self, CudaFftError> {
        let executor = GpuSumcheckExecutor::cached()?;
        let kernels = executor.get_forward_fns()?;
        Ok(Self {
            device: executor.device.clone(),
            kernels,
        })
    }

    /// Upload an M31Matrix to GPU.
    pub fn upload(&self, m: &M31Matrix) -> Result<GpuTensor, CudaFftError> {
        let data_u32: Vec<u32> = m.data.iter().map(|v| v.0).collect();
        let d_data = self.device.htod_sync_copy(&data_u32)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("upload: {:?}", e)))?;
        Ok(GpuTensor { data: d_data, rows: m.rows, cols: m.cols })
    }

    /// Download a GPU tensor to M31Matrix.
    pub fn download(&self, t: &GpuTensor) -> Result<M31Matrix, CudaFftError> {
        let n = t.rows * t.cols;
        let mut buf = vec![0u32; n];
        self.device.dtoh_sync_copy_into(&t.data, &mut buf)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("download: {:?}", e)))?;
        let data: Vec<M31> = buf.iter().map(|&v| M31::from(v)).collect();
        Ok(M31Matrix { rows: t.rows, cols: t.cols, data })
    }

    /// GPU matrix multiply: C = A × B. Both tensors stay on device.
    pub fn matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaFftError> {
        let m = a.rows;
        let k = a.cols;
        let n = b.cols;

        if k != b.rows {
            return Err(CudaFftError::KernelExecution(format!(
                "matmul dim mismatch: a.cols={k} != b.rows={}", b.rows
            )));
        }

        let d_output: CudaSlice<u32> = self.device.alloc_zeros(m * n)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        if m == 1 {
            let block_size = 256u32;
            let grid_size = (n as u32 + block_size - 1) / block_size;
            unsafe {
                self.kernels.gemv_fn.clone().launch(
                    LaunchConfig { grid_dim: (grid_size, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 },
                    (&a.data, &b.data, &d_output, k as u32, n as u32),
                ).map_err(|e| CudaFftError::KernelExecution(format!("gemv: {:?}", e)))?;
            }
        } else {
            let bx = 16u32;
            let by = 16u32;
            unsafe {
                self.kernels.gemm_fn.clone().launch(
                    LaunchConfig { grid_dim: ((n as u32 + bx - 1) / bx, (m as u32 + by - 1) / by, 1), block_dim: (bx, by, 1), shared_mem_bytes: 0 },
                    (&a.data, &b.data, &d_output, m as u32, k as u32, n as u32),
                ).map_err(|e| CudaFftError::KernelExecution(format!("gemm: {:?}", e)))?;
            }
        }

        Ok(GpuTensor { data: d_output, rows: m, cols: n })
    }

    /// GPU element-wise addition: C = A + B.
    pub fn add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaFftError> {
        let n = a.len();
        let d_output: CudaSlice<u32> = self.device.alloc_zeros(n)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block = 256u32;
        let grid = (n as u32 + block - 1) / block;
        unsafe {
            self.kernels.add_fn.clone().launch(
                LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 },
                (&a.data, &b.data, &d_output, n as u32),
            ).map_err(|e| CudaFftError::KernelExecution(format!("add: {:?}", e)))?;
        }

        Ok(GpuTensor { data: d_output, rows: a.rows, cols: a.cols })
    }

    /// GPU element-wise multiplication: C = A * B.
    pub fn mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaFftError> {
        let n = a.len();
        let d_output: CudaSlice<u32> = self.device.alloc_zeros(n)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block = 256u32;
        let grid = (n as u32 + block - 1) / block;
        unsafe {
            self.kernels.mul_fn.clone().launch(
                LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 },
                (&a.data, &b.data, &d_output, n as u32),
            ).map_err(|e| CudaFftError::KernelExecution(format!("mul: {:?}", e)))?;
        }

        Ok(GpuTensor { data: d_output, rows: a.rows, cols: a.cols })
    }

    /// GPU ReLU activation.
    pub fn relu(&self, a: &GpuTensor) -> Result<GpuTensor, CudaFftError> {
        let n = a.len();
        let d_output: CudaSlice<u32> = self.device.alloc_zeros(n)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        let block = 256u32;
        let grid = (n as u32 + block - 1) / block;
        unsafe {
            self.kernels.relu_fn.clone().launch(
                LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 },
                (&a.data, &d_output, n as u32),
            ).map_err(|e| CudaFftError::KernelExecution(format!("relu: {:?}", e)))?;
        }

        Ok(GpuTensor { data: d_output, rows: a.rows, cols: a.cols })
    }

    /// GPU RMSNorm. Returns (normalized_output, rms_sq_values, rsqrt_values).
    ///
    /// The rms_sq and rsqrt outputs are needed for the unified STARK proof.
    pub fn rmsnorm(
        &self,
        a: &GpuTensor,
        dim: usize,
        rsqrt_table: &CudaSlice<u32>,
    ) -> Result<(GpuTensor, Vec<u32>, Vec<u32>), CudaFftError> {
        let rows = a.rows;
        let d_output: CudaSlice<u32> = self.device.alloc_zeros(a.len())
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let d_rms_sq: CudaSlice<u32> = self.device.alloc_zeros(rows)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;
        let d_rsqrt: CudaSlice<u32> = self.device.alloc_zeros(rows)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        // inv_n = modular inverse of dim in M31 (precomputed)
        let p = 0x7FFFFFFFu64;
        let inv_n = mod_pow(dim as u64, p - 2, p) as u32;

        unsafe {
            self.kernels.rmsnorm_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 256 * 4,
                },
                (&a.data, &d_output, &d_rms_sq, &d_rsqrt, rsqrt_table, dim as u32, inv_n),
            ).map_err(|e| CudaFftError::KernelExecution(format!("rmsnorm: {:?}", e)))?;
        }

        let mut rms_sq = vec![0u32; rows];
        let mut rsqrt = vec![0u32; rows];
        self.device.dtoh_sync_copy_into(&d_rms_sq, &mut rms_sq)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;
        self.device.dtoh_sync_copy_into(&d_rsqrt, &mut rsqrt)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("{:?}", e)))?;

        Ok((GpuTensor { data: d_output, rows, cols: dim }, rms_sq, rsqrt))
    }

    /// Pre-upload all weight matrices to GPU, returning a map of node_id → GpuTensor.
    pub fn upload_weights(
        &self,
        weights: &crate::compiler::graph::GraphWeights,
    ) -> Result<std::collections::HashMap<usize, GpuTensor>, CudaFftError> {
        let mut gpu_weights = std::collections::HashMap::new();
        for (node_id, matrix) in &weights.weights {
            let tensor = self.upload(matrix)?;
            gpu_weights.insert(*node_id, tensor);
        }
        eprintln!("[gpu-forward] Pre-uploaded {} weight matrices to GPU", gpu_weights.len());
        Ok(gpu_weights)
    }
}

/// Modular exponentiation for M31 inverse computation.
#[cfg(feature = "cuda-runtime")]
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % modulus;
        }
        exp /= 2;
        base = base * base % modulus;
    }
    result
}

// ═══════════════════════════════════════════════════════════════════
// Stub for non-CUDA builds
// ═══════════════════════════════════════════════════════════════════

#[cfg(not(feature = "cuda-runtime"))]
pub struct GpuTensor;

#[cfg(not(feature = "cuda-runtime"))]
pub struct GpuForwardExecutor;
