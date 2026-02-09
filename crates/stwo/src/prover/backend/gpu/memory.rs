//! GPU Memory Management for Stwo.
//!
//! This module provides type-safe GPU memory allocation and transfer operations,
//! plus a thread-local FRI GPU residency cache that keeps intermediate fold
//! results on the GPU between consecutive FRI fold rounds.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐        ┌─────────────────┐
//! │   CPU Memory    │ ←───→  │   GPU Memory    │
//! │  (BaseColumn)   │  H2D   │  (GpuBuffer)    │
//! │                 │  D2H   │                 │
//! └─────────────────┘        └─────────────────┘
//! ```
//!
//! # FRI GPU Residency
//!
//! The `FRI_GPU_CACHE` thread-local stores the GPU-resident output of the
//! previous fold round, keyed by the length of the resulting
//! `SecureColumnByCoords`. Each cached slice is consumed exactly once by the
//! next fold (take semantics). Length-only keying works because the cache
//! holds at most one entry and each FRI round halves the evaluation size.
//!
//! # Design Principles
//!
//! 1. **Explicit transfers**: No implicit data movement
//! 2. **Type safety**: GpuBuffer<T> tracks element type
//! 3. **RAII**: GPU memory freed on drop
//! 4. **Error handling**: All operations return Result

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

use super::cuda_executor::CudaFftError;

#[cfg(feature = "cuda-runtime")]
use super::cuda_executor::get_cuda_executor;

#[cfg(feature = "cuda-runtime")]
use std::cell::RefCell;

// =============================================================================
// FRI GPU Residency Cache
// =============================================================================

/// State for a GPU-resident FRI fold output.
///
/// After each fold round, the output `CudaSlice<u32>` (AoS layout) is cached
/// here so the next fold can consume it without a fresh H2D transfer.
#[cfg(feature = "cuda-runtime")]
pub struct FriGpuState {
    /// AoS-layout data on the GPU (4 × u32 per QM31 element).
    pub d_data: CudaSlice<u32>,
    /// Number of QM31 elements this slice represents.
    /// Length alone is sufficient as a cache key because the FRI cache holds
    /// exactly one entry that is consumed once per fold round.
    pub len: usize,
}

#[cfg(feature = "cuda-runtime")]
thread_local! {
    /// Thread-local FRI GPU cache. Each entry is consumed exactly once.
    static FRI_GPU_CACHE: RefCell<Option<FriGpuState>> = RefCell::new(None);
}

/// Store a GPU-resident fold output for the next round.
///
/// `col` is the CPU-side `SecureColumnByCoords` that was just written.
/// `d_data` is the corresponding AoS data still on the GPU.
#[cfg(feature = "cuda-runtime")]
pub fn cache_fri_gpu_data<B: crate::prover::backend::ColumnOps<crate::core::fields::m31::BaseField>>(
    col: &crate::prover::secure_column::SecureColumnByCoords<B>,
    d_data: CudaSlice<u32>,
) {
    let len = col.len();
    FRI_GPU_CACHE.with(|cache| {
        *cache.borrow_mut() = Some(FriGpuState { d_data, len });
    });
}

/// Try to take a cached GPU slice for the given column.
///
/// Returns `Some(CudaSlice)` if the cache holds data matching the column's
/// length, consuming the cache entry. Length is sufficient because the cache
/// holds exactly one entry that is consumed once per FRI fold round.
/// Returns `None` if there is no match (first fold or different size).
#[cfg(feature = "cuda-runtime")]
pub fn take_cached_fri_gpu_data<B: crate::prover::backend::ColumnOps<crate::core::fields::m31::BaseField>>(
    col: &crate::prover::secure_column::SecureColumnByCoords<B>,
) -> Option<CudaSlice<u32>> {
    let len = col.len();
    FRI_GPU_CACHE.with(|cache| {
        let mut guard = cache.borrow_mut();
        if let Some(ref state) = *guard {
            if state.len == len {
                return guard.take().map(|s| s.d_data);
            }
        }
        None
    })
}

/// Pre-populate the FRI GPU cache with AoS data already on the device.
///
/// Call this before `FriProver::commit()` to skip the SoA→AoS conversion
/// and H2D transfer in the first `fold_circle_into_line` call. The `col`
/// must have the same length as the column that will be passed to the
/// FRI prover (length is used as the cache key).
#[cfg(feature = "cuda-runtime")]
pub fn prepopulate_fri_gpu_cache<B: crate::prover::backend::ColumnOps<crate::core::fields::m31::BaseField>>(
    col: &crate::prover::secure_column::SecureColumnByCoords<B>,
    d_data: CudaSlice<u32>,
) {
    cache_fri_gpu_data(col, d_data);
}

/// Clear the FRI GPU cache (e.g., at the end of a proof).
#[cfg(feature = "cuda-runtime")]
pub fn clear_fri_gpu_cache() {
    FRI_GPU_CACHE.with(|cache| {
        *cache.borrow_mut() = None;
    });
}

// =============================================================================
// Column GPU Residency Cache
// =============================================================================

// Thread-local cache for GPU-resident column data.
// After IFFT, the result `CudaSlice<u32>` is cached here so the subsequent
// FFT (evaluate) and Merkle commit can reuse it without a fresh H2D transfer.
// Keyed by (cpu_pointer, length) of the output `BaseColumn`.
#[cfg(feature = "cuda-runtime")]
thread_local! {
    static COLUMN_GPU_CACHE: RefCell<std::collections::HashMap<(usize, usize), CudaSlice<u32>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Store a GPU-resident column for later reuse by evaluate() or Merkle commit.
#[cfg(feature = "cuda-runtime")]
pub fn cache_column_gpu(col_ptr: usize, col_len: usize, d_data: CudaSlice<u32>) {
    COLUMN_GPU_CACHE.with(|cache| {
        cache.borrow_mut().insert((col_ptr, col_len), d_data);
    });
}

/// Take a cached GPU slice for the given column identity.
///
/// Returns `Some(CudaSlice)` if the cache holds data matching the key,
/// consuming the entry. Returns `None` on cache miss.
#[cfg(feature = "cuda-runtime")]
pub fn take_cached_column_gpu(col_ptr: usize, col_len: usize) -> Option<CudaSlice<u32>> {
    COLUMN_GPU_CACHE.with(|cache| {
        cache.borrow_mut().remove(&(col_ptr, col_len))
    })
}

/// Check if a column is cached on GPU without consuming it.
#[cfg(feature = "cuda-runtime")]
pub fn peek_cached_column_gpu(col_ptr: usize, col_len: usize) -> bool {
    COLUMN_GPU_CACHE.with(|cache| {
        cache.borrow().contains_key(&(col_ptr, col_len))
    })
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn peek_cached_column_gpu(_col_ptr: usize, _col_len: usize) -> bool {
    false
}

/// Clear the column GPU cache (e.g., at the end of a proof).
#[cfg(feature = "cuda-runtime")]
pub fn clear_column_gpu_cache() {
    COLUMN_GPU_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

// Stubs for non-CUDA builds
#[cfg(not(feature = "cuda-runtime"))]
pub fn cache_column_gpu(_col_ptr: usize, _col_len: usize, _d_data: ()) {}

#[cfg(not(feature = "cuda-runtime"))]
pub fn take_cached_column_gpu(_col_ptr: usize, _col_len: usize) -> Option<()> {
    None
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn clear_column_gpu_cache() {}

// Stubs for non-CUDA builds
#[cfg(not(feature = "cuda-runtime"))]
pub fn clear_fri_gpu_cache() {}

// =============================================================================
// Merkle Prev-Layer GPU Cache
// =============================================================================

// Thread-local cache for the GPU-resident Poseidon252 Merkle hash layer.
// After each `commit_on_layer` call, the output `CudaSlice<u64>` is cached
// here so the next layer can use it directly as `prev_layer` without
// CPU serialization + H2D re-upload. This eliminates the per-layer
// GPU→CPU→GPU round-trip in `MerkleProver::commit`.
#[cfg(feature = "cuda-runtime")]
thread_local! {
    static MERKLE_PREV_LAYER_GPU_CACHE: RefCell<Option<CudaSlice<u64>>> =
        RefCell::new(None);
}

/// Cache a GPU-resident Merkle hash layer for the next `commit_on_layer` call.
#[cfg(feature = "cuda-runtime")]
pub fn cache_merkle_prev_layer_gpu(d_layer: CudaSlice<u64>) {
    MERKLE_PREV_LAYER_GPU_CACHE.with(|cache| {
        *cache.borrow_mut() = Some(d_layer);
    });
}

/// Take the cached GPU-resident Merkle prev layer.
///
/// Returns `Some(CudaSlice<u64>)` if cached, consuming the entry.
#[cfg(feature = "cuda-runtime")]
pub fn take_merkle_prev_layer_gpu() -> Option<CudaSlice<u64>> {
    MERKLE_PREV_LAYER_GPU_CACHE.with(|cache| {
        cache.borrow_mut().take()
    })
}

/// Clear the Merkle prev-layer GPU cache.
#[cfg(feature = "cuda-runtime")]
pub fn clear_merkle_prev_layer_gpu_cache() {
    MERKLE_PREV_LAYER_GPU_CACHE.with(|cache| {
        *cache.borrow_mut() = None;
    });
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn clear_merkle_prev_layer_gpu_cache() {}

// =============================================================================
// Merkle Full-Tree Precomputed Layers Cache
// =============================================================================

// Thread-local cache for precomputed Merkle tree layers.
// When `commit_on_layer` is called for the leaf layer and all subsequent layers
// have no columns (the FRI case), we build the entire tree in one GPU pass
// with no per-layer sync or D2H. All layer results are stored here keyed by
// n_hashes (layer size). Subsequent `commit_on_layer` calls just pop results.
#[cfg(feature = "cuda-runtime")]
thread_local! {
    static MERKLE_PRECOMPUTED_LAYERS: RefCell<std::collections::HashMap<usize, Vec<u64>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Cache a precomputed Merkle layer result (keyed by n_hashes).
#[cfg(feature = "cuda-runtime")]
pub fn cache_precomputed_merkle_layer(n_hashes: usize, data: Vec<u64>) {
    MERKLE_PRECOMPUTED_LAYERS.with(|cache| {
        cache.borrow_mut().insert(n_hashes, data);
    });
}

/// Take a precomputed Merkle layer result for the given n_hashes.
#[cfg(feature = "cuda-runtime")]
pub fn take_precomputed_merkle_layer(n_hashes: usize) -> Option<Vec<u64>> {
    MERKLE_PRECOMPUTED_LAYERS.with(|cache| {
        cache.borrow_mut().remove(&n_hashes)
    })
}

/// Clear all precomputed Merkle layers.
#[cfg(feature = "cuda-runtime")]
pub fn clear_precomputed_merkle_layers() {
    MERKLE_PRECOMPUTED_LAYERS.with(|cache| {
        cache.borrow_mut().clear();
    });
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn clear_precomputed_merkle_layers() {}

// =============================================================================
// Blake2s Precomputed Merkle Layers Cache
// =============================================================================

// Thread-local cache for precomputed Blake2s Merkle tree layers.
// When `commit_on_layer` is called for the leaf layer in the FRI case,
// we build the entire tree in one GPU pass with no per-layer sync or D2H.
// All layer results are stored here keyed by n_hashes (layer size).
// Subsequent `commit_on_layer` calls just pop results.
#[cfg(feature = "cuda-runtime")]
thread_local! {
    static BLAKE2S_PRECOMPUTED_LAYERS: RefCell<std::collections::HashMap<usize, Vec<crate::core::vcs::blake2_hash::Blake2sHash>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Cache a precomputed Blake2s Merkle layer result (keyed by n_hashes).
#[cfg(feature = "cuda-runtime")]
pub fn cache_precomputed_blake2s_layer(n_hashes: usize, data: Vec<crate::core::vcs::blake2_hash::Blake2sHash>) {
    BLAKE2S_PRECOMPUTED_LAYERS.with(|cache| {
        cache.borrow_mut().insert(n_hashes, data);
    });
}

/// Take a precomputed Blake2s Merkle layer result for the given n_hashes.
#[cfg(feature = "cuda-runtime")]
pub fn take_precomputed_blake2s_layer(n_hashes: usize) -> Option<Vec<crate::core::vcs::blake2_hash::Blake2sHash>> {
    BLAKE2S_PRECOMPUTED_LAYERS.with(|cache| {
        cache.borrow_mut().remove(&n_hashes)
    })
}

/// Clear all precomputed Blake2s Merkle layers.
#[cfg(feature = "cuda-runtime")]
pub fn clear_precomputed_blake2s_layers() {
    BLAKE2S_PRECOMPUTED_LAYERS.with(|cache| {
        cache.borrow_mut().clear();
    });
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn clear_precomputed_blake2s_layers() {}

// =============================================================================
// FRI Deferred D2H Cache
// =============================================================================

/// Deferred download entry: AoS GPU data that will be downloaded lazily
/// when the CPU-side `SecureColumnByCoords` is actually needed (during decommit).
#[cfg(feature = "cuda-runtime")]
pub struct DeferredDownload {
    /// AoS-layout data on GPU (4 × u32 per QM31 element).
    pub d_aos: CudaSlice<u32>,
    /// Number of QM31 elements.
    pub n_elements: usize,
}

#[cfg(feature = "cuda-runtime")]
thread_local! {
    /// Maps column pointer (address of `SecureColumnByCoords.columns`) to
    /// its GPU-resident AoS data for deferred D2H download.
    /// Populated by `fold_line_cuda` when using the GPU-only path;
    /// consumed by decommit when CPU data is actually needed.
    static DEFERRED_D2H_CACHE: RefCell<std::collections::HashMap<usize, DeferredDownload>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Register a deferred D2H download for a `SecureColumnByCoords`.
///
/// `col_ptr` is the address of the column struct, `d_aos` is the AoS data on GPU,
/// and `n_elements` is the number of QM31 elements.
#[cfg(feature = "cuda-runtime")]
pub fn register_deferred_download(col_ptr: usize, d_aos: CudaSlice<u32>, n_elements: usize) {
    DEFERRED_D2H_CACHE.with(|cache| {
        cache.borrow_mut().insert(col_ptr, DeferredDownload { d_aos, n_elements });
    });
}

/// Execute a deferred D2H download if one exists for the given column pointer.
///
/// Returns the AoS `Vec<u32>` downloaded from GPU, consuming the cache entry.
/// Returns `None` if no deferred download is registered for this pointer.
#[cfg(feature = "cuda-runtime")]
pub fn execute_deferred_download(col_ptr: usize) -> Option<Vec<u32>> {
    DEFERRED_D2H_CACHE.with(|cache| {
        let entry = cache.borrow_mut().remove(&col_ptr)?;
        let mut cpu_data = vec![0u32; entry.n_elements * 4];
        // This is the actual D2H transfer, deferred until now
        if let Ok(executor) = super::cuda_executor::get_cuda_executor() {
            if executor.device.dtoh_sync_copy_into(&entry.d_aos, &mut cpu_data).is_ok() {
                tracing::debug!("Deferred D2H: downloaded {} QM31 elements", entry.n_elements);
                return Some(cpu_data);
            }
        }
        None
    })
}

/// Clear all deferred D2H entries.
#[cfg(feature = "cuda-runtime")]
pub fn clear_deferred_d2h_cache() {
    DEFERRED_D2H_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn clear_deferred_d2h_cache() {}

// =============================================================================
// GPU-Resident FRI Pipeline — Deferred Batch D2H
// =============================================================================
//
// During the FRI commit phase, fold outputs stay GPU-resident. Each fold round
// pushes a deferred entry (GPU AoS copy + n_output). After all fold rounds
// complete, `resolve_all_deferred_fri_folds()` batch-downloads everything to
// populate the CPU-side LineEvaluation values needed for decommitment.
//
// This eliminates per-round D2H transfers (the main GPU bottleneck) and replaces
// them with a single batch download at the end of the commit phase.

/// A single deferred fold output awaiting batch D2H.
#[cfg(feature = "cuda-runtime")]
pub struct DeferredFriFoldEntry {
    /// AoS-layout fold output on GPU (n_output * 4 u32s).
    pub d_aos: CudaSlice<u32>,
    /// Number of SecureField (QM31) elements in this fold output.
    pub n_output: usize,
}

#[cfg(feature = "cuda-runtime")]
thread_local! {
    /// Sequential list of deferred fold outputs, in fold-round order.
    /// Populated by fold_line_cuda / fold_circle_into_line_cuda.
    /// Consumed by resolve_next_deferred_fri_fold() in the same order.
    static DEFERRED_FRI_PIPELINE: RefCell<std::collections::VecDeque<DeferredFriFoldEntry>> =
        RefCell::new(std::collections::VecDeque::new());
}

/// Push a deferred fold output onto the pipeline.
#[cfg(feature = "cuda-runtime")]
pub fn push_deferred_fri_fold(d_aos: CudaSlice<u32>, n_output: usize) {
    DEFERRED_FRI_PIPELINE.with(|pipeline| {
        pipeline.borrow_mut().push_back(DeferredFriFoldEntry { d_aos, n_output });
    });
}

/// Replace the last deferred entry (used when fold_circle_into_line accumulates
/// into an evaluation that already has a deferred entry from fold_line).
#[cfg(feature = "cuda-runtime")]
pub fn replace_last_deferred_fri_fold(d_aos: CudaSlice<u32>, n_output: usize) {
    DEFERRED_FRI_PIPELINE.with(|pipeline| {
        let mut p = pipeline.borrow_mut();
        p.pop_back(); // drop previous entry for this evaluation
        p.push_back(DeferredFriFoldEntry { d_aos, n_output });
    });
}

/// Pop the next deferred fold entry (FIFO order).
#[cfg(feature = "cuda-runtime")]
pub fn pop_next_deferred_fri_fold() -> Option<DeferredFriFoldEntry> {
    DEFERRED_FRI_PIPELINE.with(|pipeline| {
        pipeline.borrow_mut().pop_front()
    })
}

/// Return number of pending deferred entries.
#[cfg(feature = "cuda-runtime")]
pub fn deferred_fri_pipeline_len() -> usize {
    DEFERRED_FRI_PIPELINE.with(|pipeline| {
        pipeline.borrow().len()
    })
}

/// Clear all deferred entries (for cleanup / error recovery).
#[cfg(feature = "cuda-runtime")]
pub fn clear_deferred_fri_pipeline() {
    DEFERRED_FRI_PIPELINE.with(|pipeline| {
        pipeline.borrow_mut().clear();
    });
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn clear_deferred_fri_pipeline() {}

// =============================================================================
// FRI SoA Column GPU Cache
// =============================================================================

// Cache for GPU-resident SoA column data from FRI fold outputs.
// After a fold kernel produces AoS output, we deinterleave into 4 SoA columns
// on GPU and cache them here. The Poseidon252 Merkle `commit_on_layer` checks
// this cache before uploading columns, eliminating the GPU→CPU→GPU round-trip.
#[cfg(feature = "cuda-runtime")]
thread_local! {
    static FRI_COLUMN_GPU_CACHE: RefCell<std::collections::HashMap<usize, CudaSlice<u32>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Cache a GPU-resident SoA column for the Poseidon252 Merkle path.
///
/// `col_ptr` is the CPU address of the `BaseColumn` data pointer.
#[cfg(feature = "cuda-runtime")]
pub fn cache_fri_column_gpu(col_ptr: usize, d_data: CudaSlice<u32>) {
    FRI_COLUMN_GPU_CACHE.with(|cache| {
        cache.borrow_mut().insert(col_ptr, d_data);
    });
}

/// Take a cached GPU-resident SoA column slice for the given CPU column pointer.
///
/// Returns `Some(CudaSlice)` if cached, consuming the entry.
#[cfg(feature = "cuda-runtime")]
pub fn take_fri_column_gpu(col_ptr: usize) -> Option<CudaSlice<u32>> {
    FRI_COLUMN_GPU_CACHE.with(|cache| {
        cache.borrow_mut().remove(&col_ptr)
    })
}

/// Clear the FRI column GPU cache.
#[cfg(feature = "cuda-runtime")]
pub fn clear_fri_column_gpu_cache() {
    FRI_COLUMN_GPU_CACHE.with(|cache| {
        cache.borrow_mut().clear();
    });
}

#[cfg(not(feature = "cuda-runtime"))]
pub fn clear_fri_column_gpu_cache() {}

// =============================================================================
// GPU Buffer - Type-safe GPU Memory Handle
// =============================================================================

/// A buffer of data residing in GPU memory.
///
/// `GpuBuffer<T>` owns GPU memory and provides safe access patterns.
/// The memory is automatically freed when the buffer is dropped.
///
/// # Type Parameters
///
/// - `T`: The element type. Must implement `DeviceRepr` for CUDA compatibility.
///
/// # Example
///
/// ```ignore
/// let cpu_data = vec![1u32, 2, 3, 4];
/// let gpu_buffer = GpuBuffer::from_cpu(&cpu_data)?;
/// // ... perform GPU operations ...
/// let result = gpu_buffer.to_cpu()?;
/// ```
#[cfg(feature = "cuda-runtime")]
pub struct GpuBuffer<T: DeviceRepr> {
    /// The underlying CUDA device memory
    slice: CudaSlice<T>,
    /// Reference to the CUDA device (for operations)
    device: Arc<CudaDevice>,
    /// Number of elements
    len: usize,
}

#[cfg(feature = "cuda-runtime")]
impl<T: DeviceRepr + Clone + Default> GpuBuffer<T> {
    /// Allocate uninitialized GPU memory for `len` elements.
    ///
    /// # Safety
    ///
    /// The returned buffer contains uninitialized memory. Reading from it
    /// before writing is undefined behavior.
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryAllocation` if allocation fails.
    pub fn alloc_uninit(len: usize) -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let device = executor.device.clone();

        // Allocate device memory
        // Safety: We're allocating uninitialized memory that will be filled by the caller
        let slice = unsafe {
            device.alloc::<T>(len)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        Ok(Self { slice, device, len })
    }

    /// Allocate GPU memory and initialize with zeros.
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryAllocation` if allocation fails.
    pub fn zeros(len: usize) -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let device = executor.device.clone();

        // Allocate and zero-initialize
        let zeros: Vec<T> = vec![T::default(); len];
        let slice = device.htod_sync_copy(&zeros)
            .map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        Ok(Self { slice, device, len })
    }

    /// Create a GPU buffer from CPU data (Host-to-Device transfer).
    ///
    /// This copies the data from CPU memory to GPU memory.
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryTransfer` if the transfer fails.
    pub fn from_cpu(data: &[T]) -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        let device = executor.device.clone();

        let slice = device.htod_sync_copy(data)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("H2D failed: {:?}", e)))?;

        Ok(Self {
            slice,
            device,
            len: data.len(),
        })
    }

    /// Copy GPU data back to CPU (Device-to-Host transfer).
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryTransfer` if the transfer fails.
    pub fn to_cpu(&self) -> Result<Vec<T>, CudaFftError> {
        let mut result = vec![T::default(); self.len];
        self.device.dtoh_sync_copy_into(&self.slice, &mut result)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("D2H failed: {:?}", e)))?;
        Ok(result)
    }

    /// Copy GPU data into an existing CPU buffer.
    ///
    /// # Panics
    ///
    /// Panics if `dst.len() != self.len()`.
    ///
    /// # Errors
    ///
    /// Returns `CudaFftError::MemoryTransfer` if the transfer fails.
    pub fn to_cpu_into(&self, dst: &mut [T]) -> Result<(), CudaFftError> {
        assert_eq!(dst.len(), self.len, "Destination buffer size mismatch");
        self.device.dtoh_sync_copy_into(&self.slice, dst)
            .map_err(|e| CudaFftError::MemoryTransfer(format!("D2H failed: {:?}", e)))?;
        Ok(())
    }

    /// Get the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the underlying CUDA slice for kernel operations.
    ///
    /// # Safety
    ///
    /// The returned slice is valid only as long as this `GpuBuffer` exists.
    pub fn as_slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// Get a mutable reference to the underlying CUDA slice.
    pub fn as_slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }

    /// Get the CUDA device handle.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

// GPU memory is automatically freed when GpuBuffer is dropped
// (CudaSlice handles this via cudarc)

// =============================================================================
// M31 Field GPU Operations
// =============================================================================

/// GPU buffer specifically for M31 field elements.
///
/// M31 elements are stored as `u32` on the GPU (the raw representation).
#[cfg(feature = "cuda-runtime")]
pub type GpuM31Buffer = GpuBuffer<u32>;

#[cfg(feature = "cuda-runtime")]
impl GpuM31Buffer {
    /// Create a GPU buffer from M31 field elements.
    pub fn from_m31(data: &[crate::core::fields::m31::M31]) -> Result<Self, CudaFftError> {
        // M31 is repr(transparent) over u32, so we can safely transmute
        let raw: &[u32] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u32,
                data.len()
            )
        };
        Self::from_cpu(raw)
    }

    /// Copy GPU data back as M31 field elements.
    pub fn to_m31(&self) -> Result<Vec<crate::core::fields::m31::M31>, CudaFftError> {
        let raw = self.to_cpu()?;
        // M31 is repr(transparent) over u32
        Ok(raw.into_iter().map(crate::core::fields::m31::M31).collect())
    }
}

// =============================================================================
// GPU Memory Pool (for reducing allocation overhead)
// =============================================================================

/// A pool of reusable GPU buffers to reduce allocation overhead.
///
/// Allocation on GPU is expensive (~100μs). For repeated operations,
/// reusing buffers from a pool can significantly improve performance.
#[cfg(feature = "cuda-runtime")]
pub struct GpuBufferPool {
    /// Available buffers, keyed by size
    available: std::collections::HashMap<usize, Vec<CudaSlice<u32>>>,
    /// The CUDA device
    device: Arc<CudaDevice>,
    /// Total allocated bytes (for monitoring)
    total_allocated: usize,
}

#[cfg(feature = "cuda-runtime")]
impl GpuBufferPool {
    /// Create a new buffer pool.
    pub fn new() -> Result<Self, CudaFftError> {
        let executor = get_cuda_executor().map_err(|e| e.clone())?;
        Ok(Self {
            available: std::collections::HashMap::new(),
            device: executor.device.clone(),
            total_allocated: 0,
        })
    }

    /// Get a buffer of at least `min_len` elements.
    ///
    /// This will reuse an existing buffer if one is available,
    /// otherwise it allocates a new one.
    pub fn get(&mut self, min_len: usize) -> Result<CudaSlice<u32>, CudaFftError> {
        // Round up to power of 2 for better reuse
        let size = min_len.next_power_of_two();

        if let Some(buffers) = self.available.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer);
            }
        }

        // Allocate new buffer
        // Safety: We're allocating uninitialized memory for the pool
        let buffer = unsafe {
            self.device.alloc::<u32>(size)
        }.map_err(|e| CudaFftError::MemoryAllocation(format!("{:?}", e)))?;

        self.total_allocated += size * std::mem::size_of::<u32>();

        Ok(buffer)
    }

    /// Return a buffer to the pool for reuse.
    pub fn put(&mut self, buffer: CudaSlice<u32>, size: usize) {
        let size = size.next_power_of_two();
        self.available.entry(size).or_default().push(buffer);
    }

    /// Get total allocated memory in bytes.
    pub fn total_allocated_bytes(&self) -> usize {
        self.total_allocated
    }

    /// Clear all pooled buffers.
    pub fn clear(&mut self) {
        self.available.clear();
    }
}

#[cfg(feature = "cuda-runtime")]
impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::new().expect("Failed to create GPU buffer pool")
    }
}

// =============================================================================
// Stubs for non-CUDA builds
// =============================================================================

#[cfg(not(feature = "cuda-runtime"))]
pub struct GpuBuffer<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(not(feature = "cuda-runtime"))]
impl<T> GpuBuffer<T> {
    pub fn from_cpu(_data: &[T]) -> Result<Self, CudaFftError> {
        Err(CudaFftError::NoDevice)
    }

    pub fn to_cpu(&self) -> Result<Vec<T>, CudaFftError> {
        Err(CudaFftError::NoDevice)
    }

    pub fn len(&self) -> usize { 0 }
    pub fn is_empty(&self) -> bool { true }
}

#[cfg(not(feature = "cuda-runtime"))]
pub type GpuM31Buffer = GpuBuffer<u32>;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::GpuBuffer;

    #[test]
    #[cfg(not(feature = "cuda-runtime"))]
    fn test_no_cuda_returns_error() {
        let result = GpuBuffer::<u32>::from_cpu(&[1, 2, 3]);
        assert!(result.is_err());
    }
}
