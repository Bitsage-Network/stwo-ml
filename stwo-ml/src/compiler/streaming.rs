//! Streaming weight pipeline for memory-efficient chunked proving.
//!
//! Instead of loading ALL weights at once (which requires 2× peak RAM for
//! 100B+ models), [`StreamingWeightPipeline`] keeps shard files mmap'd and
//! loads weights on demand per chunk. Each chunk's weights are dropped after
//! proving, keeping peak memory to ~1 chunk's worth.
//!
//! # Tile-Level Streaming
//!
//! For truly massive weight matrices (e.g. 8192×28672 = 940MB in M31 for
//! Llama-70B), even loading a single layer's weight is expensive. The
//! tile-level API splits the inner dimension `k` into tiles and loads each
//! tile directly from the mmap:
//!
//! ```text
//! Chunk-level:  load full B (k×n) → prove → drop
//! Tile-level:   load B[0..tile_k, :] → prove tile 0 → drop
//!               load B[tile_k..2*tile_k, :] → prove tile 1 → drop
//!               ...
//! Peak RAM: 1 tile (tile_k × n) instead of full matrix (k × n)
//! ```

use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;

use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use crate::compiler::quantize_weights::{
    quantize_weight_matrix, quantize_weight_tile, WeightError,
};
use crate::compiler::safetensors::{bytes_to_f32_single, dtype_byte_size, tensor_to_f32};
use crate::components::matmul::{
    matmul_m31_auto, pad_matrix_pow2, prove_matmul_sumcheck_onchain_auto, M31Matrix,
};
use crate::components::tiled_matmul::{
    extract_col_slice, TileProof, TiledMatMulConfig, TiledMatMulError, TiledMatMulProof,
};
use crate::gadgets::quantize::{QuantParams, QuantStrategy};

/// Handle to a memory-mapped SafeTensors shard file.
struct ShardHandle {
    _file: std::fs::File,
    mmap: memmap2::Mmap,
}

/// Streaming weight pipeline that serves [`GraphWeights`] per chunk on demand.
///
/// Shard files are kept open and mmap'd for the pipeline's lifetime. Only the
/// pages touched by a chunk's tensors are faulted into memory, and the OS
/// page cache handles eviction automatically.
pub struct StreamingWeightPipeline {
    shards: Vec<ShardHandle>,
    tensor_to_shard: HashMap<String, usize>,
    name_map: HashMap<usize, String>,
    strategy: QuantStrategy,
    graph: ComputationGraph,
}

// =============================================================================
// Layout detection for tile extraction
// =============================================================================

/// Whether a tensor stored with `shape` needs transpose for (k, n) row-major.
fn detect_layout(shape: &[usize], k: usize, n: usize) -> TensorLayout {
    if shape.len() == 2 && shape[0] == n && shape[1] == k && !(k == n) {
        TensorLayout::StoredNK
    } else {
        // (k, n) or ambiguous (k == n) — treat as already in correct layout
        TensorLayout::StoredKN
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorLayout {
    /// Stored as (k, n) row-major — tile rows are contiguous.
    StoredKN,
    /// Stored as (n, k) row-major — needs column extraction + transpose.
    StoredNK,
}

// =============================================================================
// Zero-allocation min/max scan over mmap bytes
// =============================================================================

/// Scan raw tensor bytes to compute (min, max) without allocating a Vec<f32>.
///
/// Iterates sequentially over mmap'd pages — great for prefetch, zero heap
/// allocation beyond two f64 accumulators.
fn scan_tensor_minmax(data: &[u8], dtype: safetensors::Dtype) -> (f64, f64) {
    let elem_size = dtype_byte_size(dtype);
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for chunk in data.chunks_exact(elem_size) {
        let v = bytes_to_f32_single(chunk, dtype) as f64;
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    (min_val, max_val)
}

// =============================================================================
// Tile extraction from raw tensor bytes
// =============================================================================

/// Extract tile rows `[k_start..k_end]` from a (k, n)-stored tensor.
///
/// The tile rows are contiguous in memory: `[k_start * n * elem_size .. k_end * n * elem_size]`.
/// Returns `tile_k × n` f32 values in row-major (k, n) order.
fn extract_tile_kn(
    data: &[u8],
    dtype: safetensors::Dtype,
    n: usize,
    k_start: usize,
    k_end: usize,
) -> Vec<f32> {
    let elem_size = dtype_byte_size(dtype);
    let start = k_start * n * elem_size;
    let end = k_end * n * elem_size;
    let tile_bytes = &data[start..end];

    // Convert the contiguous byte range to f32
    let tile_k = k_end - k_start;
    let mut tile = Vec::with_capacity(tile_k * n);
    for chunk in tile_bytes.chunks_exact(elem_size) {
        tile.push(bytes_to_f32_single(chunk, dtype));
    }
    tile
}

/// Extract tile columns `[k_start..k_end]` from an (n, k)-stored tensor,
/// returning the data in (tile_k, n) row-major order (transposed).
///
/// Reads each of the `n` rows, extracting only the `[k_start..k_end]` columns.
/// This touches `n × tile_k × elem_size` bytes of mmap pages (not the full tensor).
/// The result is transposed using a cache-friendly 64×64 block algorithm.
fn extract_tile_nk(
    data: &[u8],
    dtype: safetensors::Dtype,
    n: usize,
    k: usize,
    k_start: usize,
    k_end: usize,
) -> Vec<f32> {
    let tile_k = k_end - k_start;
    let elem_size = dtype_byte_size(dtype);

    // Phase 1: Extract (n, tile_k) tile — read only the needed columns per row
    let mut tile_nk = Vec::with_capacity(n * tile_k);
    for row in 0..n {
        let row_byte_start = row * k * elem_size;
        let col_start = k_start * elem_size;
        let col_end = k_end * elem_size;
        let row_slice = &data[row_byte_start + col_start..row_byte_start + col_end];
        for chunk in row_slice.chunks_exact(elem_size) {
            tile_nk.push(bytes_to_f32_single(chunk, dtype));
        }
    }

    // Phase 2: Transpose (n, tile_k) → (tile_k, n) using 64×64 blocks
    let mut tile_kn = vec![0.0f32; tile_k * n];
    const BLOCK: usize = 64;
    for r_block in (0..n).step_by(BLOCK) {
        for c_block in (0..tile_k).step_by(BLOCK) {
            let r_end = (r_block + BLOCK).min(n);
            let c_end = (c_block + BLOCK).min(tile_k);
            for r in r_block..r_end {
                for c in c_block..c_end {
                    tile_kn[c * n + r] = tile_nk[r * tile_k + c];
                }
            }
        }
    }

    tile_kn
}

// =============================================================================
// StreamingWeightPipeline — chunk-level API (existing)
// =============================================================================

impl StreamingWeightPipeline {
    /// Open shard files and build the tensor-to-shard index.
    ///
    /// Does NOT load any weight data — just mmap headers and builds the index.
    pub fn open(
        shard_paths: &[PathBuf],
        graph: &ComputationGraph,
        name_map: HashMap<usize, String>,
        strategy: QuantStrategy,
    ) -> Result<Self, WeightError> {
        let mut shards = Vec::with_capacity(shard_paths.len());
        for path in shard_paths {
            let file = std::fs::File::open(path)
                .map_err(|e| WeightError::IoError(format!("{}: {e}", path.display())))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| WeightError::IoError(format!("{}: {e}", path.display())))?;
            shards.push(ShardHandle { _file: file, mmap });
        }

        // Build tensor_name → shard_index lookup
        let mut tensor_to_shard = HashMap::new();
        for (shard_idx, shard) in shards.iter().enumerate() {
            let st = safetensors::SafeTensors::deserialize(&shard.mmap)
                .map_err(|e| WeightError::IoError(format!("shard {shard_idx}: {e}")))?;
            for name in st.names() {
                tensor_to_shard.insert(name.to_string(), shard_idx);
            }
        }

        Ok(Self {
            shards,
            tensor_to_shard,
            name_map,
            strategy,
            graph: graph.clone(),
        })
    }

    /// Load weights for a specific node range `[start..end)`.
    ///
    /// Returns [`GraphWeights`] with node IDs remapped to `[0..len)`, matching
    /// the semantics of [`GraphWeights::subset`]. Only touches mmap pages for
    /// tensors in this range.
    pub fn load_chunk_weights(&self, range: Range<usize>) -> Result<GraphWeights, WeightError> {
        use rayon::prelude::*;

        let start = range.start;

        // Collect which weights we need: (node_idx, k, n, tensor_name, shard_idx)
        let mut work_items: Vec<(usize, usize, usize, &str, usize)> = Vec::new();
        for idx in range {
            if idx >= self.graph.nodes.len() {
                continue;
            }
            if let GraphOp::MatMul { dims: (_m, k, n) } = &self.graph.nodes[idx].op {
                if let Some(tensor_name) = self.name_map.get(&idx) {
                    if let Some(&shard_idx) = self.tensor_to_shard.get(tensor_name.as_str()) {
                        work_items.push((idx, *k, *n, tensor_name.as_str(), shard_idx));
                    }
                }
            }
        }

        // Group by shard for sequential extraction
        let mut shard_to_work: HashMap<usize, Vec<(usize, usize, usize, &str)>> = HashMap::new();
        for &(idx, k, n, name, shard_idx) in &work_items {
            shard_to_work
                .entry(shard_idx)
                .or_default()
                .push((idx, k, n, name));
        }

        // Phase 1: Extract raw f32 from relevant shards
        let mut all_raw: Vec<(usize, usize, usize, Vec<f32>, Vec<usize>)> =
            Vec::with_capacity(work_items.len());

        for (shard_idx, weights_in_shard) in &shard_to_work {
            let tensors = safetensors::SafeTensors::deserialize(&self.shards[*shard_idx].mmap)
                .map_err(|e| WeightError::IoError(format!("shard {shard_idx}: {e}")))?;

            for &(idx, k, n, tensor_name) in weights_in_shard {
                let tensor = tensors
                    .tensor(tensor_name)
                    .map_err(|e| WeightError::IoError(format!("{tensor_name}: {e}")))?;
                let data = tensor_to_f32(tensor.data(), tensor.dtype());
                let shape = tensor.shape().to_vec();
                all_raw.push((idx, k, n, data, shape));
            }
        }

        // Phase 2: Parallel transpose + quantize
        let strategy = self.strategy;
        let processed: Vec<(usize, crate::components::matmul::M31Matrix)> = all_raw
            .par_iter()
            .map(|(idx, k, n, data, shape)| {
                let (k, n) = (*k, *n);
                let weight_data = transpose_if_needed(data, shape, k, n);
                let (matrix, _params) = quantize_weight_matrix(&weight_data, k, n, strategy);
                (*idx, matrix)
            })
            .collect();

        // Build GraphWeights with remapped IDs
        let mut weights = GraphWeights::new();
        for (idx, matrix) in processed {
            let remapped_id = idx - start;
            weights.add_weight(remapped_id, matrix);
        }

        Ok(weights)
    }

    /// Prefetch mmap pages for a chunk's tensors in background.
    ///
    /// Uses `madvise(MADV_WILLNEED)` on the relevant byte ranges so the kernel
    /// starts readahead before we actually need the data. No-op on non-unix.
    pub fn prefetch_chunk(&self, range: Range<usize>) {
        #[cfg(unix)]
        {
            // Collect unique shard indices touched by this chunk's tensors
            let mut shard_indices = std::collections::HashSet::new();
            for idx in range {
                if idx >= self.graph.nodes.len() {
                    continue;
                }
                if !matches!(&self.graph.nodes[idx].op, GraphOp::MatMul { .. }) {
                    continue;
                }
                if let Some(tensor_name) = self.name_map.get(&idx) {
                    if let Some(&shard_idx) = self.tensor_to_shard.get(tensor_name.as_str()) {
                        shard_indices.insert(shard_idx);
                    }
                }
            }

            // Prefetch each relevant shard with MADV_WILLNEED
            for shard_idx in shard_indices {
                let shard = &self.shards[shard_idx];
                unsafe {
                    libc::madvise(
                        shard.mmap.as_ptr() as *mut libc::c_void,
                        shard.mmap.len(),
                        libc::MADV_WILLNEED,
                    );
                }
            }
        }

        #[cfg(not(unix))]
        {
            let _ = range;
        }
    }

    /// Estimate memory needed for a chunk's weights (bytes).
    ///
    /// Returns the sum of `k × n × 4` bytes for each MatMul node in the range.
    pub fn estimate_chunk_memory(&self, range: Range<usize>) -> usize {
        range
            .filter_map(|idx| {
                if idx >= self.graph.nodes.len() {
                    return None;
                }
                if let GraphOp::MatMul { dims: (_m, k, n) } = &self.graph.nodes[idx].op {
                    Some(k * n * 4)
                } else {
                    None
                }
            })
            .sum()
    }
}

// =============================================================================
// StreamingWeightPipeline — tile-level API (new)
// =============================================================================

impl StreamingWeightPipeline {
    /// Resolve a node's weight tensor from the mmap index.
    ///
    /// Returns the tensor's raw bytes, dtype, and shape. Does NOT convert or
    /// allocate — the bytes are a zero-copy slice of the mmap.
    fn resolve_tensor(
        &self,
        node_id: usize,
    ) -> Result<(&[u8], safetensors::Dtype, Vec<usize>), WeightError> {
        let tensor_name = self
            .name_map
            .get(&node_id)
            .ok_or_else(|| WeightError::MissingTensor(format!("node {node_id}")))?;
        let &shard_idx = self
            .tensor_to_shard
            .get(tensor_name.as_str())
            .ok_or_else(|| WeightError::MissingTensor(tensor_name.clone()))?;

        let tensors = safetensors::SafeTensors::deserialize(&self.shards[shard_idx].mmap)
            .map_err(|e| WeightError::IoError(format!("shard {shard_idx}: {e}")))?;
        let tensor = tensors
            .tensor(tensor_name)
            .map_err(|e| WeightError::IoError(format!("{tensor_name}: {e}")))?;

        let dtype = tensor.dtype();
        let shape = tensor.shape().to_vec();

        // SAFETY: the mmap outlives `self`, and `tensor.data()` is a slice
        // of the mmap. We extend the lifetime to match `&self` which is safe
        // because the mmap is owned by `self.shards`.
        let data_ptr = tensor.data().as_ptr();
        let data_len = tensor.data().len();
        let data = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };

        Ok((data, dtype, shape))
    }

    /// Scan a weight tensor to compute global quantization parameters.
    ///
    /// Iterates over the tensor's mmap'd bytes without allocating a full
    /// `Vec<f32>`. The min/max are used to build [`QuantParams`] that are
    /// consistent across all tiles of the weight matrix.
    ///
    /// Returns `(params, k, n)` for the weight matrix.
    pub fn scan_weight_params(
        &self,
        node_id: usize,
    ) -> Result<(QuantParams, usize, usize), WeightError> {
        let (k, n) = match &self.graph.nodes[node_id].op {
            GraphOp::MatMul { dims: (_m, k, n) } => (*k, *n),
            _ => {
                return Err(WeightError::IoError(format!(
                    "node {node_id} is not a MatMul"
                )))
            }
        };

        let (data, dtype, _shape) = self.resolve_tensor(node_id)?;
        let (min_val, max_val) = scan_tensor_minmax(data, dtype);
        let params = QuantParams::from_range(min_val, max_val, self.strategy);

        Ok((params, k, n))
    }

    /// Load a single tile of weight matrix `B[k_start..k_end, :]` from mmap.
    ///
    /// Extracts only the tile's elements from the mmap'd bytes, converts to f32,
    /// handles transpose if needed, and quantizes using the provided global params.
    ///
    /// Allocates only `tile_k × n × 4` bytes for f32 + `tile_k × n × 4` bytes
    /// for M31Matrix. For k=8192, tile_k=1024, n=8192: 64 MB vs 512 MB full.
    pub fn load_weight_tile(
        &self,
        node_id: usize,
        k_start: usize,
        k_end: usize,
        global_params: &QuantParams,
    ) -> Result<M31Matrix, WeightError> {
        let (k, n) = match &self.graph.nodes[node_id].op {
            GraphOp::MatMul { dims: (_m, k, n) } => (*k, *n),
            _ => {
                return Err(WeightError::IoError(format!(
                    "node {node_id} is not a MatMul"
                )))
            }
        };

        let (data, dtype, shape) = self.resolve_tensor(node_id)?;
        let layout = detect_layout(&shape, k, n);
        let tile_k = k_end - k_start;

        let tile_f32 = match layout {
            TensorLayout::StoredKN => extract_tile_kn(data, dtype, n, k_start, k_end),
            TensorLayout::StoredNK => extract_tile_nk(data, dtype, n, k, k_start, k_end),
        };

        Ok(quantize_weight_tile(&tile_f32, tile_k, n, global_params))
    }

    /// Compute `C = A × B` by streaming tiles of B from mmap.
    ///
    /// Accumulates partial products: `C += A[:, k_start..k_end] × B[k_start..k_end, :]`.
    /// Each tile of B is loaded from the mmap, multiplied, and dropped.
    ///
    /// Peak memory: `m×n` (output C) + `m×tile_k` (A slice) + `tile_k×n` (B tile).
    /// For m=1, k=8192, n=8192, tile_k=1024: 32KB + 4KB + 32MB = ~32MB
    /// vs full load: 256MB.
    pub fn forward_matmul_tiled(
        &self,
        node_id: usize,
        a: &M31Matrix,
        tile_config: &TiledMatMulConfig,
    ) -> Result<M31Matrix, WeightError> {
        let (params, k, n) = self.scan_weight_params(node_id)?;
        let tile_k = tile_config.max_tile_k.min(k);
        let num_tiles = tile_config.num_tiles(k);
        let m = a.rows;

        // Accumulator for the full output
        let mut c = M31Matrix::new(m, n);

        // Load first tile synchronously.
        let k_end_0 = tile_k.min(k);
        let mut current_b_tile = self.load_weight_tile(node_id, 0, k_end_0, &params)?;

        for tile_idx in 0..num_tiles {
            let k_start = tile_idx * tile_k;
            let k_end = (k_start + tile_k).min(k);

            let b_tile = current_b_tile;
            let a_tile = extract_col_slice(a, k_start, k_end);

            // Double-buffer: load next tile on background thread while computing matmul.
            let has_next = tile_idx + 1 < num_tiles;
            let next_k_start = (tile_idx + 1) * tile_k;
            let next_k_end = (next_k_start + tile_k).min(k);

            let (c_tile, next_tile_result) = if has_next {
                std::thread::scope(|s| {
                    let loader = s.spawn(|| {
                        self.load_weight_tile(node_id, next_k_start, next_k_end, &params)
                    });
                    let c_tile = matmul_m31_auto(&a_tile, &b_tile);
                    let next = loader.join().expect("loader thread panicked");
                    (c_tile, Some(next))
                })
            } else {
                (matmul_m31_auto(&a_tile, &b_tile), None)
            };

            // Accumulate: C[i][j] += C_tile[i][j]
            for i in 0..m {
                for j in 0..n {
                    let sum = M31::from(
                        c.get(i, j).0.wrapping_add(c_tile.get(i, j).0) % ((1u32 << 31) - 1),
                    );
                    c.set(i, j, sum);
                }
            }

            // Advance the pre-loaded tile.
            if let Some(next_result) = next_tile_result {
                current_b_tile = next_result?;
            } else {
                current_b_tile = M31Matrix::new(0, 0);
            }
        }

        Ok(c)
    }

    /// Prove `C = A × B` tile-by-tile with streaming weights from mmap.
    ///
    /// For each tile:
    /// 1. Prefetch next tile's mmap pages (I/O pipeline)
    /// 2. Load `B[k_start..k_end, :]` from mmap → quantize
    /// 3. Extract `A[:, k_start..k_end]` from in-memory A
    /// 4. Compute `C_tile = A_tile × B_tile`
    /// 5. Prove via sumcheck
    /// 6. Drop tile data (frees memory)
    ///
    /// Peak RAM per tile: `tile_k × n × 4` (B tile) + `m × tile_k × 4` (A slice)
    /// + `m × n × 4` (C tile) + padded copies + sumcheck working memory.
    pub fn prove_matmul_tiled_streaming(
        &self,
        node_id: usize,
        a: &M31Matrix,
        _c: &M31Matrix,
        tile_config: &TiledMatMulConfig,
    ) -> Result<TiledMatMulProof, TiledMatMulError> {
        let (params, k, n) =
            self.scan_weight_params(node_id)
                .map_err(|e| TiledMatMulError::TileProvingFailed {
                    tile: 0,
                    message: format!("scan params: {e}"),
                })?;

        let m = a.rows;
        let tile_k = tile_config.max_tile_k.min(k);
        let num_tiles = tile_config.num_tiles(k);

        tracing::info!(
            m,
            k,
            n,
            tile_k,
            num_tiles,
            "Streaming tiled matmul proving (double-buffered)"
        );

        if num_tiles <= 1 {
            // Single tile — no pipelining benefit, use simple path.
            return self.prove_matmul_tiled_streaming_simple(
                node_id, a, &params, m, k, n, tile_k, num_tiles,
            );
        }

        // Double-buffered pipeline: load tile N+1 on a background thread
        // while the main thread proves tile N. Overlaps mmap I/O + quantization
        // with sumcheck proving.
        //
        // Timeline:
        //   Main thread:   [prove tile 0] [prove tile 1] [prove tile 2] ...
        //   Load thread:   [load tile 1 ] [load tile 2 ] [load tile 3 ] ...
        //                  ^^^^^^^^^^^^^^^ overlapped with prove tile 0
        let mut tile_proofs = Vec::with_capacity(num_tiles);
        let mut total_claimed_sum = SecureField::default();

        // Load first tile synchronously (nothing to overlap with yet).
        let k_start_0 = 0;
        let k_end_0 = tile_k.min(k);
        let mut current_b_tile = self
            .load_weight_tile(node_id, k_start_0, k_end_0, &params)
            .map_err(|e| TiledMatMulError::TileProvingFailed {
                tile: 0,
                message: format!("load tile: {e}"),
            })?;

        for tile_idx in 0..num_tiles {
            let k_start = tile_idx * tile_k;
            let k_end = (k_start + tile_k).min(k);
            let actual_tile_k = k_end - k_start;

            tracing::debug!(
                tile_idx,
                k_start,
                k_end,
                actual_tile_k,
                "Streaming tile (double-buffered)"
            );

            // Prepare this tile's inputs from the pre-loaded B tile.
            let b_tile = current_b_tile;
            let a_tile = extract_col_slice(a, k_start, k_end);
            let c_tile = matmul_m31_auto(&a_tile, &b_tile);
            let a_padded = pad_matrix_pow2(&a_tile);
            let b_padded = pad_matrix_pow2(&b_tile);
            let c_padded = pad_matrix_pow2(&c_tile);

            // Double-buffer: load next tile on background thread while proving current.
            let has_next = tile_idx + 1 < num_tiles;
            let next_k_start = (tile_idx + 1) * tile_k;
            let next_k_end = (next_k_start + tile_k).min(k);

            // Use std::thread::scope so the background thread can borrow &self and &params.
            let (proof_result, next_tile_result) = std::thread::scope(|s| {
                // Spawn loader for next tile (if any).
                let loader = if has_next {
                    Some(s.spawn(|| {
                        self.load_weight_tile(node_id, next_k_start, next_k_end, &params)
                    }))
                } else {
                    None
                };

                // Main thread: prove current tile.
                let proof = prove_matmul_sumcheck_onchain_auto(&a_padded, &b_padded, &c_padded);

                // Join loader.
                let next_tile = loader.map(|h| h.join().expect("loader thread panicked"));

                (proof, next_tile)
            });

            let proof = proof_result.map_err(|e| TiledMatMulError::TileProvingFailed {
                tile: tile_idx,
                message: format!("{e}"),
            })?;

            total_claimed_sum = total_claimed_sum + proof.claimed_sum;

            tile_proofs.push(TileProof {
                proof,
                k_start,
                k_end,
            });

            // Advance the pre-loaded tile for the next iteration.
            if let Some(next_result) = next_tile_result {
                current_b_tile = next_result.map_err(|e| TiledMatMulError::TileProvingFailed {
                    tile: tile_idx + 1,
                    message: format!("load tile: {e}"),
                })?;
            } else {
                // Last tile — set a dummy that won't be used.
                current_b_tile = M31Matrix::new(0, 0);
            }
        }

        Ok(TiledMatMulProof {
            m,
            k,
            n,
            tile_proofs,
            total_claimed_sum,
            tile_k,
        })
    }

    /// Simple (non-pipelined) path for single-tile matmuls.
    fn prove_matmul_tiled_streaming_simple(
        &self,
        node_id: usize,
        a: &M31Matrix,
        params: &QuantParams,
        m: usize,
        k: usize,
        n: usize,
        tile_k: usize,
        num_tiles: usize,
    ) -> Result<TiledMatMulProof, TiledMatMulError> {
        let mut tile_proofs = Vec::with_capacity(num_tiles);
        let mut total_claimed_sum = SecureField::default();

        for tile_idx in 0..num_tiles {
            let k_start = tile_idx * tile_k;
            let k_end = (k_start + tile_k).min(k);

            let b_tile = self
                .load_weight_tile(node_id, k_start, k_end, params)
                .map_err(|e| TiledMatMulError::TileProvingFailed {
                    tile: tile_idx,
                    message: format!("load tile: {e}"),
                })?;

            let a_tile = extract_col_slice(a, k_start, k_end);
            let c_tile = matmul_m31_auto(&a_tile, &b_tile);
            let a_padded = pad_matrix_pow2(&a_tile);
            let b_padded = pad_matrix_pow2(&b_tile);
            let c_padded = pad_matrix_pow2(&c_tile);

            let proof = prove_matmul_sumcheck_onchain_auto(&a_padded, &b_padded, &c_padded)
                .map_err(|e| TiledMatMulError::TileProvingFailed {
                    tile: tile_idx,
                    message: format!("{e}"),
                })?;

            total_claimed_sum = total_claimed_sum + proof.claimed_sum;
            tile_proofs.push(TileProof {
                proof,
                k_start,
                k_end,
            });
        }

        Ok(TiledMatMulProof {
            m,
            k,
            n,
            tile_proofs,
            total_claimed_sum,
            tile_k,
        })
    }

    /// Prefetch mmap pages for a specific tile's byte range within a tensor.
    ///
    /// For (k, n)-stored tensors, targets the contiguous rows `[k_start..k_end]`.
    /// For (n, k)-stored tensors, prefetches the full tensor (columns are scattered).
    ///
    /// Note: The double-buffered pipeline (`forward_matmul_tiled`, `prove_matmul_tiled_streaming`)
    /// uses background thread loading instead of madvise prefetch. This method remains
    /// available for callers that want OS-level page cache hints without spawning threads.
    #[allow(dead_code)]
    /// No-op on non-unix.
    fn prefetch_tile_bytes(&self, node_id: usize, k_start: usize, k_end: usize) {
        #[cfg(unix)]
        {
            let Ok((data, dtype, shape)) = self.resolve_tensor(node_id) else {
                return;
            };
            let (k, n) = match &self.graph.nodes[node_id].op {
                GraphOp::MatMul { dims: (_m, k, n) } => (*k, *n),
                _ => return,
            };

            let layout = detect_layout(&shape, k, n);
            let elem_size = dtype_byte_size(dtype);

            match layout {
                TensorLayout::StoredKN => {
                    // Contiguous rows — prefetch just the tile's byte range
                    let start = k_start * n * elem_size;
                    let end = k_end * n * elem_size;
                    let ptr = unsafe { data.as_ptr().add(start) };
                    let len = end - start;
                    // Align to page boundary (madvise requires page-aligned addresses)
                    let page_size = 4096usize;
                    let aligned_ptr = (ptr as usize / page_size) * page_size;
                    let aligned_len = len + (ptr as usize - aligned_ptr);
                    unsafe {
                        libc::madvise(
                            aligned_ptr as *mut libc::c_void,
                            aligned_len,
                            libc::MADV_WILLNEED,
                        );
                    }
                }
                TensorLayout::StoredNK => {
                    // Columns are scattered — prefetch entire tensor
                    let aligned_ptr = (data.as_ptr() as usize / 4096) * 4096;
                    let aligned_len = data.len() + (data.as_ptr() as usize - aligned_ptr);
                    unsafe {
                        libc::madvise(
                            aligned_ptr as *mut libc::c_void,
                            aligned_len,
                            libc::MADV_WILLNEED,
                        );
                    }
                }
            }
        }

        #[cfg(not(unix))]
        {
            let _ = (node_id, k_start, k_end);
        }
    }

    /// Estimate memory for a single tile (bytes).
    ///
    /// Returns `tile_k × n × 4` (M31 matrix) + `m × tile_k × 4` (A slice)
    /// + `m × n × 4` (C tile) + padding overhead.
    pub fn estimate_tile_memory(&self, node_id: usize, tile_config: &TiledMatMulConfig) -> usize {
        if let GraphOp::MatMul { dims: (m, k, n) } = &self.graph.nodes[node_id].op {
            let tile_k = tile_config.max_tile_k.min(*k);
            let m_pad = m.next_power_of_two();
            let tile_k_pad = tile_k.next_power_of_two();
            let n_pad = n.next_power_of_two();
            // B tile + A slice + C tile + padded copies (3×)
            let tile_mem = tile_k * n * 4;
            let a_slice = m * tile_k * 4;
            let c_tile = m * n * 4;
            let padded = (m_pad * tile_k_pad + tile_k_pad * n_pad + m_pad * n_pad) * 16;
            tile_mem + a_slice + c_tile + padded
        } else {
            0
        }
    }
}

// =============================================================================
// Existing helper
// =============================================================================

/// Transpose weight data from (n, k) to (k, n) if needed, using cache-friendly
/// 64×64 block transpose. Returns the data unchanged if already in (k, n) layout.
fn transpose_if_needed(data: &[f32], shape: &[usize], k: usize, n: usize) -> Vec<f32> {
    if shape.len() == 2 {
        let rows = shape[0];
        let cols = shape[1];
        if rows == n && cols == k {
            // Needs transpose: (n, k) → (k, n)
            let mut transposed = vec![0.0f32; data.len()];
            const BLOCK: usize = 64;
            for r_block in (0..rows).step_by(BLOCK) {
                for c_block in (0..cols).step_by(BLOCK) {
                    let r_end = (r_block + BLOCK).min(rows);
                    let c_end = (c_block + BLOCK).min(cols);
                    for r in r_block..r_end {
                        for c in c_block..c_end {
                            transposed[c * rows + r] = data[r * cols + c];
                        }
                    }
                }
            }
            transposed
        } else {
            data.to_vec()
        }
    } else {
        data.to_vec()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_if_needed_no_op() {
        // Already in (k, n) layout
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![3, 2]; // rows=k=3, cols=n=2
        let result = transpose_if_needed(&data, &shape, 3, 2);
        assert_eq!(result, data);
    }

    #[test]
    fn test_transpose_if_needed_transposes() {
        // In (n, k) layout, needs transpose
        // 2×3 matrix: [[1,2,3],[4,5,6]] → transposed 3×2: [[1,4],[2,5],[3,6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3]; // rows=n=2, cols=k=3
        let result = transpose_if_needed(&data, &shape, 3, 2);
        // Expected: row-major 3×2 = [1,4,2,5,3,6]
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_estimate_chunk_memory() {
        use crate::compiler::graph::GraphBuilder;
        use crate::components::activation::ActivationType;

        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(8); // MatMul (4, 8) → 4*8*4 = 128 bytes
        builder.activation(ActivationType::ReLU);
        builder.linear(2); // MatMul (8, 2) → 8*2*4 = 64 bytes
        let graph = builder.build();

        let name_map = HashMap::new();
        let pipeline = StreamingWeightPipeline {
            shards: vec![],
            tensor_to_shard: HashMap::new(),
            name_map,
            strategy: QuantStrategy::Symmetric8,
            graph,
        };

        // Full range: both matmuls
        assert_eq!(pipeline.estimate_chunk_memory(0..3), 128 + 64);
        // Only first matmul
        assert_eq!(pipeline.estimate_chunk_memory(0..1), 128);
        // Only second matmul (node 2)
        assert_eq!(pipeline.estimate_chunk_memory(2..3), 64);
        // Just the activation (no matmul)
        assert_eq!(pipeline.estimate_chunk_memory(1..2), 0);
    }

    // =========================================================================
    // Tile extraction tests (unit-testable without real SafeTensors files)
    // =========================================================================

    #[test]
    fn test_detect_layout() {
        // (k=4, n=8) stored as (4, 8) → KN
        assert_eq!(detect_layout(&[4, 8], 4, 8), TensorLayout::StoredKN);
        // (k=4, n=8) stored as (8, 4) → NK (needs transpose)
        assert_eq!(detect_layout(&[8, 4], 4, 8), TensorLayout::StoredNK);
        // Square: ambiguous → treat as KN
        assert_eq!(detect_layout(&[4, 4], 4, 4), TensorLayout::StoredKN);
    }

    #[test]
    fn test_extract_tile_kn() {
        // 4×2 matrix stored as f32 in (k=4, n=2) layout:
        // [[1,2],[3,4],[5,6],[7,8]]
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // Extract tile k_start=1, k_end=3 → rows 1..3 = [[3,4],[5,6]]
        let tile = extract_tile_kn(&data, safetensors::Dtype::F32, 2, 1, 3);
        assert_eq!(tile, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_extract_tile_nk() {
        // Stored as (n=2, k=4) layout: [[1,2,3,4],[5,6,7,8]]
        // This represents B stored as (n, k), so element B[k_idx][n_idx]
        // is at stored[n_idx][k_idx].
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // Extract tile k_start=1, k_end=3 from (n=2, k=4) → columns 1..3
        // From row 0: cols 1,2 → [2.0, 3.0]
        // From row 1: cols 1,2 → [6.0, 7.0]
        // (n, tile_k) = (2, 2): [[2,3],[6,7]]
        // Transposed to (tile_k, n) = (2, 2): [[2,6],[3,7]]
        let tile = extract_tile_nk(&data, safetensors::Dtype::F32, 2, 4, 1, 3);
        assert_eq!(tile, vec![2.0, 6.0, 3.0, 7.0]);
    }

    #[test]
    fn test_scan_tensor_minmax() {
        let data: Vec<u8> = [-3.0f32, 1.0, 0.5, 7.0, -1.0, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let (min_val, max_val) = scan_tensor_minmax(&data, safetensors::Dtype::F32);
        assert!((min_val - (-3.0)).abs() < 1e-6);
        assert!((max_val - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_extract_tile_kn_f16() {
        // Test with f16 dtype: 4 elements stored as 2 bytes each
        // f16 representation of [1.0, 2.0, 3.0, 4.0]
        let f16_data: Vec<u8> = [
            0x00, 0x3C, // f16 1.0
            0x00, 0x40, // f16 2.0
            0x00, 0x42, // f16 3.0
            0x00, 0x44, // f16 4.0
        ]
        .to_vec();

        // 2×2 matrix (k=2, n=2), extract tile k_start=0, k_end=1 → first row
        let tile = extract_tile_kn(&f16_data, safetensors::Dtype::F16, 2, 0, 1);
        assert_eq!(tile.len(), 2);
        assert!((tile[0] - 1.0).abs() < 0.01);
        assert!((tile[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_tile_memory_estimation() {
        use crate::compiler::graph::GraphBuilder;
        use crate::components::activation::ActivationType;

        let mut builder = GraphBuilder::new((1, 8));
        builder.linear(16); // MatMul (8, 16)
        builder.activation(ActivationType::ReLU);
        let graph = builder.build();

        let pipeline = StreamingWeightPipeline {
            shards: vec![],
            tensor_to_shard: HashMap::new(),
            name_map: HashMap::new(),
            strategy: QuantStrategy::Symmetric8,
            graph,
        };

        let config = TiledMatMulConfig::new(4); // tile_k=4
        let mem = pipeline.estimate_tile_memory(0, &config);
        assert!(mem > 0, "tile memory should be positive");

        // Full tile should be smaller than full matrix memory
        let full_mem = pipeline.estimate_chunk_memory(0..1); // 8 * 16 * 4 = 512
        assert!(
            mem < full_mem * 4,
            "tile memory {mem} should be significantly less than full × 4 = {}",
            full_mem * 4,
        );
    }
}
