//! SafeTensors weight loading.
//!
//! Loads model weights from SafeTensors format files using memory-mapped I/O
//! for efficient handling of large models. Weights are quantized to M31
//! during loading.

use std::collections::HashMap;
use std::path::Path;

use stwo::core::fields::m31::M31;

use crate::compiler::graph::{ComputationGraph, GraphWeights, GraphOp};
use crate::compiler::quantize_weights::{quantize_weight_matrix, quantize_bias_vector, WeightError};
use crate::components::matmul::M31Matrix;
use crate::gadgets::quantize::QuantStrategy;

/// Load weights from a SafeTensors file and quantize to M31.
///
/// Uses memory-mapped I/O for large models (14B = ~28GB fp16).
pub fn load_weights(
    path: &Path,
    graph: &ComputationGraph,
    strategy: QuantStrategy,
) -> Result<GraphWeights, WeightError> {
    let file = std::fs::File::open(path)
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let tensors = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let mut weights = GraphWeights::new();

    // Map tensor names to graph nodes
    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            // Try common naming conventions: weight.{idx}, layers.{idx}.weight, etc.
            let tensor_names = [
                format!("weight.{idx}"),
                format!("layers.{idx}.weight"),
                format!("model.layers.{idx}.weight"),
            ];

            for name in &tensor_names {
                if let Ok(tensor) = tensors.tensor(name) {
                    let data = tensor_to_f32(tensor.data(), tensor.dtype());
                    let (matrix, _params) = quantize_weight_matrix(
                        &data, *k, *n, strategy,
                    );
                    weights.add_weight(idx, matrix);
                    break;
                }
            }

            // Try bias
            let bias_names = [
                format!("bias.{idx}"),
                format!("layers.{idx}.bias"),
            ];
            for name in &bias_names {
                if let Ok(tensor) = tensors.tensor(name) {
                    let data = tensor_to_f32(tensor.data(), tensor.dtype());
                    let (bias, _params) = quantize_bias_vector(&data, strategy);
                    weights.add_bias(idx, bias);
                    break;
                }
            }
        }
    }

    Ok(weights)
}

/// Convert raw tensor bytes to f32 based on dtype.
fn tensor_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => {
            // Convert f16 to f32
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half_to_f32(bits)
                })
                .collect()
        }
        safetensors::Dtype::BF16 => {
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    bf16_to_f32(bits)
                })
                .collect()
        }
        _ => {
            // Fallback: treat as f32
            data.chunks_exact(4)
                .map(|c| {
                    if c.len() == 4 {
                        f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    } else {
                        0.0
                    }
                })
                .collect()
        }
    }
}

/// Convert f16 bits to f32.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31) // ±0
        } else {
            // Subnormal
            let mut e = 0i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3FF;
            let new_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (new_exp << 23) | (f << 13))
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23)) // ±Inf
        } else {
            f32::NAN
        }
    } else {
        let new_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
    }
}

/// Convert bf16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// List all tensor names in a SafeTensors file.
pub fn list_tensors(path: &Path) -> Result<Vec<String>, WeightError> {
    let file = std::fs::File::open(path)
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let tensors = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    Ok(tensors.names().into_iter().map(|s| s.to_string()).collect())
}

/// Load weights with an explicit name mapping.
///
/// `name_map` maps graph node IDs to tensor names in the SafeTensors file.
pub fn load_weights_with_mapping(
    path: &Path,
    graph: &ComputationGraph,
    name_map: &std::collections::HashMap<usize, String>,
    strategy: QuantStrategy,
) -> Result<GraphWeights, WeightError> {
    let file = std::fs::File::open(path)
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let tensors = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let mut weights = GraphWeights::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            if let Some(tensor_name) = name_map.get(&idx) {
                let tensor = tensors.tensor(tensor_name)
                    .map_err(|_| WeightError::MissingTensor(tensor_name.clone()))?;
                let data = tensor_to_f32(tensor.data(), tensor.dtype());
                let (matrix, _params) = quantize_weight_matrix(&data, *k, *n, strategy);
                weights.add_weight(idx, matrix);
            }
        }
    }

    Ok(weights)
}

/// Discover SafeTensors shard files in a directory matching a base pattern.
///
/// Finds and sorts files matching `*{base_pattern}*.safetensors`.
pub fn discover_shards(
    dir: &Path,
    base_pattern: &str,
) -> Result<Vec<std::path::PathBuf>, WeightError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let mut shards: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            name.contains(base_pattern) && name.ends_with(".safetensors")
        })
        .collect();

    shards.sort();
    Ok(shards)
}

/// List all tensors across multiple shards, with shard index.
pub fn list_tensors_sharded(
    shard_paths: &[std::path::PathBuf],
) -> Result<Vec<(String, usize)>, WeightError> {
    let mut result = Vec::new();

    for (shard_idx, path) in shard_paths.iter().enumerate() {
        let file = std::fs::File::open(path)
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| WeightError::IoError(e.to_string()))?;

        for name in tensors.names() {
            result.push((name.to_string(), shard_idx));
        }
    }

    Ok(result)
}

/// Load weights from multiple SafeTensors shard files.
///
/// Searches each shard for tensor names matching MatMul nodes in the graph.
pub fn load_weights_sharded(
    shard_paths: &[std::path::PathBuf],
    graph: &ComputationGraph,
    strategy: QuantStrategy,
) -> Result<GraphWeights, WeightError> {
    // Memory-map all shards
    let mut shard_data: Vec<(std::fs::File, memmap2::Mmap)> = Vec::new();
    for path in shard_paths {
        let file = std::fs::File::open(path)
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        shard_data.push((file, mmap));
    }

    let mut weights = GraphWeights::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            let tensor_names = [
                format!("weight.{idx}"),
                format!("layers.{idx}.weight"),
                format!("model.layers.{idx}.weight"),
            ];

            'outer: for name in &tensor_names {
                for (_file, mmap) in &shard_data {
                    let tensors = safetensors::SafeTensors::deserialize(mmap)
                        .map_err(|e| WeightError::IoError(e.to_string()))?;

                    if let Ok(tensor) = tensors.tensor(name) {
                        let data = tensor_to_f32(tensor.data(), tensor.dtype());
                        let (matrix, _params) = quantize_weight_matrix(
                            &data, *k, *n, strategy,
                        );
                        weights.add_weight(idx, matrix);
                        break 'outer;
                    }
                }
            }

            // Try bias across shards
            let bias_names = [
                format!("bias.{idx}"),
                format!("layers.{idx}.bias"),
            ];
            'bias_outer: for name in &bias_names {
                for (_file, mmap) in &shard_data {
                    let tensors = safetensors::SafeTensors::deserialize(mmap)
                        .map_err(|e| WeightError::IoError(e.to_string()))?;

                    if let Ok(tensor) = tensors.tensor(name) {
                        let data = tensor_to_f32(tensor.data(), tensor.dtype());
                        let (bias, _params) = quantize_bias_vector(&data, strategy);
                        weights.add_bias(idx, bias);
                        break 'bias_outer;
                    }
                }
            }
        }
    }

    Ok(weights)
}

/// Location of a tensor within mmap'd shard data.
#[derive(Debug, Clone)]
struct TensorLocation {
    /// Index into `LazyWeights::shard_data`.
    shard_idx: usize,
    /// Tensor name within the SafeTensors file.
    tensor_name: String,
    /// Weight matrix dimensions (rows, cols).
    rows: usize,
    cols: usize,
}

/// Location of a bias tensor within mmap'd shard data.
#[derive(Debug, Clone)]
struct BiasLocation {
    shard_idx: usize,
    tensor_name: String,
    #[allow(dead_code)]
    len: usize,
}

/// Per-layer lazy weight source backed by mmap'd SafeTensors.
///
/// Weights are loaded (quantized from raw bytes to M31) on demand and can be
/// evicted after each layer is proven, keeping peak memory to the size of a
/// single layer's weights.
///
/// # Example
///
/// ```ignore
/// let mut lazy = LazyWeights::open(&path, &graph, QuantStrategy::Direct)?;
/// // Load, use, and evict one layer at a time:
/// let w = lazy.get_weight(0);  // auto-loads from mmap
/// lazy.evict_layer(0);         // frees M31 data
/// ```
pub struct LazyWeights {
    /// mmap'd shard data (kept alive for the lifetime of LazyWeights).
    shard_data: Vec<(std::fs::File, memmap2::Mmap)>,
    /// Maps (node_id, weight_name) → tensor location in a shard.
    tensor_map: HashMap<(usize, String), TensorLocation>,
    /// Maps node_id → bias location in a shard.
    bias_map: HashMap<usize, BiasLocation>,
    /// Cached quantized weights (populated on-demand, evictable).
    cache: HashMap<(usize, String), M31Matrix>,
    /// Cached biases.
    bias_cache: HashMap<usize, Vec<M31>>,
    /// Quantization strategy.
    strategy: QuantStrategy,
}

impl LazyWeights {
    /// Open a single SafeTensors file and build the tensor index.
    ///
    /// No weights are quantized at this point — only the index mapping
    /// graph node IDs to tensor names/locations is built.
    pub fn open(
        path: &Path,
        graph: &ComputationGraph,
        strategy: QuantStrategy,
    ) -> Result<Self, WeightError> {
        let file = std::fs::File::open(path)
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| WeightError::IoError(e.to_string()))?;

        let mut lazy = Self {
            shard_data: vec![(file, mmap)],
            tensor_map: HashMap::new(),
            bias_map: HashMap::new(),
            cache: HashMap::new(),
            bias_cache: HashMap::new(),
            strategy,
        };
        lazy.build_index(graph)?;
        Ok(lazy)
    }

    /// Open multiple SafeTensors shard files and build the tensor index.
    pub fn open_sharded(
        paths: &[std::path::PathBuf],
        graph: &ComputationGraph,
        strategy: QuantStrategy,
    ) -> Result<Self, WeightError> {
        let mut shard_data = Vec::with_capacity(paths.len());
        for path in paths {
            let file = std::fs::File::open(path)
                .map_err(|e| WeightError::IoError(e.to_string()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| WeightError::IoError(e.to_string()))?;
            shard_data.push((file, mmap));
        }

        let mut lazy = Self {
            shard_data,
            tensor_map: HashMap::new(),
            bias_map: HashMap::new(),
            cache: HashMap::new(),
            bias_cache: HashMap::new(),
            strategy,
        };
        lazy.build_index(graph)?;
        Ok(lazy)
    }

    /// Build the tensor index by scanning all shards for tensors matching graph nodes.
    fn build_index(&mut self, graph: &ComputationGraph) -> Result<(), WeightError> {
        for (idx, node) in graph.nodes.iter().enumerate() {
            if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
                let tensor_names = [
                    format!("weight.{idx}"),
                    format!("layers.{idx}.weight"),
                    format!("model.layers.{idx}.weight"),
                ];

                'weight: for name in &tensor_names {
                    for (shard_idx, (_file, mmap)) in self.shard_data.iter().enumerate() {
                        let tensors = safetensors::SafeTensors::deserialize(mmap)
                            .map_err(|e| WeightError::IoError(e.to_string()))?;
                        if tensors.tensor(name).is_ok() {
                            self.tensor_map.insert(
                                (idx, String::new()),
                                TensorLocation {
                                    shard_idx,
                                    tensor_name: name.clone(),
                                    rows: *k,
                                    cols: *n,
                                },
                            );
                            break 'weight;
                        }
                    }
                }

                let bias_names = [
                    format!("bias.{idx}"),
                    format!("layers.{idx}.bias"),
                ];
                'bias: for name in &bias_names {
                    for (shard_idx, (_file, mmap)) in self.shard_data.iter().enumerate() {
                        let tensors = safetensors::SafeTensors::deserialize(mmap)
                            .map_err(|e| WeightError::IoError(e.to_string()))?;
                        if let Ok(tensor) = tensors.tensor(name) {
                            let byte_len = tensor.data().len();
                            let elem_size = match tensor.dtype() {
                                safetensors::Dtype::F32 => 4,
                                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                                _ => 4,
                            };
                            self.bias_map.insert(
                                idx,
                                BiasLocation {
                                    shard_idx,
                                    tensor_name: name.clone(),
                                    len: byte_len / elem_size,
                                },
                            );
                            break 'bias;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Load (quantize) a single layer's weights from the mmap'd data into cache.
    ///
    /// If already cached, this is a no-op.
    pub fn load_layer(&mut self, node_id: usize) -> Result<(), WeightError> {
        // Load unnamed weight
        if let Some(loc) = self.tensor_map.get(&(node_id, String::new())).cloned() {
            if !self.cache.contains_key(&(node_id, String::new())) {
                let matrix = self.quantize_tensor(&loc)?;
                self.cache.insert((node_id, String::new()), matrix);
            }
        }

        // Load bias
        if let Some(bloc) = self.bias_map.get(&node_id).cloned() {
            if !self.bias_cache.contains_key(&node_id) {
                let bias = self.quantize_bias(&bloc)?;
                self.bias_cache.insert(node_id, bias);
            }
        }

        Ok(())
    }

    /// Read raw tensor bytes from mmap and quantize to M31Matrix.
    fn quantize_tensor(&self, loc: &TensorLocation) -> Result<M31Matrix, WeightError> {
        let (_file, mmap) = &self.shard_data[loc.shard_idx];
        let tensors = safetensors::SafeTensors::deserialize(mmap)
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        let tensor = tensors.tensor(&loc.tensor_name)
            .map_err(|_| WeightError::MissingTensor(loc.tensor_name.clone()))?;
        let data = tensor_to_f32(tensor.data(), tensor.dtype());
        let (matrix, _params) = quantize_weight_matrix(&data, loc.rows, loc.cols, self.strategy);
        Ok(matrix)
    }

    /// Read raw bias bytes from mmap and quantize to Vec<M31>.
    fn quantize_bias(&self, loc: &BiasLocation) -> Result<Vec<M31>, WeightError> {
        let (_file, mmap) = &self.shard_data[loc.shard_idx];
        let tensors = safetensors::SafeTensors::deserialize(mmap)
            .map_err(|e| WeightError::IoError(e.to_string()))?;
        let tensor = tensors.tensor(&loc.tensor_name)
            .map_err(|_| WeightError::MissingTensor(loc.tensor_name.clone()))?;
        let data = tensor_to_f32(tensor.data(), tensor.dtype());
        let (bias, _params) = quantize_bias_vector(&data, self.strategy);
        Ok(bias)
    }

    /// Evict a layer's cached weights, freeing memory.
    pub fn evict_layer(&mut self, node_id: usize) {
        self.cache.remove(&(node_id, String::new()));
        self.bias_cache.remove(&node_id);
    }

    /// Get the unnamed weight for a node, auto-loading from mmap if not cached.
    pub fn get_weight(&mut self, node_id: usize) -> Option<&M31Matrix> {
        if !self.cache.contains_key(&(node_id, String::new())) {
            if self.tensor_map.contains_key(&(node_id, String::new())) {
                let _ = self.load_layer(node_id);
            }
        }
        self.cache.get(&(node_id, String::new()))
    }

    /// Get a named weight for a node (not yet used — placeholder for attention).
    pub fn get_named_weight(&mut self, node_id: usize, name: &str) -> Option<&M31Matrix> {
        let key = (node_id, name.to_string());
        if !self.cache.contains_key(&key) {
            if let Some(loc) = self.tensor_map.get(&key).cloned() {
                if let Ok(matrix) = self.quantize_tensor(&loc) {
                    self.cache.insert(key.clone(), matrix);
                }
            }
        }
        self.cache.get(&key)
    }

    /// Get the bias for a node, auto-loading from mmap if not cached.
    pub fn get_bias(&mut self, node_id: usize) -> Option<&Vec<M31>> {
        if !self.bias_cache.contains_key(&node_id) {
            if self.bias_map.contains_key(&node_id) {
                let _ = self.load_layer(node_id);
            }
        }
        self.bias_cache.get(&node_id)
    }

    /// Snapshot all weights into a `GraphWeights` (loads everything — for compatibility).
    pub fn to_graph_weights(&mut self) -> GraphWeights {
        let mut gw = GraphWeights::new();

        // Load all tensors
        let node_ids: Vec<usize> = self.tensor_map.keys()
            .filter(|(_, name)| name.is_empty())
            .map(|(id, _)| *id)
            .collect();
        for node_id in &node_ids {
            let _ = self.load_layer(*node_id);
        }

        // Copy into GraphWeights
        for ((node_id, name), matrix) in &self.cache {
            if name.is_empty() {
                gw.add_weight(*node_id, matrix.clone());
            } else {
                gw.add_named_weight(*node_id, name, matrix.clone());
            }
        }
        for (node_id, bias) in &self.bias_cache {
            gw.add_bias(*node_id, bias.clone());
        }

        gw
    }

    /// Return the node IDs of layers currently in the weight cache.
    pub fn cached_layers(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.cache.keys()
            .filter(|(_, name)| name.is_empty())
            .map(|(id, _)| *id)
            .collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// Estimated memory (bytes) used by the current M31 cache.
    ///
    /// Each M31 is 4 bytes (u32 internally).
    pub fn cache_memory_bytes(&self) -> usize {
        let weight_bytes: usize = self.cache.values()
            .map(|m| m.data.len() * 4)
            .sum();
        let bias_bytes: usize = self.bias_cache.values()
            .map(|b| b.len() * 4)
            .sum();
        weight_bytes + bias_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_to_f32() {
        // 1.0 in f16 = 0x3C00
        let val = half_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);

        // 0.0 in f16
        let val = half_to_f32(0x0000);
        assert_eq!(val, 0.0);

        // -1.0 in f16 = 0xBC00
        let val = half_to_f32(0xBC00);
        assert!((val + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32() {
        // 1.0 in bf16 = 0x3F80
        let val = bf16_to_f32(0x3F80);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_to_f32_roundtrip() {
        let original: Vec<f32> = vec![1.0, -2.5, 0.0, 100.0];
        let bytes: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();
        let result = tensor_to_f32(&bytes, safetensors::Dtype::F32);
        assert_eq!(result, original);
    }

    #[test]
    fn test_load_weights_roundtrip() {
        use std::collections::HashMap;

        // Build a simple graph: input(1,4) → matmul → output(1,2)
        let mut graph = ComputationGraph::new((1, 4));
        graph.add_node(
            GraphOp::MatMul { dims: (1, 4, 2) },
            vec![],
            (1, 2),
        );

        // Create a safetensors file with the weight
        let weight_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let weight_bytes: Vec<u8> = weight_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors_map = HashMap::new();
        tensors_map.insert(
            "weight.0".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![4, 2],
                &weight_bytes,
            ).unwrap(),
        );

        let serialized = safetensors::serialize(&tensors_map, &None).unwrap();

        // Write to temp file
        let tmp = std::env::temp_dir().join("test_weights.safetensors");
        std::fs::write(&tmp, &serialized).unwrap();

        // Load and verify
        let names = list_tensors(&tmp).unwrap();
        assert!(names.contains(&"weight.0".to_string()));

        let weights = load_weights(&tmp, &graph, QuantStrategy::Direct).unwrap();
        assert!(weights.get_weight(0).is_some());
        let w = weights.get_weight(0).unwrap();
        assert_eq!(w.rows, 4);
        assert_eq!(w.cols, 2);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_load_weights_sharded() {
        use std::collections::HashMap;

        // Build a graph with 2 MatMul nodes
        let mut graph = ComputationGraph::new((1, 4));
        graph.add_node(GraphOp::MatMul { dims: (1, 4, 2) }, vec![], (1, 2));
        graph.add_node(GraphOp::MatMul { dims: (1, 2, 3) }, vec![0], (1, 3));

        let tmp_dir = std::env::temp_dir().join("stwo_ml_shard_test");
        std::fs::create_dir_all(&tmp_dir).unwrap();

        // Shard 0: contains weight.0
        let w0_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let w0_bytes: Vec<u8> = w0_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tensors0 = HashMap::new();
        tensors0.insert(
            "weight.0".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32, vec![4, 2], &w0_bytes,
            ).unwrap(),
        );
        let shard0 = safetensors::serialize(&tensors0, &None).unwrap();
        let shard0_path = tmp_dir.join("model-00001-of-00002.safetensors");
        std::fs::write(&shard0_path, &shard0).unwrap();

        // Shard 1: contains weight.1
        let w1_data: Vec<f32> = (0..6).map(|i| (i + 10) as f32).collect();
        let w1_bytes: Vec<u8> = w1_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tensors1 = HashMap::new();
        tensors1.insert(
            "weight.1".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32, vec![2, 3], &w1_bytes,
            ).unwrap(),
        );
        let shard1 = safetensors::serialize(&tensors1, &None).unwrap();
        let shard1_path = tmp_dir.join("model-00002-of-00002.safetensors");
        std::fs::write(&shard1_path, &shard1).unwrap();

        // Load sharded
        let shard_paths = vec![shard0_path.clone(), shard1_path.clone()];
        let weights = load_weights_sharded(&shard_paths, &graph, QuantStrategy::Direct).unwrap();

        assert!(weights.get_weight(0).is_some(), "weight.0 from shard 0");
        assert!(weights.get_weight(1).is_some(), "weight.1 from shard 1");
        assert_eq!(weights.get_weight(0).unwrap().rows, 4);
        assert_eq!(weights.get_weight(0).unwrap().cols, 2);
        assert_eq!(weights.get_weight(1).unwrap().rows, 2);
        assert_eq!(weights.get_weight(1).unwrap().cols, 3);

        // List tensors across shards
        let all_tensors = list_tensors_sharded(&shard_paths).unwrap();
        assert_eq!(all_tensors.len(), 2);

        // Cleanup
        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    #[test]
    fn test_discover_shards() {
        let tmp_dir = std::env::temp_dir().join("stwo_ml_discover_test");
        std::fs::create_dir_all(&tmp_dir).unwrap();

        // Create fake shard files
        std::fs::write(tmp_dir.join("model-00001-of-00002.safetensors"), &[0u8]).unwrap();
        std::fs::write(tmp_dir.join("model-00002-of-00002.safetensors"), &[0u8]).unwrap();
        std::fs::write(tmp_dir.join("other_file.txt"), &[0u8]).unwrap();

        let shards = discover_shards(&tmp_dir, "model").unwrap();
        assert_eq!(shards.len(), 2);
        // Verify sorting
        assert!(shards[0].to_str().unwrap().contains("00001"));
        assert!(shards[1].to_str().unwrap().contains("00002"));

        // Cleanup
        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    /// Helper: create a temp SafeTensors file with weights for a graph.
    fn create_test_safetensors(
        graph: &ComputationGraph,
    ) -> (std::path::PathBuf, Vec<Vec<f32>>) {
        let tmp = std::env::temp_dir().join(format!(
            "stwo_ml_lazy_test_{}.safetensors",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mut all_data: Vec<Vec<f32>> = Vec::new();
        let mut byte_vecs: Vec<Vec<u8>> = Vec::new();

        for (idx, node) in graph.nodes.iter().enumerate() {
            if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
                let data: Vec<f32> = (0..(*k * *n))
                    .map(|i| (i as f32 + idx as f32 * 100.0) * 0.1)
                    .collect();
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                all_data.push(data);
                byte_vecs.push(bytes);
            }
        }

        // Build safetensors tensor views referencing the byte vecs
        let mut tensors_map = std::collections::HashMap::new();
        let mut weight_idx = 0;
        for (idx, node) in graph.nodes.iter().enumerate() {
            if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
                let name = format!("weight.{idx}");
                tensors_map.insert(
                    name,
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![*k, *n],
                        &byte_vecs[weight_idx],
                    ).unwrap(),
                );
                weight_idx += 1;
            }
        }

        let serialized = safetensors::serialize(&tensors_map, &None).unwrap();
        std::fs::write(&tmp, &serialized).unwrap();

        (tmp, all_data)
    }

    #[test]
    fn test_lazy_open_and_load_single_layer() {
        let mut graph = ComputationGraph::new((1, 4));
        graph.add_node(GraphOp::MatMul { dims: (1, 4, 2) }, vec![], (1, 2));

        let (tmp, _) = create_test_safetensors(&graph);
        let mut lazy = LazyWeights::open(&tmp, &graph, QuantStrategy::Direct).unwrap();

        // Nothing cached yet
        assert!(lazy.cached_layers().is_empty());
        assert_eq!(lazy.cache_memory_bytes(), 0);

        // Load layer 0
        lazy.load_layer(0).unwrap();
        assert_eq!(lazy.cached_layers(), vec![0]);
        assert!(lazy.cache_memory_bytes() > 0);

        // Weight is available
        let w = lazy.cache.get(&(0, String::new())).unwrap();
        assert_eq!(w.rows, 4);
        assert_eq!(w.cols, 2);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_lazy_evict_frees_memory() {
        let mut graph = ComputationGraph::new((1, 4));
        graph.add_node(GraphOp::MatMul { dims: (1, 4, 2) }, vec![], (1, 2));
        graph.add_node(GraphOp::MatMul { dims: (1, 2, 3) }, vec![0], (1, 3));

        let (tmp, _) = create_test_safetensors(&graph);
        let mut lazy = LazyWeights::open(&tmp, &graph, QuantStrategy::Direct).unwrap();

        lazy.load_layer(0).unwrap();
        lazy.load_layer(1).unwrap();

        let mem_both = lazy.cache_memory_bytes();
        assert!(mem_both > 0);
        assert_eq!(lazy.cached_layers().len(), 2);

        // Evict layer 0
        lazy.evict_layer(0);
        let mem_after = lazy.cache_memory_bytes();
        assert!(mem_after < mem_both, "memory should decrease after eviction");
        assert_eq!(lazy.cached_layers(), vec![1]);

        // Evict layer 1
        lazy.evict_layer(1);
        assert_eq!(lazy.cache_memory_bytes(), 0);
        assert!(lazy.cached_layers().is_empty());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_lazy_auto_load_on_get_weight() {
        let mut graph = ComputationGraph::new((1, 4));
        graph.add_node(GraphOp::MatMul { dims: (1, 4, 2) }, vec![], (1, 2));

        let (tmp, _) = create_test_safetensors(&graph);
        let mut lazy = LazyWeights::open(&tmp, &graph, QuantStrategy::Direct).unwrap();

        // get_weight should auto-load
        assert!(lazy.cached_layers().is_empty());
        let w = lazy.get_weight(0);
        assert!(w.is_some());
        assert_eq!(w.unwrap().rows, 4);
        assert_eq!(lazy.cached_layers(), vec![0]);

        // Non-existent node returns None
        let w2 = lazy.get_weight(99);
        assert!(w2.is_none());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_lazy_to_graph_weights_compat() {
        let mut graph = ComputationGraph::new((1, 4));
        graph.add_node(GraphOp::MatMul { dims: (1, 4, 2) }, vec![], (1, 2));
        graph.add_node(GraphOp::MatMul { dims: (1, 2, 3) }, vec![0], (1, 3));

        let (tmp, _) = create_test_safetensors(&graph);

        // Eager load
        let eager = load_weights(&tmp, &graph, QuantStrategy::Direct).unwrap();

        // Lazy → to_graph_weights
        let mut lazy = LazyWeights::open(&tmp, &graph, QuantStrategy::Direct).unwrap();
        let snapshot = lazy.to_graph_weights();

        // Both should produce the same weight data
        for node_id in [0, 1] {
            let ew = eager.get_weight(node_id).unwrap();
            let sw = snapshot.get_weight(node_id).unwrap();
            assert_eq!(ew.rows, sw.rows);
            assert_eq!(ew.cols, sw.cols);
            assert_eq!(ew.data, sw.data, "weight data must match for node {node_id}");
        }

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_lazy_sharded_load() {
        let mut graph = ComputationGraph::new((1, 4));
        graph.add_node(GraphOp::MatMul { dims: (1, 4, 2) }, vec![], (1, 2));
        graph.add_node(GraphOp::MatMul { dims: (1, 2, 3) }, vec![0], (1, 3));

        let tmp_dir = std::env::temp_dir().join(format!(
            "stwo_ml_lazy_shard_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        // Shard 0: weight.0
        let w0_data: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let w0_bytes: Vec<u8> = w0_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut t0 = std::collections::HashMap::new();
        t0.insert(
            "weight.0".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32, vec![4, 2], &w0_bytes,
            ).unwrap(),
        );
        let s0_path = tmp_dir.join("shard-00001.safetensors");
        std::fs::write(&s0_path, safetensors::serialize(&t0, &None).unwrap()).unwrap();

        // Shard 1: weight.1
        let w1_data: Vec<f32> = (0..6).map(|i| (i + 10) as f32).collect();
        let w1_bytes: Vec<u8> = w1_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut t1 = std::collections::HashMap::new();
        t1.insert(
            "weight.1".to_string(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32, vec![2, 3], &w1_bytes,
            ).unwrap(),
        );
        let s1_path = tmp_dir.join("shard-00002.safetensors");
        std::fs::write(&s1_path, safetensors::serialize(&t1, &None).unwrap()).unwrap();

        let shard_paths = vec![s0_path, s1_path];
        let mut lazy = LazyWeights::open_sharded(
            &shard_paths, &graph, QuantStrategy::Direct,
        ).unwrap();

        // Load layer 0 from shard 0
        lazy.load_layer(0).unwrap();
        let w0 = lazy.cache.get(&(0, String::new())).unwrap();
        assert_eq!(w0.rows, 4);
        assert_eq!(w0.cols, 2);

        // Load layer 1 from shard 1
        lazy.load_layer(1).unwrap();
        let w1 = lazy.cache.get(&(1, String::new())).unwrap();
        assert_eq!(w1.rows, 2);
        assert_eq!(w1.cols, 3);

        // Evict and verify
        lazy.evict_layer(0);
        assert_eq!(lazy.cached_layers(), vec![1]);

        std::fs::remove_dir_all(&tmp_dir).ok();
    }
}
