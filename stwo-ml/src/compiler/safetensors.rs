//! SafeTensors weight loading.
//!
//! Loads model weights from SafeTensors format files using memory-mapped I/O
//! for efficient handling of large models. Weights are quantized to M31
//! during loading.

use std::path::Path;

use crate::compiler::graph::{ComputationGraph, GraphWeights, GraphOp};
use crate::compiler::quantize_weights::{quantize_weight_matrix, quantize_bias_vector, WeightError};
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
}
