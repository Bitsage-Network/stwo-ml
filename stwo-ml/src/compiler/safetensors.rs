//! SafeTensors weight loading.
//!
//! Loads model weights from SafeTensors format files using memory-mapped I/O
//! for efficient handling of large models. Weights are quantized to M31
//! during loading.

use std::path::Path;

use stwo::core::fields::m31::M31;

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
pub fn tensor_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
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
        safetensors::Dtype::I8 => {
            data.iter().map(|&b| b as i8 as f32).collect()
        }
        safetensors::Dtype::U8 => {
            data.iter().map(|&b| b as f32).collect()
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

/// Size in bytes of a single element for a SafeTensors dtype.
pub fn dtype_byte_size(dtype: safetensors::Dtype) -> usize {
    match dtype {
        safetensors::Dtype::F32 => 4,
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
        safetensors::Dtype::I8 | safetensors::Dtype::U8 | safetensors::Dtype::BOOL => 1,
        safetensors::Dtype::I16 | safetensors::Dtype::U16 => 2,
        safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
        _ => 4, // fallback
    }
}

/// Convert a single element's raw bytes to f32, given its dtype.
///
/// The `bytes` slice must have exactly `dtype_byte_size(dtype)` bytes.
pub fn bytes_to_f32_single(bytes: &[u8], dtype: safetensors::Dtype) -> f32 {
    match dtype {
        safetensors::Dtype::F32 => {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        }
        safetensors::Dtype::F16 => {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            half_to_f32(bits)
        }
        safetensors::Dtype::BF16 => {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            bf16_to_f32(bits)
        }
        safetensors::Dtype::I8 => bytes[0] as i8 as f32,
        safetensors::Dtype::U8 => bytes[0] as f32,
        _ => {
            if bytes.len() >= 4 {
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else {
                0.0
            }
        }
    }
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

/// Unpack packed INT4 data: each byte holds 2 values (lo nibble, hi nibble).
///
/// GPTQ/AWQ models store quantized weights as packed INT4 — 2 values per byte.
/// Returns f32 values in [0, 15].
pub fn unpack_int4(data: &[u8]) -> Vec<f32> {
    data.iter().flat_map(|&byte| {
        let lo = (byte & 0x0F) as f32;
        let hi = ((byte >> 4) & 0x0F) as f32;
        [lo, hi]
    }).collect()
}

/// Unpack packed INT4 data directly to M31 field elements.
///
/// Each byte yields 2 M31 values in [0, 15]. No f32 intermediate.
pub fn unpack_int4_to_m31(data: &[u8]) -> Vec<M31> {
    data.iter().flat_map(|&byte| {
        [
            M31::from((byte & 0x0F) as u32),
            M31::from(((byte >> 4) & 0x0F) as u32),
        ]
    }).collect()
}

/// Load weights from a SafeTensors file, preserving native INT8/INT4 quantization.
///
/// Returns weights as M31 matrices alongside per-layer QuantParams detected from
/// companion `{name}_scale` and `{name}_zero_point` tensors.
///
/// Supports:
/// - Standard I8/U8 tensors (loaded as M31 directly)
/// - GPTQ format: `qweight` (packed INT4), `scales`, `qzeros`
/// - Standard F32/F16/BF16 (loaded via `tensor_to_f32` + Direct quantization)
pub fn load_weights_quantized(
    path: &Path,
    graph: &ComputationGraph,
) -> Result<(GraphWeights, Vec<(usize, crate::gadgets::quantize::QuantParams)>), WeightError> {
    use crate::gadgets::quantize::{QuantParams, QuantStrategy};

    let file = std::fs::File::open(path)
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let tensors = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| WeightError::IoError(e.to_string()))?;

    let mut weights = GraphWeights::new();
    let mut quant_params: Vec<(usize, QuantParams)> = Vec::new();

    // Check for GPTQ pattern: look for qweight/scales/qzeros tensors
    let has_gptq = tensors.names().iter().any(|n| n.contains("qweight"));

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let GraphOp::MatMul { dims: (_m, k, n) } = &node.op {
            // Try GPTQ packed INT4 format
            if has_gptq {
                let qweight_names = [
                    format!("layers.{idx}.qweight"),
                    format!("model.layers.{idx}.qweight"),
                ];
                for name in &qweight_names {
                    if let Ok(tensor) = tensors.tensor(name) {
                        let data = unpack_int4_to_m31(tensor.data());
                        let matrix = crate::components::matmul::M31Matrix {
                            rows: *k,
                            cols: *n,
                            data: if data.len() >= k * n { data[..k * n].to_vec() } else { data },
                        };
                        weights.add_weight(idx, matrix);

                        // Read scale from companion tensor
                        let scale_name = name.replace("qweight", "scales");
                        let scale = if let Ok(st) = tensors.tensor(&scale_name) {
                            let sf = tensor_to_f32(st.data(), st.dtype());
                            sf.first().copied().unwrap_or(1.0) as f64
                        } else {
                            1.0
                        };

                        // Read zero point from companion tensor
                        let zp_name = name.replace("qweight", "qzeros");
                        let zero_point = if let Ok(zt) = tensors.tensor(&zp_name) {
                            let zf = tensor_to_f32(zt.data(), zt.dtype());
                            zf.first().copied().unwrap_or(0.0) as i32
                        } else {
                            0
                        };

                        quant_params.push((idx, QuantParams {
                            strategy: QuantStrategy::Asymmetric4,
                            scale,
                            zero_point,
                            bits: 4,
                        }));
                        break;
                    }
                }
                continue;
            }

            // Try standard tensor names
            let tensor_names = [
                format!("weight.{idx}"),
                format!("layers.{idx}.weight"),
                format!("model.layers.{idx}.weight"),
            ];

            for name in &tensor_names {
                if let Ok(tensor) = tensors.tensor(name) {
                    match tensor.dtype() {
                        safetensors::Dtype::I8 | safetensors::Dtype::U8 => {
                            // Load as M31 directly (no f32 intermediate)
                            let data: Vec<M31> = match tensor.dtype() {
                                safetensors::Dtype::I8 => tensor.data().iter()
                                    .map(|&b| M31::from((b as i8 as i32 + 128) as u32))
                                    .collect(),
                                _ => tensor.data().iter()
                                    .map(|&b| M31::from(b as u32))
                                    .collect(),
                            };
                            let matrix = crate::components::matmul::M31Matrix {
                                rows: *k,
                                cols: *n,
                                data: if data.len() >= k * n { data[..k * n].to_vec() } else { data },
                            };
                            weights.add_weight(idx, matrix);

                            // Read companion scale/zero_point
                            let scale_name = format!("{name}_scale");
                            let scale = if let Ok(st) = tensors.tensor(&scale_name) {
                                let sf = tensor_to_f32(st.data(), st.dtype());
                                sf.first().copied().unwrap_or(1.0) as f64
                            } else {
                                1.0
                            };

                            let zp_name = format!("{name}_zero_point");
                            let zero_point = if let Ok(zt) = tensors.tensor(&zp_name) {
                                let zf = tensor_to_f32(zt.data(), zt.dtype());
                                zf.first().copied().unwrap_or(0.0) as i32
                            } else if tensor.dtype() == safetensors::Dtype::I8 {
                                128
                            } else {
                                0
                            };

                            let (strategy, bits) = if tensor.dtype() == safetensors::Dtype::I8 {
                                (QuantStrategy::Symmetric8, 8)
                            } else {
                                (QuantStrategy::Asymmetric8, 8)
                            };

                            quant_params.push((idx, QuantParams {
                                strategy,
                                scale,
                                zero_point,
                                bits,
                            }));
                            break;
                        }
                        _ => {
                            // F32/F16/BF16 — load via f32 path with Direct quantization
                            let data = tensor_to_f32(tensor.data(), tensor.dtype());
                            let (matrix, _) = crate::compiler::quantize_weights::quantize_weight_matrix(
                                &data, *k, *n, QuantStrategy::Direct,
                            );
                            weights.add_weight(idx, matrix);
                            break;
                        }
                    }
                }
            }
        }
    }

    Ok((weights, quant_params))
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
    fn test_unpack_int4() {
        // 0xAB = lo=0xB=11, hi=0xA=10
        let data = vec![0xAB, 0x34];
        let unpacked = super::unpack_int4(&data);
        assert_eq!(unpacked.len(), 4);
        assert_eq!(unpacked[0], 11.0); // 0xB
        assert_eq!(unpacked[1], 10.0); // 0xA
        assert_eq!(unpacked[2], 4.0);  // 0x4
        assert_eq!(unpacked[3], 3.0);  // 0x3
    }

    #[test]
    fn test_unpack_int4_to_m31() {
        let data = vec![0x12, 0xFF];
        let unpacked = super::unpack_int4_to_m31(&data);
        assert_eq!(unpacked.len(), 4);
        assert_eq!(unpacked[0], M31::from(2));  // 0x2
        assert_eq!(unpacked[1], M31::from(1));  // 0x1
        assert_eq!(unpacked[2], M31::from(15)); // 0xF
        assert_eq!(unpacked[3], M31::from(15)); // 0xF
    }

    #[test]
    fn test_tensor_to_f32_i8() {
        let data: Vec<u8> = vec![0, 127, 128, 255]; // as i8: 0, 127, -128, -1
        let result = super::tensor_to_f32(&data, safetensors::Dtype::I8);
        assert_eq!(result, vec![0.0, 127.0, -128.0, -1.0]);
    }

    #[test]
    fn test_tensor_to_f32_u8() {
        let data: Vec<u8> = vec![0, 127, 128, 255];
        let result = super::tensor_to_f32(&data, safetensors::Dtype::U8);
        assert_eq!(result, vec![0.0, 127.0, 128.0, 255.0]);
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
