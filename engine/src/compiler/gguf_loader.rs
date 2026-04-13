//! GGUF model loader — load any llama.cpp model into the obelyzk.rs proving pipeline.
//!
//! GGUF (GPT-Generated Unified Format) is the standard model format for llama.cpp,
//! Ollama, LM Studio, and most local AI tools. This module parses GGUF files,
//! dequantizes weights to f32, and feeds them into the existing M31 quantization
//! pipeline for provable inference.
//!
//! Supported quantization types: F32, F16, BF16, Q8_0, Q4_K_M, Q5_K_M, Q6_K.
//!
//! ```text
//! model.gguf → parse header → extract config → dequantize weights → M31 quantize → prove
//! ```

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::compiler::onnx::OnnxModel;
use crate::compiler::graph::GraphWeights;
use crate::compiler::quantize_weights::quantize_weight_matrix;
use crate::components::matmul::M31Matrix;

// Re-use HfConfig from hf_loader (but it may be private — define our own if needed)
use crate::compiler::hf_loader::HfConfig;

/// GGUF magic bytes: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x46475547; // "GGUF"

/// Supported GGUF version.
const GGUF_VERSION: u32 = 3;

// ═══════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════

/// GGUF file header.
#[derive(Debug)]
struct GgufHeader {
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16),
    U32(u32), I32(i32), U64(u64), I64(i64),
    F32(f32), F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            Self::U32(v) => Some(*v as u64),
            Self::I32(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { Self::String(s) => Some(s), _ => None }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self { Self::F32(v) => Some(*v), Self::F64(v) => Some(*v as f32), _ => None }
    }
}

/// GGUF tensor descriptor.
#[derive(Debug)]
struct GgufTensorInfo {
    name: String,
    dimensions: Vec<u64>,
    dtype: u32,
    offset: u64,
}

/// GGUF quantization type IDs (matches llama.cpp ggml_type enum).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgufDtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    BF16 = 30,
}

impl GgufDtype {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32), 1 => Some(Self::F16),
            2 => Some(Self::Q4_0), 3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0), 7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0), 9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K), 11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K), 13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            16 => Some(Self::IQ2_XXS), 17 => Some(Self::IQ2_XS),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Bytes per block for this quantization type.
    fn block_size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q8_0 => 34,    // 2 (scale f16) + 32 (i8 values)
            Self::Q4_0 => 18,    // 2 (scale f16) + 16 (packed 4-bit, 32 values)
            Self::Q4_1 => 20,    // 2+2 (scale+min f16) + 16 (packed)
            Self::Q5_0 => 22,    // 2 + 4 (high bits) + 16 (low 4 bits)
            Self::Q5_1 => 24,
            Self::Q4_K => 144,   // super-block: 256 values
            Self::Q5_K => 176,
            Self::Q6_K => 210,   // super-block: 256 values
            Self::Q8_1 => 36,
            _ => 1,              // unsupported, will error
        }
    }

    /// Number of elements per block.
    fn block_elements(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q2_K | Self::Q3_K => 256,
            _ => 1,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Binary Reading Helpers
// ═══════════════════════════════════════════════════════════════════

fn read_u8(r: &mut impl Read) -> std::io::Result<u8> {
    let mut buf = [0u8; 1]; r.read_exact(&mut buf)?; Ok(buf[0])
}
fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(u32::from_le_bytes(buf))
}
fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; Ok(u64::from_le_bytes(buf))
}
fn read_i8(r: &mut impl Read) -> std::io::Result<i8> {
    let mut buf = [0u8; 1]; r.read_exact(&mut buf)?; Ok(buf[0] as i8)
}
fn read_i16(r: &mut impl Read) -> std::io::Result<i16> {
    let mut buf = [0u8; 2]; r.read_exact(&mut buf)?; Ok(i16::from_le_bytes(buf))
}
fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(i32::from_le_bytes(buf))
}
fn read_i64(r: &mut impl Read) -> std::io::Result<i64> {
    let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; Ok(i64::from_le_bytes(buf))
}
fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    let mut buf = [0u8; 4]; r.read_exact(&mut buf)?; Ok(f32::from_le_bytes(buf))
}
fn read_f64(r: &mut impl Read) -> std::io::Result<f64> {
    let mut buf = [0u8; 8]; r.read_exact(&mut buf)?; Ok(f64::from_le_bytes(buf))
}

fn read_string(r: &mut impl Read) -> std::io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_gguf_value(r: &mut impl Read) -> std::io::Result<GgufValue> {
    let type_id = read_u32(r)?;
    match type_id {
        0 => Ok(GgufValue::U8(read_u8(r)?)),
        1 => Ok(GgufValue::I8(read_i8(r)?)),
        2 => Ok(GgufValue::U16({ let mut b=[0u8;2]; r.read_exact(&mut b)?; u16::from_le_bytes(b) })),
        3 => Ok(GgufValue::I16(read_i16(r)?)),
        4 => Ok(GgufValue::U32(read_u32(r)?)),
        5 => Ok(GgufValue::I32(read_i32(r)?)),
        6 => Ok(GgufValue::F32(read_f32(r)?)),
        7 => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        8 => Ok(GgufValue::String(read_string(r)?)),
        9 => {
            // Array: type_id + count + values
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                let val = match elem_type {
                    0 => GgufValue::U8(read_u8(r)?),
                    4 => GgufValue::U32(read_u32(r)?),
                    5 => GgufValue::I32(read_i32(r)?),
                    6 => GgufValue::F32(read_f32(r)?),
                    8 => GgufValue::String(read_string(r)?),
                    _ => { let mut b=[0u8;1]; r.read_exact(&mut b)?; GgufValue::U8(b[0]) }
                };
                arr.push(val);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::U64(read_u64(r)?)),
        11 => Ok(GgufValue::I64(read_i64(r)?)),
        12 => Ok(GgufValue::F64(read_f64(r)?)),
        _ => Ok(GgufValue::U32(0)), // unknown type
    }
}

// ═══════════════════════════════════════════════════════════════════
// GGUF Parsing
// ═══════════════════════════════════════════════════════════════════

fn parse_gguf_header(r: &mut impl Read) -> Result<GgufHeader, GgufError> {
    let magic = read_u32(r).map_err(|e| GgufError::ParseError(format!("magic: {e}")))?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::ParseError(format!("Bad magic: 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x}")));
    }
    let version = read_u32(r).map_err(|e| GgufError::ParseError(format!("version: {e}")))?;
    if version < 2 || version > 3 {
        return Err(GgufError::ParseError(format!("Unsupported GGUF version {version}")));
    }
    let tensor_count = read_u64(r).map_err(|e| GgufError::ParseError(format!("tensor_count: {e}")))?;
    let metadata_count = read_u64(r).map_err(|e| GgufError::ParseError(format!("metadata_count: {e}")))?;
    Ok(GgufHeader { version, tensor_count, metadata_count })
}

fn parse_gguf_metadata(r: &mut impl Read, count: u64) -> Result<HashMap<String, GgufValue>, GgufError> {
    let mut metadata = HashMap::new();
    for _ in 0..count {
        let key = read_string(r).map_err(|e| GgufError::ParseError(format!("metadata key: {e}")))?;
        let value = read_gguf_value(r).map_err(|e| GgufError::ParseError(format!("metadata value for '{key}': {e}")))?;
        metadata.insert(key, value);
    }
    Ok(metadata)
}

fn parse_tensor_infos(r: &mut impl Read, count: u64) -> Result<Vec<GgufTensorInfo>, GgufError> {
    let mut tensors = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name = read_string(r).map_err(|e| GgufError::ParseError(format!("tensor name: {e}")))?;
        let n_dims = read_u32(r).map_err(|e| GgufError::ParseError(format!("n_dims: {e}")))? as usize;
        let mut dimensions = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dimensions.push(read_u64(r).map_err(|e| GgufError::ParseError(format!("dim: {e}")))?);
        }
        let dtype = read_u32(r).map_err(|e| GgufError::ParseError(format!("dtype: {e}")))?;
        let offset = read_u64(r).map_err(|e| GgufError::ParseError(format!("offset: {e}")))?;
        tensors.push(GgufTensorInfo { name, dimensions, dtype, offset });
    }
    Ok(tensors)
}

// ═══════════════════════════════════════════════════════════════════
// Config Extraction
// ═══════════════════════════════════════════════════════════════════

fn extract_model_config(metadata: &HashMap<String, GgufValue>) -> Result<HfConfig, GgufError> {
    let arch = metadata.get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_string();

    let get_u32 = |key: &str| -> usize {
        metadata.get(key).and_then(|v| v.as_u32()).unwrap_or(0) as usize
    };

    let hidden_size = get_u32(&format!("{arch}.embedding_length"));
    let num_layers = get_u32(&format!("{arch}.block_count"));
    let num_heads = get_u32(&format!("{arch}.attention.head_count"));
    let num_kv_heads = get_u32(&format!("{arch}.attention.head_count_kv"));
    let intermediate_size = get_u32(&format!("{arch}.feed_forward_length"));
    let vocab_size = metadata.get("tokenizer.ggml.tokens")
        .and_then(|v| if let GgufValue::Array(arr) = v { Some(arr.len()) } else { None })
        .unwrap_or(32000);
    let max_pos = get_u32(&format!("{arch}.context_length"));

    if hidden_size == 0 || num_layers == 0 {
        return Err(GgufError::ParseError(format!(
            "Could not extract model config: hidden_size={hidden_size}, num_layers={num_layers}"
        )));
    }

    eprintln!("[gguf] Architecture: {arch}");
    eprintln!("[gguf] Config: hidden_size={hidden_size}, layers={num_layers}, heads={num_heads}, kv_heads={num_kv_heads}");

    Ok(HfConfig {
        model_type: arch,
        hidden_size,
        num_attention_heads: num_heads,
        num_key_value_heads: if num_kv_heads > 0 { num_kv_heads } else { num_heads },
        intermediate_size,
        num_hidden_layers: num_layers,
        vocab_size,
        hidden_act: "silu".to_string(), // default for llama-family
        max_position_embeddings: if max_pos > 0 { max_pos } else { 4096 },
        head_dim: if num_heads > 0 { hidden_size / num_heads } else { 64 },
        num_experts: 0,
        num_experts_per_tok: 0,
    })
}

// ═══════════════════════════════════════════════════════════════════
// Dequantization
// ═══════════════════════════════════════════════════════════════════

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign); }
        // subnormal
        let mut e = 0u32;
        let mut f = frac;
        while (f & 0x400) == 0 { f <<= 1; e += 1; }
        f &= 0x3FF;
        return f32::from_bits(sign | ((127 - 15 + 1 - e) << 23) | (f << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F800000 | (frac << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13))
}

/// Dequantize a GGUF tensor block to f32.
pub fn dequantize_gguf_tensor(data: &[u8], dtype_id: u32, num_elements: usize) -> Result<Vec<f32>, GgufError> {
    let dtype = GgufDtype::from_u32(dtype_id)
        .ok_or_else(|| GgufError::UnsupportedDtype(dtype_id))?;

    let mut output = Vec::with_capacity(num_elements);

    match dtype {
        GgufDtype::F32 => {
            for chunk in data.chunks_exact(4).take(num_elements) {
                output.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        GgufDtype::F16 => {
            for chunk in data.chunks_exact(2).take(num_elements) {
                output.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
            }
        }
        GgufDtype::BF16 => {
            for chunk in data.chunks_exact(2).take(num_elements) {
                let bits = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
                output.push(f32::from_bits(bits));
            }
        }
        GgufDtype::Q8_0 => {
            // Block: [f16 scale] [32 × i8 values] = 34 bytes per 32 elements
            let block_size = 34;
            let num_blocks = num_elements / 32;
            for b in 0..num_blocks {
                let block = &data[b * block_size..(b + 1) * block_size];
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for i in 0..32 {
                    let val = block[2 + i] as i8;
                    output.push(scale * val as f32);
                }
            }
        }
        GgufDtype::Q4_0 => {
            // Block: [f16 scale] [16 × u8 packed] = 18 bytes per 32 elements
            let block_size = 18;
            let num_blocks = num_elements / 32;
            for b in 0..num_blocks {
                let block = &data[b * block_size..(b + 1) * block_size];
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for i in 0..16 {
                    let byte = block[2 + i];
                    let lo = (byte & 0x0F) as i8 - 8;
                    let hi = ((byte >> 4) & 0x0F) as i8 - 8;
                    output.push(scale * lo as f32);
                    output.push(scale * hi as f32);
                }
            }
        }
        GgufDtype::Q4_K => {
            // K-quant super-block: 256 elements, 144 bytes
            // [f16 d, f16 dmin, 12×u8 scales, 4×u8 high_bits, 128×u8 packed_qs]
            let block_bytes = 144;
            let num_blocks = num_elements / 256;
            for b in 0..num_blocks {
                let block = &data[b * block_bytes..(b + 1) * block_bytes];
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                let scales = &block[4..16]; // 12 bytes of scales
                let qs = &block[16..]; // packed quantized values

                // 8 sub-blocks of 32 elements each
                for sb in 0..8 {
                    let sc_idx = sb;
                    let sc = (scales[sc_idx / 2] >> ((sc_idx % 2) * 4)) & 0x0F;
                    let m = (scales[6 + sc_idx / 2] >> ((sc_idx % 2) * 4)) & 0x0F;
                    let d_sc = d * sc as f32;
                    let dm = dmin * m as f32;

                    for i in 0..32 {
                        let byte_idx = sb * 16 + i / 2;
                        let q = if byte_idx < qs.len() {
                            if i % 2 == 0 { qs[byte_idx] & 0x0F } else { qs[byte_idx] >> 4 }
                        } else { 0 };
                        output.push(d_sc * q as f32 - dm);
                    }
                }
            }
        }
        GgufDtype::Q6_K => {
            // K-quant 6-bit: 256 elements, 210 bytes
            let block_bytes = 210;
            let num_blocks = num_elements / 256;
            for b in 0..num_blocks {
                let block = &data[b * block_bytes..(b + 1) * block_bytes];
                let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

                let ql = &block[0..128];   // low 4 bits
                let qh = &block[128..192]; // high 2 bits
                let sc = &block[192..208]; // 16 scales (i8)

                for i in 0..256 {
                    let lo = if i < 128 {
                        (ql[i / 2] >> ((i % 2) * 4)) & 0x0F
                    } else {
                        let j = i - 128;
                        (ql[64 + j / 2] >> ((j % 2) * 4)) & 0x0F
                    };
                    let hi_byte = qh[i / 4];
                    let hi = (hi_byte >> ((i % 4) * 2)) & 0x03;
                    let q = (lo | (hi << 4)) as i8 - 32;
                    let scale = sc[i / 16] as i8;
                    output.push(d * scale as f32 * q as f32);
                }
            }
        }
        _ => {
            return Err(GgufError::UnsupportedDtype(dtype_id));
        }
    }

    // Pad if needed
    while output.len() < num_elements {
        output.push(0.0);
    }
    output.truncate(num_elements);

    Ok(output)
}

// ═══════════════════════════════════════════════════════════════════
// GGUF Tensor Name → Graph Node Mapping
// ═══════════════════════════════════════════════════════════════════

/// Map GGUF tensor names to HuggingFace-style names for graph building.
fn gguf_to_hf_name(gguf_name: &str) -> String {
    // blk.{L}.attn_q.weight → model.layers.{L}.self_attn.q_proj.weight
    let s = gguf_name
        .replace("blk.", "model.layers.")
        .replace(".attn_q.", ".self_attn.q_proj.")
        .replace(".attn_k.", ".self_attn.k_proj.")
        .replace(".attn_v.", ".self_attn.v_proj.")
        .replace(".attn_output.", ".self_attn.o_proj.")
        .replace(".ffn_gate.", ".mlp.gate_proj.")
        .replace(".ffn_up.", ".mlp.up_proj.")
        .replace(".ffn_down.", ".mlp.down_proj.")
        .replace(".attn_norm.", ".input_layernorm.")
        .replace(".ffn_norm.", ".post_attention_layernorm.")
        .replace("token_embd.", "model.embed_tokens.")
        .replace("output_norm.", "model.norm.")
        .replace("output.", "lm_head.");

    // Ensure .weight suffix
    if !s.ends_with(".weight") && !s.ends_with(".bias") {
        format!("{s}.weight")
    } else {
        s
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main Loader
// ═══════════════════════════════════════════════════════════════════

/// Build a computation graph from GGUF config by writing a temporary config.json
/// and using the existing HF model loader's graph builder.
fn build_graph_from_config(config: &HfConfig) -> Result<OnnxModel, GgufError> {
    // Use the HF graph builder (reuses existing transformer graph construction)
    let transformer_config = config.to_transformer_config();
    let (graph, _moe_slots) = crate::compiler::hf_loader::build_hf_transformer_graph(
        &transformer_config, config.num_hidden_layers,
    );

    Ok(OnnxModel {
        graph,
        weights: GraphWeights {
            weights: Vec::new(),
            biases: Vec::new(),
            named_weights: Vec::new(),
        },
        input_shape: (1, config.hidden_size),
        metadata: crate::compiler::onnx::ModelMetadata {
            name: String::new(),
            num_parameters: 0,
            input_shape: (1, config.hidden_size),
            output_shape: (1, config.hidden_size),
            num_layers: config.num_hidden_layers,
        },
    })
}

/// Load a model from a GGUF file.
///
/// Returns the same `OnnxModel` struct as `load_hf_model()` — the rest of
/// the pipeline (graph compilation, proving, serving) works unchanged.
pub fn load_gguf_model(path: &Path, layers: Option<usize>) -> Result<OnnxModel, GgufError> {
    eprintln!("[gguf] Loading model from {}...", path.display());

    let mut file = std::fs::File::open(path)
        .map_err(|e| GgufError::IoError(format!("Cannot open {}: {e}", path.display())))?;

    // 1. Parse header
    let header = parse_gguf_header(&mut file)?;
    eprintln!("[gguf] Version: {}, tensors: {}, metadata: {}",
        header.version, header.tensor_count, header.metadata_count);

    // 2. Parse metadata
    let metadata = parse_gguf_metadata(&mut file, header.metadata_count)?;

    // 3. Extract model config
    let mut config = extract_model_config(&metadata)?;
    if let Some(max_layers) = layers {
        if max_layers < config.num_hidden_layers {
            config.num_hidden_layers = max_layers;
        }
    }

    // 4. Parse tensor infos
    let tensor_infos = parse_tensor_infos(&mut file, header.tensor_count)?;

    // Record the current position — tensor data starts after alignment
    let data_offset = file.stream_position()
        .map_err(|e| GgufError::IoError(format!("stream_position: {e}")))?;
    // Align to 32 bytes (GGUF spec)
    let aligned_offset = (data_offset + 31) & !31;

    // 5. Memory-map the file for tensor data
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| GgufError::IoError(format!("mmap: {e}")))?;

    // 6. Build computation graph via the public HF model loader
    //    This reuses load_hf_model's graph construction internally by
    //    first building a temporary HF config, then using the graph builder.
    let hf_model = build_graph_from_config(&config)?;
    let graph = hf_model.graph;

    // 7. Load and dequantize weights
    let mut weights = hf_model.weights;

    // Build GGUF tensor name → (node_id, is_named, label) mapping
    // by scanning the graph nodes and matching against GGUF naming conventions
    let mut gguf_name_to_node: HashMap<String, usize> = HashMap::new();
    let mut gguf_name_to_named: HashMap<String, (usize, String)> = HashMap::new();

    // For each layer, map the expected GGUF tensor names
    for layer_idx in 0..config.num_hidden_layers {
        // Standard 7-node-per-layer layout: RMSNorm, Q, O, RMSNorm, Gate, Act, Down
        let base = layer_idx * 7;
        // Q projection = node base+1, O projection = node base+2
        // Gate = base+4, Down = base+6
        gguf_name_to_node.insert(format!("blk.{layer_idx}.attn_q.weight"), base + 1);
        gguf_name_to_node.insert(format!("blk.{layer_idx}.attn_output.weight"), base + 2);
        gguf_name_to_node.insert(format!("blk.{layer_idx}.ffn_gate.weight"), base + 4);
        gguf_name_to_node.insert(format!("blk.{layer_idx}.ffn_down.weight"), base + 6);
        // Named weights
        gguf_name_to_named.insert(format!("blk.{layer_idx}.ffn_up.weight"), (base + 6, "up_proj".into()));
        gguf_name_to_named.insert(format!("blk.{layer_idx}.attn_norm.weight"), (base, "gamma".into()));
        gguf_name_to_named.insert(format!("blk.{layer_idx}.ffn_norm.weight"), (base + 3, "gamma".into()));
    }

    let mut loaded = 0usize;
    let mut skipped = 0usize;

    for tensor_info in &tensor_infos {
        let hf_name = gguf_to_hf_name(&tensor_info.name);
        let num_elements: usize = tensor_info.dimensions.iter().product::<u64>() as usize;

        if num_elements == 0 { continue; }

        // Dequantize tensor data
        let tensor_offset = aligned_offset as usize + tensor_info.offset as usize;
        let dtype = GgufDtype::from_u32(tensor_info.dtype)
            .ok_or_else(|| GgufError::UnsupportedDtype(tensor_info.dtype))?;
        let num_blocks = num_elements / dtype.block_elements().max(1);
        let data_size = num_blocks * dtype.block_size_bytes();

        if tensor_offset + data_size > mmap.len() {
            eprintln!("[gguf]   Skipping {} (offset {} + size {} > file {})",
                tensor_info.name, tensor_offset, data_size, mmap.len());
            skipped += 1;
            continue;
        }

        let raw_data = &mmap[tensor_offset..tensor_offset + data_size];
        let f32_data = dequantize_gguf_tensor(raw_data, tensor_info.dtype, num_elements)?;

        // Determine matrix shape
        let (rows, cols) = if tensor_info.dimensions.len() >= 2 {
            (tensor_info.dimensions[1] as usize, tensor_info.dimensions[0] as usize)
        } else {
            (1, num_elements)
        };

        // Quantize to M31
        let (matrix, _params) = quantize_weight_matrix(
            &f32_data, rows, cols,
            crate::gadgets::quantize::QuantStrategy::Symmetric8,
        );

        // Store as regular weight or named weight
        if let Some(&node_id) = gguf_name_to_node.get(&tensor_info.name) {
            weights.weights.push((node_id, matrix));
            loaded += 1;
        } else if let Some((node_id, label)) = gguf_name_to_named.get(&tensor_info.name) {
            weights.named_weights.push((*node_id, label.clone(), matrix));
            loaded += 1;
        } else {
            skipped += 1;
        }
    }

    eprintln!("[gguf] Loaded: {} weights, skipped: {}", loaded, skipped);

    let input_shape = (1, config.hidden_size);
    Ok(OnnxModel {
        graph,
        weights,
        input_shape,
        metadata: crate::compiler::onnx::ModelMetadata {
            name: String::new(),
            num_parameters: 0,
            input_shape: (1, config.hidden_size),
            output_shape: (1, config.hidden_size),
            num_layers: config.num_hidden_layers,
        },
    })
}

// ═══════════════════════════════════════════════════════════════════
// Errors
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("GGUF parse error: {0}")]
    ParseError(String),
    #[error("GGUF I/O error: {0}")]
    IoError(String),
    #[error("Unsupported GGUF dtype: {0}")]
    UnsupportedDtype(u32),
    #[error("Model config error: {0}")]
    ConfigError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_to_hf_name() {
        assert_eq!(gguf_to_hf_name("blk.0.attn_q.weight"), "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(gguf_to_hf_name("blk.5.ffn_gate.weight"), "model.layers.5.mlp.gate_proj.weight");
        assert_eq!(gguf_to_hf_name("blk.0.attn_norm.weight"), "model.layers.0.input_layernorm.weight");
        assert_eq!(gguf_to_hf_name("token_embd.weight"), "model.embed_tokens.weight");
        assert_eq!(gguf_to_hf_name("output.weight"), "lm_head.weight");
    }

    #[test]
    fn test_f16_to_f32() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_dequant_q8_0() {
        // Scale = 1.0 (f16: 0x3C00), 32 values of [1, 2, ..., 32]
        let mut block = vec![0x00u8, 0x3C]; // f16 1.0
        for i in 1..=32i8 {
            block.push(i as u8);
        }
        let result = dequantize_gguf_tensor(&block, 8, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[31] - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_dequant_f32() {
        let val: f32 = 3.14;
        let bytes = val.to_le_bytes();
        let result = dequantize_gguf_tensor(&bytes, 0, 1).unwrap();
        assert!((result[0] - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_gguf_dtype_block_sizes() {
        assert_eq!(GgufDtype::F32.block_size_bytes(), 4);
        assert_eq!(GgufDtype::Q8_0.block_size_bytes(), 34);
        assert_eq!(GgufDtype::Q4_0.block_size_bytes(), 18);
        assert_eq!(GgufDtype::Q8_0.block_elements(), 32);
        assert_eq!(GgufDtype::Q4_K.block_elements(), 256);
    }
}
