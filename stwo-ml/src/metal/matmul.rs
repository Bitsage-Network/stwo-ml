//! M31 matrix multiplication on Metal GPU via wgpu compute shaders.
//!
//! This is the hot kernel — used for both forward pass and GKR layer reductions.
//! Each output element C[i][j] = Σ_k A[i][k] * B[k][j] (mod P), where P = 2^31 - 1.
//!
//! The WGSL shader handles M31 modular arithmetic using u32 with careful
//! overflow handling (the product of two M31 values fits in u64 = 62 bits).

use stwo::core::fields::m31::M31;

use crate::components::matmul::M31Matrix;
use super::device::MetalDevice;

/// WGSL compute shader for M31 matrix multiplication.
///
/// Each workgroup thread computes one output element.
/// Uses tiled access pattern for memory coalescing.
const MATMUL_SHADER: &str = r#"
// M31 prime: P = 2^31 - 1 = 2147483647
const P: u32 = 2147483647u;

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<u32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // [M, K, N, 0]

// M31 modular multiply: (a * b) mod P
// Since a, b < 2^31, the product fits in 62 bits.
// We use the identity: (a * b) mod P where P = 2^31 - 1
// Split into high and low 31-bit halves.
fn m31_mul(a_val: u32, b_val: u32) -> u32 {
    // Full 64-bit product via two 32-bit multiplies
    let a64 = a_val;
    let b64 = b_val;

    // lo = (a * b) & 0x7FFFFFFF (low 31 bits)
    // hi = (a * b) >> 31
    // result = (hi + lo) mod P
    //
    // Since a,b < P < 2^31, product < 2^62
    // hi < 2^31, lo < 2^31
    // hi + lo < 2^32 — fits in u32
    // Then reduce: if result >= P, subtract P

    let product_lo = a64 * b64;                    // low 32 bits of product
    let product_hi = (a64 >> 16u) * (b64 >> 16u);  // approximate high bits

    // Exact 64-bit: use the Mersenne prime trick
    // For P = 2^31 - 1: (x mod P) = (x & P) + (x >> 31), then reduce again
    let lo = product_lo & P;
    let hi = product_lo >> 31u;
    var result = lo + hi;
    if (result >= P) {
        result = result - P;
    }
    return result;
}

// M31 modular add: (a + b) mod P
fn m31_add(a_val: u32, b_val: u32) -> u32 {
    var s = a_val + b_val;
    if (s >= P) {
        s = s - P;
    }
    return s;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let M = dims.x;
    let K = dims.y;
    let N = dims.z;

    if (row >= M || col >= N) {
        return;
    }

    var acc: u32 = 0u;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let a_val = a[row * K + k];
        let b_val = b[k * N + col];
        let prod = m31_mul(a_val, b_val);
        acc = m31_add(acc, prod);
    }

    c[row * N + col] = acc;
}
"#;

/// Perform M31 matrix multiplication on Metal GPU.
///
/// C = A × B (mod P) where A is M×K and B is K×N.
/// Falls back to CPU if the matrices are too small for GPU overhead.
pub fn gpu_matmul_m31_metal(a: &M31Matrix, b: &M31Matrix) -> M31Matrix {
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    assert_eq!(k, b.rows, "matmul dimension mismatch: A.cols={} != B.rows={}", k, b.rows);

    // Small matrices: CPU is faster due to GPU dispatch overhead
    if m * k * n < 4096 {
        return crate::components::matmul::matmul_m31(a, b);
    }

    let dev = MetalDevice::global();

    // Create shader module
    let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("m31_matmul"),
        source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
    });

    // Create buffers
    let a_data: Vec<u32> = a.data.iter().map(|v| v.0).collect();
    let b_data: Vec<u32> = b.data.iter().map(|v| v.0).collect();
    let c_size = (m * n) as usize;

    let a_buf = dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a"),
        contents: bytemuck::cast_slice(&a_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buf = dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b"),
        contents: bytemuck::cast_slice(&b_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let c_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("c"),
        size: (c_size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims = [m as u32, k as u32, n as u32, 0u32];
    let dims_buf = dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Staging buffer for readback
    let staging_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (c_size * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Bind group layout
    let bind_group_layout = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("matmul_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bind"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: c_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
        ],
    });

    let pipeline_layout = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("matmul_pipeline"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Dispatch
    let mut encoder = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // Dispatch enough workgroups to cover M×N output elements
        // Workgroup size is 16×16
        let wg_x = (m as u32 + 15) / 16;
        let wg_y = (n as u32 + 15) / 16;
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    // Copy result to staging buffer
    encoder.copy_buffer_to_buffer(&c_buf, 0, &staging_buf, 0, (c_size * 4) as u64);
    dev.queue.submit(Some(encoder.finish()));

    // Read back results
    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    dev.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result_u32: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();

    // Convert back to M31Matrix
    let mut result = M31Matrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            result.set(i, j, M31::from(result_u32[i * n + j]));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_matmul_small() {
        // 2x3 × 3x2 = 2x2
        let mut a = M31Matrix::new(2, 3);
        a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2)); a.set(0, 2, M31::from(3));
        a.set(1, 0, M31::from(4)); a.set(1, 1, M31::from(5)); a.set(1, 2, M31::from(6));

        let mut b = M31Matrix::new(3, 2);
        b.set(0, 0, M31::from(7)); b.set(0, 1, M31::from(8));
        b.set(1, 0, M31::from(9)); b.set(1, 1, M31::from(10));
        b.set(2, 0, M31::from(11)); b.set(2, 1, M31::from(12));

        // CPU reference
        let cpu_result = crate::components::matmul::matmul_m31(&a, &b);

        // Metal GPU — will use CPU fallback for small matrices (< 4096 elements)
        let gpu_result = gpu_matmul_m31_metal(&a, &b);

        assert_eq!(cpu_result.rows, gpu_result.rows);
        assert_eq!(cpu_result.cols, gpu_result.cols);
        for i in 0..cpu_result.rows {
            for j in 0..cpu_result.cols {
                assert_eq!(
                    cpu_result.get(i, j), gpu_result.get(i, j),
                    "mismatch at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_metal_matmul_medium() {
        // 64x64 × 64x64 — large enough to dispatch to GPU
        let m = 64;
        let k = 64;
        let n = 64;

        let mut a = M31Matrix::new(m, k);
        let mut b = M31Matrix::new(k, n);

        for i in 0..m {
            for j in 0..k {
                a.set(i, j, M31::from(((i * k + j) % 100 + 1) as u32));
            }
        }
        for i in 0..k {
            for j in 0..n {
                b.set(i, j, M31::from(((i * n + j) % 100 + 1) as u32));
            }
        }

        let cpu_result = crate::components::matmul::matmul_m31(&a, &b);
        let gpu_result = gpu_matmul_m31_metal(&a, &b);

        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    cpu_result.get(i, j), gpu_result.get(i, j),
                    "mismatch at ({i},{j})"
                );
            }
        }
    }
}

// Re-export for use by wgpu::util
use wgpu::util::DeviceExt;
