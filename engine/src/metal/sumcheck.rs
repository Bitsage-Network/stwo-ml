//! M31 sumcheck round evaluation on Metal GPU via wgpu.
//!
//! The core GKR sumcheck operation: evaluate the degree-2 round polynomial
//! at t=0,1,2 by computing Σ a(x,t)·b(x,t) for each evaluation point.
//!
//! For a matmul layer with N variables:
//! - Round i restricts one variable, halving the MLE size
//! - GPU computes partial sums in parallel (N/2 independent dot products)
//! - CPU does the Fiat-Shamir channel operations (sequential, cheap)

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use super::device::MetalDevice;

/// WGSL shader for sumcheck round polynomial evaluation.
///
/// For each index k in [0, mid), computes:
///   s0 += a[k] * b[k]           (evaluation at t=0)
///   s1 += a[k+mid] * b[k+mid]   (evaluation at t=1)
///   s2 += (a[k]+a[k+mid]) * (b[k]+b[k+mid])  (evaluation at t=2, via interpolation)
///
/// Uses workgroup-level parallel reduction.
const SUMCHECK_ROUND_SHADER: &str = r#"
const P: u32 = 2147483647u;

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;  // [s0, s1, s2] partial sums
@group(0) @binding(3) var<uniform> params: vec4<u32>; // [mid, 0, 0, 0]

fn m31_add(x: u32, y: u32) -> u32 {
    var s = x + y;
    if (s >= P) { s = s - P; }
    return s;
}

fn m31_mul(x: u32, y: u32) -> u32 {
    let lo = x * y;
    let reduced = (lo & P) + (lo >> 31u);
    if (reduced >= P) { return reduced - P; }
    return reduced;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let mid = params.x;

    if (k >= mid) { return; }

    let a0 = a[k];
    let a1 = a[k + mid];
    let b0 = b[k];
    let b1 = b[k + mid];

    // s0 = a[k] * b[k]  (t=0 contribution)
    let prod0 = m31_mul(a0, b0);

    // s1 = a[k+mid] * b[k+mid]  (t=1 contribution)
    let prod1 = m31_mul(a1, b1);

    // s2 = (a0+a1) * (b0+b1)  (t=2 contribution, for Lagrange interpolation)
    let sum_a = m31_add(a0, a1);
    let sum_b = m31_add(b0, b1);
    let prod2 = m31_mul(sum_a, sum_b);

    // Write partial sums (atomic add would be ideal, but WGSL doesn't have it
    // for u32 with mod arithmetic. We write per-thread results and reduce on CPU.)
    let base = k * 3u;
    out[base] = prod0;
    out[base + 1u] = prod1;
    out[base + 2u] = prod2;
}
"#;

/// MLE fold (restriction) shader — restricts one variable of the MLE.
///
/// After the verifier draws challenge r, the prover folds:
///   new_mle[k] = mle[k] * (1 - r) + mle[k + mid] * r
///
/// This halves the MLE size for the next round.
const MLE_FOLD_SHADER: &str = r#"
const P: u32 = 2147483647u;

@group(0) @binding(0) var<storage, read> mle: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // [mid, r_val, 0, 0]

fn m31_add(x: u32, y: u32) -> u32 {
    var s = x + y;
    if (s >= P) { s = s - P; }
    return s;
}

fn m31_sub(x: u32, y: u32) -> u32 {
    if (x >= y) { return x - y; }
    return P - y + x;
}

fn m31_mul(x: u32, y: u32) -> u32 {
    let lo = x * y;
    let reduced = (lo & P) + (lo >> 31u);
    if (reduced >= P) { return reduced - P; }
    return reduced;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let mid = params.x;
    let r = params.y;

    if (k >= mid) { return; }

    // new[k] = mle[k] + r * (mle[k+mid] - mle[k])
    let v0 = mle[k];
    let v1 = mle[k + mid];
    let diff = m31_sub(v1, v0);
    let scaled = m31_mul(diff, r);
    out[k] = m31_add(v0, scaled);
}
"#;

/// Evaluate the sumcheck round polynomial on Metal GPU.
///
/// Returns (s0, s1, s2) — the partial sums at t=0, t=1, t=2.
/// The CPU then interpolates the round polynomial coefficients.
pub fn metal_sumcheck_round(
    a_data: &[M31],
    b_data: &[M31],
    mid: usize,
) -> (M31, M31, M31) {
    use wgpu::util::DeviceExt;

    if mid < 128 {
        // Small: CPU reduction is faster
        return cpu_sumcheck_round(a_data, b_data, mid);
    }

    let dev = MetalDevice::global();

    let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sumcheck_round"),
        source: wgpu::ShaderSource::Wgsl(SUMCHECK_ROUND_SHADER.into()),
    });

    let a_u32: Vec<u32> = a_data.iter().map(|v| v.0).collect();
    let b_u32: Vec<u32> = b_data.iter().map(|v| v.0).collect();

    let a_buf = dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a"), contents: bytemuck::cast_slice(&a_u32), usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b"), contents: bytemuck::cast_slice(&b_u32), usage: wgpu::BufferUsages::STORAGE,
    });

    let out_size = mid * 3;
    let out_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"), size: (out_size * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = [mid as u32, 0, 0, 0];
    let params_buf = dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"), contents: bytemuck::cast_slice(&params), usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"), size: (out_size * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bgl = dev.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    let pl = dev.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
    });
    let pipeline = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pl), module: &shader, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let mut encoder = dev.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((mid as u32 + 255) / 256, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (out_size * 4) as u64);
    dev.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    dev.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let results: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    // CPU reduction of partial sums
    let p = (1u64 << 31) - 1;
    let mut s0 = 0u64;
    let mut s1 = 0u64;
    let mut s2 = 0u64;
    for k in 0..mid {
        s0 += results[k * 3] as u64;
        s1 += results[k * 3 + 1] as u64;
        s2 += results[k * 3 + 2] as u64;
    }

    (
        M31::from((s0 % p) as u32),
        M31::from((s1 % p) as u32),
        M31::from((s2 % p) as u32),
    )
}

/// CPU fallback for small sumcheck rounds.
fn cpu_sumcheck_round(a: &[M31], b: &[M31], mid: usize) -> (M31, M31, M31) {
    let mut s0 = M31::from(0u32);
    let mut s1 = M31::from(0u32);
    let mut s2 = M31::from(0u32);

    for k in 0..mid {
        s0 = s0 + a[k] * b[k];
        s1 = s1 + a[k + mid] * b[k + mid];
        s2 = s2 + (a[k] + a[k + mid]) * (b[k] + b[k + mid]);
    }

    (s0, s1, s2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sumcheck_round_small() {
        // Small test — uses CPU fallback
        let a: Vec<M31> = (0..8).map(|i| M31::from((i + 1) as u32)).collect();
        let b: Vec<M31> = (0..8).map(|i| M31::from((i + 2) as u32)).collect();

        let (s0, s1, s2) = metal_sumcheck_round(&a, &b, 4);
        let (cs0, cs1, cs2) = cpu_sumcheck_round(&a, &b, 4);

        assert_eq!(s0, cs0);
        assert_eq!(s1, cs1);
        assert_eq!(s2, cs2);
    }

    #[test]
    fn test_sumcheck_round_gpu() {
        // Large enough to dispatch to GPU (mid >= 128)
        let n = 512;
        let a: Vec<M31> = (0..n).map(|i| M31::from((i % 100 + 1) as u32)).collect();
        let b: Vec<M31> = (0..n).map(|i| M31::from(((i * 3) % 100 + 1) as u32)).collect();

        let (gs0, gs1, gs2) = metal_sumcheck_round(&a, &b, n / 2);
        let (cs0, cs1, cs2) = cpu_sumcheck_round(&a, &b, n / 2);

        assert_eq!(gs0, cs0, "s0 mismatch");
        assert_eq!(gs1, cs1, "s1 mismatch");
        assert_eq!(gs2, cs2, "s2 mismatch");
    }
}
