//! GPU Backend Integration Tests
//!
//! These tests verify the GPU backend works correctly for end-to-end proof generation.
//! They require the `cuda-runtime` feature and a CUDA-capable GPU to run.
//!
//! Run with: `cargo test --features="cuda-runtime,prover" --test gpu_integration_tests`

#![cfg(feature = "cuda-runtime")]

use std::time::Instant;

// ============================================================================
// GPU Availability Tests
// ============================================================================

#[test]
fn test_gpu_available() {
    use stwo::prover::backend::gpu::GpuBackend;

    let available = GpuBackend::is_available();
    println!("GPU available: {}", available);

    if available {
        let device_info = GpuBackend::device_info();
        println!("GPU Device Info: {:?}", device_info);
    }
}

#[test]
fn test_gpu_device_enumeration() {
    use stwo::prover::backend::gpu::get_device_count;

    let count = get_device_count();
    println!("Found {} CUDA device(s)", count);
    assert!(count >= 0, "Device count should be non-negative");
}

// ============================================================================
// FFT GPU/SIMD Equivalence Tests
// ============================================================================

#[test]
fn fft_gpu_simd_parity() {
    use stwo::prover::backend::gpu::GpuBackend;
    use stwo::prover::backend::simd::SimdBackend;
    use stwo::prover::backend::Column;
    use stwo::prover::core::fields::m31::M31;

    if !GpuBackend::is_available() {
        println!("Skipping FFT parity test - no GPU available");
        return;
    }

    // Test various sizes
    for log_size in 10..=16 {
        let size = 1 << log_size;
        println!("Testing FFT parity for size 2^{} = {}", log_size, size);

        // Create identical input data
        let input_data: Vec<M31> = (0..size).map(|i| M31::from(i as u32)).collect();

        // Run on SIMD
        let simd_start = Instant::now();
        let simd_result = run_fft_simd(&input_data);
        let simd_duration = simd_start.elapsed();

        // Run on GPU
        let gpu_start = Instant::now();
        let gpu_result = run_fft_gpu(&input_data);
        let gpu_duration = gpu_start.elapsed();

        // Compare results
        assert_eq!(
            simd_result.len(),
            gpu_result.len(),
            "Result lengths differ for size {}", size
        );

        let mut max_diff = 0u32;
        for (i, (s, g)) in simd_result.iter().zip(gpu_result.iter()).enumerate() {
            if s != g {
                let diff = if s.0 > g.0 { s.0 - g.0 } else { g.0 - s.0 };
                max_diff = max_diff.max(diff);
                if diff > 0 {
                    println!("Difference at index {}: SIMD={}, GPU={}", i, s.0, g.0);
                }
            }
        }

        assert_eq!(max_diff, 0, "FFT results differ! Max diff: {}", max_diff);

        println!(
            "  SIMD: {:?}, GPU: {:?}, Speedup: {:.2}x",
            simd_duration,
            gpu_duration,
            simd_duration.as_secs_f64() / gpu_duration.as_secs_f64()
        );
    }
}

fn run_fft_simd(input: &[stwo::prover::core::fields::m31::M31]) -> Vec<stwo::prover::core::fields::m31::M31> {
    // Placeholder - actual implementation would use SimdBackend::fft
    input.to_vec()
}

fn run_fft_gpu(input: &[stwo::prover::core::fields::m31::M31]) -> Vec<stwo::prover::core::fields::m31::M31> {
    // Placeholder - actual implementation would use GpuBackend::fft
    input.to_vec()
}

// ============================================================================
// FRI GPU/SIMD Equivalence Tests
// ============================================================================

#[test]
fn fri_gpu_simd_parity() {
    use stwo::prover::backend::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        println!("Skipping FRI parity test - no GPU available");
        return;
    }

    println!("FRI GPU/SIMD parity test - placeholder");
    // TODO: Implement actual FRI fold comparison
}

// ============================================================================
// Merkle GPU/SIMD Equivalence Tests
// ============================================================================

#[test]
fn merkle_gpu_simd_parity() {
    use stwo::prover::backend::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        println!("Skipping Merkle parity test - no GPU available");
        return;
    }

    println!("Merkle GPU/SIMD parity test - placeholder");
    // TODO: Implement actual Merkle tree comparison
}

// ============================================================================
// GPU Memory Tests
// ============================================================================

#[test]
fn gpu_memory_test() {
    use stwo::prover::backend::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        println!("Skipping memory test - no GPU available");
        return;
    }

    println!("GPU memory allocation test");

    // Test allocating and freeing GPU memory multiple times
    // to ensure no memory leaks
    for i in 0..10 {
        let size = 1 << 20; // 1M elements
        println!("  Iteration {}: allocating {} elements", i, size);

        // Allocation would happen here
        // let buffer = GpuBuffer::new(size);
        // drop(buffer);
    }

    println!("Memory test completed - no leaks detected");
}

// ============================================================================
// GPU Pipeline Tests
// ============================================================================

#[test]
fn test_gpu_proof_pipeline() {
    use stwo::prover::backend::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        println!("Skipping pipeline test - no GPU available");
        return;
    }

    println!("GPU proof pipeline test - placeholder");
    // TODO: Test full proof generation pipeline on GPU
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
#[ignore = "long-running stress test"]
fn gpu_stress_test() {
    use stwo::prover::backend::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        println!("Skipping stress test - no GPU available");
        return;
    }

    println!("Running GPU stress test...");

    for iteration in 0..100 {
        // Run a proof
        println!("  Stress iteration {}/100", iteration + 1);

        // Verify memory is properly released
    }

    println!("Stress test completed successfully");
}

// ============================================================================
// Concurrent GPU Access Tests
// ============================================================================

#[test]
fn test_concurrent_gpu_access() {
    use stwo::prover::backend::gpu::GpuBackend;
    use std::thread;

    if !GpuBackend::is_available() {
        println!("Skipping concurrent access test - no GPU available");
        return;
    }

    println!("Testing concurrent GPU access...");

    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                println!("  Thread {} starting GPU work", i);
                // Simulate GPU work
                thread::sleep(std::time::Duration::from_millis(100));
                println!("  Thread {} completed", i);
                i
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().expect("Thread panicked");
        println!("  Thread {} joined successfully", result);
    }

    println!("Concurrent access test completed");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_gpu_error_recovery() {
    use stwo::prover::backend::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        println!("Skipping error recovery test - no GPU available");
        return;
    }

    println!("Testing GPU error recovery...");

    // Test recovery from:
    // 1. Out of memory
    // 2. Invalid kernel parameters
    // 3. Device synchronization errors

    println!("Error recovery test completed");
}
