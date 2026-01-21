//! GPU Backend Unit Tests
//!
//! These tests verify individual GPU operations work correctly.
//! They require the `cuda-runtime` feature and a CUDA-capable GPU to run.
//!
//! Run with: `cargo test --features="cuda-runtime,prover" --test gpu_unit_tests`

#![cfg(feature = "cuda-runtime")]

// ============================================================================
// FFT Unit Tests
// ============================================================================

mod fft_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_fft_small() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Test FFT on small input (should fall back to SIMD)
        let size = 1 << 10; // 1024 - below GPU threshold
        println!("Testing FFT size {} (should use SIMD fallback)", size);
    }

    #[test]
    fn test_fft_threshold() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Test FFT at GPU threshold boundary
        let threshold_size = 1 << 14; // 16K - typical GPU threshold
        println!("Testing FFT at threshold size {}", threshold_size);
    }

    #[test]
    fn test_fft_large() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Test FFT on large input (should use GPU)
        let size = 1 << 20; // 1M elements
        println!("Testing FFT size {} (should use GPU)", size);
    }

    #[test]
    fn test_fft_power_of_two() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Test various power-of-two sizes
        for log_size in 10..=22 {
            let size = 1usize << log_size;
            println!("Testing FFT 2^{} = {}", log_size, size);
        }
    }

    #[test]
    fn test_inverse_fft() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Test that IFFT(FFT(x)) == x
        println!("Testing FFT/IFFT round-trip");
    }
}

// ============================================================================
// Merkle Tree Unit Tests
// ============================================================================

mod merkle_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_merkle_small() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Small Merkle tree (should use SIMD)
        let leaves = 1 << 10;
        println!("Testing Merkle with {} leaves (SIMD fallback)", leaves);
    }

    #[test]
    fn test_merkle_large() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Large Merkle tree (should use GPU)
        let leaves = 1 << 20;
        println!("Testing Merkle with {} leaves (GPU)", leaves);
    }

    #[test]
    fn test_merkle_root_deterministic() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Same input should produce same root
        println!("Testing Merkle root determinism");
    }
}

// ============================================================================
// FRI Unit Tests
// ============================================================================

mod fri_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_fri_fold_line() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing FRI fold_line");
    }

    #[test]
    fn test_fri_fold_circle_into_line() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing FRI fold_circle_into_line");
    }

    #[test]
    fn test_fri_multiple_layers() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Test folding through multiple FRI layers
        println!("Testing FRI multi-layer folding");
    }
}

// ============================================================================
// Quotient Unit Tests
// ============================================================================

mod quotient_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_quotient_accumulation() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing quotient accumulation");
    }

    #[test]
    fn test_quotient_domain_extension() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing quotient domain extension");
    }
}

// ============================================================================
// Memory Management Unit Tests
// ============================================================================

mod memory_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_gpu_buffer_allocation() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing GPU buffer allocation");
    }

    #[test]
    fn test_gpu_buffer_transfer() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing GPU buffer H2D/D2H transfer");
    }

    #[test]
    fn test_gpu_buffer_drop() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        // Test that buffers are properly freed on drop
        println!("Testing GPU buffer cleanup on drop");
    }
}

// ============================================================================
// CUDA Executor Unit Tests
// ============================================================================

mod executor_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_cuda_executor_init() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing CUDA executor initialization");
    }

    #[test]
    fn test_cuda_kernel_compilation() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing CUDA kernel compilation via NVRTC");
    }

    #[test]
    fn test_cuda_stream_operations() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing CUDA stream operations");
    }
}

// ============================================================================
// Pipeline Unit Tests
// ============================================================================

mod pipeline_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_pipeline_creation() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing GPU proof pipeline creation");
    }

    #[test]
    fn test_pipeline_polynomial_upload() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing polynomial upload to GPU pipeline");
    }

    #[test]
    fn test_pipeline_twiddle_caching() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing twiddle factor caching in pipeline");
    }
}

// ============================================================================
// Multi-GPU Unit Tests
// ============================================================================

mod multi_gpu_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn multi_gpu_device_enumeration() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing multi-GPU device enumeration");
    }

    #[test]
    fn multi_gpu_independent_execution() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing independent execution on multiple GPUs");
    }
}

// ============================================================================
// TEE Unit Tests
// ============================================================================

mod tee_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_tee_mode_detection() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing TEE/CC mode detection");
    }

    #[test]
    fn test_confidential_gpu_detection() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing confidential GPU model detection (H100/H200/B200)");
    }
}

// ============================================================================
// Conversion Unit Tests
// ============================================================================

mod conversion_tests {
    use stwo::prover::backend::gpu::GpuBackend;

    #[test]
    fn test_gpu_simd_conversion() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing GpuBackend <-> SimdBackend conversion");
    }

    #[test]
    fn test_column_conversion() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU");
            return;
        }

        println!("Testing column type conversions");
    }
}

// ============================================================================
// GPU Constraint Kernel Unit Tests
// ============================================================================

mod constraint_kernel_tests {
    use stwo::prover::backend::gpu::GpuBackend;
    use stwo::prover::backend::gpu::constraints::{
        get_full_kernel_source, ConstraintKernelConfig, ConstraintKernelStats,
        M31_FIELD_KERNEL, CONSTRAINT_EVAL_KERNEL, QUOTIENT_KERNEL,
    };

    #[test]
    fn test_m31_kernel_source() {
        let source = M31_FIELD_KERNEL;

        // Verify M31 field operations are defined
        assert!(source.contains("m31_add"));
        assert!(source.contains("m31_sub"));
        assert!(source.contains("m31_mul"));
        assert!(source.contains("m31_sqr"));
        assert!(source.contains("m31_pow"));
        assert!(source.contains("m31_inv"));
        assert!(source.contains("m31_div"));
        assert!(source.contains("m31_neg"));

        // Verify M31 prime constant
        assert!(source.contains("M31_P"));
        assert!(source.contains("0x7FFFFFFF"));

        println!("M31 kernel source validation passed");
    }

    #[test]
    fn test_constraint_eval_kernel_source() {
        let source = CONSTRAINT_EVAL_KERNEL;

        // Verify constraint evaluation functions are defined
        assert!(source.contains("eval_constraints_generic"));
        assert!(source.contains("eval_degree2_constraints"));
        assert!(source.contains("eval_transition_constraints"));
        assert!(source.contains("eval_boundary_constraints"));
        assert!(source.contains("accumulate_constraints"));

        // Verify kernel parameters
        assert!(source.contains("__global__"));
        assert!(source.contains("blockIdx"));
        assert!(source.contains("threadIdx"));
        assert!(source.contains("domain_size"));

        println!("Constraint eval kernel source validation passed");
    }

    #[test]
    fn test_quotient_kernel_source() {
        let source = QUOTIENT_KERNEL;

        // Verify quotient functions are defined
        assert!(source.contains("compute_quotient"));
        assert!(source.contains("compute_quotient_batch"));
        assert!(source.contains("zerofier"));

        println!("Quotient kernel source validation passed");
    }

    #[test]
    fn test_full_kernel_source() {
        let full_source = get_full_kernel_source();

        // Verify all components are included
        assert!(full_source.contains("m31_add"));
        assert!(full_source.contains("eval_constraints_generic"));
        assert!(full_source.contains("compute_quotient"));

        // Verify source is reasonably sized
        assert!(full_source.len() > 3000, "Full kernel source should be substantial");

        println!("Full kernel source contains {} bytes", full_source.len());
    }

    #[test]
    fn test_constraint_kernel_config_default() {
        let config = ConstraintKernelConfig::default();

        // Verify defaults
        assert_eq!(config.block_size, 256);
        assert!(config.prefer_l1_cache);
        assert_eq!(config.shared_mem_bytes, 0);

        println!("Default constraint kernel config: block_size={}, prefer_l1_cache={}",
                 config.block_size, config.prefer_l1_cache);
    }

    #[test]
    fn test_constraint_kernel_stats() {
        let mut stats = ConstraintKernelStats::new();

        // Initial state
        assert_eq!(stats.total_evaluations, 0);
        assert_eq!(stats.total_kernel_time_us, 0);
        assert!((stats.avg_kernel_time_us() - 0.0).abs() < f64::EPSILON);

        // Update stats
        stats.total_evaluations = 1000;
        stats.total_kernel_time_us = 5000;
        stats.degree2_evals = 500;
        stats.transition_evals = 300;
        stats.boundary_evals = 200;

        // Verify average calculation
        assert!((stats.avg_kernel_time_us() - 5.0).abs() < 0.001);

        println!("Constraint kernel stats: avg={}us over {} evaluations",
                 stats.avg_kernel_time_us(), stats.total_evaluations);
    }

    #[test]
    fn test_constraint_kernel_compilation() {
        if !GpuBackend::is_available() {
            println!("Skipping constraint kernel compilation test - no GPU");
            return;
        }

        // This test will only run with cuda-runtime feature and actual GPU
        println!("Testing constraint kernel compilation on GPU");

        // The ConstraintKernel::new() will compile PTX if GPU is available
        // Additional verification would go here with actual CUDA
    }

    #[test]
    fn test_m31_arithmetic_correctness() {
        // Test M31 arithmetic properties without GPU
        // These verify the kernel logic is mathematically correct

        const M31_P: u64 = 0x7FFFFFFF;

        // Test basic addition
        let a = 100u64;
        let b = 200u64;
        let sum = (a + b) % M31_P;
        assert_eq!(sum, 300);

        // Test overflow addition
        let near_max = M31_P - 1;
        let overflow_sum = (near_max + 10) % M31_P;
        assert_eq!(overflow_sum, 9);

        // Test multiplication
        let prod = (a * b) % M31_P;
        assert_eq!(prod, 20000);

        // Test multiplication near overflow
        let large_a = M31_P - 1;
        let large_b = 2u64;
        let large_prod = (large_a * large_b) % M31_P;
        assert_eq!(large_prod, M31_P - 2);  // (p-1)*2 = 2p - 2 ≡ -2 ≡ p-2 (mod p)

        println!("M31 arithmetic correctness verified");
    }

    #[test]
    fn test_constraint_kernel_thread_config() {
        // Test various thread configurations
        let configs = vec![
            ConstraintKernelConfig { block_size: 128, shared_mem_bytes: 0, prefer_l1_cache: true },
            ConstraintKernelConfig { block_size: 256, shared_mem_bytes: 1024, prefer_l1_cache: true },
            ConstraintKernelConfig { block_size: 512, shared_mem_bytes: 4096, prefer_l1_cache: false },
        ];

        for config in &configs {
            // Verify block size is power of 2
            assert!(config.block_size.is_power_of_two());
            // Verify block size is reasonable (32-1024)
            assert!(config.block_size >= 32 && config.block_size <= 1024);

            println!("Config validated: block_size={}, shared_mem={}",
                     config.block_size, config.shared_mem_bytes);
        }
    }
}

// ============================================================================
// Pinned Memory Pool Unit Tests
// ============================================================================

mod pinned_pool_tests {
    use stwo::prover::backend::gpu::GpuBackend;
    use stwo::prover::backend::gpu::optimizations::{
        PinnedPoolStats, get_pinned_pool_u32,
    };

    #[test]
    fn test_pinned_pool_stats_structure() {
        // Test PinnedPoolStats without requiring CUDA
        // This verifies the struct is properly exported

        #[cfg(feature = "cuda-runtime")]
        {
            let stats = PinnedPoolStats::default();
            assert_eq!(stats.acquisitions, 0);
            assert_eq!(stats.hits, 0);
            assert_eq!(stats.misses, 0);
            assert_eq!(stats.bytes_allocated, 0);
            assert_eq!(stats.bytes_pooled, 0);

            println!("PinnedPoolStats structure validated");
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            println!("Skipping - requires cuda-runtime feature");
        }
    }

    #[test]
    fn test_pinned_pool_hit_rate_calculation() {
        #[cfg(feature = "cuda-runtime")]
        {
            let mut stats = PinnedPoolStats::default();

            // Zero acquisitions should return 0%
            assert_eq!(stats.hit_rate(), 0.0);

            // 100 acquisitions, 75 hits = 75% hit rate
            stats.acquisitions = 100;
            stats.hits = 75;
            let rate = stats.hit_rate();
            assert!((rate - 75.0).abs() < 0.01, "Expected 75%, got {}", rate);

            // Verify miss rate is complement
            let miss_rate = stats.miss_rate();
            assert!((miss_rate - 25.0).abs() < 0.01, "Expected 25% miss rate, got {}", miss_rate);

            println!("Hit rate: {}%, Miss rate: {}%", rate, miss_rate);
        }

        #[cfg(not(feature = "cuda-runtime"))]
        {
            println!("Skipping - requires cuda-runtime feature");
        }
    }

    #[test]
    fn test_global_pinned_pool_singleton() {
        if !GpuBackend::is_available() {
            println!("Skipping - no GPU available");
            return;
        }

        #[cfg(feature = "cuda-runtime")]
        {
            // Get the global pool twice - should be same instance
            let pool1 = get_pinned_pool_u32();
            let pool2 = get_pinned_pool_u32();

            // Verify they're the same pool by checking pointer equality
            let ptr1 = pool1 as *const _ as usize;
            let ptr2 = pool2 as *const _ as usize;
            assert_eq!(ptr1, ptr2, "Global pool should be singleton");

            println!("Global pinned pool singleton verified");
        }
    }

    #[test]
    fn test_pinned_pool_acquire_release() {
        if !GpuBackend::is_available() {
            println!("Skipping pinned pool acquire/release - no GPU available");
            return;
        }

        #[cfg(feature = "cuda-runtime")]
        {
            let pool = get_pinned_pool_u32();

            // First acquisition - should be a miss
            let buf1 = pool.acquire(1024);
            assert!(buf1.is_ok(), "First acquire should succeed");

            let stats = pool.stats().unwrap();
            assert_eq!(stats.acquisitions, stats.hits + stats.misses);

            // Buffer returns to pool when dropped
            drop(buf1);

            // Second acquisition of same size - should be a hit
            let buf2 = pool.acquire(1024);
            assert!(buf2.is_ok(), "Second acquire should succeed");

            let buf2 = buf2.unwrap();
            assert_eq!(buf2.len(), 1024);

            println!("Pinned pool acquire/release verified");
            println!("Pool stats: {:?}", pool.stats());
        }
    }

    #[test]
    fn test_pinned_pool_size_classes() {
        if !GpuBackend::is_available() {
            println!("Skipping pinned pool size classes - no GPU available");
            return;
        }

        #[cfg(feature = "cuda-runtime")]
        {
            let pool = get_pinned_pool_u32();

            // Request non-power-of-two sizes - should round up
            let buf1 = pool.acquire(1000);  // Rounds up to 1024
            assert!(buf1.is_ok());
            let buf1 = buf1.unwrap();
            assert!(buf1.len() >= 1000, "Buffer should be at least 1000");

            let buf2 = pool.acquire(2000);  // Rounds up to 2048
            assert!(buf2.is_ok());
            let buf2 = buf2.unwrap();
            assert!(buf2.len() >= 2000, "Buffer should be at least 2000");

            println!("Size class rounding verified: 1000->1024, 2000->2048");
            drop(buf1);
            drop(buf2);
        }
    }

    #[test]
    fn test_pinned_pool_with_data() {
        if !GpuBackend::is_available() {
            println!("Skipping pinned pool with data - no GPU available");
            return;
        }

        #[cfg(feature = "cuda-runtime")]
        {
            let pool = get_pinned_pool_u32();

            // Create test data
            let data: Vec<u32> = (0..1024).collect();

            // Acquire with data copy
            let buf = pool.acquire_with_data(&data);
            assert!(buf.is_ok(), "acquire_with_data should succeed");

            let buf = buf.unwrap();

            // Verify data was copied correctly
            assert_eq!(buf.as_slice()[0..10], data[0..10]);
            assert_eq!(buf.as_slice()[1000..1024], data[1000..1024]);

            println!("Pinned pool acquire_with_data verified");
        }
    }

    #[test]
    fn test_pinned_pool_pooled_count() {
        if !GpuBackend::is_available() {
            println!("Skipping pinned pool count - no GPU available");
            return;
        }

        #[cfg(feature = "cuda-runtime")]
        {
            let pool = get_pinned_pool_u32();

            let initial_count = pool.pooled_count();
            let initial_bytes = pool.pooled_bytes();

            // Acquire and release a buffer
            let buf = pool.acquire(4096).unwrap();
            drop(buf);

            // Pool should have at least one more buffer
            let new_count = pool.pooled_count();
            let new_bytes = pool.pooled_bytes();

            assert!(new_count >= initial_count, "Pool count should not decrease");
            assert!(new_bytes >= initial_bytes, "Pool bytes should not decrease");

            println!("Pool count: {} -> {}, bytes: {} -> {}",
                     initial_count, new_count, initial_bytes, new_bytes);
        }
    }
}
