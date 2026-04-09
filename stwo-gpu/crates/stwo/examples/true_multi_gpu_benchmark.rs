//! True Multi-GPU Benchmark
//!
//! This benchmark uses the thread-safe multi-GPU executor pool to truly
//! parallelize work across multiple GPUs.
//!
//! Run with:
//! ```bash
//! cargo run --example true_multi_gpu_benchmark --features cuda-runtime --release
//! ```

#[cfg(feature = "cuda-runtime")]
fn main() {
    use std::time::Instant;
    use stwo::prover::backend::gpu::multi_gpu_executor::{
        get_multi_gpu_pool, TrueMultiGpuProver
    };
    
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          TRUE MULTI-GPU BENCHMARK                            ║");
    println!("║          Thread-Safe Parallel Execution                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    
    // Initialize multi-GPU pool
    println!("Initializing multi-GPU pool...");
    let pool = match get_multi_gpu_pool() {
        Ok(p) => p,
        Err(e) => {
            println!("❌ Failed to initialize multi-GPU pool: {:?}", e);
            return;
        }
    };
    
    let num_gpus = pool.gpu_count();
    println!("✓ Pool initialized with {} GPU(s)\n", num_gpus);
    
    println!("GPU Device IDs: {:?}", pool.device_ids());
    println!();
    
    // Configuration
    let log_size = 20u32;  // 2^20 = 1M elements
    let n = 1usize << log_size;
    let num_proofs = num_gpus * 4;  // 4 proofs per GPU
    
    println!("Configuration:");
    println!("  • Polynomial size: 2^{} = {} elements", log_size, n);
    println!("  • Number of proofs: {}", num_proofs);
    println!("  • Proofs per GPU: {}", num_proofs / num_gpus);
    println!();
    
    // ==========================================================================
    // Benchmark 1: Parallel FFT across GPUs
    // ==========================================================================
    
    println!("═══════════════════════════════════════════════════════════════");
    println!("BENCHMARK: Parallel FFT Execution Across {} GPUs", num_gpus);
    println!("═══════════════════════════════════════════════════════════════\n");
    
    // Create workloads
    let workloads: Vec<Vec<u32>> = (0..num_proofs)
        .map(|i| {
            (0..n)
                .map(|j| ((j as u64 * (i as u64 + 1) * 12345) % 0x7FFFFFFF) as u32)
                .collect()
        })
        .collect();
    
    println!("Created {} workloads ({:.1} MB each)\n", num_proofs, (n * 4) as f64 / (1024.0 * 1024.0));
    
    // Create prover
    let prover = match TrueMultiGpuProver::new(log_size) {
        Ok(p) => p,
        Err(e) => {
            println!("❌ Failed to create prover: {:?}", e);
            return;
        }
    };
    
    // Pre-initialize twiddles on all GPUs (warm up)
    println!("Pre-initializing twiddles on all GPUs...");
    for gpu_idx in 0..num_gpus {
        pool.with_gpu(gpu_idx, |ctx| {
            ctx.ensure_twiddles(log_size)
        }).expect("Failed to initialize twiddles");
    }
    println!("✓ All GPUs warmed up\n");
    
    println!("Processing {} proofs in parallel across {} GPUs...\n", num_proofs, num_gpus);
    
    let start = Instant::now();
    
    // Define the processing function - Full FFT pipeline!
    let results = prover.prove_parallel(workloads, |gpu_idx, ctx, data, log_size| {
        // Execute full proof pipeline: IFFT -> FFT
        let _result = ctx.execute_proof_pipeline(data, log_size)?;
        
        // Return GPU index as proof of which GPU processed this
        Ok(gpu_idx)
    });
    
    let elapsed = start.elapsed();
    
    // Count successes and failures
    let mut successes = 0;
    let mut failures = 0;
    let mut gpu_counts: Vec<usize> = vec![0; num_gpus];
    
    for result in &results {
        match result {
            Ok(gpu_idx) => {
                successes += 1;
                if *gpu_idx < gpu_counts.len() {
                    gpu_counts[*gpu_idx] += 1;
                }
            }
            Err(e) => {
                failures += 1;
                println!("  ❌ Error: {:?}", e);
            }
        }
    }
    
    println!("\n┌────────────────────────────────────────────────────┐");
    println!("│ RESULTS                                            │");
    println!("├────────────────────────────────────────────────────┤");
    println!("│ GPUs used:           {:>28} │", num_gpus);
    println!("│ Proofs attempted:    {:>28} │", num_proofs);
    println!("│ Proofs succeeded:    {:>28} │", successes);
    println!("│ Proofs failed:       {:>28} │", failures);
    println!("│ Total time:          {:>25.2}ms │", elapsed.as_secs_f64() * 1000.0);
    
    if successes > 0 {
        let per_proof_ms = elapsed.as_secs_f64() * 1000.0 / successes as f64;
        let throughput = successes as f64 / elapsed.as_secs_f64();
        
        println!("│ Per-proof time:      {:>25.2}ms │", per_proof_ms);
        println!("│ Throughput:          {:>22.1} proofs/sec │", throughput);
        println!("│ Hourly capacity:     {:>28.0} │", throughput * 3600.0);
    }
    println!("└────────────────────────────────────────────────────┘\n");
    
    // Per-GPU breakdown
    println!("Per-GPU Breakdown:");
    for (gpu_idx, count) in gpu_counts.iter().enumerate() {
        let bar_len = (*count * 20) / (num_proofs / num_gpus).max(1);
        let bar: String = "█".repeat(bar_len);
        println!("  GPU {}: {:>3} proofs {}", gpu_idx, count, bar);
    }
    
    println!();
    
    // Scaling analysis
    if successes > 0 {
        let single_gpu_estimate = 160.0; // proofs/sec from single GPU benchmark
        let actual_throughput = successes as f64 / elapsed.as_secs_f64();
        let expected_throughput = single_gpu_estimate * num_gpus as f64;
        let efficiency = (actual_throughput / expected_throughput) * 100.0;
        
        println!("Scaling Analysis:");
        println!("  • Single GPU baseline: {:.1} proofs/sec", single_gpu_estimate);
        println!("  • Expected ({} GPUs): {:.1} proofs/sec", num_gpus, expected_throughput);
        println!("  • Actual achieved: {:.1} proofs/sec", actual_throughput);
        println!("  • Scaling efficiency: {:.1}%", efficiency);
        
        if efficiency > 80.0 {
            println!("\n  🚀 Excellent scaling! Near-linear performance.");
        } else if efficiency > 50.0 {
            println!("\n  ✓ Good scaling. Some overhead from synchronization.");
        } else {
            println!("\n  ⚠️ Suboptimal scaling. May need optimization.");
        }
    }
    
    println!("\n✓ True multi-GPU benchmark complete!");
}

#[cfg(not(feature = "cuda-runtime"))]
fn main() {
    println!("True multi-GPU benchmark requires cuda-runtime feature.");
    println!("Run with: cargo run --example true_multi_gpu_benchmark --features cuda-runtime --release");
}

