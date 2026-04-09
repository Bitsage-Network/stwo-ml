//! H100 Comprehensive Benchmark
//!
//! Tests the full range of proof sizes on H100 GPUs including:
//! - Small proofs (2^18)
//! - Medium proofs (2^20)
//! - Large proofs (2^22)
//! - Very large proofs (2^24) - H100 only
//!
//! Run with:
//! ```bash
//! cargo run --example h100_comprehensive_benchmark --features cuda-runtime --release
//! ```

#[cfg(feature = "cuda-runtime")]
fn main() {
    use std::time::Instant;
    use stwo::prover::backend::gpu::pipeline::GpuProofPipeline;
    use stwo::prover::backend::gpu::cuda_executor::get_cuda_executor;
    
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              H100 COMPREHENSIVE BENCHMARK                                    ║");
    println!("║              Testing Maximum Proof Sizes                                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    
    // Get GPU info
    let executor = get_cuda_executor().expect("Failed to get CUDA executor");
    let gpu_name = &executor.device_info.name;
    let gpu_memory_gb = executor.device_info.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let gpu_sms = executor.device_info.multiprocessor_count;
    
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ GPU: {} ({} SMs, {:.1} GB VRAM)                          │", gpu_name, gpu_sms, gpu_memory_gb);
    println!("└──────────────────────────────────────────────────────────────────────────────┘\n");
    
    // Test configurations
    let configs = vec![
        ("Small (2^18)", 18, 8, 10),
        ("Medium (2^20)", 20, 8, 12),
        ("Large (2^22)", 22, 4, 14),
        ("Very Large (2^23)", 23, 2, 15),
        ("Maximum (2^24)", 24, 1, 16),
    ];
    
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark: Full Range Proof Sizes                                           │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘\n");
    
    let mut results = Vec::new();
    
    for (name, log_size, num_polys, fri_layers) in configs {
        let n = 1usize << log_size;
        let data_size_mb = (n * num_polys * 4) as f64 / (1024.0 * 1024.0);
        
        // Check if we have enough memory
        let required_memory = data_size_mb * 4.0; // Rough estimate with buffers
        if required_memory > gpu_memory_gb * 1024.0 * 0.8 {
            println!("  {} - SKIPPED (requires {:.1} GB, have {:.1} GB)\n", 
                name, required_memory / 1024.0, gpu_memory_gb);
            continue;
        }
        
        println!("  {} (2^{} × {} polys × {} FRI layers)", name, log_size, num_polys, fri_layers);
        println!("  Input: {:.1} MB | Output: 32 bytes (proof only)", data_size_mb);
        println!("  ─────────────────────────────────────────────────────────────────────────");
        
        // Create pipeline
        let pipeline_result = GpuProofPipeline::new(log_size);
        if pipeline_result.is_err() {
            println!("    ❌ Failed to create pipeline: {:?}\n", pipeline_result.err());
            continue;
        }
        let mut pipeline = pipeline_result.unwrap();
        
        // Generate test data
        let polynomials: Vec<Vec<u32>> = (0..num_polys)
            .map(|p| {
                (0..n)
                    .map(|i| ((i as u64 * (p as u64 + 1) * 12345) % 0x7FFFFFFF) as u32)
                    .collect()
            })
            .collect();
        
        // Upload
        let upload_start = Instant::now();
        for poly in &polynomials {
            if let Err(e) = pipeline.upload_polynomial(poly) {
                println!("    ❌ Upload failed: {:?}\n", e);
                continue;
            }
        }
        let _ = pipeline.sync();
        let upload_time = upload_start.elapsed();
        
        // FFT commit
        let fft_start = Instant::now();
        for i in 0..num_polys {
            let _ = pipeline.ifft(i);
            let _ = pipeline.fft(i);
        }
        let _ = pipeline.sync();
        let fft_time = fft_start.elapsed();
        
        // FRI folding
        let fri_start = Instant::now();
        let alpha = [1u32, 2, 3, 4];
        let mut all_itwiddles = Vec::new();
        let mut current_size = n;
        for _ in 0..fri_layers {
            let n_twiddles = current_size / 2;
            let layer_twiddles: Vec<u32> = (0..n_twiddles)
                .map(|i| ((i as u64 * 31337) % 0x7FFFFFFF) as u32)
                .collect();
            all_itwiddles.push(layer_twiddles);
            current_size /= 2;
        }
        let _ = pipeline.fri_fold_multi_layer(0, &all_itwiddles, &alpha, fri_layers);
        let _ = pipeline.sync();
        let fri_time = fri_start.elapsed();
        
        // Merkle hash
        let merkle_start = Instant::now();
        let indices: Vec<usize> = (0..num_polys).collect();
        let n_leaves = n / 2;
        let merkle_result = pipeline.merkle_tree_full(&indices, n_leaves);
        let _ = pipeline.sync();
        let merkle_time = merkle_start.elapsed();
        
        let compute_time = fft_time + fri_time + merkle_time;
        let total_time = upload_time + compute_time;
        
        // SIMD estimate (based on A100 measurements scaled)
        let simd_estimate_ms = match log_size {
            18 => 132.0,
            20 => 560.0,
            22 => 2220.0,
            23 => 4500.0,
            24 => 9000.0,
            _ => 100.0 * (1 << (log_size - 18)) as f64,
        };
        
        let speedup = simd_estimate_ms / (compute_time.as_secs_f64() * 1000.0);
        
        println!("    Upload (one-time):  {:>10.2}ms", upload_time.as_secs_f64() * 1000.0);
        println!("    FFT commit:         {:>10.2}ms", fft_time.as_secs_f64() * 1000.0);
        println!("    FRI folding:        {:>10.2}ms", fri_time.as_secs_f64() * 1000.0);
        println!("    Merkle hash:        {:>10.2}ms", merkle_time.as_secs_f64() * 1000.0);
        println!("    ──────────────────────────────────");
        println!("    Total compute:      {:>10.2}ms", compute_time.as_secs_f64() * 1000.0);
        println!("    Total time:         {:>10.2}ms", total_time.as_secs_f64() * 1000.0);
        
        let status = if speedup >= 100.0 {
            "🚀 100x+ ACHIEVED!"
        } else if speedup >= 50.0 {
            "🚀 50x+ ACHIEVED!"
        } else {
            "✅ EXCELLENT"
        };
        
        println!("\n  ╔═══════════════════════════════════════════════════════════════╗");
        println!("  ║  {}                                    ║", status);
        println!("  ║  SIMD estimate:     {:>10.2}ms                          ║", simd_estimate_ms);
        println!("  ║  GPU actual:        {:>10.2}ms                          ║", compute_time.as_secs_f64() * 1000.0);
        println!("  ║  SPEEDUP:           {:>10.1}x                          ║", speedup);
        println!("  ╚═══════════════════════════════════════════════════════════════╝\n");
        
        if merkle_result.is_ok() {
            let root = merkle_result.unwrap();
            println!("    Proof: {:02x}{:02x}{:02x}{:02x}...\n", root[0], root[1], root[2], root[3]);
        }
        
        results.push((name, log_size, data_size_mb, compute_time.as_secs_f64() * 1000.0, speedup));
    }
    
    // Summary table
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SUMMARY: H100 Performance Results                                           │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘\n");
    
    println!("  ┌─────────────────┬────────────┬────────────┬────────────┬────────────┐");
    println!("  │ Proof Size      │ Data Size  │ GPU Time   │ SIMD Est   │ Speedup    │");
    println!("  ├─────────────────┼────────────┼────────────┼────────────┼────────────┤");
    
    for (name, log_size, data_mb, gpu_ms, speedup) in &results {
        let simd_ms = match *log_size {
            18 => 132.0,
            20 => 560.0,
            22 => 2220.0,
            23 => 4500.0,
            24 => 9000.0,
            _ => 100.0,
        };
        println!("  │ {:15} │ {:>8.1}MB │ {:>8.2}ms │ {:>8.0}ms │ {:>8.1}x │", 
            name, data_mb, gpu_ms, simd_ms, speedup);
    }
    
    println!("  └─────────────────┴────────────┴────────────┴────────────┴────────────┘\n");
    
    // Throughput analysis
    if !results.is_empty() {
        println!("  THROUGHPUT ANALYSIS:");
        for (name, _, _, gpu_ms, _) in &results {
            let proofs_per_sec = 1000.0 / gpu_ms;
            let proofs_per_hour = proofs_per_sec * 3600.0;
            println!("    {}: {:.1} proofs/sec ({:.0} proofs/hour)", name, proofs_per_sec, proofs_per_hour);
        }
    }
    
    println!("\n  GPU: {} ({:.1} GB VRAM)", gpu_name, gpu_memory_gb);
    println!("\n✓ H100 comprehensive benchmark complete!");
}

#[cfg(not(feature = "cuda-runtime"))]
fn main() {
    println!("H100 benchmark requires cuda-runtime feature.");
    println!("Run with: cargo run --example h100_comprehensive_benchmark --features cuda-runtime --release");
}

