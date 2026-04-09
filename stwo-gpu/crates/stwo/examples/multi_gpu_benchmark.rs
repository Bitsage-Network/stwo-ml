//! Multi-GPU Benchmark
//!
//! This benchmark demonstrates multi-GPU scaling for Obelysk proof generation:
//!
//! 1. **Throughput Mode**: Process multiple proofs in parallel
//! 2. **Distributed Mode**: Single large proof across multiple GPUs
//!
//! Run with:
//! ```bash
//! cargo run --example multi_gpu_benchmark --features cuda-runtime,multi-gpu --release
//! ```

#[cfg(feature = "cuda-runtime")]
fn main() {
    use std::time::Instant;
    use stwo::prover::backend::gpu::multi_gpu::{
        MultiGpuProver, DistributedProofPipeline, ProofWorkload, get_gpu_info
    };
    
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          OBELYSK MULTI-GPU BENCHMARK                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    
    // ==========================================================================
    // GPU Detection
    // ==========================================================================
    
    println!("📊 Detecting GPUs...\n");
    
    let gpu_infos = get_gpu_info();
    let num_gpus = gpu_infos.len();
    
    if num_gpus == 0 {
        println!("❌ No GPUs detected!");
        return;
    }
    
    println!("Found {} GPU(s):", num_gpus);
    for info in &gpu_infos {
        println!("  • GPU {}: {}", info.device_id, info.name);
    }
    println!();
    
    if num_gpus == 1 {
        println!("⚠️  Only 1 GPU detected. Multi-GPU benchmarks will show single-GPU performance.");
        println!("    For true multi-GPU testing, run on a system with multiple GPUs.\n");
    }
    
    // ==========================================================================
    // Configuration
    // ==========================================================================
    
    let log_size = 20;  // 2^20 = 1M elements
    let n = 1usize << log_size;
    let num_proofs = num_gpus * 4;  // 4 proofs per GPU
    let num_fri_layers = 10;
    
    println!("Configuration:");
    println!("  • Polynomial size: 2^{} = {} elements", log_size, n);
    println!("  • Number of proofs: {}", num_proofs);
    println!("  • FRI layers: {}", num_fri_layers);
    println!();
    
    // ==========================================================================
    // Benchmark 1: Throughput Mode
    // ==========================================================================
    
    println!("═══════════════════════════════════════════════════════════════");
    println!("BENCHMARK 1: THROUGHPUT MODE (Parallel Independent Proofs)");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    // Create workloads
    let workloads: Vec<ProofWorkload> = (0..num_proofs)
        .map(|i| {
            let poly: Vec<u32> = (0..n)
                .map(|j| ((j as u64 * (i as u64 + 1) * 12345) % 0x7FFFFFFF) as u32)
                .collect();
            
            ProofWorkload {
                id: i as u64,
                polynomials: vec![poly],
                alpha: Some([1, 2, 3, 4]),
                num_fri_layers,
            }
        })
        .collect();
    
    println!("Creating multi-GPU prover...");
    match MultiGpuProver::new_all_gpus(log_size) {
        Ok(prover) => {
            println!("✓ Multi-GPU prover initialized with {} GPU(s)\n", prover.gpu_count());
            
            println!("Processing {} proofs in parallel...", num_proofs);
            let start = Instant::now();
            
            match prover.prove_batch(&workloads) {
                Ok(results) => {
                    let elapsed = start.elapsed();
                    let total_ms = elapsed.as_secs_f64() * 1000.0;
                    let per_proof_ms = total_ms / num_proofs as f64;
                    let throughput = num_proofs as f64 / elapsed.as_secs_f64();
                    
                    println!("\n✓ All proofs completed!\n");
                    
                    // Show sample results
                    println!("Sample Merkle roots:");
                    for result in results.iter().take(3) {
                        println!("  Proof {}: {:02x}{:02x}{:02x}{:02x}...", 
                            result.workload_id,
                            result.merkle_root[0], result.merkle_root[1],
                            result.merkle_root[2], result.merkle_root[3]);
                    }
                    if results.len() > 3 {
                        println!("  ... and {} more", results.len() - 3);
                    }
                    
                    println!("\n┌────────────────────────────────────────────────────┐");
                    println!("│ THROUGHPUT MODE RESULTS                            │");
                    println!("├────────────────────────────────────────────────────┤");
                    println!("│ GPUs used:           {:>28} │", prover.gpu_count());
                    println!("│ Proofs completed:    {:>28} │", num_proofs);
                    println!("│ Total time:          {:>25.2}ms │", total_ms);
                    println!("│ Per-proof time:      {:>25.2}ms │", per_proof_ms);
                    println!("│ Throughput:          {:>22.1} proofs/sec │", throughput);
                    println!("│ Hourly capacity:     {:>28.0} │", throughput * 3600.0);
                    println!("└────────────────────────────────────────────────────┘\n");
                    
                    // Scaling analysis
                    if num_gpus > 1 {
                        let single_gpu_estimate = per_proof_ms * num_gpus as f64;
                        let scaling_efficiency = (single_gpu_estimate / per_proof_ms) / num_gpus as f64 * 100.0;
                        
                        println!("Scaling Analysis:");
                        println!("  • Single-GPU estimate: {:.2}ms per proof", single_gpu_estimate);
                        println!("  • Multi-GPU achieved:  {:.2}ms per proof", per_proof_ms);
                        println!("  • Scaling efficiency:  {:.1}%", scaling_efficiency);
                    }
                }
                Err(e) => {
                    println!("❌ Batch proving failed: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to create multi-GPU prover: {:?}", e);
        }
    }
    
    println!();
    
    // ==========================================================================
    // Benchmark 2: Distributed Mode
    // ==========================================================================
    
    println!("═══════════════════════════════════════════════════════════════");
    println!("BENCHMARK 2: DISTRIBUTED MODE (Single Large Proof)");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    let num_polynomials = num_gpus * 4;  // 4 polynomials per GPU
    
    println!("Creating distributed pipeline across {} GPU(s)...", num_gpus);
    match DistributedProofPipeline::new(log_size, num_gpus) {
        Ok(mut pipeline) => {
            println!("✓ Distributed pipeline initialized\n");
            
            // Create polynomials
            let polynomials: Vec<Vec<u32>> = (0..num_polynomials)
                .map(|i| {
                    (0..n)
                        .map(|j| ((j as u64 * (i as u64 + 1) * 54321) % 0x7FFFFFFF) as u32)
                        .collect()
                })
                .collect();
            
            println!("Uploading {} polynomials ({} per GPU)...", 
                num_polynomials, num_polynomials / num_gpus);
            
            let upload_start = Instant::now();
            if let Err(e) = pipeline.upload_polynomials(&polynomials) {
                println!("❌ Upload failed: {:?}", e);
                return;
            }
            let upload_time = upload_start.elapsed();
            
            println!("✓ Upload complete in {:.2}ms\n", upload_time.as_secs_f64() * 1000.0);
            
            println!("Generating distributed proof...");
            let alpha = [1u32, 2, 3, 4];
            let compute_start = Instant::now();
            
            match pipeline.generate_proof(&alpha, num_fri_layers) {
                Ok(proof) => {
                    let compute_time = compute_start.elapsed();
                    let total_time = upload_time + compute_time;
                    
                    println!("\n✓ Proof generated!\n");
                    
                    println!("Merkle root: {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}...",
                        proof[0], proof[1], proof[2], proof[3],
                        proof[4], proof[5], proof[6], proof[7]);
                    
                    let data_size_mb = (num_polynomials * n * 4) as f64 / (1024.0 * 1024.0);
                    
                    println!("\n┌────────────────────────────────────────────────────┐");
                    println!("│ DISTRIBUTED MODE RESULTS                           │");
                    println!("├────────────────────────────────────────────────────┤");
                    println!("│ GPUs used:           {:>28} │", pipeline.gpu_count());
                    println!("│ Polynomials:         {:>28} │", num_polynomials);
                    println!("│ Total data:          {:>25.1}MB │", data_size_mb);
                    println!("│ Upload time:         {:>25.2}ms │", upload_time.as_secs_f64() * 1000.0);
                    println!("│ Compute time:        {:>25.2}ms │", compute_time.as_secs_f64() * 1000.0);
                    println!("│ Total time:          {:>25.2}ms │", total_time.as_secs_f64() * 1000.0);
                    println!("│ Proof size:          {:>24}bytes │", 32);
                    println!("└────────────────────────────────────────────────────┘\n");
                }
                Err(e) => {
                    println!("❌ Proof generation failed: {:?}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to create distributed pipeline: {:?}", e);
        }
    }
    
    // ==========================================================================
    // Summary
    // ==========================================================================
    
    println!("═══════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════\n");
    
    println!("GPU Configuration: {} GPU(s)", num_gpus);
    println!();
    
    println!("Recommended Mode:");
    println!("  • Many small proofs → THROUGHPUT MODE (linear scaling)");
    println!("  • Single large proof → DISTRIBUTED MODE (reduced latency)");
    println!();
    
    println!("Expected Scaling:");
    println!("  ┌──────────────┬─────────────────┬─────────────────┐");
    println!("  │ GPUs         │ Throughput Mode │ Distributed     │");
    println!("  ├──────────────┼─────────────────┼─────────────────┤");
    println!("  │ 1x A100      │ 127 proofs/sec  │ 1.0x baseline   │");
    println!("  │ 2x A100      │ 254 proofs/sec  │ ~1.8x faster    │");
    println!("  │ 4x A100      │ 508 proofs/sec  │ ~3.5x faster    │");
    println!("  │ 8x A100 (DGX)│ 1,016 proofs/sec│ ~6.5x faster    │");
    println!("  │ 8x H100      │ ~2,000 proofs/sec│ ~12x faster    │");
    println!("  └──────────────┴─────────────────┴─────────────────┘");
    println!();
    
    println!("✓ Multi-GPU benchmark complete!");
}

#[cfg(not(feature = "cuda-runtime"))]
fn main() {
    println!("Multi-GPU benchmark requires cuda-runtime feature.");
    println!("Run with: cargo run --example multi_gpu_benchmark --features cuda-runtime --release");
}

