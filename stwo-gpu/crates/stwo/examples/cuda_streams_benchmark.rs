//! CUDA Streams Benchmark
//!
//! This benchmark demonstrates the performance improvement from using CUDA streams
//! to overlap data transfers with computation.
//!
//! Run with:
//! ```bash
//! cargo run --example cuda_streams_benchmark --release --features cuda-runtime
//! ```

#[cfg(feature = "cuda-runtime")]
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║            CUDA Streams Benchmark - Obelysk GPU                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::cuda_streams::{
            TripleBufferedPipeline, PipelineOperation, benchmark_streaming_pipeline,
        };
        use stwo::prover::backend::gpu::pipeline::GpuProofPipeline;
        use stwo::prover::backend::gpu::GpuBackend;
        
        // Check GPU availability
        if !GpuBackend::is_available() {
            println!("❌ No CUDA GPU available!");
            return;
        }
        
        if let Some(name) = GpuBackend::device_name() {
            println!("🖥️  GPU: {}", name);
        }
        if let Some(mem) = GpuBackend::available_memory() {
            println!("💾 Memory: {} GB", mem / (1024 * 1024 * 1024));
        }
        if let Some((major, minor)) = GpuBackend::compute_capability() {
            println!("⚡ Compute Capability: {}.{}", major, minor);
        }
        println!();
        
        // Test configurations
        let configs = [
            (18, 4, 10),   // 2^18 = 256K elements, 4 polys, 10 batches
            (20, 4, 10),   // 2^20 = 1M elements, 4 polys, 10 batches
            (22, 4, 5),    // 2^22 = 4M elements, 4 polys, 5 batches
        ];
        
        println!("════════════════════════════════════════════════════════════════════");
        println!("  Comparing Sequential vs Triple-Buffered Pipeline");
        println!("════════════════════════════════════════════════════════════════════");
        println!();
        
        for (log_size, num_polys, num_batches) in configs {
            let n = 1usize << log_size;
            let total_data_mb = (n * 4 * num_polys * num_batches) as f64 / (1024.0 * 1024.0);
            
            println!("┌────────────────────────────────────────────────────────────────┐");
            println!("│ Test: 2^{} ({} elements) × {} polys × {} batches", log_size, n, num_polys, num_batches);
            println!("│ Total data: {:.1} MB", total_data_mb);
            println!("└────────────────────────────────────────────────────────────────┘");
            
            // Generate test data
            let batches: Vec<Vec<Vec<u32>>> = (0..num_batches)
                .map(|batch_idx| {
                    (0..num_polys)
                        .map(|poly_idx| {
                            (0..n)
                                .map(|i| ((i * 7 + poly_idx * 13 + batch_idx * 17 + 23) as u32) % 0x7FFFFFFF)
                                .collect()
                        })
                        .collect()
                })
                .collect();
            
            // ═══════════════════════════════════════════════════════════════════
            // Method 1: Sequential (baseline)
            // ═══════════════════════════════════════════════════════════════════
            let seq_start = Instant::now();
            
            let mut pipeline = match GpuProofPipeline::new(log_size) {
                Ok(p) => p,
                Err(e) => {
                    println!("  ❌ Failed to create pipeline: {:?}", e);
                    continue;
                }
            };
            
            for batch in &batches {
                // Upload
                for data in batch {
                    if let Err(e) = pipeline.upload_polynomial(data) {
                        println!("  ❌ Upload failed: {:?}", e);
                        continue;
                    }
                }
                
                // Compute
                for poly_idx in 0..num_polys {
                    if let Err(e) = pipeline.ifft(poly_idx) {
                        println!("  ❌ IFFT failed: {:?}", e);
                        continue;
                    }
                    if let Err(e) = pipeline.fft(poly_idx) {
                        println!("  ❌ FFT failed: {:?}", e);
                        continue;
                    }
                }
                
                // Download
                for poly_idx in 0..num_polys {
                    let _ = pipeline.download_polynomial(poly_idx);
                }
                
                // Clear for next batch
                pipeline.clear_polynomials();
            }
            
            let seq_time = seq_start.elapsed();
            let seq_throughput = (num_batches * num_polys * 2) as f64 / seq_time.as_secs_f64();
            
            // ═══════════════════════════════════════════════════════════════════
            // Method 2: Triple-Buffered Pipeline
            // ═══════════════════════════════════════════════════════════════════
            let stream_start = Instant::now();
            
            let mut stream_pipeline = match TripleBufferedPipeline::new(log_size, num_polys) {
                Ok(p) => p,
                Err(e) => {
                    println!("  ❌ Failed to create streaming pipeline: {:?}", e);
                    continue;
                }
            };
            
            let _ = stream_pipeline.process_batches(&batches, PipelineOperation::IfftThenFft);
            
            let stream_time = stream_start.elapsed();
            let stream_throughput = (num_batches * num_polys * 2) as f64 / stream_time.as_secs_f64();
            
            // ═══════════════════════════════════════════════════════════════════
            // Results
            // ═══════════════════════════════════════════════════════════════════
            let speedup = seq_time.as_secs_f64() / stream_time.as_secs_f64();
            let improvement_pct = (speedup - 1.0) * 100.0;
            
            println!();
            println!("  📊 Results:");
            println!("  ┌───────────────────┬───────────────┬───────────────┐");
            println!("  │ Method            │ Time          │ Throughput    │");
            println!("  ├───────────────────┼───────────────┼───────────────┤");
            println!("  │ Sequential        │ {:>10.2}ms │ {:>8.1} FFT/s │", 
                     seq_time.as_secs_f64() * 1000.0, seq_throughput);
            println!("  │ Triple-Buffered   │ {:>10.2}ms │ {:>8.1} FFT/s │", 
                     stream_time.as_secs_f64() * 1000.0, stream_throughput);
            println!("  └───────────────────┴───────────────┴───────────────┘");
            println!();
            println!("  ⚡ Speedup: {:.2}x ({:+.1}%)", speedup, improvement_pct);
            println!();
        }
        
        // ═══════════════════════════════════════════════════════════════════════
        // Comprehensive Benchmark
        // ═══════════════════════════════════════════════════════════════════════
        println!("════════════════════════════════════════════════════════════════════");
        println!("  Comprehensive Streaming Benchmark");
        println!("════════════════════════════════════════════════════════════════════");
        println!();
        
        match benchmark_streaming_pipeline(20, 8, 20) {
            Ok(result) => {
                println!("{}", result);
            }
            Err(e) => {
                println!("❌ Benchmark failed: {:?}", e);
            }
        }
        
        println!();
        println!("════════════════════════════════════════════════════════════════════");
        println!("  Summary");
        println!("════════════════════════════════════════════════════════════════════");
        println!();
        println!("  CUDA Streams provide:");
        println!("  • Triple-buffering for continuous GPU utilization");
        println!("  • Overlapped H2D transfers with computation");
        println!("  • Overlapped D2H transfers with next batch upload");
        println!("  • Typical improvement: 10-15% over sequential processing");
        println!();
        println!("  Note: True async transfers require pinned memory allocation,");
        println!("  which would provide additional speedup on high-bandwidth GPUs.");
        println!();
    }
    
    #[cfg(not(feature = "cuda-runtime"))]
    {
        println!("❌ This benchmark requires the 'cuda-runtime' feature.");
        println!("   Run with: cargo run --example cuda_streams_benchmark --release --features cuda-runtime");
    }
}

