//! TRUE GPU vs SIMD Benchmark - Real Side-by-Side Comparison
//!
//! This benchmark runs ACTUAL SIMD code vs ACTUAL GPU code on the same workload.
//! No estimates - just real measurements.
//!
//! Run with:
//!   cargo run --example gpu_vs_simd_real_benchmark --features cuda-runtime,prover --release

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use std::time::Instant;

// Import SIMD backend (the actual Stwo SIMD implementation)
#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use stwo::prover::backend::simd::SimdBackend;
#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use stwo::prover::backend::simd::column::BaseColumn;
#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use stwo::core::fields::m31::BaseField;
#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use stwo::core::poly::circle::CanonicCoset;
#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use stwo::prover::poly::BitReversedOrder;

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
use stwo::prover::backend::gpu::pipeline::GpuProofPipeline;

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║     TRUE GPU vs SIMD BENCHMARK - Real Code, Real Measurements               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("This benchmark runs ACTUAL Stwo SIMD code vs ACTUAL GPU code.");
    println!("No estimates - pure performance comparison.");
    println!();

    // Test configurations
    let configs = [
        (16, 4, 5),   // Small: 2^16 × 4 polys × 5 rounds
        (18, 4, 5),   // Medium: 2^18 × 4 polys × 5 rounds
        (18, 8, 10),  // Medium+: 2^18 × 8 polys × 10 rounds
        (20, 4, 5),   // Large: 2^20 × 4 polys × 5 rounds
        (20, 8, 10),  // Production: 2^20 × 8 polys × 10 rounds
    ];

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark: SIMD (Stwo's SimdBackend) vs GPU (Our CUDA Pipeline)             │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();

    for (log_size, num_polys, num_rounds) in configs {
        run_comparison(log_size, num_polys, num_rounds);
        println!();
    }

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Summary                                                                      │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  GPU Pipeline advantages:");
    println!("    • Parallel FFT across thousands of CUDA cores");
    println!("    • Data stays on GPU between operations (no transfer per-op)");
    println!("    • Optimized kernels: vectorized loads, shared memory, etc.");
    println!();
    println!("  For maximum speedup:");
    println!("    • Use larger polynomials (2^20+)");
    println!("    • Process multiple polynomials together");
    println!("    • Keep data on GPU for full proof pipeline");
}

#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn run_comparison(log_size: u32, num_polys: usize, num_rounds: usize) {
    let n = 1usize << log_size;
    let total_ffts = num_polys * num_rounds * 2; // IFFT + FFT per round
    
    println!("Config: 2^{} × {} polys × {} rounds = {} FFTs", log_size, num_polys, num_rounds, total_ffts);
    println!("─────────────────────────────────────────────────────────────────────────");

    // Generate test data
    let test_data: Vec<Vec<BaseField>> = (0..num_polys)
        .map(|i| {
            (0..n)
                .map(|j| BaseField::from_u32_unchecked(((i * n + j) % 0x7FFFFFFF) as u32))
                .collect()
        })
        .collect();

    // =========================================================================
    // SIMD BENCHMARK - Using actual Stwo SimdBackend
    // =========================================================================
    
    let simd_time = run_simd_benchmark(&test_data, log_size, num_rounds);
    
    // =========================================================================
    // GPU BENCHMARK - Using our CUDA pipeline
    // =========================================================================
    
    #[cfg(feature = "cuda-runtime")]
    let gpu_result = run_gpu_benchmark(&test_data, log_size, num_rounds);
    
    #[cfg(not(feature = "cuda-runtime"))]
    let gpu_result: Option<(std::time::Duration, std::time::Duration, std::time::Duration)> = None;

    // =========================================================================
    // RESULTS
    // =========================================================================
    
    let simd_per_fft = simd_time.as_secs_f64() * 1000.0 / total_ffts as f64;
    
    println!("  SIMD (SimdBackend):");
    println!("    Total time:      {:>10.2?}", simd_time);
    println!("    Per FFT:         {:>10.3} ms", simd_per_fft);
    
    if let Some((gpu_total, gpu_compute, gpu_transfer)) = gpu_result {
        let gpu_per_fft = gpu_compute.as_secs_f64() * 1000.0 / total_ffts as f64;
        let speedup = simd_time.as_secs_f64() / gpu_total.as_secs_f64();
        let compute_speedup = simd_time.as_secs_f64() / gpu_compute.as_secs_f64();
        
        println!();
        println!("  GPU Pipeline:");
        println!("    Total time:      {:>10.2?}", gpu_total);
        println!("    Compute only:    {:>10.2?}", gpu_compute);
        println!("    Transfer:        {:>10.2?}", gpu_transfer);
        println!("    Per FFT:         {:>10.3} ms (compute)", gpu_per_fft);
        println!();
        
        // Determine quality
        let (status, color) = if compute_speedup >= 50.0 {
            ("🚀 50x+ ACHIEVED!", "\x1b[32m")
        } else if compute_speedup >= 30.0 {
            ("✅ EXCELLENT", "\x1b[32m")
        } else if compute_speedup >= 20.0 {
            ("👍 GREAT", "\x1b[32m")
        } else if compute_speedup >= 10.0 {
            ("👌 GOOD", "\x1b[33m")
        } else {
            ("⚠️  MODEST", "\x1b[31m")
        };
        
        println!("  ╔═══════════════════════════════════════════════════════════════╗");
        println!("  ║  {}{:<20}\x1b[0m                                    ║", color, status);
        println!("  ║  End-to-End Speedup:   {:>6.1}x  (SIMD / GPU total)         ║", speedup);
        println!("  ║  Compute Speedup:      {:>6.1}x  (SIMD / GPU compute)       ║", compute_speedup);
        println!("  ║  Transfer Overhead:    {:>6.1}%                               ║", 
                 gpu_transfer.as_secs_f64() / gpu_total.as_secs_f64() * 100.0);
        println!("  ╚═══════════════════════════════════════════════════════════════╝");
    } else {
        println!();
        println!("  GPU: Not available (compile with --features cuda-runtime)");
    }
}

/// Run SIMD benchmark using actual Stwo SimdBackend
#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn run_simd_benchmark(
    test_data: &[Vec<BaseField>],
    log_size: u32,
    num_rounds: usize,
) -> std::time::Duration {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
    
    // Convert test data to CircleEvaluation
    let mut evals: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> = test_data
        .iter()
        .map(|data| {
            let col: BaseColumn = data.iter().cloned().collect();
            CircleEvaluation::new(domain, col)
        })
        .collect();
    
    // Warm up
    for eval in &evals {
        let poly = SimdBackend::interpolate(eval.clone(), &twiddles);
        let _ = SimdBackend::evaluate(&poly, domain, &twiddles);
    }
    
    // Benchmark
    let start = Instant::now();
    
    for _round in 0..num_rounds {
        for eval in &mut evals {
            // IFFT: evaluation -> coefficients
            let poly = SimdBackend::interpolate(eval.clone(), &twiddles);
            // FFT: coefficients -> evaluation
            *eval = SimdBackend::evaluate(&poly, domain, &twiddles);
        }
    }
    
    start.elapsed()
}

/// Run GPU benchmark using our CUDA pipeline
#[cfg(all(feature = "cuda-runtime", feature = "prover"))]
fn run_gpu_benchmark(
    test_data: &[Vec<BaseField>],
    log_size: u32,
    num_rounds: usize,
) -> Option<(std::time::Duration, std::time::Duration, std::time::Duration)> {
    // Create pipeline
    let mut pipeline = match GpuProofPipeline::new(log_size) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to create GPU pipeline: {:?}", e);
            return None;
        }
    };
    
    // Convert BaseField to u32
    let data_u32: Vec<Vec<u32>> = test_data
        .iter()
        .map(|v| v.iter().map(|f| f.0).collect())
        .collect();
    
    // Upload phase
    let upload_start = Instant::now();
    for data in &data_u32 {
        if pipeline.upload_polynomial(data).is_err() {
            return None;
        }
    }
    if pipeline.sync().is_err() {
        return None;
    }
    let upload_time = upload_start.elapsed();
    
    // Compute phase
    let compute_start = Instant::now();
    
    for _round in 0..num_rounds {
        for poly_idx in 0..test_data.len() {
            // IFFT
            if pipeline.ifft(poly_idx).is_err() {
                return None;
            }
            // FFT
            if pipeline.fft(poly_idx).is_err() {
                return None;
            }
        }
    }
    
    if pipeline.sync().is_err() {
        return None;
    }
    let compute_time = compute_start.elapsed();
    
    // Download phase (download all results for fair comparison)
    let download_start = Instant::now();
    for poly_idx in 0..test_data.len() {
        if pipeline.download_polynomial(poly_idx).is_err() {
            return None;
        }
    }
    let download_time = download_start.elapsed();
    
    let total_time = upload_time + compute_time + download_time;
    let transfer_time = upload_time + download_time;
    
    Some((total_time, compute_time, transfer_time))
}

#[cfg(not(all(feature = "cuda-runtime", feature = "prover")))]
fn main() {
    println!("This example requires the 'cuda-runtime' and 'prover' features.");
    println!("Run with:");
    println!("  cargo run --example gpu_vs_simd_real_benchmark --features cuda-runtime,prover --release");
}

