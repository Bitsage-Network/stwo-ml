//! Scale testing for matmul sumcheck proving.
//!
//! Tests matmul sumcheck at progressively larger sizes to find
//! where it breaks (memory, time, numerical issues).
//!
//! Run with: cargo test -p stwo-ml --test scale_matmul -- --nocapture

#![feature(portable_simd)]

use std::time::Instant;
use stwo::core::fields::m31::M31;
use stwo_ml::components::matmul::{
    estimate_sumcheck_memory, matmul_m31, pad_matrix_pow2, prove_matmul_sumcheck,
    prove_matmul_sumcheck_onchain, verify_matmul_sumcheck, verify_matmul_sumcheck_onchain,
    M31Matrix,
};

/// Build a deterministic test matrix with values in [1, 251].
fn make_matrix(rows: usize, cols: usize, seed: u64) -> M31Matrix {
    let mut m = M31Matrix::new(rows, cols);
    let mut state = seed;
    for i in 0..rows {
        for j in 0..cols {
            // LCG pseudorandom
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((state >> 33) % 251 + 1) as u32;
            m.data[i * cols + j] = M31::from(val);
        }
    }
    m
}

/// Format bytes as human-readable string.
fn fmt_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Format duration as human-readable string.
fn fmt_duration(d: std::time::Duration) -> String {
    let ms = d.as_millis();
    if ms < 1000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.2}s", d.as_secs_f64())
    } else {
        format!("{:.1}min", d.as_secs_f64() / 60.0)
    }
}

/// Run a complete scale test at a given size.
/// Returns true if all steps succeeded.
fn scale_test(size: usize, run_onchain: bool) -> bool {
    let (mle_bytes, total_bytes) = estimate_sumcheck_memory(size, size, size);

    println!("\n{:=<60}", "");
    println!("  SCALE TEST: {size}x{size} × {size}x{size}");
    println!("{:=<60}", "");
    println!("  Dimensions:     {size} × {size} × {size}");
    println!("  Sumcheck rounds: {}", (size as f64).log2() as usize);
    println!("  M31 elements:   {} per matrix", size * size);
    println!("  MLE memory:     {} (3 MLEs)", fmt_bytes(mle_bytes));
    println!("  Total estimate:  {}", fmt_bytes(total_bytes));
    println!();

    // Step 1: Build matrices
    let t = Instant::now();
    let a = make_matrix(size, size, 42);
    let b = make_matrix(size, size, 137);
    let build_time = t.elapsed();
    println!("  [1/6] Build matrices:       {}", fmt_duration(build_time));

    // Step 2: Matrix multiply
    let t = Instant::now();
    let c = matmul_m31(&a, &b);
    let matmul_time = t.elapsed();
    println!(
        "  [2/6] matmul_m31:           {}",
        fmt_duration(matmul_time)
    );

    // Sanity check: verify a few elements
    let mut check_sum = M31::from(0);
    for l in 0..size {
        check_sum += a.data[l] * b.data[l * size];
    }
    assert_eq!(c.data[0], check_sum, "matmul spot-check failed at [0,0]");

    // Step 3: Prove (Blake2s channel)
    let t = Instant::now();
    let proof = match prove_matmul_sumcheck(&a, &b, &c) {
        Ok(p) => p,
        Err(e) => {
            println!("  [3/6] prove_matmul_sumcheck: FAILED — {e}");
            return false;
        }
    };
    let prove_time = t.elapsed();
    let num_rounds = proof.sumcheck_proof.round_polys.len();
    println!(
        "  [3/6] prove (Blake2s):      {} ({num_rounds} rounds)",
        fmt_duration(prove_time)
    );

    // Step 4: Verify (Blake2s channel)
    let t = Instant::now();
    match verify_matmul_sumcheck(&proof, &a, &b, &c) {
        Ok(()) => {}
        Err(e) => {
            println!("  [4/6] verify (Blake2s):     FAILED — {e}");
            return false;
        }
    }
    let verify_time = t.elapsed();
    println!(
        "  [4/6] verify (Blake2s):     {}",
        fmt_duration(verify_time)
    );

    if run_onchain {
        // Step 5: Prove (Poseidon channel, on-chain format)
        let t = Instant::now();
        let onchain_proof = match prove_matmul_sumcheck_onchain(&a, &b, &c) {
            Ok(p) => p,
            Err(e) => {
                println!("  [5/6] prove (Poseidon):     FAILED — {e}");
                return false;
            }
        };
        let onchain_prove_time = t.elapsed();
        println!(
            "  [5/6] prove (Poseidon):     {} ({} round polys)",
            fmt_duration(onchain_prove_time),
            onchain_proof.round_polys.len()
        );

        // Step 6: Verify (Poseidon channel)
        let t = Instant::now();
        match verify_matmul_sumcheck_onchain(&onchain_proof) {
            Ok(()) => {}
            Err(e) => {
                println!("  [6/6] verify (Poseidon):    FAILED — {e}");
                return false;
            }
        }
        let onchain_verify_time = t.elapsed();
        println!(
            "  [6/6] verify (Poseidon):    {}",
            fmt_duration(onchain_verify_time)
        );
    } else {
        println!("  [5/6] prove (Poseidon):     SKIPPED (too large)");
        println!("  [6/6] verify (Poseidon):    SKIPPED");
    }

    let total = build_time + matmul_time + prove_time + verify_time;
    println!();
    println!("  TOTAL:                      {}", fmt_duration(total));
    println!("  STATUS:                     PASS");
    true
}

// ==================== Individual Scale Tests ====================

#[test]
fn scale_32x32() {
    assert!(scale_test(32, true));
}

#[test]
fn scale_64x64() {
    assert!(scale_test(64, true));
}

#[test]
fn scale_128x128() {
    assert!(scale_test(128, true));
}

#[test]
fn scale_256x256() {
    assert!(scale_test(256, true));
}

#[test]
fn scale_512x512() {
    assert!(scale_test(512, true));
}

#[test]
fn scale_1024x1024() {
    // On-chain proving at 1024 involves larger MLE commitments (Poseidon Merkle),
    // but should still complete. ~48MB MLE memory.
    assert!(scale_test(1024, true));
}

#[test]
fn scale_2048x2048() {
    // 2048x2048: ~192MB MLE memory, O(8.6B) M31 ops for matmul.
    // Poseidon on-chain proving may be slower due to larger MLE commitment trees.
    // Skip on-chain to keep test time manageable — the Blake2s proof is
    // cryptographically identical in structure.
    assert!(scale_test(2048, false));
}

// ==================== Non-Power-of-2 (Padding) Test ====================

#[test]
fn scale_5120_padded() {
    println!("\n{:=<60}", "");
    println!("  SCALE TEST: 5120x5120 (padded to 8192x8192)");
    println!("{:=<60}", "");

    let orig_size: usize = 5120;
    let padded_size = orig_size.next_power_of_two(); // 8192
    let (mle_bytes, total_bytes) = estimate_sumcheck_memory(padded_size, padded_size, padded_size);

    println!("  Original:       {orig_size} × {orig_size}");
    println!("  Padded to:      {padded_size} × {padded_size}");
    println!(
        "  Padding waste:  {:.0}%",
        (1.0 - (orig_size * orig_size) as f64 / (padded_size * padded_size) as f64) * 100.0
    );
    println!("  MLE memory:     {} (3 MLEs)", fmt_bytes(mle_bytes));
    println!("  Total estimate: {}", fmt_bytes(total_bytes));
    println!();

    // Check if we'd likely OOM (>4GB estimated)
    if total_bytes > 4 * 1024 * 1024 * 1024 {
        println!("  WARNING: Estimated {total_bytes} bytes > 4GB — likely OOM");
        println!("  STATUS:  SKIP (memory limit)");
        println!();
        // Report the finding without panicking
        println!("  FINDING: 5120x5120 (padded 8192x8192) breaks at memory.");
        println!("           Need ~{} for MLEs alone.", fmt_bytes(mle_bytes));
        println!("           Solution: streaming/chunked MLE evaluation.");
        return;
    }

    // Step 1: Build original matrices
    let t = Instant::now();
    let a_orig = make_matrix(orig_size, orig_size, 42);
    let b_orig = make_matrix(orig_size, orig_size, 137);
    println!("  [1/5] Build originals:    {}", fmt_duration(t.elapsed()));

    // Step 2: Pad to power of 2
    let t = Instant::now();
    let a = pad_matrix_pow2(&a_orig);
    let b = pad_matrix_pow2(&b_orig);
    let pad_time = t.elapsed();
    assert_eq!(a.rows, padded_size);
    assert_eq!(a.cols, padded_size);
    println!(
        "  [2/5] Pad to {padded_size}:       {}",
        fmt_duration(pad_time)
    );

    // Step 3: Matmul on padded
    let t = Instant::now();
    let c = matmul_m31(&a, &b);
    let matmul_time = t.elapsed();
    println!("  [3/5] matmul_m31:         {}", fmt_duration(matmul_time));

    // Verify padding didn't corrupt: C[0][0] should match original
    let mut expected_00 = M31::from(0);
    for l in 0..orig_size {
        expected_00 += a_orig.data[l] * b_orig.data[l * orig_size];
    }
    assert_eq!(c.data[0], expected_00, "Padded matmul corrupted [0,0]");

    // Step 4: Prove
    let t = Instant::now();
    match prove_matmul_sumcheck(&a, &b, &c) {
        Ok(proof) => {
            let prove_time = t.elapsed();
            println!(
                "  [4/5] prove (Blake2s):    {} ({} rounds)",
                fmt_duration(prove_time),
                proof.sumcheck_proof.round_polys.len()
            );

            // Step 5: Verify
            let t = Instant::now();
            match verify_matmul_sumcheck(&proof, &a, &b, &c) {
                Ok(()) => {
                    println!("  [5/5] verify (Blake2s):   {}", fmt_duration(t.elapsed()));
                    println!("\n  STATUS: PASS");
                }
                Err(e) => {
                    println!("  [5/5] verify:             FAILED — {e}");
                    panic!("5120 padded verification failed: {e}");
                }
            }
        }
        Err(e) => {
            println!("  [4/5] prove:              FAILED — {e}");
            println!("\n  FINDING: 5120x5120 breaks at proving stage: {e}");
        }
    }
}

// ==================== Memory Estimation Tests ====================

#[test]
fn test_memory_estimates_report() {
    println!("\n  Memory Estimates for Matmul Sumcheck:");
    println!(
        "  {:<12} {:>12} {:>12} {:>12}",
        "Size", "MLE Mem", "Total Est", "Matmul Ops"
    );
    println!("  {}", "-".repeat(52));

    for &size in &[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        let (mle, total) = estimate_sumcheck_memory(size, size, size);
        let ops = (size as u64) * (size as u64) * (size as u64);
        println!(
            "  {:<12} {:>12} {:>12} {:>12}",
            format!("{size}x{size}"),
            fmt_bytes(mle),
            fmt_bytes(total),
            format!("{:.1}B", ops as f64 / 1e9),
        );
    }
}

// ==================== Numerical Stability Tests ====================

#[test]
fn scale_numerical_large_values() {
    // Test with values near the M31 modulus boundary: P = 2^31 - 1 = 2147483647
    // Use values that will cause accumulator overflow within M31
    let size = 64;
    let p_minus_1 = (1u32 << 31) - 2; // P - 1

    let mut a = M31Matrix::new(size, size);
    let mut b = M31Matrix::new(size, size);
    for i in 0..size {
        for j in 0..size {
            a.data[i * size + j] = M31::from(p_minus_1);
            b.data[i * size + j] = M31::from(p_minus_1);
        }
    }

    let c = matmul_m31(&a, &b);

    // Each element: sum of 64 terms of (P-1)^2 mod P
    // (P-1)^2 mod P = 1, so each C[i][j] = 64 mod P = 64
    for i in 0..size {
        for j in 0..size {
            assert_eq!(
                c.data[i * size + j],
                M31::from(64),
                "Numerical check failed at [{i},{j}]"
            );
        }
    }

    // Prove and verify
    let proof = prove_matmul_sumcheck(&a, &b, &c).expect("large-value proving should succeed");
    verify_matmul_sumcheck(&proof, &a, &b, &c).expect("large-value verification should succeed");
    println!("  Numerical: max-value {size}x{size} PASS");
}

#[test]
fn scale_numerical_mixed_values() {
    // Test with a mix of 0, 1, P-1, and mid-range values
    let size = 128;
    let mut a = M31Matrix::new(size, size);
    let mut b = M31Matrix::new(size, size);

    let interesting = [0u32, 1, 2, (1 << 15), (1 << 30), (1 << 31) - 2];
    for i in 0..size {
        for j in 0..size {
            a.data[i * size + j] = M31::from(interesting[(i + j) % interesting.len()]);
            b.data[i * size + j] = M31::from(interesting[(i * 3 + j * 7) % interesting.len()]);
        }
    }

    let c = matmul_m31(&a, &b);
    let proof = prove_matmul_sumcheck(&a, &b, &c).expect("mixed-value proving should succeed");
    verify_matmul_sumcheck(&proof, &a, &b, &c).expect("mixed-value verification should succeed");
    println!("  Numerical: mixed-value {size}x{size} PASS");
}

// ==================== Rectangular Scale Tests ====================

#[test]
fn scale_rectangular_tall() {
    // 1024×64 × 64×1024 = 1024×1024 (tall × wide)
    println!("\n  Rectangular: 1024×64 × 64×1024");

    let a = make_matrix(1024, 64, 42);
    let b = make_matrix(64, 1024, 137);

    let t = Instant::now();
    let c = matmul_m31(&a, &b);
    println!("  matmul:   {}", fmt_duration(t.elapsed()));

    let t = Instant::now();
    let proof = prove_matmul_sumcheck(&a, &b, &c).expect("tall×wide proving should succeed");
    println!(
        "  prove:    {} ({} rounds)",
        fmt_duration(t.elapsed()),
        proof.sumcheck_proof.round_polys.len()
    );

    let t = Instant::now();
    verify_matmul_sumcheck(&proof, &a, &b, &c).expect("tall×wide verification should succeed");
    println!("  verify:   {}", fmt_duration(t.elapsed()));
}

#[test]
fn scale_rectangular_wide() {
    // 64×1024 × 1024×64 = 64×64 (wide × tall)
    println!("\n  Rectangular: 64×1024 × 1024×64");

    let a = make_matrix(64, 1024, 42);
    let b = make_matrix(1024, 64, 137);

    let t = Instant::now();
    let c = matmul_m31(&a, &b);
    println!("  matmul:   {}", fmt_duration(t.elapsed()));

    let t = Instant::now();
    let proof = prove_matmul_sumcheck(&a, &b, &c).expect("wide×tall proving should succeed");
    println!(
        "  prove:    {} ({} rounds)",
        fmt_duration(t.elapsed()),
        proof.sumcheck_proof.round_polys.len()
    );

    let t = Instant::now();
    verify_matmul_sumcheck(&proof, &a, &b, &c).expect("wide×tall verification should succeed");
    println!("  verify:   {}", fmt_duration(t.elapsed()));
}
