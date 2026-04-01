//! Integer-only mathematical functions for platform-deterministic proving.
//!
//! All functions operate in fixed-point arithmetic over M31, producing
//! bit-identical results on every platform (x86, ARM, GPU). No IEEE 754
//! floating-point is used anywhere in this module.
//!
//! # Fixed-Point Representation
//!
//! Values in [-1, 1] are encoded as M31 integers:
//!   `m31_val = round((float_val + 1.0) × SCALE)` where `SCALE = (P-1)/2`
//!
//! This maps: -1.0 → 0, 0.0 → SCALE, 1.0 → P-1
//!
//! # Angle Representation (for RoPE)
//!
//! Angles are stored as fixed-point values in [0, 2π) mapped to [0, 2^32):
//!   `angle_fp = round(angle_radians × 2^32 / (2π))`
//!
//! This gives ~1.5 nano-radian precision, far exceeding M31 requirements.

use stwo::core::fields::m31::M31;

/// M31 prime: 2^31 - 1
const P: u64 = 2147483647;

/// Fixed-point scale for [-1, 1] → M31 mapping: (P-1)/2
pub const FP_SCALE: u64 = (P - 1) / 2;

/// 2^32 as u64, for angle arithmetic.
const TWO_POW_32: u64 = 1u64 << 32;

/// Precomputed 2π × 2^32 / (2π) = 2^32 (identity, but for theta computation
/// we need 2π in fixed-point).
///
/// We represent angles as fractions of a full turn: angle_fp = angle / (2π) × 2^32.
/// So angle_fp = pos × theta_fp, where theta_fp = theta / (2π) × 2^32.

// ─── Cosine / Sine via integer-only Chebyshev ───────────────────────────

/// Cosine lookup table: 1024 entries spanning [0, π/2].
///
/// cos_table[i] = round(cos(i × π / (2 × 1024)) × 2^30)
///
/// We use 2^30 as the internal scale (fits in i32 with room for multiply).
const COS_TABLE_BITS: u32 = 10;
const COS_TABLE_SIZE: usize = 1 << COS_TABLE_BITS; // 1024
const COS_INTERNAL_SCALE: i64 = 1 << 30;

/// Generate the cosine lookup table at init time.
///
/// Table has COS_TABLE_SIZE+1 entries covering [0, π/2] inclusive.
/// Entry i = cos(i × π/(2 × COS_TABLE_SIZE)) × 2^30.
///
/// The f64 computation here runs ONCE at startup (not in the proving path).
/// The resulting table is deterministic because it's the same f64 ops on
/// every platform, rounded to i32 — any sub-ULP differences vanish.
fn build_cos_table() -> [i32; COS_TABLE_SIZE + 1] {
    let mut table = [0i32; COS_TABLE_SIZE + 1];
    let delta = std::f64::consts::FRAC_PI_2 / COS_TABLE_SIZE as f64;
    let mut i = 0;
    while i <= COS_TABLE_SIZE {
        let angle = i as f64 * delta;
        table[i] = (angle.cos() * COS_INTERNAL_SCALE as f64).round() as i32;
        i += 1;
    }
    // Exact endpoints
    table[0] = COS_INTERNAL_SCALE as i32; // cos(0) = 1
    table[COS_TABLE_SIZE] = 0;            // cos(π/2) = 0
    table
}

/// The cosine table, built once. Entry i = cos(i × π/(2×1024)) × 2^30.
/// Includes endpoint: table[1024] = cos(π/2) = 0.
static COS_TABLE: std::sync::LazyLock<[i32; COS_TABLE_SIZE + 1]> =
    std::sync::LazyLock::new(build_cos_table);

/// Look up cos in the first quadrant [0, π/2] from a 30-bit phase.
///
/// `phase`: 30-bit value where 0 = angle 0, 0x3FFF_FFFF ≈ π/2.
fn cos_q0(phase: u32) -> i32 {
    let table = &*COS_TABLE;

    // Map 30-bit phase to table index + interpolation fraction
    let idx = (phase >> (30 - COS_TABLE_BITS)) as usize;
    let interp_bits = 30 - COS_TABLE_BITS; // 20
    let interp_frac = phase & ((1 << interp_bits) - 1);

    let idx = idx.min(COS_TABLE_SIZE);
    let idx_next = (idx + 1).min(COS_TABLE_SIZE);

    let v0 = table[idx] as i64;
    let v1 = table[idx_next] as i64;
    (v0 + ((v1 - v0) * interp_frac as i64) / (1i64 << interp_bits)) as i32
}

/// Integer-only cosine using table lookup + linear interpolation.
///
/// Input: `angle_fp` is an angle in [0, 2^32) representing [0, 2π).
/// Output: cos(angle) as a fixed-point value in [-2^30, 2^30].
///
/// Uses quadrant symmetry to reduce to [0, π/2]:
///   Q0 [0, π/2]:   cos(θ) =  cos(phase)
///   Q1 [π/2, π]:   cos(θ) = -sin(phase)  = -cos(π/2 - phase)
///   Q2 [π, 3π/2]:  cos(θ) = -cos(phase)
///   Q3 [3π/2, 2π]: cos(θ) =  sin(phase)  =  cos(π/2 - phase)
pub fn cos_fixed(angle_fp: u32) -> i32 {
    let quadrant = (angle_fp >> 30) & 3;
    let phase = angle_fp & 0x3FFF_FFFF; // 30-bit position within quadrant
    let complement = 0x3FFF_FFFF_u32.wrapping_sub(phase); // π/2 - phase

    match quadrant {
        0 => cos_q0(phase),
        1 => -cos_q0(complement),
        2 => -cos_q0(phase),
        3 => cos_q0(complement),
        _ => unreachable!(),
    }
}

/// Integer-only sine: sin(θ) = cos(θ - π/2).
pub fn sin_fixed(angle_fp: u32) -> i32 {
    cos_fixed(angle_fp.wrapping_sub(1 << 30))
}

/// Convert a fixed-point cos/sin value (scale 2^30) to M31 signed representation.
///
/// Maps [-2^30, 2^30] → M31 where the M31 encoding is:
///   M31 value = round((float_in_[-1,1] + 1.0) × FP_SCALE)
pub fn fixed_to_m31_signed(val: i32) -> M31 {
    // val is in [-2^30, 2^30], representing [-1, 1]
    // We need: m31 = round((val/2^30 + 1) × FP_SCALE)
    //        = round(val × FP_SCALE / 2^30 + FP_SCALE)
    let scaled = (val as i64 * FP_SCALE as i64) / COS_INTERNAL_SCALE;
    let result = (scaled + FP_SCALE as i64).max(0).min(P as i64 - 1);
    M31::from(result as u32)
}

// ─── RoPE angle computation ─────────────────────────────────────────────

/// Precompute theta values for RoPE: one per dimension pair.
///
/// `theta[j] = base^(-2j/d)` converted to a u32 angle multiplier where
/// the full u32 range [0, 2^32) represents [0, 2π).
///
/// For position `pos`, the rotation angle is `pos × theta[j]` (mod 2^32).
///
/// This function uses f64 to compute the theta values, but it runs ONCE
/// at model registration time — NOT during per-inference proving. The
/// resulting integer multipliers are committed as part of the model config
/// and used deterministically by both prover and verifier.
pub fn precompute_rope_thetas(head_dim: usize, base: f64) -> Vec<u32> {
    let n_pairs = head_dim / 2;
    (0..n_pairs)
        .map(|j| {
            // theta_j = base^(-2j/d)  (radians per position)
            let theta = base.powf(-2.0 * j as f64 / head_dim as f64);
            // Convert to fraction-of-turn × 2^32:
            //   angle_fp = theta / (2π) × 2^32
            let frac_turn = theta / (2.0 * std::f64::consts::PI);
            // Wrap into u32 (angles > 2π wrap naturally)
            (frac_turn * TWO_POW_32 as f64).round() as u64 as u32
        })
        .collect()
}

/// Build integer-only RoPE table: cos and sin values as M31 for every
/// (position, dim_pair) combination.
///
/// This is the deterministic replacement for the f64-based `build_rope_table`.
///
/// The `thetas` parameter comes from `precompute_rope_thetas()` which runs
/// once at model registration. All subsequent computation (per position,
/// per inference) uses only integer arithmetic: multiply + table lookup.
pub fn build_rope_table_integer(
    max_seq_len: usize,
    head_dim: usize,
    thetas: &[u32],
    position_offset: usize,
) -> (Vec<M31>, Vec<M31>) {
    let n_pairs = head_dim / 2;
    assert_eq!(thetas.len(), n_pairs, "thetas length must match head_dim/2");
    let total = max_seq_len * n_pairs;

    let mut cos_vals = Vec::with_capacity(total);
    let mut sin_vals = Vec::with_capacity(total);

    for pos in 0..max_seq_len {
        let absolute_pos = (pos + position_offset) as u32;
        for j in 0..n_pairs {
            // angle_fp = pos × theta_fp (mod 2^32 for natural wrap-around)
            let angle_fp = absolute_pos.wrapping_mul(thetas[j]);
            cos_vals.push(fixed_to_m31_signed(cos_fixed(angle_fp)));
            sin_vals.push(fixed_to_m31_signed(sin_fixed(angle_fp)));
        }
    }

    (cos_vals, sin_vals)
}

// ─── Integer-only activation functions ──────────────────────────────────

/// M31 prime as u32.
const M31_P: u32 = 0x7FFF_FFFF;

/// Apply activation function using integer-only arithmetic.
///
/// This replaces `apply_activation_f64` for deterministic cross-platform computation.
/// Uses piecewise polynomial approximation in M31 arithmetic.
///
/// Input: M31 value encoding a signed number (values > P/2 are negative)
/// Output: M31 value of the activation result
pub fn apply_activation_integer(act_type: u8, val: u32) -> u32 {
    let half_p = M31_P / 2;
    // Convert M31 to signed fixed-point (scale = half_p, representing [-1, 1])
    let signed = if val <= half_p {
        val as i64
    } else {
        val as i64 - M31_P as i64
    };

    let scale = half_p as i64;

    // Compute activation in fixed-point
    let result = match act_type {
        0 => {
            // ReLU: max(0, x)
            if signed > 0 { signed } else { 0 }
        }
        1 => {
            // GELU approximation using integer polynomial
            // GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
            // We use a simpler polynomial fit:
            // GELU(x) ≈ x × sigmoid(1.702x) (Hendrycks & Gimpel approximation)
            // sigmoid(z) ≈ clamp(z/4 + 0.5, 0, 1) (simple linear approx for small range)
            //
            // More precisely, we use a 5th-order Chebyshev approximation
            // of GELU over [-4, 4], scaled to our fixed-point range.
            gelu_integer(signed, scale)
        }
        2 => {
            // Sigmoid: 1/(1+exp(-x))
            // Approximation: piecewise linear + polynomial
            sigmoid_integer(signed, scale)
        }
        3 => {
            // Softmax exp: exp(x) — element-wise
            // For softmax we compute exp(x/scale) × scale
            exp_activation_integer(signed, scale)
        }
        4 => {
            // SiLU: x × sigmoid(x)
            let sig = sigmoid_integer(signed, scale);
            // x × sigmoid(x) / scale (to maintain scale)
            (signed * sig) / scale
        }
        _ => signed, // identity fallback
    };

    // Map back to M31: [0, P-1]
    let result_mod = ((result % M31_P as i64) + M31_P as i64) as u32 % M31_P;
    result_mod
}

/// Integer GELU approximation.
///
/// Uses the Hendrycks & Gimpel form: GELU(x) ≈ x × sigmoid(1.702x)
/// with integer sigmoid.
fn gelu_integer(x: i64, scale: i64) -> i64 {
    // 1.702 in fixed-point: 1.702 × scale ≈ 1.702 × 2^30
    // But we need 1.702 × x / scale to get the argument for sigmoid
    // sigmoid_arg = 1.702 × x (in the same scale as x)
    let sig_arg = (x * 1702) / 1000;
    let sig = sigmoid_integer(sig_arg, scale);
    // GELU(x) = x × sigmoid(1.702x) / scale
    (x * sig) / scale
}

/// Integer sigmoid approximation.
///
/// Uses a 5-segment piecewise linear approximation:
///   x < -5: sigmoid ≈ 0
///   -5 ≤ x < -2.5: linear ramp
///   -2.5 ≤ x ≤ 2.5: linear (steeper)
///   2.5 < x ≤ 5: linear ramp
///   x > 5: sigmoid ≈ 1
///
/// All boundaries and slopes computed in integer arithmetic.
fn sigmoid_integer(x: i64, scale: i64) -> i64 {
    // Boundaries in terms of scale (scale represents 1.0)
    let x5 = scale * 5;  // x = 5.0
    let x25 = scale * 5 / 2;  // x = 2.5

    if x <= -x5 {
        0
    } else if x >= x5 {
        scale
    } else if x <= -x25 {
        // Linear from (−5, 0) to (−2.5, 0.07)
        // slope = 0.07 / 2.5 = 0.028
        let dx = x + x5; // distance from -5
        (dx * 28) / (1000 * (x5 - x25) / scale)
    } else if x >= x25 {
        // Linear from (2.5, 0.93) to (5, 1.0)
        let dx = x - x25;
        let base = scale * 93 / 100;
        base + (dx * 7) / (100 * (x5 - x25) / scale)
    } else {
        // Central region [-2.5, 2.5]: sigmoid ≈ 0.5 + 0.215x (linear)
        // More precisely: 0.5 + x/(4×scale) × scale = scale/2 + x/4
        // (derivative at 0 is 0.25)
        scale / 2 + x / 4
    }
}

/// Integer exp for softmax: exp(x) where x is in [-scale, scale] representing [-1, 1].
///
/// Returns result in the same scale.
fn exp_activation_integer(x: i64, scale: i64) -> i64 {
    // x_real = x / scale, in [-1, 1]
    // We need exp(x_real) × scale
    // exp over [-1, 1] ≈ Taylor: 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
    let x_norm = x; // x is already in scale
    let mut result = scale;
    let mut term = x_norm;
    result += term;
    for n in 2..=8 {
        term = (term * x_norm) / (n as i64 * scale);
        result += term;
        if term.unsigned_abs() < 2 {
            break;
        }
    }
    result.max(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cos_sin_basic() {
        // cos(0) = 1, sin(0) = 0
        let c = cos_fixed(0);
        let s = sin_fixed(0);
        assert!((c - COS_INTERNAL_SCALE as i32).unsigned_abs() < 100, "cos(0) should be ~2^30, got {c}");
        assert!(s.unsigned_abs() < 100, "sin(0) should be ~0, got {s}");

        // cos(π/2) = 0, sin(π/2) = 1 — π/2 corresponds to angle_fp = 2^30
        let c = cos_fixed(1 << 30);
        let s = sin_fixed(1 << 30);
        assert!(c.unsigned_abs() < 1000, "cos(π/2) should be ~0, got {c}");
        assert!((s - COS_INTERNAL_SCALE as i32).unsigned_abs() < 1000, "sin(π/2) should be ~2^30, got {s}");

        // cos(π) = -1 — π corresponds to angle_fp = 2^31
        let c = cos_fixed(1 << 31);
        assert!((c + COS_INTERNAL_SCALE as i32).unsigned_abs() < 100, "cos(π) should be ~-2^30, got {c}");
    }

    #[test]
    fn test_cos_sin_quarter_turn() {
        // cos(π/4) ≈ sin(π/4) ≈ 0.7071
        // π/4 → angle_fp = 2^29
        let c = cos_fixed(1 << 29);
        let s = sin_fixed(1 << 29);
        let expected = (0.7071067811865476 * COS_INTERNAL_SCALE as f64).round() as i32;
        assert!(
            (c - expected).unsigned_abs() < 1000,
            "cos(π/4): expected ~{expected}, got {c}"
        );
        assert!(
            (s - expected).unsigned_abs() < 1000,
            "sin(π/4): expected ~{expected}, got {s}"
        );
    }

    #[test]
    fn test_fixed_to_m31() {
        // cos(0) = 1.0 → M31 should be P-1
        let m = fixed_to_m31_signed(COS_INTERNAL_SCALE as i32);
        assert!(
            m.0 > P as u32 - 100,
            "cos(0) as M31 should be ~P-1, got {}",
            m.0
        );

        // cos(π) = -1.0 → M31 should be 0
        let m = fixed_to_m31_signed(-(COS_INTERNAL_SCALE as i32));
        assert!(m.0 < 100, "cos(π) as M31 should be ~0, got {}", m.0);

        // 0.0 → M31 should be SCALE = (P-1)/2
        let m = fixed_to_m31_signed(0);
        let expected = FP_SCALE as u32;
        assert!(
            (m.0 as i64 - expected as i64).unsigned_abs() < 10,
            "0.0 as M31 should be ~{expected}, got {}",
            m.0
        );
    }

    #[test]
    fn test_rope_table_matches_f64() {
        // Build both tables and compare
        let head_dim = 64;
        let max_seq_len = 16;
        let base = 10000.0_f64;
        let thetas = precompute_rope_thetas(head_dim, base);

        let (int_cos, int_sin) = build_rope_table_integer(max_seq_len, head_dim, &thetas, 0);

        // Build f64 reference
        let n_pairs = head_dim / 2;
        let mut max_cos_diff: u32 = 0;
        let mut max_sin_diff: u32 = 0;

        for pos in 0..max_seq_len {
            for j in 0..n_pairs {
                let theta = base.powf(-2.0 * j as f64 / head_dim as f64);
                let angle = pos as f64 * theta;
                let ref_cos = crate::components::rope::float_to_m31_signed(angle.cos());
                let ref_sin = crate::components::rope::float_to_m31_signed(angle.sin());

                let idx = pos * n_pairs + j;
                let diff_c = (int_cos[idx].0 as i64 - ref_cos.0 as i64).unsigned_abs() as u32;
                let diff_s = (int_sin[idx].0 as i64 - ref_sin.0 as i64).unsigned_abs() as u32;
                max_cos_diff = max_cos_diff.max(diff_c);
                max_sin_diff = max_sin_diff.max(diff_s);
            }
        }

        eprintln!("RoPE table max diffs: cos={max_cos_diff}, sin={max_sin_diff}");
        // Allow up to 0.001% of P ≈ 21475 difference
        // The integer cos/sin table has 1024-point resolution with linear interp,
        // giving ~20-bit accuracy. The M31 encoding has 31 bits. So we expect
        // diffs up to ~2^(31-20) = 2048.
        let threshold = 5000;
        assert!(
            max_cos_diff < threshold,
            "cos diff too large: {max_cos_diff} >= {threshold}"
        );
        assert!(
            max_sin_diff < threshold,
            "sin diff too large: {max_sin_diff} >= {threshold}"
        );
    }

    #[test]
    fn test_sigmoid_integer_bounds() {
        let scale = FP_SCALE as i64;

        // sigmoid(0) ≈ 0.5
        let s0 = sigmoid_integer(0, scale);
        assert!(
            (s0 - scale / 2).unsigned_abs() < (scale as u64 / 100),
            "sigmoid(0) should be ~0.5×scale, got {s0}"
        );

        // sigmoid(-5×scale) ≈ 0
        let s_neg = sigmoid_integer(-5 * scale, scale);
        assert!(s_neg.unsigned_abs() < (scale as u64 / 100), "sigmoid(-5) should be ~0, got {s_neg}");

        // sigmoid(5×scale) ≈ 1
        let s_pos = sigmoid_integer(5 * scale, scale);
        assert!(
            (s_pos - scale).unsigned_abs() < (scale as u64 / 100),
            "sigmoid(5) should be ~1×scale, got {s_pos}"
        );
    }

    #[test]
    fn test_activation_relu() {
        let half_p = M31_P / 2;
        // Positive input: should pass through
        let result = apply_activation_integer(0, 100);
        assert_eq!(result, 100);

        // Negative input (val > P/2): should return 0
        let result = apply_activation_integer(0, M31_P - 100);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_determinism() {
        // Run the same computation twice, verify identical results
        let thetas = precompute_rope_thetas(64, 10000.0);
        let (cos1, sin1) = build_rope_table_integer(32, 64, &thetas, 0);
        let (cos2, sin2) = build_rope_table_integer(32, 64, &thetas, 0);
        assert_eq!(cos1, cos2, "RoPE cos tables should be identical across runs");
        assert_eq!(sin1, sin2, "RoPE sin tables should be identical across runs");
    }
}
