// Field Tower: M31 → CM31 → QM31
//
// M31  = Z / (2^31 - 1)       — Mersenne prime field
// CM31 = M31[i] / (i² + 1)    — Complex extension
// QM31 = CM31[j] / (j² - 2-i) — Degree-4 secure field
//
// Matches STWO's field tower exactly. All arithmetic uses u64 to avoid
// overflow: max M31 product is (2^31-2)^2 ≈ 2^62, which fits in u64.

// ============================================================================
// M31: Z / (2^31 - 1)
// ============================================================================

/// The Mersenne prime 2^31 - 1 = 2147483647.
pub const M31_P: u64 = 0x7FFFFFFF;

/// 2^31 as felt252, used for QM31↔felt252 packing.
pub const M31_SHIFT: felt252 = 0x80000000;

/// M31 addition: (a + b) mod P.
pub fn m31_add(a: u64, b: u64) -> u64 {
    let sum = a + b;
    if sum >= M31_P {
        sum - M31_P
    } else {
        sum
    }
}

/// M31 subtraction: (a - b) mod P.
pub fn m31_sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        M31_P - (b - a)
    }
}

/// M31 multiplication: (a * b) mod P.
/// Max product: (2^31-2)^2 ≈ 2^62 — fits in u64.
pub fn m31_mul(a: u64, b: u64) -> u64 {
    (a * b) % M31_P
}

/// Reduce a value to M31 range [0, P). Handles edge case val == M31_P → 0.
pub fn m31_reduce(val: u64) -> u64 {
    val % M31_P
}

// ============================================================================
// CM31: M31[i] / (i² + 1) — Complex M31
// ============================================================================

/// Complex M31: a + b·i where i² = -1.
#[derive(Drop, Copy, Serde, Debug, PartialEq)]
pub struct CM31 {
    pub a: u64,
    pub b: u64,
}

pub fn cm31_add(x: CM31, y: CM31) -> CM31 {
    CM31 { a: m31_add(x.a, y.a), b: m31_add(x.b, y.b) }
}

pub fn cm31_sub(x: CM31, y: CM31) -> CM31 {
    CM31 { a: m31_sub(x.a, y.a), b: m31_sub(x.b, y.b) }
}

/// CM31 multiplication: (a+bi)(c+di) = (ac - bd) + (ad + bc)i.
pub fn cm31_mul(x: CM31, y: CM31) -> CM31 {
    let ac = m31_mul(x.a, y.a);
    let bd = m31_mul(x.b, y.b);
    let ad = m31_mul(x.a, y.b);
    let bc = m31_mul(x.b, y.a);
    CM31 { a: m31_sub(ac, bd), b: m31_add(ad, bc) }
}

// ============================================================================
// QM31: CM31[j] / (j² - (2 + i)) — Secure Field (extension degree 4)
// ============================================================================

/// Secure field element: a + b·j where j² = 2 + i.
/// Components: (a.a, a.b, b.a, b.b) — four M31 values.
/// Matches STWO's QM31(CM31(m31_0, m31_1), CM31(m31_2, m31_3)).
#[derive(Drop, Copy, Serde, Debug, PartialEq)]
pub struct QM31 {
    pub a: CM31,
    pub b: CM31,
}

pub fn qm31_new(a0: u64, a1: u64, b0: u64, b1: u64) -> QM31 {
    QM31 { a: CM31 { a: a0, b: a1 }, b: CM31 { a: b0, b: b1 } }
}

pub fn qm31_zero() -> QM31 {
    QM31 { a: CM31 { a: 0, b: 0 }, b: CM31 { a: 0, b: 0 } }
}

pub fn qm31_one() -> QM31 {
    QM31 { a: CM31 { a: 1, b: 0 }, b: CM31 { a: 0, b: 0 } }
}

pub fn qm31_add(x: QM31, y: QM31) -> QM31 {
    QM31 { a: cm31_add(x.a, y.a), b: cm31_add(x.b, y.b) }
}

pub fn qm31_sub(x: QM31, y: QM31) -> QM31 {
    QM31 { a: cm31_sub(x.a, y.a), b: cm31_sub(x.b, y.b) }
}

pub fn qm31_eq(x: QM31, y: QM31) -> bool {
    x.a.a == y.a.a && x.a.b == y.a.b && x.b.a == y.b.a && x.b.b == y.b.b
}

/// QM31 multiplication using Karatsuba over CM31.
///
/// (a + bj)(c + dj) = ac + bd·j² + (ad + bc)j
///
/// j² = 2 + i, so bd·j² = bd·(2+i).
/// If bd = (p + qi), then (2+i)(p+qi) = (2p - q) + (p + 2q)i.
///
/// Karatsuba: ad + bc = (a+b)(c+d) - ac - bd.
pub fn qm31_mul(x: QM31, y: QM31) -> QM31 {
    let ac = cm31_mul(x.a, y.a);
    let bd = cm31_mul(x.b, y.b);

    // bd × (2 + i): (p + qi)(2 + i) = (2p - q) + (p + 2q)i
    let bd_times_irred = CM31 {
        a: m31_sub(m31_add(bd.a, bd.a), bd.b), // 2·p - q
        b: m31_add(bd.a, m31_add(bd.b, bd.b)), // p + 2·q
    };

    // Real part: ac + bd·(2+i)
    let real = cm31_add(ac, bd_times_irred);

    // j-coefficient: (a+b)(c+d) - ac - bd  (Karatsuba)
    let apb = cm31_add(x.a, x.b);
    let cpd = cm31_add(y.a, y.b);
    let apb_cpd = cm31_mul(apb, cpd);
    let j_part = cm31_sub(cm31_sub(apb_cpd, ac), bd);

    QM31 { a: real, b: j_part }
}

// ============================================================================
// Polynomial Evaluation
// ============================================================================

/// Evaluate degree-2 polynomial: p(x) = c0 + c1·x + c2·x².
/// Uses Horner: p(x) = c0 + x·(c1 + x·c2).
pub fn poly_eval_degree2(c0: QM31, c1: QM31, c2: QM31, x: QM31) -> QM31 {
    let inner = qm31_add(c1, qm31_mul(x, c2));
    qm31_add(c0, qm31_mul(x, inner))
}

// ============================================================================
// Utility
// ============================================================================

/// Pack a QM31 into a single felt252.
/// Big-endian 2^31 packing starting from ONE (matching STWO's securefield_to_felt).
/// Component order: [a.a, a.b, b.a, b.b].
pub fn pack_qm31_to_felt(v: QM31) -> felt252 {
    let shift: felt252 = M31_SHIFT;
    let mut result: felt252 = 1;
    result = result * shift + v.a.a.into();
    result = result * shift + v.a.b.into();
    result = result * shift + v.b.a.into();
    result = result * shift + v.b.b.into();
    result
}

/// Compute ceil(log2(n)) for n > 0.
pub fn log2_ceil(n: u32) -> u32 {
    assert!(n > 0, "log2(0) undefined");
    let mut result: u32 = 0;
    let mut val = n - 1;
    loop {
        if val == 0 {
            break;
        }
        val = val / 2;
        result += 1;
    };
    result
}

/// Compute the next power of two >= n.
pub fn next_power_of_two(n: u32) -> u32 {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v = v | (v / 2);
    v = v | (v / 4);
    v = v | (v / 16);
    v = v | (v / 256);
    v = v | (v / 65536);
    v + 1
}

/// Compute 2^n.
pub fn pow2(n: u32) -> u32 {
    let mut result: u32 = 1;
    let mut i: u32 = 0;
    loop {
        if i >= n {
            break;
        }
        result = result * 2;
        i += 1;
    };
    result
}
