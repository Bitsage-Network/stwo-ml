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
// Field Inversions (M31 → CM31 → QM31)
// ============================================================================

/// M31 exponentiation via square-and-multiply: base^exp mod P.
pub fn m31_pow(base: u64, exp: u64) -> u64 {
    let mut result: u64 = 1;
    let mut b = base;
    let mut e = exp;
    loop {
        if e == 0 {
            break;
        }
        if e % 2 == 1 {
            result = m31_mul(result, b);
        }
        b = m31_mul(b, b);
        e = e / 2;
    };
    result
}

/// M31 multiplicative inverse via Fermat's little theorem: a^{P-2} mod P.
pub fn m31_inverse(a: u64) -> u64 {
    assert!(a != 0, "M31_ZERO_INVERSE");
    m31_pow(a, M31_P - 2)
}

/// CM31 conjugation: conj(a + bi) = a - bi.
pub fn cm31_conj(z: CM31) -> CM31 {
    CM31 { a: z.a, b: m31_sub(0, z.b) }
}

/// CM31 norm squared: |a + bi|^2 = a^2 + b^2 (an M31 value).
pub fn cm31_norm_sq(z: CM31) -> u64 {
    m31_add(m31_mul(z.a, z.a), m31_mul(z.b, z.b))
}

/// CM31 multiplicative inverse: inv(a + bi) = conj(z) / |z|^2.
pub fn cm31_inverse(z: CM31) -> CM31 {
    let ns = cm31_norm_sq(z);
    let inv_ns = m31_inverse(ns);
    CM31 { a: m31_mul(z.a, inv_ns), b: m31_sub(0, m31_mul(z.b, inv_ns)) }
}

/// QM31 multiplicative inverse via conjugation over the j-extension.
///
/// x = a + b·j where j² = 2 + i.
/// x^{-1} = conj(x) / norm(x) where:
///   conj(a + bj) = a - bj
///   norm(a + bj) = a·a - (2+i)·b·b  (a CM31 value)
pub fn qm31_inverse(x: QM31) -> QM31 {
    let a_sq = cm31_mul(x.a, x.a);
    let b_sq = cm31_mul(x.b, x.b);
    // (2+i) * (p+qi) = (2p - q) + (p + 2q)i
    let b_sq_irred = CM31 {
        a: m31_sub(m31_add(b_sq.a, b_sq.a), b_sq.b),
        b: m31_add(b_sq.a, m31_add(b_sq.b, b_sq.b)),
    };
    let norm = cm31_sub(a_sq, b_sq_irred);
    let norm_inv = cm31_inverse(norm);
    let neg_b = CM31 { a: m31_sub(0, x.b.a), b: m31_sub(0, x.b.b) };
    QM31 { a: cm31_mul(x.a, norm_inv), b: cm31_mul(neg_b, norm_inv) }
}

/// QM31 negation: -x.
pub fn qm31_neg(x: QM31) -> QM31 {
    qm31_sub(qm31_zero(), x)
}

/// Lift M31 into QM31: v → (v, 0, 0, 0).
pub fn m31_to_qm31(v: u64) -> QM31 {
    QM31 { a: CM31 { a: v, b: 0 }, b: CM31 { a: 0, b: 0 } }
}

/// Lift u32 into QM31 (for multiplicity values in LogUp).
pub fn qm31_from_u32(v: u32) -> QM31 {
    m31_to_qm31(v.into())
}

/// Montgomery batch inversion: N inversions using only 1 inverse + 3(N-1) muls.
///
/// Algorithm:
///   prefix[i] = v_0 * v_1 * ... * v_i
///   inv_acc   = prefix[N-1]^{-1}
///   Back-sweep: result[i] = prefix[i-1] * inv_acc, then inv_acc *= v_i
pub fn batch_inverse(values: Span<QM31>) -> Array<QM31> {
    let n = values.len();
    assert!(n > 0, "EMPTY_BATCH");

    // Build prefix products
    let mut prefix: Array<QM31> = array![];
    prefix.append(*values.at(0));
    let mut i: u32 = 1;
    loop {
        if i >= n {
            break;
        }
        prefix.append(qm31_mul(*prefix.at(i - 1), *values.at(i)));
        i += 1;
    };

    // Single inversion of total product
    let mut inv_acc = qm31_inverse(*prefix.at(n - 1));

    // Back-sweep: compute inverses in reverse order
    let mut rev: Array<QM31> = array![];
    let mut i: u32 = n;
    loop {
        if i == 0 {
            break;
        }
        i -= 1;
        if i == 0 {
            rev.append(inv_acc);
        } else {
            rev.append(qm31_mul(*prefix.at(i - 1), inv_acc));
        }
        inv_acc = qm31_mul(inv_acc, *values.at(i));
    };

    // Reverse to correct order
    let mut result: Array<QM31> = array![];
    let mut j: u32 = rev.len();
    loop {
        if j == 0 {
            break;
        }
        j -= 1;
        result.append(*rev.at(j));
    };
    result
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
// GKR Field Extensions
// ============================================================================

/// Evaluate degree-3 polynomial: p(x) = c0 + c1·x + c2·x² + c3·x³.
/// Uses Horner: p(x) = c0 + x·(c1 + x·(c2 + x·c3)).
pub fn poly_eval_degree3(c0: QM31, c1: QM31, c2: QM31, c3: QM31, x: QM31) -> QM31 {
    let inner = qm31_add(c2, qm31_mul(x, c3));
    let inner = qm31_add(c1, qm31_mul(x, inner));
    qm31_add(c0, qm31_mul(x, inner))
}

/// Evaluate the Lagrange kernel (eq function) of the boolean hypercube.
/// eq(x, y) = Π_i (x_i · y_i + (1 - x_i) · (1 - y_i))
/// Returns QM31::one() if x == y on {0,1}^n, zero otherwise.
pub fn eq_eval(x: Span<QM31>, y: Span<QM31>) -> QM31 {
    assert!(x.len() == y.len(), "eq_eval: length mismatch");
    let one = qm31_one();
    let mut result = one;
    let mut i: u32 = 0;
    loop {
        if i >= x.len() {
            break;
        }
        let xi = *x.at(i);
        let yi = *y.at(i);
        // term = xi*yi + (1 - xi)*(1 - yi)
        let xi_yi = qm31_mul(xi, yi);
        let one_minus_xi = qm31_sub(one, xi);
        let one_minus_yi = qm31_sub(one, yi);
        let term = qm31_add(xi_yi, qm31_mul(one_minus_xi, one_minus_yi));
        result = qm31_mul(result, term);
        i += 1;
    };
    result
}

/// Fold MLE evaluations: fold(x, v0, v1) = v0 + x · (v1 - v0) = v0·(1-x) + v1·x.
/// Matches STWO's fold_mle_evals(assignment, eval0, eval1).
pub fn fold_mle_eval(x: QM31, v0: QM31, v1: QM31) -> QM31 {
    qm31_add(v0, qm31_mul(x, qm31_sub(v1, v0)))
}

/// Evaluate a multilinear extension (MLE) at an arbitrary point.
///
/// Given 2^n evaluations on the boolean hypercube {0,1}^n and a point r in F^n,
/// computes f(r) by iterated folding:
///   For each variable r_i, fold: current[j] = current[j] + r_i * (current[j+mid] - current[j])
///   After all n variables, a single value remains.
///
/// Matches STWO's evaluate_mle (components/matmul.rs:118-132).
/// Cost: 2n - 1 QM31 operations where n = evals.len().
pub fn evaluate_mle(evals: Span<QM31>, point: Span<QM31>) -> QM31 {
    let n = evals.len();
    assert!(n > 0, "MLE_EMPTY_EVALS");

    // Copy evals into a mutable array
    let mut current: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= n {
            break;
        }
        current.append(*evals.at(i));
        i += 1;
    };

    let mut size = n;
    let mut var_idx: u32 = 0;
    loop {
        if var_idx >= point.len() {
            break;
        }
        let r = *point.at(var_idx);
        let mid = size / 2;

        // Fold in-place: for j in 0..mid, current[j] = current[j] + r*(current[j+mid] - current[j])
        let mut next: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop {
            if j >= mid {
                break;
            }
            let lo = *current.at(j);
            let hi = *current.at(j + mid);
            next.append(qm31_add(lo, qm31_mul(r, qm31_sub(hi, lo))));
            j += 1;
        };

        current = next;
        size = mid;
        var_idx += 1;
    };

    assert!(size == 1, "MLE_FOLD_INCOMPLETE");
    *current.at(0)
}

/// Evaluate an MLE at `point` by building an eq-table, then dot-product with data
/// read directly from a felt252 IO span. Eliminates the data QM31 array entirely
/// (saves 8192 appends) and replaces evaluate_mle's copy+fold (saves another 8192
/// appends + avoids the initial copy in evaluate_mle).
///
/// For batch-size-1 (padded_rows == 1): data is flat, `data_len` non-zero entries
/// followed by `padded_len - data_len` zeros. We skip padding in the dot product.
///
/// eq_table[i] = eq(binary(i), point) = prod_j((1-r_j)(1-b_j) + r_j·b_j)
/// Built incrementally (MSB-first to match evaluate_mle folding order):
///   Start: [1]
///   For each variable r_j:
///     new[2k]   = old[k] * (1 - r_j)
///     new[2k+1] = old[k] * r_j
///
/// Total: ~16K appends (eq build) vs ~25K appends (old approach)
pub fn evaluate_mle_from_io_span_1row(
    io_span: Span<felt252>,
    data_off: u32,
    data_len: u32,
    padded_len: u32,
    point: Span<QM31>,
) -> QM31 {
    let n_vars = point.len();
    assert!(padded_len > 0, "EQ_TABLE_EMPTY");
    let _ = padded_len; // validated by caller

    // Build eq table incrementally (MSB-first to match evaluate_mle folding order)
    let mut eq_table: Array<QM31> = array![qm31_one()];
    let mut var_idx: u32 = 0;
    loop {
        if var_idx >= n_vars {
            break;
        }
        let r = *point.at(var_idx);
        let one_minus_r = qm31_sub(qm31_one(), r);
        let old_len = eq_table.len();
        let old_span = eq_table.span();

        let mut new_table: Array<QM31> = array![];
        let mut k: u32 = 0;
        loop {
            if k >= old_len {
                break;
            }
            let e = *old_span.at(k);
            new_table.append(qm31_mul(e, one_minus_r));
            new_table.append(qm31_mul(e, r));
            k += 1;
        };
        eq_table = new_table;
        var_idx += 1;
    };

    // Dot product: sum(data[i] * eq_table[i]) for i in 0..data_len
    // (padding entries are zero, so their eq_table contribution is skipped)
    let eq_span = eq_table.span();
    let mut acc = qm31_zero();
    let mut i: u32 = 0;
    loop {
        if i >= data_len {
            break;
        }
        let v: u256 = (*io_span.at(data_off + i)).into();
        let v_u64: u64 = v.try_into().unwrap();
        if v_u64 != 0 {
            acc = qm31_add(acc, qm31_mul(m31_to_qm31(v_u64), *eq_span.at(i)));
        }
        i += 1;
    };
    acc
}

/// General multi-row version: handles row-major data with row/col padding.
/// data layout: row-major, data_rows x data_cols entries starting at data_off.
/// Padded to padded_rows x padded_cols (both powers of 2).
/// point has log2(padded_rows) + log2(padded_cols) variables.
pub fn evaluate_mle_from_io_span_2d(
    io_span: Span<felt252>,
    data_off: u32,
    data_rows: u32,
    data_cols: u32,
    padded_rows: u32,
    padded_cols: u32,
    point: Span<QM31>,
) -> QM31 {
    let padded_len = padded_rows * padded_cols;
    let n_vars = point.len();
    assert!(padded_len > 0, "EQ_TABLE_EMPTY");

    // Build eq table (same as 1row version)
    let mut eq_table: Array<QM31> = array![qm31_one()];
    let mut var_idx: u32 = 0;
    loop {
        if var_idx >= n_vars {
            break;
        }
        let r = *point.at(var_idx);
        let one_minus_r = qm31_sub(qm31_one(), r);
        let old_len = eq_table.len();
        let old_span = eq_table.span();

        let mut new_table: Array<QM31> = array![];
        let mut k: u32 = 0;
        loop {
            if k >= old_len {
                break;
            }
            let e = *old_span.at(k);
            new_table.append(qm31_mul(e, one_minus_r));
            new_table.append(qm31_mul(e, r));
            k += 1;
        };
        eq_table = new_table;
        var_idx += 1;
    };

    // Dot product over non-padded entries only
    let eq_span = eq_table.span();
    let mut acc = qm31_zero();
    let mut row: u32 = 0;
    loop {
        if row >= data_rows {
            break;
        }
        let mut col: u32 = 0;
        loop {
            if col >= data_cols {
                break;
            }
            let src_idx: u32 = row * data_cols + col;
            let dst_idx: u32 = row * padded_cols + col;
            let v: u256 = (*io_span.at(data_off + src_idx)).into();
            let v_u64: u64 = v.try_into().unwrap();
            if v_u64 != 0 {
                acc = qm31_add(acc, qm31_mul(m31_to_qm31(v_u64), *eq_span.at(dst_idx)));
            }
            col += 1;
        };
        row += 1;
    };
    acc
}

/// Evaluate an MLE at `point` directly from PACKED felt252s, without any intermediate
/// unpacking array. Extracts 8 M31 values per packed felt and multiplies by eq_table
/// entries in a single pass.
///
/// This eliminates the ~10K-entry unpacked array entirely (saves ~2.5M Cairo steps
/// from u128 unpacking + array appends that would otherwise be needed).
///
/// Parameters:
/// - packed_felts: Full packed IO span (8 M31 values per felt252)
/// - m31_start: Starting M31 position within the packed data (e.g., 3 for input data)
/// - data_len: Number of M31 data values to evaluate
/// - padded_len: Power-of-2 padded length for MLE
/// - point: MLE evaluation point (log2(padded_len) QM31 values)
pub fn evaluate_mle_from_packed_1row(
    packed_felts: Span<felt252>,
    m31_start: u32,
    data_len: u32,
    padded_len: u32,
    point: Span<QM31>,
) -> QM31 {
    let n_vars = point.len();
    assert!(padded_len > 0, "EQ_TABLE_EMPTY");

    // Build eq table
    let mut eq_table: Array<QM31> = array![qm31_one()];
    let mut var_idx: u32 = 0;
    loop {
        if var_idx >= n_vars {
            break;
        }
        let r = *point.at(var_idx);
        let one_minus_r = qm31_sub(qm31_one(), r);
        let old_len = eq_table.len();
        let old_span = eq_table.span();
        let mut new_table: Array<QM31> = array![];
        let mut k: u32 = 0;
        loop {
            if k >= old_len {
                break;
            }
            let e = *old_span.at(k);
            new_table.append(qm31_mul(e, one_minus_r));
            new_table.append(qm31_mul(e, r));
            k += 1;
        };
        eq_table = new_table;
        var_idx += 1;
    };
    let eq_span = eq_table.span();

    // Compute packed felt index and offset for m31_start
    let mut pi: u32 = m31_start / 8;
    let skip_in_first: u32 = m31_start % 8;

    let mut acc = qm31_zero();
    let mut data_idx: u32 = 0;

    loop {
        if data_idx >= data_len {
            break;
        }
        let val: u256 = (*packed_felts.at(pi)).into();
        let mut rem_lo: u128 = val.low;
        let mut rem_hi: u128 = val.high;

        // Determine how many M31 values to skip in this packed felt
        let skip_count = if pi == m31_start / 8 { skip_in_first } else { 0 };

        // Extract all 8 M31 values from this packed felt
        // But skip the first `skip_count` and stop when data_idx reaches data_len
        let mut pos_in_felt: u32 = 0;

        // Values 0-3 from lo limb
        let mut vi: u32 = 0;
        loop {
            if vi >= 4 {
                break;
            }
            let q: u128 = rem_lo / 0x80000000;
            let v: u128 = rem_lo - q * 0x80000000;
            if pos_in_felt >= skip_count && data_idx < data_len {
                if v != 0 {
                    let v_u64: u64 = v.try_into().unwrap();
                    acc = qm31_add(acc, qm31_mul(m31_to_qm31(v_u64), *eq_span.at(data_idx)));
                }
                data_idx += 1;
            }
            rem_lo = q;
            pos_in_felt += 1;
            vi += 1;
        };

        // Value 4 straddles lo/hi
        let hi_q4: u128 = rem_hi / 0x8000000;
        let hi_low27: u128 = rem_hi - hi_q4 * 0x8000000;
        let v4: u128 = rem_lo | (hi_low27 * 16);
        let v4_masked: u128 = v4 & 0x7FFFFFFF;
        if pos_in_felt >= skip_count && data_idx < data_len {
            if v4_masked != 0 {
                let v4_u64: u64 = v4_masked.try_into().unwrap();
                acc = qm31_add(acc, qm31_mul(m31_to_qm31(v4_u64), *eq_span.at(data_idx)));
            }
            data_idx += 1;
        }
        rem_hi = hi_q4;
        pos_in_felt += 1;

        // Values 5-6 from hi limb
        vi = 0;
        loop {
            if vi >= 2 {
                break;
            }
            let q: u128 = rem_hi / 0x80000000;
            let v: u128 = rem_hi - q * 0x80000000;
            if pos_in_felt >= skip_count && data_idx < data_len {
                if v != 0 {
                    let v_u64: u64 = v.try_into().unwrap();
                    acc = qm31_add(acc, qm31_mul(m31_to_qm31(v_u64), *eq_span.at(data_idx)));
                }
                data_idx += 1;
            }
            rem_hi = q;
            pos_in_felt += 1;
            vi += 1;
        };

        // Value 7: remaining bits
        if pos_in_felt >= skip_count && data_idx < data_len {
            let v7: u128 = rem_hi & 0x7FFFFFFF;
            if v7 != 0 {
                let v7_u64: u64 = v7.try_into().unwrap();
                acc = qm31_add(acc, qm31_mul(m31_to_qm31(v7_u64), *eq_span.at(data_idx)));
            }
            data_idx += 1;
        }

        pi += 1;
    };
    acc
}

/// Embed M31 values (as u64) into QM31 and pad to a target power-of-2 length.
///
/// Each M31 value v becomes QM31(v, 0, 0, 0). Padding uses QM31::zero().
/// Used to reconstruct an MLE from raw calldata for on-chain input/output verification.
pub fn pad_and_embed_m31s(vals: Span<u64>, target_len: u32) -> Array<QM31> {
    assert!(target_len >= vals.len(), "MLE_TARGET_TOO_SMALL");
    let mut result: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= vals.len() {
            break;
        }
        result.append(m31_to_qm31(*vals.at(i)));
        i += 1;
    };
    // Pad with zeros to target_len
    loop {
        if result.len() >= target_len {
            break;
        }
        result.append(qm31_zero());
    };
    result
}

/// Random linear combination: v_0 + alpha·v_1 + ... + alpha^(n-1)·v_{n-1}.
/// Uses Horner evaluation (matches STWO's horner_eval on values).
pub fn random_linear_combination(values: Span<QM31>, alpha: QM31) -> QM31 {
    if values.len() == 0 {
        return qm31_zero();
    }
    // Horner: fold from right: acc = acc * alpha + v_i
    let mut i: u32 = values.len();
    let mut acc = qm31_zero();
    loop {
        if i == 0 {
            break;
        }
        i -= 1;
        acc = qm31_add(qm31_mul(acc, alpha), *values.at(i));
    };
    acc
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

/// Unpack a QM31 from a single felt252 (reverse of pack_qm31_to_felt).
/// Layout: [1-bit sentinel | 31-bit a.a | 31-bit a.b | 31-bit b.a | 31-bit b.b]
pub fn unpack_qm31_from_felt(fe: felt252) -> QM31 {
    let val: u256 = fe.into();
    let mask: u256 = 0x7FFFFFFF; // 2^31 - 1
    let bb: u64 = (val & mask).try_into().unwrap();
    let ba: u64 = ((val / 0x80000000) & mask).try_into().unwrap();
    let ab: u64 = ((val / 0x4000000000000000) & mask).try_into().unwrap();
    let aa: u64 = ((val / 0x200000000000000000000000) & mask).try_into().unwrap();
    QM31 { a: CM31 { a: aa, b: ab }, b: CM31 { a: ba, b: bb } }
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
