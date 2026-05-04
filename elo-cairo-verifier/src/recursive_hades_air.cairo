/// Cairo AIR implementation for the Hades permutation verifier.
///
/// This module constrains that every Hades permutation in the recursive
/// proof trace was computed correctly, using:
///   - VerifyMul252: carry-chain multiplication over 9-bit limbs
///   - Cube252: S-box (x → x³) via two multiplications
///   - MDS: 3×3 linear matrix application
///   - Range checks via LogUp for carry bounds
///
/// The Hades AIR operates on 91-row blocks (one per Hades permutation call):
///   4 full rounds + 83 partial rounds + 4 full rounds
///
/// Connected to the chain AIR via LogUp:
///   - Chain AIR contributes +1 for each (input, output) pair
///   - Hades AIR contributes -1 at block boundaries (verified permutations)
///   - Sum = 0 proves all chain entries have valid Hades permutations

use stwo_verifier_core::fields::qm31::{QM31, QM31Zero, QM31One};
use crate::recursive_air::extract_single_val;

/// Number of 9-bit limbs per felt252.
pub const LIMBS_28: u32 = 28;

/// Number of full rounds per half.
pub const N_FULL_ROUNDS_HALF: u32 = 4;
/// Number of partial rounds.
pub const N_PARTIAL_ROUNDS: u32 = 83;
/// Total rounds per Hades permutation.
pub const N_ROUNDS: u32 = 91;

/// Base for 9-bit limbs.
pub const BASE_512: u32 = 512;

/// Carry shift: 2^19 = 524288 (shifts carry from [-2^19, 2^19) to [0, 2^20)).
pub const CARRY_SHIFT: u32 = 524288;

/// Stark prime P in 9-bit limbs (28 values).
/// P = 2^251 + 17 * 2^192 + 1
/// Computed from felt252_to_9bit_limbs(P) in the Rust side.
/// These values are constant and can be hardcoded.
/// P = 2^251 + 17·2^192 + 1, decomposed into 28 × 9-bit limbs (LSB first).
/// Matches Rust's stark_prime_9bit_limbs() exactly.
/// Nonzero limbs: [0]=1, [21]=136, [27]=256. All others zero.
pub fn stark_prime_limbs() -> Array<u32> {
    let mut limbs: Array<u32> = array![
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 0, 0, 0, 0, 0, 256
    ];
    limbs
}

/// Verify multiplication a × b ≡ c (mod P) using carry-chain constraints.
///
/// For each limb position j (0..28):
///   conv[j] = Σ_{i+l=j} a[i] * b[l]
///   conv[j] = c[j] + k * p[j] + carry[j] * 512 - carry[j-1]
///
/// Each carry must be in [-2^19, 2^19), verified via LogUp.
///
/// Parameters are QM31 evaluations at the OOD point.
pub fn verify_mul_252_at_point(
    a: Span<QM31>,          // 28 limbs
    b: Span<QM31>,          // 28 limbs
    c: Span<QM31>,          // 28 limbs
    k_limbs: Span<QM31>,    // 28 k limbs (full reduction coefficient)
    carries: Span<QM31>,    // 54 carries (55 limbs → 54 inter-limb carries)
    p_limbs: Span<u32>,     // 28 limbs of Stark prime
    is_active: QM31,
) -> Array<QM31> {
    // Returns 55 constraint quotients
    let base: QM31 = qm31_from_u32(BASE_512);
    let zero: QM31 = QM31Zero::zero();
    let mut quotients: Array<QM31> = array![];

    // Check all 55 limb positions: conv[j] - c[j] - kp[j] + carry_in = carry_out * 512
    let mut j: u32 = 0;
    loop {
        if j >= 55 { break; }

        // Convolution: Σ a[i] * b[j-i]
        let mut conv: QM31 = QM31Zero::zero();
        let mut i: u32 = 0;
        loop {
            if i > j { break; }
            if i < 28 && (j - i) < 28 {
                conv = conv + *a[i] * *b[j - i];
            }
            i += 1;
        };

        let c_j: QM31 = if j < 28 { *c[j] } else { zero };

        // kp convolution: Σ k_limbs[l] * p[j-l]
        let mut kp_j: QM31 = QM31Zero::zero();
        let mut l: u32 = 0;
        loop {
            if l > j { break; }
            if l < 28 && (j - l) < 28 {
                kp_j = kp_j + *k_limbs[l] * qm31_from_u32(*p_limbs[j - l]);
            }
            l += 1;
        };

        let carry_in: QM31 = if j == 0 { zero } else { *carries[j - 1] };
        let carry_out: QM31 = if j < 54 { *carries[j] } else { zero };

        let lhs = conv - c_j - kp_j + carry_in;
        let rhs = carry_out * base;
        quotients.append(is_active * (lhs - rhs));

        j += 1;
    };

    quotients
}

/// Verify S-box: x³ = cube_result, via x² = x*x, x³ = x²*x.
pub fn cube_252_at_point(
    x: Span<QM31>,
    x_sq: Span<QM31>,
    x_cubed: Span<QM31>,
    k_sq_limbs: Span<QM31>,     // 28 limbs
    carries_sq: Span<QM31>,     // 54 carries
    k_cube_limbs: Span<QM31>,   // 28 limbs
    carries_cube: Span<QM31>,   // 54 carries
    p_limbs: Span<u32>,
    is_active: QM31,
) -> Array<QM31> {
    let mut quotients: Array<QM31> = array![];

    // x * x = x_sq (55 constraints)
    let q1 = verify_mul_252_at_point(x, x, x_sq, k_sq_limbs, carries_sq, p_limbs, is_active);
    let q1s = q1.span();
    let mut i: u32 = 0;
    loop { if i >= q1s.len() { break; } quotients.append(*q1s[i]); i += 1; };

    // x_sq * x = x_cubed (55 constraints)
    let q2 = verify_mul_252_at_point(x_sq, x, x_cubed, k_cube_limbs, carries_cube, p_limbs, is_active);
    let q2s = q2.span();
    i = 0;
    loop { if i >= q2s.len() { break; } quotients.append(*q2s[i]); i += 1; };

    quotients
}

/// Verify MDS matrix application.
///
/// out[0] = 3*a + b + c
/// out[1] = a - b + c
/// out[2] = a + b - 2*c
///
/// Each row uses carry-chain arithmetic with 27 carries.
pub fn mds_at_point(
    a: Span<QM31>,       // post-sbox state[0], 28 limbs
    b: Span<QM31>,       // post-sbox state[1], 28 limbs
    c: Span<QM31>,       // post-sbox state[2], 28 limbs
    out0: Span<QM31>,
    out1: Span<QM31>,
    out2: Span<QM31>,
    carries0: Span<QM31>,
    carries1: Span<QM31>,
    carries2: Span<QM31>,
    k0: QM31,
    k1: QM31,
    k2: QM31,
    p_limbs: Span<u32>,
    is_active: QM31,
) -> Array<QM31> {
    let base: QM31 = qm31_from_u32(BASE_512);
    let zero: QM31 = QM31Zero::zero();
    let three: QM31 = qm31_from_u32(3);
    // -1 mod M31::P = M31::P - 1 = 2147483646
    let neg_one: QM31 = qm31_from_u32(2147483646);
    // -2 mod M31::P = M31::P - 2 = 2147483645
    let neg_two: QM31 = qm31_from_u32(2147483645);
    let one: QM31 = QM31One::one();

    let mut quotients: Array<QM31> = array![];

    // MDS row 0: 3a + b + c
    let coeffs0: Array<QM31> = array![three, one, one];
    mds_row_at_point(a, b, c, out0, carries0, k0, @coeffs0, p_limbs, base, is_active, ref quotients);

    // MDS row 1: a - b + c
    let coeffs1: Array<QM31> = array![one, neg_one, one];
    mds_row_at_point(a, b, c, out1, carries1, k1, @coeffs1, p_limbs, base, is_active, ref quotients);

    // MDS row 2: a + b - 2c
    let coeffs2: Array<QM31> = array![one, one, neg_two];
    mds_row_at_point(a, b, c, out2, carries2, k2, @coeffs2, p_limbs, base, is_active, ref quotients);

    quotients
}

fn mds_row_at_point(
    a: Span<QM31>,
    b: Span<QM31>,
    c: Span<QM31>,
    out: Span<QM31>,
    carries: Span<QM31>,
    k: QM31,
    coeffs: @Array<QM31>,
    p_limbs: Span<u32>,
    base: QM31,
    is_active: QM31,
    ref quotients: Array<QM31>,
) {
    let zero: QM31 = QM31Zero::zero();
    // 30 limb positions (28 data + 2 overflow for k*P spill)
    let mut j: u32 = 0;
    loop {
        if j >= 30 { break; }
        let p_j: QM31 = if j < 28 { qm31_from_u32(*p_limbs.at(j)) } else { zero };
        let carry_in: QM31 = if j == 0 { zero } else { *carries.at(j - 1) };
        let carry_out: QM31 = if j < 29 { *carries.at(j) } else { zero };

        let a_j: QM31 = if j < 28 { *a.at(j) } else { zero };
        let b_j: QM31 = if j < 28 { *b.at(j) } else { zero };
        let c_j: QM31 = if j < 28 { *c.at(j) } else { zero };
        let out_j: QM31 = if j < 28 { *out.at(j) } else { zero };

        let lhs = *coeffs.at(0) * a_j + *coeffs.at(1) * b_j + *coeffs.at(2) * c_j;
        let rhs = out_j + k * p_j + carry_out * base - carry_in;
        quotients.append(is_active * (lhs - rhs));
        j += 1;
    };
}

/// Convert a u32 to QM31 (embed in real component).
pub fn qm31_from_u32(v: u32) -> QM31 {
    let m = stwo_verifier_core::fields::m31::m31(v);
    let z = stwo_verifier_core::fields::m31::m31(0);
    stwo_verifier_core::fields::qm31::QM31Trait::from_fixed_array([m, z, z, z])
}

/// Evaluate all Hades constraints at the OOD point.
///
/// Reads 1225 columns from trace_vals starting at col_offset (= 56 for chain+hades).
/// Returns an array of constraint quotients (422 total).
///
/// Column layout (1225 columns, matching Rust air.rs):
///   state_before:    3 × 28 = 84
///   sbox_input:      3 × 28 = 84
///   cube_result:     3 × 28 = 84
///   cube_sq:         3 × 28 = 84
///   mul_witness:     3 × 2 × (54 + 28) = 492
///   post_sbox:       3 × 28 = 84
///   mds_result:      3 × 28 = 84
///   mds_carries:     3 × 29 = 87
///   mds_k:           3 × 1 = 3
///   shifted_next:    3 × 28 = 84
///   selectors:       5
///   repack:          50
///   TOTAL:           1225
/// Returns the array of Hades constraint quotients (422 total).
/// Called from recursive_air.cairo to append to the chain quotients
/// before a single Horner accumulation.
pub fn evaluate_hades_constraints_array(
    trace_vals: stwo_verifier_core::ColumnSpan<Span<QM31>>,
    col_offset: u32,
) -> Array<QM31> {
    let p_limbs = stark_prime_limbs();

    let mut pos: u32 = col_offset;

    // ── Read columns ────────────────────────────────────────────

    // state_before: 3 × 28
    let mut state_before: Array<Array<QM31>> = array![];
    let mut elem: u32 = 0;
    loop {
        if elem >= 3 { break; }
        let mut limbs: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop {
            if j >= 28 { break; }
            limbs.append(extract_single_val(trace_vals, pos));
            pos += 1; j += 1;
        };
        state_before.append(limbs);
        elem += 1;
    };

    // sbox_input: 3 × 28
    let mut sbox_input: Array<Array<QM31>> = array![];
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let mut limbs: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop { if j >= 28 { break; } limbs.append(extract_single_val(trace_vals, pos)); pos += 1; j += 1; };
        sbox_input.append(limbs);
        elem += 1;
    };

    // cube_result: 3 × 28
    let mut cube_result: Array<Array<QM31>> = array![];
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let mut limbs: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop { if j >= 28 { break; } limbs.append(extract_single_val(trace_vals, pos)); pos += 1; j += 1; };
        cube_result.append(limbs);
        elem += 1;
    };

    // cube_sq: 3 × 28
    let mut cube_sq: Array<Array<QM31>> = array![];
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let mut limbs: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop { if j >= 28 { break; } limbs.append(extract_single_val(trace_vals, pos)); pos += 1; j += 1; };
        cube_sq.append(limbs);
        elem += 1;
    };

    // mul_witness: 3 elements × 2 muls × (54 carries + 28 k_limbs)
    let mut mul_carries: Array<Array<Array<QM31>>> = array![];  // [elem][mul_idx] → 54 carries
    let mut mul_k_limbs: Array<Array<Array<QM31>>> = array![];  // [elem][mul_idx] → 28 k limbs
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let mut elem_carries: Array<Array<QM31>> = array![];
        let mut elem_k: Array<Array<QM31>> = array![];
        let mut mul_idx: u32 = 0;
        loop {
            if mul_idx >= 2 { break; }
            // 54 carries
            let mut carries: Array<QM31> = array![];
            let mut j: u32 = 0;
            loop { if j >= 54 { break; } carries.append(extract_single_val(trace_vals, pos)); pos += 1; j += 1; };
            elem_carries.append(carries);
            // 28 k_limbs (full reduction coefficient)
            let mut k_vals: Array<QM31> = array![];
            j = 0;
            loop { if j >= 28 { break; } k_vals.append(extract_single_val(trace_vals, pos)); pos += 1; j += 1; };
            elem_k.append(k_vals);
            mul_idx += 1;
        };
        mul_carries.append(elem_carries);
        mul_k_limbs.append(elem_k);
        elem += 1;
    };

    // post_sbox: 3 × 28
    let mut post_sbox: Array<Array<QM31>> = array![];
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let mut limbs: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop { if j >= 28 { break; } limbs.append(extract_single_val(trace_vals, pos)); pos += 1; j += 1; };
        post_sbox.append(limbs);
        elem += 1;
    };

    // mds_result: 3 × 28
    let mut mds_result: Array<Array<QM31>> = array![];
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let mut limbs: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop { if j >= 28 { break; } limbs.append(extract_single_val(trace_vals, pos)); pos += 1; j += 1; };
        mds_result.append(limbs);
        elem += 1;
    };

    // mds_carries: 3 × 29
    let mut mds_carries: Array<Array<QM31>> = array![];
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let mut carries: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop { if j >= 29 { break; } carries.append(extract_single_val(trace_vals, pos)); pos += 1; j += 1; };
        mds_carries.append(carries);
        elem += 1;
    };

    // mds_k: 3
    let mut mds_k: Array<QM31> = array![];
    elem = 0;
    loop { if elem >= 3 { break; } mds_k.append(extract_single_val(trace_vals, pos)); pos += 1; elem += 1; };

    // shifted_next_state: 3 × 28
    let mut shifted_next: Array<Array<QM31>> = array![];
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let mut limbs: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop { if j >= 28 { break; } limbs.append(extract_single_val(trace_vals, pos)); pos += 1; j += 1; };
        shifted_next.append(limbs);
        elem += 1;
    };

    // Selectors
    let is_full_round = extract_single_val(trace_vals, pos); pos += 1;
    let is_real = extract_single_val(trace_vals, pos); pos += 1;
    let is_chain_round = extract_single_val(trace_vals, pos); pos += 1;
    // Skip boundary selectors + repack columns (consumed but not constrained here)
    // _is_first_round, _is_last_round, repack (50 cols) — read to match Rust column count

    // ── Evaluate constraints ────────────────────────────────────

    let one: QM31 = QM31One::one();
    let mut quotients: Array<QM31> = array![];

    // Boolean selectors (2 constraints)
    quotients.append(is_real * (is_real - one));
    quotients.append(is_full_round * (is_full_round - one));

    // S-box cube constraints: 3 elements × 56 constraints = 168
    let sbox_s = @sbox_input;
    let csq_s = @cube_sq;
    let cres_s = @cube_result;
    let mkl_s = @mul_k_limbs;
    let mc_s = @mul_carries;
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let si_span: Span<QM31> = sbox_s.at(elem).span();
        let sq_span: Span<QM31> = csq_s.at(elem).span();
        let cr_span: Span<QM31> = cres_s.at(elem).span();
        let k_sq_span: Span<QM31> = mkl_s.at(elem).at(0).span();  // 28 k limbs for x²
        let k_cu_span: Span<QM31> = mkl_s.at(elem).at(1).span();  // 28 k limbs for x³
        let carries_sq: Span<QM31> = mc_s.at(elem).at(0).span();   // 54 carries
        let carries_cu: Span<QM31> = mc_s.at(elem).at(1).span();   // 54 carries
        let q = cube_252_at_point(
            si_span, sq_span, cr_span,
            k_sq_span, carries_sq,
            k_cu_span, carries_cu,
            p_limbs.span(), is_real,
        );
        let q_span = q.span();
        let mut i: u32 = 0;
        loop { if i >= q_span.len() { break; } quotients.append(*q_span[i]); i += 1; };
        elem += 1;
    };

    // Post-sbox linking: element 2 always cubed (28 constraints)
    let ps_s = @post_sbox;
    let ps2: Span<QM31> = ps_s.at(2).span();
    let cr2: Span<QM31> = cres_s.at(2).span();
    let mut j: u32 = 0;
    loop {
        if j >= 28 { break; }
        quotients.append(*ps2[j] - *cr2[j]);
        j += 1;
    };

    // Post-sbox interpolation: elements 0,1 (56 constraints)
    let si_s = @sbox_input;
    elem = 0;
    loop {
        if elem >= 2 { break; }
        let ps_e: Span<QM31> = ps_s.at(elem).span();
        let cr_e: Span<QM31> = cres_s.at(elem).span();
        let si_e: Span<QM31> = si_s.at(elem).span();
        j = 0;
        loop {
            if j >= 28 { break; }
            let expected = is_full_round * *cr_e[j]
                + (one - is_full_round) * *si_e[j];
            quotients.append(*ps_e[j] - expected);
            j += 1;
        };
        elem += 1;
    };

    // MDS constraints (84 constraints)
    let mr_s = @mds_result;
    let mcar_s = @mds_carries;
    let mk_s2 = @mds_k;
    let mds_q = mds_at_point(
        ps_s.at(0).span(), ps_s.at(1).span(), ps_s.at(2).span(),
        mr_s.at(0).span(), mr_s.at(1).span(), mr_s.at(2).span(),
        mcar_s.at(0).span(), mcar_s.at(1).span(), mcar_s.at(2).span(),
        *mk_s2[0], *mk_s2[1], *mk_s2[2],
        p_limbs.span(), is_real,
    );
    let mds_q_span = mds_q.span();
    let mut i: u32 = 0;
    loop { if i >= mds_q_span.len() { break; } quotients.append(*mds_q_span[i]); i += 1; };

    // Round transition: mds_result == shifted_next_state (84 constraints)
    let sn_s = @shifted_next;
    elem = 0;
    loop {
        if elem >= 3 { break; }
        let mr_e: Span<QM31> = mr_s.at(elem).span();
        let sn_e: Span<QM31> = sn_s.at(elem).span();
        j = 0;
        loop {
            if j >= 28 { break; }
            quotients.append(is_chain_round * (*mr_e[j] - *sn_e[j]));
            j += 1;
        };
        elem += 1;
    };

    quotients
}
