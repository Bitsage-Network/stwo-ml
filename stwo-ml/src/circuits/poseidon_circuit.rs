//! Poseidon2-M31 proving circuit for VM31 privacy protocol.
//!
//! Proves correct execution of N parallel Poseidon2 permutations using a
//! layered sumcheck protocol matching the GKR framework.
//!
//! # Architecture
//!
//! Each Poseidon2 permutation has 22 rounds (4 full + 14 partial + 4 full).
//! The proof walks backwards from output to input, reducing claims through:
//!
//! - **Linear layer**: degree-2 sumcheck over the 16-element state dimension
//!   (the round matrix is public, so the verifier can evaluate it directly).
//!
//! - **S-box (x^5)**: Decomposed into 3 degree-3 eq-sumchecks:
//!   1. x² = after_add · after_add
//!   2. x⁴ = x² · x²
//!   3. x⁵ = x⁴ · after_add
//!
//! - **Partial rounds**: S-box applies only to element 0. The proof decomposes
//!   `after_sbox` into `after_add` + delta correction on element 0.
//!
//! # Usage
//!
//! ```ignore
//! let inputs: Vec<[M31; 16]> = /* permutation inputs */;
//! let mut channel = PoseidonChannel::default();
//! let proof = prove_poseidon2_batch(&inputs, &mut channel);
//!
//! let mut v_channel = PoseidonChannel::default();
//! let outputs: Vec<[M31; 16]> = inputs.iter().map(|s| { let mut t = *s; poseidon2_permutation(&mut t); t }).collect();
//! assert!(verify_poseidon2_batch(&proof, &inputs, &outputs, &mut v_channel).is_ok());
//! ```

use stwo::core::fields::m31::BaseField as M31;
use stwo::core::fields::qm31::QM31;
use stwo::core::fields::FieldExpOps;
use num_traits::{One, Zero};

use crate::crypto::poseidon2_m31::{
    poseidon2_permutation, apply_external_round_matrix, apply_internal_round_matrix,
    get_round_constants, INTERNAL_DIAG_U32,
    STATE_WIDTH, N_HALF_FULL_ROUNDS, N_PARTIAL_ROUNDS, N_FULL_ROUNDS,
};
use crate::crypto::poseidon_channel::PoseidonChannel;
use crate::components::matmul::RoundPoly;
use crate::gkr::types::RoundPolyDeg3;

pub type SecureField = QM31;

// Total rounds in the Poseidon2 permutation
const N_ROUNDS: usize = N_FULL_ROUNDS + N_PARTIAL_ROUNDS; // 22

// Number of element-dimension variables
const LOG_STATE: usize = 4; // log2(16)

// ═══════════════════════════════ Proof Types ═══════════════════════════════

/// Per-round proof in the Poseidon2 circuit.
#[derive(Debug, Clone)]
pub struct Poseidon2RoundProof {
    /// Whether this is a full round or partial round.
    pub is_full_round: bool,

    // ── Linear layer sumcheck (degree-2, LOG_STATE rounds) ──
    pub linear_polys: Vec<RoundPoly>,
    pub sbox_out_eval: SecureField,

    // ── S-box mul check 3: x5 = x4 · after_add (degree-3 eq-sumcheck) ──
    pub mul3_polys: Vec<RoundPolyDeg3>,
    pub x4_eval: SecureField,
    pub after_add_eval_from_mul3: SecureField,

    // ── S-box mul check 2: x4 = x2² (degree-3 eq-sumcheck) ──
    pub mul2_polys: Vec<RoundPolyDeg3>,
    pub x2_eval: SecureField,

    // ── S-box mul check 1: x2 = after_add² (degree-3 eq-sumcheck) ──
    pub mul1_polys: Vec<RoundPolyDeg3>,
    pub after_add_eval_from_mul1: SecureField,

    // ── Dual-claim combination sumcheck (degree-2) ──
    pub combine_polys: Vec<RoundPoly>,
    pub combined_eval: SecureField,

    // ── Partial round decomposition ──
    /// For partial rounds: x5_0 eval at the instance point (S-box output for element 0).
    pub partial_x5_0_eval: Option<SecureField>,
    /// For partial rounds: after_add eval at the full point.
    pub partial_after_add_eval: Option<SecureField>,
    /// For partial rounds: after_add_0 eval at the instance point (for S-box bridge).
    pub partial_after_add_0_eval: Option<SecureField>,
}

/// Complete proof for a batch of N Poseidon2 permutations.
#[derive(Debug, Clone)]
pub struct Poseidon2BatchProof {
    /// Per-round proofs, from last round to first (output → input order).
    pub round_proofs: Vec<Poseidon2RoundProof>,
    /// Final claim value on the input state MLE.
    pub input_eval: SecureField,
}

/// Error type for Poseidon2 proving/verification.
#[derive(Debug, thiserror::Error)]
pub enum Poseidon2Error {
    #[error("sumcheck failed at round {round}, step {step}: {reason}")]
    SumcheckFailed { round: usize, step: String, reason: String },

    #[error("final check failed at round {round}: {reason}")]
    FinalCheckFailed { round: usize, reason: String },

    #[error("input mismatch: {0}")]
    InputMismatch(String),
}

// ═══════════════════════════════ Trace Generation ═══════════════════════════

/// Full execution trace for N parallel Poseidon2 permutations.
///
/// Records all intermediate states and S-box auxiliary witnesses needed for proving.
pub struct Poseidon2Trace {
    pub n: usize,
    /// Round boundary states: `states[r]` is the state AFTER round r.
    /// `states[0]` = inputs, `states[N_ROUNDS]` = outputs.
    pub states: Vec<Vec<[M31; STATE_WIDTH]>>,
    /// After adding round constants (before S-box).
    pub after_adds: Vec<Vec<[M31; STATE_WIDTH]>>,
    /// S-box intermediates x² (element-wise).
    pub x2s: Vec<Vec<[M31; STATE_WIDTH]>>,
    /// S-box intermediates x⁴ (element-wise).
    pub x4s: Vec<Vec<[M31; STATE_WIDTH]>>,
    /// S-box output x⁵ (element-wise).
    pub x5s: Vec<Vec<[M31; STATE_WIDTH]>>,
    /// After S-box application (for partial rounds, only element 0 is S-boxed).
    pub after_sboxes: Vec<Vec<[M31; STATE_WIDTH]>>,
    /// Whether each round is a full round (true) or partial round (false).
    pub is_full: Vec<bool>,
    /// Round index within the full/partial phase (for looking up round constants).
    pub rc_indices: Vec<usize>,
}

/// Generate the complete execution trace for N parallel Poseidon2 permutations.
pub fn generate_trace(inputs: &[[M31; STATE_WIDTH]]) -> Poseidon2Trace {
    let n = inputs.len();
    assert!(n > 0 && n.is_power_of_two(), "batch size must be a positive power of 2");
    let rc = get_round_constants();

    let mut states = Vec::with_capacity(N_ROUNDS + 1);
    let mut after_adds = Vec::with_capacity(N_ROUNDS);
    let mut x2s = Vec::with_capacity(N_ROUNDS);
    let mut x4s = Vec::with_capacity(N_ROUNDS);
    let mut x5s = Vec::with_capacity(N_ROUNDS);
    let mut after_sboxes = Vec::with_capacity(N_ROUNDS);
    let mut is_full = Vec::with_capacity(N_ROUNDS);
    let mut rc_indices = Vec::with_capacity(N_ROUNDS);

    states.push(inputs.to_vec());
    let mut current = inputs.to_vec();

    // First half: 4 full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        let (next, aa, sq, ft, fifth, asbox) =
            execute_full_round(&current, &rc.external[round]);
        states.push(next.clone());
        after_adds.push(aa);
        x2s.push(sq);
        x4s.push(ft);
        x5s.push(fifth);
        after_sboxes.push(asbox);
        is_full.push(true);
        rc_indices.push(round);
        current = next;
    }

    // Partial rounds: 14 rounds
    for round in 0..N_PARTIAL_ROUNDS {
        let (next, aa, sq, ft, fifth, asbox) =
            execute_partial_round(&current, rc.internal[round]);
        states.push(next.clone());
        after_adds.push(aa);
        x2s.push(sq);
        x4s.push(ft);
        x5s.push(fifth);
        after_sboxes.push(asbox);
        is_full.push(false);
        rc_indices.push(round);
        current = next;
    }

    // Second half: 4 full rounds
    for round in 0..N_HALF_FULL_ROUNDS {
        let rc_idx = round + N_HALF_FULL_ROUNDS;
        let (next, aa, sq, ft, fifth, asbox) =
            execute_full_round(&current, &rc.external[rc_idx]);
        states.push(next.clone());
        after_adds.push(aa);
        x2s.push(sq);
        x4s.push(ft);
        x5s.push(fifth);
        after_sboxes.push(asbox);
        is_full.push(true);
        rc_indices.push(rc_idx);
        current = next;
    }

    Poseidon2Trace {
        n, states, after_adds, x2s, x4s, x5s, after_sboxes, is_full, rc_indices,
    }
}

fn execute_full_round(
    states: &[[M31; STATE_WIDTH]],
    rc: &[M31; STATE_WIDTH],
) -> (
    Vec<[M31; STATE_WIDTH]>, // next state (after linear layer)
    Vec<[M31; STATE_WIDTH]>, // after_add
    Vec<[M31; STATE_WIDTH]>, // x2
    Vec<[M31; STATE_WIDTH]>, // x4
    Vec<[M31; STATE_WIDTH]>, // x5
    Vec<[M31; STATE_WIDTH]>, // after_sbox (= x5 for full rounds)
) {
    let n = states.len();
    let mut next = Vec::with_capacity(n);
    let mut after_add_all = Vec::with_capacity(n);
    let mut x2_all = Vec::with_capacity(n);
    let mut x4_all = Vec::with_capacity(n);
    let mut x5_all = Vec::with_capacity(n);

    for state in states {
        // Add round constants
        let mut aa = *state;
        for j in 0..STATE_WIDTH {
            aa[j] += rc[j];
        }

        // S-box on all elements
        let mut x2 = [M31::from_u32_unchecked(0); STATE_WIDTH];
        let mut x4 = [M31::from_u32_unchecked(0); STATE_WIDTH];
        let mut x5 = [M31::from_u32_unchecked(0); STATE_WIDTH];
        for j in 0..STATE_WIDTH {
            x2[j] = aa[j] * aa[j];
            x4[j] = x2[j] * x2[j];
            x5[j] = x4[j] * aa[j];
        }

        // Linear layer
        let mut out = x5;
        apply_external_round_matrix(&mut out);

        after_add_all.push(aa);
        x2_all.push(x2);
        x4_all.push(x4);
        x5_all.push(x5);
        next.push(out);
    }

    let after_sbox = x5_all.clone();
    (next, after_add_all, x2_all, x4_all, x5_all, after_sbox)
}

fn execute_partial_round(
    states: &[[M31; STATE_WIDTH]],
    rc_0: M31,
) -> (
    Vec<[M31; STATE_WIDTH]>,
    Vec<[M31; STATE_WIDTH]>,
    Vec<[M31; STATE_WIDTH]>,
    Vec<[M31; STATE_WIDTH]>,
    Vec<[M31; STATE_WIDTH]>,
    Vec<[M31; STATE_WIDTH]>,
) {
    let n = states.len();
    let mut next = Vec::with_capacity(n);
    let mut after_add_all = Vec::with_capacity(n);
    let mut x2_all = Vec::with_capacity(n);
    let mut x4_all = Vec::with_capacity(n);
    let mut x5_all = Vec::with_capacity(n);
    let mut after_sbox_all = Vec::with_capacity(n);

    for state in states {
        // Add round constant to element 0 only
        let mut aa = *state;
        aa[0] += rc_0;

        // S-box on element 0 only
        let mut x2 = [M31::from_u32_unchecked(0); STATE_WIDTH];
        let mut x4 = [M31::from_u32_unchecked(0); STATE_WIDTH];
        let mut x5 = [M31::from_u32_unchecked(0); STATE_WIDTH];
        x2[0] = aa[0] * aa[0];
        x4[0] = x2[0] * x2[0];
        x5[0] = x4[0] * aa[0];

        // after_sbox: element 0 = x5[0], elements 1-15 = aa[1-15]
        let mut asbox = aa;
        asbox[0] = x5[0];

        // Internal linear layer
        let mut out = asbox;
        apply_internal_round_matrix(&mut out);

        after_add_all.push(aa);
        x2_all.push(x2);
        x4_all.push(x4);
        x5_all.push(x5);
        after_sbox_all.push(asbox);
        next.push(out);
    }

    (next, after_add_all, x2_all, x4_all, x5_all, after_sbox_all)
}

/// Verify a trace against the reference implementation (direct, O(N) check).
pub fn verify_trace_direct(trace: &Poseidon2Trace) -> bool {
    for i in 0..trace.n {
        let mut state = trace.states[0][i];
        poseidon2_permutation(&mut state);
        if state != trace.states[N_ROUNDS][i] {
            return false;
        }
    }
    true
}

// ═══════════════════════════════ MLE Utilities ═══════════════════════════════

/// Flatten N arrays of STATE_WIDTH into a single flat vector for MLE construction.
fn flatten_states(states: &[[M31; STATE_WIDTH]]) -> Vec<SecureField> {
    states
        .iter()
        .flat_map(|s| s.iter().map(|&v| SecureField::from(v)))
        .collect()
}

/// Flatten N values (element-0 only) into a single vector.
fn flatten_elem0(states: &[[M31; STATE_WIDTH]]) -> Vec<SecureField> {
    states
        .iter()
        .map(|s| SecureField::from(s[0]))
        .collect()
}

/// Evaluate MLE at a point. evals[i] = f(binary_repr(i)).
fn evaluate_mle(evals: &[SecureField], point: &[SecureField]) -> SecureField {
    assert_eq!(evals.len(), 1 << point.len());
    let mut current = evals.to_vec();
    let mut size = current.len();
    for &r in point.iter() {
        let mid = size / 2;
        for i in 0..mid {
            current[i] = current[i] + r * (current[mid + i] - current[i]);
        }
        size = mid;
    }
    current[0]
}

/// Fold MLE at a challenge: new[i] = (1-r)*old[i] + r*old[mid+i].
fn fold_mle(vals: &[SecureField], r: SecureField, mid: usize) -> Vec<SecureField> {
    let one_minus_r = SecureField::one() - r;
    (0..mid)
        .map(|i| one_minus_r * vals[i] + r * vals[mid + i])
        .collect()
}

/// Build eq(r, x) evaluations for all x ∈ {0,1}^n via tensor product.
fn build_eq_evals(r: &[SecureField]) -> Vec<SecureField> {
    let n = r.len();
    let size = 1 << n;
    let mut evals = vec![SecureField::one(); size];
    for (i, &r_i) in r.iter().enumerate() {
        let half = 1 << i;
        for j in (0..half).rev() {
            evals[2 * j + 1] = evals[j] * r_i;
            evals[2 * j] = evals[j] * (SecureField::one() - r_i);
        }
    }
    evals
}

/// Compute eq(a, b) = Π_i (a_i · b_i + (1-a_i)(1-b_i)).
fn compute_eq_eval(a: &[SecureField], b: &[SecureField]) -> SecureField {
    assert_eq!(a.len(), b.len());
    let mut result = SecureField::one();
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        result = result * (ai * bi + (SecureField::one() - ai) * (SecureField::one() - bi));
    }
    result
}

/// Mix a SecureField into the Fiat-Shamir channel.
fn mix_sf(channel: &mut PoseidonChannel, v: SecureField) {
    channel.mix_u64(v.0 .0 .0 as u64);
    channel.mix_u64(v.0 .1 .0 as u64);
    channel.mix_u64(v.1 .0 .0 as u64);
    channel.mix_u64(v.1 .1 .0 as u64);
}

// ═══════════════════════════ Matrix MLE Evaluation ══════════════════════════

/// Build the MLE evaluations for the external round matrix (16×16).
/// Matrix is circ(2*M4, M4, M4, M4) applied to a state vector.
fn build_external_matrix_mle() -> Vec<SecureField> {
    let mut evals = vec![SecureField::zero(); STATE_WIDTH * STATE_WIDTH];
    // For each (row, col), compute M_ext[row][col]
    for row in 0..STATE_WIDTH {
        let mut basis = [M31::from_u32_unchecked(0); STATE_WIDTH];
        basis[row] = M31::from_u32_unchecked(1);
        // The transpose of apply_external_round_matrix gives us the row
        // Actually, we need M[row][col] = output[row] when input = e_col
        for col in 0..STATE_WIDTH {
            let mut input = [M31::from_u32_unchecked(0); STATE_WIDTH];
            input[col] = M31::from_u32_unchecked(1);
            apply_external_round_matrix(&mut input);
            evals[row * STATE_WIDTH + col] = SecureField::from(input[row]);
        }
    }
    evals
}

/// Build the MLE evaluations for the internal round matrix (16×16).
/// Matrix is M_I = J + diag(INTERNAL_DIAG), where J is all-ones.
fn build_internal_matrix_mle() -> Vec<SecureField> {
    let mut evals = vec![SecureField::zero(); STATE_WIDTH * STATE_WIDTH];
    for row in 0..STATE_WIDTH {
        for col in 0..STATE_WIDTH {
            let mut val = M31::from_u32_unchecked(1); // J component (all 1s)
            if row == col {
                val += M31::from_u32_unchecked(INTERNAL_DIAG_U32[row]);
            }
            evals[row * STATE_WIDTH + col] = SecureField::from(val);
        }
    }
    evals
}

/// Evaluate M(r_row, r_col) for a 16×16 matrix stored as flat MLE.
fn eval_matrix_mle(matrix_flat: &[SecureField], r_row: &[SecureField], r_col: &[SecureField]) -> SecureField {
    assert_eq!(r_row.len(), LOG_STATE);
    assert_eq!(r_col.len(), LOG_STATE);
    // Matrix MLE: Ṽ_M(r_row, r_col) = Σ_{i,j} eq(r_row,i) eq(r_col,j) M[i][j]
    let eq_row = build_eq_evals(r_row);
    let eq_col = build_eq_evals(r_col);
    let mut result = SecureField::zero();
    for i in 0..STATE_WIDTH {
        for j in 0..STATE_WIDTH {
            result += eq_row[i] * eq_col[j] * matrix_flat[i * STATE_WIDTH + j];
        }
    }
    result
}

/// Evaluate the round constant MLE at an element point.
/// RC is constant across instances, so its MLE only depends on the element variables.
fn eval_rc_mle(rc: &[M31; STATE_WIDTH], r_elem: &[SecureField]) -> SecureField {
    assert_eq!(r_elem.len(), LOG_STATE);
    let eq = build_eq_evals(r_elem);
    let mut result = SecureField::zero();
    for j in 0..STATE_WIDTH {
        result += eq[j] * SecureField::from(rc[j]);
    }
    result
}

/// Evaluate eq(r_elem, 0) = Π_j (1 - r_elem_j). This is the MLE selector for element 0.
fn eval_eq_zero(r_elem: &[SecureField]) -> SecureField {
    let mut result = SecureField::one();
    for &r in r_elem {
        result = result * (SecureField::one() - r);
    }
    result
}

// ══════════════════════════ Degree-2 Sumcheck (Linear Layer) ══════════════════

/// Linear layer sumcheck: proves output_eval = Σ_k M(r_elem, k) · x5(r_inst, k).
///
/// Summand: f(k) = M(r_elem, k) · x5(r_inst, k)
/// Both factors are degree-1 in each k-variable → degree-2 round polynomial.
///
/// Returns (round_polys, sumcheck_challenges, x5_eval_at_final_point).
fn prove_linear_sumcheck(
    matrix_flat: &[SecureField],
    sbox_out: &[SecureField], // N×16 flat MLE
    r_inst: &[SecureField],
    r_elem: &[SecureField],
    claimed_sum: SecureField,
    channel: &mut PoseidonChannel,
) -> (Vec<RoundPoly>, Vec<SecureField>, SecureField) {
    // Restrict sbox_out MLE to r_inst: sbox_out_restricted[k] = Ṽ_{sbox_out}(r_inst, k)
    // The MLE layout: sbox_out[inst * 16 + elem], so first LOG_STATE vars = elem, next = inst.
    // We restrict the instance variables (the UPPER portion), leaving element variables free.
    let log_n = sbox_out.len().ilog2() as usize - LOG_STATE;
    let sbox_out_at_r_inst = restrict_upper_vars(sbox_out, r_inst, log_n);
    assert_eq!(sbox_out_at_r_inst.len(), STATE_WIDTH);

    // Build M(r_elem, k) for k ∈ {0,...,15}: restrict matrix MLE at r_elem for row
    let eq_row = build_eq_evals(r_elem);
    let mut m_at_r_elem = vec![SecureField::zero(); STATE_WIDTH];
    for k in 0..STATE_WIDTH {
        for i in 0..STATE_WIDTH {
            m_at_r_elem[k] += eq_row[i] * matrix_flat[i * STATE_WIDTH + k];
        }
    }

    // Mix tag + claim
    channel.mix_u64(0x4C494E as u64); // "LIN" tag
    mix_sf(channel, claimed_sum);

    let mut f_m = m_at_r_elem;
    let mut f_s = sbox_out_at_r_inst;
    let mut polys = Vec::with_capacity(LOG_STATE);
    let mut challenges = Vec::with_capacity(LOG_STATE);
    let mut cur_n = STATE_WIDTH;

    for _ in 0..LOG_STATE {
        let mid = cur_n / 2;

        // Evaluate at t=0, 1, 2
        let s0 = compute_product_sum_at_t(&f_m, &f_s, mid, SecureField::zero());
        let s1 = compute_product_sum_at_t(&f_m, &f_s, mid, SecureField::one());
        let two = SecureField::from(M31::from(2u32));
        let s2 = compute_product_sum_at_t(&f_m, &f_s, mid, two);

        // Lagrange interpolation: p(t) = c0 + c1*t + c2*t²
        let inv2 = two.inverse();
        let c0 = s0;
        let c1 = (SecureField::from(M31::from(4u32)) * s1 - s2 - SecureField::from(M31::from(3u32)) * s0) * inv2;
        let c2 = (s2 - SecureField::from(M31::from(2u32)) * s1 + s0) * inv2;

        let rp = RoundPoly { c0, c1, c2 };
        polys.push(rp);

        // Fiat-Shamir
        mix_sf(channel, c0);
        mix_sf(channel, c1);
        mix_sf(channel, c2);
        let challenge = channel.draw_qm31();
        challenges.push(challenge);

        f_m = fold_mle(&f_m, challenge, mid);
        f_s = fold_mle(&f_s, challenge, mid);
        cur_n = mid;
    }

    assert_eq!(f_m.len(), 1);
    assert_eq!(f_s.len(), 1);
    let sbox_out_eval = f_s[0];

    mix_sf(channel, sbox_out_eval);

    (polys, challenges, sbox_out_eval)
}

fn compute_product_sum_at_t(
    f_a: &[SecureField],
    f_b: &[SecureField],
    mid: usize,
    t: SecureField,
) -> SecureField {
    let one_minus_t = SecureField::one() - t;
    let mut sum = SecureField::zero();
    for i in 0..mid {
        let a_t = one_minus_t * f_a[i] + t * f_a[mid + i];
        let b_t = one_minus_t * f_b[i] + t * f_b[mid + i];
        sum += a_t * b_t;
    }
    sum
}

/// Restrict the UPPER `n_upper` variables of an MLE, leaving lower vars free.
///
/// MLE layout: evals[lower_idx + upper_idx * (1 << n_lower)]
/// We fix upper vars to `assignments`, returning evals of size 2^n_lower.
fn restrict_upper_vars(
    evals: &[SecureField],
    assignments: &[SecureField],
    n_upper: usize,
) -> Vec<SecureField> {
    let total_vars = evals.len().ilog2() as usize;
    let n_lower = total_vars - n_upper;
    let lower_size = 1 << n_lower;
    let upper_size = 1 << n_upper;

    // Build eq weights for upper assignments
    let eq_upper = build_eq_evals(assignments);

    let mut result = vec![SecureField::zero(); lower_size];
    for upper_idx in 0..upper_size {
        let w = eq_upper[upper_idx];
        for lower_idx in 0..lower_size {
            result[lower_idx] += w * evals[lower_idx + upper_idx * lower_size];
        }
    }
    result
}

// ══════════════════════════ Degree-3 Eq-Sumcheck (Mul Check) ═════════════════

/// Degree-3 eq-sumcheck for element-wise multiplication: c = a · b.
///
/// Proves: Σ_x eq(r, x) · a(x) · b(x) = claimed_sum
/// where claimed_sum = Ṽ_c(r).
///
/// Returns (round_polys, challenges, a_eval, b_eval) at the final sumcheck point.
fn prove_mul_eq_sumcheck(
    a_evals: &[SecureField],
    b_evals: &[SecureField],
    claim_point: &[SecureField],
    claimed_sum: SecureField,
    tag: u64,
    channel: &mut PoseidonChannel,
) -> (Vec<RoundPolyDeg3>, Vec<SecureField>, SecureField, SecureField) {
    let n = a_evals.len();
    assert_eq!(n, b_evals.len());
    assert!(n.is_power_of_two());
    let num_vars = n.ilog2() as usize;

    let mut eq_evals = build_eq_evals(&claim_point[..num_vars]);
    let mut f_a = a_evals.to_vec();
    let mut f_b = b_evals.to_vec();

    // Mix tag + claimed sum
    channel.mix_u64(tag);
    mix_sf(channel, claimed_sum);

    let mut polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;

        let s0 = compute_triple_sum_at_t(&eq_evals, &f_a, &f_b, mid, SecureField::zero());
        let s1 = compute_triple_sum_at_t(&eq_evals, &f_a, &f_b, mid, SecureField::one());
        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let s2 = compute_triple_sum_at_t(&eq_evals, &f_a, &f_b, mid, two);
        let s3 = compute_triple_sum_at_t(&eq_evals, &f_a, &f_b, mid, three);

        // Newton divided differences for degree-3 interpolation
        let inv2 = two.inverse();
        let inv6 = (SecureField::from(M31::from(6u32))).inverse();

        let d1 = s1 - s0;
        let d2 = (s2 - s1 - s1 + s0) * inv2;
        let d3 = (s3 - s0 - three * (s2 - s1)) * inv6;

        let c0 = s0;
        let c1 = d1 - d2 + two * d3;
        let c2 = d2 - three * d3;
        let c3 = d3;

        let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
        polys.push(rp);

        channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        let challenge = channel.draw_qm31();
        challenges.push(challenge);

        eq_evals = fold_mle(&eq_evals, challenge, mid);
        f_a = fold_mle(&f_a, challenge, mid);
        f_b = fold_mle(&f_b, challenge, mid);
        cur_n = mid;
    }

    let a_eval = f_a[0];
    let b_eval = f_b[0];
    mix_sf(channel, a_eval);
    mix_sf(channel, b_eval);

    (polys, challenges, a_eval, b_eval)
}

fn compute_triple_sum_at_t(
    eq: &[SecureField],
    a: &[SecureField],
    b: &[SecureField],
    mid: usize,
    t: SecureField,
) -> SecureField {
    let one_minus_t = SecureField::one() - t;
    let mut sum = SecureField::zero();
    for i in 0..mid {
        let eq_t = one_minus_t * eq[i] + t * eq[mid + i];
        let a_t = one_minus_t * a[i] + t * a[mid + i];
        let b_t = one_minus_t * b[i] + t * b[mid + i];
        sum += eq_t * a_t * b_t;
    }
    sum
}

// ═════════════════════════ Degree-2 Combination Sumcheck ═════════════════════

/// Combine two claims on the same MLE into one.
///
/// Given claims: Ṽ(p1) = v1, Ṽ(p2) = v2
/// Proves: Σ_x (α·eq(p1,x) + (1-α)·eq(p2,x)) · f(x) = α·v1 + (1-α)·v2
///
/// Returns (round_polys, challenges, final_eval).
fn prove_combination_sumcheck(
    f_evals: &[SecureField],
    p1: &[SecureField],
    v1: SecureField,
    p2: &[SecureField],
    v2: SecureField,
    channel: &mut PoseidonChannel,
) -> (Vec<RoundPoly>, Vec<SecureField>, SecureField) {
    let n = f_evals.len();
    assert!(n.is_power_of_two());
    let num_vars = n.ilog2() as usize;

    // Draw combiner alpha
    let alpha = channel.draw_qm31();
    let target = alpha * v1 + (SecureField::one() - alpha) * v2;

    // Build weight function: w(x) = α·eq(p1,x) + (1-α)·eq(p2,x)
    let eq1 = build_eq_evals(p1);
    let eq2 = build_eq_evals(p2);
    let one_minus_alpha = SecureField::one() - alpha;
    let mut w: Vec<SecureField> = (0..n)
        .map(|i| alpha * eq1[i] + one_minus_alpha * eq2[i])
        .collect();

    let mut f = f_evals.to_vec();

    channel.mix_u64(0x434D42 as u64); // "CMB" tag
    mix_sf(channel, target);

    let mut polys = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    let mut cur_n = n;

    for _ in 0..num_vars {
        let mid = cur_n / 2;

        let s0 = compute_product_sum_at_t(&w, &f, mid, SecureField::zero());
        let s1 = compute_product_sum_at_t(&w, &f, mid, SecureField::one());
        let two = SecureField::from(M31::from(2u32));
        let s2 = compute_product_sum_at_t(&w, &f, mid, two);

        let inv2 = two.inverse();
        let c0 = s0;
        let c1 = (SecureField::from(M31::from(4u32)) * s1 - s2 - SecureField::from(M31::from(3u32)) * s0) * inv2;
        let c2 = (s2 - SecureField::from(M31::from(2u32)) * s1 + s0) * inv2;

        let rp = RoundPoly { c0, c1, c2 };
        polys.push(rp);

        mix_sf(channel, c0);
        mix_sf(channel, c1);
        mix_sf(channel, c2);
        let challenge = channel.draw_qm31();
        challenges.push(challenge);

        w = fold_mle(&w, challenge, mid);
        f = fold_mle(&f, challenge, mid);
        cur_n = mid;
    }

    let final_eval = f[0];
    mix_sf(channel, final_eval);

    (polys, challenges, final_eval)
}

// ═══════════════════════════════ Main Prover ═══════════════════════════════

/// Prove a batch of N Poseidon2 permutations.
///
/// The prover walks rounds from output to input, generating per-round proofs.
/// Uses Fiat-Shamir via PoseidonChannel for all random challenges.
pub fn prove_poseidon2_batch(
    inputs: &[[M31; STATE_WIDTH]],
    channel: &mut PoseidonChannel,
) -> Poseidon2BatchProof {
    let trace = generate_trace(inputs);
    let n = trace.n;
    let log_n = n.ilog2() as usize;
    let total_vars = log_n + LOG_STATE; // Variables in the N×16 MLE

    // Precompute matrix MLEs
    let ext_matrix = build_external_matrix_mle();
    let int_matrix = build_internal_matrix_mle();

    // Seed channel
    channel.mix_u64(n as u64);
    channel.mix_u64(N_ROUNDS as u64);

    // Start with output claim: evaluate output MLE at random point
    let output_flat = flatten_states(&trace.states[N_ROUNDS]);
    let r_out = channel.draw_qm31s(total_vars);
    let output_eval = evaluate_mle(&output_flat, &r_out);
    mix_sf(channel, output_eval);

    let mut current_point = r_out;
    let mut current_value = output_eval;
    let mut round_proofs = Vec::with_capacity(N_ROUNDS);

    // Walk rounds from output to input (backwards)
    for round_rev in 0..N_ROUNDS {
        let round_idx = N_ROUNDS - 1 - round_rev;
        let is_full = trace.is_full[round_idx];

        let matrix = if is_full { &ext_matrix } else { &int_matrix };

        // Split current point into (r_inst, r_elem)
        // Convention: MSB-first in evaluate_mle → instance vars first, element vars second
        // since flat layout is flat[inst * 16 + elem] (instance in upper bits).
        let r_inst = current_point[..log_n].to_vec();
        let r_elem = current_point[log_n..].to_vec();

        // ── Step 1: Linear layer sumcheck ──
        let sbox_out_flat = flatten_states(&trace.after_sboxes[round_idx]);
        let (linear_polys, lin_challenges, sbox_out_eval) =
            prove_linear_sumcheck(
                matrix,
                &sbox_out_flat,
                &r_inst,
                &r_elem,
                current_value,
                channel,
            );

        // New element point after linear layer sumcheck
        let r_elem_new = lin_challenges.clone();

        if is_full {
            // For full rounds: after_sbox = x5 (S-box on all elements)
            // Current claim: x5 at (r_inst, r_elem_new) = sbox_out_eval

            let _x5_flat = flatten_states(&trace.x5s[round_idx]);
            let x4_flat = flatten_states(&trace.x4s[round_idx]);
            let after_add_flat = flatten_states(&trace.after_adds[round_idx]);

            // Claim point for S-box checks: (r_inst, r_elem_new) in MSB-first order
            let mut sbox_point = r_inst.clone();
            sbox_point.extend_from_slice(&r_elem_new);

            // ── Step 2: x5 = x4 · after_add at sbox_point ──
            let (mul3_polys, mul3_ch, x4_eval, aa_eval_1) =
                prove_mul_eq_sumcheck(
                    &x4_flat, &after_add_flat,
                    &sbox_point, sbox_out_eval,
                    0x4D5533, // "MU3"
                    channel,
                );
            // After mul3: x4_eval = x4(mul3_ch), aa_eval_1 = aa(mul3_ch)

            // ── Step 3: x4 = x2² at mul3_ch ──
            let x2_flat = flatten_states(&trace.x2s[round_idx]);
            let (mul2_polys, mul2_ch, x2_eval, _x2_eval_dup) =
                prove_mul_eq_sumcheck(
                    &x2_flat, &x2_flat,
                    &mul3_ch, x4_eval,
                    0x4D5532, // "MU2"
                    channel,
                );
            // After mul2: x2_eval = x2(mul2_ch)

            // ── Step 4: x2 = after_add² at mul2_ch ──
            let (mul1_polys, mul1_ch, aa_eval_2, _aa_eval_2_dup) =
                prove_mul_eq_sumcheck(
                    &after_add_flat, &after_add_flat,
                    &mul2_ch, x2_eval,
                    0x4D5531, // "MU1"
                    channel,
                );
            // After mul1: aa_eval_2 = aa(mul1_ch)

            // ── Step 5: Combination sumcheck for dual after_add claims ──
            // Claim A: after_add at sbox_point = aa_eval_1 (from mul3)
            // Claim B: after_add at mul1_ch = aa_eval_2 (from mul1)
            // But we used intermediate points. Let me simplify:
            // ── Step 5: Combination sumcheck ──
            // Two claims on after_add at different points:
            //   aa(mul3_ch) = aa_eval_1  (from mul3)
            //   aa(mul1_ch) = aa_eval_2  (from mul1)
            let (combine_polys, cmb_ch, combined_eval) =
                prove_combination_sumcheck(
                    &after_add_flat,
                    &mul3_ch, aa_eval_1,
                    &mul1_ch, aa_eval_2,
                    channel,
                );

            // Resolve after_add → state by subtracting round constants
            // combined_eval is after_add at the combination challenges (cmb_ch)
            let rc = get_round_constants();
            let rc_for_round = if round_idx < N_HALF_FULL_ROUNDS {
                &rc.external[trace.rc_indices[round_idx]]
            } else if round_idx >= N_HALF_FULL_ROUNDS + N_PARTIAL_ROUNDS {
                &rc.external[trace.rc_indices[round_idx]]
            } else {
                unreachable!("full round should not be in partial range")
            };

            // The combination challenges form the next evaluation point
            let r_elem_final = &cmb_ch[log_n..];
            let rc_eval = eval_rc_mle(rc_for_round, r_elem_final);
            let state_eval = combined_eval - rc_eval;

            current_point = cmb_ch;
            current_value = state_eval;

            round_proofs.push(Poseidon2RoundProof {
                is_full_round: true,
                linear_polys,
                sbox_out_eval,
                mul3_polys,
                x4_eval,
                after_add_eval_from_mul3: aa_eval_1,
                mul2_polys,
                x2_eval,
                mul1_polys,
                after_add_eval_from_mul1: aa_eval_2,
                combine_polys,
                combined_eval,
                partial_x5_0_eval: None,
                partial_after_add_eval: None,
                partial_after_add_0_eval: None,
            });
        } else {
            // ── Partial round ──
            // after_sbox[elem] = after_add[elem] for elem > 0
            // after_sbox[0] = x5_0 (S-box applied to element 0 only)
            //
            // Claim from linear layer: Ṽ_{after_sbox}(r_inst, r_elem_new) = sbox_out_eval
            //
            // Decomposition:
            //   Ṽ_{after_sbox}(r_inst, r_elem_new) =
            //     Ṽ_{after_add}(r_inst, r_elem_new) +
            //     eq(r_elem_new, 0) · (Ṽ_{x5_0}(r_inst) - Ṽ_{after_add_0}(r_inst))

            let after_add_flat = flatten_states(&trace.after_adds[round_idx]);
            let x5_0_flat = flatten_elem0(&trace.x5s[round_idx]);
            let after_add_0_flat = flatten_elem0(&trace.after_adds[round_idx]);

            // Evaluate at the known points (r_inst, r_elem_new) in MSB-first order
            let mut full_point = r_inst.clone();
            full_point.extend_from_slice(&r_elem_new);
            let after_add_eval = evaluate_mle(&after_add_flat, &full_point);
            let x5_0_eval = evaluate_mle(&x5_0_flat, &r_inst);
            let after_add_0_eval = evaluate_mle(&after_add_0_flat, &r_inst);

            // Verify decomposition (prover-side sanity check)
            let eq_0 = eval_eq_zero(&r_elem_new);
            let expected = after_add_eval + eq_0 * (x5_0_eval - after_add_0_eval);
            debug_assert!(
                (sbox_out_eval - expected).0.0.0 == 0 && (sbox_out_eval - expected).0.1.0 == 0
                && (sbox_out_eval - expected).1.0.0 == 0 && (sbox_out_eval - expected).1.1.0 == 0,
                "partial round decomposition mismatch"
            );

            // S-box checks for element 0 only (log(N) variables)
            let x4_0_flat = flatten_elem0(&trace.x4s[round_idx]);
            let x2_0_flat = flatten_elem0(&trace.x2s[round_idx]);

            // x5_0 = x4_0 · after_add_0 at r_inst
            let (mul3_polys, mul3_ch, x4_eval, aa0_eval_from_mul3) =
                prove_mul_eq_sumcheck(
                    &x4_0_flat, &after_add_0_flat,
                    &r_inst, x5_0_eval,
                    0x4D5533,
                    channel,
                );
            // After mul3: x4_eval = x4_0(mul3_ch), aa0_eval = aa_0(mul3_ch)

            // x4_0 = x2_0² at mul3_ch
            let (mul2_polys, mul2_ch, x2_eval, _) =
                prove_mul_eq_sumcheck(
                    &x2_0_flat, &x2_0_flat,
                    &mul3_ch, x4_eval,
                    0x4D5532,
                    channel,
                );
            // After mul2: x2_eval = x2_0(mul2_ch)

            // x2_0 = after_add_0² at mul2_ch
            let (mul1_polys, _mul1_ch, aa0_eval_from_mul1, _) =
                prove_mul_eq_sumcheck(
                    &after_add_0_flat, &after_add_0_flat,
                    &mul2_ch, x2_eval,
                    0x4D5531,
                    channel,
                );
            // After mul1: aa0_eval_from_mul1 = aa_0(mul1_ch)

            // Combination: merge claims on after_add (full N×16 MLE)
            // Claim 1: after_add at (r_inst, r_elem_new) = after_add_eval
            // Claim 2: after_add at (r_inst, 0) = after_add_0_eval
            // Note: the S-box chain gives us aa_0 at mul3_ch and mul1_ch,
            // but the combination uses the FULL after_add MLE with the original points.
            let mut point_2 = r_inst.clone();
            point_2.extend(vec![SecureField::zero(); LOG_STATE]);

            let (combine_polys, cmb_ch, combined_eval) =
                prove_combination_sumcheck(
                    &after_add_flat,
                    &full_point, after_add_eval,
                    &point_2, after_add_0_eval,
                    channel,
                );

            // Resolve after_add → state using combination challenges as the point
            let rc = get_round_constants();
            let rc_internal = rc.internal[trace.rc_indices[round_idx]];
            let r_elem_final = &cmb_ch[log_n..];

            // For partial rounds: rc is only on element 0
            let eq_0_final = eval_eq_zero(r_elem_final);
            let rc_eval = eq_0_final * SecureField::from(rc_internal);
            let state_eval = combined_eval - rc_eval;

            current_point = cmb_ch;
            current_value = state_eval;

            round_proofs.push(Poseidon2RoundProof {
                is_full_round: false,
                linear_polys,
                sbox_out_eval,
                mul3_polys,
                x4_eval,
                after_add_eval_from_mul3: aa0_eval_from_mul3,
                mul2_polys,
                x2_eval,
                mul1_polys,
                after_add_eval_from_mul1: aa0_eval_from_mul1,
                combine_polys,
                combined_eval,
                partial_x5_0_eval: Some(x5_0_eval),
                partial_after_add_eval: Some(after_add_eval),
                partial_after_add_0_eval: Some(after_add_0_eval),
            });
        }
    }

    Poseidon2BatchProof {
        round_proofs,
        input_eval: current_value,
    }
}

// ════════════════════════════════ Main Verifier ════════════════════════════════

/// Verify a batch Poseidon2 proof.
///
/// Replays the Fiat-Shamir transcript and checks each round's sumcheck proofs.
pub fn verify_poseidon2_batch(
    proof: &Poseidon2BatchProof,
    inputs: &[[M31; STATE_WIDTH]],
    outputs: &[[M31; STATE_WIDTH]],
    channel: &mut PoseidonChannel,
) -> Result<(), Poseidon2Error> {
    let n = inputs.len();
    if !n.is_power_of_two() || n == 0 {
        return Err(Poseidon2Error::InputMismatch("batch size must be positive power of 2".into()));
    }
    let log_n = n.ilog2() as usize;
    let total_vars = log_n + LOG_STATE;

    let ext_matrix = build_external_matrix_mle();
    let int_matrix = build_internal_matrix_mle();

    // Seed channel (must match prover)
    channel.mix_u64(n as u64);
    channel.mix_u64(N_ROUNDS as u64);

    // Reconstruct output claim
    let output_flat = flatten_states(outputs);
    let r_out = channel.draw_qm31s(total_vars);
    let output_eval = evaluate_mle(&output_flat, &r_out);
    mix_sf(channel, output_eval);

    let mut current_point = r_out;
    let mut current_value = output_eval;

    let rc = get_round_constants();

    for (round_rev, rp) in proof.round_proofs.iter().enumerate() {
        let round_idx = N_ROUNDS - 1 - round_rev;
        let is_full = rp.is_full_round;
        let matrix = if is_full { &ext_matrix } else { &int_matrix };

        // Split current point: (r_inst, r_elem) in MSB-first order
        let r_inst = current_point[..log_n].to_vec();
        let r_elem = current_point[log_n..].to_vec();

        // ── Verify linear layer sumcheck ──
        let (_sbox_out_eval, lin_challenges) = verify_linear_sumcheck(
            matrix,
            &r_elem,
            current_value,
            &rp.linear_polys,
            rp.sbox_out_eval,
            round_idx,
            channel,
        )?;

        let r_elem_new = lin_challenges;

        // Build the sbox_point = (r_inst, r_elem_new) in MSB-first order
        let mut sbox_point = r_inst.clone();
        sbox_point.extend_from_slice(&r_elem_new);

        // The remaining verification differs for full vs partial rounds
        if is_full {
            let cmb_ch = verify_full_round_sbox(
                rp, &r_inst, &sbox_point, round_idx, channel,
            )?;

            // Resolve after_add → state using combination challenges
            let rc_idx = if round_idx < N_HALF_FULL_ROUNDS {
                round_idx
            } else {
                round_idx - N_PARTIAL_ROUNDS
            };
            let rc_for_round = &rc.external[rc_idx];

            let r_elem_final = &cmb_ch[log_n..];
            let rc_eval = eval_rc_mle(rc_for_round, r_elem_final);
            let state_eval = rp.combined_eval - rc_eval;

            current_point = cmb_ch;
            current_value = state_eval;
        } else {
            let cmb_ch = verify_partial_round_sbox(
                rp, &r_inst, &r_elem_new, log_n, round_idx, channel,
            )?;

            // Resolve after_add → state using combination challenges
            let rc_internal = rc.internal[round_idx - N_HALF_FULL_ROUNDS];
            let r_elem_final = &cmb_ch[log_n..];
            let eq_0 = eval_eq_zero(r_elem_final);
            let rc_eval = eq_0 * SecureField::from(rc_internal);
            let state_eval = rp.combined_eval - rc_eval;

            current_point = cmb_ch;
            current_value = state_eval;
        }
    }

    // Final check: input_eval should match the MLE of inputs at the final point
    let input_flat = flatten_states(inputs);
    let expected_input_eval = evaluate_mle(&input_flat, &current_point);
    if proof.input_eval != expected_input_eval {
        return Err(Poseidon2Error::FinalCheckFailed {
            round: 0,
            reason: format!(
                "input MLE mismatch: proof says {}, computed {}",
                proof.input_eval, expected_input_eval,
            ),
        });
    }
    if current_value != expected_input_eval {
        return Err(Poseidon2Error::FinalCheckFailed {
            round: 0,
            reason: format!(
                "propagated value {} != input MLE {}",
                current_value, expected_input_eval,
            ),
        });
    }

    Ok(())
}

/// Verify the linear layer sumcheck for one round.
///
/// Returns (sbox_out_eval, challenges) — the challenges form r_elem_new.
fn verify_linear_sumcheck(
    matrix: &[SecureField],
    r_elem: &[SecureField],
    claimed_sum: SecureField,
    polys: &[RoundPoly],
    sbox_out_eval: SecureField,
    round_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<(SecureField, Vec<SecureField>), Poseidon2Error> {
    channel.mix_u64(0x4C494E as u64);
    mix_sf(channel, claimed_sum);

    let mut current_sum = claimed_sum;
    let mut challenges = Vec::with_capacity(LOG_STATE);

    for (i, rp) in polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2;

        if p0 + p1 != current_sum {
            return Err(Poseidon2Error::SumcheckFailed {
                round: round_idx,
                step: "linear".into(),
                reason: format!("round {}: p(0)+p(1) != claimed_sum", i),
            });
        }

        mix_sf(channel, rp.c0);
        mix_sf(channel, rp.c1);
        mix_sf(channel, rp.c2);
        let challenge = channel.draw_qm31();
        challenges.push(challenge);

        current_sum = rp.c0 + rp.c1 * challenge + rp.c2 * challenge * challenge;
    }

    // Final check: current_sum should equal M_eval * sbox_out_eval
    let m_eval = eval_matrix_mle(matrix, r_elem, &challenges);
    if current_sum != m_eval * sbox_out_eval {
        return Err(Poseidon2Error::SumcheckFailed {
            round: round_idx,
            step: "linear final".into(),
            reason: "M_eval * sbox_out_eval != sumcheck final".into(),
        });
    }

    mix_sf(channel, sbox_out_eval);
    Ok((sbox_out_eval, challenges))
}

/// Verify degree-3 eq-sumcheck for a multiplication relation.
///
/// Returns the sumcheck challenges (the evaluation point for lhs/rhs).
fn verify_mul_eq_sumcheck(
    polys: &[RoundPolyDeg3],
    claimed_sum: SecureField,
    lhs_eval: SecureField,
    rhs_eval: SecureField,
    claim_point: &[SecureField],
    tag: u64,
    round_idx: usize,
    step_name: &str,
    channel: &mut PoseidonChannel,
) -> Result<Vec<SecureField>, Poseidon2Error> {
    let num_vars = polys.len();

    channel.mix_u64(tag);
    mix_sf(channel, claimed_sum);

    let mut current_sum = claimed_sum;
    let mut challenges = Vec::with_capacity(num_vars);

    for (i, rp) in polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

        if p0 + p1 != current_sum {
            return Err(Poseidon2Error::SumcheckFailed {
                round: round_idx,
                step: step_name.into(),
                reason: format!("round {}: p(0)+p(1) != claimed_sum", i),
            });
        }

        channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
        let challenge = channel.draw_qm31();
        challenges.push(challenge);

        current_sum = rp.eval(challenge);
    }

    // Final check: sum == eq(claim_point, challenges) * lhs * rhs
    let eq_val = compute_eq_eval(&claim_point[..num_vars], &challenges);
    let expected = eq_val * lhs_eval * rhs_eval;
    if current_sum != expected {
        return Err(Poseidon2Error::SumcheckFailed {
            round: round_idx,
            step: step_name.into(),
            reason: format!("final: sum {} != eq*a*b {}", current_sum, expected),
        });
    }

    mix_sf(channel, lhs_eval);
    mix_sf(channel, rhs_eval);

    Ok(challenges)
}

/// Verify a degree-2 combination sumcheck.
///
/// Returns the sumcheck challenges (the evaluation point for the combined MLE).
fn verify_combination_sumcheck(
    polys: &[RoundPoly],
    p1: &[SecureField],
    v1: SecureField,
    p2: &[SecureField],
    v2: SecureField,
    combined_eval: SecureField,
    round_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<Vec<SecureField>, Poseidon2Error> {
    let alpha = channel.draw_qm31();
    let target = alpha * v1 + (SecureField::one() - alpha) * v2;

    channel.mix_u64(0x434D42 as u64);
    mix_sf(channel, target);

    let mut current_sum = target;
    let mut challenges = Vec::with_capacity(polys.len());

    for (i, rp) in polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1_val = rp.c0 + rp.c1 + rp.c2;

        if p0 + p1_val != current_sum {
            return Err(Poseidon2Error::SumcheckFailed {
                round: round_idx,
                step: "combine".into(),
                reason: format!("round {}: p(0)+p(1) != sum", i),
            });
        }

        mix_sf(channel, rp.c0);
        mix_sf(channel, rp.c1);
        mix_sf(channel, rp.c2);
        let challenge = channel.draw_qm31();
        challenges.push(challenge);

        current_sum = rp.c0 + rp.c1 * challenge + rp.c2 * challenge * challenge;
    }

    // Final check: current_sum == weight_eval * combined_eval
    let eq1_val = compute_eq_eval(p1, &challenges);
    let eq2_val = compute_eq_eval(p2, &challenges);
    let weight_eval = alpha * eq1_val + (SecureField::one() - alpha) * eq2_val;

    if current_sum != weight_eval * combined_eval {
        return Err(Poseidon2Error::SumcheckFailed {
            round: round_idx,
            step: "combine final".into(),
            reason: "weight*eval != sum".into(),
        });
    }

    mix_sf(channel, combined_eval);
    Ok(challenges)
}

fn verify_full_round_sbox(
    rp: &Poseidon2RoundProof,
    _r_inst: &[SecureField],
    sbox_point: &[SecureField],
    round_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<Vec<SecureField>, Poseidon2Error> {
    // x5 = x4 · after_add (eq-sumcheck at sbox_point)
    let mul3_ch = verify_mul_eq_sumcheck(
        &rp.mul3_polys,
        rp.sbox_out_eval,
        rp.x4_eval,
        rp.after_add_eval_from_mul3,
        sbox_point,
        0x4D5533,
        round_idx,
        "mul3",
        channel,
    )?;

    // x4 = x2² (eq-sumcheck at mul3_ch — where x4 was evaluated)
    let mul2_ch = verify_mul_eq_sumcheck(
        &rp.mul2_polys,
        rp.x4_eval,
        rp.x2_eval,
        rp.x2_eval,
        &mul3_ch,
        0x4D5532,
        round_idx,
        "mul2",
        channel,
    )?;

    // x2 = after_add² (eq-sumcheck at mul2_ch — where x2 was evaluated)
    let mul1_ch = verify_mul_eq_sumcheck(
        &rp.mul1_polys,
        rp.x2_eval,
        rp.after_add_eval_from_mul1,
        rp.after_add_eval_from_mul1,
        &mul2_ch,
        0x4D5531,
        round_idx,
        "mul1",
        channel,
    )?;

    // Combination sumcheck: merge two claims on after_add at different points
    // Claim A: after_add at mul3_ch = aa_eval_from_mul3
    // Claim B: after_add at mul1_ch = aa_eval_from_mul1
    let cmb_ch = verify_combination_sumcheck(
        &rp.combine_polys,
        &mul3_ch, rp.after_add_eval_from_mul3,
        &mul1_ch, rp.after_add_eval_from_mul1,
        rp.combined_eval,
        round_idx,
        channel,
    )?;

    Ok(cmb_ch)
}

fn verify_partial_round_sbox(
    rp: &Poseidon2RoundProof,
    r_inst: &[SecureField],
    r_elem_new: &[SecureField],
    _log_n: usize,
    round_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<Vec<SecureField>, Poseidon2Error> {
    // For partial rounds, S-box mul checks operate on element-0 only (log_n variables).
    let x5_0_eval = rp.partial_x5_0_eval.ok_or_else(|| Poseidon2Error::FinalCheckFailed {
        round: round_idx,
        reason: "missing partial_x5_0_eval".into(),
    })?;
    let after_add_eval = rp.partial_after_add_eval.ok_or_else(|| Poseidon2Error::FinalCheckFailed {
        round: round_idx,
        reason: "missing partial_after_add_eval".into(),
    })?;
    let after_add_0_eval = rp.partial_after_add_0_eval.ok_or_else(|| Poseidon2Error::FinalCheckFailed {
        round: round_idx,
        reason: "missing partial_after_add_0_eval".into(),
    })?;

    // Verify decomposition: sbox_out_eval = after_add_eval + eq_0 * (x5_0_eval - after_add_0_eval)
    let eq_0 = eval_eq_zero(r_elem_new);
    let expected = after_add_eval + eq_0 * (x5_0_eval - after_add_0_eval);
    if rp.sbox_out_eval != expected {
        return Err(Poseidon2Error::SumcheckFailed {
            round: round_idx,
            step: "decomposition".into(),
            reason: "after_sbox decomposition mismatch".into(),
        });
    }

    // Chain: x5_0 → x4_0 → x2_0 → after_add_0 via sumcheck challenges.
    // x5_0 = x4_0 · after_add_0 at r_inst (claimed_sum = x5_0_eval)
    let mul3_ch = verify_mul_eq_sumcheck(
        &rp.mul3_polys,
        x5_0_eval,
        rp.x4_eval,
        rp.after_add_eval_from_mul3,
        r_inst,
        0x4D5533,
        round_idx,
        "mul3_partial",
        channel,
    )?;

    // x4_0 = x2_0² at mul3_ch
    let mul2_ch = verify_mul_eq_sumcheck(
        &rp.mul2_polys,
        rp.x4_eval,
        rp.x2_eval,
        rp.x2_eval,
        &mul3_ch,
        0x4D5532,
        round_idx,
        "mul2_partial",
        channel,
    )?;

    // x2_0 = after_add_0² at mul2_ch
    let _mul1_ch = verify_mul_eq_sumcheck(
        &rp.mul1_polys,
        rp.x2_eval,
        rp.after_add_eval_from_mul1,
        rp.after_add_eval_from_mul1,
        &mul2_ch,
        0x4D5531,
        round_idx,
        "mul1_partial",
        channel,
    )?;

    // Combination: merge claims on after_add (MSB-first: r_inst, r_elem)
    // Claim A: after_add at (r_inst, r_elem_new) = after_add_eval
    // Claim B: after_add at (r_inst, 0) = after_add_0_eval
    let mut full_point = r_inst.to_vec();
    full_point.extend_from_slice(r_elem_new);
    let mut point_2 = r_inst.to_vec();
    point_2.extend(vec![SecureField::zero(); LOG_STATE]);
    let cmb_ch = verify_combination_sumcheck(
        &rp.combine_polys,
        &full_point, after_add_eval,
        &point_2, after_add_0_eval,
        rp.combined_eval,
        round_idx,
        channel,
    )?;

    Ok(cmb_ch)
}

// ═══════════════════════════════ Tests ═══════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_inputs(n: usize) -> Vec<[M31; STATE_WIDTH]> {
        (0..n)
            .map(|i| {
                let mut s = [M31::from_u32_unchecked(0); STATE_WIDTH];
                for j in 0..STATE_WIDTH {
                    s[j] = M31::from_u32_unchecked((i * 16 + j + 1) as u32);
                }
                s
            })
            .collect()
    }

    fn compute_outputs(inputs: &[[M31; STATE_WIDTH]]) -> Vec<[M31; STATE_WIDTH]> {
        inputs
            .iter()
            .map(|s| {
                let mut out = *s;
                poseidon2_permutation(&mut out);
                out
            })
            .collect()
    }

    #[test]
    fn test_trace_generation_single() {
        let inputs = make_inputs(1);
        let trace = generate_trace(&inputs);
        assert!(verify_trace_direct(&trace));
    }

    #[test]
    fn test_trace_generation_batch() {
        let inputs = make_inputs(4);
        let trace = generate_trace(&inputs);
        assert!(verify_trace_direct(&trace));
    }

    #[test]
    fn test_trace_generation_batch_16() {
        let inputs = make_inputs(16);
        let trace = generate_trace(&inputs);
        assert!(verify_trace_direct(&trace));
    }

    #[test]
    fn test_trace_round_count() {
        let inputs = make_inputs(2);
        let trace = generate_trace(&inputs);
        assert_eq!(trace.states.len(), N_ROUNDS + 1);
        assert_eq!(trace.after_adds.len(), N_ROUNDS);
        assert_eq!(trace.x2s.len(), N_ROUNDS);
        assert_eq!(trace.is_full.len(), N_ROUNDS);

        // First 4 rounds are full, next 14 partial, last 4 full
        for i in 0..4 { assert!(trace.is_full[i], "round {} should be full", i); }
        for i in 4..18 { assert!(!trace.is_full[i], "round {} should be partial", i); }
        for i in 18..22 { assert!(trace.is_full[i], "round {} should be full", i); }
    }

    #[test]
    fn test_matrix_mle_external() {
        let matrix = build_external_matrix_mle();
        // Verify: applying M_ext to e_0 should give the first column
        let mut e0 = [M31::from_u32_unchecked(0); STATE_WIDTH];
        e0[0] = M31::from_u32_unchecked(1);
        apply_external_round_matrix(&mut e0);

        // The MLE at (i, 0) should give M[i][0] = e0[i]
        for i in 0..STATE_WIDTH {
            let val = matrix[i * STATE_WIDTH + 0];
            assert_eq!(val, SecureField::from(e0[i]),
                "M_ext[{}][0] mismatch", i);
        }
    }

    #[test]
    fn test_matrix_mle_internal() {
        let matrix = build_internal_matrix_mle();
        // M_int[i][j] = 1 + (diag[i] if i==j else 0)
        for i in 0..STATE_WIDTH {
            for j in 0..STATE_WIDTH {
                let mut expected = M31::from_u32_unchecked(1);
                if i == j {
                    expected += M31::from_u32_unchecked(INTERNAL_DIAG_U32[i]);
                }
                assert_eq!(
                    matrix[i * STATE_WIDTH + j],
                    SecureField::from(expected),
                    "M_int[{}][{}] mismatch", i, j,
                );
            }
        }
    }

    #[test]
    fn test_evaluate_mle_simple() {
        // f(0) = 3, f(1) = 7 → Ṽ(r) = 3(1-r) + 7r = 3 + 4r
        let evals = vec![
            SecureField::from(M31::from(3u32)),
            SecureField::from(M31::from(7u32)),
        ];
        let r = SecureField::from(M31::from(0u32));
        assert_eq!(evaluate_mle(&evals, &[r]), evals[0]);
        let r = SecureField::from(M31::from(1u32));
        assert_eq!(evaluate_mle(&evals, &[r]), evals[1]);
    }

    #[test]
    fn test_eq_eval_identity() {
        // eq(x, x) = 1 on the boolean hypercube
        for x in 0..4u32 {
            let point = vec![
                SecureField::from(M31::from(x & 1)),
                SecureField::from(M31::from((x >> 1) & 1)),
            ];
            let val = compute_eq_eval(&point, &point);
            assert_eq!(val, SecureField::one(), "eq(x,x) should be 1 for x={}", x);
        }
    }

    #[test]
    fn test_restrict_upper_vars() {
        // 4 evals: f(0,0)=1, f(1,0)=2, f(0,1)=3, f(1,1)=4
        // Layout: evals[lower + upper * lower_size]
        // lower = 1 bit, upper = 1 bit
        let evals = vec![
            SecureField::from(M31::from(1u32)), // (0, 0)
            SecureField::from(M31::from(2u32)), // (1, 0)
            SecureField::from(M31::from(3u32)), // (0, 1)
            SecureField::from(M31::from(4u32)), // (1, 1)
        ];

        // Restrict upper var to 0: should give f(·, 0) = [1, 2]
        let r0 = vec![SecureField::zero()];
        let result = restrict_upper_vars(&evals, &r0, 1);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], SecureField::from(M31::from(1u32)));
        assert_eq!(result[1], SecureField::from(M31::from(2u32)));

        // Restrict upper var to 1: should give f(·, 1) = [3, 4]
        let r1 = vec![SecureField::one()];
        let result = restrict_upper_vars(&evals, &r1, 1);
        assert_eq!(result[0], SecureField::from(M31::from(3u32)));
        assert_eq!(result[1], SecureField::from(M31::from(4u32)));
    }

    #[test]
    fn test_linear_sumcheck_simple() {
        // Prove output = M · x5 for a simple case
        let matrix = build_external_matrix_mle();

        // Create 1 instance with known x5
        let mut x5 = [M31::from_u32_unchecked(0); STATE_WIDTH];
        for j in 0..STATE_WIDTH {
            x5[j] = M31::from_u32_unchecked(j as u32 + 1);
        }

        // Compute output = M_ext · x5
        let mut output = x5;
        apply_external_round_matrix(&mut output);

        let x5_flat = flatten_states(&[x5]);
        let output_flat = flatten_states(&[output]);

        // Random evaluation point (element vars only, since N=1 means 0 instance vars)
        let mut channel = PoseidonChannel::default();
        channel.mix_u64(42);
        let r_elem = channel.draw_qm31s(LOG_STATE);
        let output_eval = evaluate_mle(&output_flat, &r_elem);

        let mut p_channel = PoseidonChannel::default();
        let (polys, challenges, x5_eval) = prove_linear_sumcheck(
            &matrix, &x5_flat, &[], &r_elem, output_eval, &mut p_channel,
        );

        assert_eq!(polys.len(), LOG_STATE);

        // Verify: x5_eval should match x5 MLE at challenges
        let expected_x5_eval = evaluate_mle(&x5_flat, &challenges);
        assert_eq!(x5_eval, expected_x5_eval, "x5_eval mismatch");
    }

    #[test]
    fn test_mul_eq_sumcheck_simple() {
        // Prove c = a · b for simple arrays
        let n = 4;
        let a: Vec<SecureField> = (1..=n).map(|i| SecureField::from(M31::from(i as u32))).collect();
        let b: Vec<SecureField> = (5..5+n).map(|i| SecureField::from(M31::from(i as u32))).collect();
        let c: Vec<SecureField> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

        let mut channel = PoseidonChannel::default();
        let r = channel.draw_qm31s(2); // log2(4) = 2

        let claimed_sum = evaluate_mle(&c, &r);

        let mut p_channel = PoseidonChannel::default();
        p_channel.draw_qm31s(2); // Match the random draws
        let mut p_channel2 = PoseidonChannel::default();
        let (polys, challenges, a_eval, b_eval) =
            prove_mul_eq_sumcheck(&a, &b, &r, claimed_sum, 0x5445, &mut p_channel2);

        // Verify: a_eval * b_eval * eq(r, challenges) should equal the final sum
        let eq_val = compute_eq_eval(&r, &challenges);
        let p_last = polys.last().unwrap();
        let last_ch = challenges.last().unwrap();
        let final_sum = p_last.eval(*last_ch);
        assert_eq!(final_sum, eq_val * a_eval * b_eval, "mul sumcheck final check");
    }

    #[test]
    fn test_prove_verify_single_permutation() {
        let inputs = make_inputs(1);
        let outputs = compute_outputs(&inputs);

        let mut p_channel = PoseidonChannel::default();
        let proof = prove_poseidon2_batch(&inputs, &mut p_channel);

        let mut v_channel = PoseidonChannel::default();
        let result = verify_poseidon2_batch(&proof, &inputs, &outputs, &mut v_channel);
        assert!(result.is_ok(), "verification should succeed: {:?}", result.err());
        assert_eq!(proof.round_proofs.len(), N_ROUNDS);
    }

    #[test]
    fn test_prove_verify_batch_4() {
        let inputs = make_inputs(4);
        let outputs = compute_outputs(&inputs);

        let mut p_channel = PoseidonChannel::default();
        let proof = prove_poseidon2_batch(&inputs, &mut p_channel);

        // Full verify
        let mut v_channel = PoseidonChannel::default();
        let result = verify_poseidon2_batch(&proof, &inputs, &outputs, &mut v_channel);
        assert!(result.is_ok(), "batch-4 verification should succeed: {:?}", result.err());

        assert_eq!(proof.round_proofs.len(), N_ROUNDS);

        // Verify round types
        for (i, rp) in proof.round_proofs.iter().enumerate() {
            let expected_round = N_ROUNDS - 1 - i;
            if expected_round < N_HALF_FULL_ROUNDS || expected_round >= N_HALF_FULL_ROUNDS + N_PARTIAL_ROUNDS {
                assert!(rp.is_full_round, "round proof {} should be full", i);
            } else {
                assert!(!rp.is_full_round, "round proof {} should be partial", i);
            }
        }
    }

    #[test]
    fn test_verify_rejects_wrong_output() {
        let inputs = make_inputs(1);
        let mut outputs = compute_outputs(&inputs);
        // Tamper with output
        outputs[0][0] += M31::from_u32_unchecked(1);

        let mut p_channel = PoseidonChannel::default();
        let proof = prove_poseidon2_batch(&inputs, &mut p_channel);

        let mut v_channel = PoseidonChannel::default();
        let result = verify_poseidon2_batch(&proof, &inputs, &outputs, &mut v_channel);
        assert!(result.is_err(), "verification should reject wrong output");
    }

    #[test]
    fn test_prove_verify_batch_16() {
        let inputs = make_inputs(16);
        let outputs = compute_outputs(&inputs);

        let mut p_channel = PoseidonChannel::default();
        let proof = prove_poseidon2_batch(&inputs, &mut p_channel);

        let mut v_channel = PoseidonChannel::default();
        let result = verify_poseidon2_batch(&proof, &inputs, &outputs, &mut v_channel);
        assert!(result.is_ok(), "batch-16 verification should succeed: {:?}", result.err());
    }

    #[test]
    fn test_flatten_states() {
        let states = [
            [M31::from_u32_unchecked(1); STATE_WIDTH],
            [M31::from_u32_unchecked(2); STATE_WIDTH],
        ];
        let flat = flatten_states(&states);
        assert_eq!(flat.len(), 2 * STATE_WIDTH);
        assert_eq!(flat[0], SecureField::from(M31::from(1u32)));
        assert_eq!(flat[16], SecureField::from(M31::from(2u32)));
    }

    #[test]
    fn test_eval_eq_zero() {
        // eq(0, 0) = Π(1-0) = 1
        let r = vec![SecureField::zero(); LOG_STATE];
        assert_eq!(eval_eq_zero(&r), SecureField::one());

        // eq(1, 0) = (1-1) · ... = 0 for any non-zero bit
        let mut r = vec![SecureField::zero(); LOG_STATE];
        r[0] = SecureField::one();
        assert_eq!(eval_eq_zero(&r), SecureField::zero());
    }

    #[test]
    fn test_eval_rc_mle() {
        let mut rc = [M31::from_u32_unchecked(0); STATE_WIDTH];
        rc[0] = M31::from_u32_unchecked(42);
        rc[1] = M31::from_u32_unchecked(99);

        // At binary point (0,0,0,0), should give rc[0] = 42
        let r = vec![SecureField::zero(); LOG_STATE];
        let val = eval_rc_mle(&rc, &r);
        assert_eq!(val, SecureField::from(M31::from(42u32)));

        // At binary point (0,0,0,1) (index 1 in MSB-first convention), should give rc[1] = 99
        let mut r = vec![SecureField::zero(); LOG_STATE];
        r[LOG_STATE - 1] = SecureField::one();
        let val = eval_rc_mle(&rc, &r);
        assert_eq!(val, SecureField::from(M31::from(99u32)));
    }

    #[test]
    fn test_trace_sbox_intermediates() {
        let inputs = make_inputs(2);
        let trace = generate_trace(&inputs);

        // Verify S-box intermediates for the first full round
        for i in 0..2 {
            let aa = trace.after_adds[0][i];
            let x2 = trace.x2s[0][i];
            let x4 = trace.x4s[0][i];
            let x5 = trace.x5s[0][i];
            for j in 0..STATE_WIDTH {
                assert_eq!(x2[j], aa[j] * aa[j], "x2[{}][{}]", i, j);
                assert_eq!(x4[j], x2[j] * x2[j], "x4[{}][{}]", i, j);
                assert_eq!(x5[j], x4[j] * aa[j], "x5[{}][{}]", i, j);
            }
        }

        // Verify S-box intermediates for a partial round (round 4)
        for i in 0..2 {
            let aa = trace.after_adds[4][i];
            let x2 = trace.x2s[4][i];
            let x4 = trace.x4s[4][i];
            let x5 = trace.x5s[4][i];
            // Element 0 should have S-box applied
            assert_eq!(x2[0], aa[0] * aa[0], "partial x2[{}][0]", i);
            assert_eq!(x4[0], x2[0] * x2[0], "partial x4[{}][0]", i);
            assert_eq!(x5[0], x4[0] * aa[0], "partial x5[{}][0]", i);
            // after_sbox[0] = x5[0], after_sbox[k>0] = aa[k]
            assert_eq!(trace.after_sboxes[4][i][0], x5[0], "partial after_sbox[{}][0]", i);
            for j in 1..STATE_WIDTH {
                assert_eq!(trace.after_sboxes[4][i][j], aa[j], "partial after_sbox[{}][{}]", i, j);
            }
        }
    }
}
