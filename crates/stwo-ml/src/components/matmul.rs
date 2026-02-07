//! Sumcheck-based matrix multiplication verification.
//!
//! # The Core Innovation
//!
//! Traditional zkML decomposes matrix multiplication into individual multiply-add
//! trace rows: O(m × k × n) rows for (m×k) × (k×n).
//!
//! This component uses the **sumcheck protocol over multilinear extensions** to
//! verify the same computation in O(m + k + n) verifier work:
//!
//! ```text
//! Traditional approach:
//!   C[i][j] = Σ_k A[i][k] × B[k][j]
//!   → Each multiply-add = 1 trace row
//!   → 128×128 MatMul = 2,097,152 rows
//!
//! Sumcheck approach (stwo-ml):
//!   Represent A, B, C as MLEs on boolean hypercube
//!   Prover claims: Σ_{x∈{0,1}^n} MLE_A(r_i, x) × MLE_B(x, r_j) = MLE_C(r_i, r_j)
//!   Verifier checks via n rounds of sumcheck
//!   → 128×128 MatMul ≈ 49,152 rows (42× reduction)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use stwo_ml::components::matmul::{prove_matmul, verify_matmul, M31Matrix};
//! use stwo::core::channel::Blake2sChannel;
//!
//! let a = M31Matrix::random(4, 4);
//! let b = M31Matrix::random(4, 4);
//! let c = M31Matrix::multiply(&a, &b);
//!
//! let mut prover_channel = Blake2sChannel::default();
//! let (proof, challenges) = prove_matmul(&a, &b, &c, &mut prover_channel).unwrap();
//!
//! let mut verifier_channel = Blake2sChannel::default();
//! verify_matmul(&a, &b, &c, &proof, &challenges, &mut verifier_channel).unwrap();
//! ```

use num_traits::{One, Zero};
use thiserror::Error;

use stwo::core::channel::Channel;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::backend::cpu::CpuBackend;
use stwo::prover::backend::Column;
use stwo::prover::lookups::mle::Mle;
use stwo::prover::lookups::sumcheck::{
    partially_verify, prove_batch, MultivariatePolyOracle, SumcheckProof,
};
use stwo::prover::lookups::utils::UnivariatePoly;

/// Matrix dimensions for a matmul operation.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatMulDims {
    /// Rows in A / rows in C.
    pub m: usize,
    /// Columns in A / rows in B (inner dimension).
    pub k: usize,
    /// Columns in B / columns in C.
    pub n: usize,
}

impl MatMulDims {
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        Self { m, k, n }
    }

    /// Traditional trace cost: O(m × k × n).
    pub fn naive_trace_rows(&self) -> usize {
        self.m * self.k * self.n
    }

    /// Sumcheck trace cost: O(m×n + m×k + k×n) for witness columns.
    pub fn sumcheck_trace_rows(&self) -> usize {
        self.m * self.n + self.m * self.k + self.k * self.n
    }

    /// Speedup factor: naive / sumcheck.
    pub fn speedup(&self) -> f64 {
        self.naive_trace_rows() as f64 / self.sumcheck_trace_rows() as f64
    }
}

/// Flat matrix stored in row-major order over M31.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct M31Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<M31>,
}

impl M31Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![M31::from(0); rows * cols],
        }
    }

    /// Create a matrix from a flat vector of M31 values (row-major).
    ///
    /// Returns `Err` if `data.len() != rows * cols`.
    pub fn from_data(rows: usize, cols: usize, data: Vec<M31>) -> Result<Self, MatMulError> {
        if data.len() != rows * cols {
            return Err(MatMulError::InvalidMatrixData {
                expected: rows * cols,
                got: data.len(),
            });
        }
        Ok(Self { rows, cols, data })
    }

    pub fn get(&self, i: usize, j: usize) -> M31 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, val: M31) {
        self.data[i * self.cols + j] = val;
    }

    /// Transpose this matrix.
    pub fn transpose(&self) -> M31Matrix {
        let mut t = M31Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.get(i, j));
            }
        }
        t
    }

    /// Multiply two matrices: C = A × B.
    ///
    /// Returns `Err` if inner dimensions don't match.
    pub fn multiply(a: &M31Matrix, b: &M31Matrix) -> Result<M31Matrix, MatMulError> {
        if a.cols != b.rows {
            return Err(MatMulError::DimensionMismatch {
                a_rows: a.rows,
                a_cols: a.cols,
                b_rows: b.rows,
                b_cols: b.cols,
            });
        }
        let mut c = M31Matrix::new(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut sum = M31::from(0);
                for k in 0..a.cols {
                    sum += a.get(i, k) * b.get(k, j);
                }
                c.set(i, j, sum);
            }
        }
        Ok(c)
    }
}

// ---------------------------------------------------------------------------
// Multilinear Extension helpers
// ---------------------------------------------------------------------------

/// Pad a vector to the next power of two length with zeros.
fn pad_to_pow2(v: &[SecureField]) -> Vec<SecureField> {
    let n = v.len().next_power_of_two();
    let mut padded = v.to_vec();
    padded.resize(n, SecureField::zero());
    padded
}

/// Evaluate an MLE at a point (recursive implementation for verification).
fn eval_mle_at_point(evals: &[SecureField], point: &[SecureField]) -> SecureField {
    match point {
        [] => evals[0],
        [p_i, rest @ ..] => {
            let mid = evals.len() / 2;
            let (lhs, rhs) = evals.split_at(mid);
            let lhs_eval = eval_mle_at_point(lhs, rest);
            let rhs_eval = eval_mle_at_point(rhs, rest);
            // eq(0, p_i) * lhs + eq(1, p_i) * rhs = (1 - p_i) * lhs + p_i * rhs
            *p_i * (rhs_eval - lhs_eval) + lhs_eval
        }
    }
}

/// Extract a row from matrix A as an MLE over the inner dimension k.
///
/// Given A\[i\]\[k\] for fixed i (selected by random challenges r_i),
/// produces the MLE f_A(k) = Σ_i eq(r_i, i) * A\[i\]\[k\].
fn extract_row_mle(
    matrix: &M31Matrix,
    row_challenges: &[SecureField],
) -> Vec<SecureField> {
    let num_rows = matrix.rows.next_power_of_two();
    let num_cols = matrix.cols.next_power_of_two();

    // Build eq(r_i, i) for each row index i
    let eq_evals = eq_evals_at_point(row_challenges, num_rows);

    // f_A(k) = Σ_i eq(r_i, i) * A[i][k]
    let mut result = vec![SecureField::zero(); num_cols];
    for (i, &eq_i) in eq_evals.iter().enumerate().take(matrix.rows) {
        for (k, res_k) in result.iter_mut().enumerate().take(matrix.cols) {
            *res_k += eq_i * SecureField::from(matrix.get(i, k));
        }
    }
    result
}

/// Extract a column from matrix B as an MLE over the inner dimension k.
///
/// Given B\[k\]\[j\] for fixed j (selected by random challenges r_j),
/// produces the MLE f_B(k) = Σ_j eq(r_j, j) * B\[k\]\[j\].
fn extract_col_mle(
    matrix: &M31Matrix,
    col_challenges: &[SecureField],
) -> Vec<SecureField> {
    let num_rows = matrix.rows.next_power_of_two();
    let num_cols = matrix.cols.next_power_of_two();

    // Build eq(r_j, j) for each column index j
    let eq_evals = eq_evals_at_point(col_challenges, num_cols);

    // f_B(k) = Σ_j eq(r_j, j) * B[k][j]
    let mut result = vec![SecureField::zero(); num_rows];
    for (k, res_k) in result.iter_mut().enumerate().take(matrix.rows) {
        for (j, &eq_j) in eq_evals.iter().enumerate().take(matrix.cols) {
            *res_k += eq_j * SecureField::from(matrix.get(k, j));
        }
    }
    result
}

/// Compute eq(point, i) for all i in {0,1}^n where n = ceil_log2(size).
///
/// eq(x, y) = Π_j (x_j * y_j + (1 - x_j) * (1 - y_j))
fn eq_evals_at_point(point: &[SecureField], size: usize) -> Vec<SecureField> {
    let n = point.len();
    assert!(size <= (1 << n));
    let mut evals = vec![SecureField::zero(); 1 << n];
    evals[0] = SecureField::one();

    for (j, &p_j) in point.iter().enumerate() {
        let half = 1 << j;
        // Process in reverse to avoid overwriting values we still need
        for i in (0..half).rev() {
            evals[2 * i + 1] = evals[i] * p_j;
            evals[2 * i] = evals[i] * (SecureField::one() - p_j);
        }
    }

    evals
}

// ---------------------------------------------------------------------------
// Inner-product oracle: f_A(k) * f_B(k) summed over boolean hypercube
// ---------------------------------------------------------------------------

/// Oracle for the inner product Σ_k f_A(k) * f_B(k).
///
/// This represents a degree-2 multivariate polynomial (product of two MLEs)
/// that the sumcheck protocol operates on. The degree bound of 2 per variable
/// is within STWO's MAX_DEGREE = 3.
#[derive(Debug, Clone)]
pub struct InnerProductOracle {
    /// MLE for the row slice f_A(k).
    mle_a: Mle<CpuBackend, SecureField>,
    /// MLE for the column slice f_B(k).
    mle_b: Mle<CpuBackend, SecureField>,
}

impl InnerProductOracle {
    pub fn new(
        a_evals: Vec<SecureField>,
        b_evals: Vec<SecureField>,
    ) -> Result<Self, MatMulError> {
        if a_evals.len() != b_evals.len() || !a_evals.len().is_power_of_two() {
            return Err(MatMulError::OracleDimensionMismatch {
                a_len: a_evals.len(),
                b_len: b_evals.len(),
            });
        }
        Ok(Self {
            mle_a: Mle::new(a_evals),
            mle_b: Mle::new(b_evals),
        })
    }
}

impl MultivariatePolyOracle for InnerProductOracle {
    fn n_variables(&self) -> usize {
        self.mle_a.n_variables()
    }

    /// Computes the univariate polynomial h(x_0) = Σ_{x_1,...,x_{n-1}} f_A(x_0,...) * f_B(x_0,...).
    ///
    /// Since f_A and f_B are each multilinear, h(x_0) is degree 2 in x_0.
    /// We evaluate at 3 points {0, 1, 2} and interpolate.
    fn sum_as_poly_in_first_variable(&self, _claim: SecureField) -> UnivariatePoly<SecureField> {
        let half = self.mle_a.len() / 2;

        // f_A(x_0, rest) = (1 - x_0) * a_lo + x_0 * a_hi
        // f_B(x_0, rest) = (1 - x_0) * b_lo + x_0 * b_hi
        // Product at x_0 for fixed (rest): f_A * f_B
        //   = [(1-x_0)*a_lo + x_0*a_hi] * [(1-x_0)*b_lo + x_0*b_hi]

        // Evaluate h at x_0 = 0: Σ_rest a_lo[rest] * b_lo[rest]
        let y0: SecureField = (0..half)
            .map(|i| self.mle_a.at(i) * self.mle_b.at(i))
            .sum();

        // Evaluate h at x_0 = 1: Σ_rest a_hi[rest] * b_hi[rest]
        let y1: SecureField = (0..half)
            .map(|i| self.mle_a.at(half + i) * self.mle_b.at(half + i))
            .sum();

        // Note: y0 + y1 may differ from claim if the claim is invalid.
        // The sumcheck verifier will catch this via round polynomial checks.

        // Evaluate h at x_0 = 2 for the degree-2 interpolation
        let two = SecureField::from(BaseField::from(2));
        let y2: SecureField = (0..half)
            .map(|i| {
                let a_val = two * self.mle_a.at(half + i) - self.mle_a.at(i);
                let b_val = two * self.mle_b.at(half + i) - self.mle_b.at(i);
                a_val * b_val
            })
            .sum();

        let x0 = SecureField::zero();
        let x1 = SecureField::one();
        let x2 = two;

        UnivariatePoly::interpolate_lagrange(&[x0, x1, x2], &[y0, y1, y2])
    }

    /// Fix the first variable to `challenge`, reducing both MLEs.
    fn fix_first_variable(self, challenge: SecureField) -> Self {
        let half_a = self.mle_a.len() / 2;
        let half_b = self.mle_b.len() / 2;

        // f_A(challenge, rest) = (1 - challenge) * a_lo[rest] + challenge * a_hi[rest]
        let new_a: Vec<SecureField> = (0..half_a)
            .map(|i| {
                let lo = self.mle_a.at(i);
                let hi = self.mle_a.at(half_a + i);
                challenge * (hi - lo) + lo
            })
            .collect();

        let new_b: Vec<SecureField> = (0..half_b)
            .map(|i| {
                let lo = self.mle_b.at(i);
                let hi = self.mle_b.at(half_b + i);
                challenge * (hi - lo) + lo
            })
            .collect();

        Self {
            mle_a: Mle::new(new_a),
            mle_b: Mle::new(new_b),
        }
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Error, Debug)]
pub enum MatMulError {
    #[error("dimension mismatch: A is {a_rows}×{a_cols}, B is {b_rows}×{b_cols}")]
    DimensionMismatch {
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
    },
    #[error("product matrix C has wrong dimensions: expected {expected_rows}×{expected_cols}, got {got_rows}×{got_cols}")]
    OutputDimensionMismatch {
        expected_rows: usize,
        expected_cols: usize,
        got_rows: usize,
        got_cols: usize,
    },
    #[error("invalid matrix data: expected {expected} elements, got {got}")]
    InvalidMatrixData { expected: usize, got: usize },
    #[error("MLE claim mismatch: C evaluation does not match auxiliary data")]
    ClaimMismatch,
    #[error("final evaluation mismatch: sumcheck output does not match MLE evaluations of A and B")]
    FinalEvalMismatch,
    #[error("Fiat-Shamir challenge mismatch: verifier derived different challenges than prover")]
    ChallengeMismatch,
    #[error("oracle construction error: MLE dimensions must match and be power of two (got a={a_len}, b={b_len})")]
    OracleDimensionMismatch { a_len: usize, b_len: usize },
    #[error("sumcheck verification failed: {0}")]
    SumcheckError(#[from] stwo::prover::lookups::sumcheck::SumcheckError),
}

// ---------------------------------------------------------------------------
// Proof type
// ---------------------------------------------------------------------------

/// Proof that C = A × B using the sumcheck protocol.
#[derive(Debug, Clone)]
pub struct MatMulProof {
    /// The sumcheck proof for the batched inner product claims.
    pub sumcheck_proof: SumcheckProof,
}

/// Auxiliary data produced by the prover, needed for verification.
#[derive(Debug, Clone)]
pub struct MatMulAux {
    /// Random challenges for the row dimension of C (selects a "random row").
    pub row_challenges: Vec<SecureField>,
    /// Random challenges for the column dimension of C (selects a "random column").
    pub col_challenges: Vec<SecureField>,
    /// The claimed value of MLE_C at (row_challenges, col_challenges).
    pub claimed_c_eval: SecureField,
    /// Lambda used for batching (set to 1 for single claim).
    pub lambda: SecureField,
}

// ---------------------------------------------------------------------------
// Prove
// ---------------------------------------------------------------------------

/// Prove that C = A × B using the sumcheck protocol.
///
/// The protocol:
/// 1. Draw random challenges r_i, r_j to select a "random entry" of C.
/// 2. Evaluate MLE_C(r_i, r_j) = claimed value.
/// 3. Reduce to: prove Σ_k f_A(k) * f_B(k) = claimed value
///    where f_A(k) = MLE_A(r_i, k) and f_B(k) = MLE_B(k, r_j).
/// 4. Run sumcheck on the inner product oracle.
/// 5. Verifier checks the final evaluation against MLE evaluations of A and B.
pub fn prove_matmul(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
    channel: &mut impl Channel,
) -> Result<(MatMulProof, MatMulAux), MatMulError> {
    // Validate dimensions
    if a.cols != b.rows {
        return Err(MatMulError::DimensionMismatch {
            a_rows: a.rows,
            a_cols: a.cols,
            b_rows: b.rows,
            b_cols: b.cols,
        });
    }
    if c.rows != a.rows || c.cols != b.cols {
        return Err(MatMulError::OutputDimensionMismatch {
            expected_rows: a.rows,
            expected_cols: b.cols,
            got_rows: c.rows,
            got_cols: c.cols,
        });
    }

    let m_log = a.rows.next_power_of_two().ilog2() as usize;
    let n_log = b.cols.next_power_of_two().ilog2() as usize;

    // Mix matrix dimensions into the channel for Fiat-Shamir binding
    channel.mix_u64(a.rows as u64);
    channel.mix_u64(a.cols as u64);
    channel.mix_u64(b.cols as u64);

    // Step 1: Draw random challenges for row and column selection
    let row_challenges = channel.draw_secure_felts(m_log);
    let col_challenges = channel.draw_secure_felts(n_log);

    // Step 2: Evaluate MLE_C at (r_i, r_j) — this is the claimed inner product value
    let c_mle_evals: Vec<SecureField> = c.data.iter().map(|&v| SecureField::from(v)).collect();
    let c_padded = pad_to_pow2(&c_mle_evals);

    // MLE_C has variables (row_bits, col_bits), evaluate at (row_challenges, col_challenges)
    let mut point = Vec::with_capacity(m_log + n_log);
    point.extend_from_slice(&row_challenges);
    point.extend_from_slice(&col_challenges);
    let claimed_c_eval = eval_mle_at_point(&c_padded, &point);

    // Step 3: Extract row/column slices as MLEs over inner dimension k
    let a_row = extract_row_mle(a, &row_challenges);
    let b_col = extract_col_mle(b, &col_challenges);

    // Step 4: Create the inner product oracle and run sumcheck
    let oracle = InnerProductOracle::new(a_row, b_col)?;
    let lambda = SecureField::one(); // single claim, no batching needed

    let (sumcheck_proof, _assignment, _constant_oracles, _claimed_evals) =
        prove_batch(vec![claimed_c_eval], vec![oracle], lambda, channel);

    Ok((
        MatMulProof { sumcheck_proof },
        MatMulAux {
            row_challenges,
            col_challenges,
            claimed_c_eval,
            lambda,
        },
    ))
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

/// Verify that C = A × B using a sumcheck proof.
///
/// The verifier:
/// 1. Recomputes the random challenges (from the channel).
/// 2. Evaluates MLE_C at those challenges to get the claimed inner product.
/// 3. Runs sumcheck partial verification to get the variable assignment.
/// 4. Evaluates MLE_A and MLE_B at the assigned points and checks consistency.
pub fn verify_matmul(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
    proof: &MatMulProof,
    aux: &MatMulAux,
    channel: &mut impl Channel,
) -> Result<(), MatMulError> {
    let m_log = a.rows.next_power_of_two().ilog2() as usize;
    let n_log = b.cols.next_power_of_two().ilog2() as usize;

    // Reproduce the Fiat-Shamir transcript
    channel.mix_u64(a.rows as u64);
    channel.mix_u64(a.cols as u64);
    channel.mix_u64(b.cols as u64);

    let row_challenges = channel.draw_secure_felts(m_log);
    let col_challenges = channel.draw_secure_felts(n_log);

    // Verify challenges match
    if row_challenges != aux.row_challenges || col_challenges != aux.col_challenges {
        return Err(MatMulError::ChallengeMismatch);
    }

    // Step 1: Evaluate MLE_C at (r_i, r_j)
    let c_mle_evals: Vec<SecureField> = c.data.iter().map(|&v| SecureField::from(v)).collect();
    let c_padded = pad_to_pow2(&c_mle_evals);
    let mut point = Vec::with_capacity(m_log + n_log);
    point.extend_from_slice(&row_challenges);
    point.extend_from_slice(&col_challenges);
    let claimed_c_eval = eval_mle_at_point(&c_padded, &point);

    if claimed_c_eval != aux.claimed_c_eval {
        return Err(MatMulError::ClaimMismatch);
    }

    // Step 2: Run sumcheck partial verification
    let (assignment, claimed_eval) =
        partially_verify(claimed_c_eval, &proof.sumcheck_proof, channel)?;

    // Step 3: Verify the final evaluation
    // After sumcheck, the verifier needs to check:
    //   claimed_eval == f_A(assignment) * f_B(assignment)
    // where f_A and f_B are the row/col slices at the random challenges.
    let a_row = extract_row_mle(a, &row_challenges);
    let b_col = extract_col_mle(b, &col_challenges);

    let a_eval = eval_mle_at_point(&a_row, &assignment);
    let b_eval = eval_mle_at_point(&b_col, &assignment);

    if claimed_eval != a_eval * b_eval {
        return Err(MatMulError::FinalEvalMismatch);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::channel::Blake2sChannel;

    #[test]
    fn test_dims_speedup() {
        // 128×128 matmul: classic example
        let dims = MatMulDims::new(128, 128, 128);
        assert_eq!(dims.naive_trace_rows(), 2_097_152);
        assert_eq!(dims.sumcheck_trace_rows(), 49_152);
        assert!(dims.speedup() > 42.0);

        // 768×768 (BERT attention size): the real target
        let bert = MatMulDims::new(768, 768, 768);
        assert_eq!(bert.naive_trace_rows(), 452_984_832); // 453M — impossible naive
        assert_eq!(bert.sumcheck_trace_rows(), 1_769_472); // 1.7M — fits in 4M budget!
        assert!(bert.speedup() > 255.0);
    }

    #[test]
    fn test_m31_matrix() {
        let mut m = M31Matrix::new(2, 3);
        m.set(0, 0, M31::from(42));
        m.set(1, 2, M31::from(99));
        assert_eq!(m.get(0, 0), M31::from(42));
        assert_eq!(m.get(1, 2), M31::from(99));
    }

    #[test]
    fn test_matrix_multiply() {
        // 2×2 identity multiplication
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));

        let c = M31Matrix::multiply(&a, &b).unwrap();
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(c.get(0, 0), M31::from(19));
        assert_eq!(c.get(0, 1), M31::from(22));
        assert_eq!(c.get(1, 0), M31::from(43));
        assert_eq!(c.get(1, 1), M31::from(50));
    }

    #[test]
    fn test_eq_evals() {
        let one = SecureField::one();
        let zero = SecureField::zero();

        // eq((0), i) for i in {0, 1} should be [1, 0]
        let evals = eq_evals_at_point(&[zero], 2);
        assert_eq!(evals[0], one);
        assert_eq!(evals[1], zero);

        // eq((1), i) for i in {0, 1} should be [0, 1]
        let evals = eq_evals_at_point(&[one], 2);
        assert_eq!(evals[0], zero);
        assert_eq!(evals[1], one);
    }

    #[test]
    fn test_inner_product_oracle() {
        // Simple test: f_A = [1, 2], f_B = [3, 4]
        // Inner product = 1*3 + 2*4 = 11
        let a_evals = vec![
            SecureField::from(M31::from(1)),
            SecureField::from(M31::from(2)),
        ];
        let b_evals = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(4)),
        ];
        let claim = SecureField::from(M31::from(11));

        let oracle = InnerProductOracle::new(a_evals, b_evals).unwrap();
        assert_eq!(oracle.n_variables(), 1);

        let poly = oracle.sum_as_poly_in_first_variable(claim);
        // h(0) = a[0]*b[0] = 3, h(1) = a[1]*b[1] = 8
        assert_eq!(poly.eval_at_point(SecureField::zero()), SecureField::from(M31::from(3)));
        assert_eq!(poly.eval_at_point(SecureField::one()), SecureField::from(M31::from(8)));
    }

    #[test]
    fn test_prove_verify_2x2() {
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));

        let c = M31Matrix::multiply(&a, &b).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_matmul(&a, &b, &c, &proof, &aux, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_verify_4x4() {
        // Build a 4×4 matmul with known values
        let a = M31Matrix::from_data(
            4, 4,
            (1..=16).map(M31::from).collect(),
        ).unwrap();
        let b = M31Matrix::from_data(
            4, 4,
            (17..=32).map(M31::from).collect(),
        ).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_matmul(&a, &b, &c, &proof, &aux, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_verify_8x8() {
        let a = M31Matrix::from_data(
            8, 8,
            (1..=64).map(M31::from).collect(),
        ).unwrap();
        let b = M31Matrix::from_data(
            8, 8,
            (65..=128).map(M31::from).collect(),
        ).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_matmul(&a, &b, &c, &proof, &aux, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_verify_non_square() {
        // A is 4×2, B is 2×4, C is 4×4
        let a = M31Matrix::from_data(
            4, 2,
            (1..=8).map(M31::from).collect(),
        ).unwrap();
        let b = M31Matrix::from_data(
            2, 4,
            (1..=8).map(M31::from).collect(),
        ).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c, &mut prover_channel).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_matmul(&a, &b, &c, &proof, &aux, &mut verifier_channel).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_invalid_product_panics_prover() {
        // A correct prover cannot prove a false claim — prove_batch will panic
        // when the round polynomial sum doesn't match the claim.
        let a = M31Matrix::from_data(
            2, 2,
            vec![M31::from(1), M31::from(0), M31::from(0), M31::from(1)],
        ).unwrap();
        let b = M31Matrix::from_data(
            2, 2,
            vec![M31::from(5), M31::from(6), M31::from(7), M31::from(8)],
        ).unwrap();
        let mut c = M31Matrix::multiply(&a, &b).unwrap();
        c.set(0, 0, M31::from(999)); // corrupt the result

        let mut prover_channel = Blake2sChannel::default();
        let _ = prove_matmul(&a, &b, &c, &mut prover_channel);
    }

    #[test]
    fn test_verify_rejects_tampered_proof() {
        let a = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let b = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c, &mut prover_channel).unwrap();

        // Tamper with aux: use wrong row challenges
        let mut bad_aux = aux.clone();
        bad_aux.claimed_c_eval = SecureField::from(M31::from(999));

        let mut verifier_channel = Blake2sChannel::default();
        let result = verify_matmul(&a, &b, &c, &proof, &bad_aux, &mut verifier_channel);
        assert!(result.is_err(), "tampered proof should be rejected");
    }

    #[test]
    fn test_verify_rejects_wrong_matrix() {
        let a = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let b = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let mut prover_channel = Blake2sChannel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c, &mut prover_channel).unwrap();

        // Try to verify with different matrix C
        let mut c_bad = c.clone();
        c_bad.set(0, 0, M31::from(0));

        let mut verifier_channel = Blake2sChannel::default();
        let result = verify_matmul(&a, &b, &c_bad, &proof, &aux, &mut verifier_channel);
        assert!(result.is_err(), "wrong matrix should be rejected");
    }
}
