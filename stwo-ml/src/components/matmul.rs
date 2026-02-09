//! Sumcheck-based matrix multiplication verification.
//!
//! # The Core Innovation
//!
//! Traditional zkML (including ObelyskVM today) decomposes matrix multiplication
//! into individual multiply-add trace rows: O(m × k × n) rows for (m×k) × (k×n).
//!
//! This component uses the **sumcheck protocol over multilinear extensions** to
//! verify the same computation in O(m + k + n) verifier work:
//!
//! ```text
//! Traditional (ObelyskVM):
//!   C[i][j] = Σ_k A[i][k] × B[k][j]
//!   → Each multiply-add = 1 trace row
//!   → 128×128 MatMul = 2,097,152 rows (burns half the 4M cycle budget)
//!
//! Sumcheck approach (stwo-ml):
//!   Represent A, B, C as MLEs on boolean hypercube
//!   Prover claims: Σ_{x∈{0,1}^n} MLE_A(r_i, x) × MLE_B(x, r_j) = MLE_C(r_i, r_j)
//!   Verifier checks via n rounds of sumcheck
//!   → 128×128 MatMul ≈ 49,152 rows (42× reduction)
//!   → Bounded by O(m×n + m×k + k×n) for witness, O(log(m×k×n)) for verifier
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use stwo_ml::components::matmul::MatMulComponent;
//!
//! // Define matrix dimensions
//! let component = MatMulComponent::new(128, 64, 32); // A(128×64) × B(64×32) = C(128×32)
//!
//! // Prove with witness
//! let proof = component.prove(&a_matrix, &b_matrix, &c_matrix, &mut prover)?;
//! ```

use num_traits::{One, Zero};
use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::prover::lookups::sumcheck::{
    self, MultivariatePolyOracle, SumcheckProof,
};
use stwo::prover::lookups::utils::UnivariatePoly;

use crate::crypto::poseidon_channel::{PoseidonChannel, securefield_to_felt};
use crate::crypto::mle_opening::{MleOpeningProof, commit_mle, prove_mle_opening, verify_mle_opening};

/// Matrix dimensions for a matmul operation.
#[derive(Debug, Clone, Copy)]
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

    pub fn get(&self, i: usize, j: usize) -> M31 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, val: M31) {
        self.data[i * self.cols + j] = val;
    }
}

// ===== MLE Helpers =====

/// Evaluate a multilinear extension at a given point.
///
/// `evals` contains function values on the boolean hypercube {0,1}^n
/// (standard bit ordering: evals[b_{n-1}...b_1 b_0]).
/// `point` is the evaluation point in F^n.
///
/// Uses in-place folding to minimize memory allocation.
fn evaluate_mle(evals: &[SecureField], point: &[SecureField]) -> SecureField {
    assert_eq!(evals.len(), 1 << point.len(), "evals length must be 2^n_vars");
    let mut current: Vec<SecureField> = evals.to_vec();
    let mut size = current.len();
    for &r in point.iter() {
        let mid = size / 2;
        for i in 0..mid {
            // Fold in-place: f(r, x_rest) = (1-r)*f(0, x_rest) + r*f(1, x_rest)
            current[i] = current[i] + r * (current[mid + i] - current[i]);
        }
        size = mid;
    }
    assert_eq!(size, 1);
    current[0]
}

/// Fix the first `assignments.len()` variables of an MLE, returning
/// the reduced evaluations over the remaining variables.
///
/// Uses in-place folding to minimize memory allocation.
fn restrict_mle(evals: &[SecureField], assignments: &[SecureField]) -> Vec<SecureField> {
    let mut current: Vec<SecureField> = evals.to_vec();
    let mut size = current.len();
    for &r in assignments.iter() {
        let mid = size / 2;
        for i in 0..mid {
            current[i] = current[i] + r * (current[mid + i] - current[i]);
        }
        size = mid;
    }
    current.truncate(size);
    current
}

/// Convert a row-major M31Matrix into MLE evaluations (SecureField).
/// Layout: A[i][j] at index i*cols + j.
/// Requires rows*cols to be a power of 2.
fn matrix_to_mle(matrix: &M31Matrix) -> Vec<SecureField> {
    let n = matrix.rows * matrix.cols;
    assert!(n.is_power_of_two(), "matrix size must be power of 2");
    matrix.data.iter().map(|&v| SecureField::from(v)).collect()
}

/// Convert an M31Matrix into an MLE with transposed variable ordering.
/// B[i][j] is stored at index j*rows + i (column-major).
/// This allows fixing the column variables (j) first via restrict_mle.
///
/// For large matrices (>= 65536 elements), uses rayon for parallelism.
fn matrix_to_mle_col_major(matrix: &M31Matrix) -> Vec<SecureField> {
    let n = matrix.rows * matrix.cols;
    assert!(n.is_power_of_two(), "matrix size must be power of 2");

    if n >= 65536 {
        // Parallel: build per-column chunks
        use rayon::prelude::*;
        let mut evals = vec![SecureField::zero(); n];
        evals.par_chunks_mut(matrix.rows).enumerate().for_each(|(j, col_chunk)| {
            for i in 0..matrix.rows {
                col_chunk[i] = SecureField::from(matrix.data[i * matrix.cols + j]);
            }
        });
        evals
    } else {
        let mut evals = vec![SecureField::zero(); n];
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                evals[j * matrix.rows + i] = SecureField::from(matrix.get(i, j));
            }
        }
        evals
    }
}

/// Compute C = A × B in M31 arithmetic.
///
/// For small matrices (< 64 rows), uses a simple triple loop.
/// For larger matrices, uses rayon parallelism with transposed B
/// for cache-friendly access, giving ~4-8x speedup on multi-core.
pub fn matmul_m31(a: &M31Matrix, b: &M31Matrix) -> M31Matrix {
    assert_eq!(a.cols, b.rows, "A.cols must equal B.rows");
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    // Small matrices: simple loop (avoids rayon overhead)
    if m < 64 {
        let mut c = M31Matrix::new(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = M31::from(0);
                for l in 0..k {
                    sum += a.data[i * k + l] * b.data[l * n + j];
                }
                c.data[i * n + j] = sum;
            }
        }
        return c;
    }

    // Large matrices: transpose B for cache-friendly dot products + rayon
    use rayon::prelude::*;

    let mut b_t = vec![M31::from(0); k * n];
    for i in 0..k {
        for j in 0..n {
            b_t[j * k + i] = b.data[i * n + j];
        }
    }

    let mut c_data = vec![M31::from(0); m * n];
    c_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let a_row = &a.data[i * k..(i + 1) * k];
        for j in 0..n {
            let b_col = &b_t[j * k..(j + 1) * k];
            let mut sum = M31::from(0);
            for l in 0..k {
                sum += a_row[l] * b_col[l];
            }
            row[j] = sum;
        }
    });

    M31Matrix { rows: m, cols: n, data: c_data }
}

/// Pad a matrix to the next power-of-2 dimensions (zero-padding).
///
/// Required by sumcheck protocol which operates on boolean hypercubes.
/// If dimensions are already powers of 2, returns a clone.
pub fn pad_matrix_pow2(matrix: &M31Matrix) -> M31Matrix {
    let new_rows = matrix.rows.next_power_of_two();
    let new_cols = matrix.cols.next_power_of_two();

    if new_rows == matrix.rows && new_cols == matrix.cols {
        return matrix.clone();
    }

    let mut padded = M31Matrix::new(new_rows, new_cols);
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            padded.data[i * new_cols + j] = matrix.data[i * matrix.cols + j];
        }
    }
    padded
}

/// Estimate memory needed for matmul sumcheck proving (in bytes).
///
/// Accounts for: 3 MLEs (A, B_t, C) + restrict_mle temporaries + sumcheck.
/// Returns (mle_bytes, total_estimate_bytes).
pub fn estimate_sumcheck_memory(m: usize, k: usize, n: usize) -> (usize, usize) {
    let sf_size = 16; // SecureField = QM31 = 4 × M31 = 16 bytes

    // MLEs: A is m×k, B_t is k×n (stored as n×k col-major), C is m×n
    let mle_a = m * k * sf_size;
    let mle_b = k * n * sf_size;
    let mle_c = m * n * sf_size;
    let mle_total = mle_a + mle_b + mle_c;

    // restrict_mle: for A, starts at m*k elements, folds log_m times → k elements
    // Allocates ~2*m*k elements total (geometric sum). Same for B_t.
    let restrict_a = 2 * m * k * sf_size;
    let restrict_b = 2 * k * n * sf_size;

    // Sumcheck oracle: 2 × k elements (f_a + f_b)
    let oracle = 2 * k * sf_size;

    // Input M31 matrices
    let input_m31 = (m * k + k * n + m * n) * 4;

    let total = mle_total + restrict_a + restrict_b + oracle + input_m31;
    (mle_total, total)
}

// ===== Sumcheck Oracle =====

/// Oracle for the matmul inner product sumcheck:
///   g(x) = f_a(x) × f_b(x)
/// where f_a = MLE_A restricted to random row point,
///       f_b = MLE_B restricted to random col point.
/// Degree per variable = 2 (product of two multilinear), within MAX_DEGREE=3.
pub struct MatMulOracle {
    /// MLE_A(r_i, x) evaluations on {0,1}^{log k}
    pub f_a: Vec<SecureField>,
    /// MLE_B(x, r_j) evaluations on {0,1}^{log k}
    pub f_b: Vec<SecureField>,
}

impl MultivariatePolyOracle for MatMulOracle {
    fn n_variables(&self) -> usize {
        assert_eq!(self.f_a.len(), self.f_b.len());
        self.f_a.len().ilog2() as usize
    }

    /// Compute S(t) = Σ_{x_1,...} f_a(t, x_1,...) × f_b(t, x_1,...)
    /// This is a degree-2 univariate polynomial in t.
    fn sum_as_poly_in_first_variable(&self, _claim: SecureField) -> UnivariatePoly<SecureField> {
        let n = self.f_a.len();
        let mid = n / 2;

        // Evaluate at t=0, t=1, t=2 for degree-2 Lagrange interpolation
        let mut s0 = SecureField::zero(); // S(0) = Σ a_lo × b_lo
        let mut s1 = SecureField::zero(); // S(1) = Σ a_hi × b_hi
        let mut s2 = SecureField::zero(); // S(2) = Σ (2a_hi - a_lo)(2b_hi - b_lo)

        for i in 0..mid {
            let a0 = self.f_a[i];
            let a1 = self.f_a[mid + i];
            let b0 = self.f_b[i];
            let b1 = self.f_b[mid + i];

            s0 += a0 * b0;
            s1 += a1 * b1;
            // At t=2: f(2,...) = (1-2)*f(0,...) + 2*f(1,...) = 2*f_hi - f_lo
            let a2 = a1 + a1 - a0;
            let b2 = b1 + b1 - b0;
            s2 += a2 * b2;
        }

        // Lagrange interpolation through (0, s0), (1, s1), (2, s2)
        let two = SecureField::from(M31::from(2));
        UnivariatePoly::interpolate_lagrange(
            &[SecureField::zero(), SecureField::one(), two],
            &[s0, s1, s2],
        )
    }

    /// Fix the first variable to `challenge`, folding both MLEs.
    fn fix_first_variable(self, challenge: SecureField) -> Self {
        let n = self.f_a.len();
        let mid = n / 2;

        let mut new_a = Vec::with_capacity(mid);
        let mut new_b = Vec::with_capacity(mid);

        for i in 0..mid {
            new_a.push(self.f_a[i] + challenge * (self.f_a[mid + i] - self.f_a[i]));
            new_b.push(self.f_b[i] + challenge * (self.f_b[mid + i] - self.f_b[i]));
        }

        MatMulOracle { f_a: new_a, f_b: new_b }
    }
}

// ===== Sumcheck Proof =====

/// Proof that C = A × B, generated via the sumcheck protocol.
#[derive(Debug, Clone)]
pub struct MatMulSumcheckProof {
    /// The inner sumcheck proof (round polynomials).
    pub sumcheck_proof: SumcheckProof,
    /// Random row evaluation point r_i ∈ F^{log m}.
    pub r_i: Vec<SecureField>,
    /// Random column evaluation point r_j ∈ F^{log n}.
    pub r_j: Vec<SecureField>,
    /// Claimed value: MLE_C(r_i, r_j).
    pub claimed_sum: SecureField,
    /// Final MLE_A evaluation at (r_i, r_k) where r_k = sumcheck assignment.
    pub final_a_eval: SecureField,
    /// Final MLE_B evaluation at (r_k, r_j).
    pub final_b_eval: SecureField,
    /// Sumcheck variable assignment (challenges from each round).
    pub assignment: Vec<SecureField>,
}

/// Error type for matmul proving/verification.
#[derive(Debug, thiserror::Error)]
pub enum MatMulError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Dimensions must be powers of 2: {0}")]
    NonPowerOfTwo(String),
    #[error("Sumcheck verification failed: {0}")]
    SumcheckFailed(String),
    #[error("Evaluation mismatch at final check: expected {expected}, got {actual}")]
    EvaluationMismatch { expected: SecureField, actual: SecureField },
    #[error("Claimed sum mismatch: expected {expected}, got {actual}")]
    ClaimedSumMismatch { expected: SecureField, actual: SecureField },
}

/// Prove that C = A × B using the sumcheck protocol over multilinear extensions.
///
/// Protocol:
/// 1. Build MLEs for A (row-major), B (col-major for restriction), C (row-major)
/// 2. Draw random evaluation points r_i, r_j via Fiat-Shamir
/// 3. Claim: Σ_{x ∈ {0,1}^{log k}} MLE_A(r_i, x) × MLE_B(x, r_j) = MLE_C(r_i, r_j)
/// 4. Run sumcheck on g(x) = MLE_A(r_i, x) × MLE_B(x, r_j)
/// 5. Return proof with final oracle evaluations
pub fn prove_matmul_sumcheck(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<MatMulSumcheckProof, MatMulError> {
    // Validate dimensions
    if a.cols != b.rows {
        return Err(MatMulError::DimensionMismatch(
            format!("A.cols={} != B.rows={}", a.cols, b.rows),
        ));
    }
    if c.rows != a.rows || c.cols != b.cols {
        return Err(MatMulError::DimensionMismatch(
            format!("C({},{}) != expected ({},{})", c.rows, c.cols, a.rows, b.cols),
        ));
    }

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    for (name, val) in [("m", m), ("k", k), ("n", n)] {
        if !val.is_power_of_two() {
            return Err(MatMulError::NonPowerOfTwo(format!("{name}={val}")));
        }
    }

    let log_m = m.ilog2() as usize;
    let _log_k = k.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    // Build MLEs
    let mle_a = matrix_to_mle(a);           // row-major: (row_bits, col_bits)
    let mle_b_t = matrix_to_mle_col_major(b); // col-major: (col_bits, row_bits)
    let mle_c = matrix_to_mle(c);           // row-major: (row_bits, col_bits)

    // Fiat-Shamir: seed channel with dimensions
    let mut channel = Blake2sChannel::default();
    channel.mix_felts(&[
        SecureField::from(M31::from(m as u32)),
        SecureField::from(M31::from(k as u32)),
        SecureField::from(M31::from(n as u32)),
    ]);

    // Draw random evaluation points
    let r_i = channel.draw_secure_felts(log_m);
    let r_j = channel.draw_secure_felts(log_n);

    // Compute claimed sum: MLE_C(r_i, r_j)
    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&r_j);
    let claimed_sum = evaluate_mle(&mle_c, &r_ij);

    // Mix claimed sum into channel for binding
    channel.mix_felts(&[claimed_sum]);

    // Restrict MLEs to random points
    // f_a(x) = MLE_A(r_i, x) for x ∈ {0,1}^{log k} — fix first log_m vars
    let f_a = restrict_mle(&mle_a, &r_i);
    // f_b(x) = MLE_B(x, r_j) — using transposed layout, fix first log_n vars
    let f_b = restrict_mle(&mle_b_t, &r_j);

    assert_eq!(f_a.len(), k, "f_a should have k={k} elements");
    assert_eq!(f_b.len(), k, "f_b should have k={k} elements");

    // Build oracle and run sumcheck
    let oracle = MatMulOracle { f_a, f_b };
    let lambda = SecureField::one();
    let (sumcheck_proof, assignment, final_oracles, _claimed_evals) =
        sumcheck::prove_batch(vec![claimed_sum], vec![oracle], lambda, &mut channel);

    // Extract final single-point evaluations
    let final_oracle = &final_oracles[0];
    assert_eq!(final_oracle.f_a.len(), 1);
    assert_eq!(final_oracle.f_b.len(), 1);
    let final_a_eval = final_oracle.f_a[0];
    let final_b_eval = final_oracle.f_b[0];

    Ok(MatMulSumcheckProof {
        sumcheck_proof,
        r_i,
        r_j,
        claimed_sum,
        final_a_eval,
        final_b_eval,
        assignment,
    })
}

/// Verify a matmul sumcheck proof against the original matrices.
///
/// Checks:
/// 1. MLE_C(r_i, r_j) matches the claimed sum
/// 2. Sumcheck round polynomials are valid (via STWO's partially_verify)
/// 3. Final evaluation matches MLE_A(r_i, r_k) × MLE_B(r_k, r_j)
pub fn verify_matmul_sumcheck(
    proof: &MatMulSumcheckProof,
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<(), MatMulError> {
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let log_m = m.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    // Recreate channel with identical Fiat-Shamir transcript
    let mut channel = Blake2sChannel::default();
    channel.mix_felts(&[
        SecureField::from(M31::from(m as u32)),
        SecureField::from(M31::from(k as u32)),
        SecureField::from(M31::from(n as u32)),
    ]);

    // Draw same r_i, r_j
    let r_i = channel.draw_secure_felts(log_m);
    let r_j = channel.draw_secure_felts(log_n);
    assert_eq!(r_i, proof.r_i, "r_i mismatch — channel desync");
    assert_eq!(r_j, proof.r_j, "r_j mismatch — channel desync");

    // Verify claimed sum = MLE_C(r_i, r_j)
    let mle_c = matrix_to_mle(c);
    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&r_j);
    let expected_sum = evaluate_mle(&mle_c, &r_ij);

    if expected_sum != proof.claimed_sum {
        return Err(MatMulError::ClaimedSumMismatch {
            expected: expected_sum,
            actual: proof.claimed_sum,
        });
    }

    // Mix claimed sum (same as prover)
    channel.mix_felts(&[proof.claimed_sum]);

    // Run sumcheck partial verification
    let (assignment, claimed_eval) =
        sumcheck::partially_verify(proof.claimed_sum, &proof.sumcheck_proof, &mut channel)
            .map_err(|e| MatMulError::SumcheckFailed(format!("{e}")))?;

    // Verify final evaluations against MLEs
    let mle_a = matrix_to_mle(a);
    let mle_b_t = matrix_to_mle_col_major(b);

    // MLE_A(r_i, assignment) — full point is (r_i ++ assignment)
    let mut r_full_a = Vec::with_capacity(r_i.len() + assignment.len());
    r_full_a.extend_from_slice(&r_i);
    r_full_a.extend_from_slice(&assignment);
    let expected_a = evaluate_mle(&mle_a, &r_full_a);

    // MLE_B_T(r_j, assignment) — transposed, so (r_j ++ assignment)
    let mut r_full_b = Vec::with_capacity(r_j.len() + assignment.len());
    r_full_b.extend_from_slice(&r_j);
    r_full_b.extend_from_slice(&assignment);
    let expected_b = evaluate_mle(&mle_b_t, &r_full_b);

    // The claimed eval from sumcheck should equal f_a(assignment) * f_b(assignment)
    let expected_product = expected_a * expected_b;
    if claimed_eval != expected_product {
        return Err(MatMulError::EvaluationMismatch {
            expected: expected_product,
            actual: claimed_eval,
        });
    }

    Ok(())
}

// ===== On-Chain Proof Structures =====

/// Degree-2 round polynomial: `p(x) = c0 + c1*x + c2*x^2`.
#[derive(Debug, Clone, Copy)]
pub struct RoundPoly {
    pub c0: SecureField,
    pub c1: SecureField,
    pub c2: SecureField,
}

/// MatMul proof formatted for on-chain Cairo verification.
///
/// Matches Cairo's `MatMulSumcheckProof` with 12 fields, using Poseidon
/// Fiat-Shamir channel and MLE commitments instead of Blake2s.
#[derive(Debug, Clone)]
pub struct MatMulSumcheckProofOnChain {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub num_rounds: u32,
    pub claimed_sum: SecureField,
    pub round_polys: Vec<RoundPoly>,
    pub final_a_eval: SecureField,
    pub final_b_eval: SecureField,
    pub a_commitment: FieldElement,
    pub b_commitment: FieldElement,
    pub a_opening: MleOpeningProof,
    pub b_opening: MleOpeningProof,
}

/// Prove C = A × B using the sumcheck protocol with Poseidon Fiat-Shamir
/// channel and MLE Poseidon Merkle commitments, formatted for on-chain
/// Cairo verification.
///
/// Protocol (must match Cairo verifier exactly):
/// 1. Commit to restricted MLEs via Poseidon Merkle trees
/// 2. Fiat-Shamir via PoseidonChannel (mix dimensions, draw r_i/r_j, etc.)
/// 3. Sumcheck with degree-2 round polynomials
/// 4. MLE opening proofs for final evaluations
pub fn prove_matmul_sumcheck_onchain(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<MatMulSumcheckProofOnChain, MatMulError> {
    // Validate dimensions
    if a.cols != b.rows {
        return Err(MatMulError::DimensionMismatch(
            format!("A.cols={} != B.rows={}", a.cols, b.rows),
        ));
    }
    if c.rows != a.rows || c.cols != b.cols {
        return Err(MatMulError::DimensionMismatch(
            format!("C({},{}) != expected ({},{})", c.rows, c.cols, a.rows, b.cols),
        ));
    }

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    for (name, val) in [("m", m), ("k", k), ("n", n)] {
        if !val.is_power_of_two() {
            return Err(MatMulError::NonPowerOfTwo(format!("{name}={val}")));
        }
    }

    let log_m = m.ilog2() as usize;
    let log_k = k.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    // Build MLEs
    let mle_a = matrix_to_mle(a);
    let mle_b_t = matrix_to_mle_col_major(b);
    let mle_c = matrix_to_mle(c);

    // PoseidonChannel for Fiat-Shamir
    let mut channel = PoseidonChannel::new();

    // Mix dimensions
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);

    // Draw random evaluation points
    let r_i = channel.draw_qm31s(log_m);
    let r_j = channel.draw_qm31s(log_n);

    // Compute claimed sum: MLE_C(r_i, r_j)
    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&r_j);
    let claimed_sum = evaluate_mle(&mle_c, &r_ij);

    // Mix claimed sum
    channel.mix_felt(securefield_to_felt(claimed_sum));

    // Restrict MLEs to random points
    let f_a = restrict_mle(&mle_a, &r_i);
    let f_b = restrict_mle(&mle_b_t, &r_j);

    assert_eq!(f_a.len(), k);
    assert_eq!(f_b.len(), k);

    // Commit to restricted MLEs
    let (a_commitment, _a_tree) = commit_mle(&f_a);
    let (b_commitment, _b_tree) = commit_mle(&f_b);

    // Mix commitments
    channel.mix_felt(a_commitment);
    channel.mix_felt(b_commitment);

    // === Sumcheck with Poseidon channel ===
    let num_rounds = log_k;
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut assignment = Vec::with_capacity(num_rounds);
    let mut cur_a = f_a.clone();
    let mut cur_b = f_b.clone();

    for _round in 0..num_rounds {
        let half = cur_a.len() / 2;

        // Evaluate S(t) at t=0, t=1, t=2
        let mut s0 = SecureField::zero();
        let mut s1 = SecureField::zero();
        let mut s2 = SecureField::zero();

        for i in 0..half {
            let a0 = cur_a[i];
            let a1 = cur_a[half + i];
            let b0 = cur_b[i];
            let b1 = cur_b[half + i];

            s0 += a0 * b0;
            s1 += a1 * b1;
            let a2 = a1 + a1 - a0;
            let b2 = b1 + b1 - b0;
            s2 += a2 * b2;
        }

        // Extract coefficients: p(x) = c0 + c1*x + c2*x^2
        // c0 = s0
        // c0 + c1 + c2 = s1
        // c0 + 2*c1 + 4*c2 = s2
        let c0 = s0;
        let two = SecureField::from(M31::from(2));
        let c2 = (s2 - two * s1 + s0) * SecureField::from(M31::from(2)).inverse();
        let c1 = s1 - s0 - c2;

        let rp = RoundPoly { c0, c1, c2 };
        round_polys.push(rp);

        // Mix polynomial into channel
        channel.mix_poly_coeffs(c0, c1, c2);

        // Draw challenge
        let r_k = channel.draw_qm31();
        assignment.push(r_k);

        // Fold both MLEs
        let mut new_a = Vec::with_capacity(half);
        let mut new_b = Vec::with_capacity(half);
        for i in 0..half {
            new_a.push(cur_a[i] + r_k * (cur_a[half + i] - cur_a[i]));
            new_b.push(cur_b[i] + r_k * (cur_b[half + i] - cur_b[i]));
        }
        cur_a = new_a;
        cur_b = new_b;
    }

    assert_eq!(cur_a.len(), 1);
    assert_eq!(cur_b.len(), 1);
    let final_a_eval = cur_a[0];
    let final_b_eval = cur_b[0];

    // MLE opening proofs
    let a_opening = prove_mle_opening(&f_a, &assignment, &mut channel);
    let b_opening = prove_mle_opening(&f_b, &assignment, &mut channel);

    Ok(MatMulSumcheckProofOnChain {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        num_rounds: num_rounds as u32,
        claimed_sum,
        round_polys,
        final_a_eval,
        final_b_eval,
        a_commitment,
        b_commitment,
        a_opening,
        b_opening,
    })
}

/// Verify an on-chain matmul sumcheck proof (local pre-flight check).
///
/// Replays the Fiat-Shamir transcript using PoseidonChannel, verifies
/// round polynomial consistency, and checks final evaluations.
/// Does NOT need original matrices (uses commitments).
pub fn verify_matmul_sumcheck_onchain(
    proof: &MatMulSumcheckProofOnChain,
) -> Result<(), MatMulError> {
    let m = proof.m as usize;
    let k = proof.k as usize;
    let n = proof.n as usize;
    let log_m = m.ilog2() as usize;
    let log_k = k.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    if proof.num_rounds as usize != log_k {
        return Err(MatMulError::SumcheckFailed(
            format!("num_rounds {} != log_k {}", proof.num_rounds, log_k),
        ));
    }
    if proof.round_polys.len() != log_k {
        return Err(MatMulError::SumcheckFailed(
            "round_polys count mismatch".into(),
        ));
    }

    // Recreate PoseidonChannel
    let mut channel = PoseidonChannel::new();
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);

    let _r_i = channel.draw_qm31s(log_m);
    let _r_j = channel.draw_qm31s(log_n);

    // Mix claimed sum
    channel.mix_felt(securefield_to_felt(proof.claimed_sum));

    // Mix commitments
    channel.mix_felt(proof.a_commitment);
    channel.mix_felt(proof.b_commitment);

    // Verify sumcheck rounds, collecting challenges for MLE opening verification
    let mut current_sum = proof.claimed_sum;
    let mut assignment = Vec::with_capacity(log_k);

    for rp in &proof.round_polys {
        // Check: p(0) + p(1) = current_sum
        let p_at_0 = rp.c0;
        let p_at_1 = rp.c0 + rp.c1 + rp.c2;
        let round_sum = p_at_0 + p_at_1;

        if round_sum != current_sum {
            return Err(MatMulError::SumcheckFailed(
                format!("round sum {round_sum:?} != expected {current_sum:?}"),
            ));
        }

        // Mix polynomial into channel
        channel.mix_poly_coeffs(rp.c0, rp.c1, rp.c2);

        // Draw challenge
        let r = channel.draw_qm31();
        assignment.push(r);

        // Update current_sum = p(r) = c0 + c1*r + c2*r^2
        current_sum = rp.c0 + rp.c1 * r + rp.c2 * r * r;
    }

    // Final check: current_sum should equal final_a_eval * final_b_eval
    let expected_product = proof.final_a_eval * proof.final_b_eval;
    if current_sum != expected_product {
        return Err(MatMulError::EvaluationMismatch {
            expected: expected_product,
            actual: current_sum,
        });
    }

    // Verify MLE opening proofs match the claimed final evaluations
    if proof.a_opening.final_value != proof.final_a_eval {
        return Err(MatMulError::SumcheckFailed(
            format!(
                "a_opening.final_value {:?} != final_a_eval {:?}",
                proof.a_opening.final_value, proof.final_a_eval
            ),
        ));
    }
    if proof.b_opening.final_value != proof.final_b_eval {
        return Err(MatMulError::SumcheckFailed(
            format!(
                "b_opening.final_value {:?} != final_b_eval {:?}",
                proof.b_opening.final_value, proof.final_b_eval
            ),
        ));
    }

    // Verify MLE opening proof for A against its commitment.
    // Channel state must match prover's state after sumcheck rounds.
    if !verify_mle_opening(proof.a_commitment, &proof.a_opening, &assignment, &mut channel) {
        return Err(MatMulError::SumcheckFailed(
            "MLE opening verification failed for matrix A".into(),
        ));
    }

    // Verify MLE opening proof for B against its commitment.
    // Channel continues sequentially from A's verification.
    if !verify_mle_opening(proof.b_commitment, &proof.b_opening, &assignment, &mut channel) {
        return Err(MatMulError::SumcheckFailed(
            "MLE opening verification failed for matrix B".into(),
        ));
    }

    Ok(())
}

/// Make evaluate_mle and restrict_mle publicly accessible for the crypto module.
pub fn evaluate_mle_pub(evals: &[SecureField], point: &[SecureField]) -> SecureField {
    evaluate_mle(evals, point)
}

pub fn restrict_mle_pub(evals: &[SecureField], assignments: &[SecureField]) -> Vec<SecureField> {
    restrict_mle(evals, assignments)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_matmul_m31() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3)); a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5)); b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7)); b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);
        assert_eq!(c.get(0, 0), M31::from(19));
        assert_eq!(c.get(0, 1), M31::from(22));
        assert_eq!(c.get(1, 0), M31::from(43));
        assert_eq!(c.get(1, 1), M31::from(50));
    }

    #[test]
    fn test_mle_evaluate() {
        // MLE of f(0)=1, f(1)=3 → f(x) = 1 + 2x
        let evals = vec![SecureField::from(M31::from(1)), SecureField::from(M31::from(3))];
        let point = vec![SecureField::zero()];
        assert_eq!(evaluate_mle(&evals, &point), SecureField::from(M31::from(1)));

        let point = vec![SecureField::one()];
        assert_eq!(evaluate_mle(&evals, &point), SecureField::from(M31::from(3)));
    }

    #[test]
    fn test_sumcheck_matmul_2x2() {
        // Smallest meaningful test: 2×2 × 2×2
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3)); a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5)); b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7)); b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);

        // Prove
        let proof = prove_matmul_sumcheck(&a, &b, &c).expect("proving should succeed");

        // Verify
        verify_matmul_sumcheck(&proof, &a, &b, &c).expect("verification should succeed");
    }

    #[test]
    fn test_sumcheck_matmul_4x4() {
        // 4×4 × 4×4 — 2 sumcheck rounds over inner dimension
        let mut a = M31Matrix::new(4, 4);
        let mut b = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                a.set(i, j, M31::from((i * 4 + j + 1) as u32));
                b.set(i, j, M31::from((i * 4 + j + 17) as u32));
            }
        }

        let c = matmul_m31(&a, &b);

        let proof = prove_matmul_sumcheck(&a, &b, &c).expect("4x4 proving should succeed");
        assert_eq!(proof.assignment.len(), 2, "log2(4) = 2 sumcheck rounds");

        verify_matmul_sumcheck(&proof, &a, &b, &c).expect("4x4 verification should succeed");
    }

    #[test]
    fn test_sumcheck_matmul_tampered_c_fails_verification() {
        // Prove with correct matrices
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3)); a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5)); b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7)); b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);
        let proof = prove_matmul_sumcheck(&a, &b, &c).expect("proving should succeed");

        // Tamper with C and try to verify — should fail at claimed sum check
        let mut c_wrong = c.clone();
        c_wrong.set(0, 0, M31::from(999));

        let result = verify_matmul_sumcheck(&proof, &a, &b, &c_wrong);
        assert!(result.is_err(), "verification with tampered C should fail");
    }

    #[test]
    fn test_sumcheck_matmul_rectangular() {
        // 2×4 × 4×2 = 2×2 — non-square matrices
        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..2 {
            for j in 0..4 {
                a.set(i, j, M31::from((i * 4 + j + 1) as u32));
            }
        }
        for i in 0..4 {
            for j in 0..2 {
                b.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }

        let c = matmul_m31(&a, &b);

        let proof = prove_matmul_sumcheck(&a, &b, &c).expect("rectangular proving should succeed");
        verify_matmul_sumcheck(&proof, &a, &b, &c).expect("rectangular verification should succeed");
    }

    // === On-Chain MatMul Tests ===

    #[test]
    fn test_matmul_onchain_2x2() {
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3)); a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5)); b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7)); b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);

        let proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
            .expect("on-chain 2x2 proving should succeed");
        assert_eq!(proof.m, 2);
        assert_eq!(proof.k, 2);
        assert_eq!(proof.n, 2);
        assert_eq!(proof.num_rounds, 1);
        assert_eq!(proof.round_polys.len(), 1);
        assert_ne!(proof.a_commitment, FieldElement::ZERO);
        assert_ne!(proof.b_commitment, FieldElement::ZERO);

        verify_matmul_sumcheck_onchain(&proof)
            .expect("on-chain 2x2 verification should succeed");
    }

    #[test]
    fn test_matmul_onchain_4x4() {
        let mut a = M31Matrix::new(4, 4);
        let mut b = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                a.set(i, j, M31::from((i * 4 + j + 1) as u32));
                b.set(i, j, M31::from((i * 4 + j + 17) as u32));
            }
        }

        let c = matmul_m31(&a, &b);

        let proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
            .expect("on-chain 4x4 proving should succeed");
        assert_eq!(proof.num_rounds, 2);
        assert_eq!(proof.round_polys.len(), 2);

        verify_matmul_sumcheck_onchain(&proof)
            .expect("on-chain 4x4 verification should succeed");
    }

    #[test]
    fn test_matmul_onchain_tampered_fails() {
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3)); a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5)); b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7)); b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);

        let mut proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
            .expect("proving should succeed");

        // Tamper with claimed sum
        proof.claimed_sum = SecureField::from(M31::from(999));

        let result = verify_matmul_sumcheck_onchain(&proof);
        assert!(result.is_err(), "tampered proof should fail verification");
    }

    #[test]
    fn test_matmul_onchain_rectangular() {
        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..2 {
            for j in 0..4 {
                a.set(i, j, M31::from((i * 4 + j + 1) as u32));
            }
        }
        for i in 0..4 {
            for j in 0..2 {
                b.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }

        let c = matmul_m31(&a, &b);

        let proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
            .expect("rectangular on-chain proving should succeed");
        verify_matmul_sumcheck_onchain(&proof)
            .expect("rectangular on-chain verification should succeed");
    }

    // === Fix 2 Tests: MLE opening verification in matmul verifier ===

    #[test]
    fn test_matmul_onchain_verifies_mle_openings() {
        // Full round-trip: prove and verify with MLE opening checks
        let mut a = M31Matrix::new(4, 4);
        let mut b = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                a.set(i, j, M31::from((i * 4 + j + 1) as u32));
                b.set(i, j, M31::from((i * 4 + j + 17) as u32));
            }
        }

        let c = matmul_m31(&a, &b);

        let proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
            .expect("proving should succeed");

        // Verify — now includes MLE opening proof checks
        verify_matmul_sumcheck_onchain(&proof)
            .expect("full verification with MLE openings should succeed");
    }

    #[test]
    fn test_matmul_onchain_tampered_a_opening_fails() {
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3)); a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5)); b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7)); b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);

        let mut proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
            .expect("proving should succeed");

        // Tamper with a_opening final_value (will mismatch final_a_eval)
        proof.a_opening.final_value = SecureField::from(M31::from(12345));

        let result = verify_matmul_sumcheck_onchain(&proof);
        assert!(result.is_err(), "tampered a_opening should fail verification");
    }

    #[test]
    fn test_matmul_onchain_tampered_b_commitment_fails() {
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3)); a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5)); b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7)); b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);

        let mut proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
            .expect("proving should succeed");

        // Tamper with b_commitment — MLE opening Merkle check should fail
        proof.b_commitment = FieldElement::from(99999u64);

        let result = verify_matmul_sumcheck_onchain(&proof);
        assert!(result.is_err(), "tampered b_commitment should fail verification");
    }

    #[test]
    fn test_matmul_onchain_final_eval_mismatch() {
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1)); a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3)); a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5)); b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7)); b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);

        let mut proof = prove_matmul_sumcheck_onchain(&a, &b, &c)
            .expect("proving should succeed");

        // Tamper with final_a_eval but keep a_opening.final_value consistent
        // This will fail at the product check (current_sum != a*b)
        let orig = proof.final_a_eval;
        proof.final_a_eval = orig + SecureField::from(M31::from(1));

        let result = verify_matmul_sumcheck_onchain(&proof);
        assert!(result.is_err(), "mismatched final_eval should fail verification");
    }
}
