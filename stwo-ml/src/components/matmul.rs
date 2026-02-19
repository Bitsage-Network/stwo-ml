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
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::FieldExpOps;
use stwo::prover::lookups::sumcheck::{self, MultivariatePolyOracle, SumcheckProof};
use stwo::prover::lookups::utils::UnivariatePoly;

use crate::crypto::mle_opening::{
    commit_mle_root_only, prove_mle_opening, verify_mle_opening, MleOpeningProof,
};
use crate::crypto::poseidon_channel::{securefield_to_felt, PoseidonChannel};

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
    assert_eq!(
        evals.len(),
        1 << point.len(),
        "evals length must be 2^n_vars"
    );
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
/// Uses in-place folding. Parallelized with rayon when mid >= 8192
/// (each fold level is embarrassingly parallel: current[i] and current[mid+i]
/// are independent across different i).
fn restrict_mle(evals: &[SecureField], assignments: &[SecureField]) -> Vec<SecureField> {
    use rayon::prelude::*;
    let mut current: Vec<SecureField> = evals.to_vec();
    let mut size = current.len();
    for &r in assignments.iter() {
        let mid = size / 2;
        if mid >= 8192 {
            // Parallel: split into non-overlapping lo/hi halves
            let (lo, hi) = current[..size].split_at_mut(mid);
            let hi = &*hi; // reborrow as shared for parallel read
            lo.par_iter_mut().enumerate().for_each(|(i, val)| {
                *val = *val + r * (hi[i] - *val);
            });
        } else {
            // Sequential: rayon overhead not worth it for small sizes
            for i in 0..mid {
                current[i] = current[i] + r * (current[mid + i] - current[i]);
            }
        }
        size = mid;
    }
    current.truncate(size);
    current.shrink_to_fit();
    current
}

/// Convert a row-major M31Matrix into MLE evaluations (SecureField).
/// Layout: A[i][j] at index i*cols + j.
/// Requires rows*cols to be a power of 2.
/// Parallelized with rayon for large matrices (>= 65536 elements).
fn matrix_to_mle(matrix: &M31Matrix) -> Vec<SecureField> {
    use rayon::prelude::*;
    let n = matrix.rows * matrix.cols;
    assert!(n.is_power_of_two(), "matrix size must be power of 2");
    if n >= 65536 {
        matrix
            .data
            .par_iter()
            .map(|&v| SecureField::from(v))
            .collect()
    } else {
        matrix.data.iter().map(|&v| SecureField::from(v)).collect()
    }
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
        evals
            .par_chunks_mut(matrix.rows)
            .enumerate()
            .for_each(|(j, col_chunk)| {
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

/// Convert an M31Matrix into a column-major MLE with implicit power-of-two padding.
///
/// Equivalent to:
/// `matrix_to_mle_col_major(&pad_matrix_pow2(matrix))`
/// but avoids allocating/copying an intermediate padded matrix.
fn matrix_to_mle_col_major_padded(matrix: &M31Matrix) -> Vec<SecureField> {
    let padded_rows = matrix.rows.next_power_of_two();
    let padded_cols = matrix.cols.next_power_of_two();
    let n = padded_rows * padded_cols;
    assert!(n.is_power_of_two(), "matrix size must be power of 2");

    let mut evals = vec![SecureField::zero(); n];

    if n >= 65536 {
        use rayon::prelude::*;
        let rows = padded_rows;
        let src_rows = matrix.rows;
        let src_cols = matrix.cols;
        evals
            .par_chunks_mut(rows)
            .enumerate()
            .for_each(|(j, col_chunk)| {
                if j < src_cols {
                    for i in 0..src_rows {
                        col_chunk[i] = SecureField::from(matrix.data[i * src_cols + j]);
                    }
                }
            });
    } else {
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                evals[j * padded_rows + i] = SecureField::from(matrix.get(i, j));
            }
        }
    }

    evals
}

/// Convert an M31Matrix into an MLE encoded in column-major QM31 AoS u32 words.
///
/// Layout per point: `[a,b,c,d]` where `SecureField::from(M31(x)) == (x,0,0,0)`.
/// So only the first limb is populated from matrix data, others are zero.
///
/// Output length is `rows * cols * 4` u32 words.
#[cfg(feature = "cuda-runtime")]
fn matrix_to_mle_col_major_u32(matrix: &M31Matrix) -> Vec<u32> {
    let n = matrix.rows * matrix.cols;
    assert!(n.is_power_of_two(), "matrix size must be power of 2");

    let mut evals = vec![0u32; n * 4];
    if n >= 65536 {
        use rayon::prelude::*;
        let rows = matrix.rows;
        let cols = matrix.cols;
        evals
            .par_chunks_mut(rows * 4)
            .enumerate()
            .for_each(|(j, col_chunk)| {
                for i in 0..rows {
                    col_chunk[i * 4] = matrix.data[i * cols + j].0;
                }
            });
    } else {
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                let dst = (j * matrix.rows + i) * 4;
                evals[dst] = matrix.get(i, j).0;
            }
        }
    }
    evals
}

/// Convert an M31Matrix into a column-major QM31 AoS u32 MLE with implicit power-of-two padding.
///
/// Equivalent to:
/// `matrix_to_mle_col_major_u32(&pad_matrix_pow2(matrix))`
/// but avoids allocating/copying an intermediate padded matrix.
#[cfg(feature = "cuda-runtime")]
fn matrix_to_mle_col_major_u32_padded(matrix: &M31Matrix) -> Vec<u32> {
    let padded_rows = matrix.rows.next_power_of_two();
    let padded_cols = matrix.cols.next_power_of_two();
    let n = padded_rows * padded_cols;
    assert!(n.is_power_of_two(), "matrix size must be power of 2");

    let mut evals = vec![0u32; n * 4];
    if n >= 65536 {
        use rayon::prelude::*;
        let rows = padded_rows;
        let src_rows = matrix.rows;
        let src_cols = matrix.cols;
        evals
            .par_chunks_mut(rows * 4)
            .enumerate()
            .for_each(|(j, col_chunk)| {
                if j < src_cols {
                    for i in 0..src_rows {
                        col_chunk[i * 4] = matrix.data[i * src_cols + j].0;
                    }
                }
            });
    } else {
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                let dst = (j * padded_rows + i) * 4;
                evals[dst] = matrix.get(i, j).0;
            }
        }
    }
    evals
}

/// Compute all Lagrange basis evaluations L_j(challenges) for j=0..2^v - 1.
///
/// Uses the tensor product structure:
///   L_j(c_0..c_{v-1}) = Π_l ((1-c_l)(1-j_l) + c_l·j_l)
///
/// Built bottom-up: each challenge doubles the weight table.
/// O(n log n) work, O(n) memory where n = 2^v.
fn compute_lagrange_basis(challenges: &[SecureField]) -> Vec<SecureField> {
    let mut weights = vec![SecureField::one()];
    for &c in challenges {
        let one_minus_c = SecureField::one() - c;
        let new_len = weights.len() * 2;
        let mut new_weights = Vec::with_capacity(new_len);
        for &w in &weights {
            new_weights.push(w * one_minus_c);
            new_weights.push(w * c);
        }
        weights = new_weights;
    }
    weights
}

/// Fused column-restrict: Σ_j matrix[i][j] × L_j(challenges) for each row i.
///
/// Equivalent to `restrict_mle(matrix_to_mle_col_major(matrix), challenges)` but
/// avoids allocating the full rows×cols SecureField MLE.
///
/// Memory: O(rows + cols) instead of O(rows × cols).
/// For Qwen3-14B (k=8192, n=8192): 256 KB instead of 1 GB per matrix.
///
/// The matrix-vector multiply is parallelized across rows with rayon.
pub fn restrict_cols_fused(matrix: &M31Matrix, challenges: &[SecureField]) -> Vec<SecureField> {
    use num_traits::Zero;
    use rayon::prelude::*;

    let k = matrix.rows;
    let n = matrix.cols;

    if challenges.is_empty() {
        // n=1: no column restriction, just convert each row's single element
        return (0..k)
            .map(|i| SecureField::from(matrix.data[i * n]))
            .collect();
    }

    let lagrange = compute_lagrange_basis(challenges);
    assert_eq!(lagrange.len(), n);

    // f_b[i] = Σ_j matrix.data[i*n + j] × lagrange[j]
    // Each row is independent → parallelize across rows.
    // Inner loop reads B row sequentially (cache-friendly) and reuses lagrange (fits L2).
    if k >= 64 {
        (0..k)
            .into_par_iter()
            .map(|i| {
                let row_start = i * n;
                let mut sum = SecureField::zero();
                for j in 0..n {
                    sum += SecureField::from(matrix.data[row_start + j]) * lagrange[j];
                }
                sum
            })
            .collect()
    } else {
        (0..k)
            .map(|i| {
                let row_start = i * n;
                let mut sum = SecureField::zero();
                for j in 0..n {
                    sum += SecureField::from(matrix.data[row_start + j]) * lagrange[j];
                }
                sum
            })
            .collect()
    }
}

/// Fused row-restrict: Σ_i matrix[i][j] × L_i(challenges) for each column j.
///
/// Equivalent to `restrict_mle(matrix_to_mle(matrix), challenges)` but
/// avoids the full rows×cols SecureField allocation.
///
/// Memory: O(rows + cols) instead of O(rows × cols).
pub fn restrict_rows_fused(matrix: &M31Matrix, challenges: &[SecureField]) -> Vec<SecureField> {
    use num_traits::Zero;
    use rayon::prelude::*;

    let m = matrix.rows;
    let k = matrix.cols;

    if challenges.is_empty() {
        // m=1: no row restriction, just convert the single row
        return matrix.data[..k]
            .iter()
            .map(|&v| SecureField::from(v))
            .collect();
    }

    let lagrange = compute_lagrange_basis(challenges);
    assert_eq!(lagrange.len(), m);

    // f_a[j] = Σ_i matrix.data[i*k + j] × lagrange[i]
    if k >= 64 {
        (0..k)
            .into_par_iter()
            .map(|j| {
                let mut sum = SecureField::zero();
                for i in 0..m {
                    sum += SecureField::from(matrix.data[i * k + j]) * lagrange[i];
                }
                sum
            })
            .collect()
    } else {
        (0..k)
            .map(|j| {
                let mut sum = SecureField::zero();
                for i in 0..m {
                    sum += SecureField::from(matrix.data[i * k + j]) * lagrange[i];
                }
                sum
            })
            .collect()
    }
}

/// Fused column-restrict on an UNPADDED matrix.
///
/// Same as `restrict_cols_fused` but works on the original (non-power-of-2) matrix
/// with challenges drawn for padded dimensions. Avoids the cost of copying the full
/// matrix into a padded version.
///
/// - `challenges` has `log2(padded_cols)` elements → Lagrange basis has `padded_cols` entries
/// - Only the first `matrix.cols` Lagrange weights are used (rest multiply zero-padding)
/// - Output has `padded_rows` elements: first `matrix.rows` from data, rest zero
pub fn restrict_cols_unpadded(
    matrix: &M31Matrix,
    challenges: &[SecureField],
    padded_rows: usize,
) -> Vec<SecureField> {
    use num_traits::Zero;
    use rayon::prelude::*;

    let k_orig = matrix.rows;
    let n_orig = matrix.cols;

    let lagrange = compute_lagrange_basis(challenges);
    // lagrange.len() = 2^challenges.len() = padded_cols >= n_orig

    // f_b[i] = Σ_{j=0}^{n_orig-1} matrix[i][j] × lagrange[j]
    // Only sum over actual columns; lagrange[j>=n_orig] multiply zeros.
    let mut result = if k_orig >= 64 {
        (0..k_orig)
            .into_par_iter()
            .map(|i| {
                let row_start = i * n_orig;
                let mut sum = SecureField::zero();
                for j in 0..n_orig {
                    sum += SecureField::from(matrix.data[row_start + j]) * lagrange[j];
                }
                sum
            })
            .collect::<Vec<_>>()
    } else {
        (0..k_orig)
            .map(|i| {
                let row_start = i * n_orig;
                let mut sum = SecureField::zero();
                for j in 0..n_orig {
                    sum += SecureField::from(matrix.data[row_start + j]) * lagrange[j];
                }
                sum
            })
            .collect()
    };

    // Zero-pad rows to padded_rows (padded rows contribute zero)
    result.resize(padded_rows, SecureField::zero());
    result
}

/// Fused row-restrict on an UNPADDED matrix.
///
/// Same as `restrict_rows_fused` but works on the original (non-power-of-2) matrix.
///
/// - `challenges` has `log2(padded_rows)` elements → Lagrange basis has `padded_rows` entries
/// - Only first `matrix.rows` Lagrange weights are used (rest multiply zero-padding)
/// - Output has `padded_cols` elements: first `matrix.cols` from data, rest zero
pub fn restrict_rows_unpadded(
    matrix: &M31Matrix,
    challenges: &[SecureField],
    padded_cols: usize,
) -> Vec<SecureField> {
    use num_traits::Zero;
    use rayon::prelude::*;

    let m_orig = matrix.rows;
    let k_orig = matrix.cols;

    let lagrange = compute_lagrange_basis(challenges);
    // lagrange.len() = 2^challenges.len() = padded_rows >= m_orig

    // f_a[j] = Σ_{i=0}^{m_orig-1} matrix[i][j] × lagrange[i]
    // Only sum over actual rows; lagrange[i>=m_orig] multiply zeros.
    let mut result = if k_orig >= 64 {
        (0..k_orig)
            .into_par_iter()
            .map(|j| {
                let mut sum = SecureField::zero();
                for i in 0..m_orig {
                    sum += SecureField::from(matrix.data[i * k_orig + j]) * lagrange[i];
                }
                sum
            })
            .collect::<Vec<_>>()
    } else {
        (0..k_orig)
            .map(|j| {
                let mut sum = SecureField::zero();
                for i in 0..m_orig {
                    sum += SecureField::from(matrix.data[i * k_orig + j]) * lagrange[i];
                }
                sum
            })
            .collect()
    };

    // Zero-pad columns to padded_cols (padded columns contribute zero)
    result.resize(padded_cols, SecureField::zero());
    result
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

    M31Matrix {
        rows: m,
        cols: n,
        data: c_data,
    }
}

/// GPU-accelerated M31 matrix multiply with automatic CPU fallback.
///
/// When the `cuda-runtime` feature is enabled and a GPU is available,
/// dispatches to the CUDA GEMM kernel (`gpu_matmul_m31_full`).
/// Falls back to CPU `matmul_m31` if GPU init fails or if CUDA is
/// not compiled in.
///
/// This should be the default entry point for all forward-pass matmuls.
pub fn matmul_m31_auto(a: &M31Matrix, b: &M31Matrix) -> M31Matrix {
    #[cfg(feature = "cuda-runtime")]
    {
        match crate::gpu_sumcheck::gpu_matmul_m31_full(a, b) {
            Ok(result) => return result,
            Err(e) => {
                // Log once and fall back to CPU
                eprintln!("[matmul] GPU dispatch failed, falling back to CPU: {e}");
            }
        }
    }
    matmul_m31(a, b)
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

        MatMulOracle {
            f_a: new_a,
            f_b: new_b,
        }
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
    EvaluationMismatch {
        expected: SecureField,
        actual: SecureField,
    },
    #[error("Claimed sum mismatch: expected {expected}, got {actual}")]
    ClaimedSumMismatch {
        expected: SecureField,
        actual: SecureField,
    },
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
        return Err(MatMulError::DimensionMismatch(format!(
            "A.cols={} != B.rows={}",
            a.cols, b.rows
        )));
    }
    if c.rows != a.rows || c.cols != b.cols {
        return Err(MatMulError::DimensionMismatch(format!(
            "C({},{}) != expected ({},{})",
            c.rows, c.cols, a.rows, b.cols
        )));
    }

    // Auto-pad to power-of-2 dimensions if needed (sumcheck requires boolean hypercube)
    let a = &pad_matrix_pow2(a);
    let b = &pad_matrix_pow2(b);
    let c = &pad_matrix_pow2(c);

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    // After padding, dimensions are guaranteed to be powers of 2
    debug_assert!(m.is_power_of_two());
    debug_assert!(k.is_power_of_two());
    debug_assert!(n.is_power_of_two());

    let log_m = m.ilog2() as usize;
    let _log_k = k.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    // Build MLEs
    let mle_a = matrix_to_mle(a); // row-major: (row_bits, col_bits)
    let mle_b_t = matrix_to_mle_col_major(b); // col-major: (col_bits, row_bits)
    let mle_c = matrix_to_mle(c); // row-major: (row_bits, col_bits)

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
        return Err(MatMulError::DimensionMismatch(format!(
            "A.cols={} != B.rows={}",
            a.cols, b.rows
        )));
    }
    if c.rows != a.rows || c.cols != b.cols {
        return Err(MatMulError::DimensionMismatch(format!(
            "C({},{}) != expected ({},{})",
            c.rows, c.cols, a.rows, b.cols
        )));
    }

    // Auto-pad to power-of-2 dimensions if needed (sumcheck requires boolean hypercube)
    let a = &pad_matrix_pow2(a);
    let b = &pad_matrix_pow2(b);
    let c = &pad_matrix_pow2(c);

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    // After padding, dimensions are guaranteed to be powers of 2
    debug_assert!(m.is_power_of_two());
    debug_assert!(k.is_power_of_two());
    debug_assert!(n.is_power_of_two());

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

    // Commit to restricted MLEs (root-only: tree not needed for sumcheck)
    let a_commitment = commit_mle_root_only(&f_a);
    let b_commitment = commit_mle_root_only(&f_b);

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

/// Prove with a pre-computed weight commitment.
///
/// When the restricted weight MLE (`cached_f_b`) and its Merkle root
/// (`cached_b_commitment`) are known from a previous inference, this
/// skips `matrix_to_mle_col_major(b)` + `restrict_mle` + `commit_mle_root_only`
/// for the B matrix — typically 30-40% of single-matmul proving time.
///
/// The caller must guarantee that `cached_f_b` and `cached_b_commitment`
/// were produced with the same `(m, k, n)` padding and Fiat-Shamir challenges.
pub fn prove_matmul_sumcheck_onchain_with_cached_weight(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
    cached_f_b: &[SecureField],
    cached_b_commitment: starknet_ff::FieldElement,
) -> Result<MatMulSumcheckProofOnChain, MatMulError> {
    if a.cols != b.rows {
        return Err(MatMulError::DimensionMismatch(format!(
            "A.cols={} != B.rows={}",
            a.cols, b.rows
        )));
    }
    if c.rows != a.rows || c.cols != b.cols {
        return Err(MatMulError::DimensionMismatch(format!(
            "C({},{}) != expected ({},{})",
            c.rows, c.cols, a.rows, b.cols
        )));
    }

    let a = &pad_matrix_pow2(a);
    let b_padded = &pad_matrix_pow2(b);
    let c = &pad_matrix_pow2(c);

    let m = a.rows;
    let k = a.cols;
    let n = b_padded.cols;
    let log_m = m.ilog2() as usize;
    let log_k = k.ilog2() as usize;
    let log_n = n.ilog2() as usize;

    let mle_a = matrix_to_mle(a);
    let mle_c = matrix_to_mle(c);

    let mut channel = PoseidonChannel::new();
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);

    let r_i = channel.draw_qm31s(log_m);
    let _r_j = channel.draw_qm31s(log_n);

    let mut r_ij = Vec::with_capacity(log_m + log_n);
    r_ij.extend_from_slice(&r_i);
    r_ij.extend_from_slice(&_r_j);
    let claimed_sum = evaluate_mle(&mle_c, &r_ij);
    channel.mix_felt(securefield_to_felt(claimed_sum));

    let f_a = restrict_mle(&mle_a, &r_i);
    assert_eq!(f_a.len(), k);

    let f_b = cached_f_b;
    assert_eq!(f_b.len(), k, "cached f_b length {} != k {}", f_b.len(), k);

    let a_commitment = commit_mle_root_only(&f_a);
    let b_commitment = cached_b_commitment;

    channel.mix_felt(a_commitment);
    channel.mix_felt(b_commitment);

    let num_rounds = log_k;
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut assignment = Vec::with_capacity(num_rounds);
    let mut cur_a = f_a.clone();
    let mut cur_b = f_b.to_vec();

    for _round in 0..num_rounds {
        let half = cur_a.len() / 2;
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

        let c0 = s0;
        let two = SecureField::from(M31::from(2));
        let c2 = (s2 - two * s1 + s0) * two.inverse();
        let c1 = s1 - s0 - c2;

        round_polys.push(RoundPoly { c0, c1, c2 });
        channel.mix_poly_coeffs(c0, c1, c2);
        let r_k = channel.draw_qm31();
        assignment.push(r_k);

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

    let f_b_owned = cached_f_b.to_vec();
    let a_opening = prove_mle_opening(&f_a, &assignment, &mut channel);
    let b_opening = prove_mle_opening(&f_b_owned, &assignment, &mut channel);

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
        return Err(MatMulError::SumcheckFailed(format!(
            "num_rounds {} != log_k {}",
            proof.num_rounds, log_k
        )));
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
            return Err(MatMulError::SumcheckFailed(format!(
                "round sum {round_sum:?} != expected {current_sum:?}"
            )));
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
        return Err(MatMulError::SumcheckFailed(format!(
            "a_opening.final_value {:?} != final_a_eval {:?}",
            proof.a_opening.final_value, proof.final_a_eval
        )));
    }
    if proof.b_opening.final_value != proof.final_b_eval {
        return Err(MatMulError::SumcheckFailed(format!(
            "b_opening.final_value {:?} != final_b_eval {:?}",
            proof.b_opening.final_value, proof.final_b_eval
        )));
    }

    // Verify MLE opening proof for A against its commitment.
    // Channel state must match prover's state after sumcheck rounds.
    if !verify_mle_opening(
        proof.a_commitment,
        &proof.a_opening,
        &assignment,
        &mut channel,
    ) {
        return Err(MatMulError::SumcheckFailed(
            "MLE opening verification failed for matrix A".into(),
        ));
    }

    // Verify MLE opening proof for B against its commitment.
    // Channel continues sequentially from A's verification.
    if !verify_mle_opening(
        proof.b_commitment,
        &proof.b_opening,
        &assignment,
        &mut channel,
    ) {
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

/// Public wrapper for `matrix_to_mle` (row-major MLE construction).
pub fn matrix_to_mle_pub(matrix: &M31Matrix) -> Vec<SecureField> {
    matrix_to_mle(matrix)
}

/// Public wrapper for `matrix_to_mle_col_major` (column-major MLE construction).
pub fn matrix_to_mle_col_major_pub(matrix: &M31Matrix) -> Vec<SecureField> {
    matrix_to_mle_col_major(matrix)
}

/// Public wrapper for `matrix_to_mle_col_major_padded` (implicit pow2-padded column-major MLE).
pub fn matrix_to_mle_col_major_padded_pub(matrix: &M31Matrix) -> Vec<SecureField> {
    matrix_to_mle_col_major_padded(matrix)
}

/// Public wrapper for `matrix_to_mle_col_major_u32` (QM31 AoS u32 encoding).
#[cfg(feature = "cuda-runtime")]
pub fn matrix_to_mle_col_major_u32_pub(matrix: &M31Matrix) -> Vec<u32> {
    matrix_to_mle_col_major_u32(matrix)
}

/// Public wrapper for `matrix_to_mle_col_major_u32_padded` (implicit pow2-padded QM31 AoS u32).
#[cfg(feature = "cuda-runtime")]
pub fn matrix_to_mle_col_major_u32_padded_pub(matrix: &M31Matrix) -> Vec<u32> {
    matrix_to_mle_col_major_u32_padded(matrix)
}

/// Public wrapper for `compute_lagrange_basis`.
pub fn compute_lagrange_basis_pub(challenges: &[SecureField]) -> Vec<SecureField> {
    compute_lagrange_basis(challenges)
}

/// Prove matmul sumcheck with automatic GPU dispatch.
///
/// Uses GPU when available. Falls back to CPU on GPU errors (OOM, context loss).
pub fn prove_matmul_sumcheck_auto(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<MatMulSumcheckProof, MatMulError> {
    #[cfg(feature = "cuda-runtime")]
    {
        if crate::backend::gpu_is_available() {
            match crate::gpu_sumcheck::prove_matmul_sumcheck_gpu(a, b, c) {
                Ok(proof) => return Ok(proof),
                Err(e) => {
                    tracing::warn!("GPU sumcheck failed, falling back to CPU: {e}");
                }
            }
        }
    }
    prove_matmul_sumcheck(a, b, c)
}

/// Prove on-chain matmul sumcheck with automatic GPU dispatch.
///
/// Uses GPU when available. Falls back to CPU on GPU errors (OOM, context loss).
pub fn prove_matmul_sumcheck_onchain_auto(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<MatMulSumcheckProofOnChain, MatMulError> {
    #[cfg(feature = "cuda-runtime")]
    {
        if crate::backend::gpu_is_available() {
            match crate::gpu_sumcheck::prove_matmul_sumcheck_onchain_gpu(a, b, c) {
                Ok(proof) => return Ok(proof),
                Err(e) => {
                    tracing::warn!("GPU on-chain sumcheck failed, falling back to CPU: {e}");
                }
            }
        }
    }
    prove_matmul_sumcheck_onchain(a, b, c)
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
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);
        assert_eq!(c.get(0, 0), M31::from(19));
        assert_eq!(c.get(0, 1), M31::from(22));
        assert_eq!(c.get(1, 0), M31::from(43));
        assert_eq!(c.get(1, 1), M31::from(50));
    }

    #[test]
    fn test_mle_evaluate() {
        // MLE of f(0)=1, f(1)=3 → f(x) = 1 + 2x
        let evals = vec![
            SecureField::from(M31::from(1)),
            SecureField::from(M31::from(3)),
        ];
        let point = vec![SecureField::zero()];
        assert_eq!(
            evaluate_mle(&evals, &point),
            SecureField::from(M31::from(1))
        );

        let point = vec![SecureField::one()];
        assert_eq!(
            evaluate_mle(&evals, &point),
            SecureField::from(M31::from(3))
        );
    }

    #[test]
    fn test_sumcheck_matmul_2x2() {
        // Smallest meaningful test: 2×2 × 2×2
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
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));

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
        verify_matmul_sumcheck(&proof, &a, &b, &c)
            .expect("rectangular verification should succeed");
    }

    // === On-Chain MatMul Tests ===

    #[test]
    fn test_matmul_onchain_2x2() {
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

        let c = matmul_m31(&a, &b);

        let proof =
            prove_matmul_sumcheck_onchain(&a, &b, &c).expect("on-chain 2x2 proving should succeed");
        assert_eq!(proof.m, 2);
        assert_eq!(proof.k, 2);
        assert_eq!(proof.n, 2);
        assert_eq!(proof.num_rounds, 1);
        assert_eq!(proof.round_polys.len(), 1);
        assert_ne!(proof.a_commitment, FieldElement::ZERO);
        assert_ne!(proof.b_commitment, FieldElement::ZERO);

        verify_matmul_sumcheck_onchain(&proof).expect("on-chain 2x2 verification should succeed");
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

        let proof =
            prove_matmul_sumcheck_onchain(&a, &b, &c).expect("on-chain 4x4 proving should succeed");
        assert_eq!(proof.num_rounds, 2);
        assert_eq!(proof.round_polys.len(), 2);

        verify_matmul_sumcheck_onchain(&proof).expect("on-chain 4x4 verification should succeed");
    }

    #[test]
    fn test_matmul_onchain_tampered_fails() {
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

        let c = matmul_m31(&a, &b);

        let mut proof = prove_matmul_sumcheck_onchain(&a, &b, &c).expect("proving should succeed");

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

        let proof = prove_matmul_sumcheck_onchain(&a, &b, &c).expect("proving should succeed");

        // Verify — now includes MLE opening proof checks
        verify_matmul_sumcheck_onchain(&proof)
            .expect("full verification with MLE openings should succeed");
    }

    #[test]
    fn test_matmul_onchain_tampered_a_opening_fails() {
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

        let c = matmul_m31(&a, &b);

        let mut proof = prove_matmul_sumcheck_onchain(&a, &b, &c).expect("proving should succeed");

        // Tamper with a_opening final_value (will mismatch final_a_eval)
        proof.a_opening.final_value = SecureField::from(M31::from(12345));

        let result = verify_matmul_sumcheck_onchain(&proof);
        assert!(
            result.is_err(),
            "tampered a_opening should fail verification"
        );
    }

    #[test]
    fn test_matmul_onchain_tampered_b_commitment_fails() {
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

        let c = matmul_m31(&a, &b);

        let mut proof = prove_matmul_sumcheck_onchain(&a, &b, &c).expect("proving should succeed");

        // Tamper with b_commitment — MLE opening Merkle check should fail
        proof.b_commitment = FieldElement::from(99999u64);

        let result = verify_matmul_sumcheck_onchain(&proof);
        assert!(
            result.is_err(),
            "tampered b_commitment should fail verification"
        );
    }

    #[test]
    fn test_matmul_onchain_final_eval_mismatch() {
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

        let c = matmul_m31(&a, &b);

        let mut proof = prove_matmul_sumcheck_onchain(&a, &b, &c).expect("proving should succeed");

        // Tamper with final_a_eval but keep a_opening.final_value consistent
        // This will fail at the product check (current_sum != a*b)
        let orig = proof.final_a_eval;
        proof.final_a_eval = orig + SecureField::from(M31::from(1));

        let result = verify_matmul_sumcheck_onchain(&proof);
        assert!(
            result.is_err(),
            "mismatched final_eval should fail verification"
        );
    }

    #[test]
    fn test_restrict_cols_fused_matches_original() {
        // Verify that restrict_cols_fused produces identical results to
        // matrix_to_mle_col_major + restrict_mle for various dimensions.
        use stwo::core::fields::qm31::SecureField;

        for (k, n) in [(2, 4), (4, 4), (4, 8), (8, 16)] {
            let mut b = M31Matrix::new(k, n);
            for i in 0..k {
                for j in 0..n {
                    b.set(i, j, M31::from((i * n + j + 1) as u32));
                }
            }

            let log_n = n.ilog2() as usize;
            // Generate deterministic challenges
            let challenges: Vec<SecureField> = (0..log_n)
                .map(|i| SecureField::from(M31::from(i as u32 * 7 + 3)))
                .collect();

            // Original approach: full MLE allocation + restrict
            let mle_col = matrix_to_mle_col_major(&b);
            let expected = restrict_mle(&mle_col, &challenges);

            // Fused approach: O(k + n) memory
            let actual = restrict_cols_fused(&b, &challenges);

            assert_eq!(
                expected.len(),
                actual.len(),
                "length mismatch for {}x{}",
                k,
                n
            );
            for i in 0..expected.len() {
                assert_eq!(
                    expected[i], actual[i],
                    "mismatch at index {} for {}x{}",
                    i, k, n
                );
            }
        }
    }

    #[test]
    fn test_restrict_rows_fused_matches_original() {
        // Verify that restrict_rows_fused produces identical results to
        // matrix_to_mle + restrict_mle.
        use stwo::core::fields::qm31::SecureField;

        for (m, k) in [(2, 4), (4, 4), (4, 8), (8, 16)] {
            let mut a = M31Matrix::new(m, k);
            for i in 0..m {
                for j in 0..k {
                    a.set(i, j, M31::from((i * k + j + 1) as u32));
                }
            }

            let log_m = m.ilog2() as usize;
            let challenges: Vec<SecureField> = (0..log_m)
                .map(|i| SecureField::from(M31::from(i as u32 * 11 + 5)))
                .collect();

            // Original
            let mle_row = matrix_to_mle(&a);
            let expected = restrict_mle(&mle_row, &challenges);

            // Fused
            let actual = restrict_rows_fused(&a, &challenges);

            assert_eq!(
                expected.len(),
                actual.len(),
                "length mismatch for {}x{}",
                m,
                k
            );
            for i in 0..expected.len() {
                assert_eq!(
                    expected[i], actual[i],
                    "mismatch at index {} for {}x{}",
                    i, m, k
                );
            }
        }
    }

    #[test]
    fn test_restrict_fused_empty_challenges() {
        // Empty challenges should return converted data without folding
        let mut b = M31Matrix::new(4, 1);
        for i in 0..4 {
            b.set(i, 0, M31::from(i as u32 + 10));
        }
        let result = restrict_cols_fused(&b, &[]);
        assert_eq!(result.len(), 4);
        for i in 0..4 {
            assert_eq!(result[i], SecureField::from(M31::from(i as u32 + 10)));
        }

        let mut a = M31Matrix::new(1, 4);
        for j in 0..4 {
            a.set(0, j, M31::from(j as u32 + 20));
        }
        let result = restrict_rows_fused(&a, &[]);
        assert_eq!(result.len(), 4);
        for j in 0..4 {
            assert_eq!(result[j], SecureField::from(M31::from(j as u32 + 20)));
        }
    }

    #[test]
    fn test_restrict_cols_unpadded_matches_padded() {
        // Non-POW2 matrix: 6×5, padded to 8×8
        let mut b = M31Matrix::new(6, 5);
        for i in 0..6 {
            for j in 0..5 {
                b.set(i, j, M31::from((i * 5 + j + 1) as u32));
            }
        }
        let b_padded = pad_matrix_pow2(&b);
        assert_eq!(b_padded.rows, 8);
        assert_eq!(b_padded.cols, 8);

        // 3 challenges for padded cols (log2(8)=3)
        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
            SecureField::from(M31::from(11)),
        ];

        let padded_result = restrict_cols_fused(&b_padded, &challenges);
        let unpadded_result = restrict_cols_unpadded(&b, &challenges, 8);

        assert_eq!(padded_result.len(), unpadded_result.len());
        for i in 0..padded_result.len() {
            assert_eq!(
                padded_result[i], unpadded_result[i],
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_restrict_rows_unpadded_matches_padded() {
        // Non-POW2 matrix: 5×6, padded to 8×8
        let mut a = M31Matrix::new(5, 6);
        for i in 0..5 {
            for j in 0..6 {
                a.set(i, j, M31::from((i * 6 + j + 1) as u32));
            }
        }
        let a_padded = pad_matrix_pow2(&a);
        assert_eq!(a_padded.rows, 8);
        assert_eq!(a_padded.cols, 8);

        // 3 challenges for padded rows (log2(8)=3)
        let challenges = vec![
            SecureField::from(M31::from(5)),
            SecureField::from(M31::from(2)),
            SecureField::from(M31::from(9)),
        ];

        let padded_result = restrict_rows_fused(&a_padded, &challenges);
        let unpadded_result = restrict_rows_unpadded(&a, &challenges, 8);

        assert_eq!(padded_result.len(), unpadded_result.len());
        for i in 0..padded_result.len() {
            assert_eq!(
                padded_result[i], unpadded_result[i],
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_commit_mle_root_only_matches() {
        use crate::crypto::mle_opening::{commit_mle, commit_mle_root_only};

        let evals: Vec<SecureField> = (0..64)
            .map(|i| SecureField::from(M31::from(i + 1)))
            .collect();

        let (root_full, _tree) = commit_mle(&evals);
        let root_only = commit_mle_root_only(&evals);

        assert_eq!(root_full, root_only);
    }
}
