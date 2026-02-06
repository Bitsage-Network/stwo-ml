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

use stwo::core::fields::m31::M31;

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

// TODO: Phase 1 implementation
// - Build MLE representations of A, B, C matrices
// - Implement sumcheck prover for Σ_k MLE_A(r_i, k) × MLE_B(k, r_j)
// - Use STWO's existing sumcheck.rs + mle.rs + gkr_prover.rs
// - Add GPU-accelerated MLE evaluation via backend/gpu/gkr.rs
// - Integrate with constraint framework as FrameworkComponent

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
}
