/// Matmul sumcheck component metadata.
///
/// The actual sumcheck verification happens in `sumcheck.cairo` — this module
/// provides the claim types and helper functions for matmul proof metadata.

use stwo_verifier_core::fields::qm31::{QM31, QM31Serde};

/// Claim about a single matmul layer.
#[derive(Drop, Serde, Copy)]
pub struct MatMulClaim {
    /// Layer index in the computation graph.
    pub layer_index: u32,
    /// Matrix dimensions: (m, k, n) where C[m×n] = A[m×k] × B[k×n].
    pub m: u32,
    pub k: u32,
    pub n: u32,
}

/// Interaction claim for a matmul layer (from the sumcheck).
#[derive(Drop, Serde, Copy)]
pub struct MatMulInteractionClaim {
    /// The claimed sum: MLE_C(r_i, r_j).
    pub claimed_sum: QM31,
}

/// Compute the number of sumcheck rounds for given inner dimension k.
///
/// The sumcheck runs over the boolean hypercube {0,1}^(log2(k_padded)),
/// where k_padded is the next power of 2 >= k.
pub fn num_sumcheck_rounds(k: u32) -> u32 {
    let mut log_k: u32 = 0;
    let mut val: u32 = 1;
    while val < k {
        val = val * 2;
        log_k += 1;
    };
    log_k
}
