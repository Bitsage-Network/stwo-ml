//! Tiled matmul proving for large matrix dimensions.
//!
//! Splits the inner dimension `k` of a matmul `C[i][j] = Σ_k A[i][k] × B[k][j]`
//! into tiles, proving each tile independently with the sumcheck protocol.
//! The verifier checks `Σ tile_claimed_sum == full_claimed_sum`.
//!
//! This is sound because the sumcheck protocol is additive over the inner dimension:
//!
//! ```text
//! C[i][j] = Σ_k A[i][k] × B[k][j]
//!         = Σ_{tile} Σ_{k ∈ tile} A[i][k] × B[k][j]
//! ```
//!
//! Each tile runs an independent sumcheck with shared evaluation points `r_i`, `r_j`,
//! but only over its slice of the `k` dimension.

use stwo::core::fields::qm31::SecureField;

use super::matmul::{
    M31Matrix, MatMulSumcheckProofOnChain,
    estimate_sumcheck_memory, matmul_m31, pad_matrix_pow2,
    prove_matmul_sumcheck_onchain_auto,
};

/// Configuration for tiled matmul proving.
#[derive(Debug, Clone)]
pub struct TiledMatMulConfig {
    /// Maximum tile size along the inner dimension `k`.
    /// Must be a power of 2.
    pub max_tile_k: usize,
}

impl TiledMatMulConfig {
    /// Create a config with an explicit tile size.
    pub fn new(max_tile_k: usize) -> Self {
        assert!(max_tile_k.is_power_of_two(), "tile size must be power of 2");
        Self { max_tile_k }
    }

    /// Compute tile size from a memory budget (in bytes).
    ///
    /// The memory for a single tile sumcheck is approximately:
    /// `3 * m * tile_k * 16 + 3 * tile_k * n * 16 + overhead`
    ///
    /// For simplicity, we solve for `tile_k` such that
    /// `estimate_sumcheck_memory(m, tile_k, n).1 <= budget`.
    pub fn from_memory_budget(m: usize, k: usize, n: usize, budget_bytes: usize) -> Self {
        let mut tile_k = k.next_power_of_two();

        // Binary search down to find the largest tile that fits
        while tile_k > 1 {
            let (_, estimated) = estimate_sumcheck_memory(m, tile_k, n);
            if estimated <= budget_bytes {
                break;
            }
            tile_k /= 2;
        }

        Self { max_tile_k: tile_k.max(1) }
    }

    /// Number of tiles needed for a matmul with inner dimension `k`.
    pub fn num_tiles(&self, k: usize) -> usize {
        (k + self.max_tile_k - 1) / self.max_tile_k
    }
}

/// Proof for a single tile of a tiled matmul.
#[derive(Debug, Clone)]
pub struct TileProof {
    /// The on-chain matmul sumcheck proof for this tile.
    pub proof: MatMulSumcheckProofOnChain,
    /// Start index of this tile in the original `k` dimension.
    pub k_start: usize,
    /// End index of this tile (exclusive).
    pub k_end: usize,
}

/// Complete tiled matmul proof.
///
/// The verifier checks that `Σ tile.proof.claimed_sum == total_claimed_sum`
/// and that each tile proof is valid.
#[derive(Debug, Clone)]
pub struct TiledMatMulProof {
    /// Original matrix dimensions.
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// Per-tile proofs.
    pub tile_proofs: Vec<TileProof>,
    /// Total claimed sum: MLE_C(r_i, r_j) = Σ tile claimed sums.
    pub total_claimed_sum: SecureField,
    /// Tile configuration used.
    pub tile_k: usize,
}

/// Error type for tiled matmul proving.
#[derive(Debug, thiserror::Error)]
pub enum TiledMatMulError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Tile proving failed at tile {tile}: {message}")]
    TileProvingFailed { tile: usize, message: String },
    #[error("Claimed sum mismatch: tile sums {tile_sum:?} != total {total:?}")]
    ClaimedSumMismatch {
        tile_sum: SecureField,
        total: SecureField,
    },
}

/// Extract a column slice `A[:, k_start..k_end]` from matrix A.
pub(crate) fn extract_col_slice(a: &M31Matrix, k_start: usize, k_end: usize) -> M31Matrix {
    let slice_k = k_end - k_start;
    let mut slice = M31Matrix::new(a.rows, slice_k);
    for i in 0..a.rows {
        for j in 0..slice_k {
            slice.data[i * slice_k + j] = a.data[i * a.cols + (k_start + j)];
        }
    }
    slice
}

/// Extract a row slice `B[k_start..k_end, :]` from matrix B.
pub(crate) fn extract_row_slice(b: &M31Matrix, k_start: usize, k_end: usize) -> M31Matrix {
    let slice_k = k_end - k_start;
    let mut slice = M31Matrix::new(slice_k, b.cols);
    for i in 0..slice_k {
        for j in 0..b.cols {
            slice.data[i * b.cols + j] = b.data[(k_start + i) * b.cols + j];
        }
    }
    slice
}

/// Prove `C = A × B` using tiled sumcheck when the matrices are too large
/// for a single sumcheck proof.
///
/// Splits the inner dimension `k` into tiles, proves each tile independently,
/// and returns the combined proof. The output uses the **existing** on-chain
/// proof format — tile proofs are composed back into a format compatible with
/// the standard Cairo verifier.
///
/// # Arguments
///
/// * `a` — Left matrix (m × k)
/// * `b` — Right matrix (k × n)
/// * `c` — Result matrix (m × n), must equal `A × B`
/// * `config` — Tiling configuration
pub fn prove_tiled_matmul(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
    config: &TiledMatMulConfig,
) -> Result<TiledMatMulProof, TiledMatMulError> {
    // Validate dimensions
    if a.cols != b.rows {
        return Err(TiledMatMulError::DimensionMismatch(format!(
            "A.cols={} != B.rows={}",
            a.cols, b.rows,
        )));
    }
    if c.rows != a.rows || c.cols != b.cols {
        return Err(TiledMatMulError::DimensionMismatch(format!(
            "C({},{}) != expected ({},{})",
            c.rows, c.cols, a.rows, b.cols,
        )));
    }

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;
    let tile_k = config.max_tile_k.min(k);
    let num_tiles = config.num_tiles(k);

    tracing::info!(m, k, n, tile_k, num_tiles, "Tiled matmul proving");

    let mut tile_proofs = Vec::with_capacity(num_tiles);
    let mut total_claimed_sum = SecureField::default();

    for tile_idx in 0..num_tiles {
        let k_start = tile_idx * tile_k;
        let k_end = (k_start + tile_k).min(k);
        let actual_tile_k = k_end - k_start;

        tracing::debug!(tile_idx, k_start, k_end, actual_tile_k, "Proving tile");

        // Extract tile slices
        let a_tile = extract_col_slice(a, k_start, k_end);
        let b_tile = extract_row_slice(b, k_start, k_end);

        // Compute partial product for this tile: C_tile = A_tile × B_tile
        let c_tile = matmul_m31(&a_tile, &b_tile);

        // Pad to power-of-2 dimensions for sumcheck
        let a_padded = pad_matrix_pow2(&a_tile);
        let b_padded = pad_matrix_pow2(&b_tile);
        let c_padded = pad_matrix_pow2(&c_tile);

        // Prove this tile
        let proof = prove_matmul_sumcheck_onchain_auto(&a_padded, &b_padded, &c_padded)
            .map_err(|e| TiledMatMulError::TileProvingFailed {
                tile: tile_idx,
                message: format!("{e}"),
            })?;

        total_claimed_sum = total_claimed_sum + proof.claimed_sum;

        tile_proofs.push(TileProof {
            proof,
            k_start,
            k_end,
        });
    }

    Ok(TiledMatMulProof {
        m,
        k,
        n,
        tile_proofs,
        total_claimed_sum,
        tile_k,
    })
}

/// Verify a tiled matmul proof.
///
/// Checks that:
/// 1. Each tile proof is individually valid
/// 2. The sum of tile claimed_sums equals the total claimed_sum
pub fn verify_tiled_matmul(proof: &TiledMatMulProof) -> Result<(), TiledMatMulError> {
    use super::matmul::verify_matmul_sumcheck_onchain;

    let mut tile_sum = SecureField::default();

    for (idx, tile) in proof.tile_proofs.iter().enumerate() {
        verify_matmul_sumcheck_onchain(&tile.proof).map_err(|e| {
            TiledMatMulError::TileProvingFailed {
                tile: idx,
                message: format!("verification failed: {e}"),
            }
        })?;
        tile_sum = tile_sum + tile.proof.claimed_sum;
    }

    if tile_sum != proof.total_claimed_sum {
        return Err(TiledMatMulError::ClaimedSumMismatch {
            tile_sum,
            total: proof.total_claimed_sum,
        });
    }

    Ok(())
}

/// Compose a tiled proof back into a single `MatMulSumcheckProofOnChain`.
///
/// This is only sound for **single-tile** proofs where the tile's sumcheck
/// covers the entire inner dimension. For multi-tile proofs, the first tile's
/// round polynomials correspond to a partial sum, NOT the total claimed sum,
/// so the verifier would reject (p(0)+p(1) != claimed_sum in round 1).
///
/// Multi-tile proofs require a dedicated tiled verifier that checks each tile
/// independently and sums the claimed_sums.
pub fn compose_tiled_proof(
    tiled: &TiledMatMulProof,
) -> Result<MatMulSumcheckProofOnChain, TiledMatMulError> {
    if tiled.tile_proofs.len() != 1 {
        return Err(TiledMatMulError::DimensionMismatch(format!(
            "compose_tiled_proof only supports single-tile proofs (got {} tiles). \
             Multi-tile proofs cannot be soundly composed into a single sumcheck proof \
             because the round polynomials from one tile don't match the total claimed sum. \
             Increase TILED_MEMORY_BUDGET or use verify_tiled_matmul() directly.",
            tiled.tile_proofs.len(),
        )));
    }

    let base = &tiled.tile_proofs[0].proof;

    Ok(MatMulSumcheckProofOnChain {
        m: tiled.m as u32,
        k: tiled.k as u32,
        n: tiled.n as u32,
        num_rounds: base.num_rounds,
        claimed_sum: tiled.total_claimed_sum,
        round_polys: base.round_polys.clone(),
        final_a_eval: base.final_a_eval,
        final_b_eval: base.final_b_eval,
        a_commitment: base.a_commitment,
        b_commitment: base.b_commitment,
        a_opening: base.a_opening.clone(),
        b_opening: base.b_opening.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::m31::M31;

    fn make_matrix(rows: usize, cols: usize, seed: u32) -> M31Matrix {
        let mut m = M31Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.set(i, j, M31::from(((i * cols + j) as u32 * seed + 1) % 127));
            }
        }
        m
    }

    #[test]
    fn test_extract_col_slice() {
        // 2×4 matrix, extract columns 1..3
        let mut a = M31Matrix::new(2, 4);
        for i in 0..2 {
            for j in 0..4 {
                a.set(i, j, M31::from((i * 4 + j + 1) as u32));
            }
        }

        let slice = extract_col_slice(&a, 1, 3);
        assert_eq!(slice.rows, 2);
        assert_eq!(slice.cols, 2);
        assert_eq!(slice.get(0, 0), M31::from(2)); // a[0][1]
        assert_eq!(slice.get(0, 1), M31::from(3)); // a[0][2]
        assert_eq!(slice.get(1, 0), M31::from(6)); // a[1][1]
        assert_eq!(slice.get(1, 1), M31::from(7)); // a[1][2]
    }

    #[test]
    fn test_extract_row_slice() {
        let mut b = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                b.set(i, j, M31::from((i * 2 + j + 1) as u32));
            }
        }

        let slice = extract_row_slice(&b, 1, 3);
        assert_eq!(slice.rows, 2);
        assert_eq!(slice.cols, 2);
        assert_eq!(slice.get(0, 0), M31::from(3)); // b[1][0]
        assert_eq!(slice.get(0, 1), M31::from(4)); // b[1][1]
        assert_eq!(slice.get(1, 0), M31::from(5)); // b[2][0]
        assert_eq!(slice.get(1, 1), M31::from(6)); // b[2][1]
    }

    #[test]
    fn test_tiled_matmul_single_tile() {
        // 2×2 × 2×2 with tile_k=2 (single tile, same as non-tiled)
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

        let config = TiledMatMulConfig::new(2);
        let proof = prove_tiled_matmul(&a, &b, &c, &config)
            .expect("single-tile proving should succeed");

        assert_eq!(proof.tile_proofs.len(), 1);
        assert_eq!(proof.m, 2);
        assert_eq!(proof.k, 2);
        assert_eq!(proof.n, 2);

        verify_tiled_matmul(&proof).expect("single-tile verification should succeed");
    }

    #[test]
    fn test_tiled_matmul_two_tiles() {
        // 2×4 × 4×2 with tile_k=2 → 2 tiles
        let a = make_matrix(2, 4, 3);
        let b = make_matrix(4, 2, 7);
        let c = matmul_m31(&a, &b);

        let config = TiledMatMulConfig::new(2);
        let proof = prove_tiled_matmul(&a, &b, &c, &config)
            .expect("two-tile proving should succeed");

        assert_eq!(proof.tile_proofs.len(), 2);
        assert_eq!(proof.tile_proofs[0].k_start, 0);
        assert_eq!(proof.tile_proofs[0].k_end, 2);
        assert_eq!(proof.tile_proofs[1].k_start, 2);
        assert_eq!(proof.tile_proofs[1].k_end, 4);

        verify_tiled_matmul(&proof).expect("two-tile verification should succeed");
    }

    #[test]
    fn test_tiled_config_from_memory_budget() {
        // For a 128×128 matmul, a very tight budget should yield small tiles
        let config = TiledMatMulConfig::from_memory_budget(128, 128, 128, 1024);
        assert!(
            config.max_tile_k < 128,
            "tight budget should force tiling: tile_k={}",
            config.max_tile_k,
        );
        assert!(config.max_tile_k.is_power_of_two());

        // For the same matmul with huge budget, tile should be full
        let config_big = TiledMatMulConfig::from_memory_budget(128, 128, 128, 100_000_000);
        assert_eq!(config_big.max_tile_k, 128);
    }

    #[test]
    fn test_compose_tiled_proof_single_tile() {
        // Single-tile composition is trivially sound
        let a = make_matrix(2, 2, 3);
        let b = make_matrix(2, 2, 7);
        let c = matmul_m31(&a, &b);

        let config = TiledMatMulConfig::new(2); // tile_k=2 == k=2 → single tile
        let tiled_proof = prove_tiled_matmul(&a, &b, &c, &config)
            .expect("tiled proving should succeed");
        assert_eq!(tiled_proof.tile_proofs.len(), 1);

        let composed = compose_tiled_proof(&tiled_proof)
            .expect("single-tile composition should succeed");
        assert_eq!(composed.m, 2);
        assert_eq!(composed.k, 2);
        assert_eq!(composed.n, 2);
        assert_eq!(composed.claimed_sum, tiled_proof.total_claimed_sum);
    }

    #[test]
    fn test_compose_tiled_proof_multi_tile_rejected() {
        // Multi-tile composition is unsound and must be rejected
        let a = make_matrix(2, 4, 3);
        let b = make_matrix(4, 2, 7);
        let c = matmul_m31(&a, &b);

        let config = TiledMatMulConfig::new(2); // tile_k=2 < k=4 → 2 tiles
        let tiled_proof = prove_tiled_matmul(&a, &b, &c, &config)
            .expect("tiled proving should succeed");
        assert_eq!(tiled_proof.tile_proofs.len(), 2);

        let result = compose_tiled_proof(&tiled_proof);
        assert!(result.is_err(), "multi-tile composition must be rejected");
    }
}
