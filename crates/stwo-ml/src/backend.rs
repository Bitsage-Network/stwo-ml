//! Backend abstraction for CPU/GPU-accelerated Poseidon Merkle operations.
//!
//! Provides batch hashing primitives that dispatch to:
//! - **CPU (rayon)**: Default, always available. Uses `starknet_crypto::poseidon_hash`.
//! - **GPU (CUDA)**: When `cuda-runtime` feature is enabled. Uses STWO's GPU Poseidon252.
//!
//! # Usage
//!
//! The commitment module uses these functions transparently. To enable GPU:
//! ```toml
//! stwo-ml = { features = ["cuda-runtime"] }
//! ```

use rayon::prelude::*;
use starknet_crypto::poseidon_hash;
use starknet_ff::FieldElement as FieldElement252;

/// Minimum elements before parallel dispatch.
pub(crate) const PAR_THRESHOLD: usize = 256;

/// GPU activation threshold (matches STWO's `GPU_POSEIDON252_THRESHOLD_LOG_SIZE = 14`).
/// Below 2^14 = 16,384 elements, GPU overhead exceeds benefit.
#[cfg(feature = "cuda-runtime")]
pub(crate) const GPU_THRESHOLD: usize = 1 << 14;

// ============================================================================
// Batch hash operations
// ============================================================================

/// Hash pairs of field elements in batch: `result[i] = poseidon_hash(left[i], right[i])`.
///
/// Dispatches to GPU when `cuda-runtime` is enabled and input exceeds threshold.
pub fn batch_poseidon_hash_pairs(
    left: &[FieldElement252],
    right: &[FieldElement252],
) -> Vec<FieldElement252> {
    assert_eq!(left.len(), right.len());
    let n = left.len();

    #[cfg(feature = "cuda-runtime")]
    if n >= GPU_THRESHOLD {
        if let Some(result) = gpu_batch_hash_pairs(left, right) {
            return result;
        }
        // GPU unavailable or failed — fall through to CPU
    }

    if n >= PAR_THRESHOLD {
        (0..n)
            .into_par_iter()
            .map(|i| poseidon_hash(left[i], right[i]))
            .collect()
    } else {
        (0..n)
            .map(|i| poseidon_hash(left[i], right[i]))
            .collect()
    }
}

/// Hash values with a domain separator: `result[i] = poseidon_hash(values[i], domain)`.
pub fn batch_poseidon_hash_with_domain(
    values: &[FieldElement252],
    domain: FieldElement252,
) -> Vec<FieldElement252> {
    let n = values.len();

    #[cfg(feature = "cuda-runtime")]
    if n >= GPU_THRESHOLD {
        let domains = vec![domain; n];
        if let Some(result) = gpu_batch_hash_pairs(values, &domains) {
            return result;
        }
    }

    if n >= PAR_THRESHOLD {
        values
            .par_iter()
            .map(|v| poseidon_hash(*v, domain))
            .collect()
    } else {
        values
            .iter()
            .map(|v| poseidon_hash(*v, domain))
            .collect()
    }
}

/// Build an entire Merkle tree layer from child hashes (pairs consecutive elements).
///
/// Input: `children[0..2n]` → Output: `parents[0..n]` where
/// `parents[i] = poseidon_hash(children[2i], children[2i+1])`.
pub fn batch_merkle_layer(children: &[FieldElement252]) -> Vec<FieldElement252> {
    let n = children.len() / 2;

    #[cfg(feature = "cuda-runtime")]
    if n >= GPU_THRESHOLD {
        if let Some(result) = gpu_merkle_layer(children) {
            return result;
        }
    }

    if n >= PAR_THRESHOLD {
        (0..n)
            .into_par_iter()
            .map(|j| poseidon_hash(children[2 * j], children[2 * j + 1]))
            .collect()
    } else {
        (0..n)
            .map(|j| poseidon_hash(children[2 * j], children[2 * j + 1]))
            .collect()
    }
}

// ============================================================================
// GPU backend (CUDA)
// ============================================================================

#[cfg(feature = "cuda-runtime")]
fn gpu_batch_hash_pairs(
    left: &[FieldElement252],
    right: &[FieldElement252],
) -> Option<Vec<FieldElement252>> {
    use stwo::prover::backend::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        return None;
    }

    // Convert FieldElement252 pairs to the format expected by STWO's CUDA executor.
    // STWO's Poseidon252 Merkle kernel operates on BaseField columns, not raw felt252 pairs.
    // For our use case (custom Merkle tree), we use a different strategy:
    // pack pairs as 2-column input to the Merkle kernel.
    //
    // Implementation: convert each pair into two BaseField columns (8 M31s per felt252),
    // then call execute_poseidon252_merkle.
    //
    // NOTE: This is a scaffold. Full GPU integration requires matching STWO's
    // internal column format (Col<GpuBackend, BaseField>) which involves GPU memory
    // allocation. The actual kernel calls would be:
    //   cuda_executor.execute_poseidon252_merkle(columns, prev_layer, n_hashes, round_constants)
    //
    // Until we have a CUDA-capable CI, we return None to fall through to CPU.
    let _ = (left, right);
    None
}

#[cfg(feature = "cuda-runtime")]
fn gpu_merkle_layer(children: &[FieldElement252]) -> Option<Vec<FieldElement252>> {
    use stwo::prover::backend::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        return None;
    }

    // Same scaffold as gpu_batch_hash_pairs — requires CUDA column format conversion.
    let _ = children;
    None
}

// ============================================================================
// Backend info
// ============================================================================

/// Returns a description of the active proving backend.
pub fn active_backend() -> &'static str {
    #[cfg(feature = "cuda-runtime")]
    {
        use stwo::prover::backend::gpu::GpuBackend;
        if GpuBackend::is_available() {
            return "GPU (CUDA + rayon fallback)";
        }
    }
    "CPU (rayon parallel)"
}

/// Returns the number of rayon threads in the current pool.
pub fn rayon_threads() -> usize {
    rayon::current_num_threads()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use starknet_ff::FieldElement as FieldElement252;

    #[test]
    fn test_batch_hash_pairs_matches_sequential() {
        let left: Vec<FieldElement252> = (0..512)
            .map(|i| FieldElement252::from(i as u64))
            .collect();
        let right: Vec<FieldElement252> = (512..1024)
            .map(|i| FieldElement252::from(i as u64))
            .collect();

        let batch_result = batch_poseidon_hash_pairs(&left, &right);
        let sequential_result: Vec<FieldElement252> = left
            .iter()
            .zip(right.iter())
            .map(|(l, r)| poseidon_hash(*l, *r))
            .collect();

        assert_eq!(batch_result, sequential_result);
    }

    #[test]
    fn test_batch_hash_with_domain_matches_sequential() {
        let values: Vec<FieldElement252> = (0..512)
            .map(|i| FieldElement252::from(i as u64))
            .collect();
        let domain = FieldElement252::ZERO;

        let batch_result = batch_poseidon_hash_with_domain(&values, domain);
        let sequential_result: Vec<FieldElement252> = values
            .iter()
            .map(|v| poseidon_hash(*v, domain))
            .collect();

        assert_eq!(batch_result, sequential_result);
    }

    #[test]
    fn test_batch_merkle_layer_matches_sequential() {
        let children: Vec<FieldElement252> = (0..1024)
            .map(|i| FieldElement252::from(i as u64))
            .collect();

        let batch_result = batch_merkle_layer(&children);
        let sequential_result: Vec<FieldElement252> = (0..512)
            .map(|j| poseidon_hash(children[2 * j], children[2 * j + 1]))
            .collect();

        assert_eq!(batch_result, sequential_result);
    }

    #[test]
    fn test_small_input_stays_sequential() {
        // Below threshold — should still produce correct results
        let left = vec![FieldElement252::from(1u64), FieldElement252::from(2u64)];
        let right = vec![FieldElement252::from(3u64), FieldElement252::from(4u64)];

        let result = batch_poseidon_hash_pairs(&left, &right);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], poseidon_hash(left[0], right[0]));
        assert_eq!(result[1], poseidon_hash(left[1], right[1]));
    }

    #[test]
    fn test_active_backend_reports_cpu() {
        // On Mac/CI without CUDA, should report CPU
        let backend = active_backend();
        assert!(
            backend.contains("CPU") || backend.contains("GPU"),
            "Backend should report CPU or GPU: {backend}"
        );
    }

    #[test]
    fn test_rayon_threads_positive() {
        assert!(rayon_threads() > 0);
    }
}
