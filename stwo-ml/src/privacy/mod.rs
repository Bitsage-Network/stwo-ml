//! Privacy SDK for the VM31 shielded pool.
//!
//! High-level wallet, transaction builder, and pool interaction layer
//! on top of the low-level circuits and cryptographic primitives.

pub mod note_store;
pub mod pool_client;
pub mod relayer;
pub mod serde_utils;
pub mod tree_sync;
pub mod tx_builder;
pub mod wallet;

use stwo::core::fields::m31::BaseField as M31;

/// M31 prime modulus P = 2^31 - 1.
const M31_P: u32 = 0x7FFFFFFF;

/// Reduce a u32 to a valid M31 element by masking to 31 bits.
///
/// For deterministic inputs (hex parsing, password bytes), this avoids the
/// modular bias of `val % P` (which gives values 0 and 1 an extra 1/2^32
/// probability). Masking to 31 bits maps P → 0, which is field-correct.
pub(crate) fn reduce_u32_to_m31(val: u32) -> M31 {
    let masked = val & M31_P; // [0, 2^31 - 1]
    if masked == M31_P {
        M31::from_u32_unchecked(0) // P ≡ 0 mod P
    } else {
        M31::from_u32_unchecked(masked)
    }
}

/// Generate 4 uniformly random M31 elements using rejection sampling.
///
/// Masks each 32-bit sample to 31 bits and rejects the single invalid value
/// P = 2^31 - 1 (probability 1/2^31 per limb, expected retries ≈ 0).
pub(crate) fn random_m31_quad() -> Result<[M31; 4], String> {
    let mut result = [M31::from_u32_unchecked(0); 4];
    for slot in result.iter_mut() {
        loop {
            let mut bytes = [0u8; 4];
            getrandom::getrandom(&mut bytes).map_err(|e| format!("getrandom failed: {e}"))?;
            let candidate = u32::from_le_bytes(bytes) & M31_P;
            if candidate < M31_P {
                *slot = M31::from_u32_unchecked(candidate);
                break;
            }
        }
    }
    Ok(result)
}
