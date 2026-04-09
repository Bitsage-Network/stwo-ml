//! GPU-accelerated proof-of-work grinding.
//!
//! This module implements [`GrindOps`] for [`GpuBackend`].
//!
//! Proof-of-work grinding is embarrassingly parallel and could benefit from GPU,
//! but the current implementation delegates to SIMD as the speedup is modest
//! compared to FFT.

use crate::core::channel::Blake2sChannelGeneric;
use crate::core::proof_of_work::GrindOps;
use crate::prover::backend::simd::SimdBackend;

use super::GpuBackend;

impl<const IS_M31_OUTPUT: bool> GrindOps<Blake2sChannelGeneric<IS_M31_OUTPUT>> for GpuBackend {
    fn grind(channel: &Blake2sChannelGeneric<IS_M31_OUTPUT>, pow_bits: u32) -> u64 {
        // Grinding is embarrassingly parallel but the current SIMD implementation
        // is already quite fast. GPU would help for high pow_bits values.
        // For now, delegate to SIMD.
        SimdBackend::grind(channel, pow_bits)
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod poseidon {
    use crate::core::channel::Poseidon252Channel;
    use crate::core::proof_of_work::GrindOps;
    use crate::prover::backend::simd::SimdBackend;
    use super::GpuBackend;
    
    impl GrindOps<Poseidon252Channel> for GpuBackend {
        fn grind(channel: &Poseidon252Channel, pow_bits: u32) -> u64 {
            SimdBackend::grind(channel, pow_bits)
        }
    }
}

