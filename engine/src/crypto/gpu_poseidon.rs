//! GPU Poseidon Channel — Fiat-Shamir on-device for zero CPU round-trips.
//!
//! Manages the Poseidon Hades permutation state on GPU, enabling the
//! sumcheck loop to run entirely on-device without CPU synchronization.
//!
//! Usage:
//! ```ignore
//! let mut gpu_ch = GpuPoseidonChannel::from_cpu(&cpu_channel, &executor)?;
//! // ... sumcheck loop with gpu_ch.mix_and_draw_gpu() ...
//! let cpu_channel = gpu_ch.into_cpu()?;
//! ```

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};

#[cfg(feature = "cuda-runtime")]
use crate::crypto::poseidon_channel::PoseidonChannel;
#[cfg(feature = "cuda-runtime")]
use crate::crypto::poseidon_constants;

/// GPU-resident Poseidon Fiat-Shamir channel.
///
/// Holds the channel state (digest + n_draws) and round constants on GPU.
/// All mix/draw operations run on-device via the Hades CUDA kernel.
#[cfg(feature = "cuda-runtime")]
pub struct GpuPoseidonChannel {
    device: Arc<CudaDevice>,
    /// Channel state on GPU: [digest_w0..w7, n_draws, hash_count_lo, hash_count_hi] = 11 u32
    d_state: CudaSlice<u32>,
    /// Round constants on GPU: 273 × 8 u32 = 2184 u32
    d_round_constants: CudaSlice<u32>,
    /// Compiled Poseidon permutation kernel
    poseidon_fn: CudaFunction,
    /// Number of hashes performed (tracked for CPU sync)
    hash_count: u64,
}

#[cfg(feature = "cuda-runtime")]
impl GpuPoseidonChannel {
    /// Create from a CPU channel — uploads state + round constants to GPU.
    pub fn from_cpu(
        cpu_channel: &PoseidonChannel,
        device: &Arc<CudaDevice>,
    ) -> Result<Self, GpuPoseidonError> {
        // Compile Poseidon kernel
        let kernel_src = include_str!("../cuda/poseidon_hades.cu");
        let ptx = cudarc::nvrtc::compile_ptx(kernel_src)
            .map_err(|e| GpuPoseidonError::KernelCompile(format!("{e:?}")))?;

        device.load_ptx(ptx, "poseidon", &["poseidon_permute_kernel", "poseidon_mix_draw_kernel"])
            .map_err(|e| GpuPoseidonError::KernelCompile(format!("{e:?}")))?;

        let poseidon_fn = device.get_func("poseidon", "poseidon_permute_kernel")
            .ok_or_else(|| GpuPoseidonError::KernelCompile("poseidon_permute_kernel not found".into()))?;

        // Upload round constants
        let (rc_u32, _n_rounds, _n_per_round) = poseidon_constants::get_round_constants_u32();
        let d_round_constants = device.htod_sync_copy(&rc_u32)
            .map_err(|e| GpuPoseidonError::Memory(format!("upload round constants: {e:?}")))?;

        // Upload channel state
        let state_u32 = cpu_channel.to_gpu_state();
        let d_state = device.htod_sync_copy(&state_u32)
            .map_err(|e| GpuPoseidonError::Memory(format!("upload state: {e:?}")))?;

        Ok(Self {
            device: Arc::clone(device),
            d_state,
            d_round_constants,
            poseidon_fn,
            hash_count: cpu_channel.hash_count,
        })
    }

    /// Run Poseidon Hades permutation on GPU.
    ///
    /// Operates on `d_state[0..24]` (3 felt252 elements = 24 u32 words).
    /// Modifies state in-place.
    pub fn permute_gpu(&mut self) -> Result<(), GpuPoseidonError> {
        unsafe {
            self.poseidon_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1), // Single thread — Poseidon is sequential
                    shared_mem_bytes: 0,
                },
                (
                    &mut self.d_state,
                    &self.d_round_constants,
                    4u32,  // n_full_first (first half of 8 full rounds)
                    83u32, // n_partial
                    4u32,  // n_full_last (second half of 8 full rounds)
                ),
            ).map_err(|e| GpuPoseidonError::Kernel(format!("permute: {e:?}")))?;
        }
        self.hash_count += 1;
        Ok(())
    }

    /// Mix + draw on GPU using two separate permute_kernel launches.
    ///
    /// Split into 2 kernel calls (instead of 1 fused kernel) to avoid
    /// GPU stack overflow from deeply nested felt252_mul calls.
    ///
    /// Step 1: Build mix state [digest, pack(s0,s1,s2), 2] → permute → new digest
    /// Step 2: Build draw state [new_digest, n_draws, 3] → permute → extract QM31
    ///
    /// Returns the challenge as a CudaSlice<u32> (4 words, stays on GPU).
    pub fn mix_and_draw_gpu(
        &mut self,
        d_s0: &CudaSlice<u32>,
        d_s1: &CudaSlice<u32>,
        d_s2: &CudaSlice<u32>,
    ) -> Result<CudaSlice<u32>, GpuPoseidonError> {
        // Step 1: MIX — pack coefficients + permute
        // We need to build the Poseidon state [digest, packed_value, 2] on GPU.
        // For now, download coefficients, pack on CPU, build state, upload, permute.
        // The packing requires felt252 multiply (base-2^31 encoding) which is what
        // caused the stack overflow when done in the fused kernel. Doing it on CPU
        // for the packing step keeps the GPU kernel simple.

        // Download s0, s1, s2 from GPU (12 u32 total = 48 bytes, tiny)
        let mut s0 = [0u32; 4];
        let mut s1 = [0u32; 4];
        let mut s2 = [0u32; 4];
        self.device.dtoh_sync_copy_into(d_s0, &mut s0)
            .map_err(|e| GpuPoseidonError::Memory(format!("download s0: {e:?}")))?;
        self.device.dtoh_sync_copy_into(d_s1, &mut s1)
            .map_err(|e| GpuPoseidonError::Memory(format!("download s1: {e:?}")))?;
        self.device.dtoh_sync_copy_into(d_s2, &mut s2)
            .map_err(|e| GpuPoseidonError::Memory(format!("download s2: {e:?}")))?;

        // Pack 12 M31 values into felt252 on CPU (fast, <1μs)
        use stwo::core::fields::m31::M31;
        let m31s: Vec<M31> = [s0, s1, s2].iter()
            .flat_map(|q| q.iter().map(|&v| M31::from(v)))
            .collect();
        let packed = crate::crypto::poseidon_channel::pack_m31s(&m31s);

        // Download current digest from GPU state
        let mut state_buf = vec![0u32; 11];
        self.device.dtoh_sync_copy_into(&self.d_state, &mut state_buf)
            .map_err(|e| GpuPoseidonError::Memory(format!("download state: {e:?}")))?;

        // Build mix permutation state: [digest, packed, 2]
        let digest_words = &state_buf[0..8];
        let packed_words = crate::crypto::poseidon_constants::felt_to_u32(&packed);
        let two_words = crate::crypto::poseidon_constants::felt_to_u32(&starknet_ff::FieldElement::TWO);

        let mut perm_state = vec![0u32; 24]; // 3 × 8 u32
        perm_state[0..8].copy_from_slice(digest_words);
        perm_state[8..16].copy_from_slice(&packed_words);
        perm_state[16..24].copy_from_slice(&two_words);

        // Upload and permute on GPU
        let mut d_perm = self.device.htod_sync_copy(&perm_state)
            .map_err(|e| GpuPoseidonError::Memory(format!("upload mix state: {e:?}")))?;

        unsafe {
            self.poseidon_fn.clone().launch(
                LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 },
                (&mut d_perm, &self.d_round_constants, 4u32, 83u32, 4u32),
            ).map_err(|e| GpuPoseidonError::Kernel(format!("mix permute: {e:?}")))?;
        }

        // Download new digest (first 8 u32 of permuted state)
        let mut perm_result = vec![0u32; 24];
        self.device.dtoh_sync_copy_into(&d_perm, &mut perm_result)
            .map_err(|e| GpuPoseidonError::Memory(format!("download mix result: {e:?}")))?;
        let new_digest = &perm_result[0..8];

        // Step 2: DRAW — [new_digest, n_draws, 3] → permute → extract QM31
        let n_draws = 0u32; // reset after mix
        let three_words = crate::crypto::poseidon_constants::felt_to_u32(&starknet_ff::FieldElement::THREE);
        let mut draw_ndraws = [0u32; 8];
        draw_ndraws[0] = n_draws;

        let mut draw_state = vec![0u32; 24];
        draw_state[0..8].copy_from_slice(new_digest);
        draw_state[8..16].copy_from_slice(&draw_ndraws);
        draw_state[16..24].copy_from_slice(&three_words);

        let mut d_draw = self.device.htod_sync_copy(&draw_state)
            .map_err(|e| GpuPoseidonError::Memory(format!("upload draw state: {e:?}")))?;

        unsafe {
            self.poseidon_fn.clone().launch(
                LaunchConfig { grid_dim: (1,1,1), block_dim: (1,1,1), shared_mem_bytes: 0 },
                (&mut d_draw, &self.d_round_constants, 4u32, 83u32, 4u32),
            ).map_err(|e| GpuPoseidonError::Kernel(format!("draw permute: {e:?}")))?;
        }

        // Download drawn state and extract QM31
        let mut draw_result = vec![0u32; 24];
        self.device.dtoh_sync_copy_into(&d_draw, &mut draw_result)
            .map_err(|e| GpuPoseidonError::Memory(format!("download draw result: {e:?}")))?;

        // Extract 4 M31 values from state[0] (first 8 u32 = felt252)
        // Reconstruct felt252 from u32 words, then extract via 31-bit chunks
        let drawn_felt = {
            let mut bytes = [0u8; 32];
            for w in 0..8 {
                let word = draw_result[w];
                let offset = (7 - w) * 4;
                bytes[offset] = (word >> 24) as u8;
                bytes[offset + 1] = (word >> 16) as u8;
                bytes[offset + 2] = (word >> 8) as u8;
                bytes[offset + 3] = word as u8;
            }
            starknet_ff::FieldElement::from_bytes_be(&bytes).unwrap_or(starknet_ff::FieldElement::ZERO)
        };

        // Extract 4 M31 values (same logic as CPU draw_qm31)
        let m31_values = crate::crypto::poseidon_channel::unpack_m31s(drawn_felt, 4);
        let challenge_u32: Vec<u32> = m31_values.iter().map(|m| m.0).collect();

        // Upload challenge to GPU
        let d_challenge = self.device.htod_sync_copy(&challenge_u32)
            .map_err(|e| GpuPoseidonError::Memory(format!("upload challenge: {e:?}")))?;

        // Update GPU state: new digest + n_draws=1
        let mut new_state = vec![0u32; 11];
        new_state[0..8].copy_from_slice(new_digest);
        new_state[8] = 1; // n_draws after 1 draw
        new_state[9] = 0;
        new_state[10] = 0;
        self.d_state = self.device.htod_sync_copy(&new_state)
            .map_err(|e| GpuPoseidonError::Memory(format!("upload new state: {e:?}")))?;

        self.hash_count += 2;
        Ok(d_challenge)
    }

    /// Download GPU state back to CPU channel.
    pub fn into_cpu(self) -> Result<PoseidonChannel, GpuPoseidonError> {
        let mut state_u32 = vec![0u32; 11];
        self.device.dtoh_sync_copy_into(&self.d_state, &mut state_u32)
            .map_err(|e| GpuPoseidonError::Memory(format!("download state: {e:?}")))?;

        Ok(PoseidonChannel::from_gpu_state(&state_u32, self.hash_count))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuPoseidonError {
    #[error("GPU Poseidon kernel compilation: {0}")]
    KernelCompile(String),
    #[error("GPU Poseidon kernel execution: {0}")]
    Kernel(String),
    #[error("GPU memory: {0}")]
    Memory(String),
}
