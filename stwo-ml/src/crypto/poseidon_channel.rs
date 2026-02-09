//! Poseidon Fiat-Shamir channel matching Cairo's `PoseidonChannel`.
//!
//! Must produce identical transcripts to the Cairo verifier for
//! Fiat-Shamir consistency. Operations:
//! - `mix_u64(value)`: absorb a u64 → `hades([digest, value, 2])[0]`
//! - `mix_felt(value)`: absorb a felt252 → `hades([digest, value, 2])[0]`
//! - `draw_felt252()`: squeeze a felt252 → `hades([digest, n_draws, 3])[0]`
//! - `draw_qm31()`: extract QM31 from a felt252 draw
//! - `mix_poly_coeffs(c0, c1, c2)`: absorb a degree-2 round polynomial

use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::qm31::{SecureField, QM31};

use crate::crypto::hades::hades_permutation;

/// Pack M31 values into a single felt252.
///
/// Algorithm: `acc = 1; for each m31: acc = acc * 2^31 + m31`
/// The leading 1 acts as a sentinel to preserve leading zeros.
pub fn pack_m31s(values: &[M31]) -> FieldElement {
    let shift = FieldElement::from(1u64 << 31);
    let mut acc = FieldElement::ONE;
    for &m in values {
        acc = acc * shift + FieldElement::from(m.0 as u64);
    }
    acc
}

/// Extract M31 values from a felt252 (reverse of packing).
///
/// Extracts `count` M31 values from the packed representation.
pub fn unpack_m31s(felt: FieldElement, count: usize) -> Vec<M31> {
    let modulus = FieldElement::from(1u64 << 31);
    let p_m31 = (1u64 << 31) - 1;

    let mut result = vec![M31::from(0); count];
    let mut remaining = felt;

    for i in (0..count).rev() {
        // Extract lowest 31 bits: remaining mod 2^31
        let bytes = remaining.to_bytes_be();
        let low = u64::from_be_bytes([
            bytes[24], bytes[25], bytes[26], bytes[27],
            bytes[28], bytes[29], bytes[30], bytes[31],
        ]);
        let m31_val = (low % (1u64 << 31)) as u32;
        result[i] = M31::from(m31_val % p_m31 as u32);

        // Shift right: remaining = (remaining - low_part) / 2^31
        remaining -= FieldElement::from(m31_val as u64);
        remaining = remaining.floor_div(modulus);
    }

    result
}

/// Convert a SecureField (QM31) to a felt252 by packing its 4 M31 components.
pub fn securefield_to_felt(sf: SecureField) -> FieldElement {
    pack_m31s(&[sf.0 .0, sf.0 .1, sf.1 .0, sf.1 .1])
}

/// Convert a felt252 back to a SecureField by unpacking 4 M31 components.
pub fn felt_to_securefield(fe: FieldElement) -> SecureField {
    let m31s = unpack_m31s(fe, 4);
    QM31(CM31(m31s[0], m31s[1]), CM31(m31s[2], m31s[3]))
}

/// Poseidon Fiat-Shamir channel matching Cairo's implementation.
///
/// All operations use `hades_permutation` with specific capacity values:
/// - Mix operations: capacity = 2 (same as `poseidon_hash`)
/// - Draw operations: capacity = 3 (distinct from hash)
#[derive(Debug, Clone)]
pub struct PoseidonChannel {
    digest: FieldElement,
    n_draws: u32,
}

impl PoseidonChannel {
    /// Create a new channel with zero initial state.
    pub fn new() -> Self {
        Self {
            digest: FieldElement::ZERO,
            n_draws: 0,
        }
    }

    /// Mix a u64 value into the channel.
    ///
    /// Cairo: `state = [digest, felt(value), 2]; hades(&state); digest = state[0]; n_draws = 0;`
    pub fn mix_u64(&mut self, value: u64) {
        self.mix_felt(FieldElement::from(value));
    }

    /// Mix a felt252 value into the channel.
    pub fn mix_felt(&mut self, value: FieldElement) {
        let mut state = [self.digest, value, FieldElement::TWO];
        hades_permutation(&mut state);
        self.digest = state[0];
        self.n_draws = 0;
    }

    /// Draw a raw felt252 from the channel.
    ///
    /// Cairo: `state = [digest, n_draws, 3]; hades(&state); n_draws += 1; return state[0];`
    pub fn draw_felt252(&mut self) -> FieldElement {
        let mut state = [
            self.digest,
            FieldElement::from(self.n_draws as u64),
            FieldElement::THREE,
        ];
        hades_permutation(&mut state);
        self.n_draws += 1;
        state[0]
    }

    /// Draw a QM31 value from the channel.
    ///
    /// Draws one felt252 and extracts 4 M31 components (each reduced mod P_M31).
    pub fn draw_qm31(&mut self) -> SecureField {
        let felt = self.draw_felt252();
        // Extract 4 M31 values from the felt252
        // Use byte extraction for each component
        let bytes = felt.to_bytes_be();

        // Extract 4 values from different byte positions within the felt252
        // Each M31 needs 31 bits. We extract from non-overlapping 32-bit windows.
        let p = (1u32 << 31) - 1;
        let v0 = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) % p;
        let v1 = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) % p;
        let v2 = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]) % p;
        let v3 = u32::from_be_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]) % p;

        QM31(CM31(M31::from(v0), M31::from(v1)), CM31(M31::from(v2), M31::from(v3)))
    }

    /// Draw multiple QM31 values.
    pub fn draw_qm31s(&mut self, count: usize) -> Vec<SecureField> {
        (0..count).map(|_| self.draw_qm31()).collect()
    }

    /// Mix a degree-2 polynomial (c0, c1, c2) into the channel.
    ///
    /// Packs the 12 M31 components (3 QM31s × 4 M31s each) into two felt252s:
    /// - felt1: pack first 8 M31s
    /// - felt2: pack last 4 M31s
    /// - Then hashes: digest = poseidon_hash_many([digest, felt1, felt2])
    pub fn mix_poly_coeffs(&mut self, c0: SecureField, c1: SecureField, c2: SecureField) {
        let m31s: Vec<M31> = vec![
            c0.0 .0, c0.0 .1, c0.1 .0, c0.1 .1,
            c1.0 .0, c1.0 .1, c1.1 .0, c1.1 .1,
            c2.0 .0, c2.0 .1, c2.1 .0, c2.1 .1,
        ];

        let felt1 = pack_m31s(&m31s[..8]);
        let felt2 = pack_m31s(&m31s[8..]);

        let hash = starknet_crypto::poseidon_hash_many(&[self.digest, felt1, felt2]);
        self.digest = hash;
        self.n_draws = 0;
    }

    /// Get the current digest (for debugging/testing).
    pub fn digest(&self) -> FieldElement {
        self.digest
    }
}

impl Default for PoseidonChannel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_mix_draw_deterministic() {
        let mut ch1 = PoseidonChannel::new();
        let mut ch2 = PoseidonChannel::new();

        ch1.mix_u64(42);
        ch2.mix_u64(42);

        let d1 = ch1.draw_felt252();
        let d2 = ch2.draw_felt252();
        assert_eq!(d1, d2, "same operations should produce same draws");

        let d3 = ch1.draw_felt252();
        let d4 = ch2.draw_felt252();
        assert_eq!(d3, d4);
        assert_ne!(d1, d3, "consecutive draws should differ");
    }

    #[test]
    fn test_channel_draw_qm31_components_valid() {
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(123);

        let qm31 = ch.draw_qm31();
        let p = (1u32 << 31) - 1;

        // All M31 components must be < P (2^31 - 1)
        assert!(qm31.0 .0 .0 < p);
        assert!(qm31.0 .1 .0 < p);
        assert!(qm31.1 .0 .0 < p);
        assert!(qm31.1 .1 .0 < p);
    }

    #[test]
    fn test_channel_mix_poly_coeffs() {
        let mut ch = PoseidonChannel::new();
        let initial_digest = ch.digest();

        let c0 = QM31(CM31(M31::from(1), M31::from(2)), CM31(M31::from(3), M31::from(4)));
        let c1 = QM31(CM31(M31::from(5), M31::from(6)), CM31(M31::from(7), M31::from(8)));
        let c2 = QM31(CM31(M31::from(9), M31::from(10)), CM31(M31::from(11), M31::from(12)));

        ch.mix_poly_coeffs(c0, c1, c2);

        assert_ne!(ch.digest(), initial_digest, "mixing poly coeffs should change digest");

        // Determinism: same coefficients should produce same digest
        let mut ch2 = PoseidonChannel::new();
        ch2.mix_poly_coeffs(c0, c1, c2);
        assert_eq!(ch.digest(), ch2.digest());
    }

    #[test]
    fn test_channel_mix_felt_resets_draws() {
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(1);

        // Draw something to advance n_draws
        let _d1 = ch.draw_felt252();
        let _d2 = ch.draw_felt252();

        // After mix, n_draws resets to 0
        ch.mix_felt(FieldElement::from(99u64));

        // First draw after mix should be deterministic based on new digest
        let d3 = ch.draw_felt252();
        assert_ne!(d3, FieldElement::ZERO);
    }
}
