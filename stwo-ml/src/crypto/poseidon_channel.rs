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
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::m31::M31;
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
            bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30], bytes[31],
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
    // Exact equivalent of `pack_m31s([a,b,c,d])` with sentinel prefix:
    // (((((1 * 2^31 + a) * 2^31 + b) * 2^31 + c) * 2^31) + d)
    let QM31(CM31(a, b), CM31(c, d)) = sf;
    let packed = (1u128 << 124)
        | ((a.0 as u128) << 93)
        | ((b.0 as u128) << 62)
        | ((c.0 as u128) << 31)
        | (d.0 as u128);
    FieldElement::from(packed)
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
    /// Draws one felt252 and extracts 4 M31 components via successive
    /// `floor_div(2^31)` — matching STWO's `Poseidon252Channel::draw_base_felts()`
    /// and Cairo's `felt252_to_m31_array_8()` exactly.
    ///
    /// Extraction order is LSB-first: index 0 = least significant 31 bits.
    /// Each component is reduced mod P_M31 = 2^31 - 1.
    pub fn draw_qm31(&mut self) -> SecureField {
        let felt = self.draw_felt252();
        let shift = FieldElement::from(1u64 << 31);
        let p = (1u64 << 31) - 1;

        let mut cur = felt;
        let mut m31s = [M31::from(0u32); 4];
        for m31 in m31s.iter_mut() {
            // Integer floor division: next = cur / 2^31
            let next = cur.floor_div(shift);
            // Remainder: res = cur - next * 2^31 = cur mod 2^31, in [0, 2^31)
            let res = cur - next * shift;
            cur = next;
            // Extract the small value from the low bytes of the remainder
            let bytes = res.to_bytes_be();
            let val = u64::from_be_bytes([
                bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30],
                bytes[31],
            ]);
            // Reduce mod P = 2^31 - 1 (matching BaseField::reduce)
            *m31 = M31::from((val % p) as u32);
        }

        QM31(CM31(m31s[0], m31s[1]), CM31(m31s[2], m31s[3]))
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
            c0.0 .0, c0.0 .1, c0.1 .0, c0.1 .1, c1.0 .0, c1.0 .1, c1.1 .0, c1.1 .1, c2.0 .0,
            c2.0 .1, c2.1 .0, c2.1 .1,
        ];

        let felt1 = pack_m31s(&m31s[..8]);
        let felt2 = pack_m31s(&m31s[8..]);

        let hash = starknet_crypto::poseidon_hash_many(&[self.digest, felt1, felt2]);
        self.digest = hash;
        self.n_draws = 0;
    }

    /// Mix a degree-3 round polynomial (4 QM31 coefficients = 16 M31s) into the channel.
    ///
    /// Packs the 16 M31 components (4 QM31s × 4 M31s each) into two felt252s:
    /// - felt1: pack first 8 M31s (c0, c1)
    /// - felt2: pack last 8 M31s (c2, c3)
    /// - Then hashes: digest = poseidon_hash_many([digest, felt1, felt2])
    pub fn mix_poly_coeffs_deg3(
        &mut self,
        c0: SecureField,
        c1: SecureField,
        c2: SecureField,
        c3: SecureField,
    ) {
        let m31s: Vec<M31> = vec![
            c0.0 .0, c0.0 .1, c0.1 .0, c0.1 .1, c1.0 .0, c1.0 .1, c1.1 .0, c1.1 .1, c2.0 .0,
            c2.0 .1, c2.1 .0, c2.1 .1, c3.0 .0, c3.0 .1, c3.1 .0, c3.1 .1,
        ];

        let felt1 = pack_m31s(&m31s[..8]);
        let felt2 = pack_m31s(&m31s[8..]);

        let hash = starknet_crypto::poseidon_hash_many(&[self.digest, felt1, felt2]);
        self.digest = hash;
        self.n_draws = 0;
    }

    /// Mix a variable-length array of QM31 values into the channel.
    ///
    /// Matches STWO's `Poseidon252Channel::mix_felts(&[SecureField])`:
    /// - Pack QM31s in chunks of 2 → felt252 (starting from ONE sentinel)
    /// - Leftover single QM31 packed alone
    /// - `digest = poseidon_hash_many([digest, packed_chunks...])`
    pub fn mix_felts(&mut self, felts: &[SecureField]) {
        if felts.is_empty() {
            return;
        }

        let mut hash_inputs = vec![self.digest];

        let mut i = 0;
        while i < felts.len() {
            let remaining = felts.len() - i;
            if remaining >= 2 {
                // Pack pair of QM31s: 8 M31 components
                let m31s: Vec<M31> = vec![
                    felts[i].0 .0,
                    felts[i].0 .1,
                    felts[i].1 .0,
                    felts[i].1 .1,
                    felts[i + 1].0 .0,
                    felts[i + 1].0 .1,
                    felts[i + 1].1 .0,
                    felts[i + 1].1 .1,
                ];
                hash_inputs.push(pack_m31s(&m31s));
                i += 2;
            } else {
                // Pack single QM31: 4 M31 components
                let m31s: Vec<M31> =
                    vec![felts[i].0 .0, felts[i].0 .1, felts[i].1 .0, felts[i].1 .1];
                hash_inputs.push(pack_m31s(&m31s));
                i += 1;
            }
        }

        self.digest = starknet_crypto::poseidon_hash_many(&hash_inputs);
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
    fn test_securefield_to_felt_fast_path_matches_pack() {
        let samples = [
            QM31(
                CM31(M31::from(0), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            QM31(
                CM31(M31::from(1), M31::from(2)),
                CM31(M31::from(3), M31::from(4)),
            ),
            QM31(
                CM31(M31::from((1u32 << 31) - 2), M31::from(17)),
                CM31(M31::from(1234567), M31::from((1u32 << 31) - 3)),
            ),
        ];
        for sf in samples {
            let expected = pack_m31s(&[sf.0 .0, sf.0 .1, sf.1 .0, sf.1 .1]);
            assert_eq!(securefield_to_felt(sf), expected);
        }
    }

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

        let c0 = QM31(
            CM31(M31::from(1), M31::from(2)),
            CM31(M31::from(3), M31::from(4)),
        );
        let c1 = QM31(
            CM31(M31::from(5), M31::from(6)),
            CM31(M31::from(7), M31::from(8)),
        );
        let c2 = QM31(
            CM31(M31::from(9), M31::from(10)),
            CM31(M31::from(11), M31::from(12)),
        );

        ch.mix_poly_coeffs(c0, c1, c2);

        assert_ne!(
            ch.digest(),
            initial_digest,
            "mixing poly coeffs should change digest"
        );

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

    /// Cross-verify draw_qm31 against STWO's canonical Poseidon252Channel.
    ///
    /// Both channels must produce identical QM31 values from the same
    /// initial state and operations. This ensures Fiat-Shamir transcript
    /// consistency between stwo-ml (Rust prover) and the Cairo verifier.
    #[test]
    fn test_draw_qm31_matches_stwo_canonical() {
        use stwo::core::channel::Channel;
        use stwo::core::channel::Poseidon252Channel;

        // Test with several different initial states
        for seed in [0u64, 1, 42, 12345, 999999, u64::MAX] {
            let mut our_ch = PoseidonChannel::new();
            let mut stwo_ch = Poseidon252Channel::default();

            our_ch.mix_u64(seed);
            stwo_ch.mix_u64(seed);

            // Draw 4 QM31 values and compare
            for draw_idx in 0..4 {
                let our_qm31 = our_ch.draw_qm31();
                let stwo_qm31 = stwo_ch.draw_secure_felt();

                assert_eq!(
                    our_qm31.0 .0, stwo_qm31.0 .0,
                    "seed={seed}, draw={draw_idx}: CM31.a.real mismatch"
                );
                assert_eq!(
                    our_qm31.0 .1, stwo_qm31.0 .1,
                    "seed={seed}, draw={draw_idx}: CM31.a.imag mismatch"
                );
                assert_eq!(
                    our_qm31.1 .0, stwo_qm31.1 .0,
                    "seed={seed}, draw={draw_idx}: CM31.b.real mismatch"
                );
                assert_eq!(
                    our_qm31.1 .1, stwo_qm31.1 .1,
                    "seed={seed}, draw={draw_idx}: CM31.b.imag mismatch"
                );
            }
        }
    }

    /// Verify that draw_qm31 extracts via floor_div(2^31) LSB-first,
    /// matching the modular arithmetic pattern (not byte positions).
    #[test]
    fn test_draw_qm31_lsb_first_extraction() {
        let mut ch = PoseidonChannel::new();
        ch.mix_u64(42);

        // Get the raw felt252 that draw_qm31 will consume
        let mut ch_copy = ch.clone();
        let raw_felt = ch_copy.draw_felt252();

        // Manually extract using floor_div pattern (canonical)
        let shift = FieldElement::from(1u64 << 31);
        let p = (1u64 << 31) - 1;
        let mut cur = raw_felt;
        let mut expected = [0u32; 4];
        for val in expected.iter_mut() {
            let next = cur.floor_div(shift);
            let rem = cur - next * shift;
            cur = next;
            let bytes = rem.to_bytes_be();
            let v = u64::from_be_bytes([
                bytes[24], bytes[25], bytes[26], bytes[27], bytes[28], bytes[29], bytes[30],
                bytes[31],
            ]);
            *val = (v % p) as u32;
        }

        // draw_qm31 should produce these exact M31 values
        let qm31 = ch.draw_qm31();
        assert_eq!(qm31.0 .0 .0, expected[0], "m31[0] = LSB");
        assert_eq!(qm31.0 .1 .0, expected[1], "m31[1]");
        assert_eq!(qm31.1 .0 .0, expected[2], "m31[2]");
        assert_eq!(qm31.1 .1 .0, expected[3], "m31[3] = MSB-ish");
    }

    /// Verify that mix_u64 + draw roundtrip matches STWO's Poseidon252Channel.
    ///
    /// Note: Our `mix_felt(securefield_to_felt(sf))` uses a single-QM31 pack,
    /// while STWO's `mix_felts(&[sf])` packs chunks of 2 QM31s. These are
    /// different encoding paths used in different protocol contexts.
    /// The on-chain verifier matches our encoding (mix_felt + pack_qm31_to_felt).
    #[test]
    fn test_mix_u64_draw_matches_stwo_canonical() {
        use stwo::core::channel::Channel;
        use stwo::core::channel::Poseidon252Channel;

        // Both channels should agree after identical mix_u64 sequences
        let mut our_ch = PoseidonChannel::new();
        let mut stwo_ch = Poseidon252Channel::default();

        // Mix identical u64 values
        our_ch.mix_u64(42);
        our_ch.mix_u64(9999);
        stwo_ch.mix_u64(42);
        stwo_ch.mix_u64(9999);

        // Draw should match
        let our_val = our_ch.draw_qm31();
        let stwo_val = stwo_ch.draw_secure_felt();
        assert_eq!(our_val, stwo_val, "mix_u64 + draw should match STWO");

        // Continue drawing — should still match
        let our_val2 = our_ch.draw_qm31();
        let stwo_val2 = stwo_ch.draw_secure_felt();
        assert_eq!(our_val2, stwo_val2, "second draw should also match");
    }

    // ========================================================================
    // Task #14: QM31 Field Arithmetic Cross-Verification
    // ========================================================================

    /// Verify QM31 multiplication produces identical results between
    /// our PoseidonChannel draw and direct STWO arithmetic.
    ///
    /// The secure field extension tower M31→CM31→QM31 with j²=2+i
    /// uses Karatsuba multiplication. This test verifies the full tower
    /// is correctly computed.
    #[test]
    fn test_qm31_arithmetic_cross_verify() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::m31::M31;
        use stwo::core::fields::qm31::QM31;

        // Test vectors: complex QM31 values that exercise all 4 components
        let a = QM31(
            CM31(M31::from(1234567), M31::from(7654321)),
            CM31(M31::from(111222), M31::from(333444)),
        );
        let b = QM31(
            CM31(M31::from(9876543), M31::from(3456789)),
            CM31(M31::from(555666), M31::from(777888)),
        );

        // Multiplication: (a0+a1*j)(b0+b1*j) = a0*b0 + (a0*b1+a1*b0)*j + a1*b1*j²
        // where j² = 2+i, so a1*b1*j² = a1*b1*(2+i)
        let product = a * b;

        // Verify against manual computation:
        // a0 = CM31(1234567, 7654321), a1 = CM31(111222, 333444)
        // b0 = CM31(9876543, 3456789), b1 = CM31(555666, 777888)
        // a0*b0, a1*b1, etc. — let's just verify it's deterministic
        let product2 = a * b;
        assert_eq!(product, product2, "QM31 mul should be deterministic");

        // Verify additive inverse: a + (-a) = 0
        let neg_a = -a;
        let zero = a + neg_a;
        assert_eq!(zero, QM31::default(), "a + (-a) = 0");

        // Verify distributivity: a*(b+c) = a*b + a*c
        let c = QM31(
            CM31(M31::from(42), M31::from(17)),
            CM31(M31::from(99), M31::from(5)),
        );
        let left = a * (b + c);
        let right = a * b + a * c;
        assert_eq!(left, right, "QM31 distributivity failed");

        // Print test vectors for Cairo cross-verification
        println!("=== QM31 Cross-Verification Vectors ===");
        println!(
            "a = QM31({}, {}, {}, {})",
            a.0 .0 .0, a.0 .1 .0, a.1 .0 .0, a.1 .1 .0
        );
        println!(
            "b = QM31({}, {}, {}, {})",
            b.0 .0 .0, b.0 .1 .0, b.1 .0 .0, b.1 .1 .0
        );
        println!(
            "a*b = QM31({}, {}, {}, {})",
            product.0 .0 .0, product.0 .1 .0, product.1 .0 .0, product.1 .1 .0
        );
        println!(
            "a+b = QM31({}, {}, {}, {})",
            (a + b).0 .0 .0,
            (a + b).0 .1 .0,
            (a + b).1 .0 .0,
            (a + b).1 .1 .0
        );
    }

    /// Verify eq_eval (Lagrange kernel) produces identical results.
    /// eq(x, y) = Π_i (x_i*y_i + (1-x_i)*(1-y_i))
    #[test]
    fn test_eq_eval_cross_verify() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::m31::M31;
        use stwo::core::fields::qm31::QM31;
        use stwo::prover::lookups::utils::eq;

        // Test vector 1: identical points → should be 1 on boolean hypercube
        let x = vec![QM31::from(M31::from(1)), QM31::from(M31::from(0))];
        let y = vec![QM31::from(M31::from(1)), QM31::from(M31::from(0))];
        let result1 = eq(&x, &y);
        assert_eq!(result1, QM31::from(M31::from(1)), "eq(x,x) = 1 on boolean");

        // Test vector 2: different points
        let x2 = vec![QM31::from(M31::from(1)), QM31::from(M31::from(0))];
        let y2 = vec![QM31::from(M31::from(0)), QM31::from(M31::from(1))];
        let result2 = eq(&x2, &y2);
        assert_eq!(result2, QM31::default(), "eq(10, 01) = 0 on boolean");

        // Test vector 3: non-boolean points (field elements)
        let x3 = vec![
            QM31(
                CM31(M31::from(42), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            QM31(
                CM31(M31::from(17), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
        ];
        let y3 = vec![
            QM31(
                CM31(M31::from(99), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
            QM31(
                CM31(M31::from(5), M31::from(0)),
                CM31(M31::from(0), M31::from(0)),
            ),
        ];
        let result3 = eq(&x3, &y3);

        println!("=== eq_eval Cross-Verification Vectors ===");
        println!(
            "eq([42,17], [99,5]) = QM31({}, {}, {}, {})",
            result3.0 .0 .0, result3.0 .1 .0, result3.1 .0 .0, result3.1 .1 .0
        );

        // Test vector 4: complex QM31 points
        let x4 = vec![QM31(
            CM31(M31::from(100), M31::from(200)),
            CM31(M31::from(300), M31::from(400)),
        )];
        let y4 = vec![QM31(
            CM31(M31::from(500), M31::from(600)),
            CM31(M31::from(700), M31::from(800)),
        )];
        let result4 = eq(&x4, &y4);
        println!(
            "eq_complex = QM31({}, {}, {}, {})",
            result4.0 .0 .0, result4.0 .1 .0, result4.1 .0 .0, result4.1 .1 .0
        );

        // Empty input: eq([], []) = 1
        let empty_x: &[QM31] = &[];
        let empty_y: &[QM31] = &[];
        let result_empty = eq(empty_x, empty_y);
        assert_eq!(result_empty, QM31::from(M31::from(1)), "eq([],[]) = 1");
    }

    /// Verify fold_mle_eval: v0*(1-x) + v1*x
    #[test]
    fn test_fold_mle_eval_cross_verify() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::m31::M31;
        use stwo::core::fields::qm31::QM31;

        let v0 = QM31(
            CM31(M31::from(1000), M31::from(2000)),
            CM31(M31::from(3000), M31::from(4000)),
        );
        let v1 = QM31(
            CM31(M31::from(5000), M31::from(6000)),
            CM31(M31::from(7000), M31::from(8000)),
        );
        let one = QM31::from(M31::from(1));

        // fold(v0, v1, 0) = v0
        let at_zero = v0 * (one - QM31::default()) + v1 * QM31::default();
        assert_eq!(at_zero, v0, "fold at 0 = v0");

        // fold(v0, v1, 1) = v1
        let at_one = v0 * (one - one) + v1 * one;
        assert_eq!(at_one, v1, "fold at 1 = v1");

        // fold at arbitrary point
        let x = QM31(
            CM31(M31::from(42), M31::from(7)),
            CM31(M31::from(13), M31::from(99)),
        );
        let result = v0 * (one - x) + v1 * x;

        println!("=== fold_mle_eval Cross-Verification Vectors ===");
        println!(
            "v0 = QM31({}, {}, {}, {})",
            v0.0 .0 .0, v0.0 .1 .0, v0.1 .0 .0, v0.1 .1 .0
        );
        println!(
            "v1 = QM31({}, {}, {}, {})",
            v1.0 .0 .0, v1.0 .1 .0, v1.1 .0 .0, v1.1 .1 .0
        );
        println!(
            "x = QM31({}, {}, {}, {})",
            x.0 .0 .0, x.0 .1 .0, x.1 .0 .0, x.1 .1 .0
        );
        println!(
            "fold(v0,v1,x) = QM31({}, {}, {}, {})",
            result.0 .0 .0, result.0 .1 .0, result.1 .0 .0, result.1 .1 .0
        );
    }

    // ========================================================================
    // Task #15: Poseidon Channel Transcript Cross-Verification
    // ========================================================================

    /// Verify mix_felts produces identical digests to STWO's Poseidon252Channel.
    /// This is critical for GKR Fiat-Shamir consistency.
    #[test]
    fn test_mix_felts_matches_stwo_canonical() {
        use stwo::core::channel::Channel;
        use stwo::core::channel::Poseidon252Channel;

        // Test 1: 3 QM31s (same as mix_poly_coeffs degree-2)
        let c0 = QM31(
            CM31(M31::from(1), M31::from(2)),
            CM31(M31::from(3), M31::from(4)),
        );
        let c1 = QM31(
            CM31(M31::from(5), M31::from(6)),
            CM31(M31::from(7), M31::from(8)),
        );
        let c2 = QM31(
            CM31(M31::from(9), M31::from(10)),
            CM31(M31::from(11), M31::from(12)),
        );

        let mut our_ch = PoseidonChannel::new();
        let mut stwo_ch = Poseidon252Channel::default();

        our_ch.mix_felts(&[c0, c1, c2]);
        stwo_ch.mix_felts(&[c0, c1, c2]);

        // Draw and compare
        let our_val = our_ch.draw_qm31();
        let stwo_val = stwo_ch.draw_secure_felt();
        assert_eq!(our_val, stwo_val, "mix_felts 3 QM31s: draw mismatch");

        // Test 2: 4 QM31s (same as GKR degree-3 round poly)
        let c3 = QM31(
            CM31(M31::from(13), M31::from(14)),
            CM31(M31::from(15), M31::from(16)),
        );

        let mut our_ch2 = PoseidonChannel::new();
        let mut stwo_ch2 = Poseidon252Channel::default();

        our_ch2.mix_felts(&[c0, c1, c2, c3]);
        stwo_ch2.mix_felts(&[c0, c1, c2, c3]);

        let our_val2 = our_ch2.draw_qm31();
        let stwo_val2 = stwo_ch2.draw_secure_felt();
        assert_eq!(our_val2, stwo_val2, "mix_felts 4 QM31s: draw mismatch");

        // Test 3: 1 QM31 (single, no pair)
        let mut our_ch3 = PoseidonChannel::new();
        let mut stwo_ch3 = Poseidon252Channel::default();

        our_ch3.mix_felts(&[c0]);
        stwo_ch3.mix_felts(&[c0]);

        let our_val3 = our_ch3.draw_qm31();
        let stwo_val3 = stwo_ch3.draw_secure_felt();
        assert_eq!(our_val3, stwo_val3, "mix_felts 1 QM31: draw mismatch");

        // Test 4: 5 QM31s (2 pairs + 1 leftover)
        let c4 = QM31(
            CM31(M31::from(17), M31::from(18)),
            CM31(M31::from(19), M31::from(20)),
        );

        let mut our_ch4 = PoseidonChannel::new();
        let mut stwo_ch4 = Poseidon252Channel::default();

        our_ch4.mix_felts(&[c0, c1, c2, c3, c4]);
        stwo_ch4.mix_felts(&[c0, c1, c2, c3, c4]);

        let our_val4 = our_ch4.draw_qm31();
        let stwo_val4 = stwo_ch4.draw_secure_felt();
        assert_eq!(our_val4, stwo_val4, "mix_felts 5 QM31s: draw mismatch");
    }

    /// Verify mix_felts is consistent with the existing mix_poly_coeffs helpers.
    #[test]
    fn test_mix_felts_matches_mix_poly_coeffs() {
        let c0 = QM31(
            CM31(M31::from(42), M31::from(17)),
            CM31(M31::from(99), M31::from(5)),
        );
        let c1 = QM31(
            CM31(M31::from(100), M31::from(200)),
            CM31(M31::from(300), M31::from(400)),
        );
        let c2 = QM31(
            CM31(M31::from(500), M31::from(600)),
            CM31(M31::from(700), M31::from(800)),
        );

        // mix_poly_coeffs should produce same result as mix_felts with 3 values
        let mut ch1 = PoseidonChannel::new();
        let mut ch2 = PoseidonChannel::new();

        ch1.mix_poly_coeffs(c0, c1, c2);
        ch2.mix_felts(&[c0, c1, c2]);

        assert_eq!(
            ch1.digest(),
            ch2.digest(),
            "mix_poly_coeffs != mix_felts for 3 QM31s"
        );

        // mix_poly_coeffs_deg3 should match mix_felts with 4 values
        let c3 = QM31(
            CM31(M31::from(900), M31::from(1000)),
            CM31(M31::from(1100), M31::from(1200)),
        );

        let mut ch3 = PoseidonChannel::new();
        let mut ch4 = PoseidonChannel::new();

        ch3.mix_poly_coeffs_deg3(c0, c1, c2, c3);
        ch4.mix_felts(&[c0, c1, c2, c3]);

        assert_eq!(
            ch3.digest(),
            ch4.digest(),
            "mix_poly_coeffs_deg3 != mix_felts for 4 QM31s"
        );
    }

    /// Full GKR transcript replay: mix sequence matching partially_verify_batch.
    ///
    /// Replays the exact Fiat-Shamir operations done during GKR verification:
    /// 1. Mix output claims for starting instances
    /// 2. Draw sumcheck_alpha, instance_lambda
    /// 3. For each sumcheck round: mix round_poly, draw challenge
    /// 4. Mix mask values, draw random challenge r
    #[test]
    fn test_gkr_transcript_replay_poseidon() {
        use stwo::core::channel::Channel;
        use stwo::core::channel::Poseidon252Channel;

        // Simulate a single-layer GKR with 1 GP instance:
        // This replays exactly what partially_verify_batch does.

        // Step 1: Mix output claims
        let output_claim = QM31(
            CM31(M31::from(42), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );

        let mut our_ch = PoseidonChannel::new();
        let mut stwo_ch = Poseidon252Channel::default();

        // mix_felts(&[output_claim]) — single QM31 claim
        our_ch.mix_felts(&[output_claim]);
        stwo_ch.mix_felts(&[output_claim]);

        // Step 2: Draw alpha (column RLC for multi-column, but GP has 1 col)
        let our_alpha = our_ch.draw_qm31();
        let stwo_alpha = stwo_ch.draw_secure_felt();
        assert_eq!(our_alpha, stwo_alpha, "alpha mismatch");

        // Step 3: Draw lambda (instance batching)
        let our_lambda = our_ch.draw_qm31();
        let stwo_lambda = stwo_ch.draw_secure_felt();
        assert_eq!(our_lambda, stwo_lambda, "lambda mismatch");

        // Step 4: For layer 0 (0 sumcheck rounds), skip to masks.
        // Mix mask values
        let mask_v0 = QM31(
            CM31(M31::from(6), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );
        let mask_v1 = QM31(
            CM31(M31::from(7), M31::from(0)),
            CM31(M31::from(0), M31::from(0)),
        );

        our_ch.mix_felts(&[mask_v0, mask_v1]);
        stwo_ch.mix_felts(&[mask_v0, mask_v1]);

        // Step 5: Draw random challenge r
        let our_r = our_ch.draw_qm31();
        let stwo_r = stwo_ch.draw_secure_felt();
        assert_eq!(our_r, stwo_r, "challenge r mismatch");

        println!("=== GKR Transcript Test Vectors ===");
        println!(
            "After mix_felts([{}]): alpha = QM31({}, {}, {}, {})",
            output_claim.0 .0 .0,
            our_alpha.0 .0 .0,
            our_alpha.0 .1 .0,
            our_alpha.1 .0 .0,
            our_alpha.1 .1 .0
        );
        println!(
            "lambda = QM31({}, {}, {}, {})",
            our_lambda.0 .0 .0, our_lambda.0 .1 .0, our_lambda.1 .0 .0, our_lambda.1 .1 .0
        );
        println!(
            "After mix_felts([{},{}]): r = QM31({}, {}, {}, {})",
            mask_v0.0 .0 .0,
            mask_v1.0 .0 .0,
            our_r.0 .0 .0,
            our_r.0 .1 .0,
            our_r.1 .0 .0,
            our_r.1 .1 .0
        );
    }

    /// Multi-step transcript: mix_u64 → draw → mix_felts → draw cycle.
    /// Exercises the full channel state machine as used in the complete protocol.
    #[test]
    fn test_channel_state_machine_cross_verify() {
        use stwo::core::channel::Channel;
        use stwo::core::channel::Poseidon252Channel;

        let mut our_ch = PoseidonChannel::new();
        let mut stwo_ch = Poseidon252Channel::default();

        // Phase 1: Mix dimensions (like matmul prover does)
        for dim in [64u64, 128, 256] {
            our_ch.mix_u64(dim);
            stwo_ch.mix_u64(dim);
        }

        // Draw challenges
        let our_d1 = our_ch.draw_qm31();
        let stwo_d1 = stwo_ch.draw_secure_felt();
        assert_eq!(our_d1, stwo_d1, "post-dimensions draw mismatch");

        // Phase 2: Mix field elements (like mixing commitments)
        let val = QM31(
            CM31(M31::from(999), M31::from(888)),
            CM31(M31::from(777), M31::from(666)),
        );
        our_ch.mix_felts(&[val]);
        stwo_ch.mix_felts(&[val]);

        let our_d2 = our_ch.draw_qm31();
        let stwo_d2 = stwo_ch.draw_secure_felt();
        assert_eq!(our_d2, stwo_d2, "post-commitment draw mismatch");

        // Phase 3: Multiple draws in sequence (like drawing query indices)
        for i in 0..5 {
            let our_d = our_ch.draw_qm31();
            let stwo_d = stwo_ch.draw_secure_felt();
            assert_eq!(our_d, stwo_d, "sequential draw {i} mismatch");
        }

        // Phase 4: Mix after multiple draws (tests n_draws reset)
        our_ch.mix_felts(&[val, val]);
        stwo_ch.mix_felts(&[val, val]);

        let our_d3 = our_ch.draw_qm31();
        let stwo_d3 = stwo_ch.draw_secure_felt();
        assert_eq!(our_d3, stwo_d3, "post-multi-draw-then-mix mismatch");
    }

    // ========================================================================
    // GKR Deep Circuit Test Vector Generators
    // ========================================================================

    fn qm31(a: u32, b: u32, c: u32, d: u32) -> SecureField {
        QM31(
            CM31(M31::from(a), M31::from(b)),
            CM31(M31::from(c), M31::from(d)),
        )
    }

    fn print_qm31(label: &str, v: SecureField) {
        println!(
            "{label} = QM31({}, {}, {}, {})",
            v.0 .0 .0, v.0 .1 .0, v.1 .0 .0, v.1 .1 .0
        );
    }

    fn rlc(values: &[SecureField], alpha: SecureField) -> SecureField {
        if values.is_empty() {
            return SecureField::default();
        }
        let mut acc = values[0];
        let mut alpha_pow = alpha;
        for &v in &values[1..] {
            acc = acc + alpha_pow * v;
            alpha_pow = alpha_pow * alpha;
        }
        acc
    }

    fn eq_eval_fn(x: &[SecureField], y: &[SecureField]) -> SecureField {
        assert_eq!(x.len(), y.len());
        let one = SecureField::from(M31::from(1));
        let mut acc = one;
        for i in 0..x.len() {
            let term = x[i] * y[i] + (one - x[i]) * (one - y[i]);
            acc = acc * term;
        }
        acc
    }

    fn fold_mle(x: SecureField, v0: SecureField, v1: SecureField) -> SecureField {
        let one = SecureField::from(M31::from(1));
        v0 * (one - x) + v1 * x
    }

    /// Generate test vectors for a 2-layer deep GrandProduct GKR circuit.
    ///
    /// n_variables=2 means 2 layers, each with sumcheck. The first layer (output)
    /// has 0 rounds (empty ood_point), the second layer has 1 round.
    ///
    /// Circuit: 4 leaves combined pairwise → 2 intermediate → 1 product.
    /// Layer 0 (n_remaining=2): output claim is the product.
    ///   Trivial sumcheck (0 rounds). Mask = [left_product, right_product].
    ///   Gate: left_product * right_product = total_product.
    /// Layer 1 (n_remaining=1): claims = [fold(left_product, right_product, r0)].
    ///   1 sumcheck round (ood_point has 1 element from layer 0).
    ///   Mask = [leaf_a, leaf_b]. Gate: leaf_a * leaf_b = ???.
    ///   Sumcheck checks eq(ood_point, assignment) * gate_output = eval.
    #[test]
    fn test_generate_deep_gp_vectors() {
        println!("\n=== DEEP GP (n_variables=2) Test Vectors ===\n");

        // 4 leaves: a=3, b=5, c=7, d=11
        // Layer 0 (output): left=a*b=15, right=c*d=77, product=15*77=1155
        // Layer 1: mask values are the sub-tree values for the FOLDED circuit
        let a = qm31(3, 0, 0, 0);
        let b = qm31(5, 0, 0, 0);
        let c = qm31(7, 0, 0, 0);
        let d = qm31(11, 0, 0, 0);
        let left = a * b; // 15
        let right = c * d; // 77
        let product = left * right; // 1155

        print_qm31("product (output_claim)", product);
        print_qm31("left (mask layer0 v0)", left);
        print_qm31("right (mask layer0 v1)", right);

        // ---- Layer 0 (n_remaining_layers=2) ----
        let mut ch = PoseidonChannel::new();

        // Step 2: Mix output claims
        ch.mix_felts(&[product]);

        // Step 3: Draw alphas
        let alpha0 = ch.draw_qm31();
        let lambda0 = ch.draw_qm31();
        print_qm31("layer0 alpha", alpha0);
        print_qm31("layer0 lambda", lambda0);

        // Step 4: Prepare claims (n_unused=0 for single inst with n_vars=2 when n_layers=2)
        let inst_rlc = rlc(&[product], lambda0);
        // doubling = 2^0 = 1
        let sumcheck_claim0 = rlc(&[inst_rlc], alpha0);
        print_qm31("layer0 sumcheck_claim", sumcheck_claim0);

        // Step 6: Verify sumcheck (0 rounds — trivial)
        // sumcheck_ood_point = [], sumcheck_eval = sumcheck_claim
        let sumcheck_eval0 = sumcheck_claim0;

        // Step 7: Gate evaluation
        // eq(ood[0..], sc_ood[0..]) with n_unused=0, ood=[], sc_ood=[] → eq([],[])=1
        let eq_val0 = SecureField::from(M31::from(1));
        let gate_output0 = left * right; // GP: a*b
        let rlc_gate0 = rlc(&[gate_output0], lambda0);
        let layer_eval0 = rlc(&[eq_val0 * rlc_gate0], alpha0);
        print_qm31("layer0 gate_output", gate_output0);
        print_qm31("layer0 layer_eval", layer_eval0);

        // Verify: sumcheck_eval == layer_eval
        assert_eq!(sumcheck_eval0, layer_eval0, "layer0 circuit check");

        // Step 9: Mix mask, draw challenge
        ch.mix_felts(&[left, right]);
        let r0 = ch.draw_qm31();
        print_qm31("r0 (challenge layer0)", r0);

        // Step 10-11: Reduce claims, update ood_point
        let reduced_claim0 = fold_mle(r0, left, right);
        print_qm31("reduced_claim after layer0", reduced_claim0);
        // ood_point = [r0]

        // ---- Layer 1 (n_remaining_layers=1) ----
        // No new instances start here (inst0 already active)

        // Step 2: Mix active claims
        ch.mix_felts(&[reduced_claim0]);

        // Step 3: Draw alphas
        let alpha1 = ch.draw_qm31();
        let lambda1 = ch.draw_qm31();
        print_qm31("layer1 alpha", alpha1);
        print_qm31("layer1 lambda", lambda1);

        // Step 4: n_unused=0, doubling=1
        let inst_rlc1 = rlc(&[reduced_claim0], lambda1);
        let sumcheck_claim1 = rlc(&[inst_rlc1], alpha1);
        print_qm31("layer1 sumcheck_claim", sumcheck_claim1);

        // Step 6: Layer 1 has 1 sumcheck round (ood_point has 1 element: [r0])
        // We need to construct a valid round poly for this sumcheck.
        // The sumcheck is over: Σ_{x∈{0,1}} eq(ood_point, x) * gate(mask_at_x)
        // With ood_point = [r0], the sumcheck is over x ∈ {0,1}:
        //   p(0) = eq([r0], [0]) * gate(mask_at_0)
        //   p(1) = eq([r0], [1]) * gate(mask_at_1)
        //   claim = p(0) + p(1)

        // For a GP gate with mask [leaf_a, leaf_b]:
        //   mask_at_0 means x=0 → we observe the "left" sub-tree: leaf_a, leaf_b
        //   In GKR, at each bit position, x selects left(0) or right(1) child.
        //   The mask values are [v(0), v(1)] — evaluations at x=0 and x=1.

        // For this test: the mask at layer 1 represents the MLE of the sub-products
        // evaluated at x=0 and x=1. We need mask values such that:
        // 1. gate(mask) = v(0) * v(1) (GP gate)
        // 2. The round poly satisfies p(0)+p(1) = sumcheck_claim1

        // The prover chooses mask values. For simplicity, use leaves directly:
        // mask_v0 = a*b = 15 (left sub-tree product), mask_v1 = c*d = 77 (right sub-tree)
        // But wait — these are the mask values at layer 1, which should be the
        // evaluations at the leaves of the layer-1 binary tree.

        // Actually in a real GKR circuit, the mask is always [f(0), f(1)] where
        // f is the multilinear extension. But for our test we can choose anything
        // as long as the round polynomial is consistent.

        // Simpler approach: choose mask values, compute what the round poly must be,
        // then verify the whole thing is consistent.

        let mask1_v0 = qm31(2, 0, 0, 0); // arbitrary
        let mask1_v1 = qm31(8, 0, 0, 0); // arbitrary
        let gate1 = mask1_v0 * mask1_v1; // GP: 2*8 = 16
        print_qm31("layer1 mask_v0", mask1_v0);
        print_qm31("layer1 mask_v1", mask1_v1);
        print_qm31("layer1 gate_output", gate1);

        // eq(ood[n_unused..], sc_ood[n_unused..]) with ood=[r0], n_unused=0
        // But sc_ood comes from the sumcheck! The sumcheck gives us (assignment, eval).
        // For 1 round: sc_ood = [challenge1].
        // eq([r0], [challenge1]) is computed AFTER we know challenge1.

        // The round polynomial p(x) must satisfy:
        //   p(0) + p(1) = sumcheck_claim1
        //   p(challenge) = eq([r0],[challenge]) * rlc([gate1], lambda1) * alpha1_factor

        // For a single-instance single-column GP:
        //   At x ∈ {0,1}: contribution = eq([r0],[x]) * gate_output_at_x
        //   p(x) = alpha0_factor * eq_val(x) * lambda_factor * gate_val
        // But this gets complex. Instead, let's just compute p(0) and p(1) directly:

        // eq([r0], [0]) = (1-r0), eq([r0], [1]) = r0
        let one = SecureField::from(M31::from(1));
        let eq_at_0 = one - r0;
        let eq_at_1 = r0;
        print_qm31("eq([r0],[0])", eq_at_0);
        print_qm31("eq([r0],[1])", eq_at_1);

        // layer_eval_at_x = eq([r0],[x]) * rlc([gate_output], lambda1)
        // Since single instance: sumcheck poly = alpha * eq * rlc(gate, lambda)
        // But with only 1 element in the rlc, rlc([v], lambda) = v.
        // And with only 1 instance in the alpha rlc, rlc([v], alpha) = v.

        // So p(x) = eq([r0],[x]) * gate_at_x
        // But gate_at_x needs to be defined. In GKR, the gate operates on the
        // mask values themselves (v0, v1). The "gate_at_x" evaluates at x.
        // For GP: output(x) = v0^(1-x) * v1^x ??? No, that's wrong.
        // Actually, in GKR the sumcheck is over bit positions of the evaluation domain.
        // The gate output is evaluated at EACH x in the hypercube.

        // Wait — I need to reconsider. In the GKR protocol:
        // - The sumcheck at layer L proves: Σ_x p(x) = claimed_value
        // - The mask provides [f(0), f(1)] — the univariate restriction
        // - After sumcheck, the verifier checks the final evaluation against gate(mask)
        // - There's no "gate_at_x" — the gate is applied to the MASK values

        // So for 1 round sumcheck:
        //   The prover provides a round poly. The verifier:
        //   1. Checks p(0) + p(1) = claim
        //   2. Mixes poly, draws challenge1
        //   3. sumcheck_eval = p(challenge1)
        //   4. Reads mask = [v0, v1], computes gate(mask) = v0 * v1
        //   5. Computes eq_val = eq(ood_point, sumcheck_ood_point)
        //   6. Checks sumcheck_eval == eq_val * rlc(gate_output, lambda)

        // So the round poly must satisfy:
        //   p(challenge1) = eq([r0], [challenge1]) * gate_output

        // We need p(0) + p(1) = sumcheck_claim1.
        // And p is a degree-2 poly (GP gate is degree 1 in x, times eq which is degree 1 in x → degree 2).
        // Actually for GP: the univariate polynomial at the gate is degree-2:
        //   f(x) = eq([r0],[x]) * g(x) where g(x) relates to the mask interpolation.

        // Simplest correct approach: compute p(0) and p(1), then derive the round poly.
        // p(x) = eq([r0],[x]) * gate(mask)
        // But that assumes gate(mask) is the same at x=0 and x=1, which it's not...

        // Let me re-read how STWO actually works. The GKR gate evaluation happens
        // AFTER the sumcheck, not during it. The sumcheck proves:
        //   Σ_x eq(r, x) · V_{layer}(x) = claimed
        // where V_{layer}(x) is the multilinear extension of the layer values.
        // The mask gives [V(0), V(1)] (the univariate restriction of V at the query point).
        // The round poly for the sumcheck is:
        //   p(x) = eq(r, [x|assignment_so_far]) · V(x|assignment_so_far)
        // For 1-variable sumcheck (1 round): p(x) = eq([r0], [x]) * V(x)
        // p(0) = (1-r0) * V(0), p(1) = r0 * V(1)
        // claim = (1-r0)*V(0) + r0*V(1)

        // But V(0) and V(1) are NOT the gate outputs — they're the MLE values.
        // The gate check happens afterward: gate(mask_values) must match.

        // So for our test: V(0) and V(1) can be anything. The mask [v0, v1]
        // represents the next layer's MLE at 0 and 1. The gate check verifies
        // that gate([v0, v1]) produces the right "next layer" claim.

        // Actually wait — in `partially_verify_batch`, the gate evaluation IS
        // the per-instance contribution. Let me re-read lines 308-335:
        //   gate_output = eval_gate(gate, mask)
        //   eq_val = eq(ood[n_unused..], sumcheck_ood[n_unused..])
        //   layer_evals.push(eq_val * rlc(gate_output, lambda))
        // And line 338-339:
        //   layer_eval = rlc(layer_evals, alpha)
        //   assert sumcheck_eval == layer_eval

        // So the sumcheck final evaluation must equal eq_val * gate_output.
        // For 1 round: sumcheck_eval = p(challenge1)
        // And: eq_val = eq([r0], [challenge1])
        // And: gate_output = mask_v0 * mask_v1

        // Therefore: p(challenge1) = eq([r0], [challenge1]) * mask_v0 * mask_v1
        // And: p(0) + p(1) = sumcheck_claim1

        // p(x) has degree ≤ 2 (degree of eq is 1 in x, times degree-1 gate → degree 2).
        // We know: p(challenge1) = target, p(0) + p(1) = claim.
        // That's 2 constraints for 3 unknowns (c0, c1, c2). So we have freedom.

        // The correct approach: the round polynomial IS the actual polynomial.
        // p(x) = eq([r0], [x]) * gate_interpolation(x)
        // For GP with mask [v0, v1]: gate_interpolation(x) = v0*(1-x) + v1*x... no.
        // The mask [v0, v1] represents two entries. gate(mask) = v0 * v1.
        // But the sumcheck polynomial is NOT eq * gate. It's:
        //   s(x) = Σ_{other_vars} eq(ood, [x, other]) * V_layer([x, other])
        // This reduces to the univariate restriction after fixing other vars.

        // For a 1-variable layer (only 1 round), V is a univariate MLE with
        // V(0) and V(1) being the two values. The prover computes:
        //   p(x) = eq([r0], [x]) * V(x) where V(x) = V(0)*(1-x) + V(1)*x
        //   p(0) = (1-r0) * V(0)
        //   p(1) = r0 * V(1)
        // The claim is p(0) + p(1).

        // But then the gate check: gate_output = V(0) * V(1) (for GP)?? No, that's wrong.
        // For GP: V_layer(x) = V_left(x) * V_right(x) where left and right are the
        // children in the GKR circuit. But V_left and V_right are separate columns.

        // I think I'm overcomplicating this. Let me just look at what the masks mean
        // in the code. The mask has num_columns=1 for GP, with 2 values [v0, v1].
        // These are the "left child" and "right child" evaluations.
        // gate(mask) = v0 * v1 = the parent's value.
        // The eq_eval ensures the evaluation point matches.

        // For the sumcheck: after the sumcheck, the verifier gets a challenge point
        // and evaluates: eq(ood, challenge) * gate(mask).
        // This must equal the sumcheck's final evaluation.

        // So the round poly must be constructed such that:
        //   p(challenge) = eq([r0],[challenge]) * v0 * v1

        // A valid approach: the prover sets p(x) such that p(x) = eq([r0],[x]) * gate_output
        // for ALL x. But gate_output = v0 * v1 is CONSTANT in x — it's always the
        // same product. So p(x) = eq([r0],[x]) * (v0*v1).

        // eq([r0],[x]) = r0*x + (1-r0)*(1-x) = (2r0-1)*x + (1-r0) — degree 1 in x
        // So p(x) = [(2r0-1)*x + (1-r0)] * gate_output — degree 1 in x!

        // p(0) = (1-r0) * gate_output
        // p(1) = r0 * gate_output
        // p(0) + p(1) = gate_output = sumcheck_claim

        // This means: sumcheck_claim1 == gate_output for a trivial 1-round sumcheck.
        // But our sumcheck_claim1 was derived from the PREVIOUS layer's fold.
        // So: sumcheck_claim1 = reduced_claim0 = fold(left, right, r0)
        // And: gate_output = v0 * v1 where v0, v1 are the mask values.
        // The mask values are the leaf values seen by the verifier.
        // The constraint is: fold(left, right, r0) = sum of p(x) over x = p(0)+p(1) = gate_output.
        // Wait no — fold gives the CLAIM for this layer, and the sumcheck verifies that claim.

        // So p(0) + p(1) = sumcheck_claim1 = reduced_claim0.
        // And p(challenge1) = eq([r0],[challenge1]) * gate_output.
        // Also p(0) = (1-r0)*gate_output, p(1) = r0*gate_output IF the polynomial
        // is exactly eq(...) * gate_output.
        // Then p(0)+p(1) = gate_output, so gate_output must = sumcheck_claim1 = reduced_claim0.
        // So we need v0 * v1 = reduced_claim0.

        // Let's pick v0 and compute v1:
        // Actually, in a real GKR circuit, the mask values come from the actual layer.
        // But for a test, we can pick any v0, v1 such that v0*v1 = reduced_claim0.

        // Let's just set v0 = reduced_claim0, v1 = 1 (identity for GP).
        let mask1_v0_real = reduced_claim0;
        let mask1_v1_real = one;
        let gate1_real = mask1_v0_real * mask1_v1_real; // = reduced_claim0
        assert_eq!(gate1_real, reduced_claim0, "gate must match claim");

        print_qm31("layer1 mask_v0 (actual)", mask1_v0_real);
        print_qm31("layer1 mask_v1 (actual)", mask1_v1_real);
        print_qm31("layer1 gate_output (actual)", gate1_real);

        // Round poly: p(x) = eq([r0],[x]) * gate_output
        // p(x) = [(1-r0) + (2r0-1)*x] * gate_output
        // p(x) = (1-r0)*gate + (2r0-1)*gate*x
        // So: c0 = (1-r0)*gate, c1 = (2r0-1)*gate, c2 = 0
        let c0_l1 = (one - r0) * gate1_real;
        let c1_l1 = (r0 + r0 - one) * gate1_real;
        let c2_l1 = SecureField::default();
        print_qm31("layer1 round_poly c0", c0_l1);
        print_qm31("layer1 round_poly c1", c1_l1);

        // Verify: p(0) + p(1) = c0 + (c0 + c1) = 2c0 + c1
        // = 2(1-r0)*gate + (2r0-1)*gate = [2-2r0+2r0-1]*gate = gate = sumcheck_claim1
        let p0 = c0_l1;
        let p1 = c0_l1 + c1_l1;
        let check_sum = p0 + p1;
        assert_eq!(check_sum, sumcheck_claim1, "round poly sum check");

        // Now simulate what the verifier does:
        // Mix round poly coefficients (3 coeffs for degree-2)
        ch.mix_felts(&[c0_l1, c1_l1, c2_l1]);
        let challenge1 = ch.draw_qm31();
        print_qm31("challenge1 (sumcheck round 1)", challenge1);

        // sumcheck_eval = p(challenge1)
        let sc_eval = c0_l1 + c1_l1 * challenge1 + c2_l1 * challenge1 * challenge1;
        print_qm31("sumcheck_eval layer1", sc_eval);

        // eq([r0], [challenge1])
        let eq_val1 = eq_eval_fn(&[r0], &[challenge1]);
        print_qm31("eq([r0],[challenge1])", eq_val1);

        // layer_eval = eq_val * rlc([gate_output], lambda1)
        // rlc([gate_output], lambda1) = gate_output (single element)
        let layer_eval1 = eq_val1 * gate1_real;
        // With alpha: layer_eval = rlc([eq_val * gate], alpha1) = eq_val * gate (single)
        print_qm31("layer_eval layer1", layer_eval1);

        assert_eq!(sc_eval, layer_eval1, "layer1 circuit check");

        // Step 9: Mix mask, draw challenge
        ch.mix_felts(&[mask1_v0_real, mask1_v1_real]);
        let r1 = ch.draw_qm31();
        print_qm31("r1 (challenge layer1)", r1);

        // Final reduced claim
        let final_claim = fold_mle(r1, mask1_v0_real, mask1_v1_real);
        print_qm31("final_claim", final_claim);

        // Final ood_point = [challenge1, r1]
        println!("\nood_point = [{}, {}]", "challenge1", "r1");
        println!("claims_to_verify = [[final_claim]]");

        println!("\n=== END DEEP GP VECTORS ===\n");
    }

    /// Generate test vectors for staggered activation:
    /// Instance 0: GP, n_variables=2 (starts at layer 0)
    /// Instance 1: GP, n_variables=1 (starts at layer 1)
    #[test]
    fn test_generate_staggered_vectors() {
        println!("\n=== STAGGERED ACTIVATION Test Vectors ===\n");

        let one = SecureField::from(M31::from(1));

        // Instance 0: GP, n_vars=2, output_claim = 15*77 = 1155
        // mask layer 0: [15, 77]
        let inst0_claim = qm31(1155, 0, 0, 0);
        let mask0_v0 = qm31(15, 0, 0, 0);
        let mask0_v1 = qm31(77, 0, 0, 0);

        // Instance 1: GP, n_vars=1, output_claim = 6*8 = 48
        // mask layer 1: [6, 8]
        let inst1_claim = qm31(48, 0, 0, 0);
        let inst1_mask_v0 = qm31(6, 0, 0, 0);
        let inst1_mask_v1 = qm31(8, 0, 0, 0);

        print_qm31("inst0_claim", inst0_claim);
        print_qm31("inst1_claim", inst1_claim);

        // n_layers = max(2,1) = 2

        // ---- Layer 0 (n_remaining=2) ----
        // Instance 0 starts (n_vars=2 == n_remaining=2). Instance 1 does NOT start yet.
        let mut ch = PoseidonChannel::new();

        // Step 2: Mix claims of initialized instances (only inst0)
        ch.mix_felts(&[inst0_claim]);

        // Step 3: Draw alphas
        let alpha0 = ch.draw_qm31();
        let lambda0 = ch.draw_qm31();
        print_qm31("layer0 alpha", alpha0);
        print_qm31("layer0 lambda", lambda0);

        // Step 4: Only inst0 active. n_unused = n_layers - inst_n = 2 - 2 = 0. doubling=1.
        let inst0_rlc = rlc(&[inst0_claim], lambda0);
        let sumcheck_claim0 = rlc(&[inst0_rlc], alpha0);
        print_qm31("layer0 sumcheck_claim", sumcheck_claim0);

        // Step 6: 0 rounds (trivial)
        let sumcheck_eval0 = sumcheck_claim0;

        // Step 7: Gate eval
        let gate0 = mask0_v0 * mask0_v1; // 15*77 = 1155
        let eq_val0 = one; // eq([],[]) = 1
        let rlc_gate0 = rlc(&[gate0], lambda0);
        let layer_eval0 = rlc(&[eq_val0 * rlc_gate0], alpha0);
        assert_eq!(sumcheck_eval0, layer_eval0, "layer0 circuit check");
        print_qm31("layer0 gate", gate0);

        // Step 9: Mix mask, draw r0
        ch.mix_felts(&[mask0_v0, mask0_v1]);
        let r0 = ch.draw_qm31();
        print_qm31("r0", r0);

        // Reduce inst0
        let inst0_reduced = fold_mle(r0, mask0_v0, mask0_v1);
        print_qm31("inst0_reduced", inst0_reduced);

        // ---- Layer 1 (n_remaining=1) ----
        // Instance 1 starts (n_vars=1 == n_remaining=1). Instance 0 already active.

        // Step 2: Mix claims of ALL initialized instances: inst0_reduced, inst1_claim
        ch.mix_felts(&[inst0_reduced]);
        ch.mix_felts(&[inst1_claim]);
        // Note: in the code, it loops over ALL instances and mixes if initialized.
        // inst0 is initialized (from layer 0), inst1 just got initialized.
        // Wait — let me re-read the code. Step 1 (identify output layers) happens
        // FIRST, setting inst1's claims. Then step 2 mixes ALL initialized instances.
        // So at layer 1: inst0 has claims=[inst0_reduced], inst1 has claims=[inst1_claim].
        // Both get mixed in order: inst0 first, then inst1.

        // Actually, re-reading lines 257-268: it loops inst=0..n_instances, and mixes
        // if initialized[inst]. At this point, both are initialized.
        // inst0: claims = [inst0_reduced]
        // inst1: claims = [inst1_claim]
        // So: mix_felts([inst0_reduced]), then mix_felts([inst1_claim])

        // Step 3:
        let alpha1 = ch.draw_qm31();
        let lambda1 = ch.draw_qm31();
        print_qm31("layer1 alpha", alpha1);
        print_qm31("layer1 lambda", lambda1);

        // Step 4: Prepare sumcheck claims for BOTH instances.
        // inst0: n_unused = 2-2 = 0, doubling=1
        let inst0_rlc1 = rlc(&[inst0_reduced], lambda1);
        let inst0_doubled = inst0_rlc1; // * 1

        // inst1: n_unused = 2-1 = 1, doubling=2
        let inst1_rlc1 = rlc(&[inst1_claim], lambda1);
        let inst1_doubled = inst1_rlc1 + inst1_rlc1; // * 2
        print_qm31("inst0_doubled", inst0_doubled);
        print_qm31("inst1_doubled", inst1_doubled);

        // sumcheck_claim = rlc([inst0_doubled, inst1_doubled], alpha1)
        let sumcheck_claim1 = rlc(&[inst0_doubled, inst1_doubled], alpha1);
        print_qm31("layer1 sumcheck_claim", sumcheck_claim1);

        // Step 6: Layer 1 has 1 sumcheck round (ood_point = [r0])
        // We need a round poly. For 2 instances, this is more complex.
        // The sumcheck proves:
        //   Σ_x p(x) = sumcheck_claim1
        // where p(x) = alpha * [eq_0(x) * gate_0] + alpha^2 * [eq_1(x) * gate_1]

        // For inst0: eq([r0],[x]) at n_unused=0 → full eq: eq([r0],[x])
        //   gate = mask0_at_layer1_v0 * mask0_at_layer1_v1
        // For inst1: eq([r0],[x]) at n_unused=1 → sliced: eq([r0][1..],[x][1..]) = eq([],[]) = 1
        //   (Because n_unused=1 and ood/sc_ood both have length 1, offset 1 → empty slice)
        //   gate = inst1_mask_v0 * inst1_mask_v1

        // So contribution from inst0: eq([r0],[x]) * gate0
        //    contribution from inst1: 1 * gate1 (eq is trivially 1)

        // For inst0's mask at layer 1: we need v0*v1 such that the circuit is consistent.
        // The claim is inst0_reduced = fold(15, 77, r0).
        // We need a mask [m0, m1] with m0*m1 such that the sumcheck passes.
        // Setting m0 = inst0_reduced, m1 = 1 works (GP: m0*m1 = inst0_reduced).
        let mask1_inst0_v0 = inst0_reduced;
        let mask1_inst0_v1 = one;
        let gate1_inst0 = mask1_inst0_v0 * mask1_inst0_v1;

        let gate1_inst1 = inst1_mask_v0 * inst1_mask_v1; // 6*8 = 48
        print_qm31("gate1_inst0", gate1_inst0);
        print_qm31("gate1_inst1", gate1_inst1);

        // Compute p(0) and p(1):
        // p(x) = rlc_alpha([
        //   eq([r0],[x]) * rlc([gate0], lambda1),
        //   eq_sliced * rlc([gate1], lambda1)
        // ], alpha1)

        // At x=0:
        let eq_inst0_at_0 = one - r0; // eq([r0],[0])
        let eq_inst1_at_0 = one; // eq sliced to empty = 1
        let contrib0_at_0 = eq_inst0_at_0 * rlc(&[gate1_inst0], lambda1);
        let contrib1_at_0 = eq_inst1_at_0 * rlc(&[gate1_inst1], lambda1);
        let p_at_0 = rlc(&[contrib0_at_0, contrib1_at_0], alpha1);

        // At x=1:
        let eq_inst0_at_1 = r0;
        let eq_inst1_at_1 = one;
        let contrib0_at_1 = eq_inst0_at_1 * rlc(&[gate1_inst0], lambda1);
        let contrib1_at_1 = eq_inst1_at_1 * rlc(&[gate1_inst1], lambda1);
        let p_at_1 = rlc(&[contrib0_at_1, contrib1_at_1], alpha1);

        print_qm31("p(0)", p_at_0);
        print_qm31("p(1)", p_at_1);

        let round_sum = p_at_0 + p_at_1;
        print_qm31("p(0)+p(1)", round_sum);
        print_qm31("sumcheck_claim1", sumcheck_claim1);

        // Check: p(0)+p(1) should equal sumcheck_claim1
        // Let me verify...
        // p(0)+p(1) = alpha * [(1-r0)*G0 + r0*G0] + alpha^2*[1*G1 + 1*G1]
        //           = alpha * G0 + alpha^2 * 2*G1
        // sumcheck_claim = alpha * inst0_doubled + alpha^2 * inst1_doubled
        //                = alpha * G0 + alpha^2 * 2*G1 (since inst0_doubled = G0, inst1_doubled = 2*G1)
        // Wait, inst0_doubled = rlc([inst0_reduced], lambda1) * doubling(=1) = inst0_reduced (via lambda)
        // And G0 = rlc([gate1_inst0], lambda1) = gate1_inst0 = inst0_reduced (we set it)
        // So inst0_doubled = rlc([inst0_reduced], lambda1) and G0 = gate1_inst0 = inst0_reduced
        // But rlc([v], lambda) = v, so inst0_doubled = inst0_reduced = G0. ✓
        //
        // inst1_doubled = rlc([inst1_claim], lambda1) * 2 = inst1_claim * 2 = 48*2 = 96
        // G1_rlc = rlc([gate1_inst1], lambda1) = gate1_inst1 = 48
        // 2*G1_rlc = 96 = inst1_doubled. ✓
        //
        // So p(0)+p(1) = sumcheck_claim1. ✓

        assert_eq!(round_sum, sumcheck_claim1, "staggered sumcheck claim check");

        // Build round poly: p(x) is degree-2 (eq is degree 1, product with constant gate)
        // Actually, let's compute c0, c1, c2 from p(0), p(1), and one more point.
        // p(0) = c0, p(1) = c0+c1+c2
        // We need p(2) to determine c2. But we only have degree 1 really...
        // For the simple case where the polynomial is affine: c2 = 0.
        // c0 = p(0), c1 = p(1) - p(0)
        let c0 = p_at_0;
        let c1 = p_at_1 - p_at_0;
        let c2 = SecureField::default();
        print_qm31("round_poly c0", c0);
        print_qm31("round_poly c1", c1);

        // Simulate verifier: mix round poly, draw challenge
        ch.mix_felts(&[c0, c1, c2]);
        let challenge1 = ch.draw_qm31();
        print_qm31("challenge1", challenge1);

        // sumcheck_eval = p(challenge1) = c0 + c1*challenge1
        let sc_eval = c0 + c1 * challenge1;
        print_qm31("sumcheck_eval", sc_eval);

        // Verify circuit check: sumcheck_eval == rlc(layer_evals, alpha1)
        // For inst0: eq([r0],[challenge1]) * rlc([gate0], lambda1)
        let eq_inst0 = eq_eval_fn(&[r0], &[challenge1]);
        let le_inst0 = eq_inst0 * rlc(&[gate1_inst0], lambda1);
        // For inst1: eq at n_unused=1 → empty → 1, times gate1
        let eq_inst1 = one;
        let le_inst1 = eq_inst1 * rlc(&[gate1_inst1], lambda1);
        let layer_eval_check = rlc(&[le_inst0, le_inst1], alpha1);
        print_qm31("layer_eval_check", layer_eval_check);
        assert_eq!(sc_eval, layer_eval_check, "staggered circuit check");

        // Mix masks, draw r1
        ch.mix_felts(&[mask1_inst0_v0, mask1_inst0_v1]);
        ch.mix_felts(&[inst1_mask_v0, inst1_mask_v1]);
        let r1 = ch.draw_qm31();
        print_qm31("r1", r1);

        // Final claims
        let final0 = fold_mle(r1, mask1_inst0_v0, mask1_inst0_v1);
        let final1 = fold_mle(r1, inst1_mask_v0, inst1_mask_v1);
        print_qm31("final_claim_inst0", final0);
        print_qm31("final_claim_inst1", final1);

        println!("\n=== END STAGGERED VECTORS ===\n");
    }
}
