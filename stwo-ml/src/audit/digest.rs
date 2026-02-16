//! M31-native digest utilities for the audit pipeline.
//!
//! All audit hashing uses Poseidon2-M31 digests (8 M31 elements, 248-bit).
//! This module provides conversion between M31 digests and hex strings,
//! packing into felt252 pairs for on-chain storage, and byte hashing.

use stwo::core::fields::m31::BaseField as M31;

use crate::crypto::poseidon2_m31::{poseidon2_hash, RATE};

// ─── Types ──────────────────────────────────────────────────────────────────

/// M31 digest: 8 M31 elements (248-bit, ~124-bit collision resistance).
pub type M31Digest = [M31; RATE];

/// Zero digest (all M31 zero).
pub const ZERO_DIGEST: M31Digest = [M31::from_u32_unchecked(0); RATE];

// ─── Hex Conversion ─────────────────────────────────────────────────────────

/// Convert an M31 digest to a hex string.
///
/// Format: "0x" followed by 8 zero-padded 8-char hex values (66 chars total).
/// Example: "0x0000001200000034000000560000007800000009000000ab000000cd000000ef"
pub fn digest_to_hex(digest: &M31Digest) -> String {
    let mut s = String::with_capacity(66);
    s.push_str("0x");
    for &elem in digest {
        s.push_str(&format!("{:08x}", elem.0));
    }
    s
}

/// Parse a hex string back into an M31 digest.
///
/// Accepts "0x" prefix followed by 64 hex chars (8 x 8 chars).
pub fn hex_to_digest(hex: &str) -> Result<M31Digest, String> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    if hex.len() != 64 {
        return Err(format!(
            "expected 64 hex chars for M31 digest, got {}",
            hex.len()
        ));
    }

    let mut digest = ZERO_DIGEST;
    for i in 0..RATE {
        let chunk = &hex[i * 8..(i + 1) * 8];
        let val = u32::from_str_radix(chunk, 16)
            .map_err(|e| format!("invalid hex chunk '{}': {}", chunk, e))?;
        digest[i] = M31::from_u32_unchecked(val);
    }
    Ok(digest)
}

/// Parse a hex string to M31Digest, returning ZERO_DIGEST on failure.
pub fn parse_digest_or_zero(hex: &str) -> M31Digest {
    hex_to_digest(hex).unwrap_or(ZERO_DIGEST)
}

// ─── felt252 Packing ────────────────────────────────────────────────────────

/// Pack an M31 digest into two felt252 values for on-chain storage.
///
/// `lo = m31[0] + m31[1]*2^31 + m31[2]*2^62 + m31[3]*2^93` (124 bits)
/// `hi = m31[4] + m31[5]*2^31 + m31[6]*2^62 + m31[7]*2^93` (124 bits)
///
/// Returns `(lo, hi)` as big-endian byte arrays suitable for FieldElement construction.
pub fn pack_digest_felt252(digest: &M31Digest) -> ([u8; 32], [u8; 32]) {
    let pack_half = |elems: &[M31]| -> [u8; 32] {
        let mut val = [0u8; 32];
        // Pack 4 M31 values into a 128-bit number (fits in felt252)
        let mut acc: u128 = 0;
        for (i, &elem) in elems.iter().take(4).enumerate() {
            acc |= (elem.0 as u128) << (31 * i);
        }
        // Store as big-endian in the last 16 bytes of the 32-byte array
        let bytes = acc.to_be_bytes();
        val[16..].copy_from_slice(&bytes);
        val
    };

    let lo = pack_half(&digest[..4]);
    let hi = pack_half(&digest[4..]);
    (lo, hi)
}

/// Unpack two felt252 byte arrays back into an M31 digest.
pub fn unpack_digest_felt252(lo: &[u8; 32], hi: &[u8; 32]) -> M31Digest {
    let unpack_half = |bytes: &[u8; 32]| -> [M31; 4] {
        let mut be_bytes = [0u8; 16];
        be_bytes.copy_from_slice(&bytes[16..]);
        let acc = u128::from_be_bytes(be_bytes);
        let mask = (1u128 << 31) - 1; // 0x7FFFFFFF
        [
            M31::from_u32_unchecked((acc & mask) as u32),
            M31::from_u32_unchecked(((acc >> 31) & mask) as u32),
            M31::from_u32_unchecked(((acc >> 62) & mask) as u32),
            M31::from_u32_unchecked(((acc >> 93) & mask) as u32),
        ]
    };

    let lo_parts = unpack_half(lo);
    let hi_parts = unpack_half(hi);
    [
        lo_parts[0], lo_parts[1], lo_parts[2], lo_parts[3],
        hi_parts[0], hi_parts[1], hi_parts[2], hi_parts[3],
    ]
}

/// Pack an M31 digest into two u128 values (for hex formatting as felt252).
pub fn pack_digest_u128(digest: &M31Digest) -> (u128, u128) {
    let pack_half = |elems: &[M31]| -> u128 {
        let mut acc: u128 = 0;
        for (i, &elem) in elems.iter().take(4).enumerate() {
            acc |= (elem.0 as u128) << (31 * i);
        }
        acc
    };

    (pack_half(&digest[..4]), pack_half(&digest[4..]))
}

/// Format a packed digest half as a felt252 hex string.
pub fn packed_half_to_hex(val: u128) -> String {
    format!("{:#066x}", val)
}

// ─── Integer Encoding ───────────────────────────────────────────────────────

/// Encode a u64 as two M31 limbs (31 bits each).
///
/// `lo = val & 0x7FFFFFFF` (bits 0..30)
/// `hi = (val >> 31) & 0x7FFFFFFF` (bits 31..61)
///
/// Note: u64 has 64 bits, M31 has 31 bits. Two limbs cover 62 bits,
/// which is sufficient since u64 values in practice (timestamps, counts)
/// fit in 62 bits.
pub fn u64_to_m31(val: u64) -> [M31; 2] {
    let mask = (1u64 << 31) - 1;
    [
        M31::from_u32_unchecked((val & mask) as u32),
        M31::from_u32_unchecked(((val >> 31) & mask) as u32),
    ]
}

/// Encode a u32 as a single M31 element.
///
/// Values >= 2^31-1 are reduced mod p.
pub fn u32_to_m31(val: u32) -> M31 {
    M31::from(val)
}

// ─── Byte Hashing ───────────────────────────────────────────────────────────

/// Hash arbitrary bytes to an M31 digest.
///
/// Packing: 3 bytes per M31 element (24 bits < 31 bits), length-prefixed.
/// First element is the byte length, then packed data, then Poseidon2 hash.
pub fn hash_bytes_m31(data: &[u8]) -> M31Digest {
    let mut input = Vec::with_capacity(1 + (data.len() + 2) / 3);

    // Length prefix
    input.push(M31::from(data.len() as u32));

    // Pack 3 bytes per M31 element
    for chunk in data.chunks(3) {
        let mut val = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            val |= (b as u32) << (i * 8);
        }
        input.push(M31::from(val));
    }

    poseidon2_hash(&input)
}

/// Hash a hex-encoded felt252 string to an M31 digest.
///
/// Used at the ZKML boundary: felt252 commitment values from the ZKML prover
/// are hashed into M31 space for audit aggregation.
pub fn hash_felt_hex_m31(hex: &str) -> M31Digest {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    let bytes = hex.as_bytes();
    hash_bytes_m31(bytes)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_roundtrip() {
        let digest = [
            M31::from(0x12u32),
            M31::from(0x34u32),
            M31::from(0x56u32),
            M31::from(0x78u32),
            M31::from(0x9Au32),
            M31::from(0xBCu32),
            M31::from(0xDEu32),
            M31::from(0xF0u32),
        ];

        let hex = digest_to_hex(&digest);
        assert_eq!(hex.len(), 66); // "0x" + 64 hex chars
        assert!(hex.starts_with("0x"));

        let parsed = hex_to_digest(&hex).unwrap();
        assert_eq!(parsed, digest);
    }

    #[test]
    fn test_hex_zero_digest() {
        let hex = digest_to_hex(&ZERO_DIGEST);
        assert_eq!(
            hex,
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        );
        let parsed = hex_to_digest(&hex).unwrap();
        assert_eq!(parsed, ZERO_DIGEST);
    }

    #[test]
    fn test_hex_parse_errors() {
        assert!(hex_to_digest("0x123").is_err()); // too short
        assert!(hex_to_digest("not_hex_at_all_padding_to_64_chars_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx").is_err());
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        let digest = [
            M31::from(42u32),
            M31::from(1000u32),
            M31::from(999999u32),
            M31::from(0x7FFFFFFFu32), // max M31
            M31::from(1u32),
            M31::from(0u32),
            M31::from(12345u32),
            M31::from(67890u32),
        ];

        let (lo, hi) = pack_digest_felt252(&digest);
        let unpacked = unpack_digest_felt252(&lo, &hi);
        assert_eq!(unpacked, digest);
    }

    #[test]
    fn test_pack_unpack_zero() {
        let (lo, hi) = pack_digest_felt252(&ZERO_DIGEST);
        let unpacked = unpack_digest_felt252(&lo, &hi);
        assert_eq!(unpacked, ZERO_DIGEST);
    }

    #[test]
    fn test_u64_to_m31() {
        let limbs = u64_to_m31(0);
        assert_eq!(limbs[0], M31::from(0u32));
        assert_eq!(limbs[1], M31::from(0u32));

        let limbs = u64_to_m31(1);
        assert_eq!(limbs[0], M31::from(1u32));
        assert_eq!(limbs[1], M31::from(0u32));

        // Test with value that uses both limbs
        let val = (1u64 << 31) + 42;
        let limbs = u64_to_m31(val);
        assert_eq!(limbs[0], M31::from(42u32));
        assert_eq!(limbs[1], M31::from(1u32));

        // Max 62-bit value
        let val = (1u64 << 62) - 1;
        let limbs = u64_to_m31(val);
        assert_eq!(limbs[0].0, 0x7FFFFFFF);
        assert_eq!(limbs[1].0, 0x7FFFFFFF);
    }

    #[test]
    fn test_hash_bytes_deterministic() {
        let data = b"hello world";
        let h1 = hash_bytes_m31(data);
        let h2 = hash_bytes_m31(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_bytes_different_inputs() {
        let h1 = hash_bytes_m31(b"hello");
        let h2 = hash_bytes_m31(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_bytes_empty() {
        let h = hash_bytes_m31(b"");
        assert_ne!(h, ZERO_DIGEST);
    }

    #[test]
    fn test_hash_felt_hex() {
        let h1 = hash_felt_hex_m31("0xabc");
        let h2 = hash_felt_hex_m31("0xabc");
        assert_eq!(h1, h2);

        let h3 = hash_felt_hex_m31("0xdef");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_parse_digest_or_zero() {
        let hex = digest_to_hex(&[M31::from(1u32); RATE]);
        let d = parse_digest_or_zero(&hex);
        assert_eq!(d, [M31::from(1u32); RATE]);

        let d = parse_digest_or_zero("invalid");
        assert_eq!(d, ZERO_DIGEST);
    }

    #[test]
    fn test_pack_digest_u128() {
        let digest = [
            M31::from(1u32),
            M31::from(0u32),
            M31::from(0u32),
            M31::from(0u32),
            M31::from(2u32),
            M31::from(0u32),
            M31::from(0u32),
            M31::from(0u32),
        ];
        let (lo, hi) = pack_digest_u128(&digest);
        assert_eq!(lo, 1);
        assert_eq!(hi, 2);
    }
}
