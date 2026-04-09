//! Compact binary serialization for GKR proofs.
//!
//! Replaces JSON hex serialization with bincode for ~2x size reduction and ~4x
//! faster (de)serialization. The format is not stable across versions — use
//! `PROOF_FORMAT_VERSION` to detect incompatible changes.
//!
//! # Usage
//!
//! ```ignore
//! use obelyzk::binary_serde::{proof_to_bytes, proof_from_bytes};
//!
//! let bytes = proof_to_bytes(&gkr_proof)?;
//! let recovered = proof_from_bytes(&bytes)?;
//! ```
//!
//! Requires the `binary-proof` feature flag.

use crate::gkr::GKRProof;

/// Current binary proof format version.
/// Increment when the serialized layout changes in a backward-incompatible way.
pub const PROOF_FORMAT_VERSION: u32 = 1;

/// Magic bytes identifying an ObelyZK binary proof file.
const MAGIC: [u8; 4] = *b"OZKP";

/// Header prepended to every binary proof.
#[derive(Debug, Clone, Copy)]
struct ProofHeader {
    magic: [u8; 4],
    version: u32,
    /// Byte length of the proof payload (after the header).
    payload_len: u64,
}

const HEADER_SIZE: usize = 4 + 4 + 8; // magic + version + payload_len

impl ProofHeader {
    fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..16].copy_from_slice(&self.payload_len.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Self {
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        let version = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let payload_len = u64::from_le_bytes([
            buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
        ]);
        Self {
            magic,
            version,
            payload_len,
        }
    }
}

/// Errors that can occur during binary (de)serialization.
#[derive(Debug, thiserror::Error)]
pub enum BinarySerdeError {
    #[error("bincode serialization error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("invalid magic bytes (expected OZKP, got {0:?})")]
    InvalidMagic([u8; 4]),
    #[error("unsupported proof format version {found} (expected {expected})")]
    VersionMismatch { expected: u32, found: u32 },
    #[error("proof data truncated: expected {expected} bytes, got {found}")]
    Truncated { expected: u64, found: usize },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Serialize a `GKRProof` to compact binary bytes.
///
/// Format: `[OZKP][version:u32le][payload_len:u64le][bincode payload]`
pub fn proof_to_bytes(proof: &GKRProof) -> Result<Vec<u8>, BinarySerdeError> {
    let payload = bincode::serialize(proof)?;
    let header = ProofHeader {
        magic: MAGIC,
        version: PROOF_FORMAT_VERSION,
        payload_len: payload.len() as u64,
    };
    let mut out = Vec::with_capacity(HEADER_SIZE + payload.len());
    out.extend_from_slice(&header.to_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Deserialize a `GKRProof` from binary bytes.
pub fn proof_from_bytes(data: &[u8]) -> Result<GKRProof, BinarySerdeError> {
    if data.len() < HEADER_SIZE {
        return Err(BinarySerdeError::Truncated {
            expected: HEADER_SIZE as u64,
            found: data.len(),
        });
    }
    let header = ProofHeader::from_bytes(data[..HEADER_SIZE].try_into().unwrap());
    if header.magic != MAGIC {
        return Err(BinarySerdeError::InvalidMagic(header.magic));
    }
    if header.version != PROOF_FORMAT_VERSION {
        return Err(BinarySerdeError::VersionMismatch {
            expected: PROOF_FORMAT_VERSION,
            found: header.version,
        });
    }
    let payload = &data[HEADER_SIZE..];
    if (payload.len() as u64) < header.payload_len {
        return Err(BinarySerdeError::Truncated {
            expected: header.payload_len,
            found: payload.len(),
        });
    }
    let proof = bincode::deserialize(&payload[..header.payload_len as usize])?;
    Ok(proof)
}

/// Write a `GKRProof` to a file in binary format.
pub fn proof_to_file(proof: &GKRProof, path: &std::path::Path) -> Result<(), BinarySerdeError> {
    let bytes = proof_to_bytes(proof)?;
    std::fs::write(path, &bytes)?;
    Ok(())
}

/// Read a `GKRProof` from a binary file.
pub fn proof_from_file(path: &std::path::Path) -> Result<GKRProof, BinarySerdeError> {
    let data = std::fs::read(path)?;
    proof_from_bytes(&data)
}

/// Returns the size of the binary-encoded proof without the full serialization.
/// Useful for estimating proof sizes before committing to serialization.
pub fn estimate_proof_size(proof: &GKRProof) -> Result<usize, BinarySerdeError> {
    let size = bincode::serialized_size(proof)? as usize;
    Ok(HEADER_SIZE + size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gkr::types::*;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::QM31;

    fn make_test_proof() -> GKRProof {
        let zero = QM31::from(M31::from(0));
        let one = QM31::from(M31::from(1));
        GKRProof {
            layer_proofs: vec![LayerProof::Add {
                lhs_eval: one,
                rhs_eval: zero,
                trunk_idx: 0,
            }],
            output_claim: GKRClaim {
                point: vec![one, zero],
                value: one,
            },
            input_claim: GKRClaim {
                point: vec![zero],
                value: zero,
            },
            weight_commitments: vec![],
            weight_openings: vec![],
            weight_claims: vec![],
            weight_opening_transcript_mode: WeightOpeningTranscriptMode::AggregatedOracleSumcheck,
            io_commitment: starknet_ff::FieldElement::ZERO,
            deferred_proofs: vec![],
            aggregated_binding: None,
            binding_groups: vec![],
            kv_cache_commitment: None,
            prev_kv_cache_commitment: None,
        }
    }

    #[test]
    fn test_roundtrip() {
        let proof = make_test_proof();
        let bytes = proof_to_bytes(&proof).expect("serialize");
        let recovered = proof_from_bytes(&bytes).expect("deserialize");

        assert_eq!(proof.layer_proofs.len(), recovered.layer_proofs.len());
        assert_eq!(proof.output_claim.value, recovered.output_claim.value);
        assert_eq!(proof.input_claim.value, recovered.input_claim.value);
        assert_eq!(proof.io_commitment, recovered.io_commitment);
    }

    #[test]
    fn test_header_validation() {
        let proof = make_test_proof();
        let mut bytes = proof_to_bytes(&proof).expect("serialize");

        // Corrupt magic
        bytes[0] = b'X';
        assert!(matches!(
            proof_from_bytes(&bytes),
            Err(BinarySerdeError::InvalidMagic(_))
        ));

        // Corrupt version
        let mut bytes = proof_to_bytes(&proof).expect("serialize");
        bytes[4] = 99;
        assert!(matches!(
            proof_from_bytes(&bytes),
            Err(BinarySerdeError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn test_truncated() {
        assert!(matches!(
            proof_from_bytes(&[0u8; 4]),
            Err(BinarySerdeError::Truncated { .. })
        ));
    }

    #[test]
    fn test_size_estimate() {
        let proof = make_test_proof();
        let bytes = proof_to_bytes(&proof).expect("serialize");
        let estimated = estimate_proof_size(&proof).expect("estimate");
        assert_eq!(bytes.len(), estimated);
    }

    #[test]
    fn test_compact_vs_json() {
        let proof = make_test_proof();
        let binary_bytes = proof_to_bytes(&proof).expect("serialize");
        let json_bytes = serde_json::to_vec(&proof).expect("json serialize");
        // Binary should be significantly smaller than JSON
        assert!(
            binary_bytes.len() < json_bytes.len(),
            "binary ({}) should be smaller than JSON ({})",
            binary_bytes.len(),
            json_bytes.len()
        );
    }

    #[test]
    fn test_matmul_proof_roundtrip() {
        use crate::components::matmul::RoundPoly;

        let v = |a: u32| QM31::from(M31::from(a));
        let round_poly = RoundPoly {
            c0: v(100),
            c1: v(200),
            c2: v(300),
        };

        let proof = GKRProof {
            layer_proofs: vec![
                LayerProof::MatMul {
                    round_polys: vec![round_poly; 13], // log2(8192) rounds
                    final_a_eval: v(42),
                    final_b_eval: v(99),
                },
                LayerProof::Add {
                    lhs_eval: v(1),
                    rhs_eval: v(2),
                    trunk_idx: 0,
                },
                LayerProof::Activation {
                    activation_type: crate::components::activation::ActivationType::SiLU,
                    logup_proof: None,
                    multiplicity_sumcheck: None,
                    activation_proof: None,
                    piecewise_proof: None,
                    input_eval: v(10),
                    output_eval: v(20),
                    table_commitment: starknet_ff::FieldElement::ZERO,
                    simd_combined: false,
                },
            ],
            output_claim: GKRClaim {
                point: (0..13).map(|i| v(i)).collect(),
                value: v(777),
            },
            input_claim: GKRClaim {
                point: (0..13).map(|i| v(i + 100)).collect(),
                value: v(888),
            },
            weight_commitments: vec![starknet_ff::FieldElement::ONE; 4],
            weight_openings: vec![],
            weight_claims: vec![
                WeightClaim {
                    weight_node_id: 1,
                    eval_point: vec![v(1), v(2), v(3)],
                    expected_value: v(42),
                },
            ],
            weight_opening_transcript_mode: WeightOpeningTranscriptMode::AggregatedOracleSumcheck,
            io_commitment: starknet_ff::FieldElement::TWO,
            deferred_proofs: vec![],
            aggregated_binding: None,
            binding_groups: vec![],
            kv_cache_commitment: Some(starknet_ff::FieldElement::THREE),
            prev_kv_cache_commitment: Some(starknet_ff::FieldElement::ZERO),
        };

        let bytes = proof_to_bytes(&proof).expect("serialize");
        let json_bytes = serde_json::to_vec(&proof).expect("json");

        eprintln!(
            "MatMul proof: binary={} bytes, JSON={} bytes, ratio={:.1}x",
            bytes.len(),
            json_bytes.len(),
            json_bytes.len() as f64 / bytes.len() as f64
        );

        let recovered = proof_from_bytes(&bytes).expect("deserialize");
        assert_eq!(recovered.layer_proofs.len(), 3);
        assert_eq!(recovered.weight_commitments.len(), 4);
        assert_eq!(recovered.weight_claims.len(), 1);
        assert_eq!(recovered.output_claim.point.len(), 13);
        assert!(recovered.kv_cache_commitment.is_some());

        // Verify matmul round polys survived
        if let LayerProof::MatMul { round_polys, final_a_eval, final_b_eval } = &recovered.layer_proofs[0] {
            assert_eq!(round_polys.len(), 13);
            assert_eq!(*final_a_eval, v(42));
            assert_eq!(*final_b_eval, v(99));
        } else {
            panic!("Expected MatMul layer proof");
        }
    }

    #[test]
    fn test_file_roundtrip() {
        let proof = make_test_proof();
        let tmp = std::env::temp_dir().join("obelyzk_test_proof.ozkp");
        proof_to_file(&proof, &tmp).expect("write");
        let recovered = proof_from_file(&tmp).expect("read");
        assert_eq!(proof.output_claim.value, recovered.output_claim.value);
        std::fs::remove_file(&tmp).ok();
    }
}
