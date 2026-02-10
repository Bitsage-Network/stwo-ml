//! JSON serialization for proof types.
//!
//! Provides custom JSON conversion for stwo-ml proof types without requiring
//! `#[derive(Serialize)]` on foreign STWO types. Converts proofs to
//! JSON-friendly structures for external tool integration.

use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use starknet_ff::FieldElement;

use crate::aggregation::{AggregatedModelProofOnChain, LayerClaim};
use crate::cairo_serde::MLClaimMetadata;
use crate::components::matmul::{MatMulSumcheckProofOnChain, RoundPoly};

/// Convert a SecureField (QM31) to a 4-element u32 array.
fn qm31_to_array(val: SecureField) -> [u32; 4] {
    [val.0 .0 .0, val.0 .1 .0, val.1 .0 .0, val.1 .1 .0]
}

/// Convert a FieldElement to a hex string.
fn felt_to_hex(val: &FieldElement) -> String {
    format!("0x{:x}", val)
}

/// Serialize a RoundPoly to a JSON-compatible string.
fn round_poly_to_json(rp: &RoundPoly) -> String {
    format!(
        r#"{{"c0":{:?},"c1":{:?},"c2":{:?}}}"#,
        qm31_to_array(rp.c0),
        qm31_to_array(rp.c1),
        qm31_to_array(rp.c2),
    )
}

/// Serialize a MatMulSumcheckProofOnChain to a JSON string.
fn matmul_proof_to_json(proof: &MatMulSumcheckProofOnChain) -> String {
    let round_polys: Vec<String> = proof.round_polys.iter().map(round_poly_to_json).collect();
    format!(
        r#"{{"m":{},"k":{},"n":{},"num_rounds":{},"claimed_sum":{:?},"round_polys":[{}],"final_a_eval":{:?},"final_b_eval":{:?},"a_commitment":"{}","b_commitment":"{}"}}"#,
        proof.m,
        proof.k,
        proof.n,
        proof.num_rounds,
        qm31_to_array(proof.claimed_sum),
        round_polys.join(","),
        qm31_to_array(proof.final_a_eval),
        qm31_to_array(proof.final_b_eval),
        felt_to_hex(&proof.a_commitment),
        felt_to_hex(&proof.b_commitment),
    )
}

/// Serialize a LayerClaim to a JSON string.
fn layer_claim_to_json(claim: &LayerClaim) -> String {
    format!(
        r#"{{"layer_index":{},"claimed_sum":{:?},"trace_rows":{}}}"#,
        claim.layer_index,
        qm31_to_array(claim.claimed_sum),
        claim.trace_rows,
    )
}

/// Serialize an `AggregatedModelProofOnChain` with metadata to a JSON string.
///
/// This produces a human-readable JSON representation of the proof,
/// suitable for debugging, external tooling, or archival.
pub fn proof_to_json(
    proof: &AggregatedModelProofOnChain,
    metadata: &MLClaimMetadata,
) -> String {
    let matmul_proofs: Vec<String> = proof
        .matmul_proofs
        .iter()
        .map(|(layer_idx, mp)| {
            format!(
                r#"{{"layer_index":{},"proof":{}}}"#,
                layer_idx,
                matmul_proof_to_json(mp),
            )
        })
        .collect();

    let activation_claims: Vec<String> = proof
        .activation_claims
        .iter()
        .map(layer_claim_to_json)
        .collect();

    let has_unified_stark = proof.unified_stark.is_some();

    format!(
        r#"{{"metadata":{{"model_id":"{}","num_layers":{},"activation_type":{},"io_commitment":"{}","weight_commitment":"{}"}},"matmul_proofs":[{}],"activation_claims":[{}],"has_unified_stark":{},"output_shape":[{},{}]}}"#,
        felt_to_hex(&metadata.model_id),
        metadata.num_layers,
        metadata.activation_type,
        felt_to_hex(&metadata.io_commitment),
        felt_to_hex(&metadata.weight_commitment),
        matmul_proofs.join(","),
        activation_claims.join(","),
        has_unified_stark,
        proof.execution.output.rows,
        proof.execution.output.cols,
    )
}

/// Serialize an M31Matrix to a flat JSON array of u32 values.
pub fn matrix_to_json(data: &[M31], rows: usize, cols: usize) -> String {
    let vals: Vec<String> = data.iter().map(|v| v.0.to_string()).collect();
    format!(
        r#"{{"rows":{},"cols":{},"data":[{}]}}"#,
        rows,
        cols,
        vals.join(","),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::qm31::QM31;

    #[test]
    fn test_qm31_to_array() {
        let val = QM31(CM31(M31::from(1), M31::from(2)), CM31(M31::from(3), M31::from(4)));
        let arr = qm31_to_array(val);
        assert_eq!(arr, [1, 2, 3, 4]);
    }

    #[test]
    fn test_felt_to_hex() {
        let felt = FieldElement::from(0xdeadbeefu64);
        let hex = felt_to_hex(&felt);
        assert!(hex.starts_with("0x"));
        assert!(hex.contains("deadbeef"));
    }

    #[test]
    fn test_round_poly_to_json() {
        let rp = RoundPoly {
            c0: QM31(CM31(M31::from(1), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            c1: QM31(CM31(M31::from(2), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            c2: QM31(CM31(M31::from(3), M31::from(0)), CM31(M31::from(0), M31::from(0))),
        };
        let json = round_poly_to_json(&rp);
        assert!(json.contains("\"c0\""));
        assert!(json.contains("\"c1\""));
        assert!(json.contains("\"c2\""));
    }

    #[test]
    fn test_matrix_to_json() {
        let data = vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)];
        let json = matrix_to_json(&data, 2, 2);
        assert!(json.contains("\"rows\":2"));
        assert!(json.contains("\"cols\":2"));
        assert!(json.contains("[1,2,3,4]"));
    }
}
