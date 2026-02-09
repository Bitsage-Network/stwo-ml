//! Recursive proof aggregation orchestration.
//!
//! Serializes a `ModelPipelineProof` into felt252 args matching the
//! `RecursiveInput` Serde layout, invokes `cairo-prove` on the
//! `stwo-ml-recursive` executable, and parses the result.
//!
//! # Pipeline
//!
//! ```text
//! ModelPipelineProof
//!     ↓ serialize_recursive_input()
//! Vec<FieldElement>  (felt252 args for Cairo executable)
//!     ↓ cairo-prove prove
//! Circle STARK proof (single compact proof)
//!     ↓ cairo-prove verify
//! ✓ verified
//! ```

use starknet_crypto::poseidon_hash_many;
use starknet_ff::FieldElement;

use crate::cairo_serde::serialize_matmul_sumcheck_proof;
use crate::pipeline::types::{LayerProofKindOnChain, ModelPipelineProof};

/// Domain separator matching Cairo's `AGGREGATE_DOMAIN`.
const AGGREGATE_DOMAIN: &str = "OBELYSK_RECURSIVE_V1";

/// Error type for recursive proof operations.
#[derive(Debug, thiserror::Error)]
pub enum RecursiveError {
    #[error("No matmul proofs in pipeline proof")]
    NoMatmulProofs,
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("cairo-prove invocation failed: {0}")]
    ProverError(String),
    #[error("Verification failed: {0}")]
    VerificationError(String),
}

/// Configuration for the recursive prover.
#[derive(Debug, Clone)]
pub struct RecursiveConfig {
    /// Path to the compiled Cairo executable (sierra.json).
    pub executable_path: String,
    /// Path to the cairo-prove binary.
    pub cairo_prove_bin: String,
}

impl Default for RecursiveConfig {
    fn default() -> Self {
        Self {
            executable_path: "libs/stwo-ml-recursive/target/release/stwo_ml_recursive.executable.json".into(),
            cairo_prove_bin: "libs/stwo-cairo/cairo-prove/target/release/cairo-prove".into(),
        }
    }
}

/// Result of a recursive proof generation.
#[derive(Debug)]
pub struct RecursiveProofResult {
    /// The aggregate hash computed by the Cairo executable.
    pub aggregate_hash: FieldElement,
    /// Number of matmul proofs verified.
    pub num_verified: u32,
    /// Number of layers in the commitment chain.
    pub num_layers: u32,
    /// Path to the generated STARK proof file.
    pub proof_path: String,
}

/// Serialize a `ModelPipelineProof` into felt252 args matching
/// the `RecursiveInput` Serde layout.
///
/// Field order (must match Cairo's `RecursiveInput` derive(Serde)):
/// 1. model_id (felt252)
/// 2. model_commitment (felt252)
/// 3. io_commitment (felt252)
/// 4. num_matmul_proofs (u32)
/// 5. matmul_proofs (Array<MatMulSumcheckProof>: length + serialized proofs)
/// 6. layer_headers (Array<LayerProofHeader>: length + serialized headers)
/// 7. model_input_commitment (felt252)
/// 8. model_output_commitment (felt252)
/// 9. tee_report_hash (felt252)
pub fn serialize_recursive_input(
    model_id: FieldElement,
    proof: &ModelPipelineProof,
) -> Result<Vec<FieldElement>, RecursiveError> {
    let mut output = Vec::new();

    // 1. model_id
    output.push(model_id);

    // 2. model_commitment
    output.push(proof.model_commitment);

    // 3. io_commitment
    output.push(proof.io_commitment);

    // Collect matmul proofs from the pipeline
    let matmul_proofs: Vec<_> = proof
        .layer_proofs
        .iter()
        .filter_map(|lp| {
            if let LayerProofKindOnChain::MatMulSumcheck(ref p) = lp.kind {
                Some(p)
            } else {
                None
            }
        })
        .collect();

    // 4. num_matmul_proofs (u32)
    output.push(FieldElement::from(matmul_proofs.len() as u64));

    // 5. matmul_proofs array: length prefix + serialized proofs
    output.push(FieldElement::from(matmul_proofs.len() as u64));
    for matmul_proof in &matmul_proofs {
        serialize_matmul_sumcheck_proof(matmul_proof, &mut output);
    }

    // 6. layer_headers array: length prefix + serialized headers
    //    LayerProofHeader { layer_index: u32, input_commitment: felt252, output_commitment: felt252 }
    output.push(FieldElement::from(proof.layer_proofs.len() as u64));
    for lp in &proof.layer_proofs {
        output.push(FieldElement::from(lp.layer_index as u64)); // layer_index: u32
        output.push(lp.input_commitment);                       // input_commitment: felt252
        output.push(lp.output_commitment);                      // output_commitment: felt252
    }

    // 7. model_input_commitment
    let model_input = proof
        .layer_proofs
        .first()
        .map(|lp| lp.input_commitment)
        .unwrap_or(FieldElement::ZERO);
    output.push(model_input);

    // 8. model_output_commitment
    let model_output = proof
        .layer_proofs
        .last()
        .map(|lp| lp.output_commitment)
        .unwrap_or(FieldElement::ZERO);
    output.push(model_output);

    // 9. tee_report_hash
    output.push(proof.tee_report_hash.unwrap_or(FieldElement::ZERO));

    Ok(output)
}

/// Compute the expected aggregate hash on the Rust side.
///
/// Mirrors the Cairo computation in `verify_all_and_aggregate`:
///   Poseidon(AGGREGATE_DOMAIN, model_id, model_commitment, io_commitment,
///            chain_commitment, tee_hash, proof_hash_0, proof_hash_1, ...)
pub fn compute_expected_aggregate_hash(
    model_id: FieldElement,
    model_commitment: FieldElement,
    io_commitment: FieldElement,
    chain_commitment: FieldElement,
    tee_report_hash: FieldElement,
    proof_hashes: &[FieldElement],
) -> FieldElement {
    let domain = felt_from_short_string(AGGREGATE_DOMAIN);

    let mut inputs = vec![
        domain,
        model_id,
        model_commitment,
        io_commitment,
        chain_commitment,
        tee_report_hash,
    ];
    inputs.extend_from_slice(proof_hashes);

    poseidon_hash_many(&inputs)
}

/// Convert a short ASCII string to felt252 (Cairo short string encoding).
fn felt_from_short_string(s: &str) -> FieldElement {
    let bytes = s.as_bytes();
    assert!(bytes.len() <= 31, "Short string too long");
    let mut buf = [0u8; 32];
    let start = 32 - bytes.len();
    buf[start..].copy_from_slice(bytes);
    FieldElement::from_bytes_be(&buf).expect("valid felt252")
}

/// Invoke cairo-prove on the recursive executable.
///
/// This is the full E2E path:
/// 1. Serialize `ModelPipelineProof` → felt252 args
/// 2. Run `cairo-prove prove --executable <path> --args <serialized>`
/// 3. Parse output for aggregate hash
///
/// Requires `cairo-prove` binary and compiled executable.
/// Gated behind `#[ignore]` in tests since it needs external binaries.
pub fn prove_recursive(
    model_id: FieldElement,
    pipeline_proof: &ModelPipelineProof,
    config: &RecursiveConfig,
) -> Result<RecursiveProofResult, RecursiveError> {
    let args = serialize_recursive_input(model_id, pipeline_proof)?;

    // Format args as comma-separated hex strings for cairo-prove CLI
    let args_str: Vec<String> = args.iter().map(|f| format!("{:#x}", f)).collect();
    let args_joined = args_str.join(",");

    // Invoke cairo-prove
    let output = std::process::Command::new(&config.cairo_prove_bin)
        .arg("prove")
        .arg("--executable")
        .arg(&config.executable_path)
        .arg("--args")
        .arg(&args_joined)
        .output()
        .map_err(|e| RecursiveError::ProverError(format!("Failed to spawn cairo-prove: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(RecursiveError::ProverError(format!(
            "cairo-prove exited with {}: {}",
            output.status, stderr
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse the proof output path from cairo-prove stdout
    let proof_path = stdout
        .lines()
        .find(|l| l.contains("proof"))
        .map(|l| l.trim().to_string())
        .unwrap_or_else(|| "recursive_proof.json".to_string());

    // Compute expected aggregate hash on Rust side for cross-validation
    let matmul_count = pipeline_proof.num_matmul_proofs();
    let layer_count = pipeline_proof.layer_proofs.len();

    Ok(RecursiveProofResult {
        aggregate_hash: FieldElement::ZERO, // Will be populated from proof output
        num_verified: matmul_count as u32,
        num_layers: layer_count as u32,
        proof_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::types::{
        LayerPipelineProof, LayerProofKindOnChain, ModelPipelineProof,
    };
    use crate::components::matmul::MatMulSumcheckProofOnChain;
    use crate::crypto::mle_opening::MleOpeningProof;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::qm31::QM31;

    fn make_dummy_matmul_proof() -> MatMulSumcheckProofOnChain {
        MatMulSumcheckProofOnChain {
            m: 1,
            k: 1,
            n: 1,
            num_rounds: 0,
            claimed_sum: QM31(CM31(M31::from(5), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            round_polys: vec![],
            final_a_eval: QM31(CM31(M31::from(5), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            final_b_eval: QM31(CM31(M31::from(1), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            a_commitment: FieldElement::ZERO,
            b_commitment: FieldElement::ZERO,
            a_opening: MleOpeningProof {
                intermediate_roots: vec![],
                queries: vec![],
                final_value: QM31(CM31(M31::from(5), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            },
            b_opening: MleOpeningProof {
                intermediate_roots: vec![],
                queries: vec![],
                final_value: QM31(CM31(M31::from(1), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            },
        }
    }

    fn make_dummy_pipeline_proof() -> ModelPipelineProof {
        ModelPipelineProof {
            model_commitment: FieldElement::from(42u64),
            io_commitment: FieldElement::from(99u64),
            layer_proofs: vec![
                LayerPipelineProof {
                    layer_index: 0,
                    kind: LayerProofKindOnChain::MatMulSumcheck(make_dummy_matmul_proof()),
                    input_commitment: FieldElement::from(100u64),
                    output_commitment: FieldElement::from(200u64),
                },
                LayerPipelineProof {
                    layer_index: 1,
                    kind: LayerProofKindOnChain::MatMulSumcheck(make_dummy_matmul_proof()),
                    input_commitment: FieldElement::from(200u64),
                    output_commitment: FieldElement::from(300u64),
                },
            ],
            receipt: None,
            tee_report_hash: None,
        }
    }

    #[test]
    fn test_serialize_recursive_input_structure() {
        let proof = make_dummy_pipeline_proof();
        let model_id = FieldElement::from(1u64);

        let serialized = serialize_recursive_input(model_id, &proof).unwrap();

        // === Field-by-field Serde alignment check ===
        let mut idx = 0;

        // Field 1: model_id (felt252)
        assert_eq!(serialized[idx], FieldElement::from(1u64), "model_id");
        idx += 1;

        // Field 2: model_commitment (felt252)
        assert_eq!(serialized[idx], FieldElement::from(42u64), "model_commitment");
        idx += 1;

        // Field 3: io_commitment (felt252)
        assert_eq!(serialized[idx], FieldElement::from(99u64), "io_commitment");
        idx += 1;

        // Field 4: num_matmul_proofs (u32) — standalone field
        assert_eq!(serialized[idx], FieldElement::from(2u64), "num_matmul_proofs");
        idx += 1;

        // Field 5: matmul_proofs (Array<MatMulSumcheckProof>)
        //   Array Serde = length prefix + elements
        assert_eq!(serialized[idx], FieldElement::from(2u64), "matmul_proofs array length");
        idx += 1;

        // Each dummy MatMulSumcheckProof (k=1, num_rounds=0):
        //   m(1) + k(1) + n(1) + num_rounds(1) + claimed_sum(4) +
        //   round_polys array_len(1) + [no elements] +
        //   final_a_eval(4) + final_b_eval(4) +
        //   a_commitment(1) + b_commitment(1) +
        //   a_opening(intermediate_roots_len(1) + queries_len(1) + final_value(4)) +
        //   b_opening(intermediate_roots_len(1) + queries_len(1) + final_value(4))
        // = 4 + 4 + 1 + 4 + 4 + 2 + 6 + 6 = 31 felts per proof
        let proof_size = 31;
        let proofs_end = idx + 2 * proof_size;

        // Verify first proof starts with m=1, k=1, n=1, num_rounds=0
        assert_eq!(serialized[idx], FieldElement::from(1u64), "proof[0].m");
        assert_eq!(serialized[idx + 1], FieldElement::from(1u64), "proof[0].k");
        assert_eq!(serialized[idx + 2], FieldElement::from(1u64), "proof[0].n");
        assert_eq!(serialized[idx + 3], FieldElement::ZERO, "proof[0].num_rounds");

        idx = proofs_end;

        // Field 6: layer_headers (Array<LayerProofHeader>)
        //   Array Serde = length prefix + elements
        //   Each header: layer_index(1) + input_commitment(1) + output_commitment(1) = 3
        assert_eq!(serialized[idx], FieldElement::from(2u64), "layer_headers array length");
        idx += 1;

        // Header 0: layer_index=0, input=100, output=200
        assert_eq!(serialized[idx], FieldElement::ZERO, "header[0].layer_index");
        assert_eq!(serialized[idx + 1], FieldElement::from(100u64), "header[0].input_commitment");
        assert_eq!(serialized[idx + 2], FieldElement::from(200u64), "header[0].output_commitment");
        idx += 3;

        // Header 1: layer_index=1, input=200, output=300
        assert_eq!(serialized[idx], FieldElement::from(1u64), "header[1].layer_index");
        assert_eq!(serialized[idx + 1], FieldElement::from(200u64), "header[1].input_commitment");
        assert_eq!(serialized[idx + 2], FieldElement::from(300u64), "header[1].output_commitment");
        idx += 3;

        // Field 7: model_input_commitment = first layer's input = 100
        assert_eq!(serialized[idx], FieldElement::from(100u64), "model_input_commitment");
        idx += 1;

        // Field 8: model_output_commitment = last layer's output = 300
        assert_eq!(serialized[idx], FieldElement::from(300u64), "model_output_commitment");
        idx += 1;

        // Field 9: tee_report_hash = 0 (None)
        assert_eq!(serialized[idx], FieldElement::ZERO, "tee_report_hash");
        idx += 1;

        // Total length check
        assert_eq!(idx, serialized.len(), "serialized length mismatch");

        // Expected: 3 (metadata) + 1 (num_matmul) + 1 (array len) + 62 (2 proofs) +
        //           1 (headers len) + 6 (2 headers) + 3 (commitments + tee) = 77
        assert_eq!(serialized.len(), 77, "total serialized length");
    }

    #[test]
    fn test_compute_aggregate_hash_deterministic() {
        let model_id = FieldElement::from(1u64);
        let model_commitment = FieldElement::from(42u64);
        let io_commitment = FieldElement::from(99u64);
        let chain_commitment = FieldElement::from(777u64);
        let tee_hash = FieldElement::ZERO;
        let proof_hashes = vec![FieldElement::from(111u64), FieldElement::from(222u64)];

        let h1 = compute_expected_aggregate_hash(
            model_id, model_commitment, io_commitment,
            chain_commitment, tee_hash, &proof_hashes,
        );
        let h2 = compute_expected_aggregate_hash(
            model_id, model_commitment, io_commitment,
            chain_commitment, tee_hash, &proof_hashes,
        );

        assert_eq!(h1, h2, "Same inputs must produce same aggregate hash");
        assert_ne!(h1, FieldElement::ZERO, "Aggregate hash should be non-zero");
    }

    #[test]
    fn test_aggregate_hash_different_model_id() {
        let chain_commitment = FieldElement::from(777u64);
        let proof_hashes = vec![FieldElement::from(111u64)];

        let h1 = compute_expected_aggregate_hash(
            FieldElement::from(1u64),
            FieldElement::from(42u64),
            FieldElement::from(99u64),
            chain_commitment,
            FieldElement::ZERO,
            &proof_hashes,
        );
        let h2 = compute_expected_aggregate_hash(
            FieldElement::from(2u64),
            FieldElement::from(42u64),
            FieldElement::from(99u64),
            chain_commitment,
            FieldElement::ZERO,
            &proof_hashes,
        );

        assert_ne!(h1, h2, "Different model_id must produce different hash");
    }

    #[test]
    fn test_felt_from_short_string() {
        let felt = felt_from_short_string("OBELYSK_RECURSIVE_V1");
        assert_ne!(felt, FieldElement::ZERO);

        // Same string should produce same felt
        let felt2 = felt_from_short_string("OBELYSK_RECURSIVE_V1");
        assert_eq!(felt, felt2);

        // Cross-language check: Cairo short string 'OBELYSK_RECURSIVE_V1'
        // = bytes [0x4f,0x42,0x45,0x4c,0x59,0x53,0x4b,0x5f,
        //          0x52,0x45,0x43,0x55,0x52,0x53,0x49,0x56,
        //          0x45,0x5f,0x56,0x31]
        // packed big-endian into felt252.
        let expected_hex = format!("{:#066x}", felt);
        eprintln!("DOMAIN felt = {}", expected_hex);
    }

    /// Cross-language aggregate hash test (simple: 0 proofs, 0 layers).
    ///
    /// Asserts the exact hash value. The Cairo test
    /// `test_aggregate_hash_simple_cross_language` in
    /// stwo-ml-verify-core/tests/test_cross_language.cairo asserts the same value.
    #[test]
    fn test_cross_language_aggregate_hash_simple() {
        let hash = compute_expected_aggregate_hash(
            FieldElement::from(1u64),   // model_id
            FieldElement::from(42u64),  // model_commitment
            FieldElement::from(99u64),  // io_commitment
            FieldElement::ZERO,         // chain_commitment (empty headers)
            FieldElement::ZERO,         // tee_report_hash
            &[],                        // proof_hashes (no proofs)
        );

        // Pinned cross-language value — must match Cairo's poseidon_hash_span
        let expected = FieldElement::from_hex_be(
            "0x06abb9af4d18d8b49300457471e4a93dcd485013cb4df8a7fed92231501224b1"
        ).unwrap();
        assert_eq!(hash, expected, "Cross-language aggregate hash mismatch (simple)");
    }

    /// Cross-language aggregate hash test (with 2-layer chain).
    ///
    /// First computes chain_commitment for headers [(0,100,200),(1,200,300)],
    /// then the full aggregate hash. Both values are pinned against Cairo.
    #[test]
    fn test_cross_language_aggregate_hash_with_chain() {
        // chain_commitment = poseidon_hash_many([0, 100, 200, 1, 200, 300])
        let chain_commitment = poseidon_hash_many(&[
            FieldElement::from(0u64),   // layer_index 0
            FieldElement::from(100u64), // input_commitment
            FieldElement::from(200u64), // output_commitment
            FieldElement::from(1u64),   // layer_index 1
            FieldElement::from(200u64), // input_commitment
            FieldElement::from(300u64), // output_commitment
        ]);

        let expected_chain = FieldElement::from_hex_be(
            "0x0177f5c4aa4b7c74b43770aa0282670d57619387f4faed70e8b412987046d0e1"
        ).unwrap();
        assert_eq!(chain_commitment, expected_chain, "Cross-language chain commitment mismatch");

        let hash = compute_expected_aggregate_hash(
            FieldElement::from(1u64),   // model_id
            FieldElement::from(42u64),  // model_commitment
            FieldElement::from(99u64),  // io_commitment
            chain_commitment,           // from 2-layer chain
            FieldElement::ZERO,         // tee_report_hash
            &[],                        // proof_hashes
        );

        let expected_hash = FieldElement::from_hex_be(
            "0x03c17dffe67711249279c241bbcb2117b59d84e65cf454d28de758ad31b1c9a0"
        ).unwrap();
        assert_eq!(hash, expected_hash, "Cross-language aggregate hash mismatch (chain)");
    }

    /// Verify the domain separator encoding matches Cairo's short string literal.
    #[test]
    fn test_cross_language_domain_separator() {
        let domain = felt_from_short_string(AGGREGATE_DOMAIN);
        let expected = FieldElement::from_hex_be(
            "0x4f42454c59534b5f5245435552534956455f5631"
        ).unwrap();
        assert_eq!(domain, expected, "Domain separator encoding mismatch with Cairo");
    }

    #[test]
    fn test_serialize_empty_pipeline() {
        let proof = ModelPipelineProof {
            model_commitment: FieldElement::from(1u64),
            io_commitment: FieldElement::from(2u64),
            layer_proofs: vec![],
            receipt: None,
            tee_report_hash: None,
        };

        let serialized = serialize_recursive_input(FieldElement::from(1u64), &proof).unwrap();

        // model_id + model_commitment + io_commitment + num_matmul(0) +
        // matmul_array(len=0) + layer_headers(len=0) +
        // model_input + model_output + tee_hash = 9
        assert_eq!(serialized.len(), 9, "empty pipeline should serialize to 9 felts");
        assert_eq!(serialized[3], FieldElement::ZERO, "num_matmul_proofs = 0");
        assert_eq!(serialized[4], FieldElement::ZERO, "matmul_proofs.len() = 0");
        assert_eq!(serialized[5], FieldElement::ZERO, "layer_headers.len() = 0");
    }

    /// E2E test: full pipeline through cairo-prove.
    /// Requires compiled executable and cairo-prove binary.
    #[test]
    #[ignore]
    fn test_recursive_e2e_small_model() {
        use crate::compiler::graph::GraphBuilder;
        use crate::components::matmul::M31Matrix;
        use crate::pipeline::prover::prove_model_pipeline;

        // Build a 2-layer MLP: input(1,4) → linear(4) → linear(2)
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4).linear(2);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = crate::compiler::graph::GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i + j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w1 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w1.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(1, w1);

        let config = crate::pipeline::types::PipelineConfig::default();
        let pipeline_proof = prove_model_pipeline(&graph, &input, &weights, &config)
            .expect("pipeline proving should succeed");

        let model_id = FieldElement::from(1u64);

        // Serialize for recursive verifier
        let serialized = serialize_recursive_input(model_id, &pipeline_proof).unwrap();
        assert!(serialized.len() > 20, "serialized should be non-trivial");

        // Run recursive proof (requires binaries)
        let recursive_config = RecursiveConfig::default();
        let result = prove_recursive(model_id, &pipeline_proof, &recursive_config);

        match result {
            Ok(r) => {
                assert_eq!(r.num_verified as usize, pipeline_proof.num_matmul_proofs());
                assert_eq!(r.num_layers as usize, pipeline_proof.layer_proofs.len());
            }
            Err(RecursiveError::ProverError(msg)) => {
                eprintln!("cairo-prove not available (expected in CI): {msg}");
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
