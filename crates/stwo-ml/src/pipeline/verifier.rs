//! Pipeline verifier: validates commitment chain + per-layer proofs.
//!
//! Verifies a `ModelPipelineProof` without access to original data:
//! 1. Commitment chain continuity (Poseidon linking)
//! 2. MatMul sumcheck proofs (Poseidon Fiat-Shamir)
//! 3. Receipt chain and billing (if present)

use starknet_ff::FieldElement;

use crate::components::matmul::verify_matmul_sumcheck_onchain;
use crate::receipt::ComputeReceipt;

use super::types::{LayerProofKindOnChain, ModelPipelineProof};

/// Verification result.
#[derive(Debug)]
pub struct VerificationResult {
    /// Overall pass/fail.
    pub is_valid: bool,
    /// Number of matmul proofs verified.
    pub matmul_proofs_verified: usize,
    /// Number of activation proofs present (not yet independently verified here).
    pub activation_proofs_present: usize,
    /// Commitment chain valid.
    pub chain_valid: bool,
    /// Receipt valid (if present).
    pub receipt_valid: Option<bool>,
    /// Per-layer errors.
    pub errors: Vec<String>,
}

/// Verify a pipeline proof (local pre-flight check).
///
/// Validates:
/// - Commitment chain continuity: layer[i].output == layer[i+1].input
/// - All matmul sumcheck proofs (Poseidon Fiat-Shamir replay)
/// - Receipt validity (if present)
///
/// Does NOT verify activation STARKs (use stwo-cairo-verifier on-chain).
pub fn verify_pipeline_proof(proof: &ModelPipelineProof) -> VerificationResult {
    let mut result = VerificationResult {
        is_valid: true,
        matmul_proofs_verified: 0,
        activation_proofs_present: 0,
        chain_valid: true,
        receipt_valid: None,
        errors: Vec::new(),
    };

    // 1. Verify commitment chain
    if !proof.verify_commitment_chain() {
        result.is_valid = false;
        result.chain_valid = false;
        result.errors.push("Commitment chain broken".into());
    }

    // 2. Verify each matmul sumcheck proof
    for layer_proof in &proof.layer_proofs {
        match &layer_proof.kind {
            LayerProofKindOnChain::MatMulSumcheck(matmul_proof) => {
                match verify_matmul_sumcheck_onchain(matmul_proof) {
                    Ok(()) => {
                        result.matmul_proofs_verified += 1;
                    }
                    Err(e) => {
                        result.is_valid = false;
                        result.errors.push(format!(
                            "Layer {}: matmul sumcheck verification failed: {}",
                            layer_proof.layer_index, e
                        ));
                    }
                }
            }
            LayerProofKindOnChain::ActivationStark(_) => {
                // STARK verification requires the on-chain verifier (stwo-cairo-verifier).
                // Here we just count them.
                result.activation_proofs_present += 1;
            }
            LayerProofKindOnChain::Attention(attn_proof) => {
                // Verify each sub-matmul proof within the attention layer
                let sub_proofs = [
                    ("Q_proj", &attn_proof.q_proof),
                    ("K_proj", &attn_proof.k_proof),
                    ("V_proj", &attn_proof.v_proof),
                    ("output_proj", &attn_proof.output_proof),
                ];
                for (name, mp) in &sub_proofs {
                    match verify_matmul_sumcheck_onchain(mp) {
                        Ok(()) => result.matmul_proofs_verified += 1,
                        Err(e) => {
                            result.is_valid = false;
                            result.errors.push(format!(
                                "Layer {} attention {}: {}",
                                layer_proof.layer_index, name, e
                            ));
                        }
                    }
                }
                for (h, sp) in attn_proof.score_proofs.iter().enumerate() {
                    match verify_matmul_sumcheck_onchain(sp) {
                        Ok(()) => result.matmul_proofs_verified += 1,
                        Err(e) => {
                            result.is_valid = false;
                            result.errors.push(format!(
                                "Layer {} attention score_head_{}: {}",
                                layer_proof.layer_index, h, e
                            ));
                        }
                    }
                }
                for (h, avp) in attn_proof.attn_v_proofs.iter().enumerate() {
                    match verify_matmul_sumcheck_onchain(avp) {
                        Ok(()) => result.matmul_proofs_verified += 1,
                        Err(e) => {
                            result.is_valid = false;
                            result.errors.push(format!(
                                "Layer {} attention attn_v_head_{}: {}",
                                layer_proof.layer_index, h, e
                            ));
                        }
                    }
                }
            }
            LayerProofKindOnChain::Passthrough => {}
        }
    }

    // 3. Verify receipt if present
    if let Some(receipt) = &proof.receipt {
        let receipt_ok = verify_receipt(receipt, proof.model_commitment);
        result.receipt_valid = Some(receipt_ok);
        if !receipt_ok {
            result.is_valid = false;
            result.errors.push("Receipt verification failed".into());
        }
    }

    result
}

/// Verify a compute receipt's self-consistency.
fn verify_receipt(receipt: &ComputeReceipt, expected_model_commitment: FieldElement) -> bool {
    // Model commitment must match
    if receipt.model_commitment != expected_model_commitment {
        return false;
    }

    // Chain link for first receipt
    if receipt.sequence_number == 0 && receipt.prev_receipt_hash != FieldElement::ZERO {
        return false;
    }

    // Billing arithmetic
    receipt.verify_billing()
}

/// Verify a chain of receipts (e.g., multi-turn inference session).
pub fn verify_receipt_chain(receipts: &[ComputeReceipt]) -> bool {
    if receipts.is_empty() {
        return true;
    }

    // First receipt must have prev_hash == 0
    if !receipts[0].verify_chain_link(None) {
        return false;
    }

    // Each subsequent receipt links to the previous
    for i in 1..receipts.len() {
        let prev_hash = receipts[i - 1].receipt_hash();
        if !receipts[i].verify_chain_link(Some(prev_hash)) {
            return false;
        }
        if receipts[i].sequence_number != receipts[i - 1].sequence_number + 1 {
            return false;
        }
    }

    // All must have valid billing
    receipts.iter().all(|r| r.verify_billing())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::GraphBuilder;
    use crate::compiler::onnx::generate_weights_for_graph;
    use crate::components::activation::ActivationType;
    use crate::components::matmul::M31Matrix;
    use crate::pipeline::prover::prove_model_pipeline;
    use crate::pipeline::types::PipelineConfig;
    use stwo::core::fields::m31::M31;

    fn make_input_4x4() -> M31Matrix {
        let mut m = M31Matrix::new(4, 4);
        for i in 0..16 {
            m.data[i] = M31::from((i + 1) as u32);
        }
        m
    }

    #[test]
    fn test_verify_single_matmul() {
        let mut builder = GraphBuilder::new((4, 4));
        builder.linear(4);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 42);

        let config = PipelineConfig {
            onchain_matmul: true,
            prove_activations: false,
            generate_receipt: false,
            precomputed_model_commitment: None,
        };

        let proof = prove_model_pipeline(&graph, &make_input_4x4(), &weights, &config).unwrap();
        let result = verify_pipeline_proof(&proof);

        assert!(result.is_valid, "Errors: {:?}", result.errors);
        assert_eq!(result.matmul_proofs_verified, 1);
        assert!(result.chain_valid);
    }

    #[test]
    fn test_verify_two_layer_mlp() {
        let mut builder = GraphBuilder::new((4, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(4)
            .activation(ActivationType::ReLU);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 99);

        let config = PipelineConfig::default();
        let proof = prove_model_pipeline(&graph, &make_input_4x4(), &weights, &config).unwrap();
        let result = verify_pipeline_proof(&proof);

        assert!(result.is_valid, "Errors: {:?}", result.errors);
        assert_eq!(result.matmul_proofs_verified, 2);
        assert_eq!(result.activation_proofs_present, 2);
        assert!(result.chain_valid);
    }

    #[test]
    fn test_verify_with_receipt() {
        let mut builder = GraphBuilder::new((4, 4));
        builder.linear(4);
        let graph = builder.build();
        let weights = generate_weights_for_graph(&graph, 42);

        let config = PipelineConfig {
            onchain_matmul: true,
            prove_activations: false,
            generate_receipt: true,
            precomputed_model_commitment: None,
        };

        let proof = prove_model_pipeline(&graph, &make_input_4x4(), &weights, &config).unwrap();
        let result = verify_pipeline_proof(&proof);

        assert!(result.is_valid, "Errors: {:?}", result.errors);
        assert_eq!(result.receipt_valid, Some(true));
    }

    #[test]
    fn test_receipt_chain_verification() {
        let r0 = ComputeReceipt {
            job_id: FieldElement::from(1u64),
            worker_pubkey: FieldElement::from(0u64),
            input_commitment: FieldElement::from(10u64),
            output_commitment: FieldElement::from(20u64),
            model_commitment: FieldElement::from(99u64),
            prev_receipt_hash: FieldElement::ZERO,
            gpu_time_ms: 1000,
            token_count: 10,
            peak_memory_mb: 512,
            billing_amount_sage: 120, // 1000 * 100 / 1000 + 10 * 2 = 100 + 20 = 120
            billing_rate_per_sec: 100,
            billing_rate_per_token: 2,
            tee_report_hash: FieldElement::ZERO,
            tee_timestamp: 1000,
            timestamp: 1000,
            sequence_number: 0,
        };

        let r1 = ComputeReceipt {
            job_id: FieldElement::from(2u64),
            worker_pubkey: FieldElement::from(0u64),
            input_commitment: FieldElement::from(30u64),
            output_commitment: FieldElement::from(40u64),
            model_commitment: FieldElement::from(99u64),
            prev_receipt_hash: r0.receipt_hash(),
            gpu_time_ms: 2000,
            token_count: 20,
            peak_memory_mb: 512,
            billing_amount_sage: 240, // 2000 * 100 / 1000 + 20 * 2 = 200 + 40 = 240
            billing_rate_per_sec: 100,
            billing_rate_per_token: 2,
            tee_report_hash: FieldElement::ZERO,
            tee_timestamp: 2000,
            timestamp: 2000,
            sequence_number: 1,
        };

        assert!(verify_receipt_chain(&[r0, r1]));
    }

    #[test]
    fn test_broken_receipt_chain() {
        let r0 = ComputeReceipt {
            job_id: FieldElement::from(1u64),
            worker_pubkey: FieldElement::from(0u64),
            input_commitment: FieldElement::from(10u64),
            output_commitment: FieldElement::from(20u64),
            model_commitment: FieldElement::from(99u64),
            prev_receipt_hash: FieldElement::ZERO,
            gpu_time_ms: 1000,
            token_count: 10,
            peak_memory_mb: 512,
            billing_amount_sage: 120,
            billing_rate_per_sec: 100,
            billing_rate_per_token: 2,
            tee_report_hash: FieldElement::ZERO,
            tee_timestamp: 1000,
            timestamp: 1000,
            sequence_number: 0,
        };

        let r1_bad = ComputeReceipt {
            prev_receipt_hash: FieldElement::from(999u64), // Wrong!
            sequence_number: 1,
            ..r0.clone()
        };

        assert!(!verify_receipt_chain(&[r0, r1_bad]));
    }
}
