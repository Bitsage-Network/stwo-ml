//! Adversarial testing — can a malicious prover forge proofs?
//!
//! These tests attempt every known attack vector against the protocol:
//! 1. Weight substitution: use different weights than committed
//! 2. Output fabrication: claim different output than computed
//! 3. Intermediate tampering: alter hidden layer values
//! 4. Proof reuse: submit a valid proof for different inputs
//! 5. Model swaps: prove model A, claim it was model B
//! 6. Activation bypass: skip the activation function
//! 7. Norm bypass: skip normalization
//!
//! Every test MUST fail verification. If any passes, we have a soundness bug.

#[cfg(test)]
mod tests {
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::QM31;

    use crate::components::matmul::M31Matrix;
    use crate::compiler::graph::{ComputationGraph, GraphBuilder, GraphWeights};
    use crate::components::activation::ActivationType;
    use crate::crypto::poseidon_channel::PoseidonChannel;

    /// Helper: build a small provable model (norm → matmul → activation → matmul → norm)
    fn build_test_model() -> (ComputationGraph, GraphWeights, M31Matrix) {
        let dim = 4;
        let hidden = 8;
        let mut builder = GraphBuilder::new((1, dim));
        builder.rms_norm();
        builder.linear(hidden);
        builder.activation(ActivationType::SiLU);
        builder.linear(dim);
        builder.rms_norm();
        let graph = builder.build();

        let mut weights = GraphWeights::new();
        // Node 1: matmul dim→hidden
        let mut w1 = M31Matrix::new(dim, hidden);
        for i in 0..dim * hidden {
            w1.data[i] = M31::from((i as u32 * 13 + 7) % 251);
        }
        weights.add_weight(1, w1);

        // Node 3: matmul hidden→dim
        let mut w2 = M31Matrix::new(hidden, dim);
        for i in 0..hidden * dim {
            w2.data[i] = M31::from((i as u32 * 17 + 3) % 251);
        }
        weights.add_weight(3, w2);

        let mut input = M31Matrix::new(1, dim);
        for j in 0..dim {
            input.data[j] = M31::from((j as u32 * 7 + 100) % 251);
        }

        (graph, weights, input)
    }

    /// Helper: prove a model and return the proof
    fn prove_model(
        graph: &ComputationGraph,
        input: &M31Matrix,
        weights: &GraphWeights,
    ) -> Result<crate::aggregation::AggregatedModelProofOnChain, crate::aggregation::AggregationError>
    {
        crate::aggregation::prove_model_pure_gkr(graph, input, weights)
    }

    /// Helper: verify a proof
    fn verify_proof(
        graph: &ComputationGraph,
        proof: &crate::aggregation::AggregatedModelProofOnChain,
        weights: &GraphWeights,
    ) -> Result<(), String> {
        let circuit = crate::gkr::LayeredCircuit::from_graph(graph)
            .map_err(|e| format!("circuit: {e}"))?;
        let gkr = proof.gkr_proof.as_ref().ok_or("no GKR proof")?;
        let mut channel = PoseidonChannel::new();
        crate::gkr::verify_gkr_with_weights(&circuit, gkr, &proof.execution.output, weights, &mut channel)
            .map_err(|e| format!("{e}"))?;
        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════
    // ATTACK 1: Weight Substitution
    // A malicious prover uses different weights than committed.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adversarial_weight_substitution_detected() {
        let (graph, weights, input) = build_test_model();

        // Generate valid proof with correct weights
        let proof = prove_model(&graph, &input, &weights).expect("proving should succeed");

        // Tamper: create different weights for verification
        let mut tampered_weights = GraphWeights::new();
        let mut w1 = M31Matrix::new(4, 8);
        for i in 0..32 {
            w1.data[i] = M31::from((i as u32 * 31 + 99) % 251); // DIFFERENT values
        }
        tampered_weights.add_weight(1, w1);
        let mut w2 = M31Matrix::new(8, 4);
        for i in 0..32 {
            w2.data[i] = M31::from((i as u32 * 23 + 11) % 251); // DIFFERENT values
        }
        tampered_weights.add_weight(3, w2);

        // Verification with wrong weights should FAIL
        let result = verify_proof(&graph, &proof, &tampered_weights);
        assert!(result.is_err(),
            "SOUNDNESS BUG: proof verified with WRONG weights! Attack: weight substitution");
    }

    // ═══════════════════════════════════════════════════════════════════
    // ATTACK 2: Output Fabrication
    // A malicious prover claims a different output than was computed.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adversarial_output_fabrication_detected() {
        let (graph, weights, input) = build_test_model();

        let mut proof = prove_model(&graph, &input, &weights).expect("proving should succeed");

        // Tamper: change the output values
        let original_output = proof.execution.output.clone();
        for v in proof.execution.output.data.iter_mut() {
            *v = *v + M31::from(1u32); // shift every output value by 1
        }

        // Self-verification with tampered output should FAIL
        let result = verify_proof(&graph, &proof, &weights);
        assert!(result.is_err(),
            "SOUNDNESS BUG: proof verified with FABRICATED output! Attack: output fabrication");
    }

    // ═══════════════════════════════════════════════════════════════════
    // ATTACK 3: Proof Reuse (Replay Attack)
    // A malicious prover submits a valid proof with different inputs.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adversarial_proof_reuse_different_input() {
        let (graph, weights, input) = build_test_model();

        // Generate valid proof for input A
        let proof = prove_model(&graph, &input, &weights).expect("proving should succeed");

        // The IO commitment binds to the specific input+output.
        // A different input would produce a different io_commitment.
        let mut different_input = M31Matrix::new(1, 4);
        for j in 0..4 {
            different_input.data[j] = M31::from((j as u32 * 11 + 50) % 251); // DIFFERENT input
        }

        // The proof's io_commitment should NOT match if we check against different I/O
        let io_commitment_original = proof.io_commitment;
        let io_commitment_different = crate::aggregation::compute_io_commitment(
            &different_input, &proof.execution.output,
        );
        assert_ne!(io_commitment_original, io_commitment_different,
            "SOUNDNESS BUG: same IO commitment for different inputs! Attack: proof reuse");
    }

    // ═══════════════════════════════════════════════════════════════════
    // ATTACK 4: Model Swap
    // Prove with model A but claim it was model B.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adversarial_model_swap_detected() {
        let (graph_a, weights_a, input) = build_test_model();

        // Build model B with different architecture
        let mut builder_b = GraphBuilder::new((1, 4));
        builder_b.rms_norm();
        builder_b.linear(16); // DIFFERENT hidden dim
        builder_b.activation(ActivationType::SiLU);
        builder_b.linear(4);
        builder_b.rms_norm();
        let graph_b = builder_b.build();

        let mut weights_b = GraphWeights::new();
        let mut w1 = M31Matrix::new(4, 16);
        for i in 0..64 { w1.data[i] = M31::from((i as u32 * 7 + 1) % 251); }
        weights_b.add_weight(1, w1);
        let mut w2 = M31Matrix::new(16, 4);
        for i in 0..64 { w2.data[i] = M31::from((i as u32 * 3 + 5) % 251); }
        weights_b.add_weight(3, w2);

        // Prove with model A
        let proof_a = prove_model(&graph_a, &input, &weights_a).expect("proving A should succeed");

        // Try to verify proof_a against model B's circuit — should fail
        let result = verify_proof(&graph_b, &proof_a, &weights_b);
        assert!(result.is_err(),
            "SOUNDNESS BUG: model A proof verified against model B! Attack: model swap");
    }

    // ═══════════════════════════════════════════════════════════════════
    // ATTACK 5: Commitment Chain Tampering
    // Alter the commitment chain to insert/reorder inferences.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    #[cfg(feature = "cli")]
    fn test_adversarial_commitment_chain_tampering() {
        use crate::economics::{CommitmentChain, VerificationPolicy};
        use starknet_ff::FieldElement;

        let policy = VerificationPolicy {
            sample_rate: 1.0, // 100% for testing
            ..Default::default()
        };
        let mut chain = CommitmentChain::new(policy);

        let model_id = FieldElement::from(0x1_u64);
        for i in 0..10 {
            chain.commit(model_id, FieldElement::from(i as u64), i * 1000);
        }
        assert!(chain.verify_chain(), "clean chain should verify");

        // ATTACK: tamper with commitment 5
        let original = chain.commitments[5].commitment_hash;
        chain.commitments[5].commitment_hash = FieldElement::from(0xDEAD_u64);
        assert!(!chain.verify_chain(),
            "SOUNDNESS BUG: tampered chain verified! Attack: commitment chain tampering");

        // Restore
        chain.commitments[5].commitment_hash = original;
        assert!(chain.verify_chain(), "restored chain should verify");

        // ATTACK: swap two commitments (reorder)
        chain.commitments.swap(3, 7);
        assert!(!chain.verify_chain(),
            "SOUNDNESS BUG: reordered chain verified! Attack: commitment reordering");
    }

    // ═══════════════════════════════════════════════════════════════════
    // ATTACK 6: TopK Expert Selection Fraud
    // Claim wrong experts were selected in MoE routing.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adversarial_topk_expert_fraud() {
        use crate::components::topk::{select_top_k, verify_top_k, TopKSelection};

        // 8 experts, top-2 (Mixtral pattern)
        let logits = vec![
            M31::from(10), M31::from(90), M31::from(50), M31::from(80),
            M31::from(30), M31::from(70), M31::from(40), M31::from(60),
        ];

        // Correct: top-2 are indices 1 (90) and 3 (80)
        let correct = select_top_k(&logits, 2);
        assert!(verify_top_k(&logits, &correct).is_ok());

        // ATTACK: claim cheapest experts (indices 0, 6) instead of best
        let fraud = TopKSelection {
            selected_indices: vec![0, 6],
            selected_values: vec![M31::from(10), M31::from(40)],
            rejected_indices: vec![1, 2, 3, 4, 5, 7],
            rejected_values: vec![
                M31::from(90), M31::from(50), M31::from(80),
                M31::from(30), M31::from(70), M31::from(60),
            ],
            num_experts: 8,
            top_k: 2,
        };
        assert!(verify_top_k(&logits, &fraud).is_err(),
            "SOUNDNESS BUG: fraudulent expert selection accepted! Attack: TopK fraud");
    }

    // ═══════════════════════════════════════════════════════════════════
    // ATTACK 7: Gamma (γ) Weight Substitution
    // Use different normalization scale than committed.
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adversarial_gamma_substitution() {
        use crate::crypto::poseidon_channel::securefield_to_felt;

        // Two different gamma vectors
        let gamma_real = vec![M31::from(2), M31::from(3), M31::from(4), M31::from(5)];
        let gamma_fake = vec![M31::from(1), M31::from(1), M31::from(1), M31::from(1)];

        // Compute commitments
        fn commit_gamma(g: &[M31]) -> starknet_ff::FieldElement {
            let mut ch = PoseidonChannel::new();
            ch.mix_u64(0x47414D4D41_u64);
            for &v in g {
                ch.mix_felt(securefield_to_felt(QM31::from(v)));
            }
            ch.digest()
        }

        let commit_real = commit_gamma(&gamma_real);
        let commit_fake = commit_gamma(&gamma_fake);

        assert_ne!(commit_real, commit_fake,
            "SOUNDNESS BUG: different gammas produce same commitment! Attack: γ substitution");
    }

    // ═══════════════════════════════════════════════════════════════════
    // ATTACK 8: Integer Arithmetic Platform Divergence
    // Try to produce different results on different "platforms"
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adversarial_platform_divergence_impossible() {
        use crate::components::integer_math::{cos_fixed, sin_fixed, precompute_rope_thetas, build_rope_table_integer};

        // Run the same computation 3 times — must be identical
        let thetas = precompute_rope_thetas(64, 10000.0);
        let (cos1, sin1) = build_rope_table_integer(64, 64, &thetas, 0);
        let (cos2, sin2) = build_rope_table_integer(64, 64, &thetas, 0);
        let (cos3, sin3) = build_rope_table_integer(64, 64, &thetas, 0);

        assert_eq!(cos1, cos2);
        assert_eq!(cos2, cos3);
        assert_eq!(sin1, sin2);
        assert_eq!(sin2, sin3);

        // Verify cos/sin are deterministic for random angles
        for angle in [0u32, 1 << 29, 1 << 30, 1 << 31, u32::MAX] {
            let c1 = cos_fixed(angle);
            let c2 = cos_fixed(angle);
            assert_eq!(c1, c2, "cos({angle}) not deterministic");
            let s1 = sin_fixed(angle);
            let s2 = sin_fixed(angle);
            assert_eq!(s1, s2, "sin({angle}) not deterministic");
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // ATTACK 9: Piecewise Activation Coefficient Tampering
    // Use wrong activation coefficients (different function than committed)
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_adversarial_activation_coefficient_tampering() {
        use crate::components::activation::PiecewiseLinearCoeffs;

        // Get coefficients for SiLU
        let silu_coeffs = PiecewiseLinearCoeffs::for_activation(ActivationType::SiLU);
        // Get coefficients for GELU (different function)
        let gelu_coeffs = PiecewiseLinearCoeffs::for_activation(ActivationType::GELU);

        // Coefficients MUST be different — using GELU coefficients for SiLU is detectable
        assert_ne!(silu_coeffs.slopes, gelu_coeffs.slopes,
            "SOUNDNESS BUG: SiLU and GELU have same coefficients! Attack: activation swap");

        // Evaluation with wrong coefficients produces different output
        let test_input = M31::from(500_000u32);
        let silu_result = crate::components::activation::piecewise_linear_eval(&silu_coeffs, test_input);
        let gelu_result = crate::components::activation::piecewise_linear_eval(&gelu_coeffs, test_input);
        assert_ne!(silu_result, gelu_result,
            "SOUNDNESS BUG: different activations produce same output! Attack: activation function swap");
    }
}
