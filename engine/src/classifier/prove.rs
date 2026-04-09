//! Prove pipeline: encode → classify → prove → extract score.
//!
//! This is the main entry point for the ZKML transaction classifier.
//! It takes raw transaction features, runs them through the classifier MLP,
//! generates a cryptographic proof of correct inference, and returns the
//! proven threat score + decision.

use starknet_ff::FieldElement;

use crate::policy::PolicyConfig;

use super::encoder::encode_transaction;
use super::model::ClassifierModel;
use super::types::*;

/// Evaluate a transaction through the ZKML classifier.
///
/// This is the end-to-end pipeline:
/// 1. Encode transaction features → `(1, 64)` M31 matrix
/// 2. Run classifier forward pass + GKR proof (with strict policy)
/// 3. Extract the 3 output scores (safe/suspicious/malicious)
/// 4. Compute threat score (0-100000) and decision
///
/// The returned [`ClassifierResult`] contains the proof, io_commitment,
/// policy_commitment, and decision — everything needed for on-chain submission.
pub fn evaluate_transaction(
    tx: &TransactionFeatures,
    model: &ClassifierModel,
    policy: &PolicyConfig,
) -> Result<ClassifierResult, ClassifierError> {
    let start = std::time::Instant::now();

    // 1. Encode transaction features to M31 input vector
    let input = encode_transaction(tx);

    // 2. Prove the classifier inference with GKR pipeline
    let proof = crate::aggregation::prove_model_pure_gkr_auto_with_cache(
        &model.graph,
        &input,
        &model.weights,
        None, // no weight cache (classifier is small, <1ms commitment)
        Some(policy),
    )
    .map_err(|e| ClassifierError::ProvingError(format!("{e}")))?;

    let prove_time_ms = start.elapsed().as_millis() as u64;

    // 3. Extract output scores from the proven execution trace
    let output = &proof.execution.output;
    if output.data.len() < NUM_CLASSES {
        return Err(ClassifierError::ProvingError(format!(
            "classifier output has {} values, expected at least {}",
            output.data.len(),
            NUM_CLASSES
        )));
    }

    let scores = [output.data[0].0, output.data[1].0, output.data[2].0];

    // 4. Compute threat score: (malicious / total) * 100000
    let total = scores[0] as u64 + scores[1] as u64 + scores[2] as u64;
    let threat_score = if total == 0 {
        50_000 // ambiguous → escalate
    } else {
        ((scores[2] as u64 * 100_000) / total) as u32
    };

    // 5. Derive decision from score
    let decision = Decision::from_score(threat_score);

    Ok(ClassifierResult {
        threat_score,
        decision,
        scores,
        io_commitment: proof.io_commitment,
        policy_commitment: proof.policy_commitment,
        proof_hash: FieldElement::ZERO, // computed on-chain from channel digest
        prove_time_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::model::build_test_classifier;

    fn sample_tx() -> TransactionFeatures {
        TransactionFeatures {
            target: FieldElement::from(0xDEADBEEFu64),
            value: [0, 1_000_000],
            selector: 0xa9059cbb,
            calldata_prefix: [0; 8],
            calldata_len: 68,
            agent_trust_score: 15000,
            agent_strikes: 0,
            agent_age_blocks: 1000,
            target_flags: TargetFlags::default(),
            value_features: ValueFeatures::default(),
            selector_features: SelectorFeatures {
                is_transfer: true,
                ..Default::default()
            },
            behavioral: BehavioralFeatures::default(),
        }
    }

    #[test]
    fn test_evaluate_transaction_produces_result() {
        let model = build_test_classifier();
        let policy = PolicyConfig::strict();
        let tx = sample_tx();

        let result = evaluate_transaction(&tx, &model, &policy).unwrap();

        // Score should be in valid range
        assert!(result.threat_score <= 100_000, "score out of range: {}", result.threat_score);

        // Decision should match score
        assert_eq!(result.decision, Decision::from_score(result.threat_score));

        // IO commitment should be non-zero
        assert_ne!(result.io_commitment, FieldElement::ZERO);

        // Policy commitment should match strict preset
        assert_eq!(result.policy_commitment, policy.policy_commitment());

        // Prove time should be reasonable (<60 seconds in debug, <1s in release)
        assert!(result.prove_time_ms < 60_000, "proving took {}ms", result.prove_time_ms);

        eprintln!("Classifier result:");
        eprintln!("  scores: {:?}", result.scores);
        eprintln!("  threat_score: {}", result.threat_score);
        eprintln!("  decision: {}", result.decision);
        eprintln!("  prove_time: {}ms", result.prove_time_ms);
        eprintln!("  io_commitment: {:#066x}", result.io_commitment);
        eprintln!("  policy_commitment: {:#066x}", result.policy_commitment);
    }

    #[test]
    fn test_evaluate_with_strict_policy_differs_from_standard() {
        let model = build_test_classifier();
        let tx = sample_tx();

        let strict_result = evaluate_transaction(&tx, &model, &PolicyConfig::strict()).unwrap();
        let standard_result = evaluate_transaction(&tx, &model, &PolicyConfig::standard()).unwrap();

        // Same model + same input = same scores (forward pass is deterministic)
        assert_eq!(strict_result.scores, standard_result.scores);
        assert_eq!(strict_result.threat_score, standard_result.threat_score);

        // But different policies = different policy commitments
        assert_ne!(strict_result.policy_commitment, standard_result.policy_commitment);

        // IO commitment is the same (computed from raw IO data, not Fiat-Shamir)
        // because the model output is identical regardless of policy.
        assert_eq!(strict_result.io_commitment, standard_result.io_commitment);
    }
}
