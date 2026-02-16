//! Aggregate semantic scoring for audit windows.
//!
//! Combines individual `InferenceEvaluation`s into a single
//! `AuditSemanticSummary` with score distributions and Merkle root.

use crate::audit::digest::{digest_to_hex, hash_felt_hex_m31};
use crate::audit::log::AuditMerkleTree;
use crate::audit::types::{AuditSemanticSummary, InferenceEvaluation};

/// Aggregate a set of per-inference evaluations into a summary.
///
/// Computes average score, score distribution, deterministic pass/fail counts,
/// and builds a Merkle root of `eval_io_commitment`s (if present).
pub fn aggregate_evaluations(
    evals: &[InferenceEvaluation],
    method: &str,
    evaluations_proved: bool,
) -> AuditSemanticSummary {
    if evals.is_empty() {
        return AuditSemanticSummary {
            method: method.to_string(),
            avg_quality_score: 0.0,
            excellent_count: 0,
            good_count: 0,
            fair_count: 0,
            poor_count: 0,
            deterministic_pass: 0,
            deterministic_fail: 0,
            evaluated_count: 0,
            evaluations_proved,
            eval_merkle_root: None,
            per_inference: Vec::new(),
        };
    }

    let mut total_score = 0.0f32;
    let mut scored_count = 0u32;
    let mut excellent = 0u32;
    let mut good = 0u32;
    let mut fair = 0u32;
    let mut poor = 0u32;
    let mut det_pass = 0u32;
    let mut det_fail = 0u32;
    let mut merkle_tree = AuditMerkleTree::new();

    for eval in evals {
        // Score distribution.
        if let Some(score) = eval.semantic_score {
            total_score += score;
            scored_count += 1;

            if score >= 0.9 {
                excellent += 1;
            } else if score >= 0.7 {
                good += 1;
            } else if score >= 0.5 {
                fair += 1;
            } else {
                poor += 1;
            }
        }

        // Deterministic check totals.
        for check in &eval.deterministic_checks {
            if check.passed {
                det_pass += 1;
            } else {
                det_fail += 1;
            }
        }

        // Merkle tree of eval commitments (hash felt252 hex → M31 leaf).
        if let Some(ref hex) = eval.eval_io_commitment {
            let leaf = hash_felt_hex_m31(hex);
            merkle_tree.push(leaf);
        }
    }

    let avg = if scored_count > 0 {
        total_score / scored_count as f32
    } else {
        0.0
    };

    let eval_merkle_root = if merkle_tree.len() > 0 {
        Some(digest_to_hex(&merkle_tree.root()))
    } else {
        None
    };

    AuditSemanticSummary {
        method: method.to_string(),
        avg_quality_score: avg,
        excellent_count: excellent,
        good_count: good,
        fair_count: fair,
        poor_count: poor,
        deterministic_pass: det_pass,
        deterministic_fail: det_fail,
        evaluated_count: evals.len() as u32,
        evaluations_proved,
        eval_merkle_root,
        per_inference: evals.to_vec(),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::types::DeterministicCheck;

    fn make_eval(seq: u64, score: Option<f32>, det_pass: bool) -> InferenceEvaluation {
        InferenceEvaluation {
            sequence: seq,
            deterministic_checks: vec![DeterministicCheck {
                check_type: "non_empty".to_string(),
                passed: det_pass,
                detail: None,
            }],
            semantic_score: score,
            eval_io_commitment: score.map(|_| format!("0x{:064x}", seq)),
            evaluation_proved: false,
        }
    }

    #[test]
    fn test_aggregate_empty() {
        let summary = aggregate_evaluations(&[], "combined", false);
        assert_eq!(summary.evaluated_count, 0);
        assert_eq!(summary.avg_quality_score, 0.0);
        assert!(summary.eval_merkle_root.is_none());
    }

    #[test]
    fn test_aggregate_score_distribution() {
        let evals = vec![
            make_eval(0, Some(0.95), true),  // excellent
            make_eval(1, Some(0.85), true),  // good
            make_eval(2, Some(0.75), true),  // good
            make_eval(3, Some(0.6), true),   // fair
            make_eval(4, Some(0.3), false),  // poor
        ];

        let summary = aggregate_evaluations(&evals, "self_eval", false);
        assert_eq!(summary.excellent_count, 1);
        assert_eq!(summary.good_count, 2);
        assert_eq!(summary.fair_count, 1);
        assert_eq!(summary.poor_count, 1);
        assert_eq!(summary.evaluated_count, 5);
        assert_eq!(summary.deterministic_pass, 4);
        assert_eq!(summary.deterministic_fail, 1);

        // Average: (0.95 + 0.85 + 0.75 + 0.6 + 0.3) / 5 = 0.69
        assert!((summary.avg_quality_score - 0.69).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_distribution_sums() {
        let evals: Vec<InferenceEvaluation> = (0..100)
            .map(|i| make_eval(i, Some(i as f32 / 100.0), true))
            .collect();

        let summary = aggregate_evaluations(&evals, "combined", false);
        let total = summary.excellent_count
            + summary.good_count
            + summary.fair_count
            + summary.poor_count;
        assert_eq!(total, 100);
    }

    #[test]
    fn test_aggregate_merkle_root_present() {
        let evals = vec![make_eval(0, Some(0.8), true), make_eval(1, Some(0.9), true)];
        let summary = aggregate_evaluations(&evals, "self_eval", false);
        assert!(summary.eval_merkle_root.is_some());
    }

    #[test]
    fn test_aggregate_no_scores() {
        let evals = vec![
            make_eval(0, None, true),
            make_eval(1, None, false),
        ];
        let summary = aggregate_evaluations(&evals, "deterministic", false);
        assert_eq!(summary.avg_quality_score, 0.0);
        assert_eq!(summary.excellent_count, 0);
        assert_eq!(summary.deterministic_pass, 1);
        assert_eq!(summary.deterministic_fail, 1);
    }
}
