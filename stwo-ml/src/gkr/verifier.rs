//! GKR verifier: replays Fiat-Shamir transcript and checks layer proofs.
//!
//! The verifier walks the circuit from output → input, replaying the exact
//! same PoseidonChannel operations as the prover. At each layer it checks
//! that the proof is consistent with the claimed reduction.

use num_traits::{One, Zero};
use stwo::core::fields::m31::M31;
use stwo::core::fields::FieldExpOps;

use crate::compiler::graph::{GraphExecution, GraphWeights};
use crate::components::activation::ActivationType;
use crate::components::attention::MultiHeadAttentionConfig;
use crate::components::matmul::{
    evaluate_mle_pub as evaluate_mle,
    matrix_to_mle_col_major_padded_pub as matrix_to_mle_col_major_padded,
    matrix_to_mle_pub as matrix_to_mle, pad_matrix_pow2, M31Matrix, RoundPoly,
};
use crate::crypto::aggregated_opening::{
    verify_aggregated_binding, AggregatedWeightClaim,
};
use crate::crypto::poseidon_channel::PoseidonChannel;
use crate::gadgets::lookup_table::PrecomputedTable;

use super::circuit::{LayerType, LayeredCircuit};
use super::prover::compute_rsqrt_table_commitment;
use super::types::{
    derive_weight_opening_subchannel, EmbeddingLogUpProof, GKRClaim, GKRError, GKRProof,
    LayerProof, LogUpProof, RoundPolyDeg3, SecureField, WeightOpeningTranscriptMode,
};

/// Verify a GKR proof against the model circuit and execution trace.
///
/// The verifier has access to:
/// - The circuit structure (public)
/// - The model output (public)
/// - The weight matrices (for checking weight commitments)
/// - The GKR proof
///
/// It does NOT need the intermediate values — only the prover needs those.
/// However, for the Rust-side pre-flight check we verify against the full
/// execution for confidence before on-chain submission.
pub fn verify_gkr(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    output: &M31Matrix,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    verify_gkr_inner(circuit, proof, output, None, channel)
}

/// Verify a GKR proof with access to model weights.
///
/// Needed for weight-binding transcript modes that do not carry per-weight
/// Merkle openings inside the proof payload.
pub fn verify_gkr_with_weights(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    output: &M31Matrix,
    weights: &GraphWeights,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    verify_gkr_inner(circuit, proof, output, Some(weights), channel)
}

fn verify_gkr_inner(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    output: &M31Matrix,
    weights: Option<&GraphWeights>,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let d = circuit.layers.len();

    if proof.layer_proofs.len() > d {
        return Err(GKRError::VerificationError {
            layer_idx: 0,
            reason: format!(
                "proof has {} layer proofs but circuit has {} layers",
                proof.layer_proofs.len(),
                d
            ),
        });
    }

    // Seed channel identically to prover
    channel.mix_u64(d as u64);
    channel.mix_u64(circuit.input_shape.0 as u64);
    channel.mix_u64(circuit.input_shape.1 as u64);

    // Reconstruct output claim
    let output_padded = pad_matrix_pow2(output);
    let output_mle = matrix_to_mle(&output_padded);

    let log_out_rows = output_padded.rows.ilog2() as usize;
    let log_out_cols = output_padded.cols.ilog2() as usize;

    let r_out = channel.draw_qm31s(log_out_rows + log_out_cols);
    let output_value = evaluate_mle(&output_mle, &r_out);

    mix_secure_field(channel, output_value);

    let mut current_claim = GKRClaim {
        point: r_out,
        value: output_value,
    };

    // Walk layers from output → input, verifying each proof
    let mut proof_idx = 0;
    let mut expected_weight_node_ids = Vec::new();

    for layer_idx in (0..d).rev() {
        let layer = &circuit.layers[layer_idx];

        match &layer.layer_type {
            LayerType::Identity => continue,
            LayerType::Input => break,
            _ => {}
        }

        if proof_idx >= proof.layer_proofs.len() {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: "ran out of layer proofs".to_string(),
            });
        }

        let layer_proof = &proof.layer_proofs[proof_idx];
        proof_idx += 1;

        current_claim = match (&layer.layer_type, layer_proof) {
            (
                LayerType::MatMul {
                    m,
                    k,
                    n,
                    weight_node_id,
                },
                LayerProof::MatMul {
                    round_polys,
                    final_a_eval,
                    final_b_eval,
                },
            ) => {
                expected_weight_node_ids.push(*weight_node_id);
                verify_matmul_reduction(
                    &current_claim,
                    round_polys,
                    *final_a_eval,
                    *final_b_eval,
                    *m,
                    *k,
                    *n,
                    layer_idx,
                    channel,
                )?
            }

            (
                LayerType::Add { .. },
                LayerProof::Add {
                    lhs_eval, rhs_eval, ..
                },
            ) => verify_add_reduction(
                &current_claim,
                *lhs_eval,
                *rhs_eval,
                &layer.input_layers,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Mul { .. },
                LayerProof::Mul {
                    eq_round_polys,
                    lhs_eval,
                    rhs_eval,
                },
            ) => verify_mul_reduction(
                &current_claim,
                eq_round_polys,
                *lhs_eval,
                *rhs_eval,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Activation {
                    activation_type: circuit_act_type,
                    ..
                },
                LayerProof::Activation {
                    activation_type: proof_act_type,
                    logup_proof,
                    input_eval,
                    output_eval,
                    table_commitment,
                },
            ) => {
                // Verify activation type in proof matches circuit
                if circuit_act_type != proof_act_type {
                    return Err(GKRError::VerificationError {
                        layer_idx,
                        reason: format!(
                            "activation type mismatch: circuit={:?}, proof={:?}",
                            circuit_act_type, proof_act_type,
                        ),
                    });
                }

                verify_activation_reduction(
                    &current_claim,
                    *proof_act_type,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    layer_idx,
                    channel,
                )?
            }

            (
                LayerType::LayerNorm { dim, .. },
                LayerProof::LayerNorm {
                    logup_proof,
                    linear_round_polys,
                    linear_final_evals,
                    input_eval,
                    output_eval,
                    mean,
                    rsqrt_var,
                    rsqrt_table_commitment,
                    simd_combined,
                },
            ) => verify_layernorm_reduction(
                &current_claim,
                logup_proof.as_ref(),
                linear_round_polys,
                *linear_final_evals,
                *input_eval,
                *output_eval,
                *mean,
                *rsqrt_var,
                *rsqrt_table_commitment,
                *simd_combined,
                *dim,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Dequantize { params, .. },
                LayerProof::Dequantize {
                    logup_proof,
                    input_eval,
                    output_eval,
                    table_commitment,
                },
            ) => verify_dequantize_reduction(
                &current_claim,
                params,
                logup_proof.as_ref(),
                *input_eval,
                *output_eval,
                *table_commitment,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Quantize { params, .. },
                LayerProof::Quantize {
                    logup_proof,
                    input_eval,
                    output_eval,
                    table_inputs,
                    table_outputs,
                },
            ) => verify_quantize_reduction(
                &current_claim,
                params,
                logup_proof.as_ref(),
                *input_eval,
                *output_eval,
                table_inputs,
                table_outputs,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Embedding {
                    vocab_size,
                    embed_dim,
                },
                LayerProof::Embedding {
                    logup_proof,
                    input_eval,
                    output_eval,
                    input_num_vars,
                },
            ) => verify_embedding_reduction(
                &current_claim,
                *vocab_size,
                *embed_dim,
                logup_proof.as_ref(),
                *input_eval,
                *output_eval,
                *input_num_vars,
                layer.node_id,
                layer_idx,
                channel,
                weights,
            )?,

            (
                LayerType::RMSNorm { dim, .. },
                LayerProof::RMSNorm {
                    logup_proof,
                    linear_round_polys,
                    linear_final_evals,
                    input_eval,
                    output_eval,
                    rms_sq_eval,
                    rsqrt_eval,
                    rsqrt_table_commitment,
                    simd_combined,
                },
            ) => verify_rmsnorm_reduction(
                &current_claim,
                logup_proof.as_ref(),
                linear_round_polys,
                *linear_final_evals,
                *input_eval,
                *output_eval,
                *rms_sq_eval,
                *rsqrt_eval,
                *rsqrt_table_commitment,
                *simd_combined,
                *dim,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Attention { config },
                LayerProof::Attention {
                    sub_proofs,
                    sub_claim_values,
                },
            ) => {
                verify_attention_reduction(
                    &current_claim,
                    config,
                    sub_proofs,
                    sub_claim_values,
                    None, // no SIMD in standard verify_gkr
                    layer_idx,
                    channel,
                )?
            }

            (layer_type, layer_proof) => {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "layer type {:?} does not match proof type {:?}",
                        std::mem::discriminant(layer_type),
                        std::mem::discriminant(layer_proof),
                    ),
                });
            }
        };
    }

    // Verify final claim matches proof's input_claim
    if current_claim.point != proof.input_claim.point
        || current_claim.value != proof.input_claim.value
    {
        return Err(GKRError::VerificationError {
            layer_idx: 0,
            reason: "final input claim does not match proof".to_string(),
        });
    }

    let batched_rlc_mode =
        proof.weight_opening_transcript_mode == WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1;

    // Verify deferred proofs for skip branches of DAG Add layers.
    // Fiat-Shamir order: walk → deferred proofs → weight openings.
    // Each deferred proof contains a matmul sumcheck + weight opening that
    // binds the skip_eval from an Add reduction to actual weights and input.
    for (i, deferred) in proof.deferred_proofs.iter().enumerate() {
        // Mix deferred claim into channel (must match prover)
        mix_secure_field(channel, deferred.claim.value);

        // Verify the deferred matmul sumcheck
        let (m, k, n) = deferred.dims;
        match &deferred.layer_proof {
            LayerProof::MatMul {
                round_polys,
                final_a_eval,
                final_b_eval,
            } => {
                let deferred_input_claim = verify_matmul_reduction(
                    &deferred.claim,
                    round_polys,
                    *final_a_eval,
                    *final_b_eval,
                    m,
                    k,
                    n,
                    0, // layer_idx (deferred)
                    channel,
                )?;

                // Verify the deferred input claim matches
                if deferred_input_claim.point != deferred.input_claim.point
                    || deferred_input_claim.value != deferred.input_claim.value
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!("deferred proof {} input claim mismatch", i,),
                    });
                }
            }
            _ => {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!("deferred proof {} has non-matmul layer proof type", i,),
                });
            }
        }

        if batched_rlc_mode {
            // In batched RLC mode, deferred weight binding is checked in the
            // aggregated verifier pass (no per-deferred Merkle opening).
            if deferred.weight_commitment != starknet_ff::FieldElement::ZERO {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "deferred proof {} expects zero weight_commitment in batched RLC mode",
                        i
                    ),
                });
            }
            if !deferred.weight_opening.intermediate_roots.is_empty()
                || !deferred.weight_opening.queries.is_empty()
            {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "deferred proof {} expects empty weight_opening in batched RLC mode",
                        i
                    ),
                });
            }
        } else {
            // Verify deferred weight opening
            if deferred.weight_opening.final_value != deferred.weight_claim.expected_value {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!("deferred proof {} weight opening final_value mismatch", i,),
                });
            }
            if !crate::crypto::mle_opening::verify_mle_opening(
                deferred.weight_commitment,
                &deferred.weight_opening,
                &deferred.weight_claim.eval_point,
                channel,
            ) {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "deferred proof {} weight MLE opening failed verification",
                        i,
                    ),
                });
            }
        }
    }

    // Verify weight MLE opening proofs (post-deferred channel state).
    // Each opening proves: MLE(weight_matrix, eval_point) == expected_value
    // bound to the committed Poseidon Merkle root.
    if proof.weight_claims.len() != expected_weight_node_ids.len() {
        return Err(GKRError::VerificationError {
            layer_idx: 0,
            reason: format!(
                "weight_claims count ({}) != matmul layers in circuit walk ({})",
                proof.weight_claims.len(),
                expected_weight_node_ids.len()
            ),
        });
    }
    for (i, (claim, expected_node_id)) in proof
        .weight_claims
        .iter()
        .zip(expected_weight_node_ids.iter())
        .enumerate()
    {
        if claim.weight_node_id != *expected_node_id {
            return Err(GKRError::VerificationError {
                layer_idx: 0,
                reason: format!(
                    "weight claim {} node mismatch: claim={}, expected={}",
                    i, claim.weight_node_id, expected_node_id
                ),
            });
        }
    }

    match proof.weight_opening_transcript_mode {
        WeightOpeningTranscriptMode::Sequential => {
            if proof.weight_openings.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "weight_openings count ({}) != weight_commitments count ({})",
                        proof.weight_openings.len(),
                        proof.weight_commitments.len(),
                    ),
                });
            }
            if proof.weight_claims.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "weight_claims count ({}) != weight_commitments count ({})",
                        proof.weight_claims.len(),
                        proof.weight_commitments.len(),
                    ),
                });
            }
            for (i, ((opening, commitment), claim)) in proof
                .weight_openings
                .iter()
                .zip(proof.weight_commitments.iter())
                .zip(proof.weight_claims.iter())
                .enumerate()
            {
                if opening.final_value != claim.expected_value {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "weight opening {} final_value mismatch: opening={:?}, claim={:?}",
                            i, opening.final_value, claim.expected_value,
                        ),
                    });
                }

                if !crate::crypto::mle_opening::verify_mle_opening(
                    *commitment,
                    opening,
                    &claim.eval_point,
                    channel,
                ) {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!("weight MLE opening proof {} failed verification", i),
                    });
                }
            }
        }
        WeightOpeningTranscriptMode::BatchedSubchannelV1 => {
            if proof.weight_openings.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "weight_openings count ({}) != weight_commitments count ({})",
                        proof.weight_openings.len(),
                        proof.weight_commitments.len(),
                    ),
                });
            }
            if proof.weight_claims.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "weight_claims count ({}) != weight_commitments count ({})",
                        proof.weight_claims.len(),
                        proof.weight_commitments.len(),
                    ),
                });
            }
            let opening_seed = if proof.weight_openings.is_empty() {
                None
            } else {
                Some(channel.draw_felt252())
            };

            for (i, ((opening, commitment), claim)) in proof
                .weight_openings
                .iter()
                .zip(proof.weight_commitments.iter())
                .zip(proof.weight_claims.iter())
                .enumerate()
            {
                if opening.final_value != claim.expected_value {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "weight opening {} final_value mismatch: opening={:?}, claim={:?}",
                            i, opening.final_value, claim.expected_value,
                        ),
                    });
                }

                let mut sub_channel = derive_weight_opening_subchannel(
                    opening_seed.expect("seed exists when openings are non-empty"),
                    i,
                    claim,
                );
                if !crate::crypto::mle_opening::verify_mle_opening(
                    *commitment,
                    opening,
                    &claim.eval_point,
                    &mut sub_channel,
                ) {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "batched weight MLE opening proof {} failed verification",
                            i
                        ),
                    });
                }
            }
        }
        WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1 => {
            if !proof.weight_openings.is_empty() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "batched RLC mode expects no weight openings, got {}",
                        proof.weight_openings.len()
                    ),
                });
            }
            if !proof.weight_commitments.is_empty() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "batched RLC mode expects no weight commitments, got {}",
                        proof.weight_commitments.len()
                    ),
                });
            }

            let weights = weights.ok_or_else(|| GKRError::VerificationError {
                layer_idx: 0,
                reason: "batched RLC mode requires verify_gkr_with_weights(...)".to_string(),
            })?;
            let rho = channel.draw_qm31();
            let mut rho_pow = SecureField::one();
            let mut combined_expected = SecureField::zero();
            let mut combined_actual = SecureField::zero();

            for (i, deferred) in proof.deferred_proofs.iter().enumerate() {
                let claim = &deferred.weight_claim;
                let weight =
                    weights
                        .get_weight(claim.weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: claim.weight_node_id,
                        })?;
                let actual = evaluate_weight_claim_against_matrix(weight, &claim.eval_point)
                    .map_err(|reason| GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "batched RLC deferred weight claim {} evaluation failed: {}",
                            i, reason
                        ),
                    })?;
                combined_expected = combined_expected + rho_pow * claim.expected_value;
                combined_actual = combined_actual + rho_pow * actual;
                rho_pow = rho_pow * rho;
            }

            for (i, claim) in proof.weight_claims.iter().enumerate() {
                let weight =
                    weights
                        .get_weight(claim.weight_node_id)
                        .ok_or(GKRError::MissingWeight {
                            node_id: claim.weight_node_id,
                        })?;
                let actual = evaluate_weight_claim_against_matrix(weight, &claim.eval_point)
                    .map_err(|reason| GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "batched RLC weight claim {} evaluation failed: {}",
                            i, reason
                        ),
                    })?;
                combined_expected = combined_expected + rho_pow * claim.expected_value;
                combined_actual = combined_actual + rho_pow * actual;
                rho_pow = rho_pow * rho;
            }

            mix_secure_field(channel, combined_expected);
            if combined_expected != combined_actual {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "batched RLC weight-binding mismatch: expected {:?}, actual {:?}",
                        combined_expected, combined_actual
                    ),
                });
            }
        }
        WeightOpeningTranscriptMode::AggregatedTrustlessV2
        | WeightOpeningTranscriptMode::AggregatedOpeningsV4Experimental => {
            if proof.weight_openings.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "weight_openings count ({}) != weight_commitments count ({})",
                        proof.weight_openings.len(),
                        proof.weight_commitments.len(),
                    ),
                });
            }
            if proof.weight_claims.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "weight_claims count ({}) != weight_commitments count ({})",
                        proof.weight_claims.len(),
                        proof.weight_commitments.len(),
                    ),
                });
            }
            let opening_seed = if proof.weight_openings.is_empty() {
                None
            } else {
                Some(channel.draw_felt252())
            };
            for (i, ((opening, commitment), claim)) in proof
                .weight_openings
                .iter()
                .zip(proof.weight_commitments.iter())
                .zip(proof.weight_claims.iter())
                .enumerate()
            {
                if opening.final_value != claim.expected_value {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "weight opening {} final_value mismatch: opening={:?}, claim={:?}",
                            i, opening.final_value, claim.expected_value,
                        ),
                    });
                }

                let mut sub_channel = derive_weight_opening_subchannel(
                    opening_seed.expect("seed exists when openings are non-empty"),
                    i,
                    claim,
                );
                if !crate::crypto::mle_opening::verify_mle_opening(
                    *commitment,
                    opening,
                    &claim.eval_point,
                    &mut sub_channel,
                ) {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "aggregated trustless mode weight opening {} failed verification (sub-channel transcript)",
                            i
                        ),
                    });
                }
            }
        }

        WeightOpeningTranscriptMode::AggregatedOracleSumcheck => {
            let binding_proof = proof.aggregated_binding.as_ref().ok_or_else(|| {
                GKRError::VerificationError {
                    layer_idx: 0,
                    reason: "AggregatedOracleSumcheck mode requires aggregated_binding proof"
                        .to_string(),
                }
            })?;

            // Reconstruct AggregatedWeightClaim structs for verification.
            // Main walk claims use proof.weight_commitments; deferred claims
            // use deferred_proofs[i].weight_commitment.
            let mut agg_claims = Vec::new();
            for (idx, (claim, commitment)) in proof
                .weight_claims
                .iter()
                .zip(proof.weight_commitments.iter())
                .enumerate()
            {
                agg_claims.push(AggregatedWeightClaim {
                    matrix_index: idx,
                    local_n_vars: claim.eval_point.len(),
                    eval_point: claim.eval_point.clone(),
                    expected_value: claim.expected_value,
                    commitment: *commitment,
                });
            }
            for (deferred_idx, deferred) in proof.deferred_proofs.iter().enumerate() {
                let claim = &deferred.weight_claim;
                let claim_idx = proof.weight_claims.len() + deferred_idx;
                agg_claims.push(AggregatedWeightClaim {
                    matrix_index: claim_idx,
                    local_n_vars: claim.eval_point.len(),
                    eval_point: claim.eval_point.clone(),
                    expected_value: claim.expected_value,
                    commitment: deferred.weight_commitment,
                });
            }

            if agg_claims.is_empty() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: "AggregatedOracleSumcheck mode with zero claims".to_string(),
                });
            }

            if !verify_aggregated_binding(binding_proof, &agg_claims, channel) {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: "aggregated oracle sumcheck weight binding verification failed"
                        .to_string(),
                });
            }
        }
    }

    Ok(current_claim)
}

fn evaluate_weight_claim_against_matrix(
    weight: &M31Matrix,
    eval_point: &[SecureField],
) -> Result<SecureField, String> {
    let b_mle = matrix_to_mle_col_major_padded(weight);
    let expected_vars = b_mle.len().ilog2() as usize;
    if eval_point.len() != expected_vars {
        return Err(format!(
            "eval point var count mismatch: got {}, expected {}",
            eval_point.len(),
            expected_vars
        ));
    }
    Ok(evaluate_mle(&b_mle, eval_point))
}

/// Verify a GKR proof with full execution trace (pre-flight check).
///
/// This is more thorough than `verify_gkr` — it also checks that the
/// final input claim matches the actual model input MLE evaluation.
pub fn verify_gkr_with_execution(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    execution: &GraphExecution,
    channel: &mut PoseidonChannel,
) -> Result<(), GKRError> {
    // First run the standard verification — returns the verified input claim
    let input_claim = verify_gkr(circuit, proof, &execution.output, channel)?;

    // Then verify the input claim against the actual input
    if let Some((_, input_matrix)) = execution.intermediates.first() {
        let input_padded = pad_matrix_pow2(input_matrix);
        let input_mle = matrix_to_mle(&input_padded);

        let num_vars = input_mle.len().ilog2() as usize;
        if input_claim.point.len() != num_vars {
            return Err(GKRError::VerificationError {
                layer_idx: 0,
                reason: format!(
                    "input claim point has {} vars, expected {} (MLE has {} entries)",
                    input_claim.point.len(),
                    num_vars,
                    input_mle.len(),
                ),
            });
        }
        let expected = evaluate_mle(&input_mle, &input_claim.point);
        if expected != input_claim.value {
            return Err(GKRError::VerificationError {
                layer_idx: 0,
                reason: format!(
                    "input claim value {} != actual input MLE eval {}",
                    input_claim.value, expected
                ),
            });
        }
    }

    Ok(())
}

/// Verify a GKR proof produced by `prove_gkr_simd_gpu`.
///
/// Mirrors the SIMD prover's channel state: seeds with circuit + SIMD metadata,
/// draws r_simd challenges, and walks the template layers. Attention layers
/// receive `r_simd` for dual-operand sub-proof verification.
pub fn verify_gkr_simd(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    combined_output: &[SecureField],
    channel: &mut PoseidonChannel,
) -> Result<(), GKRError> {
    let simd_config = circuit.simd_config.as_ref().ok_or_else(|| {
        GKRError::SimdError("circuit has no SIMD config — need >= 2 identical blocks".into())
    })?;

    let d = circuit.layers.len();

    // Seed channel identically to prover
    channel.mix_u64(d as u64);
    channel.mix_u64(circuit.input_shape.0 as u64);
    channel.mix_u64(circuit.input_shape.1 as u64);
    channel.mix_u64(simd_config.num_blocks as u64);

    // Draw SIMD block-selection challenges
    let r_simd = channel.draw_qm31s(simd_config.simd_log_size);

    // Reconstruct output claim from combined output MLE
    let n_out = combined_output.len();
    let log_out = n_out.ilog2() as usize;
    let r_out = channel.draw_qm31s(log_out);
    let output_value = crate::components::matmul::evaluate_mle_pub(combined_output, &r_out);
    mix_secure_field(channel, output_value);

    let mut current_claim = GKRClaim {
        point: r_out,
        value: output_value,
    };

    // Walk layers from output → input (template block structure)
    let template = &simd_config.template_range;
    let template_layers: Vec<usize> = (template.start..template.end).rev().collect();
    let mut proof_idx = 0;

    for &layer_idx in &template_layers {
        let layer = &circuit.layers[layer_idx];

        match &layer.layer_type {
            LayerType::Identity => continue,
            LayerType::Input => break,
            _ => {}
        }

        if proof_idx >= proof.layer_proofs.len() {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: "ran out of layer proofs in SIMD verify".to_string(),
            });
        }

        let layer_proof = &proof.layer_proofs[proof_idx];
        proof_idx += 1;

        current_claim = match (&layer.layer_type, layer_proof) {
            (
                LayerType::MatMul { m, k, n, .. },
                LayerProof::MatMul {
                    round_polys,
                    final_a_eval,
                    final_b_eval,
                },
            ) => verify_matmul_reduction(
                &current_claim,
                round_polys,
                *final_a_eval,
                *final_b_eval,
                *m,
                *k,
                *n,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Add { .. },
                LayerProof::Add {
                    lhs_eval, rhs_eval, ..
                },
            ) => verify_add_reduction(
                &current_claim,
                *lhs_eval,
                *rhs_eval,
                &layer.input_layers,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Mul { .. },
                LayerProof::Mul {
                    eq_round_polys,
                    lhs_eval,
                    rhs_eval,
                },
            ) => verify_mul_reduction(
                &current_claim,
                eq_round_polys,
                *lhs_eval,
                *rhs_eval,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Activation {
                    activation_type: circuit_act_type,
                    ..
                },
                LayerProof::Activation {
                    activation_type: proof_act_type,
                    logup_proof,
                    input_eval,
                    output_eval,
                    table_commitment,
                },
            ) => {
                if circuit_act_type != proof_act_type {
                    return Err(GKRError::VerificationError {
                        layer_idx,
                        reason: format!(
                            "activation type mismatch: circuit={:?}, proof={:?}",
                            circuit_act_type, proof_act_type,
                        ),
                    });
                }
                verify_activation_reduction(
                    &current_claim,
                    *proof_act_type,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    layer_idx,
                    channel,
                )?
            }

            (
                LayerType::LayerNorm { dim, .. },
                LayerProof::LayerNorm {
                    logup_proof,
                    linear_round_polys,
                    linear_final_evals,
                    input_eval,
                    output_eval,
                    mean,
                    rsqrt_var,
                    rsqrt_table_commitment,
                    simd_combined,
                },
            ) => verify_layernorm_reduction(
                &current_claim,
                logup_proof.as_ref(),
                linear_round_polys,
                *linear_final_evals,
                *input_eval,
                *output_eval,
                *mean,
                *rsqrt_var,
                *rsqrt_table_commitment,
                *simd_combined,
                *dim,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Attention { config },
                LayerProof::Attention {
                    sub_proofs,
                    sub_claim_values,
                },
            ) => verify_attention_reduction(
                &current_claim,
                config,
                sub_proofs,
                sub_claim_values,
                Some(&r_simd),
                layer_idx,
                channel,
            )?,

            (
                LayerType::Dequantize { params, .. },
                LayerProof::Dequantize {
                    logup_proof,
                    input_eval,
                    output_eval,
                    table_commitment,
                },
            ) => verify_dequantize_reduction(
                &current_claim,
                params,
                logup_proof.as_ref(),
                *input_eval,
                *output_eval,
                *table_commitment,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Quantize { params, .. },
                LayerProof::Quantize {
                    logup_proof,
                    input_eval,
                    output_eval,
                    table_inputs,
                    table_outputs,
                },
            ) => verify_quantize_reduction(
                &current_claim,
                params,
                logup_proof.as_ref(),
                *input_eval,
                *output_eval,
                table_inputs,
                table_outputs,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Embedding {
                    vocab_size,
                    embed_dim,
                },
                LayerProof::Embedding {
                    logup_proof,
                    input_eval,
                    output_eval,
                    input_num_vars,
                },
            ) => verify_embedding_reduction(
                &current_claim,
                *vocab_size,
                *embed_dim,
                logup_proof.as_ref(),
                *input_eval,
                *output_eval,
                *input_num_vars,
                layer.node_id,
                layer_idx,
                channel,
                None,
            )?,

            (
                LayerType::RMSNorm { dim, .. },
                LayerProof::RMSNorm {
                    logup_proof,
                    linear_round_polys,
                    linear_final_evals,
                    input_eval,
                    output_eval,
                    rms_sq_eval,
                    rsqrt_eval,
                    rsqrt_table_commitment,
                    simd_combined,
                },
            ) => verify_rmsnorm_reduction(
                &current_claim,
                logup_proof.as_ref(),
                linear_round_polys,
                *linear_final_evals,
                *input_eval,
                *output_eval,
                *rms_sq_eval,
                *rsqrt_eval,
                *rsqrt_table_commitment,
                *simd_combined,
                *dim,
                layer_idx,
                channel,
            )?,

            (layer_type, layer_proof) => {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "SIMD verify: layer type {:?} does not match proof type {:?}",
                        std::mem::discriminant(layer_type),
                        std::mem::discriminant(layer_proof),
                    ),
                });
            }
        };
    }

    // Verify final claim matches proof's input_claim
    if current_claim.point != proof.input_claim.point
        || current_claim.value != proof.input_claim.value
    {
        return Err(GKRError::VerificationError {
            layer_idx: 0,
            reason: "SIMD verify: final input claim does not match proof".to_string(),
        });
    }

    Ok(())
}

// ===== Per-Layer Verification =====

/// Verify a matmul reduction: replay sumcheck and check round polynomials.
fn verify_matmul_reduction(
    output_claim: &GKRClaim,
    round_polys: &[RoundPoly],
    final_a_eval: SecureField,
    final_b_eval: SecureField,
    m: usize,
    k: usize,
    n: usize,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let pm = m.next_power_of_two();
    let pk = k.next_power_of_two();
    let pn = n.next_power_of_two();

    let log_m = pm.ilog2() as usize;
    let log_k = pk.ilog2() as usize;
    let log_n = pn.ilog2() as usize;

    // Check proof has correct number of rounds
    if round_polys.len() != log_k {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "expected {} sumcheck rounds, got {}",
                log_k,
                round_polys.len()
            ),
        });
    }

    // Split output claim point into (r_i, r_j)
    let total_out_vars = log_m + log_n;
    if output_claim.point.len() < total_out_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "output claim point too short".to_string(),
        });
    }
    let r_i = &output_claim.point[..log_m];

    // Replay prover's channel operations
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);
    mix_secure_field(channel, output_claim.value);

    // Verify sumcheck rounds
    let mut current_sum = output_claim.value;
    let mut sumcheck_challenges = Vec::with_capacity(log_k);

    for (round, rp) in round_polys.iter().enumerate() {
        // Check: p(0) + p(1) == current_sum
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2;

        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "round {}: p(0)+p(1) = {} != claimed sum {}",
                    round,
                    p0 + p1,
                    current_sum
                ),
            });
        }

        // Replay Fiat-Shamir: mix round poly, draw challenge
        channel.mix_poly_coeffs(rp.c0, rp.c1, rp.c2);
        let r_k = channel.draw_qm31();
        sumcheck_challenges.push(r_k);

        // Update sum: p(r_k) = c0 + c1*r_k + c2*r_k^2
        current_sum = rp.c0 + rp.c1 * r_k + rp.c2 * r_k * r_k;
    }

    // After all rounds: current_sum should equal final_a_eval * final_b_eval
    let expected_product = final_a_eval * final_b_eval;
    if current_sum != expected_product {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "final eval check failed: sum={} != a*b={}",
                current_sum, expected_product
            ),
        });
    }

    // Mix final evals (same as prover)
    mix_secure_field(channel, final_a_eval);
    mix_secure_field(channel, final_b_eval);

    // Build input claim: A evaluated at (r_i, sumcheck_challenges)
    let mut input_point = Vec::with_capacity(log_m + log_k);
    input_point.extend_from_slice(r_i);
    input_point.extend_from_slice(&sumcheck_challenges);

    Ok(GKRClaim {
        point: input_point,
        value: final_a_eval,
    })
}

/// Verify a dual-operand SIMD matmul reduction (block-extended 3-factor sumcheck).
///
/// Replays the degree-3 sumcheck proving:
///   claim = Σ_i ext_w[i] · ext_a[i] · ext_b[i]
///
/// where ext_w, ext_a, ext_b are block-extended MLEs of length N*K.
/// The first `n_block_vars` challenges correspond to block selection,
/// and the remaining `log_k` challenges correspond to the inner dimension.
///
/// Final check: running_sum == eq(r_simd, block_challenges) * final_a * final_b
/// (because ext_w encodes the Lagrange basis of the SIMD challenges).
fn verify_matmul_dual_simd_reduction(
    output_claim: &GKRClaim,
    round_polys: &[RoundPolyDeg3],
    final_a_eval: SecureField,
    final_b_eval: SecureField,
    n_block_vars: usize,
    r_simd: &[SecureField],
    m: usize,
    k: usize,
    n: usize,
    n_blocks: usize,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let pm = m.next_power_of_two();
    let pk = k.next_power_of_two();
    let pn = n.next_power_of_two();

    let log_m = pm.ilog2() as usize;
    let log_k = pk.ilog2() as usize;
    let log_n = pn.ilog2() as usize;

    let expected_rounds = n_block_vars + log_k;
    if round_polys.len() != expected_rounds {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "dual SIMD matmul: expected {} rounds ({} block + {} k), got {}",
                expected_rounds,
                n_block_vars,
                log_k,
                round_polys.len(),
            ),
        });
    }

    if r_simd.len() < n_block_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "dual SIMD matmul: r_simd has {} vars, need at least {}",
                r_simd.len(),
                n_block_vars,
            ),
        });
    }

    let total_out_vars = log_m + log_n;
    if output_claim.point.len() < total_out_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "dual SIMD matmul: output claim point too short".to_string(),
        });
    }

    // Replay prover channel operations
    channel.mix_u64(m as u64);
    channel.mix_u64(k as u64);
    channel.mix_u64(n as u64);
    channel.mix_u64(n_blocks as u64);
    mix_secure_field(channel, output_claim.value);

    // Verify sumcheck rounds (degree-3)
    let mut current_sum = output_claim.value;
    let mut sumcheck_challenges = Vec::with_capacity(expected_rounds);

    for (round, rp) in round_polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "dual SIMD matmul round {}: p(0)+p(1) = {} != claimed sum {}",
                    round,
                    p0 + p1,
                    current_sum,
                ),
            });
        }

        channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        current_sum = rp.eval(challenge);
    }

    // Final check: the sumcheck should reduce to ext_w(s) * ext_a(s) * ext_b(s)
    // ext_w encodes the SIMD Lagrange basis: ext_w[b*K + k] = w_b
    // After folding, ext_w(s) = eq(r_simd, block_challenges) where block_challenges
    // are the first n_block_vars sumcheck challenges.
    let block_challenges = &sumcheck_challenges[..n_block_vars];
    let eq_eval = compute_eq_eval(&r_simd[..n_block_vars], block_challenges);

    let expected = eq_eval * final_a_eval * final_b_eval;
    if current_sum != expected {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "dual SIMD matmul final check failed: sum={} != eq*a*b={}",
                current_sum, expected,
            ),
        });
    }

    // Mix final evals
    mix_secure_field(channel, final_a_eval);
    mix_secure_field(channel, final_b_eval);

    // The input claim for a dual-operand matmul doesn't directly reduce to
    // a single input — both A and B MLEs are already block-extended.
    // We return a claim at the sumcheck challenge point with value = final_a_eval.
    // (This is a leaf in the attention tree — the attention verifier handles routing.)
    let r_i = &output_claim.point[..log_m];
    let k_challenges = &sumcheck_challenges[n_block_vars..];
    let mut input_point = Vec::with_capacity(log_m + log_k);
    input_point.extend_from_slice(r_i);
    input_point.extend_from_slice(k_challenges);

    Ok(GKRClaim {
        point: input_point,
        value: final_a_eval,
    })
}

/// Verify an Add reduction: check lhs_eval + rhs_eval == claimed.
fn verify_add_reduction(
    output_claim: &GKRClaim,
    lhs_eval: SecureField,
    rhs_eval: SecureField,
    input_layers: &[usize],
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let sum = lhs_eval + rhs_eval;
    if sum != output_claim.value {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "add check failed: lhs+rhs={} != claimed={}",
                sum, output_claim.value
            ),
        });
    }

    // Replay channel operations (must match prover exactly)
    mix_secure_field(channel, lhs_eval);
    mix_secure_field(channel, rhs_eval);
    let _alpha = channel.draw_qm31(); // drawn for transcript binding, not used

    // Determine trunk: the input with the higher layer index (the one the
    // sequential walk encounters next). Must match prover's logic exactly.
    let trunk_eval = if input_layers.len() >= 2 && input_layers[1] > input_layers[0] {
        rhs_eval // rhs is the trunk
    } else {
        lhs_eval // lhs is the trunk
    };

    Ok(GKRClaim {
        point: output_claim.point.clone(),
        value: trunk_eval,
    })
}

/// Verify a Mul reduction via eq-sumcheck.
///
/// Replays the degree-3 sumcheck proving:
///   Ṽ_c(r) = Σ_{x ∈ {0,1}^n} eq(r,x) · Ṽ_a(x) · Ṽ_b(x)
///
/// At each round checks p(0) + p(1) == current_sum, where p is degree 3.
/// After all rounds, checks: final_sum == eq(r, challenges) · lhs_eval · rhs_eval.
fn verify_mul_reduction(
    output_claim: &GKRClaim,
    eq_round_polys: &[RoundPolyDeg3],
    lhs_eval: SecureField,
    rhs_eval: SecureField,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let num_vars = eq_round_polys.len();

    if output_claim.point.len() < num_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "mul: claim point has {} vars, need at least {}",
                output_claim.point.len(),
                num_vars,
            ),
        });
    }

    // Replay prover's channel operations: "MUL" tag + claimed sum
    channel.mix_u64(0x4D554C as u64);
    mix_secure_field(channel, output_claim.value);

    let mut current_sum = output_claim.value;
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);

    for (round, rp) in eq_round_polys.iter().enumerate() {
        // Degree-3: p(0) = c0, p(1) = c0 + c1 + c2 + c3
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "mul round {}: p(0)+p(1) = {} != claimed sum {}",
                    round,
                    p0 + p1,
                    current_sum,
                ),
            });
        }

        // Fiat-Shamir: mix round poly, draw challenge
        channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        // Update sum: p(challenge) = c0 + c1*r + c2*r² + c3*r³
        current_sum = rp.eval(challenge);
    }

    // Final check: current_sum == eq(r, challenges) · lhs_eval · rhs_eval
    let r = &output_claim.point[..num_vars];
    let eq_val = compute_eq_eval(r, &sumcheck_challenges);
    let expected = eq_val * lhs_eval * rhs_eval;

    if current_sum != expected {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "mul final check failed: sum={} != eq*a*b={}",
                current_sum, expected,
            ),
        });
    }

    // Mix final evals (same as prover)
    mix_secure_field(channel, lhs_eval);
    mix_secure_field(channel, rhs_eval);

    // Draw combiner for reducing two claims into one
    let alpha = channel.draw_qm31();
    let combined = alpha * lhs_eval + (SecureField::one() - alpha) * rhs_eval;

    Ok(GKRClaim {
        point: output_claim.point.clone(),
        value: combined,
    })
}

/// Verify an Activation reduction via LogUp eq-sumcheck.
///
/// Proves that every (input, output) pair lies in the precomputed activation
/// table using a LogUp argument. The protocol:
///
/// 1. Verifier rebuilds the activation table from the activation type
/// 2. Draws encoding challenges γ, β
/// 3. Computes table-side LogUp sum using multiplicities from proof
/// 4. Checks the claimed sum matches
/// 5. Verifies degree-3 eq-sumcheck: Σ eq(r,x)·w(x)·d(x) = 1
///    where d_i = γ - in_i - β·out_i, w_i = 1/d_i
/// 6. Final check: eq(r,s)·w(s)·d(s) matches the sumcheck output
fn verify_activation_reduction(
    output_claim: &GKRClaim,
    activation_type: ActivationType,
    logup_proof: Option<&LogUpProof>,
    input_eval: SecureField,
    output_eval: SecureField,
    table_commitment: starknet_ff::FieldElement,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let logup = logup_proof.ok_or_else(|| GKRError::VerificationError {
        layer_idx,
        reason: "activation proof missing LogUp proof".to_string(),
    })?;

    let num_vars = logup.eq_round_polys.len();
    if num_vars == 0 {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "activation eq-sumcheck has 0 rounds".to_string(),
        });
    }

    if output_claim.point.len() < num_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "activation: claim point has {} vars, need at least {}",
                output_claim.point.len(),
                num_vars,
            ),
        });
    }

    // 1. Draw LogUp encoding challenges (same Fiat-Shamir as prover)
    channel.mix_u64(0x4C4F47_u64); // "LOG" tag
    channel.mix_u64(activation_type.type_tag() as u64);
    let gamma = channel.draw_qm31();
    let beta = channel.draw_qm31();

    // 2. Rebuild activation table from type (deterministic)
    let table_log_size = activation_type.recommended_table_log_size();
    let activation_fn = activation_type.as_fn();
    let table = PrecomputedTable::build_parallel(|x| activation_fn(x), table_log_size);

    // 3. Verify table commitment
    let expected_commitment = compute_activation_table_commitment(activation_type, table_log_size);
    if table_commitment != expected_commitment {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "activation table commitment mismatch".to_string(),
        });
    }

    // 4. Check multiplicities array length
    let table_size = 1usize << table_log_size;
    if logup.multiplicities.len() != table_size {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "multiplicities length {} != table size {}",
                logup.multiplicities.len(),
                table_size,
            ),
        });
    }

    // 5. Compute table-side LogUp sum: S = Σ mult_j / (γ - table_in_j - β·table_out_j)
    let table_sum: SecureField = table
        .inputs
        .iter()
        .zip(&table.outputs)
        .enumerate()
        .filter(|(j, _)| logup.multiplicities[*j] > 0)
        .map(|(j, (&t_in, &t_out))| {
            let m = SecureField::from(M31::from(logup.multiplicities[j]));
            let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
            m * d.inverse()
        })
        .fold(SecureField::zero(), |acc, v| acc + v);

    // 6. Check LogUp sum balance: claimed_sum == table_sum
    if logup.claimed_sum != table_sum {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "LogUp sum mismatch: claimed={}, table={}",
                logup.claimed_sum, table_sum,
            ),
        });
    }

    // 7. Mix claimed sum (same as prover)
    mix_secure_field(channel, logup.claimed_sum);

    // 8. Verify degree-3 eq-sumcheck with initial sum = 1
    //    Proves: Σ_{x ∈ {0,1}^n} eq(r,x) · w(x) · d(x) = 1
    //    Since w(x) = 1/d(x), we have w(x)·d(x) = 1 for all boolean x,
    //    and Σ eq(r,x) = 1, so the claimed sum is 1.
    let mut current_sum = SecureField::one();
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);

    for (round, rp) in logup.eq_round_polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "activation eq-sumcheck round {}: p(0)+p(1) = {} != sum {}",
                    round,
                    p0 + p1,
                    current_sum,
                ),
            });
        }

        channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        current_sum = rp.eval(challenge);
    }

    // 9. Final check: current_sum == eq(r, s) · w(s) · d(s)
    //    where s = sumcheck_challenges, d(s) = γ - in(s) - β·out(s)
    let (w_eval, in_eval_s, out_eval_s) = logup.final_evals;
    let d_eval = gamma - in_eval_s - beta * out_eval_s;
    let r = &output_claim.point[..num_vars];
    let eq_val = compute_eq_eval(r, &sumcheck_challenges);
    let expected = eq_val * w_eval * d_eval;

    if current_sum != expected {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "activation final check failed: sum={} != eq*w*d={}",
                current_sum, expected,
            ),
        });
    }

    // 10. Mix final evals (same as prover)
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    // Return claim on the activation input MLE
    Ok(GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    })
}

/// Verify a Dequantize reduction via LogUp eq-sumcheck.
///
/// Same protocol as `verify_activation_reduction` but:
/// - Table built from QuantParams (deterministic: INT4→16, INT8→256 entries)
/// - "DEQLOG" Fiat-Shamir tag
/// - Table commitment derived from (bits, scale, zero_point) hash
fn verify_dequantize_reduction(
    output_claim: &GKRClaim,
    params: &crate::gadgets::quantize::QuantParams,
    logup_proof: Option<&LogUpProof>,
    input_eval: SecureField,
    output_eval: SecureField,
    table_commitment: starknet_ff::FieldElement,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    use super::prover::compute_dequantize_table_commitment;
    use crate::components::dequantize::build_dequantize_table;

    let logup = logup_proof.ok_or_else(|| GKRError::VerificationError {
        layer_idx,
        reason: "dequantize proof missing LogUp proof".to_string(),
    })?;

    let num_vars = logup.eq_round_polys.len();
    if num_vars == 0 {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "dequantize eq-sumcheck has 0 rounds".to_string(),
        });
    }

    if output_claim.point.len() < num_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "dequantize: claim point has {} vars, need at least {}",
                output_claim.point.len(),
                num_vars,
            ),
        });
    }

    // 1. Draw LogUp encoding challenges (same Fiat-Shamir as prover)
    channel.mix_u64(0x4445514C4F47_u64); // "DEQLOG" tag
    channel.mix_u64(params.bits as u64);
    let gamma = channel.draw_qm31();
    let beta = channel.draw_qm31();

    // 2. Rebuild dequantize table from params (deterministic)
    let table = build_dequantize_table(params);

    // 3. Verify table commitment
    let expected_commitment = compute_dequantize_table_commitment(params);
    if table_commitment != expected_commitment {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "dequantize table commitment mismatch".to_string(),
        });
    }

    // 4. Check multiplicities array length
    let table_size = 1usize << params.bits;
    if logup.multiplicities.len() != table_size {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "dequantize multiplicities length {} != table size {}",
                logup.multiplicities.len(),
                table_size,
            ),
        });
    }

    // 5. Compute table-side LogUp sum: S = Σ mult_j / (γ - table_in_j - β·table_out_j)
    let table_sum: SecureField = table
        .inputs
        .iter()
        .zip(&table.outputs)
        .enumerate()
        .filter(|(j, _)| logup.multiplicities[*j] > 0)
        .map(|(j, (&t_in, &t_out))| {
            let m = SecureField::from(M31::from(logup.multiplicities[j]));
            let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
            m * d.inverse()
        })
        .fold(SecureField::zero(), |acc, v| acc + v);

    // 6. Check LogUp sum balance
    if logup.claimed_sum != table_sum {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "dequantize LogUp sum mismatch: claimed={}, table={}",
                logup.claimed_sum, table_sum,
            ),
        });
    }

    // 7. Mix claimed sum (same as prover)
    mix_secure_field(channel, logup.claimed_sum);

    // 8. Verify degree-3 eq-sumcheck with initial sum = 1
    let mut current_sum = SecureField::one();
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);

    for (round, rp) in logup.eq_round_polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "dequantize eq-sumcheck round {}: p(0)+p(1) = {} != sum {}",
                    round,
                    p0 + p1,
                    current_sum,
                ),
            });
        }

        channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);

        current_sum = rp.eval(challenge);
    }

    // 9. Final check: current_sum == eq(r, s) · w(s) · d(s)
    let (w_eval, in_eval_s, out_eval_s) = logup.final_evals;
    let d_eval = gamma - in_eval_s - beta * out_eval_s;
    let r = &output_claim.point[..num_vars];
    let eq_val = compute_eq_eval(r, &sumcheck_challenges);
    let expected = eq_val * w_eval * d_eval;

    if current_sum != expected {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "dequantize final check failed: sum={} != eq*w*d={}",
                current_sum, expected,
            ),
        });
    }

    // 10. Mix final evals (same as prover)
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    Ok(GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    })
}

fn quantize_strategy_tag(strategy: crate::gadgets::quantize::QuantStrategy) -> u64 {
    use crate::gadgets::quantize::QuantStrategy;
    match strategy {
        QuantStrategy::Direct => 0,
        QuantStrategy::Symmetric8 => 1,
        QuantStrategy::Asymmetric8 => 2,
        QuantStrategy::Symmetric4 => 3,
        QuantStrategy::Asymmetric4 => 4,
    }
}

fn project_claim_point(point: &[SecureField], n_vars: usize) -> Vec<SecureField> {
    let mut out = point.to_vec();
    if out.len() > n_vars {
        out.truncate(n_vars);
    } else if out.len() < n_vars {
        out.resize(n_vars, SecureField::zero());
    }
    out
}

/// Verify a Quantize reduction via LogUp eq-sumcheck.
fn verify_quantize_reduction(
    output_claim: &GKRClaim,
    params: &crate::gadgets::quantize::QuantParams,
    logup_proof: Option<&LogUpProof>,
    input_eval: SecureField,
    output_eval: SecureField,
    table_inputs: &[M31],
    table_outputs: &[M31],
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    use crate::gadgets::quantize::{dequantize_value, quantize_value, QuantParams, QuantStrategy};

    let logup = logup_proof.ok_or_else(|| GKRError::VerificationError {
        layer_idx,
        reason: "quantize proof missing LogUp proof".to_string(),
    })?;

    let num_vars = logup.eq_round_polys.len();
    if num_vars == 0 {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "quantize eq-sumcheck has 0 rounds".to_string(),
        });
    }

    if table_inputs.len() != table_outputs.len() {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "quantize table input/output length mismatch: {} vs {}",
                table_inputs.len(),
                table_outputs.len()
            ),
        });
    }

    if logup.multiplicities.len() != table_inputs.len() {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "quantize multiplicities length {} != table size {}",
                logup.multiplicities.len(),
                table_inputs.len()
            ),
        });
    }

    channel.mix_u64(0x514C4F47_u64); // "QLOG"
    channel.mix_u64(params.bits as u64);
    channel.mix_u64(params.zero_point.unsigned_abs() as u64);
    channel.mix_u64((params.scale * (1u64 << 32) as f64) as u64);
    channel.mix_u64(quantize_strategy_tag(params.strategy));
    let gamma = channel.draw_qm31();
    let beta = channel.draw_qm31();

    // Ensure table rows are consistent with quantization function.
    let direct_params = QuantParams {
        strategy: QuantStrategy::Direct,
        scale: 1.0,
        zero_point: 0,
        bits: 31,
    };
    for (j, (&inp, &out)) in table_inputs.iter().zip(table_outputs.iter()).enumerate() {
        if logup.multiplicities[j] == 0 {
            continue;
        }
        let f = dequantize_value(inp, &direct_params);
        let expected = quantize_value(f, params);
        if out != expected {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "quantize table row invalid: input={} expected={} got={}",
                    inp.0, expected.0, out.0
                ),
            });
        }
    }

    let table_sum: SecureField = table_inputs
        .iter()
        .zip(table_outputs.iter())
        .enumerate()
        .filter(|(j, _)| logup.multiplicities[*j] > 0)
        .map(|(j, (&t_in, &t_out))| {
            let m = SecureField::from(M31::from(logup.multiplicities[j]));
            let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
            m * d.inverse()
        })
        .fold(SecureField::zero(), |acc, v| acc + v);

    if logup.claimed_sum != table_sum {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "quantize LogUp sum mismatch: claimed={}, table={}",
                logup.claimed_sum, table_sum
            ),
        });
    }

    mix_secure_field(channel, logup.claimed_sum);

    let mut current_sum = SecureField::one();
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);
    for (round, rp) in logup.eq_round_polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;
        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "quantize eq-sumcheck round {}: p(0)+p(1) = {} != sum {}",
                    round,
                    p0 + p1,
                    current_sum
                ),
            });
        }
        channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);
        current_sum = rp.eval(challenge);
    }

    let (w_eval, in_eval_s, out_eval_s) = logup.final_evals;
    let d_eval = gamma - in_eval_s - beta * out_eval_s;
    let r = &project_claim_point(&output_claim.point, num_vars)[..num_vars];
    let eq_val = compute_eq_eval(r, &sumcheck_challenges);
    let expected = eq_val * w_eval * d_eval;
    if current_sum != expected {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "quantize final check failed: sum={} != eq*w*d={}",
                current_sum, expected
            ),
        });
    }

    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    Ok(GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    })
}

/// Verify an Embedding reduction via LogUp eq-sumcheck.
fn verify_embedding_reduction(
    output_claim: &GKRClaim,
    vocab_size: usize,
    embed_dim: usize,
    logup_proof: Option<&EmbeddingLogUpProof>,
    input_eval: SecureField,
    output_eval: SecureField,
    input_num_vars: usize,
    weight_node_id: usize,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
    weights: Option<&GraphWeights>,
) -> Result<GKRClaim, GKRError> {
    let weights = weights.ok_or_else(|| GKRError::VerificationError {
        layer_idx,
        reason: "embedding reduction requires verify_gkr_with_weights(...)".to_string(),
    })?;
    let embedding_table = weights
        .get_weight(weight_node_id)
        .ok_or(GKRError::MissingWeight {
            node_id: weight_node_id,
        })?;
    if embedding_table.rows != vocab_size || embedding_table.cols != embed_dim {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "embedding table shape mismatch: expected {}x{}, got {}x{}",
                vocab_size, embed_dim, embedding_table.rows, embedding_table.cols
            ),
        });
    }

    let logup = logup_proof.ok_or_else(|| GKRError::VerificationError {
        layer_idx,
        reason: "embedding proof missing LogUp proof".to_string(),
    })?;
    let num_vars = logup.eq_round_polys.len();
    if num_vars == 0 {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "embedding eq-sumcheck has 0 rounds".to_string(),
        });
    }
    if logup.table_tokens.len() != logup.table_cols.len()
        || logup.table_tokens.len() != logup.multiplicities.len()
    {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "embedding sparse multiplicity vectors have inconsistent lengths".to_string(),
        });
    }

    channel.mix_u64(0x454D424C4F47_u64); // "EMBLOG"
    channel.mix_u64(vocab_size as u64);
    channel.mix_u64(embed_dim as u64);
    let gamma = channel.draw_qm31();
    let beta_col = channel.draw_qm31();
    let beta_val = channel.draw_qm31();

    let table_sum: SecureField = logup
        .table_tokens
        .iter()
        .zip(logup.table_cols.iter())
        .zip(logup.multiplicities.iter())
        .map(|((&tok, &col), &mult)| {
            if (tok as usize) >= vocab_size || (col as usize) >= embed_dim {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "embedding sparse multiplicity out of bounds: token={}, col={}",
                        tok, col
                    ),
                });
            }
            let val = embedding_table.get(tok as usize, col as usize);
            let d = gamma
                - SecureField::from(M31::from(tok))
                - beta_col * SecureField::from(M31::from(col))
                - beta_val * SecureField::from(val);
            Ok(SecureField::from(M31::from(mult)) * d.inverse())
        })
        .collect::<Result<Vec<_>, GKRError>>()?
        .into_iter()
        .fold(SecureField::zero(), |acc, v| acc + v);

    if logup.claimed_sum != table_sum {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "embedding LogUp sum mismatch: claimed={}, table={}",
                logup.claimed_sum, table_sum
            ),
        });
    }

    mix_secure_field(channel, logup.claimed_sum);

    let mut current_sum = SecureField::one();
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);
    for (round, rp) in logup.eq_round_polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;
        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "embedding eq-sumcheck round {}: p(0)+p(1) = {} != sum {}",
                    round,
                    p0 + p1,
                    current_sum
                ),
            });
        }
        channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
        let challenge = channel.draw_qm31();
        sumcheck_challenges.push(challenge);
        current_sum = rp.eval(challenge);
    }

    let (w_eval, tok_eval_s, col_eval_s, val_eval_s) = logup.final_evals;
    let d_eval = gamma - tok_eval_s - beta_col * col_eval_s - beta_val * val_eval_s;
    let r = &project_claim_point(&output_claim.point, num_vars)[..num_vars];
    let eq_val = compute_eq_eval(r, &sumcheck_challenges);
    let expected = eq_val * w_eval * d_eval;
    if current_sum != expected {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "embedding final check failed: sum={} != eq*w*d={}",
                current_sum, expected
            ),
        });
    }

    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    Ok(GKRClaim {
        point: project_claim_point(&output_claim.point, input_num_vars),
        value: input_eval,
    })
}

/// Verify a LayerNorm reduction via:
/// 1. Degree-3 eq-sumcheck: output = Σ eq(r,x) · centered(x) · rsqrt(x)
/// 2. LogUp eq-sumcheck: all (variance, rsqrt) pairs ∈ rsqrt_table
///
/// The prover commits to mean and rsqrt_var evaluations, then proves
/// the linear relationship output = (input - mean) × rsqrt and that
/// the rsqrt values come from the precomputed rsqrt table.
fn verify_layernorm_reduction(
    output_claim: &GKRClaim,
    logup_proof: Option<&LogUpProof>,
    linear_round_polys: &[RoundPolyDeg3],
    linear_final_evals: (SecureField, SecureField),
    input_eval: SecureField,
    output_eval: SecureField,
    mean_eval: SecureField,
    rsqrt_eval: SecureField,
    rsqrt_table_commitment: starknet_ff::FieldElement,
    simd_combined: bool,
    dim: usize,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let num_vars = linear_round_polys.len();
    if num_vars == 0 {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "layernorm linear eq-sumcheck has 0 rounds".to_string(),
        });
    }

    if output_claim.point.len() < num_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "layernorm: claim point has {} vars, need at least {}",
                output_claim.point.len(),
                num_vars,
            ),
        });
    }

    // Non-SIMD LayerNorm MUST include a LogUp proof for rsqrt table verification.
    // The SIMD combined-product path legitimately skips LogUp since combined
    // variance/rsqrt are QM31 sums that don't map to individual table entries.
    if !simd_combined && logup_proof.is_none() {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "non-SIMD LayerNorm missing required LogUp proof".into(),
        });
    }

    // LogUp must have same number of rounds as linear sumcheck (if present)
    if let Some(logup) = logup_proof {
        if logup.eq_round_polys.len() != num_vars {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "layernorm: LogUp has {} rounds, linear has {}",
                    logup.eq_round_polys.len(),
                    num_vars,
                ),
            });
        }
    }

    // ===== Part 1: Linear transform eq-sumcheck =====
    // Proves: output_claim.value = Σ_{x} eq(r,x) · centered(x) · rsqrt(x)
    // Replay prover's Fiat-Shamir transcript exactly.

    channel.mix_u64(0x4C4E as u64); // "LN" tag
    mix_secure_field(channel, mean_eval);
    mix_secure_field(channel, rsqrt_eval);
    mix_secure_field(channel, output_claim.value);

    let mut current_sum = output_claim.value;
    let mut linear_challenges = Vec::with_capacity(num_vars);

    for (round, rp) in linear_round_polys.iter().enumerate() {
        // Degree-3: p(0) + p(1) == current_sum
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "layernorm linear round {}: p(0)+p(1) = {} != sum {}",
                    round,
                    p0 + p1,
                    current_sum,
                ),
            });
        }

        channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
        let challenge = channel.draw_qm31();
        linear_challenges.push(challenge);

        current_sum = rp.eval(challenge);
    }

    // Final check: current_sum == eq(r, challenges) · centered_final · rsqrt_final
    let (centered_final, rsqrt_final) = linear_final_evals;
    let r = &output_claim.point[..num_vars];
    let eq_val = compute_eq_eval(r, &linear_challenges);
    let expected_linear = eq_val * centered_final * rsqrt_final;

    if current_sum != expected_linear {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "layernorm linear final check failed: sum={} != eq*centered*rsqrt={}",
                current_sum, expected_linear,
            ),
        });
    }

    // Mix final linear evals (same as prover)
    mix_secure_field(channel, centered_final);
    mix_secure_field(channel, rsqrt_final);

    // ===== Part 2: rsqrt LogUp eq-sumcheck =====
    // Proves: all (variance[i], rsqrt[i]) pairs are in the rsqrt_table.
    // Skipped for SIMD combined-product path (logup_proof: None).
    if let Some(logup) = logup_proof {
        // Draw LogUp encoding challenges (same Fiat-Shamir as prover)
        channel.mix_u64(0x4C4F47 as u64); // "LOG" tag
        channel.mix_u64(0x5253 as u64); // "RS" rsqrt type tag
        let gamma = channel.draw_qm31();
        let beta = channel.draw_qm31();

        // Rebuild rsqrt table (deterministic from config)
        let config = crate::components::layernorm::LayerNormConfig::new(dim);
        let rsqrt_table =
            crate::components::layernorm::build_rsqrt_table(config.rsqrt_table_log_size);

        // Verify rsqrt table commitment
        let expected_commitment = compute_rsqrt_table_commitment(config.rsqrt_table_log_size);
        if rsqrt_table_commitment != expected_commitment {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: "rsqrt table commitment mismatch".to_string(),
            });
        }

        // Check multiplicities array length
        let table_size = 1usize << config.rsqrt_table_log_size;
        if logup.multiplicities.len() != table_size {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "rsqrt multiplicities length {} != table size {}",
                    logup.multiplicities.len(),
                    table_size,
                ),
            });
        }

        // Compute table-side LogUp sum
        let table_sum: SecureField = rsqrt_table
            .inputs
            .iter()
            .zip(&rsqrt_table.outputs)
            .enumerate()
            .filter(|(j, _)| logup.multiplicities[*j] > 0)
            .map(|(j, (&t_in, &t_out))| {
                let m = SecureField::from(M31::from(logup.multiplicities[j]));
                let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
                m * d.inverse()
            })
            .fold(SecureField::zero(), |acc, v| acc + v);

        // Check LogUp sum balance
        if logup.claimed_sum != table_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "rsqrt LogUp sum mismatch: claimed={}, table={}",
                    logup.claimed_sum, table_sum,
                ),
            });
        }

        // Mix claimed sum (same as prover)
        mix_secure_field(channel, logup.claimed_sum);

        // Verify degree-3 eq-sumcheck with initial sum = 1
        let mut logup_sum = SecureField::one();
        let mut logup_challenges = Vec::with_capacity(num_vars);

        for (round, rp) in logup.eq_round_polys.iter().enumerate() {
            let p0 = rp.c0;
            let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

            if p0 + p1 != logup_sum {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "layernorm LogUp round {}: p(0)+p(1) = {} != sum {}",
                        round,
                        p0 + p1,
                        logup_sum,
                    ),
                });
            }

            channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
            let challenge = channel.draw_qm31();
            logup_challenges.push(challenge);

            logup_sum = rp.eval(challenge);
        }

        // Final check: logup_sum == eq(r, challenges) · w_eval · d_eval
        let (w_eval, var_eval_s, rsqrt_eval_s) = logup.final_evals;
        let d_eval = gamma - var_eval_s - beta * rsqrt_eval_s;
        let eq_val_logup = compute_eq_eval(r, &logup_challenges);
        let expected_logup = eq_val_logup * w_eval * d_eval;

        if logup_sum != expected_logup {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "layernorm LogUp final check failed: sum={} != eq*w*d={}",
                    logup_sum, expected_logup,
                ),
            });
        }
    } // end if let Some(logup) = logup_proof

    // Mix final evals (same as prover)
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    // Return claim on the layernorm input MLE
    Ok(GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    })
}

/// Verify an RMSNorm reduction via:
/// 1. Degree-3 eq-sumcheck: output = Σ eq(r,x) · input(x) · rsqrt(x)
/// 2. LogUp eq-sumcheck: all (rms², rsqrt) pairs ∈ rsqrt_table
///
/// Unlike LayerNorm, RMSNorm has no mean subtraction:
///   output = input × rsqrt(mean(x²) + ε)
fn verify_rmsnorm_reduction(
    output_claim: &GKRClaim,
    logup_proof: Option<&LogUpProof>,
    linear_round_polys: &[RoundPolyDeg3],
    linear_final_evals: (SecureField, SecureField),
    input_eval: SecureField,
    output_eval: SecureField,
    rms_sq_eval: SecureField,
    rsqrt_eval: SecureField,
    rsqrt_table_commitment: starknet_ff::FieldElement,
    simd_combined: bool,
    dim: usize,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let num_vars = linear_round_polys.len();
    if num_vars == 0 {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "rmsnorm linear eq-sumcheck has 0 rounds".to_string(),
        });
    }

    if output_claim.point.len() < num_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "rmsnorm: claim point has {} vars, need at least {}",
                output_claim.point.len(),
                num_vars,
            ),
        });
    }

    // Non-SIMD RMSNorm MUST include a LogUp proof for rsqrt table verification.
    if !simd_combined && logup_proof.is_none() {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "non-SIMD RMSNorm missing required LogUp proof".into(),
        });
    }

    // LogUp must have same number of rounds as linear sumcheck (if present)
    if let Some(logup) = logup_proof {
        if logup.eq_round_polys.len() != num_vars {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "rmsnorm: LogUp has {} rounds, linear has {}",
                    logup.eq_round_polys.len(),
                    num_vars,
                ),
            });
        }
    }

    // ===== Part 1: Linear transform eq-sumcheck =====
    // Proves: output_claim.value = Σ_{x} eq(r,x) · input(x) · rsqrt(x)
    // NOTE: No mean subtraction — this is the key difference from LayerNorm.

    channel.mix_u64(0x524E as u64); // "RN" tag
    mix_secure_field(channel, rms_sq_eval);
    mix_secure_field(channel, rsqrt_eval);
    mix_secure_field(channel, output_claim.value);

    let mut current_sum = output_claim.value;
    let mut linear_challenges = Vec::with_capacity(num_vars);

    for (round, rp) in linear_round_polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "rmsnorm linear round {}: p(0)+p(1) = {} != sum {}",
                    round,
                    p0 + p1,
                    current_sum,
                ),
            });
        }

        channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
        let challenge = channel.draw_qm31();
        linear_challenges.push(challenge);

        current_sum = rp.eval(challenge);
    }

    // Final check: current_sum == eq(r, challenges) · input_final · rsqrt_final
    let (input_final, rsqrt_final) = linear_final_evals;
    let r = &output_claim.point[..num_vars];
    let eq_val = compute_eq_eval(r, &linear_challenges);
    let expected_linear = eq_val * input_final * rsqrt_final;

    if current_sum != expected_linear {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "rmsnorm linear final check failed: sum={} != eq*input*rsqrt={}",
                current_sum, expected_linear,
            ),
        });
    }

    // Mix final linear evals (same as prover)
    mix_secure_field(channel, input_final);
    mix_secure_field(channel, rsqrt_final);

    // ===== Part 2: rsqrt LogUp eq-sumcheck =====
    // Proves: all (rms²[i], rsqrt[i]) pairs are in the rsqrt_table.
    if let Some(logup) = logup_proof {
        // Draw LogUp encoding challenges (same Fiat-Shamir as prover)
        channel.mix_u64(0x4C4F47 as u64); // "LOG" tag
        channel.mix_u64(0x524E as u64); // "RN" rmsnorm type tag
        let gamma = channel.draw_qm31();
        let beta = channel.draw_qm31();

        // Rebuild rsqrt table (deterministic from config)
        let config = crate::components::rmsnorm::RMSNormConfig::new(dim);
        let rsqrt_table =
            crate::components::rmsnorm::build_rsqrt_table(config.rsqrt_table_log_size);

        // Verify rsqrt table commitment
        let expected_commitment = compute_rsqrt_table_commitment(config.rsqrt_table_log_size);
        if rsqrt_table_commitment != expected_commitment {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: "rmsnorm rsqrt table commitment mismatch".to_string(),
            });
        }

        // Check multiplicities array length
        let table_size = 1usize << config.rsqrt_table_log_size;
        if logup.multiplicities.len() != table_size {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "rmsnorm rsqrt multiplicities length {} != table size {}",
                    logup.multiplicities.len(),
                    table_size,
                ),
            });
        }

        // Compute table-side LogUp sum
        let table_sum: SecureField = rsqrt_table
            .inputs
            .iter()
            .zip(&rsqrt_table.outputs)
            .enumerate()
            .filter(|(j, _)| logup.multiplicities[*j] > 0)
            .map(|(j, (&t_in, &t_out))| {
                let m = SecureField::from(M31::from(logup.multiplicities[j]));
                let d = gamma - SecureField::from(t_in) - beta * SecureField::from(t_out);
                m * d.inverse()
            })
            .fold(SecureField::zero(), |acc, v| acc + v);

        // Check LogUp sum balance
        if logup.claimed_sum != table_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "rmsnorm rsqrt LogUp sum mismatch: claimed={}, table={}",
                    logup.claimed_sum, table_sum,
                ),
            });
        }

        // Mix claimed sum (same as prover)
        mix_secure_field(channel, logup.claimed_sum);

        // Verify degree-3 eq-sumcheck with initial sum = 1
        let mut logup_sum = SecureField::one();
        let mut logup_challenges = Vec::with_capacity(num_vars);

        for (round, rp) in logup.eq_round_polys.iter().enumerate() {
            let p0 = rp.c0;
            let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

            if p0 + p1 != logup_sum {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "rmsnorm LogUp round {}: p(0)+p(1) = {} != sum {}",
                        round,
                        p0 + p1,
                        logup_sum,
                    ),
                });
            }

            channel.mix_poly_coeffs_deg3(rp.c0, rp.c1, rp.c2, rp.c3);
            let challenge = channel.draw_qm31();
            logup_challenges.push(challenge);

            logup_sum = rp.eval(challenge);
        }

        // Final check: logup_sum == eq(r, challenges) · w_eval · d_eval
        let (w_eval, rms_sq_eval_s, rsqrt_eval_s) = logup.final_evals;
        let d_eval = gamma - rms_sq_eval_s - beta * rsqrt_eval_s;
        let eq_val_logup = compute_eq_eval(r, &logup_challenges);
        let expected_logup = eq_val_logup * w_eval * d_eval;

        if logup_sum != expected_logup {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "rmsnorm LogUp final check failed: sum={} != eq*w*d={}",
                    logup_sum, expected_logup,
                ),
            });
        }
    } // end if let Some(logup) = logup_proof

    // Mix final evals (same as prover)
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    // Return claim on the rmsnorm input MLE
    Ok(GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    })
}

/// Verify an Attention reduction by walking the decomposed sub-proofs.
///
/// Each sub-matmul gets a fresh random evaluation point drawn from the
/// channel. The prover-supplied `sub_claim_values` provide the output MLE
/// evaluations at those points. The sumcheck verification ensures these
/// values are consistent with the actual matrix products.
///
/// Sub-proofs (output → input order):
///   0: Output projection matmul (uses parent output_claim)
///   1..2H: Per-head (h = H-1..0): context matmul + score matmul (fresh claims)
///   2H+1..2H+3: V, K, Q projection matmuls (fresh claims)
///
/// Total expected: 4 + 2 × num_heads
pub(crate) fn verify_attention_reduction(
    output_claim: &GKRClaim,
    config: &MultiHeadAttentionConfig,
    sub_proofs: &[LayerProof],
    sub_claim_values: &[SecureField],
    r_simd: Option<&[SecureField]>,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let num_heads = config.num_heads;
    let seq_len = config.seq_len;
    let d_model = config.d_model;
    let d_k = config.d_k();
    let n_blocks = r_simd.map(|r| 1 << r.len()).unwrap_or(1);

    let expected_count = 4 + 2 * num_heads;
    if sub_proofs.len() != expected_count {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "attention: expected {} sub-proofs, got {}",
                expected_count,
                sub_proofs.len(),
            ),
        });
    }
    if sub_claim_values.len() != expected_count {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "attention: expected {} sub-claim values, got {}",
                expected_count,
                sub_claim_values.len(),
            ),
        });
    }

    // Replay attention metadata in channel (same as prover)
    channel.mix_u64(0x4154544E_u64); // "ATTN" tag
    channel.mix_u64(num_heads as u64);
    channel.mix_u64(seq_len as u64);
    channel.mix_u64(d_model as u64);
    channel.mix_u64(if config.causal { 1 } else { 0 });

    // Helper: verify a sub-matmul that used a fresh claim.
    // Handles both MatMul (shared-weight, degree-2) and MatMulDualSimd (dual-operand, degree-3).
    let verify_fresh_sub_matmul = |proof: &LayerProof,
                                   claimed_value: SecureField,
                                   m: usize,
                                   k: usize,
                                   n: usize,
                                   r_simd: Option<&[SecureField]>,
                                   channel: &mut PoseidonChannel|
     -> Result<GKRClaim, GKRError> {
        let pm = m.next_power_of_two();
        let pn = n.next_power_of_two();
        let log_rows = pm.ilog2() as usize;
        let log_cols = pn.ilog2() as usize;

        let r = channel.draw_qm31s(log_rows + log_cols);
        mix_secure_field(channel, claimed_value);

        let fresh_claim = GKRClaim {
            point: r,
            value: claimed_value,
        };

        match proof {
            LayerProof::MatMul {
                round_polys,
                final_a_eval,
                final_b_eval,
            } => verify_matmul_reduction(
                &fresh_claim,
                round_polys,
                *final_a_eval,
                *final_b_eval,
                m,
                k,
                n,
                layer_idx,
                channel,
            ),
            LayerProof::MatMulDualSimd {
                round_polys,
                final_a_eval,
                final_b_eval,
                n_block_vars,
            } => {
                let simd = r_simd.ok_or_else(|| GKRError::VerificationError {
                    layer_idx,
                    reason: "MatMulDualSimd sub-proof requires r_simd".to_string(),
                })?;
                verify_matmul_dual_simd_reduction(
                    &fresh_claim,
                    round_polys,
                    *final_a_eval,
                    *final_b_eval,
                    *n_block_vars,
                    simd,
                    m,
                    k,
                    n,
                    n_blocks,
                    layer_idx,
                    channel,
                )
            }
            other => Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "attention sub-proof: expected MatMul or MatMulDualSimd, got {:?}",
                    std::mem::discriminant(other),
                ),
            }),
        }
    };

    let mut proof_idx = 0;

    // --- Sub-proof 0: Output projection matmul ---
    // Uses the actual output_claim (not a fresh one)
    let _output_proj_claim = match &sub_proofs[proof_idx] {
        LayerProof::MatMul {
            round_polys,
            final_a_eval,
            final_b_eval,
        } => verify_matmul_reduction(
            output_claim,
            round_polys,
            *final_a_eval,
            *final_b_eval,
            seq_len,
            d_model,
            d_model,
            layer_idx,
            channel,
        )?,
        other => {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "attention output proj: expected MatMul, got {:?}",
                    std::mem::discriminant(other),
                ),
            });
        }
    };
    proof_idx += 1;

    // --- Per-head sub-proofs (h = H-1..0) ---
    for _h in (0..num_heads).rev() {
        // Context matmul (fresh claim): context_h = softmax_h × V_h
        let _ctx_claim = verify_fresh_sub_matmul(
            &sub_proofs[proof_idx],
            sub_claim_values[proof_idx],
            seq_len,
            seq_len,
            d_k,
            r_simd,
            channel,
        )?;
        proof_idx += 1;

        // Score matmul (fresh claim): scores_h = Q_h × K_h^T
        let _score_claim = verify_fresh_sub_matmul(
            &sub_proofs[proof_idx],
            sub_claim_values[proof_idx],
            seq_len,
            d_k,
            seq_len,
            r_simd,
            channel,
        )?;
        proof_idx += 1;
    }

    // --- Projection matmuls (fresh claims): V, K, Q ---
    let _v_claim = verify_fresh_sub_matmul(
        &sub_proofs[proof_idx],
        sub_claim_values[proof_idx],
        seq_len,
        d_model,
        d_model,
        r_simd,
        channel,
    )?;
    proof_idx += 1;

    let _k_claim = verify_fresh_sub_matmul(
        &sub_proofs[proof_idx],
        sub_claim_values[proof_idx],
        seq_len,
        d_model,
        d_model,
        r_simd,
        channel,
    )?;
    proof_idx += 1;

    // Q projection (fresh claim) — determines the final input claim
    let q_padded_rows = seq_len.next_power_of_two();
    let q_padded_cols = d_model.next_power_of_two();
    let log_rows = q_padded_rows.ilog2() as usize;
    let log_cols = q_padded_cols.ilog2() as usize;
    let r_q = channel.draw_qm31s(log_rows + log_cols);
    let q_value = sub_claim_values[proof_idx];
    mix_secure_field(channel, q_value);
    let q_claim = GKRClaim {
        point: r_q,
        value: q_value,
    };

    let final_claim = match &sub_proofs[proof_idx] {
        LayerProof::MatMul {
            round_polys,
            final_a_eval,
            final_b_eval,
        } => verify_matmul_reduction(
            &q_claim,
            round_polys,
            *final_a_eval,
            *final_b_eval,
            seq_len,
            d_model,
            d_model,
            layer_idx,
            channel,
        )?,
        other => {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "attention Q projection: expected MatMul, got {:?}",
                    std::mem::discriminant(other),
                ),
            });
        }
    };

    Ok(final_claim)
}

/// Test-accessible wrapper for `verify_attention_reduction`.
pub fn verify_attention_reduction_for_test(
    output_claim: &GKRClaim,
    config: &MultiHeadAttentionConfig,
    sub_proofs: &[LayerProof],
    sub_claim_values: &[SecureField],
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    verify_attention_reduction(
        output_claim,
        config,
        sub_proofs,
        sub_claim_values,
        None,
        layer_idx,
        channel,
    )
}

// ===== Helpers =====

/// Compute eq(a, b) = Π_i ((1 - a_i)(1 - b_i) + a_i · b_i).
fn compute_eq_eval(a: &[SecureField], b: &[SecureField]) -> SecureField {
    assert_eq!(a.len(), b.len());
    let mut result = SecureField::one();
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        result = result * ((SecureField::one() - ai) * (SecureField::one() - bi) + ai * bi);
    }
    result
}

/// Compute a deterministic commitment for an activation table.
/// Since the table is fully determined by (activation_type, table_log_size),
/// the commitment is a Poseidon hash of these parameters.
fn compute_activation_table_commitment(
    activation_type: ActivationType,
    table_log_size: u32,
) -> starknet_ff::FieldElement {
    starknet_crypto::poseidon_hash_many(&[
        starknet_ff::FieldElement::from(activation_type.type_tag() as u64),
        starknet_ff::FieldElement::from(table_log_size as u64),
    ])
}

/// Mix a SecureField into PoseidonChannel (must match prover's mix_secure_field).
fn mix_secure_field(channel: &mut PoseidonChannel, v: SecureField) {
    channel.mix_u64(v.0 .0 .0 as u64);
    channel.mix_u64(v.0 .1 .0 as u64);
    channel.mix_u64(v.1 .0 .0 as u64);
    channel.mix_u64(v.1 .1 .0 as u64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::graph::{GraphBuilder, GraphExecution, GraphWeights};
    use crate::components::activation::ActivationType;
    use crate::components::embedding::embedding_lookup;
    use crate::components::matmul::{evaluate_mle_pub, M31Matrix};
    use crate::crypto::poseidon_channel::PoseidonChannel;
    use crate::gadgets::quantize::{dequantize_value, quantize_value, QuantParams, QuantStrategy};
    use crate::gkr::circuit::LayeredCircuit;
    use crate::gkr::prover::{
        prove_gkr, reduce_activation_layer_for_test, reduce_embedding_layer_for_test,
        reduce_layernorm_layer_for_test, reduce_layernorm_simd_for_test, reduce_mul_layer_for_test,
        reduce_quantize_layer_for_test,
    };
    use num_traits::Zero;
    use stwo::core::fields::m31::M31;

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(prev) = self.prev.as_ref() {
                std::env::set_var(self.key, prev);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }

    fn matmul_forward(a: &M31Matrix, b: &M31Matrix) -> M31Matrix {
        let mut c = M31Matrix::new(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut sum = M31::zero();
                for kk in 0..a.cols {
                    sum = sum + a.get(i, kk) * b.get(kk, j);
                }
                c.set(i, j, sum);
            }
        }
        c
    }

    fn apply_relu(input: &M31Matrix) -> M31Matrix {
        let half_p = M31::from((1u32 << 30) as u32);
        let mut output = M31Matrix::new(input.rows, input.cols);
        for i in 0..input.rows {
            for j in 0..input.cols {
                let v = input.get(i, j);
                // Simple ReLU in M31: keep if < p/2, else 0
                output.set(i, j, if v.0 <= half_p.0 { v } else { M31::zero() });
            }
        }
        output
    }

    #[test]
    fn test_prove_and_verify_single_matmul() {
        // Build a single-layer matmul graph: 2×4 @ 4×2
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(2);
        let graph = builder.build();

        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        // Create weights and execute forward pass
        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..8 {
            a.data[i] = M31::from((i + 1) as u32);
        }
        for i in 0..8 {
            b.data[i] = M31::from((i + 1) as u32);
        }
        let c = matmul_forward(&a, &b);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, b.clone());

        let execution = GraphExecution {
            intermediates: vec![(0, a.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        // Prove
        let mut prover_channel = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();

        assert_eq!(proof.layer_proofs.len(), 1);
        match &proof.layer_proofs[0] {
            LayerProof::MatMul { round_polys, .. } => {
                assert_eq!(round_polys.len(), 2); // log2(4) = 2
            }
            _ => panic!("expected MatMul proof"),
        }

        // Verify with fresh channel (same initial state)
        let mut verifier_channel = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &c, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_batched_rlc_direct_eval_requires_weights_and_verifies_with_weights() {
        // Build a single-layer matmul graph: 2×4 @ 4×2
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..8 {
            a.data[i] = M31::from((i + 1) as u32);
        }
        for i in 0..8 {
            b.data[i] = M31::from((i + 1) as u32);
        }
        let c = matmul_forward(&a, &b);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, b.clone());

        let execution = GraphExecution {
            intermediates: vec![(0, a.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let mut proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();

        // Convert to batched RLC direct-eval mode: no per-weight openings/commitments.
        proof.weight_opening_transcript_mode = WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1;
        proof.weight_openings.clear();
        proof.weight_commitments.clear();

        // Legacy verifier (without weights) must reject this mode.
        let mut verifier_channel_legacy = PoseidonChannel::new();
        let err = verify_gkr(&circuit, &proof, &c, &mut verifier_channel_legacy)
            .expect_err("batched RLC mode must require verifier-side weights");
        let msg = format!("{err}");
        assert!(
            msg.contains("requires verify_gkr_with_weights"),
            "unexpected error: {msg}"
        );

        // Weight-aware verifier must accept the same proof.
        let mut verifier_channel_weighted = PoseidonChannel::new();
        verify_gkr_with_weights(
            &circuit,
            &proof,
            &c,
            &weights,
            &mut verifier_channel_weighted,
        )
        .unwrap();
    }

    #[test]
    fn test_batched_rlc_direct_eval_tampered_claim_fails() {
        // Build a single-layer matmul graph: 2×4 @ 4×2
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..8 {
            a.data[i] = M31::from((i + 1) as u32);
        }
        for i in 0..8 {
            b.data[i] = M31::from((i + 1) as u32);
        }
        let c = matmul_forward(&a, &b);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, b.clone());

        let execution = GraphExecution {
            intermediates: vec![(0, a.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let mut proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();

        proof.weight_opening_transcript_mode = WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1;
        proof.weight_openings.clear();
        proof.weight_commitments.clear();
        proof.weight_claims[0].expected_value =
            proof.weight_claims[0].expected_value + SecureField::one();

        let mut verifier_channel = PoseidonChannel::new();
        let result = verify_gkr_with_weights(&circuit, &proof, &c, &weights, &mut verifier_channel);
        assert!(
            result.is_err(),
            "tampered batched RLC claim must fail weight-binding verification"
        );
    }

    #[test]
    fn test_batched_rlc_direct_eval_with_deferred_claims() {
        // Residual DAG: x -> MatMul(0) -> MatMul(1) -> Add(with skip from MatMul(0))
        let mut builder = GraphBuilder::new((1, 4));
        builder.linear(4);
        let residual = builder.fork();
        builder.linear(4);
        builder.add_from(residual);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let mut x = M31Matrix::new(1, 4);
        for i in 0..4 {
            x.data[i] = M31::from((i + 1) as u32);
        }
        let mut w0 = M31Matrix::new(4, 4);
        let mut w1 = M31Matrix::new(4, 4);
        for i in 0..16 {
            w0.data[i] = M31::from(((i % 7) + 1) as u32);
            w1.data[i] = M31::from((((i * 3) % 11) + 1) as u32);
        }

        let y0 = matmul_forward(&x, &w0);
        let y1 = matmul_forward(&y0, &w1);
        let mut out = M31Matrix::new(1, 4);
        for j in 0..4 {
            out.set(0, j, y0.get(0, j) + y1.get(0, j));
        }

        let mut weights = GraphWeights::new();
        weights.add_weight(0, w0);
        weights.add_weight(1, w1);

        let mut node_outputs = std::collections::HashMap::new();
        node_outputs.insert(0usize, y0.clone());
        node_outputs.insert(1usize, y1.clone());
        node_outputs.insert(2usize, out.clone());

        let execution = GraphExecution {
            // Input to each node: MatMul(0)=x, MatMul(1)=y0, Add(2)=y1
            intermediates: vec![(0, x.clone()), (1, y0.clone()), (2, y1.clone())],
            node_outputs,
            output: out.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let mut proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();
        assert!(
            !proof.deferred_proofs.is_empty(),
            "residual DAG should produce deferred proofs"
        );

        // Convert to batched RLC direct-eval mode and strip all per-weight openings.
        proof.weight_opening_transcript_mode = WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1;
        proof.weight_openings.clear();
        proof.weight_commitments.clear();
        for deferred in proof.deferred_proofs.iter_mut() {
            deferred.weight_commitment = starknet_ff::FieldElement::ZERO;
            deferred.weight_opening = crate::crypto::mle_opening::MleOpeningProof {
                intermediate_roots: Vec::new(),
                queries: Vec::new(),
                final_value: SecureField::zero(),
            };
        }

        let mut verifier_channel = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &out, &weights, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_aggregated_oracle_sumcheck_requires_binding_proof() {
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..8 {
            a.data[i] = M31::from((i + 1) as u32);
            b.data[i] = M31::from((i + 1) as u32);
        }
        let c = matmul_forward(&a, &b);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, b.clone());
        let execution = GraphExecution {
            intermediates: vec![(0, a.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let mut proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();
        proof.weight_opening_transcript_mode = WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
        proof.aggregated_binding = None;

        let mut verifier_channel = PoseidonChannel::new();
        let err = verify_gkr(&circuit, &proof, &c, &mut verifier_channel)
            .expect_err("mode4 must require aggregated binding proof");
        let msg = format!("{err}");
        assert!(
            msg.contains("requires aggregated_binding proof"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_aggregated_oracle_sumcheck_tampered_binding_fails() {
        let _binding_mode = EnvVarGuard::set("STWO_WEIGHT_BINDING", "aggregated");

        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..8 {
            a.data[i] = M31::from((i + 1) as u32);
            b.data[i] = M31::from((i + 1) as u32);
        }
        let c = matmul_forward(&a, &b);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, b.clone());
        let execution = GraphExecution {
            intermediates: vec![(0, a.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let mut proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();
        assert_eq!(
            proof.weight_opening_transcript_mode,
            WeightOpeningTranscriptMode::AggregatedOracleSumcheck
        );
        assert!(
            proof.aggregated_binding.is_some(),
            "mode4 prover must include aggregated binding proof"
        );

        let mut verifier_channel_ok = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &c, &mut verifier_channel_ok)
            .expect("mode4 proof with valid binding should verify");

        if let Some(binding) = proof.aggregated_binding.as_mut() {
            binding.oracle_eval_at_s = binding.oracle_eval_at_s + SecureField::one();
        }

        let mut verifier_channel_bad = PoseidonChannel::new();
        let err = verify_gkr(&circuit, &proof, &c, &mut verifier_channel_bad)
            .expect_err("tampered mode4 binding must fail verification");
        let msg = format!("{err}");
        assert!(
            msg.contains("aggregated oracle sumcheck weight binding verification failed"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_aggregated_oracle_sumcheck_cross_mode_confusion_fails() {
        let _binding_mode = EnvVarGuard::set("STWO_WEIGHT_BINDING", "aggregated");

        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..8 {
            a.data[i] = M31::from((i + 1) as u32);
            b.data[i] = M31::from((i + 1) as u32);
        }
        let c = matmul_forward(&a, &b);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, b.clone());
        let execution = GraphExecution {
            intermediates: vec![(0, a.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let mut proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();
        assert_eq!(
            proof.weight_opening_transcript_mode,
            WeightOpeningTranscriptMode::AggregatedOracleSumcheck
        );
        assert!(proof.aggregated_binding.is_some());

        // Adversarial relabeling: try to reinterpret mode4 proof as mode3 path.
        proof.weight_opening_transcript_mode =
            WeightOpeningTranscriptMode::AggregatedOpeningsV4Experimental;

        let mut verifier_channel = PoseidonChannel::new();
        let err = verify_gkr(&circuit, &proof, &c, &mut verifier_channel)
            .expect_err("mode relabeling must fail verification");
        let msg = format!("{err}");
        assert!(
            msg.contains("weight_openings count"),
            "unexpected cross-mode error: {msg}"
        );
    }

    #[test]
    fn test_tampered_round_poly_fails() {
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..8 {
            a.data[i] = M31::from((i + 1) as u32);
        }
        for i in 0..8 {
            b.data[i] = M31::from((i + 1) as u32);
        }
        let c = matmul_forward(&a, &b);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, b.clone());

        let execution = GraphExecution {
            intermediates: vec![(0, a.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let mut proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();

        // Tamper with first round poly
        if let LayerProof::MatMul { round_polys, .. } = &mut proof.layer_proofs[0] {
            round_polys[0].c0 = round_polys[0].c0 + SecureField::one();
        }

        let mut verifier_channel = PoseidonChannel::new();
        let result = verify_gkr(&circuit, &proof, &c, &mut verifier_channel);
        assert!(result.is_err(), "tampered proof should fail verification");
    }

    #[test]
    fn test_prove_and_verify_mlp() {
        // 2-layer MLP: MatMul → ReLU → MatMul
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(4);
        builder.activation(ActivationType::ReLU);
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        // Weights
        let mut w1 = M31Matrix::new(4, 4);
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..16 {
            w1.data[i] = M31::from((i % 5 + 1) as u32);
        }
        for i in 0..8 {
            w2.data[i] = M31::from((i % 3 + 1) as u32);
        }

        // Forward pass
        let input = {
            let mut m = M31Matrix::new(2, 4);
            for i in 0..8 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };
        let hidden = matmul_forward(&input, &w1);
        let activated = apply_relu(&hidden);
        let output = matmul_forward(&activated, &w2);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, w1);
        weights.add_weight(2, w2);

        // Intermediates: node_id → INPUT to that node
        // Node 0 (first matmul): input is model input
        // Node 1 (activation): input is hidden (pre-activation)
        // Node 2 (second matmul): input is activated (post-activation)
        let execution = GraphExecution {
            intermediates: vec![
                (0, input.clone()),
                (1, hidden.clone()),
                (2, activated.clone()),
            ],
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        // Prove
        let mut prover_channel = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();
        assert_eq!(proof.layer_proofs.len(), 3); // MatMul + Activation + MatMul

        // Verify
        let mut verifier_channel = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &output, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_tampered_final_eval_fails() {
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let mut a = M31Matrix::new(2, 4);
        let mut b = M31Matrix::new(4, 2);
        for i in 0..8 {
            a.data[i] = M31::from((i + 1) as u32);
        }
        for i in 0..8 {
            b.data[i] = M31::from((i + 1) as u32);
        }
        let c = matmul_forward(&a, &b);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, b.clone());

        let execution = GraphExecution {
            intermediates: vec![(0, a.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let mut proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();

        // Tamper with final_a_eval
        if let LayerProof::MatMul { final_a_eval, .. } = &mut proof.layer_proofs[0] {
            *final_a_eval = *final_a_eval + SecureField::one();
        }

        let mut verifier_channel = PoseidonChannel::new();
        let result = verify_gkr(&circuit, &proof, &c, &mut verifier_channel);
        assert!(result.is_err(), "tampered final eval should fail");
    }

    #[test]
    fn test_mul_eq_sumcheck_prove_and_verify() {
        // Test the Mul layer's eq-sumcheck roundtrip.
        let a_vals: Vec<SecureField> = (2..=5).map(|x| SecureField::from(M31::from(x))).collect();
        let b_vals: Vec<SecureField> = (1..=4).map(|x| SecureField::from(M31::from(x))).collect();
        let c_vals: Vec<SecureField> = a_vals.iter().zip(&b_vals).map(|(&a, &b)| a * b).collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xBEEF);
        let r = prover_channel.draw_qm31s(2);

        let claimed = evaluate_mle_pub(&c_vals, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (proof, _next_claim) =
            reduce_mul_layer_for_test(&output_claim, &a_vals, &b_vals, &mut prover_channel)
                .unwrap();

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0xBEEF);
        let _r_v = verifier_channel.draw_qm31s(2);

        match &proof {
            LayerProof::Mul {
                eq_round_polys,
                lhs_eval,
                rhs_eval,
            } => {
                let result = verify_mul_reduction(
                    &output_claim,
                    eq_round_polys,
                    *lhs_eval,
                    *rhs_eval,
                    0,
                    &mut verifier_channel,
                );
                assert!(
                    result.is_ok(),
                    "valid mul proof should verify: {:?}",
                    result.err()
                );
            }
            _ => panic!("expected Mul proof"),
        }
    }

    #[test]
    fn test_mul_tampered_round_poly_fails() {
        let a_vals: Vec<SecureField> = (2..=5).map(|x| SecureField::from(M31::from(x))).collect();
        let b_vals: Vec<SecureField> = (1..=4).map(|x| SecureField::from(M31::from(x))).collect();
        let c_vals: Vec<SecureField> = a_vals.iter().zip(&b_vals).map(|(&a, &b)| a * b).collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xCAFE);
        let r = prover_channel.draw_qm31s(2);
        let claimed = evaluate_mle_pub(&c_vals, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) =
            reduce_mul_layer_for_test(&output_claim, &a_vals, &b_vals, &mut prover_channel)
                .unwrap();

        if let LayerProof::Mul { eq_round_polys, .. } = &mut proof {
            eq_round_polys[0].c0 = eq_round_polys[0].c0 + SecureField::one();
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0xCAFE);
        let _r_v = verifier_channel.draw_qm31s(2);

        match &proof {
            LayerProof::Mul {
                eq_round_polys,
                lhs_eval,
                rhs_eval,
            } => {
                let result = verify_mul_reduction(
                    &output_claim,
                    eq_round_polys,
                    *lhs_eval,
                    *rhs_eval,
                    0,
                    &mut verifier_channel,
                );
                assert!(
                    result.is_err(),
                    "tampered mul proof should fail verification"
                );
            }
            _ => panic!("expected Mul proof"),
        }
    }

    #[test]
    fn test_mul_tampered_final_eval_fails() {
        let a_vals: Vec<SecureField> = (10..=13).map(|x| SecureField::from(M31::from(x))).collect();
        let b_vals: Vec<SecureField> = (5..=8).map(|x| SecureField::from(M31::from(x))).collect();
        let c_vals: Vec<SecureField> = a_vals.iter().zip(&b_vals).map(|(&a, &b)| a * b).collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xDEAD);
        let r = prover_channel.draw_qm31s(2);
        let claimed = evaluate_mle_pub(&c_vals, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) =
            reduce_mul_layer_for_test(&output_claim, &a_vals, &b_vals, &mut prover_channel)
                .unwrap();

        if let LayerProof::Mul { lhs_eval, .. } = &mut proof {
            *lhs_eval = *lhs_eval + SecureField::one();
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0xDEAD);
        let _r_v = verifier_channel.draw_qm31s(2);

        match &proof {
            LayerProof::Mul {
                eq_round_polys,
                lhs_eval,
                rhs_eval,
            } => {
                let result = verify_mul_reduction(
                    &output_claim,
                    eq_round_polys,
                    *lhs_eval,
                    *rhs_eval,
                    0,
                    &mut verifier_channel,
                );
                assert!(
                    result.is_err(),
                    "tampered lhs_eval should fail verification"
                );
            }
            _ => panic!("expected Mul proof"),
        }
    }

    #[test]
    fn test_eq_eval_correctness() {
        let zero = vec![SecureField::zero()];
        let one = vec![SecureField::one()];
        assert_eq!(compute_eq_eval(&zero, &zero), SecureField::one());
        assert_eq!(compute_eq_eval(&zero, &one), SecureField::zero());
        assert_eq!(compute_eq_eval(&one, &zero), SecureField::zero());
        assert_eq!(compute_eq_eval(&one, &one), SecureField::one());

        let p01 = vec![SecureField::zero(), SecureField::one()];
        let p10 = vec![SecureField::one(), SecureField::zero()];
        assert_eq!(compute_eq_eval(&p01, &p01), SecureField::one());
        assert_eq!(compute_eq_eval(&p01, &p10), SecureField::zero());

        let r = vec![
            SecureField::from(M31::from(3u32)),
            SecureField::from(M31::from(7u32)),
        ];
        let x00 = vec![SecureField::zero(), SecureField::zero()];
        let expected = (SecureField::one() - r[0]) * (SecureField::one() - r[1]);
        assert_eq!(compute_eq_eval(&r, &x00), expected);
    }

    // ===== Activation LogUp Tests =====

    #[test]
    fn test_activation_logup_prove_and_verify() {
        // Build a 2×2 input with small values (all < p/2, so ReLU keeps them).
        // This tests the basic roundtrip: prove → verify succeeds.
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(5u32));
            m.set(0, 1, M31::from(10u32));
            m.set(1, 0, M31::from(3u32));
            m.set(1, 1, M31::from(7u32));
            m
        };

        // Build claim on the activation output
        let activation_fn = ActivationType::ReLU.as_fn();
        let padded = pad_matrix_pow2(&input);
        let n = padded.rows * padded.cols;
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(activation_fn(v)))
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xAC01);
        let r = prover_channel.draw_qm31s(2); // 4 elements → 2 vars
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        // Prove
        let (proof, _next_claim) = reduce_activation_layer_for_test(
            &output_claim,
            &input,
            ActivationType::ReLU,
            &mut prover_channel,
        )
        .unwrap();

        // Verify with fresh channel
        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0xAC01);
        let _r_v = verifier_channel.draw_qm31s(2);

        match &proof {
            LayerProof::Activation {
                activation_type,
                logup_proof,
                input_eval,
                output_eval,
                table_commitment,
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0,
                    &mut verifier_channel,
                );
                assert!(
                    result.is_ok(),
                    "valid activation proof should verify: {:?}",
                    result.err()
                );
            }
            _ => panic!("expected Activation proof"),
        }
    }

    #[test]
    fn test_activation_tampered_multiplicity_fails() {
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(1u32));
            m.set(0, 1, M31::from(2u32));
            m.set(1, 0, M31::from(3u32));
            m.set(1, 1, M31::from(4u32));
            m
        };

        let activation_fn = ActivationType::ReLU.as_fn();
        let padded = pad_matrix_pow2(&input);
        let n = padded.rows * padded.cols;
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(activation_fn(v)))
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xAC02);
        let r = prover_channel.draw_qm31s(2);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) = reduce_activation_layer_for_test(
            &output_claim,
            &input,
            ActivationType::ReLU,
            &mut prover_channel,
        )
        .unwrap();

        // Tamper with multiplicities
        if let LayerProof::Activation {
            logup_proof: Some(ref mut logup),
            ..
        } = &mut proof
        {
            if !logup.multiplicities.is_empty() {
                logup.multiplicities[0] = logup.multiplicities[0].wrapping_add(1);
            }
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0xAC02);
        let _r_v = verifier_channel.draw_qm31s(2);

        match &proof {
            LayerProof::Activation {
                activation_type,
                logup_proof,
                input_eval,
                output_eval,
                table_commitment,
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0,
                    &mut verifier_channel,
                );
                assert!(result.is_err(), "tampered multiplicities should fail");
            }
            _ => panic!("expected Activation proof"),
        }
    }

    #[test]
    fn test_activation_tampered_eq_round_poly_fails() {
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(2u32));
            m.set(0, 1, M31::from(4u32));
            m.set(1, 0, M31::from(6u32));
            m.set(1, 1, M31::from(8u32));
            m
        };

        let activation_fn = ActivationType::ReLU.as_fn();
        let padded = pad_matrix_pow2(&input);
        let n = padded.rows * padded.cols;
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(activation_fn(v)))
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xAC03);
        let r = prover_channel.draw_qm31s(2);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) = reduce_activation_layer_for_test(
            &output_claim,
            &input,
            ActivationType::ReLU,
            &mut prover_channel,
        )
        .unwrap();

        // Tamper with first eq-sumcheck round poly
        if let LayerProof::Activation {
            logup_proof: Some(ref mut logup),
            ..
        } = &mut proof
        {
            if !logup.eq_round_polys.is_empty() {
                logup.eq_round_polys[0].c0 = logup.eq_round_polys[0].c0 + SecureField::one();
            }
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0xAC03);
        let _r_v = verifier_channel.draw_qm31s(2);

        match &proof {
            LayerProof::Activation {
                activation_type,
                logup_proof,
                input_eval,
                output_eval,
                table_commitment,
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0,
                    &mut verifier_channel,
                );
                assert!(result.is_err(), "tampered eq round poly should fail");
            }
            _ => panic!("expected Activation proof"),
        }
    }

    #[test]
    fn test_activation_tampered_final_eval_fails() {
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(10u32));
            m.set(0, 1, M31::from(20u32));
            m.set(1, 0, M31::from(30u32));
            m.set(1, 1, M31::from(40u32));
            m
        };

        let activation_fn = ActivationType::ReLU.as_fn();
        let padded = pad_matrix_pow2(&input);
        let n = padded.rows * padded.cols;
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(activation_fn(v)))
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xAC04);
        let r = prover_channel.draw_qm31s(2);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) = reduce_activation_layer_for_test(
            &output_claim,
            &input,
            ActivationType::ReLU,
            &mut prover_channel,
        )
        .unwrap();

        // Tamper with final w_eval
        if let LayerProof::Activation {
            logup_proof: Some(ref mut logup),
            ..
        } = &mut proof
        {
            logup.final_evals.0 = logup.final_evals.0 + SecureField::one();
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0xAC04);
        let _r_v = verifier_channel.draw_qm31s(2);

        match &proof {
            LayerProof::Activation {
                activation_type,
                logup_proof,
                input_eval,
                output_eval,
                table_commitment,
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0,
                    &mut verifier_channel,
                );
                assert!(result.is_err(), "tampered final_evals should fail");
            }
            _ => panic!("expected Activation proof"),
        }
    }

    // ===== LayerNorm Tests =====

    /// Helper to build a LayerNorm forward pass for testing.
    fn layernorm_forward(input: &M31Matrix, dim: usize) -> M31Matrix {
        use crate::components::layernorm::{build_rsqrt_table, LayerNormConfig};
        let config = LayerNormConfig::new(dim);
        let rsqrt_table = build_rsqrt_table(config.rsqrt_table_log_size);
        let padded = pad_matrix_pow2(input);
        let n_active = dim.min(padded.cols);
        let p: u64 = (1u64 << 31) - 1;

        // Modular inverse of n_active in M31
        let inv_n = {
            let mut result: u64 = 1;
            let mut base = n_active as u64 % p;
            let mut exp = p - 2;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = result * base % p;
                }
                base = base * base % p;
                exp >>= 1;
            }
            M31::from(result as u32)
        };

        let mut output = M31Matrix::new(padded.rows, padded.cols);
        for row in 0..padded.rows {
            let mut sum = M31::from(0u32);
            for col in 0..n_active {
                sum = sum + padded.get(row, col);
            }
            let mean = sum * inv_n;
            let mut var_sum = M31::from(0u32);
            for col in 0..n_active {
                let diff = padded.get(row, col) - mean;
                var_sum = var_sum + diff * diff;
            }
            // Reduce variance to table range (must match prover).
            let variance_raw = var_sum * inv_n;
            let variance = M31::from(variance_raw.0 & ((1u32 << config.rsqrt_table_log_size) - 1));
            let rsqrt = rsqrt_table
                .lookup(variance)
                .expect("variance reduced to table range");
            for col in 0..padded.cols {
                if col < n_active {
                    let centered = padded.get(row, col) - mean;
                    output.set(row, col, centered * rsqrt);
                } else {
                    output.set(row, col, padded.get(row, col));
                }
            }
        }
        output
    }

    #[test]
    fn test_layernorm_prove_and_verify() {
        let input = {
            let mut m = M31Matrix::new(2, 4);
            m.set(0, 0, M31::from(10u32));
            m.set(0, 1, M31::from(20u32));
            m.set(0, 2, M31::from(30u32));
            m.set(0, 3, M31::from(40u32));
            m.set(1, 0, M31::from(5u32));
            m.set(1, 1, M31::from(15u32));
            m.set(1, 2, M31::from(25u32));
            m.set(1, 3, M31::from(35u32));
            m
        };
        let dim = 4;

        let output = layernorm_forward(&input, dim);
        let output_padded = pad_matrix_pow2(&output);
        let n = output_padded.rows * output_padded.cols;
        let output_mle: Vec<SecureField> = output_padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(v))
            .collect();
        let num_vars = output_mle.len().ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x4C4E01); // "LN01" seed
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        // Prove
        let (proof, _next_claim) =
            reduce_layernorm_layer_for_test(&output_claim, &input, dim, &mut prover_channel)
                .unwrap();

        // Verify
        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x4C4E01);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::LayerNorm {
                logup_proof,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *mean,
                    *rsqrt_var,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    &mut verifier_channel,
                );
                assert!(
                    result.is_ok(),
                    "valid layernorm proof should verify: {:?}",
                    result.err()
                );
            }
            _ => panic!("expected LayerNorm proof"),
        }
    }

    #[test]
    fn test_layernorm_tampered_linear_round_poly_fails() {
        // Row values must produce integer mean and variance in M31.
        // Pattern: [a-3d, a-d, a+d, a+3d] → mean=a, variance=5d²
        let input = {
            let mut m = M31Matrix::new(2, 4);
            // Row 0: [7, 9, 11, 13] → mean=10, variance=5
            m.set(0, 0, M31::from(7u32));
            m.set(0, 1, M31::from(9u32));
            m.set(0, 2, M31::from(11u32));
            m.set(0, 3, M31::from(13u32));
            // Row 1: [17, 19, 21, 23] → mean=20, variance=5
            m.set(1, 0, M31::from(17u32));
            m.set(1, 1, M31::from(19u32));
            m.set(1, 2, M31::from(21u32));
            m.set(1, 3, M31::from(23u32));
            m
        };
        let dim = 4;

        let output = layernorm_forward(&input, dim);
        let output_padded = pad_matrix_pow2(&output);
        let n = output_padded.rows * output_padded.cols;
        let output_mle: Vec<SecureField> = output_padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(v))
            .collect();
        let num_vars = output_mle.len().ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x4C4E02);
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) =
            reduce_layernorm_layer_for_test(&output_claim, &input, dim, &mut prover_channel)
                .unwrap();

        // Tamper with first linear round poly
        if let LayerProof::LayerNorm {
            ref mut linear_round_polys,
            ..
        } = &mut proof
        {
            if !linear_round_polys.is_empty() {
                linear_round_polys[0].c0 = linear_round_polys[0].c0 + SecureField::one();
            }
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x4C4E02);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::LayerNorm {
                logup_proof,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *mean,
                    *rsqrt_var,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    &mut verifier_channel,
                );
                assert!(result.is_err(), "tampered linear round poly should fail");
            }
            _ => panic!("expected LayerNorm proof"),
        }
    }

    #[test]
    fn test_layernorm_tampered_logup_multiplicity_fails() {
        // Row values: [a-3d, a-d, a+d, a+3d] → mean=a, variance=5d²
        let input = {
            let mut m = M31Matrix::new(2, 4);
            // Row 0: [4, 8, 12, 16] → mean=10, variance=20
            m.set(0, 0, M31::from(4u32));
            m.set(0, 1, M31::from(8u32));
            m.set(0, 2, M31::from(12u32));
            m.set(0, 3, M31::from(16u32));
            // Row 1: [14, 18, 22, 26] → mean=20, variance=20
            m.set(1, 0, M31::from(14u32));
            m.set(1, 1, M31::from(18u32));
            m.set(1, 2, M31::from(22u32));
            m.set(1, 3, M31::from(26u32));
            m
        };
        let dim = 4;

        let output = layernorm_forward(&input, dim);
        let output_padded = pad_matrix_pow2(&output);
        let n = output_padded.rows * output_padded.cols;
        let output_mle: Vec<SecureField> = output_padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(v))
            .collect();
        let num_vars = output_mle.len().ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x4C4E03);
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) =
            reduce_layernorm_layer_for_test(&output_claim, &input, dim, &mut prover_channel)
                .unwrap();

        // Tamper with LogUp multiplicities
        if let LayerProof::LayerNorm {
            logup_proof: Some(ref mut logup),
            ..
        } = &mut proof
        {
            if !logup.multiplicities.is_empty() {
                logup.multiplicities[0] = logup.multiplicities[0].wrapping_add(1);
            }
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x4C4E03);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::LayerNorm {
                logup_proof,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *mean,
                    *rsqrt_var,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    &mut verifier_channel,
                );
                assert!(result.is_err(), "tampered LogUp multiplicity should fail");
            }
            _ => panic!("expected LayerNorm proof"),
        }
    }

    #[test]
    fn test_layernorm_tampered_linear_final_eval_fails() {
        // Row values: [a-3d, a-d, a+d, a+3d] → mean=a, variance=5d²
        let input = {
            let mut m = M31Matrix::new(2, 4);
            // Row 0: [22, 26, 30, 34] → mean=28, variance=20
            m.set(0, 0, M31::from(22u32));
            m.set(0, 1, M31::from(26u32));
            m.set(0, 2, M31::from(30u32));
            m.set(0, 3, M31::from(34u32));
            // Row 1: [32, 36, 40, 44] → mean=38, variance=20
            m.set(1, 0, M31::from(32u32));
            m.set(1, 1, M31::from(36u32));
            m.set(1, 2, M31::from(40u32));
            m.set(1, 3, M31::from(44u32));
            m
        };
        let dim = 4;

        let output = layernorm_forward(&input, dim);
        let output_padded = pad_matrix_pow2(&output);
        let n = output_padded.rows * output_padded.cols;
        let output_mle: Vec<SecureField> = output_padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(v))
            .collect();
        let num_vars = output_mle.len().ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x4C4E04);
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) =
            reduce_layernorm_layer_for_test(&output_claim, &input, dim, &mut prover_channel)
                .unwrap();

        // Tamper with centered_final (first element of linear_final_evals)
        if let LayerProof::LayerNorm {
            ref mut linear_final_evals,
            ..
        } = &mut proof
        {
            linear_final_evals.0 = linear_final_evals.0 + SecureField::one();
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x4C4E04);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::LayerNorm {
                logup_proof,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *mean,
                    *rsqrt_var,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    &mut verifier_channel,
                );
                assert!(result.is_err(), "tampered linear final eval should fail");
            }
            _ => panic!("expected LayerNorm proof"),
        }
    }

    // ===== SIMD LayerNorm (combined-product, logup_proof: None) Tests =====

    /// Helper: build per-block LayerNorm MLEs (product, mean, rsqrt, input).
    fn build_layernorm_block_mles(
        input: &M31Matrix,
        dim: usize,
    ) -> (
        Vec<SecureField>,
        Vec<SecureField>,
        Vec<SecureField>,
        Vec<SecureField>,
    ) {
        use crate::components::layernorm::{build_rsqrt_table, LayerNormConfig};

        let config = LayerNormConfig::new(dim);
        let rsqrt_table = build_rsqrt_table(config.rsqrt_table_log_size);
        let padded = pad_matrix_pow2(input);
        let n = padded.rows * padded.cols;
        let cols = padded.cols;
        let n_active = dim.min(cols);
        let p: u64 = (1u64 << 31) - 1;

        let inv_n = {
            let mut result: u64 = 1;
            let mut base = n_active as u64 % p;
            let mut exp = p - 2;
            while exp > 0 {
                if exp & 1 == 1 {
                    result = result * base % p;
                }
                base = base * base % p;
                exp >>= 1;
            }
            M31::from(result as u32)
        };

        let mut product_mle = vec![SecureField::zero(); n];
        let mut mean_mle = vec![SecureField::zero(); n];
        let mut rsqrt_mle = vec![SecureField::zero(); n];
        let input_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(v))
            .collect();

        for row in 0..padded.rows {
            let mut sum = M31::from(0u32);
            for col in 0..n_active {
                sum = sum + padded.get(row, col);
            }
            let mean = sum * inv_n;

            let mut var_sum = M31::from(0u32);
            for col in 0..n_active {
                let diff = padded.get(row, col) - mean;
                var_sum = var_sum + diff * diff;
            }
            // Reduce variance to table range (must match prover).
            let variance_raw = var_sum * inv_n;
            let variance = M31::from(variance_raw.0 & ((1u32 << config.rsqrt_table_log_size) - 1));
            let rsqrt = rsqrt_table
                .lookup(variance)
                .expect("variance reduced to table range");

            let mean_sf = SecureField::from(mean);
            let rsqrt_sf = SecureField::from(rsqrt);

            for col in 0..cols {
                let idx = row * cols + col;
                mean_mle[idx] = mean_sf;
                rsqrt_mle[idx] = rsqrt_sf;

                if col < n_active {
                    let centered = padded.get(row, col) - mean;
                    product_mle[idx] = SecureField::from(centered * rsqrt);
                } else {
                    product_mle[idx] = SecureField::from(padded.get(row, col));
                }
            }
        }

        (product_mle, mean_mle, rsqrt_mle, input_mle)
    }

    #[test]
    fn test_layernorm_simd_combined_product_verify() {
        // Block 0
        let input_0 = {
            let mut m = M31Matrix::new(2, 4);
            m.set(0, 0, M31::from(10u32));
            m.set(0, 1, M31::from(20u32));
            m.set(0, 2, M31::from(30u32));
            m.set(0, 3, M31::from(40u32));
            m.set(1, 0, M31::from(5u32));
            m.set(1, 1, M31::from(15u32));
            m.set(1, 2, M31::from(25u32));
            m.set(1, 3, M31::from(35u32));
            m
        };
        // Block 1 (different values)
        let input_1 = {
            let mut m = M31Matrix::new(2, 4);
            m.set(0, 0, M31::from(7u32));
            m.set(0, 1, M31::from(9u32));
            m.set(0, 2, M31::from(11u32));
            m.set(0, 3, M31::from(13u32));
            m.set(1, 0, M31::from(17u32));
            m.set(1, 1, M31::from(19u32));
            m.set(1, 2, M31::from(21u32));
            m.set(1, 3, M31::from(23u32));
            m
        };
        let dim = 4;

        let (prod_0, mean_0, rsqrt_0, input_0_mle) = build_layernorm_block_mles(&input_0, dim);
        let (prod_1, mean_1, rsqrt_1, input_1_mle) = build_layernorm_block_mles(&input_1, dim);

        // Random block weights (simulating Lagrange basis)
        let w0 = SecureField::from(M31::from(7u32));
        let w1 = SecureField::from(M31::from(13u32));

        // Combine MLEs: Σ w_b · mle_b
        let n = prod_0.len();
        let combined_product: Vec<SecureField> =
            (0..n).map(|i| w0 * prod_0[i] + w1 * prod_1[i]).collect();
        let combined_mean: Vec<SecureField> =
            (0..n).map(|i| w0 * mean_0[i] + w1 * mean_1[i]).collect();
        let combined_rsqrt: Vec<SecureField> =
            (0..n).map(|i| w0 * rsqrt_0[i] + w1 * rsqrt_1[i]).collect();
        let combined_input: Vec<SecureField> = (0..n)
            .map(|i| w0 * input_0_mle[i] + w1 * input_1_mle[i])
            .collect();

        // Build claim on combined output (= combined_product on boolean hypercube)
        let num_vars = combined_product.len().ilog2() as usize;
        let mut setup_channel = PoseidonChannel::new();
        setup_channel.mix_u64(0x534D4C4E01);
        let r = setup_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&combined_product, &r);
        let output_claim = GKRClaim {
            point: r,
            value: claimed,
        };

        // Prove (CPU path via test wrapper)
        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x534D4C4E01);
        let _r_p = prover_channel.draw_qm31s(num_vars);

        let (proof, _next_claim) = reduce_layernorm_simd_for_test(
            &output_claim,
            combined_product,
            &combined_mean,
            &combined_rsqrt,
            &combined_input,
            dim,
            &mut prover_channel,
        )
        .unwrap();

        // Verify
        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x534D4C4E01);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::LayerNorm {
                logup_proof,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
            } => {
                assert!(
                    logup_proof.is_none(),
                    "SIMD layernorm should have no LogUp proof"
                );
                assert!(
                    *simd_combined,
                    "SIMD layernorm should have simd_combined=true"
                );
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *mean,
                    *rsqrt_var,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    &mut verifier_channel,
                );
                assert!(
                    result.is_ok(),
                    "SIMD combined-product layernorm should verify: {:?}",
                    result.err()
                );
            }
            _ => panic!("expected LayerNorm proof"),
        }
    }

    #[test]
    fn test_layernorm_simd_tampered_product_fails() {
        // Single block with weight=1 (simplest SIMD case)
        let input = {
            let mut m = M31Matrix::new(2, 4);
            m.set(0, 0, M31::from(10u32));
            m.set(0, 1, M31::from(20u32));
            m.set(0, 2, M31::from(30u32));
            m.set(0, 3, M31::from(40u32));
            m.set(1, 0, M31::from(5u32));
            m.set(1, 1, M31::from(15u32));
            m.set(1, 2, M31::from(25u32));
            m.set(1, 3, M31::from(35u32));
            m
        };
        let dim = 4;

        let (combined_product, combined_mean, combined_rsqrt, combined_input) =
            build_layernorm_block_mles(&input, dim);

        let num_vars = combined_product.len().ilog2() as usize;
        let mut setup_channel = PoseidonChannel::new();
        setup_channel.mix_u64(0x534D4C4E02);
        let r = setup_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&combined_product, &r);
        let output_claim = GKRClaim {
            point: r,
            value: claimed,
        };

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x534D4C4E02);
        let _r_p = prover_channel.draw_qm31s(num_vars);

        let (mut proof, _) = reduce_layernorm_simd_for_test(
            &output_claim,
            combined_product,
            &combined_mean,
            &combined_rsqrt,
            &combined_input,
            dim,
            &mut prover_channel,
        )
        .unwrap();

        // Tamper with a round poly coefficient
        if let LayerProof::LayerNorm {
            ref mut linear_round_polys,
            ..
        } = &mut proof
        {
            linear_round_polys[0].c1 = linear_round_polys[0].c1 + SecureField::one();
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x534D4C4E02);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::LayerNorm {
                logup_proof,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *mean,
                    *rsqrt_var,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    &mut verifier_channel,
                );
                assert!(result.is_err(), "tampered SIMD layernorm proof should fail");
            }
            _ => panic!("expected LayerNorm proof"),
        }
    }

    // === Dequantize LogUp Tests ===

    #[test]
    fn test_dequantize_logup_prove_and_verify() {
        use crate::gadgets::quantize::{
            dequantize_value, quantize_value, QuantParams, QuantStrategy,
        };

        // Build a 2×2 input with small quantized values (INT4 range: 0..15)
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(2u32));
            m.set(0, 1, M31::from(5u32));
            m.set(1, 0, M31::from(0u32));
            m.set(1, 1, M31::from(14u32));
            m
        };

        let params = QuantParams {
            strategy: QuantStrategy::Symmetric4,
            scale: 1.0 / 7.0,
            zero_point: 7,
            bits: 4,
        };

        // Compute output MLE (dequantized values as M31)
        let direct_params = QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 31,
        };
        let padded = pad_matrix_pow2(&input);
        let n = padded.rows * padded.cols;
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| {
                let f = dequantize_value(v, &params);
                SecureField::from(quantize_value(f, &direct_params))
            })
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xDE01);
        let r = prover_channel.draw_qm31s(2);
        let claimed = evaluate_mle(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        // Prove
        let (proof, _next_claim) = crate::gkr::prover::reduce_dequantize_layer_for_test(
            &output_claim,
            &input,
            &params,
            &mut prover_channel,
        )
        .unwrap();

        // Verify with fresh channel (same seed)
        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0xDE01);
        let _r_v = verifier_channel.draw_qm31s(2);

        match &proof {
            LayerProof::Dequantize {
                logup_proof,
                input_eval,
                output_eval,
                table_commitment,
            } => {
                let result = verify_dequantize_reduction(
                    &output_claim,
                    &params,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0,
                    &mut verifier_channel,
                );
                assert!(
                    result.is_ok(),
                    "valid dequantize proof should verify: {:?}",
                    result.err()
                );
            }
            _ => panic!("expected Dequantize proof"),
        }
    }

    #[test]
    fn test_dequantize_tampered_multiplicity_fails() {
        use crate::gadgets::quantize::{
            dequantize_value, quantize_value, QuantParams, QuantStrategy,
        };

        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(1u32));
            m.set(0, 1, M31::from(3u32));
            m.set(1, 0, M31::from(7u32));
            m.set(1, 1, M31::from(12u32));
            m
        };

        let params = QuantParams {
            strategy: QuantStrategy::Symmetric4,
            scale: 1.0 / 7.0,
            zero_point: 7,
            bits: 4,
        };

        let direct_params = QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 31,
        };
        let padded = pad_matrix_pow2(&input);
        let n = padded.rows * padded.cols;
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| {
                let f = dequantize_value(v, &params);
                SecureField::from(quantize_value(f, &direct_params))
            })
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xDE02);
        let r = prover_channel.draw_qm31s(2);
        let claimed = evaluate_mle(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (proof, _) = crate::gkr::prover::reduce_dequantize_layer_for_test(
            &output_claim,
            &input,
            &params,
            &mut prover_channel,
        )
        .unwrap();

        // Tamper: flip a multiplicity
        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0xDE02);
        let _r_v = verifier_channel.draw_qm31s(2);

        match proof {
            LayerProof::Dequantize {
                logup_proof: Some(mut logup),
                input_eval,
                output_eval,
                table_commitment,
            } => {
                if !logup.multiplicities.is_empty() {
                    logup.multiplicities[0] = logup.multiplicities[0].wrapping_add(1);
                }
                let result = verify_dequantize_reduction(
                    &output_claim,
                    &params,
                    Some(&logup),
                    input_eval,
                    output_eval,
                    table_commitment,
                    0,
                    &mut verifier_channel,
                );
                assert!(result.is_err(), "tampered dequantize proof should fail");
            }
            _ => panic!("expected Dequantize proof with LogUp"),
        }
    }

    #[test]
    fn test_quantize_logup_prove_and_verify() {
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(0u32));
            m.set(0, 1, M31::from(42u32));
            m.set(1, 0, M31::from(100u32));
            m.set(1, 1, M31::from(42u32));
            m
        };

        let params = QuantParams {
            strategy: QuantStrategy::Symmetric8,
            scale: 1.0 / 127.0,
            zero_point: 127,
            bits: 8,
        };
        let direct_params = QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 31,
        };

        let padded = pad_matrix_pow2(&input);
        let n = padded.rows * padded.cols;
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| {
                let f = dequantize_value(v, &direct_params);
                SecureField::from(quantize_value(f, &params))
            })
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x51414E54); // deterministic test seed
        let r = prover_channel.draw_qm31s(2);
        let claimed = evaluate_mle(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (proof, _) =
            reduce_quantize_layer_for_test(&output_claim, &input, &params, &mut prover_channel)
                .unwrap();

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x51414E54);
        let _r_v = verifier_channel.draw_qm31s(2);

        match &proof {
            LayerProof::Quantize {
                logup_proof,
                input_eval,
                output_eval,
                table_inputs,
                table_outputs,
            } => {
                let result = verify_quantize_reduction(
                    &output_claim,
                    &params,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    table_inputs,
                    table_outputs,
                    0,
                    &mut verifier_channel,
                );
                assert!(
                    result.is_ok(),
                    "valid quantize proof should verify: {:?}",
                    result.err()
                );
            }
            _ => panic!("expected Quantize proof"),
        }
    }

    #[test]
    fn test_embedding_logup_prove_and_verify_with_weights() {
        // Token input [1, 2]
        let input = {
            let mut m = M31Matrix::new(1, 2);
            m.set(0, 0, M31::from(1u32));
            m.set(0, 1, M31::from(2u32));
            m
        };

        // 4x3 embedding table
        let mut table = M31Matrix::new(4, 3);
        for r in 0..4 {
            for c in 0..3 {
                table.set(r, c, M31::from((r * 10 + c + 1) as u32));
            }
        }

        let token_u32s: Vec<u32> = input.data.iter().map(|m| m.0).collect();
        let (output, _, _, _, _) = embedding_lookup(&token_u32s, &table);

        let padded = pad_matrix_pow2(&output);
        let n = padded.rows * padded.cols;
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(v))
            .collect();
        let num_vars = output_mle.len().ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x454D4254); // deterministic test seed
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (proof, _) = reduce_embedding_layer_for_test(
            &output_claim,
            &input,
            &output,
            &table,
            &mut prover_channel,
        )
        .unwrap();

        let mut weights = GraphWeights::new();
        let node_id = 7usize;
        weights.add_weight(node_id, table.clone());

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x454D4254);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::Embedding {
                logup_proof,
                input_eval,
                output_eval,
                input_num_vars,
            } => {
                let result = verify_embedding_reduction(
                    &output_claim,
                    table.rows,
                    table.cols,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *input_num_vars,
                    node_id,
                    0,
                    &mut verifier_channel,
                    Some(&weights),
                );
                assert!(
                    result.is_ok(),
                    "valid embedding proof should verify: {:?}",
                    result.err()
                );
            }
            _ => panic!("expected Embedding proof"),
        }
    }

    // ===== Multi-Layer Chain Tests (cross-layer channel sync) =====

    /// Helper: compute matmul C = A * B in M31.
    fn matmul_m31(a: &M31Matrix, b: &M31Matrix) -> M31Matrix {
        assert_eq!(a.cols, b.rows, "matmul dimension mismatch");
        let mut c = M31Matrix::new(a.rows, b.cols);
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut sum = M31::zero();
                for k in 0..a.cols {
                    sum = sum + a.get(i, k) * b.get(k, j);
                }
                c.set(i, j, sum);
            }
        }
        c
    }

    /// Helper: apply element-wise ReLU in M31.
    fn relu_m31(input: &M31Matrix) -> M31Matrix {
        let half_p = M31::from(1u32 << 30);
        let mut output = M31Matrix::new(input.rows, input.cols);
        for i in 0..input.rows {
            for j in 0..input.cols {
                let v = input.get(i, j);
                output.set(i, j, if v.0 <= half_p.0 { v } else { M31::zero() });
            }
        }
        output
    }

    /// Helper: apply element-wise GELU approximation in M31.
    fn gelu_m31(input: &M31Matrix) -> M31Matrix {
        let half_p = M31::from(1u32 << 30);
        let mut output = M31Matrix::new(input.rows, input.cols);
        for i in 0..input.rows {
            for j in 0..input.cols {
                let v = input.get(i, j);
                // Simple M31 GELU: keep if < p/2, else 0 (same as ReLU for small values)
                output.set(i, j, if v.0 <= half_p.0 { v } else { M31::zero() });
            }
        }
        output
    }

    /// Helper: compute LayerNorm forward pass (re-uses the test module's layernorm_forward).
    fn ln_forward(input: &M31Matrix, dim: usize) -> M31Matrix {
        layernorm_forward(input, dim)
    }

    #[test]
    fn test_matmul_matmul_chain() {
        // MatMul(2x4 @ 4x4) → MatMul(2x4 @ 4x2)
        // Tests consecutive sumchecks with dimension change.
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(4);
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let input = {
            let mut m = M31Matrix::new(2, 4);
            for i in 0..8 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };
        let mut w1 = M31Matrix::new(4, 4);
        for i in 0..16 {
            w1.data[i] = M31::from((i % 3 + 1) as u32);
        }
        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..8 {
            w2.data[i] = M31::from((i % 5 + 1) as u32);
        }

        let hidden = matmul_m31(&input, &w1);
        let output = matmul_m31(&hidden, &w2);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, w1);
        weights.add_weight(1, w2);

        let execution = GraphExecution {
            intermediates: vec![(0, input.clone()), (1, hidden.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 2);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &output, &mut verifier_ch).unwrap();
    }

    #[test]
    fn test_matmul_layernorm_chain() {
        // MatMul(2x4 @ 4x4) → LayerNorm(dim=4)
        // Tests sumcheck → "LN" tag transition.
        // Row pattern [a-3d, a-d, a+d, a+3d] → mean=a (integer), variance=5d^2.
        // Use identity weights to preserve the pattern through matmul.
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(4);
        builder.layer_norm();
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        // Rows: [7,9,11,13] (mean=10,var=5) and [17,19,21,23] (mean=20,var=5)
        let input = {
            let mut m = M31Matrix::new(2, 4);
            m.set(0, 0, M31::from(7u32));
            m.set(0, 1, M31::from(9u32));
            m.set(0, 2, M31::from(11u32));
            m.set(0, 3, M31::from(13u32));
            m.set(1, 0, M31::from(17u32));
            m.set(1, 1, M31::from(19u32));
            m.set(1, 2, M31::from(21u32));
            m.set(1, 3, M31::from(23u32));
            m
        };
        // Identity weight: I_4
        let mut w1 = M31Matrix::new(4, 4);
        w1.set(0, 0, M31::from(1u32));
        w1.set(1, 1, M31::from(1u32));
        w1.set(2, 2, M31::from(1u32));
        w1.set(3, 3, M31::from(1u32));

        let hidden = matmul_m31(&input, &w1);
        let normed = ln_forward(&hidden, 4);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, w1);

        let execution = GraphExecution {
            intermediates: vec![(0, input.clone()), (1, hidden.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: normed.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 2);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &normed, &mut verifier_ch).unwrap();
    }

    #[test]
    fn test_activation_activation_chain() {
        // ReLU → GELU (back-to-back activations)
        // Tests independent gamma/beta draws for consecutive LogUp layers.
        let mut builder = GraphBuilder::new((2, 4));
        builder.activation(ActivationType::ReLU);
        builder.activation(ActivationType::GELU);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let input = {
            let mut m = M31Matrix::new(2, 4);
            for i in 0..8 {
                m.data[i] = M31::from((i + 2) as u32);
            }
            m
        };
        let after_relu = relu_m31(&input);
        let after_gelu = gelu_m31(&after_relu);

        let weights = GraphWeights::new();
        let execution = GraphExecution {
            intermediates: vec![(0, input.clone()), (1, after_relu.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: after_gelu.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 2);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &after_gelu, &mut verifier_ch).unwrap();
    }

    #[test]
    fn test_activation_matmul_chain() {
        // ReLU → MatMul(2x4 @ 4x2)
        // Tests LogUp → sumcheck transition.
        let mut builder = GraphBuilder::new((2, 4));
        builder.activation(ActivationType::ReLU);
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let input = {
            let mut m = M31Matrix::new(2, 4);
            for i in 0..8 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };
        let activated = relu_m31(&input);
        let mut w1 = M31Matrix::new(4, 2);
        for i in 0..8 {
            w1.data[i] = M31::from((i % 4 + 1) as u32);
        }
        let output = matmul_m31(&activated, &w1);

        let mut weights = GraphWeights::new();
        weights.add_weight(1, w1);

        let execution = GraphExecution {
            intermediates: vec![(0, input.clone()), (1, activated.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 2);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &output, &mut verifier_ch).unwrap();
    }

    #[test]
    fn test_matmul_matmul_matmul_chain() {
        // MatMul(2x4 @ 4x4) → MatMul(2x4 @ 4x4) → MatMul(2x4 @ 4x2)
        // Tests 3-deep consecutive sumchecks with dimension changes.
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(4);
        builder.linear(4);
        builder.linear(2);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        let input = {
            let mut m = M31Matrix::new(2, 4);
            for i in 0..8 {
                m.data[i] = M31::from((i + 1) as u32);
            }
            m
        };
        // Identity I4
        let mut w1 = M31Matrix::new(4, 4);
        w1.set(0, 0, M31::from(1u32));
        w1.set(1, 1, M31::from(1u32));
        w1.set(2, 2, M31::from(1u32));
        w1.set(3, 3, M31::from(1u32));
        // Small non-identity weight
        let mut w2 = M31Matrix::new(4, 4);
        for i in 0..16 {
            w2.data[i] = M31::from((i % 3 + 1) as u32);
        }
        let mut w3 = M31Matrix::new(4, 2);
        for i in 0..8 {
            w3.data[i] = M31::from((i % 4 + 1) as u32);
        }

        let h1 = matmul_m31(&input, &w1);
        let h2 = matmul_m31(&h1, &w2);
        let output = matmul_m31(&h2, &w3);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, w1);
        weights.add_weight(1, w2);
        weights.add_weight(2, w3);

        let execution = GraphExecution {
            intermediates: vec![(0, input.clone()), (1, h1.clone()), (2, h2.clone())],
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 3);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &output, &mut verifier_ch).unwrap();
    }

    #[test]
    fn test_deep_chain_4_layers() {
        // MatMul → ReLU → MatMul → LayerNorm (full pipeline)
        // This is the complete multi-layer chain test.
        // Row pattern [a-3d, a-d, a+d, a+3d] → mean=a (integer), variance=5d^2.
        // Identity weights + small positive values → ReLU is identity, LN gets clean values.
        let mut builder = GraphBuilder::new((2, 4));
        builder.linear(4);
        builder.activation(ActivationType::ReLU);
        builder.linear(4);
        builder.layer_norm();
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        // Rows: [7,9,11,13] and [17,19,21,23] — all < p/2, so ReLU passes them through
        let input = {
            let mut m = M31Matrix::new(2, 4);
            m.set(0, 0, M31::from(7u32));
            m.set(0, 1, M31::from(9u32));
            m.set(0, 2, M31::from(11u32));
            m.set(0, 3, M31::from(13u32));
            m.set(1, 0, M31::from(17u32));
            m.set(1, 1, M31::from(19u32));
            m.set(1, 2, M31::from(21u32));
            m.set(1, 3, M31::from(23u32));
            m
        };
        // Identity weights
        let mut w1 = M31Matrix::new(4, 4);
        w1.set(0, 0, M31::from(1u32));
        w1.set(1, 1, M31::from(1u32));
        w1.set(2, 2, M31::from(1u32));
        w1.set(3, 3, M31::from(1u32));

        let mut w2 = M31Matrix::new(4, 4);
        w2.set(0, 0, M31::from(1u32));
        w2.set(1, 1, M31::from(1u32));
        w2.set(2, 2, M31::from(1u32));
        w2.set(3, 3, M31::from(1u32));

        let hidden1 = matmul_m31(&input, &w1);
        let activated = relu_m31(&hidden1);
        let hidden2 = matmul_m31(&activated, &w2);
        let output = ln_forward(&hidden2, 4);

        let mut weights = GraphWeights::new();
        weights.add_weight(0, w1);
        weights.add_weight(2, w2);

        let execution = GraphExecution {
            intermediates: vec![
                (0, input.clone()),
                (1, hidden1.clone()),
                (2, activated.clone()),
                (3, hidden2.clone()),
            ],
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 4);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &output, &mut verifier_ch).unwrap();
    }

    #[test]
    fn test_layernorm_logup_none_rejected() {
        // Forge a LayerNorm proof with logup_proof: None and simd_combined: false.
        // The verifier should reject this as non-SIMD LayerNorm requires LogUp.
        let input = {
            let mut m = M31Matrix::new(2, 4);
            m.set(0, 0, M31::from(10u32));
            m.set(0, 1, M31::from(20u32));
            m.set(0, 2, M31::from(30u32));
            m.set(0, 3, M31::from(40u32));
            m.set(1, 0, M31::from(5u32));
            m.set(1, 1, M31::from(15u32));
            m.set(1, 2, M31::from(25u32));
            m.set(1, 3, M31::from(35u32));
            m
        };
        let dim = 4;

        let output = layernorm_forward(&input, dim);
        let output_padded = pad_matrix_pow2(&output);
        let n = output_padded.rows * output_padded.cols;
        let output_mle: Vec<SecureField> = output_padded
            .data
            .iter()
            .take(n)
            .map(|&v| SecureField::from(v))
            .collect();
        let num_vars = output_mle.len().ilog2() as usize;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0x4C4E0E);
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        // Produce a valid LayerNorm proof (has logup_proof: Some(...))
        let (mut proof, _) =
            reduce_layernorm_layer_for_test(&output_claim, &input, dim, &mut prover_channel)
                .unwrap();

        // Tamper: strip LogUp proof AND set simd_combined=false
        if let LayerProof::LayerNorm {
            ref mut logup_proof,
            ref mut simd_combined,
            ..
        } = &mut proof
        {
            *logup_proof = None;
            *simd_combined = false;
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x4C4E0E);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::LayerNorm {
                logup_proof,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *mean,
                    *rsqrt_var,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    &mut verifier_channel,
                );
                assert!(
                    result.is_err(),
                    "non-SIMD LayerNorm without LogUp should be rejected"
                );
                let err_msg = format!("{:?}", result.err().unwrap());
                assert!(
                    err_msg.contains("non-SIMD LayerNorm missing required LogUp proof"),
                    "unexpected error: {err_msg}",
                );
            }
            _ => panic!("expected LayerNorm proof"),
        }
    }

    /// Test the dual SIMD matmul verification on a manually constructed proof.
    ///
    /// Constructs extended MLEs (ext_w, ext_a, ext_b) for 2 blocks with k=4,
    /// runs the 3-factor eq-sumcheck on CPU, and verifies the resulting proof.
    #[test]
    fn test_dual_simd_matmul_verify_roundtrip() {
        use super::super::types::RoundPolyDeg3;

        let n_blocks = 2usize;
        let k = 4usize;
        let pk = k.next_power_of_two(); // 4
        let n_block_vars = n_blocks.ilog2() as usize; // 1
        let log_k = pk.ilog2() as usize; // 2
        let total_vars = n_block_vars + log_k; // 3
        let ext_len = n_blocks * pk; // 8

        // Build fake block weights (Lagrange basis for r_simd)
        let r_simd = vec![SecureField::from(M31::from(3u32))]; // 1 SIMD variable
        let w_0 = SecureField::one() - r_simd[0]; // 1 - 3 = -2
        let w_1 = r_simd[0]; // 3

        // Build fake restricted A and B vectors (per block, length pk)
        let f_a_0: Vec<SecureField> = (1..=4).map(|x| SecureField::from(M31::from(x))).collect();
        let f_a_1: Vec<SecureField> = (5..=8).map(|x| SecureField::from(M31::from(x))).collect();
        let f_b_0: Vec<SecureField> = (10..=13).map(|x| SecureField::from(M31::from(x))).collect();
        let f_b_1: Vec<SecureField> = (14..=17).map(|x| SecureField::from(M31::from(x))).collect();

        // Build extended MLEs
        let mut ext_w = vec![SecureField::zero(); ext_len];
        let mut ext_a = vec![SecureField::zero(); ext_len];
        let mut ext_b = vec![SecureField::zero(); ext_len];
        for ki in 0..pk {
            ext_w[0 * pk + ki] = w_0;
            ext_w[1 * pk + ki] = w_1;
            ext_a[0 * pk + ki] = f_a_0[ki];
            ext_a[1 * pk + ki] = f_a_1[ki];
            ext_b[0 * pk + ki] = f_b_0[ki];
            ext_b[1 * pk + ki] = f_b_1[ki];
        }

        // Compute claim: Σ ext_w[i] * ext_a[i] * ext_b[i]
        let claimed_sum: SecureField = (0..ext_len)
            .map(|i| ext_w[i] * ext_a[i] * ext_b[i])
            .fold(SecureField::zero(), |acc, x| acc + x);

        // Build output claim with a fake point
        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(999);
        let m = 2usize;
        let n = 2usize;
        let log_m = m.next_power_of_two().ilog2() as usize;
        let log_n = n.next_power_of_two().ilog2() as usize;
        let r_out = prover_channel.draw_qm31s(log_m + log_n);
        let output_claim = GKRClaim {
            point: r_out,
            value: claimed_sum,
        };

        // Run 3-factor eq-sumcheck on CPU (same logic as prover)
        prover_channel.mix_u64(m as u64);
        prover_channel.mix_u64(k as u64);
        prover_channel.mix_u64(n as u64);
        prover_channel.mix_u64(n_blocks as u64);
        mix_secure_field(&mut prover_channel, claimed_sum);

        let mut round_polys = Vec::with_capacity(total_vars);
        let mut cur_w = ext_w;
        let mut cur_a = ext_a;
        let mut cur_b = ext_b;
        let mut cur_n = ext_len;

        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let inv2 = {
            use stwo::core::fields::FieldExpOps;
            two.inverse()
        };
        let inv6 = {
            use stwo::core::fields::FieldExpOps;
            (SecureField::from(M31::from(6u32))).inverse()
        };

        fn sum_at_t(
            w: &[SecureField],
            a: &[SecureField],
            b: &[SecureField],
            mid: usize,
            t: SecureField,
        ) -> SecureField {
            let one_minus_t = SecureField::one() - t;
            let mut sum = SecureField::zero();
            for i in 0..mid {
                let w_t = one_minus_t * w[i] + t * w[mid + i];
                let a_t = one_minus_t * a[i] + t * a[mid + i];
                let b_t = one_minus_t * b[i] + t * b[mid + i];
                sum = sum + w_t * a_t * b_t;
            }
            sum
        }

        fn fold(vals: &[SecureField], r: SecureField, mid: usize) -> Vec<SecureField> {
            let one_minus_r = SecureField::one() - r;
            (0..mid)
                .map(|i| one_minus_r * vals[i] + r * vals[mid + i])
                .collect()
        }

        for _ in 0..total_vars {
            let mid = cur_n / 2;
            let s0 = sum_at_t(&cur_w, &cur_a, &cur_b, mid, SecureField::zero());
            let s1 = sum_at_t(&cur_w, &cur_a, &cur_b, mid, SecureField::one());
            let s2 = sum_at_t(&cur_w, &cur_a, &cur_b, mid, two);
            let s3 = sum_at_t(&cur_w, &cur_a, &cur_b, mid, three);

            let d1 = s1 - s0;
            let d2 = (s2 - s1 - s1 + s0) * inv2;
            let d3 = (s3 - s0 - three * (s2 - s1)) * inv6;

            let c0 = s0;
            let c1 = d1 - d2 + two * d3;
            let c2 = d2 - three * d3;
            let c3 = d3;

            let rp = RoundPolyDeg3 { c0, c1, c2, c3 };
            round_polys.push(rp);

            prover_channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
            let challenge = prover_channel.draw_qm31();

            cur_w = fold(&cur_w, challenge, mid);
            cur_a = fold(&cur_a, challenge, mid);
            cur_b = fold(&cur_b, challenge, mid);
            cur_n = mid;
        }

        let final_a_eval = cur_a[0];
        let final_b_eval = cur_b[0];
        mix_secure_field(&mut prover_channel, final_a_eval);
        mix_secure_field(&mut prover_channel, final_b_eval);

        // Now verify
        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(999);
        let _r_out_v = verifier_channel.draw_qm31s(log_m + log_n);

        let result = verify_matmul_dual_simd_reduction(
            &output_claim,
            &round_polys,
            final_a_eval,
            final_b_eval,
            n_block_vars,
            &r_simd,
            m,
            k,
            n,
            n_blocks,
            0,
            &mut verifier_channel,
        );

        assert!(
            result.is_ok(),
            "dual SIMD matmul verification failed: {:?}",
            result.err()
        );
    }

    /// Test that tampering with final_a_eval in a MatMulDualSimd proof is rejected.
    #[test]
    fn test_dual_simd_matmul_tampered_rejected() {
        use super::super::types::RoundPolyDeg3;

        let n_blocks = 2usize;
        let k = 4usize;
        let pk = k;
        let n_block_vars = 1;
        let total_vars = 3;
        let ext_len = 8;

        let r_simd = vec![SecureField::from(M31::from(3u32))];
        let w_0 = SecureField::one() - r_simd[0];
        let w_1 = r_simd[0];

        let f_a_0: Vec<SecureField> = (1..=4).map(|x| SecureField::from(M31::from(x))).collect();
        let f_a_1: Vec<SecureField> = (5..=8).map(|x| SecureField::from(M31::from(x))).collect();
        let f_b_0: Vec<SecureField> = (10..=13).map(|x| SecureField::from(M31::from(x))).collect();
        let f_b_1: Vec<SecureField> = (14..=17).map(|x| SecureField::from(M31::from(x))).collect();

        let mut ext_w = vec![SecureField::zero(); ext_len];
        let mut ext_a = vec![SecureField::zero(); ext_len];
        let mut ext_b = vec![SecureField::zero(); ext_len];
        for ki in 0..pk {
            ext_w[ki] = w_0;
            ext_w[pk + ki] = w_1;
            ext_a[ki] = f_a_0[ki];
            ext_a[pk + ki] = f_a_1[ki];
            ext_b[ki] = f_b_0[ki];
            ext_b[pk + ki] = f_b_1[ki];
        }

        let claimed_sum: SecureField = (0..ext_len)
            .map(|i| ext_w[i] * ext_a[i] * ext_b[i])
            .fold(SecureField::zero(), |acc, x| acc + x);

        let m = 2;
        let n = 2;
        let log_m = 1;
        let log_n = 1;

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(888);
        let r_out = prover_channel.draw_qm31s(log_m + log_n);
        let output_claim = GKRClaim {
            point: r_out,
            value: claimed_sum,
        };

        prover_channel.mix_u64(m as u64);
        prover_channel.mix_u64(k as u64);
        prover_channel.mix_u64(n as u64);
        prover_channel.mix_u64(n_blocks as u64);
        mix_secure_field(&mut prover_channel, claimed_sum);

        let two = SecureField::from(M31::from(2u32));
        let three = SecureField::from(M31::from(3u32));
        let inv2 = {
            use stwo::core::fields::FieldExpOps;
            two.inverse()
        };
        let inv6 = {
            use stwo::core::fields::FieldExpOps;
            (SecureField::from(M31::from(6u32))).inverse()
        };

        fn sum_at_t(
            w: &[SecureField],
            a: &[SecureField],
            b: &[SecureField],
            mid: usize,
            t: SecureField,
        ) -> SecureField {
            let one_minus_t = SecureField::one() - t;
            (0..mid)
                .map(|i| {
                    let wt = one_minus_t * w[i] + t * w[mid + i];
                    let at = one_minus_t * a[i] + t * a[mid + i];
                    let bt = one_minus_t * b[i] + t * b[mid + i];
                    wt * at * bt
                })
                .fold(SecureField::zero(), |a, b| a + b)
        }

        fn fold(v: &[SecureField], r: SecureField, mid: usize) -> Vec<SecureField> {
            (0..mid)
                .map(|i| (SecureField::one() - r) * v[i] + r * v[mid + i])
                .collect()
        }

        let mut round_polys = Vec::new();
        let mut cw = ext_w;
        let mut ca = ext_a;
        let mut cb = ext_b;
        let mut cn = ext_len;

        for _ in 0..total_vars {
            let mid = cn / 2;
            let s0 = sum_at_t(&cw, &ca, &cb, mid, SecureField::zero());
            let s1 = sum_at_t(&cw, &ca, &cb, mid, SecureField::one());
            let s2 = sum_at_t(&cw, &ca, &cb, mid, two);
            let s3 = sum_at_t(&cw, &ca, &cb, mid, three);

            let d1 = s1 - s0;
            let d2 = (s2 - s1 - s1 + s0) * inv2;
            let d3 = (s3 - s0 - three * (s2 - s1)) * inv6;
            let c0 = s0;
            let c1 = d1 - d2 + two * d3;
            let c2 = d2 - three * d3;
            let c3 = d3;

            round_polys.push(RoundPolyDeg3 { c0, c1, c2, c3 });
            prover_channel.mix_poly_coeffs_deg3(c0, c1, c2, c3);
            let ch = prover_channel.draw_qm31();
            cw = fold(&cw, ch, mid);
            ca = fold(&ca, ch, mid);
            cb = fold(&cb, ch, mid);
            cn = mid;
        }

        let final_a_eval = ca[0];
        let final_b_eval = cb[0];

        // Tamper: modify final_a_eval
        let tampered_a = final_a_eval + SecureField::one();

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(888);
        let _r = verifier_channel.draw_qm31s(log_m + log_n);

        let result = verify_matmul_dual_simd_reduction(
            &output_claim,
            &round_polys,
            tampered_a,
            final_b_eval,
            n_block_vars,
            &r_simd,
            m,
            k,
            n,
            n_blocks,
            0,
            &mut verifier_channel,
        );

        assert!(
            result.is_err(),
            "tampered dual SIMD matmul should fail verification"
        );
        let err_msg = format!("{:?}", result.err().unwrap());
        assert!(
            err_msg.contains("final check failed"),
            "expected 'final check failed', got: {err_msg}"
        );
    }
}
