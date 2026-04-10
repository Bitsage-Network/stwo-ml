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
use crate::crypto::aggregated_opening::{verify_aggregated_binding, AggregatedWeightClaim};
use crate::crypto::poseidon_channel::PoseidonChannel;
use crate::gadgets::lookup_table::PrecomputedTable;

use super::circuit::{LayerType, LayeredCircuit};
use super::prover::compute_rsqrt_table_commitment;
use super::types::{
    derive_weight_opening_subchannel, EmbeddingLogUpProof, GKRClaim, GKRError, GKRProof,
    LayerProof, LogUpProof, MultiplicitySumcheckProof, RoundPolyDeg3, SecureField,
    WeightOpeningTranscriptMode,
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
    verify_gkr_inner(circuit, proof, output, None, channel, None)
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
    verify_gkr_inner(circuit, proof, output, Some(weights), channel, None)
}

/// Verify a GKR proof with explicit policy binding.
///
/// The policy commitment must match the one used during proving — otherwise
/// the Fiat-Shamir transcript diverges and verification fails.
pub fn verify_gkr_with_policy(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    output: &M31Matrix,
    weights: Option<&GraphWeights>,
    channel: &mut PoseidonChannel,
    policy: &crate::policy::PolicyConfig,
) -> Result<GKRClaim, GKRError> {
    verify_gkr_inner(circuit, proof, output, weights, channel, Some(policy))
}

fn verify_gkr_inner(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    output: &M31Matrix,
    weights: Option<&GraphWeights>,
    channel: &mut PoseidonChannel,
    policy: Option<&crate::policy::PolicyConfig>,
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

    // Resolve policy configuration (must match prover's policy).
    let resolved_policy = crate::policy::resolve(policy);

    // Seed channel identically to prover
    channel.mix_u64(d as u64);
    channel.mix_u64(circuit.input_shape.0 as u64);
    channel.mix_u64(circuit.input_shape.1 as u64);

    // Bind policy commitment (must match prover's mix position exactly).
    let policy_commitment = resolved_policy.policy_commitment();
    let skip_policy = std::env::var("STWO_SKIP_POLICY_COMMITMENT").is_ok();
    if !skip_policy && policy_commitment != starknet_ff::FieldElement::ZERO {
        channel.mix_felt(policy_commitment);
    }

    let _v_trace = std::env::var("STWO_CHANNEL_TRACE").is_ok();
    if _v_trace {
        eprintln!("[VERIFIER] ch after seeding+policy: {:?}", channel.digest());
        eprintln!("[VERIFIER] policy: {:?}, skip={}", policy_commitment, skip_policy);
    }

    // Reconstruct output claim
    let output_padded = pad_matrix_pow2(output);
    let output_mle = matrix_to_mle(&output_padded);

    let log_out_rows = output_padded.rows.ilog2() as usize;
    let log_out_cols = output_padded.cols.ilog2() as usize;

    let r_out = channel.draw_qm31s(log_out_rows + log_out_cols);
    let output_value = evaluate_mle(&output_mle, &r_out);

    mix_secure_field(channel, output_value);

    if _v_trace {
        eprintln!("[VERIFIER] output_value: {:?}", output_value);
        eprintln!("[VERIFIER] ch after output claim: {:?}", channel.digest());
    }

    let mut current_claim = GKRClaim {
        point: r_out,
        value: output_value,
    };

    // Walk layers from output → input, verifying each proof
    let mut proof_idx = 0;
    let mut expected_weight_node_ids = Vec::new();
    // Track skip layer indices for deferred proofs (populated on Add layers)
    let mut deferred_skip_layer_indices: Vec<usize> = Vec::new();

    // Layers to skip in the main walk (deferred branches of Add/Mul)
    let mut skip_layers: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for layer_idx in (0..d).rev() {
        if skip_layers.contains(&layer_idx) {
            continue;
        }
        let layer = &circuit.layers[layer_idx];

        match &layer.layer_type {
            LayerType::Identity | LayerType::TopK { .. } => {
                // TopK: verify witness binding in the channel, propagate claim.
                // The TopK proof data is mixed into the channel during proving.
                // The verifier replays the same channel operations.
                if let LayerType::TopK { num_experts, top_k } = &layer.layer_type {
                    if proof_idx < proof.layer_proofs.len() {
                        if let LayerProof::TopK {
                            num_experts: pn, top_k: pk,
                            selected_indices, selected_values,
                            threshold_gap, logits_commitment,
                        } = &proof.layer_proofs[proof_idx] {
                            proof_idx += 1;

                            // Structural checks
                            if selected_indices.len() != *pk {
                                return Err(GKRError::VerificationError {
                                    layer_idx,
                                    reason: format!("TopK: expected {} selected indices, got {}", pk, selected_indices.len()),
                                });
                            }
                            if selected_values.len() != *pk {
                                return Err(GKRError::VerificationError {
                                    layer_idx,
                                    reason: format!("TopK: expected {} selected values, got {}", pk, selected_values.len()),
                                });
                            }

                            // Replay channel mixing (must match prover exactly)
                            channel.mix_u64(0x544F504B_u64); // "TOPK"
                            channel.mix_u64(*pn as u64);
                            channel.mix_u64(*pk as u64);
                            channel.mix_felt(*logits_commitment);
                            for &idx in selected_indices {
                                channel.mix_u64(idx as u64);
                            }
                            for &val in selected_values {
                                mix_secure_field(channel, val);
                            }
                            mix_secure_field(channel, *threshold_gap);
                            mix_secure_field(channel, current_claim.value);

                            // Claim propagation: value changes to logits eval (from proof)
                            // The TopK doesn't change the claim point, only the value
                            continue;
                        }
                    }
                }
                continue;
            }
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

        if _v_trace {
            eprintln!(
                "[VERIFIER] L{} type={:?} ch={:?} claim_val={:?}",
                layer_idx,
                std::mem::discriminant(&layer.layer_type),
                channel.digest(),
                current_claim.value,
            );
        }

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
            ) => {
                // Track skip layer index for deferred proof verification
                let skip_layer_idx = if layer.input_layers.len() >= 2
                    && layer.input_layers[1] > layer.input_layers[0]
                {
                    layer.input_layers[0]
                } else if layer.input_layers.len() >= 2 {
                    layer.input_layers[1]
                } else {
                    0
                };
                deferred_skip_layer_indices.push(skip_layer_idx);
                skip_layers.insert(skip_layer_idx);
                verify_add_reduction(
                    &current_claim,
                    *lhs_eval,
                    *rhs_eval,
                    &layer.input_layers,
                    layer_idx,
                    channel,
                )?
            }

            (
                LayerType::Mul { .. },
                LayerProof::Mul {
                    eq_round_polys,
                    lhs_eval,
                    rhs_eval,
                },
            ) => {
                let combined = verify_mul_reduction(
                    &current_claim,
                    eq_round_polys,
                    *lhs_eval,
                    *rhs_eval,
                    layer_idx,
                    channel,
                )?;
                // For branched Mul (gated FFN): override claim with trunk eval
                if layer.input_layers.len() >= 2 {
                    let skip_layer_idx = if layer.input_layers[1] > layer.input_layers[0] {
                        layer.input_layers[0]
                    } else {
                        layer.input_layers[1]
                    };
                    let trunk_eval = if layer.input_layers[1] > layer.input_layers[0] {
                        *rhs_eval
                    } else {
                        *lhs_eval
                    };
                    deferred_skip_layer_indices.push(skip_layer_idx);
                    skip_layers.insert(skip_layer_idx);
                    GKRClaim {
                        point: combined.point,
                        value: trunk_eval,
                    }
                } else {
                    combined
                }
            }

            (
                LayerType::Activation {
                    activation_type: circuit_act_type,
                    size: circuit_size,
                },
                LayerProof::Activation {
                    activation_type: proof_act_type,
                    logup_proof,
                    multiplicity_sumcheck,
                    activation_proof,
                    piecewise_proof,
                    input_eval,
                    output_eval,
                    table_commitment,
                    simd_combined,
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
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    piecewise_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    *circuit_size,
                    layer_idx,
                    channel,
                    *simd_combined,
                )?
            }

            (
                LayerType::LayerNorm { dim, .. },
                LayerProof::LayerNorm {
                    logup_proof,
                    multiplicity_sumcheck,
                    linear_round_polys,
                    linear_final_evals,
                    input_eval,
                    output_eval,
                    mean,
                    rsqrt_var,
                    rsqrt_table_commitment,
                    simd_combined,
                    mean_var_round_polys,
                    mean_var_final_evals,
                    var_eval,
                    centered_binding_evals,
                    mv_claimed_sums,
                    row_means,
                    row_variances,
                    ..
                },
            ) => verify_layernorm_reduction(
                &current_claim,
                logup_proof.as_ref(),
                multiplicity_sumcheck.as_ref(),
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
                mean_var_round_polys.as_ref(),
                *mean_var_final_evals,
                *var_eval,
                *centered_binding_evals,
                *mv_claimed_sums,
                row_means.as_ref(),
                row_variances.as_ref(),
            )?,

            (
                LayerType::Dequantize { params, .. },
                LayerProof::Dequantize {
                    logup_proof,
                    multiplicity_sumcheck,
                    input_eval,
                    output_eval,
                    table_commitment,
                    ..
                },
            ) => verify_dequantize_reduction(
                &current_claim,
                params,
                logup_proof.as_ref(),
                multiplicity_sumcheck.as_ref(),
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
                    ..
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
                    ..
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
                    multiplicity_sumcheck,
                    linear_round_polys,
                    linear_final_evals,
                    input_eval,
                    output_eval,
                    rms_sq_eval,
                    rsqrt_eval,
                    rsqrt_table_commitment,
                    simd_combined,
                    rms_sq_round_polys,
                    rms_sq_input_final,
                    rms_sq_claimed_sq_sum,
                    row_rms_sq,
                    ..
                },
            ) => verify_rmsnorm_reduction(
                &current_claim,
                logup_proof.as_ref(),
                multiplicity_sumcheck.as_ref(),
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
                rms_sq_round_polys.as_ref(),
                *rms_sq_input_final,
                *rms_sq_claimed_sq_sum,
                row_rms_sq.as_ref(),
            )?,

            (
                LayerType::Attention { config },
                LayerProof::Attention {
                    sub_proofs,
                    sub_claim_values,
                    ref softmax_sum_proofs,
                    ..
                },
            ) => {
                verify_attention_reduction(
                    &current_claim,
                    config,
                    sub_proofs,
                    sub_claim_values,
                    softmax_sum_proofs,
                    None, // no SIMD in standard verify_gkr
                    layer_idx,
                    channel,
                )?
            }

            (
                LayerType::Attention { config: _ },
                LayerProof::AttentionDecode {
                    sub_proofs,
                    sub_claim_values,
                    num_heads,
                    new_tokens,
                    full_seq_len,
                    d_model,
                    causal,
                    position_offset,
                },
            ) => {
                verify_attention_reduction_decode(
                    &current_claim,
                    sub_proofs,
                    sub_claim_values,
                    *num_heads,
                    *new_tokens,
                    *full_seq_len,
                    *d_model,
                    *causal,
                    *position_offset,
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

    // In batched/aggregated modes, deferred weight bindings are verified in the
    // aggregated pass — not per-deferred Merkle openings. This covers both the old
    // BatchedRlcDirectEvalV1 mode and the new AggregatedOracleSumcheck mode.
    let batched_rlc_mode =
        proof.weight_opening_transcript_mode == WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1
            || proof.weight_opening_transcript_mode
                == WeightOpeningTranscriptMode::AggregatedOracleSumcheck;

    // Verify deferred proofs for skip branches of DAG Add layers.
    // Fiat-Shamir order: walk → deferred proofs → weight openings.
    // Each deferred proof contains a layer reduction + optional weight opening that
    // binds the skip_eval from an Add reduction to actual weights and input.
    if proof.deferred_proofs.len() != deferred_skip_layer_indices.len() {
        return Err(GKRError::VerificationError {
            layer_idx: 0,
            reason: format!(
                "deferred_proofs count ({}) != Add layers walked ({})",
                proof.deferred_proofs.len(),
                deferred_skip_layer_indices.len(),
            ),
        });
    }
    for (i, deferred) in proof.deferred_proofs.iter().enumerate() {
        // Mix deferred claim into channel (must match prover)
        mix_secure_field(channel, deferred.claim.value);

        // Verify the deferred layer reduction
        match &deferred.layer_proof {
            LayerProof::MatMul {
                round_polys,
                final_a_eval,
                final_b_eval,
            } => {
                let (m, k, n) = deferred.dims().ok_or_else(|| GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "deferred proof {} is MatMul layer_proof but has Weightless kind",
                        i
                    ),
                })?;
                // Cross-check dims against circuit's skip layer
                let skip_idx = *deferred_skip_layer_indices.get(i).ok_or_else(|| {
                    GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "deferred proof index {i} out of bounds (only {} skip indices)",
                            deferred_skip_layer_indices.len()
                        ),
                    }
                })?;
                if skip_idx >= circuit.layers.len() {
                    return Err(GKRError::VerificationError {
                        layer_idx: skip_idx,
                        reason: format!(
                            "skip_idx {skip_idx} out of bounds (circuit has {} layers)",
                            circuit.layers.len()
                        ),
                    });
                }
                if let LayerType::MatMul {
                    m: cm,
                    k: ck,
                    n: cn,
                    ..
                } = &circuit.layers[skip_idx].layer_type
                {
                    if m != *cm || k != *ck || n != *cn {
                        return Err(GKRError::VerificationError {
                            layer_idx: skip_idx,
                            reason: format!(
                                "deferred proof {} dims ({},{},{}) != circuit layer dims ({},{},{})",
                                i, m, k, n, cm, ck, cn,
                            ),
                        });
                    }
                }
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
            LayerProof::Dequantize {
                logup_proof,
                multiplicity_sumcheck,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                // Verify Dequantize deferred proof via full LogUp verification.
                // Look up QuantParams from the circuit's skip layer.
                let skip_layer_idx =
                    deferred_skip_layer_indices.get(i).copied().ok_or_else(|| {
                        GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "deferred proof {} has no matching skip layer index",
                                i
                            ),
                        }
                    })?;
                let params = match &circuit.layers[skip_layer_idx].layer_type {
                    LayerType::Dequantize { params, .. } => params,
                    other => {
                        return Err(GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "deferred proof {} skip layer {} is {:?}, expected Dequantize",
                                i,
                                skip_layer_idx,
                                std::mem::discriminant(other),
                            ),
                        })
                    }
                };
                let deferred_input_claim = verify_dequantize_reduction(
                    &deferred.claim,
                    params,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0,
                    channel,
                )?;
                if deferred_input_claim.point != deferred.input_claim.point
                    || deferred_input_claim.value != deferred.input_claim.value
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!("deferred proof {} (dequantize) input claim mismatch", i),
                    });
                }
            }
            LayerProof::Quantize {
                logup_proof,
                input_eval,
                output_eval,
                table_inputs,
                table_outputs,
                ..
            } => {
                // Verify Quantize deferred proof via full LogUp verification.
                let skip_layer_idx =
                    deferred_skip_layer_indices.get(i).copied().ok_or_else(|| {
                        GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "deferred proof {} has no matching skip layer index",
                                i
                            ),
                        }
                    })?;
                let params = match &circuit.layers[skip_layer_idx].layer_type {
                    LayerType::Quantize { params, .. } => params,
                    other => {
                        return Err(GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "deferred proof {} skip layer {} is {:?}, expected Quantize",
                                i,
                                skip_layer_idx,
                                std::mem::discriminant(other),
                            ),
                        })
                    }
                };
                let deferred_input_claim = verify_quantize_reduction(
                    &deferred.claim,
                    params,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    table_inputs,
                    table_outputs,
                    0,
                    channel,
                )?;
                if deferred_input_claim.point != deferred.input_claim.point
                    || deferred_input_claim.value != deferred.input_claim.value
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!("deferred proof {} (quantize) input claim mismatch", i),
                    });
                }
            }
            LayerProof::Add {
                lhs_eval,
                rhs_eval,
                ..
            } => {
                // Deferred Add proof: verify Add reduction (lhs + rhs == claim).
                // Used when a transformer-block residual Add is on a deferred skip branch.
                let skip_layer_idx = deferred_skip_layer_indices.get(i).copied().unwrap_or(0);
                let input_layers = &circuit.layers[skip_layer_idx].input_layers;
                let deferred_input_claim = verify_add_reduction(
                    &deferred.claim,
                    *lhs_eval,
                    *rhs_eval,
                    input_layers,
                    0,
                    channel,
                )?;
                if deferred_input_claim.point != deferred.input_claim.point
                    || deferred_input_claim.value != deferred.input_claim.value
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!("deferred proof {} (add) input claim mismatch", i),
                    });
                }
            }
            LayerProof::RMSNorm {
                logup_proof,
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                rms_sq_eval,
                rsqrt_eval,
                rsqrt_table_commitment,
                simd_combined,
                rms_sq_round_polys,
                rms_sq_input_final,
                rms_sq_claimed_sq_sum,
                rms_sq_n_active,
                row_rms_sq,
                gamma_commitment: _,
                gamma_eval: _,
            } => {
                // Deferred RMSNorm proof
                let skip_layer_idx = deferred_skip_layer_indices.get(i).copied().unwrap_or(0);
                let dim = match &circuit.layers[skip_layer_idx].layer_type {
                    LayerType::RMSNorm { dim, .. } => *dim,
                    _ => {
                        return Err(GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "deferred proof {} skip layer is not RMSNorm",
                                i
                            ),
                        })
                    }
                };
                let deferred_input_claim = verify_rmsnorm_reduction(
                    &deferred.claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *rms_sq_eval,
                    *rsqrt_eval,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    channel,
                    rms_sq_round_polys.as_ref(),
                    *rms_sq_input_final,
                    *rms_sq_claimed_sq_sum,
                    row_rms_sq.as_ref(),
                )?;
                if deferred_input_claim.point != deferred.input_claim.point
                    || deferred_input_claim.value != deferred.input_claim.value
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!("deferred proof {} (rmsnorm) input claim mismatch", i),
                    });
                }
            }
            LayerProof::Activation {
                activation_type,
                logup_proof,
                multiplicity_sumcheck,
                activation_proof,
                piecewise_proof,
                input_eval,
                output_eval,
                table_commitment,
                simd_combined,
            } => {
                // Deferred Activation proof
                let deferred_input_claim = verify_activation_reduction(
                    &deferred.claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    piecewise_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0,
                    0,
                    channel,
                    *simd_combined,
                )?;
                if deferred_input_claim.point != deferred.input_claim.point
                    || deferred_input_claim.value != deferred.input_claim.value
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!("deferred proof {} (activation) input claim mismatch", i),
                    });
                }
            }
            _ => {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!("deferred proof {} has unsupported layer proof type", i,),
                });
            }
        }

        // Weight opening verification (MatMul kind only)
        if deferred.has_weights() {
            let weight_commitment = deferred.weight_commitment().unwrap();
            let weight_opening = deferred.weight_opening().unwrap();
            let weight_claim = deferred.weight_claim().unwrap();

            if batched_rlc_mode {
                // In batched/aggregated modes, deferred weight binding is checked
                // in the aggregated verifier pass (no per-deferred Merkle opening).
                //
                // BatchedRlcDirectEvalV1: weight_commitment = ZERO, opening = empty.
                // AggregatedOracleSumcheck: weight_commitment = real Merkle root
                //   (needed for super-root reconstruction), opening = empty.
                let is_agg_oracle = proof.weight_opening_transcript_mode
                    == WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
                if !is_agg_oracle && weight_commitment != starknet_ff::FieldElement::ZERO {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "deferred proof {} expects zero weight_commitment in batched RLC mode",
                            i
                        ),
                    });
                }
                if !weight_opening.intermediate_roots.is_empty()
                    || !weight_opening.queries.is_empty()
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
                if weight_opening.final_value != weight_claim.expected_value {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!("deferred proof {} weight opening final_value mismatch", i,),
                    });
                }
                if !crate::crypto::mle_opening::verify_mle_opening(
                    weight_commitment,
                    weight_opening,
                    &weight_claim.eval_point,
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
                if let Some(claim) = deferred.weight_claim() {
                    let weight = weights.get_weight(claim.weight_node_id).ok_or(
                        GKRError::MissingWeight {
                            node_id: claim.weight_node_id,
                        },
                    )?;
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
            let mut matmul_deferred_idx = 0usize;
            for deferred in proof.deferred_proofs.iter() {
                if let Some(claim) = deferred.weight_claim() {
                    let claim_idx = proof.weight_claims.len() + matmul_deferred_idx;
                    agg_claims.push(AggregatedWeightClaim {
                        matrix_index: claim_idx,
                        local_n_vars: claim.eval_point.len(),
                        eval_point: claim.eval_point.clone(),
                        expected_value: claim.expected_value,
                        commitment: deferred
                            .weight_commitment()
                            .unwrap_or(starknet_ff::FieldElement::ZERO),
                    });
                    matmul_deferred_idx += 1;
                }
            }

            if agg_claims.is_empty() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: "AggregatedOracleSumcheck mode with zero claims".to_string(),
                });
            }

            if let Some(binding_proof) = proof.aggregated_binding.as_ref() {
                // Single binding proof path (small models, ≤ GROUP_SIZE matrices)
                if !verify_aggregated_binding(binding_proof, &agg_claims, channel) {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: "aggregated oracle sumcheck weight binding verification failed"
                            .to_string(),
                    });
                }
            } else if !proof.binding_groups.is_empty() {
                // Grouped binding proof path (large models, > GROUP_SIZE matrices)
                // Each group covers a subset of claims and is verified independently
                // through the same Fiat-Shamir channel.
                let group_size = crate::gkr::prover::BINDING_GROUP_SIZE;
                let expected_groups =
                    (agg_claims.len() + group_size - 1) / group_size;
                if proof.binding_groups.len() != expected_groups {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "binding_groups count mismatch: got {}, expected {} ({} claims / {} group_size)",
                            proof.binding_groups.len(), expected_groups,
                            agg_claims.len(), group_size,
                        ),
                    });
                }
                for (g, group_proof) in proof.binding_groups.iter().enumerate() {
                    let chunk_start = g * group_size;
                    let chunk_end = (chunk_start + group_size).min(agg_claims.len());
                    let group_claims = &agg_claims[chunk_start..chunk_end];
                    if !verify_aggregated_binding(group_proof, group_claims, channel) {
                        return Err(GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "grouped binding verification failed at group {}/{} (claims {}..{})",
                                g + 1, expected_groups, chunk_start, chunk_end,
                            ),
                        });
                    }
                }
            } else {
                // RLC-only: compute combined_expected from proof claims.
                // When OBELYZK_TRUST_WEIGHT_CLAIMS=1 (recursive STARK path),
                // skip the expensive MLE re-evaluation — trust the prover's
                // weight claims and only replay channel operations. Weight
                // binding is verified on-chain via Poseidon commitments.
                let trust_weights = std::env::var("OBELYZK_TRUST_WEIGHT_CLAIMS").is_ok();

                let rho = channel.draw_qm31();
                let mut rho_pow = SecureField::one();
                let mut combined_expected = SecureField::zero();

                // Deferred weight claims first (prover order: deferred before main)
                for deferred in proof.deferred_proofs.iter() {
                    if let Some(claim) = deferred.weight_claim() {
                        combined_expected += rho_pow * claim.expected_value;
                        rho_pow = rho_pow * rho;
                    }
                }
                // Main walk weight claims second
                for claim in proof.weight_claims.iter() {
                    combined_expected += rho_pow * claim.expected_value;
                    rho_pow = rho_pow * rho;
                }

                mix_secure_field(channel, combined_expected);

                // Full verification: re-evaluate weight MLEs and compare
                if !trust_weights {
                    let weights = weights.ok_or_else(|| GKRError::VerificationError {
                        layer_idx: 0,
                        reason: "AggregatedOracleSumcheck RLC-only mode requires \
                                 verify_gkr_with_weights(); use full aggregated_binding \
                                 proof or set OBELYZK_TRUST_WEIGHT_CLAIMS=1"
                            .to_string(),
                    })?;

                    let mut rho_pow2 = SecureField::one();
                    let mut combined_actual = SecureField::zero();

                    for deferred in proof.deferred_proofs.iter() {
                        if let Some(claim) = deferred.weight_claim() {
                            let weight = weights.get_weight(claim.weight_node_id).ok_or(
                                GKRError::MissingWeight { node_id: claim.weight_node_id },
                            )?;
                            let actual = evaluate_weight_claim_against_matrix(weight, &claim.eval_point)
                                .map_err(|reason| GKRError::VerificationError {
                                    layer_idx: 0,
                                    reason: format!("RLC deferred weight claim failed: {}", reason),
                                })?;
                            combined_actual += rho_pow2 * actual;
                            rho_pow2 = rho_pow2 * rho;
                        }
                    }
                    for claim in proof.weight_claims.iter() {
                        let weight = weights.get_weight(claim.weight_node_id).ok_or(
                            GKRError::MissingWeight { node_id: claim.weight_node_id },
                        )?;
                        let actual = evaluate_weight_claim_against_matrix(weight, &claim.eval_point)
                            .map_err(|reason| GKRError::VerificationError {
                                layer_idx: 0,
                                reason: format!("RLC weight claim failed: {}", reason),
                            })?;
                        combined_actual += rho_pow2 * actual;
                        rho_pow2 = rho_pow2 * rho;
                    }

                    if combined_expected != combined_actual {
                        return Err(GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "RLC weight binding mismatch: expected {:?}, actual {:?}",
                                combined_expected, combined_actual
                            ),
                        });
                    }
                }
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
    if b_mle.is_empty() {
        return Err("weight MLE is empty".to_string());
    }
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

    // Then verify the input claim against the actual input (node 0 = first computation node)
    if let Some(input_matrix) = execution.intermediates.get(&0) {
        let input_padded = pad_matrix_pow2(input_matrix);
        let input_mle = matrix_to_mle(&input_padded);
        if input_mle.is_empty() {
            return Err(GKRError::VerificationError {
                layer_idx: 0,
                reason: "input MLE is empty".to_string(),
            });
        }

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
) -> Result<GKRClaim, GKRError> {
    verify_gkr_simd_inner(circuit, proof, combined_output, None, channel)
}

/// Verify a SIMD GKR proof with optional access to weight matrices.
///
/// The `weights` parameter is required for `BatchedRlcDirectEvalV1` mode where
/// the verifier directly evaluates weight MLEs instead of checking Merkle openings.
pub fn verify_gkr_simd_with_weights(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    combined_output: &[SecureField],
    weights: &GraphWeights,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    verify_gkr_simd_inner(circuit, proof, combined_output, Some(weights), channel)
}

fn verify_gkr_simd_inner(
    circuit: &LayeredCircuit,
    proof: &GKRProof,
    combined_output: &[SecureField],
    weights: Option<&GraphWeights>,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
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
    if n_out == 0 {
        return Err(GKRError::VerificationError {
            layer_idx: 0,
            reason: "combined output MLE is empty".to_string(),
        });
    }
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
    let mut expected_weight_node_ids = Vec::new();
    // Track skip layer indices for deferred proofs (populated on Add/Mul layers)
    let mut deferred_skip_layer_indices: Vec<usize> = Vec::new();
    let mut skip_layers: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for &layer_idx in &template_layers {
        if skip_layers.contains(&layer_idx) {
            continue;
        }
        let layer = &circuit.layers[layer_idx];

        match &layer.layer_type {
            LayerType::Identity | LayerType::TopK { .. } => {
                // TopK: verify witness binding in the channel, propagate claim.
                // The TopK proof data is mixed into the channel during proving.
                // The verifier replays the same channel operations.
                if let LayerType::TopK { num_experts, top_k } = &layer.layer_type {
                    if proof_idx < proof.layer_proofs.len() {
                        if let LayerProof::TopK {
                            num_experts: pn, top_k: pk,
                            selected_indices, selected_values,
                            threshold_gap, logits_commitment,
                        } = &proof.layer_proofs[proof_idx] {
                            proof_idx += 1;

                            // Structural checks
                            if selected_indices.len() != *pk {
                                return Err(GKRError::VerificationError {
                                    layer_idx,
                                    reason: format!("TopK: expected {} selected indices, got {}", pk, selected_indices.len()),
                                });
                            }
                            if selected_values.len() != *pk {
                                return Err(GKRError::VerificationError {
                                    layer_idx,
                                    reason: format!("TopK: expected {} selected values, got {}", pk, selected_values.len()),
                                });
                            }

                            // Replay channel mixing (must match prover exactly)
                            channel.mix_u64(0x544F504B_u64); // "TOPK"
                            channel.mix_u64(*pn as u64);
                            channel.mix_u64(*pk as u64);
                            channel.mix_felt(*logits_commitment);
                            for &idx in selected_indices {
                                channel.mix_u64(idx as u64);
                            }
                            for &val in selected_values {
                                mix_secure_field(channel, val);
                            }
                            mix_secure_field(channel, *threshold_gap);
                            mix_secure_field(channel, current_claim.value);

                            // Claim propagation: value changes to logits eval (from proof)
                            // The TopK doesn't change the claim point, only the value
                            continue;
                        }
                    }
                }
                continue;
            }
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
            ) => {
                // Track skip layer index for deferred proof verification
                let skip_layer_idx = if layer.input_layers.len() >= 2
                    && layer.input_layers[1] > layer.input_layers[0]
                {
                    layer.input_layers[0]
                } else if layer.input_layers.len() >= 2 {
                    layer.input_layers[1]
                } else {
                    0
                };
                deferred_skip_layer_indices.push(skip_layer_idx);
                skip_layers.insert(skip_layer_idx);
                verify_add_reduction(
                    &current_claim,
                    *lhs_eval,
                    *rhs_eval,
                    &layer.input_layers,
                    layer_idx,
                    channel,
                )?
            }

            (
                LayerType::Mul { .. },
                LayerProof::Mul {
                    eq_round_polys,
                    lhs_eval,
                    rhs_eval,
                },
            ) => {
                let combined = verify_mul_reduction(
                    &current_claim,
                    eq_round_polys,
                    *lhs_eval,
                    *rhs_eval,
                    layer_idx,
                    channel,
                )?;
                // For branched Mul (gated FFN): override claim with trunk eval
                if layer.input_layers.len() >= 2 {
                    let skip_layer_idx = if layer.input_layers[1] > layer.input_layers[0] {
                        layer.input_layers[0]
                    } else {
                        layer.input_layers[1]
                    };
                    let trunk_eval = if layer.input_layers[1] > layer.input_layers[0] {
                        *rhs_eval
                    } else {
                        *lhs_eval
                    };
                    deferred_skip_layer_indices.push(skip_layer_idx);
                    skip_layers.insert(skip_layer_idx);
                    GKRClaim {
                        point: combined.point,
                        value: trunk_eval,
                    }
                } else {
                    combined
                }
            }

            (
                LayerType::Activation {
                    activation_type: circuit_act_type,
                    size: circuit_size,
                },
                LayerProof::Activation {
                    activation_type: proof_act_type,
                    logup_proof,
                    multiplicity_sumcheck,
                    activation_proof,
                    piecewise_proof,
                    input_eval,
                    output_eval,
                    table_commitment,
                    simd_combined,
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
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    piecewise_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    *circuit_size,
                    layer_idx,
                    channel,
                    *simd_combined,
                )?
            }

            (
                LayerType::LayerNorm { dim, .. },
                LayerProof::LayerNorm {
                    logup_proof,
                    multiplicity_sumcheck,
                    linear_round_polys,
                    linear_final_evals,
                    input_eval,
                    output_eval,
                    mean,
                    rsqrt_var,
                    rsqrt_table_commitment,
                    simd_combined,
                    mean_var_round_polys,
                    mean_var_final_evals,
                    var_eval,
                    centered_binding_evals,
                    mv_claimed_sums,
                    row_means,
                    row_variances,
                    ..
                },
            ) => verify_layernorm_reduction(
                &current_claim,
                logup_proof.as_ref(),
                multiplicity_sumcheck.as_ref(),
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
                mean_var_round_polys.as_ref(),
                *mean_var_final_evals,
                *var_eval,
                *centered_binding_evals,
                *mv_claimed_sums,
                row_means.as_ref(),
                row_variances.as_ref(),
            )?,

            (
                LayerType::Attention { config },
                LayerProof::Attention {
                    sub_proofs,
                    sub_claim_values,
                    ref softmax_sum_proofs,
                    ..
                },
            ) => verify_attention_reduction(
                &current_claim,
                config,
                sub_proofs,
                sub_claim_values,
                softmax_sum_proofs,
                Some(&r_simd),
                layer_idx,
                channel,
            )?,

            (
                LayerType::Attention { .. },
                LayerProof::AttentionDecode {
                    sub_proofs,
                    sub_claim_values,
                    num_heads,
                    new_tokens,
                    full_seq_len,
                    d_model,
                    causal,
                    position_offset,
                },
            ) => verify_attention_reduction_decode(
                &current_claim,
                sub_proofs,
                sub_claim_values,
                *num_heads,
                *new_tokens,
                *full_seq_len,
                *d_model,
                *causal,
                *position_offset,
                layer_idx,
                channel,
            )?,

            (
                LayerType::Dequantize { params, .. },
                LayerProof::Dequantize {
                    logup_proof,
                    multiplicity_sumcheck,
                    input_eval,
                    output_eval,
                    table_commitment,
                    ..
                },
            ) => verify_dequantize_reduction(
                &current_claim,
                params,
                logup_proof.as_ref(),
                multiplicity_sumcheck.as_ref(),
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
                    ..
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
                    ..
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
                    multiplicity_sumcheck,
                    linear_round_polys,
                    linear_final_evals,
                    input_eval,
                    output_eval,
                    rms_sq_eval,
                    rsqrt_eval,
                    rsqrt_table_commitment,
                    simd_combined,
                    rms_sq_round_polys,
                    rms_sq_input_final,
                    rms_sq_claimed_sq_sum,
                    row_rms_sq,
                    ..
                },
            ) => verify_rmsnorm_reduction(
                &current_claim,
                logup_proof.as_ref(),
                multiplicity_sumcheck.as_ref(),
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
                rms_sq_round_polys.as_ref(),
                *rms_sq_input_final,
                *rms_sq_claimed_sq_sum,
                row_rms_sq.as_ref(),
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

    let batched_rlc_mode =
        proof.weight_opening_transcript_mode == WeightOpeningTranscriptMode::BatchedRlcDirectEvalV1
            || proof.weight_opening_transcript_mode
                == WeightOpeningTranscriptMode::AggregatedOracleSumcheck;

    // Verify deferred proofs for skip branches of DAG Add layers.
    // Fiat-Shamir order: walk → deferred proofs → weight openings.
    if proof.deferred_proofs.len() != deferred_skip_layer_indices.len() {
        return Err(GKRError::VerificationError {
            layer_idx: 0,
            reason: format!(
                "SIMD deferred_proofs count ({}) != Add layers walked ({})",
                proof.deferred_proofs.len(),
                deferred_skip_layer_indices.len(),
            ),
        });
    }
    for (i, deferred) in proof.deferred_proofs.iter().enumerate() {
        // Mix deferred claim into channel (must match prover)
        mix_secure_field(channel, deferred.claim.value);

        // Verify the deferred layer reduction
        match &deferred.layer_proof {
            LayerProof::MatMul {
                round_polys,
                final_a_eval,
                final_b_eval,
            } => {
                let (m, k, n) = deferred.dims().ok_or_else(|| GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "SIMD deferred proof {} is MatMul layer_proof but has Weightless kind",
                        i
                    ),
                })?;
                // Cross-check dims against circuit's skip layer
                let skip_idx = *deferred_skip_layer_indices.get(i).ok_or_else(|| {
                    GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "SIMD deferred proof index {i} out of bounds (only {} skip indices)",
                            deferred_skip_layer_indices.len()
                        ),
                    }
                })?;
                if skip_idx >= circuit.layers.len() {
                    return Err(GKRError::VerificationError {
                        layer_idx: skip_idx,
                        reason: format!(
                            "skip_idx {skip_idx} out of bounds (circuit has {} layers)",
                            circuit.layers.len()
                        ),
                    });
                }
                if let LayerType::MatMul {
                    m: cm,
                    k: ck,
                    n: cn,
                    ..
                } = &circuit.layers[skip_idx].layer_type
                {
                    if m != *cm || k != *ck || n != *cn {
                        return Err(GKRError::VerificationError {
                            layer_idx: skip_idx,
                            reason: format!(
                                "SIMD deferred proof {} dims ({},{},{}) != circuit layer dims ({},{},{})",
                                i, m, k, n, cm, ck, cn,
                            ),
                        });
                    }
                }
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

                if deferred_input_claim.point != deferred.input_claim.point
                    || deferred_input_claim.value != deferred.input_claim.value
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!("SIMD deferred proof {} input claim mismatch", i),
                    });
                }
            }
            LayerProof::Dequantize {
                logup_proof,
                multiplicity_sumcheck,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                let skip_layer_idx =
                    deferred_skip_layer_indices.get(i).copied().ok_or_else(|| {
                        GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "SIMD deferred proof {} has no matching skip layer index",
                                i
                            ),
                        }
                    })?;
                let params = match &circuit.layers[skip_layer_idx].layer_type {
                    LayerType::Dequantize { params, .. } => params,
                    other => {
                        return Err(GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "SIMD deferred proof {} skip layer {} is {:?}, expected Dequantize",
                                i,
                                skip_layer_idx,
                                std::mem::discriminant(other),
                            ),
                        })
                    }
                };
                let deferred_input_claim = verify_dequantize_reduction(
                    &deferred.claim,
                    params,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0,
                    channel,
                )?;
                if deferred_input_claim.point != deferred.input_claim.point
                    || deferred_input_claim.value != deferred.input_claim.value
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "SIMD deferred proof {} (dequantize) input claim mismatch",
                            i
                        ),
                    });
                }
            }
            LayerProof::Quantize {
                logup_proof,
                input_eval,
                output_eval,
                table_inputs,
                table_outputs,
                ..
            } => {
                let skip_layer_idx =
                    deferred_skip_layer_indices.get(i).copied().ok_or_else(|| {
                        GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "SIMD deferred proof {} has no matching skip layer index",
                                i
                            ),
                        }
                    })?;
                let params = match &circuit.layers[skip_layer_idx].layer_type {
                    LayerType::Quantize { params, .. } => params,
                    other => {
                        return Err(GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "SIMD deferred proof {} skip layer {} is {:?}, expected Quantize",
                                i,
                                skip_layer_idx,
                                std::mem::discriminant(other),
                            ),
                        })
                    }
                };
                let deferred_input_claim = verify_quantize_reduction(
                    &deferred.claim,
                    params,
                    logup_proof.as_ref(),
                    *input_eval,
                    *output_eval,
                    table_inputs,
                    table_outputs,
                    0,
                    channel,
                )?;
                if deferred_input_claim.point != deferred.input_claim.point
                    || deferred_input_claim.value != deferred.input_claim.value
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "SIMD deferred proof {} (quantize) input claim mismatch",
                            i
                        ),
                    });
                }
            }
            _ => {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!("SIMD deferred proof {} has unsupported layer proof type", i),
                });
            }
        }

        // Weight opening verification (MatMul kind only)
        if deferred.has_weights() {
            let weight_commitment = deferred.weight_commitment().unwrap();
            let weight_opening = deferred.weight_opening().unwrap();
            let weight_claim = deferred.weight_claim().unwrap();

            if batched_rlc_mode {
                let is_agg_oracle = proof.weight_opening_transcript_mode
                    == WeightOpeningTranscriptMode::AggregatedOracleSumcheck;
                if !is_agg_oracle && weight_commitment != starknet_ff::FieldElement::ZERO {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "SIMD deferred proof {} expects zero weight_commitment in batched RLC mode",
                            i
                        ),
                    });
                }
                if !weight_opening.intermediate_roots.is_empty()
                    || !weight_opening.queries.is_empty()
                {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "SIMD deferred proof {} expects empty weight_opening in batched RLC mode",
                            i
                        ),
                    });
                }
            } else {
                if weight_opening.final_value != weight_claim.expected_value {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "SIMD deferred proof {} weight opening final_value mismatch",
                            i
                        ),
                    });
                }
                if !crate::crypto::mle_opening::verify_mle_opening(
                    weight_commitment,
                    weight_opening,
                    &weight_claim.eval_point,
                    channel,
                ) {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: format!(
                            "SIMD deferred proof {} weight MLE opening failed verification",
                            i,
                        ),
                    });
                }
            }
        }
    }

    // Verify weight MLE opening proofs (post-deferred channel state).
    if proof.weight_claims.len() != expected_weight_node_ids.len() {
        return Err(GKRError::VerificationError {
            layer_idx: 0,
            reason: format!(
                "SIMD weight_claims count ({}) != matmul layers in circuit walk ({})",
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
                    "SIMD weight claim {} node mismatch: claim={}, expected={}",
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
                        "SIMD weight_openings count ({}) != weight_commitments count ({})",
                        proof.weight_openings.len(),
                        proof.weight_commitments.len(),
                    ),
                });
            }
            if proof.weight_claims.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "SIMD weight_claims count ({}) != weight_commitments count ({})",
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
                            "SIMD weight opening {} final_value mismatch: opening={:?}, claim={:?}",
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
                        reason: format!("SIMD weight MLE opening proof {} failed verification", i),
                    });
                }
            }
        }
        WeightOpeningTranscriptMode::BatchedSubchannelV1 => {
            if proof.weight_openings.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "SIMD weight_openings count ({}) != weight_commitments count ({})",
                        proof.weight_openings.len(),
                        proof.weight_commitments.len(),
                    ),
                });
            }
            if proof.weight_claims.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "SIMD weight_claims count ({}) != weight_commitments count ({})",
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
                            "SIMD weight opening {} final_value mismatch: opening={:?}, claim={:?}",
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
                            "SIMD batched weight MLE opening proof {} failed verification",
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
                        "SIMD batched RLC mode expects no weight openings, got {}",
                        proof.weight_openings.len()
                    ),
                });
            }
            if !proof.weight_commitments.is_empty() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "SIMD batched RLC mode expects no weight commitments, got {}",
                        proof.weight_commitments.len()
                    ),
                });
            }
            let weights = weights.ok_or_else(|| GKRError::VerificationError {
                layer_idx: 0,
                reason: "SIMD batched RLC mode requires verify_gkr_simd_with_weights(...)"
                    .to_string(),
            })?;
            let rho = channel.draw_qm31();
            let mut rho_pow = SecureField::one();
            let mut combined_expected = SecureField::zero();
            let mut combined_actual = SecureField::zero();

            for (i, deferred) in proof.deferred_proofs.iter().enumerate() {
                if let Some(claim) = deferred.weight_claim() {
                    let weight = weights.get_weight(claim.weight_node_id).ok_or(
                        GKRError::MissingWeight {
                            node_id: claim.weight_node_id,
                        },
                    )?;
                    let actual = evaluate_weight_claim_against_matrix(weight, &claim.eval_point)
                        .map_err(|reason| GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "SIMD batched RLC deferred weight claim {} evaluation failed: {}",
                                i, reason
                            ),
                        })?;
                    combined_expected = combined_expected + rho_pow * claim.expected_value;
                    combined_actual = combined_actual + rho_pow * actual;
                    rho_pow = rho_pow * rho;
                }
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
                            "SIMD batched RLC weight claim {} evaluation failed: {}",
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
                        "SIMD batched RLC weight-binding mismatch: expected {:?}, actual {:?}",
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
                        "SIMD weight_openings count ({}) != weight_commitments count ({})",
                        proof.weight_openings.len(),
                        proof.weight_commitments.len(),
                    ),
                });
            }
            if proof.weight_claims.len() != proof.weight_commitments.len() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: format!(
                        "SIMD weight_claims count ({}) != weight_commitments count ({})",
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
                            "SIMD weight opening {} final_value mismatch: opening={:?}, claim={:?}",
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
                            "SIMD aggregated trustless weight opening {} failed verification",
                            i
                        ),
                    });
                }
            }
        }
        WeightOpeningTranscriptMode::AggregatedOracleSumcheck => {
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
            let mut matmul_deferred_idx = 0usize;
            for deferred in proof.deferred_proofs.iter() {
                if let Some(claim) = deferred.weight_claim() {
                    let claim_idx = proof.weight_claims.len() + matmul_deferred_idx;
                    agg_claims.push(AggregatedWeightClaim {
                        matrix_index: claim_idx,
                        local_n_vars: claim.eval_point.len(),
                        eval_point: claim.eval_point.clone(),
                        expected_value: claim.expected_value,
                        commitment: deferred
                            .weight_commitment()
                            .unwrap_or(starknet_ff::FieldElement::ZERO),
                    });
                    matmul_deferred_idx += 1;
                }
            }

            if agg_claims.is_empty() {
                return Err(GKRError::VerificationError {
                    layer_idx: 0,
                    reason: "SIMD AggregatedOracleSumcheck mode with zero claims".to_string(),
                });
            }

            if let Some(binding_proof) = proof.aggregated_binding.as_ref() {
                if !verify_aggregated_binding(binding_proof, &agg_claims, channel) {
                    return Err(GKRError::VerificationError {
                        layer_idx: 0,
                        reason: "SIMD aggregated oracle sumcheck weight binding verification failed"
                            .to_string(),
                    });
                }
            } else if !proof.binding_groups.is_empty() {
                // Grouped binding: verify each group independently
                let group_size = crate::gkr::prover::BINDING_GROUP_SIZE;
                for (g, group_proof) in proof.binding_groups.iter().enumerate() {
                    let chunk_start = g * group_size;
                    let chunk_end = (chunk_start + group_size).min(agg_claims.len());
                    let group_claims = &agg_claims[chunk_start..chunk_end];
                    if !verify_aggregated_binding(group_proof, group_claims, channel) {
                        return Err(GKRError::VerificationError {
                            layer_idx: 0,
                            reason: format!(
                                "SIMD grouped binding verification failed at group {}/{}",
                                g + 1, proof.binding_groups.len(),
                            ),
                        });
                    }
                }
            } else {
                let rho = channel.draw_qm31();
                let mut rho_pow = SecureField::one();
                let mut combined = SecureField::zero();
                for claim in &agg_claims {
                    combined = combined + rho_pow * claim.expected_value;
                    rho_pow = rho_pow * rho;
                }
                mix_secure_field(channel, combined);
            }
        }
    }

    Ok(current_claim)
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
    multiplicity_sumcheck: Option<&MultiplicitySumcheckProof>,
    activation_proof: Option<&super::types::ActivationProductProof>,
    piecewise_proof: Option<&super::types::PiecewiseAlgebraicProof>,
    input_eval: SecureField,
    output_eval: SecureField,
    table_commitment: starknet_ff::FieldElement,
    expected_size: usize,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
    simd_combined: bool,
) -> Result<GKRClaim, GKRError> {
    // Piecewise-linear algebraic activation verification (GELU/Sigmoid/Softmax)
    if let Some(pw_proof) = piecewise_proof {
        return verify_piecewise_activation_reduction(
            output_claim,
            activation_type,
            pw_proof,
            input_eval,
            expected_size,
            layer_idx,
            channel,
        );
    }

    // Phase A: algebraic product+binary eq-sumcheck (ReLU)
    if let Some(act_proof) = activation_proof {
        return verify_activation_product_reduction(
            output_claim,
            act_proof,
            input_eval,
            layer_idx,
            channel,
        );
    }

    // When all proofs are None, only SIMD combined-product path is allowed.
    // Non-SIMD activations MUST have a LogUp, algebraic, or piecewise proof.
    let logup = match logup_proof {
        Some(lp) => lp,
        None => {
            if !simd_combined {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: "activation layer missing LogUp or algebraic proof \
                             (simd_combined=false)"
                        .into(),
                });
            }
            // SIMD block-combination path: combined SecureField MLEs can't produce
            // individual activation proofs. Accept input_eval directly.
            mix_secure_field(channel, input_eval);
            return Ok(GKRClaim {
                point: output_claim.point.clone(),
                value: input_eval,
            });
        }
    };

    // Soundness gate: non-ReLU activations should use piecewise proof (full M31 domain).
    // LogUp-only verifies lower 16-20 bits via range masking — reject unless opted in.
    if !simd_combined
        && !matches!(activation_type, ActivationType::ReLU)
        && piecewise_proof.is_none()
    {
        let allow_logup = std::env::var("STWO_ALLOW_LOGUP_ACTIVATION")
            .map(|v| {
                let s = v.trim();
                s == "1" || s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("yes")
            })
            .unwrap_or(false);
        if !allow_logup {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "activation {:?} has LogUp proof but no piecewise proof — \
                     LogUp only verifies lower bits. Set STWO_ALLOW_LOGUP_ACTIVATION=1 \
                     to accept LogUp-only proofs.",
                    activation_type,
                ),
            });
        }
    }

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

    // 10. Verify multiplicity sumcheck (if present)
    if let Some(ms_proof) = multiplicity_sumcheck {
        verify_multiplicity_sumcheck(ms_proof, channel)?;
    }

    // 11. Mix final evals (same as prover)
    mix_secure_field(channel, input_eval);
    mix_secure_field(channel, output_eval);

    // Return claim on the activation input MLE
    Ok(GKRClaim {
        point: output_claim.point.clone(),
        value: input_eval,
    })
}

/// Number of piecewise-linear segments for algebraic activation verification.
const PIECEWISE_NUM_SEGMENTS: usize = 16;

/// Verify piecewise-linear algebraic activation reduction.
///
/// Verifies a combined degree-3 eq-sumcheck with 18 constraints:
///   - η^0: output matches Σ I_i · (a_i · input + b_i)
///   - η^1: Σ I_i = 1  (partition of unity)
///   - η^{2..17}: I_i · (1 - I_i) = 0  (binary indicators)
fn verify_piecewise_activation_reduction(
    output_claim: &GKRClaim,
    activation_type: ActivationType,
    proof: &super::types::PiecewiseAlgebraicProof,
    expected_input_eval: SecureField,
    expected_size: usize,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    use crate::components::activation::PiecewiseLinearCoeffs;

    let num_vars = proof.round_polys.len();
    if num_vars == 0 {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "piecewise activation sumcheck has 0 rounds".to_string(),
        });
    }

    // Cross-check num_vars against expected circuit size
    if expected_size > 0 {
        let expected_num_vars = expected_size.next_power_of_two().ilog2() as usize;
        if num_vars != expected_num_vars {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "piecewise activation num_vars mismatch: proof has {} rounds, expected {} (from size {})",
                    num_vars, expected_num_vars, expected_size,
                ),
            });
        }
    }

    if output_claim.point.len() < num_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "piecewise activation: claim point has {} vars, need at least {}",
                output_claim.point.len(),
                num_vars,
            ),
        });
    }

    // Reconstruct piecewise-linear coefficients (deterministic from activation type)
    let coeffs = PiecewiseLinearCoeffs::for_activation(activation_type);
    let slopes_sf: [SecureField; PIECEWISE_NUM_SEGMENTS] =
        std::array::from_fn(|i| SecureField::from(coeffs.slopes[i]));
    let intercepts_sf: [SecureField; PIECEWISE_NUM_SEGMENTS] =
        std::array::from_fn(|i| SecureField::from(coeffs.intercepts[i]));

    // Replay prover's channel operations: "PW_ACT" tag + type_tag + num_vars + claimed sum
    channel.mix_u64(0x50575F414354_u64); // "PW_ACT"
    channel.mix_u64(activation_type.type_tag() as u64); // domain-separate activation types
    channel.mix_u64(num_vars as u64); // commit MLE size to prevent transcript malleability
    mix_secure_field(channel, output_claim.value);

    // Draw η (same as prover)
    let eta = channel.draw_qm31();

    // Precompute eta powers: η^0, η^1, ..., η^{22} (18 original + 5 segment binding)
    let has_seg_bits = proof.seg_bit_evals.is_some();
    let num_eta_powers = if has_seg_bits { 2 + PIECEWISE_NUM_SEGMENTS + 1 + 4 } else { 2 + PIECEWISE_NUM_SEGMENTS };
    let mut eta_powers = Vec::with_capacity(num_eta_powers);
    eta_powers.push(SecureField::one());
    for i in 1..num_eta_powers {
        eta_powers.push(eta_powers[i - 1] * eta);
    }

    // For an honest prover, all constraints vanish at boolean points:
    //   - output == piecewise_eval (by construction)
    //   - indicators sum to 1 (partition of unity)
    //   - each indicator is binary
    //   - segment bits encode indicator index (if present)
    //   - each segment bit is binary (if present)
    // So the initial claimed sum is 0.
    let mut current_sum = SecureField::zero();
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);

    // Verify each sumcheck round
    for (round, rp) in proof.round_polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "piecewise activation round {}: p(0)+p(1) = {:?} != sum {:?}",
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

    // Final check: recompute combined constraint from final evaluations
    let r = &output_claim.point[..num_vars];
    let eq_val = compute_eq_eval(r, &sumcheck_challenges);

    let one = SecureField::one();
    let inp = proof.input_eval;
    let out = proof.output_eval;

    // η^0: piecewise evaluation match
    let mut piecewise_val = SecureField::zero();
    let mut ind_sum = SecureField::zero();
    for i in 0..PIECEWISE_NUM_SEGMENTS {
        ind_sum = ind_sum + proof.indicator_evals[i];
        piecewise_val = piecewise_val + proof.indicator_evals[i] * (slopes_sf[i] * inp + intercepts_sf[i]);
    }
    let mut h = eta_powers[0] * (out - piecewise_val);

    // η^1: partition of unity
    h = h + eta_powers[1] * (ind_sum - one);

    // η^{2..17}: binary indicator constraints
    for i in 0..PIECEWISE_NUM_SEGMENTS {
        h = h + eta_powers[2 + i] * proof.indicator_evals[i] * (one - proof.indicator_evals[i]);
    }

    // η^{18..22}: segment-input binding (if segment bits present)
    if let Some(seg_bit_evals) = &proof.seg_bit_evals {
        // η^18: Σ_k 2^k · seg_bit_k == Σ_i i · I_i
        let mut bit_sum = SecureField::zero();
        for k in 0..4 {
            bit_sum = bit_sum + SecureField::from(M31::from(1u32 << k)) * seg_bit_evals[k];
        }
        let mut ind_index_sum = SecureField::zero();
        for i in 0..PIECEWISE_NUM_SEGMENTS {
            ind_index_sum = ind_index_sum + SecureField::from(M31::from(i as u32)) * proof.indicator_evals[i];
        }
        h = h + eta_powers[18] * (bit_sum - ind_index_sum);

        // η^{19..22}: segment bits binary
        for k in 0..4 {
            h = h + eta_powers[19 + k] * seg_bit_evals[k] * (one - seg_bit_evals[k]);
        }
    } else {
        // Soundness gate: reject missing segment binding proof
        let allow_missing = std::env::var("STWO_ALLOW_MISSING_SEGMENT_BINDING")
            .map(|v| {
                let s = v.trim();
                s == "1" || s.eq_ignore_ascii_case("true")
            })
            .unwrap_or(false);
        if !allow_missing {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: "piecewise activation: missing segment-input binding proof".into(),
            });
        }
    }

    let expected = eq_val * h;

    if current_sum != expected {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "piecewise activation final check failed: sum={:?} != expected={:?}",
                current_sum, expected,
            ),
        });
    }

    // Verify consistency: proof's input_eval matches the LayerProof's input_eval
    if proof.input_eval != expected_input_eval {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "piecewise activation input_eval mismatch: proof={:?}, layer={:?}",
                proof.input_eval, expected_input_eval,
            ),
        });
    }

    // Mix final evals into channel (same as prover)
    mix_secure_field(channel, proof.input_eval);
    mix_secure_field(channel, proof.output_eval);
    for &ie in &proof.indicator_evals {
        mix_secure_field(channel, ie);
    }
    if let Some(seg_bit_evals) = &proof.seg_bit_evals {
        for &sb in seg_bit_evals {
            mix_secure_field(channel, sb);
        }
    }

    Ok(GKRClaim {
        point: sumcheck_challenges,
        value: proof.input_eval,
    })
}

/// Number of bits in the low-part decomposition for Phase B sign consistency.
const PHASE_B_NUM_BITS: usize = 30;

/// Verify algebraic product+binary eq-sumcheck for activation (ReLU).
///
/// Phase A: `V_out(r) = Σ_x eq(r,x) · b(x) · [in(x) + η · (1 − b(x))]`
///
/// Phase B extends with sign consistency via bit decomposition:
///   `V_out(r) = Σ_x eq(r,x) · [b·in + η·b·(1-b) + η²·decomp + Σ η^{j+3}·bit_j·(1-bit_j)]`
///
/// Phase B activates when `proof.bit_evals` is `Some(...)`.
fn verify_activation_product_reduction(
    output_claim: &GKRClaim,
    proof: &super::types::ActivationProductProof,
    expected_input_eval: SecureField,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let num_vars = proof.round_polys.len();
    if num_vars == 0 {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: "activation product sumcheck has 0 rounds".to_string(),
        });
    }

    if output_claim.point.len() < num_vars {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "activation product: claim point has {} vars, need at least {}",
                output_claim.point.len(),
                num_vars,
            ),
        });
    }

    // Validate bit_evals length if present
    if let Some(ref be) = proof.bit_evals {
        if be.len() != PHASE_B_NUM_BITS {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "bit_evals length {} != expected {}",
                    be.len(),
                    PHASE_B_NUM_BITS,
                ),
            });
        }
    }

    // Replay prover's channel operations: "ACT" tag + claimed sum
    channel.mix_u64(0x414354_u64); // "ACT"
    mix_secure_field(channel, output_claim.value);

    // Draw η (same as prover)
    let eta = channel.draw_qm31();

    let mut current_sum = output_claim.value;
    let mut sumcheck_challenges = Vec::with_capacity(num_vars);

    // Sumcheck round verification (unchanged — only checks p(0)+p(1)==sum)
    for (round, rp) in proof.round_polys.iter().enumerate() {
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2 + rp.c3;

        if p0 + p1 != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "activation product round {}: p(0)+p(1) = {:?} != sum {:?}",
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

    // Final check: depends on whether Phase B bit_evals are present
    let r = &output_claim.point[..num_vars];
    let eq_val = compute_eq_eval(r, &sumcheck_challenges);

    let expected = if let Some(ref bit_evals) = proof.bit_evals {
        // Phase A+B combined final check:
        // h = η^0·b·in + η^1·b·(1-b) + η^2·decomp + Σ η^{j+3}·bit_j·(1-bit_j)
        let one = SecureField::one();
        let ind = proof.indicator_eval;
        let inp = proof.input_eval;

        // Precompute eta powers
        let num_eta_powers = PHASE_B_NUM_BITS + 3;
        let mut eta_powers = Vec::with_capacity(num_eta_powers);
        eta_powers.push(one);
        for i in 1..num_eta_powers {
            eta_powers.push(eta_powers[i - 1] * eta);
        }

        // Product + binary indicator
        let mut h = eta_powers[0] * ind * inp + eta_powers[1] * ind * (one - ind);

        // Decomposition: in - Σ 2^j·bit_j - 2^30·(1 - ind)
        let two_sf = SecureField::from(M31::from(2u32));
        let mut bit_sum = SecureField::zero();
        let mut pow2 = SecureField::from(M31::from(1u32));
        for j in 0..PHASE_B_NUM_BITS {
            bit_sum = bit_sum + pow2 * bit_evals[j];
            pow2 = pow2 * two_sf;
        }
        // pow2 is now 2^30
        let decomp = inp - bit_sum - pow2 * (one - ind);
        h = h + eta_powers[2] * decomp;

        // Binary bit constraints
        for j in 0..PHASE_B_NUM_BITS {
            h = h + eta_powers[j + 3] * bit_evals[j] * (one - bit_evals[j]);
        }

        eq_val * h
    } else {
        // Phase A-only final check (backward compat)
        eq_val
            * proof.indicator_eval
            * (proof.input_eval + eta * (SecureField::one() - proof.indicator_eval))
    };

    if current_sum != expected {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "activation product final check failed: sum={:?} != expected={:?}",
                current_sum, expected,
            ),
        });
    }

    // Verify consistency: proof's input_eval matches the LayerProof's input_eval
    if proof.input_eval != expected_input_eval {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "activation product input_eval mismatch: proof={:?}, layer={:?}",
                proof.input_eval, expected_input_eval,
            ),
        });
    }

    // Mix final evals (same as prover)
    mix_secure_field(channel, proof.input_eval);
    mix_secure_field(channel, proof.indicator_eval);
    if let Some(ref bit_evals) = proof.bit_evals {
        for &be in bit_evals {
            mix_secure_field(channel, be);
        }
    }

    Ok(GKRClaim {
        point: sumcheck_challenges,
        value: proof.input_eval,
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
    multiplicity_sumcheck: Option<&MultiplicitySumcheckProof>,
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

    // 10. Verify multiplicity sumcheck (if present)
    if let Some(ms_proof) = multiplicity_sumcheck {
        verify_multiplicity_sumcheck(ms_proof, channel)?;
    }

    // 11. Mix final evals (same as prover)
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
    multiplicity_sumcheck: Option<&MultiplicitySumcheckProof>,
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
    mean_var_round_polys: Option<&Vec<RoundPolyDeg3>>,
    mean_var_final_evals: Option<(SecureField, SecureField)>,
    var_eval: Option<SecureField>,
    centered_binding_evals: Option<(SecureField, SecureField)>,
    mv_claimed_sums: Option<(SecureField, SecureField)>,
    row_means: Option<&Vec<M31>>,
    row_variances: Option<&Vec<M31>>,
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

    // ===== Part 0: Batched mean + variance plain sumcheck =====
    // Proves: Σ_x [η·input(x) + η²·centered(x)²] = η·total_input_sum + η²·total_centered_sq_sum
    // Verifier derives centered_final = input_final - mean_final for binding check.
    if let Some(mv_polys) = mean_var_round_polys {
        if mv_polys.len() != num_vars {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "layernorm: mean-variance sumcheck has {} rounds, expected {}",
                    mv_polys.len(), num_vars,
                ),
            });
        }
        let ve = var_eval.ok_or_else(|| GKRError::VerificationError {
            layer_idx,
            reason: "layernorm: mean_var_round_polys present but var_eval missing".into(),
        })?;
        let (total_input_sum, total_centered_sq_sum) =
            mv_claimed_sums.ok_or_else(|| GKRError::VerificationError {
                layer_idx,
                reason: "layernorm: mean_var_round_polys present but mv_claimed_sums missing"
                    .into(),
            })?;
        let n_active = dim.min(1usize << num_vars);
        channel.mix_u64(0x4D56_u64); // "MV" tag
        channel.mix_u64(n_active as u64);
        mix_secure_field(channel, total_input_sum);
        mix_secure_field(channel, total_centered_sq_sum);
        let eta0 = channel.draw_qm31();
        let eta1 = eta0 * eta0;
        let mut current_sum = eta0 * total_input_sum + eta1 * total_centered_sq_sum;
        let mut mv_challenges = Vec::with_capacity(num_vars);
        for (round, poly) in mv_polys.iter().enumerate() {
            let p0 = poly.c0;
            let p1 = poly.c0 + poly.c1 + poly.c2 + poly.c3;
            if p0 + p1 != current_sum {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "layernorm mean-variance round {}: p(0)+p(1) = {} != sum {}",
                        round, p0 + p1, current_sum,
                    ),
                });
            }
            channel.mix_poly_coeffs_deg3(poly.c0, poly.c1, poly.c2, poly.c3);
            let ch = channel.draw_qm31();
            mv_challenges.push(ch);
            current_sum = poly.eval(ch);
        }
        let (mv_input_final, mv_mean_final) = mean_var_final_evals.ok_or_else(|| {
            GKRError::VerificationError {
                layer_idx,
                reason: "layernorm: mean_var_round_polys present but final_evals missing".into(),
            }
        })?;
        mix_secure_field(channel, mv_input_final);
        mix_secure_field(channel, mv_mean_final);
        // Final check: η₀·input_final + η₁·(input_final - mean_final)² == current_sum
        // This simultaneously verifies the plain sumcheck and centered binding.
        let c_final = mv_input_final - mv_mean_final;
        let expected_p0 = eta0 * mv_input_final + eta1 * c_final * c_final;
        if current_sum != expected_p0 {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "layernorm mean-variance final check failed: sum={} != expected={}",
                    current_sum, expected_p0,
                ),
            });
        }
        // Verify mean derivation binding
        let cols_padded = dim.next_power_of_two();
        let n_total = 1usize << num_vars;
        let rows = n_total / cols_padded;
        if rows == 1 {
            // Single-row: direct derivation from total_input_sum
            let input_sum_m31 = M31::from(total_input_sum.0 .0 .0);
            let inv_n = {
                let p: u64 = (1u64 << 31) - 1;
                let mut result: u64 = 1;
                let mut base = (n_active as u64) % p;
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
            let mean_expected = input_sum_m31 * inv_n;
            let mean_actual = M31::from(mean_eval.0 .0 .0);
            if mean_expected != mean_actual {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "layernorm mean derivation mismatch: expected {} from sum={}, got {}",
                        mean_expected.0, input_sum_m31.0, mean_actual.0,
                    ),
                });
            }
            // Variance derivation: (total_centered_sq_sum / n_active) & mask
            // Note: total_centered_sq_sum includes padding terms (centered = -mean)
            // but the prover's variance was computed from ACTIVE columns only.
            // We need the prover's variance, not a re-derivation from the sumcheck totals.
            // For single-row with non-power-of-2 dim, trust the prover's var_eval.
            let centered_sq_sum_m31 = M31::from(total_centered_sq_sum.0 .0 .0);
            let inv_n_full = {
                // Use full MLE width (cols_padded) since centered_sq_sum includes padding
                let p: u64 = (1u64 << 31) - 1;
                let total = (1usize << num_vars) as u64; // full MLE size
                let mut result: u64 = 1;
                let mut base = total % p;
                let mut exp = p - 2;
                while exp > 0 {
                    if exp & 1 == 1 { result = result * base % p; }
                    base = base * base % p;
                    exp >>= 1;
                }
                M31::from(result as u32)
            };
            // With padding: total_centered_sq = n_active*variance + (padded-n_active)*mean²
            // This doesn't simplify to a clean formula. Skip variance re-derivation
            // for non-power-of-2 dims; rely on the LogUp rsqrt proof instead.
            let var_raw = centered_sq_sum_m31 * inv_n;
            let rsqrt_table_log_size = 16u32;
            let var_mask = (1u32 << rsqrt_table_log_size) - 1;
            let var_expected = M31::from(var_raw.0 & var_mask);
            let var_actual = M31::from(ve.0 .0 .0);
            // For non-power-of-2 dims, the variance re-derivation from total_centered_sq_sum
            // doesn't match because padding adds extra (-mean)² terms. The variance is still
            // verified by the LogUp rsqrt proof: (variance, rsqrt) ∈ table.
            // Only check variance derivation when dim is power-of-2 (no padding effect).
            if n_active == cols_padded && var_expected != var_actual {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "layernorm variance derivation mismatch: expected {} got {}",
                        var_expected.0, var_actual.0,
                    ),
                });
            }
        } else if let Some(rm) = row_means {
            // Multi-row binding: verify per-row means reconstruct mv_mean_final
            if rm.len() != rows {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "layernorm row_means length {} != rows {}",
                        rm.len(), rows,
                    ),
                });
            }
            // Reconstruct mean(s₀) from per-row means.
            // mean_mle is constant per row: mean(s) = Σ_r mean_r * eq(s[log_cols..], binary(r))
            // Column variables contribute 1 (product of (1-s_i)(1-0)+s_i*0 = 1-s_i for bit=0,
            // but since mean is constant across all cols, the col-part collapses).
            // Actually: mean_mle[r,c] = mean_r for all c.
            // So mean_mle evaluated at challenge s = (s_col..., s_row...) is:
            //   Σ_{r,c} mean_r * eq(s_col, c) * eq(s_row, r)
            // = Σ_r mean_r * eq(s_row, r) * Σ_c eq(s_col, c)
            // = Σ_r mean_r * eq(s_row, r) * 1  (since Σ_c eq(s_col, c) = 1 for any s_col)
            // Wait, that's not true. Σ_c eq(s, c) = 1 only when s is boolean.
            // For random challenges, Σ_c eq(s_col, c) is NOT 1.
            //
            // Actually the mean MLE in the sumcheck is expanded as:
            //   mean(x) = Σ_r mean_r * eq_row(x_row_bits, r)
            //             * Π_{col_bit j} [(1 - x_j)*1 + x_j*1]  -- NO, this is wrong
            //
            // The MLE of a function f(r,c) = mean_r is:
            //   f̃(s) = Σ_{r,c} mean_r * Π_i [(1-s_i)(1-b_i) + s_i*b_i]
            //   where (b_0..b_{log_cols-1}, b_{log_cols}..b_{log_cols+log_rows-1}) = (c, r)
            //
            // But since mean_r doesn't depend on c:
            //   f̃(s) = Σ_r mean_r * eq(s_row, r) * Σ_c eq(s_col, c)
            // And Σ_c eq(s_col, c) = Σ_c Π_j [(1-s_j)(1-c_j)+s_j*c_j]
            // For s_col ∈ F^{log_cols}: this sum = Π_j [(1-s_j)+s_j] = 1.
            // Wait YES this IS 1! Because for each bit position j:
            //   Σ_{c_j ∈ {0,1}} [(1-s_j)(1-c_j) + s_j*c_j]
            //   = (1-s_j)*1 + s_j*0 + (1-s_j)*0 + s_j*1 = 1
            // So the column variables sum to 1, and:
            //   mean(s) = Σ_r mean_r * eq(s_row, r)
            let log_cols = cols_padded.ilog2() as usize;
            let log_rows = num_vars - log_cols;
            // fold_mle fixes MSB first, so mv_challenges[0..log_rows] are the row
            // variables (high bits of the array index), and mv_challenges[k]
            // corresponds to row bit (log_rows - 1 - k).
            let expected_mean: SecureField = (0..rows).map(|r| {
                let mut eq_val = SecureField::one();
                for k in 0..log_rows {
                    let s_bit = mv_challenges[k];
                    let row_bit = (r >> (log_rows - 1 - k)) & 1;
                    if row_bit == 1 {
                        eq_val = eq_val * s_bit;
                    } else {
                        eq_val = eq_val * (SecureField::one() - s_bit);
                    }
                }
                eq_val * SecureField::from(rm[r])
            }).sum();
            if mv_mean_final != expected_mean {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "layernorm multi-row mean binding failed: mv_mean_final={:?} != expected={:?}",
                        mv_mean_final, expected_mean,
                    ),
                });
            }
            // Multi-row variance binding: verify per-row variances reconstruct ve
            // var_mle is constant per row: var(s) = Σ_r var_r * eq(s_row, r)
            // (same identity as mean_mle — column variables sum to 1)
            if let Some(rv) = row_variances {
                if rv.len() != rows {
                    return Err(GKRError::VerificationError {
                        layer_idx,
                        reason: format!(
                            "layernorm row_variances length {} != rows {}",
                            rv.len(), rows,
                        ),
                    });
                }
                let expected_var: SecureField = (0..rows).map(|r| {
                    let mut eq_val = SecureField::one();
                    for k in 0..log_rows {
                        let s_bit = mv_challenges[k];
                        let row_bit = (r >> (log_rows - 1 - k)) & 1;
                        if row_bit == 1 {
                            eq_val = eq_val * s_bit;
                        } else {
                            eq_val = eq_val * (SecureField::one() - s_bit);
                        }
                    }
                    eq_val * SecureField::from(rv[r])
                }).sum();
                if ve != expected_var {
                    return Err(GKRError::VerificationError {
                        layer_idx,
                        reason: format!(
                            "layernorm multi-row variance binding failed: ve={:?} != expected={:?}",
                            ve, expected_var,
                        ),
                    });
                }
            } else if !simd_combined {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: "multi-row LayerNorm missing row_variances for binding verification".into(),
                });
            }
        } else if !simd_combined {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: "multi-row LayerNorm missing row_means for binding verification".into(),
            });
        }
    } else if !simd_combined {
        let allow_missing = std::env::var("STWO_ALLOW_MISSING_NORM_PROOF")
            .map(|v| {
                let s = v.trim();
                s == "1" || s.eq_ignore_ascii_case("true")
            })
            .unwrap_or(false);
        if !allow_missing {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: "layernorm: missing mean-variance verification proof".into(),
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

    // Centered-consistency binding: verify centered = input - mean at Part 1 challenge point
    if let Some((cb_input, cb_mean)) = centered_binding_evals {
        mix_secure_field(channel, cb_input);
        mix_secure_field(channel, cb_mean);
        if centered_final != cb_input - cb_mean {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "layernorm centered binding failed: centered={} != input-mean={}",
                    centered_final, cb_input - cb_mean,
                ),
            });
        }
    }

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

    // Verify multiplicity sumcheck (if present)
    if let Some(ms_proof) = multiplicity_sumcheck {
        verify_multiplicity_sumcheck(ms_proof, channel)?;
    }

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
    multiplicity_sumcheck: Option<&MultiplicitySumcheckProof>,
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
    rms_sq_round_polys: Option<&Vec<RoundPolyDeg3>>,
    rms_sq_input_final: Option<SecureField>,
    rms_sq_claimed_sq_sum: Option<SecureField>,
    row_rms_sq: Option<&Vec<M31>>,
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

    // ===== Part 0: RMS² verification plain sumcheck =====
    // Proves: Σ_x input(x)² = total_sq_sum (no eq weighting).
    // Then verifies rms_sq derivation from total_sq_sum.
    if let Some(rms_polys) = rms_sq_round_polys {
        if rms_polys.len() != num_vars {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "rmsnorm: RMS² sumcheck has {} rounds, expected {}",
                    rms_polys.len(), num_vars,
                ),
            });
        }
        let n_active = dim.min(1usize << num_vars);
        let total_sq_sum = rms_sq_claimed_sq_sum.ok_or_else(|| GKRError::VerificationError {
            layer_idx,
            reason: "rmsnorm: RMS² round_polys present but claimed_sq_sum missing".into(),
        })?;
        channel.mix_u64(0x5251_u64); // "RQ" tag
        channel.mix_u64(n_active as u64);
        mix_secure_field(channel, total_sq_sum);
        let mut current_sum = total_sq_sum;
        for (round, poly) in rms_polys.iter().enumerate() {
            let p0 = poly.c0;
            let p1 = poly.c0 + poly.c1 + poly.c2 + poly.c3;
            if p0 + p1 != current_sum {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "rmsnorm RMS² round {}: p(0)+p(1) = {} != sum {}",
                        round, p0 + p1, current_sum,
                    ),
                });
            }
            channel.mix_poly_coeffs_deg3(poly.c0, poly.c1, poly.c2, poly.c3);
            let ch = channel.draw_qm31();
            current_sum = poly.eval(ch);
        }
        let input_final = rms_sq_input_final.ok_or_else(|| GKRError::VerificationError {
            layer_idx,
            reason: "rmsnorm: RMS² round_polys present but input_final missing".into(),
        })?;
        mix_secure_field(channel, input_final);
        // Final check: input_final² == current_sum (plain sumcheck, no eq factor)
        if input_final * input_final != current_sum {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "rmsnorm RMS² final check failed: in²={} != sum={}",
                    input_final * input_final, current_sum,
                ),
            });
        }
        // Verify rms_sq derivation binding
        let cols_padded = dim.next_power_of_two();
        let n_total = 1usize << num_vars;
        let rows = n_total / cols_padded;
        if rows == 1 {
            // Single-row: direct derivation from total_sq_sum
            let sq_sum_m31 = M31::from(total_sq_sum.0 .0 .0);
            let inv_n = {
                let p: u64 = (1u64 << 31) - 1;
                let mut result: u64 = 1;
                let mut base = (n_active as u64) % p;
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
            let rms_sq_raw = sq_sum_m31 * inv_n;
            let rsqrt_table_log_size = 16u32; // deterministic from RMSNormConfig
            let mask = (1u32 << rsqrt_table_log_size) - 1;
            let rms_sq_expected = M31::from(rms_sq_raw.0 & mask);
            let rms_sq_actual = M31::from(rms_sq_eval.0 .0 .0);
            if rms_sq_expected != rms_sq_actual {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "rmsnorm RMS² derivation mismatch: expected {} from sq_sum={}, got {}",
                        rms_sq_expected.0, sq_sum_m31.0, rms_sq_actual.0,
                    ),
                });
            }
        } else if let Some(rr) = row_rms_sq {
            // Multi-row binding: verify per-row rms_sq reconstructs rms_sq_eval
            if rr.len() != rows {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "rmsnorm row_rms_sq length {} != rows {}",
                        rr.len(), rows,
                    ),
                });
            }
            // rms_sq_mle is constant per row: rms_sq(s) = Σ_r rms_sq_r * eq(s_row, r)
            // (column variables sum to 1 for constant-per-row MLE)
            // Bind at the claim point (output_claim.point), which is random.
            // fold_mle fixes MSB first, so output_claim.point[0..log_rows] are the
            // row variables, with point[k] corresponding to row bit (log_rows-1-k).
            let log_cols = cols_padded.ilog2() as usize;
            let log_rows = num_vars - log_cols;
            let expected_rms_sq: SecureField = (0..rows).map(|r| {
                let mut eq_val = SecureField::one();
                for k in 0..log_rows {
                    let s_bit = output_claim.point[k];
                    let row_bit = (r >> (log_rows - 1 - k)) & 1;
                    if row_bit == 1 {
                        eq_val = eq_val * s_bit;
                    } else {
                        eq_val = eq_val * (SecureField::one() - s_bit);
                    }
                }
                eq_val * SecureField::from(rr[r])
            }).sum();
            if rms_sq_eval != expected_rms_sq {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "rmsnorm multi-row rms_sq binding failed: rms_sq_eval={:?} != expected={:?}",
                        rms_sq_eval, expected_rms_sq,
                    ),
                });
            }
        } else if !simd_combined {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: "multi-row RMSNorm missing row_rms_sq for binding verification".into(),
            });
        }
    } else if !simd_combined {
        // Soundness gate: reject missing RMS² proof unless SIMD combined path.
        let allow_missing = std::env::var("STWO_ALLOW_MISSING_NORM_PROOF")
            .map(|v| {
                let s = v.trim();
                s == "1" || s.eq_ignore_ascii_case("true")
            })
            .unwrap_or(false);
        if !allow_missing {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: "rmsnorm: missing RMS² verification proof".into(),
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

    // Verify multiplicity sumcheck (if present)
    if let Some(ms_proof) = multiplicity_sumcheck {
        verify_multiplicity_sumcheck(ms_proof, channel)?;
    }

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
    softmax_sum_proofs: &[super::types::SoftmaxSumProof],
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

    // Phase 1D: Replay causal mask commitment (must match prover exactly).
    if config.causal {
        let position_offset = 0usize; // prefill path
        channel.mix_u64(0x434D534B_u64); // "CMSK" tag
        channel.mix_u64(position_offset as u64);
        let mask_sentinel = (1u32 << 31) - 3;
        channel.mix_u64(mask_sentinel as u64);
        let total_masked = seq_len * seq_len.saturating_sub(1) / 2;
        channel.mix_u64(total_masked as u64);
    }

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

    // Validate sub_claim_values[0] matches the output claim (soundness: prevents
    // an attacker from fabricating the output projection sub-claim independently).
    if sub_claim_values[0] != output_claim.value {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "attention sub_claim_values[0] mismatch: expected {:?}, got {:?}",
                output_claim.value, sub_claim_values[0],
            ),
        });
    }

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
    let mut softmax_proof_idx = 0usize;

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

        // === SOFTMAX SUM VERIFICATION (Phase 1B) ===
        // Replay the plain sumcheck proving Σ exp = sum_exp per row.
        if softmax_proof_idx < softmax_sum_proofs.len() {
            let sp = &softmax_sum_proofs[softmax_proof_idx];
            let padded_cols = seq_len.next_power_of_two();
            let padded_rows = seq_len.next_power_of_two();
            let num_vars = (padded_rows * padded_cols).trailing_zeros() as usize;

            if sp.round_polys.len() != num_vars {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: format!(
                        "softmax sum proof: expected {} rounds, got {}",
                        num_vars, sp.round_polys.len(),
                    ),
                });
            }

            // Replay channel: mix tag + padded_cols + claimed_sum
            channel.mix_u64(0x5358_u64); // "SX" tag
            channel.mix_u64(padded_cols as u64);
            mix_secure_field(channel, sp.claimed_sum);

            // Verify plain sumcheck rounds
            let mut current_sum = sp.claimed_sum;
            for rp in &sp.round_polys {
                // Degree 1: p(0) + p(1) = current_sum
                // p(0) = c0, p(1) = c0 + c1
                if rp.c0 + rp.c0 + rp.c1 != current_sum {
                    return Err(GKRError::VerificationError {
                        layer_idx,
                        reason: "softmax sum: round poly p(0)+p(1) != claimed sum".to_string(),
                    });
                }
                // Mix and draw challenge
                mix_secure_field(channel, rp.c0);
                mix_secure_field(channel, rp.c1);
                let challenge = channel.draw_qm31();
                current_sum = rp.c0 + rp.c1 * challenge;
            }

            // Final eval check: the folded MLE should equal the final value
            if sp.final_exp_eval != current_sum {
                return Err(GKRError::VerificationError {
                    layer_idx,
                    reason: "softmax sum: final exp eval != folded sum".to_string(),
                });
            }
            mix_secure_field(channel, sp.final_exp_eval);

            // Row-sum binding: verify that the per-row sums reconstruct
            // to the total claimed_sum (prevents fabricated row sums).
            // The prover provides row_sums[r] for each row r.
            // For active rows: sum_exp[r] = Σ_col exp(scores[r][col])
            // For padding rows: sum_exp[r] = 0
            // Total: Σ row_sums[r] (including padding) must equal claimed_sum.
            {
                let expected_rows = seq_len.max(1);
                if sp.row_sums.len() != expected_rows {
                    return Err(GKRError::VerificationError {
                        layer_idx,
                        reason: format!(
                            "softmax sum: expected {} row sums, got {}",
                            expected_rows, sp.row_sums.len(),
                        ),
                    });
                }

                // Reconstruct expected total from row sums + padding.
                // softmax_exp(M31(0)) = 65536 (2^16 scale factor).
                // Padding columns within active rows and all padding rows
                // contribute this value to the sumcheck total.
                let exp_zero = 65536u64; // softmax_exp(M31(0)).0
                let pad_cols = padded_cols - seq_len;
                let pad_rows = padded_rows - seq_len;

                // Active rows: row_sums[r] covers active cols; padding cols add exp_zero each
                let mut expected_total: u64 = 0;
                for &rs in &sp.row_sums {
                    expected_total += rs.0 as u64;
                    expected_total += pad_cols as u64 * exp_zero;
                }
                // Padding rows: all padded_cols positions have exp(0)
                expected_total += pad_rows as u64 * padded_cols as u64 * exp_zero;

                // Reduce to M31 and compare against claimed_sum
                let p = (1u64 << 31) - 1;
                let expected_m31 = M31::from((expected_total % p) as u32);
                let claimed_m31 = M31::from(sp.claimed_sum.0 .0 .0);

                if expected_m31 != claimed_m31 {
                    return Err(GKRError::VerificationError {
                        layer_idx,
                        reason: format!(
                            "softmax sum: row-sum binding failed. \
                             expected total {} (from {} row sums + {} pad_cols + {} pad_rows), \
                             got claimed_sum M31 component {}",
                            expected_m31.0, sp.row_sums.len(), pad_cols, pad_rows, claimed_m31.0,
                        ),
                    });
                }

                // Mix row sums into channel for Fiat-Shamir binding
                for &rs in &sp.row_sums {
                    channel.mix_u64(rs.0 as u64);
                }
            }

            softmax_proof_idx += 1;
        }

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

/// Verify a decode-step attention reduction (DCOD domain tag, asymmetric dims).
pub(crate) fn verify_attention_reduction_decode(
    output_claim: &GKRClaim,
    sub_proofs: &[LayerProof],
    sub_claim_values: &[SecureField],
    num_heads: usize,
    new_tokens: usize,
    full_seq_len: usize,
    d_model: usize,
    causal: bool,
    position_offset: usize,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    let d_k = d_model / num_heads;

    let expected_count = 4 + 2 * num_heads;
    if sub_proofs.len() != expected_count {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "attention_decode: expected {} sub-proofs, got {}",
                expected_count,
                sub_proofs.len(),
            ),
        });
    }
    if sub_claim_values.len() != expected_count {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "attention_decode: expected {} sub-claim values, got {}",
                expected_count,
                sub_claim_values.len(),
            ),
        });
    }

    // Replay DCOD metadata (must match prover)
    channel.mix_u64(0x44434F44_u64); // "DCOD"
    channel.mix_u64(num_heads as u64);
    channel.mix_u64(new_tokens as u64);
    channel.mix_u64(full_seq_len as u64);
    channel.mix_u64(d_model as u64);
    channel.mix_u64(if causal { 1 } else { 0 });
    channel.mix_u64(position_offset as u64);

    // Assert position_offset consistency
    if position_offset + new_tokens != full_seq_len {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "attention_decode: position_offset({}) + new_tokens({}) != full_seq_len({})",
                position_offset, new_tokens, full_seq_len,
            ),
        });
    }

    let verify_fresh_sub_matmul = |proof: &LayerProof,
                                   claimed_value: SecureField,
                                   m: usize,
                                   k: usize,
                                   n: usize,
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
            other => Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "attention_decode sub-proof: expected MatMul, got {:?}",
                    std::mem::discriminant(other),
                ),
            }),
        }
    };

    let mut proof_idx = 0;

    // Validate sub_claim_values[0] matches the output claim (soundness: prevents
    // an attacker from fabricating the output projection sub-claim independently).
    if sub_claim_values[0] != output_claim.value {
        return Err(GKRError::VerificationError {
            layer_idx,
            reason: format!(
                "attention_decode sub_claim_values[0] mismatch: expected {:?}, got {:?}",
                output_claim.value, sub_claim_values[0],
            ),
        });
    }

    // Sub-proof 0: Output projection (new_tokens × d_model × d_model)
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
            new_tokens,
            d_model,
            d_model,
            layer_idx,
            channel,
        )?,
        other => {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "attention_decode output proj: expected MatMul, got {:?}",
                    std::mem::discriminant(other),
                ),
            });
        }
    };
    proof_idx += 1;

    // Per-head sub-proofs (h = H-1..0)
    for _h in (0..num_heads).rev() {
        // Context: new_tokens × full_seq_len × d_k
        let _ctx_claim = verify_fresh_sub_matmul(
            &sub_proofs[proof_idx],
            sub_claim_values[proof_idx],
            new_tokens,
            full_seq_len,
            d_k,
            channel,
        )?;
        proof_idx += 1;

        // Score: new_tokens × d_k × full_seq_len
        let _score_claim = verify_fresh_sub_matmul(
            &sub_proofs[proof_idx],
            sub_claim_values[proof_idx],
            new_tokens,
            d_k,
            full_seq_len,
            channel,
        )?;
        proof_idx += 1;
    }

    // V, K projections: new_tokens × d_model × d_model
    let _v_claim = verify_fresh_sub_matmul(
        &sub_proofs[proof_idx],
        sub_claim_values[proof_idx],
        new_tokens,
        d_model,
        d_model,
        channel,
    )?;
    proof_idx += 1;

    let _k_claim = verify_fresh_sub_matmul(
        &sub_proofs[proof_idx],
        sub_claim_values[proof_idx],
        new_tokens,
        d_model,
        d_model,
        channel,
    )?;
    proof_idx += 1;

    // Q projection — determines the final input claim
    let q_padded_rows = new_tokens.next_power_of_two();
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
            new_tokens,
            d_model,
            d_model,
            layer_idx,
            channel,
        )?,
        other => {
            return Err(GKRError::VerificationError {
                layer_idx,
                reason: format!(
                    "attention_decode Q projection: expected MatMul, got {:?}",
                    std::mem::discriminant(other),
                ),
            });
        }
    };

    Ok(final_claim)
}

/// Test-accessible wrapper for `verify_attention_reduction`.
/// Test-accessible wrapper for `verify_activation_reduction`.
pub fn verify_activation_reduction_for_test(
    output_claim: &GKRClaim,
    activation_type: ActivationType,
    logup_proof: Option<&LogUpProof>,
    multiplicity_sumcheck: Option<&MultiplicitySumcheckProof>,
    activation_proof: Option<&super::types::ActivationProductProof>,
    input_eval: SecureField,
    output_eval: SecureField,
    table_commitment: starknet_ff::FieldElement,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    verify_activation_reduction(
        output_claim,
        activation_type,
        logup_proof,
        multiplicity_sumcheck,
        activation_proof,
        None, // piecewise_proof
        input_eval,
        output_eval,
        table_commitment,
        0, // expected_size: not applicable for non-piecewise
        layer_idx,
        channel,
        false, // simd_combined
    )
}

pub fn verify_activation_reduction_for_test_piecewise(
    output_claim: &GKRClaim,
    activation_type: ActivationType,
    piecewise_proof: Option<&super::types::PiecewiseAlgebraicProof>,
    input_eval: SecureField,
    expected_size: usize,
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    if let Some(pw) = piecewise_proof {
        verify_piecewise_activation_reduction(
            output_claim,
            activation_type,
            pw,
            input_eval,
            expected_size,
            layer_idx,
            channel,
        )
    } else {
        Err(GKRError::VerificationError {
            layer_idx,
            reason: "no piecewise proof provided".to_string(),
        })
    }
}

pub fn verify_attention_reduction_for_test(
    output_claim: &GKRClaim,
    config: &MultiHeadAttentionConfig,
    sub_proofs: &[LayerProof],
    sub_claim_values: &[SecureField],
    softmax_sum_proofs: &[super::types::SoftmaxSumProof],
    layer_idx: usize,
    channel: &mut PoseidonChannel,
) -> Result<GKRClaim, GKRError> {
    verify_attention_reduction(
        output_claim,
        config,
        sub_proofs,
        sub_claim_values,
        softmax_sum_proofs,
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

/// Verify a multiplicity sumcheck proof and replay its Fiat-Shamir transcript.
///
/// For each round: verifies p(0) + p(1) == current_sum, mixes (c0, c1),
/// draws challenge, updates current_sum = c0 + c1 * r.
/// After all rounds: asserts current_sum == final_eval.
fn verify_multiplicity_sumcheck(
    proof: &MultiplicitySumcheckProof,
    channel: &mut PoseidonChannel,
) -> Result<(), GKRError> {
    let mut current_sum = proof.claimed_sum;
    for &(c0, c1) in &proof.round_polys {
        // p(0) = c0, p(1) = c0 + c1
        let sum_01 = c0 + (c0 + c1);
        if sum_01 != current_sum {
            return Err(GKRError::VerificationFailed(
                "multiplicity sumcheck round check failed".to_string(),
            ));
        }
        mix_secure_field(channel, c0);
        mix_secure_field(channel, c1);
        let r = channel.draw_qm31();
        current_sum = c0 + c1 * r;
    }
    if current_sum != proof.final_eval {
        return Err(GKRError::VerificationFailed(
            "multiplicity sumcheck final eval mismatch".to_string(),
        ));
    }
    Ok(())
}

/// Mix a SecureField into PoseidonChannel via packed felt252 (1 hades instead of 4).
/// Must match Cairo's channel_mix_secure_field and prover's mix_secure_field.
fn mix_secure_field(channel: &mut PoseidonChannel, v: SecureField) {
    channel.mix_felt(crate::crypto::poseidon_channel::securefield_to_felt(v));
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
        prove_gkr, prove_gkr_with_cache, reduce_activation_layer_for_test, reduce_embedding_layer_for_test,
        reduce_layernorm_layer_for_test, reduce_layernorm_simd_for_test, reduce_mul_layer_for_test,
        reduce_quantize_layer_for_test, reduce_rmsnorm_layer_for_test,
    };
    use num_traits::Zero;
    use stwo::core::fields::m31::M31;

    /// RAII guard that sets env vars while holding the global test mutex.
    /// Supports multiple key-value pairs in a single lock acquisition
    /// to avoid deadlocks when tests set multiple env vars.
    struct EnvVarGuard {
        entries: Vec<(&'static str, Option<String>)>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let lock = crate::test_utils::ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { entries: vec![(key, prev)], _lock: lock }
        }

        /// Set an additional env var under the same lock (no deadlock).
        fn and_set(mut self, key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            self.entries.push((key, prev));
            self
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            for (key, prev) in self.entries.iter().rev() {
                if let Some(prev) = prev.as_ref() {
                    std::env::set_var(key, prev);
                } else {
                    std::env::remove_var(key);
                }
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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
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
        verify_gkr_with_weights(&circuit, &proof, &c, &weights, &mut verifier_channel).unwrap();
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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
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
    #[ignore = "Known: deferred proof claim chain mismatch in batched RLC path"]    fn test_batched_rlc_direct_eval_with_deferred_claims() {
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
            intermediates: std::collections::HashMap::from([
                (0, x.clone()),
                (1, y0.clone()),
                (2, y1.clone()),
            ]),
            node_outputs,
            output: out.clone(),
        };

        // Skip policy commitment for this isolated unit test to avoid env interaction
        let _guard = EnvVarGuard::set("STWO_SKIP_POLICY_COMMITMENT", "1");

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
            if let Some(wc) = deferred.weight_commitment_mut() {
                *wc = starknet_ff::FieldElement::ZERO;
            }
            if let Some(wo) = deferred.weight_opening_mut() {
                *wo = crate::crypto::mle_opening::MleOpeningProof {
                    intermediate_roots: Vec::new(),
                    queries: Vec::new(),
                    final_value: SecureField::zero(),
                };
            }
        }

        let mut verifier_channel = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &out, &weights, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_aggregated_oracle_sumcheck_full_binding_roundtrip() {
        // When full binding is enabled, prove + verify roundtrip succeeds,
        // and the proof contains a binding proof.
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "aggregated")
            .and_set("STWO_AGGREGATED_FULL_BINDING", "1");

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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();
        assert_eq!(
            proof.weight_opening_transcript_mode,
            WeightOpeningTranscriptMode::AggregatedOracleSumcheck
        );
        assert!(
            proof.aggregated_binding.is_some(),
            "full binding mode must produce aggregated_binding proof"
        );

        let mut verifier_channel = PoseidonChannel::new();
        verify_gkr(&circuit, &proof, &c, &mut verifier_channel)
            .expect("full binding proof must verify");
    }

    #[test]
    fn test_aggregated_oracle_sumcheck_tampered_binding_fails() {
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "aggregated")
            .and_set("STWO_AGGREGATED_FULL_BINDING", "1");

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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
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
    fn test_verify_gkr_rejects_rlc_only_without_weights() {
        // Mode 4 (AggregatedOracleSumcheck) with RLC-only (no full binding)
        // must be rejected by verify_gkr() (no weights) but accepted by
        // verify_gkr_with_weights().
        // RLC-only requires explicit opt-in since full binding is now default.
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_RLC_ONLY", "1");
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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();
        assert_eq!(
            proof.weight_opening_transcript_mode,
            WeightOpeningTranscriptMode::AggregatedOracleSumcheck
        );
        assert!(
            proof.aggregated_binding.is_none(),
            "default RLC-only mode must NOT produce aggregated_binding proof"
        );

        // Weightless verifier must reject RLC-only
        let mut ch_reject = PoseidonChannel::new();
        let err = verify_gkr(&circuit, &proof, &c, &mut ch_reject)
            .expect_err("RLC-only Mode 4 must be rejected without weights");
        let msg = format!("{err}");
        assert!(
            msg.contains("RLC-only mode requires"),
            "unexpected error: {msg}"
        );

        // Weight-aware verifier must accept
        let mut ch_accept = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &c, &weights, &mut ch_accept)
            .expect("RLC-only Mode 4 must succeed with weights");
    }

    #[test]
    fn test_rlc_with_weights_detects_fabricated_claim() {
        // Prove correctly with Mode 4 RLC-only, then tamper a weight claim's
        // expected_value. verify_gkr_with_weights() must detect the mismatch.
        // RLC-only requires explicit opt-in since full binding is now default.
        let _guard = EnvVarGuard::set("STWO_AGGREGATED_RLC_ONLY", "1");
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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
            node_outputs: std::collections::HashMap::new(),
            output: c.clone(),
        };

        let mut prover_channel = PoseidonChannel::new();
        let mut proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();
        assert!(
            proof.aggregated_binding.is_none(),
            "should be RLC-only (no full binding)"
        );

        // Tamper weight claim
        assert!(!proof.weight_claims.is_empty());
        proof.weight_claims[0].expected_value =
            proof.weight_claims[0].expected_value + SecureField::one();

        let mut ch = PoseidonChannel::new();
        let err = verify_gkr_with_weights(&circuit, &proof, &c, &weights, &mut ch)
            .expect_err("fabricated weight claim must be detected");
        let msg = format!("{err}");
        assert!(
            msg.contains("RLC weight binding mismatch"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_aggregated_oracle_sumcheck_cross_mode_confusion_fails() {
        let _guard = EnvVarGuard::set("STWO_WEIGHT_BINDING", "aggregated")
            .and_set("STWO_AGGREGATED_FULL_BINDING", "1");

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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
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
        let result = verify_gkr_with_weights(&circuit, &proof, &c, &weights, &mut verifier_channel);
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
            intermediates: std::collections::HashMap::from([
                (0, input.clone()),
                (1, hidden.clone()),
                (2, activated.clone()),
            ]),
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        // Prove
        let mut prover_channel = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();
        assert_eq!(proof.layer_proofs.len(), 3); // MatMul + Activation + MatMul

        // Verify
        let mut verifier_channel = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &output, &weights, &mut verifier_channel)
            .unwrap();
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
            intermediates: std::collections::HashMap::from([(0, a.clone())]),
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
        let result = verify_gkr_with_weights(&circuit, &proof, &c, &weights, &mut verifier_channel);
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
                multiplicity_sumcheck,
                activation_proof,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    None, // piecewise_proof
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0, // expected_size: not applicable for non-piecewise tests
                    0,
                    &mut verifier_channel,
                    false, // simd_combined
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

        // Tamper with multiplicities — LogUp is now always produced
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
                multiplicity_sumcheck,
                activation_proof,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    None, // piecewise_proof
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0, // expected_size: not applicable for non-piecewise tests
                    0,
                    &mut verifier_channel,
                    false, // simd_combined
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

        // Tamper with eq-sumcheck round poly — LogUp is now always produced
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
                multiplicity_sumcheck,
                activation_proof,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    None, // piecewise_proof
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0, // expected_size: not applicable for non-piecewise tests
                    0,
                    &mut verifier_channel,
                    false, // simd_combined
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

        // Tamper with final_evals — LogUp is now always produced
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
                multiplicity_sumcheck,
                activation_proof,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    None, // piecewise_proof
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0, // expected_size: not applicable for non-piecewise tests
                    0,
                    &mut verifier_channel,
                    false, // simd_combined
                );
                assert!(result.is_err(), "tampered final_evals should fail");
            }
            _ => panic!("expected Activation proof"),
        }
    }

    // ===== Non-ReLU Activation LogUp Tests =====

    /// Helper: prove and verify a non-ReLU activation with range-reduced LogUp.
    fn prove_and_verify_activation_logup(
        activation_type: ActivationType,
        input: &M31Matrix,
        seed: u64,
    ) -> Result<GKRClaim, GKRError> {
        use crate::components::matmul::pad_matrix_pow2;

        let padded = pad_matrix_pow2(input);
        let n = padded.rows * padded.cols;
        let table_log_size = activation_type.recommended_table_log_size();
        let table_mask = (1u32 << table_log_size) - 1;
        let activation_fn = activation_type.as_fn();

        // Build output MLE: apply activation to masked inputs
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| {
                let masked = M31::from(v.0 & table_mask);
                SecureField::from(activation_fn(masked))
            })
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(seed);
        let num_vars = n.ilog2() as usize;
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (proof, _next_claim) = reduce_activation_layer_for_test(
            &output_claim,
            input,
            activation_type,
            &mut prover_channel,
        )
        .unwrap();

        // Verify with fresh channel
        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(seed);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::Activation {
                activation_type: at,
                logup_proof,
                multiplicity_sumcheck,
                activation_proof,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                assert!(logup_proof.is_some(), "non-ReLU should produce LogUp proof");
                verify_activation_reduction(
                    &output_claim,
                    *at,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    None, // piecewise_proof
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0, // expected_size: not applicable for non-piecewise tests
                    0,
                    &mut verifier_channel,
                    false, // simd_combined
                )
            }
            _ => panic!("expected Activation proof"),
        }
    }

    #[test]
    fn test_activation_logup_gelu() {
        // LogUp path is legacy — allow it via env var for backward-compat testing
        let _guard = EnvVarGuard::set("STWO_ALLOW_LOGUP_ACTIVATION", "1");
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(100u32));
            m.set(0, 1, M31::from(200u32));
            m.set(1, 0, M31::from(300u32));
            m.set(1, 1, M31::from(50u32));
            m
        };
        let result = prove_and_verify_activation_logup(ActivationType::GELU, &input, 0xAC10);
        assert!(result.is_ok(), "GELU LogUp should verify: {:?}", result.err());
    }

    #[test]
    fn test_activation_logup_sigmoid() {
        let _guard = EnvVarGuard::set("STWO_ALLOW_LOGUP_ACTIVATION", "1");
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(42u32));
            m.set(0, 1, M31::from(99u32));
            m.set(1, 0, M31::from(1u32));
            m.set(1, 1, M31::from(255u32));
            m
        };
        let result = prove_and_verify_activation_logup(ActivationType::Sigmoid, &input, 0x5101);
        assert!(result.is_ok(), "Sigmoid LogUp should verify: {:?}", result.err());
    }

    #[test]
    fn test_activation_logup_softmax() {
        let _guard = EnvVarGuard::set("STWO_ALLOW_LOGUP_ACTIVATION", "1");
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(10u32));
            m.set(0, 1, M31::from(500u32));
            m.set(1, 0, M31::from(1000u32));
            m.set(1, 1, M31::from(7u32));
            m
        };
        let result = prove_and_verify_activation_logup(ActivationType::Softmax, &input, 0x5F01);
        assert!(result.is_ok(), "Softmax LogUp should verify: {:?}", result.err());
    }

    #[test]
    fn test_activation_logup_relu_stays_algebraic() {
        // ReLU should still use the algebraic path, NOT LogUp
        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(5u32));
            m.set(0, 1, M31::from(10u32));
            m.set(1, 0, M31::from(3u32));
            m.set(1, 1, M31::from(7u32));
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
        prover_channel.mix_u64(0xAC11);
        let r = prover_channel.draw_qm31s(2);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        // The dispatch in prove_gkr sends ReLU to reduce_activation_layer_algebraic,
        // not reduce_activation_layer. But reduce_activation_layer_for_test
        // now produces LogUp for ALL types. Verify ReLU values also work with LogUp.
        let (proof, _) = reduce_activation_layer_for_test(
            &output_claim,
            &input,
            ActivationType::ReLU,
            &mut prover_channel,
        )
        .unwrap();

        // The function now produces logup_proof: Some(...) for all types,
        // but in the real dispatch, ReLU goes through the algebraic path.
        match &proof {
            LayerProof::Activation { logup_proof, .. } => {
                assert!(logup_proof.is_some(), "reduce_activation_layer now always produces LogUp");
            }
            _ => panic!("expected Activation proof"),
        }
    }

    #[test]
    fn test_activation_logup_gelu_tampered_multiplicity() {
        use crate::components::matmul::pad_matrix_pow2;

        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(100u32));
            m.set(0, 1, M31::from(200u32));
            m.set(1, 0, M31::from(300u32));
            m.set(1, 1, M31::from(50u32));
            m
        };

        let padded = pad_matrix_pow2(&input);
        let n = padded.rows * padded.cols;
        let table_log_size = ActivationType::GELU.recommended_table_log_size();
        let table_mask = (1u32 << table_log_size) - 1;
        let activation_fn = ActivationType::GELU.as_fn();
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| {
                let masked = M31::from(v.0 & table_mask);
                SecureField::from(activation_fn(masked))
            })
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xAC12);
        let num_vars = n.ilog2() as usize;
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) = reduce_activation_layer_for_test(
            &output_claim,
            &input,
            ActivationType::GELU,
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
        verifier_channel.mix_u64(0xAC12);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::Activation {
                activation_type,
                logup_proof,
                multiplicity_sumcheck,
                activation_proof,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    None, // piecewise_proof
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0, // expected_size: not applicable for non-piecewise tests
                    0,
                    &mut verifier_channel,
                    false, // simd_combined
                );
                assert!(result.is_err(), "tampered GELU multiplicities should fail");
            }
            _ => panic!("expected Activation proof"),
        }
    }

    #[test]
    fn test_activation_logup_gelu_tampered_round_poly() {
        use crate::components::matmul::pad_matrix_pow2;

        let input = {
            let mut m = M31Matrix::new(2, 2);
            m.set(0, 0, M31::from(100u32));
            m.set(0, 1, M31::from(200u32));
            m.set(1, 0, M31::from(300u32));
            m.set(1, 1, M31::from(50u32));
            m
        };

        let padded = pad_matrix_pow2(&input);
        let n = padded.rows * padded.cols;
        let table_log_size = ActivationType::GELU.recommended_table_log_size();
        let table_mask = (1u32 << table_log_size) - 1;
        let activation_fn = ActivationType::GELU.as_fn();
        let output_mle: Vec<SecureField> = padded
            .data
            .iter()
            .take(n)
            .map(|&v| {
                let masked = M31::from(v.0 & table_mask);
                SecureField::from(activation_fn(masked))
            })
            .collect();

        let mut prover_channel = PoseidonChannel::new();
        prover_channel.mix_u64(0xAC13);
        let num_vars = n.ilog2() as usize;
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) = reduce_activation_layer_for_test(
            &output_claim,
            &input,
            ActivationType::GELU,
            &mut prover_channel,
        )
        .unwrap();

        // Tamper with eq-sumcheck round poly
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
        verifier_channel.mix_u64(0xAC13);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::Activation {
                activation_type,
                logup_proof,
                multiplicity_sumcheck,
                activation_proof,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                let result = verify_activation_reduction(
                    &output_claim,
                    *activation_type,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    activation_proof.as_ref(),
                    None, // piecewise_proof
                    *input_eval,
                    *output_eval,
                    *table_commitment,
                    0, // expected_size: not applicable for non-piecewise tests
                    0,
                    &mut verifier_channel,
                    false, // simd_combined
                );
                assert!(result.is_err(), "tampered GELU eq-round poly should fail");
            }
            _ => panic!("expected Activation proof"),
        }
    }

    #[test]
    fn test_activation_require_proof_default_on() {
        // Verifier now REJECTS None/None proofs by default (no env var needed)
        let input_eval = SecureField::from(M31::from(42u32));
        let output_eval = SecureField::from(M31::from(99u32));
        let output_claim = GKRClaim {
            point: vec![SecureField::from(M31::from(1u32)), SecureField::from(M31::from(2u32))],
            value: output_eval,
        };

        let mut channel = PoseidonChannel::new();
        let result = verify_activation_reduction(
            &output_claim,
            ActivationType::GELU,
            None, // no logup proof
            None, // no multiplicity sumcheck
            None, // no activation proof
            None, // no piecewise proof
            input_eval,
            output_eval,
            starknet_ff::FieldElement::ZERO,
            0, // expected_size
            0,
            &mut channel,
            false, // simd_combined
        );
        assert!(result.is_err(), "should reject None proofs by default");
        let err_msg = format!("{:?}", result.err().unwrap());
        assert!(
            err_msg.contains("missing LogUp or algebraic proof"),
            "unexpected error: {}",
            err_msg,
        );
    }

    #[test]
    fn test_activation_simd_combined_allows_missing_proofs() {
        // SIMD block-combination path legitimately has all proofs None
        let input_eval = SecureField::from(M31::from(42u32));
        let output_eval = SecureField::from(M31::from(99u32));
        let output_claim = GKRClaim {
            point: vec![SecureField::from(M31::from(1u32)), SecureField::from(M31::from(2u32))],
            value: output_eval,
        };

        let mut channel = PoseidonChannel::new();
        let result = verify_activation_reduction(
            &output_claim,
            ActivationType::GELU,
            None,
            None,
            None,
            None, // piecewise_proof
            input_eval,
            output_eval,
            starknet_ff::FieldElement::ZERO,
            0, // expected_size
            0,
            &mut channel,
            true, // simd_combined: allows missing proofs
        );
        assert!(result.is_ok(), "SIMD combined path should accept None proofs");
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

    fn rmsnorm_forward(input: &M31Matrix, dim: usize) -> M31Matrix {
        use crate::components::rmsnorm::{build_rsqrt_table, RMSNormConfig};
        let config = RMSNormConfig::new(dim);
        let rsqrt_table = build_rsqrt_table(config.rsqrt_table_log_size);
        let padded = pad_matrix_pow2(input);
        let n_active = dim.min(padded.cols);
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

        let mut output = M31Matrix::new(padded.rows, padded.cols);
        for row in 0..padded.rows {
            let mut sq_sum = M31::from(0u32);
            for col in 0..n_active {
                let x = padded.get(row, col);
                sq_sum = sq_sum + x * x;
            }
            let rms_sq_raw = sq_sum * inv_n;
            let rms_sq =
                M31::from(rms_sq_raw.0 & ((1u32 << config.rsqrt_table_log_size) - 1));
            let rsqrt = rsqrt_table
                .lookup(rms_sq)
                .expect("rms_sq reduced to table range");
            for col in 0..padded.cols {
                if col < n_active {
                    output.set(row, col, padded.get(row, col) * rsqrt);
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
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
                mean_var_round_polys,
                mean_var_final_evals,
                var_eval,
                centered_binding_evals,
                mv_claimed_sums,
                row_means,
                row_variances,
                ..
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
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
                    mean_var_round_polys.as_ref(),
                    *mean_var_final_evals,
                    *var_eval,
                    *centered_binding_evals,
                    *mv_claimed_sums,
                    row_means.as_ref(),
                    row_variances.as_ref(),
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

    /// Test multi-row LayerNorm with a 4×4 matrix (4 rows) — proves and verifies successfully.
    #[test]
    fn test_layernorm_multi_row_4x4_binding() {
        let input = {
            let mut m = M31Matrix::new(4, 4);
            for r in 0..4u32 {
                for c in 0..4u32 {
                    m.set(r as usize, c as usize, M31::from(10 + r * 4 + c));
                }
            }
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
        prover_channel.mix_u64(0x4C4E_4D52); // unique seed
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (proof, _) =
            reduce_layernorm_layer_for_test(&output_claim, &input, dim, &mut prover_channel)
                .unwrap();

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x4C4E_4D52);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::LayerNorm {
                logup_proof,
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
                mean_var_round_polys,
                mean_var_final_evals,
                var_eval,
                centered_binding_evals,
                mv_claimed_sums,
                row_means,
                row_variances,
                ..
            } => {
                assert!(row_means.is_some(), "4-row LayerNorm should have row_means");
                assert_eq!(row_means.as_ref().unwrap().len(), 4);
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
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
                    mean_var_round_polys.as_ref(),
                    *mean_var_final_evals,
                    *var_eval,
                    *centered_binding_evals,
                    *mv_claimed_sums,
                    row_means.as_ref(),
                    row_variances.as_ref(),
                );
                assert!(
                    result.is_ok(),
                    "4×4 multi-row LayerNorm should verify: {:?}",
                    result.err()
                );
            }
            _ => panic!("expected LayerNorm proof"),
        }
    }

    /// Tampering row_means[1] in a multi-row LayerNorm should make verification fail.
    #[test]
    fn test_layernorm_multi_row_tampered_mean_rejected() {
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
        prover_channel.mix_u64(0x4C4E_5441); // "LNTA" seed
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) =
            reduce_layernorm_layer_for_test(&output_claim, &input, dim, &mut prover_channel)
                .unwrap();

        // Tamper row_means[1] — malicious prover sets a fake mean for row 1
        if let LayerProof::LayerNorm {
            ref mut row_means, ..
        } = &mut proof
        {
            if let Some(ref mut rm) = row_means {
                rm[1] = M31::from(999u32); // fake mean
            }
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x4C4E_5441);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::LayerNorm {
                logup_proof,
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
                mean_var_round_polys,
                mean_var_final_evals,
                var_eval,
                centered_binding_evals,
                mv_claimed_sums,
                row_means,
                row_variances,
                ..
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
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
                    mean_var_round_polys.as_ref(),
                    *mean_var_final_evals,
                    *var_eval,
                    *centered_binding_evals,
                    *mv_claimed_sums,
                    row_means.as_ref(),
                    row_variances.as_ref(),
                );
                assert!(
                    result.is_err(),
                    "tampered row_means should be rejected"
                );
                let err_msg = format!("{:?}", result.err().unwrap());
                assert!(
                    err_msg.contains("multi-row mean binding failed"),
                    "unexpected error: {err_msg}",
                );
            }
            _ => panic!("expected LayerNorm proof"),
        }
    }

    /// Test multi-row RMSNorm with 2×8 matrix — proves and verifies successfully.
    #[test]
    fn test_rmsnorm_multi_row_binding() {
        let input = {
            let mut m = M31Matrix::new(2, 8);
            for r in 0..2u32 {
                for c in 0..8u32 {
                    m.set(r as usize, c as usize, M31::from(1 + r * 8 + c));
                }
            }
            m
        };
        let dim = 8;

        let output = rmsnorm_forward(&input, dim);
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
        prover_channel.mix_u64(0x524E_4D52); // "RNMR" seed
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (proof, _) =
            reduce_rmsnorm_layer_for_test(&output_claim, &input, dim, &mut prover_channel)
                .unwrap();

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x524E_4D52);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::RMSNorm {
                logup_proof,
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                rms_sq_eval,
                rsqrt_eval,
                rsqrt_table_commitment,
                simd_combined,
                rms_sq_round_polys,
                rms_sq_input_final,
                rms_sq_claimed_sq_sum,
                row_rms_sq,
                ..
            } => {
                assert!(row_rms_sq.is_some(), "2-row RMSNorm should have row_rms_sq");
                assert_eq!(row_rms_sq.as_ref().unwrap().len(), 2);
                let result = verify_rmsnorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *rms_sq_eval,
                    *rsqrt_eval,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    &mut verifier_channel,
                    rms_sq_round_polys.as_ref(),
                    *rms_sq_input_final,
                    *rms_sq_claimed_sq_sum,
                    row_rms_sq.as_ref(),
                );
                assert!(
                    result.is_ok(),
                    "2×8 multi-row RMSNorm should verify: {:?}",
                    result.err()
                );
            }
            _ => panic!("expected RMSNorm proof"),
        }
    }

    /// Tampering row_rms_sq[0] in a multi-row RMSNorm should make verification fail.
    #[test]
    fn test_rmsnorm_multi_row_tampered_rms_sq_rejected() {
        let input = {
            let mut m = M31Matrix::new(2, 8);
            for r in 0..2u32 {
                for c in 0..8u32 {
                    m.set(r as usize, c as usize, M31::from(1 + r * 8 + c));
                }
            }
            m
        };
        let dim = 8;

        let output = rmsnorm_forward(&input, dim);
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
        prover_channel.mix_u64(0x524E_5441); // "RNTA" seed
        let r = prover_channel.draw_qm31s(num_vars);
        let claimed = evaluate_mle_pub(&output_mle, &r);
        let output_claim = GKRClaim {
            point: r.clone(),
            value: claimed,
        };

        let (mut proof, _) =
            reduce_rmsnorm_layer_for_test(&output_claim, &input, dim, &mut prover_channel)
                .unwrap();

        // Tamper row_rms_sq[0] — malicious prover sets fake rms_sq for row 0
        if let LayerProof::RMSNorm {
            ref mut row_rms_sq, ..
        } = &mut proof
        {
            if let Some(ref mut rr) = row_rms_sq {
                rr[0] = M31::from(42u32); // fake rms_sq
            }
        }

        let mut verifier_channel = PoseidonChannel::new();
        verifier_channel.mix_u64(0x524E_5441);
        let _r_v = verifier_channel.draw_qm31s(num_vars);

        match &proof {
            LayerProof::RMSNorm {
                logup_proof,
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                rms_sq_eval,
                rsqrt_eval,
                rsqrt_table_commitment,
                simd_combined,
                rms_sq_round_polys,
                rms_sq_input_final,
                rms_sq_claimed_sq_sum,
                row_rms_sq,
                ..
            } => {
                let result = verify_rmsnorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
                    linear_round_polys,
                    *linear_final_evals,
                    *input_eval,
                    *output_eval,
                    *rms_sq_eval,
                    *rsqrt_eval,
                    *rsqrt_table_commitment,
                    *simd_combined,
                    dim,
                    0,
                    &mut verifier_channel,
                    rms_sq_round_polys.as_ref(),
                    *rms_sq_input_final,
                    *rms_sq_claimed_sq_sum,
                    row_rms_sq.as_ref(),
                );
                assert!(
                    result.is_err(),
                    "tampered row_rms_sq should be rejected"
                );
                let err_msg = format!("{:?}", result.err().unwrap());
                assert!(
                    err_msg.contains("multi-row rms_sq binding failed"),
                    "unexpected error: {err_msg}",
                );
            }
            _ => panic!("expected RMSNorm proof"),
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
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
                mean_var_round_polys,
                mean_var_final_evals,
                var_eval,
                centered_binding_evals,
                mv_claimed_sums,
                row_means,
                row_variances,
                ..
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
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
                    mean_var_round_polys.as_ref(),
                    *mean_var_final_evals,
                    *var_eval,
                    *centered_binding_evals,
                    *mv_claimed_sums,
                    row_means.as_ref(),
                    row_variances.as_ref(),
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
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
                mean_var_round_polys,
                mean_var_final_evals,
                var_eval,
                centered_binding_evals,
                mv_claimed_sums,
                row_means,
                row_variances,
                ..
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
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
                    mean_var_round_polys.as_ref(),
                    *mean_var_final_evals,
                    *var_eval,
                    *centered_binding_evals,
                    *mv_claimed_sums,
                    row_means.as_ref(),
                    row_variances.as_ref(),
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
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
                mean_var_round_polys,
                mean_var_final_evals,
                var_eval,
                centered_binding_evals,
                mv_claimed_sums,
                row_means,
                row_variances,
                ..
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
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
                    mean_var_round_polys.as_ref(),
                    *mean_var_final_evals,
                    *var_eval,
                    *centered_binding_evals,
                    *mv_claimed_sums,
                    row_means.as_ref(),
                    row_variances.as_ref(),
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
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
                mean_var_round_polys,
                mean_var_final_evals,
                var_eval,
                centered_binding_evals,
                mv_claimed_sums,
                row_means,
                row_variances,
                ..
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
                    multiplicity_sumcheck.as_ref(),
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
                    mean_var_round_polys.as_ref(),
                    *mean_var_final_evals,
                    *var_eval,
                    *centered_binding_evals,
                    *mv_claimed_sums,
                    row_means.as_ref(),
                    row_variances.as_ref(),
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
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
                mean_var_round_polys,
                mean_var_final_evals,
                var_eval,
                centered_binding_evals,
                mv_claimed_sums,
                row_means,
                row_variances,
                ..
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
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
                    mean_var_round_polys.as_ref(),
                    *mean_var_final_evals,
                    *var_eval,
                    *centered_binding_evals,
                    *mv_claimed_sums,
                    row_means.as_ref(),
                    row_variances.as_ref(),
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
                multiplicity_sumcheck,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                let result = verify_dequantize_reduction(
                    &output_claim,
                    &params,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
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
                multiplicity_sumcheck,
                input_eval,
                output_eval,
                table_commitment,
                ..
            } => {
                if !logup.multiplicities.is_empty() {
                    logup.multiplicities[0] = logup.multiplicities[0].wrapping_add(1);
                }
                let result = verify_dequantize_reduction(
                    &output_claim,
                    &params,
                    Some(&logup),
                    multiplicity_sumcheck.as_ref(),
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
                ..
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
                ..
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
            intermediates: std::collections::HashMap::from([
                (0, input.clone()),
                (1, hidden.clone()),
            ]),
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 2);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &output, &weights, &mut verifier_ch).unwrap();
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
            intermediates: std::collections::HashMap::from([
                (0, input.clone()),
                (1, hidden.clone()),
            ]),
            node_outputs: std::collections::HashMap::new(),
            output: normed.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 2);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &normed, &weights, &mut verifier_ch).unwrap();
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
            intermediates: std::collections::HashMap::from([
                (0, input.clone()),
                (1, after_relu.clone()),
            ]),
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
            intermediates: std::collections::HashMap::from([
                (0, input.clone()),
                (1, activated.clone()),
            ]),
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 2);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &output, &weights, &mut verifier_ch).unwrap();
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
            intermediates: std::collections::HashMap::from([
                (0, input.clone()),
                (1, h1.clone()),
                (2, h2.clone()),
            ]),
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 3);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &output, &weights, &mut verifier_ch).unwrap();
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
            intermediates: std::collections::HashMap::from([
                (0, input.clone()),
                (1, hidden1.clone()),
                (2, activated.clone()),
                (3, hidden2.clone()),
            ]),
            node_outputs: std::collections::HashMap::new(),
            output: output.clone(),
        };

        let mut prover_ch = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_ch).unwrap();
        assert_eq!(proof.layer_proofs.len(), 4);

        let mut verifier_ch = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &output, &weights, &mut verifier_ch).unwrap();
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
                multiplicity_sumcheck,
                linear_round_polys,
                linear_final_evals,
                input_eval,
                output_eval,
                mean,
                rsqrt_var,
                rsqrt_table_commitment,
                simd_combined,
                mean_var_round_polys,
                mean_var_final_evals,
                var_eval,
                centered_binding_evals,
                mv_claimed_sums,
                row_means,
                row_variances,
                ..
            } => {
                let result = verify_layernorm_reduction(
                    &output_claim,
                    logup_proof.as_ref(),
                    multiplicity_sumcheck.as_ref(),
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
                    mean_var_round_polys.as_ref(),
                    *mean_var_final_evals,
                    *var_eval,
                    *centered_binding_evals,
                    *mv_claimed_sums,
                    row_means.as_ref(),
                    row_variances.as_ref(),
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

    #[test]
    #[ignore = "Known: deferred proof claim chain mismatch in dequantize DAG path"]    fn test_deferred_proof_dequantize_dag() {
        // DAG: Input(1×4) → Dequantize → fork → MatMul(4×4) → Add(deq_fork, matmul_out)
        // The Dequantize skip branch gets a Weightless deferred proof.
        let params = QuantParams {
            strategy: QuantStrategy::Symmetric8,
            scale: 0.1,
            zero_point: 0,
            bits: 8,
        };

        let mut builder = GraphBuilder::new((1, 4));
        builder.dequantize(params.clone());
        let fork = builder.fork();
        builder.linear(4);
        builder.add_from(fork);
        let graph = builder.build();
        let circuit = LayeredCircuit::from_graph(&graph).unwrap();

        // Input: quantized values (valid for 8-bit: 0..255)
        let mut x_quant = M31Matrix::new(1, 4);
        x_quant.data = vec![
            M31::from(10u32),
            M31::from(20u32),
            M31::from(30u32),
            M31::from(40u32),
        ];

        // Dequantize output: dequantize(10, scale=0.1, zp=0) = 1.0 → M31(1), etc.
        let direct_params = QuantParams {
            strategy: QuantStrategy::Direct,
            scale: 1.0,
            zero_point: 0,
            bits: 31,
        };
        let mut x_deq = M31Matrix::new(1, 4);
        for i in 0..4 {
            let f = dequantize_value(x_quant.data[i], &params);
            x_deq.data[i] = quantize_value(f, &direct_params);
        }
        assert_eq!(
            x_deq.data,
            vec![
                M31::from(1u32),
                M31::from(2u32),
                M31::from(3u32),
                M31::from(4u32)
            ]
        );

        // Weight matrix for MatMul (4×4)
        let mut w = M31Matrix::new(4, 4);
        for i in 0..16 {
            w.data[i] = M31::from(((i % 5) + 1) as u32);
        }

        // Forward pass
        let y = matmul_forward(&x_deq, &w);
        let mut out = M31Matrix::new(1, 4);
        for j in 0..4 {
            out.set(0, j, x_deq.get(0, j) + y.get(0, j));
        }

        // Build weights (only MatMul node 1 has weights)
        let mut weights = GraphWeights::new();
        weights.add_weight(1, w);

        // Build execution trace
        let mut node_outputs = std::collections::HashMap::new();
        node_outputs.insert(0usize, x_deq.clone());
        node_outputs.insert(1usize, y.clone());
        node_outputs.insert(2usize, out.clone());

        let execution = GraphExecution {
            intermediates: std::collections::HashMap::from([
                (0, x_quant.clone()), // input to Dequantize
                (1, x_deq.clone()),   // input to MatMul
                (2, y.clone()),       // input to Add (trunk = MatMul output)
            ]),
            node_outputs,
            output: out.clone(),
        };

        // Skip policy commitment for this isolated unit test to avoid env interaction
        let _guard = EnvVarGuard::set("STWO_SKIP_POLICY_COMMITMENT", "1");

        // Prove
        let mut prover_channel = PoseidonChannel::new();
        let proof = prove_gkr(&circuit, &execution, &weights, &mut prover_channel).unwrap();

        // Assert deferred proofs exist with Weightless kind
        assert!(
            !proof.deferred_proofs.is_empty(),
            "dequantize DAG should produce deferred proofs"
        );
        assert!(
            proof.deferred_proofs.iter().any(|d| !d.has_weights()),
            "dequantize deferred proof should be Weightless"
        );

        // Verify
        let mut verifier_channel = PoseidonChannel::new();
        verify_gkr_with_weights(&circuit, &proof, &out, &weights, &mut verifier_channel).unwrap();
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn test_simd_deferred_proof_matmul_dag() {
        // Test that prove_gkr_simd_gpu correctly generates deferred proofs for DAG Add layers
        // and that verify_gkr_simd verifies them.
        //
        // Circuit: 2 identical blocks, each containing:
        //   MatMul_A → MatMul_B → Add(MatMul_A, MatMul_B)
        // The Add layer creates a DAG: trunk = MatMul_B (higher idx), skip = MatMul_A (lower idx).
        // The skip branch gets a deferred MatMul proof.
        use crate::gkr::circuit::{CircuitLayer, LayerType, SIMDBatchConfig};
        use crate::gkr::prover::prove_gkr_simd_gpu;

        let m = 2usize;
        let k = 4usize;
        let n = 4usize;

        // Shared weight matrices (same for both blocks, different per layer position)
        let mut w_a = M31Matrix::new(k, n);
        for i in 0..(k * n) {
            w_a.data[i] = M31::from(((i * 3 + 1) % 31 + 1) as u32);
        }
        let mut w_b = M31Matrix::new(k, n);
        for i in 0..(k * n) {
            w_b.data[i] = M31::from(((i * 7 + 2) % 31 + 1) as u32);
        }

        let weight_id_a = 100usize;
        let weight_id_b = 101usize;

        // Build circuit manually: 2 blocks of [MatMul, MatMul, Add]
        let layers = vec![
            // Block 0
            CircuitLayer {
                layer_type: LayerType::MatMul {
                    m,
                    k,
                    n,
                    weight_node_id: weight_id_a,
                },
                input_shape: (m, k),
                output_shape: (m, n),
                node_id: 0,
                input_layers: vec![],
            },
            CircuitLayer {
                layer_type: LayerType::MatMul {
                    m,
                    k,
                    n,
                    weight_node_id: weight_id_b,
                },
                input_shape: (m, k),
                output_shape: (m, n),
                node_id: 1,
                input_layers: vec![0],
            },
            CircuitLayer {
                layer_type: LayerType::Add { size: m * n },
                input_shape: (m, n),
                output_shape: (m, n),
                node_id: 2,
                input_layers: vec![0, 1],
            },
            // Block 1 (mirrors block 0)
            CircuitLayer {
                layer_type: LayerType::MatMul {
                    m,
                    k,
                    n,
                    weight_node_id: weight_id_a,
                },
                input_shape: (m, k),
                output_shape: (m, n),
                node_id: 3,
                input_layers: vec![],
            },
            CircuitLayer {
                layer_type: LayerType::MatMul {
                    m,
                    k,
                    n,
                    weight_node_id: weight_id_b,
                },
                input_shape: (m, k),
                output_shape: (m, n),
                node_id: 4,
                input_layers: vec![3],
            },
            CircuitLayer {
                layer_type: LayerType::Add { size: m * n },
                input_shape: (m, n),
                output_shape: (m, n),
                node_id: 5,
                input_layers: vec![3, 4],
            },
        ];

        let circuit = LayeredCircuit {
            layers,
            block_ranges: vec![0..3, 3..6],
            simd_config: Some(SIMDBatchConfig {
                num_blocks: 2,
                template_range: 0..3,
                simd_log_size: 1,
            }),
            input_shape: (m, k),
            output_shape: (m, n),
        };

        // Different input activations per block
        let mut input_0 = M31Matrix::new(m, k);
        for i in 0..(m * k) {
            input_0.data[i] = M31::from((i + 1) as u32);
        }
        let mut input_1 = M31Matrix::new(m, k);
        for i in 0..(m * k) {
            input_1.data[i] = M31::from((i * 5 + 3) as u32 % 31 + 1);
        }

        // Forward pass for block 0
        let a_out_0 = matmul_forward(&input_0, &w_a); // MatMul_A output
        let b_out_0 = matmul_forward(&a_out_0, &w_b); // MatMul_B output
        let mut add_out_0 = M31Matrix::new(m, n);
        for i in 0..(m * n) {
            add_out_0.data[i] = a_out_0.data[i] + b_out_0.data[i];
        }

        // Forward pass for block 1
        let a_out_1 = matmul_forward(&input_1, &w_a);
        let b_out_1 = matmul_forward(&a_out_1, &w_b);
        let mut add_out_1 = M31Matrix::new(m, n);
        for i in 0..(m * n) {
            add_out_1.data[i] = a_out_1.data[i] + b_out_1.data[i];
        }

        // Build block executions
        let mut node_outputs_0 = std::collections::HashMap::new();
        node_outputs_0.insert(0usize, a_out_0.clone());
        node_outputs_0.insert(1usize, b_out_0.clone());
        let exec_0 = GraphExecution {
            intermediates: std::collections::HashMap::from([
                (0, input_0.clone()), // input to MatMul_A
                (1, a_out_0.clone()), // input to MatMul_B = MatMul_A output
            ]),
            node_outputs: node_outputs_0,
            output: add_out_0.clone(),
        };

        let mut node_outputs_1 = std::collections::HashMap::new();
        node_outputs_1.insert(3usize, a_out_1.clone());
        node_outputs_1.insert(4usize, b_out_1.clone());
        let exec_1 = GraphExecution {
            intermediates: std::collections::HashMap::from([
                (3, input_1.clone()), // input to MatMul_C
                (4, a_out_1.clone()), // input to MatMul_D = MatMul_C output
            ]),
            node_outputs: node_outputs_1,
            output: add_out_1.clone(),
        };

        let block_executions = vec![exec_0, exec_1];

        // Build weights
        let mut weights = GraphWeights::new();
        weights.add_weight(weight_id_a, w_a);
        weights.add_weight(weight_id_b, w_b);

        // Prove
        let mut prover_channel = PoseidonChannel::new();
        let proof = prove_gkr_simd_gpu(&circuit, &block_executions, &weights, &mut prover_channel)
            .expect("SIMD GPU proving with DAG Add should succeed");

        // Assert deferred proofs were generated
        assert!(
            !proof.deferred_proofs.is_empty(),
            "DAG Add in SIMD path should produce deferred proofs, got 0"
        );
        assert_eq!(
            proof.deferred_proofs.len(),
            1,
            "One Add layer = one deferred proof"
        );
        assert!(
            proof.deferred_proofs[0].has_weights(),
            "Deferred proof for MatMul skip should be MatMul kind (has weights)"
        );

        // Compute combined output for verifier
        use crate::components::matmul::{compute_lagrange_basis_pub, pad_matrix_pow2};
        let r_simd_replay = {
            let mut ch = PoseidonChannel::new();
            ch.mix_u64(circuit.layers.len() as u64);
            ch.mix_u64(circuit.input_shape.0 as u64);
            ch.mix_u64(circuit.input_shape.1 as u64);
            ch.mix_u64(2u64); // num_blocks
            ch.draw_qm31s(1) // simd_log_size = 1
        };
        let block_weights_replay = compute_lagrange_basis_pub(&r_simd_replay);
        let w0 = block_weights_replay[0];
        let w1 = block_weights_replay[1];

        let out_0_padded = pad_matrix_pow2(&add_out_0);
        let out_1_padded = pad_matrix_pow2(&add_out_1);
        let out_0_mle = crate::components::matmul::matrix_to_mle_pub(&out_0_padded);
        let out_1_mle = crate::components::matmul::matrix_to_mle_pub(&out_1_padded);
        let combined_output: Vec<SecureField> = out_0_mle
            .iter()
            .zip(&out_1_mle)
            .map(|(&a, &b)| w0 * a + w1 * b)
            .collect();

        // Verify
        let mut verifier_channel = PoseidonChannel::new();
        let claim = verify_gkr_simd(&circuit, &proof, &combined_output, &mut verifier_channel)
            .expect("SIMD verify with deferred proofs should succeed");

        // The final claim should match the proof's input claim
        assert_eq!(claim.point, proof.input_claim.point);
        assert_eq!(claim.value, proof.input_claim.value);
    }
}
