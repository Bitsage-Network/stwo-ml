// Full GKR Model Walk Verifier
//
// Walks layers output → input, dispatching by tag to per-layer verifiers.
// This is the core of the 100% on-chain ZKML verification pipeline:
// no STARK, no FRI, no dicts — only field ops + Poseidon + sumcheck.
//
// Entry point: `verify_gkr_model()` takes flat felt252 proof data
// (serialized by stwo-ml/src/cairo_serde.rs:serialize_gkr_model_proof)
// and walks each layer in proof order, verifying via per-layer verifiers.
//
// Layer tags (matching Rust gkr/types.rs):
//   0=MatMul, 1=Add, 2=Mul, 3=Activation, 4=LayerNorm,
//   5=Attention, 6=Dequantize, 7=MatMulDualSimd, 8=RMSNorm

use crate::field::{QM31, CM31, qm31_zero, log2_ceil, next_power_of_two};
use crate::channel::{PoseidonChannel, channel_mix_secure_field};
use crate::types::{RoundPoly, GkrRoundPoly, GKRClaim};
use crate::layer_verifiers::{
    verify_add_layer, verify_mul_layer, verify_matmul_layer,
    verify_activation_layer, verify_dequantize_layer,
    verify_layernorm_layer, verify_rmsnorm_layer,
    clone_point,
};

/// Weight claim collected during the GKR walk.
/// Each MatMul layer produces one: the evaluation point and expected value
/// for the weight MLE opening proof.
#[derive(Drop)]
pub struct WeightClaimData {
    /// Evaluation point: [r_j || sumcheck_challenges]
    pub eval_point: Array<QM31>,
    /// Expected value: final_b_eval from the matmul sumcheck
    pub expected_value: QM31,
}

// ============================================================================
// Proof Data Reader
// ============================================================================

/// Offset-based reader for flat felt252 proof data.
/// Advances through the data one field at a time.
#[derive(Drop, Copy)]
struct ProofReader {
    data: Span<felt252>,
    offset: u32,
}

fn reader_new(data: Span<felt252>) -> ProofReader {
    ProofReader { data, offset: 0 }
}

fn read_felt(ref r: ProofReader) -> felt252 {
    assert!(r.offset < r.data.len(), "PROOF_DATA_TRUNCATED");
    let v = *r.data.at(r.offset);
    r.offset += 1;
    v
}

fn read_u32(ref r: ProofReader) -> u32 {
    let f = read_felt(ref r);
    let v: u256 = f.into();
    v.try_into().unwrap()
}

fn read_u64(ref r: ProofReader) -> u64 {
    let f = read_felt(ref r);
    let v: u256 = f.into();
    v.try_into().unwrap()
}

fn read_qm31(ref r: ProofReader) -> QM31 {
    let aa = read_u64(ref r);
    let ab = read_u64(ref r);
    let ba = read_u64(ref r);
    let bb = read_u64(ref r);
    QM31 { a: CM31 { a: aa, b: ab }, b: CM31 { a: ba, b: bb } }
}

fn read_deg2_poly(ref r: ProofReader) -> RoundPoly {
    let c0 = read_qm31(ref r);
    let c1 = read_qm31(ref r);
    let c2 = read_qm31(ref r);
    RoundPoly { c0, c1, c2 }
}

fn read_deg3_poly(ref r: ProofReader) -> GkrRoundPoly {
    let c0 = read_qm31(ref r);
    let c1 = read_qm31(ref r);
    let c2 = read_qm31(ref r);
    let c3 = read_qm31(ref r);
    GkrRoundPoly { c0, c1, c2, c3, num_coeffs: 4 }
}

fn read_deg2_polys(ref r: ProofReader, count: u32) -> Array<RoundPoly> {
    let mut result: Array<RoundPoly> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= count {
            break;
        }
        result.append(read_deg2_poly(ref r));
        i += 1;
    };
    result
}

fn read_deg3_polys(ref r: ProofReader, count: u32) -> Array<GkrRoundPoly> {
    let mut result: Array<GkrRoundPoly> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= count {
            break;
        }
        result.append(read_deg3_poly(ref r));
        i += 1;
    };
    result
}

/// Read an optional LogUp proof from flat data.
/// Returns (has_logup, round_polys, final_w, final_in, final_out, claimed_sum).
/// Multiplicities are skipped (table-side verification done externally).
fn read_optional_logup(
    ref r: ProofReader,
) -> (bool, Array<GkrRoundPoly>, QM31, QM31, QM31, QM31) {
    let has_logup = read_u32(ref r);
    if has_logup == 0 {
        return (false, array![], qm31_zero(), qm31_zero(), qm31_zero(), qm31_zero());
    }

    let claimed_sum = read_qm31(ref r);
    let num_rounds = read_u32(ref r);
    let polys = read_deg3_polys(ref r, num_rounds);
    let final_w = read_qm31(ref r);
    let final_in = read_qm31(ref r);
    let final_out = read_qm31(ref r);

    // Skip multiplicities (consumed but not used — table check is external)
    let num_mults = read_u32(ref r);
    let mut i: u32 = 0;
    loop {
        if i >= num_mults {
            break;
        }
        let _ = read_u32(ref r);
        i += 1;
    };

    (true, polys, final_w, final_in, final_out, claimed_sum)
}

// ============================================================================
// Per-Layer Dispatch
// ============================================================================

/// Parse and verify a Tag 0 (MatMul) layer proof.
/// Returns (new_claim, final_b_eval) — final_b_eval is needed for weight opening verification.
fn dispatch_matmul(
    current_claim: @GKRClaim,
    m: u32,
    k: u32,
    n: u32,
    ref reader: ProofReader,
    ref ch: PoseidonChannel,
) -> (GKRClaim, QM31) {
    let num_rounds = read_u32(ref reader);
    let round_polys = read_deg2_polys(ref reader, num_rounds);
    let final_a = read_qm31(ref reader);
    let final_b = read_qm31(ref reader);

    let claim = verify_matmul_layer(
        current_claim, round_polys.span(), final_a, final_b, m, k, n, ref ch,
    );
    (claim, final_b)
}

/// Parse and verify a Tag 1 (Add) layer proof.
fn dispatch_add(
    current_claim: @GKRClaim,
    ref reader: ProofReader,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    let lhs = read_qm31(ref reader);
    let rhs = read_qm31(ref reader);
    let trunk_idx = read_u32(ref reader);

    verify_add_layer(current_claim, lhs, rhs, trunk_idx, ref ch)
}

/// Parse and verify a Tag 2 (Mul) layer proof.
fn dispatch_mul(
    current_claim: @GKRClaim,
    ref reader: ProofReader,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    let num_rounds = read_u32(ref reader);
    let round_polys = read_deg3_polys(ref reader, num_rounds);
    let lhs = read_qm31(ref reader);
    let rhs = read_qm31(ref reader);

    verify_mul_layer(current_claim, round_polys.span(), lhs, rhs, ref ch)
}

/// Parse and verify a Tag 3 (Activation) layer proof.
fn dispatch_activation(
    current_claim: @GKRClaim,
    ref reader: ProofReader,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    let act_type_tag = read_u64(ref reader);
    let input_eval = read_qm31(ref reader);
    let output_eval = read_qm31(ref reader);
    let _table_commitment = read_felt(ref reader);

    let (has_logup, logup_polys, w, in_e, out_e, claimed) = read_optional_logup(ref reader);
    assert!(has_logup, "ACTIVATION_MISSING_LOGUP");

    verify_activation_layer(
        current_claim, act_type_tag,
        logup_polys.span(), w, in_e, out_e, claimed,
        input_eval, output_eval, ref ch,
    )
}

/// Parse and verify a Tag 4 (LayerNorm) layer proof.
fn dispatch_layernorm(
    current_claim: @GKRClaim,
    ref reader: ProofReader,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    let input_eval = read_qm31(ref reader);
    let output_eval = read_qm31(ref reader);
    let mean = read_qm31(ref reader);
    let rsqrt_var = read_qm31(ref reader);
    let _rsqrt_table_commitment = read_felt(ref reader);
    let _simd_combined = read_u32(ref reader);

    let num_linear_rounds = read_u32(ref reader);
    let linear_polys = read_deg3_polys(ref reader, num_linear_rounds);
    let centered_final = read_qm31(ref reader);
    let rsqrt_final = read_qm31(ref reader);

    let (has_logup, logup_polys, w, in_e, out_e, claimed) = read_optional_logup(ref reader);

    verify_layernorm_layer(
        current_claim,
        linear_polys.span(), centered_final, rsqrt_final,
        mean, rsqrt_var,
        has_logup, logup_polys.span(), w, in_e, out_e, claimed,
        input_eval, output_eval, ref ch,
    )
}

/// Parse and verify a Tag 6 (Dequantize) layer proof.
fn dispatch_dequantize(
    current_claim: @GKRClaim,
    bits: u64,
    ref reader: ProofReader,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    let input_eval = read_qm31(ref reader);
    let output_eval = read_qm31(ref reader);
    let _table_commitment = read_felt(ref reader);

    let (has_logup, logup_polys, w, in_e, out_e, claimed) = read_optional_logup(ref reader);
    assert!(has_logup, "DEQUANTIZE_MISSING_LOGUP");

    verify_dequantize_layer(
        current_claim, bits,
        logup_polys.span(), w, in_e, out_e, claimed,
        input_eval, output_eval, ref ch,
    )
}

/// Parse and verify a Tag 8 (RMSNorm) layer proof.
fn dispatch_rmsnorm(
    current_claim: @GKRClaim,
    ref reader: ProofReader,
    ref ch: PoseidonChannel,
) -> GKRClaim {
    let input_eval = read_qm31(ref reader);
    let output_eval = read_qm31(ref reader);
    let rms_sq = read_qm31(ref reader);
    let rsqrt_eval = read_qm31(ref reader);
    let _rsqrt_table_commitment = read_felt(ref reader);
    let _simd_combined = read_u32(ref reader);

    let num_linear_rounds = read_u32(ref reader);
    let linear_polys = read_deg3_polys(ref reader, num_linear_rounds);
    let input_final = read_qm31(ref reader);
    let rsqrt_final = read_qm31(ref reader);

    let (has_logup, logup_polys, w, in_e, out_e, claimed) = read_optional_logup(ref reader);

    verify_rmsnorm_layer(
        current_claim,
        linear_polys.span(), input_final, rsqrt_final,
        rms_sq, rsqrt_eval,
        has_logup, logup_polys.span(), w, in_e, out_e, claimed,
        input_eval, output_eval, ref ch,
    )
}

// ============================================================================
// Top-Level Model Verifier
// ============================================================================

/// Verify a complete GKR model proof by walking layers output → input.
///
/// The proof_data is a flat felt252 array containing tag-dispatched
/// per-layer proofs (serialized by cairo_serde.rs:serialize_gkr_model_proof).
///
/// The caller is responsible for:
///   1. Seeding the channel: mix_u64(d), mix_u64(input_rows), mix_u64(input_cols)
///   2. Reconstructing the initial output claim (draw r_out, evaluate output MLE)
///   3. Mixing the output value: mix_secure_field(output_value)
///   4. Passing the initial claim
///
/// Circuit dimensions not in the proof:
///   - matmul_dims: flat [m0,k0,n0, m1,k1,n1, ...] — one triple per MatMul layer
///   - dequantize_bits: flat [bits0, bits1, ...] — one per Dequantize layer
///
/// Returns (final_input_claim, weight_claims). The caller should:
///   1. Verify final_input_claim matches the committed input data
///   2. Verify each weight_claim via MLE opening proofs against registered roots
pub fn verify_gkr_model(
    proof_data: Span<felt252>,
    num_layers: u32,
    matmul_dims: Span<u32>,
    dequantize_bits: Span<u64>,
    initial_claim: GKRClaim,
    ref ch: PoseidonChannel,
) -> (GKRClaim, Array<WeightClaimData>) {
    let (final_claim, weight_claims, _layer_tags, _deferred_weight_commitments) =
        verify_gkr_model_with_trace(
            proof_data,
            num_layers,
            matmul_dims,
            dequantize_bits,
            initial_claim,
            ref ch,
        );
    (final_claim, weight_claims)
}

/// Verify a complete GKR model proof and return additional trace metadata.
///
/// Returns:
///   - final_input_claim
///   - weight_claims (main + deferred)
///   - layer_tags observed in proof order (for circuit hash binding)
///   - deferred_weight_commitments (one per deferred matmul proof)
pub fn verify_gkr_model_with_trace(
    proof_data: Span<felt252>,
    num_layers: u32,
    matmul_dims: Span<u32>,
    dequantize_bits: Span<u64>,
    initial_claim: GKRClaim,
    ref ch: PoseidonChannel,
) -> (GKRClaim, Array<WeightClaimData>, Array<u32>, Array<felt252>) {
    let mut reader = reader_new(proof_data);
    let mut current_claim = initial_claim;
    let mut weight_claims: Array<WeightClaimData> = array![];
    let mut layer_tags: Array<u32> = array![];
    let mut deferred_weight_commitments: Array<felt252> = array![];

    // Counters for per-type dimension arrays
    let mut matmul_idx: u32 = 0;
    let mut dequantize_idx: u32 = 0;

    // Save claim points at each Add layer for deferred proof reconstruction.
    // DAG Add layers (residual connections) produce deferred proofs for skip
    // branches. The deferred claim's point = walk claim point at the Add layer.
    let mut deferred_add_points: Array<Array<QM31>> = array![];

    let mut layer_idx: u32 = 0;
    loop {
        if layer_idx >= num_layers {
            break;
        }

        let tag = read_u32(ref reader);
        layer_tags.append(tag);

        if tag == 0 {
            // MatMul — collect weight claim for MLE opening verification
            let dims_base = matmul_idx * 3;
            assert!(dims_base + 2 < matmul_dims.len(), "MATMUL_DIMS_UNDERRUN");
            let m = *matmul_dims.at(dims_base);
            let k = *matmul_dims.at(dims_base + 1);
            let n = *matmul_dims.at(dims_base + 2);
            matmul_idx += 1;

            // Capture r_j from current claim before reduction
            let log_m = log2_ceil(next_power_of_two(m));
            let log_n = log2_ceil(next_power_of_two(n));

            let (new_claim, final_b_eval) = dispatch_matmul(
                @current_claim, m, k, n, ref reader, ref ch,
            );

            // Build weight evaluation point: [r_j || sumcheck_challenges]
            // r_j = current_claim.point[log_m..log_m+log_n]
            // sumcheck_challenges = new_claim.point[log_m..]
            let mut eval_point: Array<QM31> = array![];
            let mut j: u32 = 0;
            loop {
                if j >= log_n {
                    break;
                }
                eval_point.append(*current_claim.point.at(log_m + j));
                j += 1;
            };
            j = log_m;
            loop {
                if j >= new_claim.point.len() {
                    break;
                }
                eval_point.append(*new_claim.point.at(j));
                j += 1;
            };

            weight_claims.append(WeightClaimData {
                eval_point,
                expected_value: final_b_eval,
            });

            current_claim = new_claim;
        } else if tag == 1 {
            // Add — save claim point for deferred proof (skip connection)
            let claim_snap = @current_claim;
            deferred_add_points.append(clone_point(claim_snap.point));
            current_claim = dispatch_add(@current_claim, ref reader, ref ch);
        } else if tag == 2 {
            // Mul
            current_claim = dispatch_mul(@current_claim, ref reader, ref ch);
        } else if tag == 3 {
            // Activation
            current_claim = dispatch_activation(@current_claim, ref reader, ref ch);
        } else if tag == 4 {
            // LayerNorm
            current_claim = dispatch_layernorm(@current_claim, ref reader, ref ch);
        } else if tag == 6 {
            // Dequantize
            assert!(dequantize_idx < dequantize_bits.len(), "DEQUANTIZE_BITS_UNDERRUN");
            let bits = *dequantize_bits.at(dequantize_idx);
            dequantize_idx += 1;
            current_claim = dispatch_dequantize(@current_claim, bits, ref reader, ref ch);
        } else if tag == 8 {
            // RMSNorm
            current_claim = dispatch_rmsnorm(@current_claim, ref reader, ref ch);
        } else {
            assert!(false, "UNKNOWN_LAYER_TAG");
        }

        layer_idx += 1;
    };

    assert!(matmul_idx * 3 == matmul_dims.len(), "MATMUL_DIMS_TRAILING");
    assert!(dequantize_idx == dequantize_bits.len(), "DEQUANTIZE_BITS_TRAILING");

    // ========================================================================
    // Deferred Proofs (DAG Add skip connections)
    // ========================================================================
    // After the main walk, verify deferred matmul sumcheck proofs for skip
    // branches of Add layers. Each Add layer saved its claim point above.
    // Fiat-Shamir order: walk -> deferred proofs -> weight openings.
    let num_deferred = read_u32(ref reader);
    assert!(
        num_deferred <= deferred_add_points.len(),
        "DEFERRED_COUNT_EXCEEDS_ADDS",
    );

    let deferred_points_span = deferred_add_points.span();
    let mut def_idx: u32 = 0;
    loop {
        if def_idx >= num_deferred {
            break;
        }

        // Read deferred claim value (skip_eval from Add reduction)
        let claim_value = read_qm31(ref reader);

        // Reconstruct deferred claim point from saved Add layer point
        let point_snap = deferred_points_span.at(def_idx);
        let deferred_point = clone_point(point_snap);

        // Mix claim value into Fiat-Shamir channel (matches Rust prover)
        channel_mix_secure_field(ref ch, claim_value);

        // Read MatMul dimensions for the skip branch
        let m = read_u32(ref reader);
        let k = read_u32(ref reader);
        let n = read_u32(ref reader);

        let log_m = log2_ceil(next_power_of_two(m));
        let log_n = log2_ceil(next_power_of_two(n));

        // Construct and verify deferred matmul sumcheck
        let deferred_claim = GKRClaim { point: deferred_point, value: claim_value };
        let (new_claim, final_b_eval) = dispatch_matmul(
            @deferred_claim, m, k, n, ref reader, ref ch,
        );

        // Build weight evaluation point: [r_j || sumcheck_challenges]
        let mut eval_point: Array<QM31> = array![];
        let mut j: u32 = 0;
        loop {
            if j >= log_n {
                break;
            }
            eval_point.append(*deferred_claim.point.at(log_m + j));
            j += 1;
        };
        j = log_m;
        loop {
            if j >= new_claim.point.len() {
                break;
            }
            eval_point.append(*new_claim.point.at(j));
            j += 1;
        };

        weight_claims.append(WeightClaimData {
            eval_point,
            expected_value: final_b_eval,
        });

        // Read deferred weight commitment (bound by caller against registration)
        let deferred_weight_commitment = read_felt(ref reader);
        deferred_weight_commitments.append(deferred_weight_commitment);

        def_idx += 1;
    };

    assert!(reader.offset == reader.data.len(), "PROOF_DATA_TRAILING");
    (current_claim, weight_claims, layer_tags, deferred_weight_commitments)
}
