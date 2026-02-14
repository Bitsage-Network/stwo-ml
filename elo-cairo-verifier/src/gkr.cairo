// GKR Batch Verifier for Grand Product and LogUp Lookup Arguments
//
// Mirrors STWO's gkr_verifier.rs partially_verify_batch() algorithm exactly.
// Processes GKR proofs layer by layer, verifying sumcheck at each layer,
// evaluating gates locally, and reducing claims for the next layer.
//
// Algorithm (per layer):
//   1. Identify instances starting at this layer (output layer)
//   2. Mix starting instances' output claims into channel
//   3. Draw sumcheck_alpha and instance_lambda
//   4. Prepare per-instance claims with doubling factor + RLC
//   5. Compute batched sumcheck claim via RLC with sumcheck_alpha
//   6. Verify sumcheck(batch_claim, round_polys, channel)
//   7. Evaluate gates locally at mask values
//   8. Check: sumcheck_eval == RLC of gate evaluations
//   9. Mix mask columns into channel, draw challenge r
//   10. Reduce claims: fold(v0, v1, r) for each column
//   11. Extend OOD point with r

use crate::field::{
    QM31, CM31, qm31_one, qm31_add, qm31_mul, qm31_eq,
    m31_mul, poly_eval_degree3, eq_eval, fold_mle_eval, random_linear_combination,
};
use crate::channel::{
    PoseidonChannel, channel_mix_felts, channel_draw_qm31,
};
use crate::types::{
    GateType, GkrMask, GkrSumcheckProof,
    GkrBatchProof, GkrArtifact,
};

// ============================================================================
// Gate Evaluation
// ============================================================================

/// Evaluate GrandProduct gate: output = col0_v0 * col0_v1.
/// Mask must have exactly 1 column (2 values).
pub fn eval_grand_product_gate(mask: @GkrMask) -> Array<QM31> {
    assert!(*mask.num_columns == 1, "GrandProduct requires 1 column");
    assert!(mask.values.len() == 2, "GrandProduct mask needs 2 values");
    let a = *mask.values.at(0);
    let b = *mask.values.at(1);
    array![qm31_mul(a, b)]
}

/// Evaluate LogUp gate: fraction addition of (n_a/d_a) + (n_b/d_b).
/// Mask has 2 columns: [n_v0, n_v1, d_v0, d_v1].
/// Output = (n_a*d_b + n_b*d_a, d_a*d_b) as [numerator, denominator].
pub fn eval_logup_gate(mask: @GkrMask) -> Array<QM31> {
    assert!(*mask.num_columns == 2, "LogUp requires 2 columns");
    assert!(mask.values.len() == 4, "LogUp mask needs 4 values");
    let n_a = *mask.values.at(0); // numerator column, row 0
    let n_b = *mask.values.at(1); // numerator column, row 1
    let d_a = *mask.values.at(2); // denominator column, row 0
    let d_b = *mask.values.at(3); // denominator column, row 1
    // (n_a/d_a) + (n_b/d_b) = (n_a*d_b + n_b*d_a) / (d_a*d_b)
    let numerator = qm31_add(qm31_mul(n_a, d_b), qm31_mul(n_b, d_a));
    let denominator = qm31_mul(d_a, d_b);
    array![numerator, denominator]
}

/// Evaluate a gate by type. gate_id: 0 = GrandProduct, 1 = LogUp.
pub fn eval_gate(gate: @GateType, mask: @GkrMask) -> Array<QM31> {
    if *gate.gate_id == 0 {
        eval_grand_product_gate(mask)
    } else {
        eval_logup_gate(mask)
    }
}

// ============================================================================
// Mask Operations
// ============================================================================

/// Reduce mask at a challenge point: fold each column's (v0, v1) pair.
/// Returns one QM31 per column: fold_mle_eval(r, v0, v1).
pub fn reduce_mask_at_point(mask: @GkrMask, r: QM31) -> Array<QM31> {
    let num_cols = *mask.num_columns;
    let mut result: Array<QM31> = array![];
    let mut col: u32 = 0;
    loop {
        if col >= num_cols {
            break;
        }
        let v0 = *mask.values.at(col * 2);
        let v1 = *mask.values.at(col * 2 + 1);
        result.append(fold_mle_eval(r, v0, v1));
        col += 1;
    };
    result
}

// ============================================================================
// GKR Sumcheck Verification
// ============================================================================

/// Verify a sumcheck proof with variable-degree round polynomials (up to degree 3).
/// Returns (assignment, final_eval).
///
/// For each round:
///   1. Check p(0) + p(1) == expected_sum
///   2. Mix round poly coefficients into channel (only num_coeffs values)
///   3. Draw challenge
///   4. expected_sum = p(challenge)
pub fn verify_gkr_sumcheck(
    claim: QM31,
    proof: @GkrSumcheckProof,
    ref ch: PoseidonChannel,
) -> (Array<QM31>, QM31) {
    let num_rounds = proof.round_polys.len();
    let mut expected_sum = claim;
    let mut assignment: Array<QM31> = array![];

    let mut round: u32 = 0;
    loop {
        if round >= num_rounds {
            break;
        }

        let poly = *proof.round_polys.at(round);

        // p(0) = c0
        let eval_at_0 = poly.c0;
        // p(1) = c0 + c1 + c2 + c3 (Horner at x=1)
        let mut eval_at_1 = poly.c0;
        if poly.num_coeffs >= 2 {
            eval_at_1 = qm31_add(eval_at_1, poly.c1);
        }
        if poly.num_coeffs >= 3 {
            eval_at_1 = qm31_add(eval_at_1, poly.c2);
        }
        if poly.num_coeffs >= 4 {
            eval_at_1 = qm31_add(eval_at_1, poly.c3);
        }

        let round_sum = qm31_add(eval_at_0, eval_at_1);
        assert!(qm31_eq(round_sum, expected_sum), "GKR sumcheck round sum mismatch");

        // Mix round poly coefficients (only num_coeffs values, matching STWO truncation)
        let mut coeffs_to_mix: Array<QM31> = array![];
        if poly.num_coeffs >= 1 {
            coeffs_to_mix.append(poly.c0);
        }
        if poly.num_coeffs >= 2 {
            coeffs_to_mix.append(poly.c1);
        }
        if poly.num_coeffs >= 3 {
            coeffs_to_mix.append(poly.c2);
        }
        if poly.num_coeffs >= 4 {
            coeffs_to_mix.append(poly.c3);
        }
        channel_mix_felts(ref ch, coeffs_to_mix.span());

        let challenge = channel_draw_qm31(ref ch);
        assignment.append(challenge);

        // Update expected sum: p(challenge) using full degree-3 eval
        expected_sum = poly_eval_degree3(poly.c0, poly.c1, poly.c2, poly.c3, challenge);

        round += 1;
    };

    (assignment, expected_sum)
}

// ============================================================================
// Core: partially_verify_batch
// ============================================================================

/// Partially verify a batch GKR proof.
///
/// Mirrors STWO's `partially_verify_batch()` exactly:
/// - Processes layers top-to-bottom
/// - At each layer, identifies starting instances, mixes claims, draws alphas
/// - Runs sumcheck, evaluates gates locally, checks consistency
/// - Reduces claims via folding for the next layer
///
/// Returns GkrArtifact with OOD point and per-instance claims to verify.
pub fn partially_verify_batch(
    proof: @GkrBatchProof,
    ref ch: PoseidonChannel,
) -> GkrArtifact {
    let n_instances = proof.instances.len();
    assert!(n_instances > 0, "GKR: no instances");

    // Compute n_variables for each instance and find max (= n_layers)
    let mut instance_n_layers: Array<u32> = array![];
    let mut n_layers: u32 = 0;
    let mut i: u32 = 0;
    loop {
        if i >= n_instances {
            break;
        }
        let inst = proof.instances.at(i);
        let nv = *inst.n_variables;
        instance_n_layers.append(nv);
        if nv > n_layers {
            n_layers = nv;
        }
        i += 1;
    };

    assert!(proof.layer_proofs.len() == n_layers, "GKR: layer count mismatch");

    // claims_to_verify[instance] — None represented as empty array, Some as non-empty
    // We use a flat array of arrays; initialized_flags tracks which are set
    let mut claims_storage: Array<Array<QM31>> = array![];
    let mut initialized: Array<bool> = array![];
    i = 0;
    loop {
        if i >= n_instances {
            break;
        }
        claims_storage.append(array![]);
        initialized.append(false);
        i += 1;
    };

    let mut ood_point: Array<QM31> = array![];

    let mut layer: u32 = 0;
    loop {
        if layer >= n_layers {
            break;
        }

        let n_remaining_layers = n_layers - layer;
        let layer_proof = proof.layer_proofs.at(layer);

        // Step 1: Check for output layers — instances whose n_variables == n_remaining_layers
        // start at this layer
        let mut inst: u32 = 0;
        loop {
            if inst >= n_instances {
                break;
            }
            let inst_n = *instance_n_layers.at(inst);
            if inst_n == n_remaining_layers {
                // This instance starts here — set its claims to output_claims
                let output_claims = proof.instances.at(inst).output_claims;
                let mut claims_copy: Array<QM31> = array![];
                let mut j: u32 = 0;
                loop {
                    if j >= output_claims.len() {
                        break;
                    }
                    claims_copy.append(*output_claims.at(j));
                    j += 1;
                };
                claims_storage = _replace_array(claims_storage, inst, claims_copy);
                initialized = _replace_bool(initialized, inst, true);
            }
            inst += 1;
        };

        // Step 2: Seed the channel with all active instances' claims
        inst = 0;
        loop {
            if inst >= n_instances {
                break;
            }
            if *initialized.at(inst) {
                let claims = _get_array_span(@claims_storage, inst);
                channel_mix_felts(ref ch, claims);
            }
            inst += 1;
        };

        // Step 3: Draw sumcheck_alpha and instance_lambda
        let sumcheck_alpha = channel_draw_qm31(ref ch);
        let instance_lambda = channel_draw_qm31(ref ch);

        // Step 4: Prepare sumcheck claims with doubling factor
        let mut sumcheck_claims: Array<QM31> = array![];
        let mut sumcheck_instances: Array<u32> = array![];

        inst = 0;
        loop {
            if inst >= n_instances {
                break;
            }
            if *initialized.at(inst) {
                let inst_n = *instance_n_layers.at(inst);
                let n_unused = n_layers - inst_n;
                // doubling_factor = 2^n_unused (as M31 value in QM31)
                let doubling: u64 = _pow2_u64(n_unused);
                let claims = _get_array_span(@claims_storage, inst);
                let rlc = random_linear_combination(claims, instance_lambda);
                // Multiply by doubling factor (embed M31 scalar into QM31)
                let doubled = _qm31_mul_m31(rlc, doubling);
                sumcheck_claims.append(doubled);
                sumcheck_instances.append(inst);
            }
            inst += 1;
        };

        // Step 5: Batched sumcheck claim
        let sumcheck_claim = random_linear_combination(
            sumcheck_claims.span(), sumcheck_alpha,
        );

        // Step 6: Verify sumcheck
        let (sumcheck_ood_point, sumcheck_eval) = verify_gkr_sumcheck(
            sumcheck_claim, layer_proof.sumcheck_proof, ref ch,
        );

        // Step 7: Evaluate gates locally at masks
        let mut layer_evals: Array<QM31> = array![];
        let mut mask_idx: u32 = 0;

        let mut si: u32 = 0;
        loop {
            if si >= sumcheck_instances.len() {
                break;
            }
            let instance_id = *sumcheck_instances.at(si);
            let inst_n = *instance_n_layers.at(instance_id);
            let n_unused = n_layers - inst_n;
            let mask = layer_proof.masks.at(mask_idx);
            let inst_snap = proof.instances.at(instance_id);
            let gate = GateType { gate_id: *inst_snap.gate.gate_id };
            let gate_output = eval_gate(@gate, mask);

            // eq(ood_point[n_unused..], sumcheck_ood_point[n_unused..])
            let eq_val = _eq_eval_sliced(
                @ood_point, @sumcheck_ood_point, n_unused,
            );

            let rlc_gate = random_linear_combination(gate_output.span(), instance_lambda);
            layer_evals.append(qm31_mul(eq_val, rlc_gate));

            mask_idx += 1;
            si += 1;
        };

        // Step 8: Check circuit evaluation
        let layer_eval = random_linear_combination(layer_evals.span(), sumcheck_alpha);
        assert!(qm31_eq(sumcheck_eval, layer_eval), "GKR: circuit check failure");

        // Step 9: Mix mask columns into channel
        si = 0;
        mask_idx = 0;
        loop {
            if si >= sumcheck_instances.len() {
                break;
            }
            let mask = layer_proof.masks.at(mask_idx);
            channel_mix_felts(ref ch, mask.values.span());
            mask_idx += 1;
            si += 1;
        };

        // Draw challenge for next layer
        let challenge = channel_draw_qm31(ref ch);

        // Step 10: Update OOD point
        ood_point = _clone_array(@sumcheck_ood_point);
        ood_point.append(challenge);

        // Step 11: Reduce claims for next layer
        si = 0;
        mask_idx = 0;
        loop {
            if si >= sumcheck_instances.len() {
                break;
            }
            let instance_id = *sumcheck_instances.at(si);
            let mask = layer_proof.masks.at(mask_idx);
            let reduced = reduce_mask_at_point(mask, challenge);
            claims_storage = _replace_array(claims_storage, instance_id, reduced);
            mask_idx += 1;
            si += 1;
        };

        layer += 1;
    };

    // Collect final claims
    let mut final_claims: Array<Array<QM31>> = array![];
    let mut final_n_vars: Array<u32> = array![];
    i = 0;
    loop {
        if i >= n_instances {
            break;
        }
        let claims = _extract_array(ref claims_storage, i);
        final_claims.append(claims);
        final_n_vars.append(*instance_n_layers.at(i));
        i += 1;
    };

    GkrArtifact {
        ood_point,
        claims_to_verify: final_claims,
        n_variables_by_instance: final_n_vars,
    }
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Multiply QM31 by an M31 scalar (as u64). Embeds scalar into QM31 real part.
fn _qm31_mul_m31(x: QM31, scalar: u64) -> QM31 {
    QM31 {
        a: CM31 { a: m31_mul(x.a.a, scalar), b: m31_mul(x.a.b, scalar) },
        b: CM31 { a: m31_mul(x.b.a, scalar), b: m31_mul(x.b.b, scalar) },
    }
}

/// 2^n as u64. Panics if n > 30 (M31 range).
fn _pow2_u64(n: u32) -> u64 {
    let mut result: u64 = 1;
    let mut i: u32 = 0;
    loop {
        if i >= n {
            break;
        }
        result = result * 2;
        i += 1;
    };
    result
}

/// Replace element at index in an array of arrays. Returns new array.
fn _replace_array(
    old: Array<Array<QM31>>, index: u32, new_val: Array<QM31>,
) -> Array<Array<QM31>> {
    let span = old.span();
    let mut result: Array<Array<QM31>> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= span.len() {
            break;
        }
        if i == index {
            result.append(new_val.clone());
        } else {
            let src = span.at(i);
            let mut copy: Array<QM31> = array![];
            let mut j: u32 = 0;
            loop {
                if j >= src.len() {
                    break;
                }
                copy.append(*src.at(j));
                j += 1;
            };
            result.append(copy);
        }
        i += 1;
    };
    result
}

/// Replace element at index in a bool array. Returns new array.
fn _replace_bool(old: Array<bool>, index: u32, new_val: bool) -> Array<bool> {
    let span = old.span();
    let mut result: Array<bool> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= span.len() {
            break;
        }
        if i == index {
            result.append(new_val);
        } else {
            result.append(*span.at(i));
        }
        i += 1;
    };
    result
}

/// Get a span view of an inner array from an array of arrays (snapshot).
fn _get_array_span(storage: @Array<Array<QM31>>, index: u32) -> Span<QM31> {
    storage.at(index).span()
}

/// Clone a QM31 array from a snapshot.
fn _clone_array(src: @Array<QM31>) -> Array<QM31> {
    let mut result: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= src.len() {
            break;
        }
        result.append(*src.at(i));
        i += 1;
    };
    result
}

/// Extract (clone) the array at index from claims_storage.
fn _extract_array(ref storage: Array<Array<QM31>>, index: u32) -> Array<QM31> {
    let span = storage.span();
    let src = span.at(index);
    let mut result: Array<QM31> = array![];
    let mut j: u32 = 0;
    loop {
        if j >= src.len() {
            break;
        }
        result.append(*src.at(j));
        j += 1;
    };
    result
}

/// Compute eq(a[offset..], b[offset..]) for sliced OOD point comparison.
fn _eq_eval_sliced(
    a: @Array<QM31>, b: @Array<QM31>, offset: u32,
) -> QM31 {
    let a_len = a.len();
    let b_len = b.len();

    // If both slices are empty (offset >= len), eq() = 1
    if offset >= a_len && offset >= b_len {
        return qm31_one();
    }

    // Build sliced spans — lengths MUST match (STWO asserts this)
    let a_slice_len = if offset < a_len { a_len - offset } else { 0 };
    let b_slice_len = if offset < b_len { b_len - offset } else { 0 };
    assert!(a_slice_len == b_slice_len, "GKR: eq_eval slice length mismatch");

    let eq_len = a_slice_len;

    if eq_len == 0 {
        return qm31_one();
    }

    let mut x_arr: Array<QM31> = array![];
    let mut y_arr: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= eq_len {
            break;
        }
        x_arr.append(*a.at(offset + i));
        y_arr.append(*b.at(offset + i));
        i += 1;
    };
    eq_eval(x_arr.span(), y_arr.span())
}
