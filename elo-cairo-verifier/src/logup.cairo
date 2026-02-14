// LogUp Table-Side Sum Verification
//
// Verifies the table side of a LogUp argument [Haboeck 2022]:
//   S_table = sum_j mult_j / (gamma - in_j - beta * out_j)
//
// Uses Montgomery batch inversion to compute all reciprocals with
// a single field inversion + 3(N-1) multiplications.
//
// This module is used by activation, layernorm, rmsnorm, dequantize,
// and embedding layer verifiers when full table-side verification
// is required (ADR-2 Approach B).

use crate::field::{
    QM31, qm31_zero, qm31_add, qm31_sub, qm31_mul,
    m31_to_qm31, qm31_from_u32, batch_inverse,
};

/// Verify the LogUp table-side sum matches the claimed sum.
///
/// For each table entry j with multiplicity mult_j > 0:
///   d_j = gamma - M31(table_input_j) - beta * M31(table_output_j)
///   contribution_j = mult_j / d_j
///
/// Returns true iff sum of all contributions == claimed_sum.
///
/// Arguments:
///   gamma, beta: Fiat-Shamir challenges drawn by the verifier
///   table_inputs: M31 input values of the lookup table (as u64)
///   table_outputs: M31 output values of the lookup table (as u64)
///   multiplicities: per-entry access counts
///   claimed_sum: prover's claimed total LogUp sum
pub fn verify_logup_table_sum(
    gamma: QM31,
    beta: QM31,
    table_inputs: Span<u64>,
    table_outputs: Span<u64>,
    multiplicities: Span<u32>,
    claimed_sum: QM31,
) -> bool {
    let n = table_inputs.len();
    assert!(n == table_outputs.len(), "LOGUP_TABLE_LEN_MISMATCH");
    assert!(n == multiplicities.len(), "LOGUP_MULT_LEN_MISMATCH");

    // Collect non-zero entries and their denominators
    let mut denoms: Array<QM31> = array![];
    let mut mults: Array<u32> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= n {
            break;
        }
        let m = *multiplicities.at(i);
        if m > 0 {
            // d = gamma - in_j - beta * out_j
            let in_qm31 = m31_to_qm31(*table_inputs.at(i));
            let out_qm31 = m31_to_qm31(*table_outputs.at(i));
            let d = qm31_sub(qm31_sub(gamma, in_qm31), qm31_mul(beta, out_qm31));
            denoms.append(d);
            mults.append(m);
        }
        i += 1;
    };

    if denoms.len() == 0 {
        // No entries accessed: sum should be zero
        return claimed_sum == qm31_zero();
    }

    // Batch invert all denominators
    let inv_denoms = batch_inverse(denoms.span());

    // Accumulate: S = sum_j mult_j * inv_d_j
    let mut sum = qm31_zero();
    let mut i: u32 = 0;
    loop {
        if i >= inv_denoms.len() {
            break;
        }
        let mult_qm31 = qm31_from_u32(*mults.at(i));
        sum = qm31_add(sum, qm31_mul(mult_qm31, *inv_denoms.at(i)));
        i += 1;
    };

    sum == claimed_sum
}
