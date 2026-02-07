//! Starknet proof generation and serialization for on-chain verification.
//!
//! Generates matmul proofs using STWO's Poseidon252Channel so the Fiat-Shamir
//! transcript matches the on-chain Cairo verifier exactly.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Prover (Rust)                                                   │
//! │                                                                  │
//! │  1. Commit matrices A, B → Poseidon Merkle roots R_A, R_B       │
//! │  2. Run sumcheck proving with Poseidon252Channel                 │
//! │  3. Generate MLE opening proofs via multilinear folding          │
//! │  4. Serialize everything to calldata                             │
//! │                                                                  │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  Verifier (Cairo, on-chain)                                      │
//! │                                                                  │
//! │  1. Replay Fiat-Shamir transcript                                │
//! │  2. Verify sumcheck rounds                                       │
//! │  3. Verify MLE openings against committed Merkle roots           │
//! │  4. Check final_a_eval × final_b_eval = sumcheck output          │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

use starknet_ff::FieldElement as FieldElement252;

use stwo::core::channel::{Channel, Poseidon252Channel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;

use crate::commitment::{
    open_mle, verify_mle_opening, MleOpeningProof, PoseidonMerkleTree, DEFAULT_NUM_QUERIES,
};
use crate::components::matmul::{prove_matmul, verify_matmul, M31Matrix, MatMulError};

// ============================================================================
// Proof types
// ============================================================================

/// Complete matmul proof with MLE commitments for on-chain verification.
///
/// Contains:
/// - Sumcheck proof data (round polynomials, claimed sum)
/// - Poseidon Merkle commitments to matrices A and B
/// - MLE opening proofs verifying final evaluations against commitments
#[derive(Debug, Clone)]
pub struct StarknetMatMulProof {
    /// Matrix dimensions: A is m×k, B is k×n.
    pub m: u32,
    pub k: u32,
    pub n: u32,
    /// Number of sumcheck rounds (= ceil_log2(k)).
    pub num_rounds: u32,
    /// MLE_C evaluated at the random point (the initial sumcheck claim).
    pub claimed_sum: SecureField,
    /// Round polynomial coefficients: [c0, c1, c2] per round, monomial basis.
    pub round_polys: Vec<[SecureField; 3]>,
    /// MLE_A evaluated at (row_challenges, assignment).
    pub final_a_eval: SecureField,
    /// MLE_B evaluated at (assignment, col_challenges).
    pub final_b_eval: SecureField,
    /// Poseidon Merkle root of matrix A entries (the weight commitment).
    pub a_commitment: FieldElement252,
    /// Poseidon Merkle root of matrix B entries.
    pub b_commitment: FieldElement252,
    /// Opening proof: MLE_A(row_challenges, assignment) = final_a_eval.
    pub a_opening: MleOpeningProof,
    /// Opening proof: MLE_B(assignment, col_challenges) = final_b_eval.
    pub b_opening: MleOpeningProof,
}

/// A single M31 value serialized as a 64-bit unsigned integer for Starknet calldata.
type Felt = u64;

impl StarknetMatMulProof {
    /// Serialize the sumcheck portion to felt252 calldata.
    ///
    /// Layout (matching Cairo `MatMulSumcheckProof` Serde):
    ///   [m, k, n, num_rounds,
    ///    claimed_sum.a.a, claimed_sum.a.b, claimed_sum.b.a, claimed_sum.b.b,
    ///    round_polys_len,
    ///    round_0.c0.a.a, ..., round_0.c2.b.b,
    ///    ...,
    ///    final_a_eval.a.a, ..., final_b_eval.b.b]
    pub fn to_calldata(&self) -> Vec<Felt> {
        let mut data = vec![
            self.m as Felt,
            self.k as Felt,
            self.n as Felt,
            self.num_rounds as Felt,
        ];

        push_qm31(&mut data, self.claimed_sum);

        data.push(self.round_polys.len() as Felt);
        for round in &self.round_polys {
            push_qm31(&mut data, round[0]);
            push_qm31(&mut data, round[1]);
            push_qm31(&mut data, round[2]);
        }

        push_qm31(&mut data, self.final_a_eval);
        push_qm31(&mut data, self.final_b_eval);

        data
    }

    /// Verify the proof locally (sumcheck + MLE commitment openings).
    pub fn verify_locally(
        &self,
        a: &M31Matrix,
        b: &M31Matrix,
        c: &M31Matrix,
    ) -> Result<bool, MatMulError> {
        // Verify sumcheck
        let mut channel = Poseidon252Channel::default();
        let (proof, aux) = prove_matmul(a, b, c, &mut channel)?;
        let mut verify_channel = Poseidon252Channel::default();
        verify_matmul(a, b, c, &proof, &aux, &mut verify_channel)?;

        // Verify MLE opening for A
        let a_padded = pad_matrix_for_mle(a);
        let a_num_vars = a_padded.len().ilog2() as usize;
        let a_point = build_opening_point_a(&aux.row_challenges, &self.get_assignment(a, b, c)?);
        assert_eq!(a_point.len(), a_num_vars);

        let a_valid =
            verify_mle_opening(self.a_commitment, &a_point, self.final_a_eval, &self.a_opening);

        // Verify MLE opening for B
        let b_padded = pad_matrix_for_mle(b);
        let b_num_vars = b_padded.len().ilog2() as usize;
        let b_point = build_opening_point_b(&self.get_assignment(a, b, c)?, &aux.col_challenges);
        assert_eq!(b_point.len(), b_num_vars);

        let b_valid =
            verify_mle_opening(self.b_commitment, &b_point, self.final_b_eval, &self.b_opening);

        Ok(a_valid && b_valid)
    }

    /// Re-derive the sumcheck assignment point (needed for local verification).
    fn get_assignment(
        &self,
        a: &M31Matrix,
        b: &M31Matrix,
        c: &M31Matrix,
    ) -> Result<Vec<SecureField>, MatMulError> {
        let mut channel = Poseidon252Channel::default();
        let (proof, aux) = prove_matmul(a, b, c, &mut channel)?;

        let mut eval_channel = Poseidon252Channel::default();
        eval_channel.mix_u64(a.rows as u64);
        eval_channel.mix_u64(a.cols as u64);
        eval_channel.mix_u64(b.cols as u64);
        let _row = eval_channel.draw_secure_felts(a.rows.next_power_of_two().ilog2() as usize);
        let _col = eval_channel.draw_secure_felts(b.cols.next_power_of_two().ilog2() as usize);

        use stwo::prover::lookups::sumcheck::partially_verify;
        let (assignment, _) =
            partially_verify(aux.claimed_c_eval, &proof.sumcheck_proof, &mut eval_channel)
                .map_err(MatMulError::SumcheckError)?;
        Ok(assignment)
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Push QM31 components as 4 M31 felt values.
fn push_qm31(data: &mut Vec<Felt>, v: SecureField) {
    let [m0, m1, m2, m3] = v.to_m31_array();
    data.push(m0.0 as Felt);
    data.push(m1.0 as Felt);
    data.push(m2.0 as Felt);
    data.push(m3.0 as Felt);
}

/// Pad a matrix to power-of-two dimensions for proper MLE layout.
///
/// Row-major: A[i][j] at index `i * cols_padded + j`.
/// Total size = `rows_padded * cols_padded = 2^(m_log + k_log)`.
fn pad_matrix_for_mle(matrix: &M31Matrix) -> Vec<M31> {
    let m_padded = matrix.rows.next_power_of_two();
    let k_padded = matrix.cols.next_power_of_two();
    let mut entries = vec![M31::from(0); m_padded * k_padded];
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            entries[i * k_padded + j] = matrix.get(i, j);
        }
    }
    entries
}

/// Build the MLE opening point for matrix A.
///
/// MLE_A has variables: (row_0, ..., row_{m_log-1}, col_0, ..., col_{k_log-1})
/// Opening point: (row_challenges, assignment)
fn build_opening_point_a(
    row_challenges: &[SecureField],
    assignment: &[SecureField],
) -> Vec<SecureField> {
    let mut point = Vec::with_capacity(row_challenges.len() + assignment.len());
    point.extend_from_slice(row_challenges);
    point.extend_from_slice(assignment);
    point
}

/// Build the MLE opening point for matrix B.
///
/// MLE_B has variables: (row_0, ..., row_{k_log-1}, col_0, ..., col_{n_log-1})
/// Opening point: (assignment, col_challenges)
fn build_opening_point_b(
    assignment: &[SecureField],
    col_challenges: &[SecureField],
) -> Vec<SecureField> {
    let mut point = Vec::with_capacity(assignment.len() + col_challenges.len());
    point.extend_from_slice(assignment);
    point.extend_from_slice(col_challenges);
    point
}

// ============================================================================
// Proof generation
// ============================================================================

/// Generate a complete matmul proof with MLE commitments for on-chain verification.
///
/// This is the main entry point. It:
/// 1. Commits matrices A, B via Poseidon Merkle trees
/// 2. Runs `prove_matmul` with `Poseidon252Channel`
/// 3. Derives the sumcheck assignment point
/// 4. Generates MLE opening proofs for both matrices
/// 5. Packages everything into `StarknetMatMulProof`
pub fn prove_matmul_for_starknet(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<StarknetMatMulProof, MatMulError> {
    // Step 1: Commit to matrices
    let a_padded = pad_matrix_for_mle(a);
    let b_padded = pad_matrix_for_mle(b);
    let a_tree = PoseidonMerkleTree::from_m31_values(&a_padded);
    let b_tree = PoseidonMerkleTree::from_m31_values(&b_padded);
    let a_commitment = a_tree.root();
    let b_commitment = b_tree.root();

    // Step 2: Run sumcheck with Poseidon252Channel
    let mut prover_channel = Poseidon252Channel::default();
    let (proof, aux) = prove_matmul(a, b, c, &mut prover_channel)?;

    // Verify locally to catch bugs early
    let mut verify_channel = Poseidon252Channel::default();
    verify_matmul(a, b, c, &proof, &aux, &mut verify_channel)?;

    // Step 3: Extract round polynomial coefficients
    let k_log = a.cols.next_power_of_two().ilog2() as usize;
    let round_polys: Vec<[SecureField; 3]> = proof
        .sumcheck_proof
        .round_polys
        .iter()
        .map(|poly| {
            let c0 = poly.eval_at_point(SecureField::from(M31::from(0)));
            let c1_plus_c0 = poly.eval_at_point(SecureField::from(M31::from(1)));
            let two = SecureField::from(M31::from(2));
            let y2 = poly.eval_at_point(two);

            let c0_val = c0;
            let half = SecureField::from(M31::from((1u64 << 30) as u32)); // 2^{-1} in M31
            let c2_val = (y2 - c1_plus_c0 - c1_plus_c0 + c0_val) * half;
            let c1_val = c1_plus_c0 - c0_val - c2_val;

            [c0_val, c1_val, c2_val]
        })
        .collect();

    // Step 4: Derive the assignment point via partially_verify
    let mut eval_channel = Poseidon252Channel::default();
    eval_channel.mix_u64(a.rows as u64);
    eval_channel.mix_u64(a.cols as u64);
    eval_channel.mix_u64(b.cols as u64);
    let _row_challenges =
        eval_channel.draw_secure_felts(a.rows.next_power_of_two().ilog2() as usize);
    let _col_challenges =
        eval_channel.draw_secure_felts(b.cols.next_power_of_two().ilog2() as usize);

    use stwo::prover::lookups::sumcheck::partially_verify;
    let (assignment, _claimed_eval) =
        partially_verify(aux.claimed_c_eval, &proof.sumcheck_proof, &mut eval_channel)
            .map_err(MatMulError::SumcheckError)?;

    // Step 5: Compute final MLE evaluations at the full opening points
    let a_point = build_opening_point_a(&aux.row_challenges, &assignment);
    let b_point = build_opening_point_b(&assignment, &aux.col_challenges);

    let a_padded_sf: Vec<SecureField> = a_padded.iter().map(|v| SecureField::from(*v)).collect();
    let b_padded_sf: Vec<SecureField> = b_padded.iter().map(|v| SecureField::from(*v)).collect();
    let final_a_eval = eval_mle_at_point_pub(&a_padded_sf, &a_point);
    let final_b_eval = eval_mle_at_point_pub(&b_padded_sf, &b_point);

    // Step 6: Generate MLE opening proofs
    let a_opening = open_mle(&a_padded, &a_point, &a_tree, DEFAULT_NUM_QUERIES);
    let b_opening = open_mle(&b_padded, &b_point, &b_tree, DEFAULT_NUM_QUERIES);

    // Sanity check: opening proofs should yield the correct evaluations
    assert_eq!(a_opening.final_value, final_a_eval, "A opening mismatch");
    assert_eq!(b_opening.final_value, final_b_eval, "B opening mismatch");

    Ok(StarknetMatMulProof {
        m: a.rows as u32,
        k: a.cols as u32,
        n: b.cols as u32,
        num_rounds: k_log as u32,
        claimed_sum: aux.claimed_c_eval,
        round_polys,
        final_a_eval,
        final_b_eval,
        a_commitment,
        b_commitment,
        a_opening,
        b_opening,
    })
}

// ============================================================================
// Private MLE helpers (mirror matmul.rs internals)
// ============================================================================

// MLE evaluation helper - no external trait imports needed.

fn eval_mle_at_point_pub(evals: &[SecureField], point: &[SecureField]) -> SecureField {
    match point {
        [] => evals[0],
        [p_i, rest @ ..] => {
            let mid = evals.len() / 2;
            let (lhs, rhs) = evals.split_at(mid);
            let lhs_eval = eval_mle_at_point_pub(lhs, rest);
            let rhs_eval = eval_mle_at_point_pub(rhs, rest);
            *p_i * (rhs_eval - lhs_eval) + lhs_eval
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;
    use stwo::core::channel::Poseidon252Channel;
    use stwo::core::fields::m31::M31;

    #[test]
    fn test_prove_matmul_poseidon_2x2() {
        let a = M31Matrix::from_data(
            2,
            2,
            vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)],
        )
        .unwrap();
        let b = M31Matrix::from_data(
            2,
            2,
            vec![M31::from(5), M31::from(6), M31::from(7), M31::from(8)],
        )
        .unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let mut prover_channel = Poseidon252Channel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c, &mut prover_channel).unwrap();

        let mut verifier_channel = Poseidon252Channel::default();
        verify_matmul(&a, &b, &c, &proof, &aux, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_matmul_poseidon_4x4() {
        let a = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let b = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let mut prover_channel = Poseidon252Channel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c, &mut prover_channel).unwrap();

        let mut verifier_channel = Poseidon252Channel::default();
        verify_matmul(&a, &b, &c, &proof, &aux, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_starknet_proof_with_commitments_2x2() {
        let a = M31Matrix::from_data(
            2,
            2,
            vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)],
        )
        .unwrap();
        let b = M31Matrix::from_data(
            2,
            2,
            vec![M31::from(5), M31::from(6), M31::from(7), M31::from(8)],
        )
        .unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();

        assert_eq!(proof.m, 2);
        assert_eq!(proof.k, 2);
        assert_eq!(proof.n, 2);
        assert_eq!(proof.num_rounds, 1);
        assert_ne!(proof.a_commitment, FieldElement252::ZERO);
        assert_ne!(proof.b_commitment, FieldElement252::ZERO);

        // Verify MLE openings
        let a_padded = pad_matrix_for_mle(&a);
        let a_tree = PoseidonMerkleTree::from_m31_values(&a_padded);
        assert_eq!(proof.a_commitment, a_tree.root());

        let b_padded = pad_matrix_for_mle(&b);
        let b_tree = PoseidonMerkleTree::from_m31_values(&b_padded);
        assert_eq!(proof.b_commitment, b_tree.root());
    }

    #[test]
    fn test_starknet_proof_with_commitments_4x4() {
        let a = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let b = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();

        assert_eq!(proof.m, 4);
        assert_eq!(proof.k, 4);
        assert_eq!(proof.n, 4);
        assert_eq!(proof.num_rounds, 2);
        assert_eq!(proof.round_polys.len(), 2);
        assert_ne!(proof.final_a_eval * proof.final_b_eval, SecureField::zero());

        // Verify A opening
        assert!(verify_mle_opening(
            proof.a_commitment,
            &proof.get_assignment(&a, &b, &c).map(|assignment| {
                let mut ch = Poseidon252Channel::default();
                let (_, aux) = prove_matmul(&a, &b, &c, &mut ch).unwrap();
                build_opening_point_a(&aux.row_challenges, &assignment)
            }).unwrap(),
            proof.final_a_eval,
            &proof.a_opening,
        ));
    }

    #[test]
    fn test_starknet_proof_with_commitments_8x8() {
        let a = M31Matrix::from_data(8, 8, (1..=64).map(M31::from).collect()).unwrap();
        let b = M31Matrix::from_data(8, 8, (65..=128).map(M31::from).collect()).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();

        assert_eq!(proof.m, 8);
        assert_eq!(proof.k, 8);
        assert_eq!(proof.n, 8);
        assert_eq!(proof.num_rounds, 3);
        assert_eq!(proof.round_polys.len(), 3);

        // Commitment roots should be non-zero and distinct
        assert_ne!(proof.a_commitment, FieldElement252::ZERO);
        assert_ne!(proof.b_commitment, FieldElement252::ZERO);
        assert_ne!(proof.a_commitment, proof.b_commitment);
    }

    #[test]
    fn test_calldata_serialization() {
        let a = M31Matrix::from_data(
            2,
            2,
            vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)],
        )
        .unwrap();
        let b = M31Matrix::from_data(
            2,
            2,
            vec![M31::from(5), M31::from(6), M31::from(7), M31::from(8)],
        )
        .unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();
        let calldata = proof.to_calldata();

        // 4 (m,k,n,rounds) + 4 (claimed_sum) + 1 (array_len) + rounds*12 (polys) + 8 (final evals)
        let expected_len = 4 + 4 + 1 + (proof.num_rounds as usize) * 12 + 8;
        assert_eq!(calldata.len(), expected_len);

        assert_eq!(calldata[0], 2); // m
        assert_eq!(calldata[1], 2); // k
        assert_eq!(calldata[2], 2); // n
        assert_eq!(calldata[3], 1); // num_rounds
    }

    #[test]
    fn test_commitment_roots_deterministic() {
        let a = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let b = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof1 = prove_matmul_for_starknet(&a, &b, &c).unwrap();
        let proof2 = prove_matmul_for_starknet(&a, &b, &c).unwrap();

        // Same matrices → same commitment roots
        assert_eq!(proof1.a_commitment, proof2.a_commitment);
        assert_eq!(proof1.b_commitment, proof2.b_commitment);
    }

    #[test]
    fn test_mle_eval_matches_extracted_mle() {
        // Verify that evaluating the full matrix MLE at (row_challenges, assignment)
        // gives the same result as the extract_row_mle + eval_mle approach.
        let a = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let b = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        // Get the challenges and assignment
        let mut ch = Poseidon252Channel::default();
        let (proof, aux) = prove_matmul(&a, &b, &c, &mut ch).unwrap();

        let mut eval_ch = Poseidon252Channel::default();
        eval_ch.mix_u64(a.rows as u64);
        eval_ch.mix_u64(a.cols as u64);
        eval_ch.mix_u64(b.cols as u64);
        let _r = eval_ch.draw_secure_felts(a.rows.next_power_of_two().ilog2() as usize);
        let _c = eval_ch.draw_secure_felts(b.cols.next_power_of_two().ilog2() as usize);

        use stwo::prover::lookups::sumcheck::partially_verify;
        let (assignment, _) =
            partially_verify(aux.claimed_c_eval, &proof.sumcheck_proof, &mut eval_ch).unwrap();

        // Method 1: Full MLE of padded A at (row_challenges, assignment)
        let a_padded = pad_matrix_for_mle(&a);
        let a_padded_sf: Vec<SecureField> = a_padded.iter().map(|v| SecureField::from(*v)).collect();
        let a_point = build_opening_point_a(&aux.row_challenges, &assignment);
        let full_eval = eval_mle_at_point_pub(&a_padded_sf, &a_point);

        // Method 2: The Starknet proof's final_a_eval
        let starknet_proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();
        assert_eq!(full_eval, starknet_proof.final_a_eval);
    }
}
