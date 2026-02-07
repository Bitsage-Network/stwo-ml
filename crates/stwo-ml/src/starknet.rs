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
use tracing::{info_span, instrument};

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

/// Starknet calldata element — a native felt252.
///
/// Cairo `Serde` deserializes each field from `Span<felt252>`, so every element
/// in calldata is a single felt252. Small values (u32, u64, M31) are embedded
/// directly; Poseidon hashes use the full 252-bit range.
type Felt = FieldElement252;

impl StarknetMatMulProof {
    /// Serialize the complete proof to felt252 calldata.
    ///
    /// Each element is a native `felt252`. Layout matches Cairo `MatMulSumcheckProof`
    /// Serde deserialization exactly:
    ///   [m, k, n, num_rounds,
    ///    claimed_sum(4 QM31 components),
    ///    round_polys_len, round_0.c0(4), round_0.c1(4), round_0.c2(4), ...,
    ///    final_a_eval(4), final_b_eval(4),
    ///    a_commitment(1 felt252), b_commitment(1 felt252),
    ///    a_opening: { intermediate_roots_len, [root(1 felt252 each)...],
    ///                 queries_len, [query: { pair_index,
    ///                   rounds_len, [round: { left(4), right(4),
    ///                     left_siblings_len, [sibling(1 felt252)...],
    ///                     right_siblings_len, [sibling(1 felt252)...] }] }] ,
    ///                 final_value(4) },
    ///    b_opening: { ... } ]
    pub fn to_calldata(&self) -> Vec<Felt> {
        let mut data = vec![
            FieldElement252::from(self.m as u64),
            FieldElement252::from(self.k as u64),
            FieldElement252::from(self.n as u64),
            FieldElement252::from(self.num_rounds as u64),
        ];

        push_qm31(&mut data, self.claimed_sum);

        data.push(FieldElement252::from(self.round_polys.len() as u64));
        for round in &self.round_polys {
            push_qm31(&mut data, round[0]);
            push_qm31(&mut data, round[1]);
            push_qm31(&mut data, round[2]);
        }

        push_qm31(&mut data, self.final_a_eval);
        push_qm31(&mut data, self.final_b_eval);

        // a_commitment as felt252 (single felt for Cairo)
        push_felt252(&mut data, self.a_commitment);

        // b_commitment as felt252
        push_felt252(&mut data, self.b_commitment);

        // a_opening
        push_mle_opening(&mut data, &self.a_opening);

        // b_opening
        push_mle_opening(&mut data, &self.b_opening);

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

/// Push QM31 as 4 felt252 values (one per M31 component).
///
/// Cairo layout: `QM31 { a: CM31 { a: u64, b: u64 }, b: CM31 { a: u64, b: u64 } }`
/// Serde order: a.a, a.b, b.a, b.b → 4 felt252 elements.
fn push_qm31(data: &mut Vec<Felt>, v: SecureField) {
    let [m0, m1, m2, m3] = v.to_m31_array();
    data.push(FieldElement252::from(m0.0 as u64));
    data.push(FieldElement252::from(m1.0 as u64));
    data.push(FieldElement252::from(m2.0 as u64));
    data.push(FieldElement252::from(m3.0 as u64));
}

/// Push a felt252 as a single calldata element.
///
/// Cairo's `Serde` for `felt252` reads one element from `Span<felt252>`.
fn push_felt252(data: &mut Vec<Felt>, v: FieldElement252) {
    data.push(v);
}

/// Serialize an MLE opening proof to calldata.
///
/// Layout (matching Cairo `MleOpeningProof` Serde):
///   intermediate_roots_len, [root_0(4 limbs), root_1(4 limbs), ...],
///   queries_len, [query_0, query_1, ...],
///   final_value(4 QM31 components)
fn push_mle_opening(data: &mut Vec<Felt>, proof: &MleOpeningProof) {
    // intermediate_roots: Array<felt252>
    data.push(FieldElement252::from(proof.intermediate_roots.len() as u64));
    for root in &proof.intermediate_roots {
        push_felt252(data, *root);
    }

    // queries: Array<MleQueryProof>
    data.push(FieldElement252::from(proof.queries.len() as u64));
    for query in &proof.queries {
        // initial_pair_index: u32
        data.push(FieldElement252::from(query.initial_pair_index as u64));

        // rounds: Array<MleQueryRoundData>
        data.push(FieldElement252::from(query.rounds.len() as u64));
        for round in &query.rounds {
            // left_value: QM31
            push_qm31(data, round.left_value);
            // right_value: QM31
            push_qm31(data, round.right_value);

            // left_siblings: Array<felt252>
            data.push(FieldElement252::from(round.left_proof.siblings.len() as u64));
            for sibling in &round.left_proof.siblings {
                push_felt252(data, *sibling);
            }

            // right_siblings: Array<felt252>
            data.push(FieldElement252::from(round.right_proof.siblings.len() as u64));
            for sibling in &round.right_proof.siblings {
                push_felt252(data, *sibling);
            }
        }
    }

    // final_value: QM31
    push_qm31(data, proof.final_value);
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
// Prover configuration
// ============================================================================

/// Configuration for the proving pipeline.
#[derive(Debug, Clone)]
pub struct ProverConfig {
    /// Run local verification after proving (catches bugs but doubles sumcheck time).
    /// Default: true (development). Set false for production throughput.
    pub verify_after_prove: bool,
    /// Number of spot-check queries for MLE opening proofs.
    pub num_queries: usize,
}

impl Default for ProverConfig {
    fn default() -> Self {
        Self {
            verify_after_prove: true,
            num_queries: DEFAULT_NUM_QUERIES,
        }
    }
}

impl ProverConfig {
    /// Production config: skip redundant local verification for maximum throughput.
    pub fn production() -> Self {
        Self {
            verify_after_prove: false,
            num_queries: DEFAULT_NUM_QUERIES,
        }
    }
}

// ============================================================================
// Proof generation
// ============================================================================

/// Generate a complete matmul proof with MLE commitments for on-chain verification.
///
/// Uses the default [`ProverConfig`] (with local verification enabled).
pub fn prove_matmul_for_starknet(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
) -> Result<StarknetMatMulProof, MatMulError> {
    prove_matmul_for_starknet_with_config(a, b, c, &ProverConfig::default())
}

/// Generate a complete matmul proof with custom prover configuration.
///
/// Pipeline stages (all A/B operations run in parallel):
/// 1. Commit matrices A, B via Poseidon Merkle trees
/// 2. Run sumcheck proving with Poseidon252Channel
/// 3. Derive the sumcheck assignment point
/// 4. Compute final MLE evaluations
/// 5. Generate MLE opening proofs
/// 6. Package into `StarknetMatMulProof`
#[instrument(skip_all, fields(size = %format!("{}x{}x{}", a.rows, a.cols, b.cols)))]
pub fn prove_matmul_for_starknet_with_config(
    a: &M31Matrix,
    b: &M31Matrix,
    c: &M31Matrix,
    config: &ProverConfig,
) -> Result<StarknetMatMulProof, MatMulError> {
    // Step 1: Commit to matrices (A and B tree builds run in parallel)
    let (a_padded, b_padded, a_tree, b_tree) = {
        let _span = info_span!("merkle_commit").entered();
        let (a_padded, b_padded) = rayon::join(
            || pad_matrix_for_mle(a),
            || pad_matrix_for_mle(b),
        );
        let (a_tree, b_tree) = rayon::join(
            || PoseidonMerkleTree::from_m31_values(&a_padded),
            || PoseidonMerkleTree::from_m31_values(&b_padded),
        );
        (a_padded, b_padded, a_tree, b_tree)
    };
    let a_commitment = a_tree.root();
    let b_commitment = b_tree.root();

    // Step 2: Run sumcheck with Poseidon252Channel
    let (proof, aux) = {
        let _span = info_span!("sumcheck_prove").entered();
        let mut prover_channel = Poseidon252Channel::default();
        prove_matmul(a, b, c, &mut prover_channel)?
    };

    // Optional local verification (skip in production for 2x sumcheck throughput)
    if config.verify_after_prove {
        let _span = info_span!("sumcheck_verify").entered();
        let mut verify_channel = Poseidon252Channel::default();
        verify_matmul(a, b, c, &proof, &aux, &mut verify_channel)?;
    }

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

    // Step 5: Compute final MLE evaluations at the full opening points (parallel A/B)
    let a_point = build_opening_point_a(&aux.row_challenges, &assignment);
    let b_point = build_opening_point_b(&assignment, &aux.col_challenges);

    let (final_a_eval, final_b_eval) = {
        let _span = info_span!("mle_eval").entered();
        rayon::join(
            || {
                let sf: Vec<SecureField> =
                    a_padded.iter().map(|v| SecureField::from(*v)).collect();
                eval_mle_at_point_pub(&sf, &a_point)
            },
            || {
                let sf: Vec<SecureField> =
                    b_padded.iter().map(|v| SecureField::from(*v)).collect();
                eval_mle_at_point_pub(&sf, &b_point)
            },
        )
    };

    // Step 6: Generate MLE opening proofs (parallel A/B)
    let (a_opening, b_opening) = {
        let _span = info_span!("mle_open").entered();
        rayon::join(
            || open_mle(&a_padded, &a_point, &a_tree, config.num_queries),
            || open_mle(&b_padded, &b_point, &b_tree, config.num_queries),
        )
    };

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

        // Now each felt252 is one element (not 4 limbs), so calldata is more compact.
        // Base: 4 dims + 4 claimed_sum + 1 round_polys_len + rounds*12 + 8 evals
        // + 1 a_commitment + 1 b_commitment + MLE openings
        let sumcheck_base = 4 + 4 + 1 + (proof.num_rounds as usize) * 12 + 8;
        assert!(calldata.len() > sumcheck_base + 2, "should include MLE openings");

        let f = |v: u64| FieldElement252::from(v);
        assert_eq!(calldata[0], f(2)); // m
        assert_eq!(calldata[1], f(2)); // k
        assert_eq!(calldata[2], f(2)); // n
        assert_eq!(calldata[3], f(1)); // num_rounds

        // Calldata is deterministic
        let calldata2 = prove_matmul_for_starknet(&a, &b, &c).unwrap().to_calldata();
        assert_eq!(calldata, calldata2);
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
    fn test_starknet_proof_16x16() {
        let a = M31Matrix::from_data(16, 16, (1..=256).map(M31::from).collect()).unwrap();
        let b = M31Matrix::from_data(16, 16, (257..=512).map(M31::from).collect()).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();

        assert_eq!(proof.m, 16);
        assert_eq!(proof.k, 16);
        assert_eq!(proof.n, 16);
        assert_eq!(proof.num_rounds, 4); // log2(16) = 4
        assert_eq!(proof.round_polys.len(), 4);
        assert_ne!(proof.a_commitment, proof.b_commitment);

        // Verify MLE openings locally
        let a_padded = pad_matrix_for_mle(&a);
        let a_tree = PoseidonMerkleTree::from_m31_values(&a_padded);
        assert_eq!(proof.a_commitment, a_tree.root());
    }

    #[test]
    fn test_starknet_proof_32x32() {
        let a = M31Matrix::from_data(
            32, 32,
            (0..1024).map(|i| M31::from((i * 7 + 3) % 1000)).collect(),
        ).unwrap();
        let b = M31Matrix::from_data(
            32, 32,
            (0..1024).map(|i| M31::from((i * 13 + 11) % 1000)).collect(),
        ).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();

        assert_eq!(proof.num_rounds, 5); // log2(32) = 5
        assert_eq!(proof.a_opening.queries.len(), DEFAULT_NUM_QUERIES);
        assert_eq!(proof.b_opening.queries.len(), DEFAULT_NUM_QUERIES);
        // Each query should have 10 rounds for A (5 row + 5 col = 10 vars)
        assert_eq!(
            proof.a_opening.queries[0].rounds.len(),
            10 // log2(32) + log2(32) = 10 variables
        );
    }

    #[test]
    fn test_starknet_proof_64x64() {
        let a = M31Matrix::from_data(
            64, 64,
            (0..4096).map(|i| M31::from((i * 7 + 3) % 2147483647)).collect(),
        ).unwrap();
        let b = M31Matrix::from_data(
            64, 64,
            (0..4096).map(|i| M31::from((i * 13 + 11) % 2147483647)).collect(),
        ).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();

        assert_eq!(proof.num_rounds, 6); // log2(64) = 6
        assert_eq!(proof.a_opening.intermediate_roots.len(), 11); // 12 vars - 1
    }

    #[test]
    fn test_starknet_proof_128x128() {
        let a = M31Matrix::from_data(
            128, 128,
            (0..16384).map(|i| M31::from((i * 7 + 3) % 2147483647)).collect(),
        ).unwrap();
        let b = M31Matrix::from_data(
            128, 128,
            (0..16384).map(|i| M31::from((i * 13 + 11) % 2147483647)).collect(),
        ).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();

        assert_eq!(proof.num_rounds, 7); // log2(128) = 7
        assert_eq!(proof.round_polys.len(), 7);
        assert_eq!(proof.a_opening.queries.len(), DEFAULT_NUM_QUERIES);
        assert_eq!(proof.b_opening.queries.len(), DEFAULT_NUM_QUERIES);
        // A has 14 variables (7 row + 7 col), so 13 intermediate roots
        assert_eq!(proof.a_opening.intermediate_roots.len(), 13);
        // Each query should have 14 rounds
        assert_eq!(proof.a_opening.queries[0].rounds.len(), 14);

        // Verify commitment roots are consistent
        let a_padded = pad_matrix_for_mle(&a);
        let a_tree = PoseidonMerkleTree::from_m31_values(&a_padded);
        assert_eq!(proof.a_commitment, a_tree.root());

        // Calldata should serialize
        let calldata = proof.to_calldata();
        assert!(calldata.len() > 100);
    }

    #[test]
    fn test_starknet_proof_256x256() {
        let a = M31Matrix::from_data(
            256, 256,
            (0..65536).map(|i| M31::from((i * 7 + 3) % 2147483647)).collect(),
        ).unwrap();
        let b = M31Matrix::from_data(
            256, 256,
            (0..65536).map(|i| M31::from((i * 13 + 11) % 2147483647)).collect(),
        ).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();

        assert_eq!(proof.num_rounds, 8); // log2(256) = 8
        assert_eq!(proof.round_polys.len(), 8);
        assert_eq!(proof.a_opening.queries.len(), DEFAULT_NUM_QUERIES);
        assert_eq!(proof.b_opening.queries.len(), DEFAULT_NUM_QUERIES);
        // A has 16 variables (8 row + 8 col), so 15 intermediate roots
        assert_eq!(proof.a_opening.intermediate_roots.len(), 15);

        // Verify calldata size is reasonable
        let calldata = proof.to_calldata();
        assert!(calldata.len() > 200);
    }

    #[test]
    fn test_calldata_full_serialization() {
        // Verify calldata includes MLE opening proofs
        let a = M31Matrix::from_data(
            2, 2,
            vec![M31::from(1), M31::from(2), M31::from(3), M31::from(4)],
        ).unwrap();
        let b = M31Matrix::from_data(
            2, 2,
            vec![M31::from(5), M31::from(6), M31::from(7), M31::from(8)],
        ).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();
        let calldata = proof.to_calldata();

        // Base: 4 dims + 4 claimed_sum + 1 len + rounds*12 + 8 evals + 2 commitments + openings
        assert!(calldata.len() > 29, "calldata should include MLE openings, got len={}", calldata.len());

        // First elements are still m, k, n, num_rounds
        let f = |v: u64| FieldElement252::from(v);
        assert_eq!(calldata[0], f(2)); // m
        assert_eq!(calldata[1], f(2)); // k
        assert_eq!(calldata[2], f(2)); // n
        assert_eq!(calldata[3], f(1)); // num_rounds

        // Calldata should be deterministic
        let calldata2 = proof.to_calldata();
        assert_eq!(calldata, calldata2);
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

    #[test]
    fn test_production_config_matches_default() {
        let a = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
        let b = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let default_proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();
        let prod_proof =
            prove_matmul_for_starknet_with_config(&a, &b, &c, &ProverConfig::production())
                .unwrap();

        // Same matrices → same proof (verify_after_prove doesn't affect output)
        assert_eq!(default_proof.a_commitment, prod_proof.a_commitment);
        assert_eq!(default_proof.b_commitment, prod_proof.b_commitment);
        assert_eq!(default_proof.final_a_eval, prod_proof.final_a_eval);
        assert_eq!(default_proof.final_b_eval, prod_proof.final_b_eval);
        assert_eq!(default_proof.claimed_sum, prod_proof.claimed_sum);
        assert_eq!(
            default_proof.to_calldata(),
            prod_proof.to_calldata()
        );
    }

    #[test]
    fn test_production_config_64x64() {
        let a = M31Matrix::from_data(
            64, 64,
            (0..4096).map(|i| M31::from((i * 7 + 3) % 2147483647)).collect(),
        ).unwrap();
        let b = M31Matrix::from_data(
            64, 64,
            (0..4096).map(|i| M31::from((i * 13 + 11) % 2147483647)).collect(),
        ).unwrap();
        let c = M31Matrix::multiply(&a, &b).unwrap();

        let proof =
            prove_matmul_for_starknet_with_config(&a, &b, &c, &ProverConfig::production())
                .unwrap();

        assert_eq!(proof.num_rounds, 6);
        assert_ne!(proof.a_commitment, FieldElement252::ZERO);

        // Verify the proof is still valid by checking MLE openings
        let a_padded = pad_matrix_for_mle(&a);
        let a_tree = PoseidonMerkleTree::from_m31_values(&a_padded);
        assert_eq!(proof.a_commitment, a_tree.root());
    }
}
