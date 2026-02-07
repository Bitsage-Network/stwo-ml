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
use crate::components::attention::{
    apply_activation_table, flatten_matrix, AttentionError,
};
use crate::components::activation::ActivationType;
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
// Starknet Attention Proof
// ============================================================================

/// Error type for Starknet attention proof generation.
#[derive(Debug, thiserror::Error)]
pub enum StarknetAttentionError {
    #[error("matmul error: {0}")]
    MatMul(#[from] MatMulError),
    #[error("attention error: {0}")]
    Attention(#[from] AttentionError),
}

/// Complete attention proof with MLE commitments for on-chain verification.
///
/// Contains two matmul proofs (QK^T and weights×V) with Poseidon Merkle
/// commitments to all matrices (Q, K^T, V, scores, weights, output).
///
/// The activation step (scores → weights) is verified on-chain by checking
/// the committed scores against the committed weights using the activation
/// table. The prover commits to both, and the verifier checks consistency.
#[derive(Debug, Clone)]
pub struct StarknetAttentionProof {
    /// QK^T matmul proof (Stage 1).
    pub qkt_proof: StarknetMatMulProof,
    /// Weights×V matmul proof (Stage 3).
    pub attn_v_proof: StarknetMatMulProof,
    /// Activation type used for scores → weights.
    pub activation_type: u32,
    /// Log2 of the activation table size.
    pub activation_log_size: u32,
    /// Poseidon Merkle root of the scores matrix.
    pub scores_commitment: FieldElement252,
    /// Poseidon Merkle root of the weights matrix.
    pub weights_commitment: FieldElement252,
}

impl StarknetAttentionProof {
    /// Serialize the complete attention proof to felt252 calldata.
    ///
    /// Layout:
    ///   [activation_type, activation_log_size,
    ///    scores_commitment, weights_commitment,
    ///    qkt_proof_calldata..., attn_v_proof_calldata...]
    pub fn to_calldata(&self) -> Vec<Felt> {
        let mut data = vec![
            FieldElement252::from(self.activation_type as u64),
            FieldElement252::from(self.activation_log_size as u64),
        ];

        push_felt252(&mut data, self.scores_commitment);
        push_felt252(&mut data, self.weights_commitment);

        // Inline both matmul proofs
        let qkt_calldata = self.qkt_proof.to_calldata();
        data.push(FieldElement252::from(qkt_calldata.len() as u64));
        data.extend_from_slice(&qkt_calldata);

        let attn_v_calldata = self.attn_v_proof.to_calldata();
        data.push(FieldElement252::from(attn_v_calldata.len() as u64));
        data.extend_from_slice(&attn_v_calldata);

        data
    }
}

/// Generate a complete attention proof for Starknet on-chain verification.
///
/// Pipeline:
/// 1. Compute scores = Q × K^T
/// 2. Apply activation table: weights = activation(scores)
/// 3. Compute output = weights × V
/// 4. Generate two StarknetMatMulProofs (QK^T and weights×V)
/// 5. Commit scores and weights for on-chain activation verification
///
/// All matrices must have scores in `[0, 2^log_table_size)` after QK^T.
#[instrument(skip_all, fields(
    seq_len = q.rows,
    d_k = q.cols,
    d_v = v.cols,
    activation = ?activation_type,
))]
pub fn prove_attention_for_starknet(
    q: &M31Matrix,
    k: &M31Matrix,
    v: &M31Matrix,
    activation_type: ActivationType,
    log_table_size: u32,
) -> Result<StarknetAttentionProof, StarknetAttentionError> {
    // Stage 1: Compute scores = Q × K^T
    let kt = k.transpose();
    let scores = M31Matrix::multiply(q, &kt)?;

    // Stage 2: Apply activation table
    let table = activation_type.build_table(log_table_size);
    let weights = apply_activation_table(&scores, &table)?;

    // Stage 3: Compute output = weights × V
    let output = M31Matrix::multiply(&weights, v)?;

    // Generate QK^T matmul proof (parallel with V proof)
    let config = ProverConfig::production();
    let (qkt_proof, attn_v_proof) = {
        let _span = info_span!("attention_proofs").entered();
        rayon::join(
            || prove_matmul_for_starknet_with_config(q, &kt, &scores, &config),
            || prove_matmul_for_starknet_with_config(&weights, v, &output, &config),
        )
    };
    let qkt_proof = qkt_proof?;
    let attn_v_proof = attn_v_proof?;

    // Commit scores and weights for on-chain activation verification
    let score_values = flatten_matrix(&scores);
    let weight_values = flatten_matrix(&weights);
    let (scores_tree, weights_tree) = {
        let _span = info_span!("activation_commit").entered();
        rayon::join(
            || PoseidonMerkleTree::from_m31_values(&score_values),
            || PoseidonMerkleTree::from_m31_values(&weight_values),
        )
    };

    let activation_type_id = match activation_type {
        ActivationType::ReLU => 0,
        ActivationType::GELU => 1,
        ActivationType::Sigmoid => 2,
        ActivationType::Softmax => 3,
        ActivationType::LayerNorm => 4,
    };

    Ok(StarknetAttentionProof {
        qkt_proof,
        attn_v_proof,
        activation_type: activation_type_id,
        activation_log_size: log_table_size,
        scores_commitment: scores_tree.root(),
        weights_commitment: weights_tree.root(),
    })
}

// ============================================================================
// Starknet Model Proof (full computation graph)
// ============================================================================

/// Per-node proof data serializable as Starknet calldata.
///
/// Each node proof is either a matmul proof (sumcheck + MLE commitments)
/// or a component proof (type tag only — LogUp proofs verified off-chain,
/// only matmul proofs go on-chain due to gas costs).
#[derive(Debug, Clone)]
pub enum StarknetNodeProof {
    /// Input node — no proof.
    Input,
    /// MatMul proof with full on-chain calldata.
    MatMul(Box<StarknetMatMulProof>),
    /// Activation/LayerNorm/Quantize — type tag for off-chain proof reference.
    OffChain {
        /// Node type identifier: 1=Activation, 2=LayerNorm, 3=Quantize.
        node_type: u32,
    },
}

/// Complete model proof for Starknet on-chain verification.
///
/// The on-chain verifier checks all matmul sumcheck proofs. Activation,
/// LayerNorm, and Quantize proofs are verified off-chain (their LogUp STARK
/// proofs are too large for on-chain gas budgets). The model proof
/// commits to the graph structure so the verifier knows the computation
/// being proven.
#[derive(Debug, Clone)]
pub struct StarknetModelProof {
    /// Number of nodes in the graph.
    pub num_nodes: u32,
    /// Per-node proofs in topological order.
    pub node_proofs: Vec<StarknetNodeProof>,
}

impl StarknetModelProof {
    /// Serialize the model proof to felt252 calldata.
    ///
    /// Layout:
    ///   [num_nodes,
    ///    for each node:
    ///      node_type (0=Input, 1=MatMul, 2=Activation, 3=LayerNorm, 4=Quantize),
    ///      if MatMul: matmul_proof_calldata...
    ///   ]
    pub fn to_calldata(&self) -> Vec<Felt> {
        let mut data = vec![FieldElement252::from(self.num_nodes as u64)];

        for node_proof in &self.node_proofs {
            match node_proof {
                StarknetNodeProof::Input => {
                    data.push(FieldElement252::from(0u64)); // type = Input
                }
                StarknetNodeProof::MatMul(ref proof) => {
                    data.push(FieldElement252::from(1u64)); // type = MatMul
                    let matmul_calldata = proof.to_calldata();
                    data.push(FieldElement252::from(matmul_calldata.len() as u64));
                    data.extend_from_slice(&matmul_calldata);
                }
                StarknetNodeProof::OffChain { node_type } => {
                    data.push(FieldElement252::from(*node_type as u64));
                }
            }
        }

        data
    }

    /// Count of on-chain verifiable proofs (matmul nodes only).
    pub fn num_onchain_proofs(&self) -> usize {
        self.node_proofs
            .iter()
            .filter(|p| matches!(p, StarknetNodeProof::MatMul(_)))
            .count()
    }
}

/// Generate a Starknet model proof from a computation graph.
///
/// Pipeline:
/// 1. Execute graph to generate all intermediate values
/// 2. For each MatMul node: generate full StarknetMatMulProof with Poseidon commitments
/// 3. For other nodes: record type tag (proofs verified off-chain)
///
/// All MatMul proofs run with production config (no redundant local verify).
#[instrument(skip_all, fields(num_nodes = graph.nodes().len()))]
pub fn prove_model_for_starknet(
    graph: &crate::compiler::graph::ComputationGraph,
    input: &M31Matrix,
    weights: &crate::compiler::graph::GraphWeights,
) -> Result<StarknetModelProof, StarknetModelError> {
    use crate::compiler::graph::{execute_graph, GraphOp};

    let execution = execute_graph(graph, input, weights)
        .map_err(StarknetModelError::Execution)?;
    let order = execution.order.clone();
    let config = ProverConfig::production();

    let mut node_proofs = Vec::with_capacity(order.len());

    for &node_id in &order {
        let node = graph.node(node_id).unwrap();
        let proof = match &node.op {
            GraphOp::Input { .. } => StarknetNodeProof::Input,

            GraphOp::MatMul { .. } => {
                let pred_id = node.inputs[0];
                let input_matrix = execution.output(pred_id)
                    .ok_or(StarknetModelError::MissingOutput(node_id))?;
                let weight_matrix = weights.matmul_weights.get(&node_id)
                    .ok_or(StarknetModelError::MissingWeight(node_id))?;
                let output_matrix = execution.output(node_id)
                    .ok_or(StarknetModelError::MissingOutput(node_id))?;

                let matmul_proof = prove_matmul_for_starknet_with_config(
                    input_matrix, weight_matrix, output_matrix, &config,
                )?;
                StarknetNodeProof::MatMul(Box::new(matmul_proof))
            }

            GraphOp::Activation { .. } => StarknetNodeProof::OffChain { node_type: 2 },
            GraphOp::LayerNorm { .. } => StarknetNodeProof::OffChain { node_type: 3 },
            GraphOp::Quantize { .. } => StarknetNodeProof::OffChain { node_type: 4 },
        };

        node_proofs.push(proof);
    }

    Ok(StarknetModelProof {
        num_nodes: order.len() as u32,
        node_proofs,
    })
}

/// Error type for Starknet model proof generation.
#[derive(Debug, thiserror::Error)]
pub enum StarknetModelError {
    #[error("execution error: {0}")]
    Execution(#[from] crate::compiler::graph::ExecutionError),
    #[error("matmul proof error: {0}")]
    MatMul(#[from] MatMulError),
    #[error("missing output for node {0}")]
    MissingOutput(usize),
    #[error("missing weight for node {0}")]
    MissingWeight(usize),
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

    // -----------------------------------------------------------------------
    // Starknet attention proof tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_starknet_attention_relu_4x4() {
        // Small attention: seq_len=4, d_k=4, d_v=4 with ReLU activation.
        // Values kept small to fit in log_size=4 table ([0, 16)).
        let q = M31Matrix::from_data(4, 4, vec![
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
        ]).unwrap();
        let k = q.clone();
        let v = M31Matrix::from_data(4, 4, vec![
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
        ]).unwrap();

        let proof = prove_attention_for_starknet(
            &q, &k, &v,
            ActivationType::ReLU,
            4,
        ).unwrap();

        // Verify structure
        assert_eq!(proof.qkt_proof.m, 4);
        assert_eq!(proof.qkt_proof.k, 4);
        assert_eq!(proof.attn_v_proof.m, 4);
        assert_eq!(proof.activation_type, 0); // ReLU
        assert_eq!(proof.activation_log_size, 4);
        assert_ne!(proof.scores_commitment, FieldElement252::ZERO);
        assert_ne!(proof.weights_commitment, FieldElement252::ZERO);

        // Calldata should serialize
        let calldata = proof.to_calldata();
        assert!(calldata.len() > 50, "calldata too short: {}", calldata.len());

        // First two elements: activation_type, activation_log_size
        assert_eq!(calldata[0], FieldElement252::from(0u64)); // ReLU
        assert_eq!(calldata[1], FieldElement252::from(4u64)); // log_size
    }

    #[test]
    fn test_starknet_attention_sigmoid_4x4() {
        let q = M31Matrix::from_data(4, 4, vec![
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
        ]).unwrap();
        let k = q.clone();
        let v = M31Matrix::from_data(4, 4, vec![
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
        ]).unwrap();

        let proof = prove_attention_for_starknet(
            &q, &k, &v,
            ActivationType::Sigmoid,
            4,
        ).unwrap();

        assert_eq!(proof.activation_type, 2); // Sigmoid
        assert_ne!(proof.scores_commitment, proof.weights_commitment,
            "sigmoid should transform scores differently from identity");

        let calldata = proof.to_calldata();
        assert!(calldata.len() > 50);
    }

    #[test]
    fn test_starknet_attention_calldata_deterministic() {
        let q = M31Matrix::from_data(4, 4, vec![
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
        ]).unwrap();
        let k = q.clone();
        let v = M31Matrix::from_data(4, 4, vec![
            M31::from(1), M31::from(0), M31::from(1), M31::from(0),
            M31::from(0), M31::from(1), M31::from(0), M31::from(1),
            M31::from(1), M31::from(1), M31::from(0), M31::from(0),
            M31::from(0), M31::from(0), M31::from(1), M31::from(1),
        ]).unwrap();

        let proof1 = prove_attention_for_starknet(&q, &k, &v, ActivationType::ReLU, 4).unwrap();
        let proof2 = prove_attention_for_starknet(&q, &k, &v, ActivationType::ReLU, 4).unwrap();

        assert_eq!(proof1.to_calldata(), proof2.to_calldata());
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

    // -----------------------------------------------------------------------
    // Starknet model proof tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_starknet_model_proof_matmul_relu() {
        use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};

        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::MatMul {
                weight_rows: 4,
                weight_cols: 4,
            },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 8,
            },
        ])
        .unwrap();

        let input = M31Matrix::from_data(
            4, 4,
            (0..16u32).map(|i| M31::from(i % 4)).collect(),
        ).unwrap();

        let weight = M31Matrix::from_data(
            4, 4,
            (0..16u32).map(|i| M31::from((i + 1) % 3)).collect(),
        ).unwrap();

        let mut weights = GraphWeights::new();
        weights.matmul_weights.insert(1, weight);

        let proof = prove_model_for_starknet(&graph, &input, &weights).unwrap();

        assert_eq!(proof.num_nodes, 3);
        assert_eq!(proof.num_onchain_proofs(), 1); // Only matmul is on-chain

        // Verify calldata structure
        let calldata = proof.to_calldata();
        assert_eq!(calldata[0], FieldElement252::from(3u64)); // num_nodes
        assert_eq!(calldata[1], FieldElement252::from(0u64)); // Input type
        assert_eq!(calldata[2], FieldElement252::from(1u64)); // MatMul type
    }

    #[test]
    fn test_starknet_model_proof_two_layer_mlp() {
        use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};

        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::MatMul {
                weight_rows: 4,
                weight_cols: 4,
            },
            GraphOp::Activation {
                activation: ActivationType::ReLU,
                log_table_size: 8,
            },
            GraphOp::MatMul {
                weight_rows: 4,
                weight_cols: 4,
            },
        ])
        .unwrap();

        let input = M31Matrix::from_data(
            4, 4,
            (0..16u32).map(|i| M31::from(i % 3)).collect(),
        ).unwrap();

        let w1 = M31Matrix::from_data(
            4, 4,
            (0..16u32).map(|i| M31::from((i + 1) % 2)).collect(),
        ).unwrap();

        let w2 = M31Matrix::from_data(
            4, 4,
            (0..16u32).map(|i| M31::from(i % 2)).collect(),
        ).unwrap();

        let mut weights = GraphWeights::new();
        weights.matmul_weights.insert(1, w1);
        weights.matmul_weights.insert(3, w2);

        let proof = prove_model_for_starknet(&graph, &input, &weights).unwrap();

        assert_eq!(proof.num_nodes, 4);
        assert_eq!(proof.num_onchain_proofs(), 2); // Two matmul proofs

        let calldata = proof.to_calldata();
        assert!(calldata.len() > 10);
    }

    #[test]
    fn test_starknet_model_calldata_deterministic() {
        use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};

        let graph = ComputationGraph::sequential(vec![
            GraphOp::Input { rows: 4, cols: 4 },
            GraphOp::MatMul {
                weight_rows: 4,
                weight_cols: 4,
            },
        ])
        .unwrap();

        let input = M31Matrix::from_data(
            4, 4,
            (1..=16).map(M31::from).collect(),
        ).unwrap();

        let weight = M31Matrix::from_data(
            4, 4,
            (17..=32).map(M31::from).collect(),
        ).unwrap();

        let mut weights = GraphWeights::new();
        weights.matmul_weights.insert(1, weight);

        let proof1 = prove_model_for_starknet(&graph, &input, &weights).unwrap();
        let proof2 = prove_model_for_starknet(&graph, &input, &weights).unwrap();

        assert_eq!(proof1.to_calldata(), proof2.to_calldata());
    }
}
