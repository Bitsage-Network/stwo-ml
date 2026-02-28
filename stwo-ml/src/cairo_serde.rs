//! Cairo serialization bridge: Rust `StarkProof` → `Vec<felt252>`.
//!
//! Converts STWO's Rust `StarkProof<Blake2sMerkleHasher>` into a flat
//! `Vec<FieldElement>` that the stwo-cairo-verifier can deserialize via
//! Cairo's `Serde::deserialize(ref serialized: Span<felt252>)`.
//!
//! # Serialization Order
//!
//! The output matches Cairo's auto-derived `Serde` deserialization order:
//!
//! ```text
//! StarkProof → CommitmentSchemeProof
//! ├── config.pow_bits                          [1 felt]
//! ├── config.fri_config.log_blowup_factor      [1 felt]
//! ├── config.fri_config.log_last_layer_deg     [1 felt]
//! ├── config.fri_config.n_queries              [1 felt]
//! ├── commitments (per-tree array of Blake2sHash)
//! ├── sampled_values (nested TreeVec<ColumnVec<Vec<QM31>>>)
//! ├── decommitments (TreeVec<MerkleDecommitment>)
//! ├── queried_values (TreeVec<ColumnVec<Vec<M31>>>)
//! ├── proof_of_work_nonce                      [1 felt]
//! └── fri_proof
//! ```
//!
//! # Type Mapping
//!
//! | Rust Type | Cairo Type | felt252 count |
//! |-----------|-----------|---------------|
//! | `M31` | `M31` | 1 |
//! | `QM31` | `QM31` | 4 |
//! | `Blake2sHash` | `Blake2sHash` | 8 (8 × u32) |
//! | `u32` | `u32` | 1 |
//! | `u64` | `u64` | 1 |
//! | `Vec<T>` | `Span<T>` | 1 (length) + N × size(T) |

use starknet_ff::FieldElement;

use stwo::core::channel::MerkleChannel;
use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::core::fri::{FriConfig, FriLayerProof, FriProof};
use stwo::core::pcs::quotients::CommitmentSchemeProof;
use stwo::core::pcs::PcsConfig;
use stwo::core::pcs::TreeVec;
use stwo::core::poly::line::LinePoly;
use stwo::core::proof::StarkProof;
use stwo::core::vcs::blake2_hash::Blake2sHash;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo::core::vcs_lifted::verifier::MerkleDecommitmentLifted;
use stwo::core::ColumnVec;

/// The lifted Merkle hasher type for Blake2s channel.
type Hasher = <Blake2sMerkleChannel as MerkleChannel>::H;

/// Concrete proof type for Blake2s channel.
pub type Blake2sProof = StarkProof<Hasher>;

/// Error type for Cairo serialization.
#[derive(Debug, thiserror::Error)]
pub enum CairoSerdeError {
    #[error("Empty proof")]
    EmptyProof,
    #[error("Serialization overflow: {0}")]
    Overflow(String),
}

/// Serialize a complete STARK proof into a flat `Vec<FieldElement>` (felt252 array)
/// matching the stwo-cairo-verifier's expected deserialization format.
///
/// The output can be passed directly to the Cairo verifier's
/// `Serde::<StarkProof>::deserialize(ref serialized)`.
pub fn serialize_proof(proof: &Blake2sProof) -> Vec<FieldElement> {
    let mut output = Vec::new();
    serialize_commitment_scheme_proof(&proof.0, &mut output);
    output
}

/// Returns the estimated number of felt252 elements for the serialized proof.
pub fn estimate_calldata_size(proof: &Blake2sProof) -> usize {
    serialize_proof(proof).len()
}

// === Primitive serialization ===

pub(crate) fn serialize_u32(val: u32, output: &mut Vec<FieldElement>) {
    output.push(FieldElement::from(val as u64));
}

fn serialize_u64(val: u64, output: &mut Vec<FieldElement>) {
    output.push(FieldElement::from(val));
}

fn serialize_usize(val: usize, output: &mut Vec<FieldElement>) {
    output.push(FieldElement::from(val as u64));
}

fn serialize_m31(val: M31, output: &mut Vec<FieldElement>) {
    output.push(FieldElement::from(val.0 as u64));
}

/// QM31 = ((a + bi) + (c + di)u) → [a, b, c, d] as 4 felt252.
pub(crate) fn serialize_qm31(val: SecureField, output: &mut Vec<FieldElement>) {
    // QM31(CM31(a, b), CM31(c, d))
    serialize_m31(val.0 .0, output); // a
    serialize_m31(val.0 .1, output); // b
    serialize_m31(val.1 .0, output); // c
    serialize_m31(val.1 .1, output); // d
}

/// QM31 packed into 1 felt252 (124 bits) via sentinel-prefix packing.
/// Layout: [1-bit sentinel | 31-bit a | 31-bit b | 31-bit c | 31-bit d].
pub(crate) fn serialize_qm31_packed(val: SecureField, output: &mut Vec<FieldElement>) {
    output.push(crate::crypto::poseidon_channel::securefield_to_felt(val));
}

/// Pack two QM31 values into a single felt252 (248 bits, no sentinel).
/// Layout: [a.a.a(31) | a.a.b(31) | a.b.a(31) | a.b.b(31) | b.a.a(31) | b.a.b(31) | b.b.a(31) | b.b.b(31)]
/// Uses the same Horner-style packing as Cairo's pack_qm31_pair_to_felt.
pub(crate) fn serialize_qm31_pair_packed(a: SecureField, b: SecureField, output: &mut Vec<FieldElement>) {
    use stwo::core::fields::qm31::QM31;
    use stwo::core::fields::cm31::CM31;
    let QM31(CM31(a0, a1), CM31(a2, a3)) = a;
    let QM31(CM31(b0, b1), CM31(b2, b3)) = b;
    let shift = FieldElement::from(1u64 << 31); // 2^31
    // Horner: result = a0 * 2^217 + a1 * 2^186 + ... + b3
    let mut result = FieldElement::from(a0.0 as u64);
    result = result * shift + FieldElement::from(a1.0 as u64);
    result = result * shift + FieldElement::from(a2.0 as u64);
    result = result * shift + FieldElement::from(a3.0 as u64);
    result = result * shift + FieldElement::from(b0.0 as u64);
    result = result * shift + FieldElement::from(b1.0 as u64);
    result = result * shift + FieldElement::from(b2.0 as u64);
    result = result * shift + FieldElement::from(b3.0 as u64);
    output.push(result);
}

/// Unpack two QM31 values from a single double-packed felt252.
pub(crate) fn deserialize_qm31_pair_packed(fe: FieldElement) -> (SecureField, SecureField) {
    use stwo::core::fields::qm31::QM31;
    use stwo::core::fields::cm31::CM31;
    let m31s = crate::crypto::poseidon_channel::unpack_m31s(fe, 8);
    let a = QM31(CM31(m31s[0], m31s[1]), CM31(m31s[2], m31s[3]));
    let b = QM31(CM31(m31s[4], m31s[5]), CM31(m31s[6], m31s[7]));
    (a, b)
}

/// Blake2sHash([u8; 32]) → 8 × u32 words → 8 felt252.
///
/// Converts the 32-byte hash into 8 little-endian u32 words,
/// matching Cairo's `Blake2sHash { hash: Box<[u32; 8]> }`.
fn serialize_blake2s_hash(hash: &Blake2sHash, output: &mut Vec<FieldElement>) {
    for chunk in hash.0.chunks_exact(4) {
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        output.push(FieldElement::from(word as u64));
    }
}

// === Span/Array serialization ===

/// Serialize a `Vec<T>` as a Cairo `Span<T>`: length prefix + elements.
fn serialize_span<T>(
    items: &[T],
    serialize_item: impl Fn(&T, &mut Vec<FieldElement>),
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(items.len() as u32, output);
    for item in items {
        serialize_item(item, output);
    }
}

// === Config serialization ===

fn serialize_fri_config(config: &FriConfig, output: &mut Vec<FieldElement>) {
    serialize_u32(config.log_blowup_factor, output);
    serialize_u32(config.log_last_layer_degree_bound, output);
    serialize_usize(config.n_queries, output);
}

fn serialize_pcs_config(config: &PcsConfig, output: &mut Vec<FieldElement>) {
    serialize_u32(config.pow_bits, output);
    serialize_fri_config(&config.fri_config, output);
}

// === TreeVec serialization ===
//
// Cairo uses `TreeArray<T>` / `TreeSpan<T>` which are arrays-of-arrays.
// Serialized as: length (num trees) + per-tree serialization.

/// Serialize `TreeVec<Blake2sHash>` (commitments) as nested Span.
fn serialize_tree_hashes(tree: &TreeVec<Blake2sHash>, output: &mut Vec<FieldElement>) {
    serialize_u32(tree.0.len() as u32, output);
    for hash in &tree.0 {
        serialize_blake2s_hash(hash, output);
    }
}

/// Serialize `TreeVec<ColumnVec<Vec<SecureField>>>` (sampled values).
///
/// Cairo layout: num_trees, then per tree: num_columns, then per column: num_values, values.
fn serialize_sampled_values(
    tree: &TreeVec<ColumnVec<Vec<SecureField>>>,
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(tree.0.len() as u32, output);
    for columns in &tree.0 {
        serialize_u32(columns.len() as u32, output);
        for values in columns {
            serialize_span(values, |v, o| serialize_qm31(*v, o), output);
        }
    }
}

/// Serialize `TreeVec<MerkleDecommitmentLifted<Hasher>>` (decommitments).
///
/// Each decommitment has:
/// - hash_witness: Span<Blake2sHash>
/// - column_witness: Span<M31> (empty in Rust, required by Cairo)
fn serialize_decommitments(
    tree: &TreeVec<MerkleDecommitmentLifted<Hasher>>,
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(tree.0.len() as u32, output);
    for decommitment in &tree.0 {
        serialize_merkle_decommitment(decommitment, output);
    }
}

fn serialize_merkle_decommitment(
    decommitment: &MerkleDecommitmentLifted<Hasher>,
    output: &mut Vec<FieldElement>,
) {
    // hash_witness: Span<Blake2sHash>
    serialize_span(&decommitment.hash_witness, serialize_blake2s_hash, output);
    // column_witness: Span<M31> — empty in Rust's MerkleDecommitmentLifted,
    // but Cairo's MerkleDecommitment expects it. Serialize as empty span.
    serialize_u32(0, output); // length = 0
}

/// Serialize `TreeVec<ColumnVec<Vec<BaseField>>>` (queried values).
fn serialize_queried_values(
    tree: &TreeVec<ColumnVec<Vec<BaseField>>>,
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(tree.0.len() as u32, output);
    for columns in &tree.0 {
        serialize_u32(columns.len() as u32, output);
        for values in columns {
            serialize_span(values, |v, o| serialize_m31(*v, o), output);
        }
    }
}

// === FRI proof serialization ===

fn serialize_fri_layer_proof(layer: &FriLayerProof<Hasher>, output: &mut Vec<FieldElement>) {
    // fri_witness: Span<QM31>
    serialize_span(&layer.fri_witness, |v, o| serialize_qm31(*v, o), output);
    // decommitment: MerkleDecommitment
    serialize_merkle_decommitment(&layer.decommitment, output);
    // commitment: Blake2sHash
    serialize_blake2s_hash(&layer.commitment, output);
}

/// Serialize `LinePoly` as Cairo's `LinePoly { coeffs: Array<QM31>, log_size: u32 }`.
///
/// Coefficients are serialized in bit-reversed order (internal storage format).
fn serialize_line_poly(poly: &LinePoly, output: &mut Vec<FieldElement>) {
    let coeffs: &[SecureField] = poly; // LinePoly derefs to [SecureField]
    serialize_span(coeffs, |v, o| serialize_qm31(*v, o), output);
    serialize_u32(poly.len().ilog2(), output);
}

fn serialize_fri_proof(fri: &FriProof<Hasher>, output: &mut Vec<FieldElement>) {
    // first_layer: FriLayerProof
    serialize_fri_layer_proof(&fri.first_layer, output);
    // inner_layers: Span<FriLayerProof>
    serialize_span(&fri.inner_layers, serialize_fri_layer_proof, output);
    // last_layer_poly: LinePoly
    serialize_line_poly(&fri.last_layer_poly, output);
}

// === Top-level proof serialization ===

/// Serialize the full `CommitmentSchemeProof`.
///
/// Field order matches Cairo's `#[derive(Serde)]` on `CommitmentSchemeProof`:
/// config, commitments, sampled_values, decommitments, queried_values,
/// proof_of_work_nonce, fri_proof.
fn serialize_commitment_scheme_proof(
    proof: &CommitmentSchemeProof<Hasher>,
    output: &mut Vec<FieldElement>,
) {
    // 1. config: PcsConfig
    serialize_pcs_config(&proof.config, output);

    // 2. commitments: TreeVec<Blake2sHash>
    serialize_tree_hashes(&proof.commitments, output);

    // 3. sampled_values: TreeVec<ColumnVec<Vec<SecureField>>>
    serialize_sampled_values(&proof.sampled_values, output);

    // 4. decommitments: TreeVec<MerkleDecommitmentLifted<Hasher>>
    serialize_decommitments(&proof.decommitments, output);

    // 5. queried_values: TreeVec<ColumnVec<Vec<BaseField>>>
    serialize_queried_values(&proof.queried_values, output);

    // 6. proof_of_work_nonce: u64
    // Note: Rust field is `proof_of_work`, Cairo field is `proof_of_work_nonce`
    serialize_u64(proof.proof_of_work, output);

    // 7. fri_proof: FriProof
    serialize_fri_proof(&proof.fri_proof, output);
}

// === MatMul Sumcheck + MLE Proof Serialization ===

use crate::components::matmul::{CompressedRoundPoly, MatMulSumcheckProofOnChain, RoundPoly};
use crate::crypto::mle_opening::MleOpeningProof;

/// Serialize a `RoundPoly` (3 QM31 values = 12 felt252s).
fn serialize_round_poly(rp: &RoundPoly, output: &mut Vec<FieldElement>) {
    serialize_qm31(rp.c0, output);
    serialize_qm31(rp.c1, output);
    serialize_qm31(rp.c2, output);
}

/// Serialize a compressed `RoundPoly` (2 QM31 values = 8 felt252s).
///
/// Omits c1 — the verifier reconstructs it via:
/// `c1 = current_sum - 2*c0 - c2`
///
/// Saves 4 felts (33%) per sumcheck round.
pub fn serialize_round_poly_compressed(rp: &CompressedRoundPoly, output: &mut Vec<FieldElement>) {
    serialize_qm31(rp.c0, output);
    serialize_qm31(rp.c2, output);
}

/// Serialize an `MleOpeningProof` matching Cairo's `#[derive(Serde)]` layout.
pub fn serialize_mle_opening_proof(proof: &MleOpeningProof, output: &mut Vec<FieldElement>) {
    // intermediate_roots: Array<felt252>
    serialize_u32(proof.intermediate_roots.len() as u32, output);
    for root in &proof.intermediate_roots {
        output.push(*root);
    }
    // queries: Array<MleQueryProof>
    serialize_u32(proof.queries.len() as u32, output);
    for query in &proof.queries {
        serialize_u32(query.initial_pair_index, output);
        serialize_u32(query.rounds.len() as u32, output);
        for round in &query.rounds {
            serialize_qm31(round.left_value, output);
            serialize_qm31(round.right_value, output);
            // left_siblings: Span<felt252>
            serialize_u32(round.left_siblings.len() as u32, output);
            for s in &round.left_siblings {
                output.push(*s);
            }
            // right_siblings: Span<felt252>
            serialize_u32(round.right_siblings.len() as u32, output);
            for s in &round.right_siblings {
                output.push(*s);
            }
        }
    }
    // final_value: QM31
    serialize_qm31(proof.final_value, output);
}

/// Serialize a `MatMulSumcheckProofOnChain` matching the Cairo verifier's
/// 12-field `MatMulSumcheckProof` layout.
///
/// Field order:
/// 1. m (u32)
/// 2. k (u32)
/// 3. n (u32)
/// 4. num_rounds (u32)
/// 5. claimed_sum (QM31 = 4 felts)
/// 6. round_polys (Array<RoundPoly>: length + n * 12 felts)
/// 7. final_a_eval (QM31 = 4 felts)
/// 8. final_b_eval (QM31 = 4 felts)
/// 9. a_commitment (felt252)
/// 10. b_commitment (felt252)
/// 11. a_opening (MleOpeningProof)
/// 12. b_opening (MleOpeningProof)
pub fn serialize_matmul_sumcheck_proof(
    proof: &MatMulSumcheckProofOnChain,
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(proof.m, output);
    serialize_u32(proof.k, output);
    serialize_u32(proof.n, output);
    serialize_u32(proof.num_rounds, output);
    serialize_qm31(proof.claimed_sum, output);
    // round_polys: Array<RoundPoly>
    serialize_u32(proof.round_polys.len() as u32, output);
    for rp in &proof.round_polys {
        serialize_round_poly(rp, output);
    }
    serialize_qm31(proof.final_a_eval, output);
    serialize_qm31(proof.final_b_eval, output);
    output.push(proof.a_commitment);
    output.push(proof.b_commitment);
    serialize_mle_opening_proof(&proof.a_opening, output);
    serialize_mle_opening_proof(&proof.b_opening, output);
}

/// Serialize a `MatMulSumcheckProofOnChain` with **compressed** round polys.
///
/// Same as `serialize_matmul_sumcheck_proof` but round polys use 8 felts (c0, c2)
/// instead of 12 felts (c0, c1, c2). The verifier reconstructs c1 via:
/// `c1 = current_sum - 2*c0 - c2` at each round.
///
/// Savings: `num_rounds × 4` felts per proof (~33% round poly reduction).
pub fn serialize_matmul_sumcheck_proof_compressed(
    proof: &MatMulSumcheckProofOnChain,
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(proof.m, output);
    serialize_u32(proof.k, output);
    serialize_u32(proof.n, output);
    serialize_u32(proof.num_rounds, output);
    serialize_qm31(proof.claimed_sum, output);
    // round_polys: Array<CompressedRoundPoly> (8 felts each instead of 12)
    serialize_u32(proof.round_polys.len() as u32, output);
    for rp in &proof.round_polys {
        serialize_round_poly_compressed(&rp.compress(), output);
    }
    serialize_qm31(proof.final_a_eval, output);
    serialize_qm31(proof.final_b_eval, output);
    output.push(proof.a_commitment);
    output.push(proof.b_commitment);
    serialize_mle_opening_proof(&proof.a_opening, output);
    serialize_mle_opening_proof(&proof.b_opening, output);
}

// === MLProof serialization for recursive verification ===
//
// Serializes an AggregatedModelProofOnChain into the felt252[] layout
// matching the Cairo `MLProof` struct's Serde deserialization.

use crate::aggregation::{AggregatedModelProofOnChain, BatchedMatMulProofOnChain, LayerClaim};
use crate::components::attention::AttentionProofOnChain;

/// Metadata about the ML model, needed to construct the Cairo MLClaim.
pub struct MLClaimMetadata {
    pub model_id: FieldElement,
    pub num_layers: u32,
    pub activation_type: u8,
    pub io_commitment: FieldElement,
    pub weight_commitment: FieldElement,
    /// Optional TEE attestation hash (Poseidon hash of NVIDIA CC attestation report).
    /// `None` or `Some(FieldElement::ZERO)` means no TEE attestation.
    /// `Some(hash)` means the proof was generated on CC-On hardware.
    /// Serialized as: 0 (no TEE) or 1 + hash (with TEE).
    pub tee_attestation_hash: Option<FieldElement>,
}

/// Serialize a lightweight Add or Mul layer claim.
///
/// Layout:
/// - `layer_index` (u32, 1 felt)
/// - `trace_rows` (u32, 1 felt)
///
/// Add/Mul are pure AIR (no LogUp), verified inside the unified STARK.
fn serialize_elementwise_claim(claim: &LayerClaim, output: &mut Vec<FieldElement>) {
    serialize_u32(claim.layer_index as u32, output);
    serialize_u32(claim.trace_rows as u32, output);
}

/// Serialize a lightweight LayerNorm layer claim.
///
/// Layout:
/// - `layer_index` (u32, 1 felt)
/// - `trace_rows` (u32, 1 felt)
/// - `claimed_sum` (QM31, 4 felts — LogUp claimed sum)
///
/// LayerNorm uses LogUp for range-checking rsqrt values, so we include
/// the claimed_sum for the verifier to check the LogUp argument.
fn serialize_layernorm_claim(claim: &LayerClaim, output: &mut Vec<FieldElement>) {
    serialize_u32(claim.layer_index as u32, output);
    serialize_u32(claim.trace_rows as u32, output);
    serialize_qm31(claim.claimed_sum, output);
}

/// Serialize a lightweight Embedding layer claim.
///
/// Layout:
/// - `layer_index` (u32, 1 felt)
/// - `trace_rows` (u32, 1 felt)
///
/// Embedding is verified inside the unified STARK (pure AIR, no LogUp).
fn serialize_embedding_claim(claim: &LayerClaim, output: &mut Vec<FieldElement>) {
    serialize_u32(claim.layer_index as u32, output);
    serialize_u32(claim.trace_rows as u32, output);
}

/// Serialize an on-chain attention proof (all matmul sub-proofs).
///
/// Layout:
/// - `num_heads` (u32, 1 felt) — derived from score_proofs.len()
/// - `q_proof`: MatMulSumcheckProofOnChain (10-field)
/// - `k_proof`: MatMulSumcheckProofOnChain (10-field)
/// - `v_proof`: MatMulSumcheckProofOnChain (10-field)
/// - `score_proofs`: Array<MatMulSumcheckProofOnChain>
/// - `attn_v_proofs`: Array<MatMulSumcheckProofOnChain>
/// - `output_proof`: MatMulSumcheckProofOnChain (10-field)
fn serialize_attention_proof_onchain(
    proof: &AttentionProofOnChain,
    output: &mut Vec<FieldElement>,
) {
    // num_heads (for the verifier to know how many score/attn_v proofs to expect)
    serialize_u32(proof.score_proofs.len() as u32, output);
    // Q, K, V projections
    serialize_matmul_for_recursive(&proof.q_proof, output);
    serialize_matmul_for_recursive(&proof.k_proof, output);
    serialize_matmul_for_recursive(&proof.v_proof, output);
    // Per-head score proofs
    serialize_u32(proof.score_proofs.len() as u32, output);
    for sp in &proof.score_proofs {
        serialize_matmul_for_recursive(sp, output);
    }
    // Per-head attn×V proofs
    serialize_u32(proof.attn_v_proofs.len() as u32, output);
    for ap in &proof.attn_v_proofs {
        serialize_matmul_for_recursive(ap, output);
    }
    // Output projection
    serialize_matmul_for_recursive(&proof.output_proof, output);
}

/// Serialize an ML proof for the recursive Cairo verifier.
///
/// Output format matches Cairo's `MLProof { claim, matmul_proofs, channel_salt, unified_stark_proof }`:
///
/// 1. MLClaim: model_id, num_layers, activation_type, io_commitment, weight_commitment
/// 2. matmul_proofs: Array<MatMulSumcheckProofOnChain> (10-field version, no MLE openings)
/// 3. channel_salt: Option<u64> (0 for None, 1 + value for Some)
/// 4. unified_stark_proof: Option<UnifiedStarkProof> — contains ALL non-matmul component claims
///    (activations, add, mul, layernorm, embedding) plus interaction claims and the STARK proof
/// 5. add_claims: Array<ElementwiseClaim> (layer_index + trace_rows) — graph-level metadata
/// 6. mul_claims: Array<ElementwiseClaim> (layer_index + trace_rows) — graph-level metadata
/// 7. layernorm_claims: Array<LayerNormClaim> (layer_index + trace_rows + claimed_sum) — graph-level metadata
/// 8. embedding_claims: Array<EmbeddingClaim> (layer_index + trace_rows) — graph-level metadata
/// 9. attention_proofs: Array<(layer_index, AttentionProofOnChain)> — attention sub-proofs
pub fn serialize_ml_proof_for_recursive(
    proof: &AggregatedModelProofOnChain,
    metadata: &MLClaimMetadata,
    channel_salt: Option<u64>,
) -> Vec<FieldElement> {
    let mut output = Vec::new();

    // 1. MLClaim
    output.push(metadata.model_id);
    serialize_u32(metadata.num_layers, &mut output);
    // activation_type is u8, serialize as felt252
    output.push(FieldElement::from(metadata.activation_type as u64));
    output.push(metadata.io_commitment);
    output.push(metadata.weight_commitment);

    // 2. matmul_proofs: Array<MatMulSumcheckProofOnChain> (10-field Cairo layout)
    serialize_u32(proof.matmul_proofs.len() as u32, &mut output);
    for (_layer_idx, matmul) in &proof.matmul_proofs {
        serialize_matmul_for_recursive(matmul, &mut output);
    }

    // 2b. batched_matmul_proofs: Array<BatchedMatMulProofOnChain>
    serialize_u32(proof.batched_matmul_proofs.len() as u32, &mut output);
    for batch in &proof.batched_matmul_proofs {
        serialize_batched_matmul_for_recursive(batch, &mut output);
    }

    // 3. channel_salt: Option<u64>
    match channel_salt {
        None => {
            serialize_u32(0, &mut output); // Cairo Option::None variant index
        }
        Some(salt) => {
            serialize_u32(1, &mut output); // Cairo Option::Some variant index
            serialize_u64(salt, &mut output);
        }
    }

    // 4. unified_stark_proof: Option<UnifiedStarkProof>
    match &proof.unified_stark {
        None => {
            serialize_u32(0, &mut output); // Cairo Option::None variant index
        }
        Some(stark_proof) => {
            serialize_u32(1, &mut output); // Cairo Option::Some variant index
            serialize_unified_stark_proof(
                stark_proof,
                &proof.activation_claims,
                &proof.add_claims,
                &proof.mul_claims,
                &proof.layernorm_claims,
                &proof.embedding_claims,
                metadata.activation_type,
                &mut output,
            );
        }
    }

    // 5. add_claims: Array<ElementwiseClaim>
    serialize_u32(proof.add_claims.len() as u32, &mut output);
    for claim in &proof.add_claims {
        serialize_elementwise_claim(claim, &mut output);
    }

    // 6. mul_claims: Array<ElementwiseClaim>
    serialize_u32(proof.mul_claims.len() as u32, &mut output);
    for claim in &proof.mul_claims {
        serialize_elementwise_claim(claim, &mut output);
    }

    // 7. layernorm_claims: Array<LayerNormClaim>
    serialize_u32(proof.layernorm_claims.len() as u32, &mut output);
    for claim in &proof.layernorm_claims {
        serialize_layernorm_claim(claim, &mut output);
    }

    // 8. embedding_claims: Array<EmbeddingClaim>
    serialize_u32(proof.embedding_claims.len() as u32, &mut output);
    for claim in &proof.embedding_claims {
        serialize_embedding_claim(claim, &mut output);
    }

    // 9. attention_proofs: Array<(layer_index, AttentionProofOnChain)>
    serialize_u32(proof.attention_proofs.len() as u32, &mut output);
    for (layer_idx, attn_proof) in &proof.attention_proofs {
        serialize_u32(*layer_idx as u32, &mut output);
        serialize_attention_proof_onchain(attn_proof, &mut output);
    }

    // 10. tee_attestation_hash: Option<felt252>
    //    Serialized as Cairo Option: 0 = None, 1 + value = Some.
    match metadata.tee_attestation_hash {
        Some(hash) if hash != FieldElement::ZERO => {
            serialize_u32(1, &mut output); // Some variant
            output.push(hash);
        }
        _ => {
            serialize_u32(0, &mut output); // None variant
        }
    }

    output
}

/// Serialize a matmul proof in the 10-field Cairo layout (no MLE openings).
///
/// This matches the Cairo `MatMulSumcheckProofOnChain` struct which has:
/// m, k, n, num_rounds, claimed_sum, round_polys, final_a_eval, final_b_eval,
/// a_commitment, b_commitment
fn serialize_matmul_for_recursive(
    proof: &MatMulSumcheckProofOnChain,
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(proof.m, output);
    serialize_u32(proof.k, output);
    serialize_u32(proof.n, output);
    serialize_u32(proof.num_rounds, output);
    serialize_qm31(proof.claimed_sum, output);
    // round_polys: Array<RoundPoly>
    serialize_u32(proof.round_polys.len() as u32, output);
    for rp in &proof.round_polys {
        serialize_round_poly(rp, output);
    }
    serialize_qm31(proof.final_a_eval, output);
    serialize_qm31(proof.final_b_eval, output);
    output.push(proof.a_commitment);
    output.push(proof.b_commitment);
}

/// Serialize a batched matmul proof for recursive verification.
///
/// Layout:
/// - k, num_rounds (2 felts)
/// - lambda (4 felts, QM31)
/// - combined_claimed_sum (4 felts, QM31)
/// - round_polys: [len] + len × 12 felts
/// - entries: [len] + per-entry: node_id, m, n, claimed_sum, final_a, final_b, a_commit, b_commit
fn serialize_batched_matmul_for_recursive(
    batch: &BatchedMatMulProofOnChain,
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(batch.k, output);
    serialize_u32(batch.num_rounds, output);
    serialize_qm31(batch.lambda, output);
    serialize_qm31(batch.combined_claimed_sum, output);
    // round_polys
    serialize_u32(batch.round_polys.len() as u32, output);
    for rp in &batch.round_polys {
        serialize_round_poly(rp, output);
    }
    // entries
    serialize_u32(batch.entries.len() as u32, output);
    for entry in &batch.entries {
        serialize_u32(entry.node_id as u32, output);
        serialize_u32(entry.m, output);
        serialize_u32(entry.n, output);
        serialize_qm31(entry.claimed_sum, output);
        serialize_qm31(entry.final_a_eval, output);
        serialize_qm31(entry.final_b_eval, output);
        output.push(entry.a_commitment);
        output.push(entry.b_commitment);
    }
}

// === Unified STARK Proof Serialization ===

/// Bridge from Rust `LayerClaim` to Cairo's `ActivationClaim` fields.
///
/// Cairo layout: `ActivationClaim { layer_index: u32, log_size: u32, activation_type: u8 }`
pub struct ActivationClaimForSerde {
    pub layer_index: u32,
    pub log_size: u32,
    pub activation_type: u8,
}

impl ActivationClaimForSerde {
    pub fn from_layer_claim(claim: &LayerClaim, activation_type: u8) -> Self {
        Self {
            layer_index: claim.layer_index as u32,
            log_size: claim.trace_rows.ilog2(),
            activation_type,
        }
    }
}

/// Bridge from Rust `LayerClaim` to Cairo's `ElementwiseComponentClaim` fields.
///
/// Cairo layout: `ElementwiseComponentClaim { layer_index: u32, log_size: u32 }`
/// Used for Add/Mul (pure AIR, no LogUp claimed_sum).
pub struct ElementwiseClaimForSerde {
    pub layer_index: u32,
    pub log_size: u32,
}

impl ElementwiseClaimForSerde {
    pub fn from_layer_claim(claim: &LayerClaim) -> Self {
        Self {
            layer_index: claim.layer_index as u32,
            log_size: claim.trace_rows.ilog2(),
        }
    }
}

/// Bridge from Rust `LayerClaim` to Cairo's `LayerNormComponentClaim` fields.
///
/// Cairo layout: `LayerNormComponentClaim { layer_index: u32, log_size: u32 }`
/// The LogUp claimed_sum is serialized separately in the interaction claims.
pub struct LayerNormClaimForSerde {
    pub layer_index: u32,
    pub log_size: u32,
}

impl LayerNormClaimForSerde {
    pub fn from_layer_claim(claim: &LayerClaim) -> Self {
        Self {
            layer_index: claim.layer_index as u32,
            log_size: claim.trace_rows.ilog2(),
        }
    }
}

/// Serialize an `ActivationClaim` (3 felt252s: layer_index, log_size, activation_type).
fn serialize_activation_claim(claim: &ActivationClaimForSerde, output: &mut Vec<FieldElement>) {
    serialize_u32(claim.layer_index, output);
    serialize_u32(claim.log_size, output);
    output.push(FieldElement::from(claim.activation_type as u64));
}

/// Serialize an `ElementwiseComponentClaim` (2 felt252s: layer_index, log_size).
fn serialize_elementwise_component_claim(
    claim: &ElementwiseClaimForSerde,
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(claim.layer_index, output);
    serialize_u32(claim.log_size, output);
}

/// Serialize a `LayerNormComponentClaim` (2 felt252s: layer_index, log_size).
fn serialize_layernorm_component_claim(
    claim: &LayerNormClaimForSerde,
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(claim.layer_index, output);
    serialize_u32(claim.log_size, output);
}

/// Serialize an interaction claim (4 felt252s: claimed_sum as QM31).
fn serialize_interaction_claim(sum: SecureField, output: &mut Vec<FieldElement>) {
    serialize_qm31(sum, output);
}

/// Serialize the total `MLInteractionClaim` (4 felt252s: total LogUp sum as QM31).
fn serialize_ml_interaction_claim(sum: SecureField, output: &mut Vec<FieldElement>) {
    serialize_qm31(sum, output);
}

/// Serialize a complete unified STARK proof covering all non-matmul components.
///
/// This matches the Cairo `UnifiedStarkProof` struct's `#[derive(Serde)]` field order:
///
/// 1. `activation_claims: Array<ActivationClaim>` — len + per-claim (3 felts each)
/// 2. `activation_interaction_claims: Array<QM31>` — len + per-claim (4 felts each)
/// 3. `add_claims: Array<ElementwiseComponentClaim>` — len + per-claim (2 felts each)
/// 4. `mul_claims: Array<ElementwiseComponentClaim>` — len + per-claim (2 felts each)
/// 5. `layernorm_claims: Array<LayerNormComponentClaim>` — len + per-claim (2 felts each)
/// 6. `layernorm_interaction_claims: Array<QM31>` — len + per-claim (4 felts each)
/// 7. `embedding_claims: Array<EmbeddingComponentClaim>` — len + per-claim (2 felts each)
/// 8. `interaction_claim: MLInteractionClaim` — 4 felts (total LogUp sum across activations + layernorms)
/// 9. `pcs_config: PcsConfig` — 4 felts (pow_bits + fri_config)
/// 10. `interaction_pow: u64` — 1 felt (0: matches INTERACTION_POW_BITS=0 in Cairo verifier)
/// 11. `stark_proof: StarkProof` — CommitmentSchemeProof (existing serializer)
fn serialize_unified_stark_proof(
    stark_proof: &Blake2sProof,
    activation_claims: &[LayerClaim],
    add_claims: &[LayerClaim],
    mul_claims: &[LayerClaim],
    layernorm_claims: &[LayerClaim],
    embedding_claims: &[LayerClaim],
    activation_type: u8,
    output: &mut Vec<FieldElement>,
) {
    // 1. activation_claims: Array<ActivationClaim>
    let serde_claims: Vec<ActivationClaimForSerde> = activation_claims
        .iter()
        .map(|c| ActivationClaimForSerde::from_layer_claim(c, activation_type))
        .collect();
    serialize_span(&serde_claims, serialize_activation_claim, output);

    // 2. activation_interaction_claims: Array<QM31>
    serialize_u32(activation_claims.len() as u32, output);
    for claim in activation_claims {
        serialize_interaction_claim(claim.claimed_sum, output);
    }

    // 3. add_claims: Array<ElementwiseComponentClaim>
    let add_serde: Vec<ElementwiseClaimForSerde> = add_claims
        .iter()
        .map(ElementwiseClaimForSerde::from_layer_claim)
        .collect();
    serialize_span(&add_serde, serialize_elementwise_component_claim, output);

    // 4. mul_claims: Array<ElementwiseComponentClaim>
    let mul_serde: Vec<ElementwiseClaimForSerde> = mul_claims
        .iter()
        .map(ElementwiseClaimForSerde::from_layer_claim)
        .collect();
    serialize_span(&mul_serde, serialize_elementwise_component_claim, output);

    // 5. layernorm_claims: Array<LayerNormComponentClaim>
    let ln_serde: Vec<LayerNormClaimForSerde> = layernorm_claims
        .iter()
        .map(LayerNormClaimForSerde::from_layer_claim)
        .collect();
    serialize_span(&ln_serde, serialize_layernorm_component_claim, output);

    // 6. layernorm_interaction_claims: Array<QM31>
    serialize_u32(layernorm_claims.len() as u32, output);
    for claim in layernorm_claims {
        serialize_interaction_claim(claim.claimed_sum, output);
    }

    // 7. embedding_claims: Array<EmbeddingComponentClaim>
    let emb_serde: Vec<ElementwiseClaimForSerde> = embedding_claims
        .iter()
        .map(ElementwiseClaimForSerde::from_layer_claim)
        .collect();
    serialize_span(&emb_serde, serialize_elementwise_component_claim, output);

    // 8. interaction_claim: MLInteractionClaim (sum of ALL LogUp claimed_sums)
    let total_sum: SecureField = activation_claims
        .iter()
        .chain(layernorm_claims.iter())
        .map(|c| c.claimed_sum)
        .fold(SecureField::default(), |acc, s| acc + s);
    serialize_ml_interaction_claim(total_sum, output);

    // 9. pcs_config: PcsConfig (from the proof itself)
    serialize_pcs_config(&stark_proof.0.config, output);

    // 10. interaction_pow: u64 (0: INTERACTION_POW_BITS=0, prover mixes this into channel)
    serialize_u64(0, output);

    // 11. stark_proof: StarkProof → CommitmentSchemeProof
    serialize_commitment_scheme_proof(&stark_proof.0, output);
}

/// Convert the serialized felt252[] to a JSON array of hex strings,
/// suitable for cairo-prove's `--arguments-file`.
pub fn serialize_ml_proof_to_arguments_file(felts: &[FieldElement]) -> String {
    let hex_strings: Vec<String> = felts.iter().map(|f| format!("\"0x{:x}\"", f)).collect();
    format!("[{}]", hex_strings.join(","))
}

/// Stream-write the serialized felt252[] directly to a file as a JSON hex array.
///
/// Avoids building the full String in memory — writes each felt directly via BufWriter.
/// For 100k+ felts, this saves ~10 MB of intermediate allocations and the O(n) join.
pub fn serialize_ml_proof_to_file(
    felts: &[FieldElement],
    path: &std::path::Path,
) -> std::io::Result<usize> {
    use std::io::Write;

    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file); // 1 MB buffer

    writer.write_all(b"[")?;
    let mut bytes_written = 1;

    for (i, f) in felts.iter().enumerate() {
        if i > 0 {
            writer.write_all(b",")?;
            bytes_written += 1;
        }
        let hex = format!("\"0x{:x}\"", f);
        writer.write_all(hex.as_bytes())?;
        bytes_written += hex.len();
    }

    writer.write_all(b"]")?;
    bytes_written += 1;
    writer.flush()?;

    Ok(bytes_written)
}

// === Direct Model Proof Serialization (Phase 9 — no Cairo VM) ===

/// Metadata for constructing the direct model proof.
pub struct DirectProofMetadata {
    pub model_id: FieldElement,
    /// IO commitment: Poseidon(inputs || outputs).
    pub io_commitment: FieldElement,
    /// Weight commitment (Poseidon hash of weight matrices).
    pub weight_commitment: FieldElement,
    pub num_layers: u32,
    pub activation_type: u8,
}

/// Serialize a complete model proof for direct on-chain verification.
///
/// This bypasses the Cairo VM recursive proving step entirely.
/// The output is split into:
///   1. `batched_calldata` — serialized batch sumcheck proofs (fits in main tx)
///   2. `stark_chunks` — serialized activation STARK proof (uploaded in prior txs)
///
/// Layout of batched_calldata (matches `BatchedMatMulProof`'s Serde order):
///   [batched_proof_0, batched_proof_1, ...]
///
/// Layout of stark_chunks:
///   Flat felt252 array containing the serialized ActivationStarkProof
///   (activation_claims + interaction_claims + interaction_claim + PCS + pow + StarkProof)
pub fn serialize_model_proof_direct(
    proof: &AggregatedModelProofOnChain,
    metadata: &DirectProofMetadata,
) -> DirectSerializedProof {
    // 1. Serialize batched matmul proofs
    let mut batched_calldata: Vec<Vec<FieldElement>> = Vec::new();
    for batch in &proof.batched_matmul_proofs {
        let mut buf = Vec::new();
        serialize_batched_matmul_for_recursive(batch, &mut buf);
        batched_calldata.push(buf);
    }

    // 2. Serialize activation STARK proof (if present)
    let stark_data = if let Some(ref stark_proof) = proof.unified_stark {
        let mut output = Vec::new();
        serialize_unified_stark_proof(
            stark_proof,
            &proof.activation_claims,
            &proof.add_claims,
            &proof.mul_claims,
            &proof.layernorm_claims,
            &proof.embedding_claims,
            metadata.activation_type,
            &mut output,
        );
        Some(output)
    } else {
        None
    };

    // 3. Split STARK data into chunks for multi-tx upload
    let max_chunk_size = 4000; // ~4000 felts per tx (well within Starknet limits)
    let stark_chunks = if let Some(ref data) = stark_data {
        data.chunks(max_chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    } else {
        Vec::new()
    };

    DirectSerializedProof {
        model_id: metadata.model_id,
        io_commitment: metadata.io_commitment,
        weight_commitment: metadata.weight_commitment,
        num_layers: metadata.num_layers,
        activation_type: metadata.activation_type,
        batched_calldata,
        stark_chunks,
        has_activation_stark: stark_data.is_some(),
    }
}

/// Serialized direct model proof ready for on-chain submission.
#[derive(Debug)]
pub struct DirectSerializedProof {
    /// Model identifier.
    pub model_id: FieldElement,
    /// Poseidon(inputs || outputs).
    pub io_commitment: FieldElement,
    /// Poseidon hash of weight matrices.
    pub weight_commitment: FieldElement,
    /// Number of model layers.
    pub num_layers: u32,
    /// Activation function type.
    pub activation_type: u8,
    /// Per-batch serialized sumcheck proofs.
    pub batched_calldata: Vec<Vec<FieldElement>>,
    /// Activation STARK proof split into upload chunks.
    pub stark_chunks: Vec<Vec<FieldElement>>,
    /// Whether an activation STARK proof is present.
    pub has_activation_stark: bool,
}

// === Deserialization (Cairo felt252[] → Rust types) ===

// === Tiled MatMul Proof Serialization ===

use crate::components::tiled_matmul::TiledMatMulProof;

/// Serialize a `TiledMatMulProof` to felt252 array.
///
/// Layout: `m, k, n, tile_k, num_tiles, [tile_proofs...], total_claimed_sum`
///
/// Each tile proof uses the standard `serialize_matmul_sumcheck_proof` format.
/// This allows the Cairo verifier to deserialize and verify each tile independently,
/// then check `sum(tile_claimed_sums) == total_claimed_sum`.
pub fn serialize_tiled_matmul_proof(proof: &TiledMatMulProof, output: &mut Vec<FieldElement>) {
    serialize_u32(proof.m as u32, output);
    serialize_u32(proof.k as u32, output);
    serialize_u32(proof.n as u32, output);
    serialize_u32(proof.tile_k as u32, output);
    serialize_u32(proof.tile_proofs.len() as u32, output);

    for tile in &proof.tile_proofs {
        // k_start and k_end for this tile
        serialize_u32(tile.k_start as u32, output);
        serialize_u32(tile.k_end as u32, output);
        // The standard matmul proof for this tile
        serialize_matmul_sumcheck_proof(&tile.proof, output);
    }

    // Total claimed sum (QM31 = 4 felts)
    serialize_qm31(proof.total_claimed_sum, output);
}

// ============================================================================
// GKR Proof Serialization
// ============================================================================

/// Serialize a GKR LogUp proof (activation/layernorm lookup arguments).
///
/// Layout: `claimed_sum, num_rounds, [c0,c1,c2,c3]×num_rounds, w_eval, in_eval, out_eval, num_mults, [mult]×num_mults`
fn serialize_logup_proof(lup: &crate::gkr::types::LogUpProof, output: &mut Vec<FieldElement>) {
    serialize_logup_proof_inner(lup, output, true)
}

/// Inner function with `include_multiplicities` flag.
///
/// On-chain mode (`include_multiplicities = false`): writes num_mults=0.
/// The eq-sumcheck binds the LogUp identity via Fiat-Shamir —
/// table-side sum verification is not needed for soundness.
fn serialize_logup_proof_inner(
    lup: &crate::gkr::types::LogUpProof,
    output: &mut Vec<FieldElement>,
    include_multiplicities: bool,
) {
    serialize_qm31(lup.claimed_sum, output);
    serialize_u32(lup.eq_round_polys.len() as u32, output);
    for rp in &lup.eq_round_polys {
        // Compressed: omit c1 (verifier reconstructs from current_sum)
        serialize_qm31(rp.c0, output);
        serialize_qm31(rp.c2, output);
        serialize_qm31(rp.c3, output);
    }
    let (w_eval, in_eval, out_eval) = lup.final_evals;
    serialize_qm31(w_eval, output);
    serialize_qm31(in_eval, output);
    serialize_qm31(out_eval, output);
    if include_multiplicities {
        serialize_u32(lup.multiplicities.len() as u32, output);
        for &mult in &lup.multiplicities {
            serialize_u32(mult, output);
        }
    } else {
        serialize_u32(0, output); // no multiplicities for on-chain
    }
}

/// Packed variant: QM31 values as 1 felt each instead of 4.
fn serialize_logup_proof_inner_packed(
    lup: &crate::gkr::types::LogUpProof,
    output: &mut Vec<FieldElement>,
    include_multiplicities: bool,
) {
    serialize_qm31_packed(lup.claimed_sum, output);
    serialize_u32(lup.eq_round_polys.len() as u32, output);
    for rp in &lup.eq_round_polys {
        // Compressed: omit c1 (verifier reconstructs from current_sum)
        serialize_qm31_packed(rp.c0, output);
        serialize_qm31_packed(rp.c2, output);
        serialize_qm31_packed(rp.c3, output);
    }
    let (w_eval, in_eval, out_eval) = lup.final_evals;
    serialize_qm31_packed(w_eval, output);
    serialize_qm31_packed(in_eval, output);
    serialize_qm31_packed(out_eval, output);
    if include_multiplicities {
        serialize_u32(lup.multiplicities.len() as u32, output);
        for &mult in &lup.multiplicities {
            serialize_u32(mult, output);
        }
    } else {
        serialize_u32(0, output);
    }
}

/// Serialize a complete GKR model proof for on-chain verification.
///
/// Layout: `num_layers, [layer_proof]×num_layers, input_claim, weight_commitments, io_commitment`
///
/// Layer proof tags: 0=MatMul, 1=Add, 2=Mul, 3=Activation, 4=LayerNorm, 5=Attention.
pub fn serialize_gkr_model_proof(proof: &crate::gkr::GKRProof, output: &mut Vec<FieldElement>) {
    use crate::gkr::types::LayerProof;

    serialize_u32(proof.layer_proofs.len() as u32, output);

    for layer_proof in &proof.layer_proofs {
        match layer_proof {
            LayerProof::MatMul {
                round_polys,
                final_a_eval,
                final_b_eval,
            } => {
                serialize_u32(0, output); // tag: MatMul
                serialize_u32(round_polys.len() as u32, output);
                for rp in round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                }
                serialize_qm31(*final_a_eval, output);
                serialize_qm31(*final_b_eval, output);
            }
            LayerProof::Add {
                lhs_eval,
                rhs_eval,
                trunk_idx,
            } => {
                serialize_u32(1, output); // tag: Add
                serialize_qm31(*lhs_eval, output);
                serialize_qm31(*rhs_eval, output);
                serialize_u32(*trunk_idx as u32, output);
            }
            LayerProof::Mul {
                eq_round_polys,
                lhs_eval,
                rhs_eval,
            } => {
                serialize_u32(2, output); // tag: Mul
                serialize_u32(eq_round_polys.len() as u32, output);
                for rp in eq_round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                    serialize_qm31(rp.c3, output);
                }
                serialize_qm31(*lhs_eval, output);
                serialize_qm31(*rhs_eval, output);
            }
            LayerProof::Activation {
                activation_type,
                logup_proof,
                input_eval,
                output_eval,
                table_commitment,
            } => {
                serialize_u32(3, output); // tag: Activation
                                          // CRITICAL: use type_tag(), NOT enum discriminant — prover mixes type_tag()
                serialize_u32(activation_type.type_tag(), output);
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                output.push(*table_commitment);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_logup_proof(lup, output);
                    }
                    None => {
                        serialize_u32(0, output);
                    }
                }
            }
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
                serialize_u32(4, output); // tag: LayerNorm
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                serialize_qm31(*mean, output);
                serialize_qm31(*rsqrt_var, output);
                output.push(*rsqrt_table_commitment);
                // simd_combined flag: 1 = SIMD path (LogUp optional), 0 = standard (LogUp required)
                serialize_u32(if *simd_combined { 1 } else { 0 }, output);
                serialize_u32(linear_round_polys.len() as u32, output);
                for rp in linear_round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                    serialize_qm31(rp.c3, output);
                }
                let (centered_eval, rsqrt_eval) = *linear_final_evals;
                serialize_qm31(centered_eval, output);
                serialize_qm31(rsqrt_eval, output);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_logup_proof(lup, output);
                    }
                    None => {
                        serialize_u32(0, output);
                    }
                }
            }
            LayerProof::Attention {
                sub_proofs,
                sub_claim_values,
            } => {
                serialize_u32(5, output); // tag: Attention
                serialize_u32(sub_proofs.len() as u32, output);
                // Serialize sub_claim_values
                for val in sub_claim_values {
                    serialize_qm31(*val, output);
                }
                for sub in sub_proofs {
                    let tmp = crate::gkr::GKRProof {
                        layer_proofs: vec![sub.clone()],
                        output_claim: proof.output_claim.clone(),
                        input_claim: proof.input_claim.clone(),
                        weight_commitments: vec![],
                        weight_openings: vec![],
                        weight_claims: vec![],
                        weight_opening_transcript_mode:
                            crate::gkr::types::WeightOpeningTranscriptMode::Sequential,
                        io_commitment: proof.io_commitment,
                        deferred_proofs: vec![],
                        aggregated_binding: None,
                    };
                    let mut sub_buf = Vec::new();
                    serialize_gkr_model_proof(&tmp, &mut sub_buf);
                    // Skip the num_layers=1 prefix felt
                    output.extend_from_slice(&sub_buf[1..]);
                }
            }
            LayerProof::Dequantize {
                logup_proof,
                input_eval,
                output_eval,
                table_commitment,
            } => {
                serialize_u32(6, output); // tag: Dequantize
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                output.push(*table_commitment);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_logup_proof(lup, output);
                    }
                    None => {
                        serialize_u32(0, output);
                    }
                }
            }
            LayerProof::Quantize {
                logup_proof,
                input_eval,
                output_eval,
                table_inputs,
                table_outputs,
            } => {
                serialize_u32(9, output); // tag: Quantize
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_logup_proof(lup, output);
                    }
                    None => serialize_u32(0, output),
                }
                serialize_u32(table_inputs.len() as u32, output);
                for (&inp, &out) in table_inputs.iter().zip(table_outputs.iter()) {
                    serialize_m31(inp, output);
                    serialize_m31(out, output);
                }
            }
            LayerProof::Embedding {
                logup_proof,
                input_eval,
                output_eval,
                input_num_vars,
            } => {
                serialize_u32(10, output); // tag: Embedding
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                serialize_u32(*input_num_vars as u32, output);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_u32(lup.eq_round_polys.len() as u32, output);
                        for rp in &lup.eq_round_polys {
                            serialize_qm31(rp.c0, output);
                            serialize_qm31(rp.c1, output);
                            serialize_qm31(rp.c2, output);
                            serialize_qm31(rp.c3, output);
                        }
                        serialize_qm31(lup.claimed_sum, output);
                        let (w, tok, col, val) = lup.final_evals;
                        serialize_qm31(w, output);
                        serialize_qm31(tok, output);
                        serialize_qm31(col, output);
                        serialize_qm31(val, output);
                        serialize_u32(lup.table_tokens.len() as u32, output);
                        for i in 0..lup.table_tokens.len() {
                            serialize_u32(lup.table_tokens[i], output);
                            serialize_u32(lup.table_cols[i], output);
                            serialize_u32(lup.multiplicities[i], output);
                        }
                    }
                    None => serialize_u32(0, output),
                }
            }
            LayerProof::MatMulDualSimd {
                round_polys,
                final_a_eval,
                final_b_eval,
                n_block_vars,
            } => {
                serialize_u32(7, output); // tag: MatMulDualSimd
                serialize_u32(*n_block_vars as u32, output);
                serialize_u32(round_polys.len() as u32, output);
                for rp in round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                    serialize_qm31(rp.c3, output);
                }
                serialize_qm31(*final_a_eval, output);
                serialize_qm31(*final_b_eval, output);
            }

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
            } => {
                serialize_u32(8, output); // tag: RMSNorm
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                serialize_qm31(*rms_sq_eval, output);
                serialize_qm31(*rsqrt_eval, output);
                output.push(*rsqrt_table_commitment);
                serialize_u32(if *simd_combined { 1 } else { 0 }, output);
                serialize_u32(linear_round_polys.len() as u32, output);
                for rp in linear_round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                    serialize_qm31(rp.c3, output);
                }
                let (input_final, rsqrt_final) = *linear_final_evals;
                serialize_qm31(input_final, output);
                serialize_qm31(rsqrt_final, output);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_logup_proof(lup, output);
                    }
                    None => {
                        serialize_u32(0, output);
                    }
                }
            }
        }
    }

    // Input claim: point + value
    serialize_u32(proof.input_claim.point.len() as u32, output);
    for p in &proof.input_claim.point {
        serialize_qm31(*p, output);
    }
    serialize_qm31(proof.input_claim.value, output);

    // Weight commitments
    serialize_u32(proof.weight_commitments.len() as u32, output);
    for commitment in &proof.weight_commitments {
        output.push(*commitment);
    }

    // Weight opening proofs (one per MatMul layer).
    // Each opening proves: MLE(weight, eval_point) == expected_value.
    serialize_u32(proof.weight_openings.len() as u32, output);
    for (opening, claim) in proof.weight_openings.iter().zip(proof.weight_claims.iter()) {
        // Eval point
        serialize_u32(claim.eval_point.len() as u32, output);
        for &p in &claim.eval_point {
            serialize_qm31(p, output);
        }
        // Expected value (== final_b_eval from matmul sumcheck)
        serialize_qm31(claim.expected_value, output);
        // Full MLE opening proof (Merkle auth paths + intermediate roots)
        serialize_mle_opening_proof(opening, output);
    }

    // IO commitment
    output.push(proof.io_commitment);
}

/// Serialize ONLY the per-layer GKR proof data (tags + layer-specific fields).
///
/// This omits the header (num_layers) and footer (input_claim, weight_commitments,
/// io_commitment) that `serialize_gkr_model_proof` includes. Used to produce
/// the `proof_data: Array<felt252>` parameter for the contract's `verify_model_gkr()`.
pub fn serialize_gkr_proof_data_only(proof: &crate::gkr::GKRProof, output: &mut Vec<FieldElement>) {
    use crate::gkr::types::LayerProof;

    for layer_proof in &proof.layer_proofs {
        match layer_proof {
            LayerProof::MatMul {
                round_polys,
                final_a_eval,
                final_b_eval,
            } => {
                serialize_u32(0, output);
                serialize_u32(round_polys.len() as u32, output);
                for rp in round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                }
                serialize_qm31(*final_a_eval, output);
                serialize_qm31(*final_b_eval, output);
            }
            LayerProof::Add {
                lhs_eval,
                rhs_eval,
                trunk_idx,
            } => {
                serialize_u32(1, output);
                serialize_qm31(*lhs_eval, output);
                serialize_qm31(*rhs_eval, output);
                serialize_u32(*trunk_idx as u32, output);
            }
            LayerProof::Mul {
                eq_round_polys,
                lhs_eval,
                rhs_eval,
            } => {
                serialize_u32(2, output);
                serialize_u32(eq_round_polys.len() as u32, output);
                for rp in eq_round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                    serialize_qm31(rp.c3, output);
                }
                serialize_qm31(*lhs_eval, output);
                serialize_qm31(*rhs_eval, output);
            }
            LayerProof::Activation {
                activation_type,
                logup_proof,
                input_eval,
                output_eval,
                table_commitment,
            } => {
                serialize_u32(3, output);
                // CRITICAL: use type_tag(), NOT enum discriminant — prover mixes type_tag()
                serialize_u32(activation_type.type_tag(), output);
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                output.push(*table_commitment);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        // Strip multiplicities for on-chain (saves ~65K calldata for ReLU)
                        serialize_logup_proof_inner(lup, output, false);
                    }
                    None => serialize_u32(0, output),
                }
            }
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
                serialize_u32(4, output);
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                serialize_qm31(*mean, output);
                serialize_qm31(*rsqrt_var, output);
                output.push(*rsqrt_table_commitment);
                serialize_u32(if *simd_combined { 1 } else { 0 }, output);
                serialize_u32(linear_round_polys.len() as u32, output);
                for rp in linear_round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                    serialize_qm31(rp.c3, output);
                }
                let (centered_eval, rsqrt_eval) = *linear_final_evals;
                serialize_qm31(centered_eval, output);
                serialize_qm31(rsqrt_eval, output);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_logup_proof_inner(lup, output, false);
                    }
                    None => serialize_u32(0, output),
                }
            }
            LayerProof::Attention {
                sub_proofs,
                sub_claim_values,
            } => {
                serialize_u32(5, output);
                serialize_u32(sub_proofs.len() as u32, output);
                for val in sub_claim_values {
                    serialize_qm31(*val, output);
                }
                for sub in sub_proofs {
                    let tmp = crate::gkr::GKRProof {
                        layer_proofs: vec![sub.clone()],
                        output_claim: proof.output_claim.clone(),
                        input_claim: proof.input_claim.clone(),
                        weight_commitments: vec![],
                        weight_openings: vec![],
                        weight_claims: vec![],
                        weight_opening_transcript_mode:
                            crate::gkr::types::WeightOpeningTranscriptMode::Sequential,
                        io_commitment: proof.io_commitment,
                        deferred_proofs: vec![],
                        aggregated_binding: None,
                    };
                    let mut sub_buf = Vec::new();
                    serialize_gkr_model_proof(&tmp, &mut sub_buf);
                    output.extend_from_slice(&sub_buf[1..]);
                }
            }
            LayerProof::Dequantize {
                logup_proof,
                input_eval,
                output_eval,
                table_commitment,
            } => {
                serialize_u32(6, output);
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                output.push(*table_commitment);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_logup_proof_inner(lup, output, false);
                    }
                    None => serialize_u32(0, output),
                }
            }
            LayerProof::Quantize {
                logup_proof,
                input_eval,
                output_eval,
                table_inputs,
                table_outputs,
            } => {
                serialize_u32(9, output);
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_logup_proof_inner(lup, output, true);
                    }
                    None => serialize_u32(0, output),
                }
                serialize_u32(table_inputs.len() as u32, output);
                for (&inp, &out) in table_inputs.iter().zip(table_outputs.iter()) {
                    serialize_m31(inp, output);
                    serialize_m31(out, output);
                }
            }
            LayerProof::Embedding {
                logup_proof,
                input_eval,
                output_eval,
                input_num_vars,
            } => {
                serialize_u32(10, output);
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                serialize_u32(*input_num_vars as u32, output);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_u32(lup.eq_round_polys.len() as u32, output);
                        for rp in &lup.eq_round_polys {
                            serialize_qm31(rp.c0, output);
                            serialize_qm31(rp.c1, output);
                            serialize_qm31(rp.c2, output);
                            serialize_qm31(rp.c3, output);
                        }
                        serialize_qm31(lup.claimed_sum, output);
                        let (w, tok, col, val) = lup.final_evals;
                        serialize_qm31(w, output);
                        serialize_qm31(tok, output);
                        serialize_qm31(col, output);
                        serialize_qm31(val, output);
                        serialize_u32(lup.table_tokens.len() as u32, output);
                        for i in 0..lup.table_tokens.len() {
                            serialize_u32(lup.table_tokens[i], output);
                            serialize_u32(lup.table_cols[i], output);
                            serialize_u32(lup.multiplicities[i], output);
                        }
                    }
                    None => serialize_u32(0, output),
                }
            }
            LayerProof::MatMulDualSimd {
                round_polys,
                final_a_eval,
                final_b_eval,
                n_block_vars,
            } => {
                serialize_u32(7, output);
                serialize_u32(*n_block_vars as u32, output);
                serialize_u32(round_polys.len() as u32, output);
                for rp in round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                    serialize_qm31(rp.c3, output);
                }
                serialize_qm31(*final_a_eval, output);
                serialize_qm31(*final_b_eval, output);
            }
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
            } => {
                serialize_u32(8, output);
                serialize_qm31(*input_eval, output);
                serialize_qm31(*output_eval, output);
                serialize_qm31(*rms_sq_eval, output);
                serialize_qm31(*rsqrt_eval, output);
                output.push(*rsqrt_table_commitment);
                serialize_u32(if *simd_combined { 1 } else { 0 }, output);
                serialize_u32(linear_round_polys.len() as u32, output);
                for rp in linear_round_polys {
                    // Compressed: omit c1 (verifier reconstructs from current_sum)
                    serialize_qm31(rp.c0, output);
                    serialize_qm31(rp.c2, output);
                    serialize_qm31(rp.c3, output);
                }
                let (input_final, rsqrt_final) = *linear_final_evals;
                serialize_qm31(input_final, output);
                serialize_qm31(rsqrt_final, output);
                match logup_proof {
                    Some(lup) => {
                        serialize_u32(1, output);
                        serialize_logup_proof_inner(lup, output, false);
                    }
                    None => serialize_u32(0, output),
                }
            }
        }
    }

    // Serialize deferred proofs (DAG Add layer skip connections)
    serialize_u32(proof.deferred_proofs.len() as u32, output);
    for deferred in &proof.deferred_proofs {
        // Claim value (the skip_eval from the Add reduction)
        serialize_qm31(deferred.claim.value, output);
        // MatMul dimensions
        let (m, k, n) = deferred.dims().unwrap();
        serialize_u32(m as u32, output);
        serialize_u32(k as u32, output);
        serialize_u32(n as u32, output);
        // MatMul sumcheck proof (same format as Tag 0 but without the tag)
        if let LayerProof::MatMul {
            round_polys,
            final_a_eval,
            final_b_eval,
        } = &deferred.layer_proof
        {
            serialize_u32(round_polys.len() as u32, output);
            for rp in round_polys {
                // Compressed: omit c1 (verifier reconstructs from current_sum)
                serialize_qm31(rp.c0, output);
                serialize_qm31(rp.c2, output);
            }
            serialize_qm31(*final_a_eval, output);
            serialize_qm31(*final_b_eval, output);
        }
        // Weight commitment
        output.push(deferred.weight_commitment().unwrap());
    }
}

/// Serialize a single GKR layer proof in packed format.
///
/// Extracted for incremental/streaming serialization: callers can serialize
/// layer proofs one at a time as they are produced by the GKR prover.
pub fn serialize_layer_proof_packed(layer_proof: &crate::gkr::types::LayerProof, output: &mut Vec<FieldElement>) {
    serialize_layer_proof_packed_inner(layer_proof, output, None);
}

fn serialize_layer_proof_packed_inner(
    layer_proof: &crate::gkr::types::LayerProof,
    output: &mut Vec<FieldElement>,
    proof_for_attention: Option<&crate::gkr::GKRProof>,
) {
    use crate::gkr::types::LayerProof;
    match layer_proof {
        LayerProof::MatMul {
            round_polys,
            final_a_eval,
            final_b_eval,
        } => {
            serialize_u32(0, output);
            serialize_u32(round_polys.len() as u32, output);
            for rp in round_polys {
                serialize_qm31_packed(rp.c0, output);
                serialize_qm31_packed(rp.c2, output);
            }
            serialize_qm31_packed(*final_a_eval, output);
            serialize_qm31_packed(*final_b_eval, output);
        }
        LayerProof::Add {
            lhs_eval,
            rhs_eval,
            trunk_idx,
        } => {
            serialize_u32(1, output);
            serialize_qm31_packed(*lhs_eval, output);
            serialize_qm31_packed(*rhs_eval, output);
            serialize_u32(*trunk_idx as u32, output);
        }
        LayerProof::Mul {
            eq_round_polys,
            lhs_eval,
            rhs_eval,
        } => {
            serialize_u32(2, output);
            serialize_u32(eq_round_polys.len() as u32, output);
            for rp in eq_round_polys {
                serialize_qm31_packed(rp.c0, output);
                serialize_qm31_packed(rp.c2, output);
                serialize_qm31_packed(rp.c3, output);
            }
            serialize_qm31_packed(*lhs_eval, output);
            serialize_qm31_packed(*rhs_eval, output);
        }
        LayerProof::Activation {
            activation_type,
            logup_proof,
            input_eval,
            output_eval,
            table_commitment,
        } => {
            serialize_u32(3, output);
            serialize_u32(activation_type.type_tag(), output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            output.push(*table_commitment);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_packed(lup, output, false);
                }
                None => serialize_u32(0, output),
            }
        }
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
            serialize_u32(4, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            serialize_qm31_packed(*mean, output);
            serialize_qm31_packed(*rsqrt_var, output);
            output.push(*rsqrt_table_commitment);
            serialize_u32(if *simd_combined { 1 } else { 0 }, output);
            serialize_u32(linear_round_polys.len() as u32, output);
            for rp in linear_round_polys {
                serialize_qm31_packed(rp.c0, output);
                serialize_qm31_packed(rp.c2, output);
                serialize_qm31_packed(rp.c3, output);
            }
            let (centered_eval, rsqrt_eval) = *linear_final_evals;
            serialize_qm31_packed(centered_eval, output);
            serialize_qm31_packed(rsqrt_eval, output);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_packed(lup, output, false);
                }
                None => serialize_u32(0, output),
            }
        }
        LayerProof::Attention {
            sub_proofs,
            sub_claim_values,
        } => {
            serialize_u32(5, output);
            serialize_u32(sub_proofs.len() as u32, output);
            for val in sub_claim_values {
                serialize_qm31_packed(*val, output);
            }
            if let Some(proof) = proof_for_attention {
                for sub in sub_proofs {
                    let tmp = crate::gkr::GKRProof {
                        layer_proofs: vec![sub.clone()],
                        output_claim: proof.output_claim.clone(),
                        input_claim: proof.input_claim.clone(),
                        weight_commitments: vec![],
                        weight_openings: vec![],
                        weight_claims: vec![],
                        weight_opening_transcript_mode:
                            crate::gkr::types::WeightOpeningTranscriptMode::Sequential,
                        io_commitment: proof.io_commitment,
                        deferred_proofs: vec![],
                        aggregated_binding: None,
                    };
                    let mut sub_buf = Vec::new();
                    serialize_gkr_model_proof(&tmp, &mut sub_buf);
                    output.extend_from_slice(&sub_buf[1..]);
                }
            }
        }
        LayerProof::Dequantize {
            logup_proof,
            input_eval,
            output_eval,
            table_commitment,
        } => {
            serialize_u32(6, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            output.push(*table_commitment);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_packed(lup, output, false);
                }
                None => serialize_u32(0, output),
            }
        }
        LayerProof::MatMulDualSimd {
            round_polys,
            final_a_eval,
            final_b_eval,
            n_block_vars,
        } => {
            serialize_u32(7, output);
            serialize_u32(*n_block_vars as u32, output);
            serialize_u32(round_polys.len() as u32, output);
            for rp in round_polys {
                serialize_qm31_packed(rp.c0, output);
                serialize_qm31_packed(rp.c1, output);
                serialize_qm31_packed(rp.c2, output);
                serialize_qm31_packed(rp.c3, output);
            }
            serialize_qm31_packed(*final_a_eval, output);
            serialize_qm31_packed(*final_b_eval, output);
        }
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
        } => {
            serialize_u32(8, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            serialize_qm31_packed(*rms_sq_eval, output);
            serialize_qm31_packed(*rsqrt_eval, output);
            output.push(*rsqrt_table_commitment);
            serialize_u32(if *simd_combined { 1 } else { 0 }, output);
            serialize_u32(linear_round_polys.len() as u32, output);
            for rp in linear_round_polys {
                serialize_qm31_packed(rp.c0, output);
                serialize_qm31_packed(rp.c2, output);
                serialize_qm31_packed(rp.c3, output);
            }
            let (input_final, rsqrt_final) = *linear_final_evals;
            serialize_qm31_packed(input_final, output);
            serialize_qm31_packed(rsqrt_final, output);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_packed(lup, output, false);
                }
                None => serialize_u32(0, output),
            }
        }
        LayerProof::Quantize {
            logup_proof,
            input_eval,
            output_eval,
            table_inputs,
            table_outputs,
        } => {
            serialize_u32(9, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_packed(lup, output, true);
                }
                None => serialize_u32(0, output),
            }
            serialize_u32(table_inputs.len() as u32, output);
            for (&inp, &out) in table_inputs.iter().zip(table_outputs.iter()) {
                serialize_m31(inp, output);
                serialize_m31(out, output);
            }
        }
        LayerProof::Embedding {
            logup_proof,
            input_eval,
            output_eval,
            input_num_vars,
        } => {
            serialize_u32(10, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            serialize_u32(*input_num_vars as u32, output);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_u32(lup.eq_round_polys.len() as u32, output);
                    for rp in &lup.eq_round_polys {
                        serialize_qm31_packed(rp.c0, output);
                        serialize_qm31_packed(rp.c1, output);
                        serialize_qm31_packed(rp.c2, output);
                        serialize_qm31_packed(rp.c3, output);
                    }
                    serialize_qm31_packed(lup.claimed_sum, output);
                    let (w, tok, col, val) = lup.final_evals;
                    serialize_qm31_packed(w, output);
                    serialize_qm31_packed(tok, output);
                    serialize_qm31_packed(col, output);
                    serialize_qm31_packed(val, output);
                    serialize_u32(lup.table_tokens.len() as u32, output);
                    for i in 0..lup.table_tokens.len() {
                        serialize_u32(lup.table_tokens[i], output);
                        serialize_u32(lup.table_cols[i], output);
                        serialize_u32(lup.multiplicities[i], output);
                    }
                }
                None => serialize_u32(0, output),
            }
        }
    }
}

/// Packed variant of `serialize_gkr_proof_data_only`: QM31 values as 1 felt each.
///
/// Achieves ~4x compression on proof_data (75K → ~19K felts for Qwen3-14B 40 layers).
/// Tags, u32 metadata, and felt252 commitments are unchanged.
pub fn serialize_gkr_proof_data_only_packed(proof: &crate::gkr::GKRProof, output: &mut Vec<FieldElement>) {
    for layer_proof in &proof.layer_proofs {
        serialize_layer_proof_packed_inner(layer_proof, output, Some(proof));
    }

    // Serialize deferred proofs (DAG Add layer skip connections) — also packed
    serialize_u32(proof.deferred_proofs.len() as u32, output);
    for deferred in &proof.deferred_proofs {
        serialize_qm31_packed(deferred.claim.value, output);
        let (m, k, n) = deferred.dims().unwrap();
        serialize_u32(m as u32, output);
        serialize_u32(k as u32, output);
        serialize_u32(n as u32, output);
        if let crate::gkr::types::LayerProof::MatMul {
            round_polys,
            final_a_eval,
            final_b_eval,
        } = &deferred.layer_proof
        {
            serialize_u32(round_polys.len() as u32, output);
            for rp in round_polys {
                // Compressed: omit c1 (verifier reconstructs from current_sum)
                serialize_qm31_packed(rp.c0, output);
                serialize_qm31_packed(rp.c2, output);
            }
            serialize_qm31_packed(*final_a_eval, output);
            serialize_qm31_packed(*final_b_eval, output);
        }
        output.push(deferred.weight_commitment().unwrap());
    }
}

/// Double-packed variant: degree-2 round polys (c0, c2) packed as QM31 pairs (1 felt each),
/// degree-3 round polys (c0, c2) as pair + c3 as single packed QM31.
/// All other QM31 values remain single-packed. Saves ~50% on proof_data section.
pub fn serialize_gkr_proof_data_only_double_packed(proof: &crate::gkr::GKRProof, output: &mut Vec<FieldElement>) {
    for layer_proof in &proof.layer_proofs {
        serialize_layer_proof_double_packed_inner(layer_proof, output, Some(proof));
    }

    // Serialize deferred proofs — also double-packed
    serialize_u32(proof.deferred_proofs.len() as u32, output);
    for deferred in &proof.deferred_proofs {
        serialize_qm31_packed(deferred.claim.value, output);
        let (m, k, n) = deferred.dims().unwrap();
        serialize_u32(m as u32, output);
        serialize_u32(k as u32, output);
        serialize_u32(n as u32, output);
        if let crate::gkr::types::LayerProof::MatMul {
            round_polys,
            final_a_eval,
            final_b_eval,
        } = &deferred.layer_proof
        {
            serialize_u32(round_polys.len() as u32, output);
            for rp in round_polys {
                // Double-packed: c0+c2 in one felt
                serialize_qm31_pair_packed(rp.c0, rp.c2, output);
            }
            serialize_qm31_packed(*final_a_eval, output);
            serialize_qm31_packed(*final_b_eval, output);
        }
        output.push(deferred.weight_commitment().unwrap());
    }
}

fn serialize_layer_proof_double_packed_inner(
    layer_proof: &crate::gkr::types::LayerProof,
    output: &mut Vec<FieldElement>,
    proof_for_attention: Option<&crate::gkr::GKRProof>,
) {
    use crate::gkr::types::LayerProof;
    match layer_proof {
        LayerProof::MatMul {
            round_polys,
            final_a_eval,
            final_b_eval,
        } => {
            serialize_u32(0, output);
            serialize_u32(round_polys.len() as u32, output);
            for rp in round_polys {
                // Double-packed: c0+c2 in one felt252
                serialize_qm31_pair_packed(rp.c0, rp.c2, output);
            }
            serialize_qm31_packed(*final_a_eval, output);
            serialize_qm31_packed(*final_b_eval, output);
        }
        LayerProof::Add {
            lhs_eval,
            rhs_eval,
            trunk_idx,
        } => {
            serialize_u32(1, output);
            serialize_qm31_packed(*lhs_eval, output);
            serialize_qm31_packed(*rhs_eval, output);
            serialize_u32(*trunk_idx as u32, output);
        }
        LayerProof::Mul {
            eq_round_polys,
            lhs_eval,
            rhs_eval,
        } => {
            serialize_u32(2, output);
            serialize_u32(eq_round_polys.len() as u32, output);
            for rp in eq_round_polys {
                // Double-packed: (c0, c2) as pair, c3 as single
                serialize_qm31_pair_packed(rp.c0, rp.c2, output);
                serialize_qm31_packed(rp.c3, output);
            }
            serialize_qm31_packed(*lhs_eval, output);
            serialize_qm31_packed(*rhs_eval, output);
        }
        LayerProof::Activation {
            activation_type,
            logup_proof,
            input_eval,
            output_eval,
            table_commitment,
        } => {
            serialize_u32(3, output);
            serialize_u32(activation_type.type_tag(), output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            output.push(*table_commitment);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_double_packed(lup, output, false);
                }
                None => serialize_u32(0, output),
            }
        }
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
            serialize_u32(4, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            serialize_qm31_packed(*mean, output);
            serialize_qm31_packed(*rsqrt_var, output);
            output.push(*rsqrt_table_commitment);
            serialize_u32(if *simd_combined { 1 } else { 0 }, output);
            serialize_u32(linear_round_polys.len() as u32, output);
            for rp in linear_round_polys {
                // Double-packed: (c0, c2) as pair, c3 as single
                serialize_qm31_pair_packed(rp.c0, rp.c2, output);
                serialize_qm31_packed(rp.c3, output);
            }
            let (centered_eval, rsqrt_eval) = *linear_final_evals;
            serialize_qm31_packed(centered_eval, output);
            serialize_qm31_packed(rsqrt_eval, output);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_double_packed(lup, output, false);
                }
                None => serialize_u32(0, output),
            }
        }
        LayerProof::Attention {
            sub_proofs,
            sub_claim_values,
        } => {
            // Attention not commonly used with double-packed; fall back to packed
            serialize_u32(5, output);
            serialize_u32(sub_proofs.len() as u32, output);
            for val in sub_claim_values {
                serialize_qm31_packed(*val, output);
            }
            if let Some(proof) = proof_for_attention {
                for sub in sub_proofs {
                    let tmp = crate::gkr::GKRProof {
                        layer_proofs: vec![sub.clone()],
                        output_claim: proof.output_claim.clone(),
                        input_claim: proof.input_claim.clone(),
                        weight_commitments: vec![],
                        weight_openings: vec![],
                        weight_claims: vec![],
                        weight_opening_transcript_mode:
                            crate::gkr::types::WeightOpeningTranscriptMode::Sequential,
                        io_commitment: proof.io_commitment,
                        deferred_proofs: vec![],
                        aggregated_binding: None,
                    };
                    let mut sub_buf = Vec::new();
                    serialize_gkr_model_proof(&tmp, &mut sub_buf);
                    output.extend_from_slice(&sub_buf[1..]);
                }
            }
        }
        LayerProof::Dequantize {
            logup_proof,
            input_eval,
            output_eval,
            table_commitment,
        } => {
            serialize_u32(6, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            output.push(*table_commitment);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_double_packed(lup, output, false);
                }
                None => serialize_u32(0, output),
            }
        }
        LayerProof::MatMulDualSimd {
            round_polys,
            final_a_eval,
            final_b_eval,
            n_block_vars,
        } => {
            serialize_u32(7, output);
            serialize_u32(*n_block_vars as u32, output);
            serialize_u32(round_polys.len() as u32, output);
            for rp in round_polys {
                // MatMulDualSimd has deg-3 polys with all 4 coeffs — pack (c0,c1) and (c2,c3)
                serialize_qm31_pair_packed(rp.c0, rp.c1, output);
                serialize_qm31_pair_packed(rp.c2, rp.c3, output);
            }
            serialize_qm31_packed(*final_a_eval, output);
            serialize_qm31_packed(*final_b_eval, output);
        }
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
        } => {
            serialize_u32(8, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            serialize_qm31_packed(*rms_sq_eval, output);
            serialize_qm31_packed(*rsqrt_eval, output);
            output.push(*rsqrt_table_commitment);
            serialize_u32(if *simd_combined { 1 } else { 0 }, output);
            serialize_u32(linear_round_polys.len() as u32, output);
            for rp in linear_round_polys {
                // Double-packed: (c0, c2) as pair, c3 as single
                serialize_qm31_pair_packed(rp.c0, rp.c2, output);
                serialize_qm31_packed(rp.c3, output);
            }
            let (input_final, rsqrt_final) = *linear_final_evals;
            serialize_qm31_packed(input_final, output);
            serialize_qm31_packed(rsqrt_final, output);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_double_packed(lup, output, false);
                }
                None => serialize_u32(0, output),
            }
        }
        LayerProof::Quantize {
            logup_proof,
            input_eval,
            output_eval,
            table_inputs,
            table_outputs,
        } => {
            serialize_u32(9, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_logup_proof_inner_double_packed(lup, output, true);
                }
                None => serialize_u32(0, output),
            }
            serialize_u32(table_inputs.len() as u32, output);
            for (&inp, &out) in table_inputs.iter().zip(table_outputs.iter()) {
                serialize_m31(inp, output);
                serialize_m31(out, output);
            }
        }
        LayerProof::Embedding {
            logup_proof,
            input_eval,
            output_eval,
            input_num_vars,
        } => {
            serialize_u32(10, output);
            serialize_qm31_packed(*input_eval, output);
            serialize_qm31_packed(*output_eval, output);
            serialize_u32(*input_num_vars as u32, output);
            match logup_proof {
                Some(lup) => {
                    serialize_u32(1, output);
                    serialize_u32(lup.eq_round_polys.len() as u32, output);
                    for rp in &lup.eq_round_polys {
                        // Embedding uses all 4 coefficients
                        serialize_qm31_pair_packed(rp.c0, rp.c1, output);
                        serialize_qm31_pair_packed(rp.c2, rp.c3, output);
                    }
                    serialize_qm31_packed(lup.claimed_sum, output);
                    let (w, tok, col, val) = lup.final_evals;
                    serialize_qm31_packed(w, output);
                    serialize_qm31_packed(tok, output);
                    serialize_qm31_packed(col, output);
                    serialize_qm31_packed(val, output);
                    serialize_u32(lup.table_tokens.len() as u32, output);
                    for i in 0..lup.table_tokens.len() {
                        serialize_u32(lup.table_tokens[i], output);
                        serialize_u32(lup.table_cols[i], output);
                        serialize_u32(lup.multiplicities[i], output);
                    }
                }
                None => serialize_u32(0, output),
            }
        }
    }
}

/// Double-packed variant of serialize_logup_proof_inner_packed.
fn serialize_logup_proof_inner_double_packed(
    lup: &crate::gkr::types::LogUpProof,
    output: &mut Vec<FieldElement>,
    include_multiplicities: bool,
) {
    serialize_qm31_packed(lup.claimed_sum, output);
    serialize_u32(lup.eq_round_polys.len() as u32, output);
    for rp in &lup.eq_round_polys {
        // Degree-3: (c0, c2) as pair, c3 as single
        serialize_qm31_pair_packed(rp.c0, rp.c2, output);
        serialize_qm31_packed(rp.c3, output);
    }
    let (w_eval, in_eval, out_eval) = lup.final_evals;
    serialize_qm31_packed(w_eval, output);
    serialize_qm31_packed(in_eval, output);
    serialize_qm31_packed(out_eval, output);
    if include_multiplicities {
        serialize_u32(lup.multiplicities.len() as u32, output);
        for &mult in &lup.multiplicities {
            serialize_u32(mult, output);
        }
    } else {
        serialize_u32(0, output);
    }
}

/// Packed variant with layer boundary tracking for streaming verification (v25).
///
/// Returns the felt offset at each layer boundary, enabling precise splitting
/// of proof data into batches for streaming calldata-only verification.
///
/// boundaries[i] = felt offset where layer i begins (relative to output start)
/// boundaries[len] = final offset (total felts for all layers)
///
/// Deferred proofs are NOT included (they go in the finalize TX).
pub fn serialize_gkr_proof_data_only_packed_with_boundaries(
    proof: &crate::gkr::GKRProof,
    output: &mut Vec<FieldElement>,
) -> Vec<usize> {
    let base = output.len();
    let mut boundaries = Vec::with_capacity(proof.layer_proofs.len() + 1);
    for layer_proof in &proof.layer_proofs {
        boundaries.push(output.len() - base);
        serialize_layer_proof_packed_inner(layer_proof, output, Some(proof));
    }
    boundaries.push(output.len() - base); // final offset
    boundaries
}

/// Split packed proof data into batches for streaming GKR verification.
///
/// Each batch contains consecutive layers whose total felt count fits within
/// `max_felts_per_batch`. Returns a Vec of (start_layer, end_layer, felt_slice)
/// tuples suitable for building streaming calldata.
pub fn split_proof_into_stream_batches(
    proof: &crate::gkr::GKRProof,
    max_felts_per_batch: usize,
) -> (Vec<FieldElement>, Vec<StreamBatchInfo>) {
    split_proof_into_stream_batches_with_layer_cap(
        proof,
        max_felts_per_batch,
        crate::starknet::MAX_STREAM_BATCH_LAYERS,
    )
}

pub fn split_proof_into_stream_batches_with_layer_cap(
    proof: &crate::gkr::GKRProof,
    max_felts_per_batch: usize,
    max_layers_per_batch: usize,
) -> (Vec<FieldElement>, Vec<StreamBatchInfo>) {
    let mut all_felts = Vec::new();
    let boundaries = serialize_gkr_proof_data_only_packed_with_boundaries(proof, &mut all_felts);

    let num_layers = proof.layer_proofs.len();
    let mut batches = Vec::new();
    let mut batch_start = 0;

    while batch_start < num_layers {
        let mut batch_end = batch_start + 1;
        // Greedily pack consecutive layers (respecting both felt and layer limits)
        while batch_end < num_layers {
            let batch_felts = boundaries[batch_end + 1] - boundaries[batch_start];
            let batch_layers = batch_end - batch_start + 1;
            if batch_felts > max_felts_per_batch || batch_layers > max_layers_per_batch {
                break;
            }
            batch_end += 1;
        }
        // If a single layer exceeds the felt limit, include it anyway (1 layer minimum)
        let felt_start = boundaries[batch_start];
        let felt_end = boundaries[batch_end];
        batches.push(StreamBatchInfo {
            start_layer: batch_start,
            end_layer: batch_end,
            felt_start,
            felt_end,
        });
        batch_start = batch_end;
    }

    (all_felts, batches)
}

/// Metadata for a single streaming batch.
#[derive(Debug, Clone)]
pub struct StreamBatchInfo {
    /// First layer index in this batch (inclusive).
    pub start_layer: usize,
    /// Last layer index in this batch (exclusive).
    pub end_layer: usize,
    /// Byte offset into the flat felt array where this batch starts.
    pub felt_start: usize,
    /// Byte offset into the flat felt array where this batch ends.
    pub felt_end: usize,
}

/// Serialize a STWO-native `GkrBatchProof` into felt252 layout matching Cairo's
/// `GkrBatchProof` Serde deserialization.
///
/// # Format (matches Cairo auto-derived Serde)
///
/// ```text
/// GkrBatchProof:
///   instances: Array<GkrInstance>        // length + per-instance
///   layer_proofs: Array<GkrLayerProof>   // length + per-layer
///
/// GkrInstance:
///   gate: GateType { gate_id: u32 }      // 0=GrandProduct, 1=LogUp
///   n_variables: u32
///   output_claims: Array<QM31>           // length + claims (4 felts each)
///
/// GkrLayerProof:
///   sumcheck_proof: GkrSumcheckProof { round_polys: Array<GkrRoundPoly> }
///   masks: Array<GkrMask>
///
/// GkrRoundPoly:
///   c0, c1, c2, c3: QM31                // 16 felts
///   num_coeffs: u32                       // actual number of nonzero coefficients
///
/// GkrMask:
///   values: Array<QM31>                  // [col0_v0, col0_v1, col1_v0, col1_v1, ...]
///   num_columns: u32
/// ```
pub fn serialize_stwo_gkr_batch_proof(
    proof: &stwo::prover::lookups::gkr_verifier::GkrBatchProof,
    gate_types: &[stwo::prover::lookups::gkr_verifier::Gate],
    n_variables: &[usize],
    output: &mut Vec<FieldElement>,
) {
    use std::ops::Deref;
    use stwo::prover::lookups::gkr_verifier::Gate;

    let n_instances = proof.output_claims_by_instance.len();

    // === instances: Array<GkrInstance> ===
    serialize_u32(n_instances as u32, output);
    for i in 0..n_instances {
        // gate: GateType { gate_id }
        let gate_id = match gate_types[i] {
            Gate::GrandProduct => 0u32,
            Gate::LogUp => 1u32,
        };
        serialize_u32(gate_id, output);

        // n_variables: u32
        serialize_u32(n_variables[i] as u32, output);

        // output_claims: Array<QM31>
        let claims = &proof.output_claims_by_instance[i];
        serialize_u32(claims.len() as u32, output);
        for claim in claims {
            serialize_qm31(*claim, output);
        }
    }

    // === layer_proofs: Array<GkrLayerProof> ===
    let n_layers = proof.sumcheck_proofs.len();
    serialize_u32(n_layers as u32, output);

    for layer_idx in 0..n_layers {
        let sumcheck = &proof.sumcheck_proofs[layer_idx];

        // sumcheck_proof: GkrSumcheckProof { round_polys: Array<GkrRoundPoly> }
        serialize_u32(sumcheck.round_polys.len() as u32, output);
        for poly in &sumcheck.round_polys {
            // GkrRoundPoly { c0, c1, c2, c3, num_coeffs }
            // UnivariatePoly implements Deref<Target=[SecureField]> for coefficients
            let coeffs: &[SecureField] = poly.deref();
            let num_coeffs = coeffs.len();
            let c0 = if num_coeffs > 0 {
                coeffs[0]
            } else {
                SecureField::default()
            };
            let c1 = if num_coeffs > 1 {
                coeffs[1]
            } else {
                SecureField::default()
            };
            let c2 = if num_coeffs > 2 {
                coeffs[2]
            } else {
                SecureField::default()
            };
            let c3 = if num_coeffs > 3 {
                coeffs[3]
            } else {
                SecureField::default()
            };
            serialize_qm31(c0, output);
            serialize_qm31(c1, output);
            serialize_qm31(c2, output);
            serialize_qm31(c3, output);
            serialize_u32(num_coeffs as u32, output);
        }

        // masks: Array<GkrMask>
        // layer_masks_by_instance[instance_idx][layer_idx] gives the mask for that instance at that layer
        serialize_u32(n_instances as u32, output);
        for inst_idx in 0..n_instances {
            let mask = &proof.layer_masks_by_instance[inst_idx][layer_idx];
            let columns = mask.columns();
            // values: Array<QM31> — flattened [col0_v0, col0_v1, col1_v0, col1_v1, ...]
            serialize_u32((columns.len() * 2) as u32, output);
            for [v0, v1] in columns {
                serialize_qm31(*v0, output);
                serialize_qm31(*v1, output);
            }
            // num_columns: u32
            serialize_u32(columns.len() as u32, output);
        }
    }
}

/// Deserialize a receipt hash from a felt252 value.
pub fn felt_to_receipt_hash(felt: &FieldElement) -> [u8; 32] {
    felt.to_bytes_be()
}

/// Convert a FieldElement to M31 (truncating to 31 bits).
pub fn felt_to_m31(felt: &FieldElement) -> M31 {
    let bytes = felt.to_bytes_be();
    // Take the last 4 bytes as u32, then mod P
    let val = u32::from_be_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
    M31::from(val % ((1u32 << 31) - 1))
}

// ============================================================================
// Raw IO Serialization (for IO commitment recomputation on-chain)
// ============================================================================

/// Serialize raw input and output matrices for IO commitment recomputation.
///
/// Layout matches `compute_io_commitment()` in aggregation.rs exactly:
/// `[input_rows, input_cols, input_len, input_data..., output_rows, output_cols, output_len, output_data...]`
///
/// Each M31 value is serialized as 1 felt252.
/// The on-chain verifier hashes this array with Poseidon to recompute the IO commitment.
pub fn serialize_raw_io(
    input: &crate::components::matmul::M31Matrix,
    output: &crate::components::matmul::M31Matrix,
) -> Vec<FieldElement> {
    let mut buf = Vec::with_capacity(6 + input.data.len() + output.data.len());
    buf.push(FieldElement::from(input.rows as u64));
    buf.push(FieldElement::from(input.cols as u64));
    buf.push(FieldElement::from(input.data.len() as u64));
    for &v in &input.data {
        buf.push(FieldElement::from(v.0 as u64));
    }
    buf.push(FieldElement::from(output.rows as u64));
    buf.push(FieldElement::from(output.cols as u64));
    buf.push(FieldElement::from(output.data.len() as u64));
    for &v in &output.data {
        buf.push(FieldElement::from(v.0 as u64));
    }
    buf
}

// ============================================================================
// Weight MLE Opening Serialization
// ============================================================================

/// A weight MLE opening proof for on-chain binding.
///
/// Proves that the weight matrix MLE at the sumcheck challenge point equals a
/// claimed value, using the committed Poseidon Merkle root.
#[derive(Debug, Clone)]
pub struct WeightMleOpening {
    /// MatMul layer index in the circuit.
    pub layer_idx: usize,
    /// Evaluation point (the sumcheck challenge vector).
    pub point: Vec<SecureField>,
    /// MLE evaluation at the point.
    pub value: SecureField,
    /// Poseidon Merkle root of the weight matrix MLE.
    pub commitment: FieldElement,
}

/// Serialize weight MLE openings for on-chain weight binding verification.
///
/// Layout per opening:
/// `[layer_idx(1), point_len(1), point...(4×point_len), value(4), commitment(1)]`
///
/// The Cairo verifier uses these to check that each matmul layer's weight matrix
/// matches the registered commitment at the sumcheck challenge point.
pub fn serialize_weight_mle_openings(
    openings: &[WeightMleOpening],
    output: &mut Vec<FieldElement>,
) {
    serialize_u32(openings.len() as u32, output);
    for opening in openings {
        serialize_usize(opening.layer_idx, output);
        serialize_u32(opening.point.len() as u32, output);
        for &p in &opening.point {
            serialize_qm31(p, output);
        }
        serialize_qm31(opening.value, output);
        output.push(opening.commitment);
    }
}

/// Serialize an aggregated weight binding proof into felt252 layout for Starknet.
pub fn serialize_aggregated_binding_proof(
    proof: &crate::crypto::aggregated_opening::AggregatedWeightBindingProof,
    output: &mut Vec<FieldElement>,
) {
    // config
    serialize_usize(proof.config.selector_bits, output);
    serialize_usize(proof.config.n_max, output);
    serialize_usize(proof.config.m_padded, output);
    serialize_usize(proof.config.n_global, output);
    serialize_usize(proof.config.n_claims, output);

    // sumcheck_round_polys: Vec<(QM31, QM31, QM31)>
    serialize_u32(proof.sumcheck_round_polys.len() as u32, output);
    for (c0, c1, c2) in &proof.sumcheck_round_polys {
        serialize_qm31(*c0, output);
        serialize_qm31(*c1, output);
        serialize_qm31(*c2, output);
    }

    // oracle_eval_at_s: QM31
    serialize_qm31(proof.oracle_eval_at_s, output);

    // opening_proof: MleOpeningProof
    serialize_mle_opening_proof(&proof.opening_proof, output);

    // super_root
    output.push(proof.super_root.root);
    serialize_u32(proof.super_root.subtree_roots.len() as u32, output);
    for root in &proof.super_root.subtree_roots {
        output.push(*root);
    }
    output.push(proof.super_root.zero_tree_root);
    serialize_usize(proof.super_root.top_levels, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_m31() {
        let mut out = Vec::new();
        serialize_m31(M31::from(42), &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], FieldElement::from(42u64));
    }

    #[test]
    fn test_serialize_qm31() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        let val = QM31(
            CM31(M31::from(1), M31::from(2)),
            CM31(M31::from(3), M31::from(4)),
        );
        let mut out = Vec::new();
        serialize_qm31(val, &mut out);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], FieldElement::from(1u64));
        assert_eq!(out[1], FieldElement::from(2u64));
        assert_eq!(out[2], FieldElement::from(3u64));
        assert_eq!(out[3], FieldElement::from(4u64));
    }

    #[test]
    fn test_serialize_qm31_packed_roundtrip() {
        use crate::crypto::poseidon_channel::{securefield_to_felt, felt_to_securefield};
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        // Test with various M31 values including boundary cases
        let test_cases = vec![
            QM31(CM31(M31::from(0), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            QM31(CM31(M31::from(1), M31::from(2)), CM31(M31::from(3), M31::from(4))),
            QM31(
                CM31(M31::from(2147483646), M31::from(1000000)),
                CM31(M31::from(999999), M31::from(12345)),
            ),
        ];

        for val in test_cases {
            let mut out = Vec::new();
            serialize_qm31_packed(val, &mut out);
            assert_eq!(out.len(), 1, "packed QM31 should be 1 felt");

            // Verify round-trip via felt_to_securefield
            let recovered = felt_to_securefield(out[0]);
            assert_eq!(recovered, val, "packed round-trip failed for {:?}", val);

            // Also verify it matches securefield_to_felt
            assert_eq!(out[0], securefield_to_felt(val));
        }
    }

    #[test]
    fn test_packed_vs_unpacked_size() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        let val = QM31(
            CM31(M31::from(100), M31::from(200)),
            CM31(M31::from(300), M31::from(400)),
        );

        let mut unpacked = Vec::new();
        serialize_qm31(val, &mut unpacked);
        assert_eq!(unpacked.len(), 4);

        let mut packed = Vec::new();
        serialize_qm31_packed(val, &mut packed);
        assert_eq!(packed.len(), 1);
    }

    #[test]
    fn test_serialize_blake2s_hash() {
        let mut hash_bytes = [0u8; 32];
        // Set first 4 bytes to [1, 0, 0, 0] → word 0 = 1 (LE)
        hash_bytes[0] = 1;
        // Set bytes 4-7 to [2, 0, 0, 0] → word 1 = 2 (LE)
        hash_bytes[4] = 2;

        let hash = Blake2sHash(hash_bytes);
        let mut out = Vec::new();
        serialize_blake2s_hash(&hash, &mut out);

        assert_eq!(out.len(), 8, "Blake2sHash should serialize to 8 felt252s");
        assert_eq!(out[0], FieldElement::from(1u64), "word 0");
        assert_eq!(out[1], FieldElement::from(2u64), "word 1");
        // Words 2-7 should be 0
        for (i, val) in out.iter().enumerate().take(8).skip(2) {
            assert_eq!(*val, FieldElement::ZERO, "word {i} should be zero");
        }
    }

    #[test]
    fn test_serialize_span_empty() {
        let items: Vec<M31> = vec![];
        let mut out = Vec::new();
        serialize_span(&items, |v, o| serialize_m31(*v, o), &mut out);
        assert_eq!(out.len(), 1, "empty span = length prefix only");
        assert_eq!(out[0], FieldElement::ZERO);
    }

    #[test]
    fn test_serialize_span_with_values() {
        let items = vec![M31::from(10), M31::from(20), M31::from(30)];
        let mut out = Vec::new();
        serialize_span(&items, |v, o| serialize_m31(*v, o), &mut out);
        assert_eq!(out.len(), 4, "3 items + 1 length prefix");
        assert_eq!(out[0], FieldElement::from(3u64));
        assert_eq!(out[1], FieldElement::from(10u64));
    }

    #[test]
    fn test_serialize_pcs_config() {
        let config = PcsConfig::default();
        let mut out = Vec::new();
        serialize_pcs_config(&config, &mut out);
        assert_eq!(out.len(), 4, "PcsConfig = pow_bits + 3 FriConfig fields");
    }

    #[test]
    fn test_serialize_merkle_decommitment_empty() {
        let decommit = MerkleDecommitmentLifted::<Hasher> {
            hash_witness: Vec::new(),
        };
        let mut out = Vec::new();
        serialize_merkle_decommitment(&decommit, &mut out);
        // hash_witness length (0) + column_witness length (0) = 2
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_felt_to_m31_roundtrip() {
        let original = M31::from(12345);
        let mut out = Vec::new();
        serialize_m31(original, &mut out);
        let back = felt_to_m31(&out[0]);
        assert_eq!(back, original);
    }

    #[test]
    fn test_serialize_real_proof_calldata() {
        // Generate a real proof using the activation layer prover
        use crate::compiler::prove::prove_activation_layer;
        use crate::gadgets::lookup_table::activations;
        use crate::gadgets::lookup_table::PrecomputedTable;
        use stwo::prover::backend::simd::SimdBackend;

        let table = PrecomputedTable::build(activations::relu, 4);
        let inputs = vec![M31::from(0), M31::from(1), M31::from(3), M31::from(5)];
        let outputs: Vec<M31> = inputs.iter().map(|&x| activations::relu(x)).collect();

        let config = PcsConfig::default();
        let (_component, proof) = prove_activation_layer::<SimdBackend, Blake2sMerkleChannel>(
            &inputs, &outputs, &table, config,
        )
        .expect("proving should succeed");

        // Serialize the proof
        let calldata = serialize_proof(&proof);

        // Verify it's non-empty and reasonable size
        assert!(!calldata.is_empty(), "calldata should be non-empty");
        assert!(
            calldata.len() > 20,
            "calldata should have meaningful content"
        );
        assert!(
            calldata.len() < 100_000,
            "calldata shouldn't be absurdly large"
        );

        // Verify the first 4 felts are PcsConfig
        let estimated = estimate_calldata_size(&proof);
        assert_eq!(estimated, calldata.len());
    }

    // === Sumcheck + MLE Serialization Tests ===

    #[test]
    fn test_serialize_round_poly() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        let rp = RoundPoly {
            c0: QM31(
                CM31(M31::from(1), M31::from(2)),
                CM31(M31::from(3), M31::from(4)),
            ),
            c1: QM31(
                CM31(M31::from(5), M31::from(6)),
                CM31(M31::from(7), M31::from(8)),
            ),
            c2: QM31(
                CM31(M31::from(9), M31::from(10)),
                CM31(M31::from(11), M31::from(12)),
            ),
        };

        let mut out = Vec::new();
        serialize_round_poly(&rp, &mut out);
        assert_eq!(out.len(), 12, "3 QM31s = 12 felt252s");
        assert_eq!(out[0], FieldElement::from(1u64)); // c0.a
        assert_eq!(out[4], FieldElement::from(5u64)); // c1.a
        assert_eq!(out[8], FieldElement::from(9u64)); // c2.a
    }

    #[test]
    fn test_compressed_round_poly_roundtrip() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        let rp = RoundPoly {
            c0: QM31(
                CM31(M31::from(1), M31::from(2)),
                CM31(M31::from(3), M31::from(4)),
            ),
            c1: QM31(
                CM31(M31::from(5), M31::from(6)),
                CM31(M31::from(7), M31::from(8)),
            ),
            c2: QM31(
                CM31(M31::from(9), M31::from(10)),
                CM31(M31::from(11), M31::from(12)),
            ),
        };

        // Compute the sum that would be the running sumcheck value:
        // current_sum = p(0) + p(1) = c0 + (c0 + c1 + c2)
        let p0 = rp.c0;
        let p1 = rp.c0 + rp.c1 + rp.c2;
        let current_sum = p0 + p1;

        // Compress and decompress
        let compressed = rp.compress();
        let decompressed = compressed.decompress(current_sum);

        assert_eq!(decompressed.c0, rp.c0, "c0 mismatch");
        assert_eq!(decompressed.c1, rp.c1, "c1 mismatch after reconstruction");
        assert_eq!(decompressed.c2, rp.c2, "c2 mismatch");

        // Verify compressed serialization is 8 felts (vs 12 uncompressed)
        let mut out_compressed = Vec::new();
        serialize_round_poly_compressed(&compressed, &mut out_compressed);
        assert_eq!(out_compressed.len(), 8, "compressed = 2 QM31s = 8 felt252s");
        assert_eq!(out_compressed[0], FieldElement::from(1u64)); // c0.a
        assert_eq!(out_compressed[4], FieldElement::from(9u64)); // c2.a (skipped c1)
    }

    #[test]
    fn test_serialize_matmul_sumcheck_field_order() {
        use crate::components::matmul::{matmul_m31, prove_matmul_sumcheck_onchain, M31Matrix};

        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);
        let proof = prove_matmul_sumcheck_onchain(&a, &b, &c).unwrap();

        let mut out = Vec::new();
        serialize_matmul_sumcheck_proof(&proof, &mut out);

        // First 8 felts: m, k, n, num_rounds, claimed_sum(4)
        assert_eq!(out[0], FieldElement::from(2u64)); // m
        assert_eq!(out[1], FieldElement::from(2u64)); // k
        assert_eq!(out[2], FieldElement::from(2u64)); // n
        assert_eq!(out[3], FieldElement::from(1u64)); // num_rounds (log2(2) = 1)
                                                      // out[4..8] = claimed_sum (4 felts for QM31)

        // Total should be non-trivially large
        assert!(
            out.len() > 20,
            "serialized proof too small: {} felts",
            out.len()
        );
    }

    #[test]
    fn test_serialize_mle_opening_proof_structure() {
        use crate::crypto::mle_opening::prove_mle_opening;
        use crate::crypto::poseidon_channel::PoseidonChannel;

        let evals: Vec<SecureField> = (0..4)
            .map(|i| SecureField::from(M31::from((i + 1) as u32)))
            .collect();

        let challenges = vec![
            SecureField::from(M31::from(3)),
            SecureField::from(M31::from(7)),
        ];

        let mut ch = PoseidonChannel::new();
        ch.mix_u64(42);
        let proof = prove_mle_opening(&evals, &challenges, &mut ch);

        let mut out = Vec::new();
        serialize_mle_opening_proof(&proof, &mut out);

        // Should contain: intermediate_roots array, queries array, final_value
        assert!(
            out.len() > 4,
            "serialized MLE proof too small: {} felts",
            out.len()
        );

        // Last 4 felts should be final_value (QM31)
        let fv_start = out.len() - 4;
        let final_a = out[fv_start];
        // Should be non-zero since we used non-zero inputs
        assert_ne!(final_a, FieldElement::ZERO);
    }

    #[test]
    fn test_serialize_ml_proof_for_recursive_structure() {
        use crate::components::matmul::{matmul_m31, prove_matmul_sumcheck_onchain, M31Matrix};

        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));

        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));

        let c = matmul_m31(&a, &b);
        let matmul_proof = prove_matmul_sumcheck_onchain(&a, &b, &c).unwrap();

        let aggregated = AggregatedModelProofOnChain {
            unified_stark: None,
            matmul_proofs: vec![(0, matmul_proof)],
            batched_matmul_proofs: Vec::new(),
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            rmsnorm_claims: Vec::new(),
            execution: crate::compiler::prove::GraphExecution {
                intermediates: vec![],
                output: M31Matrix::new(1, 1),
            },
            activation_claims: vec![],
            attention_proofs: Vec::new(),
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            dequantize_claims: Vec::new(),
            layer_chain_commitment: starknet_ff::FieldElement::ZERO,
            io_commitment: starknet_ff::FieldElement::ZERO,
            layernorm_mean_var_commitments: Vec::new(),
            quantize_params_commitment: starknet_ff::FieldElement::ZERO,
            tiled_matmul_proofs: Vec::new(),
            gkr_proof: None,
            gkr_batch_data: None,
        };

        let metadata = MLClaimMetadata {
            model_id: FieldElement::from(0x42u64),
            num_layers: 1,
            activation_type: 0, // ReLU
            io_commitment: FieldElement::from(0xdeadbeefu64),
            weight_commitment: FieldElement::from(0xcafeu64),
            tee_attestation_hash: None,
        };

        let felts = serialize_ml_proof_for_recursive(&aggregated, &metadata, None);

        // Verify structure:
        // MLClaim: model_id(1) + num_layers(1) + activation_type(1) + io_commitment(1) + weight_commitment(1) = 5
        assert_eq!(felts[0], FieldElement::from(0x42u64), "model_id");
        assert_eq!(felts[1], FieldElement::from(1u64), "num_layers");
        assert_eq!(
            felts[2],
            FieldElement::from(0u64),
            "activation_type (ReLU=0)"
        );
        assert_eq!(felts[3], FieldElement::from(0xdeadbeefu64), "io_commitment");
        assert_eq!(felts[4], FieldElement::from(0xcafeu64), "weight_commitment");

        // matmul_proofs: length = 1
        assert_eq!(felts[5], FieldElement::from(1u64), "matmul_proofs length");

        // First matmul proof starts at index 6: m=2, k=2, n=2, num_rounds=1
        assert_eq!(felts[6], FieldElement::from(2u64), "m");
        assert_eq!(felts[7], FieldElement::from(2u64), "k");
        assert_eq!(felts[8], FieldElement::from(2u64), "n");
        assert_eq!(felts[9], FieldElement::from(1u64), "num_rounds");

        // Total should be reasonable
        assert!(
            felts.len() > 20,
            "serialized too small: {} felts",
            felts.len()
        );

        // Trailing sections: unified_stark(0), add_claims(0), mul_claims(0), layernorm_claims(0), tee(0)
        assert_eq!(
            felts[felts.len() - 5],
            FieldElement::ZERO,
            "unified_stark_proof = None"
        );
        assert_eq!(
            felts[felts.len() - 4],
            FieldElement::ZERO,
            "add_claims count = 0"
        );
        assert_eq!(
            felts[felts.len() - 3],
            FieldElement::ZERO,
            "mul_claims count = 0"
        );
        assert_eq!(
            felts[felts.len() - 2],
            FieldElement::ZERO,
            "layernorm_claims count = 0"
        );
        assert_eq!(
            felts[felts.len() - 1],
            FieldElement::ZERO,
            "tee_attestation_hash = None"
        );

        // Test arguments file generation
        let json = serialize_ml_proof_to_arguments_file(&felts);
        assert!(json.starts_with("[\"0x"));
        assert!(json.ends_with("\"]"));
    }

    #[test]
    fn test_serialize_ml_proof_with_salt() {
        use crate::components::matmul::{matmul_m31, prove_matmul_sumcheck_onchain, M31Matrix};

        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(0));
        a.set(1, 0, M31::from(0));
        a.set(1, 1, M31::from(1));
        let c = matmul_m31(&a, &a);
        let matmul_proof = prove_matmul_sumcheck_onchain(&a, &a, &c).unwrap();

        let aggregated = AggregatedModelProofOnChain {
            unified_stark: None,
            matmul_proofs: vec![(0, matmul_proof)],
            batched_matmul_proofs: Vec::new(),
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            rmsnorm_claims: Vec::new(),
            execution: crate::compiler::prove::GraphExecution {
                intermediates: vec![],
                output: M31Matrix::new(1, 1),
            },
            activation_claims: vec![],
            attention_proofs: Vec::new(),
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            dequantize_claims: Vec::new(),
            layer_chain_commitment: starknet_ff::FieldElement::ZERO,
            io_commitment: starknet_ff::FieldElement::ZERO,
            layernorm_mean_var_commitments: Vec::new(),
            quantize_params_commitment: starknet_ff::FieldElement::ZERO,
            tiled_matmul_proofs: Vec::new(),
            gkr_proof: None,
            gkr_batch_data: None,
        };

        let metadata = MLClaimMetadata {
            model_id: FieldElement::ONE,
            num_layers: 1,
            activation_type: 0,
            io_commitment: FieldElement::ZERO,
            weight_commitment: FieldElement::ZERO,
            tee_attestation_hash: None,
        };

        // With salt
        let with_salt = serialize_ml_proof_for_recursive(&aggregated, &metadata, Some(12345));
        // Without salt
        let no_salt = serialize_ml_proof_for_recursive(&aggregated, &metadata, None);

        // Salt adds 1 extra felt (variant index 1 + value) vs (variant index 0)
        assert_eq!(with_salt.len(), no_salt.len() + 1);

        // Both end with: unified_stark(0), add_claims(0), mul_claims(0), layernorm_claims(0), embedding(0), attention(0), tee(0)
        assert_eq!(no_salt[no_salt.len() - 1], FieldElement::ZERO, "tee = None");
        assert_eq!(
            no_salt[no_salt.len() - 2],
            FieldElement::ZERO,
            "attention_proofs = 0"
        );
        assert_eq!(
            no_salt[no_salt.len() - 3],
            FieldElement::ZERO,
            "embedding_claims = 0"
        );
        assert_eq!(
            no_salt[no_salt.len() - 4],
            FieldElement::ZERO,
            "layernorm_claims = 0"
        );
        assert_eq!(
            no_salt[no_salt.len() - 5],
            FieldElement::ZERO,
            "mul_claims = 0"
        );
        assert_eq!(
            no_salt[no_salt.len() - 6],
            FieldElement::ZERO,
            "add_claims = 0"
        );
        assert_eq!(
            no_salt[no_salt.len() - 7],
            FieldElement::ZERO,
            "unified_stark = None"
        );

        assert_eq!(
            with_salt[with_salt.len() - 1],
            FieldElement::ZERO,
            "tee = None"
        );
        assert_eq!(
            with_salt[with_salt.len() - 4],
            FieldElement::ZERO,
            "layernorm_claims = 0"
        );
        assert_eq!(
            with_salt[with_salt.len() - 7],
            FieldElement::ZERO,
            "unified_stark = None"
        );

        // no_salt layout ends: ..., channel_salt=None(0), unified_stark=None(0), add(0), mul(0), layernorm(0), embedding(0), attention(0), tee(0)
        assert_eq!(
            no_salt[no_salt.len() - 8],
            FieldElement::ZERO,
            "channel_salt = None"
        );

        // with_salt layout ends: ..., channel_salt=Some(1), 12345, unified_stark=None(0), add(0), mul(0), layernorm(0), embedding(0), attention(0), tee(0)
        assert_eq!(
            with_salt[with_salt.len() - 9],
            FieldElement::from(1u64),
            "channel_salt = Some"
        );
        assert_eq!(
            with_salt[with_salt.len() - 8],
            FieldElement::from(12345u64),
            "salt value"
        );
    }

    #[test]
    fn test_serialize_activation_claim_layout() {
        let claim = ActivationClaimForSerde {
            layer_index: 3,
            log_size: 10,
            activation_type: 1, // GELU
        };

        let mut out = Vec::new();
        serialize_activation_claim(&claim, &mut out);

        assert_eq!(out.len(), 3, "ActivationClaim = 3 felt252s");
        assert_eq!(out[0], FieldElement::from(3u64), "layer_index");
        assert_eq!(out[1], FieldElement::from(10u64), "log_size");
        assert_eq!(out[2], FieldElement::from(1u64), "activation_type (GELU=1)");
    }

    #[test]
    fn test_serialize_activation_claim_from_layer_claim() {
        let layer_claim = LayerClaim {
            layer_index: 2,
            claimed_sum: SecureField::from(M31::from(42)),
            trace_rows: 1024, // ilog2(1024) = 10
        };

        let serde_claim = ActivationClaimForSerde::from_layer_claim(&layer_claim, 0);
        assert_eq!(serde_claim.layer_index, 2);
        assert_eq!(serde_claim.log_size, 10);
        assert_eq!(serde_claim.activation_type, 0);

        let mut out = Vec::new();
        serialize_activation_claim(&serde_claim, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], FieldElement::from(2u64));
        assert_eq!(out[1], FieldElement::from(10u64));
        assert_eq!(out[2], FieldElement::from(0u64));
    }

    #[test]
    fn test_serialize_ml_proof_activation_none_vs_some_size() {
        use crate::compiler::prove::prove_activation_layer;
        use crate::components::matmul::{matmul_m31, prove_matmul_sumcheck_onchain, M31Matrix};
        use crate::gadgets::lookup_table::activations;
        use crate::gadgets::lookup_table::PrecomputedTable;
        use stwo::prover::backend::simd::SimdBackend;

        // Create a matmul proof for the base
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));
        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));
        let c_mat = matmul_m31(&a, &b);
        let matmul_proof = prove_matmul_sumcheck_onchain(&a, &b, &c_mat).unwrap();

        // Create a real activation STARK proof
        let table = PrecomputedTable::build(activations::relu, 4);
        let inputs = vec![M31::from(0), M31::from(1), M31::from(3), M31::from(5)];
        let outputs: Vec<M31> = inputs.iter().map(|&x| activations::relu(x)).collect();
        let config = PcsConfig::default();
        let (_component, activation_proof) = prove_activation_layer::<
            SimdBackend,
            Blake2sMerkleChannel,
        >(&inputs, &outputs, &table, config)
        .expect("proving should succeed");

        let metadata = MLClaimMetadata {
            model_id: FieldElement::from(0x42u64),
            num_layers: 1,
            activation_type: 0,
            io_commitment: FieldElement::ZERO,
            weight_commitment: FieldElement::ZERO,
            tee_attestation_hash: None,
        };

        // Without activation STARK
        let aggregated_none = AggregatedModelProofOnChain {
            unified_stark: None,
            matmul_proofs: vec![(0, matmul_proof.clone())],
            batched_matmul_proofs: Vec::new(),
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            rmsnorm_claims: Vec::new(),
            execution: crate::compiler::prove::GraphExecution {
                intermediates: vec![],
                output: M31Matrix::new(1, 1),
            },
            activation_claims: vec![],
            attention_proofs: Vec::new(),
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            dequantize_claims: Vec::new(),
            layer_chain_commitment: starknet_ff::FieldElement::ZERO,
            io_commitment: starknet_ff::FieldElement::ZERO,
            layernorm_mean_var_commitments: Vec::new(),
            quantize_params_commitment: starknet_ff::FieldElement::ZERO,
            tiled_matmul_proofs: Vec::new(),
            gkr_proof: None,
            gkr_batch_data: None,
        };
        let felts_none = serialize_ml_proof_for_recursive(&aggregated_none, &metadata, None);

        // With activation STARK
        let activation_claims = vec![LayerClaim {
            layer_index: 0,
            claimed_sum: SecureField::default(),
            trace_rows: 4,
        }];
        let aggregated_some = AggregatedModelProofOnChain {
            unified_stark: Some(activation_proof),
            matmul_proofs: vec![(0, matmul_proof)],
            batched_matmul_proofs: Vec::new(),
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            rmsnorm_claims: Vec::new(),
            execution: crate::compiler::prove::GraphExecution {
                intermediates: vec![],
                output: M31Matrix::new(1, 1),
            },
            activation_claims,
            attention_proofs: Vec::new(),
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            dequantize_claims: Vec::new(),
            layer_chain_commitment: starknet_ff::FieldElement::ZERO,
            io_commitment: starknet_ff::FieldElement::ZERO,
            layernorm_mean_var_commitments: Vec::new(),
            quantize_params_commitment: starknet_ff::FieldElement::ZERO,
            tiled_matmul_proofs: Vec::new(),
            gkr_proof: None,
            gkr_batch_data: None,
        };
        let felts_some = serialize_ml_proof_for_recursive(&aggregated_some, &metadata, None);

        // Some(activation) should be substantially larger than None
        assert!(
            felts_some.len() > felts_none.len() + 10,
            "Some(activation) should add many felts: none={}, some={}",
            felts_none.len(),
            felts_some.len(),
        );

        // None ends with: channel_salt=None(0), unified_stark=None(0), add(0), mul(0), layernorm(0), embedding(0), attention(0), tee(0)
        assert_eq!(
            felts_none[felts_none.len() - 1],
            FieldElement::ZERO,
            "tee = None"
        );
        assert_eq!(
            felts_none[felts_none.len() - 4],
            FieldElement::ZERO,
            "layernorm_claims = 0"
        );
        assert_eq!(
            felts_none[felts_none.len() - 7],
            FieldElement::ZERO,
            "unified_stark = None"
        );
        assert_eq!(
            felts_none[felts_none.len() - 8],
            FieldElement::ZERO,
            "channel_salt = None"
        );

        // Some path: both share the same prefix up to channel_salt
        // The None path is: [MLClaim(5) + matmul_array + batched(0) + channel_salt(1) + activation_none(1) + add(0) + mul(0) + layernorm(0) + embedding(0) + attention(0) + tee(0)]
        // The Some path is: [MLClaim(5) + matmul_array + batched(0) + channel_salt(1) + activation_some(1) + data... + add(0) + mul(0) + layernorm(0) + embedding(0) + attention(0) + tee(0)]
        // The divergence point is at felts_none.len() - 7 (activation discriminant)
        let diverge_idx = felts_none.len() - 7;
        assert_eq!(
            felts_none[diverge_idx],
            FieldElement::ZERO,
            "None discriminant"
        );
        assert_eq!(
            felts_some[diverge_idx],
            FieldElement::from(1u64),
            "Some discriminant"
        );
    }

    #[test]
    fn test_serialize_unified_stark_proof_with_real_proof() {
        use crate::compiler::prove::prove_activation_layer;
        use crate::components::matmul::{matmul_m31, prove_matmul_sumcheck_onchain, M31Matrix};
        use crate::gadgets::lookup_table::activations;
        use crate::gadgets::lookup_table::PrecomputedTable;
        use stwo::prover::backend::simd::SimdBackend;

        // Create matmul proof
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));
        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));
        let c_mat = matmul_m31(&a, &b);
        let matmul_proof = prove_matmul_sumcheck_onchain(&a, &b, &c_mat).unwrap();

        // Create real activation STARK proof
        let table = PrecomputedTable::build(activations::relu, 4);
        let inputs = vec![M31::from(0), M31::from(1), M31::from(3), M31::from(5)];
        let outputs: Vec<M31> = inputs.iter().map(|&x| activations::relu(x)).collect();
        let config = PcsConfig::default();
        let (_component, activation_proof) = prove_activation_layer::<
            SimdBackend,
            Blake2sMerkleChannel,
        >(&inputs, &outputs, &table, config)
        .expect("proving should succeed");

        let activation_claims = vec![LayerClaim {
            layer_index: 0,
            claimed_sum: SecureField::default(),
            trace_rows: 4,
        }];

        let aggregated = AggregatedModelProofOnChain {
            unified_stark: Some(activation_proof),
            matmul_proofs: vec![(0, matmul_proof)],
            batched_matmul_proofs: Vec::new(),
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            rmsnorm_claims: Vec::new(),
            execution: crate::compiler::prove::GraphExecution {
                intermediates: vec![],
                output: M31Matrix::new(1, 1),
            },
            activation_claims,
            attention_proofs: Vec::new(),
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            dequantize_claims: Vec::new(),
            layer_chain_commitment: starknet_ff::FieldElement::ZERO,
            io_commitment: starknet_ff::FieldElement::ZERO,
            layernorm_mean_var_commitments: Vec::new(),
            quantize_params_commitment: starknet_ff::FieldElement::ZERO,
            tiled_matmul_proofs: Vec::new(),
            gkr_proof: None,
            gkr_batch_data: None,
        };

        let metadata = MLClaimMetadata {
            model_id: FieldElement::from(0xABCDu64),
            num_layers: 1,
            activation_type: 0, // ReLU
            io_commitment: FieldElement::from(0x111u64),
            weight_commitment: FieldElement::from(0x222u64),
            tee_attestation_hash: None,
        };

        let felts = serialize_ml_proof_for_recursive(&aggregated, &metadata, None);

        // === Verify full MLProof structure ===

        // Field 1: MLClaim (5 felts)
        let mut idx = 0;
        assert_eq!(felts[idx], FieldElement::from(0xABCDu64), "model_id");
        idx += 1;
        assert_eq!(felts[idx], FieldElement::from(1u64), "num_layers");
        idx += 1;
        assert_eq!(
            felts[idx],
            FieldElement::from(0u64),
            "activation_type (ReLU=0)"
        );
        idx += 1;
        assert_eq!(felts[idx], FieldElement::from(0x111u64), "io_commitment");
        idx += 1;
        assert_eq!(
            felts[idx],
            FieldElement::from(0x222u64),
            "weight_commitment"
        );
        idx += 1;

        // Field 2: matmul_proofs array (length=1)
        assert_eq!(felts[idx], FieldElement::from(1u64), "matmul_proofs length");
        idx += 1;
        // Skip over the matmul proof (m=2, k=2, n=2, ...)
        assert_eq!(felts[idx], FieldElement::from(2u64), "matmul m");
        // Advance past the entire matmul proof to find channel_salt
        // 10-field layout: m(1) + k(1) + n(1) + num_rounds(1) + claimed_sum(4)
        //   + round_polys_len(1) + round_polys(num_rounds*12) + final_a(4) + final_b(4)
        //   + a_commit(1) + b_commit(1)
        let num_rounds = 1u32; // log2(2) = 1
        let matmul_felts = 4 + 4 + 1 + (num_rounds as usize * 12) + 4 + 4 + 1 + 1;
        idx += matmul_felts;

        // Field 2b: batched_matmul_proofs array (length=0)
        assert_eq!(
            felts[idx],
            FieldElement::ZERO,
            "batched_matmul_proofs length = 0"
        );
        idx += 1;

        // Field 3: channel_salt = Option::None (1 felt = 0)
        assert_eq!(felts[idx], FieldElement::ZERO, "channel_salt = None");
        idx += 1;

        // Field 4: unified_stark_proof = Option::Some (discriminant = 1)
        assert_eq!(felts[idx], FieldElement::from(1u64), "unified_stark = Some");
        idx += 1;

        // Inside UnifiedStarkProof:
        // 4.1: activation_claims array (length=1)
        assert_eq!(
            felts[idx],
            FieldElement::from(1u64),
            "activation_claims length"
        );
        idx += 1;
        // Each ActivationClaim = 3 felts: layer_index(0), log_size(ilog2(4)=2), activation_type(0)
        assert_eq!(felts[idx], FieldElement::from(0u64), "claim.layer_index");
        idx += 1;
        assert_eq!(
            felts[idx],
            FieldElement::from(2u64),
            "claim.log_size (ilog2(4))"
        );
        idx += 1;
        assert_eq!(
            felts[idx],
            FieldElement::from(0u64),
            "claim.activation_type (ReLU)"
        );
        idx += 1;

        // 4.2: activation_interaction_claims array (length=1)
        assert_eq!(
            felts[idx],
            FieldElement::from(1u64),
            "activation_interaction_claims length"
        );
        idx += 1;
        // Each interaction claim = 4 felts (QM31 claimed_sum = default = 0,0,0,0)
        for i in 0..4 {
            assert_eq!(
                felts[idx + i],
                FieldElement::ZERO,
                "activation_interaction_claim[{i}]"
            );
        }
        idx += 4;

        // 4.3: add_claims array (length=0, empty for this proof)
        assert_eq!(felts[idx], FieldElement::ZERO, "add_claims length = 0");
        idx += 1;

        // 4.4: mul_claims array (length=0, empty for this proof)
        assert_eq!(felts[idx], FieldElement::ZERO, "mul_claims length = 0");
        idx += 1;

        // 4.5: layernorm_claims array (length=0, empty for this proof)
        assert_eq!(
            felts[idx],
            FieldElement::ZERO,
            "layernorm_claims length = 0"
        );
        idx += 1;

        // 4.6: layernorm_interaction_claims array (length=0, empty for this proof)
        assert_eq!(
            felts[idx],
            FieldElement::ZERO,
            "layernorm_interaction_claims length = 0"
        );
        idx += 1;

        // 4.7: embedding_claims array (length=0, empty for this proof)
        assert_eq!(
            felts[idx],
            FieldElement::ZERO,
            "embedding_claims length = 0"
        );
        idx += 1;

        // 4.8: interaction_claim: MLInteractionClaim = 4 felts (sum of all LogUp claimed_sums = 0)
        for i in 0..4 {
            assert_eq!(
                felts[idx + i],
                FieldElement::ZERO,
                "ml_interaction_claim[{i}]"
            );
        }
        idx += 4;

        // 4.9: pcs_config = 4 felts (pow_bits + fri_config)
        let default_config = PcsConfig::default();
        assert_eq!(
            felts[idx],
            FieldElement::from(default_config.pow_bits as u64),
            "pcs_config.pow_bits"
        );
        idx += 1;
        assert_eq!(
            felts[idx],
            FieldElement::from(default_config.fri_config.log_blowup_factor as u64),
            "fri_config.log_blowup_factor"
        );
        idx += 1;
        assert_eq!(
            felts[idx],
            FieldElement::from(default_config.fri_config.log_last_layer_degree_bound as u64),
            "fri_config.log_last_layer_deg"
        );
        idx += 1;
        assert_eq!(
            felts[idx],
            FieldElement::from(default_config.fri_config.n_queries as u64),
            "fri_config.n_queries"
        );
        idx += 1;

        // 4.10: interaction_pow = 0 (u64)
        assert_eq!(felts[idx], FieldElement::ZERO, "interaction_pow = 0");
        idx += 1;

        // 4.11: stark_proof (CommitmentSchemeProof) — starts with PcsConfig again
        assert_eq!(
            felts[idx],
            FieldElement::from(default_config.pow_bits as u64),
            "stark_proof config.pow_bits"
        );
        idx += 4; // skip past the 4 PcsConfig felts

        // Remaining felts are the rest of CommitmentSchemeProof (commitments, sampled_values, etc.)
        assert!(
            felts.len() > idx + 10,
            "STARK proof should have substantial remaining data: idx={}, total={}",
            idx,
            felts.len(),
        );
    }

    #[test]
    fn test_serialize_elementwise_claim() {
        let claim = LayerClaim {
            layer_index: 3,
            claimed_sum: SecureField::default(),
            trace_rows: 16,
        };
        let mut output = Vec::new();
        serialize_elementwise_claim(&claim, &mut output);

        // Layout: layer_index(1) + trace_rows(1)
        assert_eq!(output.len(), 2, "elementwise claim = 2 felt252s");
        assert_eq!(output[0], FieldElement::from(3u64), "layer_index");
        assert_eq!(output[1], FieldElement::from(16u64), "trace_rows");
    }

    #[test]
    fn test_serialize_layernorm_claim() {
        let claim = LayerClaim {
            layer_index: 7,
            claimed_sum: SecureField::from(M31::from(42)),
            trace_rows: 16,
        };
        let mut output = Vec::new();
        serialize_layernorm_claim(&claim, &mut output);

        // Layout: layer_index(1) + trace_rows(1) + claimed_sum(4)
        assert_eq!(output.len(), 6, "layernorm claim = 6 felt252s");
        assert_eq!(output[0], FieldElement::from(7u64), "layer_index");
        assert_eq!(output[1], FieldElement::from(16u64), "trace_rows");
        // claimed_sum at [2..6] — first component should be 42
        assert_eq!(output[2], FieldElement::from(42u64), "claimed_sum[0]");
    }

    #[test]
    fn test_serialize_ml_proof_with_add() {
        use crate::aggregation::prove_model_aggregated_onchain;
        use crate::compiler::graph::GraphBuilder;
        use crate::components::activation::ActivationType;
        use crate::components::matmul::M31Matrix;

        // Build a model with a residual Add
        let mut builder = GraphBuilder::new((1, 8));
        builder.linear(8);
        let branch = builder.fork();
        builder.activation(ActivationType::ReLU);
        builder.linear(8);
        builder.add_from(branch);
        builder.linear(4);
        let graph = builder.build();

        let mut input = M31Matrix::new(1, 8);
        for j in 0..8 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut weights = crate::compiler::graph::GraphWeights::new();
        let mut w0 = M31Matrix::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                w0.set(i, j, M31::from(((i + j) % 5 + 1) as u32));
            }
        }
        weights.add_weight(0, w0);
        let mut w2 = M31Matrix::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                w2.set(i, j, M31::from(((i * j) % 7 + 1) as u32));
            }
        }
        weights.add_weight(2, w2);
        let mut w4 = M31Matrix::new(8, 4);
        for i in 0..8 {
            for j in 0..4 {
                w4.set(i, j, M31::from((i + j + 1) as u32));
            }
        }
        weights.add_weight(4, w4);

        let proof = prove_model_aggregated_onchain(&graph, &input, &weights)
            .expect("on-chain aggregated proving with Add should succeed");

        assert_eq!(proof.add_claims.len(), 1, "should have 1 Add claim");

        let metadata = MLClaimMetadata {
            model_id: FieldElement::from(0x42u64),
            num_layers: 5,
            activation_type: 0,
            io_commitment: FieldElement::ZERO,
            weight_commitment: FieldElement::ZERO,
            tee_attestation_hash: None,
        };

        let felts = serialize_ml_proof_for_recursive(&proof, &metadata, None);

        // Trailing sections: add_claims(1 claim), mul_claims(empty), layernorm_claims(empty), embedding(empty), attention(empty), tee(None)
        // Claims are lightweight: add_claim = layer_index(1) + trace_rows(1) = 2 felts per claim
        let last = felts.len() - 1;
        assert_eq!(
            felts[last],
            FieldElement::ZERO,
            "tee_attestation_hash = None"
        );
        assert_eq!(
            felts[last - 1],
            FieldElement::ZERO,
            "attention_proofs count = 0"
        );
        assert_eq!(
            felts[last - 2],
            FieldElement::ZERO,
            "embedding_claims count = 0"
        );
        assert_eq!(
            felts[last - 3],
            FieldElement::ZERO,
            "layernorm_claims count = 0"
        );
        assert_eq!(felts[last - 4], FieldElement::ZERO, "mul_claims count = 0");
        // add_claims section: count(1) + 1 claim * 2 felts = 3 felts before mul_claims
        // add_claims count should be 1
        assert_eq!(
            felts[last - 7],
            FieldElement::from(1u64),
            "add_claims count = 1"
        );
        assert!(felts.len() > 20, "should have data: {} felts", felts.len());
    }

    #[test]
    fn test_serialize_empty_add_mul_layernorm_backward_compat() {
        use crate::components::matmul::{matmul_m31, prove_matmul_sumcheck_onchain, M31Matrix};

        // Build a matmul-only proof (no add/mul/layernorm)
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));
        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));
        let c = matmul_m31(&a, &b);
        let matmul_proof = prove_matmul_sumcheck_onchain(&a, &b, &c).unwrap();

        let aggregated = AggregatedModelProofOnChain {
            unified_stark: None,
            matmul_proofs: vec![(0, matmul_proof)],
            batched_matmul_proofs: Vec::new(),
            add_claims: Vec::new(),
            mul_claims: Vec::new(),
            layernorm_claims: Vec::new(),
            rmsnorm_claims: Vec::new(),
            execution: crate::compiler::prove::GraphExecution {
                intermediates: vec![],
                output: M31Matrix::new(1, 1),
            },
            activation_claims: vec![],
            attention_proofs: Vec::new(),
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            dequantize_claims: Vec::new(),
            layer_chain_commitment: starknet_ff::FieldElement::ZERO,
            io_commitment: starknet_ff::FieldElement::ZERO,
            layernorm_mean_var_commitments: Vec::new(),
            quantize_params_commitment: starknet_ff::FieldElement::ZERO,
            tiled_matmul_proofs: Vec::new(),
            gkr_proof: None,
            gkr_batch_data: None,
        };

        let metadata = MLClaimMetadata {
            model_id: FieldElement::ONE,
            num_layers: 1,
            activation_type: 0,
            io_commitment: FieldElement::ZERO,
            weight_commitment: FieldElement::ZERO,
            tee_attestation_hash: None,
        };

        let felts = serialize_ml_proof_for_recursive(&aggregated, &metadata, None);

        // Last 6 felts should be [0, 0, 0, 0, 0, 0] (empty add, mul, layernorm, embedding, attention, tee)
        let len = felts.len();
        assert_eq!(felts[len - 6], FieldElement::ZERO, "add_claims count = 0");
        assert_eq!(felts[len - 5], FieldElement::ZERO, "mul_claims count = 0");
        assert_eq!(
            felts[len - 4],
            FieldElement::ZERO,
            "layernorm_claims count = 0"
        );
        assert_eq!(
            felts[len - 3],
            FieldElement::ZERO,
            "embedding_claims count = 0"
        );
        assert_eq!(
            felts[len - 2],
            FieldElement::ZERO,
            "attention_proofs count = 0"
        );
        assert_eq!(
            felts[len - 1],
            FieldElement::ZERO,
            "tee_attestation_hash = None"
        );

        // unified_stark=None(0), add(0), mul(0), layernorm(0), embedding(0), attention(0), tee(0)
        assert_eq!(felts[len - 7], FieldElement::ZERO, "unified_stark = None");
    }

    #[test]
    fn test_serialize_unified_stark_with_add_claims() {
        use crate::compiler::prove::prove_activation_layer;
        use crate::components::matmul::{matmul_m31, prove_matmul_sumcheck_onchain, M31Matrix};
        use crate::gadgets::lookup_table::activations;
        use crate::gadgets::lookup_table::PrecomputedTable;
        use stwo::prover::backend::simd::SimdBackend;

        // Create matmul proof
        let mut a = M31Matrix::new(2, 2);
        a.set(0, 0, M31::from(1));
        a.set(0, 1, M31::from(2));
        a.set(1, 0, M31::from(3));
        a.set(1, 1, M31::from(4));
        let mut b = M31Matrix::new(2, 2);
        b.set(0, 0, M31::from(5));
        b.set(0, 1, M31::from(6));
        b.set(1, 0, M31::from(7));
        b.set(1, 1, M31::from(8));
        let c_mat = matmul_m31(&a, &b);
        let matmul_proof = prove_matmul_sumcheck_onchain(&a, &b, &c_mat).unwrap();

        // Create real activation STARK proof
        let table = PrecomputedTable::build(activations::relu, 4);
        let inputs = vec![M31::from(0), M31::from(1), M31::from(3), M31::from(5)];
        let outputs: Vec<M31> = inputs.iter().map(|&x| activations::relu(x)).collect();
        let config = PcsConfig::default();
        let (_component, activation_proof) = prove_activation_layer::<
            SimdBackend,
            Blake2sMerkleChannel,
        >(&inputs, &outputs, &table, config)
        .expect("proving should succeed");

        let activation_claims = vec![LayerClaim {
            layer_index: 1,
            claimed_sum: SecureField::from(M31::from(10)),
            trace_rows: 4,
        }];

        let add_claims = vec![LayerClaim {
            layer_index: 3,
            claimed_sum: SecureField::default(), // pure AIR
            trace_rows: 8,
        }];

        let mul_claims = vec![LayerClaim {
            layer_index: 5,
            claimed_sum: SecureField::default(), // pure AIR
            trace_rows: 16,
        }];

        let layernorm_claims = vec![LayerClaim {
            layer_index: 7,
            claimed_sum: SecureField::from(M31::from(20)),
            trace_rows: 4,
        }];

        let aggregated = AggregatedModelProofOnChain {
            unified_stark: Some(activation_proof),
            matmul_proofs: vec![(0, matmul_proof)],
            batched_matmul_proofs: Vec::new(),
            add_claims: add_claims.clone(),
            mul_claims: mul_claims.clone(),
            layernorm_claims: layernorm_claims.clone(),
            rmsnorm_claims: Vec::new(),
            execution: crate::compiler::prove::GraphExecution {
                intermediates: vec![],
                output: M31Matrix::new(1, 1),
            },
            activation_claims: activation_claims.clone(),
            attention_proofs: Vec::new(),
            embedding_claims: Vec::new(),
            quantize_claims: Vec::new(),
            dequantize_claims: Vec::new(),
            layer_chain_commitment: starknet_ff::FieldElement::ZERO,
            io_commitment: starknet_ff::FieldElement::ZERO,
            layernorm_mean_var_commitments: Vec::new(),
            quantize_params_commitment: starknet_ff::FieldElement::ZERO,
            tiled_matmul_proofs: Vec::new(),
            gkr_proof: None,
            gkr_batch_data: None,
        };

        let metadata = MLClaimMetadata {
            model_id: FieldElement::from(0x77u64),
            num_layers: 8,
            activation_type: 0,
            io_commitment: FieldElement::ZERO,
            weight_commitment: FieldElement::ZERO,
            tee_attestation_hash: None,
        };

        let felts = serialize_ml_proof_for_recursive(&aggregated, &metadata, None);

        // Find the unified_stark section: skip MLClaim(5) + matmul array + channel_salt + discriminant
        // After MLClaim(5), matmul_proofs(len+proof), channel_salt(1), discriminant(1)
        // we enter the unified STARK proof. Walk from the start:
        let mut idx = 5; // past MLClaim
                         // matmul_proofs: 1 + proof felts
        idx += 1; // matmul array length
        let num_rounds = 1u32;
        let matmul_felts = 4 + 4 + 1 + (num_rounds as usize * 12) + 4 + 4 + 1 + 1;
        idx += matmul_felts;
        // batched_matmul_proofs: length=0
        idx += 1;
        // channel_salt: None = 1 felt
        idx += 1;
        // unified_stark discriminant: Some = 1 felt
        assert_eq!(felts[idx], FieldElement::from(1u64), "unified_stark = Some");
        idx += 1;

        // === Inside UnifiedStarkProof ===

        // 1. activation_claims: length=1
        assert_eq!(
            felts[idx],
            FieldElement::from(1u64),
            "activation_claims len"
        );
        idx += 1;
        // ActivationClaim: layer_index(1), log_size(ilog2(4)=2), activation_type(0)
        assert_eq!(
            felts[idx],
            FieldElement::from(1u64),
            "act claim layer_index"
        );
        idx += 1;
        assert_eq!(felts[idx], FieldElement::from(2u64), "act claim log_size");
        idx += 1;
        assert_eq!(
            felts[idx],
            FieldElement::from(0u64),
            "act claim activation_type"
        );
        idx += 1;

        // 2. activation_interaction_claims: length=1, 4 felts
        assert_eq!(felts[idx], FieldElement::from(1u64), "act interaction len");
        idx += 1;
        // claimed_sum = SecureField::from(M31::from(10)) = (10, 0, 0, 0)
        assert_eq!(
            felts[idx],
            FieldElement::from(10u64),
            "act interaction sum[0]"
        );
        idx += 4;

        // 3. add_claims: length=1
        assert_eq!(felts[idx], FieldElement::from(1u64), "add_claims len");
        idx += 1;
        // ElementwiseComponentClaim: layer_index(3), log_size(ilog2(8)=3)
        assert_eq!(
            felts[idx],
            FieldElement::from(3u64),
            "add claim layer_index"
        );
        idx += 1;
        assert_eq!(felts[idx], FieldElement::from(3u64), "add claim log_size");
        idx += 1;

        // 4. mul_claims: length=1
        assert_eq!(felts[idx], FieldElement::from(1u64), "mul_claims len");
        idx += 1;
        // ElementwiseComponentClaim: layer_index(5), log_size(ilog2(16)=4)
        assert_eq!(
            felts[idx],
            FieldElement::from(5u64),
            "mul claim layer_index"
        );
        idx += 1;
        assert_eq!(felts[idx], FieldElement::from(4u64), "mul claim log_size");
        idx += 1;

        // 5. layernorm_claims: length=1
        assert_eq!(felts[idx], FieldElement::from(1u64), "layernorm_claims len");
        idx += 1;
        // LayerNormComponentClaim: layer_index(7), log_size(ilog2(4)=2)
        assert_eq!(felts[idx], FieldElement::from(7u64), "ln claim layer_index");
        idx += 1;
        assert_eq!(felts[idx], FieldElement::from(2u64), "ln claim log_size");
        idx += 1;

        // 6. layernorm_interaction_claims: length=1, 4 felts
        assert_eq!(felts[idx], FieldElement::from(1u64), "ln interaction len");
        idx += 1;
        // claimed_sum = SecureField::from(M31::from(20)) = (20, 0, 0, 0)
        assert_eq!(
            felts[idx],
            FieldElement::from(20u64),
            "ln interaction sum[0]"
        );
        idx += 4;

        // 7. embedding_claims: length=0 (empty for this proof)
        assert_eq!(felts[idx], FieldElement::ZERO, "embedding_claims len = 0");
        idx += 1;

        // 8. interaction_claim: total sum = activation(10) + layernorm(20) = 30
        assert_eq!(
            felts[idx],
            FieldElement::from(30u64),
            "total interaction sum[0]"
        );
        idx += 4;

        // 9. pcs_config
        let default_config = PcsConfig::default();
        assert_eq!(
            felts[idx],
            FieldElement::from(default_config.pow_bits as u64),
            "pcs_config.pow_bits"
        );
        idx += 4;

        // 10. interaction_pow = 0
        assert_eq!(felts[idx], FieldElement::ZERO, "interaction_pow");
        idx += 1;

        // 11. stark_proof data follows
        assert!(
            felts.len() > idx + 10,
            "STARK proof should have data after header"
        );

        // === Outer MLProof sections (graph-level metadata) ===
        // Trailing: add(count+claim) + mul(count+claim) + layernorm(count+claim+sum) + embedding(count) + attention(count) + tee(1)
        let len = felts.len();
        // tee_attestation_hash: None = 1 felt (0)
        assert_eq!(
            felts[len - 1],
            FieldElement::ZERO,
            "tee_attestation_hash = None"
        );
        assert_eq!(
            felts[len - 2],
            FieldElement::ZERO,
            "attention_proofs count = 0"
        );
        assert_eq!(
            felts[len - 3],
            FieldElement::ZERO,
            "embedding_claims count = 0"
        );
        // layernorm_claims: count(1) + claim(layer_index + trace_rows + claimed_sum) = 1 + 6 = 7
        // mul_claims: count(1) + claim(layer_index + trace_rows) = 1 + 2 = 3
        // add_claims: count(1) + claim(layer_index + trace_rows) = 1 + 2 = 3
        // Total trailing before embedding = 7 + 3 + 3 = 13 felts
        let ln_end = len - 3; // skip tee + attention + embedding counts
        let ln_start = ln_end - 7; // 1 count + 1 claim * (1 + 1 + 4) = 7
        assert_eq!(
            felts[ln_start],
            FieldElement::from(1u64),
            "outer layernorm count"
        );
        assert_eq!(
            felts[ln_start + 1],
            FieldElement::from(7u64),
            "outer ln layer_index"
        );
    }

    #[test]
    fn test_serialize_elementwise_component_claim_serde() {
        let claim = LayerClaim {
            layer_index: 5,
            claimed_sum: SecureField::default(),
            trace_rows: 32,
        };
        let serde = ElementwiseClaimForSerde::from_layer_claim(&claim);
        assert_eq!(serde.layer_index, 5);
        assert_eq!(serde.log_size, 5); // ilog2(32)

        let mut out = Vec::new();
        serialize_elementwise_component_claim(&serde, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], FieldElement::from(5u64));
        assert_eq!(out[1], FieldElement::from(5u64));
    }

    #[test]
    fn test_serialize_layernorm_component_claim_serde() {
        let claim = LayerClaim {
            layer_index: 9,
            claimed_sum: SecureField::from(M31::from(100)),
            trace_rows: 64,
        };
        let serde = LayerNormClaimForSerde::from_layer_claim(&claim);
        assert_eq!(serde.layer_index, 9);
        assert_eq!(serde.log_size, 6); // ilog2(64)

        let mut out = Vec::new();
        serialize_layernorm_component_claim(&serde, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], FieldElement::from(9u64));
        assert_eq!(out[1], FieldElement::from(6u64));
    }

    // === Raw IO Serialization Tests ===

    #[test]
    fn test_serialize_raw_io_basic() {
        use crate::components::matmul::M31Matrix;

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut output = M31Matrix::new(1, 2);
        output.set(0, 0, M31::from(10));
        output.set(0, 1, M31::from(20));

        let serialized = serialize_raw_io(&input, &output);

        // Layout: [1, 4, 4, 1,2,3,4, 1, 2, 2, 10,20]
        assert_eq!(
            serialized.len(),
            6 + 4 + 2,
            "6 header fields + 4 input + 2 output"
        );
        // Input header
        assert_eq!(serialized[0], FieldElement::from(1u64), "input rows");
        assert_eq!(serialized[1], FieldElement::from(4u64), "input cols");
        assert_eq!(serialized[2], FieldElement::from(4u64), "input data len");
        // Input data
        assert_eq!(serialized[3], FieldElement::from(1u64));
        assert_eq!(serialized[4], FieldElement::from(2u64));
        assert_eq!(serialized[5], FieldElement::from(3u64));
        assert_eq!(serialized[6], FieldElement::from(4u64));
        // Output header
        assert_eq!(serialized[7], FieldElement::from(1u64), "output rows");
        assert_eq!(serialized[8], FieldElement::from(2u64), "output cols");
        assert_eq!(serialized[9], FieldElement::from(2u64), "output data len");
        // Output data
        assert_eq!(serialized[10], FieldElement::from(10u64));
        assert_eq!(serialized[11], FieldElement::from(20u64));
    }

    #[test]
    fn test_serialize_raw_io_matches_io_commitment() {
        use crate::aggregation::compute_io_commitment;
        use crate::components::matmul::M31Matrix;

        let mut input = M31Matrix::new(2, 3);
        for i in 0..2 {
            for j in 0..3 {
                input.set(i, j, M31::from((i * 3 + j + 1) as u32));
            }
        }

        let mut output = M31Matrix::new(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                output.set(i, j, M31::from((i * 2 + j + 10) as u32));
            }
        }

        // The serialized IO data should hash to the same IO commitment
        let serialized = serialize_raw_io(&input, &output);
        let commitment = compute_io_commitment(&input, &output);

        // Hash the serialized data with Poseidon
        let recomputed = starknet_crypto::poseidon_hash_many(&serialized);
        assert_eq!(
            recomputed, commitment,
            "serialize_raw_io hash must match compute_io_commitment"
        );
    }

    #[test]
    fn test_serialize_raw_io_packed_commitment() {
        use crate::aggregation::{compute_io_commitment_packed};
        use crate::components::matmul::M31Matrix;
        use crate::starknet::pack_m31_io_data;

        let mut input = M31Matrix::new(2, 3);
        for i in 0..2 {
            for j in 0..3 {
                input.set(i, j, M31::from((i * 3 + j + 1) as u32));
            }
        }

        let mut output = M31Matrix::new(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                output.set(i, j, M31::from((i * 2 + j + 10) as u32));
            }
        }

        // Packed commitment: Poseidon(original_io_len, packed_felts...)
        let serialized = serialize_raw_io(&input, &output);
        let packed = pack_m31_io_data(&serialized);
        let mut hash_inputs = vec![FieldElement::from(serialized.len() as u64)];
        hash_inputs.extend_from_slice(&packed);
        let manual = starknet_crypto::poseidon_hash_many(&hash_inputs);

        let from_fn = compute_io_commitment_packed(&input, &output);
        assert_eq!(
            manual, from_fn,
            "manual packed hash must match compute_io_commitment_packed"
        );

        // Packed commitment must differ from unpacked commitment
        let unpacked = crate::aggregation::compute_io_commitment(&input, &output);
        assert_ne!(
            from_fn, unpacked,
            "packed and unpacked commitments must differ"
        );
    }

    #[test]
    fn test_serialize_raw_io_matches_io_commitment_packed() {
        use crate::aggregation::compute_io_commitment_packed;
        use crate::components::matmul::M31Matrix;
        use crate::starknet::pack_m31_io_data;

        let mut input = M31Matrix::new(1, 8);
        for j in 0..8 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut output = M31Matrix::new(1, 4);
        for j in 0..4 {
            output.set(0, j, M31::from((j + 100) as u32));
        }

        let serialized = serialize_raw_io(&input, &output);
        let packed = pack_m31_io_data(&serialized);

        // Packed commitment: Poseidon(original_io_len, packed_data...)
        let mut hash_inputs = Vec::with_capacity(1 + packed.len());
        hash_inputs.push(FieldElement::from(serialized.len() as u64));
        hash_inputs.extend_from_slice(&packed);
        let recomputed = starknet_crypto::poseidon_hash_many(&hash_inputs);

        let commitment = compute_io_commitment_packed(&input, &output);
        assert_eq!(
            recomputed, commitment,
            "packed IO hash must match compute_io_commitment_packed"
        );
    }

    // === Weight MLE Opening Serialization Tests ===

    #[test]
    fn test_serialize_weight_mle_openings_empty() {
        let openings: Vec<WeightMleOpening> = Vec::new();
        let mut out = Vec::new();
        serialize_weight_mle_openings(&openings, &mut out);
        assert_eq!(out.len(), 1, "empty openings = just the count");
        assert_eq!(out[0], FieldElement::ZERO);
    }

    #[test]
    fn test_serialize_weight_mle_openings_single() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        let opening = WeightMleOpening {
            layer_idx: 3,
            point: vec![QM31(
                CM31(M31::from(10), M31::from(20)),
                CM31(M31::from(30), M31::from(40)),
            )],
            value: QM31(
                CM31(M31::from(100), M31::from(200)),
                CM31(M31::from(300), M31::from(400)),
            ),
            commitment: FieldElement::from(0x1234u64),
        };

        let mut out = Vec::new();
        serialize_weight_mle_openings(&[opening], &mut out);

        // Layout: count(1) + [layer_idx(1) + point_len(1) + point(4) + value(4) + commitment(1)]
        assert_eq!(out.len(), 1 + 1 + 1 + 4 + 4 + 1, "single opening");
        assert_eq!(out[0], FieldElement::from(1u64), "count = 1");
        assert_eq!(out[1], FieldElement::from(3u64), "layer_idx = 3");
        assert_eq!(out[2], FieldElement::from(1u64), "point_len = 1");
        // Point QM31 = [10, 20, 30, 40]
        assert_eq!(out[3], FieldElement::from(10u64));
        assert_eq!(out[4], FieldElement::from(20u64));
        assert_eq!(out[5], FieldElement::from(30u64));
        assert_eq!(out[6], FieldElement::from(40u64));
        // Value QM31 = [100, 200, 300, 400]
        assert_eq!(out[7], FieldElement::from(100u64));
        assert_eq!(out[8], FieldElement::from(200u64));
        assert_eq!(out[9], FieldElement::from(300u64));
        assert_eq!(out[10], FieldElement::from(400u64));
        // Commitment
        assert_eq!(out[11], FieldElement::from(0x1234u64));
    }

    #[test]
    fn test_serialize_qm31_pair_packed_roundtrip() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        let test_pairs = vec![
            // (a, b) — both zero
            (
                QM31(CM31(M31::from(0), M31::from(0)), CM31(M31::from(0), M31::from(0))),
                QM31(CM31(M31::from(0), M31::from(0)), CM31(M31::from(0), M31::from(0))),
            ),
            // Small values
            (
                QM31(CM31(M31::from(1), M31::from(2)), CM31(M31::from(3), M31::from(4))),
                QM31(CM31(M31::from(5), M31::from(6)), CM31(M31::from(7), M31::from(8))),
            ),
            // Max M31 values (2^31 - 2)
            (
                QM31(
                    CM31(M31::from(2147483646u32), M31::from(1000000u32)),
                    CM31(M31::from(999999u32), M31::from(12345u32)),
                ),
                QM31(
                    CM31(M31::from(2147483646u32), M31::from(2147483646u32)),
                    CM31(M31::from(2147483646u32), M31::from(2147483646u32)),
                ),
            ),
            // Mixed zero and max
            (
                QM31(CM31(M31::from(0), M31::from(2147483646u32)), CM31(M31::from(0), M31::from(0))),
                QM31(CM31(M31::from(2147483646u32), M31::from(0)), CM31(M31::from(0), M31::from(2147483646u32))),
            ),
        ];

        for (a, b) in &test_pairs {
            let mut out = Vec::new();
            serialize_qm31_pair_packed(*a, *b, &mut out);
            assert_eq!(out.len(), 1, "double-packed QM31 pair should be 1 felt");

            // Verify round-trip via deserialize_qm31_pair_packed
            let (recovered_a, recovered_b) = deserialize_qm31_pair_packed(out[0]);
            assert_eq!(recovered_a, *a, "pair pack/unpack failed for a={:?}", a);
            assert_eq!(recovered_b, *b, "pair pack/unpack failed for b={:?}", b);
        }
    }

    #[test]
    fn test_double_packed_smaller_than_packed() {
        // Verify that double-packed serialization of a MatMul round poly
        // produces fewer felts than regular packed serialization.
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        let c0 = QM31(CM31(M31::from(100), M31::from(200)), CM31(M31::from(300), M31::from(400)));
        let c2 = QM31(CM31(M31::from(500), M31::from(600)), CM31(M31::from(700), M31::from(800)));

        // Regular packed: 2 felts (one per QM31)
        let mut packed = Vec::new();
        serialize_qm31_packed(c0, &mut packed);
        serialize_qm31_packed(c2, &mut packed);
        assert_eq!(packed.len(), 2);

        // Double-packed: 1 felt (both QM31 in one)
        let mut dp = Vec::new();
        serialize_qm31_pair_packed(c0, c2, &mut dp);
        assert_eq!(dp.len(), 1);

        // Verify round-trip
        let (rc0, rc2) = deserialize_qm31_pair_packed(dp[0]);
        assert_eq!(rc0, c0);
        assert_eq!(rc2, c2);
    }

    #[test]
    fn test_double_packed_deg3_poly_roundtrip() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        // Degree-3 round polys serialize (c0, c2) as a pair + c3 as single packed
        let c0 = QM31(
            CM31(M31::from(111u32), M31::from(222u32)),
            CM31(M31::from(333u32), M31::from(444u32)),
        );
        let c2 = QM31(
            CM31(M31::from(555u32), M31::from(666u32)),
            CM31(M31::from(777u32), M31::from(888u32)),
        );
        let c3 = QM31(
            CM31(M31::from(999u32), M31::from(1111u32)),
            CM31(M31::from(2222u32), M31::from(3333u32)),
        );

        // Serialize (c0, c2) as double-packed pair
        let mut dp_pair = Vec::new();
        serialize_qm31_pair_packed(c0, c2, &mut dp_pair);
        assert_eq!(dp_pair.len(), 1);

        // Serialize c3 as single-packed
        let mut c3_packed = Vec::new();
        serialize_qm31_packed(c3, &mut c3_packed);
        assert_eq!(c3_packed.len(), 1);

        // Round-trip pair
        let (rc0, rc2) = deserialize_qm31_pair_packed(dp_pair[0]);
        assert_eq!(rc0, c0, "deg3 pair c0 mismatch");
        assert_eq!(rc2, c2, "deg3 pair c2 mismatch");

        // Round-trip single
        let rc3 = crate::crypto::poseidon_channel::felt_to_securefield(c3_packed[0]);
        assert_eq!(rc3, c3, "deg3 single c3 mismatch");

        // Total felts for a deg-3 round poly: 2 (pair + single) vs 3 (packed c0 + c2 + c3)
        assert_eq!(dp_pair.len() + c3_packed.len(), 2);
    }

    #[test]
    fn test_double_packed_edge_values() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        let zero = M31::from(0u32);
        let max_m31 = M31::from(2147483646u32); // 2^31 - 2 (max valid M31)

        let edge_cases = vec![
            // All zeros
            (
                QM31(CM31(zero, zero), CM31(zero, zero)),
                QM31(CM31(zero, zero), CM31(zero, zero)),
                "all_zeros",
            ),
            // All max
            (
                QM31(CM31(max_m31, max_m31), CM31(max_m31, max_m31)),
                QM31(CM31(max_m31, max_m31), CM31(max_m31, max_m31)),
                "all_max",
            ),
            // Alternating zero/max in all 8 positions
            (
                QM31(CM31(zero, max_m31), CM31(zero, max_m31)),
                QM31(CM31(zero, max_m31), CM31(zero, max_m31)),
                "alternating_0_max",
            ),
            // Max in first, zero in rest
            (
                QM31(CM31(max_m31, zero), CM31(zero, zero)),
                QM31(CM31(zero, zero), CM31(zero, zero)),
                "max_first_only",
            ),
            // Zero in first, max in last
            (
                QM31(CM31(zero, zero), CM31(zero, zero)),
                QM31(CM31(zero, zero), CM31(zero, max_m31)),
                "max_last_only",
            ),
            // One value in each QM31 component
            (
                QM31(CM31(M31::from(1u32), zero), CM31(zero, zero)),
                QM31(CM31(zero, zero), CM31(zero, M31::from(1u32))),
                "sparse_ones",
            ),
        ];

        for (a, b, label) in &edge_cases {
            let mut buf = Vec::new();
            serialize_qm31_pair_packed(*a, *b, &mut buf);
            assert_eq!(buf.len(), 1, "edge case {}: expected 1 felt", label);
            let (ra, rb) = deserialize_qm31_pair_packed(buf[0]);
            assert_eq!(ra, *a, "edge case {}: a mismatch", label);
            assert_eq!(rb, *b, "edge case {}: b mismatch", label);
        }
    }

    #[test]
    fn test_double_packed_full_proof_roundtrip() {
        use stwo::core::fields::cm31::CM31;
        use stwo::core::fields::qm31::QM31;

        // Build a small GKR proof with multiple layer types to verify full
        // serialize→deserialize round-trip for both packed and double-packed.
        use crate::components::matmul::RoundPoly;
        use crate::gkr::types::{
            GKRClaim, GKRProof, LayerProof, RoundPolyDeg3,
            WeightOpeningTranscriptMode,
        };
        use num_traits::Zero;

        let q = |a: u32, b: u32, c: u32, d: u32| QM31(CM31(M31::from(a), M31::from(b)), CM31(M31::from(c), M31::from(d)));

        let matmul_proof = LayerProof::MatMul {
            round_polys: vec![
                RoundPoly { c0: q(10, 20, 30, 40), c1: q(50, 60, 70, 80), c2: q(90, 100, 110, 120) },
                RoundPoly { c0: q(130, 140, 150, 160), c1: q(170, 180, 190, 200), c2: q(210, 220, 230, 240) },
            ],
            final_a_eval: q(1000, 2000, 3000, 4000),
            final_b_eval: q(5000, 6000, 7000, 8000),
        };

        let mul_proof = LayerProof::Mul {
            eq_round_polys: vec![
                RoundPolyDeg3 { c0: q(11, 22, 33, 44), c1: q(55, 66, 77, 88), c2: q(99, 111, 122, 133), c3: q(144, 155, 166, 177) },
            ],
            lhs_eval: q(1001, 2001, 3001, 4001),
            rhs_eval: q(5001, 6001, 7001, 8001),
        };

        let proof = GKRProof {
            layer_proofs: vec![matmul_proof, mul_proof],
            output_claim: GKRClaim { point: vec![], value: QM31::zero() },
            input_claim: GKRClaim { point: vec![], value: QM31::zero() },
            weight_commitments: vec![],
            weight_openings: vec![],
            weight_claims: vec![],
            weight_opening_transcript_mode: WeightOpeningTranscriptMode::Sequential,
            io_commitment: FieldElement::ZERO,
            deferred_proofs: vec![],
            aggregated_binding: None,
        };

        // Serialize both ways
        let mut packed = Vec::new();
        serialize_gkr_proof_data_only_packed(&proof, &mut packed);

        let mut double_packed = Vec::new();
        serialize_gkr_proof_data_only_double_packed(&proof, &mut double_packed);

        // Double-packed must be smaller
        assert!(
            double_packed.len() < packed.len(),
            "double_packed ({}) should be smaller than packed ({})",
            double_packed.len(),
            packed.len()
        );

        // Verify round-trip of the double-packed data by walking it manually:
        // MatMul layer: tag(1) + num_rounds(1) + 2*(pair=1) + final_a(1) + final_b(1) = 6
        // Mul layer: tag(1) + num_rounds(1) + 1*(pair=1 + c3=1) + lhs(1) + rhs(1) = 6
        // deferred count(1) = 1
        // Total = 13
        assert_eq!(
            double_packed.len(), 13,
            "expected 13 felts for double-packed proof, got {}",
            double_packed.len()
        );

        // Walk the double-packed data and verify key values
        let felt_to_u32 = |f: FieldElement| -> u32 {
            let b = f.to_bytes_be();
            u32::from_be_bytes([b[28], b[29], b[30], b[31]])
        };
        let mut off = 0usize;

        // MatMul layer
        let tag = felt_to_u32(double_packed[off]); off += 1;
        assert_eq!(tag, 0, "MatMul tag");
        let nrounds = felt_to_u32(double_packed[off]); off += 1;
        assert_eq!(nrounds, 2, "MatMul rounds");

        // Round 0: (c0, c2) pair
        let (rc0, rc2) = deserialize_qm31_pair_packed(double_packed[off]); off += 1;
        assert_eq!(rc0, q(10, 20, 30, 40), "MatMul round 0 c0");
        assert_eq!(rc2, q(90, 100, 110, 120), "MatMul round 0 c2");

        // Round 1: (c0, c2) pair
        let (rc0, rc2) = deserialize_qm31_pair_packed(double_packed[off]); off += 1;
        assert_eq!(rc0, q(130, 140, 150, 160), "MatMul round 1 c0");
        assert_eq!(rc2, q(210, 220, 230, 240), "MatMul round 1 c2");

        // final_a, final_b
        let final_a = crate::crypto::poseidon_channel::felt_to_securefield(double_packed[off]); off += 1;
        assert_eq!(final_a, q(1000, 2000, 3000, 4000), "MatMul final_a");
        let final_b = crate::crypto::poseidon_channel::felt_to_securefield(double_packed[off]); off += 1;
        assert_eq!(final_b, q(5000, 6000, 7000, 8000), "MatMul final_b");

        // Mul layer
        let tag = felt_to_u32(double_packed[off]); off += 1;
        assert_eq!(tag, 2, "Mul tag");
        let nrounds = felt_to_u32(double_packed[off]); off += 1;
        assert_eq!(nrounds, 1, "Mul rounds");

        // Round 0: (c0, c2) pair + c3 single
        let (rc0, rc2) = deserialize_qm31_pair_packed(double_packed[off]); off += 1;
        assert_eq!(rc0, q(11, 22, 33, 44), "Mul round 0 c0");
        assert_eq!(rc2, q(99, 111, 122, 133), "Mul round 0 c2");
        let rc3 = crate::crypto::poseidon_channel::felt_to_securefield(double_packed[off]); off += 1;
        assert_eq!(rc3, q(144, 155, 166, 177), "Mul round 0 c3");

        // lhs, rhs
        let lhs = crate::crypto::poseidon_channel::felt_to_securefield(double_packed[off]); off += 1;
        assert_eq!(lhs, q(1001, 2001, 3001, 4001), "Mul lhs");
        let rhs = crate::crypto::poseidon_channel::felt_to_securefield(double_packed[off]); off += 1;
        assert_eq!(rhs, q(5001, 6001, 7001, 8001), "Mul rhs");

        // Deferred count
        let deferred_count = felt_to_u32(double_packed[off]); off += 1;
        assert_eq!(deferred_count, 0, "deferred count");

        assert_eq!(off, double_packed.len(), "consumed all double-packed data");
    }
}
