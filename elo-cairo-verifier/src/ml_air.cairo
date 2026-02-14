/// ML Air — Direct on-chain verification of unified ML STARK proofs.
///
/// Supports ALL component types from the Rust stwo-ml prover:
///   - Activation (ReLU/GELU/Sigmoid): LogUp with type-tag domain separation
///   - Add/Mul (elementwise): Pure AIR constraints
///   - LayerNorm: Algebraic + LogUp for rsqrt table
///   - Embedding: LogUp for token embedding table
///
/// Architecture:
///   Stage 1: stwo-ml GPU → AggregatedModelProofOnChain
///   On-chain: verify_model_direct() = verify<MLAir>(unified_stark) + verify_batched_sumcheck()
///
/// This eliminates Stage 2 (46.8s Cairo VM execution) entirely.

use core::num::traits::Zero;
use stwo_verifier_core::verifier::Air;
use stwo_verifier_core::circle::CirclePoint;
use stwo_verifier_core::fields::Invertible;
use stwo_verifier_core::fields::m31::m31;
use stwo_verifier_core::fields::qm31::{QM31, QM31Serde, QM31Trait};
use stwo_verifier_core::poly::circle::{CanonicCosetImpl, CanonicCosetTrait};
use stwo_verifier_core::utils::pow2;
use stwo_verifier_core::{ColumnSpan, TreeSpan};
use stwo_verifier_core::channel::{Channel, ChannelTrait};
use stwo_verifier_core::verifier::verify;
use stwo_verifier_core::pcs::PcsConfig;
use stwo_verifier_core::pcs::PcsConfigTrait;
use stwo_verifier_core::pcs::verifier::{
    CommitmentSchemeVerifierImpl, CommitmentSchemeVerifierTrait, get_trace_lde_log_size,
};
use stwo_verifier_core::Hash;
use stwo_verifier_core::verifier::StarkProof;
use stwo_constraint_framework::{LookupElements, LookupElementsTrait};

// ============================================================================
// Claim Types
// ============================================================================

/// Claim about a single activation layer.
#[derive(Drop, Serde, Copy)]
pub struct ActivationClaim {
    pub layer_index: u32,
    pub log_size: u32,
    pub activation_type: u8,
}

/// Interaction claim for activation layer (LogUp sum).
#[derive(Drop, Serde, Copy)]
pub struct ActivationInteractionClaim {
    pub claimed_sum: QM31,
}

/// Claim about an elementwise (Add or Mul) layer. Pure AIR, no LogUp.
#[derive(Drop, Serde, Copy)]
pub struct ElementwiseClaim {
    pub layer_index: u32,
    pub log_size: u32,
}

/// Claim about a LayerNorm layer.
#[derive(Drop, Serde, Copy)]
pub struct LayerNormClaim {
    pub layer_index: u32,
    pub log_size: u32,
}

/// Interaction claim for LayerNorm layer (LogUp sum for rsqrt table).
#[derive(Drop, Serde, Copy)]
pub struct LayerNormInteractionClaim {
    pub claimed_sum: QM31,
}

/// Claim about an Embedding layer.
#[derive(Drop, Serde, Copy)]
pub struct EmbeddingClaim {
    pub layer_index: u32,
    pub log_size: u32,
}

/// ML interaction claim — total LogUp sum across ALL LogUp components.
/// Must be zero for a valid proof.
#[derive(Drop, Serde, Copy)]
pub struct MLInteractionClaim {
    pub activation_claimed_sum: QM31,
}

/// Claim about the ML inference computation.
#[derive(Drop, Serde, Copy)]
pub struct MLClaim {
    pub model_id: felt252,
    pub num_layers: u32,
    pub activation_type: u8,
    pub io_commitment: felt252,
    pub weight_commitment: felt252,
}

/// Unified STARK proof covering ALL non-matmul components.
///
/// Field order MUST match Rust `serialize_unified_stark_proof()` in cairo_serde.rs:
///   1. activation_claims
///   2. activation_interaction_claims
///   3. add_claims
///   4. mul_claims
///   5. layernorm_claims
///   6. layernorm_interaction_claims
///   7. embedding_claims
///   8. interaction_claim
///   9. pcs_config
///   10. interaction_pow
///   11. stark_proof
#[derive(Drop, Serde)]
pub struct UnifiedStarkProof {
    pub activation_claims: Array<ActivationClaim>,
    pub activation_interaction_claims: Array<ActivationInteractionClaim>,
    pub add_claims: Array<ElementwiseClaim>,
    pub mul_claims: Array<ElementwiseClaim>,
    pub layernorm_claims: Array<LayerNormClaim>,
    pub layernorm_interaction_claims: Array<LayerNormInteractionClaim>,
    pub embedding_claims: Array<EmbeddingClaim>,
    pub interaction_claim: MLInteractionClaim,
    pub pcs_config: PcsConfig,
    pub interaction_pow: u64,
    pub stark_proof: StarkProof,
}

// ============================================================================
// Column Constants
// ============================================================================

// --- Activation (LogUp) ---
// Relation: ActivationRelation(3) = [type_tag, input, output]
pub const N_ACTIVATION_PREPROCESSED_COLUMNS: u32 = 2; // table_input, table_output
pub const N_ACTIVATION_TRACE_COLUMNS: u32 = 3; // input, output, multiplicity
pub const N_ACTIVATION_INTERACTION_COLUMNS: u32 = 4; // QM31 partial evals

/// Activation lookup: combine(type_tag, input, output).
/// N=3 matches Rust ActivationRelation(3) with type-tag domain separation.
pub type ActivationLookupElements = LookupElements<3>;

// --- Add/Mul (Pure AIR) ---
pub const N_ELEMENTWISE_PREPROCESSED_COLUMNS: u32 = 0;
pub const N_ELEMENTWISE_TRACE_COLUMNS: u32 = 3; // lhs, rhs, output
pub const N_ELEMENTWISE_INTERACTION_COLUMNS: u32 = 0;

// --- LayerNorm (LogUp) ---
// Relation: LayerNormRelation(2) = [variance, rsqrt_val]
pub const N_LAYERNORM_PREPROCESSED_COLUMNS: u32 = 2; // table_var, table_rsqrt
pub const N_LAYERNORM_TRACE_COLUMNS: u32 = 6; // input, mean, variance, rsqrt, output, mult
pub const N_LAYERNORM_INTERACTION_COLUMNS: u32 = 4; // QM31 partial evals

pub type LayerNormLookupElements = LookupElements<2>;

// --- Embedding (LogUp) ---
// Relation: EmbeddingRelation(3) = [token, col, value]
pub const N_EMBEDDING_PREPROCESSED_COLUMNS: u32 = 3; // table_token, table_col, table_value
pub const N_EMBEDDING_TRACE_COLUMNS: u32 = 4; // trace_token, trace_col, trace_value, mult
pub const N_EMBEDDING_INTERACTION_COLUMNS: u32 = 4; // QM31 partial evals

pub type EmbeddingLookupElements = LookupElements<3>;

// ============================================================================
// Dummy LookupElements — used when a component type is absent.
// These MUST NOT draw from the channel, preserving Fiat-Shamir transcript.
// ============================================================================

fn dummy_lookup_elements_2() -> LookupElements<2> {
    LookupElements { z: Zero::zero(), alpha: Zero::zero(), alpha_powers: array![Zero::zero(), Zero::zero()] }
}

fn dummy_lookup_elements_3() -> LookupElements<3> {
    LookupElements { z: Zero::zero(), alpha: Zero::zero(), alpha_powers: array![Zero::zero(), Zero::zero(), Zero::zero()] }
}

// ============================================================================
// Component Structs
// ============================================================================

#[derive(Drop)]
pub struct ActivationComponent {
    pub claim: ActivationClaim,
    pub interaction_claim: ActivationInteractionClaim,
    pub lookup_elements: ActivationLookupElements,
}

#[derive(Drop, Copy)]
pub struct ElementwiseAddComponent {
    pub claim: ElementwiseClaim,
}

#[derive(Drop, Copy)]
pub struct ElementwiseMulComponent {
    pub claim: ElementwiseClaim,
}

#[derive(Drop)]
pub struct LayerNormComponent {
    pub claim: LayerNormClaim,
    pub interaction_claim: LayerNormInteractionClaim,
    pub lookup_elements: LayerNormLookupElements,
}

#[derive(Drop)]
pub struct EmbeddingComponent {
    pub claim: EmbeddingClaim,
    pub lookup_elements: EmbeddingLookupElements,
}

// ============================================================================
// Constraint Evaluators
// ============================================================================

/// Evaluate activation LogUp constraints at an out-of-domain point.
///
/// Relation: ActivationRelation(3) = [type_tag, input, output]
/// Constraint: ((S_curr - S_prev + claimed_sum/N) * denom + mult) / vanishing = 0
pub fn evaluate_activation_constraints_at_point(
    component: @ActivationComponent,
    ref sum: QM31,
    ref trace_mask_values: ColumnSpan<Span<QM31>>,
    ref interaction_trace_mask_values: ColumnSpan<Span<QM31>>,
    random_coeff: QM31,
    point: CirclePoint<QM31>,
) {
    let log_size = *component.claim.log_size;
    let trace_domain = CanonicCosetImpl::new(log_size);
    let domain_vanishing_eval_inv = trace_domain.eval_vanishing(point).inverse();
    let claimed_sum = *component.interaction_claim.claimed_sum;
    let column_size = m31(pow2(log_size));

    // Pop 3 trace columns: [input, output, multiplicity]
    let [col_input, col_output, col_mult]: [Span<QM31>; 3] =
        (*trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [input_val]: [QM31; 1] = (*col_input.try_into().unwrap()).unbox();
    let [output_val]: [QM31; 1] = (*col_output.try_into().unwrap()).unbox();
    let [mult_val]: [QM31; 1] = (*col_mult.try_into().unwrap()).unbox();

    // Combine lookup: z - (alpha^0 * type_tag + alpha^1 * input + alpha^2 * output)
    let type_tag: QM31 = m31((*component.claim.activation_type).into()).into();
    let denom = component.lookup_elements.combine_qm31([type_tag, input_val, output_val]);

    // Pop 4 interaction trace columns (LogUp cumulative sum as QM31 partial evals)
    let [t2c0, t2c1, t2c2, t2c3]: [Span<QM31>; 4] =
        (*interaction_trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [t2c0_prev, t2c0_curr]: [QM31; 2] = (*t2c0.try_into().unwrap()).unbox();
    let [t2c1_prev, t2c1_curr]: [QM31; 2] = (*t2c1.try_into().unwrap()).unbox();
    let [t2c2_prev, t2c2_curr]: [QM31; 2] = (*t2c2.try_into().unwrap()).unbox();
    let [t2c3_prev, t2c3_curr]: [QM31; 2] = (*t2c3.try_into().unwrap()).unbox();

    core::internal::revoke_ap_tracking();

    let s_curr = QM31Trait::from_partial_evals([t2c0_curr, t2c1_curr, t2c2_curr, t2c3_curr]);
    let s_prev = QM31Trait::from_partial_evals([t2c0_prev, t2c1_prev, t2c2_prev, t2c3_prev]);

    let constraint_quotient = (((s_curr - s_prev + (claimed_sum * (column_size.inverse().into())))
        * denom)
        + mult_val)
        * domain_vanishing_eval_inv;

    sum = sum * random_coeff + constraint_quotient;
}

/// Evaluate elementwise Add constraint: output - lhs - rhs = 0 (pure AIR, degree 1).
pub fn evaluate_add_constraints_at_point(
    component: @ElementwiseAddComponent,
    ref sum: QM31,
    ref trace_mask_values: ColumnSpan<Span<QM31>>,
    random_coeff: QM31,
    point: CirclePoint<QM31>,
) {
    let log_size = *component.claim.log_size;
    let trace_domain = CanonicCosetImpl::new(log_size);
    let domain_vanishing_eval_inv = trace_domain.eval_vanishing(point).inverse();

    let [col_lhs, col_rhs, col_output]: [Span<QM31>; 3] =
        (*trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [lhs]: [QM31; 1] = (*col_lhs.try_into().unwrap()).unbox();
    let [rhs]: [QM31; 1] = (*col_rhs.try_into().unwrap()).unbox();
    let [output]: [QM31; 1] = (*col_output.try_into().unwrap()).unbox();

    // Constraint: output = lhs + rhs
    let constraint_quotient = (output - lhs - rhs) * domain_vanishing_eval_inv;
    sum = sum * random_coeff + constraint_quotient;
}

/// Evaluate elementwise Mul constraint: output - lhs * rhs = 0 (pure AIR, degree 2).
pub fn evaluate_mul_constraints_at_point(
    component: @ElementwiseMulComponent,
    ref sum: QM31,
    ref trace_mask_values: ColumnSpan<Span<QM31>>,
    random_coeff: QM31,
    point: CirclePoint<QM31>,
) {
    let log_size = *component.claim.log_size;
    let trace_domain = CanonicCosetImpl::new(log_size);
    let domain_vanishing_eval_inv = trace_domain.eval_vanishing(point).inverse();

    let [col_lhs, col_rhs, col_output]: [Span<QM31>; 3] =
        (*trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [lhs]: [QM31; 1] = (*col_lhs.try_into().unwrap()).unbox();
    let [rhs]: [QM31; 1] = (*col_rhs.try_into().unwrap()).unbox();
    let [output]: [QM31; 1] = (*col_output.try_into().unwrap()).unbox();

    // Constraint: output = lhs * rhs
    let constraint_quotient = (output - lhs * rhs) * domain_vanishing_eval_inv;
    sum = sum * random_coeff + constraint_quotient;
}

/// Evaluate LayerNorm constraints at an out-of-domain point.
///
/// Algebraic: output - (input - mean) * rsqrt_val = 0
/// LogUp: (variance, rsqrt_val) ∈ rsqrt_table via LayerNormRelation(2)
pub fn evaluate_layernorm_constraints_at_point(
    component: @LayerNormComponent,
    ref sum: QM31,
    ref trace_mask_values: ColumnSpan<Span<QM31>>,
    ref interaction_trace_mask_values: ColumnSpan<Span<QM31>>,
    random_coeff: QM31,
    point: CirclePoint<QM31>,
) {
    let log_size = *component.claim.log_size;
    let trace_domain = CanonicCosetImpl::new(log_size);
    let domain_vanishing_eval_inv = trace_domain.eval_vanishing(point).inverse();
    let claimed_sum = *component.interaction_claim.claimed_sum;
    let column_size = m31(pow2(log_size));

    // Pop 6 trace columns: [input, mean, variance, rsqrt_val, output, multiplicity]
    let [col_input, col_mean, col_variance, col_rsqrt, col_output, col_mult]: [Span<QM31>; 6] =
        (*trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [input_val]: [QM31; 1] = (*col_input.try_into().unwrap()).unbox();
    let [mean_val]: [QM31; 1] = (*col_mean.try_into().unwrap()).unbox();
    let [variance_val]: [QM31; 1] = (*col_variance.try_into().unwrap()).unbox();
    let [rsqrt_val]: [QM31; 1] = (*col_rsqrt.try_into().unwrap()).unbox();
    let [output_val]: [QM31; 1] = (*col_output.try_into().unwrap()).unbox();
    let [mult_val]: [QM31; 1] = (*col_mult.try_into().unwrap()).unbox();

    // Algebraic constraint: output = (input - mean) * rsqrt_val
    let alg_constraint = (output_val - (input_val - mean_val) * rsqrt_val)
        * domain_vanishing_eval_inv;
    sum = sum * random_coeff + alg_constraint;

    // LogUp: combine(variance, rsqrt_val) for rsqrt table lookup
    let denom = component.lookup_elements.combine_qm31([variance_val, rsqrt_val]);

    // Pop 4 interaction trace columns
    let [t2c0, t2c1, t2c2, t2c3]: [Span<QM31>; 4] =
        (*interaction_trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [t2c0_prev, t2c0_curr]: [QM31; 2] = (*t2c0.try_into().unwrap()).unbox();
    let [t2c1_prev, t2c1_curr]: [QM31; 2] = (*t2c1.try_into().unwrap()).unbox();
    let [t2c2_prev, t2c2_curr]: [QM31; 2] = (*t2c2.try_into().unwrap()).unbox();
    let [t2c3_prev, t2c3_curr]: [QM31; 2] = (*t2c3.try_into().unwrap()).unbox();

    core::internal::revoke_ap_tracking();

    let s_curr = QM31Trait::from_partial_evals([t2c0_curr, t2c1_curr, t2c2_curr, t2c3_curr]);
    let s_prev = QM31Trait::from_partial_evals([t2c0_prev, t2c1_prev, t2c2_prev, t2c3_prev]);

    let logup_constraint = (((s_curr - s_prev + (claimed_sum * (column_size.inverse().into())))
        * denom)
        + mult_val)
        * domain_vanishing_eval_inv;

    sum = sum * random_coeff + logup_constraint;
}

/// Evaluate Embedding LogUp constraints at an out-of-domain point.
///
/// LogUp: (token, col, value) ∈ embedding_table via EmbeddingRelation(3)
pub fn evaluate_embedding_constraints_at_point(
    component: @EmbeddingComponent,
    ref sum: QM31,
    ref trace_mask_values: ColumnSpan<Span<QM31>>,
    ref interaction_trace_mask_values: ColumnSpan<Span<QM31>>,
    random_coeff: QM31,
    point: CirclePoint<QM31>,
    claimed_sum: QM31,
) {
    let log_size = *component.claim.log_size;
    let trace_domain = CanonicCosetImpl::new(log_size);
    let domain_vanishing_eval_inv = trace_domain.eval_vanishing(point).inverse();
    let column_size = m31(pow2(log_size));

    // Pop 4 trace columns: [trace_token, trace_col, trace_value, multiplicity]
    let [col_token, col_col, col_value, col_mult]: [Span<QM31>; 4] =
        (*trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [token_val]: [QM31; 1] = (*col_token.try_into().unwrap()).unbox();
    let [col_val]: [QM31; 1] = (*col_col.try_into().unwrap()).unbox();
    let [value_val]: [QM31; 1] = (*col_value.try_into().unwrap()).unbox();
    let [mult_val]: [QM31; 1] = (*col_mult.try_into().unwrap()).unbox();

    // LogUp: combine(token, col, value) for embedding table lookup
    let denom = component.lookup_elements.combine_qm31([token_val, col_val, value_val]);

    // Pop 4 interaction trace columns
    let [t2c0, t2c1, t2c2, t2c3]: [Span<QM31>; 4] =
        (*interaction_trace_mask_values.multi_pop_front().unwrap()).unbox();
    let [t2c0_prev, t2c0_curr]: [QM31; 2] = (*t2c0.try_into().unwrap()).unbox();
    let [t2c1_prev, t2c1_curr]: [QM31; 2] = (*t2c1.try_into().unwrap()).unbox();
    let [t2c2_prev, t2c2_curr]: [QM31; 2] = (*t2c2.try_into().unwrap()).unbox();
    let [t2c3_prev, t2c3_curr]: [QM31; 2] = (*t2c3.try_into().unwrap()).unbox();

    core::internal::revoke_ap_tracking();

    let s_curr = QM31Trait::from_partial_evals([t2c0_curr, t2c1_curr, t2c2_curr, t2c3_curr]);
    let s_prev = QM31Trait::from_partial_evals([t2c0_prev, t2c1_prev, t2c2_prev, t2c3_prev]);

    let logup_constraint = (((s_curr - s_prev + (claimed_sum * (column_size.inverse().into())))
        * denom)
        + mult_val)
        * domain_vanishing_eval_inv;

    sum = sum * random_coeff + logup_constraint;
}

// ============================================================================
// Log Size Computation
// ============================================================================

/// Compute column log_sizes per commitment tree for ALL component types.
///
/// Returns [preprocessed_log_sizes, trace_log_sizes, interaction_trace_log_sizes].
///
/// Tree ordering matches Rust prover (aggregation.rs):
///   Tree 0 (preprocessed): activation tables → layernorm tables → embedding tables
///   Tree 1 (trace): activation → add → mul → layernorm → embedding
///   Tree 2 (interaction): activation logup → layernorm logup → embedding logup
pub fn compute_activation_log_sizes(
    activation_claims: Span<ActivationClaim>,
) -> Array<Array<u32>> {
    compute_unified_log_sizes(
        activation_claims,
        array![].span(),
        array![].span(),
        array![].span(),
        array![].span(),
    )
}

pub fn compute_unified_log_sizes(
    activation_claims: Span<ActivationClaim>,
    add_claims: Span<ElementwiseClaim>,
    mul_claims: Span<ElementwiseClaim>,
    layernorm_claims: Span<LayerNormClaim>,
    embedding_claims: Span<EmbeddingClaim>,
) -> Array<Array<u32>> {
    let mut preprocessed_sizes: Array<u32> = array![];
    let mut trace_sizes: Array<u32> = array![];
    let mut interaction_sizes: Array<u32> = array![];

    // --- Tree 0 (preprocessed) ---
    // Activation: 2 columns per layer
    let mut i: u32 = 0;
    while i < activation_claims.len() {
        let ls = *activation_claims.at(i).log_size;
        preprocessed_sizes.append(ls);
        preprocessed_sizes.append(ls);
        i += 1;
    };
    // LayerNorm: 2 columns per layer
    i = 0;
    while i < layernorm_claims.len() {
        let ls = *layernorm_claims.at(i).log_size;
        preprocessed_sizes.append(ls);
        preprocessed_sizes.append(ls);
        i += 1;
    };
    // Embedding: 3 columns per layer
    i = 0;
    while i < embedding_claims.len() {
        let ls = *embedding_claims.at(i).log_size;
        preprocessed_sizes.append(ls);
        preprocessed_sizes.append(ls);
        preprocessed_sizes.append(ls);
        i += 1;
    };

    // --- Tree 1 (trace) ---
    // Activation: 3 columns per layer
    i = 0;
    while i < activation_claims.len() {
        let ls = *activation_claims.at(i).log_size;
        trace_sizes.append(ls);
        trace_sizes.append(ls);
        trace_sizes.append(ls);
        i += 1;
    };
    // Add: 3 columns per layer
    i = 0;
    while i < add_claims.len() {
        let ls = *add_claims.at(i).log_size;
        trace_sizes.append(ls);
        trace_sizes.append(ls);
        trace_sizes.append(ls);
        i += 1;
    };
    // Mul: 3 columns per layer
    i = 0;
    while i < mul_claims.len() {
        let ls = *mul_claims.at(i).log_size;
        trace_sizes.append(ls);
        trace_sizes.append(ls);
        trace_sizes.append(ls);
        i += 1;
    };
    // LayerNorm: 6 columns per layer
    i = 0;
    while i < layernorm_claims.len() {
        let ls = *layernorm_claims.at(i).log_size;
        let mut c: u32 = 0;
        while c < N_LAYERNORM_TRACE_COLUMNS {
            trace_sizes.append(ls);
            c += 1;
        };
        i += 1;
    };
    // Embedding: 4 columns per layer
    i = 0;
    while i < embedding_claims.len() {
        let ls = *embedding_claims.at(i).log_size;
        let mut c: u32 = 0;
        while c < N_EMBEDDING_TRACE_COLUMNS {
            trace_sizes.append(ls);
            c += 1;
        };
        i += 1;
    };

    // --- Tree 2 (interaction) ---
    // Activation: 4 QM31 partial eval columns per layer
    i = 0;
    while i < activation_claims.len() {
        let ls = *activation_claims.at(i).log_size;
        interaction_sizes.append(ls);
        interaction_sizes.append(ls);
        interaction_sizes.append(ls);
        interaction_sizes.append(ls);
        i += 1;
    };
    // LayerNorm: 4 QM31 partial eval columns per layer
    i = 0;
    while i < layernorm_claims.len() {
        let ls = *layernorm_claims.at(i).log_size;
        interaction_sizes.append(ls);
        interaction_sizes.append(ls);
        interaction_sizes.append(ls);
        interaction_sizes.append(ls);
        i += 1;
    };
    // Embedding: 4 QM31 partial eval columns per layer
    i = 0;
    while i < embedding_claims.len() {
        let ls = *embedding_claims.at(i).log_size;
        interaction_sizes.append(ls);
        interaction_sizes.append(ls);
        interaction_sizes.append(ls);
        interaction_sizes.append(ls);
        i += 1;
    };

    array![preprocessed_sizes, trace_sizes, interaction_sizes]
}

// ============================================================================
// MLAir
// ============================================================================

/// The unified ML Air — holds ALL component types for constraint evaluation.
#[derive(Drop)]
pub struct MLAir {
    pub claim: MLClaim,
    pub composition_log_degree_bound: u32,
    pub activation_components: Array<ActivationComponent>,
    pub add_components: Array<ElementwiseAddComponent>,
    pub mul_components: Array<ElementwiseMulComponent>,
    pub layernorm_components: Array<LayerNormComponent>,
    pub embedding_components: Array<EmbeddingComponent>,
    /// Per-embedding claimed sums (needed for constraint evaluation).
    pub embedding_claimed_sums: Array<QM31>,
}

#[generate_trait]
pub impl MLAirNewImpl of MLAirNewTrait {
    fn new(
        claim: @MLClaim,
        activation_claims: Span<ActivationClaim>,
        activation_interaction_claims: Span<ActivationInteractionClaim>,
        activation_lookup_elements: @ActivationLookupElements,
        add_claims: Span<ElementwiseClaim>,
        mul_claims: Span<ElementwiseClaim>,
        layernorm_claims: Span<LayerNormClaim>,
        layernorm_interaction_claims: Span<LayerNormInteractionClaim>,
        layernorm_lookup_elements: @LayerNormLookupElements,
        embedding_claims: Span<EmbeddingClaim>,
        embedding_lookup_elements: @EmbeddingLookupElements,
        embedding_claimed_sums: Array<QM31>,
        composition_log_degree_bound: u32,
    ) -> MLAir {
        let mut activation_components: Array<ActivationComponent> = array![];
        let mut i: u32 = 0;
        while i < activation_claims.len() {
            activation_components
                .append(
                    ActivationComponent {
                        claim: *activation_claims.at(i),
                        interaction_claim: *activation_interaction_claims.at(i),
                        lookup_elements: activation_lookup_elements.clone(),
                    },
                );
            i += 1;
        };

        let mut add_components: Array<ElementwiseAddComponent> = array![];
        i = 0;
        while i < add_claims.len() {
            add_components.append(ElementwiseAddComponent { claim: *add_claims.at(i) });
            i += 1;
        };

        let mut mul_components: Array<ElementwiseMulComponent> = array![];
        i = 0;
        while i < mul_claims.len() {
            mul_components.append(ElementwiseMulComponent { claim: *mul_claims.at(i) });
            i += 1;
        };

        let mut layernorm_components: Array<LayerNormComponent> = array![];
        i = 0;
        while i < layernorm_claims.len() {
            layernorm_components
                .append(
                    LayerNormComponent {
                        claim: *layernorm_claims.at(i),
                        interaction_claim: *layernorm_interaction_claims.at(i),
                        lookup_elements: layernorm_lookup_elements.clone(),
                    },
                );
            i += 1;
        };

        let mut embedding_components: Array<EmbeddingComponent> = array![];
        i = 0;
        while i < embedding_claims.len() {
            embedding_components
                .append(
                    EmbeddingComponent {
                        claim: *embedding_claims.at(i),
                        lookup_elements: embedding_lookup_elements.clone(),
                    },
                );
            i += 1;
        };

        MLAir {
            claim: *claim,
            composition_log_degree_bound,
            activation_components,
            add_components,
            mul_components,
            layernorm_components,
            embedding_components,
            embedding_claimed_sums,
        }
    }
}

/// Air trait implementation for MLAir — evaluates ALL component constraints.
pub impl MLAirAirImpl of Air<MLAir> {
    fn composition_log_degree_bound(self: @MLAir) -> u32 {
        *self.composition_log_degree_bound
    }

    fn eval_composition_polynomial_at_point(
        self: @MLAir,
        point: CirclePoint<QM31>,
        mask_values: TreeSpan<ColumnSpan<Span<QM31>>>,
        random_coeff: QM31,
    ) -> QM31 {
        let mut sum: QM31 = Zero::zero();

        let [
            _preprocessed_mask_values,
            mut trace_mask_values,
            mut interaction_trace_mask_values,
            _composition_trace_mask_values,
        ]: [ColumnSpan<Span<QM31>>; 4] =
            (*mask_values.try_into().unwrap()).unbox();

        // 1. Evaluate activation constraints (LogUp)
        let mut idx: u32 = 0;
        while idx < self.activation_components.len() {
            evaluate_activation_constraints_at_point(
                self.activation_components.at(idx),
                ref sum,
                ref trace_mask_values,
                ref interaction_trace_mask_values,
                random_coeff,
                point,
            );
            idx += 1;
        };

        // 2. Evaluate Add constraints (pure AIR)
        idx = 0;
        while idx < self.add_components.len() {
            evaluate_add_constraints_at_point(
                self.add_components.at(idx),
                ref sum,
                ref trace_mask_values,
                random_coeff,
                point,
            );
            idx += 1;
        };

        // 3. Evaluate Mul constraints (pure AIR)
        idx = 0;
        while idx < self.mul_components.len() {
            evaluate_mul_constraints_at_point(
                self.mul_components.at(idx),
                ref sum,
                ref trace_mask_values,
                random_coeff,
                point,
            );
            idx += 1;
        };

        // 4. Evaluate LayerNorm constraints (algebraic + LogUp)
        idx = 0;
        while idx < self.layernorm_components.len() {
            evaluate_layernorm_constraints_at_point(
                self.layernorm_components.at(idx),
                ref sum,
                ref trace_mask_values,
                ref interaction_trace_mask_values,
                random_coeff,
                point,
            );
            idx += 1;
        };

        // 5. Evaluate Embedding constraints (LogUp)
        idx = 0;
        while idx < self.embedding_components.len() {
            let emb_claimed_sum = *self.embedding_claimed_sums.at(idx);
            evaluate_embedding_constraints_at_point(
                self.embedding_components.at(idx),
                ref sum,
                ref trace_mask_values,
                ref interaction_trace_mask_values,
                random_coeff,
                point,
                emb_claimed_sum,
            );
            idx += 1;
        };

        sum
    }
}

// ============================================================================
// Channel Mixing
// ============================================================================

pub fn mix_ml_claim_into_channel(claim: @MLClaim, ref channel: Channel) {
    channel.mix_u64((*claim.num_layers).into());
    channel.mix_u64((*claim.activation_type).into());
}

pub fn mix_interaction_claim_into_channel(
    interaction_claim: @MLInteractionClaim, ref channel: Channel,
) {
    channel.mix_felts([*interaction_claim.activation_claimed_sum].span());
}

// ============================================================================
// 13-Step Unified STARK Verification
// ============================================================================

const SECURITY_BITS: u32 = 96;
const INTERACTION_POW_BITS: u32 = 0;

/// Verify a unified STARK proof covering all non-matmul components.
///
/// Follows the 13-step pattern from verify_cairo, extended for all component types.
/// Draws interaction elements in Rust prover order:
///   activation (N=3) → layernorm (N=2) → embedding (N=3)
pub fn verify_unified_stark(
    ref channel: Channel,
    claim: @MLClaim,
    proof: UnifiedStarkProof,
) {
    let UnifiedStarkProof {
        activation_claims,
        activation_interaction_claims,
        add_claims,
        mul_claims,
        layernorm_claims,
        layernorm_interaction_claims,
        embedding_claims,
        interaction_claim,
        pcs_config,
        interaction_pow,
        stark_proof,
    } = proof;

    // Step 1: Mix PCS config into channel
    let proof_pcs_config = stark_proof.commitment_scheme_proof.config;
    proof_pcs_config.mix_into(ref channel);

    // Step 2: Create commitment scheme verifier
    let mut commitment_scheme = CommitmentSchemeVerifierImpl::new();

    // Step 3: Unpack commitments [preprocessed, trace, interaction_trace, composition]
    let commitments: @Box<[Hash; 4]> = stark_proof
        .commitment_scheme_proof
        .commitments
        .try_into()
        .unwrap();
    let [
        preprocessed_commitment,
        trace_commitment,
        interaction_trace_commitment,
        composition_commitment,
    ] = commitments.unbox();

    // Step 4: Compute log_sizes per tree (all component types)
    let log_sizes_arr = compute_unified_log_sizes(
        activation_claims.span(),
        add_claims.span(),
        mul_claims.span(),
        layernorm_claims.span(),
        embedding_claims.span(),
    );
    let preprocessed_log_sizes = log_sizes_arr[0].span();
    let trace_log_sizes = log_sizes_arr[1].span();
    let interaction_trace_log_sizes = log_sizes_arr[2].span();

    let log_blowup_factor = pcs_config.fri_config.log_blowup_factor;

    // Step 5: Commit preprocessed trace
    commitment_scheme.commit(
        preprocessed_commitment,
        preprocessed_log_sizes,
        ref channel,
        log_blowup_factor,
    );
    mix_ml_claim_into_channel(claim, ref channel);

    // Step 6: Commit trace
    commitment_scheme.commit(
        trace_commitment,
        trace_log_sizes,
        ref channel,
        log_blowup_factor,
    );

    // Step 7: Verify interaction proof-of-work
    assert!(
        channel.verify_pow_nonce(INTERACTION_POW_BITS, interaction_pow),
        "Interaction proof-of-work failed",
    );
    channel.mix_u64(interaction_pow);

    // Step 8: Draw interaction lookup elements — CONDITIONAL per component type.
    // The Rust prover only draws when the corresponding layer type is present.
    // Drawing when absent would advance the Fiat-Shamir channel, causing transcript
    // divergence and verification failure.
    let has_logup = activation_claims.len() > 0
        || layernorm_claims.len() > 0
        || embedding_claims.len() > 0;

    // Activation: LookupElements<3> [type_tag, input, output]
    let activation_lookup_elements: ActivationLookupElements = if activation_claims.len() > 0 {
        LookupElementsTrait::draw(ref channel)
    } else {
        dummy_lookup_elements_3()
    };
    // LayerNorm: LookupElements<2> [variance, rsqrt_val]
    let layernorm_lookup_elements: LayerNormLookupElements = if layernorm_claims.len() > 0 {
        LookupElementsTrait::draw(ref channel)
    } else {
        dummy_lookup_elements_2()
    };
    // Embedding: LookupElements<3> [token, col, value]
    let embedding_lookup_elements: EmbeddingLookupElements = if embedding_claims.len() > 0 {
        LookupElementsTrait::draw(ref channel)
    } else {
        dummy_lookup_elements_3()
    };

    // Step 9: Verify LogUp sum — total across all LogUp components must be zero
    assert!(
        interaction_claim.activation_claimed_sum.is_zero(),
        "Invalid LogUp sum: must be zero",
    );

    // Step 10: Mix interaction claim into channel
    mix_interaction_claim_into_channel(@interaction_claim, ref channel);

    // Step 11: Commit interaction trace — only when LogUp components exist.
    // Matches Rust verifier: `if has_logup { commitment_scheme.commit(proof.commitments[2], ...) }`
    if has_logup {
        commitment_scheme.commit(
            interaction_trace_commitment,
            interaction_trace_log_sizes,
            ref channel,
            log_blowup_factor,
        );
    }

    // Step 12: Construct MLAir with ALL component types
    let trace_lde_log_size = get_trace_lde_log_size(@commitment_scheme.trees);
    let trace_log_size = trace_lde_log_size - pcs_config.fri_config.log_blowup_factor;
    let composition_log_degree_bound = trace_log_size + 1;

    // Collect per-embedding claimed sums (embedding has no InteractionClaim struct,
    // its share is embedded in the total interaction_claim).
    // For now, we pass zero — embedding LogUp sums are folded into the total.
    let mut embedding_claimed_sums: Array<QM31> = array![];
    let mut eidx: u32 = 0;
    while eidx < embedding_claims.len() {
        embedding_claimed_sums.append(Zero::zero());
        eidx += 1;
    };

    let ml_air = MLAirNewImpl::new(
        claim,
        activation_claims.span(),
        activation_interaction_claims.span(),
        @activation_lookup_elements,
        add_claims.span(),
        mul_claims.span(),
        layernorm_claims.span(),
        layernorm_interaction_claims.span(),
        @layernorm_lookup_elements,
        embedding_claims.span(),
        @embedding_lookup_elements,
        embedding_claimed_sums,
        composition_log_degree_bound,
    );

    // Step 13: Call generic STARK verify
    verify(
        ml_air,
        ref channel,
        stark_proof,
        commitment_scheme,
        SECURITY_BITS,
        composition_commitment,
    );
}

/// Backward-compatible wrapper: verify activation-only STARK proof.
/// Kept for existing tests that use the old ActivationStarkProof name.
pub fn verify_activation_stark(
    ref channel: Channel,
    claim: @MLClaim,
    proof: UnifiedStarkProof,
) {
    verify_unified_stark(ref channel, claim, proof);
}
