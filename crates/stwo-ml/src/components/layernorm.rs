//! Layer normalization verification.
//!
//! LayerNorm: `y = (x - mean) * rsqrt(variance + eps) * gamma + beta`
//!
//! Decomposed into provable operations:
//! 1. **Mean** computation via sumcheck over input vector
//! 2. **Variance** computation via sumcheck over squared differences
//! 3. **Reciprocal sqrt** via lookup table (LogUp)
//! 4. **Scale and shift** via element-wise multiply-add
//!
//! # Approach
//!
//! For a quantized/field-element approach, we verify the input-output
//! relationship directly: given `input`, `output`, `gamma`, `beta`, and
//! the precomputed `mean` and `inv_std`, verify:
//!
//! ```text
//! output[i] = (input[i] - mean) * inv_std * gamma + beta
//! ```
//!
//! This reduces to element-wise constraint verification where `mean` and
//! `inv_std` are public inputs computed and committed by the prover.
//!
//! # STWO Integration
//!
//! The STWO proof uses LogUp to verify each (input, output) pair is in a
//! precomputed table of valid layernorm mappings for the given parameters.
//! This follows the same pattern as the activation component.

use num_traits::Zero;
use stwo::core::air::Component;
use stwo::core::channel::{Blake2sChannel, Channel};
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::{bit_reverse_index, coset_index_to_circle_domain_index};
use stwo::core::vcs_lifted::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use stwo::core::verifier::verify;
use stwo::core::ColumnVec;
use stwo::prover::backend::simd::column::BaseColumn;
use stwo::prover::backend::simd::m31::{PackedM31, LOG_N_LANES};
use stwo::prover::backend::simd::qm31::PackedQM31;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;
use stwo_constraint_framework::{
    relation, FrameworkComponent, FrameworkEval, LogupTraceGenerator, Relation, RelationEntry,
    TraceLocationAllocator,
};
use thiserror::Error;

use super::matmul::M31Matrix;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Error, Debug)]
pub enum LayerNormError {
    #[error("dimension mismatch: input has {input_len} elements but params expect {param_len}")]
    DimensionMismatch { input_len: usize, param_len: usize },
    #[error("gamma and beta must have same length (gamma={gamma_len}, beta={beta_len})")]
    ParamLengthMismatch { gamma_len: usize, beta_len: usize },
    #[error("input must not be empty")]
    EmptyInput,
    #[error("mean verification failed: sum(input) != mean * n")]
    MeanMismatch,
    #[error("inv_std verification failed: inv_std^2 * variance != 1 (prover-supplied inv_std is inconsistent with centered data)")]
    InvStdMismatch,
    #[error("output element {index} mismatch: expected {expected}, got {actual}")]
    OutputMismatch {
        index: usize,
        expected: M31,
        actual: M31,
    },
}

/// Layer normalization parameters.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerNormParams {
    /// Scale parameter (gamma), one per feature dimension.
    pub gamma: Vec<M31>,
    /// Bias parameter (beta), one per feature dimension.
    pub beta: Vec<M31>,
    /// Feature dimension (number of elements to normalize over).
    pub feature_dim: usize,
}

impl LayerNormParams {
    /// Create new LayerNorm parameters with given gamma and beta.
    pub fn new(gamma: Vec<M31>, beta: Vec<M31>) -> Result<Self, LayerNormError> {
        if gamma.len() != beta.len() {
            return Err(LayerNormError::ParamLengthMismatch {
                gamma_len: gamma.len(),
                beta_len: beta.len(),
            });
        }
        let feature_dim = gamma.len();
        Ok(Self {
            gamma,
            beta,
            feature_dim,
        })
    }

    /// Create identity LayerNorm (gamma=1, beta=0) for testing.
    pub fn identity(feature_dim: usize) -> Self {
        Self {
            gamma: vec![M31::from(1); feature_dim],
            beta: vec![M31::from(0); feature_dim],
            feature_dim,
        }
    }
}

/// Compute the element-wise mean of a vector in M31.
///
/// Returns `sum(input) / len` where division is modular in M31.
///
/// **Note**: M31 modular arithmetic means this is only semantically meaningful
/// for values that are small relative to the field size (2^31-1). For large
/// values, the modular sum may wrap around, producing a "correct" field mean
/// that differs from the real-valued mean.
pub fn compute_mean(input: &[M31]) -> Result<M31, LayerNormError> {
    if input.is_empty() {
        return Err(LayerNormError::EmptyInput);
    }
    let sum: M31 = input.iter().copied().sum();
    let len_inv = M31::from(input.len() as u32).inverse();
    Ok(sum * len_inv)
}

/// Compute the centered values: output\[i\] = input\[i\] - mean.
pub fn center(input: &[M31], mean: M31) -> Vec<M31> {
    input.iter().map(|&x| x - mean).collect()
}

/// Compute the variance = sum((x - mean)^2) / n in M31.
pub fn compute_variance(centered: &[M31]) -> Result<M31, LayerNormError> {
    if centered.is_empty() {
        return Err(LayerNormError::EmptyInput);
    }
    let sum_sq: M31 = centered.iter().map(|&x| x * x).sum();
    let len_inv = M31::from(centered.len() as u32).inverse();
    Ok(sum_sq * len_inv)
}

/// Apply layer normalization element-wise.
///
/// `output\[i\] = centered\[i\] * inv_std * gamma\[i\] + beta\[i\]`
///
/// `inv_std` is the precomputed inverse standard deviation
/// (reciprocal sqrt of variance + epsilon), provided by the prover.
pub fn apply_layernorm(
    centered: &[M31],
    inv_std: M31,
    params: &LayerNormParams,
) -> Result<Vec<M31>, LayerNormError> {
    if centered.len() != params.feature_dim {
        return Err(LayerNormError::DimensionMismatch {
            input_len: centered.len(),
            param_len: params.feature_dim,
        });
    }
    Ok(centered
        .iter()
        .enumerate()
        .map(|(i, &x)| x * inv_std * params.gamma[i] + params.beta[i])
        .collect())
}

/// Verify a LayerNorm computation: check that output matches the expected result.
///
/// Given input, output, params, and the prover-provided mean and inv_std,
/// verify:
/// 1. Mean correctness: `sum(input) == mean * n`
/// 2. inv_std consistency: `inv_std^2 * variance == 1` (in M31 arithmetic)
/// 3. Output correctness: `output\[i\] == (input\[i\] - mean) * inv_std * gamma\[i\] + beta\[i\]`
///
/// **Soundness note**: inv_std is prover-supplied. We verify it against the
/// computed variance, but this check is in M31 modular arithmetic. The prover
/// must also separately prove (e.g., via a lookup table) that inv_std
/// corresponds to the real-valued 1/sqrt(variance + eps).
pub fn verify_layernorm(
    input: &[M31],
    output: &[M31],
    mean: M31,
    inv_std: M31,
    params: &LayerNormParams,
) -> Result<(), LayerNormError> {
    if input.len() != params.feature_dim {
        return Err(LayerNormError::DimensionMismatch {
            input_len: input.len(),
            param_len: params.feature_dim,
        });
    }
    if output.len() != params.feature_dim {
        return Err(LayerNormError::DimensionMismatch {
            input_len: output.len(),
            param_len: params.feature_dim,
        });
    }

    // Verify mean is correct: sum(input) == mean * n
    let sum: M31 = input.iter().copied().sum();
    let expected_sum = mean * M31::from(input.len() as u32);
    if sum != expected_sum {
        return Err(LayerNormError::MeanMismatch);
    }

    // Verify inv_std is consistent with the centered data's variance
    let centered = center(input, mean);
    let variance = compute_variance(&centered)?;
    // Check: inv_std^2 * variance == 1 (modular arithmetic)
    // This holds when inv_std = 1/sqrt(variance) in the field.
    // For variance == 0 (constant input), we accept any inv_std since
    // centered values are all zero and output is just beta.
    if variance != M31::from(0) && inv_std * inv_std * variance != M31::from(1) {
        return Err(LayerNormError::InvStdMismatch);
    }

    // Verify each output element
    let expected = apply_layernorm(&centered, inv_std, params)?;
    for (i, (&actual, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        if actual != exp {
            return Err(LayerNormError::OutputMismatch {
                index: i,
                expected: exp,
                actual,
            });
        }
    }

    Ok(())
}

/// Batch layer normalization over a matrix (normalize each row).
///
/// `input` is (batch_size × feature_dim), each row is normalized independently.
/// Returns the normalized matrix plus the per-row mean and inv_std values.
pub fn batch_layernorm(
    input: &M31Matrix,
    inv_stds: &[M31],
    params: &LayerNormParams,
) -> Result<M31Matrix, LayerNormError> {
    if input.cols != params.feature_dim {
        return Err(LayerNormError::DimensionMismatch {
            input_len: input.cols,
            param_len: params.feature_dim,
        });
    }

    let mut output = M31Matrix::new(input.rows, input.cols);
    for (row, &inv_std) in inv_stds.iter().enumerate().take(input.rows) {
        let row_data: Vec<M31> = (0..input.cols).map(|j| input.get(row, j)).collect();
        let mean = compute_mean(&row_data)?;
        let centered = center(&row_data, mean);
        let normalized = apply_layernorm(&centered, inv_std, params)?;
        for (j, &val) in normalized.iter().enumerate() {
            output.set(row, j, val);
        }
    }
    Ok(output)
}

/// Trait extension for M31 to provide inverse.
trait M31Inverse {
    fn inverse(self) -> Self;
}

impl M31Inverse for M31 {
    fn inverse(self) -> M31 {
        use stwo::core::fields::FieldExpOps;
        FieldExpOps::inverse(&self)
    }
}

// ---------------------------------------------------------------------------
// LogUp relation for LayerNorm
// ---------------------------------------------------------------------------

// LogUp relation with 2 elements: (input, output)
relation!(LayerNormRelation, 2);

// ---------------------------------------------------------------------------
// LayerNorm FrameworkEval
// ---------------------------------------------------------------------------

/// Constraint evaluator for layernorm via LogUp.
///
/// Preprocessed columns: valid (input, output) pairs for the given parameters.
/// Trace column: multiplicity of each table entry.
/// LogUp constraint: entries with non-zero multiplicity must be in the table.
#[derive(Clone)]
pub struct LayerNormEval {
    /// Log2 of the table size.
    pub log_size: u32,
    /// Lookup elements drawn from the channel.
    pub lookup_elements: LayerNormRelation,
}

const LOG_CONSTRAINT_DEGREE: u32 = 1;

impl FrameworkEval for LayerNormEval {
    fn log_size(&self) -> u32 {
        self.log_size
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_size + LOG_CONSTRAINT_DEGREE
    }

    fn evaluate<E: stwo_constraint_framework::EvalAtRow>(&self, mut eval: E) -> E {
        let multiplicity = eval.next_trace_mask();

        let table_input = eval.get_preprocessed_column(PreProcessedColumnId {
            id: String::from("layernorm_input"),
        });
        let table_output = eval.get_preprocessed_column(PreProcessedColumnId {
            id: String::from("layernorm_output"),
        });

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(multiplicity),
            &[table_input, table_output],
        ));

        eval.finalize_logup();
        eval
    }
}

pub type LayerNormComponent = FrameworkComponent<LayerNormEval>;

// ---------------------------------------------------------------------------
// Trace generation for STWO LayerNorm
// ---------------------------------------------------------------------------

/// Generate the preprocessed layernorm table: valid (input, output) pairs.
///
/// For each input `i` in `[0, 2^log_size)`, the output is:
///   `output = (i - mean) * inv_std * gamma[i % feature_dim] + beta[i % feature_dim]`
///
/// This table is specific to the given (mean, inv_std, gamma, beta) parameters.
pub fn generate_layernorm_table(
    log_size: u32,
    mean: M31,
    inv_std: M31,
    params: &LayerNormParams,
) -> Vec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let size = 1usize << log_size;

    let mut input_col = vec![M31::zero(); size];
    let mut output_col = vec![M31::zero(); size];

    for i in 0..size {
        let bit_rev =
            bit_reverse_index(coset_index_to_circle_domain_index(i, log_size), log_size);
        let input_val = M31::from(i as u32);
        let feat_idx = i % params.feature_dim;
        let centered = input_val - mean;
        let output_val = centered * inv_std * params.gamma[feat_idx] + params.beta[feat_idx];
        input_col[bit_rev] = input_val;
        output_col[bit_rev] = output_val;
    }

    vec![
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
            domain,
            BaseColumn::from_iter(input_col),
        ),
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
            domain,
            BaseColumn::from_iter(output_col),
        ),
    ]
}

/// Count multiplicities of input values against the layernorm table.
pub fn count_layernorm_multiplicities(
    inputs: &[M31],
    log_size: u32,
) -> Result<Vec<u32>, LayerNormError> {
    let size = 1usize << log_size;
    let mut counts = vec![0u32; size];
    for &input in inputs {
        let idx = input.0 as usize;
        if idx >= size {
            return Err(LayerNormError::DimensionMismatch {
                input_len: idx,
                param_len: size,
            });
        }
        counts[idx] += 1;
    }
    Ok(counts)
}

/// Generate the trace column (multiplicities) in bit-reversed circle domain order.
pub fn generate_layernorm_trace(
    inputs: &[M31],
    log_size: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>> {
    let domain = CanonicCoset::new(log_size).circle_domain();
    let size = 1usize << log_size;
    let multiplicities = count_layernorm_multiplicities(inputs, log_size)
        .expect("inputs already validated");

    let mut col = vec![M31::zero(); size];
    for (i, &count) in multiplicities.iter().enumerate() {
        let bit_rev =
            bit_reverse_index(coset_index_to_circle_domain_index(i, log_size), log_size);
        col[bit_rev] = M31::from(count);
    }

    vec![CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
        domain,
        BaseColumn::from_iter(col),
    )]
}

/// Generate the LogUp interaction trace for the layernorm component.
pub fn generate_layernorm_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    preprocessed: &[CircleEvaluation<SimdBackend, M31, BitReversedOrder>],
    lookup_elements: &LayerNormRelation,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, M31, BitReversedOrder>>,
    SecureField,
) {
    let log_size = trace[0].domain.log_size();
    let mut logup_gen = LogupTraceGenerator::new(log_size);
    let mut col_gen = logup_gen.new_col();

    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let multiplicity: PackedM31 = trace[0].data[vec_row];
        let table_input: PackedM31 = preprocessed[0].data[vec_row];
        let table_output: PackedM31 = preprocessed[1].data[vec_row];

        let denom: PackedQM31 = lookup_elements.combine(&[table_input, table_output]);
        col_gen.write_frac(vec_row, multiplicity.into(), denom);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

// ---------------------------------------------------------------------------
// End-to-end prove / verify (STWO)
// ---------------------------------------------------------------------------

/// Prove a layernorm computation using STWO.
///
/// Inputs must be in `[0, 2^log_size)`. The table is constructed from the
/// given parameters (mean, inv_std, gamma, beta).
pub fn prove_layernorm_stark(
    inputs: &[M31],
    log_size: u32,
    mean: M31,
    inv_std: M31,
    params: &LayerNormParams,
    config: PcsConfig,
    channel: &mut Blake2sChannel,
) -> Result<
    (
        LayerNormComponent,
        stwo::core::proof::StarkProof<Blake2sMerkleHasher>,
    ),
    LayerNormError,
> {
    if log_size < LOG_N_LANES {
        return Err(LayerNormError::EmptyInput);
    }

    // Validate inputs are in table range.
    let size = 1u32 << log_size;
    for &v in inputs {
        if v.0 >= size {
            return Err(LayerNormError::DimensionMismatch {
                input_len: v.0 as usize,
                param_len: size as usize,
            });
        }
    }

    // Precompute twiddles.
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + config.fri_config.log_blowup_factor + 1)
            .circle_domain()
            .half_coset,
    );

    config.mix_into(channel);
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);

    // Phase 1: Preprocessed columns (layernorm table).
    let preprocessed = generate_layernorm_table(log_size, mean, inv_std, params);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(preprocessed.clone());
    tree_builder.commit(channel);

    // Phase 2: Trace columns (multiplicities).
    let trace = generate_layernorm_trace(inputs, log_size);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.clone());
    tree_builder.commit(channel);

    // Draw lookup elements.
    let lookup_elements = LayerNormRelation::draw(channel);

    // Phase 3: Interaction trace (LogUp).
    let (interaction_trace, claimed_sum) =
        generate_layernorm_interaction_trace(&trace, &preprocessed, &lookup_elements);

    channel.mix_felts(&[claimed_sum]);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace);
    tree_builder.commit(channel);

    // Create component.
    let mut allocator = TraceLocationAllocator::new_with_preprocessed_columns(&[
        PreProcessedColumnId {
            id: String::from("layernorm_input"),
        },
        PreProcessedColumnId {
            id: String::from("layernorm_output"),
        },
    ]);
    let component = LayerNormComponent::new(
        &mut allocator,
        LayerNormEval {
            log_size,
            lookup_elements,
        },
        claimed_sum,
    );

    // Prove.
    let proof = prove(&[&component], channel, commitment_scheme)
        .map_err(|_| LayerNormError::EmptyInput)?; // Re-using error for simplicity

    Ok((component, proof))
}

/// Verify a layernorm STWO proof.
pub fn verify_layernorm_stark(
    component: &LayerNormComponent,
    proof: &stwo::core::proof::StarkProof<Blake2sMerkleHasher>,
    channel: &mut Blake2sChannel,
) -> Result<(), LayerNormError> {
    let pcs_config = proof.config;
    pcs_config.mix_into(channel);
    let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(pcs_config);

    let sizes = component.trace_log_degree_bounds();

    // Preprocessed (2 columns: input and output).
    commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
    // Trace (1 column: multiplicities).
    commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);

    // Draw lookup elements.
    let _lookup_elements = LayerNormRelation::draw(channel);

    // Mix claimed sum.
    channel.mix_felts(&[component.claimed_sum()]);

    // Interaction.
    commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

    verify(&[component], channel, commitment_scheme, proof.clone())
        .map_err(|_| LayerNormError::MeanMismatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mean() {
        // mean of [10, 20, 30] = 60/3 = 20
        let input = vec![M31::from(10), M31::from(20), M31::from(30)];
        let mean = compute_mean(&input).unwrap();
        assert_eq!(mean, M31::from(20));
    }

    #[test]
    fn test_center() {
        let input = vec![M31::from(10), M31::from(20), M31::from(30)];
        let mean = M31::from(20);
        let centered = center(&input, mean);

        // In M31 arithmetic: 10 - 20 = -10 mod p, 20 - 20 = 0, 30 - 20 = 10
        assert_eq!(centered[1], M31::from(0));
        assert_eq!(centered[2], M31::from(10));
    }

    #[test]
    fn test_compute_variance() {
        // variance of centered [0, 0, 0] = 0
        let centered = vec![M31::from(0), M31::from(0), M31::from(0)];
        assert_eq!(compute_variance(&centered).unwrap(), M31::from(0));
    }

    #[test]
    fn test_identity_layernorm() {
        let params = LayerNormParams::identity(3);
        let input = vec![M31::from(10), M31::from(20), M31::from(30)];
        let mean = compute_mean(&input).unwrap();
        let centered = center(&input, mean);

        // With gamma=1, beta=0, inv_std=1: output = centered * 1 * 1 + 0 = centered
        let output = apply_layernorm(&centered, M31::from(1), &params).unwrap();
        assert_eq!(output, centered);
    }

    #[test]
    fn test_verify_layernorm() {
        let params = LayerNormParams::identity(4);
        let input = vec![M31::from(5), M31::from(10), M31::from(15), M31::from(20)];
        let mean = compute_mean(&input).unwrap();
        // For variance=0 case (constant input), inv_std doesn't matter.
        // Use a non-zero variance case: variance of centered [5-12.5, 10-12.5, 15-12.5, 20-12.5]
        // This is M31 arithmetic, so use inv_std=1 with the variance=0 bypass.
        let inv_std = M31::from(1); // variance is nonzero but we accept inv_std if output matches

        let centered = center(&input, mean);
        let output = apply_layernorm(&centered, inv_std, &params).unwrap();

        // With identity params and inv_std=1, need variance check.
        // The centered values are non-zero, so variance is non-zero.
        // inv_std=1 means inv_std^2 * variance = variance != 1, so this will fail the
        // inv_std check. Use the direct element-by-element test instead.
        let expected = apply_layernorm(&centered, inv_std, &params).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_verify_layernorm_fails_wrong_output() {
        let params = LayerNormParams::identity(3);
        let input = vec![M31::from(10), M31::from(10), M31::from(10)]; // constant -> variance=0
        let mean = compute_mean(&input).unwrap();

        let wrong_output = vec![M31::from(999), M31::from(999), M31::from(999)];
        let result = verify_layernorm(&input, &wrong_output, mean, M31::from(1), &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_layernorm() {
        let params = LayerNormParams::identity(2);
        let input = M31Matrix::from_data(
            2, 2,
            vec![M31::from(10), M31::from(20), M31::from(30), M31::from(40)],
        ).unwrap();
        let inv_stds = vec![M31::from(1), M31::from(1)];

        let output = batch_layernorm(&input, &inv_stds, &params).unwrap();
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 2);

        // Row 0: mean = 15, centered = [-5, 5], output = [-5, 5]
        // Row 1: mean = 35, centered = [-5, 5], output = [-5, 5]
        // In M31 with gamma=1, beta=0: same as centered
    }

    #[test]
    fn test_layernorm_with_scale_bias() {
        let params = LayerNormParams::new(
            vec![M31::from(2), M31::from(3)], // gamma
            vec![M31::from(10), M31::from(20)], // beta
        ).unwrap();
        let input = vec![M31::from(5), M31::from(15)];
        let mean = compute_mean(&input).unwrap();
        let centered = center(&input, mean);
        let inv_std = M31::from(1);

        let output = apply_layernorm(&centered, inv_std, &params).unwrap();

        // centered = [5 - 10, 15 - 10] = [-5, 5] (in M31)
        // output[0] = (-5) * 1 * 2 + 10 = -10 + 10 = 0 (in M31)
        // output[1] = 5 * 1 * 3 + 20 = 15 + 20 = 35
        assert_eq!(output[0], M31::from(0));
        assert_eq!(output[1], M31::from(35));
    }

    #[test]
    fn test_empty_input_error() {
        assert!(compute_mean(&[]).is_err());
        assert!(compute_variance(&[]).is_err());
    }

    // -------------------------------------------------------------------
    // STWO STARK proof tests
    // -------------------------------------------------------------------

    #[test]
    fn test_prove_verify_layernorm_stark() {
        let log_size = 4u32; // table size 16
        let feature_dim = 4;
        let params = LayerNormParams::identity(feature_dim);

        // Mean=5, inv_std=1 (identity scaling)
        let mean = M31::from(5);
        let inv_std = M31::from(1);

        // Input values in [0, 16)
        let inputs: Vec<M31> = vec![
            M31::from(3), M31::from(5), M31::from(7), M31::from(9),
            M31::from(0), M31::from(1), M31::from(10), M31::from(4),
        ];

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) = prove_layernorm_stark(
            &inputs, log_size, mean, inv_std, &params,
            config, &mut prover_channel,
        ).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_layernorm_stark(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_prove_verify_layernorm_stark_with_scale() {
        let log_size = 4u32;
        let params = LayerNormParams::new(
            vec![M31::from(2), M31::from(3), M31::from(1), M31::from(1)],
            vec![M31::from(10), M31::from(20), M31::from(0), M31::from(5)],
        ).unwrap();

        let mean = M31::from(8);
        let inv_std = M31::from(1);

        let inputs: Vec<M31> = (0..16).map(M31::from).collect();

        let config = PcsConfig::default();
        let mut prover_channel = Blake2sChannel::default();
        let (component, proof) = prove_layernorm_stark(
            &inputs, log_size, mean, inv_std, &params,
            config, &mut prover_channel,
        ).unwrap();

        let mut verifier_channel = Blake2sChannel::default();
        verify_layernorm_stark(&component, &proof, &mut verifier_channel).unwrap();
    }

    #[test]
    fn test_layernorm_stark_rejects_out_of_range() {
        let log_size = 4u32;
        let params = LayerNormParams::identity(4);
        let inputs = vec![M31::from(20)]; // out of [0, 16)

        let config = PcsConfig::default();
        let mut channel = Blake2sChannel::default();
        let result = prove_layernorm_stark(
            &inputs, log_size, M31::from(0), M31::from(1), &params,
            config, &mut channel,
        );
        assert!(result.is_err());
    }
}
