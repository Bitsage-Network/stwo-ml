//! Rotary Positional Embedding (RoPE) verification.
//!
//! RoPE applies position-dependent rotations to Q/K vectors:
//!   x' = x·cos(θ·m) - y·sin(θ·m)
//!   y' = x·sin(θ·m) + y·cos(θ·m)
//!
//! where θ_j = base^(-2j/d), m = position index.
//!
//! Decomposed into provable operations:
//! 1. Precompute rotation table: (pos, dim_pair) → (cos_val, sin_val) in M31
//! 2. Element-wise rotation using table values
//! 3. LogUp proof: all (cos, sin) pairs come from the precomputed table
//!
//! The rotation factors are deterministic from (seq_len, d_model, base),
//! so the verifier can reconstruct the table independently.

use stwo::core::fields::m31::{BaseField, M31};
use stwo::core::fields::qm31::SecureField;
use stwo::prover::poly::circle::CircleEvaluation;
use stwo::prover::poly::BitReversedOrder;
use stwo::core::poly::circle::CanonicCoset;
use stwo::prover::backend::{Col, Column, ColumnOps};
use stwo_constraint_framework::{
    FrameworkEval, EvalAtRow, RelationEntry,
};
use stwo_constraint_framework::preprocessed_columns::PreProcessedColumnId;

use crate::components::matmul::M31Matrix;
use crate::gadgets::lookup_table::PrecomputedTable;

/// M31 prime: 2^31 - 1
const P: u64 = 2147483647;

// Relation for RoPE rotation lookup: (cos_val, sin_val).
// The table is indexed by (position, dim_pair) but the LogUp relation
// only needs to verify the (cos, sin) pair membership.
stwo_constraint_framework::relation!(RoPERelation, 2);

impl RoPERelation {
    pub fn lookup_elements(&self) -> &stwo_constraint_framework::logup::LookupElements<2> {
        &self.0
    }
}

/// RoPE configuration.
#[derive(Debug, Clone, Copy)]
pub struct RoPEConfig {
    /// Sequence length (number of positions).
    pub seq_len: usize,
    /// Dimension of Q/K vectors per head (must be even).
    pub head_dim: usize,
    /// Frequency base (default: 10000).
    pub base: f64,
    /// Maximum sequence length for table precomputation.
    pub max_seq_len: usize,
}

impl RoPEConfig {
    pub fn new(seq_len: usize, head_dim: usize) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        Self {
            seq_len,
            head_dim,
            base: 10000.0,
            max_seq_len: seq_len,
        }
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }

    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Number of dimension pairs (d/2).
    pub fn num_pairs(&self) -> usize {
        self.head_dim / 2
    }

    /// Total number of rotation entries: positions × dim_pairs.
    pub fn table_size(&self) -> usize {
        self.max_seq_len * self.num_pairs()
    }

    /// Log2 of the padded table size (next power of 2).
    pub fn table_log_size(&self) -> u32 {
        let sz = self.table_size().next_power_of_two();
        sz.ilog2()
    }
}

/// Precomputed rotation factors for RoPE.
///
/// For each (position m, dim_pair j):
///   angle = m × base^(-2j/d)
///   cos_val = quantize(cos(angle))
///   sin_val = quantize(sin(angle))
///
/// Stored as M31 values using scaled fixed-point representation:
///   M31 value = round((float_value + 1.0) × scale)
/// where scale = (P-1)/2 so the full [-1, 1] range maps to [0, P-1].
#[derive(Debug, Clone)]
pub struct RoPETable {
    /// cos values as M31, indexed by [pos * num_pairs + pair_idx]
    pub cos_vals: Vec<M31>,
    /// sin values as M31, indexed by [pos * num_pairs + pair_idx]
    pub sin_vals: Vec<M31>,
    /// Configuration used to build this table.
    pub config: RoPEConfig,
}

/// Fixed-point scale: maps [-1, 1] → [0, P-1].
/// scale = (P-1)/2 = 1073741823
const FP_SCALE: u64 = (P - 1) / 2;

/// Convert a float in [-1, 1] to M31 via fixed-point encoding.
pub fn float_to_m31_signed(val: f64) -> M31 {
    // Map [-1, 1] to [0, P-1]: m31_val = round((val + 1.0) * scale)
    let scaled = ((val + 1.0) * FP_SCALE as f64).round() as u64;
    M31::from(scaled.min(P - 1) as u32)
}

/// Convert M31 back to float in [-1, 1].
pub fn m31_to_float_signed(val: M31) -> f64 {
    (val.0 as f64) / (FP_SCALE as f64) - 1.0
}

/// Build the precomputed RoPE rotation table.
pub fn build_rope_table(config: &RoPEConfig) -> RoPETable {
    let n_pairs = config.num_pairs();
    let n_pos = config.max_seq_len;
    let total = n_pos * n_pairs;

    let mut cos_vals = Vec::with_capacity(total);
    let mut sin_vals = Vec::with_capacity(total);

    for pos in 0..n_pos {
        for j in 0..n_pairs {
            let theta = config.base.powf(-2.0 * j as f64 / config.head_dim as f64);
            let angle = pos as f64 * theta;
            cos_vals.push(float_to_m31_signed(angle.cos()));
            sin_vals.push(float_to_m31_signed(angle.sin()));
        }
    }

    RoPETable { cos_vals, sin_vals, config: *config }
}

/// Build a PrecomputedTable for LogUp verification.
/// The table has entries: (cos_val, sin_val) for each (pos, dim_pair).
pub fn build_rope_lookup_table(config: &RoPEConfig) -> PrecomputedTable {
    let table = build_rope_table(config);
    let pairs: Vec<(M31, M31)> = table.cos_vals.into_iter()
        .zip(table.sin_vals)
        .collect();
    let log_size = config.table_log_size();
    PrecomputedTable::from_pairs(pairs, log_size)
}

/// Apply RoPE to a Q or K matrix in-place.
///
/// Input: matrix of shape (seq_len, head_dim) with M31 values.
/// Each row corresponds to a position m.
/// Adjacent pairs (col 2j, col 2j+1) are rotated by angle θ_j × m.
///
/// The rotation in M31 fixed-point arithmetic:
///   x' = x·cos - y·sin  (mod P, with re-centering)
///   y' = x·sin + y·cos  (mod P, with re-centering)
pub fn apply_rope(
    matrix: &M31Matrix,
    table: &RoPETable,
) -> (M31Matrix, Vec<M31>, Vec<M31>) {
    let seq_len = matrix.rows;
    let head_dim = matrix.cols;
    let n_pairs = head_dim / 2;

    assert!(seq_len <= table.config.max_seq_len,
        "seq_len {} exceeds table max_seq_len {}", seq_len, table.config.max_seq_len);
    assert_eq!(head_dim, table.config.head_dim,
        "head_dim mismatch: matrix has {}, table has {}", head_dim, table.config.head_dim);

    let mut out_data = Vec::with_capacity(seq_len * head_dim);
    let mut cos_used = Vec::with_capacity(seq_len * n_pairs);
    let mut sin_used = Vec::with_capacity(seq_len * n_pairs);

    for pos in 0..seq_len {
        for j in 0..n_pairs {
            let x = matrix.data[pos * head_dim + 2 * j];
            let y = matrix.data[pos * head_dim + 2 * j + 1];

            let table_idx = pos * n_pairs + j;
            let cos_m31 = table.cos_vals[table_idx];
            let sin_m31 = table.sin_vals[table_idx];

            cos_used.push(cos_m31);
            sin_used.push(sin_m31);

            // Fixed-point rotation in M31:
            // cos/sin are encoded as (float + 1.0) * scale
            // We decode, multiply, re-encode.
            let cos_f = m31_to_float_signed(cos_m31);
            let sin_f = m31_to_float_signed(sin_m31);
            let x_f = m31_to_float_signed(x);
            let y_f = m31_to_float_signed(y);

            let x_rot = x_f * cos_f - y_f * sin_f;
            let y_rot = x_f * sin_f + y_f * cos_f;

            out_data.push(float_to_m31_signed(x_rot.clamp(-1.0, 1.0)));
            out_data.push(float_to_m31_signed(y_rot.clamp(-1.0, 1.0)));
        }
    }

    let out = M31Matrix {
        rows: seq_len,
        cols: head_dim,
        data: out_data,
    };

    (out, cos_used, sin_used)
}

/// Evaluator for RoPE constraints.
///
/// Trace layout (7 columns):
///   Col 0: input_x     — first element of pair
///   Col 1: input_y     — second element of pair
///   Col 2: cos_val     — rotation cosine from table
///   Col 3: sin_val     — rotation sine from table
///   Col 4: output_x    — rotated first element
///   Col 5: output_y    — rotated second element
///   Col 6: multiplicity — LogUp multiplicity
///
/// Preprocessed (2 columns):
///   Col 0: table cos values
///   Col 1: table sin values
///
/// Constraints:
///   output_x = input_x * cos_val - input_y * sin_val  (modular)
///   output_y = input_x * sin_val + input_y * cos_val  (modular)
///   LogUp: (cos_val, sin_val) ∈ precomputed table
#[derive(Debug, Clone)]
pub struct RoPEEval {
    pub log_n_rows: u32,
    pub lookup_elements: RoPERelation,
    pub claimed_sum: SecureField,
}

impl FrameworkEval for RoPEEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // Preprocessed table columns
        let table_cos = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "rope_table_cos".into(),
        });
        let table_sin = eval.get_preprocessed_column(PreProcessedColumnId {
            id: "rope_table_sin".into(),
        });

        // Execution trace columns
        let input_x = eval.next_trace_mask();
        let input_y = eval.next_trace_mask();
        let cos_val = eval.next_trace_mask();
        let sin_val = eval.next_trace_mask();
        let output_x = eval.next_trace_mask();
        let output_y = eval.next_trace_mask();
        let multiplicity = eval.next_trace_mask();

        // Constraint: output_x = input_x * cos_val - input_y * sin_val
        eval.add_constraint(
            output_x - (input_x.clone() * cos_val.clone() - input_y.clone() * sin_val.clone()),
        );

        // Constraint: output_y = input_x * sin_val + input_y * cos_val
        eval.add_constraint(
            output_y - (input_x * sin_val.clone() + input_y * cos_val.clone()),
        );

        // LogUp: table side
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            -E::EF::from(multiplicity),
            &[table_cos, table_sin],
        ));

        // LogUp: trace side — proves (cos_val, sin_val) ∈ table
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::from(E::F::from(BaseField::from(1))),
            &[cos_val, sin_val],
        ));

        eval.finalize_logup_in_pairs();

        eval
    }
}

/// Generate execution trace columns for RoPE verification.
///
/// Returns 7 trace columns + multiplicities vector for LogUp.
pub fn generate_rope_trace<B: stwo::prover::backend::Backend>(
    input_x: &[M31],
    input_y: &[M31],
    cos_vals: &[M31],
    sin_vals: &[M31],
    output_x: &[M31],
    output_y: &[M31],
    table: &PrecomputedTable,
    log_size: u32,
) -> (Vec<CircleEvaluation<B, BaseField, BitReversedOrder>>, Vec<M31>)
where
    B: ColumnOps<BaseField>,
{
    let size = 1usize << log_size;
    let n = input_x.len().min(size);
    let domain = CanonicCoset::new(log_size).circle_domain();

    // Compute multiplicities
    let multiplicities = crate::components::activation::compute_multiplicities(
        cos_vals,
        table,
    );

    // Build 7 trace columns
    let mut col_ix = Col::<B, BaseField>::zeros(size);
    let mut col_iy = Col::<B, BaseField>::zeros(size);
    let mut col_cos = Col::<B, BaseField>::zeros(size);
    let mut col_sin = Col::<B, BaseField>::zeros(size);
    let mut col_ox = Col::<B, BaseField>::zeros(size);
    let mut col_oy = Col::<B, BaseField>::zeros(size);
    let mut col_mult = Col::<B, BaseField>::zeros(size);

    // Padding values (use first table entry)
    let pad_cos = table.inputs.first().copied().unwrap_or(M31::from(0));
    let pad_sin = table.outputs.first().copied().unwrap_or(M31::from(0));

    for i in 0..n {
        col_ix.set(i, input_x[i]);
        col_iy.set(i, input_y[i]);
        col_cos.set(i, cos_vals[i]);
        col_sin.set(i, sin_vals[i]);
        col_ox.set(i, output_x[i]);
        col_oy.set(i, output_y[i]);
        col_mult.set(i, multiplicities[i]);
    }

    // Padding rows reference valid table entries
    for i in n..size {
        col_cos.set(i, pad_cos);
        col_sin.set(i, pad_sin);
    }

    let evals = vec![
        CircleEvaluation::new(domain, col_ix),
        CircleEvaluation::new(domain, col_iy),
        CircleEvaluation::new(domain, col_cos),
        CircleEvaluation::new(domain, col_sin),
        CircleEvaluation::new(domain, col_ox),
        CircleEvaluation::new(domain, col_oy),
        CircleEvaluation::new(domain, col_mult),
    ];

    let mults_vec: Vec<M31> = (0..size)
        .map(|i| if i < multiplicities.len() { multiplicities[i] } else { M31::from(0) })
        .collect();

    (evals, mults_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_m31_roundtrip() {
        // Test encoding/decoding of float values
        let vals = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.707107, -0.707107];
        for v in vals {
            let m = float_to_m31_signed(v);
            let back = m31_to_float_signed(m);
            assert!((back - v).abs() < 1e-6, "roundtrip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_build_rope_table() {
        let config = RoPEConfig::new(4, 4); // 4 positions, 4-dim (2 pairs)
        let table = build_rope_table(&config);

        assert_eq!(table.cos_vals.len(), 8); // 4 pos × 2 pairs
        assert_eq!(table.sin_vals.len(), 8);

        // Position 0: angle = 0 for all pairs → cos=1, sin=0
        let cos_00 = m31_to_float_signed(table.cos_vals[0]);
        let sin_00 = m31_to_float_signed(table.sin_vals[0]);
        assert!((cos_00 - 1.0).abs() < 1e-6, "cos(0) should be 1, got {}", cos_00);
        assert!(sin_00.abs() < 1e-6, "sin(0) should be 0, got {}", sin_00);
    }

    #[test]
    fn test_apply_rope() {
        let config = RoPEConfig::new(2, 4); // 2 positions, 4-dim
        let table = build_rope_table(&config);

        // Simple input: identity-like
        let input = M31Matrix {
            rows: 2,
            cols: 4,
            data: vec![
                float_to_m31_signed(1.0), float_to_m31_signed(0.0),
                float_to_m31_signed(0.0), float_to_m31_signed(1.0),
                float_to_m31_signed(0.5), float_to_m31_signed(0.5),
                float_to_m31_signed(-0.5), float_to_m31_signed(0.5),
            ],
        };

        let (rotated, cos_used, sin_used) = apply_rope(&input, &table);
        assert_eq!(rotated.rows, 2);
        assert_eq!(rotated.cols, 4);
        assert_eq!(cos_used.len(), 4); // 2 positions × 2 pairs
        assert_eq!(sin_used.len(), 4);

        // Position 0: angle = 0, so output should equal input
        let x0 = m31_to_float_signed(rotated.data[0]);
        let y0 = m31_to_float_signed(rotated.data[1]);
        assert!((x0 - 1.0).abs() < 0.01, "pos 0, pair 0: x should be ~1.0, got {}", x0);
        assert!(y0.abs() < 0.01, "pos 0, pair 0: y should be ~0.0, got {}", y0);
    }

    #[test]
    fn test_rope_config() {
        let config = RoPEConfig::new(128, 64);
        assert_eq!(config.num_pairs(), 32);
        assert_eq!(config.table_size(), 128 * 32);
        assert!(config.table_log_size() >= 12); // 4096 entries → log2 ≥ 12
    }

    #[test]
    fn test_rope_rotation_orthogonality() {
        // Verify rotation preserves vector norm (approximately)
        let config = RoPEConfig::new(8, 4);
        let table = build_rope_table(&config);

        for pos in 0..8 {
            for j in 0..2 {
                let idx = pos * 2 + j;
                let c = m31_to_float_signed(table.cos_vals[idx]);
                let s = m31_to_float_signed(table.sin_vals[idx]);
                // cos² + sin² ≈ 1
                let norm = c * c + s * s;
                assert!((norm - 1.0).abs() < 0.001,
                    "pos={}, pair={}: cos²+sin²={}, expected 1.0", pos, j, norm);
            }
        }
    }
}
