//! Element-wise Add/Mul verification via AIR constraints.
//!
//! # Add constraint (degree 1):
//!   output[i] - lhs[i] - rhs[i] = 0
//!
//! # Mul constraint (degree 2):
//!   output[i] - lhs[i] * rhs[i] = 0

use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};

/// Evaluator for element-wise addition: output = lhs + rhs.
#[derive(Debug, Clone)]
pub struct ElementwiseAddEval {
    pub log_n_rows: u32,
}

impl FrameworkEval for ElementwiseAddEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        // output - lhs - rhs = 0  (degree 1 constraint, but bounded by log_size + 1)
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let lhs = eval.next_trace_mask();
        let rhs = eval.next_trace_mask();
        let output = eval.next_trace_mask();

        // Constraint: output - lhs - rhs = 0
        eval.add_constraint(output - lhs - rhs);

        eval
    }
}

/// Type alias for the element-wise add component.
pub type ElementwiseAddComponent = FrameworkComponent<ElementwiseAddEval>;

/// Evaluator for element-wise multiplication: output = lhs * rhs.
#[derive(Debug, Clone)]
pub struct ElementwiseMulEval {
    pub log_n_rows: u32,
}

impl FrameworkEval for ElementwiseMulEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        // output - lhs * rhs = 0  (degree 2 constraint)
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let lhs = eval.next_trace_mask();
        let rhs = eval.next_trace_mask();
        let output = eval.next_trace_mask();

        // Constraint: output - lhs * rhs = 0
        eval.add_constraint(output - lhs * rhs);

        eval
    }
}

/// Type alias for the element-wise mul component.
pub type ElementwiseMulComponent = FrameworkComponent<ElementwiseMulEval>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_eval_log_size() {
        let eval = ElementwiseAddEval { log_n_rows: 4 };
        assert_eq!(eval.log_size(), 4);
        assert_eq!(eval.max_constraint_log_degree_bound(), 5);
    }

    #[test]
    fn test_mul_eval_log_size() {
        let eval = ElementwiseMulEval { log_n_rows: 6 };
        assert_eq!(eval.log_size(), 6);
        assert_eq!(eval.max_constraint_log_degree_bound(), 7);
    }
}
