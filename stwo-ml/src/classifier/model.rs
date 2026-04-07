//! Classifier model construction and weight loading.
//!
//! Builds the `64 → 64 (ReLU) → 32 (ReLU) → 3` MLP graph and loads
//! quantized weights from raw M31 arrays or file paths.

use stwo::core::fields::m31::M31;

use crate::compiler::graph::{ComputationGraph, GraphBuilder, GraphWeights};
use crate::components::activation::ActivationType;
use crate::components::matmul::M31Matrix;

use super::types::*;

/// A loaded classifier model ready for proving.
pub struct ClassifierModel {
    /// Computation graph (5 nodes: linear → relu → linear → relu → linear).
    pub graph: ComputationGraph,
    /// Quantized weight matrices keyed by node ID.
    pub weights: GraphWeights,
}

/// Build the classifier computation graph.
///
/// Architecture: `(1, 64) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(3)`
///
/// Node IDs:
/// - 0: Linear(64→64), weight shape (64, 64)
/// - 1: ReLU(64)
/// - 2: Linear(64→32), weight shape (64, 32)
/// - 3: ReLU(32)
/// - 4: Linear(32→3), weight shape (32, 3)
pub fn build_classifier_graph() -> ComputationGraph {
    let mut builder = GraphBuilder::new((1, INPUT_DIM));
    builder
        .linear(64)
        .activation(ActivationType::ReLU)
        .linear(32)
        .activation(ActivationType::ReLU)
        .linear(NUM_CLASSES);
    builder.build()
}

/// Create a classifier model with random weights (for testing).
///
/// Weights are deterministic based on a seed pattern — same seed produces
/// identical weights across runs.
pub fn build_test_classifier() -> ClassifierModel {
    let graph = build_classifier_graph();
    let mut weights = GraphWeights::new();

    // Layer 0: Linear(64→64) — node 0
    let mut w0 = M31Matrix::new(64, 64);
    for i in 0..64 {
        for j in 0..64 {
            w0.set(i, j, M31::from(((i * 64 + j) * 7 + 3) as u32 % 127));
        }
    }
    weights.add_weight(0, w0);

    // Layer 2: Linear(64→32) — node 2 (node 1 is ReLU)
    let mut w2 = M31Matrix::new(64, 32);
    for i in 0..64 {
        for j in 0..32 {
            w2.set(i, j, M31::from(((i * 32 + j) * 13 + 5) as u32 % 97));
        }
    }
    weights.add_weight(2, w2);

    // Layer 4: Linear(32→3) — node 4 (node 3 is ReLU)
    let mut w4 = M31Matrix::new(32, 3);
    for i in 0..32 {
        for j in 0..3 {
            w4.set(i, j, M31::from(((i * 3 + j) * 11 + 7) as u32 % 61));
        }
    }
    weights.add_weight(4, w4);

    ClassifierModel { graph, weights }
}

/// Create a classifier model with trained weights.
///
/// These weights were trained on 75K labeled transactions (10 real exploit
/// patterns + 8 safe DeFi patterns) and quantized to M31. See
/// `training/train.py` and `training/data_sources.py` for the pipeline.
///
/// The weights are embedded as const arrays — no file I/O at runtime.
pub fn build_trained_classifier() -> ClassifierModel {
    use super::trained_weights::*;

    let graph = build_classifier_graph();
    let weights = load_weights_from_arrays(
        &LAYER0_WEIGHTS,
        &LAYER2_WEIGHTS,
        &LAYER4_WEIGHTS,
    )
    .expect("trained weight dimensions must match architecture");

    ClassifierModel { graph, weights }
}

/// Load classifier weights from raw M31 arrays.
///
/// Each array contains row-major weight values. The arrays must have
/// exactly the right number of elements for each layer:
/// - `layer0`: 64 * 64 = 4096 values
/// - `layer2`: 64 * 32 = 2048 values
/// - `layer4`: 32 * 3 = 96 values
pub fn load_weights_from_arrays(
    layer0: &[u32],
    layer2: &[u32],
    layer4: &[u32],
) -> Result<GraphWeights, ClassifierError> {
    if layer0.len() != 64 * 64 {
        return Err(ClassifierError::ModelError(format!(
            "layer0 has {} values, expected {}",
            layer0.len(),
            64 * 64
        )));
    }
    if layer2.len() != 64 * 32 {
        return Err(ClassifierError::ModelError(format!(
            "layer2 has {} values, expected {}",
            layer2.len(),
            64 * 32
        )));
    }
    if layer4.len() != 32 * 3 {
        return Err(ClassifierError::ModelError(format!(
            "layer4 has {} values, expected {}",
            layer4.len(),
            32 * 3
        )));
    }

    let mut weights = GraphWeights::new();

    let mut w0 = M31Matrix::new(64, 64);
    for (i, &v) in layer0.iter().enumerate() {
        w0.data[i] = M31::from(v & 0x7FFF_FFFF);
    }
    weights.add_weight(0, w0);

    let mut w2 = M31Matrix::new(64, 32);
    for (i, &v) in layer2.iter().enumerate() {
        w2.data[i] = M31::from(v & 0x7FFF_FFFF);
    }
    weights.add_weight(2, w2);

    let mut w4 = M31Matrix::new(32, 3);
    for (i, &v) in layer4.iter().enumerate() {
        w4.data[i] = M31::from(v & 0x7FFF_FFFF);
    }
    weights.add_weight(4, w4);

    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_classifier_graph_structure() {
        let graph = build_classifier_graph();
        // 5 nodes: linear(64), relu, linear(32), relu, linear(3)
        assert_eq!(graph.nodes.len(), 5);
        assert_eq!(graph.output_shape, (1, NUM_CLASSES));
    }

    #[test]
    fn test_build_test_classifier() {
        let model = build_test_classifier();
        assert_eq!(model.graph.nodes.len(), 5);
        assert!(model.weights.get_weight(0).is_some(), "layer 0 weight should exist");
        assert!(model.weights.get_weight(2).is_some(), "layer 2 weight should exist");
        assert!(model.weights.get_weight(4).is_some(), "layer 4 weight should exist");

        let w0 = model.weights.get_weight(0).unwrap();
        assert_eq!(w0.rows, 64);
        assert_eq!(w0.cols, 64);

        let w4 = model.weights.get_weight(4).unwrap();
        assert_eq!(w4.rows, 32);
        assert_eq!(w4.cols, NUM_CLASSES);
    }

    #[test]
    fn test_load_weights_dimension_check() {
        let too_short = vec![0u32; 100];
        let result = load_weights_from_arrays(&too_short, &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_trained_classifier() {
        let model = build_trained_classifier();
        assert_eq!(model.graph.nodes.len(), 5);

        let w0 = model.weights.get_weight(0).unwrap();
        assert_eq!(w0.rows, 64);
        assert_eq!(w0.cols, 64);

        let w2 = model.weights.get_weight(2).unwrap();
        assert_eq!(w2.rows, 64);
        assert_eq!(w2.cols, 32);

        let w4 = model.weights.get_weight(4).unwrap();
        assert_eq!(w4.rows, 32);
        assert_eq!(w4.cols, NUM_CLASSES);

        // Verify weights are non-trivial (not all zeros)
        let sum: u64 = w0.data.iter().map(|m| m.0 as u64).sum();
        assert!(sum > 0, "trained weights should be non-zero");
    }
}
