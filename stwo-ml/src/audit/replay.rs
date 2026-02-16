//! Replay verification for audit proving.
//!
//! Before proving a logged inference, replay verification re-executes
//! the forward pass and checks that the `io_commitment` matches.
//! This catches log corruption or data integrity issues before
//! spending GPU time on proving.

use std::collections::HashMap;

use starknet_ff::FieldElement;

use crate::aggregation::compute_io_commitment;
use crate::audit::log::InferenceLog;
use crate::audit::types::{AuditError, InferenceLogEntry};
use crate::compiler::graph::{ComputationGraph, GraphOp, GraphWeights};
use crate::compiler::prove::{
    apply_activation_pub, apply_layernorm_pub, apply_rmsnorm_detailed,
    elementwise_add, elementwise_mul,
};
use crate::components::matmul::{matmul_m31, M31Matrix};

use stwo::core::fields::m31::M31;

/// Result of replaying a single inference.
#[derive(Debug)]
pub struct ReplayResult {
    /// Sequence number from the log.
    pub sequence: u64,
    /// Whether the replay matched the logged io_commitment.
    pub matched: bool,
    /// The io_commitment computed from the replayed forward pass.
    pub replayed_commitment: FieldElement,
    /// The io_commitment from the log.
    pub logged_commitment: FieldElement,
}

/// Execute a forward pass through a computation graph without proving.
///
/// Handles the core ops (MatMul, Activation, LayerNorm, RMSNorm, Add, Mul,
/// Attention). Quantize/Dequantize/Embedding/Conv2D/Identity/RoPE are treated
/// as passthrough, matching how `prove_model_with` handles them.
pub fn execute_forward_pass(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,
) -> Result<M31Matrix, AuditError> {
    let mut node_outputs: HashMap<usize, M31Matrix> = HashMap::new();
    let mut current = input.clone();

    let topo = graph.topological_order();
    for &node_id in &topo {
        let node = &graph.nodes[node_id];

        // Resolve input from predecessor.
        if let Some(&first_input) = node.inputs.first() {
            if let Some(inp) = node_outputs.get(&first_input) {
                current = inp.clone();
            }
        }

        let output = match &node.op {
            GraphOp::MatMul { .. } => {
                let weight = weights.get_weight(node.id).ok_or_else(|| {
                    AuditError::ProvingFailed(format!("missing weight for node {}", node.id))
                })?;
                matmul_m31(&current, weight)
            }
            GraphOp::Activation { activation_type, .. } => {
                let f = activation_type.as_fn();
                apply_activation_pub(&current, &*f)
            }
            GraphOp::LayerNorm { dim } => apply_layernorm_pub(&current, *dim),
            GraphOp::RMSNorm { dim } => apply_rmsnorm_detailed(&current, *dim).output_matrix,
            GraphOp::Add { .. } => {
                let lhs_id = node.inputs.get(0).copied().unwrap_or(0);
                let rhs_id = node.inputs.get(1).copied().unwrap_or(0);
                let lhs = node_outputs.get(&lhs_id).unwrap_or(&current);
                let rhs = node_outputs.get(&rhs_id).unwrap_or(&current);
                elementwise_add(lhs, rhs)
            }
            GraphOp::Mul { .. } => {
                let lhs_id = node.inputs.get(0).copied().unwrap_or(0);
                let rhs_id = node.inputs.get(1).copied().unwrap_or(0);
                let lhs = node_outputs.get(&lhs_id).unwrap_or(&current);
                let rhs = node_outputs.get(&rhs_id).unwrap_or(&current);
                elementwise_mul(lhs, rhs)
            }
            GraphOp::Attention { config } => {
                // Extract attention weights using named weight convention.
                let w_q = weights.get_named_weight(node.id, "w_q");
                let w_k = weights.get_named_weight(node.id, "w_k");
                let w_v = weights.get_named_weight(node.id, "w_v");
                let w_o = weights.get_named_weight(node.id, "w_o");

                if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (w_q, w_k, w_v, w_o) {
                    let attn_weights = crate::components::attention::AttentionWeights {
                        w_q: wq.clone(),
                        w_k: wk.clone(),
                        w_v: wv.clone(),
                        w_o: wo.clone(),
                    };
                    let intermediates = crate::components::attention::attention_forward(
                        &current,
                        &attn_weights,
                        config,
                        false,
                    );
                    intermediates.final_output
                } else {
                    // Missing attention weights — passthrough.
                    current.clone()
                }
            }
            // Passthrough ops (matching prove_model_with behavior).
            GraphOp::Quantize { .. }
            | GraphOp::Dequantize { .. }
            | GraphOp::Embedding { .. }
            | GraphOp::Conv2D { .. }
            | GraphOp::Identity { .. }
            | GraphOp::RoPE { .. } => current.clone(),
        };

        node_outputs.insert(node.id, output.clone());
        current = output;
    }

    Ok(current)
}

/// Verify a single logged inference by replaying the forward pass.
///
/// Loads M31 input from the binary sidecar, runs the full forward pass,
/// and compares `compute_io_commitment(input, output)` with the logged value.
pub fn verify_replay(
    graph: &ComputationGraph,
    weights: &GraphWeights,
    entry: &InferenceLogEntry,
    log: &InferenceLog,
) -> Result<ReplayResult, AuditError> {
    // Load M31 input matrix from sidecar.
    let (rows, cols, data) = log.read_matrix(entry.matrix_offset, entry.matrix_size)?;
    let input = M31Matrix {
        rows: rows as usize,
        cols: cols as usize,
        data: data.iter().map(|&v| M31::from(v)).collect(),
    };

    // Execute forward pass (no proving).
    let output = execute_forward_pass(graph, &input, weights)?;

    // Compute io_commitment from replayed data.
    let replayed_commitment = compute_io_commitment(&input, &output);
    let logged_commitment = FieldElement::from_hex_be(&entry.io_commitment).map_err(|_| {
        AuditError::LogError(format!("invalid io_commitment hex: {}", entry.io_commitment))
    })?;

    let matched = replayed_commitment == logged_commitment;

    if !matched {
        return Err(AuditError::ReplayMismatch {
            sequence: entry.sequence_number,
            expected: entry.io_commitment.clone(),
            actual: format!("{:#066x}", replayed_commitment),
        });
    }

    Ok(ReplayResult {
        sequence: entry.sequence_number,
        matched,
        replayed_commitment,
        logged_commitment,
    })
}

/// Batch replay verification for multiple entries.
///
/// Continues past individual failures, collecting all results.
pub fn verify_replay_batch(
    graph: &ComputationGraph,
    weights: &GraphWeights,
    entries: &[InferenceLogEntry],
    log: &InferenceLog,
) -> Vec<Result<ReplayResult, AuditError>> {
    entries
        .iter()
        .map(|entry| verify_replay(graph, weights, entry, log))
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::log::InferenceLog;
    use crate::audit::types::InferenceLogEntry;
    use crate::compiler::graph::{GraphBuilder, GraphWeights};
    use crate::components::activation::ActivationType;
    use stwo::core::fields::m31::M31;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> std::path::PathBuf {
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("stwo_ml_replay_{}", d))
    }

    fn make_entry_for_replay(
        log: &mut InferenceLog,
        input: &M31Matrix,
        io_commitment: FieldElement,
    ) -> InferenceLogEntry {
        let input_data: Vec<u32> = input.data.iter().map(|m| m.0).collect();
        let (mat_off, mat_sz) = log
            .write_matrix(input.rows as u32, input.cols as u32, &input_data)
            .unwrap();

        InferenceLogEntry {
            inference_id: 0,
            sequence_number: 0,
            model_id: "0x2".to_string(),
            weight_commitment: "0xabc".to_string(),
            model_name: "test".to_string(),
            num_layers: 2,
            input_tokens: vec![1, 2, 3, 4],
            output_tokens: vec![],
            matrix_offset: mat_off,
            matrix_size: mat_sz,
            input_rows: input.rows as u32,
            input_cols: input.cols as u32,
            output_rows: 0,
            output_cols: 0,
            io_commitment: format!("{:#066x}", io_commitment),
            layer_chain_commitment: "0x0".to_string(),
            prev_entry_hash: String::new(),
            entry_hash: String::new(),
            timestamp_ns: 1_000_000_000_000,
            latency_ms: 100,
            gpu_device: "test".to_string(),
            tee_report_hash: "0x0".to_string(),
            task_category: None,
            input_preview: None,
            output_preview: None,
        }
    }

    #[test]
    fn test_replay_valid() {
        let dir = temp_dir();

        // Build: linear(4->4) -> relu -> linear(4->2)
        // Node 0 = matmul, node 1 = relu, node 2 = matmul
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU)
            .linear(2);
        let graph = builder.build();

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i * 4 + j) + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        let mut w2 = M31Matrix::new(4, 2);
        for i in 0..4 {
            for j in 0..2 {
                w2.set(i, j, M31::from(((i * 2 + j) + 1) as u32));
            }
        }
        weights.add_weight(2, w2);

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        // Get correct output and commitment.
        let output = execute_forward_pass(&graph, &input, &weights).unwrap();
        let io_commitment = compute_io_commitment(&input, &output);

        let mut log = InferenceLog::new(&dir, "0x2", "0xabc", "test-model").unwrap();
        let entry = make_entry_for_replay(&mut log, &input, io_commitment);
        log.append(entry).unwrap();

        let result = verify_replay(&graph, &weights, &log.entries()[0], &log).unwrap();
        assert!(result.matched);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_replay_tampered_commitment_fails() {
        let dir = temp_dir();

        // Build: linear(4->4) -> relu
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU);
        let graph = builder.build();

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i * 4 + j) + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 1) as u32));
        }

        let mut log = InferenceLog::new(&dir, "0x2", "0xabc", "test-model").unwrap();
        let wrong_commitment = FieldElement::from(0xDEADBEEFu64);
        let entry = make_entry_for_replay(&mut log, &input, wrong_commitment);
        log.append(entry).unwrap();

        let result = verify_replay(&graph, &weights, &log.entries()[0], &log);
        assert!(result.is_err());
        match result.unwrap_err() {
            AuditError::ReplayMismatch { sequence, .. } => assert_eq!(sequence, 0),
            other => panic!("expected ReplayMismatch, got: {:?}", other),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_forward_pass_deterministic() {
        let mut builder = GraphBuilder::new((1, 4));
        builder
            .linear(4)
            .activation(ActivationType::ReLU);
        let graph = builder.build();

        let mut weights = GraphWeights::new();
        let mut w0 = M31Matrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                w0.set(i, j, M31::from(((i * 4 + j) + 1) as u32));
            }
        }
        weights.add_weight(0, w0);

        let mut input = M31Matrix::new(1, 4);
        for j in 0..4 {
            input.set(0, j, M31::from((j + 5) as u32));
        }

        let out1 = execute_forward_pass(&graph, &input, &weights).unwrap();
        let out2 = execute_forward_pass(&graph, &input, &weights).unwrap();
        assert_eq!(out1.data, out2.data);
        assert_eq!(out1.rows, out2.rows);
        assert_eq!(out1.cols, out2.cols);
    }
}
