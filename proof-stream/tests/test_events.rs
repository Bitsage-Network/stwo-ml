//! Serde round-trip tests for every `ProofEvent` variant.

use proof_stream::events::*;
use proof_stream::ProofEvent;

fn round_trip(event: ProofEvent) -> ProofEvent {
    let json = serde_json::to_string(&event).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[test]
fn test_circuit_compiled_round_trip() {
    let event = ProofEvent::CircuitCompiled {
        total_layers: 5,
        input_shape: (1, 512),
        output_shape: (1, 32000),
        nodes: vec![CircuitNodeMeta {
            layer_idx: 0,
            node_id: 1,
            kind: LayerKind::MatMul,
            input_shape: (1, 512),
            output_shape: (1, 2048),
            trace_cost: 1048576,
            input_layers: vec![],
        }],
        has_simd: false,
        simd_num_blocks: 0,
    };
    let rt = round_trip(event.clone());
    let json1 = serde_json::to_string(&event).unwrap();
    let json2 = serde_json::to_string(&rt).unwrap();
    assert_eq!(json1, json2);
}

#[test]
fn test_layer_activation_round_trip() {
    let event = ProofEvent::LayerActivation {
        layer_idx: 2,
        node_id: 3,
        kind: LayerKind::Activation,
        output_shape: (1, 1024),
        output_sample: vec![0, 1000, 2000, 3000],
        stats: ActivationStats {
            mean: 0.5,
            std_dev: 0.1,
            min: 0.0,
            max: 1.0,
            sparsity: 0.2,
        },
    };
    round_trip(event);
}

#[test]
fn test_attention_heatmap_round_trip() {
    let event = ProofEvent::AttentionHeatmap {
        layer_idx: 1,
        head_idx: 0,
        num_heads: 32,
        seq_len: 4,
        scores: vec![1.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
    };
    round_trip(event);
}

#[test]
fn test_proof_start_round_trip() {
    let event = ProofEvent::ProofStart {
        model_name: Some("qwen3-14b".into()),
        backend: "gpu".into(),
        num_layers: 40,
        input_shape: (1, 512),
        output_shape: (1, 32000),
    };
    round_trip(event);
}

#[test]
fn test_layer_start_round_trip() {
    let event = ProofEvent::LayerStart {
        layer_idx: 3,
        kind: LayerKind::MatMul,
        input_shape: (1, 5120),
        output_shape: (1, 17408),
        trace_cost: 89128960,
        claim_value_approx: 0.42,
        gpu_device: Some(0),
    };
    round_trip(event);
}

#[test]
fn test_sumcheck_round_round_trip() {
    let sf = SecureFieldMirror { a: 1234567, b: 0, c: 0, d: 0 };
    let event = ProofEvent::SumcheckRound {
        layer_idx: 0,
        round: 5,
        total_rounds: 24,
        poly_deg2: Some(RoundPolyViz { c0: sf, c1: sf, c2: sf }),
        poly_deg3: None,
        claim_value_approx: 0.00001,
    };
    round_trip(event);
}

#[test]
fn test_layer_end_round_trip() {
    let event = ProofEvent::LayerEnd {
        layer_idx: 2,
        kind: LayerProofKind::Sumcheck,
        final_claim_value_approx: 0.0,
        duration_ms: 310,
        rounds_completed: 24,
    };
    round_trip(event);
}

#[test]
fn test_proof_complete_round_trip() {
    let event = ProofEvent::ProofComplete {
        duration_ms: 3040,
        num_layer_proofs: 40,
        num_weight_openings: 80,
        weight_binding_mode: "aggregated".into(),
    };
    round_trip(event);
}

#[test]
fn test_gpu_status_round_trip() {
    let event = ProofEvent::GpuStatus {
        devices: vec![GpuSnapshot {
            device_id: 0,
            device_name: "NVIDIA H100".into(),
            utilization: 0.87,
            free_memory_bytes: Some(40_000_000_000),
        }],
        matmul_done: 10,
        matmul_total: 80,
        layers_done: 5,
        layers_total: 40,
    };
    round_trip(event);
}

#[test]
fn test_stark_proof_events_round_trip() {
    round_trip(ProofEvent::StarkProofStart {
        num_activation_layers: 40,
        num_add_layers: 40,
        num_layernorm_layers: 40,
    });
    round_trip(ProofEvent::StarkProofEnd { duration_ms: 320 });
}

#[test]
fn test_log_round_trip() {
    round_trip(ProofEvent::Log {
        level: LogLevel::Info,
        message: "hello proof-stream".into(),
    });
}

#[test]
fn test_weight_opening_round_trip() {
    round_trip(ProofEvent::WeightOpeningStart { weight_node_id: 7, eval_point_len: 24 });
    round_trip(ProofEvent::WeightOpeningEnd {
        weight_node_id: 7,
        duration_ms: 50,
        commitment_hex: "deadbeef01234567".into(),
    });
}

#[test]
fn test_aggregated_binding_round_trip() {
    round_trip(ProofEvent::AggregatedBindingStart { num_claims: 80, num_matrices: 80 });
    round_trip(ProofEvent::AggregatedBindingEnd {
        duration_ms: 1200,
        estimated_calldata_felts: 3000,
    });
}

#[test]
fn test_layer_kind_colors() {
    // Every variant should have a non-zero color
    for kind in [
        LayerKind::MatMul, LayerKind::Activation, LayerKind::LayerNorm,
        LayerKind::RMSNorm, LayerKind::Add, LayerKind::Mul,
        LayerKind::Attention, LayerKind::Embedding, LayerKind::Input,
    ] {
        let [r, g, b] = kind.color_rgb();
        assert!(r > 0 || g > 0 || b > 0, "kind {kind:?} has zero color");
    }
}
