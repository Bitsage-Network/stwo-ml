//! Integration test: verify proof-stream events are emitted during proving.
//!
//! Uses a `CollectingSink` to record events emitted during the test,
//! verifying the sink install/teardown contract and manual emission via
//! the thread-local.

#![cfg(feature = "proof-stream")]

use proof_stream::sink::CollectingSink;
use proof_stream::{ProofEvent, ProofSink};
use stwo_ml::gkr::prover::{set_proof_sink, PROOF_SINK};

#[test]
fn test_sink_install_teardown() {
    let collector = CollectingSink::new();
    let sink = ProofSink::new(collector.clone());

    // No sink installed yet — should be None
    PROOF_SINK.with(|s| assert!(s.borrow().is_none()));

    let _guard = set_proof_sink(sink);

    // Sink now installed
    PROOF_SINK.with(|s| assert!(s.borrow().is_some()));

    drop(_guard);

    // Sink removed after drop
    PROOF_SINK.with(|s| assert!(s.borrow().is_none()));
}

#[test]
fn test_manual_emit_via_thread_local() {
    use proof_stream::events::LogLevel;

    let collector = CollectingSink::new();
    let sink = ProofSink::new(collector.clone());
    let _guard = set_proof_sink(sink);

    // Emit via thread-local (same path as emit_proof_event! macro)
    PROOF_SINK.with(|s| {
        if let Some(sink) = s.borrow().as_ref() {
            sink.emit(ProofEvent::ProofStart {
                model_name: Some("test-model".into()),
                backend: "cpu".into(),
                num_layers: 2,
                input_shape: (1, 4),
                output_shape: (1, 4),
            });
            sink.emit(ProofEvent::Log {
                level: LogLevel::Info,
                message: "layer 0 started".into(),
            });
            sink.emit(ProofEvent::ProofComplete {
                duration_ms: 100,
                num_layer_proofs: 1,
                num_weight_openings: 1,
                weight_binding_mode: "sequential".into(),
            });
        }
    });

    let events = collector.snapshot();
    assert_eq!(events.len(), 3);

    assert!(matches!(events[0], ProofEvent::ProofStart { .. }));
    assert!(matches!(events[1], ProofEvent::Log { .. }));
    assert!(matches!(events[2], ProofEvent::ProofComplete { .. }));
}

#[test]
fn test_collect_from_multiple_threads() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    // Each thread has its own independent thread-local, so we test independence.
    let barrier = Arc::new(Barrier::new(2));
    let barrier2 = Arc::clone(&barrier);

    let handle = thread::spawn(move || {
        let collector = CollectingSink::new();
        let sink = ProofSink::new(collector.clone());
        let _guard = set_proof_sink(sink);

        barrier2.wait();

        PROOF_SINK.with(|s| {
            assert!(
                s.borrow().is_some(),
                "thread-local should be set in spawned thread"
            );
        });

        collector.snapshot().len()
    });

    barrier.wait();

    // Main thread has no sink
    PROOF_SINK.with(|s| assert!(s.borrow().is_none()));

    let n = handle.join().unwrap();
    assert_eq!(n, 0, "spawned thread should have 0 events (no emissions)");
}
