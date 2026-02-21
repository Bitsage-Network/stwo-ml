//! Zero-overhead tests for `ProofSink` / `NullSink`.

use proof_stream::events::{LogLevel, ProofEvent};
use proof_stream::sink::ProofSink;

fn make_log_event() -> ProofEvent {
    ProofEvent::Log {
        level: LogLevel::Info,
        message: "bench".into(),
    }
}

/// `NullSink::emit` must be a no-op — calling it 1 000 000 times at opt-level=2
/// should complete well under 100ms.
#[test]
fn test_null_sink_zero_overhead() {
    let sink = ProofSink::none();
    let iters = 1_000_000usize;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        // Closure must never be called when sink is inactive
        sink.emit_if(make_log_event);
    }
    let elapsed = start.elapsed();
    // 1ms generous threshold — real time should be ~0µs (branch eliminated)
    assert!(
        elapsed.as_millis() < 100,
        "NullSink took {}ms for {iters} iterations",
        elapsed.as_millis()
    );
}

/// `CollectingSink` records every event.
#[test]
fn test_collecting_sink() {
    use proof_stream::sink::CollectingSink;

    let collector = CollectingSink::new();
    let sink = ProofSink::new(collector.clone());

    assert!(sink.is_active());

    sink.emit(ProofEvent::Log {
        level: LogLevel::Info,
        message: "first".into(),
    });
    sink.emit(ProofEvent::Log {
        level: LogLevel::Warn,
        message: "second".into(),
    });

    let events = collector.snapshot();
    assert_eq!(events.len(), 2);

    // Verify discriminants
    assert!(matches!(events[0], ProofEvent::Log { .. }));
    assert!(matches!(events[1], ProofEvent::Log { .. }));
}

/// `emit_if` closure is never called on an inactive sink.
#[test]
fn test_emit_if_not_called_when_inactive() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let sink = ProofSink::none();
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = Arc::clone(&call_count);

    for _ in 0..10 {
        sink.emit_if(|| {
            cc.fetch_add(1, Ordering::Relaxed);
            make_log_event()
        });
    }

    assert_eq!(call_count.load(Ordering::Relaxed), 0);
}

/// `ChannelSink` delivers events to the receiver.
#[test]
fn test_channel_sink() {
    use proof_stream::sink::ChannelSink;

    let (chan_sink, rx) = ChannelSink::new(64);
    let sink = ProofSink::new(chan_sink);

    sink.emit(ProofEvent::StarkProofStart {
        num_activation_layers: 40,
        num_add_layers: 40,
        num_layernorm_layers: 40,
    });
    sink.emit(ProofEvent::StarkProofEnd { duration_ms: 320 });

    let e1 = rx.recv().unwrap();
    let e2 = rx.recv().unwrap();
    assert!(matches!(e1, ProofEvent::StarkProofStart { .. }));
    assert!(matches!(e2, ProofEvent::StarkProofEnd { .. }));
}
