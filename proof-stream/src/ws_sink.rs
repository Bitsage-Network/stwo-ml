//! `WsBroadcastSink` — fan-out proof events to WebSocket subscribers.
//!
//! Enabled by the `ws-server` feature. The sink wraps a
//! `tokio::sync::broadcast::Sender<String>` so that every emitted `ProofEvent`
//! is serialised to JSON and broadcast to all connected WebSocket clients.
//!
//! `broadcast::Sender::send` is synchronous and O(subscribers) — it never
//! parks the calling thread, so the prover hot-path is unaffected.

#[cfg(feature = "ws-server")]
pub use inner::WsBroadcastSink;

#[cfg(feature = "ws-server")]
mod inner {
    use crate::{ProofEvent, ProofEventSink};

    /// A `ProofEventSink` that broadcasts serialised JSON to WebSocket subscribers.
    ///
    /// Cloning is cheap (wraps `Arc` internally via `broadcast::Sender`).
    #[derive(Clone)]
    pub struct WsBroadcastSink {
        tx: tokio::sync::broadcast::Sender<String>,
    }

    impl WsBroadcastSink {
        /// Create a new sink with `capacity` slots in the broadcast channel.
        ///
        /// Lagged subscribers (those that fall behind by more than `capacity`
        /// events) will receive a `RecvError::Lagged` on their next recv, which
        /// the WebSocket handler converts to a `{"Log":{"level":"Warn",...}}` frame.
        pub fn new(capacity: usize) -> Self {
            let (tx, _) = tokio::sync::broadcast::channel(capacity);
            Self { tx }
        }

        /// Subscribe to the broadcast stream. Each subscriber gets its own
        /// independent receiver; slow subscribers do not stall the prover.
        pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<String> {
            self.tx.subscribe()
        }
    }

    impl ProofEventSink for WsBroadcastSink {
        #[inline]
        fn emit(&self, event: ProofEvent) {
            if let Ok(json) = serde_json::to_string(&event) {
                // Ignore SendError (no active subscribers) — fire and forget.
                let _ = self.tx.send(json);
            }
        }

        fn flush(&self) {}
    }
}
