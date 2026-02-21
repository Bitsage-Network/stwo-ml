//! `ProofEventSink` trait + `ProofSink` wrapper.
//!
//! `ProofSink` is the handle that stwo-ml stores in a thread-local and calls
//! into during proving. It wraps an optional boxed sink so that the inactive
//! (no visualization) path is a single `Option::is_some()` branch with no
//! heap allocation.

use crate::events::ProofEvent;

// ─── Trait ───────────────────────────────────────────────────────────────────

/// A consumer of proof events.
///
/// Implementations must be `Send + Sync + 'static` so they can be stored in a
/// thread-local and potentially forwarded across threads.
pub trait ProofEventSink: Send + Sync + 'static {
    /// Handle one event. Must not block the calling thread.
    fn emit(&self, event: ProofEvent);

    /// Called at the end of a proof session. Default: no-op.
    fn flush(&self) {}
}

// ─── NullSink ────────────────────────────────────────────────────────────────

/// A sink that discards every event. Used as the default when visualization is
/// disabled. The compiler should inline and eliminate all calls through this.
pub struct NullSink;

impl ProofEventSink for NullSink {
    #[inline(always)]
    fn emit(&self, _: ProofEvent) {}
}

// ─── ProofSink ───────────────────────────────────────────────────────────────

/// Wrapper around an optional `ProofEventSink`.
///
/// When `inner` is `None` (the common, non-visualization path), all hot-path
/// calls go through a single branch and generate zero allocations.
pub struct ProofSink {
    inner: Option<Box<dyn ProofEventSink>>,
}

impl ProofSink {
    /// Create an inactive sink (no visualization).
    pub fn none() -> Self {
        Self { inner: None }
    }

    /// Create an active sink wrapping `s`.
    pub fn new(s: impl ProofEventSink) -> Self {
        Self {
            inner: Some(Box::new(s)),
        }
    }

    /// Returns `true` if a sink is installed (visualization is active).
    #[inline]
    pub fn is_active(&self) -> bool {
        self.inner.is_some()
    }

    /// Emit an event. No-op when inactive.
    #[inline]
    pub fn emit(&self, event: ProofEvent) {
        if let Some(s) = &self.inner {
            s.emit(event);
        }
    }

    /// Emit the result of `f()` only when active.
    ///
    /// The closure is **never called** when no sink is installed, so constructing
    /// an event in a hot loop (e.g. sumcheck rounds) costs nothing when disabled.
    #[inline]
    pub fn emit_if(&self, f: impl FnOnce() -> ProofEvent) {
        if self.inner.is_some() {
            self.emit(f());
        }
    }

    /// Flush the inner sink. Called at the end of a proof session.
    pub fn flush(&self) {
        if let Some(s) = &self.inner {
            s.flush();
        }
    }
}

// ─── ChannelSink ─────────────────────────────────────────────────────────────

/// A sink backed by a `crossbeam` bounded channel.
///
/// Events are `try_send`'d — if the channel is full the event is silently
/// dropped rather than blocking the prover.
pub struct ChannelSink {
    tx: crossbeam::channel::Sender<ProofEvent>,
}

impl ChannelSink {
    /// Create a new `ChannelSink`/receiver pair with `capacity` slots.
    pub fn new(capacity: usize) -> (Self, crossbeam::channel::Receiver<ProofEvent>) {
        let (tx, rx) = crossbeam::channel::bounded(capacity);
        (Self { tx }, rx)
    }
}

impl ProofEventSink for ChannelSink {
    #[inline]
    fn emit(&self, event: ProofEvent) {
        // Silently drop events if the consumer is falling behind.
        let _ = self.tx.try_send(event);
    }
}

// ─── CollectingSink (useful in tests) ────────────────────────────────────────

use std::sync::{Arc, Mutex};

/// A sink that collects all emitted events into a `Vec`. Useful in integration
/// tests that need to assert on the sequence of events.
#[derive(Clone)]
pub struct CollectingSink {
    events: Arc<Mutex<Vec<ProofEvent>>>,
}

impl CollectingSink {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Drain and return all collected events.
    pub fn drain(&self) -> Vec<ProofEvent> {
        self.events.lock().unwrap().drain(..).collect()
    }

    /// Return a snapshot of collected events (clone).
    pub fn snapshot(&self) -> Vec<ProofEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl Default for CollectingSink {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofEventSink for CollectingSink {
    fn emit(&self, event: ProofEvent) {
        self.events.lock().unwrap().push(event);
    }
}
