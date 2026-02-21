//! Helpers for building `ProofEvent::GpuStatus` from device info.

use crate::events::{GpuSnapshot, ProofEvent};

/// Build a `GpuStatus` event from a slice of device snapshots and progress
/// counters.
pub fn gpu_status_event(
    devices: Vec<GpuSnapshot>,
    matmul_done: usize,
    matmul_total: usize,
    layers_done: usize,
    layers_total: usize,
) -> ProofEvent {
    ProofEvent::GpuStatus {
        devices,
        matmul_done,
        matmul_total,
        layers_done,
        layers_total,
    }
}

/// Create a `GpuSnapshot` with unknown utilization (useful when NVML is not
/// available).
pub fn unknown_gpu_snapshot(device_id: usize, name: impl Into<String>) -> GpuSnapshot {
    GpuSnapshot {
        device_id,
        device_name: name.into(),
        utilization: 0.0,
        free_memory_bytes: None,
    }
}
