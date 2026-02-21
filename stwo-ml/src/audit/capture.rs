//! Non-blocking capture hook for model servers.
//!
//! Sits between the model server and the inference log. When an inference
//! completes, the server calls `record()` which returns immediately.
//! A background thread computes commitments and appends to the log.
//!
//! ```text
//! Model Server ──record()──> CaptureHook ──crossbeam──> Background Thread
//!  (returns                    (non-blocking send)         │
//!   immediately)                                           ├─ compute io_commitment
//!                                                          ├─ write M31 matrices
//!                                                          └─ append to InferenceLog
//! ```

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use crate::aggregation::compute_io_commitment;
use crate::audit::log::InferenceLog;
use crate::audit::types::{AuditError, InferenceLogEntry};
use crate::components::matmul::M31Matrix;
use crossbeam::channel::{self, Receiver, Sender};

/// A job submitted to the capture hook's background thread.
pub struct CaptureJob {
    /// Tokenized input (raw token IDs).
    pub input_tokens: Vec<u32>,
    /// Tokenized output (raw token IDs).
    pub output_tokens: Vec<u32>,
    /// M31-quantized input matrix (for prover replay).
    pub input_m31: M31Matrix,
    /// M31-quantized output matrix (for prover replay).
    pub output_m31: M31Matrix,
    /// Inference timestamp (Unix epoch nanoseconds).
    pub timestamp_ns: u64,
    /// Inference latency in milliseconds.
    pub latency_ms: u64,
    /// GPU device name.
    pub gpu_device: String,
    /// TEE attestation hash (hex felt252, "0x0" if no TEE).
    pub tee_report_hash: String,
    /// Task category for report aggregation.
    pub task_category: Option<String>,
    /// First 100 chars of detokenized input.
    pub input_preview: Option<String>,
    /// First 200 chars of detokenized output.
    pub output_preview: Option<String>,
}

/// Internal message for the background thread.
enum CaptureMsg {
    Job(CaptureJob),
    Flush(Sender<()>),
    Shutdown,
}

/// Non-blocking capture hook for inference logging.
///
/// `record()` sends a job to a background thread via crossbeam channel
/// and returns immediately. The background thread computes commitments,
/// writes M31 matrices to the binary sidecar, and appends to the JSONL log.
pub struct CaptureHook {
    tx: Sender<CaptureMsg>,
    /// Counter of entries successfully written (updated by background thread).
    entry_count: Arc<AtomicU64>,
    /// Background thread handle.
    handle: Option<thread::JoinHandle<()>>,
}

impl CaptureHook {
    /// Create a new capture hook for a model session.
    ///
    /// Spawns a background thread that processes capture jobs.
    /// The `model_id` and `weight_commitment` are hex felt252 strings.
    pub fn new(
        log_dir: impl AsRef<Path>,
        model_id: &str,
        weight_commitment: &str,
        model_name: &str,
    ) -> Result<Self, AuditError> {
        let log = InferenceLog::new(log_dir, model_id, weight_commitment, model_name)?;
        Self::with_log(log)
    }

    /// Create a capture hook from an existing (possibly reloaded) log.
    pub fn with_log(log: InferenceLog) -> Result<Self, AuditError> {
        let (tx, rx) = channel::unbounded();
        let entry_count = Arc::new(AtomicU64::new(log.entry_count()));

        let count_ref = entry_count.clone();
        let handle = thread::Builder::new()
            .name("audit-capture".to_string())
            .spawn(move || {
                background_worker(log, rx, count_ref);
            })
            .map_err(|e| AuditError::LogError(format!("failed to spawn capture thread: {}", e)))?;

        Ok(Self {
            tx,
            entry_count,
            handle: Some(handle),
        })
    }

    /// Record an inference. Returns immediately (non-blocking).
    ///
    /// The background thread will compute commitments and append to the log.
    pub fn record(&self, job: CaptureJob) {
        // Unbounded send never blocks.
        let _ = self.tx.send(CaptureMsg::Job(job));
    }

    /// Flush all pending records to disk. Blocks until complete.
    pub fn flush(&self) {
        let (done_tx, done_rx) = channel::bounded(0);
        let _ = self.tx.send(CaptureMsg::Flush(done_tx));
        // Block until the background thread has processed all preceding messages.
        let _ = done_rx.recv();
    }

    /// Number of entries successfully written to the log.
    pub fn entry_count(&self) -> u64 {
        self.entry_count.load(Ordering::Relaxed)
    }
}

impl Drop for CaptureHook {
    fn drop(&mut self) {
        let _ = self.tx.send(CaptureMsg::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Background worker that processes capture jobs.
fn background_worker(mut log: InferenceLog, rx: Receiver<CaptureMsg>, entry_count: Arc<AtomicU64>) {
    for msg in rx {
        match msg {
            CaptureMsg::Job(job) => {
                if let Err(e) = process_job(&mut log, &job) {
                    tracing::error!("capture hook: failed to process job: {}", e);
                    continue;
                }
                entry_count.store(log.entry_count(), Ordering::Relaxed);
            }
            CaptureMsg::Flush(done) => {
                // All preceding messages have been processed (channel is FIFO).
                let _ = done.send(());
            }
            CaptureMsg::Shutdown => {
                break;
            }
        }
    }
}

/// Process a single capture job: compute commitments, write matrices, append entry.
fn process_job(log: &mut InferenceLog, job: &CaptureJob) -> Result<(), AuditError> {
    // Compute IO commitment from M31 matrices.
    let io_commitment = compute_io_commitment(&job.input_m31, &job.output_m31);

    // Write input matrix to binary sidecar.
    let input_data: Vec<u32> = job.input_m31.data.iter().map(|m| m.0).collect();
    let (matrix_offset, input_size) = log.write_matrix(
        job.input_m31.rows as u32,
        job.input_m31.cols as u32,
        &input_data,
    )?;

    // Write output matrix to binary sidecar.
    let output_data: Vec<u32> = job.output_m31.data.iter().map(|m| m.0).collect();
    let (_out_offset, output_size) = log.write_matrix(
        job.output_m31.rows as u32,
        job.output_m31.cols as u32,
        &output_data,
    )?;

    // matrix_size stores only the input matrix size — the prover reads back the
    // input from (matrix_offset, matrix_size) and re-executes the forward pass.
    // The output matrix is stored adjacently in the sidecar for archival purposes.
    let _ = output_size;

    // Build the log entry.
    let entry = InferenceLogEntry {
        inference_id: log.entry_count(),
        sequence_number: 0, // Assigned by log.append().
        model_id: log.model_id().to_string(),
        weight_commitment: log.weight_commitment().to_string(),
        model_name: String::new(), // Not needed per-entry (in session meta).
        num_layers: 0,             // Set by server if known.
        input_tokens: job.input_tokens.clone(),
        output_tokens: job.output_tokens.clone(),
        matrix_offset,
        matrix_size: input_size,
        input_rows: job.input_m31.rows as u32,
        input_cols: job.input_m31.cols as u32,
        output_rows: job.output_m31.rows as u32,
        output_cols: job.output_m31.cols as u32,
        io_commitment: format!("{:#066x}", io_commitment),
        layer_chain_commitment: "0x0".to_string(), // Set during replay/proving.
        prev_entry_hash: String::new(),            // Set by log.append().
        entry_hash: String::new(),                 // Set by log.append().
        timestamp_ns: job.timestamp_ns,
        latency_ms: job.latency_ms,
        gpu_device: job.gpu_device.clone(),
        tee_report_hash: job.tee_report_hash.clone(),
        task_category: job.task_category.clone(),
        input_preview: job.input_preview.clone(),
        output_preview: job.output_preview.clone(),
    };

    log.append(entry)?;
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use starknet_ff::FieldElement;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};
    use stwo::core::fields::m31::M31;

    fn temp_dir() -> std::path::PathBuf {
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("stwo_ml_capture_{}", d))
    }

    fn make_m31_matrix(rows: usize, cols: usize, base: u32) -> M31Matrix {
        let data: Vec<M31> = (0..(rows * cols) as u32)
            .map(|i| M31::from(base + i))
            .collect();
        M31Matrix { rows, cols, data }
    }

    fn make_job(idx: u32) -> CaptureJob {
        CaptureJob {
            input_tokens: vec![1, 2, 3],
            output_tokens: vec![4, 5],
            input_m31: make_m31_matrix(1, 3, idx * 100),
            output_m31: make_m31_matrix(1, 2, idx * 100 + 50),
            timestamp_ns: 1_000_000_000_000 + idx as u64 * 1_000_000,
            latency_ms: 100,
            gpu_device: "test-gpu".to_string(),
            tee_report_hash: "0x0".to_string(),
            task_category: Some("test".to_string()),
            input_preview: Some("test input".to_string()),
            output_preview: Some("test output".to_string()),
        }
    }

    #[test]
    fn test_record_1000_entries() {
        let dir = temp_dir();
        let hook = CaptureHook::new(&dir, "0x2", "0xabc", "test-model").expect("create hook");

        for i in 0..1000 {
            hook.record(make_job(i));
        }

        hook.flush();
        assert_eq!(hook.entry_count(), 1000);

        // Reload and verify chain.
        drop(hook);
        let log = InferenceLog::load(&dir).expect("load");
        assert_eq!(log.entry_count(), 1000);
        log.verify_chain().expect("chain valid");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_record_is_nonblocking() {
        let dir = temp_dir();
        let hook = CaptureHook::new(&dir, "0x2", "0xabc", "test-model").expect("create hook");

        let start = Instant::now();
        for i in 0..100 {
            hook.record(make_job(i));
        }
        let elapsed = start.elapsed();

        // 100 records should complete in well under 10ms (just channel sends).
        assert!(
            elapsed.as_millis() < 10,
            "record should be non-blocking, took {}ms",
            elapsed.as_millis()
        );

        hook.flush();
        assert_eq!(hook.entry_count(), 100);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_commitments_match_manual() {
        let dir = temp_dir();
        let hook = CaptureHook::new(&dir, "0x2", "0xabc", "test-model").expect("create hook");

        let input = make_m31_matrix(1, 3, 10);
        let output = make_m31_matrix(1, 2, 50);

        // Compute expected commitment manually.
        let expected = compute_io_commitment(&input, &output);

        hook.record(CaptureJob {
            input_tokens: vec![1, 2, 3],
            output_tokens: vec![4, 5],
            input_m31: input,
            output_m31: output,
            timestamp_ns: 1_000_000_000_000,
            latency_ms: 100,
            gpu_device: "test-gpu".to_string(),
            tee_report_hash: "0x0".to_string(),
            task_category: None,
            input_preview: None,
            output_preview: None,
        });

        hook.flush();

        // Load the log and check the commitment.
        drop(hook);
        let log = InferenceLog::load(&dir).expect("load");
        let entry = &log.entries()[0];
        let stored = FieldElement::from_hex_be(&entry.io_commitment).unwrap();
        assert_eq!(
            stored, expected,
            "io_commitment should match manual computation"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
