//! Append-only inference log with M31-native Poseidon2 Merkle commitment.
//!
//! The inference log captures every inference during model serving.
//! Entries are chain-linked via Poseidon2-M31 hashes and indexed by a
//! Merkle tree for audit-time batch proofs.
//!
//! # Storage Layout
//!
//! ```text
//! ~/.obelysk/logs/session_{timestamp}/
//!   meta.json         <- session metadata
//!   log.jsonl         <- InferenceLogEntry per line
//!   matrices.bin      <- binary M31 matrices (input/output per entry)
//! ```

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read as IoRead, Seek, SeekFrom, Write as IoWrite};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::audit::digest::{
    digest_to_hex, hash_felt_hex_m31, parse_digest_or_zero, u64_to_m31, M31Digest, ZERO_DIGEST,
};
use crate::audit::types::{AuditError, InferenceLogEntry, LogWindow};
use crate::crypto::poseidon2_m31::{poseidon2_compress, poseidon2_hash};

// ─── Entry Hash ─────────────────────────────────────────────────────────────

/// Compute the Poseidon2-M31 hash of an inference log entry.
///
/// Hash covers identity, commitments, chain link, and timing fields.
/// This is the canonical hash used for chain linking and Merkle leaves.
///
/// Input layout (~43 M31 elements):
///   inference_id (2), sequence_number (2), model_id (8 via hash),
///   weight_commitment (8), io_commitment (8), layer_chain_commitment (8),
///   prev_entry_hash (8), timestamp_ns (2), latency_ms (2),
///   tee_report_hash (8)
pub fn compute_entry_hash(entry: &InferenceLogEntry) -> M31Digest {
    let mut input = Vec::with_capacity(56);

    // Identity fields (u64 → 2 M31 limbs each)
    input.extend_from_slice(&u64_to_m31(entry.inference_id));
    input.extend_from_slice(&u64_to_m31(entry.sequence_number));

    // Model fields (hex strings → M31 digest via hash)
    input.extend_from_slice(&hash_felt_hex_m31(&entry.model_id));

    // Commitment fields (hex strings → M31 digest via hash)
    input.extend_from_slice(&hash_felt_hex_m31(&entry.weight_commitment));
    input.extend_from_slice(&hash_felt_hex_m31(&entry.io_commitment));
    input.extend_from_slice(&hash_felt_hex_m31(&entry.layer_chain_commitment));

    // Chain link (hex string → M31 digest via hash)
    input.extend_from_slice(&hash_felt_hex_m31(&entry.prev_entry_hash));

    // Timing fields (u64 → 2 M31 limbs each)
    input.extend_from_slice(&u64_to_m31(entry.timestamp_ns));
    input.extend_from_slice(&u64_to_m31(entry.latency_ms));

    // TEE report hash
    input.extend_from_slice(&hash_felt_hex_m31(&entry.tee_report_hash));

    poseidon2_hash(&input)
}

// ─── Audit Merkle Tree (M31-native) ─────────────────────────────────────────

/// Append-only Poseidon2-M31 Merkle tree for audit logs.
///
/// Leaves are M31 digests (8 M31 elements). Internal nodes use
/// `poseidon2_compress`. Leaves are appended in O(1). Root and proofs
/// are built on demand from the full leaf set (O(N)).
pub struct AuditMerkleTree {
    leaves: Vec<M31Digest>,
}

impl AuditMerkleTree {
    pub fn new() -> Self {
        Self { leaves: Vec::new() }
    }

    /// Append a leaf digest.
    pub fn push(&mut self, leaf: M31Digest) {
        self.leaves.push(leaf);
    }

    /// Compute the current Merkle root (rebuilds from leaves, padded to power of 2).
    pub fn root(&self) -> M31Digest {
        build_merkle_root_m31(&self.leaves)
    }

    /// Number of leaves in the tree.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Get all leaf digests.
    pub fn leaves(&self) -> &[M31Digest] {
        &self.leaves
    }

    /// Compute root covering only the first `count` leaves.
    pub fn root_at(&self, count: usize) -> M31Digest {
        if count == 0 {
            return ZERO_DIGEST;
        }
        let count = count.min(self.leaves.len());
        build_merkle_root_m31(&self.leaves[..count])
    }

    /// Generate a Merkle proof (sibling digests from leaf to root).
    pub fn proof(&self, index: usize) -> Vec<M31Digest> {
        if index >= self.leaves.len() {
            return vec![];
        }
        build_merkle_proof_m31(&self.leaves, index)
    }

    /// Verify a Merkle inclusion proof.
    pub fn verify_proof(
        leaf: M31Digest,
        index: usize,
        proof: &[M31Digest],
        root: M31Digest,
    ) -> bool {
        let mut current = leaf;
        let mut idx = index;
        for sibling in proof {
            if idx % 2 == 0 {
                current = poseidon2_compress(&current, sibling);
            } else {
                current = poseidon2_compress(sibling, &current);
            }
            idx /= 2;
        }
        current == root
    }
}

/// Build Merkle root from a slice of M31 digest leaves (pad to power of 2).
fn build_merkle_root_m31(leaves: &[M31Digest]) -> M31Digest {
    if leaves.is_empty() {
        return ZERO_DIGEST;
    }
    if leaves.len() == 1 {
        return leaves[0];
    }
    let n = leaves.len().next_power_of_two();
    let mut layer: Vec<M31Digest> = leaves.to_vec();
    layer.resize(n, ZERO_DIGEST);

    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len() / 2);
        for chunk in layer.chunks(2) {
            next.push(poseidon2_compress(&chunk[0], &chunk[1]));
        }
        layer = next;
    }
    layer[0]
}

/// Build a Merkle proof for the leaf at `index`.
fn build_merkle_proof_m31(leaves: &[M31Digest], index: usize) -> Vec<M31Digest> {
    if leaves.is_empty() || index >= leaves.len() {
        return vec![];
    }
    let n = leaves.len().next_power_of_two();
    let mut layer: Vec<M31Digest> = leaves.to_vec();
    layer.resize(n, ZERO_DIGEST);

    let mut proof = Vec::new();
    let mut idx = index;

    while layer.len() > 1 {
        let sibling_idx = idx ^ 1;
        proof.push(layer[sibling_idx]);

        let mut next = Vec::with_capacity(layer.len() / 2);
        for chunk in layer.chunks(2) {
            next.push(poseidon2_compress(&chunk[0], &chunk[1]));
        }
        layer = next;
        idx /= 2;
    }
    proof
}

// ─── Session Metadata ───────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct SessionMeta {
    model_id: String,
    weight_commitment: String,
    model_name: String,
    created_at: String,
    entry_count: u64,
    /// Hash version: "2" = M31 Poseidon2 (current). Absent or "1" = legacy felt252.
    #[serde(default = "default_hash_version")]
    hash_version: String,
}

fn default_hash_version() -> String {
    "1".to_string()
}

// ─── Inference Log ──────────────────────────────────────────────────────────

/// Append-only inference log with M31-native Poseidon2 Merkle commitment.
///
/// During serving, entries are appended to disk (JSONL + binary sidecar)
/// and to an in-memory Merkle tree. The Merkle root changes with each
/// append, enabling audit proofs over any subset of entries.
pub struct InferenceLog {
    /// Session directory path.
    log_dir: PathBuf,
    /// Model identifier.
    model_id: String,
    /// Weight commitment hash.
    weight_commitment: String,
    /// Human-readable model name.
    model_name: String,

    // ── In-memory state ──
    /// All entries in insertion order.
    entries: Vec<InferenceLogEntry>,
    /// M31-native Merkle tree of entry hashes.
    merkle: AuditMerkleTree,
    /// Next sequence number to assign.
    next_sequence: u64,
    /// Hash of the most recent entry (M31 digest hex, "0x0" if empty).
    last_entry_hash: String,

    // ── File handles ──
    /// JSONL writer (one entry per line).
    log_writer: BufWriter<File>,
    /// Binary M31 matrix sidecar writer.
    matrix_writer: BufWriter<File>,
    /// Current byte offset in matrices.bin.
    matrix_offset: u64,
}

impl InferenceLog {
    /// Create a new inference log session.
    ///
    /// Creates the session directory and initializes:
    /// - `meta.json` — session metadata (hash_version = "2")
    /// - `log.jsonl` — empty JSONL file
    /// - `matrices.bin` — empty binary sidecar
    pub fn new(
        log_dir: impl AsRef<Path>,
        model_id: &str,
        weight_commitment: &str,
        model_name: &str,
    ) -> Result<Self, AuditError> {
        let log_dir = log_dir.as_ref().to_path_buf();
        fs::create_dir_all(&log_dir)?;

        // Write session metadata.
        let meta = SessionMeta {
            model_id: model_id.to_string(),
            weight_commitment: weight_commitment.to_string(),
            model_name: model_name.to_string(),
            created_at: now_iso8601(),
            entry_count: 0,
            hash_version: "2".to_string(),
        };
        let meta_path = log_dir.join("meta.json");
        let meta_file = File::create(&meta_path)?;
        serde_json::to_writer_pretty(meta_file, &meta)
            .map_err(|e| AuditError::Serde(e.to_string()))?;

        // Open JSONL log for append.
        let log_path = log_dir.join("log.jsonl");
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;
        let log_writer = BufWriter::new(log_file);

        // Open binary sidecar for append.
        let matrix_path = log_dir.join("matrices.bin");
        let matrix_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&matrix_path)?;
        let matrix_writer = BufWriter::new(matrix_file);

        Ok(Self {
            log_dir,
            model_id: model_id.to_string(),
            weight_commitment: weight_commitment.to_string(),
            model_name: model_name.to_string(),
            entries: Vec::new(),
            merkle: AuditMerkleTree::new(),
            next_sequence: 0,
            last_entry_hash: "0x0".to_string(),
            log_writer,
            matrix_writer,
            matrix_offset: 0,
        })
    }

    /// Load an existing inference log from disk.
    ///
    /// Reads `meta.json`, replays `log.jsonl` to rebuild in-memory state,
    /// and opens files for continued appending.
    pub fn load(log_dir: impl AsRef<Path>) -> Result<Self, AuditError> {
        let log_dir = log_dir.as_ref().to_path_buf();

        // Read session metadata.
        let meta_path = log_dir.join("meta.json");
        let meta_file = File::open(&meta_path)?;
        let meta: SessionMeta =
            serde_json::from_reader(meta_file).map_err(|e| AuditError::Serde(e.to_string()))?;

        // Replay log entries to rebuild in-memory state.
        let log_path = log_dir.join("log.jsonl");
        let mut entries = Vec::new();
        let mut merkle = AuditMerkleTree::new();
        let mut last_entry_hash = "0x0".to_string();

        if log_path.exists() {
            let file = File::open(&log_path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let entry: InferenceLogEntry =
                    serde_json::from_str(&line).map_err(|e| AuditError::Serde(e.to_string()))?;

                let hash = parse_digest_or_zero(&entry.entry_hash);
                merkle.push(hash);
                last_entry_hash = entry.entry_hash.clone();
                entries.push(entry);
            }
        }

        let next_sequence = entries.len() as u64;

        // Current matrix sidecar offset.
        let matrix_path = log_dir.join("matrices.bin");
        let matrix_offset = if matrix_path.exists() {
            fs::metadata(&matrix_path)?.len()
        } else {
            0
        };

        // Open files for continued appending.
        let log_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_dir.join("log.jsonl"))?;
        let log_writer = BufWriter::new(log_file);

        let matrix_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&matrix_path)?;
        let matrix_writer = BufWriter::new(matrix_file);

        Ok(Self {
            log_dir,
            model_id: meta.model_id,
            weight_commitment: meta.weight_commitment,
            model_name: meta.model_name,
            entries,
            merkle,
            next_sequence,
            last_entry_hash,
            log_writer,
            matrix_writer,
            matrix_offset,
        })
    }

    /// Append an inference entry to the log.
    ///
    /// 1. Assigns sequence number and chain link
    /// 2. Computes M31 Poseidon2 entry hash
    /// 3. Appends to JSONL file
    /// 4. Updates Merkle tree
    ///
    /// Returns the assigned sequence number.
    pub fn append(&mut self, mut entry: InferenceLogEntry) -> Result<u64, AuditError> {
        // Assign sequence number.
        entry.sequence_number = self.next_sequence;

        // Chain link to previous entry.
        entry.prev_entry_hash = self.last_entry_hash.clone();

        // Compute entry hash (M31 Poseidon2).
        let hash_digest = compute_entry_hash(&entry);
        entry.entry_hash = digest_to_hex(&hash_digest);

        // Append to JSONL.
        serde_json::to_writer(&mut self.log_writer, &entry)
            .map_err(|e| AuditError::Serde(e.to_string()))?;
        self.log_writer.write_all(b"\n")?;
        self.log_writer.flush()?;

        // Update Merkle tree with M31 digest.
        self.merkle.push(hash_digest);

        // Update state.
        self.last_entry_hash = entry.entry_hash.clone();
        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.entries.push(entry);

        // Persist entry count.
        self.update_meta()?;

        Ok(seq)
    }

    /// Write M31 matrix data to the binary sidecar.
    ///
    /// Returns `(offset, size)` for the entry's `matrix_offset` and `matrix_size` fields.
    /// Call this **before** `append()` and set the returned values on the entry.
    ///
    /// Binary format: `[rows: u32 LE][cols: u32 LE][data: u32 LE * rows*cols]`
    pub fn write_matrix(
        &mut self,
        rows: u32,
        cols: u32,
        data: &[u32],
    ) -> Result<(u64, u64), AuditError> {
        let offset = self.matrix_offset;

        self.matrix_writer.write_all(&rows.to_le_bytes())?;
        self.matrix_writer.write_all(&cols.to_le_bytes())?;
        for &val in data {
            self.matrix_writer.write_all(&val.to_le_bytes())?;
        }
        self.matrix_writer.flush()?;

        let size = 8 + (data.len() as u64 * 4);
        self.matrix_offset += size;

        Ok((offset, size))
    }

    /// Read M31 matrix data from the binary sidecar.
    ///
    /// Returns `(rows, cols, data)`.
    pub fn read_matrix(&self, offset: u64, size: u64) -> Result<(u32, u32, Vec<u32>), AuditError> {
        let matrix_path = self.log_dir.join("matrices.bin");
        let mut file = File::open(&matrix_path)?;
        file.seek(SeekFrom::Start(offset))?;

        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        let rows = u32::from_le_bytes(buf);
        file.read_exact(&mut buf)?;
        let cols = u32::from_le_bytes(buf);

        let data_count = ((size - 8) / 4) as usize;
        let mut data = Vec::with_capacity(data_count);
        for _ in 0..data_count {
            file.read_exact(&mut buf)?;
            data.push(u32::from_le_bytes(buf));
        }

        Ok((rows, cols, data))
    }

    /// Query entries within a time window `[start_ns, end_ns]`.
    ///
    /// Returns a `LogWindow` with matching entries, their M31 Merkle root,
    /// and whether chain integrity holds for the subset.
    pub fn query_window(&self, start_ns: u64, end_ns: u64) -> LogWindow {
        let entries: Vec<InferenceLogEntry> = self
            .entries
            .iter()
            .filter(|e| e.timestamp_ns >= start_ns && e.timestamp_ns <= end_ns)
            .cloned()
            .collect();

        // Build an M31 Merkle root over just the window entries.
        let leaf_hashes: Vec<M31Digest> = entries
            .iter()
            .map(|e| parse_digest_or_zero(&e.entry_hash))
            .collect();
        let merkle_root = build_merkle_root_m31(&leaf_hashes);

        let chain_verified = verify_chain_subset(&entries);

        LogWindow {
            entries,
            merkle_root,
            start_ns,
            end_ns,
            chain_verified,
        }
    }

    /// Current Merkle root of the entire log (M31 digest).
    pub fn merkle_root(&self) -> M31Digest {
        self.merkle.root()
    }

    /// Merkle root covering only the first `count` entries.
    pub fn merkle_root_at(&self, count: usize) -> M31Digest {
        self.merkle.root_at(count)
    }

    /// Merkle inclusion proof for the entry at `index`.
    pub fn merkle_proof(&self, index: usize) -> Vec<M31Digest> {
        self.merkle.proof(index)
    }

    /// Verify chain link integrity of all entries.
    ///
    /// Checks:
    /// 1. First entry has `prev_entry_hash == "0x0"`
    /// 2. Each entry's `entry_hash` matches recomputation
    /// 3. Each entry's `prev_entry_hash` matches the previous entry's `entry_hash`
    pub fn verify_chain(&self) -> Result<(), AuditError> {
        if self.entries.is_empty() {
            return Ok(());
        }

        // First entry must link to 0x0.
        if self.entries[0].prev_entry_hash != "0x0" {
            return Err(AuditError::ChainBroken {
                sequence: 0,
                detail: format!(
                    "first entry prev_entry_hash should be 0x0, got {}",
                    self.entries[0].prev_entry_hash
                ),
            });
        }

        // Verify each entry's hash matches recomputation.
        for entry in &self.entries {
            let computed = digest_to_hex(&compute_entry_hash(entry));
            if computed != entry.entry_hash {
                return Err(AuditError::ChainBroken {
                    sequence: entry.sequence_number,
                    detail: format!(
                        "entry_hash mismatch: stored {}, computed {}",
                        entry.entry_hash, computed
                    ),
                });
            }
        }

        // Verify chain links.
        for i in 1..self.entries.len() {
            if self.entries[i].prev_entry_hash != self.entries[i - 1].entry_hash {
                return Err(AuditError::ChainBroken {
                    sequence: self.entries[i].sequence_number,
                    detail: format!(
                        "prev_entry_hash {} does not match previous entry_hash {}",
                        self.entries[i].prev_entry_hash,
                        self.entries[i - 1].entry_hash
                    ),
                });
            }
        }

        Ok(())
    }

    /// Number of entries in the log.
    pub fn entry_count(&self) -> u64 {
        self.entries.len() as u64
    }

    /// Session directory path.
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }

    /// Model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Weight commitment.
    pub fn weight_commitment(&self) -> &str {
        &self.weight_commitment
    }

    /// All entries in insertion order.
    pub fn entries(&self) -> &[InferenceLogEntry] {
        &self.entries
    }

    /// Hash of the most recent entry ("0x0" if empty).
    pub fn last_entry_hash(&self) -> &str {
        &self.last_entry_hash
    }

    /// Update meta.json with current entry count.
    fn update_meta(&self) -> Result<(), AuditError> {
        let meta = SessionMeta {
            model_id: self.model_id.clone(),
            weight_commitment: self.weight_commitment.clone(),
            model_name: self.model_name.clone(),
            created_at: now_iso8601(),
            entry_count: self.entries.len() as u64,
            hash_version: "2".to_string(),
        };
        let meta_path = self.log_dir.join("meta.json");
        let meta_file = File::create(&meta_path)?;
        serde_json::to_writer_pretty(meta_file, &meta)
            .map_err(|e| AuditError::Serde(e.to_string()))?;
        Ok(())
    }
}

/// Verify chain integrity for a subset of entries.
fn verify_chain_subset(entries: &[InferenceLogEntry]) -> bool {
    if entries.len() <= 1 {
        return true;
    }
    for i in 1..entries.len() {
        if entries[i].sequence_number == entries[i - 1].sequence_number + 1
            && entries[i].prev_entry_hash != entries[i - 1].entry_hash
        {
            return false;
        }
    }
    true
}

/// Simple ISO 8601 timestamp from system clock.
fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let days = secs / 86400;
    let rem = secs % 86400;
    let hours = rem / 3600;
    let mins = (rem % 3600) / 60;
    let s = rem % 60;

    let (year, month, day) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, mins, s
    )
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_days: &[u64] = if is_leap(year) {
        &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &md in month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::digest::hex_to_digest;
    use std::time::{SystemTime, UNIX_EPOCH};
    use stwo::core::fields::m31::BaseField as M31;

    /// Create a test entry with given sequence.
    fn make_entry(seq: u64, timestamp_ns: u64) -> InferenceLogEntry {
        InferenceLogEntry {
            inference_id: seq,
            sequence_number: 0, // Assigned by log.
            model_id: "0x2".to_string(),
            weight_commitment: "0xabc".to_string(),
            model_name: "test-model".to_string(),
            num_layers: 4,
            input_tokens: vec![1, 2, 3],
            output_tokens: vec![4, 5],
            matrix_offset: 0,
            matrix_size: 0,
            input_rows: 1,
            input_cols: 3,
            output_rows: 1,
            output_cols: 2,
            io_commitment: "0xdef".to_string(),
            layer_chain_commitment: "0x123".to_string(),
            prev_entry_hash: String::new(), // Set by log.
            entry_hash: String::new(),      // Set by log.
            timestamp_ns,
            latency_ms: 100,
            gpu_device: "H100".to_string(),
            tee_report_hash: "0x0".to_string(),
            task_category: Some("qa".to_string()),
            input_preview: Some("What is 2+2?".to_string()),
            output_preview: Some("4".to_string()),
        }
    }

    fn temp_dir() -> PathBuf {
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("stwo_ml_test_log_{}", d))
    }

    #[test]
    fn test_append_and_chain_integrity() {
        let dir = temp_dir();
        let mut log = InferenceLog::new(&dir, "0x2", "0xabc", "test-model").expect("create log");

        let base_ts = 1_000_000_000_000u64;
        for i in 0..100 {
            let entry = make_entry(i, base_ts + i * 1_000_000);
            log.append(entry).expect("append");
        }

        assert_eq!(log.entry_count(), 100);
        log.verify_chain().expect("chain should be valid");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_query_window() {
        let dir = temp_dir();
        let mut log = InferenceLog::new(&dir, "0x2", "0xabc", "test-model").expect("create log");

        let base_ts = 1_000_000_000_000u64;
        for i in 0..20 {
            let entry = make_entry(i, base_ts + i * 1_000_000);
            log.append(entry).expect("append");
        }

        let window = log.query_window(base_ts + 5_000_000, base_ts + 14_000_000);
        assert_eq!(window.entries.len(), 10);
        assert_eq!(window.entries[0].inference_id, 5);
        assert_eq!(window.entries[9].inference_id, 14);
        assert!(window.chain_verified);

        let empty = log.query_window(base_ts + 100_000_000, base_ts + 200_000_000);
        assert!(empty.entries.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_merkle_root_deterministic() {
        let dir1 = temp_dir();
        let dir2 = temp_dir();
        let mut log1 = InferenceLog::new(&dir1, "0x2", "0xabc", "test-model").expect("create log1");
        let mut log2 = InferenceLog::new(&dir2, "0x2", "0xabc", "test-model").expect("create log2");

        let base_ts = 1_000_000_000_000u64;
        for i in 0..10 {
            let entry1 = make_entry(i, base_ts + i * 1_000_000);
            let entry2 = make_entry(i, base_ts + i * 1_000_000);
            log1.append(entry1).expect("append log1");
            log2.append(entry2).expect("append log2");
        }

        assert_eq!(log1.merkle_root(), log2.merkle_root());
        assert_ne!(log1.merkle_root(), ZERO_DIGEST);

        let _ = fs::remove_dir_all(&dir1);
        let _ = fs::remove_dir_all(&dir2);
    }

    #[test]
    fn test_merkle_proof_verifies() {
        let dir = temp_dir();
        let mut log = InferenceLog::new(&dir, "0x2", "0xabc", "test-model").expect("create log");

        let base_ts = 1_000_000_000_000u64;
        for i in 0..8 {
            let entry = make_entry(i, base_ts + i * 1_000_000);
            log.append(entry).expect("append");
        }

        let root = log.merkle_root();
        for i in 0..8 {
            let proof = log.merkle_proof(i);
            let leaf = parse_digest_or_zero(&log.entries()[i].entry_hash);
            assert!(
                AuditMerkleTree::verify_proof(leaf, i, &proof, root),
                "proof failed for index {}",
                i
            );
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_log_survives_reload() {
        let dir = temp_dir();

        {
            let mut log =
                InferenceLog::new(&dir, "0x2", "0xabc", "test-model").expect("create log");
            let base_ts = 1_000_000_000_000u64;
            for i in 0..25 {
                let entry = make_entry(i, base_ts + i * 1_000_000);
                log.append(entry).expect("append");
            }
        }

        let loaded = InferenceLog::load(&dir).expect("load log");
        assert_eq!(loaded.entry_count(), 25);
        loaded
            .verify_chain()
            .expect("chain should be valid after reload");

        let fresh_root = {
            let mut log2 =
                InferenceLog::new(temp_dir(), "0x2", "0xabc", "test-model").expect("create");
            let base_ts = 1_000_000_000_000u64;
            for i in 0..25 {
                let entry = make_entry(i, base_ts + i * 1_000_000);
                log2.append(entry).expect("append");
            }
            log2.merkle_root()
        };
        assert_eq!(loaded.merkle_root(), fresh_root);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_empty_log() {
        let dir = temp_dir();
        let log = InferenceLog::new(&dir, "0x2", "0xabc", "test-model").expect("create log");

        assert_eq!(log.entry_count(), 0);
        assert_eq!(log.merkle_root(), ZERO_DIGEST);
        log.verify_chain().expect("empty chain is valid");

        let window = log.query_window(0, u64::MAX);
        assert!(window.entries.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_matrix_sidecar_roundtrip() {
        let dir = temp_dir();
        let mut log = InferenceLog::new(&dir, "0x2", "0xabc", "test-model").expect("create log");

        let data1: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let (off1, sz1) = log.write_matrix(2, 3, &data1).expect("write matrix 1");
        let data2: Vec<u32> = vec![10, 20, 30, 40];
        let (off2, sz2) = log.write_matrix(1, 4, &data2).expect("write matrix 2");

        let (r1, c1, d1) = log.read_matrix(off1, sz1).expect("read matrix 1");
        assert_eq!((r1, c1), (2, 3));
        assert_eq!(d1, data1);

        let (r2, c2, d2) = log.read_matrix(off2, sz2).expect("read matrix 2");
        assert_eq!((r2, c2), (1, 4));
        assert_eq!(d2, data2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_merkle_tree_incremental_matches_batch() {
        let mut tree = AuditMerkleTree::new();
        let leaves: Vec<M31Digest> = (0..17)
            .map(|i| {
                let val = M31::from((i + 1) as u32);
                [val, val, val, val, val, val, val, val]
            })
            .collect();

        for &leaf in &leaves {
            tree.push(leaf);
        }

        let incremental_root = tree.root();
        let batch_root = build_merkle_root_m31(&leaves);
        assert_eq!(incremental_root, batch_root);
    }

    #[test]
    fn test_merkle_root_at() {
        let mut tree = AuditMerkleTree::new();
        let leaves: Vec<M31Digest> = (0..8)
            .map(|i| {
                let val = M31::from((i + 1) as u32);
                [val, val, val, val, val, val, val, val]
            })
            .collect();

        let mut roots = Vec::new();
        for &leaf in &leaves {
            tree.push(leaf);
            roots.push(tree.root());
        }

        for i in 1..=8 {
            assert_eq!(tree.root_at(i), roots[i - 1]);
        }
    }

    #[test]
    fn test_append_continues_after_reload() {
        let dir = temp_dir();

        {
            let mut log = InferenceLog::new(&dir, "0x2", "0xabc", "test-model").expect("create");
            let base_ts = 1_000_000_000_000u64;
            for i in 0..10 {
                log.append(make_entry(i, base_ts + i * 1_000_000))
                    .expect("append");
            }
        }

        let mut log = InferenceLog::load(&dir).expect("load");
        let base_ts = 1_000_000_000_000u64;
        for i in 10..20 {
            log.append(make_entry(i, base_ts + i * 1_000_000))
                .expect("append after reload");
        }

        assert_eq!(log.entry_count(), 20);
        log.verify_chain()
            .expect("chain valid after reload + append");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_entry_hash_uses_m31_format() {
        let dir = temp_dir();
        let mut log = InferenceLog::new(&dir, "0x2", "0xabc", "test-model").expect("create log");

        let entry = make_entry(0, 1_000_000_000_000);
        log.append(entry).expect("append");

        // Entry hash should be in M31 digest hex format (66 chars: 0x + 64)
        let hash = &log.entries()[0].entry_hash;
        assert_eq!(hash.len(), 66, "M31 digest hex should be 66 chars");
        assert!(hash.starts_with("0x"));

        // Should be parseable as M31 digest
        let parsed = hex_to_digest(hash);
        assert!(parsed.is_ok(), "entry hash should parse as M31 digest");

        let _ = fs::remove_dir_all(&dir);
    }
}
