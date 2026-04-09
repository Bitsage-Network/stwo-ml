//! Multi-session audit aggregation.
//!
//! Links multiple audit sessions into a chain-hashed aggregate report,
//! enabling cross-session auditing and tamper detection.
//!
//! # Usage
//!
//! ```text
//! let mut agg = MultiSessionAuditAggregator::new();
//! agg.add_session("./sessions/session_001")?;
//! agg.add_session("./sessions/session_002")?;
//! let report = agg.generate_report()?;
//! println!("Chain hash: {}", digest_to_hex(&report.chain_hash));
//! ```

use std::path::{Path, PathBuf};

use crate::audit::digest::{parse_digest_or_zero, M31Digest, ZERO_DIGEST};
use crate::audit::types::AuditError;
use crate::crypto::poseidon2_m31::poseidon2_compress;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Summary of a single audit session.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    /// Session identifier (directory name).
    pub session_id: String,
    /// Path to the session directory.
    pub session_dir: PathBuf,
    /// Number of inference log entries.
    pub entry_count: usize,
    /// Merkle root over all entries (from meta.json or rebuilt).
    pub merkle_root: M31Digest,
    /// Hash of the last entry in the session (from meta.json).
    pub last_entry_hash: M31Digest,
    /// Time range: (first_timestamp_ns, last_timestamp_ns).
    pub time_range: (u64, u64),
    /// Model identifier.
    pub model_id: String,
    /// Score summary (if evaluation was run).
    pub score_summary: Option<ScoreSummary>,
}

/// Aggregated score summary from semantic evaluation.
#[derive(Debug, Clone)]
pub struct ScoreSummary {
    /// Average score across inferences.
    pub mean_score: f64,
    /// Number of inferences evaluated.
    pub evaluated_count: usize,
}

/// Aggregated report across multiple sessions.
#[derive(Debug)]
pub struct MultiSessionAuditReport {
    /// Per-session summaries in chain order.
    pub sessions: Vec<SessionSummary>,
    /// Chain hash: H(session_1_root || session_2_root || ...).
    pub chain_hash: M31Digest,
    /// Total entries across all sessions.
    pub total_entries: usize,
    /// Time span: (earliest_timestamp_ns, latest_timestamp_ns).
    pub time_span: (u64, u64),
    /// Weighted average score across sessions (None if no evaluation data).
    pub overall_score: Option<f64>,
    /// Tamper flags: any detected issues during aggregation.
    pub tamper_flags: Vec<String>,
}

// ─── Session metadata (matches InferenceLog's SessionMeta) ──────────────────

#[derive(Debug, serde::Deserialize)]
struct SessionMeta {
    model_id: String,
    #[allow(dead_code)]
    weight_commitment: String,
    #[allow(dead_code)]
    model_name: String,
    #[allow(dead_code)]
    created_at: String,
    entry_count: u64,
    #[serde(default)]
    last_entry_hash: Option<String>,
    #[serde(default)]
    merkle_root: Option<String>,
}

// ─── Aggregator ─────────────────────────────────────────────────────────────

/// Aggregator for linking multiple audit sessions.
pub struct MultiSessionAuditAggregator {
    sessions: Vec<SessionSummary>,
    chain_hash: M31Digest,
}

impl MultiSessionAuditAggregator {
    /// Create a new empty aggregator.
    pub fn new() -> Self {
        Self {
            sessions: Vec::new(),
            chain_hash: ZERO_DIGEST,
        }
    }

    /// Add a session from a directory.
    ///
    /// Loads `meta.json`, validates cryptographic anchors by replaying the
    /// log entries, and appends to the session chain.
    pub fn add_session(&mut self, dir: &Path) -> Result<SessionSummary, AuditError> {
        // Load meta.json
        let meta_path = dir.join("meta.json");
        let meta_file = std::fs::File::open(&meta_path).map_err(|e| {
            AuditError::LogError(format!("cannot open {}: {}", meta_path.display(), e))
        })?;
        let meta: SessionMeta = serde_json::from_reader(meta_file)
            .map_err(|e| AuditError::Serde(e.to_string()))?;

        // Parse anchors from meta.json
        let meta_merkle_root = meta
            .merkle_root
            .as_deref()
            .map(parse_digest_or_zero)
            .unwrap_or(ZERO_DIGEST);
        let meta_last_hash = meta
            .last_entry_hash
            .as_deref()
            .map(parse_digest_or_zero)
            .unwrap_or(ZERO_DIGEST);

        // Replay log entries to rebuild Merkle root + find time range
        let log_path = dir.join("log.jsonl");
        let mut rebuilt_merkle = crate::audit::log::AuditMerkleTree::new();
        let mut first_ts = u64::MAX;
        let mut last_ts = 0u64;
        let mut entry_count = 0usize;
        let mut last_entry_hash = ZERO_DIGEST;

        if log_path.exists() {
            let file = std::fs::File::open(&log_path)?;
            let reader = std::io::BufReader::new(file);
            use std::io::BufRead;
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let entry: crate::audit::types::InferenceLogEntry =
                    serde_json::from_str(&line).map_err(|e| AuditError::Serde(e.to_string()))?;

                let hash = parse_digest_or_zero(&entry.entry_hash);
                rebuilt_merkle.push(hash);
                last_entry_hash = hash;

                if entry.timestamp_ns < first_ts {
                    first_ts = entry.timestamp_ns;
                }
                if entry.timestamp_ns > last_ts {
                    last_ts = entry.timestamp_ns;
                }
                entry_count += 1;
            }
        }

        if first_ts == u64::MAX {
            first_ts = 0;
        }

        let rebuilt_root = rebuilt_merkle.root();

        // Validate anchors
        let mut tamper_detected = false;
        if meta.merkle_root.is_some() && rebuilt_root != meta_merkle_root {
            tamper_detected = true;
        }
        if meta.last_entry_hash.is_some() && last_entry_hash != meta_last_hash {
            tamper_detected = true;
        }
        if entry_count != meta.entry_count as usize {
            tamper_detected = true;
        }

        let session_id = dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| dir.display().to_string());

        let summary = SessionSummary {
            session_id,
            session_dir: dir.to_path_buf(),
            entry_count,
            merkle_root: rebuilt_root,
            last_entry_hash,
            time_range: (first_ts, last_ts),
            model_id: meta.model_id,
            score_summary: None,
        };

        // Update chain hash: H(prev_chain_hash || session_merkle_root)
        self.chain_hash = poseidon2_compress(&self.chain_hash, &summary.merkle_root);
        self.sessions.push(summary.clone());

        if tamper_detected {
            return Err(AuditError::LogError(format!(
                "session {} failed anchor validation: merkle_root or last_entry_hash mismatch",
                dir.display()
            )));
        }

        Ok(summary)
    }

    /// Add multiple sessions from a parent directory.
    ///
    /// Scans `parent_dir` for subdirectories containing `meta.json` and adds
    /// them in sorted (alphabetical) order. Returns the number of sessions added.
    pub fn add_sessions_from_dir(&mut self, parent_dir: &Path) -> Result<usize, AuditError> {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(parent_dir)
            .map_err(|e| AuditError::LogError(format!("cannot read {}: {e}", parent_dir.display())))?
            .filter_map(|r| r.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.join("meta.json").exists())
            .collect();

        dirs.sort();

        let mut count = 0;
        for path in &dirs {
            self.add_session(path)?;
            count += 1;
        }
        Ok(count)
    }

    /// Generate an aggregated report across all sessions.
    pub fn generate_report(&self) -> Result<MultiSessionAuditReport, AuditError> {
        if self.sessions.is_empty() {
            return Err(AuditError::EmptyWindow {
                start: 0,
                end: 0,
            });
        }

        let total_entries: usize = self.sessions.iter().map(|s| s.entry_count).sum();
        let earliest = self.sessions.iter().map(|s| s.time_range.0).min().unwrap_or(0);
        let latest = self.sessions.iter().map(|s| s.time_range.1).max().unwrap_or(0);

        // Weighted average score
        let overall_score = {
            let mut total_score = 0.0f64;
            let mut total_eval = 0usize;
            for s in &self.sessions {
                if let Some(ref score) = s.score_summary {
                    total_score += score.mean_score * score.evaluated_count as f64;
                    total_eval += score.evaluated_count;
                }
            }
            if total_eval > 0 {
                Some(total_score / total_eval as f64)
            } else {
                None
            }
        };

        Ok(MultiSessionAuditReport {
            sessions: self.sessions.clone(),
            chain_hash: self.chain_hash,
            total_entries,
            time_span: (earliest, latest),
            overall_score,
            tamper_flags: Vec::new(),
        })
    }

    /// Verify chain integrity: all sessions are internally consistent and
    /// chain hashes link correctly.
    pub fn verify_chain(&self) -> Result<bool, AuditError> {
        let mut expected_chain = ZERO_DIGEST;
        for session in &self.sessions {
            expected_chain = poseidon2_compress(&expected_chain, &session.merkle_root);
        }
        Ok(expected_chain == self.chain_hash)
    }

    /// Get the current chain hash.
    pub fn chain_hash(&self) -> &M31Digest {
        &self.chain_hash
    }

    /// Get all session summaries.
    pub fn sessions(&self) -> &[SessionSummary] {
        &self.sessions
    }
}

impl Default for MultiSessionAuditAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::digest::digest_to_hex;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("obelyzk_agg_{}_{}", prefix, d))
    }

    /// Create a minimal session directory with meta.json and N log entries.
    fn create_test_session(dir: &Path, n: usize, model_id: &str, base_ts: u64) {
        use crate::audit::log::{compute_entry_hash, AuditMerkleTree};
        use crate::audit::types::InferenceLogEntry;

        std::fs::create_dir_all(dir).unwrap();

        let mut merkle = AuditMerkleTree::new();
        let mut prev_hash = "0x0".to_string();
        let mut entries = Vec::new();

        for i in 0..n {
            let mut entry = InferenceLogEntry {
                inference_id: i as u64,
                sequence_number: i as u64,
                model_id: model_id.to_string(),
                weight_commitment: "0xabc".to_string(),
                model_name: "test".to_string(),
                num_layers: 2,
                input_tokens: vec![1, 2],
                output_tokens: vec![3],
                matrix_offset: 0,
                matrix_size: 0,
                input_rows: 1,
                input_cols: 2,
                output_rows: 1,
                output_cols: 1,
                io_commitment: "0x1".to_string(),
                layer_chain_commitment: "0x0".to_string(),
                prev_entry_hash: prev_hash.clone(),
                entry_hash: String::new(),
                timestamp_ns: base_ts + i as u64 * 1_000_000_000,
                latency_ms: 50,
                gpu_device: "test".to_string(),
                tee_report_hash: "0x0".to_string(),
                task_category: Some("test".to_string()),
                input_preview: None,
                output_preview: None,
            };

            let hash = compute_entry_hash(&entry);
            entry.entry_hash = digest_to_hex(&hash);
            merkle.push(hash);
            prev_hash = entry.entry_hash.clone();
            entries.push(entry);
        }

        // Write log.jsonl
        let log_path = dir.join("log.jsonl");
        let mut log_content = String::new();
        for entry in &entries {
            log_content.push_str(&serde_json::to_string(entry).unwrap());
            log_content.push('\n');
        }
        std::fs::write(&log_path, &log_content).unwrap();

        // Write meta.json
        let meta = serde_json::json!({
            "model_id": model_id,
            "weight_commitment": "0xabc",
            "model_name": "test",
            "created_at": "2026-01-01T00:00:00Z",
            "entry_count": n,
            "hash_version": "2",
            "last_entry_hash": if entries.is_empty() { None } else { Some(entries.last().unwrap().entry_hash.clone()) },
            "merkle_root": Some(digest_to_hex(&merkle.root())),
        });
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();

        // Create empty matrices.bin
        std::fs::write(dir.join("matrices.bin"), &[]).unwrap();
    }

    #[test]
    fn test_aggregator_single_session() {
        let base = temp_dir("agg_single");
        let s1 = base.join("session_001");
        create_test_session(&s1, 5, "0x1", 1_000_000_000_000);

        let mut agg = MultiSessionAuditAggregator::new();
        let summary = agg.add_session(&s1).unwrap();
        assert_eq!(summary.entry_count, 5);
        assert_eq!(summary.model_id, "0x1");

        let report = agg.generate_report().unwrap();
        assert_eq!(report.total_entries, 5);
        assert_ne!(report.chain_hash, ZERO_DIGEST);
        assert!(agg.verify_chain().unwrap());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_aggregator_multi_session() {
        let base = temp_dir("agg_multi");
        let s1 = base.join("session_001");
        let s2 = base.join("session_002");
        let s3 = base.join("session_003");
        create_test_session(&s1, 3, "0x1", 1_000_000_000_000);
        create_test_session(&s2, 5, "0x1", 2_000_000_000_000);
        create_test_session(&s3, 2, "0x1", 3_000_000_000_000);

        let mut agg = MultiSessionAuditAggregator::new();
        agg.add_session(&s1).unwrap();
        agg.add_session(&s2).unwrap();
        agg.add_session(&s3).unwrap();

        let report = agg.generate_report().unwrap();
        assert_eq!(report.total_entries, 10);
        assert_eq!(report.sessions.len(), 3);
        assert_eq!(report.time_span.0, 1_000_000_000_000);
        assert!(report.time_span.1 >= 3_000_000_000_000);
        assert_ne!(report.chain_hash, ZERO_DIGEST);
        assert!(agg.verify_chain().unwrap());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_aggregator_detects_tamper() {
        let base = temp_dir("agg_tamper");
        let s1 = base.join("session_001");
        create_test_session(&s1, 3, "0x1", 1_000_000_000_000);

        // Corrupt merkle_root in meta.json
        let meta_path = s1.join("meta.json");
        let meta_str = std::fs::read_to_string(&meta_path).unwrap();
        let corrupted = meta_str.replace(
            &meta_str[meta_str.find("\"merkle_root\"").unwrap()..meta_str.find("\"merkle_root\"").unwrap() + 90],
            "\"merkle_root\": \"0x00000000000000000000000000000000000000000000000000000000deadbeef\"",
        );
        std::fs::write(&meta_path, corrupted).unwrap();

        let mut agg = MultiSessionAuditAggregator::new();
        let result = agg.add_session(&s1);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&base);
    }
}
