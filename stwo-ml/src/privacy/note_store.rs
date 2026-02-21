//! Local note tracking for the VM31 privacy pool.
//!
//! Maintains a persistent JSON store of owned notes at `~/.vm31/notes.json`.
//! Supports scanning encrypted memos, balance queries, and note selection.
//!
//! File locking: `save()` acquires an exclusive advisory lock and `load()`
//! acquires a shared lock on `<path>.lock` so concurrent CLI processes cannot
//! corrupt or partially-read the store.

use std::path::{Path, PathBuf};

use stwo::core::fields::m31::BaseField as M31;

use crate::crypto::commitment::{Note, NoteCommitment, PublicKey};
use crate::crypto::encryption::{
    decrypt_note_memo, derive_key, poseidon2_decrypt, poseidon2_encrypt,
};
use crate::crypto::poseidon2_m31::RATE;

// ─── Advisory file locking (Unix flock) ──────────────────────────────────

/// RAII guard that holds a `flock` advisory lock.  The lock is released when
/// the guard is dropped (the file descriptor is closed).
#[cfg(unix)]
struct FileLockGuard {
    _file: std::fs::File,
}

#[cfg(unix)]
impl FileLockGuard {
    /// Acquire a **shared** lock (multiple readers allowed, blocks writers).
    fn shared(lock_path: &Path) -> Result<Self, std::io::Error> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(lock_path)?;
        // SAFETY: `file` is a valid open File; `as_raw_fd()` returns its fd which
        // remains valid for the lifetime of `file` held by the guard. `flock(2)` is
        // async-signal-safe and the lock is released when the fd is closed (Drop).
        let ret =
            unsafe { libc::flock(std::os::unix::io::AsRawFd::as_raw_fd(&file), libc::LOCK_SH) };
        if ret != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { _file: file })
    }

    /// Acquire an **exclusive** lock (single writer, blocks everyone).
    fn exclusive(lock_path: &Path) -> Result<Self, std::io::Error> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(lock_path)?;
        // SAFETY: same as `shared()` above — valid fd, flock is safe, released on drop.
        let ret =
            unsafe { libc::flock(std::os::unix::io::AsRawFd::as_raw_fd(&file), libc::LOCK_EX) };
        if ret != 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { _file: file })
    }
}

/// Compute the lockfile path: `<data_path>.lock`.
fn lock_path_for(data_path: &Path) -> PathBuf {
    let mut p = data_path.as_os_str().to_owned();
    p.push(".lock");
    PathBuf::from(p)
}

// ─── Types ────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum NoteStoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(String),
    #[error("note not found: {0}")]
    NotFound(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoteStatus {
    Confirmed,
    Pending,
    Spent,
}

/// Serializable note data (M31 values stored as u32).
#[derive(Clone, Debug)]
pub struct NoteData {
    pub owner_pubkey: [u32; 4],
    pub asset_id: u32,
    pub amount_lo: u32,
    pub amount_hi: u32,
    pub blinding: [u32; 4],
}

impl NoteData {
    pub fn from_note(note: &Note) -> Self {
        Self {
            owner_pubkey: [
                note.owner_pubkey[0].0,
                note.owner_pubkey[1].0,
                note.owner_pubkey[2].0,
                note.owner_pubkey[3].0,
            ],
            asset_id: note.asset_id.0,
            amount_lo: note.amount_lo.0,
            amount_hi: note.amount_hi.0,
            blinding: [
                note.blinding[0].0,
                note.blinding[1].0,
                note.blinding[2].0,
                note.blinding[3].0,
            ],
        }
    }

    pub fn to_note(&self) -> Note {
        Note::new(
            self.owner_pubkey.map(M31::from_u32_unchecked),
            M31::from_u32_unchecked(self.asset_id),
            M31::from_u32_unchecked(self.amount_lo),
            M31::from_u32_unchecked(self.amount_hi),
            self.blinding.map(M31::from_u32_unchecked),
        )
    }

    pub fn amount(&self) -> u64 {
        self.amount_lo as u64 + (self.amount_hi as u64) * (1u64 << 31)
    }
}

/// A tracked note with metadata.
#[derive(Clone, Debug)]
pub struct TrackedNote {
    pub note: NoteData,
    pub commitment: String,
    pub merkle_index: usize,
    pub status: NoteStatus,
}

/// In-memory note store with JSON persistence.
pub struct NoteStore {
    pub notes: Vec<TrackedNote>,
    pub path: PathBuf,
}

impl NoteStore {
    /// Default notes path: `~/.vm31/notes.json`.
    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".vm31").join("notes.json")
    }

    /// Load from file, or create empty. If the file is encrypted (v2),
    /// `encryption_key` is required to decrypt.
    ///
    /// Acquires a shared advisory lock on `<path>.lock` so concurrent readers
    /// are allowed but writers are blocked until the read completes.
    pub fn load(path: &Path, encryption_key: Option<&[M31; RATE]>) -> Result<Self, NoteStoreError> {
        if !path.exists() {
            return Ok(Self {
                notes: Vec::new(),
                path: path.to_path_buf(),
            });
        }

        // Acquire shared lock (readers don't block each other)
        #[cfg(unix)]
        let _lock = FileLockGuard::shared(&lock_path_for(path))?;

        let contents = std::fs::read_to_string(path)?;
        let parsed: serde_json::Value =
            serde_json::from_str(&contents).map_err(|e| NoteStoreError::Json(e.to_string()))?;

        let version = parsed["version"].as_u64().unwrap_or(1);

        let notes_value = if version >= 2 {
            // v2: encrypted notes — decrypt first
            let key = encryption_key.ok_or_else(|| {
                NoteStoreError::Json("encrypted note store (v2) requires encryption key".into())
            })?;
            let enc_hex = parsed["encrypted_notes"]
                .as_str()
                .ok_or_else(|| NoteStoreError::Json("missing encrypted_notes".into()))?;
            let nonce_hex = parsed["nonce"]
                .as_str()
                .ok_or_else(|| NoteStoreError::Json("missing nonce".into()))?;
            let data_len = parsed["data_len"]
                .as_u64()
                .ok_or_else(|| NoteStoreError::Json("missing data_len".into()))?
                as usize;

            let encrypted = hex_to_m31_vec(enc_hex)?;
            let nonce = parse_4_m31_hex(nonce_hex)?;
            let decrypted = poseidon2_decrypt(key, &nonce, &encrypted);

            let bytes = m31_to_bytes_packed(&decrypted, data_len);
            let json_str = String::from_utf8(bytes).map_err(|e| {
                NoteStoreError::Json(format!("decryption produced invalid UTF-8: {e}"))
            })?;
            let inner: serde_json::Value = serde_json::from_str(&json_str)
                .map_err(|e| NoteStoreError::Json(format!("decrypted JSON parse error: {e}")))?;
            inner
        } else {
            // v1: plaintext
            if encryption_key.is_some() {
                eprintln!("Warning: note store is plaintext (v1). Re-save with encryption key to encrypt.");
            }
            parsed.clone()
        };

        let notes = parse_notes_array(&notes_value);

        Ok(Self {
            notes,
            path: path.to_path_buf(),
        })
    }

    /// Save to file. If `encryption_key` is provided, notes are encrypted at rest (v2).
    /// Without a key, notes are saved in plaintext (v1, with warning).
    ///
    /// Acquires an exclusive advisory lock on `<path>.lock` so no other reader
    /// or writer can access the file concurrently.
    pub fn save(&self, encryption_key: Option<&[M31; RATE]>) -> Result<(), NoteStoreError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Acquire exclusive lock (blocks all readers and writers)
        #[cfg(unix)]
        let _lock = FileLockGuard::exclusive(&lock_path_for(&self.path))?;

        let notes_json: Vec<serde_json::Value> = self
            .notes
            .iter()
            .map(|n| {
                let status_str = match n.status {
                    NoteStatus::Confirmed => "confirmed",
                    NoteStatus::Pending => "pending",
                    NoteStatus::Spent => "spent",
                };
                serde_json::json!({
                    "owner_pubkey": n.note.owner_pubkey,
                    "asset_id": n.note.asset_id,
                    "amount_lo": n.note.amount_lo,
                    "amount_hi": n.note.amount_hi,
                    "blinding": n.note.blinding,
                    "commitment": n.commitment,
                    "merkle_index": n.merkle_index,
                    "status": status_str,
                })
            })
            .collect();

        let output = if let Some(key) = encryption_key {
            // v2: encrypt notes at rest
            let inner_json = serde_json::json!({ "notes": notes_json });
            let inner_str = serde_json::to_string(&inner_json)
                .map_err(|e| NoteStoreError::Json(e.to_string()))?;
            let inner_bytes = inner_str.as_bytes();

            // Pack bytes into M31 (3 bytes per element, safe: max 2^24 < 2^31)
            let plaintext_m31 = bytes_to_m31_packed(inner_bytes);

            // Generate random nonce
            let nonce = generate_nonce()?;
            let encrypted = poseidon2_encrypt(key, &nonce, &plaintext_m31);

            let enc_hex = m31_vec_to_hex(&encrypted);
            let nonce_hex = format!(
                "0x{:08x}{:08x}{:08x}{:08x}",
                nonce[0].0, nonce[1].0, nonce[2].0, nonce[3].0,
            );

            format!(
                "{{\n  \"version\": 2,\n  \"encrypted_notes\": \"{enc_hex}\",\n  \"nonce\": \"{nonce_hex}\",\n  \"data_len\": {}\n}}",
                inner_bytes.len()
            )
        } else {
            eprintln!(
                "WARNING: saving notes in PLAINTEXT. Use wallet encryption key for production."
            );
            let json = serde_json::json!({ "version": 1, "notes": notes_json });
            serde_json::to_string_pretty(&json).map_err(|e| NoteStoreError::Json(e.to_string()))?
        };

        std::fs::write(&self.path, output)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            if let Err(e) = std::fs::set_permissions(&self.path, perms) {
                eprintln!("Warning: could not set notes file permissions to 0600: {e}");
            }
        }

        Ok(())
    }

    /// Add a new note to the store.
    pub fn add_note(&mut self, note: &Note, commitment_hex: &str, merkle_index: usize) {
        self.notes.push(TrackedNote {
            note: NoteData::from_note(note),
            commitment: commitment_hex.to_string(),
            merkle_index,
            status: NoteStatus::Confirmed,
        });
    }

    /// Add a new note with Pending status (not yet confirmed on-chain).
    pub fn add_note_pending(&mut self, note: &Note, commitment_hex: &str) {
        self.notes.push(TrackedNote {
            note: NoteData::from_note(note),
            commitment: commitment_hex.to_string(),
            merkle_index: 0,
            status: NoteStatus::Pending,
        });
    }

    /// Update merkle indices for pending notes by searching the synced tree.
    ///
    /// Notes with `merkle_index == 0` and non-Spent status are checked against
    /// the tree. If their commitment is found, the index and status are updated.
    pub fn update_merkle_indices<F>(&mut self, find_commitment: F)
    where
        F: Fn(&[M31; RATE]) -> Option<usize>,
    {
        for note in &mut self.notes {
            if note.merkle_index == 0 && note.status != NoteStatus::Spent {
                let reconstructed = note.note.to_note();
                let commitment = reconstructed.commitment();
                if let Some(idx) = find_commitment(&commitment) {
                    note.merkle_index = idx;
                    note.status = NoteStatus::Confirmed;
                }
            }
        }
    }

    /// Mark a note as spent by commitment hex.
    pub fn mark_spent(&mut self, commitment_hex: &str) -> bool {
        for note in &mut self.notes {
            if note.commitment == commitment_hex && note.status != NoteStatus::Spent {
                note.status = NoteStatus::Spent;
                return true;
            }
        }
        false
    }

    /// Trial-decrypt encrypted memos to find incoming notes.
    /// Returns the number of new notes found.
    ///
    /// Accepts both legacy 8-element memos and new 11-element authenticated memos.
    pub fn scan_memos(
        &mut self,
        memos: &[(Vec<M31>, [M31; 4])], // (encrypted_memo, nonce)
        viewing_key: &[M31; 4],
        pubkey: &PublicKey,
        starting_merkle_index: usize,
    ) -> usize {
        let enc_key = derive_key(viewing_key);
        let mut found = 0;

        for (i, (memo, nonce)) in memos.iter().enumerate() {
            if let Some((asset_id, amount_lo, amount_hi, blinding)) =
                decrypt_note_memo(&enc_key, nonce, memo)
            {
                let note = Note::new(*pubkey, asset_id, amount_lo, amount_hi, blinding);
                let commitment = note.commitment();
                let commitment_hex = commitment_to_hex(&commitment);

                // Avoid duplicates
                if !self.notes.iter().any(|n| n.commitment == commitment_hex) {
                    self.add_note(&note, &commitment_hex, starting_merkle_index + i);
                    found += 1;
                }
            }
        }

        found
    }

    /// Get spendable (confirmed, unspent) notes for a given asset.
    pub fn spendable_notes(&self, asset_id: u32) -> Vec<&TrackedNote> {
        let mut notes: Vec<&TrackedNote> = self
            .notes
            .iter()
            .filter(|n| n.status == NoteStatus::Confirmed && n.note.asset_id == asset_id)
            .collect();
        // Sort by amount descending (greedy selection)
        notes.sort_by(|a, b| b.note.amount().cmp(&a.note.amount()));
        notes
    }

    /// Total spendable balance for an asset.
    pub fn balance(&self, asset_id: u32) -> u64 {
        self.spendable_notes(asset_id)
            .iter()
            .map(|n| n.note.amount())
            .sum()
    }

    /// Select notes to cover a target amount (greedy largest-first).
    /// Returns selected notes and any change amount.
    pub fn select_notes(&self, asset_id: u32, target: u64) -> Option<(Vec<&TrackedNote>, u64)> {
        let available = self.spendable_notes(asset_id);
        let mut selected = Vec::new();
        let mut total = 0u64;

        for note in &available {
            if total >= target {
                break;
            }
            selected.push(*note);
            total += note.note.amount();
        }

        if total >= target {
            Some((selected, total - target))
        } else {
            None
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn commitment_to_hex(commitment: &NoteCommitment) -> String {
    let mut s = String::with_capacity(66);
    s.push_str("0x");
    for &elem in commitment {
        s.push_str(&format!("{:08x}", elem.0));
    }
    s
}

fn parse_u32_array_4(val: &serde_json::Value, key: &str) -> [u32; 4] {
    let arr = val[key].as_array();
    match arr {
        Some(a) if a.len() >= 4 => [
            a[0].as_u64().unwrap_or(0) as u32,
            a[1].as_u64().unwrap_or(0) as u32,
            a[2].as_u64().unwrap_or(0) as u32,
            a[3].as_u64().unwrap_or(0) as u32,
        ],
        _ => [0; 4],
    }
}

/// Parse notes from a JSON value containing a "notes" array.
fn parse_notes_array(parsed: &serde_json::Value) -> Vec<TrackedNote> {
    let mut notes = Vec::new();
    if let Some(arr) = parsed["notes"].as_array() {
        for entry in arr {
            let note_data = NoteData {
                owner_pubkey: parse_u32_array_4(entry, "owner_pubkey"),
                asset_id: entry["asset_id"].as_u64().unwrap_or(0) as u32,
                amount_lo: entry["amount_lo"].as_u64().unwrap_or(0) as u32,
                amount_hi: entry["amount_hi"].as_u64().unwrap_or(0) as u32,
                blinding: parse_u32_array_4(entry, "blinding"),
            };
            let commitment = entry["commitment"].as_str().unwrap_or("").to_string();
            let merkle_index = entry["merkle_index"].as_u64().unwrap_or(0) as usize;
            let status = match entry["status"].as_str().unwrap_or("confirmed") {
                "pending" => NoteStatus::Pending,
                "spent" => NoteStatus::Spent,
                _ => NoteStatus::Confirmed,
            };
            notes.push(TrackedNote {
                note: note_data,
                commitment,
                merkle_index,
                status,
            });
        }
    }
    notes
}

/// Pack bytes into M31 elements (3 bytes per element, safe: max 2^24 < 2^31).
fn bytes_to_m31_packed(bytes: &[u8]) -> Vec<M31> {
    bytes
        .chunks(3)
        .map(|chunk| {
            let mut val = 0u32;
            for (i, &b) in chunk.iter().enumerate() {
                val |= (b as u32) << (i * 8);
            }
            M31::from_u32_unchecked(val)
        })
        .collect()
}

/// Unpack M31 elements to bytes (3 bytes per element), truncated to `original_len`.
fn m31_to_bytes_packed(elems: &[M31], original_len: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(elems.len() * 3);
    for &e in elems {
        let val = e.0;
        bytes.push((val & 0xFF) as u8);
        bytes.push(((val >> 8) & 0xFF) as u8);
        bytes.push(((val >> 16) & 0xFF) as u8);
    }
    bytes.truncate(original_len);
    bytes
}

fn m31_vec_to_hex(v: &[M31]) -> String {
    let mut s = String::with_capacity(2 + v.len() * 8);
    s.push_str("0x");
    for &elem in v {
        s.push_str(&format!("{:08x}", elem.0));
    }
    s
}

fn hex_to_m31_vec(hex: &str) -> Result<Vec<M31>, NoteStoreError> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    if hex.len() % 8 != 0 {
        return Err(NoteStoreError::Json(format!(
            "hex length {} not multiple of 8",
            hex.len()
        )));
    }
    let mut result = Vec::with_capacity(hex.len() / 8);
    for i in 0..(hex.len() / 8) {
        let chunk = &hex[i * 8..(i + 1) * 8];
        let val = u32::from_str_radix(chunk, 16)
            .map_err(|e| NoteStoreError::Json(format!("invalid hex '{chunk}': {e}")))?;
        result.push(M31::from_u32_unchecked(val));
    }
    Ok(result)
}

fn parse_4_m31_hex(hex: &str) -> Result<[M31; 4], NoteStoreError> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    if hex.len() != 32 {
        return Err(NoteStoreError::Json(format!(
            "expected 32 hex chars, got {}",
            hex.len()
        )));
    }
    let mut result = [M31::from_u32_unchecked(0); 4];
    for i in 0..4 {
        let chunk = &hex[i * 8..(i + 1) * 8];
        let val = u32::from_str_radix(chunk, 16)
            .map_err(|e| NoteStoreError::Json(format!("invalid hex '{chunk}': {e}")))?;
        result[i] = super::reduce_u32_to_m31(val);
    }
    Ok(result)
}

fn generate_nonce() -> Result<[M31; 4], NoteStoreError> {
    super::random_m31_quad().map_err(|e| NoteStoreError::Json(e))
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::commitment::{derive_pubkey, SpendingKey};
    use crate::crypto::encryption::encrypt_note_memo;

    fn test_sk() -> SpendingKey {
        [42, 99, 7, 13].map(M31::from_u32_unchecked)
    }

    fn test_note(sk: &SpendingKey) -> Note {
        let pk = derive_pubkey(sk);
        Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            [1, 2, 3, 4].map(M31::from_u32_unchecked),
        )
    }

    #[test]
    fn test_note_data_roundtrip() {
        let sk = test_sk();
        let note = test_note(&sk);
        let data = NoteData::from_note(&note);
        let recovered = data.to_note();
        assert_eq!(note.commitment(), recovered.commitment());
    }

    #[test]
    fn test_note_store_save_load_plaintext() {
        let tmp = std::env::temp_dir().join("vm31_test_notes_plain.json");
        let sk = test_sk();
        let note = test_note(&sk);
        let commitment_hex = commitment_to_hex(&note.commitment());

        let mut store = NoteStore::load(&tmp, None).unwrap();
        store.add_note(&note, &commitment_hex, 0);
        store.save(None).unwrap();

        let loaded = NoteStore::load(&tmp, None).unwrap();
        assert_eq!(loaded.notes.len(), 1);
        assert_eq!(loaded.notes[0].commitment, commitment_hex);
        assert_eq!(loaded.notes[0].note.amount(), 1000);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_note_store_save_load_encrypted() {
        let tmp = std::env::temp_dir().join("vm31_test_notes_enc.json");
        let sk = test_sk();
        let vk = crate::crypto::commitment::derive_viewing_key(&sk);
        let enc_key = derive_key(&vk);
        let note = test_note(&sk);
        let commitment_hex = commitment_to_hex(&note.commitment());

        let mut store = NoteStore::load(&tmp, Some(&enc_key)).unwrap();
        store.add_note(&note, &commitment_hex, 0);
        store.save(Some(&enc_key)).unwrap();

        // Verify the file is encrypted (no plaintext blinding values)
        let contents = std::fs::read_to_string(&tmp).unwrap();
        assert!(
            contents.contains("\"version\": 2"),
            "should be v2 encrypted"
        );
        assert!(
            contents.contains("\"encrypted_notes\""),
            "should have encrypted data"
        );
        assert!(
            !contents.contains("\"blinding\""),
            "M6: blinding must NOT appear in plaintext"
        );

        // Load and verify roundtrip
        let loaded = NoteStore::load(&tmp, Some(&enc_key)).unwrap();
        assert_eq!(loaded.notes.len(), 1);
        assert_eq!(loaded.notes[0].commitment, commitment_hex);
        assert_eq!(loaded.notes[0].note.amount(), 1000);
        assert_eq!(loaded.notes[0].note.blinding, [1, 2, 3, 4]);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_note_store_encrypted_wrong_key_fails() {
        let tmp = std::env::temp_dir().join("vm31_test_notes_wrongkey.json");
        let sk = test_sk();
        let vk = crate::crypto::commitment::derive_viewing_key(&sk);
        let enc_key = derive_key(&vk);
        let note = test_note(&sk);
        let commitment_hex = commitment_to_hex(&note.commitment());

        let mut store = NoteStore {
            notes: Vec::new(),
            path: tmp.clone(),
        };
        store.add_note(&note, &commitment_hex, 0);
        store.save(Some(&enc_key)).unwrap();

        // Try loading with a different key
        let wrong_key = [M31::from_u32_unchecked(99); RATE];
        let result = NoteStore::load(&tmp, Some(&wrong_key));
        assert!(result.is_err(), "M6: wrong key must fail decryption");

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_note_store_v1_backward_compat() {
        let tmp = std::env::temp_dir().join("vm31_test_notes_v1compat.json");
        let sk = test_sk();
        let note = test_note(&sk);
        let commitment_hex = commitment_to_hex(&note.commitment());

        // Save as plaintext v1
        let mut store = NoteStore {
            notes: Vec::new(),
            path: tmp.clone(),
        };
        store.add_note(&note, &commitment_hex, 0);
        store.save(None).unwrap();

        // Load with an encryption key — should still work (v1 compat)
        let enc_key = [M31::from_u32_unchecked(42); RATE];
        let loaded = NoteStore::load(&tmp, Some(&enc_key)).unwrap();
        assert_eq!(loaded.notes.len(), 1);
        assert_eq!(loaded.notes[0].note.amount(), 1000);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_mark_spent() {
        let sk = test_sk();
        let note = test_note(&sk);
        let commitment_hex = commitment_to_hex(&note.commitment());

        let mut store = NoteStore {
            notes: Vec::new(),
            path: PathBuf::from("/tmp/test"),
        };
        store.add_note(&note, &commitment_hex, 0);
        assert_eq!(store.balance(0), 1000);

        store.mark_spent(&commitment_hex);
        assert_eq!(store.balance(0), 0);
    }

    #[test]
    fn test_scan_memos() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let vk = crate::crypto::commitment::derive_viewing_key(&sk);
        let enc_key = derive_key(&vk);
        let nonce = [10, 20, 30, 40].map(M31::from_u32_unchecked);

        let asset_id = M31::from_u32_unchecked(0);
        let amount_lo = M31::from_u32_unchecked(500);
        let amount_hi = M31::from_u32_unchecked(0);
        let blinding = [5, 6, 7, 8].map(M31::from_u32_unchecked);

        let encrypted =
            encrypt_note_memo(&enc_key, &nonce, asset_id, amount_lo, amount_hi, &blinding);

        let mut store = NoteStore {
            notes: Vec::new(),
            path: PathBuf::from("/tmp/test"),
        };
        let found = store.scan_memos(&[(encrypted.clone(), nonce)], &vk, &pk, 0);
        assert_eq!(found, 1);
        assert_eq!(store.balance(0), 500);
    }

    #[test]
    fn test_update_merkle_indices() {
        let sk = test_sk();
        let note = test_note(&sk);
        let commitment = note.commitment();
        let commitment_hex = commitment_to_hex(&commitment);

        let mut store = NoteStore {
            notes: Vec::new(),
            path: PathBuf::from("/tmp/test"),
        };

        // Add as pending (index 0)
        store.add_note_pending(&note, &commitment_hex);
        assert_eq!(store.notes[0].status, NoteStatus::Pending);
        assert_eq!(store.notes[0].merkle_index, 0);

        // Simulate tree sync finding this commitment at index 5
        store.update_merkle_indices(|c| if *c == commitment { Some(5) } else { None });

        assert_eq!(store.notes[0].status, NoteStatus::Confirmed);
        assert_eq!(store.notes[0].merkle_index, 5);
    }

    #[test]
    fn test_update_merkle_indices_skips_spent() {
        let sk = test_sk();
        let note = test_note(&sk);
        let commitment = note.commitment();
        let commitment_hex = commitment_to_hex(&commitment);

        let mut store = NoteStore {
            notes: Vec::new(),
            path: PathBuf::from("/tmp/test"),
        };

        store.add_note(&note, &commitment_hex, 0);
        store.mark_spent(&commitment_hex);

        // Should not update spent notes
        store.update_merkle_indices(|_| Some(99));
        assert_eq!(store.notes[0].status, NoteStatus::Spent);
        assert_eq!(store.notes[0].merkle_index, 0);
    }

    #[test]
    fn test_select_notes() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);

        let mut store = NoteStore {
            notes: Vec::new(),
            path: PathBuf::from("/tmp/test"),
        };

        // Add notes of different amounts
        for amount in [100u32, 300, 500, 200] {
            let note = Note::new(
                pk,
                M31::from_u32_unchecked(0),
                M31::from_u32_unchecked(amount),
                M31::from_u32_unchecked(0),
                [amount, amount + 1, amount + 2, amount + 3].map(M31::from_u32_unchecked),
            );
            let hex = commitment_to_hex(&note.commitment());
            store.add_note(&note, &hex, 0);
        }

        let (selected, change) = store.select_notes(0, 700).unwrap();
        let total: u64 = selected.iter().map(|n| n.note.amount()).sum();
        assert!(total >= 700);
        assert_eq!(total - change, 700);

        // Can't select more than balance
        assert!(store.select_notes(0, 2000).is_none());
    }

    // ── M8 regression: file locking ──────────────────────────────────────

    #[test]
    fn test_m8_concurrent_saves_no_corruption() {
        // Two threads race to save different notes to the same file.
        // Without locking, one thread's write could be partially overwritten
        // by the other. With flock, they serialize correctly.
        use std::sync::{Arc, Barrier};

        let tmp = std::env::temp_dir().join("vm31_test_m8_concurrent.json");
        let _ = std::fs::remove_file(&tmp);
        let lock_file = lock_path_for(&tmp);
        let _ = std::fs::remove_file(&lock_file);

        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let barrier = Arc::new(Barrier::new(2));

        let handles: Vec<_> = (0..2u32)
            .map(|thread_id| {
                let path = tmp.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait(); // start at the same time
                    for i in 0..5u32 {
                        let amount = thread_id * 1000 + i;
                        let note = Note::new(
                            pk,
                            M31::from_u32_unchecked(0),
                            M31::from_u32_unchecked(amount),
                            M31::from_u32_unchecked(0),
                            [amount, amount + 1, amount + 2, amount + 3]
                                .map(M31::from_u32_unchecked),
                        );
                        let hex = commitment_to_hex(&note.commitment());

                        // Load → add → save under exclusive lock
                        let mut store = NoteStore::load(&path, None).unwrap();
                        store.add_note(&note, &hex, i as usize);
                        store.save(None).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // The final file must be valid JSON and parseable
        let final_store = NoteStore::load(&tmp, None).unwrap();
        assert!(
            !final_store.notes.is_empty(),
            "M8: concurrent saves must not produce empty/corrupt file"
        );
        // With proper locking, the file is always valid JSON (no partial writes)
        let contents = std::fs::read_to_string(&tmp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert!(
            parsed["notes"].is_array(),
            "M8: file must be valid JSON with notes array"
        );

        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&lock_file);
    }

    #[test]
    fn test_m8_lock_path_derivation() {
        let path = Path::new("/tmp/notes.json");
        let lp = lock_path_for(path);
        assert_eq!(lp, PathBuf::from("/tmp/notes.json.lock"));
    }
}
