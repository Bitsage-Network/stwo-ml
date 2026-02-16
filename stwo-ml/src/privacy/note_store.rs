//! Local note tracking for the VM31 privacy pool.
//!
//! Maintains a persistent JSON store of owned notes at `~/.vm31/notes.json`.
//! Supports scanning encrypted memos, balance queries, and note selection.

use std::path::{Path, PathBuf};

use stwo::core::fields::m31::BaseField as M31;

use crate::crypto::commitment::{Note, NoteCommitment, PublicKey};
use crate::crypto::encryption::{decrypt_note_memo, derive_key};
use crate::crypto::poseidon2_m31::RATE;

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

    /// Load from file, or create empty.
    pub fn load(path: &Path) -> Result<Self, NoteStoreError> {
        if !path.exists() {
            return Ok(Self {
                notes: Vec::new(),
                path: path.to_path_buf(),
            });
        }

        let contents = std::fs::read_to_string(path)?;
        let parsed: serde_json::Value =
            serde_json::from_str(&contents).map_err(|e| NoteStoreError::Json(e.to_string()))?;

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
                let commitment = entry["commitment"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
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

        Ok(Self {
            notes,
            path: path.to_path_buf(),
        })
    }

    /// Save to file.
    pub fn save(&self) -> Result<(), NoteStoreError> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

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

        let json = serde_json::json!({ "version": 1, "notes": notes_json });
        let output = serde_json::to_string_pretty(&json)
            .map_err(|e| NoteStoreError::Json(e.to_string()))?;
        std::fs::write(&self.path, output)?;
        Ok(())
    }

    /// Add a new note to the store.
    pub fn add_note(
        &mut self,
        note: &Note,
        commitment_hex: &str,
        merkle_index: usize,
    ) {
        self.notes.push(TrackedNote {
            note: NoteData::from_note(note),
            commitment: commitment_hex.to_string(),
            merkle_index,
            status: NoteStatus::Confirmed,
        });
    }

    /// Add a new note with Pending status (not yet confirmed on-chain).
    pub fn add_note_pending(
        &mut self,
        note: &Note,
        commitment_hex: &str,
    ) {
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
    pub fn scan_memos(
        &mut self,
        memos: &[([M31; RATE], [M31; 4])], // (encrypted_memo, nonce)
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
            .filter(|n| {
                n.status == NoteStatus::Confirmed
                    && n.note.asset_id == asset_id
            })
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
    pub fn select_notes(
        &self,
        asset_id: u32,
        target: u64,
    ) -> Option<(Vec<&TrackedNote>, u64)> {
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
    fn test_note_store_save_load() {
        let tmp = std::env::temp_dir().join("vm31_test_notes.json");
        let sk = test_sk();
        let note = test_note(&sk);
        let commitment_hex = commitment_to_hex(&note.commitment());

        let mut store = NoteStore::load(&tmp).unwrap();
        store.add_note(&note, &commitment_hex, 0);
        store.save().unwrap();

        let loaded = NoteStore::load(&tmp).unwrap();
        assert_eq!(loaded.notes.len(), 1);
        assert_eq!(loaded.notes[0].commitment, commitment_hex);
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

        let encrypted = encrypt_note_memo(&enc_key, &nonce, asset_id, amount_lo, amount_hi, &blinding);

        let mut store = NoteStore {
            notes: Vec::new(),
            path: PathBuf::from("/tmp/test"),
        };
        let found = store.scan_memos(&[(encrypted, nonce)], &vk, &pk, 0);
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
        store.update_merkle_indices(|c| {
            if *c == commitment {
                Some(5)
            } else {
                None
            }
        });

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
}
