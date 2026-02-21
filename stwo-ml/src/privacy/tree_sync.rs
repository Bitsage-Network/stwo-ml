//! Merkle tree sync for the VM31 privacy pool.
//!
//! Syncs the local Merkle tree with on-chain state by scanning `NoteInserted`
//! events from the pool contract. Provides Merkle proofs against the real
//! pool root for withdraw and spend transactions.

use std::path::{Path, PathBuf};

use stwo::core::fields::m31::BaseField as M31;

use crate::crypto::merkle_m31::{verify_merkle_proof, Digest, MerklePath, PoseidonMerkleTreeM31};
use crate::crypto::poseidon2_m31::RATE;

#[cfg(feature = "audit-http")]
use super::pool_client::{CrossVerifyResult, PoolClient};

// ─── Types ────────────────────────────────────────────────────────────────

const TREE_DEPTH: usize = 20;

#[derive(Debug, thiserror::Error)]
pub enum TreeSyncError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(String),
    #[error("pool error: {0}")]
    Pool(String),
    #[error("sync error: {0}")]
    Sync(String),
    #[error("proof error: {0}")]
    Proof(String),
}

/// Result of a sync operation.
#[derive(Debug)]
pub struct SyncResult {
    pub total_leaves: usize,
    pub events_added: usize,
    pub root_verified: bool,
    /// C6: cross-RPC verification was performed (verify RPCs were configured).
    pub cross_verified: bool,
    /// C6: number of independent RPCs that confirmed the root.
    pub verify_confirmed: u32,
    /// C6: total number of independent RPCs queried.
    pub verify_total: u32,
}

/// Merkle tree synced from the on-chain pool.
///
/// Wraps a `PoseidonMerkleTreeM31` (depth 20) and manages syncing
/// with the pool contract via `NoteInserted` events.
pub struct TreeSync {
    tree: PoseidonMerkleTreeM31,
    last_synced_block: u64,
    cache_path: Option<PathBuf>,
    /// Store leaf commitments for find_commitment lookups.
    leaves: Vec<Digest>,
}

impl TreeSync {
    /// Create a new empty tree (depth 20).
    pub fn new() -> Self {
        Self {
            tree: PoseidonMerkleTreeM31::new(TREE_DEPTH),
            last_synced_block: 0,
            cache_path: None,
            leaves: Vec::new(),
        }
    }

    /// Default cache path: `~/.vm31/tree_cache.json`.
    pub fn default_cache_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".vm31").join("tree_cache.json")
    }

    /// Load from cache file, or create empty if not found / corrupt.
    pub fn load_or_create(path: &Path) -> Result<Self, TreeSyncError> {
        if !path.exists() {
            let mut sync = Self::new();
            sync.cache_path = Some(path.to_path_buf());
            return Ok(sync);
        }

        let contents = std::fs::read_to_string(path)?;
        let parsed: serde_json::Value =
            serde_json::from_str(&contents).map_err(|e| TreeSyncError::Json(e.to_string()))?;

        let version = parsed["version"].as_u64().unwrap_or(0);
        if version != 1 {
            // Incompatible version, start fresh
            let mut sync = Self::new();
            sync.cache_path = Some(path.to_path_buf());
            return Ok(sync);
        }

        let last_synced_block = parsed["last_synced_block"].as_u64().unwrap_or(0);

        let leaf_hexes = parsed["leaves"]
            .as_array()
            .ok_or_else(|| TreeSyncError::Json("missing leaves array".into()))?;

        let mut tree = PoseidonMerkleTreeM31::new(TREE_DEPTH);
        let mut leaves = Vec::with_capacity(leaf_hexes.len());

        for hex_val in leaf_hexes {
            let hex = hex_val.as_str().unwrap_or("");
            let digest = parse_digest_hex(hex)?;
            tree.append(digest);
            leaves.push(digest);
        }

        Ok(Self {
            tree,
            last_synced_block,
            cache_path: Some(path.to_path_buf()),
            leaves,
        })
    }

    /// Sync the tree from the pool contract.
    ///
    /// 1. Query on-chain tree size
    /// 2. If local tree is up-to-date, cross-verify root and return
    /// 3. Otherwise, fetch NoteInserted events and append new leaves
    /// 4. Cross-verify root (C6: primary RPC + independent verification RPCs)
    /// 5. Save cache
    ///
    /// C6: The root is verified against independent RPCs (if configured via
    /// `STARKNET_VERIFY_RPC`) to prevent a malicious primary RPC from feeding
    /// fake events AND confirming a fake root from the same source.
    #[cfg(feature = "audit-http")]
    pub fn sync(&mut self, pool: &PoolClient) -> Result<SyncResult, TreeSyncError> {
        let on_chain_size = pool
            .get_tree_size()
            .map_err(|e| TreeSyncError::Pool(e.to_string()))? as usize;

        let local_size = self.tree.size();

        if local_size == on_chain_size {
            let local_root = self.tree.root();
            let cv = Self::verify_root_cross_rpc(pool, &local_root)?;
            if !cv.is_confirmed() {
                return Err(TreeSyncError::Sync(
                    "local root is not a known pool root (tree may be corrupt)".into(),
                ));
            }
            return Ok(SyncResult {
                total_leaves: on_chain_size,
                events_added: 0,
                root_verified: true,
                cross_verified: cv.cross_verified,
                verify_confirmed: cv.verify_confirmed,
                verify_total: cv.verify_total,
            });
        }

        if local_size > on_chain_size {
            return Err(TreeSyncError::Sync(format!(
                "local tree ({local_size} leaves) is ahead of on-chain ({on_chain_size} leaves)"
            )));
        }

        // Fetch new events starting from last synced block
        let events = pool
            .get_note_inserted_events(self.last_synced_block)
            .map_err(|e| TreeSyncError::Pool(e.to_string()))?;

        let mut events_added = 0;

        for event in &events {
            let expected_index = self.tree.size() as u64;

            // Skip events we already have
            if event.leaf_index < expected_index {
                continue;
            }

            // Verify sequential insertion
            if event.leaf_index != expected_index {
                return Err(TreeSyncError::Sync(format!(
                    "gap in events: expected index {expected_index}, got {}",
                    event.leaf_index
                )));
            }

            self.tree.append(event.commitment);
            self.leaves.push(event.commitment);
            events_added += 1;

            // Track the latest block
            if event.block_number > self.last_synced_block {
                self.last_synced_block = event.block_number;
            }
        }

        // After sync, our tree may have fewer or more leaves than on_chain_size
        // if events arrived during the sync. Check that our root is known.
        if self.tree.size() < on_chain_size {
            return Err(TreeSyncError::Sync(format!(
                "after sync: local size {} < on-chain size {on_chain_size} (missing events?)",
                self.tree.size()
            )));
        }

        // C6: Cross-verify root against independent RPCs
        let local_root = self.tree.root();
        let cv = Self::verify_root_cross_rpc(pool, &local_root)?;
        if !cv.is_confirmed() {
            return Err(TreeSyncError::Sync(
                "root mismatch after sync — root not confirmed by independent RPCs".into(),
            ));
        }

        // Save cache
        self.save_cache()?;

        Ok(SyncResult {
            total_leaves: self.tree.size(),
            events_added,
            root_verified: true,
            cross_verified: cv.cross_verified,
            verify_confirmed: cv.verify_confirmed,
            verify_total: cv.verify_total,
        })
    }

    /// C6: Verify a root via cross-RPC verification.
    ///
    /// If verify RPCs are configured, the root must be confirmed by the primary
    /// AND at least one independent RPC. If no verify RPCs are configured,
    /// falls back to primary-only (backward compatible).
    #[cfg(feature = "audit-http")]
    fn verify_root_cross_rpc(
        pool: &PoolClient,
        root: &Digest,
    ) -> Result<CrossVerifyResult, TreeSyncError> {
        pool.cross_verify_root(root)
            .map_err(|e| TreeSyncError::Pool(e.to_string()))
    }

    /// Generate a Merkle proof for the leaf at `leaf_index`.
    pub fn prove(&self, leaf_index: usize) -> Result<MerklePath, TreeSyncError> {
        if leaf_index >= self.tree.size() {
            return Err(TreeSyncError::Proof(format!(
                "leaf index {leaf_index} out of range (tree has {} leaves)",
                self.tree.size()
            )));
        }
        Ok(self
            .tree
            .prove(leaf_index)
            .map_err(|e| TreeSyncError::Proof(e.to_string()))?)
    }

    /// Get the current tree root.
    pub fn root(&self) -> Digest {
        self.tree.root()
    }

    /// Number of leaves in the tree.
    pub fn size(&self) -> usize {
        self.tree.size()
    }

    /// Find a commitment's leaf index in the tree.
    pub fn find_commitment(&self, commitment: &Digest) -> Option<usize> {
        self.leaves.iter().position(|leaf| leaf == commitment)
    }

    /// Append a leaf directly (for testing or manual reconstruction).
    pub fn append(&mut self, leaf: Digest) -> usize {
        let idx = self.tree.append(leaf);
        self.leaves.push(leaf);
        idx
    }

    /// Save cache to disk.
    pub fn save_cache(&self) -> Result<(), TreeSyncError> {
        let path = match &self.cache_path {
            Some(p) => p,
            None => return Ok(()),
        };

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let leaf_hexes: Vec<String> = self.leaves.iter().map(digest_to_hex).collect();

        let json = serde_json::json!({
            "version": 1,
            "last_synced_block": self.last_synced_block,
            "leaves": leaf_hexes,
        });

        let output =
            serde_json::to_string_pretty(&json).map_err(|e| TreeSyncError::Json(e.to_string()))?;
        std::fs::write(path, output)?;
        Ok(())
    }

    /// Verify a proof locally against the current root.
    pub fn verify_proof(&self, leaf: &Digest, path: &MerklePath) -> bool {
        verify_merkle_proof(&self.tree.root(), leaf, path, self.tree.depth())
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn digest_to_hex(digest: &Digest) -> String {
    let mut s = String::with_capacity(66);
    s.push_str("0x");
    for &elem in digest {
        s.push_str(&format!("{:08x}", elem.0));
    }
    s
}

fn parse_digest_hex(hex: &str) -> Result<Digest, TreeSyncError> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    if hex.len() != 64 {
        return Err(TreeSyncError::Json(format!(
            "expected 64 hex chars for digest, got {}",
            hex.len()
        )));
    }
    let mut digest = [M31::from_u32_unchecked(0); RATE];
    for i in 0..RATE {
        let chunk = &hex[i * 8..(i + 1) * 8];
        let val = u32::from_str_radix(chunk, 16)
            .map_err(|e| TreeSyncError::Json(format!("invalid hex '{}': {}", chunk, e)))?;
        digest[i] = M31::from_u32_unchecked(val);
    }
    Ok(digest)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_commitment(val: u32) -> Digest {
        let mut d = [M31::from_u32_unchecked(0); RATE];
        d[0] = M31::from_u32_unchecked(val);
        d
    }

    #[test]
    fn test_empty_tree() {
        let sync = TreeSync::new();
        assert_eq!(sync.size(), 0);
        // Root should be the zero hash at depth 20
        let empty = PoseidonMerkleTreeM31::new(TREE_DEPTH);
        assert_eq!(sync.root(), empty.root());
    }

    #[test]
    fn test_append_and_prove() {
        let mut sync = TreeSync::new();
        let c1 = make_commitment(42);
        let c2 = make_commitment(99);
        let c3 = make_commitment(7);

        sync.append(c1);
        sync.append(c2);
        sync.append(c3);

        assert_eq!(sync.size(), 3);

        // Prove each leaf and verify
        for (i, commitment) in [c1, c2, c3].iter().enumerate() {
            let path = sync.prove(i).unwrap();
            assert!(sync.verify_proof(commitment, &path));
        }
    }

    #[test]
    fn test_find_commitment() {
        let mut sync = TreeSync::new();
        let c1 = make_commitment(10);
        let c2 = make_commitment(20);
        let c3 = make_commitment(30);

        sync.append(c1);
        sync.append(c2);
        sync.append(c3);

        assert_eq!(sync.find_commitment(&c1), Some(0));
        assert_eq!(sync.find_commitment(&c2), Some(1));
        assert_eq!(sync.find_commitment(&c3), Some(2));
    }

    #[test]
    fn test_find_missing_returns_none() {
        let mut sync = TreeSync::new();
        sync.append(make_commitment(42));

        let missing = make_commitment(999);
        assert_eq!(sync.find_commitment(&missing), None);
    }

    #[test]
    fn test_cache_roundtrip() {
        let tmp = std::env::temp_dir().join("vm31_test_tree_cache.json");

        // Build a tree and save
        let mut sync = TreeSync::new();
        sync.cache_path = Some(tmp.clone());
        sync.append(make_commitment(1));
        sync.append(make_commitment(2));
        sync.append(make_commitment(3));
        let original_root = sync.root();
        let original_size = sync.size();
        sync.save_cache().unwrap();

        // Load from cache
        let loaded = TreeSync::load_or_create(&tmp).unwrap();
        assert_eq!(loaded.size(), original_size);
        assert_eq!(loaded.root(), original_root);

        // Proofs still work
        let path = loaded.prove(1).unwrap();
        assert!(loaded.verify_proof(&make_commitment(2), &path));

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_incremental_rebuild() {
        let tmp = std::env::temp_dir().join("vm31_test_tree_incremental.json");

        // Save with 2 leaves
        let mut sync = TreeSync::new();
        sync.cache_path = Some(tmp.clone());
        sync.append(make_commitment(1));
        sync.append(make_commitment(2));
        sync.save_cache().unwrap();

        // Load and add more
        let mut loaded = TreeSync::load_or_create(&tmp).unwrap();
        assert_eq!(loaded.size(), 2);
        loaded.append(make_commitment(3));
        loaded.append(make_commitment(4));
        assert_eq!(loaded.size(), 4);

        // Verify all proofs
        for (i, val) in [1u32, 2, 3, 4].iter().enumerate() {
            let path = loaded.prove(i).unwrap();
            assert!(loaded.verify_proof(&make_commitment(*val), &path));
        }

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_root_matches_direct_tree() {
        let mut sync = TreeSync::new();
        let mut direct = PoseidonMerkleTreeM31::new(TREE_DEPTH);

        for i in 1..=10 {
            let c = make_commitment(i);
            sync.append(c);
            direct.append(c);
        }

        assert_eq!(sync.root(), direct.root());
    }

    #[test]
    fn test_proof_verifies() {
        let mut sync = TreeSync::new();
        for i in 0..20 {
            sync.append(make_commitment(i + 1));
        }

        // Verify proof for leaf 7
        let leaf = make_commitment(8);
        let path = sync.prove(7).unwrap();
        assert!(verify_merkle_proof(&sync.root(), &leaf, &path, 20));
    }

    #[test]
    fn test_prove_out_of_range() {
        let mut sync = TreeSync::new();
        sync.append(make_commitment(1));

        assert!(sync.prove(1).is_err());
        assert!(sync.prove(100).is_err());
    }
}
