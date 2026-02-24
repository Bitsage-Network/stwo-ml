//! mmap-backed Merkle tree cache for weight opening proofs.
//!
//! Caches full Poseidon Merkle trees to disk so that subsequent proofs of the
//! same model can extract auth paths via O(log n) random mmap reads instead
//! of rebuilding the tree (~15s GPU / ~80s CPU per matrix).
//!
//! On-disk format:
//! ```text
//! Header (48 bytes):
//!   magic:    [u8; 4]  = "SMTC"
//!   version:  u32      = 1
//!   n_layers: u32
//!   _pad:     u32      = 0
//!   total_u64s: u64    (total u64 elements across all layers)
//!   root:     [u8; 32] (root hash for integrity check)
//!
//! Layer table (n_layers × 16 bytes):
//!   offset: u64  (byte offset from start of data section)
//!   count:  u64  (number of hash elements in this layer)
//!
//! Data section:
//!   Layer 0: count_0 × 4 × u64 (leaf hash layer)
//!   Layer 1: count_1 × 4 × u64
//!   ...
//! ```
//!
//! Each element is 4 × u64 = 32 bytes (Poseidon252 field element in limb form).
//!
//! Controlled via `STWO_MERKLE_TREE_CACHE_DIR` environment variable.
//! When set, trees are cached to `$STWO_MERKLE_TREE_CACHE_DIR/<node_id>_round<N>.smtc`.

use std::io::{self, Write as IoWrite};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use starknet_ff::FieldElement;

use crate::crypto::poseidon_merkle::MerkleAuthPath;

/// Magic bytes for cached Merkle tree files.
const MAGIC: [u8; 4] = [b'S', b'M', b'T', b'C'];
/// File format version.
const VERSION: u32 = 1;
/// Header size: 4 (magic) + 4 (version) + 4 (n_layers) + 4 (pad) + 8 (total_u64s) + 32 (root) = 56
const HEADER_SIZE: usize = 56;
/// Each layer table entry: 8 (offset) + 8 (count) = 16 bytes
const LAYER_ENTRY_SIZE: usize = 16;
/// Each element is 4 u64s = 32 bytes
const ELEMENT_SIZE: usize = 32;

/// mmap-backed Merkle tree for instant auth path extraction.
pub struct MmapMerkleTree {
    mmap: Mmap,
    /// (byte_offset_in_file, element_count) per layer
    layer_offsets: Vec<(usize, usize)>,
    root: FieldElement,
}

impl MmapMerkleTree {
    /// Open an existing cached tree file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too small for header",
            ));
        }

        // Validate magic
        if mmap[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a Merkle tree cache file (bad magic)",
            ));
        }

        // Validate version
        let version = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version {version}, expected {VERSION}"),
            ));
        }

        let n_layers = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;

        // Read root hash
        let mut root_bytes = [0u8; 32];
        root_bytes.copy_from_slice(&mmap[24..56]);
        let root = FieldElement::from_bytes_be(&root_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid root: {e:?}")))?;

        // Read layer table
        let table_start = HEADER_SIZE;
        let table_end = table_start + n_layers * LAYER_ENTRY_SIZE;
        if mmap.len() < table_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too small for layer table",
            ));
        }

        let data_section_start = table_end;
        let mut layer_offsets = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let entry_start = table_start + i * LAYER_ENTRY_SIZE;
            let offset = u64::from_le_bytes(
                mmap[entry_start..entry_start + 8].try_into().unwrap(),
            ) as usize;
            let count = u64::from_le_bytes(
                mmap[entry_start + 8..entry_start + 16].try_into().unwrap(),
            ) as usize;
            // offset is relative to data section start
            layer_offsets.push((data_section_start + offset, count));
        }

        Ok(Self {
            mmap,
            layer_offsets,
            root,
        })
    }

    /// Build a Merkle tree from leaf hash data, write to disk, return mmap handle.
    ///
    /// `leaf_felts` are the FieldElement leaves (packed QM31 values).
    /// The tree is built using CPU Poseidon hashing, then each layer is written
    /// to the cache file. The file is then mmap'd for fast random access.
    pub fn build_and_cache(
        leaf_felts: &[FieldElement],
        path: &Path,
    ) -> io::Result<Self> {
        use crate::crypto::poseidon_merkle::PoseidonMerkleTree;

        // Build the full tree on CPU
        let tree = PoseidonMerkleTree::build_parallel(leaf_felts.to_vec());
        let root = tree.root();

        // Extract all layers as FieldElement vecs
        let layers = tree.all_layers_as_felts();
        let n_layers = layers.len();

        // Compute layout
        let mut total_u64s = 0u64;
        let mut layer_data_offsets: Vec<(u64, u64)> = Vec::with_capacity(n_layers);
        let mut current_offset = 0u64;
        for layer in &layers {
            let count = layer.len() as u64;
            layer_data_offsets.push((current_offset, count));
            let layer_bytes = count * ELEMENT_SIZE as u64;
            current_offset += layer_bytes;
            total_u64s += count * 4;
        }

        // Write file
        let file = std::fs::File::create(path)?;
        let mut w = io::BufWriter::with_capacity(1 << 20, file);

        // Header
        w.write_all(&MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(n_layers as u32).to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?; // padding
        w.write_all(&total_u64s.to_le_bytes())?;
        w.write_all(&root.to_bytes_be())?;

        // Layer table
        for &(offset, count) in &layer_data_offsets {
            w.write_all(&offset.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
        }

        // Layer data — each FieldElement stored as 32 raw big-endian bytes
        for layer in &layers {
            for fe in layer {
                w.write_all(&fe.to_bytes_be())?;
            }
        }

        w.flush()?;
        drop(w);

        // Reopen as mmap
        Self::open(path)
    }

    /// Read a single element at (layer, index) via mmap.
    pub fn element_at(&self, layer: usize, index: usize) -> Option<FieldElement> {
        if layer >= self.layer_offsets.len() {
            return None;
        }
        let (byte_offset, count) = self.layer_offsets[layer];
        if index >= count {
            return None;
        }
        let elem_start = byte_offset + index * ELEMENT_SIZE;
        let elem_end = elem_start + ELEMENT_SIZE;
        if elem_end > self.mmap.len() {
            return None;
        }

        // Read 32 raw big-endian bytes
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&self.mmap[elem_start..elem_end]);
        FieldElement::from_bytes_be(&bytes).ok()
    }

    /// Extract auth path for a leaf at the given index.
    ///
    /// Returns sibling nodes from leaf to root, matching PoseidonMerkleTree::prove().
    pub fn prove(&self, leaf_index: usize) -> Option<MerkleAuthPath> {
        if self.layer_offsets.is_empty() {
            return None;
        }

        let n_layers = self.layer_offsets.len();
        // Skip root layer (last one with 1 element) — matches PoseidonMerkleTree::prove()
        let n_sibling_layers = n_layers.saturating_sub(1);
        let mut siblings = Vec::with_capacity(n_sibling_layers);
        let mut idx = leaf_index;

        for layer in 0..n_sibling_layers {
            let sib_idx = idx ^ 1;
            let sib = self.element_at(layer, sib_idx)?;
            siblings.push(sib);
            idx >>= 1;
        }

        Some(MerkleAuthPath { siblings })
    }

    /// Root hash of the cached tree.
    pub fn root(&self) -> FieldElement {
        self.root
    }

    /// Number of layers in the cached tree.
    pub fn num_layers(&self) -> usize {
        self.layer_offsets.len()
    }

    /// Number of elements in a given layer.
    pub fn layer_count(&self, layer: usize) -> usize {
        if layer < self.layer_offsets.len() {
            self.layer_offsets[layer].1
        } else {
            0
        }
    }
}

/// Get the Merkle tree cache directory from env var, if set.
pub fn merkle_cache_dir() -> Option<PathBuf> {
    std::env::var("STWO_MERKLE_TREE_CACHE_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

/// Compute the cache file path for a given node and round.
pub fn merkle_cache_path(cache_dir: &Path, node_id: usize, round: usize) -> PathBuf {
    cache_dir.join(format!("{}_round{}.smtc", node_id, round))
}

/// Try to open a cached Merkle tree for a specific node/round.
///
/// Returns `None` if no cache dir is configured or the file doesn't exist.
pub fn open_merkle_cache(node_id: usize, round: usize) -> Option<MmapMerkleTree> {
    let cache_dir = merkle_cache_dir()?;
    let path = merkle_cache_path(&cache_dir, node_id, round);
    if !path.exists() {
        return None;
    }
    match MmapMerkleTree::open(&path) {
        Ok(tree) => Some(tree),
        Err(e) => {
            eprintln!(
                "[GKR] Failed to open Merkle cache {}: {e}",
                path.display()
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::poseidon_merkle::PoseidonMerkleTree;
    use crate::crypto::poseidon_channel::securefield_to_felt;
    use stwo::core::fields::cm31::CM31;
    use stwo::core::fields::m31::M31;
    use stwo::core::fields::qm31::QM31;

    fn make_leaves(n: usize) -> Vec<FieldElement> {
        (0..n)
            .map(|i| {
                let sf = QM31(
                    CM31(M31::from((i * 7 + 3) as u32), M31::from((i * 13 + 1) as u32)),
                    CM31(M31::from((i * 5 + 2) as u32), M31::from((i * 11 + 4) as u32)),
                );
                securefield_to_felt(sf)
            })
            .collect()
    }

    #[test]
    fn test_build_and_open_roundtrip() {
        let dir = std::env::temp_dir().join("stwo_merkle_cache_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_tree.smtc");

        let leaves = make_leaves(1024);
        let cached = MmapMerkleTree::build_and_cache(&leaves, &path).unwrap();

        // Compare root with CPU tree
        let cpu_tree = PoseidonMerkleTree::build_parallel(leaves.clone());
        assert_eq!(cached.root(), cpu_tree.root());

        // Close and reopen
        drop(cached);
        let reopened = MmapMerkleTree::open(&path).unwrap();
        assert_eq!(reopened.root(), cpu_tree.root());

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_auth_paths_match_cpu() {
        let dir = std::env::temp_dir().join("stwo_merkle_cache_test2");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_auth.smtc");

        let leaves = make_leaves(256);
        let cached = MmapMerkleTree::build_and_cache(&leaves, &path).unwrap();
        let cpu_tree = PoseidonMerkleTree::build_parallel(leaves.clone());

        // Verify auth paths match for several indices
        for idx in [0, 1, 7, 42, 100, 127, 200, 255] {
            let cached_path = cached.prove(idx).unwrap();
            let cpu_path = cpu_tree.prove(idx);
            assert_eq!(
                cached_path.siblings, cpu_path.siblings,
                "auth path mismatch at index {idx}"
            );
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_corrupt_magic_fails() {
        let dir = std::env::temp_dir().join("stwo_merkle_cache_test3");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bad_magic.smtc");

        let mut data = vec![0u8; 100];
        data[..4].copy_from_slice(b"XXXX");
        std::fs::write(&path, &data).unwrap();

        let result = MmapMerkleTree::open(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_element_at_bounds() {
        let dir = std::env::temp_dir().join("stwo_merkle_cache_test4");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_bounds.smtc");

        let leaves = make_leaves(16);
        let cached = MmapMerkleTree::build_and_cache(&leaves, &path).unwrap();

        // Valid access
        assert!(cached.element_at(0, 0).is_some());
        // Out of bounds layer
        assert!(cached.element_at(100, 0).is_none());
        // Out of bounds index
        assert!(cached.element_at(0, 10000).is_none());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_verify_all_layers() {
        let dir = std::env::temp_dir().join("stwo_merkle_cache_test5");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_layers.smtc");

        let leaves = make_leaves(64);
        let cached = MmapMerkleTree::build_and_cache(&leaves, &path).unwrap();

        // Verify we can verify a complete path from leaf to root
        let auth_path = cached.prove(13).unwrap();
        assert!(PoseidonMerkleTree::verify(
            cached.root(),
            13,
            leaves[13],
            &auth_path,
        ));

        // Wrong leaf should fail
        assert!(!PoseidonMerkleTree::verify(
            cached.root(),
            13,
            leaves[14], // wrong leaf
            &auth_path,
        ));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
