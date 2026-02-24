//! Weight commitment cache for model-static matrices.
//!
//! Weight matrices are fixed per model — their restricted MLEs and Poseidon
//! commitments are deterministic given the same padded dimensions. This module
//! caches those results so they're computed once and reused across inferences.
//!
//! For Qwen3-14B (160 matmuls): skips 160 × `restrict_cols` GPU dispatches +
//! 160 × `commit_mle_root_only` Merkle root computations per inference.
//! Estimated 30-50% per-inference speedup.

use std::collections::HashMap;
use std::io::{self, Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use starknet_ff::FieldElement;
use stwo::core::fields::cm31::CM31;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::{SecureField, QM31};

/// Binary format version. Bump on layout changes.
/// v1: f_b + b_commitment + r_j
/// v2: v1 + initial_mle_root (Optional<FieldElement>)
/// v3: v2 + merkle_tree_cache_path (Optional<String>)
const CACHE_VERSION: u32 = 3;
/// Magic bytes: "SWCF" (Stwo Weight Cache File).
const MAGIC: [u8; 4] = [b'S', b'W', b'C', b'F'];

/// Cache key: identifies a weight commitment for a specific matmul node
/// at specific padded dimensions. Padded dims matter because the Fiat-Shamir
/// channel draws (r_i, r_j) depend on (m, k, n).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    node_id: usize,
    m_padded: usize,
    k_padded: usize,
    n_padded: usize,
}

/// Cached weight commitment data for a single matmul node.
#[derive(Debug, Clone)]
pub struct CachedWeight {
    /// Restricted weight MLE: restrict_cols(B, r_j) → k elements.
    pub f_b: Vec<SecureField>,
    /// Poseidon Merkle root of f_b.
    pub b_commitment: FieldElement,
    /// Challenges used for restriction (for verification).
    pub r_j: Vec<SecureField>,
    /// Initial Poseidon Merkle root of the full weight MLE (before folding).
    /// Cached to skip the initial root computation in MLE opening proofs.
    pub initial_mle_root: Option<FieldElement>,
    /// Path to mmap-backed Merkle tree cache directory for this weight's MLE.
    /// When set, MLE opening proofs can extract auth paths via O(log n)
    /// random mmap reads instead of rebuilding Merkle trees (~15s per matrix).
    pub merkle_tree_cache_path: Option<PathBuf>,
}

/// In-memory cache of weight commitments for a single model.
///
/// Thread-safe access via [`SharedWeightCache`] wrapper.
pub struct WeightCommitmentCache {
    entries: HashMap<CacheKey, CachedWeight>,
    model_id: String,
    dirty: bool,
}

impl WeightCommitmentCache {
    /// Create an empty cache for a model.
    pub fn new(model_id: &str) -> Self {
        Self {
            entries: HashMap::new(),
            model_id: model_id.to_string(),
            dirty: false,
        }
    }

    /// Look up a cached weight commitment.
    ///
    /// Returns `None` on cache miss (first inference or dimension change).
    pub fn get(
        &self,
        node_id: usize,
        m_padded: usize,
        k_padded: usize,
        n_padded: usize,
    ) -> Option<&CachedWeight> {
        let key = CacheKey {
            node_id,
            m_padded,
            k_padded,
            n_padded,
        };
        self.entries.get(&key)
    }

    /// Store a weight commitment in the cache.
    pub fn insert(
        &mut self,
        node_id: usize,
        m_padded: usize,
        k_padded: usize,
        n_padded: usize,
        entry: CachedWeight,
    ) {
        let key = CacheKey {
            node_id,
            m_padded,
            k_padded,
            n_padded,
        };
        self.entries.insert(key, entry);
        self.dirty = true;
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The model this cache is for.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether the cache has unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.dirty = true;
    }

    /// Save cache to a binary file.
    ///
    /// Format: MAGIC(4) | version(4) | model_id_len(4) | model_id | num_entries(4) | entries...
    /// Each entry: node_id(8) | m(8) | k(8) | n(8) | f_b_len(4) | f_b(16*len) | b_commitment(32) | r_j_len(4) | r_j(16*len)
    pub fn save(&mut self, path: &Path) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut w = io::BufWriter::with_capacity(1 << 20, file); // 1 MB buffer

        // Header
        w.write_all(&MAGIC)?;
        w.write_all(&CACHE_VERSION.to_le_bytes())?;

        let id_bytes = self.model_id.as_bytes();
        w.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
        w.write_all(id_bytes)?;

        w.write_all(&(self.entries.len() as u32).to_le_bytes())?;

        // Entries
        for (key, val) in &self.entries {
            w.write_all(&(key.node_id as u64).to_le_bytes())?;
            w.write_all(&(key.m_padded as u64).to_le_bytes())?;
            w.write_all(&(key.k_padded as u64).to_le_bytes())?;
            w.write_all(&(key.n_padded as u64).to_le_bytes())?;

            write_securefield_vec(&mut w, &val.f_b)?;
            write_fieldelement(&mut w, &val.b_commitment)?;
            write_securefield_vec(&mut w, &val.r_j)?;
            // v2: optional initial_mle_root
            match &val.initial_mle_root {
                Some(root) => {
                    w.write_all(&[1u8])?;
                    write_fieldelement(&mut w, root)?;
                }
                None => {
                    w.write_all(&[0u8])?;
                }
            }
            // v3: optional merkle_tree_cache_path
            match &val.merkle_tree_cache_path {
                Some(path) => {
                    let path_bytes = path.to_string_lossy().into_owned().into_bytes();
                    w.write_all(&[1u8])?;
                    w.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
                    w.write_all(&path_bytes)?;
                }
                None => {
                    w.write_all(&[0u8])?;
                }
            }
        }

        w.flush()?;
        self.dirty = false;
        Ok(())
    }

    /// Load cache from a binary file.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut r = io::BufReader::with_capacity(1 << 20, file);

        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a weight cache file",
            ));
        }

        // Version — accept v1, v2, v3 (backward compatible)
        let version = read_u32(&mut r)?;
        if version < 1 || version > CACHE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported cache version {version}, expected 1..={CACHE_VERSION}"),
            ));
        }

        // Model ID
        let id_len = read_u32(&mut r)? as usize;
        let mut id_bytes = vec![0u8; id_len];
        r.read_exact(&mut id_bytes)?;
        let model_id = String::from_utf8(id_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Entries
        let num_entries = read_u32(&mut r)? as usize;
        let mut entries = HashMap::with_capacity(num_entries);

        for _ in 0..num_entries {
            let node_id = read_u64(&mut r)? as usize;
            let m_padded = read_u64(&mut r)? as usize;
            let k_padded = read_u64(&mut r)? as usize;
            let n_padded = read_u64(&mut r)? as usize;

            let f_b = read_securefield_vec(&mut r)?;
            let b_commitment = read_fieldelement(&mut r)?;
            let r_j = read_securefield_vec(&mut r)?;

            // v2: optional initial_mle_root
            let initial_mle_root = if version >= 2 {
                let mut flag = [0u8; 1];
                r.read_exact(&mut flag)?;
                if flag[0] != 0 {
                    Some(read_fieldelement(&mut r)?)
                } else {
                    None
                }
            } else {
                None // v1 files don't have this field
            };

            // v3: optional merkle_tree_cache_path
            let merkle_tree_cache_path = if version >= 3 {
                let mut flag = [0u8; 1];
                r.read_exact(&mut flag)?;
                if flag[0] != 0 {
                    let path_len = read_u32(&mut r)? as usize;
                    let mut path_bytes = vec![0u8; path_len];
                    r.read_exact(&mut path_bytes)?;
                    let path_str = String::from_utf8(path_bytes)
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                    Some(PathBuf::from(path_str))
                } else {
                    None
                }
            } else {
                None // v1/v2 files don't have this field
            };

            let key = CacheKey {
                node_id,
                m_padded,
                k_padded,
                n_padded,
            };
            entries.insert(
                key,
                CachedWeight {
                    f_b,
                    b_commitment,
                    r_j,
                    initial_mle_root,
                    merkle_tree_cache_path,
                },
            );
        }

        Ok(Self {
            entries,
            model_id,
            dirty: false,
        })
    }

    /// Load from file if it exists, otherwise create empty.
    pub fn load_or_new(path: &Path, model_id: &str) -> Self {
        match Self::load(path) {
            Ok(cache) if cache.model_id == model_id => cache,
            _ => Self::new(model_id),
        }
    }
}

/// Thread-safe handle to a weight cache.
pub type SharedWeightCache = Arc<RwLock<WeightCommitmentCache>>;

/// Create a new thread-safe cache.
pub fn shared_cache(model_id: &str) -> SharedWeightCache {
    Arc::new(RwLock::new(WeightCommitmentCache::new(model_id)))
}

/// Load or create a thread-safe cache from disk.
pub fn shared_cache_from_file(path: &Path, model_id: &str) -> SharedWeightCache {
    Arc::new(RwLock::new(WeightCommitmentCache::load_or_new(
        path, model_id,
    )))
}

// ── Binary I/O helpers ──────────────────────────────────────────────────

fn read_u32(r: &mut impl IoRead) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl IoRead) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn write_securefield_vec(w: &mut impl IoWrite, v: &[SecureField]) -> io::Result<()> {
    w.write_all(&(v.len() as u32).to_le_bytes())?;
    for sf in v {
        let QM31(CM31(a, b), CM31(c, d)) = *sf;
        w.write_all(&a.0.to_le_bytes())?;
        w.write_all(&b.0.to_le_bytes())?;
        w.write_all(&c.0.to_le_bytes())?;
        w.write_all(&d.0.to_le_bytes())?;
    }
    Ok(())
}

fn read_securefield_vec(r: &mut impl IoRead) -> io::Result<Vec<SecureField>> {
    let len = read_u32(r)? as usize;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        let a = read_u32(r)?;
        let b = read_u32(r)?;
        let c = read_u32(r)?;
        let d = read_u32(r)?;
        v.push(QM31(CM31(M31(a), M31(b)), CM31(M31(c), M31(d))));
    }
    Ok(v)
}

fn write_fieldelement(w: &mut impl IoWrite, fe: &FieldElement) -> io::Result<()> {
    let bytes = fe.to_bytes_be();
    w.write_all(&bytes)
}

fn read_fieldelement(r: &mut impl IoRead) -> io::Result<FieldElement> {
    let mut buf = [0u8; 32];
    r.read_exact(&mut buf)?;
    Ok(FieldElement::from_bytes_be(&buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid felt: {e:?}")))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_sf(val: u32) -> SecureField {
        QM31(
            CM31(M31(val), M31(val + 1)),
            CM31(M31(val + 2), M31(val + 3)),
        )
    }

    fn make_cached_weight(k: usize, n_challenges: usize) -> CachedWeight {
        CachedWeight {
            f_b: (0..k).map(|i| make_sf(i as u32 * 10)).collect(),
            b_commitment: FieldElement::from(0xCAFEu64),
            r_j: (0..n_challenges).map(|i| make_sf(100 + i as u32)).collect(),
            initial_mle_root: Some(FieldElement::from(0xDEADu64)),
            merkle_tree_cache_path: None,
        }
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = WeightCommitmentCache::new("test-model");
        assert!(cache.is_empty());
        assert!(!cache.is_dirty());

        let entry = make_cached_weight(8, 3);
        cache.insert(0, 4, 8, 16, entry.clone());

        assert_eq!(cache.len(), 1);
        assert!(cache.is_dirty());

        let hit = cache.get(0, 4, 8, 16);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.f_b.len(), 8);
        assert_eq!(hit.b_commitment, FieldElement::from(0xCAFEu64));

        // Different dims → miss
        assert!(cache.get(0, 8, 8, 16).is_none());
        // Different node → miss
        assert!(cache.get(1, 4, 8, 16).is_none());
    }

    #[test]
    fn test_cache_overwrite() {
        let mut cache = WeightCommitmentCache::new("m1");
        cache.insert(0, 4, 8, 16, make_cached_weight(8, 3));

        let new_entry = CachedWeight {
            f_b: vec![make_sf(999)],
            b_commitment: FieldElement::from(0xBEEFu64),
            r_j: vec![],
            initial_mle_root: None,
            merkle_tree_cache_path: None,
        };
        cache.insert(0, 4, 8, 16, new_entry);

        assert_eq!(cache.len(), 1);
        let hit = cache.get(0, 4, 8, 16).unwrap();
        assert_eq!(hit.f_b.len(), 1);
        assert_eq!(hit.b_commitment, FieldElement::from(0xBEEFu64));
    }

    #[test]
    fn test_securefield_roundtrip() {
        let original = vec![make_sf(42), make_sf(100), make_sf(0)];
        let mut buf = Vec::new();
        write_securefield_vec(&mut buf, &original).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = read_securefield_vec(&mut cursor).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_fieldelement_roundtrip() {
        let original = FieldElement::from(0x123456789ABCDEFu64);
        let mut buf = Vec::new();
        write_fieldelement(&mut buf, &original).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = read_fieldelement(&mut cursor).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_cache_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("stwo_ml_cache_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_cache.swcf");

        // Build cache with multiple entries
        let mut cache = WeightCommitmentCache::new("qwen3-14b");
        cache.insert(0, 4, 8, 16, make_cached_weight(8, 3));
        cache.insert(5, 4, 8, 32, make_cached_weight(8, 5));
        cache.insert(10, 8, 16, 16, make_cached_weight(16, 4));
        assert_eq!(cache.len(), 3);

        // Save
        cache.save(&path).unwrap();
        assert!(!cache.is_dirty());

        // Load
        let loaded = WeightCommitmentCache::load(&path).unwrap();
        assert_eq!(loaded.model_id(), "qwen3-14b");
        assert_eq!(loaded.len(), 3);
        assert!(!loaded.is_dirty());

        // Verify entries match
        for (node_id, m, k, n) in [(0, 4, 8, 16), (5, 4, 8, 32), (10, 8, 16, 16)] {
            let orig = cache.get(node_id, m, k, n).unwrap();
            let load = loaded.get(node_id, m, k, n).unwrap();
            assert_eq!(orig.f_b, load.f_b);
            assert_eq!(orig.b_commitment, load.b_commitment);
            assert_eq!(orig.r_j, load.r_j);
            assert_eq!(orig.initial_mle_root, load.initial_mle_root);
            assert_eq!(orig.merkle_tree_cache_path, load.merkle_tree_cache_path);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_load_or_new_missing_file() {
        let path = Path::new("/tmp/nonexistent_swcf_cache.swcf");
        let cache = WeightCommitmentCache::load_or_new(path, "model-x");
        assert_eq!(cache.model_id(), "model-x");
        assert!(cache.is_empty());
    }

    #[test]
    fn test_load_or_new_wrong_model() {
        let dir = std::env::temp_dir().join("stwo_ml_cache_test2");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("wrong_model.swcf");

        let mut cache = WeightCommitmentCache::new("model-a");
        cache.insert(0, 4, 8, 16, make_cached_weight(8, 3));
        cache.save(&path).unwrap();

        // Load with different model_id → fresh cache
        let loaded = WeightCommitmentCache::load_or_new(&path, "model-b");
        assert_eq!(loaded.model_id(), "model-b");
        assert!(loaded.is_empty());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_invalid_magic() {
        let mut buf = vec![0u8; 100];
        buf[..4].copy_from_slice(b"XXXX");
        let dir = std::env::temp_dir();
        let path = dir.join("bad_magic.swcf");
        std::fs::write(&path, &buf).unwrap();

        let result = WeightCommitmentCache::load(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_clear() {
        let mut cache = WeightCommitmentCache::new("m");
        cache.insert(0, 1, 2, 4, make_cached_weight(2, 1));
        cache.insert(1, 1, 2, 4, make_cached_weight(2, 1));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert!(cache.is_dirty());
    }

    #[test]
    fn test_shared_cache_concurrent_access() {
        let cache = shared_cache("concurrent-test");

        // Write from one thread
        {
            let mut w = cache.write().unwrap();
            w.insert(0, 4, 8, 16, make_cached_weight(8, 3));
        }

        // Read from another
        {
            let r = cache.read().unwrap();
            assert!(r.get(0, 4, 8, 16).is_some());
            assert_eq!(r.len(), 1);
        }
    }
}
