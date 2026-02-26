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
/// v4: v3 + weight_fingerprint header (32 bytes, validates weight content)
const CACHE_VERSION: u32 = 4;

/// Number of M31 elements sampled per weight matrix for fingerprinting.
/// With 8 samples × 160 matrices = 1280 elements hashed → ~5µs total.
const FINGERPRINT_SAMPLES_PER_MATRIX: usize = 8;
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
    /// Content-based fingerprint of weight matrices. Detects weight changes
    /// (e.g., fine-tuning) that would invalidate cached Merkle roots.
    fingerprint: [u8; 32],
    dirty: bool,
}

impl WeightCommitmentCache {
    /// Create an empty cache for a model (zero fingerprint).
    pub fn new(model_id: &str) -> Self {
        Self {
            entries: HashMap::new(),
            model_id: model_id.to_string(),
            fingerprint: [0u8; 32],
            dirty: false,
        }
    }

    /// Create a cache with a weight fingerprint computed from `GraphWeights`.
    ///
    /// The fingerprint samples a fixed set of elements from each weight matrix
    /// and hashes them. This detects weight changes (fine-tuning, corruption)
    /// that would invalidate cached Merkle roots.
    pub fn new_with_fingerprint(
        model_id: &str,
        weights: &crate::compiler::graph::GraphWeights,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            model_id: model_id.to_string(),
            fingerprint: compute_weight_fingerprint(weights),
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

    /// The stored weight fingerprint.
    pub fn fingerprint(&self) -> &[u8; 32] {
        &self.fingerprint
    }

    /// Set the fingerprint (marks cache dirty if changed).
    pub fn set_fingerprint(&mut self, fp: [u8; 32]) {
        if self.fingerprint != fp {
            self.fingerprint = fp;
            self.dirty = true;
        }
    }

    /// Validate that the stored fingerprint matches the given weights.
    ///
    /// Returns `true` if fingerprints match (cache is valid), `false` if
    /// weights have changed and cache should be invalidated.
    /// A zero fingerprint (from `new()`) always returns `true` — the cache
    /// was created without fingerprinting and should be trusted.
    pub fn validate_fingerprint(
        &self,
        weights: &crate::compiler::graph::GraphWeights,
    ) -> bool {
        // Zero fingerprint = no validation (backward compat with v1-v3 caches)
        if self.fingerprint == [0u8; 32] {
            return true;
        }
        let current = compute_weight_fingerprint(weights);
        self.fingerprint == current
    }

    /// Look up a cached full-MLE Merkle root for a weight matrix.
    ///
    /// Used by the aggregated oracle sumcheck path which commits the full
    /// weight MLE (not the restricted one). Uses `m_padded = 0` sentinel
    /// to distinguish from regular cache entries.
    pub fn get_root(
        &self,
        node_id: usize,
        rows_padded: usize,
        cols_padded: usize,
    ) -> Option<FieldElement> {
        let key = CacheKey {
            node_id,
            m_padded: 0, // sentinel: full-MLE root entry
            k_padded: rows_padded,
            n_padded: cols_padded,
        };
        self.entries
            .get(&key)
            .and_then(|e| e.initial_mle_root)
    }

    /// Store a full-MLE Merkle root for a weight matrix.
    ///
    /// Counterpart of [`get_root`] for the aggregated oracle sumcheck path.
    pub fn insert_root(
        &mut self,
        node_id: usize,
        rows_padded: usize,
        cols_padded: usize,
        root: FieldElement,
    ) {
        let key = CacheKey {
            node_id,
            m_padded: 0, // sentinel: full-MLE root entry
            k_padded: rows_padded,
            n_padded: cols_padded,
        };
        self.entries.insert(
            key,
            CachedWeight {
                f_b: Vec::new(),
                b_commitment: FieldElement::ZERO,
                r_j: Vec::new(),
                initial_mle_root: Some(root),
                merkle_tree_cache_path: None,
            },
        );
        self.dirty = true;
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.dirty = true;
    }

    /// Save cache to a binary file.
    ///
    /// Format: MAGIC(4) | version(4) | model_id_len(4) | model_id | fingerprint(32) | num_entries(4) | entries...
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

        // v4: weight fingerprint (32 bytes)
        w.write_all(&self.fingerprint)?;

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

        // Version — accept v1, v2, v3, v4 (backward compatible)
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

        // v4: weight fingerprint
        let fingerprint = if version >= 4 {
            let mut fp = [0u8; 32];
            r.read_exact(&mut fp)?;
            fp
        } else {
            [0u8; 32] // v1-v3 files: zero fingerprint (no validation)
        };

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
            fingerprint,
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

    /// Load from file, validate fingerprint against current weights.
    ///
    /// Returns the cached data only if both model_id and weight fingerprint match.
    /// If the fingerprint doesn't match (weights changed), creates a fresh cache
    /// with the correct fingerprint.
    pub fn load_or_new_validated(
        path: &Path,
        model_id: &str,
        weights: &crate::compiler::graph::GraphWeights,
    ) -> Self {
        let current_fp = compute_weight_fingerprint(weights);
        match Self::load(path) {
            Ok(cache) if cache.model_id == model_id => {
                if cache.fingerprint == [0u8; 32] || cache.fingerprint == current_fp {
                    // Valid: zero fp (legacy) or matching fp
                    let mut cache = cache;
                    if cache.fingerprint == [0u8; 32] {
                        // Upgrade legacy cache with fingerprint
                        cache.fingerprint = current_fp;
                        cache.dirty = true;
                    }
                    cache
                } else {
                    // Fingerprint mismatch — weights changed, start fresh
                    eprintln!(
                        "  [weight_cache] fingerprint mismatch for model '{}' — weights changed, invalidating cache",
                        model_id,
                    );
                    Self::new_with_fingerprint(model_id, weights)
                }
            }
            _ => Self::new_with_fingerprint(model_id, weights),
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

/// Load or create a thread-safe cache next to a model directory.
///
/// Cache file is stored at `<model_dir>/.stwo_weight_cache.swcf`.
/// Creates the cache file's parent directory if needed.
pub fn shared_cache_for_model(model_dir: &Path, model_id: &str) -> SharedWeightCache {
    let cache_path = model_dir.join(".stwo_weight_cache.swcf");
    shared_cache_from_file(&cache_path, model_id)
}

/// Load or create a validated cache next to a model directory.
///
/// Like [`shared_cache_for_model`] but also validates the weight fingerprint.
/// If weights have changed (fine-tuning, different checkpoint), the cache
/// is invalidated and a fresh one created with the correct fingerprint.
pub fn shared_cache_for_model_validated(
    model_dir: &Path,
    model_id: &str,
    weights: &crate::compiler::graph::GraphWeights,
) -> SharedWeightCache {
    let cache_path = model_dir.join(".stwo_weight_cache.swcf");
    Arc::new(RwLock::new(WeightCommitmentCache::load_or_new_validated(
        &cache_path, model_id, weights,
    )))
}

/// Save a shared weight cache to its model directory.
///
/// Only writes if the cache has been modified since last save.
/// Returns `Ok(true)` if written, `Ok(false)` if clean/skipped.
pub fn save_shared_cache(
    cache: &SharedWeightCache,
    model_dir: &Path,
) -> io::Result<bool> {
    let cache_path = model_dir.join(".stwo_weight_cache.swcf");
    let mut w = cache
        .write()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("lock poisoned: {e}")))?;
    if !w.is_dirty() {
        return Ok(false);
    }
    w.save(&cache_path)?;
    Ok(true)
}

// ── Weight fingerprinting ────────────────────────────────────────────────

/// Compute a 32-byte fingerprint of weight matrices by sampling elements.
///
/// Samples [`FINGERPRINT_SAMPLES_PER_MATRIX`] elements from each weight matrix
/// at deterministic positions (spread evenly across the data). The sampled
/// values are mixed into a running hash using FNV-1a-like mixing.
///
/// For Qwen3-14B (160 matrices): ~1280 u32 reads → <10µs.
pub fn compute_weight_fingerprint(
    weights: &crate::compiler::graph::GraphWeights,
) -> [u8; 32] {
    // FNV-1a 64-bit basis and prime
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let mut state = [FNV_OFFSET; 4]; // 4 × 64-bit = 256-bit state

    for (node_id, matrix) in &weights.weights {
        let n = matrix.data.len();
        if n == 0 {
            continue;
        }

        // Mix node metadata
        state[0] = state[0].wrapping_mul(FNV_PRIME) ^ (*node_id as u64);
        state[1] = state[1].wrapping_mul(FNV_PRIME) ^ (matrix.rows as u64);
        state[2] = state[2].wrapping_mul(FNV_PRIME) ^ (matrix.cols as u64);
        state[3] = state[3].wrapping_mul(FNV_PRIME) ^ (n as u64);

        // Sample elements at evenly spaced positions
        let step = n.max(1) / FINGERPRINT_SAMPLES_PER_MATRIX.max(1);
        let step = step.max(1);
        for i in 0..FINGERPRINT_SAMPLES_PER_MATRIX.min(n) {
            let idx = (i * step).min(n - 1);
            let val = matrix.data[idx].0 as u64;
            let lane = i % 4;
            state[lane] = state[lane].wrapping_mul(FNV_PRIME) ^ val;
        }
    }

    // Finalize: convert 4×u64 → 32 bytes
    let mut out = [0u8; 32];
    for (i, &s) in state.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&s.to_le_bytes());
    }
    out
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

// ── Background cache pre-warming ─────────────────────────────────────

/// Pre-compute Poseidon Merkle roots for all uncached weight matrices.
///
/// Designed to run on a background thread immediately after model load.
/// While the main thread runs forward pass + GKR walk, this function
/// fills the cache so that `apply_aggregated_oracle_sumcheck()` finds
/// all roots pre-computed.
///
/// On single-GPU machines, falls back to CPU to avoid GPU contention
/// with the prover. Use [`prewarm_weight_roots_gpu_exclusive`] when the
/// caller guarantees no concurrent GPU usage (e.g., blocking prewarm
/// before proving starts).
///
/// Returns the number of newly computed roots.
pub fn prewarm_weight_roots(
    weights: &crate::compiler::graph::GraphWeights,
    cache: &SharedWeightCache,
    model_dir: Option<&std::path::Path>,
) -> usize {
    prewarm_weight_roots_inner(weights, cache, model_dir, false)
}

/// Pre-compute Poseidon Merkle roots using GPU even on single-GPU machines.
///
/// Use this when the caller guarantees no concurrent GPU usage — e.g.,
/// blocking prewarm before proving starts, or `--generate-cache` mode.
/// On single-GPU machines this is ~10x faster than the CPU fallback path.
///
/// Returns the number of newly computed roots.
pub fn prewarm_weight_roots_gpu_exclusive(
    weights: &crate::compiler::graph::GraphWeights,
    cache: &SharedWeightCache,
    model_dir: Option<&std::path::Path>,
) -> usize {
    prewarm_weight_roots_inner(weights, cache, model_dir, true)
}

fn prewarm_weight_roots_inner(
    weights: &crate::compiler::graph::GraphWeights,
    cache: &SharedWeightCache,
    model_dir: Option<&std::path::Path>,
    force_gpu: bool,
) -> usize {
    let uncached: Vec<(usize, usize, usize)> = {
        let r = match cache.read() {
            Ok(r) => r,
            Err(_) => return 0,
        };
        weights
            .weights
            .iter()
            .filter_map(|(node_id, matrix)| {
                let rp = matrix.rows.next_power_of_two();
                let cp = matrix.cols.next_power_of_two();
                if r.get_root(*node_id, rp, cp).is_some() {
                    None
                } else {
                    Some((*node_id, rp, cp))
                }
            })
            .collect()
    };

    if uncached.is_empty() {
        return 0;
    }

    let total = uncached.len();
    eprintln!("[prewarm] starting Merkle root computation for {total} weight matrices");
    let t_start = std::time::Instant::now();

    // GPU contention guard: on single-GPU machines, the prover's pipelined
    // weight-commit loop also needs the CUDA executor.  Running prewarm on
    // GPU simultaneously would cause scheduler thrashing and kernel launch
    // contention. Use GPU prewarm only when >=2 GPUs are available (prewarm
    // takes a secondary device); otherwise fall back to CPU so the prover
    // gets exclusive GPU access.
    //
    // Exception: when force_gpu is true, the caller guarantees no concurrent
    // GPU usage (blocking prewarm before proving, or --generate-cache mode).
    #[cfg(feature = "cuda-runtime")]
    let computed = {
        let gpu_count = {
            #[cfg(feature = "multi-gpu")]
            { crate::multi_gpu::device_count() }
            #[cfg(not(feature = "multi-gpu"))]
            { 1usize }
        };
        if gpu_count >= 2 || force_gpu {
            if force_gpu && gpu_count < 2 {
                eprintln!("[prewarm] GPU exclusive mode — using single GPU (no contention)");
            }
            prewarm_gpu(weights, &uncached, cache, total, &t_start)
        } else {
            eprintln!("[prewarm] single-GPU detected — using CPU path to avoid GPU contention");
            prewarm_cpu(weights, &uncached, cache, total, &t_start)
        }
    };

    #[cfg(not(feature = "cuda-runtime"))]
    let computed = {
        let _ = force_gpu; // unused without cuda-runtime
        prewarm_cpu(weights, &uncached, cache, total, &t_start)
    };

    // Save cache to disk if we computed anything.
    if computed > 0 {
        if let Some(dir) = model_dir {
            match save_shared_cache(cache, dir) {
                Ok(true) => eprintln!("[prewarm] cache saved to disk ({computed} new roots)"),
                Ok(false) => {}
                Err(e) => eprintln!("[prewarm] warning: cache save failed: {e}"),
            }
        }
    }

    eprintln!(
        "[prewarm] completed {computed}/{total} roots in {:.1}s",
        t_start.elapsed().as_secs_f64(),
    );
    computed
}

#[cfg(feature = "cuda-runtime")]
fn prewarm_gpu(
    weights: &crate::compiler::graph::GraphWeights,
    uncached: &[(usize, usize, usize)],
    cache: &SharedWeightCache,
    total: usize,
    t_start: &std::time::Instant,
) -> usize {
    use stwo::prover::backend::gpu::cuda_executor::{
        get_cuda_executor, is_cuda_available, upload_poseidon252_round_constants,
    };

    if !is_cuda_available() || get_cuda_executor().is_err() {
        eprintln!("[prewarm] GPU unavailable, falling back to CPU");
        return prewarm_cpu(weights, uncached, cache, total, t_start);
    }

    let executor = match get_cuda_executor() {
        Ok(e) => e,
        Err(_) => return prewarm_cpu(weights, uncached, cache, total, t_start),
    };
    let d_rc = match upload_poseidon252_round_constants(&executor.device) {
        Ok(rc) => rc,
        Err(_) => return prewarm_cpu(weights, uncached, cache, total, t_start),
    };

    eprintln!("[prewarm] using GPU pipelined Merkle root computation");

    // Find max padded size for reusable limb buffers (double-buffer).
    let max_padded_n = uncached
        .iter()
        .map(|(_, rp, cp)| rp * cp)
        .max()
        .unwrap_or(0);
    let limb_buf_size = max_padded_n * 4;
    let mut limb_bufs = [vec![0u64; limb_buf_size], vec![0u64; limb_buf_size]];
    let mut cur_buf_idx: usize = 0;
    let mut computed = 0usize;

    // Prepare first matrix
    let (first_node_id, first_rp, first_cp) = uncached[0];
    let first_matrix = match weights.get_weight(first_node_id) {
        Some(m) => m,
        None => return 0,
    };
    let first_n = first_rp * first_cp;
    limb_bufs[0][..first_n * 4].fill(0);
    let _ = crate::components::matmul::matrix_to_mle_col_major_all_padded(
        first_matrix,
        &mut limb_bufs[0],
    );

    for idx in 0..uncached.len() {
        let (node_id, rp, cp) = uncached[idx];
        let cur_n = rp * cp;
        let next_buf_idx = 1 - cur_buf_idx;

        if idx + 1 < uncached.len() {
            let (next_node_id, next_rp, next_cp) = uncached[idx + 1];
            let next_n = next_rp * next_cp;

            let (buf_lo, buf_hi) = limb_bufs.split_at_mut(1);
            let (cur_buf, next_buf) = if cur_buf_idx == 0 {
                (&buf_lo[0][..], &mut buf_hi[0][..])
            } else {
                (&buf_hi[0][..], &mut buf_lo[0][..])
            };
            next_buf[..next_n * 4].fill(0);

            std::thread::scope(|s| {
                // CPU: prepare next matrix
                let cpu_handle = s.spawn(|| {
                    if let Some(m) = weights.get_weight(next_node_id) {
                        let _ = crate::components::matmul::matrix_to_mle_col_major_all_padded(
                            m, next_buf,
                        );
                    }
                });

                // GPU: commit current matrix
                if let Ok(root) = crate::crypto::mle_opening::commit_mle_root_only_gpu_from_limbs(
                    cur_buf, cur_n, executor, &d_rc,
                ) {
                    if let Ok(mut w) = cache.write() {
                        w.insert_root(node_id, rp, cp, root);
                        computed += 1;
                    }
                }

                cpu_handle.join().expect("prewarm CPU prep panicked");
            });
        } else {
            // Last matrix — no overlap
            if let Ok(root) = crate::crypto::mle_opening::commit_mle_root_only_gpu_from_limbs(
                &limb_bufs[cur_buf_idx], cur_n, executor, &d_rc,
            ) {
                if let Ok(mut w) = cache.write() {
                    w.insert_root(node_id, rp, cp, root);
                    computed += 1;
                }
            }
        }

        cur_buf_idx = next_buf_idx;

        let finished = idx + 1;
        if finished % 20 == 0 || finished == total {
            eprintln!(
                "[prewarm] {finished}/{total} ({:.1}s)",
                t_start.elapsed().as_secs_f64(),
            );
        }
    }
    computed
}

fn prewarm_cpu(
    weights: &crate::compiler::graph::GraphWeights,
    uncached: &[(usize, usize, usize)],
    cache: &SharedWeightCache,
    total: usize,
    t_start: &std::time::Instant,
) -> usize {
    eprintln!("[prewarm] using CPU parallel Merkle root computation");
    let mut computed = 0usize;

    for (idx, &(node_id, _rp, _cp)) in uncached.iter().enumerate() {
        let matrix = match weights.get_weight(node_id) {
            Some(m) => m,
            None => continue,
        };
        let rp = matrix.rows.next_power_of_two();
        let cp = matrix.cols.next_power_of_two();
        let mle = crate::components::matmul::matrix_to_mle_col_major_padded_pub(matrix);
        let root = crate::crypto::mle_opening::commit_mle_root_only(&mle);
        if let Ok(mut w) = cache.write() {
            w.insert_root(node_id, rp, cp, root);
            computed += 1;
        }

        let finished = idx + 1;
        if finished % 10 == 0 || finished == total {
            eprintln!(
                "[prewarm] {finished}/{total} ({:.1}s)",
                t_start.elapsed().as_secs_f64(),
            );
        }
    }
    computed
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

    #[test]
    fn test_root_cache_insert_and_get() {
        let mut cache = WeightCommitmentCache::new("root-test");
        let root = FieldElement::from(0xABCDu64);

        // Insert root for node 5, padded 16x32
        cache.insert_root(5, 16, 32, root);
        assert_eq!(cache.len(), 1);
        assert!(cache.is_dirty());

        // Hit
        let hit = cache.get_root(5, 16, 32);
        assert_eq!(hit, Some(root));

        // Miss: different node
        assert!(cache.get_root(6, 16, 32).is_none());
        // Miss: different dims
        assert!(cache.get_root(5, 32, 32).is_none());
    }

    #[test]
    fn test_root_cache_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("stwo_ml_root_cache_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("root_cache.swcf");

        let mut cache = WeightCommitmentCache::new("root-model");
        cache.insert_root(0, 8, 16, FieldElement::from(0x1111u64));
        cache.insert_root(5, 16, 32, FieldElement::from(0x2222u64));
        cache.insert_root(10, 32, 64, FieldElement::from(0x3333u64));
        cache.save(&path).unwrap();

        let loaded = WeightCommitmentCache::load(&path).unwrap();
        assert_eq!(loaded.get_root(0, 8, 16), Some(FieldElement::from(0x1111u64)));
        assert_eq!(loaded.get_root(5, 16, 32), Some(FieldElement::from(0x2222u64)));
        assert_eq!(loaded.get_root(10, 32, 64), Some(FieldElement::from(0x3333u64)));
        assert!(loaded.get_root(99, 8, 16).is_none());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_root_cache_does_not_collide_with_regular() {
        let mut cache = WeightCommitmentCache::new("mixed");

        // Insert regular entry at (node=0, m=4, k=8, n=16)
        cache.insert(0, 4, 8, 16, make_cached_weight(8, 3));

        // Insert root entry at (node=0, rows=8, cols=16) → uses m_padded=0
        cache.insert_root(0, 8, 16, FieldElement::from(0xFACEu64));

        // Both should exist independently
        assert_eq!(cache.len(), 2);
        assert!(cache.get(0, 4, 8, 16).is_some());
        assert_eq!(cache.get_root(0, 8, 16), Some(FieldElement::from(0xFACEu64)));

        // Root entry doesn't interfere with regular
        assert!(cache.get(0, 0, 8, 16).is_some()); // m_padded=0 is the root sentinel
        assert_eq!(cache.get(0, 0, 8, 16).unwrap().initial_mle_root, Some(FieldElement::from(0xFACEu64)));
    }

    #[test]
    fn test_fingerprint_deterministic() {
        use crate::components::matmul::M31Matrix;
        use crate::compiler::graph::GraphWeights;

        let w1 = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 2, cols: 4, data: (0..8).map(|i| M31(i * 100)).collect() }),
                (1, M31Matrix { rows: 4, cols: 2, data: (0..8).map(|i| M31(i * 200 + 1)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };

        let fp1 = compute_weight_fingerprint(&w1);
        let fp2 = compute_weight_fingerprint(&w1);
        assert_eq!(fp1, fp2, "fingerprint must be deterministic");
        assert_ne!(fp1, [0u8; 32], "fingerprint must not be zero");
    }

    #[test]
    fn test_fingerprint_changes_with_data() {
        use crate::components::matmul::M31Matrix;
        use crate::compiler::graph::GraphWeights;

        let w1 = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 2, cols: 4, data: (0..8).map(|i| M31(i * 100)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };
        let w2 = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 2, cols: 4, data: (0..8).map(|i| M31(i * 100 + 1)).collect() }), // different data
            ],
            biases: vec![],
            named_weights: vec![],
        };

        let fp1 = compute_weight_fingerprint(&w1);
        let fp2 = compute_weight_fingerprint(&w2);
        assert_ne!(fp1, fp2, "fingerprint must change when weight data changes");
    }

    #[test]
    fn test_fingerprint_save_load_roundtrip() {
        use crate::components::matmul::M31Matrix;
        use crate::compiler::graph::GraphWeights;

        let dir = std::env::temp_dir().join("stwo_ml_fp_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("fp_cache.swcf");

        let weights = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 2, cols: 4, data: (0..8).map(|i| M31(i * 100)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };

        // Create cache with fingerprint
        let mut cache = WeightCommitmentCache::new_with_fingerprint("fp-model", &weights);
        let fp = *cache.fingerprint();
        assert_ne!(fp, [0u8; 32]);

        cache.insert_root(0, 2, 4, FieldElement::from(0x1234u64));
        cache.save(&path).unwrap();

        // Load and verify fingerprint survives roundtrip
        let loaded = WeightCommitmentCache::load(&path).unwrap();
        assert_eq!(*loaded.fingerprint(), fp);
        assert!(loaded.validate_fingerprint(&weights));

        // Mutated weights should fail validation
        let mutated_weights = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 2, cols: 4, data: (0..8).map(|i| M31(i * 999)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };
        assert!(!loaded.validate_fingerprint(&mutated_weights));

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_load_or_new_validated_invalidates_on_weight_change() {
        use crate::components::matmul::M31Matrix;
        use crate::compiler::graph::GraphWeights;

        let dir = std::env::temp_dir().join("stwo_ml_validated_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("validated.swcf");

        let original_weights = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 2, cols: 4, data: (0..8).map(|i| M31(i * 100)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };

        // Create and save with original weights
        let mut cache = WeightCommitmentCache::new_with_fingerprint("val-model", &original_weights);
        cache.insert_root(0, 2, 4, FieldElement::from(0xAAAAu64));
        cache.save(&path).unwrap();
        assert_eq!(cache.len(), 1);

        // Load with same weights → should keep entries
        let reloaded = WeightCommitmentCache::load_or_new_validated(&path, "val-model", &original_weights);
        assert_eq!(reloaded.len(), 1);
        assert_eq!(reloaded.get_root(0, 2, 4), Some(FieldElement::from(0xAAAAu64)));

        // Load with different weights → should invalidate
        let new_weights = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 2, cols: 4, data: (0..8).map(|i| M31(i * 999)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };
        let invalidated = WeightCommitmentCache::load_or_new_validated(&path, "val-model", &new_weights);
        assert!(invalidated.is_empty(), "cache should be invalidated on weight change");
        assert_ne!(*invalidated.fingerprint(), [0u8; 32], "new cache should have fingerprint");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    // ── prewarm_weight_roots tests ──────────────────────────────────

    #[test]
    fn test_prewarm_cpu_computes_roots() {
        use crate::compiler::graph::GraphWeights;
        use crate::components::matmul::M31Matrix;

        let weights = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 4, cols: 4, data: (0..16).map(|i| M31(i + 1)).collect() }),
                (1, M31Matrix { rows: 2, cols: 8, data: (0..16).map(|i| M31(i * 3 + 1)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };
        let cache = shared_cache("prewarm-test");

        // Initially empty
        assert_eq!(cache.read().unwrap().len(), 0);

        // Prewarm without disk save (no model_dir)
        let count = prewarm_weight_roots(&weights, &cache, None);
        assert_eq!(count, 2, "should compute roots for both matrices");

        // Cache should now have 2 root entries
        let r = cache.read().unwrap();
        assert!(r.get_root(0, 4, 4).is_some(), "missing root for node 0");
        assert!(r.get_root(1, 2, 8).is_some(), "missing root for node 1");
    }

    #[test]
    fn test_prewarm_skips_already_cached() {
        use crate::compiler::graph::GraphWeights;
        use crate::components::matmul::M31Matrix;

        let weights = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 4, cols: 4, data: (0..16).map(|i| M31(i + 1)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };
        let cache = shared_cache("prewarm-skip-test");

        // Pre-insert a fake root
        cache.write().unwrap().insert_root(0, 4, 4, FieldElement::from(0xBEEFu64));

        // Prewarm should find all cached
        let count = prewarm_weight_roots(&weights, &cache, None);
        assert_eq!(count, 0, "should skip already-cached matrix");

        // Original fake root should be unchanged
        let r = cache.read().unwrap();
        assert_eq!(r.get_root(0, 4, 4), Some(FieldElement::from(0xBEEFu64)));
    }

    #[test]
    fn test_prewarm_roots_match_commit_mle_root_only() {
        use crate::compiler::graph::GraphWeights;
        use crate::components::matmul::M31Matrix;

        let weights = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 4, cols: 8, data: (0..32).map(|i| M31(i * 7 + 3)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };

        // Compute root via prewarm
        let cache = shared_cache("prewarm-match-test");
        prewarm_weight_roots(&weights, &cache, None);
        let prewarm_root = cache.read().unwrap().get_root(0, 4, 8).unwrap();

        // Compute root directly for comparison
        let mle = crate::components::matmul::matrix_to_mle_col_major_padded_pub(
            weights.get_weight(0).unwrap(),
        );
        let direct_root = crate::crypto::mle_opening::commit_mle_root_only(&mle);

        assert_eq!(prewarm_root, direct_root, "prewarm root must match direct computation");
    }

    #[test]
    fn test_prewarm_saves_to_disk() {
        use crate::compiler::graph::GraphWeights;
        use crate::components::matmul::M31Matrix;

        let dir = std::env::temp_dir().join("prewarm_disk_test");
        let _ = std::fs::create_dir_all(&dir);

        let weights = GraphWeights {
            weights: vec![
                (0, M31Matrix { rows: 2, cols: 4, data: (0..8).map(|i| M31(i + 1)).collect() }),
            ],
            biases: vec![],
            named_weights: vec![],
        };
        let cache = shared_cache("disk-test");
        prewarm_weight_roots(&weights, &cache, Some(&dir));

        // Cache file should exist
        let cache_path = dir.join(".stwo_weight_cache.swcf");
        assert!(cache_path.exists(), "cache file should be saved");

        // Reload and verify
        let loaded = shared_cache_from_file(&cache_path, "disk-test");
        let r = loaded.read().unwrap();
        assert!(r.get_root(0, 2, 4).is_some(), "loaded cache should contain root");

        let _ = std::fs::remove_file(&cache_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
