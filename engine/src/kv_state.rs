//! KV-cache state serialization for incremental decode proving.
//!
//! Provides `KVCacheState` which can serialize a live `ModelKVCache` +
//! `IncrementalKVCommitment` pair to JSON and restore them. M31 values are
//! stored as `u32`; the Merkle commitment trees are rebuilt from the KV data
//! on load (since `IncrementalPoseidonMerkle` fields are private).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use stwo::core::fields::m31::M31;

use crate::aggregation::IncrementalKVCommitment;
use crate::components::attention::{KVCache, ModelKVCache};
use crate::components::matmul::M31Matrix;

/// Serializable snapshot of KV-cache state + incremental commitment metadata.
#[derive(Serialize, Deserialize)]
pub struct KVCacheState {
    pub layers: BTreeMap<usize, LayerKVState>,
    /// Capacity hint used when rebuilding Merkle trees on load.
    pub merkle_capacity: usize,
}

/// Per-layer KV cache data in serializable form.
#[derive(Serialize, Deserialize)]
pub struct LayerKVState {
    /// Per-head K cache: each inner Vec is a flattened (cached_len × d_k) matrix as u32.
    pub k_cache: Vec<Vec<u32>>,
    /// Per-head V cache: same layout.
    pub v_cache: Vec<Vec<u32>>,
    pub cached_len: usize,
    pub num_kv_heads: usize,
    pub d_k: usize,
}

impl KVCacheState {
    /// Snapshot live KV cache and commitment into a serializable form.
    pub fn from_live(
        kv_cache: &ModelKVCache,
        _commitment: &IncrementalKVCommitment,
    ) -> Self {
        let mut layers = BTreeMap::new();
        let mut max_len = 0usize;

        for (&layer_id, cache) in &kv_cache.layers {
            let mut k_data = Vec::with_capacity(cache.num_kv_heads);
            let mut v_data = Vec::with_capacity(cache.num_kv_heads);

            for h in 0..cache.num_kv_heads {
                let k = &cache.k_cache[h];
                let v = &cache.v_cache[h];
                let mut k_vals = Vec::with_capacity(cache.cached_len * cache.d_k);
                let mut v_vals = Vec::with_capacity(cache.cached_len * cache.d_k);
                for row in 0..cache.cached_len {
                    for col in 0..cache.d_k {
                        k_vals.push(k.get(row, col).0);
                        v_vals.push(v.get(row, col).0);
                    }
                }
                k_data.push(k_vals);
                v_data.push(v_vals);
            }

            if cache.cached_len > max_len {
                max_len = cache.cached_len;
            }

            layers.insert(layer_id, LayerKVState {
                k_cache: k_data,
                v_cache: v_data,
                cached_len: cache.cached_len,
                num_kv_heads: cache.num_kv_heads,
                d_k: cache.d_k,
            });
        }

        Self {
            layers,
            merkle_capacity: max_len.next_power_of_two().max(16),
        }
    }

    /// Restore live KV cache and rebuild the incremental Merkle commitment.
    ///
    /// The Merkle trees are rebuilt from scratch (O(N) per layer) since
    /// `IncrementalPoseidonMerkle` fields are private.
    pub fn to_live(&self) -> (ModelKVCache, IncrementalKVCommitment) {
        let mut model_cache = ModelKVCache::new();

        for (&layer_id, layer) in &self.layers {
            // Reconstruct per-head M31Matrix directly from serialized u32 data
            let k_cache: Vec<M31Matrix> = layer.k_cache.iter().map(|k_vals| {
                let mut m = M31Matrix::new(layer.cached_len, layer.d_k);
                for (i, &v) in k_vals.iter().enumerate() {
                    m.data[i] = M31::from(v);
                }
                m
            }).collect();

            let v_cache: Vec<M31Matrix> = layer.v_cache.iter().map(|v_vals| {
                let mut m = M31Matrix::new(layer.cached_len, layer.d_k);
                for (i, &v) in v_vals.iter().enumerate() {
                    m.data[i] = M31::from(v);
                }
                m
            }).collect();

            let cache = KVCache {
                k_cache,
                v_cache,
                cached_len: layer.cached_len,
                num_kv_heads: layer.num_kv_heads,
                d_k: layer.d_k,
            };
            model_cache.layers.insert(layer_id, cache);
        }

        // Rebuild Merkle commitment from the restored KV cache
        let commitment = IncrementalKVCommitment::from_kv_cache(
            &model_cache,
            self.merkle_capacity,
        );

        (model_cache, commitment)
    }

    /// Save to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load from a JSON file.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
