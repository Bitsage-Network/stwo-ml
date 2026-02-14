# Tile-Level Streaming Architecture

> Double-buffered tile pipeline with precomputed matmul injection. Eliminates weight re-loading and overlaps I/O with proving.

## Problem

The original `prove_model_chunked_streaming_tiled` had a critical inefficiency. While Phases 1 and 2 used tile-level streaming (loading weight tiles on demand from mmap), Phase 3 fell back to `pipeline.load_chunk_weights()` and called the monolithic `prove_model_aggregated_onchain`. This re-loaded full weight matrices and re-proved all matmuls — negating the tile-level memory savings.

```text
BEFORE (wasteful):
  Phase 1: tile-by-tile forward pass    → A, C per matmul  (good)
  Phase 2: tile-by-tile proving          → TiledProofs      (good)
  Phase 3: load_chunk_weights()          → full B matrices   (BAD — re-loads everything)
           prove_model_aggregated_onchain → re-proves matmuls (BAD — duplicated work)
```

## Solution: PrecomputedMatmuls Injection

The aggregation pipeline (`prove_model_aggregated_onchain_with_cache`) was refactored to accept an optional `PrecomputedMatmuls` struct that provides pre-computed matmul outputs and proofs. When present:

- **Phase 1** (forward pass): Uses `precomputed.outputs[node_id]` instead of `weights.get_weight() + matmul_m31()`.
- **Phase 2** (proving): Skipped entirely — proofs come from `precomputed.proofs` and `precomputed.tiled_proofs`.
- **Phase 3** (STARK): Runs normally for non-matmul components (activations, add, mul, layernorm, etc.).

```text
AFTER (efficient):
  Phase 1: tile-by-tile forward pass    → A, C per matmul
  Phase 2: tile-by-tile proving          → TiledProofs
  Phase 3: PrecomputedMatmuls injected   → NO weight loading
           aggregation pipeline           → NO matmul re-proving
                                          → Only STARK for non-matmul components
```

## Key Types

### `PrecomputedMatmuls` (aggregation.rs)

```rust
pub(crate) struct PrecomputedMatmuls {
    /// Pre-computed matmul output matrices (node_id -> C matrix).
    /// Phase 1 uses these instead of weights.get_weight() + matmul_m31().
    pub outputs: HashMap<usize, M31Matrix>,

    /// Pre-composed single-tile matmul proofs (node_id, proof).
    /// Phase 2 is skipped entirely for these.
    pub proofs: Vec<(usize, MatMulSumcheckProofOnChain)>,

    /// Multi-tile proofs that can't be composed into a single proof.
    pub tiled_proofs: Vec<(usize, TiledMatMulProof)>,
}
```

### `AggregatedModelProofOnChain` — new field

```rust
pub struct AggregatedModelProofOnChain {
    // ... existing fields ...

    /// Multi-tile matmul proofs that couldn't be composed into a single proof.
    /// Present when tile-level streaming is used with multi-tile matmuls.
    pub tiled_matmul_proofs: Vec<(usize, TiledMatMulProof)>,

    // ... existing fields ...
}
```

## API

### `prove_model_aggregated_onchain_with_precomputed`

```rust
pub(crate) fn prove_model_aggregated_onchain_with_precomputed(
    graph: &ComputationGraph,
    input: &M31Matrix,
    weights: &GraphWeights,  // Can be empty — weights not needed
    precomputed: PrecomputedMatmuls,
) -> Result<AggregatedModelProofOnChain, AggregationError>
```

Thin wrapper around `prove_model_aggregated_onchain_with_cache` that passes `Some(precomputed)`.

## Data Flow

```text
prove_model_chunked_streaming_tiled:

  ┌─────────────────────────────────────────────────────────┐
  │ Phase 1: Forward Pass (tile-level streaming)            │
  │                                                         │
  │   for each matmul node:                                 │
  │     pipeline.forward_matmul_tiled(node_id, A, config)   │
  │       → double-buffered: loads tile N+1 while computing │
  │         matmul for tile N (std::thread::scope)          │
  │       → accumulates C = A × B tile by tile              │
  │       → peak mem: 2 × tile_k × n (current + next tile) │
  │     stores (node_id, A, C) in chunk_matmul_data         │
  └─────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Phase 2: Prove (tile-level streaming)                   │
  │                                                         │
  │   for each matmul:                                      │
  │     pipeline.prove_matmul_tiled_streaming(...)           │
  │       → double-buffered: loads tile N+1 while proving   │
  │         tile N via sumcheck (std::thread::scope)        │
  │       → verify_tiled_matmul() sanity check              │
  │     stores (node_id, TiledMatMulProof)                  │
  └─────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Phase 3: Aggregate (precomputed injection)              │
  │                                                         │
  │   Build PrecomputedMatmuls:                             │
  │     outputs: node_id → C from Phase 1                   │
  │     proofs: single-tile → compose_tiled_proof()         │
  │     tiled_proofs: multi-tile (passed through as-is)     │
  │                                                         │
  │   prove_model_aggregated_onchain_with_precomputed(      │
  │     graph, input, GraphWeights::new(), precomputed      │
  │   )                                                     │
  │     → Phase 1: uses precomputed C (no weight lookup)    │
  │     → Phase 2: uses precomputed proofs (no re-proving)  │
  │     → Phase 3: builds STARK for activations/add/mul/... │
  └─────────────────────────────────────────────────────────┘
```

## Double-Buffered Tile Pipeline

Both `forward_matmul_tiled` and `prove_matmul_tiled_streaming` use a double-buffered pipeline that overlaps tile loading with computation using `std::thread::scope`.

### Timeline

```text
Sequential (before):
  [load tile 0] [compute tile 0] [load tile 1] [compute tile 1] [load tile 2] ...
  |----- T0 ----|----- T1 -------|----- T2 ----|----- T3 -------|----- T4 ----|

Double-buffered (after):
  [load tile 0] [compute tile 0 ||||| load tile 1] [compute tile 1 ||||| load tile 2] ...
  |----- T0 ----|------------ T1 ----------------|-----------  T2 -------------------|
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 I/O hidden behind compute — wall time reduced by load latency
```

### How It Works

1. **Tile 0**: Loaded synchronously (nothing to overlap with yet).
2. **Tiles 1..N**: `std::thread::scope` spawns a loader thread for tile N+1 while the main thread computes/proves tile N. The loader performs mmap read + f32 conversion + M31 quantization.
3. **Single-tile case**: Falls through to a simple non-pipelined path (no thread overhead for 1 tile).

### Implementation (streaming.rs)

```rust
// forward_matmul_tiled — overlaps matmul with next tile load
let (c_tile, next_tile_result) = std::thread::scope(|s| {
    let loader = s.spawn(|| {
        self.load_weight_tile(node_id, next_k_start, next_k_end, &params)
    });
    let c_tile = matmul_m31(&a_tile, &b_tile);
    let next = loader.join().expect("loader thread panicked");
    (c_tile, Some(next))
});

// prove_matmul_tiled_streaming — overlaps sumcheck proving with next tile load
let (proof_result, next_tile_result) = std::thread::scope(|s| {
    let loader = s.spawn(|| {
        self.load_weight_tile(node_id, next_k_start, next_k_end, &params)
    });
    let proof = prove_matmul_sumcheck_onchain_auto(&a_padded, &b_padded, &c_padded);
    let next = loader.join().expect("loader thread panicked");
    (proof, next)
});
```

### Why `std::thread::scope`

- **Borrows `&self`**: The loader thread needs `&self` to call `load_weight_tile`. Scoped threads allow borrowing from the enclosing stack frame without `'static` bounds.
- **No `Arc`/`Mutex` overhead**: The pipeline reference is shared read-only (mmap is `Sync`).
- **Panic safety**: If the loader panics, `join()` propagates it cleanly. The scoped thread is always joined before the scope exits.

### Latency Hiding

The `load_weight_tile` call performs:
1. mmap page fault (kernel I/O, ~0.5-2ms per tile for NVMe)
2. f32 conversion from dtype (bf16/f16 → f32, ~0.1ms per tile)
3. M31 quantization (~0.2ms per tile)

For the proving path, `prove_matmul_sumcheck_onchain_auto` takes 50-500ms per tile (depending on dimensions). The ~1-3ms of tile loading is completely hidden behind the proving time — effectively free I/O.

For the forward path, `matmul_m31` takes 1-50ms per tile. The loading is partially or fully hidden depending on tile size.

## Memory Impact

For a matmul with dimensions `m × k × n` and tile size `tile_k`:

| Stage | Before | After |
|-------|--------|-------|
| Forward pass | `k × n × 4` bytes (full weight) | `tile_k × n × 4` bytes (1 tile) |
| Proving | `k × n × 4` bytes (full weight, loaded twice) | `tile_k × n × 4` bytes (1 tile) |
| Aggregation | `k × n × 4` bytes (full weight, loaded AGAIN) | **0 bytes** (precomputed) |

Example: Qwen3-14B matmul 5120×14336:
- Full weight: 280 MB per matrix
- Tile (tile_k=1024): 56 MB per tile
- Aggregation weight load eliminated: **280 MB saved per matmul**
- For 160 matmuls in a transformer block: **~44 GB of redundant weight I/O eliminated**

## Proof Composition

Single-tile proofs (where `TiledMatMulProof.tile_proofs.len() == 1`) are composed into standard `MatMulSumcheckProofOnChain` via `compose_tiled_proof()`. This makes them indistinguishable from non-tiled proofs in the final `AggregatedModelProofOnChain`.

Multi-tile proofs (2+ tiles) cannot be composed into a single sumcheck proof because the Fiat-Shamir challenges differ per tile. These are stored in the `tiled_matmul_proofs` field for separate verification.

## Files Modified

| File | Changes |
|------|---------|
| `src/aggregation.rs` | `PrecomputedMatmuls` struct, `tiled_matmul_proofs` field on `AggregatedModelProofOnChain`, precomputed injection in `_with_cache`, `prove_model_aggregated_onchain_with_precomputed` wrapper |
| `src/compiler/chunked.rs` | Rewrote Phase 3 of `prove_model_chunked_streaming_tiled` — builds `PrecomputedMatmuls`, calls `_with_precomputed`, no more `load_chunk_weights` |
| `src/compiler/streaming.rs` | Double-buffered `forward_matmul_tiled` and `prove_matmul_tiled_streaming` — background thread loads tile N+1 while main thread computes/proves tile N |
| `src/cairo_serde.rs` | Added `tiled_matmul_proofs: Vec::new()` to all 7 test struct literals |

## Backward Compatibility

All existing callers pass `None` for the new `precomputed` parameter, making this fully backward compatible. The new `tiled_matmul_proofs` field on `AggregatedModelProofOnChain` defaults to `Vec::new()` in all non-streaming paths.
