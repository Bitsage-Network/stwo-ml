# Environment Variables Reference

All `STWO_*` environment variables that control proving, GPU, calldata, caching, and debug behavior.

## Proving

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_WEIGHT_BINDING` | `aggregated` | Weight binding mode. Set to `individual`, `sequential`, `0`, `off`, or `false` to opt out of aggregated oracle sumcheck. |
| `STWO_AGGREGATED_FULL_BINDING` | `1` | Enable full aggregated binding verification (default). Ensures weight commitments are independently verified on-chain. |
| `STWO_AGGREGATED_RLC_ONLY` | `0` | When `1`, use RLC-only binding (no full aggregated proof). Not accepted for streaming submission. |
| `STWO_FORCE_STREAMING` | `0` | Force streaming GKR calldata layout even when single-TX would fit. |
| `STWO_STARKNET_GKR_V4` | `1` | Use V4 on-chain entrypoint (packed IO, aggregated openings). |

## GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_GPU_MERKLE_THRESHOLD` | `4096` | Minimum tree size (leaves) before GPU Merkle acceleration kicks in. Lower values increase GPU usage but may cause OOM. |
| `STWO_GPU_MLE_FOLD` | unset | Control GPU MLE fold backend. Set to `cpu` to force CPU fallback. |
| `STWO_GPU_ONLY` | `0` | When `1`, fail instead of falling back to CPU for GPU operations. |
| `STWO_GPU_MLE_OPENING_TIMING` | `0` | When `1`, print per-layer GPU MLE opening timing. |
| `STWO_PARALLEL_GPU_COMMIT` | `0` | When `1`, overlap weight commitment computation with proving on GPU. May cause contention on single-GPU machines. |
| `STWO_GPU_COMMIT_STRICT` | `0` | When `1`, panic instead of CPU-fallback if GPU commitment fails. |
| `STWO_GPU_COMMIT_HARDEN` | `0` | When `1`, cross-check GPU commitment results against CPU. Slower but catches GPU bugs. |

## Calldata

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_NO_IO_PACK` | `0` | When `1`, disable IO packing (use raw M31 values instead of 8-per-felt). |
| `STWO_NO_PACKED` | `0` | When `1`, disable packed calldata mode. |

## Caching

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_MERKLE_TREE_CACHE_DIR` | unset | Directory for persistent Merkle tree caches. If unset, uses model directory. |
| `STWO_WEIGHT_PROGRESS_EVERY` | `1` | Print weight commitment progress every N matrices. |
| `STWO_WEIGHT_COMMIT_SEGMENTS` | `4096` | Number of parallel segments per weight matrix during commitment. |

## Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_CHANNEL_TRACE` | `0` | When `1`, print Fiat-Shamir channel operations for debugging transcript mismatches. |
| `STWO_LOG_SUMCHECK` | `0` | When `1`, log sumcheck round details. |

## MLE Opening

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_MLE_N_QUERIES` | `3` | Number of MLE opening queries. Cairo reads this from the proof, so both sides must agree. |

## External (non-STWO)

| Variable | Description |
|----------|-------------|
| `STARKNET_ACCOUNT` | Starknet account address for on-chain submission. Used by pipeline scripts. |
| `STARKNET_PRIVATE_KEY` | Private key for signing Starknet transactions. |
| `STARKNET_RPC` | Starknet RPC endpoint URL. Defaults to Alchemy Sepolia. |
| `CONTRACT_ADDRESS` | GKR verifier contract address. Defaults to `0x0121d1...`. |
| `SESSION_ID` | Session ID for resuming streaming submission with `--skip-session`. |
| `AVNU_API_KEY` | API key for AVNU paymaster (gasless transactions). |
| `NEXT_PUBLIC_PROVER_WS_URL` | WebSocket URL for the marketplace to connect to the prove-server. |
