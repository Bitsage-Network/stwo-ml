# Configuration Reference

## Policy Presets (Recommended)

Instead of setting individual environment variables, use the `--policy` flag to select a preset that configures all proving options at once. The policy is cryptographically bound to the proof via its Poseidon commitment hash.

```bash
# Production: all soundness gates enforced, full weight binding
prove-model --model-dir ./qwen --gkr --policy strict

# On-chain streaming: default, matches prove-server behavior
prove-model --model-dir ./qwen --gkr --policy standard

# Development: all gates permissive, fastest proving
prove-model --model-dir ./qwen --gkr --policy relaxed
```

### What Each Preset Does

| Setting | strict | standard | relaxed |
|---------|--------|----------|---------|
| Missing norm proofs | reject | allow | allow |
| LogUp activation | reject | allow | allow |
| Missing segment binding | reject | allow | allow |
| RMS Part 0 skip | no | yes | yes |
| Piecewise activation | yes | no | yes |
| Batch token skip | no | yes | yes |
| Unified STARK skip | no | yes | yes |
| Weight binding | aggregated + full | aggregated + full | aggregated |
| IO packing | yes | yes | yes |
| Packed proof | yes | yes | yes |
| Decode chain validation | enforced | enforced | off |

### Policy Commitment

Every proof carries a `policy_commitment` — a Poseidon hash of the policy configuration mixed into the Fiat-Shamir channel:

```
strict:   0x0370c9348ed6edddf310baf5d8104d57c07f36962deea9738dd00519d9948449
standard: 0x05baf1be3d54bcd383072f79923316ac7124670a117bd5c809b67b651209424b
relaxed:  0x02fba808267ad15ef03f2db8ac9a09a87194ea32edb5aa41333976ac4425d06c
```

This means:

- The on-chain verifier can check which policy was used
- Proofs generated under different policies produce different commitments
- You cannot retroactively change which policy a proof was generated under

### Custom Policies

For advanced use cases, create a JSON policy file:

```json
{
  "allow_missing_norm_proof": false,
  "allow_logup_activation": false,
  "allow_missing_segment_binding": false,
  "skip_rms_sq_proof": false,
  "piecewise_activation": true,
  "skip_batch_tokens": false,
  "skip_unified_stark": false,
  "weight_binding_mode": "Aggregated",
  "aggregated_full_binding": true,
  "aggregated_rlc_only": false,
  "io_packing": true,
  "packed_proof": true,
  "double_packed_proof": true,
  "validate_decode_chain": true
}
```

```bash
prove-model --model-dir ./qwen --gkr --policy-file my_policy.json
```

### Precedence

When multiple configuration sources exist:

```
--policy-file flag  >  --policy flag  >  STWO_POLICY env  >  individual STWO_* env vars  >  defaults
```

If you pass `--policy strict` but also have `STWO_ALLOW_MISSING_NORM_PROOF=1` set, the CLI will warn:

```
Note: --policy strict overrides 1 STWO_* env var(s): STWO_ALLOW_MISSING_NORM_PROOF
```

### API Usage

When using the prove-server HTTP API, pass `"policy"` in the request body:

```json
POST /api/v1/prove
{
  "model_id": "smollm2-135m",
  "input": [...],
  "policy": "strict"
}
```

The response includes the policy used and its commitment:

```json
{
  "policy": "strict",
  "policy_commitment": "0x074043587ac5abf3...",
  "calldata": [...],
  "io_commitment": "0x..."
}
```

---

## Environment Variables (Legacy)

Individual `STWO_*` variables are still supported for backward compatibility. When no `--policy` flag is set, these variables control proving behavior directly.

### STWO_POLICY

Set this to a preset name to configure all proving options at once (equivalent to `--policy`):

```bash
export STWO_POLICY=strict
prove-model --model-dir ./qwen --gkr   # uses strict policy
```

### Proving

| Variable | Default | PolicyConfig Field | Description |
|----------|---------|-------------------|-------------|
| `STWO_WEIGHT_BINDING` | `aggregated` | `weight_binding_mode` | Weight binding mode: `aggregated`, `individual`, `sequential`. |
| `STWO_AGGREGATED_FULL_BINDING` | `0` | `aggregated_full_binding` | Full MLE opening proofs for trustless on-chain streaming. |
| `STWO_AGGREGATED_RLC_ONLY` | `0` | `aggregated_rlc_only` | Force RLC-only binding (weaker, for debugging). |
| `STWO_ALLOW_MISSING_NORM_PROOF` | `0` | `allow_missing_norm_proof` | Allow proofs with missing LayerNorm/RMSNorm sub-proofs. |
| `STWO_ALLOW_LOGUP_ACTIVATION` | `0` | `allow_logup_activation` | Allow reduced-precision LogUp activation proofs. |
| `STWO_ALLOW_MISSING_SEGMENT_BINDING` | `0` | `allow_missing_segment_binding` | Allow missing piecewise segment binding. |
| `STWO_SKIP_RMS_SQ_PROOF` | unset | `skip_rms_sq_proof` | Skip RMSNorm Part 0 self-verification (Cairo handles it). |
| `STWO_PIECEWISE_ACTIVATION` | `1` | `piecewise_activation` | Use piecewise algebraic activation proofs. Set to `0` to disable. |
| `STWO_SKIP_BATCH_TOKENS` | `0` | `skip_batch_tokens` | Skip batch token accumulation proofs. |
| `STWO_PURE_GKR_SKIP_UNIFIED_STARK` | `0` | `skip_unified_stark` | Skip unified STARK layer in pure-GKR mode. |
| `STWO_FORCE_STREAMING` | `0` | — | Force streaming GKR calldata even when single-TX fits. |

### Calldata

| Variable | Default | PolicyConfig Field | Description |
|----------|---------|-------------------|-------------|
| `STWO_NO_IO_PACK` | unset | `io_packing` (negated) | Disable IO packing (use raw M31 values). |
| `STWO_NO_PACKED` | unset | `packed_proof` (negated) | Disable packed calldata mode. |
| `STWO_NO_DOUBLE_PACK` | unset | `double_packed_proof` (negated) | Disable double-packed output format. |

### GPU

These are not part of PolicyConfig (they affect performance, not soundness):

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_GPU_MERKLE_THRESHOLD` | `4096` | Minimum tree size for GPU Merkle acceleration. |
| `STWO_GPU_MLE_FOLD` | unset | GPU MLE fold backend. Set to `cpu` for CPU fallback. |
| `STWO_GPU_ONLY` | `0` | Fail instead of CPU fallback for GPU operations. |
| `STWO_GPU_MLE_OPENING_TIMING` | `0` | Print per-layer GPU MLE opening timing. |
| `STWO_PARALLEL_GPU_COMMIT` | `0` | Overlap weight commitment with proving on GPU. |
| `STWO_GPU_COMMIT_STRICT` | `0` | Panic instead of CPU fallback on GPU commit failure. |
| `STWO_GPU_COMMIT_HARDEN` | `0` | Cross-check GPU commitments against CPU. |

### Caching

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_MERKLE_TREE_CACHE_DIR` | unset | Directory for persistent Merkle tree caches. |
| `STWO_WEIGHT_PROGRESS_EVERY` | `1` | Print weight commitment progress every N matrices. |
| `STWO_WEIGHT_COMMIT_SEGMENTS` | `4096` | Parallel segments per weight matrix during commitment. |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_CHANNEL_TRACE` | `0` | Print Fiat-Shamir channel operations. |
| `STWO_LOG_SUMCHECK` | `0` | Log sumcheck round details. |
| `STWO_PROFILE` | `0` | Enable per-phase timing breakdown. |

### MLE Opening

| Variable | Default | Description |
|----------|---------|-------------|
| `STWO_MLE_N_QUERIES` | `3` | Number of MLE opening queries. Must match between prover and verifier. |

### External

| Variable | Description |
|----------|-------------|
| `STARKNET_ACCOUNT` | Starknet account address for on-chain submission. |
| `STARKNET_PRIVATE_KEY` | Private key for signing Starknet transactions. Required for `--on-chain` flag. |
| `STARKNET_RPC` | Starknet RPC endpoint URL. Defaults to Alchemy Sepolia. Do not hardcode API keys -- use env vars. |
| `RECURSIVE_CONTRACT` | Recursive verifier contract address. Defaults to trustless verifier on Sepolia (`0x1c208a5...`). |
| `OBELYSK_RECURSIVE_SCRIPT` | Path to `submit_recursive.mjs`. The CLI auto-detects if not set. |
| `CONTRACT_ADDRESS` | Streaming GKR verifier contract address. |
| `SESSION_ID` | Session ID for resuming streaming submission. |
| `AVNU_API_KEY` | API key for AVNU paymaster (gasless transactions). |
| `PROVE_SERVER_API_KEY` | Bearer token for prove-server authentication. |
| `PROVE_SERVER_MODEL_DIR` | Allowlist directory for model loading in prove-server. |
| `PROVE_SERVER_RATE_LIMIT` | Requests per minute (default 60). |
