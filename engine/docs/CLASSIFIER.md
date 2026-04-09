# ZKML Transaction Classifier

Verifiable AI guardrails for autonomous agent transactions. A small MLP classifier scores transaction intent, and stwo-ml proves the inference ran correctly. The proof verifies on-chain — no oracle, no relay, no trust assumptions.

## How It Works

```
Agent wants to execute a transaction
       |
  1. ENCODE     Transaction features → 64 M31 values
       |
  2. CLASSIFY   MLP forward pass (64→64→32→3)
       |
  3. PROVE      GKR + STARK proof (~10s debug, <1s release)
       |
  4. SUBMIT     6 streaming TXs to Starknet verifier
       |
  5. RESOLVE    Firewall reads proven score → approve / escalate / block
```

## Quick Start

### Rust

```rust
use stwo_ml::classifier::*;
use stwo_ml::policy::PolicyConfig;

// Load classifier (test model with deterministic weights)
let model = build_test_classifier();
let policy = PolicyConfig::strict();

// Describe the transaction
let tx = TransactionFeatures {
    target: FieldElement::from(0x1234u64),
    value: [0, 1_000_000_000],          // 1 ETH in wei
    selector: 0xa9059cbb,                // ERC20 transfer(address,uint256)
    calldata_prefix: [0x5678, 0, 0, 0, 0, 0, 0, 0],
    calldata_len: 68,
    agent_trust_score: 15000,            // current on-chain score
    agent_strikes: 0,
    agent_age_blocks: 50000,
    target_flags: TargetFlags {
        is_verified: true,
        is_proxy: false,
        has_source: true,
        interaction_count: 200,
    },
    value_features: ValueFeatures {
        log2_value: 60,                  // log2(1e18)
        value_balance_ratio: 5000,       // 5% of balance
        is_max_approval: false,
        is_zero_value: false,
    },
    selector_features: SelectorFeatures {
        is_transfer: true,
        is_approve: false,
        is_swap: false,
        is_unknown: false,
    },
    behavioral: BehavioralFeatures {
        tx_frequency: 10,
        unique_targets_24h: 5,
        avg_value_24h: 500_000,
        max_value_24h: 2_000_000,
    },
};

// Evaluate: encode → classify → prove → score
let result = evaluate_transaction(&tx, &model, &policy)?;

println!("Decision: {}", result.decision);         // "approve"
println!("Threat score: {}", result.threat_score);  // 0-100000
println!("Scores: {:?}", result.scores);            // [safe, suspicious, malicious]
println!("IO commitment: {:#x}", result.io_commitment);
println!("Policy: {:#x}", result.policy_commitment);
```

### API (coming soon)

```bash
curl -X POST https://prover.bitsage.network/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "target": "0x1234...",
    "value": "1000000000",
    "selector": "0xa9059cbb",
    "calldata": "0x5678...",
    "agent_id": "my-agent",
    "policy": "strict"
  }'

# Response:
# {
#   "decision": "approve",
#   "threat_score": 12500,
#   "scores": [85000, 10000, 5000],
#   "proof_hash": "0x...",
#   "io_commitment": "0x...",
#   "policy_commitment": "0x0370c934...",
#   "prove_time_ms": 450
# }
```

### SDK (coming soon)

```typescript
import { AgentFirewallSDK } from '@obelyzk/firewall-sdk'

const firewall = new AgentFirewallSDK({
  proverUrl: 'https://prover.bitsage.network',
  firewallContract: '0x...',
  verifierContract: '0x0121d1...',
  rpcUrl: process.env.STARKNET_RPC,
  account: myAccount,
})

const result = await firewall.evaluateAction({
  agentId: 'my-agent',
  target: '0x1234...',
  value: '1000000000',
  data: '0xa9059cbb...',
})

if (result.decision === 'approve') {
  // safe to execute
  await executeTransaction(...)
} else if (result.decision === 'escalate') {
  // needs human review
  await notifyOperator(result)
} else {
  // blocked — probable attack
  console.error(`Blocked: score ${result.threatScore}`)
}
```

## Architecture

### Model: 64→64→32→3 MLP

| Layer | Type | Input | Output | Params |
|-------|------|-------|--------|--------|
| 0 | Linear | 64 | 64 | 4,096 |
| 1 | ReLU | 64 | 64 | 0 |
| 2 | Linear | 64 | 32 | 2,048 |
| 3 | ReLU | 32 | 32 | 0 |
| 4 | Linear | 32 | 3 | 96 |
| **Total** | | | | **6,240** |

Why this size:
- 64 input: power-of-2 for efficient MLE evaluation in GKR
- 6K params: proves in <1s (release build), 100,000x smaller than Qwen3-14B
- 3 outputs: safe / suspicious / malicious (maps to approve / escalate / block)
- ReLU: simplest activation to prove (LogUp lookup table)

### Input Encoding (48 features → 64 with padding)

| Index | Feature | Encoding |
|-------|---------|----------|
| 0-7 | Target address | 8 x 31-bit chunks from felt252 |
| 8-15 | Value (u256) | 8 x 31-bit chunks |
| 16 | Function selector | u32 as M31 |
| 17-24 | Calldata prefix | First 8 words as M31 |
| 25 | Calldata length | u32 |
| 26-28 | Agent metadata | trust_score, strikes, age |
| 29-32 | Target flags | is_verified, is_proxy, has_source, interactions |
| 33-36 | Value features | log2, ratio, is_max_approval, is_zero |
| 37-40 | Selector features | is_transfer, is_approve, is_swap, is_unknown |
| 41-44 | Behavioral | frequency, unique_targets, avg_value, max_value |
| 45-63 | Padding | Zero (MLE alignment) |

### Scoring (ENShell-compatible)

The 3 output neurons map to a 0-100000 threat score:

```
threat_score = (malicious_score * 100000) / (safe + suspicious + malicious)
```

| Score Range | Decision | Action |
|-------------|----------|--------|
| 0 - 15,000 | Approve | Routine, safe to execute |
| 15,001 - 40,000 | Approve | Low risk, log for audit |
| 40,001 - 70,000 | Escalate | Suspicious, needs human review |
| 70,001 - 100,000 | Block | Probable attack, reject |

### Policy Binding

The classifier MUST use `PolicyConfig::strict()`:

```
Commitment: 0x0370c9348ed6edddf310baf5d8104d57c07f36962deea9738dd00519d9948449
```

This is registered on-chain via `register_model_policy(classifier_model_id, 0x0370...)`.
The on-chain verifier rejects proofs generated with any other policy.

Strict policy guarantees:
- All soundness gates enforced (no missing norm/activation proofs)
- Full weight binding with MLE opening proofs
- Piecewise activation verification
- Decode chain validation enforced

### On-Chain Flow

```
                         Agent SDK
                            |
                 1. encode + classify + prove
                            |
                    prove-server (/api/v1/classify)
                            |
            2. submit_action(agent_id, target, value, io_commitment)
                            |
                    AgentFirewallZK contract
                            |
            3. streaming verify (6 TXs → ObelyskVerifier)
                            |
            4. resolve_action_with_proof(action_id, proof_hash, threat_score)
                            |
                    AgentFirewallZK checks:
                      - is_proof_verified(proof_hash)? ✓
                      - policy == strict? ✓
                      - io_commitment matches? ✓
                            |
                    5. EMA trust score update
                       Strike mechanism if suspicious
                       Auto-freeze at 5 strikes
                            |
            6. is_action_approved(action_id) → true/false
                            |
                    External contract executes (or blocks) TX
```

### Comparison with ENShell

| | ENShell | ZKML Classifier |
|---|---|---|
| **Scoring engine** | Claude API (LLM) | Trained MLP (6K params) |
| **Trust model** | Chainlink CRE oracle signs verdict | Cryptographic STARK proof |
| **Verifiability** | DON-signed, trust oracle | Mathematically verified on-chain |
| **Latency** | 5-30s (API + CRE workflow) | <1s proving + ~30s on-chain |
| **Attack surface** | Prompt injection fools Claude | Must modify model weights (committed on-chain) |
| **Cost** | CRE fees + Claude API | STRK gas only (~5-8 STRK) |
| **Upgradability** | Change Claude prompt | Retrain model, re-register weights |
| **Policy binding** | None | Cryptographic (`PolicyConfig::strict`) |
| **Agent reputation** | ENS TXT records (writable by CRE) | On-chain EMA (writable only by verified proof) |

Key tradeoff: we sacrifice Claude's natural language reasoning for cryptographic verifiability and immutable scoring. Mitigated by escalating ambiguous scores (40K-70K) to human review.

## File Structure

```
src/classifier/
    mod.rs          — Module docs, re-exports
    types.rs        — TransactionFeatures, Decision, ClassifierResult, thresholds
    encoder.rs      — encode_transaction() → M31Matrix(1, 64)
    model.rs        — build_classifier_graph(), build_test_classifier(), load_weights_from_arrays()
    prove.rs        — evaluate_transaction() → ClassifierResult
```

## Tests

| Test | What it verifies |
|------|-----------------|
| `test_encode_produces_correct_dimensions` | Output is (1, 64) |
| `test_encode_is_deterministic` | Same input → same output |
| `test_encode_padding_is_zero` | Features 48-63 are zero |
| `test_encode_selector_feature` | Selector at index 16 is correctly masked |
| `test_encode_trust_score` | Trust score at index 26 |
| `test_build_classifier_graph_structure` | 5 nodes, output (1, 3) |
| `test_build_test_classifier` | Weights exist at correct nodes |
| `test_load_weights_dimension_check` | Rejects wrong dimensions |
| `test_evaluate_transaction_produces_result` | Full pipeline: encode → prove → score |
| `test_evaluate_with_strict_policy_differs_from_standard` | Different policies produce different policy commitments |

## Training (Guide)

### 1. Prepare training data

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Features: 48 values per transaction (padded to 64 at inference)
# Labels: 0=safe, 1=suspicious, 2=malicious
X_train = torch.tensor(features, dtype=torch.float32)  # (N, 48)
y_train = torch.tensor(labels, dtype=torch.long)         # (N,)
```

### 2. Define and train the model

```python
model = torch.nn.Sequential(
    torch.nn.Linear(64, 64),  # pad input to 64 at inference
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 3),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    # Pad input to 64 features
    X_padded = torch.nn.functional.pad(X_train, (0, 16))  # (N, 64)
    logits = model(X_padded)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 3. Export weights

```python
import numpy as np

# Extract and quantize weights to M31 range
def quantize_weights(tensor, bits=16):
    """Quantize float weights to unsigned integers in [0, 2^bits)."""
    t = tensor.detach().numpy()
    scale = (2**bits - 1) / (t.max() - t.min() + 1e-8)
    quantized = np.clip((t - t.min()) * scale, 0, 2**bits - 1).astype(np.uint32)
    return quantized.flatten().tolist()

layer0 = quantize_weights(model[0].weight)  # 64x64 = 4096 values
layer2 = quantize_weights(model[2].weight)  # 64x32 = 2048 values
layer4 = quantize_weights(model[4].weight)  # 32x3 = 96 values

# Save as JSON for Rust loading
import json
with open('classifier_weights.json', 'w') as f:
    json.dump({'layer0': layer0, 'layer2': layer2, 'layer4': layer4}, f)
```

### 4. Load in Rust

```rust
use stwo_ml::classifier::model::load_weights_from_arrays;

let weights_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("classifier_weights.json")?)?;
let layer0: Vec<u32> = weights_json["layer0"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
let layer2: Vec<u32> = weights_json["layer2"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
let layer4: Vec<u32> = weights_json["layer4"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();

let weights = load_weights_from_arrays(&layer0, &layer2, &layer4)?;
let model = ClassifierModel {
    graph: build_classifier_graph(),
    weights,
};
```

## Data Sources for Training

| Source | Type | Size | Labels |
|--------|------|------|--------|
| ENShell scored actions | Claude analysis results | Variable | 0-100K threat scores |
| Forta Network alerts | Labeled DeFi attacks | ~100K | attack type |
| Etherscan labeled addresses | Known scam/phishing | ~50K | binary safe/malicious |
| Synthetic generators | Approval drains, flash loans | Unlimited | programmatic labels |
| Agent behavioral logs | Normal agent activity | Variable | baseline (safe) |

## Deployed Infrastructure (Starknet Sepolia)

| Component | Address | Status |
|-----------|---------|--------|
| **AgentFirewallZK** | [`0x043b51f6f571137d0e7c3afa4ca689e84271ba97c5b6fc83349a3fe1275634f0`](https://sepolia.starkscan.co/contract/0x043b51f6f571137d0e7c3afa4ca689e84271ba97c5b6fc83349a3fe1275634f0) | Live, agent registered |
| **ContractRegistry** | [`0x075f9812753666ee506509de0de10bdea3ad1a79d4ed31817a0e2534c9d90607`](https://sepolia.starkscan.co/contract/0x075f9812753666ee506509de0de10bdea3ad1a79d4ed31817a0e2534c9d90607) | Live, linked to firewall |
| **ObelyskVerifier** | [`0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) | Live, streaming GKR |
| **Recursive Verifier** | [`0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7`](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) | Live, 1-TX STARK |
| **Prover (A10G GPU)** | `https://prover.bitsage.network` | Live, `/api/v1/classify` |

## Prove-Server API

```bash
curl -X POST https://prover.bitsage.network/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "target": "0x049d36570d4e46f48e99674bd3fcc84644ddd6b96f7c741b1562b82f9e004dc7",
    "value": "1000000000000000000",
    "selector": "0xa9059cbb",
    "agent_trust_score": 5000,
    "agent_strikes": 0,
    "agent_age_blocks": 50000,
    "target_verified": true,
    "target_has_source": true,
    "tx_frequency": 10,
    "unique_targets_24h": 5
  }'
```

Response (282ms on A10G GPU):
```json
{
  "request_id": "20059293-7b30-4389-bd19-bf2d5adfc864",
  "decision": "approve",
  "threat_score": 24005,
  "scores": [745069812, 2062743355, 886939347],
  "io_commitment": "0x0543908225041becc8c0048743b5674e2840d0f226ee290fb0003988a1940b3f",
  "policy_commitment": "0x0370c9348ed6edddf310baf5d8104d57c07f36962deea9738dd00519d9948449",
  "prove_time_ms": 282
}
```

Request fields: `target` (required), `value`, `selector`, `calldata`, `agent_trust_score`, `agent_strikes`, `agent_age_blocks`, `target_verified`, `target_is_proxy`, `target_has_source`, `target_interaction_count`, `tx_frequency`, `unique_targets_24h`, `avg_value_24h`, `max_value_24h`.

## AgentFirewallZK Contract

The on-chain firewall contract at `0x043b51...` manages agents, queues actions, and resolves them using verified ZKML proofs with **21 sequential security checks** in `resolve_action_with_proof`:

```
 1. ACTION_NOT_FOUND
 2. ACTION_ALREADY_RESOLVED
 3. CONTRACT_PAUSED (emergency pause)
 4. ACTION_EXPIRED (MAX_ACTION_AGE=3600s)
 5. NOT_AGENT_OR_CONTRACT_OWNER
 6. AGENT_FROZEN
 7. PROOF_NOT_VERIFIED (cross-contract ObelyskVerifier)
 8. PROOF_ALREADY_USED (replay protection)
 9. IO_COMMITMENT_MISMATCH (Poseidon recomputation from calldata)
10. PROOF_IO_MISMATCH (double-check against proof storage)
11-16. IO structure validation (6 bounds checks)
17. INPUT_TARGET_MISMATCH (248-bit address reconstruction)
18. INPUT_SELECTOR_MISMATCH (31-bit match)
19. INPUT_TRUST_SCORE_MISMATCH (±5000 vs on-chain)
20. INPUT_STRIKES_MISMATCH (exact match)
21. INPUT_AGE_MISMATCH (±100 blocks)
```

Plus value/selector feature derivation, behavioral tracking cross-checks, ContractRegistry lookups, ERC20 balance queries, weight root hash verification, and policy enforcement.

**Threat score is computed ON-CHAIN** from the proven IO data — not caller-supplied. The 3 output neurons (safe, suspicious, malicious) are extracted from packed M31 values and the score is computed as `(malicious * 100000) / total`.

**Asymmetric EMA trust scoring**: alpha_up=0.5 (bad actions hit hard), alpha_down=0.1 (safe actions forgive slowly). Auto-freeze at 5 strikes.

## TypeScript SDK

```typescript
import { AgentFirewallSDK } from '@obelyzk/sdk/firewall'

const firewall = new AgentFirewallSDK({
  proverUrl: 'https://prover.bitsage.network',
  firewallContract: '0x043b51f6f571137d0e7c3afa4ca689e84271ba97c5b6fc83349a3fe1275634f0',
  verifierContract: '0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005',
  rpcUrl: process.env.STARKNET_RPC,
  account: myAccount,
})

// Classify a transaction (282ms GPU proving)
const result = await firewall.classify({
  target: '0x049d36570d4e46f48e99674bd3fcc84644ddd6b96f7c741b1562b82f9e004dc7',
  value: '1000000000000000000',
  selector: '0xa9059cbb',
  agentTrustScore: 5000,
})
console.log(result.decision)      // "approve" | "escalate" | "block"
console.log(result.threatScore)   // 0-100000
console.log(result.ioCommitment)  // for on-chain submission

// Register an agent
await firewall.registerAgent('my-agent-001')

// Check trust status
const status = await firewall.getAgentStatus('my-agent-001')
console.log(status.trusted)       // true
console.log(status.trustScore)    // 0
console.log(status.strikes)       // 0
```

## Recursive STARK (Single TX)

The classifier proof compresses via recursive STARK:

```
GKR proof (2277 felts) → Recursive STARK (635 felts) → 1 TX on Starknet
```

```bash
prove-model --model-dir ./smollm2 --layers 1 --gkr --format ml_gkr \
  --policy strict --recursive --output proof.json
```

Performance on A10G:
- GKR prove: 1.4s
- Recursive compress: 0.65s
- Total: 2.1s
- Calldata: 635 felts (single TX)

## Security Audit Status

| Attack | Status | Fix |
|--------|--------|-----|
| Score manipulation | **CLOSED** | On-chain extraction, 12 bounds checks |
| Model substitution | **CLOSED** | Weight root hash cross-check |
| Feature forgery | **95% CLOSED** | 47/48 features verified on-chain |
| EMA dilution | **CLOSED** | Asymmetric decay (0.5 up, 0.1 down) |
| Classifier capacity | **By design** | Escalation path for ambiguous scores |
| Action expiry | **CLOSED** | MAX_ACTION_AGE=3600s |
| Emergency controls | **Added** | Pause/unpause, 2-step ownership transfer |
| Rate limiting | **Added** | 10 pending actions per agent max |
| Proof replay | **Added** | used_proof_hashes map |

## Roadmap

| Item | Status |
|------|--------|
| PolicyConfig framework | **DONE** — 21 tests, Fiat-Shamir bound |
| ZKML classifier module | **DONE** — 10 tests, 282ms GPU |
| AgentFirewallZK contract | **DONE** — 30 tests, deployed Sepolia |
| ContractRegistry | **DONE** — deployed Sepolia |
| proof_io_commitment getter | **DONE** — verifier extended |
| `/api/v1/classify` endpoint | **DONE** — live on A10G |
| Firewall SDK (TypeScript) | **DONE** — `@obelyzk/sdk/firewall` |
| Recursive STARK with policy | **DONE** — 635 felts, 1 TX |
| On-chain behavioral tracking | **DONE** — 24h window, 5 features |
| Real training data | Remaining |
| Trained production model | Remaining |
