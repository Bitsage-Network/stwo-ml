# ObelyZK / stwo-ml — Project Status & Roadmap

*Last updated: April 6, 2026*

---

## What This Project Does

ObelyZK proves that machine learning models ran correctly. You give it a model and an input, it runs inference over the Mersenne-31 prime field, generates a GKR+STARK proof, and verifies it on Starknet — either in 6 streaming transactions or 1 recursive transaction.

On top of the prover, we built a **transaction classifier** and **on-chain firewall** that scores autonomous AI agent transactions before they execute. The classifier is a small MLP whose inference is cryptographically proven. The firewall contract reads the proven score and gates the transaction — no oracle, no trust, pure math.

---

## System Architecture

```
                        Agent / SDK / Claude Code
                                |
                    ┌───────────┴───────────┐
                    │   /api/v1/classify     │  ← 282ms on A10G GPU
                    │   Transaction features │
                    │   → MLP classifier     │
                    │   → GKR+STARK proof    │
                    │   → threat score       │
                    └───────────┬───────────┘
                                |
                    ┌───────────┴───────────┐
                    │   Starknet Sepolia     │
                    │                        │
                    │   ObelyskVerifier      │  ← streaming (6 TX) or recursive (1 TX)
                    │   AgentFirewallZK      │  ← 21 security checks
                    │   ContractRegistry     │  ← target attestations
                    └───────────────────────┘
```

---

## Component Status

### Tier 1: Production-Ready

| Component | Status | Evidence |
|-----------|--------|----------|
| **GKR Prover (CPU+GPU)** | Production | 936 tests, 7 models proven, 103s for 40-layer Qwen3-14B |
| **Recursive STARK v2** | Production | 48-col chain AIR, 38 constraints, ~4,934 felts, 160-bit security, 4 verified TXs on Sepolia (latest: [`0x021512dd...`](https://sepolia.starkscan.co/tx/0x021512dd991a1c317a1aa93a382bed322af2e63d9fa01b9c5a3b133cf1ceebb8)), contract `0x0121d1...8c005`, 9 security layers, two-level recursion |
| **Recursive STARK v1** | Production | 28-col AIR, 27 constraints, 981 felts, verified on Sepolia ([TX](https://sepolia.starkscan.co/tx/0x276c6a448829c0f3975080914a89c2a9611fc41912aff1fddfe29d8f3364ddc)) |
| **Streaming GKR Verifier** | Production | 6/6 steps verified on Sepolia, 14/14 streaming steps |
| **PolicyConfig** | Production | 21 tests, 3 presets, Fiat-Shamir bound, Cairo enforced |
| **Model Loading** | Production | HuggingFace SafeTensors, ONNX, any LLaMA/Qwen/Phi architecture |
| **Weight Commitment** | Production | Poseidon Merkle roots, aggregated binding, full MLE openings |

### Tier 2: Deployed, Needs Integration Testing

| Component | Status | What's Missing |
|-----------|--------|----------------|
| **AgentFirewallZK** | Deployed on Sepolia | Never resolved a real proof on-chain |
| **ContractRegistry** | Deployed on Sepolia | No attestations populated yet |
| **ZKML Classifier** | Live on A10G | Uses random test weights, not trained |
| **`/api/v1/classify`** | Live endpoint | Returns scores but model is untrained |
| **Recursive + Policy** | Working locally | 635 felts, needs on-chain submission test |

### Tier 3: Published, Thin Wrappers

| Component | Status | What's Missing |
|-----------|--------|----------------|
| **npm `@obelyzk/sdk@1.2.0`** | Published | `evaluateAction()` full flow untested against live contract |
| **PyPI `bitsage-sdk@0.2.0`** | Published | `classify()` works, no contract interaction |
| **crates.io `bitsage-sdk@0.2.0`** | Published | `classify()` works, no contract interaction |

### Tier 4: Not Built

| Component | Description | Effort |
|-----------|-------------|--------|
| **Trained classifier model** | PyTorch MLP on labeled DeFi data | 1-2 weeks (data collection + training) |
| **E2E on-chain flow** | classify → prove → verify → resolve → trust update | 1 day |
| **Claude Code MCP tool** | `classify_transaction` tool in MCP server | 1 day |
| **Per-agent policy routing** | Different models per domain (DeFi, NFT, governance) | 1 week |
| **Agent developer tutorial** | Step-by-step: register agent → classify → execute | 1 day |

---

## Deployed Contracts (Starknet Sepolia)

| Contract | Address | Purpose |
|----------|---------|---------|
| **Recursive Verifier v2** | [`0x0121d1e9...`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) | Production: 48-col chain AIR, 38 constraints, 160-bit security, 9 security layers, two-level recursion |
| **Recursive Verifier v1** | [`0x1c208a5f...`](https://sepolia.starkscan.co/contract/0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7) | Original: 28-col AIR, 27 constraints, 1-TX STARK verification |
| **Streaming Verifier** | [`0x0121d1e9...`](https://sepolia.starkscan.co/contract/0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005) | 6-TX GKR streaming verification |
| **AgentFirewallZK** | [`0x043b51f6...`](https://sepolia.starkscan.co/contract/0x043b51f6f571137d0e7c3afa4ca689e84271ba97c5b6fc83349a3fe1275634f0) | Agent management, action gating, trust scoring |
| **ContractRegistry** | [`0x075f9812...`](https://sepolia.starkscan.co/contract/0x075f9812753666ee506509de0de10bdea3ad1a79d4ed31817a0e2534c9d90607) | Target contract attestations (is_verified, has_source) |

---

## Published SDKs

| Package | Registry | Version | Install |
|---------|----------|---------|---------|
| `@obelyzk/sdk` | npm | 1.2.0 | `npm install @obelyzk/sdk` |
| `bitsage-sdk` | PyPI | 0.2.0 | `pip install bitsage-sdk` |
| `bitsage-sdk` | crates.io | 0.2.0 | `cargo add bitsage-sdk` |
| `@obelyzk/mcp-server` | npm | 0.1.0 | `npm install @obelyzk/mcp-server` |
| `@bitsagecli/cli` | npm | 0.2.1 | `npm install -g @bitsagecli/cli` |

---

## Security Audit Status

Six attack vectors identified via adversarial audit. Five closed, one by-design:

| Attack | Severity | Status | Fix |
|--------|----------|--------|-----|
| **1. Score manipulation** | CRITICAL | **CLOSED** | On-chain extraction from proven IO data. 12 bounds checks. Threat score computed inside the contract, not caller-supplied. |
| **2. Model substitution** | HIGH | **CLOSED** | `classifier_weight_root_hash` stored in firewall, cross-checked against verifier's registered Poseidon Merkle roots on every resolution. |
| **3. Feature forgery** | HIGH | **95% CLOSED** | 47/48 input features verified against on-chain state. Target address (248-bit reconstruction), selector, agent trust/strikes/age, value features, selector features, behavioral stats — all cross-checked. Remaining: `is_proxy` (needs try-call). |
| **4. EMA dilution** | MEDIUM | **CLOSED** | Asymmetric decay: alpha_up=0.5 (bad actions raise score fast), alpha_down=0.1 (safe actions lower score slowly). Takes 8 safe actions to recover from 75K, vs 2 before. |
| **5. Classifier capacity** | LOW | **BY DESIGN** | 6K-param MLP is a first-pass filter, not a replacement for LLM reasoning. Escalation path (40K-70K) routes ambiguous scores to human review. |
| **6. Action expiry** | MEDIUM | **CLOSED** | MAX_ACTION_AGE=3600s. Stale actions cannot be resolved. Checked on resolve, approve, and reject. |

**Additional hardening:** emergency pause, 2-step ownership transfer, per-agent rate limiting (10 pending max), proof replay protection, escalation strikes (reject = +1 strike), ContractRegistry for target attestations, IERC20 balance queries for value_balance_ratio.

---

## How Agents Get Policies Today

```typescript
import { AgentFirewallSDK } from '@obelyzk/sdk/firewall'

const firewall = new AgentFirewallSDK({
  proverUrl: 'https://prover.bitsage.network',
  firewallContract: '0x043b51f6f571137d0e7c3afa4ca689e84271ba97c5b6fc83349a3fe1275634f0',
  verifierContract: '0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005',
  rpcUrl: process.env.STARKNET_RPC,
  account: myAccount,
})

// Before every transaction:
const result = await firewall.classify({
  target: tx.to,
  value: tx.value.toString(),
  selector: tx.data.slice(0, 10),
})

if (result.decision === 'approve') {
  await executeTransaction(tx)
} else if (result.decision === 'escalate') {
  await notifyOperator(result)
} else {
  console.error(`BLOCKED: score ${result.threatScore}`)
}
```

The policy (`PolicyConfig::strict`) is enforced at the prover level. Every proof carries a policy commitment hash (`0x0370c934...`) that the on-chain verifier checks. An agent cannot downgrade from strict to relaxed without the contract rejecting the proof.

**What's missing:** per-agent policy customization (different models per domain), and a Claude Code integration.

---

## Supported Models

### For Inference Proving (core ZKML)

Any HuggingFace transformer with SafeTensors weights:

| Model | Params | Prove Time (A10G) | Recursive Felts |
|-------|--------|-------------------|-----------------|
| SmolLM2-135M (1 layer) | 135M | 3.3s | ~4,934 (v2) / 981 (v1) |
| SmolLM2-135M (30 layers) | 135M | 95s | ~4,934 (v2) / 981 (v1) |
| Qwen2-0.5B | 494M | ~20s | ~4,934 (v2) / 981 (v1) |
| Llama-3.2-3B | 3B | ~35s | ~4,934 (v2) / 981 (v1) |
| Phi-3-mini | 3.8B | ~45s | ~4,934 (v2) / 981 (v1) |
| Mistral-7B | 7B | ~90s | ~4,934 (v2) / 981 (v1) |
| Qwen3-14B (H100) | 14B | 103s | ~4,934 (v2) / 981 (v1) |

### For Transaction Classification

Only the 6K-param MLP (64→64→32→3) right now. 282ms on A10G. Test weights (not trained).

---

## Claude Code Integration Path

Claude Code supports MCP (Model Context Protocol) servers that provide tools. The path to integration:

### Phase 1: MCP Tool (1 day)

Add a `classify_transaction` tool to `@obelyzk/mcp-server`:

```json
{
  "name": "classify_transaction",
  "description": "Score a blockchain transaction for safety using ZKML-proven classifier",
  "parameters": {
    "target": "contract address",
    "value": "transaction value in wei",
    "selector": "function selector (4 bytes)",
    "calldata": "transaction calldata"
  }
}
```

Claude Code calls this before executing any on-chain action. The tool calls `/api/v1/classify` and returns the decision + score.

### Phase 2: Agent Registration (1 day)

Register Claude Code as an agent on the firewall:
```typescript
await firewall.registerAgent('claude-code-session-001')
```

Every transaction goes through `submit_action` → classify → resolve. If Claude gets prompt-injected into approving a drain, the classifier catches it.

### Phase 3: Claude Code Hooks (Future)

Claude Code has pre/post hooks for tool execution. A pre-hook could automatically classify every `Bash` or contract call:

```json
{
  "hooks": {
    "pre_tool_call": {
      "command": "node classify_hook.js",
      "tools": ["Bash", "contract_call"]
    }
  }
}
```

### Phase 4: Per-Session Policies (Future)

Different Claude Code sessions get different policy profiles:
- **Development session**: `PolicyConfig::relaxed` — fast, no blocking
- **Production deployment**: `PolicyConfig::strict` — full verification
- **Financial operations**: Custom policy with lower escalation threshold

---

## Roadmap

### Sprint 1: Close the Loop (This Week)

| Task | Effort | Impact |
|------|--------|--------|
| **E2E on-chain flow** — classify → prove → verify → resolve on Sepolia | 1 day | Proves the entire system works |
| **Claude Code MCP tool** — `classify_transaction` in `@obelyzk/mcp-server` | 1 day | Claude Code agents get guardrails |
| **Agent developer tutorial** — step-by-step from register to execute | 1 day | First external users |

### Sprint 2: Train the Classifier (Next 2 Weeks)

| Task | Effort | Impact |
|------|--------|--------|
| **Data collection** — Forta alerts, Etherscan labels, synthetic patterns | 1 week | Training data for the MLP |
| **PyTorch training pipeline** — train, validate, export weights | 3 days | Replace random test weights |
| **Model registration on-chain** — weight commitments + strict policy | 1 day | Binds trained model to contract |
| **Retrain automation** — weekly retrain from new labeled data | 2 days | Model stays current |

### Sprint 3: Per-Agent Policies (Month 2)

| Task | Effort | Impact |
|------|--------|--------|
| **Multiple classifier models** — DeFi, NFT, governance, cross-chain | 1 week | Domain-specific scoring |
| **Per-agent policy registration** — agents choose their model | 3 days | Customization |
| **Policy marketplace** — share/sell trained models | 2 weeks | Ecosystem |
| **`is_proxy` detection** — try-call pattern for proxy contracts | 1 day | Last unverified feature |

### Sprint 4: Production Hardening (Month 3)

| Task | Effort | Impact |
|------|--------|--------|
| **Mainnet deployment** — contracts + prover on mainnet Starknet | 1 week | Real money at stake |
| **Multi-prover fleet** — load balancing across GPUs | 1 week | Availability |
| **Formal verification** — Cairo contract audit | External | Trust |
| **Larger classifier** — 50K+ params, attention layers | 2 weeks | Better detection |

---

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| Rust full (single-threaded) | 936 | 2 pre-existing failures |
| Rust policy + classifier | 31 | All passing |
| Rust E2E firewall | 4 | All passing (zero mocks) |
| Cairo full | 138 | 13 pre-existing failures |
| Cairo firewall | 30 | All passing |
| Cairo policy | 4 | All passing |
| **Total** | **1143** | **15 pre-existing, 0 new** |

---

## Performance

| Operation | Time | Hardware |
|-----------|------|----------|
| Classify + prove (6K-param MLP) | 282ms | A10G GPU |
| GKR prove (1 layer SmolLM2) | 3.3s | A10G GPU |
| GKR prove (30 layers SmolLM2) | 95s | A10G GPU |
| GKR prove (40 layers Qwen3-14B) | 103s | H100 GPU |
| Recursive STARK compression | 0.65s | A10G GPU |
| On-chain verify (recursive, 1 TX) | ~$0.02 | Starknet Sepolia |
| On-chain verify (streaming, 6 TX) | ~$0.12 | Starknet Sepolia |

---

## Repository Structure

```
libs/
  stwo-ml/                          # Core ZKML prover
    src/
      classifier/                   # Transaction classifier (encoder, model, prove)
      policy.rs                     # PolicyConfig (strict/standard/relaxed)
      gkr/                          # GKR sumcheck prover + verifier
      aggregation.rs                # Proof aggregation pipeline
      starknet.rs                   # Calldata serialization + self-verification
      recursive/                    # Recursive STARK composition
      bin/
        prove-model.rs              # CLI binary (--policy, --recursive)
        prove-server.rs             # HTTP API (/api/v1/classify, /api/v1/prove)
    tests/
      e2e_firewall.rs               # Full prove→verify→score→freeze (no mocks)
    docs/
      CLASSIFIER.md                 # Full architecture + API + security
      ENV_VARS.md                   # Policy presets + all config
      STATUS.md                     # This file
    scripts/
      deploy_firewall.mjs           # Deploy AgentFirewallZK to Sepolia
      deploy_policy_upgrade.mjs     # Deploy PolicyConfig-enabled verifier

  elo-cairo-verifier/               # Cairo smart contracts
    src/
      verifier.cairo                # ObelyskVerifier (streaming + policy)
      firewall.cairo                # AgentFirewallZK (21 checks, behavioral tracking)
      registry.cairo                # ContractRegistry (target attestations)
    tests/
      test_firewall.cairo           # 30 tests
      test_model_gkr_contract.cairo # Policy tests

sdk/
  typescript/                       # @obelyzk/sdk (npm)
    src/firewall/                   # AgentFirewallSDK
  python/                           # bitsage-sdk (PyPI)
    bitsage/firewall.py             # FirewallClient
  rust/                             # bitsage-sdk (crates.io)
    src/firewall.rs                 # FirewallClient
```
