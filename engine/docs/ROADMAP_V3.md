# ObelyZK Roadmap v3 -- Gap Closure and Production Readiness

**Date**: April 6, 2026
**Author**: Internal Engineering
**Baseline**: STATUS_REPORT.md (same date)
**Goal**: Close all known gaps, reach production-ready state

---

## Phase 0: Critical Fixes (Week 1-2)

These must be fixed before any public demo, investor meeting, or external audit engagement. They represent gaps where the system's claims do not match its actual behavior.

---

### Item 1: Fix Weight Binding On-Chain

**Description**: Pass real `circuit_hash` and `weight_super_root` values to `register_model_recursive()` instead of the current placeholder `0x1` values. The Rust prover already computes these -- the weight Merkle roots are built in `apply_aggregated_oracle_sumcheck` and the circuit descriptor is available from `LayeredCircuit`. The gap is in the submission scripts (`submit_recursive.mjs`, `deploy_recursive_v2.mjs`) which hardcode `0x1`.

**Why it matters**: Without real binding, the on-chain contract cannot distinguish between proofs from different models. This is the single largest security gap. Any valid STARK proof will be accepted for any registered model_id. The entire trust model of "this proof attests to this specific model's inference" is broken.

**Estimated effort**: 3-5 days

**Dependencies**: None. All computation exists in the Rust prover.

**Work items**:
1. Add `circuit_hash` computation to `prove_model_pure_gkr` output: Poseidon hash of `LayeredCircuit` descriptor (layer types, shapes, weight node IDs). The function `compute_circuit_hash()` may already exist in `recursive/types.rs`; verify and wire it.
2. Add `weight_super_root` to the recursive proof output JSON. This is the Poseidon Merkle root of all weight matrices, already computed during proving.
3. Update `serialize_recursive_proof_calldata()` in `cairo_serde.rs` to include real values.
4. Update `submit_recursive.mjs` to read `circuit_hash` and `weight_super_root` from the proof JSON and pass them to `register_model_recursive()`.
5. Update `deploy_recursive_v2.mjs` similarly.
6. Verify on Sepolia: register a model with real values, submit proof, confirm acceptance. Then register a second model_id with different values and confirm the same proof is rejected.

**Acceptance criteria**:
- `register_model_recursive()` is called with non-placeholder values that match the prover's computation.
- A proof generated for model A is rejected when submitted against model B's registration.
- The circuit_hash and weight_super_root in the on-chain event match the values computed by the Rust prover.

---

### Item 2: Fix Streaming Self-Verification Bug

**Description**: Debug the Fiat-Shamir transcript divergence between `prove_model_pure_gkr` (prover) and `replay_verify_serialized_proof` (self-verifier). The divergence occurs at layer 1 (`MATMUL_FINAL_MISMATCH`) and layer 210 (RMSNorm linear round) for the 30-layer SmolLM2 model.

**Why it matters**: The streaming path is the fallback when recursive proving is not available. If it cannot self-verify, it cannot be trusted for on-chain submission. Additionally, this bug may indicate a deeper issue in the Fiat-Shamir transcript that coincidentally does not manifest in the recursive path due to a different code path.

**Estimated effort**: 3-5 days (debugging, possibly 1-2 days if the root cause is a simple serialization mismatch)

**Dependencies**: None.

**Work items**:
1. Add debug logging to both `prove_gkr` and `replay_verify_serialized_proof` to dump the Poseidon channel state (digest) after every mix/draw operation.
2. Run both paths on a 1-layer model and diff the channel state logs. Identify the first point of divergence.
3. If the 1-layer model works, repeat with 2-layer, 5-layer, etc. to find the minimum reproduction case.
4. Fix the root cause (likely a missing or extra `mix` call in one of the paths, or a serialization/deserialization mismatch in the proof format).
5. Run the full 30-layer model and confirm `cryptographic_self_verify: PASSED`.

**Acceptance criteria**:
- `replay_verify_serialized_proof` returns success for the 30-layer SmolLM2 proof.
- `cryptographic_self_verify: PASSED` appears in the prove-model output.
- The streaming path can submit to the v32 contract and all 14 verification steps pass for the 30-layer model.

---

### Item 3: Enable Policy Commitment

**Description**: Remove `STWO_SKIP_POLICY_COMMITMENT=1` from deployment scripts and thread the policy commitment through the recursive path. The `PolicyConfig` system already computes Poseidon commitments for each preset. The gap is that the commitment is not mixed into the Fiat-Shamir channel (skipped by the env var) and the Cairo contract does not read or emit it.

**Why it matters**: Without policy commitment, proofs generated under the `relaxed` policy (which skips soundness checks) are indistinguishable from `strict` policy proofs. An attacker could use relaxed mode to generate proofs faster while claiming strict verification.

**Estimated effort**: 5-7 days

**Dependencies**: The Cairo recursive verifier contract may need a minor update to accept and emit the policy_hash as a public input. This requires a contract upgrade (which itself depends on the recursive contract having an upgrade mechanism -- see Item 12).

**Work items**:
1. Remove `STWO_SKIP_POLICY_COMMITMENT=1` from `standard` policy preset in `src/policy.rs`.
2. Verify that `prove_recursive_with_policy()` correctly mixes the policy commitment into the Fiat-Shamir channel.
3. Update the recursive witness generator to include the policy commitment mix in its instrumented channel replay.
4. Update the recursive AIR public inputs to include `policy_hash`.
5. Update the Cairo `recursive_verifier.cairo` to accept `policy_hash` as an additional parameter and emit it in the `ProofVerified` event.
6. Update submission scripts to pass the policy_hash.
7. Test: prove with `strict` and `standard` policies, verify both produce valid proofs with different policy_hash values, verify the contract emits the correct hash.

**Acceptance criteria**:
- Proofs carry a non-zero `policy_commitment` in their metadata and Fiat-Shamir transcript.
- On-chain `ProofVerified` events include the `policy_hash` field.
- A proof generated with `--policy standard` has a different policy_hash than one generated with `--policy strict`.
- The recursive STARK proof remains valid after policy commitment is enabled.

---

### Item 4: Wire Real SDK Prover Client

**Description**: Create actual `ProverClient` classes in TypeScript and Python that call the prove-server's `/api/v1/prove` and `/api/v1/attest` endpoints. Replace the current BitSage distributed compute client wrappers in `@obelyzk/sdk` and the `obelyzk` PyPI package.

**Why it matters**: The SDK is the primary developer-facing interface. If developers install it and it does not work, trust is lost. The current packages contain code for a different product (BitSage compute network).

**Estimated effort**: 5-7 days

**Dependencies**: Prove-server must be deployed and accessible (see Item 9). For this phase, the SDK can target `localhost:8080` with documentation for self-hosting.

**Work items**:
1. Create `@obelyzk/sdk` v2 with a `ProverClient` class:
   - `prove(modelId, input, options)` -> `{ output, proof_hash, calldata_felts }`
   - `attest(modelId, input, options)` -> `{ output, proof_hash, tx_hash, verified }`
   - `verify(proofHash)` -> `{ verified, model_id, io_commitment }`
   - Constructor accepts `{ proverUrl, starknetRpc }`.
2. Create `obelyzk` Python package v2 with `ObelyzkClient`:
   - Same methods as TypeScript, using `httpx` or `requests`.
3. Add integration tests that spin up a local prove-server and exercise all endpoints.
4. Update README and package descriptions to accurately reflect the API.
5. Deprecate old package versions with a notice pointing to v2.

**Acceptance criteria**:
- `npm install @obelyzk/sdk && node -e "const { ProverClient } = require('@obelyzk/sdk'); const c = new ProverClient({proverUrl: 'http://localhost:8080'}); c.prove('smollm2-135m', 'Hello').then(console.log)"` works against a running prove-server.
- `pip install obelyzk && python -c "from obelyzk import ObelyzkClient; ..."` works equivalently.
- All methods return well-typed responses with proper error handling.

---

## Phase 1: Hardening (Week 3-4)

These items strengthen the security posture and expand verification coverage.

---

### Item 5: Adversarial Proof Testing

**Description**: Submit malformed, truncated, replayed, and tampered proofs to the trustless recursive contract on Sepolia. Verify all are rejected with appropriate error messages.

**Why it matters**: The contract has been tested only with valid proofs. In production, adversaries will submit invalid proofs. We need confidence that the Cairo verifier's deserialization and verification logic correctly rejects all invalid inputs without panicking, consuming excessive gas, or entering undefined states.

**Estimated effort**: 5-7 days

**Dependencies**: None (uses existing deployed contract).

**Work items**:
1. Generate a valid recursive proof for SmolLM2-135M.
2. Create a test harness (JavaScript) that submits modified versions of the proof:
   - Bit-flip in each major proof section (commitments, sampled_values, decommitments, fri_proof).
   - Truncated proof (missing last N felts for N in {1, 10, 100}).
   - Proof replay: same proof submitted with different `io_commitment`.
   - Proof replay: same proof submitted for different `model_id`.
   - Wrong model_id with valid proof.
   - Oversized calldata (append garbage felts).
   - Zero-length proof.
   - All-zeros proof.
3. For each test case, verify the transaction reverts (not succeeds silently).
4. Log gas consumption for each rejection (ensure no gas-bomb attacks).
5. Document results.

**Acceptance criteria**:
- All malformed proofs are rejected by the contract (transaction reverts).
- No test case causes the contract to accept an invalid proof.
- No test case consumes more than 2x the gas of a valid proof verification.
- Results documented with TX hashes for each test case.

---

### Item 6: Cairo Recursive Verifier Tests

**Description**: Write snforge tests for the recursive verifier contract. The streaming verifier has 41 test files; the recursive verifier (the primary path) has zero.

**Why it matters**: The recursive contract is the trust anchor of the entire system. It must have automated tests that run in CI and catch regressions.

**Estimated effort**: 5-7 days

**Dependencies**: snforge setup for the elo-cairo-verifier project.

**Work items**:
1. Set up snforge test infrastructure for `recursive_verifier.cairo`.
2. Write tests:
   - `test_register_model_recursive`: Register a model, verify storage.
   - `test_register_model_unauthorized`: Non-owner cannot register (if access control exists).
   - `test_verify_recursive_valid`: Submit a valid proof, verify acceptance and event emission.
   - `test_verify_recursive_tampered_commitment`: Tamper with a commitment felt, verify rejection.
   - `test_verify_recursive_tampered_fri`: Tamper with FRI proof data, verify rejection.
   - `test_verify_recursive_wrong_model`: Valid proof for model A submitted for model B, verify rejection.
   - `test_verify_recursive_wrong_io`: Valid proof with wrong io_commitment, verify rejection.
   - `test_verify_recursive_duplicate`: Submit same proof twice, verify behavior (should succeed or fail gracefully).
   - `test_is_recursive_proof_verified`: Query verification status.
   - `test_get_recursive_verification_count`: Count increments on successful verification.
3. Embed a valid proof as a test fixture (serialized felt252 array).
4. Add to CI pipeline.

**Acceptance criteria**:
- At least 10 snforge tests passing for the recursive verifier.
- Tests run in CI on every PR to the cairo verifier.
- At least one tampered-proof rejection test per major proof component (commitment, FRI, PoW).

---

### Item 7: Close Soundness Gates

**Description**: For each of the five weakened soundness gates, either (a) implement the missing verification in the recursive AIR, or (b) document the exact attack surface created by the skip and assess the risk.

**Why it matters**: The weakened gates mean that only MatMul operations are fully verified. Activation functions, normalization, and batch tokens are unverified in the on-chain path. This is the difference between "we verify ML inference" and "we verify matrix multiplication."

**Estimated effort**: 2-4 weeks (this is the largest single item)

**Dependencies**: Items 1-3 should be completed first (they are simpler and higher-impact).

**Work items**:
1. **`STWO_SKIP_RMS_SQ_PROOF`** (Priority: HIGH): The RMSNorm variance sub-proof is already implemented in the Rust prover (`reduce_rmsnorm_layer_with_gamma`). The gap is that it is not included in the recursive AIR witness. Extend the witness generator to record the variance sumcheck operations, and add corresponding constraints to the recursive AIR. Alternative: document that without this, a prover can substitute arbitrary RMS values.
2. **`STWO_ALLOW_MISSING_NORM_PROOF`** (Priority: HIGH): Similar to above. The LayerNorm/RMSNorm LogUp proof exists in the Rust prover but is not part of the recursive AIR. Either extend the AIR or document the gap.
3. **`STWO_PIECEWISE_ACTIVATION`** (Priority: MEDIUM): Piecewise algebraic activation proofs exist but are disabled. These prove that activation function outputs fall on the correct piecewise-linear approximation segments. Evaluate whether the on-chain cost of including them in the recursive AIR is acceptable.
4. **`STWO_ALLOW_LOGUP_ACTIVATION`** (Priority: MEDIUM): LogUp activation proofs are skipped because M31 matmul outputs exceed the precomputed table range. Either increase table size, implement range reduction before LogUp, or document the gap.
5. **`STWO_SKIP_BATCH_TOKENS`** (Priority: LOW): Batch token proofs are relevant only for batched inference (multiple inputs in one proof). Document the gap for single-inference use case; implement for batch use case.

For each gate, produce one of:
- A PR that extends the recursive AIR to include the missing verification, OR
- A written security assessment with: (a) what exactly a malicious prover can do, (b) the probability of exploitability, (c) a mitigation timeline.

**Acceptance criteria**:
- At least `STWO_SKIP_RMS_SQ_PROOF` and `STWO_ALLOW_MISSING_NORM_PROOF` are closed (either in the AIR or with documented mitigation).
- The `strict` policy produces proofs that pass full on-chain verification with these gates enforced.
- Updated STATUS_REPORT.md reflecting the new state.

---

### Item 8: Multi-Model On-Chain Verification

**Description**: Prove and verify at least two additional models on Starknet Sepolia via the recursive path: Qwen2-0.5B (smallest, fastest) and Phi-3-mini-3.8B (different architecture, fused QKV).

**Why it matters**: Demonstrating that the system works for multiple models with different architectures validates the "any model" claim. Currently only SmolLM2-135M has been verified on-chain.

**Estimated effort**: 3-5 days (assuming Item 1 is completed first so weight binding works)

**Dependencies**: Item 1 (weight binding fix) must be completed so that each model's registration has unique, real circuit_hash and weight_super_root values.

**Work items**:
1. Prove Qwen2-0.5B on A10G with `--recursive` flag. Record proof JSON.
2. Register Qwen2-0.5B on the recursive contract with real binding values.
3. Submit recursive proof. Verify acceptance.
4. Repeat for Phi-3-mini-3.8B.
5. Verify that the Qwen2 proof is rejected if submitted against the Phi-3 model_id (and vice versa).
6. Document TX hashes and gas costs for each model.

**Acceptance criteria**:
- At least 3 different models verified on Sepolia (SmolLM2 + Qwen2 + Phi-3).
- Each model has a unique circuit_hash and weight_super_root on-chain.
- Cross-model proof substitution is rejected.

---

## Phase 2: Production Infrastructure (Week 5-8)

These items build the infrastructure needed for external users and production deployment.

---

### Item 9: Deploy Prove-Server as Hosted API

**Description**: Deploy the `prove-server` binary on the A10G instance behind `api.bitsage.network`. Add health checks, rate limiting, API key authentication, and monitoring.

**Why it matters**: The hosted API is the primary access point for SDK users. Without it, all proving requires building from source and having GPU access. This is the bridge between "research project" and "developer product."

**Estimated effort**: 5-7 days

**Dependencies**: A10G instance is already running. DNS for `api.bitsage.network` must be configured.

**Work items**:
1. Set up nginx reverse proxy on A10G with TLS (Let's Encrypt).
2. Deploy `prove-server` binary behind nginx.
3. Add health check endpoint (`/health`) that returns server status, GPU availability, and loaded models.
4. Add API key authentication via `X-API-Key` header. Store keys in environment, not in code.
5. Add rate limiting (e.g., 10 requests/minute per API key for prove, 100/minute for verify).
6. Add request logging and basic monitoring (request count, latency, error rate).
7. Pre-load SmolLM2-135M and Qwen2-0.5B models on startup.
8. Verify SDK clients can connect and get proofs.
9. Document the API in a minimal OpenAPI spec.

**Acceptance criteria**:
- `curl https://api.bitsage.network/health` returns 200 with server status.
- `curl -X POST https://api.bitsage.network/api/v1/prove -H "X-API-Key: ..." -d '{"model":"smollm2-135m","input":"Hello"}'` returns a valid proof.
- Rate limiting rejects excessive requests with 429.
- Invalid API keys receive 401.

---

### Item 10: CLI Alignment

**Description**: Align the CLI binary commands with documentation. The `obelysk` binary should support `obelysk prove`, `obelysk submit`, `obelysk verify`, and `obelysk models`. Remove BitSage worker/validator commands from `@obelyzk/cli` (move to a separate package).

**Why it matters**: Users who install the CLI and run documented commands should not encounter unknown command errors or unrelated BitSage functionality.

**Estimated effort**: 3-5 days

**Dependencies**: None.

**Work items**:
1. Audit current `src/bin/obelysk.rs` (or equivalent) for supported subcommands.
2. Implement or wire missing subcommands:
   - `obelysk prove --model <path> --input <text> [--recursive] [--policy <preset>]` -> outputs proof JSON.
   - `obelysk submit --proof <path> [--recursive]` -> submits to Starknet.
   - `obelysk verify --proof-hash <hash>` -> queries on-chain verification status.
   - `obelysk models` -> lists registered models on-chain.
3. Remove or gate BitSage-specific commands (`bitsage deploy`, `bitsage stake`) behind a feature flag.
4. Update `@obelyzk/cli` npm package to match.
5. Update documentation to reflect actual commands.

**Acceptance criteria**:
- `obelysk prove --model ./smollm2-135m --input "Hello" --recursive` produces a proof file.
- `obelysk --help` shows only ZKML-relevant commands.
- No BitSage worker/validator commands in the default CLI.

---

### Item 11: Key Rotation and Access Control

**Description**: Rotate all testnet deployer keys. Add multisig or timelock to contract admin functions. Remove plaintext keys from all documentation and scripts.

**Why it matters**: The deployer private key is in plaintext in `docs/DEPLOYER_ACCOUNTS.md`. While this is testnet, it sets a bad precedent and would be a critical vulnerability if any mainnet deployment used the same key or procedure.

**Estimated effort**: 3-5 days

**Dependencies**: None.

**Work items**:
1. Generate a new deployer account (v3) with a fresh private key.
2. Fund the new account with testnet STRK.
3. Update all deployment scripts to read the private key from environment variable only (no hardcoded values).
4. Remove the plaintext private key from `docs/DEPLOYER_ACCOUNTS.md`. Replace with `<set via STARKNET_PRIVATE_KEY env var>`.
5. Document the key rotation procedure.
6. Evaluate multisig options for contract admin (e.g., Argent multisig on Starknet).
7. Update contract ownership to the new deployer.

**Acceptance criteria**:
- No plaintext private keys in any committed file (grep confirms).
- All scripts read keys from environment variables.
- Key rotation procedure documented.

---

### Item 12: Contract Upgrade Path for Recursive Verifier

**Description**: Add a timelock upgrade mechanism to the recursive verifier contract, mirroring the streaming contract's `propose_upgrade` -> delay -> `execute_upgrade` pattern.

**Why it matters**: If a bug is found in the recursive contract (which is the primary verification path), there is currently no way to fix it without deploying a new contract and migrating all model registrations. A timelock mechanism allows upgrades while giving users time to react.

**Estimated effort**: 3-5 days

**Dependencies**: Item 11 (key rotation) should be done first so the upgrade mechanism is controlled by a secure key.

**Work items**:
1. Add `propose_upgrade(new_class_hash)` to `recursive_verifier.cairo`.
2. Add `execute_upgrade()` with a 24-hour timelock (longer than the streaming contract's 5-minute timelock, since the recursive contract is higher stakes).
3. Add `cancel_upgrade()` for the owner to abort a pending upgrade.
4. Add `get_pending_upgrade()` view function so users can see if an upgrade is pending.
5. Emit events for all upgrade lifecycle stages.
6. Deploy the updated contract class.
7. Test the full upgrade cycle on Sepolia.

**Acceptance criteria**:
- `propose_upgrade` records pending upgrade with timestamp.
- `execute_upgrade` before 24 hours reverts.
- `execute_upgrade` after 24 hours succeeds and upgrades the contract class.
- `cancel_upgrade` clears pending upgrade.
- All stages emit events.

---

## Phase 3: Completeness (Week 9-12)

These items fill out the remaining gaps for a complete, auditable system.

---

### Item 13: Publish stwo-ml to crates.io

**Description**: Make `stwo-ml` installable via `cargo install stwo-ml` or usable as a dependency via `cargo add stwo-ml`.

**Why it matters**: Path dependencies on a local STWO fork prevent any Rust developer from using the library without cloning the entire monorepo. This is the biggest barrier to adoption for Rust developers.

**Estimated effort**: 3-5 days

**Dependencies**: None, but requires coordination with the STWO fork.

**Work items**:
1. Evaluate two approaches:
   - (a) Publish the STWO fork as `obelyzk-stwo` on crates.io, then reference it as a crates.io dependency.
   - (b) Convert path dependencies to git dependencies (`stwo = { git = "https://github.com/...", rev = "..." }`).
2. Option (b) is simpler and does not require maintaining a separate crate publication. Choose (b) unless there is a reason to prefer (a).
3. Update `Cargo.toml` to use git dependencies.
4. Verify `cargo publish --dry-run` succeeds.
5. Publish to crates.io.
6. Verify `cargo install stwo-ml` works on a clean machine.

**Acceptance criteria**:
- `cargo add stwo-ml` works.
- `cargo install stwo-ml` builds and installs the `prove-model` and `obelysk` binaries.
- The crate builds on stable Rust (or clearly documents the nightly requirement).

---

### Item 14: Activation/Norm Sub-Proof Verification in Recursive AIR

**Description**: Extend the recursive AIR to include LogUp activation proofs and LayerNorm/RMSNorm sub-proofs. This is the full implementation of Item 7's option (a) for all remaining soundness gates.

**Why it matters**: This closes the gap between "we verify MatMul" and "we verify the entire forward pass." Without this, the recursive STARK only proves that the GKR sumcheck chain was executed correctly, but does not verify that activation functions and normalization layers produced correct outputs.

**Estimated effort**: 2-3 weeks

**Dependencies**: Item 7 (assessment of each gate) should be completed first to prioritize which sub-proofs to include.

**Work items**:
1. Extend `InstrumentedChannel` to record activation LogUp operations.
2. Add `ActivationLogUpComponent` to the recursive AIR (similar to the existing `PoseidonChainComponent`).
3. Add `NormSubProofComponent` for RMSNorm/LayerNorm variance verification.
4. Update trace layout to accommodate new columns.
5. Update the Cairo recursive AIR to match (`recursive_air.cairo`).
6. Benchmark the recursive proving overhead with the additional components.
7. If overhead is acceptable (< 2x current), enable by default. If not, make it configurable.

**Acceptance criteria**:
- `--policy strict` produces recursive proofs that include activation and norm sub-proofs.
- These proofs verify on-chain.
- All soundness gates can be set to enforcing mode.
- The `strict` policy produces proofs where every operation in the forward pass is verified.

---

### Item 15: Mainnet Preparation

**Description**: Deploy contracts to Starknet mainnet. Implement gas estimation, fee market integration, and payment/billing for the hosted prover.

**Why it matters**: Sepolia is a testnet. Production verification requires mainnet deployment with real economic guarantees.

**Estimated effort**: 2-3 weeks

**Dependencies**: Items 1, 5, 6, 11, 12 should all be completed (security-critical items before mainnet).

**Work items**:
1. Audit all contract code one final time before mainnet declaration.
2. Deploy recursive verifier to Starknet mainnet.
3. Deploy deployer account on mainnet with proper key management (hardware wallet or multisig).
4. Implement gas estimation for proof submission (calldata cost + execution cost).
5. Add gas price monitoring and fee market integration to submission scripts.
6. Design and implement billing system for hosted prover (metered API keys, usage tracking).
7. Set up mainnet monitoring and alerting.

**Acceptance criteria**:
- Recursive verifier contract deployed on Starknet mainnet.
- At least one model registered and verified on mainnet.
- Gas costs documented and predictable.
- Billing system operational for hosted API.

---

### Item 16: Third-Party Audit

**Description**: Engage a security firm to audit four components: (a) the GKR prover soundness, (b) the recursive AIR constraints, (c) the Cairo contract, (d) the SDK/API attack surface.

**Why it matters**: Internal security audits are necessary but not sufficient. A third-party audit provides independent verification and is expected by institutional users and investors.

**Estimated effort**: 4-6 weeks (including remediation of findings)

**Dependencies**: Items 1-12 should be completed so the audit covers the production-ready codebase, not a moving target.

**Work items**:
1. Prepare audit scope document covering all four areas.
2. Engage 1-2 audit firms with STARK/ZK expertise (candidates: Trail of Bits, Zellic, OtterSec, Spearbit).
3. Provide auditors with documentation, test vectors, and access to a Sepolia deployment.
4. Triage findings into immediate/deferred.
5. Remediate all critical and high findings before public disclosure.
6. Publish audit report (or summary) on the project website.

**Acceptance criteria**:
- Audit report received covering all four scoped areas.
- All critical findings remediated.
- All high findings remediated or mitigated with documented timeline.
- Audit report published.

---

## Phase 4: Scale (Week 13-20)

These items extend the system's capabilities for larger models and production workloads.

---

### Item 17: Multi-GPU Proving

**Description**: Enable the `multi-gpu` feature for models with more than 1B parameters. The codebase already has `#[cfg(feature = "multi-gpu")]` blocks in `gkr/prover.rs` with greedy bin-packing partitioning and per-device guards. This item makes it production-ready.

**Why it matters**: Qwen3-14B takes 103s on a single H100. With 4xH100, the target is <60s. For production throughput, multi-GPU is essential.

**Estimated effort**: 2-3 weeks

**Dependencies**: CUDA runtime and multi-GPU hardware (4xH100 or equivalent).

**Work items**:
1. Benchmark current multi-GPU path on 4xH100 with Qwen3-14B.
2. Identify bottlenecks (device synchronization, memory transfer, load imbalance).
3. Optimize `multi_gpu::partition_by_size()` bin-packing for balanced GPU utilization.
4. Add device memory management (avoid OOM on individual GPUs).
5. Add multi-GPU CI tests (requires GPU CI runner).
6. Benchmark and document results.

**Acceptance criteria**:
- Qwen3-14B proves in <60s on 4xH100.
- Multi-GPU path produces identical proofs to single-GPU path.
- No GPU memory leaks across repeated proving.

---

### Item 18: Prefill Batch Proving

**Description**: Prove multiple inference requests in a single GKR proof via batched sumcheck. Each batch shares the same model weights but has different inputs/outputs.

**Why it matters**: Amortizes proving cost across multiple requests. If a single proof costs 3s, a batch of 8 should cost significantly less than 24s due to shared weight commitments and batched sumcheck rounds.

**Estimated effort**: 2-3 weeks

**Dependencies**: None (can proceed in parallel with Item 17).

**Work items**:
1. Extend `prove_model_pure_gkr` to accept a batch of input vectors.
2. Run forward pass for each input, collecting per-input intermediates.
3. Batch the MLE evaluations: weight MLEs are shared, activation MLEs are stacked.
4. Implement batched sumcheck: shared challenges across batch elements for weight terms.
5. Update IO commitment to cover all inputs/outputs in the batch.
6. Update recursive path to handle batched proofs.
7. Benchmark cost per request as a function of batch size.

**Acceptance criteria**:
- Batch of 8 inputs for SmolLM2-135M proves in <50% of 8 individual proofs.
- Each input in the batch has a separate IO commitment that can be independently verified.
- Recursive proof covers the entire batch.

---

### Item 19: KV-Cache Decode Proving

**Description**: Incremental proofs for autoregressive decoding. Each decode step proves one new token's inference against the existing KV-cache commitment chain.

**Why it matters**: LLM inference is autoregressive -- each token depends on all previous tokens. Proving a full 512-token generation as a single forward pass is prohibitively expensive. Incremental decode proofs amortize the cost to O(1) per token (after the initial prefill).

**Estimated effort**: 3-4 weeks

**Dependencies**: The KV-cache commitment chain infrastructure already exists (Poseidon binding, incremental Merkle, `AttentionDecode` variant). This item makes it production-ready and verified on-chain.

**Work items**:
1. Implement `prove_decode_step()` that takes: previous KV-cache commitment, new K/V vectors, query vector, and produces: updated KV-cache commitment + GKR proof for the decode attention.
2. Wire the `DCOD` tag through the streaming and recursive paths.
3. Implement on-chain verification of decode step proofs (chain of KV commitments).
4. Benchmark cost per decode step on A10G.
5. End-to-end test: prefill + 10 decode steps, all verified.

**Acceptance criteria**:
- Decode step proves in <0.5s on A10G.
- KV-cache commitment chain is verifiable on-chain.
- 10 sequential decode steps produce a chain of verified proofs.

---

### Item 20: MoE (Mixture of Experts)

**Description**: Support Mixtral-style Mixture of Experts routing with verifiable expert selection. The codebase already has MoE detection and Mixtral tensor name mapping in the HuggingFace loader. This item extends the GKR prover to handle expert routing.

**Why it matters**: MoE architectures are increasingly common (Mixtral, DeepSeek, etc.). Without MoE support, these models cannot be proven.

**Estimated effort**: 3-4 weeks

**Dependencies**: Item 17 (multi-GPU) is helpful since MoE models tend to be large.

**Work items**:
1. Implement expert routing verification: prove that the top-K expert selection (gating network output) is correct.
2. Implement per-expert forward pass proving: each selected expert is a standard transformer block proved via GKR.
3. Implement expert aggregation proving: prove that the weighted combination of expert outputs is correct.
4. Wire MoE through the recursive path.
5. End-to-end test with Mixtral-8x7B (or a smaller MoE model).

**Acceptance criteria**:
- Mixtral-style MoE model proves and verifies end-to-end.
- Expert selection is cryptographically verified (a prover cannot substitute different experts).
- Expert output aggregation is verified.

---

## Timeline Visualization

```
Week  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20
      [--- Phase 0 ---]
      [1: Weight binding ]
      [2: Self-verify bug]
         [3: Policy commit ]
         [4: SDK client     ]
            [--- Phase 1 ---]
            [5: Adversarial  ]
            [6: Cairo tests  ]
            [7: Soundness gates    ]
               [8: Multi-model]
                     [------- Phase 2 -------]
                     [9: Hosted API    ]
                     [10: CLI align    ]
                        [11: Key rotation]
                        [12: Upgrade path]
                              [-------- Phase 3 --------]
                              [13: crates.io  ]
                              [14: Activation AIR      ]
                                    [15: Mainnet prep   ]
                                    [16: Third-party audit       ]
                                                   [------- Phase 4 -------]
                                                   [17: Multi-GPU   ]
                                                   [18: Batch prove ]
                                                      [19: KV decode    ]
                                                         [20: MoE        ]
```

---

## Summary of Priorities

| Priority | Item | Risk if Skipped |
|----------|------|-----------------|
| P0 | Weight binding (Item 1) | System cannot bind proofs to models. Any proof accepted for any model. |
| P0 | Self-verify bug (Item 2) | Streaming fallback path is broken for large models. |
| P1 | Policy commitment (Item 3) | Proofs do not carry policy guarantees. |
| P1 | SDK client (Item 4) | Developer experience does not match documentation. |
| P1 | Adversarial testing (Item 5) | Unknown contract behavior under attack. |
| P1 | Cairo tests (Item 6) | No automated regression testing for trust anchor. |
| P1 | Soundness gates (Item 7) | Only MatMul is verified; activations/norms unverified. |
| P2 | Multi-model verification (Item 8) | "Any model" claim unsubstantiated. |
| P2 | Hosted API (Item 9) | SDK clients have no endpoint to connect to. |
| P2 | CLI alignment (Item 10) | User confusion, broken commands. |
| P2 | Key rotation (Item 11) | Plaintext keys in docs. Bad security posture. |
| P2 | Upgrade path (Item 12) | Cannot fix bugs in production contract. |
| P3 | crates.io (Item 13) | Adoption barrier for Rust developers. |
| P3 | Activation AIR (Item 14) | Soundness gap remains open. |
| P3 | Mainnet (Item 15) | System remains testnet-only. |
| P3 | Audit (Item 16) | No independent security validation. |
| P4 | Multi-GPU (Item 17) | Large models prove slowly. |
| P4 | Batch proving (Item 18) | Per-request cost not optimized. |
| P4 | KV decode (Item 19) | Cannot prove autoregressive generation. |
| P4 | MoE (Item 20) | Cannot prove Mixtral-family models. |

---

*This roadmap should be reviewed and updated every 2 weeks. Items may shift between phases based on customer/partner feedback and engineering velocity.*
