# ObelyZK Threat Model

**Version**: 1.0
**Date**: April 6, 2026
**Status**: Prepared for third-party security audit engagement

---

## 1. Overview

ObelyZK is a zero-knowledge machine learning (ZKML) proving system that generates
cryptographic proofs of correct neural network inference. Given a registered model
(weights + circuit topology) and an input, the prover executes a forward pass in
the M31 prime field (p = 2^31 - 1) and produces a GKR (Goldwasser-Kalai-Rothblum)
interactive proof composed into a non-interactive STARK via Fiat-Shamir. The proof
is verified on-chain by a Cairo smart contract deployed on Starknet.

### Security Properties Claimed

1. **Inference Integrity**: A valid proof guarantees that the claimed output was
   produced by running the registered model on the claimed input. A malicious prover
   cannot forge a proof for an incorrect output except with negligible probability
   (2^{-128} soundness via QM31 extension field).

2. **Model Binding**: Each proof is cryptographically bound to a specific model via
   a Poseidon Merkle root over all weight matrices (the "weight super root") and a
   Poseidon hash of the circuit topology (the "circuit hash"). Substituting a
   different model requires breaking Poseidon2 collision resistance.

3. **IO Commitment**: Inputs and outputs are committed via a packed Poseidon hash
   (8 M31 elements per felt252) that is mixed into the Fiat-Shamir channel, binding
   the proof to specific input/output values.

4. **Replay Prevention**: The on-chain verifier stores proof hashes in contract
   storage and rejects duplicate submissions.

5. **Policy Binding**: Proof generation parameters (soundness gates, weight binding
   mode, packing format) are captured in a `PolicyConfig` struct whose Poseidon hash
   is mixed into the Fiat-Shamir channel, preventing proofs generated under relaxed
   policies from being accepted by strict verifiers.

---

## 2. Adversary Model

### 2.1 Malicious Prover

**Goal**: Generate a proof that a model produced output Y on input X, when in fact
the model would produce a different output Y' on input X (or any output on a
different input).

**Capabilities**: Full control over the proving process. Can modify prover code,
choose arbitrary inputs to cryptographic functions, and submit any data as a "proof"
to the on-chain verifier. Cannot modify the on-chain verifier contract or the
registered model parameters (circuit hash, weight super root).

**Threat Level**: HIGH. This is the primary adversary the system is designed to
resist.

### 2.2 Malicious Model Operator

**Goal**: Register a model with one set of weights or circuit topology but actually
prove inference using a different (possibly weaker or backdoored) model.

**Capabilities**: Controls the model registration transaction. Can register
arbitrary circuit hashes and weight super roots. May attempt to register a model
whose weights differ from what users expect, or register a simpler circuit that
omits security-critical layers (e.g., skip normalization).

**Threat Level**: MEDIUM. Mitigated by public model registration and the ability for
users to independently verify registered parameters against known model
checkpoints.

### 2.3 Network Observer

**Goal**: Extract model weights, private inputs, or other sensitive information from
proofs, on-chain transactions, or network traffic.

**Capabilities**: Can observe all on-chain transactions, submitted proofs, and
network traffic between client and prove-server. Cannot break standard
cryptographic assumptions.

**Threat Level**: LOW-MEDIUM. STARK proofs are not zero-knowledge by default.
Inputs and outputs are committed but may be partially leaked through proof
structure. Model weights are committed via Merkle roots but weight values flow
through the GKR protocol as MLE evaluations.

### 2.4 Denial of Service

**Goal**: Prevent legitimate users from generating or verifying proofs.

**Capabilities**: Can flood the prove-server with requests, submit invalid proofs to
waste verifier gas, or attempt to exhaust prover resources.

**Threat Level**: MEDIUM. Proving is computationally expensive (seconds to minutes
per inference), making resource exhaustion attacks practical.

---

## 3. Attack Vectors

### 3.1 Proof Forgery (Malicious Prover)

**3.1.1 Forge STARK proof**: Construct a STARK proof that passes the FRI, OODS, and
Merkle verification without performing the actual computation.

- **Mitigation**: STWO STARK provides 128-bit soundness via the QM31 extension
  field. Forging requires finding collisions in the Merkle commitment scheme or
  inverting the FRI protocol, both of which are computationally infeasible.
- **Residual risk**: Depends on the correctness of STWO's STARK implementation,
  which is an external dependency.

**3.1.2 Manipulate Fiat-Shamir transcript**: Selectively reorder or omit channel
mixing steps to steer verifier challenges toward favorable values.

- **Mitigation**: The Fiat-Shamir channel is instantiated with a Poseidon2-based
  `PoseidonChannel` that absorbs all public parameters (circuit hash, weight super
  root, IO commitment, policy commitment) in a fixed order. Both Rust and Cairo
  verifiers must produce identical channel states.
- **Residual risk**: If the Rust and Cairo channel implementations diverge (e.g.,
  different mixing order), a prover could exploit the discrepancy.

**3.1.3 GKR claim manipulation**: Submit incorrect intermediate claims between GKR
layers (e.g., wrong MatMul output, fabricated activation values).

- **Mitigation**: The GKR protocol reduces each layer's claim to the previous
  layer via sumcheck. The verifier independently evaluates the multilinear
  extension at challenge points. Any inconsistency causes sumcheck verification
  to fail.
- **Residual risk**: Soundness gates (Section 3.6) may allow certain intermediate
  values to go unchecked.

### 3.2 Weight Substitution (Malicious Model Operator)

**3.2.1 Register model A, prove model B**: Register the circuit hash and weight
super root of a legitimate model, but use different weights during proving.

- **Mitigation**: The aggregated weight binding protocol computes an RLC
  (random linear combination) of all weight MLE evaluations at verifier-chosen
  challenge points, then verifies the RLC against Merkle opening proofs rooted at
  the registered weight super root. Substituting weights would require either
  finding a Poseidon collision or producing a valid MLE opening at the challenge
  point, both computationally infeasible.
- **Residual risk**: If `STWO_AGGREGATED_RLC_ONLY=1` is set (no full MLE openings),
  the binding is weaker. The streaming verifier rejects RLC-only proofs as of
  March 2026.

**3.2.2 Circuit substitution**: Register a circuit that omits layers (e.g., drop
normalization or activation functions) to make forgery easier.

- **Mitigation**: The circuit hash commits to the full `LayeredCircuit` topology
  including all layer types, dimensions, and ordering. Omitting layers changes the
  hash.
- **Residual risk**: Users must independently verify that the registered circuit
  hash corresponds to the expected model architecture. No on-chain registry of
  "known-good" circuit hashes exists.

### 3.3 IO Manipulation (Malicious Prover)

**3.3.1 Claim wrong input/output commitment**: Submit a proof with a valid
computation trace but bind it to different input/output values than what was
actually computed.

- **Mitigation**: The IO commitment is a packed Poseidon hash of the input and
  output tensors (8 M31 per felt252). This commitment is mixed into the
  Fiat-Shamir channel at proof generation time. Changing the IO values would
  produce a different commitment, leading to different verifier challenges and
  proof verification failure.
- **Known limitation**: The IO commitment parameter passed to the on-chain
  verifier and the IO values embedded in the proof header use different encodings.
  The on-chain verifier does not currently cross-check these two representations
  directly (see Section 7).

### 3.4 Replay Attack (Malicious Prover)

**3.4.1 Resubmit old proof**: Take a valid proof from a previous inference and
submit it as if it were a new computation.

- **Mitigation**: The on-chain verifier computes a `proof_hash` (Poseidon hash of
  the full calldata) and stores it in contract storage. Duplicate `proof_hash`
  values are rejected.
- **Residual risk**: If the contract is redeployed or storage is reset, old proofs
  could be replayed. Cross-contract replay is possible if multiple verifier
  instances do not share a deduplication registry.

### 3.5 Soundness Gate Bypass (Malicious Prover)

**3.5.1 Exploit permissive environment variables**: Generate a proof under relaxed
policy settings (e.g., `STWO_ALLOW_MISSING_NORM_PROOF=1`) and submit it to a
verifier that accepts such proofs.

- **Mitigation**: The `PolicyConfig` commitment is mixed into the Fiat-Shamir
  channel. A proof generated under a relaxed policy will have a different channel
  state than one generated under a strict policy, causing verification failure
  under the strict policy.
- **Caveat**: If the on-chain verifier does not enforce policy commitment
  checking (e.g., legacy contract versions), a relaxed-policy proof could be
  accepted. The `STWO_SKIP_POLICY_COMMITMENT` environment variable, if set,
  disables policy checking in the Rust verifier.

**3.5.2 Soundness gates that weaken verification**:

| Gate | Default | Effect When Open |
|------|---------|------------------|
| `STWO_ALLOW_MISSING_NORM_PROOF` | `false` | Accepts proofs without LayerNorm/RMSNorm sub-proofs |
| `STWO_PIECEWISE_ACTIVATION` | `true` | When `false`, activation verification covers only lower 16-20 bits |
| `STWO_ALLOW_LOGUP_ACTIVATION` | `false` | Accepts proofs without full activation LogUp proofs |
| `STWO_SKIP_BATCH_TOKENS` | `false` | Skips batch token accumulation verification |

See `docs/SOUNDNESS_GATES_AUDIT.md` for detailed analysis of each gate.

### 3.6 Side-Channel Attacks (Network Observer)

**3.6.1 Timing attacks on prover**: Observe proving time to infer model complexity
or input characteristics.

- **Mitigation**: None implemented. Proving time is correlated with model size and
  is observable by network observers.
- **Residual risk**: Low. Model architecture is typically public. Input-dependent
  timing variation is minimal since all operations are over fixed-size tensors.

**3.6.2 Weight extraction from proofs**: Analyze proof structure to recover model
weights.

- **Mitigation**: Weight values appear only as Poseidon Merkle commitments and MLE
  evaluations at random challenge points. Recovering the full weight matrix from a
  polynomial number of evaluation points requires solving a system with
  exponentially many unknowns.
- **Residual risk**: Multiple proofs with different challenge points could in theory
  leak partial weight information. Formal analysis of weight privacy under the GKR
  protocol is an open research question.

---

## 4. Trust Boundaries

### 4.1 Prover (Untrusted) <-> Verifier (On-Chain, Trusted)

The core trust boundary. The prover is fully untrusted and may be adversarial. The
on-chain Cairo verifier is the root of trust. All security properties ultimately
depend on the verifier correctly rejecting invalid proofs.

- **Data crossing boundary**: Serialized proof (felt252 array), model ID,
  IO commitment, PCS configuration.
- **Serialization format**: Rust `cairo_serde.rs` encodes proof elements as
  felt252 values; Cairo deserializes and verifies.
- **Critical invariant**: The Rust serializer and Cairo deserializer must be
  perfectly aligned. Any mismatch could cause valid proofs to be rejected or
  invalid proofs to be accepted.

### 4.2 Client (Untrusted) <-> Prove-Server (Semi-Trusted)

The prove-server accepts inference requests and returns proofs. It is semi-trusted:
it could refuse to generate proofs (liveness failure) but should not be able to
generate invalid proofs (the on-chain verifier catches those).

- **Data crossing boundary**: Model identifier, input tensor, proof parameters.
- **Trust assumption**: The prove-server runs the correct prover code. A
  compromised prove-server could return invalid proofs, but these would be rejected
  on-chain.

### 4.3 Off-Chain Computation <-> On-Chain Verification

All computation (forward pass, GKR proving, STARK composition) happens off-chain.
Only the final proof and public inputs are submitted on-chain.

- **Data crossing boundary**: Proof calldata, gas payment.
- **Trust assumption**: Starknet sequencer processes transactions correctly and
  does not censor verification transactions.

### 4.4 Model Registry <-> Proof Verification

The on-chain model registry stores `(model_id, circuit_hash, weight_super_root)`
tuples. The verifier reads these during proof verification.

- **Trust assumption**: Only the contract owner can register models
  (`owner_only` access control). The registry is append-only; models cannot be
  unregistered or modified after registration.

---

## 5. Security Assumptions

### 5.1 Cryptographic Assumptions

- **Poseidon2 hash function**: Preimage resistance and collision resistance over
  the Starknet field (felt252). Used for Merkle commitments, Fiat-Shamir channel,
  IO commitments, policy commitments, and weight super root computation.

- **M31/QM31 field arithmetic**: Correctness of arithmetic operations in the
  Mersenne-31 field (p = 2^31 - 1) and its degree-4 extension QM31. All GKR
  sumcheck evaluations, MLE computations, and FRI operations rely on field
  arithmetic being correct.

- **STWO STARK soundness**: The STWO proving system provides 128-bit security via
  the QM31 extension field, assuming correct implementation of FRI (Fast
  Reed-Solomon Interactive Oracle Proofs), OODS (Out-of-Domain Sampling), Merkle
  commitment scheme, and proof-of-work grinding resistance. The upgraded recursive
  STARK (v2) provides 160-bit security via PcsConfig (pow_bits=20, log_blowup=5,
  n_queries=28, log_last_layer_deg=0) and is protected by 9 independent security
  layers including amortized accumulator constraints, carry-chain modular addition,
  hades_commitment binding for two-level recursion, and boundary constraints.

### 5.2 Infrastructure Assumptions

- **Starknet sequencer liveness**: The L1/L2 sequencer must be live for proofs to
  be verified on-chain. Sequencer downtime prevents verification but does not
  compromise soundness.

- **Cairo VM determinism**: The Cairo VM executes contract code deterministically.
  The same proof calldata always produces the same verification result.

- **Contract immutability**: Once deployed, the verifier contract code cannot be
  modified except through the upgrade mechanism (propose + timelock + execute).

### 5.3 Operational Assumptions

- **Deployer key security**: The contract deployer key controls model registration
  and contract upgrades. Compromise of this key allows registering malicious models
  or upgrading to a weak verifier.

- **Environment variable discipline**: Production deployments must use
  `PolicyConfig::strict()` or equivalent. Relaxed environment variables weaken
  soundness.

---

## 6. Mitigations

### 6.1 Circuit Hash Binding

The `LayeredCircuit` structure (layer types, dimensions, activation functions,
connectivity) is hashed via Poseidon to produce a deterministic `circuit_hash`. This
hash is mixed into the Fiat-Shamir channel and stored in the on-chain model
registry. Any modification to the circuit topology produces a different hash.

**Implementation**: `src/gkr/circuit.rs` computes the circuit hash.
`src/starknet.rs` serializes it into calldata. The Cairo verifier reads it from the
model registry and mixes it into its channel.

### 6.2 Weight Super Root Binding

All weight matrices are committed via a Poseidon Merkle tree. Per-layer weight roots
are combined into a single "weight super root" that is mixed into the Fiat-Shamir
channel. The aggregated weight binding protocol (Mode 4) verifies that weight MLE
evaluations at random challenge points are consistent with the committed Merkle
roots via batched opening proofs.

**Implementation**: `src/crypto/poseidon_merkle.rs` for Merkle construction.
`src/crypto/aggregated_opening.rs` for batched MLE opening proofs.
`src/gkr/prover.rs` for aggregated oracle sumcheck.

### 6.3 IO Commitment

Input and output tensors are serialized as packed felt252 values (8 M31 elements
per felt) and hashed via Poseidon. This IO commitment is mixed into the Fiat-Shamir
channel and passed as a parameter to the on-chain verifier.

**Implementation**: `src/aggregation.rs::compute_io_commitment_packed()` on the
Rust side. The Cairo verifier recomputes the commitment from the IO span in calldata.

### 6.4 Policy Commitment in Fiat-Shamir Channel

The `PolicyConfig` struct captures all proof generation parameters. Its Poseidon
hash (`policy_commitment()`) is mixed into the Fiat-Shamir channel after KV-cache
binding and before circuit metadata. Both the Rust and Cairo verifiers mix at the
same channel position.

**Implementation**: `src/policy.rs` defines the struct, presets, and hash
computation. Domain separator `0x504F4C01` ("POL" + version 0x01) is prepended.

### 6.5 Replay Prevention

The on-chain verifier computes a `proof_hash` from the submitted calldata and checks
it against contract storage. Previously verified proof hashes are rejected.

**Implementation**: Cairo contract storage mapping. The `proof_hash` is computed
as a Poseidon hash of the full calldata array.

### 6.6 Proof-of-Work Grinding Resistance

The STARK proof includes a proof-of-work nonce. The verifier requires `pow_bits >= 10`
(configurable via PCS config), making it computationally expensive for an adversary
to grind through many candidate proofs searching for one that passes verification
by chance.

**Implementation**: STWO's PCS configuration. The `pow_bits` value is serialized in
calldata and checked by the Cairo verifier.

### 6.7 Owner-Only Model Registration

Only the contract owner address can call `register_model()` to add new model entries
to the on-chain registry. This prevents adversaries from registering malicious models
that users might mistakenly trust.

**Implementation**: Cairo contract access control (`assert(caller == owner)`).

---

## 7. Known Limitations

### 7.1 Open Soundness Gates

Four soundness gates remain open in certain deployment configurations. Each gate,
when permissive, allows proofs that skip certain verification steps:

1. **`STWO_ALLOW_MISSING_NORM_PROOF`** (default: `false`): When set to `true`,
   accepts proofs without LayerNorm/RMSNorm sub-proofs. The prover does not yet
   generate full norm binding proofs in all code paths, so some deployments
   require this gate open.

2. **`STWO_PIECEWISE_ACTIVATION`** (default: `true`): When set to `false`, falls
   back to legacy LogUp activation proofs that only verify the lower 16-20 bits of
   activation function outputs. Upper bits (20-30) of M31 values remain unverified,
   allowing a malicious prover to produce activation outputs that are correct only
   in lower bits.

3. **`STWO_ALLOW_LOGUP_ACTIVATION`** (default: `false`): When set to `true`,
   accepts proofs without full LogUp activation proofs. Coupled with the piecewise
   gate above.

4. **`STWO_SKIP_BATCH_TOKENS`** (default: `false`): When set to `true`, skips
   batch token accumulation proofs. Affects multi-token inference; single-token
   inference is unaffected.

See `docs/SOUNDNESS_GATES_AUDIT.md` for the full audit of each gate.

### 7.2 IO Commitment Encoding Mismatch

The IO commitment parameter passed to the on-chain verifier and the IO values
embedded in the proof header use different internal encodings (packed felt252 vs.
raw M31 arrays). The on-chain verifier does not currently cross-check these two
representations directly. A prover could theoretically submit a valid proof with
one set of IO values while passing a different IO commitment parameter, though
Fiat-Shamir binding provides indirect protection.

### 7.3 Single Deployer Key

The contract deployer/owner is a single externally owned account (EOA). There is
no multisig, governance, or social recovery mechanism. Compromise of this key
allows:
- Registering arbitrary models
- Proposing contract upgrades to malicious code
- (After timelock) executing upgrades

### 7.4 Contract Upgrade Timelock on Recursive Verifier

Both recursive verifier contracts support upgrades via a propose-then-execute pattern
with a timelock:

- **v2 contract** (`0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005`):
  Production 48-column chain AIR with 38 constraints and 160-bit security. Two-level
  recursion architecture. Includes upgrade timelock mechanism.

- **v1 contract** (`0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7`):
  Original 28-column/27-constraint design. Has 5-minute timelock matching the
  streaming GKR verifier.

The timelock duration may be insufficient for the community to detect and respond to
a malicious upgrade proposal.

### 7.5 Activation Function Partial Verification

When piecewise activation mode is disabled (legacy path), activation function outputs
(GELU, SiLU, Sigmoid, Softmax) are verified only in their lower 16-20 bits via
range-reduced LogUp tables. The upper bits of the M31 field element are not
constrained, allowing a malicious prover to produce activation outputs that differ
from the true function value by multiples of 2^16 to 2^20.

### 7.6 Weight Privacy

The GKR protocol does not provide zero-knowledge guarantees. MLE evaluations of
weight matrices at verifier-chosen challenge points are revealed in the proof. While
recovering full weight matrices from a small number of evaluation points is
computationally infeasible, accumulating evaluations across many proofs could leak
partial information about model weights. Applications requiring weight
confidentiality should consider additional protections.

### 7.7 Proof Size and Gas

The production recursive STARK (v2) proofs are approximately 4,934 felts for a 30-layer
model due to the 48-column chain AIR with 38 constraints and two-level recursion. The original
v1 system compressed to approximately 942-981 felts. Both versions verify in a single
Starknet transaction. On-chain verification gas costs remain significant. Extremely
large models may approach the Starknet sequencer step limit (10M steps), though the
constant-size property ensures model size does not affect verification cost.

### 7.8 Recursive STARK Security Layers

The v2 recursive STARK system uses 9 independent security layers. A failure in
any single layer does not compromise the overall system, but the effectiveness of
certain layers depends on the correctness of others:

1. **Fiat-Shamir channel binding** depends on Poseidon2 collision resistance
2. **Amortized accumulator** is unconditional (no selector dependency)
3. **n_poseidon_perms validation** depends on correct trace size reporting
4. **seed_digest checkpoint** depends on Fiat-Shamir binding
5. **pass1_final_digest binding** depends on correct witness generation
6. **Carry-chain modular addition** depends on correct M31 arithmetic
7. **hades_commitment binding** depends on Level 1 cairo-prove integrity (two-level recursion)
8. **Boundary constraints** depend on correct initial/final digest computation
9. **160-bit STARK security** depends on PcsConfig enforcement (pow=20, blowup=5, queries=28)

The two-level recursion architecture strengthens security: Level 1 (cairo-prove)
verifies 145 Hades permutations off-chain, and Level 2 (chain STARK) binds to the
Level 1 commitment on-chain. All 9 layers provide on-chain or cryptographic protection.
