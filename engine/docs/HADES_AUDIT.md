# Hades Pairs — Independent Audit Procedure

The on-chain v4 recursive verifier stores a `level1_proof_hash` per model at
registration. This document describes how to independently verify it matches
the off-chain Hades pairs sidecar bytes.

This is the Hades Phase A guarantee: the chain STARK's `hades_commitment`
(proof header felt [18]) is bound to a specific set of (input, output) Hades
permutation pairs whose witness bytes hash to the on-chain registered value.

The chain STARK's AIR proves Hades correctness internally — Phase A's hash
binding does *not* re-validate the pairs via a separate cairo-prove run. That's
Phase B (Level-0 compressor), future work. Phase A is "the witness bytes are
publicly committed" — anyone can fetch + recompute and confirm match.

## What you need

- The `chain_manifest.json` from a decode-chain run.
- The per-step `chain_step_<N>.recursive.json.hades_args.json` sidecars.
- Network access to Sepolia (any RPC).
- Node 18+ (for `@noble/hashes` keccak256).

The proof directory contents are designed to be portable — anyone who pulls
`chain_manifest.json` and the sidecars (e.g., from IPFS, a release artifact,
or a GitHub repo) can audit independently.

## Procedure

### 1. Recompute the aggregate Hades hash

```bash
cd libs/engine
scripts/prove_hades_level1.sh <PATH_TO_PROOF_DIR>
```

This is a deterministic re-derivation of the same hash that was registered
on-chain. Output looks like:

```
  step 0: 511 felts, hash=0xa9f9a477e7a43c3f...
  step 1: 511 felts, hash=0xa9f9a477e7a43c3f...
aggregate keccak (raw 256-bit): 0x2a7cd5a47584758f6b8865caf2263d6fe1808130d2a170a45ac511d39955666c
aggregate keccak (felt252 truncated): 0x27cd5a47584758f6b8865caf2263d6fe1808130d2a170a45ac511d39955666c
```

The truncated value is what gets registered on-chain.

### 2. Fetch the on-chain registered hash

```bash
node -e "
const { hash, RpcProvider } = require('starknet');
const provider = new RpcProvider({ nodeUrl: 'https://starknet-sepolia-rpc.publicnode.com' });
const CONTRACT = '0x05736b0fb338a5de1e00f751bae3e2b65f0d8051952a5888d9cbf2f0a929e92a';
const MODEL_ID = '<your model_id from chain_manifest.json>';
provider.callContract({
  contractAddress: CONTRACT,
  entrypoint: 'get_level1_proof_hash',
  calldata: [MODEL_ID],
}).then(r => console.log('on-chain level1_proof_hash:', r[0]));
"
```

### 3. Compare

The recomputed aggregate hash from step 1 MUST equal the on-chain hash from
step 2. If they match, the off-chain sidecar bytes are exactly what was
registered on-chain.

If they don't match, one of:
- The sidecar bytes have been modified post-registration.
- The model registration used a different sidecar set.
- The hash algorithm in `prove_hades_level1.sh` was changed (check git blame).

### 4. Verify each step's `hades_commitment` matches the sidecar

The chain STARK's recursive proof includes `hades_commitment` at proof header
felt [18]. This is a Poseidon chain over the (input, output) Hades pairs:

```
commitment_0 = 0
commitment_{i+1} = Poseidon(commitment_i, output_i[0], 2)[0]
```

(Implementation: `libs/engine/src/recursive/prover.rs::compute_hades_commitment`.)

Anyone can re-run this Poseidon chain over the sidecar's pairs and compare
to the chain STARK's recorded hades_commitment. The on-chain verifier already
mixed this value into Fiat-Shamir during proof verification, so a tampered
sidecar would have caused proof rejection at submission time — this step is
mostly a sanity check that confirms sidecar ↔ proof bytes are consistent.

## What this audit *does* prove

- **Witness integrity**: the Hades pairs witness wasn't modified between
  proving time and now.
- **Public commitment**: any party can fetch the witness and reproduce the
  hash; this binds the chain STARK to a specific set of pairs.
- **Continuity within the chain**: pairs from each step's sidecar feed
  into the chain STARK's AIR, which checks digest consistency in-circuit.

## What this audit does *not* prove (yet)

- **Hades correctness**: the actual computation `hades_permutation(in) == out`
  for each pair is checked *inside* the chain STARK's AIR — this audit step
  doesn't independently re-run that check. Phase B (Level-0 compressor)
  adds an external cairo-prove proof of the same fact, accessible via a
  vanilla Cairo verifier.
- **Cross-prover compatibility**: the cairo-prove `hades_permutation` built-in
  uses different round constants than our recursive STARK's Hades. A vanilla
  Cairo verifier cannot validate our pairs without first reconciling these.
  Phase B work or a custom Cairo Hades verifier with our round constants
  will close this.

## Reference: Hades chain construction

The Poseidon hash used for `hades_commitment`:

```rust
// libs/engine/src/recursive/prover.rs
pub fn compute_hades_commitment(pairs: &[(Felt, Felt)]) -> FieldElement {
    let mut commitment = FieldElement::ZERO;
    for (_input, output) in pairs {
        let h = hades_permutation(commitment, output[0], FieldElement::from(2));
        commitment = h.0[0];
    }
    commitment
}
```

This commitment is what the chain STARK's AIR's row-by-row checks reduce to —
it's the "single felt that summarizes the entire Hades witness."

## Future work: Phase B (Level-0 compressor)

Today's hash-only audit closes the bytes-integrity gap. To close the
correctness gap fully on-chain:

1. Build a Cairo Hades verifier program that uses the **same** round constants
   as our prover's Hades.
2. Run cairo-prove to generate a Level-1 STARK proof over the pairs witness.
3. Recursively compress that ~278K-felt Level-1 proof into a ~5K-felt Level-0
   proof using a new `recursive_hades` Rust module (mirror of the existing
   `recursive` module).
4. Add a `verify_hades_level0(level0_proof, model_id)` entrypoint to the
   on-chain verifier that validates Level-0 and confirms its bound
   hades_commitment matches the chain STARK's `hades_commitment`.

The chain STARK + Level-0 then form a closed cryptographic loop:
- Chain STARK: "I ran the GKR verifier with hades_commitment X."
- Level-0 STARK: "I verified a cairo-prove proof that pairs producing X are
  correct under Cairo's Hades semantics."

Together they give: "the GKR verifier accepted on a model whose Hades
witness is independently provably correct."
