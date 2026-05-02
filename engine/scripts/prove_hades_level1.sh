#!/usr/bin/env bash
# prove_hades_level1.sh — Hades Phase A: auditable hash binding.
#
# For every `chain_step_<N>.recursive.json.hades_args.json` sidecar in PROOF_DIR,
# computes keccak256 over the canonical felt-concatenation. The resulting hash
# binds the published sidecar bytes to the on-chain registration's
# `level1_proof_hash` field. Anyone can independently:
#
#   1. Fetch the sidecar (published next to the proof).
#   2. Recompute the hash via this script's algorithm.
#   3. Confirm it matches `get_level1_proof_hash(model_id)` on-chain.
#   4. Verify the pairs in the sidecar reproduce the chain STARK's
#      `hades_commitment` (proof header felt [18]).
#
# Step 4 closes the loop: the chain STARK's `hades_commitment` is bound to the
# pairs whose witness bytes produce the on-chain `level1_proof_hash`. The pairs
# themselves are proven correct *inside* the chain STARK's AIR (not by an
# external cairo-prove proof — the built-in `hades_permutation` round constants
# don't match our prover's Hades, so secondary cairo-prove validation isn't
# trivially available). Phase B (Level-0 compressor) is the future work that
# adds independent on-chain Hades validation.
#
# Usage: prove_hades_level1.sh <PROOF_DIR>
#
# Side-effects:
#   - Writes <PROOF_DIR>/<step>.hades_level1.hash.txt (single line, 0x… keccak)
#   - Writes <PROOF_DIR>/level1_aggregate.hash.txt (overall hash for registration)
#   - Mutates <PROOF_DIR>/chain_manifest.json: sets `level1_proof_hash` field

set -euo pipefail

PROOF_DIR="${1:-${PROOF_DIR:-}}"
if [ -z "$PROOF_DIR" ] || [ ! -d "$PROOF_DIR" ]; then
  echo "Usage: $0 <PROOF_DIR>" >&2
  exit 1
fi

MANIFEST="$PROOF_DIR/chain_manifest.json"
if [ ! -f "$MANIFEST" ]; then
  echo "Missing chain_manifest.json in $PROOF_DIR" >&2
  exit 1
fi

# We use Node + @noble/hashes (already in libs/engine/node_modules) for keccak.
# Run from the engine dir so the require resolves.
ENGINE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ENGINE_DIR"

node <<NODE_SCRIPT
const fs = require('fs');
const path = require('path');
const { keccak_256 } = require('@noble/hashes/sha3');

const proofDir = '${PROOF_DIR}';
const manifestPath = path.join(proofDir, 'chain_manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));

const stepHashes = [];
for (const step of manifest.steps) {
  const sidecarPath = path.join(proofDir, step.proof_file + '.hades_args.json');
  if (!fs.existsSync(sidecarPath)) {
    console.error('Missing Hades sidecar for step ' + step.step_idx + ': ' + sidecarPath);
    process.exit(1);
  }

  // Canonical felt concatenation: each felt is encoded as 32-byte big-endian.
  // The sidecar's JSON formatting (whitespace, prefix-padding) doesn't matter —
  // only the felt values do. This makes the hash reproducible from any
  // implementation that can parse felt252 hex strings.
  const sidecar = JSON.parse(fs.readFileSync(sidecarPath, 'utf-8'));
  const concat = Buffer.alloc(sidecar.length * 32);
  for (let i = 0; i < sidecar.length; i++) {
    let h = sidecar[i].toString().toLowerCase();
    if (h.startsWith('0x')) h = h.slice(2);
    if (h.length > 64) {
      console.error('Felt too large in sidecar[' + i + ']: ' + h);
      process.exit(1);
    }
    h = h.padStart(64, '0');
    Buffer.from(h, 'hex').copy(concat, i * 32);
  }
  const hashBytes = keccak_256(concat);
  const hashHex = '0x' + Buffer.from(hashBytes).toString('hex');

  // Per-step hash file (audit artifact).
  fs.writeFileSync(
    path.join(proofDir, step.proof_file + '.hades_level1.hash.txt'),
    hashHex + '\n'
  );

  console.log('  step ' + step.step_idx + ': ' + sidecar.length + ' felts, hash=' + hashHex);
  stepHashes.push(hashHex);
}

// Aggregate hash = keccak256(concat(per-step hashes)).
// This is what we register on-chain as level1_proof_hash. Anyone audit can
// recompute it from the per-step sidecars and verify against the on-chain
// registration.
const aggregateInput = Buffer.alloc(stepHashes.length * 32);
for (let i = 0; i < stepHashes.length; i++) {
  Buffer.from(stepHashes[i].slice(2), 'hex').copy(aggregateInput, i * 32);
}
const aggregateBytes = keccak_256(aggregateInput);
const aggregateHex = '0x' + Buffer.from(aggregateBytes).toString('hex');

// felt252 fits 251 bits; truncate the keccak to 250 bits to be safe.
// Mask top byte with 0x03 (clears top 6 bits → max 250-bit value).
const safeBytes = Buffer.from(aggregateBytes);
safeBytes[0] = safeBytes[0] & 0x03;
const safeHex = '0x' + safeBytes.toString('hex').replace(/^0+/, '') || '0x0';
console.log('aggregate keccak (raw 256-bit): ' + aggregateHex);
console.log('aggregate keccak (felt252 truncated): ' + safeHex);

fs.writeFileSync(path.join(proofDir, 'level1_aggregate.hash.txt'), safeHex + '\n');

manifest.level1_proof_hash = safeHex;
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
console.log('chain_manifest.json updated with level1_proof_hash=' + safeHex);
NODE_SCRIPT
