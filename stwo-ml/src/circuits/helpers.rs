//! Permutation-recording helpers for composed circuit proofs.
//!
//! These functions replicate `poseidon2_hash`, `poseidon2_compress`, Merkle verification,
//! and key derivation — but record every permutation's `(input_state, output_state)` pair.
//! This is needed because `prove_poseidon2_batch` operates on raw `[M31; 16]` I/O.

use stwo::core::fields::m31::BaseField as M31;

use crate::crypto::merkle_m31::{Digest, MerklePath};
use crate::crypto::poseidon2_m31::{poseidon2_permutation, RATE, STATE_WIDTH};

/// A recorded permutation: (input_state, output_state).
pub type PermRecord = ([M31; STATE_WIDTH], [M31; STATE_WIDTH]);

/// Hash variable-length M31 input, recording every permutation.
///
/// Matches `poseidon2_hash` exactly (poseidon2_m31.rs:211-234):
/// 1. `state = [0; 16]`, `state[RATE] = input.len()` (domain separation)
/// 2. For each chunk of 8: `state[i] += chunk[i]`, snapshot input, permute, snapshot output
/// 3. Empty input: still 1 permutation
/// 4. Return `state[..8]`
pub fn record_hash_permutations(input: &[M31]) -> ([M31; RATE], Vec<PermRecord>) {
    let mut state = [M31::from_u32_unchecked(0); STATE_WIDTH];
    let mut perms = Vec::new();

    // Domain separation: encode input length in first capacity element
    state[RATE] = M31::from_u32_unchecked(input.len() as u32);

    // Absorb phase: process input in chunks of RATE
    for chunk in input.chunks(RATE) {
        for (i, &val) in chunk.iter().enumerate() {
            state[i] += val;
        }
        let input_snapshot = state;
        poseidon2_permutation(&mut state);
        perms.push((input_snapshot, state));
    }

    // For empty input, still apply one permutation
    if input.is_empty() {
        let input_snapshot = state;
        poseidon2_permutation(&mut state);
        perms.push((input_snapshot, state));
    }

    // Squeeze: return rate portion
    let mut output = [M31::from_u32_unchecked(0); RATE];
    output.copy_from_slice(&state[..RATE]);
    (output, perms)
}

/// 2-to-1 compression, recording the single permutation.
///
/// Matches `poseidon2_compress` (poseidon2_m31.rs:252-261):
/// `state[..8] = left`, `state[8..] = right`, permute, return `state[..8]`.
pub fn record_compress_permutation(
    left: &[M31; RATE],
    right: &[M31; RATE],
) -> ([M31; RATE], PermRecord) {
    let mut state = [M31::from_u32_unchecked(0); STATE_WIDTH];
    state[..RATE].copy_from_slice(left);
    state[RATE..].copy_from_slice(right);

    let input_snapshot = state;
    poseidon2_permutation(&mut state);
    let record = (input_snapshot, state);

    let mut output = [M31::from_u32_unchecked(0); RATE];
    output.copy_from_slice(&state[..RATE]);
    (output, record)
}

/// Verify a Merkle path, recording all compress permutations.
///
/// Matches `verify_merkle_proof` (merkle_m31.rs:143-158):
/// `index & 1 == 0` → current is left child.
pub fn record_merkle_permutations(leaf: &Digest, path: &MerklePath) -> (Digest, Vec<PermRecord>) {
    let mut current = *leaf;
    let mut index = path.index;
    let mut perms = Vec::with_capacity(path.siblings.len());

    for sibling in &path.siblings {
        let (result, record) = if index & 1 == 0 {
            // Current node is left child
            record_compress_permutation(&current, sibling)
        } else {
            // Current node is right child
            record_compress_permutation(sibling, &current)
        };
        current = result;
        perms.push(record);
        index >>= 1;
    }

    (current, perms)
}

/// Derive public key from spending key, recording the hash permutation.
///
/// Matches `derive_pubkey` (commitment.rs:94-105):
/// `DOMAIN_SPEND = 0x766D3331`, input = `[DOMAIN_SPEND, sk[0..4]]`.
pub fn record_ownership_permutations(spending_key: &[M31; 4]) -> ([M31; 4], Vec<PermRecord>) {
    const DOMAIN_SPEND: M31 = M31::from_u32_unchecked(0x766D3331);
    let input = [
        DOMAIN_SPEND,
        spending_key[0],
        spending_key[1],
        spending_key[2],
        spending_key[3],
    ];
    let (hash, perms) = record_hash_permutations(&input);
    ([hash[0], hash[1], hash[2], hash[3]], perms)
}

/// Verify sponge chain continuity between consecutive permutations of a multi-perm hash.
///
/// For an N-element hash requiring `perm_count` permutations, checks that between
/// consecutive permutations, the unchained state positions are preserved:
/// - Rate positions that don't receive new absorbed data
/// - All capacity positions
pub fn verify_sponge_chain(
    perm_inputs: &[[M31; STATE_WIDTH]],
    perm_outputs: &[[M31; STATE_WIDTH]],
    start_idx: usize,
    perm_count: usize,
    total_elements: usize,
) -> Result<(), String> {
    if perm_count <= 1 {
        return Ok(()); // single-perm hash has no chain to verify
    }
    for p in 0..perm_count - 1 {
        let out_idx = start_idx + p;
        let in_idx = start_idx + p + 1;

        // How many elements does chunk (p+1) absorb?
        let remaining = total_elements.saturating_sub((p + 1) * RATE);
        let k = remaining.min(RATE);

        // Unchained rate positions: k..RATE must match previous output
        for j in k..RATE {
            if perm_inputs[in_idx][j] != perm_outputs[out_idx][j] {
                return Err(format!(
                    "sponge chain broken at perm {}→{}: rate position {} (got {}, expected {})",
                    p,
                    p + 1,
                    j,
                    perm_inputs[in_idx][j].0,
                    perm_outputs[out_idx][j].0
                ));
            }
        }

        // Capacity positions: RATE..STATE_WIDTH must match previous output
        for j in RATE..STATE_WIDTH {
            if perm_inputs[in_idx][j] != perm_outputs[out_idx][j] {
                return Err(format!(
                    "sponge chain broken at perm {}→{}: capacity position {} (got {}, expected {})",
                    p,
                    p + 1,
                    j,
                    perm_inputs[in_idx][j].0,
                    perm_outputs[out_idx][j].0
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::commitment::derive_pubkey;
    use crate::crypto::merkle_m31::{verify_merkle_proof, PoseidonMerkleTreeM31};
    use crate::crypto::poseidon2_m31::{poseidon2_compress, poseidon2_hash};

    #[test]
    fn test_record_hash_matches_poseidon2_hash() {
        // Test multiple input lengths: empty, 1, 5, 8 (exact), 11 (multi-chunk), 16
        let test_cases: Vec<Vec<M31>> = vec![
            vec![],
            vec![M31::from_u32_unchecked(42)],
            (1..=5).map(|i| M31::from_u32_unchecked(i)).collect(),
            (1..=8).map(|i| M31::from_u32_unchecked(i)).collect(),
            (1..=11).map(|i| M31::from_u32_unchecked(i)).collect(),
            (1..=16).map(|i| M31::from_u32_unchecked(i)).collect(),
        ];

        for input in &test_cases {
            let expected = poseidon2_hash(input);
            let (result, _perms) = record_hash_permutations(input);
            assert_eq!(
                result,
                expected,
                "hash mismatch for input len {}",
                input.len()
            );
        }
    }

    #[test]
    fn test_record_hash_permutation_count() {
        // empty → 1 perm
        let (_, perms) = record_hash_permutations(&[]);
        assert_eq!(perms.len(), 1);

        // 1..=8 → 1 perm (fits in one RATE chunk)
        for len in 1..=8 {
            let input: Vec<M31> = (0..len)
                .map(|i| M31::from_u32_unchecked(i as u32))
                .collect();
            let (_, perms) = record_hash_permutations(&input);
            assert_eq!(perms.len(), 1, "expected 1 perm for len {len}");
        }

        // 9..=16 → 2 perms
        for len in 9..=16 {
            let input: Vec<M31> = (0..len)
                .map(|i| M31::from_u32_unchecked(i as u32))
                .collect();
            let (_, perms) = record_hash_permutations(&input);
            assert_eq!(perms.len(), 2, "expected 2 perms for len {len}");
        }

        // 11 → 2 perms (commitment hash: 8 + 3)
        let input: Vec<M31> = (0..11).map(|i| M31::from_u32_unchecked(i)).collect();
        let (_, perms) = record_hash_permutations(&input);
        assert_eq!(perms.len(), 2);

        // 12 → 2 perms (nullifier hash: 8 + 4)
        let input: Vec<M31> = (0..12).map(|i| M31::from_u32_unchecked(i)).collect();
        let (_, perms) = record_hash_permutations(&input);
        assert_eq!(perms.len(), 2);
    }

    #[test]
    fn test_record_compress_matches_poseidon2_compress() {
        let left = [1, 2, 3, 4, 5, 6, 7, 8].map(M31::from_u32_unchecked);
        let right = [9, 10, 11, 12, 13, 14, 15, 16].map(M31::from_u32_unchecked);

        let expected = poseidon2_compress(&left, &right);
        let (result, _record) = record_compress_permutation(&left, &right);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_record_merkle_matches_verify_merkle_proof() {
        // Build a depth-4 tree with 8 leaves
        let mut tree = PoseidonMerkleTreeM31::new(4);
        let leaves: Vec<Digest> = (1..=8)
            .map(|v| {
                let mut d = [M31::from_u32_unchecked(0); RATE];
                d[0] = M31::from_u32_unchecked(v);
                d
            })
            .collect();

        for leaf in &leaves {
            tree.append(*leaf);
        }

        let root = tree.root();

        // Verify each leaf's proof matches
        for (i, leaf) in leaves.iter().enumerate() {
            let path = tree.prove(i);
            assert!(verify_merkle_proof(&root, leaf, &path));

            let (computed_root, perms) = record_merkle_permutations(leaf, &path);
            assert_eq!(computed_root, root, "Merkle root mismatch for leaf {i}");
            assert_eq!(perms.len(), 4, "depth-4 tree should have 4 compress perms");
        }
    }

    #[test]
    fn test_record_ownership_matches_derive_pubkey() {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let expected = derive_pubkey(&sk);
        let (result, perms) = record_ownership_permutations(&sk);
        assert_eq!(result, expected);
        // 5-element input → 1 perm (fits in one RATE chunk)
        assert_eq!(perms.len(), 1);
    }

    #[test]
    fn test_verify_sponge_chain_valid() {
        // Commitment hash: 11 elements → 2 perms. Chain should be valid.
        let input: Vec<M31> = (1..=11).map(|i| M31::from_u32_unchecked(i)).collect();
        let (_, perms) = record_hash_permutations(&input);

        let inputs: Vec<[M31; STATE_WIDTH]> = perms.iter().map(|p| p.0).collect();
        let outputs: Vec<[M31; STATE_WIDTH]> = perms.iter().map(|p| p.1).collect();

        verify_sponge_chain(&inputs, &outputs, 0, 2, 11).expect("valid sponge chain should pass");
    }

    #[test]
    fn test_verify_sponge_chain_broken_capacity() {
        let input: Vec<M31> = (1..=11).map(|i| M31::from_u32_unchecked(i)).collect();
        let (_, perms) = record_hash_permutations(&input);

        let mut inputs: Vec<[M31; STATE_WIDTH]> = perms.iter().map(|p| p.0).collect();
        let outputs: Vec<[M31; STATE_WIDTH]> = perms.iter().map(|p| p.1).collect();

        // Break capacity position 10 in perm1
        inputs[1][10] = M31::from_u32_unchecked(999999);

        let result = verify_sponge_chain(&inputs, &outputs, 0, 2, 11);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("capacity position 10"));
    }

    #[test]
    fn test_verify_sponge_chain_broken_rate() {
        // Nullifier: 12 elements → 2 perms. Chunk 1 absorbs 4 elements (k=4).
        // Rate positions 4..8 should be unchained.
        let input: Vec<M31> = (1..=12).map(|i| M31::from_u32_unchecked(i)).collect();
        let (_, perms) = record_hash_permutations(&input);

        let mut inputs: Vec<[M31; STATE_WIDTH]> = perms.iter().map(|p| p.0).collect();
        let outputs: Vec<[M31; STATE_WIDTH]> = perms.iter().map(|p| p.1).collect();

        // Break unchained rate position 5 in perm1
        inputs[1][5] = M31::from_u32_unchecked(999999);

        let result = verify_sponge_chain(&inputs, &outputs, 0, 2, 12);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("rate position 5"));
    }

    #[test]
    fn test_permutation_validity() {
        // Every recorded permutation should be a valid Poseidon2 permutation:
        // applying poseidon2_permutation to the recorded input should yield the recorded output.
        let input: Vec<M31> = (1..=11).map(|i| M31::from_u32_unchecked(i)).collect();
        let (_, perms) = record_hash_permutations(&input);

        for (i, (perm_in, perm_out)) in perms.iter().enumerate() {
            let mut state = *perm_in;
            poseidon2_permutation(&mut state);
            assert_eq!(state, *perm_out, "permutation {i} is not valid");
        }

        // Also test compress
        let left = [1, 2, 3, 4, 5, 6, 7, 8].map(M31::from_u32_unchecked);
        let right = [9, 10, 11, 12, 13, 14, 15, 16].map(M31::from_u32_unchecked);
        let (_, record) = record_compress_permutation(&left, &right);
        let mut state = record.0;
        poseidon2_permutation(&mut state);
        assert_eq!(state, record.1, "compress permutation is not valid");
    }
}
