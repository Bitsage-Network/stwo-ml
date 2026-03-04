// Tests for vm31_merkle: Poseidon2-M31 on-chain primitives.
// Cross-validates against Rust test vectors from stwo-ml/src/crypto/poseidon2_m31.rs.

use elo_cairo_verifier::vm31_merkle::{
    poseidon2_m31_permutation, poseidon2_m31_hash, poseidon2_m31_compress,
    poseidon2_m31_compress_packed,
    pack_m31x8, unpack_m31x8,
    verify_merkle_proof,
    compute_note_commitment, compute_nullifier,
    PackedDigest,
};

// ============================================================================
// Test 1: Permutation of all-zeros matches Rust test vector
// ============================================================================

#[test]
fn test_permutation_all_zeros() {
    let input: Array<u64> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let output = poseidon2_m31_permutation(input.span());

    // Test vector from Rust: permutation of all-zeros state
    assert!(*output.at(0) == 141925539, "perm[0] mismatch");
    assert!(*output.at(1) == 2022402577, "perm[1] mismatch");
    assert!(*output.at(2) == 90857687, "perm[2] mismatch");
    assert!(*output.at(3) == 37707072, "perm[3] mismatch");
    assert!(*output.at(4) == 908263051, "perm[4] mismatch");
    assert!(*output.at(5) == 112043401, "perm[5] mismatch");
    assert!(*output.at(6) == 155381440, "perm[6] mismatch");
    assert!(*output.at(7) == 1719032434, "perm[7] mismatch");
    assert!(*output.at(8) == 659955956, "perm[8] mismatch");
    assert!(*output.at(9) == 1976645536, "perm[9] mismatch");
    assert!(*output.at(10) == 1334159862, "perm[10] mismatch");
    assert!(*output.at(11) == 653787337, "perm[11] mismatch");
    assert!(*output.at(12) == 125388620, "perm[12] mismatch");
    assert!(*output.at(13) == 1278990130, "perm[13] mismatch");
    assert!(*output.at(14) == 1676336433, "perm[14] mismatch");
    assert!(*output.at(15) == 840772834, "perm[15] mismatch");
}

// ============================================================================
// Test 2: Permutation of [1..16] matches Rust test vector
// ============================================================================

#[test]
fn test_permutation_sequential() {
    let input: Array<u64> = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let output = poseidon2_m31_permutation(input.span());

    assert!(*output.at(0) == 1952311145, "seq[0] mismatch");
    assert!(*output.at(1) == 1366111949, "seq[1] mismatch");
    assert!(*output.at(2) == 1659008477, "seq[2] mismatch");
    assert!(*output.at(3) == 15715750, "seq[3] mismatch");
    assert!(*output.at(4) == 2072970988, "seq[4] mismatch");
    assert!(*output.at(5) == 1381954016, "seq[5] mismatch");
    assert!(*output.at(6) == 1009725444, "seq[6] mismatch");
    assert!(*output.at(7) == 475559150, "seq[7] mismatch");
}

// ============================================================================
// Test 3: Hash of [42] matches Rust test vector
// ============================================================================

#[test]
fn test_hash_42() {
    let input: Array<u64> = array![42];
    let output = poseidon2_m31_hash(input.span());

    assert!(output.len() == 8, "hash output must be 8 elements");
    assert!(*output.at(0) == 1803232797, "hash[0] mismatch");
    assert!(*output.at(1) == 390016128, "hash[1] mismatch");
    assert!(*output.at(2) == 1166274131, "hash[2] mismatch");
    assert!(*output.at(3) == 1025409010, "hash[3] mismatch");
    assert!(*output.at(4) == 1740080392, "hash[4] mismatch");
    assert!(*output.at(5) == 1578330152, "hash[5] mismatch");
    assert!(*output.at(6) == 332743797, "hash[6] mismatch");
    assert!(*output.at(7) == 1931424693, "hash[7] mismatch");
}

// ============================================================================
// Test 4: Permutation is deterministic
// ============================================================================

#[test]
fn test_permutation_deterministic() {
    let input: Array<u64> = array![7, 13, 42, 99, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let out1 = poseidon2_m31_permutation(input.span());
    let out2 = poseidon2_m31_permutation(input.span());

    let mut i: u32 = 0;
    loop {
        if i >= 16 {
            break;
        }
        assert!(*out1.at(i) == *out2.at(i), "non-deterministic");
        i += 1;
    };
}

// ============================================================================
// Test 5: Hash is deterministic
// ============================================================================

#[test]
fn test_hash_deterministic() {
    let input: Array<u64> = array![1, 2, 3, 4, 5];
    let h1 = poseidon2_m31_hash(input.span());
    let h2 = poseidon2_m31_hash(input.span());

    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        assert!(*h1.at(i) == *h2.at(i), "hash non-deterministic");
        i += 1;
    };
}

// ============================================================================
// Test 6: Different inputs produce different hashes
// ============================================================================

#[test]
fn test_hash_collision_resistance() {
    let h1 = poseidon2_m31_hash(array![1_u64].span());
    let h2 = poseidon2_m31_hash(array![2_u64].span());

    // At least one element must differ
    let mut differs = false;
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        if *h1.at(i) != *h2.at(i) {
            differs = true;
        }
        i += 1;
    };
    assert!(differs, "different inputs should produce different hashes");
}

// ============================================================================
// Test 7: Length domain separation
// ============================================================================

#[test]
fn test_hash_length_domain_separation() {
    let h1 = poseidon2_m31_hash(array![1_u64, 2].span());
    let h2 = poseidon2_m31_hash(array![1_u64, 2, 0].span());

    let mut differs = false;
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        if *h1.at(i) != *h2.at(i) {
            differs = true;
        }
        i += 1;
    };
    assert!(differs, "length domain separation should work");
}

// ============================================================================
// Test 8: Compress order matters
// ============================================================================

#[test]
fn test_compress_order_matters() {
    let a: Array<u64> = array![1, 2, 3, 4, 5, 6, 7, 8];
    let b: Array<u64> = array![9, 10, 11, 12, 13, 14, 15, 16];

    let h_ab = poseidon2_m31_compress(a.span(), b.span());
    let h_ba = poseidon2_m31_compress(b.span(), a.span());

    let mut differs = false;
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        if *h_ab.at(i) != *h_ba.at(i) {
            differs = true;
        }
        i += 1;
    };
    assert!(differs, "compress(a,b) != compress(b,a)");
}

// ============================================================================
// Test 9: PackedDigest pack/unpack roundtrip
// ============================================================================

#[test]
fn test_packed_digest_roundtrip() {
    let vals: Array<u64> = array![42, 99, 1000, 2000000, 7, 13, 256, 65536];
    let packed = pack_m31x8(vals.span());
    let unpacked = unpack_m31x8(packed);

    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        assert!(*unpacked.at(i) == *vals.at(i), "pack/unpack roundtrip failed");
        i += 1;
    };
}

// ============================================================================
// Test 10: Compress packed matches compress unpacked
// ============================================================================

#[test]
fn test_compress_packed_consistency() {
    let left: Array<u64> = array![1, 2, 3, 4, 5, 6, 7, 8];
    let right: Array<u64> = array![9, 10, 11, 12, 13, 14, 15, 16];

    // Unpacked version
    let h_raw = poseidon2_m31_compress(left.span(), right.span());

    // Packed version
    let left_packed = pack_m31x8(left.span());
    let right_packed = pack_m31x8(right.span());
    let h_packed = poseidon2_m31_compress_packed(left_packed, right_packed);

    // Unpack and compare
    let h_unpacked = unpack_m31x8(h_packed);
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        assert!(*h_raw.at(i) == *h_unpacked.at(i), "packed/unpacked compress mismatch");
        i += 1;
    };
}

// ============================================================================
// Test 11: Simple Merkle proof (depth 1)
// ============================================================================

#[test]
fn test_merkle_proof_depth_1() {
    let left = pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span());
    let right = pack_m31x8(array![9_u64, 10, 11, 12, 13, 14, 15, 16].span());
    let root = poseidon2_m31_compress_packed(left, right);

    // Prove left leaf
    let path: Array<PackedDigest> = array![right];
    let indices: Array<u8> = array![0]; // left child
    assert!(verify_merkle_proof(left, path.span(), indices.span(), root), "left proof failed");

    // Prove right leaf
    let path2: Array<PackedDigest> = array![left];
    let indices2: Array<u8> = array![1]; // right child
    assert!(verify_merkle_proof(right, path2.span(), indices2.span(), root), "right proof failed");
}

// ============================================================================
// Test 12: Merkle proof rejects wrong root
// ============================================================================

#[test]
fn test_merkle_proof_rejects_wrong_root() {
    let left = pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span());
    let right = pack_m31x8(array![9_u64, 10, 11, 12, 13, 14, 15, 16].span());
    let wrong_root = pack_m31x8(array![99_u64, 0, 0, 0, 0, 0, 0, 0].span());

    let path: Array<PackedDigest> = array![right];
    let indices: Array<u8> = array![0];
    assert!(!verify_merkle_proof(left, path.span(), indices.span(), wrong_root), "should reject");
}

// ============================================================================
// Test 13: Note commitment deterministic
// ============================================================================

#[test]
fn test_note_commitment_deterministic() {
    let pk: Array<u64> = array![100, 200, 300, 400];
    let blinding: Array<u64> = array![500, 600, 700, 800];

    let c1 = compute_note_commitment(pk.span(), 1, 42, 0, blinding.span());
    let c2 = compute_note_commitment(pk.span(), 1, 42, 0, blinding.span());
    assert!(c1 == c2, "note commitment not deterministic");
}

// ============================================================================
// Test 14: Different notes produce different commitments
// ============================================================================

#[test]
fn test_note_commitment_unique() {
    let pk: Array<u64> = array![100, 200, 300, 400];
    let blinding: Array<u64> = array![500, 600, 700, 800];

    let c1 = compute_note_commitment(pk.span(), 1, 42, 0, blinding.span());
    let c2 = compute_note_commitment(pk.span(), 1, 43, 0, blinding.span()); // different amount
    assert!(c1 != c2, "different amounts should produce different commitments");
}

// ============================================================================
// Test 15: Nullifier computation
// ============================================================================

#[test]
fn test_nullifier_deterministic() {
    let sk: Array<u64> = array![111, 222, 333, 444];
    let commitment = pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span());

    let n1 = compute_nullifier(sk.span(), commitment);
    let n2 = compute_nullifier(sk.span(), commitment);
    assert!(n1 == n2, "nullifier not deterministic");
}

// ============================================================================
// Test 16: Different secrets produce different nullifiers
// ============================================================================

#[test]
fn test_nullifier_unique() {
    let sk1: Array<u64> = array![111, 222, 333, 444];
    let sk2: Array<u64> = array![111, 222, 333, 445]; // different secret
    let commitment = pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span());

    let n1 = compute_nullifier(sk1.span(), commitment);
    let n2 = compute_nullifier(sk2.span(), commitment);
    assert!(n1 != n2, "different secrets should produce different nullifiers");
}

// ============================================================================
// Test 17: Permutation is not identity
// ============================================================================

#[test]
fn test_permutation_not_identity() {
    let input: Array<u64> = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let output = poseidon2_m31_permutation(input.span());

    let mut differs = false;
    let mut i: u32 = 0;
    loop {
        if i >= 16 {
            break;
        }
        if *output.at(i) != *input.at(i) {
            differs = true;
        }
        i += 1;
    };
    assert!(differs, "permutation should not be identity");
}

// ============================================================================
// Test 18: Hash of long input (>8 elements, multiple absorb rounds)
// ============================================================================

#[test]
fn test_hash_long_input() {
    let input: Array<u64> = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    let h1 = poseidon2_m31_hash(input.span());
    let h2 = poseidon2_m31_hash(input.span());

    // Deterministic
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        assert!(*h1.at(i) == *h2.at(i), "long hash non-deterministic");
        i += 1;
    };

    // Should differ from shorter input
    let short: Array<u64> = array![1, 2, 3, 4, 5, 6, 7, 8];
    let h_short = poseidon2_m31_hash(short.span());
    let mut differs = false;
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        if *h1.at(i) != *h_short.at(i) {
            differs = true;
        }
        i += 1;
    };
    assert!(differs, "long and short hashes should differ");
}
