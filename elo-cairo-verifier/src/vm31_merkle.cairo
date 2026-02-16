// On-chain Poseidon2-M31 primitives for the VM31 privacy protocol.
//
// Implements the exact same Poseidon2 hash function as the Rust stwo-ml crate
// (crypto/poseidon2_m31.rs), enabling on-chain Merkle tree verification and
// note commitment validation.
//
// Parameters (matching Plonky3 / STWO):
//   - State width: t = 16
//   - Rate: 8, Capacity: 8
//   - S-box: x^5 over M31
//   - Full rounds: R_f = 8 (4 + 4)
//   - Partial rounds: R_p = 14
//   - External matrix: circ(2*M4, M4, M4, M4) from HorizenLabs
//   - Internal diagonal: Plonky3 DiffusionMatrixMersenne31
//   - Round constants: xorshift64 PRNG seeded with "Poseidon2-M31"

use crate::field::{m31_add, m31_mul};

// ============================================================================
// Constants
// ============================================================================

pub const STATE_WIDTH: u32 = 16;
pub const RATE: u32 = 8;
pub const CAPACITY: u32 = 8;
pub const N_FULL_ROUNDS: u32 = 8;
pub const N_HALF_FULL_ROUNDS: u32 = 4;
pub const N_PARTIAL_ROUNDS: u32 = 14;
pub const M31_P: u64 = 0x7FFFFFFF;

// Merkle tree depth (supports up to 2^20 = 1M notes)
pub const MERKLE_DEPTH: u32 = 20;

// ============================================================================
// PackedDigest: 8 M31 values packed as 2 felt252
// ============================================================================

// Each felt252 stores 4 M31 values: v0 + v1*2^31 + v2*2^62 + v3*2^93
// Total: 4 × 31 = 124 bits per felt252 (fits in 252 bits)
#[derive(Drop, Copy, Serde, starknet::Store, PartialEq, Debug)]
pub struct PackedDigest {
    pub lo: felt252,  // M31 values [0..4]
    pub hi: felt252,  // M31 values [4..8]
}

pub fn packed_digest_zero() -> PackedDigest {
    PackedDigest { lo: 0, hi: 0 }
}

// Pack 8 u64 M31 values into a PackedDigest
pub fn pack_m31x8(v: Span<u64>) -> PackedDigest {
    assert!(v.len() == 8, "pack_m31x8: need exactly 8 values");
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        assert!(*v.at(i) < M31_P, "pack_m31x8: value out of M31 range");
        i += 1;
    };
    let shift: felt252 = 0x80000000; // 2^31
    let lo: felt252 = (*v.at(0)).into()
        + (*v.at(1)).into() * shift
        + (*v.at(2)).into() * shift * shift
        + (*v.at(3)).into() * shift * shift * shift;
    let hi: felt252 = (*v.at(4)).into()
        + (*v.at(5)).into() * shift
        + (*v.at(6)).into() * shift * shift
        + (*v.at(7)).into() * shift * shift * shift;
    PackedDigest { lo, hi }
}

// Unpack a PackedDigest into 8 u64 M31 values
pub fn unpack_m31x8(d: PackedDigest) -> Array<u64> {
    let shift: u256 = 0x80000000; // 2^31
    let mask: u256 = 0x7FFFFFFF; // 2^31 - 1

    let mut result: Array<u64> = array![];

    let lo_u256: u256 = d.lo.into();
    result.append((lo_u256 & mask).try_into().unwrap());
    result.append(((lo_u256 / shift) & mask).try_into().unwrap());
    result.append(((lo_u256 / (shift * shift)) & mask).try_into().unwrap());
    result.append(((lo_u256 / (shift * shift * shift)) & mask).try_into().unwrap());

    let hi_u256: u256 = d.hi.into();
    result.append((hi_u256 & mask).try_into().unwrap());
    result.append(((hi_u256 / shift) & mask).try_into().unwrap());
    result.append(((hi_u256 / (shift * shift)) & mask).try_into().unwrap());
    result.append(((hi_u256 / (shift * shift * shift)) & mask).try_into().unwrap());

    result
}

// ============================================================================
// Internal diagonal vector (Plonky3 DiffusionMatrixMersenne31)
// ============================================================================

fn get_internal_diag(i: u32) -> u64 {
    // -2 mod p = 2147483645
    if i == 0 { return 2147483645; }
    if i == 1 { return 1; }
    if i == 2 { return 2; }
    if i == 3 { return 4; }
    if i == 4 { return 8; }
    if i == 5 { return 16; }
    if i == 6 { return 32; }
    if i == 7 { return 64; }
    if i == 8 { return 128; }
    if i == 9 { return 256; }
    if i == 10 { return 1024; }
    if i == 11 { return 4096; }
    if i == 12 { return 8192; }
    if i == 13 { return 16384; }
    if i == 14 { return 32768; }
    65536 // i == 15
}

// ============================================================================
// Round Constants (xorshift64 PRNG seeded with "Poseidon2-M31")
// ============================================================================

fn get_external_rc(round: u32, idx: u32) -> u64 {
    // 8 rounds × 16 elements = 128 constants
    // Stored as a flat lookup for gas efficiency
    let flat = round * 16 + idx;
    // Round 0
    if flat == 0 { return 1904805405; }
    if flat == 1 { return 1096573395; }
    if flat == 2 { return 325583922; }
    if flat == 3 { return 1922497245; }
    if flat == 4 { return 1238611110; }
    if flat == 5 { return 1782572124; }
    if flat == 6 { return 1214633699; }
    if flat == 7 { return 425244868; }
    if flat == 8 { return 368977990; }
    if flat == 9 { return 1938949003; }
    if flat == 10 { return 857728839; }
    if flat == 11 { return 657187509; }
    if flat == 12 { return 1292182440; }
    if flat == 13 { return 1155991041; }
    if flat == 14 { return 665120200; }
    if flat == 15 { return 632342137; }
    // Round 1
    if flat == 16 { return 67460776; }
    if flat == 17 { return 1347329608; }
    if flat == 18 { return 1871055643; }
    if flat == 19 { return 624516163; }
    if flat == 20 { return 1103701336; }
    if flat == 21 { return 1940111874; }
    if flat == 22 { return 1930359101; }
    if flat == 23 { return 1396428560; }
    if flat == 24 { return 1780094021; }
    if flat == 25 { return 603972502; }
    if flat == 26 { return 538668740; }
    if flat == 27 { return 1377983905; }
    if flat == 28 { return 763555589; }
    if flat == 29 { return 1484943348; }
    if flat == 30 { return 123454433; }
    if flat == 31 { return 248439189; }
    // Round 2
    if flat == 32 { return 1867417593; }
    if flat == 33 { return 986287536; }
    if flat == 34 { return 606008889; }
    if flat == 35 { return 73979116; }
    if flat == 36 { return 1322555314; }
    if flat == 37 { return 1499910744; }
    if flat == 38 { return 1276293876; }
    if flat == 39 { return 1430788068; }
    if flat == 40 { return 542774866; }
    if flat == 41 { return 471498949; }
    if flat == 42 { return 1166024235; }
    if flat == 43 { return 474821153; }
    if flat == 44 { return 1171382481; }
    if flat == 45 { return 1425437182; }
    if flat == 46 { return 711989992; }
    if flat == 47 { return 1190070539; }
    // Round 3
    if flat == 48 { return 825883997; }
    if flat == 49 { return 407968301; }
    if flat == 50 { return 828103240; }
    if flat == 51 { return 396959544; }
    if flat == 52 { return 254805600; }
    if flat == 53 { return 405629793; }
    if flat == 54 { return 1736078245; }
    if flat == 55 { return 161376884; }
    if flat == 56 { return 1762339952; }
    if flat == 57 { return 60701464; }
    if flat == 58 { return 1027360218; }
    if flat == 59 { return 1528437821; }
    if flat == 60 { return 1639818656; }
    if flat == 61 { return 820804151; }
    if flat == 62 { return 1694124839; }
    if flat == 63 { return 674178797; }
    // Round 4
    if flat == 64 { return 1951345086; }
    if flat == 65 { return 667628229; }
    if flat == 66 { return 1412910229; }
    if flat == 67 { return 1526417058; }
    if flat == 68 { return 1582191717; }
    if flat == 69 { return 1465503729; }
    if flat == 70 { return 1514991590; }
    if flat == 71 { return 723260968; }
    if flat == 72 { return 1038341032; }
    if flat == 73 { return 87125145; }
    if flat == 74 { return 1113380561; }
    if flat == 75 { return 1916769929; }
    if flat == 76 { return 883101163; }
    if flat == 77 { return 500806420; }
    if flat == 78 { return 441461154; }
    if flat == 79 { return 624402420; }
    // Round 5
    if flat == 80 { return 1879880798; }
    if flat == 81 { return 1427664595; }
    if flat == 82 { return 1528919036; }
    if flat == 83 { return 1701115451; }
    if flat == 84 { return 717907989; }
    if flat == 85 { return 498367125; }
    if flat == 86 { return 39443273; }
    if flat == 87 { return 559133583; }
    if flat == 88 { return 1693915992; }
    if flat == 89 { return 1588914461; }
    if flat == 90 { return 1444895204; }
    if flat == 91 { return 2002477838; }
    if flat == 92 { return 929976106; }
    if flat == 93 { return 685581961; }
    if flat == 94 { return 1175651740; }
    if flat == 95 { return 502929573; }
    // Round 6
    if flat == 96 { return 927043549; }
    if flat == 97 { return 1495546862; }
    if flat == 98 { return 919607960; }
    if flat == 99 { return 1562745368; }
    if flat == 100 { return 1969008016; }
    if flat == 101 { return 1653795331; }
    if flat == 102 { return 2038349847; }
    if flat == 103 { return 1649824183; }
    if flat == 104 { return 321040687; }
    if flat == 105 { return 2060370837; }
    if flat == 106 { return 996839186; }
    if flat == 107 { return 652263400; }
    if flat == 108 { return 565955495; }
    if flat == 109 { return 653444965; }
    if flat == 110 { return 2008703010; }
    if flat == 111 { return 54846370; }
    // Round 7
    if flat == 112 { return 1556376433; }
    if flat == 113 { return 134987890; }
    if flat == 114 { return 627602907; }
    if flat == 115 { return 1808498223; }
    if flat == 116 { return 604190690; }
    if flat == 117 { return 2002863080; }
    if flat == 118 { return 1637851708; }
    if flat == 119 { return 621652046; }
    if flat == 120 { return 1980469812; }
    if flat == 121 { return 1531936506; }
    if flat == 122 { return 828286260; }
    if flat == 123 { return 412743697; }
    if flat == 124 { return 968980913; }
    if flat == 125 { return 2132095013; }
    if flat == 126 { return 1743262036; }
    1549107277 // flat == 127
}

fn get_internal_rc(round: u32) -> u64 {
    if round == 0 { return 861470954; }
    if round == 1 { return 593081428; }
    if round == 2 { return 1279665870; }
    if round == 3 { return 52671424; }
    if round == 4 { return 1177440899; }
    if round == 5 { return 2121690958; }
    if round == 6 { return 1455540962; }
    if round == 7 { return 438352440; }
    if round == 8 { return 1523388190; }
    if round == 9 { return 397307856; }
    if round == 10 { return 1049387486; }
    if round == 11 { return 1488926401; }
    if round == 12 { return 1656484477; }
    331202396 // round == 13
}

// ============================================================================
// S-box: x^5 over M31
// ============================================================================

fn sbox(x: u64) -> u64 {
    let x2 = m31_mul(x, x);
    let x4 = m31_mul(x2, x2);
    m31_mul(x4, x)
}

// ============================================================================
// Matrix operations
// ============================================================================

// Apply 4x4 MDS sub-matrix (HorizenLabs / STWO):
//   [[5, 7, 1, 3],
//    [4, 6, 1, 1],
//    [1, 3, 5, 7],
//    [1, 1, 4, 6]]
// Implemented with only additions.
fn apply_m4(ref s0: u64, ref s1: u64, ref s2: u64, ref s3: u64) {
    let t0 = m31_add(s0, s1);
    let t02 = m31_add(t0, t0);
    let t1 = m31_add(s2, s3);
    let t12 = m31_add(t1, t1);
    let t2 = m31_add(m31_add(s1, s1), t1);
    let t3 = m31_add(m31_add(s3, s3), t0);
    let t4 = m31_add(m31_add(t12, t12), t3);
    let t5 = m31_add(m31_add(t02, t02), t2);
    s0 = m31_add(t3, t5);
    s1 = t5;
    s2 = m31_add(t2, t4);
    s3 = t4;
}

// Apply external round matrix: circ(2*M4, M4, M4, M4)
fn apply_external_round_matrix(ref state: Array<u64>) -> Array<u64> {
    // Apply M4 to each 4-element block
    let mut s = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 16 {
            break;
        }
        s.append(*state.at(i));
        i += 1;
    };

    // Block 0
    let mut a0 = *s.at(0); let mut a1 = *s.at(1);
    let mut a2 = *s.at(2); let mut a3 = *s.at(3);
    apply_m4(ref a0, ref a1, ref a2, ref a3);

    // Block 1
    let mut b0 = *s.at(4); let mut b1 = *s.at(5);
    let mut b2 = *s.at(6); let mut b3 = *s.at(7);
    apply_m4(ref b0, ref b1, ref b2, ref b3);

    // Block 2
    let mut c0 = *s.at(8); let mut c1 = *s.at(9);
    let mut c2 = *s.at(10); let mut c3 = *s.at(11);
    apply_m4(ref c0, ref c1, ref c2, ref c3);

    // Block 3
    let mut d0 = *s.at(12); let mut d1 = *s.at(13);
    let mut d2 = *s.at(14); let mut d3 = *s.at(15);
    apply_m4(ref d0, ref d1, ref d2, ref d3);

    // Cross-block column sums
    let cs0 = m31_add(m31_add(a0, b0), m31_add(c0, d0));
    let cs1 = m31_add(m31_add(a1, b1), m31_add(c1, d1));
    let cs2 = m31_add(m31_add(a2, b2), m31_add(c2, d2));
    let cs3 = m31_add(m31_add(a3, b3), m31_add(c3, d3));

    let mut result: Array<u64> = array![
        m31_add(a0, cs0), m31_add(a1, cs1), m31_add(a2, cs2), m31_add(a3, cs3),
        m31_add(b0, cs0), m31_add(b1, cs1), m31_add(b2, cs2), m31_add(b3, cs3),
        m31_add(c0, cs0), m31_add(c1, cs1), m31_add(c2, cs2), m31_add(c3, cs3),
        m31_add(d0, cs0), m31_add(d1, cs1), m31_add(d2, cs2), m31_add(d3, cs3),
    ];
    result
}

// Apply internal round matrix: M_I = J + diag(v)
// result[i] = v[i] * state[i] + sum(state)
fn apply_internal_round_matrix(ref state: Array<u64>) -> Array<u64> {
    let mut sum: u64 = 0;
    let mut i: u32 = 0;
    loop {
        if i >= 16 {
            break;
        }
        sum = m31_add(sum, *state.at(i));
        i += 1;
    };

    let mut result: Array<u64> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 16 {
            break;
        }
        result.append(m31_add(m31_mul(*state.at(i), get_internal_diag(i)), sum));
        i += 1;
    };
    result
}

// ============================================================================
// Poseidon2 Permutation
// ============================================================================

// Full Poseidon2 permutation over M31[16].
// Structure:
//   - 4 full rounds: AddConst → S-box(all) → External matrix
//   - 14 partial rounds: AddConst[0] → S-box[0] → Internal matrix
//   - 4 full rounds: AddConst → S-box(all) → External matrix
pub fn poseidon2_m31_permutation(input: Span<u64>) -> Array<u64> {
    assert!(input.len() == 16, "poseidon2: state must be 16 elements");

    // Copy input into working state
    let mut state: Array<u64> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 16 {
            break;
        }
        state.append(*input.at(i));
        i += 1;
    };

    // First half: 4 full rounds
    let mut round: u32 = 0;
    loop {
        if round >= N_HALF_FULL_ROUNDS {
            break;
        }
        // Add round constants + S-box on all
        let mut next: Array<u64> = array![];
        let mut i: u32 = 0;
        loop {
            if i >= 16 {
                break;
            }
            next.append(sbox(m31_add(*state.at(i), get_external_rc(round, i))));
            i += 1;
        };
        // External linear layer
        state = apply_external_round_matrix(ref next);
        round += 1;
    };

    // Middle: 14 partial rounds
    let mut round: u32 = 0;
    loop {
        if round >= N_PARTIAL_ROUNDS {
            break;
        }
        // Add round constant to first element + S-box on first only
        let mut next: Array<u64> = array![];
        next.append(sbox(m31_add(*state.at(0), get_internal_rc(round))));
        let mut i: u32 = 1;
        loop {
            if i >= 16 {
                break;
            }
            next.append(*state.at(i));
            i += 1;
        };
        // Internal linear layer
        state = apply_internal_round_matrix(ref next);
        round += 1;
    };

    // Second half: 4 full rounds
    let mut round: u32 = 0;
    loop {
        if round >= N_HALF_FULL_ROUNDS {
            break;
        }
        let rc_idx = round + N_HALF_FULL_ROUNDS;
        let mut next: Array<u64> = array![];
        let mut i: u32 = 0;
        loop {
            if i >= 16 {
                break;
            }
            next.append(sbox(m31_add(*state.at(i), get_external_rc(rc_idx, i))));
            i += 1;
        };
        state = apply_external_round_matrix(ref next);
        round += 1;
    };

    state
}

// ============================================================================
// Sponge Hash
// ============================================================================

// Hash variable-length M31 input to 8 M31 elements.
// Uses sponge construction with rate=8, capacity=8.
// Domain separation: input length encoded in state[8] (first capacity position).
pub fn poseidon2_m31_hash(input: Span<u64>) -> Array<u64> {
    // Initialize state to zeros
    let mut state: Array<u64> = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    // Domain separation: encode input length in first capacity element
    let input_len: u64 = input.len().into();
    let mut ds_state: Array<u64> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 16 {
            break;
        }
        if i == 8 {
            ds_state.append(input_len);
        } else {
            ds_state.append(*state.at(i));
        }
        i += 1;
    };
    state = ds_state;

    // Absorb phase: process input in chunks of RATE (8)
    let n_chunks = if input.len() == 0 { 0_u32 } else { (input.len() - 1) / 8 + 1 };
    let mut chunk_idx: u32 = 0;
    loop {
        if chunk_idx >= n_chunks {
            break;
        }
        // Absorb: add chunk elements to state[0..8]
        let chunk_start = chunk_idx * 8;
        let mut absorbed: Array<u64> = array![];
        let mut i: u32 = 0;
        loop {
            if i >= 16 {
                break;
            }
            if i < 8 {
                let data_idx = chunk_start + i;
                if data_idx < input.len() {
                    absorbed.append(m31_add(*state.at(i), *input.at(data_idx)));
                } else {
                    absorbed.append(*state.at(i));
                }
            } else {
                absorbed.append(*state.at(i));
            }
            i += 1;
        };
        // Permute
        state = poseidon2_m31_permutation(absorbed.span());
        chunk_idx += 1;
    };

    // For empty input, still apply one permutation (domain sep is in capacity)
    if input.len() == 0 {
        state = poseidon2_m31_permutation(state.span());
    }

    // Squeeze: return rate portion (first 8 elements)
    let mut output: Array<u64> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        output.append(*state.at(i));
        i += 1;
    };
    output
}

// Hash and return as PackedDigest
pub fn poseidon2_m31_hash_packed(input: Span<u64>) -> PackedDigest {
    let h = poseidon2_m31_hash(input);
    pack_m31x8(h.span())
}

// ============================================================================
// 2-to-1 Compression (for Merkle trees)
// ============================================================================

// Takes two 8-element digests, loads into full 16-element state, permutes,
// returns rate portion. Jive/overwrite mode (standard 2-to-1 compression).
pub fn poseidon2_m31_compress(left: Span<u64>, right: Span<u64>) -> Array<u64> {
    assert!(left.len() == 8, "compress: left must be 8 elements");
    assert!(right.len() == 8, "compress: right must be 8 elements");

    let mut state: Array<u64> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        state.append(*left.at(i));
        i += 1;
    };
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        state.append(*right.at(i));
        i += 1;
    };

    let permuted = poseidon2_m31_permutation(state.span());

    // Return rate portion
    let mut output: Array<u64> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        output.append(*permuted.at(i));
        i += 1;
    };
    output
}

// Compress with PackedDigest types
pub fn poseidon2_m31_compress_packed(left: PackedDigest, right: PackedDigest) -> PackedDigest {
    let l = unpack_m31x8(left);
    let r = unpack_m31x8(right);
    let h = poseidon2_m31_compress(l.span(), r.span());
    pack_m31x8(h.span())
}

// ============================================================================
// Merkle Tree Operations
// ============================================================================

// Default empty leaf digest: hash of empty input
// This is a fixed value used for empty tree positions.
// We compute it dynamically since Cairo doesn't support complex const initialization.
pub fn empty_leaf() -> PackedDigest {
    packed_digest_zero()
}

// Compute the root of a Merkle path from leaf to root.
// path: array of sibling digests from leaf level to root.
// leaf: the leaf digest.
// path_indices: bit array indicating left(0) or right(1) at each level.
pub fn compute_merkle_root(
    leaf: PackedDigest,
    path: Span<PackedDigest>,
    path_indices: Span<u8>,
) -> PackedDigest {
    assert!(path.len() == path_indices.len(), "merkle: path/index length mismatch");

    let mut current = leaf;
    let mut i: u32 = 0;
    loop {
        if i >= path.len() {
            break;
        }
        let sibling = *path.at(i);
        let idx = *path_indices.at(i);

        if idx == 0 {
            // Current node is the left child
            current = poseidon2_m31_compress_packed(current, sibling);
        } else {
            // Current node is the right child
            current = poseidon2_m31_compress_packed(sibling, current);
        }
        i += 1;
    };
    current
}

// Verify a Merkle proof: given a leaf, path, indices, and expected root.
pub fn verify_merkle_proof(
    leaf: PackedDigest,
    path: Span<PackedDigest>,
    path_indices: Span<u8>,
    expected_root: PackedDigest,
) -> bool {
    let computed_root = compute_merkle_root(leaf, path, path_indices);
    computed_root == expected_root
}

// Compute a note commitment digest for use as a Merkle leaf.
// Matches Rust's Note: Poseidon2(pk[4] || asset || amt_lo || amt_hi || blinding[4])
pub fn compute_note_commitment(
    pk: Span<u64>,         // 4 M31 elements: public key
    asset_id: u64,         // asset identifier
    amount_lo: u64,        // amount low limb
    amount_hi: u64,        // amount high limb
    blinding: Span<u64>,   // 4 M31 elements: blinding factor
) -> PackedDigest {
    assert!(pk.len() == 4, "note: pk must be 4 elements");
    assert!(blinding.len() == 4, "note: blinding must be 4 elements");

    let mut input: Array<u64> = array![];
    // pk[0..4]
    input.append(*pk.at(0));
    input.append(*pk.at(1));
    input.append(*pk.at(2));
    input.append(*pk.at(3));
    // asset_id, amount_lo, amount_hi
    input.append(asset_id);
    input.append(amount_lo);
    input.append(amount_hi);
    // blinding[0..4]
    input.append(*blinding.at(0));
    input.append(*blinding.at(1));
    input.append(*blinding.at(2));
    input.append(*blinding.at(3));

    poseidon2_m31_hash_packed(input.span())
}

// Compute a nullifier: Poseidon2(sk[4] || commitment[8])
pub fn compute_nullifier(
    sk: Span<u64>,              // 4 M31 elements: secret key
    commitment: PackedDigest,   // note commitment
) -> PackedDigest {
    assert!(sk.len() == 4, "nullifier: sk must be 4 elements");

    let commitment_vals = unpack_m31x8(commitment);
    let mut input: Array<u64> = array![];
    // sk[0..4]
    input.append(*sk.at(0));
    input.append(*sk.at(1));
    input.append(*sk.at(2));
    input.append(*sk.at(3));
    // commitment[0..8]
    let mut i: u32 = 0;
    loop {
        if i >= 8 {
            break;
        }
        input.append(*commitment_vals.at(i));
        i += 1;
    };

    poseidon2_m31_hash_packed(input.span())
}
