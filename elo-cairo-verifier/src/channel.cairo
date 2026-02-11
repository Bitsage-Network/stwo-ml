// Poseidon252-Compatible Fiat-Shamir Channel
//
// Exactly matches STWO's Poseidon252Channel (stwo/src/core/channel/poseidon252.rs):
//
//   draw_secure_felt252():
//     state = [digest, n_draws, THREE(=3)]
//     hades_permutation(state)
//     return state[0], n_draws += 1
//
//   draw_base_felts():
//     felt252 → 8 M31 values via successive floor_div(2^31)
//     (LSB first: index 0 = least significant 31 bits)
//
//   draw_secure_felt():
//     8 M31 → take first 4 → QM31(CM31(m0,m1), CM31(m2,m3))
//
//   mix_u64(value):
//     digest = poseidon_hash(digest, value) = hades_permutation(digest, value, 2)[0]
//     n_draws = 0
//
//   mix_felts(&[SecureField]):
//     Pack QM31 pairs: fold(ONE, |acc, m31| acc * 2^31 + m31)
//     digest = poseidon_hash_many([digest, packed_values...])
//     n_draws = 0

use core::poseidon::{poseidon_hash_span, hades_permutation};
use crate::field::{CM31, QM31, M31_SHIFT, m31_reduce};

/// Channel state matching STWO's Poseidon252Channel { digest, n_draws }.
#[derive(Drop, Copy)]
pub struct PoseidonChannel {
    pub digest: felt252,
    pub n_draws: u32,
}

/// Create a new channel with zero initial state.
pub fn channel_default() -> PoseidonChannel {
    PoseidonChannel { digest: 0, n_draws: 0 }
}

/// Mix a u64 value into the channel.
/// Matches: self.update_digest(poseidon_hash(self.digest, value.into()))
/// poseidon_hash(a, b) in starknet_crypto = hades_permutation(a, b, 2)[0]
pub fn channel_mix_u64(ref ch: PoseidonChannel, value: u64) {
    let (s0, _, _) = hades_permutation(ch.digest, value.into(), 2);
    ch.digest = s0;
    ch.n_draws = 0;
}

/// Mix a felt252 value into the channel.
/// Matches Rust's PoseidonChannel::mix_felt(value).
pub fn channel_mix_felt(ref ch: PoseidonChannel, value: felt252) {
    let (s0, _, _) = hades_permutation(ch.digest, value, 2);
    ch.digest = s0;
    ch.n_draws = 0;
}

/// Draw a raw felt252 from the channel.
/// Domain separator THREE(=3) distinguishes draws from mixes.
fn channel_draw_felt252(ref ch: PoseidonChannel) -> felt252 {
    let (s0, _, _) = hades_permutation(ch.digest, ch.n_draws.into(), 3);
    ch.n_draws += 1;
    s0
}

/// Extract 8 M31 values from a felt252 by successive floor_div(2^31).
/// LSB first (index 0 = least significant 31 bits).
fn felt252_to_m31_array_8(
    value: felt252,
) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
    let shift: u256 = 0x80000000; // 2^31
    let mut cur: u256 = value.into();

    let r0: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r1: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r2: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r3: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r4: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r5: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r6: u64 = (cur % shift).try_into().unwrap();
    cur = cur / shift;
    let r7: u64 = (cur % shift).try_into().unwrap();

    (
        m31_reduce(r0), m31_reduce(r1), m31_reduce(r2), m31_reduce(r3),
        m31_reduce(r4), m31_reduce(r5), m31_reduce(r6), m31_reduce(r7),
    )
}

/// Draw a single QM31 challenge from the channel.
/// Matches: draw_secure_felt() = draw_base_felts()[0..4] → QM31.
/// Discards the upper 4 M31 values.
pub fn channel_draw_qm31(ref ch: PoseidonChannel) -> QM31 {
    let felt = channel_draw_felt252(ref ch);
    let (m0, m1, m2, m3, _, _, _, _) = felt252_to_m31_array_8(felt);
    QM31 {
        a: CM31 { a: m0, b: m1 },
        b: CM31 { a: m2, b: m3 },
    }
}

/// Draw multiple QM31 challenges from the channel.
pub fn channel_draw_qm31s(ref ch: PoseidonChannel, count: u32) -> Array<QM31> {
    let mut result: Array<QM31> = array![];
    let mut i: u32 = 0;
    loop {
        if i >= count {
            break;
        }
        result.append(channel_draw_qm31(ref ch));
        i += 1;
    };
    result
}

/// Pack a QM31's 4 M31 components into a running felt252 accumulator.
/// Implements: fold(acc, m31) = acc * 2^31 + m31.
/// Component order: [a.a, a.b, b.a, b.b].
fn pack_qm31_into_felt(mut cur: felt252, v: QM31) -> felt252 {
    cur = cur * M31_SHIFT + v.a.a.into();
    cur = cur * M31_SHIFT + v.a.b.into();
    cur = cur * M31_SHIFT + v.b.a.into();
    cur = cur * M31_SHIFT + v.b.b.into();
    cur
}

/// Mix degree-2 polynomial coefficients [c0, c1, c2] into the channel.
///
/// Packing (Poseidon252Channel::mix_felts with chunks(2)):
///   Chunk [c0, c1] → 8 M31 → 1 felt252 (starting from ONE)
///   Chunk [c2]     → 4 M31 → 1 felt252 (starting from ONE)
///   digest = poseidon_hash_many([digest, packed1, packed2])
pub fn channel_mix_poly_coeffs(ref ch: PoseidonChannel, c0: QM31, c1: QM31, c2: QM31) {
    let mut packed1: felt252 = 1;
    packed1 = pack_qm31_into_felt(packed1, c0);
    packed1 = pack_qm31_into_felt(packed1, c1);

    let mut packed2: felt252 = 1;
    packed2 = pack_qm31_into_felt(packed2, c2);

    ch.digest = poseidon_hash_span(array![ch.digest, packed1, packed2].span());
    ch.n_draws = 0;
}

/// Draw query indices from the channel.
/// Matches Rust's draw_query_indices: draw felt252, extract low 64 bits, mod range.
pub fn channel_draw_query_indices(
    ref ch: PoseidonChannel, half_n: u32, n_queries: u32,
) -> Array<u32> {
    let mut indices: Array<u32> = array![];
    let half_n_u64: u64 = half_n.into();
    let mut i: u32 = 0;
    loop {
        if i >= n_queries {
            break;
        }
        let felt = channel_draw_felt252(ref ch);
        let hash_u256: u256 = felt.into();
        let val_u64: u64 = (hash_u256 % 0x10000000000000000).try_into().unwrap();
        let index: u32 = (val_u64 % half_n_u64).try_into().unwrap();
        indices.append(index);
        i += 1;
    };
    indices
}
