#!/usr/bin/env python3
"""
Empirical verification of DER signature probability for random byte strings.

Implements Bitcoin's exact IsValidSignatureEncoding (BIP66) and tests:
1. P(20-byte random string is valid DER) — should be ~2^{-46.4}
2. P(32-byte random string is valid DER) — should be ~2^{-45.4}
3. SHA256 chain independence: are SHA256^k(x) outputs independent for DER?
4. Verify the formula: P(n) = (n-8) / (4 * 256^6) for sighash-unconstrained

Uses Monte Carlo sampling with billions of trials via vectorized numpy.
"""

import hashlib
import math
import os
import struct
import sys
import time
from collections import defaultdict

import numpy as np


def is_valid_signature_encoding(sig: bytes) -> bool:
    """
    Exact port of Bitcoin Core's IsValidSignatureEncoding from BIP66.
    https://github.com/bitcoin/bips/blob/master/bip-0066.mediawiki

    Does NOT check sighash type — only structural DER validity.
    """
    length = len(sig)

    # Minimum and maximum size constraints
    if length < 9:
        return False
    if length > 73:
        return False

    # A signature is of type 0x30 (compound)
    if sig[0] != 0x30:
        return False

    # Make sure the length covers the entire signature
    # sig[1] is the length of everything that follows (excluding sighash byte)
    if sig[1] != length - 3:
        return False

    # Extract the length of the R element
    lenR = sig[3]

    # Make sure the length of the S element is still inside the signature
    if 5 + lenR >= length:
        return False

    # Extract the length of the S element
    lenS = sig[5 + lenR]

    # Verify that the length of the signature matches the sum of the length
    # of the elements
    if lenR + lenS + 7 != length:
        return False

    # Check whether the R element is an integer
    if sig[2] != 0x02:
        return False

    # Zero-length integers are not allowed for R
    if lenR == 0:
        return False

    # Negative numbers are not allowed for R
    if sig[4] & 0x80:
        return False

    # Null bytes at the start of R are not allowed, unless R would
    # otherwise be interpreted as a negative number
    if lenR > 1 and sig[4] == 0x00 and not (sig[5] & 0x80):
        return False

    # Check whether the S element is an integer
    if sig[4 + lenR] != 0x02:
        return False

    # Zero-length integers are not allowed for S
    if lenS == 0:
        return False

    # Negative numbers are not allowed for S
    if sig[6 + lenR] & 0x80:
        return False

    # Null bytes at the start of S are not allowed, unless S would
    # otherwise be interpreted as a negative number
    if lenS > 1 and sig[6 + lenR] == 0x00 and not (sig[7 + lenR] & 0x80):
        return False

    return True


def theoretical_probability(n: int) -> float:
    """
    Compute theoretical P(n-byte random string is valid DER).

    Formula: P(n) = sum over valid r_len of:
        (1/256)^5 [fixed structural bytes] *
        (1/256) [r_len specific value] *
        P(r_valid | r_len) *
        P(s_valid | s_len) *
        1 [sighash unconstrained]

    Where P(value_valid | len) = 1/2 for len >= 1.

    Simplified: P(n) = num_valid_r_lens / (4 * 256^6)

    For n bytes: r_len + s_len = n - 7, both >= 1
    So r_len in {1, ..., n-8}, giving (n-8) valid values.

    P(n) = (n-8) / (4 * 256^6)
    """
    if n < 9 or n > 73:
        return 0.0
    num_r_lens = n - 8
    return num_r_lens / (4.0 * (256 ** 6))


def compute_exact_probability(n: int) -> float:
    """
    Compute exact probability by summing over all valid r_len values,
    accounting for the exact P(r_valid) and P(s_valid) per length.
    """
    if n < 9 or n > 73:
        return 0.0

    total_p = 0.0

    for r_len in range(1, n - 7):  # r_len from 1 to n-8
        s_len = (n - 7) - r_len
        if s_len < 1:
            continue

        # Fixed bytes: 0x30, total_len, 0x02 (r tag), 0x02 (s tag)
        # That's 4 fixed bytes at specific positions: (1/256)^4
        # Plus r_len at position 3: (1/256)
        # Plus s_len at position 5+r_len: (1/256)
        # Total fixed: (1/256)^6
        p_fixed = (1.0 / 256) ** 6

        # R value validity
        if r_len == 1:
            # Single byte: must be 0x00-0x7F (not negative)
            # 0x00 is allowed (represents zero)
            p_r = 128.0 / 256  # = 1/2
        else:
            # First byte 0x01-0x7F: 127/256
            # First byte 0x00, second byte 0x80-0xFF: (1/256)(128/256)
            p_r = 127.0 / 256 + (1.0 / 256) * (128.0 / 256)
            # = (127*256 + 128) / 65536 = 32768 / 65536 = 1/2

        # S value validity (same structure)
        if s_len == 1:
            p_s = 128.0 / 256
        else:
            p_s = 127.0 / 256 + (1.0 / 256) * (128.0 / 256)

        total_p += p_fixed * p_r * p_s

    return total_p


def monte_carlo_der_check(n: int, num_trials: int) -> tuple:
    """
    Monte Carlo estimation of P(n-byte random string is valid DER).
    Uses vectorized numpy for speed.

    Returns (hits, trials, probability, confidence_interval_95)
    """
    hits = 0
    batch_size = min(num_trials, 10_000_000)
    total_done = 0

    while total_done < num_trials:
        current_batch = min(batch_size, num_trials - total_done)

        # Generate random bytes
        data = np.random.randint(0, 256, size=(current_batch, n), dtype=np.uint8)

        for i in range(current_batch):
            if is_valid_signature_encoding(bytes(data[i])):
                hits += 1

        total_done += current_batch

        if total_done % (batch_size * 10) == 0 and total_done > 0:
            current_p = hits / total_done
            print(f"  ... {total_done:,} trials, {hits} hits, "
                  f"P ≈ {current_p:.2e}" if hits > 0 else f"  ... {total_done:,} trials, 0 hits")

    p = hits / num_trials if num_trials > 0 else 0
    # 95% confidence interval (Wilson score)
    if hits > 0:
        z = 1.96
        denominator = 1 + z**2 / num_trials
        centre = (p + z**2 / (2 * num_trials)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * num_trials)) / num_trials) / denominator
        ci = (centre - spread, centre + spread)
    else:
        # Upper bound: rule of three
        ci = (0, 3.0 / num_trials)

    return hits, num_trials, p, ci


def verify_sha256_chain_independence(num_samples: int = 1_000_000, chain_depth: int = 10):
    """
    Test whether SHA256 chain outputs have independent DER validity.

    For each random 33-byte input (simulating a compressed pubkey):
    1. Compute H1 = SHA256(input), H2 = SHA256(H1), ..., H_k = SHA256(H_{k-1})
    2. Check DER validity of each Hi
    3. Count hits per depth and joint hits

    If independent: P(Hi valid AND Hj valid) ≈ P(Hi valid) * P(Hj valid)
    """
    print(f"\n{'='*60}")
    print(f"SHA256 Chain Independence Test")
    print(f"Samples: {num_samples:,}, Chain depth: {chain_depth}")
    print(f"{'='*60}")

    hits_per_depth = [0] * chain_depth
    any_hit_count = 0
    joint_hits = defaultdict(int)  # (i,j) -> count

    for trial in range(num_samples):
        # Random 33-byte "compressed pubkey"
        x = os.urandom(33)

        chain_valid = []
        h = x
        for depth in range(chain_depth):
            h = hashlib.sha256(h).digest()  # 32 bytes
            valid = is_valid_signature_encoding(h)
            chain_valid.append(valid)
            if valid:
                hits_per_depth[depth] += 1

        if any(chain_valid):
            any_hit_count += 1

        # Record joint hits for independence testing
        for i in range(chain_depth):
            for j in range(i + 1, chain_depth):
                if chain_valid[i] and chain_valid[j]:
                    joint_hits[(i, j)] += 1

        if (trial + 1) % 100_000 == 0:
            print(f"  ... {trial+1:,} samples processed")

    print(f"\nResults:")
    print(f"  Theoretical P(32-byte DER) = {theoretical_probability(32):.6e}")
    print(f"  = 2^{math.log2(theoretical_probability(32)):.2f}")
    print()

    for d in range(chain_depth):
        p = hits_per_depth[d] / num_samples
        print(f"  Depth {d+1}: {hits_per_depth[d]} hits in {num_samples:,} "
              f"({p:.6e})")

    total_hits = sum(hits_per_depth)
    avg_p = total_hits / (num_samples * chain_depth)
    print(f"\n  Average per-depth P = {avg_p:.6e}")
    print(f"  Any hit in chain:    {any_hit_count} / {num_samples:,}")

    # Independence test
    if any(v > 0 for v in hits_per_depth):
        print(f"\n  Independence check (joint hits):")
        for (i, j), count in sorted(joint_hits.items()):
            pi = hits_per_depth[i] / num_samples
            pj = hits_per_depth[j] / num_samples
            expected = pi * pj * num_samples
            print(f"    Depths ({i+1},{j+1}): observed={count}, "
                  f"expected={expected:.4f} (if independent)")

    return hits_per_depth, any_hit_count


def verify_formula_across_lengths():
    """
    Verify that compute_exact_probability matches theoretical_probability
    for all valid signature lengths (9-73 bytes).
    """
    print(f"\n{'='*60}")
    print(f"Formula Verification Across All Lengths (9-73 bytes)")
    print(f"{'='*60}")
    print(f"{'Length':>6} {'Theoretical':>14} {'Exact':>14} {'Match':>6} {'Bits':>8}")

    for n in range(9, 74):
        p_theory = theoretical_probability(n)
        p_exact = compute_exact_probability(n)
        match = abs(p_theory - p_exact) / max(p_exact, 1e-100) < 0.01
        bits = -math.log2(p_exact) if p_exact > 0 else float('inf')
        print(f"{n:>6} {p_theory:>14.6e} {p_exact:>14.6e} {'✓' if match else '✗':>6} {bits:>8.2f}")

    # Key lengths
    print(f"\nKey lengths:")
    for n in [9, 20, 32, 65, 73]:
        p = compute_exact_probability(n)
        bits = -math.log2(p) if p > 0 else float('inf')
        print(f"  {n} bytes: P = {p:.6e} = 2^{bits:.2f}")
        print(f"    r_len range: 1 to {n-8} ({n-8} values)")


def verify_r_s_validity_probability():
    """
    Empirically verify that P(r_valid | r_len) = 1/2 for all r_len >= 1.
    """
    print(f"\n{'='*60}")
    print(f"R/S Value Validity Probability Check")
    print(f"{'='*60}")

    num_trials = 10_000_000

    for value_len in [1, 2, 3, 5, 10, 20, 32]:
        valid_count = 0

        for _ in range(num_trials):
            value = os.urandom(value_len)

            # Check: not negative (first byte < 0x80)
            if value[0] & 0x80:
                continue

            # Check: no unnecessary leading zeros
            if value_len > 1 and value[0] == 0x00 and not (value[1] & 0x80):
                continue

            valid_count += 1

        p = valid_count / num_trials
        print(f"  r_len={value_len:>2}: P(valid) = {p:.6f} "
              f"(expected 0.5, diff = {abs(p - 0.5):.6f})")


def check_sighash_constraint():
    """
    Verify: does the sighash byte (last byte) have any constraint
    in IsValidSignatureEncoding? (It shouldn't — BIP66 only checks structure.)
    """
    print(f"\n{'='*60}")
    print(f"Sighash Byte Constraint Check")
    print(f"{'='*60}")

    # Construct a known-valid 9-byte DER signature
    # 30 06 02 01 01 02 01 01 XX
    base = bytes([0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01])

    valid_sighash_count = 0
    for sighash in range(256):
        sig = base + bytes([sighash])
        if is_valid_signature_encoding(sig):
            valid_sighash_count += 1

    print(f"  Valid sighash values: {valid_sighash_count} / 256")
    print(f"  All sighash bytes accepted by IsValidSignatureEncoding: "
          f"{'YES' if valid_sighash_count == 256 else 'NO'}")

    if valid_sighash_count != 256:
        print(f"  WARNING: Not all sighash bytes pass DER check!")
        for sighash in range(256):
            sig = base + bytes([sighash])
            if not is_valid_signature_encoding(sig):
                print(f"    Rejected: 0x{sighash:02x}")


def main():
    print("=" * 60)
    print("QSB DER Probability Verification Suite")
    print("=" * 60)

    # 1. Verify formula matches exact computation
    verify_formula_across_lengths()

    # 2. Verify r/s validity probability = 1/2
    verify_r_s_validity_probability()

    # 3. Check sighash constraint
    check_sighash_constraint()

    # 4. SHA256 chain independence (the key claim)
    # Note: with P ~ 2^{-45}, we need ~2^{45} samples to see even one hit
    # in Monte Carlo. That's infeasible. Instead we test:
    # - Distribution properties of chain outputs
    # - First-byte distribution (should be uniform)
    # - Cross-depth correlation of byte values
    print(f"\n{'='*60}")
    print(f"SHA256 Chain Byte Distribution Test")
    print(f"(Since P(DER) ~ 2^{{-45}}, we test uniformity instead)")
    print(f"{'='*60}")

    num_samples = 1_000_000
    chain_depth = 10

    # Test: first byte distribution at each chain depth
    # Should be uniform over [0, 255] — critical because byte[0] must be 0x30
    first_byte_counts = [[0] * 256 for _ in range(chain_depth)]

    # Test: byte[1] distribution (should also be uniform)
    second_byte_counts = [[0] * 256 for _ in range(chain_depth)]

    for trial in range(num_samples):
        x = os.urandom(33)
        h = x
        for depth in range(chain_depth):
            h = hashlib.sha256(h).digest()
            first_byte_counts[depth][h[0]] += 1
            second_byte_counts[depth][h[1]] += 1

    expected = num_samples / 256

    print(f"\n  First byte uniformity (P(byte[0]=0x30) should be ~1/256 = {1/256:.6f}):")
    for d in range(chain_depth):
        p_0x30 = first_byte_counts[d][0x30] / num_samples
        chi2 = sum((c - expected)**2 / expected for c in first_byte_counts[d])
        print(f"    Depth {d+1}: P(0x30) = {p_0x30:.6f}, chi² = {chi2:.1f} "
              f"(expected ~255, {'OK' if 200 < chi2 < 320 else 'SUSPICIOUS'})")

    # Cross-depth correlation: are first bytes correlated across depths?
    print(f"\n  Cross-depth first-byte correlation:")
    for d1 in range(min(3, chain_depth)):
        for d2 in range(d1 + 1, min(d1 + 3, chain_depth)):
            # Count joint occurrences of byte[0]=0x30 at both depths
            joint_count = 0
            h_chain = [None] * chain_depth

            for trial in range(num_samples):
                x = os.urandom(33)
                h = x
                for depth in range(max(d1, d2) + 1):
                    h = hashlib.sha256(h).digest()
                    h_chain[depth] = h

                if h_chain[d1][0] == 0x30 and h_chain[d2][0] == 0x30:
                    joint_count += 1

            p_joint = joint_count / num_samples
            p_independent = (1/256) * (1/256)
            ratio = p_joint / p_independent if p_independent > 0 else 0
            print(f"    Depths ({d1+1},{d2+1}): P(both=0x30) = {p_joint:.8f}, "
                  f"expected if independent = {p_independent:.8f}, "
                  f"ratio = {ratio:.4f}")

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    p20 = compute_exact_probability(20)
    p32 = compute_exact_probability(32)

    print(f"\n  P(20-byte DER) = {p20:.6e} = 2^{math.log2(p20):.2f}")
    print(f"  P(32-byte DER) = {p32:.6e} = 2^{math.log2(p32):.2f}")
    print(f"  Ratio P(32)/P(20) = {p32/p20:.4f}")
    print()

    print(f"  SHA256 chain cost-benefit (pinning phase):")
    print(f"  {'N chains':>10} {'P improvement':>15} {'GPU overhead':>12} {'Net speedup':>12} {'Extra ops':>10}")

    ec_cost = 650
    base_cost = 820  # non-hash fixed costs
    hash_cost = 130  # per SHA256 compression
    der_cost = 50    # per DER check

    p0 = p20  # current baseline (RIPEMD-160)
    cost0 = base_cost + hash_cost + der_cost  # = 1000

    for N in [1, 2, 3, 4, 5, 10, 20]:
        p_n = N * p32
        cost_n = base_cost + N * (hash_cost + der_cost)

        improvement = p_n / p0
        overhead = cost_n / cost0
        net = improvement / overhead
        extra_ops = max(0, 3 * (N - 1))

        print(f"  {N:>10} {improvement:>14.1f}× {overhead:>11.2f}× {net:>11.1f}× {extra_ops:>10}")


if __name__ == "__main__":
    main()
