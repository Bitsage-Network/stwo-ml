// Tests for vm31_verifier: batch public input hashing and IO binding.

use elo_cairo_verifier::vm31_verifier::{
    BatchPublicInputs, DepositPublicInput, WithdrawPublicInput, SpendPublicInput,
    hash_batch_public_inputs, verify_batch_public_inputs,
    reconstruct_amount, batch_tx_count,
};
use elo_cairo_verifier::vm31_merkle::pack_m31x8;

// ============================================================================
// Helpers
// ============================================================================

fn sample_deposit() -> DepositPublicInput {
    DepositPublicInput {
        commitment: pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span()),
        amount_lo: 42,
        amount_hi: 0,
        asset_id: 1,
    }
}

fn sample_withdraw() -> WithdrawPublicInput {
    WithdrawPublicInput {
        merkle_root: pack_m31x8(array![10_u64, 20, 30, 40, 50, 60, 70, 80].span()),
        nullifier: pack_m31x8(array![100_u64, 200, 300, 400, 500, 600, 700, 800].span()),
        amount_lo: 100,
        amount_hi: 0,
        asset_id: 1,
        withdrawal_binding: pack_m31x8(array![0_u64, 0, 0, 0, 0, 0, 0, 0].span()),
    }
}

fn sample_spend() -> SpendPublicInput {
    SpendPublicInput {
        merkle_root: pack_m31x8(array![10_u64, 20, 30, 40, 50, 60, 70, 80].span()),
        nullifier_0: pack_m31x8(array![11_u64, 22, 33, 44, 55, 66, 77, 88].span()),
        nullifier_1: pack_m31x8(array![111_u64, 222, 333, 444, 555, 666, 777, 888].span()),
        output_commitment_0: pack_m31x8(array![1_u64, 2, 3, 4, 5, 6, 7, 8].span()),
        output_commitment_1: pack_m31x8(array![9_u64, 10, 11, 12, 13, 14, 15, 16].span()),
    }
}

// ============================================================================
// Test 1: Hash is deterministic
// ============================================================================

#[test]
fn test_batch_hash_deterministic() {
    let inputs = BatchPublicInputs {
        deposits: array![sample_deposit()],
        withdrawals: array![],
        spends: array![],
    };
    let h1 = hash_batch_public_inputs(@inputs);

    let inputs2 = BatchPublicInputs {
        deposits: array![sample_deposit()],
        withdrawals: array![],
        spends: array![],
    };
    let h2 = hash_batch_public_inputs(@inputs2);

    assert!(h1 == h2, "batch hash should be deterministic");
}

// ============================================================================
// Test 2: Different inputs produce different hashes
// ============================================================================

#[test]
fn test_batch_hash_different_inputs() {
    let inputs1 = BatchPublicInputs {
        deposits: array![sample_deposit()],
        withdrawals: array![],
        spends: array![],
    };
    let h1 = hash_batch_public_inputs(@inputs1);

    // Modify amount
    let mut dep2 = sample_deposit();
    dep2.amount_lo = 43;
    let inputs2 = BatchPublicInputs {
        deposits: array![dep2],
        withdrawals: array![],
        spends: array![],
    };
    let h2 = hash_batch_public_inputs(@inputs2);

    assert!(h1 != h2, "different amounts should produce different hashes");
}

// ============================================================================
// Test 3: Verify batch public inputs succeeds with correct hash
// ============================================================================

#[test]
fn test_verify_batch_public_inputs_correct() {
    let inputs = BatchPublicInputs {
        deposits: array![sample_deposit()],
        withdrawals: array![sample_withdraw()],
        spends: array![],
    };
    let hash = hash_batch_public_inputs(@inputs);

    // Should succeed without panicking
    let result = verify_batch_public_inputs(@inputs, hash);
    assert!(result == hash, "verify should return the hash");
}

// ============================================================================
// Test 4: Verify batch public inputs fails with wrong hash
// ============================================================================

#[test]
#[should_panic(expected: "VM31: batch public input hash mismatch")]
fn test_verify_batch_public_inputs_wrong_hash() {
    let inputs = BatchPublicInputs {
        deposits: array![sample_deposit()],
        withdrawals: array![],
        spends: array![],
    };
    let wrong_hash = pack_m31x8(array![99_u64, 0, 0, 0, 0, 0, 0, 0].span());

    verify_batch_public_inputs(@inputs, wrong_hash);
}

// ============================================================================
// Test 5: Reconstruct amount
// ============================================================================

#[test]
fn test_reconstruct_amount() {
    assert!(reconstruct_amount(42, 0) == 42, "simple amount");
    assert!(reconstruct_amount(0, 1) == 0x80000000, "high limb only");
    assert!(reconstruct_amount(100, 2) == 100 + 2 * 0x80000000, "combined");
}

// ============================================================================
// Test 6: Batch tx count
// ============================================================================

#[test]
fn test_batch_tx_count() {
    let inputs = BatchPublicInputs {
        deposits: array![sample_deposit(), sample_deposit()],
        withdrawals: array![sample_withdraw()],
        spends: array![sample_spend()],
    };
    assert!(batch_tx_count(@inputs) == 4, "2 + 1 + 1 = 4");
}

// ============================================================================
// Test 7: Mixed batch hash includes all types
// ============================================================================

#[test]
fn test_mixed_batch_hash() {
    // Deposits only
    let dep_only = BatchPublicInputs {
        deposits: array![sample_deposit()],
        withdrawals: array![],
        spends: array![],
    };
    let h_dep = hash_batch_public_inputs(@dep_only);

    // Same deposit + a withdrawal
    let mixed = BatchPublicInputs {
        deposits: array![sample_deposit()],
        withdrawals: array![sample_withdraw()],
        spends: array![],
    };
    let h_mixed = hash_batch_public_inputs(@mixed);

    assert!(h_dep != h_mixed, "adding withdrawal should change hash");
}

// ============================================================================
// Test 8: Empty deposits/withdrawals/spends count hashed
// ============================================================================

#[test]
fn test_empty_types_different_hash() {
    let no_dep = BatchPublicInputs {
        deposits: array![],
        withdrawals: array![sample_withdraw()],
        spends: array![],
    };
    let no_wit = BatchPublicInputs {
        deposits: array![],
        withdrawals: array![],
        spends: array![sample_spend()],
    };
    let h1 = hash_batch_public_inputs(@no_dep);
    let h2 = hash_batch_public_inputs(@no_wit);
    assert!(h1 != h2, "different type compositions should have different hashes");
}
