//! Poseidon2-M31 commitment and nullifier primitives for VM31 privacy protocol.
//!
//! Note commitment: Poseidon2(owner_pubkey || asset_id || amount_lo || amount_hi || blinding)
//! Nullifier: Poseidon2(spending_key || commitment)
//! Key derivation: pubkey = Poseidon2(spending_key || "spend")

use stwo::core::fields::m31::BaseField as M31;

use super::poseidon2_m31::{poseidon2_hash, RATE};

/// A note in the VM31 pool.
/// Contains all data needed to spend or verify the note.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Note {
    /// Owner's public key (Poseidon2 hash of spending key)
    pub owner_pubkey: [M31; 4],
    /// Asset identifier (0 = STRK, 1 = ETH, etc.)
    pub asset_id: M31,
    /// Amount: low limb (31 bits)
    pub amount_lo: M31,
    /// Amount: high limb (31 bits). Total value = lo + hi * 2^31.
    pub amount_hi: M31,
    /// Random blinding factor (4 M31 elements for 124-bit hiding)
    pub blinding: [M31; 4],
}

/// Commitment to a note: 8 M31 elements (248-bit, ~124-bit collision resistance).
pub type NoteCommitment = [M31; RATE];

/// Nullifier: 8 M31 elements. Uniquely identifies a spent note.
pub type Nullifier = [M31; RATE];

/// Spending key: 4 M31 elements (private).
pub type SpendingKey = [M31; 4];

/// Public key: 4 M31 elements (derived from spending key).
pub type PublicKey = [M31; 4];

/// Viewing key: 4 M31 elements (semi-private, shared with authorized viewers).
/// Holders of the viewing key can decrypt incoming note memos but cannot spend.
pub type ViewingKey = [M31; 4];

impl Note {
    /// Create a new note.
    pub fn new(
        owner_pubkey: [M31; 4],
        asset_id: M31,
        amount_lo: M31,
        amount_hi: M31,
        blinding: [M31; 4],
    ) -> Self {
        Self {
            owner_pubkey,
            asset_id,
            amount_lo,
            amount_hi,
            blinding,
        }
    }

    /// Compute the commitment to this note.
    /// commitment = Poseidon2(pk[0..4] || asset_id || amount_lo || amount_hi || blinding[0..4])
    pub fn commitment(&self) -> NoteCommitment {
        let input = [
            self.owner_pubkey[0],
            self.owner_pubkey[1],
            self.owner_pubkey[2],
            self.owner_pubkey[3],
            self.asset_id,
            self.amount_lo,
            self.amount_hi,
            self.blinding[0],
            self.blinding[1],
            self.blinding[2],
            self.blinding[3],
        ];
        poseidon2_hash(&input)
    }

    /// Compute the nullifier for this note given the spending key.
    /// nullifier = Poseidon2(sk[0..4] || commitment[0..8])
    pub fn nullifier(&self, spending_key: &SpendingKey) -> Nullifier {
        let commitment = self.commitment();
        let mut input = [M31::from_u32_unchecked(0); 12];
        input[0] = spending_key[0];
        input[1] = spending_key[1];
        input[2] = spending_key[2];
        input[3] = spending_key[3];
        input[4..12].copy_from_slice(&commitment[..8]);
        poseidon2_hash(&input)
    }
}

/// Derive a public key from a spending key.
/// pubkey = Poseidon2("vm31-spend" || sk)[0..4]
///
/// Domain separation constant: "vm31-spend" encoded as M31 = 0x766D3331 (truncated).
pub fn derive_pubkey(spending_key: &SpendingKey) -> PublicKey {
    const DOMAIN_SPEND: M31 = M31::from_u32_unchecked(0x766D3331); // "vm31"
    let input = [
        DOMAIN_SPEND,
        spending_key[0],
        spending_key[1],
        spending_key[2],
        spending_key[3],
    ];
    let h = poseidon2_hash(&input);
    [h[0], h[1], h[2], h[3]]
}

/// Derive a viewing key from a spending key.
/// vk = Poseidon2("vm31-view" || sk)[0..4]
pub fn derive_viewing_key(spending_key: &SpendingKey) -> [M31; 4] {
    const DOMAIN_VIEW: M31 = M31::from_u32_unchecked(0x76696577); // "view"
    let input = [
        DOMAIN_VIEW,
        spending_key[0],
        spending_key[1],
        spending_key[2],
        spending_key[3],
    ];
    let h = poseidon2_hash(&input);
    [h[0], h[1], h[2], h[3]]
}

/// M31 modulus: p = 2^31 − 1. Both `M31(0)` and `M31(P)` are field-zero.
const M31_MODULUS: u32 = 0x7FFFFFFF;

/// Returns true if an M31 element is zero in the field (either canonical 0 or
/// the non-canonical representation P = 0x7FFFFFFF).
#[inline]
fn is_field_zero(m: &M31) -> bool {
    m.0 == 0 || m.0 == M31_MODULUS
}

/// Validate that a spending key is not all-zero in the M31 field.
///
/// A zero spending key `[0,0,0,0]` means anyone can compute nullifiers,
/// making the note trivially spendable by any observer.
///
/// Checks both canonical zero (0) and non-canonical zero (0x7FFFFFFF ≡ 0 mod p).
pub fn validate_spending_key(sk: &SpendingKey) -> Result<(), &'static str> {
    if sk.iter().all(is_field_zero) {
        return Err("zero spending key: anyone could compute nullifiers");
    }
    Ok(())
}

/// Minimum non-zero M31 elements required in a blinding factor.
///
/// Each non-zero M31 element contributes ~31 bits of hiding.  Two non-zero
/// elements give ≥62 bits — the minimum for computational security in a
/// privacy pool where notes are ephemeral.
///
/// With `getrandom`-sourced blinding the probability of fewer than 2
/// non-zero elements is ~4/2^31 ≈ 2×10⁻⁹, so legitimate blinding always
/// passes.
const MIN_NONZERO_BLINDING: usize = 2;

/// Validate that a blinding factor has sufficient entropy for hiding.
///
/// Requires at least [`MIN_NONZERO_BLINDING`] (2) non-zero M31 elements,
/// giving ≥62 bits of hiding.  A single non-zero element (~31 bits) is
/// brute-forceable on modern hardware and is rejected (H5).
///
/// Checks both canonical zero (0) and non-canonical zero (0x7FFFFFFF ≡ 0 mod p).
pub fn validate_blinding(blinding: &[M31; 4]) -> Result<(), &'static str> {
    let nonzero_count = blinding.iter().filter(|m| !is_field_zero(m)).count();
    if nonzero_count < MIN_NONZERO_BLINDING {
        return Err("blinding has < 2 non-zero elements: insufficient hiding (need >= 62 bits)");
    }
    Ok(())
}

/// Compute nullifier directly from a spending key and note commitment.
pub fn compute_nullifier(spending_key: &SpendingKey, commitment: &NoteCommitment) -> Nullifier {
    let mut input = [M31::from_u32_unchecked(0); 12];
    input[..4].copy_from_slice(spending_key);
    input[4..12].copy_from_slice(commitment);
    poseidon2_hash(&input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> SpendingKey {
        [42, 99, 7, 13].map(M31::from_u32_unchecked)
    }

    fn test_note(sk: &SpendingKey) -> Note {
        let pk = derive_pubkey(sk);
        Note::new(
            pk,
            M31::from_u32_unchecked(0),                // STRK
            M31::from_u32_unchecked(1000),             // amount_lo
            M31::from_u32_unchecked(0),                // amount_hi
            [1, 2, 3, 4].map(M31::from_u32_unchecked), // blinding
        )
    }

    #[test]
    fn test_commitment_deterministic() {
        let sk = test_key();
        let note = test_note(&sk);
        let c1 = note.commitment();
        let c2 = note.commitment();
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_commitment_hiding() {
        // Same note data with different blinding → different commitment
        let pk = derive_pubkey(&test_key());
        let note1 = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(100),
            M31::from_u32_unchecked(0),
            [1, 2, 3, 4].map(M31::from_u32_unchecked),
        );
        let note2 = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(100),
            M31::from_u32_unchecked(0),
            [5, 6, 7, 8].map(M31::from_u32_unchecked),
        );
        assert_ne!(
            note1.commitment(),
            note2.commitment(),
            "Different blinding should produce different commitments"
        );
    }

    #[test]
    fn test_commitment_binding() {
        // Different amounts → different commitments
        let pk = derive_pubkey(&test_key());
        let blinding = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let note1 = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(100),
            M31::from_u32_unchecked(0),
            blinding,
        );
        let note2 = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(200),
            M31::from_u32_unchecked(0),
            blinding,
        );
        assert_ne!(note1.commitment(), note2.commitment());
    }

    #[test]
    fn test_nullifier_deterministic() {
        let sk = test_key();
        let note = test_note(&sk);
        let n1 = note.nullifier(&sk);
        let n2 = note.nullifier(&sk);
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_nullifier_different_keys() {
        let sk1 = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let sk2 = [5, 6, 7, 8].map(M31::from_u32_unchecked);
        let pk1 = derive_pubkey(&sk1);
        let note = Note::new(
            pk1,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(100),
            M31::from_u32_unchecked(0),
            [9, 10, 11, 12].map(M31::from_u32_unchecked),
        );
        let n1 = note.nullifier(&sk1);
        let n2 = note.nullifier(&sk2);
        assert_ne!(n1, n2, "Different keys should produce different nullifiers");
    }

    #[test]
    fn test_nullifier_unlinkable() {
        // Two different notes from same owner have different nullifiers
        let sk = test_key();
        let pk = derive_pubkey(&sk);
        let note1 = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(100),
            M31::from_u32_unchecked(0),
            [1, 2, 3, 4].map(M31::from_u32_unchecked),
        );
        let note2 = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(200),
            M31::from_u32_unchecked(0),
            [5, 6, 7, 8].map(M31::from_u32_unchecked),
        );
        assert_ne!(
            note1.nullifier(&sk),
            note2.nullifier(&sk),
            "Different notes should produce different nullifiers"
        );
    }

    #[test]
    fn test_derive_pubkey_deterministic() {
        let sk = test_key();
        let pk1 = derive_pubkey(&sk);
        let pk2 = derive_pubkey(&sk);
        assert_eq!(pk1, pk2);
    }

    #[test]
    fn test_derive_pubkey_different_keys() {
        let sk1 = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let sk2 = [5, 6, 7, 8].map(M31::from_u32_unchecked);
        assert_ne!(derive_pubkey(&sk1), derive_pubkey(&sk2));
    }

    #[test]
    fn test_derive_viewing_key_different_from_pubkey() {
        let sk = test_key();
        let pk = derive_pubkey(&sk);
        let vk = derive_viewing_key(&sk);
        assert_ne!(pk, vk, "Viewing key should differ from public key");
    }

    #[test]
    fn test_compute_nullifier_matches_note_method() {
        let sk = test_key();
        let note = test_note(&sk);
        let commitment = note.commitment();
        let n1 = note.nullifier(&sk);
        let n2 = compute_nullifier(&sk, &commitment);
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_different_assets_different_commitments() {
        let pk = derive_pubkey(&test_key());
        let blinding = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let note_strk = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(100),
            M31::from_u32_unchecked(0),
            blinding,
        );
        let note_eth = Note::new(
            pk,
            M31::from_u32_unchecked(1),
            M31::from_u32_unchecked(100),
            M31::from_u32_unchecked(0),
            blinding,
        );
        assert_ne!(note_strk.commitment(), note_eth.commitment());
    }

    #[test]
    fn test_two_limb_amount() {
        // Verify amount_lo and amount_hi produce different commitments
        let pk = derive_pubkey(&test_key());
        let blinding = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let note_lo = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            blinding,
        );
        let note_hi = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(1000),
            blinding,
        );
        assert_ne!(note_lo.commitment(), note_hi.commitment());
    }

    #[test]
    fn test_validate_zero_spending_key_rejected() {
        let zero_sk = [0, 0, 0, 0].map(M31::from_u32_unchecked);
        assert!(validate_spending_key(&zero_sk).is_err());
    }

    #[test]
    fn test_validate_zero_blinding_rejected() {
        let zero_blinding = [0, 0, 0, 0].map(M31::from_u32_unchecked);
        assert!(validate_blinding(&zero_blinding).is_err());
    }

    #[test]
    fn test_validate_partial_zero_spending_key_ok() {
        // Spending key: 1 non-zero element is sufficient (key ownership, not hiding)
        let partial_sk = [0, 0, 0, 1].map(M31::from_u32_unchecked);
        assert!(validate_spending_key(&partial_sk).is_ok());
    }

    /// H5: single non-zero blinding element (~31 bits) is now rejected.
    #[test]
    fn test_h5_single_nonzero_blinding_rejected() {
        let single = [0, 0, 0, 1].map(M31::from_u32_unchecked);
        assert!(validate_blinding(&single).is_err());
    }

    #[test]
    fn test_validate_normal_key_ok() {
        let sk = test_key();
        assert!(validate_spending_key(&sk).is_ok());
        let blinding = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        assert!(validate_blinding(&blinding).is_ok());
    }

    // ── M7 regression: non-canonical zero (0x7FFFFFFF ≡ 0 mod p) ─────────

    #[test]
    fn test_validate_m7_noncanonical_zero_spending_key_rejected() {
        // M31(0x7FFFFFFF) is semantically zero in the field: p ≡ 0 mod p
        let p_sk = [0x7FFFFFFF; 4].map(M31::from_u32_unchecked);
        assert!(validate_spending_key(&p_sk).is_err());
    }

    #[test]
    fn test_validate_m7_noncanonical_zero_blinding_rejected() {
        let p_blinding = [0x7FFFFFFF; 4].map(M31::from_u32_unchecked);
        assert!(validate_blinding(&p_blinding).is_err());
    }

    #[test]
    fn test_validate_m7_mixed_canonical_noncanonical_zero_rejected() {
        // Mix of 0 and 0x7FFFFFFF — all are field-zero, should be rejected
        let mixed_sk: SpendingKey = [
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(0x7FFFFFFF),
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(0x7FFFFFFF),
        ];
        assert!(validate_spending_key(&mixed_sk).is_err());
    }

    #[test]
    fn test_validate_m7_partial_noncanonical_zero_ok() {
        // One element is non-zero in the field → overall key is non-zero
        let partial: SpendingKey = [
            M31::from_u32_unchecked(0x7FFFFFFF),
            M31::from_u32_unchecked(0x7FFFFFFF),
            M31::from_u32_unchecked(0x7FFFFFFF),
            M31::from_u32_unchecked(1),
        ];
        assert!(validate_spending_key(&partial).is_ok());
    }

    // ── H5 regression: blinding entropy minimum ─────────────────────────

    /// H5: two non-zero elements (≥62 bits) passes.
    #[test]
    fn test_h5_two_nonzero_blinding_ok() {
        let blinding = [0, 0, 42, 99].map(M31::from_u32_unchecked);
        assert!(validate_blinding(&blinding).is_ok());
    }

    /// H5: all-zero blinding still rejected.
    #[test]
    fn test_h5_all_zero_blinding_rejected() {
        let blinding = [0, 0, 0, 0].map(M31::from_u32_unchecked);
        assert!(validate_blinding(&blinding).is_err());
    }

    /// H5: single non-zero with non-canonical zeros still rejected.
    #[test]
    fn test_h5_single_nonzero_with_noncanonical_zeros_rejected() {
        // Three 0x7FFFFFFF (field zero) + one real non-zero = only 1 non-zero
        let blinding: [M31; 4] = [
            M31::from_u32_unchecked(0x7FFFFFFF),
            M31::from_u32_unchecked(0x7FFFFFFF),
            M31::from_u32_unchecked(0x7FFFFFFF),
            M31::from_u32_unchecked(42),
        ];
        assert!(validate_blinding(&blinding).is_err());
    }

    /// H5: four non-zero elements (full ~124 bits) passes.
    #[test]
    fn test_h5_full_blinding_ok() {
        let blinding = [10, 20, 30, 40].map(M31::from_u32_unchecked);
        assert!(validate_blinding(&blinding).is_ok());
    }
}
