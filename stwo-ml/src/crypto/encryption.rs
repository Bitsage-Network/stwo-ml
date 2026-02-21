//! Poseidon2-M31 symmetric encryption for VM31 privacy protocol.
//!
//! Counter-mode encryption using Poseidon2-M31 as the PRF.
//! Key derivation via Poseidon2 hash of shared secret.
//! Designed for encrypting note memos (small, fixed-size payloads).
//!
//! For the audit system: implements the `AuditEncryption` trait behind
//! the `poseidon-m31-encryption` feature flag (default).

use stwo::core::fields::m31::BaseField as M31;

use super::poseidon2_m31::{poseidon2_hash, poseidon2_permutation, RATE, STATE_WIDTH};

/// Domain separation constants.
const DOMAIN_ENCRYPT: M31 = M31::from_u32_unchecked(0x656E6372); // "encr"
const DOMAIN_KDF: M31 = M31::from_u32_unchecked(0x6B646600); // "kdf\0"

/// Derive an encryption key from a shared secret.
/// key = Poseidon2("kdf" || secret[0..n])[0..8]
pub fn derive_key(secret: &[M31]) -> [M31; RATE] {
    let mut input = Vec::with_capacity(1 + secret.len());
    input.push(DOMAIN_KDF);
    input.extend_from_slice(secret);
    poseidon2_hash(&input)
}

/// Generate a keystream block for counter-mode encryption.
/// block = Poseidon2-permutation(key[0..8] || counter || domain || nonce[0..4] || padding)[0..8]
///
/// The nonce occupies capacity positions 10..13, providing ~124 bits of
/// nonce space. Callers MUST use a unique nonce per encryption with the
/// same key to avoid keystream reuse.
fn keystream_block(key: &[M31; RATE], nonce: &[M31; 4], counter: u32) -> [M31; RATE] {
    let mut state = [M31::from_u32_unchecked(0); STATE_WIDTH];
    // Load key into rate portion
    state[..RATE].copy_from_slice(key);
    // Load counter and domain into capacity portion
    state[RATE] = M31::from_u32_unchecked(counter);
    state[RATE + 1] = DOMAIN_ENCRYPT;
    // Load nonce into capacity positions 10..13
    state[RATE + 2] = nonce[0];
    state[RATE + 3] = nonce[1];
    state[RATE + 4] = nonce[2];
    state[RATE + 5] = nonce[3];

    poseidon2_permutation(&mut state);

    let mut block = [M31::from_u32_unchecked(0); RATE];
    block.copy_from_slice(&state[..RATE]);
    block
}

/// Encrypt M31 elements using Poseidon2-M31 counter mode.
///
/// Ciphertext = plaintext + keystream (addition in M31 field).
/// Returns ciphertext of same length as plaintext.
///
/// The `nonce` MUST be unique per encryption with the same key.
/// It is not secret and should be transmitted alongside the ciphertext.
pub fn poseidon2_encrypt(key: &[M31; RATE], nonce: &[M31; 4], plaintext: &[M31]) -> Vec<M31> {
    let mut ciphertext = Vec::with_capacity(plaintext.len());
    let mut counter = 0u32;

    for chunk in plaintext.chunks(RATE) {
        let ks = keystream_block(key, nonce, counter);
        for (i, &pt) in chunk.iter().enumerate() {
            ciphertext.push(pt + ks[i]);
        }
        counter += 1;
    }

    ciphertext
}

/// Decrypt M31 elements using Poseidon2-M31 counter mode.
///
/// Since we use additive keystream, decryption subtracts the same keystream.
/// plaintext = ciphertext - keystream.
///
/// The `nonce` must match the one used during encryption.
pub fn poseidon2_decrypt(key: &[M31; RATE], nonce: &[M31; 4], ciphertext: &[M31]) -> Vec<M31> {
    let mut plaintext = Vec::with_capacity(ciphertext.len());
    let mut counter = 0u32;

    for chunk in ciphertext.chunks(RATE) {
        let ks = keystream_block(key, nonce, counter);
        for (i, &ct) in chunk.iter().enumerate() {
            plaintext.push(ct - ks[i]);
        }
        counter += 1;
    }

    plaintext
}

/// Encrypt a note memo: (asset_id, amount_lo, amount_hi, blinding[0..4]) = 7 M31 elements.
/// Returns: encrypted_memo (7 M31 elements) + checksum (1 M31 element) = 8 elements total.
///
/// The checksum allows recipients to detect valid decryption during scanning.
/// The `nonce` MUST be unique per encryption with the same key.
pub fn encrypt_note_memo(
    key: &[M31; RATE],
    nonce: &[M31; 4],
    asset_id: M31,
    amount_lo: M31,
    amount_hi: M31,
    blinding: &[M31; 4],
) -> [M31; RATE] {
    // Compute checksum: hash of all fields (enables scan-time validation)
    let checksum_input = [
        asset_id,
        amount_lo,
        amount_hi,
        blinding[0],
        blinding[1],
        blinding[2],
        blinding[3],
    ];
    let checksum_hash = poseidon2_hash(&checksum_input);
    let checksum = checksum_hash[0]; // First element as checksum

    let plaintext = [
        asset_id,
        amount_lo,
        amount_hi,
        blinding[0],
        blinding[1],
        blinding[2],
        blinding[3],
        checksum,
    ];

    let encrypted = poseidon2_encrypt(key, nonce, &plaintext);
    let mut result = [M31::from_u32_unchecked(0); RATE];
    result.copy_from_slice(&encrypted);
    result
}

/// Decrypt a note memo. Returns Some((asset_id, amount_lo, amount_hi, blinding)) if
/// the checksum validates, None otherwise (wrong key).
///
/// The `nonce` must match the one used during encryption.
pub fn decrypt_note_memo(
    key: &[M31; RATE],
    nonce: &[M31; 4],
    encrypted_memo: &[M31; RATE],
) -> Option<(M31, M31, M31, [M31; 4])> {
    let decrypted = poseidon2_decrypt(key, nonce, encrypted_memo);

    let asset_id = decrypted[0];
    let amount_lo = decrypted[1];
    let amount_hi = decrypted[2];
    let blinding = [decrypted[3], decrypted[4], decrypted[5], decrypted[6]];
    let checksum = decrypted[7];

    // Verify checksum
    let checksum_input = [
        asset_id,
        amount_lo,
        amount_hi,
        blinding[0],
        blinding[1],
        blinding[2],
        blinding[3],
    ];
    let expected_checksum = poseidon2_hash(&checksum_input)[0];

    if checksum == expected_checksum {
        Some((asset_id, amount_lo, amount_hi, blinding))
    } else {
        None
    }
}

/// Trait for audit system encryption (feature-gated handoff point).
pub trait AuditEncryption {
    /// Encrypt arbitrary bytes for a set of recipients (identified by their M31 keys).
    /// The `nonce` MUST be unique per encryption with the same key.
    fn encrypt_for_keys(
        &self,
        plaintext: &[M31],
        nonce: &[M31; 4],
        recipient_keys: &[[M31; RATE]],
    ) -> Vec<Vec<M31>>;

    /// Decrypt ciphertext with a private key.
    /// The `nonce` must match the one used during encryption.
    fn decrypt_with_key(
        &self,
        ciphertext: &[M31],
        nonce: &[M31; 4],
        key: &[M31; RATE],
    ) -> Option<Vec<M31>>;
}

/// Poseidon2-M31 implementation of AuditEncryption.
pub struct PoseidonM31Encryption;

impl AuditEncryption for PoseidonM31Encryption {
    fn encrypt_for_keys(
        &self,
        plaintext: &[M31],
        nonce: &[M31; 4],
        recipient_keys: &[[M31; RATE]],
    ) -> Vec<Vec<M31>> {
        recipient_keys
            .iter()
            .map(|key| poseidon2_encrypt(key, nonce, plaintext))
            .collect()
    }

    fn decrypt_with_key(
        &self,
        ciphertext: &[M31],
        nonce: &[M31; 4],
        key: &[M31; RATE],
    ) -> Option<Vec<M31>> {
        Some(poseidon2_decrypt(key, nonce, ciphertext))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [M31; RATE] {
        [42, 99, 7, 13, 256, 1024, 5, 11].map(M31::from_u32_unchecked)
    }

    fn test_nonce() -> [M31; 4] {
        [1, 2, 3, 4].map(M31::from_u32_unchecked)
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = test_key();
        let nonce = test_nonce();
        let plaintext: Vec<M31> = (1..=10).map(|i| M31::from_u32_unchecked(i)).collect();

        let ciphertext = poseidon2_encrypt(&key, &nonce, &plaintext);
        let recovered = poseidon2_decrypt(&key, &nonce, &ciphertext);

        assert_eq!(plaintext, recovered);
    }

    #[test]
    fn test_ciphertext_differs_from_plaintext() {
        let key = test_key();
        let nonce = test_nonce();
        let plaintext = [1, 2, 3, 4].map(M31::from_u32_unchecked);

        let ciphertext = poseidon2_encrypt(&key, &nonce, &plaintext);

        // Ciphertext should differ from plaintext
        let differs = ciphertext.iter().zip(plaintext.iter()).any(|(c, p)| c != p);
        assert!(differs, "Ciphertext should differ from plaintext");
    }

    #[test]
    fn test_wrong_key_gives_wrong_plaintext() {
        let key1 = test_key();
        let key2 = [1, 2, 3, 4, 5, 6, 7, 8].map(M31::from_u32_unchecked);
        let nonce = test_nonce();
        let plaintext = [10, 20, 30, 40].map(M31::from_u32_unchecked);

        let ciphertext = poseidon2_encrypt(&key1, &nonce, &plaintext);
        let wrong_decrypted = poseidon2_decrypt(&key2, &nonce, &ciphertext);

        assert_ne!(
            plaintext.to_vec(),
            wrong_decrypted,
            "Wrong key should not recover plaintext"
        );
    }

    #[test]
    fn test_encrypt_empty() {
        let key = test_key();
        let nonce = test_nonce();
        let ciphertext = poseidon2_encrypt(&key, &nonce, &[]);
        assert!(ciphertext.is_empty());
    }

    #[test]
    fn test_encrypt_long_message() {
        let key = test_key();
        let nonce = test_nonce();
        let plaintext: Vec<M31> = (0..100).map(|i| M31::from_u32_unchecked(i + 1)).collect();

        let ciphertext = poseidon2_encrypt(&key, &nonce, &plaintext);
        assert_eq!(ciphertext.len(), 100);

        let recovered = poseidon2_decrypt(&key, &nonce, &ciphertext);
        assert_eq!(plaintext, recovered);
    }

    #[test]
    fn test_note_memo_roundtrip() {
        let key = test_key();
        let nonce = test_nonce();
        let asset_id = M31::from_u32_unchecked(0);
        let amount_lo = M31::from_u32_unchecked(1000);
        let amount_hi = M31::from_u32_unchecked(0);
        let blinding = [1, 2, 3, 4].map(M31::from_u32_unchecked);

        let encrypted = encrypt_note_memo(&key, &nonce, asset_id, amount_lo, amount_hi, &blinding);
        let decrypted = decrypt_note_memo(&key, &nonce, &encrypted);

        assert!(decrypted.is_some(), "Valid key should decrypt");
        let (a, lo, hi, b) = decrypted.unwrap();
        assert_eq!(a, asset_id);
        assert_eq!(lo, amount_lo);
        assert_eq!(hi, amount_hi);
        assert_eq!(b, blinding);
    }

    #[test]
    fn test_note_memo_wrong_key() {
        let key1 = test_key();
        let key2 = [5, 6, 7, 8, 9, 10, 11, 12].map(M31::from_u32_unchecked);
        let nonce = test_nonce();

        let encrypted = encrypt_note_memo(
            &key1,
            &nonce,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            &[1, 2, 3, 4].map(M31::from_u32_unchecked),
        );

        let decrypted = decrypt_note_memo(&key2, &nonce, &encrypted);
        assert!(
            decrypted.is_none(),
            "Wrong key should fail checksum validation"
        );
    }

    #[test]
    fn test_derive_key_deterministic() {
        let secret = [42, 99].map(M31::from_u32_unchecked);
        let k1 = derive_key(&secret);
        let k2 = derive_key(&secret);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_derive_key_different_secrets() {
        let k1 = derive_key(&[M31::from_u32_unchecked(1)]);
        let k2 = derive_key(&[M31::from_u32_unchecked(2)]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_audit_encryption_trait() {
        let enc = PoseidonM31Encryption;
        let key1 = test_key();
        let key2 = [5, 6, 7, 8, 9, 10, 11, 12].map(M31::from_u32_unchecked);
        let nonce = test_nonce();

        let plaintext = [100, 200, 300].map(M31::from_u32_unchecked);
        let ciphertexts = enc.encrypt_for_keys(&plaintext, &nonce, &[key1, key2]);

        assert_eq!(ciphertexts.len(), 2);

        // Each recipient can decrypt with their key
        let d1 = enc
            .decrypt_with_key(&ciphertexts[0], &nonce, &key1)
            .unwrap();
        let d2 = enc
            .decrypt_with_key(&ciphertexts[1], &nonce, &key2)
            .unwrap();

        assert_eq!(d1, plaintext.to_vec());
        assert_eq!(d2, plaintext.to_vec());
    }

    #[test]
    fn test_different_keys_produce_different_ciphertext() {
        let key1 = test_key();
        let key2 = [5, 6, 7, 8, 9, 10, 11, 12].map(M31::from_u32_unchecked);
        let nonce = test_nonce();
        let plaintext = [42].map(M31::from_u32_unchecked);

        let ct1 = poseidon2_encrypt(&key1, &nonce, &plaintext);
        let ct2 = poseidon2_encrypt(&key2, &nonce, &plaintext);

        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_same_key_different_nonce_different_ciphertext() {
        let key = test_key();
        let nonce1 = [1, 2, 3, 4].map(M31::from_u32_unchecked);
        let nonce2 = [5, 6, 7, 8].map(M31::from_u32_unchecked);
        let plaintext = [42].map(M31::from_u32_unchecked);

        let ct1 = poseidon2_encrypt(&key, &nonce1, &plaintext);
        let ct2 = poseidon2_encrypt(&key, &nonce2, &plaintext);
        assert_ne!(
            ct1, ct2,
            "Different nonces must produce different ciphertexts"
        );
    }
}
