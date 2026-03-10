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
const DOMAIN_MAC: M31 = M31::from_u32_unchecked(0x6D616300); // "mac\0"
const DOMAIN_SIV: M31 = M31::from_u32_unchecked(0x73697600); // "siv\0"

/// Maximum message length in M31 elements for counter-mode encryption.
/// RATE * u32::MAX ≈ 34 billion elements — checked before the encrypt loop.
const MAX_MESSAGE_LEN: usize = RATE * (u32::MAX as usize);

/// Maximum encryptions per key before rotation is needed (~1 trillion).
const MAX_ENCRYPTIONS_PER_KEY: u64 = 1 << 40;

/// Encryption error type.
#[derive(Debug, thiserror::Error)]
pub enum EncryptError {
    #[error("empty plaintext")]
    EmptyPlaintext,
    #[error("message too large: {0} elements exceeds counter space")]
    MessageTooLarge(usize),
    #[error("key rotation needed: {0} encryptions performed")]
    KeyRotationNeeded(u64),
    #[error("RNG failure")]
    RngFailed,
}

/// SIV ciphertext: nonce derived from content, ciphertext, and MAC tag.
pub struct SivCiphertext {
    pub nonce: [M31; 4],
    pub ciphertext: Vec<M31>,
    pub mac: [M31; 4],
}

/// Tracks how many encryptions have been performed with a key.
///
/// After `MAX_ENCRYPTIONS_PER_KEY` (~1 trillion) encryptions, the tracker
/// returns an error signaling the caller should rotate the key.
pub struct KeyUsageTracker {
    encryptions: u64,
}

impl KeyUsageTracker {
    pub fn new() -> Self {
        Self { encryptions: 0 }
    }

    /// Record one encryption. Returns `Err` if the key has been used too many times.
    pub fn record_encryption(&mut self) -> Result<(), EncryptError> {
        self.encryptions += 1;
        if self.encryptions > MAX_ENCRYPTIONS_PER_KEY {
            return Err(EncryptError::KeyRotationNeeded(self.encryptions));
        }
        Ok(())
    }

    /// Number of remaining encryptions before rotation is needed.
    pub fn remaining(&self) -> u64 {
        MAX_ENCRYPTIONS_PER_KEY.saturating_sub(self.encryptions)
    }
}

/// Generate a cryptographically secure random nonce (4 M31 elements, ~124 bits).
///
/// Uses `getrandom` crate for OS-level entropy.
/// Available when `audit`, `cli`, or `server` features are enabled.
#[cfg(any(feature = "audit", feature = "cli", feature = "server"))]
pub fn generate_secure_nonce() -> Result<[M31; 4], EncryptError> {
    const P: u32 = 0x7FFFFFFF; // M31 prime = 2^31 - 1
    let mut result = [M31::from_u32_unchecked(0); 4];
    for elem in result.iter_mut() {
        loop {
            let mut buf = [0u8; 4];
            getrandom::getrandom(&mut buf).map_err(|_| EncryptError::RngFailed)?;
            let v = u32::from_le_bytes(buf) >> 1; // 31 bits, uniform in [0, 2^31)
            if v < P {
                *elem = M31::from_u32_unchecked(v);
                break;
            }
            // v == P: reject and retry (probability ~4.7e-10)
        }
    }
    Ok(result)
}

/// Derive a synthetic nonce from key + plaintext (SIV construction).
///
/// nonce = Poseidon2(DOMAIN_SIV || key || plaintext)[0..4]
fn derive_synthetic_nonce(key: &[M31; RATE], plaintext: &[M31]) -> [M31; 4] {
    let mut input = Vec::with_capacity(1 + RATE + plaintext.len());
    input.push(DOMAIN_SIV);
    input.extend_from_slice(key);
    input.extend_from_slice(plaintext);
    let hash = poseidon2_hash(&input);
    [hash[0], hash[1], hash[2], hash[3]]
}

/// SIV encrypt: derive nonce from content, then counter-mode encrypt.
///
/// Returns `SivCiphertext` containing the derived nonce, ciphertext, and MAC.
/// Nonce reuse is structurally impossible since the nonce is derived from
/// `Hash(key, plaintext)` — identical plaintext produces the same ciphertext
/// (deterministic), but different plaintext always gets a unique nonce.
pub fn poseidon2_encrypt_siv(
    key: &[M31; RATE],
    plaintext: &[M31],
) -> Result<SivCiphertext, EncryptError> {
    if plaintext.is_empty() {
        return Err(EncryptError::EmptyPlaintext);
    }
    if plaintext.len() > MAX_MESSAGE_LEN {
        return Err(EncryptError::MessageTooLarge(plaintext.len()));
    }
    let nonce = derive_synthetic_nonce(key, plaintext);
    let ciphertext = poseidon2_encrypt_checked(key, &nonce, plaintext)?;
    let mac = compute_mac(key, &nonce, &ciphertext);
    Ok(SivCiphertext {
        nonce,
        ciphertext,
        mac,
    })
}

/// Decrypt a SIV ciphertext. Verifies MAC, then decrypts, then verifies the
/// nonce matches `Hash(key, recovered_plaintext)`.
pub fn poseidon2_decrypt_siv(
    key: &[M31; RATE],
    siv: &SivCiphertext,
) -> Option<Vec<M31>> {
    // Verify MAC before decryption
    let expected_mac = compute_mac(key, &siv.nonce, &siv.ciphertext);
    if !verify_mac(&expected_mac, &siv.mac) {
        return None;
    }
    let plaintext = poseidon2_decrypt(key, &siv.nonce, &siv.ciphertext);
    // Verify nonce matches derived nonce (SIV integrity)
    let expected_nonce = derive_synthetic_nonce(key, &plaintext);
    if expected_nonce != siv.nonce {
        return None;
    }
    Some(plaintext)
}

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

/// Encrypt M31 elements using Poseidon2-M31 counter mode (checked variant).
///
/// Returns `Err` on empty plaintext or if the message exceeds counter space.
/// Prefer `poseidon2_encrypt_siv()` for new code to eliminate nonce management.
pub fn poseidon2_encrypt_checked(
    key: &[M31; RATE],
    nonce: &[M31; 4],
    plaintext: &[M31],
) -> Result<Vec<M31>, EncryptError> {
    if plaintext.is_empty() {
        return Err(EncryptError::EmptyPlaintext);
    }
    if plaintext.len() > MAX_MESSAGE_LEN {
        return Err(EncryptError::MessageTooLarge(plaintext.len()));
    }
    let mut ciphertext = Vec::with_capacity(plaintext.len());
    let mut counter = 0u32;

    for chunk in plaintext.chunks(RATE) {
        let ks = keystream_block(key, nonce, counter);
        for (i, &pt) in chunk.iter().enumerate() {
            ciphertext.push(pt + ks[i]);
        }
        counter = counter.checked_add(1).ok_or_else(|| {
            EncryptError::MessageTooLarge(plaintext.len())
        })?;
    }

    Ok(ciphertext)
}

/// Encrypt M31 elements using Poseidon2-M31 counter mode.
///
/// Ciphertext = plaintext + keystream (addition in M31 field).
/// Returns ciphertext of same length as plaintext.
///
/// Callers MUST use a unique nonce per encryption with the same key;
/// use `getrandom` for random nonces. Nonce reuse breaks counter-mode security.
///
/// **Deprecated**: prefer `poseidon2_encrypt_siv()` which derives nonces from
/// content, making nonce reuse structurally impossible.
///
/// # Panics
///
/// Panics if `plaintext` is empty.
pub fn poseidon2_encrypt(key: &[M31; RATE], nonce: &[M31; 4], plaintext: &[M31]) -> Vec<M31> {
    poseidon2_encrypt_checked(key, nonce, plaintext)
        .expect("BUG: encrypting empty or oversized plaintext")
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

/// Compute a MAC (Message Authentication Code) over ciphertext using Poseidon2.
///
/// MAC = Poseidon2(DOMAIN_MAC || key || nonce || ciphertext)[0..4]
///
/// This provides ciphertext integrity: any bit-flip in the ciphertext will
/// cause MAC verification to fail (IND-CCA2 security via Encrypt-then-MAC).
pub fn compute_mac(key: &[M31; RATE], nonce: &[M31; 4], ciphertext: &[M31]) -> [M31; 4] {
    let mut input = Vec::with_capacity(1 + RATE + 4 + ciphertext.len());
    input.push(DOMAIN_MAC);
    input.extend_from_slice(key);
    input.extend_from_slice(nonce);
    input.extend_from_slice(ciphertext);
    let hash = poseidon2_hash(&input);
    [hash[0], hash[1], hash[2], hash[3]]
}

/// Verify a MAC tag against the expected value (constant-time comparison).
fn verify_mac(expected: &[M31; 4], actual: &[M31; 4]) -> bool {
    // Use XOR accumulation to avoid early-exit timing side-channels.
    let mut diff = 0u32;
    for i in 0..4 {
        diff |= expected[i].0 ^ actual[i].0;
    }
    diff == 0
}

/// Encrypt a note memo with Encrypt-then-MAC authentication.
///
/// Payload: (asset_id, amount_lo, amount_hi, blinding[0..4]) = 7 M31 elements.
/// Returns: encrypted payload (7 M31) + MAC tag (4 M31) = 11 elements total.
///
/// The MAC provides ciphertext integrity: any tampering is detected by
/// `decrypt_note_memo` before decryption is returned.
///
/// The `nonce` MUST be unique per encryption with the same key.
///
/// **Deprecated**: prefer `poseidon2_encrypt_siv()` which derives nonces from
/// content, making nonce reuse structurally impossible. Combine
/// `siv.ciphertext` + `siv.mac` for the same 11-element format.
pub fn encrypt_note_memo(
    key: &[M31; RATE],
    nonce: &[M31; 4],
    asset_id: M31,
    amount_lo: M31,
    amount_hi: M31,
    blinding: &[M31; 4],
) -> Vec<M31> {
    let plaintext = [
        asset_id,
        amount_lo,
        amount_hi,
        blinding[0],
        blinding[1],
        blinding[2],
        blinding[3],
    ];

    let ciphertext = poseidon2_encrypt(key, nonce, &plaintext);
    let mac = compute_mac(key, nonce, &ciphertext);

    let mut result = Vec::with_capacity(ciphertext.len() + 4);
    result.extend_from_slice(&ciphertext);
    result.extend_from_slice(&mac);
    result
}

/// Decrypt a note memo with MAC verification (Encrypt-then-MAC).
///
/// Verifies the MAC tag BEFORE decrypting. Returns `None` if:
/// - MAC verification fails (ciphertext was tampered with)
/// - Wrong key was used
///
/// Accepts both the new 11-element format (7 ciphertext + 4 MAC) and the
/// legacy 8-element format (7 payload + 1 checksum, encrypted together) for
/// backward compatibility with existing on-chain memos.
pub fn decrypt_note_memo(
    key: &[M31; RATE],
    nonce: &[M31; 4],
    encrypted_memo: &[M31],
) -> Option<(M31, M31, M31, [M31; 4])> {
    if encrypted_memo.len() == 11 {
        // New authenticated format: 7 ciphertext + 4 MAC
        let ciphertext = &encrypted_memo[..7];
        let mac_tag = [
            encrypted_memo[7],
            encrypted_memo[8],
            encrypted_memo[9],
            encrypted_memo[10],
        ];

        // Verify MAC BEFORE decryption (Encrypt-then-MAC)
        let expected_mac = compute_mac(key, nonce, ciphertext);
        if !verify_mac(&expected_mac, &mac_tag) {
            return None;
        }

        let decrypted = poseidon2_decrypt(key, nonce, ciphertext);
        let asset_id = decrypted[0];
        let amount_lo = decrypted[1];
        let amount_hi = decrypted[2];
        let blinding = [decrypted[3], decrypted[4], decrypted[5], decrypted[6]];
        Some((asset_id, amount_lo, amount_hi, blinding))
    } else if encrypted_memo.len() == RATE {
        // Legacy format: 8 encrypted elements (7 payload + 1 checksum)
        let decrypted = poseidon2_decrypt(key, nonce, encrypted_memo);

        let asset_id = decrypted[0];
        let amount_lo = decrypted[1];
        let amount_hi = decrypted[2];
        let blinding = [decrypted[3], decrypted[4], decrypted[5], decrypted[6]];
        let checksum = decrypted[7];

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

    /// Decrypt ciphertext with a private key (raw, no MAC verification).
    ///
    /// WARNING: This does NOT verify ciphertext integrity. Callers MUST use
    /// `decrypt_note_memo` (which checks the Encrypt-then-MAC tag) for any
    /// security-sensitive path. This raw decryption is only suitable for
    /// audit log replay where integrity is verified externally.
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
    #[should_panic(expected = "EmptyPlaintext")]
    fn test_encrypt_empty_panics() {
        let key = test_key();
        let nonce = test_nonce();
        let _ = poseidon2_encrypt(&key, &nonce, &[]);
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
        assert_eq!(
            encrypted.len(),
            11,
            "C4: new format should be 7 ciphertext + 4 MAC"
        );
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
            "Wrong key should fail MAC verification"
        );
    }

    #[test]
    fn test_c4_mac_detects_ciphertext_tampering() {
        let key = test_key();
        let nonce = test_nonce();

        let mut encrypted = encrypt_note_memo(
            &key,
            &nonce,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            &[1, 2, 3, 4].map(M31::from_u32_unchecked),
        );

        // Flip a single ciphertext element
        encrypted[0] = encrypted[0] + M31::from_u32_unchecked(1);

        let decrypted = decrypt_note_memo(&key, &nonce, &encrypted);
        assert!(
            decrypted.is_none(),
            "C4: tampered ciphertext must fail MAC verification"
        );
    }

    #[test]
    fn test_c4_mac_detects_mac_tampering() {
        let key = test_key();
        let nonce = test_nonce();

        let mut encrypted = encrypt_note_memo(
            &key,
            &nonce,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(500),
            M31::from_u32_unchecked(0),
            &[5, 6, 7, 8].map(M31::from_u32_unchecked),
        );

        // Flip a MAC element (last 4 elements)
        let mac_idx = encrypted.len() - 1;
        encrypted[mac_idx] = encrypted[mac_idx] + M31::from_u32_unchecked(1);

        let decrypted = decrypt_note_memo(&key, &nonce, &encrypted);
        assert!(
            decrypted.is_none(),
            "C4: tampered MAC must fail verification"
        );
    }

    #[test]
    fn test_c4_legacy_8_element_format_still_decrypts() {
        // Test backward compatibility with old 8-element format
        let key = test_key();
        let nonce = test_nonce();
        let asset_id = M31::from_u32_unchecked(0);
        let amount_lo = M31::from_u32_unchecked(1000);
        let amount_hi = M31::from_u32_unchecked(0);
        let blinding = [1, 2, 3, 4].map(M31::from_u32_unchecked);

        // Manually construct legacy format (7 payload + 1 checksum, encrypted)
        let checksum_input = [
            asset_id,
            amount_lo,
            amount_hi,
            blinding[0],
            blinding[1],
            blinding[2],
            blinding[3],
        ];
        let checksum = poseidon2_hash(&checksum_input)[0];
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
        let legacy_encrypted = poseidon2_encrypt(&key, &nonce, &plaintext);
        let legacy_array: [M31; RATE] = legacy_encrypted.try_into().unwrap();

        let decrypted = decrypt_note_memo(&key, &nonce, &legacy_array);
        assert!(
            decrypted.is_some(),
            "C4: legacy 8-element format must still decrypt"
        );
        let (a, lo, hi, b) = decrypted.unwrap();
        assert_eq!(a, asset_id);
        assert_eq!(lo, amount_lo);
        assert_eq!(hi, amount_hi);
        assert_eq!(b, blinding);
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

    // ── SIV tests ──────────────────────────────────────────────

    #[test]
    fn test_siv_encrypt_decrypt_roundtrip() {
        let key = test_key();
        let plaintext: Vec<M31> = (1..=10).map(|i| M31::from_u32_unchecked(i)).collect();

        let siv = poseidon2_encrypt_siv(&key, &plaintext).expect("encrypt");
        let recovered = poseidon2_decrypt_siv(&key, &siv);
        assert_eq!(recovered, Some(plaintext));
    }

    #[test]
    fn test_siv_deterministic() {
        let key = test_key();
        let plaintext: Vec<M31> = (1..=5).map(|i| M31::from_u32_unchecked(i)).collect();

        let siv1 = poseidon2_encrypt_siv(&key, &plaintext).expect("encrypt 1");
        let siv2 = poseidon2_encrypt_siv(&key, &plaintext).expect("encrypt 2");

        assert_eq!(siv1.nonce, siv2.nonce, "SIV nonces should be deterministic");
        assert_eq!(siv1.ciphertext, siv2.ciphertext);
        assert_eq!(siv1.mac, siv2.mac);
    }

    #[test]
    fn test_siv_different_plaintext_different_nonce() {
        let key = test_key();
        let pt1: Vec<M31> = (1..=5).map(|i| M31::from_u32_unchecked(i)).collect();
        let pt2: Vec<M31> = (10..=14).map(|i| M31::from_u32_unchecked(i)).collect();

        let siv1 = poseidon2_encrypt_siv(&key, &pt1).expect("encrypt 1");
        let siv2 = poseidon2_encrypt_siv(&key, &pt2).expect("encrypt 2");

        assert_ne!(siv1.nonce, siv2.nonce, "Different plaintext → different nonce");
    }

    #[test]
    fn test_siv_tampered_ciphertext_fails() {
        let key = test_key();
        let plaintext: Vec<M31> = (1..=5).map(|i| M31::from_u32_unchecked(i)).collect();

        let mut siv = poseidon2_encrypt_siv(&key, &plaintext).expect("encrypt");
        siv.ciphertext[0] = siv.ciphertext[0] + M31::from_u32_unchecked(1);

        assert!(poseidon2_decrypt_siv(&key, &siv).is_none(), "Tampered ciphertext must fail");
    }

    #[test]
    fn test_siv_tampered_mac_fails() {
        let key = test_key();
        let plaintext: Vec<M31> = (1..=5).map(|i| M31::from_u32_unchecked(i)).collect();

        let mut siv = poseidon2_encrypt_siv(&key, &plaintext).expect("encrypt");
        siv.mac[0] = siv.mac[0] + M31::from_u32_unchecked(1);

        assert!(poseidon2_decrypt_siv(&key, &siv).is_none(), "Tampered MAC must fail");
    }

    #[test]
    fn test_siv_wrong_key_fails() {
        let key1 = test_key();
        let key2 = [5, 6, 7, 8, 9, 10, 11, 12].map(M31::from_u32_unchecked);
        let plaintext: Vec<M31> = (1..=5).map(|i| M31::from_u32_unchecked(i)).collect();

        let siv = poseidon2_encrypt_siv(&key1, &plaintext).expect("encrypt");
        assert!(poseidon2_decrypt_siv(&key2, &siv).is_none(), "Wrong key must fail");
    }

    #[test]
    fn test_siv_empty_plaintext_error() {
        let key = test_key();
        let result = poseidon2_encrypt_siv(&key, &[]);
        assert!(matches!(result, Err(EncryptError::EmptyPlaintext)));
    }

    // ── Checked encrypt tests ──────────────────────────────────

    #[test]
    fn test_checked_encrypt_empty_error() {
        let key = test_key();
        let nonce = test_nonce();
        let result = poseidon2_encrypt_checked(&key, &nonce, &[]);
        assert!(matches!(result, Err(EncryptError::EmptyPlaintext)));
    }

    #[test]
    fn test_checked_encrypt_roundtrip() {
        let key = test_key();
        let nonce = test_nonce();
        let plaintext: Vec<M31> = (1..=20).map(|i| M31::from_u32_unchecked(i)).collect();

        let ct = poseidon2_encrypt_checked(&key, &nonce, &plaintext).expect("encrypt");
        let recovered = poseidon2_decrypt(&key, &nonce, &ct);
        assert_eq!(plaintext, recovered);
    }

    // ── Key rotation tracker tests ─────────────────────────────

    #[test]
    fn test_key_usage_tracker_basic() {
        let mut tracker = KeyUsageTracker::new();
        assert!(tracker.remaining() > 0);
        tracker.record_encryption().expect("first use ok");
        assert_eq!(tracker.remaining(), MAX_ENCRYPTIONS_PER_KEY - 1);
    }

    #[test]
    fn test_key_usage_tracker_limit() {
        let mut tracker = KeyUsageTracker { encryptions: MAX_ENCRYPTIONS_PER_KEY };
        let result = tracker.record_encryption();
        assert!(matches!(result, Err(EncryptError::KeyRotationNeeded(_))));
    }

    // ── Secure nonce tests ─────────────────────────────────────

    #[cfg(any(feature = "audit", feature = "cli", feature = "server"))]
    #[test]
    fn test_generate_secure_nonce() {
        let n1 = generate_secure_nonce().expect("nonce 1");
        let n2 = generate_secure_nonce().expect("nonce 2");
        // With 124 bits of entropy, collision probability is negligible
        assert_ne!(n1, n2, "Two random nonces should differ");
    }
}
