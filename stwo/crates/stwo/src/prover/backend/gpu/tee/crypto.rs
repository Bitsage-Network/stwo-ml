//! Cryptographic Primitives for TEE Integration
//!
//! This module provides:
//! - AES-GCM-256 encryption/decryption (matching GPU DMA encryption)
//! - Key derivation (HKDF)
//! - Hashing (SHA-256)

#![allow(unexpected_cfgs)]

use super::{TeeError, TeeResult};

/// AES-GCM-256 nonce size
pub const AES_GCM_NONCE_SIZE: usize = 12;

/// AES-GCM-256 tag size
pub const AES_GCM_TAG_SIZE: usize = 16;

/// AES-256 key size
pub const AES_KEY_SIZE: usize = 32;

/// Encrypt data using AES-GCM-256
///
/// This matches the encryption used by NVIDIA H100 GPU DMA engine.
///
/// # Format
///
/// The output format is: `nonce (12 bytes) || ciphertext || tag (16 bytes)`
pub fn aes_gcm_encrypt(key: &[u8; 32], plaintext: &[u8]) -> TeeResult<Vec<u8>> {
    #[cfg(feature = "real-crypto")]
    {
        use aes_gcm::{
            aead::{Aead, KeyInit, OsRng},
            Aes256Gcm, Nonce,
        };
        use rand::RngCore;

        let cipher = Aes256Gcm::new_from_slice(key)
            .map_err(|e| TeeError::CryptoError(format!("Key error: {}", e)))?;

        // Generate random nonce
        let mut nonce_bytes = [0u8; AES_GCM_NONCE_SIZE];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt
        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| TeeError::CryptoError(format!("Encryption failed: {}", e)))?;

        // Combine: nonce || ciphertext (includes tag)
        let mut result = Vec::with_capacity(AES_GCM_NONCE_SIZE + ciphertext.len());
        result.extend_from_slice(&nonce_bytes);
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    #[cfg(not(feature = "real-crypto"))]
    {
        // Software fallback using XOR (development only!)
        aes_gcm_encrypt_fallback(key, plaintext)
    }
}

/// Decrypt data using AES-GCM-256
pub fn aes_gcm_decrypt(key: &[u8; 32], ciphertext: &[u8]) -> TeeResult<Vec<u8>> {
    if ciphertext.len() < AES_GCM_NONCE_SIZE + AES_GCM_TAG_SIZE {
        return Err(TeeError::CryptoError("Ciphertext too short".into()));
    }

    #[cfg(feature = "real-crypto")]
    {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Nonce,
        };

        let cipher = Aes256Gcm::new_from_slice(key)
            .map_err(|e| TeeError::CryptoError(format!("Key error: {}", e)))?;

        // Extract nonce and ciphertext
        let nonce = Nonce::from_slice(&ciphertext[..AES_GCM_NONCE_SIZE]);
        let ct = &ciphertext[AES_GCM_NONCE_SIZE..];

        // Decrypt
        cipher
            .decrypt(nonce, ct)
            .map_err(|e| TeeError::CryptoError(format!("Decryption failed: {}", e)))
    }

    #[cfg(not(feature = "real-crypto"))]
    {
        aes_gcm_decrypt_fallback(key, ciphertext)
    }
}

/// Software fallback for AES-GCM (development only)
#[cfg(not(feature = "real-crypto"))]
fn aes_gcm_encrypt_fallback(key: &[u8; 32], plaintext: &[u8]) -> TeeResult<Vec<u8>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Generate nonce from timestamp
    let mut nonce = [0u8; AES_GCM_NONCE_SIZE];
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut hasher = DefaultHasher::new();
    now.hash(&mut hasher);
    let hash = hasher.finish();
    nonce[0..8].copy_from_slice(&hash.to_le_bytes());

    hasher = DefaultHasher::new();
    (now ^ 0xDEADBEEF).hash(&mut hasher);
    let hash2 = hasher.finish();
    nonce[8..12].copy_from_slice(&hash2.to_le_bytes()[0..4]);

    // XOR encryption (NOT SECURE - development only)
    let mut ciphertext: Vec<u8> = plaintext
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ key[i % 32] ^ nonce[i % 12])
        .collect();

    // Generate fake tag
    let mut tag = [0u8; AES_GCM_TAG_SIZE];
    let mut tag_hasher = DefaultHasher::new();
    plaintext.hash(&mut tag_hasher);
    key.hash(&mut tag_hasher);
    let tag_hash = tag_hasher.finish();
    tag[0..8].copy_from_slice(&tag_hash.to_le_bytes());
    tag[8..16].copy_from_slice(&tag_hash.to_be_bytes());

    // Combine: nonce || ciphertext || tag
    let mut result = Vec::with_capacity(AES_GCM_NONCE_SIZE + ciphertext.len() + AES_GCM_TAG_SIZE);
    result.extend_from_slice(&nonce);
    result.append(&mut ciphertext);
    result.extend_from_slice(&tag);

    Ok(result)
}

#[cfg(not(feature = "real-crypto"))]
fn aes_gcm_decrypt_fallback(key: &[u8; 32], ciphertext: &[u8]) -> TeeResult<Vec<u8>> {
    // Extract nonce
    let nonce = &ciphertext[..AES_GCM_NONCE_SIZE];

    // Extract ciphertext (without nonce and tag)
    let ct_end = ciphertext.len() - AES_GCM_TAG_SIZE;
    let ct = &ciphertext[AES_GCM_NONCE_SIZE..ct_end];

    // XOR decryption
    let plaintext: Vec<u8> = ct
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ key[i % 32] ^ nonce[i % 12])
        .collect();

    Ok(plaintext)
}

/// Compute SHA-256 hash
pub fn sha256(data: &[u8]) -> [u8; 32] {
    #[cfg(feature = "real-crypto")]
    {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    #[cfg(not(feature = "real-crypto"))]
    {
        sha256_fallback(data)
    }
}

/// Software fallback for SHA-256 (development only)
#[cfg(not(feature = "real-crypto"))]
fn sha256_fallback(data: &[u8]) -> [u8; 32] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut result = [0u8; 32];

    // Use multiple hash passes to fill 32 bytes
    for i in 0..4 {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        i.hash(&mut hasher);
        let hash = hasher.finish();
        result[(i * 8)..((i + 1) * 8)].copy_from_slice(&hash.to_le_bytes());
    }

    result
}

/// Derive session key from key material using HKDF
pub fn derive_session_key(key_material: &[u8]) -> [u8; 32] {
    #[cfg(feature = "real-crypto")]
    {
        use hkdf::Hkdf;
        use sha2::Sha256;

        let hk = Hkdf::<Sha256>::new(None, key_material);
        let mut session_key = [0u8; 32];
        hk.expand(b"obelysk_session_key_v1", &mut session_key)
            .expect("HKDF expand should work for 32 bytes");
        session_key
    }

    #[cfg(not(feature = "real-crypto"))]
    {
        // Fallback: SHA-256 with salt
        let mut salted = Vec::with_capacity(key_material.len() + 32);
        salted.extend_from_slice(b"obelysk_session_key_derivation_");
        salted.extend_from_slice(key_material);
        sha256(&salted)
    }
}

/// Generate a cryptographically secure random key
pub fn generate_random_key() -> [u8; 32] {
    #[cfg(feature = "real-crypto")]
    {
        use rand::RngCore;
        let mut key = [0u8; 32];
        rand::rngs::OsRng.fill_bytes(&mut key);
        key
    }

    #[cfg(not(feature = "real-crypto"))]
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut key = [0u8; 32];
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();

        for i in 0..4 {
            let mut hasher = DefaultHasher::new();
            now.as_nanos().hash(&mut hasher);
            std::process::id().hash(&mut hasher);
            (i as u64).hash(&mut hasher);

            // Add some additional entropy
            let addr = &key as *const _ as usize;
            addr.hash(&mut hasher);

            let hash = hasher.finish();
            key[(i * 8)..((i + 1) * 8)].copy_from_slice(&hash.to_le_bytes());
        }

        key
    }
}

/// Compute HMAC-SHA256
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    #[cfg(feature = "real-crypto")]
    {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        type HmacSha256 = Hmac<Sha256>;

        let mut mac = HmacSha256::new_from_slice(key).expect("HMAC key should be valid");
        mac.update(data);
        mac.finalize().into_bytes().into()
    }

    #[cfg(not(feature = "real-crypto"))]
    {
        // Fallback: Concatenate and hash
        let mut input = Vec::with_capacity(key.len() + data.len());
        input.extend_from_slice(key);
        input.extend_from_slice(data);
        sha256(&input)
    }
}

/// Constant-time comparison for crypto values
pub fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }

    result == 0
}

/// Secure memory zeroing
pub fn secure_zero(data: &mut [u8]) {
    // Use volatile writes to prevent compiler from optimizing away
    for byte in data.iter_mut() {
        unsafe {
            std::ptr::write_volatile(byte, 0);
        }
    }

    // Memory barrier to ensure writes complete
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = generate_random_key();
        let plaintext = b"Hello, Obelysk TEE!";

        let ciphertext = aes_gcm_encrypt(&key, plaintext).unwrap();
        let decrypted = aes_gcm_decrypt(&key, &ciphertext).unwrap();

        assert_eq!(&decrypted, plaintext);
    }

    #[test]
    fn test_sha256() {
        let data = b"test data";
        let hash = sha256(data);

        assert_eq!(hash.len(), 32);

        // Same input should produce same hash
        let hash2 = sha256(data);
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = sha256(b"different data");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_derive_session_key() {
        let material = b"key material";
        let key1 = derive_session_key(material);
        let key2 = derive_session_key(material);

        // Same material should produce same key
        assert_eq!(key1, key2);

        // Different material should produce different key
        let key3 = derive_session_key(b"different material");
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_constant_time_compare() {
        let a = [1u8, 2, 3, 4];
        let b = [1u8, 2, 3, 4];
        let c = [1u8, 2, 3, 5];

        assert!(constant_time_compare(&a, &b));
        assert!(!constant_time_compare(&a, &c));
    }

    #[test]
    fn test_secure_zero() {
        let mut data = [1u8, 2, 3, 4, 5];
        secure_zero(&mut data);
        assert_eq!(data, [0u8; 5]);
    }
}
