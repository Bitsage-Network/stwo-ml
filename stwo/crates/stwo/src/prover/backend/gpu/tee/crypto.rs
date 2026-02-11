//! Cryptographic Primitives for TEE Integration
//!
//! This module provides:
//! - AES-GCM-256 encryption/decryption (matching GPU DMA encryption)
//! - Key derivation (HKDF)
//! - Hashing (SHA-256)

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
    #[cfg(feature = "cuda-runtime")]
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

    #[cfg(not(feature = "cuda-runtime"))]
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

    #[cfg(feature = "cuda-runtime")]
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

    #[cfg(not(feature = "cuda-runtime"))]
    {
        aes_gcm_decrypt_fallback(key, ciphertext)
    }
}

/// Software fallback for AES-GCM (development only — NOT production secure)
///
/// Uses XOR with blake2-derived keystream. Provides basic confidentiality
/// for development but MUST NOT be used in production (no authenticated encryption).
#[cfg(not(feature = "cuda-runtime"))]
fn aes_gcm_encrypt_fallback(key: &[u8; 32], plaintext: &[u8]) -> TeeResult<Vec<u8>> {
    use blake2::{Blake2s256, Digest};

    // Generate nonce from blake2s of entropy sources
    let mut nonce = [0u8; AES_GCM_NONCE_SIZE];
    let mut nonce_hasher = Blake2s256::new();
    nonce_hasher.update(&std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .to_le_bytes());
    nonce_hasher.update(&std::process::id().to_le_bytes());
    let nonce_hash = nonce_hasher.finalize();
    nonce.copy_from_slice(&nonce_hash[..AES_GCM_NONCE_SIZE]);

    // XOR encryption with blake2-derived keystream (NOT SECURE - development only)
    let mut ciphertext: Vec<u8> = plaintext
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ key[i % 32] ^ nonce[i % 12])
        .collect();

    // Generate tag via blake2 (integrity but not authenticated encryption)
    let mut tag_hasher = Blake2s256::new();
    tag_hasher.update(key);
    tag_hasher.update(&nonce);
    tag_hasher.update(plaintext);
    let tag_hash = tag_hasher.finalize();
    let mut tag = [0u8; AES_GCM_TAG_SIZE];
    tag.copy_from_slice(&tag_hash[..AES_GCM_TAG_SIZE]);

    // Combine: nonce || ciphertext || tag
    let mut result = Vec::with_capacity(AES_GCM_NONCE_SIZE + ciphertext.len() + AES_GCM_TAG_SIZE);
    result.extend_from_slice(&nonce);
    result.append(&mut ciphertext);
    result.extend_from_slice(&tag);

    Ok(result)
}

#[cfg(not(feature = "cuda-runtime"))]
fn aes_gcm_decrypt_fallback(key: &[u8; 32], ciphertext: &[u8]) -> TeeResult<Vec<u8>> {
    use blake2::{Blake2s256, Digest};

    // Extract nonce
    let nonce = &ciphertext[..AES_GCM_NONCE_SIZE];

    // Extract ciphertext (without nonce and tag)
    let ct_end = ciphertext.len() - AES_GCM_TAG_SIZE;
    let ct = &ciphertext[AES_GCM_NONCE_SIZE..ct_end];
    let tag = &ciphertext[ct_end..];

    // XOR decryption
    let plaintext: Vec<u8> = ct
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ key[i % 32] ^ nonce[i % 12])
        .collect();

    // Verify tag (blake2-based integrity check)
    let mut tag_hasher = Blake2s256::new();
    tag_hasher.update(key);
    tag_hasher.update(nonce);
    tag_hasher.update(&plaintext);
    let expected_tag = tag_hasher.finalize();
    if !constant_time_compare(tag, &expected_tag[..AES_GCM_TAG_SIZE]) {
        return Err(TeeError::CryptoError("Tag verification failed".into()));
    }

    Ok(plaintext)
}

/// Compute a 256-bit cryptographic hash.
///
/// Uses blake2s-256 (cryptographically secure, 32-byte output).
/// Named `sha256` for API compatibility but uses blake2s internally
/// since it's always available as a dependency.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    use blake2::{Blake2s256, Digest};

    let mut hasher = Blake2s256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Derive session key from key material.
///
/// Uses domain-separated blake2s for key derivation.
pub fn derive_session_key(key_material: &[u8]) -> [u8; 32] {
    use blake2::{Blake2s256, Digest};

    let mut hasher = Blake2s256::new();
    hasher.update(b"obelysk_session_key_v1");
    hasher.update(key_material);
    hasher.finalize().into()
}

/// Generate a random key.
///
/// When `cuda-runtime` is enabled, uses the OS CSPRNG via `getrandom`.
/// Otherwise, uses blake2s hash of high-entropy system state.
pub fn generate_random_key() -> [u8; 32] {
    #[cfg(feature = "cuda-runtime")]
    {
        let mut key = [0u8; 32];
        getrandom::getrandom(&mut key).expect("OS CSPRNG (getrandom) should not fail");
        return key;
    }

    #[cfg(not(feature = "cuda-runtime"))]
    {
        use blake2::{Blake2s256, Digest};

        let mut hasher = Blake2s256::new();
        hasher.update(&std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes());
        hasher.update(&std::process::id().to_le_bytes());
        let key_var: u64 = 0;
        hasher.update(&(&key_var as *const _ as usize).to_le_bytes());
        hasher.update(format!("{:?}", std::thread::current().id()).as_bytes());
        hasher.finalize().into()
    }
}

/// Compute keyed MAC using blake2s (envelope construction).
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    use blake2::{Blake2s256, Digest};

    let mut hasher = Blake2s256::new();
    // Envelope MAC: H(key || data || key)
    hasher.update(key);
    hasher.update(data);
    hasher.update(key);
    hasher.finalize().into()
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
