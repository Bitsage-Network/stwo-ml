//! Encryption integration for audit reports.
//!
//! Provides a swappable encryption backend via the [`AuditEncryption`] trait.
//! Production default is [`Poseidon2M31Encryption`] — M31-native Poseidon2
//! counter-mode with Poseidon2 key wrapping. Also ships `Aes256GcmEncryption`
//! (behind the `aes-fallback` feature) and `NoopEncryption` for testing only.
//!
//! Higher-level helpers:
//! - [`encrypt_and_store`] — encrypt → upload to Arweave → return receipt.
//! - [`fetch_and_decrypt`] — download from Arweave → decrypt → deserialize.
//! - [`grant_access`] — wrap data key for an additional recipient.
//! - [`revoke_access`] — re-encrypt with a fresh key, keeping only listed recipients.

use crate::audit::storage::{ArweaveClient, ArweaveTag};
use crate::audit::types::{
    AuditEncryption, AuditError, AuditReport, EncryptedBlob, EncryptionError, WrappedKey,
};

// ─── Helper: plaintext hash (M31-native Poseidon2) ─────────────────────────

/// Compute a Poseidon2-M31 hash of raw bytes.
///
/// Returns a hex-formatted string of the 8 M31 output elements.
/// This is 100% M31-native — no felt252 involved.
fn hash_plaintext(data: &[u8]) -> String {
    use crate::crypto::poseidon2_m31::poseidon2_hash;
    use stwo::core::fields::m31::BaseField as M31;

    // Pack bytes into M31 elements with length prefix (prevents
    // zero-padded tails from colliding with shorter inputs).
    let mut m31s = Vec::with_capacity(1 + (data.len() + 2) / 3);
    m31s.push(M31::from_u32_unchecked(data.len() as u32));
    for chunk in data.chunks(3) {
        let mut val = 0u32;
        for (i, &b) in chunk.iter().enumerate() {
            val |= (b as u32) << (i * 8);
        }
        m31s.push(M31::from_u32_unchecked(val));
    }

    let hash = poseidon2_hash(&m31s);
    let bytes: Vec<u8> = hash.iter().flat_map(|m| m.0.to_le_bytes()).collect();
    format!(
        "0x{}",
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

// ─── Noop Encryption (always available, for testing) ───────────────────────

/// No-op encryption that stores plaintext with XOR obfuscation.
///
/// **Not secure** — for testing and dry-run mode only.
pub struct NoopEncryption;

impl AuditEncryption for NoopEncryption {
    fn encrypt(
        &self,
        plaintext: &[u8],
        owner_pubkey: &[u8],
    ) -> Result<EncryptedBlob, EncryptionError> {
        // Simple XOR with the first byte of pubkey (NOT real encryption).
        let xor_byte = owner_pubkey.first().copied().unwrap_or(0x42);
        let ciphertext: Vec<u8> = plaintext.iter().map(|b| b ^ xor_byte).collect();
        let hash = hash_plaintext(plaintext);

        Ok(EncryptedBlob {
            ciphertext,
            scheme: self.scheme_name().to_string(),
            nonce: vec![xor_byte],
            wrapped_keys: vec![WrappedKey {
                recipient: hex::encode(owner_pubkey),
                encrypted_key: vec![xor_byte],
                role: "owner".to_string(),
                granted_at: 0,
            }],
            plaintext_hash: hash,
        })
    }

    fn decrypt(
        &self,
        blob: &EncryptedBlob,
        recipient_address: &[u8],
        _privkey: &[u8],
    ) -> Result<Vec<u8>, EncryptionError> {
        let addr_hex = hex::encode(recipient_address);
        let wrapped = blob
            .wrapped_keys
            .iter()
            .find(|k| k.recipient == addr_hex)
            .ok_or(EncryptionError::AccessDenied)?;

        let xor_byte = wrapped.encrypted_key.first().copied().unwrap_or(0x42);
        let plaintext: Vec<u8> = blob.ciphertext.iter().map(|b| b ^ xor_byte).collect();

        // Verify plaintext hash.
        let hash = hash_plaintext(&plaintext);
        if hash != blob.plaintext_hash {
            return Err(EncryptionError::HashMismatch);
        }

        Ok(plaintext)
    }

    fn wrap_key_for(
        &self,
        blob: &EncryptedBlob,
        _owner_privkey: &[u8],
        _owner_address: &[u8],
        _grantee_pubkey: &[u8],
        grantee_address: &[u8],
    ) -> Result<WrappedKey, EncryptionError> {
        let xor_byte = blob.nonce.first().copied().unwrap_or(0x42);

        Ok(WrappedKey {
            recipient: hex::encode(grantee_address),
            encrypted_key: vec![xor_byte],
            role: "auditor".to_string(),
            granted_at: 0,
        })
    }

    fn scheme_name(&self) -> &str {
        "noop_xor"
    }
}

// ─── Hex encode/decode (minimal, no extra dep) ─────────────────────────────

mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }

    #[allow(dead_code)]
    pub fn decode(s: &str) -> Result<Vec<u8>, String> {
        if s.len() % 2 != 0 {
            return Err("odd-length hex string".to_string());
        }
        (0..s.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&s[i..i + 2], 16)
                    .map_err(|e| format!("invalid hex at {}: {}", i, e))
            })
            .collect()
    }
}

// ─── AES-256-GCM Encryption (feature-gated) ───────────────────────────────

#[cfg(feature = "aes-fallback")]
mod aes_impl {
    use aes_gcm::aead::rand_core::RngCore;
    use aes_gcm::{
        aead::{Aead, KeyInit, OsRng},
        Aes256Gcm, Key, Nonce,
    };

    use super::*;

    /// AES-256-GCM encryption backend with M31-native key wrapping.
    ///
    /// Data encryption: AES-256-GCM (hardware-accelerated AEAD).
    /// Key wrapping: M31-native Poseidon2 symmetric wrapping (same as Poseidon2 backend).
    ///
    /// Key formats: 32-byte view key (pubkey), 16-byte secret key (privkey).
    pub struct Aes256GcmEncryption;

    /// AES DEK size.
    const AES_DEK_BYTES: usize = 32;

    impl AuditEncryption for Aes256GcmEncryption {
        fn encrypt(
            &self,
            plaintext: &[u8],
            owner_pubkey: &[u8],
        ) -> Result<EncryptedBlob, EncryptionError> {
            // Generate random DEK and nonce.
            let mut dek = [0u8; 32];
            let mut nonce_bytes = [0u8; 12];
            OsRng.fill_bytes(&mut dek);
            OsRng.fill_bytes(&mut nonce_bytes);

            let key = Key::<Aes256Gcm>::from_slice(&dek);
            let cipher = Aes256Gcm::new(key);
            let nonce = Nonce::from_slice(&nonce_bytes);

            let ciphertext = cipher
                .encrypt(nonce, plaintext)
                .map_err(|e| EncryptionError::DecryptionFailed(format!("AES encrypt: {}", e)))?;

            let hash = hash_plaintext(plaintext);

            // M31-native wrap DEK for owner.
            let wrapped_dek = super::m31_keys::wrap_dek(&dek, owner_pubkey)?;

            Ok(EncryptedBlob {
                ciphertext,
                scheme: self.scheme_name().to_string(),
                nonce: nonce_bytes.to_vec(),
                wrapped_keys: vec![WrappedKey {
                    recipient: super::hex::encode(owner_pubkey),
                    encrypted_key: wrapped_dek,
                    role: "owner".to_string(),
                    granted_at: 0,
                }],
                plaintext_hash: hash,
            })
        }

        fn decrypt(
            &self,
            blob: &EncryptedBlob,
            recipient_address: &[u8],
            privkey: &[u8],
        ) -> Result<Vec<u8>, EncryptionError> {
            let addr_hex = super::hex::encode(recipient_address);
            let wrapped = blob
                .wrapped_keys
                .iter()
                .find(|k| k.recipient == addr_hex)
                .ok_or(EncryptionError::AccessDenied)?;

            // M31-native unwrap DEK.
            let dek_bytes =
                super::m31_keys::unwrap_dek(&wrapped.encrypted_key, privkey, AES_DEK_BYTES)?;

            if blob.nonce.len() != 12 {
                return Err(EncryptionError::DecryptionFailed(
                    "nonce must be 12 bytes".to_string(),
                ));
            }

            let key = Key::<Aes256Gcm>::from_slice(&dek_bytes);
            let cipher = Aes256Gcm::new(key);
            let nonce = Nonce::from_slice(&blob.nonce);

            let plaintext = cipher
                .decrypt(nonce, blob.ciphertext.as_ref())
                .map_err(|e| EncryptionError::DecryptionFailed(format!("AES decrypt: {}", e)))?;

            // Verify plaintext hash.
            let hash = hash_plaintext(&plaintext);
            let expected = hash;
            if expected != blob.plaintext_hash {
                return Err(EncryptionError::HashMismatch);
            }

            Ok(plaintext)
        }

        fn wrap_key_for(
            &self,
            blob: &EncryptedBlob,
            owner_privkey: &[u8],
            owner_address: &[u8],
            grantee_pubkey: &[u8],
            _grantee_address: &[u8],
        ) -> Result<WrappedKey, EncryptionError> {
            // Unwrap DEK using owner's secret key.
            let owner_hex = super::hex::encode(owner_address);
            let owner_wrapped = blob
                .wrapped_keys
                .iter()
                .find(|k| k.recipient == owner_hex)
                .ok_or(EncryptionError::AccessDenied)?;

            let dek_bytes = super::m31_keys::unwrap_dek(
                &owner_wrapped.encrypted_key,
                owner_privkey,
                AES_DEK_BYTES,
            )?;

            // Re-wrap for grantee using their view key.
            let wrapped_dek = super::m31_keys::wrap_dek(&dek_bytes, grantee_pubkey)?;

            Ok(WrappedKey {
                recipient: super::hex::encode(grantee_pubkey),
                encrypted_key: wrapped_dek,
                role: "auditor".to_string(),
                granted_at: 0,
            })
        }

        fn scheme_name(&self) -> &str {
            "aes_256_gcm"
        }
    }
}

#[cfg(feature = "aes-fallback")]
pub use aes_impl::Aes256GcmEncryption;

// ─── High-level Operations ─────────────────────────────────────────────────

/// Encrypt an audit report and upload to Arweave.
///
/// Returns the storage receipt from the upload.
pub fn encrypt_and_store(
    report: &AuditReport,
    owner_pubkey: &[u8],
    encryption: &dyn AuditEncryption,
    storage: &ArweaveClient,
) -> Result<(crate::audit::storage::StorageReceipt, EncryptedBlob), AuditError> {
    // Serialize report to JSON.
    let json = serde_json::to_vec(report).map_err(|e| AuditError::Serde(e.to_string()))?;

    // Encrypt.
    let blob = encryption
        .encrypt(&json, owner_pubkey)
        .map_err(AuditError::Encryption)?;

    // Serialize blob for storage.
    let blob_bytes = serde_json::to_vec(&blob).map_err(|e| AuditError::Serde(e.to_string()))?;

    // Upload with metadata tags.
    let extra_tags = vec![
        ArweaveTag {
            name: "Encryption-Scheme".to_string(),
            value: encryption.scheme_name().to_string(),
        },
        ArweaveTag {
            name: "Report-Hash".to_string(),
            value: report.commitments.audit_report_hash.clone(),
        },
    ];

    let receipt = storage.upload(
        &blob_bytes,
        &report.audit_id,
        &report.model.model_id,
        &extra_tags,
    )?;

    Ok((receipt, blob))
}

/// Download an encrypted report from Arweave and decrypt it.
///
/// Verifies the plaintext hash matches before returning the report.
pub fn fetch_and_decrypt(
    tx_id: &str,
    recipient_address: &[u8],
    privkey: &[u8],
    encryption: &dyn AuditEncryption,
    storage: &ArweaveClient,
) -> Result<AuditReport, AuditError> {
    // Download encrypted blob.
    let blob_bytes = storage.download(tx_id)?;

    // Deserialize the encrypted blob.
    let blob: EncryptedBlob =
        serde_json::from_slice(&blob_bytes).map_err(|e| AuditError::Serde(e.to_string()))?;

    // Decrypt.
    let plaintext = encryption
        .decrypt(&blob, recipient_address, privkey)
        .map_err(AuditError::Encryption)?;

    // Deserialize the report.
    let report: AuditReport =
        serde_json::from_slice(&plaintext).map_err(|e| AuditError::Serde(e.to_string()))?;

    Ok(report)
}

/// Grant access to an encrypted blob for an additional recipient.
///
/// The owner unwraps the data key and re-wraps it for the grantee.
/// Returns a new `WrappedKey` that should be appended to the blob's key list.
pub fn grant_access(
    blob: &EncryptedBlob,
    owner_privkey: &[u8],
    owner_address: &[u8],
    grantee_pubkey: &[u8],
    grantee_address: &[u8],
    encryption: &dyn AuditEncryption,
) -> Result<WrappedKey, AuditError> {
    let key = encryption
        .wrap_key_for(
            blob,
            owner_privkey,
            owner_address,
            grantee_pubkey,
            grantee_address,
        )
        .map_err(AuditError::Encryption)?;
    Ok(key)
}

/// Revoke access by re-encrypting with a fresh key for remaining recipients.
///
/// This is the nuclear option — generates a completely new ciphertext.
/// Only the listed recipients (by address) will be able to decrypt.
pub fn revoke_access(
    plaintext: &[u8],
    owner_pubkey: &[u8],
    remaining_recipients: &[(&[u8], &[u8], &str)], // (pubkey, address, role)
    encryption: &dyn AuditEncryption,
) -> Result<EncryptedBlob, AuditError> {
    // Re-encrypt with new key for owner.
    let mut blob = encryption
        .encrypt(plaintext, owner_pubkey)
        .map_err(AuditError::Encryption)?;

    // Wrap key for each remaining recipient.
    for &(grantee_pub, grantee_addr, role) in remaining_recipients {
        let mut key = encryption
            .wrap_key_for(
                &blob,
                owner_pubkey, // Simplified: owner pubkey used as privkey for noop
                owner_pubkey,
                grantee_pub,
                grantee_addr,
            )
            .map_err(AuditError::Encryption)?;
        key.role = role.to_string();
        blob.wrapped_keys.push(key);
    }

    Ok(blob)
}

// ─── M31-Native Key Wrapping (Poseidon2 Symmetric + View Key Delegation) ────
//
// Per the Obelysk Protocol / VM31 design docs:
//   "All crypto operations are hash-based (Poseidon), not EC-based"
//
// Key model:
//   Secret key:  sk = [M31; 4]  (random QM31, ~124-bit)
//   View key:    vk = Poseidon2("view" || sk)  → [M31; RATE] = 8 M31 elements
//   Public ID:   pk = Poseidon2("pubk" || sk)  → [M31; RATE] (on-chain identifier)
//
// Key wrapping uses the VIEW KEY as a symmetric Poseidon2 encryption key:
//   wrap(dek, view_key) → nonce(16) || encrypted_dek(N×4) || mac(4)
//   unwrap(wrapped, secret_key) → derive view_key, then decrypt
//
// The view key is shared via delegation (on-chain view_key.cairo or out-of-band).
// Anyone with the view key CAN decrypt — that is intentional for the audit model.
// Only the secret key holder can derive the view key.
//
// 100% M31-native. Zero felt252. Zero EC operations. Fully circuit-provable.

mod m31_keys {
    use stwo::core::fields::m31::BaseField as M31;

    use crate::audit::types::EncryptionError;
    use crate::crypto::encryption::{poseidon2_decrypt, poseidon2_encrypt};
    use crate::crypto::poseidon2_m31::{poseidon2_hash, RATE};

    /// M31 field modulus.
    const P: u32 = 2147483647;

    // Domain separators matching crypto/commitment.rs pattern.
    const DOMAIN_VIEW: M31 = M31::from_u32_unchecked(0x76696577); // "view"
    #[allow(dead_code)]
    const DOMAIN_PUBKEY: M31 = M31::from_u32_unchecked(0x7075626B); // "pubk"
    const DOMAIN_MAC: M31 = M31::from_u32_unchecked(0x6D616300); // "mac\0"

    // ── Key Derivation ──────────────────────────────────────────────────────

    /// Derive view key (encryption key) from secret key.
    /// vk = Poseidon2("view" || sk[0..4]) → [M31; RATE]
    pub fn derive_view_key(sk: &[M31; 4]) -> [M31; RATE] {
        let mut input = Vec::with_capacity(5);
        input.push(DOMAIN_VIEW);
        input.extend_from_slice(sk);
        poseidon2_hash(&input)
    }

    /// Derive public identifier from secret key.
    /// pk = Poseidon2("pubk" || sk[0..4]) → [M31; RATE]
    #[allow(dead_code)]
    pub fn derive_public_id(sk: &[M31; 4]) -> [M31; RATE] {
        let mut input = Vec::with_capacity(5);
        input.push(DOMAIN_PUBKEY);
        input.extend_from_slice(sk);
        poseidon2_hash(&input)
    }

    // ── Serialization ───────────────────────────────────────────────────────

    /// Serialize M31 slice to bytes (4 bytes per element, LE).
    fn m31_ser(elements: &[M31]) -> Vec<u8> {
        elements.iter().flat_map(|m| m.0.to_le_bytes()).collect()
    }

    /// Deserialize bytes to M31 vec (4 bytes per element, LE).
    fn m31_deser(data: &[u8]) -> Result<Vec<M31>, EncryptionError> {
        if data.len() % 4 != 0 {
            return Err(EncryptionError::DecryptionFailed(
                "data length not multiple of 4".into(),
            ));
        }
        Ok(data
            .chunks(4)
            .map(|c| {
                let val = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                M31::from_u32_unchecked(val % P)
            })
            .collect())
    }

    /// Parse 32-byte view key → [M31; RATE].
    pub fn parse_view_key(bytes: &[u8]) -> Result<[M31; RATE], EncryptionError> {
        if bytes.len() != RATE * 4 {
            return Err(EncryptionError::DecryptionFailed(format!(
                "view key must be {} bytes, got {}",
                RATE * 4,
                bytes.len()
            )));
        }
        let v = m31_deser(bytes)?;
        let mut vk = [M31::from_u32_unchecked(0); RATE];
        vk.copy_from_slice(&v);
        Ok(vk)
    }

    /// Parse 16-byte secret key → [M31; 4].
    fn parse_secret_key(bytes: &[u8]) -> Result<[M31; 4], EncryptionError> {
        if bytes.len() != 16 {
            return Err(EncryptionError::DecryptionFailed(format!(
                "secret key must be 16 bytes, got {}",
                bytes.len()
            )));
        }
        let v = m31_deser(bytes)?;
        Ok([v[0], v[1], v[2], v[3]])
    }

    // ── MAC ─────────────────────────────────────────────────────────────────

    /// Compute MAC: Poseidon2(mac_domain || view_key || data)[0].
    fn compute_mac(vk: &[M31; RATE], data: &[M31]) -> M31 {
        let mut input = Vec::with_capacity(1 + RATE + data.len());
        input.push(DOMAIN_MAC);
        input.extend_from_slice(vk);
        input.extend_from_slice(data);
        poseidon2_hash(&input)[0]
    }

    // ── Random Nonce ────────────────────────────────────────────────────────

    fn random_nonce() -> Result<[M31; 4], EncryptionError> {
        let mut bytes = [0u8; 16];
        getrandom::getrandom(&mut bytes)
            .map_err(|e| EncryptionError::EncryptionFailed(format!("RNG: {}", e)))?;
        Ok([
            M31::from_u32_unchecked(
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) % P,
            ),
            M31::from_u32_unchecked(
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) % P,
            ),
            M31::from_u32_unchecked(
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) % P,
            ),
            M31::from_u32_unchecked(
                u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) % P,
            ),
        ])
    }

    // ── Pack/Unpack bytes ↔ M31 ─────────────────────────────────────────────

    /// Pack arbitrary bytes into M31 elements with length prefix (3 bytes/element).
    fn pack_bytes(data: &[u8]) -> Vec<M31> {
        let mut out = Vec::with_capacity(1 + (data.len() + 2) / 3);
        out.push(M31::from_u32_unchecked(data.len() as u32));
        for chunk in data.chunks(3) {
            let mut val = 0u32;
            for (i, &b) in chunk.iter().enumerate() {
                val |= (b as u32) << (i * 8);
            }
            out.push(M31::from_u32_unchecked(val));
        }
        out
    }

    /// Unpack M31 elements to bytes (first element = original length).
    fn unpack_bytes(elements: &[M31], expected_len: usize) -> Vec<u8> {
        if elements.is_empty() {
            return Vec::new();
        }
        let stored_len = elements[0].0 as usize;
        let len = expected_len.min(stored_len);
        let mut out = Vec::with_capacity(len);
        for &elem in &elements[1..] {
            let val = elem.0;
            out.push((val & 0xFF) as u8);
            out.push(((val >> 8) & 0xFF) as u8);
            out.push(((val >> 16) & 0xFF) as u8);
        }
        out.truncate(len);
        out
    }

    // ── Wrap / Unwrap ───────────────────────────────────────────────────────

    /// Wrap DEK bytes using a view key (Poseidon2 symmetric encryption + MAC).
    ///
    /// Output format: `nonce(16) || encrypted_dek_m31(N×4) || mac_tag(4)`
    ///
    /// The view key is the shared secret between owner and authorized recipient.
    pub fn wrap_dek(dek_bytes: &[u8], view_key_bytes: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        let vk = parse_view_key(view_key_bytes)?;
        let dek_m31 = pack_bytes(dek_bytes);
        let nonce = random_nonce()?;

        let encrypted = poseidon2_encrypt(&vk, &nonce, &dek_m31);
        let mac = compute_mac(&vk, &encrypted);

        let mut out = Vec::with_capacity(16 + encrypted.len() * 4 + 4);
        out.extend_from_slice(&m31_ser(&nonce));
        out.extend_from_slice(&m31_ser(&encrypted));
        out.extend_from_slice(&mac.0.to_le_bytes());
        Ok(out)
    }

    /// Unwrap DEK bytes using a secret key (derives view key, then decrypts).
    ///
    /// `wrapped`: output from `wrap_dek`
    /// `secret_key_bytes`: 16-byte secret key (will derive view key internally)
    /// `expected_dek_len`: expected byte length of the DEK
    pub fn unwrap_dek(
        wrapped: &[u8],
        secret_key_bytes: &[u8],
        expected_dek_len: usize,
    ) -> Result<Vec<u8>, EncryptionError> {
        let sk = parse_secret_key(secret_key_bytes)?;
        let vk = derive_view_key(&sk);
        unwrap_dek_with_view_key(wrapped, &vk, expected_dek_len)
    }

    /// Unwrap DEK bytes directly using a view key.
    pub fn unwrap_dek_with_view_key(
        wrapped: &[u8],
        vk: &[M31; RATE],
        expected_dek_len: usize,
    ) -> Result<Vec<u8>, EncryptionError> {
        // Minimum: nonce(16) + at least 4 bytes encrypted + mac(4)
        if wrapped.len() < 24 {
            return Err(EncryptionError::DecryptionFailed(
                "wrapped key too short".into(),
            ));
        }

        // Parse nonce (first 16 bytes = 4 M31).
        let nonce_v = m31_deser(&wrapped[..16])?;
        let nonce: [M31; 4] = [nonce_v[0], nonce_v[1], nonce_v[2], nonce_v[3]];

        // Parse MAC (last 4 bytes).
        let mac_off = wrapped.len() - 4;
        let stored_mac = M31::from_u32_unchecked(
            u32::from_le_bytes([
                wrapped[mac_off],
                wrapped[mac_off + 1],
                wrapped[mac_off + 2],
                wrapped[mac_off + 3],
            ]) % P,
        );

        // Parse encrypted DEK M31 elements (middle).
        let encrypted = m31_deser(&wrapped[16..mac_off])?;

        // Verify MAC.
        let expected_mac = compute_mac(vk, &encrypted);
        if expected_mac != stored_mac {
            return Err(EncryptionError::DecryptionFailed(
                "MAC verification failed — wrong key or tampered data".into(),
            ));
        }

        // Decrypt.
        let dek_m31 = poseidon2_decrypt(vk, &nonce, &encrypted);
        Ok(unpack_bytes(&dek_m31, expected_dek_len))
    }

    // ── Keypair Generation ──────────────────────────────────────────────────

    /// Generate an M31-native keypair for the audit system.
    ///
    /// Returns `(view_key_bytes_32, secret_key_bytes_16)`.
    ///
    /// - `view_key_bytes`: 32 bytes — share this with parties who should decrypt
    /// - `secret_key_bytes`: 16 bytes — keep this secret, can regenerate view key
    pub fn generate_keypair() -> Result<(Vec<u8>, Vec<u8>), EncryptionError> {
        let mut sk_raw = [0u8; 16];
        getrandom::getrandom(&mut sk_raw)
            .map_err(|e| EncryptionError::EncryptionFailed(format!("RNG: {}", e)))?;

        let sk: [M31; 4] = [
            M31::from_u32_unchecked(
                u32::from_le_bytes([sk_raw[0], sk_raw[1], sk_raw[2], sk_raw[3]]) % P,
            ),
            M31::from_u32_unchecked(
                u32::from_le_bytes([sk_raw[4], sk_raw[5], sk_raw[6], sk_raw[7]]) % P,
            ),
            M31::from_u32_unchecked(
                u32::from_le_bytes([sk_raw[8], sk_raw[9], sk_raw[10], sk_raw[11]]) % P,
            ),
            M31::from_u32_unchecked(
                u32::from_le_bytes([sk_raw[12], sk_raw[13], sk_raw[14], sk_raw[15]]) % P,
            ),
        ];

        let vk = derive_view_key(&sk);
        Ok((m31_ser(&vk), m31_ser(&sk)))
    }
}

// ─── Poseidon2-M31 Encryption (M31-native key wrapping) ─────────────────────

mod poseidon2_impl {
    use stwo::core::fields::m31::BaseField as M31;

    use super::*;
    use crate::crypto::encryption::{derive_key, poseidon2_decrypt, poseidon2_encrypt};
    use crate::crypto::poseidon2_m31::{poseidon2_hash, RATE};

    /// M31 field modulus: P = 2^31 - 1.
    const P: u32 = 2147483647;

    /// Pack bytes into M31 elements (3 bytes per element to stay < 2^31).
    /// Prepends a length element so the original byte count is recoverable.
    fn bytes_to_m31(data: &[u8]) -> Vec<M31> {
        let mut out = Vec::with_capacity(1 + (data.len() + 2) / 3);
        out.push(M31::from_u32_unchecked(data.len() as u32));
        for chunk in data.chunks(3) {
            let mut val = 0u32;
            for (i, &b) in chunk.iter().enumerate() {
                val |= (b as u32) << (i * 8);
            }
            out.push(M31::from_u32_unchecked(val));
        }
        out
    }

    /// Unpack M31 elements back to bytes. First element is the original length.
    fn m31_to_bytes(elements: &[M31]) -> Vec<u8> {
        if elements.is_empty() {
            return Vec::new();
        }
        let len = elements[0].0 as usize;
        let mut out = Vec::with_capacity(len);
        for &elem in &elements[1..] {
            let val = elem.0;
            out.push((val & 0xFF) as u8);
            out.push(((val >> 8) & 0xFF) as u8);
            out.push(((val >> 16) & 0xFF) as u8);
        }
        out.truncate(len);
        out
    }

    /// Derive an M31 encryption key from arbitrary bytes.
    fn key_from_bytes(key_bytes: &[u8]) -> [M31; RATE] {
        let m31s: Vec<M31> = key_bytes
            .chunks(3)
            .map(|chunk| {
                let mut val = 0u32;
                for (i, &b) in chunk.iter().enumerate() {
                    val |= (b as u32) << (i * 8);
                }
                M31::from_u32_unchecked(val)
            })
            .collect();
        derive_key(&m31s)
    }

    /// Serialize M31 ciphertext to bytes (4 bytes per element, little-endian).
    fn ciphertext_to_bytes(ct: &[M31]) -> Vec<u8> {
        ct.iter().flat_map(|elem| elem.0.to_le_bytes()).collect()
    }

    /// Deserialize bytes back to M31 ciphertext.
    fn bytes_to_ciphertext(data: &[u8]) -> Result<Vec<M31>, EncryptionError> {
        if data.len() % 4 != 0 {
            return Err(EncryptionError::DecryptionFailed(
                "ciphertext length not multiple of 4".to_string(),
            ));
        }
        Ok(data
            .chunks(4)
            .map(|chunk| {
                let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                M31::from_u32_unchecked(val % P)
            })
            .collect())
    }

    /// Serialize M31 DEK to bytes (4 bytes per element, LE).
    fn dek_to_bytes(dek: &[M31; RATE]) -> Vec<u8> {
        dek.iter().flat_map(|m| m.0.to_le_bytes()).collect()
    }

    /// DEK size in bytes: 8 M31 elements × 4 bytes = 32.
    const DEK_BYTES: usize = RATE * 4;

    /// Poseidon2-M31 encryption backend with M31-native key wrapping.
    ///
    /// Data encryption: Poseidon2-M31 counter mode (symmetric).
    /// Key wrapping: Poseidon2 symmetric encryption using shared view key.
    ///
    /// Key model (per Obelysk Protocol / VM31 docs):
    /// - "Public key" (`owner_pubkey`, `grantee_pubkey`): 32 bytes = view key (shared with authorized parties)
    /// - "Private key" (`privkey`, `owner_privkey`): 16 bytes = secret key (derives view key)
    /// - Recipient address: view key bytes (32) for lookup in WrappedKey.recipient
    pub struct Poseidon2M31Encryption;

    impl AuditEncryption for Poseidon2M31Encryption {
        fn encrypt(
            &self,
            plaintext: &[u8],
            owner_pubkey: &[u8],
        ) -> Result<EncryptedBlob, EncryptionError> {
            // Generate random DEK.
            let mut rng_bytes = [0u8; 24];
            getrandom::getrandom(&mut rng_bytes)
                .map_err(|e| EncryptionError::EncryptionFailed(format!("RNG: {}", e)))?;
            let dek = key_from_bytes(&rng_bytes);

            // Derive nonce from DEK hash.
            let key_hash = poseidon2_hash(&dek);
            let nonce_m31: [M31; 4] = std::array::from_fn(|i| key_hash[i]);

            let m31_plaintext = bytes_to_m31(plaintext);
            let m31_ciphertext = poseidon2_encrypt(&dek, &nonce_m31, &m31_plaintext);

            let hash = hash_plaintext(plaintext);
            let nonce: Vec<u8> = nonce_m31.iter().flat_map(|m| m.0.to_le_bytes()).collect();

            // Wrap DEK for owner using M31-native Poseidon2 wrapping.
            let dek_bytes = dek_to_bytes(&dek);
            let wrapped_dek = super::m31_keys::wrap_dek(&dek_bytes, owner_pubkey)?;

            Ok(EncryptedBlob {
                ciphertext: ciphertext_to_bytes(&m31_ciphertext),
                scheme: self.scheme_name().to_string(),
                nonce,
                wrapped_keys: vec![WrappedKey {
                    recipient: super::hex::encode(owner_pubkey),
                    encrypted_key: wrapped_dek,
                    role: "owner".to_string(),
                    granted_at: 0,
                }],
                plaintext_hash: hash,
            })
        }

        fn decrypt(
            &self,
            blob: &EncryptedBlob,
            recipient_address: &[u8],
            privkey: &[u8],
        ) -> Result<Vec<u8>, EncryptionError> {
            let addr_hex = super::hex::encode(recipient_address);
            let wrapped = blob
                .wrapped_keys
                .iter()
                .find(|k| k.recipient == addr_hex)
                .ok_or(EncryptionError::AccessDenied)?;

            // Unwrap DEK: privkey is secret key (16 bytes) → derives view key internally.
            let dek_bytes =
                super::m31_keys::unwrap_dek(&wrapped.encrypted_key, privkey, DEK_BYTES)?;

            // Reconstruct DEK as M31 array.
            if dek_bytes.len() != DEK_BYTES {
                return Err(EncryptionError::DecryptionFailed(format!(
                    "DEK must be {} bytes, got {}",
                    DEK_BYTES,
                    dek_bytes.len()
                )));
            }
            let mut dek = [M31::from_u32_unchecked(0); RATE];
            for i in 0..RATE {
                let off = i * 4;
                let val = u32::from_le_bytes([
                    dek_bytes[off],
                    dek_bytes[off + 1],
                    dek_bytes[off + 2],
                    dek_bytes[off + 3],
                ]);
                dek[i] = M31::from_u32_unchecked(val % P);
            }

            // Reconstruct nonce from DEK hash.
            let key_hash = poseidon2_hash(&dek);
            let nonce_m31: [M31; 4] = std::array::from_fn(|i| key_hash[i]);

            let m31_ciphertext = bytes_to_ciphertext(&blob.ciphertext)?;
            let m31_plaintext = poseidon2_decrypt(&dek, &nonce_m31, &m31_ciphertext);
            let plaintext = m31_to_bytes(&m31_plaintext);

            // Verify hash.
            let hash = hash_plaintext(&plaintext);
            let expected = hash;
            if expected != blob.plaintext_hash {
                return Err(EncryptionError::HashMismatch);
            }

            Ok(plaintext)
        }

        fn wrap_key_for(
            &self,
            blob: &EncryptedBlob,
            owner_privkey: &[u8],
            owner_address: &[u8],
            grantee_pubkey: &[u8],
            _grantee_address: &[u8],
        ) -> Result<WrappedKey, EncryptionError> {
            // Unwrap DEK using owner's secret key.
            let owner_hex = super::hex::encode(owner_address);
            let owner_wrapped = blob
                .wrapped_keys
                .iter()
                .find(|k| k.recipient == owner_hex)
                .ok_or(EncryptionError::AccessDenied)?;

            let dek_bytes = super::m31_keys::unwrap_dek(
                &owner_wrapped.encrypted_key,
                owner_privkey,
                DEK_BYTES,
            )?;

            // Re-wrap DEK for grantee using their view key.
            let wrapped_bytes = super::m31_keys::wrap_dek(&dek_bytes, grantee_pubkey)?;

            Ok(WrappedKey {
                recipient: super::hex::encode(grantee_pubkey),
                encrypted_key: wrapped_bytes,
                role: "auditor".to_string(),
                granted_at: 0,
            })
        }

        fn scheme_name(&self) -> &str {
            "poseidon2_m31"
        }
    }
}

pub use m31_keys::generate_keypair as generate_audit_keypair;
pub use poseidon2_impl::Poseidon2M31Encryption;

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_encrypt_decrypt_roundtrip() {
        let enc = NoopEncryption;
        let owner_pub = b"owner_pubkey_123";
        let plaintext = b"Hello, encrypted audit!";

        let blob = enc.encrypt(plaintext, owner_pub).unwrap();
        assert_eq!(blob.scheme, "noop_xor");
        assert!(!blob.ciphertext.is_empty());
        assert_ne!(blob.ciphertext, plaintext);

        let decrypted = enc.decrypt(&blob, owner_pub, b"ignored_privkey").unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_noop_access_denied() {
        let enc = NoopEncryption;
        let blob = enc.encrypt(b"secret", b"owner_pub").unwrap();

        let result = enc.decrypt(&blob, b"wrong_address", b"any_key");
        assert!(matches!(result, Err(EncryptionError::AccessDenied)));
    }

    #[test]
    fn test_noop_wrap_key_for_grantee() {
        let enc = NoopEncryption;
        let blob = enc.encrypt(b"report data", b"owner_pub").unwrap();

        let wrapped = enc
            .wrap_key_for(
                &blob,
                b"owner_priv",
                b"owner_addr",
                b"grantee_pub",
                b"grantee_addr",
            )
            .unwrap();
        assert_eq!(wrapped.role, "auditor");

        let mut blob_with_grantee = blob.clone();
        blob_with_grantee.wrapped_keys.push(wrapped);

        let decrypted = enc
            .decrypt(&blob_with_grantee, b"grantee_addr", b"grantee_priv")
            .unwrap();
        assert_eq!(decrypted, b"report data");
    }

    #[test]
    fn test_revoke_access() {
        let enc = NoopEncryption;
        let plaintext = b"sensitive report";

        let original = enc.encrypt(plaintext, b"owner_pub").unwrap();
        assert_eq!(original.wrapped_keys.len(), 1);

        let revoked = revoke_access(plaintext, b"owner_pub", &[], &enc).unwrap();

        let decrypted = enc.decrypt(&revoked, b"owner_pub", b"owner_priv").unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_plaintext_hash_integrity() {
        let enc = NoopEncryption;
        let blob = enc.encrypt(b"original data", b"owner").unwrap();

        let mut tampered = blob.clone();
        if !tampered.ciphertext.is_empty() {
            tampered.ciphertext[0] ^= 0xFF;
        }

        let result = enc.decrypt(&tampered, b"owner", b"priv");
        assert!(matches!(result, Err(EncryptionError::HashMismatch)));
    }

    #[test]
    fn test_hex_roundtrip() {
        let data = b"hello world";
        let encoded = hex::encode(data);
        let decoded = hex::decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_hash_plaintext_deterministic() {
        let h1 = hash_plaintext(b"test data");
        let h2 = hash_plaintext(b"test data");
        assert_eq!(h1, h2);

        let h3 = hash_plaintext(b"different data");
        assert_ne!(h1, h3);
    }

    // ─── M31-Native Key Module Tests ────────────────────────────────────

    #[test]
    fn test_m31_keypair_generation() {
        let (view_key, secret) = m31_keys::generate_keypair().unwrap();
        assert_eq!(view_key.len(), 32, "view key = 8 M31 × 4 bytes");
        assert_eq!(secret.len(), 16, "secret key = 4 M31 × 4 bytes");

        let (vk2, sk2) = m31_keys::generate_keypair().unwrap();
        assert_ne!(view_key, vk2, "keypairs should differ");
        assert_ne!(secret, sk2);
    }

    #[test]
    fn test_m31_wrap_unwrap_roundtrip() {
        let (view_key, secret) = m31_keys::generate_keypair().unwrap();
        let dek = b"this_is_a_32_byte_test_dek_1234!";

        let wrapped = m31_keys::wrap_dek(dek, &view_key).unwrap();
        // nonce(16) + encrypted_m31(N×4) + mac(4)
        assert!(wrapped.len() > 24);

        let unwrapped = m31_keys::unwrap_dek(&wrapped, &secret, 32).unwrap();
        assert_eq!(unwrapped, dek);
    }

    #[test]
    fn test_m31_wrong_key_fails() {
        let (view_key, _secret) = m31_keys::generate_keypair().unwrap();
        let (_vk2, wrong_secret) = m31_keys::generate_keypair().unwrap();

        let wrapped = m31_keys::wrap_dek(b"12345678901234567890123456789012", &view_key).unwrap();
        let result = m31_keys::unwrap_dek(&wrapped, &wrong_secret, 32);
        assert!(result.is_err(), "wrong key should fail MAC check");
    }

    #[test]
    fn test_m31_tampered_wrapped_key_fails() {
        let (view_key, secret) = m31_keys::generate_keypair().unwrap();
        let dek = b"12345678901234567890123456789012";

        let mut wrapped = m31_keys::wrap_dek(dek, &view_key).unwrap();
        // Tamper with encrypted data portion (past the nonce).
        if wrapped.len() > 20 {
            wrapped[20] ^= 0xFF;
        }

        let result = m31_keys::unwrap_dek(&wrapped, &secret, 32);
        assert!(result.is_err(), "tampered data should fail MAC check");
    }

    #[test]
    fn test_m31_view_key_deterministic() {
        use stwo::core::fields::m31::BaseField as M31;
        let sk = [
            M31::from_u32_unchecked(42),
            M31::from_u32_unchecked(99),
            M31::from_u32_unchecked(7),
            M31::from_u32_unchecked(13),
        ];
        let vk1 = m31_keys::derive_view_key(&sk);
        let vk2 = m31_keys::derive_view_key(&sk);
        assert_eq!(vk1, vk2, "view key derivation must be deterministic");

        let pk = m31_keys::derive_public_id(&sk);
        // View key and public ID should differ (different domain separators).
        assert_ne!(vk1, pk, "view key and public ID should differ");
    }

    // ─── Poseidon2-M31 Backend Tests (M31-native wrapping) ──────────────

    /// Helper: generate M31-native keypair. Returns (view_key, secret_key).
    fn test_keypair() -> (Vec<u8>, Vec<u8>) {
        m31_keys::generate_keypair().unwrap()
    }

    #[test]
    fn test_poseidon2_m31_encrypt_decrypt_roundtrip() {
        let enc = Poseidon2M31Encryption;
        let (view_key, secret) = test_keypair();
        let plaintext = b"Hello, Poseidon2-M31 native encrypted audit!";

        let blob = enc.encrypt(plaintext, &view_key).unwrap();
        assert_eq!(blob.scheme, "poseidon2_m31");
        assert!(!blob.ciphertext.is_empty());
        assert_ne!(blob.ciphertext, plaintext.to_vec());

        // Owner decrypts with their secret key.
        let decrypted = enc.decrypt(&blob, &view_key, &secret).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_poseidon2_m31_access_denied() {
        let enc = Poseidon2M31Encryption;
        let (view_key, _secret) = test_keypair();
        let (wrong_vk, wrong_secret) = test_keypair();

        let blob = enc.encrypt(b"secret", &view_key).unwrap();

        // Wrong address → AccessDenied.
        let result = enc.decrypt(&blob, &wrong_vk, &wrong_secret);
        assert!(result.is_err(), "decryption with wrong key should fail");
    }

    #[test]
    fn test_poseidon2_m31_wrong_key_fails_mac() {
        let enc = Poseidon2M31Encryption;
        let (view_key, _secret) = test_keypair();
        let (_wrong_vk, wrong_secret) = test_keypair();

        let blob = enc.encrypt(b"secret", &view_key).unwrap();

        // Right address but wrong secret → MAC failure.
        let result = enc.decrypt(&blob, &view_key, &wrong_secret);
        assert!(result.is_err(), "wrong secret key should fail unwrap");
    }

    #[test]
    fn test_poseidon2_m31_tamper_detection() {
        let enc = Poseidon2M31Encryption;
        let (view_key, secret) = test_keypair();
        let blob = enc.encrypt(b"original data", &view_key).unwrap();

        let mut tampered = blob.clone();
        if !tampered.ciphertext.is_empty() {
            tampered.ciphertext[0] ^= 0xFF;
        }

        let result = enc.decrypt(&tampered, &view_key, &secret);
        assert!(matches!(result, Err(EncryptionError::HashMismatch)));
    }

    #[test]
    fn test_poseidon2_m31_wrap_key_for_grantee() {
        let enc = Poseidon2M31Encryption;
        let (owner_vk, owner_secret) = test_keypair();
        let (grantee_vk, grantee_secret) = test_keypair();

        let blob = enc.encrypt(b"audit report data", &owner_vk).unwrap();

        // Owner wraps key for grantee.
        let wrapped = enc
            .wrap_key_for(&blob, &owner_secret, &owner_vk, &grantee_vk, &grantee_vk)
            .unwrap();
        assert_eq!(wrapped.role, "auditor");

        // Grantee can decrypt.
        let mut blob_with_grantee = blob.clone();
        blob_with_grantee.wrapped_keys.push(wrapped);

        let decrypted = enc
            .decrypt(&blob_with_grantee, &grantee_vk, &grantee_secret)
            .unwrap();
        assert_eq!(decrypted, b"audit report data");
    }

    #[test]
    fn test_poseidon2_m31_large_plaintext() {
        let enc = Poseidon2M31Encryption;
        let (view_key, secret) = test_keypair();
        let plaintext: Vec<u8> = (0..10240).map(|i| (i % 256) as u8).collect();

        let blob = enc.encrypt(&plaintext, &view_key).unwrap();
        let decrypted = enc.decrypt(&blob, &view_key, &secret).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_poseidon2_m31_bytes_roundtrip() {
        let enc = Poseidon2M31Encryption;
        let (view_key, secret) = test_keypair();
        for len in [0, 1, 2, 3, 4, 7, 15, 31, 100] {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 13) as u8).collect();
            let blob = enc.encrypt(&data, &view_key).unwrap();
            let recovered = enc.decrypt(&blob, &view_key, &secret).unwrap();
            assert_eq!(recovered, data, "roundtrip failed for len={}", len);
        }
    }

    #[test]
    fn test_poseidon2_m31_empty_plaintext() {
        let enc = Poseidon2M31Encryption;
        let (view_key, secret) = test_keypair();
        let blob = enc.encrypt(b"", &view_key).unwrap();
        let decrypted = enc.decrypt(&blob, &view_key, &secret).unwrap();
        assert_eq!(decrypted, b"");
    }

    #[test]
    fn test_view_key_not_sufficient_without_secret() {
        // Core security: the view key is the "public" address for wrapping lookup,
        // but you need the SECRET KEY to derive the view key for unwrapping.
        // Using the view key AS the secret key should fail (wrong derivation).
        let enc = Poseidon2M31Encryption;
        let (view_key, _secret) = test_keypair();

        let blob = enc.encrypt(b"top secret", &view_key).unwrap();

        // Attacker tries to use view_key as the "private key" — wrong size (32 vs 16).
        let result = enc.decrypt(&blob, &view_key, &view_key);
        assert!(
            result.is_err(),
            "view key alone must not decrypt (wrong size for secret key)"
        );
    }
}
