//! VM31 wallet: spending key management, key derivation, and keystore persistence.
//!
//! Spending key → public key, viewing key, encryption key — all derived via Poseidon2-M31.
//! Keystore format: JSON file at `~/.vm31/wallet.json` with optional password encryption.

use std::path::{Path, PathBuf};

use stwo::core::fields::m31::BaseField as M31;

use crate::crypto::commitment::{derive_pubkey, derive_viewing_key, PublicKey, SpendingKey};
use crate::crypto::encryption::{derive_key, poseidon2_decrypt, poseidon2_encrypt};
use crate::crypto::poseidon2_m31::{poseidon2_hash, RATE};

/// Number of Poseidon2 hash iterations for password key derivation.
/// At ~1μs per hash on modern CPUs, 100k iterations ≈ 100ms — enough to
/// make brute-force dictionary attacks expensive while remaining interactive.
const KDF_ITERATIONS: u32 = 100_000;

// ─── Types ────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum WalletError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("crypto error: {0}")]
    Crypto(String),
    #[error("json error: {0}")]
    Json(String),
    #[error("invalid wallet file: {0}")]
    InvalidFormat(String),
    #[error("decryption failed (wrong password?)")]
    DecryptionFailed,
    #[error("hex parse error: {0}")]
    HexParse(String),
}

/// VM31 wallet containing a spending key and derived keys.
#[derive(Clone, Debug)]
pub struct Wallet {
    pub spending_key: SpendingKey,
    pub public_key: PublicKey,
    pub viewing_key: [M31; 4],
    pub encryption_key: [M31; RATE],
}

impl Wallet {
    /// Generate a new wallet with a random spending key.
    pub fn generate() -> Result<Self, WalletError> {
        let mut bytes = [0u8; 16];
        getrandom::getrandom(&mut bytes)
            .map_err(|e| WalletError::Crypto(format!("getrandom failed: {e}")))?;

        let spending_key: SpendingKey = [
            M31::from_u32_unchecked(
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) & 0x7FFFFFFF,
            ),
            M31::from_u32_unchecked(
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) & 0x7FFFFFFF,
            ),
            M31::from_u32_unchecked(
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) & 0x7FFFFFFF,
            ),
            M31::from_u32_unchecked(
                u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) & 0x7FFFFFFF,
            ),
        ];

        Ok(Self::from_spending_key(spending_key))
    }

    /// Construct wallet from an existing spending key.
    pub fn from_spending_key(spending_key: SpendingKey) -> Self {
        let public_key = derive_pubkey(&spending_key);
        let viewing_key = derive_viewing_key(&spending_key);
        let encryption_key = derive_key(&viewing_key);
        Self {
            spending_key,
            public_key,
            viewing_key,
            encryption_key,
        }
    }

    /// Parse a spending key from hex (e.g. "0x0000002a00000063000000070000000d").
    pub fn from_hex(hex: &str) -> Result<Self, WalletError> {
        let sk = parse_spending_key_hex(hex)?;
        Ok(Self::from_spending_key(sk))
    }

    /// Default wallet path: `~/.vm31/wallet.json`.
    pub fn default_path() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".vm31").join("wallet.json")
    }

    /// Hex representation of the public key.
    pub fn address(&self) -> String {
        format!(
            "0x{:08x}{:08x}{:08x}{:08x}",
            self.public_key[0].0, self.public_key[1].0, self.public_key[2].0, self.public_key[3].0,
        )
    }

    /// Save wallet to JSON file. If `password` is provided, the spending key
    /// is encrypted with Poseidon2-M31 counter-mode using a key derived from the password.
    pub fn save(&self, path: &Path, password: Option<&str>) -> Result<(), WalletError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let pk_hex = format!(
            "0x{:08x}{:08x}{:08x}{:08x}",
            self.public_key[0].0, self.public_key[1].0, self.public_key[2].0, self.public_key[3].0,
        );
        let vk_hex = format!(
            "0x{:08x}{:08x}{:08x}{:08x}",
            self.viewing_key[0].0,
            self.viewing_key[1].0,
            self.viewing_key[2].0,
            self.viewing_key[3].0,
        );

        let json = if let Some(pw) = password {
            let salt = generate_salt()?;
            let pw_key = password_to_key(pw, &salt);
            let nonce = generate_nonce()?;
            let sk_elems: Vec<M31> = self.spending_key.to_vec();
            let encrypted = poseidon2_encrypt(&pw_key, &nonce, &sk_elems);

            let enc_hex = m31_vec_to_hex(&encrypted);
            let nonce_hex = format!(
                "0x{:08x}{:08x}{:08x}{:08x}",
                nonce[0].0, nonce[1].0, nonce[2].0, nonce[3].0,
            );
            let salt_hex = format!(
                "0x{:08x}{:08x}{:08x}{:08x}",
                salt[0].0, salt[1].0, salt[2].0, salt[3].0,
            );

            format!(
                "{{\n  \"version\": 2,\n  \"public_key\": \"{pk_hex}\",\n  \"viewing_key\": \"{vk_hex}\",\n  \"encrypted_spending_key\": \"{enc_hex}\",\n  \"spending_key_nonce\": \"{nonce_hex}\",\n  \"kdf_salt\": \"{salt_hex}\",\n  \"kdf_iterations\": {KDF_ITERATIONS}\n}}"
            )
        } else {
            eprintln!(
                "WARNING: saving spending key in PLAINTEXT. Use --password for production wallets."
            );
            let sk_hex = format!(
                "0x{:08x}{:08x}{:08x}{:08x}",
                self.spending_key[0].0,
                self.spending_key[1].0,
                self.spending_key[2].0,
                self.spending_key[3].0,
            );
            format!(
                "{{\n  \"version\": 2,\n  \"public_key\": \"{pk_hex}\",\n  \"viewing_key\": \"{vk_hex}\",\n  \"spending_key_plaintext\": \"{sk_hex}\"\n}}"
            )
        };

        std::fs::write(path, json)?;

        // Set restrictive permissions (0600) on wallet file
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(path, perms).ok();
        }

        Ok(())
    }

    /// Load wallet from JSON file. If the file is encrypted, `password` is required.
    /// Supports both v1 (legacy single-pass KDF) and v2 (salted iterated KDF).
    pub fn load(path: &Path, password: Option<&str>) -> Result<Self, WalletError> {
        let contents = std::fs::read_to_string(path)?;
        let parsed: serde_json::Value =
            serde_json::from_str(&contents).map_err(|e| WalletError::Json(e.to_string()))?;

        let version = parsed["version"].as_u64().unwrap_or(0);
        if version != 1 && version != 2 {
            return Err(WalletError::InvalidFormat(format!(
                "unsupported wallet version {version}"
            )));
        }

        // Try plaintext spending key first
        if let Some(sk_hex) = parsed["spending_key_plaintext"].as_str() {
            let sk = parse_spending_key_hex(sk_hex)?;
            return Ok(Self::from_spending_key(sk));
        }

        // Try encrypted spending key
        let enc_hex = parsed["encrypted_spending_key"]
            .as_str()
            .ok_or_else(|| WalletError::InvalidFormat("missing spending key".into()))?;
        let nonce_hex = parsed["spending_key_nonce"]
            .as_str()
            .ok_or_else(|| WalletError::InvalidFormat("missing nonce".into()))?;

        let pw = password.ok_or(WalletError::DecryptionFailed)?;

        // Derive key based on wallet version
        let pw_key = if version >= 2 {
            // v2: salted iterated KDF
            let salt_hex = parsed["kdf_salt"]
                .as_str()
                .ok_or_else(|| WalletError::InvalidFormat("v2 wallet missing kdf_salt".into()))?;
            let salt = parse_4_m31_hex(salt_hex)?;
            password_to_key(pw, &salt)
        } else {
            // v1: legacy single-pass (backward compatibility)
            password_to_key_v1(pw)
        };

        let nonce = parse_4_m31_hex(nonce_hex)?;
        let encrypted = hex_to_m31_vec(enc_hex)?;
        let decrypted = poseidon2_decrypt(&pw_key, &nonce, &encrypted);

        if decrypted.len() < 4 {
            return Err(WalletError::DecryptionFailed);
        }

        let sk: SpendingKey = [decrypted[0], decrypted[1], decrypted[2], decrypted[3]];
        let wallet = Self::from_spending_key(sk);

        // Verify public key matches (integrity check)
        let pk_hex = parsed["public_key"].as_str().unwrap_or("");
        let expected_pk_hex = wallet.address();
        if !pk_hex.is_empty() && pk_hex != expected_pk_hex {
            return Err(WalletError::DecryptionFailed);
        }

        Ok(wallet)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Derive an encryption key from a password + salt using iterated Poseidon2-M31.
///
/// Concatenates password bytes + salt, hashes once, then iterates KDF_ITERATIONS
/// times. The salt prevents rainbow table attacks; iteration count makes
/// brute-force expensive (~100ms on a modern CPU).
fn password_to_key(pw: &str, salt: &[M31; 4]) -> [M31; RATE] {
    // Convert password to M31 elements
    let pw_m31: Vec<M31> = pw
        .bytes()
        .collect::<Vec<u8>>()
        .chunks(4)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            M31::from_u32_unchecked(u32::from_le_bytes(bytes) & 0x7FFFFFFF)
        })
        .collect();

    // Initial hash: H(password || salt)
    let mut input = pw_m31;
    input.extend_from_slice(salt);
    let mut state = poseidon2_hash(&input);

    // Iterate: state = H(state || salt || iteration_counter)
    for i in 0..KDF_ITERATIONS {
        let mut round_input: Vec<M31> = state.to_vec();
        round_input.extend_from_slice(salt);
        round_input.push(M31::from_u32_unchecked(i));
        state = poseidon2_hash(&round_input);
    }

    state
}

/// Legacy single-pass KDF for v1 wallet files (backward compatibility).
fn password_to_key_v1(pw: &str) -> [M31; RATE] {
    let pw_m31: Vec<M31> = pw
        .bytes()
        .collect::<Vec<u8>>()
        .chunks(4)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            M31::from_u32_unchecked(u32::from_le_bytes(bytes) & 0x7FFFFFFF)
        })
        .collect();
    derive_key(&pw_m31)
}

fn generate_salt() -> Result<[M31; 4], WalletError> {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes)
        .map_err(|e| WalletError::Crypto(format!("getrandom failed: {e}")))?;
    Ok([
        M31::from_u32_unchecked(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) & 0x7FFFFFFF,
        ),
        M31::from_u32_unchecked(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) & 0x7FFFFFFF,
        ),
        M31::from_u32_unchecked(
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) & 0x7FFFFFFF,
        ),
        M31::from_u32_unchecked(
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) & 0x7FFFFFFF,
        ),
    ])
}

fn generate_nonce() -> Result<[M31; 4], WalletError> {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes)
        .map_err(|e| WalletError::Crypto(format!("getrandom failed: {e}")))?;
    Ok([
        M31::from_u32_unchecked(
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) & 0x7FFFFFFF,
        ),
        M31::from_u32_unchecked(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) & 0x7FFFFFFF,
        ),
        M31::from_u32_unchecked(
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) & 0x7FFFFFFF,
        ),
        M31::from_u32_unchecked(
            u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) & 0x7FFFFFFF,
        ),
    ])
}

fn parse_spending_key_hex(hex: &str) -> Result<SpendingKey, WalletError> {
    let vals = parse_4_m31_hex(hex)?;
    Ok(vals)
}

fn parse_4_m31_hex(hex: &str) -> Result<[M31; 4], WalletError> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    if hex.len() != 32 {
        return Err(WalletError::HexParse(format!(
            "expected 32 hex chars (4 x 8), got {}",
            hex.len()
        )));
    }
    let mut result = [M31::from_u32_unchecked(0); 4];
    for i in 0..4 {
        let chunk = &hex[i * 8..(i + 1) * 8];
        let val = u32::from_str_radix(chunk, 16)
            .map_err(|e| WalletError::HexParse(format!("invalid hex '{chunk}': {e}")))?;
        result[i] = M31::from_u32_unchecked(val & 0x7FFFFFFF);
    }
    Ok(result)
}

fn m31_vec_to_hex(v: &[M31]) -> String {
    let mut s = String::with_capacity(2 + v.len() * 8);
    s.push_str("0x");
    for &elem in v {
        s.push_str(&format!("{:08x}", elem.0));
    }
    s
}

fn hex_to_m31_vec(hex: &str) -> Result<Vec<M31>, WalletError> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    if !hex.len().is_multiple_of(8) {
        return Err(WalletError::HexParse(format!(
            "hex length {} not multiple of 8",
            hex.len()
        )));
    }
    let mut result = Vec::with_capacity(hex.len() / 8);
    for i in 0..(hex.len() / 8) {
        let chunk = &hex[i * 8..(i + 1) * 8];
        let val = u32::from_str_radix(chunk, 16)
            .map_err(|e| WalletError::HexParse(format!("invalid hex '{chunk}': {e}")))?;
        result.push(M31::from_u32_unchecked(val));
    }
    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_wallet() {
        let w = Wallet::generate().unwrap();
        // Public key should be derived from spending key
        assert_eq!(w.public_key, derive_pubkey(&w.spending_key));
        assert_eq!(w.viewing_key, derive_viewing_key(&w.spending_key));
    }

    #[test]
    fn test_from_spending_key_deterministic() {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let w1 = Wallet::from_spending_key(sk);
        let w2 = Wallet::from_spending_key(sk);
        assert_eq!(w1.public_key, w2.public_key);
        assert_eq!(w1.viewing_key, w2.viewing_key);
        assert_eq!(w1.address(), w2.address());
    }

    #[test]
    fn test_from_hex_roundtrip() {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let hex = format!(
            "0x{:08x}{:08x}{:08x}{:08x}",
            sk[0].0, sk[1].0, sk[2].0, sk[3].0,
        );
        let w = Wallet::from_hex(&hex).unwrap();
        assert_eq!(w.spending_key, sk);
    }

    #[test]
    fn test_save_load_plaintext() {
        let w = Wallet::generate().unwrap();
        let tmp = std::env::temp_dir().join("vm31_test_wallet_plain.json");
        w.save(&tmp, None).unwrap();
        let loaded = Wallet::load(&tmp, None).unwrap();
        assert_eq!(loaded.spending_key, w.spending_key);
        assert_eq!(loaded.public_key, w.public_key);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_save_load_encrypted() {
        let w = Wallet::generate().unwrap();
        let tmp = std::env::temp_dir().join("vm31_test_wallet_enc.json");
        w.save(&tmp, Some("testpassword")).unwrap();
        let loaded = Wallet::load(&tmp, Some("testpassword")).unwrap();
        assert_eq!(loaded.spending_key, w.spending_key);
        assert_eq!(loaded.public_key, w.public_key);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_wrong_password_fails() {
        let w = Wallet::generate().unwrap();
        let tmp = std::env::temp_dir().join("vm31_test_wallet_wrongpw.json");
        w.save(&tmp, Some("correct")).unwrap();
        let result = Wallet::load(&tmp, Some("wrong"));
        assert!(result.is_err());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_address_format() {
        let sk = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let w = Wallet::from_spending_key(sk);
        let addr = w.address();
        assert!(addr.starts_with("0x"));
        assert_eq!(addr.len(), 34); // 0x + 4*8 hex chars
    }

    #[test]
    fn test_generate_uniqueness() {
        let w1 = Wallet::generate().unwrap();
        let w2 = Wallet::generate().unwrap();
        assert_ne!(w1.spending_key, w2.spending_key);
        assert_ne!(w1.public_key, w2.public_key);
    }
}
