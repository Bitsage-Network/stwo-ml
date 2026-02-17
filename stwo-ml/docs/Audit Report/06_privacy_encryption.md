# Stage 6: Privacy & Encryption

**Status**: Design Ready
**Readiness**: 30% — Obelysk Protocol has ElGamal + Pedersen, Marketplace has IndexedDB key store. Integration is new.
**Depends on**: Stage 4 (Audit Report Format)
**Blocks**: Stage 7 (Access Control)

---

## Purpose

Audit reports contain sensitive data — real user queries, model outputs, performance metrics. The operator must control who sees this data while still proving on-chain that the computation was correct.

The ZK proof already achieves the hard part: **verification without revelation.** The on-chain verifier checks the proof is valid without ever seeing the actual inputs or outputs. The only missing piece is encrypting the off-chain report.

---

## Privacy Tiers

| Tier | On-Chain | Off-Chain Report | Who Can Read | Use Case |
|------|----------|-----------------|-------------|----------|
| **0: Public** | Full metadata | Unencrypted, stored publicly | Anyone | Transparency reports, open-source models |
| **1: Private** | Hashes only | AES-256-GCM encrypted | Owner only | Proprietary AI operations |
| **2: Selective** | Hashes only | AES-256-GCM encrypted | Owner + granted addresses | Regulatory compliance, partner audits |

The operator chooses per-audit. Some audits might be public (quarterly transparency reports), others fully private (daily operations).

---

## Encryption Architecture

```
Audit Report (JSON)
        │
        ▼
┌──────────────────────┐
│  Generate random     │
│  AES-256-GCM key     │──> data_key (32 bytes)
│  (per-report)        │
└──────┬───────────────┘
       │
       ├──> Encrypt report ──> ciphertext + IV + auth_tag
       │
       ├──> Wrap data_key with owner's public key ──> wrapped_key_owner
       │
       └──> (Optional) Wrap data_key with grantee's public key ──> wrapped_key_grantee
                                                                       │
┌──────────────────────────────────────────────────────────────────────┘
│
▼
Storage (Bitsage Marketplace / IPFS)
┌─────────────────────────────────┐
│  encrypted_report.enc           │  ← AES-256-GCM ciphertext
│  key_store.json                 │  ← Wrapped keys per-recipient
│  metadata.json                  │  ← audit_id, storage_backend, etc.
└─────────────────────────────────┘
```

---

## Existing Primitives

### Obelysk Protocol — ElGamal over STARK Curve

From `Obelysk-Protocol/contracts/src/elgamal.cairo`:

- **Curve**: STARK curve `y^2 = x^3 + x + beta (mod p)`
- **Generator G**: `(0x1ef15c18..., 0x5668060a...)`
- **Generator H**: `(0x73bd2c94..., 0x1bd58ea5...)` — derived via hash-to-curve with domain `"OBELYSK_PEDERSEN_H_V1"`
- **Encryption**: `Enc(m, pk, r) = (r*G, m*G + r*pk)` — homomorphic, additively
- **Key wrapping**: Encrypt the AES data key's bytes as STARK curve points

### Bitsage Marketplace — IndexedDB Key Store

From `bitsage-marketplace/apps/web/src/lib/crypto/keyStore.ts`:

```typescript
interface StoredPrivacyKey {
  publicKey: ECPoint,
  encryptedPrivateKey: string,  // AES-GCM encrypted with KEK
  iv: string,
  salt: string,
  version: 1,
  createdAt: number
}
```

Key derivation: User signs message with Starknet wallet → signature → HKDF → KEK → encrypts ElGamal private key → stored in IndexedDB.

### Poseidon-M31 (VM31 Future)

From Obelysk Protocol VM31 spec:
- **Poseidon-M31**: 6,058 M31 muls per permutation (12.2x cheaper than Poseidon-252)
- **Commitment**: `C = Poseidon-M31(v, r)` — hash-based, not EC-based
- Future optimization for inference log Merkle trees

---

## Implementation

### Report Encryption

```rust
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::Aead;
use rand::RngCore;

/// Encrypted audit report.
pub struct EncryptedReport {
    /// AES-256-GCM ciphertext of the JSON report.
    pub ciphertext: Vec<u8>,
    /// 96-bit nonce/IV.
    pub nonce: [u8; 12],
    /// 128-bit authentication tag (appended to ciphertext by AES-GCM).
    pub auth_tag_included: bool,
    /// Wrapped data keys — one per authorized recipient.
    pub wrapped_keys: Vec<WrappedKey>,
    /// Poseidon hash of the plaintext report (matches on-chain audit_report_hash).
    pub report_hash: FieldElement,
    /// Privacy tier (1 = private, 2 = selective).
    pub privacy_tier: u8,
}

/// A data key wrapped (encrypted) for a specific Starknet address.
pub struct WrappedKey {
    /// Starknet address of the recipient.
    pub recipient: FieldElement,
    /// The AES data key, encrypted with the recipient's public key.
    /// For ElGamal: (r*G, data_key + r*pk) serialized.
    pub encrypted_key: Vec<u8>,
    /// Role: "owner", "auditor", "regulator"
    pub role: String,
    /// When access was granted.
    pub granted_at: u64,
}

/// Encrypt an audit report.
///
/// 1. Generate random AES-256-GCM key (data_key)
/// 2. Encrypt report JSON with data_key
/// 3. Wrap data_key with owner's public key
/// 4. Return EncryptedReport
pub fn encrypt_report(
    report: &AuditReport,
    owner_pubkey: &ECPoint,
) -> Result<EncryptedReport, EncryptionError> {
    // Serialize report to canonical JSON
    let plaintext = serde_json::to_vec(report)?;

    // Generate random AES-256-GCM key
    let mut data_key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut data_key);

    // Generate random nonce
    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);

    // Encrypt
    let cipher = Aes256Gcm::new_from_slice(&data_key)?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher.encrypt(nonce, plaintext.as_ref())?;

    // Wrap data_key with owner's ElGamal public key
    let wrapped_owner = elgamal_wrap_key(&data_key, owner_pubkey)?;

    // Compute report hash (same as on-chain)
    let report_hash = compute_report_hash(report);

    Ok(EncryptedReport {
        ciphertext,
        nonce: nonce_bytes,
        auth_tag_included: true,
        wrapped_keys: vec![WrappedKey {
            recipient: pubkey_to_address(owner_pubkey),
            encrypted_key: wrapped_owner,
            role: "owner".to_string(),
            granted_at: now_epoch(),
        }],
        report_hash,
        privacy_tier: 1,
    })
}

/// Decrypt an audit report using the recipient's private key.
pub fn decrypt_report(
    encrypted: &EncryptedReport,
    recipient_privkey: &FieldElement,
    recipient_address: &FieldElement,
) -> Result<AuditReport, EncryptionError> {
    // Find the wrapped key for this recipient
    let wrapped = encrypted.wrapped_keys.iter()
        .find(|wk| wk.recipient == *recipient_address)
        .ok_or(EncryptionError::AccessDenied)?;

    // Unwrap the data key with ElGamal decryption
    let data_key = elgamal_unwrap_key(&wrapped.encrypted_key, recipient_privkey)?;

    // Decrypt the report
    let cipher = Aes256Gcm::new_from_slice(&data_key)?;
    let nonce = Nonce::from_slice(&encrypted.nonce);
    let plaintext = cipher.decrypt(nonce, encrypted.ciphertext.as_ref())?;

    // Verify report hash
    let report: AuditReport = serde_json::from_slice(&plaintext)?;
    let computed_hash = compute_report_hash(&report);
    if computed_hash != encrypted.report_hash {
        return Err(EncryptionError::HashMismatch);
    }

    Ok(report)
}
```

### ElGamal Key Wrapping

```rust
/// Wrap a 32-byte AES key using ElGamal encryption over the STARK curve.
///
/// The key is split into 31-byte chunks (each fits in a felt252),
/// and each chunk is encrypted independently with ElGamal.
fn elgamal_wrap_key(
    data_key: &[u8; 32],
    recipient_pubkey: &ECPoint,
) -> Result<Vec<u8>, EncryptionError> {
    // Split key into two felt252s (first 31 bytes, last 1 byte padded)
    let mut chunk1 = [0u8; 32];
    chunk1[1..].copy_from_slice(&data_key[..31]);
    let m1 = FieldElement::from_bytes_be(&chunk1)?;

    let mut chunk2 = [0u8; 32];
    chunk2[31] = data_key[31];
    let m2 = FieldElement::from_bytes_be(&chunk2)?;

    // ElGamal encrypt each chunk: (r*G, m + r*pk)
    let c1 = elgamal_encrypt(m1, recipient_pubkey)?;
    let c2 = elgamal_encrypt(m2, recipient_pubkey)?;

    // Serialize: [c1.R.x, c1.R.y, c1.S.x, c1.S.y, c2.R.x, c2.R.y, c2.S.x, c2.S.y]
    let mut result = Vec::with_capacity(8 * 32);
    result.extend_from_slice(&c1.ephemeral.x.to_bytes_be());
    result.extend_from_slice(&c1.ephemeral.y.to_bytes_be());
    result.extend_from_slice(&c1.ciphertext.x.to_bytes_be());
    result.extend_from_slice(&c1.ciphertext.y.to_bytes_be());
    result.extend_from_slice(&c2.ephemeral.x.to_bytes_be());
    result.extend_from_slice(&c2.ephemeral.y.to_bytes_be());
    result.extend_from_slice(&c2.ciphertext.x.to_bytes_be());
    result.extend_from_slice(&c2.ciphertext.y.to_bytes_be());

    Ok(result)
}
```

### Grant Access

Adding a recipient requires:
1. Decrypt the data key with the owner's private key
2. Re-encrypt (wrap) the data key with the grantee's public key
3. Append the new `WrappedKey` to the encrypted report
4. Update the on-chain access control (Stage 7)

```rust
/// Grant access to an encrypted report.
///
/// The owner decrypts the data key and re-wraps it for the grantee.
pub fn grant_access(
    encrypted: &mut EncryptedReport,
    owner_privkey: &FieldElement,
    owner_address: &FieldElement,
    grantee_pubkey: &ECPoint,
    grantee_address: &FieldElement,
    role: &str,
) -> Result<(), EncryptionError> {
    // Decrypt data key with owner's key
    let owner_wrapped = encrypted.wrapped_keys.iter()
        .find(|wk| wk.recipient == *owner_address)
        .ok_or(EncryptionError::NotOwner)?;
    let data_key = elgamal_unwrap_key(&owner_wrapped.encrypted_key, owner_privkey)?;

    // Wrap for grantee
    let grantee_wrapped = elgamal_wrap_key(&data_key, grantee_pubkey)?;

    // Append
    encrypted.wrapped_keys.push(WrappedKey {
        recipient: *grantee_address,
        encrypted_key: grantee_wrapped,
        role: role.to_string(),
        granted_at: now_epoch(),
    });

    encrypted.privacy_tier = 2; // Selective disclosure

    Ok(())
}
```

### Revoke Access

Revocation requires re-encrypting with a new data key:

```rust
/// Revoke access from a recipient.
///
/// Re-encrypts the report with a new data key, re-wraps for all
/// remaining authorized parties. The revoked party's old wrapped key
/// becomes useless (decrypts to old data key, which no longer decrypts
/// the new ciphertext).
pub fn revoke_access(
    encrypted: &mut EncryptedReport,
    report: &AuditReport,  // Need plaintext to re-encrypt
    revokee_address: &FieldElement,
    remaining_pubkeys: &[(FieldElement, ECPoint, String)],  // (address, pubkey, role)
) -> Result<(), EncryptionError> {
    // Re-encrypt with new random key
    let mut new_data_key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut new_data_key);

    let mut new_nonce = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut new_nonce);

    let plaintext = serde_json::to_vec(report)?;
    let cipher = Aes256Gcm::new_from_slice(&new_data_key)?;
    let nonce = Nonce::from_slice(&new_nonce);
    let new_ciphertext = cipher.encrypt(nonce, plaintext.as_ref())?;

    // Re-wrap for all remaining parties
    let mut new_wrapped = Vec::new();
    for (addr, pubkey, role) in remaining_pubkeys {
        let wrapped = elgamal_wrap_key(&new_data_key, pubkey)?;
        new_wrapped.push(WrappedKey {
            recipient: *addr,
            encrypted_key: wrapped,
            role: role.clone(),
            granted_at: now_epoch(),
        });
    }

    encrypted.ciphertext = new_ciphertext;
    encrypted.nonce = new_nonce;
    encrypted.wrapped_keys = new_wrapped;

    Ok(())
}
```

---

## Storage Architecture: Arweave + Bitsage Marketplace

Encrypted reports are stored on **Arweave** (permanent, decentralized, ~$0.005/MB). The **Bitsage Marketplace** provides the UX layer — wallet-authenticated file browser, indexing, caching, and access control UI.

### Why Arweave (Not S3)

| Factor | S3 | Arweave |
|--------|------|---------|
| **Cost** | $0.023/GB/month (recurring) | ~$5/GB one-time (permanent) |
| **Permanence** | While you pay | Forever |
| **Decentralized** | No (AWS controls) | Yes (200+ miners) |
| **Censorship-resistant** | No | Yes |
| **Content-addressed** | No | Yes (tx_id = hash) |
| **Privacy** | Encrypted blobs are safe | Encrypted blobs are safe |

Since the report is AES-256-GCM encrypted before upload, it doesn't matter that Arweave is public — the ciphertext is meaningless without the wrapped key. Arweave nodes store data they cannot read.

### Storage Flow

```
Encrypt report
      │
      ├──> Upload to Arweave ──> arweave_tx_id (permanent)
      │        ~$0.005, one-time
      │
      ├──> Index in Marketplace DB ──> file_id (fast lookup)
      │        stores: {arweave_tx_id, audit_id, wallet, timestamp, size}
      │        caches: encrypted blob for fast subsequent access
      │
      └──> On-chain AuditRecord ──> includes arweave_tx_id
               ~$0.02, paymaster-sponsored
```

### Arweave Client

```rust
/// Upload encrypted data to Arweave permanent storage.
pub struct ArweaveStorage {
    gateway: String,       // https://arweave.net or bundlr node
    wallet_jwk: PathBuf,   // Arweave wallet for signing uploads
}

impl ArweaveStorage {
    /// Upload encrypted audit report to Arweave.
    ///
    /// Returns the transaction ID (content-addressed, permanent).
    /// Cost: ~$0.005 per MB (one-time, paid in AR tokens).
    pub async fn upload(
        &self,
        encrypted: &EncryptedReport,
        tags: &[(&str, &str)],  // Arweave tags for discoverability
    ) -> Result<String, StorageError> {
        let data = serialize_encrypted_report(encrypted)?;

        // Tags for indexing (visible, but content is encrypted)
        let mut all_tags = vec![
            ("Content-Type", "application/octet-stream"),
            ("App-Name", "Obelysk-Audit"),
            ("App-Version", "1.0"),
        ];
        all_tags.extend_from_slice(tags);

        // Upload via Arweave bundler (Irys/Bundlr for instant finality)
        let tx_id = self.bundler_upload(&data, &all_tags).await?;

        Ok(tx_id)
    }

    /// Fetch encrypted report from Arweave.
    pub async fn download(&self, tx_id: &str) -> Result<EncryptedReport, StorageError> {
        let url = format!("{}/{}", self.gateway, tx_id);
        let response = reqwest::get(&url).await?;
        let bytes = response.bytes().await?;
        deserialize_encrypted_report(&bytes)
    }
}
```

### Marketplace Caching Layer

```rust
/// Bitsage Marketplace — UX layer over Arweave.
///
/// Provides: wallet-scoped indexing, fast cached access, file browser UI,
/// access control management. Falls back to Arweave gateway on cache miss.
pub struct MarketplaceStorage {
    api_url: String,
    wallet_address: String,
    arweave: ArweaveStorage,  // Fallback for cache misses
}

impl MarketplaceStorage {
    /// Upload to Arweave + index in Marketplace.
    pub async fn upload_and_index(
        &self,
        encrypted: &EncryptedReport,
        audit_id: &str,
    ) -> Result<StorageReceipt, StorageError> {
        // 1. Upload to Arweave (permanent)
        let arweave_tx_id = self.arweave.upload(encrypted, &[
            ("Audit-ID", audit_id),
            ("Model-ID", &encrypted.report_hash.to_string()),
        ]).await?;

        // 2. Index in Marketplace (fast lookup)
        let response = self.client.post(&format!("{}/api/storage/index", self.api_url))
            .header("X-Wallet-Address", &self.wallet_address)
            .json(&serde_json::json!({
                "audit_id": audit_id,
                "arweave_tx_id": arweave_tx_id,
                "file_type": "audit_report",
                "size_bytes": encrypted.ciphertext.len(),
                "report_hash": encrypted.report_hash.to_string(),
            }))
            .send()
            .await?;

        Ok(StorageReceipt {
            arweave_tx_id: Some(arweave_tx_id),
            marketplace_file_id: Some(response.file_id),
            size_bytes: encrypted.ciphertext.len(),
            uploaded_at: now_epoch(),
            ..Default::default()
        })
    }

    /// Download — try Marketplace cache first, fall back to Arweave.
    pub async fn download(
        &self,
        audit_id: &str,
    ) -> Result<EncryptedReport, StorageError> {
        // Try Marketplace cache
        match self.marketplace_fetch(audit_id).await {
            Ok(report) => Ok(report),
            Err(_) => {
                // Cache miss — resolve arweave_tx_id from index, fetch from gateway
                let tx_id = self.resolve_arweave_tx(audit_id).await?;
                self.arweave.download(&tx_id).await
            }
        }
    }
}
```

### Required Marketplace Changes

1. **New API**: `POST /api/storage/index` — index an Arweave upload (wallet-scoped)
2. **New API**: `GET /api/storage/audits` — list user's audit reports
3. **New API**: `GET /api/storage/audits/{id}` — fetch cached blob (or redirect to Arweave)
4. **Database migration**: Add `audit_files` table linking `wallet_address` + `arweave_tx_id` + `audit_id`
5. **Frontend**: Audit report file browser in the storage page (browse, download, share, revoke)
6. **Cache layer**: Optional S3/local cache of frequently accessed Arweave blobs

---

## Obelysk Protocol Integration

### View Key Delegation

From `Obelysk-Protocol/contracts/src/stealth_payments.cairo`:

The view key pattern enables read-only access without ownership transfer:

```
Owner: (sk_spend, sk_view, pk_spend, pk_view)
Auditor: receives sk_view → can decrypt, cannot sign transactions
```

This maps directly to audit access:
- **Owner** has both spending key (can submit audits) and viewing key (can read reports)
- **Auditor** receives only the viewing key (can read reports, cannot submit new audits)
- **Revoking** a viewing key: re-encrypt with new key, re-derive viewing key

### Privacy Pools Compliance

From `Obelysk-Protocol/contracts/src/privacy_pools.cairo`:

For regulated environments, the audit submitter can prove membership in an approved ASP (Association Set Provider):

```
"I am in the set of approved medical AI operators"
→ Inclusion proof in Merkle tree
→ Submitted alongside audit
→ On-chain: audit + compliance proof in one TX
```

This enables privacy WITH compliance — the regulator knows the audit came from an approved operator without knowing which specific operator.

---

## Encryption Costs

| Operation | Time | Notes |
|-----------|------|-------|
| AES-256-GCM encrypt (1MB report) | ~0.5ms | Hardware-accelerated on modern CPUs |
| ElGamal key wrapping | ~2ms | Two EC scalar multiplications |
| ElGamal key unwrapping | ~1ms | One EC scalar multiplication |
| Poseidon report hash | ~5ms | Depends on report size |
| **Total encryption overhead** | **~8ms** | Negligible vs proving time |

---

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `src/audit/encryption.rs` | **New** | ~400 (encrypt, decrypt, wrap, unwrap, grant, revoke) |
| `src/audit/storage.rs` | **Modify** | ~100 (add marketplace backend) |
| `Cargo.toml` | **Modify** | +2 (aes-gcm, rand dependencies) |

---

## Verification Criteria

- [ ] Encrypt → decrypt round-trip preserves report exactly
- [ ] Report hash before encryption matches after decryption
- [ ] Owner can always decrypt their own reports
- [ ] Granted party can decrypt with their wrapped key
- [ ] Revoked party's old key fails to decrypt re-encrypted report
- [ ] Encryption adds < 10ms overhead
- [ ] Marketplace upload/download works with wallet auth
- [ ] Key store integrates with existing IndexedDB ElGamal keys
