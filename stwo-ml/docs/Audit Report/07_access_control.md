# Stage 7: Access Control

**Status**: New Contract
**Readiness**: 20% — View keys exist in Obelysk Protocol, ACL contract is new
**Depends on**: Stage 6 (Privacy/Encryption)

---

## Purpose

On-chain access control for encrypted audit reports. The owner of an audit can grant and revoke read access to specific Starknet addresses. The access list is stored on-chain, making it transparent and auditable — everyone can see *who* has access, but only key holders can read *what* was audited.

---

## Design

### On-Chain ACL

```cairo
/// Access control for a single audit's encrypted report.
#[derive(Drop, Serde, starknet::Store)]
struct AuditAccess {
    /// Starknet address of the authorized party.
    address: ContractAddress,
    /// Role: 0 = owner, 1 = auditor, 2 = regulator, 3 = partner
    role: u8,
    /// When access was granted (block number).
    granted_at_block: u64,
    /// Whether access is currently active.
    is_active: bool,
}
```

### Contract Interface

```cairo
#[starknet::interface]
trait IAuditAccessControl<TContractState> {
    // ─── Grant/Revoke ────────────────────────────────────────
    /// Grant read access to an audit report.
    ///
    /// Only the audit submitter (owner) can grant access.
    /// The wrapped_key is the AES data key encrypted with the grantee's
    /// public key — stored on-chain for the grantee to retrieve.
    fn grant_audit_access(
        ref self: TContractState,
        audit_id: felt252,
        grantee: ContractAddress,
        wrapped_key: felt252,     // ElGamal-encrypted AES key (compact)
        role: u8,
    );

    /// Revoke read access from a party.
    ///
    /// Only the owner can revoke. After revocation:
    /// 1. On-chain: grantee marked inactive, wrapped_key zeroed
    /// 2. Off-chain: owner re-encrypts report with new data key,
    ///    re-wraps for remaining parties, re-uploads
    fn revoke_audit_access(
        ref self: TContractState,
        audit_id: felt252,
        revokee: ContractAddress,
    );

    /// Batch grant access to multiple parties.
    fn grant_audit_access_batch(
        ref self: TContractState,
        audit_id: felt252,
        grantees: Span<ContractAddress>,
        wrapped_keys: Span<felt252>,
        roles: Span<u8>,
    );

    // ─── Query ───────────────────────────────────────────────
    /// Get the access list for an audit.
    fn get_audit_access_list(
        self: @TContractState,
        audit_id: felt252,
    ) -> Span<AuditAccess>;

    /// Check if an address has access to an audit.
    fn has_audit_access(
        self: @TContractState,
        audit_id: felt252,
        address: ContractAddress,
    ) -> bool;

    /// Get the wrapped key for a specific grantee.
    ///
    /// The grantee calls this to retrieve their encrypted data key,
    /// then decrypts it with their Starknet private key.
    fn get_wrapped_key(
        self: @TContractState,
        audit_id: felt252,
        grantee: ContractAddress,
    ) -> felt252;

    /// Get the owner of an audit.
    fn get_audit_owner(
        self: @TContractState,
        audit_id: felt252,
    ) -> ContractAddress;

    /// Get the number of active access grants for an audit.
    fn get_access_count(
        self: @TContractState,
        audit_id: felt252,
    ) -> u32;
}
```

### Storage Layout

```cairo
#[storage]
struct Storage {
    // ─── Access Control ──────────────────────────────────
    /// (audit_id, grantee_address) -> AuditAccess
    audit_access: LegacyMap<(felt252, ContractAddress), AuditAccess>,
    /// (audit_id, grantee_address) -> wrapped_key (ElGamal encrypted)
    audit_wrapped_keys: LegacyMap<(felt252, ContractAddress), felt252>,
    /// (audit_id, index) -> grantee_address (for enumeration)
    audit_access_list: LegacyMap<(felt252, u32), ContractAddress>,
    /// audit_id -> number of grantees (including owner)
    audit_access_count: LegacyMap<felt252, u32>,
    /// audit_id -> owner address
    audit_owner: LegacyMap<felt252, ContractAddress>,
}
```

### Events

```cairo
#[derive(Drop, starknet::Event)]
struct AccessGranted {
    #[key]
    audit_id: felt252,
    #[key]
    grantee: ContractAddress,
    role: u8,
    granted_by: ContractAddress,
}

#[derive(Drop, starknet::Event)]
struct AccessRevoked {
    #[key]
    audit_id: felt252,
    #[key]
    revokee: ContractAddress,
    revoked_by: ContractAddress,
}
```

---

## Implementation: `grant_audit_access`

```cairo
fn grant_audit_access(
    ref self: ContractState,
    audit_id: felt252,
    grantee: ContractAddress,
    wrapped_key: felt252,
    role: u8,
) {
    // 1. Verify caller is the audit owner
    let caller = get_caller_address();
    let owner = self.audit_owner.read(audit_id);
    assert(caller == owner, 'Only owner can grant access');

    // 2. Check grantee doesn't already have active access
    let existing = self.audit_access.read((audit_id, grantee));
    assert(!existing.is_active, 'Access already granted');

    // 3. Store access record
    let access = AuditAccess {
        address: grantee,
        role,
        granted_at_block: starknet::get_block_number(),
        is_active: true,
    };
    self.audit_access.write((audit_id, grantee), access);

    // 4. Store wrapped key
    self.audit_wrapped_keys.write((audit_id, grantee), wrapped_key);

    // 5. Add to enumeration list
    let count = self.audit_access_count.read(audit_id);
    self.audit_access_list.write((audit_id, count), grantee);
    self.audit_access_count.write(audit_id, count + 1);

    // 6. Emit event
    self.emit(AccessGranted {
        audit_id,
        grantee,
        role,
        granted_by: caller,
    });
}
```

---

## Decryption Flow (Client-Side)

When a grantee wants to read an audit report:

```
1. Get wrapped key from contract:
   wrapped_key = contract.get_wrapped_key(audit_id, my_address)

2. Decrypt wrapped key with Starknet private key:
   data_key = elgamal_decrypt(wrapped_key, my_private_key)

3. Fetch encrypted report from storage:
   encrypted = marketplace.download_report(audit_id)

4. Decrypt report with data key:
   report = aes_gcm_decrypt(encrypted.ciphertext, data_key, encrypted.nonce)

5. Verify report hash matches on-chain:
   assert(poseidon(report) == contract.get_audit(audit_id).audit_report_hash)
```

### TypeScript Client (Marketplace Web App)

```typescript
import { Account, RpcProvider, ec } from 'starknet';
import { loadKeyPair } from '@/lib/crypto/keyStore';

async function readAuditReport(auditId: string): Promise<AuditReport> {
  const provider = new RpcProvider({ nodeUrl: SEPOLIA_RPC });
  const walletAddress = localStorage.getItem('wallet_address');

  // 1. Check access
  const hasAccess = await provider.callContract({
    contractAddress: AUDIT_CONTRACT,
    entrypoint: 'has_audit_access',
    calldata: [auditId, walletAddress],
  });
  if (!hasAccess[0]) throw new Error('No access');

  // 2. Get wrapped key from contract
  const wrappedKey = await provider.callContract({
    contractAddress: AUDIT_CONTRACT,
    entrypoint: 'get_wrapped_key',
    calldata: [auditId, walletAddress],
  });

  // 3. Decrypt wrapped key with ElGamal private key from IndexedDB
  const kek = await deriveKEK(walletAddress); // From wallet signature
  const keyPair = await loadKeyPair(walletAddress, kek);
  const dataKey = elgamalDecrypt(wrappedKey[0], keyPair.privateKey);

  // 4. Fetch encrypted report from marketplace storage
  const response = await fetch(`/api/storage/${auditId}`, {
    headers: { 'X-Wallet-Address': walletAddress },
  });
  const encryptedData = await response.arrayBuffer();

  // 5. Decrypt with AES-GCM
  const cryptoKey = await crypto.subtle.importKey(
    'raw', dataKey, 'AES-GCM', false, ['decrypt']
  );
  const decrypted = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: extractIV(encryptedData) },
    cryptoKey,
    extractCiphertext(encryptedData)
  );

  // 6. Parse and verify hash
  const report = JSON.parse(new TextDecoder().decode(decrypted));
  // Verify poseidon(report) == on-chain audit_report_hash
  return report;
}
```

---

## View Key Delegation (Obelysk Protocol Pattern)

For long-term auditor relationships, the owner can delegate a **view key** instead of granting per-audit access:

```
Owner meta-address: (pk_spend, pk_view)
Auditor receives: sk_view (derived from owner's view key tree)

With sk_view, the auditor can:
- Scan for new audits (check access)
- Decrypt any audit the owner has published
- Cannot submit new audits (no sk_spend)

This is one grant for unlimited future audits.
```

### View Key Storage

```cairo
/// Long-term view key delegation.
///
/// Instead of per-audit grants, the owner shares a view key
/// that works for all future audits.
struct ViewKeyDelegation {
    owner: ContractAddress,
    delegate: ContractAddress,
    /// Encrypted view key (ElGamal with delegate's public key).
    encrypted_view_key: felt252,
    /// Valid from (block number).
    valid_from: u64,
    /// Valid until (block number, 0 = forever).
    valid_until: u64,
    /// Whether currently active.
    is_active: bool,
}

fn delegate_view_key(
    ref self: ContractState,
    delegate: ContractAddress,
    encrypted_view_key: felt252,
    valid_until: u64,
);

fn revoke_view_key(
    ref self: ContractState,
    delegate: ContractAddress,
);
```

---

## Privacy Pools Compliance Integration

For regulated environments, combine audit access with ASP membership:

```
Regulator grants:
  "Any ASP-approved auditor can access Model 0x2's audits"

On-chain:
  fn grant_asp_access(
      audit_id: felt252,
      asp_id: felt252,        // Association set containing approved auditors
      wrapped_key: felt252,   // Encrypted with ASP's public key
  );

  fn claim_asp_access(
      audit_id: felt252,
      asp_id: felt252,
      membership_proof: Span<felt252>,  // Merkle proof of ASP membership
  ) -> felt252;  // Returns re-wrapped key for the claimer
```

This enables: "Any FDA-approved auditor can read this medical AI audit" without knowing which specific auditor claims access.

---

## End-to-End Flow

```
1. AUDIT COMPLETED
   ├── Audit report generated (JSON)
   ├── Report encrypted (AES-256-GCM)
   ├── Data key wrapped for owner (ElGamal)
   ├── Encrypted report uploaded to Bitsage Marketplace
   └── submit_audit on-chain (report_hash, merkle_root, proof)

2. OWNER READS REPORT
   ├── get_wrapped_key(audit_id, owner_address) → wrapped key
   ├── ElGamal decrypt → data key
   ├── Fetch encrypted report from marketplace
   ├── AES-GCM decrypt → report JSON
   └── Verify poseidon(report) == on-chain hash

3. OWNER GRANTS ACCESS TO AUDITOR
   ├── Owner decrypts data key (as above)
   ├── Owner wraps data key with auditor's public key
   ├── grant_audit_access(audit_id, auditor, wrapped_key, role=1)
   └── Auditor can now follow step 2 with their own key

4. OWNER REVOKES ACCESS
   ├── Owner re-encrypts report with new data key
   ├── Owner re-wraps for remaining parties
   ├── Owner re-uploads encrypted report to marketplace
   ├── revoke_audit_access(audit_id, auditor)
   └── Auditor's old wrapped key is now useless
```

---

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `libs/elo-cairo-verifier/src/access_control.cairo` | **New** | ~250 (ACL contract) |
| `libs/elo-cairo-verifier/src/view_key.cairo` | **New** | ~100 (view key delegation) |
| `libs/elo-cairo-verifier/src/verifier.cairo` | **Modify** | ~30 (integrate ACL module) |
| `libs/elo-cairo-verifier/tests/test_access_control.cairo` | **New** | ~200 (unit tests) |

---

## Verification Criteria

- [ ] Only audit owner can grant access
- [ ] Only audit owner can revoke access
- [ ] Grantee can retrieve wrapped key from contract
- [ ] Grantee can decrypt report with retrieved wrapped key
- [ ] Revoked party's wrapped key no longer works after re-encryption
- [ ] View key delegation works across multiple audits
- [ ] ASP membership proof grants access to approved auditors
- [ ] Access list is enumerable on-chain
- [ ] Events emitted for all grant/revoke operations
- [ ] Backward compatible with existing verifier contract
