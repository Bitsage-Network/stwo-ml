//! Transaction construction and proving for the VM31 privacy pool.
//!
//! Accumulates pending transactions, builds witnesses, and calls `prove_privacy_batch()`.

use stwo::core::fields::m31::BaseField as M31;

use crate::circuits::batch::{prove_privacy_batch, BatchError, PrivacyBatch, PrivacyBatchProof};
use crate::circuits::deposit::DepositWitness;
use crate::circuits::spend::{InputNoteWitness, OutputNoteWitness, SpendWitness};
use crate::circuits::withdraw::WithdrawWitness;
use std::collections::HashSet;

use crate::crypto::commitment::{
    validate_blinding, Note, NoteCommitment, PublicKey, SpendingKey, ViewingKey,
};
use crate::crypto::encryption::{derive_key, encrypt_note_memo};
use crate::crypto::merkle_m31::{verify_merkle_proof, Digest, MerklePath};
use crate::crypto::poseidon2_m31::RATE;

use super::serde_utils::EncryptedMemoJson;

// ─── Types ────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum TxBuilderError {
    #[error("no transactions to prove")]
    EmptyBatch,
    #[error("insufficient balance: need {need}, have {have}")]
    InsufficientBalance { need: u64, have: u64 },
    #[error("note not found at index {0}")]
    NoteNotFound(usize),
    #[error("need at least 2 input notes for transfer, have {0}")]
    NotEnoughInputs(usize),
    #[error("batch proving error: {0}")]
    BatchError(#[from] BatchError),
    #[error("cryptographic randomness unavailable: {0}")]
    Rng(String),
    #[error("zero amount not allowed")]
    ZeroAmount,
    #[error("nonce collision after {0} attempts (RNG failure?)")]
    NonceCollision(u32),
    #[error("merkle path does not verify against provided root")]
    MerklePathInvalid,
    #[error("amount {amount} exceeds max {max}")]
    AmountOverflow { amount: u64, max: u64 },
    #[error("withdrawal amount {requested} does not match note amount {note_amount}")]
    WithdrawAmountMismatch { requested: u64, note_amount: u64 },
    #[error("asset id {requested} does not match note asset {note_asset}")]
    AssetMismatch { requested: u32, note_asset: u32 },
}

/// A pending transaction to include in the batch.
#[derive(Clone, Debug)]
pub enum PendingTx {
    Deposit {
        amount: u64,
        asset_id: u32,
        recipient_pubkey: PublicKey,
        /// Viewing key for memo encryption. Required for the recipient to detect this note.
        recipient_viewing_key: ViewingKey,
    },
    Withdraw {
        amount: u64,
        asset_id: u32,
        /// The note to spend (from NoteStore).
        note: Note,
        spending_key: SpendingKey,
        merkle_path: MerklePath,
        merkle_root: Digest,
        /// Optional proof-bound bridge/app recipient binding.
        withdrawal_binding: Digest,
    },
    Transfer {
        amount: u64,
        asset_id: u32,
        recipient_pubkey: PublicKey,
        /// Viewing key for encrypting memo to recipient.
        recipient_viewing_key: ViewingKey,
        /// Sender's viewing key for encrypting change memo.
        sender_viewing_key: ViewingKey,
        /// Two input notes for the 2-in/2-out spend.
        input_notes: [(Note, SpendingKey, MerklePath); 2],
        merkle_root: Digest,
    },
}

/// Result of proving a batch.
pub struct ProvenTransaction {
    pub proof: PrivacyBatchProof,
    pub encrypted_memos: Vec<EncryptedMemoJson>,
    pub new_commitments: Vec<(NoteCommitment, Note)>,
    pub spent_commitments: Vec<String>,
}

/// Builder pattern for accumulating and proving privacy transactions.
#[derive(Debug)]
pub struct TxBuilder {
    pending: Vec<PendingTx>,
}

impl Default for TxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TxBuilder {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Add a deposit transaction.
    ///
    /// `recipient_viewing_key` is required so the recipient can detect and
    /// decrypt this note. For self-deposits, pass your own viewing key.
    pub fn deposit(
        &mut self,
        amount: u64,
        asset_id: u32,
        recipient_pubkey: PublicKey,
        recipient_viewing_key: ViewingKey,
    ) -> Result<&mut Self, TxBuilderError> {
        validate_amount(amount)?;
        self.pending.push(PendingTx::Deposit {
            amount,
            asset_id,
            recipient_pubkey,
            recipient_viewing_key,
        });
        Ok(self)
    }

    /// Add a withdraw transaction.
    ///
    /// `amount` and `asset_id` must match the note — a withdrawal spends the
    /// entire UTXO.  Mismatches are rejected upfront rather than failing deep
    /// in the circuit prover.
    pub fn withdraw(
        &mut self,
        amount: u64,
        asset_id: u32,
        note: Note,
        spending_key: SpendingKey,
        merkle_path: MerklePath,
        merkle_root: Digest,
    ) -> Result<&mut Self, TxBuilderError> {
        validate_amount(amount)?;
        validate_note_matches(&note, amount, asset_id)?;
        validate_merkle_path(&note.commitment(), &merkle_path, &merkle_root)?;
        self.pending.push(PendingTx::Withdraw {
            amount,
            asset_id,
            note,
            spending_key,
            merkle_path,
            merkle_root,
            withdrawal_binding: [M31::from_u32_unchecked(0); RATE],
        });
        Ok(self)
    }

    /// Add a withdraw transaction with an explicit proof-bound binding digest.
    pub fn withdraw_with_binding(
        &mut self,
        amount: u64,
        asset_id: u32,
        note: Note,
        spending_key: SpendingKey,
        merkle_path: MerklePath,
        merkle_root: Digest,
        withdrawal_binding: Digest,
    ) -> Result<&mut Self, TxBuilderError> {
        validate_amount(amount)?;
        validate_note_matches(&note, amount, asset_id)?;
        validate_merkle_path(&note.commitment(), &merkle_path, &merkle_root)?;
        self.pending.push(PendingTx::Withdraw {
            amount,
            asset_id,
            note,
            spending_key,
            merkle_path,
            merkle_root,
            withdrawal_binding,
        });
        Ok(self)
    }

    /// Add a transfer transaction (2-in/2-out spend).
    ///
    /// Both `recipient_viewing_key` and `sender_viewing_key` are required
    /// to encrypt memos: one for the recipient note, one for the change note.
    pub fn transfer(
        &mut self,
        amount: u64,
        asset_id: u32,
        recipient_pubkey: PublicKey,
        recipient_viewing_key: ViewingKey,
        sender_viewing_key: ViewingKey,
        input_notes: [(Note, SpendingKey, MerklePath); 2],
        merkle_root: Digest,
    ) -> Result<&mut Self, TxBuilderError> {
        validate_amount(amount)?;
        for (note, _, path) in &input_notes {
            validate_merkle_path(&note.commitment(), path, &merkle_root)?;
        }
        self.pending.push(PendingTx::Transfer {
            amount,
            asset_id,
            recipient_pubkey,
            recipient_viewing_key,
            sender_viewing_key,
            input_notes,
            merkle_root,
        });
        Ok(self)
    }

    /// Number of pending transactions.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Build witnesses and prove the batch.
    pub fn prove(self) -> Result<ProvenTransaction, TxBuilderError> {
        if self.pending.is_empty() {
            return Err(TxBuilderError::EmptyBatch);
        }

        let mut deposits = Vec::new();
        let mut withdrawals = Vec::new();
        let mut spends = Vec::new();
        let mut memos = Vec::new();
        let mut new_commitments = Vec::new();
        let mut spent_commitments = Vec::new();
        let mut used_nonces: HashSet<[u32; 4]> = HashSet::new();

        for tx in &self.pending {
            match tx {
                PendingTx::Deposit {
                    amount,
                    asset_id,
                    recipient_pubkey,
                    recipient_viewing_key,
                } => {
                    let (witness, note) =
                        build_deposit_witness(*amount, *asset_id, *recipient_pubkey)?;
                    new_commitments.push((note.commitment(), note.clone()));

                    // Encrypt memo using recipient's viewing key
                    let memo = build_encrypted_memo(
                        &note,
                        recipient_pubkey,
                        recipient_viewing_key,
                        &mut used_nonces,
                    )?;
                    memos.push(memo);

                    deposits.push(witness);
                }
                PendingTx::Withdraw {
                    note,
                    spending_key,
                    merkle_path,
                    merkle_root,
                    withdrawal_binding,
                    ..
                } => {
                    let commitment_hex = commitment_to_hex(&note.commitment());
                    spent_commitments.push(commitment_hex);
                    withdrawals.push(WithdrawWitness {
                        note: note.clone(),
                        spending_key: *spending_key,
                        merkle_path: merkle_path.clone(),
                        merkle_root: *merkle_root,
                        withdrawal_binding: *withdrawal_binding,
                    });
                }
                PendingTx::Transfer {
                    amount,
                    asset_id,
                    recipient_pubkey,
                    recipient_viewing_key,
                    sender_viewing_key,
                    input_notes,
                    merkle_root,
                } => {
                    let (witness, out_notes) = build_spend_witness(
                        *amount,
                        *asset_id,
                        *recipient_pubkey,
                        input_notes,
                        *merkle_root,
                    )?;

                    for out_note in &out_notes {
                        new_commitments.push((out_note.commitment(), out_note.clone()));
                    }

                    // Mark input notes as spent
                    for (note, _, _) in input_notes {
                        spent_commitments.push(commitment_to_hex(&note.commitment()));
                    }

                    // Encrypt memos for output notes
                    // First output → recipient (encrypted with recipient's viewing key)
                    let memo_recipient = build_encrypted_memo(
                        &out_notes[0],
                        recipient_pubkey,
                        recipient_viewing_key,
                        &mut used_nonces,
                    )?;
                    memos.push(memo_recipient);

                    // Change note → sender (encrypted with sender's viewing key)
                    let sender_pk = input_notes[0].0.owner_pubkey;
                    let memo_change = build_encrypted_memo(
                        &out_notes[1],
                        &sender_pk,
                        sender_viewing_key,
                        &mut used_nonces,
                    )?;
                    memos.push(memo_change);

                    spends.push(witness);
                }
            }
        }

        let batch = PrivacyBatch {
            deposits,
            withdrawals,
            spends,
        };

        let proof = prove_privacy_batch(&batch)?;

        Ok(ProvenTransaction {
            proof,
            encrypted_memos: memos,
            new_commitments,
            spent_commitments,
        })
    }
}

// ─── Witness builders ─────────────────────────────────────────────────────

/// Maximum amount that fits in two M31 limbs: (2^31 - 1) + (2^31 - 1) * 2^31
const MAX_NOTE_AMOUNT: u64 = ((1u64 << 31) - 1) + ((1u64 << 31) - 1) * (1u64 << 31);

fn validate_amount(amount: u64) -> Result<(), TxBuilderError> {
    if amount == 0 {
        return Err(TxBuilderError::ZeroAmount);
    }
    if amount > MAX_NOTE_AMOUNT {
        return Err(TxBuilderError::AmountOverflow {
            amount,
            max: MAX_NOTE_AMOUNT,
        });
    }
    Ok(())
}

/// Check that the withdrawal `amount` and `asset_id` match what the note
/// actually contains.  A withdrawal spends the entire UTXO — partial
/// withdrawals are not supported.
fn validate_note_matches(note: &Note, amount: u64, asset_id: u32) -> Result<(), TxBuilderError> {
    let note_amount = note.amount_lo.0 as u64 + (note.amount_hi.0 as u64) * (1u64 << 31);
    if amount != note_amount {
        return Err(TxBuilderError::WithdrawAmountMismatch {
            requested: amount,
            note_amount,
        });
    }
    if asset_id != note.asset_id.0 {
        return Err(TxBuilderError::AssetMismatch {
            requested: asset_id,
            note_asset: note.asset_id.0,
        });
    }
    Ok(())
}

/// Tree depth for the privacy pool Merkle tree.
const TREE_DEPTH: usize = 20;

/// Verify that the Merkle path is valid against the provided root.
///
/// This catches mismatched paths/roots early, before they reach the circuit
/// prover, and prevents submitting proofs against roots the pool doesn't know.
fn validate_merkle_path(
    commitment: &NoteCommitment,
    path: &MerklePath,
    root: &Digest,
) -> Result<(), TxBuilderError> {
    if !verify_merkle_proof(root, commitment, path, TREE_DEPTH) {
        return Err(TxBuilderError::MerklePathInvalid);
    }
    Ok(())
}

fn build_deposit_witness(
    amount: u64,
    asset_id: u32,
    recipient_pubkey: PublicKey,
) -> Result<(DepositWitness, Note), TxBuilderError> {
    validate_amount(amount)?;
    let blinding = generate_random_blinding()?;
    validate_blinding(&blinding).map_err(|e| TxBuilderError::Rng(e.to_string()))?;
    let amount_lo = M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32);
    let amount_hi = M31::from_u32_unchecked(((amount >> 31) & 0x7FFFFFFF) as u32);
    let asset_m31 = M31::from_u32_unchecked(asset_id);

    let note = Note::new(recipient_pubkey, asset_m31, amount_lo, amount_hi, blinding);
    let witness = DepositWitness {
        note: note.clone(),
        amount,
        asset_id: asset_m31,
    };
    Ok((witness, note))
}

fn build_spend_witness(
    amount: u64,
    asset_id: u32,
    recipient_pubkey: PublicKey,
    input_notes: &[(Note, SpendingKey, MerklePath); 2],
    merkle_root: Digest,
) -> Result<(SpendWitness, [Note; 2]), TxBuilderError> {
    validate_amount(amount)?;
    let input_total: u64 = input_notes
        .iter()
        .map(|(n, _, _)| n.amount_lo.0 as u64 + (n.amount_hi.0 as u64) * (1u64 << 31))
        .sum();

    if input_total < amount {
        return Err(TxBuilderError::InsufficientBalance {
            need: amount,
            have: input_total,
        });
    }

    let change = input_total - amount;
    let asset_m31 = M31::from_u32_unchecked(asset_id);

    // Output 1: to recipient
    let blinding1 = generate_random_blinding()?;
    validate_blinding(&blinding1).map_err(|e| TxBuilderError::Rng(e.to_string()))?;
    let out_note1 = Note::new(
        recipient_pubkey,
        asset_m31,
        M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32),
        M31::from_u32_unchecked(((amount >> 31) & 0x7FFFFFFF) as u32),
        blinding1,
    );

    // Output 2: change back to sender
    let sender_pk = input_notes[0].0.owner_pubkey;
    let blinding2 = generate_random_blinding()?;
    validate_blinding(&blinding2).map_err(|e| TxBuilderError::Rng(e.to_string()))?;
    let out_note2 = Note::new(
        sender_pk,
        asset_m31,
        M31::from_u32_unchecked((change & 0x7FFFFFFF) as u32),
        M31::from_u32_unchecked(((change >> 31) & 0x7FFFFFFF) as u32),
        blinding2,
    );

    let witness = SpendWitness {
        inputs: [
            InputNoteWitness {
                note: input_notes[0].0.clone(),
                spending_key: input_notes[0].1,
                merkle_path: input_notes[0].2.clone(),
            },
            InputNoteWitness {
                note: input_notes[1].0.clone(),
                spending_key: input_notes[1].1,
                merkle_path: input_notes[1].2.clone(),
            },
        ],
        outputs: [
            OutputNoteWitness {
                note: out_note1.clone(),
            },
            OutputNoteWitness {
                note: out_note2.clone(),
            },
        ],
        merkle_root,
    };

    Ok((witness, [out_note1, out_note2]))
}

fn build_encrypted_memo(
    note: &Note,
    recipient_pubkey: &PublicKey,
    viewing_key: &ViewingKey,
    used_nonces: &mut HashSet<[u32; 4]>,
) -> Result<EncryptedMemoJson, TxBuilderError> {
    // Encrypt using the recipient's viewing key. Only holders of the viewing
    // key (the recipient and authorized viewers) can decrypt. The public key
    // alone is NOT sufficient — this prevents anyone from reading memos just
    // by observing the chain.
    let enc_key = derive_key(viewing_key);
    let nonce = generate_unique_nonce(used_nonces)?;

    let encrypted = encrypt_note_memo(
        &enc_key,
        &nonce,
        note.asset_id,
        note.amount_lo,
        note.amount_hi,
        &note.blinding,
    );

    Ok(EncryptedMemoJson {
        encrypted_data: m31_array_to_hex(&encrypted),
        nonce: format!(
            "0x{:08x}{:08x}{:08x}{:08x}",
            nonce[0].0, nonce[1].0, nonce[2].0, nonce[3].0,
        ),
        recipient_pubkey: format!(
            "0x{:08x}{:08x}{:08x}{:08x}",
            recipient_pubkey[0].0,
            recipient_pubkey[1].0,
            recipient_pubkey[2].0,
            recipient_pubkey[3].0,
        ),
    })
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn generate_random_blinding() -> Result<[M31; 4], TxBuilderError> {
    super::random_m31_quad().map_err(|e| TxBuilderError::Rng(e))
}

fn generate_random_nonce() -> Result<[M31; 4], TxBuilderError> {
    generate_random_blinding()
}

/// Generate a random nonce guaranteed unique within this batch.
///
/// Retries up to 8 times on collision (astronomically unlikely with 124-bit
/// nonce space, but structurally enforced for defense-in-depth).
fn generate_unique_nonce(used: &mut HashSet<[u32; 4]>) -> Result<[M31; 4], TxBuilderError> {
    const MAX_RETRIES: u32 = 8;
    for attempt in 0..MAX_RETRIES {
        let nonce = generate_random_nonce()?;
        let key = [nonce[0].0, nonce[1].0, nonce[2].0, nonce[3].0];
        if used.insert(key) {
            return Ok(nonce);
        }
        // Collision — retry (should never happen with 124-bit nonce space)
        if attempt == MAX_RETRIES - 1 {
            return Err(TxBuilderError::NonceCollision(MAX_RETRIES));
        }
    }
    unreachable!()
}

fn commitment_to_hex(commitment: &NoteCommitment) -> String {
    m31_array_to_hex(commitment)
}

fn m31_array_to_hex(arr: &[M31]) -> String {
    let mut s = String::with_capacity(2 + arr.len() * 8);
    s.push_str("0x");
    for &elem in arr {
        s.push_str(&format!("{:08x}", elem.0));
    }
    s
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::commitment::{derive_pubkey, derive_viewing_key};
    use crate::crypto::merkle_m31::PoseidonMerkleTreeM31;

    fn test_sk() -> SpendingKey {
        [42, 99, 7, 13].map(M31::from_u32_unchecked)
    }

    #[test]
    fn test_builder_deposit() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let vk = derive_viewing_key(&sk);

        let mut builder = TxBuilder::new();
        builder.deposit(1000, 0, pk, vk).unwrap();
        assert_eq!(builder.len(), 1);

        let result = builder.prove().unwrap();
        assert_eq!(result.proof.public_inputs.deposits.len(), 1);
        assert_eq!(result.proof.public_inputs.deposits[0].amount, 1000);
        assert!(!result.new_commitments.is_empty());
        assert!(!result.encrypted_memos.is_empty());
    }

    #[test]
    fn test_builder_multiple_deposits() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let vk = derive_viewing_key(&sk);

        let mut builder = TxBuilder::new();
        builder.deposit(100, 0, pk, vk).unwrap();
        builder.deposit(200, 0, pk, vk).unwrap();
        builder.deposit(300, 0, pk, vk).unwrap();

        let result = builder.prove().unwrap();
        assert_eq!(result.proof.public_inputs.deposits.len(), 3);
        assert_eq!(result.new_commitments.len(), 3);
    }

    #[test]
    fn test_builder_withdraw() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let note = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            [10, 20, 30, 40].map(M31::from_u32_unchecked),
        );

        let commitment = note.commitment();
        let mut tree = PoseidonMerkleTreeM31::new(20);
        tree.append(commitment);
        let path = tree.prove(0).unwrap();
        let root = tree.root();

        let mut builder = TxBuilder::new();
        builder.withdraw(1000, 0, note, sk, path, root).unwrap();

        let result = builder.prove().unwrap();
        assert_eq!(result.proof.public_inputs.withdrawals.len(), 1);
        assert_eq!(result.spent_commitments.len(), 1);
    }

    #[test]
    fn test_builder_transfer() {
        let sk1 = [42, 99, 7, 13].map(M31::from_u32_unchecked);
        let sk2 = [50, 60, 70, 80].map(M31::from_u32_unchecked);
        let recipient_sk = [100, 200, 300, 400].map(M31::from_u32_unchecked);
        let pk1 = derive_pubkey(&sk1);
        let pk2 = derive_pubkey(&sk2);
        let recipient_pk = derive_pubkey(&recipient_sk);
        let recipient_vk = derive_viewing_key(&recipient_sk);
        let sender_vk = derive_viewing_key(&sk1);

        let note1 = Note::new(
            pk1,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            [1, 2, 3, 4].map(M31::from_u32_unchecked),
        );
        let note2 = Note::new(
            pk2,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(2000),
            M31::from_u32_unchecked(0),
            [5, 6, 7, 8].map(M31::from_u32_unchecked),
        );

        let mut tree = PoseidonMerkleTreeM31::new(20);
        tree.append(note1.commitment());
        tree.append(note2.commitment());
        let root = tree.root();
        let path1 = tree.prove(0).unwrap();
        let path2 = tree.prove(1).unwrap();

        let mut builder = TxBuilder::new();
        builder
            .transfer(
                1500,
                0,
                recipient_pk,
                recipient_vk,
                sender_vk,
                [(note1, sk1, path1), (note2, sk2, path2)],
                root,
            )
            .unwrap();

        let result = builder.prove().unwrap();
        assert_eq!(result.proof.public_inputs.spends.len(), 1);
        assert_eq!(result.new_commitments.len(), 2); // recipient + change
        assert_eq!(result.spent_commitments.len(), 2);
        assert_eq!(result.encrypted_memos.len(), 2); // one per output
    }

    #[test]
    fn test_builder_empty_fails() {
        let builder = TxBuilder::new();
        let result = builder.prove();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_deposit_zero_rejected() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let vk = derive_viewing_key(&sk);
        let mut builder = TxBuilder::new();
        let result = builder.deposit(0, 0, pk, vk);
        assert!(result.is_err());
        match result.unwrap_err() {
            TxBuilderError::ZeroAmount => {}
            e => panic!("expected ZeroAmount, got: {e}"),
        }
    }

    #[test]
    fn test_builder_deposit_overflow_rejected() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let vk = derive_viewing_key(&sk);
        let mut builder = TxBuilder::new();
        let result = builder.deposit(u64::MAX, 0, pk, vk);
        assert!(result.is_err());
        match result.unwrap_err() {
            TxBuilderError::AmountOverflow { .. } => {}
            e => panic!("expected AmountOverflow, got: {e}"),
        }
    }

    #[test]
    fn test_builder_withdraw_zero_rejected() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let note = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            [10, 20, 30, 40].map(M31::from_u32_unchecked),
        );
        let mut tree = PoseidonMerkleTreeM31::new(20);
        tree.append(note.commitment());
        let path = tree.prove(0).unwrap();
        let root = tree.root();

        let mut builder = TxBuilder::new();
        let result = builder.withdraw(0, 0, note, sk, path, root);
        assert!(result.is_err());
        match result.unwrap_err() {
            TxBuilderError::ZeroAmount => {}
            e => panic!("expected ZeroAmount, got: {e}"),
        }
    }

    // ── M9 regression: withdrawal amount must match note ─────────────────

    // ── H7 regression: amount limb split must use bitmask, not modulo ────

    #[test]
    fn test_h7_amount_limb_split_at_p_boundary() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let vk = derive_viewing_key(&sk);

        // amount = 2^31 = P + 1: lo should be 0, hi should be 1
        let amount = 1u64 << 31;
        let mut builder = TxBuilder::new();
        builder.deposit(amount, 0, pk, vk).unwrap();
        let result = builder.prove().unwrap();

        // Reconstruct from the note to verify round-trip
        let (_commitment, note) = &result.new_commitments[0];
        let reconstructed = note.amount_lo.0 as u64 + (note.amount_hi.0 as u64) * (1u64 << 31);
        assert_eq!(
            reconstructed, amount,
            "H7: amount {amount} must round-trip through limb split (got {reconstructed})"
        );
    }

    #[test]
    fn test_h7_amount_limb_split_just_below_p() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let vk = derive_viewing_key(&sk);

        // amount = P - 1 = 2^31 - 2: lo = P-1, hi = 0
        let amount = (1u64 << 31) - 2;
        let mut builder = TxBuilder::new();
        builder.deposit(amount, 0, pk, vk).unwrap();
        let result = builder.prove().unwrap();

        let (_commitment, note) = &result.new_commitments[0];
        let reconstructed = note.amount_lo.0 as u64 + (note.amount_hi.0 as u64) * (1u64 << 31);
        assert_eq!(
            reconstructed, amount,
            "H7: amount {amount} must round-trip (got {reconstructed})"
        );
    }

    #[test]
    fn test_h7_amount_limb_split_two_limbs() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let vk = derive_viewing_key(&sk);

        // amount = 3 * 2^31 + 42 → lo = 42, hi = 3
        let amount = 3u64 * (1u64 << 31) + 42;
        let mut builder = TxBuilder::new();
        builder.deposit(amount, 0, pk, vk).unwrap();
        let result = builder.prove().unwrap();

        let (_commitment, note) = &result.new_commitments[0];
        let reconstructed = note.amount_lo.0 as u64 + (note.amount_hi.0 as u64) * (1u64 << 31);
        assert_eq!(
            reconstructed, amount,
            "H7: two-limb amount {amount} must round-trip (got {reconstructed})"
        );
    }

    #[test]
    fn test_m9_withdraw_amount_mismatch_rejected() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let note = Note::new(
            pk,
            M31::from_u32_unchecked(0),    // STRK
            M31::from_u32_unchecked(1000), // amount_lo
            M31::from_u32_unchecked(0),    // amount_hi
            [10, 20, 30, 40].map(M31::from_u32_unchecked),
        );
        let mut tree = PoseidonMerkleTreeM31::new(20);
        tree.append(note.commitment());
        let path = tree.prove(0).unwrap();
        let root = tree.root();

        let mut builder = TxBuilder::new();
        // Request 2000 but note only contains 1000
        let result = builder.withdraw(2000, 0, note, sk, path, root);
        assert!(result.is_err());
        match result.unwrap_err() {
            TxBuilderError::WithdrawAmountMismatch {
                requested: 2000,
                note_amount: 1000,
            } => {}
            e => panic!("expected WithdrawAmountMismatch, got: {e}"),
        }
    }

    #[test]
    fn test_m9_withdraw_asset_mismatch_rejected() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let note = Note::new(
            pk,
            M31::from_u32_unchecked(0), // STRK (asset 0)
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            [10, 20, 30, 40].map(M31::from_u32_unchecked),
        );
        let mut tree = PoseidonMerkleTreeM31::new(20);
        tree.append(note.commitment());
        let path = tree.prove(0).unwrap();
        let root = tree.root();

        let mut builder = TxBuilder::new();
        // Claim asset 1 (ETH) but note is asset 0 (STRK)
        let result = builder.withdraw(1000, 1, note, sk, path, root);
        assert!(result.is_err());
        match result.unwrap_err() {
            TxBuilderError::AssetMismatch {
                requested: 1,
                note_asset: 0,
            } => {}
            e => panic!("expected AssetMismatch, got: {e}"),
        }
    }

    #[test]
    fn test_m9_withdraw_matching_amount_accepted() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let note = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(1000),
            M31::from_u32_unchecked(0),
            [10, 20, 30, 40].map(M31::from_u32_unchecked),
        );
        let mut tree = PoseidonMerkleTreeM31::new(20);
        tree.append(note.commitment());
        let path = tree.prove(0).unwrap();
        let root = tree.root();

        let mut builder = TxBuilder::new();
        // Correct amount and asset — should succeed
        assert!(builder.withdraw(1000, 0, note, sk, path, root).is_ok());
    }

    #[test]
    fn test_m9_withdraw_with_binding_amount_mismatch_rejected() {
        let sk = test_sk();
        let pk = derive_pubkey(&sk);
        let note = Note::new(
            pk,
            M31::from_u32_unchecked(0),
            M31::from_u32_unchecked(500),
            M31::from_u32_unchecked(0),
            [10, 20, 30, 40].map(M31::from_u32_unchecked),
        );
        let mut tree = PoseidonMerkleTreeM31::new(20);
        tree.append(note.commitment());
        let path = tree.prove(0).unwrap();
        let root = tree.root();
        let binding = [M31::from_u32_unchecked(0); RATE];

        let mut builder = TxBuilder::new();
        let result = builder.withdraw_with_binding(999, 0, note, sk, path, root, binding);
        assert!(result.is_err());
        match result.unwrap_err() {
            TxBuilderError::WithdrawAmountMismatch {
                requested: 999,
                note_amount: 500,
            } => {}
            e => panic!("expected WithdrawAmountMismatch, got: {e}"),
        }
    }
}
