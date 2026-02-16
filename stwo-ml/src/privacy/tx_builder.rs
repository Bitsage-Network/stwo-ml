//! Transaction construction and proving for the VM31 privacy pool.
//!
//! Accumulates pending transactions, builds witnesses, and calls `prove_privacy_batch()`.

use stwo::core::fields::m31::BaseField as M31;

use crate::circuits::batch::{
    PrivacyBatch, PrivacyBatchProof, prove_privacy_batch, BatchError,
};
use crate::circuits::deposit::DepositWitness;
use crate::circuits::spend::{InputNoteWitness, OutputNoteWitness, SpendWitness};
use crate::circuits::withdraw::WithdrawWitness;
use crate::crypto::commitment::{Note, NoteCommitment, PublicKey, SpendingKey, ViewingKey};
use crate::crypto::encryption::{derive_key, encrypt_note_memo};
use crate::crypto::merkle_m31::{Digest, MerklePath};
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
    ) -> &mut Self {
        self.pending.push(PendingTx::Deposit {
            amount,
            asset_id,
            recipient_pubkey,
            recipient_viewing_key,
        });
        self
    }

    /// Add a withdraw transaction.
    pub fn withdraw(
        &mut self,
        amount: u64,
        asset_id: u32,
        note: Note,
        spending_key: SpendingKey,
        merkle_path: MerklePath,
        merkle_root: Digest,
    ) -> &mut Self {
        self.pending.push(PendingTx::Withdraw {
            amount,
            asset_id,
            note,
            spending_key,
            merkle_path,
            merkle_root,
            withdrawal_binding: [M31::from_u32_unchecked(0); RATE],
        });
        self
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
    ) -> &mut Self {
        self.pending.push(PendingTx::Withdraw {
            amount,
            asset_id,
            note,
            spending_key,
            merkle_path,
            merkle_root,
            withdrawal_binding,
        });
        self
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
    ) -> &mut Self {
        self.pending.push(PendingTx::Transfer {
            amount,
            asset_id,
            recipient_pubkey,
            recipient_viewing_key,
            sender_viewing_key,
            input_notes,
            merkle_root,
        });
        self
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
                    )?;
                    memos.push(memo_recipient);

                    // Change note → sender (encrypted with sender's viewing key)
                    let sender_pk = input_notes[0].0.owner_pubkey;
                    let memo_change = build_encrypted_memo(
                        &out_notes[1],
                        &sender_pk,
                        sender_viewing_key,
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
    if amount > MAX_NOTE_AMOUNT {
        return Err(TxBuilderError::InsufficientBalance {
            need: amount,
            have: MAX_NOTE_AMOUNT,
        });
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
    let amount_lo = M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32);
    let amount_hi = M31::from_u32_unchecked((amount >> 31) as u32);
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
    let out_note1 = Note::new(
        recipient_pubkey,
        asset_m31,
        M31::from_u32_unchecked((amount & 0x7FFFFFFF) as u32),
        M31::from_u32_unchecked((amount >> 31) as u32),
        generate_random_blinding()?,
    );

    // Output 2: change back to sender
    let sender_pk = input_notes[0].0.owner_pubkey;
    let out_note2 = Note::new(
        sender_pk,
        asset_m31,
        M31::from_u32_unchecked((change & 0x7FFFFFFF) as u32),
        M31::from_u32_unchecked((change >> 31) as u32),
        generate_random_blinding()?,
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
) -> Result<EncryptedMemoJson, TxBuilderError> {
    // Encrypt using the recipient's viewing key. Only holders of the viewing
    // key (the recipient and authorized viewers) can decrypt. The public key
    // alone is NOT sufficient — this prevents anyone from reading memos just
    // by observing the chain.
    let enc_key = derive_key(viewing_key);
    let nonce = generate_random_nonce()?;

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
            recipient_pubkey[0].0, recipient_pubkey[1].0,
            recipient_pubkey[2].0, recipient_pubkey[3].0,
        ),
    })
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn generate_random_blinding() -> Result<[M31; 4], TxBuilderError> {
    let mut bytes = [0u8; 16];
    getrandom::getrandom(&mut bytes)
        .map_err(|e| TxBuilderError::Rng(format!("{e}")))?;
    Ok([
        M31::from_u32_unchecked(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) & 0x7FFFFFFF),
        M31::from_u32_unchecked(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) & 0x7FFFFFFF),
        M31::from_u32_unchecked(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) & 0x7FFFFFFF),
        M31::from_u32_unchecked(u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) & 0x7FFFFFFF),
    ])
}

fn generate_random_nonce() -> Result<[M31; 4], TxBuilderError> {
    generate_random_blinding()
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
        builder.deposit(1000, 0, pk, vk);
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
        builder.deposit(100, 0, pk, vk);
        builder.deposit(200, 0, pk, vk);
        builder.deposit(300, 0, pk, vk);

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
        let path = tree.prove(0);
        let root = tree.root();

        let mut builder = TxBuilder::new();
        builder.withdraw(1000, 0, note, sk, path, root);

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
        let path1 = tree.prove(0);
        let path2 = tree.prove(1);

        let mut builder = TxBuilder::new();
        builder.transfer(
            1500,
            0,
            recipient_pk,
            recipient_vk,
            sender_vk,
            [(note1, sk1, path1), (note2, sk2, path2)],
            root,
        );

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
}
