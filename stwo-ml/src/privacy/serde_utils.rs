//! JSON serialization types for privacy batch proofs and CLI output.

use stwo::core::fields::m31::BaseField as M31;

use crate::circuits::batch::BatchPublicInputs;
use crate::crypto::poseidon2_m31::poseidon2_hash;

// ─── Output Types ─────────────────────────────────────────────────────────

/// Top-level batch proof output (saved to JSON).
#[derive(Clone, Debug)]
pub struct BatchProofOutput {
    pub format: String,
    pub version: u32,
    pub num_deposits: usize,
    pub num_withdrawals: usize,
    pub num_spends: usize,
    pub proof_hash: String,
    pub public_inputs_hash: String,
    pub prove_time_ms: u64,
    pub deposits: Vec<DepositPublicInputJson>,
    pub withdrawals: Vec<WithdrawPublicInputJson>,
    pub spends: Vec<SpendPublicInputJson>,
    pub encrypted_memos: Vec<EncryptedMemoJson>,
}

#[derive(Clone, Debug)]
pub struct DepositPublicInputJson {
    pub commitment: String,
    pub amount: u64,
    pub asset_id: u32,
}

#[derive(Clone, Debug)]
pub struct WithdrawPublicInputJson {
    pub merkle_root: String,
    pub nullifier: String,
    pub amount_lo: u32,
    pub amount_hi: u32,
    pub asset_id: u32,
    pub withdrawal_binding: String,
}

#[derive(Clone, Debug)]
pub struct SpendPublicInputJson {
    pub merkle_root: String,
    pub nullifiers: [String; 2],
    pub output_commitments: [String; 2],
}

#[derive(Clone, Debug)]
pub struct EncryptedMemoJson {
    pub encrypted_data: String,
    pub nonce: String,
    pub recipient_pubkey: String,
}

// ─── Transaction file schema (for --tx-file batch input) ──────────────────

#[derive(Clone, Debug)]
pub struct TxFileEntry {
    pub tx_type: String, // "deposit", "withdraw", "transfer"
    pub amount: u64,
    pub asset_id: u32,
    pub recipient_pubkey: Option<[u32; 4]>,
}

// ─── Serialization helpers ────────────────────────────────────────────────

fn m31_array_to_hex(arr: &[M31]) -> String {
    let mut s = String::with_capacity(2 + arr.len() * 8);
    s.push_str("0x");
    for &elem in arr {
        s.push_str(&format!("{:08x}", elem.0));
    }
    s
}

/// Build a BatchProofOutput from batch public inputs.
pub fn build_batch_proof_output(
    public_inputs: &BatchPublicInputs,
    proof_hash: &str,
    prove_time_ms: u64,
    encrypted_memos: Vec<EncryptedMemoJson>,
) -> BatchProofOutput {
    let deposits: Vec<DepositPublicInputJson> = public_inputs
        .deposits
        .iter()
        .map(|d| DepositPublicInputJson {
            commitment: m31_array_to_hex(&d.commitment),
            amount: d.amount,
            asset_id: d.asset_id.0,
        })
        .collect();

    let withdrawals: Vec<WithdrawPublicInputJson> = public_inputs
        .withdrawals
        .iter()
        .map(|w| WithdrawPublicInputJson {
            merkle_root: m31_array_to_hex(&w.merkle_root),
            nullifier: m31_array_to_hex(&w.nullifier),
            amount_lo: w.amount_lo.0,
            amount_hi: w.amount_hi.0,
            asset_id: w.asset_id.0,
            withdrawal_binding: m31_array_to_hex(&w.withdrawal_binding),
        })
        .collect();

    let spends: Vec<SpendPublicInputJson> = public_inputs
        .spends
        .iter()
        .map(|s| SpendPublicInputJson {
            merkle_root: m31_array_to_hex(&s.merkle_root),
            nullifiers: [
                m31_array_to_hex(&s.nullifiers[0]),
                m31_array_to_hex(&s.nullifiers[1]),
            ],
            output_commitments: [
                m31_array_to_hex(&s.output_commitments[0]),
                m31_array_to_hex(&s.output_commitments[1]),
            ],
        })
        .collect();

    // Hash all public inputs
    let mut pi_data: Vec<M31> = Vec::new();
    pi_data.push(M31::from_u32_unchecked(public_inputs.deposits.len() as u32));
    pi_data.push(M31::from_u32_unchecked(public_inputs.withdrawals.len() as u32));
    pi_data.push(M31::from_u32_unchecked(public_inputs.spends.len() as u32));
    for d in &public_inputs.deposits {
        pi_data.extend_from_slice(&d.commitment);
    }
    for w in &public_inputs.withdrawals {
        pi_data.extend_from_slice(&w.nullifier);
    }
    for s in &public_inputs.spends {
        for nul in &s.nullifiers {
            pi_data.extend_from_slice(nul);
        }
    }
    let pi_hash = poseidon2_hash(&pi_data);
    let pi_hash_hex = m31_array_to_hex(&pi_hash);

    BatchProofOutput {
        format: "vm31_privacy_batch".to_string(),
        version: 1,
        num_deposits: public_inputs.deposits.len(),
        num_withdrawals: public_inputs.withdrawals.len(),
        num_spends: public_inputs.spends.len(),
        proof_hash: proof_hash.to_string(),
        public_inputs_hash: pi_hash_hex,
        prove_time_ms,
        deposits,
        withdrawals,
        spends,
        encrypted_memos,
    }
}

/// Serialize a BatchProofOutput to pretty JSON.
pub fn batch_proof_to_json(output: &BatchProofOutput) -> String {
    let deposits_json: Vec<serde_json::Value> = output
        .deposits
        .iter()
        .map(|d| {
            serde_json::json!({
                "commitment": d.commitment,
                "amount": d.amount,
                "asset_id": d.asset_id,
            })
        })
        .collect();

    let withdrawals_json: Vec<serde_json::Value> = output
        .withdrawals
        .iter()
        .map(|w| {
            serde_json::json!({
                "merkle_root": w.merkle_root,
                "nullifier": w.nullifier,
                "amount_lo": w.amount_lo,
                "amount_hi": w.amount_hi,
                "asset_id": w.asset_id,
                "withdrawal_binding": w.withdrawal_binding,
            })
        })
        .collect();

    let spends_json: Vec<serde_json::Value> = output
        .spends
        .iter()
        .map(|s| {
            serde_json::json!({
                "merkle_root": s.merkle_root,
                "nullifiers": s.nullifiers,
                "output_commitments": s.output_commitments,
            })
        })
        .collect();

    let memos_json: Vec<serde_json::Value> = output
        .encrypted_memos
        .iter()
        .map(|m| {
            serde_json::json!({
                "encrypted_data": m.encrypted_data,
                "nonce": m.nonce,
                "recipient_pubkey": m.recipient_pubkey,
            })
        })
        .collect();

    let json = serde_json::json!({
        "format": output.format,
        "version": output.version,
        "num_deposits": output.num_deposits,
        "num_withdrawals": output.num_withdrawals,
        "num_spends": output.num_spends,
        "proof_hash": output.proof_hash,
        "public_inputs_hash": output.public_inputs_hash,
        "prove_time_ms": output.prove_time_ms,
        "deposits": deposits_json,
        "withdrawals": withdrawals_json,
        "spends": spends_json,
        "encrypted_memos": memos_json,
    });

    serde_json::to_string_pretty(&json).unwrap_or_else(|_| "{}".to_string())
}

/// Parse a tx-file JSON into entries.
pub fn parse_tx_file(contents: &str) -> Result<Vec<TxFileEntry>, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(contents).map_err(|e| format!("invalid tx-file JSON: {e}"))?;

    let arr = parsed
        .as_array()
        .ok_or_else(|| "tx-file must be a JSON array".to_string())?;

    let mut entries = Vec::with_capacity(arr.len());
    for (i, item) in arr.iter().enumerate() {
        let tx_type = item["type"]
            .as_str()
            .ok_or_else(|| format!("tx[{i}]: missing 'type'"))?
            .to_string();
        let amount = item["amount"]
            .as_u64()
            .ok_or_else(|| format!("tx[{i}]: missing 'amount'"))?;
        let asset_id = item["asset_id"].as_u64().unwrap_or(0) as u32;
        let recipient_pubkey = item["recipient_pubkey"]
            .as_array()
            .and_then(|a| {
                if a.len() >= 4 {
                    Some([
                        a[0].as_u64()? as u32,
                        a[1].as_u64()? as u32,
                        a[2].as_u64()? as u32,
                        a[3].as_u64()? as u32,
                    ])
                } else {
                    None
                }
            });

        entries.push(TxFileEntry {
            tx_type,
            amount,
            asset_id,
            recipient_pubkey,
        });
    }

    Ok(entries)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::batch::BatchPublicInputs;
    use crate::circuits::deposit::DepositPublicInputs;
    use crate::crypto::poseidon2_m31::RATE;

    #[test]
    fn test_build_batch_proof_output() {
        let pi = BatchPublicInputs {
            deposits: vec![DepositPublicInputs {
                commitment: [M31::from_u32_unchecked(42); RATE],
                amount: 1000,
                asset_id: M31::from_u32_unchecked(0),
            }],
            withdrawals: vec![],
            spends: vec![],
        };

        let output = build_batch_proof_output(&pi, "0xdeadbeef", 4320, vec![]);
        assert_eq!(output.num_deposits, 1);
        assert_eq!(output.num_withdrawals, 0);
        assert_eq!(output.prove_time_ms, 4320);
        assert!(output.deposits[0].commitment.starts_with("0x"));
    }

    #[test]
    fn test_batch_proof_to_json() {
        let pi = BatchPublicInputs {
            deposits: vec![DepositPublicInputs {
                commitment: [M31::from_u32_unchecked(42); RATE],
                amount: 1000,
                asset_id: M31::from_u32_unchecked(0),
            }],
            withdrawals: vec![],
            spends: vec![],
        };
        let output = build_batch_proof_output(&pi, "0xdeadbeef", 100, vec![]);
        let json_str = batch_proof_to_json(&output);
        assert!(json_str.contains("vm31_privacy_batch"));
        assert!(json_str.contains("\"amount\": 1000"));
    }

    #[test]
    fn test_parse_tx_file() {
        let json = r#"[
            {"type": "deposit", "amount": 1000, "asset_id": 0},
            {"type": "transfer", "amount": 500, "asset_id": 0, "recipient_pubkey": [42, 99, 7, 13]}
        ]"#;
        let entries = parse_tx_file(json).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].tx_type, "deposit");
        assert_eq!(entries[0].amount, 1000);
        assert_eq!(entries[1].tx_type, "transfer");
        assert!(entries[1].recipient_pubkey.is_some());
    }
}
