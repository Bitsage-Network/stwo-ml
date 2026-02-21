//! On-chain pool interaction: read-only Starknet queries via JSON-RPC.
//!
//! Provides `PoolClient` for querying the VM31 pool contract state
//! and `serialize_batch_for_pool` for building on-chain calldata.

#[cfg(any(feature = "audit-http", test))]
use stwo::core::fields::m31::BaseField as M31;

#[cfg(any(feature = "audit-http", test))]
use crate::crypto::poseidon2_m31::RATE;

// ─── Types ────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum PoolClientError {
    #[error("RPC error: {0}")]
    Rpc(String),
    #[error("response parse error: {0}")]
    Parse(String),
    #[error("contract error: {0}")]
    Contract(String),
}

/// Configuration for pool client.
#[derive(Clone, Debug)]
pub struct PoolClientConfig {
    pub rpc_url: String,
    pub pool_address: String,
    pub network: String,
}

impl PoolClientConfig {
    /// Build config from env vars or defaults.
    pub fn from_env(network: &str) -> Self {
        let rpc_url = std::env::var("STARKNET_RPC").unwrap_or_else(|_| match network {
            "mainnet" => "https://free-rpc.nethermind.io/mainnet-juno/".to_string(),
            _ => "https://free-rpc.nethermind.io/sepolia-juno/".to_string(),
        });
        let pool_address = std::env::var("VM31_POOL_ADDRESS").unwrap_or_else(|_| match network {
            "sepolia" => {
                "0x07cf94e27a60b94658ec908a00a9bb6dfff03358e952d9d48a8ed0be080ce1f9".to_string()
            }
            _ => String::new(),
        });

        Self {
            rpc_url,
            pool_address,
            network: network.to_string(),
        }
    }
}

/// Pool client for read-only Starknet queries.
pub struct PoolClient {
    pub config: PoolClientConfig,
}

impl PoolClient {
    pub fn new(config: PoolClientConfig) -> Self {
        Self { config }
    }

    /// Call a read-only function on the pool contract.
    #[cfg(feature = "audit-http")]
    fn starknet_call(
        &self,
        function: &str,
        calldata: &[&str],
    ) -> Result<Vec<String>, PoolClientError> {
        let calldata_json: Vec<serde_json::Value> = calldata
            .iter()
            .map(|s| serde_json::Value::String(s.to_string()))
            .collect();

        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "starknet_call",
            "params": [{
                "contract_address": self.config.pool_address,
                "entry_point_selector": function_selector(function),
                "calldata": calldata_json,
            }, "latest"]
        });

        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| PoolClientError::Parse(format!("serialize: {e}")))?;

        let mut resp = ureq::post(&self.config.rpc_url)
            .header("Content-Type", "application/json")
            .send(&body_bytes)
            .map_err(|e| PoolClientError::Rpc(format!("{e}")))?;

        let resp_bytes = resp
            .body_mut()
            .read_to_vec()
            .map_err(|e| PoolClientError::Parse(format!("read body: {e}")))?;

        let resp: serde_json::Value = serde_json::from_slice(&resp_bytes)
            .map_err(|e| PoolClientError::Parse(format!("json: {e}")))?;

        if let Some(error) = resp.get("error") {
            return Err(PoolClientError::Contract(error.to_string()));
        }

        let result = resp["result"]
            .as_array()
            .ok_or_else(|| PoolClientError::Parse("missing result array".into()))?
            .iter()
            .map(|v| v.as_str().unwrap_or("0x0").to_string())
            .collect();

        Ok(result)
    }

    /// Get the current Merkle root of the pool.
    #[cfg(feature = "audit-http")]
    pub fn get_merkle_root(&self) -> Result<[M31; RATE], PoolClientError> {
        let result = self.starknet_call("get_merkle_root", &[])?;
        parse_m31_digest_from_result(&result)
    }

    /// Get the tree size (number of leaves inserted).
    #[cfg(feature = "audit-http")]
    pub fn get_tree_size(&self) -> Result<u64, PoolClientError> {
        let result = self.starknet_call("get_tree_size", &[])?;
        if result.is_empty() {
            return Ok(0);
        }
        let hex = result[0].strip_prefix("0x").unwrap_or(&result[0]);
        u64::from_str_radix(hex, 16).map_err(|e| PoolClientError::Parse(e.to_string()))
    }

    /// Check if a nullifier has been spent.
    #[cfg(feature = "audit-http")]
    pub fn is_nullifier_spent(&self, nullifier: &[M31; RATE]) -> Result<bool, PoolClientError> {
        let calldata: Vec<String> = nullifier.iter().map(|m| format!("0x{:x}", m.0)).collect();
        let calldata_refs: Vec<&str> = calldata.iter().map(|s| s.as_str()).collect();
        let result = self.starknet_call("is_nullifier_spent", &calldata_refs)?;
        Ok(!result.is_empty() && result[0] != "0x0")
    }

    /// Check if a root is a known historical root.
    #[cfg(feature = "audit-http")]
    pub fn is_known_root(&self, root: &[M31; RATE]) -> Result<bool, PoolClientError> {
        let calldata: Vec<String> = root.iter().map(|m| format!("0x{:x}", m.0)).collect();
        let calldata_refs: Vec<&str> = calldata.iter().map(|s| s.as_str()).collect();
        let result = self.starknet_call("is_known_root", &calldata_refs)?;
        Ok(!result.is_empty() && result[0] != "0x0")
    }

    /// Get batch status (0=none, 1=submitted, 2=finalized).
    #[cfg(feature = "audit-http")]
    pub fn get_batch_status(&self, batch_id: &str) -> Result<u8, PoolClientError> {
        let result = self.starknet_call("get_batch_status", &[batch_id])?;
        if result.is_empty() {
            return Ok(0);
        }
        let hex = result[0].strip_prefix("0x").unwrap_or(&result[0]);
        Ok(u8::from_str_radix(hex, 16).unwrap_or(0))
    }
}

// ─── Event Types ─────────────────────────────────────────────────────────

/// A NoteInserted event from the pool contract.
#[cfg(any(feature = "audit-http", test))]
#[derive(Clone, Debug)]
pub struct NoteEvent {
    pub leaf_index: u64,
    pub commitment: [M31; RATE],
    pub block_number: u64,
}

// ─── Event Scanning ──────────────────────────────────────────────────────

#[cfg(feature = "audit-http")]
impl PoolClient {
    /// Fetch all NoteInserted events from the pool contract since `from_block`.
    ///
    /// Uses `starknet_getEvents` RPC with pagination. Events are returned
    /// sorted by leaf_index ascending.
    ///
    /// Event data layout (matching vm31_pool.cairo `NoteInserted`):
    ///   data[0] = leaf_index
    ///   data[1] = commitment.lo (packed felt252)
    ///   data[2] = commitment.hi (packed felt252)
    pub fn get_note_inserted_events(
        &self,
        from_block: u64,
    ) -> Result<Vec<NoteEvent>, PoolClientError> {
        let note_inserted_key = event_key("NoteInserted");
        let mut all_events = Vec::new();
        let mut continuation_token: Option<String> = None;
        let max_pages = 100;

        for _ in 0..max_pages {
            let mut filter = serde_json::json!({
                "from_block": { "block_number": from_block },
                "to_block": "latest",
                "address": self.config.pool_address,
                "keys": [[note_inserted_key]],
                "chunk_size": 100
            });

            if let Some(ref token) = continuation_token {
                filter["continuation_token"] = serde_json::Value::String(token.clone());
            }

            let body = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "starknet_getEvents",
                "params": [filter]
            });

            let body_bytes = serde_json::to_vec(&body)
                .map_err(|e| PoolClientError::Parse(format!("serialize: {e}")))?;

            let mut resp = ureq::post(&self.config.rpc_url)
                .header("Content-Type", "application/json")
                .send(&body_bytes)
                .map_err(|e| PoolClientError::Rpc(format!("{e}")))?;

            let resp_bytes = resp
                .body_mut()
                .read_to_vec()
                .map_err(|e| PoolClientError::Parse(format!("read body: {e}")))?;

            let resp: serde_json::Value = serde_json::from_slice(&resp_bytes)
                .map_err(|e| PoolClientError::Parse(format!("json: {e}")))?;

            if let Some(error) = resp.get("error") {
                return Err(PoolClientError::Rpc(error.to_string()));
            }

            let result = resp
                .get("result")
                .ok_or_else(|| PoolClientError::Parse("missing result".into()))?;

            let events = result
                .get("events")
                .and_then(|e| e.as_array())
                .ok_or_else(|| PoolClientError::Parse("missing events array".into()))?;

            for event in events {
                let data = event
                    .get("data")
                    .and_then(|d| d.as_array())
                    .ok_or_else(|| PoolClientError::Parse("missing event data".into()))?;

                if data.len() < 3 {
                    continue; // malformed event, skip
                }

                let leaf_index = parse_felt_u64(data[0].as_str().unwrap_or("0x0"))?;
                let commitment = parse_lo_hi_commitment(
                    data[1].as_str().unwrap_or("0x0"),
                    data[2].as_str().unwrap_or("0x0"),
                )?;

                let block_number = event
                    .get("block_number")
                    .and_then(|b| b.as_u64())
                    .unwrap_or(0);

                all_events.push(NoteEvent {
                    leaf_index,
                    commitment,
                    block_number,
                });
            }

            // Check for continuation token
            continuation_token = result
                .get("continuation_token")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string());

            if continuation_token.is_none() {
                break;
            }
        }

        // Sort by leaf_index ascending
        all_events.sort_by_key(|e| e.leaf_index);
        Ok(all_events)
    }
}

/// Parse a hex felt string to u64.
#[cfg(feature = "audit-http")]
fn parse_felt_u64(hex: &str) -> Result<u64, PoolClientError> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    u64::from_str_radix(hex, 16).map_err(|e| PoolClientError::Parse(format!("felt u64: {e}")))
}

/// Parse (lo, hi) felt252 pair to [M31; 8] commitment.
///
/// Matches `pack_m31x8` in vm31_merkle.cairo:
///   lo = v[0] + v[1]*2^31 + v[2]*2^62 + v[3]*2^93
///   hi = v[4] + v[5]*2^31 + v[6]*2^62 + v[7]*2^93
#[cfg(any(feature = "audit-http", test))]
fn parse_lo_hi_commitment(lo_hex: &str, hi_hex: &str) -> Result<[M31; RATE], PoolClientError> {
    let parse_half = |hex: &str| -> Result<[M31; 4], PoolClientError> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        let val = u128::from_str_radix(hex, 16)
            .map_err(|e| PoolClientError::Parse(format!("lo/hi parse: {e}")))?;
        let mask = (1u128 << 31) - 1;
        Ok([
            M31::from_u32_unchecked((val & mask) as u32),
            M31::from_u32_unchecked(((val >> 31) & mask) as u32),
            M31::from_u32_unchecked(((val >> 62) & mask) as u32),
            M31::from_u32_unchecked(((val >> 93) & mask) as u32),
        ])
    };

    let lo = parse_half(lo_hex)?;
    let hi = parse_half(hi_hex)?;
    Ok([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]])
}

// ─── Calldata Serialization ───────────────────────────────────────────────

/// Build `submit_batch_proof` calldata for sncast invoke.
///
/// Matches the Cairo `Serde` layout expected by `vm31_pool::submit_batch_proof`.
pub fn build_submit_batch_calldata(
    num_deposits: usize,
    num_withdrawals: usize,
    num_spends: usize,
    proof_data: &[String],
) -> Vec<String> {
    let mut calldata = Vec::new();

    // Transaction counts
    calldata.push(format!("0x{num_deposits:x}"));
    calldata.push(format!("0x{num_withdrawals:x}"));
    calldata.push(format!("0x{num_spends:x}"));

    // Proof data length + elements
    calldata.push(format!("0x{:x}", proof_data.len()));
    calldata.extend(proof_data.iter().cloned());

    calldata
}

/// Build a pool status summary as JSON string.
pub fn format_pool_status(root: &str, tree_size: u64, network: &str, pool_address: &str) -> String {
    serde_json::json!({
        "merkle_root": root,
        "tree_size": tree_size,
        "network": network,
        "pool_address": pool_address,
    })
    .to_string()
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Compute the Starknet function selector: sn_keccak(name) truncated to 250 bits.
#[cfg(feature = "audit-http")]
fn function_selector(name: &str) -> String {
    use sha3::{Digest as Sha3Digest, Keccak256};

    let hash = Keccak256::digest(name.as_bytes());
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&hash);
    // Truncate to 250 bits (sn_keccak convention)
    bytes[0] &= 0x03;
    // Convert to felt252 hex, trimming leading zeros
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("0x{}", hex.trim_start_matches('0'))
}

/// Compute an event key using sn_keccak (same as function_selector).
#[cfg(feature = "audit-http")]
fn event_key(name: &str) -> String {
    function_selector(name)
}

#[cfg(feature = "audit-http")]
fn parse_m31_digest_from_result(result: &[String]) -> Result<[M31; RATE], PoolClientError> {
    if result.len() < RATE {
        return Err(PoolClientError::Parse(format!(
            "expected {} values, got {}",
            RATE,
            result.len()
        )));
    }
    let mut digest = [M31::from_u32_unchecked(0); RATE];
    for i in 0..RATE {
        let hex = result[i].strip_prefix("0x").unwrap_or(&result[i]);
        let val = u32::from_str_radix(hex, 16)
            .map_err(|e| PoolClientError::Parse(format!("invalid hex '{}': {}", result[i], e)))?;
        digest[i] = M31::from_u32_unchecked(val);
    }
    Ok(digest)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_client_config_from_env() {
        let config = PoolClientConfig::from_env("sepolia");
        assert!(config.rpc_url.contains("nethermind") || config.rpc_url.contains("localhost"));
        assert_eq!(config.network, "sepolia");
    }

    #[test]
    fn test_build_submit_batch_calldata() {
        let calldata =
            build_submit_batch_calldata(2, 1, 0, &["0xaa".to_string(), "0xbb".to_string()]);
        assert_eq!(calldata[0], "0x2"); // deposits
        assert_eq!(calldata[1], "0x1"); // withdrawals
        assert_eq!(calldata[2], "0x0"); // spends
        assert_eq!(calldata[3], "0x2"); // proof data len
        assert_eq!(calldata[4], "0xaa");
        assert_eq!(calldata[5], "0xbb");
    }

    #[test]
    fn test_format_pool_status() {
        let json = format_pool_status("0xabc", 42, "sepolia", "0x123");
        assert!(json.contains("\"tree_size\":42"));
        assert!(json.contains("\"merkle_root\":\"0xabc\""));
    }

    #[test]
    fn test_function_selector_sn_keccak() {
        // sn_keccak("get_merkle_root") should produce a known selector.
        // We verify the selector is a non-empty hex string with 0x prefix
        // and that the top bits are truncated (first byte <= 0x03).
        #[cfg(feature = "audit-http")]
        {
            let sel = function_selector("get_merkle_root");
            assert!(sel.starts_with("0x"), "selector should start with 0x");
            assert!(sel.len() > 2, "selector should not be empty");

            // Verify truncation: parse back and check top 6 bits are 0
            let hex = sel.strip_prefix("0x").unwrap();
            let padded = format!("{:0>64}", hex);
            let first_byte = u8::from_str_radix(&padded[..2], 16).unwrap();
            assert!(
                first_byte <= 0x03,
                "top 6 bits should be 0 (sn_keccak truncation)"
            );
        }
    }

    #[test]
    fn test_event_key_note_inserted() {
        #[cfg(feature = "audit-http")]
        {
            let key = event_key("NoteInserted");
            assert!(key.starts_with("0x"));
            assert!(key.len() > 2);
        }
    }

    #[test]
    fn test_parse_lo_hi_commitment() {
        // lo encodes [1, 0, 0, 0], hi encodes [2, 0, 0, 0]
        let lo = "0x1"; // 1
        let hi = "0x2"; // 2
        let commitment = parse_lo_hi_commitment(lo, hi).unwrap();
        assert_eq!(commitment[0], M31::from_u32_unchecked(1));
        assert_eq!(commitment[1], M31::from_u32_unchecked(0));
        assert_eq!(commitment[4], M31::from_u32_unchecked(2));
        assert_eq!(commitment[5], M31::from_u32_unchecked(0));
    }

    #[test]
    fn test_parse_lo_hi_roundtrip() {
        // Pack known M31 values and verify round-trip
        let vals: [u32; 8] = [42, 1000, 999999, 0x7FFFFFFF, 1, 0, 12345, 67890];

        // Pack lo half: v[0] + v[1]*2^31 + v[2]*2^62 + v[3]*2^93
        let lo: u128 = (vals[0] as u128)
            | ((vals[1] as u128) << 31)
            | ((vals[2] as u128) << 62)
            | ((vals[3] as u128) << 93);
        let hi: u128 = (vals[4] as u128)
            | ((vals[5] as u128) << 31)
            | ((vals[6] as u128) << 62)
            | ((vals[7] as u128) << 93);

        let lo_hex = format!("0x{lo:x}");
        let hi_hex = format!("0x{hi:x}");

        let commitment = parse_lo_hi_commitment(&lo_hex, &hi_hex).unwrap();
        for i in 0..8 {
            assert_eq!(
                commitment[i],
                M31::from_u32_unchecked(vals[i]),
                "mismatch at index {i}"
            );
        }
    }
}
