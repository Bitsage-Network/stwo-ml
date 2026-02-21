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
    /// C7: plaintext HTTP rejected for RPC URLs.
    #[error("insecure URL rejected: {0}")]
    InsecureUrl(String),
}

/// C7: Validate that an RPC URL uses HTTPS (TLS).
///
/// Rejects `http://` URLs because pool state (commitments, nullifiers, roots)
/// would be visible in cleartext to any network observer.
///
/// Exceptions:
/// - `localhost`, `127.0.0.1`, `[::1]` — local dev/testing
/// - `VM31_ALLOW_INSECURE_RPC=1` env var — explicit opt-out for CI
///
/// Returns `Ok(())` if the URL is safe, `Err(InsecureUrl)` otherwise.
pub fn validate_rpc_url(url: &str) -> Result<(), PoolClientError> {
    // Allow HTTPS always
    if url.starts_with("https://") {
        return Ok(());
    }

    // Reject non-http schemes (or missing scheme)
    if !url.starts_with("http://") {
        return Err(PoolClientError::InsecureUrl(format!(
            "'{url}' — URL must start with https:// (or http:// for localhost only)"
        )));
    }

    // http:// — check if it's localhost
    let after_scheme = &url["http://".len()..];
    let host_with_port = after_scheme.split('/').next().unwrap_or("");

    // Handle IPv6 bracket notation: [::1]:5050
    let host_no_port = if host_with_port.starts_with('[') {
        // Extract content up to ']'
        host_with_port
            .split(']')
            .next()
            .unwrap_or(host_with_port)
            .strip_prefix('[')
            .unwrap_or(host_with_port)
    } else {
        // Strip port after last ':'
        host_with_port.split(':').next().unwrap_or(host_with_port)
    };

    let is_local = matches!(host_no_port, "localhost" | "127.0.0.1" | "::1");

    if is_local {
        return Ok(());
    }

    // Check explicit opt-out
    if std::env::var("VM31_ALLOW_INSECURE_RPC").as_deref() == Ok("1") {
        eprintln!("WARNING: C7 TLS check bypassed via VM31_ALLOW_INSECURE_RPC=1 for '{url}'");
        return Ok(());
    }

    Err(PoolClientError::InsecureUrl(format!(
        "'{url}' — http:// rejected for non-localhost RPC (pool state visible in cleartext). \
         Use https://, or set VM31_ALLOW_INSECURE_RPC=1 to override."
    )))
}

/// Configuration for pool client.
#[derive(Clone, Debug)]
pub struct PoolClientConfig {
    pub rpc_url: String,
    pub pool_address: String,
    pub network: String,
    /// C6: independent RPC URLs for cross-verifying Merkle roots.
    ///
    /// Events are fetched from `rpc_url` (primary), but the final root
    /// is verified against these independent RPCs. If a malicious primary
    /// RPC feeds fake events, the verification RPCs will reject the root.
    pub verify_rpc_urls: Vec<String>,
}

impl PoolClientConfig {
    /// Build config from env vars or defaults.
    ///
    /// Reads `STARKNET_VERIFY_RPC` as a comma-separated list of independent
    /// RPC URLs for cross-verification (C6).
    ///
    /// C7: Warns at construction if any URL uses plaintext HTTP for a non-local
    /// host. Enforcement happens at call time (`starknet_call_at` / events).
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
        let verify_rpc_urls: Vec<String> = std::env::var("STARKNET_VERIFY_RPC")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // C7: early warning (enforcement at call time)
        if let Err(e) = validate_rpc_url(&rpc_url) {
            eprintln!("WARNING: primary RPC URL will be rejected at call time: {e}");
        }
        for url in &verify_rpc_urls {
            if let Err(e) = validate_rpc_url(url) {
                eprintln!("WARNING: verify RPC URL will be rejected at call time: {e}");
            }
        }

        Self {
            rpc_url,
            pool_address,
            network: network.to_string(),
            verify_rpc_urls,
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

    /// Call a read-only function on the pool contract via the primary RPC.
    #[cfg(feature = "audit-http")]
    fn starknet_call(
        &self,
        function: &str,
        calldata: &[&str],
    ) -> Result<Vec<String>, PoolClientError> {
        Self::starknet_call_at(
            &self.config.rpc_url,
            &self.config.pool_address,
            function,
            calldata,
        )
    }

    /// Call a read-only function on the pool contract via a specific RPC URL.
    ///
    /// C7: Rejects plaintext HTTP for non-localhost URLs.
    #[cfg(feature = "audit-http")]
    fn starknet_call_at(
        rpc_url: &str,
        contract_address: &str,
        function: &str,
        calldata: &[&str],
    ) -> Result<Vec<String>, PoolClientError> {
        validate_rpc_url(rpc_url)?;

        let calldata_json: Vec<serde_json::Value> = calldata
            .iter()
            .map(|s| serde_json::Value::String(s.to_string()))
            .collect();

        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "starknet_call",
            "params": [{
                "contract_address": contract_address,
                "entry_point_selector": function_selector(function),
                "calldata": calldata_json,
            }, "latest"]
        });

        let body_bytes = serde_json::to_vec(&body)
            .map_err(|e| PoolClientError::Parse(format!("serialize: {e}")))?;

        let mut resp = ureq::post(rpc_url)
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

    /// C6: Cross-verify a Merkle root against independent RPCs.
    ///
    /// Checks the primary RPC first, then queries each verification RPC.
    /// Returns a `CrossVerifyResult` with details about which RPCs agreed.
    ///
    /// If no verification RPCs are configured, `verified` is false (primary-only).
    /// If verification RPCs are configured but ALL are unreachable, returns error.
    #[cfg(feature = "audit-http")]
    pub fn cross_verify_root(
        &self,
        root: &[M31; RATE],
    ) -> Result<CrossVerifyResult, PoolClientError> {
        let primary_ok = self.is_known_root(root)?;

        if self.config.verify_rpc_urls.is_empty() {
            return Ok(CrossVerifyResult {
                primary_confirmed: primary_ok,
                cross_verified: false,
                verify_total: 0,
                verify_confirmed: 0,
                verify_unreachable: 0,
            });
        }

        let calldata: Vec<String> = root.iter().map(|m| format!("0x{:x}", m.0)).collect();
        let calldata_refs: Vec<&str> = calldata.iter().map(|s| s.as_str()).collect();

        let mut confirmed = 0u32;
        let mut unreachable = 0u32;

        for url in &self.config.verify_rpc_urls {
            match Self::starknet_call_at(
                url,
                &self.config.pool_address,
                "is_known_root",
                &calldata_refs,
            ) {
                Ok(result) => {
                    if !result.is_empty() && result[0] != "0x0" {
                        confirmed += 1;
                    }
                }
                Err(_) => {
                    unreachable += 1;
                }
            }
        }

        let total = self.config.verify_rpc_urls.len() as u32;

        // If ALL verify RPCs were unreachable, that's an error — we can't
        // degrade to primary-only when verification was explicitly configured.
        if unreachable == total {
            return Err(PoolClientError::Rpc(format!(
                "C6: all {} verification RPCs unreachable — cannot verify root",
                total
            )));
        }

        Ok(CrossVerifyResult {
            primary_confirmed: primary_ok,
            cross_verified: true,
            verify_total: total,
            verify_confirmed: confirmed,
            verify_unreachable: unreachable,
        })
    }
}

/// Result of cross-RPC root verification (C6).
#[derive(Debug, Clone)]
pub struct CrossVerifyResult {
    /// Primary RPC confirmed the root.
    pub primary_confirmed: bool,
    /// Whether cross-verification was performed (false = no verify RPCs configured).
    pub cross_verified: bool,
    /// Total number of verification RPCs queried.
    pub verify_total: u32,
    /// Number of verification RPCs that confirmed the root.
    pub verify_confirmed: u32,
    /// Number of verification RPCs that were unreachable.
    pub verify_unreachable: u32,
}

impl CrossVerifyResult {
    /// True if root is confirmed: primary says yes AND (no cross-verify, or at least one
    /// independent RPC agrees).
    pub fn is_confirmed(&self) -> bool {
        if !self.primary_confirmed {
            return false;
        }
        if !self.cross_verified {
            return true; // no verify RPCs configured, primary-only
        }
        // Require majority of reachable verification RPCs to agree.
        let reachable = self.verify_total - self.verify_unreachable;
        reachable == 0 || self.verify_confirmed * 2 > reachable
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
    /// C7: Rejects plaintext HTTP for non-localhost URLs.
    pub fn get_note_inserted_events(
        &self,
        from_block: u64,
    ) -> Result<Vec<NoteEvent>, PoolClientError> {
        validate_rpc_url(&self.config.rpc_url)?;

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

    // ─── C6 Cross-RPC Verification Tests ─────────────────────────────────

    #[test]
    fn test_c6_config_parses_verify_rpcs() {
        // Default config with no env var should have empty verify list
        let config = PoolClientConfig::from_env("sepolia");
        // (STARKNET_VERIFY_RPC not set in test env)
        // Just verify the field exists and is a Vec
        assert!(config.verify_rpc_urls.len() <= 10); // sanity
    }

    #[test]
    fn test_c6_config_manual_verify_rpcs() {
        let config = PoolClientConfig {
            rpc_url: "https://primary.example.com".to_string(),
            pool_address: "0x123".to_string(),
            network: "sepolia".to_string(),
            verify_rpc_urls: vec![
                "https://verify1.example.com".to_string(),
                "https://verify2.example.com".to_string(),
            ],
        };
        assert_eq!(config.verify_rpc_urls.len(), 2);
        assert_eq!(config.verify_rpc_urls[0], "https://verify1.example.com");
    }

    #[test]
    fn test_c6_cross_verify_result_primary_only() {
        // No verify RPCs configured → primary-only
        let cv = CrossVerifyResult {
            primary_confirmed: true,
            cross_verified: false,
            verify_total: 0,
            verify_confirmed: 0,
            verify_unreachable: 0,
        };
        assert!(
            cv.is_confirmed(),
            "primary-only should confirm when primary says yes"
        );

        let cv_fail = CrossVerifyResult {
            primary_confirmed: false,
            cross_verified: false,
            verify_total: 0,
            verify_confirmed: 0,
            verify_unreachable: 0,
        };
        assert!(
            !cv_fail.is_confirmed(),
            "primary-only should reject when primary says no"
        );
    }

    #[test]
    fn test_c6_cross_verify_result_with_verify_rpcs() {
        // Primary + 2 verify RPCs, both confirm
        let cv = CrossVerifyResult {
            primary_confirmed: true,
            cross_verified: true,
            verify_total: 2,
            verify_confirmed: 2,
            verify_unreachable: 0,
        };
        assert!(cv.is_confirmed());

        // Primary yes, but all verify RPCs reject → NOT confirmed
        let cv_reject = CrossVerifyResult {
            primary_confirmed: true,
            cross_verified: true,
            verify_total: 2,
            verify_confirmed: 0,
            verify_unreachable: 0,
        };
        assert!(
            !cv_reject.is_confirmed(),
            "C6: primary-only confirmation must be rejected when verify RPCs disagree"
        );

        // Primary yes, 1 of 2 confirm → confirmed (at least one)
        let cv_partial = CrossVerifyResult {
            primary_confirmed: true,
            cross_verified: true,
            verify_total: 2,
            verify_confirmed: 1,
            verify_unreachable: 1,
        };
        assert!(cv_partial.is_confirmed());
    }

    #[test]
    fn test_c6_cross_verify_primary_no_overrides_verify() {
        // Even if verify RPCs say yes, primary must also confirm
        let cv = CrossVerifyResult {
            primary_confirmed: false,
            cross_verified: true,
            verify_total: 2,
            verify_confirmed: 2,
            verify_unreachable: 0,
        };
        assert!(
            !cv.is_confirmed(),
            "C6: primary rejection must override verify confirmations"
        );
    }

    // ─── C7 TLS Enforcement Tests ────────────────────────────────────────

    #[test]
    fn test_c7_https_accepted() {
        assert!(validate_rpc_url("https://rpc.example.com").is_ok());
        assert!(validate_rpc_url("https://rpc.example.com:443/v1").is_ok());
        assert!(validate_rpc_url("https://free-rpc.nethermind.io/sepolia-juno/").is_ok());
    }

    #[test]
    fn test_c7_http_remote_rejected() {
        let result = validate_rpc_url("http://rpc.example.com");
        assert!(
            result.is_err(),
            "C7: http:// to remote host must be rejected"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("insecure"),
            "error should mention insecure: {err}"
        );
    }

    #[test]
    fn test_c7_http_localhost_accepted() {
        assert!(validate_rpc_url("http://localhost:5050").is_ok());
        assert!(validate_rpc_url("http://localhost/rpc").is_ok());
        assert!(validate_rpc_url("http://127.0.0.1:5050").is_ok());
        assert!(validate_rpc_url("http://127.0.0.1").is_ok());
        assert!(validate_rpc_url("http://[::1]:5050").is_ok());
    }

    #[test]
    fn test_c7_missing_scheme_rejected() {
        let result = validate_rpc_url("rpc.example.com");
        assert!(result.is_err(), "C7: URL without scheme must be rejected");
    }

    #[test]
    fn test_c7_http_various_remote_hosts_rejected() {
        // Various non-localhost hosts that should all be rejected
        for url in &[
            "http://10.0.0.1:8545",
            "http://192.168.1.100:8545",
            "http://my-node.internal:8545",
            "http://rpc.starknet.io",
        ] {
            assert!(validate_rpc_url(url).is_err(), "C7: should reject {url}");
        }
    }
}
