//! Arweave storage client for encrypted audit reports.
//!
//! Provides upload/download/status operations against an Arweave gateway
//! (or Irys bundler). The HTTP transport is abstracted via the
//! [`HttpTransport`] trait so callers can plug in reqwest, ureq, or mocks.

use crate::audit::types::AuditError;

// ─── HTTP Transport Abstraction ─────────────────────────────────────────────

/// HTTP transport trait — lets us swap reqwest, ureq, curl, or a mock.
pub trait HttpTransport: Send + Sync {
    /// POST bytes to a URL with headers. Returns the response body.
    fn post(&self, url: &str, headers: &[(&str, &str)], body: &[u8])
        -> Result<Vec<u8>, AuditError>;

    /// GET a URL. Returns the response body.
    fn get(&self, url: &str) -> Result<Vec<u8>, AuditError>;
}

// ─── Arweave Tags ───────────────────────────────────────────────────────────

/// Tag attached to an Arweave transaction for querying.
#[derive(Debug, Clone)]
pub struct ArweaveTag {
    pub name: String,
    pub value: String,
}

// ─── Transaction Status ─────────────────────────────────────────────────────

/// Status of an Arweave transaction.
#[derive(Debug, Clone)]
pub enum TxStatus {
    /// Transaction is pending confirmation.
    Pending,
    /// Transaction is confirmed at the given block height.
    Confirmed { block_height: u64 },
    /// Transaction not found.
    NotFound,
}

// ─── Storage Receipt ────────────────────────────────────────────────────────

/// Receipt from a successful Arweave upload.
#[derive(Debug, Clone)]
pub struct StorageReceipt {
    /// Arweave transaction ID.
    pub tx_id: String,
    /// Size of the uploaded data in bytes.
    pub size_bytes: usize,
    /// Gateway URL to retrieve the data.
    pub gateway_url: String,
}

// ─── Arweave Client ─────────────────────────────────────────────────────────

/// Client for uploading/downloading audit reports to/from Arweave.
///
/// Uses an [`HttpTransport`] for actual HTTP calls, making the client
/// testable and transport-agnostic.
pub struct ArweaveClient {
    /// Arweave gateway URL (e.g., "https://arweave.net").
    gateway: String,
    /// Irys/Bundlr upload endpoint (e.g., "https://node1.irys.xyz").
    bundler: String,
    /// HTTP transport implementation.
    transport: Box<dyn HttpTransport>,
    /// Optional Irys API auth token for authenticated uploads.
    auth_token: Option<String>,
}

impl ArweaveClient {
    /// Create a new Arweave client.
    ///
    /// - `gateway`: Arweave read gateway (e.g., `"https://arweave.net"`).
    /// - `bundler`: Irys upload endpoint (e.g., `"https://node1.irys.xyz"`).
    /// - `transport`: HTTP transport implementation.
    pub fn new(
        gateway: impl Into<String>,
        bundler: impl Into<String>,
        transport: Box<dyn HttpTransport>,
    ) -> Self {
        Self {
            gateway: gateway.into(),
            bundler: bundler.into(),
            transport,
            auth_token: None,
        }
    }

    /// Create with default gateways.
    pub fn with_defaults(transport: Box<dyn HttpTransport>) -> Self {
        Self::new("https://arweave.net", "https://node1.irys.xyz", transport)
    }

    /// Set an Irys API auth token for authenticated uploads.
    pub fn with_auth(mut self, token: impl Into<String>) -> Self {
        self.auth_token = Some(token.into());
        self
    }

    /// Upload data to Arweave via Irys bundler.
    ///
    /// Tags are attached for queryability:
    /// - `App-Name: Obelysk-Audit`
    /// - `Audit-ID: {audit_id}`
    /// - `Model-ID: {model_id}`
    /// - Plus any additional tags.
    ///
    /// The upload uses Irys's `/tx/arweave` endpoint with a JSON body:
    /// `{"data": "<base64>", "tags": [{"name": "...", "value": "..."}]}`.
    pub fn upload(
        &self,
        data: &[u8],
        audit_id: &str,
        model_id: &str,
        extra_tags: &[ArweaveTag],
    ) -> Result<StorageReceipt, AuditError> {
        // Build tags for queryability.
        let mut tags = vec![
            ArweaveTag {
                name: "App-Name".to_string(),
                value: "Obelysk-Audit".to_string(),
            },
            ArweaveTag {
                name: "Content-Type".to_string(),
                value: "application/octet-stream".to_string(),
            },
            ArweaveTag {
                name: "Audit-ID".to_string(),
                value: audit_id.to_string(),
            },
            ArweaveTag {
                name: "Model-ID".to_string(),
                value: model_id.to_string(),
            },
        ];
        tags.extend_from_slice(extra_tags);

        // Irys upload: POST JSON envelope to /tx/arweave endpoint.
        let url = format!("{}/tx/arweave", self.bundler);

        // Build JSON body: {"data": "<base64>", "tags": [{"name": ..., "value": ...}]}
        let data_b64 = base64_encode(data);
        let tags_json: Vec<String> = tags
            .iter()
            .map(|t| {
                format!(
                    "{{\"name\":\"{}\",\"value\":\"{}\"}}",
                    escape_json_string(&t.name),
                    escape_json_string(&t.value)
                )
            })
            .collect();

        let body = format!(
            "{{\"data\":\"{}\",\"tags\":[{}]}}",
            data_b64,
            tags_json.join(",")
        );

        let mut headers: Vec<(&str, &str)> = vec![("Content-Type", "application/json")];
        let auth_val;
        if let Some(ref token) = self.auth_token {
            auth_val = format!("Bearer {}", token);
            headers.push(("Authorization", &auth_val));
        }

        let response = self.transport.post(&url, &headers, body.as_bytes())?;

        // Parse response to extract tx_id.
        let tx_id = parse_upload_response(&response)?;

        Ok(StorageReceipt {
            tx_id: tx_id.clone(),
            size_bytes: data.len(),
            gateway_url: format!("{}/{}", self.gateway, tx_id),
        })
    }

    /// Download data from Arweave by transaction ID.
    pub fn download(&self, tx_id: &str) -> Result<Vec<u8>, AuditError> {
        let url = format!("{}/{}", self.gateway, tx_id);
        self.transport.get(&url)
    }

    /// Check the status of an Arweave transaction.
    pub fn status(&self, tx_id: &str) -> Result<TxStatus, AuditError> {
        let url = format!("{}/tx/{}/status", self.gateway, tx_id);
        match self.transport.get(&url) {
            Ok(body) => {
                let text = String::from_utf8_lossy(&body);
                if text.contains("block_height") {
                    // Parse block height from JSON.
                    if let Some(height) = extract_block_height(&text) {
                        Ok(TxStatus::Confirmed {
                            block_height: height,
                        })
                    } else {
                        Ok(TxStatus::Pending)
                    }
                } else {
                    Ok(TxStatus::Pending)
                }
            }
            Err(_) => Ok(TxStatus::NotFound),
        }
    }

    /// Get the gateway URL for this client.
    pub fn gateway(&self) -> &str {
        &self.gateway
    }
}

// ─── Encoding Helpers ───────────────────────────────────────────────────────

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn base64_decode(encoded: &str) -> Option<Vec<u8>> {
    fn b64_val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            b'=' => Some(0),
            _ => None,
        }
    }
    let bytes: Vec<u8> = encoded
        .bytes()
        .filter(|&b| b != b'\n' && b != b'\r')
        .collect();
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let a = b64_val(chunk[0])?;
        let b = b64_val(chunk[1])?;
        let c = b64_val(chunk[2])?;
        let d = b64_val(chunk[3])?;
        out.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            out.push((b << 4) | (c >> 2));
        }
        if chunk[3] != b'=' {
            out.push((c << 6) | d);
        }
    }
    Some(out)
}

fn extract_data_from_json_envelope(body: &[u8]) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(body).ok()?;
    // Find "data":"<base64>"
    let data_key = text.find("\"data\"")?;
    let after = &text[data_key + 6..];
    let colon = after.find(':')?;
    let after_colon = after[colon + 1..].trim();
    if !after_colon.starts_with('"') {
        return None;
    }
    let end_quote = after_colon[1..].find('"')?;
    let b64_str = &after_colon[1..1 + end_quote];
    base64_decode(b64_str)
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

// ─── Response Parsing ───────────────────────────────────────────────────────

fn parse_upload_response(response: &[u8]) -> Result<String, AuditError> {
    let text = String::from_utf8_lossy(response);

    // Try to parse as JSON with "id" field.
    if let Some(id_start) = text.find("\"id\"") {
        let after = &text[id_start + 4..];
        if let Some(colon) = after.find(':') {
            let after_colon = after[colon + 1..].trim();
            if after_colon.starts_with('"') {
                if let Some(end) = after_colon[1..].find('"') {
                    return Ok(after_colon[1..1 + end].to_string());
                }
            }
        }
    }

    // Fallback: treat entire response as tx_id if it looks like one.
    let trimmed = text.trim().trim_matches('"');
    if !trimmed.is_empty()
        && trimmed.len() <= 64
        && trimmed
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Ok(trimmed.to_string());
    }

    Err(AuditError::Storage(format!(
        "Failed to parse upload response: {}",
        &text[..text.len().min(200)]
    )))
}

fn extract_block_height(json: &str) -> Option<u64> {
    if let Some(start) = json.find("\"block_height\"") {
        let after = &json[start + 14..];
        if let Some(colon) = after.find(':') {
            let num_str: String = after[colon + 1..]
                .trim()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            return num_str.parse().ok();
        }
    }
    None
}

// ─── Relay Client ────────────────────────────────────────────────────────────

/// Client that proxies audit uploads through the Obelysk relay service.
///
/// When users don't have an IRYS_TOKEN, the relay (running on the coordinator EC2)
/// holds the token server-side and forwards uploads to Irys. Users never need
/// their own Arweave/Irys account.
///
/// Default relay URL: `https://relay.obelysk.xyz` (override via `OBELYSK_RELAY_URL`).
pub struct RelayClient {
    /// Relay service URL (e.g., "https://relay.obelysk.xyz").
    relay_url: String,
    /// Arweave gateway URL for download/status (reads don't need relay).
    gateway: String,
    /// HTTP transport implementation.
    transport: Box<dyn HttpTransport>,
    /// Optional API key for relay authentication.
    api_key: Option<String>,
}

impl RelayClient {
    /// Default relay URL used when no override is provided.
    pub const DEFAULT_RELAY_URL: &'static str = "https://relay.obelysk.xyz";

    /// Create a new relay client.
    pub fn new(relay_url: impl Into<String>, transport: Box<dyn HttpTransport>) -> Self {
        Self {
            relay_url: relay_url.into(),
            gateway: "https://arweave.net".to_string(),
            transport,
            api_key: None,
        }
    }

    /// Create with default relay URL from environment or fallback.
    pub fn from_env(transport: Box<dyn HttpTransport>) -> Self {
        let url = std::env::var("OBELYSK_RELAY_URL")
            .unwrap_or_else(|_| Self::DEFAULT_RELAY_URL.to_string());
        Self::new(url, transport)
    }

    /// Set an API key for relay authentication.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Upload data via the relay service.
    pub fn upload(
        &self,
        data: &[u8],
        audit_id: &str,
        model_id: &str,
        extra_tags: &[ArweaveTag],
    ) -> Result<StorageReceipt, AuditError> {
        let url = format!("{}/v1/audit/upload", self.relay_url);

        let data_b64 = base64_encode(data);

        let tags_json: Vec<String> = extra_tags
            .iter()
            .map(|t| {
                format!(
                    "{{\"name\":\"{}\",\"value\":\"{}\"}}",
                    escape_json_string(&t.name),
                    escape_json_string(&t.value)
                )
            })
            .collect();

        let body = format!(
            "{{\"data\":\"{}\",\"audit_id\":\"{}\",\"model_id\":\"{}\",\"tags\":[{}]}}",
            data_b64,
            escape_json_string(audit_id),
            escape_json_string(model_id),
            tags_json.join(",")
        );

        let mut headers: Vec<(&str, &str)> = vec![("Content-Type", "application/json")];
        let api_key_val;
        if let Some(ref key) = self.api_key {
            api_key_val = key.clone();
            headers.push(("X-Api-Key", &api_key_val));
        }

        let response = self.transport.post(&url, &headers, body.as_bytes())?;

        let tx_id = parse_upload_response(&response)?;

        Ok(StorageReceipt {
            tx_id: tx_id.clone(),
            size_bytes: data.len(),
            gateway_url: format!("{}/{}", self.gateway, tx_id),
        })
    }

    /// Download data from Arweave (reads go directly to the gateway, no relay needed).
    pub fn download(&self, tx_id: &str) -> Result<Vec<u8>, AuditError> {
        let url = format!("{}/{}", self.gateway, tx_id);
        self.transport.get(&url)
    }

    /// Check the status of an Arweave transaction.
    pub fn status(&self, tx_id: &str) -> Result<TxStatus, AuditError> {
        let url = format!("{}/v1/audit/{}", self.relay_url, tx_id);
        match self.transport.get(&url) {
            Ok(body) => {
                let text = String::from_utf8_lossy(&body);
                if text.contains("confirmed") {
                    if let Some(height) = extract_block_height(&text) {
                        Ok(TxStatus::Confirmed {
                            block_height: height,
                        })
                    } else {
                        Ok(TxStatus::Confirmed { block_height: 0 })
                    }
                } else if text.contains("pending") {
                    Ok(TxStatus::Pending)
                } else {
                    Ok(TxStatus::NotFound)
                }
            }
            Err(_) => Ok(TxStatus::NotFound),
        }
    }
}

// ─── Marketplace Client ─────────────────────────────────────────────────────

/// Client that routes audit uploads through the BitSage Marketplace.
///
/// The marketplace handles encryption, Arweave upload, and database indexing.
/// Pipeline users get auto-provisioned accounts and can view their proofs
/// in the marketplace dashboard at `/storage`.
///
/// Default marketplace URL: from `MARKETPLACE_URL` env var.
pub struct MarketplaceClient {
    /// Marketplace API URL (e.g., "https://marketplace.bitsage.xyz").
    marketplace_url: String,
    /// API key for marketplace authentication.
    api_key: String,
    /// HTTP transport implementation.
    transport: Box<dyn HttpTransport>,
}

impl MarketplaceClient {
    /// Create a new marketplace client.
    pub fn new(
        marketplace_url: impl Into<String>,
        api_key: impl Into<String>,
        transport: Box<dyn HttpTransport>,
    ) -> Self {
        Self {
            marketplace_url: marketplace_url.into(),
            api_key: api_key.into(),
            transport,
        }
    }

    /// Create from environment variables.
    ///
    /// Reads `MARKETPLACE_URL` and `MARKETPLACE_API_KEY`.
    pub fn from_env(transport: Box<dyn HttpTransport>) -> Option<Self> {
        let url = std::env::var("MARKETPLACE_URL").ok()?;
        let key = std::env::var("MARKETPLACE_API_KEY").ok()?;
        if url.is_empty() || key.is_empty() {
            return None;
        }
        Some(Self::new(url, key, transport))
    }

    /// Upload audit data via the marketplace API.
    ///
    /// The marketplace encrypts the data, uploads to Arweave, and indexes it.
    pub fn upload(
        &self,
        data: &[u8],
        audit_id: &str,
        model_id: &str,
        extra_tags: &[ArweaveTag],
    ) -> Result<StorageReceipt, AuditError> {
        let url = format!("{}/api/v1/pipeline/upload", self.marketplace_url);

        let data_b64 = base64_encode(data);

        // Build tags JSON array
        let tags_json: Vec<String> = extra_tags
            .iter()
            .map(|t| {
                format!(
                    "{{\"name\":\"{}\",\"value\":\"{}\"}}",
                    escape_json_string(&t.name),
                    escape_json_string(&t.value)
                )
            })
            .collect();

        // Extract model name from model_id (or use as-is)
        let model_name = model_id;

        // Build the JSON body matching the marketplace UploadSchema
        let now = chrono_utc_now();
        let body = format!(
            concat!(
                "{{\"data\":\"{}\",",
                "\"auditId\":\"{}\",",
                "\"modelName\":\"{}\",",
                "\"inferenceCount\":0,",
                "\"timeWindowStart\":\"{}\",",
                "\"timeWindowEnd\":\"{}\",",
                "\"tags\":[{}]}}"
            ),
            data_b64,
            escape_json_string(audit_id),
            escape_json_string(model_name),
            now,
            now,
            tags_json.join(",")
        );

        let api_key_header = self.api_key.clone();
        let headers: Vec<(&str, &str)> = vec![
            ("Content-Type", "application/json"),
            ("X-API-Key", &api_key_header),
        ];

        let response = self.transport.post(&url, &headers, body.as_bytes())?;

        // Parse response: {"arweaveTxId": "...", "gatewayUrl": "...", ...}
        let tx_id = parse_marketplace_response(&response)?;

        Ok(StorageReceipt {
            tx_id: tx_id.clone(),
            size_bytes: data.len(),
            gateway_url: format!("https://arweave.net/{}", tx_id),
        })
    }

    /// Download data from Arweave gateway (reads go direct, no marketplace needed).
    pub fn download(&self, tx_id: &str) -> Result<Vec<u8>, AuditError> {
        let url = format!("https://arweave.net/{}", tx_id);
        self.transport.get(&url)
    }

    /// Check status via Arweave gateway.
    pub fn status(&self, tx_id: &str) -> Result<TxStatus, AuditError> {
        let url = format!("https://arweave.net/tx/{}/status", tx_id);
        match self.transport.get(&url) {
            Ok(body) => {
                let text = String::from_utf8_lossy(&body);
                if let Some(height) = extract_block_height(&text) {
                    Ok(TxStatus::Confirmed {
                        block_height: height,
                    })
                } else {
                    Ok(TxStatus::Pending)
                }
            }
            Err(_) => Ok(TxStatus::NotFound),
        }
    }
}

/// Parse marketplace upload response to extract arweaveTxId.
fn parse_marketplace_response(response: &[u8]) -> Result<String, AuditError> {
    let text = String::from_utf8_lossy(response);

    // Look for "arweaveTxId":"..."
    if let Some(start) = text.find("\"arweaveTxId\"") {
        let after = &text[start + 13..];
        if let Some(colon) = after.find(':') {
            let after_colon = after[colon + 1..].trim();
            if after_colon.starts_with('"') {
                if let Some(end) = after_colon[1..].find('"') {
                    return Ok(after_colon[1..1 + end].to_string());
                }
            }
        }
    }

    // Fallback to generic "id" field
    parse_upload_response(response)
}

/// Generate a UTC ISO 8601 timestamp without pulling in chrono.
fn chrono_utc_now() -> String {
    // Use std::time to compute a reasonable timestamp
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Convert to ISO 8601 (simplified — good enough for API calls)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Compute year/month/day from days since epoch (1970-01-01)
    let (year, month, day) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 0u64;
    for (i, &md) in month_days.iter().enumerate() {
        if days < md {
            month = (i + 1) as u64;
            break;
        }
        days -= md;
    }
    if month == 0 {
        month = 12;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Create the appropriate storage client based on available credentials.
///
/// Priority chain:
/// 1. `IRYS_TOKEN` set → direct `ArweaveClient` (user's own token)
/// 2. `MARKETPLACE_URL` + `MARKETPLACE_API_KEY` set → `MarketplaceClient` (via marketplace)
/// 3. Otherwise → `RelayClient` (proxied through coordinator EC2)
///
/// Returns `(uploader, backend_name)`.
#[cfg(feature = "audit-http")]
pub fn create_audit_storage() -> (Box<dyn AuditUploader>, &'static str) {
    // 1. Direct Irys upload (user has their own token)
    if let Ok(token) = std::env::var("IRYS_TOKEN") {
        if !token.is_empty() {
            let transport = Box::new(UreqTransport);
            let client = ArweaveClient::with_defaults(transport).with_auth(token);
            return (Box::new(ArweaveUploader(client)), "irys-direct");
        }
    }

    // 2. Marketplace (auto-provisioned, handles encryption + Arweave + indexing)
    {
        let transport = Box::new(UreqTransport);
        if let Some(client) = MarketplaceClient::from_env(transport) {
            return (Box::new(MarketplaceUploader(client)), "marketplace");
        }
    }

    // 3. Relay fallback (coordinator EC2 holds the token)
    let transport = Box::new(UreqTransport);
    let relay = RelayClient::from_env(transport);
    (Box::new(RelayUploader(relay)), "relay")
}

/// Trait for abstracting upload behavior (direct vs relay).
pub trait AuditUploader: Send + Sync {
    fn upload(
        &self,
        data: &[u8],
        audit_id: &str,
        model_id: &str,
        extra_tags: &[ArweaveTag],
    ) -> Result<StorageReceipt, AuditError>;

    fn download(&self, tx_id: &str) -> Result<Vec<u8>, AuditError>;

    fn status(&self, tx_id: &str) -> Result<TxStatus, AuditError>;
}

#[cfg(feature = "audit-http")]
struct ArweaveUploader(ArweaveClient);

#[cfg(feature = "audit-http")]
impl AuditUploader for ArweaveUploader {
    fn upload(
        &self,
        data: &[u8],
        audit_id: &str,
        model_id: &str,
        extra_tags: &[ArweaveTag],
    ) -> Result<StorageReceipt, AuditError> {
        self.0.upload(data, audit_id, model_id, extra_tags)
    }
    fn download(&self, tx_id: &str) -> Result<Vec<u8>, AuditError> {
        self.0.download(tx_id)
    }
    fn status(&self, tx_id: &str) -> Result<TxStatus, AuditError> {
        self.0.status(tx_id)
    }
}

#[cfg(feature = "audit-http")]
struct RelayUploader(RelayClient);

#[cfg(feature = "audit-http")]
impl AuditUploader for RelayUploader {
    fn upload(
        &self,
        data: &[u8],
        audit_id: &str,
        model_id: &str,
        extra_tags: &[ArweaveTag],
    ) -> Result<StorageReceipt, AuditError> {
        self.0.upload(data, audit_id, model_id, extra_tags)
    }
    fn download(&self, tx_id: &str) -> Result<Vec<u8>, AuditError> {
        self.0.download(tx_id)
    }
    fn status(&self, tx_id: &str) -> Result<TxStatus, AuditError> {
        self.0.status(tx_id)
    }
}

#[cfg(feature = "audit-http")]
struct MarketplaceUploader(MarketplaceClient);

#[cfg(feature = "audit-http")]
impl AuditUploader for MarketplaceUploader {
    fn upload(
        &self,
        data: &[u8],
        audit_id: &str,
        model_id: &str,
        extra_tags: &[ArweaveTag],
    ) -> Result<StorageReceipt, AuditError> {
        self.0.upload(data, audit_id, model_id, extra_tags)
    }
    fn download(&self, tx_id: &str) -> Result<Vec<u8>, AuditError> {
        self.0.download(tx_id)
    }
    fn status(&self, tx_id: &str) -> Result<TxStatus, AuditError> {
        self.0.status(tx_id)
    }
}

// ─── Ureq Transport (real HTTP, feature-gated) ──────────────────────────────

#[cfg(feature = "audit-http")]
pub struct UreqTransport;

#[cfg(feature = "audit-http")]
impl HttpTransport for UreqTransport {
    fn post(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Vec<u8>, AuditError> {
        let mut req = ureq::post(url);
        for &(k, v) in headers {
            req = req.header(k, v);
        }
        let mut resp = req
            .send(body)
            .map_err(|e| AuditError::Storage(format!("POST {}: {}", url, e)))?;
        resp.body_mut()
            .read_to_vec()
            .map_err(|e| AuditError::Storage(format!("read body: {}", e)))
    }

    fn get(&self, url: &str) -> Result<Vec<u8>, AuditError> {
        let mut resp = ureq::get(url)
            .call()
            .map_err(|e| AuditError::Storage(format!("GET {}: {}", url, e)))?;
        resp.body_mut()
            .read_to_vec()
            .map_err(|e| AuditError::Storage(format!("read body: {}", e)))
    }
}

// ─── Mock Transport (for tests) ─────────────────────────────────────────────

/// In-memory mock transport for testing.
pub struct MockTransport {
    /// Stored data keyed by tx_id.
    store: std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>,
    /// Counter for generating tx IDs.
    counter: std::sync::atomic::AtomicU64,
    /// Last headers received by POST (for test assertions).
    last_headers: std::sync::Mutex<Vec<(String, String)>>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self {
            store: std::sync::Mutex::new(std::collections::HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(1),
            last_headers: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Return the headers from the last POST request.
    pub fn last_post_headers(&self) -> Vec<(String, String)> {
        self.last_headers.lock().unwrap().clone()
    }
}

impl HttpTransport for MockTransport {
    fn post(
        &self,
        _url: &str,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Vec<u8>, AuditError> {
        *self.last_headers.lock().unwrap() = headers
            .iter()
            .map(|&(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let id = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let tx_id = format!("mock_tx_{}", id);

        // Extract data from JSON envelope if present, otherwise store raw body.
        let data_to_store = extract_data_from_json_envelope(body).unwrap_or_else(|| body.to_vec());

        self.store
            .lock()
            .unwrap()
            .insert(tx_id.clone(), data_to_store);

        Ok(format!("{{\"id\":\"{}\"}}", tx_id).into_bytes())
    }

    fn get(&self, url: &str) -> Result<Vec<u8>, AuditError> {
        // Extract tx_id from URL (last path segment).
        let tx_id = url.rsplit('/').next().unwrap_or("");

        if tx_id == "status" {
            // Status endpoint — always return confirmed for mock.
            return Ok(b"{\"block_height\": 12345}".to_vec());
        }

        self.store
            .lock()
            .unwrap()
            .get(tx_id)
            .cloned()
            .ok_or_else(|| AuditError::Storage(format!("Not found: {}", tx_id)))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upload_and_download() {
        let transport = Box::new(MockTransport::new());
        let client = ArweaveClient::with_defaults(transport);

        let data = b"Hello, Arweave!";
        let receipt = client.upload(data, "audit_0x1", "model_0x2", &[]).unwrap();

        assert!(!receipt.tx_id.is_empty());
        assert_eq!(receipt.size_bytes, 15);
        assert!(receipt.gateway_url.contains(&receipt.tx_id));

        let downloaded = client.download(&receipt.tx_id).unwrap();
        assert_eq!(downloaded, data);
    }

    #[test]
    fn test_upload_returns_valid_receipt() {
        let transport = Box::new(MockTransport::new());
        let client = ArweaveClient::with_defaults(transport);

        let data = vec![0u8; 1024];
        let receipt = client.upload(&data, "audit_0x1", "model_0x2", &[]).unwrap();

        assert!(receipt.tx_id.starts_with("mock_tx_"));
        assert_eq!(receipt.size_bytes, 1024);
    }

    #[test]
    fn test_download_not_found() {
        let transport = Box::new(MockTransport::new());
        let client = ArweaveClient::with_defaults(transport);

        let result = client.download("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_status_confirmed() {
        let transport = Box::new(MockTransport::new());
        let client = ArweaveClient::with_defaults(transport);

        let data = b"test";
        let receipt = client.upload(data, "a", "m", &[]).unwrap();

        // Mock always returns confirmed.
        let status = client.status(&receipt.tx_id).unwrap();
        match status {
            TxStatus::Confirmed { block_height } => assert_eq!(block_height, 12345),
            _ => panic!("Expected Confirmed"),
        }
    }

    #[test]
    fn test_extra_tags() {
        let transport = Box::new(MockTransport::new());
        let client = ArweaveClient::with_defaults(transport);

        let extra = vec![ArweaveTag {
            name: "Privacy-Tier".to_string(),
            value: "encrypted".to_string(),
        }];
        let receipt = client.upload(b"data", "a", "m", &extra).unwrap();
        assert!(!receipt.tx_id.is_empty());
    }

    #[test]
    fn test_parse_upload_response_json() {
        let resp = br#"{"id": "bIj9E1osXwY8"}"#;
        assert_eq!(parse_upload_response(resp).unwrap(), "bIj9E1osXwY8");
    }

    #[test]
    fn test_parse_upload_response_plain() {
        let resp = b"abcdef123456";
        assert_eq!(parse_upload_response(resp).unwrap(), "abcdef123456");
    }

    #[test]
    fn test_base64_roundtrip() {
        let data = b"Hello, Arweave!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64_roundtrip_binary() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = base64_encode(&data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_json_envelope_extraction() {
        let data = b"test data";
        let b64 = base64_encode(data);
        let json = format!("{{\"data\":\"{}\",\"tags\":[]}}", b64);
        let extracted = extract_data_from_json_envelope(json.as_bytes()).unwrap();
        assert_eq!(extracted, data);
    }

    #[test]
    fn test_upload_sends_json_envelope() {
        // Verify the mock correctly decodes the JSON envelope back to original data
        let transport = Box::new(MockTransport::new());
        let client = ArweaveClient::with_defaults(transport);

        let data = b"binary\x00\x01\x02payload";
        let receipt = client.upload(data, "audit_1", "model_1", &[]).unwrap();
        let downloaded = client.download(&receipt.tx_id).unwrap();
        assert_eq!(downloaded, data);
    }

    #[test]
    fn test_with_auth_sets_token() {
        let transport = Box::new(MockTransport::new());
        let client = ArweaveClient::with_defaults(transport).with_auth("my-test-token");
        assert_eq!(client.auth_token.as_deref(), Some("my-test-token"));
    }

    #[test]
    fn test_auth_header_included_in_upload() {
        use std::sync::Arc;
        // Use Arc<MockTransport> so we can inspect headers after upload.
        let mock = Arc::new(MockTransport::new());

        // Wrap in a newtype that delegates HttpTransport.
        struct ArcTransport(Arc<MockTransport>);
        impl HttpTransport for ArcTransport {
            fn post(
                &self,
                url: &str,
                headers: &[(&str, &str)],
                body: &[u8],
            ) -> Result<Vec<u8>, AuditError> {
                self.0.post(url, headers, body)
            }
            fn get(&self, url: &str) -> Result<Vec<u8>, AuditError> {
                self.0.get(url)
            }
        }

        let mock_ref = Arc::clone(&mock);
        let client = ArweaveClient::new(
            "https://arweave.net",
            "https://node1.irys.xyz",
            Box::new(ArcTransport(mock_ref)),
        )
        .with_auth("irys-secret-token");

        client.upload(b"test", "a", "m", &[]).unwrap();

        let headers = mock.last_post_headers();
        let auth = headers.iter().find(|(k, _)| k == "Authorization");
        assert_eq!(auth.unwrap().1, "Bearer irys-secret-token");
    }

    #[test]
    fn test_marketplace_upload_and_download() {
        let transport = Box::new(MockTransport::new());
        let client =
            MarketplaceClient::new("https://marketplace.test", "bsk_test_key_12345", transport);

        let data = b"audit report data";
        let receipt = client.upload(data, "audit_0x42", "qwen3-14b", &[]).unwrap();

        assert!(!receipt.tx_id.is_empty());
        assert_eq!(receipt.size_bytes, data.len());

        let downloaded = client.download(&receipt.tx_id).unwrap();
        assert_eq!(downloaded, data);
    }

    #[test]
    fn test_marketplace_sends_api_key_header() {
        use std::sync::Arc;
        let mock = Arc::new(MockTransport::new());

        struct ArcTransport(Arc<MockTransport>);
        impl HttpTransport for ArcTransport {
            fn post(
                &self,
                url: &str,
                headers: &[(&str, &str)],
                body: &[u8],
            ) -> Result<Vec<u8>, AuditError> {
                self.0.post(url, headers, body)
            }
            fn get(&self, url: &str) -> Result<Vec<u8>, AuditError> {
                self.0.get(url)
            }
        }

        let mock_ref = Arc::clone(&mock);
        let client = MarketplaceClient::new(
            "https://marketplace.test",
            "bsk_my_secret_key",
            Box::new(ArcTransport(mock_ref)),
        );

        client.upload(b"test", "a", "m", &[]).unwrap();

        let headers = mock.last_post_headers();
        let api_key = headers.iter().find(|(k, _)| k == "X-API-Key");
        assert_eq!(api_key.unwrap().1, "bsk_my_secret_key");
    }

    #[test]
    fn test_chrono_utc_now_format() {
        let ts = chrono_utc_now();
        // Should match ISO 8601: YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(ts.len(), 20);
        assert!(ts.ends_with('Z'));
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[7..8], "-");
        assert_eq!(&ts[10..11], "T");
    }

    #[test]
    fn test_parse_marketplace_response() {
        let resp = br#"{"status":"created","arweaveTxId":"abc123","reportHash":"deadbeef"}"#;
        assert_eq!(parse_marketplace_response(resp).unwrap(), "abc123");
    }

    #[test]
    fn test_no_auth_header_without_token() {
        use std::sync::Arc;
        let mock = Arc::new(MockTransport::new());

        struct ArcTransport(Arc<MockTransport>);
        impl HttpTransport for ArcTransport {
            fn post(
                &self,
                url: &str,
                headers: &[(&str, &str)],
                body: &[u8],
            ) -> Result<Vec<u8>, AuditError> {
                self.0.post(url, headers, body)
            }
            fn get(&self, url: &str) -> Result<Vec<u8>, AuditError> {
                self.0.get(url)
            }
        }

        let mock_ref = Arc::clone(&mock);
        let client = ArweaveClient::new(
            "https://arweave.net",
            "https://node1.irys.xyz",
            Box::new(ArcTransport(mock_ref)),
        );
        // No .with_auth()

        client.upload(b"test", "a", "m", &[]).unwrap();

        let headers = mock.last_post_headers();
        let auth = headers.iter().find(|(k, _)| k == "Authorization");
        assert!(
            auth.is_none(),
            "No Authorization header expected without auth token"
        );
    }
}
