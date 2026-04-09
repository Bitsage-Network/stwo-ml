//! TLS Attestation — cryptographic proof of API interactions.
//!
//! Proves that a specific HTTPS request was sent to a specific server
//! and a specific response was received. Two trust levels:
//!
//! - **Level 1 (Commitment)**: Proxy captures request/response, verifies TLS
//!   certificate chain, computes Poseidon commitment. Trusted proxy model.
//!
//! - **Level 2 (TLS Notary)**: Full MPC-based TLS transcript proof.
//!   No trusted proxy — the TLS session itself is the proof. (Future)
//!
//! Both levels produce an `TlsAttestation` that can be verified on-chain.

use starknet_ff::FieldElement;

/// TLS attestation record — proves an API interaction occurred.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TlsAttestation {
    /// Unique attestation identifier.
    pub attestation_id: String,
    /// The server domain (e.g., "api.anthropic.com").
    pub server_domain: String,
    /// SHA-256 fingerprint of the server's TLS certificate.
    pub cert_fingerprint: String,
    /// HTTP method + path (e.g., "POST /v1/messages").
    pub request_method_path: String,
    /// Poseidon hash of the full request body.
    pub request_body_hash: FieldElement,
    /// Poseidon hash of the full response body.
    pub response_body_hash: FieldElement,
    /// Combined IO commitment: H(request || response || cert || timestamp).
    pub io_commitment: FieldElement,
    /// HTTP response status code.
    pub status_code: u16,
    /// Unix timestamp (seconds) when the request was made.
    pub timestamp: u64,
    /// Attestation level.
    pub level: AttestationLevel,
    /// The extracted text response (for display).
    pub response_text: String,
    /// Provider name (e.g., "anthropic", "openai", "google").
    pub provider: String,
    /// Model used (e.g., "claude-sonnet-4-20250514").
    pub model: String,
}

/// Trust level of the attestation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AttestationLevel {
    /// Proxy-verified: the ObelyZK proxy captured and committed the exchange.
    /// Trust assumption: the proxy is honest.
    ProxyCommitment,
    /// TLS Notary: MPC-based proof of the TLS transcript.
    /// No trust assumption beyond TLS itself.
    TlsNotary,
}

/// Compute the IO commitment for a TLS-attested API call.
///
/// `H(domain_tag, server_domain_hash, request_hash, response_hash, cert_hash, timestamp)`
pub fn compute_tls_io_commitment(
    server_domain: &str,
    request_body: &[u8],
    response_body: &[u8],
    cert_fingerprint: &str,
    timestamp: u64,
) -> FieldElement {
    let domain_hash = hash_string(server_domain);
    let request_hash = hash_bytes(request_body);
    let response_hash = hash_bytes(response_body);
    let cert_hash = hash_string(cert_fingerprint);

    starknet_crypto::poseidon_hash_many(&[
        FieldElement::from(0x544C5341u64), // "TLSA" domain tag
        domain_hash,
        request_hash,
        response_hash,
        cert_hash,
        FieldElement::from(timestamp),
    ])
}

/// Create an attestation from a captured HTTPS exchange.
///
/// The caller (proxy) provides the raw request/response bodies and the
/// TLS certificate fingerprint. This function computes all commitments.
pub fn create_proxy_attestation(
    attestation_id: &str,
    server_domain: &str,
    request_method_path: &str,
    request_body: &[u8],
    response_body: &[u8],
    response_text: &str,
    cert_fingerprint: &str,
    status_code: u16,
    provider: &str,
    model: &str,
) -> TlsAttestation {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let io_commitment = compute_tls_io_commitment(
        server_domain,
        request_body,
        response_body,
        cert_fingerprint,
        timestamp,
    );

    TlsAttestation {
        attestation_id: attestation_id.to_string(),
        server_domain: server_domain.to_string(),
        cert_fingerprint: cert_fingerprint.to_string(),
        request_method_path: request_method_path.to_string(),
        request_body_hash: hash_bytes(request_body),
        response_body_hash: hash_bytes(response_body),
        io_commitment,
        status_code,
        timestamp,
        level: AttestationLevel::ProxyCommitment,
        response_text: response_text.to_string(),
        provider: provider.to_string(),
        model: model.to_string(),
    }
}

/// Hash a byte slice into a FieldElement via Poseidon.
fn hash_bytes(data: &[u8]) -> FieldElement {
    let mut felts = vec![FieldElement::from(data.len() as u64)];
    for chunk in data.chunks(31) {
        let mut buf = [0u8; 32];
        buf[32 - chunk.len()..].copy_from_slice(chunk);
        felts.push(FieldElement::from_bytes_be(&buf).unwrap_or(FieldElement::ZERO));
    }
    starknet_crypto::poseidon_hash_many(&felts)
}

/// Hash a string into a FieldElement.
fn hash_string(s: &str) -> FieldElement {
    hash_bytes(s.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls_attestation_commitment() {
        let att = create_proxy_attestation(
            "att-test-001",
            "api.anthropic.com",
            "POST /v1/messages",
            b"{\"model\":\"claude-sonnet\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}",
            b"{\"content\":[{\"text\":\"Hello! How can I help?\"}]}",
            "Hello! How can I help?",
            "sha256:abcdef1234567890",
            200,
            "anthropic",
            "claude-sonnet-4-20250514",
        );

        assert_eq!(att.server_domain, "api.anthropic.com");
        assert_eq!(att.level, AttestationLevel::ProxyCommitment);
        assert_ne!(att.io_commitment, FieldElement::ZERO);
        assert_ne!(att.request_body_hash, att.response_body_hash);
    }

    #[test]
    fn test_deterministic_commitment() {
        let c1 = compute_tls_io_commitment(
            "api.openai.com", b"request", b"response", "cert123", 1000,
        );
        let c2 = compute_tls_io_commitment(
            "api.openai.com", b"request", b"response", "cert123", 1000,
        );
        assert_eq!(c1, c2);

        // Different response → different commitment
        let c3 = compute_tls_io_commitment(
            "api.openai.com", b"request", b"different", "cert123", 1000,
        );
        assert_ne!(c1, c3);
    }
}
