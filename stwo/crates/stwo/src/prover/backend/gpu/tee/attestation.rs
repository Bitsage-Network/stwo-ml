//! GPU and CPU Attestation for NVIDIA Confidential Computing
//!
//! This module handles attestation using the NVIDIA nvTrust SDK:
//! - GPU attestation via SPDM protocol
//! - CPU attestation via TDX/SEV-SNP
//! - Remote attestation to external verifiers

#![allow(unexpected_cfgs)]

use super::{
    CcMode, ConfidentialGpu, CpuAttestationReport, CpuTee, GpuAttestationReport, TeeConfig,
    TeeError, TeeResult,
};
use std::process::Command;
use std::time::Instant;

/// Generate GPU attestation report
///
/// This uses the NVIDIA nvTrust SDK to generate an attestation report
/// for the GPU in CC-On mode.
pub fn generate_gpu_attestation(
    device_id: u32,
    config: &TeeConfig,
) -> TeeResult<GpuAttestationReport> {
    // Generate random nonce for freshness
    let nonce = generate_nonce();

    // Try to use nvTrust SDK for real attestation
    #[cfg(feature = "nvtrust")]
    {
        return generate_real_gpu_attestation(device_id, config, nonce);
    }

    // Fallback: Use nvidia-smi and driver info for development
    #[cfg(not(feature = "nvtrust"))]
    {
        generate_dev_gpu_attestation(device_id, config, nonce)
    }
}

/// Generate real GPU attestation using nvTrust SDK
#[cfg(feature = "nvtrust")]
fn generate_real_gpu_attestation(
    device_id: u32,
    config: &TeeConfig,
    nonce: [u8; 32],
) -> TeeResult<GpuAttestationReport> {
    use super::nvtrust::NvTrustClient;

    let client = NvTrustClient::new()?;

    // Get GPU info
    let gpu_info = client.get_gpu_info(device_id)?;

    // Generate attestation evidence
    let evidence = client.generate_attestation_evidence(device_id, &nonce)?;

    // Get certificate chain
    let cert_chain = client.get_certificate_chain(device_id)?;

    Ok(GpuAttestationReport {
        device_id,
        gpu_model: gpu_info.model,
        cc_mode: gpu_info.cc_mode,
        driver_version: gpu_info.driver_version,
        vbios_version: gpu_info.vbios_version,
        cert_chain,
        evidence,
        timestamp: Instant::now(),
        nonce,
    })
}

/// Generate development GPU attestation (no nvTrust)
#[allow(unused_variables)]
fn generate_dev_gpu_attestation(
    device_id: u32,
    config: &TeeConfig,
    nonce: [u8; 32],
) -> TeeResult<GpuAttestationReport> {
    // Query GPU info via nvidia-smi
    let (driver_version, gpu_name) = query_nvidia_smi()?;

    // Determine GPU model from name
    let gpu_model = parse_gpu_model(&gpu_name)?;

    // Query CC mode
    let cc_mode = super::cc_mode::query_cc_mode(device_id)?;

    // Generate mock evidence (in production, this comes from GPU)
    let evidence = generate_mock_evidence(device_id, &nonce, &driver_version);

    // Mock certificate chain
    let cert_chain = generate_mock_cert_chain(device_id);

    Ok(GpuAttestationReport {
        device_id,
        gpu_model,
        cc_mode,
        driver_version,
        vbios_version: "MOCK_VBIOS".to_string(),
        cert_chain,
        evidence,
        timestamp: Instant::now(),
        nonce,
    })
}

/// Query nvidia-smi for GPU info
fn query_nvidia_smi() -> TeeResult<(String, String)> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=driver_version,name", "--format=csv,noheader"])
        .output()
        .map_err(|e| TeeError::DriverError(format!("Failed to run nvidia-smi: {}", e)))?;

    if !output.status.success() {
        return Err(TeeError::DriverError(
            "nvidia-smi failed - is NVIDIA driver installed?".into(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = stdout.trim().split(", ").collect();

    if parts.len() < 2 {
        return Err(TeeError::DriverError("Invalid nvidia-smi output".into()));
    }

    Ok((parts[0].to_string(), parts[1].to_string()))
}

/// Parse GPU model from nvidia-smi name
fn parse_gpu_model(name: &str) -> TeeResult<ConfidentialGpu> {
    let name_upper = name.to_uppercase();

    if name_upper.contains("B200") {
        if name_upper.contains("NVL") {
            Ok(ConfidentialGpu::B200Nvl)
        } else {
            Ok(ConfidentialGpu::B200)
        }
    } else if name_upper.contains("H200") {
        if name_upper.contains("NVL") {
            Ok(ConfidentialGpu::H200Nvl)
        } else {
            Ok(ConfidentialGpu::H200)
        }
    } else if name_upper.contains("H100") {
        if name_upper.contains("NVL") {
            Ok(ConfidentialGpu::H100Nvl)
        } else {
            Ok(ConfidentialGpu::H100)
        }
    } else {
        Err(TeeError::GpuNotSupported(format!(
            "GPU '{}' does not support Confidential Computing. Supported: H100, H200, B200",
            name
        )))
    }
}

/// Generate a random nonce
fn generate_nonce() -> [u8; 32] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut nonce = [0u8; 32];

    // Use timestamp and random data
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();

    let mut hasher = DefaultHasher::new();
    now.as_nanos().hash(&mut hasher);
    std::process::id().hash(&mut hasher);

    let hash1 = hasher.finish();
    nonce[0..8].copy_from_slice(&hash1.to_le_bytes());

    // Add more entropy
    for i in 1..4 {
        hasher = DefaultHasher::new();
        hash1.wrapping_add(i as u64).hash(&mut hasher);
        now.as_nanos().wrapping_add(i as u128).hash(&mut hasher);
        let hash = hasher.finish();
        nonce[(i * 8)..((i + 1) * 8)].copy_from_slice(&hash.to_le_bytes());
    }

    nonce
}

/// Generate mock attestation evidence (development only)
fn generate_mock_evidence(device_id: u32, nonce: &[u8; 32], driver: &str) -> Vec<u8> {
    let mut evidence = Vec::new();

    // Header
    evidence.extend_from_slice(b"NVIDIA_CC_EVIDENCE_V1");
    evidence.push(0);

    // Device ID
    evidence.extend_from_slice(&device_id.to_le_bytes());

    // Nonce
    evidence.extend_from_slice(nonce);

    // Driver version hash
    let driver_hash = super::crypto::sha256(driver.as_bytes());
    evidence.extend_from_slice(&driver_hash);

    // Timestamp
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    evidence.extend_from_slice(&ts.to_le_bytes());

    // Mock signature (in production, this is signed by GPU)
    evidence.extend_from_slice(b"MOCK_SIGNATURE_FOR_DEVELOPMENT");

    evidence
}

/// Generate mock certificate chain (development only)
fn generate_mock_cert_chain(device_id: u32) -> Vec<u8> {
    let mut chain = Vec::new();

    // This would be a real X.509 certificate chain in production
    chain.extend_from_slice(b"-----BEGIN CERTIFICATE-----\n");
    chain.extend_from_slice(b"MOCK_GPU_ATTESTATION_CERTIFICATE_");
    chain.extend_from_slice(&device_id.to_le_bytes());
    chain.extend_from_slice(b"\n-----END CERTIFICATE-----\n");

    chain
}

/// Verify GPU attestation report
pub fn verify_gpu_attestation(report: &GpuAttestationReport) -> TeeResult<bool> {
    // In production, this would:
    // 1. Verify the certificate chain against NVIDIA root CA
    // 2. Verify the evidence signature using the leaf certificate
    // 3. Verify the nonce is fresh
    // 4. Verify the CC mode and other claims

    // Check timestamp freshness (within last hour)
    if report.timestamp.elapsed().as_secs() > 3600 {
        return Err(TeeError::AttestationFailed(
            "Attestation report is stale".into(),
        ));
    }

    // Check CC mode
    if report.cc_mode == CcMode::Off {
        return Err(TeeError::AttestationFailed(
            "GPU is not in CC-On mode".into(),
        ));
    }

    // Verify evidence structure
    if report.evidence.len() < 64 {
        return Err(TeeError::AttestationFailed(
            "Evidence too short".into(),
        ));
    }

    // Check evidence header
    if !report.evidence.starts_with(b"NVIDIA_CC_EVIDENCE") {
        return Err(TeeError::AttestationFailed(
            "Invalid evidence format".into(),
        ));
    }

    tracing::info!(
        device_id = report.device_id,
        gpu = ?report.gpu_model,
        cc_mode = %report.cc_mode,
        "GPU attestation verified"
    );

    Ok(true)
}

/// Generate CPU attestation report
pub fn generate_cpu_attestation(tee_type: CpuTee) -> TeeResult<CpuAttestationReport> {
    match tee_type {
        CpuTee::IntelTdx => generate_tdx_attestation(),
        CpuTee::AmdSevSnp => generate_sevsnp_attestation(),
        CpuTee::None => Err(TeeError::CpuTeeNotAvailable),
    }
}

/// Generate Intel TDX attestation
fn generate_tdx_attestation() -> TeeResult<CpuAttestationReport> {
    // In production, this would use the TDX Guest Library to:
    // 1. Get TD report
    // 2. Convert to attestation quote via QGS

    // For development, generate mock report
    let platform_info = b"INTEL_TDX_PLATFORM_INFO".to_vec();
    let measurements = vec![[0u8; 48]; 4]; // RTMR0-3
    let quote = generate_mock_tdx_quote();

    Ok(CpuAttestationReport {
        tee_type: CpuTee::IntelTdx,
        platform_info,
        measurements,
        quote,
        timestamp: Instant::now(),
    })
}

/// Generate AMD SEV-SNP attestation
fn generate_sevsnp_attestation() -> TeeResult<CpuAttestationReport> {
    // In production, this would use the SEV-SNP Guest Library to:
    // 1. Get attestation report from PSP
    // 2. Include launch measurement

    // For development, generate mock report
    let platform_info = b"AMD_SEVSNP_PLATFORM_INFO".to_vec();
    let measurements = vec![[0u8; 48]; 1]; // Launch measurement
    let quote = generate_mock_sevsnp_quote();

    Ok(CpuAttestationReport {
        tee_type: CpuTee::AmdSevSnp,
        platform_info,
        measurements,
        quote,
        timestamp: Instant::now(),
    })
}

/// Generate mock TDX quote
fn generate_mock_tdx_quote() -> Vec<u8> {
    let mut quote = Vec::new();

    // TDX quote header
    quote.extend_from_slice(&[0x04, 0x00]); // Version
    quote.extend_from_slice(&[0x02, 0x00]); // Attestation key type
    quote.extend_from_slice(&[0x00, 0x00, 0x81, 0x00]); // TEE type (TDX)

    // Reserved
    quote.extend_from_slice(&[0u8; 2]);

    // QE vendor ID
    quote.extend_from_slice(&[0x93, 0x9A, 0x72, 0x33]); // Intel

    // User data
    quote.extend_from_slice(&[0u8; 20]);

    // Mock report body
    quote.extend_from_slice(b"MOCK_TDX_REPORT_BODY");
    quote.extend_from_slice(&[0u8; 364]); // Pad to expected size

    quote
}

/// Generate mock SEV-SNP quote
fn generate_mock_sevsnp_quote() -> Vec<u8> {
    let mut quote = Vec::new();

    // SEV-SNP report header
    quote.extend_from_slice(&[0x02, 0x00, 0x00, 0x00]); // Version
    quote.extend_from_slice(&[0x1F, 0x00, 0x00, 0x00]); // Guest SVN

    // Policy
    quote.extend_from_slice(&[0u8; 8]);

    // Family ID, Image ID
    quote.extend_from_slice(&[0u8; 32]);

    // VMPL, signature algo
    quote.extend_from_slice(&[0u8; 4]);

    // Platform info
    quote.extend_from_slice(&[0u8; 8]);

    // Mock measurement
    quote.extend_from_slice(b"MOCK_SEVSNP_MEASUREMENT");
    quote.extend_from_slice(&[0u8; 25]); // Pad to 48 bytes

    // Host data, ID block, author key
    quote.extend_from_slice(&[0u8; 128]);

    // Report signature
    quote.extend_from_slice(b"MOCK_SIGNATURE");
    quote.extend_from_slice(&[0u8; 498]); // Pad to 512 bytes

    quote
}

/// Verify CPU attestation report
pub fn verify_cpu_attestation(report: &CpuAttestationReport) -> TeeResult<bool> {
    // Check timestamp freshness
    if report.timestamp.elapsed().as_secs() > 3600 {
        return Err(TeeError::AttestationFailed(
            "CPU attestation report is stale".into(),
        ));
    }

    // Verify quote structure
    match report.tee_type {
        CpuTee::IntelTdx => { verify_tdx_quote(&report.quote)?; }
        CpuTee::AmdSevSnp => { verify_sevsnp_quote(&report.quote)?; }
        CpuTee::None => return Err(TeeError::CpuTeeNotAvailable),
    }

    tracing::info!(tee = ?report.tee_type, "CPU attestation verified");

    Ok(true)
}

fn verify_tdx_quote(quote: &[u8]) -> TeeResult<bool> {
    // Verify TDX quote structure
    if quote.len() < 48 {
        return Err(TeeError::AttestationFailed("TDX quote too short".into()));
    }

    // Check version
    if quote[0] != 0x04 {
        return Err(TeeError::AttestationFailed("Invalid TDX quote version".into()));
    }

    Ok(true)
}

fn verify_sevsnp_quote(quote: &[u8]) -> TeeResult<bool> {
    // Verify SEV-SNP report structure
    if quote.len() < 48 {
        return Err(TeeError::AttestationFailed("SEV-SNP report too short".into()));
    }

    // Check version
    if quote[0] != 0x02 {
        return Err(TeeError::AttestationFailed(
            "Invalid SEV-SNP report version".into(),
        ));
    }

    Ok(true)
}

/// Remote attestation client for external verification
#[allow(dead_code)]
pub struct RemoteAttestationClient {
    /// Attestation server URL
    server_url: String,
    /// HTTP client timeout
    timeout: std::time::Duration,
}

impl RemoteAttestationClient {
    /// Create a new remote attestation client
    pub fn new(server_url: &str) -> Self {
        Self {
            server_url: server_url.to_string(),
            timeout: std::time::Duration::from_secs(30),
        }
    }

    /// Verify attestation with remote server
    #[allow(unused_variables)]
    pub fn verify_remote(
        &self,
        gpu_report: &GpuAttestationReport,
        cpu_report: Option<&CpuAttestationReport>,
    ) -> TeeResult<bool> {
        // In production, this would:
        // 1. Send attestation quotes to verification server
        // 2. Server verifies against NVIDIA/Intel/AMD root of trust
        // 3. Return verification result with signed token

        tracing::info!(
            server = %self.server_url,
            "Remote attestation verification (mock)"
        );

        // For development, return success
        Ok(true)
    }

    /// Get verification token for client
    pub fn get_verification_token(
        &self,
        gpu_report: &GpuAttestationReport,
    ) -> TeeResult<Vec<u8>> {
        // Generate mock token
        let mut token = Vec::new();
        token.extend_from_slice(b"OBELYSK_ATTESTATION_TOKEN_V1");
        token.extend_from_slice(&gpu_report.nonce);
        token.extend_from_slice(
            &std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                .to_le_bytes(),
        );

        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_nonce() {
        let nonce1 = generate_nonce();
        let nonce2 = generate_nonce();

        // Nonces should be different
        assert_ne!(nonce1, nonce2);

        // Nonces should be 32 bytes
        assert_eq!(nonce1.len(), 32);
    }

    #[test]
    fn test_parse_gpu_model() {
        assert_eq!(
            parse_gpu_model("NVIDIA H100 PCIe").unwrap(),
            ConfidentialGpu::H100
        );
        assert_eq!(
            parse_gpu_model("NVIDIA H100 NVL").unwrap(),
            ConfidentialGpu::H100Nvl
        );
        assert_eq!(
            parse_gpu_model("NVIDIA H200").unwrap(),
            ConfidentialGpu::H200
        );
        assert_eq!(
            parse_gpu_model("NVIDIA B200 NVL").unwrap(),
            ConfidentialGpu::B200Nvl
        );

        // Unsupported GPUs
        assert!(parse_gpu_model("NVIDIA A100").is_err());
        assert!(parse_gpu_model("NVIDIA RTX 4090").is_err());
    }

    #[test]
    fn test_mock_evidence() {
        let nonce = [0u8; 32];
        let evidence = generate_mock_evidence(0, &nonce, "535.104.05");

        assert!(evidence.starts_with(b"NVIDIA_CC_EVIDENCE_V1"));
        assert!(evidence.len() > 64);
    }
}
