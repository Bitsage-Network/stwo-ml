//! NVIDIA nvTrust SDK Integration
//!
//! This module provides integration with the NVIDIA nvTrust SDK for
//! GPU attestation in Confidential Computing environments.
//!
//! # Requirements
//!
//! - NVIDIA H100/H200/B200 GPU in CC-On mode
//! - CUDA 12.4+ with r550+ driver
//! - nvTrust SDK installed (Python or native)
//!
//! # Usage
//!
//! The nvTrust SDK can be used via:
//! 1. Python SDK (`nv-attestation-sdk` package)
//! 2. CLI tools (`nvidia-smi conf-compute`)
//! 3. Direct SPDM communication (advanced)

use super::{CcMode, ConfidentialGpu, TeeError, TeeResult};
use std::process::Command;

/// GPU information from nvTrust
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU model
    pub model: ConfidentialGpu,
    /// CC mode
    pub cc_mode: CcMode,
    /// Driver version
    pub driver_version: String,
    /// VBIOS version
    pub vbios_version: String,
    /// GPU UUID
    pub uuid: String,
    /// PCIe BDF (Bus:Device.Function)
    pub pcie_bdf: String,
}

/// nvTrust client for attestation operations
pub struct NvTrustClient {
    /// Use Python SDK vs CLI
    use_python_sdk: bool,
    /// Python interpreter path
    python_path: String,
}

impl NvTrustClient {
    /// Create a new nvTrust client
    pub fn new() -> TeeResult<Self> {
        // Check if Python SDK is available
        let use_python_sdk = check_python_sdk_available();

        Ok(Self {
            use_python_sdk,
            python_path: "python3".to_string(),
        })
    }

    /// Create with custom Python path
    pub fn with_python(python_path: &str) -> TeeResult<Self> {
        Ok(Self {
            use_python_sdk: true,
            python_path: python_path.to_string(),
        })
    }

    /// Get GPU information
    pub fn get_gpu_info(&self, device_id: u32) -> TeeResult<GpuInfo> {
        if self.use_python_sdk {
            self.get_gpu_info_python(device_id)
        } else {
            self.get_gpu_info_cli(device_id)
        }
    }

    /// Get GPU info via CLI
    fn get_gpu_info_cli(&self, device_id: u32) -> TeeResult<GpuInfo> {
        // Query basic info via nvidia-smi
        let output = Command::new("nvidia-smi")
            .args([
                "-i",
                &device_id.to_string(),
                "--query-gpu=name,driver_version,uuid,pci.bus_id",
                "--format=csv,noheader",
            ])
            .output()
            .map_err(|e| TeeError::NvTrustError(format!("nvidia-smi failed: {}", e)))?;

        if !output.status.success() {
            return Err(TeeError::NvTrustError("nvidia-smi query failed".into()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split(", ").collect();

        if parts.len() < 4 {
            return Err(TeeError::NvTrustError("Invalid nvidia-smi output".into()));
        }

        let model = parse_gpu_model(parts[0])?;
        let cc_mode = super::cc_mode::query_cc_mode(device_id)?;

        Ok(GpuInfo {
            model,
            cc_mode,
            driver_version: parts[1].to_string(),
            vbios_version: "Unknown".to_string(), // Would query separately
            uuid: parts[2].to_string(),
            pcie_bdf: parts[3].to_string(),
        })
    }

    /// Get GPU info via Python SDK
    fn get_gpu_info_python(&self, device_id: u32) -> TeeResult<GpuInfo> {
        let script = format!(
            r#"
import json
try:
    from nv_attestation_sdk import attestation
    client = attestation.AttestationClient()
    gpu_info = client.get_gpu_info({device_id})
    print(json.dumps({{
        'name': gpu_info.get('name', 'Unknown'),
        'driver_version': gpu_info.get('driver_version', 'Unknown'),
        'vbios_version': gpu_info.get('vbios_version', 'Unknown'),
        'uuid': gpu_info.get('uuid', 'Unknown'),
        'pcie_bdf': gpu_info.get('pcie_bdf', 'Unknown'),
        'cc_mode': gpu_info.get('cc_mode', 'off')
    }}, indent=2))
except ImportError:
    # Fallback to nvidia-smi
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '-i', '{device_id}', '--query-gpu=name,driver_version,uuid,pci.bus_id', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    parts = result.stdout.strip().split(', ')
    print(json.dumps({{
        'name': parts[0] if len(parts) > 0 else 'Unknown',
        'driver_version': parts[1] if len(parts) > 1 else 'Unknown',
        'vbios_version': 'Unknown',
        'uuid': parts[2] if len(parts) > 2 else 'Unknown',
        'pcie_bdf': parts[3] if len(parts) > 3 else 'Unknown',
        'cc_mode': 'unknown'
    }}, indent=2))
"#,
            device_id = device_id
        );

        let output = Command::new(&self.python_path)
            .args(["-c", &script])
            .output()
            .map_err(|e| TeeError::NvTrustError(format!("Python failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TeeError::NvTrustError(format!("Python error: {}", stderr)));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Simple JSON parsing without serde_json
        let (name, driver_version, vbios_version, uuid, pcie_bdf, cc_mode_str) =
            parse_simple_json(&stdout);

        let model = parse_gpu_model(&name)?;

        let cc_mode = match cc_mode_str.to_lowercase().as_str() {
            "on" | "enabled" => CcMode::On,
            "devtools" | "dev" => CcMode::DevTools,
            _ => CcMode::Off,
        };

        Ok(GpuInfo {
            model,
            cc_mode,
            driver_version,
            vbios_version,
            uuid,
            pcie_bdf,
        })
    }

    /// Generate attestation evidence for a GPU
    pub fn generate_attestation_evidence(
        &self,
        device_id: u32,
        nonce: &[u8; 32],
    ) -> TeeResult<Vec<u8>> {
        if self.use_python_sdk {
            self.generate_evidence_python(device_id, nonce)
        } else {
            self.generate_evidence_cli(device_id, nonce)
        }
    }

    /// Generate evidence via CLI
    fn generate_evidence_cli(&self, device_id: u32, nonce: &[u8; 32]) -> TeeResult<Vec<u8>> {
        // Use nvidia-smi conf-compute for attestation
        let nonce_hex = hex::encode(nonce);

        let output = Command::new("nvidia-smi")
            .args([
                "conf-compute",
                "-ga", // Generate attestation
                "-i",
                &device_id.to_string(),
                "--nonce",
                &nonce_hex,
            ])
            .output()
            .map_err(|e| TeeError::NvTrustError(format!("Attestation command failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!(
                device_id = device_id,
                stderr = %stderr,
                "GPU attestation command failed — real attestation unavailable"
            );
            return Err(TeeError::NvTrustError(format!(
                "nvidia-smi conf-compute attestation failed for device {}: {}",
                device_id, stderr
            )));
        }

        // Parse attestation report from output
        let stdout = String::from_utf8_lossy(&output.stdout);

        // The attestation report is typically base64 encoded
        for line in stdout.lines() {
            if line.starts_with("Attestation Report:") || line.contains("evidence") {
                if let Some(b64) = line.split(':').nth(1) {
                    if let Ok(evidence) = base64_decode(b64.trim()) {
                        return Ok(evidence);
                    }
                }
            }
        }

        // If we can't parse, return raw output as evidence
        Ok(output.stdout)
    }

    /// Generate evidence via Python SDK
    fn generate_evidence_python(&self, device_id: u32, nonce: &[u8; 32]) -> TeeResult<Vec<u8>> {
        let nonce_hex = hex::encode(nonce);

        let script = format!(
            r#"
import base64
try:
    from nv_attestation_sdk import attestation
    client = attestation.AttestationClient()
    nonce = bytes.fromhex('{nonce_hex}')
    evidence = client.get_gpu_attestation_evidence({device_id}, nonce)
    print(base64.b64encode(evidence).decode())
except Exception as e:
    print(f"ERROR: {{e}}")
"#,
            device_id = device_id,
            nonce_hex = nonce_hex
        );

        let output = Command::new(&self.python_path)
            .args(["-c", &script])
            .output()
            .map_err(|e| TeeError::NvTrustError(format!("Python failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();

        if stdout.starts_with("ERROR:") {
            tracing::error!(
                device_id = device_id,
                error = %stdout,
                "Python attestation SDK failed — real attestation unavailable"
            );
            return Err(TeeError::NvTrustError(format!(
                "Python attestation failed for device {}: {}",
                device_id, stdout
            )));
        }

        base64_decode(&stdout)
            .map_err(|e| TeeError::NvTrustError(format!("Base64 decode failed: {}", e)))
    }

    /// Get certificate chain for a GPU
    pub fn get_certificate_chain(&self, device_id: u32) -> TeeResult<Vec<u8>> {
        if self.use_python_sdk {
            self.get_cert_chain_python(device_id)
        } else {
            self.get_cert_chain_cli(device_id)
        }
    }

    /// Get cert chain via CLI
    fn get_cert_chain_cli(&self, device_id: u32) -> TeeResult<Vec<u8>> {
        let output = Command::new("nvidia-smi")
            .args([
                "conf-compute",
                "-gc", // Get certificate
                "-i",
                &device_id.to_string(),
            ])
            .output()
            .map_err(|e| TeeError::NvTrustError(format!("Get cert failed: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!(
                device_id = device_id,
                stderr = %stderr,
                "Failed to retrieve GPU certificate chain"
            );
            return Err(TeeError::NvTrustError(format!(
                "nvidia-smi conf-compute get-cert failed for device {}: {}",
                device_id, stderr
            )));
        }

        Ok(output.stdout)
    }

    /// Get cert chain via Python SDK
    fn get_cert_chain_python(&self, device_id: u32) -> TeeResult<Vec<u8>> {
        let script = format!(
            r#"
import base64
try:
    from nv_attestation_sdk import attestation
    client = attestation.AttestationClient()
    cert_chain = client.get_gpu_certificate_chain({device_id})
    print(base64.b64encode(cert_chain).decode())
except Exception as e:
    print(f"ERROR: {{e}}")
"#,
            device_id = device_id
        );

        let output = Command::new(&self.python_path)
            .args(["-c", &script])
            .output()
            .map_err(|e| TeeError::NvTrustError(format!("Python failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();

        if stdout.starts_with("ERROR:") {
            tracing::error!(
                device_id = device_id,
                error = %stdout,
                "Python SDK failed to retrieve certificate chain"
            );
            return Err(TeeError::NvTrustError(format!(
                "Python cert chain retrieval failed for device {}: {}",
                device_id, stdout
            )));
        }

        base64_decode(&stdout)
            .map_err(|e| TeeError::NvTrustError(format!("Base64 decode failed: {}", e)))
    }

    /// Verify attestation evidence with NVIDIA's verification service
    pub fn verify_with_nvidia(&self, evidence: &[u8], cert_chain: &[u8]) -> TeeResult<bool> {
        let evidence_b64 = base64_encode(evidence);
        let cert_b64 = base64_encode(cert_chain);

        let script = format!(
            r#"
import base64
try:
    from nv_attestation_sdk import attestation
    verifier = attestation.RemoteVerifier()
    evidence = base64.b64decode('{evidence_b64}')
    cert_chain = base64.b64decode('{cert_b64}')
    result = verifier.verify(evidence, cert_chain)
    print("VERIFIED" if result else "FAILED")
except Exception as e:
    print(f"ERROR: {{e}}")
"#,
            evidence_b64 = evidence_b64,
            cert_b64 = cert_b64
        );

        let output = Command::new(&self.python_path)
            .args(["-c", &script])
            .output()
            .map_err(|e| TeeError::NvTrustError(format!("Python failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();

        match stdout.as_str() {
            "VERIFIED" => Ok(true),
            "FAILED" => Ok(false),
            _ => {
                tracing::error!(result = %stdout, "Unexpected verification response from NVIDIA");
                Err(TeeError::NvTrustError(format!(
                    "Unexpected verification result: {}", stdout
                )))
            }
        }
    }
}

/// Check if Python SDK is available
fn check_python_sdk_available() -> bool {
    let result = Command::new("python3")
        .args(["-c", "from nv_attestation_sdk import attestation; print('ok')"])
        .output();

    matches!(result, Ok(output) if output.status.success())
}

/// Simple JSON parsing for GPU info (avoids serde_json dependency)
fn parse_simple_json(json_str: &str) -> (String, String, String, String, String, String) {
    fn extract_value(json: &str, key: &str) -> String {
        let pattern = format!("\"{}\":", key);
        if let Some(start) = json.find(&pattern) {
            let rest = &json[start + pattern.len()..];
            let rest = rest.trim_start();
            if rest.starts_with('"') {
                let rest = &rest[1..];
                if let Some(end) = rest.find('"') {
                    return rest[..end].to_string();
                }
            }
        }
        "Unknown".to_string()
    }

    (
        extract_value(json_str, "name"),
        extract_value(json_str, "driver_version"),
        extract_value(json_str, "vbios_version"),
        extract_value(json_str, "uuid"),
        extract_value(json_str, "pcie_bdf"),
        extract_value(json_str, "cc_mode"),
    )
}

/// Parse GPU model from name
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
        Err(TeeError::GpuNotSupported(name.to_string()))
    }
}

/// Generate mock evidence for testing only.
#[cfg(test)]
fn generate_mock_evidence(device_id: u32, nonce: &[u8; 32]) -> Vec<u8> {
    let mut evidence = Vec::new();
    evidence.extend_from_slice(b"NVIDIA_MOCK_EVIDENCE_V1\0");
    evidence.extend_from_slice(&device_id.to_le_bytes());
    evidence.extend_from_slice(nonce);
    evidence.extend_from_slice(
        &std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_le_bytes(),
    );
    evidence.extend_from_slice(b"MOCK_SIGNATURE");
    evidence
}

/// Generate mock certificate chain for testing only.
#[cfg(test)]
#[allow(dead_code)]
fn generate_mock_cert_chain() -> Vec<u8> {
    let mut chain = Vec::new();
    chain.extend_from_slice(b"-----BEGIN CERTIFICATE-----\n");
    chain.extend_from_slice(b"MOCK_NVIDIA_GPU_ATTESTATION_CERTIFICATE\n");
    chain.extend_from_slice(b"-----END CERTIFICATE-----\n");
    chain
}

/// Base64 encode
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut result = String::new();
    let chunks = data.chunks(3);

    for chunk in chunks {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        } else {
            result.push('=');
        }
    }

    result
}

/// Base64 decode
fn base64_decode(input: &str) -> Result<Vec<u8>, &'static str> {
    const DECODE_TABLE: [i8; 128] = [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1,
        -1, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4,
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1,
        -1, -1, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
        46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1,
    ];

    let input = input.trim().trim_end_matches('=');
    let mut result = Vec::with_capacity(input.len() * 3 / 4);

    let chars: Vec<u8> = input
        .chars()
        .filter_map(|c| {
            if c.is_ascii() && DECODE_TABLE[c as usize] >= 0 {
                Some(DECODE_TABLE[c as usize] as u8)
            } else {
                None
            }
        })
        .collect();

    for chunk in chars.chunks(4) {
        if chunk.len() >= 2 {
            result.push((chunk[0] << 2) | (chunk[1] >> 4));
        }
        if chunk.len() >= 3 {
            result.push((chunk[1] << 4) | (chunk[2] >> 2));
        }
        if chunk.len() >= 4 {
            result.push((chunk[2] << 6) | chunk[3]);
        }
    }

    Ok(result)
}

/// Hex encoding
mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_roundtrip() {
        let data = b"Hello, nvTrust!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(data.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_parse_gpu_model() {
        assert_eq!(
            parse_gpu_model("NVIDIA H100 PCIe").unwrap(),
            ConfidentialGpu::H100
        );
        assert_eq!(
            parse_gpu_model("H200 NVL").unwrap(),
            ConfidentialGpu::H200Nvl
        );
        assert_eq!(parse_gpu_model("B200").unwrap(), ConfidentialGpu::B200);
    }

    #[test]
    fn test_mock_evidence() {
        let nonce = [0u8; 32];
        let evidence = generate_mock_evidence(0, &nonce);
        assert!(evidence.starts_with(b"NVIDIA_MOCK_EVIDENCE"));
    }
}
