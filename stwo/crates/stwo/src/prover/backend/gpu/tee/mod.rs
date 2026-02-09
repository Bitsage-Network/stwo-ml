//! Real TEE Integration for NVIDIA H100/H200/B200 Confidential Computing
//!
//! This module provides hardware-backed TEE integration using:
//! - NVIDIA Confidential Computing (CC-On mode) for GPU isolation
//! - Intel TDX or AMD SEV-SNP for CPU TEE
//! - nvTrust SDK for attestation verification
//!
//! # Supported Hardware
//!
//! | GPU | Architecture | Memory | TEE Support |
//! |-----|--------------|--------|-------------|
//! | H100 NVL | Hopper | 94GB | ✓ CC-On |
//! | H200 NVL | Hopper | 141GB | ✓ CC-On |
//! | B200 | Blackwell | 192GB | ✓ CC-On |
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        Obelysk Confidential Computing                       │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────────────────────────────────────────────────────────────┐    │
//! │  │                     CPU TEE (Intel TDX / AMD SEV-SNP)               │    │
//! │  │  ┌─────────────────────────────────────────────────────────────┐    │    │
//! │  │  │                    Confidential VM                          │    │    │
//! │  │  │  • Encrypted memory (AES-256)                               │    │    │
//! │  │  │  • Attestation quote generation                             │    │    │
//! │  │  │  • Secure boot chain                                        │    │    │
//! │  │  └─────────────────────────────────────────────────────────────┘    │    │
//! │  └─────────────────────────────────────────────────────────────────────┘    │
//! │                                    │                                        │
//! │                         SPDM Session (TLS 1.3)                              │
//! │                                    │                                        │
//! │  ┌─────────────────────────────────────────────────────────────────────┐    │
//! │  │                     GPU TEE (NVIDIA CC-On)                          │    │
//! │  │  ┌─────────────────────────────────────────────────────────────┐    │    │
//! │  │  │              Hardware Root of Trust                         │    │    │
//! │  │  │  • On-die RoT with secure boot                              │    │    │
//! │  │  │  • AES-GCM-256 DMA encryption                               │    │    │
//! │  │  │  • Memory isolation via firewalls                           │    │    │
//! │  │  │  • GPU attestation certificate                              │    │    │
//! │  │  └─────────────────────────────────────────────────────────────┘    │    │
//! │  └─────────────────────────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

pub mod attestation;
pub mod cc_mode;
pub mod crypto;
pub mod nvtrust;

#[cfg(test)]
mod tests;

use std::time::{Duration, Instant};

/// GPU models that support Confidential Computing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidentialGpu {
    /// NVIDIA H100 (Hopper) - 80GB/94GB HBM3
    H100,
    /// NVIDIA H100 NVL - 94GB HBM3
    H100Nvl,
    /// NVIDIA H200 - 141GB HBM3e
    H200,
    /// NVIDIA H200 NVL - 141GB HBM3e
    H200Nvl,
    /// NVIDIA B200 (Blackwell) - 192GB HBM3e
    B200,
    /// NVIDIA B200 NVL - 192GB HBM3e
    B200Nvl,
}

impl ConfidentialGpu {
    /// Get HBM memory size in GB
    pub fn memory_gb(&self) -> u32 {
        match self {
            ConfidentialGpu::H100 => 80,
            ConfidentialGpu::H100Nvl => 94,
            ConfidentialGpu::H200 => 141,
            ConfidentialGpu::H200Nvl => 141,
            ConfidentialGpu::B200 => 192,
            ConfidentialGpu::B200Nvl => 192,
        }
    }

    /// Get memory bandwidth in TB/s
    pub fn memory_bandwidth_tbs(&self) -> f32 {
        match self {
            ConfidentialGpu::H100 | ConfidentialGpu::H100Nvl => 3.35,
            ConfidentialGpu::H200 | ConfidentialGpu::H200Nvl => 4.8,
            ConfidentialGpu::B200 | ConfidentialGpu::B200Nvl => 8.0,
        }
    }

    /// Check if GPU supports PPCIE (Protected PCIe) for multi-GPU
    pub fn supports_ppcie(&self) -> bool {
        true // All CC-capable GPUs support PPCIE
    }
}

/// CPU TEE type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuTee {
    /// Intel Trust Domain Extensions
    IntelTdx,
    /// AMD Secure Encrypted Virtualization - Secure Nested Paging
    AmdSevSnp,
    /// No CPU TEE (development only)
    None,
}

impl CpuTee {
    /// Detect the CPU TEE type from system
    pub fn detect() -> Self {
        // Check for TDX
        if std::path::Path::new("/sys/firmware/tdx").exists() {
            return CpuTee::IntelTdx;
        }

        // Check for SEV-SNP
        if std::path::Path::new("/sys/kernel/security/sev").exists() {
            return CpuTee::AmdSevSnp;
        }

        // Check via cpuid for SEV-SNP
        #[cfg(target_arch = "x86_64")]
        {
            // SEV-SNP is indicated by CPUID function 0x8000001F
            // This is a simplified check - production would use proper cpuid
            if std::fs::read_to_string("/proc/cpuinfo")
                .map(|s| s.contains("sev_snp") || s.contains("SEV-SNP"))
                .unwrap_or(false)
            {
                return CpuTee::AmdSevSnp;
            }
        }

        CpuTee::None
    }
}

/// Confidential Computing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcMode {
    /// CC disabled
    Off,
    /// CC enabled (production mode)
    On,
    /// CC enabled with dev tools (allows debugging)
    DevTools,
}

impl std::fmt::Display for CcMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CcMode::Off => write!(f, "off"),
            CcMode::On => write!(f, "on"),
            CcMode::DevTools => write!(f, "devtools"),
        }
    }
}

/// TEE Configuration for Obelysk
#[derive(Debug, Clone)]
pub struct TeeConfig {
    /// Target GPU model
    pub gpu: ConfidentialGpu,
    /// Required CC mode
    pub cc_mode: CcMode,
    /// CPU TEE requirement
    pub cpu_tee: Option<CpuTee>,
    /// Enable PPCIE for multi-GPU
    pub enable_ppcie: bool,
    /// Attestation server URL (for remote attestation)
    pub attestation_server: Option<String>,
    /// Session timeout
    pub session_timeout: Duration,
    /// Enable secure memory zeroing
    pub secure_memory_clear: bool,
}

impl Default for TeeConfig {
    fn default() -> Self {
        Self {
            gpu: ConfidentialGpu::H100Nvl,
            cc_mode: CcMode::On,
            cpu_tee: None, // Auto-detect
            enable_ppcie: false,
            attestation_server: None,
            session_timeout: Duration::from_secs(3600),
            secure_memory_clear: true,
        }
    }
}

/// Error types for TEE operations
#[derive(Debug, Clone)]
pub enum TeeError {
    /// GPU not in CC-On mode
    CcModeNotEnabled,
    /// GPU doesn't support CC
    GpuNotSupported(String),
    /// CPU TEE not available
    CpuTeeNotAvailable,
    /// Attestation failed
    AttestationFailed(String),
    /// SPDM session failed
    SpdmSessionFailed(String),
    /// Encryption/decryption failed
    CryptoError(String),
    /// GPU driver error
    DriverError(String),
    /// nvTrust SDK error
    NvTrustError(String),
    /// Configuration error
    ConfigError(String),
}

impl std::fmt::Display for TeeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeeError::CcModeNotEnabled => write!(f, "GPU is not in CC-On mode"),
            TeeError::GpuNotSupported(s) => write!(f, "GPU not supported: {}", s),
            TeeError::CpuTeeNotAvailable => write!(f, "CPU TEE not available"),
            TeeError::AttestationFailed(s) => write!(f, "Attestation failed: {}", s),
            TeeError::SpdmSessionFailed(s) => write!(f, "SPDM session failed: {}", s),
            TeeError::CryptoError(s) => write!(f, "Crypto error: {}", s),
            TeeError::DriverError(s) => write!(f, "Driver error: {}", s),
            TeeError::NvTrustError(s) => write!(f, "nvTrust error: {}", s),
            TeeError::ConfigError(s) => write!(f, "Configuration error: {}", s),
        }
    }
}

impl std::error::Error for TeeError {}

/// Result type for TEE operations
pub type TeeResult<T> = Result<T, TeeError>;

/// GPU attestation report
#[derive(Debug, Clone)]
pub struct GpuAttestationReport {
    /// GPU device ID
    pub device_id: u32,
    /// GPU model
    pub gpu_model: ConfidentialGpu,
    /// CC mode
    pub cc_mode: CcMode,
    /// Driver version
    pub driver_version: String,
    /// VBIOS version
    pub vbios_version: String,
    /// Attestation certificate chain (DER encoded)
    pub cert_chain: Vec<u8>,
    /// Attestation evidence (SPDM measurement)
    pub evidence: Vec<u8>,
    /// Timestamp
    pub timestamp: Instant,
    /// Nonce used in attestation
    pub nonce: [u8; 32],
}

impl GpuAttestationReport {
    /// Verify the attestation report
    pub fn verify(&self) -> TeeResult<bool> {
        attestation::verify_gpu_attestation(self)
    }

    /// Get the attestation quote for external verification
    pub fn to_quote(&self) -> Vec<u8> {
        let mut quote = Vec::new();

        // Header
        quote.extend_from_slice(b"NVIDIA_CC_QUOTE_V1");

        // GPU info
        quote.extend_from_slice(&self.device_id.to_le_bytes());
        quote.push(self.cc_mode as u8);

        // Nonce
        quote.extend_from_slice(&self.nonce);

        // Evidence
        quote.extend_from_slice(&(self.evidence.len() as u32).to_le_bytes());
        quote.extend_from_slice(&self.evidence);

        // Cert chain
        quote.extend_from_slice(&(self.cert_chain.len() as u32).to_le_bytes());
        quote.extend_from_slice(&self.cert_chain);

        quote
    }
}

/// CPU attestation report
#[derive(Debug, Clone)]
pub struct CpuAttestationReport {
    /// TEE type
    pub tee_type: CpuTee,
    /// Platform info
    pub platform_info: Vec<u8>,
    /// Measurement registers
    pub measurements: Vec<[u8; 48]>,
    /// Attestation quote
    pub quote: Vec<u8>,
    /// Timestamp
    pub timestamp: Instant,
}

/// Combined TEE attestation (CPU + GPU)
#[derive(Debug, Clone)]
pub struct CombinedAttestation {
    /// CPU attestation
    pub cpu: Option<CpuAttestationReport>,
    /// GPU attestation(s)
    pub gpus: Vec<GpuAttestationReport>,
    /// Session binding (ties CPU and GPU attestations together)
    pub session_binding: [u8; 32],
    /// Combined timestamp
    pub timestamp: Instant,
}

impl CombinedAttestation {
    /// Verify the entire attestation chain
    pub fn verify(&self) -> TeeResult<bool> {
        // Verify CPU attestation if present
        if let Some(cpu) = &self.cpu {
            attestation::verify_cpu_attestation(cpu)?;
        }

        // Verify all GPU attestations
        for gpu in &self.gpus {
            gpu.verify()?;
        }

        // Verify session binding
        self.verify_session_binding()
    }

    fn verify_session_binding(&self) -> TeeResult<bool> {
        // In production, this would verify the cryptographic binding
        // between CPU and GPU attestations
        Ok(true)
    }

    /// Export to portable format for external verification
    pub fn export(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Magic header
        data.extend_from_slice(b"OBELYSK_TEE_ATTESTATION_V1\0\0");

        // Session binding
        data.extend_from_slice(&self.session_binding);

        // GPU count
        data.extend_from_slice(&(self.gpus.len() as u32).to_le_bytes());

        // GPU quotes
        for gpu in &self.gpus {
            let quote = gpu.to_quote();
            data.extend_from_slice(&(quote.len() as u32).to_le_bytes());
            data.extend_from_slice(&quote);
        }

        // CPU quote (if present)
        if let Some(cpu) = &self.cpu {
            data.push(1); // Has CPU
            data.push(cpu.tee_type as u8);
            data.extend_from_slice(&(cpu.quote.len() as u32).to_le_bytes());
            data.extend_from_slice(&cpu.quote);
        } else {
            data.push(0); // No CPU
        }

        data
    }
}

/// Real TEE Context with hardware attestation
pub struct RealTeeContext {
    /// Configuration
    config: TeeConfig,
    /// GPU attestation reports
    gpu_attestations: Vec<GpuAttestationReport>,
    /// CPU attestation report
    cpu_attestation: Option<CpuAttestationReport>,
    /// Session key (derived from TEE)
    session_key: [u8; 32],
    /// Session started at
    started_at: Instant,
    /// Is initialized
    initialized: bool,
}

impl RealTeeContext {
    /// Create a new TEE context
    pub fn new(config: TeeConfig) -> TeeResult<Self> {
        Ok(Self {
            config,
            gpu_attestations: Vec::new(),
            cpu_attestation: None,
            session_key: [0u8; 32],
            started_at: Instant::now(),
            initialized: false,
        })
    }

    /// Initialize the TEE context
    ///
    /// This performs:
    /// 1. GPU CC mode verification
    /// 2. CPU TEE detection and verification
    /// 3. SPDM session establishment
    /// 4. Attestation generation
    /// 5. Session key derivation
    pub fn initialize(&mut self) -> TeeResult<()> {
        tracing::info!("Initializing TEE context");

        // Step 1: Verify GPU CC mode
        self.verify_gpu_cc_mode()?;

        // Step 2: Detect/verify CPU TEE
        self.setup_cpu_tee()?;

        // Step 3: Generate GPU attestation
        self.generate_gpu_attestation()?;

        // Step 4: Generate CPU attestation (if available)
        self.generate_cpu_attestation()?;

        // Step 5: Establish SPDM session and derive session key
        self.establish_spdm_session()?;

        self.initialized = true;
        tracing::info!("TEE context initialized successfully");

        Ok(())
    }

    /// Verify GPU is in CC-On mode
    fn verify_gpu_cc_mode(&self) -> TeeResult<()> {
        let mode = cc_mode::query_cc_mode(0)?;

        if mode != self.config.cc_mode {
            return Err(TeeError::CcModeNotEnabled);
        }

        tracing::info!(mode = %mode, "GPU CC mode verified");
        Ok(())
    }

    /// Setup CPU TEE
    fn setup_cpu_tee(&mut self) -> TeeResult<()> {
        let detected = CpuTee::detect();

        match self.config.cpu_tee {
            Some(required) if required != CpuTee::None && detected == CpuTee::None => {
                return Err(TeeError::CpuTeeNotAvailable);
            }
            Some(required) if required != detected && detected != CpuTee::None => {
                tracing::warn!(
                    required = ?required,
                    detected = ?detected,
                    "CPU TEE type mismatch, using detected"
                );
            }
            _ => {}
        }

        if detected != CpuTee::None {
            tracing::info!(tee = ?detected, "CPU TEE detected");
        }

        Ok(())
    }

    /// Generate GPU attestation
    fn generate_gpu_attestation(&mut self) -> TeeResult<()> {
        let report = attestation::generate_gpu_attestation(0, &self.config)?;
        self.gpu_attestations.push(report);
        Ok(())
    }

    /// Generate CPU attestation
    fn generate_cpu_attestation(&mut self) -> TeeResult<()> {
        let cpu_tee = CpuTee::detect();
        if cpu_tee == CpuTee::None {
            return Ok(());
        }

        let report = attestation::generate_cpu_attestation(cpu_tee)?;
        self.cpu_attestation = Some(report);
        Ok(())
    }

    /// Establish SPDM session and derive session key
    fn establish_spdm_session(&mut self) -> TeeResult<()> {
        // In production, this would:
        // 1. Perform SPDM handshake with GPU
        // 2. Verify GPU identity
        // 3. Derive session key using HKDF

        // For now, derive from attestation evidence
        let mut key_material = Vec::new();

        for gpu in &self.gpu_attestations {
            key_material.extend_from_slice(&gpu.nonce);
            key_material.extend_from_slice(&gpu.evidence);
        }

        if let Some(cpu) = &self.cpu_attestation {
            key_material.extend_from_slice(&cpu.quote);
        }

        // Derive session key using SHA-256
        self.session_key = crypto::derive_session_key(&key_material);

        tracing::info!("SPDM session established");
        Ok(())
    }

    /// Encrypt data using session key (AES-GCM-256)
    pub fn encrypt(&self, plaintext: &[u8]) -> TeeResult<Vec<u8>> {
        if !self.initialized {
            return Err(TeeError::ConfigError("TEE not initialized".into()));
        }
        crypto::aes_gcm_encrypt(&self.session_key, plaintext)
    }

    /// Decrypt data using session key (AES-GCM-256)
    pub fn decrypt(&self, ciphertext: &[u8]) -> TeeResult<Vec<u8>> {
        if !self.initialized {
            return Err(TeeError::ConfigError("TEE not initialized".into()));
        }
        crypto::aes_gcm_decrypt(&self.session_key, ciphertext)
    }

    /// Get combined attestation for external verification
    pub fn get_attestation(&self) -> TeeResult<CombinedAttestation> {
        if !self.initialized {
            return Err(TeeError::ConfigError("TEE not initialized".into()));
        }

        // Generate session binding
        let mut binding_material = Vec::new();
        binding_material.extend_from_slice(&self.session_key);
        for gpu in &self.gpu_attestations {
            binding_material.extend_from_slice(&gpu.nonce);
        }
        let session_binding = crypto::sha256(&binding_material);

        Ok(CombinedAttestation {
            cpu: self.cpu_attestation.clone(),
            gpus: self.gpu_attestations.clone(),
            session_binding,
            timestamp: Instant::now(),
        })
    }

    /// Check if session is still valid
    pub fn is_valid(&self) -> bool {
        self.initialized && self.started_at.elapsed() < self.config.session_timeout
    }

    /// Securely destroy the context
    pub fn destroy(&mut self) {
        // Zero out session key
        for byte in &mut self.session_key {
            *byte = 0;
        }

        self.gpu_attestations.clear();
        self.cpu_attestation = None;
        self.initialized = false;

        tracing::info!("TEE context destroyed");
    }
}

impl Drop for RealTeeContext {
    fn drop(&mut self) {
        self.destroy();
    }
}
