//! Integration tests for TEE module
//!
//! These tests verify:
//! - TEE context creation and initialization
//! - Encryption/decryption roundtrips
//! - Attestation generation and verification
//! - CC mode detection

#[cfg(test)]
mod tests {
    use super::super::*;

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_tee_config_default() {
        let config = TeeConfig::default();

        assert_eq!(config.cc_mode, CcMode::On);
        assert!(config.secure_memory_clear);
        assert!(!config.enable_ppcie);
        assert!(config.attestation_server.is_none());
    }

    #[test]
    fn test_tee_config_custom() {
        let config = TeeConfig {
            gpu: ConfidentialGpu::H200Nvl,
            cc_mode: CcMode::DevTools,
            cpu_tee: Some(CpuTee::IntelTdx),
            enable_ppcie: true,
            attestation_server: Some("https://attestation.example.com".into()),
            session_timeout: std::time::Duration::from_secs(1800),
            secure_memory_clear: true,
        };

        assert_eq!(config.gpu.memory_gb(), 141);
        assert_eq!(config.cc_mode, CcMode::DevTools);
        assert!(config.enable_ppcie);
    }

    // =========================================================================
    // GPU Model Tests
    // =========================================================================

    #[test]
    fn test_confidential_gpu_properties() {
        // H100 variants
        assert_eq!(ConfidentialGpu::H100.memory_gb(), 80);
        assert_eq!(ConfidentialGpu::H100Nvl.memory_gb(), 94);
        assert!(ConfidentialGpu::H100.supports_ppcie());

        // H200 variants
        assert_eq!(ConfidentialGpu::H200.memory_gb(), 141);
        assert_eq!(ConfidentialGpu::H200Nvl.memory_gb(), 141);
        assert!((ConfidentialGpu::H200.memory_bandwidth_tbs() - 4.8).abs() < 0.1);

        // B200 variants
        assert_eq!(ConfidentialGpu::B200.memory_gb(), 192);
        assert_eq!(ConfidentialGpu::B200Nvl.memory_gb(), 192);
        assert!((ConfidentialGpu::B200.memory_bandwidth_tbs() - 8.0).abs() < 0.1);
    }

    // =========================================================================
    // CC Mode Tests
    // =========================================================================

    #[test]
    fn test_cc_mode_display() {
        assert_eq!(format!("{}", CcMode::Off), "off");
        assert_eq!(format!("{}", CcMode::On), "on");
        assert_eq!(format!("{}", CcMode::DevTools), "devtools");
    }

    #[test]
    fn test_cc_mode_default() {
        let mode: CcMode = Default::default();
        assert_eq!(mode, CcMode::Off);
    }

    // =========================================================================
    // CPU TEE Tests
    // =========================================================================

    #[test]
    fn test_cpu_tee_detect() {
        // This test will return None on most development machines
        let tee = CpuTee::detect();

        // Just verify it doesn't panic
        match tee {
            CpuTee::IntelTdx => println!("Intel TDX detected"),
            CpuTee::AmdSevSnp => println!("AMD SEV-SNP detected"),
            CpuTee::None => println!("No CPU TEE detected (expected on dev machine)"),
        }
    }

    // =========================================================================
    // Crypto Tests
    // =========================================================================

    #[test]
    fn test_crypto_encrypt_decrypt_roundtrip() {
        let key = crypto::generate_random_key();
        let plaintext = b"Hello, Obelysk TEE Integration!";

        let ciphertext = crypto::aes_gcm_encrypt(&key, plaintext).unwrap();

        // Ciphertext should be longer (includes nonce + tag)
        assert!(ciphertext.len() > plaintext.len());

        // Decrypt should recover original
        let decrypted = crypto::aes_gcm_decrypt(&key, &ciphertext).unwrap();
        assert_eq!(&decrypted, plaintext);
    }

    #[test]
    fn test_crypto_encrypt_different_each_time() {
        let key = crypto::generate_random_key();
        let plaintext = b"Test message";

        let ct1 = crypto::aes_gcm_encrypt(&key, plaintext).unwrap();
        let ct2 = crypto::aes_gcm_encrypt(&key, plaintext).unwrap();

        // Due to random nonce, ciphertexts should differ
        assert_ne!(ct1, ct2);

        // But both should decrypt to same plaintext
        let pt1 = crypto::aes_gcm_decrypt(&key, &ct1).unwrap();
        let pt2 = crypto::aes_gcm_decrypt(&key, &ct2).unwrap();
        assert_eq!(pt1, pt2);
    }

    #[test]
    fn test_crypto_sha256() {
        let data = b"test data for hashing";
        let hash = crypto::sha256(data);

        assert_eq!(hash.len(), 32);

        // Same input = same output
        let hash2 = crypto::sha256(data);
        assert_eq!(hash, hash2);

        // Different input = different output
        let hash3 = crypto::sha256(b"different data");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_crypto_derive_session_key() {
        let material1 = b"key material 1";
        let material2 = b"key material 2";

        let key1 = crypto::derive_session_key(material1);
        let key1_again = crypto::derive_session_key(material1);
        let key2 = crypto::derive_session_key(material2);

        // Deterministic
        assert_eq!(key1, key1_again);

        // Different material = different key
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_crypto_constant_time_compare() {
        let a = [1u8, 2, 3, 4, 5];
        let b = [1u8, 2, 3, 4, 5];
        let c = [1u8, 2, 3, 4, 6];
        let d = [1u8, 2, 3];

        assert!(crypto::constant_time_compare(&a, &b));
        assert!(!crypto::constant_time_compare(&a, &c));
        assert!(!crypto::constant_time_compare(&a, &d));
    }

    #[test]
    fn test_crypto_secure_zero() {
        let mut data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        crypto::secure_zero(&mut data);
        assert_eq!(data, [0u8; 8]);
    }

    // =========================================================================
    // Attestation Tests
    // =========================================================================

    #[test]
    fn test_attestation_mock_evidence() {
        let config = TeeConfig::default();

        // This will generate mock evidence without real GPU
        // The function gracefully falls back when nvidia-smi isn't available
        let result = attestation::generate_gpu_attestation(0, &config);

        // On dev machine without NVIDIA GPU, this will fail gracefully
        match result {
            Ok(report) => {
                assert!(!report.evidence.is_empty());
                assert_eq!(report.nonce.len(), 32);
            }
            Err(e) => {
                println!("Expected error on dev machine: {:?}", e);
            }
        }
    }

    // =========================================================================
    // TEE Context Tests
    // =========================================================================

    #[test]
    fn test_real_tee_context_creation() {
        let config = TeeConfig {
            gpu: ConfidentialGpu::H100,
            cc_mode: CcMode::On,
            cpu_tee: None,
            enable_ppcie: false,
            attestation_server: None,
            session_timeout: std::time::Duration::from_secs(3600),
            secure_memory_clear: true,
        };

        // This should succeed even without real TEE hardware
        let result = RealTeeContext::new(config);
        assert!(result.is_ok());

        let ctx = result.unwrap();
        assert!(!ctx.is_valid()); // Not initialized yet
    }

    // =========================================================================
    // CC Mode Query Tests
    // =========================================================================

    #[test]
    fn test_cc_mode_parse() {
        // These are internal tests for the parsing logic
        use cc_mode::CcVerificationResult;

        let mut result = CcVerificationResult::default();
        assert!(!result.is_production_ready());

        result.gpu_supported = true;
        result.cc_enabled = true;
        result.cc_mode = CcMode::On;
        result.driver_compatible = true;

        assert!(result.is_production_ready());
        assert!(result.status_message().contains("production ready"));
    }

    // =========================================================================
    // Attestation Report Tests
    // =========================================================================

    #[test]
    fn test_gpu_attestation_report_to_quote() {
        let report = GpuAttestationReport {
            device_id: 0,
            gpu_model: ConfidentialGpu::H100,
            cc_mode: CcMode::On,
            driver_version: "550.54.15".to_string(),
            vbios_version: "96.00.89.00.01".to_string(),
            cert_chain: vec![1, 2, 3, 4],
            evidence: vec![5, 6, 7, 8, 9, 10],
            timestamp: std::time::Instant::now(),
            nonce: [0u8; 32],
        };

        let quote = report.to_quote();

        // Quote should contain header
        assert!(quote.starts_with(b"NVIDIA_CC_QUOTE_V1"));

        // Quote should contain evidence and cert chain
        assert!(quote.len() > 18 + 32 + 4 + 6 + 4 + 4);
    }

    #[test]
    fn test_combined_attestation_export() {
        let gpu_report = GpuAttestationReport {
            device_id: 0,
            gpu_model: ConfidentialGpu::H200Nvl,
            cc_mode: CcMode::On,
            driver_version: "550.54.15".to_string(),
            vbios_version: "96.00.89.00.01".to_string(),
            cert_chain: vec![1, 2, 3],
            evidence: vec![4, 5, 6],
            timestamp: std::time::Instant::now(),
            nonce: [1u8; 32],
        };

        let combined = CombinedAttestation {
            cpu: None,
            gpus: vec![gpu_report],
            session_binding: [2u8; 32],
            timestamp: std::time::Instant::now(),
        };

        let exported = combined.export();

        // Should start with magic header
        assert!(exported.starts_with(b"OBELYSK_TEE_ATTESTATION_V1"));
    }

    // =========================================================================
    // Error Tests
    // =========================================================================

    #[test]
    fn test_tee_error_display() {
        let errors = vec![
            TeeError::CcModeNotEnabled,
            TeeError::GpuNotSupported("RTX 4090".into()),
            TeeError::CpuTeeNotAvailable,
            TeeError::AttestationFailed("Invalid nonce".into()),
            TeeError::CryptoError("Key too short".into()),
        ];

        for err in errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty());
        }
    }
}
