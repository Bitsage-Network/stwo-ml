//! Confidential Computing Mode Management for NVIDIA GPUs
//!
//! This module provides functionality to query and configure CC mode on
//! NVIDIA H100/H200/B200 GPUs.
//!
//! # CC Mode States
//!
//! - **Off**: Confidential Computing disabled
//! - **On**: CC enabled for production use
//! - **DevTools**: CC enabled with debugging capabilities

use super::{CcMode, TeeError, TeeResult};
use std::process::Command;

/// Query the current CC mode of a GPU
pub fn query_cc_mode(device_id: u32) -> TeeResult<CcMode> {
    // Try nvidia-smi first for CC mode
    if let Ok(mode) = query_cc_mode_nvidia_smi(device_id) {
        return Ok(mode);
    }

    // Try nvtrust-rs style query
    if let Ok(mode) = query_cc_mode_nvtrust(device_id) {
        return Ok(mode);
    }

    // Fallback: Check GPU driver sysfs
    query_cc_mode_sysfs(device_id)
}

/// Query CC mode via nvidia-smi
fn query_cc_mode_nvidia_smi(device_id: u32) -> TeeResult<CcMode> {
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            &device_id.to_string(),
            "--query-gpu=cc_mode",
            "--format=csv,noheader",
        ])
        .output()
        .map_err(|e| TeeError::DriverError(format!("nvidia-smi failed: {}", e)))?;

    if !output.status.success() {
        // CC mode query might not be available on older drivers
        return Err(TeeError::DriverError(
            "nvidia-smi cc_mode query not supported".into(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_cc_mode(stdout.trim())
}

/// Query CC mode via nvtrust tool
#[allow(unused_variables)]
fn query_cc_mode_nvtrust(device_id: u32) -> TeeResult<CcMode> {
    // Try the official nvidia-smi conf-compute command
    let output = Command::new("nvidia-smi")
        .args(["conf-compute", "-gcs"])
        .output()
        .map_err(|e| TeeError::DriverError(format!("nvidia-smi conf-compute failed: {}", e)))?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse the output for CC mode
        for line in stdout.lines() {
            if line.contains("CC Mode") || line.contains("Confidential Compute") {
                if line.contains("ON") || line.contains("Enabled") {
                    return Ok(CcMode::On);
                } else if line.contains("DEVTOOLS") || line.contains("DevTools") {
                    return Ok(CcMode::DevTools);
                } else if line.contains("OFF") || line.contains("Disabled") {
                    return Ok(CcMode::Off);
                }
            }
        }
    }

    Err(TeeError::DriverError("Could not parse CC mode".into()))
}

/// Query CC mode via sysfs
fn query_cc_mode_sysfs(device_id: u32) -> TeeResult<CcMode> {
    // Check sysfs for CC mode
    let _sysfs_path = format!(
        "/sys/bus/pci/devices/*/nvidia/{}/conf_compute_mode",
        device_id
    );

    // Use glob to find the actual path
    if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
        for entry in entries.flatten() {
            let cc_path = entry.path().join(format!("nvidia/{}/conf_compute_mode", device_id));
            if cc_path.exists() {
                if let Ok(mode_str) = std::fs::read_to_string(&cc_path) {
                    return parse_cc_mode(mode_str.trim());
                }
            }
        }
    }

    // Default to Off if we can't determine
    tracing::warn!(
        device_id = device_id,
        "Could not determine CC mode, assuming Off"
    );
    Ok(CcMode::Off)
}

/// Parse CC mode from string
fn parse_cc_mode(s: &str) -> TeeResult<CcMode> {
    let s_lower = s.to_lowercase();

    if s_lower.contains("on") || s_lower.contains("enabled") || s == "1" {
        Ok(CcMode::On)
    } else if s_lower.contains("devtools") || s_lower.contains("dev") || s == "2" {
        Ok(CcMode::DevTools)
    } else if s_lower.contains("off") || s_lower.contains("disabled") || s == "0" {
        Ok(CcMode::Off)
    } else {
        Err(TeeError::DriverError(format!(
            "Unknown CC mode: {}",
            s
        )))
    }
}

/// Set CC mode on a GPU (requires reboot)
///
/// # Warning
///
/// Changing CC mode requires a GPU reset or system reboot.
/// This should only be done during system setup, not during operation.
pub fn set_cc_mode(device_id: u32, mode: CcMode) -> TeeResult<()> {
    let mode_str = match mode {
        CcMode::Off => "off",
        CcMode::On => "on",
        CcMode::DevTools => "devtools",
    };

    // Use nvidia-smi conf-compute
    let output = Command::new("nvidia-smi")
        .args([
            "conf-compute",
            "-scc",
            mode_str,
            "-i",
            &device_id.to_string(),
        ])
        .output()
        .map_err(|e| TeeError::DriverError(format!("Failed to set CC mode: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(TeeError::DriverError(format!(
            "Failed to set CC mode: {}",
            stderr
        )));
    }

    tracing::info!(
        device_id = device_id,
        mode = %mode,
        "CC mode set (reboot required)"
    );

    Ok(())
}

/// Check if GPU supports Confidential Computing
pub fn supports_cc(device_id: u32) -> TeeResult<bool> {
    // Query GPU name
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            &device_id.to_string(),
            "--query-gpu=name",
            "--format=csv,noheader",
        ])
        .output()
        .map_err(|e| TeeError::DriverError(format!("nvidia-smi failed: {}", e)))?;

    if !output.status.success() {
        return Err(TeeError::DriverError("Could not query GPU name".into()));
    }

    let name = String::from_utf8_lossy(&output.stdout);
    let name_upper = name.to_uppercase();

    // Only Hopper (H100, H200) and Blackwell (B200) support CC
    let supported = name_upper.contains("H100")
        || name_upper.contains("H200")
        || name_upper.contains("B200")
        || name_upper.contains("B100");

    Ok(supported)
}

/// Get CC settings (detailed configuration)
pub fn query_cc_settings(device_id: u32) -> TeeResult<CcSettings> {
    let output = Command::new("nvidia-smi")
        .args(["conf-compute", "-gcs", "-i", &device_id.to_string()])
        .output()
        .map_err(|e| TeeError::DriverError(format!("nvidia-smi conf-compute failed: {}", e)))?;

    if !output.status.success() {
        return Err(TeeError::DriverError(
            "Failed to query CC settings".into(),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_cc_settings(&stdout)
}

/// CC Settings from the GPU
#[derive(Debug, Clone)]
pub struct CcSettings {
    /// Current CC mode
    pub mode: CcMode,
    /// CC mode that will take effect after reset
    pub pending_mode: Option<CcMode>,
    /// Protected PCIe enabled
    pub ppcie_enabled: bool,
    /// Multi-GPU mode
    pub multi_gpu_mode: Option<String>,
    /// Driver version
    pub driver_version: String,
    /// VBIOS version
    pub vbios_version: String,
}

fn parse_cc_settings(output: &str) -> TeeResult<CcSettings> {
    let mut settings = CcSettings {
        mode: CcMode::Off,
        pending_mode: None,
        ppcie_enabled: false,
        multi_gpu_mode: None,
        driver_version: String::new(),
        vbios_version: String::new(),
    };

    for line in output.lines() {
        let line = line.trim();

        if line.contains("CC Mode") || line.contains("Current CC Mode") {
            if line.contains("ON") {
                settings.mode = CcMode::On;
            } else if line.contains("DEVTOOLS") {
                settings.mode = CcMode::DevTools;
            }
        } else if line.contains("Pending CC Mode") {
            if line.contains("ON") {
                settings.pending_mode = Some(CcMode::On);
            } else if line.contains("DEVTOOLS") {
                settings.pending_mode = Some(CcMode::DevTools);
            } else if line.contains("OFF") {
                settings.pending_mode = Some(CcMode::Off);
            }
        } else if line.contains("PPCIE") || line.contains("Protected PCIe") {
            settings.ppcie_enabled = line.contains("Enabled") || line.contains("ON");
        } else if line.contains("Multi-GPU") {
            if let Some(mode) = line.split(':').nth(1) {
                settings.multi_gpu_mode = Some(mode.trim().to_string());
            }
        } else if line.contains("Driver Version") {
            if let Some(version) = line.split(':').nth(1) {
                settings.driver_version = version.trim().to_string();
            }
        } else if line.contains("VBIOS") {
            if let Some(version) = line.split(':').nth(1) {
                settings.vbios_version = version.trim().to_string();
            }
        }
    }

    Ok(settings)
}

/// Verify the GPU is properly configured for CC
pub fn verify_cc_configuration(device_id: u32) -> TeeResult<CcVerificationResult> {
    let mut result = CcVerificationResult::default();

    // Check if GPU supports CC
    result.gpu_supported = supports_cc(device_id)?;

    if !result.gpu_supported {
        return Ok(result);
    }

    // Check current CC mode
    result.cc_mode = query_cc_mode(device_id)?;
    result.cc_enabled = matches!(result.cc_mode, CcMode::On | CcMode::DevTools);

    // Check driver version
    if let Ok(settings) = query_cc_settings(device_id) {
        result.driver_version = settings.driver_version;
        result.ppcie_enabled = settings.ppcie_enabled;

        // Verify driver version is compatible (r550+)
        if let Some(major) = result.driver_version.split('.').next() {
            if let Ok(major_num) = major.parse::<u32>() {
                result.driver_compatible = major_num >= 550;
            }
        }
    }

    Ok(result)
}

/// Result of CC configuration verification
#[derive(Debug, Clone, Default)]
pub struct CcVerificationResult {
    /// GPU supports CC
    pub gpu_supported: bool,
    /// CC is currently enabled
    pub cc_enabled: bool,
    /// Current CC mode
    pub cc_mode: CcMode,
    /// PPCIE enabled
    pub ppcie_enabled: bool,
    /// Driver version
    pub driver_version: String,
    /// Driver is compatible (r550+)
    pub driver_compatible: bool,
}

impl Default for CcMode {
    fn default() -> Self {
        CcMode::Off
    }
}

impl CcVerificationResult {
    /// Check if configuration is ready for production
    pub fn is_production_ready(&self) -> bool {
        self.gpu_supported
            && self.cc_enabled
            && self.cc_mode == CcMode::On
            && self.driver_compatible
    }

    /// Get human-readable status
    pub fn status_message(&self) -> String {
        if !self.gpu_supported {
            return "GPU does not support Confidential Computing (need H100/H200/B200)".into();
        }

        if !self.cc_enabled {
            return format!(
                "CC mode is {}, needs to be enabled via: nvidia-smi conf-compute -scc on",
                self.cc_mode
            );
        }

        if !self.driver_compatible {
            return format!(
                "Driver version {} is not compatible, need r550 or later",
                self.driver_version
            );
        }

        if self.cc_mode == CcMode::DevTools {
            return "CC mode is DevTools (debugging enabled) - use 'on' for production".into();
        }

        "Configuration is production ready".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cc_mode() {
        assert_eq!(parse_cc_mode("on").unwrap(), CcMode::On);
        assert_eq!(parse_cc_mode("ON").unwrap(), CcMode::On);
        assert_eq!(parse_cc_mode("Enabled").unwrap(), CcMode::On);
        assert_eq!(parse_cc_mode("1").unwrap(), CcMode::On);

        assert_eq!(parse_cc_mode("off").unwrap(), CcMode::Off);
        assert_eq!(parse_cc_mode("OFF").unwrap(), CcMode::Off);
        assert_eq!(parse_cc_mode("Disabled").unwrap(), CcMode::Off);
        assert_eq!(parse_cc_mode("0").unwrap(), CcMode::Off);

        assert_eq!(parse_cc_mode("devtools").unwrap(), CcMode::DevTools);
        assert_eq!(parse_cc_mode("DEVTOOLS").unwrap(), CcMode::DevTools);
        assert_eq!(parse_cc_mode("2").unwrap(), CcMode::DevTools);
    }

    #[test]
    fn test_verification_result() {
        let mut result = CcVerificationResult::default();
        assert!(!result.is_production_ready());

        result.gpu_supported = true;
        result.cc_enabled = true;
        result.cc_mode = CcMode::On;
        result.driver_compatible = true;

        assert!(result.is_production_ready());
    }
}
