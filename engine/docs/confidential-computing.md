# Confidential Computing Plan

**Status**: PLANNED (not yet implemented)
**Priority**: After prover stabilization

## Overview

The ZKML prover currently supports GPU-only TEE attestation (NVIDIA H100 CC mode via `nvattest`). This encrypts GPU memory but the host can still read CPU memory. Full confidentiality requires a Confidential VM (CVM) where CPU memory is also hardware-encrypted.

## Current State (GPU-Only CC)

- `src/tee.rs`: GPU CC detection, nvattest attestation, Poseidon hash for on-chain
- `SecurityLevel` enum: `ZkOnly`, `ZkPlusTee`, `Auto`
- Feature flag: `tee = ["cuda-runtime"]`
- `tee_attestation_hash` flows through: prove -> cairo_serde -> ObelyskVerifier contract
- Works on any H100/H200/B200 with CC mode enabled (e.g., Shadeform)

### What GPU CC Protects

- GPU HBM memory (AES hardware encryption)
- PCIe bus (encrypted CPU-GPU link)
- Cross-tenant GPU memory isolation

### What GPU CC Does NOT Protect

- CPU-side memory (model weights, inputs, outputs in host RAM)
- Filesystem (model files, proof outputs)
- Network traffic
- Process visibility (host can see what's running)

## Full-Stack CC Architecture

```
              Full-Stack Confidential Computing
             /                                  \
    CPU Attestation                     GPU Attestation (existing)
   /              \                            |
AMD SEV-SNP    Intel TDX                 NVIDIA nvattest
(/dev/sev-guest) (/dev/tdx-guest)        (H100 CC mode)
```

### Combined Attestation Hash

- GPU-only (no CVM): `combined_hash = gpu.report_hash_felt()` (backwards-compatible)
- CPU-only (no GPU CC): `combined_hash = Poseidon(0, cpu_hash)`
- Both (full-stack): `combined_hash = Poseidon(cpu_hash, gpu_hash)`
- Neither: `combined_hash = FieldElement::ZERO`

Cross-attestation binding: GPU report hash is embedded in CPU report's `report_data` field (64 bytes), cryptographically binding the two attestations.

## Implementation Plan

### New Feature Flags

```toml
cpu-tee = ["dep:sev", "dep:tdx-guest"]    # CPU attestation
full-cc = ["tee", "cpu-tee"]               # Full-stack (GPU + CPU)
```

### New Dependencies (linux x86_64 only)

- `sev` crate (v7+): AMD SEV-SNP `/dev/sev-guest` ioctl via Rust (no CLI needed)
- `tdx-guest` crate (v0.1+): Intel TDX `/dev/tdx_guest` ioctl via Rust

### New Types (`src/tee.rs`)

- `CpuPlatform { SevSnp, Tdx, None }`
- `CpuAttestation { platform, report, measurement: [u8; 48], timestamp, report_data, platform_version }`
- `FullStackAttestation { cpu: Option<CpuAttestation>, gpu: Option<TeeAttestation>, combined_hash }`
- `FullStackCapability { gpu: TeeCapability, cpu_platform, cpu_device_available }`

### Files to Modify

| File | Change |
|------|--------|
| `src/tee.rs` | ~300 lines: new types, CPU attestation via sev/tdx-guest crates, combined hash |
| `Cargo.toml` | New optional deps + feature flags |
| `src/gpu.rs` | Full-stack attestation method on GpuModelProver |
| `src/lib.rs` | New re-exports |
| `src/cairo_serde.rs` | Convenience method for FullStackAttestation |
| `src/starknet.rs` | New proof builder functions |
| `src/receipt.rs` | cpu_tee_report_hash field |
| `src/bin/prove_model.rs` | Full-stack detection + generation in CLI |
| `src/bin/prove_server.rs` | Health endpoint: cpu_tee_available, full_stack_cc |

### No On-Chain Contract Changes

The existing `tee_attestation_hash: felt252` field carries the combined hash. GPU-only hash is identical to current format. No contract upgrade needed.

## Cloud Deployment

### Azure NCCads_H100_v5 (AMD SEV-SNP + H100 CC)

- 40 AMD EPYC cores, 320 GiB RAM, 1x H100 NVL 94GB
- Full TEE: CPU (SEV-SNP) + GPU (H100 CC auto-enabled by hypervisor)
- Regions: East US 2, West Europe
- Cost: ~$3-4/hr

```bash
az vm create \
    --size Standard_NCCads_H100_v5 \
    --security-type ConfidentialVM \
    --enable-secure-boot true \
    --enable-vtpm true
```

### GCP A3 Confidential (TDX/SEV-SNP + H100 CC)

- A3 instances with confidential computing
- GPU CC auto-enabled by hypervisor
- Regions: europe-west4-c, us-central1-a, us-east5-a

```bash
gcloud compute instances create my-prover \
    --machine-type a3-highgpu-1g \
    --confidential-compute \
    --provisioning-model SPOT
```

### Detection Script (`scripts/setup_cc.sh`)

Checks `/dev/sev-guest` (SEV-SNP) or `/dev/tdx-guest` (TDX) for CPU CC, `nvidia-smi conf-compute -gcs` for GPU CC, and recommends feature flags.

## Security Comparison

| Threat | Regular VM | GPU CC Only | Full-Stack CC |
|--------|-----------|-------------|---------------|
| GPU memory snooping | Vulnerable | Protected | Protected |
| CPU memory snooping | Vulnerable | Vulnerable | Protected |
| Malicious host admin | Vulnerable | Partially protected | Protected |
| Cold boot attacks | Vulnerable | GPU protected | Fully protected |
| ZK proof validity | Valid | Valid | Valid |
| Hardware attestation | None | GPU only | CPU + GPU |
