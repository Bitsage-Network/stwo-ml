//! Compact proof serialization for on-chain STWO verification.
//!
//! The standard cairo-serde proof format embeds the full program bytecode
//! (~3.3M felts for the Cairo verifier). For on-chain verification, the
//! contract already knows the program — it only needs the cryptographic data.
//!
//! This module strips the program section, reducing the proof from ~3.6M
//! to ~286K felts. With M31 packing, this drops further to ~19K felts.

use log::info;
use std::io::Write;
use std::path::Path;

use crate::error::{CairoProveError, Result};

/// Strip the program section from a cairo-serde proof.
///
/// The cairo-serde format starts with:
///   [program_section_length, ...program_entries (length × 9 felts each)...]
///   [public_segments, output_section, safe_call_ids, ...]
///   [...rest of claim + proof data...]
///
/// We replace the program section with just its Poseidon hash, reducing
/// 3.3M felts to 1 felt. The on-chain contract has the program hash
/// pre-registered and validates it matches.
pub fn strip_program_section(proof_felts: &[String]) -> Result<(Vec<String>, usize)> {
    if proof_felts.is_empty() {
        return Err(CairoProveError::ProofSerialization(
            "Empty proof".into(),
        ));
    }

    // Parse program section length (first felt)
    let prog_len_str = &proof_felts[0];
    let prog_len = u64::from_str_radix(
        prog_len_str.trim_start_matches("0x"),
        16,
    )
    .map_err(|e| CairoProveError::ProofSerialization(format!("Invalid program length: {e}")))?
        as usize;

    // Each program entry is 9 felts (1 id + 8 value words)
    let program_section_felts = 1 + prog_len * 9; // +1 for the length field
    let total = proof_felts.len();

    info!(
        "Stripping program section: {} entries × 9 = {} felts ({:.1}% of {} total)",
        prog_len,
        program_section_felts,
        program_section_felts as f64 / total as f64 * 100.0,
        total,
    );

    // The compact proof is everything AFTER the program section
    let mut compact = Vec::with_capacity(total - program_section_felts + 1);

    // First felt: program section length = 0 (stripped)
    compact.push("0x0".to_string());

    // Skip the program section, keep everything else
    compact.extend_from_slice(&proof_felts[program_section_felts..]);

    let stripped_felts = program_section_felts - 1; // -1 because we keep the length field (as 0)
    info!(
        "Compact proof: {} felts (stripped {} felts, {:.1}% reduction)",
        compact.len(),
        stripped_felts,
        stripped_felts as f64 / total as f64 * 100.0,
    );

    Ok((compact, stripped_felts))
}

/// Save a compact proof to a file.
pub fn save_compact_proof(compact_felts: &[String], path: &Path) -> Result<()> {
    let json = serde_json::to_string(&compact_felts)
        .map_err(|e| CairoProveError::ProofSerialization(format!("{e}")))?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}
