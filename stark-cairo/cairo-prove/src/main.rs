use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use cairo_air::utils::{ProofFormat, serialize_proof_to_file};
use cairo_air::verifier::verify_cairo;
use cairo_air::{CairoProof, CairoProofForRustVerifier};
use cairo_lang_runner::Arg;
use cairo_prove::args::{Cli, Commands, ProgramArguments};
use cairo_prove::error::{CairoProveError, Result};
use cairo_prove::execute::execute;
use cairo_prove::prove::{prove, prove_poseidon, prover_input_from_runner};
use clap::Parser;
use log::{error, info};
use stwo_cairo_prover::stwo::core::fri::FriConfig;
use stwo_cairo_prover::stwo::core::pcs::PcsConfig;
use stwo_cairo_prover::stwo::core::vcs_lifted::blake2_merkle::{
    Blake2sMerkleChannel, Blake2sMerkleHasher,
};
use stwo_cairo_prover::stwo::core::vcs_lifted::poseidon252_merkle::{
    Poseidon252MerkleChannel, Poseidon252MerkleHasher,
};

fn secure_pcs_config() -> PcsConfig {
    PcsConfig {
        pow_bits: 26,
        fri_config: FriConfig {
            log_last_layer_degree_bound: 0,
            log_blowup_factor: 1,
            n_queries: 70,
            fold_step: 1,
        },
        lifting_log_size: None,
    }
}

/// Compact PCS config for the recursive level — fewer queries since the inner
/// proof is already verified. Reduces verifier step count and Level 2 proof size.
fn recursive_pcs_config() -> PcsConfig {
    PcsConfig {
        pow_bits: 20,
        fri_config: FriConfig {
            log_last_layer_degree_bound: 0,
            log_blowup_factor: 1,
            n_queries: 20,
            fold_step: 1,
        },
        lifting_log_size: None,
    }
}

fn handle_prove(
    target: &Path,
    proof: &Path,
    proof_format: ProofFormat,
    poseidon: bool,
    args: ProgramArguments,
) -> Result<()> {
    info!("Generating proof for target: {:?}", target);
    let start = Instant::now();

    let target_str = target
        .to_str()
        .ok_or_else(|| CairoProveError::InvalidPath {
            path: target.to_path_buf(),
        })?;
    let file = std::fs::File::open(target_str)?;
    let executable = serde_json::from_reader(file)?;
    let runner = execute(executable, args.read_arguments())?;
    let prover_input = prover_input_from_runner(&runner)?;

    if poseidon {
        info!("[Poseidon252] On-chain recursive path enabled.");
        let cairo_proof = prove_poseidon(prover_input, secure_pcs_config())?;
        serialize_proof_to_file::<Poseidon252MerkleHasher>(
            &cairo_proof,
            proof.into(),
            proof_format,
        )
        .map_err(|e| CairoProveError::ProofSerialization(format!("{:?}", e)))?;
    } else {
        let cairo_proof = prove(prover_input, secure_pcs_config())?;
        serialize_proof_to_file::<Blake2sMerkleHasher>(&cairo_proof, proof.into(), proof_format)
            .map_err(|e| CairoProveError::ProofSerialization(format!("{:?}", e)))?;
    }

    info!("Proof saved to: {:?}", proof);
    info!("Proof generation completed in {:.2?}", start.elapsed());
    Ok(())
}

fn handle_verify(proof: &Path, _with_pedersen: bool) -> Result<()> {
    info!("Verifying proof from: {:?}", proof);

    let proof_path = proof
        .to_str()
        .ok_or_else(|| CairoProveError::InvalidPath {
            path: proof.to_path_buf(),
        })?;

    let proof_str = std::fs::read_to_string(proof_path)?;

    // Try Blake2s first, then Poseidon252
    if let Ok(cairo_proof) =
        serde_json::from_str::<CairoProofForRustVerifier<Blake2sMerkleHasher>>(&proof_str)
    {
        verify_cairo::<Blake2sMerkleChannel>(cairo_proof)
            .map_err(|e| CairoProveError::Verification(format!("{:?}", e)))?;
    } else {
        let cairo_proof: CairoProofForRustVerifier<Poseidon252MerkleHasher> =
            serde_json::from_str(&proof_str)?;
        verify_cairo::<Poseidon252MerkleChannel>(cairo_proof)
            .map_err(|e| CairoProveError::Verification(format!("{:?}", e)))?;
    }

    info!("Verification successful");
    Ok(())
}

fn handle_prove_ml(
    verifier_executable: &Path,
    ml_proof: &Path,
    output: &Path,
    proof_format: ProofFormat,
    gpu: bool,
    poseidon: bool,
) -> Result<()> {
    info!(
        "Generating recursive proof for ML STARK proof: {:?}",
        ml_proof
    );
    if poseidon {
        info!("[Poseidon252] On-chain recursive path — proof will be ~19K felts.");
    }
    let start = Instant::now();

    // Load verifier executable
    let exec_str = verifier_executable
        .to_str()
        .ok_or_else(|| CairoProveError::InvalidPath {
            path: verifier_executable.to_path_buf(),
        })?;
    let exec_file = std::fs::File::open(exec_str)?;
    let executable: cairo_lang_executable::executable::Executable =
        serde_json::from_reader(exec_file)
            .map_err(|e| CairoProveError::ProofSerialization(format!("Failed to parse: {e}")))?;
    info!("Verifier executable loaded.");

    // Load proof arguments
    let args_file = std::fs::File::open(ml_proof)?;
    let as_vec: Vec<cairo_lang_utils::bigint::BigUintAsHex> = serde_json::from_reader(args_file)
        .map_err(|e| CairoProveError::ProofSerialization(format!("Failed to parse args: {e}")))?;
    let args: Vec<Arg> = as_vec
        .into_iter()
        .map(|v| Arg::Value(v.value.into()))
        .collect();
    info!("Proof arguments loaded: {} felt252 values.", args.len());

    // Execute verifier in Cairo VM
    let exec_start = Instant::now();
    let runner = execute(executable, args)?;
    info!("Verifier executed in {:.2?}.", exec_start.elapsed());

    // Generate recursive proof
    let prove_start = Instant::now();
    let prover_input = prover_input_from_runner(&runner)?;

    // Use recursive PCS config (fewer queries) for the Level 2 proof
    let pcs = if poseidon {
        recursive_pcs_config()
    } else {
        secure_pcs_config()
    };

    if poseidon {
        let cairo_proof = {
            #[cfg(feature = "cuda-runtime")]
            {
                if gpu {
                    info!("[GPU+Poseidon252] CUDA + native Poseidon channel.");
                    cairo_prove::prove::prove_gpu_poseidon(prover_input, pcs)?
                } else {
                    prove_poseidon(prover_input, pcs)?
                }
            }
            #[cfg(not(feature = "cuda-runtime"))]
            {
                let _ = gpu;
                prove_poseidon(prover_input, pcs)?
            }
        };
        info!(
            "Recursive STARK proof (Poseidon252) generated in {:.2?}.",
            prove_start.elapsed()
        );
        serialize_proof_to_file::<Poseidon252MerkleHasher>(
            &cairo_proof,
            output.into(),
            proof_format,
        )
        .map_err(|e| CairoProveError::ProofSerialization(format!("{:?}", e)))?;
    } else {
        let cairo_proof = {
            #[cfg(feature = "cuda-runtime")]
            {
                if gpu {
                    info!("[GPU] Using GpuBackend for STARK proving.");
                    cairo_prove::prove::prove_gpu(prover_input, pcs)?
                } else {
                    prove(prover_input, pcs)?
                }
            }
            #[cfg(not(feature = "cuda-runtime"))]
            {
                let _ = gpu;
                prove(prover_input, pcs)?
            }
        };
        info!(
            "Recursive STARK proof generated in {:.2?}.",
            prove_start.elapsed()
        );
        serialize_proof_to_file::<Blake2sMerkleHasher>(&cairo_proof, output.into(), proof_format)
            .map_err(|e| CairoProveError::ProofSerialization(format!("{:?}", e)))?;
    }

    info!(
        "Recursive proof saved to: {:?} ({:.2?} total)",
        output,
        start.elapsed()
    );
    Ok(())
}

fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Prove {
            target,
            proof,
            proof_format,
            poseidon,
            program_arguments,
        } => {
            handle_prove(&target, &proof, proof_format, poseidon, program_arguments)?;
        }
        Commands::Verify {
            proof,
            with_pedersen,
        } => {
            handle_verify(&proof, with_pedersen)?;
        }
        Commands::ProveMl {
            verifier_executable,
            ml_proof,
            output,
            proof_format,
            gpu,
            poseidon,
        } => {
            handle_prove_ml(
                &verifier_executable,
                &ml_proof,
                &output,
                proof_format,
                gpu,
                poseidon,
            )?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}
