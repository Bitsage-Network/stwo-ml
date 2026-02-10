use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use cairo_air::utils::{ProofFormat, serialize_proof_to_file};
use cairo_air::verifier::verify_cairo;
use cairo_air::{CairoProof, PreProcessedTraceVariant};
use cairo_lang_runner::Arg;
use cairo_prove::args::{Cli, Commands, ProgramArguments};
use cairo_prove::error::{CairoProveError, Result};
use cairo_prove::execute::execute;
use cairo_prove::prove::{prove, prover_input_from_runner};
use clap::Parser;
use log::{error, info};
use stwo_cairo_prover::stwo::core::fri::FriConfig;
use stwo_cairo_prover::stwo::core::pcs::PcsConfig;
use stwo_cairo_prover::stwo::core::vcs::blake2_merkle::{
    Blake2sMerkleChannel, Blake2sMerkleHasher,
};

fn execute_and_prove(
    target_path: &Path,
    args: Vec<Arg>,
    pcs_config: PcsConfig,
) -> Result<CairoProof<Blake2sMerkleHasher>> {
    // Read and parse executable
    let target_str = target_path
        .to_str()
        .ok_or_else(|| CairoProveError::InvalidPath {
            path: target_path.to_path_buf(),
        })?;

    let file = std::fs::File::open(target_str)?;
    let executable = serde_json::from_reader(file)?;

    // Execute
    let runner = execute(executable, args)?;

    // Prove
    let prover_input = prover_input_from_runner(&runner)?;
    prove(prover_input, pcs_config)
}

fn secure_pcs_config() -> PcsConfig {
    PcsConfig {
        pow_bits: 26,
        fri_config: FriConfig {
            log_last_layer_degree_bound: 0,
            log_blowup_factor: 1,
            n_queries: 70,
        },
    }
}

fn handle_prove(
    target: &Path,
    proof: &Path,
    proof_format: ProofFormat,
    args: ProgramArguments,
) -> Result<()> {
    info!("Generating proof for target: {:?}", target);
    let start = Instant::now();

    let cairo_proof = execute_and_prove(target, args.read_arguments(), secure_pcs_config())?;

    let elapsed = start.elapsed();

    serialize_proof_to_file::<Blake2sMerkleChannel>(&cairo_proof, proof.into(), proof_format)
        .map_err(|e| CairoProveError::ProofSerialization(format!("{:?}", e)))?;

    info!("Proof saved to: {:?}", proof);
    info!("Proof generation completed in {:.2?}", elapsed);
    Ok(())
}

fn handle_verify(proof: &Path, with_pedersen: bool) -> Result<()> {
    info!("Verifying proof from: {:?}", proof);

    let proof_path = proof
        .to_str()
        .ok_or_else(|| CairoProveError::InvalidPath {
            path: proof.to_path_buf(),
        })?;

    let proof_str = std::fs::read_to_string(proof_path)?;
    let cairo_proof = serde_json::from_str(&proof_str)?;

    let preprocessed_trace = match with_pedersen {
        true => PreProcessedTraceVariant::Canonical,
        false => PreProcessedTraceVariant::CanonicalWithoutPedersen,
    };

    verify_cairo::<Blake2sMerkleChannel>(cairo_proof, secure_pcs_config(), preprocessed_trace)
        .map_err(|e| CairoProveError::Verification(format!("{:?}", e)))?;

    info!("Verification successful");
    Ok(())
}

fn handle_prove_ml(
    verifier_executable: &Path,
    ml_proof: &Path,
    output: &Path,
    proof_format: ProofFormat,
    gpu: bool,
) -> Result<()> {
    info!("Generating recursive proof for ML STARK proof: {:?}", ml_proof);
    info!("Using ML verifier executable: {:?}", verifier_executable);
    if gpu {
        info!("GPU requested for recursive proving.");
        #[cfg(feature = "cuda-runtime")]
        info!("cuda-runtime feature enabled — GPU will be used for STARK proving.");
        #[cfg(not(feature = "cuda-runtime"))]
        info!("cuda-runtime feature NOT enabled — falling back to SimdBackend. \
               Build with --features cuda-runtime to enable GPU.");
    } else {
        info!("GPU backend: disabled");
    }
    let start = Instant::now();

    // Step 1: Load the ML verifier executable (compiled Cairo Sierra JSON)
    let exec_str = verifier_executable
        .to_str()
        .ok_or_else(|| CairoProveError::InvalidPath {
            path: verifier_executable.to_path_buf(),
        })?;
    let exec_file = std::fs::File::open(exec_str)?;
    let executable: cairo_lang_executable::executable::Executable =
        serde_json::from_reader(exec_file)
            .map_err(|e| CairoProveError::ProofSerialization(
                format!("Failed to parse ML verifier executable: {e}")
            ))?;
    info!("ML verifier executable loaded.");

    // Step 2: Load ML proof arguments (felt252 hex array)
    let args_file = std::fs::File::open(ml_proof)?;
    let as_vec: Vec<cairo_lang_utils::bigint::BigUintAsHex> =
        serde_json::from_reader(args_file)
            .map_err(|e| CairoProveError::ProofSerialization(
                format!("Failed to parse ML proof arguments: {e}")
            ))?;
    let args: Vec<Arg> = as_vec
        .into_iter()
        .map(|v| Arg::Value(v.value.into()))
        .collect();
    info!("ML proof arguments loaded: {} felt252 values.", args.len());

    // Step 3: Execute the ML verifier in Cairo VM
    let exec_start = Instant::now();
    let runner = execute(executable, args)?;
    info!("ML verifier executed in {:.2?}.", exec_start.elapsed());

    // Step 4: Generate recursive STARK proof
    let prove_start = Instant::now();
    let prover_input = prover_input_from_runner(&runner)?;
    let cairo_proof = prove(prover_input, secure_pcs_config())?;
    info!("Recursive STARK proof generated in {:.2?}.", prove_start.elapsed());

    // Step 5: Save the proof
    serialize_proof_to_file::<Blake2sMerkleChannel>(&cairo_proof, output.into(), proof_format)
        .map_err(|e| CairoProveError::ProofSerialization(format!("{:?}", e)))?;

    let elapsed = start.elapsed();
    info!("Recursive proof saved to: {:?} ({:.2?} total)", output, elapsed);
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
            program_arguments,
        } => {
            handle_prove(&target, &proof, proof_format, program_arguments)?;
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
        } => {
            handle_prove_ml(&verifier_executable, &ml_proof, &output, proof_format, gpu)?;
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use cairo_vm::Felt252;
    use num_bigint::BigInt;

    use super::*;

    #[test]
    fn test_e2e() {
        let target_path = Path::new("./example/target/release/example.executable.json");
        let args = vec![Arg::Value(Felt252::from(BigInt::from(100)))];
        let proof = execute_and_prove(target_path, args, PcsConfig::default())
            .expect("Proof generation failed");
        let pcs_config = PcsConfig::default();
        let preprocessed_trace = PreProcessedTraceVariant::CanonicalWithoutPedersen;
        let result = verify_cairo::<Blake2sMerkleChannel>(proof, pcs_config, preprocessed_trace);
        assert!(result.is_ok());
    }
}
