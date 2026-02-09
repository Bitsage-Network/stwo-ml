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
