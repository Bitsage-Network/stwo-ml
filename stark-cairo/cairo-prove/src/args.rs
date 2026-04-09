use std::path::PathBuf;

use cairo_air::utils::ProofFormat;
use cairo_lang_runner::Arg;
use cairo_lang_utils::bigint::BigUintAsHex;
use camino::Utf8PathBuf;
use clap::{Parser, Subcommand};
use num_bigint::BigInt;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generate a proof for a target file
    Prove {
        /// Path to the target file
        target: PathBuf,
        /// Path to the proof file
        proof: PathBuf,
        /// The format of the proof output.
        /// - json: Standard JSON format (default)
        /// - cairo_serde: Array of field elements serialized as hex strings, ex. `["0x1", "0x2"]`
        #[arg(long, value_enum, default_value_t = ProofFormat::Json)]
        proof_format: ProofFormat,
        /// Program arguments
        #[command(flatten)]
        program_arguments: ProgramArguments,
    },
    /// Verify a proof
    Verify {
        /// Path to the proof JSON file
        proof: PathBuf,
        /// Canonical trace, if Pedersen is included in the program.
        #[arg(short, long)]
        with_pedersen: bool,
    },
    /// Generate a recursive proof for an ML STARK proof.
    ///
    /// Takes a serialized ML proof (from stwo-ml), runs the ML verifier
    /// executable in the Cairo VM, and generates a compact recursive STARK
    /// proof that can be verified on-chain in a single transaction.
    ProveMl {
        /// Path to the compiled ML verifier executable JSON
        /// (from `scarb build --package obelysk_ml_verifier`).
        #[arg(long)]
        verifier_executable: PathBuf,
        /// Path to the ML proof arguments file (felt252 hex array from stwo-ml serializer).
        #[arg(long)]
        ml_proof: PathBuf,
        /// Path to write the output recursive proof.
        #[arg(long)]
        output: PathBuf,
        /// Proof output format.
        #[arg(long, value_enum, default_value_t = ProofFormat::Json)]
        proof_format: ProofFormat,
        /// Use GPU backend for STARK proving (requires cuda-runtime feature).
        #[arg(long, default_value = "false")]
        gpu: bool,
    },
}

#[derive(Parser, Debug, Clone)]
pub struct ProgramArguments {
    /// Serialized arguments to the executable function.
    #[arg(long, value_delimiter = ',')]
    pub arguments: Vec<BigInt>,

    /// Serialized arguments to the executable function from a file.
    #[arg(long, conflicts_with = "arguments")]
    pub arguments_file: Option<Utf8PathBuf>,
}
impl ProgramArguments {
    pub fn read_arguments(self) -> Vec<Arg> {
        if let Some(path) = self.arguments_file {
            let file = std::fs::File::open(&path).unwrap();
            let as_vec: Vec<BigUintAsHex> = serde_json::from_reader(file).unwrap();
            as_vec
                .into_iter()
                .map(|v| Arg::Value(v.value.into()))
                .collect()
        } else {
            self.arguments
                .iter()
                .map(|v| Arg::Value(v.into()))
                .collect()
        }
    }
}
