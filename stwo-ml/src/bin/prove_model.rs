//! `prove-model` CLI binary.
//!
//! Standalone tool for ZKML proving and verifiable inference auditing.
//!
//! **Subcommands:**
//!
//! ```text
//! # Prove a model (default):
//! prove-model prove \
//!   --model model.onnx \
//!   --input input.json \
//!   --output proof.json
//!
//! # Audit inferences over a time window:
//! prove-model audit \
//!   --log-dir ~/.obelysk/logs/qwen3-14b \
//!   --model-dir /path/to/qwen3-14b \
//!   --start "1h ago" --end now \
//!   --evaluate --submit
//!
//! # Legacy (no subcommand = prove):
//! prove-model --model model.onnx --input input.json
//! ```

#![feature(portable_simd)]

use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::{Parser, Subcommand};
use starknet_ff::FieldElement;
use stwo::core::fields::m31::M31;
#[cfg(feature = "cuda-runtime")]
use stwo::prover::backend::gpu::cuda_executor::{
    get_cuda_executor, upload_poseidon252_round_constants,
};

use stwo_ml::aggregation::compute_io_commitment;
use stwo_ml::cairo_serde::{
    serialize_ml_proof_for_recursive, serialize_ml_proof_to_file, DirectProofMetadata,
    MLClaimMetadata,
};
use stwo_ml::compiler::inspect::summarize_model;
use stwo_ml::compiler::onnx::{load_onnx, OnnxModel};
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::gadgets::quantize::{quantize_tensor, QuantStrategy};
use stwo_ml::json_serde;
use stwo_ml::starknet::build_starknet_proof_direct;
use stwo_ml::tee::{detect_tee_capability, SecurityLevel};

/// Output format for the proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    CairoSerde,
    Json,
    Direct,
    /// Full ML GKR model proof artifact.
    /// Uses `prove_model_pure_gkr_auto` → `build_gkr_serializable_proof` pipeline.
    /// Output: JSON with GKR calldata, IO calldata, weight openings/claims, and
    /// submission readiness metadata.
    MlGkr,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cairo_serde" | "cairo-serde" | "felt" => Ok(OutputFormat::CairoSerde),
            "json" => Ok(OutputFormat::Json),
            "direct" => Ok(OutputFormat::Direct),
            "ml_gkr" | "ml-gkr" | "gkr" => Ok(OutputFormat::MlGkr),
            _ => Err(format!(
                "unknown format '{s}', expected 'cairo_serde', 'json', 'direct', or 'ml_gkr'"
            )),
        }
    }
}

/// stwo-ml prove-model: ZKML prover and verifiable inference audit CLI.
#[derive(Parser, Debug)]
#[command(name = "prove-model", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    // ── Legacy flat flags (backward compat: no subcommand = prove) ────────
    /// Path to ONNX model file.
    #[arg(long, group = "model_source")]
    model: Option<PathBuf>,

    /// Path to a HuggingFace model directory (with config.json + *.safetensors).
    #[arg(long, group = "model_source")]
    model_dir: Option<PathBuf>,

    /// Number of transformer layers to prove (default: all from config.json).
    #[arg(long)]
    layers: Option<usize>,

    /// Path to JSON input file (array of f32 values).
    #[arg(long)]
    input: Option<PathBuf>,

    /// Path to write output proof.
    #[arg(long, default_value = "proof.json")]
    output: PathBuf,

    /// Output format: 'cairo_serde', 'json', 'direct', or 'ml_gkr'.
    #[arg(long, default_value = "cairo_serde")]
    format: OutputFormat,

    /// Model ID (hex or decimal) for the on-chain claim.
    #[arg(long, default_value = "0x1")]
    model_id: String,

    /// Use GPU backend if available.
    #[arg(long)]
    gpu: bool,

    /// Distribute proving across all available GPUs.
    #[arg(long)]
    multi_gpu: bool,

    /// Memory budget per chunk in GB (used with --multi-gpu).
    #[arg(long, default_value = "16")]
    chunk_budget_gb: f64,

    /// Security level: 'auto', 'tee', or 'zk-only'.
    #[arg(long, default_value = "auto")]
    security: SecurityLevel,

    /// Print model summary and exit (no proving).
    #[arg(long)]
    inspect: bool,

    /// Validate model directory only.
    #[arg(long)]
    validate: bool,

    /// Use STWO-native GKR for LogUp component verification.
    #[arg(long)]
    gkr: bool,

    /// Channel salt for Fiat-Shamir (optional).
    #[arg(long)]
    salt: Option<u64>,

    /// Skip weight commitment computation.
    #[arg(long)]
    skip_commitment: bool,

    /// Verify an existing proof file and exit.
    #[arg(long)]
    verify_proof: Option<PathBuf>,

    /// Submit the GKR proof on-chain via `sncast invoke`.
    #[arg(long)]
    submit_gkr: bool,

    /// Contract address for --submit-gkr (hex).
    #[arg(
        long,
        default_value = "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005"
    )]
    contract: String,

    /// sncast account name for --submit-gkr.
    #[arg(long, default_value = "deployer")]
    account: String,

    /// Starknet network for --submit-gkr.
    #[arg(long, default_value = "sepolia")]
    network: String,

    /// Max fee in ETH for --submit-gkr transactions.
    #[arg(long, default_value = "0.05")]
    max_fee: String,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Prove a single inference (default behavior).
    Prove,
    /// Audit inferences over a time window.
    ///
    /// Loads an inference log, batch-proves all inferences in the window,
    /// runs semantic evaluation, and produces an audit report.
    ///
    /// Examples:
    ///   prove-model audit --log-dir ~/.obelysk/logs/qwen3 --model-dir ./qwen3 --start "1h ago"
    ///   prove-model audit --log-dir ./logs --model model.onnx --evaluate --submit
    Audit(AuditCmd),
    /// Retrieve and decrypt an encrypted audit report from Arweave.
    ///
    /// Fetches the encrypted blob by Arweave TX ID, decrypts it with the
    /// provided private key, and writes the plaintext report to a file.
    ///
    /// Examples:
    ///   prove-model retrieve --tx-id abc123 --privkey 0xdead... --output report.json
    ///   prove-model retrieve --tx-id abc123 --privkey 0xdead... --encryption aes
    Retrieve(RetrieveCmd),
    /// Manage VM31 wallets (generate, show, import, export viewing key).
    Wallet(WalletCmd),
    /// Shield tokens into the VM31 privacy pool.
    Deposit(DepositCmd),
    /// Unshield tokens from the VM31 privacy pool.
    Withdraw(WithdrawCmd),
    /// Private shielded transfer within the pool.
    Transfer(TransferCmd),
    /// Prove a batch of privacy transactions from a file.
    Batch(BatchCmd),
    /// Query the on-chain pool state.
    PoolStatus(PoolStatusCmd),
    /// Scan for incoming notes.
    Scan(ScanCmd),
    /// Capture inference logs by running forward passes through the prover.
    ///
    /// Loads the model, runs N forward passes through `execute_forward_pass()`
    /// (the same code path the audit verifier checks), and records each via
    /// `CaptureHook`. This produces a chain-linked inference log suitable for
    /// auditing.
    ///
    /// Examples:
    ///   prove-model capture --model-dir ./qwen3-14b --log-dir /tmp/logs --count 10
    ///   prove-model capture --model model.onnx --log-dir /tmp/logs --count 5 --skip-commitment
    Capture(CaptureCmd),
}

/// CLI arguments for the `audit` subcommand.
#[derive(Parser, Debug)]
struct AuditCmd {
    /// Directory containing the inference log (log.jsonl + matrices.bin + meta.json).
    #[arg(long)]
    log_dir: PathBuf,

    /// Path to ONNX model file (for replay verification).
    #[arg(long, group = "audit_model")]
    model: Option<PathBuf>,

    /// Path to HuggingFace model directory (for replay verification).
    #[arg(long, group = "audit_model")]
    model_dir: Option<PathBuf>,

    /// Number of transformer layers (only with --model-dir).
    #[arg(long)]
    layers: Option<usize>,

    /// Start of audit window. Supports:
    ///   - Relative: "1h ago", "30m ago", "2d ago"
    ///   - Unix timestamp (nanoseconds): "1707000000000000000"
    ///   - "all" (from beginning)
    #[arg(long, default_value = "all")]
    start: String,

    /// End of audit window. Supports:
    ///   - "now" (current time)
    ///   - Relative: "30m ago"
    ///   - Unix timestamp (nanoseconds)
    #[arg(long, default_value = "now")]
    end: String,

    /// Model ID (hex felt252) to audit.
    #[arg(long)]
    model_id: Option<String>,

    /// Human-readable model name (used in report metadata).
    #[arg(long)]
    model_name: Option<String>,

    /// Run semantic evaluation (deterministic checks + forward-pass scoring).
    #[arg(long)]
    evaluate: bool,

    /// Prove evaluation forward passes (adds ZK proofs for eval outputs).
    #[arg(long)]
    prove_evals: bool,

    /// Maximum number of inferences to prove (0 = all in window).
    #[arg(long, default_value = "0")]
    max_inferences: usize,

    /// Privacy tier: "public" (default), "private", "selective".
    #[arg(long, default_value = "public")]
    privacy: String,

    /// Serialize on-chain calldata and write alongside the report.
    #[arg(long)]
    submit: bool,

    /// Contract address for on-chain submission.
    #[arg(
        long,
        default_value = "0x03f937cb00db86933c94a680ce2cb2df3296e7680df3547c610aa929ffba860c"
    )]
    contract: String,

    /// Starknet network for on-chain submission.
    #[arg(long, default_value = "sepolia")]
    network: String,

    /// Submitter account address (or set STARKNET_ACCOUNT env var).
    #[arg(long)]
    account: Option<String>,

    /// Submitter private key (or set STARKNET_PRIVATE_KEY env var).
    /// Prefer env var over CLI to avoid leaking keys in process listings.
    #[arg(long)]
    private_key: Option<String>,

    /// Path to write the audit report JSON.
    #[arg(long, default_value = "audit_report.json")]
    output: PathBuf,

    /// Use GPU backend for proving.
    #[arg(long)]
    gpu: bool,

    /// Dry-run: prove + evaluate + report, but skip encryption and on-chain.
    #[arg(long)]
    dry_run: bool,

    /// Arweave gateway URL.
    #[arg(long, default_value = "https://arweave.net")]
    arweave_gateway: String,

    /// Arweave bundler URL.
    #[arg(long, default_value = "https://node1.irys.xyz")]
    arweave_bundler: String,

    /// Encryption mode: "poseidon2" (default, production), "aes", "noop" (test only), "none".
    #[arg(long, default_value = "poseidon2")]
    encryption: String,

    /// Owner public key hex (for encryption -- who can decrypt the report).
    #[arg(long)]
    owner_pubkey: Option<String>,

    /// Irys API token for authenticated Arweave uploads (or set IRYS_TOKEN env var).
    #[arg(long, env = "IRYS_TOKEN")]
    irys_token: Option<String>,

    /// Proof mode: "direct" (fast, aggregated STARK — default), "gkr" (full GKR with
    /// weight openings), or "legacy" (Blake2s, not on-chain verifiable).
    #[arg(long, default_value = "direct")]
    mode: String,
}

/// CLI arguments for the `retrieve` subcommand.
#[derive(Parser, Debug)]
struct RetrieveCmd {
    /// Arweave transaction ID of the encrypted audit report.
    #[arg(long)]
    tx_id: String,

    /// Private key hex for decryption.
    #[arg(long)]
    privkey: String,

    /// Recipient address hex (your Starknet address).
    #[arg(long)]
    address: Option<String>,

    /// Encryption mode: "poseidon2" (default, production), "aes", or "noop" (test only).
    #[arg(long, default_value = "poseidon2")]
    encryption: String,

    /// Arweave gateway URL.
    #[arg(long, default_value = "https://arweave.net")]
    arweave_gateway: String,

    /// Path to write the decrypted audit report JSON.
    #[arg(long, default_value = "audit_report.json")]
    output: PathBuf,
}

// ─── Privacy Subcommand Structs ──────────────────────────────────────────

/// CLI arguments for the `wallet` subcommand.
#[derive(Parser, Debug)]
struct WalletCmd {
    /// Wallet action: generate, show, import, export-viewing-key.
    #[arg(default_value = "show")]
    action: String,

    /// Path to wallet file.
    #[arg(long, default_value = "~/.vm31/wallet.json")]
    wallet: String,

    /// Password for encrypted wallets.
    #[arg(long)]
    password: Option<String>,

    /// Output path (for generate).
    #[arg(long)]
    output: Option<String>,

    /// Hex spending key (for import).
    #[arg(long)]
    spending_key: Option<String>,
}

/// CLI arguments for the `deposit` subcommand.
#[derive(Parser, Debug)]
struct DepositCmd {
    /// Amount to deposit (integer, smallest unit).
    #[arg(long)]
    amount: u64,

    /// Asset ID (0 = STRK, 1 = ETH, etc.).
    #[arg(long, default_value = "0")]
    asset: u32,

    /// Recipient public key hex (default: self).
    #[arg(long)]
    recipient: Option<String>,

    /// Path to wallet file.
    #[arg(long, default_value = "~/.vm31/wallet.json")]
    wallet: String,

    /// Password for encrypted wallets.
    #[arg(long)]
    password: Option<String>,

    /// Path to write proof output.
    #[arg(long, default_value = "deposit_proof.json")]
    output: PathBuf,

    /// Submit proof on-chain after proving.
    #[arg(long)]
    submit: bool,

    /// Starknet network.
    #[arg(long, default_value = "sepolia")]
    priv_network: String,

    /// sncast account name.
    #[arg(long, default_value = "deployer")]
    priv_account: String,
}

/// CLI arguments for the `withdraw` subcommand.
#[derive(Parser, Debug)]
struct WithdrawCmd {
    /// Amount to withdraw.
    #[arg(long)]
    amount: u64,

    /// Asset ID.
    #[arg(long, default_value = "0")]
    asset: u32,

    /// Path to wallet file.
    #[arg(long, default_value = "~/.vm31/wallet.json")]
    wallet: String,

    /// Password for encrypted wallets.
    #[arg(long)]
    password: Option<String>,

    /// Path to write proof output.
    #[arg(long, default_value = "withdraw_proof.json")]
    output: PathBuf,

    /// Submit proof on-chain after proving.
    #[arg(long)]
    submit: bool,

    /// Starknet network.
    #[arg(long, default_value = "sepolia")]
    priv_network: String,

    /// Pool contract address (override env: VM31_POOL_ADDRESS).
    #[arg(long)]
    pool_contract: Option<String>,

    /// Payout recipient Starknet address for proof-bound withdrawal binding.
    #[arg(long)]
    payout_recipient: Option<String>,

    /// Credit recipient Starknet address for proof-bound withdrawal binding.
    /// Defaults to payout recipient if omitted.
    #[arg(long)]
    credit_recipient: Option<String>,

    /// sncast account name.
    #[arg(long, default_value = "deployer")]
    priv_account: String,
}

/// CLI arguments for the `transfer` subcommand.
#[derive(Parser, Debug)]
struct TransferCmd {
    /// Amount to transfer.
    #[arg(long)]
    amount: u64,

    /// Asset ID.
    #[arg(long, default_value = "0")]
    asset: u32,

    /// Recipient public key hex.
    #[arg(long)]
    to: String,

    /// Recipient viewing key hex (required for memo encryption).
    /// The recipient must share this out-of-band for you to send them notes.
    #[arg(long)]
    to_viewing_key: String,

    /// Path to wallet file.
    #[arg(long, default_value = "~/.vm31/wallet.json")]
    wallet: String,

    /// Password for encrypted wallets.
    #[arg(long)]
    password: Option<String>,

    /// Path to write proof output.
    #[arg(long, default_value = "transfer_proof.json")]
    output: PathBuf,

    /// Submit proof on-chain after proving.
    #[arg(long)]
    submit: bool,

    /// Starknet network.
    #[arg(long, default_value = "sepolia")]
    priv_network: String,

    /// Pool contract address (override env: VM31_POOL_ADDRESS).
    #[arg(long)]
    pool_contract: Option<String>,

    /// sncast account name.
    #[arg(long, default_value = "deployer")]
    priv_account: String,
}

/// CLI arguments for the `batch` subcommand.
#[derive(Parser, Debug)]
struct BatchCmd {
    /// Path to JSON transaction file.
    #[arg(long)]
    tx_file: PathBuf,

    /// Path to wallet file.
    #[arg(long, default_value = "~/.vm31/wallet.json")]
    wallet: String,

    /// Password for encrypted wallets.
    #[arg(long)]
    password: Option<String>,

    /// Path to write proof output.
    #[arg(long, default_value = "batch_proof.json")]
    output: PathBuf,

    /// Submit proof on-chain after proving.
    #[arg(long)]
    submit: bool,

    /// Starknet network.
    #[arg(long, default_value = "sepolia")]
    priv_network: String,

    /// sncast account name.
    #[arg(long, default_value = "deployer")]
    priv_account: String,
}

/// CLI arguments for the `pool-status` subcommand.
#[derive(Parser, Debug)]
struct PoolStatusCmd {
    /// Starknet network.
    #[arg(long, default_value = "sepolia")]
    priv_network: String,

    /// Pool contract address (override env: VM31_POOL_ADDRESS).
    #[arg(long)]
    pool_contract: Option<String>,

    /// Check if a specific nullifier is spent (hex).
    #[arg(long)]
    check_nullifier: Option<String>,

    /// Check if a specific root is known (hex).
    #[arg(long)]
    check_root: Option<String>,
}

/// CLI arguments for the `scan` subcommand.
#[derive(Parser, Debug)]
struct ScanCmd {
    /// Path to wallet file.
    #[arg(long, default_value = "~/.vm31/wallet.json")]
    wallet: String,

    /// Password for encrypted wallets.
    #[arg(long)]
    password: Option<String>,

    /// Starting Merkle index to scan from.
    #[arg(long, default_value = "0")]
    from_index: usize,

    /// Starknet network.
    #[arg(long, default_value = "sepolia")]
    priv_network: String,

    /// Pool contract address (override env: VM31_POOL_ADDRESS).
    #[arg(long)]
    pool_contract: Option<String>,
}

/// CLI arguments for the `capture` subcommand.
#[derive(Parser, Debug)]
struct CaptureCmd {
    /// Path to HuggingFace model directory.
    #[arg(long, group = "capture_model")]
    model_dir: Option<PathBuf>,

    /// Path to ONNX model file.
    #[arg(long, group = "capture_model")]
    model: Option<PathBuf>,

    /// Number of transformer layers (only with --model-dir).
    #[arg(long)]
    layers: Option<usize>,

    /// Directory to write the inference log.
    #[arg(long)]
    log_dir: PathBuf,

    /// Number of forward passes to capture.
    #[arg(long, default_value = "10")]
    count: usize,

    /// Model ID (hex felt252) for the log.
    #[arg(long, default_value = "0x1")]
    model_id: String,

    /// Path to JSON input file (if provided, uses this instead of diverse generation).
    #[arg(long)]
    input: Option<PathBuf>,

    /// Skip weight commitment computation (faster, but log won't have real commitment).
    #[arg(long)]
    skip_commitment: bool,

    /// Human-readable model name for the log metadata.
    #[arg(long)]
    model_name: Option<String>,
}

fn parse_model_id(s: &str) -> FieldElement {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        FieldElement::from_hex_be(hex).unwrap_or_else(|_| {
            eprintln!("Error: invalid hex model-id '{s}'");
            process::exit(1);
        })
    } else {
        let val: u64 = s.parse().unwrap_or_else(|_| {
            eprintln!("Error: invalid model-id '{s}', expected hex (0x...) or decimal");
            process::exit(1);
        });
        FieldElement::from(val)
    }
}

fn load_input_json(path: &PathBuf) -> Vec<f32> {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error: cannot read input file '{}': {e}", path.display());
        process::exit(1);
    });
    serde_json::from_str::<Vec<f32>>(&contents).unwrap_or_else(|e| {
        eprintln!("Error: invalid JSON input (expected array of f32): {e}");
        process::exit(1);
    })
}

/// Quantize f32 input values to M31 field elements.
fn quantize_input(values: &[f32], rows: usize, cols: usize) -> M31Matrix {
    let (quantized, _params) = quantize_tensor(values, QuantStrategy::Symmetric8);
    let mut matrix = M31Matrix::new(rows, cols);
    for (i, &v) in quantized.iter().enumerate().take(rows * cols) {
        matrix.data[i] = v;
    }
    matrix
}

/// Generate a deterministic random input for a given shape.
fn generate_random_input(rows: usize, cols: usize) -> M31Matrix {
    let mut matrix = M31Matrix::new(rows, cols);
    for i in 0..(rows * cols) {
        // Simple deterministic pseudo-random values
        matrix.data[i] = M31::from((i as u32 * 7 + 13) % (1 << 20));
    }
    matrix
}

fn load_model(cli: &Cli) -> OnnxModel {
    if let Some(ref model_dir) = cli.model_dir {
        // HuggingFace directory mode — validation is built into load_hf_model
        eprintln!("Loading HuggingFace model from: {}", model_dir.display());
        stwo_ml::compiler::hf_loader::load_hf_model(model_dir, cli.layers).unwrap_or_else(|e| {
            eprintln!("Error loading model directory: {e}");
            process::exit(1);
        })
    } else if let Some(ref model_path) = cli.model {
        // ONNX mode
        eprintln!("Loading ONNX model: {}", model_path.display());
        load_onnx(model_path).unwrap_or_else(|e| {
            eprintln!("Error loading ONNX model: {e}");
            process::exit(1);
        })
    } else {
        eprintln!(
            "Error: specify either --model (ONNX file) or --model-dir (HuggingFace directory)"
        );
        process::exit(1);
    }
}

fn main() {
    let cli = Cli::parse();

    // Dispatch to subcommands if specified
    match cli.command {
        Some(Command::Audit(ref audit_cmd)) => {
            run_audit_command(audit_cmd, &cli);
            return;
        }
        Some(Command::Retrieve(ref retrieve_cmd)) => {
            run_retrieve_command(retrieve_cmd);
            return;
        }
        Some(Command::Wallet(ref cmd)) => {
            run_wallet_command(cmd);
            return;
        }
        Some(Command::Deposit(ref cmd)) => {
            run_deposit_command(cmd);
            return;
        }
        Some(Command::Withdraw(ref cmd)) => {
            run_withdraw_command(cmd);
            return;
        }
        Some(Command::Transfer(ref cmd)) => {
            run_transfer_command(cmd);
            return;
        }
        Some(Command::Batch(ref cmd)) => {
            run_batch_command(cmd);
            return;
        }
        Some(Command::PoolStatus(ref cmd)) => {
            run_pool_status_command(cmd);
            return;
        }
        Some(Command::Scan(ref cmd)) => {
            run_scan_command(cmd);
            return;
        }
        Some(Command::Capture(ref cmd)) => {
            run_capture_command(cmd);
            return;
        }
        _ => {}
    }

    // --verify-proof: verify an existing proof file and exit
    if let Some(ref proof_path) = cli.verify_proof {
        eprintln!("Verifying proof: {}", proof_path.display());
        let contents = std::fs::read_to_string(proof_path).unwrap_or_else(|e| {
            eprintln!(
                "Error: cannot read proof file '{}': {e}",
                proof_path.display()
            );
            process::exit(1);
        });

        let proof_json: serde_json::Value = serde_json::from_str(&contents).unwrap_or_else(|e| {
            eprintln!("Error: invalid JSON in proof file: {e}");
            process::exit(1);
        });

        let format = proof_json
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match format {
            "ml_gkr" => {
                // Verify GKR proof structure
                let calldata = proof_json.get("gkr_calldata").and_then(|v| v.as_array());
                let io = proof_json.get("io_calldata").and_then(|v| v.as_array());
                match (calldata, io) {
                    (Some(c), Some(i)) if !c.is_empty() && !i.is_empty() => {
                        eprintln!("  format: ml_gkr");
                        eprintln!("  gkr_calldata: {} felts", c.len());
                        eprintln!("  io_calldata: {} felts", i.len());
                        eprintln!("Proof structure valid.");
                        process::exit(0);
                    }
                    _ => {
                        eprintln!("Error: GKR proof missing or empty calldata");
                        process::exit(1);
                    }
                }
            }
            _ => {
                // Generic: check it has expected top-level fields
                let has_matmul = proof_json.get("batched_calldata").is_some()
                    || proof_json.get("gkr_calldata").is_some()
                    || proof_json.get("model_id").is_some();
                if has_matmul {
                    eprintln!(
                        "  format: {}",
                        if format.is_empty() { "unknown" } else { format }
                    );
                    eprintln!("Proof structure valid.");
                    process::exit(0);
                } else {
                    eprintln!("Error: proof file does not contain recognized proof fields");
                    process::exit(1);
                }
            }
        }
    }

    // --validate: run validation only and exit
    if cli.validate {
        if let Some(ref model_dir) = cli.model_dir {
            let report =
                stwo_ml::compiler::hf_loader::validate_model_directory(model_dir, cli.layers);
            eprintln!();
            eprintln!("  ── Model Validation ──");
            eprintln!("{}", report.format_report());
            eprintln!();

            if report.passed() {
                eprintln!("Model is valid and ready for proving.");
                process::exit(0);
            } else {
                eprintln!("Model validation FAILED. Fix the issues above.");
                process::exit(1);
            }
        } else {
            eprintln!("--validate requires --model-dir");
            process::exit(1);
        }
    }

    let model = load_model(&cli);

    // --inspect: print summary and exit
    if cli.inspect {
        let summary = summarize_model(&model);
        println!("{summary}");
        return;
    }

    // Build input
    let input = if let Some(ref input_path) = cli.input {
        // Load from JSON
        eprintln!("Loading input: {}", input_path.display());
        let input_f32 = load_input_json(input_path);
        let (in_rows, in_cols) = model.input_shape;
        let expected_len = in_rows * in_cols;
        if input_f32.len() != expected_len {
            eprintln!(
                "Error: input has {} values but model expects {} ({} x {})",
                input_f32.len(),
                expected_len,
                in_rows,
                in_cols,
            );
            process::exit(1);
        }
        quantize_input(&input_f32, in_rows, in_cols)
    } else {
        // Generate random input for --model-dir mode
        let (in_rows, in_cols) = model.input_shape;
        eprintln!("Generating random input: {} x {}", in_rows, in_cols);
        generate_random_input(in_rows, in_cols)
    };

    // Detect TEE capability and display status
    let tee_cap = detect_tee_capability();
    let resolved = cli.security.resolve();
    eprintln!("Security: {} (resolved: {:?})", cli.security, resolved);
    eprintln!("TEE: {}", tee_cap);

    if matches!(cli.security, SecurityLevel::ZkPlusTee) && !tee_cap.cc_active {
        eprintln!(
            "Error: --security tee requires NVIDIA H100/H200/B200 with CC-On firmware.\n  {}",
            tee_cap.status_message
        );
        process::exit(1);
    }

    // Pre-compute values needed by both proving and commitment paths
    let model_id = parse_model_id(&cli.model_id);
    let activation_type = model
        .graph
        .nodes
        .iter()
        .find_map(|n| match &n.op {
            stwo_ml::compiler::graph::GraphOp::Activation {
                activation_type, ..
            } => Some(*activation_type as u8),
            _ => None,
        })
        .unwrap_or(0);

    // Check weight commitment cache BEFORE entering the parallel scope.
    // If cached, no background thread needed.
    let cached_commitment = if cli.skip_commitment {
        Some(FieldElement::ZERO)
    } else {
        check_commitment_cache(cli.model_dir.as_deref())
    };

    if cached_commitment.is_some() && !cli.skip_commitment {
        eprintln!("Weight commitment loaded from cache (fingerprint validated)");
    } else if cli.skip_commitment {
        eprintln!("Skipping weight commitment (--skip-commitment)");
    }

    // Pipeline: run proving and weight commitment in parallel via std::thread::scope.
    // Both only need &model.weights (shared ref), so this is safe.
    // Commitment prefers GPU Poseidon hash-many when available, with strict/harden
    // flags to fail closed or cross-check against CPU.
    let use_gpu = cli.gpu || cli.multi_gpu;

    eprintln!(
        "Proving {} layers (gpu={}, multi_gpu={}, security={})",
        model.graph.num_layers(),
        use_gpu,
        cli.multi_gpu,
        cli.security,
    );

    // Multi-GPU: print device discovery info
    #[cfg(feature = "multi-gpu")]
    if cli.multi_gpu {
        let devices = stwo_ml::multi_gpu::discover_devices();
        if devices.is_empty() {
            eprintln!("Error: --multi-gpu specified but no GPUs found");
            process::exit(1);
        }
        eprintln!("Multi-GPU: {} device(s) detected", devices.len());
        for d in &devices {
            eprintln!(
                "  GPU {}: {} ({:.1} GB, SM {}.{}, {} SMs)",
                d.ordinal,
                d.name,
                d.total_memory as f64 / 1e9,
                d.compute_capability.0,
                d.compute_capability.1,
                d.sm_count,
            );
        }
    }

    let t0 = Instant::now();

    let run_proof = || {
        if cli.format == OutputFormat::MlGkr {
            // Full ML GKR pipeline: all layers via GKR sumcheck (no individual matmul proofs)
            eprintln!("Using full ML GKR pipeline (--format ml_gkr)");
            stwo_ml::aggregation::prove_model_pure_gkr_auto(&model.graph, &input, &model.weights)
        } else if cli.multi_gpu {
            // Multi-GPU path: chunk model and distribute across GPUs
            #[cfg(feature = "multi-gpu")]
            {
                let memory_budget = (cli.chunk_budget_gb * 1e9) as usize;
                let (chunks, metrics) =
                    stwo_ml::compiler::chunked::prove_model_chunked_multi_gpu_with_metrics(
                        &model.graph,
                        &input,
                        &model.weights,
                        memory_budget,
                    )
                    .map_err(|e| {
                        stwo_ml::compiler::prove::ModelError::ProvingError {
                            layer: 0,
                            message: format!("Multi-GPU chunked proving: {e}"),
                        }
                    })?;

                eprintln!(
                    "Multi-GPU proving: {:.2}s total, {} chunks across {} devices",
                    metrics.total_elapsed.as_secs_f64(),
                    chunks.len(),
                    metrics.device_stats.len(),
                );
                for stat in &metrics.device_stats {
                    eprintln!(
                        "  GPU {}: {} chunks, {} matmuls, {:.2}s",
                        stat.device_id,
                        stat.chunks_proven.len(),
                        stat.matmuls_proven,
                        stat.elapsed.as_secs_f64(),
                    );
                }

                stwo_ml::compiler::chunked::compose_chunk_proofs_auto(
                    &chunks,
                    &model.graph,
                    &input,
                    &model.weights,
                )
                .map_err(|e| stwo_ml::compiler::prove::ModelError::ProvingError {
                    layer: 0,
                    message: format!("Chunk composition: {e}"),
                })
            }
            #[cfg(not(feature = "multi-gpu"))]
            {
                eprintln!("Error: --multi-gpu requires the 'multi-gpu' feature");
                process::exit(1);
            }
        } else if cli.gkr {
            // LogUp GKR pipeline: standard pipeline + STWO-native GKR for LogUp components
            eprintln!("Using STWO-native GKR for LogUp verification");
            if use_gpu {
                stwo_ml::aggregation::prove_model_aggregated_onchain_logup_gkr_auto(
                    &model.graph,
                    &input,
                    &model.weights,
                )
            } else {
                stwo_ml::aggregation::prove_model_aggregated_onchain_logup_gkr(
                    &model.graph,
                    &input,
                    &model.weights,
                )
            }
        } else if use_gpu {
            stwo_ml::aggregation::prove_model_aggregated_onchain_auto(
                &model.graph,
                &input,
                &model.weights,
            )
        } else {
            stwo_ml::aggregation::prove_model_aggregated_onchain(
                &model.graph,
                &input,
                &model.weights,
            )
        }
    };

    let allow_parallel_gpu_commit = {
        #[cfg(feature = "cuda-runtime")]
        {
            gpu_commit_flag_enabled("STWO_PARALLEL_GPU_COMMIT")
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            false
        }
    };

    // On single-GPU runs, do GPU commitment before proving by default.
    // Parallel overlap made sense when commitment was CPU-bound, but now both
    // proving and commitment are GPU-heavy and can contend on one device.
    let serialize_gpu_commit =
        cached_commitment.is_none() && use_gpu && !cli.multi_gpu && !allow_parallel_gpu_commit;

    let (proof_result, weight_commitment, commit_elapsed) = if serialize_gpu_commit {
        eprintln!(
            "[BG] Single-GPU mode: running weight commitment before proving to avoid GPU contention."
        );
        eprintln!("[BG] Set STWO_PARALLEL_GPU_COMMIT=1 to force overlapping commitment + proving.");
        let t_commit = Instant::now();
        let commitment = compute_weight_commitment(&model.weights, cli.model_dir.as_deref());
        let commit_time = t_commit.elapsed();
        let proof = run_proof();
        (proof, commitment, commit_time)
    } else {
        std::thread::scope(|s| {
            // Spawn weight commitment on background thread if not cached
            let commit_handle = if cached_commitment.is_none() {
                Some(s.spawn(|| {
                    let t_commit = Instant::now();
                    let commitment =
                        compute_weight_commitment(&model.weights, cli.model_dir.as_deref());
                    (commitment, t_commit.elapsed())
                }))
            } else {
                None
            };

            let proof = run_proof();

            // Collect commitment result
            let (commitment, commit_time) = match (cached_commitment, commit_handle) {
                (Some(c), _) => (c, std::time::Duration::ZERO),
                (None, Some(handle)) => {
                    let (c, elapsed) = handle.join().expect("commitment thread panicked");
                    (c, elapsed)
                }
                _ => (FieldElement::ZERO, std::time::Duration::ZERO),
            };

            (proof, commitment, commit_time)
        })
    };

    let prove_elapsed = t0.elapsed();

    let proof = proof_result.unwrap_or_else(|e| {
        eprintln!("Error: proving failed: {e}");
        process::exit(1);
    });

    if !commit_elapsed.is_zero() {
        eprintln!(
            "Weight commitment computed in {:.2}s (pipelined with proving — zero added latency)",
            commit_elapsed.as_secs_f64(),
        );
    }

    // Generate TEE attestation if applicable
    let tee_attestation = if matches!(resolved, stwo_ml::tee::ResolvedSecurityLevel::ZkPlusTee) {
        eprintln!("Generating TEE attestation...");
        match stwo_ml::tee::TeeModelProver::with_security(cli.security) {
            Ok(tee_prover) if tee_prover.is_tee() => {
                eprintln!(
                    "  TEE: {} (hw={}, secboot={}, dbg={})",
                    tee_prover.attestation.device_id,
                    tee_prover.attestation.hw_model,
                    tee_prover.attestation.secure_boot,
                    tee_prover.attestation.debug_status,
                );
                Some(tee_prover.attestation.clone())
            }
            Ok(_) => None,
            Err(e) => {
                eprintln!("Warning: TEE attestation failed: {e}");
                None
            }
        }
    } else {
        None
    };

    eprintln!("Proving completed in {:.2}s", prove_elapsed.as_secs_f64());

    // Build metadata
    let io_commitment = compute_io_commitment(&input, &proof.execution.output);

    // Compute TEE attestation hash for on-chain commitment
    let tee_hash = tee_attestation
        .as_ref()
        .filter(|a| a.has_report())
        .map(|a| a.report_hash_felt());

    let metadata = MLClaimMetadata {
        model_id,
        num_layers: model.graph.num_layers() as u32,
        activation_type,
        io_commitment,
        weight_commitment,
        tee_attestation_hash: tee_hash,
    };

    // Serialize
    eprintln!("Serializing proof (format={:?})...", cli.format);
    let t_ser = Instant::now();
    let output_bytes = match cli.format {
        OutputFormat::CairoSerde => {
            let felts = serialize_ml_proof_for_recursive(&proof, &metadata, cli.salt);
            eprintln!("  {} felt252 values, streaming to file...", felts.len());
            serialize_ml_proof_to_file(&felts, &cli.output).unwrap_or_else(|e| {
                eprintln!("Error writing output to '{}': {e}", cli.output.display());
                process::exit(1);
            })
        }
        OutputFormat::Json => {
            let output_str = json_serde::proof_to_json(&proof, &metadata);
            let len = output_str.len();
            std::fs::write(&cli.output, &output_str).unwrap_or_else(|e| {
                eprintln!("Error writing output to '{}': {e}", cli.output.display());
                process::exit(1);
            });
            len
        }
        OutputFormat::MlGkr => {
            use stwo_ml::starknet::{
                build_gkr_serializable_proof, build_verify_model_gkr_calldata,
                build_verify_model_gkr_v2_calldata, build_verify_model_gkr_v3_calldata,
                build_verify_model_gkr_v4_calldata,
            };

            let gkr_proof =
                build_gkr_serializable_proof(&proof, model_id, &input).unwrap_or_else(|e| {
                    eprintln!("Error building GKR proof artifact: {e}");
                    eprintln!(
                        "Hint: --format ml_gkr requires the ML GKR pipeline (pure GKR proving)."
                    );
                    process::exit(1);
                });

            eprintln!(
                "  gkr_calldata: {} felts, io_calldata: {} felts, weight_openings: {} felts, weight_claims: {} felts",
                gkr_proof.gkr_calldata.len(),
                gkr_proof.io_calldata.len(),
                gkr_proof.weight_opening_calldata.len(),
                gkr_proof.weight_claim_calldata.len(),
            );
            eprintln!(
                "  num_layer_proofs: {}, estimated_gas: {}, total_calldata: {} felts",
                gkr_proof.num_layer_proofs, gkr_proof.estimated_gas, gkr_proof.total_calldata_size,
            );
            eprintln!(
                "  weight_opening_mode: {:?}, submission_ready: {}",
                gkr_proof.weight_opening_mode, gkr_proof.submission_ready,
            );
            if let Some(reason) = &gkr_proof.soundness_gate_error {
                eprintln!("  Warning: Starknet soundness gate status: {reason}");
            }

            let use_starknet_gkr_v4 = std::env::var("STWO_STARKNET_GKR_V4")
                .ok()
                .map(|v| {
                    let v = v.trim().to_ascii_lowercase();
                    !v.is_empty() && v != "0" && v != "false" && v != "off"
                })
                .unwrap_or(false);
            let use_starknet_gkr_v3 = std::env::var("STWO_STARKNET_GKR_V3")
                .ok()
                .map(|v| {
                    let v = v.trim().to_ascii_lowercase();
                    !v.is_empty() && v != "0" && v != "false" && v != "off"
                })
                .unwrap_or(false);
            let use_starknet_gkr_v2 = !use_starknet_gkr_v4
                && !use_starknet_gkr_v3
                && std::env::var("STWO_STARKNET_GKR_V2")
                    .ok()
                    .map(|v| {
                        let v = v.trim().to_ascii_lowercase();
                        !v.is_empty() && v != "0" && v != "false" && v != "off"
                    })
                    .unwrap_or(false);
            let verify_entrypoint = if use_starknet_gkr_v4 {
                "verify_model_gkr_v4"
            } else if use_starknet_gkr_v3 {
                "verify_model_gkr_v3"
            } else if use_starknet_gkr_v2 {
                "verify_model_gkr_v2"
            } else {
                "verify_model_gkr"
            };
            eprintln!("  starknet_entrypoint: {verify_entrypoint}");

            // Build complete verify_model_gkr calldata (all parameters pre-assembled)
            let verify_calldata_obj = if let Some(gkr_p) = proof.gkr_proof.as_ref() {
                match stwo_ml::gkr::LayeredCircuit::from_graph(&model.graph) {
                    Ok(circuit) => {
                        let raw_io =
                            stwo_ml::cairo_serde::serialize_raw_io(&input, &proof.execution.output);
                        let verify_result = if use_starknet_gkr_v4 {
                            build_verify_model_gkr_v4_calldata(gkr_p, &circuit, model_id, &raw_io)
                        } else if use_starknet_gkr_v3 {
                            build_verify_model_gkr_v3_calldata(gkr_p, &circuit, model_id, &raw_io)
                        } else if use_starknet_gkr_v2 {
                            build_verify_model_gkr_v2_calldata(gkr_p, &circuit, model_id, &raw_io)
                        } else {
                            build_verify_model_gkr_calldata(gkr_p, &circuit, model_id, &raw_io)
                        };
                        match verify_result {
                            Ok(vc) => {
                                eprintln!(
                                    "  verify_calldata: {} parts (ready for submission)",
                                    vc.total_felts
                                );
                                serde_json::json!({
                                    "schema_version": 1,
                                    "entrypoint": verify_entrypoint,
                                    "calldata": vc.calldata_parts,
                                    "total_felts": vc.total_felts,
                                    "upload_chunks": Vec::<Vec<String>>::new(),
                                })
                            }
                            Err(e) => {
                                eprintln!(
                                    "  Warning: soundness gate rejected verify_calldata: {e}"
                                );
                                serde_json::json!({
                                    "schema_version": 1,
                                    "entrypoint": "unsupported",
                                    "calldata": Vec::<String>::new(),
                                    "total_felts": 0,
                                    "upload_chunks": Vec::<Vec<String>>::new(),
                                    "reason": e.to_string(),
                                })
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("  Warning: could not build verify_calldata: {e}");
                        serde_json::json!({
                            "schema_version": 1,
                            "entrypoint": "unsupported",
                            "calldata": Vec::<String>::new(),
                            "total_felts": 0,
                            "upload_chunks": Vec::<Vec<String>>::new(),
                            "reason": e.to_string(),
                        })
                    }
                }
            } else {
                serde_json::json!({
                    "schema_version": 1,
                    "entrypoint": "unsupported",
                    "calldata": Vec::<String>::new(),
                    "total_felts": 0,
                    "upload_chunks": Vec::<Vec<String>>::new(),
                    "reason": "missing GKR proof in aggregated artifact",
                })
            };

            let json_obj = serde_json::json!({
                "format": "ml_gkr",
                "model_id": format!("0x{:064x}", gkr_proof.model_id),
                "io_commitment": format!("0x{:064x}", gkr_proof.io_commitment),
                "layer_chain_commitment": format!("0x{:064x}", gkr_proof.layer_chain_commitment),
                "num_layer_proofs": gkr_proof.num_layer_proofs,
                "estimated_gas": gkr_proof.estimated_gas,
                "total_calldata_size": gkr_proof.total_calldata_size,
                "gkr_calldata": gkr_proof.gkr_calldata.iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect::<Vec<_>>(),
                "io_calldata": gkr_proof.io_calldata.iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect::<Vec<_>>(),
                "weight_commitments": gkr_proof.weight_commitments.iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect::<Vec<_>>(),
                "weight_opening_calldata": gkr_proof.weight_opening_calldata.iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect::<Vec<_>>(),
                "weight_claim_calldata": gkr_proof.weight_claim_calldata.iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect::<Vec<_>>(),
                "weight_binding_schema_version": gkr_proof.weight_binding_schema_version,
                "weight_binding_mode_id": gkr_proof.weight_binding_mode_id,
                "weight_binding_data_calldata": gkr_proof.weight_binding_data_calldata.iter()
                    .map(|f| format!("0x{:x}", f))
                    .collect::<Vec<_>>(),
                "weight_opening_mode": format!("{:?}", gkr_proof.weight_opening_mode),
                "submission_ready": gkr_proof.submission_ready,
                "soundness_gate_error": gkr_proof.soundness_gate_error,
                "verify_calldata": verify_calldata_obj,
            });
            let output_str = serde_json::to_string_pretty(&json_obj).unwrap();
            let len = output_str.len();
            std::fs::write(&cli.output, &output_str).unwrap_or_else(|e| {
                eprintln!("Error writing output to '{}': {e}", cli.output.display());
                process::exit(1);
            });
            len
        }
        OutputFormat::Direct => {
            let direct_metadata = DirectProofMetadata {
                model_id,
                io_commitment,
                weight_commitment,
                num_layers: model.graph.num_layers() as u32,
                activation_type,
            };
            let direct_proof = build_starknet_proof_direct(&proof, &input, direct_metadata);
            let verify_calldata = stwo_ml::starknet::build_verify_model_direct_calldata(
                &direct_proof,
                "__SESSION_ID__",
            );

            eprintln!(
                "  batched_calldata: {} batches, stark_chunks: {} chunks, has_stark: {}",
                direct_proof.batched_calldata.len(),
                direct_proof.stark_chunks.len(),
                direct_proof.has_activation_stark,
            );
            eprintln!(
                "  estimated_gas: {}, total_calldata: {} felts",
                direct_proof.estimated_gas, direct_proof.total_calldata_size,
            );
            eprintln!(
                "  verify_calldata: {} parts, {} upload chunks",
                verify_calldata.total_felts,
                verify_calldata.upload_chunks.len(),
            );

            // Write as JSON for inspection / pipeline consumption
            let json_obj = serde_json::json!({
                "format": "direct",
                "model_id": format!("0x{:064x}", direct_proof.model_id),
                "io_commitment": format!("0x{:064x}", io_commitment),
                "raw_io_data_len": direct_proof.raw_io_data.len(),
                "weight_commitment": format!("0x{:064x}", direct_proof.weight_commitment),
                "num_layers": direct_proof.num_layers,
                "activation_type": direct_proof.activation_type,
                "has_activation_stark": direct_proof.has_activation_stark,
                "estimated_gas": direct_proof.estimated_gas,
                "total_calldata_size": direct_proof.total_calldata_size,
                "batched_calldata": direct_proof.batched_calldata.iter()
                    .map(|batch| batch.iter().map(|f| format!("0x{:x}", f)).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                "stark_chunks": direct_proof.stark_chunks.iter()
                    .map(|chunk| chunk.iter().map(|f| format!("0x{:x}", f)).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                "verify_calldata": {
                    "schema_version": 1,
                    "entrypoint": "verify_model_direct",
                    "calldata": verify_calldata.calldata_parts,
                    "total_felts": verify_calldata.total_felts,
                    "upload_chunks": verify_calldata.upload_chunks,
                },
            });
            let output_str = serde_json::to_string_pretty(&json_obj).unwrap();
            let len = output_str.len();
            std::fs::write(&cli.output, &output_str).unwrap_or_else(|e| {
                eprintln!("Error writing output to '{}': {e}", cli.output.display());
                process::exit(1);
            });
            len
        }
    };

    let ser_elapsed = t_ser.elapsed();

    eprintln!(
        "Proof written to {} ({} bytes, format={:?}, serialize={:.2}s)",
        cli.output.display(),
        output_bytes,
        cli.format,
        ser_elapsed.as_secs_f64(),
    );
    eprintln!(
        "  matmul_proofs: {} individual + {} batched ({} total), activation_claims: {}",
        proof.matmul_proofs.len(),
        proof.batched_matmul_proofs.len(),
        proof.total_matmul_count(),
        proof.activation_claims.len(),
    );
    eprintln!(
        "  prove_time: {:.2}s, model: {}, layers: {}",
        prove_elapsed.as_secs_f64(),
        model.metadata.name,
        model.metadata.num_layers,
    );

    // Print TEE attestation summary
    if let Some(ref att) = tee_attestation {
        let hash_felt = att.report_hash_felt();
        eprintln!(
            "  tee: hw={}, report_hash=0x{:x}, timestamp={}",
            att.hw_model, hash_felt, att.timestamp,
        );
    } else {
        eprintln!("  tee: none (zk-only)");
    }

    // --submit-gkr: submit GKR proof on-chain via sncast
    if cli.submit_gkr {
        submit_gkr_onchain(&cli, &model, &proof, &input, model_id, io_commitment);
    }
}

/// Submit a GKR proof on-chain via `sncast invoke`.
///
/// Steps:
/// 1. Compile LayeredCircuit from the model graph
/// 2. Build calldata matching `verify_model_gkr()` contract signature
/// 3. Write calldata to temp file (may exceed shell arg limit)
/// 4. Invoke sncast with the calldata
fn submit_gkr_onchain(
    cli: &Cli,
    model: &OnnxModel,
    proof: &stwo_ml::aggregation::AggregatedModelProofOnChain,
    input: &M31Matrix,
    model_id: FieldElement,
    _io_commitment: FieldElement,
) {
    use stwo_ml::starknet::{
        build_circuit_descriptor, build_register_gkr_calldata, build_verify_model_gkr_calldata,
        build_verify_model_gkr_v2_calldata, build_verify_model_gkr_v3_calldata,
    };

    if cli.format != OutputFormat::MlGkr {
        eprintln!("Error: --submit-gkr requires --format ml_gkr");
        process::exit(1);
    }

    let gkr_proof = match &proof.gkr_proof {
        Some(p) => p,
        None => {
            eprintln!("Error: --submit-gkr requires a GKR proof (use --format ml_gkr)");
            process::exit(1);
        }
    };

    // Compile the LayeredCircuit for dimension extraction
    let circuit = stwo_ml::gkr::LayeredCircuit::from_graph(&model.graph).unwrap_or_else(|e| {
        eprintln!("Error compiling GKR circuit: {e}");
        process::exit(1);
    });

    eprintln!();
    eprintln!("=== On-Chain GKR Submission ===");
    eprintln!("  Contract: {}", cli.contract);
    eprintln!("  Account:  {}", cli.account);
    eprintln!("  Network:  {}", cli.network);
    eprintln!("  Fee:      auto-estimated");

    let use_starknet_gkr_v3 = std::env::var("STWO_STARKNET_GKR_V3")
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        })
        .unwrap_or(false);
    let use_starknet_gkr_v2 = !use_starknet_gkr_v3
        && std::env::var("STWO_STARKNET_GKR_V2")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !v.is_empty() && v != "0" && v != "false" && v != "off"
            })
            .unwrap_or(false);
    let verify_entrypoint = if use_starknet_gkr_v3 {
        "verify_model_gkr_v3"
    } else if use_starknet_gkr_v2 {
        "verify_model_gkr_v2"
    } else {
        "verify_model_gkr"
    };
    eprintln!("  Entrypoint: {}", verify_entrypoint);

    // Step 1: Build verify_model_gkr calldata (raw IO data for on-chain recomputation)
    let raw_io_data = stwo_ml::cairo_serde::serialize_raw_io(input, &proof.execution.output);
    let verify_calldata = if use_starknet_gkr_v3 {
        build_verify_model_gkr_v3_calldata(gkr_proof, &circuit, model_id, &raw_io_data)
    } else if use_starknet_gkr_v2 {
        build_verify_model_gkr_v2_calldata(gkr_proof, &circuit, model_id, &raw_io_data)
    } else {
        build_verify_model_gkr_calldata(gkr_proof, &circuit, model_id, &raw_io_data)
    }
    .unwrap_or_else(|e| {
        eprintln!("Error: soundness gate rejected GKR calldata: {e}");
        process::exit(1);
    });

    eprintln!(
        "  Calldata: {} parts ({} estimated felts)",
        verify_calldata.total_felts, verify_calldata.total_felts
    );

    // Write calldata to temp file (may exceed shell arg limit for large proofs)
    let calldata_path = cli.output.with_extension("calldata.txt");
    let calldata_str = verify_calldata.calldata_parts.join(" ");
    std::fs::write(&calldata_path, &calldata_str).unwrap_or_else(|e| {
        eprintln!(
            "Error writing calldata to '{}': {e}",
            calldata_path.display()
        );
        process::exit(1);
    });
    eprintln!("  Calldata written to: {}", calldata_path.display());

    // Step 2: Check if model needs registration first
    eprintln!("  Checking model registration...");
    let check_result = std::process::Command::new("sncast")
        .args([
            "--account",
            &cli.account,
            "call",
            "--network",
            &cli.network,
            "--contract-address",
            &cli.contract,
            "--function",
            "get_model_circuit_hash",
            "--calldata",
            &format!("0x{:x}", model_id),
        ])
        .output();

    let needs_registration = match check_result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // If circuit_hash is 0, model is not registered
            stdout.contains("0x0") || !output.status.success()
        }
        Err(_) => {
            eprintln!("  Warning: could not check registration (sncast not found?)");
            eprintln!("  Attempting to register anyway...");
            true
        }
    };

    if needs_registration {
        eprintln!("  Model not registered — registering for GKR verification...");
        let circuit_desc = build_circuit_descriptor(&circuit);
        let mut register_weight_commitments = gkr_proof.weight_commitments.clone();
        for deferred in &gkr_proof.deferred_proofs {
            if let Some(wc) = deferred.weight_commitment() {
                register_weight_commitments.push(wc);
            }
        }
        let register_calldata =
            build_register_gkr_calldata(model_id, &register_weight_commitments, &circuit_desc);
        let register_str = register_calldata.join(" ");

        let register_result = std::process::Command::new("sncast")
            .args([
                "--account",
                &cli.account,
                "invoke",
                "--network",
                &cli.network,
                "--contract-address",
                &cli.contract,
                "--function",
                "register_model_gkr",
                "--calldata",
                &register_str,
            ])
            .output();

        match register_result {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                eprintln!("  Registration submitted: {}", stdout.trim());
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                // May fail if already registered — that's OK
                if stderr.contains("already registered") || stdout.contains("already registered") {
                    eprintln!("  Model already registered (OK)");
                } else {
                    eprintln!("  Warning: registration may have failed:");
                    eprintln!("    stdout: {}", stdout.trim());
                    eprintln!("    stderr: {}", stderr.trim());
                    eprintln!("  Continuing with verification attempt...");
                }
            }
            Err(e) => {
                eprintln!("  Error: could not run sncast: {e}");
                eprintln!("  Make sure sncast is installed and in PATH");
                process::exit(1);
            }
        }
    } else {
        eprintln!("  Model already registered (OK)");
    }

    // Step 3: Submit verification
    eprintln!("  Submitting {}...", verify_entrypoint);

    // For large calldatas, read from file to avoid shell arg limit
    let verify_result = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "sncast --account '{}' invoke \
             --network '{}' \
             --contract-address '{}' \
             --function {} \
             --calldata $(cat '{}')",
            cli.account,
            cli.network,
            cli.contract,
            verify_entrypoint,
            calldata_path.display(),
        ))
        .output();

    match verify_result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            eprintln!("  Verification submitted successfully!");
            eprintln!("  {}", stdout.trim());
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            eprintln!("  Error: verification submission failed");
            eprintln!("    stdout: {}", stdout.trim());
            eprintln!("    stderr: {}", stderr.trim());
            eprintln!(
                "  Calldata saved to: {} (retry manually)",
                calldata_path.display()
            );
            process::exit(1);
        }
        Err(e) => {
            eprintln!("  Error: could not run sncast: {e}");
            process::exit(1);
        }
    }

    eprintln!("==============================");
}

// ─── Privacy Subcommands ─────────────────────────────────────────────────

fn resolve_wallet_path(path_str: &str) -> PathBuf {
    if path_str.starts_with("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(&path_str[2..])
    } else {
        PathBuf::from(path_str)
    }
}

/// Build a PoolClientConfig from common CLI args (network + pool_contract).
#[cfg(feature = "audit-http")]
fn build_pool_config(
    network: &str,
    pool_contract: Option<&str>,
) -> stwo_ml::privacy::pool_client::PoolClientConfig {
    use stwo_ml::privacy::pool_client::PoolClientConfig;

    let mut config = PoolClientConfig::from_env(network);
    if let Some(addr) = pool_contract {
        config.pool_address = addr.to_string();
    }
    config
}

fn run_wallet_command(cmd: &WalletCmd) {
    use stwo_ml::privacy::wallet::Wallet;

    match cmd.action.as_str() {
        "generate" => {
            let wallet = Wallet::generate().unwrap_or_else(|e| {
                eprintln!("Error: failed to generate wallet: {e}");
                std::process::exit(1);
            });
            let output_path = cmd
                .output
                .as_deref()
                .map(|p| PathBuf::from(p))
                .unwrap_or_else(Wallet::default_path);

            eprintln!();
            eprintln!("  prove-model wallet generate");
            eprintln!("  ───────────────────────────");
            wallet
                .save(&output_path, cmd.password.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("Error: {e}");
                    process::exit(1);
                });
            eprintln!("  address:  {}", wallet.address());
            eprintln!("  saved:    {}", output_path.display());
            if cmd.password.is_some() {
                eprintln!("  encrypted: yes");
            }

            // JSON on stdout
            println!(
                "{{\"address\":\"{}\",\"path\":\"{}\"}}",
                wallet.address(),
                output_path.display(),
            );
        }
        "show" => {
            let wallet_path = resolve_wallet_path(&cmd.wallet);
            let wallet = Wallet::load(&wallet_path, cmd.password.as_deref()).unwrap_or_else(|e| {
                eprintln!("Error loading wallet: {e}");
                process::exit(1);
            });

            eprintln!();
            eprintln!("  prove-model wallet show");
            eprintln!("  ───────────────────────");
            eprintln!("  address:     {}", wallet.address());
            eprintln!(
                "  viewing_key: 0x{:08x}{:08x}{:08x}{:08x}",
                wallet.viewing_key[0].0,
                wallet.viewing_key[1].0,
                wallet.viewing_key[2].0,
                wallet.viewing_key[3].0,
            );
            eprintln!("  wallet:      {}", wallet_path.display());

            println!(
                "{{\"address\":\"{}\",\"viewing_key\":\"0x{:08x}{:08x}{:08x}{:08x}\"}}",
                wallet.address(),
                wallet.viewing_key[0].0,
                wallet.viewing_key[1].0,
                wallet.viewing_key[2].0,
                wallet.viewing_key[3].0,
            );
        }
        "import" => {
            let sk_hex = cmd.spending_key.as_deref().unwrap_or_else(|| {
                eprintln!("Error: --spending-key required for import");
                process::exit(1);
            });
            let wallet = Wallet::from_hex(sk_hex).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                process::exit(1);
            });
            let output_path = cmd
                .output
                .as_deref()
                .map(|p| PathBuf::from(p))
                .unwrap_or_else(Wallet::default_path);

            wallet
                .save(&output_path, cmd.password.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("Error: {e}");
                    process::exit(1);
                });

            eprintln!("Wallet imported: {}", wallet.address());
            eprintln!("Saved to: {}", output_path.display());
            println!(
                "{{\"address\":\"{}\",\"path\":\"{}\"}}",
                wallet.address(),
                output_path.display()
            );
        }
        "export-viewing-key" => {
            let wallet_path = resolve_wallet_path(&cmd.wallet);
            let wallet = Wallet::load(&wallet_path, cmd.password.as_deref()).unwrap_or_else(|e| {
                eprintln!("Error: {e}");
                process::exit(1);
            });
            let vk = format!(
                "0x{:08x}{:08x}{:08x}{:08x}",
                wallet.viewing_key[0].0,
                wallet.viewing_key[1].0,
                wallet.viewing_key[2].0,
                wallet.viewing_key[3].0,
            );
            println!("{vk}");
        }
        other => {
            eprintln!("Error: unknown wallet action '{other}'. Use: generate, show, import, export-viewing-key");
            process::exit(1);
        }
    }
}

fn run_deposit_command(cmd: &DepositCmd) {
    use stwo_ml::privacy::note_store::NoteStore;
    use stwo_ml::privacy::serde_utils::{batch_proof_to_json, build_batch_proof_output};
    use stwo_ml::privacy::tx_builder::TxBuilder;
    use stwo_ml::privacy::wallet::Wallet;

    let wallet_path = resolve_wallet_path(&cmd.wallet);
    let wallet = Wallet::load(&wallet_path, cmd.password.as_deref()).unwrap_or_else(|e| {
        eprintln!("Error loading wallet: {e}");
        process::exit(1);
    });

    // For self-deposits, use own keys. For deposits to others, recipient
    // must provide their viewing key (required for memo encryption).
    let (recipient_pk, recipient_vk) = if let Some(ref hex) = cmd.recipient {
        let pk = parse_pubkey_hex(hex);
        // When depositing to someone else, we need their viewing key.
        // For now, self-deposit is the common case. External deposits
        // require the recipient to share their viewing key out-of-band.
        eprintln!("Warning: depositing to external recipient without their viewing key.");
        eprintln!("  The memo will be encrypted with a derived key that the recipient");
        eprintln!("  may not be able to decrypt unless they share their viewing key.");
        (pk, wallet.viewing_key) // fallback: not ideal
    } else {
        (wallet.public_key, wallet.viewing_key)
    };

    eprintln!();
    eprintln!("  prove-model deposit");
    eprintln!("  ────────────────────");
    eprintln!("  amount:    {}", cmd.amount);
    eprintln!("  asset:     {}", cmd.asset);
    eprintln!(
        "  recipient: 0x{:08x}{:08x}...",
        recipient_pk[0].0, recipient_pk[1].0
    );

    let t0 = Instant::now();
    eprintln!();
    eprintln!("  Building deposit witness...");
    eprintln!("  Proving batch (1 deposit, 0 withdrawals, 0 spends)...");

    let mut builder = TxBuilder::new();
    builder
        .deposit(cmd.amount, cmd.asset, recipient_pk, recipient_vk)
        .unwrap_or_else(|e| {
            eprintln!("Error: invalid deposit parameters: {e}");
            process::exit(1);
        });

    let result = builder.prove().unwrap_or_else(|e| {
        eprintln!("Error: proving failed: {e}");
        process::exit(1);
    });

    let prove_time = t0.elapsed();
    eprintln!("  Proved in {:.2}s", prove_time.as_secs_f64());

    // Save proof
    let output = build_batch_proof_output(
        &result.proof.public_inputs,
        "batch_proof",
        prove_time.as_millis() as u64,
        result.encrypted_memos,
    );
    let json = batch_proof_to_json(&output);
    std::fs::write(&cmd.output, &json).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        process::exit(1);
    });

    // Update note store
    let notes_path = NoteStore::default_path();
    let mut note_store = NoteStore::load(&notes_path, None).unwrap_or_else(|e| {
        eprintln!("Warning: could not load note store: {e}");
        NoteStore {
            notes: Vec::new(),
            path: notes_path.clone(),
        }
    });
    for (commitment, note) in &result.new_commitments {
        let hex = format!(
            "0x{}",
            commitment
                .iter()
                .map(|m| format!("{:08x}", m.0))
                .collect::<String>()
        );
        note_store.add_note(note, &hex, 0);
    }
    note_store.save(None).ok();

    eprintln!();
    if !output.deposits.is_empty() {
        eprintln!("  commitment: {}", output.deposits[0].commitment);
    }
    eprintln!("  proof saved: {}", cmd.output.display());

    // JSON on stdout
    println!(
        "{{\"commitment\":\"{}\",\"proof_hash\":\"{}\",\"prove_time_ms\":{}}}",
        output
            .deposits
            .first()
            .map(|d| d.commitment.as_str())
            .unwrap_or(""),
        output.proof_hash,
        output.prove_time_ms,
    );
}

fn run_withdraw_command(cmd: &WithdrawCmd) {
    use stwo_ml::privacy::note_store::NoteStore;
    use stwo_ml::privacy::relayer::compute_withdrawal_binding_digest;
    use stwo_ml::privacy::serde_utils::{batch_proof_to_json, build_batch_proof_output};
    use stwo_ml::privacy::tx_builder::TxBuilder;
    use stwo_ml::privacy::wallet::Wallet;

    let wallet_path = resolve_wallet_path(&cmd.wallet);
    let wallet = Wallet::load(&wallet_path, cmd.password.as_deref()).unwrap_or_else(|e| {
        eprintln!("Error loading wallet: {e}");
        process::exit(1);
    });

    eprintln!();
    eprintln!("  prove-model withdraw");
    eprintln!("  ─────────────────────");
    eprintln!("  amount:  {}", cmd.amount);
    eprintln!("  asset:   {}", cmd.asset);
    eprintln!("  address: {}", wallet.address());

    // Load note store and find a spendable note
    let notes_path = NoteStore::default_path();
    let note_store = NoteStore::load(&notes_path, None).unwrap_or_else(|e| {
        eprintln!("Error: could not load note store: {e}");
        process::exit(1);
    });

    let balance = note_store.balance(cmd.asset);
    if balance < cmd.amount {
        eprintln!(
            "Error: insufficient balance. Have {balance}, need {}",
            cmd.amount
        );
        process::exit(1);
    }

    // For withdraw, we need exactly one note with enough balance
    let spendable = note_store.spendable_notes(cmd.asset);
    let note_entry = spendable.iter().find(|n| n.note.amount() >= cmd.amount);
    let note_entry = note_entry.unwrap_or_else(|| {
        eprintln!(
            "Error: no single note covers amount {}. Largest note: {}",
            cmd.amount,
            spendable.first().map(|n| n.note.amount()).unwrap_or(0)
        );
        eprintln!("Hint: use 'transfer' for splitting/merging notes");
        process::exit(1);
    });

    let note = note_entry.note.to_note();
    let commitment = note.commitment();
    #[cfg(feature = "audit-http")]
    let leaf_index = note_entry.merkle_index;
    let note_amount = (note.amount_lo.0 as u64) | ((note.amount_hi.0 as u64) << 31);
    if note_amount != cmd.amount {
        eprintln!(
            "Error: withdraw proves full-note exits only. Selected note amount is {}, requested {}",
            note_amount, cmd.amount
        );
        eprintln!("Hint: use 'transfer' to split/merge notes, then withdraw an exact-size note");
        process::exit(1);
    }

    // Proof-bound recipients for V2 withdrawal binding.
    let payout_recipient = cmd.payout_recipient.as_deref().unwrap_or_else(|| {
        eprintln!("Error: --payout-recipient is required for withdrawal proof binding");
        process::exit(1);
    });
    let credit_recipient = cmd.credit_recipient.as_deref().unwrap_or(payout_recipient);
    let amount_lo = cmd.amount & 0x7FFF_FFFF;
    let amount_hi = cmd.amount >> 31;
    let withdrawal_binding = compute_withdrawal_binding_digest(
        payout_recipient,
        credit_recipient,
        cmd.asset as u64,
        amount_lo,
        amount_hi,
        0, // single-withdraw command always occupies withdrawal index 0
    )
    .unwrap_or_else(|e| {
        eprintln!("Error: invalid withdrawal binding inputs: {e}");
        process::exit(1);
    });

    // Sync Merkle tree from on-chain pool
    #[cfg(feature = "audit-http")]
    let (path, root) = {
        use stwo_ml::privacy::pool_client::PoolClient;
        use stwo_ml::privacy::tree_sync::TreeSync;

        let pool_config = build_pool_config(&cmd.priv_network, cmd.pool_contract.as_deref());
        if pool_config.pool_address.is_empty() {
            eprintln!("Error: pool address not configured");
            eprintln!("Hint: set VM31_POOL_ADDRESS or use --pool-contract");
            process::exit(1);
        }
        let pool_client = PoolClient::new(pool_config);

        let cache_path = TreeSync::default_cache_path();
        let mut tree_sync = TreeSync::load_or_create(&cache_path).unwrap_or_else(|e| {
            eprintln!("Warning: cache load failed ({e}), starting fresh sync");
            TreeSync::new()
        });

        eprintln!("  Syncing Merkle tree...");
        let sync_result = tree_sync.sync(&pool_client).unwrap_or_else(|e| {
            eprintln!("Error: Merkle tree sync failed: {e}");
            process::exit(1);
        });
        eprintln!(
            "  Synced: {} leaves ({} new)",
            sync_result.total_leaves, sync_result.events_added
        );

        let path = tree_sync.prove(leaf_index).unwrap_or_else(|e| {
            eprintln!("Error: Merkle proof failed for index {leaf_index}: {e}");
            process::exit(1);
        });
        let root = tree_sync.root();

        // Local verification before proving
        use stwo_ml::crypto::merkle_m31::verify_merkle_proof;
        if !verify_merkle_proof(&root, &commitment, &path, 20) {
            eprintln!("Error: local Merkle proof verification failed");
            eprintln!("Hint: delete ~/.vm31/tree_cache.json and re-sync");
            process::exit(1);
        }

        (path, root)
    };

    // Fallback when audit-http is not available: ephemeral single-note tree
    #[cfg(not(feature = "audit-http"))]
    let (path, root) = {
        use stwo_ml::crypto::merkle_m31::PoseidonMerkleTreeM31;
        eprintln!("  Warning: no audit-http feature, using ephemeral tree");
        let mut tree = PoseidonMerkleTreeM31::new(20);
        tree.append(commitment);
        let path = tree.prove(0).unwrap();
        let root = tree.root();
        (path, root)
    };

    let t0 = Instant::now();
    eprintln!();
    eprintln!("  Proving batch (0 deposits, 1 withdrawal, 0 spends)...");
    eprintln!("  payout_recipient: {}", payout_recipient);
    eprintln!("  credit_recipient: {}", credit_recipient);

    let mut builder = TxBuilder::new();
    builder
        .withdraw_with_binding(
            cmd.amount,
            cmd.asset,
            note,
            wallet.spending_key,
            path,
            root,
            withdrawal_binding,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error: invalid withdraw parameters: {e}");
            process::exit(1);
        });

    let result = builder.prove().unwrap_or_else(|e| {
        eprintln!("Error: proving failed: {e}");
        process::exit(1);
    });

    let prove_time = t0.elapsed();
    eprintln!("  Proved in {:.2}s", prove_time.as_secs_f64());

    let output = build_batch_proof_output(
        &result.proof.public_inputs,
        "batch_proof",
        prove_time.as_millis() as u64,
        result.encrypted_memos,
    );
    let json = batch_proof_to_json(&output);
    std::fs::write(&cmd.output, &json).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        process::exit(1);
    });

    // Mark note as spent
    let mut note_store = NoteStore::load(&notes_path, None).unwrap_or_else(|_| NoteStore {
        notes: Vec::new(),
        path: notes_path.clone(),
    });
    for c in &result.spent_commitments {
        note_store.mark_spent(c);
    }
    note_store.save(None).ok();

    eprintln!("  proof saved: {}", cmd.output.display());
    if !output.withdrawals.is_empty() {
        println!(
            "{{\"nullifier\":\"{}\",\"withdrawal_binding\":\"{}\",\"payout_recipient\":\"{}\",\"credit_recipient\":\"{}\",\"prove_time_ms\":{}}}",
            output.withdrawals[0].nullifier,
            output.withdrawals[0].withdrawal_binding,
            payout_recipient,
            credit_recipient,
            output.prove_time_ms,
        );
    }
}

fn run_transfer_command(cmd: &TransferCmd) {
    use stwo_ml::privacy::note_store::NoteStore;
    use stwo_ml::privacy::serde_utils::{batch_proof_to_json, build_batch_proof_output};
    use stwo_ml::privacy::tx_builder::TxBuilder;
    use stwo_ml::privacy::wallet::Wallet;

    let wallet_path = resolve_wallet_path(&cmd.wallet);
    let wallet = Wallet::load(&wallet_path, cmd.password.as_deref()).unwrap_or_else(|e| {
        eprintln!("Error loading wallet: {e}");
        process::exit(1);
    });

    let recipient_pk = parse_pubkey_hex(&cmd.to);
    let recipient_vk = parse_pubkey_hex(&cmd.to_viewing_key); // same format: 4 M31 elements

    eprintln!();
    eprintln!("  prove-model transfer");
    eprintln!("  ─────────────────────");
    eprintln!("  amount: {}", cmd.amount);
    eprintln!("  asset:  {}", cmd.asset);
    eprintln!("  to:     {}", cmd.to);

    // Load note store and select 2 input notes
    let notes_path = NoteStore::default_path();
    let note_store = NoteStore::load(&notes_path, None).unwrap_or_else(|e| {
        eprintln!("Error: could not load note store: {e}");
        process::exit(1);
    });

    let selection = note_store.select_notes(cmd.asset, cmd.amount);
    let (selected, change) = selection.unwrap_or_else(|| {
        let balance = note_store.balance(cmd.asset);
        eprintln!(
            "Error: insufficient balance. Have {balance}, need {}",
            cmd.amount
        );
        process::exit(1);
    });

    if selected.len() < 2 {
        eprintln!(
            "Error: need at least 2 notes for transfer, have {}",
            selected.len()
        );
        eprintln!("Hint: deposit more notes or use a single withdraw");
        process::exit(1);
    }

    let note1 = selected[0].note.to_note();
    let note2 = selected[1].note.to_note();
    #[cfg(feature = "audit-http")]
    let idx1 = selected[0].merkle_index;
    #[cfg(feature = "audit-http")]
    let idx2 = selected[1].merkle_index;

    // Sync Merkle tree from on-chain pool
    #[cfg(feature = "audit-http")]
    let (path1, path2, root) = {
        use stwo_ml::privacy::pool_client::PoolClient;
        use stwo_ml::privacy::tree_sync::TreeSync;

        let pool_config = build_pool_config(&cmd.priv_network, cmd.pool_contract.as_deref());
        if pool_config.pool_address.is_empty() {
            eprintln!("Error: pool address not configured");
            eprintln!("Hint: set VM31_POOL_ADDRESS or use --pool-contract");
            process::exit(1);
        }
        let pool_client = PoolClient::new(pool_config);

        let cache_path = TreeSync::default_cache_path();
        let mut tree_sync = TreeSync::load_or_create(&cache_path).unwrap_or_else(|e| {
            eprintln!("Warning: cache load failed ({e}), starting fresh sync");
            TreeSync::new()
        });

        eprintln!("  Syncing Merkle tree...");
        let sync_result = tree_sync.sync(&pool_client).unwrap_or_else(|e| {
            eprintln!("Error: Merkle tree sync failed: {e}");
            process::exit(1);
        });
        eprintln!(
            "  Synced: {} leaves ({} new)",
            sync_result.total_leaves, sync_result.events_added
        );

        let p1 = tree_sync.prove(idx1).unwrap_or_else(|e| {
            eprintln!("Error: Merkle proof failed for index {idx1}: {e}");
            process::exit(1);
        });
        let p2 = tree_sync.prove(idx2).unwrap_or_else(|e| {
            eprintln!("Error: Merkle proof failed for index {idx2}: {e}");
            process::exit(1);
        });
        let root = tree_sync.root();

        // Local verification before proving
        use stwo_ml::crypto::merkle_m31::verify_merkle_proof;
        if !verify_merkle_proof(&root, &note1.commitment(), &p1, 20) {
            eprintln!("Error: local Merkle proof verification failed for note 1");
            eprintln!("Hint: delete ~/.vm31/tree_cache.json and re-sync");
            process::exit(1);
        }
        if !verify_merkle_proof(&root, &note2.commitment(), &p2, 20) {
            eprintln!("Error: local Merkle proof verification failed for note 2");
            eprintln!("Hint: delete ~/.vm31/tree_cache.json and re-sync");
            process::exit(1);
        }

        (p1, p2, root)
    };

    // Fallback when audit-http is not available: ephemeral 2-note tree
    #[cfg(not(feature = "audit-http"))]
    let (path1, path2, root) = {
        use stwo_ml::crypto::merkle_m31::PoseidonMerkleTreeM31;
        eprintln!("  Warning: no audit-http feature, using ephemeral tree");
        let mut tree = PoseidonMerkleTreeM31::new(20);
        tree.append(note1.commitment());
        tree.append(note2.commitment());
        let root = tree.root();
        let path1 = tree.prove(0).unwrap();
        let path2 = tree.prove(1).unwrap();
        (path1, path2, root)
    };

    let t0 = Instant::now();
    eprintln!();
    eprintln!("  Proving batch (0 deposits, 0 withdrawals, 1 spend)...");

    let mut builder = TxBuilder::new();
    builder
        .transfer(
            cmd.amount,
            cmd.asset,
            recipient_pk,
            recipient_vk,
            wallet.viewing_key,
            [
                (note1, wallet.spending_key, path1),
                (note2, wallet.spending_key, path2),
            ],
            root,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error: invalid transfer parameters: {e}");
            process::exit(1);
        });

    let result = builder.prove().unwrap_or_else(|e| {
        eprintln!("Error: proving failed: {e}");
        process::exit(1);
    });

    let prove_time = t0.elapsed();
    eprintln!("  Proved in {:.2}s", prove_time.as_secs_f64());

    let output = build_batch_proof_output(
        &result.proof.public_inputs,
        "batch_proof",
        prove_time.as_millis() as u64,
        result.encrypted_memos,
    );
    let json = batch_proof_to_json(&output);
    std::fs::write(&cmd.output, &json).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        process::exit(1);
    });

    // Update note store
    let mut note_store = NoteStore::load(&notes_path, None).unwrap_or_else(|_| NoteStore {
        notes: Vec::new(),
        path: notes_path.clone(),
    });
    for c in &result.spent_commitments {
        note_store.mark_spent(c);
    }
    for (commitment, note) in &result.new_commitments {
        let hex = format!(
            "0x{}",
            commitment
                .iter()
                .map(|m| format!("{:08x}", m.0))
                .collect::<String>()
        );
        note_store.add_note_pending(note, &hex);
    }
    note_store.save(None).ok();

    eprintln!("  proof saved: {}", cmd.output.display());
    println!(
        "{{\"num_spends\":1,\"change\":{change},\"prove_time_ms\":{}}}",
        output.prove_time_ms,
    );
}

fn run_batch_command(cmd: &BatchCmd) {
    use stwo_ml::privacy::note_store::NoteStore;
    use stwo_ml::privacy::serde_utils::{
        batch_proof_to_json, build_batch_proof_output, parse_tx_file,
    };
    use stwo_ml::privacy::tx_builder::TxBuilder;
    use stwo_ml::privacy::wallet::Wallet;

    let wallet_path = resolve_wallet_path(&cmd.wallet);
    let wallet = Wallet::load(&wallet_path, cmd.password.as_deref()).unwrap_or_else(|e| {
        eprintln!("Error loading wallet: {e}");
        process::exit(1);
    });

    let tx_contents = std::fs::read_to_string(&cmd.tx_file).unwrap_or_else(|e| {
        eprintln!("Error reading tx-file '{}': {e}", cmd.tx_file.display());
        process::exit(1);
    });

    let entries = parse_tx_file(&tx_contents).unwrap_or_else(|e| {
        eprintln!("Error parsing tx-file: {e}");
        process::exit(1);
    });

    eprintln!();
    eprintln!("  prove-model batch");
    eprintln!("  ──────────────────");
    eprintln!("  tx-file: {}", cmd.tx_file.display());
    eprintln!("  entries: {}", entries.len());

    let mut builder = TxBuilder::new();
    for (i, entry) in entries.iter().enumerate() {
        match entry.tx_type.as_str() {
            "deposit" => {
                let pk = entry
                    .recipient_pubkey
                    .map(|arr| arr.map(M31::from_u32_unchecked))
                    .unwrap_or(wallet.public_key);
                // Batch deposits use wallet's viewing key (self-deposit).
                // External deposits in batch mode not yet supported.
                builder
                    .deposit(entry.amount, entry.asset_id, pk, wallet.viewing_key)
                    .unwrap_or_else(|e| {
                        eprintln!("Error: tx[{i}] invalid deposit: {e}");
                        process::exit(1);
                    });
            }
            "withdraw" | "transfer" => {
                eprintln!(
                    "Error: tx[{}] type '{}' requires merkle paths and note selection. \
                     Use the '{}' subcommand instead of batch for this transaction type.",
                    i, entry.tx_type, entry.tx_type
                );
                process::exit(1);
            }
            other => {
                eprintln!(
                    "Error: tx[{}] unknown type '{}'. Supported: deposit, withdraw, transfer",
                    i, other
                );
                process::exit(1);
            }
        }
    }

    if builder.is_empty() {
        eprintln!("Error: no supported transactions in tx-file");
        process::exit(1);
    }

    let t0 = Instant::now();
    eprintln!("  Proving batch ({} transactions)...", builder.len());

    let result = builder.prove().unwrap_or_else(|e| {
        eprintln!("Error: proving failed: {e}");
        process::exit(1);
    });

    let prove_time = t0.elapsed();
    eprintln!("  Proved in {:.2}s", prove_time.as_secs_f64());

    let output = build_batch_proof_output(
        &result.proof.public_inputs,
        "batch_proof",
        prove_time.as_millis() as u64,
        result.encrypted_memos,
    );
    let json = batch_proof_to_json(&output);
    std::fs::write(&cmd.output, &json).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        process::exit(1);
    });

    // Update note store
    let notes_path = NoteStore::default_path();
    let mut note_store = NoteStore::load(&notes_path, None).unwrap_or_else(|_| NoteStore {
        notes: Vec::new(),
        path: notes_path.clone(),
    });
    for (commitment, note) in &result.new_commitments {
        let hex = format!(
            "0x{}",
            commitment
                .iter()
                .map(|m| format!("{:08x}", m.0))
                .collect::<String>()
        );
        note_store.add_note(note, &hex, 0);
    }
    note_store.save(None).ok();

    eprintln!("  proof saved: {}", cmd.output.display());
    println!(
        "{{\"num_deposits\":{},\"num_withdrawals\":{},\"num_spends\":{},\"prove_time_ms\":{}}}",
        output.num_deposits, output.num_withdrawals, output.num_spends, output.prove_time_ms,
    );
}

fn run_pool_status_command(cmd: &PoolStatusCmd) {
    use stwo_ml::privacy::pool_client::{format_pool_status, PoolClientConfig};

    let config = PoolClientConfig::from_env(&cmd.priv_network);
    let pool_addr = cmd.pool_contract.as_deref().unwrap_or(&config.pool_address);

    eprintln!();
    eprintln!("  prove-model pool-status");
    eprintln!("  ───────────────────────");
    eprintln!("  network: {}", cmd.priv_network);
    eprintln!("  pool:    {}", pool_addr);

    if pool_addr.is_empty() {
        eprintln!("  status: pool address not configured");
        eprintln!("  Hint: set VM31_POOL_ADDRESS env var or use --pool-contract");
        println!("{{\"error\":\"pool_address_not_configured\"}}");
        return;
    }

    #[cfg(feature = "audit-http")]
    {
        use stwo_ml::privacy::pool_client::PoolClient;

        let client_config = PoolClientConfig {
            rpc_url: config.rpc_url.clone(),
            pool_address: pool_addr.to_string(),
            network: cmd.priv_network.clone(),
            verify_rpc_urls: Vec::new(),
        };
        let client = PoolClient::new(client_config);

        eprintln!("  Querying on-chain state...");

        // Get Merkle root
        let root_str = match client.get_merkle_root() {
            Ok(root) => {
                let hex: String = root.iter().map(|m| format!("{:08x}", m.0)).collect();
                format!("0x{hex}")
            }
            Err(e) => {
                eprintln!("  Warning: get_merkle_root failed: {e}");
                "error".to_string()
            }
        };

        // Get tree size
        let tree_size = match client.get_tree_size() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  Warning: get_tree_size failed: {e}");
                0
            }
        };

        eprintln!("  root:      {}", root_str);
        eprintln!("  tree_size: {}", tree_size);

        // Optional nullifier check
        if let Some(ref nul_hex) = cmd.check_nullifier {
            match parse_m31_digest_hex(nul_hex) {
                Ok(nul) => match client.is_nullifier_spent(&nul) {
                    Ok(spent) => eprintln!("  nullifier_spent: {spent}"),
                    Err(e) => eprintln!("  Warning: nullifier check failed: {e}"),
                },
                Err(e) => eprintln!("  Warning: invalid nullifier hex: {e}"),
            }
        }

        // Optional root check
        if let Some(ref root_hex) = cmd.check_root {
            match parse_m31_digest_hex(root_hex) {
                Ok(root) => match client.is_known_root(&root) {
                    Ok(known) => eprintln!("  root_known: {known}"),
                    Err(e) => eprintln!("  Warning: root check failed: {e}"),
                },
                Err(e) => eprintln!("  Warning: invalid root hex: {e}"),
            }
        }

        let json = format_pool_status(&root_str, tree_size, &cmd.priv_network, pool_addr);
        println!("{json}");
    }

    #[cfg(not(feature = "audit-http"))]
    {
        eprintln!("  Note: build with --features audit-http for live RPC queries");
        let json = format_pool_status("unavailable", 0, &cmd.priv_network, pool_addr);
        println!("{json}");
    }
}

/// Parse a hex string into [M31; 8] digest for pool-status nullifier/root checks.
#[cfg(feature = "audit-http")]
fn parse_m31_digest_hex(hex: &str) -> Result<[M31; 8], String> {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    if hex.len() != 64 {
        return Err(format!("expected 64 hex chars (8 x 8), got {}", hex.len()));
    }
    let mut result = [M31::from(0u32); 8];
    for i in 0..8 {
        let chunk = &hex[i * 8..(i + 1) * 8];
        let val =
            u32::from_str_radix(chunk, 16).map_err(|e| format!("invalid hex '{chunk}': {e}"))?;
        result[i] = M31::from_u32_unchecked(val % 0x7FFFFFFF);
    }
    Ok(result)
}

fn run_scan_command(cmd: &ScanCmd) {
    use stwo_ml::privacy::note_store::NoteStore;
    use stwo_ml::privacy::wallet::Wallet;

    let wallet_path = resolve_wallet_path(&cmd.wallet);
    let wallet = Wallet::load(&wallet_path, cmd.password.as_deref()).unwrap_or_else(|e| {
        eprintln!("Error loading wallet: {e}");
        process::exit(1);
    });

    let notes_path = NoteStore::default_path();
    #[cfg(feature = "audit-http")]
    let mut note_store = NoteStore::load(&notes_path, None).unwrap_or_else(|_| NoteStore {
        notes: Vec::new(),
        path: notes_path.clone(),
    });
    #[cfg(not(feature = "audit-http"))]
    let note_store = NoteStore::load(&notes_path, None).unwrap_or_else(|_| NoteStore {
        notes: Vec::new(),
        path: notes_path.clone(),
    });

    eprintln!();
    eprintln!("  prove-model scan");
    eprintln!("  ─────────────────");
    eprintln!("  address: {}", wallet.address());

    // Sync tree and update pending note indices
    #[cfg(feature = "audit-http")]
    {
        use stwo_ml::privacy::pool_client::PoolClient;
        use stwo_ml::privacy::tree_sync::TreeSync;

        let pool_config = build_pool_config(&cmd.priv_network, cmd.pool_contract.as_deref());
        if !pool_config.pool_address.is_empty() {
            let pool_client = PoolClient::new(pool_config);

            let cache_path = TreeSync::default_cache_path();
            let mut tree_sync = TreeSync::load_or_create(&cache_path).unwrap_or_else(|e| {
                eprintln!("  Warning: cache load failed ({e}), starting fresh sync");
                TreeSync::new()
            });

            eprintln!("  Syncing Merkle tree...");
            match tree_sync.sync(&pool_client) {
                Ok(sync_result) => {
                    eprintln!("  tree_size: {}", sync_result.total_leaves);
                    eprintln!("  new_events: {}", sync_result.events_added);

                    // Update pending notes with correct merkle indices
                    note_store
                        .update_merkle_indices(|commitment| tree_sync.find_commitment(commitment));
                    note_store.save(None).ok();
                }
                Err(e) => {
                    eprintln!("  Warning: tree sync failed: {e}");
                }
            }
        } else {
            eprintln!("  Note: pool address not configured, skipping tree sync");
        }
    }

    #[cfg(not(feature = "audit-http"))]
    {
        eprintln!("  Note: build with --features audit-http for tree sync");
    }

    eprintln!("  notes:      {}", note_store.notes.len());
    eprintln!("  balance(0): {}", note_store.balance(0));
    eprintln!("  balance(1): {}", note_store.balance(1));

    println!(
        "{{\"address\":\"{}\",\"notes\":{},\"balance_strk\":{},\"balance_eth\":{}}}",
        wallet.address(),
        note_store.notes.len(),
        note_store.balance(0),
        note_store.balance(1),
    );
}

fn parse_pubkey_hex(hex: &str) -> [M31; 4] {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    if hex.len() != 32 {
        eprintln!(
            "Error: public key hex must be 32 chars (4 x 8), got {}",
            hex.len()
        );
        process::exit(1);
    }
    let mut result = [M31::from(0u32); 4];
    for i in 0..4 {
        let chunk = &hex[i * 8..(i + 1) * 8];
        let val = u32::from_str_radix(chunk, 16).unwrap_or_else(|e| {
            eprintln!("Error: invalid hex in pubkey '{}': {}", chunk, e);
            process::exit(1);
        });
        result[i] = M31::from_u32_unchecked(val % 0x7FFFFFFF);
    }
    result
}

// ─── Audit Subcommand ─────────────────────────────────────────────────────

/// Parse a human-friendly time spec into Unix nanoseconds.
///
/// Supports:
/// - `"now"` → current time
/// - `"all"` → 0 (from beginning)
/// - `"Xh ago"`, `"Xm ago"`, `"Xd ago"` → relative to now
/// - Bare integer → Unix nanoseconds
fn parse_time_spec(spec: &str) -> u64 {
    let spec = spec.trim().to_lowercase();

    if spec == "now" {
        return std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
    }
    if spec == "all" {
        return 0;
    }

    // Relative: "1h ago", "30m ago", "2d ago", "90s ago"
    if spec.ends_with(" ago") || spec.ends_with("ago") {
        let trimmed = spec.trim_end_matches(" ago").trim_end_matches("ago").trim();
        let (num_str, unit) = if trimmed.ends_with('h') {
            (&trimmed[..trimmed.len() - 1], 3_600_000_000_000u64)
        } else if trimmed.ends_with('m') {
            (&trimmed[..trimmed.len() - 1], 60_000_000_000u64)
        } else if trimmed.ends_with('d') {
            (&trimmed[..trimmed.len() - 1], 86_400_000_000_000u64)
        } else if trimmed.ends_with('s') {
            (&trimmed[..trimmed.len() - 1], 1_000_000_000u64)
        } else {
            eprintln!(
                "Error: invalid relative time '{spec}'. Use Xh/Xm/Xd/Xs ago (e.g., '1h ago')"
            );
            process::exit(1);
        };

        let num: u64 = num_str.parse().unwrap_or_else(|_| {
            eprintln!("Error: invalid number in time spec '{spec}'");
            process::exit(1);
        });

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        return now_ns.saturating_sub(num * unit);
    }

    // Bare integer: Unix nanoseconds
    spec.parse::<u64>().unwrap_or_else(|_| {
        eprintln!(
            "Error: invalid time spec '{spec}'. \
             Use 'now', 'all', 'Xh ago', 'Xm ago', 'Xd ago', or a Unix timestamp in nanoseconds."
        );
        process::exit(1);
    })
}

/// Generate a deterministic but diverse M31 input matrix for capture iteration `i`.
///
/// Uses xorshift64-based PRNG seeded per iteration so each captured inference
/// exercises meaningfully different values through the forward pass.
fn generate_diverse_input(rows: usize, cols: usize, iteration: usize) -> M31Matrix {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_0000 ^ (iteration as u64 * 0x9E37_79B9_7F4A_7C15);
    let mut matrix = M31Matrix::new(rows, cols);
    let p = (1u32 << 31) - 1; // M31 modulus
    for i in 0..(rows * cols) {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let val = ((state >> 16) as u32) % p;
        // Scale to a reasonable input range (values 0..2^20 like quantized inputs)
        matrix.data[i] = M31::from(val % (1 << 20));
        // Extra mixing per element
        state = state
            .wrapping_add(i as u64)
            .wrapping_mul(0x517C_C1B7_2722_0A95);
    }
    matrix
}

/// Run the `prove-model capture` subcommand.
fn run_capture_command(cmd: &CaptureCmd) {
    use std::time::{SystemTime, UNIX_EPOCH};
    use stwo_ml::audit::capture::{CaptureHook, CaptureJob};
    use stwo_ml::audit::replay::execute_forward_pass;

    eprintln!();
    eprintln!("  prove-model capture");
    eprintln!("  ───────────────────");

    // ── Load model ──────────────────────────────────────────────────────
    let onnx = if let Some(ref model_dir) = cmd.model_dir {
        eprintln!("Loading model: {} (HuggingFace)", model_dir.display());
        stwo_ml::compiler::hf_loader::load_hf_model(model_dir, cmd.layers).unwrap_or_else(|e| {
            eprintln!("Error loading model directory: {e}");
            process::exit(1);
        })
    } else if let Some(ref model_path) = cmd.model {
        eprintln!("Loading model: {} (ONNX)", model_path.display());
        load_onnx(model_path).unwrap_or_else(|e| {
            eprintln!("Error loading ONNX model: {e}");
            process::exit(1);
        })
    } else {
        eprintln!("Error: specify --model (ONNX) or --model-dir (HuggingFace)");
        process::exit(1);
    };

    let graph = &onnx.graph;
    let weights = &onnx.weights;
    let (input_rows, input_cols) = onnx.input_shape;
    let model_name = cmd
        .model_name
        .as_deref()
        .unwrap_or(&onnx.metadata.name)
        .to_string();

    eprintln!("  model: {}", model_name);
    eprintln!("  input_shape: ({}, {})", input_rows, input_cols);
    eprintln!("  layers: {}", graph.num_layers());
    eprintln!("  count: {}", cmd.count);
    eprintln!("  log_dir: {}", cmd.log_dir.display());

    // ── Compute weight commitment ───────────────────────────────────────
    let weight_commitment_hex = if cmd.skip_commitment {
        eprintln!("  weight_commitment: 0x0 (skipped)");
        "0x0".to_string()
    } else {
        let commitment = compute_weight_commitment(weights, cmd.model_dir.as_deref());
        let hex = format!("{:#066x}", commitment);
        eprintln!("  weight_commitment: {}", hex);
        hex
    };

    // ── Create capture hook ─────────────────────────────────────────────
    let model_id_str = &cmd.model_id;
    let hook = CaptureHook::new(
        &cmd.log_dir,
        model_id_str,
        &weight_commitment_hex,
        &model_name,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error creating capture hook: {e}");
        process::exit(1);
    });

    // ── Load fixed input if provided ────────────────────────────────────
    let fixed_input = cmd.input.as_ref().map(|path| {
        let values = load_input_json(path);
        quantize_input(&values, input_rows, input_cols)
    });

    // ── Run forward passes ──────────────────────────────────────────────
    let t_start = Instant::now();
    eprintln!();

    for i in 0..cmd.count {
        let input = if let Some(ref fixed) = fixed_input {
            fixed.clone()
        } else {
            generate_diverse_input(input_rows, input_cols, i)
        };

        let t_inference = Instant::now();
        let output = execute_forward_pass(graph, &input, weights).unwrap_or_else(|e| {
            eprintln!("Error: forward pass failed on iteration {}: {e}", i);
            process::exit(1);
        });
        let latency_ms = t_inference.elapsed().as_millis() as u64;

        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        hook.record(CaptureJob {
            input_tokens: vec![], // No tokenization in capture mode.
            output_tokens: vec![],
            input_m31: input,
            output_m31: output,
            timestamp_ns: now_ns,
            latency_ms,
            gpu_device: "cpu".to_string(),
            tee_report_hash: "0x0".to_string(),
            task_category: Some("capture".to_string()),
            input_preview: Some(format!("capture_iter_{}", i)),
            output_preview: None,
        });

        eprintln!("  [{}/{}] forward pass: {}ms", i + 1, cmd.count, latency_ms);
    }

    // ── Flush and report ────────────────────────────────────────────────
    hook.flush();
    let total_ms = t_start.elapsed().as_millis();
    let entry_count = hook.entry_count();

    eprintln!();
    eprintln!("  Capture complete:");
    eprintln!("    entries: {}", entry_count);
    eprintln!("    total_time: {}ms", total_ms);
    eprintln!("    log_dir: {}", cmd.log_dir.display());
    eprintln!();

    // Machine-readable output for shell parsing.
    println!("CAPTURE_LOG_DIR={}", cmd.log_dir.display());
    println!("CAPTURE_COUNT={}", entry_count);
    println!("CAPTURE_MODEL={}", model_name);
}

/// Run the `prove-model audit` subcommand.
fn run_audit_command(cmd: &AuditCmd, _cli: &Cli) {
    use stwo_ml::audit::log::InferenceLog;
    use stwo_ml::audit::orchestrator::{run_audit, run_audit_dry, AuditPipelineConfig};
    use stwo_ml::audit::submit::{explorer_url, SubmitConfig};
    use stwo_ml::audit::types::{AuditRequest, ModelInfo};

    eprintln!();
    eprintln!("  prove-model audit");
    eprintln!("  ──────────────────");

    // ── Load inference log ────────────────────────────────────────────────
    eprintln!("Loading inference log: {}", cmd.log_dir.display());
    let log = InferenceLog::load(&cmd.log_dir).unwrap_or_else(|e| {
        eprintln!(
            "Error: cannot load inference log from '{}': {e}",
            cmd.log_dir.display()
        );
        eprintln!("Hint: the directory should contain meta.json, log.jsonl, and matrices.bin");
        process::exit(1);
    });

    let entry_count = log.entry_count();
    let model_id_from_log = log.model_id().to_string();
    let weight_commitment_from_log = log.weight_commitment().to_string();

    eprintln!("  entries: {}", entry_count);
    eprintln!("  model_id: {}", model_id_from_log);
    eprintln!("  weight_commitment: {}", weight_commitment_from_log);

    if entry_count == 0 {
        eprintln!("Error: inference log is empty — nothing to audit");
        process::exit(1);
    }

    // ── Load model ───────────────────────────────────────────────────────
    let onnx = if let Some(ref model_dir) = cmd.model_dir {
        eprintln!("Loading model: {} (HuggingFace)", model_dir.display());
        stwo_ml::compiler::hf_loader::load_hf_model(model_dir, cmd.layers).unwrap_or_else(|e| {
            eprintln!("Error loading model directory: {e}");
            process::exit(1);
        })
    } else if let Some(ref model_path) = cmd.model {
        eprintln!("Loading model: {} (ONNX)", model_path.display());
        load_onnx(model_path).unwrap_or_else(|e| {
            eprintln!("Error loading ONNX model: {e}");
            process::exit(1);
        })
    } else {
        eprintln!(
            "Error: specify --model (ONNX) or --model-dir (HuggingFace) for replay verification"
        );
        process::exit(1);
    };

    let graph = &onnx.graph;
    let weights = &onnx.weights;

    // ── Parse time window ────────────────────────────────────────────────
    let start_ns = parse_time_spec(&cmd.start);
    let end_ns = parse_time_spec(&cmd.end);

    if end_ns <= start_ns && cmd.start != "all" {
        eprintln!(
            "Error: end time ({}) must be after start time ({})",
            cmd.end, cmd.start
        );
        process::exit(1);
    }

    // Preview window
    let preview = log.query_window(start_ns, end_ns);
    eprintln!(
        "Audit window: {} inferences [{} → {}]",
        preview.entries.len(),
        if start_ns == 0 {
            "beginning".to_string()
        } else {
            format_ns(start_ns)
        },
        if end_ns == u64::MAX {
            "now".to_string()
        } else {
            format_ns(end_ns)
        },
    );

    if preview.entries.is_empty() {
        eprintln!("Error: no inferences found in the specified time window");
        process::exit(1);
    }

    // ── Build model info ─────────────────────────────────────────────────
    let model_id = cmd
        .model_id
        .as_deref()
        .unwrap_or(&model_id_from_log)
        .to_string();
    let model_name = cmd
        .model_name
        .as_deref()
        .unwrap_or(&onnx.metadata.name)
        .to_string();

    let model_info = ModelInfo {
        model_id: model_id.clone(),
        name: model_name.clone(),
        architecture: "mlp".to_string(), // Derived from graph structure
        parameters: format!("{}", onnx.metadata.num_parameters),
        layers: graph.num_layers() as u32,
        weight_commitment: weight_commitment_from_log.clone(),
    };

    // ── Build audit request ──────────────────────────────────────────────
    let request = AuditRequest {
        start_ns,
        end_ns,
        model_id: model_id.clone(),
        mode: cmd.mode.clone(),
        evaluate_semantics: cmd.evaluate,
        max_inferences: cmd.max_inferences,
        gpu_device: if cmd.gpu { Some(0) } else { None },
    };

    // ── Run audit ────────────────────────────────────────────────────────
    let t0 = Instant::now();

    if cmd.dry_run {
        // Dry-run: prove + evaluate + report, skip storage/on-chain
        eprintln!();
        eprintln!("Running dry-run audit...");
        let report = run_audit_dry(&log, graph, weights, request, model_info).unwrap_or_else(|e| {
            eprintln!("Error: audit failed: {e}");
            process::exit(1);
        });

        let elapsed = t0.elapsed();
        print_audit_report(&report, elapsed, &cmd.output);

        // Write report JSON
        let report_json = serde_json::to_string_pretty(&report).unwrap();
        std::fs::write(&cmd.output, &report_json).unwrap_or_else(|e| {
            eprintln!("Error writing report to '{}': {e}", cmd.output.display());
            process::exit(1);
        });
        eprintln!("Report written to: {}", cmd.output.display());
    } else {
        // Full pipeline
        eprintln!();
        eprintln!("Running full audit pipeline...");
        eprintln!("  evaluate: {}", cmd.evaluate);
        eprintln!("  privacy: {}", cmd.privacy);
        eprintln!("  submit: {}", cmd.submit);

        let submit_config = if cmd.submit {
            Some(SubmitConfig {
                contract_address: cmd.contract.clone(),
                network: cmd.network.clone(),
                ..SubmitConfig::default()
            })
        } else {
            None
        };

        // Construct encryption backend
        let encryption_backend: Option<Box<dyn stwo_ml::audit::types::AuditEncryption>> = match cmd
            .encryption
            .as_str()
        {
            "poseidon2" | "poseidon2_m31" => {
                Some(Box::new(stwo_ml::audit::encryption::Poseidon2M31Encryption))
            }
            "aes" => {
                #[cfg(feature = "aes-fallback")]
                {
                    Some(Box::new(stwo_ml::audit::encryption::Aes256GcmEncryption))
                }
                #[cfg(not(feature = "aes-fallback"))]
                {
                    eprintln!("Error: --encryption aes requires the 'aes-fallback' feature");
                    process::exit(1);
                }
            }
            "noop" => {
                eprintln!("Warning: --encryption noop is for testing only, NOT production-safe");
                Some(Box::new(stwo_ml::audit::encryption::NoopEncryption))
            }
            _ => None, // "none"
        };

        // Construct storage client for private audits
        #[allow(unused_assignments, unused_mut)]
        let mut storage_client: Option<stwo_ml::audit::storage::ArweaveClient> = None;
        if cmd.privacy != "public" && encryption_backend.is_some() {
            #[cfg(feature = "audit-http")]
            {
                let mut client = stwo_ml::audit::storage::ArweaveClient::new(
                    &cmd.arweave_gateway,
                    &cmd.arweave_bundler,
                    Box::new(stwo_ml::audit::storage::UreqTransport),
                );
                if let Some(ref token) = cmd.irys_token {
                    client = client.with_auth(token);
                }
                storage_client = Some(client);
            }

            if cmd.irys_token.is_none() {
                eprintln!("Warning: --privacy {} requires Arweave upload, but no --irys-token or IRYS_TOKEN provided", cmd.privacy);
                eprintln!("  Set IRYS_TOKEN=<your-token> or pass --irys-token <token>");
                eprintln!("  Get a token at https://irys.xyz");
            }
        }

        // Parse owner pubkey
        let owner_pubkey = cmd
            .owner_pubkey
            .as_deref()
            .map(|hex_str| {
                let hex_str = hex_str.trim_start_matches("0x");
                let mut bytes = Vec::with_capacity(hex_str.len() / 2);
                for i in (0..hex_str.len()).step_by(2) {
                    if let Ok(b) = u8::from_str_radix(&hex_str[i..i + 2], 16) {
                        bytes.push(b);
                    }
                }
                bytes
            })
            .unwrap_or_default();

        let config = AuditPipelineConfig {
            request,
            model_info,
            evaluate_semantics: cmd.evaluate,
            prove_evaluations: cmd.prove_evals,
            privacy_tier: cmd.privacy.clone(),
            owner_pubkey,
            submit_config,
            billing: None,
        };

        let result = run_audit(
            &log,
            graph,
            weights,
            &config,
            encryption_backend.as_deref(),
            storage_client.as_ref(),
        )
        .unwrap_or_else(|e| {
            eprintln!("Error: audit failed: {e}");
            process::exit(1);
        });

        let elapsed = t0.elapsed();
        print_audit_report(&result.report, elapsed, &cmd.output);

        // Write report JSON
        let report_json = serde_json::to_string_pretty(&result.report).unwrap();
        std::fs::write(&cmd.output, &report_json).unwrap_or_else(|e| {
            eprintln!("Error writing report to '{}': {e}", cmd.output.display());
            process::exit(1);
        });
        eprintln!("Report written to: {}", cmd.output.display());

        // Write calldata and submit on-chain
        if let Some(ref calldata) = result.calldata {
            let calldata_strs: Vec<String> =
                calldata.iter().map(|f| format!("0x{:x}", f)).collect();

            // Write calldata JSON for reference
            let calldata_path = cmd.output.with_extension("calldata.json");
            let calldata_json = serde_json::to_string_pretty(&calldata_strs).unwrap();
            std::fs::write(&calldata_path, &calldata_json).unwrap_or_else(|e| {
                eprintln!(
                    "Error writing calldata to '{}': {e}",
                    calldata_path.display()
                );
                process::exit(1);
            });
            eprintln!(
                "Calldata written to: {} ({} felts)",
                calldata_path.display(),
                calldata.len()
            );

            // Submit via Avnu paymaster (gasless)
            eprintln!();
            eprintln!("=== On-Chain Audit Submission (Avnu Paymaster) ===");
            eprintln!("  Contract: {}", cmd.contract);
            eprintln!("  Network:  {}", cmd.network);
            let submit_mode = if std::env::var("AVNU_API_KEY").is_ok() { "sponsored" } else { "direct" };
            eprintln!("  Fee:      {submit_mode}");

            // Resolve account address and private key from CLI or env
            let acct_addr = cmd
                .account
                .clone()
                .or_else(|| std::env::var("STARKNET_ACCOUNT").ok())
                .unwrap_or_default();
            let priv_key = cmd
                .private_key
                .clone()
                .or_else(|| std::env::var("STARKNET_PRIVATE_KEY").ok())
                .unwrap_or_default();

            if acct_addr.is_empty() || priv_key.is_empty() {
                eprintln!("  Error: --account + --private-key (or STARKNET_ACCOUNT + STARKNET_PRIVATE_KEY env vars) required for submission");
                eprintln!("  Calldata saved to: {}", calldata_path.display());
                eprintln!("  Submit manually:");
                eprintln!(
                    "    node scripts/pipeline/paymaster_submit.mjs --contract {} --calldata-file {} --account-address $STARKNET_ACCOUNT --private-key $STARKNET_PRIVATE_KEY --network {}",
                    cmd.contract, calldata_path.display(), cmd.network
                );
            } else {
                eprintln!("  Account: {}", acct_addr);

                // Write calldata as space-separated felts for the paymaster script
                let calldata_txt = cmd.output.with_extension("calldata.txt");
                let calldata_flat = calldata_strs.join(" ");
                std::fs::write(&calldata_txt, &calldata_flat).unwrap_or_else(|e| {
                    eprintln!("Error writing calldata txt: {e}");
                    process::exit(1);
                });

                // Find the paymaster script relative to the binary
                let paymaster_script = {
                    let exe = std::env::current_exe().unwrap_or_default();
                    let repo_root = exe
                        .parent() // target/release
                        .and_then(|p| p.parent()) // target
                        .and_then(|p| p.parent()); // repo root
                    match repo_root {
                        Some(root) => root.join("scripts/pipeline/paymaster_submit.mjs"),
                        None => PathBuf::from("scripts/pipeline/paymaster_submit.mjs"),
                    }
                };

                eprintln!("  Submitting via paymaster...");
                let mut submit_cmd = std::process::Command::new("node");
                submit_cmd
                    .arg(&paymaster_script)
                    .arg("--contract")
                    .arg(&cmd.contract)
                    .arg("--function")
                    .arg("submit_audit")
                    .arg("--calldata-file")
                    .arg(&calldata_txt)
                    .arg("--account-address")
                    .arg(&acct_addr)
                    .arg("--private-key")
                    .arg(&priv_key)
                    .arg("--network")
                    .arg(&cmd.network);

                // Select paymaster mode:
                //   - sponsored: if AVNU_API_KEY is set (dApp pays gas)
                //   - direct: default (account pays gas in STRK)
                // Note: "gasless" mode requires SNIP-9 compatible account (Argent/Braavos)
                if let Ok(api_key) = std::env::var("AVNU_API_KEY") {
                    submit_cmd.arg("--mode").arg("sponsored");
                    submit_cmd.env("AVNU_API_KEY", api_key);
                } else {
                    submit_cmd.arg("--mode").arg("direct");
                }

                let submit_result = submit_cmd.output();

                match submit_result {
                    Ok(output) if output.status.success() => {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        eprintln!("{}", stderr.trim());
                        eprintln!("  Audit submitted successfully!");

                        // Parse JSON output for tx hash
                        if let Ok(json) =
                            serde_json::from_str::<serde_json::Value>(stdout.trim())
                        {
                            if let Some(tx) = json.get("transaction_hash").and_then(|v| v.as_str())
                            {
                                eprintln!("  TX hash: {}", tx);
                            }
                            if let Some(url) = json.get("explorer_url").and_then(|v| v.as_str()) {
                                eprintln!("  Explorer: {}", url);
                            }
                        }
                    }
                    Ok(output) => {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        eprintln!("  Warning: submission may have failed:");
                        if !stderr.is_empty() {
                            eprintln!("    {}", stderr.trim());
                        }
                        if !stdout.is_empty() {
                            eprintln!("    {}", stdout.trim());
                        }
                        eprintln!("  Calldata saved to: {}", calldata_txt.display());
                    }
                    Err(e) => {
                        eprintln!("  Error: could not run paymaster script: {e}");
                        eprintln!(
                            "  Make sure Node.js is installed and starknet package is available"
                        );
                        eprintln!("  Calldata saved to: {}", calldata_txt.display());
                    }
                }
            }
        }

        // Storage receipt
        if let Some(ref receipt) = result.storage_receipt {
            eprintln!("Arweave: stored as tx {}", receipt.tx_id);
        }

        eprintln!(
            "Pipeline: {:.2}s total ({} ms)",
            elapsed.as_secs_f64(),
            result.total_time_ms,
        );
    }
}

// ─── Retrieve Subcommand ─────────────────────────────────────────────────

fn run_retrieve_command(cmd: &RetrieveCmd) {
    #[cfg(not(feature = "audit-http"))]
    let _ = cmd;

    #[cfg(not(feature = "audit-http"))]
    {
        eprintln!();
        eprintln!("  prove-model retrieve");
        eprintln!("  ─────────────────────");
        eprintln!("Error: audit-http feature not enabled.");
        eprintln!(
            "Build with: cargo build --release --bin prove-model --features cli,audit,audit-http"
        );
        process::exit(1);
    }

    #[cfg(feature = "audit-http")]
    {
        use stwo_ml::audit::encryption::fetch_and_decrypt;

        eprintln!();
        eprintln!("  prove-model retrieve");
        eprintln!("  ─────────────────────");
        eprintln!("  TX ID:      {}", cmd.tx_id);
        eprintln!("  Encryption: {}", cmd.encryption);
        eprintln!("  Gateway:    {}", cmd.arweave_gateway);
        eprintln!("  Output:     {}", cmd.output.display());

        // Parse private key
        let privkey_hex = cmd.privkey.trim_start_matches("0x");
        let privkey: Vec<u8> = (0..privkey_hex.len())
            .step_by(2)
            .filter_map(|i| u8::from_str_radix(&privkey_hex[i..i + 2], 16).ok())
            .collect();

        if privkey.is_empty() {
            eprintln!("Error: invalid --privkey hex");
            process::exit(1);
        }

        // Parse recipient address (defaults to first 20 bytes of pubkey derivation)
        let address = cmd
            .address
            .as_deref()
            .map(|hex| {
                let hex = hex.trim_start_matches("0x");
                (0..hex.len())
                    .step_by(2)
                    .filter_map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
                    .collect::<Vec<u8>>()
            })
            .unwrap_or_else(|| privkey.clone()); // Fallback: use privkey as address

        // Build encryption backend
        let encryption: Box<dyn stwo_ml::audit::types::AuditEncryption> = match cmd
            .encryption
            .as_str()
        {
            "poseidon2" | "poseidon2_m31" => {
                Box::new(stwo_ml::audit::encryption::Poseidon2M31Encryption)
            }
            "aes" => {
                #[cfg(feature = "aes-fallback")]
                {
                    Box::new(stwo_ml::audit::encryption::Aes256GcmEncryption)
                }
                #[cfg(not(feature = "aes-fallback"))]
                {
                    eprintln!("Error: --encryption aes requires the 'aes-fallback' feature");
                    process::exit(1);
                }
            }
            "noop" => {
                eprintln!("Warning: --encryption noop is for testing only, NOT production-safe");
                Box::new(stwo_ml::audit::encryption::NoopEncryption)
            }
            other => {
                eprintln!(
                    "Error: unknown encryption mode '{}' (expected 'poseidon2', 'aes', or 'noop')",
                    other
                );
                process::exit(1);
            }
        };

        // Build storage client (read-only, bundler not used for downloads)
        let transport: Box<dyn stwo_ml::audit::storage::HttpTransport> =
            Box::new(stwo_ml::audit::storage::UreqTransport);
        let storage = stwo_ml::audit::storage::ArweaveClient::new(
            &cmd.arweave_gateway,
            "https://node1.irys.xyz",
            transport,
        );

        eprintln!("Fetching and decrypting...");

        let report = fetch_and_decrypt(
            &cmd.tx_id,
            &address,
            &privkey,
            encryption.as_ref(),
            &storage,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error: failed to retrieve audit report: {e}");
            process::exit(1);
        });

        // Write decrypted report
        let report_json = serde_json::to_string_pretty(&report).unwrap();
        if let Some(parent) = cmd.output.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&cmd.output, &report_json).unwrap_or_else(|e| {
            eprintln!("Error writing report to '{}': {e}", cmd.output.display());
            process::exit(1);
        });

        eprintln!();
        eprintln!(
            "Audit report decrypted and written to: {}",
            cmd.output.display()
        );
        eprintln!("  Audit ID:     {}", report.audit_id);
        eprintln!("  Model:        {}", report.model.name);
        eprintln!(
            "  Inferences:   {}",
            report.inference_summary.total_inferences
        );
        eprintln!(
            "  Time window:  {} → {}",
            report.time_window.start_epoch_ns, report.time_window.end_epoch_ns
        );
    }
}

// ─── Report Formatting Helpers ───────────────────────────────────────────

const W: usize = 72;

fn section_line(title: &str) -> String {
    let prefix = format!("─── {} ", title);
    let prefix_w = prefix.chars().count();
    let fill = W.saturating_sub(prefix_w);
    format!("{}{}", prefix, "─".repeat(fill))
}

fn section_line_info(title: &str, info: &str) -> String {
    let prefix = format!("─── {} ", title);
    let suffix = format!(" {} ───", info);
    let prefix_w = prefix.chars().count();
    let suffix_w = suffix.chars().count();
    let fill = W.saturating_sub(prefix_w + suffix_w);
    format!("{}{}{}", prefix, "─".repeat(fill.max(1)), suffix)
}

fn bar_chart(value: u32, max: u32, width: usize) -> String {
    if max == 0 {
        return "\u{2591}".repeat(width); // ░
    }
    let filled = ((value as f64 / max as f64) * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(empty))
}

fn duration_human(secs: u64) -> String {
    if secs == 0 {
        "< 1s".to_string()
    } else if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("{}m", m)
        } else {
            format!("{}m {}s", m, s)
        }
    } else {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        if m == 0 {
            format!("{}h", h)
        } else {
            format!("{}h {}m", h, m)
        }
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Print a well-structured audit report to stderr.
fn print_audit_report(
    report: &stwo_ml::audit::types::AuditReport,
    elapsed: std::time::Duration,
    output: &std::path::Path,
) {
    // Box inner width = W - 2 (for ╔/╗ corners), so total display = W
    let inner = W - 2;
    let border: String = "\u{2550}".repeat(inner);
    let footer: String = "\u{2550}".repeat(W);

    // ── Title ────────────────────────────────────────────────────────────
    eprintln!();
    eprintln!("\u{2554}{}\u{2557}", border); // ╔═══╗
    let title = "VERIFIABLE INFERENCE AUDIT REPORT";
    let pad = inner.saturating_sub(title.len());
    let lpad = pad / 2;
    let rpad = pad - lpad;
    eprintln!(
        "\u{2551}{}{}{}\u{2551}",
        " ".repeat(lpad),
        title,
        " ".repeat(rpad)
    );
    eprintln!("\u{255a}{}\u{255d}", border); // ╚═══╝

    eprintln!();
    eprintln!("  Audit       {}", report.audit_id);
    eprintln!("  Generated   {}", report.metadata.generated_at);

    // ── Model ────────────────────────────────────────────────────────────
    eprintln!();
    eprintln!("{}", section_line("MODEL"));
    eprintln!();
    eprintln!("  {:<56}ID  {}", report.model.name, report.model.model_id);
    eprintln!(
        "  {} \u{00b7} {} parameters \u{00b7} {} layers",
        report.model.architecture, report.model.parameters, report.model.layers,
    );
    eprintln!("  Weight  {}", report.model.weight_commitment);

    // ── Time Window ──────────────────────────────────────────────────────
    eprintln!();
    let dur = duration_human(report.time_window.duration_seconds);
    eprintln!("{}", section_line_info("TIME WINDOW", &dur));
    eprintln!();
    eprintln!(
        "  {}  \u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}  {}",
        report.time_window.start, report.time_window.end,
    );

    // ── Inference Summary ────────────────────────────────────────────────
    let s = &report.inference_summary;
    eprintln!();
    let inf_label = format!("{} inferences", s.total_inferences);
    eprintln!("{}", section_line_info("INFERENCE SUMMARY", &inf_label));
    eprintln!();
    eprintln!(
        "  {:<38}Throughput  {:.1} tok/s",
        format!(
            "Tokens    {} in \u{00b7} {} out",
            s.total_input_tokens, s.total_output_tokens
        ),
        s.throughput_tokens_per_sec,
    );
    let latency_str = format!(
        "Latency   avg {}ms \u{00b7} p95 {}ms",
        s.avg_latency_ms, s.p95_latency_ms
    );
    if !s.categories.is_empty() {
        let mut cats: Vec<_> = s.categories.iter().collect();
        cats.sort_by(|a, b| b.1.cmp(a.1));
        let cat_str: Vec<String> = cats.iter().map(|(k, v)| format!("{} ({})", k, v)).collect();
        eprintln!("  {:<38}Categories  {}", latency_str, cat_str.join(", "));
    } else {
        eprintln!("  {}", latency_str);
    }

    // ── Semantic Evaluation ──────────────────────────────────────────────
    if let Some(ref sem) = report.semantic_evaluation {
        eprintln!();
        let sem_info = format!(
            "{} \u{00b7} {}/{} checked",
            sem.method, sem.evaluated_count, s.total_inferences,
        );
        eprintln!("{}", section_line_info("SEMANTIC EVALUATION", &sem_info));
        eprintln!();
        eprintln!("  Average Quality   {:.1}%", sem.avg_quality_score * 100.0);
        eprintln!();

        // Bar chart — scale bars to max bucket
        let max_bucket = sem
            .excellent_count
            .max(sem.good_count)
            .max(sem.fair_count)
            .max(sem.poor_count)
            .max(1); // avoid div-by-zero in display; bars are all empty if max=1 and val=0
        let bw = 20;
        eprintln!(
            "  Excellent  {}  {:>3}",
            bar_chart(sem.excellent_count, max_bucket, bw),
            sem.excellent_count
        );
        eprintln!(
            "  Good       {}  {:>3}",
            bar_chart(sem.good_count, max_bucket, bw),
            sem.good_count
        );
        eprintln!(
            "  Fair       {}  {:>3}",
            bar_chart(sem.fair_count, max_bucket, bw),
            sem.fair_count
        );
        eprintln!(
            "  Poor       {}  {:>3}",
            bar_chart(sem.poor_count, max_bucket, bw),
            sem.poor_count
        );

        eprintln!();
        let det_total = sem.deterministic_pass + sem.deterministic_fail;
        if det_total > 0 {
            let pct = if det_total > 0 {
                (sem.deterministic_pass as f64 / det_total as f64 * 100.0) as u32
            } else {
                0
            };
            eprintln!(
                "  Deterministic   {} pass \u{00b7} {} fail ({}% pass rate)",
                sem.deterministic_pass, sem.deterministic_fail, pct,
            );
        }
        if sem.evaluations_proved {
            eprintln!("  Eval proofs     ZK-proved evaluation forward passes");
        }
        if let Some(ref root) = sem.eval_merkle_root {
            eprintln!("  Eval root       {}", root);
        }

        // Per-inference table
        if !sem.per_inference.is_empty() {
            let show_count = sem.per_inference.len().min(10);
            eprintln!();
            eprintln!(
                "  Per-Inference ({}/{}):",
                show_count,
                sem.per_inference.len()
            );
            eprintln!(
                "  {:>4}  {:>5}  {:>7}  {:>5}  {}",
                "#", "Seq", "Score", "Det", "Status"
            );
            eprintln!(
                "  {:─>4}  {:─>5}  {:─>7}  {:─>5}  {:─>8}",
                "", "", "", "", ""
            );

            for eval in sem.per_inference.iter().take(show_count) {
                let score_str = eval
                    .semantic_score
                    .map(|sc| format!("{:.1}%", sc * 100.0))
                    .unwrap_or_else(|| "\u{2014}".to_string()); // —
                let det_pass = eval
                    .deterministic_checks
                    .iter()
                    .filter(|c| c.passed)
                    .count();
                let det_total = eval.deterministic_checks.len();
                let det_str = format!("{}/{}", det_pass, det_total);
                let status = if eval.deterministic_checks.iter().all(|c| c.passed) {
                    if eval.evaluation_proved {
                        "proved"
                    } else {
                        "ok"
                    }
                } else {
                    "FAIL"
                };

                eprintln!(
                    "  {:>4}  {:>5}  {:>7}  {:>5}  {}",
                    eval.sequence, eval.sequence, score_str, det_str, status,
                );

                // Show failing check details inline
                for check in &eval.deterministic_checks {
                    if !check.passed {
                        let detail = check.detail.as_deref().unwrap_or("");
                        eprintln!(
                            "  {:>4}  {:>5}  {:>7}  {:>5}  \u{2514}\u{2500} {} {}",
                            "", "", "", "", check.check_type, detail,
                        );
                    }
                }
            }
            if sem.per_inference.len() > show_count {
                eprintln!(
                    "  {:>4}  ... {} more (see report JSON)",
                    "",
                    sem.per_inference.len() - show_count
                );
            }
        }
    } else {
        eprintln!();
        eprintln!("{}", section_line("SEMANTIC EVALUATION"));
        eprintln!();
        eprintln!("  Skipped (use --evaluate to enable)");
    }

    // ── Proof & Infrastructure ───────────────────────────────────────────
    let proof = &report.proof;
    let infra = &report.infrastructure;
    eprintln!();
    eprintln!("{}", section_line("PROOF & INFRASTRUCTURE"));
    eprintln!();

    // Proof line
    let mut proof_parts: Vec<String> = vec![proof.mode.clone()];
    if proof.proving_time_seconds > 0 {
        proof_parts.push(duration_human(proof.proving_time_seconds));
    }
    if proof.proof_size_bytes > 0 {
        proof_parts.push(format_bytes(proof.proof_size_bytes));
    }

    let mut gpu_str = infra.gpu_device.clone();
    if infra.gpu_count > 1 {
        gpu_str = format!("{} (\u{00d7}{})", gpu_str, infra.gpu_count);
    }

    eprintln!(
        "  {:<38}GPU         {}",
        format!("Proof       {}", proof_parts.join(" \u{00b7} ")),
        gpu_str,
    );
    let tee_str = if infra.tee_active {
        "active"
    } else {
        "inactive"
    };
    eprintln!(
        "  {:<38}TEE         {}",
        format!("Prover      stwo-ml v{}", infra.prover_version),
        tee_str,
    );
    if !infra.cuda_version.is_empty() {
        eprintln!("  CUDA        {}", infra.cuda_version);
    }
    if infra.tee_active {
        if let Some(ref hash) = infra.tee_attestation_hash {
            eprintln!("  TEE hash    {}", hash);
        }
    }

    // On-chain status
    if proof.on_chain_tx.is_some() || proof.arweave_tx_id.is_some() {
        eprintln!();
        if let Some(ref tx) = proof.on_chain_tx {
            let verified = proof.on_chain_verified.unwrap_or(false);
            eprintln!(
                "  On-chain    {}  {}",
                tx,
                if verified { "(verified)" } else { "(pending)" },
            );
        }
        if let Some(ref arweave) = proof.arweave_tx_id {
            eprintln!("  Arweave     {}", arweave);
        }
        if let Some(ref record_id) = proof.audit_record_id {
            eprintln!("  Record ID   {}", record_id);
        }
    }

    // ── Commitments ──────────────────────────────────────────────────────
    eprintln!();
    eprintln!("{}", section_line("COMMITMENTS"));
    eprintln!();
    eprintln!("  IO root      {}", report.commitments.io_merkle_root);
    eprintln!(
        "  Log root     {}",
        report.commitments.inference_log_merkle_root
    );
    eprintln!("  Weight       {}", report.commitments.weight_commitment);
    eprintln!(
        "  Chain        {}",
        report.commitments.combined_chain_commitment
    );
    eprintln!("  Report hash  {}", report.commitments.audit_report_hash);

    // ── Inferences ───────────────────────────────────────────────────────
    if !report.inferences.is_empty() {
        let show_count = report.inferences.len().min(5);
        eprintln!();
        let inf_info = format!("showing {} of {}", show_count, report.inferences.len());
        eprintln!("{}", section_line_info("INFERENCES", &inf_info));

        for entry in report.inferences.iter().take(show_count) {
            eprintln!();
            let score_str = entry
                .semantic_score
                .map(|sc| format!("  score {:.0}%", sc * 100.0))
                .unwrap_or_default();
            let cat = entry.category.as_deref().unwrap_or("-");
            // Extract just the time portion from ISO timestamp
            let time_short = entry
                .timestamp
                .find('T')
                .map(|i| &entry.timestamp[i + 1..entry.timestamp.len().min(i + 9)])
                .unwrap_or(&entry.timestamp);
            eprintln!(
                "  #{:<3} {}  {}ms  {}{}",
                entry.index, time_short, entry.latency_ms, cat, score_str,
            );
            if let Some(ref preview) = entry.input_preview {
                let truncated = if preview.len() > 64 {
                    &preview[..64]
                } else {
                    preview.as_str()
                };
                eprintln!("       \u{25b8} {}", truncated);
            }
            if let Some(ref preview) = entry.output_preview {
                let truncated = if preview.len() > 64 {
                    &preview[..64]
                } else {
                    preview.as_str()
                };
                eprintln!("       \u{25c2} {}", truncated);
            }
        }
        if report.inferences.len() > show_count {
            eprintln!();
            eprintln!("  ... {} more", report.inferences.len() - show_count);
        }
    }

    // ── Privacy ──────────────────────────────────────────────────────────
    if let Some(ref priv_info) = report.privacy {
        eprintln!();
        eprintln!("{}", section_line("PRIVACY"));
        eprintln!();
        eprintln!(
            "  {:<38}Encryption  {}",
            format!("Tier        {}", priv_info.tier),
            priv_info.encryption_scheme,
        );
        if let Some(ref tx) = priv_info.arweave_tx_id {
            eprintln!("  Storage TX  {}", tx);
        }
    }

    // ── Footer ───────────────────────────────────────────────────────────
    eprintln!();
    eprintln!("{}", footer);
    let time_str = format!("Completed in {:.2}s", elapsed.as_secs_f64());
    let file_str = format!("{}", output.display());
    let gap = W.saturating_sub(2 + time_str.len() + file_str.len());
    eprintln!("  {}{}{}", time_str, " ".repeat(gap), file_str);
    eprintln!("{}", footer);
}

/// Format nanosecond timestamp as human-readable string.
fn format_ns(ns: u64) -> String {
    let secs = ns / 1_000_000_000;
    let datetime = std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs);
    let dur = datetime
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let total_secs = dur.as_secs();
    let hours = (total_secs % 86400) / 3600;
    let mins = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    format!("{}:{:02}:{:02} UTC ({})", hours, mins, s, secs)
}

/// Check if a cached weight commitment exists with valid fingerprint.
fn check_commitment_cache(model_dir: Option<&std::path::Path>) -> Option<FieldElement> {
    let d = model_dir?;
    let fingerprint = compute_fingerprint(d)?;
    let cache_path = d.join(format!(".weight_commitment_{fingerprint}.hex"));
    let hex = std::fs::read_to_string(&cache_path).ok()?;
    FieldElement::from_hex_be(hex.trim().trim_start_matches("0x")).ok()
}

const WEIGHT_COMMITMENT_SCHEME_VERSION: &str = "v3_gpu_m31_segments";

/// Compute a fingerprint from safetensors file metadata (sizes + mtimes).
fn compute_fingerprint(model_dir: &std::path::Path) -> Option<String> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    WEIGHT_COMMITMENT_SCHEME_VERSION.hash(&mut hasher);
    std::env::var("STWO_WEIGHT_COMMIT_SEGMENTS")
        .unwrap_or_else(|_| "".to_string())
        .hash(&mut hasher);
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    for f in &files {
        f.to_str().unwrap_or("").hash(&mut hasher);
        if let Ok(meta) = std::fs::metadata(f) {
            meta.len().hash(&mut hasher);
            if let Ok(mtime) = meta.modified() {
                mtime.hash(&mut hasher);
            }
        }
    }
    Some(format!("{:016x}", hasher.finish()))
}

#[cfg(feature = "cuda-runtime")]
fn gpu_commit_flag_enabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && v != "0" && v != "false" && v != "off"
        }
        Err(_) => false,
    }
}

#[cfg(feature = "cuda-runtime")]
fn gpu_commit_strict_mode() -> bool {
    gpu_commit_flag_enabled("STWO_GPU_COMMIT_STRICT")
}

#[cfg(feature = "cuda-runtime")]
fn gpu_commit_hardening_enabled() -> bool {
    gpu_commit_flag_enabled("STWO_GPU_COMMIT_HARDEN")
}

#[cfg(feature = "cuda-runtime")]
fn field_element_to_u64_limbs(fe: &FieldElement) -> [u64; 4] {
    let bytes = fe.to_bytes_be();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let offset = 24 - i * 8;
        let mut val = 0u64;
        for j in 0..8 {
            val = (val << 8) | bytes[offset + j] as u64;
        }
        *limb = val;
    }
    limbs
}

#[cfg(feature = "cuda-runtime")]
fn u64_limbs_to_field_element(limbs: &[u64]) -> Result<FieldElement, String> {
    if limbs.len() != 4 {
        return Err(format!("expected 4 limbs, got {}", limbs.len()));
    }
    let mut bytes = [0u8; 32];
    for i in 0..4 {
        let limb = limbs[3 - i];
        let offset = i * 8;
        bytes[offset..offset + 8].copy_from_slice(&limb.to_be_bytes());
    }
    // Match STWO GPU Poseidon conversion: keep only the low 251 bits.
    bytes[0] &= 0x07;
    FieldElement::from_bytes_be(&bytes).map_err(|_| {
        format!(
            "invalid felt252 limbs after top-bit mask: [{:#x}, {:#x}, {:#x}, {:#x}]",
            limbs[0], limbs[1], limbs[2], limbs[3]
        )
    })
}

fn pack_m31_chunk_to_felt(values: &[M31]) -> FieldElement {
    // Pack little-endian base-2^31 digits:
    // felt = v0 + v1*2^31 + v2*2^62 + ...
    // This is mathematically identical to the previous Horner packing, but
    // avoids expensive field multiplications on CPU.
    let mut limbs = [0u64; 4]; // little-endian 256-bit
    for (i, &v) in values.iter().enumerate() {
        let bit_pos = i * 31;
        let limb_idx = bit_pos / 64;
        let bit_off = bit_pos % 64;
        let val = v.0 as u64;

        limbs[limb_idx] |= val << bit_off;
        if bit_off + 31 > 64 && limb_idx + 1 < 4 {
            limbs[limb_idx + 1] |= val >> (64 - bit_off);
        }
    }

    let mut bytes = [0u8; 32];
    for i in 0..4 {
        let limb = limbs[3 - i];
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    FieldElement::from_bytes_be(&bytes).expect("packed 217-bit value always fits felt252 field")
}

#[cfg(feature = "cuda-runtime")]
struct GpuCommitHasher {
    d_round_constants: cudarc::driver::CudaSlice<u64>,
}

#[cfg(feature = "cuda-runtime")]
impl GpuCommitHasher {
    fn new() -> Result<Self, String> {
        let executor = get_cuda_executor().map_err(|e| format!("CUDA init: {e}"))?;
        let d_round_constants = upload_poseidon252_round_constants(&executor.device)
            .map_err(|e| format!("Poseidon252 round constants upload: {e}"))?;
        Ok(Self { d_round_constants })
    }

    fn hash_segments(
        &self,
        segments: &[Vec<FieldElement>],
        chunk_size: usize,
    ) -> Result<Vec<FieldElement>, String> {
        if segments.is_empty() {
            return Ok(Vec::new());
        }

        let total_inputs: usize = segments.iter().map(|seg| seg.len()).sum();
        let mut offsets = Vec::with_capacity(segments.len());
        let mut lengths = Vec::with_capacity(segments.len());
        let mut inputs_limbs = Vec::with_capacity(total_inputs * 4);

        let mut cursor: u32 = 0;
        for seg in segments {
            let len_u32 = u32::try_from(seg.len())
                .map_err(|_| format!("segment too large for u32: {}", seg.len()))?;
            offsets.push(cursor);
            lengths.push(len_u32);
            cursor = cursor
                .checked_add(len_u32)
                .ok_or_else(|| "segment offsets overflow u32".to_string())?;

            for fe in seg {
                inputs_limbs.extend_from_slice(&field_element_to_u64_limbs(fe));
            }
        }

        let executor = get_cuda_executor().map_err(|e| format!("CUDA init: {e}"))?;
        let output_limbs = executor
            .execute_poseidon252_hash_many_chunked(
                &inputs_limbs,
                &offsets,
                &lengths,
                chunk_size,
                &self.d_round_constants,
            )
            .map_err(|e| format!("poseidon252_hash_many_chunked: {e}"))?;

        let mut out = Vec::with_capacity(segments.len());
        for chunk in output_limbs.chunks_exact(4) {
            out.push(u64_limbs_to_field_element(chunk)?);
        }
        Ok(out)
    }

    fn hash_segments_m31(
        &self,
        segments: &[&[M31]],
        chunk_size: usize,
    ) -> Result<Vec<FieldElement>, String> {
        if segments.is_empty() {
            return Ok(Vec::new());
        }

        let total_inputs: usize = segments.iter().map(|seg| seg.len()).sum();
        let mut offsets = Vec::with_capacity(segments.len());
        let mut lengths = Vec::with_capacity(segments.len());
        let mut inputs_m31 = Vec::with_capacity(total_inputs);

        let mut cursor: u32 = 0;
        for seg in segments {
            let len_u32 = u32::try_from(seg.len())
                .map_err(|_| format!("segment too large for u32: {}", seg.len()))?;
            offsets.push(cursor);
            lengths.push(len_u32);
            cursor = cursor
                .checked_add(len_u32)
                .ok_or_else(|| "segment offsets overflow u32".to_string())?;
            inputs_m31.extend(seg.iter().map(|v| v.0));
        }

        let executor = get_cuda_executor().map_err(|e| format!("CUDA init: {e}"))?;
        let output_limbs = executor
            .execute_poseidon252_hash_many_chunked_m31(
                &inputs_m31,
                &offsets,
                &lengths,
                chunk_size,
                &self.d_round_constants,
            )
            .map_err(|e| format!("poseidon252_hash_many_chunked_m31: {e}"))?;

        let mut out = Vec::with_capacity(segments.len());
        for chunk in output_limbs.chunks_exact(4) {
            out.push(u64_limbs_to_field_element(chunk)?);
        }
        Ok(out)
    }
}

/// Compute weight commitment: Poseidon hash of all weight matrices.
/// Packed 7:1 (7 M31 values per FieldElement) + parallel Merkle segments.
/// Caches result with fingerprint for instant validated reuse on subsequent runs.
fn compute_weight_commitment(
    weights: &stwo_ml::compiler::graph::GraphWeights,
    model_dir: Option<&std::path::Path>,
) -> FieldElement {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    let n_weights = weights.weights.len();
    eprintln!(
        "[BG] Computing weight commitment ({} matrices, packed 7:1, parallel)...",
        n_weights
    );
    eprintln!("[BG]   First run — will cache with fingerprint for instant validated reuse.");
    let t_commit = Instant::now();
    let progress_every = std::env::var("STWO_WEIGHT_PROGRESS_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(1);

    let weight_list: Vec<_> = weights.weights.iter().collect();
    let done_count = Arc::new(AtomicUsize::new(0));
    let total = weight_list.len();
    eprintln!(
        "[BG]   Progress update cadence: every {} matrix/matrices",
        progress_every
    );
    let ticker_stop = Arc::new(AtomicBool::new(false));
    let ticker_done_count = Arc::clone(&done_count);
    let ticker_stop_flag = Arc::clone(&ticker_stop);
    let ticker = std::thread::spawn(move || {
        while !ticker_stop_flag.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_secs(15));
            if ticker_stop_flag.load(Ordering::Relaxed) {
                break;
            }
            let finished = ticker_done_count.load(Ordering::Relaxed);
            let elapsed = t_commit.elapsed().as_secs_f64();
            let pct = if total > 0 {
                100.0 * finished as f64 / total as f64
            } else {
                100.0
            };
            eprintln!(
                "[BG]   Working... {}/{} ({:.1}%, {:.1}s elapsed)",
                finished, total, pct, elapsed,
            );
        }
    });

    let target_segments = std::env::var("STWO_WEIGHT_COMMIT_SEGMENTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(4096);
    let chunk_size = 4096usize;
    eprintln!(
        "[BG]   Segment parallelism target: {} segments per matrix",
        target_segments
    );

    let hash_packed_segment_cpu = |packed: &[FieldElement]| -> FieldElement {
        if packed.is_empty() {
            return FieldElement::ZERO;
        }
        let mut hash_inputs: Vec<FieldElement> = Vec::with_capacity(chunk_size + 1);
        let mut running = FieldElement::ZERO;
        for chunk in packed.chunks(chunk_size) {
            hash_inputs.clear();
            hash_inputs.push(running);
            hash_inputs.extend_from_slice(chunk);
            running = starknet_crypto::poseidon_hash_many(&hash_inputs);
        }
        running
    };

    #[cfg(feature = "cuda-runtime")]
    let mut gpu_hasher = match GpuCommitHasher::new() {
        Ok(hasher) => {
            let strict = gpu_commit_strict_mode();
            let harden = gpu_commit_hardening_enabled();
            eprintln!(
                "[BG]   GPU Poseidon hash-many enabled for segment hashing (strict={}, harden={})",
                strict, harden
            );
            if harden {
                eprintln!(
                    "[BG]   Hardening enabled: GPU results are cross-checked on CPU (expect slower first run)"
                );
            }
            Some(hasher)
        }
        Err(e) => {
            if gpu_commit_strict_mode() {
                panic!("GPU commitment strict mode enabled, but GPU hasher init failed: {e}");
            }
            eprintln!("[BG]   GPU Poseidon unavailable ({e}), falling back to CPU segment hashing");
            None
        }
    };

    let per_matrix_hashes: Vec<FieldElement> = weight_list.iter().enumerate().map(|(idx, (layer_id, w))| {
        eprintln!(
            "[BG]   Matrix {}/{} start: layer={}, shape={}x{} ({} elements)",
            idx + 1,
            total,
            layer_id,
            w.rows,
            w.cols,
            w.data.len(),
        );
        let segment_count = target_segments.min(w.data.len().max(1));
        let seg_size = (w.data.len() + segment_count - 1) / segment_count;
        let raw_segments: Vec<&[M31]> = w.data
            .chunks(seg_size.max(1))
            .collect();

        let cpu_hash_segments = || -> Vec<FieldElement> {
            raw_segments
                .par_iter()
                .map(|segment| {
                    let packed: Vec<FieldElement> = segment
                        .chunks(7)
                        .map(pack_m31_chunk_to_felt)
                        .collect();
                    hash_packed_segment_cpu(&packed)
                })
                .collect()
        };

        let segment_hashes: Vec<FieldElement> = {
            #[cfg(feature = "cuda-runtime")]
            {
                if let Some(hasher) = gpu_hasher.as_ref() {
                    match hasher.hash_segments_m31(&raw_segments, chunk_size) {
                        Ok(gpu_hashes) => {
                            if gpu_commit_hardening_enabled() {
                                let cpu_hashes = cpu_hash_segments();
                                if gpu_hashes != cpu_hashes {
                                    panic!(
                                        "GPU weight commitment hardening failed: segment hash mismatch on layer {}",
                                        layer_id
                                    );
                                }
                            }
                            gpu_hashes
                        }
                        Err(e) => {
                            if gpu_commit_strict_mode() {
                                panic!(
                                    "GPU commitment strict mode enabled, but GPU hashing failed for layer {}: {}",
                                    layer_id, e
                                );
                            }
                            eprintln!(
                                "[BG]   Layer {} GPU Poseidon hash-many failed ({}), using CPU fallback",
                                layer_id, e
                            );
                            gpu_hasher = None;
                            cpu_hash_segments()
                        }
                    }
                } else {
                    cpu_hash_segments()
                }
            }
            #[cfg(not(feature = "cuda-runtime"))]
            {
                cpu_hash_segments()
            }
        };

        let mut final_inputs: Vec<FieldElement> = Vec::with_capacity(segment_hashes.len() + 3);
        final_inputs.push(FieldElement::from(*layer_id as u64));
        final_inputs.push(FieldElement::from((w.rows as u64) << 32 | w.cols as u64));
        final_inputs.push(FieldElement::from(raw_segments.len() as u64));
        final_inputs.extend_from_slice(&segment_hashes);
        let matrix_hash = {
            #[cfg(feature = "cuda-runtime")]
            {
                if let Some(hasher) = gpu_hasher.as_ref() {
                    match hasher.hash_segments(std::slice::from_ref(&final_inputs), chunk_size) {
                        Ok(hashes) => {
                            let gpu_hash = hashes[0];
                            if gpu_commit_hardening_enabled() {
                                let cpu_hash = starknet_crypto::poseidon_hash_many(&final_inputs);
                                if gpu_hash != cpu_hash {
                                    panic!(
                                        "GPU weight commitment hardening failed: matrix hash mismatch on layer {}",
                                        layer_id
                                    );
                                }
                            }
                            gpu_hash
                        }
                        Err(e) => {
                            if gpu_commit_strict_mode() {
                                panic!(
                                    "GPU commitment strict mode enabled, but GPU matrix hash failed for layer {}: {}",
                                    layer_id, e
                                );
                            }
                            eprintln!(
                                "[BG]   Layer {} GPU matrix hash failed ({}), using CPU fallback",
                                layer_id, e
                            );
                            gpu_hasher = None;
                            starknet_crypto::poseidon_hash_many(&final_inputs)
                        }
                    }
                } else {
                    starknet_crypto::poseidon_hash_many(&final_inputs)
                }
            }
            #[cfg(not(feature = "cuda-runtime"))]
            {
                starknet_crypto::poseidon_hash_many(&final_inputs)
            }
        };

        let finished = done_count.fetch_add(1, Ordering::Relaxed) + 1;
        if finished % progress_every == 0 || finished == total {
            let elapsed = t_commit.elapsed().as_secs_f64();
            let eta = if finished > 0 { elapsed * (total as f64 / finished as f64 - 1.0) } else { 0.0 };
            let pct = if total > 0 {
                100.0 * finished as f64 / total as f64
            } else {
                100.0
            };
            eprintln!(
                "[BG]   Weight commitment: {}/{} ({:.1}%, {:.1}s elapsed, ~{:.0}s remaining)",
                finished, total, pct, elapsed, eta,
            );
        }
        matrix_hash
    }).collect();
    ticker_stop.store(true, Ordering::Relaxed);
    let _ = ticker.join();

    let commitment = if per_matrix_hashes.is_empty() {
        FieldElement::ZERO
    } else {
        #[cfg(feature = "cuda-runtime")]
        {
            if let Some(hasher) = gpu_hasher.as_ref() {
                match hasher.hash_segments(std::slice::from_ref(&per_matrix_hashes), chunk_size) {
                    Ok(hashes) => {
                        let gpu_hash = hashes[0];
                        if gpu_commit_hardening_enabled() {
                            let cpu_hash = starknet_crypto::poseidon_hash_many(&per_matrix_hashes);
                            if gpu_hash != cpu_hash {
                                panic!(
                                    "GPU weight commitment hardening failed: root hash mismatch"
                                );
                            }
                        }
                        gpu_hash
                    }
                    Err(e) => {
                        if gpu_commit_strict_mode() {
                            panic!(
                                "GPU commitment strict mode enabled, but GPU root hash failed: {}",
                                e
                            );
                        }
                        eprintln!("[BG]   GPU root hash failed ({}), using CPU fallback", e);
                        starknet_crypto::poseidon_hash_many(&per_matrix_hashes)
                    }
                }
            } else {
                starknet_crypto::poseidon_hash_many(&per_matrix_hashes)
            }
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            starknet_crypto::poseidon_hash_many(&per_matrix_hashes)
        }
    };
    eprintln!(
        "[BG] Weight commitment computed in {:.2}s",
        t_commit.elapsed().as_secs_f64()
    );

    // Cache with fingerprint for validated reuse
    if let Some(d) = model_dir {
        if let Some(fp) = compute_fingerprint(d) {
            let cache_path = d.join(format!(".weight_commitment_{fp}.hex"));
            let hex = format!("0x{:064x}", commitment);
            if let Err(e) = std::fs::write(&cache_path, &hex) {
                eprintln!(
                    "[BG]   Warning: could not cache to {}: {e}",
                    cache_path.display()
                );
            } else {
                eprintln!(
                    "[BG]   Cached to {} (auto-invalidates if weights change)",
                    cache_path.display()
                );
            }
        }
    }
    commitment
}
