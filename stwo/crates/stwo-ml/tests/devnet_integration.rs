//! End-to-end integration test: Rust proof → calldata → Cairo verifier on devnet.
//!
//! Requires:
//! - `starknet-devnet` binary in PATH
//! - `sncast` binary in PATH (starknet-foundry)
//! - Cairo contracts built: `scarb build` in BitSage-Cairo-Smart-Contracts/
//!
//! Run with: `cargo test -p stwo-ml --test devnet_integration -- --ignored --nocapture`
//! (ignored by default to avoid needing devnet for CI)

use std::process::{Child, Command};
use std::sync::Arc;
use std::time::Duration;

use starknet::accounts::{Account, ExecutionEncoding, SingleOwnerAccount};
use starknet::core::types::{BlockId, BlockTag, Call, Felt, FunctionCall, StarknetError};
use starknet::core::utils::get_selector_from_name;
use starknet::macros::felt;
use starknet::providers::jsonrpc::HttpTransport;
use starknet::providers::{JsonRpcClient, Provider, ProviderError, Url};
use starknet::signers::{LocalWallet, SigningKey};

use stwo::core::fields::m31::M31;
use stwo_ml::components::matmul::M31Matrix;
use stwo_ml::starknet::prove_matmul_for_starknet;

// Devnet pre-funded account #0 (seed=42)
const DEVNET_ACCOUNT_ADDRESS: &str =
    "0x34ba56f92265f0868c57d3fe72ecab144fc96f97954bbbc4252cef8e8a979ba";
const DEVNET_PRIVATE_KEY: &str = "0xb137668388dbe9acdfa3bc734cc2c469";

const CONTRACT_ARTIFACT_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../BitSage-Cairo-Smart-Contracts"
);

const DEVNET_PORT: u16 = 5051;

/// Start starknet-devnet and return child process.
fn start_devnet() -> Child {
    let child = Command::new("starknet-devnet")
        .args(["--port", &DEVNET_PORT.to_string(), "--seed", "42"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("Failed to start starknet-devnet — is it installed?");

    std::thread::sleep(Duration::from_secs(3));
    child
}

fn rpc_provider() -> JsonRpcClient<HttpTransport> {
    let url = Url::parse(&format!("http://127.0.0.1:{}/rpc", DEVNET_PORT)).unwrap();
    JsonRpcClient::new(HttpTransport::new(url))
}

/// Convert starknet-ff FieldElement to starknet-rs Felt.
fn felt_from_fe252(v: starknet_ff::FieldElement) -> Felt {
    Felt::from_bytes_be(&v.to_bytes_be())
}

/// Create sncast accounts file for devnet.
fn create_sncast_accounts() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("stwo-ml-devnet-test");
    std::fs::create_dir_all(&dir).unwrap();
    let accounts_file = dir.join("accounts.json");

    let json = format!(
        r#"{{
  "alpha-sepolia": {{
    "devnet0": {{
      "private_key": "{DEVNET_PRIVATE_KEY}",
      "public_key": "0x5a5e37c60e77a0318643b111f88413a76af6233c891a0cfb2804106372006d4",
      "address": "{DEVNET_ACCOUNT_ADDRESS}",
      "salt": "0x0",
      "deployed": true,
      "class_hash": "0x5b4b537eaa2399e3aa99c4e2e0208ebd6c71bc1467938cd52c798c601e43564",
      "legacy": false,
      "type": "open_zeppelin"
    }}
  }}
}}"#
    );

    std::fs::write(&accounts_file, json).unwrap();
    accounts_file
}

/// Declare and deploy the SumcheckVerifierContract via sncast.
/// Returns the deployed contract address as a hex string.
fn declare_and_deploy_contract(accounts_file: &str) -> String {
    let rpc_url = format!("http://127.0.0.1:{}/rpc", DEVNET_PORT);

    // Declare from the Scarb workspace
    let declare_output = Command::new("sncast")
        .args([
            "--account",
            "devnet0",
            "--accounts-file",
            accounts_file,
            "declare",
            "--url",
            &rpc_url,
            "--contract-name",
            "SumcheckVerifierContract",
        ])
        .current_dir(CONTRACT_ARTIFACT_DIR)
        .output()
        .expect("sncast declare failed to execute");

    let declare_stdout = String::from_utf8_lossy(&declare_output.stdout);
    let declare_stderr = String::from_utf8_lossy(&declare_output.stderr);
    println!("declare stdout: {declare_stdout}");
    if !declare_stderr.is_empty() {
        println!("declare stderr: {declare_stderr}");
    }

    assert!(
        declare_output.status.success(),
        "sncast declare failed: {declare_stderr}"
    );

    // Extract class hash
    let class_hash = declare_stdout
        .lines()
        .find(|l| l.contains("Class Hash:"))
        .map(|l| l.split_whitespace().last().unwrap().trim().to_string())
        .expect("Could not find class hash in sncast output");

    println!("Class hash: {class_hash}");

    // Deploy with owner = devnet account
    let deploy_output = Command::new("sncast")
        .args([
            "--account",
            "devnet0",
            "--accounts-file",
            accounts_file,
            "deploy",
            "--url",
            &rpc_url,
            "--class-hash",
            &class_hash,
            "--salt",
            "0x1234",
            "--arguments",
            DEVNET_ACCOUNT_ADDRESS,
        ])
        .output()
        .expect("sncast deploy failed to execute");

    let deploy_stdout = String::from_utf8_lossy(&deploy_output.stdout);
    let deploy_stderr = String::from_utf8_lossy(&deploy_output.stderr);
    println!("deploy stdout: {deploy_stdout}");
    if !deploy_stderr.is_empty() {
        println!("deploy stderr: {deploy_stderr}");
    }

    assert!(
        deploy_output.status.success(),
        "sncast deploy failed: {deploy_stderr}"
    );

    // Extract contract address
    deploy_stdout
        .lines()
        .find(|l| l.contains("Contract Address:"))
        .map(|l| l.split_whitespace().last().unwrap().trim().to_string())
        .expect("Could not find contract address in sncast output")
}

#[tokio::test]
#[ignore] // Run manually: cargo test -p stwo-ml --test devnet_integration -- --ignored --nocapture
async fn test_rust_proof_verified_on_cairo_devnet() {
    // ── 1. Start devnet ──────────────────────────────────────────────────
    let mut devnet = start_devnet();
    let _guard = DevnetGuard(&mut devnet);

    let rpc = Arc::new(rpc_provider());

    let chain_id = rpc.chain_id().await.expect("devnet not responding");
    println!("Connected to devnet, chain_id: {:#x}", chain_id);

    // ── 2. Declare + Deploy via sncast ───────────────────────────────────
    let accounts_file = create_sncast_accounts();
    let accounts_path = accounts_file.to_str().unwrap();

    let contract_addr_hex = declare_and_deploy_contract(accounts_path);
    let contract_address = Felt::from_hex(&contract_addr_hex).unwrap();
    println!("Contract deployed at: {:#x}", contract_address);

    // ── 3. Set up starknet-rs account for invoke calls ───────────────────
    let signer = LocalWallet::from(SigningKey::from_secret_scalar(
        Felt::from_hex(DEVNET_PRIVATE_KEY).unwrap(),
    ));
    let account = SingleOwnerAccount::new(
        rpc.clone(),
        signer,
        Felt::from_hex(DEVNET_ACCOUNT_ADDRESS).unwrap(),
        chain_id,
        ExecutionEncoding::New,
    );

    // ── 4. Generate a proof in Rust (4×4 matmul) ─────────────────────────
    let a = M31Matrix::from_data(4, 4, (1..=16).map(M31::from).collect()).unwrap();
    let b = M31Matrix::from_data(4, 4, (17..=32).map(M31::from).collect()).unwrap();
    let c = M31Matrix::multiply(&a, &b).unwrap();

    let proof = prove_matmul_for_starknet(&a, &b, &c).unwrap();
    let calldata = proof.to_calldata();

    println!(
        "Generated proof: {}x{}, {} rounds, calldata len={}",
        proof.m,
        proof.k,
        proof.num_rounds,
        calldata.len()
    );

    // ── 5. Register the model ────────────────────────────────────────────
    let model_id = felt!("0x4d4c5f544553545f4d4f44454c"); // "ML_TEST_MODEL"
    let weight_commitment = felt_from_fe252(proof.a_commitment);

    println!(
        "Registering model: id={:#x}, commitment={:#x}",
        model_id, weight_commitment
    );

    let register_tx = account
        .execute_v3(vec![Call {
            to: contract_address,
            selector: get_selector_from_name("register_model").unwrap(),
            calldata: vec![model_id, weight_commitment],
        }])
        .send()
        .await
        .expect("register_model failed");
    println!("register_model tx: {:#x}", register_tx.transaction_hash);
    tokio::time::sleep(Duration::from_secs(2)).await;

    // ── 6. Call verify_matmul via RPC call (read-only) ───────────────────
    let mut verify_calldata = vec![model_id];
    for felt252_val in &calldata {
        verify_calldata.push(felt_from_fe252(*felt252_val));
    }

    println!(
        "Calling verify_matmul with {} calldata felts...",
        verify_calldata.len()
    );

    let verify_result = rpc
        .call(
            FunctionCall {
                contract_address,
                entry_point_selector: get_selector_from_name("verify_matmul").unwrap(),
                calldata: verify_calldata.clone(),
            },
            BlockId::Tag(BlockTag::Latest),
        )
        .await;

    match verify_result {
        Ok(result) => {
            println!("verify_matmul returned: {:?}", result);
            assert_eq!(result.len(), 1, "Expected single bool return");
            assert_eq!(
                result[0],
                Felt::ONE,
                "Proof verification should return true"
            );
            println!("SUCCESS: On-chain verification passed!");
        }
        Err(e) => {
            eprintln!("verify_matmul call failed: {e}");
            eprintln!("This may indicate a calldata serialization mismatch.");
            eprintln!("Calldata length: {}", verify_calldata.len());

            if let ProviderError::StarknetError(StarknetError::ContractError(data)) = &e {
                eprintln!("Contract error: {:?}", data);
            }

            panic!("verify_matmul failed: {e}");
        }
    }
}

/// Guard that kills devnet on drop.
struct DevnetGuard<'a>(&'a mut Child);

impl Drop for DevnetGuard<'_> {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
