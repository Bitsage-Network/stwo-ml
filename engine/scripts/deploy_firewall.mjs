#!/usr/bin/env node
// Deploy the AgentFirewallZK + ContractRegistry contracts to Starknet Sepolia.
//
// Steps:
//   1. Declare ContractRegistry class
//   2. Deploy ContractRegistry with owner = deployer
//   3. Declare AgentFirewallZK class
//   4. Deploy AgentFirewallZK with:
//      - owner = deployer
//      - verifier = existing ObelyskVerifier
//      - classifier_model_id = 0x42 (test model)
//      - classifier_weight_root_hash = 0x0 (test, no weight binding yet)
//   5. Configure: set_contract_registry on firewall
//   6. Print all addresses
//
// Usage:
//   export STARKNET_RPC=https://starknet-sepolia.public.blastapi.io/rpc/v0_7
//   export STARKNET_ACCOUNT=0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f
//   export STARKNET_PRIVATE_KEY=<your key>
//   node scripts/deploy_firewall.mjs

import { Account, RpcProvider, json, CallData, stark } from "starknet";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Config ──────────────────────────────────────────────────────────────────
const RPC_URL = process.env.STARKNET_RPC || "https://starknet-sepolia.public.blastapi.io/rpc/v0_7";
const ACCOUNT_ADDRESS = process.env.STARKNET_ACCOUNT || "0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f";
const PRIVATE_KEY = process.env.STARKNET_PRIVATE_KEY;

if (!PRIVATE_KEY) {
  console.error("FATAL: STARKNET_PRIVATE_KEY env var required");
  process.exit(1);
}

// Existing ObelyskVerifier contract
const VERIFIER_CONTRACT = process.env.VERIFIER_CONTRACT ||
  "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005";

// Classifier model ID (test)
const CLASSIFIER_MODEL_ID = "0x42";
// Weight root hash (0 = no weight binding for test deployment)
const CLASSIFIER_WEIGHT_ROOT_HASH = "0x0";

// ── Paths ───────────────────────────────────────────────────────────────────
const CAIRO_DIR = join(__dirname, "../../elo-cairo-verifier/target/dev");

const REGISTRY_CLASS = join(CAIRO_DIR, "elo_cairo_verifier_ContractRegistry.contract_class.json");
const REGISTRY_CASM = join(CAIRO_DIR, "elo_cairo_verifier_ContractRegistry.compiled_contract_class.json");
const FIREWALL_CLASS = join(CAIRO_DIR, "elo_cairo_verifier_AgentFirewallZK.contract_class.json");
const FIREWALL_CASM = join(CAIRO_DIR, "elo_cairo_verifier_AgentFirewallZK.compiled_contract_class.json");

// ── Main ────────────────────────────────────────────────────────────────────
const provider = new RpcProvider({ nodeUrl: RPC_URL });
const account = new Account(provider, ACCOUNT_ADDRESS, PRIVATE_KEY);

async function declareIfNeeded(name, classPath, casmPath) {
  console.log(`\nDeclaring ${name}...`);
  const contract = json.parse(readFileSync(classPath, "utf-8"));
  const casm = json.parse(readFileSync(casmPath, "utf-8"));

  try {
    const result = await account.declare({ contract, casm });
    console.log(`  TX: ${result.transaction_hash}`);
    await provider.waitForTransaction(result.transaction_hash);
    console.log(`  Class hash: ${result.class_hash}`);
    return result.class_hash;
  } catch (e) {
    const msg = e.message || "";
    if (msg.includes("already declared") || msg.includes("CLASS_ALREADY_DECLARED")) {
      // Extract class hash from the Sierra contract
      const classHash = stark.computeSierraContractClassHash(contract);
      console.log(`  Already declared: ${classHash}`);
      return classHash;
    }
    throw e;
  }
}

async function deploy(name, classHash, constructorCalldata) {
  console.log(`\nDeploying ${name}...`);
  const result = await account.deployContract({
    classHash,
    constructorCalldata,
  });
  console.log(`  TX: ${result.transaction_hash}`);
  await provider.waitForTransaction(result.transaction_hash);
  console.log(`  Address: ${result.contract_address}`);
  return result.contract_address;
}

async function main() {
  console.log("╔══════════════════════════════════════════════╗");
  console.log("║  AgentFirewallZK Deployment — Starknet Sepolia  ║");
  console.log("╚══════════════════════════════════════════════╝");
  console.log(`  Deployer: ${ACCOUNT_ADDRESS}`);
  console.log(`  Verifier: ${VERIFIER_CONTRACT}`);
  console.log(`  RPC:      ${RPC_URL}`);

  // 1. Declare + Deploy ContractRegistry
  const registryClassHash = await declareIfNeeded("ContractRegistry", REGISTRY_CLASS, REGISTRY_CASM);
  const registryAddress = await deploy("ContractRegistry", registryClassHash,
    CallData.compile({ owner: ACCOUNT_ADDRESS })
  );

  // 2. Declare + Deploy AgentFirewallZK
  const firewallClassHash = await declareIfNeeded("AgentFirewallZK", FIREWALL_CLASS, FIREWALL_CASM);
  const firewallAddress = await deploy("AgentFirewallZK", firewallClassHash,
    CallData.compile({
      owner: ACCOUNT_ADDRESS,
      verifier_address: VERIFIER_CONTRACT,
      classifier_model_id: CLASSIFIER_MODEL_ID,
      classifier_weight_root_hash: CLASSIFIER_WEIGHT_ROOT_HASH,
    })
  );

  // 3. Configure firewall: set contract registry
  console.log("\nConfiguring firewall...");
  const configTx = await account.execute({
    contractAddress: firewallAddress,
    entrypoint: "set_contract_registry",
    calldata: CallData.compile({ registry_address: registryAddress }),
  });
  console.log(`  set_contract_registry TX: ${configTx.transaction_hash}`);
  await provider.waitForTransaction(configTx.transaction_hash);

  // 4. Print summary
  console.log("\n╔══════════════════════════════════════════════╗");
  console.log("║  Deployment Complete                            ║");
  console.log("╚══════════════════════════════════════════════╝");
  console.log(`  ContractRegistry:  ${registryAddress}`);
  console.log(`    Class hash:      ${registryClassHash}`);
  console.log(`  AgentFirewallZK:   ${firewallAddress}`);
  console.log(`    Class hash:      ${firewallClassHash}`);
  console.log(`    Verifier:        ${VERIFIER_CONTRACT}`);
  console.log(`    Model ID:        ${CLASSIFIER_MODEL_ID}`);
  console.log(`    Registry:        ${registryAddress}`);
  console.log();
  console.log("Next steps:");
  console.log("  1. Register an agent:");
  console.log(`     sncast invoke --contract-address ${firewallAddress} --function register_agent --calldata 0x1`);
  console.log("  2. Classify a transaction:");
  console.log("     curl -X POST http://localhost:8080/api/v1/classify -d '{\"target\":\"0x1234\",\"value\":\"1000000\"}'");
  console.log("  3. Submit + resolve with proof");
}

main().catch((e) => {
  console.error("\nFATAL:", e.message || e);
  process.exit(1);
});
