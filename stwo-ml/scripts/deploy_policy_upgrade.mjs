#!/usr/bin/env node
// Deploy the PolicyConfig-enabled Cairo verifier contract upgrade.
//
// This script:
//   1. Declares the new contract class (with policy support)
//   2. Proposes the upgrade (5-min timelock)
//   3. Waits for the timelock to expire
//   4. Executes the upgrade
//   5. Registers the "standard" policy for the active model
//
// Prerequisites:
//   - scarb build (in elo-cairo-verifier/)
//   - STARKNET_ACCOUNT, STARKNET_PRIVATE_KEY, STARKNET_RPC env vars
//
// Usage:
//   node scripts/deploy_policy_upgrade.mjs [--model-id 0x...] [--skip-declare] [--execute-only]
//
// Policy commitment hashes (deterministic, computed by PolicyConfig::policy_commitment()):
//   strict:   0x0370c9348ed6edddf310baf5d8104d57c07f36962deea9738dd00519d9948449
//   standard: 0x05baf1be3d54bcd383072f79923316ac7124670a117bd5c809b67b651209424b
//   relaxed:  0x02fba808267ad15ef03f2db8ac9a09a87194ea32edb5aa41333976ac4425d06c

import { Account, RpcProvider, json, CallData } from "starknet";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// ── Config ──────────────────────────────────────────────────────────────────
const RPC_URL = process.env.STARKNET_RPC;
if (!RPC_URL) { console.error("FATAL: STARKNET_RPC env var required"); process.exit(1); }
const ACCOUNT_ADDRESS = process.env.STARKNET_ACCOUNT;
const PRIVATE_KEY = process.env.STARKNET_PRIVATE_KEY;
if (!ACCOUNT_ADDRESS || !PRIVATE_KEY) {
  console.error("FATAL: STARKNET_ACCOUNT and STARKNET_PRIVATE_KEY required");
  process.exit(1);
}
const CONTRACT =
  process.env.CONTRACT_ADDRESS ||
  "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005";

// Policy commitment hashes
const POLICY_COMMITMENTS = {
  strict:   "0x0370c9348ed6edddf310baf5d8104d57c07f36962deea9738dd00519d9948449",
  standard: "0x05baf1be3d54bcd383072f79923316ac7124670a117bd5c809b67b651209424b",
  relaxed:  "0x02fba808267ad15ef03f2db8ac9a09a87194ea32edb5aa41333976ac4425d06c",
};

// ── Parse args ──────────────────────────────────────────────────────────────
const args = process.argv.slice(2);
const skipDeclare = args.includes("--skip-declare");
const executeOnly = args.includes("--execute-only");
const modelIdArg = args.find((_, i, a) => a[i - 1] === "--model-id");
const policyArg = args.find((_, i, a) => a[i - 1] === "--policy") || "standard";
const classHashArg = args.find((_, i, a) => a[i - 1] === "--class-hash");

const provider = new RpcProvider({ nodeUrl: RPC_URL });
const account = new Account(provider, ACCOUNT_ADDRESS, PRIVATE_KEY);

// ── Contract artifact path ──────────────────────────────────────────────────
const ARTIFACT_PATH = join(
  __dirname,
  "../../elo-cairo-verifier/target/dev/elo_cairo_verifier_SumcheckVerifierContract.contract_class.json"
);

async function main() {
  console.log("=== PolicyConfig Contract Upgrade ===");
  console.log(`  Contract: ${CONTRACT}`);
  console.log(`  Account:  ${ACCOUNT_ADDRESS}`);
  console.log(`  Policy:   ${policyArg} (${POLICY_COMMITMENTS[policyArg] || "custom"})`);
  console.log();

  let newClassHash = classHashArg;

  // ── Step 1: Declare ──────────────────────────────────────────────────────
  if (!skipDeclare && !executeOnly) {
    console.log("Step 1: Declaring new contract class...");
    try {
      const artifact = json.parse(readFileSync(ARTIFACT_PATH, "utf-8"));
      // CASM artifact
      const casmPath = ARTIFACT_PATH.replace(".contract_class.json", ".compiled_contract_class.json");
      const casm = json.parse(readFileSync(casmPath, "utf-8"));

      const declareResponse = await account.declare({ contract: artifact, casm });
      console.log(`  TX hash: ${declareResponse.transaction_hash}`);
      await provider.waitForTransaction(declareResponse.transaction_hash);
      newClassHash = declareResponse.class_hash;
      console.log(`  Class hash: ${newClassHash}`);
    } catch (e) {
      if (e.message?.includes("already declared") || e.message?.includes("StarknetErrorCode.CLASS_ALREADY_DECLARED")) {
        console.log("  Class already declared, continuing...");
        // Try to extract class hash from error or use provided one
        if (!newClassHash) {
          console.error("  ERROR: Class already declared but no --class-hash provided. Use --class-hash 0x...");
          process.exit(1);
        }
      } else {
        throw e;
      }
    }
    console.log();
  }

  if (!newClassHash) {
    console.error("ERROR: No class hash available. Run without --skip-declare or pass --class-hash 0x...");
    process.exit(1);
  }

  // ── Step 2: Propose upgrade ──────────────────────────────────────────────
  if (!executeOnly) {
    console.log("Step 2: Proposing upgrade (5-min timelock)...");
    const proposeTx = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "propose_upgrade",
      calldata: CallData.compile({ new_class_hash: newClassHash }),
    });
    console.log(`  TX hash: ${proposeTx.transaction_hash}`);
    await provider.waitForTransaction(proposeTx.transaction_hash);
    console.log("  Upgrade proposed. Waiting 5 minutes for timelock...");
    console.log();

    // Wait for timelock (5 min + 30s buffer)
    const TIMELOCK_SECS = 5 * 60 + 30;
    for (let i = TIMELOCK_SECS; i > 0; i -= 10) {
      process.stdout.write(`  Timelock: ${Math.ceil(i / 60)}m ${i % 60}s remaining...\r`);
      await new Promise((r) => setTimeout(r, 10000));
    }
    console.log("\n  Timelock expired.");
    console.log();
  }

  // ── Step 3: Execute upgrade ──────────────────────────────────────────────
  console.log("Step 3: Executing upgrade...");
  try {
    const executeTx = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "execute_upgrade",
      calldata: [],
    });
    console.log(`  TX hash: ${executeTx.transaction_hash}`);
    await provider.waitForTransaction(executeTx.transaction_hash);
    console.log("  Upgrade executed successfully.");
  } catch (e) {
    if (e.message?.includes("No upgrade pending")) {
      console.log("  No upgrade pending (already executed?)");
    } else {
      throw e;
    }
  }
  console.log();

  // ── Step 4: Register policy for model ────────────────────────────────────
  if (modelIdArg) {
    const policyHash = POLICY_COMMITMENTS[policyArg] || policyArg;
    console.log(`Step 4: Registering policy for model ${modelIdArg}...`);
    console.log(`  Policy: ${policyArg}`);
    console.log(`  Hash:   ${policyHash}`);

    const registerTx = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "register_model_policy",
      calldata: CallData.compile({
        model_id: modelIdArg,
        policy_hash: policyHash,
      }),
    });
    console.log(`  TX hash: ${registerTx.transaction_hash}`);
    await provider.waitForTransaction(registerTx.transaction_hash);
    console.log("  Policy registered.");
  } else {
    console.log("Step 4: Skipped (no --model-id provided).");
    console.log("  To register a policy later:");
    console.log(`  node scripts/deploy_policy_upgrade.mjs --skip-declare --execute-only --model-id 0x... --policy standard`);
  }

  console.log();
  console.log("=== Done ===");
  console.log();
  console.log("Next steps:");
  console.log("  1. Prove with: prove-model --policy standard --format ml_gkr ...");
  console.log("  2. Submit with: node scripts/pipeline/register_and_submit.mjs proof.json");
  console.log("  3. The streaming init TX will include the policy_hash automatically.");
  console.log("  4. The on-chain verifier will validate policy_hash matches the registered policy.");
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
