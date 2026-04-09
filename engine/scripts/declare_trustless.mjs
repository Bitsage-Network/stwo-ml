#!/usr/bin/env node
// Declare the fully trustless recursive STARK verifier class on Starknet Sepolia.
// Works with starknet.js v7.x (positional Account constructor).
//
// Usage:
//   STARKNET_PRIVATE_KEY=0x... node scripts/declare_trustless.mjs
//
// Optional:
//   STARKNET_RPC          — RPC endpoint (tries multiple if not set)
//   STARKNET_ACCOUNT      — deployer account address

import { Account, RpcProvider, json } from "starknet";
import { readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const CAIRO_DIR = resolve(ROOT, "../elo-cairo-verifier");

const SIERRA_PATH = resolve(CAIRO_DIR, "target/dev/elo_cairo_verifier_RecursiveVerifierContract.contract_class.json");
const CASM_PATH = resolve(CAIRO_DIR, "target/dev/elo_cairo_verifier_RecursiveVerifierContract.compiled_contract_class.json");

const DEFAULT_ACCOUNT = "0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f";
const PHASE1_CONTRACT = "0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7";

const RPCS = [
  process.env.STARKNET_RPC,
  "https://free-rpc.nethermind.io/sepolia-juno/v0_8",
  "https://starknet-sepolia.public.blastapi.io/rpc/v0_7",
  "https://api.cartridge.gg/x/starknet/sepolia",
  "http://localhost:6060",
].filter(Boolean);

const ADDR = process.env.STARKNET_ACCOUNT || DEFAULT_ACCOUNT;
const KEY = process.env.STARKNET_PRIVATE_KEY;

if (!KEY) {
  console.error("ERROR: STARKNET_PRIVATE_KEY is not set");
  process.exit(1);
}

console.log("Loading contract artifacts...");
const sierraRaw = readFileSync(SIERRA_PATH, "utf-8");
const casmRaw = readFileSync(CASM_PATH, "utf-8");
const sierra = json.parse(sierraRaw);
const casm = json.parse(casmRaw);
console.log(`  Sierra: ${(sierraRaw.length / 1024).toFixed(0)} KB`);
console.log(`  CASM:   ${(casmRaw.length / 1024).toFixed(0)} KB`);

async function tryDeclare(rpcUrl) {
  console.log(`\nTrying RPC: ${rpcUrl}`);
  const provider = new RpcProvider({ nodeUrl: rpcUrl });

  // Check RPC is reachable
  try {
    const block = await provider.getBlockNumber();
    console.log(`  Block: ${block}`);
  } catch (e) {
    console.log(`  RPC unreachable: ${(e.message || "").slice(0, 100)}`);
    return null;
  }

  const account = new Account(provider, ADDR, KEY);

  try {
    const result = await account.declareIfNot({
      contract: sierra,
      casm: casm,
    });

    if (result.transaction_hash) {
      console.log(`  TX: ${result.transaction_hash}`);
      console.log(`  Class hash: ${result.class_hash}`);
      console.log("  Waiting for confirmation...");
      const receipt = await provider.waitForTransaction(result.transaction_hash);
      console.log(`  Status: ${receipt.execution_status || receipt.status}`);
      return result;
    } else {
      console.log(`  Already declared: ${result.class_hash}`);
      return result;
    }
  } catch (e) {
    const msg = String(e.message || e);
    if (msg.includes("already declared") || msg.includes("is already declared")) {
      const classHash = msg.match(/0x[0-9a-fA-F]{60,66}/)?.[0];
      console.log(`  Already declared: ${classHash || "(hash not extracted)"}`);
      return { class_hash: classHash, already_declared: true };
    }
    console.log(`  Failed: ${msg.slice(0, 300)}`);
    return null;
  }
}

async function main() {
  let result = null;

  for (const rpc of RPCS) {
    result = await tryDeclare(rpc);
    if (result) break;
  }

  if (!result) {
    console.error("\nAll RPCs failed. Options:");
    console.error("  1. Start Juno with snap sync: docker run -d --name juno-sepolia ...");
    console.error("  2. Set STARKNET_RPC to a provider with high payload limits");
    process.exit(1);
  }

  console.log("\n================================================================");
  console.log("  TRUSTLESS RECURSIVE VERIFIER DECLARED");
  console.log("================================================================");
  console.log(`  Class hash:      ${result.class_hash}`);
  console.log(`  Phase 1 contract: ${PHASE1_CONTRACT}`);
  console.log("");
  console.log("  Next: upgrade Phase 1 contract to trustless class");
  console.log("================================================================");
}

main().catch((e) => {
  console.error("Fatal:", (e.message || "").slice(0, 500));
  process.exit(1);
});
