#!/usr/bin/env node
// submit_recursive.mjs — Submit a recursive STARK proof on-chain in a single TX.
//
// Usage:
//   node scripts/submit_recursive.mjs /tmp/proof.json
//
// Environment:
//   STARKNET_PRIVATE_KEY  — deployer private key (required)
//   STARKNET_RPC          — Starknet RPC URL (default: Alchemy Sepolia)
//   RECURSIVE_CONTRACT    — recursive verifier contract address
//   STARKNET_ACCOUNT      — deployer account address

import { Account, RpcProvider, CallData } from "starknet";
import { readFileSync } from "fs";

const DEFAULT_RPC = process.env.STARKNET_RPC || "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo";
const DEFAULT_CONTRACT = "0x707819dea6210ab58b358151419a604ffdb16809b568bf6f8933067c2a28715";
const DEFAULT_ACCOUNT = "0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f";

const RPC = process.env.STARKNET_RPC || DEFAULT_RPC;
const CONTRACT = process.env.RECURSIVE_CONTRACT || DEFAULT_CONTRACT;
const ADDR = process.env.STARKNET_ACCOUNT || DEFAULT_ACCOUNT;
const KEY = process.env.STARKNET_PRIVATE_KEY;

if (!KEY) {
  console.error("ERROR: STARKNET_PRIVATE_KEY is not set");
  process.exit(1);
}

const proofPath = process.argv[2];
if (!proofPath) {
  console.error("Usage: node submit_recursive.mjs <proof.json>");
  process.exit(1);
}

const provider = new RpcProvider({ nodeUrl: RPC });
const account = new Account({ provider, address: ADDR, signer: KEY });

const GAS = {
  l2_gas: { max_amount: 0x40000000n, max_price_per_unit: 0x2cb417800n },
  l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n },
  l1_data_gas: { max_amount: 0x10000n, max_price_per_unit: 0x7ed13c779n },
};

async function main() {
  const raw = JSON.parse(readFileSync(proofPath, "utf-8"));

  // Support both top-level format and nested recursive_proof
  const modelId = raw.model_id || raw.verify_calldata?.model_id || "0x1";
  const ioCommitment = raw.io_commitment || "0x1";
  const recursive = raw.recursive_proof;

  if (!recursive || !recursive.calldata || recursive.calldata.length === 0) {
    console.error("ERROR: No recursive_proof.calldata found in proof file");
    process.exit(1);
  }

  const calldata = recursive.calldata;
  console.log("Contract:      " + CONTRACT);
  console.log("Model ID:      " + modelId);
  console.log("IO Commitment: " + ioCommitment);
  console.log("Calldata:      " + calldata.length + " felts");

  // Step 1: Register model (skip if already registered)
  try {
    const r = await provider.callContract({
      contractAddress: CONTRACT,
      entrypoint: "get_recursive_verification_count",
      calldata: [modelId],
    });
    // If the call succeeds, model is registered (or at least the contract responds)
    console.log("Register:      skip (already registered, count=" + Number(BigInt(r[0])) + ")");
  } catch {
    // Model not registered — register it
    process.stdout.write("Register:      ");
    try {
      const regTx = await account.execute({
        contractAddress: CONTRACT,
        entrypoint: "register_model_recursive",
        calldata: CallData.compile({
          model_id: modelId,
          circuit_hash: "0x1",
          weight_super_root: "0x1",
        }),
      });
      await provider.waitForTransaction(regTx.transaction_hash);
      console.log("done (tx=" + regTx.transaction_hash.slice(0, 18) + "...)");
    } catch (e) {
      // "already registered" is fine
      if ((e.message || "").includes("already") || (e.message || "").includes("registered")) {
        console.log("skip (already registered)");
      } else {
        console.error("FAILED: " + (e.message || "").slice(0, 300));
        process.exit(1);
      }
    }
  }

  // Step 2: Submit verify_recursive
  process.stdout.write("Submitting:    ");
  const tx = await account.execute(
    {
      contractAddress: CONTRACT,
      entrypoint: "verify_recursive",
      calldata: CallData.compile({
        model_id: modelId,
        io_commitment: ioCommitment,
        stark_proof_data: calldata,
      }),
    },
    { resourceBounds: GAS }
  );
  console.log("tx=" + tx.transaction_hash);

  // Step 3: Wait for confirmation
  process.stdout.write("Confirming:    ");
  const receipt = await provider.waitForTransaction(tx.transaction_hash);
  if (receipt.execution_status === "REVERTED") {
    console.log("REVERTED");
    console.error("Revert reason: " + (receipt.revert_reason || "").slice(0, 400));
    // Output structured result even on revert so caller can parse
    console.log("RESULT_JSON:" + JSON.stringify({
      success: false,
      tx_hash: tx.transaction_hash,
      error: (receipt.revert_reason || "").slice(0, 400),
    }));
    process.exit(1);
  }
  console.log("confirmed");

  // Step 4: Query verification count
  let verificationCount = 0;
  try {
    const count = await provider.callContract({
      contractAddress: CONTRACT,
      entrypoint: "get_recursive_verification_count",
      calldata: [modelId],
    });
    verificationCount = Number(BigInt(count[0]));
  } catch {
    // Non-fatal — contract might not have this view function
  }

  const explorerUrl = "https://sepolia.starkscan.co/tx/" + tx.transaction_hash;

  console.log("");
  console.log("================================================================");
  console.log("  RECURSIVE STARK VERIFIED ON-CHAIN");
  console.log("================================================================");
  console.log("  TX:            " + tx.transaction_hash);
  console.log("  Felts:         " + calldata.length);
  console.log("  Verifications: " + verificationCount);
  console.log("  Explorer:      " + explorerUrl);
  console.log("================================================================");

  // Output structured JSON for machine parsing (CLI reads this line)
  console.log("RESULT_JSON:" + JSON.stringify({
    success: true,
    tx_hash: tx.transaction_hash,
    explorer_url: explorerUrl,
    verification_count: verificationCount,
    felts: calldata.length,
  }));
}

main().catch((e) => {
  console.error("Fatal: " + (e.message || "").slice(0, 500));
  // Output structured result even on error
  console.log("RESULT_JSON:" + JSON.stringify({
    success: false,
    error: (e.message || "").slice(0, 500),
  }));
  process.exit(1);
});
