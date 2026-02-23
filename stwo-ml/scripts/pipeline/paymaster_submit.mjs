#!/usr/bin/env node
// Avnu paymaster submission — gasless on-chain audit submission.
// No STRK needed on the submitter account. Avnu sponsors gas.
//
// Usage:
//   node paymaster_submit.mjs \
//     --contract 0x03f937cb... \
//     --function submit_audit \
//     --calldata-file /path/to/calldata.txt \
//     --account-address 0x01b17c... \
//     --private-key 0x0abc... \
//     --network sepolia
//
// Env vars (optional overrides):
//   AVNU_API_KEY — Avnu paymaster API key
//   STARKNET_ACCOUNT — account address
//   STARKNET_PRIVATE_KEY — account private key

import { Account, RpcProvider, stark, ec } from "starknet";
import { readFileSync } from "fs";

// ─── Config ──────────────────────────────────────────────────────────
const AVNU_URLS = {
  mainnet: "https://starknet.api.avnu.fi",
  sepolia: "https://sepolia.api.avnu.fi",
};

const RPC_URLS = {
  mainnet: "https://starknet-mainnet.public.blastapi.io/rpc/v0_7",
  sepolia: "https://starknet-sepolia.public.blastapi.io/rpc/v0_7",
};

const DEFAULT_API_KEY = process.env.AVNU_API_KEY || "";

// ─── Parse args ──────────────────────────────────────────────────────
function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    contract: "",
    function: "submit_audit",
    calldataFile: "",
    accountAddress: process.env.STARKNET_ACCOUNT || "",
    privateKey: process.env.STARKNET_PRIVATE_KEY || "",
    network: "sepolia",
    apiKey: DEFAULT_API_KEY,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--contract":
        opts.contract = args[++i];
        break;
      case "--function":
        opts.function = args[++i];
        break;
      case "--calldata-file":
        opts.calldataFile = args[++i];
        break;
      case "--account-address":
        opts.accountAddress = args[++i];
        break;
      case "--private-key":
        opts.privateKey = args[++i];
        break;
      case "--network":
        opts.network = args[++i];
        break;
      case "--api-key":
        opts.apiKey = args[++i];
        break;
    }
  }

  if (!opts.contract || !opts.calldataFile || !opts.accountAddress || !opts.privateKey) {
    console.error("Missing required args: --contract, --calldata-file, --account-address, --private-key");
    process.exit(1);
  }
  if (!opts.apiKey) {
    console.error("Missing AVNU_API_KEY env var or --api-key flag");
    process.exit(1);
  }

  return opts;
}

// ─── Main ────────────────────────────────────────────────────────────
async function main() {
  const opts = parseArgs();
  const avnuBase = AVNU_URLS[opts.network] || AVNU_URLS.sepolia;
  const rpcUrl = RPC_URLS[opts.network] || RPC_URLS.sepolia;

  // Read calldata from file (space-separated felts)
  const raw = readFileSync(opts.calldataFile, "utf-8").trim();
  const calldata = raw.split(/\s+/);

  console.error(`Paymaster submit:`);
  console.error(`  Contract:  ${opts.contract}`);
  console.error(`  Function:  ${opts.function}`);
  console.error(`  Calldata:  ${calldata.length} felts`);
  console.error(`  Account:   ${opts.accountAddress}`);
  console.error(`  Network:   ${opts.network}`);
  console.error(`  Avnu:      ${avnuBase}`);

  // Set up starknet.js account
  const provider = new RpcProvider({ nodeUrl: rpcUrl });
  const account = new Account(provider, opts.accountAddress, opts.privateKey);

  const calls = [
    {
      contractAddress: opts.contract,
      entrypoint: opts.function,
      calldata,
    },
  ];

  const headers = {
    "Content-Type": "application/json",
    "api-key": opts.apiKey,
  };

  // Step 1: Build typed data (sponsored — gasTokenAddress=null)
  console.error(`  Building typed data...`);
  const buildResp = await fetch(`${avnuBase}/paymaster/v1/build-typed-data`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      userAddress: opts.accountAddress,
      calls,
      gasTokenAddress: null,
      maxGasTokenAmount: null,
    }),
  });

  if (!buildResp.ok) {
    const errText = await buildResp.text();
    console.error(`  Error building typed data: ${buildResp.status} ${errText}`);
    process.exit(1);
  }

  const typedData = await buildResp.json();
  console.error(`  Typed data received, signing...`);

  // Step 2: Sign the typed data
  const signature = await account.signMessage(typedData);
  const sigArray = stark.formatSignature(signature);

  // Step 3: Execute via Avnu paymaster
  console.error(`  Submitting to Avnu paymaster...`);
  const execResp = await fetch(`${avnuBase}/paymaster/v1/execute`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      userAddress: opts.accountAddress,
      typedData: JSON.stringify(typedData),
      signature: sigArray,
    }),
  });

  if (!execResp.ok) {
    const errText = await execResp.text();
    console.error(`  Error executing: ${execResp.status} ${errText}`);
    process.exit(1);
  }

  const result = await execResp.json();
  const txHash = result.transactionHash;

  // Output
  const explorerBase =
    opts.network === "mainnet"
      ? "https://starkscan.co"
      : "https://sepolia.starkscan.co";

  console.error(`  Transaction submitted!`);
  console.error(`  TX hash:  ${txHash}`);
  console.error(`  Explorer: ${explorerBase}/tx/${txHash}`);

  // Print JSON to stdout for machine consumption
  console.log(
    JSON.stringify({
      transaction_hash: txHash,
      explorer_url: `${explorerBase}/tx/${txHash}`,
      network: opts.network,
      contract: opts.contract,
      function: opts.function,
      calldata_count: calldata.length,
    })
  );
}

main().catch((err) => {
  console.error(`Fatal: ${err.message || err}`);
  process.exit(1);
});
