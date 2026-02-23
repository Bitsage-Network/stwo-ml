#!/usr/bin/env node
// On-chain audit submission via Avnu paymaster or direct execution.
//
// Modes:
//   1. Gasless (default) — user pays gas in STRK via Avnu paymaster. No API key needed.
//   2. Sponsored — Avnu pays gas. Requires AVNU_API_KEY with credits (free on Sepolia).
//   3. Direct — standard starknet.js execute. Account pays gas normally.
//
// Usage:
//   node paymaster_submit.mjs \
//     --contract 0x03f937cb... \
//     --function submit_audit \
//     --calldata-file /path/to/calldata.txt \
//     --account-address 0x01b17c... \
//     --private-key 0x0abc... \
//     --network sepolia \
//     --mode gasless
//
// Env vars:
//   STARKNET_ACCOUNT      — account address (overridden by --account-address)
//   STARKNET_PRIVATE_KEY  — account private key (overridden by --private-key)
//   AVNU_API_KEY          — Avnu API key (only needed for --mode sponsored)

import { Account, RpcProvider, PaymasterRpc } from "starknet";
import { readFileSync } from "fs";

// ─── Config ──────────────────────────────────────────────────────────
const PAYMASTER_URLS = {
  mainnet: "https://starknet.paymaster.avnu.fi",
  sepolia: "https://sepolia.paymaster.avnu.fi",
};

const RPC_URLS = {
  mainnet: "https://starknet-mainnet.public.blastapi.io/rpc/v0_7",
  sepolia: "https://starknet-sepolia.public.blastapi.io/rpc/v0_7",
};

// STRK token address (same on mainnet and sepolia)
const STRK_TOKEN = "0x04718f5a0fc34cc1af16a1cdee98ffb20c31f5cd61d6ab07201858f4287c938d";

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
    mode: "gasless", // gasless | sponsored | direct
    apiKey: process.env.AVNU_API_KEY || "",
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
      case "--mode":
        opts.mode = args[++i];
        break;
      case "--api-key":
        opts.apiKey = args[++i];
        break;
    }
  }

  if (!opts.contract || !opts.calldataFile || !opts.accountAddress || !opts.privateKey) {
    console.error(
      "Missing required args: --contract, --calldata-file, --account-address, --private-key"
    );
    process.exit(1);
  }

  if (opts.mode === "sponsored" && !opts.apiKey) {
    console.error("Sponsored mode requires AVNU_API_KEY env var or --api-key flag");
    process.exit(1);
  }

  return opts;
}

// ─── Main ────────────────────────────────────────────────────────────
async function main() {
  const opts = parseArgs();
  const rpcUrl = RPC_URLS[opts.network] || RPC_URLS.sepolia;
  const paymasterUrl = PAYMASTER_URLS[opts.network] || PAYMASTER_URLS.sepolia;

  // Read calldata from file (space-separated felts)
  const raw = readFileSync(opts.calldataFile, "utf-8").trim();
  const calldata = raw.split(/\s+/);

  console.error(`Submit on-chain:`);
  console.error(`  Contract:  ${opts.contract}`);
  console.error(`  Function:  ${opts.function}`);
  console.error(`  Calldata:  ${calldata.length} felts`);
  console.error(`  Account:   ${opts.accountAddress}`);
  console.error(`  Network:   ${opts.network}`);
  console.error(`  Mode:      ${opts.mode}`);

  const provider = new RpcProvider({ nodeUrl: rpcUrl });
  const account = new Account(provider, opts.accountAddress, opts.privateKey);

  const calls = [
    {
      contractAddress: opts.contract,
      entrypoint: opts.function,
      calldata,
    },
  ];

  let txHash;

  if (opts.mode === "direct") {
    // ─── Direct execution (account pays gas in STRK) ──────────────
    console.error(`  Executing directly...`);
    const result = await account.execute(calls);
    txHash = result.transaction_hash;
    console.error(`  Waiting for confirmation...`);
    await provider.waitForTransaction(txHash);
  } else {
    // ─── Paymaster execution (gasless or sponsored) ───────────────
    const paymasterOpts = { nodeUrl: paymasterUrl };

    if (opts.mode === "sponsored" && opts.apiKey) {
      paymasterOpts.headers = { "x-paymaster-api-key": opts.apiKey };
      console.error(`  Paymaster: ${paymasterUrl} (sponsored)`);
    } else {
      console.error(`  Paymaster: ${paymasterUrl} (gasless, pay in STRK)`);
    }

    const paymaster = new PaymasterRpc(paymasterOpts);

    let feeMode;
    if (opts.mode === "sponsored") {
      feeMode = { mode: "sponsored" };
    } else {
      // Gasless: user pays gas in STRK via paymaster
      feeMode = { mode: "default", gasToken: STRK_TOKEN };
    }

    console.error(`  Building transaction...`);
    const result = await account.execute(calls, { paymaster, feeMode });
    txHash = result.transaction_hash;
    console.error(`  Waiting for confirmation...`);
    await provider.waitForTransaction(txHash);
  }

  // Output
  const explorerBase =
    opts.network === "mainnet"
      ? "https://starkscan.co"
      : "https://sepolia.starkscan.co";

  console.error(`  Transaction confirmed!`);
  console.error(`  TX hash:  ${txHash}`);
  console.error(`  Explorer: ${explorerBase}/tx/${txHash}`);

  // Print JSON to stdout for machine consumption
  console.log(
    JSON.stringify({
      transaction_hash: txHash,
      explorer_url: `${explorerBase}/tx/${txHash}`,
      network: opts.network,
      mode: opts.mode,
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
