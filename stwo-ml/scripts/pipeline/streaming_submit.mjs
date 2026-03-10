#!/usr/bin/env node
// Streaming multi-TX submission with exponential backoff retry.
//
// Orchestrates the multi-step GKR streaming verification flow:
//   1. stream_init           — opens session, returns session_id
//   2. stream_output_mle     — M chunks of output MLE data (MUST be before layers)
//   3. stream_layers         — N batches of layer proofs
//   4. stream_weight_binding — packed aggregated binding verification
//   5. stream_finalize_input_mle — input MLE chunks
//   6. stream_finalize       — final check + proof recording
//
// Usage:
//   node streaming_submit.mjs \
//     --contract 0x... \
//     --calldata-dir /path/to/streaming_calldata/ \
//     --account-address 0x... \
//     --private-key 0x... \
//     --network sepolia \
//     --mode gasless \
//     --max-retries 3
//
// Calldata dir structure:
//   stream_init.txt
//   stream_layers_0.txt
//   stream_layers_1.txt
//   ...
//   stream_output_mle_0.txt (optional)
//   stream_finalize_input_mle.txt
//   stream_finalize.txt
//
// Each file contains space-separated felts. The stream_init step returns
// a session_id which is injected into subsequent calldata files by
// replacing the placeholder "__SESSION_ID__".
//
// Env vars:
//   STARKNET_ACCOUNT      — account address
//   STARKNET_PRIVATE_KEY  — private key
//   AVNU_API_KEY          — Avnu API key (sponsored mode only)

import { Account, RpcProvider, Signer } from "starknet";
import { readFileSync, readdirSync } from "fs";

// ── Config ───────────────────────────────────────────────────────────

const PAYMASTER_URLS = {
  mainnet: "https://starknet.paymaster.avnu.fi",
  sepolia: "https://sepolia.paymaster.avnu.fi",
};

const RPC_URLS = {
  mainnet: process.env.STARKNET_RPC || "https://json-rpc.starknet-mainnet.public.lavanet.xyz",
  sepolia: process.env.STARKNET_RPC || "https://api.cartridge.gg/x/starknet/sepolia",
};

const STRK_TOKEN =
  "0x04718f5a0fc34cc1af16a1cdee98ffb20c31f5cd61d6ab07201858f4287c938d";

// ── Parse args ───────────────────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    contract: "",
    calldataDir: "",
    accountAddress: process.env.STARKNET_ACCOUNT || "",
    privateKey: process.env.STARKNET_PRIVATE_KEY || "",
    network: "sepolia",
    mode: "gasless",
    apiKey: process.env.AVNU_API_KEY || "",
    maxRetries: 3,
    baseDelayMs: 2000,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--contract":
        opts.contract = args[++i];
        break;
      case "--calldata-dir":
        opts.calldataDir = args[++i];
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
      case "--max-retries":
        opts.maxRetries = parseInt(args[++i], 10);
        break;
    }
  }

  if (
    !opts.contract ||
    !opts.calldataDir ||
    !opts.accountAddress ||
    !opts.privateKey
  ) {
    console.error(
      "Required: --contract, --calldata-dir, --account-address, --private-key"
    );
    process.exit(1);
  }

  return opts;
}

// ── Helpers ──────────────────────────────────────────────────────────

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readCalldata(filePath, sessionId) {
  let raw = readFileSync(filePath, "utf-8").trim();
  if (sessionId) {
    raw = raw.replace(/__SESSION_ID__/g, sessionId);
  }
  return raw.split(/\s+/);
}

function discoverSteps(dir) {
  const files = readdirSync(dir).filter((f) => f.endsWith(".txt")).sort();
  const steps = [];

  // Order: init, output_mle (sorted), layers (sorted), weight_binding, finalize_input_mle, finalize
  const init = files.find((f) => f.startsWith("stream_init"));
  if (init) steps.push({ name: "stream_init", file: init, entrypoint: "verify_gkr_stream_init" });

  // Output MLE must come BEFORE layers (channel state dependency)
  const outputMle = files
    .filter((f) => f.startsWith("stream_output_mle_"))
    .sort((a, b) => {
      const na = parseInt(a.match(/(\d+)/)?.[1] || "0", 10);
      const nb = parseInt(b.match(/(\d+)/)?.[1] || "0", 10);
      return na - nb;
    });
  for (const f of outputMle) {
    steps.push({
      name: f.replace(".txt", ""),
      file: f,
      entrypoint: "verify_gkr_stream_init_output_mle",
    });
  }

  const layers = files
    .filter((f) => f.startsWith("stream_layers_"))
    .sort((a, b) => {
      const na = parseInt(a.match(/(\d+)/)?.[1] || "0", 10);
      const nb = parseInt(b.match(/(\d+)/)?.[1] || "0", 10);
      return na - nb;
    });
  for (const f of layers) {
    steps.push({ name: f.replace(".txt", ""), file: f, entrypoint: "verify_gkr_stream_layers" });
  }

  // Weight binding (packed QM31, separate TX before input MLE)
  const weightBinding = files.find((f) => f.startsWith("stream_weight_binding"));
  if (weightBinding) {
    steps.push({
      name: "stream_weight_binding",
      file: weightBinding,
      entrypoint: "verify_gkr_stream_weight_binding",
    });
  }

  const finInputMleFiles = files
    .filter((f) => f.startsWith("stream_finalize_input_mle"))
    .sort((a, b) => {
      const na = parseInt(a.match(/(\d+)/)?.[1] || "0", 10);
      const nb = parseInt(b.match(/(\d+)/)?.[1] || "0", 10);
      return na - nb;
    });
  for (const f of finInputMleFiles) {
    steps.push({
      name: f.replace(".txt", ""),
      file: f,
      entrypoint: "verify_gkr_stream_finalize_input_mle",
    });
  }

  const fin = files.find(
    (f) => f === "stream_finalize.txt"
  );
  if (fin) {
    steps.push({
      name: "stream_finalize",
      file: fin,
      entrypoint: "verify_gkr_stream_finalize",
    });
  }

  return steps;
}

// ── Submit with retry ────────────────────────────────────────────────

async function submitWithRetry(account, provider, call, opts, stepName) {
  let lastError;

  for (let attempt = 1; attempt <= opts.maxRetries; attempt++) {
    try {
      let txHash;

      if (opts.mode === "direct") {
        const result = await account.execute([call]);
        txHash = result.transaction_hash;
      } else {
        const feeMode =
          opts.mode === "sponsored"
            ? { mode: "sponsored" }
            : { mode: "default", gasToken: STRK_TOKEN };
        const paymasterDetails = { feeMode };
        const feeEstimate = await account.estimatePaymasterTransactionFee(
          [call],
          paymasterDetails
        );
        const result = await account.executePaymasterTransaction(
          [call],
          paymasterDetails,
          feeEstimate.suggested_max_fee_in_gas_token
        );
        txHash = result.transaction_hash;
      }

      console.error(`  [${stepName}] TX submitted: ${txHash}`);
      console.error(`  [${stepName}] Waiting for confirmation...`);
      await provider.waitForTransaction(txHash);
      console.error(`  [${stepName}] Confirmed.`);
      return txHash;
    } catch (err) {
      lastError = err;
      const delay = opts.baseDelayMs * Math.pow(2, attempt - 1);
      console.error(
        `  [${stepName}] Attempt ${attempt}/${opts.maxRetries} failed: ${
          err.message || err
        }`
      );
      if (attempt < opts.maxRetries) {
        console.error(`  [${stepName}] Retrying in ${delay}ms...`);
        await sleep(delay);
      }
    }
  }

  throw new Error(
    `[${stepName}] Failed after ${opts.maxRetries} attempts: ${
      lastError?.message || lastError
    }`
  );
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  const opts = parseArgs();
  const rpcUrl = RPC_URLS[opts.network] || RPC_URLS.sepolia;
  const paymasterUrl = PAYMASTER_URLS[opts.network] || PAYMASTER_URLS.sepolia;

  const provider = new RpcProvider({ nodeUrl: rpcUrl });

  const signer = new Signer(opts.privateKey);
  const account = new Account({ provider, address: opts.accountAddress, signer });

  // Discover streaming steps from calldata directory
  const steps = discoverSteps(opts.calldataDir);
  if (steps.length === 0) {
    console.error("No streaming calldata files found in " + opts.calldataDir);
    process.exit(1);
  }

  console.error(`Streaming submission: ${steps.length} steps`);
  console.error(`  Contract: ${opts.contract}`);
  console.error(`  Network:  ${opts.network}`);
  console.error(`  Mode:     ${opts.mode}`);
  console.error(`  Retries:  ${opts.maxRetries}`);
  console.error("");

  const results = [];
  let sessionId = null;

  for (const step of steps) {
    const filePath = `${opts.calldataDir}/${step.file}`;
    const calldata = readCalldata(filePath, sessionId);

    console.error(
      `Step: ${step.name} (${step.entrypoint}, ${calldata.length} felts)`
    );

    const call = {
      contractAddress: opts.contract,
      entrypoint: step.entrypoint,
      calldata,
    };

    const txHash = await submitWithRetry(
      account,
      provider,
      call,
      opts,
      step.name
    );

    // Extract session_id from init step event.
    // session_id is #[key] in GkrSessionOpened event, so it's in keys[1]
    // (keys[0] is the event selector hash).
    if (step.name === "stream_init" && !sessionId) {
      try {
        const receipt = await provider.getTransactionReceipt(txHash);
        if (receipt.events && receipt.events.length > 0) {
          const evt = receipt.events[0];
          sessionId = evt.keys?.[1] || evt.data?.[0];
          if (sessionId) {
            console.error(`  Session ID: ${sessionId}`);
          }
        }
        if (!sessionId) {
          console.error("  FATAL: Could not extract session_id from init receipt!");
          process.exit(1);
        }
      } catch (e) {
        console.error(
          `  FATAL: could not extract session_id from receipt: ${e.message}`
        );
        process.exit(1);
      }
    }

    results.push({
      step: step.name,
      entrypoint: step.entrypoint,
      tx_hash: txHash,
      calldata_felts: calldata.length,
    });

    console.error("");
  }

  // Output summary
  const explorerBase =
    opts.network === "mainnet"
      ? "https://starkscan.co"
      : "https://sepolia.starkscan.co";

  console.error("All steps completed successfully!");
  console.error("");

  const output = {
    success: true,
    network: opts.network,
    contract: opts.contract,
    session_id: sessionId,
    steps: results.map((r) => ({
      ...r,
      explorer_url: `${explorerBase}/tx/${r.tx_hash}`,
    })),
    total_steps: results.length,
    total_felts: results.reduce((sum, r) => sum + r.calldata_felts, 0),
  };

  console.log(JSON.stringify(output, null, 2));
}

main().catch((err) => {
  console.error(`Fatal: ${err.message || err}`);
  process.exit(1);
});
