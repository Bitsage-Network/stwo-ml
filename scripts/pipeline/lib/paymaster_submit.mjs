#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════════
// Obelysk Pipeline — Paymaster Submission Helper
// ═══════════════════════════════════════════════════════════════════════
//
// Uses starknet.js v8 native paymaster support (AVNU sponsored mode)
// to submit proofs gaslessly on Starknet.
//
// Zero-config flow (no env vars needed):
//   1. Auto-generate ephemeral Stark keypair
//   2. Compute counterfactual account address
//   3. Deploy account + submit proof in ONE paymaster-sponsored TX
//      via PaymasterDetails.deploymentData
//
// Commands:
//   verify  — Submit proof via AVNU paymaster (auto-deploys account if needed)
//   setup   — Deploy agent account via factory (ERC-8004 identity, needs deployer key)
//   status  — Check account and verification status
//
// Usage:
//   node paymaster_submit.mjs verify --proof proof.json --contract 0x... --model-id 0x1
//   node paymaster_submit.mjs setup --network sepolia
//   node paymaster_submit.mjs status --contract 0x... --model-id 0x1
//
import {
  Account,
  PaymasterRpc,
  RpcProvider,
  CallData,
  ETransactionVersion,
  ec,
  hash,
  num,
  byteArray,
} from "starknet";
import { readFileSync, writeFileSync, mkdirSync, existsSync, unlinkSync, renameSync, statSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { createHash } from "crypto";

// ─── Constants ────────────────────────────────────────────────────────

const NETWORKS = {
  sepolia: {
    rpcPublic: "https://api.cartridge.gg/x/starknet/sepolia",
    paymasterUrl: "https://sepolia.paymaster.avnu.fi",
    explorer: "https://sepolia.starkscan.co/tx/",
    factory: "0x2f69e566802910359b438ccdb3565dce304a7cc52edbf9fd246d6ad2cd89ce4",
    // Argent Account v0.4.0 — SNIP-9 compatible (required by AVNU paymaster).
    // Constructor: (owner: felt252, guardian: felt252).
    accountClassHash:
      "0x029927c8af6bccf3f6fda035981e765a7bdbf18a2dc0d630494f8758aa908e2b",
    identityRegistry:
      "0x72eb37b0389e570bf8b158ce7f0e1e3489de85ba43ab3876a0594df7231631",
  },
  mainnet: {
    rpcPublic: "https://free-rpc.nethermind.io/mainnet-juno/",
    paymasterUrl: "https://starknet.paymaster.avnu.fi",
    explorer: "https://starkscan.co/tx/",
    factory: "",
    accountClassHash: "",
    identityRegistry: "",
    notReady: true,
  },
};

const ACCOUNT_CONFIG_DIR = join(homedir(), ".obelysk", "starknet");
const ACCOUNT_CONFIG_FILE = join(ACCOUNT_CONFIG_DIR, "pipeline_account.json");

function parsePositiveIntEnv(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === null || String(raw).trim() === "") return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n <= 0) {
    throw new Error(`${name} must be a positive integer (got: ${raw})`);
  }
  return n;
}

const MAX_GKR_CALLDATA_FELTS = parsePositiveIntEnv(
  "OBELYSK_MAX_GKR_CALLDATA_FELTS",
  300000
);
const MAX_GKR_MODE4_CALLDATA_FELTS = parsePositiveIntEnv(
  "OBELYSK_MAX_GKR_MODE4_CALLDATA_FELTS",
  120000
);
const MIN_GKR_MODE4_CALLDATA_FELTS = parsePositiveIntEnv(
  "OBELYSK_MIN_GKR_MODE4_CALLDATA_FELTS",
  1000
);
// Per-chunk felt count for upload_gkr_chunk. The contract wraps each chunk in
// ~3 felts overhead (session_id, chunk_idx, array_len), so the total TX calldata
// is CHUNK_SIZE_FELTS + 3. Public RPCs (Cartridge, Nethermind) drop TXs
// with >~2500 felt calldata from the mempool (RECEIVED but never ACCEPTED_ON_L2).
// 2000 felts + 3 = 2003 is well within limits, while reducing chunk count by ~25%
// vs the prior 1500 default. Use OBELYSK_CHUNK_SIZE=1500 if your RPC is flaky.
const CHUNK_SIZE_FELTS = (() => {
  const raw = process.env.OBELYSK_CHUNK_SIZE;
  if (raw === undefined || raw === null || String(raw).trim() === "") return 2000;
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 1) {
    throw new Error(`OBELYSK_CHUNK_SIZE must be a positive integer (got: ${raw})`);
  }
  return n;
})();

// ─── Argument Parsing ─────────────────────────────────────────────────

function parseArgs(argv) {
  const args = {};
  const positional = [];
  for (let i = 2; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const key = argv[i].slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        args[key] = next;
        i++;
      } else {
        args[key] = true;
      }
    } else {
      positional.push(argv[i]);
    }
  }
  return { command: positional[0], ...args };
}

// ─── Helpers ──────────────────────────────────────────────────────────

function getProvider(network) {
  const rpcUrl =
    process.env.STARKNET_RPC ||
    (process.env.ALCHEMY_KEY
      ? `https://starknet-${network}.g.alchemy.com/starknet/version/rpc/v0_8/${process.env.ALCHEMY_KEY}`
      : NETWORKS[network].rpcPublic);
  return new RpcProvider({ nodeUrl: rpcUrl, batch: 0 });
}

function normalizePrivateKey(rawKey) {
  // starknet.js versions may return randomPrivateKey() as:
  // - hex string
  // - bigint/number
  // - byte array (Uint8Array / number[])
  if (typeof rawKey === "string") {
    return num.toHex(rawKey);
  }
  if (typeof rawKey === "bigint" || typeof rawKey === "number") {
    return num.toHex(rawKey);
  }
  if (rawKey && typeof rawKey === "object") {
    const isTypedArray =
      ArrayBuffer.isView(rawKey) || Array.isArray(rawKey);
    if (isTypedArray) {
      const hex = Buffer.from(rawKey).toString("hex");
      return num.toHex(`0x${hex}`);
    }
  }
  throw new Error("Unsupported private key format returned by starknet.js");
}

function loadAccountConfig() {
  if (!existsSync(ACCOUNT_CONFIG_FILE)) return null;
  return JSON.parse(readFileSync(ACCOUNT_CONFIG_FILE, "utf-8"));
}

function saveAccountConfig(config) {
  mkdirSync(ACCOUNT_CONFIG_DIR, { recursive: true });
  writeFileSync(ACCOUNT_CONFIG_FILE, JSON.stringify(config, null, 2));
}

function getAccount(provider, privateKey, address, network = "sepolia") {
  const net = NETWORKS[network];
  const paymasterOpts = net && net.paymasterUrl
    ? { nodeUrl: net.paymasterUrl }
    : { default: true };
  // AVNU requires an API key for sponsored mode (even on Sepolia).
  // Obtain one at https://portal.avnu.fi — set AVNU_PAYMASTER_API_KEY env var.
  const apiKey = process.env.AVNU_PAYMASTER_API_KEY;
  if (apiKey && paymasterOpts.nodeUrl) {
    paymasterOpts.headers = { "x-paymaster-api-key": apiKey };
  }
  return new Account({
    provider,
    address,
    signer: privateKey,
    transactionVersion: ETransactionVersion.V3,
    paymaster: new PaymasterRpc(paymasterOpts),
  });
}

function jsonOutput(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function die(msg) {
  process.stderr.write(`[ERR] ${msg}\n`);
  process.exit(1);
}

// BigInt-safe JSON serialization + atomic file write (write to temp, then rename).
// Prevents corrupted resume files from partial writes or BigInt TypeError.
function safeWriteJson(filePath, obj) {
  const json = JSON.stringify(obj, (_, v) => typeof v === "bigint" ? v.toString() : v, 2);
  const tmp = filePath + `.tmp.${process.pid}`;
  writeFileSync(tmp, json);
  renameSync(tmp, filePath);
}

function truncateRpcError(e) {
  const msg = e.message || String(e);
  // starknet.js RpcError embeds the full request params (100K+ chars of
  // calldata JSON) in the error message.  Extract just the error code.
  //
  // Error format: "RPC: <method> with params { <giant JSON> }\n\n <code>: "<reason>""
  // The error code + reason is at the END, after the params JSON.
  const codeAtEnd = msg.match(/\}\s*\n\s*(\d+):\s*"?(.{1,500})/);
  if (codeAtEnd) return `RPC error ${codeAtEnd[1]}: ${codeAtEnd[2]}`;
  const codeMatch = msg.match(/(-\d+):\s*"?(.{1,300})/);
  if (codeMatch) return `RPC ${codeMatch[1]}: ${codeMatch[2]}`;
  // Look for "Transaction execution error" or "Contract error" patterns
  const execErr = msg.match(/execution error[:\s]*(.{1,500})/i);
  if (execErr) return `Execution error: ${execErr[1]}`;
  const contractErr = msg.match(/Contract error[:\s]*(.{1,500})/i);
  if (contractErr) return `Contract error: ${contractErr[1]}`;
  // Fallback: last 500 chars (often contains the actual error).
  // IMPORTANT: Preserve critical keywords that callers depend on for control flow.
  // E.g., "INSUFFICIENT" triggers fee cache invalidation; if truncated away, the
  // stale fee cache is never cleared, causing cascading failures.
  if (msg.length > 500) {
    const tail = msg.slice(-500);
    // Check if critical keywords are in the full message but not in the tail.
    const criticalKeywords = ["INSUFFICIENT", "insufficient", "rate limit", "429", "REVERTED"];
    const preserved = criticalKeywords.filter(kw => msg.includes(kw) && !tail.includes(kw));
    if (preserved.length > 0) {
      return `[${preserved.join(",")}] ...${tail}`;
    }
    return "..." + tail;
  }
  return msg;
}

function info(msg) {
  process.stderr.write(`[INFO] ${msg}\n`);
}

function abiHasEntrypoint(abiEntries, entrypoint) {
  if (!Array.isArray(abiEntries)) return false;
  const stack = [...abiEntries];
  while (stack.length > 0) {
    const entry = stack.pop();
    if (!entry || typeof entry !== "object") continue;

    if (entry.type === "function" && typeof entry.name === "string") {
      const fullName = entry.name;
      const shortName = fullName.split("::").pop();
      if (fullName === entrypoint || shortName === entrypoint) {
        return true;
      }
    }

    if (Array.isArray(entry.items)) {
      for (const item of entry.items) stack.push(item);
    }
  }
  return false;
}

async function preflightContractEntrypoint(provider, contractAddress, entrypoint) {
  let classAt;
  try {
    classAt = await provider.getClassAt(contractAddress);
  } catch (e) {
    info(
      `Entrypoint preflight skipped (failed to fetch contract class at ${contractAddress}): ${e.message || e}`
    );
    return;
  }

  let abi = classAt?.abi;
  if (typeof abi === "string") {
    try {
      abi = JSON.parse(abi);
    } catch {
      abi = null;
    }
  }
  if (!Array.isArray(abi)) {
    info(
      `Entrypoint preflight skipped (contract ABI unavailable for ${contractAddress})`
    );
    return;
  }

  if (!abiHasEntrypoint(abi, entrypoint)) {
    if (
      entrypoint === "verify_model_gkr_v2" ||
      entrypoint === "verify_model_gkr_v3" ||
      entrypoint === "verify_model_gkr_v4"
    ) {
      die(
        `Contract ${contractAddress} does not expose ${entrypoint}. ` +
          "Deploy the upgraded verifier, or submit with v1 (Sequential mode)."
      );
    }
    die(`Contract ${contractAddress} does not expose required entrypoint: ${entrypoint}`);
  }
}

// ─── Ephemeral Account ───────────────────────────────────────────────
//
// Generates a keypair and computes the counterfactual address for the
// agent account class (already declared on Sepolia). The account doesn't
// need to exist on-chain yet — deploymentData in the paymaster TX will
// deploy it atomically alongside the proof verification call.
//
// Argent Account v0.4.0 constructor: (owner: felt252, guardian: felt252)
// For ephemeral accounts we pass guardian = 0x0 (no guardian).

function generateEphemeralAccount(network) {
  const net = NETWORKS[network];
  if (!net?.accountClassHash) die(`No account class hash for ${network}`);

  const privateKey = normalizePrivateKey(ec.starkCurve.utils.randomPrivateKey());
  const publicKey = ec.starkCurve.getStarkKey(privateKey);
  const salt = publicKey;

  // Argent account constructor calldata: [owner=publicKey, guardian=0x0]
  const constructorCalldata = CallData.compile([publicKey, "0x0"]);

  // Deterministic address: hash(deployer=0, salt, classHash, constructorCalldata)
  const address = hash.calculateContractAddressFromHash(
    salt,
    net.accountClassHash,
    constructorCalldata,
    0 // deployer = 0 → self-deployed via DEPLOY_ACCOUNT
  );

  return { privateKey, publicKey, salt, address, constructorCalldata };
}

// ─── Paymaster Execution ─────────────────────────────────────────────

async function deployAccountDirect(provider, account, deploymentData, network) {
  // Deploy account via standard DEPLOY_ACCOUNT transaction.
  // This uses the account's own key to self-deploy (gas paid by the account
  // or estimated from the provider). On Sepolia, a small balance is needed
  // to cover gas. If the account has no balance, we try paymaster first as
  // a fallback.
  const net = NETWORKS[network];

  // Try paymaster-sponsored deployment first (empty calls + deploymentData)
  try {
    info("Attempting paymaster-sponsored account deployment...");
    const deployFeeDetails = {
      feeMode: { mode: "sponsored" },
      deploymentData: { ...deploymentData, version: 1 },
    };
    const deployEstimation = await account.estimatePaymasterTransactionFee(
      [],
      deployFeeDetails
    );
    const deployResult = await account.executePaymasterTransaction(
      [],
      deployFeeDetails,
      deployEstimation.suggested_max_fee_in_gas_token
    );
    const deployTxHash = deployResult.transaction_hash;
    info(`Account deploy TX (paymaster): ${deployTxHash}`);
    info("Waiting for account deployment confirmation...");
    const deployReceipt = await provider.waitForTransaction(deployTxHash);
    const deployStatus = deployReceipt.execution_status ?? deployReceipt.status ?? "unknown";
    if (deployStatus === "REVERTED") {
      throw new Error(`TX reverted: ${deployReceipt.revert_reason || "unknown"}`);
    }
    info("Account deployed successfully via paymaster.");
    return deployTxHash;
  } catch (e) {
    info(`Paymaster account deploy failed: ${truncateRpcError(e)}`);
    info("Falling back to standard DEPLOY_ACCOUNT...");
  }

  // ── Fund the ephemeral account if a funder key is available ──
  // OBELYSK_FUNDER_KEY + OBELYSK_FUNDER_ADDRESS: a pre-deployed, funded account
  // (e.g. the deployer) that can transfer STRK to cover deploy gas.
  const funderKey = process.env.OBELYSK_FUNDER_KEY;
  const funderAddress = process.env.OBELYSK_FUNDER_ADDRESS;
  if (funderKey && funderAddress) {
    const STRK_TOKEN = "0x04718f5a0fc34cc1af16a1cdee98ffb20c31f5cd61d6ab07201858f4287c938d";
    const FUND_AMOUNT = "0x16345785D8A0000"; // 0.1 STRK (10^17 wei)
    const funder = new Account({
      provider,
      address: funderAddress,
      signer: funderKey,
      transactionVersion: ETransactionVersion.V3,
    });
    info(`Funding ephemeral account from ${funderAddress}...`);
    try {
      const fundResult = await funder.execute([{
        contractAddress: STRK_TOKEN,
        entrypoint: "transfer",
        calldata: CallData.compile([deploymentData.address || account.address, FUND_AMOUNT, "0x0"]),
      }]);
      info(`Fund TX: ${fundResult.transaction_hash}`);
      await provider.waitForTransaction(fundResult.transaction_hash);
      info("Ephemeral account funded with 0.01 STRK.");
    } catch (fundErr) {
      info(`Funding failed: ${truncateRpcError(fundErr)}`);
      die("Cannot deploy ephemeral account: paymaster rejected deploy, and funding failed. " +
          "Set OBELYSK_FUNDER_KEY + OBELYSK_FUNDER_ADDRESS to a funded account, " +
          "or use STARKNET_PRIVATE_KEY + STARKNET_ACCOUNT_ADDRESS for a pre-deployed account.");
    }
  }

  // Fallback: standard deploy_account (needs gas balance on the account)
  const deployPayload = {
    classHash: deploymentData.class_hash || deploymentData.classHash,
    constructorCalldata: deploymentData.calldata || deploymentData.constructorCalldata,
    addressSalt: deploymentData.salt,
  };

  let transaction_hash;
  try {
    const result = await account.deployAccount(deployPayload);
    transaction_hash = result.transaction_hash;
  } catch (deployErr) {
    const fullMsg = deployErr.message || String(deployErr);
    info(`Standard DEPLOY_ACCOUNT error (full): ${fullMsg.slice(0, 2000)}`);
    if (!funderKey) {
      die("Cannot deploy ephemeral account: no STRK balance and no funder key. " +
          "Set OBELYSK_FUNDER_KEY + OBELYSK_FUNDER_ADDRESS to a funded Starknet account, " +
          "or use STARKNET_PRIVATE_KEY + STARKNET_ACCOUNT_ADDRESS for a pre-deployed account.");
    }
    throw deployErr;
  }
  info(`Account deploy TX (standard): ${transaction_hash}`);
  info("Waiting for account deployment confirmation...");
  const receipt = await provider.waitForTransaction(transaction_hash);
  const status = receipt.execution_status ?? receipt.status ?? "unknown";
  if (status === "REVERTED") {
    die(`Account deployment reverted: ${receipt.revert_reason || "unknown"}`);
  }
  info("Account deployed successfully.");
  return transaction_hash;
}

async function executeViaPaymaster(account, calls) {
  const callsArray = Array.isArray(calls) ? calls : [calls];
  const feeDetails = { feeMode: { mode: "sponsored" } };

  info("Estimating paymaster fee...");
  const estimation = await account.estimatePaymasterTransactionFee(
    callsArray,
    feeDetails
  );
  if (!estimation || !estimation.suggested_max_fee_in_gas_token) {
    throw new Error("Paymaster fee estimation returned empty/invalid result");
  }

  info("Executing via AVNU paymaster (sponsored mode)...");
  const result = await account.executePaymasterTransaction(
    callsArray,
    feeDetails,
    estimation.suggested_max_fee_in_gas_token
  );
  if (!result || !result.transaction_hash) {
    throw new Error(
      `Paymaster executePaymasterTransaction returned invalid result: ${JSON.stringify(result)?.slice(0, 200)}`
    );
  }

  return result.transaction_hash;
}


// ─── Calldata Validation ──────────────────────────────────────────────
//
// IMPORTANT: The canonical implementation of proof calldata validation lives in
// validate_proof.py (same directory). This JS parseVerifyCalldata function
// duplicates that logic for use within the Node.js paymaster flow.
// When updating validation rules, update validate_proof.py FIRST, then
// mirror changes here to keep both implementations in sync.
//

function parseVerifyCalldata(proofData, fallbackModelId) {
  const verifyCalldata = proofData.verify_calldata;
  if (!verifyCalldata || typeof verifyCalldata !== "object" || Array.isArray(verifyCalldata)) {
    die("Proof file missing 'verify_calldata' object");
  }

  const schemaVersion = Number(verifyCalldata.schema_version);
  if (schemaVersion !== 1 && schemaVersion !== 2) {
    die(`verify_calldata.schema_version must be 1 or 2 (got: ${JSON.stringify(verifyCalldata.schema_version)})`);
  }

  // ── Schema v2: chunked session mode ──
  if (schemaVersion === 2) {
    const entrypoint = verifyCalldata.entrypoint;
    if (entrypoint !== "verify_gkr_from_session") {
      die(`schema_version 2 requires entrypoint=verify_gkr_from_session (got: ${entrypoint})`);
    }
    if (verifyCalldata.mode !== "chunked") {
      die(`schema_version 2 requires mode=chunked (got: ${verifyCalldata.mode})`);
    }
    const chunks = verifyCalldata.chunks;
    if (!Array.isArray(chunks) || chunks.length === 0) {
      die("schema_version 2 requires non-empty chunks array");
    }
    const totalFelts = verifyCalldata.total_felts;
    if (typeof totalFelts !== "number" || totalFelts <= 0) {
      die("schema_version 2 requires positive total_felts");
    }
    const numChunks = verifyCalldata.num_chunks;
    if (typeof numChunks !== "number" || numChunks !== chunks.length) {
      die(`num_chunks mismatch: declared=${numChunks} actual=${chunks.length}`);
    }
    const circuitDepth = verifyCalldata.circuit_depth;
    const numLayers = verifyCalldata.num_layers;
    const weightBindingMode = Number(verifyCalldata.weight_binding_mode);
    if (typeof circuitDepth !== "number" || circuitDepth <= 0) {
      die("schema_version 2 requires positive circuit_depth");
    }
    if (typeof numLayers !== "number" || numLayers <= 0) {
      die("schema_version 2 requires positive num_layers");
    }
    if (![3, 4].includes(weightBindingMode)) {
      die(`schema_version 2 requires weight_binding_mode in {3,4} (got: ${JSON.stringify(verifyCalldata.weight_binding_mode)})`);
    }
    // Validate chunk sizes.
    let feltCount = 0;
    for (let i = 0; i < chunks.length; i++) {
      if (!Array.isArray(chunks[i]) || chunks[i].length === 0) {
        die(`chunk[${i}] must be a non-empty array`);
      }
      if (chunks[i].length > 4000) {
        die(`chunk[${i}] exceeds max 4000 felts (has ${chunks[i].length})`);
      }
      feltCount += chunks[i].length;
    }
    if (feltCount !== totalFelts) {
      die(`chunk felt total mismatch: sum=${feltCount} declared=${totalFelts}`);
    }
    const modelId = verifyCalldata.model_id || proofData.model_id || fallbackModelId;

    // For Mode 4 RLC-only: strip weight_opening_proofs from session data.
    // The on-chain verifier never uses them, and including them pushes
    // total storage reads past the per-TX step limit.
    let finalChunks = chunks;
    let finalTotalFelts = totalFelts;
    let finalNumChunks = numChunks;
    if (weightBindingMode === 4) {
      // Flatten chunks → parse sections → check if Mode 4 RLC → strip openings
      const flat = [];
      for (const c of chunks) for (const f of c) flat.push(f);
      const readNatV2 = (i, label) => {
        if (i < 0 || i >= flat.length) {
          die(`schema v2 Mode 4: index out of bounds reading ${label}: idx=${i}, flat.length=${flat.length}`);
        }
        const s = String(flat[i]);
        if (s === "undefined" || s === "null" || s === "") {
          die(`schema v2 Mode 4: empty/null value at index ${i} for ${label}`);
        }
        let n;
        try {
          n = s.startsWith("0x") || s.startsWith("0X") ? Number(BigInt(s)) : Number(s);
        } catch (e) {
          die(`schema v2 Mode 4: invalid number at index ${i} for ${label}: ${s}`);
        }
        if (!Number.isSafeInteger(n) || n < 0) {
          die(`schema v2 Mode 4: invalid ${label} at index ${i}: ${s}`);
        }
        return n;
      };
      let off = 0;
      let sec6Off = 0, sec6Len = 0;
      const isIoPacked = verifyCalldata.io_packed === true;
      // Walk through 6 length-prefixed sections, capturing section 6 (weight_binding_data) position.
      for (let sec = 0; sec < 6; sec++) {
        if (sec === 5) { sec6Off = off; }
        if (sec === 0 && isIoPacked) {
          // Section 1 (raw_io_data) packed: [original_len, packed_count, packed_data...]
          const _origLen = readNatV2(off, "raw_io_original_len");
          const packedCount = readNatV2(off + 1, "raw_io_packed_count");
          off += 2 + packedCount;
        } else {
          const secLen = readNatV2(off, `section_${sec + 1}_len`);
          if (sec === 5) { sec6Len = secLen; }
          off += 1 + secLen;
        }
        if (off > flat.length) {
          die(`schema v2 Mode 4: section ${sec + 1} overflows flat data (off=${off}, flat.length=${flat.length})`);
        }
      }
      // off now points at weight_opening_proofs start
      const wopFelts = flat.length - off;
      // Check if weight_binding_data (section 6) is RLC marker
      let isRlc = false;
      if (sec6Len === 2 && (sec6Off + 1) < flat.length) {
        try { isRlc = BigInt(flat[sec6Off + 1]) === 0x524C43n; } catch { /* not RLC */ }
      }
      if (isRlc && wopFelts > 1) {
        // Replace opening proofs with empty array [0]
        const stripped = flat.slice(0, off);
        stripped.push("0");
        info(`  Mode 4 RLC v2: stripping ${wopFelts} felts of weight_opening_proofs (${flat.length} → ${stripped.length})`);
        // Re-chunk
        const chunkSize = CHUNK_SIZE_FELTS;
        finalChunks = [];
        for (let i = 0; i < stripped.length; i += chunkSize) {
          finalChunks.push(stripped.slice(i, i + chunkSize));
        }
        finalTotalFelts = stripped.length;
        finalNumChunks = finalChunks.length;
      }
    }

    // ── Re-chunk if any chunk exceeds the on-chain TX calldata limit ──
    // Cartridge/public RPCs silently drop TXs with calldata > ~2000 felts from
    // the mempool (TX gets RECEIVED but never ACCEPTED_ON_L2). The Rust prover
    // pre-chunks at 4000 felts, so we re-chunk to a smaller size here.
    const maxChunkFelts = CHUNK_SIZE_FELTS;
    const needsRechunk = finalChunks.some((c) => c.length > maxChunkFelts);
    if (needsRechunk) {
      const flat = [];
      for (const c of finalChunks) for (const f of c) flat.push(f);
      finalChunks = [];
      for (let i = 0; i < flat.length; i += maxChunkFelts) {
        finalChunks.push(flat.slice(i, i + maxChunkFelts));
      }
      finalNumChunks = finalChunks.length;
      info(`  Re-chunked: ${numChunks} chunks @ ≤${chunks[0].length} → ${finalNumChunks} chunks @ ≤${maxChunkFelts} felts`);
    }

    return {
      entrypoint,
      calldata: [],
      uploadChunks: finalChunks,
      sessionId: null,
      modelId,
      schemaVersion,
      chunked: true,
      totalFelts: finalTotalFelts,
      numChunks: finalNumChunks,
      circuitDepth,
      numLayers,
      weightBindingMode,
      packed: verifyCalldata.packed === true || verifyCalldata.packed === undefined,
      ioPacked: verifyCalldata.io_packed === true,
    };
  }

  const entrypoint = verifyCalldata.entrypoint;
  if (typeof entrypoint !== "string" || entrypoint.length === 0) {
    die("verify_calldata.entrypoint must be a non-empty string");
  }
  const allowedEntrypoints = new Set([
    "verify_model_gkr",
    "verify_model_gkr_v2",
    "verify_model_gkr_v3",
    "verify_model_gkr_v4",
    "verify_model_gkr_v4_packed",
    "verify_model_gkr_v4_packed_io",
  ]);
  if (!allowedEntrypoints.has(entrypoint)) {
    const reason = verifyCalldata.reason || "(no reason recorded)";
    die(
      `Unsupported entrypoint in hardened pipeline (got: ${entrypoint}). Reason: ${reason}`
    );
  }

  const rawCalldata = verifyCalldata.calldata;
  if (!Array.isArray(rawCalldata) || rawCalldata.length === 0) {
    die("verify_calldata.calldata must be a non-empty array");
  }

  // ── Auto-convert v1 → chunked v2 when calldata exceeds single-TX limit ──
  // Starknet enforces ~5000-felt max calldata per TX. When a v4 proof exceeds
  // this threshold, we parse the flat v4 calldata and rebuild it as a chunked
  // session data buffer (schema_version 2) on the fly.
  const CHUNKED_THRESHOLD = 5000;
  // Chunk size from CHUNK_SIZE_FELTS (default 2000). Public RPCs drop TXs with
  // calldata >~2500 felts from mempool (RECEIVED but never ACCEPTED_ON_L2).
  const MAX_CHUNK_FELTS = CHUNK_SIZE_FELTS;
  if (entrypoint === "verify_model_gkr_v4" && rawCalldata.length > CHUNKED_THRESHOLD) {
    info(`Auto-converting v1 calldata (${rawCalldata.length} felts) to chunked session format...`);
    const cd = rawCalldata.map((v) => String(v));
    const readNat = (idx, label) => {
      if (idx >= cd.length) {
        die(`auto-chunk: index out of bounds reading ${label}: idx=${idx} >= cd.length=${cd.length}`);
      }
      const s = cd[idx];
      const n = s.startsWith("0x") || s.startsWith("0X") ? Number(BigInt(s)) : Number(s);
      if (!Number.isSafeInteger(n) || n < 0) die(`auto-chunk: invalid ${label} at idx ${idx}: ${s}`);
      return n;
    };

    // Parse v4 calldata structure:
    // [0] model_id, [1] raw_io_len, raw_io..., circuit_depth, num_layers,
    // matmul_dims_len, matmul_dims..., deq_bits_len, deq_bits...,
    // proof_data_len, proof_data..., wc_len, wc..., weight_binding_mode,
    // wbd_len, wbd..., wop_count, wop_data...
    let idx = 0;
    // Helper: advance idx by a section length, dying if it overshoots calldata.
    const advanceSection = (len, label) => {
      idx += len;
      if (idx > cd.length) {
        die(`auto-chunk: ${label} overflows calldata (idx=${idx} > cd.length=${cd.length})`);
      }
    };

    const modelIdStr = cd[idx]; idx += 1;

    // raw_io_data section
    const rawIoLen = readNat(idx, "raw_io_len"); idx += 1;
    const rawIoStart = idx; advanceSection(rawIoLen, "raw_io_data");

    // circuit_depth, num_layers (scalars — stored in session metadata)
    const circuitDepth = readNat(idx, "circuit_depth"); idx += 1;
    const numLayers = readNat(idx, "num_layers"); idx += 1;

    // matmul_dims section
    const matmulDimsLen = readNat(idx, "matmul_dims_len"); idx += 1;
    const matmulDimsStart = idx; advanceSection(matmulDimsLen, "matmul_dims");

    // dequantize_bits section
    const deqBitsLen = readNat(idx, "deq_bits_len"); idx += 1;
    const deqBitsStart = idx; advanceSection(deqBitsLen, "dequantize_bits");

    // proof_data section
    const proofDataLen = readNat(idx, "proof_data_len"); idx += 1;
    const proofDataStart = idx; advanceSection(proofDataLen, "proof_data");

    // weight_commitments section
    const wcLen = readNat(idx, "wc_len"); idx += 1;
    const wcStart = idx; advanceSection(wcLen, "weight_commitments");

    // weight_binding_mode (scalar — stored in session metadata)
    const weightBindingMode = readNat(idx, "weight_binding_mode"); idx += 1;

    // weight_binding_data section
    const wbdLen = readNat(idx, "wbd_len"); idx += 1;
    const wbdStart = idx; advanceSection(wbdLen, "weight_binding_data");

    // weight_opening_proofs (rest of calldata = Serde-serialized Array<MleOpeningProof>)
    const wopStart = idx;
    if (wopStart > cd.length) {
      die(`auto-chunk: calldata truncated — expected weight_opening_proofs at idx ${wopStart} but cd.length=${cd.length}`);
    }
    const wopLen = cd.length - wopStart;

    // Build session data: length-prefixed sections, no model_id/circuit_depth/num_layers/binding_mode
    const sessionData = [];
    // 1. raw_io_data
    sessionData.push(String(rawIoLen));
    for (let i = 0; i < rawIoLen; i++) sessionData.push(cd[rawIoStart + i]);
    // 2. matmul_dims
    sessionData.push(String(matmulDimsLen));
    for (let i = 0; i < matmulDimsLen; i++) sessionData.push(cd[matmulDimsStart + i]);
    // 3. dequantize_bits
    sessionData.push(String(deqBitsLen));
    for (let i = 0; i < deqBitsLen; i++) sessionData.push(cd[deqBitsStart + i]);
    // 4. proof_data
    sessionData.push(String(proofDataLen));
    for (let i = 0; i < proofDataLen; i++) sessionData.push(cd[proofDataStart + i]);
    // 5. weight_commitments
    sessionData.push(String(wcLen));
    for (let i = 0; i < wcLen; i++) sessionData.push(cd[wcStart + i]);
    // 6. weight_binding_data
    sessionData.push(String(wbdLen));
    for (let i = 0; i < wbdLen; i++) sessionData.push(cd[wbdStart + i]);
    // 7. weight_opening_proofs (raw Serde — count + serialized data)
    //    For Mode 4 RLC-only: the on-chain verifier never uses opening proofs.
    //    Omitting them drops ~53K felts, reducing total from ~86K to ~33K felts
    //    which fits within Starknet's per-TX step limit for storage reads.
    // Standardize RLC marker detection via BigInt to handle any hex casing or decimal format.
    let isMode4Rlc = false;
    if (weightBindingMode === 4 && wbdLen === 2 && wbdStart < cd.length) {
      try { isMode4Rlc = BigInt(cd[wbdStart]) === 0x524C43n; } catch { /* not RLC */ }
    }
    if (isMode4Rlc) {
      sessionData.push("0");  // empty Array<MleOpeningProof>
      info(`  Mode 4 RLC: stripping ${wopLen} felts of weight_opening_proofs`);
    } else {
      for (let i = 0; i < wopLen; i++) sessionData.push(cd[wopStart + i]);
    }

    // Split into chunks
    const chunks = [];
    for (let i = 0; i < sessionData.length; i += MAX_CHUNK_FELTS) {
      chunks.push(sessionData.slice(i, i + MAX_CHUNK_FELTS));
    }
    const totalFelts = sessionData.length;
    const numChunks = chunks.length;

    info(`  Session data: ${totalFelts} felts → ${numChunks} chunks (max ${MAX_CHUNK_FELTS}/chunk)`);
    info(`  Model: ${modelIdStr}, circuit_depth=${circuitDepth}, num_layers=${numLayers}, binding_mode=${weightBindingMode}`);

    return {
      entrypoint: "verify_gkr_from_session",
      calldata: [],
      uploadChunks: chunks,
      sessionId: null,
      modelId: modelIdStr,
      schemaVersion: 2,
      chunked: true,
      totalFelts,
      numChunks,
      circuitDepth,
      numLayers,
      weightBindingMode,
      // V4 single-TX calldata from build_verify_model_gkr_v4_calldata is UNPACKED
      // (4 felts per QM31). The Rust chunked builder (build_chunked_gkr_calldata)
      // does packing internally, but this JS auto-convert path works on unpacked data.
      // Setting packed=true here would cause the Cairo verifier to misinterpret felts.
      packed: false,
      ioPacked: false,
    };
  }

  if (rawCalldata.length > MAX_GKR_CALLDATA_FELTS) {
    die(
      `calldata too large for hardened submit path: ${rawCalldata.length} felts ` +
        `(max ${MAX_GKR_CALLDATA_FELTS}). ` +
        "Likely legacy per-opening mode; use --submit to generate mode-4 aggregated proof."
    );
  }

  if (proofData.submission_ready === false) {
    const mode = proofData.weight_opening_mode ?? "unknown";
    const reason =
      verifyCalldata.reason ?? proofData.soundness_gate_error ?? "unspecified";
    die(
      `proof is marked submission_ready=false (entrypoint=${entrypoint}, weight_opening_mode=${mode}, reason=${reason})`
    );
  }
  const weightOpeningMode =
    proofData.weight_opening_mode !== undefined
      ? String(proofData.weight_opening_mode)
      : undefined;
  if (entrypoint === "verify_model_gkr") {
    if (weightOpeningMode !== undefined && weightOpeningMode !== "Sequential") {
      die(
        `${entrypoint} requires weight_opening_mode=Sequential (got: ${proofData.weight_opening_mode})`
      );
    }
  } else if (entrypoint === "verify_model_gkr_v2" || entrypoint === "verify_model_gkr_v3") {
    const allowedModes = new Set(["Sequential", "BatchedSubchannelV1"]);
    if (weightOpeningMode !== undefined && !allowedModes.has(weightOpeningMode)) {
      die(
        `${entrypoint} requires weight_opening_mode in {Sequential,BatchedSubchannelV1} (got: ${proofData.weight_opening_mode})`
      );
    }
  } else if (entrypoint === "verify_model_gkr_v4") {
    const allowedV4Modes = new Set([
      "AggregatedOpeningsV4Experimental",
      "AggregatedOracleSumcheck",
    ]);
    if (weightOpeningMode !== undefined && !allowedV4Modes.has(weightOpeningMode)) {
      die(
        `${entrypoint} requires weight_opening_mode in {AggregatedOpeningsV4Experimental,AggregatedOracleSumcheck} ` +
          `(got: ${proofData.weight_opening_mode})`
      );
    }
  }

  const rawChunks = verifyCalldata.upload_chunks ?? [];
  if (!Array.isArray(rawChunks)) {
    die("verify_calldata.upload_chunks must be an array");
  }
  if (rawChunks.length > 0) {
    die("verify_model_gkr(*) payload must not include upload_chunks");
  }

  const hasSessionPlaceholder = rawCalldata.some((v) => String(v) === "__SESSION_ID__");
  if (hasSessionPlaceholder) {
    die("verify_model_gkr(*) calldata must not include __SESSION_ID__ placeholder");
  }

  const calldata = rawCalldata.map((v) => String(v));
  const parseNat = (token, label) => {
    const s = String(token);
    let n;
    try {
      n = s.startsWith("0x") || s.startsWith("0X") ? Number(BigInt(s)) : Number(s);
    } catch (e) {
      die(`invalid ${label}: ${s} (${e.message || e})`);
    }
    if (!Number.isSafeInteger(n) || n < 0) {
      die(`invalid ${label}: ${s}`);
    }
    return n;
  };
  const parseFeltBigInt = (token, label) => {
    const s = String(token);
    try {
      return BigInt(s);
    } catch (e) {
      die(`invalid ${label}: ${s} (${e.message || e})`);
    }
  };

  if (
    entrypoint === "verify_model_gkr_v2" ||
    entrypoint === "verify_model_gkr_v3" ||
    entrypoint === "verify_model_gkr_v4"
  ) {
    // model_id, raw_io_data, circuit_depth, num_layers, matmul_dims,
    // dequantize_bits, proof_data, weight_commitments, weight_binding_mode, weight_openings...
    let idx = 0;
    idx += 1;
    if (idx >= calldata.length) die("v2 calldata truncated before raw_io length");
    const rawIoLen = parseNat(calldata[idx], "raw_io_data length");
    idx += 1 + rawIoLen;
    idx += 2;
    if (idx >= calldata.length) die("v2 calldata truncated before matmul_dims length");
    const matmulLen = parseNat(calldata[idx], "matmul_dims length");
    idx += 1 + matmulLen;
    if (idx >= calldata.length) die("v2 calldata truncated before dequantize_bits length");
    const deqLen = parseNat(calldata[idx], "dequantize_bits length");
    idx += 1 + deqLen;
    if (idx >= calldata.length) die("v2 calldata truncated before proof_data length");
    const proofDataLen = parseNat(calldata[idx], "proof_data length");
    idx += 1 + proofDataLen;
    if (idx >= calldata.length) die("v2 calldata truncated before weight_commitments length");
    const weightCommitmentsLen = parseNat(calldata[idx], "weight_commitments length");
    idx += 1 + weightCommitmentsLen;
    if (idx >= calldata.length) die("v2 calldata truncated before weight_binding_mode");
    const weightBindingMode = parseNat(calldata[idx], "weight_binding_mode");
    if (entrypoint === "verify_model_gkr_v2" && !new Set([0, 1]).has(weightBindingMode)) {
      die(`${entrypoint} requires weight_binding_mode in {0,1} (got ${weightBindingMode})`);
    }
    if (entrypoint === "verify_model_gkr_v4" && !new Set([3, 4]).has(weightBindingMode)) {
      die(`${entrypoint} requires weight_binding_mode in {3,4} (got ${weightBindingMode})`);
    }
    let expectedMode = null;
    if (weightOpeningMode === "Sequential") {
      expectedMode = 0;
    } else if (weightOpeningMode === "BatchedSubchannelV1") {
      expectedMode = 1;
    } else if (weightOpeningMode === "AggregatedTrustlessV2") {
      expectedMode = 2;
    } else if (weightOpeningMode === "AggregatedOpeningsV4Experimental") {
      expectedMode = 3;
    } else if (weightOpeningMode === "AggregatedOracleSumcheck") {
      expectedMode = 4;
    }
    if (expectedMode !== null && weightBindingMode !== expectedMode) {
      die(
        `${entrypoint} expected weight_binding_mode=${expectedMode} for weight_opening_mode=${weightOpeningMode} (got ${weightBindingMode})`
      );
    }
    const allowedModes =
      entrypoint === "verify_model_gkr_v3"
        ? new Set([0, 1, 2])
        : entrypoint === "verify_model_gkr_v4"
          ? new Set([3, 4])
          : new Set([0, 1]);
    if (expectedMode === null && !allowedModes.has(weightBindingMode)) {
      die(
        `${entrypoint} requires weight_binding_mode in {${[...allowedModes].join(",")}} (got ${weightBindingMode})`
      );
    }
    if (proofData.weight_binding_mode_id !== undefined && proofData.weight_binding_mode_id !== null) {
      const artifactModeId = parseNat(proofData.weight_binding_mode_id, "weight_binding_mode_id");
      if (artifactModeId !== weightBindingMode) {
        die(
          `weight_binding_mode_id mismatch: artifact=${artifactModeId} calldata=${weightBindingMode}`
        );
      }
    }
    if (entrypoint === "verify_model_gkr_v3" || entrypoint === "verify_model_gkr_v4") {
      idx += 1; // consume weight_binding_mode
      if (idx >= calldata.length) die(`${entrypoint} calldata truncated before weight_binding_data length`);
      const weightBindingDataLen = parseNat(calldata[idx], "weight_binding_data length");
      idx += 1 + weightBindingDataLen;
      if (new Set([0, 1]).has(weightBindingMode) && weightBindingDataLen !== 0) {
        die(
          `${entrypoint} mode ${weightBindingMode} requires empty weight_binding_data (got len=${weightBindingDataLen})`
        );
      }
      if (weightBindingMode === 2 && weightBindingDataLen === 0) {
        die(`${entrypoint} mode 2 requires non-empty weight_binding_data`);
      }
      if (weightBindingMode === 3 && weightBindingDataLen === 0) {
        die(`${entrypoint} mode 3 requires non-empty weight_binding_data`);
      }
      if (weightBindingMode === 4 && weightBindingDataLen === 0) {
        die(`${entrypoint} mode 4 requires non-empty weight_binding_data`);
      }
      if (entrypoint === "verify_model_gkr_v4" && weightBindingMode === 4) {
        if (calldata.length > MAX_GKR_MODE4_CALLDATA_FELTS) {
          die(
            `${entrypoint} mode 4 calldata unexpectedly large: ${calldata.length} felts ` +
              `(max ${MAX_GKR_MODE4_CALLDATA_FELTS}). ` +
              "This looks like non-aggregated payload; regenerate proof with --submit."
          );
        }
        if (calldata.length < MIN_GKR_MODE4_CALLDATA_FELTS) {
          die(
            `${entrypoint} mode 4 calldata unexpectedly small: ${calldata.length} felts ` +
              `(min ${MIN_GKR_MODE4_CALLDATA_FELTS}).`
          );
        }
      }
      if (
        Array.isArray(proofData.weight_binding_data_calldata) &&
        proofData.weight_binding_data_calldata.length !== weightBindingDataLen
      ) {
        die(
          `weight_binding_data_calldata length mismatch: artifact=${proofData.weight_binding_data_calldata.length} calldata=${weightBindingDataLen}`
        );
      }
      if (Array.isArray(proofData.weight_binding_data_calldata)) {
        const calldataBindingData = calldata.slice(
          idx - weightBindingDataLen,
          idx
        );
        for (let j = 0; j < weightBindingDataLen; j++) {
          const artifactVal = parseFeltBigInt(
            proofData.weight_binding_data_calldata[j],
            `weight_binding_data_calldata[${j}]`
          );
          const calldataVal = parseFeltBigInt(
            calldataBindingData[j],
            `weight_binding_data[${j}]`
          );
          if (artifactVal !== calldataVal) {
            die(
              `weight_binding_data_calldata mismatch at index ${j}: artifact=${proofData.weight_binding_data_calldata[j]} calldata=${calldataBindingData[j]}`
            );
          }
        }
      }
    }
  }

  const uploadChunks = [];
  const sessionId = null;

  const modelId = calldata.length > 0 ? String(calldata[0]) : fallbackModelId;

  return {
    entrypoint,
    calldata,
    uploadChunks,
    sessionId,
    modelId,
    schemaVersion,
    chunked: false,
  };
}

async function fetchVerificationCount(provider, contract, modelId) {
  try {
    const result = await provider.callContract({
      contractAddress: contract,
      entrypoint: "get_verification_count",
      calldata: CallData.compile([modelId]),
    });
    return result.result ? BigInt(result.result[0] || "0") : 0n;
  } catch {
    return null;
  }
}

async function fetchProofVerifiedFlag(provider, contract, modelId) {
  try {
    const result = await provider.callContract({
      contractAddress: contract,
      entrypoint: "is_proof_verified",
      calldata: CallData.compile([modelId]),
    });
    return !!(result.result && BigInt(result.result[0] || "0") > 0n);
  } catch {
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Command: verify
// ═══════════════════════════════════════════════════════════════════════
//
// Three account resolution paths:
//   1. STARKNET_PRIVATE_KEY env var → use user's existing account
//   2. ~/.obelysk/starknet/pipeline_account.json → use saved account
//   3. Neither → generate ephemeral keypair, deploy + verify in one TX
//      via PaymasterDetails.deploymentData (true zero-config)

async function cmdVerify(args) {
  const network = args.network || "sepolia";
  const net = NETWORKS[network];
  if (!net) die(`Unknown network: ${network}`);
  if (net.notReady) {
    die(`${network} is not yet configured for paymaster submissions. ` +
        `Deploy the agent factory and account class on ${network} first, ` +
        `then populate NETWORKS.${network} in paymaster_submit.mjs.`);
  }

  const proofPath = args.proof;
  const contract = args.contract;
  const modelIdArg = args["model-id"] || "0x1";

  if (!proofPath) die("--proof is required");
  if (!contract) die("--contract is required");

  // Validate model-id: must be a non-zero hex value.
  {
    const mid = String(modelIdArg).trim();
    if (!mid || mid === "" || mid === "0x" || mid === "0x0" || mid === "0X" || mid === "0X0" || mid === "0") {
      die(
        `Invalid --model-id: "${modelIdArg}"\n` +
          "  Model ID must be a non-zero value (e.g., 0x1, 0x2).\n" +
          "  This typically comes from the contract's registered model index."
      );
    }
    // Basic hex/decimal format check
    if (!/^(0x[0-9a-fA-F]+|[1-9]\d*)$/.test(mid)) {
      die(
        `Invalid --model-id format: "${modelIdArg}"\n` +
          "  Expected a hex string (0x...) or positive decimal integer."
      );
    }
  }

  const provider = getProvider(network);

  // ── Resolve account ──
  let privateKey, accountAddress;
  let needsDeploy = false;
  let ephemeral = null;

  if (process.env.STARKNET_PRIVATE_KEY) {
    // Path 1: User-provided key
    privateKey = process.env.STARKNET_PRIVATE_KEY;
    accountAddress = process.env.STARKNET_ACCOUNT_ADDRESS;
    if (!accountAddress)
      die("STARKNET_ACCOUNT_ADDRESS required when using STARKNET_PRIVATE_KEY");
    info("Using user-provided account");
  } else {
    let config = loadAccountConfig();
    // Validate network match: a sepolia account must not be used on mainnet and vice versa.
    if (config && config.network && config.network !== network) {
      info(`Saved account is for ${config.network}, but current network is ${network}. Generating fresh keypair.`);
      config = null;
    }
    if (config) {
      // Path 2: Saved pipeline account
      privateKey = config.privateKey;
      accountAddress = config.address;
      info(`Using saved pipeline account: ${accountAddress}`);

      // Check if it's actually deployed
      try {
        await provider.getClassHashAt(accountAddress);
      } catch {
        info("Saved account not yet deployed on-chain, will deploy with TX");
        needsDeploy = true;
        ephemeral = {
          publicKey: config.publicKey,
          salt: config.publicKey,
          constructorCalldata: CallData.compile([config.publicKey, "0x0"]),
        };
      }
    } else {
      // Path 3: Zero-config — generate ephemeral keypair
      info("No account found. Generating ephemeral keypair...");
      ephemeral = generateEphemeralAccount(network);
      privateKey = ephemeral.privateKey;
      accountAddress = ephemeral.address;
      needsDeploy = true;

      info(`Ephemeral account: ${accountAddress}`);
      info(
        "Account will be deployed atomically with the proof verification TX"
      );

      // Save for reuse in future runs
      saveAccountConfig({
        address: accountAddress,
        privateKey,
        publicKey: ephemeral.publicKey,
        agentId: null,
        network,
        ephemeral: true,
        createdAt: new Date().toISOString(),
      });
      info(`Keypair saved to ${ACCOUNT_CONFIG_FILE}`);
    }
  }

  const account = getAccount(provider, privateKey, accountAddress, network);

  // ── Read proof file ──
  info(`Reading proof: ${proofPath}`);
  // Guard against OOM: check file size before reading into memory.
  const maxProofSizeMb = parsePositiveIntEnv("OBELYSK_MAX_PROOF_SIZE_MB", 512);
  try {
    const fileStat = statSync(proofPath);
    const sizeMb = fileStat.size / (1024 * 1024);
    if (sizeMb > maxProofSizeMb) {
      die(
        `Proof file is ${sizeMb.toFixed(0)}MB, exceeds ${maxProofSizeMb}MB limit.\n` +
          "  Set OBELYSK_MAX_PROOF_SIZE_MB to increase the limit, or ensure the\n" +
          "  proof was generated correctly (Qwen3-14B 40-layer is typically ~100-200MB)."
      );
    }
  } catch (e) {
    if (e.code === "ENOENT") die(`Proof file not found: ${proofPath}`);
    // statSync failure on existing file — proceed and let readFileSync handle it
  }
  let proofData;
  try {
    proofData = JSON.parse(readFileSync(proofPath, "utf-8"));
  } catch (e) {
    die(`Failed to read proof file: ${e.message}`);
  }

  const verifyPayload = parseVerifyCalldata(proofData, modelIdArg);
  const modelId = verifyPayload.modelId || modelIdArg;

  if (
    args["model-id"] &&
    String(args["model-id"]).toLowerCase() !== String(modelId).toLowerCase()
  ) {
    info(
      `--model-id ${args["model-id"]} differs from proof artifact model_id ${modelId}; using proof artifact value`
    );
  }

  // ── Deploy account if needed ──
  const noPaymaster = args["no-paymaster"] === true || args["no-paymaster"] === "true";
  if (needsDeploy && ephemeral) {
    const rawCalldata = Array.isArray(ephemeral.constructorCalldata)
      ? ephemeral.constructorCalldata
      : CallData.compile(ephemeral.constructorCalldata);
    const deployData = {
      class_hash: net.accountClassHash,
      salt: ephemeral.salt,
      calldata: rawCalldata.map((v) => num.toHex(v)),
      address: accountAddress,
    };
    info("Deploying ephemeral account...");
    await deployAccountDirect(provider, account, deployData, network);
    const config = loadAccountConfig();
    if (config) {
      config.deployedAt = new Date().toISOString();
      config.ephemeral = true;
      saveAccountConfig(config);
    }
    needsDeploy = false;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Chunked Session Flow (schema_version 2)
  // ═══════════════════════════════════════════════════════════════════
  if (verifyPayload.chunked) {
    const e2eStart = Date.now();
    const e2ePhase = (label) => {
      const elapsed = ((Date.now() - e2eStart) / 1000).toFixed(1);
      info(`[E2E] ${label} (${elapsed}s elapsed)`);
    };
    info(`Chunked GKR session mode: ${verifyPayload.totalFelts} felts in ${verifyPayload.numChunks} chunks`);

    await preflightContractEntrypoint(provider, contract, "open_gkr_session");

    // ── Auto-register model if needed ──
    // Check if the model is registered by calling get_model_circuit_hash.
    // If it returns 0x0 or the call fails, register using calldata from the proof file.
    let needsRegistration = false;
    try {
      const regResult = await provider.callContract({
        contractAddress: contract,
        entrypoint: "get_model_circuit_hash",
        calldata: CallData.compile([modelId]),
      });
      const circuitHash = regResult.result ? regResult.result[0] : "0x0";
      try {
        needsRegistration = !circuitHash || circuitHash === "0x0" || BigInt(circuitHash) === 0n;
      } catch {
        needsRegistration = true;
      }
      if (!needsRegistration) {
        info(`Model already registered (circuit_hash: ${circuitHash})`);
      }
    } catch {
      info("Could not check model registration (will attempt registration)");
      needsRegistration = true;
    }

    if (needsRegistration) {
      const registerCalldata = proofData.register_calldata;
      if (!registerCalldata || !Array.isArray(registerCalldata) || registerCalldata.length === 0) {
        die(
          "Model is not registered and proof file does not contain register_calldata.\n" +
          "  Re-generate the proof with the latest prove-model binary to include register_calldata,\n" +
          "  or register the model manually via sncast: sncast invoke --function register_model_gkr ..."
        );
      }
      info(`Registering model (${registerCalldata.length} calldata felts)...`);
      const regCalls = [{
        contractAddress: contract,
        entrypoint: "register_model_gkr",
        calldata: CallData.compile(registerCalldata),
      }];
      try {
        let regTxHash;
        if (noPaymaster) {
          const execResult = await account.execute(regCalls);
          if (!execResult || !execResult.transaction_hash) {
            throw new Error(`account.execute() returned invalid result: ${JSON.stringify(execResult)?.slice(0, 200)}`);
          }
          regTxHash = execResult.transaction_hash;
        } else {
          regTxHash = await executeViaPaymaster(account, regCalls);
        }
        info(`  Registration TX: ${regTxHash}`);
        const regReceipt = await provider.waitForTransaction(regTxHash, { retryInterval: 4000 });
        const regStatus = regReceipt.execution_status ?? regReceipt.status ?? "unknown";
        if (regStatus === "REVERTED") {
          const reason = regReceipt.revert_reason || "unknown";
          // "already registered" or "Only owner" are not fatal — model may be pre-registered by deployer
          if (/already.?registered/i.test(reason)) {
            info("  Model already registered (OK)");
          } else if (/only.?owner|not.?owner|unauthorized|caller.?is.?not/i.test(reason)) {
            info("  Registration reverted (not owner) — model may already be registered by deployer, continuing...");
          } else {
            die(`  Registration reverted: ${reason}`);
          }
        } else {
          info(`  Model registered successfully (status: ${regStatus})`);
        }
      } catch (regErr) {
        const errMsg = truncateRpcError(regErr);
        // Tolerate "already registered" or ownership errors (model pre-registered by deployer)
        if (/already.?registered/i.test(errMsg)) {
          info("  Model already registered (OK)");
        } else if (/only.?owner|not.?owner|unauthorized|caller.?is.?not/i.test(errMsg)) {
          info("  Registration failed (not owner) — model may already be registered by deployer, continuing...");
        } else {
          die(`  Registration failed: ${errMsg}`);
        }
      }
    }

    const verificationCountBefore = await fetchVerificationCount(provider, contract, modelId);
    if (verificationCountBefore !== null) {
      info(`Verification count (before): ${verificationCountBefore.toString()}`);
    }

    // ── Resumability: check for saved session ──
    const sessionDir = join(homedir(), ".obelysk", "chunked_sessions");
    mkdirSync(sessionDir, { recursive: true });

    // Session keyed by contract+modelId (not session_id, which we may not have yet)
    const contractShort = contract.slice(-12).replace(/^0x/i, "");
    const modelShort = String(modelId).slice(-8).replace(/^0x/i, "");
    const resumeFile = join(sessionDir, `resume_${contractShort}_${modelShort}.json`);
    const GKR_SESSION_TIMEOUT_BLOCKS = 10000; // Must match contract constant

    // Compute a content hash of the proof data to detect changes between runs.
    // Uses first 32 bytes of SHA-256 over the flattened chunk data (fast, collision-resistant).
    const proofContentHash = (() => {
      const h = createHash("sha256");
      for (const chunk of verifyPayload.uploadChunks) {
        for (const felt of chunk) h.update(String(felt));
      }
      return h.digest("hex").slice(0, 32);
    })();

    let resumeState = null;
    let resumeIsSealedOnly = false; // True when resuming a sealed (but not yet verified) session
    if (existsSync(resumeFile)) {
      try {
        resumeState = JSON.parse(readFileSync(resumeFile, "utf-8"));
        const isUploadResume = resumeState.status === "uploading" && resumeState.sessionId && resumeState.chunksUploaded > 0;
        const isSealedResume = resumeState.status === "sealed" && resumeState.sessionId;
        if (isUploadResume || isSealedResume) {
          if (isSealedResume) {
            info(`Found sealed session: ${resumeState.sessionId} — resuming from verify step`);
            resumeIsSealedOnly = true;
          } else {
            info(`Found resumable session: ${resumeState.sessionId} (${resumeState.chunksUploaded}/${resumeState.numChunks} chunks uploaded)`);
          }
          // Verify proof content matches (catches different weights with same architecture)
          if (!isSealedResume && resumeState.proofContentHash && resumeState.proofContentHash !== proofContentHash) {
            info(`  Proof content changed since last run (hash: ${resumeState.proofContentHash} → ${proofContentHash})`);
            info(`  Starting fresh session (old session will be orphaned on-chain)`);
            resumeState = null;
          }
          // Verify chunk count matches current proof (upload resume only)
          if (!isSealedResume && resumeState && (resumeState.numChunks !== verifyPayload.numChunks || resumeState.totalFelts !== verifyPayload.totalFelts)) {
            info(`  Proof changed since last run (chunks: ${resumeState.numChunks}→${verifyPayload.numChunks}, felts: ${resumeState.totalFelts}→${verifyPayload.totalFelts})`);
            info(`  Starting fresh session (old session will be orphaned on-chain)`);
            resumeState = null;
          }
          // Check session TTL: contract expires sessions after GKR_SESSION_TIMEOUT_BLOCKS
          if (resumeState && resumeState.createdAt_block) {
            try {
              const currentBlock = await provider.getBlockNumber();
              const blocksRemaining = (resumeState.createdAt_block + GKR_SESSION_TIMEOUT_BLOCKS) - currentBlock;
              if (blocksRemaining <= 0) {
                info(`  Session expired on-chain (created at block ${resumeState.createdAt_block}, now ${currentBlock}, TTL=${GKR_SESSION_TIMEOUT_BLOCKS})`);
                info(`  Starting fresh session`);
                resumeState = null;
                resumeIsSealedOnly = false;
              } else if (blocksRemaining < 500) {
                info(`  WARNING: Session nearing expiry (${blocksRemaining} blocks remaining, ~${Math.round(blocksRemaining * 6 / 60)} min)`);
              }
            } catch {
              info("  WARNING: Cannot verify session TTL (getBlockNumber unavailable). If session expired, contract will reject with GKR_SESSION_EXPIRED.");
            }
          }
        } else {
          resumeState = null;
        }
      } catch {
        resumeState = null;
      }
    }

    // Helper: execute a single call
    async function waitWithTimeout(txHash, timeoutMs = 300000) {
      let timer;
      try {
        return await Promise.race([
          provider.waitForTransaction(txHash, { retryInterval: 4000 }),
          new Promise((_, reject) => {
            timer = setTimeout(() => reject(new Error(`TX ${txHash} not confirmed after ${timeoutMs/1000}s`)), timeoutMs);
          }),
        ]);
      } finally {
        // Always clear the timeout to prevent timer leaks when waitForTransaction wins the race.
        if (timer) clearTimeout(timer);
      }
    }

    // Cached paymaster fee estimate from the first successful chunk.
    // Reused for all subsequent chunks to avoid nonce-stale estimateFee failures.
    // For paymaster mode: caches the `suggested_max_fee_in_gas_token` value.
    // For non-paymaster mode: not used (account.execute() handles fees).
    let cachedPaymasterMaxFee = null;
    const STRK_TOKEN_ADDR = "0x04718f5a0fc34cc1af16a1cdee98ffb20c31f5cd61d6ab07201858f4287c938d";

    async function execCall(entrypoint, calldata, label, opts = {}) {
      const calls = [{
        contractAddress: contract,
        entrypoint,
        calldata: CallData.compile(calldata),
      }];
      info(`  ${label}...`);
      let txHash;
      if (noPaymaster) {
        try {
          // Build resource bounds — either from fee estimation or conservative fallback.
          // All values must be BigInt to avoid "Cannot mix BigInt and other types" errors.
          let rb;
          try {
            const est = await account.estimateInvokeFee(calls);
            if (est.resourceBounds) {
              const e = est.resourceBounds;
              rb = {
                l1_gas: { max_amount: BigInt(e.l1_gas.max_amount), max_price_per_unit: BigInt(e.l1_gas.max_price_per_unit) },
                l2_gas: { max_amount: BigInt(e.l2_gas.max_amount), max_price_per_unit: BigInt(e.l2_gas.max_price_per_unit) },
                l1_data_gas: { max_amount: BigInt(e.l1_data_gas.max_amount), max_price_per_unit: BigInt(e.l1_data_gas.max_price_per_unit) },
              };
            }
          } catch (estErr) {
            info(`  Fee estimation failed (${(estErr.message || "").slice(0, 120)}), using fallback bounds`);
          }

          if (!rb) {
            // Conservative fallback — fits in ~8 STRK total
            rb = {
              l1_gas: { max_amount: 100n, max_price_per_unit: 100000000000000n },         // ~0.01 STRK
              l2_gas: { max_amount: 500000000n, max_price_per_unit: 12000000000n },        // ~6 STRK
              l1_data_gas: { max_amount: 5000n, max_price_per_unit: 300000000000000n },    // ~1.5 STRK
            };
          }

          // Cap total bounds to 80% of account balance to leave room for subsequent TXs.
          const bal = await provider.callContract({
            contractAddress: STRK_TOKEN_ADDR,
            entrypoint: "balanceOf",
            calldata: CallData.compile([account.address]),
          });
          const balance = BigInt(bal[0]);
          const maxSpend = balance * 80n / 100n;
          const totalBounds = rb.l1_gas.max_amount * rb.l1_gas.max_price_per_unit +
            rb.l2_gas.max_amount * rb.l2_gas.max_price_per_unit +
            rb.l1_data_gas.max_amount * rb.l1_data_gas.max_price_per_unit;

          if (totalBounds > maxSpend && maxSpend > 0n) {
            // Scale down all prices proportionally to fit within balance
            const scale_num = maxSpend;
            const scale_den = totalBounds;
            rb.l1_gas.max_price_per_unit = rb.l1_gas.max_price_per_unit * scale_num / scale_den;
            rb.l2_gas.max_price_per_unit = rb.l2_gas.max_price_per_unit * scale_num / scale_den;
            rb.l1_data_gas.max_price_per_unit = rb.l1_data_gas.max_price_per_unit * scale_num / scale_den;
            const capped = rb.l1_gas.max_amount * rb.l1_gas.max_price_per_unit +
              rb.l2_gas.max_amount * rb.l2_gas.max_price_per_unit +
              rb.l1_data_gas.max_amount * rb.l1_data_gas.max_price_per_unit;
            info(`  Bounds capped: ${Number(totalBounds / 10n**15n) / 1000} → ${Number(capped / 10n**15n) / 1000} STRK (bal: ${Number(balance / 10n**15n) / 1000})`);
          }

          const finalTotal = rb.l1_gas.max_amount * rb.l1_gas.max_price_per_unit +
            rb.l2_gas.max_amount * rb.l2_gas.max_price_per_unit +
            rb.l1_data_gas.max_amount * rb.l1_data_gas.max_price_per_unit;
          info(`  Max fee: ~${Number(finalTotal / 10n**15n) / 1000} STRK`);

          const result = await account.execute(calls, { resourceBounds: rb });
          if (!result || !result.transaction_hash) {
            throw new Error(`account.execute() returned invalid result: ${JSON.stringify(result)?.slice(0, 200)}`);
          }
          txHash = result.transaction_hash;
        } catch (execErr) {
          info(`  execute() failed: ${truncateRpcError(execErr)}`);
          throw execErr;
        }
      } else if (opts.cachedMaxFee) {
        // Skip estimation — reuse cached fee from a previous successful chunk.
        // This avoids nonce-stale errors during rapid sequential chunk uploads.
        const feeDetails = { feeMode: { mode: "sponsored" } };
        const result = await account.executePaymasterTransaction(
          Array.isArray(calls) ? calls : [calls],
          feeDetails,
          opts.cachedMaxFee
        );
        if (!result || typeof result !== "object" || !result.transaction_hash) {
          throw new Error(`executePaymasterTransaction returned invalid result: ${JSON.stringify(result)?.slice(0, 200)}`);
        }
        txHash = result.transaction_hash;
      } else {
        txHash = await executeViaPaymaster(account, calls);
      }
      // Validate txHash before proceeding — catch undefined/null from broken RPC responses.
      if (!txHash || typeof txHash !== "string" || !/^0x[0-9a-fA-F]+$/i.test(txHash)) {
        throw new Error(`${label}: received invalid txHash: ${JSON.stringify(txHash)}`);
      }
      info(`  TX: ${txHash}`);
      const receipt = await waitWithTimeout(txHash);
      const execStatus = receipt.execution_status ?? receipt.finality_status ?? receipt.status ?? "unknown";
      if (execStatus === "REVERTED") {
        const reason = receipt.revert_reason || "unknown";
        // Nonce-related reverts are retryable — throw instead of die() so outer retry loops can handle them.
        if (/nonce|desynchroni|already used|too (high|old)/i.test(reason)) {
          throw new Error(`${label} reverted (retryable nonce error): ${reason}`);
        }
        die(`  ${label} reverted: ${reason}`);
      }
      if (execStatus !== "SUCCEEDED" && execStatus !== "ACCEPTED_ON_L2") {
        // TX was RECEIVED but never confirmed, or unknown status.
        // This indicates the RPC mempool dropped the TX (common with large calldata).
        const msg = `${label} TX status: ${execStatus} (expected SUCCEEDED or ACCEPTED_ON_L2)`;
        if (execStatus === "RECEIVED" || execStatus === "PENDING") {
          // TX was accepted by RPC but never included in a block — likely mempool drop.
          throw new Error(`${msg}. TX likely dropped by RPC mempool. Retry with smaller chunks (OBELYSK_CHUNK_SIZE).`);
        }
        throw new Error(msg);
      }
      return { txHash, receipt };
    }

    // ── Step 1: open_gkr_session (or resume) ──
    e2ePhase("Opening GKR session");
    let sessionId = null;
    let sessionState = null;
    let sessionFile = null;
    let resumeFromChunk = 0;

    if (resumeState) {
      // Resume existing session (uploading or sealed)
      sessionId = resumeState.sessionId;
      sessionState = resumeState;
      sessionFile = resumeFile;
      resumeFromChunk = resumeIsSealedOnly ? verifyPayload.numChunks : resumeState.chunksUploaded;
      if (resumeIsSealedOnly) {
        info(`  Resuming sealed session ${sessionId} — skipping to verify`);
      } else {
        info(`  Resuming session ${sessionId} from chunk ${resumeFromChunk}`);
      }
    } else {
      // Open new session
      const { txHash: openTxHash, receipt: openReceipt } = await execCall(
        "open_gkr_session",
        [
          modelId,
          String(verifyPayload.totalFelts),
          String(verifyPayload.circuitDepth),
          String(verifyPayload.numLayers),
          String(verifyPayload.weightBindingMode),
          verifyPayload.packed ? "1" : "0",
          verifyPayload.ioPacked ? "1" : "0",
        ],
        "open_gkr_session"
      );

      // Parse session_id from events.
      // Look for GkrSessionOpened event: key[0]=selector, key[1]=session_id.
      // CRITICAL: Only accept events emitted by our contract (from_address match).
      const events = openReceipt.events ?? [];
      const contractNorm = contract.toLowerCase().replace(/^0x0*/, "0x");
      for (const ev of events) {
        const evFrom = (ev.from_address || "").toLowerCase().replace(/^0x0*/, "0x");
        if (evFrom !== contractNorm) continue; // Skip events from other contracts
        if (ev.keys && ev.keys.length >= 2) {
          const candidateId = ev.keys[1];
          if (candidateId && candidateId !== "0x0") {
            try { if (BigInt(candidateId) !== 0n) { sessionId = candidateId; break; } } catch { /* skip malformed */ }
          }
        }
      }
      if (!sessionId) {
        // Fallback: parse from data[0] — some contract versions emit session_id in data.
        // Still only accept events from our contract.
        for (const ev of events) {
          const evFrom = (ev.from_address || "").toLowerCase().replace(/^0x0*/, "0x");
          if (evFrom !== contractNorm) continue;
          if (ev.data && ev.data.length >= 1 && ev.data[0] !== "0x0") {
            try { if (BigInt(ev.data[0]) !== 0n) { sessionId = ev.data[0]; break; } } catch { /* skip malformed */ }
          }
        }
      }
      if (!sessionId) {
        die("Could not parse session_id from open_gkr_session events. " +
            `Receipt had ${events.length} events. ` +
            "Ensure the contract emits GkrSessionOpened with session_id in keys[1] or data[0].");
      }
      info(`  Session ID: ${sessionId}`);

      // Save session state for resumability — persist IMMEDIATELY to prevent
      // duplicate sessions if the script crashes before the first chunk upload.
      sessionFile = resumeFile;
      const openBlockNumber = (() => {
        if (!openReceipt.block_number) return null;
        const n = Number(openReceipt.block_number);
        return Number.isFinite(n) && n >= 0 ? n : null;
      })();
      sessionState = {
        sessionId,
        modelId,
        contract,
        totalFelts: verifyPayload.totalFelts,
        numChunks: verifyPayload.numChunks,
        chunksUploaded: 0,
        txHashes: [openTxHash],
        status: "uploading",
        createdAt: new Date().toISOString(),
        createdAt_block: openBlockNumber,
        proofContentHash,
      };
      safeWriteJson(sessionFile, sessionState);
    }

    // ── Step 2: upload_gkr_chunk × N ──
    // Pre-upload integrity check: verify chunk felt counts sum to total_felts.
    // Catches data corruption from re-chunking, truncated JSON, or version mismatches.
    {
      const chunkFeltSum = verifyPayload.uploadChunks.reduce((sum, c) => sum + c.length, 0);
      if (chunkFeltSum !== verifyPayload.totalFelts) {
        die(
          `Chunk data integrity check failed: sum of chunk lengths (${chunkFeltSum}) !== total_felts (${verifyPayload.totalFelts}).\n` +
          `  This indicates data corruption in the proof file or a chunking bug.`
        );
      }
      info(`Chunk integrity OK: ${chunkFeltSum} felts across ${verifyPayload.numChunks} chunks`);
    }

    e2ePhase("Uploading chunks");
    // First chunk uses normal fee estimation (may need retries for nonce sync).
    // Subsequent chunks reuse cached paymaster max fee with 1.5x safety margin,
    // skipping estimateFee entirely to avoid nonce-stale errors on public RPCs.
    const MAX_RETRIES = 5;
    // Default inter-chunk delay reduced from 3s to 1.5s.
    // Inter-chunk delay: 500ms default is sufficient — Starknet blocks arrive ~6s
    // apart, and the retry loop handles nonce errors if we submit too fast.
    // Override: OBELYSK_CHUNK_DELAY_MS=1500 for conservative public RPCs.
    const INTER_CHUNK_DELAY_MS = (() => {
      const raw = process.env.OBELYSK_CHUNK_DELAY_MS;
      if (raw === undefined || raw === null || String(raw).trim() === "") return 500;
      const n = parseInt(raw, 10);
      if (!Number.isFinite(n) || n < 0) {
        die(`OBELYSK_CHUNK_DELAY_MS must be a non-negative integer (got: "${raw}")`);
      }
      return n;
    })();
    const uploadStartTime = Date.now();
    if (resumeFromChunk > 0) {
      info(`Skipping ${resumeFromChunk} already-uploaded chunks...`);
    }
    // Re-estimate paymaster fee every N chunks. Default 30 is conservative:
    // gas prices rarely shift enough in ~3min to cause INSUFFICIENT errors,
    // and the INSUFFICIENT handler already clears the cache reactively.
    const FEE_RE_ESTIMATE_INTERVAL = parsePositiveIntEnv("OBELYSK_FEE_RE_ESTIMATE_INTERVAL", 30);
    for (let i = resumeFromChunk; i < verifyPayload.numChunks; i++) {
      const chunk = verifyPayload.uploadChunks[i];
      const pct = ((i / verifyPayload.numChunks) * 100).toFixed(1);
      // ETA based on average upload time so far
      const chunksUploaded = i - resumeFromChunk;
      let etaStr = "";
      if (chunksUploaded > 0) {
        const avgMs = (Date.now() - uploadStartTime) / chunksUploaded;
        const remainMs = avgMs * (verifyPayload.numChunks - i);
        const remainSec = Math.round(remainMs / 1000);
        etaStr = `, ETA ${Math.floor(remainSec / 60)}m ${remainSec % 60}s`;
      }
      // Periodically re-estimate fee to handle gas price spikes during long uploads
      if (cachedPaymasterMaxFee && chunksUploaded > 0 && chunksUploaded % FEE_RE_ESTIMATE_INTERVAL === 0) {
        info(`  Re-estimating paymaster fee (every ${FEE_RE_ESTIMATE_INTERVAL} chunks)...`);
        cachedPaymasterMaxFee = null;
      }
      for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        // Rebuild execOpts on every attempt so cache clears take effect immediately.
        const execOpts = cachedPaymasterMaxFee ? { cachedMaxFee: cachedPaymasterMaxFee } : {};
        try {
          const { txHash: chunkTxHash, receipt: chunkReceipt } = await execCall(
            "upload_gkr_chunk",
            [sessionId, String(i), String(chunk.length), ...chunk],
            `upload_gkr_chunk[${i + 1}/${verifyPayload.numChunks}] (${chunk.length} felts, ${pct}%${etaStr})`,
            execOpts
          );
          // Cache the paymaster max fee from the first successful chunk.
          // Subsequent chunks reuse this fee, skipping estimateFee entirely
          // to avoid nonce-stale errors on public RPCs during rapid uploads.
          if (!cachedPaymasterMaxFee && !noPaymaster) {
            // Re-estimate once after first chunk succeeds to get a valid fee baseline.
            // The first chunk went through executeViaPaymaster (full estimation),
            // so we estimate the same call shape and cache the result.
            try {
              const callsArray = [{
                contractAddress: contract,
                entrypoint: "upload_gkr_chunk",
                calldata: CallData.compile([sessionId, String(0), String(chunk.length), ...chunk]),
              }];
              const feeDetails = { feeMode: { mode: "sponsored" } };
              const estimation = await account.estimatePaymasterTransactionFee(callsArray, feeDetails);
              // Apply 2x safety margin on the estimated fee to handle gas price fluctuations.
              const feeStr = estimation?.suggested_max_fee_in_gas_token;
              if (!feeStr) throw new Error("Fee estimation missing suggested_max_fee_in_gas_token");
              const fee = BigInt(feeStr);
              cachedPaymasterMaxFee = "0x" + (fee * 2n).toString(16);
              info(`  Cached paymaster max fee: ${cachedPaymasterMaxFee} (2x margin)`);
            } catch (feeErr) {
              info(`  Could not cache paymaster fee: ${truncateRpcError(feeErr)}`);
              // Will continue with full estimation on each chunk (slower but safe)
            }
          }
          sessionState.chunksUploaded = i + 1;
          sessionState.txHashes.push(chunkTxHash);
          safeWriteJson(sessionFile, sessionState);
          break;
        } catch (e) {
          const errMsg = truncateRpcError(e);
          info(`  Chunk ${i} attempt ${attempt + 1} failed: ${errMsg}`);
          // If cached fee caused the failure, clear it and retry with full estimation.
          if (cachedPaymasterMaxFee && (errMsg.includes("INSUFFICIENT") || errMsg.includes("insufficient"))) {
            info(`  Clearing cached paymaster fee, will re-estimate...`);
            cachedPaymasterMaxFee = null;
          }
          if (attempt < MAX_RETRIES - 1) {
            // Rate-limited responses (429, "too many") need much longer backoff.
            const isRateLimit = /429|rate.?limit|too many|throttl/i.test(errMsg);
            const backoffMs = isRateLimit
              ? Math.min((attempt + 1) * 30000, 60000)
              : Math.min((attempt + 1) * 5000, 20000);
            if (isRateLimit) info(`  Rate limited — using extended backoff`);
            info(`  Retrying in ${backoffMs / 1000}s...`);
            await new Promise((r) => setTimeout(r, backoffMs));
          } else {
            die(
              `Failed to upload chunk ${i} after ${MAX_RETRIES} attempts.\n` +
              `  Session ${sessionId} is still open on-chain. You can:\n` +
              `  1. Re-run the script to auto-resume from chunk ${i}\n` +
              `  2. Delete ${resumeFile} to start a fresh session`
            );
          }
        }
      }
      // Inter-chunk delay: combines minimum sleep with lightweight nonce check.
      // The retry loop handles nonce errors, so this is best-effort to reduce retries.
      if (i < verifyPayload.numChunks - 1 && INTER_CHUNK_DELAY_MS > 0) {
        await new Promise((r) => setTimeout(r, INTER_CHUNK_DELAY_MS));
        // Quick nonce check — if stale, wait a bit more before next chunk
        try {
          const nonce = BigInt(await provider.getNonceForAddress(account.address, "latest"));
          // We've submitted (i+1) chunks + open_session TX = (i+2) total TXs from this account.
          // If the RPC nonce lags significantly, add a short extra wait.
          if (nonce < BigInt(i + 1)) {
            await new Promise((r) => setTimeout(r, 2000));
          }
        } catch { /* proceed — retry loop handles failures */ }
      }
    }
    const uploadDuration = ((Date.now() - uploadStartTime) / 1000).toFixed(1);
    info(`All ${verifyPayload.numChunks} chunks uploaded in ${uploadDuration}s.`);
    e2ePhase(`Chunks uploaded (${uploadDuration}s for ${verifyPayload.numChunks} chunks)`);

    // Helper: wait for on-chain nonce to reach or exceed a target value.
    // Polls the RPC every 3s until nonce >= target or timeout.
    async function waitForNonce(targetNonceStr, maxWaitMs = 90000) {
      let target;
      try { target = BigInt(String(targetNonceStr).trim()); }
      catch { throw new Error(`Invalid nonce target: ${targetNonceStr}`); }
      const start = Date.now();
      let lastNonce = 0n;
      while (Date.now() - start < maxWaitMs) {
        try {
          const nonceResult = await provider.getNonceForAddress(account.address, "latest");
          let current;
          try { current = BigInt(nonceResult); }
          catch { throw new Error(`Invalid nonce from RPC: ${JSON.stringify(nonceResult)?.slice(0, 100)}`); }
          if (current >= target) {
            info(`  Nonce synced: ${current} (target was ${target})`);
            return true;
          }
          if (current !== lastNonce) {
            info(`  Nonce progress: ${current} → target ${target}`);
            lastNonce = current;
          }
        } catch {
          // getNonceForAddress may fail on some RPCs; fall back to time-based wait
        }
        // 5s aligns with Starknet block time (~6s). Polling faster wastes RPC calls.
        await new Promise((r) => setTimeout(r, 5000));
      }
      info(`  Nonce wait timed out after ${maxWaitMs / 1000}s`);
      return false;
    }

    // Helper: retry execCall for nonce-stale estimateFee errors.
    async function execCallWithRetry(entrypoint, calldata, label, opts = {}, maxRetries = 10) {
      let consecutiveNonceErrors = 0;
      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          const result = await execCall(entrypoint, calldata, label, opts);
          consecutiveNonceErrors = 0;
          return result;
        } catch (e) {
          const fullMsg = e.message || String(e);
          const isNonceError = /nonce|desynchroni|already used|too (high|old)|nonce mismatch/i.test(fullMsg);
          const isDroppedTx = /dropped|mempool/i.test(fullMsg);
          // Permanent errors: don't waste retries on unrecoverable failures
          const isPermanent = /class.?not.?found|not.?deployed|entry.?point.?not.?found|invalid.?txHash/i.test(fullMsg);
          if (isPermanent) {
            throw new Error(`Permanent error (not retrying): ${fullMsg}`);
          }
          // Circuit breaker: 3 consecutive nonce errors likely means unrecoverable state
          if (isNonceError) {
            consecutiveNonceErrors++;
            if (consecutiveNonceErrors > 3) {
              throw new Error(`Nonce error persisted after ${consecutiveNonceErrors} consecutive attempts: ${fullMsg}`);
            }
          } else {
            consecutiveNonceErrors = 0;
          }
          const msg = truncateRpcError(e);
          info(`  ${label} attempt ${attempt + 1} failed: ${msg}`);
          if (attempt < maxRetries - 1) {
            if (isNonceError) {
              // Active nonce polling instead of blind sleep
              info(`  Nonce error detected — polling for nonce sync...`);
              try {
                const currentNonce = await provider.getNonceForAddress(account.address, "latest");
                // Wait for nonce to advance past current value
                const waitTarget = String(BigInt(currentNonce) + 1n);
                await waitForNonce(waitTarget, 60000);
              } catch {
                // Fallback to time-based wait
                const backoffMs = Math.min((attempt + 1) * 10000, 60000);
                info(`  Nonce poll unavailable, waiting ${backoffMs / 1000}s...`);
                await new Promise((r) => setTimeout(r, backoffMs));
              }
            } else {
              const backoffMs = isDroppedTx
                ? Math.min((attempt + 1) * 8000, 30000)
                : Math.min((attempt + 1) * 5000, 20000);
              info(`  Retrying in ${backoffMs / 1000}s...`);
              await new Promise((r) => setTimeout(r, backoffMs));
            }
          } else {
            throw e;
          }
        }
      }
    }

    // Skip nonce wait + seal if resuming a sealed session.
    if (!resumeIsSealedOnly) {
    // Each chunk's execCall already calls waitWithTimeout, confirming block inclusion.
    // A brief settle is sufficient before sealing — no 90s nonce poll needed.
    info("All chunks confirmed. Brief settle before seal...");
    await new Promise((r) => setTimeout(r, 2000));

    // ── Step 3: seal_gkr_session ──
    e2ePhase("Sealing session");
    const { txHash: sealTxHash } = await execCallWithRetry(
      "seal_gkr_session",
      [sessionId],
      "seal_gkr_session"
    );
    sessionState.status = "sealed";
    sessionState.txHashes.push(sealTxHash);
    safeWriteJson(sessionFile, sessionState);

    // Seal TX is already block-confirmed by waitWithTimeout inside execCallWithRetry.
    // A short settle delay is sufficient for RPC state propagation before verify.
    info("Seal TX confirmed. Brief settle before verify...");
    await new Promise((r) => setTimeout(r, 2000));
    } // end !resumeIsSealedOnly

    // ── Step 4: verify_gkr_from_session ──
    // After seal, the session cannot be resumed (sealed sessions are immutable).
    // Wrap in try/catch to ensure resume file cleanup even on verify failure.
    let verifyTxHash;
    try {
      e2ePhase("Verifying session");
      const result = await execCallWithRetry(
        "verify_gkr_from_session",
        [sessionId],
        "verify_gkr_from_session"
      );
      verifyTxHash = result.txHash;
      sessionState.status = "verified";
      sessionState.txHashes.push(verifyTxHash);
      safeWriteJson(sessionFile, sessionState);
    } catch (verifyErr) {
      // Session is sealed — keep resume file so next run can retry verify.
      // Only delete on permanent (non-retryable) errors.
      const verifyErrMsg = truncateRpcError(verifyErr);
      const isPermanentVerifyErr = /entry.?point.?not.?found|class.?not.?found|not.?deployed|session.?not.?found|invalid.?session/i.test(verifyErrMsg);
      if (isPermanentVerifyErr) {
        try { if (existsSync(resumeFile)) unlinkSync(resumeFile); } catch { /* ignore */ }
        die(`verify_gkr_from_session permanently failed: ${verifyErrMsg}`);
      }
      // Transient error: keep resume file with "sealed" status for retry on next run
      info(`verify_gkr_from_session failed (transient): ${verifyErrMsg}`);
      info(`  Session ${sessionId} is sealed. Re-run the script to retry verification.`);
      info(`  Resume file: ${resumeFile}`);
      die(`verify_gkr_from_session failed after seal: ${verifyErrMsg}`);
    }

    // ── Check verification result ──
    // Retry verification count query up to 3 times (RPC may lag behind L2 state)
    // Retry verification count query with exponential backoff.
    // Block propagation on Starknet can take 10-60s after TX confirmation,
    // especially on congested blocks or when using public RPCs.
    let verificationCountAfter = null;
    const VC_MAX_ATTEMPTS = 8;
    for (let vcAttempt = 0; vcAttempt < VC_MAX_ATTEMPTS; vcAttempt++) {
      verificationCountAfter = await fetchVerificationCount(provider, contract, modelId);
      if (verificationCountAfter !== null) break;
      const waitMs = Math.min(3000 * Math.pow(1.5, vcAttempt), 15000);
      info(`  Verification count query attempt ${vcAttempt + 1}/${VC_MAX_ATTEMPTS} failed, retrying in ${(waitMs / 1000).toFixed(0)}s...`);
      await new Promise((r) => setTimeout(r, waitMs));
    }

    let verificationCountDelta = null;
    let acceptedOnchain = false;
    let acceptanceEvidence = "unknown";
    if (verificationCountAfter !== null && verificationCountBefore !== null) {
      verificationCountDelta = verificationCountAfter - verificationCountBefore;
      acceptedOnchain = verificationCountDelta > 0n;
      acceptanceEvidence = "verification_count_delta";
    } else if (verificationCountAfter !== null) {
      acceptedOnchain = verificationCountAfter > 0n;
      acceptanceEvidence = "verification_count_observed";
    } else {
      // Last resort: try is_proof_verified as fallback
      const proofVerifiedFlag = await fetchProofVerifiedFlag(provider, contract, modelId);
      if (proofVerifiedFlag !== null) {
        acceptedOnchain = proofVerifiedFlag;
        acceptanceEvidence = "is_proof_verified_fallback";
      } else {
        // Cannot confirm — DO NOT assume success
        acceptedOnchain = false;
        acceptanceEvidence = "unconfirmed_rpc_unavailable";
        info("WARNING: Could not confirm on-chain acceptance. RPC view methods unavailable.");
        info("  The verify_gkr_from_session TX succeeded, but acceptance is unconfirmed.");
        info("  Check manually: is_proof_verified(" + modelId + ") on " + contract);
      }
    }

    // Clean up resume file on completion (success or confirmed failure)
    try { if (existsSync(resumeFile)) unlinkSync(resumeFile); } catch { /* ignore */ }

    const totalTxs = sessionState.txHashes.length;
    const e2eTotalSec = ((Date.now() - e2eStart) / 1000).toFixed(1);
    const e2eMinutes = Math.floor(e2eTotalSec / 60);
    const e2eRemainSec = Math.round(e2eTotalSec % 60);
    const explorerUrl = `${net.explorer}${verifyTxHash}`;

    e2ePhase("Complete");
    info(`Session complete: ${totalTxs} TXs in ${e2eTotalSec}s (${e2eMinutes}m ${e2eRemainSec}s)`);
    info(`Explorer (verify): ${explorerUrl}`);
    info(`Accepted on-chain: ${acceptedOnchain}`);

    jsonOutput({
      txHash: verifyTxHash,
      explorerUrl,
      isVerified: acceptedOnchain,
      acceptedOnchain,
      fullGkrVerified: acceptedOnchain,
      hasAnyVerification: acceptedOnchain,
      verificationCountBefore:
        verificationCountBefore === null ? null : verificationCountBefore.toString(),
      verificationCountAfter:
        verificationCountAfter === null ? null : verificationCountAfter.toString(),
      verificationCountDelta:
        verificationCountDelta === null ? null : verificationCountDelta.toString(),
      acceptanceEvidence,
      gasSponsored: !noPaymaster,
      accountDeployed: false,
      executionStatus: acceptedOnchain ? "ACCEPTED_ON_L2" : "UNCONFIRMED",
      entrypoint: "verify_gkr_from_session",
      onchainAssurance: acceptedOnchain ? "full_gkr" : "unconfirmed",
      sessionId,
      uploadedChunks: verifyPayload.numChunks,
      totalTxs,
      allTxHashes: sessionState.txHashes,
      e2eDurationSeconds: parseFloat(e2eTotalSec),
    });
    return;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Single-TX Flow (schema_version 1)
  // ═══════════════════════════════════════════════════════════════════
  const singleTxStart = Date.now();
  if (
    !new Set([
      "verify_model_gkr", "verify_model_gkr_v2", "verify_model_gkr_v3", "verify_model_gkr_v4",
      "verify_model_gkr_v4_packed", "verify_model_gkr_v4_packed_io",
    ]).has(verifyPayload.entrypoint)
  ) {
    die(
      `Unsupported entrypoint in hardened pipeline (got: ${verifyPayload.entrypoint})`
    );
  }

  await preflightContractEntrypoint(provider, contract, verifyPayload.entrypoint);

  // ── Auto-register model if needed (single-TX path) ──
  {
    let needsReg = false;
    try {
      const rr = await provider.callContract({
        contractAddress: contract,
        entrypoint: "get_model_circuit_hash",
        calldata: CallData.compile([modelId]),
      });
      const ch = rr.result ? rr.result[0] : "0x0";
      try {
        needsReg = !ch || ch === "0x0" || BigInt(ch) === 0n;
      } catch {
        needsReg = true;
      }
      if (!needsReg) info(`Model already registered (circuit_hash: ${ch})`);
    } catch {
      info("Could not check model registration (will attempt registration)");
      needsReg = true;
    }
    if (needsReg) {
      const rc = proofData.register_calldata;
      if (!rc || !Array.isArray(rc) || rc.length === 0) {
        die(
          "Model is not registered and proof file does not contain register_calldata.\n" +
          "  Re-generate the proof with the latest prove-model binary, or register manually."
        );
      }
      info(`Registering model (${rc.length} calldata felts)...`);
      const regCalls = [{
        contractAddress: contract,
        entrypoint: "register_model_gkr",
        calldata: CallData.compile(rc),
      }];
      try {
        let rTx;
        if (noPaymaster) {
          const execResult = await account.execute(regCalls);
          if (!execResult || !execResult.transaction_hash) {
            throw new Error(`account.execute() returned invalid result: ${JSON.stringify(execResult)?.slice(0, 200)}`);
          }
          rTx = execResult.transaction_hash;
        } else {
          rTx = await executeViaPaymaster(account, regCalls);
        }
        info(`  Registration TX: ${rTx}`);
        const rReceipt = await waitWithTimeout(rTx, 120000);
        const rStatus = rReceipt.execution_status ?? rReceipt.status ?? "unknown";
        if (rStatus === "REVERTED") {
          const reason = rReceipt.revert_reason || "unknown";
          if (/already.?registered/i.test(reason)) {
            info("  Model already registered (OK)");
          } else if (/only.?owner|not.?owner|unauthorized|caller.?is.?not/i.test(reason)) {
            info("  Registration reverted (not owner) — model may already be registered by deployer, continuing...");
          } else {
            die(`  Registration reverted: ${reason}`);
          }
        } else {
          info(`  Model registered (status: ${rStatus})`);
        }
      } catch (regErr) {
        const errMsg = truncateRpcError(regErr);
        if (/already.?registered/i.test(errMsg)) {
          info("  Model already registered (OK)");
        } else if (/only.?owner|not.?owner|unauthorized|caller.?is.?not/i.test(errMsg)) {
          info("  Registration failed (not owner) — model may already be registered by deployer, continuing...");
        } else {
          die(`  Registration failed: ${errMsg}`);
        }
      }
    }
  }

  const verificationCountBefore = await fetchVerificationCount(
    provider,
    contract,
    modelId
  );
  if (verificationCountBefore !== null) {
    info(`Verification count (before): ${verificationCountBefore.toString()}`);
  }

  // ── Build verification call (GKR only) ──
  const calls = [
    {
      contractAddress: contract,
      entrypoint: verifyPayload.entrypoint,
      calldata: CallData.compile(verifyPayload.calldata),
    },
  ];

  info(`Submitting ${verifyPayload.entrypoint} for model ${modelId}...`);
  info(`Contract: ${contract}`);
  info(`Calldata elements: ${verifyPayload.calldata.length}`);

  // ── Execute with retry (matches chunked path reliability) ──
  let txHash;
  let gasSponsored = !noPaymaster;
  const SINGLE_TX_MAX_RETRIES = 5;
  let consecutiveNonceErrors = 0;

  for (let attempt = 0; attempt < SINGLE_TX_MAX_RETRIES; attempt++) {
    try {
      if (noPaymaster) {
        if (attempt === 0) info("Submitting directly (no paymaster)...");
        const result = await account.execute(calls);
        if (!result || !result.transaction_hash) {
          throw new Error(`account.execute() returned invalid result: ${JSON.stringify(result)?.slice(0, 200)}`);
        }
        txHash = result.transaction_hash;
      } else {
        txHash = await executeViaPaymaster(account, calls);
      }
      break; // success
    } catch (e) {
      const msg = truncateRpcError(e);
      // Nonce circuit breaker: 2 consecutive nonce errors → account state is broken
      const isNonceError = /nonce|desynchroni|already used|too (high|old)|nonce mismatch/i.test(msg);
      if (isNonceError) {
        consecutiveNonceErrors++;
        if (consecutiveNonceErrors >= 2) {
          die(
            `Nonce error persisted after ${consecutiveNonceErrors} attempts: ${msg}\n` +
              "  The account nonce is desynchronized. This can happen when:\n" +
              "  - A previous TX is stuck in the mempool\n" +
              "  - Another process submitted TXs from the same account\n" +
              "  Wait a few minutes and retry, or use a fresh ephemeral account."
          );
        }
      } else {
        consecutiveNonceErrors = 0;
      }
      // Permanent errors: don't retry
      const isPermanent = /not.?eligible|not.?supported|SNIP-9|class.?not.?found|entry.?point.?not.?found|insufficient.?balance|balance.?too.?low|insufficient.?funds/i.test(msg);
      if (isPermanent) {
        if (msg.includes("not eligible") || msg.includes("not supported") || msg.includes("SNIP-9")) {
          die(
            `Paymaster rejected transaction: ${msg}\n` +
              "This may mean:\n" +
              "  - Account is not SNIP-9 compatible (needed for paymaster)\n" +
              "  - The dApp is not registered with AVNU for sponsored mode\n" +
              "  - Daily gas limit exceeded\n" +
              "Try: --no-paymaster (account pays gas directly)"
          );
        }
        if (/insufficient.?balance|balance.?too.?low|insufficient.?funds/i.test(msg)) {
          die(
            `Account has insufficient balance: ${msg}\n` +
              "The account does not have enough STRK/ETH for gas.\n" +
              "  - If using --no-paymaster: fund the account before retrying\n" +
              "  - If using paymaster: the paymaster may have rejected sponsorship;\n" +
              "    check the AVNU dashboard or try again later"
          );
        }
        die(`${noPaymaster ? "Direct" : "Paymaster"} submission failed (permanent): ${msg}`);
      }
      info(`  Attempt ${attempt + 1}/${SINGLE_TX_MAX_RETRIES} failed: ${msg}`);
      if (attempt < SINGLE_TX_MAX_RETRIES - 1) {
        const isRateLimit = /429|rate.?limit|too many|throttl/i.test(msg);
        const backoffMs = isRateLimit
          ? Math.min((attempt + 1) * 30000, 60000)
          : Math.min((attempt + 1) * 5000, 20000);
        info(`  Retrying in ${backoffMs / 1000}s...`);
        await new Promise((r) => setTimeout(r, backoffMs));
      } else {
        die(`${noPaymaster ? "Direct" : "Paymaster"} submission failed after ${SINGLE_TX_MAX_RETRIES} attempts: ${msg}`);
      }
    }
  }

  info(`TX submitted: ${txHash}`);
  info("Waiting for confirmation...");

  const receipt = await waitWithTimeout(txHash);
  const execStatus = receipt.execution_status ?? receipt.finality_status ?? receipt.status ?? "unknown";

  if (execStatus === "REVERTED") {
    die(`TX reverted: ${receipt.revert_reason || "unknown reason"}`);
  }

  // ── Check post-submit verification status with assurance separation ──
  // Retry verification count fetch with exponential backoff (matches chunked path).
  let verificationCountAfter = null;
  const VC_MAX_ATTEMPTS = 8;
  for (let vcAttempt = 0; vcAttempt < VC_MAX_ATTEMPTS; vcAttempt++) {
    verificationCountAfter = await fetchVerificationCount(provider, contract, modelId);
    if (verificationCountAfter !== null) break;
    if (vcAttempt < VC_MAX_ATTEMPTS - 1) {
      const waitMs = Math.min(3000 * Math.pow(1.5, vcAttempt), 15000);
      info(`  Verification count query attempt ${vcAttempt + 1}/${VC_MAX_ATTEMPTS} failed, retrying in ${(waitMs / 1000).toFixed(0)}s...`);
      await new Promise((r) => setTimeout(r, waitMs));
    }
  }
  let verificationCountDelta = null;
  let hasAnyVerification = false;
  let acceptedOnchain = false;
  let acceptanceEvidence = "unknown";

  if (verificationCountAfter !== null) {
    hasAnyVerification = verificationCountAfter > 0n;
    if (verificationCountBefore !== null) {
      verificationCountDelta = verificationCountAfter - verificationCountBefore;
      acceptedOnchain = verificationCountDelta > 0n;
      acceptanceEvidence = "verification_count_delta";
    } else {
      acceptedOnchain = hasAnyVerification;
      acceptanceEvidence = "verification_count_observed";
    }
  } else {
    const proofVerifiedFlag = await fetchProofVerifiedFlag(provider, contract, modelId);
    if (proofVerifiedFlag !== null) {
      hasAnyVerification = proofVerifiedFlag;
      acceptedOnchain = proofVerifiedFlag;
      acceptanceEvidence = "is_proof_verified_fallback";
    } else {
      // Cannot confirm on-chain acceptance — DO NOT assume success.
      // Match chunked flow behavior: report as unconfirmed.
      acceptedOnchain = false;
      hasAnyVerification = false;
      acceptanceEvidence = "unconfirmed_rpc_unavailable";
      info("WARNING: Could not confirm on-chain acceptance. RPC view methods unavailable.");
      info("  The verify TX succeeded, but acceptance is unconfirmed.");
      info("  Check manually: is_proof_verified(" + modelId + ") on " + contract);
    }
  }

  // If verification count didn't increase, try is_proof_verified fallback before giving up.
  if (!acceptedOnchain && acceptanceEvidence === "verification_count_delta") {
    info("WARNING: verification_count did not increase. Checking is_proof_verified fallback...");
    const proofVerifiedFallback = await fetchProofVerifiedFlag(provider, contract, modelId);
    if (proofVerifiedFallback === true) {
      acceptedOnchain = true;
      hasAnyVerification = true;
      acceptanceEvidence = "is_proof_verified_fallback";
      info("  Confirmed via is_proof_verified fallback.");
    } else {
      info("WARNING: Could not confirm verification via either method.");
      info("  This may indicate a replayed proof or verifier-side rejection.");
      info("  Check manually: is_proof_verified(" + modelId + ") on " + contract);
    }
  }

  const onchainAssurance = acceptedOnchain ? "full_gkr" : "unconfirmed";
  const fullGkrVerified = acceptedOnchain;
  const isVerified = fullGkrVerified;

  const explorerUrl = `${net.explorer}${txHash}`;
  info(`Explorer: ${explorerUrl}`);
  info(`Accepted on-chain: ${acceptedOnchain}`);
  info(`Full GKR verified: ${fullGkrVerified}`);
  if (verificationCountBefore !== null) {
    info(`Verification count (before): ${verificationCountBefore.toString()}`);
  }
  if (verificationCountAfter !== null) {
    info(`Verification count (after): ${verificationCountAfter.toString()}`);
  }
  if (verificationCountDelta !== null) {
    info(`Verification count delta: ${verificationCountDelta.toString()}`);
  }

  jsonOutput({
    txHash,
    explorerUrl,
    isVerified,
    acceptedOnchain,
    fullGkrVerified,
    hasAnyVerification,
    verificationCountBefore:
      verificationCountBefore === null
        ? null
        : verificationCountBefore.toString(),
    verificationCountAfter:
      verificationCountAfter === null ? null : verificationCountAfter.toString(),
    verificationCountDelta:
      verificationCountDelta === null ? null : verificationCountDelta.toString(),
    acceptanceEvidence,
    gasSponsored,
    accountDeployed: false,
    executionStatus: execStatus,
    entrypoint: verifyPayload.entrypoint,
    onchainAssurance,
    sessionId: verifyPayload.sessionId || null,
    uploadedChunks: verifyPayload.uploadChunks ? verifyPayload.uploadChunks.length : 0,
    totalTxs: 1,
    allTxHashes: [txHash],
    e2eDurationSeconds: parseFloat(((Date.now() - singleTxStart) / 1000).toFixed(1)),
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Command: setup  (factory path — ERC-8004 identity)
// ═══════════════════════════════════════════════════════════════════════
//
// Deploys an agent account via the AgentAccountFactory. This registers
// the account in the identity registry and mints an ERC-8004 identity NFT.
// Requires OBELYSK_DEPLOYER_KEY + OBELYSK_DEPLOYER_ADDRESS.

async function cmdSetup(args) {
  const network = args.network || "sepolia";
  const net = NETWORKS[network];
  if (!net) die(`Unknown network: ${network}`);
  if (!net.factory) die(`Factory not deployed on ${network}`);

  const deployerKey = process.env.OBELYSK_DEPLOYER_KEY;
  if (!deployerKey) {
    die(
      "OBELYSK_DEPLOYER_KEY is required for factory deployment (ERC-8004 identity).\n" +
        "For zero-config without identity, just run 'verify' directly — it auto-deploys."
    );
  }
  const deployerAddress = process.env.OBELYSK_DEPLOYER_ADDRESS;
  if (!deployerAddress) {
    die(
      "OBELYSK_DEPLOYER_ADDRESS is required for factory deployment.\n" +
        "This is the address of the deployer account."
    );
  }

  const provider = getProvider(network);
  const deployer = getAccount(provider, deployerKey, deployerAddress, network);

  // Generate new keypair for the pipeline account
  info("Generating new Stark keypair...");
  const privateKey = normalizePrivateKey(ec.starkCurve.utils.randomPrivateKey());
  const publicKey = ec.starkCurve.getStarkKey(privateKey);
  info(`Public key: ${publicKey}`);

  // Build factory deploy_account call
  const salt = publicKey;
  const tokenUri = byteArray.byteArrayFromString(
    'data:application/json,{"name":"ObelyskPipeline","description":"Proof submission agent","agentType":"prover"}'
  );

  const deployCall = {
    contractAddress: args.factory || net.factory,
    entrypoint: "deploy_account",
    calldata: CallData.compile({
      public_key: publicKey,
      salt,
      token_uri: tokenUri,
    }),
  };

  info("Deploying agent account via factory (ERC-8004 identity)...");
  const txHash = await executeViaPaymaster(deployer, deployCall);
  info(`TX: ${txHash}`);

  info("Waiting for confirmation...");
  const receipt = await waitWithTimeout(txHash, 180000);
  const execStatus = receipt.execution_status ?? receipt.finality_status ?? receipt.status ?? "unknown";
  if (execStatus === "REVERTED") {
    die(`Account deployment reverted: ${receipt.revert_reason || "unknown"}`);
  }

  // Parse AccountDeployed event — filter by from_address to avoid matching
  // events from other contracts (e.g. identity registry mint, ERC-721 transfer).
  let accountAddress = null;
  let agentId = null;
  const factoryAddress = args.factory || net.factory;
  const events = receipt.events || [];
  for (const event of events) {
    // Only consider events emitted by the factory contract
    if (event.from_address) {
      try {
        if (BigInt(event.from_address) !== BigInt(factoryAddress)) continue;
      } catch { continue; }
    }
    if (
      event.keys &&
      event.keys.length >= 3 &&
      event.data &&
      event.data.length >= 3
    ) {
      try {
        const possiblePubKey = event.keys[2];
        if (possiblePubKey && BigInt(possiblePubKey) === BigInt(publicKey)) {
          accountAddress = "0x" + BigInt(event.keys[1]).toString(16);
          const idLow = BigInt(event.data[0] || "0");
          const idHigh = BigInt(event.data[1] || "0");
          agentId = (idLow + (idHigh << 128n)).toString();
          break;
        }
      } catch { /* skip malformed event data */ }
    }
  }

  if (!accountAddress) {
    // CRITICAL: Deploy TX succeeded but we can't find the account address.
    // Save partial config so the private key isn't lost — the account exists on-chain.
    const partialConfig = {
      address: null,
      privateKey,
      publicKey,
      agentId: null,
      network,
      ephemeral: false,
      factory: factoryAddress,
      deployedAt: new Date().toISOString(),
      deployTxHash: txHash,
      parseFailure: true,
      eventCount: events.length,
    };
    saveAccountConfig(partialConfig);
    die(
      `Deploy TX succeeded (${txHash}) but could not parse AccountDeployed event.\n` +
        `  ${events.length} events found in receipt, none matched factory ${factoryAddress}.\n` +
        `  Private key SAVED to ${ACCOUNT_CONFIG_FILE} — do NOT delete this file.\n` +
        `  To recover: inspect TX events on the explorer (${net.explorer}${txHash})\n` +
        `  and manually set the "address" field in ${ACCOUNT_CONFIG_FILE}.`
    );
  }

  // Save config
  const config = {
    address: accountAddress,
    privateKey,
    publicKey,
    agentId,
    network,
    ephemeral: false,
    factory: args.factory || net.factory,
    deployedAt: new Date().toISOString(),
    deployTxHash: txHash,
  };
  saveAccountConfig(config);
  info(`Account saved to ${ACCOUNT_CONFIG_FILE}`);

  jsonOutput({
    address: accountAddress,
    agentId,
    txHash,
    explorerUrl: `${net.explorer}${txHash}`,
    identity: true,
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Command: status
// ═══════════════════════════════════════════════════════════════════════

async function cmdStatus(args) {
  const network = args.network || "sepolia";
  const net = NETWORKS[network];
  if (!net) die(`Unknown network: ${network}`);

  const contract = args.contract;
  const modelId = args["model-id"] || "0x1";

  const provider = getProvider(network);
  const config = loadAccountConfig();

  // Account status
  let accountStatus = { configured: false };
  if (config) {
    accountStatus = {
      configured: true,
      address: config.address,
      agentId: config.agentId,
      ephemeral: config.ephemeral ?? true,
      network: config.network,
      createdAt: config.createdAt || config.deployedAt,
    };

    // Check if deployed on-chain
    try {
      const classHash = await provider.getClassHashAt(config.address);
      accountStatus.deployedOnChain = !!classHash;
      accountStatus.classHash = classHash;
    } catch {
      accountStatus.deployedOnChain = false;
    }
  }

  // Verification status
  let verificationStatus = { checked: false };
  if (contract) {
    try {
      const count = await fetchVerificationCount(provider, contract, modelId);
      const countStr = count === null ? null : count.toString();
      const hasAnyVerification = count !== null ? count > 0n : false;
      verificationStatus = {
        checked: true,
        contract,
        modelId,
        verificationCount: countStr,
        hasAnyVerification,
        // Strict signal remains false here because status endpoint is mode-agnostic.
        fullGkrVerified: false,
        isVerified: false,
        assuranceNote:
          "verification_count is mode-agnostic; use verify command output for assurance-separated status",
      };
    } catch (e) {
      verificationStatus = { checked: true, error: e.message };
    }
  }

  jsonOutput({ account: accountStatus, verification: verificationStatus });
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════

const args = parseArgs(process.argv);

try {
  switch (args.command) {
    case "verify":
      await cmdVerify(args);
      break;
    case "setup":
      await cmdSetup(args);
      break;
    case "status":
      await cmdStatus(args);
      break;
    default:
      process.stderr.write(
        "Usage: node paymaster_submit.mjs <command> [options]\n\n" +
          "Commands:\n" +
          "  verify   Submit proof via AVNU paymaster (auto-deploys account if needed)\n" +
          "  setup    Deploy agent account via factory (ERC-8004 identity)\n" +
          "  status   Check account and verification status\n\n" +
          "With funder account (for ephemeral account deploy gas):\n" +
          "  OBELYSK_FUNDER_KEY=0x... OBELYSK_FUNDER_ADDRESS=0x... \\\n" +
          "    node paymaster_submit.mjs verify --proof proof.json --contract 0x... --model-id 0x1\n\n" +
          "With pre-deployed account (skip ephemeral):\n" +
          "  STARKNET_PRIVATE_KEY=0x... STARKNET_ACCOUNT_ADDRESS=0x... \\\n" +
          "    node paymaster_submit.mjs verify --proof proof.json --contract 0x... --model-id 0x1\n"
      );
      process.exit(1);
  }
} catch (e) {
  die(truncateRpcError(e));
}
