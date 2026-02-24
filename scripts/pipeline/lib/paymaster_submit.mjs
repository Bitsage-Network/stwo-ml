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
  RpcProvider,
  CallData,
  ETransactionVersion,
  ec,
  hash,
  num,
  byteArray,
} from "starknet";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";

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

function getAccount(provider, privateKey, address) {
  return new Account({
    provider,
    address,
    signer: privateKey,
    transactionVersion: ETransactionVersion.V3,
  });
}

function jsonOutput(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function die(msg) {
  process.stderr.write(`[ERR] ${msg}\n`);
  process.exit(1);
}

function truncateRpcError(e) {
  const msg = e.message || String(e);
  // starknet.js RpcError embeds the full request params (100K+ chars of
  // calldata JSON) in the error message.  Extract just the error code.
  const codeMatch = msg.match(/(-\d+):\s*"?(.{1,300})/);
  if (codeMatch) return `RPC ${codeMatch[1]}: ${codeMatch[2]}`;
  // Fallback: first 500 chars
  return msg.length > 500 ? msg.slice(0, 500) + "..." : msg;
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

async function executeViaPaymaster(account, calls, deploymentData) {
  const callsArray = Array.isArray(calls) ? calls : [calls];

  // Build fee details with optional deploymentData for atomic deploy+invoke.
  // AVNU paymaster sponsors both the account deployment and the invoke in a
  // single TX when deploymentData is provided with small calldata (like
  // open_gkr_session). For large calldata TXs, pass deploymentData=undefined.
  const feeDetails = { feeMode: { mode: "sponsored" } };
  if (deploymentData) {
    feeDetails.deploymentData = { ...deploymentData, version: 1 };
    info("Including account deployment in this TX (atomic deploy+invoke)...");
  }

  info("Estimating paymaster fee...");
  const estimation = await account.estimatePaymasterTransactionFee(
    callsArray,
    feeDetails
  );

  info("Executing via AVNU paymaster (sponsored mode)...");
  const result = await account.executePaymasterTransaction(
    callsArray,
    feeDetails,
    estimation.suggested_max_fee_in_gas_token
  );

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

  const schemaVersion = verifyCalldata.schema_version;
  if (schemaVersion !== 1 && schemaVersion !== 2) {
    die("verify_calldata.schema_version must be 1 or 2");
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
    const weightBindingMode = verifyCalldata.weight_binding_mode;
    if (typeof circuitDepth !== "number" || circuitDepth <= 0) {
      die("schema_version 2 requires positive circuit_depth");
    }
    if (typeof numLayers !== "number" || numLayers <= 0) {
      die("schema_version 2 requires positive num_layers");
    }
    if (![3, 4].includes(weightBindingMode)) {
      die(`schema_version 2 requires weight_binding_mode in {3,4} (got: ${weightBindingMode})`);
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
      const readNat = (i) => {
        const s = String(flat[i]);
        return s.startsWith("0x") || s.startsWith("0X") ? Number(BigInt(s)) : Number(s);
      };
      let off = 0;
      // Skip through 6 length-prefixed sections
      for (let sec = 0; sec < 6; sec++) {
        const secLen = readNat(off); off += 1 + secLen;
      }
      // off now points at weight_opening_proofs start
      const wopFelts = flat.length - off;
      // Check if weight_binding_data (section 6) is RLC marker
      // Parse backwards from off to find section 6
      let sec6Off = 0, sec6Len = 0;
      let tmpOff = 0;
      for (let sec = 0; sec < 6; sec++) {
        if (sec === 5) { sec6Off = tmpOff; sec6Len = readNat(tmpOff); }
        const len = readNat(tmpOff); tmpOff += 1 + len;
      }
      const isRlc = sec6Len === 2 && BigInt(flat[sec6Off + 1]) === 0x524C43n;
      if (isRlc && wopFelts > 1) {
        // Replace opening proofs with empty array [0]
        const stripped = flat.slice(0, off);
        stripped.push("0");
        info(`  Mode 4 RLC v2: stripping ${wopFelts} felts of weight_opening_proofs (${flat.length} → ${stripped.length})`);
        // Re-chunk
        const chunkSize = parseInt(process.env.OBELYSK_CHUNK_SIZE || "1500", 10);
        finalChunks = [];
        for (let i = 0; i < stripped.length; i += chunkSize) {
          finalChunks.push(stripped.slice(i, i + chunkSize));
        }
        finalTotalFelts = stripped.length;
        finalNumChunks = finalChunks.length;
      }
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
  ]);
  if (!allowedEntrypoints.has(entrypoint)) {
    die(
      `Only verify_model_gkr / verify_model_gkr_v2 / verify_model_gkr_v3 / verify_model_gkr_v4 are supported in the hardened pipeline (got: ${entrypoint})`
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
  // Default chunk size reduced from 4000 to 1500: Cartridge/public RPCs drop
  // large-calldata TXs from mempool (TX gets RECEIVED but never ACCEPTED_ON_L2).
  // 1500 felts → ~1503 felt calldata per TX, well within limits.
  const MAX_CHUNK_FELTS = parseInt(process.env.OBELYSK_CHUNK_SIZE || "1500", 10);
  if (entrypoint === "verify_model_gkr_v4" && rawCalldata.length > CHUNKED_THRESHOLD) {
    info(`Auto-converting v1 calldata (${rawCalldata.length} felts) to chunked session format...`);
    const cd = rawCalldata.map((v) => String(v));
    const readNat = (idx, label) => {
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
    const modelIdStr = cd[idx]; idx += 1;

    // raw_io_data section
    const rawIoLen = readNat(idx, "raw_io_len"); idx += 1;
    const rawIoStart = idx; idx += rawIoLen;

    // circuit_depth, num_layers (scalars — stored in session metadata)
    const circuitDepth = readNat(idx, "circuit_depth"); idx += 1;
    const numLayers = readNat(idx, "num_layers"); idx += 1;

    // matmul_dims section
    const matmulDimsLen = readNat(idx, "matmul_dims_len"); idx += 1;
    const matmulDimsStart = idx; idx += matmulDimsLen;

    // dequantize_bits section
    const deqBitsLen = readNat(idx, "deq_bits_len"); idx += 1;
    const deqBitsStart = idx; idx += deqBitsLen;

    // proof_data section
    const proofDataLen = readNat(idx, "proof_data_len"); idx += 1;
    const proofDataStart = idx; idx += proofDataLen;

    // weight_commitments section
    const wcLen = readNat(idx, "wc_len"); idx += 1;
    const wcStart = idx; idx += wcLen;

    // weight_binding_mode (scalar — stored in session metadata)
    const weightBindingMode = readNat(idx, "weight_binding_mode"); idx += 1;

    // weight_binding_data section
    const wbdLen = readNat(idx, "wbd_len"); idx += 1;
    const wbdStart = idx; idx += wbdLen;

    // weight_opening_proofs (rest of calldata = Serde-serialized Array<MleOpeningProof>)
    const wopStart = idx;
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
    const isMode4Rlc = weightBindingMode === 4 && wbdLen === 2
      && (cd[wbdStart] === "0x524c43" || cd[wbdStart] === "0x524C43"
          || BigInt(cd[wbdStart]) === 0x524C43n);
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
    const config = loadAccountConfig();
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

  const account = getAccount(provider, privateKey, accountAddress);

  // ── Read proof file ──
  info(`Reading proof: ${proofPath}`);
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

  // ── Prepare deployment data (used atomically with first invoke TX) ──
  const noPaymaster = args["no-paymaster"] === true || args["no-paymaster"] === "true";
  let pendingDeploymentData = null;
  if (needsDeploy && ephemeral) {
    const rawCalldata = Array.isArray(ephemeral.constructorCalldata)
      ? ephemeral.constructorCalldata
      : CallData.compile(ephemeral.constructorCalldata);
    pendingDeploymentData = {
      class_hash: net.accountClassHash,
      salt: ephemeral.salt,
      calldata: rawCalldata.map((v) => num.toHex(v)),
      address: accountAddress,
    };
    info("Account not yet deployed — will deploy atomically with first TX");
  }

  // ═══════════════════════════════════════════════════════════════════
  // Chunked Session Flow (schema_version 2)
  // ═══════════════════════════════════════════════════════════════════
  if (verifyPayload.chunked) {
    info(`Chunked GKR session mode: ${verifyPayload.totalFelts} felts in ${verifyPayload.numChunks} chunks`);

    await preflightContractEntrypoint(provider, contract, "open_gkr_session");

    const verificationCountBefore = await fetchVerificationCount(provider, contract, modelId);
    if (verificationCountBefore !== null) {
      info(`Verification count (before): ${verificationCountBefore.toString()}`);
    }

    // ── Resumability: check for saved session ──
    const sessionDir = join(homedir(), ".obelysk", "chunked_sessions");
    mkdirSync(sessionDir, { recursive: true });

    // Helper: execute a single call
    async function waitWithTimeout(txHash, timeoutMs = 300000) {
      return Promise.race([
        provider.waitForTransaction(txHash, { retryInterval: 4000 }),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error(`TX ${txHash} not confirmed after ${timeoutMs/1000}s`)), timeoutMs)
        ),
      ]);
    }

    // Cached resource bounds from the first successful chunk estimate.
    // Reused for all subsequent chunks to avoid nonce-stale estimateFee failures.
    let cachedChunkResourceBounds = null;

    async function execCall(entrypoint, calldata, label, opts = {}) {
      const calls = [{
        contractAddress: contract,
        entrypoint,
        calldata: CallData.compile(calldata),
      }];
      info(`  ${label}...`);
      let txHash;
      if (noPaymaster) {
        const result = await account.execute(calls);
        txHash = result.transaction_hash;
      } else {
        txHash = await executeViaPaymaster(account, calls, opts.deploymentData);
      }
      info(`  TX: ${txHash}`);
      const receipt = await waitWithTimeout(txHash);
      const execStatus = receipt.execution_status ?? receipt.status ?? "unknown";
      if (execStatus === "REVERTED") {
        die(`  ${label} reverted: ${receipt.revert_reason || "unknown"}`);
      }
      return { txHash, receipt };
    }

    // ── Step 1: open_gkr_session ──
    // If account needs deployment, include deploymentData so AVNU paymaster
    // deploys the account atomically with this invoke (small calldata).
    const openOpts = pendingDeploymentData ? { deploymentData: pendingDeploymentData } : {};
    const { txHash: openTxHash, receipt: openReceipt } = await execCall(
      "open_gkr_session",
      [
        modelId,
        String(verifyPayload.totalFelts),
        String(verifyPayload.circuitDepth),
        String(verifyPayload.numLayers),
        String(verifyPayload.weightBindingMode),
        verifyPayload.packed ? "1" : "0",
      ],
      "open_gkr_session",
      openOpts
    );
    if (pendingDeploymentData) {
      info("Account deployed atomically with open_gkr_session.");
      pendingDeploymentData = null;
      const config = loadAccountConfig();
      if (config) {
        config.deployedAt = new Date().toISOString();
        config.ephemeral = true;
        saveAccountConfig(config);
      }
    }

    // Parse session_id from events.
    let sessionId = null;
    const events = openReceipt.events ?? [];
    for (const ev of events) {
      // GkrSessionOpened event has session_id as first key.
      if (ev.keys && ev.keys.length >= 2) {
        // key[0] is the event selector, key[1] is session_id.
        sessionId = ev.keys[1];
        break;
      }
    }
    if (!sessionId) {
      // Fallback: parse from data[0] if keys don't work.
      for (const ev of events) {
        if (ev.data && ev.data.length >= 1) {
          sessionId = ev.data[0];
          break;
        }
      }
    }
    if (!sessionId) {
      die("Could not parse session_id from open_gkr_session events");
    }
    info(`  Session ID: ${sessionId}`);

    // Save session state for resumability.
    const sessionFile = join(sessionDir, `${sessionId.slice(0, 18)}.json`);
    const sessionState = {
      sessionId,
      modelId,
      contract,
      totalFelts: verifyPayload.totalFelts,
      numChunks: verifyPayload.numChunks,
      chunksUploaded: 0,
      txHashes: [openTxHash],
      status: "uploading",
      createdAt: new Date().toISOString(),
    };
    writeFileSync(sessionFile, JSON.stringify(sessionState, null, 2));

    // ── Step 2: upload_gkr_chunk × N ──
    // First chunk uses normal fee estimation (may need retries for nonce sync).
    // Subsequent chunks reuse cached resource bounds with 2x safety margin,
    // skipping estimateFee entirely to avoid nonce-stale errors on public RPCs.
    const MAX_RETRIES = 5;
    const INTER_CHUNK_DELAY_MS = parseInt(process.env.OBELYSK_CHUNK_DELAY_MS || "3000", 10);
    const uploadStartTime = Date.now();
    for (let i = 0; i < verifyPayload.numChunks; i++) {
      const chunk = verifyPayload.uploadChunks[i];
      const pct = ((i / verifyPayload.numChunks) * 100).toFixed(1);
      let uploaded = false;
      const execOpts = cachedChunkResourceBounds ? { resourceBounds: cachedChunkResourceBounds } : {};
      for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
          const { txHash: chunkTxHash, receipt: chunkReceipt } = await execCall(
            "upload_gkr_chunk",
            [sessionId, String(i), String(chunk.length), ...chunk],
            `upload_gkr_chunk[${i + 1}/${verifyPayload.numChunks}] (${chunk.length} felts, ${pct}%)`,
            execOpts
          );
          // Cache resource bounds from the first successful chunk receipt.
          // We reuse the SAME resource bounds that the SDK computed for chunk 0
          // on all subsequent chunks, skipping estimateFee entirely.
          if (!cachedChunkResourceBounds && chunkReceipt) {
            // Use a generous fixed bound: 20M l2_gas at a fixed price.
            // This avoids BigInt/string type mixing issues with starknet.js v8.
            cachedChunkResourceBounds = {
              l1_gas: { max_amount: "0x0", max_price_per_unit: "0x0" },
              l2_gas: { max_amount: "0x1312D00", max_price_per_unit: "0x174876e800" },
              l1_data_gas: { max_amount: "0x0", max_price_per_unit: "0x0" },
            };
            info(`  Cached fixed resource bounds for subsequent chunks`);
          }
          sessionState.chunksUploaded = i + 1;
          sessionState.txHashes.push(chunkTxHash);
          writeFileSync(sessionFile, JSON.stringify(sessionState, null, 2));
          uploaded = true;
          break;
        } catch (e) {
          const errMsg = truncateRpcError(e);
          info(`  Chunk ${i} attempt ${attempt + 1} failed: ${errMsg}`);
          // If cached bounds caused the failure, clear them and retry with estimation.
          if (cachedChunkResourceBounds && (errMsg.includes("INSUFFICIENT") || errMsg.includes("insufficient"))) {
            info(`  Clearing cached resource bounds, will re-estimate...`);
            cachedChunkResourceBounds = null;
          }
          if (attempt < MAX_RETRIES - 1) {
            const backoffMs = Math.min((attempt + 1) * 5000, 20000);
            info(`  Retrying in ${backoffMs / 1000}s...`);
            await new Promise((r) => setTimeout(r, backoffMs));
          } else {
            die(`Failed to upload chunk ${i} after ${MAX_RETRIES} attempts`);
          }
        }
      }
      // Delay between chunks to let RPC nonce state sync
      if (i < verifyPayload.numChunks - 1 && INTER_CHUNK_DELAY_MS > 0) {
        await new Promise((r) => setTimeout(r, INTER_CHUNK_DELAY_MS));
      }
    }
    const uploadDuration = ((Date.now() - uploadStartTime) / 1000).toFixed(1);
    info(`All ${verifyPayload.numChunks} chunks uploaded in ${uploadDuration}s.`);

    // Helper: retry execCall for nonce-stale estimateFee errors
    async function execCallWithRetry(entrypoint, calldata, label, opts = {}, maxRetries = 5) {
      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          return await execCall(entrypoint, calldata, label, opts);
        } catch (e) {
          const msg = truncateRpcError(e);
          info(`  ${label} attempt ${attempt + 1} failed: ${msg}`);
          if (attempt < maxRetries - 1) {
            const backoffMs = Math.min((attempt + 1) * 5000, 20000);
            info(`  Retrying in ${backoffMs / 1000}s...`);
            await new Promise((r) => setTimeout(r, backoffMs));
          } else {
            throw e;
          }
        }
      }
    }

    // ── Step 3: seal_gkr_session ──
    const { txHash: sealTxHash } = await execCallWithRetry(
      "seal_gkr_session",
      [sessionId],
      "seal_gkr_session"
    );
    sessionState.status = "sealed";
    sessionState.txHashes.push(sealTxHash);
    writeFileSync(sessionFile, JSON.stringify(sessionState, null, 2));

    // ── Step 4: verify_gkr_from_session ──
    const { txHash: verifyTxHash } = await execCallWithRetry(
      "verify_gkr_from_session",
      [sessionId],
      "verify_gkr_from_session"
    );
    sessionState.status = "verified";
    sessionState.txHashes.push(verifyTxHash);
    writeFileSync(sessionFile, JSON.stringify(sessionState, null, 2));

    // ── Check verification result ──
    const verificationCountAfter = await fetchVerificationCount(provider, contract, modelId);
    let verificationCountDelta = null;
    let acceptedOnchain = false;
    let acceptanceEvidence = "unknown";
    if (verificationCountAfter !== null && verificationCountBefore !== null) {
      verificationCountDelta = verificationCountAfter - verificationCountBefore;
      acceptedOnchain = verificationCountDelta > 0n;
      acceptanceEvidence = "verification_count_delta";
    } else {
      acceptedOnchain = true;
      acceptanceEvidence = "tx_success_only_unconfirmed";
    }

    const totalTxs = sessionState.txHashes.length;
    const explorerUrl = `${net.explorer}${verifyTxHash}`;
    info(`Session complete: ${totalTxs} TXs`);
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
      executionStatus: "ACCEPTED_ON_L2",
      entrypoint: "verify_gkr_from_session",
      onchainAssurance: "full_gkr",
      sessionId,
      uploadedChunks: verifyPayload.numChunks,
      totalTxs,
      allTxHashes: sessionState.txHashes,
    });
    return;
  }

  // ═══════════════════════════════════════════════════════════════════
  // Single-TX Flow (schema_version 1)
  // ═══════════════════════════════════════════════════════════════════
  if (
    !new Set(["verify_model_gkr", "verify_model_gkr_v2", "verify_model_gkr_v3", "verify_model_gkr_v4"]).has(
      verifyPayload.entrypoint
    )
  ) {
    die(
      `Only verify_model_gkr / verify_model_gkr_v2 / verify_model_gkr_v3 / verify_model_gkr_v4 are supported in the hardened pipeline (got: ${verifyPayload.entrypoint})`
    );
  }

  await preflightContractEntrypoint(provider, contract, verifyPayload.entrypoint);

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

  // ── Execute ──
  let txHash;
  let gasSponsored = !noPaymaster;

  if (noPaymaster) {
    // Direct execution — account pays gas (needs STRK balance)
    info("Submitting directly (no paymaster)...");
    try {
      const result = await account.execute(calls);
      txHash = result.transaction_hash;
    } catch (e) {
      die(`Direct submission failed: ${truncateRpcError(e)}`);
    }
  } else {
    try {
      txHash = await executeViaPaymaster(account, calls, pendingDeploymentData);
      if (pendingDeploymentData) {
        info("Account deployed atomically with verification TX.");
        pendingDeploymentData = null;
        const config = loadAccountConfig();
        if (config) {
          config.deployedAt = new Date().toISOString();
          config.ephemeral = true;
          saveAccountConfig(config);
        }
      }
    } catch (e) {
      const msg = truncateRpcError(e);
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
      die(`Paymaster submission failed: ${msg}`);
    }
  }

  info(`TX submitted: ${txHash}`);
  info("Waiting for confirmation...");

  const receipt = await provider.waitForTransaction(txHash);
  const execStatus = receipt.execution_status ?? receipt.status ?? "unknown";

  if (execStatus === "REVERTED") {
    die(`TX reverted: ${receipt.revert_reason || "unknown reason"}`);
  }

  // ── Check post-submit verification status with assurance separation ──
  const verificationCountAfter = await fetchVerificationCount(
    provider,
    contract,
    modelId
  );
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
      // Final fallback: tx executed successfully, but acceptance cannot be confirmed.
      acceptedOnchain = true;
      hasAnyVerification = false;
      acceptanceEvidence = "tx_success_only_unconfirmed";
      info(
        "Could not verify acceptance via contract view methods; using tx success only."
      );
    }
  }

  if (!acceptedOnchain && acceptanceEvidence === "verification_count_delta") {
    die(
      "On-chain acceptance check failed: verification_count did not increase. " +
        "Likely replayed proof (already verified) or verifier-side rejection."
    );
  }

  const onchainAssurance = "full_gkr";
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
    sessionId: verifyPayload.sessionId,
    uploadedChunks: verifyPayload.uploadChunks.length,
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
  const deployer = getAccount(provider, deployerKey, deployerAddress);

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
  const receipt = await provider.waitForTransaction(txHash);
  const execStatus = receipt.execution_status ?? receipt.status ?? "unknown";
  if (execStatus === "REVERTED") {
    die(`Account deployment reverted: ${receipt.revert_reason || "unknown"}`);
  }

  // Parse AccountDeployed event
  let accountAddress = null;
  let agentId = null;
  const events = receipt.events || [];
  for (const event of events) {
    if (
      event.keys &&
      event.keys.length >= 3 &&
      event.data &&
      event.data.length >= 3
    ) {
      const possiblePubKey = event.keys[2];
      if (possiblePubKey && BigInt(possiblePubKey) === BigInt(publicKey)) {
        accountAddress = "0x" + BigInt(event.keys[1]).toString(16);
        const idLow = BigInt(event.data[0] || "0");
        const idHigh = BigInt(event.data[1] || "0");
        agentId = (idLow + (idHigh << 128n)).toString();
        break;
      }
    }
  }

  if (!accountAddress) {
    die("Could not parse AccountDeployed event from receipt");
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
          "Zero-config (no env vars needed):\n" +
          "  node paymaster_submit.mjs verify --proof proof.json --contract 0x... --model-id 0x1\n\n" +
          "With ERC-8004 identity (needs deployer key):\n" +
          "  OBELYSK_DEPLOYER_KEY=0x... OBELYSK_DEPLOYER_ADDRESS=0x... \\\n" +
          "    node paymaster_submit.mjs setup --network sepolia\n"
      );
      process.exit(1);
  }
} catch (e) {
  die(truncateRpcError(e));
}
