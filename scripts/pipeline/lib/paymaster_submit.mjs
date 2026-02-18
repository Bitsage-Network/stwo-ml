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
    rpcPublic: "https://free-rpc.nethermind.io/sepolia-juno/",
    paymasterUrl: "https://sepolia.paymaster.avnu.fi",
    explorer: "https://sepolia.starkscan.co/tx/",
    factory: "0x2f69e566802910359b438ccdb3565dce304a7cc52edbf9fd246d6ad2cd89ce4",
    accountClassHash:
      "0x14d44fb938b43e5fbcec27894670cb94898d759e2ef30e7af70058b4da57e7f",
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
  },
};

const ACCOUNT_CONFIG_DIR = join(homedir(), ".obelysk", "starknet");
const ACCOUNT_CONFIG_FILE = join(ACCOUNT_CONFIG_DIR, "pipeline_account.json");

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

function info(msg) {
  process.stderr.write(`[INFO] ${msg}\n`);
}

// ─── Ephemeral Account ───────────────────────────────────────────────
//
// Generates a keypair and computes the counterfactual address for the
// agent account class (already declared on Sepolia). The account doesn't
// need to exist on-chain yet — deploymentData in the paymaster TX will
// deploy it atomically alongside the proof verification call.
//
// Constructor: (public_key: felt252, factory: ContractAddress)
// For ephemeral accounts we pass factory = 0x0 (no identity registration).

function generateEphemeralAccount(network) {
  const net = NETWORKS[network];
  if (!net?.accountClassHash) die(`No account class hash for ${network}`);

  const privateKey = normalizePrivateKey(ec.starkCurve.utils.randomPrivateKey());
  const publicKey = ec.starkCurve.getStarkKey(privateKey);
  const salt = publicKey;

  // Agent account constructor calldata: [public_key, factory=0x0]
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

async function executeViaPaymaster(account, calls, deploymentData) {
  const callsArray = Array.isArray(calls) ? calls : [calls];
  const feeDetails = { feeMode: { mode: "sponsored" } };

  if (deploymentData) {
    feeDetails.deploymentData = deploymentData;
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


function parseVerifyCalldata(proofData, fallbackModelId) {
  const verifyCalldata = proofData.verify_calldata;
  if (!verifyCalldata || typeof verifyCalldata !== "object" || Array.isArray(verifyCalldata)) {
    die("Proof file missing 'verify_calldata' object");
  }

  const schemaVersion = verifyCalldata.schema_version;
  if (schemaVersion !== 1) {
    die("verify_calldata.schema_version must be 1");
  }

  const entrypoint = verifyCalldata.entrypoint;
  if (typeof entrypoint !== "string" || entrypoint.length === 0) {
    die("verify_calldata.entrypoint must be a non-empty string");
  }
  const allowedEntrypoints = new Set(["verify_model_gkr", "verify_model_gkr_v2"]);
  if (!allowedEntrypoints.has(entrypoint)) {
    die(
      `Only verify_model_gkr / verify_model_gkr_v2 are supported in the hardened pipeline (got: ${entrypoint})`
    );
  }

  const rawCalldata = verifyCalldata.calldata;
  if (!Array.isArray(rawCalldata) || rawCalldata.length === 0) {
    die("verify_calldata.calldata must be a non-empty array");
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
  } else if (entrypoint === "verify_model_gkr_v2") {
    const allowedModes = new Set(["Sequential", "BatchedSubchannelV1"]);
    if (weightOpeningMode !== undefined && !allowedModes.has(weightOpeningMode)) {
      die(
        `${entrypoint} requires weight_opening_mode in {Sequential,BatchedSubchannelV1} (got: ${proofData.weight_opening_mode})`
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

  if (entrypoint === "verify_model_gkr_v2") {
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
    let expectedMode = null;
    if (weightOpeningMode === "Sequential") {
      expectedMode = 0;
    } else if (weightOpeningMode === "BatchedSubchannelV1") {
      expectedMode = 1;
    }
    if (expectedMode !== null && weightBindingMode !== expectedMode) {
      die(
        `verify_model_gkr_v2 expected weight_binding_mode=${expectedMode} for weight_opening_mode=${weightOpeningMode} (got ${weightBindingMode})`
      );
    }
    if (expectedMode === null && !new Set([0, 1]).has(weightBindingMode)) {
      die(
        `verify_model_gkr_v2 requires weight_binding_mode in {0,1} (got ${weightBindingMode})`
      );
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

  if (!new Set(["verify_model_gkr", "verify_model_gkr_v2"]).has(verifyPayload.entrypoint)) {
    die(
      `Only verify_model_gkr / verify_model_gkr_v2 are supported in the hardened pipeline (got: ${verifyPayload.entrypoint})`
    );
  }

  if (
    args["model-id"] &&
    String(args["model-id"]).toLowerCase() !== String(modelId).toLowerCase()
  ) {
    info(
      `--model-id ${args["model-id"]} differs from proof artifact model_id ${modelId}; using proof artifact value`
    );
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
  if (verifyPayload.sessionId) {
    info(`Session ID: ${verifyPayload.sessionId}`);
  }

  // ── Build deploymentData if account needs deploying ──
  let deploymentData = undefined;
  if (needsDeploy && ephemeral) {
    deploymentData = {
      classHash: net.accountClassHash,
      salt: ephemeral.salt,
      uniqueDeployerAddress: "0x0",
      constructorCalldata: ephemeral.constructorCalldata,
    };
    info("Including deploymentData — account will be deployed in this TX");
  }

  // ── Execute ──
  let txHash;
  try {
    txHash = await executeViaPaymaster(account, calls, deploymentData);
  } catch (e) {
    const msg = e.message || String(e);
    if (msg.includes("not eligible") || msg.includes("not supported")) {
      die(
        `Paymaster rejected transaction: ${msg}\n` +
          "This may mean:\n" +
          "  - The dApp is not registered with AVNU for sponsored mode\n" +
          "  - Daily gas limit exceeded\n" +
          "  - The account class is not whitelisted\n" +
          "Try: STARKNET_PRIVATE_KEY=0x... ./04_verify_onchain.sh --submit --no-paymaster"
      );
    }
    throw e;
  }

  info(`TX submitted: ${txHash}`);
  info("Waiting for confirmation...");

  const receipt = await provider.waitForTransaction(txHash);
  const execStatus = receipt.execution_status ?? receipt.status ?? "unknown";

  if (execStatus === "REVERTED") {
    die(`TX reverted: ${receipt.revert_reason || "unknown reason"}`);
  }

  // Update saved config with deployment TX
  if (needsDeploy && ephemeral) {
    const config = loadAccountConfig();
    if (config) {
      config.deployTxHash = txHash;
      config.deployedAt = new Date().toISOString();
      config.ephemeral = true;
      saveAccountConfig(config);
    }
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
    gasSponsored: true,
    accountDeployed: needsDeploy,
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
  die(e.message || String(e));
}
