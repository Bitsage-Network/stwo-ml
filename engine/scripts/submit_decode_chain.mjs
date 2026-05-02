#!/usr/bin/env node
// Submit a multi-token decode chain to a v4+ recursive verifier on Sepolia.
//
// Required env:
//   KEYSTORE_PATH       — path to OZ-style scrypt keystore for the submitter
//   KEYSTORE_PASSWORD   — keystore password
//   ACCOUNT_ADDRESS     — submitter address (must match keystore privkey)
//   CONTRACT            — v4 recursive verifier contract address
//   PROOF_DIR           — directory containing chain_manifest.json + per-step proof files
//
// Optional env:
//   STARKNET_RPC        — defaults to Alchemy Sepolia demo
//   N_MATMULS           — defaults from manifest, fallback 6
//   HIDDEN_SIZE         — defaults from manifest, fallback 576
//   NUM_TRANSFORMER_BLOCKS — defaults from manifest, fallback 1
//
// chain_manifest.json schema (produced by prove-model --decode --recursive):
//   {
//     "model_id": "0x...",
//     "circuit_hash": "0x...",
//     "weight_super_root": "0x...",
//     "policy_commitment": "0x0",
//     "n_matmuls": 6,
//     "hidden_size": 576,
//     "num_transformer_blocks": 1,
//     "level1_proof_hash": "0x0",
//     "steps": [
//       { "step_idx": 0, "is_prefill": true,  "proof_file": "prefill.recursive.json" },
//       { "step_idx": 1, "is_prefill": false, "proof_file": "decode_0.recursive.json" },
//       ...
//     ]
//   }

import { Account, RpcProvider, CallData, ec } from "starknet";
import { keccak_256 } from "@noble/hashes/sha3";
import { readFileSync } from "fs";
import { resolve as pathResolve } from "path";
import { scrypt as scryptCb, createDecipheriv } from "crypto";
import { promisify } from "util";
const scrypt = promisify(scryptCb);

const RPC = process.env.STARKNET_RPC || "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo";
const ADDR = process.env.ACCOUNT_ADDRESS;
const KS = process.env.KEYSTORE_PATH;
const PW = process.env.KEYSTORE_PASSWORD;
const CONTRACT = process.env.CONTRACT;
const PROOF_DIR = process.env.PROOF_DIR;

if (!ADDR || !KS || !PW || !CONTRACT || !PROOF_DIR) {
  console.error("Need ACCOUNT_ADDRESS, KEYSTORE_PATH, KEYSTORE_PASSWORD, CONTRACT, PROOF_DIR");
  process.exit(1);
}

async function decryptKeystore(path, password) {
  const ks = JSON.parse(readFileSync(path, "utf-8"));
  const c = ks.crypto;
  const params = c.kdfparams;
  const dk = await scrypt(Buffer.from(password, "utf-8"), Buffer.from(params.salt, "hex"), params.dklen, {
    N: params.n, r: params.r, p: params.p, maxmem: 512 * 1024 * 1024,
  });
  const ciphertext = Buffer.from(c.ciphertext, "hex");
  const macReal = Buffer.from(keccak_256(Buffer.concat([dk.slice(16, 32), ciphertext]))).toString("hex");
  if (macReal !== c.mac) throw new Error("MAC mismatch");
  const iv = Buffer.from(c.cipherparams.iv, "hex");
  const decipher = createDecipheriv("aes-128-ctr", dk.slice(0, 16), iv);
  return "0x" + Buffer.concat([decipher.update(ciphertext), decipher.final()]).toString("hex");
}

function loadStepProof(proofDir, stepFile) {
  const proofPath = pathResolve(proofDir, stepFile);
  return JSON.parse(readFileSync(proofPath, "utf-8"));
}

function decodeProofMetadata(proof) {
  const r = proof.recursive_proof;
  if (!r?.calldata?.length) throw new Error("missing recursive_proof.calldata in " + proof.model_id);
  return {
    calldata: r.calldata,
    circuit_hash: r.circuit_hash,
    weight_super_root: r.weight_super_root,
    io_commitment_param: r.calldata[19],         // proof body's io_commitment_felt252
    n_layers:        parseInt(r.calldata[12], 16),
    trace_log_size:  parseInt(r.calldata[22], 16),
    n_pos_perms:     parseInt(r.calldata[13], 16),
    prev_kv:         r.calldata[24],             // NEW v4 header field
    new_kv:          r.calldata[25],             // NEW v4 header field
    policy_commitment: proof.policy_commitment || r.policy_commitment || "0x0",
  };
}

async function ensureModelRegistered(account, provider, manifest) {
  const probe = await provider.callContract({
    contractAddress: CONTRACT,
    entrypoint: "get_recursive_model_info",
    calldata: [manifest.model_id],
  }).catch(() => null);

  const alreadyRegistered = probe && (probe[0] || "0x0").toLowerCase() !== "0x0";
  if (alreadyRegistered) {
    console.log("Model already registered, skipping");
    return;
  }

  console.log("Registering model on v4 (with level1_proof_hash + KV-aware metadata)...");
  // Use first step's n_pos_perms as expected (registration metadata).
  const firstStep = loadStepProof(PROOF_DIR, manifest.steps[0].proof_file);
  const firstMeta = decodeProofMetadata(firstStep);

  const tx = await account.execute({
    contractAddress: CONTRACT,
    entrypoint: "register_model_recursive",
    calldata: CallData.compile({
      model_id: manifest.model_id,
      circuit_hash: manifest.circuit_hash || firstMeta.circuit_hash,
      weight_super_root: manifest.weight_super_root || firstMeta.weight_super_root,
      policy_commitment: manifest.policy_commitment || "0x0",
      n_matmuls: parseInt(process.env.N_MATMULS || manifest.n_matmuls || 6),
      hidden_size: parseInt(process.env.HIDDEN_SIZE || manifest.hidden_size || 576),
      num_transformer_blocks: parseInt(process.env.NUM_TRANSFORMER_BLOCKS || manifest.num_transformer_blocks || 1),
      expected_n_poseidon_perms: firstMeta.n_pos_perms,
      level1_proof_hash: manifest.level1_proof_hash || "0x0",
    }),
  });
  await provider.waitForTransaction(tx.transaction_hash);
  console.log("  registered tx:", tx.transaction_hash);
}

async function startSession(account, provider, modelId, initialKvCommitment) {
  console.log("Starting decode session (initial_kv=" + initialKvCommitment.slice(0, 18) + "...)");
  const tx = await account.execute({
    contractAddress: CONTRACT,
    entrypoint: "start_decode_session",
    calldata: CallData.compile({
      model_id: modelId,
      initial_kv_commitment: initialKvCommitment,
    }),
  });
  const receipt = await provider.waitForTransaction(tx.transaction_hash);
  console.log("  start tx:", tx.transaction_hash);

  // Find DecodeSessionStarted event. The event's #[key] fields appear in
  // ev.keys after the event selector at keys[0]:
  //   keys = [selector, session_id, model_id]
  //   data = [initiator, started_at, initial_kv_commitment]
  const events = receipt.events || [];
  const norm = (s) => (s || "").toLowerCase().replace(/^0x0*/, "0x");
  for (const ev of events) {
    if (norm(ev.from_address) !== norm(CONTRACT)) continue;
    if (!ev.keys || ev.keys.length < 2) continue;
    const sessionId = BigInt(ev.keys[1]);
    if (sessionId > 0n) return { sessionId, tx: tx.transaction_hash };
  }
  throw new Error("could not parse session_id from start_decode_session events");
}

async function verifyDecodeStep(account, provider, sessionId, stepIdx, manifest, stepProof) {
  const meta = decodeProofMetadata(stepProof);
  console.log("Step " + stepIdx + ": prev_kv=" + meta.prev_kv.slice(0, 18) + "... new_kv=" + meta.new_kv.slice(0, 18) + "... felts=" + meta.calldata.length);

  const tx = await account.execute({
    contractAddress: CONTRACT,
    entrypoint: "verify_decode_step",
    calldata: CallData.compile({
      session_id: sessionId.toString(),
      expected_step_idx: stepIdx,
      model_id: manifest.model_id,
      io_commitment: meta.io_commitment_param,
      circuit_hash: meta.circuit_hash,
      weight_super_root: meta.weight_super_root,
      n_layers: meta.n_layers,
      n_matmuls: parseInt(process.env.N_MATMULS || manifest.n_matmuls || 6),
      hidden_size: parseInt(process.env.HIDDEN_SIZE || manifest.hidden_size || 576),
      num_transformer_blocks: parseInt(process.env.NUM_TRANSFORMER_BLOCKS || manifest.num_transformer_blocks || 1),
      policy_commitment: meta.policy_commitment,
      trace_log_size: meta.trace_log_size,
      stark_proof_data: meta.calldata,
    }),
  });
  const receipt = await provider.waitForTransaction(tx.transaction_hash);
  if (receipt.execution_status === "REVERTED") {
    throw new Error("step " + stepIdx + " REVERTED: " + (receipt.revert_reason || "").slice(0, 400));
  }
  return tx.transaction_hash;
}

async function finalizeSession(account, provider, sessionId) {
  console.log("Finalizing session...");
  const tx = await account.execute({
    contractAddress: CONTRACT,
    entrypoint: "finalize_decode_session",
    calldata: CallData.compile({ session_id: sessionId.toString() }),
  });
  await provider.waitForTransaction(tx.transaction_hash);
  return tx.transaction_hash;
}

async function main() {
  const PRIV = await decryptKeystore(KS, PW);
  console.log("keystore decrypted; pubkey=" + ec.starkCurve.getStarkKey(PRIV).slice(0, 18) + "...");
  const provider = new RpcProvider({ nodeUrl: RPC });
  const account = new Account(provider, ADDR, PRIV);

  const manifestPath = pathResolve(PROOF_DIR, "chain_manifest.json");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf-8"));
  console.log("Chain manifest: " + manifest.steps.length + " steps, model " + manifest.model_id);

  await ensureModelRegistered(account, provider, manifest);
  const initialKv = manifest.initial_kv_commitment || "0x0";
  const { sessionId, tx: startTx } = await startSession(account, provider, manifest.model_id, initialKv);
  console.log("Session ID:", sessionId.toString());

  const stepTxs = [];
  for (const step of manifest.steps) {
    const stepProof = loadStepProof(PROOF_DIR, step.proof_file);
    const txHash = await verifyDecodeStep(account, provider, sessionId, step.step_idx, manifest, stepProof);
    stepTxs.push({ step_idx: step.step_idx, tx_hash: txHash });
    console.log("  step " + step.step_idx + " tx:", txHash);
  }

  const finalizeTx = await finalizeSession(account, provider, sessionId);
  console.log("");
  console.log("================================================================");
  console.log("  MULTI-TOKEN DECODE CHAIN VERIFIED ON-CHAIN");
  console.log("================================================================");
  console.log("  Session ID: " + sessionId.toString());
  console.log("  Steps:      " + manifest.steps.length);
  console.log("  Start TX:   " + startTx);
  for (const s of stepTxs) console.log("  Step " + s.step_idx + " TX: " + s.tx_hash);
  console.log("  Finalize TX:" + finalizeTx);
  console.log("  Explorer:   https://sepolia.starkscan.co/tx/" + finalizeTx);
  console.log("================================================================");

  console.log("RESULT_JSON:" + JSON.stringify({
    success: true,
    session_id: sessionId.toString(),
    contract: CONTRACT,
    start_tx: startTx,
    step_txs: stepTxs,
    finalize_tx: finalizeTx,
    explorer_url: "https://sepolia.starkscan.co/tx/" + finalizeTx,
  }));
}

main().catch((e) => {
  console.error("Fatal:", (e.message || "").slice(0, 800));
  console.log("RESULT_JSON:" + JSON.stringify({ success: false, error: (e.message || "").slice(0, 800) }));
  process.exit(1);
});
