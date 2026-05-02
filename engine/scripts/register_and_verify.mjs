#!/usr/bin/env node
// Register model + verify_recursive on an existing contract.
// Required env: KEYSTORE_PATH, KEYSTORE_PASSWORD, ACCOUNT_ADDRESS, CONTRACT, PROOF_FILE
// Optional env: N_MATMULS, HIDDEN_SIZE, NUM_TRANSFORMER_BLOCKS

import { Account, RpcProvider, CallData, ec } from "starknet";
import { keccak_256 } from "@noble/hashes/sha3";
import { readFileSync } from "fs";
import { scrypt as scryptCb, createDecipheriv } from "crypto";
import { promisify } from "util";
const scrypt = promisify(scryptCb);

const RPC = process.env.STARKNET_RPC || "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo";
const ADDR = process.env.ACCOUNT_ADDRESS;
const KS = process.env.KEYSTORE_PATH;
const PW = process.env.KEYSTORE_PASSWORD;
const CONTRACT = process.env.CONTRACT;
const PROOF_FILE = process.env.PROOF_FILE;
const N_MATMULS = parseInt(process.env.N_MATMULS || "210");
const HIDDEN_SIZE = parseInt(process.env.HIDDEN_SIZE || "576");
const NUM_BLOCKS = parseInt(process.env.NUM_TRANSFORMER_BLOCKS || "30");

if (!ADDR || !KS || !PW || !CONTRACT || !PROOF_FILE) {
  console.error("Need ACCOUNT_ADDRESS, KEYSTORE_PATH, KEYSTORE_PASSWORD, CONTRACT, PROOF_FILE");
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

async function main() {
  const PRIV = await decryptKeystore(KS, PW);
  console.log("keystore decrypted; pubkey=" + ec.starkCurve.getStarkKey(PRIV).slice(0, 18) + "...");
  const provider = new RpcProvider({ nodeUrl: RPC });
  const account = new Account(provider, ADDR, PRIV);

  const raw = JSON.parse(readFileSync(PROOF_FILE, "utf-8"));
  const r = raw.recursive_proof;
  const ioCommitment = r.calldata[19];
  const nLayers = parseInt(r.calldata[12], 16);
  const nPosPerms = parseInt(r.calldata[13], 16);
  const traceLog = parseInt(r.calldata[22], 16);

  console.log("Contract:    " + CONTRACT);
  console.log("Model ID:    " + raw.model_id);
  console.log("circuit_hash:" + r.circuit_hash);
  console.log("weight_root: " + r.weight_super_root);
  console.log("Proof:       " + r.calldata.length + " felts (n_layers=" + nLayers + ", n_pos_perms=" + nPosPerms + ", trace_log=" + traceLog + ")");
  console.log("Architecture:n_matmuls=" + N_MATMULS + " hidden_size=" + HIDDEN_SIZE + " num_blocks=" + NUM_BLOCKS);

  // Step 1: register if not already
  let needsRegister = true;
  try {
    const info = await provider.callContract({
      contractAddress: CONTRACT,
      entrypoint: "get_recursive_model_info",
      calldata: [raw.model_id],
    });
    needsRegister = (info[0] || "0x0").toLowerCase() === "0x0";
  } catch {}

  if (needsRegister) {
    console.log("Registering model...");
    const regTx = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "register_model_recursive",
      calldata: CallData.compile({
        model_id: raw.model_id,
        circuit_hash: r.circuit_hash,
        weight_super_root: r.weight_super_root,
        policy_commitment: raw.policy_commitment || r.policy_commitment || "0x0",
        n_matmuls: N_MATMULS,
        hidden_size: HIDDEN_SIZE,
        num_transformer_blocks: NUM_BLOCKS,
        expected_n_poseidon_perms: nPosPerms,
      }),
    });
    await provider.waitForTransaction(regTx.transaction_hash);
    console.log("  registered tx:", regTx.transaction_hash);
  } else {
    console.log("Model already registered, skipping");
  }

  // Step 2: verify_recursive
  console.log("Submitting verify_recursive...");
  const tx = await account.execute({
    contractAddress: CONTRACT,
    entrypoint: "verify_recursive",
    calldata: CallData.compile({
      model_id: raw.model_id,
      io_commitment: ioCommitment,
      circuit_hash: r.circuit_hash,
      weight_super_root: r.weight_super_root,
      n_layers: nLayers,
      n_matmuls: N_MATMULS,
      hidden_size: HIDDEN_SIZE,
      num_transformer_blocks: NUM_BLOCKS,
      policy_commitment: raw.policy_commitment || r.policy_commitment || "0x0",
      trace_log_size: traceLog,
      stark_proof_data: r.calldata,
    }),
  });
  console.log("  verify tx:", tx.transaction_hash);
  const receipt = await provider.waitForTransaction(tx.transaction_hash);
  if (receipt.execution_status === "REVERTED") {
    console.log("REVERTED:", (receipt.revert_reason || "").slice(0, 600));
    console.log("RESULT_JSON:" + JSON.stringify({ success: false, tx_hash: tx.transaction_hash, error: (receipt.revert_reason || "").slice(0, 600) }));
    process.exit(1);
  }
  const explorer = "https://sepolia.starkscan.co/tx/" + tx.transaction_hash;
  console.log("");
  console.log("================================================================");
  console.log("  FULL " + NUM_BLOCKS + "-LAYER MODEL VERIFIED ON-CHAIN — 1 TX");
  console.log("================================================================");
  console.log("  TX:       " + tx.transaction_hash);
  console.log("  Felts:    " + r.calldata.length);
  console.log("  GKR layers (incl. embed/norm/etc): " + nLayers);
  console.log("  Poseidon perms in chain: " + nPosPerms);
  console.log("  Explorer: " + explorer);
  console.log("================================================================");
  console.log("RESULT_JSON:" + JSON.stringify({ success: true, tx_hash: tx.transaction_hash, explorer_url: explorer, felts: r.calldata.length, n_layers: nLayers }));
}

main().catch((e) => {
  console.error("Fatal:", (e.message || "").slice(0, 800));
  console.log("RESULT_JSON:" + JSON.stringify({ success: false, error: (e.message || "").slice(0, 800) }));
  process.exit(1);
});
