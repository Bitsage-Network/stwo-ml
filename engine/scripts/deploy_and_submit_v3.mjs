#!/usr/bin/env node
// Deploy fresh recursive verifier from current Cairo source + register + submit.
// Required env: KEYSTORE_PATH, KEYSTORE_PASSWORD, ACCOUNT_ADDRESS, PROOF_FILE
// Optional env: STARKNET_RPC

import { Account, RpcProvider, CallData, hash, json, ec } from "starknet";
import { keccak_256 } from "@noble/hashes/sha3";
import { readFileSync } from "fs";
import { scrypt as scryptCb, createDecipheriv } from "crypto";
import { promisify } from "util";
const scrypt = promisify(scryptCb);

// PublicNode RPC bypasses Alchemy's 10K-felt cap (sierra is 29K felts during declare).
// PublicNode lacks 'pending' tag support; starknet.js account flow needs that.
// Use Alchemy for getNonce/estimateFee (small calls) and PublicNode for the heavy declare/verify.
const RPC = process.env.STARKNET_RPC || "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo";
const RPC_NOCAP = process.env.STARKNET_RPC_NOCAP || "https://starknet-sepolia-rpc.publicnode.com";
const ADDR = process.env.ACCOUNT_ADDRESS;
const KS = process.env.KEYSTORE_PATH;
const PW = process.env.KEYSTORE_PASSWORD;
const PROOF_FILE = process.env.PROOF_FILE;

if (!ADDR || !KS || !PW || !PROOF_FILE) {
  console.error("Need ACCOUNT_ADDRESS, KEYSTORE_PATH, KEYSTORE_PASSWORD, PROOF_FILE");
  process.exit(1);
}

const SIERRA_PATH = "/Users/vaamx/bitsage-network/libs/elo-cairo-verifier/target/release/elo_cairo_verifier_RecursiveVerifierContract.contract_class.json";
const CASM_PATH   = "/Users/vaamx/bitsage-network/libs/elo-cairo-verifier/target/release/elo_cairo_verifier_RecursiveVerifierContract.compiled_contract_class.json";

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
  const pub = ec.starkCurve.getStarkKey(PRIV);
  console.log("keystore decrypted; pubkey=" + pub.slice(0, 18) + "...");
  console.log("RPC: " + RPC);

  const provider = new RpcProvider({ nodeUrl: RPC });
  const account = new Account(provider, ADDR, PRIV);

  const sierra = json.parse(readFileSync(SIERRA_PATH, "utf-8"));
  const casm = json.parse(readFileSync(CASM_PATH, "utf-8"));
  const classHash = hash.computeContractClassHash(sierra);
  console.log("Class hash:", classHash);

  // Step 1: declare — assume done externally (sncast). Just verify the class exists.
  if (process.env.SKIP_DECLARE !== "1") {
    try {
      await provider.getClassByHash(classHash);
      console.log("Class already declared on chain");
    } catch (e) {
      console.error("Class not found on chain. Run sncast declare first or set SKIP_DECLARE=1.");
      console.error("Class:", classHash);
      process.exit(1);
    }
  }

  // Step 2: deploy a new contract instance
  console.log("Deploying new contract instance...");
  const deploy = await account.deployContract({
    classHash,
    constructorCalldata: CallData.compile({ owner: ADDR }),
    salt: "0x" + Date.now().toString(16),
  });
  await provider.waitForTransaction(deploy.transaction_hash);
  const CONTRACT = deploy.contract_address;
  console.log("  deploy tx:", deploy.transaction_hash);
  console.log("  contract: ", CONTRACT);

  // Step 3: parse proof
  const raw = JSON.parse(readFileSync(PROOF_FILE, "utf-8"));
  const r = raw.recursive_proof;
  const ioCommitment = r.calldata[19];
  const nLayers = parseInt(r.calldata[12], 16);
  const nPosPerms = parseInt(r.calldata[13], 16);
  const traceLog = parseInt(r.calldata[22], 16);
  console.log("Proof: " + r.calldata.length + " felts, n_layers=" + nLayers + ", n_pos_perms=" + nPosPerms + ", trace_log=" + traceLog);

  // Step 4: register model (8-param v2 signature)
  console.log("Registering model...");
  const regTx = await account.execute({
    contractAddress: CONTRACT,
    entrypoint: "register_model_recursive",
    calldata: CallData.compile({
      model_id: raw.model_id,
      circuit_hash: r.circuit_hash,
      weight_super_root: r.weight_super_root,
      policy_commitment: raw.policy_commitment || r.policy_commitment || "0x0",
      n_matmuls: 6,
      hidden_size: 576,
      num_transformer_blocks: 1,
      expected_n_poseidon_perms: nPosPerms,
    }),
  });
  await provider.waitForTransaction(regTx.transaction_hash);
  console.log("  registered tx:", regTx.transaction_hash);

  // Step 5: submit verify_recursive
  console.log("Submitting verify_recursive...");
  const verifyCalldata = CallData.compile({
    model_id: raw.model_id,
    io_commitment: ioCommitment,
    circuit_hash: r.circuit_hash,
    weight_super_root: r.weight_super_root,
    n_layers: nLayers,
    n_matmuls: 6,
    hidden_size: 576,
    num_transformer_blocks: 1,
    policy_commitment: raw.policy_commitment || r.policy_commitment || "0x0",
    trace_log_size: traceLog,
    stark_proof_data: r.calldata,
  });
  const verifyTx = await account.execute({
    contractAddress: CONTRACT,
    entrypoint: "verify_recursive",
    calldata: verifyCalldata,
  });
  console.log("  verify tx:", verifyTx.transaction_hash);
  const receipt = await provider.waitForTransaction(verifyTx.transaction_hash);
  if (receipt.execution_status === "REVERTED") {
    console.log("  REVERTED:", (receipt.revert_reason || "").slice(0, 600));
    console.log("RESULT_JSON:" + JSON.stringify({ success: false, contract: CONTRACT, class: classHash, tx_hash: verifyTx.transaction_hash, error: (receipt.revert_reason || "").slice(0, 600) }));
    process.exit(1);
  }
  console.log("");
  console.log("================================================================");
  console.log("  RECURSIVE STARK VERIFIED ON-CHAIN — 1 TX");
  console.log("================================================================");
  console.log("  Class:    " + classHash);
  console.log("  Contract: " + CONTRACT);
  console.log("  TX:       " + verifyTx.transaction_hash);
  console.log("  Felts:    " + r.calldata.length);
  console.log("  Explorer: https://sepolia.starkscan.co/tx/" + verifyTx.transaction_hash);
  console.log("================================================================");
  console.log("RESULT_JSON:" + JSON.stringify({ success: true, contract: CONTRACT, class: classHash, tx_hash: verifyTx.transaction_hash, explorer_url: "https://sepolia.starkscan.co/tx/" + verifyTx.transaction_hash, felts: r.calldata.length }));
}

main().catch((e) => {
  console.error("Fatal:", (e.message || "").slice(0, 800));
  console.log("RESULT_JSON:" + JSON.stringify({ success: false, error: (e.message || "").slice(0, 800) }));
  process.exit(1);
});
