#!/usr/bin/env node
// Hand-rolled INVOKE_V3 raw submission — bypasses starknet.js's hardcoded 'pending'
// block tag so we can use RPCs (PublicNode) that lift the 10K-felt calldata cap.
//
// Required env: KEYSTORE_PATH, KEYSTORE_PASSWORD, ACCOUNT_ADDRESS
// Optional env: RPC_URL (default PublicNode), CONTRACT (default v1 recursive verifier)

import { hash, ec, CallData, transaction, num, constants } from "starknet";
import { keccak_256 } from "@noble/hashes/sha3";
import { readFileSync } from "fs";
import { scrypt as scryptCb, createDecipheriv } from "crypto";
import { promisify } from "util";
const scrypt = promisify(scryptCb);

const RPC = process.env.RPC_URL || "https://starknet-sepolia-rpc.publicnode.com";
const CONTRACT = process.env.CONTRACT || "0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7";
const ADDR = process.env.ACCOUNT_ADDRESS;
const KS = process.env.KEYSTORE_PATH;
const PW = process.env.KEYSTORE_PASSWORD;

if (!ADDR || !KS || !PW) { console.error("Need ACCOUNT_ADDRESS, KEYSTORE_PATH, KEYSTORE_PASSWORD"); process.exit(1); }
const proofPath = process.argv[2];
if (!proofPath) { console.error("Usage: submit_recursive_raw.mjs <proof.json>"); process.exit(1); }

async function rpc(method, params) {
  const r = await fetch(RPC, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
  });
  return r.json();
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
  if (macReal !== c.mac) throw new Error("MAC mismatch (wrong password)");
  const iv = Buffer.from(c.cipherparams.iv, "hex");
  const decipher = createDecipheriv("aes-128-ctr", dk.slice(0, 16), iv);
  return "0x" + Buffer.concat([decipher.update(ciphertext), decipher.final()]).toString("hex");
}

// V3 resource bounds for big-calldata verify_recursive call.
// Prices are denominated in fri (STRK base unit). Sequencer charges actual×actual_price.
// Budgets target ~2 STRK worst case (deployer has ~7.5 STRK).
async function resourceBounds() {
  // Query current prices and bid 2x.
  const blk = await rpc("starknet_getBlockWithTxHashes", ["latest"]);
  const p = blk.result;
  const l1Cur = BigInt(p.l1_gas_price.price_in_fri);
  const l1dCur = BigInt(p.l1_data_gas_price.price_in_fri);
  const l2Cur = BigInt(p.l2_gas_price.price_in_fri);
  const toHex = (b) => "0x" + b.toString(16);
  return {
    // l1_gas: only used for L2->L1 messages, none here. Tiny budget.
    l1_gas:      { max_amount: "0x1000",      max_price_per_unit: toHex(l1Cur * 2n) },
    // l2_gas: dominant cost — STARK verification compute. Bid 200M units at 2x current price.
    l2_gas:      { max_amount: "0xbebc200",   max_price_per_unit: toHex(l2Cur * 2n) },
    // l1_data_gas: charged per calldata felt that's posted to L1. ~36K felts → bid 100K.
    l1_data_gas: { max_amount: "0x186a0",     max_price_per_unit: toHex(l1dCur * 2n) },
  };
}

async function main() {
  const PRIV = await decryptKeystore(KS, PW);
  const pub = ec.starkCurve.getStarkKey(PRIV);
  console.log("keystore decrypted; pubkey=" + pub.slice(0, 18) + "...");
  console.log("RPC: " + RPC);
  console.log("Contract: " + CONTRACT);
  console.log("Account:  " + ADDR);

  const raw = JSON.parse(readFileSync(proofPath, "utf-8"));
  const r = raw.recursive_proof;
  if (!r?.calldata?.length) { console.error("No recursive_proof.calldata"); process.exit(1); }

  // The contract checks `io_commitment param == proof_io_commitment_felt252` where
  // proof_io_commitment_felt252 is calldata[19] (the lossy QM31->felt252 conversion stored
  // in the proof header). Top-level j.io_commitment is the full 252-bit Poseidon hash —
  // different from what the proof body bound. Use proof body's value.
  const ioCommitmentParam = r.calldata[19];
  console.log("io_commitment (from proof header [19]): " + ioCommitmentParam);

  // Read n_layers + trace_log_size from proof body (cd[12] and cd[22]).
  // Other metadata (n_matmuls, hidden_size, num_transformer_blocks) is informational
  // — the contract doesn't cross-check it against the proof body, only stores it for events.
  const nLayersFromProof = parseInt(r.calldata[12], 16);
  const traceLogSizeFromProof = parseInt(r.calldata[22], 16);
  console.log(`metadata from proof body: n_layers=${nLayersFromProof}, trace_log_size=${traceLogSizeFromProof}`);

  const verifyCalldata = CallData.compile({
    model_id: raw.model_id,
    io_commitment: ioCommitmentParam,
    circuit_hash: r.circuit_hash,
    weight_super_root: r.weight_super_root,
    n_layers: nLayersFromProof,
    n_matmuls: parseInt(process.env.N_MATMULS || "6"),
    hidden_size: parseInt(process.env.HIDDEN_SIZE || "576"),
    num_transformer_blocks: parseInt(process.env.NUM_TRANSFORMER_BLOCKS || "1"),
    policy_commitment: raw.policy_commitment || r.policy_commitment || "0x0",
    trace_log_size: traceLogSizeFromProof,
    stark_proof_data: r.calldata,
  });

  const calls = [{
    contractAddress: CONTRACT,
    entrypoint: "verify_recursive",
    calldata: verifyCalldata,
  }];
  // Cairo 1 account __execute__ calldata: [n_calls, [{contract, selector, calldata_len, calldata...}]]
  const compiledCalldata = transaction.fromCallsToExecuteCalldata_cairo1(calls);
  console.log("execute calldata felts: " + compiledCalldata.length);

  // Get nonce via 'latest' (PublicNode-compatible)
  const nonceRes = await rpc("starknet_getNonce", ["latest", ADDR]);
  if (nonceRes.error) { console.error("getNonce err:", nonceRes.error); process.exit(1); }
  const nonce = nonceRes.result;
  console.log("nonce: " + nonce);

  const rb = await resourceBounds();
  console.log("resource bounds: l2_gas=" + rb.l2_gas.max_amount + " @ " + rb.l2_gas.max_price_per_unit);
  const args = {
    senderAddress: ADDR,
    version: "0x3",
    compiledCalldata,
    chainId: constants.StarknetChainId.SN_SEPOLIA,
    nonce,
    accountDeploymentData: [],
    nonceDataAvailabilityMode: 0,    // L1
    feeDataAvailabilityMode: 0,      // L1
    resourceBounds: rb,
    tip: "0x0",
    paymasterData: [],
  };
  const txHash = hash.calculateInvokeTransactionHash(args);
  console.log("computed tx hash: " + txHash);

  const sig = ec.starkCurve.sign(txHash, PRIV);
  // sig has .r .s as bigints
  const signature = ["0x" + sig.r.toString(16), "0x" + sig.s.toString(16)];

  // Build INVOKE_V3 envelope (RPC 0.8 format)
  const invokeTx = {
    type: "INVOKE",
    version: "0x3",
    sender_address: ADDR,
    calldata: compiledCalldata.map(v => "0x" + BigInt(v).toString(16)),
    signature,
    nonce,
    resource_bounds: rb,
    tip: "0x0",
    paymaster_data: [],
    account_deployment_data: [],
    nonce_data_availability_mode: "L1",
    fee_data_availability_mode: "L1",
  };

  console.log("submitting INVOKE_V3...");
  // RPC 0.8: params = [invoke_transaction] (positional array, flat envelope)
  const submitRes = await rpc("starknet_addInvokeTransaction", [invokeTx]);
  if (submitRes.error) {
    console.error("submit error:", JSON.stringify(submitRes.error).slice(0, 1000));
    console.log("RESULT_JSON:" + JSON.stringify({ success: false, error: JSON.stringify(submitRes.error).slice(0, 800) }));
    process.exit(1);
  }
  const txHashOnChain = submitRes.result.transaction_hash;
  console.log("submitted: " + txHashOnChain);

  // Poll for receipt
  process.stdout.write("Confirming");
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 5000));
    process.stdout.write(".");
    const rcpt = await rpc("starknet_getTransactionReceipt", [txHashOnChain]);
    if (rcpt.result) {
      console.log("");
      const r = rcpt.result;
      console.log("execution_status: " + r.execution_status);
      console.log("finality_status:  " + r.finality_status);
      if (r.execution_status === "REVERTED") {
        console.log("revert: " + (r.revert_reason || "").slice(0, 600));
        console.log("RESULT_JSON:" + JSON.stringify({ success: false, tx_hash: txHashOnChain, revert: (r.revert_reason || "").slice(0, 600) }));
        process.exit(1);
      }
      const explorer = "https://sepolia.starkscan.co/tx/" + txHashOnChain;
      console.log("\n================================================================");
      console.log("  RECURSIVE STARK VERIFIED ON-CHAIN");
      console.log("================================================================");
      console.log("  TX:       " + txHashOnChain);
      console.log("  Felts:    " + r.calldata?.length || "n/a");
      console.log("  Explorer: " + explorer);
      console.log("================================================================");
      console.log("RESULT_JSON:" + JSON.stringify({ success: true, tx_hash: txHashOnChain, explorer_url: explorer }));
      return;
    }
  }
  console.log("\ntimed out waiting for receipt");
  console.log("RESULT_JSON:" + JSON.stringify({ success: false, tx_hash: txHashOnChain, error: "receipt timeout" }));
}

main().catch(e => {
  console.error("Fatal:", (e.message || "").slice(0, 800));
  console.log("RESULT_JSON:" + JSON.stringify({ success: false, error: (e.message || "").slice(0, 800) }));
  process.exit(1);
});
