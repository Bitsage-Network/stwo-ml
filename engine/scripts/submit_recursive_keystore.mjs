#!/usr/bin/env node
// In-process wrapper: decrypt keystore + submit recursive proof.
// Private key never leaves the JS heap (no env/argv/file).
//
// Required env: KEYSTORE_PATH, KEYSTORE_PASSWORD, ACCOUNT_ADDRESS
// Optional env: RECURSIVE_CONTRACT, STARKNET_RPC

import { Account, RpcProvider, CallData, ec } from "starknet";
import { keccak_256 } from "@noble/hashes/sha3";
import { readFileSync } from "fs";
import { scrypt as scryptCb } from "crypto";
import { promisify } from "util";
const scrypt = promisify(scryptCb);

const RPC = process.env.STARKNET_RPC || "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo";
// PublicNode bypasses Alchemy/Cartridge's 10K-felt RPC-side calldata cap. Used only for the verify_recursive submit.
const RPC_NOCAP = process.env.STARKNET_RPC_NOCAP || "https://starknet-sepolia-rpc.publicnode.com";
const CONTRACT = process.env.RECURSIVE_CONTRACT || "0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7";
const ADDR = process.env.ACCOUNT_ADDRESS;
const KS = process.env.KEYSTORE_PATH;
const PW = process.env.KEYSTORE_PASSWORD;

if (!ADDR || !KS || !PW) {
  console.error("Need ACCOUNT_ADDRESS, KEYSTORE_PATH, KEYSTORE_PASSWORD");
  process.exit(1);
}
const proofPath = process.argv[2];
if (!proofPath) { console.error("Usage: submit_recursive_keystore.mjs <proof.json>"); process.exit(1); }

function packQm31ToFelt252(limbs) {
  const shift = 1n << 31n;
  const parse = (x) => BigInt(typeof x === "string" ? x : "0x" + x.toString(16));
  let r = parse(limbs[0]);
  r = r * shift + parse(limbs[1]);
  r = r * shift + parse(limbs[2]);
  r = r * shift + parse(limbs[3]);
  return "0x" + r.toString(16);
}

async function decryptKeystore(path, password) {
  const ks = JSON.parse(readFileSync(path, "utf-8"));
  const c = ks.crypto;
  const params = c.kdfparams;
  const salt = Buffer.from(params.salt, "hex");
  const dk = await scrypt(Buffer.from(password, "utf-8"), salt, params.dklen, {
    N: params.n, r: params.r, p: params.p, maxmem: 512 * 1024 * 1024,
  });
  const ciphertext = Buffer.from(c.ciphertext, "hex");
  const macReal = Buffer.from(keccak_256(Buffer.concat([dk.slice(16, 32), ciphertext]))).toString("hex");
  if (macReal !== c.mac) throw new Error("keystore MAC mismatch (wrong password?)");
  const iv = Buffer.from(c.cipherparams.iv, "hex");
  const { createDecipheriv } = await import("crypto");
  const decipher = createDecipheriv("aes-128-ctr", dk.slice(0, 16), iv);
  const plain = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
  return "0x" + plain.toString("hex");
}

async function main() {
  const PRIV = await decryptKeystore(KS, PW);
  const pub = ec.starkCurve.getStarkKey(PRIV);
  console.log("Decrypted keystore. pubkey=" + pub.slice(0, 18) + "...");

  const provider = new RpcProvider({ nodeUrl: RPC });
  const account = new Account(provider, ADDR, PRIV);

  const raw = JSON.parse(readFileSync(proofPath, "utf-8"));
  const modelId = raw.model_id || raw.verify_calldata?.model_id || "0x1";
  const ioCommitment = raw.io_commitment || "0x1";
  const recursive = raw.recursive_proof;
  if (!recursive?.calldata?.length) { console.error("No recursive_proof.calldata"); process.exit(1); }
  const calldata = recursive.calldata;
  const circuitHash = recursive.circuit_hash || packQm31ToFelt252(calldata.slice(0, 4));
  const weightSuperRoot = recursive.weight_super_root || packQm31ToFelt252(calldata.slice(8, 12));

  console.log("Contract:      " + CONTRACT);
  console.log("Account:       " + ADDR);
  console.log("Model ID:      " + modelId);
  console.log("IO Commitment: " + ioCommitment);
  console.log("Circuit Hash:  " + circuitHash);
  console.log("Weight Root:   " + weightSuperRoot);
  console.log("Calldata:      " + calldata.length + " felts");

  // Step 1: register if needed
  let needsRegister = false;
  try {
    const info = await provider.callContract({
      contractAddress: CONTRACT,
      entrypoint: "get_recursive_model_info",
      calldata: [modelId],
    });
    const chainRoot = (info[1] || "0x0").toLowerCase();
    const proofRoot = weightSuperRoot.toLowerCase();
    needsRegister = chainRoot === "0x0" || (chainRoot !== proofRoot && proofRoot !== "0x0");
    console.log("Register check: chainRoot=" + chainRoot.slice(0, 18) + "... proofRoot=" + proofRoot.slice(0, 18) + "... needsRegister=" + needsRegister);
  } catch (e) {
    console.log("get_recursive_model_info failed (treating as not-registered): " + (e.message || "").slice(0, 120));
    needsRegister = true;
  }

  if (needsRegister) {
    process.stdout.write("Registering:   ");
    // v2 contract requires 8 params; v1 takes 4. Detect by trying long form first.
    const longCalldata = CallData.compile({
      model_id: modelId,
      circuit_hash: circuitHash,
      weight_super_root: weightSuperRoot,
      policy_commitment: raw.policy_commitment || recursive.policy_commitment || "0x0",
      n_matmuls: 6,
      hidden_size: 576,
      num_transformer_blocks: 1,
      expected_n_poseidon_perms: parseInt(recursive.calldata?.[13] || "0x420", 16) || 1056,
    });
    try {
      const regTx = await account.execute({
        contractAddress: CONTRACT,
        entrypoint: "register_model_recursive",
        calldata: longCalldata,
      });
      await provider.waitForTransaction(regTx.transaction_hash);
      console.log("done tx=" + regTx.transaction_hash);
    } catch (e) {
      const m = (e.message || "").toLowerCase();
      if (m.includes("already") || m.includes("registered")) console.log("skip (already registered)");
      else { console.error("FAILED: " + (e.message || "").slice(0, 400)); process.exit(1); }
    }
  } else {
    console.log("Register:      skip (already up to date)");
  }

  // Step 2: verify_recursive
  const metadata = raw.metadata || {};

  // Pre-flight: simulate the call directly via RPC so we can see the real revert reason.
  if (process.env.SIMULATE_FIRST === "1") {
    process.stdout.write("Simulating:    ");
    try {
      const sim = await fetch(RPC, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0", id: 1, method: "starknet_call",
          params: [{
            contract_address: CONTRACT,
            entry_point_selector: "0x10cce72c6a97f20e432264e466c8f2d1f344a5ba920ef93333c899d9011fe5a",
            calldata: CallData.compile({
              model_id: modelId,
              io_commitment: ioCommitment,
              circuit_hash: circuitHash,
              weight_super_root: weightSuperRoot,
              n_layers: metadata.n_layers || 9,
              n_matmuls: metadata.n_matmuls || 6,
              hidden_size: metadata.hidden_size || 576,
              num_transformer_blocks: metadata.num_transformer_blocks || 1,
              policy_commitment: raw.policy_commitment || recursive.policy_commitment || "0x0",
              trace_log_size: metadata.trace_log_size || recursive.log_size || 11,
              stark_proof_data: calldata,
            }),
          }, "latest"]
        }),
      });
      const j = await sim.json();
      if (j.error) {
        const data = typeof j.error.data === "string" ? j.error.data : JSON.stringify(j.error.data || {});
        console.log("REVERT");
        console.log("  code:    " + j.error.code);
        console.log("  message: " + (j.error.message || "").slice(0, 200));
        console.log("  data:    " + data.slice(0, 800));
        process.exit(2);
      } else {
        console.log("simulate OK; would return " + (j.result?.length || 0) + " felts");
      }
    } catch (e) {
      console.log("simulate fetch error: " + (e.message || "").slice(0, 200));
      process.exit(3);
    }
  }

  process.stdout.write("Submitting:    ");
  const verifyCalldata = CallData.compile({
    model_id: modelId,
    io_commitment: ioCommitment,
    circuit_hash: circuitHash,
    weight_super_root: weightSuperRoot,
    n_layers: metadata.n_layers || 9,
    n_matmuls: metadata.n_matmuls || 6,
    hidden_size: metadata.hidden_size || 576,
    num_transformer_blocks: metadata.num_transformer_blocks || 1,
    policy_commitment: raw.policy_commitment || recursive.policy_commitment || "0x0",
    trace_log_size: metadata.trace_log_size || recursive.log_size || 15,
    stark_proof_data: calldata,
  });

  // Get nonce from a 'latest'-only RPC (PublicNode for big-calldata path), then submit.
  // starknet.js v7 default uses 'pending' which PublicNode rejects, so we set nonce manually.
  const submitProvider = new RpcProvider({ nodeUrl: RPC_NOCAP });
  const submitAccount = new Account(submitProvider, ADDR, PRIV);
  const nonceHex = await submitProvider.getNonceForAddress(ADDR, "latest");
  console.log("(via " + RPC_NOCAP.replace(/^https?:\/\//, "").slice(0, 30) + ", nonce=" + nonceHex + ")");

  // Conservative resource bounds for a ~35K-felt verify call.
  // L1 data gas dominates (calldata-on-chain costs).
  const resourceBounds = {
    l1_gas: { max_amount: "0x100000", max_price_per_unit: "0x100000000000" },
    l2_gas: { max_amount: "0x40000000", max_price_per_unit: "0x100000000" },
    l1_data_gas: { max_amount: "0x100000", max_price_per_unit: "0x100000000000" },
  };

  const tx = await submitAccount.execute(
    [{ contractAddress: CONTRACT, entrypoint: "verify_recursive", calldata: verifyCalldata }],
    { nonce: nonceHex, resourceBounds, skipValidate: true, version: 3, blockIdentifier: "latest" }
  );
  console.log("tx=" + tx.transaction_hash);

  process.stdout.write("Confirming:    ");
  const receipt = await submitProvider.waitForTransaction(tx.transaction_hash);
  if (receipt.execution_status === "REVERTED") {
    console.log("REVERTED");
    console.error("Revert reason: " + (receipt.revert_reason || "").slice(0, 600));
    console.log("RESULT_JSON:" + JSON.stringify({ success: false, tx_hash: tx.transaction_hash, error: (receipt.revert_reason || "").slice(0, 600) }));
    process.exit(1);
  }
  console.log("confirmed");

  let verificationCount = 0;
  try {
    const c = await provider.callContract({ contractAddress: CONTRACT, entrypoint: "get_recursive_verification_count", calldata: [modelId] });
    verificationCount = Number(BigInt(c[0]));
  } catch {}

  const explorer = "https://sepolia.starkscan.co/tx/" + tx.transaction_hash;
  console.log("");
  console.log("================================================================");
  console.log("  RECURSIVE STARK VERIFIED ON-CHAIN");
  console.log("================================================================");
  console.log("  TX:            " + tx.transaction_hash);
  console.log("  Felts:         " + calldata.length);
  console.log("  Verifications: " + verificationCount);
  console.log("  Explorer:      " + explorer);
  console.log("================================================================");
  console.log("RESULT_JSON:" + JSON.stringify({ success: true, tx_hash: tx.transaction_hash, explorer_url: explorer, verification_count: verificationCount, felts: calldata.length }));
}

main().catch((e) => {
  console.error("Fatal: " + (e.message || "").slice(0, 800));
  console.log("RESULT_JSON:" + JSON.stringify({ success: false, error: (e.message || "").slice(0, 800) }));
  process.exit(1);
});
