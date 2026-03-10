#!/usr/bin/env node
// Register model + open session + submit streaming calldata.
//
// Handles the full pipeline:
//   1. register_model_gkr (weight commitments + circuit descriptor)
//   2. register_model_gkr_streaming_circuit (layer tags)
//   3. open_gkr_session (creates session, gets session_id)
//   4. upload proof chunks → seal
//   5. Delegates to streaming_submit.mjs steps (init → output_mle → layers → weight_binding → input_mle → finalize)
//
// Usage:
//   node register_and_submit.mjs <proof.json> [--skip-register] [--skip-session]
//
// Env vars:
//   STARKNET_ACCOUNT, STARKNET_PRIVATE_KEY, AVNU_API_KEY
//   CONTRACT_ADDRESS (default: 0x0121d1...)

import { Account, RpcProvider, CallData, Signer } from "starknet";
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const RPC_URL = process.env.STARKNET_RPC;
if (!RPC_URL) { console.error("FATAL: STARKNET_RPC env var required"); process.exit(1); }
const CONTRACT =
  process.env.CONTRACT_ADDRESS ||
  "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005";
const ACCOUNT_ADDRESS = process.env.STARKNET_ACCOUNT;
if (!ACCOUNT_ADDRESS) { console.error("FATAL: STARKNET_ACCOUNT env var required"); process.exit(1); }
const PRIVATE_KEY = process.env.STARKNET_PRIVATE_KEY;
if (!PRIVATE_KEY) { console.error("FATAL: STARKNET_PRIVATE_KEY env var required"); process.exit(1); }

const args = process.argv.slice(2);
const proofPath = args.find((a) => !a.startsWith("--"));
const skipRegister = args.includes("--skip-register");
const skipSession = args.includes("--skip-session");

if (!proofPath) {
  console.error(
    "Usage: node register_and_submit.mjs <proof.json> [--skip-register] [--skip-session]"
  );
  process.exit(1);
}

const proof = JSON.parse(readFileSync(proofPath, "utf-8"));
const vc = proof.verify_calldata;

if (!vc || vc.mode !== "streaming") {
  console.error("Proof does not contain streaming calldata");
  process.exit(1);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function main() {
  const provider = new RpcProvider({ nodeUrl: RPC_URL });
  const signer = new Signer(PRIVATE_KEY);
  const account = new Account({ provider, address: ACCOUNT_ADDRESS, signer });

  const modelId = vc.model_id;
  console.log(`Model ID: ${modelId}`);
  console.log(`Contract: ${CONTRACT}`);

  // Step 1: Register model (weight commitments + circuit)
  if (!skipRegister) {
    // Check if already registered
    try {
      const result = await provider.callContract({
        contractAddress: CONTRACT,
        entrypoint: "get_model_gkr_weight_count",
        calldata: [modelId],
      });
      const count = Number(BigInt(result[0]));
      if (count > 0) {
        console.log(
          `Model already registered with ${count} weight commitments — skipping registration`
        );
      } else {
        throw new Error("not registered");
      }
    } catch {
      // Register
      const weightCommitments = proof.weight_commitments || [];
      const circuitDescriptor = vc.layer_tags || [];
      console.log(
        `Registering model: ${weightCommitments.length} weights, ${circuitDescriptor.length} tags`
      );
      const tx = await account.execute({
        contractAddress: CONTRACT,
        entrypoint: "register_model_gkr",
        calldata: CallData.compile({
          model_id: modelId,
          weight_commitments: weightCommitments,
          circuit_descriptor: circuitDescriptor,
        }),
      });
      console.log(`  register TX: ${tx.transaction_hash}`);
      await provider.waitForTransaction(tx.transaction_hash);
      console.log(`  registered.`);
    }

    // Register streaming circuit
    try {
      const tx = await account.execute({
        contractAddress: CONTRACT,
        entrypoint: "register_model_gkr_streaming_circuit",
        calldata: CallData.compile({
          model_id: modelId,
          circuit_depth: vc.circuit_depth,
          layer_tags: vc.layer_tags || [],
        }),
      });
      console.log(`  streaming circuit TX: ${tx.transaction_hash}`);
      await provider.waitForTransaction(tx.transaction_hash);
      console.log(`  streaming circuit registered.`);
    } catch (e) {
      console.log(`  streaming circuit registration: ${e.message} (may already exist)`);
    }
  }

  // Step 2: Open session + upload + seal
  let sessionId;
  if (!skipSession) {
    // Open session (7 params: model_id, total_felts, circuit_depth, num_layers, weight_binding_mode, packed, io_packed)
    const totalFelts = vc.total_felts;
    const circuitDepth = vc.circuit_depth || 8;
    const numLayers = (vc.layer_tags || []).length || circuitDepth;
    const weightBindingMode = vc.weight_binding_mode || 4;
    const packed = vc.packed ? 1 : 0;
    const ioPacked = vc.io_packed ? 1 : 0;
    console.log(`\nOpening GKR session: ${totalFelts} felts, depth=${circuitDepth}, layers=${numLayers}, binding_mode=${weightBindingMode}`);
    const openTx = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "open_gkr_session",
      calldata: CallData.compile({
        model_id: modelId,
        total_felts: totalFelts,
        circuit_depth: circuitDepth,
        num_layers: numLayers,
        weight_binding_mode: weightBindingMode,
        packed: packed,
        io_packed: ioPacked,
      }),
    });
    console.log(`  open TX: ${openTx.transaction_hash}`);
    const receipt = await provider.waitForTransaction(openTx.transaction_hash);

    // Extract session_id from events
    // session_id is #[key] in GkrSessionOpened, so it's in keys[1] (keys[0] is event selector)
    if (receipt.events && receipt.events.length > 0) {
      const evt = receipt.events[0];
      sessionId = evt.keys?.[1] || evt.data?.[0];
      console.log(`  Session ID: ${sessionId}`);
      console.log(`  Event keys: ${JSON.stringify(evt.keys)}`);
      console.log(`  Event data: ${JSON.stringify(evt.data)}`);
    } else {
      console.error("  Could not extract session ID from receipt!");
      process.exit(1);
    }

    // Upload chunks
    const chunks = vc.chunks || [];
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      console.log(`  Uploading chunk ${i + 1}/${chunks.length} (${chunk.length} felts)`);
      const tx = await account.execute({
        contractAddress: CONTRACT,
        entrypoint: "upload_gkr_chunk",
        calldata: [sessionId, `${i}`, `${chunk.length}`, ...chunk],
      });
      console.log(`    TX: ${tx.transaction_hash}`);
      await provider.waitForTransaction(tx.transaction_hash);
    }

    // Seal
    console.log("  Sealing session...");
    const sealTx = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "seal_gkr_session",
      calldata: [sessionId],
    });
    console.log(`  seal TX: ${sealTx.transaction_hash}`);
    await provider.waitForTransaction(sealTx.transaction_hash);
    console.log("  Sealed.");
  } else {
    sessionId = process.env.SESSION_ID;
    if (!sessionId) {
      console.error("--skip-session requires SESSION_ID env var");
      process.exit(1);
    }
  }

  // Step 3: Extract calldata files and submit
  const outDir = `/tmp/streaming_calldata_${Date.now()}`;
  mkdirSync(outDir, { recursive: true });

  function writeCalldata(filename, calldata) {
    // Replace __SESSION_ID__ with actual session_id
    const resolved = calldata.map((f) =>
      f === "__SESSION_ID__" ? sessionId : f
    );
    writeFileSync(join(outDir, filename), resolved.join(" ") + "\n");
    console.log(`  ${filename}: ${resolved.length} felts`);
  }

  console.log(`\nWriting calldata to ${outDir}/`);
  writeCalldata("stream_init.txt", vc.init_calldata);

  if (vc.output_mle_chunks) {
    for (let i = 0; i < vc.output_mle_chunks.length; i++) {
      writeCalldata(
        `stream_output_mle_${i}.txt`,
        vc.output_mle_chunks[i].calldata
      );
    }
  }

  if (vc.stream_batches) {
    for (let i = 0; i < vc.stream_batches.length; i++) {
      writeCalldata(`stream_layers_${i}.txt`, vc.stream_batches[i].calldata);
    }
  }

  if (vc.weight_binding_calldata) {
    writeCalldata("stream_weight_binding.txt", vc.weight_binding_calldata);
  }

  if (vc.input_mle_chunks) {
    for (let i = 0; i < vc.input_mle_chunks.length; i++) {
      writeCalldata(
        `stream_finalize_input_mle_${i}.txt`,
        vc.input_mle_chunks[i].calldata
      );
    }
  }

  writeCalldata("stream_finalize.txt", vc.finalize_calldata);

  // Step 4: Submit streaming steps
  console.log(`\nSubmitting streaming verification steps...`);

  const files = readdirSync(outDir)
    .filter((f) => f.endsWith(".txt"))
    .sort();
  const steps = discoverSteps(outDir, files);

  console.log(`  ${steps.length} steps to submit\n`);

  for (const step of steps) {
    const filePath = join(outDir, step.file);
    const raw = readFileSync(filePath, "utf-8").trim();
    const calldata = raw.split(/\s+/);

    console.log(
      `Step: ${step.name} (${step.entrypoint}, ${calldata.length} felts)`
    );

    try {
      const tx = await account.execute({
        contractAddress: CONTRACT,
        entrypoint: step.entrypoint,
        calldata,
      });
      console.log(`  TX: ${tx.transaction_hash}`);
      await provider.waitForTransaction(tx.transaction_hash);
      console.log(`  Confirmed.\n`);
    } catch (err) {
      console.error(`  FAILED: ${err.message || err}`);

      // Try fee estimation for better error message
      try {
        await account.estimateInvokeFee({
          contractAddress: CONTRACT,
          entrypoint: step.entrypoint,
          calldata,
        });
      } catch (feeErr) {
        console.error(`  Fee estimation error: ${feeErr.message || feeErr}`);
      }
      process.exit(1);
    }
  }

  console.log("All steps completed successfully!");

  // Verify
  try {
    const result = await provider.callContract({
      contractAddress: CONTRACT,
      entrypoint: "is_proof_verified",
      calldata: [proof.proof_hash || "0x0"],
    });
    console.log(`is_proof_verified: ${result[0]}`);
  } catch {
    console.log("(could not check is_proof_verified)");
  }
}

function discoverSteps(dir, files) {
  const steps = [];

  const init = files.find((f) => f.startsWith("stream_init"));
  if (init)
    steps.push({
      name: "stream_init",
      file: init,
      entrypoint: "verify_gkr_stream_init",
    });

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
    steps.push({
      name: f.replace(".txt", ""),
      file: f,
      entrypoint: "verify_gkr_stream_layers",
    });
  }

  const weightBinding = files.find((f) =>
    f.startsWith("stream_weight_binding")
  );
  if (weightBinding) {
    steps.push({
      name: "stream_weight_binding",
      file: weightBinding,
      entrypoint: "verify_gkr_stream_weight_binding",
    });
  }

  const finInputMle = files
    .filter((f) => f.startsWith("stream_finalize_input_mle"))
    .sort((a, b) => {
      const na = parseInt(a.match(/(\d+)/)?.[1] || "0", 10);
      const nb = parseInt(b.match(/(\d+)/)?.[1] || "0", 10);
      return na - nb;
    });
  for (const f of finInputMle) {
    steps.push({
      name: f.replace(".txt", ""),
      file: f,
      entrypoint: "verify_gkr_stream_finalize_input_mle",
    });
  }

  const fin = files.find((f) => f === "stream_finalize.txt");
  if (fin) {
    steps.push({
      name: "stream_finalize",
      file: fin,
      entrypoint: "verify_gkr_stream_finalize",
    });
  }

  return steps;
}

main().catch((err) => {
  console.error(`Fatal: ${err.message || err}`);
  process.exit(1);
});
