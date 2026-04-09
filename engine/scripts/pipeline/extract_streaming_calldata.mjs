#!/usr/bin/env node
// Extract streaming calldata from a proof JSON file into text files
// for use with streaming_submit.mjs.
//
// Usage:
//   node extract_streaming_calldata.mjs proof.json [output-dir]
//
// Output files:
//   stream_init.txt
//   stream_output_mle_0.txt, stream_output_mle_1.txt, ...
//   stream_layers_0.txt, stream_layers_1.txt, ...
//   stream_weight_binding.txt
//   stream_finalize_input_mle_0.txt, ...
//   stream_finalize.txt

import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { join } from "path";

const proofPath = process.argv[2];
if (!proofPath) {
  console.error("Usage: node extract_streaming_calldata.mjs <proof.json> [output-dir]");
  process.exit(1);
}

const outDir = process.argv[3] || "streaming_calldata";
mkdirSync(outDir, { recursive: true });

const proof = JSON.parse(readFileSync(proofPath, "utf-8"));
const vc = proof.verify_calldata;

if (!vc || vc.mode !== "streaming") {
  console.error("Proof does not contain streaming calldata (mode != streaming)");
  process.exit(1);
}

let fileCount = 0;

function writeCalldata(filename, calldata) {
  const path = join(outDir, filename);
  writeFileSync(path, calldata.join(" ") + "\n");
  console.log(`  ${filename}: ${calldata.length} felts`);
  fileCount++;
}

// 1. stream_init
writeCalldata("stream_init.txt", vc.init_calldata);

// 2. output_mle chunks
if (vc.output_mle_chunks) {
  for (let i = 0; i < vc.output_mle_chunks.length; i++) {
    writeCalldata(`stream_output_mle_${i}.txt`, vc.output_mle_chunks[i].calldata);
  }
}

// 3. stream_layers batches
if (vc.stream_batches) {
  for (let i = 0; i < vc.stream_batches.length; i++) {
    writeCalldata(`stream_layers_${i}.txt`, vc.stream_batches[i].calldata);
  }
}

// 4. weight_binding (chunked or legacy)
if (vc.weight_binding_chunks) {
  for (let i = 0; i < vc.weight_binding_chunks.length; i++) {
    writeCalldata(`stream_weight_binding_${i}.txt`, vc.weight_binding_chunks[i].calldata);
  }
} else if (vc.weight_binding_calldata) {
  // Legacy: single weight binding calldata
  writeCalldata("stream_weight_binding_0.txt", vc.weight_binding_calldata);
}

// 5. input_mle chunks
if (vc.input_mle_chunks) {
  for (let i = 0; i < vc.input_mle_chunks.length; i++) {
    writeCalldata(`stream_finalize_input_mle_${i}.txt`, vc.input_mle_chunks[i].calldata);
  }
}

// 6. finalize
writeCalldata("stream_finalize.txt", vc.finalize_calldata);

console.log(`\nExtracted ${fileCount} calldata files to ${outDir}/`);
console.log(`Model ID: ${vc.model_id}`);
console.log(`Layer tags: [${(vc.layer_tags || []).join(", ")}]`);
console.log(`\nSubmit with:`);
console.log(`  node streaming_submit.mjs \\`);
console.log(`    --contract <CONTRACT_ADDRESS> \\`);
console.log(`    --calldata-dir ${outDir} \\`);
console.log(`    --account-address $STARKNET_ACCOUNT \\`);
console.log(`    --private-key $STARKNET_PRIVATE_KEY`);
