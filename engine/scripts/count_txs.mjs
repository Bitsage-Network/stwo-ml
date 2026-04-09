import { readFileSync } from "fs";

const proof = JSON.parse(readFileSync("/tmp/proof_final2.json", "utf-8"));
const vc = proof.verify_calldata;

console.log("=== 1-Layer SmolLM2 Transaction Breakdown ===\n");

let txCount = 0;
const chunks = vc.chunks || [];
console.log(`1. open_gkr_session:     1 TX`); txCount++;
console.log(`2. upload_gkr_chunk:     ${chunks.length} TX (${chunks.map(c=>c.length).join(" + ")} felts)`); txCount += chunks.length;
console.log(`3. seal_gkr_session:     1 TX`); txCount++;
console.log(`4. stream_init:          1 TX (${(vc.init_calldata||[]).length} felts)`); txCount++;

const omle = vc.output_mle_chunks || [];
console.log(`5. output_mle:           ${omle.length} TX (${omle.map(c=>c.calldata.length).join(" + ")} felts)`); txCount += omle.length;

const batches = vc.stream_batches || [];
console.log(`6. layers:               ${batches.length} TX (${batches.map(c=>c.calldata.length).join(" + ")} felts)`); txCount += batches.length;

const wbChunksArr = vc.weight_binding_chunks || [];
const wbLen = wbChunksArr.reduce((s,c) => s + c.calldata.length, 0) || (vc.weight_binding_calldata||[]).length;
const wbChunkCount = wbChunksArr.length || Math.ceil(wbLen / 4500);
console.log(`7. weight_binding:       ${wbChunkCount} TX (${wbLen} felts${wbChunkCount > 1 ? ", chunked" : ""})`); txCount += wbChunkCount;

const imle = vc.input_mle_chunks || [];
console.log(`8. input_mle:            ${imle.length} TX (${imle.map(c=>c.calldata.length).join(" + ")} felts)`); txCount += imle.length;

console.log(`9. finalize:             1 TX (${(vc.finalize_calldata||[]).length} felts)`); txCount++;

console.log(`\n  TOTAL: ${txCount} transactions (1-layer model)`);
console.log(`\n  Of which ${txCount - chunks.length - 2} are verification steps`);
console.log(`  (the rest are session management: open + ${chunks.length} uploads + seal)`);
