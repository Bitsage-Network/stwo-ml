import { readFileSync } from "fs";
const p = JSON.parse(readFileSync("/tmp/proof_chunked.json", "utf-8"));
const vc = p.verify_calldata;
console.log("mode:", vc.mode);
console.log("weight_binding_chunks:", (vc.weight_binding_chunks || []).length);
for (const c of (vc.weight_binding_chunks || [])) {
  console.log("  chunk", c.chunk_idx, ":", c.calldata.length, "felts, ep:", c.entrypoint, "last:", c.is_last);
}
console.log("legacy weight_binding_calldata:", !!(vc.weight_binding_calldata));
