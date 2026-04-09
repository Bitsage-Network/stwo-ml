import { RpcProvider } from "starknet";
import { readFileSync } from "fs";

const provider = new RpcProvider({ nodeUrl: "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo" });
const proof = JSON.parse(readFileSync("/tmp/proof_final.json", "utf-8"));
const vc = proof.verify_calldata;
const sid = "0x48";
const cd = vc.output_mle_chunks[0].calldata.map(f => f === "__SESSION_ID__" ? sid : f);

try {
  await provider.callContract({
    contractAddress: "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005",
    entrypoint: "verify_gkr_stream_init_output_mle",
    calldata: cd,
  });
  console.log("OK");
} catch (e) {
  const s = JSON.stringify(e);
  // Find common error keywords
  for (const kw of ["NOT_SESSION_OWNER","INVALID","OUTPUT_MLE","MLE_MISMATCH","WRONG","BAD","ASSERT","STEP","SEALED","POLICY"]) {
    if (s.includes(kw)) console.log("Found:", kw);
  }
  console.log("Error tail:", s.slice(-300));
}
