import { Account, RpcProvider, CallData } from "starknet";
import { readFileSync } from "fs";

const RPC = "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo";
const CONTRACT = "0x376fa0c4a9cf3d069e6a5b91bad6e131e7a800f9fced49bd72253a0b0983039";
const ADDR = "0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f";
const KEY = process.env.STARKNET_PRIVATE_KEY;

const provider = new RpcProvider({ nodeUrl: RPC });
const account = new Account({ provider, address: ADDR, signer: KEY });
const proof = JSON.parse(readFileSync("/tmp/proof_small_chunks.json", "utf-8"));
const vc = proof.verify_calldata;

// Gas bounds per step type
const GAS = {
  small: { l2_gas: { max_amount: 0x5000000n, max_price_per_unit: 0x2cb417800n }, l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n }, l1_data_gas: { max_amount: 0x5000n, max_price_per_unit: 0x7ed13c779n } },
  med:   { l2_gas: { max_amount: 0x20000000n, max_price_per_unit: 0x2cb417800n }, l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n }, l1_data_gas: { max_amount: 0x30000n, max_price_per_unit: 0x7ed13c779n } },
  // For storing ~1000 felts to contract storage
  wbStore: { l2_gas: { max_amount: 0x30000000n, max_price_per_unit: 0x2cb417800n }, l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n }, l1_data_gas: { max_amount: 0x30000n, max_price_per_unit: 0x7ed13c779n } },
  // For final chunk: read all back + verify binding proof
  wbFinal: { l2_gas: { max_amount: 0x40000000n, max_price_per_unit: 0x2cb417800n }, l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n }, l1_data_gas: { max_amount: 0x50000n, max_price_per_unit: 0x7ed13c779n } },
};

async function tx(ep, cd, gasType) {
  const opts = gasType ? { resourceBounds: GAS[gasType] } : {};
  const t = await account.execute({ contractAddress: CONTRACT, entrypoint: ep, calldata: cd }, opts);
  const r = await provider.waitForTransaction(t.transaction_hash);
  if (r.execution_status === "REVERTED") throw new Error("REVERTED: " + (r.revert_reason || "").slice(0, 200));
  return t.transaction_hash;
}

async function main() {
  console.log("Contract:", CONTRACT);

  // Register (skip if done)
  try {
    const r = await provider.callContract({ contractAddress: CONTRACT, entrypoint: "get_model_gkr_weight_count", calldata: [vc.model_id] });
    if (Number(BigInt(r[0])) > 0) console.log("register... skip");
    else throw 0;
  } catch {
    process.stdout.write("register... ");
    await tx("register_model_gkr", CallData.compile({ model_id: vc.model_id, weight_commitments: proof.weight_commitments||[], circuit_descriptor: vc.layer_tags||[] }), "med");
    console.log("✓");
  }

  // Session
  process.stdout.write("open... ");
  const otx = await account.execute({ contractAddress: CONTRACT, entrypoint: "open_gkr_session",
    calldata: CallData.compile({ model_id: vc.model_id, total_felts: vc.total_felts, circuit_depth: vc.circuit_depth||8,
      num_layers: (vc.layer_tags||[]).length||8, weight_binding_mode: vc.weight_binding_mode||4, packed: vc.packed?1:0, io_packed: vc.io_packed?1:0 }) });
  const or = await provider.waitForTransaction(otx.transaction_hash);
  const sid = or.events?.[0]?.keys?.[1] || or.events?.[0]?.data?.[0];
  console.log("✓ sid=" + sid);

  for (let i = 0; i < (vc.chunks||[]).length; i++) {
    process.stdout.write("up" + i + "...");
    await tx("upload_gkr_chunk", [sid, ""+i, ""+vc.chunks[i].length, ...vc.chunks[i]], "small");
    console.log("✓");
  }
  process.stdout.write("seal...");
  await tx("seal_gkr_session", [sid], "small");
  console.log("✓");

  // Verification
  const steps = [];
  steps.push({ n: "init", cd: vc.init_calldata, ep: "verify_gkr_stream_init", gas: null });
  (vc.output_mle_chunks||[]).forEach((c,i) => steps.push({ n: "out_mle", cd: c.calldata, ep: "verify_gkr_stream_init_output_mle", gas: null }));
  (vc.stream_batches||[]).forEach((c,i) => steps.push({ n: "layers", cd: c.calldata, ep: "verify_gkr_stream_layers", gas: "med" }));

  const wbcs = vc.weight_binding_chunks || [];
  wbcs.forEach((c) => steps.push({
    n: "wb_" + c.chunk_idx, cd: c.calldata, ep: c.entrypoint,
    gas: c.is_last ? "wbFinal" : "wbStore"
  }));

  (vc.input_mle_chunks||[]).forEach((c,i) => steps.push({ n: "in_mle", cd: c.calldata, ep: "verify_gkr_stream_finalize_input_mle", gas: "med" }));
  steps.push({ n: "finalize", cd: vc.finalize_calldata, ep: "verify_gkr_stream_finalize", gas: "med" });

  console.log("\n" + steps.length + " verification steps:");
  for (const s of steps) {
    const cd = s.cd.map(f => f === "__SESSION_ID__" ? sid : f);
    process.stdout.write("  " + s.n + "(" + cd.length + ")... ");
    try {
      await tx(s.ep, cd, s.gas);
      console.log("✓");
    } catch (e) {
      console.log("FAIL:", (e.message||"").slice(0, 200));
      process.exit(1);
    }
  }

  console.log("\n══════════════════════════════════════════");
  console.log("  ALL " + steps.length + "/" + steps.length + " VERIFIED ON-CHAIN ✓");
  console.log("══════════════════════════════════════════");
}

main().catch(e => { console.error("Fatal:", (e.message||"").slice(0,300)); process.exit(1); });
