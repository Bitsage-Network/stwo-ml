import { Account, RpcProvider, CallData, json } from "starknet";
import { readFileSync } from "fs";

function packQm31ToFelt252(limbs) {
  const shift = 1n << 31n;
  const parse = (x) => BigInt(typeof x === "string" ? x : "0x" + x.toString(16));
  let r = parse(limbs[0]);
  r = r * shift + parse(limbs[1]);
  r = r * shift + parse(limbs[2]);
  r = r * shift + parse(limbs[3]);
  return "0x" + r.toString(16);
}

const RPC = process.env.STARKNET_RPC;
const ADDR = process.env.DEPLOYER_ADDRESS;
const KEY = process.env.STARKNET_PRIVATE_KEY;
if (!RPC || !ADDR || !KEY) { console.error("ERROR: Set STARKNET_RPC, DEPLOYER_ADDRESS, STARKNET_PRIVATE_KEY env vars"); process.exit(1); }

const provider = new RpcProvider({ nodeUrl: RPC });
const account = new Account({ provider, address: ADDR, signer: KEY });

async function main() {
  const sierra = json.parse(readFileSync("/tmp/recursive_class_v3.json", "utf-8"));
  const casm = json.parse(readFileSync("/tmp/recursive_casm_v3.json", "utf-8"));

  console.log("Declaring Phase 2 recursive verifier (with STARK verify)...");
  const decl = await account.declare({ contract: sierra, casm });
  console.log("Class:", decl.class_hash);
  await provider.waitForTransaction(decl.transaction_hash);

  const deploy = await account.deployContract({
    classHash: decl.class_hash,
    constructorCalldata: CallData.compile({ owner: ADDR }),
    salt: "0x" + Date.now().toString(16),
  });
  await provider.waitForTransaction(deploy.transaction_hash);
  const CONTRACT = deploy.contract_address;

  console.log("\n  Recursive Verifier v3 (Phase 2 — trustless STARK verify)");
  console.log("  Contract:", CONTRACT);
  console.log("  Class:   ", decl.class_hash);

  // Register + submit 1-layer proof
  const proof = JSON.parse(readFileSync("/tmp/proof_recursive.json", "utf-8"));
  const rp = proof.recursive_proof;

  const calldata = rp.calldata;
  const circuitHash = rp.circuit_hash || packQm31ToFelt252(calldata.slice(0, 4));
  const weightSuperRoot = rp.weight_super_root || packQm31ToFelt252(calldata.slice(8, 12));
  console.log("  Circuit Hash:", circuitHash);
  console.log("  Weight Root: ", weightSuperRoot);

  await (async () => {
    const t = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "register_model_recursive",
      calldata: CallData.compile({ model_id: proof.model_id, circuit_hash: circuitHash, weight_super_root: weightSuperRoot, policy_commitment: proof.policy_commitment || "0x0" }),
    });
    await provider.waitForTransaction(t.transaction_hash);
    console.log("  Registered");
  })();

  console.log("  Submitting recursive proof (" + rp.total_felts + " felts)...");
  try {
    const rb = {
      l2_gas: { max_amount: 0x40000000n, max_price_per_unit: 0x2cb417800n },
      l1_gas: { max_amount: 0n, max_price_per_unit: 0x59d9328b3166n },
      l1_data_gas: { max_amount: 0x10000n, max_price_per_unit: 0x7ed13c779n },
    };
    const tx = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "verify_recursive",
      calldata: CallData.compile({
        model_id: proof.model_id,
        io_commitment: proof.io_commitment || "0x1",
        stark_proof_data: rp.calldata,
      }),
    }, { resourceBounds: rb });
    console.log("  TX:", tx.transaction_hash);
    const receipt = await provider.waitForTransaction(tx.transaction_hash);
    if (receipt.execution_status === "REVERTED") {
      console.log("  REVERTED:", receipt.revert_reason?.slice(0, 400));
    } else {
      const count = await provider.callContract({
        contractAddress: CONTRACT,
        entrypoint: "get_recursive_verification_count",
        calldata: [proof.model_id],
      });
      console.log("\n  ================================================================");
      console.log("  TRUSTLESS RECURSIVE STARK VERIFIED ON-CHAIN — 1 TX");
      console.log("  ================================================================");
      console.log("  TX:", tx.transaction_hash);
      console.log("  Felts:", rp.total_felts);
      console.log("  Verifications:", Number(BigInt(count[0])));
      console.log("  Explorer: https://sepolia.starkscan.co/tx/" + tx.transaction_hash);
      console.log("  ================================================================");
    }
  } catch (e) {
    console.log("  Error:", (e.message || "").slice(-300));
  }
}

main().catch(e => console.error("Fatal:", (e.message || "").slice(0, 500)));
