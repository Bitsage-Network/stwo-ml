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

// All addresses via env vars — no hardcoded defaults for production safety.
const RPC = process.env.STARKNET_RPC;
const ADDR = process.env.DEPLOYER_ADDRESS;
if (!RPC || !ADDR) { console.error("ERROR: Set STARKNET_RPC and DEPLOYER_ADDRESS env vars"); process.exit(1); }
const KEY = process.env.STARKNET_PRIVATE_KEY;

const provider = new RpcProvider({ nodeUrl: RPC });
const account = new Account({ provider, address: ADDR, signer: KEY });

async function main() {
  // Declare recursive verifier class
  const sierraClass = json.parse(readFileSync("/tmp/recursive_class.json", "utf-8"));
  const casmClass = json.parse(readFileSync("/tmp/recursive_casm.json", "utf-8"));
  console.log("Sierra:", JSON.stringify(sierraClass).length, "bytes");

  console.log("Declaring recursive verifier...");
  const declareResp = await account.declare({ contract: sierraClass, casm: casmClass });
  console.log("Declare TX:", declareResp.transaction_hash);
  console.log("Class hash:", declareResp.class_hash);
  await provider.waitForTransaction(declareResp.transaction_hash);
  console.log("Declared!");

  // Deploy
  console.log("\nDeploying...");
  const deployResp = await account.deployContract({
    classHash: declareResp.class_hash,
    constructorCalldata: CallData.compile({ owner: ADDR }),
    salt: "0x" + Date.now().toString(16),
  });
  console.log("Deploy TX:", deployResp.transaction_hash);
  await provider.waitForTransaction(deployResp.transaction_hash);

  console.log("\n═══════════════════════════════════════════════════");
  console.log("  ObelyZK Recursive STARK Verifier (Sepolia)");
  console.log("═══════════════════════════════════════════════════");
  console.log("  Contract:", deployResp.contract_address);
  console.log("  Class:   ", declareResp.class_hash);
  console.log("  Admin:   ", ADDR);
  console.log("═══════════════════════════════════════════════════");

  // Register model + submit proof
  const proof = JSON.parse(readFileSync("/tmp/proof_recursive.json", "utf-8"));
  const rp = proof.recursive_proof;
  const modelId = proof.model_id;
  const CONTRACT = deployResp.contract_address;

  // Extract real circuit_hash and weight_super_root from proof
  const calldata = rp.calldata;
  const circuitHash = rp.circuit_hash || packQm31ToFelt252(calldata.slice(0, 4));
  const weightSuperRoot = rp.weight_super_root || packQm31ToFelt252(calldata.slice(8, 12));
  console.log("\nCircuit Hash:  " + circuitHash);
  console.log("Weight Root:   " + weightSuperRoot);

  console.log("\nRegistering model...");
  await (async () => {
    const t = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "register_model_recursive",
      calldata: CallData.compile({
        model_id: modelId,
        circuit_hash: circuitHash,
        weight_super_root: weightSuperRoot,
        policy_commitment: proof.policy_commitment || "0x0",
      }),
    });
    await provider.waitForTransaction(t.transaction_hash);
    console.log("Registered ✓");
  })();

  // Submit recursive proof (SINGLE TX!)
  console.log("\nSubmitting recursive proof (" + rp.total_felts + " felts)...");
  const submitTx = await account.execute({
    contractAddress: CONTRACT,
    entrypoint: "verify_recursive",
    calldata: CallData.compile({
      model_id: modelId,
      io_commitment: proof.io_commitment || "0x1",
      stark_proof_data: rp.calldata,
    }),
  });
  console.log("TX:", submitTx.transaction_hash);
  const receipt = await provider.waitForTransaction(submitTx.transaction_hash);

  if (receipt.execution_status === "REVERTED") {
    console.log("REVERTED:", receipt.revert_reason?.slice(0, 200));
    return;
  }

  console.log("\n══════════════════════════════════════════════════════════");
  console.log("  RECURSIVE PROOF VERIFIED ON-CHAIN IN 1 TX! ✓");
  console.log("══════════════════════════════════════════════════════════");
  console.log("  TX:", submitTx.transaction_hash);
  console.log("  Explorer: https://sepolia.starkscan.co/tx/" + submitTx.transaction_hash);
  console.log("  Felts:", rp.total_felts);

  // Verify it's recorded
  const verified = await provider.callContract({
    contractAddress: CONTRACT,
    entrypoint: "get_recursive_verification_count",
    calldata: [modelId],
  });
  console.log("  Verification count:", Number(BigInt(verified[0])));
}

main().catch(e => console.error("Fatal:", (e.message || "").slice(0, 500)));
