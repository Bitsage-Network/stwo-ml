import { Account, RpcProvider, CallData, hash } from "starknet";

const RPC = "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo";
const ADDR = "0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f";
const KEY = process.env.STARKNET_PRIVATE_KEY;
const CLASS_HASH = "0x5dca646786c36f9d68bab802d5c5c4995c37aa7c25bfa59ff20144a283f0956";

async function main() {
  const provider = new RpcProvider({ nodeUrl: RPC });
  const account = new Account({ provider, address: ADDR, signer: KEY });

  console.log("Deploying new contract with class:", CLASS_HASH);
  console.log("Admin (deployer):", ADDR);

  // The contract constructor takes an admin address
  const constructorCalldata = CallData.compile({ admin: ADDR });

  try {
    const deployResponse = await account.deployContract({
      classHash: CLASS_HASH,
      constructorCalldata,
      salt: "0x" + Date.now().toString(16), // unique salt
    });
    console.log("Deploy TX:", deployResponse.transaction_hash);
    console.log("Contract address:", deployResponse.contract_address);
    await provider.waitForTransaction(deployResponse.transaction_hash);
    console.log("Deployed!");
    console.log("\n═══════════════════════════════════════════");
    console.log("  New GKR Verifier Contract (v32)");
    console.log("═══════════════════════════════════════════");
    console.log("  Address:", deployResponse.contract_address);
    console.log("  Class:  ", CLASS_HASH);
    console.log("  Admin:  ", ADDR);
    console.log("  TX:     ", deployResponse.transaction_hash);
    console.log("═══════════════════════════════════════════");
  } catch (e) {
    console.error("Deploy error:", (e.message || "").slice(0, 500));
  }
}

main();
