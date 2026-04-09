import { Account, RpcProvider, json, CallData, hash } from "starknet";
import { readFileSync } from "fs";

const RPC = "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo";
const CONTRACT = "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005";
const ADDR = "0x57a93709bb92879f0f9f2cb81a87f9ca47d2d7e54af87dbde2831b0b7e81c1f";
const KEY = process.env.STARKNET_PRIVATE_KEY;

async function main() {
  const provider = new RpcProvider({ nodeUrl: RPC });
  const account = new Account({ provider, address: ADDR, signer: KEY });

  // Read compiled contract class + CASM
  const contractClass = json.parse(readFileSync("/tmp/contract_class.json", "utf-8"));
  const casmClass = json.parse(readFileSync("/tmp/contract_casm.json", "utf-8"));
  console.log("Contract class loaded, sierra:", JSON.stringify(contractClass).length, "bytes, casm:", JSON.stringify(casmClass).length, "bytes");

  // Declare
  console.log("Declaring new class...");
  try {
    const declareResponse = await account.declare({
      contract: contractClass,
      casm: casmClass,
    });
    console.log("Declare TX:", declareResponse.transaction_hash);
    console.log("Class hash:", declareResponse.class_hash);
    await provider.waitForTransaction(declareResponse.transaction_hash);
    console.log("Declared!");

    // Upgrade contract
    console.log("\nUpgrading contract to new class...");
    const upgradeTx = await account.execute({
      contractAddress: CONTRACT,
      entrypoint: "propose_upgrade",
      calldata: CallData.compile({ new_class_hash: declareResponse.class_hash }),
    });
    console.log("Propose upgrade TX:", upgradeTx.transaction_hash);
    await provider.waitForTransaction(upgradeTx.transaction_hash);
    console.log("Upgrade proposed! Wait 5 min for timelock, then execute_upgrade.");
    console.log("\nNew class hash:", declareResponse.class_hash);
  } catch (e) {
    const msg = String(e.message || e);
    if (msg.includes("already declared") || msg.includes("is already declared")) {
      console.log("Class already declared. Extracting class hash...");
      const classHash = msg.match(/0x[0-9a-fA-F]{60,66}/)?.[0];
      if (classHash) {
        console.log("Class hash:", classHash);
      }
    } else {
      console.error("Error:", msg.slice(0, 500));
    }
  }
}

main().catch(e => console.error("Fatal:", (e.message || "").slice(0, 300)));
