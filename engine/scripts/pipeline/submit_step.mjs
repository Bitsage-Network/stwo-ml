import { Account, RpcProvider } from "starknet";
import { readFileSync } from "fs";

const RPC = "https://starknet-sepolia-rpc.publicnode.com";
const ACCOUNT = "0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344";
const KEY = "0x154de503c7553e078b28044f15b60323899d9437bd44e99d9ab629acbada47a";
const CONTRACT = "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005";

const provider = new RpcProvider({ nodeUrl: RPC });
const account = new Account(provider, ACCOUNT, KEY);

const step = process.argv[2];
const file = process.argv[3];

const calldata = readFileSync(file, "utf-8").trim().split(/\s+/);
console.log(`Submitting ${step} with ${calldata.length} felts...`);

try {
    const result = await account.execute([{
        contractAddress: CONTRACT,
        entrypoint: step,
        calldata,
    }]);
    console.log(`TX: ${result.transaction_hash}`);
    console.log("Waiting...");
    await provider.waitForTransaction(result.transaction_hash);
    console.log("CONFIRMED ✓");
} catch (e) {
    console.error("FAILED:", e.message?.substring(0, 500) || e);
}
