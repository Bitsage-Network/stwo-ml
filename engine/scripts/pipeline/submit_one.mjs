import { Account, RpcProvider } from "starknet";
import { readFileSync } from "fs";

const provider = new RpcProvider({ nodeUrl: "https://starknet-sepolia-rpc.publicnode.com" });
const account = new Account(provider,
    "0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344",
    "0x154de503c7553e078b28044f15b60323899d9437bd44e99d9ab629acbada47a");
const CONTRACT = "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005";

const entrypoint = process.argv[2];
const file = process.argv[3];
const calldata = readFileSync(file, "utf-8").trim().split(/\s+/);
console.log(`${entrypoint}: ${calldata.length} felts (session=${calldata[0]})`);

try {
    const result = await account.execute([{ contractAddress: CONTRACT, entrypoint, calldata }]);
    console.log(`  TX: ${result.transaction_hash}`);
    await provider.waitForTransaction(result.transaction_hash, { retryInterval: 5000 });
    console.log(`  CONFIRMED ✓`);
} catch (e) {
    const msg = e.message || String(e);
    // Extract the Cairo error if present
    const match = msg.match(/0x[0-9a-f]+ \('([^']+)'\)/);
    console.error(`  FAILED: ${match ? match[1] : msg.substring(0, 300)}`);
}
