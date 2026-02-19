#!/usr/bin/env node
// Declare a Sierra class on Starknet using starknet.js
// Usage: node declare_class.mjs <sierra_json_path> <casm_json_path>
//
// Uses starknet.js Account.declareIfNot() which handles the compiled_class_hash
// computation from the actual CASM file.

import { Account, RpcProvider, json, Signer } from 'starknet';
import { readFileSync } from 'fs';

const RPC_URL = process.env.STARKNET_RPC || 'https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/GUBwFqKhSgn4mwVbN6Sbn';
const ACCOUNT_ADDRESS = '0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344';
const PRIVATE_KEY = '0x0154de503c7553e078b28044f15b60323899d9437bd44e99d9ab629acbada47a';

const sierraPath = process.argv[2];
const casmPath = process.argv[3];

if (!sierraPath || !casmPath) {
  console.error('Usage: node declare_class.mjs <sierra.json> <casm.json>');
  process.exit(1);
}

console.log(`RPC: ${RPC_URL}`);
console.log(`Account: ${ACCOUNT_ADDRESS}`);

const provider = new RpcProvider({ nodeUrl: RPC_URL });
const signer = new Signer(PRIVATE_KEY);
const account = new Account({ provider, address: ACCOUNT_ADDRESS, signer });

const sierraContract = json.parse(readFileSync(sierraPath, 'utf-8'));
const casmContract = json.parse(readFileSync(casmPath, 'utf-8'));

console.log('Declaring class...');

try {
  const declareResponse = await account.declareIfNot({
    contract: sierraContract,
    casm: casmContract,
  });

  if (declareResponse.transaction_hash) {
    console.log(`Transaction hash: ${declareResponse.transaction_hash}`);
    console.log(`Class hash: ${declareResponse.class_hash}`);
    console.log('Waiting for confirmation...');
    const receipt = await provider.waitForTransaction(declareResponse.transaction_hash);
    console.log(`Status: ${receipt.execution_status || receipt.status}`);
  } else {
    console.log(`Class already declared: ${declareResponse.class_hash}`);
  }
} catch (err) {
  console.error('Declare failed:', err.message || err);
  if (err.data) console.error('Data:', JSON.stringify(err.data, null, 2));
  process.exit(1);
}
