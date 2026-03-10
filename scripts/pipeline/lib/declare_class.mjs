#!/usr/bin/env node
// Declare a Sierra class on Starknet using starknet.js
// Usage: node declare_class.mjs <sierra_json_path> <casm_json_path>
//
// Uses starknet.js Account.declareIfNot() which handles the compiled_class_hash
// computation from the actual CASM file.

import { Account, RpcProvider, json, Signer } from 'starknet';
import { readFileSync } from 'fs';

const RPC_URL = process.env.STARKNET_RPC;
if (!RPC_URL) {
  console.error('FATAL: STARKNET_RPC env var required');
  process.exit(1);
}
const ACCOUNT_ADDRESS = process.env.STARKNET_ACCOUNT;
if (!ACCOUNT_ADDRESS) {
  console.error('FATAL: STARKNET_ACCOUNT env var required');
  process.exit(1);
}
const PRIVATE_KEY = process.env.STARKNET_PRIVATE_KEY;
if (!PRIVATE_KEY) {
  console.error('FATAL: STARKNET_PRIVATE_KEY env var required');
  process.exit(1);
}

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
