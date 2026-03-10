import { Account, RpcProvider, json, Signer, CallData, Contract } from 'starknet';
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
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS || '0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005';
const NEW_CLASS_HASH = '0x6a6b7a75d5ec1f63d715617d352bc0d353042b2a033d98fa28ffbaf6c5b5439';

const provider = new RpcProvider({ nodeUrl: RPC_URL });
const signer = new Signer(PRIVATE_KEY);
const account = new Account({ provider, address: ACCOUNT_ADDRESS, signer });

const step = process.argv[2] || 'propose';

if (step === 'propose') {
  console.log('Proposing upgrade to class:', NEW_CLASS_HASH);
  const tx = await account.execute({
    contractAddress: CONTRACT_ADDRESS,
    entrypoint: 'propose_upgrade',
    calldata: CallData.compile({ new_class_hash: NEW_CLASS_HASH }),
  });
  console.log('TX:', tx.transaction_hash);
  const receipt = await provider.waitForTransaction(tx.transaction_hash);
  console.log('Status:', receipt.execution_status || receipt.status);
  console.log('\nWait 5+ minutes, then run: node _upgrade.mjs execute');
} else if (step === 'execute') {
  console.log('Executing upgrade...');
  const tx = await account.execute({
    contractAddress: CONTRACT_ADDRESS,
    entrypoint: 'execute_upgrade',
    calldata: [],
  });
  console.log('TX:', tx.transaction_hash);
  const receipt = await provider.waitForTransaction(tx.transaction_hash);
  console.log('Status:', receipt.execution_status || receipt.status);
  console.log('Upgrade complete! New class:', NEW_CLASS_HASH);
} else if (step === 'check') {
  // Check pending upgrade
  const result = await provider.callContract({
    contractAddress: CONTRACT_ADDRESS,
    entrypoint: 'get_pending_upgrade',
    calldata: [],
  });
  console.log('Pending class_hash:', result[0]);
  console.log('Proposed at timestamp:', BigInt(result[1]).toString());
  const now = Math.floor(Date.now() / 1000);
  const proposedAt = Number(BigInt(result[1]));
  const elapsed = now - proposedAt;
  console.log('Elapsed:', elapsed, 'seconds');
  if (elapsed >= 300) {
    console.log('Timelock expired! Ready to execute.');
  } else {
    console.log('Wait', 300 - elapsed, 'more seconds.');
  }
} else {
  console.error('Usage: node _upgrade.mjs [propose|execute|check]');
}
