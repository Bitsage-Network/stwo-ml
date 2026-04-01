#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════
// Obelysk — Streaming GKR Verification on Starknet Sepolia
//
// Submits the 6-step streaming GKR proof verification pipeline.
// Reads calldata from /tmp/streaming_calldata_fixed/ files.
//
// Usage:
//   STARKNET_RPC=... STARKNET_ACCOUNT=... STARKNET_PRIVATE_KEY=... \
//     node submit_streaming.mjs [--skip-init]
// ═══════════════════════════════════════════════════════════════════

import { Account, RpcProvider, Signer } from 'starknet';
import { readFileSync } from 'fs';

const RPCS = [
  process.env.STARKNET_RPC,
  'https://starknet-sepolia.public.blastapi.io/rpc/v0_8',
  'https://free-rpc.nethermind.io/sepolia-juno/v0_8',
  'https://api.cartridge.gg/x/starknet/sepolia',
].filter(Boolean);

const ADDR = process.env.STARKNET_ACCOUNT;
if (!ADDR) { console.error('FATAL: STARKNET_ACCOUNT env var required'); process.exit(1); }
const KEY = process.env.STARKNET_PRIVATE_KEY;
if (!KEY) { console.error('FATAL: STARKNET_PRIVATE_KEY env var required'); process.exit(1); }

const C = '0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005';
const CALLDATA_DIR = process.env.CALLDATA_DIR || '/tmp/streaming_calldata_fixed';
const SKIP_INIT = process.argv.includes('--skip-init');

// starknet.js v8: options object constructor
const provider = new RpcProvider({ nodeUrl: RPCS[0] });
const account = new Account({ provider, address: ADDR, signer: new Signer(KEY) });

function loadCalldata(filename) {
  const raw = readFileSync(`${CALLDATA_DIR}/${filename}`, 'utf-8').trim();
  return raw.split(/\s+/);
}

async function ex(entrypoint, calldata, label, maxRetries = 3) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const tag = attempt > 0 ? ` [retry ${attempt}]` : '';
      process.stdout.write(`  ${label}${tag} (${calldata.length} felts) ... `);

      const r = await account.execute([{
        contractAddress: C,
        entrypoint,
        calldata,
      }]);

      const txHash = r.transaction_hash;
      console.log(`TX: ${txHash}`);

      // Poll for receipt
      let receipt = null;
      for (let poll = 0; poll < 60; poll++) {
        await new Promise(r => setTimeout(r, 5000));
        try {
          const rc = await provider.getTransactionReceipt(txHash);
          if (rc && (rc.execution_status || (rc.finality_status && rc.finality_status !== 'RECEIVED'))) {
            receipt = rc;
            break;
          }
          process.stdout.write('.');
        } catch (e) {
          const em = e.message || '';
          if ((em.includes('not found') || em.includes('TXN_HASH_NOT_FOUND')) && poll > 12) {
            throw new Error('TX_EVICTED');
          }
          process.stdout.write('?');
        }
      }

      if (!receipt) throw new Error('TX_TIMEOUT');

      const status = receipt.execution_status || receipt.finality_status || 'UNKNOWN';
      console.log(`  Status: ${status}`);

      if (status === 'REVERTED') {
        console.error(`  REVERT: ${receipt.revert_reason}`);
        process.exit(1);
      }

      return { receipt, txHash };
    } catch (e) {
      const em = e.message || String(e);
      if (attempt < maxRetries && !em.includes('REVERTED')) {
        const wait = 15 + attempt * 10;
        console.log(`\n  Error: ${em.slice(0, 120)}, retrying in ${wait}s...`);
        await new Promise(r => setTimeout(r, wait * 1000));
        continue;
      }
      throw e;
    }
  }
  throw new Error('Max retries exceeded');
}

// ─── Streaming Steps ─────────────────────────────────────────────

const steps = [
  { file: 'stream_init.txt',                  entry: 'verify_gkr_stream_init',                  label: '[1/6] stream_init' },
  { file: 'stream_output_mle_0.txt',          entry: 'verify_gkr_stream_init_output_mle',       label: '[2/6] output_mle' },
  { file: 'stream_layers_0.txt',              entry: 'verify_gkr_stream_layers',                label: '[3/6] layers' },
  { file: 'stream_weight_binding.txt',        entry: 'verify_gkr_stream_weight_binding',        label: '[4/6] weight_binding' },
  { file: 'stream_finalize_input_mle_0.txt',  entry: 'verify_gkr_stream_finalize_input_mle',    label: '[5/6] finalize_input_mle' },
  { file: 'stream_finalize.txt',              entry: 'verify_gkr_stream_finalize',              label: '[6/6] finalize' },
];

const startIdx = SKIP_INIT ? 1 : 0;
const txHashes = [];

console.log(`\n═══════════════════════════════════════════════════`);
console.log(`  Obelysk Streaming GKR Verification`);
console.log(`  Contract: ${C}`);
console.log(`  Steps: ${startIdx + 1}-6 of 6${SKIP_INIT ? ' (skipping init — already submitted)' : ''}`);
console.log(`═══════════════════════════════════════════════════\n`);

const t0 = Date.now();

for (let i = startIdx; i < steps.length; i++) {
  const step = steps[i];
  const calldata = loadCalldata(step.file);

  console.log(`\n─── ${step.label} ───`);
  const { receipt, txHash } = await ex(step.entry, calldata, step.label);
  txHashes.push({ step: step.label, txHash, status: receipt.execution_status || receipt.finality_status });

  // Breathing room between steps
  if (i < steps.length - 1) {
    await new Promise(r => setTimeout(r, 5000));
  }
}

const elapsed = ((Date.now() - t0) / 1000).toFixed(0);

console.log(`\n═══════════════════════════════════════════════════`);
console.log(`  STREAMING GKR VERIFICATION COMPLETE`);
console.log(`  Time: ${elapsed}s`);
console.log(`═══════════════════════════════════════════════════\n`);

console.log('Transaction hashes:');
for (const tx of txHashes) {
  console.log(`  ${tx.step}: ${tx.txHash}`);
  console.log(`    https://sepolia.voyager.online/tx/${tx.txHash}`);
}
console.log('');

// Check verification status
try {
  const cnt = await provider.callContract({
    contractAddress: C,
    entrypoint: 'get_verification_count',
    calldata: ['0x1'],
  });
  console.log('Verification count:', cnt);
} catch (e) {
  console.log('(Could not query verification count)');
}
