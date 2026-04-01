#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════
// Obelysk — Direct Streaming GKR Submission (no paymaster)
//
// Submits all 6 streaming verification steps using the deployer account.
// Pays gas directly from account STRK balance.
// ═══════════════════════════════════════════════════════════════════

import { Account, RpcProvider, Signer, num } from 'starknet';
import { readFileSync } from 'fs';

const RPC = process.env.STARKNET_RPC || 'https://api.cartridge.gg/x/starknet/sepolia';
const KEY = process.env.STARKNET_PRIVATE_KEY;
const ADDR = process.env.STARKNET_ACCOUNT_ADDRESS;
if (!KEY || !ADDR) {
  console.error('Set STARKNET_PRIVATE_KEY and STARKNET_ACCOUNT_ADDRESS');
  process.exit(1);
}

const C = '0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005';
const proofPath = process.argv[2] || '/tmp/proof_1layer_streaming.json';

const provider = new RpcProvider({ nodeUrl: RPC });
const account = new Account({ provider, address: ADDR, signer: new Signer(KEY) });

const proof = JSON.parse(readFileSync(proofPath, 'utf-8'));
const vc = proof.verify_calldata;

if (vc.schema_version !== 3) {
  console.error(`Expected schema_version 3, got ${vc.schema_version}`);
  process.exit(1);
}

// Explicit resource bounds to bypass broken fee estimation on Sepolia
// L1 gas ~60T/unit, L1DataGas ~60K/unit, L2 gas ~1/unit
// Minimal bounds that cover validation + execution.
// Max per TX: ~0.5 STRK (total 6 TXs = ~3 STRK of 31.5 STRK)
const RESOURCE_BOUNDS = {
  l1_gas: { max_amount: 0x3000n, max_price_per_unit: 0x5AF3107A4000n },      // ~1.2 STRK
  l2_gas: { max_amount: 0x800000n, max_price_per_unit: 0x174876E800n },       // ~0.8 STRK
  l1_data_gas: { max_amount: 0x10000n, max_price_per_unit: 0x100000n },       // ~0.004 STRK
};

async function submit(entrypoint, calldata, label) {
  const maxRetries = 3;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const tag = attempt > 0 ? ` [retry ${attempt}]` : '';
      console.log(`\n${label}${tag} — ${calldata.length} felts`);

      const r = await account.execute([{
        contractAddress: C,
        entrypoint,
        calldata,
      }], { resourceBounds: RESOURCE_BOUNDS });

      const txHash = r.transaction_hash;
      console.log(`  TX: ${txHash}`);
      console.log(`  https://sepolia.voyager.online/tx/${txHash}`);

      // Wait for confirmation (up to 30 min — Sepolia can stall)
      for (let i = 0; i < 360; i++) {
        await new Promise(r => setTimeout(r, 5000));
        try {
          const rc = await provider.getTransactionReceipt(txHash);
          if (rc && rc.execution_status) {
            console.log(`  Status: ${rc.execution_status} (block ${rc.block_number})`);
            if (rc.execution_status === 'REVERTED') {
              console.error(`  REVERT: ${rc.revert_reason}`);
              process.exit(1);
            }
            return { txHash, receipt: rc };
          }
          if (rc && rc.finality_status && rc.finality_status !== 'RECEIVED') {
            console.log(`  Status: ${rc.finality_status} (block ${rc.block_number})`);
            return { txHash, receipt: rc };
          }
        } catch (e) {
          const em = e.message || '';
          if ((em.includes('not found') || em.includes('TXN_HASH_NOT_FOUND')) && i > 120) {
            throw new Error('TX_EVICTED');
          }
        }
        if (i % 60 === 59) console.log(`  Waiting... (${Math.floor(i * 5 / 60)}min)`);
        else if (i % 12 === 11) process.stdout.write('.');
      }
      throw new Error('TX_TIMEOUT');
    } catch (e) {
      if (attempt < maxRetries) {
        console.log(`  Error: ${(e.message || '').slice(0, 100)}, retrying in 15s...`);
        await new Promise(r => setTimeout(r, 15000));
        continue;
      }
      throw e;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════
// Execute streaming pipeline
// ═══════════════════════════════════════════════════════════════════

const t0 = Date.now();
const results = [];

console.log('═══════════════════════════════════════════════════');
console.log('  Obelysk Streaming GKR — Direct Submission');
console.log(`  Contract: ${C}`);
console.log(`  Account:  ${ADDR}`);
console.log(`  Proof:    ${vc.total_felts} felts, ${vc.num_layers} layers`);
console.log('═══════════════════════════════════════════════════');

const SKIP_INIT = process.argv.includes('--skip-init');

// Step 1: stream_init
if (!SKIP_INIT) {
  const initCalldata = vc.init_calldata.map(String);
  const r1 = await submit('verify_gkr_stream_init', initCalldata, '[1/6] verify_gkr_stream_init');
  results.push({ step: 'stream_init', ...r1 });
  await new Promise(r => setTimeout(r, 3000));
} else {
  console.log('\n[1/6] verify_gkr_stream_init — SKIPPED (--skip-init)');
}

// Step 2: output_mle
const outputChunks = vc.output_mle_chunks || [vc.output_mle_calldata];
for (let i = 0; i < outputChunks.length; i++) {
  const cd = outputChunks[i].map(String);
  const r = await submit('verify_gkr_stream_init_output_mle', cd, `[2/6] verify_gkr_stream_init_output_mle (chunk ${i + 1}/${outputChunks.length})`);
  results.push({ step: `output_mle_${i}`, ...r });
  await new Promise(r => setTimeout(r, 3000));
}

// Step 3: layers
for (let i = 0; i < vc.stream_batches.length; i++) {
  const batch = vc.stream_batches[i];
  const cd = batch.calldata.map(String);
  const r = await submit('verify_gkr_stream_layers', cd, `[3/6] verify_gkr_stream_layers (batch ${i + 1}/${vc.stream_batches.length}, ${batch.num_layers} layers)`);
  results.push({ step: `layers_${i}`, ...r });
  await new Promise(r => setTimeout(r, 3000));
}

// Step 4: weight_binding
if (vc.weight_binding_calldata) {
  const cd = vc.weight_binding_calldata.map(String);
  const r = await submit('verify_gkr_stream_weight_binding', cd, '[4/6] verify_gkr_stream_weight_binding');
  results.push({ step: 'weight_binding', ...r });
  await new Promise(r => setTimeout(r, 3000));
}

// Step 5: finalize_input_mle
const inputChunks = vc.input_mle_chunks || [];
for (let i = 0; i < inputChunks.length; i++) {
  const cd = inputChunks[i].map(String);
  const r = await submit('verify_gkr_stream_finalize_input_mle', cd, `[5/6] verify_gkr_stream_finalize_input_mle (chunk ${i + 1}/${inputChunks.length})`);
  results.push({ step: `input_mle_${i}`, ...r });
  await new Promise(r => setTimeout(r, 3000));
}

// Step 6: finalize
const finalizeCalldata = vc.finalize_calldata.map(String);
const r6 = await submit('verify_gkr_stream_finalize', finalizeCalldata, '[6/6] verify_gkr_stream_finalize');
results.push({ step: 'finalize', ...r6 });

const elapsed = ((Date.now() - t0) / 1000).toFixed(0);

console.log('\n═══════════════════════════════════════════════════');
console.log('  STREAMING GKR VERIFICATION COMPLETE');
console.log(`  Total time: ${elapsed}s`);
console.log('═══════════════════════════════════════════════════\n');

console.log('All transaction hashes:');
for (const r of results) {
  console.log(`  ${r.step}: ${r.txHash}`);
  console.log(`    https://sepolia.voyager.online/tx/${r.txHash}`);
}
console.log('');
