import { Account, RpcProvider, Signer } from 'starknet';
import { readFileSync } from 'fs';

const RPC = 'https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/GUBwFqKhSgn4mwVbN6Sbn';
const ADDR = '0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344';
const KEY = '0x0154de503c7553e078b28044f15b60323899d9437bd44e99d9ab629acbada47a';
const C = '0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005';

const CHUNK_SIZE = 1500;
const MODEL_ID = '0x3'; // already registered with matching weight commitments

const provider = new RpcProvider({ nodeUrl: RPC });
const account = new Account({ provider, address: ADDR, signer: new Signer(KEY) });
const proof = JSON.parse(readFileSync('/tmp/unpacked_proof_2.json', 'utf-8'));
const vc = proof.verify_calldata;

const allFelts = vc.chunks.flat();
const chunks = [];
for (let i = 0; i < allFelts.length; i += CHUNK_SIZE) {
  chunks.push(allFelts.slice(i, i + CHUNK_SIZE));
}

const totalSteps = 3 + chunks.length;
const t0 = Date.now();

console.log(`UNPACKED A/B TEST`);
console.log(`Proof: ${vc.total_felts} felts, ${chunks.length} chunks of ${CHUNK_SIZE}`);
console.log(`packed=${vc.packed}, circuit_depth=${vc.circuit_depth}, layers=${vc.num_layers}, mode=${vc.weight_binding_mode}`);
console.log(`Total TXs: ${totalSteps}`);

async function ex(ep, cd, msg, maxRetries = 3) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      process.stdout.write(msg + (attempt > 0 ? ` [retry ${attempt}]` : '') + ' ... ');
      const r = await account.execute(
        [{ contractAddress: C, entrypoint: ep, calldata: cd }]
      );
      const txHash = r.transaction_hash;
      console.log('TX:', txHash.slice(0, 22) + '...');

      let finalReceipt = null;
      for (let poll = 0; poll < 60; poll++) {
        await new Promise(r => setTimeout(r, 5000));
        try {
          const rc = await provider.getTransactionReceipt(txHash);
          if (rc && rc.execution_status) {
            finalReceipt = rc;
            break;
          }
          if (rc && rc.finality_status && rc.finality_status !== 'RECEIVED') {
            finalReceipt = rc;
            break;
          }
          process.stdout.write('.');
        } catch (e) {
          const em = e.message || '';
          if (em.includes('not found') || em.includes('TXN_HASH_NOT_FOUND')) {
            if (poll > 12) {
              console.log(' evicted');
              throw new Error('TX_EVICTED');
            }
          }
          process.stdout.write('?');
        }
      }

      if (!finalReceipt) {
        console.log(' timeout');
        throw new Error('TX_TIMEOUT');
      }

      const s = finalReceipt.execution_status || finalReceipt.finality_status || 'UNKNOWN';
      console.log(' Status:', s);
      if (s === 'REVERTED') {
        console.error('  REVERT:', finalReceipt.revert_reason);
        process.exit(1);
      }
      return finalReceipt;
    } catch (e) {
      const em = e.message || String(e);
      if (em.includes('EVICTED') || em.includes('TTL') || em.includes('TX_EVICTED') || em.includes('TX_TIMEOUT')) {
        if (attempt < maxRetries) {
          const wait = 20 + attempt * 15;
          console.log(`  Mempool issue, waiting ${wait}s...`);
          await new Promise(r => setTimeout(r, wait * 1000));
          continue;
        }
      }
      if (attempt < maxRetries && !em.includes('REVERTED')) {
        console.log(`  Error: ${em.slice(0, 100)}, retrying in 20s...`);
        await new Promise(r => setTimeout(r, 20000));
        continue;
      }
      throw e;
    }
  }
  throw new Error('Max retries exceeded');
}

// Open session with packed=0 (UNPACKED)
console.log('\n--- Open GKR session (UNPACKED) ---');
const or = await ex('open_gkr_session',
  [MODEL_ID, String(vc.total_felts), String(vc.circuit_depth), String(vc.num_layers), String(vc.weight_binding_mode), '0'],
  `[1/${totalSteps}] open`);

let sid = null;
for (const e of (or.events || [])) {
  if (e.keys && e.keys.length >= 2) { sid = e.keys[1]; break; }
}
if (sid === null) {
  for (const e of (or.events || [])) {
    if (e.data && e.data.length >= 1) { sid = e.data[0]; break; }
  }
}
if (!sid) { console.error('No session ID in events'); process.exit(1); }
console.log('  Session:', sid);

// Upload chunks
console.log('\n--- Upload chunks ---');
for (let i = 0; i < chunks.length; i++) {
  await ex('upload_gkr_chunk',
    [sid, String(i), String(chunks[i].length), ...chunks[i]],
    `[${2 + i}/${totalSteps}] chunk_${i} (${chunks[i].length} felts)`);
  if (i < chunks.length - 1) await new Promise(r => setTimeout(r, 3000));
}

// Seal
console.log('\n--- Seal ---');
await ex('seal_gkr_session', [sid], `[${2 + chunks.length}/${totalSteps}] seal`);

// Verify
console.log('\n--- Verify ---');
await ex('verify_gkr_from_session', [sid], `[${3 + chunks.length}/${totalSteps}] VERIFY`);

const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
console.log(`\n========================================`);
console.log(`UNPACKED QM31 ON-CHAIN VERIFICATION SUCCEEDED`);
console.log(`Time: ${elapsed}s | Felts: ${vc.total_felts} | Chunks: ${chunks.length}`);
console.log(`========================================`);
