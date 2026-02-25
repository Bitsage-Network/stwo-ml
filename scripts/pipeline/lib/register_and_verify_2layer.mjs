import { Account, RpcProvider, Signer } from 'starknet';
import { readFileSync } from 'fs';

const RPC = 'https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/GUBwFqKhSgn4mwVbN6Sbn';
const ADDR = '0x0759a4374389b0e3cfcc59d49310b6bc75bb12bbf8ce550eb5c2f026918bb344';
const KEY = '0x0154de503c7553e078b28044f15b60323899d9437bd44e99d9ab629acbada47a';
const C = '0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005';

const provider = new RpcProvider({ nodeUrl: RPC });
const account = new Account({ provider, address: ADDR, signer: new Signer(KEY) });
const proof = JSON.parse(readFileSync('/tmp/packed_proof_2.json', 'utf-8'));
const vc = proof.verify_calldata;
const all = vc.chunks.flat();

// Extract sections
let off = 0;
function readSection() {
  const len = Number(BigInt(all[off]));
  const data = all.slice(off + 1, off + 1 + len);
  off += 1 + len;
  return data;
}
const raw_io = readSection();       // raw_io_data
const matmul_dims = readSection();   // matmul_dims
const deq_bits = readSection();      // dequantize_bits
const proof_data = readSection();    // proof_data
const weight_commitments = readSection(); // weight_commitments

console.log('weight_commitments:', weight_commitments.length);
console.log('proof_data:', proof_data.length, 'felts');
console.log('circuit_depth:', vc.circuit_depth, ', num_layers:', vc.num_layers);

// Build circuit descriptor: [circuit_depth, tag0, tag1, ...]
// For 2 transformer layers (15 proof layers), the tags are in proof walk order
// We need to extract them from proof_data
// Tags: 8=RMSNorm, 0=MatMul, 1=Add, 3=Activation, etc.
// For Qwen3 2-layer: RMSNorm, MatMul, MatMul, Activation, Add, RMSNorm, MatMul, MatMul, Activation, Add, RMSNorm, MatMul, MatMul, Add, RMSNorm
// Let me parse: first value is the tag
let pd_off = 0;
const tags = [];

function parseTag() {
  const tag = Number(BigInt(proof_data[pd_off]));
  tags.push(tag);
  pd_off += 1;
  return tag;
}

// Parse 15 layers to extract tags
for (let l = 0; l < vc.num_layers; l++) {
  const tag = parseTag();

  if (tag === 8) { // RMSNorm
    pd_off += 4; // input, output, rms_sq, rsqrt (packed QM31)
    pd_off += 1; // rsqrt_table_commitment
    pd_off += 1; // simd_combined
    const nrounds = Number(BigInt(proof_data[pd_off])); pd_off += 1;
    pd_off += nrounds * 4; // 4 QM31 per round
    pd_off += 2; // final evals
    // logup
    const has = Number(BigInt(proof_data[pd_off])); pd_off += 1;
    if (has === 1) {
      pd_off += 1; // claimed_sum
      const eq_rounds = Number(BigInt(proof_data[pd_off])); pd_off += 1;
      pd_off += eq_rounds * 4;
      pd_off += 3; // final evals
      const num_mults = Number(BigInt(proof_data[pd_off])); pd_off += 1;
      pd_off += num_mults;
    }
  } else if (tag === 0) { // MatMul
    const nrounds = Number(BigInt(proof_data[pd_off])); pd_off += 1;
    pd_off += nrounds * 3; // 3 QM31 per round
    pd_off += 2; // final evals
  } else if (tag === 1) { // Add
    pd_off += 2; // lhs, rhs
    pd_off += 1; // trunk_idx
  } else if (tag === 3) { // Activation
    pd_off += 1; // act_type_tag
    pd_off += 2; // input, output
    pd_off += 1; // table_commitment
    const has = Number(BigInt(proof_data[pd_off])); pd_off += 1;
    if (has === 1) {
      pd_off += 1; // claimed_sum
      const eq_rounds = Number(BigInt(proof_data[pd_off])); pd_off += 1;
      pd_off += eq_rounds * 4;
      pd_off += 3;
      const num_mults = Number(BigInt(proof_data[pd_off])); pd_off += 1;
      pd_off += num_mults;
    }
  } else {
    console.error('Unknown tag:', tag, 'at layer', l);
    process.exit(1);
  }
}

// Check for deferred proofs at end
const num_deferred = Number(BigInt(proof_data[pd_off]));
pd_off += 1;
console.log('deferred proofs:', num_deferred);
console.log('Tags:', tags);
console.log('Remaining proof_data:', proof_data.length - pd_off);

// Build circuit descriptor: [circuit_depth, ...tags]
const circuit_descriptor = [String(vc.circuit_depth), ...tags.map(String)];

async function ex(ep, cd, msg) {
  process.stdout.write(msg + ' ... ');
  const r = await account.execute([{ contractAddress: C, entrypoint: ep, calldata: cd }]);
  const rc = await provider.waitForTransaction(r.transaction_hash, { retryInterval: 5000 });
  const s = rc.execution_status || rc.status;
  console.log(s, r.transaction_hash.slice(0, 22) + '...');
  if (s === 'REVERTED') { console.error(rc.revert_reason); process.exit(1); }
  return rc;
}

// Step 1: Register model_id 0x3
console.log('\n--- Register model 0x3 ---');
const registerCalldata = [
  '0x3', // model_id
  String(weight_commitments.length), ...weight_commitments,
  String(circuit_descriptor.length), ...circuit_descriptor,
];
await ex('register_model_gkr', registerCalldata, 'register_model_gkr');

// Step 2: Open session
console.log('\n--- Open session ---');
const CHUNK_SIZE = 1500;
const rechunked = [];
for (let i = 0; i < all.length; i += CHUNK_SIZE) {
  rechunked.push(all.slice(i, i + CHUNK_SIZE));
}

const totalSteps = 3 + rechunked.length;
const or = await ex('open_gkr_session',
  ['0x3', String(vc.total_felts), String(vc.circuit_depth), String(vc.num_layers), String(vc.weight_binding_mode), '1'],
  '[1/' + totalSteps + '] open');

let sid = null;
for (const e of (or.events || [])) {
  if (e.keys && e.keys.length >= 2) { sid = e.keys[1]; break; }
}
if (!sid) {
  for (const e of (or.events || [])) {
    if (e.data && e.data.length >= 1) { sid = e.data[0]; break; }
  }
}
console.log('Session:', sid);

// Step 3: Upload chunks
console.log('\n--- Upload chunks ---');
for (let i = 0; i < rechunked.length; i++) {
  await ex('upload_gkr_chunk',
    [sid, String(i), String(rechunked[i].length), ...rechunked[i]],
    `[${2 + i}/${totalSteps}] chunk_${i} (${rechunked[i].length})`);
  if (i < rechunked.length - 1) await new Promise(r => setTimeout(r, 3000));
}

// Step 4: Seal
console.log('\n--- Seal ---');
await ex('seal_gkr_session', [sid], `[${2 + rechunked.length}/${totalSteps}] seal`);

// Step 5: Verify
console.log('\n--- Verify ---');
await ex('verify_gkr_from_session', [sid], `[${3 + rechunked.length}/${totalSteps}] VERIFY`);

console.log('\nPACKED QM31 VERIFICATION SUCCEEDED for model 0x3 (2-layer)');
