import { Account, RpcProvider, json, Signer, CallData } from 'starknet';
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

const provider = new RpcProvider({ nodeUrl: RPC_URL });
const signer = new Signer(PRIVATE_KEY);
const account = new Account({ provider, address: ACCOUNT_ADDRESS, signer });

const sc = json.parse(readFileSync('/Users/vaamx/bitsage-network/libs/elo-cairo-verifier/target/dev/elo_cairo_verifier_SumcheckVerifierContract.contract_class.json', 'utf-8'));
const cc = json.parse(readFileSync('/Users/vaamx/bitsage-network/libs/elo-cairo-verifier/target/dev/elo_cairo_verifier_SumcheckVerifierContract.compiled_contract_class.json', 'utf-8'));

console.log('Sierra felts:', sc.sierra_program?.length);

// Check balance
const balance = await provider.callContract({
  contractAddress: '0x04718f5a0fc34cc1af16a1cdee98ffb20c31f5cd61d6ab07201858f4287c938d',
  entrypoint: 'balanceOf',
  calldata: CallData.compile({ account: ACCOUNT_ADDRESS }),
});
const balWei = BigInt(balance[0]);
const balStrk = Number(balWei) / 1e18;
console.log('Balance:', balStrk.toFixed(4), 'STRK');

// Estimate fee first
try {
  const feeEstimate = await account.estimateDeclareFee({ contract: sc, casm: cc });
  console.log('Estimated fee:');
  console.log('  overall_fee:', feeEstimate.overall_fee?.toString());
  console.log('  gas_consumed:', feeEstimate.gas_consumed?.toString());
  console.log('  gas_price:', feeEstimate.gas_price?.toString());
  console.log('  suggestedMaxFee:', feeEstimate.suggestedMaxFee?.toString());

  const overallFee = BigInt(feeEstimate.overall_fee || '0');
  const feeStrk = Number(overallFee) / 1e18;
  console.log('  Fee in STRK:', feeStrk.toFixed(4));

  if (overallFee > balWei) {
    console.error('INSUFFICIENT FUNDS: need', feeStrk.toFixed(4), 'STRK but have', balStrk.toFixed(4));
    console.error('Top up deployer at:', ACCOUNT_ADDRESS);
    process.exit(1);
  }

  console.log('Fee OK, proceeding with declare...');
} catch (err) {
  const msg = err.message || '';
  console.error('Fee estimation failed:', msg.substring(Math.max(0, msg.length - 400)));
  // Try declaring anyway
}

try {
  const r = await account.declareIfNot({ contract: sc, casm: cc });
  if (r.transaction_hash) {
    console.log('TX:', r.transaction_hash);
    console.log('Class:', r.class_hash);
    const receipt = await provider.waitForTransaction(r.transaction_hash);
    console.log('Status:', receipt.execution_status || receipt.status);
  } else {
    console.log('Already declared:', r.class_hash);
  }
} catch (err) {
  if (typeof err.code !== 'undefined') console.error('Code:', err.code);
  const msg = err.message || '';
  console.error('Error tail:', msg.substring(Math.max(0, msg.length - 600)));
  process.exit(1);
}
