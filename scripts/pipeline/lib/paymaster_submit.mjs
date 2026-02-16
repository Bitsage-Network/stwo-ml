#!/usr/bin/env node
import { spawnSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '../../../../');
const target = path.join(repoRoot, 'scripts/pipeline/lib/paymaster_submit.mjs');

const result = spawnSync(process.execPath, [target, ...process.argv.slice(2)], {
  stdio: 'inherit',
  env: process.env,
});

if (result.error) {
  console.error(`[ERR] Failed to execute canonical paymaster script: ${result.error.message}`);
  process.exit(1);
}
process.exit(result.status ?? 1);
