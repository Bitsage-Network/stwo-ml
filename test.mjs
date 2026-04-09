import { createStwoProverClient } from '@obelyzk/sdk';

const prover = createStwoProverClient();
console.log('Package:  @obelyzk/sdk 1.1.0');
console.log('Default:  https://prover.bitsage.network');

const health = await fetch('https://prover.bitsage.network/health').then(r => r.json());
console.log('GPU:     ', health.device_name);
console.log('Status:  ', health.status);
console.log('Models:  ', health.loaded_models);
console.log('');
console.log('Install:  npm install @obelyzk/sdk');
