// Verify the JS g2p port against the g2p_en Python oracle.
import { readFileSync } from 'node:fs';
import { G2p } from '../dist/g2p.js';
import { loadG2pAssets } from '../dist/assets-node.js';

const oracle = JSON.parse(readFileSync(new URL('./g2p_oracle.json', import.meta.url)));
const g2p = new G2p(loadG2pAssets());

let pass = 0;
const fails = [];
for (const { text, phones } of oracle) {
  const got = g2p.convert(text);
  const ok = got.length === phones.length && got.every((p, i) => p === phones[i]);
  if (ok) pass++;
  else fails.push({ text, got: got.join(' '), exp: phones.join(' ') });
  console.log(`${ok ? 'OK ' : '** '} ${text}`);
}

if (fails.length) {
  console.log('\n--- mismatches ---');
  for (const f of fails) {
    console.log(`text: ${f.text}`);
    console.log(`  js:  ${f.got}`);
    console.log(`  py:  ${f.exp}`);
  }
}
console.log(`\n${pass}/${oracle.length} cases match the g2p_en oracle`);
process.exit(pass === oracle.length ? 0 : 1);
