import { readFileSync } from 'node:fs';
import { G2pM } from '../dist/g2pm.js';
import { loadG2pmAssets } from '../dist/assets-node.js';
const oracle = JSON.parse(readFileSync(new URL('./g2pm_oracle.json', import.meta.url)));
const g = new G2pM(loadG2pmAssets());
let pass = 0;
for (const { text, pinyin } of oracle) {
  const got = g.convert(text);
  const ok = got.length === pinyin.length && got.every((p, i) => p === pinyin[i]);
  if (ok) pass++; else { console.log('** ', text); console.log('  js:', got.join(' ')); console.log('  py:', pinyin.join(' ')); }
  if (ok) console.log('OK ', text, '->', got.join(' '));
}
console.log(`\n${pass}/${oracle.length} g2pM cases match`);
process.exit(pass === oracle.length ? 0 : 1);
