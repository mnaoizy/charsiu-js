// Fully standalone E2E: text + audio -> alignment in pure JS (no Python).
// Verifies (a) JS phone ids == charsiu's, (b) JS alignment == charsiu oracle.
import { readFileSync, writeFileSync } from 'node:fs';
import { createNodeAligner } from '../dist/aligner-node.js';
import { toTextGrid } from '../dist/index.js';

function readWavPCM16(path) {
  const buf = readFileSync(path);
  let off = 12, dataOff = 0, dataLen = 0, channels = 1;
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === 'fmt ') channels = buf.readUInt16LE(off + 10);
    else if (id === 'data') { dataOff = off + 8; dataLen = size; }
    off += 8 + size + (size & 1);
  }
  const n = dataLen / 2 / channels;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = buf.readInt16LE(dataOff + i * 2 * channels) / 32768;
  return out;
}

const root = new URL('../', import.meta.url);
const text = readFileSync(new URL('sample/transcript.txt', root), 'utf8').trim();
const waveform = readWavPCM16(new URL('sample/sample.wav', root).pathname);
const oracle = JSON.parse(readFileSync(new URL('sample/align_oracle.json', root)));
const oracleWords = JSON.parse(readFileSync(new URL('sample/align_oracle_words.json', root)));
const spec = JSON.parse(readFileSync(new URL('sample/align_spec.json', root)));

console.log(`text: "${text}"`);
const aligner = await createNodeAligner();
const t0 = performance.now();
const { phones, words: wordSegs, phoneIds } = await aligner.align(waveform, text);
console.log(`aligned in ${(performance.now() - t0).toFixed(0)}ms\n`);

// (a) phone ids match charsiu's get_phone_ids
const idsMatch = phoneIds.length === spec.phone_ids.length && phoneIds.every((v, i) => v === spec.phone_ids[i]);
console.log(`phone ids:  js[${phoneIds.length}] vs charsiu[${spec.phone_ids.length}] -> ${idsMatch ? 'OK' : 'MISMATCH'}`);
if (!idsMatch) { console.log('  js:     ', phoneIds.join(',')); console.log('  charsiu:', spec.phone_ids.join(',')); }

// (b) alignment matches oracle
const seqMatch = phones.length === oracle.length && phones.every((s, i) => s[2] === oracle[i][2]);
let maxDiff = 0;
for (let i = 0; i < Math.min(phones.length, oracle.length); i++)
  maxDiff = Math.max(maxDiff, Math.abs(phones[i][0] - oracle[i][0]), Math.abs(phones[i][1] - oracle[i][1]));

console.log(`\nphones: ${phones.map((p) => p[2]).filter((p) => p !== '[SIL]').join(' ')}`);
console.log(`segments match oracle: ${seqMatch ? 'OK' : 'MISMATCH'} | max boundary diff: ${maxDiff.toFixed(2)}s`);

// (c) word-level alignment matches oracle
const wordsMatch = wordSegs.length === oracleWords.length &&
  wordSegs.every((s, i) => s[2] === oracleWords[i][2] && Math.abs(s[0] - oracleWords[i][0]) < 0.02 && Math.abs(s[1] - oracleWords[i][1]) < 0.02);
console.log('\nword alignment:');
for (const [s, e, w] of wordSegs) console.log(`  ${s.toFixed(2)}-${e.toFixed(2)}  ${w}`);
console.log(`words match oracle: ${wordsMatch ? 'OK' : 'MISMATCH'}`);

// (d) TextGrid output
const tg = toTextGrid([
  { name: 'words', intervals: wordSegs },
  { name: 'phones', intervals: phones },
]);
writeFileSync(new URL('sample/sample.TextGrid', root), tg);
console.log(`\nwrote sample/sample.TextGrid (${tg.split('\n').length} lines)`);

const pass = idsMatch && seqMatch && maxDiff <= 0.02 && wordsMatch;
console.log(`\n${pass ? 'PASS: standalone JS text->alignment (phones + words + TextGrid) matches charsiu' : 'FAIL'}`);
process.exit(pass ? 0 : 1);
