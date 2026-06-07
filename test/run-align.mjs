// End-to-end proof: align the synthetic sample in pure JS and compare to the
// charsiu Python oracle (sample/align_oracle.json).
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';
import { align } from '../dist/align.js';
import { loadWav16k } from '../dist/assets-node.js';

const root = new URL('../', import.meta.url);
const sampleRate = 16000; // loadWav16k guarantees 16 kHz
const waveform = loadWav16k(new URL('sample/sample.wav', root));
const spec = JSON.parse(readFileSync(new URL('sample/align_spec.json', root)));
const oracle = JSON.parse(readFileSync(new URL('sample/align_oracle.json', root)));
console.log(`audio: ${waveform.length} samples @ ${sampleRate}Hz (${(waveform.length / sampleRate).toFixed(2)}s)`);

const session = await ort.InferenceSession.create(new URL('models/en_w2v2_fc_10ms/model.onnx', root).pathname);
const t0 = performance.now();
const segs = await align(session, ort, waveform, spec);
console.log(`aligned in ${(performance.now() - t0).toFixed(0)}ms -> ${segs.length} segments\n`);

// --- compare to oracle ---
const phonesJS = segs.map((s) => s[2]);
const phonesOracle = oracle.map((s) => s[2]);
const seqMatch = phonesJS.length === phonesOracle.length && phonesJS.every((p, i) => p === phonesOracle[i]);

let maxBoundaryDiff = 0;
const n = Math.min(segs.length, oracle.length);
for (let i = 0; i < n; i++) {
  maxBoundaryDiff = Math.max(maxBoundaryDiff, Math.abs(segs[i][0] - oracle[i][0]), Math.abs(segs[i][1] - oracle[i][1]));
}

console.log('  JS                       oracle');
for (let i = 0; i < n; i++) {
  const a = segs[i], b = oracle[i];
  const mark = a[2] === b[2] && Math.abs(a[0] - b[0]) < 0.02 && Math.abs(a[1] - b[1]) < 0.02 ? ' ' : '*';
  console.log(`${mark} ${a[0].toFixed(2)}-${a[1].toFixed(2)} ${String(a[2]).padEnd(6)}   ${b[0].toFixed(2)}-${b[1].toFixed(2)} ${b[2]}`);
}

console.log(`\nphone sequence match: ${seqMatch ? 'OK' : 'MISMATCH'}`);
console.log(`max boundary diff:    ${maxBoundaryDiff.toFixed(2)}s`);
const pass = seqMatch && maxBoundaryDiff <= 0.02;
console.log(`\n${pass ? 'PASS: JS forced alignment matches the charsiu oracle' : 'FAIL'}`);
process.exit(pass ? 0 : 1);
