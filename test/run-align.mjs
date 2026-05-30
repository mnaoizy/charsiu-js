// End-to-end proof: align the synthetic sample in pure JS and compare to the
// charsiu Python oracle (sample/align_oracle.json).
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';
import { align } from '../dist/align.js';

// --- minimal 16-bit PCM mono WAV reader -> Float32 in [-1, 1] ---
function readWavPCM16(path) {
  const buf = readFileSync(path);
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE')
    throw new Error('not a WAV file');
  let off = 12, sampleRate = 0, dataOff = 0, dataLen = 0, bits = 16, channels = 1;
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === 'fmt ') { channels = buf.readUInt16LE(off + 10); sampleRate = buf.readUInt32LE(off + 12); bits = buf.readUInt16LE(off + 22); }
    else if (id === 'data') { dataOff = off + 8; dataLen = size; }
    off += 8 + size + (size & 1);
  }
  if (bits !== 16) throw new Error(`expected 16-bit PCM, got ${bits}`);
  const n = dataLen / 2 / channels;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = buf.readInt16LE(dataOff + i * 2 * channels) / 32768;
  return { waveform: out, sampleRate };
}

const root = new URL('../', import.meta.url);
const { waveform, sampleRate } = readWavPCM16(new URL('sample/sample.wav', root).pathname);
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
