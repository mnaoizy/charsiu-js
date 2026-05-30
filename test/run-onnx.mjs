// Proof that the converted charsiu model runs in JS (Node) via onnxruntime,
// producing the SAME frame logits as PyTorch.
//
// Reads models/<short>/parity.json (saved by scripts/convert.py): the exact
// input_values fed to torch, plus torch/onnx reference outputs. We run the ONNX
// graph in JS on that same input and confirm the logits match.
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';

const SHORT = process.argv[2] ?? 'en_w2v2_fc_10ms';
const dir = new URL(`../models/${SHORT}/`, import.meta.url);
const parity = JSON.parse(readFileSync(new URL('parity.json', dir)));

const input = Float32Array.from(parity.input_values);
const [B, FRAMES, VOCAB] = parity.logits_shape;
console.log(`model: ${SHORT}`);
console.log(`input: ${input.length} samples (1s @ 16kHz) | expected logits: [${parity.logits_shape}]`);

const session = await ort.InferenceSession.create(
  new URL('model.onnx', dir).pathname,
);

const feeds = {
  input_values: new ort.Tensor('float32', input, [1, input.length]),
};
const t0 = performance.now();
const { logits } = await session.run(feeds);
const ms = (performance.now() - t0).toFixed(0);

console.log(`ran in ${ms}ms | output dims: [${logits.dims}]`);

// 1) shape parity
const shapeOk = logits.dims[0] === B && logits.dims[1] === FRAMES && logits.dims[2] === VOCAB;

// 2) value parity on frame 0 vs the torch/onnx reference
const frame0 = logits.data.slice(0, VOCAB);
let maxDiff = 0;
for (let i = 0; i < VOCAB; i++) maxDiff = Math.max(maxDiff, Math.abs(frame0[i] - parity.logits_frame0[i]));

// 3) argmax-per-frame parity (the thing alignment actually consumes)
const jsArgmax = [];
for (let f = 0; f < FRAMES; f++) {
  let best = 0, bestv = -Infinity;
  for (let c = 0; c < VOCAB; c++) {
    const v = logits.data[f * VOCAB + c];
    if (v > bestv) { bestv = v; best = c; }
  }
  jsArgmax.push(best);
}
const argmaxMatches = jsArgmax.filter((v, i) => v === parity.argmax_per_frame[i]).length;

console.log('\n--- parity vs PyTorch reference ---');
console.log(`shape match:        ${shapeOk ? 'OK' : 'MISMATCH'}`);
console.log(`max|js-ref| frame0: ${maxDiff.toExponential(3)}`);
console.log(`argmax frames match: ${argmaxMatches}/${FRAMES}`);

const pass = shapeOk && maxDiff < 1e-2 && argmaxMatches === FRAMES;
console.log(`\n${pass ? 'PASS: charsiu model runs in JS with PyTorch parity' : 'FAIL'}`);
process.exit(pass ? 0 : 1);
