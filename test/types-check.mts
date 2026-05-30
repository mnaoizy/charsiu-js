// Consumer-facing TypeScript check: imports the BUILT package (dist .d.ts) the way
// a published-package user would, and asserts the public API is type-safe.
// Run after `npm run build` via `tsc -p tsconfig.test.json`.
import {
  createNodeAligner, createNodeAlignerZh,
  ForcedAligner, toTextGrid, G2p, G2pM, PhonemizerEn, PhonemizerZh,
} from '../dist/aligner-node.js';
import type { AlignResult, Segment, PhonemizerLike } from '../dist/index.js';
import { loadG2pAssets, loadPhoneVocab } from '../dist/assets-node.js';

async function main() {
  const aligner = await createNodeAligner({ modelPath: './model.onnx' });
  const zh = await createNodeAlignerZh({ modelUrl: 'https://example/model.onnx' });

  const r: AlignResult = await aligner.align(new Float32Array(16000), 'hello world');
  const phones: Segment[] = r.phones;
  const [start, end, label]: Segment = phones[0];
  const _check: [number, number, string] = [start, end, label];

  const tg: string = toTextGrid([{ name: 'words', intervals: r.words }]);

  const phonemizer: PhonemizerLike = new PhonemizerEn(new G2p(loadG2pAssets()), loadPhoneVocab());
  const manual = new ForcedAligner({ session: {} as never, ort: {} as never, phonemizer });
  await manual.align(new Float32Array(1), 'x');

  // @ts-expect-error waveform must be a Float32Array, not a string
  await aligner.align('not audio', 'x');

  // @ts-expect-error align takes (waveform, text), not a number
  await zh.align(123, 'x');

  return { r, tg, _check, manual, G2pM, PhonemizerZh };
}
void main;
