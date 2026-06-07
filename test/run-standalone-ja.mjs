// Standalone E2E (Japanese): text + audio -> alignment in pure JS, via
// createNodeAlignerJa (tokana + hubert phoneme-CTC). Verifies (a) the phone
// sequence matches the pyopenjtalk oracle (devoicing-collapsed), and (b) phone +
// word boundaries match the committed snapshot.
//
// Needs a dev checkout with the model + dictionary built (both gitignored):
//   npm run convert prj-beatrice/japanese-hubert-base-phoneme-ctc
//   npm run quantize japanese-hubert-base-phoneme-ctc   # optional (uses model.onnx otherwise)
//   npm run setup-dict                                  # -> models/ipadic-dict
// If either is missing the test SKIPs (exit 0) so `npm test` still passes.
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createNodeAlignerJa } from '../dist/aligner-node.js';
import { loadWav16k } from '../dist/assets-node.js';

const root = new URL('../', import.meta.url);
const A = (p) => fileURLToPath(new URL(p, root));
const MODEL = A('models/japanese-hubert-base-phoneme-ctc/model.onnx');
const DICT = A('models/ipadic-dict');

if (!existsSync(MODEL) || !existsSync(DICT)) {
  console.log('SKIP: Japanese model and/or IPADIC dict not built (see header for setup).');
  process.exit(0);
}

const text = readFileSync(new URL('sample/ja_transcript.txt', root), 'utf8').trim();
const wav = loadWav16k(new URL('sample/ja_sample.wav', root));
const phonesOracle = JSON.parse(readFileSync(new URL('sample/ja_phones_oracle.json', root)));
const snap = JSON.parse(readFileSync(new URL('sample/ja_align_oracle.json', root)));
const snapW = JSON.parse(readFileSync(new URL('sample/ja_align_oracle_words.json', root)));

const aligner = await createNodeAlignerJa({ dicPath: DICT, modelPath: MODEL });
console.log('text:', text);
const t0 = performance.now();
const { phones, words } = await aligner.align(wav, text);
console.log(`aligned in ${(performance.now() - t0).toFixed(0)}ms`);

// (a) phone sequence == pyopenjtalk (devoicing-collapsed)
const seq = phones.filter((p) => p[2] !== '[SIL]').map((p) => p[2]);
const seqOk = seq.length === phonesOracle.length && seq.every((p, i) => p === phonesOracle[i]);
console.log('phones:', seq.join(' '));
console.log('phones match pyopenjtalk:', seqOk ? 'OK' : 'MISMATCH');

// (b) phone + word boundaries match the committed snapshot
function cmp(got, exp) {
  let ok = got.length === exp.length, md = 0;
  for (let i = 0; i < Math.min(got.length, exp.length); i++) {
    if (got[i][2] !== exp[i][2]) ok = false;
    md = Math.max(md, Math.abs(got[i][0] - exp[i][0]), Math.abs(got[i][1] - exp[i][1]));
  }
  return { ok: ok && md <= 0.02, md };
}
const pc = cmp(phones, snap), wc = cmp(words, snapW);
console.log(`phone boundaries vs snapshot: ${pc.ok ? 'OK' : 'MISMATCH'} (max diff ${pc.md.toFixed(2)}s)`);
console.log('words:', words.map((w) => w[2]).join(' '));
console.log(`word boundaries vs snapshot:  ${wc.ok ? 'OK' : 'MISMATCH'} (max diff ${wc.md.toFixed(2)}s)`);

const pass = seqOk && pc.ok && wc.ok;
console.log(`\n${pass ? 'PASS: standalone JS Japanese alignment matches the oracle' : 'FAIL'}`);
process.exit(pass ? 0 : 1);
