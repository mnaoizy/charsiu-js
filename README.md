# charsiu-js

Neural phonetic **forced aligner** for **Node and the browser** — a JavaScript port
of [charsiu](https://github.com/lingjzhu/charsiu). Give it audio and a transcript;
get back **phone- and word-level time alignments**, fully client-side, no Python.

```
the quick brown fox …
0.00–0.10  the     DH AH
0.10–0.41  quick   K W IH K
0.41–0.76  brown   B R AW N
…
```

It runs a wav2vec2 frame-classification model (via [onnxruntime](https://onnxruntime.ai/))
to get per-frame phone probabilities, converts text to phones with a ported
[g2p_en](https://github.com/Kyubyong/g2p), and aligns the two with DTW — matching
charsiu's Python output bit-for-bit (see [Verification](#verification)).

> Status: **English and Mandarin** both work end-to-end. See [Languages](#languages).

## Install

```bash
npm install charsiu-js
# plus the onnxruntime for your runtime:
npm install onnxruntime-node    # Node
npm install onnxruntime-web     # browser bundlers
```

Written in TypeScript; ships compiled ESM + `.d.ts`, so it's type-safe out of the box.

## What you need besides the package

`npm install charsiu-js` gives you the code and the g2p data. Two more things:

1. **An onnxruntime** (peer dependency) — `onnxruntime-node` for Node, or
   `onnxruntime-web` for the browser. https://onnxruntime.ai/
2. **An acoustic ONNX model** (not bundled; ~123 MB EN / ~90 MB ZH). The upstream
   charsiu weights live on the Hugging Face Hub but are PyTorch, so convert them
   to ONNX (see [Model setup](#model-setup)) or host a converted copy yourself:
   - English: https://huggingface.co/charsiu/en_w2v2_fc_10ms
   - Mandarin: https://huggingface.co/charsiu/zh_w2v2_tiny_fc_10ms
   - tokenizers / other languages: https://huggingface.co/charsiu

   Pass the model via `modelPath` (local file) or `modelUrl` (downloaded + cached).

## Model setup

The g2p assets (~7.5 MB) ship inside the package. The acoustic model does **not**
(~123 MB quantized) — point the aligner at one you host or convert:

```bash
# convert + quantize charsiu/en_w2v2_fc_10ms -> models/en_w2v2_fc_10ms/
npm run convert charsiu/en_w2v2_fc_10ms
npm run quantize en_w2v2_fc_10ms
```

Then host `model_quantized.onnx` (e.g. on the Hugging Face Hub) and pass its URL,
or pass a local path.

## Usage — Node

```js
import { createNodeAligner } from 'charsiu-js';

const aligner = await createNodeAligner({
  // one of:
  modelPath: './models/en_w2v2_fc_10ms/model_quantized.onnx',
  // modelUrl: 'https://huggingface.co/<you>/charsiu-en-onnx/resolve/main/model_quantized.onnx',
});

// waveform: Float32Array, 16 kHz mono, samples in [-1, 1]
const { phones, words } = await aligner.align(waveform, 'the quick brown fox');
// phones: [[start, end, 'DH'], ...]   words: [[start, end, 'the'], ...]
```

## Usage — browser

```js
import { G2p, PhonemizerEn, ForcedAligner } from 'charsiu-js/core';
import { loadG2pAssets, loadPhoneVocab, decodeToMono16k } from 'charsiu-js/assets-web';
import * as ort from 'onnxruntime-web';

const phonemizer = new PhonemizerEn(new G2p(await loadG2pAssets('/assets/')),
                                    await loadPhoneVocab('/assets/'));
const session = await ort.InferenceSession.create('/model_quantized.onnx');
const aligner = new ForcedAligner({ session, ort, phonemizer });

const waveform = await decodeToMono16k(await file.arrayBuffer()); // any audio file
const { phones, words } = await aligner.align(waveform, transcript);
```

A runnable demo lives in `web/`:

```bash
npm run convert && npm run quantize   # produce the local model
npm run serve                         # http://localhost:8080/web/index.html
```

## API

- `createNodeAligner({ modelPath?, modelUrl?, cacheDir? }) → Promise<ForcedAligner>`
  (Node) — wires onnxruntime-node + bundled assets; downloads & caches `modelUrl`.
- `new ForcedAligner({ session, ort, phonemizer, silThreshold?, resolution? })` —
  runtime-agnostic core.
- `aligner.align(waveform, text) → { phones, words, phoneIds }` — segments are
  `[startSec, endSec, label]`; `[SIL]` marks silence.
- `toTextGrid([{ name, intervals }, …]) → string` — Praat TextGrid (short format).
- Lower-level building blocks are exported too: `G2p`, `PhonemizerEn`, `normalize`,
  `softmaxRows`, `forcedAlign`, `seq2duration`.

Input must be **16 kHz mono**. In the browser, `decodeToMono16k` handles decoding
and resampling; in Node, resample yourself (e.g. with the model's expected rate).

## Verification

Every stage is checked against the original Python, bit-for-bit:

```bash
npm test
```

| Test | Checks |
|------|--------|
| `test:onnx` | ONNX frame logits == PyTorch |
| `test:g2p` | g2p output == g2p_en (15/15, incl. OOV, numbers, punctuation) |
| `test:standalone` | text→phones+words == charsiu oracle (Node) |
| `test:browser` | same, in headless Chrome via onnxruntime-web |

## Languages

- **English** — complete. g2p is a full port of `g2p_en` (CMUdict + GRU OOV
  predictor). Homographs (~371 words like "read") currently use their default
  pronunciation; the POS tagger g2p_en uses to disambiguate them isn't ported yet.
- **Mandarin (Standard Chinese)** — complete. g2p is a full port of `g2pM`
  (CEDICT + a BiLSTM that disambiguates polyphonic characters, e.g. 长→cháng/zhǎng,
  行→xíng/háng). Uses `charsiu/zh_w2v2_tiny_fc_10ms` (~90 MB, 210 tonal phones).

```js
import { createNodeAlignerZh } from 'charsiu-js';
const zh = await createNodeAlignerZh({ modelPath: './models/zh_w2v2_tiny_fc_10ms/model.onnx' });
const { phones, words } = await zh.align(waveform, '快速的棕色狐狸');
// phones: [[0, .08,'k'], [.08,.22,'uai4'], …]   words: [[0,.22,'快'], …]
```

For the browser, build a `ForcedAligner` with `G2pM` + `PhonemizerZh` and the zh
assets, exactly like the English browser example.

## How it works / internals

See [`FINDINGS.md`](./FINDINGS.md) for the full design notes: model architecture
(`Wav2Vec2ForFrameClassification` ≡ `Wav2Vec2ForCTC`), the DTW setup
(`step_sizes [[1,1],[1,0]]`, silence handling), and the g2p port.

## Development

Source is TypeScript in `src/`; `npm run build` compiles it to `dist/` (ESM +
`.d.ts`). `npm test` builds, runs the Python-parity tests (ONNX/g2p/alignment in
Node and in headless Chrome), and type-checks the public API as a consumer would.
`npm run typecheck` checks types without emitting. The Python scripts in `scripts/`
(model conversion, quantization, asset export, oracle generation) need the
`.venv` from the reproduce steps above.

## License

MIT — see [LICENSE](./LICENSE). Bundled data/models retain their own licenses;
see [ATTRIBUTION.md](./ATTRIBUTION.md) (charsiu: MIT, g2p_en: Apache-2.0,
CMUdict: BSD-style).
