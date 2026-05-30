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

The g2p data ships inside the package. You need **an onnxruntime** (peer
dependency) — `onnxruntime-node` for Node, or `onnxruntime-web` for the browser
(https://onnxruntime.ai/).

The acoustic ONNX model is **not** bundled (it's large), but in Node it's
**downloaded automatically** on first use (and cached in `~/.cache/charsiu-js`)
from a hosted, INT8-quantized copy: https://huggingface.co/mnaoizyyy/charsiu-js-models
(EN ~123 MB, ZH ~40 MB). So `createNodeAligner()` just works with no setup.

To use your own model, pass `modelPath` (local file) or `modelUrl`. The upstream
PyTorch weights and tokenizers live at https://huggingface.co/charsiu — convert
them to ONNX with the scripts (see [Model setup](#model-setup-custom--your-own-models)).

## Usage — Node

```js
import { createNodeAligner } from 'charsiu-js';

// model auto-downloads from the Hugging Face Hub on first use, then caches
const aligner = await createNodeAligner();
// ...or bring your own:
//   createNodeAligner({ modelPath: './model_quantized.onnx' })
//   createNodeAligner({ modelUrl: 'https://…/model_quantized.onnx' })

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

A runnable demo lives in `web/` (the page imports the compiled `dist/`):

```bash
npm run build                         # compile src -> dist
npm run convert && npm run quantize   # produce a local model (needs the Python venv)
npm run serve                         # http://localhost:8080/web/index.html
```

(Or skip convert/quantize and edit `MODEL_URL` in `web/demo.mjs` to the hosted
model on the Hub.)

## API

- `createNodeAligner({ modelPath?, modelUrl?, cacheDir? }) → Promise<ForcedAligner>`
  (Node) — wires onnxruntime-node + bundled assets; with no options it downloads
  and caches the hosted default model. `createNodeAlignerZh(...)` is the Mandarin
  equivalent.
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
| `test:g2p` | English g2p == g2p_en (15/15, incl. OOV, numbers, punctuation) |
| `test:g2pm` | Mandarin g2p == g2pM (10/10, incl. polyphone disambiguation) |
| `test:standalone` | English text→phones+words == charsiu oracle (Node) |
| `test:standalone-zh` | Mandarin text→phones+words == charsiu oracle (Node) |
| `test:browser` | English alignment in headless Chrome via onnxruntime-web |

`npm test` also type-checks the public API against the built `.d.ts` as a consumer.

## Languages

- **English** — complete. g2p is a full port of `g2p_en` (CMUdict + GRU OOV
  predictor). Homographs (~371 words like "read") currently use their default
  pronunciation; the POS tagger g2p_en uses to disambiguate them isn't ported yet.
- **Mandarin (Standard Chinese)** — complete. g2p is a full port of `g2pM`
  (CEDICT + a BiLSTM that disambiguates polyphonic characters, e.g. 长→cháng/zhǎng,
  行→xíng/háng). Uses `charsiu/zh_w2v2_tiny_fc_10ms` (~40 MB quantized, 210 tonal
  phones).

```js
import { createNodeAlignerZh } from 'charsiu-js';
const zh = await createNodeAlignerZh();   // model auto-downloads on first use
const { phones, words } = await zh.align(waveform, '快速的棕色狐狸');
// phones: [[0, .08,'k'], [.08,.22,'uai4'], …]   words: [[0,.22,'快'], …]
```

## Model setup (custom / your own models)

The default models are downloaded from the Hub automatically. To build your own
(e.g. for another charsiu language), convert + quantize with the scripts, then
pass the result via `modelPath`/`modelUrl`:

```bash
npm run convert charsiu/en_w2v2_fc_10ms    # PyTorch -> models/<name>/model.onnx
npm run quantize en_w2v2_fc_10ms           # -> model_quantized.onnx (single file)
```

For Mandarin in the browser, construct `G2pM` + `PhonemizerZh` and fetch the
`g2pm_*` assets and `vocab_zh.json` yourself — the bundled `assets-web` loader
currently covers English only.

## How it works / internals

See [`FINDINGS.md`](./FINDINGS.md) for the full design notes: model architecture
(`Wav2Vec2ForFrameClassification` ≡ `Wav2Vec2ForCTC`), the DTW setup
(`step_sizes [[1,1],[1,0]]`, silence handling), and the g2p port.

## Development

Source is TypeScript in `src/`; `npm run build` compiles it to `dist/` (ESM +
`.d.ts`). `npm test` builds, runs the Python-parity tests (ONNX/g2p/alignment in
Node and in headless Chrome), and type-checks the public API as a consumer would.
`npm run typecheck` checks types without emitting.

The Python scripts in `scripts/` (model conversion, quantization, asset export,
oracle generation) need a virtualenv with the ML deps:

```bash
python -m venv .venv
.venv/bin/pip install torch transformers optimum onnx onnxruntime onnxscript \
  huggingface_hub numpy soundfile librosa nltk g2p_en g2pM praatio
```

The g2p assets in `assets/` are generated from these (`npm run export-g2p`); the
oracle JSONs the JS tests compare against are produced by `scripts/*_oracle.py`.

## License

MIT — see [LICENSE](./LICENSE). Bundled data/models retain their own licenses;
see [ATTRIBUTION.md](./ATTRIBUTION.md) (charsiu: MIT, g2p_en: Apache-2.0,
CMUdict: BSD-style).
