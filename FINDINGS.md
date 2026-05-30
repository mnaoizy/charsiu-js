# Feasibility proof: charsiu forced aligner in JS

**Verdict: feasible.** The charsiu wav2vec2 aligner runs in JavaScript via ONNX with
bit-exact parity to PyTorch.

## What was proven

1. **Architecture is JS-friendly.** charsiu's `Wav2Vec2ForFrameClassification`
   subclasses `Wav2Vec2ForCTC` and has an *identical* network graph (wav2vec2
   backbone + dropout + linear `lm_head`, output `(batch, frames, vocab)`). Only the
   training loss differs. So it loads cleanly as a standard `Wav2Vec2ForCTC`.
2. **Conversion works.** `scripts/convert.py` loads `charsiu/en_w2v2_fc_10ms`,
   normalizes the config to `Wav2Vec2ForCTC`, and exports to ONNX. onnxruntime
   (Python) matches PyTorch with `max|diff| = 3.7e-5`.
3. **Runs in JS.** `test/run-onnx.mjs` loads the ONNX in Node via `onnxruntime-node`
   and reproduces the PyTorch logits exactly: shape `[1,98,42]`, frame-0
   `max|diff| = 0`, argmax `98/98` frames match, ~168 ms for 1 s of audio.

## Key facts for the package design

- **Input is raw 16 kHz mono waveform** (Float32), normalized to zero-mean/unit-var.
  No mel spectrogram — preprocessing in JS is trivial (resample + normalize).
- **Frame rate is 10 ms** (98 frames for 1 s), matching the model name `_10ms`.
- **Model size: 378 MB fp32** (wav2vec2-base, ~94 M params), exported as
  `model.onnx` (graph, 1.6 MB) + `model.onnx.data` (weights). Needs quantization
  (q8 ≈ ~95 MB) for browser delivery.
- **Browser path is the same engine:** `onnxruntime-web` / `@huggingface/transformers`
  use the identical ORT runtime as `onnxruntime-node`, so the browser result will match.

## End-to-end alignment proven (JS == charsiu)

A full forced alignment of synthetic speech now runs in pure JS and matches
charsiu's own Python pipeline **exactly**.

- **Sample:** macOS `say` -> "the quick brown fox jumps over the lazy dog",
  16 kHz mono (`sample/sample.wav`, 2.74 s).
- **Oracle** (`scripts/align_oracle.py`): charsiu's real `CharsiuPreprocessor_en`
  (g2p_en + `charsiu/tokenizer_en_cmu` vocab) + `utils.forced_align` (DTW) +
  `seq2duration`, over logits from the published weights.
- **JS** (`src/align.mjs`, `npm run test:align`): WAV read -> normalize -> ONNX ->
  softmax -> silence mask -> **DTW** (librosa step sizes `[[1,1],[1,0]]`) -> merge
  silence -> `seq2duration`.
- **Result:** 32/32 segments identical, phone sequence match, **max boundary
  diff 0.00 s**, ~450 ms in Node.

### Alignment algorithm (confirmed)

- charsiu uses **DTW**, not Viterbi: cost = `-softmax(logits)[:, phoneIds]`, step
  sizes `[[1,1],[1,0]]` (each frame advances; phone index is non-decreasing).
- Silence handling: argmax runs of the `[SIL]` id shorter than `sil_threshold=4`
  are demoted to speech; long ones are excluded from DTW then merged back.
- Resolution 0.01 s (10 ms frames).

## Standalone JS package — built and verified

The whole pipeline now runs in JS (Node and browser) with no Python at inference,
each stage verified bit-for-bit against the charsiu/g2p_en Python oracles.

| Stage | File | Verification |
|-------|------|--------------|
| g2p (English) | `src/g2p.mjs` (+ `assets/`) | `npm run test:g2p` — 15/15 cases match g2p_en (incl. OOV GRU, numbers, punctuation) |
| phone vocab / ids | `src/phonemize-en.mjs` (+ `assets/vocab_en.json`) | phone ids 33/33 match charsiu |
| model inference | ONNX via onnxruntime | `npm run test:onnx` — frame logits == PyTorch |
| DTW alignment | `src/align.mjs` | `npm run test:align` — 0.00 s vs oracle |
| word align + TextGrid | `src/index.mjs` | `npm run test:standalone` — phones + words match, writes `sample/sample.TextGrid` |
| INT8 quantization | `scripts/quantize.py` | 379 MB -> **123 MB**; alignment preserved (0.01 s drift) |
| **browser (onnxruntime-web)** | `web/`, `src/assets-web.mjs` | `npm run test:browser` — headless Chrome, 0.01 s vs oracle |

Public API: `ForcedAligner.align(waveform, text)` -> `{ phones, words, phoneIds }`
(`src/index.mjs`); Node factory `createNodeAligner()` (`src/aligner-node.mjs`);
browser demo `npm run serve`.

### g2p port notes
- `g2p_en` GRU seq2seq for OOV words is ported exactly (argmax-identical on
  "supercalifragilistic…", "javascript", "blockchain", …). Weights in
  `assets/g2p_gru.bin` (3.3 MB) + manifest.
- Homographs (~371 words) currently fall back to their default pronunciation; the
  nltk POS tagger that g2p_en uses for them is not yet ported. Only affects
  homographs.

## Mandarin (Standard Chinese) — complete

Verified end-to-end in JS against the charsiu oracle, same as English:
- **Model:** `charsiu/zh_w2v2_tiny_fc_10ms` -> ONNX (90 MB, 210 tonal phone
  classes). DTW alignment (`src/align.mjs`) is reused unchanged.
- **g2p:** `src/g2pm.mjs` ports `g2pM` — CEDICT lookup plus a 1-layer BiLSTM +
  2-layer FC that disambiguates polyphonic characters. `npm run test:g2pm` →
  10/10 vs the Python (incl. 长→cháng/zhǎng, 行→xíng/háng, 重→zhòng, 乐→yuè).
- **pinyin->phones:** `src/phonemize-zh.mjs` ports `CharsiuPreprocessor_zh`
  (`transform_dict`, `_separate_syllable`, er/rhyme maps, `tokenizer_zh_pinyin`
  vocab).
- **E2E:** `npm run test:standalone-zh` — 快速的棕色狐狸 → `k uai4 s u4 d e5 …`,
  words 快 速 的 …; phones/words/ids match the oracle, 0.00 s drift.
- Factory: `createNodeAlignerZh()`. Bundled assets: `g2pm_*` (cedict 190 KB,
  BiLSTM 1.6 MB) + `vocab_zh.json`.

One gotcha fixed along the way: the alignment must NOT strip digits from phone
labels — EN vocab labels are stress-less already, but ZH labels carry tone digits
(`uai4`), so digit-stripping was removed from `align.mjs`.

## Remaining for publish
- `npm publish` + host the ONNX models (HF Hub) — needs the maintainer's accounts.
- Optional: POS tagger for English homographs; quantize the zh model; WebGPU EP;
  a zh path in the browser demo.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install torch transformers onnx onnxruntime onnxscript huggingface_hub numpy
.venv/bin/python scripts/convert.py charsiu/en_w2v2_fc_10ms   # -> models/en_w2v2_fc_10ms/
npm install
npm run test:onnx                                             # -> PASS
```
