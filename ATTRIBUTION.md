# Attribution

charsiu-js is a JavaScript port of existing open-source work. The alignment
algorithm, models, and g2p resources are derived from the projects below.

## charsiu
- https://github.com/lingjzhu/charsiu — MIT License, Copyright (c) 2021 jzhu
- The forced-alignment pipeline (audio normalization, DTW alignment over
  wav2vec2 frame logits, `seq2duration`, word alignment) is ported from charsiu's
  `Charsiu.py` / `processors.py` / `utils.py`.
- The acoustic models (`charsiu/en_w2v2_fc_10ms`, `charsiu/zh_w2v2_tiny_fc_10ms`,
  …) are published on the Hugging Face Hub under the charsiu organization. The
  ONNX exports used by this package are conversions of those weights.

## g2p_en
- https://github.com/Kyubyong/g2p — Apache License 2.0,
  Copyright Kyubyong Park & Jongseok Kim
- `src/g2p.mjs` ports the g2p_en algorithm (text normalization, CMUdict lookup,
  homograph handling, and the GRU seq2seq OOV predictor). The bundled
  `assets/g2p_gru.bin` contains the g2p_en `checkpoint20.npz` weights.

## CMU Pronouncing Dictionary (CMUdict)
- http://www.speech.cs.cmu.edu/cgi-bin/cmudict — BSD-2-Clause-style license,
  Copyright (c) Carnegie Mellon University.
- Bundled (first pronunciation per word) as `assets/cmudict.json`.

## g2pM (Mandarin)
- https://github.com/kakaobrain/g2pM — the reference for Mandarin
  grapheme-to-phoneme. `src/g2pm.ts` ports g2pM (CEDICT lookup + a BiLSTM that
  disambiguates polyphonic characters); weights/data are bundled as
  `assets/g2pm_*`. See the g2pM repository (and CC-CEDICT) for their licenses.

## Japanese
- **tokana** — https://github.com/mnaoizy/tokana — morphological analysis, used
  via the optional `tokana` peer dependency by `createNodeAlignerJa`.
- **mecab-ipadic** — the IPADIC dictionary, compiled with tokana for tokenization
  (not bundled; built locally via `npm run setup-dict`). Distributed under its own
  permissive license; see the mecab-ipadic source.
- **mecab-ipadic-NEologd** — https://github.com/neologd/mecab-ipadic-neologd —
  Apache License 2.0. Optional larger dictionary (`npm run setup-dict -- neologd`).
- **prj-beatrice/japanese-hubert-base-phoneme-ctc** —
  https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc — Apache
  License 2.0. The Japanese acoustic model (HuBERT phoneme-CTC); the ONNX export is
  a conversion of these weights. Fine-tuned from `rinna/japanese-hubert-base`
  (Apache-2.0); phoneme labels were produced with pyopenjtalk-plus.
- **OpenJTalk / pyopenjtalk-plus** — the reference for the OpenJTalk phone set used
  by `src/g2p-ja.ts`, and used at dev time to generate the Japanese phoneme oracle.
