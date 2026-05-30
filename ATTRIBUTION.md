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

## g2pM (for planned Mandarin support)
- https://github.com/kakaobrain/g2pM — used as the reference for Mandarin
  grapheme-to-phoneme; not yet ported.
