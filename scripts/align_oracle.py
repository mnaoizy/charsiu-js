"""
Oracle: run charsiu's REAL forced-alignment pipeline (its own processor + g2p +
DTW utils) on the synthetic sample, to produce ground-truth phone boundaries.

We reuse charsiu's text side (CharsiuPreprocessor_en) and alignment utils
(forced_align / seq2duration) verbatim, and the published wav2vec2 weights loaded
as Wav2Vec2ForCTC (proven bit-identical to the frame-classification graph).

Outputs:
  sample/align_oracle.json   [[start, end, phone], ...]  ground truth
  sample/align_spec.json     {phone_ids, sil_idx, sil_threshold, resolution,
                              id2phone, audio_len} for the JS port to consume
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.join("charsiu_src", "src"))
from itertools import groupby
import nltk
for pkg in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"]:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass

from processors import CharsiuPreprocessor_en
from utils import forced_align, seq2duration
from transformers import AutoModelForCTC, AutoConfig

MODEL_ID = "charsiu/en_w2v2_fc_10ms"
RESOLUTION = 0.01
SIL_THRESHOLD = 4

text = open("sample/transcript.txt").read().strip()
print("transcript:", text)

print("[1/5] init charsiu English processor (tokenizer + g2p_en)")
proc = CharsiuPreprocessor_en()
sil_idx = proc.sil_idx
print("    sil token:", proc.sil, "| sil_idx:", sil_idx)

print("[2/5] g2p + phone ids")
phones, words = proc.get_phones_and_words(text)
phone_ids = proc.get_phone_ids(phones)
print("    phones:", phones)
print("    phone_ids (w/ sil ends):", phone_ids)

print("[3/5] load + run model")
config = AutoConfig.from_pretrained(MODEL_ID); config.architectures = ["Wav2Vec2ForCTC"]
model = AutoModelForCTC.from_pretrained(MODEL_ID, config=config).eval()
audio = proc.audio_preprocess("sample/sample.wav", sr=16000)
audio_t = torch.Tensor(audio).unsqueeze(0)
with torch.no_grad():
    logits = model(audio_t).logits
cost = torch.softmax(logits, dim=-1).numpy().squeeze()  # (frames, vocab)
print("    cost (softmax) shape:", cost.shape)

print("[4/5] sil mask + DTW forced align (charsiu utils)")
preds = np.argmax(cost, axis=-1)
sil_mask = []
for key, group in groupby(preds):
    group = list(group)
    if key == sil_idx and len(group) < SIL_THRESHOLD:
        sil_mask += [-1 for _ in group]
    else:
        sil_mask += group
sil_mask = np.array(sil_mask)
nonsil_idx = np.argwhere(sil_mask != sil_idx).squeeze()
aligned_phone_ids = forced_align(cost[nonsil_idx, :], phone_ids[1:-1])
aligned_phones = [proc.mapping_id2phone(phone_ids[1:-1][i]) for i in aligned_phone_ids]

# merge silence back
pred_phones = []
count = 0
for i in sil_mask:
    if i == sil_idx:
        pred_phones.append("[SIL]")
    else:
        pred_phones.append(aligned_phones[count]); count += 1
duration = seq2duration(pred_phones, resolution=RESOLUTION)

print("[5/5] alignment:")
for s, e, p in duration:
    print(f"    {s:5.2f} - {e:5.2f}  {p}")

word_dur = proc.align_words(duration, phones, words)
print("words:", word_dur)

os.makedirs("sample", exist_ok=True)
json.dump([[s, e, p] for s, e, p in duration], open("sample/align_oracle.json", "w"), indent=1)
json.dump([[s, e, w] for s, e, w in word_dur], open("sample/align_oracle_words.json", "w"), indent=1, ensure_ascii=False)
# id2phone for the full vocab so JS can label
id2phone = {str(i): proc.mapping_id2phone(i) for i in range(cost.shape[1])}
json.dump({
    "phones": phones, "words": words, "phone_ids": phone_ids,
    "sil_idx": int(sil_idx), "sil_threshold": SIL_THRESHOLD, "resolution": RESOLUTION,
    "id2phone": id2phone, "frames": int(cost.shape[1]) and int(cost.shape[0]),
}, open("sample/align_spec.json", "w"), indent=1, ensure_ascii=False)
print("\nsaved sample/align_oracle.json + sample/align_spec.json")
