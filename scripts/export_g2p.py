"""Export g2p_en resources into JS-loadable assets under ./assets/."""
import os, json, struct
import numpy as np
import g2p_en
from nltk.corpus import cmudict
import nltk
for pkg in ["cmudict"]:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass

G = os.path.dirname(g2p_en.__file__)
os.makedirs("assets", exist_ok=True)

# 1) CMUdict — first pronunciation per word (that's all g2p_en uses)
cmu = cmudict.dict()
first = {w: " ".join(prons[0]) for w, prons in cmu.items()}
json.dump(first, open("assets/cmudict.json", "w"))
print(f"cmudict.json: {len(first)} words, {os.path.getsize('assets/cmudict.json')/1e6:.1f} MB")

# 2) homographs.en -> JSON {word: [pron1[], pron2[], pos1]}
homo = {}
for line in open(os.path.join(G, "homographs.en"), encoding="utf8").read().splitlines():
    if line.startswith("#") or "|" not in line: continue
    headword, pron1, pron2, pos1 = line.strip().split("|")
    homo[headword.lower()] = [pron1.split(), pron2.split(), pos1]
json.dump(homo, open("assets/homographs.json", "w"))
print(f"homographs.json: {len(homo)} entries")

# 3) GRU seq2seq weights -> one float32 blob + manifest
v = np.load(os.path.join(G, "checkpoint20.npz"))
manifest, blob = {}, bytearray()
for k in v.files:
    arr = v[k].astype(np.float32).ravel(order="C")
    manifest[k] = {"offset": len(blob) // 4, "shape": list(v[k].shape)}
    blob += arr.tobytes()
open("assets/g2p_gru.bin", "wb").write(blob)
json.dump(manifest, open("assets/g2p_gru.json", "w"), indent=1)
print(f"g2p_gru.bin: {len(blob)/1e6:.1f} MB | arrays: {list(manifest)}")

# 4) vocab (graphemes/phonemes) so JS matches exactly
from g2p_en import G2p
g = G2p.__new__(G2p)
G2p.__init__.__wrapped__ if False else None
graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0','AA1','AA2','AE0','AE1','AE2','AH0','AH1','AH2','AO0',
    'AO1','AO2','AW0','AW1','AW2','AY0','AY1','AY2','B','CH','D','DH',
    'EH0','EH1','EH2','ER0','ER1','ER2','EY0','EY1','EY2','F','G','HH',
    'IH0','IH1','IH2','IY0','IY1','IY2','JH','K','L','M','N','NG','OW0','OW1',
    'OW2','OY0','OY1','OY2','P','R','S','SH','T','TH','UH0','UH1','UH2','UW',
    'UW0','UW1','UW2','V','W','Y','Z','ZH']
json.dump({"graphemes": graphemes, "phonemes": phonemes}, open("assets/g2p_vocab.json", "w"))
print(f"g2p_vocab.json: {len(graphemes)} graphemes, {len(phonemes)} phonemes")
