"""Export g2pM (Mandarin hanzi->pinyin) resources into JS-loadable assets."""
import os, json, pickle
import numpy as np
import g2pM

G = os.path.dirname(g2pM.__file__)
os.makedirs("assets", exist_ok=True)
load = lambda f: pickle.load(open(os.path.join(G, f), "rb"))

cedict = load("digest_cedict.pkl")
char2idx = load("char2idx.pkl")
class2idx = load("class2idx.pkl")
sd = load("np_ckpt.pkl")

json.dump(cedict, open("assets/g2pm_cedict.json", "w"), ensure_ascii=False)
json.dump(char2idx, open("assets/g2pm_char2idx.json", "w"), ensure_ascii=False)
# idx2class as an array indexed by class id
idx2class = [None] * len(class2idx)
for pron, idx in class2idx.items():
    idx2class[idx] = pron
json.dump(idx2class, open("assets/g2pm_idx2class.json", "w"), ensure_ascii=False)

# BiLSTM + FC weights -> one float32 blob + manifest
manifest, blob = {}, bytearray()
for k, v in sd.items():
    arr = v.astype(np.float32).ravel(order="C")
    manifest[k] = {"offset": len(blob) // 4, "shape": list(v.shape)}
    blob += arr.tobytes()
open("assets/g2pm_lstm.bin", "wb").write(blob)
json.dump(manifest, open("assets/g2pm_lstm.json", "w"), indent=1)

print(f"cedict {len(cedict)} | char2idx {len(char2idx)} | classes {len(idx2class)} | "
      f"weights {len(blob)/1e6:.1f}MB")
print("special tokens:", {t: char2idx[t] for t in ["<PAD>", "<UNK>", "시", "끝"]})
