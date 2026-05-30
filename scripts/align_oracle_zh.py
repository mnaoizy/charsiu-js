import sys, os, json
import numpy as np, torch
sys.path.insert(0, os.path.join("charsiu_src","src"))
from itertools import groupby
import nltk
from processors import CharsiuPreprocessor_zh
from utils import forced_align, seq2duration
from transformers import AutoModelForCTC, AutoConfig

MODEL_ID="charsiu/zh_w2v2_tiny_fc_10ms"; RES=0.01; SILTH=4
text=open("sample/zh_transcript.txt").read().strip()
print("text:",text)
proc=CharsiuPreprocessor_zh()
sil_idx=proc.sil_idx; print("sil_idx:",sil_idx)
phones,words=proc.get_phones_and_words(text)
phone_ids=proc.get_phone_ids(phones)
print("phones:",phones); print("phone_ids:",phone_ids,"max",max(phone_ids))
cfg=AutoConfig.from_pretrained(MODEL_ID); cfg.architectures=["Wav2Vec2ForCTC"]
model=AutoModelForCTC.from_pretrained(MODEL_ID,config=cfg).eval()
print("model out dim:",model.config.vocab_size)
audio=proc.audio_preprocess("sample/zh_sample.wav",sr=16000)
with torch.no_grad(): logits=model(torch.Tensor(audio).unsqueeze(0)).logits
cost=torch.softmax(logits,dim=-1).numpy().squeeze()
print("cost shape:",cost.shape)
preds=np.argmax(cost,axis=-1); sil_mask=[]
for k,g in groupby(preds):
    g=list(g)
    sil_mask += ([-1]*len(g)) if (k==sil_idx and len(g)<SILTH) else g
sil_mask=np.array(sil_mask); nonsil=np.argwhere(sil_mask!=sil_idx).squeeze()
ap=forced_align(cost[nonsil,:],phone_ids[1:-1])
aphones=[proc.mapping_id2phone(phone_ids[1:-1][i]) for i in ap]
merged=[]; c=0
for i in sil_mask:
    if i==sil_idx: merged.append("[SIL]")
    else: merged.append(aphones[c]); c+=1
dur=seq2duration(merged,resolution=RES)
wdur=proc.align_words(dur,phones,words)
for s,e,p in dur: print(f"  {s:.2f}-{e:.2f} {p}")
print("words:",wdur)
os.makedirs("sample",exist_ok=True)
json.dump([[s,e,p] for s,e,p in dur],open("sample/zh_align_oracle.json","w"),ensure_ascii=False,indent=1)
json.dump([[s,e,w] for s,e,w in wdur],open("sample/zh_align_oracle_words.json","w"),ensure_ascii=False,indent=1)
json.dump({"phone_ids":phone_ids,"sil_idx":int(sil_idx)},open("sample/zh_align_spec.json","w"),indent=1)
print("saved zh oracle")
