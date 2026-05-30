import json
from g2pM import G2pM
m = G2pM()
TESTS = ["快速的棕色狐狸","我爱北京天安门","他长大了头发很长","行不行银行","音乐很好听",
         "重要的重量","一行白鹭上青天","中国人民银行","我们都喜欢学习","这个东西不便宜"]
out = [{"text": t, "pinyin": m(t, tone=True, char_split=False)} for t in TESTS]
json.dump(out, open("test/g2pm_oracle.json","w"), ensure_ascii=False, indent=0)
print("wrote", len(out), "cases")
for c in out[:3]: print(" ", c["text"], "->", " ".join(c["pinyin"]))
