"""Generate g2p_en ground truth for a test set -> test/g2p_oracle.json"""
import json, os, sys
sys.path.insert(0, os.path.join("charsiu_src", "src"))
import nltk
for pkg in ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"]:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass
from g2p_en import G2p

TESTS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world",
    "she sells seashells by the seashore",
    "supercalifragilisticexpialidocious",  # OOV -> GRU
    "I'm gonna test the aligner's robustness",
    "antidisestablishmentarianism",          # OOV
    "transformers run in javascript now",     # 'javascript' likely OOV
    "a blockchain cryptocurrency wallet",     # several OOV
    "numbers like 250 and 1999 and 42",       # number expansion
    "well-known multi-word phrases",
    "what time is it, please?",
    "the cat sat on the mat",
    "voice activity detection threshold",
    "phonetic alignment with neural networks",
    "onomatopoeia and serendipity",
]

g2p = G2p()
out = [{"text": t, "phones": g2p(t)} for t in TESTS]
json.dump(out, open("test/g2p_oracle.json", "w"), ensure_ascii=False, indent=0)
print(f"wrote {len(out)} cases to test/g2p_oracle.json")
for c in out[:4]:
    print(" ", c["text"], "->", " ".join(c["phones"]))
