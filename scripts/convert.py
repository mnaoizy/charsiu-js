"""
Proof-of-concept: convert a charsiu Wav2Vec2ForFrameClassification aligner to ONNX.

Key insight: charsiu's Wav2Vec2ForFrameClassification subclasses Wav2Vec2ForCTC and
shares an *identical* network graph (wav2vec2 backbone + dropout + linear lm_head,
output shape (batch, frames, vocab)). Only the loss differs. So we can load the
published weights as a standard Wav2Vec2ForCTC and export that graph to ONNX.

Outputs (into ./models/<short>/):
  - model.onnx              the exported graph
  - labels.json            id -> phone label map + feature-extractor settings
  - parity.json            reference {input, torch_logits, onnx_logits, max_abs_diff}
"""
import json, sys, os
import numpy as np
import torch
from transformers import AutoModelForCTC, AutoConfig, AutoFeatureExtractor

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "charsiu/en_w2v2_fc_10ms"
SHORT = MODEL_ID.split("/")[-1]
OUT = os.path.join("models", SHORT)
os.makedirs(OUT, exist_ok=True)

print(f"[1/6] loading config for {MODEL_ID}")
config = AutoConfig.from_pretrained(MODEL_ID)
# charsiu publishes architectures=["Wav2Vec2ForFrameClassification"]; the graph is
# identical to CTC, so normalize it so every downstream loader agrees.
config.architectures = ["Wav2Vec2ForCTC"]
print("    model_type:", config.model_type, "| vocab/num_labels:", getattr(config, "vocab_size", None))

print("[2/6] loading weights as Wav2Vec2ForCTC")
model = AutoModelForCTC.from_pretrained(MODEL_ID, config=config)
model.eval()

# feature extractor: wav2vec2 just normalizes raw 16k waveform (no mel) -> easy in JS
try:
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    fe_cfg = {
        "sampling_rate": getattr(fe, "sampling_rate", 16000),
        "do_normalize": getattr(fe, "do_normalize", True),
        "return_attention_mask": getattr(fe, "return_attention_mask", False),
    }
except Exception as e:
    print("    (no feature extractor, using defaults)", e)
    fe_cfg = {"sampling_rate": 16000, "do_normalize": True, "return_attention_mask": False}

# id -> phone label
id2label = config.id2label if getattr(config, "id2label", None) else {}
labels = {str(i): id2label.get(i, id2label.get(str(i), str(i))) for i in range(getattr(config, "vocab_size", len(id2label)))}
with open(os.path.join(OUT, "labels.json"), "w") as f:
    json.dump({"id2label": labels, "feature_extractor": fe_cfg, "model_id": MODEL_ID}, f, indent=2, ensure_ascii=False)
print(f"    saved {len(labels)} labels + feature-extractor cfg")

print("[3/6] torch reference forward pass")
SAMPLES = 16000  # 1 second @ 16kHz
torch.manual_seed(0)
dummy = torch.randn(1, SAMPLES)
# normalize like the feature extractor (zero-mean unit-var)
if fe_cfg["do_normalize"]:
    dummy = (dummy - dummy.mean()) / torch.sqrt(dummy.var() + 1e-7)
with torch.no_grad():
    torch_logits = model(dummy).logits  # (1, frames, vocab)
print("    torch logits shape:", tuple(torch_logits.shape))

print("[4/6] exporting to ONNX")
onnx_path = os.path.join(OUT, "model.onnx")
torch.onnx.export(
    model,
    (dummy,),
    onnx_path,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={"input_values": {0: "batch", 1: "samples"},
                  "logits": {0: "batch", 1: "frames"}},
    opset_version=14,
    do_constant_folding=True,
)
print("    wrote", onnx_path, f"({os.path.getsize(onnx_path)/1e6:.1f} MB)")

print("[5/6] verifying with onnxruntime")
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
onnx_logits = sess.run(["logits"], {"input_values": dummy.numpy()})[0]
diff = float(np.max(np.abs(onnx_logits - torch_logits.numpy())))
print("    onnx logits shape:", onnx_logits.shape, "| max|torch-onnx| =", diff)

print("[6/6] saving parity reference for the JS side")
# store a short slice so the JS test can confirm bit-for-bit-ish parity
parity = {
    "input_values": dummy.numpy().astype(np.float32).flatten().tolist(),
    "logits_shape": list(onnx_logits.shape),
    # first frame's logits as the comparison fingerprint
    "logits_frame0": onnx_logits[0, 0, :].astype(np.float32).tolist(),
    "argmax_per_frame": onnx_logits[0].argmax(-1).astype(int).tolist(),
    "max_abs_diff_torch_onnx": diff,
}
with open(os.path.join(OUT, "parity.json"), "w") as f:
    json.dump(parity, f)
print("    done. frames:", onnx_logits.shape[1], "| ~", onnx_logits.shape[1], "frames for 1s audio")
print("\nSUCCESS" if diff < 1e-3 else "\nWARN: parity diff high")
