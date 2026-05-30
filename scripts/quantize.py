"""Dynamic INT8 quantization of a converted charsiu ONNX model for web delivery.
Produces model_quantized.onnx next to model.onnx (transformers.js naming)."""
import sys, os
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

SHORT = sys.argv[1] if len(sys.argv) > 1 else "en_w2v2_fc_10ms"
d = os.path.join("models", SHORT)
src = os.path.join(d, "model.onnx")
prep = os.path.join(d, "model_prep.onnx")
dst = os.path.join(d, "model_quantized.onnx")

print(f"pre-processing {src}")
quant_pre_process(src, prep, skip_symbolic_shape=True)

print("dynamic INT8 quantization (MatMul/Gemm — the transformer weights)")
# wav2vec2's feature-encoder Conv weights are computed (weight-norm), not
# initializers, so Conv can't be dynamically quantized; the transformer MatMuls
# are the bulk of the size anyway.
quantize_dynamic(prep, dst, weight_type=QuantType.QInt8,
                 op_types_to_quantize=["MatMul", "Gemm"])

def size(p):
    total = os.path.getsize(p)
    if os.path.exists(p + ".data"): total += os.path.getsize(p + ".data")
    return total / 1e6

for f in (prep, prep + ".data"):
    if os.path.exists(f): os.remove(f)
print(f"\nfp32: {size(src):.1f} MB -> q8: {size(dst):.1f} MB  ({size(dst)/size(src)*100:.0f}%)")
print("wrote", dst)
