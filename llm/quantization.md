# Quantization

## Concepts
```
int8: [-128, 127] or [0, 255]
int4: [-8, 7] or [0, 15]

during quantization, we have: weight = scale * q + zero_point
scale = (Vmax-Vmin)/(Qmax-Qmin)
zero_point = Qmin - round(Vmin/scale)

symmetric: zero_point = 0, only store scale
asymmetric (affine): zero_point != 0, store both scale and zero_point

int4 requires weight packing (2 int4 packed together into int8 for storage and transfer)
int4 needs advanced techniques like gptq/awq to get good performance

fp8 (e5m2 or e4m3) provides more accurate result than int8, but requires hardware support (H100/MI300)
fp4 (e1m2 or e2m1) only supported in B200

Post-Training Quantization (PTQ): Quantization is applied after the model is fully trained.
Quantization-Aware Training (QAT): Quantization effects are simulated during training by inserting “fake quantization”, lets the model adapt to quantization.
```

## QQUF
```
inference with llama.cpp or ollama
cannot be converted back to original model, information is lost

Q4_K_M, Q8_0

Q — Quantized
Digit after Q (e.g. Q4) — the number of bits in each quantized weight memory storage.
K — Grouped quantization (uses per-group scale + zero point)
0 — Ungrouped (older style quantization)
1 — Another legacy quantization format
S — Small precision (fastest, lower accuracy)
M — Medium precision
L —Large precision (slowest, higher accuracy)

Grouped quantization:
weight = scale * q + zero_point
eg, origin values: [0.11, 0.09, 0.12, 0.10, …, 0.15]
if use Q4_K_M, scale=0.005, zero_point=0.08,
quantized values: [6, 2, 8, 4, …, 14]

Ungrouped quantization:
Use global quantization, a single scale and zero point
shared across the entire tensor
```

## GPTQ
```
method: quantizes each weight by finding a compressed version of that weight
given a layer l, with a weight matrix W, layer input Xl, find quantized weight W^:
W^l = argmin(||WlX-W^lX||^2)
```

## AWQ

## TorchAO

## Evalution
- relative perplexity, quantized model vs float32/16 model
- specific task accuracy