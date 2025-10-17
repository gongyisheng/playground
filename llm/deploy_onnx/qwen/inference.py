import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

from configs import HF_MODEL_NAME, ONNX_MODEL_PATH, EXAMPLE_QUERY

# ---------- LOADERS ----------
print("🔹 Loading tokenizer and PyTorch model...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
pytorch_model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
pytorch_model.eval()

# ---------- Load ONNX model ----------
print(f"🔹 Loading ONNX model from: {ONNX_MODEL_PATH}")
onnx_model = ORTModelForCausalLM.from_pretrained(ONNX_MODEL_PATH)

# ---------- Prepare test prompt ----------
test_prompt = EXAMPLE_QUERY
print(f"\n🔹 Test prompt: '{test_prompt}'")

# Generation parameters
max_new_tokens = 100
temperature = 0.7
do_sample = True

# ---------- Run PyTorch generation ----------
print("\n🔹 Running PyTorch text generation...")
inputs = tokenizer(test_prompt, return_tensors="pt")

start = time.time()
pytorch_outputs = pytorch_model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=do_sample,
    temperature=temperature,
    pad_token_id=tokenizer.eos_token_id
)
pytorch_time = time.time() - start

pytorch_text = tokenizer.decode(pytorch_outputs[0], skip_special_tokens=True)
pytorch_tokens_generated = pytorch_outputs.shape[1] - inputs['input_ids'].shape[1]

print(f"  Tokens generated: {pytorch_tokens_generated}")
print(f"  Time: {pytorch_time:.4f}s")
print(f"  Tokens/sec: {pytorch_tokens_generated/pytorch_time:.2f}")
print(f"\n  Generated text:\n  {pytorch_text}\n")

# ---------- Run ONNX generation ----------
print("🔹 Running ONNX text generation...")

start = time.time()
onnx_outputs = onnx_model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=do_sample,
    temperature=temperature,
    pad_token_id=tokenizer.eos_token_id
)
onnx_time = time.time() - start

onnx_text = tokenizer.decode(onnx_outputs[0], skip_special_tokens=True)
onnx_tokens_generated = onnx_outputs.shape[1] - inputs['input_ids'].shape[1]

print(f"  Tokens generated: {onnx_tokens_generated}")
print(f"  Time: {onnx_time:.4f}s")
print(f"  Tokens/sec: {onnx_tokens_generated/onnx_time:.2f}")
print(f"\n  Generated text:\n  {onnx_text}\n")

# ---------- Compare outputs ----------
print("🔹 Performance Comparison:")
print(f"  PyTorch:  {pytorch_time:.4f}s ({pytorch_tokens_generated/pytorch_time:.2f} tokens/sec)")
print(f"  ONNX:     {onnx_time:.4f}s ({onnx_tokens_generated/onnx_time:.2f} tokens/sec)")
speedup = pytorch_time / onnx_time if onnx_time > 0 else 0
print(f"  Speedup:  {speedup:.2f}x")

# ---------- Single forward pass comparison ----------
print("\n🔹 Single forward pass comparison (logits accuracy):")
inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)

with torch.no_grad():
    pytorch_output = pytorch_model(**inputs)
    onnx_output = onnx_model(**inputs)

pytorch_logits = pytorch_output.logits
onnx_logits = onnx_output.logits

# Calculate mean absolute difference in logits
pytorch_logits_np = pytorch_logits.cpu().numpy()
onnx_logits_np = onnx_logits.cpu().numpy()
logits_diff = np.abs(pytorch_logits_np - onnx_logits_np)
mean_diff = np.mean(logits_diff)
max_diff = np.max(logits_diff)

print(f"  Mean logits difference: {mean_diff:.6f}")
print(f"  Max logits difference: {max_diff:.6f}")

# Check next token prediction
pytorch_next_token = torch.argmax(pytorch_logits[0, -1, :]).item()
onnx_next_token = torch.argmax(onnx_logits[0, -1, :]).item()
print(f"  Same next token prediction: {pytorch_next_token == onnx_next_token}")

print("\n✅ Inference complete.")
