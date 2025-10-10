import time
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
import onnxruntime as ort

from configs import HF_MODEL_NAME, ONNX_MODEL_PATH, EXAMPLE_QUERY, EXAMPLE_DOCUMENTS

# ---------- LOADERS ----------
print("🔹 Loading tokenizer and reranker model...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
reranker = CrossEncoder(HF_MODEL_NAME)

# ---------- ONNX Runtime session ----------
providers = ["CPUExecutionProvider"]
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

# ---------- Prepare inputs (query-document pairs) ----------
# Rerankers take pairs of [query, document] as input
pairs = [[EXAMPLE_QUERY, doc] for doc in EXAMPLE_DOCUMENTS]

print(f"\n🔹 Query: {EXAMPLE_QUERY}")
print(f"🔹 Ranking {len(EXAMPLE_DOCUMENTS)} documents...\n")

# ---------- Run inference ----------
print("🔹 Running inference...")

# SentenceTransformers CrossEncoder
start = time.time()
st_scores = reranker.predict(pairs)
st_time = time.time() - start

# ONNX - Tokenize pairs
inputs = tokenizer(pairs, padding=True, return_tensors="pt")
onnx_inputs = {
    "input_ids": inputs["input_ids"].cpu().numpy(),
    "attention_mask": inputs["attention_mask"].cpu().numpy(),
}

start = time.time()
onnx_outputs = session.run(None, onnx_inputs)

# The output is logits with shape [batch, 1] or [batch]
# For reranker models, we need to squeeze to get scores
logits = onnx_outputs[0]
if logits.ndim > 1 and logits.shape[-1] == 1:
    logits = logits.squeeze(-1)  # [batch, 1] -> [batch]

# Apply sigmoid to normalize scores to [0, 1] range (same as CrossEncoder)
# torch.sigmoid is numerically stable for both large positive and negative values
onnx_scores = torch.sigmoid(torch.from_numpy(logits)).numpy()

onnx_time = time.time() - start

# ---------- Display Results ----------
print("\n🔹 Results:")
print(f"Speed: CrossEncoder = {st_time:.4f}s | ONNX = {onnx_time:.4f}s")

# Sort documents by ONNX scores
ranked_indices = np.argsort(onnx_scores)[::-1]

# Compare CrossEncoder vs ONNX scores
print("\n🔹 Score Comparison (CrossEncoder vs ONNX):")
for i, doc in enumerate(EXAMPLE_DOCUMENTS):
    print(f"  Doc {i+1}: CrossEncoder={st_scores[i]:.4f}, ONNX={onnx_scores[i]:.4f}, Diff={abs(st_scores[i]-onnx_scores[i]):.4f}")

print("\n✅ Reranking complete.")
