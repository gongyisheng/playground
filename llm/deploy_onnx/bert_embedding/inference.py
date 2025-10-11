import time
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
from numpy.linalg import norm

from configs import HF_MODEL_NAME, ONNX_MODEL_PATH, EXAMPLE_TEXTS

# ---------- HELPER FUNCTIONS ----------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ---------- LOADERS ----------
print("🔹 Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
sentence_model = SentenceTransformer(HF_MODEL_NAME)

# ---------- ONNX Runtime session ----------
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

# ---------- Prepare inputs ----------
inputs = tokenizer(EXAMPLE_TEXTS, padding=True, return_tensors="pt")

# Convert to NumPy for ONNX
onnx_inputs = {
    "input_ids": inputs["input_ids"].cpu().numpy(),
    "attention_mask": inputs["attention_mask"].cpu().numpy(),
}

# ---------- Run inference ----------
print("🔹 Running inference...")

# PyTorch (SentenceTransformer)
start = time.time()
st_embeddings = sentence_model.encode(EXAMPLE_TEXTS, convert_to_numpy=True)
st_time = time.time() - start

# ONNX
start = time.time()
onnx_outputs = session.run(None, onnx_inputs)
onnx_time = time.time() - start

# Mean pool for ONNX output (last_hidden_state)
last_hidden_state = onnx_outputs[0]
attention_mask = onnx_inputs["attention_mask"]
mask_expanded = np.expand_dims(attention_mask, -1)
onnx_embeddings = (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1)

# ---------- Compare ----------
print("\n🔹 Results:")
print(f"SentenceTransformer embedding shape: {st_embeddings.shape}")
print(f"ONNX embedding shape: {onnx_embeddings.shape}")
print(f"Speed: SentenceTransformer = {st_time:.4f}s | ONNX = {onnx_time:.4f}s")

# Cosine similarity between ONNX and SentenceTransformer for each sentence
for i in range(len(EXAMPLE_TEXTS)):
    sim = cosine_similarity(st_embeddings[i], onnx_embeddings[i])
    print(f"  • Cosine similarity for sentence {i+1}: {sim:.4f}")

print("\n✅ Comparison complete.")
