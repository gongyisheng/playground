import time
from sentence_transformers import SentenceTransformer
import torch

from configs import HF_MODEL_NAME, EXAMPLE_TEXTS

# ---------- FP32 ----------
sentence_model = SentenceTransformer(HF_MODEL_NAME)

start = time.time()
st_embeddings = sentence_model.encode(EXAMPLE_TEXTS, convert_to_numpy=True)
st_time = time.time() - start

print(f"FP32 Inference time: {st_time:.3f}s")


# ---------- FP16 ----------
sentence_model = SentenceTransformer(HF_MODEL_NAME, model_kwargs={"torch_dtype": torch.float16})

start = time.time()
st_embeddings = sentence_model.encode(EXAMPLE_TEXTS, convert_to_numpy=True)
st_time = time.time() - start

print(f"FP16 Inference time: {st_time:.3f}s")


# ---------- BP16 ----------
sentence_model = SentenceTransformer(HF_MODEL_NAME, model_kwargs={"torch_dtype": torch.bfloat16})

start = time.time()
st_embeddings = sentence_model.encode(EXAMPLE_TEXTS, convert_to_numpy=True)
st_time = time.time() - start

print(f"BP16 Inference time: {st_time:.3f}s")

print("✅ Comparison complete.")