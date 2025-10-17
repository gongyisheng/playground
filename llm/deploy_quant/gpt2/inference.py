import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from configs import HF_MODEL_NAME, EXAMPLE_QUERY

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# ---------- FP32 ----------
print("=" * 50)
print("Running FP32 Inference")
print("=" * 50)
tokenizer = GPT2Tokenizer.from_pretrained(HF_MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(HF_MODEL_NAME)
model.to(device)
model.eval()

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

text = EXAMPLE_QUERY
print(f"Input text: {text}\n")

start = time.time()
encoded_input = tokenizer(text, return_tensors='pt').to(device)
input_length = encoded_input['input_ids'].shape[1]

# Generate text
with torch.no_grad():
    output = model.generate(
        encoded_input['input_ids'],
        max_length=input_length + 50,  # Generate 50 additional tokens
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )

st_time = time.time() - start

# Decode the full output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
tokens_generated = output.shape[1] - input_length

print(f"[FP32] Tokens generated: {tokens_generated}")
print(f"[FP32] Time: {st_time:.4f}s")
print(f"[FP32] Tokens/sec: {tokens_generated/st_time:.2f}")
print(f"[FP32] Generated text:\n  {generated_text}\n")

# Clean up
del model
if device == "cuda":
    torch.cuda.empty_cache()


# ---------- FP16 ----------
print("=" * 50)
print("Running FP16 Inference")
print("=" * 50)
tokenizer = GPT2Tokenizer.from_pretrained(HF_MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(HF_MODEL_NAME, torch_dtype=torch.float16)
model.to(device)
model.eval()

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

text = EXAMPLE_QUERY
print(f"Input text: {text}\n")

start = time.time()
encoded_input = tokenizer(text, return_tensors='pt').to(device)
input_length = encoded_input['input_ids'].shape[1]

# Generate text
with torch.no_grad():
    output = model.generate(
        encoded_input['input_ids'],
        max_length=input_length + 50,  # Generate 50 additional tokens
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )

st_time = time.time() - start

# Decode the full output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
tokens_generated = output.shape[1] - input_length

print(f"[FP16] Tokens generated: {tokens_generated}")
print(f"[FP16] Time: {st_time:.4f}s")
print(f"[FP16] Tokens/sec: {tokens_generated/st_time:.2f}")
print(f"[FP16] Generated text:\n  {generated_text}\n")

# Clean up
del model
if device == "cuda":
    torch.cuda.empty_cache()


# ---------- BF16 (BFloat16) ----------
print("=" * 50)
print("Running BF16 Inference")
print("=" * 50)
tokenizer = GPT2Tokenizer.from_pretrained(HF_MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(HF_MODEL_NAME, torch_dtype=torch.bfloat16)
model.to(device)
model.eval()

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

text = EXAMPLE_QUERY
print(f"Input text: {text}\n")

start = time.time()
encoded_input = tokenizer(text, return_tensors='pt').to(device)
input_length = encoded_input['input_ids'].shape[1]

# Generate text
with torch.no_grad():
    output = model.generate(
        encoded_input['input_ids'],
        max_length=input_length + 50,  # Generate 50 additional tokens
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )

st_time = time.time() - start

# Decode the full output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
tokens_generated = output.shape[1] - input_length

print(f"[BF16] Tokens generated: {tokens_generated}")
print(f"[BF16] Time: {st_time:.4f}s")
print(f"[BF16] Tokens/sec: {tokens_generated/st_time:.2f}")
print(f"[BF16] Generated text:\n  {generated_text}\n")

# Clean up
del model
if device == "cuda":
    torch.cuda.empty_cache()

print("✅ Inference complete.")
