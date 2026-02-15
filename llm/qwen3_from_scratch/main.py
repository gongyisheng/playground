import torch
import time

from config import Qwen3Config
from tokenizer import Qwen3Tokenizer
from model import Qwen3Model
from weights import load_weights
from generate import generate

MODEL_DIR = "checkpoint/Qwen3-0.6B"

# load config, tokenizer, model
config = Qwen3Config.from_model_dir(MODEL_DIR)
tokenizer = Qwen3Tokenizer.from_model_dir(MODEL_DIR)

model = Qwen3Model(config)
load_weights(model, MODEL_DIR)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# prepare prompt
messages = [{"role": "user", "content": "Give me 100 meow, just meow no other words"}]
formatted = tokenizer.apply_chat_template(messages, enable_thinking=True)
token_ids = tokenizer.encode(formatted)
prompt_tensor = torch.tensor(token_ids, device=device)

print(f"Model: {MODEL_DIR}")
print(f"Device: {device}")
print(f"Prompt tokens: {len(token_ids)}")
print("Generating...\n")

# generate
start = time.time()
with torch.no_grad():
    output_ids = generate(
        model,
        prompt_tensor,
        max_new_tokens=1024,
        temperature=1,
        top_k=-1,
        eos_token_id=config.eos_token_id,
    )
elapsed = time.time() - start

# decode and print
output_text = tokenizer.decode(output_ids)
new_tokens = len(output_ids) - len(token_ids)
print(output_text)
print(f"\n--- {new_tokens} tokens in {elapsed:.2f}s ({new_tokens/elapsed:.1f} tok/s) ---")
