import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from prompt import SFT_PROMPT
from utils import DataWriter

# Configuration
model_path = "./outputs/qwen3-0.6b-sft/checkpoint-1300"
input_path = "outputs/random_tasks.jsonl"
output_path = "outputs/eval_result.jsonl"

# Load model and tokenizer
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model = model.to("cuda", dtype=torch.bfloat16)
model.eval()

# Load input data
print(f"Loading data from {input_path}...")
data = []
with open(input_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print(f"Evaluating {len(data)} examples...")

# Evaluate probabilities for each option
with DataWriter(output_path, mode='w') as writer:
    for record in tqdm(data):
        task = record['task']
        options = record['options']

        # Format prompt
        options_str = ", ".join(options)
        prompt_text = SFT_PROMPT.format(task=task, options=options_str)

        # Loop through each option and calculate its probability
        for option in options:
            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": option},
            ]

            # Apply chat template for full text
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Tokenize full text
            full_encoding = tokenizer(full_text, return_tensors="pt").to("cuda")
            input_ids = full_encoding['input_ids']

            # Tokenize user text to find where assistant starts
            user_text = tokenizer.apply_chat_template(
                messages[:1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            user_encoding = tokenizer(user_text, return_tensors="pt")
            user_length = user_encoding['input_ids'].shape[1]

            # Tokenize assistant tokens
            assistant_tokens = tokenizer.encode(option + tokenizer.eos_token, add_special_tokens=False)
            assistant_length = len(assistant_tokens)

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits

            # Calculate probability for assistant tokens
            log_probs = []
            for i in range(assistant_length):
                token_idx = user_length + i - 1  # -1 because logits are shifted
                target_token_id = assistant_tokens[i]

                token_logits = logits[0, token_idx, :]
                token_log_probs = F.log_softmax(token_logits, dim=-1)
                log_prob = token_log_probs[target_token_id].item()
                log_probs.append(log_prob)

            # Calculate total probability (sum of log probs = log of product of probs)
            total_log_prob = sum(log_probs)
            total_proba = torch.exp(torch.tensor(total_log_prob)).item()

            # Save one line per option
            result = {
                "uuid": record.get("uuid"),
                "task": task,
                "options": options,
                "chosen_option": option,
                "proba": total_proba
            }
            writer.write(result)

print(f"Done! Results saved to {output_path}")
