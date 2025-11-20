import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------------------
# Full fine-tuning setup (optimized for 3060 12GB)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # ~1.1B params, fits on 12GB
NEW_MODEL_NAME = "manual_sft_model"
MAX_LENGTH = 256
BATCH_SIZE = 4  # Adjust based on your GPU memory
LR = 2e-5  # Lower LR for full fine-tuning (was 2e-4 for LoRA)
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------
# 2. CUSTOM DATASET CLASS (The Core Logic)
# ------------------------------------------------------------------------
class SFTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the raw text (Assumes data has a 'text' column or similar)
        # Format: ### Human: ... ### Assistant: ...
        record = self.dataset[idx]
        full_text = record["text"]
        
        # Split into Prompt vs Response so we can mask the prompt
        # NOTE: This splitting logic depends strictly on your dataset format
        split_text = full_text.split("### Assistant:")
        if len(split_text) < 2:
            # Skip bad data in real life, for now return truncated
            prompt_text = full_text[:10] 
            response_text = full_text[10:]
        else:
            prompt_text = split_text[0] + "### Assistant:"
            response_text = split_text[1]

        # 1. Tokenize the Prompt (Instructions)
        prompt_ids = self.tokenizer.encode(
            prompt_text, 
            add_special_tokens=False
        )
        
        # 2. Tokenize the Response (Target)
        response_ids = self.tokenizer.encode(
            response_text, 
            add_special_tokens=False
        ) + [self.tokenizer.eos_token_id] # Add EOS token at the end

        # 3. Combine them
        input_ids = prompt_ids + response_ids
        
        # 4. Create Labels (Mask prompt with -100)
        # -100 is the default "ignore_index" for PyTorch's CrossEntropyLoss
        labels = [-100] * len(prompt_ids) + response_ids

        # 5. Truncate or Pad
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor([1 if x != self.tokenizer.pad_token_id else 0 for x in input_ids])
        }

# ------------------------------------------------------------------------
# 3. PREPARE RESOURCES
# ------------------------------------------------------------------------
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.unk_token # Handle padding for Llama

print("Loading Dataset...")
# Using a subset for speed
dataset_raw = load_dataset("timdettmers/openassistant-guanaco", split="train").select(range(100)) 
train_dataset = SFTDataset(dataset_raw, tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Loading Model (FP16 for memory efficiency)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ------------------------------------------------------------------------
# 4. MANUAL TRAINING LOOP
# ------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=10, num_training_steps=len(train_loader) * EPOCHS
)

print("Starting Training Loop...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to GPU
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Forward pass
        # The model automatically calculates loss if 'labels' are provided
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

# ------------------------------------------------------------------------
# 5. SAVE
# ------------------------------------------------------------------------
print("Saving Model...")
model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print(f"Model and tokenizer saved to {NEW_MODEL_NAME}")