import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class SFTConfig:
    """Configuration class for SFT training"""
    # Model configuration
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    new_model_name: str = "manual_sft_model"

    # Training configuration
    max_length: int = 256
    batch_size: int = 4
    learning_rate: float = 2e-5
    epochs: int = 1
    num_warmup_steps: int = 10

    # Dataset configuration
    dataset_name: str = "timdettmers/openassistant-guanaco"
    dataset_split: str = "train"
    dataset_size: int = 100

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
    use_gradient_checkpointing: bool = True

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


class SFTTrainer:
    """SFT Trainer class similar to TRL's SFTTrainer pattern"""

    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        config: SFTConfig = None
    ):
        """
        Initialize the SFT Trainer

        Args:
            model: Pre-initialized model for training
            tokenizer: Pre-initialized tokenizer
            train_dataloader: Pre-initialized data loader
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.config = config if config is not None else SFTConfig()

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=len(self.train_dataloader) * self.config.epochs
        )

    def train(self):
        """Run the training loop"""
        print("Starting Training Loop...")
        self.model.train()

        for epoch in range(self.config.epochs):
            total_loss = 0
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.epochs}"
            )

            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Optimization step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    def save_model(self, output_dir: str = None):
        """Save the trained model and tokenizer"""
        if output_dir is None:
            output_dir = self.config.new_model_name

        print(f"Saving Model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")


def main(config: SFTConfig = None):
    """Main training function for SFT"""
    # Initialize configuration
    if config is None:
        config = SFTConfig()

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.unk_token # Handle padding for Llama

    print("Loading Dataset...")
    # Using a subset for speed
    dataset_raw = load_dataset(config.dataset_name, split=config.dataset_split).select(range(config.dataset_size))
    train_dataset = SFTDataset(dataset_raw, tokenizer, config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    print("Loading Model (FP16 for memory efficiency)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.use_fp16 else torch.float32,
        device_map="auto"
    )

    # Enable gradient checkpointing to save memory
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize trainer with pre-initialized components
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_loader,
        config=config
    )

    # Run training
    trainer.train()

    # Save model
    trainer.save_model()


if __name__ == "__main__":
    main()