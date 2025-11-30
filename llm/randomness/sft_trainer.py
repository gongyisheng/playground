import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, Dataset as HFDataset
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Callable, Union
import os
import wandb


@dataclass
class SFTConfig:
    """Configuration class for SFT training - similar to TRL's SFTConfig"""
    # Training configuration
    max_length: int = 256
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    num_warmup_steps: int = 10
    gradient_accumulation_steps: int = 1

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    output_dir: str = "./sft_output"

    # Optimization
    use_fp16: bool = True
    use_gradient_checkpointing: bool = True

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # SFT-specific options (like TRL)
    packing: bool = False
    dataset_text_field: str = "text"
    max_seq_length: Optional[int] = None

    # Wandb configuration
    use_wandb: bool = True
    wandb_project: Optional[str] = "sft-training"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Loss calculation options
    use_manual_loss: bool = False
    loss_mask_fn: Optional[Callable] = None  # Custom function to create loss mask

    def __post_init__(self):
        if self.max_seq_length is None:
            self.max_seq_length = self.max_length

class SFTDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length=512,
        formatting_func: Optional[Callable] = None,
        text_field: str = "text"
    ):
        """
        Dataset class for SFT training

        Args:
            dataset: HuggingFace dataset
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            formatting_func: Optional function to format examples
            text_field: Field name containing text data
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formatting_func = formatting_func
        self.text_field = text_field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]

        # Apply formatting function if provided
        if self.formatting_func is not None:
            full_text = self.formatting_func(record)
        else:
            # Default: get text from specified field
            full_text = record[self.text_field]

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
    """SFT Trainer class matching TRL's SFTTrainer API"""

    def __init__(
        self,
        model: Union[str, PreTrainedModel] = None,
        args: Optional[SFTConfig] = None,
        train_dataset: Optional[HFDataset] = None,
        eval_dataset: Optional[HFDataset] = None,
        processing_class: Optional[PreTrainedTokenizer] = None,
        data_collator: Optional[Callable] = None,
        formatting_func: Optional[Callable] = None,
        model_init_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the SFT Trainer matching TRL's API

        Args:
            model: Model ID (string) or PreTrainedModel instance
            args: SFTConfig instance for training configuration
            train_dataset: Raw HuggingFace dataset for training
            eval_dataset: Optional evaluation dataset
            processing_class: Tokenizer; auto-loaded if None
            data_collator: Custom collator for batching
            formatting_func: Function to preprocess dataset examples
            model_init_kwargs: Additional kwargs for model loading
        """
        self.args = args if args is not None else SFTConfig()
        self.model_init_kwargs = model_init_kwargs or {}
        self.formatting_func = formatting_func

        # Auto-load model if string is provided
        if isinstance(model, str):
            print(f"Loading model from: {model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32,
                device_map="auto",
                **self.model_init_kwargs
            )
        else:
            self.model = model

        # Auto-load tokenizer if not provided
        if processing_class is None:
            if isinstance(model, str):
                print(f"Loading tokenizer from: {model}")
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            else:
                raise ValueError("Must provide processing_class when model is PreTrainedModel")
        else:
            self.tokenizer = processing_class

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        # Enable gradient checkpointing
        if self.args.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Process dataset
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Create data collator
        if data_collator is None:
            self.data_collator = self._default_data_collator
        else:
            self.data_collator = data_collator

        # Prepare dataset and dataloader
        if train_dataset is not None:
            processed_dataset = self._prepare_dataset(train_dataset)
            self.train_dataloader = DataLoader(
                processed_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=self.data_collator
            )
        else:
            self.train_dataloader = None

        # Setup optimizer and scheduler
        if self.train_dataloader is not None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate
            )
            total_steps = len(self.train_dataloader) * self.args.num_train_epochs
            total_steps = total_steps // self.args.gradient_accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=total_steps
            )

        self.global_step = 0

        # Initialize wandb
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_run_name,
                entity=self.args.wandb_entity,
                config={
                    "learning_rate": self.args.learning_rate,
                    "batch_size": self.args.batch_size,
                    "num_train_epochs": self.args.num_train_epochs,
                    "max_length": self.args.max_length,
                    "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                    "use_fp16": self.args.use_fp16,
                    "use_gradient_checkpointing": self.args.use_gradient_checkpointing,
                }
            )

    def _prepare_dataset(self, dataset: HFDataset) -> Dataset:
        """Convert HuggingFace dataset to PyTorch dataset with tokenization"""
        return SFTDataset(
            dataset,
            self.tokenizer,
            max_length=self.args.max_seq_length,
            formatting_func=self.formatting_func,
            text_field=self.args.dataset_text_field
        )

    def _default_data_collator(self, batch):
        """Default collator for batching examples"""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def compute_loss(self, logits, labels, attention_mask=None):
        """
        Manually compute cross-entropy loss with custom masking.

        Args:
            logits: Model output logits, shape (batch_size, seq_len, vocab_size)
            labels: Target labels, shape (batch_size, seq_len)
            attention_mask: Optional attention mask (currently unused)

        Returns:
            loss: Scalar loss value
            loss_info: Dictionary with additional loss statistics
        """
        _ = attention_mask  # Reserved for future use
        batch_size, seq_len, vocab_size = logits.shape

        # Shift logits and labels for next-token prediction
        # logits: predict token at position i from tokens 0..i-1
        # labels: token at position i
        shift_logits = logits[:, :-1, :].contiguous()  # (batch, seq_len-1, vocab)
        shift_labels = labels[:, 1:].contiguous()      # (batch, seq_len-1)

        # Flatten for cross-entropy calculation
        shift_logits = shift_logits.view(-1, vocab_size)  # (batch * seq_len-1, vocab)
        shift_labels = shift_labels.view(-1)               # (batch * seq_len-1)

        # Create loss mask
        # By default, mask out -100 labels (standard behavior)
        loss_mask = (shift_labels != -100).float()

        # Apply custom loss mask function if provided
        if self.args.loss_mask_fn is not None:
            # Reshape back to 2D for custom masking
            loss_mask_2d = loss_mask.view(batch_size, seq_len - 1)
            shift_labels_2d = shift_labels.view(batch_size, seq_len - 1)

            # Apply custom mask function
            custom_mask = self.args.loss_mask_fn(
                shift_labels_2d,
                loss_mask_2d,
                self.tokenizer
            )
            loss_mask = custom_mask.view(-1).float()

        # Compute cross-entropy loss (without reduction first)
        loss_per_token = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction='none'
        )

        # Apply mask and compute mean over valid tokens only
        masked_loss = loss_per_token * loss_mask
        num_valid_tokens = loss_mask.sum()

        if num_valid_tokens > 0:
            loss = masked_loss.sum() / num_valid_tokens
        else:
            loss = masked_loss.sum()  # Will be 0 if no valid tokens

        # Gather statistics
        loss_info = {
            'loss': loss.item(),
            'num_valid_tokens': num_valid_tokens.item(),
            'total_tokens': loss_mask.numel(),
            'mask_ratio': (num_valid_tokens / loss_mask.numel()).item() if loss_mask.numel() > 0 else 0.0,
            'mean_loss_per_token': loss.item(),
        }

        return loss, loss_info

    def train(self):
        """Run the training loop"""
        if self.train_dataloader is None:
            raise ValueError("No training dataset provided")

        print("Starting Training Loop...")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        self.model.train()

        for epoch in range(self.args.num_train_epochs):
            total_loss = 0
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.args.num_train_epochs}"
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.args.device)
                attention_mask = batch["attention_mask"].to(self.args.device)
                labels = batch["labels"].to(self.args.device)

                # Forward pass
                if self.args.use_manual_loss:
                    # Manual loss calculation
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits
                    loss, loss_info = self.compute_loss(logits, labels, attention_mask)
                else:
                    # Use built-in loss calculation
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss_info = {'loss': loss.item()}

                # Handle gradient accumulation
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Optimization step
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                total_loss += loss.item()

                # Prepare progress bar postfix
                postfix_dict = {
                    "loss": loss.item() * self.args.gradient_accumulation_steps,
                    "lr": self.scheduler.get_last_lr()[0]
                }
                if self.args.use_manual_loss and 'mask_ratio' in loss_info:
                    postfix_dict["mask%"] = f"{loss_info['mask_ratio']*100:.1f}"

                progress_bar.set_postfix(postfix_dict)

                # Logging
                if self.global_step % self.args.logging_steps == 0 and (step + 1) % self.args.gradient_accumulation_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(f"\nStep {self.global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}")

                    # Log to wandb
                    if self.args.use_wandb:
                        log_dict = {
                            "train/loss": loss.item() * self.args.gradient_accumulation_steps,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": self.global_step,
                        }
                        # Add additional loss info if using manual loss
                        if self.args.use_manual_loss:
                            log_dict.update({
                                "train/num_valid_tokens": loss_info.get('num_valid_tokens', 0),
                                "train/mask_ratio": loss_info.get('mask_ratio', 0),
                            })
                        wandb.log(log_dict)

                # Save checkpoint
                if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0 and (step + 1) % self.args.gradient_accumulation_steps == 0:
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                    self.save_model(checkpoint_dir)

            avg_loss = total_loss / len(self.train_dataloader)
            print(f"\nEpoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

            # Log epoch metrics to wandb
            if self.args.use_wandb:
                wandb.log({
                    "train/epoch_loss": avg_loss,
                    "train/epoch": epoch + 1,
                })

        print("Training completed!")

        # Finish wandb run
        if self.args.use_wandb:
            wandb.finish()

    def save_model(self, output_dir: str = None):
        """Save the trained model and tokenizer"""
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload model",
        private: bool = False,
        token: Optional[str] = None
    ):
        """
        Push the trained model to Hugging Face Hub

        Args:
            repo_id: Repository ID on Hugging Face Hub (e.g., "username/model-name")
            commit_message: Commit message for the push
            private: Whether to create a private repository
            token: Hugging Face API token (uses cached token if None)
        """
        print(f"Pushing model to Hugging Face Hub: {repo_id}")
        self.model.push_to_hub(
            repo_id,
            commit_message=commit_message,
            private=private,
            token=token
        )
        self.tokenizer.push_to_hub(
            repo_id,
            commit_message=commit_message,
            private=private,
            token=token
        )
        print(f"Model successfully pushed to: https://huggingface.co/{repo_id}")


def main():
    """Main training function demonstrating TRL-style SFTTrainer API"""

    # Example 1: Simple usage with minimal configuration (TRL-style)
    print("="*50)
    print("Example 1: TRL-style Simple API")
    print("="*50)

    # Load dataset
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    dataset = dataset.select(range(100))  # Using subset for speed

    # Create trainer with string model name (auto-loads model and tokenizer)
    trainer = SFTTrainer(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        train_dataset=dataset,
        args=SFTConfig(
            max_length=256,
            batch_size=4,
            num_train_epochs=1,
            output_dir="./sft_output",
        )
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

    # Optionally push to hub (uncomment to use)
    # trainer.push_to_hub("username/my-sft-model")

    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)


if __name__ == "__main__":
    main()