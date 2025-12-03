import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Callable
import os
import wandb

from dataset import CustomSFTDataset


@dataclass
class CustomSFTConfig:
    """Configuration class for SFT training - similar to TRL's SFTConfig"""
    # Training configuration
    max_length: int = 1024
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    num_warmup_steps: int = 10
    gradient_accumulation_steps: int = 1

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    output_dir: str = "./sft_output"

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Wandb configuration
    use_wandb: bool = True
    wandb_project: Optional[str] = "sft-training"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None


class CustomSFTTrainer:

    def __init__(
        self,
        model: PreTrainedModel,
        processing_class: PreTrainedTokenizer,
        args: CustomSFTConfig,
        train_dataset: CustomSFTDataset,
        eval_dataset: CustomSFTDataset,
        # data_collator: Optional[Callable] = None,
        # formatting_func: Optional[Callable] = None,
        # model_init_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the SFT Trainer matching TRL's API

        Args:
            model: Model ID (string) or PreTrainedModel instance
            args: SFTConfig instance for training configuration
            train_dataset: Path to JSONL file (str) or HuggingFace dataset for training
            eval_dataset: Optional path to JSONL file (str) or evaluation dataset
            processing_class: Tokenizer; auto-loaded if None
            data_collator: Custom collator for batching
            formatting_func: Function to preprocess dataset examples
        """
        self.model = model
        self.tokenizer = processing_class
        self.args = args

        # Process dataset
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
            
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        # Setup optimizer and scheduler
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

    def compute_hard_label_loss(self, logits, labels, ignore_index=-100, reduction='mean'):
        """
        Standard cross-entropy loss for hard labels (integer indices).
        Equivalent to F.cross_entropy but with custom masking support.

        Args:
            logits: (B, L, V) unnormalized logits where B=batch, L=length, V=vocab_size
            labels: (B, L) integer token indices
            ignore_index: token index to ignore in loss calculation
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: scalar loss value
            loss_info: dict with logging information
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Shift logits and labels for causal LM (predict next token)
        shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = labels[:, 1:].contiguous()  # (B, L-1)

        # Flatten for loss calculation
        shift_logits = shift_logits.view(-1, vocab_size)  # (B*(L-1), V)
        shift_labels = shift_labels.view(-1)  # (B*(L-1),)

        # Create loss mask based on labels (mask out ignore_index)
        loss_mask = (shift_labels != ignore_index)  # (B*(L-1),)

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (B*(L-1), V)

        # Gather log probs for target tokens (standard cross-entropy)
        loss_per_token = -log_probs[torch.arange(shift_logits.size(0), device=logits.device), shift_labels]

        # Apply loss mask
        masked_loss = loss_per_token * loss_mask.float()

        # Compute final loss based on reduction
        num_valid_tokens = loss_mask.sum()
        if reduction == 'none':
            loss = masked_loss
        elif reduction == 'sum':
            loss = masked_loss.sum()
        elif reduction == 'mean':
            loss = masked_loss.sum() / (num_valid_tokens + 1e-8)  # Add epsilon to avoid division by zero
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")

        # Prepare logging info
        loss_info = {
            'num_valid_tokens': num_valid_tokens.item(),
            'mask_ratio': num_valid_tokens.item() / loss_mask.numel() if loss_mask.numel() > 0 else 0.0,
            'loss_type': 'hard',
        }

        return loss, loss_info

    def compute_soft_label_loss(self, logits, labels, reduction='mean'):
        """
        Cross-entropy loss for soft labels (probability distributions).
        Computes: -sum(soft_labels * log_softmax(logits))

        Args:
            logits: (B, L, V) unnormalized logits where B=batch, L=length, V_logits=vocab_size
            labels: (B, L, V) probability distributions
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: scalar loss value
            loss_info: dict with logging information
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Shift logits and labels for causal LM (predict next token)
        shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = labels[:, 1:, :].contiguous()  # (B, L-1, V)

        # Flatten for loss calculation
        shift_logits = shift_logits.view(-1, vocab_size)  # (B*(L-1), V)
        shift_labels = shift_labels.view(-1, vocab_size)  # (B*(L-1), V)

        # Create loss mask based on labels (if all probabilities sum to 0, treat as masked)
        # A position is valid if the label distribution sums to something non-zero
        label_sums = shift_labels.sum(dim=-1)  # (B*(L-1),)
        loss_mask = (label_sums > 1e-8)  # (B*(L-1),)

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (B*(L-1), V)

        # Compute cross-entropy with probability distributions
        # Loss = -sum(target_probs * log_probs) for each position
        loss_per_token = -(shift_labels * log_probs).sum(dim=-1)  # (B*(L-1),)

        # Apply loss mask
        masked_loss = loss_per_token * loss_mask.float()

        # Compute final loss based on reduction
        num_valid_tokens = loss_mask.sum()
        if reduction == 'none':
            loss = masked_loss
        elif reduction == 'sum':
            loss = masked_loss.sum()
        elif reduction == 'mean':
            loss = masked_loss.sum() / (num_valid_tokens + 1e-8)  # Add epsilon to avoid division by zero
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")

        # Prepare logging info
        loss_info = {
            'num_valid_tokens': num_valid_tokens.item(),
            'mask_ratio': num_valid_tokens.item() / loss_mask.numel() if loss_mask.numel() > 0 else 0.0,
            'loss_type': 'soft',
        }

        return loss, loss_info

    def compute_loss(self, logits, labels, ignore_index=-100, reduction='mean'):
        """
        Unified loss computation interface that dispatches to hard or soft label loss.
        Automatically detects whether labels are hard (integer indices) or soft (probability distributions) based on shape.

        Args:
            logits: (B, L, V) unnormalized logits where B=batch, L=length, V=vocab_size
            labels: (B, L) integer indices for hard labels OR (B, L, V) probability distributions for soft labels
            ignore_index: token index to ignore in loss calculation (only for hard labels)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            loss: scalar loss value
            loss_info: dict with logging information
        """
        # Determine if labels are soft (3D) or hard (2D) based on shape
        if labels.dim() == 3:
            # Soft labels: (B, L, V)
            return self.compute_soft_label_loss(
                logits, labels, reduction
            )
        elif labels.dim() == 2:
            # Hard labels: (B, L)
            return self.compute_hard_label_loss(
                logits, labels, ignore_index, reduction
            )
        else:
            raise ValueError(f"Invalid labels shape: {labels.shape}. Expected 2D (hard labels) or 3D (soft labels)")

    def train(self):
        """Run the training loop"""
        if self.train_dataloader is None:
            raise ValueError("No training dataset provided")

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
                }
            )

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
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                # Compute loss (automatically detects soft/hard labels based on shape)
                # Loss mask is calculated internally based on labels
                loss, loss_info = self.compute_loss(
                    logits,
                    labels=labels,
                )

                # Handle gradient accumulation
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
                if 'mask_ratio' in loss_info:
                    postfix_dict["mask%"] = f"{loss_info['mask_ratio']*100:.1f}"
                if 'loss_type' in loss_info:
                    postfix_dict["type"] = loss_info['loss_type'][:2]  # 'kl' or 'ce'

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
                            "train/num_valid_tokens": loss_info.get('num_valid_tokens', 0),
                            "train/mask_ratio": loss_info.get('mask_ratio', 0),
                        }
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

    # Load dataset
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    dataset = dataset.select(range(100))  # Using subset for speed

    # Create trainer with string model name (auto-loads model and tokenizer)
    trainer = CustomSFTTrainer(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        train_dataset=dataset,
        args=CustomSFTConfig(
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

    print("Training completed!")


if __name__ == "__main__":
    main()