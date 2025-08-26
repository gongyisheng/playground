from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from config import Word2VecConfig
from dataset import get_dataset, get_split_dataloader
from model import CbowModel, SkipGramModel
from tokenizer import Word2VecTokenizer

class Word2VecTrainer:
    def __init__(self, config: Word2VecConfig):
        self.model_name = config.model_name
        self.model_path = config.model_path
        if self.model_name == "cbow":
            self.model = CbowModel(config)
        elif self.model_name == "skip_gram":
            self.model = SkipGramModel(config)

        self.config = config
        self.device = config.device
        print(f"Using device: {config.device}")
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        best_val_loss = float("inf")
        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch(train_dataloader)
            val_loss, test_loss = self._validate_epoch(val_dataloader, test_dataloader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model, self.model_path)
                print(f"Saved best model to {self.model_path}")

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}",
            )

            # log metrics to wandb
            wandb.log(
                {
                    "Loss_train": train_loss,
                    "loss_val": val_loss,
                    "loss_test": test_loss,
                    "perplexity_train": torch.exp(torch.tensor(train_loss)),
                    "perplexity_val": torch.exp(torch.tensor(val_loss)),
                    "perplexity_test": torch.exp(torch.tensor(test_loss)),
                },
            )
    
    def _train_epoch(self, train_dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for input_batch, target_batch in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            input_batch_device, target_batch_device = (
                input_batch.to(self.device),
                target_batch.to(self.device),
            )
            loss = self.calc_loss_batch(input_batch_device, target_batch_device)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * input_batch.size(0)  # Multiply by batch size
            total_samples += input_batch.size(0)
        train_loss = total_loss / total_samples
        return train_loss

    def _validate_epoch(self, val_dataloader: DataLoader, test_dataloader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            val_loss = self.calc_loss_loader(val_dataloader)
            test_loss = self.calc_loss_loader(test_dataloader)
        return val_loss, test_loss
    
    def calc_loss_batch(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss

    def calc_loss_loader(self, data_loader: DataLoader) -> float:
        total_loss = 0.0
        total_samples = 0  # Track total samples
        for input_batch, target_batch in tqdm(data_loader):
            input_batch_device, target_batch_device = (
                input_batch.to(self.device),
                target_batch.to(self.device),
            )
            loss = self.calc_loss_batch(input_batch_device, target_batch_device)
            total_loss += loss.item() * input_batch.size(0)  # Multiply by batch size
            total_samples += input_batch.size(0)
        return total_loss / total_samples  # Correct average


def train_tokenizer(config: Word2VecConfig) -> None:
    if not Path(config.tokenizer_path).exists():
        tokenizer = Word2VecTokenizer(config)
        dataset = get_dataset(config)
        tokenizer.train(dataset["train"])
    else:
        print(f"Tokenizer already exists at {config.tokenizer_path}")


def train_model(config: Word2VecConfig) -> None:
    model_name = config.model_name
    wandb.init(
        # set the wandb project where this run will be logged
        project=config.wandb_project,
        name=config.wandb_name,
        # track hyperparameters and run metadata
        config=asdict(config),
    )
    dataset = get_dataset(config)
    tokenizer = Word2VecTokenizer(config).load()
    train_dataloader = get_split_dataloader(
        dataset,
        "train",
        tokenizer=tokenizer,
        config=config,
        model_name=model_name,
    )
    val_dataloader = get_split_dataloader(
        dataset,
        "validation",
        tokenizer=tokenizer,
        config=config,
        model_name=model_name,
    )
    test_dataloader = get_split_dataloader(
        dataset,
        "test",
        tokenizer=tokenizer,
        config=config,
        model_name=model_name,
    )

    trainer = Word2VecTrainer(config)
    trainer.train(train_dataloader, val_dataloader, test_dataloader)
    wandb.finish()

def train_cbow():
    for dim in [128, 256, 512, 1024]:
        print(f"Training dim {dim}")
        config = Word2VecConfig(
            model_name="cbow",
            dataset_path="Salesforce/wikitext",
            dataset_name="wikitext-103-raw-v1",
            embedding_dim=dim,
            learning_rate=1e-3,
            batch_size=256*256//dim,
            num_workers=8,
            epochs=5,
        )
        print(config.embedding_dim)
        train_tokenizer(config)
        train_model(config)


def train_skip_gram():
    config = Word2VecConfig(
        model_name="skip_gram",
        dataset_path="Salesforce/wikitext",
        dataset_name="wikitext-103-raw-v1",
        embedding_dim=100,
        learning_rate=0.01,
        batch_size=64,
        num_workers=8,
        epochs=5,
    )

    train_tokenizer(config)
    train_model(config)


if __name__ == "__main__":
    train_cbow()
    # train_skip_gram()