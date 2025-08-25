from dataclasses import dataclass
from typing import Literal

import torch

BASE_DIR = "/media/hdddisk/yisheng/replicate/word2vec"

@dataclass
class Word2VecConfig:
    # model selection
    model_name: Literal["cbow", "skip_gram"] = "cbow"

    # dataset configs
    dataset_path: str = "Salesforce/wikitext"
    dataset_name: str = "wikitext-103-raw-v1"

    # model configs
    vocab_size: int = 20000
    min_freq: int = 50
    embedding_dim: int = 256

    n_words: int = 4

    model_path: str = f"{BASE_DIR}/{model_name}_dim={embedding_dim}_nwords={n_words}"
    tokenizer_path: str = f"{BASE_DIR}/tokenizer"

    # training configs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    batch_size: int = 256
    num_workers: int = 8
    shuffle: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # wandb configs
    wandb_name: str | None = f"word2vec_{model_name}_dim={embedding_dim}_nwords={n_words}"
    wandb_project: str = "paper_reproduce"
