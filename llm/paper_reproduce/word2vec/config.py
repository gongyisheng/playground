from typing import Literal, Any


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

    cbow_n_words: int = 4
    skip_gram_n_words: int = 4

    model_path: str = ""
    tokenizer_path: str = ""

    # training configs
    device: str = "auto"
    epochs: int = 50
    batch_size: int = 256
    num_workers: int = 8
    shuffle: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # wandb configs
    wandb_name: str | None = "paper_reproduce"
    wandb_project: str = "word2vec"
