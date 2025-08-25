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

    model_path: str = ""
    tokenizer_path: str = ""
