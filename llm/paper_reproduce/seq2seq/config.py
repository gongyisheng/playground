from dataclasses import dataclass, field, asdict
from typing import Any

BASE_DIR = "/media/hdddisk/yisheng/replicate/seq2seq"

@dataclass
class Seq2SeqConfig:

    # dataset configs
    dataset_path: str = "wmt/wmt14"
    dataset_name: str | None = "fr-en"
    source_lang: str = "fr"
    target_lang: str = "en"
    max_length: int = 1000
    batch_size: int = 256
    num_workers: int = 4
    shuffle: bool = True
    # tokenizer configs
    min_frequency: int = 1
    num_proc: int = 8
    special_tokens: list[str] = field(
        default_factory=lambda: ["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
    )
    # model configs
    source_vocab_size: int = 8000
    target_vocab_size: int = 6000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout_ratio: float = 0.5
    teacher_forcing_ratio: float = 0.5
    # optimizer configs
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    # training configs
    epochs: int = 10
    clip_norm: float | None = None  # Gradient clipping norm, None means no clipping
    # inference configs
    inference_max_length: int = 50
    # evaluation configs
    evaluation_max_samples: int | None = None
    evaluation_batch_size: int = 32
    # wandb configs
    wandb_name: str | None = None
    wandb_project: str = "Seq2Seq"
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.wandb_name is None:
            self.wandb_name = self._get_wandb_name()

    def get_lang_vocab_size(self, lang: str) -> int:
        """Get vocabulary size for a specific language."""
        if lang == self.source_lang:
            return self.source_vocab_size
        if lang == self.target_lang:
            return self.target_vocab_size
        msg = f"Language '{lang}' not supported. Use '{self.source_lang}' or '{self.target_lang}'"
        raise ValueError(msg)

    def _get_wandb_name(self) -> str:
        return f"E:{self.embedding_dim},H:{self.hidden_dim},L:{self.num_layers}"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return asdict(self)