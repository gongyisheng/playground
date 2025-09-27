from datasets import Dataset

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Punctuation, Whitespace
from tokenizers.trainers import WordLevelTrainer

from config import Word2VecConfig


class Word2VecTokenizer:
    def __init__(self, config: Word2VecConfig):
        self.config = config
        self.tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
            ],
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                Punctuation(behavior="removed"),
                Whitespace(),
            ],
        )

    def train(self, dataset: Dataset) -> Tokenizer:
        trainer = WordLevelTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_freq,
            special_tokens=["[UNK]"],
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer)
        self.tokenizer.save(self.config.tokenizer_path)
        return self.tokenizer

    def load(self) -> Tokenizer:
        return Tokenizer.from_file(self.config.tokenizer_path)


if __name__ == "__main__":
    config = Word2VecConfig()
    tokenizer = Word2VecTokenizer(config).load()
    print(tokenizer.encode("Hello, World!").ids)
    print(tokenizer.decode([0, 1, 2, 3, 4, 5]))
