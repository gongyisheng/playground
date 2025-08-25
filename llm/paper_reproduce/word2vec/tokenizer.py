from datasets import Dataset

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Punctuation, Whitespace
from tokenizers.trainers import WordLevelTrainer

from config import Word2VecConfig

corpus = [
    "I love natural language processing.",
    "Word2Vec is simple but powerful.",
    "Tokenization is the first step.",
]


class Word2VecTokenizer:
    def __init__(self):
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

    def train(
        self,
        dataset: Dataset,
        vocab_size: int = Word2VecConfig.vocab_size,
        min_freq: int = Word2VecConfig.min_freq,
    ):

        trainer = WordLevelTrainer(
            vocab_size=vocab_size, min_frequency=min_freq, special_tokens=["[UNK]"]
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer)

    def save(self, path: str = Word2VecConfig.tokenizer_path):
        self.tokenizer.save(path)

    def load(self, path: str = Word2VecConfig.tokenizer_path):
        self.tokenizer = Tokenizer.from_file(path)


if __name__ == "__main__":
    word2vec_tokenizer = Word2VecTokenizer().load()
    print(word2vec_tokenizer.encode("Hello, World!").ids)
    print(word2vec_tokenizer.decode([0, 1, 2, 3, 4, 5]))
