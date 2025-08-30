import random
from typing import Literal

from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader

from config import Word2VecConfig


def get_dataset(config: Word2VecConfig) -> DatasetDict:
    dataset = load_dataset(path=config.dataset_path, name=config.dataset_name)
    return dataset


def collate_cbow_fn(
    batch: dict[str, list[str]],
    tokenizer: Tokenizer,
    config: Word2VecConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    n_words = config.n_words
    n_negative = config.n_negative
    vocab_size = config.vocab_size
    batch_input = []
    batch_target = []
    batch_label = []

    for line in batch:
        token_ids = tokenizer.encode(line["text"]).ids

        # skip too short sentences
        if len(token_ids) < n_words * 2 + 1:
            continue

        # get context token, predict target token
        for i in range(n_words, len(token_ids) - n_words):
            context = (
                token_ids[i - n_words : i]
                + token_ids[i + 1 : i + n_words + 1]
            )
            target = token_ids[i]
            batch_input.append(context)
            batch_target.append(target)
            batch_label.append(1.0)

            if n_negative > 0:
                for _ in range(n_negative):
                    batch_input.append(context)
                    random_target = random.randint(0, vocab_size-1)
                    while random_target == target:
                        random_target = random.randint(0, vocab_size-1)
                    batch_target.append(random_target)
                    batch_label.append(0.0)
    
    batch_input_tensor = torch.tensor(batch_input, dtype=torch.long)
    batch_target_tensor = torch.tensor(batch_target, dtype=torch.long)
    batch_labels_tensor = torch.tensor(batch_label, dtype=torch.float32)
    
    return batch_input_tensor, batch_target_tensor, batch_labels_tensor


def collate_skip_gram_fn(
    batch: dict[str, list[str]],
    tokenizer: Tokenizer,
    config: Word2VecConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    n_words = config.n_words
    n_negative = config.n_negative
    vocab_size = config.vocab_size
    batch_input = []
    batch_target = []
    batch_label = []

    for line in batch:
        token_ids = tokenizer.encode(line["text"]).ids

        # skip too short sentences
        if len(token_ids) < n_words * 2 + 1:
            continue

        for i in range(n_words, len(token_ids) - n_words):
            context_words = (
                token_ids[i - n_words : i]
                + token_ids[i + 1 : i + n_words + 1]
            )
            input_word = token_ids[i]
            for context in context_words:
                batch_input.append(input_word)
                batch_target.append(context)
                batch_label.append(1.0)
                
                if n_negative > 0:
                    for _ in range(n_negative):
                        batch_input.append(input_word)
                        random_target = random.randint(0, vocab_size-1)
                        while random_target == context:
                            random_target = random.randint(0, vocab_size-1)
                        batch_target.append(random_target)
                        batch_label.append(0.0)
    
    batch_input_tensor = torch.tensor(batch_input, dtype=torch.long)
    batch_target_tensor = torch.tensor(batch_target, dtype=torch.long)
    batch_labels_tensor = torch.tensor(batch_label, dtype=torch.float32)
    
    return batch_input_tensor, batch_target_tensor, batch_labels_tensor


def get_split_dataloader(
    dataset: DatasetDict,
    split: str,
    tokenizer: Tokenizer,
    config: Word2VecConfig,
    model_name: Literal["cbow", "skip_gram"] = "cbow",
) -> DataLoader:
    if model_name == "cbow":

        def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
            return collate_cbow_fn(batch, tokenizer, config)

    else:

        def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
            return collate_skip_gram_fn(batch, tokenizer, config)

    dataloader = DataLoader(
        dataset=dataset[split],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        collate_fn=collate_fn,  # collate_fn tells dataloader how to merge a list of samples into a batch of tensors.
        drop_last=True,
    )
    return dataloader


if __name__ == "__main__":
    config = Word2VecConfig()
    dataset = get_dataset(config)
    print(dataset["train"])
