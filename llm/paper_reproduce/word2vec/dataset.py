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
) -> tuple[torch.Tensor, torch.Tensor]:

    n_words = config.n_words
    batch_input = []
    batch_target = []

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

    # transform to tensor
    batch_input_tensor = torch.tensor(
        batch_input, dtype=torch.long
    )  # [batch_size, n_words * 2]
    batch_target_tensor = torch.tensor(batch_target, dtype=torch.long)  # [batch_size,]

    return batch_input_tensor, batch_target_tensor


def collate_skip_gram_fn(
    batch: dict[str, list[str]],
    tokenizer: Tokenizer,
    config: Word2VecConfig,
) -> tuple[torch.Tensor, torch.Tensor]:

    n_words = config.n_words
    batch_input = []
    batch_target = []

    for line in batch:
        token_ids = tokenizer.encode(line["text"]).ids

        # skip too short sentences
        if len(token_ids) < n_words * 2 + 1:
            continue

        # get input word, predict target context words
        for i in range(n_words, len(token_ids) - n_words):
            context_words = (
                token_ids[i - n_words : i]
                + token_ids[i + 1 : i + n_words + 1]
            )
            input_word = token_ids[i]
            for context in context_words:
                batch_input.append(input_word)
                batch_target.append(context)

    # transform to tensor
    batch_input_tensor = torch.tensor(batch_input, dtype=torch.long)  # [batch_size,]
    batch_target_tensor = torch.tensor(
        batch_target, dtype=torch.long
    )  # [batch_size, n_words * 2]

    return batch_input_tensor, batch_target_tensor


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
