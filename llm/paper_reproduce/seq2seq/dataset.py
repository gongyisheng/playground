from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from config import Seq2SeqConfig

def get_dataset(config: Seq2SeqConfig) -> DatasetDict:
    dataset = load_dataset(path=config.dataset_path, name=config.dataset_name)
    return dataset


def collate_fn(
    batch: dict[str, list[str]],
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    source_lang: str,
    target_lang: str,
    max_length: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_input = []
    batch_target = []
    input_pad_id = source_tokenizer.token_to_id("[PAD]")
    target_pad_id = target_tokenizer.token_to_id("[PAD]")

    for item in batch:
        src_tensor = source_tokenizer.encode(item[source_lang]).ids  # type: ignore[call-arg,index]
        tgt_tensor = target_tokenizer.encode(item[target_lang]).ids  # type: ignore[call-arg,index]
        batch_input.append(torch.tensor(src_tensor[:max_length], dtype=torch.long))
        batch_target.append(torch.tensor(tgt_tensor[:max_length], dtype=torch.long))
    batch_input_tensor = pad_sequence(batch_input, padding_value=input_pad_id, batch_first=True)
    batch_target_tensor = pad_sequence(batch_target, padding_value=target_pad_id, batch_first=True)
    return batch_input_tensor, batch_target_tensor


def get_split_dataloader(
    dataset: DatasetDict,
    split: str,
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    dataset_config: Seq2SeqConfig,
) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset[split], 
        batch_size=dataset_config.batch_size,
        num_workers=dataset_config.num_workers,
        shuffle=dataset_config.shuffle,
        collate_fn=lambda batch: collate_fn(
            batch,
            source_tokenizer,
            target_tokenizer,
            dataset_config.source_lang,
            dataset_config.target_lang,
            dataset_config.max_length,
        ),
    )
    return dataloader


if __name__ == "__main__":
    config = Seq2SeqConfig()
    dataset = get_dataset(config)