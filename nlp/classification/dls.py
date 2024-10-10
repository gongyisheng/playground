# Prepare dataloaders for HF models
# Here we're using torch DataLoader but HF datasets Dataset; don't confuse those

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from pathlib import Path
from config import CFG
from utils import get_num_col_names_iter


def create_dataloaders(data_df:pd.DataFrame, val_fold:int, CFG: CFG):
    """Create train and validation dataloaders from pandas df, using specific functions below"""
    # Create Datasets
    col_list = get_col_list(CFG)

    ds_train_valid = DatasetDict(
        {
            "train": Dataset.from_pandas(
                data_df.loc[data_df.fold != val_fold, col_list]
                .reset_index(drop=True)
                .rename({CFG.item_col_name: "text"}, axis="columns")
            ),
            "valid": Dataset.from_pandas(
                data_df.loc[data_df.fold == val_fold, col_list]
                .reset_index(drop=True)
                .rename({CFG.item_col_name: "text"}, axis="columns")
            ),
        }
    )
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.checkpoint)

    # If CFG.max_len == "auto", set max_len = length of longest tokenized input
    max_len = get_max_len(data_df, tokenizer, CFG)
    print(f"tokenizer max len: {max_len}")
    # Tokenize Datasets
    ds_tok_train_valid = ds_train_valid.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"max_len": max_len, "tokenizer": tokenizer},
    )
    ds_tok_train_valid = ds_tok_train_valid.remove_columns(["text"])

    # Convert elements to torch tensors.
    ds_tok_train_valid.set_format("torch")

    # Create Dataloaders from Datasets
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        ds_tok_train_valid["train"],
        shuffle=True,
        batch_size=CFG.bs,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        ds_tok_train_valid["valid"],
        shuffle=False,
        batch_size=CFG.bs,
        collate_fn=data_collator,
    )

    return train_dataloader, eval_dataloader, max_len, tokenizer


def get_col_list(CFG: CFG):
    """Returns list of columns to be included in dataloaders, optionally including numerical columns"""
    col_list = [CFG.item_col_name, "labels"]
    if CFG.num_col_names:
        num_col_list = get_num_col_names_iter(
            CFG.num_col_names
        )  # necessary in case num_col_names is a single str
        col_list.extend(num_col_list)
    return col_list


def tokenize_function(example, max_len, tokenizer):
    """HF inspired function to apply tokenizer"""
    return tokenizer(example["text"], truncation=True, max_length=max_len)


def get_max_len(data_df: pd.DataFrame, tokenizer, CFG: CFG) -> int:
    """Calculates length of longest tokenized input for use if CFG.max_len == 'auto'"""
    max_len = (
        data_df[CFG.item_col_name].apply(lambda x: len(tokenizer.tokenize(x))).max()
        if CFG.max_len == "auto"
        else CFG.max_len
    )
    return max_len
