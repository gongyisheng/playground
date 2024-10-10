# Run model inference on new data
# This is the convenience module that we are giving to analysts to run in their inference notebooks

import torch
import json
from pathlib import Path
import pandas as pd
import re
import numpy as np
import datetime
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers.utils import logging
from utils import (
    tokenize_function,
    set_device,
    get_num_col_names_iter,
)
from storage import (
    load_model,
    load_le,
    load_scaler,
    get_model_checkpoint,
)


# Disable warnings from HF
logging.set_verbosity(40)


def run_inference(
    item_desc_df,
    model_name,
    proj_directory,
    item_col_name="item",
    num_col_names=None,
    scale_num_cols=False,
    max_len=128,
    bs=64,
    proba=False,
    k=3,
    device="auto",
    custom_model_type=None,
    progress_bars=True
):
    """
    Runs inference for a selected pretrained model

    Parameters
    ----------
    item_desc_df: pyspark.sql.dataframe.DataFrame or pd.DataFrame
        df containing a string field for item descriptions you wish to classify and optional numeric fields
    model_name: str
        name of pretrained model used for inference
    proj_directory: str
        directory containing saved pretrained model
    item_col_name: str (default: "item")
        name of sole column in item_desc_df
    num_col_names: list[str] or str
        name(s) of column(s) in item_desc_df containing numerical features. Can input single column as string.
    scale_num_cols: bool
        whether to scale numerical inputs using saved fitted scaler.
    max_len: int
        max_len of tokenized input data. Tokenized inputs over max_len will be truncated
    bs: int
        batch size
    proba: bool or str
        proba == False returns the predicted category
        proba == True returns multiple columns with label probabilities for all labels
        proba == "proba_max" returns multiple columns with the probability of the predicted label
        proba == "topk" returns k*2 columns with the top k predicted labels, along with their probabilities
    k: int
        the number of most probable labels to be returned. Used only when proba == "topk".
    proba_max: bool
        whether to return another column with probability of predicted label. Only works if proba == False.
    device: str
        device for inference. Supported devices: "cuda", "mps", "cpu", and "auto"
    progress_bars: bool
        whether to display tqdm progress bars & huggingface datasets progress bars

    Returns
    -------
    Returns a pd.DataFrame combining item_desc_df and predicted labels, or probabilities of labels if proba = True
    """

    # disable HF datasets 'map' progress bars
    if not progress_bars:
        datasets.logging.disable_progress_bar()

    # Convert spark df to pandas df
    if not isinstance(item_desc_df, pd.DataFrame):
        item_desc_df = item_desc_df.toPandas()
    item_desc_df = item_desc_df.reset_index(drop=True)

    # Device management
    device = set_device(device)
    print(f"Using device {device}")

    # Load model and tokenizer
    print("Loading model...")
    model, model_path = load_model(
        model_name, proj_directory, custom=bool(num_col_names), device=device, custom_model_type=custom_model_type
    )
    model.to(device)
    print(f"Loaded model from {model_path}")
    print("Loading tokenizer...")
    tokenizer, checkpoint = load_tokenizer(proj_directory, model_name, custom_model_type)
    print(f"Loaded tokenizer for checkpoint {checkpoint}")

    # Numerical scaling
    if scale_num_cols:
        print("Loading scaler...")
        scaler, scaler_path = load_scaler(model_name, proj_directory)
        print(f"Scaler loaded from {scaler_path}")
        item_desc_df = scale_inf_data(scaler, item_desc_df, num_col_names)

    # Create Dataloader
    print("Creating Dataloader...")
    inference_dl = create_inference_dl(
        item_desc_df, item_col_name, num_col_names, max_len, bs, tokenizer
    )
    print("Dataloader created.")

    # Run inference
    print("Running inference...")
    start_dt = datetime.datetime.now()
    results = get_preds_fork(inference_dl, model, device, proba, k, progress_bars)
    print(f"Inference complete. Time elapsed: {datetime.datetime.now() - start_dt}")

    # Get label encoder cat map
    print("Loading label encoder...")
    base_model_name = re.sub("__\d$", "", model_name)
    le, le_path = load_le(base_model_name, proj_directory)
    print(f"Loaded label encoder from {le_path}")

    # Get inference df
    print("Creating inference df...")
    inference_df = get_inference_df_fork(item_desc_df, results, le, proba, k)
    print("Inference df created.")

    return inference_df


def load_tokenizer(proj_directory, model_name, custom_model_type):
    """Load HF tokenizer using model name with AutoTokenizer"""
    if custom_model_type == "metric_classification":
        config_path = Path(proj_directory) / "models"/ model_name /"config.json"
        with config_path.open("rb") as f:
            config = json.load(f)
            checkpoint = config["checkpoint"]
    else:
        checkpoint = get_model_checkpoint(model_name, proj_directory)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    return tokenizer, checkpoint


def scale_inf_data(scaler, df: pd.DataFrame, num_col_names):
    """Scale num data with scaler fit to training data"""
    num_col_list = get_num_col_names_iter(num_col_names)
    if len(num_col_names) == 1:
        df[num_col_list] = scaler.transform(df[num_col_list].values.reshape(-1, 1))
    else:
        df[num_col_list] = scaler.transform(df[num_col_list].values)
    return df


def create_inference_dl(
    item_desc_df, item_col_name, num_col_names, max_len, bs, tokenizer
):
    """Inference dataloaders"""
    col_list = get_inf_col_list(item_col_name, num_col_names)
    ds_inference = Dataset.from_pandas(
        item_desc_df.loc[:, col_list].rename({item_col_name: "text"}, axis="columns")
    )
    ds_tok_inference = ds_inference.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"max_len": max_len, "tokenizer": tokenizer},
    )
    ds_tok_inference = ds_tok_inference.remove_columns(["text"])

    ds_tok_inference.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    inference_dl = DataLoader(ds_tok_inference, batch_size=bs, collate_fn=data_collator)

    return inference_dl


def get_inf_col_list(item_col_name, num_col_names):
    """Get list of columns in df in dataloaders"""
    col_list = [item_col_name]
    if num_col_names:
        num_col_list = get_num_col_names_iter(num_col_names)
        col_list += num_col_list
    return col_list


def get_preds_fork(inference_dataloader, model, device, proba, k, progress_bars):
    """Get inference results"""
    if proba == "proba_max":
        preds_probas = get_preds_proba_max(inference_dataloader, model, device, progress_bars)
        return preds_probas
    elif proba == "topk":
        topk_probas = get_preds_proba_topk(inference_dataloader, model, device, k, progress_bars)
        return topk_probas
    elif proba:
        probas = get_preds_proba(inference_dataloader, model, device, progress_bars)
        return probas
    else:
        preds = get_preds(inference_dataloader, model, device, progress_bars)
        return preds


def get_preds(inference_dataloader, model, device, progress_bars):
    """Get batched inference predictions as list"""
    preds = []
    model.eval()
    for batch in tqdm(inference_dataloader, disable=(not progress_bars)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        preds += list(predictions.cpu().numpy())
    return preds

def get_preds_proba_max(inference_dataloader, model, device, progress_bars):
    """Get batched inference predictions and prediction probabilities as list of lists"""
    preds = []
    probas_max = []
    model.eval()
    for batch in tqdm(inference_dataloader, disable=(not progress_bars)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logit()
        predictions = torch.argmax(logits, dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        prediction_probabilities = torch.max(probabilities, dim=-1).values

        preds += list(predictions.cpu().numpy())
        probas_max += list(prediction_probabilities.cpu().numpy())
    return [preds, probas_max]

def get_preds_proba_topk(inference_dataloader, model, device, k, progress_bars):
    model.eval()
    for i, batch in enumerate(tqdm(inference_dataloader, disable=(not progress_bars))):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        probas = F.softmax(logits, dim=-1)
        if i == 0:
            probas_comb = probas.clone()
        else:
            probas_comb = torch.vstack([probas_comb, probas])

    topk_probas = torch.topk(probas_comb, k, dim=-1)
    return topk_probas

def get_preds_proba(inference_dataloader, model, device, progress_bars):
    """Get batched inference probabilities per class as array"""
    probas = []
    model.eval()
    for batch in tqdm(inference_dataloader, disable=(not progress_bars)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        probas += list(probabilities.cpu().numpy())

    return np.array(probas)


def get_inference_df_fork(item_desc_df, results, le, proba, k):
    """Get inference dataframe"""
    if proba == "proba_max":
        inference_df = get_inference_df_proba_max(item_desc_df, results, le)
    elif proba == "topk":
        inference_df = get_inference_df_proba_topk(item_desc_df, results, le, k)
    elif proba:
        inference_df = get_inference_df_proba(item_desc_df, results, le)
    else:
        inference_df = get_inference_df(item_desc_df, results, le)
    return inference_df


def get_inference_df(item_desc_df, preds, le):
    """Add decoded category labels to inference dataframe"""
    inference_df = item_desc_df.copy()
    inference_df["category"] = preds
    inference_df["category"] = le.inverse_transform(inference_df["category"])

    return inference_df

def get_inference_df_proba_max(item_desc_df, preds_probas, le):
    """Add decoded category labels and corresponding probabilities to inference dataframe"""
    inference_df = item_desc_df.copy()
    preds, probas = preds_probas
    inference_df["category"] = preds
    inference_df["category"] = le.inverse_transform(inference_df["category"])
    inference_df["proba"] = probas

    return inference_df

def get_inference_df_proba_topk(item_desc_df, topk_probas, le, k):
    inference_df = item_desc_df.copy()
    probas_df = pd.DataFrame()
    topk_probas, topk_preds = topk_probas.values.cpu().numpy(), topk_probas.indices.cpu().numpy()
    for i in range(k):
        probas_df[f"pred_{i+1}"] = topk_preds[:,i]
        probas_df[f"pred_{i+1}"] = le.inverse_transform(probas_df[f"pred_{i+1}"])
        probas_df[f"proba_{i+1}"] = topk_probas[:,i]
    inference_df = pd.concat([inference_df, probas_df], axis=1)
    return inference_df

def get_inference_df_proba(item_desc_df, probas, le):
    """Add decoded category labels and probabilities to inference dataframe"""
    le_classes = list(le.classes_)
    inference_df = pd.DataFrame(data=probas, columns=le_classes)
    inference_df = pd.concat([item_desc_df.copy(), inference_df], axis=1)
    return inference_df