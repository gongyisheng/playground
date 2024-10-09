import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os
import torch
import random
from config import Config


def setup(config: Config = Config):
    # Time keeping
    run_start_dt = datetime.datetime.utcnow()
    # model ID
    base_model_id = get_base_model_id(config, run_start_dt)
    # Set random seed
    set_seed(config)
    # Print run details
    if config.mlflow:
        print(
            "------------------------------------\n"
            "MLFlow Experiment\n"
            f"experiment name: {Path(config.experiment_path).name}\n"
            f"experiment id: {config._experiment_id}\n"
            f"base_model_id/run_name: {base_model_id}\n"
            f"n_epochs: {config.n_epochs}\n"
            f"n_val_folds: {config.val_folds}\n"
            f"dt: {str(run_start_dt)}\n"
            "------------------------------------"
        )
    else:
        print(
            "------------------------------------\n"
            f"base_model_id: {base_model_id}\n"
            f"n_epochs: {config.n_epochs}\n"
            f"n_val_folds: {config.val_folds}\n"
            f"dt: {str(run_start_dt)}\n"
            "------------------------------------"
        )

    return base_model_id, run_start_dt


def get_base_model_id(config: Config, run_start_dt: datetime.datetime):
    """Returns the ID of the experiment run, concatenating checkpoint name and datetime of the run"""
    name_parts = [
        config.checkpoint.replace("/", "_"),
        run_start_dt.strftime("%Y_%m_%d_%H_%M"),
    ]
    if config.model_name:
        name_parts.insert(0, config.model_name)
    return "__".join(name_parts)


def set_seed(config: Config):
    """Set random seeds for reproducibility"""
    if config.random_seed != None:
        random.seed(config.random_seed)  # Python
        np.random.seed(config.random_seed)  # cpu vars
        torch.manual_seed(config.random_seed)  # cpu vars
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
            torch.cuda.manual_seed_all(config.random_seed)  # gpu vars
        if torch.backends.cudnn.is_available:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_device(device):
    """Device management: running on Mac M1 GPU with mps"""
    if device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif device == "cpu":
        device = torch.device("cpu")
    elif device == "cuda":
        device = torch.device("cuda")
    elif device == "mps":
        device = torch.device("mps")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return device


def get_val_fold_iter(val_folds):
    """Returns list of validation folds; specific validation folds can be specified as a list, otherwise the int value is number of folds"""
    if isinstance(val_folds, list):
        return val_folds
    elif isinstance(val_folds, int):
        return list(np.arange(val_folds))


def get_num_col_names_iter(num_col_names):
    if isinstance(num_col_names, str):
        return [num_col_names]
    elif isinstance(num_col_names, list):
        return num_col_names


def tokenize_function(example, max_len, tokenizer):
    """Simple function to tokenize our datasets"""
    return tokenizer(example["text"], truncation=True, max_length=max_len)


def create_train_log_df(
    train_results: list,
    base_model_id: str,
    model_id: str,
    val_fold: int,
    auto_max_len: int,
    config: Config,
):
    """Initiate the training log file"""
    df = pd.DataFrame(train_results)
    df["model_id"] = model_id
    df["base_model_id"] = base_model_id
    df["fold"] = val_fold
    df["max_len"] = config.max_len
    df["auto_max_len"] = auto_max_len if config.max_len == "auto" else None
    df["subset"] = config.subset
    df["pool_mode"] = config.pool_mode
    df["bs"] = config.bs
    df["build_custom_model_type"] = config.build_custom_model_type
    df["checkpoint"] = config.checkpoint
    df["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    df["lr"] = config.lr
    df["n_epochs"] = config.n_epochs
    df["random_seed"] = config.random_seed
    df["loss"] = config.loss
    df["loss_distance"] = config.loss_distance
    df["loss_reducer"] = config.loss_reducer
    df["loss_optimizer"] = config.loss_optimizer
    df["loss_lr"] = config.loss_lr
    df["loss_margin"] = config.loss_margin
    df["miner_type_of_triplets"] = config.miner_type_of_triplets
    df["optimizer_name"] = config.optimizer_name
    df["save_flag"] = config.save_flag
    df["data_path"] = config.data_path
    df["item_col_name"] = config.item_col_name
    df["cat_col_name"] = config.cat_col_name
    df["num_col_names"] = config.num_col_names
    df["scale_num_cols"] = config.scale_num_cols
    df["group_col_name"] = config.group_col_name
    df["fold_col_name"] = config.group_col_name
    df["mlflow"] = config.mlflow 
    df["experiment_path"] = config.experiment_path
    df["experiment_id"] = config._experiment_id 
    df["run_description"] = config.run_description
    return df
