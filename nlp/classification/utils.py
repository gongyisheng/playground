import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os
import torch
import random
from config import CFG
from storage import make_proj_dir_tree, clear_temp


def setup(CFG: CFG = CFG):
    # Time keeping
    run_start_dt = datetime.datetime.utcnow()
    # model ID
    base_model_id = get_base_model_id(CFG, run_start_dt)
    # Set random seed
    set_seed(CFG)
    # Print run details
    if CFG.mlflow:
        print(
            "------------------------------------\n"
            "MLFlow Experiment\n"
            f"experiment name: {Path(CFG.experiment_path).name}\n"
            f"experiment id: {CFG._experiment_id}\n"
            f"base_model_id/run_name: {base_model_id}\n"
            f"n_epochs: {CFG.n_epochs}\n"
            f"n_val_folds: {CFG.val_folds}\n"
            f"dt: {str(run_start_dt)}\n"
            "------------------------------------"
        )
    else:
        print(
            "------------------------------------\n"
            f"base_model_id: {base_model_id}\n"
            f"n_epochs: {CFG.n_epochs}\n"
            f"n_val_folds: {CFG.val_folds}\n"
            f"dt: {str(run_start_dt)}\n"
            "------------------------------------"
        )
    # Create project directory
    make_proj_dir_tree(CFG)
    clear_temp(CFG)

    return base_model_id, run_start_dt


def get_base_model_id(CFG: CFG, run_start_dt: datetime.datetime):
    """Returns the ID of the experiment run, concatenating checkpoint name and datetime of the run"""
    name_parts = [
        CFG.checkpoint.replace("/", "_"),
        run_start_dt.strftime("%Y_%m_%d_%H_%M"),
    ]
    if CFG.model_name:
        name_parts.insert(0, CFG.model_name)
    return "__".join(name_parts)


def set_seed(CFG: CFG):
    """Set random seeds for reproducibility"""
    if CFG.random_seed != None:
        random.seed(CFG.random_seed)  # Python
        np.random.seed(CFG.random_seed)  # cpu vars
        torch.manual_seed(CFG.random_seed)  # cpu vars
        if torch.cuda.is_available():
            torch.cuda.manual_seed(CFG.random_seed)
            torch.cuda.manual_seed_all(CFG.random_seed)  # gpu vars
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
    CFG: CFG,
):
    """Initiate the training log file"""
    df = pd.DataFrame(train_results)
    df["model_id"] = model_id
    df["base_model_id"] = base_model_id
    df["fold"] = val_fold
    df["max_len"] = CFG.max_len
    df["auto_max_len"] = auto_max_len if CFG.max_len == "auto" else None
    df["subset"] = CFG.subset
    df["pool_mode"] = CFG.pool_mode
    df["bs"] = CFG.bs
    df["build_custom_model_type"] = CFG.build_custom_model_type
    df["checkpoint"] = CFG.checkpoint
    df["gradient_accumulation_steps"] = CFG.gradient_accumulation_steps
    df["lr"] = CFG.lr
    df["n_epochs"] = CFG.n_epochs
    df["random_seed"] = CFG.random_seed
    df["loss"] = CFG.loss
    df["loss_distance"] = CFG.loss_distance
    df["loss_reducer"] = CFG.loss_reducer
    df["loss_optimizer"] = CFG.loss_optimizer
    df["loss_lr"] = CFG.loss_lr
    df["loss_margin"] = CFG.loss_margin
    df["miner_type_of_triplets"] = CFG.miner_type_of_triplets
    df["optimizer_name"] = CFG.optimizer_name
    df["save_flag"] = CFG.save_flag
    df["data_path"] = CFG.data_path
    df["item_col_name"] = CFG.item_col_name
    df["cat_col_name"] = CFG.cat_col_name
    df["num_col_names"] = CFG.num_col_names
    df["scale_num_cols"] = CFG.scale_num_cols
    df["group_col_name"] = CFG.group_col_name
    df["fold_col_name"] = CFG.group_col_name
    df["mlflow"] = CFG.mlflow 
    df["experiment_path"] = CFG.experiment_path
    df["experiment_id"] = CFG._experiment_id 
    df["run_description"] = CFG.run_description
    return df
