from config import CFG
import mlflow

LOGGED_PARAMS = [
    "cat_col_name", 
    "item_col_name", 
    "num_col_names", 
    "scale_num_cols", 
    "pool_mode", 
    "subset",
    "n_folds",
    "val_folds",
    "random_seed",
    "checkpoint",
    "custom",
    "max_len",
    "bs",
    "gradient_accumulation_steps",
    "optimizer_name",
    "lr",
    "n_epochs",
    "loss",
    "loss_distance",
    "loss_reducer",
    "loss_optimizer",
    "loss_lr",
    "loss_margin",
    "miner_type_of_triplets",
    "class_weights",
    "fl_gamma",
    "data_path", 
    "proj_dir",
    "build_custom_model_type",
    "group_col_name",
    "fold_col_name",
    ]

def log_config_params(CFG:CFG, optional_params={}):
    params_to_log = {param:value for param, value in vars(CFG).items() if param in LOGGED_PARAMS}
    if optional_params:
        params_to_log.update(optional_params)
    mlflow.log_params(params_to_log)