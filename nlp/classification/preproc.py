# Data preprocessing functions
# This is everything pre dataloader / datablock

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from config import CFG
from storage import save_le, save_scaler
from utils import get_num_col_names_iter


def preprocess(data_df: pd.DataFrame, base_model_id: str, CFG: CFG):
    """Single-function preprocessing, chaining the specific preproc functions below"""
    data_df = apply_labels(data_df, CFG)
    data_df, label_encoder = encode_labels(data_df)
    data_df = subset_data(data_df, CFG)
    if not CFG.fold_col_name:
        data_df = assign_folds(data_df, CFG)
    if CFG.save_flag:
        save_le(label_encoder, base_model_id, CFG)
    elif CFG.temp_flag: 
        save_le(label_encoder, base_model_id, CFG, temp=True)
    return data_df, label_encoder 


def apply_labels(data_df: pd.DataFrame, CFG: CFG):
    """Optional subsetting of category list, with all other categories being grouped into Other
    Supplied with CFG.target_cat_list
    """
    data_df["cat"] = (
        data_df[CFG.cat_col_name].apply(
            lambda x: x if x in CFG.target_cat_list else "Other"
        )
        if CFG.target_cat_list
        else data_df[CFG.cat_col_name]
    )
    return data_df


def encode_labels(data_df: pd.DataFrame):
    """Label encoding fit_transform step"""
    label_encoder = LabelEncoder()
    data_df["labels"] = label_encoder.fit_transform(data_df["cat"])
    return data_df, label_encoder


def subset_data(data_df: pd.DataFrame, CFG: CFG):
    """Optional sub-sampling step, with sample fraction supplied by CFG.subset"""
    subset_df = (
        data_df.groupby("labels", group_keys=False)
        .apply(lambda x: x.sample(frac=CFG.subset, random_state=CFG.random_seed))
        .sample(frac=1.0, random_state=CFG.random_seed)
        .reset_index(drop=True)
    )
    if CFG.subset < 1.0:
        print(f"Running on input data subset: {(CFG.subset * 100):.2f}%")
    return subset_df


def assign_folds(data_df: pd.DataFrame, CFG: CFG):
    """Create CV folds for number of folds n_folds"""
    data_df = data_df.reset_index(drop=True)
    if CFG.group_col_name:
        skf = StratifiedGroupKFold(
            n_splits=CFG.n_folds, shuffle=True, random_state=CFG.random_seed
        )
        for fold, (_, val_index) in enumerate(skf.split(X=data_df, y=data_df["cat"], groups=data_df[CFG.group_col_name])):
            data_df.loc[val_index, "fold"] = fold
    else:
        skf = StratifiedKFold(
            n_splits=CFG.n_folds, shuffle=True, random_state=CFG.random_seed
        )
        # standard val index assign
        for fold, (_, val_index) in enumerate(skf.split(X=data_df, y=data_df["cat"])):
            data_df.loc[val_index, "fold"] = fold

    # this will be useful for fastai
    # data_df["is_valid"] == null

    return data_df


def create_and_apply_num_scaler(
    data_df: pd.DataFrame, CFG: CFG, val_fold: int, model_id: str
):
    """Instantiate and apply scaler to numerical data"""
    num_col_list = get_num_col_names_iter(CFG.num_col_names)
    scaler = fit_scaler(data_df, num_col_list, val_fold)
    scaled_df = scale_data(scaler, data_df, num_col_list)
    if CFG.save_flag:
        save_scaler(scaler, model_id, CFG)
    elif CFG.temp_flag:
        save_scaler(scaler, model_id, CFG, temp=True)
    return scaled_df, scaler


def fit_scaler(df: pd.DataFrame, num_col_list: list, val_fold: int):
    """Fit an sklearn scaler to training data numerical columns"""
    scaler = MinMaxScaler()
    if len(num_col_list) == 1:
        scaler.fit(df.loc[df["fold"] != val_fold, num_col_list].values.reshape(-1, 1))
    else:
        scaler.fit(df.loc[df["fold"] != val_fold, num_col_list].values)
    return scaler


def scale_data(scaler: MinMaxScaler, df: pd.DataFrame, num_col_list: list):
    """Scale num data with scaler fit to training data"""
    if len(num_col_list) == 1:
        df[num_col_list] = scaler.transform(df[num_col_list].values.reshape(-1, 1))
    else:
        df[num_col_list] = scaler.transform(df[num_col_list].values)
    return df
