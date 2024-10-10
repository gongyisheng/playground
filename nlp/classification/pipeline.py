# End-to-end classification pipeline

import datetime
import pandas as pd
from config import CFG, validate_and_transform_config
from preproc import preprocess
from utils import setup, get_val_fold_iter
from train_1_fold import train_1_fold


def run_training_pipeline(df: pd.DataFrame, CFG: CFG = CFG):
    """Run the pipeline; only inputs are input df and CFG"""
    # Validate
    validate_and_transform_config(CFG, df)

    # Setup
    base_model_id, run_start_dt = setup(CFG)

    # Copy input df
    data_df = df.copy()

    # Preprocessing
    print("Preprocessing data...")
    preproc_start_dt = datetime.datetime.utcnow()
    preproc_data_df, label_encoder = preprocess(data_df, base_model_id, CFG)
    print(
        f"Preprocessing complete. Time elapsed: {datetime.datetime.utcnow() - preproc_start_dt}"
    )

    # Iterate over validation folds
    val_fold_list = get_val_fold_iter(CFG.val_folds)
    cv_results = [] #TODO: Should this be list (in order of fold), or dict mapping run/vald_fold to fold_results?
    for val_fold in val_fold_list:
        fold_results = train_1_fold(
            preproc_data_df, label_encoder, val_fold, base_model_id, CFG, val_fold_list
        )

        cv_results.append(fold_results)

    if not CFG.save_flag and CFG.temp_flag:
        print("- NOTE: all temporarily saved files will be deleted next pipeline run")

    print(
        f"Pipeline run complete. Time elapsed: {datetime.datetime.utcnow() - run_start_dt}"
    )

    return cv_results