# End-to-end classification pipeline

import datetime
import pandas as pd

from config import Config, validate_and_transform_config
from utils import setup
import gc


def run_training_pipeline(df: pd.DataFrame, config: Config = Config()):
    """Run the pipeline; only inputs are input df and config"""
    # Validate
    validate_and_transform_config(config, df)

    # Setup
    base_model_id, run_start_dt = setup(config)

    # # Copy input df
    # data_df = df.copy()

    # # Preprocessing
    # print("Preprocessing data...")
    # preproc_start_dt = datetime.datetime.utcnow()
    # preproc_data_df, label_encoder = preprocess(data_df, base_model_id, CFG)
    # print(
    #     f"Preprocessing complete. Time elapsed: {datetime.datetime.utcnow() - preproc_start_dt}"
    # )

    # # Iterate over validation folds
    # val_fold_list = get_val_fold_iter(CFG.val_folds)
    # cv_results = [] #TODO: Should this be list (in order of fold), or dict mapping run/vald_fold to fold_results?
    # for val_fold in val_fold_list:

    #     fold_results = train_1_fold(
    #         preproc_data_df, label_encoder, val_fold, base_model_id, CFG, val_fold_list
    #     )

    #     cv_results.append(fold_results)

    # if CFG.return_only_log:
    #     cv_results = pd.concat([fold_result_dict["log"] for fold_result_dict in cv_results], ignore_index=True)
    #     gc.collect()

    #     # if CFG.return_artifacts:
    #     #     cv_results[str(val_fold)] = fold_results

    #     # leaving this in for backwards compatibility
    #     # next_train_log_df, next_oof_df = fold_results["log"], fold_results["oof"]

    #     # if val_fold_list.index(val_fold) == 0:
    #     #     train_log_df = next_train_log_df.copy()
    #     #     oof_df = next_oof_df.copy()
    #     # else:
    #     #     train_log_df = pd.concat(
    #     #         [train_log_df, next_train_log_df], ignore_index=True
    #     #     )
    #     #     oof_df = pd.concat([oof_df, next_oof_df], ignore_index=True)

    # if not CFG.save_flag and CFG.temp_flag:
    #     print("- NOTE: all temporarily saved files will be deleted next pipeline run")

    # print(
    #     f"Pipeline run complete. Time elapsed: {datetime.datetime.utcnow() - run_start_dt}"
    # )

    # return cv_results

    # # if CFG.return_artifacts:
    # #     if len(cv_results) == 1:
    # #         return cv_results[str(val_fold)]
    # #     else:
    # #         return cv_results
    # # else:
    # #     return {"log":train_log_df, "oof":oof_df}
