# Utility functions for artifact storage: saving, loading, copying, and removing

import shutil
import torch
import pickle
import json
import pandas as pd
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from sklearn.preprocessing import LabelEncoder
from re import search

from config import CFG
from model import SeqEmbedModel, NumSeqModel

# Internal


def make_proj_dir_tree(CFG: CFG):
    """Creates the directory tree if it doesn't exist yet; see the next functions for details
    Folders:
    - models/            - trained models
    - label_encoders/    - trained categorical label encoders
    - preproc/           - fitted preprocessing artifacts, such as scalers
    - temp/              - temporary storage for models
    - logs/              - holds log and oof files from pipeline run
    Files:
    - model_checkpoint_map*json
    """
    proj_dir = Path(CFG.proj_dir)
    subdir_list = ["models", "label_encoders", "preproc", "temp", "logs"]

    make_dirs(proj_dir, subdir_list)
    make_model_checkpoint_map_file(proj_dir)


def make_dirs(proj_dir: Path, subdir_list: list):
    """Creates directories in project directory tree"""
    dir_list = [proj_dir] + [proj_dir / subdir for subdir in subdir_list]
    for i, path in enumerate(dir_list):
        if not path.is_dir():
            path.mkdir()
            print(
                f"- Made {'project' if i==0 else path.name.replace('_', ' ')} directory {str(path)}"
            )


def make_model_checkpoint_map_file(proj_dir: Path):
    """Creates model_check_map*json file:
    This file connects trained model IDs to checkpoint names. Primarily done as convenience
    to reload the corresponding HF tokenizer upon inference. Could also solve this by saving
    the tokenizer separately, but current solution is more efficient and pretty cheap.
    """
    path = proj_dir / "model_checkpoint_map.json"
    if not path.is_file():
        blank_json = json.dumps({})
        with path.open("w") as f:
            f.write(blank_json)
        print(f"- Made model-checkpoint map file {str(path)}")


def save_model(model: PreTrainedModel, model_id: str, CFG: CFG, temp: bool = False):
    """Saves model to either models/ or temp/ directory:
    Purpose is to rescue a run where we forgot to set save_model_flag = True but realised we want to save the model.
    Every experiment run saves their models and label encoders. This prevents having to rerun (long) experiments.
    Temp save files get overwritten in the next run, so make sure to carefully check after each experiment.
    """
    model_path = (
        Path(CFG.proj_dir) / f"{'temp' if temp else 'models'}" / model_id
    )
    model.save_pretrained(model_path)
    # torch.save(model, model_path)
    print(f"- Model {'temporarily ' if temp else ''}saved at {str(model_path)}")


def save_le(label_encoder: LabelEncoder, model_id: str, CFG: CFG, temp: bool = False):
    """Same as save_model but for label encoders"""
    le_path = (
        Path(CFG.proj_dir)
        / f"{'temp' if temp else 'label_encoders'}"
        / f"le_{model_id}.pkl"
    )
    with le_path.open("wb") as f:
        pickle.dump(label_encoder, f)
    print(f"- Label encoder {'temporarily ' if temp else ''}saved at {str(le_path)}")


def save_scaler(scaler, model_id: str, CFG: CFG, temp: bool = False):
    """Same as save_model but for scalers"""
    scaler_path = (
        Path(CFG.proj_dir)
        / f"{'temp' if temp else 'preproc'}"
        / f"scaler_{model_id}.pkl"
    )
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    print(f"- Scaler {'temporarily ' if temp else ''}saved at {str(scaler_path)}")


def save_train_log(train_log_df: pd.DataFrame, model_id: str, CFG: CFG):
    """Save oof and log dfs for each model to csv files"""
    train_log_df.to_csv(
        Path(CFG.proj_dir) / "logs" / f"{model_id}_log.csv", index=False
    )

    print(f"- Log data saved as csv file in {str(Path(CFG.proj_dir) / 'logs')}")


# User-facing and Internal


def load_model(
    model_id: str, proj_dir: str, from_temp: bool = False, device: torch.device = None, custom_model_type:str=None, custom:bool=None
):
    """Load a trained HF Sequence Classification model from our project directory"""
    model_path = (
        Path(proj_dir)
        / f"{'temp' if from_temp else 'models'}"
        / f"{model_id if not custom else model_id + '.pt'}"
    )
    if custom: # for backwards compatibility with Shoppee's pt NumModel for Indonesia
        if device == torch.device("cpu"):
            model = torch.load(model_path, map_location=device)
        else:
            model = torch.load(model_path)
        return model, str(model_path)
    # hard-coded AutoModelForSequenceClassification here; keep in mind when adjusting to other NLP tasks
    if not custom_model_type:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    elif custom_model_type == "num_metric":
        model = NumSeqModel.from_pretrained(model_path)
    elif custom_model_type == "metric":
        model = SeqEmbedModel.from_pretrained(model_path)
    return model, str(model_path)


def load_le(base_model_id: str, proj_dir: str, from_temp: bool = False):
    """Reload fitted label encoder from our project directory"""
    le_path = (
        Path(proj_dir)
        / f"{'temp' if from_temp else 'label_encoders'}"
        / f"le_{base_model_id}.pkl"
    )
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder, str(le_path)


def load_scaler(model_id: str, proj_dir: str, from_temp: bool = False):
    """Reload fitted scaler from our project directory"""
    scaler_path = (
        Path(proj_dir)
        / f"{'temp' if from_temp else 'preproc'}"
        / f"scaler_{model_id}.pkl"
    )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler, str(scaler_path)


def get_model_checkpoint(model_id: str, proj_dir: str):
    """Get the correct checkpoint from the model_name using the model_checkpoint_map"""
    model_checkpoint_map_path = Path(proj_dir) / "model_checkpoint_map.json"
    with open(model_checkpoint_map_path, "r") as map_json_file:
        map_json = json.load(map_json_file)
        checkpoint = map_json[model_id]

    return checkpoint


def save_model_checkpoint_map(model_id: str, checkpoint: str, proj_dir: str):
    """Add a model ID to an existing model_checkpoint_map dictionary (written by make_model_checkpoint_map_file)"""
    path = Path(proj_dir) / "model_checkpoint_map.json"
    with path.open("r") as f:
        model_checkpoint_map_json = json.load(f)
    new_mapping = {model_id: checkpoint}
    model_checkpoint_map_json.update(new_mapping)
    with path.open("w") as f:
        json.dump(model_checkpoint_map_json, f)
    print(f"- Added map {new_mapping} to {str(path)}")


def clear_temp(proj_dir_or_CFG: str):
    """Removes all contents of the temp directory. Preserves temp dir."""
    if isinstance(proj_dir_or_CFG, str):
        path = Path(proj_dir_or_CFG) / "temp"
    elif isinstance(proj_dir_or_CFG, CFG) and proj_dir_or_CFG.clear_temp:
        path = Path(proj_dir_or_CFG.proj_dir) / "temp"
    else:
        return None
    try:
        temp_contents = [f for f in path.iterdir()]
    except FileNotFoundError:
        return None
    if len(temp_contents) > 0:
        for item_path in temp_contents:
            shutil.rmtree(item_path) if item_path.is_dir() else item_path.unlink()
        print(f"- Deleted {len(temp_contents)} items from temporary directory")


# User-facing


def copy_from_temp(
    model_temp_id: str,
    model_new_id: str,
    proj_dir: str,
    checkpoint: str = None,
    save_map: bool = False,
    save_le: bool = True,
    delete_temp_copy: bool = True,
    clear_temp: bool = False,
    overwrite: bool = False,
    custom: bool = False,
):
    """Bring back models and (optionally) label encoders that were saved to temp.
    Requires to specify model names to write to in the model/ directory.
    If artifacts with name model_new_id already exist, must pass overwrite = True to overwrite them.

    Parameters:
    - model_temp_id: name of model dir in temp/
    - model_new_id: new name of model in models/, label encoder in label_encoders/, and model_id key in model_checkpoint_map.json
    - proj_dir: project directory path containing models/, label_encoders/, temp/, and model_checkpoint_map.json
    - checkpoint: name of model checkpoint to be inputted in model_checkpoint_map.json
    - save_map: whether to append model_new_id : checkpoint pair to model_checkpoint_map.json
    - save_le: whether to copy label_encoder from temp to label_necoders/
    - delete_temp_copy: whether to delete temp/ copy of model and label_encoder after copying to permanent location
    - clear_temp: whether to erase all contents of temp/ after copying specified artifacts
    - overwrite: whether to overwrite artifacts with name model_new_id if they already exist before copying
    """
    # copy model
    if not custom:
        copy_model_from_temp(
            model_temp_id, model_new_id, proj_dir, delete_temp_copy, overwrite=overwrite
        )
    else:
        copy_custom_model_from_temp(
            model_temp_id, model_new_id, proj_dir, delete_temp_copy, overwrite=overwrite
        )
    # copy le
    if save_le:
        copy_le_from_temp(model_new_id, proj_dir, delete_temp_copy, overwrite=overwrite)
    # Save model checkpoint map
    if save_map:
        if checkpoint:
            save_model_checkpoint_map(model_new_id, checkpoint, proj_dir)
        else:
            raise ValueError("No checkpoint specified")
    # clear temp
    if clear_temp:
        clear_temp(proj_dir)


def copy_model_from_temp(
    model_temp_id: str,
    model_new_id: str,
    proj_dir: str,
    delete_temp_copy: bool = True,
    overwrite: bool = False,
):
    """Copy model, saved with HF, from temp/ to models/"""
    model_new_path = Path(proj_dir) / "models" / model_new_id
    model_temp_path = Path(proj_dir) / "temp" / model_temp_id
    shutil.copytree(model_temp_path, model_new_path, dirs_exist_ok=overwrite)
    print(f"Copied model from {str(model_temp_path)} to {str(model_new_path)}")
    if delete_temp_copy:
        shutil.rmtree(model_temp_path)
        print(f"Deleted model from temp")


def copy_custom_model_from_temp(
    model_temp_id: str,
    model_new_id: str,
    proj_dir: str,
    delete_temp_copy: bool = True,
    overwrite: bool = False,
):
    """Copy custom model, saved with torch, from temp/ to models/"""
    model_new_path = Path(proj_dir) / "models" / model_new_id + ".pt"
    model_temp_path = Path(proj_dir) / "temp" / model_temp_id + ".pt"
    if os.path.exists(model_new_path) and not overwrite:
        raise FileExistsError(
            f"{model_new_path} already exists. To overwrite, set overwrite=True"
        )
    else:
        shutil.copy(model_temp_path, model_new_path)
    print(f"Copied model from {str(model_temp_path)} to {str(model_new_path)}")
    if delete_temp_copy:
        os.remove(model_temp_path)
        print(f"Deleted model from temp")


def copy_le_from_temp(
    model_new_id: str,
    proj_dir: str,
    delete_temp_copy: bool = True,
    overwrite: bool = False,
):
    """Copy label encoder from temp/ to label_encoders/"""
    le_new_path = Path(proj_dir) / "label_encoders" / f"le_{model_new_id}.pkl"
    le_temp_path = find_temp_le(proj_dir)
    if le_new_path.is_file() and not overwrite:
        raise FileExistsError(f"Label encoder already exists: {str(le_new_path)}")
    else:
        shutil.copy(le_temp_path, le_new_path)
        print(f"Copied label encoder from {str(le_temp_path)} to {str(le_new_path)}")
        if delete_temp_copy:
            le_temp_path.unlink()
            print(f"Deleted label encoder from temp")


def find_temp_le(proj_dir: str):
    """Get path of label encoder from temp directory if there is a single one there, else raise error"""
    temp_dir = Path(proj_dir) / "temp"
    le_in_temp_list = [item for item in os.listdir(temp_dir) if search("^le_", item)]
    if len(le_in_temp_list) == 0:
        raise FileNotFoundError(f"No label encoder found in {str(temp_dir)}")
    elif len(le_in_temp_list) > 1:
        raise FileNotFoundError(
            f"Multiple label encoders found in {str(temp_dir)}. Save manually."
        )
    else:
        return temp_dir / le_in_temp_list[0]


def load_logs(proj_dir:str, regex: str = None):
    """Load multiple log csvs in a project director into one pd dataframe."""
    if not regex:
        regex = ".*"
        
    base_path = Path(proj_dir) / "logs"
    logs = [item for item in os.listdir(base_path) if search(regex, item)]
    log_paths = [base_path / Path(log) for log in logs]
    
    log_dfs = [pd.read_csv(log_path) for log_path in log_paths]
    log_df = pd.concat(log_dfs, axis=0, ignore_index=True)
    
    return log_df

def load_log(proj_dir: str, model_id: str):
    """Load log csv from model run into one pd dataframe."""
    log_df = pd.read_csv(Path(proj_dir) / "logs" / f"{model_id}_log.csv")
    return log_df