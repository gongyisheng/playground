import json
import mlflow
from huggingface_hub import list_repo_files
import pandas as pd
import pandas.api.types as ptypes
import random
from pathlib import Path

class CFG(object):
    """
    Config class defining pipeline parameters. Must be instantiated before running pipeline.

    Attributes:
    - model name: str or None                    - The base_model_id to be used in storing the trained model, label encoder, and logs.
        default: None                              If None, base_model_id defined by get_base_model_id().
    - save_flag: bool                            - Save the model in models/ and label encoder in label_encoders/,
        default: False                             and append "{model_id}": "{checkpoint}" to model_checkpoint_map.json.
    - temp_flag: bool                            - Save the model and artifacts in temp/ if save_flag=False. This is a failsafe in case you forget to
        default: True                              permanently save a model that you ultimately want to keep.
    - clear_temp: bool                           - Clear temp folder at beginning of run.
        default: True
    - target_cat_list: list[str] or None         - List of categories in data_df to use in training. All categories in data_df not in this list
        default: None                              will be mapped to "Other" category.
                                                   If None, all categories in data_df will be used and no "Other" category will be constructed.
    - cat_col_name: str                          - Name of column in data_df containing categories.
        default: "category"
    - item_col_name: str                         - Name of column in data_df containing text to be categorized.
        default: "item"
    - num_col_names: list[str] or str            - Name(s) of column(s) in data_df containing numerical features. If specified, a custom model incorporating
        default: None                              both text embeddings and numerical features will be built as a wrapper over the pretrained base model
                                                   specified by the checkpoint parameter. Can input single column as string.
    - scale_num_cols: bool                       - Whether to scale (between 0 and 1) cols in num_col_names. Scaler will be saved for use in inference.
        default: False
    - pool_mode: str                             - Pooler to be used to combine token embeddings into sequence embedding. If num_col_names is specified, pooling occurs
        default: "mean"                            before numerical features are appended to embeddings. List of supported poolers found in self._pool_modes.
    - subset: float                              - Proportion of data_df to be used in pipeline. Subset is taken by stratifying by category.
        default: 1.0                               Must be between 0.0 and 1.0 (inclusive).
    - n_folds: int                               - Number of CV folds to be created. Not all folds must be used for validation.
        default: 5
    - val_folds: list[int] or int                - If list[int], specific folds to be used for validation. Elements must be distinct and <= n_folds.
        default: 5                                 If int, number of folds to be used for validation. Validation folds are selected sequentially
                                                   from 0 (e.g. val_folds = 3 -> folds 0, 1, 2 used for validation). Must be <= n_folds.
    - random_seed: int                           - Random seed used for numerous functions in pipeline. Must be >= 0.
    - checkpoint: str                            - Name of model checkpoint used to load model from HF Hub, or a path to a model checkpoint saved locally.
        default: "distilroberta-base"              If checkpoint points to a model on HF Hub, must be a model that permits AutoModelForSequenceClassification.
    - max_len: int or "auto"                     - Maximum length of tokenized inputs permitted before truncation by tokenizer.
        default: "auto"                            If "auto", largest length of tokenized inputs found by get_max_len() and used.
    - bs: int                                    - Batch size used in creation of dataloaders. Must be >= 1.
        default: 64
    - gradient_accumulation_steps: int           - Number of batches passed to model before gradients are calculated. Gradients are
        default: 1                                 summed before calculation. If > 1, effectively increases batch_size to steps * bs.
    - optimizer_name: str                        - Name of optimizer to be used in training. Supported optimizers can be found in
        default: "AdamW"                           self._optimizers.
    - lr: float                                  - Learning rate. Must be > 0.0.
        default: 3e-5
    - n_epochs: int                              - Number of training epochs. Must be > 0.
        default: 3
    - eval_metrics: list[str]                    - List of evaluation metrics to be calculated on the eval set during training. Supported metrics can
        default: ["map@r"]                         be found in self._metrics.
    - loss: str                                  - Loss function used in training. Supported loss functions can be found in self._loss_fns
        default: "tripletmargin"
    - device: str                                - Device pipeline operations will use. If "auto", priority of devices can be found in set_device().
        default: "auto"                            Supported devices can be found in self._devices.
    - proj_dir: str                              - Root directory (local or remote) that pipeline artifacts will be saved in. Project directory tree
        default: "."                               will be built inside.
    - return_embeddings: bool                    - Whether to return train and test embeddings, and corresponding preprocessed train and test dfs
        default: True
    - data_path: str or None                     - Path to training data. Should be either a table name or a dbfs path.
        default: None
    - return_artifacts: bool                     - Whether to return all model run artifacts in addition to log and oof. 
        default: False
    - group_col_name: str or None                - Name of column used to group items for GroupedStratifiedKFold. 
        default: None
    - fold_col_name: str or None                 - Name of column used for custom folds
        default: None
    - build_custom_model_type: str or None       - Name of custom model type to build, if not none. Supported model types can be found in self._custom_model_types.
        default: None                              "metric_classification" model type requires that a SeqEmbedModel be saved locally with path specified in self.checkpoint.

   MLFLOW-specific attributes
    - mlflow: bool                               - Whether to log params, metrics, and model in mlflow. Whether or not model is saved depends
        default: False                             on if save_flag == true
    - experiment_path: str or None               - Workspace path in which the experiment lives
        default: None
    - _experiment_id: NOT ACCESSIBLE             - Validated and identified by class automatically.
    - run_description: str or None               - Optionally add a description of the run
        default: None
    """

    _is_frozen = False

    _pool_modes = ["mean", "max"]
    _optimizers = ["AdamW"]
    _metrics = ["precision@1", "ami", "map@r"]
    _loss_fns = ["tripletmargin", "arcface", "proxyanchor"]
    _class_weights_fns = ["inv_size", "balanced"]
    _devices = ["auto", "cuda", "mps", "cpu"]
    _miner_type_of_triplets = ["default", "all", "hard", "semihard", "easy"]
    _loss_distances = ["default", "cosine", "dot", "lp", "snr"]
    _loss_reducers = ["default", "avgnonzero", "mean", "sum"]
    _custom_model_types = ["num_metric", "metric"]

    def __init__(self):
        """Initialize default attr values and freeze object."""
        self._set_defaults()
        self._freeze()

    def _set_defaults(self):
        """Set config defaults."""
        self.model_name = None
        self.save_flag = False
        self.temp_flag = True
        self.clear_temp = True
        self.target_cat_list = None
        self.cat_col_name = "category"
        self.item_col_name = "item"
        self.num_col_names = None
        self.scale_num_cols = False
        self.pool_mode = "mean"
        self.subset = 1.0
        self.n_folds = 5
        self.val_folds = 5
        self.random_seed = 4321
        self.build_custom_model_type = "metric"
        self.checkpoint = "distilroberta-base"
        self.max_len = "auto"
        self.bs = 64
        self.gradient_accumulation_steps = 1
        self.optimizer_name = "AdamW"
        self.lr = 3e-5
        self.n_epochs = 3
        self.eval_metrics = ["map@r"]
        self.loss = "tripletmargin"
        self.loss_distance = "default" #TODO: add to docstring
        self.loss_reducer = "default" #TODO: add to docstring
        self.loss_optimizer = "AdamW" #TODO: add to docstring
        self.loss_lr = 1e-2 #TODO: add to docstring
        self.loss_margin = "default" #TODO: add to docstring
        self.miner_type_of_triplets = "default" #TODO: add to docstring
        self.device = "auto"
        self.proj_dir = "."
        self.return_embeddings = False
        self.data_path = None
        self.mlflow = False
        self.experiment_path = None
        self._experiment_id = None
        self.run_description = None
        self.return_artifacts = False 
        self.group_col_name = None
        self.end_last_run = True # For debugging
        self.metric_calc_device = "cpu" #TODO: add to docstring
        self.fold_col_name = None

    def _validate(self, key: str, value) -> bool:
        """Validate attribute value according to specified conditions."""
        try:
            if key == "model_name":
                assert not value or isinstance(value, str)
            if key == "save_flag":
                assert isinstance(value, bool)
            if key == "temp_flag":
                assert isinstance(value, bool)
            if key == "clear_temp":
                assert isinstance(value, bool)
            if key == "target_cat_list":
                assert not value or isinstance(value, list)
            if key == "cat_col_name":
                assert isinstance(value, str)
            if key == "item_col_name":
                assert isinstance(value, str)
            if key == "num_col_names":
                assert not value or isinstance(value, str) or isinstance(value, list)
            if key == "scale_num_cols":
                assert isinstance(value, bool)
            if key == "pool_mode":
                assert value in self._pool_modes
            if key == "subset":
                assert isinstance(value, float) and value <= 1 and value > 0
            if key == "n_folds":
                assert isinstance(value, int) and value > 1
            if key == "val_folds":
                assert (
                    isinstance(value, int) and value >= 1 and value <= self.n_folds
                ) or (
                    isinstance(value, list)
                    and len(value) >= 1
                    and len(value) <= self.n_folds
                    and max(value) <= self.n_folds
                    and min(value) >= 0
                )
            if key == "random_seed":
                assert isinstance(value, int) and value >= 0
            if key == "checkpoint":
                assert isinstance(value, str)
            if key == "max_len":
                assert (isinstance(value, int) and value > 0) or value == "auto"
            if key == "bs":
                assert isinstance(value, int) and value >= 1
            if key == "gradient_accumulation_steps":
                assert isinstance(value, int)
            if key == "optimizer_name":
                assert isinstance(value, str) and value in self._optimizers
            if key == "lr":
                assert isinstance(value, float) and value > 0 and value < 1
            if key == "n_epochs":
                assert isinstance(value, int) and value > 0
            if key == "eval_metrics":
                assert (
                    isinstance(value, list) and set(value).issubset(set(self._metrics))
                ) or (isinstance(value, str) and value in self._metrics)
            if key == "loss":
                assert isinstance(value, str) and value in self._loss_fns
            if key == "loss_distance":
                assert isinstance(value, str) and value in self._loss_distances
            if key == "loss_reducer":
                assert isinstance(value, str) and value in self._loss_reducers
            if key == "loss_optimizer":
                assert isinstance(value, str) and value in self._optimizers
            if key == "loss_lr":
                assert isinstance(value, float) and value > 0 and value < 1
            if key == "loss_margin":
                assert (value == "default") or (isinstance(value, float) and value > 0)
            if key == "miner_type_of_triplets":
                assert isinstance(value, str) and value in self._miner_type_of_triplets
            if key == "device":
                assert isinstance(value, str) and value in self._devices
            if key == "proj_dir":
                assert isinstance(value, str)
                assert Path(value).parent.exists() # Parent dir needs to exist to create directory tree
            if key == "return_embeddings":
                assert isinstance(value, bool)
            if key == "data_path":
                assert not value or isinstance(value, str)
            if key == "return_artifacts":
                assert isinstance(value, bool)
            if key == "group_col_name":
                assert not value or isinstance(value, str)
            if key == "mlflow":
                assert isinstance(value, bool)
            if key == "experiment_path":
                assert not value or isinstance(value, str)
                if value:
                    try:
                        self._experiment_id = int(dict(mlflow.get_experiment_by_name(value))["experiment_id"])
                    except:
                        self._experiment_id = None
                        raise ValueError(f"Invalid experiment path. Experiment does not exist")
                else:
                    self._experiment_id = None
            if key == "run_description":
                assert not value or isinstance(value, str)
            if key == "build_custom_model_type":
                assert value in self._custom_model_types
            if key == "metric_calc_device":
                assert isinstance(value, str) and value in ["cpu", "cuda"]
            if key == "end_last_run":
                assert isinstance(value, bool)
            if key == "fold_col_name":
                assert not value or isinstance(value, str)
            return True
        except AssertionError:
            return False

    def _freeze(self):
        """Freeze object after instantiation."""
        self._is_frozen = True

    def __setattr__(self, key: str, value):
        """If attr is not set in init or is not valid, throw error."""
        if not hasattr(self, key) and self._is_frozen:
            raise TypeError(f"CFG does not support {key}")
        elif not self._validate(key, value):
            raise ValueError(f"Illegal value for CFG.{key}")
        object.__setattr__(self, key, value)

    def _get_config_dictionary(self):
        """Convert attr, value pairs to dict."""
        config = {}
        for value in [k for k in dict(vars(self)).keys() if "__" not in k]:
            config[value] = getattr(self, value)
        return config

    def __repr__(self):
        config = self._get_config_dictionary()
        return json.dumps(config, indent=4)

def validate_and_transform_config(CFG:CFG, df:pd.DataFrame):
    if CFG.fold_col_name:
        validate_and_transform_config_for_custom_folds(CFG, df)

    if isinstance(CFG.val_folds, int): # We always want this to be an iterator
        CFG.val_folds = list(range(CFG.val_folds)) 

    if isinstance(CFG.num_col_names, str): # We always want this to be an iterator
        CFG.num_col_names = [CFG.num_col_names]

    if CFG.build_custom_model_type:
        validate_and_transform_config_for_custom_build_type(CFG, df)

    assert CFG.item_col_name in df.columns # Make sure item col name exists in df
    assert CFG.cat_col_name in df.columns # Make sure cat col name exists in df
    assert CFG.item_col_name != CFG.cat_col_name # Make sure these are distinct

    if CFG.target_cat_list: # Make sure all cats exist in df
        assert set(CFG.target_cat_list).issubset(set(df[CFG.cat_col_name]))

    assert list_repo_files(CFG.checkpoint) 



def validate_and_transform_config_for_custom_folds(CFG:CFG, df:pd.DataFrame):
    assert (CFG.fold_col_name in df.columns) and ptypes.is_numeric_dtype(df[CFG.fold_col_name])
    CFG.n_folds = df[CFG.fold_col_name].nunique()

    if isinstance(CFG.val_folds, int):
        assert CFG.val_folds <= CFG.n_folds
        CFG.val_folds = sorted(random.sample(df[CFG.fold_col_name].unique().tolist(), k=CFG.val_folds))
    else:
        assert set(CFG.val_folds).issubset(set(df[CFG.fold_col_name]))

    assert not CFG.group_col_name, "Can't do both GroupKFold and custom folds!"

def validate_and_transform_config_for_custom_build_type(CFG:CFG, df:pd.DataFrame):
    if CFG.build_custom_model_type == "num_metric":
        assert CFG.num_col_names
        for num_col_name in CFG.num_col_names:
            assert (num_col_name in df.columns) and ptypes.is_numeric_dtype(df[num_col_name])
