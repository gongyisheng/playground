import pandas as pd

from config import Config as ml_CFG
from pipeline import run_training_pipeline


cfg = ml_CFG()

cfg.model_name = "test_model"
cfg.save_flag=True
cfg.temp_flag=False
cfg.proj_dir = "/media/hdddisk/nlp-classification"
cfg.n_folds = 5
cfg.val_folds = 2
cfg.eval_metrics = []
cfg.metric_calc_device = "cpu"
cfg.random_seed=420
cfg.subset=1.0
cfg.item_col_name = "item"
cfg.cat_col_name = "coalesced_vm_label"
cfg.pool_mode="mean"
cfg.num_col_names= None
cfg.checkpoint = "sentence-transformers/all-distilroberta-v1"
cfg.gradient_accumulation_steps = 1
cfg.max_len = 128
cfg.bs=64
cfg.return_embeddings = False
cfg.n_epochs = 10
cfg.loss="arcface"
cfg.mlflow = False
cfg.fold_col_name = "fold"
cfg.run_description = "test run"
# cfg.experiment_path = "/Data Engineering/Project Vulcan/V3 Metric Learning/Full Model Training Pipeline"

data = {
    'coalesced_vm_label': ['abc', 'def', 'abc'],
    'item': ['abc company', 'def company', 'abc company'],
    'fold': [0, 1, 2],
}

# Step 2: Create DataFrame
df = pd.DataFrame(data)

run_training_pipeline(df, cfg)