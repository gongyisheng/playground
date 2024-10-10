import pandas as pd

from inference import run_inference

pdf = pd.DataFrame({
    'item': ['abc company', 'def company', 'abc company'],
})

pdf = run_inference(
    pdf,
    "test_model__sentence-transformers_all-distilroberta-v1__2024_10_10_19_50__1",
    proj_directory="/media/hdddisk/nlp-classification",
    item_col_name="item",
    max_len=128,
    bs=64,
    proba="proba_max",
    device="auto",
    progress_bars=True,
    custom_model_type="metric"
)

print(pdf)