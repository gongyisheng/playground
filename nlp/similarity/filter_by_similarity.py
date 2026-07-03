import faiss
import pandas as pd
import random
import tensorflow_hub as hub
from tqdm import tqdm

# build faiss index
d = 512
nlist = 100
m = 16
n_bits = 8
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def calc_vector(texts):
    embeddings = embed(texts)
    return embeddings.numpy()

df = pd.read_parquet("/data/yisheng/useless_classifier/neg_employee_related.parquet").reset_index()
raw_texts = df["item"].tolist()

sample_texts = random.sample(raw_texts, min(100, len(raw_texts)))
embeddings = calc_vector(sample_texts)
index.train(embeddings)

skip_count = 0
selected_index = []
for i in tqdm(df.index):
    match_str = df.loc[i, "item"]
    embedding = calc_vector([match_str])
    need_skip = False
    if i != 0:
        D, I = index.search(embedding, 5)
        if D[0][0] > 0.98:
            need_skip = True
    if not need_skip:
        index.add(embedding)
        selected_index.append(i)
    else:
        skip_count += 1
print(f"skip_count={skip_count}")

df_selected = df.loc[selected_index]
print(f"selected_count={len(df_selected)}")
