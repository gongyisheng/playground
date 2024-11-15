import numpy as np
import pickle
import faiss
import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
USE_MODEL = hub.load(module_url)
print("module %s loaded" % module_url)

d = 512                     # Dimensionality of the vectors
amount = 10000              # Number of database vectors
nlist = 100                 # Number of clusters

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

with open("/media/hdddisk/spendhound-data/llm_rag/match_phrase_rag_data.pkl", "rb") as f:
    _data = pickle.load(f)
training_data = []

for item in _data:
    std_vn = item[0]
    raw_mfs = item[1]
    for raw_mf in raw_mfs:
        for i in range(600):
            if len(training_data) >= amount:
                break
            training_data.append(raw_mf + f"-{i}")

vectors = USE_MODEL(training_data).numpy()
vectors = vectors.astype('float32')

index.train(vectors)
index.add(vectors)

print(index.ntotal)
print(index.is_trained)
faiss.write_index(index, f"/media/hdddisk/vectordb-test-data/faiss_index_{amount}_trained.index")

index = faiss.read_index(f"/media/hdddisk/vectordb-test-data/faiss_index_{amount}_trained.index")
print(index.is_trained)