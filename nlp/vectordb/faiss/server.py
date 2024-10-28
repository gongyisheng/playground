import os
import time
from typing import List

from fastapi import FastAPI
import faiss
import numpy as np
from pydantic import BaseModel
import uvicorn

app = FastAPI()

np.random.seed(1234) # Set seed for reproducibility
faiss.omp_set_num_threads(4) # Set number of threads for Faiss

# Assume dimensions of vectors
# amount = 10000 # 10k
# amount = 100000 # 100k
amount = 1000000 # 1M
index_path = f"/media/hdddisk/vectordb-test-data/faiss_index_{amount}.index"
save_index = None

d = 512
nlist = 100

if not os.path.exists(index_path):
    cpu_quantizer = faiss.IndexFlatL2(d)
    print(f"Index created with dimension {d} at {index_path}")
    save_index = True
else:
    cpu_quantizer = faiss.read_index(index_path)
    print(f"Index loaded from {index_path}")
    save_index = False

cpu_index = faiss.IndexIVFFlat(cpu_quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

# gpu index creation
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# Set the index to be used
index = cpu_index
print(index.is_trained)

@app.get("/ping/")
async def ping():
    return {"status": "success"}

@app.post("/add/")
async def add_vectors(vectors: List[List[float]]):
    # Convert input to NumPy array and add to index
    st_time = time.time()
    np_vectors = np.array(vectors).astype('float32')
    index.add(np_vectors)
    ed_time = time.time()
    print(f"Time taken: {ed_time - st_time}, {len(vectors)} vectors added")
    return {"status": "success", "num": len(vectors), "time": ed_time - st_time}

class SearchSchema(BaseModel):
    vector: List[float]
    k: int

@app.post("/search/")
async def search(body: SearchSchema):
    vector = body.vector
    k = body.k
    np_vector = np.array(vector).astype('float32').reshape(1, -1)
    st_time = time.time()
    distances, indices = index.search(np_vector, k)
    ed_time = time.time()
    print(f"Time taken: {ed_time - st_time}, top {k} vectors retrieved")
    return {"distances": distances.tolist(), "indices": indices.tolist(), "time": ed_time - st_time}

if __name__ == "__main__":
    # Run the FastAPI app using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    if save_index:
        faiss.write_index(index, index_path)