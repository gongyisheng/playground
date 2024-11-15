import asyncio
import pickle
import random

import httpx
import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
USE_MODEL = hub.load(module_url)
print("module %s loaded" % module_url)

METRICS = {}

async def send_add_vector_request(vectors, repeat=1):
    # Define the endpoint URL
    url = "http://localhost:8000/add/"
    for i in range(repeat):
        # Send a POST request
        headers = {'Content-Type': 'application/json'}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, headers=headers, json=vectors)
            # Print the response from the server
            print("Response Status Code:", response.status_code)
            print("Response JSON:", response.json())
            METRICS["add_num"] = METRICS.get("add_num", 0) + response.json()["num"]
            METRICS["add_time"] = METRICS.get("add_time", 0) + response.json()["time"]
        except Exception as e:
            print(f"Request failed: {str(e)}")

async def add_vector(texts, repeat=1, concurrency=1):
    text_embeddings = USE_MODEL(texts)
    vectors = text_embeddings.numpy().tolist()

    tasks = [asyncio.create_task(send_add_vector_request(vectors, repeat=repeat)) for _ in range(concurrency)]
    await asyncio.gather(*tasks)

    print(f"Total vectors added: {METRICS['add_num']}, Total time taken: {METRICS['add_time']}")
    print(f"Average time taken: {METRICS['add_time'] / METRICS['add_num'] * 1000} per 1k vectors")

async def run_add_test(limit=100000, repeat=10, concurrency=1):
    data_dir = "/Users/temp"
    with open(f"{data_dir}/match_phrase_rag_data.pkl", "rb") as f:
        rag_data = pickle.load(f)
    
    texts = []

    for item in rag_data:
        std_vn = item[0]
        raw_mfs = item[1]
        for raw_mf in raw_mfs:
            for i in range(100):
                if len(texts) >= limit:
                    break
                texts.append(raw_mf + f"-{i}")
    
    await add_vector(texts, repeat=repeat, concurrency=concurrency)

async def send_search_vector_request(vectors, repeat=1, topk=10):
    # Define the endpoint URL
    url = "http://localhost:8000/search/"

    for vector in vectors:
        payload = {"vector": vector, "k": topk}
        for i in range(repeat):
            # Send a POST request
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(url, json=payload)
                # Print the response from the server
                print("Response Status Code:", response.status_code)
                print("Response JSON:", response.json())
                METRICS["search_num"] = METRICS.get("search_num", 0) + 1
                METRICS["search_time"] = METRICS.get("search_time", 0) + response.json()["time"]
            except Exception as e:
                print(f"Request failed: {str(e)}")

async def search_vector(texts, repeat=1, concurrency=1, topk=10):
    text_embeddings = USE_MODEL(texts)
    vectors = text_embeddings.numpy().tolist()

    # Prepare the payload
    tasks = [asyncio.create_task(send_search_vector_request(vectors, repeat=repeat)) for _ in range(concurrency)]
    await asyncio.gather(*tasks)

    print(f"Total vectors searched: {METRICS['search_num']}, Total time taken: {METRICS['search_time']}")
    print(f"QPS: {METRICS['search_num'] / METRICS['search_time']}")

async def run_search_test(limit=10, repeat=1, concurrency=1, topk=10):
    data_dir = "/Users/temp/Downloads"
    with open(f"{data_dir}/match_phrase_rag_data.pkl", "rb") as f:
        rag_data = pickle.load(f)
    
    texts = []

    for item in rag_data:
        std_vn = item[0]
        raw_mfs = item[1]
        for raw_mf in raw_mfs:
            for i in range(100):
                texts.append(raw_mf + f"-{i}")
    
    # random choose 10 texts
    texts = random.sample(texts, limit)

    await search_vector(texts, repeat=repeat, concurrency=concurrency, topk=topk)

if __name__ == "__main__":
    # asyncio.run(run_add_test(limit=10000, repeat=50, concurrency=2))
    asyncio.run(run_search_test(limit=10, repeat=10, concurrency=10, topk=10))