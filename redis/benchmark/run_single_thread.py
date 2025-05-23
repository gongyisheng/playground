import asyncio
import time
import uuid

from redis.asyncio import Redis, BlockingConnectionPool
from prometheus_client import Counter, Histogram
from prometheus_client import start_http_server

MAX_CLIENT = 1000
EXPIRE = 30
STAGE_INTERVAL_MS = 30*1000
REQUEST_DELAY_MS = 30

HOST = 'cluster-n0.local'
PORT = 6379

async def run_single_thread(redis_client: Redis, request_count: Counter, request_latency: Histogram):
    await asyncio.sleep(REQUEST_DELAY_MS / 1000)
    id = uuid.uuid4()

    st_time = time.time()
    await redis_client.set(f'key_{id}', f'value_{id}', ex=EXPIRE)
    et_time = time.time()

    request_latency.observe(et_time - st_time, {'cmd': 'set'})
    request_count.inc()

    st_time = time.time()
    await redis_client.get(f'key_{id}')
    et_time = time.time()
    request_latency.observe(et_time - st_time, {'cmd': 'get'})

async def main():
    connection_pool = BlockingConnectionPool(max_connections=MAX_CLIENT)
    redis_client = Redis(host=HOST, port=PORT, connection_pool=connection_pool)
    request_count = Counter('redis_requests_total', 'Number of requests to Redis')
    request_latency = Histogram('redis_request_latency_seconds', 'Latency of requests to Redis')    

    for i in range(MAX_CLIENT):
        st_time = time.time()
        while time.time() - st_time < STAGE_INTERVAL_MS / 1000:
            tasks = [asyncio.create_task(run_single_thread(redis_client, request_count, request_latency)) for j in range(i)]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    start_http_server(50052)
    asyncio.run(main())
