import asyncio
import time
import uuid

from redis.asyncio import Redis, BlockingConnectionPool

MAX_CLIENT = 1000
EXPIRE = 30
STAGE_INTERVAL_MS = 30*1000
REQUEST_DELAY_MS = 30

HOST = 'cluster-n0.local'
PORT = 6379

async def run_single_thread(redis_client: Redis):
    await asyncio.sleep(REQUEST_DELAY_MS / 1000)
    id = uuid.uuid4()
    await redis_client.set(f'key_{id}', f'value_{id}', ex=EXPIRE)
    await redis_client.get(f'key_{id}')

async def main(max_client, stage_interval):
    connection_pool = BlockingConnectionPool(max_connections=max_client)
    redis_client = Redis(host=HOST, port=PORT, connection_pool=connection_pool)
    for i in range(max_client):
        st_time = time.time()
        while time.time() - st_time < stage_interval / 1000:
            tasks = [asyncio.create_task(run_single_thread(redis_client)) for j in range(max_client)]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
