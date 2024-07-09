import asyncio
import os

from redis import asyncio as aioredis
import random

async def set_key(redis:aioredis.Redis, key, value):
    while True:
        try:
            await redis.set(key, value)
            break
        except Exception as e:
            print(f'error set key {key} with value {value}: {e}')

async def main():
    host = os.getenv('REDIS_HOST', '1.2.3.4')
    max_connections = os.getenv('MAX_CONNECTIONS', 200)
    connection_pool = aioredis.BlockingConnectionPool(host=host, max_connections=max_connections)
    redis = aioredis.Redis(connection_pool=connection_pool)
    count = 0
    while True:
        concurrency = max_connections - random.randint(0,1)*5
        print(f'start set key round {count}, concurrency {concurrency}')
        tasks = [asyncio.create_task(set_key(redis, f'key_{i}', f'value_{i}')) for i in range(concurrency)]
        await asyncio.gather(*tasks)
        print(f'done set key round {count}, concurrency {concurrency}')
        count += 1
        await asyncio.sleep(3)

if __name__ == '__main__':
    asyncio.run(main())