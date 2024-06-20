import asyncio
from redis import asyncio as aioredis

async def set_key(redis:aioredis.Redis, key, value):
    while True:
        try:
            await redis.set(key, value)
            break
        except Exception as e:
            print(f'error set key {key} with value {value}: {e}')

async def main():
    connection_pool = aioredis.BlockingConnectionPool(host="1.2.3.4", max_connections=200)
    redis = aioredis.Redis(connection_pool=connection_pool)
    count = 0
    while True:
        for i in range(10):
            concurrency = 200-i*20
            print(f'start set key round {count}, concurrency {concurrency}')
            tasks = [asyncio.create_task(set_key(redis, f'key_{i}', f'value_{i}')) for i in range(concurrency)]
            await asyncio.gather(*tasks)
            print(f'done set key round {count}, concurrency {concurrency}')
            count += 1
            await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.run(main())