import asyncio
from redis import asyncio as aioredis

async def set_key(redis:aioredis.Redis, key, value):
    await redis.set(key, value)
    print(f'set key {key}')

async def main():
    connection_pool = aioredis.BlockingConnectionPool(host="127.0.0.1", port=8080, max_connections=200)
    redis = aioredis.Redis(connection_pool=connection_pool)
    count = 0
    while True:
        tasks = [asyncio.create_task(set_key(redis, f'key_{i}', f'value_{i}')) for i in range(2<<10)]
        await asyncio.gather(*tasks)
        print(f'done set key round {count}')
        count += 1

if __name__ == '__main__':
    asyncio.run(main())