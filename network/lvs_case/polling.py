import asyncio
from redis import asyncio as aioredis

async def set_key(redis:aioredis.Redis, semaphore: asyncio.Semaphore, key, value):
    async with semaphore:
        await redis.set(key, value)

async def main():
    connection_pool = aioredis.BlockingConnectionPool(host="127.0.0.1", port=8080, max_connections=1)
    redis = aioredis.Redis(connection_pool=connection_pool)
    semaphore = asyncio.Semaphore(50)
    tasks = [asyncio.create_task(set_key(redis, semaphore, f'key_{i}', f'value_{i}')) for i in range(2<<10)]
    await asyncio.gather(*tasks)
    print('done set key')
    await asyncio.sleep(600)

if __name__ == '__main__':
    asyncio.run(main())