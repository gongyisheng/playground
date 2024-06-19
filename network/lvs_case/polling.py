import asyncio
from redis import asyncio as aioredis

async def set_key(redis, key, value):
    await redis.set(key, value)

async def main():
    connection_pool = aioredis.ConnectionPool(host="127.0.0.1", port=8080, max_connections=10)
    redis = aioredis.Redis(connection_pool=connection_pool)
    tasks = [asyncio.create_task(set_key(redis, f'key_{i}', f'value_{i}')) for i in range(20)]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())