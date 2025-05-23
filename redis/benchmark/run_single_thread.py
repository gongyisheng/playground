import asyncio
from redis.asyncio import Redis

async def run_single_thread(redis_client, num):
    await redis_client.set(f'key_{num}', f'value_{num}')
    await redis_client.get(f'key_{num}')

async def main():
    redis_client = Redis(host='localhost', port=6379, db=0)
    for i in range(1000000000):
        await run_single_thread(redis_client, i)
        if i % 1000000 == 0:
            print(f'Processed {i} keys')

if __name__ == "__main__":
    asyncio.run(main())