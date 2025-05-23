from argparse import ArgumentParser
import asyncio

from redis.asyncio import Redis, BlockingConnectionPool

async def run_single_thread(redis_client, num):
    await redis_client.set(f'key_{num}', f'value_{num}')
    await redis_client.get(f'key_{num}')

async def main(concurrency):
    connection_pool = BlockingConnectionPool(max_connections=concurrency)
    redis_client = Redis(host='localhost', port=6379, db=0, connection_pool=connection_pool)
    tasks = []
    for i in range(1,000,000,000):
        tasks.append(asyncio.create_task(run_single_thread(redis_client, i)))
        if len(tasks) >= concurrency:
            await asyncio.gather(*tasks)
            tasks = []
        if i % 1000000 == 0:
            print(f'Processed {i} keys')

if __name__ == "__main__":
    parser = ArgumentParser(description='Redis Benchmarking Tool')
    parser.add_argument("-c", "--concurrency", dest="concurrency", type=int, default=10, help="Number of concurrent connections to Redis")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency))