from argparse import ArgumentParser
import asyncio
from redis.asyncio import Redis, BlockingConnectionPool

async def run_single_thread(redis_client, num):
    await redis_client.set(f'key_{num}', f'value_{num}')
    await redis_client.get(f'key_{num}')

async def main(max_client):
    connection_pool = BlockingConnectionPool(max_connections=max_client)
    redis_client = Redis(host='cluster-n0.local', port=6379, db=0, connection_pool=connection_pool)
    for i in range(max_client):
        tasks = [asyncio.create_task(run_single_thread(redis_client, j)) for j in range(i)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = ArgumentParser(description='Redis Benchmarking Tool')
    parser.add_argument("--max-client", dest="max_client", type=int, default=1000, help="Number of max connections to Redis")
    args = parser.parse_args()
    asyncio.run(main(args.max_client))
