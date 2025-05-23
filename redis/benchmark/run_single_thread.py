from argparse import ArgumentParser
import asyncio
import time
import uuid

from redis.asyncio import Redis, BlockingConnectionPool

async def run_single_thread(redis_client, id):
    await redis_client.set(f'key_{id}', f'value_{id}')
    await redis_client.get(f'key_{id}')

async def main(max_client, stage_interval):
    connection_pool = BlockingConnectionPool(max_connections=max_client)
    redis_client = Redis(host='cluster-n0.local', port=6379, db=0, connection_pool=connection_pool)
    for i in range(max_client):
        st_time = time.time()
        while time.time() - st_time < stage_interval:
            tasks = [asyncio.create_task(run_single_thread(redis_client, uuid.uuid4())) for j in range(max_client)]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = ArgumentParser(description='Redis Benchmarking Tool')
    parser.add_argument("--max-client", dest="max_client", type=int, default=1000, help="Number of max connections to Redis")
    parser.add_argument("--stage-interval", dest="stage_interval", type=int, default=1, help="Interval for each stage")
    args = parser.parse_args()
    asyncio.run(main(args.max_client, args.stage_interval))
