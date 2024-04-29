import asyncio
import time
from redis import asyncio as aioredis
import random
import string

# Generate random 10KB data


async def producer(num, task_id):
    # Connect to Redis
    r = aioredis.Redis(host='localhost', port=6379, db=0)

    # Start the timer
    start_time = time.time()

    data = ''.join(random.choices(string.ascii_letters, k=10240))

    # Add data to stream
    for i in range(num):
        await r.xadd(f'mystream_{task_id}', {'data': data})

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print benchmark results
    print(f"[{task_id}] Added {num} messages with 10KB data each to Redis stream in {elapsed_time} seconds")

async def consumer(num, task_id):
    # Connect to Redis
    r = aioredis.Redis(host='localhost', port=6379, db=0)

    # Start the timer
    start_time = time.time()

    # Read data from stream
    for i in range(num):
        res = await r.xread({f'mystream_{task_id}': '0-0'}, count=1)
        id = res[0][1][0][0].decode("utf-8")
        print(id)

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print benchmark results
    print(f"[{task_id}] Read {num} messages with 10KB data from Redis stream in {elapsed_time} seconds")


async def replay(num, task_id, start_id, stop_id):
    # Connect to Redis
    r = aioredis.Redis(host='localhost', port=6379, db=0)

    # Start the timer
    start_time = time.time()

    # Read data from stream
    messages = await r.xrange(f"mystream_{task_id}", start=start_id, stop=stop_id)

    for entry_id, data in messages:
        print(f"Message ID: {entry_id}, Data: {data}")

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print benchmark results
    print(f"[{task_id}] Read {num} messages with 10KB data from Redis stream in {elapsed_time} seconds")



async def main(num, concurrency):
    tasks = []
    tasks += [asyncio.create_task(producer(num, i)) for i in range(concurrency)]
    tasks += [asyncio.create_task(consumer(num, i)) for i in range(concurrency)]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    import sys
    num = int(sys.argv[1])
    concurrency = int(sys.argv[2])
    asyncio.run(main(num, concurrency))