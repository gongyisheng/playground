import asyncio
import time
from redis import asyncio as aioredis
import random
import string

# redis config
# config set appendonly yes
# config set appendfsync always

# Generate random data
data_size = 10

async def producer(num, task_id, batch_size=10):
    # Connect to Redis
    r = aioredis.Redis(host='localhost', port=6379, db=0)

    # Start the timer
    start_time = time.time()

    data = ''.join(random.choices(string.ascii_letters, k=data_size))
    print(f"Data size: {len(data)/1024} KB, batch_data_size={len(data)*batch_size/1024} KB")

    # Add data to stream
    for i in range(num//batch_size):
        pipe = r.pipeline()
        for i in range(batch_size):
            pipe.xadd(f'mystream_{task_id}', {'data': data})
        res = await pipe.execute()
        print(res)
        # print(res)

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print benchmark results
    print(f"[{task_id}] Added {num} messages with 10KB data each to Redis stream in {elapsed_time} seconds")

async def consumer(num, task_id, batch_size=10):
    await asyncio.sleep(0.1)
    # Connect to Redis
    r = aioredis.Redis(host='localhost', port=6379, db=0)

    # Start the timer
    start_time = time.time()

    # Read data from stream
    for i in range(num//batch_size):
        for j in range(batch_size):
            res = await r.xread({f'mystream_{task_id}': '0-0'}, count=batch_size)
            print(res)
            stream_data = res[0][1]
            stream_ids = []
            for item in stream_data:
                id = item[0].decode('utf-8')
                stream_ids.append(id)
            print(stream_ids, id)
            await r.xack(f'mystream_{task_id}', "default_group", *stream_ids)

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



async def main(num, concurrency, batch_size=10):
    tasks = []
    tasks += [asyncio.create_task(producer(num, i, batch_size)) for i in range(concurrency)]
    tasks += [asyncio.create_task(consumer(num, i, batch_size)) for i in range(concurrency)]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    import sys
    num = int(sys.argv[1])
    concurrency = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    asyncio.run(main(num, concurrency, batch_size))