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
group_name = "default_group"
consumer_name = "consumer_0"
min_idle_time = 5
block_timeout = 5

async def init(stream_name):
    r = aioredis.Redis(host='localhost', port=6379, db=0)
    resp = await r.xinfo_groups(stream_name)
    has_group = False
    for group in resp:
        if group['name'].decode('utf-8') == group_name:
            has_group = True
            print("Group already exists")
            break
    if not has_group:
        print("Create group")
        resp = await r.xgroup_create(stream_name, group_name, id='$', mkstream=True)
        if resp != True:
            print("Create group failed, response: ", resp)
    
    resp = await r.xinfo_consumers(stream_name, group_name)
    has_consumer = False
    for consumer in resp:
        if consumer['name'].decode('utf-8') == consumer_name:
            has_consumer = True
            print("Consumer already exists")
            break
    if not has_consumer:
        print("Create consumer")
        resp = await r.xgroup_createconsumer(stream_name, group_name, consumer_name)
        if resp != 1:
            print("Create consumer failed, response: ", resp)
    print(f"Stream {stream_name} initialized")

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
        resp = await pipe.execute()
        print("produce ", resp)

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

    await r.xclaim(f'mystream_{task_id}', group_name, consumer_name, min_idle_time, ['0-0'])

    # Read data from stream
    count = 0
    last_id = '>'
    while count < num:
        res = await r.xreadgroup(
            group_name,
            consumer_name,
            streams={f'mystream_{task_id}': last_id},
            count=batch_size,
            block=block_timeout,
        )
        if len(res) == 0:
            print("No more data to read")
            continue
        stream_data = res[0][1]
        stream_ids = []
        for item in stream_data:
            id = item[0].decode('utf-8')
            stream_ids.append(id)
        count += len(stream_ids)
        print("consume ", stream_ids, last_id)
        resp = await r.xack(f'mystream_{task_id}', group_name, *stream_ids)

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

async def cleanup():
    r = aioredis.Redis(host='localhost', port=6379, db=0)
    await r.flushdb()

async def main(num, concurrency, batch_size=10):
    #await cleanup()

    tasks = []
    tasks += [asyncio.create_task(init(f'mystream_{i}')) for i in range(concurrency)]
    await asyncio.gather(*tasks)

    tasks = []
    tasks += [asyncio.create_task(producer(num, i, batch_size)) for i in range(concurrency)]
    tasks += [asyncio.create_task(consumer(num, i, batch_size)) for i in range(concurrency)]
    await asyncio.gather(*tasks)

    # await cleanup()

if __name__ == '__main__':
    import sys
    num = int(sys.argv[1])
    concurrency = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    asyncio.run(main(num, concurrency, batch_size))