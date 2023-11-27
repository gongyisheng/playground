import asyncio
import time

from redis import asyncio as aioredis

MAX_BACKOFF_TIME = 300000  # 300s
SOCKET_TIMEOUT = 30
GET_CONNECTION_TIMEOUT = 60
TOPIC = "test_queue"
POOL_SIZE = 16

def cpu_work():
    a = 0
    for i in range(1000000):
        a += 1

def new_redis(host):
    cpool = aioredis.BlockingConnectionPool(
        host=host,
        max_connections=5,
        timeout=GET_CONNECTION_TIMEOUT,
        socket_timeout=SOCKET_TIMEOUT,
        retry_on_error=[ConnectionError],
    )
    redis_client = aioredis.Redis(connection_pool=cpool)
    return redis_client

async def single_consume(host):
    node = new_redis(host)
    while True:
        message = await node.brpop(TOPIC, timeout=1)
        start = time.time()
        cpu_work()
        print(f"cpu work time={time.time()-start}")
    
async def main(host):
    task = [asyncio.create_task(single_consume(host)) for i in range(POOL_SIZE)]
    await asyncio.gather(*task)

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"

    asyncio.run(main(host))
