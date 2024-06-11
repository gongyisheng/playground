import asyncio
import random
import time

from redis import asyncio as aioredis

MAX_BACKOFF_TIME = 300000  # 300s
SOCKET_TIMEOUT = 30
GET_CONNECTION_TIMEOUT = 60
TOPIC = "test_queue"
POOL_SIZE = 16

SMALL_MSG = "a" * 10
BIG_MSG = "a" * 1024 * 600

MAX_PRODUCE_NUMBER = 10000
CURR_PRODUCE_NUMBER = 0


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


async def single_produce(host):
    global CURR_PRODUCE_NUMBER, MAX_PRODUCE_NUMBER
    node = new_redis(host)
    while CURR_PRODUCE_NUMBER < MAX_PRODUCE_NUMBER:
        if random.random() <= 0.99:
            message = SMALL_MSG
        else:
            message = BIG_MSG
        await node.lpush(TOPIC, message)
        CURR_PRODUCE_NUMBER += 1


async def main(host):
    task = [asyncio.create_task(single_produce(host))]
    await asyncio.gather(*task)


if __name__ == "__main__":
    import sys

    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"

    asyncio.run(main(host))
