import asyncio
import redis.asyncio as aioredis
import time

pool = aioredis.BlockingConnectionPool(
    host="localhost", port=6379, db=0, max_connections=1
)
node = aioredis.Redis(connection_pool=pool)


async def freq_update():
    while True:
        await node.set("my_key", f"my_value_{time.time()}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(freq_update())
