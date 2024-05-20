import asyncio
import redis.asyncio as aioredis

pool = aioredis.BlockingConnectionPool(
    host="localhost", port=6379, db=0, max_connections=1
)
node = aioredis.Redis(connection_pool=pool)


async def freq_update():
    for i in range(10000):
        await node.set(f"my_key_{i}", "my_value")
        if i % 1000 == 0:
            print(f"set {i} keys")

    for i in range(15, 1, -1):
        await asyncio.sleep(1)
        print(f"time to flushdb: {i}")
    # await node.flushdb()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(freq_update())
