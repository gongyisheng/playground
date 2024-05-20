import asyncio
from redis import asyncio as aioredis
import time

pool = aioredis.BlockingConnectionPool(
    host="127.0.0.1", port=6379, db=0, max_connections=5
)
node = aioredis.Redis(connection_pool=pool)


async def freq_update():
    async with node.monitor() as m:
        async for command in m.listen():
            print(command)


async def get_key():
    for i in range(5):
        await node.get("my_key")
        await asyncio.sleep(1)


async def main():
    monitor_task = asyncio.create_task(freq_update())
    await asyncio.gather(asyncio.create_task(get_key()))
    print(monitor_task.cancel())


if __name__ == "__main__":
    asyncio.run(main())
