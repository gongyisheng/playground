import asyncio
from asyncio.queues import LifoQueue

q = LifoQueue()


async def get():
    while True:
        n = await q.get()
        print(f"get {n}")
        yield n
        print(f"yield {n}")


async def put(n):
    print(f"put {n}")
    await q.put(n)


async def listen():
    async for i in get():
        print(i)


async def write():
    await put(0)
    await asyncio.sleep(1)
    await put(1)
    await asyncio.sleep(1)
    await put(2)
    await asyncio.sleep(1)
    await put(3)

asyncio.run(asyncio.wait([listen(), write()]))