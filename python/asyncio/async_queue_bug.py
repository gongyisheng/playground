import asyncio
from asyncio.queues import LifoQueue

q = LifoQueue()


async def get():
    while True:
        n = await q.get()
        print(f"get {n}")
        return n


async def put(n):
    print(f"put {n}")
    await q.put(n)


async def listen():
    while True:
        await get()


async def write():
    await put(0)
    await asyncio.sleep(1)
    await put(1)
    await asyncio.sleep(1)
    await put(2)
    await asyncio.sleep(1)
    await put(3)


async def main():
    # asyncio.wait([write(), listen()])
    await asyncio.wait_for(asyncio.gather(write(), listen()), timeout=30)


if __name__ == "__main__":
    # following code works in python 3.8.10
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
    # following code doesn't work in python 3.8.10, but works in 3.10.0
    # asyncio.run(main())
