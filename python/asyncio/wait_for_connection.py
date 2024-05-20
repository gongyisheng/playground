import asyncio
import time

pool = asyncio.Queue(5)


async def worker(i):
    while True:
        obj = await pool.get()
        print(f"{time.time()} worker {i} start")
        await asyncio.sleep(1)
        await pool.put(obj)
        print(f"{time.time()} worker {i} end")


async def main():
    for i in range(5):
        await pool.put(object())
    tasks = [asyncio.create_task(worker(i)) for i in range(20)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
