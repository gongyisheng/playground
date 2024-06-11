import asyncio
import random
import time

# This will not cause race condition, because only one coroutine will be able to execute at a time.

cache = {}


async def get(key):
    return cache.get(key, None)


async def set(key, value):
    cache[key] = value


async def background_get(key, task_id):
    while True:
        await asyncio.sleep(random.randint(0, 1000) / 1000)
        val = await get(key)
        print(f"Get[{task_id}]: key={key}, val={val}")


async def background_set(key):
    while True:
        await asyncio.sleep(1)
        value = time.time()
        await set(key, value)
        print(f"Set: key={key}, val={value}")


async def main():
    await asyncio.gather(
        *[background_get("key", i) for i in range(100)], background_set("key")
    )


if __name__ == "__main__":
    asyncio.run(main())
