import asyncio

global_lock = asyncio.Lock()


async def raise_inside_lock():
    async with global_lock:
        print("get lock")
        raise Exception("test")


async def func_run():
    try:
        await raise_inside_lock()
    except Exception as e:
        print("catch exception:", e)


async def func_test():
    tasks = [asyncio.create_task(func_run()) for i in range(10)]
    await asyncio.gather(*tasks)


class Test:
    def __init__(self):
        pass

    async def raise_inside_lock(self):
        async with global_lock:
            print("get lock")
            raise Exception("test")


t = Test()


async def class_run():
    try:
        await t.raise_inside_lock()
    except Exception as e:
        print("catch exception:", e)


async def class_test():
    tasks = [asyncio.create_task(class_run()) for i in range(10)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(func_test())
    loop.run_until_complete(class_test())
