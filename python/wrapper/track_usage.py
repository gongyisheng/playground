import asyncio
import time
import inspect
from functools import wraps

usage_data = {}


def track_count(metric):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _metric = metric + "_count"
            usage_data[_metric] = usage_data.get(_metric, 0) + 1
            print(f"record count: _metric={_metric}, data={usage_data[_metric]}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def track_time(metric):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _metric = metric + "_time"
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            usage_data[_metric] = usage_data.get(_metric, 0) + (end - start) * 1000
            print(f"record time: _metric={_metric}, data={usage_data[_metric]}")
            return result

        async def async_wrapper(*args, **kwargs):
            _metric = metric + "_time"
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
            usage_data[_metric] = usage_data.get(_metric, 0) + (end - start) * 1000
            print(f"record time: _metric={_metric}, data={usage_data[_metric]}")
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        elif asyncio.iscoroutine(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


@track_count("IO")
@track_time("IO")
def IO(wait=1):
    time.sleep(wait)
    pass


@track_count("IO")
@track_time("IO")
async def async_IO(wait=1):
    await asyncio.sleep(wait)
    pass


def display_usage():
    print(usage_data)


def main():
    for i in range(5):
        IO(i)


async def async_main():
    for i in range(5):
        await async_IO(i)


async def async_test():
    await async_main()
    display_usage()


class Test:
    def __init__(self):
        pass

    # avoid track time first
    @track_count("IO")
    @track_time("IO")
    async def IO(self, wait=1):
        await asyncio.sleep(wait)
        return 1, 2, 3


async def async_test2():
    test = Test()
    print(inspect.iscoroutinefunction(test.IO))
    for i in range(5):
        a, b, c = await test.IO(i)
    display_usage()


if __name__ == "__main__":
    # asyncio.run(async_test())
    asyncio.run(async_test2())
