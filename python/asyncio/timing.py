# See the coresponding stackoverflow post:
# https://stackoverflow.com/a/34827291/2846140
# Author: vxgmichel
# Source: https://gist.github.com/vxgmichel/620eb3a02d97d3da9dacdc508a5d5321

import time
import asyncio
import selectors
import contextlib

class TimedSelector(selectors.DefaultSelector):
    select_time = 0.

    def reset_select_time(self):
        self.select_time = 0.

    def select(self, timeout=None):
        if timeout <= 0:
            return super().select(timeout)
        start = time.perf_counter()
        try:
            return super().select(timeout)
        finally:
            self.select_time += time.perf_counter() - start


class TimedEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        selector = TimedSelector()
        return asyncio.DefaultEventLoopPolicy._loop_factory(selector)

# Method 1: Using a context manager
@contextlib.contextmanager
def print_timing():
    asyncio.get_event_loop()._selector.reset_select_time()
    real_time = time.perf_counter()
    process_time = time.process_time()
    yield
    real_time = time.perf_counter() - real_time
    cpu_time = time.process_time() - process_time
    select_time = asyncio.get_event_loop()._selector.select_time
    other_io_time = max(0., real_time - cpu_time - select_time)
    print(f"CPU time:      {cpu_time:.3f} s")
    print(f"Select time:   {select_time:.3f} s")
    print(f"Other IO time: {other_io_time:.3f} s")
    print(f"Real time:     {real_time:.3f} s")

# Method 2: Using a decorator
def log_timing(func):
    async def wrapper(*args, **kwargs):
        asyncio.get_event_loop()._selector.reset_select_time()
        real_time = time.perf_counter()
        process_time = time.process_time()

        result = await func(*args, **kwargs)

        real_time = time.perf_counter() - real_time
        cpu_time = time.process_time() - process_time
        select_time = asyncio.get_event_loop()._selector.select_time
        other_io_time = max(0., real_time - cpu_time - select_time)
        print(f"CPU time:      {cpu_time:.3f} s")
        print(f"Select time:   {select_time:.3f} s")
        print(f"Other IO time: {other_io_time:.3f} s")
        print(f"Real time:     {real_time:.3f} s")
        return result

    return wrapper

@log_timing
async def doSomething():
    time.sleep(0.2)
    await asyncio.sleep(0.8)
    return sum(range(10**6))

# Testing

async def main():
    print("~ Correct IO management ~")
    with print_timing():
        await asyncio.sleep(1)
        sum(range(10**6))
    print()

    print("~ Incorrect IO management ~")
    with print_timing():
        time.sleep(0.2)
        await asyncio.sleep(0.8)
        sum(range(10**6))
    print()

    print("~ Correct IO management ~")
    res = await doSomething()
    print(res)
    print()


if __name__ == "__main__":
    asyncio.set_event_loop_policy(TimedEventLoopPolicy())
    asyncio.run(main())