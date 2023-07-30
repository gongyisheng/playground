import asyncio
from contextlib import contextmanager

@contextmanager
def profile_and_print():
    import cProfile, pstats

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        yield
    finally:
        profiler.disable()

    pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)

def fib_seq(n):
    seq = []
    if n > 0:
        seq.extend(fib_seq(n - 1))
    seq.append(fib(n))
    return seq

@profile_and_print()
async def combined(n):
    for i in range(n):
        fib_seq(10)
        await asyncio.sleep(1)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(combined(5))
