import asyncio
import time

total_time = 0


def init(pool_size):
    pool = asyncio.Semaphore(pool_size)
    return pool


async def io_work(pool, io_time, round):
    _total_time = 0
    for i in range(round):
        start = time.time()
        await pool.acquire()
        await asyncio.sleep(io_time)
        pool.release()
        end = time.time()
        _total_time += end - start
    global total_time
    total_time += _total_time
    print(f"avg time: {_total_time/round}")


async def main(pool_size, coro_num, io_time, round):
    pool = init(pool_size)
    tasks = [
        asyncio.create_task(io_work(pool, io_time, round)) for _ in range(coro_num)
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    round_num = 10000
    pool_size = 5
    coro_num = 15
    io_time = 0.001

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        main(pool_size=pool_size, coro_num=coro_num, io_time=io_time, round=round_num)
    )
    print(f"overall avg time: {total_time/(round_num*coro_num)}")
