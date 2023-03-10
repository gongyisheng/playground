import asyncio
import random
import aioredis
import time

redis_conf = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'max_connections': 5,
    'socket_timeout': 300,
    'socket_connect_timeout': 5,
    'timeout': None
}
pool = aioredis.BlockingConnectionPool(**redis_conf)
node = aioredis.Redis(connection_pool=pool)
node.set('foo', 'bar')

concurrent_coro_num = 16
semaphore = asyncio.Semaphore(concurrent_coro_num)

def get_log_id():
    return random.randint(0, 1000000)

def cpu_work(log_id):
    start = time.time()
    print(f"[{log_id}]cpu work start")
    sum = 0
    for i in range(1000000):
        sum += i
    print(f"[{log_id}]cpu work end")
    end = time.time()
    print(f"[{log_id}]cpu work time: {(end - start)*1000}ms")

async def io_work(log_id):
    start = time.time()
    print(f"[{log_id}]io work start")
    data = node.get('foo')
    print(f"[{log_id}]io work redis get end")
    await asyncio.sleep(0.03)
    print(f"[{log_id}]io work end")
    end = time.time()
    print(f"[{log_id}]io work time: {(end - start)*1000}ms")

async def comb_work(log_id, round=7):
    start = time.time()
    print(f"[{log_id}]comb work start")
    for i in range(round):
        await io_work(log_id)
        cpu_work(log_id)
    semaphore.release()
    print(f"[{log_id}]comb work end")
    end = time.time()
    print(f"[{log_id}]comb work time: {(end - start)*1000}ms")

async def main():
    fd = open('operation.txt', 'r')
    while True:
        line = fd.readline()
        if not line:
            await asyncio.sleep(0.1)
            continue
        try:
            round = int(line)
        except:
            continue
        spawn = 0
        start = time.time()
        while spawn < round:
            try:
                log_id = get_log_id()
                await asyncio.wait_for(semaphore.acquire(), timeout=1)
                asyncio.create_task(comb_work(log_id))
                spawn += 1
            except asyncio.TimeoutError:
                print("Pool is full")
                continue
            print(f"spawn {spawn} tasks")
        while semaphore._value != concurrent_coro_num:
            print(f"semaphore_value = {semaphore._value}")
            await asyncio.sleep(0.1)
            continue
        end = time.time()
        print(f"spawn {round} tasks, time: {(end - start)*1000}ms, each task time: {(end - start)*1000/round}ms")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
