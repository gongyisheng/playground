import asyncio
import logging
import random
from redis import asyncio as aioredis
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
# node.set('foo', 'bar')

concurrent_coro_num = 16
semaphore = asyncio.Semaphore(concurrent_coro_num)

def setup_logger():
    logger = logging.getLogger()
    fh = logging.FileHandler('redis_test.log')
    fh.setFormatter(logging.Formatter('%(levelname)s: [%(asctime)s]%(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)

def get_log_id():
    return random.randint(0, 1000000)

def cpu_work(log_id):
    start = time.time()
    logging.info(f"[{log_id}]cpu work start")
    sum = 0
    for i in range(500000):
        sum += i
    logging.info(f"[{log_id}]cpu work end")
    end = time.time()
    logging.info(f"[{log_id}]cpu work time: {(end - start)*1000}ms")

async def io_work(log_id):
    start = time.time()
    logging.info(f"[{log_id}]io work start")
    data = await node.get('foo')
    logging.info(f"[{log_id}]io work redis get end")
    await asyncio.sleep(0.03)
    logging.info(f"[{log_id}]io work end")
    end = time.time()
    logging.info(f"[{log_id}]io work time: {(end - start)*1000}ms")

async def comb_work(log_id, round=7):
    start = time.time()
    logging.info(f"[{log_id}]comb work start")
    for i in range(round):
        await io_work(log_id)
        cpu_work(log_id)
    semaphore.release()
    logging.info(f"[{log_id}]comb work end")
    end = time.time()
    logging.info(f"[{log_id}]comb work time: {(end - start)*1000}ms")

async def main():
    setup_logger()
    fd = open('operation.txt', 'r')
    while True:
        line = fd.readline()
        if not line:
            await asyncio.sleep(0.1)
            continue
        try:
            round = int(line)
            if round <= 0:
                break
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
                logging.info("Pool is full")
                continue
            logging.info(f"spawn {spawn} tasks")
        while semaphore._value != concurrent_coro_num:
            logging.info(f"semaphore_value = {semaphore._value}")
            await asyncio.sleep(0.1)
            continue
        end = time.time()
        logging.info(f"spawn {round} tasks, time: {(end - start)*1000}ms, each task time: {(end - start)*1000/round}ms")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
