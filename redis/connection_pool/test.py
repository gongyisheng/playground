import asyncio
import logging
import random
from redis import asyncio as aioredis
import time
import uuid

from contextvars import ContextVar
request = ContextVar("request")

# add following code to redis/asyncio/client.py L474-476
# logging.info("[get conn]available connection: %s", pool.pool.qsize())
# conn = self.connection or await pool.get_connection(command_name, **options)
# await asyncio.sleep(0.005)

# add following code to redis/asyncio/client.py L486-488
# if not self.connection:
#     await pool.release(conn)
#     logging.info("[release conn]available connection: %s", pool.pool.qsize())

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

concurrent_coro_num = 24
semaphore = asyncio.Semaphore(concurrent_coro_num)

def get_log_formatter():
    formatter = logging.Formatter('%(levelname)s: [%(asctime)s][%(request)s]%(message)s')
    return formatter

def get_log_filter():
    filter = logging.Filter()
    def _filter(record):
        if not request.get(None):
            request.set(str(uuid.uuid4()).split('-')[0])
        record.request = request.get()
        return True
    filter.filter = _filter
    return filter

def setup_logger():
    logger = logging.getLogger()
    formatter = get_log_formatter()
    filter = get_log_filter()

    fh = logging.FileHandler('redis_test.log')
    fh.setFormatter(formatter)
    fh.addFilter(filter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.addFilter(filter)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    logger.setLevel(logging.DEBUG)

def cpu_work():
    start = time.time()
    logging.debug(f"cpu work start")
    sum = 0
    for i in range(500000):
        sum += i
    logging.debug(f"cpu work end")
    end = time.time()
    logging.debug(f"cpu work time: {(end - start)*1000}ms")

async def io_work():
    start = time.time()
    logging.debug(f"io work start")
    data = await node.get('foo')
    logging.debug(f"io work redis get end")
    await asyncio.sleep(0.01)
    logging.debug(f"io work end")
    end = time.time()
    logging.info(f"io work time: {(end - start)*1000}ms")

async def comb_work(round=7):
    request.set(str(uuid.uuid4()).split('-')[0])
    start = time.time()
    logging.debug(f"comb work start")
    for i in range(round):
        await io_work()
        cpu_work()
    semaphore.release()
    logging.info(f"release semaphore, value: {semaphore._value}")
    logging.debug(f"comb work end")
    end = time.time()
    logging.info(f"comb work time: {(end - start)*1000}ms")

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
                logging.info(f"acquire semaphore, value: {semaphore._value}")
                await asyncio.wait_for(semaphore.acquire(), timeout=1)
                asyncio.create_task(comb_work())
                spawn += 1
            except asyncio.TimeoutError:
                logging.warning("Pool is full")
                continue
            logging.info(f"spawn {spawn} tasks")
        while semaphore._value != concurrent_coro_num:
            await asyncio.sleep(0.1)
            continue
        end = time.time()
        logging.info(f"spawn {round} tasks, time: {(end - start)*1000}ms, each task time: {(end - start)*1000/round}ms")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
