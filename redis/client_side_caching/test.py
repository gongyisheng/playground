import asyncio
from contextvars import ContextVar
import logging
import signal_state_aio as signal_state
import uuid

from redis.asyncio import BlockingConnectionPool
from broadcast import CachedRedis

request = ContextVar("request")

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

    fh = logging.FileHandler('redis_client_side_caching.log')
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

async def init(*args, **kwargs):
    setup_logger()
    signal_state.register_exit_signal()

    pool = BlockingConnectionPool(host="localhost", port=6379, db=0, max_connections=10)
    client = CachedRedis(*args, connection_pool=pool, **kwargs)
    asyncio.create_task(client.run())
    return client

async def test():
    client = await init()
    await client.set("my_key", "my_value")
    for i in range(50):
        logging.info(await client.get("my_key"))
        await asyncio.sleep(1)
        if signal_state.ALIVE == False:
            break
    await client.stop()

async def test_prefix():
    client = await init(prefix=["test", "my"])
    await client.set("my_key", "my_value")
    for i in range(120):
        logging.info(await client.get("my_key"))
        await asyncio.sleep(1)
        if signal_state.ALIVE == False:
            break
    await client.stop()

async def test_short_expire_time():
    client = await init(expire_threshold=2)
    await client.set("my_key", "my_value")
    for i in range(50):
        logging.info(await client.get("my_key"))
        await asyncio.sleep(1)
        if signal_state.ALIVE == False:
            break
    await client.stop()

async def test_short_check_health():
    client = await init(check_health_interval=2)
    await client.set("my_key", "my_value")
    for i in range(50):
        logging.info(await client.get("my_key"))
        await asyncio.sleep(1)
        if signal_state.ALIVE == False:
            break
    await client.stop()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test())
    # loop.run_until_complete(test_prefix())
    # loop.run_until_complete(test_short_expire_time())
    # loop.run_until_complete(test_short_check_health())