import asyncio
from contextvars import ContextVar
import logging
import signal_state_aio as signal_state
import time
import uuid

from redis.asyncio import BlockingConnectionPool, ConnectionError
from broadcast import CachedRedis

request = ContextVar("request")

def get_log_formatter():
    formatter = logging.Formatter('%(levelname)s: [%(asctime)s][%(filename)s:%(lineno)s][%(request)s]%(message)s')
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
    signal_state.register_exit_signal()
    pool = BlockingConnectionPool(host="localhost", port=6379, db=0, max_connections=5)
    client = CachedRedis(*args, connection_pool=pool, **kwargs)
    daemon_task = asyncio.create_task(client.run())
    return client, daemon_task

async def test():
    client, daemon_task = await init()
    await client.set("my_key", "my_value")
    while signal_state.ALIVE:
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            continue
        assert value[:8] == b"my_value"
        await asyncio.sleep(1)
    await asyncio.gather(daemon_task)

async def test_prefix():
    client, daemon_task = await init(prefix=["test", "my"])
    await client.set("my_key", "my_value")
    while signal_state.ALIVE:
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            continue
        assert value[:8] == b"my_value"
        await asyncio.sleep(1)
    await asyncio.gather(daemon_task)

async def test_frequent_get():
    client, daemon_task = await init()
    await client.set("my_key", "my_value")
    while signal_state.ALIVE:
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            continue
        assert value[:8] == b"my_value"
    await asyncio.gather(daemon_task)

async def test_short_expire_time():
    client, daemon_task = await init(cache_expire_threshold=2)
    await client.set("my_key", "my_value")
    while signal_state.ALIVE:
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            continue
        assert value[:8] == b"my_value"
        await asyncio.sleep(1)
    await asyncio.gather(daemon_task)

async def test_short_health_check():
    client, daemon_task = await init(pubsub_health_check_interval=2)
    await client.set("my_key", "my_value")
    while signal_state.ALIVE:
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            continue
        assert value[:8] == b"my_value"
        await asyncio.sleep(1)
    await asyncio.gather(daemon_task)

async def test_concurrent_get():
    async def _get():
        while signal_state.ALIVE:
            request.set(str(uuid.uuid4()).split('-')[0])
            try:
                value = await client.get("my_key")
                assert value[:8] == b"my_value"
                diff = time.time() - float(value.decode('ascii')[9:])
                logging.info(f"diff: {int(diff*1000)}ms")
                assert diff < 1
            except Exception as e:
                logging.error(e, value)
                raise e
    pool_num = 10
    client, daemon_task = await init(prefix=["test", "my"])
    await client.set("my_key", f"my_value_{time.time()}")
    task = [asyncio.create_task(_get()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)

async def test_concurrent_get_short_expire_time():
    async def _get():
        while signal_state.ALIVE:
            request.set(str(uuid.uuid4()).split('-')[0])
            try:
                value = await client.get("my_key")
                assert value[:8] == b"my_value"
            except Exception as e:
                logging.error(e, value)
                raise e
    pool_num = 10
    client, daemon_task = await init(prefix=["test", "my"], cache_expire_threshold=0.001)
    await client.set("my_key", "my_value")
    task = [asyncio.create_task(_get()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)

async def test_concurrent_get_short_health_check():
    async def _get():
        request.set(str(uuid.uuid4()).split('-')[0])
        while signal_state.ALIVE:
            try:
                value = await client.get("my_key")
                assert value[:8] == b"my_value"
            except Exception as e:
                logging.error(e, value)
                raise e
    pool_num = 10
    client, daemon_task = await init(prefix=["test", "my"], pubsub_health_check_interval=0.001)
    await client.set("my_key", "my_value")
    task = [asyncio.create_task(_get()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)

async def test_extreme_case():
    async def _get():
        while signal_state.ALIVE:
            request.set(str(uuid.uuid4()).split('-')[0])
            try:
                value = await client.get("my_key")
                assert value[:8] == b"my_value"
            except ConnectionError as e:
                logging.error(e, value)
            except Exception as e:
                logging.error(e, value)
                raise e
    pool_num = 10
    client, daemon_task = await init(prefix=["test", "my"], pubsub_health_check_interval=0.001,  cache_expire_threshold=0.001)
    await client.set("my_key", "my_value")
    task = [asyncio.create_task(_get()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)

if __name__ == "__main__":
    setup_logger()
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(test())
    # loop.run_until_complete(test_prefix())
    # loop.run_until_complete(test_frequent_get())
    # loop.run_until_complete(test_short_expire_time())
    # loop.run_until_complete(test_short_health_check())
    # loop.run_until_complete(test_concurrent_get())
    # loop.run_until_complete(test_concurrent_get_short_expire_time())
    # loop.run_until_complete(test_concurrent_get_short_health_check())
    loop.run_until_complete(test_extreme_case())

    