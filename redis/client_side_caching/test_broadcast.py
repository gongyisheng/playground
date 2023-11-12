# Before running this test, you need to start redis server first
# Test Dependencies:
#   pip install pytest pytest-asyncio pytest-repeat

import asyncio
from contextvars import ContextVar
import logging
import pytest
# import pytest_asyncio
import random
import signal_state_aio as signal_state
import time
import traceback
import uuid

from redis.asyncio import Redis, BlockingConnectionPool, ConnectionError
from broadcast import CachedRedis

request = ContextVar("request")
LOG_SETUP_FLAG = False

# redis configs
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_MAX_CONNECTIONS = 5

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
    global LOG_SETUP_FLAG
    if LOG_SETUP_FLAG:
        return

    logger = logging.getLogger()
    formatter = get_log_formatter()
    filter = get_log_filter()

    fh = logging.FileHandler('cached_redis_unittest.log')
    fh.setFormatter(formatter)
    fh.addFilter(filter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.addFilter(filter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    logger.setLevel(logging.DEBUG)
    LOG_SETUP_FLAG = True

# WARNING: if func(info) raise exception, monitor coroutine will stop but no exception will be raised
async def monitor(redis, callback_func=[]):
    async with redis.monitor() as m:
        async for info in m.listen():
            for func in callback_func:
                func(info)
            logging.debug(f"monitor redis: command={info.get('command')}, time={info.get('time')}")

async def init(**kwargs):
    setup_logger()
    signal_state.ALIVE = True
    signal_state.register_exit_signal()
    pool = BlockingConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, max_connections=REDIS_MAX_CONNECTIONS)
    redis = Redis(connection_pool=pool) # You can also test decode_responses=True, it should also work
    await redis.flushdb()

    client = CachedRedis(redis, **kwargs)
    client.TASK_NAME += f"_{uuid.uuid4().hex[:16]}"
    client.HASHKEY_PREFIX = uuid.uuid4().hex[:16]
    callback_funcs = kwargs.get("listen_invalidate_callback", [])
    for func in callback_funcs:
        client.register_listen_invalidate_callback(func)

    monitor_task = asyncio.create_task(monitor(redis, kwargs.get("monitor_callback", [])))
    daemon_task = asyncio.create_task(client.run())

    await asyncio.sleep(0.5)
    return client, daemon_task, monitor_task

@pytest.mark.asyncio
async def test_get():
    GET_COUNT = 0
    def audit_get(info):
        # Expect only one GET my_key command to redis
        nonlocal GET_COUNT
        if info['command'].startswith("GET my_key"):
            GET_COUNT += 1
            assert GET_COUNT <= 1

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], monitor_callback=[audit_get])
    await client.set("my_key", "my_value")

    for i in range(5):
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            logging.error(f"error={e}, trackback={traceback.format_exc()}")
            raise e
        assert value[:8] == b"my_value"
        await asyncio.sleep(1)

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert GET_COUNT == 1

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_hget():
    HGET_COUNT = 0
    def audit_hget(info):
        # Expect only one HGET my_key my_field command to redis
        nonlocal HGET_COUNT
        if info['command'].startswith("HGET my_key my_field"):
            HGET_COUNT += 1

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], monitor_callback=[audit_hget])
    await client.hset("my_key", "my_field", "my_value")

    for i in range(5):
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.hget("my_key", "my_field")
        except Exception as e:
            logging.error(f"error={e}, trackback={traceback.format_exc()}")
            raise e
        assert value[:8] == b"my_value"
        await asyncio.sleep(1)

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert HGET_COUNT == 1

    await client._redis.close(close_connection_pool=True)

# TODO: test long prefix and many prefix
@pytest.mark.asyncio
async def test_prefix():
    cache_prefix = ["my_key", "test"]
    for i in range(62):
        cache_prefix.append(uuid.uuid4().hex[:16])

    CLIENT_TRACKING_ON_COMMAND = None
    CLIENT_TRACKING_OFF_COMMAND = None
    def audit_client_tracking(info):
        nonlocal CLIENT_TRACKING_ON_COMMAND
        nonlocal CLIENT_TRACKING_OFF_COMMAND
        # Expect only one HGET my_key my_field command to redis
        if info['command'].startswith("CLIENT TRACKING ON"):
            CLIENT_TRACKING_ON_COMMAND = info['command']
        elif info['command'].startswith("CLIENT TRACKING OFF"):
            CLIENT_TRACKING_OFF_COMMAND = info['command']

    client, daemon_task, monitor_task = await init(cache_prefix=cache_prefix, monitor_callback=[audit_client_tracking])
    await client.set("my_key", "my_value")

    for i in range(5):
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            logging.error(f"error={e}, trackback={traceback.format_exc()}")
            raise e
        assert value[:8] == b"my_value"
        await asyncio.sleep(1)

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert CLIENT_TRACKING_ON_COMMAND is not None
    assert CLIENT_TRACKING_OFF_COMMAND is not None
    for prefix in cache_prefix:
        assert prefix in set(CLIENT_TRACKING_ON_COMMAND.split(' '))
        assert prefix in set(CLIENT_TRACKING_OFF_COMMAND.split(' '))

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_synchronized_get():
    GET_COUNT = 0
    def audit_get(info):
        # Expect only one GET my_key command to redis
        nonlocal GET_COUNT
        if info['command'].startswith("GET my_key"):
            GET_COUNT += 1

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], monitor_callback=[audit_get])
    await client.set("my_key", "my_value")

    start = time.time()
    while time.time()-start <= 5:
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            logging.error(f"error={e}, trackback={traceback.format_exc()}")
            raise e
        assert value[:8] == b"my_value"

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert GET_COUNT == 1

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_synchronized_hget():
    HGET_COUNT = 0
    def audit_hget(info):
        # Expect only one HGET my_key my_field command to redis
        nonlocal HGET_COUNT
        if info['command'].startswith("HGET my_key my_field"):
            HGET_COUNT += 1

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], monitor_callback=[audit_hget])
    await client.hset("my_key", "my_field", "my_value")

    start = time.time()
    while time.time()-start <= 5:
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.hget("my_key", "my_field")
        except Exception as e:
            logging.error(f"error={e}, trackback={traceback.format_exc()}")
            raise e
        assert value[:8] == b"my_value"

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert HGET_COUNT == 1

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_short_cache_ttl():
    # This function introduces a random value in cache ttl
    # You can run this test multiple times to make sure that any value in cache ttl is working
    # pytest -k test_short_cache_ttl --count X
    CACHE_TTL = max(random.random() * 5, 0.1)
    GET_TIMESTAMPS = []

    def audit_get(info):
        nonlocal GET_TIMESTAMPS
        if info['command'].startswith("GET my_key"):
            GET_TIMESTAMPS.append(info['time'])

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], cache_ttl=CACHE_TTL, monitor_callback=[audit_get])
    await client.set("my_key", "my_value")

    start = time.time()
    while time.time()-start <= 5:
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            logging.warning(f"error={e}, trackback={traceback.format_exc()}")
            break
        assert value[:8] == b"my_value"

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    max_cache_ttl = CACHE_TTL*(1-client.cache_ttl_deviation)
    for i in range(len(GET_TIMESTAMPS)-1):
        diff = GET_TIMESTAMPS[i+1] - GET_TIMESTAMPS[i]
        assert diff >= max_cache_ttl
    assert len(GET_TIMESTAMPS) == 5//(CACHE_TTL*(1-client.cache_ttl_deviation)) + 1

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_short_health_check():
    # This function introduces a random value in health check interval
    # You can run this test multiple times to make sure that any value in cache ttl is working
    # pytest -k test_short_cache_ttl --count X
    HEALTH_CHECK_INTERVAL = max(random.random() * 5, 0.1)
    HEALTH_CHECK_TIMESTAMPS = []
    def audit_health_check(info):
        nonlocal HEALTH_CHECK_TIMESTAMPS
        if info['command'].startswith("PING"):
            HEALTH_CHECK_TIMESTAMPS.append(info['time'])

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], health_check_interval=HEALTH_CHECK_INTERVAL, monitor_callback=[audit_health_check])
    await client.set("my_key", "my_value")

    start = time.time()
    while time.time()-start <= 5:
        request.set(str(uuid.uuid4()).split('-')[0])
        try:
            value = await client.get("my_key")
        except Exception as e:
            logging.warning(f"error={e}, trackback={traceback.format_exc()}")
            raise e
        assert value[:8] == b"my_value"

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    for i in range(len(HEALTH_CHECK_TIMESTAMPS)-1):
        diff = HEALTH_CHECK_TIMESTAMPS[i+1] - HEALTH_CHECK_TIMESTAMPS[i]
        assert diff >= HEALTH_CHECK_INTERVAL
    assert len(HEALTH_CHECK_TIMESTAMPS) <= min(5//HEALTH_CHECK_INTERVAL + 1, 6)

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_concurrent_get():
    GET_COUNT = 0
    def audit_get(info):
        nonlocal GET_COUNT
        if info['command'].startswith("GET my_key"):
            GET_COUNT += 1

    ERROR = []
    async def _get():
        nonlocal ERROR
        try:
            start = time.time()
            while time.time()-start <= 5 and signal_state.ALIVE:
                request.set(str(uuid.uuid4()).split('-')[0])
                value = await client.get("my_key")
                assert value[:8] == b"my_value"
        except Exception as e:
            logging.error(e, value)
            ERROR.append(e)
            raise e
        finally:
            signal_state.ALIVE = False

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], monitor_callback=[audit_get])
    await client.set("my_key", "my_value")

    pool_num = 10
    task = [asyncio.create_task(_get()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    assert GET_COUNT == 1
    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_concurrent_hget():
    HGET_COUNT = 0
    def audit_hget(info):
        nonlocal HGET_COUNT
        if info['command'].startswith("HGET my_key my_field"):
            HGET_COUNT += 1

    ERROR = []
    async def _hget():
        nonlocal ERROR
        try:
            start = time.time()
            while time.time()-start <= 5 and signal_state.ALIVE:
                request.set(str(uuid.uuid4()).split('-')[0])
                value = await client.hget("my_key", "my_field")
                assert value[:8] == b"my_value"
        except Exception as e:
            logging.error(e, value)
            ERROR.append(e)
            raise e
        finally:
            signal_state.ALIVE = False

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], monitor_callback=[audit_hget])
    await client.hset("my_key", "my_field", "my_value")

    pool_num = 10
    task = [asyncio.create_task(_hget()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    assert HGET_COUNT == 1
    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_concurrent_hget_with_deviation():
    HGET_COUNT = 0
    HGET_SERVER_TIMESTAMP = None
    HGET_CLIENT_TIMESTAMP = None

    def audit_hget(info):
        nonlocal HGET_COUNT
        nonlocal HGET_SERVER_TIMESTAMP
        if info['command'].startswith("HGET my_key my_field"):
            HGET_COUNT += 1
            if HGET_SERVER_TIMESTAMP is None:
                HGET_SERVER_TIMESTAMP = info['time']

    ERROR = []
    async def _hget():
        nonlocal ERROR
        nonlocal HGET_CLIENT_TIMESTAMP
        try:
            start = time.time()
            if HGET_CLIENT_TIMESTAMP is None:
                HGET_CLIENT_TIMESTAMP = start
            while time.time()-start <= 5 and signal_state.ALIVE:
                request.set(str(uuid.uuid4()).split('-')[0])
                value = await client.hget("my_key", "my_field")
                assert value[:8] == b"my_value"
        except Exception as e:
            logging.error(e, value)
            ERROR.append(e)
            raise e
        finally:
            signal_state.ALIVE = False

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], hget_deviation_option={"my_key": 10}, monitor_callback=[audit_hget])
    await client.hset("my_key", "my_field", f"my_value")

    pool_num = 10
    task = [asyncio.create_task(_hget()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    assert HGET_COUNT == 1
    assert len(ERROR) == 0
    assert HGET_SERVER_TIMESTAMP is not None
    assert HGET_CLIENT_TIMESTAMP is not None
    diff = HGET_SERVER_TIMESTAMP - HGET_CLIENT_TIMESTAMP
    assert diff >= 0.01 and diff <= 10

    # If don't have hget deviation option, diff is usually less than 0.005
    # FAILED test_concurrent_hget_with_deviation[1-10] - assert 0.003918886184692383 >= 0.01
    # FAILED test_concurrent_hget_with_deviation[2-10] - assert 0.002978801727294922 >= 0.01
    # FAILED test_concurrent_hget_with_deviation[3-10] - assert 0.0022521018981933594 >= 0.01
    # FAILED test_concurrent_hget_with_deviation[4-10] - assert 0.0016889572143554688 >= 0.01
    # FAILED test_concurrent_hget_with_deviation[5-10] - assert 0.0029039382934570312 >= 0.01
    # FAILED test_concurrent_hget_with_deviation[6-10] - assert 0.003061056137084961 >= 0.01
    # FAILED test_concurrent_hget_with_deviation[7-10] - assert 0.0022199153900146484 >= 0.01
    # FAILED test_concurrent_hget_with_deviation[8-10] - assert 0.0019199848175048828 >= 0.01
    # FAILED test_concurrent_hget_with_deviation[9-10] - assert 0.0018799304962158203 >= 0.01
    # FAILED test_concurrent_hget_with_deviation[10-10] - assert 0.0030639171600341797 >= 0.01

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_noevict_get():
    NOEVICT_KEY_GET_COUNT = 0
    def audit_get(info):
        nonlocal NOEVICT_KEY_GET_COUNT
        if info['command'].startswith("GET my_key_no_evict"):
            NOEVICT_KEY_GET_COUNT += 1

    ERROR = []
    async def _get(key: str, expected_value: bytes):
        nonlocal ERROR
        try:
            start = time.time()
            while time.time()-start <= 5 and signal_state.ALIVE:
                request.set(str(uuid.uuid4()).split('-')[0])
                value = await client.get(key)
                assert value == expected_value
        except Exception as e:
            logging.error(e, value)
            ERROR.append(e)
            raise e
        finally:
            signal_state.ALIVE = False

    pool_num = 10
    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], cache_noevict_prefix=["my_key_no_evict"], cache_size=5, monitor_callback=[audit_get])
    await client.set("my_key_no_evict", "my_value_no_evict")
    for i in range(pool_num):
        await client.set(f"my_key_{i}", f"my_value_{i}")
    
    task = [asyncio.create_task(_get(f"my_key_{i}", f"my_value_{i}".encode('ascii'))) for i in range(pool_num)]
    noevict_task = asyncio.create_task(_get("my_key_no_evict", b"my_value_no_evict"))
    await asyncio.gather(daemon_task, *task, noevict_task)
    monitor_task.cancel()

    assert NOEVICT_KEY_GET_COUNT == 1
    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_noevict_hget():
    NOEVICT_KEY_GET_COUNT = 0
    def audit_hget(info):
        nonlocal NOEVICT_KEY_GET_COUNT
        if info['command'].startswith("HGET my_key_no_evict"):
            NOEVICT_KEY_GET_COUNT += 1

    ERROR = []
    async def _hget(key: str, field: str, expected_value: bytes):
        nonlocal ERROR
        try:
            start = time.time()
            while time.time()-start <= 5 and signal_state.ALIVE:
                request.set(str(uuid.uuid4()).split('-')[0])
                value = await client.hget(key, field)
                assert value == expected_value
        except Exception as e:
            logging.error(e, value)
            ERROR.append(e)
            raise e
        finally:
            signal_state.ALIVE = False

    pool_num = 10
    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], cache_noevict_prefix=["my_key_no_evict"], cache_size=5, monitor_callback=[audit_hget])
    await client.hset("my_key_no_evict", "my_field_no_evict", "my_value_no_evict")
    for i in range(pool_num):
        await client.hset(f"my_key_{i}", f"my_field_{i}", f"my_value_{i}")

    task = [asyncio.create_task(_hget(f"my_key_{i}", f"my_field_{i}", f"my_value_{i}".encode('ascii'))) for i in range(pool_num)]
    noevict_task = asyncio.create_task(_hget("my_key_no_evict", "my_field_no_evict", b"my_value_no_evict"))
    await asyncio.gather(daemon_task, *task, noevict_task)
    monitor_task.cancel()

    assert NOEVICT_KEY_GET_COUNT == 1
    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_concurrent_get_short_expire_time():
    # This function introduces a random value in cache ttl
    # You can run this test multiple times to make sure that any value in cache ttl is working
    # pytest -k test_short_cache_ttl --count X
    CACHE_TTL = max(random.random() * 5, 0.1)
    GET_TIMESTAMPS = []
    def audit_get(info):
        nonlocal GET_TIMESTAMPS
        if info['command'].startswith("GET my_key"):
            GET_TIMESTAMPS.append(info['time'])

    ERROR = []
    async def _get():
        nonlocal ERROR
        try:
            start = time.time()
            while time.time()-start <= 5 and signal_state.ALIVE:
                request.set(str(uuid.uuid4()).split('-')[0])
                value = await client.get("my_key")
                assert value[:8] == b"my_value"
        except Exception as e:
            logging.error(e, value)
            raise e
        finally:
            signal_state.ALIVE = False

    client, daemon_task, monitor_task = await init(cache_prefix=["my_key", "test"], cache_ttl=CACHE_TTL, monitor_callback=[audit_get])
    await client.set("my_key", "my_value")

    pool_num = 10
    task = [asyncio.create_task(_get()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    max_cache_ttl = CACHE_TTL*(1-client.cache_ttl_deviation)
    for i in range(len(GET_TIMESTAMPS)-1):
        diff = GET_TIMESTAMPS[i+1] - GET_TIMESTAMPS[i]
        assert diff >= max_cache_ttl
    assert len(GET_TIMESTAMPS) == 5//(CACHE_TTL*(1-client.cache_ttl_deviation)) + 1

    await client._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_concurrent_hget_short_expire_time():
    async def _hget():
        while signal_state.ALIVE:
            request.set(str(uuid.uuid4()).split('-')[0])
            try:
                value = await client.hget("my_key", "my_field")
                assert value[:8] == b"my_value"
            except Exception as e:
                logging.error(e, value)
                raise e
    pool_num = 10
    client, daemon_task = await init(cache_prefix=["my_key", "test"], cache_ttl=0.001)
    await client.hset("my_key", "my_field", "my_value")
    task = [asyncio.create_task(_hget()) for _ in range(pool_num)]
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
    client, daemon_task = await init(cache_prefix=["my_key", "test"], health_check_interval=0.001)
    await client.set("my_key", "my_value")
    task = [asyncio.create_task(_get()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)

async def test_concurrent_hget_short_health_check():
    async def _hget():
        request.set(str(uuid.uuid4()).split('-')[0])
        while signal_state.ALIVE:
            try:
                value = await client.hget("my_key", "my_field")
                assert value[:8] == b"my_value"
            except Exception as e:
                logging.error(e, value)
                raise e
    pool_num = 10
    client, daemon_task = await init(cache_prefix=["my_key", "test"], health_check_interval=0.001)
    await client.hset("my_key", "my_field", "my_value")
    task = [asyncio.create_task(_hget()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)

async def test_get_extreme_case():
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
    client, daemon_task = await init(cache_prefix=["my_key", "test"], cache_ttl=0.001, health_check_interval=0.001)
    await client.set("my_key", "my_value")
    task = [asyncio.create_task(_get()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)

async def test_hget_extreme_case():
    async def _hget():
        while signal_state.ALIVE:
            request.set(str(uuid.uuid4()).split('-')[0])
            try:
                value = await client.hget("my_key", "my_field")
                assert value[:8] == b"my_value"
            except ConnectionError as e:
                logging.error(e, value)
            except Exception as e:
                logging.error(e, value)
                raise e
    pool_num = 10
    client, daemon_task = await init(cache_prefix=["my_key", "test"], cache_ttl=0.001, health_check_interval=0.001)
    await client.hset("my_key", "my_field", "my_value")
    task = [asyncio.create_task(_hget()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)

async def test_1000_client_listen_invalidate():
    async def _get():
        client, daemon_task = await init(cache_prefix=["my_key", "test"], cache_ttl=86400)
        await client.set("my_key", "my_value")
        await asyncio.sleep(60)

    async def _set():
        pool = BlockingConnectionPool(host="localhost", port=6379, db=0, max_connections=1)
        redis = Redis(connection_pool=pool)
        await asyncio.sleep(10)
        await redis.set("my_key", "not_my_value")

    pool_num = 1000
    task = [asyncio.create_task(_get()) for _ in range(pool_num)] + [asyncio.create_task(_set())]
    await asyncio.gather(*task)

async def test_frequent_set():
    pool = BlockingConnectionPool(host="localhost", port=6379, db=0, max_connections=1)
    redis = Redis(connection_pool=pool)
    await asyncio.sleep(10)

    start = time.time()
    for i in range(1000):
        await redis.set("my_key", "not_my_value")
        logging.info(f"SET {i} times")
    end = time.time()
    logging.info(f"SET 1000 times in {end - start} seconds, qps = {int(1000 / (end - start))}")

async def test_concurrent_set():
    pool = BlockingConnectionPool(host="localhost", port=6379, db=0, max_connections=5)
    redis = Redis(connection_pool=pool)
    await asyncio.sleep(10)

    async def _set():
        start = time.time()
        for i in range(1000):
            await redis.set("my_key", "not_my_value")
            logging.info(f"SET {i} times")
        end = time.time()
        logging.info(f"SET 1000 times in {end - start} seconds, qps = {int(1000 / (end - start))}")
    pool_size = 5
    task = [asyncio.create_task(_set()) for _ in range(pool_size)]
    await asyncio.gather(*task)

if __name__ == "__main__":
    import sys

    func_name = sys.argv[1]
    try:
        profiling = sys.argv[2] == "1"
    except Exception:
        profiling = False

    if not profiling:
        asyncio.run(globals()[func_name]())
    else:
        import cProfile
        import pstats

        profiler = cProfile.Profile(timer=time.process_time)
        profiler.enable()
        asyncio.run(globals()[func_name]())
        profiler.disable()

        stream = open('profiling_output.log', 'w')
        stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
        stats.sort_stats('cumtime')
        s = stats.print_stats()
        stream.close()
    