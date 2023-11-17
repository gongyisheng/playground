# Before running this test, you need to start redis server first
#    redis-server --maxclients 65535
#    ulimit -n 65535 (avoid OSError: [Errno 24] Too many open files)
#    redis configs
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_MAX_CONNECTIONS = 5
# Test Dependencies:
#    pip install pytest pytest-asyncio pytest-repeat
# Test command:
#    pytest
#    pytest -v -s (show print)
#    pytest -k <test function name>
#    pytest -k <test function name> --count 10 (repeat 10 times)
#    pytest -k <test function name> -s (show print)
#    python <test function name> run test function directly, usually for demo
#    python <test function name> run test function directly and profiling it

import asyncio
from contextvars import ContextVar
from functools import partial
import logging
import pytest
import random
import signal_state_aio as signal_state
import time
from typing import Optional, Union
import uuid

from redis.asyncio import Redis, BlockingConnectionPool, ConnectionError
from broadcast import CachedRedis

request = ContextVar("request")
LOG_SETUP_FLAG = False


def get_log_formatter():
    formatter = logging.Formatter(
        "%(levelname)s: [%(asctime)s][%(filename)s:%(lineno)s][%(request)s]%(message)s"
    )
    return formatter


def get_log_filter():
    filter = logging.Filter()

    def _filter(record):
        if not request.get(None):
            request.set(str(uuid.uuid4()).split("-")[0])
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

    fh = logging.FileHandler("cached_redis_unittest.log")
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
            logging.debug(
                f"monitor redis: command={info.get('command')}, time={info.get('time')}"
            )


async def kill_client(redis: Redis, client_id: int):
    await redis.client_kill(client_id)


async def init(**kwargs):
    setup_logger()
    signal_state.ALIVE = True
    signal_state.register_exit_signal()
    pool = BlockingConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        max_connections=REDIS_MAX_CONNECTIONS,
    )
    redis = Redis(
        connection_pool=pool
    )  # You can also test decode_responses=True, it should also work
    await redis.flushdb()

    client = CachedRedis(redis, **kwargs)
    client.TASK_NAME += f"_{uuid.uuid4().hex[:16]}"
    client.HASHKEY_PREFIX = uuid.uuid4().hex[:16]
    callback_funcs = kwargs.get("listen_invalidate_callback", [])
    for func in callback_funcs:
        client.register_listen_invalidate_callback(func)

    monitor_task = asyncio.create_task(
        monitor(redis, kwargs.get("monitor_callback", []))
    )
    daemon_task = asyncio.create_task(client.run())

    await asyncio.sleep(0.5)
    return client, daemon_task, monitor_task


# Utils get/hget functions for test
async def _synchronized_get(
    client: CachedRedis,
    key: str,
    expected_value: str,
    error_return: Optional[list] = None,
    raise_error: bool = True,
    sleep: Optional[int] = None,
    duration: int = 5,
):
    try:
        start = time.time()
        expected_value = expected_value.encode("ascii")
        while time.time() - start <= duration and signal_state.ALIVE:
            request.set(str(uuid.uuid4()).split("-")[0])
            value = await client.get(key)
            assert value == expected_value
            if sleep is not None:
                await asyncio.sleep(sleep)
    except Exception as e:
        logging.error(
            f"Got error during get test: key={key}, expected_value={expected_value}, actual_value={value}",
            exc_info=True,
        )
        if error_return is not None:
            error_return.append((time.time(), e))
        if raise_error:
            raise e
    finally:
        signal_state.ALIVE = False


async def _synchronized_hget(
    client: CachedRedis,
    key: str,
    field: str,
    expected_value: str,
    error_return: Optional[list] = None,
    raise_error: bool = True,
    sleep: Optional[int] = None,
    duration: int = 5,
):
    try:
        start = time.time()
        expected_value = expected_value.encode("ascii")
        while time.time() - start <= duration and signal_state.ALIVE:
            request.set(str(uuid.uuid4()).split("-")[0])
            value = await client.hget(key, field)
            assert value == expected_value
            if sleep is not None:
                await asyncio.sleep(sleep)
    except Exception as e:
        logging.error(
            f"Got error during hget test: key={key}, field={field}, expected_value={expected_value}, actual_value={value}",
            exc_info=True,
        )
        if error_return is not None:
            error_return.append((time.time(), e))
        if raise_error:
            raise e
    finally:
        signal_state.ALIVE = False


# Monitor audit functions
def _audit(
    info: dict,
    data_return: Optional[list] = None,
    keyword: Union[set, str, None] = None,
):
    if data_return is None or keyword is None:
        return
    if isinstance(keyword, str):
        keyword = {keyword}
    for k in keyword:
        if info["command"].startswith(k):
            data_return.append(
                {
                    "command": info["command"],
                    "time": info["time"],
                }
            )
            break


CLIENT_TRACKING_AUDIT_KEYWORDS = {
    "CLIENT TRACKING ON",
    "CLIENT TRACKING OFF",
    "SUBSCRIBE __redis__:invalidate",
    "UNSUBSCRIBE __redis__:invalidate",
}


async def demo():
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
    )
    await client.set("my_key", "my_value")
    await _synchronized_get(client, "my_key", "my_value", sleep=1, duration=60)

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_get():
    AUDIT_DATA_RETURN = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        monitor_callback=[
            partial(_audit, data_return=AUDIT_DATA_RETURN, keyword="GET my_key")
        ],
    )
    await client.set("my_key", "my_value")
    await _synchronized_get(client, "my_key", "my_value", sleep=1)

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_hget():
    AUDIT_DATA_RETURN = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        monitor_callback=[
            partial(
                _audit, data_return=AUDIT_DATA_RETURN, keyword="HGET my_key my_field"
            )
        ],
    )
    await client.hset("my_key", "my_field", "my_value")
    await _synchronized_hget(client, "my_key", "my_field", "my_value", sleep=1)

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1

    await client._redis.close(close_connection_pool=True)


# TODO: test long prefix and many prefix
@pytest.mark.asyncio
async def test_prefix():
    AUDIT_RETURN = []
    cache_prefix = ["my_key", "test"]
    for i in range(62):
        cache_prefix.append(uuid.uuid4().hex[:16])

    client, daemon_task, monitor_task = await init(
        cache_prefix=cache_prefix,
        monitor_callback=[
            partial(
                _audit,
                data_return=AUDIT_RETURN,
                keywords=CLIENT_TRACKING_AUDIT_KEYWORDS,
            )
        ],
    )
    await client.set("my_key", "my_value")
    await _synchronized_get(client, "my_key", "my_value", sleep=1)

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert len(AUDIT_RETURN) == 4
    CLIENT_TRACKING_ON_COMMANDS = [
        d["command"]
        for d in AUDIT_RETURN
        if d["command"].startswith("CLIENT TRACKING ON")
    ]
    CLIENT_TRACKING_OFF_COMMANDS = [
        d["command"]
        for d in AUDIT_RETURN
        if d["command"].startswith("CLIENT TRACKING OFF")
    ]
    assert len(CLIENT_TRACKING_ON_COMMANDS) == 1
    assert len(CLIENT_TRACKING_OFF_COMMANDS) == 1
    for prefix in cache_prefix:
        assert prefix in set(CLIENT_TRACKING_ON_COMMANDS[0].split(" "))
        assert prefix in set(CLIENT_TRACKING_OFF_COMMANDS[0].split(" "))

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_synchronized_get():
    AUDIT_RETURN = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        monitor_callback=[
            partial(_audit, data_return=AUDIT_RETURN, keyword="GET my_key")
        ],
    )
    await client.set("my_key", "my_value")
    await _synchronized_get(client, "my_key", "my_value")

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert len(AUDIT_RETURN) == 1

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_synchronized_hget():
    AUDIT_RETURN = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        monitor_callback=[
            partial(_audit, data_return=AUDIT_RETURN, keyword="HGET my_key my_field")
        ],
    )
    await client.hset("my_key", "my_field", "my_value")
    await _synchronized_hget(client, "my_key", "my_field", "my_value")

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    assert len(AUDIT_RETURN) == 1

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_short_cache_ttl():
    # This function introduces a random value in cache ttl
    # You can run this test multiple times to make sure that any value in cache ttl is working
    # pytest -k test_short_cache_ttl --count X
    CACHE_TTL = max(random.random() * 5, 0.1)
    AUDIT_RETURN = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        cache_ttl=CACHE_TTL,
        monitor_callback=[
            partial(_audit, data_return=AUDIT_RETURN, keyword="GET my_key")
        ],
    )
    await client.set("my_key", "my_value")
    await _synchronized_get(client, "my_key", "my_value")

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    max_cache_ttl = CACHE_TTL * (1 - client.cache_ttl_deviation)
    for i in range(len(AUDIT_RETURN) - 1):
        diff = AUDIT_RETURN[i + 1]["time"] - AUDIT_RETURN[i]["time"]
        assert diff >= max_cache_ttl
    expected_count = 5 // (CACHE_TTL * (1 - client.cache_ttl_deviation)) + 1
    assert len(AUDIT_RETURN) in [expected_count - 1, expected_count]

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_short_health_check():
    # This function introduces a random value in health check interval
    # You can run this test multiple times to make sure that any value in cache ttl is working
    # pytest -k test_short_health_check --count X
    HEALTH_CHECK_INTERVAL = max(random.random() * 5, 0.1)
    AUDIT_RETURN = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        health_check_interval=HEALTH_CHECK_INTERVAL,
        monitor_callback=[partial(_audit, data_return=AUDIT_RETURN, keyword="PING")],
    )
    await client.set("my_key", "my_value")
    await _synchronized_get(client, "my_key", "my_value")

    signal_state.ALIVE = False
    await asyncio.gather(daemon_task)
    monitor_task.cancel()

    for i in range(len(AUDIT_RETURN) - 1):
        diff = AUDIT_RETURN[i + 1]["time"] - AUDIT_RETURN[i]["time"]
        assert diff >= HEALTH_CHECK_INTERVAL
    assert len(AUDIT_RETURN) <= min(5 // HEALTH_CHECK_INTERVAL + 1, 6)

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_concurrent_get():
    AUDIT_RETURN = []
    ERROR = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        monitor_callback=[
            partial(_audit, data_return=AUDIT_RETURN, keyword="GET my_key")
        ],
    )
    await client.set("my_key", "my_value")

    pool_num = 10
    task = [
        asyncio.create_task(
            _synchronized_get(client, "my_key", "my_value", error_return=ERROR)
        )
        for _ in range(pool_num)
    ]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    assert len(AUDIT_RETURN) == 1
    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_concurrent_hget():
    AUDIT_RETURN = []
    ERROR = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        monitor_callback=[
            partial(_audit, data_return=AUDIT_RETURN, keyword="HGET my_key my_field")
        ],
    )
    await client.hset("my_key", "my_field", "my_value")

    pool_num = 10
    task = [
        asyncio.create_task(
            _synchronized_hget(
                client, "my_key", "my_field", "my_value", error_return=ERROR
            )
        )
        for _ in range(pool_num)
    ]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    assert len(AUDIT_RETURN) == 1
    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_concurrent_hget_with_deviation():
    HGET_CLIENT_TIMESTAMP = None
    AUDIT_RETURN = []
    ERROR = []

    async def _hget():
        nonlocal ERROR
        nonlocal HGET_CLIENT_TIMESTAMP
        start = time.time()
        if HGET_CLIENT_TIMESTAMP is None:
            HGET_CLIENT_TIMESTAMP = start
        await _synchronized_hget(
            client, "my_key", "my_field", "my_value", error_return=ERROR
        )

    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        hget_deviation_option={"my_key": 10},
        monitor_callback=[
            partial(_audit, data_return=AUDIT_RETURN, keyword="HGET my_key my_field")
        ],
    )
    await client.hset("my_key", "my_field", f"my_value")

    pool_num = 10
    task = [asyncio.create_task(_hget()) for _ in range(pool_num)]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    assert len(AUDIT_RETURN) == 1
    assert len(ERROR) == 0
    HGET_SERVER_TIMESTAMP = AUDIT_RETURN[0]["time"]
    assert HGET_SERVER_TIMESTAMP is not None
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
    AUDIT_RETURN = []
    ERROR = []
    pool_num = 10
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        cache_noevict_prefix=["my_key_no_evict"],
        cache_size=5,
        monitor_callback=[
            partial(_audit, data_return=AUDIT_RETURN, keyword="GET my_key_no_evict")
        ],
    )
    await client.set("my_key_no_evict", "my_value_no_evict")
    for i in range(pool_num):
        await client.set(f"my_key_{i}", f"my_value_{i}")

    task = [
        asyncio.create_task(
            _synchronized_get(
                client, f"my_key_{i}", f"my_value_{i}", error_return=ERROR
            )
        )
        for i in range(pool_num)
    ]
    noevict_task = asyncio.create_task(
        _synchronized_get(
            client, "my_key_no_evict", "my_value_no_evict", error_return=ERROR
        )
    )
    await asyncio.gather(daemon_task, *task, noevict_task)
    monitor_task.cancel()

    assert len(AUDIT_RETURN) == 1
    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_noevict_hget():
    AUDIT_RETURN = []
    ERROR = []

    pool_num = 10
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        cache_noevict_prefix=["my_key_no_evict"],
        cache_size=5,
        monitor_callback=[
            partial(_audit, data_return=AUDIT_RETURN, keyword="HGET my_key_no_evict")
        ],
    )
    for i in range(pool_num):
        await client.hset(
            "my_key_no_evict", f"my_field_no_evict_{i}", f"my_value_no_evict_{i}"
        )
        await client.hset(f"my_key_{i}", f"my_field_{i}", f"my_value_{i}")

    task = [
        asyncio.create_task(
            _synchronized_hget(
                client,
                f"my_key_{i}",
                f"my_field_{i}",
                f"my_value_{i}",
                error_return=ERROR,
            )
        )
        for i in range(pool_num)
    ]
    noevict_task = [
        asyncio.create_task(
            _synchronized_hget(
                client,
                "my_key_no_evict",
                f"my_field_no_evict_{i}",
                f"my_value_no_evict_{i}",
                error_return=ERROR,
            )
        )
        for i in range(pool_num)
    ]
    await asyncio.gather(daemon_task, *task, *noevict_task)
    monitor_task.cancel()

    FIELD_COUNT = {}
    for data in AUDIT_RETURN:
        field = data["command"].split(" ")[2]
        FIELD_COUNT[field] = FIELD_COUNT.get(field, 0) + 1
    for field in FIELD_COUNT:
        assert FIELD_COUNT[field] == 1
    assert len(FIELD_COUNT) == pool_num
    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_concurrent_get_short_expire_time():
    # This function introduces a random value in cache ttl
    # You can run this test multiple times to make sure that any value in cache ttl is working
    # pytest -k test_concurrent_get_short_expire_time --count X
    CACHE_TTL = max(random.random() * 5, 0.1)
    AUDIT_RETURN = []
    ERROR = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        cache_ttl=CACHE_TTL,
        monitor_callback=[
            partial(_audit, data_return=AUDIT_RETURN, keyword="GET my_key")
        ],
    )
    await client.set("my_key", "my_value")

    pool_num = 10
    task = [
        asyncio.create_task(
            _synchronized_get(client, "my_key", "my_value", error_return=ERROR)
        )
        for _ in range(pool_num)
    ]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    max_cache_ttl = CACHE_TTL * (1 - client.cache_ttl_deviation)
    for i in range(len(AUDIT_RETURN) - 1):
        diff = AUDIT_RETURN[i + 1]["time"] - AUDIT_RETURN[i]["time"]
        assert diff >= max_cache_ttl
    expected_count = 5 // (CACHE_TTL * (1 - client.cache_ttl_deviation)) + 1
    assert len(AUDIT_RETURN) in [expected_count - 1, expected_count]

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_concurrent_hget_short_expire_time():
    # This function introduces a random value in cache ttl
    # You can run this test multiple times to make sure that any value in cache ttl is working
    # pytest -k test_concurrent_hget_short_expire_time --count X
    CACHE_TTL = max(random.random() * 5, 0.1)
    HGET_TIMESTAMPS = {}

    def audit_hget(info):
        nonlocal HGET_TIMESTAMPS
        if info["command"].startswith("HGET my_key my_field"):
            field = info["command"].split(" ")[2]
            if field not in HGET_TIMESTAMPS:
                HGET_TIMESTAMPS[field] = []
            HGET_TIMESTAMPS[field].append(info["time"])

    ERROR = []
    pool_num = 10
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        cache_ttl=CACHE_TTL,
        monitor_callback=[audit_hget],
    )
    for i in range(pool_num):
        await client.hset("my_key", f"my_field_{i}", f"my_value_{i}")

    task = [
        asyncio.create_task(
            _synchronized_hget(
                client, "my_key", f"my_field_{i}", f"my_value_{i}", error_return=ERROR
            )
        )
        for i in range(pool_num)
    ]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    max_cache_ttl = CACHE_TTL * (1 - client.cache_ttl_deviation)
    for field in HGET_TIMESTAMPS:
        for i in range(0, len(HGET_TIMESTAMPS[field]) - 1):
            diff = HGET_TIMESTAMPS[field][i + 1] - HGET_TIMESTAMPS[field][i]
            assert diff >= max_cache_ttl
        expected_count = 5 // (CACHE_TTL * (1 - client.cache_ttl_deviation)) + 1
        assert len(HGET_TIMESTAMPS[field]) in [expected_count - 1, expected_count]
    assert len(HGET_TIMESTAMPS) == 10
    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_concurrent_short_health_check():
    # This function introduces a random value in cache ttl
    # You can run this test multiple times to make sure that any value in cache ttl is working
    # pytest -k test_concurrent_short_health_check --count X
    HEALTH_CHECK_INTERVAL = max(random.random() * 5, 0.1)
    PING_AUDIT_RETURN = []
    GET_AUDIT_RETURN = []
    ERROR = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"],
        health_check_interval=HEALTH_CHECK_INTERVAL,
        monitor_callback=[
            partial(_audit, data_return=PING_AUDIT_RETURN, keyword="PING"),
            partial(_audit, data_return=GET_AUDIT_RETURN, keyword="GET my_key"),
        ],
    )
    await client.set("my_key", "my_value")

    pool_num = 10
    task = [
        asyncio.create_task(
            _synchronized_get(client, "my_key", "my_value", error_return=ERROR)
        )
        for _ in range(pool_num)
    ]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    assert len(ERROR) == 0
    assert len(GET_AUDIT_RETURN) == 1
    for i in range(0, len(PING_AUDIT_RETURN) - 1):
        assert (
            PING_AUDIT_RETURN[i + 1]["time"] - PING_AUDIT_RETURN[i]["time"]
            >= HEALTH_CHECK_INTERVAL
        )

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_get_extreme_case():
    ERROR = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"], cache_ttl=0.001, health_check_interval=0.001
    )
    await client.set("my_key", "my_value")

    pool_num = 10
    task = [
        asyncio.create_task(
            _synchronized_get(client, "my_key", "my_value", error_return=ERROR)
        )
        for _ in range(pool_num)
    ]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_hget_extreme_case():
    ERROR = []
    client, daemon_task, monitor_task = await init(
        cache_prefix=["my_key", "test"], cache_ttl=0.001, health_check_interval=0.001
    )
    await client.hset("my_key", "my_field", "my_value")

    pool_num = 10
    task = [
        asyncio.create_task(
            _synchronized_hget(
                client, "my_key", "my_field", "my_value", error_return=ERROR
            )
        )
        for _ in range(pool_num)
    ]
    await asyncio.gather(daemon_task, *task)
    monitor_task.cancel()

    assert len(ERROR) == 0

    await client._redis.close(close_connection_pool=True)


async def start_1000_clients():
    signal_state.register_exit_signal()
    pool_num = 1000

    async def _get():
        pool = BlockingConnectionPool(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, max_connections=5
        )
        redis = Redis(connection_pool=pool)

        client = CachedRedis(redis, cache_prefix=["my_key", "test"])
        client.TASK_NAME += f"_{uuid.uuid4().hex[:16]}"
        client.HASHKEY_PREFIX = uuid.uuid4().hex[:16]

        await client.run()

    get_task = [asyncio.create_task(_get()) for _ in range(pool_num)]
    await asyncio.sleep(20)

    signal_state.ALIVE = False
    await asyncio.gather(*get_task)


@pytest.mark.asyncio
async def test_1000_client_listen_invalidate_once():
    pool = BlockingConnectionPool(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, max_connections=1
    )
    redis = Redis(connection_pool=pool)
    await redis.flushdb()

    start = time.time()
    await redis.set("my_key", "not_my_value")
    end = time.time()

    print(f"SET 1 time cost: {(end-start)*1000} ms")

    proc = await asyncio.create_subprocess_exec(
        "python", "test_broadcast.py", "start_1000_clients"
    )
    proc_task = asyncio.create_task(proc.communicate())
    await asyncio.sleep(2)

    start = time.time()
    await redis.set("my_key", "not_my_value")
    end = time.time()

    assert end - start < 1
    print(
        f"SET 1 time and send invalidate message to 1000 clients cost: {(end-start)*1000} ms"
    )

    await redis.close(close_connection_pool=True)
    try:
        proc.kill()
        proc_task.cancel()
    except Exception as e:
        pass


@pytest.mark.asyncio
async def test_1000_client_listen_invalidate_multi():
    pool = BlockingConnectionPool(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, max_connections=1
    )
    redis = Redis(connection_pool=pool)
    await redis.flushdb()

    start = time.time()
    for i in range(1000):
        await redis.set("my_key", "not_my_value")
    end = time.time()

    print(f"SET 1000 time cost: avg = {(end-start)} ms, qps = {int(1000/(end-start))}")

    proc = await asyncio.create_subprocess_exec(
        "python", "test_broadcast.py", "start_1000_clients"
    )
    proc_task = asyncio.create_task(proc.communicate())
    await asyncio.sleep(2)

    start = time.time()
    for i in range(1000):
        await redis.set("my_key", "not_my_value")
    end = time.time()

    avg_rumtime = end - start
    qps = int(1000 / avg_rumtime)
    assert end - start < 1000

    print(
        f"SET 1000 time and send invalidate message to 1000 clients cost: avg = {(avg_rumtime)} ms, qps = {qps}"
    )

    await redis.close(close_connection_pool=True)
    try:
        proc.kill()
        proc_task.cancel()
    except Exception as e:
        pass


@pytest.mark.asyncio
async def test_1000_client_listen_invalidate_concurrent():
    pool = BlockingConnectionPool(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, max_connections=10
    )
    redis = Redis(connection_pool=pool)
    await redis.flushdb()

    RUMTIME = []

    async def _set():
        start = time.time()
        for i in range(1000):
            await redis.set("my_key", "not_my_value")
        end = time.time()
        RUMTIME.append(end - start)

    pool_size = 10

    task = [asyncio.create_task(_set()) for _ in range(pool_size)]
    await asyncio.gather(*task)

    avg_runtime = sum(RUMTIME) / len(RUMTIME)
    qps = int(1000 / (sum(RUMTIME) / len(RUMTIME)))
    print(f"SET 1000 time concurrently cost: avg = {avg_runtime} ms, qps = {qps}")
    RUMTIME.clear()

    proc = await asyncio.create_subprocess_exec(
        "python", "test_broadcast.py", "start_1000_clients"
    )
    proc_task = asyncio.create_task(proc.communicate())
    await asyncio.sleep(2)

    task = [asyncio.create_task(_set()) for _ in range(pool_size)]
    await asyncio.gather(*task)

    avg_runtime = sum(RUMTIME) / len(RUMTIME)
    qps = int(1000 / (sum(RUMTIME) / len(RUMTIME)))
    assert avg_runtime < 1000
    print(
        f"SET 1000 time concurrently and send invalidate message to 1000 clients cost: avg = {avg_runtime} ms, qps = {qps}"
    )

    await redis.close(close_connection_pool=True)
    try:
        proc.kill()
        proc_task.cancel()
    except Exception as e:
        pass


@pytest.mark.asyncio
async def test_synchronized_get_close_connection_single():
    pass


@pytest.mark.asyncio
async def test_concurrent_get_close_connection_single():
    pass


@pytest.mark.asyncio
async def test_synchronized_get_close_connection_multi():
    pass


@pytest.mark.asyncio
async def test_concurrent_get_close_connection_multi():
    pass


@pytest.mark.asyncio
async def test_redis_down():
    pass


@pytest.mark.asyncio
async def test_redis_down_and_up():
    pass


@pytest.mark.asyncio
async def test_short_key_ttl():
    pass


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

        stream = open("profiling_output.log", "w")
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.sort_stats("cumtime")
        s = stats.print_stats()
        stream.close()
