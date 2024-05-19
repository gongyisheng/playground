# Before running this test, you need to start redis server first
#    redis-server --maxclients 65535
#    ulimit -n 65535 (avoid OSError: [Errno 24] Too many open files)
#    config set appendonly yes (data persistence requirement)
#    config set appendfsync always (data persistence requirement)
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_MAX_CONNECTIONS = 5
# Test Dependencies:
#    pip install pytest pytest-asyncio pytest-repeat
# Test command:
#    pytest -v
#    pytest -v -s (show print)
#    pytest -k <test function name>
#    pytest -k <test function name> --count 10 (repeat 10 times)
#    pytest -k <test function name> -s (show print)
#    python <test function name> run test function directly, usually for demo
#    python <test function name> run test function directly and profiling it

import asyncio
from contextvars import ContextVar
from functools import partial
import json
import logging
import pytest
import random
import signal_state_aio as signal_state
import string
import time
from typing import Callable, List, Optional, Union
import uuid

from redis.asyncio import Redis, BlockingConnectionPool
from stream import RedisStream

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

    fh = logging.FileHandler("redis_stream_unittest.log")
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


async def monitor(redis, callback_func=[]):
    async with redis.monitor() as m:
        async for info in m.listen():
            for func in callback_func:
                func(message=info)
            logging.debug(
                f"server side monitor: command={info.get('command')}, time={info.get('time')}"
            )


# Monitor audit functions
def _audit_monitor(
    message: dict,
    data_return: Optional[list] = None,
    prefix: Union[list, str, None] = None,
):
    if message is None or data_return is None or prefix is None:
        return
    if isinstance(prefix, str):
        prefix = [prefix]
    for p in prefix:
        if message["command"].startswith(p):
            data_return.append(
                {
                    "command": message["command"],
                    "time": message["time"],
                }
            )
            break


async def init(**kwargs):
    setup_logger()
    signal_state.ALIVE = True
    signal_state.register_exit_signal()
    pool = BlockingConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        max_connections=kwargs.get("max_connections", REDIS_MAX_CONNECTIONS + 1),
        timeout=30,
    )
    redis = Redis(
        connection_pool=pool
    )  # You can also test decode_responses=True, it should also work
    await redis.flushdb()

    redis_stream = RedisStream(redis, "test_stream")
    await redis_stream.init()

    monitor_task = asyncio.create_task(
        monitor(redis, kwargs.get("monitor_callback", []))
    )

    await asyncio.sleep(0.5)
    return redis_stream, monitor_task


def gen_test_messages(size=8, count=10) -> List[str]:
    messages = []
    for _ in range(count):
        messages.append(
            "".join(
                random.choice(string.ascii_letters + string.digits) for _ in range(size)
            )
        )
    return messages


async def demo():
    redis_stream, monitor_task = await init()
    put_messages = gen_test_messages(count=10)
    await redis_stream.batch_put(values=put_messages)

    get_messages = await redis_stream.batch_get(count=10)
    ack_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages]
    await redis_stream.batch_ack(ack_ids)

    assert len(put_messages) == len(get_messages)
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    assert monitor_task.done() is False
    monitor_task.cancel()

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_put_succ():
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XADD", "MULTI", "EXEC"],
            ),
        ],
    )
    message_count = 10
    put_messages = gen_test_messages(count=message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == message_count + 2
    assert AUDIT_DATA_RETURN[0]["command"] == "MULTI"
    assert AUDIT_DATA_RETURN[-1]["command"] == "EXEC"
    for i in range(1, message_count + 1):
        assert AUDIT_DATA_RETURN[i]["command"].startswith("XADD")

    await redis_stream._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_batch_put_fail():
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XADD", "MULTI", "EXEC"],
            ),
        ],
    )

    await redis_stream._redis.flushdb()
    await redis_stream._redis.set("test_stream", "test_value")

    retry = 3
    message_count = 10
    put_messages = gen_test_messages(count=message_count)
    succ = await redis_stream.batch_put(values=put_messages, retry=retry)
    assert succ is False

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == retry * (message_count + 2)
    for i in range(retry):
        assert AUDIT_DATA_RETURN[i * (message_count + 2)]["command"] == "MULTI"
        assert AUDIT_DATA_RETURN[(i + 1) * (message_count + 2) - 1]["command"] == "EXEC"
        for j in range(1, message_count + 1):
            assert AUDIT_DATA_RETURN[i * (message_count + 2) + j]["command"].startswith("XADD")

    await redis_stream._redis.close(close_connection_pool=True)

@pytest.mark.asyncio
async def test_batch_get_succ():
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XREADGROUP"],
            ),
        ],
    )
    message_count = 10
    put_messages = gen_test_messages(count=message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_messages = await redis_stream.batch_get(count=message_count)
    assert len(get_messages) == message_count

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    for i in range(message_count):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    await redis_stream._redis.close(close_connection_pool=True)


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
