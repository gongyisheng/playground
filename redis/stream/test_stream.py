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

STREAM_NAME = "test_stream"
GET_BLOCK_TIME = 2500

import asyncio
from contextvars import ContextVar
from functools import partial
import logging
import pytest
import random
import signal_state_aio as signal_state
import string
import time
from typing import List, Optional, Union
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

    redis_stream = RedisStream(redis, STREAM_NAME)
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
async def test_batch_put_error():
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
    await redis_stream._redis.set(STREAM_NAME, "test_value")

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
            assert AUDIT_DATA_RETURN[i * (message_count + 2) + j]["command"].startswith(
                "XADD"
            )

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_get_succ1():
    """
    Success case for batch get: get message count is less than put message count
    """
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
    put_message_count = 10
    get_message_count = random.randint(1, put_message_count)
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {get_message_count}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"BLOCK {GET_BLOCK_TIME}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"STREAMS {STREAM_NAME} >" in AUDIT_DATA_RETURN[0]["command"]
    for i in range(get_message_count):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_get_succ2():
    """
    Success case for batch get: get message count is more than put message count
    """
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
    put_message_count = 10
    get_message_count = random.randint(put_message_count + 1, put_message_count * 2)
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == put_message_count

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {get_message_count}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"BLOCK {GET_BLOCK_TIME}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"STREAMS {STREAM_NAME} >" in AUDIT_DATA_RETURN[0]["command"]
    for i in range(put_message_count):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_get_unicode():
    """
    Success case for batch get: put and get unicode messages
    """
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

    put_messages = [
        "这是一段中文测试",  # Chinese
        "🎡🐖🚿↕️🏓♉️🌨🤓😟🔁💔🎆🌆🔍😡🍧🚷🌇↗️🦀❎⏏🍃🤐🙍👤🎧🌹⏺🕦🏮⛑",  # emoji
        "\t\n\r",  # control characters
        "هذا اختبار عربي",  # Arabic
        "ဒါက ဗမာဘာသာစကား စမ်းသပ်မှုပါ။",  # Burmese
        "これは日本語のテストです",  # Japanese
        "이것은 한국어 테스트입니다",  # Korean
        "Esta es una prueba de español",  # Spanish
        "Este é um teste de português",  # Portuguese
        "Ceci est un test en français",  # French
        "यह एक हिंदी परीक्षा है",  # Hindi
        "Это тест на русском",  # Russian
        "זהו מבחן בעברית",  # Hebrew
        "ນີ້ແມ່ນການທົດສອບພາສາລາວ",  # Lao
        "Αυτό είναι ένα ελληνικό τεστ",  # Greek
        "این یک آزمون فارسی است",  # Persian
        "ეს არის ქართული ტესტი",  # Georgian
        "ഇത് ഒരു മലയാളം പരീക്ഷണമാണ്",  # Malayalam
        "Đây là bài kiểm tra tiếng Việt",  # Vietnamese
        "මෙය සිංහල පරිවර්තනයකි",  # Sinhala
        "இது தமிழ் சோதனை",  # Tamil
        "ఇది తెలుగు పరీక్ష",  # Telugu
        "\u00C0\u00C1\u00C2\u00C3\u00C4\u00C5\u00C6\u00C7\u00C8\u00C9\u00CA\u00CB\u00CC\u00CD\u00CE\u00CF",  # Latin-1 Supplement
        "🀱🀲🀳🀴🀵🀶🀷🀸🀹🀺🀻🀼🀽🀾🀿",  # Domino Tiles
        "🀠🀡🀢🀣🀤🀥🀦🀧🀨🀩🀪",  # Mahjong Tiles
        "𓅯𓆝𓆟𓆜𓆞𓆝𓆟𓆜𓆞",  # Egyptian Hieroglyph
    ]
    put_message_count = len(put_messages)
    get_message_count = put_message_count
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == put_message_count

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {get_message_count}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"BLOCK {GET_BLOCK_TIME}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"STREAMS {STREAM_NAME} >" in AUDIT_DATA_RETURN[0]["command"]
    for i in range(put_message_count):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_get_empty_result1():
    """
    Empty result case for batch get: stream is empty
    """
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

    get_message_count = 10
    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )

    assert isinstance(get_messages, list)
    assert len(get_messages) == 0

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {get_message_count}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"BLOCK {GET_BLOCK_TIME}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"STREAMS {STREAM_NAME} >" in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_get_block():
    """
    Block time test for batch get
    """
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

    get_message_count = random.randint(1, 10)
    get_block_time = random.randint(1000, 5000)
    start_time = time.time()
    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=get_block_time
    )
    end_time = time.time()
    elapsed_time = int((end_time - start_time) * 1000)

    assert isinstance(get_messages, list)
    assert len(get_messages) == 0
    assert elapsed_time >= get_block_time
    logging.info(f"block time: {get_block_time}, elapsed time: {elapsed_time}")

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {get_message_count}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"BLOCK {get_block_time}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"STREAMS {STREAM_NAME} >" in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_get_error():
    """
    Error case for batch get: WRONGTYPE Operation against a key holding the wrong kind of value
    """
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
    await redis_stream._redis.flushdb()
    await redis_stream._redis.set(STREAM_NAME, "test_value")

    get_message_count = 10
    has_error = False
    try:
        get_messages = await redis_stream.batch_get(
            count=get_message_count, block=GET_BLOCK_TIME
        )
    except Exception as ex:
        logging.error("Error caught: %s", ex)
        has_error = True

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert has_error is True
    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {get_message_count}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"BLOCK {GET_BLOCK_TIME}" in AUDIT_DATA_RETURN[0]["command"]
    assert f"STREAMS {STREAM_NAME} >" in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_ack_succ1():
    """
    Success case for batch get: get message count is less than put message count
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XACK"],
            ),
        ],
    )
    put_message_count = 10
    get_message_count = random.randint(1, put_message_count)
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages]
    succ = await redis_stream.batch_ack(successful_ids)
    assert succ is True

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    for id in successful_ids:
        assert id in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_ack_succ2():
    """
    Success case for batch ack: ack duplicated message id
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XACK"],
            ),
        ],
    )
    put_message_count = 10
    get_message_count = random.randint(1, put_message_count)
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages] + [
        get_messages[0][RedisStream.MESSAGE_ID_KEY]
    ]
    succ = await redis_stream.batch_ack(successful_ids)
    assert succ is True

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    for id in set(successful_ids):
        assert id in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_ack_error():
    """
    Error case for batch get: ack non-exist message id
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XACK"],
            ),
        ],
    )
    put_message_count = 10
    get_message_count = random.randint(1, put_message_count)
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    retry = 3
    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages] + [
        "non_exist_id"
    ]
    succ = await redis_stream.batch_ack(successful_ids, retry=retry)
    assert succ is False

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == retry
    for i in range(retry):
        for id in successful_ids:
            assert id in AUDIT_DATA_RETURN[i]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_ack_with_delete():
    """
    Case for batch ack: ack with delete message
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XACK"],
            ),
        ],
    )
    put_message_count = 10
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_message_count = put_message_count
    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    # Trim the stream
    await asyncio.sleep(1)
    delete_count, stream_length = await redis_stream.trim(
        0, approximate=False, force_delete=True
    )
    assert delete_count == get_message_count
    assert stream_length == 0

    # Ack the deleted messages
    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages]
    succ = await redis_stream.batch_ack(successful_ids)
    assert succ is True

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    for id in successful_ids:
        assert id in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_monitor_xpel_succ():
    """
    Success case for monitor pending list
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XPENDING"],
            ),
        ],
    )
    put_message_count = 10
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_message_count = put_message_count
    get_messages = await redis_stream.batch_get(count=get_message_count)
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages]
    for i in range(len(successful_ids)):
        length, min_time = await redis_stream.monitor_xpel()
        assert length == get_message_count - i
        assert min_time != -1
        await redis_stream.batch_ack([successful_ids[i]])

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == get_message_count

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_claim_succ1():
    """
    Success case for batch claim: claim message count is less than pending list message count
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XAUTOCLAIM"],
            ),
        ],
    )
    put_message_count = 10
    get_message_count = random.randint(1, put_message_count)
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    claim_message_count = random.randint(1, get_message_count)
    claim_messages = await redis_stream.batch_claim(count=claim_message_count)
    assert len(claim_messages) == claim_message_count
    for i in range(claim_message_count):
        assert put_messages[i] == claim_messages[i][RedisStream.MESSAGE_DATA_KEY]

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {claim_message_count}" in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_claim_succ2():
    """
    Success case for batch claim: claim message count is more than pending list message count
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XAUTOCLAIM"],
            ),
        ],
    )
    put_message_count = 10
    get_message_count = random.randint(1, put_message_count)
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count

    claim_message_count = random.randint(get_message_count + 1, put_message_count * 2)
    claim_messages = await redis_stream.batch_claim(count=claim_message_count)
    assert len(claim_messages) == get_message_count
    for i in range(len(claim_messages)):
        assert put_messages[i] == claim_messages[i][RedisStream.MESSAGE_DATA_KEY]

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {claim_message_count}" in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_claim_empty_result():
    """
    Empty result case for batch claim: no message in the pending list
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XAUTOCLAIM"],
            ),
        ],
    )
    put_message_count = 10
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    claim_message_count = random.randint(1, put_message_count)
    claim_messages = await redis_stream.batch_claim(count=claim_message_count)
    assert isinstance(claim_messages, list)
    assert len(claim_messages) == 0

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {claim_message_count}" in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_claim_error():
    """
    Error case for batch claim: WRONGTYPE Operation against a key holding the wrong kind of value
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XAUTOCLAIM"],
            ),
        ],
    )
    await redis_stream._redis.flushdb()
    await redis_stream._redis.set(STREAM_NAME, "test_value")

    claim_message_count = 10
    has_error = False
    try:
        claim_messages = await redis_stream.batch_claim(count=claim_message_count)
    except Exception as ex:
        logging.error("Error caught: %s", ex)
        has_error = True

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert has_error is True
    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {claim_message_count}" in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_batch_claim_with_delete():
    """
    Successful case for batch claim: contains delete message
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XAUTOCLAIM"],
            ),
        ],
    )
    put_message_count = 10
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_message_count = put_message_count
    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    # Trim the stream
    delete_count, stream_length = await redis_stream.trim(
        0, approximate=False, force_delete=True
    )
    assert delete_count == put_message_count
    assert stream_length == 0

    # Claim message with delete
    claim_message_count = random.randint(1, get_message_count)
    claim_messages = await redis_stream.batch_claim(count=claim_message_count)
    assert isinstance(claim_messages, list)
    assert len(claim_messages) == 0

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1
    assert f"COUNT {claim_message_count}" in AUDIT_DATA_RETURN[0]["command"]

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_trim_succ():
    """
    Success case for trim: trim message that has inserted time older than the specified time
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XTRIM"],
            ),
        ],
    )
    put_message_count = 10
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    print(int(time.time() * 1000))
    assert succ is True

    await asyncio.sleep(2)
    checkpoint_time = int(time.time())

    await asyncio.sleep(2)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    delete_count, stream_length = await redis_stream.trim(3600, force_delete=True)
    assert delete_count == 0
    assert stream_length == put_message_count * 2

    delete_count, stream_length = await redis_stream.trim(
        int(time.time()) - checkpoint_time, approximate=False, force_delete=False
    )
    assert delete_count == 0
    assert stream_length == put_message_count * 2

    delete_count, stream_length = await redis_stream.trim(
        int(time.time()) - checkpoint_time, approximate=False, force_delete=True
    )
    assert delete_count == put_message_count
    assert stream_length == put_message_count

    delete_count, stream_length = await redis_stream.trim(
        0, approximate=False, force_delete=True
    )
    assert delete_count == put_message_count
    assert stream_length == 0

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 4

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_range_replay_succ1():
    """
    Success case for range replay: replay message that is in the specified range, left_closed=True
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XRANGE"],
            ),
        ],
    )
    put_message_count = 10
    get_message_count = random.randint(1, put_message_count)
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    start_time = int(time.time())

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages]
    succ = await redis_stream.batch_ack(successful_ids)
    assert succ is True

    await asyncio.sleep(1)
    end_time = int(time.time())

    replay_messages = await redis_stream.range_replay(
        start_id=f"{start_time*1000}-0",
        end_id=f"{end_time*1000}-0",
        count=put_message_count,
    )
    assert len(replay_messages) == put_message_count
    for i in range(len(replay_messages)):
        assert put_messages[i] == replay_messages[i][RedisStream.MESSAGE_DATA_KEY]

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_range_replay_succ2():
    """
    Success case for range replay: replay message that is in the specified range, left_closed=False
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XRANGE"],
            ),
        ],
    )
    put_message_count = 10
    get_message_count = random.randint(1, put_message_count)
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    start_time = int(time.time())

    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages]
    succ = await redis_stream.batch_ack(successful_ids)
    assert succ is True

    await asyncio.sleep(1)
    end_time = int(time.time())

    start_id = (
        await redis_stream.range_replay(
            start_id=f"{start_time*1000}-0",
            end_id=f"{end_time*1000}-0",
            count=1,
            left_closed=True,
        )
    )[0][RedisStream.MESSAGE_ID_KEY]

    replay_messages = await redis_stream.range_replay(
        start_id=start_id,
        end_id=f"{end_time*1000}-0",
        count=put_message_count,
        left_closed=False,
    )
    assert len(replay_messages) == put_message_count - 1
    for i in range(len(replay_messages)):
        assert put_messages[i + 1] == replay_messages[i][RedisStream.MESSAGE_DATA_KEY]

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 2

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_range_replay_empty_result():
    """
    Empty result case for range replay: replay message that is not in the specified range
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XRANGE"],
            ),
        ],
    )
    put_message_count = 10
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_message_count = random.randint(1, put_message_count)
    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]

    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages]
    succ = await redis_stream.batch_ack(successful_ids)
    assert succ is True

    await asyncio.sleep(1)

    start_time = 0
    end_time = int(time.time()) - 3600
    replay_messages = await redis_stream.range_replay(
        start_id=f"{start_time*1000}-0",
        end_id=f"{end_time*1000}-0",
        count=put_message_count,
    )
    assert len(replay_messages) == 0

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == 1

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_item_replay_succ():
    """
    Success case for item replay: replay message with specified message id
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XRANGE"],
            ),
        ],
    )
    MAPPING = {}
    put_message_count = 10
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_message_count = put_message_count
    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]
        MAPPING[get_messages[i][RedisStream.MESSAGE_ID_KEY]] = get_messages[i][
            RedisStream.MESSAGE_DATA_KEY
        ]

    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages]
    succ = await redis_stream.batch_ack(successful_ids)
    assert succ is True

    await asyncio.sleep(1)

    random.shuffle(successful_ids)
    replay_message_count = put_message_count
    replay_messages = await redis_stream.item_replay(successful_ids)
    assert len(replay_messages) == replay_message_count
    for i in range(len(replay_messages)):
        assert (
            MAPPING[replay_messages[i][RedisStream.MESSAGE_ID_KEY]]
            == replay_messages[i][RedisStream.MESSAGE_DATA_KEY]
        )

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == replay_message_count

    await redis_stream._redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_item_replay_empty_result():
    """
    Empty result case for item replay: replay message with non-exist message id
    """
    AUDIT_DATA_RETURN = []
    redis_stream, monitor_task = await init(
        monitor_callback=[
            partial(
                _audit_monitor,
                data_return=AUDIT_DATA_RETURN,
                prefix=["XRANGE"],
            ),
        ],
    )
    MAPPING = {}
    put_message_count = 10
    put_messages = gen_test_messages(count=put_message_count)
    succ = await redis_stream.batch_put(values=put_messages)
    assert succ is True

    get_message_count = put_message_count
    get_messages = await redis_stream.batch_get(
        count=get_message_count, block=GET_BLOCK_TIME
    )
    assert len(get_messages) == get_message_count
    for i in range(len(get_messages)):
        assert put_messages[i] == get_messages[i][RedisStream.MESSAGE_DATA_KEY]
        MAPPING[get_messages[i][RedisStream.MESSAGE_ID_KEY]] = get_messages[i][
            RedisStream.MESSAGE_DATA_KEY
        ]

    successful_ids = [item[RedisStream.MESSAGE_ID_KEY] for item in get_messages]
    succ = await redis_stream.batch_ack(successful_ids)
    assert succ is True

    await asyncio.sleep(1)

    replay_ids = ["0-0", "1-1", "2-2"]
    replay_messages = await redis_stream.item_replay(replay_ids)
    assert len(replay_messages) == 0

    assert monitor_task.done() is False
    monitor_task.cancel()

    assert len(AUDIT_DATA_RETURN) == len(replay_ids)

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
