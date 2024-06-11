import asyncio
import pytest
import pytest_asyncio
from redis import asyncio as aioredis


async def init():
    return redis


@pytest_asyncio.fixture
async def redis():
    pool = aioredis.BlockingConnectionPool(
        host="localhost", port=6379, db=0, max_connections=5
    )
    redis = aioredis.Redis(
        connection_pool=pool
    )  # You can also test decode_responses=True, it should also work
    await redis.flushdb()
    await redis.set("key", "value")

    yield redis

    await redis.close(close_connection_pool=True)


@pytest.mark.asyncio
async def test_get(redis):
    assert (await redis.get("key")) == b"value"


@pytest.mark.asyncio
async def test_frequent_get(redis):
    redis = await redis
    for i in range(1000):
        assert (await redis.get("key")) == b"value"
