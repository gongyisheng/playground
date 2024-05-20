import asyncio
import pytest


async def async_add(x, y):
    await asyncio.sleep(1)
    return x + y


@pytest.mark.asyncio
async def test_async_addition():
    assert await async_add(2, 3) == 5


@pytest.mark.asyncio
async def test_async_addition_negative():
    assert await async_add(-1, 1) == 0


# This command will only run async tests:
# pytest -k asyncio
