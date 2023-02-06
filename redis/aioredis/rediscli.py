import asyncio

from redis.asyncio import BlockingConnectionPool, Redis


class ClientSideCache(object):
    def __init__(self, redis_host):
        self._pool = BlockingConnectionPool(host=redis_host, decode_responses=True, max_connections=10)

    def __await__(self):
        return self.init().__await__()

    async def init(self):
        self._redis = await Redis(connection_pool=self._pool)
        return self

    async def set(self, key, value):
        await self._redis.set(key, value)

    async def get(self, key):
        return await self._redis.get(key)

async def main():
    client = await ClientSideCache("localhost")
    await client.set("my_key", "my_value")
    print(await client.get("my_key"))

if __name__ == "__main__":
    asyncio.run(main())