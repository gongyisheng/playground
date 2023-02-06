import asyncio

from redis.asyncio import BlockingConnectionPool, Redis


class ClientSideCache(object):
    def __init__(self, redis_host):
        self._pool = BlockingConnectionPool(host=redis_host, decode_responses=True, max_connections=10)
        self._local_cache = {}

    def __await__(self):
        return self.init().__await__()

    async def init(self):
        self._redis = await Redis(connection_pool=self._pool)
        asyncio.create_task(self._listen_invalidate())
        return self

    async def set(self, key, value):
        self._local_cache[key] = value
        await self._redis.set(key, value)

    async def get(self, key):
        if key in self._local_cache:
            return self._local_cache[key]
        value = await self._redis.get(key)
        if value is not None:
            self._local_cache[key] = value
        return value
    
    async def _listen_invalidate(self):
        pubsub = self._redis.pubsub()
        await pubsub.execute_command(b"CLIENT", b"ID")
        client_id = await pubsub.connection.read_response()
        await pubsub.execute_command(f"CLIENT TRACKING on REDIRECT {client_id} BCAST")
        await pubsub.connection.read_response()
        await pubsub.subscribe("__redis__:invalidate")

        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if message is None or not message.get("data"):
                continue
            key = message["data"][0]
            del self._local_cache[key]

async def main():
    client = await ClientSideCache("localhost")
    await client.set("my_key", "my_value")
    for i in range(100):
        print(await client.get("my_key"))
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())