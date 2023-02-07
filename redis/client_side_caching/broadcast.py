import asyncio
import time
import traceback
from redis.asyncio import BlockingConnectionPool, Redis


class ClientSideCache(object):
    def __init__(self, redis_host, perfix=[], expire_threshold=86400):
        self._pool = BlockingConnectionPool(host=redis_host, decode_responses=True, max_connections=10)
        self._local_cache = {}
        self._local_cache_update_time = {}
        self._pubsub = None
        self._pubsub_client_id = None
        self.expire_threshold = expire_threshold
        self.perfix_command = "".join([f"PREFIX {p} " for p in set(perfix) if len(p)>0])

    def __await__(self):
        return self.init().__await__()

    async def init(self):
        self._redis = await Redis(connection_pool=self._pool)
        asyncio.create_task(self._background_listen_invalidate())
        return self

    async def set(self, key, value):
        await self._redis.set(key, value)
        # optional: cache writes to local cache
        # optional: use NOLOOP option to tell the server 
        # client don't want to receive invalidation messages 
        # for this keys that it modified.

    async def get(self, key):
        if key in self._local_cache:
            if int(time.time()) - self._local_cache_update_time[key] < self.expire_threshold:
                print(f"Get key from client-side cache: {key}")
                return self._local_cache[key]
        print(f"Get key from redis server: {key}")
        value = await self._redis.get(key)
        if value is not None:
            self._local_cache[key] = value
            self._local_cache_update_time[key] = int(time.time())
        return value
    
    def flush_cache(self):
        self._local_cache = {}
        self._local_cache_update_time = {}
    
    async def _background_listen_invalidate(self):
        while True:
            if self._pubsub is not None:
                continue
            else:
                await asyncio.gather(self._listen_invalidate())
            await asyncio.sleep(5)
    
    async def _listen_invalidate_on_open(self):
        try:
            self._pubsub = self._redis.pubsub()
            # get client id
            await self._pubsub.execute_command("CLIENT ID")
            self._pubsub_client_id = await self._pubsub.connection.read_response()
            if self._pubsub_client_id is None:
                raise Exception(f"CLIENT ID failed. resp=None")

            # client tracking
            await self._pubsub.execute_command(f"CLIENT TRACKING on REDIRECT {self._pubsub_client_id} BCAST {self.perfix_command}")
            resp = await self._pubsub.connection.read_response()
            if resp != "OK":
                raise Exception(f"CLIENT TRACKING on failed. resp={resp}")

            # subscribe __redis__:invalidate
            await self._pubsub.subscribe("__redis__:invalidate")
            resp = await self._pubsub.connection.read_response()
            if resp[-1] != 1:
                raise Exception(f"SUBCRIBE __redis__:invalidate failed. resp={resp}")
            print(f"Listen invalidate on open success. client_id={self._pubsub_client_id}")
        except Exception as e:
            print(f"Listen invalidate on open failed. error={e}, traceback={traceback.format_exc()}")
            self._pubsub = None
            self._pubsub_client_id = None
    
    async def _listen_invalidate_on_close(self):
        try:
            self.flush_cache()
            # release connection
            await self._pubsub.close()
        except Exception as e:
            print(f"Listen invalidate on close failed. error={e}, traceback={traceback.format_exc()}")
        finally:
            print(f"Listen invalidate on close complete. client_id={self._pubsub_client_id}")
            self._pubsub = None
            self._pubsub_client_id = None

    async def _listen_invalidate(self):
        await self._listen_invalidate_on_open()
        while self._pubsub is not None:
            try:
                message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
            except Exception as e:
                print(f"Listen invalidate failed. error={e}, traceback={traceback.format_exc()}")
                break
            if message is None or not message.get("data"):
                await asyncio.sleep(1)
                continue
            key = message["data"][0]
            if key in self._local_cache:
                del self._local_cache[key]
                print(f"Invalidate key: {key}")
        await self._listen_invalidate_on_close()

    async def stop(self):
        try:
            # unsubscribe __redis__:invalidate
            resp = await self._redis.execute_command("UNSUBSCRIBE __redis__:invalidate")
            if resp[-1] != 0:
                raise Exception(f"UNSUBCRIBE __redis__:invalidate failed. resp={resp}")
            # client tracking off
            resp = await self._redis.execute_command("CLIENT TRACKING off")
            if resp != "OK":
                raise Exception(f"CLIENT TRACKING off failed. resp={resp}")
        except Exception as e:
            print(f"Stop failed. error={e}, traceback={traceback.format_exc()}")

async def test():
    client = await ClientSideCache("localhost")
    #await client.set("my_key", "my_value")
    for i in range(50):
        print(await client.get("my_key"))
        await asyncio.sleep(1)
    await client.stop()

if __name__ == "__main__":
    asyncio.run(test())