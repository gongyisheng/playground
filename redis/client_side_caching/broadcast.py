import asyncio
import time
import traceback
from redis.asyncio import BlockingConnectionPool, Redis

CACHING_PLACEHOLDER = "__in_progress__"

# todo: limit cache size
# todo: use logging to replace print

class ClientSideCache(object):
    def __init__(self, redis_host, perfix=[], expire_threshold=86400, check_health_interval=60):
        self._pool = BlockingConnectionPool(host=redis_host, decode_responses=True, max_connections=10)

        self._local_cache = {}
        self._local_cache_expire_time = {}

        self._pubsub = None
        self._pubsub_client_id = None

        self.expire_threshold = expire_threshold
        self.check_health_interval = check_health_interval
        self._next_check_heath_time = 0

        self.perfix_command = "".join([f"PREFIX {p} " for p in set(perfix) if len(p)>0])

    def __await__(self):
        return self.init().__await__()

    async def init(self):
        self._redis = await Redis(connection_pool=self._pool)
        asyncio.create_task(self._background_listen_invalidate())
        return self

    async def set(self, key, value):
        """
        Set kv pair to redis server, nothing different from redis.set()
        TODO: add optional parameter to set expire time
        TODO(optional): add optional parameter to set NOLOOP
        """
        await self._redis.set(key, value)
        # optional: cache writes to local cache
        # optional: use NOLOOP option to tell the server 
        # client don't want to receive invalidation messages 
        # for this keys that it modified.

    async def get(self, key):
        """
        Get value from redis server or client side cache.
        """
        # Get value from local cache
        # 1. check if key exist in local cache
        # 2. check if key expire time is not reached
        if key in self._local_cache:
            if int(time.time()) < self._local_cache_expire_time[key]:
                print(f"Get key from client-side cache: {key}")
                return self._local_cache[key]
        
        # Get value from redis server and set to cache
        # 1. set value to local cache to avoid race condition
        # 2. get value from redis server
        # 3. get expire time from redis server
        # 4. set value to local cache
        # 5. set expire time to local cache

        self._local_cache[key] = CACHING_PLACEHOLDER
        value = await self._redis.get(key)
        ttl = await self._redis.ttl(key)
        print(f"Get key from redis server: {key}, value={value}, ttl={ttl}")

        if value is not None: # Key exist
            if ttl >= 0: # Integer expire time
                ttl = min(ttl, self.expire_threshold)
            elif ttl == -1: # No expire time or integer expire time
                ttl = self.expire_threshold
            elif ttl == -2: # Key not exist
                self.flush_key(key)
                return None
            # check if the value is deleted by _listen_invalidate
            if self._local_cache.get(key, None) == CACHING_PLACEHOLDER:
                self._local_cache[key] = value
                self._local_cache_expire_time[key] = int(time.time())+ttl
            else:
                self.flush_key(key)
                return None
        else: # Key not exist
            self.flush_key(key)
            return None
        return value
    
    def flush_cache(self):
        """
        clean whole cache
        """
        self._local_cache = {}
        self._local_cache_update_time = {}
    
    def flush_key(self, key):
        """
        delete key from local cache
        """
        if key in self._local_cache:
            del self._local_cache[key]
        if key in self._local_cache_update_time:
            del self._local_cache_update_time[key]
    
    async def _background_listen_invalidate(self):
        """
        create another listen invalidate coroutine in case the current connection is broken
        """
        while True:
            if self._pubsub is not None:
                continue
            else:
                await asyncio.gather(self._listen_invalidate())
            await asyncio.sleep(5)
    
    async def _listen_invalidate_on_open(self):
        """
        Steps to open listen invalidate coroutine
        1. get client id
        2. enable client tracking, redirect invalidate message to this connection
        3. subscribe __redis__:invalidate channel
        If any step failed, set self._pubsub to None to trigger a new listen invalidate coroutine
        """
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
        """
        Steps to close listen invalidate coroutine
        1. flush whole client side cache
        2. close pubsub and release connection

        This function is called when:
        1. the connection is broken
        2. redis server failed
        3. the client is closed
        """
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
    
    async def _listen_invalidate_check_health(self):
        """
        check if the current listen invalidate connection is healthy
        By default, the connections in the pool does not has health_check_interval
        But we do need health check for THIS, EXACTLY THIS listen invalidate connection
        """
        if self._pubsub is not None:
            try:
                resp = await self._pubsub.ping()
                if resp != "PONG":
                    raise Exception(f"Listen invalidate connection is broken. resp={resp}")
                return True
            except Exception as e:
                print(f"Listen invalidate connection is broken. error={e}, traceback={traceback.format_exc()}")
        else:
            print(f"Listen invalidate connection is broken. self._pubsub=None")
        return False

    async def _listen_invalidate(self):
        """
        listen invalidate message from redis server
        TODO: discuss a better timeout value
        """
        await self._listen_invalidate_on_open()
        while self._pubsub is not None:
            now = int(time.time())
            if self._next_check_heath_time < now:
                if not await self._listen_invalidate_check_health():
                    break
                else:
                    self._next_check_heath_time = now + self.check_health_interval
            try:
                message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=0.2)
            except Exception as e:
                print(f"Listen invalidate failed. error={e}, traceback={traceback.format_exc()}")
                break
            if message is None or not message.get("data"):
                await asyncio.sleep(1)
                continue
            key = message["data"][0]
            self.flush_key(key)
            print(f"Invalidate key: {key}")
        await self._listen_invalidate_on_close()

    async def stop(self):
        """
        Steps to stop client side cache
        1. unsubscribe __redis__:invalidate channel
        2. disable client tracking

        This function is called when the client is closed or the object is about to delete
        """
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

    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.stop())
            else:
                loop.run_until_complete(self.stop())
        except Exception:
            pass

async def test():
    client = await ClientSideCache("localhost")
    await client.set("my_key", "my_value")
    for i in range(50):
        print(await client.get("my_key"))
        await asyncio.sleep(1)
    await client.stop()

if __name__ == "__main__":
    asyncio.run(test())