import asyncio
import logging
import time
import traceback
from typing import Union, Tuple

from redis.asyncio import Redis
from lru import LRU
import signal_state_aio as signal_state


CACHING_PLACEHOLDER = "2c017ac168a7d22180e1c2fd60e70b0a"

# TODO: discuss several timeout/sleep values (search for keyword `timeout` or `sleep`)

class CachedRedis(Redis):

    VALUE_SLOT = 0
    TTL_SLOT = 1

    def __init__(self, *args, **kwargs) -> None:
        self.perfix = kwargs.pop("perfix", [])
        self.expire_threshold = kwargs.pop("expire_threshold", 86400)
        self.check_health_interval = kwargs.pop("check_health_interval", 60)
        self.cache_size = kwargs.pop("cache_size", 10000)

        self._pubsub = None
        self._pubsub_client_id = None
        self._next_check_heath_time = 0

        self._local_cache = LRU(self.cache_size)
        self.perfix_command = "".join([f"PREFIX {p} " for p in set(self.perfix) if len(p)>0])
        super().__init__(*args, **kwargs)

    async def run(self) -> None:
        asyncio.create_task(self._background_listen_invalidate())

    async def set(self, key: str, value: Union[None, str, int, float]) -> None:
        """
        Set kv pair to redis server, nothing different from redis.set()
        TODO: add optional parameter to set expire time
        TODO(optional): add optional parameter to set NOLOOP
        """
        await super().set(key, value)
        # optional: cache writes to local cache
        # optional: use NOLOOP option to tell the server 
        # client don't want to receive invalidation messages 
        # for this keys that it modified.

    async def get(self, key: str) -> Union[None, str, int, float]:
        """
        Get value from redis server or client side cache.
        """
        if self._pubsub is None:
            value, _ = await self._get_from_redis(key, only_value=True)
            return value
        # Get value from local cache
        # 1. check if key exist in local cache
        # 2. check if key expire time is not reached
        if key in self._local_cache:
            if int(time.time()) < self._local_cache[key][self.TTL_SLOT]:
                logging.info(f"Get key from client-side cache: {key}")
                return self._local_cache[key][self.VALUE_SLOT]
            else:
                logging.info(f"Key exists in clien-side cache but expired: {key}")
        
        # Get value from redis server and set to cache
        # 1. set value to local cache to avoid race condition
        # 2. get value from redis server
        # 3. get expire time from redis server
        # 4. set value to local cache
        # 5. set expire time to local cache

        self._local_cache[key] = (CACHING_PLACEHOLDER, None)
        value, ttl = await self._get_from_redis(key)

        if value is not None: # Key exist
            if ttl >= 0: # Integer expire time
                ttl = min(ttl, self.expire_threshold)
            elif ttl == -1: # No expire time or integer expire time
                ttl = self.expire_threshold
            elif ttl == -2: # Key not exist
                self.flush_key(key)
                return None
            # check if the value is deleted by _listen_invalidate
            if self._local_cache.get(key, (None, None))[self.VALUE_SLOT] == CACHING_PLACEHOLDER:
                self._local_cache[key] = (value, int(time.time())+ttl)
                logging.info(f"Set key to client-side cache: {key}, value={value[:64]}, ttl={ttl}")
            else:
                self.flush_key(key)
                return None
        else: # Key not exist
            self.flush_key(key)
            return None
        return value
    
    async def _get_from_redis(self, key: str, only_value: bool=False) -> Tuple[Union[None, str, int, float], int]:
        value = ttl = None
        if only_value:
            value = await super().get(key)
        else:
            pipe = super().pipeline()
            pipe.get(key)
            pipe.ttl(key)
            value, ttl = await pipe.execute()
        logging.info(f"Get key from redis server: {key}, value={value}, ttl={ttl}")
        return value, ttl
    
    def flush_cache(self) -> None:
        """
        clean whole cache
        """
        self._local_cache.clear()
        logging.info("Flush client-side cache")
    
    def flush_key(self, key: str) -> None:
        """
        delete key from local cache
        """
        if key in self._local_cache:
            del self._local_cache[key]
        logging.info(f"Flush key from client-side cache: {key}")
    
    async def _background_listen_invalidate(self) -> None:
        """
        create another listen invalidate coroutine in case the current connection is broken
        """
        while signal_state:
            if self._pubsub is not None:
                continue
            else:
                await asyncio.gather(self._listen_invalidate())
            await asyncio.sleep(5)
        await self.stop()
    
    async def _listen_invalidate_on_open(self) -> None:
        """
        Steps to open listen invalidate coroutine
        1. get client id
        2. enable client tracking, redirect invalidate message to this connection
        3. subscribe __redis__:invalidate channel
        If any step failed, set self._pubsub to None to trigger a new listen invalidate coroutine
        """
        try:
            self._pubsub = super().pubsub()
            # get client id
            await self._pubsub.execute_command("CLIENT ID")
            self._pubsub_client_id = await self._pubsub.connection.read_response()
            if self._pubsub_client_id is None:
                raise Exception(f"CLIENT ID failed. resp=None")

            # client tracking
            await self._pubsub.execute_command(f"CLIENT TRACKING on REDIRECT {self._pubsub_client_id} BCAST {self.perfix_command}")
            resp = await self._pubsub.connection.read_response()
            if resp != b'OK':
                raise Exception(f"CLIENT TRACKING on failed. resp={resp}")

            # subscribe __redis__:invalidate
            await self._pubsub.subscribe("__redis__:invalidate")
            resp = await self._pubsub.connection.read_response()
            if resp[-1] != 1:
                raise Exception(f"SUBCRIBE __redis__:invalidate failed. resp={resp}")
            logging.info(f"Listen invalidate on open success. client_id={self._pubsub_client_id}")
        except Exception as e:
            logging.info(f"Listen invalidate on open failed. error={e}, traceback={traceback.format_exc()}")
            self._pubsub = None
            self._pubsub_client_id = None
    
    async def _listen_invalidate_on_close(self) -> None:
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
            if self._pubsub is not None:
                await self._pubsub.close()
        except Exception as e:
            logging.info(f"Listen invalidate on close failed. error={e}, traceback={traceback.format_exc()}")
        finally:
            logging.info(f"Listen invalidate on close complete. client_id={self._pubsub_client_id}")
            self._pubsub = None
            self._pubsub_client_id = None
    
    async def _listen_invalidate_check_health(self) -> bool:
        """
        check if the current listen invalidate connection is healthy
        By default, the connections in the pool does not has health_check_interval
        But we do need health check for THIS, EXACTLY THIS listen invalidate connection
        """
        if self._pubsub is not None:
            try:
                await self._pubsub.ping()
                logging.info(f"Listen invalidate connection is healthy. client_id={self._pubsub_client_id}")
                return True
            except Exception as e:
                logging.info(f"Listen invalidate connection is broken. error={e}, traceback={traceback.format_exc()}")
        else:
            logging.info(f"Listen invalidate connection is broken. self._pubsub=None")
        return False

    async def _listen_invalidate(self) -> None:
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
                logging.info(f"Listen invalidate failed. error={e}, traceback={traceback.format_exc()}")
                break
            if message is None or not message.get("data"):
                await asyncio.sleep(1)
                continue
            key = message["data"][0]
            self.flush_key(key)
            logging.info(f"Invalidate key: {key}")
        await self._listen_invalidate_on_close()

    async def stop(self) -> None:
        """
        Steps to stop client side cache
        1. unsubscribe __redis__:invalidate channel
        2. disable client tracking

        This function is called when the client is closed or the object is about to delete
        """
        try:
            # unsubscribe __redis__:invalidate
            resp = await super().execute_command("UNSUBSCRIBE __redis__:invalidate")
            if resp[-1] != 0:
                raise Exception(f"UNSUBCRIBE __redis__:invalidate failed. resp={resp}")
            # client tracking off
            resp = await super().execute_command("CLIENT TRACKING off")
            if resp != b'OK':
                raise Exception(f"CLIENT TRACKING off failed. resp={resp}")
        except Exception as e:
            logging.info(f"Stop failed. error={e}, traceback={traceback.format_exc()}")