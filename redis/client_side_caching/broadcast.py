import asyncio
import logging
import time
import traceback
from typing import Union, Tuple

import redis.asyncio as aioredis
from lru import LRU
import signal_state_aio as signal_state


# TODO: discuss several timeout/sleep values (search for keyword `timeout` or `sleep`)

class CachedRedis(aioredis.Redis):

    VALUE_SLOT = 0
    EXPIRE_TIME_SLOT = 1

    SET_CACHE_LOCK = asyncio.Lock()
    SET_CACHE_EVENT = asyncio.Event()
    SET_CACHE_PLACEHOLDER = object()

    LISTEN_INVALIDATE_COROUTINE_EVENT = asyncio.Event()

    TASK_NAME = 'task-cached_redis'

    def __init__(self, *args, **kwargs) -> None:
        self.prefix = kwargs.pop("prefix", [])
        self.expire_threshold = kwargs.pop("expire_threshold", 86400)
        self.check_health_interval = kwargs.pop("check_health_interval", 60)
        self.cache_size = kwargs.pop("cache_size", 10000)

        self._pubsub_connection = None
        self._pubsub_client_id = None
        self._next_check_heath_time = 0

        self._local_cache = LRU(self.cache_size)
        self.prefix_commands = [f"PREFIX {p}" for p in set(self.prefix)] if len(self.prefix) > 0 else [""]
        super().__init__(*args, **kwargs)

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
        if self._pubsub_connection is None:
            value, _ = await self._get_from_redis(key, only_value=True)
            return value
        # Get value from local cache
        # 1. check if key exist in local cache
        # 2. check if key expire time is not reached
        if key in self._local_cache:
            if self._local_cache[key][self.VALUE_SLOT] == self.SET_CACHE_PLACEHOLDER:
                await self.SET_CACHE_EVENT.wait()
            if int(time.time()) <= self._local_cache[key][self.EXPIRE_TIME_SLOT]:
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

        async with self.SET_CACHE_LOCK:
            self._local_cache[key] = (self.SET_CACHE_PLACEHOLDER, None)
            value, ttl = await self._get_from_redis(key)

            if value is not None: 
                # Key exist
                ttl = min(ttl, self.expire_threshold) if ttl >=0 else self.expire_threshold

                # check if the value is deleted by _listen_invalidate
                if self._local_cache.get(key, (None, None))[self.VALUE_SLOT] == self.SET_CACHE_PLACEHOLDER:
                    self._local_cache[key] = (value, int(time.time())+ttl)
                    logging.info(f"Set key to client-side cache: {key}, ttl={ttl}")
                else:
                    value = None
                    self.flush_key(key)
            else: 
                # Key not exist
                self.flush_key(key)
            # notify other coroutines waiting for this key, and clear the event
            self.SET_CACHE_EVENT.set()
            self.SET_CACHE_EVENT.clear()
        return value
    
    async def _get_from_redis(self, key: str, only_value: bool=False) -> Tuple[Union[None, str, int, float], int]:
        value = ttl = None
        if only_value:
            value = await super().get(key)
        else:
            # Use pipelien to execute a transaction
            pipe = super().pipeline()
            pipe.get(key)
            pipe.ttl(key)
            value, ttl = await pipe.execute()
        logging.info(f"Get key from redis server: {key}, ttl={ttl}")
        return value, ttl
    
    def flush_cache(self) -> None:
        """
        clean whole cache
        """
        self._local_cache.clear()
        logging.info("Flush ALL client-side cache")
    
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
        logging.info("Start _background_listen_invalidate")
        try:
            while signal_state.ALIVE:
                asyncio.create_task(self._listen_invalidate(), name=self.TASK_NAME)
                self.LISTEN_INVALIDATE_COROUTINE_EVENT.clear()
                await self.LISTEN_INVALIDATE_COROUTINE_EVENT.wait()
        except Exception as e:
            logging.error("Error in _background_listen_invalidate", error=e)
        finally:
            await self.stop()
            logging.info("Exit _background_listen_invalidate")
    
    async def _listen_invalidate_on_open(self) -> None:
        """
        Steps to open listen invalidate coroutine
        1. get client id
        2. enable client tracking, redirect invalidate message to this connection
        3. subscribe __redis__:invalidate channel
        If any step failed, set self._pubsub_connection to None to trigger a new listen invalidate coroutine
        """
        try:
            self._pubsub_connection = await self.connection_pool.get_connection("_")
            # get client id
            await self._pubsub_connection.send_command("CLIENT ID")
            self._pubsub_client_id = await self._pubsub_connection.read_response()
            if self._pubsub_client_id is None:
                raise Exception(f"CLIENT ID failed. resp=None")

            # client tracking
            prefix_command = " ".join(self.prefix_commands)
            await self._pubsub_connection.send_command(f"CLIENT TRACKING on REDIRECT {self._pubsub_client_id} BCAST {prefix_command}")
            resp = await self._pubsub_connection.read_response()
            if resp != b'OK':
                raise Exception(f"CLIENT TRACKING on failed. resp={resp}")

            # subscribe __redis__:invalidate
            await self._pubsub_connection.send_command("SUBSCRIBE __redis__:invalidate")
            resp = await self._pubsub_connection.read_response()
            if resp[-1] != 1:
                raise Exception(f"SUBCRIBE __redis__:invalidate failed. resp={resp}")
            logging.info(f"Listen invalidate on open success. client_id={self._pubsub_client_id}")
        except Exception as e:
            logging.error(f"Listen invalidate on open failed. error={e}, traceback={traceback.format_exc()}")
            self._pubsub_connection = None
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
            if self._pubsub_connection is not None:
                await self._pubsub_connection.disconnect()
        except Exception as e:
            logging.error(f"Listen invalidate on close failed. error={e}, traceback={traceback.format_exc()}")
        finally:
            self.connection_pool.pool.put_nowait(None)
            logging.info(f"Listen invalidate on close complete. client_id={self._pubsub_client_id}")
            self._pubsub_connection = None
            self._pubsub_client_id = None
    
    async def _listen_invalidate_check_health(self) -> bool:
        """
        check if the current listen invalidate connection is healthy
        By default, the connections in the pool does not has health_check_interval
        But we do need health check for THIS, EXACTLY THIS listen invalidate connection
        """
        if self._pubsub_connection is not None:
            try:
                await self._pubsub_connection.check_health()
                logging.info(f"Listen invalidate connection is healthy. client_id={self._pubsub_client_id}")
                return True
            except Exception as e:
                logging.error(f"Listen invalidate connection is broken. error={e}, traceback={traceback.format_exc()}")
        else:
            logging.error(f"Listen invalidate connection is broken. self._pubsub_connection=None")
        return False

    async def _listen_invalidate(self) -> None:
        """
        listen invalidate message from redis server
        TODO: discuss a better timeout value
        """
        await self._listen_invalidate_on_open()
        while signal_state.ALIVE and self._pubsub_connection is not None:
            now = int(time.time())
            if self._next_check_heath_time < now:
                if not await self._listen_invalidate_check_health():
                    break
                else:
                    self._next_check_heath_time = now + self.check_health_interval
            try:
                message = await self._pubsub_connection.read_response(timeout=1)
            except Exception as e:
                logging.error(f"Listen invalidate failed. error={e}, traceback={traceback.format_exc()}")
                break

            if message is None or not message[-1]:
                continue
            key = message[-1][0].decode('ascii')
            self.flush_key(key)
            logging.info(f"Invalidate key {key} because received invalidate message from redis server")
        await self._listen_invalidate_on_close()
        self.LISTEN_INVALIDATE_COROUTINE_EVENT.set()

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
            logging.error(f"Stop failed. error={e}, traceback={traceback.format_exc()}")
    
    async def run(self) -> None:
        task_list = [asyncio.create_task(self._background_listen_invalidate())]
        await asyncio.gather(*task_list)

        try:
            logging.info('Waiting for all CachedRedis task to finish')
            task_list = [task for task in asyncio.all_tasks() if task.get_name() == self.TASK_NAME]
            await asyncio.wait_for(asyncio.gather(*task_list), timeout=30)
            logging.info('All CachedRedis tasks are done')
        except asyncio.TimeoutError:
            logging.error('Timeout when finish CachedRedis task')
        except Exception as e:
            logging.error('CachedRedis task error', error=e)