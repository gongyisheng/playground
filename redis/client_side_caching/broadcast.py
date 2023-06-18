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
    HEALTH_CHECK_MSG = b'cached-redis-py-health-check'
    LISTEN_INVALIDATE_CHANNEL = b"__redis__:invalidate"
    SUBSCRIBE_SUCCESS_MSG = {
        'type':"subscribe", 
        'channel': LISTEN_INVALIDATE_CHANNEL, 
        'data': 1
    }
    UNSUBSCRIBE_SUCCESS_MSG = {
        'type':"unsubscribe",
        'channel': LISTEN_INVALIDATE_CHANNEL,
        'data': 0
    }

    def __init__(self, *args, **kwargs) -> None:
        self.prefix = kwargs.pop("prefix", [])
        self.expire_threshold = kwargs.pop("expire_threshold", 86400)
        self.pubsub_health_check_interval = kwargs.pop("pubsub_health_check_interval", 60)
        self.cache_size = kwargs.pop("cache_size", 10000)
        
        self._pubsub = None
        self._pubsub_client_id = None

        self.health_check_ongoing_flag = False
        self.health_check_timeout = 10
        self._last_health_check_time = 0
        self._next_health_check_time = 0

        self._local_cache = LRU(self.cache_size)
        
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
        1. If value is not in client side cache, get it from redis server and cache it.
        2. If value is in client side cache, return it.
        3. Lock and events are used to prevent race condition.
        4. Keys in local cache may be evicted for two reasons: LRU full or expire time reached.
        """
        value = None

        # get value from redis
        if self._pubsub is None:
            value, _ = await self._get_from_redis(key, only_value=True)
        
        # Get value from local cache
        # 1. check if key exist in local cache
        # 2. check if key expire time is not reached
        if value is None and key in self._local_cache:
            if self._local_cache[key][self.VALUE_SLOT] == self.SET_CACHE_PLACEHOLDER:
                await self.SET_CACHE_EVENT.wait()
            if int(time.time()) <= self._local_cache[key][self.EXPIRE_TIME_SLOT]:
                value = self._local_cache[key][self.VALUE_SLOT]
                logging.info(f"Get key from client-side cache: {key}, value: {str(value)[:16]}")
            else:
                logging.info(f"Key exists in clien-side cache but expired: {key}")
        
        # Get value from redis server and set to cache
        # 1. set value to local cache to avoid race condition
        # 2. get value from redis server
        # 3. get expire time from redis server
        # 4. set value to local cache
        # 5. set expire time to local cache

        if value is None:
            async with self.SET_CACHE_LOCK:
                self.SET_CACHE_EVENT.clear()
                self._local_cache[key] = (self.SET_CACHE_PLACEHOLDER, None)
                try:
                    value, ttl = await self._get_from_redis(key)
                except Exception as e:
                    self.flush_key(key)
                    # notify other coroutines waiting for this key, and clear the event
                    self.SET_CACHE_EVENT.set()
                    raise e

                if value is not None: 
                    # Key exist
                    ttl = min(ttl, self.expire_threshold) if ttl >=0 else self.expire_threshold

                    # check if the value is deleted by _listen_invalidate
                    if self._local_cache.get(key, (None, None))[self.VALUE_SLOT] == self.SET_CACHE_PLACEHOLDER:
                        # if it's not deleted, set the value to local cache
                        self._local_cache[key] = (value, int(time.time())+ttl)
                        logging.info(f"Set key to client-side cache: {key}, ttl={ttl}")
                    else:
                        # if it's deleted, flsuh the key from local cache and return the stale value
                        self.flush_key(key)
                else: 
                    # Key not exist
                    self.flush_key(key)
                # notify other coroutines waiting for this key, and clear the event
                self.SET_CACHE_EVENT.set()

        # Ensure that other tasks on the event loop get a chance to run
        # if we didn't have to block for I/O anywhere.
        await asyncio.sleep(0)
        return value
    
    async def _get_from_redis(self, key: str, only_value: bool=False) -> Tuple[Union[None, str, int, float], int]:
        """
        Get key and ttl (optional) from redis server
        This function may raise exceptions
        """
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
    
    def flush_all(self) -> None:
        """
        Flush the whole local cache
        """
        self._local_cache.clear()
        logging.info("Flush ALL client-side cache")
    
    def flush_key(self, key: str) -> None:
        """
        Delete key from local cache
        """
        if key in self._local_cache:
            del self._local_cache[key]
        logging.info(f"Flush key from client-side cache: {key}")
    
    async def _background_listen_invalidate(self) -> None:
        """
        Create another listen invalidate coroutine in case the current connection is broken
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
            logging.info("Exit _background_listen_invalidate")
    
    async def _listen_invalidate_on_open(self) -> None:
        """
        Steps to open listen invalidate coroutine
        1. create pubsub object
        2. get client id
        3. enable client tracking, redirect invalidate message to this connection
        4. subscribe __redis__:invalidate channel
        If any step failed, set self._pubsub_connection to None to trigger a new listen invalidate coroutine
        """
        try:
            # create pubsub object
            self._pubsub = self.pubsub()

            # get client id
            await self._pubsub.execute_command("CLIENT ID")
            self._pubsub_client_id = await self._pubsub.parse_response(block=False, timeout=1)
            if self._pubsub_client_id is None:
                raise Exception(f"CLIENT ID failed. resp={self._pubsub_client_id}")

            # subscribe __redis__:invalidate
            await self._pubsub.subscribe(self.LISTEN_INVALIDATE_CHANNEL)
            resp = await self._pubsub.get_message(timeout=1)
            for k,v in self.SUBSCRIBE_SUCCESS_MSG.items():
                if k not in resp or v != resp[k]:
                    raise Exception(f"SUBCRIBE {self.LISTEN_INVALIDATE_CHANNEL} failed. resp={resp}")

            # client tracking
            resp = await self.client_tracking_on(clientid=self._pubsub_client_id, bcast=True, prefix=self.prefix)
            if resp != b"OK":
                raise Exception(f"CLIENT TRACKING failed. resp={resp}")
            
            # disable built-in health check interval
            self._pubsub.connection.health_check_interval = None
            logging.info(f"Listen invalidate on open success. client_id={self._pubsub_client_id}, channel={self.LISTEN_INVALIDATE_CHANNEL}")
        except Exception as e:
            logging.error(f"Listen invalidate on open failed. error={e}, traceback={traceback.format_exc()}")
            self._pubsub = None
            self._pubsub_client_id = None
    
    async def _listen_invalidate_on_close(self) -> None:
        """
        Steps to close listen invalidate coroutine
        1. flush whole client side cache
        2. disable client tracking
        3. unsubscribe __redis__:invalidate channel
        4. close pubsub connection and release it to connection pool

        This function is called when:
        1. the connection is broken
        2. redis server failed
        3. the client is closed
        """
        try:
            self.flush_all()
            if self._pubsub is not None:
                # client tracking off
                resp = await self.client_tracking_off(clientid=self._pubsub_client_id, bcast=True, prefix=self.prefix)
                if resp != b'OK':
                    raise Exception(f"CLIENT TRACKING off failed. resp={resp}")

                # unsubscribe __redis__:invalidate
                await self._pubsub.unsubscribe(self.LISTEN_INVALIDATE_CHANNEL)
                resp = await self._pubsub.get_message(timeout=1)
                for k,v in self.UNSUBSCRIBE_SUCCESS_MSG.items():
                    if k not in resp or v != resp[k]:
                        raise Exception(f"UNSUBCRIBE {self.LISTEN_INVALIDATE_CHANNEL} failed. resp={resp}")
            logging.info(f"Listen invalidate on close complete. client_id={self._pubsub_client_id}")
        except Exception as e:
            logging.error(f"Listen invalidate on close failed. error={str(e)}, traceback={traceback.format_exc()}")
        finally:
            await self._pubsub.reset()
            self._pubsub = None
            self._pubsub_client_id = None

    async def _listen_invalidate(self) -> None:
        """
        Listen invalidate message from redis server 
        as well as connection health check
        1. if receive a invalidate message, flush the key from local cache
        2. if receive a health check message, update health check status
        TODO: discuss a better timeout value
        """
        await self._listen_invalidate_on_open()
        while signal_state.ALIVE and self._pubsub is not None:
            now = time.time()
            try:
                # check health
                if self.health_check_ongoing_flag:
                    if now-self._last_health_check_time > self.health_check_timeout:
                        raise Exception(f"check health timeout. now={now}, last_health_check_time={self._last_health_check_time}")
                elif now > self._next_health_check_time:
                    await self._pubsub.ping(message=self.HEALTH_CHECK_MSG)
                    self._last_health_check_time = now
                    self._next_health_check_time = now + self.pubsub_health_check_interval
                    self.health_check_ongoing_flag = True
                
                # listen pubsub messages
                resp = await self._pubsub.get_message(timeout=1)

                # message handling
                if resp is None:
                    pass
                elif resp['type'] == 'message':
                    if resp['channel'] == self.LISTEN_INVALIDATE_CHANNEL:
                        if resp['data'] is None:
                            # flushdb or flushall
                            self.flush_all()
                        else:
                            for key in resp['data']:
                                key = key.decode('ascii')
                                self.flush_key(key)
                                logging.info(f"Invalidate key {key} because received invalidate message from redis server")
                elif resp['type'] == 'pong':
                    if resp['data'] == self.HEALTH_CHECK_MSG:
                        self.health_check_ongoing_flag = False
                        logging.info("Receive health check message from redis server")
            except Exception as e:
                logging.error(f"Listen invalidate failed. error={e}, traceback={traceback.format_exc()}")
                break

        await self._listen_invalidate_on_close()
        self.LISTEN_INVALIDATE_COROUTINE_EVENT.set()
    
    async def run(self) -> None:
        task_list = [asyncio.create_task(self._background_listen_invalidate(), name=self.TASK_NAME)]
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