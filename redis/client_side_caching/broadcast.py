import asyncio
import logging
import random
import string
import time
from typing import Callable, Optional, Union, Tuple

from redis import asyncio as aioredis
from lru import LRU
import signal_state_aio as signal_state

# KNOWN ISSUE:
#  1. If redis server closes the connection, the available connection number will -1
#     but the connection pool will not create new connection to replace it.
#     This will cause "No connection available" error if all the connections are closed.
#  2. If redis is shutdown and rebooted, it's recommend to restart the client. 
#     The closed connection in the pool will not be reconnected, and there're chances 
#     to cause "No connection available" error.
#  3. One connection will be used for subscribing invalidation channel, make sure there're
#     enough connections in the pool (at least 2 available connections) otherwise it will
#     raise "No connection available" error.
#  4. CachedRedis does not support redis cluster for now.
class CachedRedis(object):

    VALUE_SLOT = 0
    EXPIRE_TIME_SLOT = 1
    INSERT_TIME_SLOT = 2

    WRITE_CACHE_LOCK = None
    WRITE_CACHE_PLACEHOLDER = object()
    READ_CACHE_EVENT = None
    RW_CONFLICT_MAX_RETRY = 5

    HASHKEY_PREFIX = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(16))

    LISTEN_INVALIDATE_COROUTINE_EVENT = None

    TASK_NAME = 'task-cached_redis'
    HEALTH_CHECK_MSG = b'cached-redis-py-health-check'
    LISTEN_INVALIDATE_CHANNEL = b"__redis__:invalidate"
    SUBSCRIBE_SUCCESS_MSG = {
        'type':"subscribe", 
        'channel': LISTEN_INVALIDATE_CHANNEL
    }

    def __init__(self, redis: aioredis.Redis, **kwargs):
        """
        :param cache_prefix: list of key prefix to be cached, empty list is not allowed
        :param cache_noevict_prefix: list of key prefix that can not be evicted from local cache, 
                                    empty means all keys can be evicted
        :param cache_size: max number of keys in local cache
        :param cache_ttl: max time to live for keys in local cache
        :param cache_ttl_deviation: deviation for cache_ttl to avoid all keys expire at the same time, 
                                    should be in [0, 1], 0.01 means 1% deviation of cache_ttl
        :param hget_deviation_option: deviation for hget, to avoid a lot of pods running hget at the same time
        :param health_check_interval: interval for health check, default is 60s
        """
        def _validate_option(_dict: dict):
            for k, v in _dict.items():
                if type(k) is not str:
                    raise TypeError("get_deviation_option key should be str")
                if type(v) is not float and type(v) is not int:
                    raise TypeError("get_deviation_option value should be float or int")
                if v <= 0:
                    raise ValueError("get_deviation_option value should be larger than 0")

        self._redis = redis
        if self._redis is None:
            raise ValueError("Can't init cached redis because redis is None")

        # PubSub related
        self._pubsub = self._redis.pubsub()
        self._pubsub_client_id = None
        self._pubsub_is_alive = False

        # Cache prefix related
        self.cache_prefix = kwargs.pop("cache_prefix", []) # empty means cache NO keys
        self.cache_prefix_tuple = tuple(self.cache_prefix)
        if len(self.cache_prefix) == 0:
            raise ValueError("cache_prefix is empty, please specify the key prefix to be cached")
        self.cache_noevict_prefix = kwargs.pop("cache_noevict_prefix", []) # empty means all keys can be evicted
        self.cache_noevict_prefix_tuple = tuple([f"{self.HASHKEY_PREFIX}:{p}" for p in self.cache_noevict_prefix] + self.cache_noevict_prefix)
        
        # Local cache related
        self.cache_size = kwargs.pop("cache_size", 10000)

        self.cache_ttl = kwargs.pop("cache_ttl", 86400)
        if self.cache_ttl <= 0:
            logging.warning("cache_ttl should be larger than 0, set to 86400")
            self.cache_ttl = 86400

        self.cache_ttl_deviation = kwargs.pop("cache_ttl_deviation", 0.01)
        if self.cache_ttl_deviation < 0 or self.cache_ttl_deviation > 1:
            logging.warning("cache_ttl_deviation should be in [0, 1], set to 0.01")
            self.cache_ttl_deviation = 0.01

        self.cache_ttl_max_deviation = self.cache_ttl * self.cache_ttl_deviation
        if self.cache_ttl_max_deviation < 1:
            logging.error("cache_ttl * cache_ttl_deviation is less than 1, please increase cache_ttl or cache_ttl_deviation to avoid cache avalanche")

        self.hget_deviation_option = kwargs.pop("hget_deviation_option", {})
        _validate_option(self.hget_deviation_option)

        self._local_lru_cache = LRU(self.cache_size)
        self._local_noevict_cache = {}
        self._hashkey_field_map = {} # key -> [hashkey_prefix:key:field] metadata, for listen invalidate of hashkey

        # Listen invalidate related
        self._listen_invalidate_callback = []
        self._listen_invalidate_callback_enabled = False

        # Health check related
        self.health_check_interval = kwargs.pop("health_check_interval", 60)
        if self.health_check_interval <= 0:
            logging.warning("health_check_interval should be larger than 0, set to 60")
            self.health_check_interval = 60

        self.health_check_ongoing_flag = False
        self.health_check_timeout = 10
        self._last_health_check_time = 0
        self._next_health_check_time = 0
    
    def __get_write_cache_lock(self):
        if self.WRITE_CACHE_LOCK is None:
            self.WRITE_CACHE_LOCK = asyncio.Lock()
        return self.WRITE_CACHE_LOCK
    
    def __get_read_cache_event(self):
        if self.READ_CACHE_EVENT is None:
            self.READ_CACHE_EVENT = asyncio.Event()
        return self.READ_CACHE_EVENT
    
    def _make_cache_key(self, key: str, field: Optional[str] = None) -> str:
        if field is None:
            return key
        else:
            return f"{self.HASHKEY_PREFIX}:{key}:{field}"
    
    def _choose_cache(self, key: str) -> dict:
        if key.startswith(self.cache_noevict_prefix_tuple):
            cache = self._local_noevict_cache
        else:
            cache = self._local_lru_cache
        return cache
    
    async def get(self, key: str):
        logging.debug(f"Get key={key}")
        return await self._get(key)

    async def hget(self, key: str, field: str):
        logging.debug(f"Hget key={key} field={field}")
        return await self._get(key, field=field)
    
    async def set(self, *args, **kwargs):
        return await self._redis.set(*args, **kwargs)

    async def hset(self, *args, **kwargs):
        return await self._redis.hset(*args, **kwargs)
    
    async def setnx(self, *args, **kwargs):
        return await self._redis.setnx(*args, **kwargs)
    
    async def expire(self, *args, **kwargs):
        return await self._redis.expire(*args, **kwargs)

    async def _get(self, key: str, field: Optional[str] = None):
        """
        Generalized get function to retrive value from redis server or client side cache.
        1. If value is not in client side cache, get it from redis server and cache it, then return it.
        2. If value is in client side cache, return it.
        3. Lock and events are used to prevent race condition.
        4. Keys in cache may be removed for two reasons: LRU full or expire time reached.
        """
        
        gmode = "hget" if field else "get"

        redis_getter = getattr(self, f"_{gmode}_from_redis")
        cache_getter = getattr(self, f"_{gmode}_from_cache")
            
        value = None

        # If pubsub is not alive, get value from redis server
        _pubsub_skip_flag = not self._pubsub_is_alive
        # If key prefix is not in prefix list, get value from redis server
        _key_prefix_skip_flag = (len(self.cache_prefix_tuple) > 0) & (not key.startswith(self.cache_prefix_tuple))

        if _pubsub_skip_flag | _key_prefix_skip_flag:
            logging.debug(f"Get value from redis server directly. key: {key}, pubsub_skip_flag: {_pubsub_skip_flag}, key_prefix_skip_flag: {_key_prefix_skip_flag}")
            value, _ = await redis_getter(key, field=field, with_ttl=False)
            # Ensure that other tasks on the event loop get a chance to run
            # if we didn't have to block for I/O anywhere.
            await asyncio.sleep(0)
            return value
        
        cache = self._choose_cache(key)
        # Get value from local cache
        # If read-write conflict is heavy, get value from redis server
        value, rw_conflict_fail = await cache_getter(cache, key, field=field)
        if rw_conflict_fail:
            logging.warning(f"Heavy read-write conflict, get value from redis server, key: {key}")
            value, _ = await redis_getter(key, with_ttl=False, field=field)
            return value

        # Get value from redis server and set to cache
        # 1. Dup check that value is not in local cache
        # 2. Set value to placeholder and block other coroutines trying to read the same key
        # 3. Get value and ttl from redis server
        # 4. Check whether the value is deleted by _listen_invalidate
        # 5. Set key, value and expire time to local cache
        # 6. Notify other coroutines waiting for this key
        if value is None:
            async with self.__get_write_cache_lock():
                # Dup check from memory cache
                value, _ = await cache_getter(cache, key, field=field)
                if value is None:
                    value = await self._cache_key(cache, key, field=field)

        # Ensure that other tasks on the event loop get a chance to run
        # If we didn't have to block for I/O anywhere.
        await asyncio.sleep(0)
        return value
    
    async def _get_from_redis(self, key: str, with_ttl: bool = True, **kwargs) -> Tuple[Union[bytes, str], int]:
        """
        Get key and ttl (optional) from redis server
        This function may raise exceptions
        """
        value = ttl = None
        if with_ttl:
            # Use pipeline to execute a transaction
            pipe = self._redis.pipeline()
            pipe.get(key)
            pipe.ttl(key)
            value, ttl = await pipe.execute()
        else:
            value = await self._redis.get(key)
        logging.debug(f"Get key from redis server: {key}, ttl={ttl}")
        return value, ttl
    
    async def _hget_from_redis(self, key: str, field: str, with_ttl: bool = True) -> Tuple[Union[bytes, str], int]:
        """
        Hget key, field from redis server
        This function may raise exceptions
        """
        value = ttl = None

        if key in self.hget_deviation_option:
            deviation = random.random() * self.hget_deviation_option[key]
            await asyncio.sleep(deviation)

        if with_ttl:
            # Use pipeline to execute a transaction
            pipe = self._redis.pipeline()
            pipe.hget(key, field)
            pipe.ttl(key)
            value, ttl = await pipe.execute()
        else:
            value = await self._redis.hget(key, field)
        logging.debug(f"Hget key, field from redis server: {key}, {field} ttl={ttl}")
        return value, ttl
    
    async def _get_from_cache(self, cache: dict, key: str, **kwargs) -> Tuple[Union[bytes, str], bool]:
        """
        Get value from local cache
        1. Check if key exist in local cache. 
           If it's hold by placeholder, wait for READ_CACHE_EVENT
        2. Check if key expire time is not reached
        """
        value = None

        if key not in cache:
            return None, False

        _retry = self.RW_CONFLICT_MAX_RETRY
        while cache[key][self.VALUE_SLOT] == self.WRITE_CACHE_PLACEHOLDER:
            await self.__get_read_cache_event().wait()
            _retry -= 1
            if key not in cache:
                logging.debug(f"Key becomes invalid after waiting for READ_CACHE_EVENT: {key}")
                return None, False
            if _retry <= 0:
                logging.debug(f"Heavy read-write conflict, retry times exceeded: {key}")
                return None, True

        _value, expire_time, insert_time = cache[key]
        if time.time() <= expire_time:
            value = _value
            logging.debug(f"Get key from client-side cache: {key}, expire_time={expire_time}, insert_time={insert_time}")
        else:
            self.flush_key(key)
            logging.debug(f"Key exists in clien-side cache but expired: {key}, expire_time={expire_time}, insert_time={insert_time}")
        return value, False
    
    async def _hget_from_cache(self, cache: dict, key: str, field: str) -> Tuple[Union[bytes, str], bool]:
        """
        Get value associated with field in hash stored at key from local cache
        """ 
        cache_key = self._make_cache_key(key, field)
        _value, rw_conflict_fail = await self._get_from_cache(cache, cache_key)
        if rw_conflict_fail | (_value is None):
            return None, rw_conflict_fail
        else:
            return _value, rw_conflict_fail
    
    def _set_to_cache(self, cache: dict, key: str, value: str, ttl: Union[int, float], **kwargs) -> None:
        """
        Set key, value and expire time to local cache
        """
        # Key exist
        if value is not None: 
            # If the key has ttl in redis server and less than or equal to cache_ttl, 
            # use it to keep the expire time consistent. Otherwise, use cache_ttl and 
            # make it a little bit random to avoid cache avalanche.
            if ttl < 0 or ttl > self.cache_ttl:
                ttl = self.cache_ttl - random.random() * self.cache_ttl_max_deviation

            # Check if the value is deleted by _listen_invalidate
            if cache.get(key, (None, None, None))[self.VALUE_SLOT] == self.WRITE_CACHE_PLACEHOLDER and self._pubsub_is_alive:
                # If it's not deleted, set the value to local cache
                insert_time = time.time()
                cache[key] = (value, insert_time+ttl, insert_time)
                logging.debug(f"Set key to client-side cache: {key}, ttl={ttl}, insert_time={insert_time}")
            else:
                # If it's deleted, flush the key from local cache and return the stale value
                self.flush_key(key)
                logging.debug(f"Key becomes invalid before set to cache, return the stale value: {key}")
        else: 
            # Key not exist
            self.flush_key(key)
            logging.debug(f"Key not exist in redis server: {key}")

    def _hset_to_cache(self, cache: dict, key: str, value: str, ttl: Union[int, float], field: str) -> None:
        """
        Set hashed key-field, value and expire time to local cache
        """
        cache_key = self._make_cache_key(key, field)
        if key not in self._hashkey_field_map:
            self._hashkey_field_map[key] = set()
        self._hashkey_field_map[key].add(cache_key)
        self._set_to_cache(cache, cache_key, value, ttl)
    
    async def _cache_key(self, cache: dict, key: str, field: Optional[str] = None) -> Optional[str]:

        gmode = "hget" if field else "get"
        smode = "hset" if field else "set"

        redis_getter = getattr(self, f"_{gmode}_from_redis")
        cache_setter = getattr(self, f"_{smode}_to_cache")
        
        cache_key = self._make_cache_key(key, field)

        # Block other coroutines trying to read the same key
        self.__get_read_cache_event().clear()
        cache[cache_key] = (self.WRITE_CACHE_PLACEHOLDER, None)
        try:
            value, ttl = await redis_getter(key, field=field)
            cache_setter(cache, key, value, ttl, field=field)
        except Exception as e:
            self.flush_key(key)
            raise e
        finally:
            # Notify other coroutines waiting for this key
            self.__get_read_cache_event().set()
        return value
    
    def flush_all(self) -> None:
        """
        Flush the whole local cache
        """
        self._local_lru_cache.clear()
        self._local_noevict_cache.clear()
        logging.info("Flush ALL client-side cache")
    
    def flush_key(self, key: str) -> None:
        """
        Delete key from local cache
        """
        cache = self._choose_cache(key)

        # string key
        if key in cache:
            del cache[key]

        # hash key
        if key in self._hashkey_field_map:
            for cache_key in self._hashkey_field_map[key]:
                if cache_key in cache:
                    del cache[cache_key]
            del self._hashkey_field_map[key]

        logging.info(f"Flush key from client-side cache: {key}")
    
    async def _background_listen_invalidate(self) -> None:
        """
        Create another listen invalidate coroutine in case the current connection is broken
        """
        logging.info("Start _background_listen_invalidate")
        while signal_state.ALIVE:
            try:
                await asyncio.gather(asyncio.create_task(self._listen_invalidate(), name=self.TASK_NAME))
            except Exception as e:
                logging.error(f"Error in _background_listen_invalidate. error={e}", exc_info=True)
            finally:
                await asyncio.sleep(1)
        logging.info("Exit _background_listen_invalidate")
    
    async def _listen_invalidate_on_open(self) -> None:
        """
        Steps to open listen invalidate coroutine
        1. Create pubsub object
        2. Get client id
        3. Enable client tracking, redirect invalidate message to this connection
        4. Subscribe __redis__:invalidate channel
        If any step failed, set self._pubsub_is_alive to None to trigger a new listen invalidate coroutine
        """
        try:
            # Get client id
            await self._pubsub.execute_command("CLIENT ID")
            self._pubsub_client_id = await self._pubsub.parse_response(block=False, timeout=1)
            if self._pubsub_client_id is None:
                raise Exception(f"CLIENT ID failed. resp={self._pubsub_client_id}")

            # Subscribe __redis__:invalidate
            await self._pubsub.subscribe(self.LISTEN_INVALIDATE_CHANNEL)
            resp = await self._pubsub.get_message(timeout=1)
            for k,v in self.SUBSCRIBE_SUCCESS_MSG.items():
                if k not in resp or v != resp[k]:
                    raise Exception(f"SUBCRIBE {self.LISTEN_INVALIDATE_CHANNEL} failed. resp={resp}")

            # Client tracking
            resp = await self._redis.client_tracking_on(clientid=self._pubsub_client_id, bcast=True, prefix=self.cache_prefix)
            if resp != b"OK":
                raise Exception(f"CLIENT TRACKING ON failed. resp={resp}")
            
            # Disable built-in health check interval
            self._pubsub.connection.health_check_interval = None
            self._pubsub_is_alive = True
            logging.info(f"Listen invalidate on open success. client_id={self._pubsub_client_id}, channel={self.LISTEN_INVALIDATE_CHANNEL}")
        except Exception as e:
            logging.error(f"Listen invalidate on open failed. error={e}", exc_info=True)
            self._pubsub_is_alive = False
    
    async def _listen_invalidate_on_close(self) -> None:
        """
        Steps to close listen invalidate coroutine
        1. Flush whole client side cache
        2. Disable client tracking
        3. Unsubscribe __redis__:invalidate channel
        4. Close pubsub connection and release it to connection pool

        This function is called when:
        1. The connection is broken
        2. Redis server failed
        3. The client is closed
        """
        # Flush whole client side cache
        # Set _pubsub_is_alive to false to prevent read/write message from local cache
        self.flush_all()
        # Client tracking off
        try:
            resp = await self._redis.client_tracking_off(bcast=True, prefix=self.cache_prefix)
            if resp != b'OK':
                raise Exception(f"CLIENT TRACKING OFF resp is not OK. resp={resp}")
        except Exception as e:
            logging.info(f"CLIENT TRACKING OFF failed. error={e}", exc_info=True)

        # Unsubscribe __redis__:invalidate and reset connection
        try:
            await self._pubsub.reset()
            logging.info(f"Pubsub reset complete.")
        except Exception as e:
            logging.info(f"Pubsub reset complete with error. error={e}", exc_info=True)

        logging.info(f"Listen invalidate on close complete. client_id={self._pubsub_client_id}")
        self._pubsub_client_id = None
        self._pubsub_is_alive = False
        self.health_check_ongoing_flag = False
        self._last_health_check_time = 0
        self._next_health_check_time = 0

    async def _listen_invalidate(self) -> None:
        """
        Listen invalidate message from redis server 
        as well as connection health check
        1. If receive a invalidate message, flush the key from local cache
        2. If receive a health check message, update health check status
        TODO: discuss a better timeout value
        """
        await self._listen_invalidate_on_open()
        while signal_state.ALIVE and self._pubsub_is_alive:
            now = time.time()
            try:
                # Health check
                if self.health_check_ongoing_flag:
                    if now-self._last_health_check_time > self.health_check_timeout:
                        raise Exception(f"health check timeout. now={now}, last_health_check_time={self._last_health_check_time}")
                elif now > self._next_health_check_time:
                    await self._pubsub.ping(message=self.HEALTH_CHECK_MSG)
                    self._last_health_check_time = now
                    self._next_health_check_time = now + self.health_check_interval
                    self.health_check_ongoing_flag = True
                
                # Listen pubsub messages
                resp = await self._pubsub.get_message(timeout=1)

                # Message handling
                if resp is None:
                    pass
                elif resp['type'] == 'message':
                    if resp['channel'] == self.LISTEN_INVALIDATE_CHANNEL:
                        if resp['data'] is None:
                            # Flushdb or Flushall
                            self.flush_all()
                        else:
                            for key in resp['data']:
                                key = key.decode('ascii')
                                self.flush_key(key)
                                logging.info(f"Invalidate key {key} because received invalidate message from redis server")
                    self.run_listen_invalidate_callback(resp)
                elif resp['type'] == 'pong':
                    if resp['data'] == self.HEALTH_CHECK_MSG:
                        self.health_check_ongoing_flag = False
                        logging.info("Receive health check message from redis server, interval=%.3fms" % ((time.time()-self._last_health_check_time)*1000))
            except Exception as e:
                # If any exception occurs, set _pubsub_is_alive to False
                self._pubsub_is_alive = False
                logging.error(f"Listen invalidate failed. error={e}", exc_info=True)
                break

        await self._listen_invalidate_on_close()
    
    def register_listen_invalidate_callback(self, func: Callable, *args, **kwargs) -> None:
        """
        Register callback function for listen invalidate
        """
        self._listen_invalidate_callback_enabled = True
        self._listen_invalidate_callback.append([func, args, kwargs])
    
    def run_listen_invalidate_callback(self, message: dict) -> None:
        """
        Run all callback functions
        """
        if self._listen_invalidate_callback_enabled:
            for func, args, kwargs in self._listen_invalidate_callback:
                func(message=message, *args, **kwargs)
    
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
            logging.error(f"CachedRedis task exit error. error={e}", exc_info=True)