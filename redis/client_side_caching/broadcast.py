import asyncio
import logging
import random
import time
import traceback

import redis.asyncio as aioredis
from lru import LRU
import signal_state_aio as signal_state


# TODO: discuss several timeout/sleep values (search for keyword `timeout` or `sleep`)
# TODO: support hget/hset
# TODO: support set nolppo

# KNOWN ISSUE:
#  1. if redis server closes the connection, the available connection number will -1
#     but the connection pool will not create new connection to replace it.
#     This will cause "No connection available" error if all the connections are closed.
#  2. If redis is shutdown and rebooted, it's recommend to restart the client. 
#     The closed connection in the pool will not be reconnected, and there're chances 
#     to cause "No connection available" error.
class CachedRedis(aioredis.Redis):

    VALUE_SLOT = 0
    EXPIRE_TIME_SLOT = 1
    INSERT_TIME_SLOT = 2

    WRITE_CACHE_LOCK = asyncio.Lock()
    WRITE_CACHE_PLACEHOLDER = object()
    READ_CACHE_EVENT = asyncio.Event()
    RW_CONFLICT_MAX_RETRY = 5

    LISTEN_INVALIDATE_COROUTINE_EVENT = asyncio.Event()

    TASK_NAME = 'task-cached_redis'
    HEALTH_CHECK_MSG = b'cached-redis-py-health-check'
    LISTEN_INVALIDATE_CHANNEL = b"__redis__:invalidate"
    SUBSCRIBE_SUCCESS_MSG = {
        'type':"subscribe", 
        'channel': LISTEN_INVALIDATE_CHANNEL
    }
    UNSUBSCRIBE_SUCCESS_MSG = {
        'type':"unsubscribe",
        'channel': LISTEN_INVALIDATE_CHANNEL,
        'data': 0
    }

    def __init__(self, *args, **kwargs):
        """
        :param cache_prefix: list of key prefix to be cached, empty means cache all keys
        :param cache_noevict_prefix: list of key prefix that can not be evicted from local cache, 
                                    empty means all keys can be evicted
        :param cache_size: max number of keys in local cache
        :param cache_ttl: max time to live for keys in local cache
        :param cache_ttl_deviation: deviation for cache_ttl to avoid all keys expire at the same time, 
                                    should be in [0, 1], 0.01 means 1% deviation of cache_ttl
        :param hget_deviation: deviation for hget, to avoid a lot of pods running hget at the same time
        """

        # PubSub related
        self._pubsub = None
        self._pubsub_client_id = None
        self._pubsub_is_alive = False

        # Cache prefix related
        self.cache_prefix = kwargs.pop("cache_prefix", []) # empty means cache all keys
        self.cache_prefix_tuple = tuple(self.cache_prefix)
        self.cache_noevict_prefix = kwargs.pop("cache_noevict_prefix", []) # empty means all keys can be evicted
        self.cache_noevict_prefix_tuple = tuple(self.cache_noevict_prefix)
        
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
            logging.error("cache_ttl * cache_ttl_deviation is less than 1, \
            please increase cache_ttl or decrease cache_ttl_deviation to avoid cache avalanche")
        self._local_cache = LRU(self.cache_size)
        self._local_dict_cache = {}

        # Listen invalidate related
        self._listen_invalidate_callback = []
        self._listen_invalidate_callback_enabled = False

        # Health check related
        self.pubsub_health_check_interval = kwargs.pop("pubsub_health_check_interval", 60)
        if self.pubsub_health_check_interval <= 0:
            logging.warning("pubsub_health_check_interval should be larger than 0, set to 60")
            self.pubsub_health_check_interval = 60
        self.health_check_ongoing_flag = False
        self.health_check_timeout = 10
        self._last_health_check_time = 0
        self._next_health_check_time = 0
        
        super().__init__(*args, **kwargs)

    async def set(self, key, value):
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

    async def get(self, key):
        """
        Get value from redis server or client side cache.
        1. If value is not in client side cache, get it from redis server and cache it.
        2. If value is in client side cache, return it.
        3. Lock and events are used to prevent race condition.
        4. Keys in local cache may be evicted for two reasons: LRU full or expire time reached.
        """
        value = None

        # If pubsub is not alive, get value from redis server
        _pubsub_skip_flag = not self._pubsub_is_alive
        # If key prefix is not in prefix list, get value from redis server
        _key_prefix_skip_flag = (not self.cache_prefix_tuple) | (not key.startswith(self.cache_prefix_tuple))

        if _pubsub_skip_flag | _key_prefix_skip_flag:
            logging.info(f"Get value from redis server directly. key: {key}, pubsub_skip_flag: {_pubsub_skip_flag}, key_prefix_skip_flag: {_key_prefix_skip_flag}")
            value, _ = await self._get_from_redis(key, only_value=True)
            # Ensure that other tasks on the event loop get a chance to run
            # if we didn't have to block for I/O anywhere.
            await asyncio.sleep(0)
            return value
        
        # Get value from local cache
        # If read-write conflict is heavy, get value from redis server
        value, rw_conflict_fail = await self._get_from_local_cache(key)
        if rw_conflict_fail:
            logging.info(f"Heavy read-write conflict, get value from redis server, key: {key}")
            value, _ = await self._get_from_redis(key, only_value=True)
            return value

        # Get value from redis server and set to cache
        # 1. Dup check that value is not in local cache
        # 2. Set value to placeholder and block other coroutines trying to read the same key
        # 3. Get value and ttl from redis server
        # 4. Check whether the value is deleted by _listen_invalidate
        # 5. Set key, value and expire time to local cache
        # 6. Notify other coroutines waiting for this key
        if value is None:
            async with self.WRITE_CACHE_LOCK:
                # Dup check from memory cache
                value, _ = await self._get_from_local_cache(key)
                if value is None:
                    # Block other coroutines trying to read the same key
                    self.READ_CACHE_EVENT.clear()
                    self._local_cache[key] = (self.WRITE_CACHE_PLACEHOLDER, None)
                    try:
                        value, ttl = await self._get_from_redis(key)
                        self._set_to_local_cache(key, value, ttl)
                    except Exception as e:
                        self.flush_key(key)
                        raise e
                    finally:
                        # Notify other coroutines waiting for this key
                        self.READ_CACHE_EVENT.set()

        # Ensure that other tasks on the event loop get a chance to run
        # If we didn't have to block for I/O anywhere.
        await asyncio.sleep(0)
        return value
    
    async def _get_from_redis(self, key, only_value=False):
        """
        Get key and ttl (optional) from redis server
        This function may raise exceptions
        """
        value = ttl = None
        if only_value:
            value = await super().get(key)
        else:
            # Use pipeline to execute a transaction
            pipe = super().pipeline()
            pipe.get(key)
            pipe.ttl(key)
            value, ttl = await pipe.execute()
        logging.info(f"Get key from redis server: {key}, ttl={ttl}, only_value={only_value}")
        return value, ttl
    
    async def _get_from_local_cache(self, key):
        """
        Get value from local cache
        1. Check if key exist in local cache. 
           If it's hold by placeholder, wait for READ_CACHE_EVENT
        2. Check if key expire time is not reached
        """
        value = None
        cache = None

        # choose cache, either lru cache or dict cache
        if key in self._local_cache:
            cache = self._local_cache
        elif key in self._local_dict_cache:
            cache = self._local_dict_cache
        else:
            return None, False
        
        _retry = self.RW_CONFLICT_MAX_RETRY
        while cache[key][self.VALUE_SLOT] == self.WRITE_CACHE_PLACEHOLDER:
            await self.READ_CACHE_EVENT.wait()
            _retry -= 1
            if key not in cache:
                logging.info(f"Key becomes invalid after waiting for READ_CACHE_EVENT: {key}")
                return None, False
            if _retry <= 0:
                logging.info(f"Heavy read-write conflict, retry times exceeded: {key}")
                return None, True

        _value, expire_time, insert_time = cache[key]
        if time.time() <= expire_time:
            value = _value
            logging.info(f"Get key from client-side cache: {key}, expire_time={expire_time}, insert_time={insert_time}")
        else:
            logging.info(f"Key exists in clien-side cache but expired: {key}, expire_time={expire_time}, insert_time={insert_time}")
        return value, False
    
    def _hget_from_local_cache(self, key, field):
        """
        Get value associated with field in hash stored at key from local cache
        """ 
        _value, rw_conflict_fail = self._get_from_local_cache(key)
        if rw_conflict_fail | (_value is None):
            return None, rw_conflict_fail
        else:
            return _value.get(field, None), rw_conflict_fail
    
    def _set_to_local_cache(self, key, value, ttl):
        """
        Set key, value and expire time to local cache
        """
        # Key exist
        if value is not None: 
            # If the key has ttl in redis server and less than or equal to cache_ttl, 
            # use it to keep the expire time consistent. Otherwise, use cache_ttl and 
            # make it a little bit random to avoid cache avalanche.
            if ttl < 0 and ttl > self.cache_ttl:
                ttl = self.cache_ttl - random.random() * self.cache_ttl_max_deviation

            # Check if the value is deleted by _listen_invalidate
            if self._local_cache.get(key, (None, None, None))[self.VALUE_SLOT] == self.WRITE_CACHE_PLACEHOLDER and self._pubsub_is_alive:
                # If it's not deleted, set the value to local cache
                insert_time = time.time()
                self._local_cache[key] = (value, insert_time+ttl, insert_time)
                logging.info(f"Set key to client-side cache: {key}, ttl={ttl}, insert_time={insert_time}")
            else:
                # If it's deleted, flush the key from local cache and return the stale value
                self.flush_key(key)
                logging.info(f"Key becomes invalid before set to cache, return the stale value: {key}")
        else: 
            # Key not exist
            self.flush_key(key)
            logging.info(f"Key not exist in redis server: {key}")
    
    def flush_all(self):
        """
        Flush the whole local cache
        """
        self._local_cache.clear()
        logging.info("Flush ALL client-side cache")
    
    def flush_key(self, key: str):
        """
        Delete key from local cache
        """
        if key in self._local_cache:
            del self._local_cache[key]
        logging.info(f"Flush key from client-side cache: {key}")
    
    async def _background_listen_invalidate(self):
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
    
    async def _listen_invalidate_on_open(self):
        """
        Steps to open listen invalidate coroutine
        1. Create pubsub object
        2. Get client id
        3. Enable client tracking, redirect invalidate message to this connection
        4. Subscribe __redis__:invalidate channel
        If any step failed, set self._pubsub_is_alive to None to trigger a new listen invalidate coroutine
        """
        try:
            # Create pubsub object
            self._pubsub = self.pubsub()

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
            resp = await self.client_tracking_on(clientid=self._pubsub_client_id, bcast=True, prefix=self.cache_prefix)
            if resp != b"OK":
                raise Exception(f"CLIENT TRACKING failed. resp={resp}")
            
            # Disable built-in health check interval
            self._pubsub.connection.health_check_interval = None
            self._pubsub_is_alive = True
            logging.info(f"Listen invalidate on open success. client_id={self._pubsub_client_id}, channel={self.LISTEN_INVALIDATE_CHANNEL}")
        except Exception as e:
            logging.error(f"Listen invalidate on open failed. error={e}, traceback={traceback.format_exc()}")
            self._pubsub = None
            self._pubsub_client_id = None
            self._pubsub_is_alive = False
    
    async def _listen_invalidate_on_close(self):
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
        try:
            self.flush_all()
            if self._pubsub_is_alive:
                # Client tracking off
                resp = await self.client_tracking_off(clientid=self._pubsub_client_id, bcast=True, prefix=self.cache_prefix)
                if resp != b'OK':
                    raise Exception(f"CLIENT TRACKING off failed. resp={resp}")

                # Unsubscribe __redis__:invalidate
                await self._pubsub.unsubscribe(self.LISTEN_INVALIDATE_CHANNEL)
                resp = await self._pubsub.get_message(timeout=1)
                while resp is not None:
                    if resp['type'] != 'unsubscribe':
                        # Read out other messages that are left in the channel
                        resp = await self._pubsub.get_message(timeout=1)
                    else:
                        for k,v in self.UNSUBSCRIBE_SUCCESS_MSG.items():
                            if k not in resp or v != resp[k]:
                                raise Exception(f"UNSUBCRIBE {self.LISTEN_INVALIDATE_CHANNEL} failed. resp={resp}")
                        break
                logging.info(f"Listen invalidate on close complete. client_id={self._pubsub_client_id}")
            else:
                logging.info("Listen invalidate on close skipped. _pubsub_is_alive=False")
        except Exception as e:
            logging.warning(f"Listen invalidate on close failed. error={e}, traceback={traceback.format_exc()}")
        finally:
            self._pubsub = None
            self._pubsub_client_id = None
            logging.info("PubSub reset complete")

    async def _listen_invalidate(self):
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
                        raise Exception(f"check health timeout. now={now}, last_health_check_time={self._last_health_check_time}")
                elif now > self._next_health_check_time:
                    await self._pubsub.ping(message=self.HEALTH_CHECK_MSG)
                    self._last_health_check_time = now
                    self._next_health_check_time = now + self.pubsub_health_check_interval
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
                        logging.info("Receive health check message from redis server, interval=%.3fms" % ((now-self._last_health_check_time)*1000))
            except Exception as e:
                # If any exception occurs, set _pubsub_is_alive to False
                self._pubsub_is_alive = False
                logging.error(f"Listen invalidate failed. error={e}, traceback={traceback.format_exc()}")
                break

        await self._listen_invalidate_on_close()
        self.LISTEN_INVALIDATE_COROUTINE_EVENT.set()
    
    def register_listen_invalidate_callback(self, func, *args, **kwargs):
        """
        Register callback function for listen invalidate
        """
        self._listen_invalidate_callback_enabled = True
        self._listen_invalidate_callback.append([func, args, kwargs])
    
    def run_listen_invalidate_callback(self, message_resp):
        """
        Run all callback functions
        """
        if self._listen_invalidate_callback_enabled:
            for func, args, kwargs in self._listen_invalidate_callback:
                func(message_resp, *args, **kwargs)
    
    async def run(self):
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