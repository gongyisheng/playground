import asyncio
import logging
import random
import signal_state_aio as signal_state

class BaseChaosMonkey(object):
    TYPE = "BASE"
    METHOD_FREQ_MAP = {
        'get': 10,
        'set': 10,
        'hget': 10, 
        'hset': 10,
        'delete': 10,
        'hdel': 10,
        'expire': 10,
        'flushdb': 1,
    }
    KEY_LIST = []

    def __init__(self, redis, sleep_time=0.1):
        self._redis = redis
        self._sleep_time = sleep_time
    
    async def get(self, key):
        logging.info(f"[ChaosMonkey({self.TYPE})] get: {key}")
        return await self._redis.get(key)
    
    async def set(self, key, value):
        logging.info(f"[ChaosMonkey({self.TYPE})] set: {key}={value}")
        return await self._redis.set(key, value)
    
    async def hget(self, key, field):
        logging.info(f"[ChaosMonkey({self.TYPE})] hget: {key}.{field}")
        return await self._redis.hget(key, field)
    
    async def hset(self, key, field, value):
        logging.info(f"[ChaosMonkey({self.TYPE})] hset: {key}.{field}={value}")
        return await self._redis.hset(key, field, value)
    
    async def delete(self, key):
        logging.info(f"[ChaosMonkey({self.TYPE})] delete: {key}")
        return await self._redis.delete(key)
    
    async def hdel(self, key, field):
        logging.info(f"[ChaosMonkey({self.TYPE})] hdel: {key}.{field}")
        return await self._redis.hdel(key, field)
    
    async def expire(self, key, seconds):
        logging.info(f"[ChaosMonkey({self.TYPE})] expire: {key}={seconds}")
        return await self._redis.expire(key, seconds)
    
    async def flushdb(self):
        logging.info(f"[ChaosMonkey({self.TYPE})] flushdb")
        return await self._redis.flushdb()
    
    # async def close_client(self):
    #     logging.info(f"[ChaosMonkey({self.TYPE})] close_client")
    #     return await self._redis.close_client()
    
    async def run(self):
        method_list = []
        for method, freq in self.METHOD_FREQ_MAP.items():
            method_list.extend([method] * freq)
        
        while signal_state.ALIVE:
            try:
                method = method_list[random.randint(0, len(method_list) - 1)]
                await getattr(self, method)()
            except Exception as e:
                logging.error(f"[ChaosMonkey({self.TYPE})] execute method failed. Error={e}")
            finally:
                await asyncio.sleep(self._sleep_time)

        
    
