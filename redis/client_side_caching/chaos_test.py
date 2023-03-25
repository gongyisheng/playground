import asyncio
import logging

class ChaosMonkey(object):
    def __init__(self, redis):
        self._redis = redis
        self.allowed_methods = ['get', 'set', 'hget', 'hset', 'delete', 'hdel', 'expire', 'flushdb']
    
    async def get(self, key):
        logging.info(f"[ChaosMonkey] get: {key}")
        return await self._redis.get(key)
    
    async def set(self, key, value):
        logging.info(f"[ChaosMonkey] set: {key}={value}")
        return await self._redis.set(key, value)
    
    async def hget(self, key, field):
        logging.info(f"[ChaosMonkey] hget: {key}.{field}")
        return await self._redis.hget(key, field)
    
    async def hset(self, key, field, value):
        logging.info(f"[ChaosMonkey] hset: {key}.{field}={value}")
        return await self._redis.hset(key, field, value)
    
    async def delete(self, key):
        logging.info(f"[ChaosMonkey] delete: {key}")
        return await self._redis.delete(key)
    
    async def hdel(self, key, field):
        logging.info(f"[ChaosMonkey] hdel: {key}.{field}")
        return await self._redis.hdel(key, field)
    
    async def expire(self, key, seconds):
        logging.info(f"[ChaosMonkey] expire: {key}={seconds}")
        return await self._redis.expire(key, seconds)
    
    async def flushdb(self):
        logging.info(f"[ChaosMonkey] flushdb")
        return await self._redis.flushdb()
    
    async def close_client(self):
        logging.info(f"[ChaosMonkey] close_client")
        return await self._redis.close_client()
    
