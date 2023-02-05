import redis
from collections import defaultdict
from client import Redis as CachedRedis
from cache import Cache
from crc import crc64

class Manager(object):
    ''' Redis Assisted Client Side Cache Manager '''
    def __init__(self : object, pool : redis.ConnectionPool, capacity : int = 128, sleep_time : int = 0):
        self.pool = pool
        self.capacity = capacity
        self.sleep_time = sleep_time
        self.client = redis.Redis(connection_pool=self.pool)
        self.client_id = None
        self.reset()
        self.start()

    def __del__(self):
        self.stop()

    def _handler(self, message):
        ''' Handles invalidation messages '''
        slot = message['data']
        self.invalidate(slot)

    def reset(self):
        ''' Resets the manager '''
        self.slots = defaultdict(set)
        self.cache = Cache(self, maxsize=self.capacity)

    def start(self):
        ''' Starts the manager '''
        self.client_id = self.client.client_id()
        self._pubsub = self.client.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(**{'__redis__:invalidate': self._handler})
        self._thread = self._pubsub.run_in_thread(sleep_time=self.sleep_time)

    def stop(self):
        ''' Stops the manager '''
        if self.client_id is not None:
            self._thread.stop()
            self.client_id = None

    @staticmethod
    def slot(key):
        ''' Returns the slot for a key '''
        crc = crc64(key)
        crc &= 0xffffff
        return crc

    def add(self, key):
        ''' Adds a key to the internal tracking table '''
        slot = self.slot(key)
        self.slots[slot].add(key)

    def discard(self, key):
        ''' Removes a key from the internal tracking table '''
        slot = self.slot(key)
        self.slots[slot].discard(key)

    def invalidate(self, slot):
        ''' Invalidates a slot's keys '''
        slot = int(slot)
        while self.slots[slot]:
           key = self.slots[slot].pop()
           del self.cache[key]

    def get_connection(self, *args, **kwargs):
        ''' Returns a cached Redis connection '''
        conn = CachedRedis(self, connection_pool=self.pool, *args, **kwargs)
        return conn