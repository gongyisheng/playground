'''
A local cache-aware Redis client
'''
import redis

class Redis(redis.Redis):
    def __init__(self, manager, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self._manager = manager
        self._client_id = super().client_id()
        self.execute_command('CLIENT', 'TRACKING', 'ON',
            'redirect', self._manager.client_id)

    def close(self):
        self.execute_command('CLIENT', 'TRACKING', 'OFF')
        super().close()

    def get(self, name):
        try:
            value = self._manager.cache[name]
        except KeyError:
            value = super().get(name)
            self._manager.cache[name] = value
        return value