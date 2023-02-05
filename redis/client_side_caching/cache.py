from collections import OrderedDict

class Cache(OrderedDict):
    'Limit size, evicting the least recently looked-up key when full'

    def __init__(self, manager, maxsize=128, *args, **kwds):
        self.maxsize = maxsize
        self.manager = manager
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        # self.manager.drain()
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.manager.add(key)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]
            self.manager.discard(oldest)