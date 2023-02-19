### Problem:
How to create an object with many fields and nested objects?
### Solution:
Use builder functions to help create the object, step by step.
### When to use:
Complex object that requires laborious, step-by-step initialization.
### Steps:
1. Set up several builder functions that help initialize one field of the object.
2. (Optional) Create `builder` classes implement the same set of building steps but in a different manner.
3. Execute a series of these steps on a builder object. You can call only those steps that are necessary.
4. (Optional) Extract a series of calls to the builder steps you use to construct a product into a separate `director` class.
### Example:
Naive builder:
```
class RedisCli(object):
    def __init__(self):
        self.host = 'localhost'
        self.port = '6379'
        self.password = ''
        self.db = 0
        self._redis = None
    
    def build_host(self, host):
        self.host = host
    
    def build_port(self, port):
        self.port = port
    
    def build_password(self, password):
        self.password = password
    
    def build_db(self, db):
        self.db = db
    
    def build(self):
        self._redis = redis.StrictRedis(host=self.host, port=self.port, password=self.password, db=self.db)
```
Complex builder:
```
class Builder(object):

    @abstractmethod
    def product(self) -> None:
        pass

class RedisCliBuilder(Builder):

    def __init__(self):
        self.reset()

    def build_host(self, host):
        self.host = host

    def build_port(self, port):
        self.port = port

    def build_password(self, password):
        self.password = password

    def build_db(self, db):
        self.db = db
    
    def reset(self):
        self._redis = None
        self.host = 'localhost'
        self.port = '6379'
        self.password = ''
        self.db = 0
    
    def _build(self):
        self._redis = redis.StrictRedis(host=self.host, port=self.port, password=self.password, db=self.db)

    @property
    def product(self):
        if self._redis is None:
            self._build()
        return self._redis

class RedisCli(object):
    def __init__(self):
        self._redis = None

    def setRedis(self, redis):
        self._redis = redis
    
    def getRedis(self):
        return self._redis

class Director(object):
    
        def __init__(self):
            self._builder = None
    
        @property
        def builder(self):
            return self._builder
    
        @builder.setter
        def builder(self, builder):
            self._builder = builder
    
        def build_product_without_pwd(self):
            self.builder.build_host('localhost')
            self.builder.build_port('6379')
            self.builder.build_db(0)
    
        def build_product_with_pwd(self):
            self.builder.build_host('localhost')
            self.builder.build_port('6379')
            self.builder.build_password('<password>')
            self.builder.build_db(0)
    
        def get_product(self):
            return self.builder.product
```