### Problem:
How to control how many instances a class has?
How to control the access to a global class to prevent dirty read/write?
### Solution:
Ensure a class has only one instance. Provide a global point of access to it.
### When to use:
Global variable, External service connection (redis, db).
### Steps:
1. Make the default constructor private, to prevent other objects from using the new operator with the Singleton class
2. Create a static creation method that acts as a constructor. Under the hood, this method calls the private constructor to create an object and saves it in a static field. All following calls to this method return the cached object.

### Example:
Naive Singleton:
```
class RedisCli(object):

    _instances = {}

    def __new__(cls, *args, **kwargs):
        # You can also define customized key for each instance here
        if cls not in cls._instances:
            cls._instances[cls] = super(RedisCli, cls).__new__(cls)
        return cls._instances[cls]

    def __init__(self, *args, **kwargs):
        pass
```

Thread-Safe Singleton:
```
from threading import Lock

class RedisCli(object):

    _instances = {}
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            # You can also define customized key for each instance here
            if cls not in cls._instances:
                cls._instances[cls] = super(RedisCli, cls).__new__(cls)
            return cls._instances[cls]

    def __init__(self, *args, **kwargs):
        pass
```
Note: `__new__` is classmethod. super(RedisCli, cls) calls the parent class, which is python built-in object. 