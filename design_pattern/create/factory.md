### Problem:
One product, Several providers, Which one to use?
### Solution:
Create a factory that will provide different product for you. You choose the product.
### When to use:
Complex object creation
### Steps:
1. Declare interfaces for each distinct product of the product family (e.g. Database, Logger, Cache, EmailProtocol, etc.)
2. Declare an abstract factory interface that declares a factory method for each product interface (e.g. createDatabase(), createLogger(), createCache(), createEmailProtocol(), etc.)
3. For each concrete product, declare a class that implements the corresponding product interface (e.g. Database->(MySQL, MongoDB, Postgre, etc), Logger->(FileLogger, etc), Cache->(MemoryCache, RedisCache, DiskCache), EmailProtocol->(IMPA, SMTP, POP, etc))

### Example:
```
Cache():
    get(key)->str
    set(key, value)->bool
    delete(key)->str
```

```
MemoryCache(Cache):
    get(key)->str
        _get_from_memory(key)
    set(key, value)->bool
        _set_to_memory(key, value)
    delete(key)->bool
        _delete_from_memory(key)
```

```
RedisCache(Cache):
    get(key)->str
        _get_from_redis(key)
    set(key, value)->bool
        _set_to_redis(key, value)
    delete(key)->bool
        _delete_from_redis(key)
```

```
DiskCache(Cache):
    get(key)->str
        _get_from_disk(key)
    set(key, value)->bool
        _set_to_disk(key, value)
    delete(key)->bool
        _delete_from_disk(key)
```