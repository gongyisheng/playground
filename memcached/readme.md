# Memcached Documentation

## Overview

Memcached is a light-weight, high-performance, distributed memory kv storange system. It supports multi thread natively. It has performance better than redis.

## Key Features
- in-memory KV storage, no disk persistence, no auth
- logic half in server, half in client
- servers are disconnected from each other
- only support lightweight O(1) operations
- use limited memory + LRU by default

## Basic Configuration

Default configuration:
- Port: 11211
- Memory: 64MB
- Max connections: 1024

Configuration file location: `/etc/memcached.conf`

example (memory=4GiB, client=10000, )
```
-m 4096
-c 10000
-r
```

## Memcached Commands

```
set
add
replace
append
prepend
cas
get
gets
delete
flush_all
incr
decr
touch
stats
stats items
stats slabs
stats sizes
version
quit
```

add vs set: add fail if a key already exists, set will overwrite
get vs gets: gets is for cas command, it will return a CAS identifier. The identifier will be used in next cas command.
cas: check and set (similar to upsert)

## Interesting use cases
1. store set or list 
    ```
    a list can be stored by 
    - a serialized json (overwrite it all)
    - append/prepend (only add, can't delete)
    
    alternative to store as a list
    - (set) store kv objects, use multi-get
    - (list) store a small id-list, then use mult-get to get values by id

    avoid big key
    - put items in chunks. eg, store list 0_500 in a key, 501_1000 in another key
    ```

2. reduce key size 
    ```
    base64/gzip encoding big numbers
    ```

3. scaling expiration
    ```
    set small soft expiration in value (1hr)
    set big hard expiration in memcached (1hr30m)

    this can help with cache stampede problem
    when curr_time > cache soft expire time, choose some of the clients to update the cache.

    the ratio of clients (probability) can be one of following options:
    - linear ramp 
        probability = (now - soft_timeout) / (hard_timeout - soft_timeout)
    - exponential ramp
        ratio = (now - soft_timeout) / (hard_timeout - soft_timeout)
        probability = ratio ** 2  # or ratio ** 3
    ```
4. hot key problem
    ```
    since memcached server does not talk to each other, and client uses consistent hashing and one key only is stored in one server, it will suffer from hot key problems

    work around:
    1. (temporary) use virtual node
    2. manual partition of hot keys, split it into buckets
    3. (best) use a load aware proxy (twemproxy / mcrouter)
    4. switch to redis cluster 
    ```
