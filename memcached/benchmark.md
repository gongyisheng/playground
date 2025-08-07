## Benchmarking with memtier_benchmark

### Install memtier_benchmark
```bash
# Ubuntu/Debian
sudo apt-get install memtier-benchmark
```

### Benchmark Memcached
```bash
# Basic memcached benchmark
memtier_benchmark -s cluster-n0.local -p 11211 --protocol=memcache_text

# Detailed memcached benchmark
memtier_benchmark -s cluster-n0.local -p 11211 --protocol=memcache_text \
  -t 4 -c 50 -n 10000 --ratio=1:10 --data-size=1024 \
  --key-pattern=R:R --hide-histogram

# Write-heavy workload
memtier_benchmark -s cluster-n0.local -p 11211 --protocol=memcache_text \
  -t 8 -c 25 -n 5000 --ratio=10:1 --data-size=512

# Read-heavy workload  
memtier_benchmark -s cluster-n0.local -p 11211 --protocol=memcache_text \
  -t 8 -c 25 -n 5000 --ratio=1:50 --data-size=512
```

### Benchmark Redis
```bash
# Basic redis benchmark
memtier_benchmark -s cluster-n0.local -p 6379 --protocol=redis

# Detailed redis benchmark
memtier_benchmark -s cluster-n0.local -p 6379 --protocol=redis \
  -t 4 -c 50 -n 10000 --ratio=1:10 --data-size=1024 \
  --key-pattern=R:R --hide-histogram

# Write-heavy workload
memtier_benchmark -s cluster-n0.local -p 6379 --protocol=redis \
  -t 8 -c 25 -n 5000 --ratio=10:1 --data-size=512

# Read-heavy workload
memtier_benchmark -s cluster-n0.local -p 6379 --protocol=redis \
  -t 8 -c 25 -n 5000 --ratio=1:50 --data-size=512
```

### Parameter Explanations
```
-s, --server: Server hostname (default: cluster-n0.local)
-p, --port: Server port
-t, --threads: Number of threads (default: 4)
-c, --clients: Number of clients per thread (default: 50)
-n, --requests: Number of requests per client (default: 10000)
--ratio: Set:Get ratio (default: 1:10)
--data-size: Object data size in bytes (default: 32)
--key-pattern: Key pattern (R:R = random:random, S:S = sequential:sequential)
--protocol: Protocol to use (redis, memcache_text, memcache_binary)
--hide-histogram: Hide latency histogram in output
--run-count: Number of full test iterations to perform
--test-time: Test duration in seconds (instead of --requests)
```

### Result
env:
```
raspberrypi 5b
Linux cluster-n0 6.14.0-1005-raspi

memcached: 1.6.38
redis: 8.0
valkey: 8.1
```


standard
```
config
4         Threads
50        Connections per thread
10000     Requests per client

memcached
ALL STATS
============================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
----------------------------------------------------------------------------------------------------------------------------
Sets         8763.43          ---          ---         1.75666         1.47100         6.71900        12.99100       597.92 
Gets        87538.04         0.00     87538.04         1.75173         1.47100         6.71900        12.15900      2213.18 
Waits           0.00          ---          ---             ---             ---             ---             ---          --- 
Totals      96301.47         0.00     87538.04         1.75218         1.47100         6.71900        12.22300      2811.09

redis
ALL STATS
============================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
----------------------------------------------------------------------------------------------------------------------------
Sets         4462.70          ---          ---         4.07752         3.77500         7.29500        11.26300       343.71 
Gets        44577.98         0.00     44577.98         4.07273         3.77500         7.23100        11.26300      1736.50 
Waits           0.00          ---          ---             ---             ---             ---             ---          --- 
Totals      49040.69         0.00     44577.98         4.07316         3.77500         7.23100        11.26300      2080.21 

valkey
ALL STATS
============================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
----------------------------------------------------------------------------------------------------------------------------
Sets         7541.32          ---          ---         2.59159         2.23900         5.72700        26.11100       580.81 
Gets        75330.33         0.00     75330.33         2.61256         2.23900         5.75900        66.04700      2934.44 
Waits           0.00          ---          ---             ---             ---             ---             ---          --- 
Totals      82871.65         0.00     75330.33         2.61065         2.23900         5.75900        64.51100      3515.26 
```

write heavy
```
config 
8         Threads
25        Connections per thread
5000      Requests per client

memcached
ALL STATS
============================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
----------------------------------------------------------------------------------------------------------------------------
Sets        89845.63          ---          ---         2.02790         1.80700         5.66300        16.63900     48334.86 
Gets         8972.70         0.00      8972.70         2.02614         1.80700         5.56700        17.02300       226.82 
Waits           0.00          ---          ---             ---             ---             ---             ---          --- 
Totals      98818.33         0.00      8972.70         2.02774         1.80700         5.66300        16.63900     48561.68 

redis
ALL STATS
============================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
----------------------------------------------------------------------------------------------------------------------------
Sets        41313.14          ---          ---         4.40080         3.98300         7.45500        11.45500     22588.61 
Gets         4125.86         0.00      4125.86         4.39837         3.98300         7.48700        11.58300       160.70 
Waits           0.00          ---          ---             ---             ---             ---             ---          --- 
Totals      45439.00         0.00      4125.86         4.40058         3.98300         7.45500        11.45500     22749.32 

valkey
ALL STATS
============================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
----------------------------------------------------------------------------------------------------------------------------
Sets        85775.40          ---          ---         2.45621         1.80700         7.58300        85.50300     46899.06 
Gets         8566.22         0.00      8566.22         2.44543         1.80700         8.09500        84.99100       333.66 
Waits           0.00          ---          ---             ---             ---             ---             ---          --- 
Totals      94341.62         0.00      8566.22         2.45523         1.80700         7.58300        85.50300     47232.72 
```

read heavy
```
config
8         Threads
25        Connections per thread
5000      Requests per client

memcached
ALL STATS
============================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
----------------------------------------------------------------------------------------------------------------------------
Sets         1884.55          ---          ---         2.09544         2.00700         5.75900        17.91900      1013.83 
Gets        93294.91     84823.94      8470.97         2.10553         2.00700         6.01500        19.45500     47329.25 
Waits           0.00          ---          ---             ---             ---             ---             ---          --- 
Totals      95179.46     84823.94      8470.97         2.10533         2.00700         6.01500        19.32700     48343.08 

redis
ALL STATS
============================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
----------------------------------------------------------------------------------------------------------------------------
Sets          898.40          ---          ---         4.37812         3.93500         7.51900        11.64700       491.21 
Gets        44475.31     40437.05      4038.26         4.40319         3.95100         7.51900        11.51900     22069.45 
Waits           0.00          ---          ---             ---             ---             ---             ---          --- 
Totals      45373.71     40437.05      4038.26         4.40269         3.95100         7.51900        11.51900     22560.66

valkey
ALL STATS
============================================================================================================================
Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
----------------------------------------------------------------------------------------------------------------------------
Sets         1493.57          ---          ---         2.98953         2.65500         6.49500        12.54300       816.62 
Gets        73939.09     67225.58      6713.51         2.97921         2.65500         6.36700        10.75100     36689.91 
Waits           0.00          ---          ---             ---             ---             ---             ---          --- 
Totals      75432.66     67225.58      6713.51         2.97941         2.65500         6.36700        10.81500     37506.53 
```