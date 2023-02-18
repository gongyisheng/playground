Observation:
- Overall system performance goes down.
- Redis memory goes up for X10 times, finally OOM
- Client side error: Timeout reading from socket
- Network output limit at 300 Mbps

Analysis:
- redis-cli, MEMORY STATS, shows that `clients.normal` is too big (about 1G), `dataset.percentage` is only 14%, not caused by a burst of newly added data.
- redis-cli, CLIENT LIST, shows that there are too many clients with tot-mem=7M/5M, oll=1

Root Cause:
- Redis big key, a single key is about 7Mb, if there're 100 pods request the same key in a second, it requires 700 Mbps network brandwith.
- Network output limit at 300 Mbps, set by the hardware limit of AWS instance we use.
- The data to be sent out line up in the network output buffer, which can be read from `clients.normal` and takes more and more memory, finally OOM.

Solution:
- Avoid redis big key and frequent request for big key.
- Use client side caching if possible.