from redis import asyncio as aioredis
import asyncio
import logging
import socket
import time
import logging
import sys

MAX_BACKOFF_TIME = 300000  # 300s
SOCKET_TIMEOUT = 30
GET_CONNECTION_TIMEOUT = 60

logging.basicConfig(
    stream=sys.stdout,
    # filename=logfile,
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    datefmt='%H:%M:%S'
)

class PageQueue(object):

    def __init__(self, host="127.0.0.1", redis_max_connections=5):

        self._host = host
        self._host_ips = [self._host]
        self._last_host_update_time = 0
        self._redis_max_connections = redis_max_connections
        self._redis_clients = {self._host: self._new_redis_client(self._host)}
        self._out_queues = {}  # dict[queue_name, asyncio.Queue]
        self._background_brpop_tasks = {}  # dict[queue_name, Task]
        self._background_get_tasks = {}

        self._refresh_host_ip_task = asyncio.create_task(self._backgroud_refresh_host_ips())

    async def ping_redis(self, rds, retry):
        for i in range(retry):
            try:
                await rds.ping()
                return
            except Exception as e:
                logging.error('ping fail, err=%s' % e)
                await asyncio.sleep(1)
        raise Exception('ping fail')

    async def _refresh_host_ips(self):
        try:
            result = await asyncio.get_running_loop().getaddrinfo(self._host, None,
                        family=socket.AF_INET, type=socket.SOCK_DGRAM, proto=socket.IPPROTO_IP, flags=socket.AI_CANONNAME)
        except Exception as e:
            logging.error('Socket error - error[%s] host[%s]', e, self._host)
            return

        host_ips = []
        ips = set([x[4][0] for x in result] if result else [])
        for ip in set(self._host_ips).union(ips):
            if not self._redis_clients.get(ip):
                self._redis_clients[ip] = self._new_redis_client(ip)
            redis_client = self._redis_clients[ip]

            try:
                await self.ping_redis(redis_client, 2)
                logging.info('ping success, ip=%s' % ip)
            except Exception as e:
                continue

            if ip != self._host:
                host_ips.append(ip)

        if host_ips:
            self._host_ips = host_ips
            self._last_host_update_time = int(time.time())

        if not self._host_ips:
            self._host_ips = [self._host]

    def _new_redis_client(self, host):
        cpool = aioredis.BlockingConnectionPool(
            host=host,
            max_connections=self._redis_max_connections,
            timeout=GET_CONNECTION_TIMEOUT,
            socket_timeout=SOCKET_TIMEOUT,
            retry_on_error=[ConnectionError],
        )
        redis_client = aioredis.Redis(connection_pool=cpool)
        return redis_client
    
    async def _backgroud_refresh_host_ips(self):
        while True:
            try:
                await self._refresh_host_ips()
            except Exception as e:
                logging.error('_refresh_host_ips err=%s' % e)
            await asyncio.sleep(20)
    
    async def get(self, key):
        try:
            val = await self._redis_clients[self._host].get(key)
            logging.info("get key=%s, val=%s" % (key, val))
        except Exception as e:
            logging.error('get key=%s err=%s' % (key, e))
            val = None
        return val


async def loop_once(pq):
    tasks = [asyncio.create_task(pq.get('key')) for _ in range(6)]
    await asyncio.gather(*tasks)
    await asyncio.sleep(30)

async def test():

    pq = PageQueue()
    for i in range(100):
        await loop_once(pq)
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test())
