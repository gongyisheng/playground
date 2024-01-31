import asyncio
from redis.asyncio import Redis, BlockingConnectionPool

pool = BlockingConnectionPool(max_connections=10, timeout=20, port=6379, db=0)
r = Redis(connection_pool=pool)

async def main():
    await r.flushdb()
    await r.hset('test', 'f1', 'v1')
    await r.hset('test', 'f2', 'v2')
    await r.hset('test', 'f3', 'v3')
    print(await r.hkeys('test'))
    print(await r.hgetall('test'))

if __name__ == '__main__':
    asyncio.run(main())
