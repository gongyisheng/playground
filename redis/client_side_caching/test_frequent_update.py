import asyncio
import redis.asyncio as aioredis

pool = aioredis.BlockingConnectionPool(host='localhost', port=6379, db=0, max_connections=1)
node = aioredis.Redis(connection_pool=pool)

async def freq_update():
    for i in range(100000000):
        await node.set("my_key", "my_value")
    
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(freq_update())