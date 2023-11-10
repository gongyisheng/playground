import asyncio
from redis import asyncio as aioredis
import time

pool = aioredis.BlockingConnectionPool(host='127.0.0.1', port=6379, db=0, max_connections=5)
node = aioredis.Redis(connection_pool=pool)

async def freq_update():
    async with node.monitor() as m:
        async for command in m.listen():
            print(command)

async def get_key():
    while True:
        await node.get("my_key")
        await asyncio.sleep(1)
    
async def main():
    tasks = [
        asyncio.create_task(freq_update()),
        asyncio.create_task(get_key())
    ]
    await asyncio.gather(*tasks)
    
if __name__ == '__main__':
    asyncio.run(main())