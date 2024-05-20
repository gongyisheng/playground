from redis import asyncio as aioredis
import asyncio

# set: redis server
# CONFIG SET timeout 10
config = {
    "host": "localhost",
    "port": 6379,
    "max_connections": 20,
    "timeout": None,
    "socket_timeout": 100,
    "socket_connect_timeout": 5,
}


async def timeout_proc(r):
    await r.lpush("list", "hello")
    await asyncio.sleep(15)
    print(await r.brpop("list"))


async def main():
    pool = aioredis.BlockingConnectionPool(**config)

    r = aioredis.Redis(connection_pool=pool)
    tasks = [asyncio.create_task(timeout_proc(r))]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
