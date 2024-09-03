import asyncio
from redis import asyncio as aioredis

async def main():
    while True:
        try:
            pool = aioredis.BlockingConnectionPool(
                host="localhost", 
                port=6379, 
                db=0,
                socket_timeout=300, 
                socket_connect_timeout=5,
                max_connections=1
            )
            r = aioredis.Redis(connection_pool=pool)
            pubsub = r.pubsub()
            await pubsub.execute_command("CLIENT ID")
            client_id = await pubsub.parse_response(block=False, timeout=1)
            print(f"Client ID: {client_id}")
            await pubsub.reset()
        except Exception as e:
            print(f"Exception: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    for i in range(10000):
        asyncio.run(main())