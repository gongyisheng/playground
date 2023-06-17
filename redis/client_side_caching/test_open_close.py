import asyncio
import redis.asyncio as aioredis

pool = aioredis.BlockingConnectionPool(host='localhost', port=6379, db=0, max_connections=1)
node = aioredis.Redis(connection_pool=pool)
prefix = ['abc', 'zoo']

async def open_close():
    connection = await node.connection_pool.get_connection('_')
    await connection.send_command("CLIENT ID")
    connection_id = await connection.read_response()
    print(connection_id)

    prefix_command = " ".join(f"PREFIX {p}" for p in prefix)
    await connection.send_command(f"CLIENT TRACKING on REDIRECT {connection_id} BCAST {prefix_command}")
    resp = await connection.read_response()
    print(resp)

    await connection.disconnect()
    node.connection_pool.pool.put_nowait(None)

async def main(round):
    for i in range(round):
        await open_close()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(10))