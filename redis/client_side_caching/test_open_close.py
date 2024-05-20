import asyncio
import redis.asyncio as aioredis

pool = aioredis.BlockingConnectionPool(
    host="localhost", port=6379, db=0, max_connections=20
)
node = aioredis.Redis(connection_pool=pool)
prefix = ["abc", "zoo"]
LISTEN_INVALIDATE_CHANNEL = b"__redis__:invalidate"


async def open_close():
    _pubsub = node.pubsub()
    await _pubsub.execute_command("CLIENT ID")
    _pubsub_client_id = await _pubsub.parse_response(block=False, timeout=1)
    print(_pubsub_client_id)

    await _pubsub.subscribe(LISTEN_INVALIDATE_CHANNEL)
    resp = await _pubsub.get_message(timeout=1)
    print(resp)

    resp = await node.client_tracking_on(
        clientid=_pubsub_client_id, bcast=True, prefix=prefix
    )
    print(resp)

    resp = await node.client_tracking_off(clientid=_pubsub_client_id, bcast=True)
    print(resp)

    await _pubsub.unsubscribe(LISTEN_INVALIDATE_CHANNEL)
    resp = await _pubsub.get_message(timeout=1)
    print(resp)

    await _pubsub.reset()


async def main(round):
    for i in range(round):
        await open_close()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(10))
