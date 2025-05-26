import asyncio
import logging
import time
import random
import aiohttp

logging.basicConfig(level=logging.DEBUG)

async def do_stuff(sess):
    cpu_hog = False
    while True:
        resp = await sess.get('http://127.0.0.1:12345/')
        # print('status:', resp.status)
        got = 0
        async for chunk in resp.content.iter_chunked(8192):
            got += len(chunk)
            lunch = random.random() * 0.001
            if cpu_hog:
                time.sleep(lunch)
            else:
                await asyncio.sleep(lunch)

        print('<<<', got, 'bytes')
        lunch = random.random() * 0.1
        if cpu_hog:
            time.sleep(lunch)
        else:
            await asyncio.sleep(lunch)

async def main():
    c = aiohttp.TCPConnector(
        limit=3,
        keepalive_timeout=1,
        # timeout_ceil_threshold=0.5,
    )
    async with aiohttp.ClientSession(connector=c) as sess:
        tasks = [asyncio.create_task(do_stuff(sess)) for _ in range(20)]
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())