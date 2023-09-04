import asyncio
import logging
import sys

from lazy_init_base import Global
from lazy_init_pq import summary_stat

logging.basicConfig(
    stream=sys.stdout,
    # filename=logfile,
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s:%(message)s',
    datefmt='%H:%M:%S'
)

async def loop_once():
    tasks = []
    for _ in range(10_000):
        tasks.append(Global.attachment_client.add_message())
    await asyncio.gather(*tasks)

async def main():
    for i in range(100):
        await loop_once()
    summary_stat()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())