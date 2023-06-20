import asyncio
from lru import LRU
import logging

async def main():
    lru = LRU(1)
    lru['a'] = 1
    get_pool = 10

    async def _get():
        for i in range(2000):
            logging.info(f"get done. a:{lru.get('a')}, items:{lru.items()}")
            await asyncio.sleep(0)
    
    async def _clear():
        await asyncio.sleep(0.1)
        lru.clear()
        logging.info(f"clear done. items:{lru.items()}")
    
    get_task = [asyncio.create_task(_get()) for _ in range(get_pool)]
    clear_task = asyncio.create_task(_clear())
    await asyncio.gather(*get_task, clear_task)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="lru_debug.log", filemode="w", format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(main())