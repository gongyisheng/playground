import aiohttp
import asyncio
import tracemalloc

FETCH_CNT = 0

async def fetch(url):
    global FETCH_CNT
    FETCH_CNT += 1
    try:
        timeout = aiohttp.ClientTimeout(connect=5, sock_connect=10, sock_read=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await session.post(url, data="a"*100000)
            dict_result = await response.json()
            return dict_result
    except Exception as e:
        # print(f"Fail to get embedding. url=[{url}], error=[{e}]")
        return None
    
async def main(): 
    tracemalloc.start()
    url = "http://localhost:3000"
    try:
        while True:
            tasks = [asyncio.create_task(fetch(url)) for _ in range(10)]
            await asyncio.gather(*tasks)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass  # end and clean things up
    finally:
        memory_used = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('lineno')
        for stat in stats[:10]:
            print(stat)

if __name__ == "__main__":
    asyncio.run(main())
