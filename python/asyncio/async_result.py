import asyncio

async def run_one_task(i):
    await asyncio.sleep(0.1)
    return i

async def run_all():
    tasks = [asyncio.create_task(run_one_task(i)) for i in range(10)]
    results = await asyncio.gather(*tasks)
    print(results)

if __name__ == "__main__":
    asyncio.run(run_all())