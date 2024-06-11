import asyncio


async def generator(limit=100):
    for i in range(limit):
        await asyncio.sleep(0.1)
        yield i


async def IO(id):
    await asyncio.sleep(1)
    print(f"id={id}")


async def main():
    iter = generator()
    async for i in iter:
        await IO(i)
        await IO(str(i) + "_")
    print("Done")


if __name__ == "__main__":
    asyncio.run(main())
