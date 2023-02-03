import asyncio

async def generator():
    for i in range(20):
        await asyncio.sleep(0.1)
        yield i

async def main1():
    iter = generator()
    # async for i in iter:
    #     print(i)
    print([i async for i in iter])

class Foo(object):
    def __init__(self, _iter):
        self._iter = _iter
        self._bar = None

    async def _init(self):
        self._bar = [i async for i in self._iter]

async def main2():
    iter = generator()
    foo = Foo(iter)
    await foo._init()
    print(foo._bar)

if __name__ == '__main__':
    asyncio.run(main2())
