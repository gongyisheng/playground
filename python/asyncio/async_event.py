import asyncio

class Test:

    KEY = 'test'
    PLACEHOLDER = object()

    def __init__(self):
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        self._cache = {}

    async def wait(self):
        print('waiting for it...')
        await self.event.wait()
        print('...got it!')

    def set(self):
        print('setting it')
        self.event.set()
    
    def clear(self):
        print('clearing it')
        self.event.clear()
    
    async def main(self):
        if self.KEY in self._cache:
            if self._cache[self.KEY] == self.PLACEHOLDER:
                await self.wait()
            print(f"get: {self._cache[self.KEY]}")
        else:
            await self.lock.acquire()
            self._cache[self.KEY] = self.PLACEHOLDER
            await asyncio.sleep(2)
            self._cache[self.KEY] = 'value'
            self.lock.release()
            self.set()
            self.clear()


async def test():
    t = Test()
    tasks = [t.main() for _ in range(10)]
    await asyncio.gather(*tasks)

    del t._cache[t.KEY]
    tasks = [t.main() for _ in range(10)]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(test())