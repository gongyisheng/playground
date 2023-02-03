import asyncio

class Foo(object):
    @property
    def bar(self):
        return self._bar
    
    @bar.setter
    def bar(self, value):
        self._bar = value
    
    @bar.getter
    async def bar(self):
        await asyncio.sleep(1)
        if isinstance(self._bar, str):
            return self._bar.upper()
        else:
            return self._bar

async def main():
    foo = Foo()
    foo.bar = 'foo'
    print((await foo.bar)[1])

if __name__ == '__main__':
    asyncio.run(main())