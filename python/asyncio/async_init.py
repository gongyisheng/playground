import asyncio

# object with async __init__

# Most magic methods aren't designed to work with async def/await
# https://docs.python.org/3/reference/datamodel.html#special-method-names

# correct example 
# FooCorrect.__init__ -> FooCorrect.__await__ -> FooCorrect.init -> FooCorrect.init: Bar
class FooCorrect(object):
    def __init__(self, name="Bar"):
        print("FooCorrect.__init__")
        self.name = name
    
    def __await__(self):
        # return a Generator. 
        # ref: https://docs.python.org/3/reference/datamodel.html#awaitable-objects
        print("FooCorrect.__await__")
        return self.init().__await__()

    async def init(self):
        print("FooCorrect.init")
        await asyncio.sleep(1)
        print(f"FooCorrect.init: {self.name}")


# wrong example
# raise error: RuntimeWarning: coroutine 'FooWrong.__init__' was never awaited
class FooWrong(object):
    async def __init__(self, name="Bar"):
        self.name = name
        await self.init()
    
    async def init(self):
        await asyncio.sleep(1)
        print(f"FooWrong.init: {self.name}")

async def testcorrect():
    print("------ test foo = FooCorrect() -------")
    foo = FooCorrect()
    print("------ test foo = await FooCorrect() -------")
    foo = await FooCorrect()

async def testwrong():
    print("------ test foo = FooWrong() -------")
    try:
        foo = FooWrong()
    except Exception as e:
        print(f"error: {e}")
    
    print("------ test foo = await FooWrong() -------")
    try:
        foo = await FooWrong()
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    asyncio.run(testcorrect())
    asyncio.run(testwrong())