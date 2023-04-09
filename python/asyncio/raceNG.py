# Use race condition to get a random value from a list
# Doesn't work as expected 
import asyncio

class raceNG(object):
    def __init__(self, available_values=[]):
        self.available_values = available_values
        self.value = None
        self.set_value_event = asyncio.Event()
    
    async def get(self):
        self.set_value_event.set()
        while self.value is None:
            await asyncio.sleep(0.001)
        self.set_value_event.clear()
        return_value = self.value
        self.value = None
        return return_value

    async def task(self, id):
        while True:
            await self.set_value_event.wait()
            await asyncio.sleep(0.001)
            self.value = self.available_values[id]
    
    async def run(self):
        await asyncio.gather(*[asyncio.create_task(self.task(i)) for i in range(len(self.available_values))])
    
async def test():
    rand = raceNG([1,2,3,4,5,6,7,8,9,10])
    asyncio.create_task(rand.run())
    for i in range(10):
        print(await rand.get())

if __name__ == '__main__':
    asyncio.run(test())
