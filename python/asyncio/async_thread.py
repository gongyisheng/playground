import asyncio 
import threading

# run async code in sync mode
class AsyncLoopThread:                                                        
      def __init__(self):                                                       
          self.loop = asyncio.new_event_loop()                                  
          self._thread = threading.Thread(target=self._run, daemon=True)        
          self._thread.start()                                                  
                                                                                
      def _run(self):                                                           
          asyncio.set_event_loop(self.loop)                                     
          self.loop.run_forever()                                               
                                                                                
      def run(self, coro):                                                      
          return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

def test1():
    async def fetch_data(i):
        await asyncio.sleep(0.1)
        print(f"fetch data {i}")
    async_runner = AsyncLoopThread()                                              
    result = async_runner.run(fetch_data(1))
    result = async_runner.run(fetch_data(2))


from concurrent.futures import ThreadPoolExecutor
import time

# run sync code in async mode
executor = ThreadPoolExecutor(max_workers=4)

async def test2():
    def block_func(i):
        time.sleep(0.1)
        print(f"block func {i}")
                                                         
    loop = asyncio.get_event_loop()                                           
    # Run blocking function without blocking the event loop                   
    result = await loop.run_in_executor(executor, block_func, 1)
    result = await loop.run_in_executor(executor, block_func, 2)                                                                  

if __name__ == "__main__":
    test1()
    asyncio.run(test2())