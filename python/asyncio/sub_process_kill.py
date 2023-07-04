import asyncio
import functools
from multiprocessing import Manager
from multiprocessing import Event
from concurrent.futures import ProcessPoolExecutor, wait

def cpu_bond_func(event):
    a = False
    while True:
        a = not a
        if event.is_set():
            return

async def run_func_in_subprocess(func, *args, timeout=None, **kwargs):
    """
    Run a function in subprocess
    """

    with Manager() as manager:
        event = manager.Event()
        with ProcessPoolExecutor() as executor:
            try:
                _future = executor.submit(cpu_bond_func, event)
                await asyncio.sleep(timeout)
            except asyncio.TimeoutError as e:
                print(f'[func_subprocess] get timeout error')
                raise e
            except Exception as e:
                print(f'[func_subprocess] get uncaught error: {e}')
                raise e
            finally:
                print('Waiting...')
                event.set()
                print('All done.')
                print(f'[func_subprocess] process exit complete')

async def test():
    await run_func_in_subprocess(cpu_bond_func, timeout=5)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test())