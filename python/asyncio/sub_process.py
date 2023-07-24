import asyncio
import time

global_lock = asyncio.Semaphore(1)

async def async_procedure(cmd, timeout=None):
    async with global_lock:
        start = time.time()
        proc = await asyncio.create_subprocess_shell("exec "+cmd)
        try:
            if timeout:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            else:
                await proc.wait()
            print(f'[{cmd!r} exited with {proc.returncode}]')
        except asyncio.TimeoutError as e:
            print(f'[{cmd!r} get timeout error]')
        except Exception as e:
            print(f'[{cmd!r} get uncaught exception {e}]')
        finally:
            if proc.returncode is None:
                print("kill")
                proc.kill()
            end = time.time()
            print(f"cost {end-start}")

async def main(cmd, timeout=None):
    tasks = [async_procedure(cmd, timeout=timeout) for i in range(10)]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(main('sleep 3', 15))
    # loop.run_until_complete(main('python deadlock.py', 15))
    # loop.run_until_complete(main('sleep 3', 3.050))
    # loop.run_until_complete(main('python3 -c \"from pdf import pdf2html\"', 3.050))
    # loop.run_until_complete(main('pdf2htmlEX', 0.5))
    loop.run_until_complete(main('sleep 1', 1.017))