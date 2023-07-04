import asyncio
import os

global_lock = asyncio.Semaphore(2)

async def get_running_process_number(pattern):
    proc = await asyncio.create_subprocess_shell(f'ps aux | grep {pattern} -c', stdout=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    return int(stdout.decode().strip())

async def async_procedure(cmd, timeout=None):
    async with global_lock:
        start_proc_number = await get_running_process_number('python')
        print("start:",start_proc_number)
        proc = await asyncio.create_subprocess_shell("exec "+cmd)
        try:
            if timeout:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            else:
                stdout, stderr = await proc.communicate()
            print(f'[{cmd!r} exited with {proc.returncode}]')
            if stdout:
                print(f'[stdout]\n{stdout.decode()}')
            if stderr:
                print(f'[stderr]\n{stderr.decode()}')
        except Exception as e:
            pass
        finally:
            if proc.returncode is None:
                print("kill")
                proc.kill()
            end_proc_number = await get_running_process_number('python')
            print("end:",end_proc_number)

async def main(cmd, timeout=None):
    tasks = [async_procedure(cmd, timeout=timeout) for i in range(10)]
    await asyncio.gather(*tasks)
    await asyncio.sleep(100)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(main('sleep 3', 15))
    loop.run_until_complete(main('python deadlock.py', 15))