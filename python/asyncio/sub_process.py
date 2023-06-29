import asyncio
import os

global_lock = asyncio.Lock()

async def get_running_process_number(pattern):
    proc = await asyncio.create_subprocess_shell(f'ps aux | grep {pattern} -c', stdout=asyncio.subprocess.PIPE)
    stdout, stderr = await proc.communicate()
    return int(stdout.decode().strip())

async def async_procedure(cmd):
    async with global_lock:
        start_proc_number = await get_running_process_number('python')
        print("start:",start_proc_number)
        proc = await asyncio.create_subprocess_shell(cmd)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), 5)
            print(f'[{cmd!r} exited with {proc.returncode}]')
            if stdout:
                print(f'[stdout]\n{stdout.decode()}')
            if stderr:
                print(f'[stderr]\n{stderr.decode()}')
        except Exception as e:
            raise e
        finally:
            os.system(f'kill -9 {proc.pid}')
            print("kill:",proc.pid)
            retry = 6
            for i in range(retry):
                end_proc_number = await get_running_process_number('python')
                print("end:",end_proc_number)
                if end_proc_number == start_proc_number:
                    break
                await asyncio.sleep(10)

async def main():
    tasks = [async_procedure('sleep 1') for i in range(10)]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())