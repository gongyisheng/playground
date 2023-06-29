import asyncio

async def timeout_run_func(timeout_sec, func, *args, **kwargs):
    async def wrap_func():
        return await func(*args, **kwargs)

    res = await asyncio.wait_for(
        wrap_func(),
        timeout=timeout_sec,
    )
    return res

async def async_procedure(cmd):
    proc = await asyncio.create_subprocess_shell(cmd)
    stdout, stderr = await proc.communicate()
    print(f'[{cmd!r} exited with {proc.returncode}]')
    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')

async def main(cmd):
    try:
        await timeout_run_func(1, async_procedure, cmd)
    except asyncio.TimeoutError:
        print('timeout')

if __name__ == '__main__':
    asyncio.run(main('sleep 5'))
