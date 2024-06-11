import asyncio
import time
import random

global_lock = asyncio.Semaphore(1)


async def run_cmd(cmd, timeout=None, task_id=None):
    async with global_lock:
        print(f"{task_id} subprocess start")
        start = time.time()
        proc = await asyncio.create_subprocess_shell("exec " + cmd)
        try:
            if timeout:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            else:
                await proc.wait()
            print(f"[{cmd!r} exited with {proc.returncode}]")
        except asyncio.TimeoutError as e:
            print(f"[{cmd!r} get timeout error]")
        except Exception as e:
            print(f"[{cmd!r} get uncaught exception {e}]")
        finally:
            if proc.returncode is None:
                print(f"{task_id} kill")
                proc.kill()
            end = time.time()
            print(f"{task_id} subprocess cost {end-start}")
        print(f"{task_id} subprocess end")


def cpu(task_id):
    print(f"{task_id} cpu start")
    for i in range(10000000):
        i = i + 1
    print(f"{task_id} cpu end")


async def io(task_id):
    print(f"{task_id} io start")
    await asyncio.sleep(0.05)
    print(f"{task_id} io end")


async def single_round(cmd, timeout=None, task_id=None):
    await run_cmd(cmd, timeout=timeout, task_id=task_id)
    await io(task_id)
    cpu(task_id)
    await io(task_id)


async def main(timeout):
    tasks = [
        single_round(
            f"sleep {random.random()*timeout}",
            timeout=timeout,
            task_id=random.randint(0, 1000),
        )
        for i in range(5)
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(8))
