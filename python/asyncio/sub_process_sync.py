import asyncio
import time
import threading
from threading import Timer
import subprocess, shlex
import random

global_lock = threading.Lock()


def kill_func(proc, cmd, timeout_sec):
    proc.kill()
    print("cmd[%s] timeout[%s]" % (cmd, timeout_sec))


def timeout_run(cmd, timeout_sec):
    proc = subprocess.Popen(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    timer = Timer(timeout_sec, kill_func, [proc, cmd, timeout_sec])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
    finally:
        timer.cancel()


def run_cmd(cmd, timeout=None, task_id=None):
    global_lock.acquire()
    print(f"{task_id} subprocess start")
    start = time.time()
    try:
        timeout_run(cmd, timeout)
    except Exception as e:
        print(f"[{cmd!r} get exception {e}]")
    finally:
        end = time.time()
        print(f"{task_id} subprocess cost {end-start}")
        print(f"{task_id} subprocess end")
        global_lock.release()


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
    run_cmd(cmd, timeout=timeout, task_id=task_id)
    await io(task_id)
    cpu(task_id)
    await io(task_id)


async def main(timeout=None):
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
