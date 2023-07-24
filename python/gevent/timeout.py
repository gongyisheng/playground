
import gevent
from gevent.monkey import saved
import subprocess, shlex
from threading import Timer

def kill_func(proc, cmd, timeout_sec):
    proc.kill()
    print("cmd[%s] timeout[%s]" % (cmd, timeout_sec))

def timeout_run(cmd, timeout_sec):
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = Timer(timeout_sec, kill_func, [proc, cmd, timeout_sec])
    try:
        timer.start()
        stdout,stderr = proc.communicate()
    finally:
        timer.cancel()

def timeout_run_func(timeout_sec, func, *args, **kwargs):
    if 'sys' not in saved:
        return func(*args, **kwargs)
    p = gevent.spawn(func, *args, **kwargs)
    p.join(timeout_sec)
    p.kill()
    return p.value

def inf_loop():
    a = True
    while True:
        a = not a

def main_func():
    timeout_run_func(2, inf_loop)

def main_cmd():
    timeout_run("taskset -c bin/inf_loop", 2)

if __name__ == "__main__":
    #main_func()
    main_cmd()