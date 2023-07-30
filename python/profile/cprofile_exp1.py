import cProfile
import pstats
import asyncio

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)

def fib_seq(n):
    seq = []
    if n > 0:
        seq.extend(fib_seq(n - 1))
    seq.append(fib(n))
    return seq

async def combined(n):
    for i in range(n):
        fib_seq(10)
        await asyncio.sleep(1)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    import time
    p = cProfile.Profile(timer=time.process_time) # you can set up your custom timer
    # p.run('fib_seq(30)')
    p.run('loop.run_until_complete(combined(5))')
    p.dump_stats('output.prof')

    stream = open('output.txt', 'w')
    stats = pstats.Stats('output.prof', stream=stream)
    stats.sort_stats('cumtime')
    s = stats.print_stats()
    for k,v in s.stats.items():
        cc, nc, tt, ct, callers = v
        per_call_1 = 0 if nc==0 else float(tt) / nc
        per_call_2 = 0 if cc==0 else float(ct) / cc
        # the output will not be sorted because it's from dict
        print("[cProfile STATS] file: %s, line: %s, func: %s, direct_call: %s, total_call: %s, total_time: %s, tt_per_call: %s, cum_time: %s, ct_per_call: %s" % (k[0], k[1], k[2], str(cc), str(nc), str(tt), str(per_call_1), str(ct), str(per_call_2)))
    stream.close()
