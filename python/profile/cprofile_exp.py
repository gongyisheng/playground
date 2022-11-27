import cProfile
import profile
import pstats

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


if __name__ == "__main__":
    p = cProfile.Profile()
    p.run('fib_seq(30),fib_seq(35)')
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
        print("[cProfile STATS] file: %s, line: %s, func: %s, direct_call: %s, total_call: %s, total_time: %s, tt_per_call: %s, cum_time: %s, ct_per_call, %s" % (k[0], k[1], k[2], str(cc), str(nc), str(tt), str(per_call_1), str(ct), str(per_call_2)))
    stream.close()
