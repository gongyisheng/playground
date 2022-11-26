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
    p.run('fib_seq(30)')
    p.dump_stats('output.prof')

    stream = open('output.txt', 'w')
    stats = pstats.Stats('output.prof', stream=stream)
    stats.sort_stats('cumtime')
    s = stats.print_stats()
    for k,v in s.stats.items():
        print(f"file: {k[0]}, line: {k[1]}, func: {k[2]}, ncalls: {v[0]}, tottime: {v[1]}, cumtime: {v[2]}")
