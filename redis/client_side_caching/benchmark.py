import argparse
import json
import redis
from collections import defaultdict
from time import perf_counter
from manager import Manager

# A global for holding time measurements
timings = defaultdict(list)

def timeit(fn):
    ' Wraps a function call and times it '
    def doit(*args):
        global timings
        start = perf_counter()
        res = fn(*args)
        end = perf_counter()
        ms = (end - start) * 1000
        name = f'{fn.__name__} {args[0].__module__} {args[1]}'
        timings[name].append(ms)
        return res
    return doit

def populate_db(conn, number):
    ' Populates the database'
    v = 'x' * 42
    p = conn.pipeline(transaction=False)
    for i in range(number):
        p.set(f'benchmark:{i}', v)
        if i % 100 == 0:
            p.execute()
    p.execute()

@timeit
def single_read(conn, number):
    for i in range(number):
        conn.get(f'benchmark:{i}')

@timeit
def eleven_reads(conn, number):
    for i in range(number):
        conn.get(f'benchmark:{i}')
        for j in range(10):
            conn.get(f'benchmark:{i}')

@timeit
def write_and_reads(conn, number, conn2):
    for i in range(number):
        conn.get(f'benchmark:{i}')
        conn2.set(f'benchmark:{i % 10}', 'x' * 42)
        for j in range(10):
            conn.get(f'benchmark:{i}')

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--capacity', help='Cache capacity', type=int, default=100)
    parser.add_argument('-n', '--number', help='Number of keys', type=int, default=1000)
    parser.add_argument('-r', '--repeat', help='Number of repeats', type=int, default=5)
    parser.add_argument('-u', '--url', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    args = parser.parse_args()

    # Set up Redis connections
    pool = redis.ConnectionPool.from_url(args.url)
    manager = Manager(pool, capacity=args.capacity)
    writer = redis.Redis(connection_pool=pool)
    normal = redis.Redis(connection_pool=pool)
    cached = manager.get_connection()
    conns = [normal, cached]

    # Populate the database
    populate_db(writer, args.number)

    # Reading a single key, all keys
    for r in range(args.repeat):
        for c in conns:
            single_read(c, args.number)

    # Reading a single key, but only up to cache's capacity
    for r in range(args.repeat):
        for c in conns:
            single_read(c, args.capacity)

    # Reading a single key and another ten keys that are always the same
    for r in range(args.repeat):
        for c in conns:
            eleven_reads(c, args.number)

    # Reading a single key, changing one of and reading ten keys that are always the same
    for r in range(args.repeat):
        for c in conns:
            write_and_reads(c, args.number, writer)

    manager.stop()
    print(json.dumps(timings, indent=2))