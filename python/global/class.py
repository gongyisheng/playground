import time
from lru import LRU
import re
cache = LRU(10)
HIT_CACHE = 0

class Test():
    def __init__(self) -> None:
        pass
    def run(self):
        cache[time.time()] = re.compile(str(time.time()))

def proc():
    a = Test()
    a.run()
    print(len(cache))

def main():
    proc()
    proc()
    proc()
    print(cache.items())
    
main()