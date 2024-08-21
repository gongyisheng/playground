from lru import LRU

def test():
    cache = LRU(10)
    for i in range(100):
        cache[i] = i

if __name__ == "__main__":
    test()