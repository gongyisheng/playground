from lru import LRU

def test():
    cache = LRU(10)
    for i in range(100):
        cache[i] = object()
        if i % 10 == 0 and i > 0:
            for j in range(5):
                cache[i-j]

if __name__ == "__main__":
    test()