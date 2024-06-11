import random
import time


def miss(count):
    nums = []
    for i in range(count):
        nums.append(random.randint(0, count))

    start = time.time()
    total = 0
    for n in nums:
        if n >= count // 2:
            total = total + 1
    print(f"total: {total}")
    end = time.time()
    print(f"cost {end-start} s")


def hit(count):
    nums = []
    for i in range(count):
        nums.append(random.randint(0, count))
    nums.sort()

    start = time.time()
    total = 0
    for n in nums:
        if n >= count // 2:
            total = total + 1
    print(f"total: {total}")
    end = time.time()
    print(f"cost {end-start} s")


if __name__ == "__main__":
    miss(30000000)
    hit(30000000)
