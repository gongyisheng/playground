import random

def divide(x, y):
    sign = -1 if (x > 0) ^ (y > 0) else 1
    x, y = abs(x), abs(y)
    if x == y:
        return 1 if not sign else -1
    res = 0
    while x >= y:
        p = 0
        while x >= (y << p):
            p += 1
        p -= 1
        res += 1 << p
        x -= y << p
    return sign * res

def rand5():
    return random.randint(0, 4)

def rand5_to_rand7():
    while True:
        num = 5 * rand5() + rand5()
        if num < 21:
            return num % 7

if __name__ == "__main__":
    print(divide(10, -3))
    print(rand5_to_rand7())