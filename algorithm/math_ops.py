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

if __name__ == "__main__":
    print(divide(10, -3))