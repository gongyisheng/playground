def inner():
    yield 1
    yield 2
    return "done"

def outer():
    result = yield from inner()  # yields 1, then 2
    print(result)                # prints "done"


if __name__ == "__main__":
    for value in outer():
        print(value)