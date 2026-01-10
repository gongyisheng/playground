from functools import wraps


def with_defer(deferred_func):
    def decorator(fn):
        @wraps(fn) # reserves the original function's metadata, eg call fn.__name__, or fn.__doc__
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                deferred_func()

        return wrapper

    return decorator

@with_defer(lambda: print("defer"))
def test():
    print("test")

if __name__ == "__main__":
    test()