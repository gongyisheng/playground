import time

usage_data = {}

def track_count(metric):
    def decorator(func):
        def wrapper(*args, **kwargs):
            _metric = metric + "_count"
            usage_data[_metric] = usage_data.get(_metric, 0) + 1
            print(f"record count: _metric={_metric}, data={usage_data[_metric]}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def track_time(metric):
    def decorator(func):
        def wrapper(*args, **kwargs):
            _metric = metric + "_time"
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            usage_data[_metric] = usage_data.get(_metric, 0) + (end - start)*1000
            print(f"record time: _metric={_metric}, data={usage_data[_metric]}")
        return wrapper
    return decorator

@track_count("IO")
@track_time("IO")
def IO(wait=1):
    time.sleep(wait)
    pass

def display_usage():
    print(usage_data)

def main():
    for i in range(5):
        IO(i)

if __name__ == '__main__':
    main()
    display_usage()