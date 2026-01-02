import ray
import time

# task can be a function (basic)
@ray.remote
def calculate_square(number):
    """
    A simple remote function that runs on a Ray worker.
    """
    print(f"Worker is processing square for {number}...")
    time.sleep(1) # Simulate some work
    return number * number

def test_calculate_squre():
    ray.init()
    futures = [calculate_square.remote(i) for i in range(5)]

    results = ray.get(futures)

    print(f"Input list: [0, 1, 2, 3, 4]")
    print(f"Results: {results}")

    ray.shutdown()


if __name__ == "__main__":
    test_calculate_squre()
    