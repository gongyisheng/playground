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

# task can be a class (actor)
@ray.remote
class Counter:
    """
    An Actor that maintains state (self.count).
    """
    def __init__(self):
        self.count = 0

    def increment(self, value):
        self.count += value
        return self.count

    def get_count(self):
        return self.count

def test_counter():
    ray.init()
    counter_actor = Counter.remote()

    ref1 = counter_actor.increment.remote(5)
    ref2 = counter_actor.increment.remote(10)
    ref3 = counter_actor.get_count.remote()
    
    result_ref1 = ray.get(ref1)
    result_ref2 = ray.get(ref2)
    final_count = ray.get(ref3)
    
    print(f"Count after first increment: {result_ref1}")
    print(f"Count after second increment: {result_ref2}")
    print(f"Final count retrieved: {final_count}")

    ray.shutdown()


if __name__ == "__main__":
    test_calculate_squre()
    test_counter()