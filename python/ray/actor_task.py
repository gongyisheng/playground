import ray

REMOTE_RAY_ADDRESS = "ray://10.0.0.195:10001" 
print(f"Connecting to Ray cluster at {REMOTE_RAY_ADDRESS}...")

try:
    ray.init(address=REMOTE_RAY_ADDRESS, log_to_driver=True)
    print("Successfully connected to the remote Ray cluster.")
except Exception as e:
    print(f"Error connecting to Ray: {e}")
    exit()

@ray.remote
class Counter:
    """
    An Actor that maintains state (self.count).
    """
    def __init__(self):
        # Initial state
        self.count = 0

    def increment(self, value):
        self.count += value
        return self.count

    def get_count(self):
        return self.count

if __name__ == "__main__":
    print("--- Creating Remote Actor ---")
    
    counter_actor = Counter.remote()
    print("Actor instantiated. State is initialized.")

    ref1 = counter_actor.increment.remote(5)
    ref2 = counter_actor.increment.remote(10)
    ref3 = counter_actor.get_count.remote()
    
    result_ref1 = ray.get(ref1)
    result_ref2 = ray.get(ref2)
    final_count = ray.get(ref3)
    
    print("\n--- Actor State Results ---")
    print(f"Count after first increment: {result_ref1}")
    print(f"Count after second increment: {result_ref2}")
    print(f"Final count retrieved: {final_count}")