import ray
import time

REMOTE_RAY_ADDRESS = "ray://10.0.0.195:10001" 
print(f"Connecting to Ray cluster at {REMOTE_RAY_ADDRESS}...")

try:
    ray.init(address=REMOTE_RAY_ADDRESS, log_to_driver=True)
    print("Successfully connected to the remote Ray cluster.")
except Exception as e:
    print(f"Error connecting to Ray: {e}")
    exit()

@ray.remote
def calculate_square(number):
    """
    A simple remote function that runs on a Ray worker.
    """
    print(f"Worker is processing square for {number}...")
    time.sleep(1) # Simulate some work
    return number * number

if __name__ == "__main__":
    print("--- Starting Remote Tasks ---")
    
    futures = [calculate_square.remote(i) for i in range(5)]
    print("Tasks submitted. Results are pending (ObjectRefs created).")

    results = ray.get(futures)
    
    print("\n--- Results Retrieved ---")
    print(f"Input list: [0, 1, 2, 3, 4]")
    print(f"Results: {results}")

    ray.shutdown()