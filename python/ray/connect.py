import ray
import socket

REMOTE_RAY_ADDRESS = "ray://10.0.0.195:10001" 
print(f"Connecting to Ray cluster at {REMOTE_RAY_ADDRESS}...")

try:
    ray.init(address=REMOTE_RAY_ADDRESS, log_to_driver=True)
    print("Successfully connected to the remote Ray cluster.")
except Exception as e:
    print(f"Error connecting to Ray: {e}")
    exit()

# Define a remote task
@ray.remote
def get_node_host():
    # This task will run on a worker node in the remote cluster (not your client PC)
    return socket.gethostname()

if __name__ == "__main__":

    future = get_node_host.remote()
    remote_host = ray.get(future)
    
    print("\n--- Execution Result ---")
    print(f"The task was executed on a worker: {remote_host}")
    print(f"Your client machine's hostname: {socket.gethostname()}")
    
    ray.shutdown()