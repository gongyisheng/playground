import ray

@ray.remote(num_gpus=1)
class GPUActor:
    def __init__(self):
        pass

    def work(self):
        ip_addr = ray.util.get_node_ip_address()
        print(f"doing GPU work, ip_addr[{ip_addr}]")

def test_gpu_actor():
    ray.init(num_gpus=4)
    # actor num must be <= 4
    # actor still holds the gpu resource when task finishes 
    # resource will be held until class is recycled by GC
    actors = [GPUActor.remote() for _ in range(4)] 
    ray.get([actor.work.remote() for actor in actors])

if __name__ == "__main__":
    test_gpu_actor()