import ray

@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]

@ray.remote(num_gpus=1)
class GPUActor:
    def __init__(self):
        pass

    def work(self):
        print("doing GPU work")

def test_gpu_actor():
    ray.init(num_gpus=4)
    actors = [GPUActor.remote() for _ in range(5)]
    ray.get([actor.work.remote() for actor in actors])

if __name__ == "__main__":
    test_gpu_actor()