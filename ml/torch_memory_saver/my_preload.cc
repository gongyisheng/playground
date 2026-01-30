import ctypes
import time
from typing import Callable

import torch

# print('change_current_allocator')
# new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
#     str(Path(__file__).parent / 'my_alloc.so'), 'my_malloc', 'my_free')
# torch.cuda.memory.change_current_allocator(new_alloc)


# TODO ok?
cdll_my_preload_so = ctypes.CDLL('./my_preload.so')
print(f'{cdll_my_preload_so=}')


def _ptr(x):
    assert isinstance(x, torch.Tensor)
    return hex(x.data_ptr())


class KVCache:
    def __init__(self):
        self.create_buffers(1)

    def create_buffers(self, value):
        # or model weights, etc
        self.kv_buffer = torch.tensor([value, value, value], dtype=torch.int32, device='cuda')
        print(f'create_buffers {_ptr(self.kv_buffer)=}')

    def clear_buffers(self):
        del self.kv_buffer

    def execute(self, arg: torch.Tensor) -> torch.Tensor:
        # print(f'KVCache.execute {arg=} {self.kv_buffer=}')
        return arg + self.kv_buffer


# https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
def create_cuda_graph(fn: Callable):
    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        print('with torch.cuda.stream(s) execute fn')
        fn()
    torch.cuda.current_stream().wait_stream(s)

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        print('with torch.cuda.graph(g) execute fn')
        fn()

    return g


def run():
    cache = KVCache()
    static_input = torch.zeros((3,), dtype=torch.int32, device='cuda')
    static_output = torch.zeros((3,), dtype=torch.int32, device='cuda')
    print(f'{_ptr(static_input)=} {_ptr(static_output)=}')

    def fn():
        nonlocal static_output
        static_output = cache.execute(static_input)

    g = create_cuda_graph(fn)

    print('replay #1')
    static_input[...] = 100
    g.replay()
    print(f'{static_output=}')
    assert torch.all(
        static_output == torch.tensor([101, 101, 101], dtype=torch.int32, device='cuda')), f'{static_output=}'

    # cache.clear_buffers()

    big_tensor = torch.zeros((2_000_000_000,), dtype=torch.uint8, device='cuda')
    print(f'{big_tensor=}')

    print('torch.cuda.empty_cache()')
    torch.cuda.empty_cache()

    print('sleep...')
    time.sleep(3)

    print('call hack_release_occupation')
    cdll_my_preload_so.hack_release_occupation()

    print('sleep...')
    time.sleep(3)

    # this should fail
    # print(f'{cache.kv_buffer=}')

    print('call hack_resume_occupation')
    cdll_my_preload_so.hack_resume_occupation()

    dummy = torch.zeros((3,), device='cuda')
    print(f'{_ptr(dummy)=}')

    # cache.create_buffers(2)

    cache.kv_buffer[...] = 2

    print('replay #2')
    static_input[...] = 200
    g.replay()
    print(f'{static_output=}')
    assert torch.all(
        static_output == torch.tensor([202, 202, 202], dtype=torch.int32, device='cuda')), f'{static_output=}'

    print('sleep...')
    time.sleep(3)

    print(f'{big_tensor=}')
    print(f'{dummy=}')


if __name__ == '__main__':
    run()