#include <cstdio>

// dim3: A built-in CUDA type representing 3D dimensions (x, y, z).

// nvcc -o build/print_id print_id.cu && ./build/print_id

__global__ void print_ids() {
    int row = blockIdx.x;
    int col = blockIdx.y;
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("row=%d, col=%d, tid=%d, global_id=%d\n", row, col, tid, global_id);
}

int main() {
    dim3 grid(2, 2);   // 2x2 blocks
    dim3 block(4);     // 4 threads per block

    print_ids<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}
