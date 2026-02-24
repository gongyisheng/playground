#include "utils.h"

/*
 * CUDA Kernel Launch - Execution Configuration
 *
 * Syntax: kernel<<<numBlocks, threadsPerBlock>>>(args)
 *
 * Thread hierarchy:
 *   Grid (all blocks) -> Block -> Thread
 *
 *   <<<2, 4>>> means:
 *   ┌─────────────┐ ┌─────────────┐
 *   │   Block 0   │ │   Block 1   │
 *   │[T0 T1 T2 T3]│ |[T0 T1 T2 T3]│
 *   └─────────────┘ └─────────────┘
 *   Total: 2 blocks × 4 threads = 8 threads
 *
 * Built-in variables inside kernel:
 *   blockIdx.x  - which block this thread is in (0 to numBlocks-1)
 *   threadIdx.x - which thread within the block (0 to threadsPerBlock-1)
 *   blockDim.x  - threads per block
 *   gridDim.x   - number of blocks
 *
 * Global thread index formula:
 *   int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *
 * Block:
 *   - Group of threads that run on same SM (Streaming Multiprocessor)
 *   - Threads in same block can share memory (__shared__) and sync (__syncthreads)
 *   - Max 1024 threads per block (hardware limit)
 *   - RTX 3060 has 28 SMs (multiProcessorCount), can run many blocks in parallel
 *   - Check SM count: cudaGetDeviceProperties(&prop, 0); prop.multiProcessorCount
 *
 * How to calculate numBlocks:
 *   int n = 1000000;                                          // Total elements
 *   int blockSize = 256;                                      // Threads per block (128/256/512)
 *   int numBlocks = (n + blockSize - 1) / blockSize;          // Ceiling division
 *   kernel<<<numBlocks, blockSize>>>(data, n);
 *
 *   Example: n=1000, blockSize=256
 *   numBlocks = (1000 + 255) / 256 = 4 blocks
 *   Total threads = 4 × 256 = 1024 (covers all 1000, some threads idle)
 *
 * Warp (32 threads):
 *   - GPU executes threads in groups of 32 called "warps"
 *   - warps per block = blockSize / 32
 *   - Always use blockSize as multiple of 32 to avoid wasted threads
 *     <<<N, 100>>>  // Bad: 4 warps but 28 threads idle
 *     <<<N, 128>>>  // Good: 4 warps, no waste
 *   - All 32 threads in warp execute same instruction (SIMT)
 *   - "Warp divergence" (if/else) causes serialization - avoid when possible
 *
 * Why blockSize=256?
 *   - 256 = 8 warps, good balance for most kernels
 *   - 128: use when kernel needs many registers/shared memory
 *   - 512/1024: use for simple kernels with few registers
 */

// nvcc -o build/kernel basic/kernel.cu && ./build/kernel

// Simple kernel - each thread prints its IDs
__global__ void print_ids() {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("  Block %d, Thread %d -> Global index: %d\n",
           blockIdx.x, threadIdx.x, globalIdx);
}

// Vector add kernel - real-world example
__global__ void vector_add(int *a, int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // 1. Basic kernel launch - show thread IDs
    printf("=== Kernel Launch: <<<2, 4>>> ===\n");
    printf("2 blocks × 4 threads = 8 threads total\n\n");
    print_ids<<<2, 4>>>();
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    printf("\n=== Kernel Launch: <<<1, 8>>> ===\n");
    printf("1 block × 8 threads = 8 threads total\n\n");
    print_ids<<<1, 8>>>();
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // 2. Practical example - vector addition
    printf("\n=== Vector Addition Example ===\n");
    int n = 10;
    int h_a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int h_b[10] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    int h_c[10] = {0};

    int *d_a, *d_b, *d_c;
    CUDA_ERROR_CHECK(cudaMalloc(&d_a, n * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_b, n * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_c, n * sizeof(int)));

    CUDA_ERROR_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice));

    // Calculate grid size: numBlocks = ceil(n / blockSize)
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;  // = 1 block for n=10
    printf("n=%d, blockSize=%d, numBlocks=%d, total threads=%d\n",
           n, blockSize, numBlocks, numBlocks * blockSize);
    vector_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    CUDA_ERROR_CHECK(cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost));

    printf("a:   ");
    for (int i = 0; i < n; i++) printf("%2d ", h_a[i]);
    printf("\nb:   ");
    for (int i = 0; i < n; i++) printf("%2d ", h_b[i]);
    printf("\na+b: ");
    for (int i = 0; i < n; i++) printf("%2d ", h_c[i]);
    printf("\n");

    CUDA_ERROR_CHECK(cudaFree(d_a));
    CUDA_ERROR_CHECK(cudaFree(d_b));
    CUDA_ERROR_CHECK(cudaFree(d_c));

    return 0;
}
