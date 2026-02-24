#include "utils.h"
#include <unistd.h>

/*
 * Demo: CUDA operations are async - CPU doesn't wait for GPU
 *
 * This shows that kernel launch returns immediately to CPU,
 * similar to spawning a thread.
 */

// nvcc -o build/sync basic/sync.cu && ./build/sync

// A slow kernel that takes ~1 second on GPU
__global__ void slow_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Busy loop to simulate work (~1 sec)
        for (long i = 0; i < 100000000L; i++) {
            data[idx] = i % 100;
        }
    }
}

int main() {
    int *d_data;
    CUDA_ERROR_CHECK(cudaMalloc(&d_data, sizeof(int)));

    printf("=== Async Demo ===\n\n");

    // Example 1: Without sync - CPU continues immediately
    printf("1. Without cudaDeviceSynchronize:\n");
    printf("   Launching slow kernel...\n");
    slow_kernel<<<1, 1>>>(d_data, 1);
    printf("   CPU reached here IMMEDIATELY (GPU still working in background)\n\n");

    // Wait for GPU before next example
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    printf("   GPU finished.\n\n");

    // Example 2: With sync - CPU waits
    printf("2. With cudaDeviceSynchronize:\n");
    printf("   Launching slow kernel...\n");
    slow_kernel<<<1, 1>>>(d_data, 1);
    printf("   Waiting for GPU...\n");
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    printf("   CPU reached here AFTER GPU finished\n\n");

    // Analogy summary
    printf("=== Analogy ===\n");
    printf("Thread:  pthread_create() -> printf() -> pthread_join()\n");
    printf("CUDA:    kernel<<<>>>()   -> printf() -> cudaDeviceSynchronize()\n");
    printf("Both launch work and return immediately to caller.\n");

    CUDA_ERROR_CHECK(cudaFree(d_data));
    return 0;
}
