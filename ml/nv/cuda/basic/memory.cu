#include "utils.h"

/*
 * CUDA Runtime API - Memory Operations
 *
 * APIs used:
 *   cudaMalloc            - Allocate device memory
 *   cudaFree              - Free device memory
 *   cudaMemcpy            - Copy data between host and device
 *   cudaDeviceSynchronize - Block until device completes all tasks
 */

// nvcc -o build/memory basic/memory.cu && ./build/memory

int main() {
    // 1. cudaMalloc / cudaFree - memory allocation
    printf("=== Memory Allocation ===\n");
    size_t allocSize = 1024 * 1024 * 100;  // 100 MB
    void *d_ptr = NULL;

    size_t freeMem = 0, totalMem = 0;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("Before allocation - Free: %.2f GB\n", freeMem / (1024.0 * 1024.0 * 1024.0));

    CHECK_CUDA(cudaMalloc(&d_ptr, allocSize));
    printf("Allocated %.2f MB on device\n", allocSize / (1024.0 * 1024.0));

    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("After allocation - Free: %.2f GB\n", freeMem / (1024.0 * 1024.0 * 1024.0));

    CHECK_CUDA(cudaFree(d_ptr));
    printf("Freed device memory\n");

    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("After free - Free: %.2f GB\n\n", freeMem / (1024.0 * 1024.0 * 1024.0));

    // 2. cudaMemcpy - memory copy
    printf("=== Memory Copy ===\n");
    int n = 10;
    int h_src[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int h_dst[10] = {0};
    int *d_data = NULL;

    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_src, n * sizeof(int), cudaMemcpyHostToDevice));
    printf("Copied %d integers from host to device\n", n);

    CHECK_CUDA(cudaMemcpy(h_dst, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Copied %d integers from device to host\n", n);

    printf("Verification: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_dst[i]);
    }
    printf("\n\n");

    CHECK_CUDA(cudaFree(d_data));

    // 3. cudaDeviceSynchronize
    printf("=== Device Synchronization ===\n");
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Device synchronized successfully\n");

    return 0;
}
