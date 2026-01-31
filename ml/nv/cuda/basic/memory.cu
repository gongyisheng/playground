#include "utils.h"

/*
 * CUDA Runtime API - Memory Operations
 *
 * APIs used:
 *   cudaMemGetInfo        - Get free and total device memory
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
    CUDA_ERROR_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    printf("Before allocation - Free: %.2f GB\n", freeMem / (1024.0 * 1024.0 * 1024.0));

    CUDA_ERROR_CHECK(cudaMalloc(&d_ptr, allocSize));
    printf("Allocated %.2f MB on device\n", allocSize / (1024.0 * 1024.0));

    CUDA_ERROR_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    printf("After allocation - Free: %.2f GB\n", freeMem / (1024.0 * 1024.0 * 1024.0));

    CUDA_ERROR_CHECK(cudaFree(d_ptr));
    printf("Freed device memory\n");

    CUDA_ERROR_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    printf("After free - Free: %.2f GB\n\n", freeMem / (1024.0 * 1024.0 * 1024.0));

    // 2. OOM error example - try to allocate more than available
    printf("=== OOM Error Example ===\n");
    CUDA_ERROR_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    printf("Available memory: %.2f GB\n", freeMem / (1024.0 * 1024.0 * 1024.0));

    size_t oomSize = (size_t)1024 * 1024 * 1024 * 1024;  // 1 TB - way more than available
    printf("Attempting to allocate: %.2f TB\n", oomSize / (1024.0 * 1024.0 * 1024.0 * 1024.0));

    void *d_oom = NULL;
    cudaError_t err = cudaMalloc(&d_oom, oomSize);
    if (err != cudaSuccess) {
        printf("OOM Error: %s (error code: %d)\n\n", cudaGetErrorString(err), err);
        // Clear the error state so subsequent CUDA calls work
        cudaGetLastError();
    } else {
        printf("Allocation succeeded (unexpected!)\n\n");
        cudaFree(d_oom);
    }

    // 3. cudaMemcpy - memory copy
    printf("=== Memory Copy ===\n");
    int n = 10;
    int h_src[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int h_dst[10] = {0};
    int *d_data = NULL;

    CUDA_ERROR_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
    CUDA_ERROR_CHECK(cudaMemcpy(d_data, h_src, n * sizeof(int), cudaMemcpyHostToDevice));
    printf("Copied %d integers from host to device\n", n);

    CUDA_ERROR_CHECK(cudaMemcpy(h_dst, d_data, n * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Copied %d integers from device to host\n", n);

    printf("Verification: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_dst[i]);
    }
    printf("\n\n");

    CUDA_ERROR_CHECK(cudaFree(d_data));

    // 4. cudaDeviceSynchronize
    printf("=== Device Synchronization ===\n");
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    printf("Device synchronized successfully\n");

    return 0;
}
