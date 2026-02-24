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

void print_memory(const char* prefix) {
    if (prefix == nullptr) {
        prefix = "Memory";
    }
    size_t freeMem = 0, totalMem = 0;
    CUDA_ERROR_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    printf("[%s] - free: %.2f GB, total: %.2f GB\n", prefix, freeMem / (1024.0 * 1024.0 * 1024.0), totalMem / (1024.0 * 1024.0 * 1024.0));
}

void test_cudaMalloc() {
    size_t allocSize = 1024 * 1024 * 128; // 128 MiB
    void *d_ptr = nullptr;

    print_memory("Before malloc");
    CUDA_ERROR_CHECK(cudaMalloc(&d_ptr, allocSize));

    print_memory("After malloc");
    CUDA_ERROR_CHECK(cudaFree(d_ptr));

    print_memory("After free");

}

int main() {
    // 1. cudaMalloc / cudaFree - memory allocation
    printf("=== Memory Allocation ===\n");
    test_cudaMalloc();

    // 2. OOM error example - try to allocate more than available
    printf("=== OOM Error Example ===\n");

    size_t oomSize = (size_t)1024 * 1024 * 1024 * 1024;  // 1 TB - way more than available
    printf("Attempting to allocate: %.2f TB\n", oomSize / (1024.0 * 1024.0 * 1024.0 * 1024.0));

    void *d_oom = NULL;
    cudaError_t err = cudaMalloc(&d_oom, oomSize);
    if (err != cudaSuccess) {
        // CUDA maintains a "sticky" error state. 
        // After an error occurs, it stays set until you explicitly clear it.
        // Clear the error state so subsequent CUDA calls work
        cudaError_t err = cudaGetLastError();
        printf("OOM Error: %s (error code: %d)\n\n", cudaGetErrorString(err), err);
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

    // clang always pass copies, to modify its original we need to pass its address
    // if pass d_data, the pointer is copied and value is assigned to the copy of pointer internally
    // cannot pass to the pointer outside of it
    CUDA_ERROR_CHECK(cudaMalloc(&d_data, n * sizeof(int))); // void** devPtr, size_t size
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

    // 4. cudaDeviceSynchronize - block CPU until all GPU operations complete
    //    CUDA ops are async by default - CPU launches work and continues immediately
    //    Use cases:
    //      - Timing: ensure GPU work done before measuring time
    //      - Before reading results from device
    //      - Debugging: catch async errors at known point
    //    Example without sync:
    //      kernel<<<...>>>();      // CPU continues immediately
    //      printf("Done!\n");      // May print before GPU finishes!
    //    Example with sync:
    //      kernel<<<...>>>();
    //      cudaDeviceSynchronize(); // CPU waits here
    //      printf("Done!\n");       // GPU is guaranteed finished
    printf("=== Device Synchronization ===\n");
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    printf("Device synchronized successfully\n");

    return 0;
}
