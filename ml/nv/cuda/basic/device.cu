#include "utils.h"

/*
 * CUDA Runtime API - Device Management
 *
 * APIs used:
 *   cudaGetDeviceCount    - Get number of CUDA-capable devices
 *   cudaGetDevice         - Get current device ID
 *   cudaGetDeviceProperties - Get device hardware properties
 *   cudaDeviceGetAttribute - Get specific device attribute (preferred for deprecated fields)
 *   cudaMemGetInfo        - Get free and total device memory
 *   cudaDeviceReset       - Destroy all allocations and reset device state
 */

// nvcc -o build/device basic/device.cu && ./build/device

int main() {
    // 1. Get device count
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    printf("=== CUDA Device Count ===\n");
    printf("Number of CUDA devices: %d\n\n", deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    // 2. Get/Set current device
    int currentDevice = 0;
    CHECK_CUDA(cudaGetDevice(&currentDevice));
    printf("=== Current Device ===\n");
    printf("Current device: %d\n\n", currentDevice);

    // 3. Get device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, currentDevice));
    printf("=== Device Properties ===\n");
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared memory per block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads dim: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid size: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    // Note: clockRate and memoryClockRate deprecated in CUDA 12+
    // Use cudaDeviceGetAttribute for these values instead
    int clockRate, memoryClockRate;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, currentDevice);
    cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, currentDevice);
    printf("Clock rate: %.2f GHz\n", clockRate / 1e6);
    printf("Memory clock rate: %.2f GHz\n", memoryClockRate / 1e6);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("L2 cache size: %.2f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
    printf("Multi-processor count: %d\n\n", prop.multiProcessorCount);

    // 4. cudaMemGetInfo - get free and total memory
    size_t freeMem = 0, totalMem = 0;
    CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    printf("=== Memory Info (cudaMemGetInfo) ===\n");
    printf("Free memory: %.2f GB\n", freeMem / (1024.0 * 1024.0 * 1024.0));
    printf("Total memory: %.2f GB\n", totalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Used memory: %.2f GB\n\n", (totalMem - freeMem) / (1024.0 * 1024.0 * 1024.0));

    // 5. cudaDeviceReset
    printf("=== Device Reset ===\n");
    CHECK_CUDA(cudaDeviceReset());
    printf("Device reset successfully\n");

    return 0;
}
