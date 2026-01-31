#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK(expr) \
    do { \
        cudaError_t __result = (expr); \
        if (__result != cudaSuccess) { \
            const char* err_str = cudaGetErrorString(__result); \
            fprintf(stderr, "[CUDA Error] code=%d (%s) file=%s func=%s line=%d\n", \
                    __result, err_str ? err_str : "Unknown error", \
                    __FILE__, __func__, __LINE__); \
            exit(1); \
        } \
    } while (0)

#endif // CUDA_UTILS_H
