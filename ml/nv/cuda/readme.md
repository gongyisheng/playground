# CUDA readme

## nvcc

When compiling with `nvcc`, CUDA runtime headers are automatically included.

### headers

| Header | Location | Provides |
|--------|----------|----------|
| `cuda_runtime.h` | `/usr/local/cuda/include/cuda_runtime.h` | `cudaDeviceSynchronize`, `cudaMalloc`, `cudaMemcpy`, etc. |
| `vector_types.h` | `/usr/local/cuda/include/vector_types.h` | `dim3`, `int2`, `float4`, etc. |

### types
- dim3: built-in struct for specifying grid and block dimensions:
    ```cuda
    struct dim3 {
        unsigned int x;
        unsigned int y;
        unsigned int z;
    };

    // Constructors
    dim3 a;           // x=1, y=1, z=1
    dim3 b(4);        // x=4, y=1, z=1
    dim3 c(4, 2);     // x=4, y=2, z=1
    dim3 d(4, 2, 3);  // x=4, y=2, z=3
    ```
