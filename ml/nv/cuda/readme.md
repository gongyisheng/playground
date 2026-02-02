# CUDA readme

## GPU hardware arch
A GPU contains multiple SMs

### memory 
- registers (per thread, fastest)
- L1 cache (SRAM, ~100KB), `__shared__` lives here
- L2 cache (SRAM, ~40MB)
- Global memory (HBM/GDDR, ~8-80GB), `cudaMalloc` lives here

| Memory Type | Location | Latency | Scope |
|-------------|----------|---------|-------|
| Registers | On-chip | 1 cycle | Per thread |
| L1 Cache | On-chip | ~20-30 cycles | Per block |
| L2 Cache | On-chip | ~200 cycles | All SMs |
| Global Memory | Off-chip VRAM | ~400-600 cycles | All threads |

## execution model

CUDA organizes parallel execution in a two-level hierarchy:

```
Grid (entire kernel launch)
├── Block (0,0)          ├── Block (0,1)
│   ├── Thread 0         │   ├── Thread 0
│   ├── Thread 1         │   ├── Thread 1
│   └── ...              │   └── ...
└── Block (1,0)          └── Block (1,1)
    └── ...                  └── ...
```

kernel launch syntax: `kernel<<<grid, block>>>(args)`

| Level | Description | Size Limit |
|-------|-------------|------------|
| Grid | Collection of blocks | Millions of blocks |
| Block | Group of cooperating threads | 1024 threads max |
| Thread | Single execution unit | - |

key properties:
- Threads in same block can share memory and synchronize
- Blocks are independent and run in any order
- Grid scales automatically across available SMs

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
