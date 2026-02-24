# triton
A python compiler module for writing highly efficient GPU kernels, by openai

## install
```
uv pip install triton
```
- issue: Python.h not found:
    ```
    error: 
    /tmp/tmpixs2etub/__triton_launcher.c:7:10: fatal error: Python.h: No such file or directory
        7 | #include <Python.h>
        |          ^~~~~~~~~~
    
    solve: 
    sudo apt update
    sudo apt install python3.12-dev
    ```

## concepts
1. tl.program_id(axis)
    ```
    returns the index of current program instance (thread block)

    tl.program_id(0) ---> CUDA blockIdx.x
    tl.program_id(1) ---> CUDA blockIdx.y
    tl.program_id(2) ---> CUDA blockIdx.z
    ```
