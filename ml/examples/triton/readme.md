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