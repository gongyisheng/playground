# Compile
c compile: `gcc <input.c> -o <output>`  
cuda compile: `nvcc <input.cu> -o <output>` 

# Common workflow of cuda program
1. Allocate host memory and initialized host data
2. Allocate device memory
3. Transfer input data from host to device memory
4. Execute kernels
5. Transfer output from device memory to host
Note: You can't ask cuda code to write result directly to host memory