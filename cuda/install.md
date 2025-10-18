# Install CUDA
cuda 12.8
```
check https://developer.nvidia.com/cuda-12-8-0-download-archive

add to ~/.bashrc:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

switch cuda version:
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.8 /usr/local/cuda
```

output
```
yisheng@pc:/usr/local$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```