# cuda/cuDNN

## install driver
use upgrade from 570 to 580 as an example
```
## network repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

## remove 570
sudo apt remove --purge '^nvidia-.*570'
sudo apt autoremove
sudo apt autoclean

## install 580
sudo apt update
sudo apt install nvidia-driver-pinning-580
sudo apt install nvidia-driver-580

sudo reboot
```
ref: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html  
ref: https://www.nvidia.com/en-in/drivers/  

## install cuda
cuda 12.8
```
check https://developer.nvidia.com/cuda-12-8-0-download-archive

## add to ~/.bashrc:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

## switch cuda version:
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

## install cuDNN
```
check https://developer.nvidia.com/cudnn-downloads
```

## check version
cuda: `nvcc -V`
cuDNN: `dpkg -l | grep cudnn`

# nvidia-smi
- `watch nvidia-smi`: check gpu health 
    ```
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.274.02             Driver Version: 535.274.02   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA L40S                    Off | 00000000:38:00.0 Off |                    0 |
    | N/A   49C    P0             144W / 350W |  27426MiB / 46068MiB |     90%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
    |   1  NVIDIA L40S                    Off | 00000000:3A:00.0 Off |                    0 |
    | N/A   51C    P0             145W / 350W |  26136MiB / 46068MiB |     93%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
    |   2  NVIDIA L40S                    Off | 00000000:3C:00.0 Off |                    0 |
    | N/A   49C    P0             142W / 350W |  26136MiB / 46068MiB |     93%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
    |   3  NVIDIA L40S                    Off | 00000000:3E:00.0 Off |                    0 |
    | N/A   48C    P0             137W / 350W |  26136MiB / 46068MiB |     92%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+

    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |    0   N/A  N/A     12806      C   /usr/bin/python                           26130MiB |
    |    0   N/A  N/A     12807      C   /usr/bin/python                             426MiB |
    |    0   N/A  N/A     12808      C   /usr/bin/python                             426MiB |
    |    0   N/A  N/A     12809      C   /usr/bin/python                             426MiB |
    |    1   N/A  N/A     12807      C   /usr/bin/python                           26130MiB |
    |    2   N/A  N/A     12808      C   /usr/bin/python                           26130MiB |
    |    3   N/A  N/A     12809      C   /usr/bin/python                           26130MiB |
    +---------------------------------------------------------------------------------------+
    ```
- `nvidia-smi topo -m`: check multi-gpu connectivity 
    ```
        GPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID
    GPU0	 X 	SYS	SYS	SYS	0-47	0		N/A
    GPU1	SYS	 X 	SYS	SYS	0-47	0		N/A
    GPU2	SYS	SYS	 X 	SYS	0-47	0		N/A
    GPU3	SYS	SYS	SYS	 X 	0-47	0		N/A

    Legend:

    X    = Self
    SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
    NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
    PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
    PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
    PIX  = Connection traversing at most a single PCIe bridge
    NV#  = Connection traversing a bonded set of # NVLinks
    ```

- `nvidia-smi dmon`: device monitoring
```
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk 
# Idx      W      C      C      %      %      %      %      %      %    MHz    MHz 
    0    164     87      -    100     85      0      0      0      0   7501   1777 
    0    168     87      -     99     87      0      0      0      0   7501   1785 
    0    169     87      -     99     87      0      0      0      0   7501   1785 
    0    169     87      -    100     85      0      0      0      0   7501   1770 
    0    168     86      -    100     85      0      0      0      0   7501   1777 
    0    166     87      -    100     85      0      0      0      0   7501   1792 

sm: streaming multiprocessor utilization, similar to cpu%
mem: memory controller utilization
pwr: power draw
gtemp: gpu temperature
mtemp: memory temperature

- memory bottleneck: mem high, sm high, pwr does not hit limit
- cuda bottleneck: mem low, sm high, pwr hit limit
- cpu limit: mem low, sm < 50%, pwr does not hit limit
```

- `nvidia-smi dmon -s t`: pcie bandwidth usage monitoring
```
# gpu  rxpci  txpci 
# Idx   MB/s   MB/s 
    0     60     13 
    0     72     12 

rxpci: data received by pcie, host->gpu
txpci: data transmitted by pcie, gpu->host

| PCIe Gen | x16 Bandwidth (each direction) |
|----------|--------------------------------|
| Gen 3    | ~16 GB/s                       |
| Gen 4    | ~32 GB/s                       |
| Gen 5    | ~64 GB/s                       |

use pcie gen4 x 16 as an example, it has 16 lanes per direction, each lane has dedicated direction (rx or tx)
so it's impossible to see tx=64, rx=0
```

# NCCL
Known issues:
- OS error on docker + multi-gpu instance: `Cuda failure 304 'OS call failed or operation not supported on this OS'`. 
    `bash run_test_nccl.sh`: check PyTorch + NCCL + multi-GPU communication works on your system. 
    fix: add following opts in docker command: `--ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864`
    - `ipc=host`: so that container can direct access to the host's shared memory and IPC resources
    - `seccomp=unconfined`: disable docker syscall filtering, docker daemon can accept all the sys calls
    - `memlock`: some GPU operations require locking host memory, docker default to 64kb,limit too low may cause the error
    - `stack`: stack controls the maximum stack size per thread, default to 8M, may be too low
    ```
    sudo docker create \
    --runtime=nvidia \
    --gpus all \
    --net=host \
    --ipc=host \
    --shm-size="100g" \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v .:/workspace/verl \
    --name verl \
    verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 \
    sleep infinity
    ```
