# Jetson AGX Orin Setup Manual

## Boot from nvme SSD
```
```

## ML Env

### pytorch
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cusparselt-cuda-12
sudo apt-get -y install cudss
sudo apt-get install -y python3-pip libopenblas-dev libcusparselt-dev

## cuda 12.6 + python 3.10
sudo apt-get install -y cuda-cupti-12-6
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

## cuda 12.9 + python 3.12
sudo apt-get install -y cuda-cupti-12-6
uv pip install torch==2.9.0 torchvision==0.24.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu129
```

# vllm
```
## cuda 12.6 + python 3.10
uv pip install vllm==0.10.2+cu126 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```