# miles

## install
```
sudo docker pull radixark/miles:latest
sudo docker create --gpus all --ipc=host --shm-size=16g --ulimit nofile=1048576:1048576 --ulimit memlock=-1 --ulimit stack=67108864 -v .:/root/workspace --name miles radixark/miles:latest sleep infinity
sudo docker start miles
sudo docker exec -it miles bash

rm -r /root/miles
cd /root/
git clone https://github.com/radixark/miles.git
cd miles
pip install -e .
pip install -e --no-deps

# (dev)
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e --no-deps "python"

wandb login
```

## example - qwen3-4b
```
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024


cd /root/miles
source scripts/models/qwen3-4B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist

cd /root/miles
nohup bash scripts/run-qwen3-4B.sh 2>&1 &
```

## lora megatron development
```
sudo docker create --gpus all --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged --shm-size 32G --ulimit memlock=-1 --ulimit stack=67108864 -v /home/yisheng:/workspace --name miles_lora_megatron_yisheng radixark/miles:latest sleep infinity
sudo docker start miles_lora_megatron_yisheng
sudo docker exec -it miles_lora_megatron_yisheng bash

# clone miles
git clone --branch miles-lora-megatron --single-branch https://github.com/yushengsu-thu/miles.git 
cd miles
pip install -e .

# clone sglang
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python"

# clone megatron-bridge
git clone --branch merged-megatron-0.16.0rc0 --single-branch https://github.com/yushengsu-thu/Megatron-Bridge.git
cd Megatron-Bridge
pip install -e . --no-deps --no-build-isolation
pip install megatron-energon --no-deps
pip install multi-storage-client --no-deps
```