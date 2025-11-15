# verl
## install from source
```
git clone https://github.com/volcengine/verl.git # official version
git clone https://github.com/gongyisheng/verl.git # customized version

cd verl
pip3 install --no-deps -e .
cd ..
```

## install by image
```
sudo docker create --runtime=nvidia --gpus all --net=host --shm-size="100g" --cap-add=SYS_ADMIN --security-opt seccomp=unconfined  -v .:/workspace/verl --name verl-dev verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 sleep infinity

# if you get NCCL OS error:
sudo docker create --runtime=nvidia --gpus all --net=host --ipc=host --shm-size="100g" --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 -v .:/workspace/verl --name verl-dev verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 sleep infinity

sudo docker start verl 
sudo docker exec -it verl bash

# install from source

# wandb
wandb login
```