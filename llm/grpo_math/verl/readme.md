# Use verl to run GRPO training

## install
```
# use verl docker image
sudo docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 sleep infinity

git clone https://github.com/volcengine/verl.git
cd verl
pip3 install --no-deps -e .

wandb login
```

## run experiment
```
mkdir -p /root/datasets/gsm8k/verl
cd llm/grpo_math/verl
python3 data_preprocess.py
bash train_2x4090.sh
```

## configuration
```
1. set actor_rollout_ref.rollout.tensor_model_parallel_size=1 if use single process training
```

## result
Qwen3-0.6B-Base:
After training for 100 steps (on 2x4090), reward stable at 0.75
The inference result shows that base model already has enough math ability for gsk8m, but do not know to follow instructions and return output (after ####). RL teaches model to return output properly.