## claude code task example
focus on tasks that can run for hours

## RL example local setup
env:
```
miles docker image
https://github.com/radixark/miles/pull/409
```
prompt:
```
note that i m running the code on a rtx3060 instance, the gpu only has 12gb. your job is to 1. update @examples/reproducibility/run-qwen2.5-0.5B-gsm8k.sh marco configs to make sure it can run successfully. 2. tune the macro configs to allow the training process to be stable (batch size as big as possible, truncation ratio as low as possible). 
for miles args you can ref to @miles/utils/arguments.py , for sglang args you can run `python3 -m sglang.launch_server --help`, for gpu monitoring you can run nvidia-smi (may run in background)
your should document your changes in args (ideally in a experiment notes md file), when nescessary, do controlled experiment to find out the causal relationship.
after everything is done, make a document of arguments updated/added, with the reason behind, and failure attempts if has
```