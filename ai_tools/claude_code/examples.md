## claude code YOLO task example
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
result: 
```
30-50 min single task
quite good on 3060 12g (fail) and a10g (success) after about 6 rounds, both get right result
note that the temp log files are everywhere, may tell LLM somewhere to put, away from scripts / changlog
```

## prompt engineering
env:
```
vm agent, based on dev instance env
```
prompt:
```
I want you to do prompt engineering for this agent, your goal is to keep vendor accuracy of @../scripts/run_eval_normal.sh as high as >93% and keep it concise and organized, consume tokens as low as possible. you are allowed to update @../prompts.py , or update @../utils.py to add some logs, you are not allowed to update any other files. when update prompts you are not allowed to directly add eval examples to prompt, but you are allowed to extract the abstract rules and reorganize the prompt. do controled experiment when possible, document all your changes in a changlog.md file. work hard and run multiple rounds until you cannot improve it
```
result:
```
30 min task
85%->93% after rounds, but fail to improve, start to see tendency of hacking (hardcode example to prompt), has to stop and ask it to reorganize and restucture the prompt, and focus on common pattern of failed case, suggest infra change, after that see accuracy ~95%
```

## vendor negotiation agent
env:
N/A
prompt:
```
your job is to create an llm agent for helping finance team to negotiate with vendors. the agent should be able to use the benchmark tool defiled in @sh_bench_mcp/ to answer the questions defined in @../Agent Eval QA.csv , and the agent should be able to connect to a frontend (copilotkit). the agent should be well designed to handle different kinds of corner cases. 
you should deliver the fully functional backend agent service and tests (including the eval test), make sure the agent can pass the eval test and get the expected answer, and a frontend that can connect to it successfully. use claude agent development kit to develop it. 
you can create an virtual env with uv, and only install minimal dependencies that will be used in the project, record all install dependencies in requirements.txt

refs:
uv: https://github.com/astral-sh/uv
claude adk: https://github.com/anthropics/claude-agent-sdk-python
copilotkit connect to python backend: https://docs.copilotkit.ai/direct-to-llm/guides/backend-actions/remote-backend-endpoint
```
result:
```
15 min task
not quite good, the tests are hardcoded, need to tune test and run again
```