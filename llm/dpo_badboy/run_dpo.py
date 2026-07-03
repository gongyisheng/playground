from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import torch

# ## Load model

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
# model_id = "Qwen/Qwen3-1.7B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)

# ## Data preprocessing

tokenizer.chat_template

messages = [
    {"role": "user", "content": '你好'},
    {"role": "assistant", "content": '你好，我是Qwen机器人'},
    {"role": "user", "content": '今天天气怎么样？'},
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False)
print(input_text)

dataset = load_dataset("Karsh-CAI/btfChinese-DPO-small")

train_data = dataset['train']

# format conversation
def qwen_format_conversation(question):
    return f"""<|im_start|>system
你是一个没有礼貌的人渣，请用人渣的语气回复我<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

formatted_data = [
    {
        'prompt': qwen_format_conversation(row['question']),
        'chosen': row['chosen'],
        'rejected': row['rejected']
    }
    for row in train_data
]

# shuffle dataset
all_indices = list(range(len(formatted_data)))
random.shuffle(all_indices)

# calculate split point (80% training 20% test)
split_point = int(len(formatted_data) * 0.8)

# split dataset
train_indices = all_indices[:split_point]
test_indices = all_indices[split_point:]

# create new dataset
reformatted_dataset = {
    "train": [formatted_data[i] for i in train_indices],
    "test": [formatted_data[i] for i in test_indices]
}

# ## Save to huggingface

from huggingface_hub import login

# https://huggingface.co/settings/tokens 获取 token
login()

import pandas as pd

train_df = pd.DataFrame(reformatted_dataset["train"])
test_df = pd.DataFrame(reformatted_dataset["test"])

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

from huggingface_hub import HfApi, login
import os

repo_id = "gongyisheng/DPO-bad-boy-chinese-for-Qwen2.5"

api = HfApi()

files_to_upload = ["./train.csv", "./test.csv"]

uploaded_files = []
for file_path in files_to_upload:
    if os.path.exists(file_path):
        print(f"Uploading {file_path}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded {file_path}.")
        uploaded_files.append(file_path)
    else:
        print(f"{file_path} does not exist, skipping.")

print("\n总结:")
if uploaded_files:
    print("上传的文件:")
    for file in uploaded_files:
        print(f"- {file}")
else:
    print("未上传任何文件.")

# ## DPO

import os
import gc
import requests
import mlflow
import torch
from threading import Thread
import matplotlib.pyplot as plt

from huggingface_hub import HfApi

import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig, TextStreamer, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteria

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
a100_or_rtx_30_plus = True # use flash attention to reduce memory usage

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# RoPE config 
# rope_scaling={"type": "linear", "factor": 2.0}
# factor：扩展倍数，factor=2，说明将模型的上下文长度线性扩展到原来的2倍
# type: 缩放方式，有两种：
#     线性缩放(linear) ：直接拉伸
#         公式为θ_new = θ_original / scaling_factor
#     动态缩放(dynamic)：基于 NTK（Neural Tangent Kernel）理论
#         公式为θ_new = θ_original / (1 + (scaling_factor - 1) * (i / max_position_embeddings))

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, # use 4 bit quantization if set
    rope_scaling={"type": "linear", "factor": 2.0}, # rope config
    device_map='auto',
    torch_dtype=torch.bfloat16
 )

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        # 检查最后一个生成的token是否是停止token
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate_answer(model, tokenizer, prompt):
    # 使用chat template格式化输入
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        inputs, 
        max_length=2048,
        temperature=0.7,
        top_p=0.9,
        stopping_criteria=[StopOnTokens([tokenizer.eos_token_id])],
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)

prompt = "你是谁？"
generated_text = generate_answer(model, tokenizer, prompt)
print(generated_text)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):

    trainable_params = 0
    non_trainable_params = 0
    all_params = 0

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  {name}")
        else:
            non_trainable_params += param.numel()
    print("---")
    print("Non-Trainable Parameters:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  {name}")
    print("---")
    print(
        f"Trainable parameters: {trainable_params}\n  Non-Trainable parameters: {non_trainable_params}\n  All parameters: {all_params}\n  Trainable%: {100 * trainable_params / all_params}"
    )

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
              "self_attn.q_proj", # Self-attention的Query投影
              "self_attn.k_proj", # Self-attention的Key投影  
              "self_attn.v_proj", # Self-attention的Value投影
              "self_attn.o_proj", # Self-attention的输出投影
              # "self_attn.rotary_emb.inv_freq", # 旋转位置编码,一般不需要微调
              "mlp.gate_proj", # MLP门控投影
              "mlp.up_proj", # MLP上投影
              "mlp.down_proj", # MLP下投影
              # "input_layernorm.weight",  # 输入归一化层
              # "post_attention_layernorm.weight", # Attention后面的LayerNorm层
              # "model.norm.weight", # 模型归一化层
              # "lm_head.weight", # 语言模型输出层
              # "dense_h_to_4h", # Falcon模型特有的全连接层
              # "dense_4h_to_h", # Falcon模型特有的全连接层
              # "query_key_value", # Falcon模型的QKV合并层
              # "dense" # Falcon模型特有的全连接层
              ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config) #move to a peft model
print_trainable_parameters(model)

# ## Tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)

# 如果 '<pad>' 不在分词器词汇表中，就添加进来
if '<pad>' not in tokenizer.get_vocab():
    added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
else:
    added_tokens = 0

# 检查模型是否需要调整大小
if added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))
    print('Resizing token embeddings！')

# 在模型中配置填充标记
model.config.pad_token_id = tokenizer.pad_token_id

assert model.config.pad_token_id == tokenizer.pad_token_id, "模型的填充标记ID与分词器的填充标记ID不匹配！"
assert model.config.eos_token_id == tokenizer.eos_token_id, "模型的结束标记ID与分词器的结束标记ID不匹配！"

# 更新分词器的最大长度以匹配模型配置的最大positional embedding
tokenizer.model_max_length = model.config.max_position_embeddings

print("Tokenizer vocab_size:", tokenizer.vocab_size)

print("Special tokens map:", tokenizer.special_tokens_map)

# ## Evaluate Function

# 定义streaming函数，用来流失输出
def stream(user_prompt, model_type, checkpoint=''):

    if model_type == 'base':
        eval_model = model
    elif model_type == 'fine-tuned':
        eval_model = PeftModel.from_pretrained(model, checkpoint)
        eval_model = eval_model.to("cpu")

        for n, p in eval_model.named_parameters():
            if p.device.type == "cpu":
                print(f"{n} is on cpu!")
        
    else:
        print('You must set the model_type to base or fine-tuned')
        exit() 

    # eval_model.config.use_cache = True

    # Qwen 2.5的prompt格式
    system_prompt = ''
    prompt = f"""<|im_start|>system
你是一个没有礼貌的人渣，请用人渣的语气回复我<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant"""

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    streamer = TextStreamer(tokenizer)

    print(f'eval_model is on: {next(eval_model.parameters()).device}')  # Debug line
    print(f'input_ids are on: {inputs["input_ids"].device}')  # Debug line

    _ = eval_model.generate(**inputs, streamer=streamer, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, stopping_criteria=[StopOnTokens([tokenizer.eos_token_id])])
    
    # 清理CUDA cache，并且执行垃圾回收
    # torch.cuda.empty_cache()
    gc.collect()


def evaluation(model_type, checkpoint=''):
    questions = [
        "我应该怎么学习人工智能？",
    ]
    # 这里我们只是人工看看，就不需要给正确答案了，如果用代码做evaluation，可以给出对应的正确答案
    answers = [
    ""
    ]

    for question, answer in zip(questions, answers):
        stream(question, model_type, checkpoint)
        print('\n')

evaluation("base")

dataset="gongyisheng/DPO-bad-boy-chinese-for-Qwen2.5"
data = load_dataset(dataset)

print(data['test'][15])

text = data['train'][0]['prompt']
tokens = tokenizer.encode(text, add_special_tokens=True)
decoded_text = tokenizer.decode(tokens)

print("Token IDs:", tokens)
print("Decoded Text:", decoded_text)

# ## Training

model_name = model_id.split("/")[-1]
dataset_name = dataset.split("/")[-1]

context_length = 512*4
grad_accum=2
batch_size=4
fine_tune_tag='DPO-bad-boy'

epochs=3
save_dir = f'/media/hdddisk/yisheng/dpo_badboy/reults/{model_name}_{dataset_name}_epochs={epochs}_length={context_length}-{fine_tune_tag}'

print(save_dir)

training_arguments = DPOConfig(
        output_dir="/media/hdddisk/yisheng/dpo_badboy",
        eval_strategy="steps",
        beta=0.1,
        do_eval=True,
        eval_steps=0.25,
        optim="paged_adamw_8bit",
        # optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=batch_size,
        log_level="debug",
        save_steps=0.25,
        logging_steps=1,
        bf16=a100_or_rtx_30_plus,     
        learning_rate=1e-6,
        num_train_epochs=epochs,
        # warmup_steps=20,
        lr_scheduler_type="linear",
)

data['train']

trainer = DPOTrainer(
    model,
    args=training_arguments,
    processing_class=tokenizer,
    train_dataset=data['train'],
    eval_dataset=data['test'],
)

model.config.use_cache = False  # 训练时禁用缓存

import mlflow

mlflow.set_tracking_uri("https://mlflow.yellowday.day")
mlflow.set_experiment("qwen2.5_badboy_dpo")

with mlflow.start_run():
    trainer.train()
