# # 环境准备
# 
# 首先我们需要安装一些必要的依赖包:
# 
# - transformers: Hugging Face的主要工具包
# - bitsandbytes: 用于模型量化
# - peft: 参数高效微调工具
# - accelerate: 分布式训练加速
# - datasets: 数据集处理工具
# - trl: DPO训练的核心包
# - flash-attn: 快速注意力计算
# - sentencepiece: 分词器
# - wandb: 训练过程可视化

# !pip install -U transformers bitsandbytes peft accelerate datasets trl flash-attn sentencepiece wandb

# 导入相关的包

import os
import gc
import requests
import wandb
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

# ## Weights & Biases (wandb)
# 
# Weights & Biases (wandb) 是一个非常流行的机器学习实验跟踪工具。它可以帮助我们:
# 
# 1. 可视化训练过程 - 实时查看loss、accuracy等指标的变化
# 2. 记录实验配置 - 保存每次实验的超参数和设置
# 3. 对比实验结果 - 方便比较不同实验的效果
# 4. 版本管理 - 追踪模型和数据集的版本
# 5. 协作分享 - 团队成员可以查看彼此的实验
# 
# 我们将使用wandb来监控DPO训练的整个过程。首先需要去注册一个wandb账号,然后运行下面代码登录，之后训练结果会自动被记录:

wandb.login()

# 设置模型缓存目录和硬件环境参数
# 
# **cache_dir**: 模型缓存的本地目录路径,空字符串表示使用默认路径
# 
# **a100_or_rtx_30_plus**: 是否使用A100或RTX 30系列及以上的GPU，如果是这些比较新的GPU,可以设为True来启用flash attention加速训练

cache_dir=''
a100_or_rtx_30_plus = False

# ## 加载模型和分词器
# 
# 1. **选择基础模型** - 使用Qwen2.5-1.5B-Instruct作为基础模型
# 
# 2. **配置量化参数**:
#    - 使用4bit量化减少显存占用
#    - 启用double quantization进一步压缩
#    - 使用nf4量化类型
#    - 计算使用bfloat16精度
# 
# 3. **加载模型介绍**:
#    - 使用auto device map自动分配设备
#    - 使用bfloat16，方便后面混合精度训练
#    - 根据GPU型号决定是否启用flash attention 2.0
#    - 支持自定义缓存目录
# 
# 4. **RoPE配置**：
#     - `rope_scaling={"type": "linear", "factor": 2.0}`
#     - `factor`：扩展倍数，`factor=2`，说明将模型的上下文长度线性扩展到原来的2倍
#     - `type`: 缩放方式，有两种：
#         - 线性缩放(`linear`) ：直接拉伸，公式为`θ_new = θ_original / scaling_factor`
#         - 动态缩放(`dynamic`)：基于 NTK（Neural Tangent Kernel）理论，公式为`θ_new = θ_original / (1 + (scaling_factor - 1) * (i / max_position_embeddings))`
# 
# 5. **device_map**: 是用于控制模型如何在可用设备（GPU、CPU）上分配的参数。
#     - device_map = "balanced"  # 在多GPU间平衡分配
#     - device_map = "sequential"  # 按顺序填充GPU
#     - device_map = "cpu"  # 全部加载到CPU
#     - device_map = 0  # 全部加载到第一个GPU
# 
# 6. **torch_dtype**: 指定模型权重的数据类型（精度），影响内存占用和速度。
#     - `torch.float32/torch.float`：标准的32位浮点精度，最高精度，但内存占用最大
#     - `torch.float16`：内存占用是float32的一半，可能会有数值不稳定的问题
#     - `torch.bfloat16`（建议）:16位的"大脑浮点"格式，比float16更稳定，特别适合大语言模型，在保持数值稳定性的同时节省内存
#     - `torch.int8`:8位整数格式,用于极端量化场景,精度损失最大，但内存占用最小
# 
# 7. **Flash Attention**:
#    - 是否支持启用Flash Attention加速训练，仅支持A100/30X0系及以上，类似V100之类的显卡是不支持的

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config, # 如果开启，则使用4bit量化
    # rope_scaling={"type": "linear", "factor": 2.0},
    device_map='auto',
    torch_dtype=torch.bfloat16,
    use_flash_attention_2= a100_or_rtx_30_plus, # 降低内存需求，如果是A100及RTX 30系列及以上可以选择为True
    cache_dir=cache_dir)

# - **StopOnTokens类**
#    - 继承自StoppingCriteria
#    - 用于控制文本生成的停止条件
#    - 当生成的token是预定义的stop token时结束生成
# 
# - **generate_answer函数**
#    - 输入参数:model(模型)、tokenizer(分词器)、prompt(提示文本)
#    - 使用chat template格式化用户输入
#    - 设置生成参数:
#      - max_length=2048:控制生成文本最大长度
#      - temperature=0.7:控制生成文本的随机性
#      - top_p=0.9:使用nucleus sampling控制采样范围
#    - 使用StopOnTokens控制生成停止条件
#    - 返回解码后的生成文本

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

# 在训练之前测试一下模型的输出

tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)

prompt = "你是谁？"
generated_text = generate_answer(model, tokenizer, prompt)
print(generated_text)

# 确保模型没有错误加载到了内存(meta)

for n, p in model.named_parameters():
    if p.device.type == "meta":
        print(f"{n} is on meta!")

# ### 启用梯度检查点和kbit训练
# 
# `gradient_checkpointing_enable()` 开启梯度检查点,可以减少显存占用,但会略微降低训练速度。
# 
# `prepare_model_for_kbit_training` 为kbit量化训练做准备,包括:
# 
# 1. 启用输入梯度
# 2. 启用梯度检查点  
# 3. 确保模型参数在正确的设备上

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

# ### 模型量化和LoRA配置
# 
# 下面我们使用LoRA(Low-Rank Adaptation)做fine-tuning。LoRA通过在原始权重旁边添加小的可训练rank分解矩阵来实现高效微调,可以大幅度减少训练参数量。
# 
# 主要配置说明:
# 
# - r=8: LoRA的秩,决定了低秩矩阵的大小
# - lora_alpha=32: LoRA的缩放因子
# - target_modules: 需要应用LoRA的模块列表,主要包括attention和MLP相关层
# - lora_dropout=0.1: LoRA的dropout率
# - bias="none": 不对bias进行训练
# - task_type="CAUSAL_LM": 指定任务类型为因果语言模型
# 
# 这些配置参考了Llama的成功经验,在保持模型性能的同时显著降低了训练成本。

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

# 只有0.59%的trainable parameters

model = get_peft_model(model, peft_config) #move to a peft model
print_trainable_parameters(model)

# ## Tokenizer 配置
# 
# ### 加载Tokenizer
# 
# 从预训练模型加载对应的 tokenizer

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

# ## 准备Evaluation函数
# 
# 这一部分我们将设置评估环境,主要包括:
# 
# 1. 创建一个流式输出函数`stream`,用于实时查看模型生成的文本
# 2. 支持评估原始基础模型和经过DPO微调后的模型
# 3. 设置系统提示词,让模型以"暴躁"的语气回复
# 4. 处理输入提示词的格式化和模型推理
# 
# 下面让我们看看具体的代码实现:

# 定义streaming函数，用来流失输出
def stream(user_prompt, model_type, checkpoint=''):

    if model_type == 'base':
        eval_model = model
    elif model_type == 'fine-tuned':
        eval_model = PeftModel.from_pretrained(model, checkpoint)
        eval_model = eval_model.to("cuda")

        for n, p in eval_model.named_parameters():
            if p.device.type == "cpu":
                print(f"{n} is on cpu!")
        
    else:
        print('You must set the model_type to base or fine-tuned')
        exit() 

    eval_model.config.use_cache = True

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
    torch.cuda.empty_cache()
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

# 原始的模型还是很有礼貌的，我们需要进行DPO微调，让它变得“暴躁”起来！

evaluation("base")

# ## 加载数据集
# 
# 这里可以换成你自己构建的数据集，或者直接用我这个

dataset="MLZoo/DPO-bad-boy-chinese-for-Qwen2.5"
data = load_dataset(dataset)

# 打印一下，可以看到chosen是我们要模型学习的暴躁回答，rejected是要遗忘的正经回答。

print(data['test'][15])

# 我们可以使用tokenizer.encode把文字进行encoding。
# 
# 也可以使用tokenizer.decode把token反向解析回文字。

text = data['train'][0]['prompt']
tokens = tokenizer.encode(text, add_special_tokens=True)
decoded_text = tokenizer.decode(tokens)

print("Token IDs:", tokens)
print("Decoded Text:", decoded_text)

# ## 配置训练
# 
# ### TRL Trainer

model_name = model_id.split("/")[-1]
dataset_name = dataset.split("/")[-1]

context_length = 512*4
grad_accum=2
batch_size=4
fine_tune_tag='DPO-bad-boy'

epochs=3
save_dir = f'./results/{model_name}_{dataset_name}_epochs={epochs}_length={context_length}-{fine_tune_tag}'

print(save_dir)

# ### DPO 训练配置
# 
# 
# #### 基本配置
# 
# - `output_dir`: 模型和日志的输出目录
# - `evaluation_strategy`: 评估策略,这里设为按步数评估
# - `beta`: DPO算法的温度系数,控制偏好强度,这里设为0.1
# - `do_eval`: 是否进行评估
# - `eval_steps`: 每训练25%的数据进行一次评估
# - `bf16`: 是否启用混合精度，也是要A100/30X0以上才可以
# 
# #### 优化器配置  
# 
# - `optim`: 使用 AdamW 优化器的 PyTorch 实现，如果显存不足，可以改为稍慢一些的`bitsandbytes`库提供的8位量化版本的AdamW `paged_adamw_8bit`，
# - `learning_rate`: 学习率设为 1e-6
# - `lr_scheduler_type`: `linear`是使用线性学习率调度器，其他选择：
#     - `cosine`: 余弦退火调度
#     - `constant`: 固定学习率
#     - `constant_with_warmup`: 预热后固定学习率
#     - `polynomial`: 多项式衰减
#     - `cosine_with_restarts`: 带重启的余弦退火
# 
# #### 批处理配置
# 
# - `per_device_train_batch_size`: 每个设备的训练批次大小
# - `per_device_eval_batch_size`: 每个设备的评估批次大小  
# - `gradient_accumulation_steps`: 梯度累积步数,用于增大等效批次大小
# 
# #### 训练进度记录
# 
# - `save_steps`: 每训练25%的数据保存一次模型
# - `logging_steps`: 每步都记录训练日志
# - `log_level`: 日志级别设为debug,便于调试
# - `num_train_epochs`: 训练轮数

training_arguments = DPOConfig(
        output_dir="./results",
        evaluation_strategy="steps",
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

# 注意，训练的时候，不要使用注意力缓存（attention cache），降低显存消耗。因为在训练时我们是用到teacher forcing，不需要逐个token预测。
# 这是一个非常有趣的点，没有系统学习过Transformer的同学会忽略：
#    - 训练时我们已经有完整的目标序列（ground truth）
#    - 模型可以并行地看到整个输入序列和目标序列
#    - 使用 teacher forcing 技术，即使用真实的前一个token而不是模型预测的token
#    
# `use_cache`设置为`True`时，模型会缓存每一层的key和value值，这个在生成任务中特别有用，可以避免重复计算之前的token的attention值
#

trainer = DPOTrainer(
    model,
    args=training_arguments,
    train_dataset=data['train'],
    eval_dataset=data['test'],
    tokenizer=tokenizer,
)

model.config.use_cache = False  # 训练时禁用缓存
trainer.train()

# ## 绘制训练和评估损失

# 初始化存储训练和评估过程中的损失值和步数的列表
train_losses = []  # 训练损失值列表
eval_losses = []   # 评估损失值列表
train_steps = []   # 训练步数列表
eval_steps = []    # 评估步数列表

# 从训练器的日志历史中提取损失值和对应步数
for entry in trainer.state.log_history:
    # 提取训练过程的损失值和步数
    if 'loss' in entry:
        train_losses.append(entry['loss'])
        train_steps.append(entry['step'])
    # 提取评估过程的损失值和步数    
    if 'eval_loss' in entry:
        eval_losses.append(entry['eval_loss'])
        eval_steps.append(entry['step'])

# 绘制训练过程的损失曲线图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(train_steps, train_losses, label='训练损失', color='blue', linestyle='-')
plt.plot(eval_steps, eval_losses, label='验证损失', color='red', linestyle='--')
plt.xlabel('训练步数')
plt.ylabel('损失值')
plt.title('模型训练和验证损失曲线')
plt.grid(True)  # 添加网格线
plt.legend()
plt.show()

# ## 模型效果验证
# 
# 这次可以把推理加速改回来

# 加速推理
model.config.use_cache = True

evaluation("base")

# # Merge Adapters and Save Model to Hub

new_model = f"MLZoo/{model_name}-{fine_tune_tag}"

# 保存trainable参数

os.makedirs(save_dir, exist_ok=True)
trainable_params_state_dict = {n: p.data for n, p in model.named_parameters() if p.requires_grad}

final_save_path = os.path.join(save_dir, "trainable_params_final.bin")
torch.save(trainable_params_state_dict, final_save_path)

# 保存LoRA权重

# upload the trainable_params as well

from huggingface_hub import HfApi

# Initialize the HfApi class
api = HfApi()

# Specify the repository where you want to upload the files
repo_id = adapter_model

# Array of local file paths you want to upload
local_file_paths = [
    save_dir + "/trainable_params_final.bin",
]

# Loop through each file and upload it
for local_file_path in local_file_paths:
    # Extract the file name from the local file path
    file_name = local_file_path.split("/")[-1]

    # Specify the path where you want the file to be uploaded in the repository
    path_in_repo = file_name  # Using file_name directly, adjust as needed
    
    # Upload the file
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",  # Assuming it's a model; can be "dataset" or "space" as well
    )
    print(f"Uploaded {file_name} to {repo_id}")

# 把原始模型权重和LoRA权重整合到一起再push到huggingface

model = model.merge_and_unload()

# 把tokenizer push到huggingface

tokenizer.push_to_hub(new_model, use_auth_token=True)

# 把model+LoRA权重 push到huggingface

model.push_to_hub(new_model, use_auth_token=True, max_shard_size="10GB", use_safetensors=True)

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, TextStreamer
from transformers.generation.stopping_criteria import StoppingCriteria

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        # 检查最后一个生成的token是否是停止token
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def generate_response_streaming(prompt):
    # 对输入进行编码
    messages = [
        {"role": "system", "content":  "你是一个没素质的人，请用暴躁的语气回复我，尽量多说一些。"},
        {"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 使用 streamer 进行生成
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    # 设置生成参数
    generation_kwargs = {
        "inputs": inputs["input_ids"],
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
        "stopping_criteria": [StopOnTokens([tokenizer.eos_token_id])],
    }
    
    # 在单独的线程中进行生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 实时输出生成的文本
    generated_text = ""
    # 获取输入的长度
    input_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    first_token = True
    token_counts = 0
    for new_text in streamer:
        if token_counts < 4:
            token_counts += 1
            continue
        print(new_text, end="", flush=True)
        generated_text += new_text

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


questions = [
    "那些年轻人为什么不生孩子？",
    "为什么年轻人不买房了？",
    "怎么理解“真传一句话，假传万卷书”？"
]
for question in questions:
    print("question:", question)
    print("answer:", end=" ")
    response = generate_response_streaming(question)
    print('\n')

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, TextStreamer
from transformers.generation.stopping_criteria import StoppingCriteria


model_name = "MLZoo/Qwen2.5-1.5B-Instruct-DPO-bad-boy"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

questions = [
    "那些年轻人为什么不生孩子？",
    "为什么年轻人不买房了？",
    "怎么理解“真传一句话，假传万卷书”？"
]
for question in questions:
    print("question:", question)
    print("answer:", end=" ")
    response = generate_response_streaming(question)
    print('\n')
