from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA

# HF → Megatron
bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
bridge.load_hf_weights(model)

# Apply LoRA adapters
lora = LoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=8,
    alpha=16,
)
model = lora(model, training=True)

# Export adapter weights
for i, (name, tensor) in enumerate(bridge.export_adapter_weights(model, cpu=True)):
    print(name, tuple(tensor.shape))
    if i > 25:
        break
