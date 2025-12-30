import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name):
    """Load the tokenizer and model once."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model


def run_generate(tokenizer, model):
    """Run a single generation step."""
    prompts = [
        "introduce yourself",
        "which is bigger, 9.9 or 9.11",
        "how many Rs in \"strawberry\""
    ]
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    model_inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(model.device)

    with record_function("model_generate"):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )

    for i in range(len(prompts)):
        output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist()
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True)
        print("Output:", outputs)


def main():
    model_name = "Qwen/Qwen3-0.6B"

    # Load model outside of profiling (we don't want to profile model loading)
    print("Loading model...")
    tokenizer, model = load_model(model_name)
    print("Model loaded.")

    # Number of steps: wait(1) + warmup(1) + active(3) = 5
    num_steps = 5

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler_demo'),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        for step in range(num_steps):
            print(f"\n--- Step {step + 1}/{num_steps} ---")
            run_generate(tokenizer, model)
            prof.step()

    # Print profiling summary
    print("\n" + "=" * 80)
    print("Profiling Summary (sorted by CUDA time):")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
