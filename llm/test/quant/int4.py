import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "Qwen/Qwen3-14B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", # normal float 4, based on normal (Gaussian) distribution assumption of weights
        bnb_4bit_use_double_quant=True, # nested quantization, save more memory
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # prepare the model input
    prompts = [
        "introduce yourself",
        "which is bigger, 9.9 or 9.11",
        "how many Rs in \"strawberry\""
    ]
    # formatted_prompts = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": prompt}],
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    #     for prompt in prompts
    # ]
    formatted_prompts = prompts
    model_inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    for i in range(len(prompts)):
        output_ids = generated_ids[i][len(model_inputs.input_ids[i]):].tolist() 
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True)
        print("Output:", outputs)

if __name__ == "__main__":
    main()