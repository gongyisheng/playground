from transformers import AutoModelForCausalLM, AutoTokenizer

def main(padding_side="left"):
    model_name = "Qwen/Qwen3-0.6B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = padding_side
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="bfloat16",
        device_map="auto"
    )

    # prepare the model input
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
    # main(padding_side="left") # correct
    main(padding_side="right") # wrong
