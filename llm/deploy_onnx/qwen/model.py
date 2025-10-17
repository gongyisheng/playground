from transformers import AutoModelForCausalLM, AutoTokenizer
from configs import HF_MODEL_NAME

print(f"Loading model: {HF_MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

# Test with a simple prompt
prompt = "Hello, my name is"
inputs = tokenizer(prompt, return_tensors="pt")
print(f"\nInput prompt: '{prompt}'")
print(f"Input shape: {inputs['input_ids'].shape}")

# Generate text
output = model.generate(
    inputs["input_ids"],
    max_new_tokens=20,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nGenerated text: {generated_text}")

# Also test forward pass to see output structure
with tokenizer.as_target_tokenizer():
    forward_output = model(**inputs)
print(f"\nForward pass output keys: {forward_output.keys()}")
print(f"Logits shape: {forward_output.logits.shape}")