from transformers import AutoModelForTokenClassification, AutoTokenizer
from configs import HF_MODEL_NAME

model = AutoModelForTokenClassification.from_pretrained(HF_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
inputs = tokenizer("Hello world", return_tensors="pt")
print(inputs)

output = model(**inputs)
print(output.keys())