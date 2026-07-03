from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer_name = "bert-base-uncased"
model_name = "/media/hdddisk/bert-classify-smsspam-data/results/checkpoint-5580"  # or your specific model name
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()  # Put the model in evaluation mode

texts = [
    "Your input text here",
    "Some texts",
    "Great news! You're selected"
]
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():  # Disable gradients for inference
   outputs = model(**inputs)
   logits = outputs.logits  # Get the logits (raw scores before activation)

prob = torch.softmax(logits, dim=1)  # Apply softmax along the class dimension

probs_pos = prob[:, 1].tolist()  # Probability of label 1
probs_neg = prob[:, 0].tolist()  # Probability of label 0

for i, text in enumerate(texts):
    print(f"Text: {text}")
    print(f"Probability of label 1: {probs_pos[i]}")
    print(f"Probability of label 0: {probs_neg[i]}")
    print("---")
