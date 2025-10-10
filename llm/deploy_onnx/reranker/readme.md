# Onnx Export (BERT Reranker)
1. input/output
```
input: input_ids, attention_mask
output: logits

note that reranker model has a classification head, need to use AutoModelForSequenceClassification to load model
```

2. score calculation
```
step1. prepare input
convert sentence pair into a single sentence
[CLS] query [SEP] document [SEP]

step2. model forward pass
Input tokens → BERT Encoder → [CLS] token embedding → Linear Layer → Logit (raw score)

step3. apply sigmoid
note that please use calculation stable sigmoid function (torch.sigmoid)
```