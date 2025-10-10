# Onnx Export (BERT Embedding)
1. onnx model input
```
input_ids: the actual text content
attention_mask: tell which token is real, which is padding

neglect token_type_ids here because out input only has one sentence, not sentence pair. 
token_type_ids is used to tell model of difference sentences
```

2. onnx model output
```
last_hidden_state: [batch_size, sequence_length, hidden_size] vec, each one is 768-dim vector for tokens of sentence
pooler_output: [batch_size, hidden_size] vec, each one is embedding of [CLS] token (first token)

use case:
- last_hidden_state: NER, POS tagging, question answering
- pooler_output: sentence-level classification, sentence representation
```

3. test input/output  
input/output is decided by implementation of BERT model  
test with following code:  
```
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world", return_tensors="pt")
print(inputs)

>>>
{
    'input_ids': tensor([[ 101, 7592, 2088,  102]]), 
    'token_type_ids': tensor([[0, 0, 0, 0]]), 
    'attention_mask': tensor([[1, 1, 1, 1]])
}

output = model(**inputs)
print(output.keys())

>>>
odict_keys(['last_hidden_state', 'pooler_output'])
```

4. embedding calculation
```
mask_expanded = np.expand_dims(attention_mask, -1)
onnx_embeddings = (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1)

step1. expand mask dimensions
mask_expanded = np.expand_dims(attention_mask, -1)
Shape changes: [3, 10] → [3, 10, 1]

step2. calculate mean pooling
last_hidden_state * mask_expanded ---> masked_hidden_state
   [3, 10, 768] * [3, 10, 1] = [3, 10, 768] 
.sum(1) ---> sum along axis 1 (sequence dimension)
   [3, 10, 768] ---> [3, 768]
/ mask_expanded.sum(1) ---> divide by token number, calculate average
   [3, 768] ---> [3, 768]
```

5. dynamic_axes
```
It indicates the axis that is dynamic during inference
eg, "input_ids": {0: "batch_size", 1: "sequence_length"}, 
0,1: axis number, important
batch_size, sequence_length: for documentation
```

6. do_constant_folding=True  
```
optimize graph during onnx export, no need to run optimize command separately
```

7. opset_version
```
17, works for onnxruntime >=1.16
14-16, works for onnxruntime < 1.16

see onnx version:
python3 -c "import onnxruntime; print(onnxruntime.__version__)"
```