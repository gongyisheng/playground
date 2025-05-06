# Transformer related knowledge
## function
- normalization
- subword tokenization
- token to id mapping
- padding and truncation
- add special tokens
- generate attention mask (in transformers)

## limitation
- Case sensitive. eg, "Hello" and "hello" are 2 different tokens
- Hard to deal with numbers: 100 is one token but 111222333444 is separated into 3 tokens
- Space makes a difference: "hello" and " hello" can be 2 different tokens

## padding
Padding is used to solve the problem that batched inputs needs fixed-size tensor. It adds a special padding token after input string to make them have same length.
```
encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
```
3 strategies:
- `max_length`: padding to max_length, but may introduce a lot of noise
- `True` or `longest`: padding to longest string length in a batch
- `False`: no padding

padding side: left or right  

Q: Why LLM uses left padding?  
A: LLMs are decoder-only architectures, meaning they continue to iterate on your input prompt. If your inputs do not have the same length, they need to be padded. Since LLMs are not trained to continue from pad tokens, your input needs to be left-padded.  

## truncation
Truncation is also used like padding to solve long sentence issues  
suggest to use `True` or `'longest_first'`  