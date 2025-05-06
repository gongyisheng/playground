# Transformer related knowledge
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
Truncation is also used like padding to solve 