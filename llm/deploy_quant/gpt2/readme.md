## Test Result
```
(3.12.9) yisheng@pc:/media/hdddisk/yisheng/playground/llm/deploy_quant/gpt2$ python3 inference.py
Using device: cuda

==================================================
Running FP32 Inference
==================================================
Input text: Replace me by any text you'd like.

The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
[FP32] Tokens generated: 50
[FP32] Time: 0.4227s
[FP32] Tokens/sec: 118.29
[FP32] Generated text:
  Replace me by any text you'd like.

This is one of the most beautiful things I've ever experienced. I was thinking about my mom when I came up with this idea.

"I don't know, maybe I'm so stupid, but I've never heard of a

==================================================
Running FP16 Inference
==================================================
Input text: Replace me by any text you'd like.

[FP16] Tokens generated: 50
[FP16] Time: 0.2608s
[FP16] Tokens/sec: 191.68
[FP16] Generated text:
  Replace me by any text you'd like.

If you've got a question, please don't hesitate to ask us! We're happy to assist you on any questions you may have!

Thank you for participating in our free survey! We'd love to hear from you!

==================================================
Running BF16 Inference
==================================================
Input text: Replace me by any text you'd like.

[BF16] Tokens generated: 50
[BF16] Time: 0.2561s
[BF16] Tokens/sec: 195.26
[BF16] Generated text:
  Replace me by any text you'd like. [To the woman] Thank you. Please don't touch me with your hand."


As if to prove he was innocent, the woman left the room.


This is the second time the girl has confessed to the man she married.


✅ Inference complete.
```